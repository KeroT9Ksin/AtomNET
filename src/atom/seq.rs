// src/atom/seq.rs
//! atom::seq — последовательностное ядро для «настоящей» сети.
//!
//! Причинное (causal) самовнимание с KV-кэшем, RoPE-позиции, sampler.
//! f32, работает на simd-ядрах AtomNET; Q16-вариант — следующим шагом.
//! Качество обучения гарантируется градиентной проверкой (gradcheck)
//! в тестах: аналитический backward сверяется с конечными разностями.

use crate::atom::math::simd;

/// Кэш причинного внимания: scores (L×L) и вероятности softmax (L×L).
pub struct AttnCache {
    pub scores: Vec<f32>,
    pub probs: Vec<f32>,
    pub l: usize,
}

/// Причинное внимание: out_i = Σ_{j≤i} p_ij·v_j,
/// p = softmax(q·k/√d) по строке (j≤i; j>i маскируется).
pub fn causal_attn_fwd(q: &[f32], k: &[f32], v: &[f32], d: usize) -> (Vec<f32>, AttnCache) {
    let l = q.len() / d;
    assert_eq!(k.len(), l * d);
    assert_eq!(v.len(), l * d);
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; l * d];
    let mut scores = vec![f32::NEG_INFINITY; l * l];
    let mut probs = vec![0.0f32; l * l];
    for i in 0..l {
        let qi = &q[i * d..(i + 1) * d];
        // scores
        for j in 0..=i {
            let kj = &k[j * d..(j + 1) * d];
            scores[i * l + j] = simd::dot(qi, kj) * scale;
        }
        // softmax по строке: ASM max/exp/sum, сразу в probs.
        simd::softmax_into(
            &scores[i * l..(i + 1) * l],
            &mut probs[i * l..(i + 1) * l],
        );
        // out_i = Σ p_ij v_j
        for j in 0..=i {
            let p = probs[i * l + j];
            if p != 0.0 {
                let vj = &v[j * d..(j + 1) * d];
                let oi = &mut out[i * d..(i + 1) * d];
                for t in 0..d {
                    oi[t] += p * vj[t];
                }
            }
        }
    }
    (out, AttnCache { scores, probs, l })
}

/// Backward причинного внимания. Возвращает (dq, dk, dv).
pub fn causal_attn_bwd(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cache: &AttnCache,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let l = cache.l;
    let scale = 1.0 / (d as f32).sqrt();
    let mut dq = vec![0.0f32; l * d];
    let mut dk = vec![0.0f32; l * d];
    let mut dv = vec![0.0f32; l * d];
    for i in 0..l {
        let di = &dout[i * d..(i + 1) * d];
        let out_i: Vec<f32> = (0..d)
            .map(|t| {
                let mut s = 0.0f32;
                for j in 0..=i {
                    s += cache.probs[i * l + j] * v[j * d + t];
                }
                s
            })
            .collect();
        let d_out_dot = simd::dot(di, &out_i);
        // dS_ij = scale · p_ij · (di·v_j − di·out_i)
        // dq_i = Σ_j dS_ij·k_j ; dk_j += dS_ij·q_i ; dv_j += p_ij·di
        for j in 0..=i {
            let p = cache.probs[i * l + j];
            let vj = &v[j * d..(j + 1) * d];
            let di_vj = simd::dot(di, vj);
            let ds = scale * p * (di_vj - d_out_dot);
            // dq_i
            let q_i = &mut dq[i * d..(i + 1) * d];
            let kj = &k[j * d..(j + 1) * d];
            for t in 0..d {
                q_i[t] += ds * kj[t];
            }
            // dk_j
            let dk_j = &mut dk[j * d..(j + 1) * d];
            let q_i_ro = &q[i * d..(i + 1) * d];
            for t in 0..d {
                dk_j[t] += ds * q_i_ro[t];
            }
            // dv_j
            let dv_j = &mut dv[j * d..(j + 1) * d];
            for t in 0..d {
                dv_j[t] += p * di[t];
            }
        }
    }
    (dq, dk, dv)
}

/// GQA (Grouped-Query Attention): Q-голов делятся на группы KV-голов.
/// Раскладка q/k/v — ПО ТОКЕНАМ: [t][head][d] (как в transformer-моделях).
/// Внутри головы собираются во временные непрерывные буферы.
pub fn gqa_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_heads: usize,
    kv_heads: usize,
    d: usize,
) -> (Vec<f32>, Vec<AttnCache>) {
    let l = q.len() / (q_heads * d);
    assert_eq!(k.len(), l * kv_heads * d);
    assert_eq!(v.len(), l * kv_heads * d);
    let group = q_heads / kv_heads.max(1);
    let mut out = vec![0.0f32; l * q_heads * d];
    let mut caches = Vec::with_capacity(q_heads);
    for h in 0..q_heads {
        let kv = h / group;
        // собрать голову h из пер-токен раскладки
        let mut qh = vec![0.0f32; l * d];
        for t in 0..l {
            let base = t * q_heads * d + h * d;
            qh[t * d..(t + 1) * d].copy_from_slice(&q[base..base + d]);
        }
        let mut kh = vec![0.0f32; l * d];
        let mut vh = vec![0.0f32; l * d];
        for t in 0..l {
            let base = t * kv_heads * d + kv * d;
            kh[t * d..(t + 1) * d].copy_from_slice(&k[base..base + d]);
            vh[t * d..(t + 1) * d].copy_from_slice(&v[base..base + d]);
        }
        let (oh, cache) = causal_attn_fwd(&qh, &kh, &vh, d);
        for t in 0..l {
            let base = t * q_heads * d + h * d;
            out[base..base + d].copy_from_slice(&oh[t * d..(t + 1) * d]);
        }
        caches.push(cache);
    }
    (out, caches)
}

/// Backward GQA: dq по головам; dk/dv накапливаются от всех Q-голов группы.
pub fn gqa_bwd(
    dout: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    caches: &[AttnCache],
    q_heads: usize,
    kv_heads: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let l = q.len() / (q_heads * d);
    let group = q_heads / kv_heads.max(1);
    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.len()];
    for h in 0..q_heads {
        let kv = h / group;
        let mut qh = vec![0.0f32; l * d];
        let mut kh = vec![0.0f32; l * d];
        let mut vh = vec![0.0f32; l * d];
        let mut dh = vec![0.0f32; l * d];
        for t in 0..l {
            let bq = t * q_heads * d + h * d;
            let bk = t * kv_heads * d + kv * d;
            qh[t * d..(t + 1) * d].copy_from_slice(&q[bq..bq + d]);
            kh[t * d..(t + 1) * d].copy_from_slice(&k[bk..bk + d]);
            vh[t * d..(t + 1) * d].copy_from_slice(&v[bk..bk + d]);
            dh[t * d..(t + 1) * d].copy_from_slice(&dout[bq..bq + d]);
        }
        let (dqh, dkh, dvh) = causal_attn_bwd(&dh, &qh, &kh, &vh, &caches[h], d);
        for t in 0..l {
            let bq = t * q_heads * d + h * d;
            let bk = t * kv_heads * d + kv * d;
            for j in 0..d {
                dq[bq + j] += dqh[t * d + j];
                dk[bk + j] += dkh[t * d + j];
                dv[bk + j] += dvh[t * d + j];
            }
        }
    }
    (dq, dk, dv)
}

fn rope_apply_mode(
    x: &mut [f32],
    pos: usize,
    theta: f32,
    d: usize,
    sign: f32,
    legacy: bool,
) {
    for i in (0..d).step_by(2) {
        if i + 1 >= d {
            break;
        }
        let angle = if legacy {
            // Старое (ошибочное) поведение: theta^i. Нужно только для
            // чтения чекпойнтов v1, обученных до исправления RoPE.
            sign * pos as f32 / theta.powf(i as f32)
        } else {
            // Стандартный RoPE: inv_freq(pair) = theta^(-2*pair/d).
            // Здесь i — индекс координаты (0,2,4,...), поэтому показатель
            // равен -i/d.
            sign * pos as f32 * theta.powf(-(i as f32) / d as f32)
        };
        let (s, c) = angle.sin_cos();
        let (a, b) = (x[i], x[i + 1]);
        x[i] = a * c - b * s;
        x[i + 1] = a * s + b * c;
    }
}

/// RoPE: вращение пар координат позицией pos (sign=1 прямое, −1 обратное).
pub fn rope_apply_sign(x: &mut [f32], pos: usize, theta: f32, d: usize, sign: f32) {
    rope_apply_mode(x, pos, theta, d, sign, false);
}

/// Старый RoPE для обратной совместимости с чекпойнтами v1.
pub fn rope_apply_legacy(x: &mut [f32], pos: usize, theta: f32, d: usize, sign: f32) {
    rope_apply_mode(x, pos, theta, d, sign, true);
}

/// RoPE: вращение пар координат позицией pos.
pub fn rope_apply(x: &mut [f32], pos: usize, theta: f32, d: usize) {
    rope_apply_sign(x, pos, theta, d, 1.0);
}

/// Простой sampler: temperature + top-k. Возвращает индекс токена.
pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize, rng: f32) -> usize {
    let t = temperature.max(1e-3);
    let mut probs: Vec<f32> = logits.iter().map(|&l| l / t).collect();
    simd::softmax(&mut probs);
    let k = top_k.max(1).min(probs.len());
    // top-k маска: оставляем k крупнейших вероятностей.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = probs[order[k - 1]];
    for (idx, p) in probs.iter_mut().enumerate() {
        if *p < threshold && !order[..k].contains(&idx) {
            *p = 0.0;
        }
    }
    let sum2: f32 = probs.iter().sum();
    let u = (rng.clamp(0.0, 0.999999) * sum2).max(0.0);
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if acc >= u {
            return i;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_uses_standard_frequencies() {
        let mut x = vec![1.0f32, 0.0, 1.0, 0.0];
        rope_apply(&mut x, 1, 10_000.0, 4);
        let (s0, c0) = 1.0f32.sin_cos();
        let (s1, c1) = 10_000f32.powf(-2.0 / 4.0).sin_cos();
        assert!((x[0] - c0).abs() < 1e-5);
        assert!((x[1] - s0).abs() < 1e-5);
        assert!((x[2] - c1).abs() < 1e-5);
        assert!((x[3] - s1).abs() < 1e-5);

        // Вторая пара не должна быть «заморожена»: для d=16 её угол
        // при pos=1 равен theta^(-1/8) ~ 0.316, а не ~1e-4.
        let mut y = vec![0.0f32; 16];
        y[2] = 1.0;
        rope_apply(&mut y, 1, 10_000.0, 16);
        assert!(y[3].abs() > 0.1, "вторая RoPE-пара почти не вращается: {}", y[3]);
    }

    fn grad_numeric<F: Fn(&[f32]) -> Vec<f32>>(
        fwd: F,
        analytic: &[f32],
        param: &[f32],
        eps: f32,
        loss: &dyn Fn(&[f32]) -> f32,
    ) {
        for (idx, &a) in analytic.iter().enumerate() {
            let mut p_hi = param.to_vec();
            let mut p_lo = param.to_vec();
            p_hi[idx] += eps;
            p_lo[idx] -= eps;
            let num = (loss(&fwd(&p_hi)) - loss(&fwd(&p_lo))) / (2.0 * eps);
            let diff = (num - a).abs();
            assert!(
                diff < 1e-4 || diff < 0.05 * a.abs(),
                "[{idx}]: analytic={a}, numeric={num}"
            );
        }
    }

    /// Градиентная проверка: dq/dk/dv против конечных разностей.
    #[test]
    fn attn_gradcheck() {
        let l = 4usize;
        let d = 3usize;
        let eps = 1e-3f32;

        let mut r = 0x1234_5678u64;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            ((r >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };
        let mut gen = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };

        let q = gen(l * d);
        let k = gen(l * d);
        let v = gen(l * d);
        let (out, cache) = causal_attn_fwd(&q, &k, &v, d);
        let loss = |o: &[f32]| o.iter().map(|x| x * x).sum::<f32>() / 2.0;
        let (dq, dk, dv) = causal_attn_bwd(&out, &q, &k, &v, &cache, d);

        grad_numeric(|q2| causal_attn_fwd(q2, &k, &v, d).0, &dq, &q, eps, &loss);
        grad_numeric(|k2| causal_attn_fwd(&q, k2, &v, d).0, &dk, &k, eps, &loss);
        grad_numeric(|v2| causal_attn_fwd(&q, &k, v2, d).0, &dv, &v, eps, &loss);
    }

    /// GQA (2 Q-головы, 1 KV-голова = MQA): градиенты dk/dv суммируются
    /// от обеих Q-голов группы.
    #[test]
    fn gqa_gradcheck() {
        let l = 3usize;
        let d = 3usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let eps = 1e-3f32;

        let mut r = 0xDEAD_BEEFu64;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            ((r >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };
        let mut gen = |n: usize| -> Vec<f32> { (0..n).map(|_| next()).collect() };

        let q = gen(l * q_heads * d);
        let k = gen(l * kv_heads * d);
        let v = gen(l * kv_heads * d);
        let (out, caches) = gqa_fwd(&q, &k, &v, q_heads, kv_heads, d);
        let loss = |o: &[f32]| o.iter().map(|x| x * x).sum::<f32>() / 2.0;
        let (dq, dk, dv) = gqa_bwd(&out, &q, &k, &v, &caches, q_heads, kv_heads, d);

        grad_numeric(|q2| gqa_fwd(q2, &k, &v, q_heads, kv_heads, d).0, &dq, &q, eps, &loss);
        grad_numeric(|k2| gqa_fwd(&q, k2, &v, q_heads, kv_heads, d).0, &dk, &k, eps, &loss);
        grad_numeric(|v2| gqa_fwd(&q, &k, v2, q_heads, kv_heads, d).0, &dv, &v, eps, &loss);
    }
}
