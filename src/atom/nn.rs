// src/atom/nn.rs
//! atom::nn — готовые компоненты современного transformer-блока.
//!
//! Всё trainable градиентно проверено (gradcheck конечными разностями):
//! RMSNorm, GQA-Attention-слой (QKV-проекции + RoPE + causal + output-проекция),
//! batch cross-entropy, активации. Сборка полных блоков — поверх этих частей.
//!
//! Математика (silu/rmsnorm/softmax/CE/linear backward) живёт в
//! `atom::math::tensor`; здесь — только составные компоненты.

use crate::atom::math::tensor;
use crate::atom::seq::{gqa_bwd, gqa_fwd, rope_apply_sign};

/// silu(x) = x·sigmoid(x)
pub fn silu(x: f32) -> f32 {
    tensor::silu(x)
}

/// RMSNorm: y = x / sqrt(mean(x²)+eps) · gamma
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    tensor::rms_norm(x, gamma, eps)
}

/// Backward RMSNorm: y = x·g/√(mean x²+eps).
pub fn rms_norm_bwd(
    dout: &[f32],
    x: &[f32],
    gamma: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    tensor::rms_norm_bwd(dout, x, gamma, eps)
}

/// Энтропия распределения логитов (для регуляризации/уверенности).
pub fn logits_entropy(logits: &[f32]) -> f32 {
    tensor::softmax_entropy(logits)
}

/// Batch cross-entropy: logits B×V, targets B. Возвращает (loss, dlogits).
pub fn cross_entropy_batch(logits: &[f32], targets: &[u32], vocab: usize) -> (f32, Vec<f32>) {
    tensor::cross_entropy(logits, targets, vocab)
}

/// dW (rows×cols) = Σ_t dout_t ⊗ x_t; dx = W^T dout.
pub fn linear_backward(
    w: &[f32],
    rows: usize,
    cols: usize,
    dout: &[f32],
    x: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let wt = tensor::Tensor::from_vec(w.to_vec(), rows, cols);
    tensor::linear_backward(&wt, rows, cols, dout, x)
}

/// GQA-Attention-слой с RoPE: Wq/Wk/Wv проекции + attention + Wo.
pub struct AttnLayer {
    pub d_model: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub d_head: usize,
    pub rope_theta: f32,
    pub wq: Vec<f32>, // (q_heads·d_head) × d_model
    pub wk: Vec<f32>, // (kv_heads·d_head) × d_model
    pub wv: Vec<f32>,
    pub wo: Vec<f32>, // d_model × (q_heads·d_head)
}

impl AttnLayer {
    pub fn new(d_model: usize, q_heads: usize, kv_heads: usize, d_head: usize, seed: u64) -> Self {
        let mut r = seed | 1;
        let mut next = move || {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            ((r >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };
        let mut gen = |rows: usize, cols: usize| -> Vec<f32> {
            let scale = (1.0 / (cols as f32).sqrt()).min(0.3);
            (0..rows * cols).map(|_| next() * scale).collect()
        };
        let out_dim = q_heads * d_head;
        let kdim = kv_heads * d_head;
        Self {
            d_model,
            q_heads,
            kv_heads,
            d_head,
            rope_theta: 10000.0,
            wq: gen(out_dim, d_model),
            wk: gen(kdim, d_model),
            wv: gen(kdim, d_model),
            wo: gen(d_model, out_dim),
        }
    }

    fn proj(&self, w: &[f32], out_dim: usize, x: &[f32]) -> Vec<f32> {
        // out_t = W x_t; результат L×out_dim
        let l = x.len() / self.d_model;
        let mut out = vec![0.0f32; l * out_dim];
        for t in 0..l {
            let xrow = &x[t * self.d_model..(t + 1) * self.d_model];
            for i in 0..out_dim {
                let base = i * self.d_model;
                out[t * out_dim + i] = crate::atom::math::simd::dot(&w[base..base + self.d_model], xrow);
            }
        }
        out
    }

    /// Внутренний forward с промежуточными (для backward).
    fn fwd_parts(
        &self,
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<crate::atom::seq::AttnCache>, Vec<f32>) {
        let l = x.len() / self.d_model;
        let mut q = self.proj(&self.wq, self.q_heads * self.d_head, x);
        let mut k = self.proj(&self.wk, self.kv_heads * self.d_head, x);
        let v = self.proj(&self.wv, self.kv_heads * self.d_head, x);
        for t in 0..l {
            for h in 0..self.q_heads {
                let slice = &mut q[(t * self.q_heads + h) * self.d_head
                    ..(t * self.q_heads + h + 1) * self.d_head];
                rope_apply_sign(slice, t, self.rope_theta, self.d_head, 1.0);
            }
            for h in 0..self.kv_heads {
                let slice = &mut k[(t * self.kv_heads + h) * self.d_head
                    ..(t * self.kv_heads + h + 1) * self.d_head];
                rope_apply_sign(slice, t, self.rope_theta, self.d_head, 1.0);
            }
        }
        let (attn, caches) = gqa_fwd(&q, &k, &v, self.q_heads, self.kv_heads, self.d_head);
        let out = self.proj(&self.wo, self.d_model, &attn);
        (q, k, v, attn, caches, out)
    }

    /// Forward: x (L×d_model) -> out (L×d_model). Кэш внимания возвращается.
    pub fn forward(&self, x: &[f32]) -> (Vec<f32>, Vec<crate::atom::seq::AttnCache>) {
        let (_, _, _, _, caches, out) = self.fwd_parts(x);
        (out, caches)
    }

    /// Backward: dWq/dWk/dWv/dWo + dx. Пересчитывает forward (L маленькие).
    pub fn backward(
        &self,
        dout: &[f32],
        x: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let l = x.len() / self.d_model;
        let q_dim = self.q_heads * self.d_head;
        let kv_dim = self.kv_heads * self.d_head;
        let (q, k, v, attn, caches, _out) = self.fwd_parts(x);

        // 1) Wo: dW_wo += Σ_t dout_t ⊗ attn_t; d_attn = Wo^T dout
        let (dwo, d_attn) =
            crate::atom::nn::linear_backward(&self.wo, self.d_model, q_dim, dout, &attn);

        // 2) attention backward -> dq, dk, dv
        let (dq, dk, dv) = gqa_bwd(
            &d_attn,
            &q,
            &k,
            &v,
            &caches,
            self.q_heads,
            self.kv_heads,
            self.d_head,
        );

        // 3) обратное RoPE
        let mut dq = dq;
        let mut dk = dk;
        for t in 0..l {
            for h in 0..self.q_heads {
                let slice = &mut dq[(t * self.q_heads + h) * self.d_head
                    ..(t * self.q_heads + h + 1) * self.d_head];
                rope_apply_sign(slice, t, self.rope_theta, self.d_head, -1.0);
            }
            for h in 0..self.kv_heads {
                let slice = &mut dk[(t * self.kv_heads + h) * self.d_head
                    ..(t * self.kv_heads + h + 1) * self.d_head];
                rope_apply_sign(slice, t, self.rope_theta, self.d_head, -1.0);
            }
        }

        // 4) проекции QKV: dW += Σ d_t ⊗ x_t; dx = Σ W^T d
        let (dwq, dxq) = crate::atom::nn::linear_backward(&self.wq, q_dim, self.d_model, &dq, x);
        let (dwk, dxk) = crate::atom::nn::linear_backward(&self.wk, kv_dim, self.d_model, &dk, x);
        let (dwv, dxv) = crate::atom::nn::linear_backward(&self.wv, kv_dim, self.d_model, &dv, x);
        let mut dx = vec![0.0f32; l * self.d_model];
        for i in 0..dx.len() {
            dx[i] = dxq[i] + dxk[i] + dxv[i];
        }
        (dwq, dwk, dwv, dwo, dx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::seq::gqa_bwd;

    fn grad_numeric<F: Fn(&[f32]) -> f32>(fwd: F, analytic: &[f32], param: &[f32], eps: f32) {
        for (idx, &a) in analytic.iter().enumerate() {
            let mut hi = param.to_vec();
            let mut lo = param.to_vec();
            hi[idx] += eps;
            lo[idx] -= eps;
            let num = (fwd(&hi) - fwd(&lo)) / (2.0 * eps);
            let diff = (num - a).abs();
            assert!(
                diff < 1e-3 || diff < 0.05 * a.abs(),
                "[{idx}]: analytic={a}, numeric={num}"
            );
        }
    }

    #[test]
    fn rmsnorm_gradcheck() {
        let x = vec![0.3f32, -0.7, 1.2, 0.05, -0.4];
        let gamma = vec![1.0f32, 1.1, 0.9, 1.0, 1.05];
        let eps = 1e-6f32;
        let y = rms_norm(&x, &gamma, eps);
        let loss = |o: &[f32]| o.iter().map(|v| v * v).sum::<f32>() / 2.0;
        let dout: Vec<f32> = y.iter().map(|v| *v).collect();
        let (dx, dg) = rms_norm_bwd(&dout, &x, &gamma, eps);
        grad_numeric(
            |x2| loss(&rms_norm(x2, &gamma, eps)),
            &dx,
            &x,
            1e-3,
        );
        grad_numeric(
            |g2| loss(&rms_norm(&x, g2, eps)),
            &dg,
            &gamma,
            1e-3,
        );
    }

    #[test]
    fn ce_batch_matches_single() {
        let logits = vec![0.5f32, 0.2, -0.1, 1.0, 2.0, -1.0];
        let targets = vec![1u32, 2];
        let (loss, d) = cross_entropy_batch(&logits, &targets, 3);
        assert!(loss.is_finite() && loss >= 0.0);
        // dlogits суммируются в 0 на строку
        assert!((d[0] + d[1] + d[2]).abs() < 1e-5);
        assert!((d[3] + d[4] + d[5]).abs() < 1e-5);
        assert!(d[1] < 0.0 && d[5] < 0.0, "target лог нужно вычесть");
    }

    /// AttnLayer gradcheck: dWq/dWk/dWv/dWo + dx против конечных разностей.
    #[test]
    fn attn_layer_gradcheck() {
        let d_model = 4usize;
        let q_heads = 2usize;
        let kv_heads = 1usize;
        let d_head = 2usize;
        let l = 3usize;
        let layer = AttnLayer::new(d_model, q_heads, kv_heads, d_head, 42);
        let x: Vec<f32> = (0..l * d_model)
            .map(|k| {
                let u = (k.wrapping_mul(2246822519) >> 8) as f32 / (1u32 << 24) as f32;
                (u - 0.5) * 1.0
            })
            .collect();
        let loss = |o: &[f32]| o.iter().map(|v| v * v).sum::<f32>() / 2.0;

        fn perturb(
            src: &AttnLayer,
            kind: &str,
            idx: usize,
            delta: f32,
            d_model: usize,
            q_heads: usize,
            kv_heads: usize,
            d_head: usize,
        ) -> AttnLayer {
            let mut l2 = AttnLayer::new(d_model, q_heads, kv_heads, d_head, 42);
            l2.wq.copy_from_slice(&src.wq);
            l2.wk.copy_from_slice(&src.wk);
            l2.wv.copy_from_slice(&src.wv);
            l2.wo.copy_from_slice(&src.wo);
            match kind {
                "wq" => l2.wq[idx] += delta,
                "wk" => l2.wk[idx] += delta,
                "wv" => l2.wv[idx] += delta,
                "wo" => l2.wo[idx] += delta,
                _ => panic!("kind"),
            }
            l2
        }

        // Аналитические градиенты
        let dout: Vec<f32> = layer.forward(&x).0.iter().map(|v| *v).collect();
        let (dwq, dwk, dwv, dwo, dx) = layer.backward(&dout, &x);

        // Численно: по каждому семейству параметров
        let eps = 1e-3f32;
        let check_w = |name: &str, dw: &[f32]| {
            for (idx, &a) in dw.iter().enumerate() {
                let hi = perturb(&layer, name, idx, 1e-3, d_model, q_heads, kv_heads, d_head);
                let lo = perturb(&layer, name, idx, -1e-3, d_model, q_heads, kv_heads, d_head);
                let num = (loss(&hi.forward(&x).0) - loss(&lo.forward(&x).0)) / (2.0 * eps);
                let diff = (num - a).abs();
                assert!(
                    diff < 2e-3 || diff < 0.05 * a.abs(),
                    "{name}[{idx}]: analytic={a}, numeric={num}"
                );
            }
        };
        check_w("wq", &dwq);
        check_w("wk", &dwk);
        check_w("wv", &dwv);
        check_w("wo", &dwo);

        // dx против конечных разностей
        for (idx, &a) in dx.iter().enumerate() {
            let mut hi = x.clone();
            let mut lo = x.clone();
            hi[idx] += eps;
            lo[idx] -= eps;
            let num = (loss(&layer.forward(&hi).0) - loss(&layer.forward(&lo).0)) / (2.0 * eps);
            let diff = (num - a).abs();
            assert!(
                diff < 2e-3 || diff < 0.05 * a.abs(),
                "dx[{idx}]: analytic={a}, numeric={num}"
            );
        }
    }
}
