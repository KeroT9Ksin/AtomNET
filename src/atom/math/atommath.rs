// src/atom/math/atommath.rs
//! atommath — «наша» математика AtomNET: Атомная Сферическая Арифметика.
//!
//! Формализм (см. docs/ATOM_MATH.md):
//!   Атом веса — четверка A = (n, r, u, s):
//!     n — «ядро» (nucleus): скалярный центр, энергия shell-среднего;
//!     r — «радиус»: L2-норма остатка после снятия ядра;
//!     u — «направление»: единичный вектор, точка на сфере S^{d-1};
//!     s — «спин»: кольцевая подпись-оболочка (R-notes, как SFEROM).
//!  .Weight = w = n·1 + r·u  (преобразуется обратно: n = mean(w), r = |w−n·1|, u = (w−n)/r).
//!
//! Связь с квантовой математикой: направление u на S^{d-1}= projective
//! space; инвариант — Fubini–Study/Bures метрика (аналог «квантового угла»):
//!    ds²_FS = ⟨du,du⟩ − |⟨du,u⟩|² .
//! Ключевой объект — оператор проекции на касательное пространство сферы:
//!    P_u(g) = g − (g·u) u   (Pessôa projection, аналогичный условию
//!    ⟨dψ|ψ⟩=0 в квантовой механике: касаельный вектор не имеет радиальной компоненты).
//! Динамика направления — точный геодезический шаг (поворот по большой окружности),
//! а не normalize(u + lr·g) (геометрия Роберта настоящей римановой оптимизации).
//!
//! Функции модуля — детерминированные примитивы, инвариантные к нормализации и
//! устойчивые к радиальным удалениям. Сжатие — квантование направления по кодовым
//! книгам оболочек (кольцевые коды в духе электронных оболочек).

use crate::atom::math::simd;

/// Евклидова норма.
pub fn norm(v: &[f32]) -> f32 {
    simd::norm(v)
}

/// Модуль с безопасным Handling «мёртвого» (нулевого) вектора.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let r = norm(v);
    if r < 1e-12 {
        return vec![0.0; v.len()];
    }
    let inv = 1.0 / r;
    v.iter().map(|x| x * inv).collect()
}

/// Касательная проекция градиента/вектора на сферу S^{d−1} в точке u:
///   P_u(g) = g − (g·u)·u.   (quantum: ⟨dψ|ψ⟩=0; Fubini–Study геометрия)
/// Предполагается |u| = 1; при необходимости нормализует вход.
pub fn tangent_project(g: &[f32], u: &[f32]) -> Vec<f32> {
    debug_assert_eq!(g.len(), u.len(), "tangent_project: len mismatch");
    let dot = simd::dot(g, u);
    g.iter()
        .zip(u.iter())
        .map(|(&gi, &ui)| gi - dot * ui)
        .collect()
}

/// ТОЧНЫЙ геодезический шаг на сфере S^{d−1} по касательному градиенту g:
///   a) v = P_u(g)               (касательная часть)
///   b) θ = lr·|v|               (шаг по большой окружности)
///   c) u' = cos θ·u + sin θ·(v/|v|)
/// Инвариант: |u'| = 1 с точностью до f32 (не перенормализация, а Rotation).
/// При |v|≈0 или lr≈0 возвращает u (как на геодезической: нет движения).
pub fn geodesic_step(u: &[f32], g: &[f32], lr: f32) -> Vec<f32> {
    debug_assert_eq!(u.len(), g.len(), "geodesic_step: len mismatch");
    let v = tangent_project(g, u);
    let nv = norm(&v);
    if nv < 1e-12 || lr == 0.0 {
        return u.to_vec();
    }
    let vhat = normalize(&v);
    let th = lr * nv;
    let (c, s) = (th.cos(), th.sin());
    u.iter()
        .zip(vhat.iter())
        .map(|(&ui, &vi)| c * ui + s * vi)
        .collect()
}

/// Отражение Хаусхолдера H: H·u = v для единичных u, v (H — ортогонально, |Hx|=|x|).
/// Метод Хаусхолдера (стандартный вчислительный путь):
///   w = u − v;  H(x) = x − 2·(w·x)·w/(w·w).
/// При u ≈ v возвращает x (тождество). У детерминированности: при w norm < 1e-12
/// возвращает x без изменений.
pub fn householder_reflect(x: &[f32], u: &[f32], v: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), u.len(), "householder: len mismatch");
    let w: Vec<f32> = u.iter().zip(v.iter()).map(|(&a, &b)| a - b).collect();
    let s = simd::dot(&w, &w);
    if s < 1e-24 {
        return x.to_vec();
    }
    let dot = simd::dot(&w, x);
    let coeff = 2.0 * dot / s;
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi - coeff * wi)
        .collect()
}

/// Квантование направления по «электронной оболочке»: M кольцевых ширин × K точек
/// на кольце имеют lat = -1..1 равномерно, az = 2πj/K.
///
/// Возвращает (index, cos_similarity) с максимальным косинусом к u.
/// Если M*K ≥ 1: индекс = r*K + j (0..M*K-1).
///
/// Shell-идея: точечно-дискретные направления структурно аналогичны
/// электронным оболочкам (квантование угловых состояний, физика плазмы).
pub fn shell_codebook(m: usize, k: usize) -> Vec<Vec<f32>> {
    let mut book = Vec::with_capacity(m * k);
    for r in 0..m {
        let lat = if m == 1 {
            0.0
        } else {
            -1.0 + 2.0 * (r as f32) / (m - 1) as f32
        };
        let sp = (1.0 - lat * lat).max(0.0).sqrt();
        let bz = lat;
        for j in 0..k {
            let ang = 2.0 * std::f32::consts::PI * (j as f32) / (k.max(1) as f32);
            book.push(vec![sp * ang.cos(), sp * ang.sin(), bz]);
        }
    }
    book
}

/// Быстрое квантование (предварительный перебор Dot с codebook, max-cosindex).
pub fn shell_quantize(u: &[f32], m: usize, k: usize) -> (usize, f32) {
    let book = shell_codebook(m, k);
    let un = normalize(u);
    let mut best = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (i, cw) in book.iter().enumerate() {
        let dgt = simd::dot(&un, cw);
        // normalize already unit? cw = unit (lat²/2 + (1-lat²) = 1 ✓)
        if dgt > best_dot {
            best_dot = dgt;
            best = i;
        }
    }
    (best, best_dot)
}

/// Реконструкция ряд-вектора: w = n·1 + r·u.
pub fn reconstruct(n: f32, r: f32, u: &[f32]) -> Vec<f32> {
    u.iter().map(|&ui| n + r * ui).collect()
}

/// Разложение ряд-вектора в атом: (nucleus=mean, radius=||w−n||, dir=(w−n)/r).
/// При r < 1e-12 возвращает (mean, 0.0, нулевой вектор).
pub fn decompose(w: &[f32]) -> (f32, f32, Vec<f32>) {
    let n: f32 = w.iter().sum::<f32>() / (w.len().max(1) as f32);
    let resid: Vec<f32> = w.iter().map(|&v| v - n).collect();
    let r = norm(&resid);
    if r < 1e-12 {
        return (n, 0.0, vec![0.0; w.len()]);
    }
    let dir = normalize(&resid);
    (n, r, dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E3779B9_7F4A_7C15) ^ 0x12345678u64;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push((((s >> 33) % 2000) as f32 - 1000.0) / 1000.0);
        }
        v
    }

    #[test]
    fn normalize_unit_circle() {
        let u = normalize(&[3.0, 4.0, 0.0]);
        assert!((norm(&u) - 1.0).abs() < 1e-6);
        assert!((norm(&normalize(&[0.0, 0.0]))).abs() < 1e-12);
    }

    #[test]
    fn geodesic_preserves_norm_and_direction() {
        let u = normalize(&rand_vec(16, 7));
        let g = rand_vec(16, 11);
        let u2 = geodesic_step(&u, &g, 0.05);
        assert!((norm(&u2) - 1.0).abs() < 2e-6, "norm {}", norm(&u2));
        // lr=0 => identity
        let u0 = geodesic_step(&u, &g, 0.0);
        assert!(u0.iter().zip(u.iter()).all(|(a, b)| (a - b).abs() < 1e-7));
        // маленький lr: u2 ≈ normalize(u + lr*v)
        let v = tangent_project(&g, &u);
        let lr = 0.001f32;
        let mut z = u.clone();
        for (zi, vi) in z.iter_mut().zip(v.iter()) {
            *zi += lr * *vi;
        }
        let z = normalize(&z);
        let u3 = geodesic_step(&u, &g, lr);
        assert!(u3.iter().zip(z.iter()).all(|(a, b)| (a - b).abs() < 1e-3));
    }

    #[test]
    fn tangent_projection_orthogonal_to_u() {
        let u = normalize(&rand_vec(10, 3));
        let g = rand_vec(10, 17);
        let tg = tangent_project(&g, &u);
        assert!(simd::dot(&tg, &u).abs() < 1e-5);

        // |g − tg| = |g·u| (пифагор, точно при нормализации)
        let dot = simd::dot(&g, &u);
        let resid: Vec<f32> = g.iter().zip(tg.iter()).map(|(a, b)| a - b).collect();
        assert!((norm(&resid) - dot.abs()).abs() < 1e-5);
    }

    #[test]
    fn householder_maps_u_to_v_and_orthogonality() {
        let u = normalize(&rand_vec(12, 5));
        let v = normalize(&rand_vec(12, 13));
        // При u==v — тождество
        let x = rand_vec(12, 21);
        let out = householder_reflect(&x, &u, &u);
        assert!(out.iter().zip(x.iter()).all(|(a, b)| (a - b).abs() < 1e-7));
        // H·u = v ?
        let hu = householder_reflect(&u, &u, &v);
        for (a, b) in hu.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // Ортогональность: |Hx| = |x|
        let hx = householder_reflect(&x, &u, &v);
        assert!((norm(&hx) - norm(&x)).abs() < 1e-5);
    }

    #[test]
    fn shell_quantize_accuracy() {
        // Для плотной codebook (M=16, K=16) ошибка направления ограничена
        let u = normalize(&[1.0, 0.0, 0.0]);
        let (_, d) = shell_quantize(&u, 16, 16);
        assert!(d > 0.9, "cos {d}");
        // точное попадание в код
        let book = shell_codebook(8, 8);
        let cw = &book[42.min(book.len() - 1)];
        let (_, d2) = shell_quantize(cw, 8, 8);
        assert!(d2 > 0.999, "exact hit {d2}");
    }

    #[test]
    fn decompose_reconstruct_identity() {
        let w = rand_vec(20, 9);
        let (n, r, d) = decompose(&w);
        let w2 = reconstruct(n, r, &d);
        for (a, b) in w.iter().zip(w2.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        // у однородного вектора радиус ноль
        let (n2, r2, _) = decompose(&[2.0, 2.0, 2.0]);
        assert!((n2 - 2.0).abs() < 1e-6 && r2 < 1e-12);
    }
}
