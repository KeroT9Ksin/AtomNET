// src/atom_math/qairo.rs
//! Qairo: атомарная замена линейного/аффинного слоя.
//!
//! Обычный слой:
//!     y = Wx + b
//!
//! Qairo:
//!     каждая строка W раскладывается на атом:
//!         w = nucleus + radius * direction
//!
//!     вход x раскладывается на атом:
//!         x = nucleus + radius * direction
//!
//!     score = d * n_w * n_x + r_w * r_x * dot(u_w, u_x)
//!
//! Это математически эквивалентно обычному dot при корректном разложении,
//! если остатки имеют нулевое среднее.

use crate::atom::math::{enigma, simd};

#[derive(Clone)]
pub struct QairoAtom {
    pub nucleus: f32,
    pub radius: f32,
    pub dir: Vec<f32>,
    pub sig: u64,
}

/// Разложение вектора на атом: nucleus + radius * dir.
pub fn to_atom(x: &[f32]) -> QairoAtom {
    let dim = x.len().max(1);
    let nucleus: f32 = x.iter().sum::<f32>() / dim as f32;

    let mut residual: Vec<f32> = x.iter().map(|v| v - nucleus).collect();
    let radius = simd::norm(&residual);

    if radius < 1e-12 {
        return QairoAtom {
            nucleus,
            radius: 0.0,
            dir: vec![0.0; x.len()],
            sig: 0,
        };
    }

    let inv_r = 1.0 / radius;
    for v in residual.iter_mut() {
        *v *= inv_r;
    }

    let sig = simd::xor_signature(&residual);

    QairoAtom {
        nucleus,
        radius,
        dir: residual,
        sig,
    }
}

/// Атомарный dot между двумя атомами.
///
/// Для векторов длины d:
///     dot(a, b) ≈ d * n_a * n_b + r_a * r_b * dot(u_a, u_b)
pub fn atom_dot(a: &QairoAtom, b: &QairoAtom, dim: f32) -> f32 {
    let nucleus_part = dim * a.nucleus * b.nucleus;
    let sphere_part = a.radius * b.radius * simd::dot(&a.dir, &b.dir);
    nucleus_part + sphere_part
}

/// Восстановление вектора из атома.
pub fn reconstruct(atom: &QairoAtom) -> Vec<f32> {
    atom.dir
        .iter()
        .map(|&u| atom.nucleus + atom.radius * u)
        .collect()
}

/// Проверка качества разложения.
/// Возвращает максимальную ошибку реконструкции.
pub fn reconstruction_error(original: &[f32], atom: &QairoAtom) -> f32 {
    let rec = reconstruct(atom);

    original
        .iter()
        .zip(rec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Проверка живости атома через XOR.
pub fn atom_is_alive(atom: &QairoAtom) -> bool {
    atom.sig.count_ones() > 10
}

/// Qairo forward: y_i = b_i + atom_dot(row_i, x)
pub fn forward(
    x: &[f32],
    rows: &[QairoAtom],
    bias: &[f32],
) -> Vec<f32> {
    let x_atom = to_atom(x);
    let dim = x.len() as f32;

    rows.iter()
        .zip(bias.iter())
        .map(|(row, b)| {
            b + atom_dot(row, &x_atom, dim)
        })
        .collect()
}

/// Qairo forward с Enigma-гейтом.
///
/// Это уже не чистый линейный слой, а атомарный нелинейный слой:
///     y = b + gate(radius_x) * atom_dot(row, x)
pub fn forward_gate(
    x: &[f32],
    rows: &[QairoAtom],
    bias: &[f32],
) -> Vec<f32> {
    let x_atom = to_atom(x);
    let dim = x.len() as f32;
    let g = enigma::gate(x_atom.radius);

    rows.iter()
        .zip(bias.iter())
        .map(|(row, b)| {
            b + g * atom_dot(row, &x_atom, dim)
        })
        .collect()
}
