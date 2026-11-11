// src/atom_math/atom_codec.rs
//! AtomCodec: многоуровневое атомарное сжатие.
//!
//! Идея:
//! Блок x раскладывается как:
//!
//!     x = nucleus + radius * direction + residual_levels
//!
//! Уровни:
//! 0. nucleus = mean(x)
//! 1. radius + quantized direction
//! 2+. остаточные уровни:
//!
//!     residual_0 = x - (nucleus + radius * dir_hat)
//!     residual_k = residual_{k-1} - scale_k * q_k
//!
//! Это развитие Micron, а не его удаление.
//! Micron остаётся как проверенный int8 baseline.

use crate::atom::math::simd;


#[derive(Clone)]
pub struct AtomLevel {
    pub bits: u8,
    pub scale: f32,
    /// Коды упакованы по `bits` бит на элемент (не один i8 на код).
    pub q: Vec<u8>,
}

#[derive(Clone)]
pub struct AtomCompressed {
    pub len: usize,
    pub nucleus: f32,
    pub radius: f32,
    pub dir_bits: u8,
    pub dir_max_code: f32,
    /// Коды упакованы по `dir_bits` бит на элемент.
    pub dir_q: Vec<u8>,
    pub residual_levels: Vec<AtomLevel>,
    pub sig: u64,
}

fn max_abs(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()))
}

/// Упаковать знаковые коды в битовый поток (little-endian по битам).
fn pack_codes(codes: &[i8], bits: u8) -> Vec<u8> {
    let bits = bits.clamp(1, 8) as usize;
    let mask = (1u32 << bits) - 1;
    let total_bits = codes.len().saturating_mul(bits);
    let mut out = vec![0u8; total_bits.div_ceil(8)];
    for (i, &v) in codes.iter().enumerate() {
        let raw = (v as i32 as u32) & mask;
        let bitpos = i * bits;
        let byte = bitpos / 8;
        let shift = bitpos % 8;
        out[byte] |= ((raw << shift) & 0xff) as u8;
        if shift + bits > 8 {
            out[byte + 1] |= (raw >> (8 - shift)) as u8;
        }
    }
    out
}

/// Распаковать один знаковый код; выход в диапазоне 2^(bits-1).
fn unpack_code(data: &[u8], bits: u8, idx: usize) -> i32 {
    let bits = bits.clamp(1, 8) as usize;
    let mask = (1u32 << bits) - 1;
    let bitpos = idx * bits;
    let byte = bitpos / 8;
    let shift = bitpos % 8;
    let mut raw = (data.get(byte).copied().unwrap_or(0) as u32) >> shift;
    if shift + bits > 8 {
        let next = data.get(byte + 1).copied().unwrap_or(0) as u32;
        raw |= next << (8 - shift);
    }
    let raw = (raw & mask) as i32;
    let sign = 1i32 << (bits - 1);
    if raw & sign != 0 {
        raw - (1i32 << bits)
    } else {
        raw
    }
}

/// Сжать блок в многоуровневый атомарный формат.
///
/// `bits`:
/// - bits[0] — битность направления (обычно 8, 4)
/// - bits[1..] — битность остаточных уровней (например, 4, 2)
///
/// Примеры:
/// - [8]       = только атомарное направление, похоже на Micron baseline
/// - [8, 4]    = направление 8 бит + остаток 4 бита
/// - [8, 4, 2] = направление 8 бит + остаток 4 бита + остаток 2 бита
/// - [4, 2]    = агрессивный режим экономии памяти
pub fn compress_atom(x: &[f32], bits: &[u8]) -> AtomCompressed {
    let len = x.len();
    let dim = len.max(1) as f32;

    // Уровень 0: nucleus.
    let nucleus: f32 = x.iter().sum::<f32>() / dim;

    // Уровень 1: остаток и радиус.
    // Нормируем на max_abs, а не на L2: при L2-радиусе шаг квантования
    // растёт как sqrt(dim) и качество в разы хуже int8-Micron.
    let mut residual: Vec<f32> = x.iter().map(|v| v - nucleus).collect();
    let radius = max_abs(&residual);

    let dir_bits = bits.first().copied().unwrap_or(8).clamp(2, 8);
    let dir_max_code = ((1u32 << ((dir_bits - 1) as u32)) - 1) as f32;

    let mut dir_codes = vec![0i8; len];

    if radius > 1e-12 {
        let inv_r = 1.0 / radius;

        for i in 0..len {
            let u = residual[i] * inv_r;
            let code = (u * dir_max_code).round();
            dir_codes[i] = code.clamp(-dir_max_code, dir_max_code) as i8;
        }

        // Сразу пересчитываем остаток после первого уровня.
        for i in 0..len {
            let u_hat = dir_codes[i] as f32 / dir_max_code;
            let x_hat = nucleus + radius * u_hat;
            residual[i] = x[i] - x_hat;
        }
    } else {
        residual.fill(0.0);
    }
    let dir_q = pack_codes(&dir_codes, dir_bits);

    // Уровни 2+.
    let mut residual_levels = Vec::new();

    for &b in bits.iter().skip(1) {
        if b < 2 || b > 8 {
            continue;
        }

        let max_code = ((1u32 << ((b - 1) as u32)) - 1) as f32;
        let m = max_abs(&residual).max(1e-12);
        let scale = m / max_code;
        let inv_scale = 1.0 / scale;

        let mut codes = vec![0i8; len];

        for i in 0..len {
            let code = (residual[i] * inv_scale).round();
            codes[i] = code.clamp(-max_code, max_code) as i8;
        }

        for i in 0..len {
            residual[i] -= scale * codes[i] as f32;
        }

        residual_levels.push(AtomLevel {
            bits: b,
            scale,
            q: pack_codes(&codes, b),
        });
    }

    let sig = simd::xor_signature(x);

    AtomCompressed {
        len,
        nucleus,
        radius,
        dir_bits,
        dir_max_code,
        dir_q,
        residual_levels,
        sig,
    }
}

/// Восстановить блок из многоуровневого атомарного формата.
pub fn decompress_atom(c: &AtomCompressed) -> Vec<f32> {
    let len = c.len;
    let mut out = vec![c.nucleus; len];

    if c.radius > 1e-12 && c.dir_max_code > 1e-12 {
        for i in 0..len {
            let u_hat = unpack_code(&c.dir_q, c.dir_bits, i) as f32 / c.dir_max_code;
            out[i] += c.radius * u_hat;
        }
    }

    for level in &c.residual_levels {
        for i in 0..len {
            let code = unpack_code(&level.q, level.bits, i) as f32;
            out[i] += level.scale * code;
        }
    }

    out
}

/// Максимальная ошибка реконструкции.
pub fn atom_reconstruction_error(original: &[f32], c: &AtomCompressed) -> f32 {
    let rec = decompress_atom(c);

    original
        .iter()
        .zip(rec.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Проверка живости сжатого блока через XOR.
pub fn atom_is_alive(c: &AtomCompressed) -> bool {
    c.sig.count_ones() > 10
}

/// Фактическая память кодов + метаданные (без накладных расходов Vec).
/// Коды действительно упакованы по `bits`, поэтому это не «теоретическая»
/// оценка: `dir_q.len()` и `level.q.len()` — реальные длины буферов.
pub fn estimated_bytes(c: &AtomCompressed) -> f32 {
    let dir_bytes = c.dir_q.len() as f32;
    let residual_bytes: f32 = c.residual_levels.iter().map(|l| l.q.len() as f32).sum();
    // nucleus + radius + направление + scales + metadata
    let metadata_bytes = 16.0 + 4.0 * c.residual_levels.len() as f32;
    dir_bytes + residual_bytes + metadata_bytes
}

/// Байты на один параметр.
pub fn bytes_per_param(c: &AtomCompressed) -> f32 {
    if c.len == 0 {
        return 0.0;
    }

    estimated_bytes(c) / c.len as f32
}
