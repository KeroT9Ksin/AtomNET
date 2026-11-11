// src/atom_math/micron.rs
//! Micron: математика сжатия. int8-блоки с гарантией + электронные подписи + cold start.
use crate::atom::math::simd;


pub const MICRON_BLOCK: usize = 256;

/// q_i = clamp(round((x_i - mu)/s), -128, 127); гарантия |x - x_hat| <= s/2
pub fn quantize_int8(b: &[f32]) -> (Vec<i8>, f32, f32) {
    // Чистый итератор (без rayon) — быстрее для средних массивов
    let mu: f32 = b.iter().sum::<f32>() / b.len().max(1) as f32;
    let m: f32 = b.iter().map(|&x| (x - mu).abs()).fold(0.0f32, f32::max);
    let sc = (m / 127.0).max(1e-12);
    let inv = 1.0 / sc;
    let mut q = vec![0i8; b.len()];
    
    if simd::has_avx2() {
        unsafe { quant_i8_avx2_runtime(b, mu, inv, &mut q) };
    } else {
        for (i, &x) in b.iter().enumerate() {
            q[i] = ((x - mu) * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q, mu, sc)
}

pub fn dequantize_int8(q: &[i8], mu: f32, sc: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; q.len()];
    if simd::has_avx2() {
        unsafe { dequant_i8_avx2_runtime(q, sc, &mut out) };
        for v in out.iter_mut() { *v += mu; }
    } else {
        for (i, &v) in q.iter().enumerate() {
            out[i] = mu + sc * v as f32;
        }
    }
    out
}

/// XOR-подпись блока (связь с Enigma/Rineta)
pub fn micron_signature(b: &[f32]) -> u64 {
    simd::xor_signature(b)
}

/// Хэммингово расстояние между двумя подписями
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Cold start: "холодная" подпись (все нули) для детектирования первого использования
pub const COLD_SIGNATURE: u64 = 0x0000_0000_0000_0000;

/// Hamming distance от cold start: сколько бит изменилось с "холодного" состояния
pub fn cold_start_hamming(b: &[f32]) -> u32 {
    let sig = micron_signature(b);
    (sig ^ COLD_SIGNATURE).count_ones()
}

/// Проверка целостности: совпадают ли подписи до/после квантования
pub fn verify_integrity(original: &[f32], quantized: &[i8], mu: f32, sc: f32) -> (bool, u32) {
    let sig_orig = simd::xor_signature(original);
    let dequant: Vec<f32> = quantized.iter().map(|&v| mu + sc * v as f32).collect();
    let sig_dequant = simd::xor_signature(&dequant);
    let h = (sig_orig ^ sig_dequant).count_ones();
    (h == 0, h)
}

/// Cold start для Micron: проверка, что квантованные данные не "мёртвые".
pub fn micron_cold_start_hamming(b: &[f32]) -> u32 {
    let sig = micron_signature(b);
    sig.count_ones()
}

/// Проверка "живости" блока Micron.
pub fn micron_is_alive(b: &[f32]) -> bool {
    micron_cold_start_hamming(b) > 10
}

// ═══════════════════════════════════════════════════════════
// Runtime dispatch (без #[target_feature], чтобы избежать проблем с вызовом)
// ═══════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
unsafe fn quant_i8_avx2_runtime(data: &[f32], mu: f32, inv: f32, out: &mut [i8]) {
    use std::arch::x86_64::*;
    let n = data.len().min(out.len());
    let chunks = n / 8;
    let vmu = _mm256_set1_ps(mu);
    let vinv = _mm256_set1_ps(inv);
    let mut t = [0i32; 8];
    for c in 0..chunks {
        let o = c * 8;
        let v = _mm256_sub_ps(_mm256_loadu_ps(data.as_ptr().add(o)), vmu);
        let s = _mm256_mul_ps(v, vinv);
        let r = _mm256_round_ps::<0x8>(s);
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, _mm256_cvtps_epi32(r));
        for i in 0..8 { out[o + i] = t[i].clamp(-128, 127) as i8; }
    }
    for i in chunks * 8..n {
        out[i] = ((data[i] - mu) * inv).round().clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn quant_i8_avx2_runtime(_data: &[f32], _mu: f32, _inv: f32, _out: &mut [i8]) {
    unreachable!("AVX2 not available on this architecture");
}

#[cfg(target_arch = "x86_64")]
unsafe fn dequant_i8_avx2_runtime(q: &[i8], scale: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = q.len().min(out.len());
    let chunks = n / 8;
    let vs = _mm256_set1_ps(scale);
    for c in 0..chunks {
        let o = c * 8;
        let bytes = _mm_loadu_si64(q.as_ptr().add(o) as *const _);
        let f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
        _mm256_storeu_ps(out.as_mut_ptr().add(o), _mm256_mul_ps(f, vs));
    }
    for i in chunks * 8..n { out[i] = q[i] as f32 * scale; }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dequant_i8_avx2_runtime(_q: &[i8], _scale: f32, _out: &mut [f32]) {
    unreachable!("AVX2 not available on this architecture");
}

/// Двухуровневое квантование: int4-база + int8-остаток («f32-подобие»).
pub struct Q4Resid {
    pub q4: Vec<u8>,   // упакованные нибблы
    pub q8: Vec<i8>,   // остаток
    pub s4: Vec<f32>, pub z4: Vec<f32>,
    pub s8: Vec<f32>, pub z8: Vec<f32>,
    pub n: usize,
    pub block: usize,
}

pub fn quantize_q4_resid(w: &[f32], block: usize) -> Q4Resid {
    let n = w.len();
    let nb = (n + block - 1) / block;
    let mut q = Q4Resid {
        q4: vec![0u8; (n + 1) / 2],
        q8: vec![0i8; n],
        s4: Vec::with_capacity(nb), z4: Vec::with_capacity(nb),
        s8: Vec::with_capacity(nb), z8: Vec::with_capacity(nb),
        n, block,
    };
    let mut codes = vec![0i8; block];

    for b in 0..nb {
        let o = b * block;
        let upto = (o + block).min(n);

        // Уровень 1: int4
        let mut m4 = 0.0f32; for i in o..upto { m4 += w[i]; } m4 /= (upto - o) as f32;
        let mut M4 = 0.0f32; for i in o..upto { M4 = M4.max((w[i] - m4).abs()); }
        let s4 = (M4 / 7.0).max(1e-12);
        for (j, i) in (o..upto).enumerate() {
            codes[j] = ((w[i] - m4) / s4).round().clamp(-8.0, 7.0) as i8;
        }
        for (j, i) in (o..upto).enumerate() {
            let k = i / 2;
            if i % 2 == 0 { q.q4[k] = (codes[j] + 8) as u8; }
            else { q.q4[k] |= ((codes[j] + 8) as u8) << 4; }
        }

        // Уровень 2: int8-остаток
        let mut m8 = 0.0f32;
        for (j, i) in (o..upto).enumerate() { m8 += w[i] - (codes[j] as f32 * s4 + m4); }
        m8 /= (upto - o) as f32;
        let mut M8 = 0.0f32;
        for (j, i) in (o..upto).enumerate() {
            let r = w[i] - (codes[j] as f32 * s4 + m4);
            M8 = M8.max((r - m8).abs());
        }
        let s8 = (M8 / 127.0).max(1e-12);
        for (j, i) in (o..upto).enumerate() {
            let r = w[i] - (codes[j] as f32 * s4 + m4);
            q.q8[i] = ((r - m8) / s8).round().clamp(-127.0, 127.0) as i8;
        }

        q.s4.push(s4); q.z4.push(m4); q.s8.push(s8); q.z8.push(m8);
    }
    q
}

pub fn dequantize_q4_resid(q: &Q4Resid, out: &mut [f32]) {
    for i in 0..q.n {
        let b = i / q.block;
        let byte = q.q4[i / 2];
        let c4 = if i % 2 == 0 { (byte & 0x0F) as f32 } else { (byte >> 4) as f32 };
        let base = (c4 - 8.0) * q.s4[b] + q.z4[b];
        out[i] = base + q.q8[i] as f32 * q.s8[b] + q.z8[b];
    }
}

pub fn q4_resid_bytes(q: &Q4Resid) -> usize {
    q.q4.len() + q.q8.len() + (q.s4.len() + q.z4.len() + q.s8.len() + q.z8.len()) * 4
}
