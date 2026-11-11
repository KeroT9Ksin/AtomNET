// src/atom/math/enigma.rs
//! Enigma: математика преобразований на слитых SIMD-ядрах, данные — в Flash.
//! Сфера, тангенс, нелинейный гейт + XOR-подписи для связи с Rineta/Micron.
//!
//! Архитектура: вычисления идут через Flash-фасад (диспетчер над SIMD),
//! поэтому данные живут в Flash-буферах, а математика — слитые AVX2-ядра
//! (детерминированный порядок → без численного шума).

use crate::atom::flash::flash::Aligned64;
use crate::atom::flash::flash_facade::Flash;
use crate::atom::math::simd;

/// Проекция на сферу: w ← w/||w||. Возвращает норму до проекции.
pub fn project_sphere(w: &mut [f32]) -> f32 {
    let n = Flash::norm(w);
    if n > 1e-12 {
        Flash::scale(w, 1.0 / n);
    }
    n
}

/// Спроецировать src на сферу, записав результат в dst без аллокаций:
/// norm(src) одним проходом + dst = src·(1/n) вторым (scale_to, AVX2).
pub fn project_into(src: &[f32], dst: &mut [f32]) -> f32 {
    let n = Flash::norm(src);
    if n > 1e-12 {
        let inv = 1.0 / n;
        if std::ptr::eq(src.as_ptr(), dst.as_ptr()) {
            Flash::scale(dst, inv);
        } else {
            Flash::scale_to(dst, src, inv);
        }
    } else if dst.len() == src.len() {
        dst.copy_from_slice(src);
    }
    n
}

/// Риманов тангенциальный шаг (чистая геометрия, без boost).
/// g_тан = g − (g·ŵ)ŵ ;  w ← normalize(w − lr·g_тан).
/// Слито в один FMA-проход: w ← (1+lr·gw)·w + (−lr)·g.
pub fn tangent_step(w: &mut [f32], g: &[f32], lr: f32) {
    let gw = Flash::dot(g, w);
    let a = 1.0 + lr * gw;
    let b = -lr;
    Flash::axpby(w, a, g, b);
    project_sphere(w);
}

/// Нелинейный гейт (swish): x / (1 + e^{−x}).
#[inline(always)]
pub fn gate(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Чистое Enigma-преобразование (без boost):
/// y = r·(x/||x||)·gate(||x||).  Слито: y = k·x, k = r·gate(n)/n.
pub fn transform(y: &mut [f32], x: &[f32], r: f32) {
    let n = Flash::norm(x);
    if n < 1e-12 {
        y.fill(0.0);
        return;
    }
    let k = r * gate(n) / n;
    Flash::scale_to(y, x, k);
}

// ═══════════════════════════════════════════════════════════
// Данные в Flash: версии для Flash-буферов (Aligned64<f32>)
// ═══════════════════════════════════════════════════════════

/// Проекция Flash-буфера на сферу.
pub fn project_sphere_flash(buf: &mut Aligned64<f32>) -> f32 {
    project_sphere(buf.as_mut_slice())
}

/// Enigma-преобразование из Flash-буфера в Flash-буфер.
pub fn transform_flash(y: &mut Aligned64<f32>, x: &Aligned64<f32>, r: f32) {
    transform(y.as_mut_slice(), x.as_slice(), r);
}

// ═══════════════════════════════════════════════════════════
// XOR-интеграция: связь Enigma с Rineta и Micron через подписи
// ═══════════════════════════════════════════════════════════

/// Enigma-подпись: XOR-подпись вектора ПОСЛЕ проекции на сферу.
pub fn enigma_signature(w: &[f32]) -> u64 {
    let mut copy = w.to_vec();
    project_sphere(&mut copy);
    simd::xor_signature(&copy)
}

/// Проверка: сохранилась ли сферическая XOR-структура после преобразования?
pub fn verify_sphere(original: &[f32], transformed: &[f32]) -> (bool, u32) {
    let sig_orig = enigma_signature(original);
    let sig_trans = enigma_signature(transformed);
    let hamming = (sig_orig ^ sig_trans).count_ones();
    (hamming <= 4, hamming)
}

/// Связь Enigma ↔ Micron: потеря информации при квантовании на сфере.
pub fn verify_with_micron(
    original: &[f32],
    micron_quantized: &[i8],
    mu: f32,
    sc: f32,
) -> (bool, u32, f32) {
    let sig_orig = enigma_signature(original);
    let mut dequant: Vec<f32> = micron_quantized
        .iter()
        .map(|&v| mu + sc * v as f32)
        .collect();
    project_sphere(&mut dequant);
    let sig_micron = simd::xor_signature(&dequant);
    let hamming = (sig_orig ^ sig_micron).count_ones();
    let loss_pct = hamming as f32 / 64.0 * 100.0;
    (hamming <= 8, hamming, loss_pct)
}

/// Cold start для Enigma: проверка, что сферические веса не "холодные".
pub fn enigma_cold_start_hamming(w: &[f32]) -> u32 {
    enigma_signature(w).count_ones()
}

/// Проверка "живости" сферического состояния.
pub fn enigma_is_alive(w: &[f32]) -> bool {
    enigma_cold_start_hamming(w) > 10
}

/// Связь Enigma ↔ Rineta: сохраняет ли импульс сферическую структуру.
pub fn coherence_with_rin(w: &[f32], rin_data: &[f32]) -> (bool, u32) {
    let sig_w = enigma_signature(w);
    let sig_rin = simd::xor_signature(rin_data);
    let hamming = (sig_w ^ sig_rin).count_ones();
    (hamming <= 16, hamming)
}
