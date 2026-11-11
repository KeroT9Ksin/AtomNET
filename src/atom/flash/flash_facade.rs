// src/atom/flash/flash_facade.rs
//! Flash: универсальный фасад backend'а. Все компоненты зовут его.

use crate::atom::flash::flash::FlashDense;
use crate::atom::flash::xor_alloc::XorAlloc;
use crate::atom::math::simd;

pub struct Flash;

impl Flash {
    // ── BLAS-ядра ──
    #[inline] pub fn dot(a: &[f32], b: &[f32]) -> f32 { simd::dot(a, b) }
    #[inline] pub fn axpy(y: &mut [f32], k: f32, x: &[f32]) { simd::axpy(y, k, x) }
    #[inline] pub fn norm(a: &[f32]) -> f32 { simd::norm(a) }
    #[inline] pub fn gemv(d: &FlashDense, x: &[f32]) -> Vec<f32> { d.gemv(x) }
    #[inline] pub fn gemv_bias(d: &FlashDense, x: &[f32]) -> Vec<f32> { d.gemv_bias(x) }
    #[inline] pub fn gemv_t(d: &FlashDense, g: &[f32]) -> Vec<f32> { d.gemv_t(g) }

    // ── Битовая аллокация ──
    #[inline] pub fn liveness(sig: u64) -> u32 { XorAlloc::liveness(sig) }
    #[inline] pub fn sim(a: u64, b: u64) -> u32 { XorAlloc::sim(a, b) }
    #[inline] pub fn plan(s: &[u64], h: &[u32], cr: usize) -> Vec<u8> { XorAlloc::plan(s, h, cr) }
    #[inline] pub fn scale(y: &mut [f32], k: f32) { simd::scale(y, k) }
    #[inline] pub fn scale_to(y: &mut [f32], x: &[f32], k: f32) { simd::scale_to(y, x, k) }
    #[inline] pub fn axpby(y: &mut [f32], a: f32, x: &[f32], b: f32) { simd::axpby(y, a, x, b) }
    #[inline] pub fn add_elem(y: &mut [f32], x: &[f32]) { simd::add_elem(y, x) }
    #[inline] pub fn mul_elem(y: &mut [f32], x: &[f32]) { simd::mul_elem(y, x) }
    #[inline] pub fn relu(y: &mut [f32]) { simd::relu(y) }
    #[inline] pub fn silu(y: &mut [f32]) { simd::silu(y) }
    #[inline] pub fn silu_fast(y: &mut [f32]) { simd::silu_fast(y) }
}
