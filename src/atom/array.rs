// src/atom/array.rs
//! AtomArray — физический массив AtomNET: замена NumPy-буфера для ML.
//!
//! Каждый массив — row-major f32, contiguous, с физической метадатой:
//! XOR-подпись (целостность), жизнь (AtomLife), энергия.
//! Все горячие операции идут через SIMD/FLASH-ядра AtomNET (ноль копий
//! между matmul → активация → softmax), а не через numpy-мост.

use crate::atom::flash::flash_gemm;
use crate::atom::life::life::{AtomLife, LifeParams};
use crate::atom::math::{micron, simd};

// ═══════════════════════════════════════════════════════════
// Xoroshiro256** — детерминированный RNG для фабрик.
// Тот же алгоритм, что в torchx (фиксированный seed → воспроизводимо).
// ═══════════════════════════════════════════════════════════

#[derive(Clone)]
pub struct Xoroshiro {
    s: [u64; 4],
}

impl Xoroshiro {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed;
        for slot in s.iter_mut() {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut t = z;
            t = (t ^ (t >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            t = (t ^ (t >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = t ^ (t >> 31);
        }
        Self { s }
    }

    pub fn next_f32(&mut self) -> f32 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9) as f64;
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        (result / (u64::MAX as f64)) as f32
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ═══════════════════════════════════════════════════════════
// AtomArray
// ═══════════════════════════════════════════════════════════

/// Физический массив AtomNET (f32, row-major, contiguous).
/// Физика: sig (XOR-подпись), life (AtomLife), energy (норма).
pub struct AtomArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub sig: u64,
    pub life: AtomLife,
    pub energy: f32,
}

impl AtomArray {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "AtomArray: data len {} != numel {}", data.len(), n);
        // Ленивая физика: sig — один дешёвый XOR-проход, энергия/норма
        // считаются по требованию norm() (иначе каждый elementwise-результат
        // платил бы 2 лишних O(n) прохода). Life получает energy=1.0 как
        // плейсхолдер: живость определяется по sig (биты), а не по энергии.
        let sig = simd::xor_fold(&data);
        let mut life = AtomLife::new(LifeParams::default());
        life.observe(sig, 1.0, 1.0);
        Self { shape, data, sig, life, energy: 0.0 }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Пересчитать физику после in-place мутации (только sig — дёшево;
    /// энергия ленивая, см. norm()/energy()).
    pub fn sync_physics(&mut self) {
        self.sig = simd::xor_fold(&self.data);
        self.energy = 0.0;
        self.life.observe(self.sig, 1.0, 1.0);
    }

    /// Энергия массива (норма, считается по требованию).
    pub fn energy(&self) -> f32 {
        if self.energy != 0.0 {
            self.energy
        } else {
            simd::norm(&self.data)
        }
    }

    // ── фабрики (конструкция без numpy) ──────────────────────

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![0.0f32; n])
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![1.0f32; n])
    }

    pub fn empty(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![0.0f32; n])
    }

    pub fn full(shape: Vec<usize>, val: f32) -> Self {
        let n: usize = shape.iter().product();
        Self::new(shape, vec![val; n])
    }

    pub fn arange(n: usize) -> Self {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Self::new(vec![n], data)
    }

    pub fn linspace(start: f32, stop: f32, num: usize) -> Self {
        let data: Vec<f32> = if num <= 1 {
            vec![start]
        } else {
            let step = (stop - start) / (num - 1) as f32;
            (0..num).map(|i| start + step * i as f32).collect()
        };
        Self::new(vec![num], data)
    }

    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(vec![n, n], data)
    }

    pub fn rand(shape: Vec<usize>, seed: u64) -> Self {
        let n: usize = shape.iter().product();
        let mut rng = Xoroshiro::new(seed);
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        Self::new(shape, data)
    }

    pub fn randn(shape: Vec<usize>, seed: u64) -> Self {
        let n: usize = shape.iter().product();
        let mut rng = Xoroshiro::new(seed);
        let data: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        Self::new(shape, data)
    }

    // ── pojo-совместимость: строки (2D) ─────────────────────

    pub fn rows(&self) -> usize {
        if self.shape.len() == 2 { self.shape[0] } else { 1 }
    }

    pub fn cols(&self) -> usize {
        if self.shape.len() == 2 { self.shape[1] }
        else if self.shape.len() == 1 { self.shape[0] }
        else { self.numel() }
    }

    // ── elementwise (out-of-place, новые массивы) ────────────

    pub fn add(&self, other: &AtomArray) -> AtomArray {
        assert_eq!(self.numel(), other.numel(), "add: numel mismatch");
        let mut data = self.data.clone();
        simd::add_elem(&mut data, &other.data);
        AtomArray::new(self.shape.clone(), data)
    }

    pub fn sub(&self, other: &AtomArray) -> AtomArray {
        assert_eq!(self.numel(), other.numel(), "sub: numel mismatch");
        let mut data = self.data.clone();
        for (d, o) in data.iter_mut().zip(&other.data) {
            *d -= *o;
        }
        AtomArray::new(self.shape.clone(), data)
    }

    pub fn mul(&self, other: &AtomArray) -> AtomArray {
        assert_eq!(self.numel(), other.numel(), "mul: numel mismatch");
        let mut data = self.data.clone();
        simd::mul_elem(&mut data, &other.data);
        AtomArray::new(self.shape.clone(), data)
    }

    pub fn div(&self, other: &AtomArray) -> AtomArray {
        assert_eq!(self.numel(), other.numel(), "div: numel mismatch");
        let mut data = self.data.clone();
        for (d, o) in data.iter_mut().zip(&other.data) {
            *d /= *o;
        }
        AtomArray::new(self.shape.clone(), data)
    }

    pub fn add_scalar(&self, v: f32) -> AtomArray {
        let mut data = self.data.clone();
        for x in data.iter_mut() { *x += v; }
        AtomArray::new(self.shape.clone(), data)
    }

    pub fn mul_scalar(&self, v: f32) -> AtomArray {
        let mut data = self.data.clone();
        simd::scale(&mut data, v);
        AtomArray::new(self.shape.clone(), data)
    }

    pub fn neg(&self) -> AtomArray {
        let mut data = self.data.clone();
        simd::scale(&mut data, -1.0);
        AtomArray::new(self.shape.clone(), data)
    }

    // ── активации in-place (быстрые, без аллокаций) ──────────

    pub fn relu_(&mut self) {
        simd::relu(&mut self.data);
        self.sync_physics();
    }

    pub fn silu_(&mut self) {
        simd::silu(&mut self.data);
        self.sync_physics();
    }

    pub fn silu_fast_(&mut self) {
        simd::silu_fast(&mut self.data);
        self.sync_physics();
    }

    pub fn softmax_(&mut self) {
        // 2D: softmax по строкам (последняя ось); иначе плоско.
        if self.ndim() == 2 {
            let cols = self.cols();
            for r in 0..self.rows() {
                let s = r * cols;
                simd::softmax(&mut self.data[s..s + cols]);
            }
        } else {
            simd::softmax(&mut self.data);
        }
        self.sync_physics();
    }

    pub fn gelu_(&mut self) {
        for v in self.data.iter_mut() {
            *v = crate::atom::math::tensor::gelu(*v);
        }
        self.sync_physics();
    }

    pub fn sigmoid_(&mut self) {
        for v in self.data.iter_mut() {
            *v = crate::atom::math::tensor::sigmoid(*v);
        }
        self.sync_physics();
    }

    pub fn tanh_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.tanh();
        }
        self.sync_physics();
    }

    pub fn exp_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.exp();
        }
        self.sync_physics();
    }

    pub fn log_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.ln();
        }
        self.sync_physics();
    }

    pub fn sqrt_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.sqrt();
        }
        self.sync_physics();
    }

    pub fn abs_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.abs();
        }
        self.sync_physics();
    }

    pub fn sin_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.sin();
        }
        self.sync_physics();
    }

    pub fn cos_(&mut self) {
        for v in self.data.iter_mut() {
            *v = v.cos();
        }
        self.sync_physics();
    }

    pub fn clip_(&mut self, lo: f32, hi: f32) {
        for v in self.data.iter_mut() {
            *v = v.clamp(lo, hi);
        }
        self.sync_physics();
    }

    pub fn affine_(&mut self, scale: f32, shift: f32) {
        // y = x*scale + shift через один FMA-проход (без временного массива)
        let mut out = vec![0.0f32; self.data.len()];
        simd::affine_into(&self.data, scale, shift, &mut out);
        std::mem::swap(&mut self.data, &mut out);
        self.sync_physics();
    }

    // ── редукции (один проход, без временных массивов) ───────

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() { 0.0 } else { self.sum() / self.data.len() as f32 }
    }

    pub fn var(&self) -> f32 {
        simd::mean_var(&self.data).1
    }

    pub fn std(&self) -> f32 {
        self.var().max(0.0).sqrt()
    }

    pub fn mean_var(&self) -> (f32, f32) {
        simd::mean_var(&self.data)
    }

    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0)
    }

    pub fn argmin(&self) -> usize {
        self.data.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i).unwrap_or(0)
    }

    pub fn norm(&self) -> f32 {
        simd::norm(&self.data)
    }

    pub fn dot(&self, other: &AtomArray) -> f32 {
        assert_eq!(self.numel(), other.numel(), "dot: numel mismatch");
        simd::dot(&self.data, &other.data)
    }

    pub fn mse(&self, other: &AtomArray) -> f32 {
        assert_eq!(self.numel(), other.numel(), "mse: numel mismatch");
        simd::mse(&self.data, &other.data)
    }

    pub fn allclose(&self, other: &AtomArray, atol: f32) -> bool {
        if self.numel() != other.numel() { return false; }
        self.data.iter().zip(&other.data).all(|(a, b)| (a - b).abs() <= atol)
    }

    // ── линейная алгебра 2D ──────────────────────────────────

    pub fn matmul(&self, other: &AtomArray) -> AtomArray {
        assert!(self.ndim() == 2 && other.ndim() == 2, "matmul: 2D expected");
        let (m, k) = (self.rows(), self.cols());
        let (k2, n) = (other.rows(), other.cols());
        assert_eq!(k, k2, "matmul: inner dims mismatch {} vs {}", k, k2);
        let parts = flash_gemm::gemm_auto_parts(0, m, k, n);
        let mut c = vec![0.0f32; m * n];
        flash_gemm::gemm_blocked_into(&self.data, m, k, &other.data, n, parts, &mut c);
        AtomArray::new(vec![m, n], c)
    }

    pub fn matvec(&self, x: &AtomArray) -> AtomArray {
        assert!(self.ndim() == 2 && x.ndim() == 1, "matvec: W 2D, x 1D expected");
        let (m, n) = (self.rows(), self.cols());
        assert_eq!(n, x.numel(), "matvec: len(x) != cols");
        let y = crate::atom::math::tensor::matvec_slice(&self.data, m, n, &x.data);
        AtomArray::new(vec![m], y)
    }

    pub fn transpose(&self) -> AtomArray {
        assert!(self.ndim() == 2, "transpose: 2D expected");
        let (m, n) = (self.rows(), self.cols());
        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                data[j * m + i] = self.data[i * n + j];
            }
        }
        AtomArray::new(vec![n, m], data)
    }

    pub fn t(&self) -> AtomArray {
        self.transpose()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> AtomArray {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: numel mismatch");
        AtomArray::new(shape, self.data.clone())
    }

    // ── физика: сфера, подписи, сжатие ───────────────────────

    /// Проекция на сферу (flat). Возвращает норму до проекции.
    pub fn project_sphere(&mut self) -> f32 {
        let n = crate::atom::math::enigma::project_sphere(&mut self.data);
        self.sync_physics();
        n
    }

    pub fn xor_signature(&self) -> u64 {
        simd::xor_signature(&self.data)
    }

    pub fn xor_fold(&self) -> u64 {
        simd::xor_fold(&self.data)
    }

    pub fn is_alive(&self) -> bool {
        self.life.is_active()
    }

    pub fn gate(&self) -> f32 {
        self.life.gate()
    }

    pub fn quantize_int8(&self) -> (Vec<i8>, f32, f32) {
        micron::quantize_int8(&self.data)
    }

    pub fn dequantize_int8(q: &[i8], mu: f32, sc: f32, shape: Vec<usize>) -> AtomArray {
        AtomArray::new(shape, micron::dequantize_int8(q, mu, sc))
    }

    pub fn pack_signs(&self) -> Vec<u8> {
        simd::pack_signs(&self.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factories_and_reductions() {
        let a = AtomArray::zeros(vec![2, 3]);
        assert_eq!(a.numel(), 6);
        assert_eq!(a.sum(), 0.0);
        let b = AtomArray::ones(vec![2, 3]);
        assert_eq!(b.sum(), 6.0);
        assert_eq!(b.mean(), 1.0);
        let c = AtomArray::arange(4);
        assert_eq!(c.data, vec![0.0, 1.0, 2.0, 3.0]);
        let (m, v) = c.mean_var();
        assert!((m - 1.5).abs() < 1e-6);
        assert!((v - 1.25).abs() < 1e-6);
    }

    #[test]
    fn elementwise_and_activations() {
        let a = AtomArray::new(vec![2], vec![-1.0, 2.0]);
        let b = AtomArray::new(vec![2], vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![2.0, 6.0]);
        let mut d = AtomArray::new(vec![2], vec![-1.0, 2.0]);
        d.relu_();
        assert_eq!(d.data, vec![0.0, 2.0]);
        d = AtomArray::new(vec![2], vec![0.0, 1.0]);
        d.sigmoid_();
        assert!((d.data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn matmul_2x2() {
        let a = AtomArray::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = AtomArray::new(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
        let t = a.transpose();
        assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn physics_bits() {
        let mut a = AtomArray::new(vec![3], vec![1.0, 2.0, 3.0]);
        let n = a.project_sphere();
        assert!((n - 14.0f32.sqrt()).abs() < 1e-5);
        assert!((a.norm() - 1.0).abs() < 1e-5);
        let (q, mu, sc) = a.quantize_int8();
        let back = AtomArray::dequantize_int8(&q, mu, sc, vec![3]);
        assert!(back.allclose(&a, sc.max(1e-6) * 0.6 + 1e-5));
    }
}
