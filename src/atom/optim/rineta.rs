// src/atom_math/rineta.rs
//! Rineta: резонансная математика. Волны, когерентность, gain, rin[]-состояния.
use crate::atom::math::{enigma, simd};

pub const RIN_BLOCK: usize = 4096;

/// Направленная когерентность cos(p, g)
pub fn coherence_pg(p: &[f32], g: &[f32]) -> f32 {
    (simd::dot(p, g) / (simd::norm(p) * simd::norm(g) + 1e-8)).clamp(-1.0, 1.0)
}

/// Быстрый gain без powf (alpha ∈ {2, 1.5, 1, 0.5})
#[inline(always)]
pub fn gain_fast(s: f32, alpha: f32) -> f32 {
    if alpha == 2.0 { (1.0 + s * s).sqrt().recip() }
    else if alpha == 1.0 { (1.0 + s).recip() }
    else if alpha == 0.5 { let t = 1.0 + s.sqrt(); (t * t).recip() }
    else { let r = s * s.sqrt(); let t = 1.0 + r; (t * t).cbrt().recip() }
}

// ═══════════════════════════════════════════════════════════
// Rin — вектор с волновым состоянием (ядро физики)
// ═══════════════════════════════════════════════════════════
pub struct Rin {
    pub data: Vec<f32>,
    pub coherence: f32,
    pub flip_ema: f32,
    pub energy_ema: f32,
    sign_prev: f32,
}

impl Rin {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data, coherence: 0.0, flip_ema: 0.0, energy_ema: 0.0, sign_prev: 1.0 }
    }

    /// Направленная когерентность + обновление волнового состояния
    pub fn coherence_with(&mut self, g: &[f32]) -> f32 {
        let d = simd::dot(&self.data, g);
        let np = simd::norm(&self.data);
        let ng = simd::norm(g);
        let c = (d / (np * ng + 1e-8)).clamp(-1.0, 1.0);
        self.coherence = 0.9 * self.coherence + 0.1 * c;
        let sign = if d >= 0.0 { 1.0 } else { -1.0 };
        self.flip_ema = 0.9 * self.flip_ema + 0.1 * if sign != self.sign_prev { 1.0 } else { 0.0 };
        self.sign_prev = sign;
        self.energy_ema = 0.9 * self.energy_ema + 0.1 * ng;
        c
    }

    /// Резонансный множитель (даёт буст SIMD-ядрам)
    #[inline]
    pub fn boost(&self) -> f32 {
        1.0 + 0.55 * (5.0 * self.coherence).tanh() / (1.0 + 0.5 * self.flip_ema)
    }

    /// Импульс + boost-обновление весов in-place
    pub fn apply_into(&mut self, w: &mut [f32], g: &[f32], lr: f32, b1: f32) -> f32 {
        let c = self.coherence_with(g);
        simd::momentum_update(&mut self.data, g, b1);
        simd::axpy(w, -lr * self.boost(), &self.data);
        c
    }

    /// Сферический шаг (Enigma-интеграция)
    pub fn spherical_step(&mut self, w: &mut [f32], g: &[f32], lr: f32, b1: f32) -> f32 {
        let c = self.coherence_with(g);
        simd::momentum_update(&mut self.data, g, b1);
        let gw = simd::dot(g, w);
        let boost = self.boost();
        for (wv, &gv) in w.iter_mut().zip(g) {
            *wv -= lr * boost * (gv - gw * *wv);
        }
        enigma::project_sphere(w);
        c
    }

    /// Enigma-преобразование с boost от Rin
    pub fn enigma_transform(&self, y: &mut [f32], x: &[f32], r: f32) {
        let n = simd::norm(x);
        if n < 1e-12 { y.fill(0.0); return; }
        let g = enigma::gate(n);
        let boost = self.boost();
        for (yv, xv) in y.iter_mut().zip(x) {
            *yv = r * boost * (*xv / n) * g;
        }
    }

    /// XOR-подпись состояния (связь с Micron)
    pub fn signature(&self) -> u64 {
        simd::xor_signature(&self.data)
    }

    /// Сколько бит изменилось с прошлой подписи
    pub fn state_changed(&self, prev_sig: u64) -> u32 {
        simd::hamming_distance(prev_sig, self.signature())
    }
    
    /// Cold start: отдалённость от нулевого состояния (0 = мёртвое, 32 = живое).
pub fn cold_start_hamming(&self) -> u32 {
    let sig = self.signature();
    sig.count_ones()  // Hamming distance от нуля
}

/// Проверка "живости" состояния: должно быть > 10 бит из 64.
pub fn is_alive(&self) -> bool {
    self.cold_start_hamming() > 10
}

/// "Пробуждение" состояния: если мёртвое, добавляем шум для инициализации.
pub fn warmup(&mut self, noise_scale: f32) {
    if !self.is_alive() {
        let mut rng = XorShift64::new(42);
        for v in &mut self.data {
            *v += (rng.next_f32() * 2.0 - 1.0) * noise_scale;
        }
    }
}
}

// ═══════════════════════════════════════════════════════════
// RinArray — блочный массив [Rin]
// ═══════════════════════════════════════════════════════════
pub struct RinArray {
    pub blocks: Vec<Rin>,
    pub scales: Vec<f32>,
    pub block_size: usize,
}

impl RinArray {
    pub fn from_vec(v: Vec<f32>, block_size: usize) -> Self {
        let mut blocks = Vec::new();
        let mut scales = Vec::new();
        for chunk in v.chunks(block_size) {
            let m = chunk.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-12);
            scales.push(m);
            blocks.push(Rin::new(chunk.to_vec()));
        }
        Self { blocks, scales, block_size }
    }
    pub fn boost(&self) -> f32 {
        if self.blocks.is_empty() { 1.0 }
        else { self.blocks.iter().map(|b| b.boost()).sum::<f32>() / self.blocks.len() as f32 }
    }
}

// ═══════════════════════════════════════════════════════════
// Legacy Rineta v4 (слоевой бустер) — для совместимости PyRineta
// ═══════════════════════════════════════════════════════════
pub struct Rineta {
    pub beta1: f32,
    pub alpha: f32,
    pub kappa: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub warmup_steps: usize,
    pub current_step: usize,
    momentum: Vec<f32>,
    layer_coherences: Vec<f32>,
    layer_energy: Vec<f32>,
    layer_energy_history: Vec<Vec<f32>>,
}

struct XorShift64 { state: u64 }
impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_f32(&mut self) -> f32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x;
        (x >> 40) as f32 / (1_u64 << 24) as f32
    }
}

impl Rineta {
    pub fn new(beta1: f32, alpha: f32, kappa: f32, gamma: f32, lambda: f32,
               warmup_steps: usize, num_layers: usize) -> Self {
        Self {
            beta1, alpha, kappa, gamma, lambda, warmup_steps, current_step: 0,
            momentum: vec![0.0; num_layers],
            layer_coherences: vec![0.0; num_layers],
            layer_energy: vec![0.0; num_layers],
            layer_energy_history: (0..num_layers).map(|_| Vec::with_capacity(5)).collect(),
        }
    }

    pub fn update_layer(&mut self, grads: &mut [f32], li: usize) -> f32 {
        if self.current_step < self.warmup_steps { return 1.0; }
        let grad_norm: f32 = grads.iter().map(|&g| g * g).sum::<f32>().sqrt();
        let p_prev = self.momentum[li];
        let p_new = self.beta1 * p_prev + (1.0 - self.beta1) * grad_norm;
        self.momentum[li] = p_new;
        let dot: f32 = grads.iter().map(|&g| g * g).sum();
        let c = dot / ((p_new.sqrt() + 1e-8) * grad_norm + 1e-8);
        self.layer_coherences[li] = c;
        if self.layer_energy_history[li].len() >= 5 { self.layer_energy_history[li].remove(0); }
        self.layer_energy_history[li].push(grad_norm);
        let h = &self.layer_energy_history[li];
        let delta_e = if h.len() > 1 { (h[0] - h[h.len() - 1]).abs() } else { 0.0 };
        self.layer_energy[li] = delta_e;
        let amp = 1.0 + self.alpha * (self.kappa * c).tanh() * (1.0 + self.gamma * delta_e);
        let d = grads.len() as f32;
        let mean = grads.iter().sum::<f32>() / d;
        let var = grads.iter().map(|&g| (g - mean).powi(2)).sum::<f32>() / d;
        let s = p_new.sqrt() / (d.sqrt() * var.sqrt() + 1e-8);
        let gain = (1.0 + s.powf(self.alpha)).powf(-1.0 / self.alpha);
        let scale = amp * gain * (1.0 - self.lambda * (1.0 - c).max(0.0));
        grads.iter_mut().for_each(|g| *g *= scale);
        let h_eff = h.iter().sum::<f32>() / h.len().max(1) as f32;
        let q_amp = if delta_e > 0.0 && h_eff > 1e-8 { (delta_e / h_eff).powf(1.0 / d) } else { 0.0 };
        if q_amp > 0.01 {
            let a = q_amp.min(0.5);
            let mut rng = XorShift64::new(li as u64 ^ self.current_step as u64);
            for g in grads.iter_mut() { *g += (rng.next_f32() * 2.0 - 1.0) * a; }
        }
        scale
    }

    pub fn get_stats(&self) -> (f32, f32, f32) {
        let len = self.layer_coherences.len().max(1) as f32;
        (self.layer_coherences.iter().sum::<f32>() / len,
         self.layer_coherences.iter().map(|&c| 1.0 + self.alpha * (self.kappa * c).tanh()).sum::<f32>() / len,
         self.layer_energy.iter().sum::<f32>() / len)
    }

    pub fn reset(&mut self) {
        self.current_step = 0;
        self.momentum.fill(0.0);
        self.layer_coherences.fill(0.0);
        self.layer_energy.fill(0.0);
        for h in &mut self.layer_energy_history { h.clear(); }
    }
}
