// src/atom/life/qairo_life.rs
//! Qairo v2 + QairoCursor: живой атомарный слой на ядрах FLASH.
//! API не изменён — python.rs не переписываем.

use crate::atom::flash::flash::FlashDense;
use crate::atom::flash::flash_compressed::FlashCompressed;
use crate::atom::life::life::{AtomLife, LifeParams, LifeState};
use crate::atom::life::qairo_cursor::QairoCursor;
use crate::atom::math::simd;
use crate::atom::flash::flash_arena::{FlashArena, RegionManager};

pub struct QairoLivingAtom {
    pub life: AtomLife,
    pub nucleus: f32,
    pub radius: f32,
    pub sig: u64,
}

impl QairoLivingAtom {
    #[inline]
    pub fn gate(&self) -> f32 {
        self.life.gate()
    }
    #[inline]
    pub fn is_alive(&self) -> bool {
        self.life.is_active()
    }
}

pub struct QairoLivingLayer {
    pub dense: FlashDense,
    pub rows: Vec<QairoLivingAtom>,
    pub dim: usize,
    pub compressed: Option<FlashCompressed>,
    pub cursor: QairoCursor, // ← курсор ТОЛЬКО здесь
    pub arena: FlashArena,
    pub region: RegionManager,
}

impl QairoLivingLayer {
    /// Сигнатура не изменена: (weight_rows, bias) -> Option<Self>.
    pub fn new(weight_rows: Vec<Vec<f32>>, bias: Vec<f32>) -> Option<Self> {
        if weight_rows.is_empty() || weight_rows.len() != bias.len() {
            return None;
        }
        let dim = weight_rows[0].len();
        for r in &weight_rows {
            if r.len() != dim {
                return None;
            }
        }

                let dense = FlashDense::from_rows(&weight_rows, bias.clone())?;

        let dim_f = dim.max(1) as f32;
        let mut rows = Vec::with_capacity(weight_rows.len());
        for r in &weight_rows {
            let nucleus: f32 = r.iter().sum::<f32>() / dim_f;
            let mut sumsq = 0.0f32;
            let mut residual = Vec::with_capacity(dim);
            for &v in r {
                let e = v - nucleus;
                residual.push(e);
                sumsq += e * e;
            }
            let radius = sumsq.sqrt();
            let sig = simd::xor_signature(&residual);
            rows.push(QairoLivingAtom {
                life: AtomLife::new(LifeParams::default()),
                nucleus,
                radius,
                sig,
            });
        }

        // Инициализация курсора — в конструкторе
        let mut cursor = QairoCursor::new(rows.len());
        cursor.seed(rows.iter().map(|a| a.sig).collect());
                // Собрать плоский массив весов для арены
        let mut w_flat = Vec::with_capacity(weight_rows.len() * dim);
        for r in &weight_rows {
            w_flat.extend_from_slice(r);
        }
        let arena = FlashArena::from_dense(&w_flat, weight_rows.len(), dim, bias.clone());
        let region = RegionManager::new(16 * 1024 * 1024); // бюджет 16 МБ на слой

        Some(Self { dense, rows, dim, compressed: None, cursor, arena, region })
    }

    // ─────────────────────────────────────────────────────────
    // Курсор / холодный старт
    // ─────────────────────────────────────────────────────────

    /// Явный холодный старт: прогреть веса до первого forward.
    pub fn cold_start(&mut self) -> u64 {
        if self.cursor.needs_cold_start() {
            self.cursor.warm(self.dense.data.as_slice())
        } else {
            0
        }
    }

    fn ensure_warm(&mut self) {
        if self.cursor.needs_cold_start() {
            self.cursor.warm(self.dense.data.as_slice());
        }
        self.cursor.touch();
    }

    // ─────────────────────────────────────────────────────────
    // Forward
    // ─────────────────────────────────────────────────────────

    /// Сырой сигнал БЕЗ bias (для living-режима).
    fn raw_signal(&self, x: &[f32]) -> Vec<f32> {
        match &self.compressed {
            Some(c) => {
                let mut r = c.gemv(x);
                for i in 0..r.len() {
                    r[i] -= self.dense.bias[i];
                }
                r
            }
            None => self.dense.gemv(x),
        }
    }

    /// Чистый аффинный forward: y = W x + b. (&self — не греет кэш)
    pub fn forward_linear(&self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.dim {
            return Vec::new();
        }
        match &self.compressed {
            Some(c) => c.gemv(x),
            None => self.dense.gemv_bias(x),
        }
    }

    /// Живой forward: y = b + gate * (W x). Сам следит за прогревом.
    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.dim {
            return Vec::new();
        }
        self.ensure_warm(); // ← только здесь (&mut self)

        let raw = self.raw_signal(x);
        let sum_x: f32 = x.iter().sum();

        let mut y = vec![0.0f32; self.dense.rows];
        for i in 0..self.dense.rows {
            let a = &mut self.rows[i];
            let dir_dot = if a.radius > 1e-12 {
                ((raw[i] - a.nucleus * sum_x) / a.radius).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            a.life.observe(a.sig, raw[i].abs(), dir_dot);
            y[i] = self.dense.bias[i] + a.gate() * raw[i];
        }
        y
    }

    /// Детерминированный живой forward без обновления Life.
    pub fn forward_gated_frozen(&self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.dim {
            return Vec::new();
        }
        let raw = self.raw_signal(x);
        let mut y = vec![0.0f32; self.dense.rows];
        for i in 0..self.dense.rows {
            y[i] = self.dense.bias[i] + self.rows[i].gate() * raw[i];
        }
        y
    }

    // ─────────────────────────────────────────────────────────
    // Backward / обучение
    // ─────────────────────────────────────────────────────────

    pub fn backward_input(&self, grad_out: &[f32]) -> Vec<f32> {
        self.dense.gemv_t(grad_out)
    }

    pub fn distribute_grad(&mut self, grad: &[f32], input_norm: f32) {
        if grad.len() != self.dense.rows {
            return;
        }
        for i in 0..self.dense.rows {
            let e = grad[i].abs() * input_norm;
            let sig = self.rows[i].sig;
            self.rows[i].life.observe(sig, e, 0.0);
            self.arena.note_energy(i, e);
        }
    }

    pub fn sgd_step(&mut self, grad_rows: &[f32], grad_bias: &[f32], lr: f32) {
        let cols = self.dense.cols;
        let rows = self.dense.rows;
        if grad_rows.len() < rows * cols {
            return;
        }
        let dim_f = cols.max(1) as f32;
        let stride = self.dense.stride;
        let d = self.dense.data.as_mut_slice();

        for i in 0..rows {
            let o = i * stride;
            let g = &grad_rows[i * cols..(i + 1) * cols];
            simd::axpy(&mut d[o..o + cols], -lr, g);

            if i < grad_bias.len() {
                self.dense.bias[i] -= lr * grad_bias[i];
            }

            let row = &d[o..o + cols];
            let nucleus: f32 = row.iter().sum::<f32>() / dim_f;
            let mut sumsq = 0.0f32;
            let mut residual = Vec::with_capacity(cols);
            for &v in row {
                let e = v - nucleus;
                residual.push(e);
                sumsq += e * e;
            }
            let radius = sumsq.sqrt();
            let sig = simd::xor_signature(&residual);

            let a = &mut self.rows[i];
            a.nucleus = nucleus;
            a.radius = radius;
            a.sig = sig;
        }

        // Веса изменились → int4-снимок устарел
        self.compressed = None;
    }

    // ─────────────────────────────────────────────────────────
    // int4-снимок
    // ─────────────────────────────────────────────────────────

    pub fn is_int4(&self) -> bool {
        self.compressed.is_some()
    }

    pub fn quantize_to_int4(&mut self) {
        if self.compressed.is_some() {
            return;
        }
        let rows = self.dense.rows;
        let cols = self.dense.cols;
        let mut w = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            w.extend_from_slice(self.dense.row(i));
        }
        let bias = self.dense.bias.clone();
        let mut c = FlashCompressed::from_dense(&w, rows, cols, bias);
        c.xor_cold_start();
        self.compressed = Some(c);
    }

    pub fn drop_int4(&mut self) {
        self.compressed = None;
    }

    pub fn memory_bytes(&self) -> usize {
        self.dense.bytes()
            + self.compressed.as_ref().map(|c| c.memory_bytes()).unwrap_or(0)
    }

    pub fn state_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mut dead = 0;
        let mut cold = 0;
        let mut born = 0;
        let mut alive = 0;
        let mut stable = 0;
        let mut decaying = 0;
        for r in &self.rows {
            match r.life.state {
                LifeState::Dead => dead += 1,
                LifeState::Cold => cold += 1,
                LifeState::Born => born += 1,
                LifeState::Alive => alive += 1,
                LifeState::Stable => stable += 1,
                LifeState::Decaying => decaying += 1,
            }
        }
        (dead, cold, born, alive, stable, decaying)
    }
        /// Вызывать между эпохами: освободить холодные чанки, остудить курсор.
    pub fn epoch_tick(&mut self) {
        self.arena.release_cold();              // точное освобождение мёртвых чанков
        self.region.rebalance(&mut self.arena); // бюджетная защита
        self.cursor.tick();
    }
}
