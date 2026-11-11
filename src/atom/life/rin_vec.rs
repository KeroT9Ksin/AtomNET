// src/atom_math/rin_vec.rs
//! RinVec: вектороподобный массив с математикой Rineta.
//!
//! Это НЕ оптимизатор.
//! Он не делает шаг обучения.
//!
//! Он хранит:
//! - данные;
//! - жизнь;
//! - энергию;
//! - когерентность;
//! - XOR-подпись.
use crate::atom::life::life::{AtomLife, LifeParams};
use crate::atom::math::{
    simd,
};

pub struct RinVec {
    pub data: Vec<f32>,
    pub life: AtomLife,
}

impl RinVec {
    pub fn new(data: Vec<f32>) -> Self {
        let mut v = Self {
            data,
            life: AtomLife::new(LifeParams::default()),
        };

        v.update_life_with_coherence(0.0);
        v
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Обновить жизнь по текущему вектору.
    pub fn update_life_with_coherence(&mut self, coherence: f32) {
        let sig = simd::xor_signature(&self.data);
        let energy = simd::norm(&self.data);
        self.life.observe(sig, energy, coherence);
    }

    pub fn norm(&self) -> f32 {
        simd::norm(&self.data)
    }

    pub fn dot(&self, other: &Self) -> f32 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }

        simd::dot(&self.data, &other.data)
    }

    /// Направленная когерентность между двумя векторами:
    ///
    ///     C = dot(x, y) / (||x|| * ||y|| + eps)
    pub fn coherence_with(&mut self, other: &Self) -> f32 {
        if self.data.len() != other.data.len() {
            self.update_life_with_coherence(0.0);
            return 0.0;
        }

        let dot = simd::dot(&self.data, &other.data);
        let na = simd::norm(&self.data);
        let nb = simd::norm(&other.data);

        let c = (dot / (na * nb + 1e-8)).clamp(-1.0, 1.0);

        self.update_life_with_coherence(c);

        c
    }

    /// Нормализация вектора.
    /// Возвращает норму до нормализации.
    pub fn normalize(&mut self) -> f32 {
        let n = simd::norm(&self.data);

        if n > 1e-12 {
            let inv = 1.0 / n;

            for v in self.data.iter_mut() {
                *v *= inv;
            }
        }

        self.update_life_with_coherence(0.0);

        n
    }

    /// self += alpha * other
    pub fn add_scaled(&mut self, other: &Self, alpha: f32) -> bool {
        if self.data.len() != other.data.len() {
            return false;
        }

        for i in 0..self.data.len() {
            self.data[i] += alpha * other.data[i];
        }

        self.update_life_with_coherence(0.0);

        true
    }

    /// Dot с учётом lifetime-гейтов обоих векторов.
    ///
    /// Это уже не чистый dot, а живой математический dot.
    pub fn gated_dot(&self, other: &Self) -> f32 {
        if self.data.len() != other.data.len() {
            return 0.0;
        }

        let raw = simd::dot(&self.data, &other.data);

        raw * self.life.gate() * other.life.gate()
    }

    pub fn gate(&self) -> f32 {
        self.life.gate()
    }

    pub fn is_alive(&self) -> bool {
        self.life.is_active()
    }

    pub fn is_dead(&self) -> bool {
        self.life.is_dead()
    }
}
