// src/atom_math/xor_cursor.rs
//! XorCursor: курсор состояния блока.
//!
//! Задача:
//! 1. Отслеживать XOR-подпись блока.
//! 2. Определять cold start.
//! 3. Определять, можно ли сжимать блок сильнее.
//! 4. Определять, нужно ли вернуть более высокую точность.
//!
//! Это управляющий контур для AtomCodec.

use crate::atom::math::simd;

#[derive(Clone)]
pub struct XorCursor {
    pub sig: u64,
    pub prev_sig: u64,
    pub alive_bits: u32,
    pub hamming: u32,
    pub steps_alive: u32,
    pub compression_level: u8,
    pub initialized: bool,
}

impl XorCursor {
    pub fn new() -> Self {
        Self {
            sig: 0,
            prev_sig: 0,
            alive_bits: 0,
            hamming: 0,
            steps_alive: 0,
            compression_level: 0,
            initialized: false,
        }
    }

    /// Обновить курсор по реальному блоку данных.
    pub fn update(&mut self, block: &[f32]) {
        let sig = simd::xor_signature(block);
        self.update_from_signature(sig);
    }

    /// Обновить курсор только по уже известной подписи.
    /// Полезно, если подпись уже была вычислена в другом модуле.
    pub fn update_from_signature(&mut self, sig: u64) {
        self.alive_bits = sig.count_ones();

        if !self.initialized {
            self.sig = sig;
            self.prev_sig = sig;
            self.hamming = 0;
            self.initialized = true;

            if self.is_alive() {
                self.steps_alive = 1;
            } else {
                self.steps_alive = 0;
            }

            return;
        }

        self.prev_sig = self.sig;
        self.sig = sig;
        self.hamming = (self.sig ^ self.prev_sig).count_ones();

        if self.is_alive() {
            self.steps_alive += 1;
        } else {
            self.steps_alive = 0;
        }
    }

    /// Холодный блок: подпись почти пустая.
    pub fn is_cold(&self) -> bool {
        self.alive_bits <= 10
    }

    /// Живой блок: достаточно бит в XOR-подписи.
    pub fn is_alive(&self) -> bool {
        self.alive_bits > 10
    }

    /// Стабильный блок: живой и почти не меняется.
    pub fn is_stable(&self) -> bool {
        self.alive_bits > 20 && self.hamming <= 2
    }

    /// Блок меняется слишком быстро.
    pub fn is_changing_fast(&self) -> bool {
        self.initialized && self.hamming > 8
    }

    /// Нужна более высокая точность:
    /// холодный блок или быстрые изменения.
    pub fn needs_more_precision(&self) -> bool {
        self.is_cold() || self.is_changing_fast()
    }

    /// Можно сжимать сильнее:
    /// блок долго живой и стабильный.
    pub fn can_compress_more(&self) -> bool {
        self.steps_alive >= 8 && self.is_stable()
    }

    /// Автоматически адаптировать уровень сжатия.
    ///
    /// Уровень:
    /// 0 = максимальная точность
    /// 1 = базовое сжатие
    /// 2 = агрессивное сжатие
    /// 3 = экстремальное сжатие
    pub fn adapt_compression(&mut self, max_level: u8) {
        if self.needs_more_precision() {
            if self.compression_level > 0 {
                self.compression_level -= 1;
            }
        } else if self.can_compress_more() {
            if self.compression_level < max_level {
                self.compression_level += 1;
            }
        }
    }

    /// Рекомендованные биты для AtomCodec.
    ///
    /// Формат:
    /// [биты направления, остаточный уровень 1, остаточный уровень 2, ...]
    pub fn suggested_bits(&self) -> Vec<u8> {
        if self.is_cold() {
            // Холодный блок: только точное направление, без остатков.
            vec![8]
        } else if self.needs_more_precision() {
            // Нестабильный блок: точность важнее памяти.
            vec![8, 4]
        } else if self.can_compress_more() {
            // Стабильный блок: можно жать сильнее.
            vec![8, 4, 2]
        } else {
            // Обычный рабочий блок.
            vec![8, 4]
        }
    }
}

impl Default for XorCursor {
    fn default() -> Self {
        Self::new()
    }
}
