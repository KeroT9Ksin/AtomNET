// src/atom/life/qairo_cursor.rs
//! QairoCursor: курсор состояния слоя Qairo на XOR-подписях и «тепле» строк.
//! Управляет FLASH: холодный старт, выбор ядра, prefetch, роутинг строк.

pub const CURSOR_HOT_BITS: u32 = 24;  // popcount выше — строка «живая»
pub const CURSOR_COLD_BITS: u32 = 8;  // ниже — «холодная»

pub struct QairoCursor {
    pub row_sigs: Vec<u64>,   // XOR-подписи строк
    pub row_heat: Vec<u32>,   // накопленное «тепло» (активность)
    pub layer_sig: u64,       // агрегированная подпись слоя
    pub step: u64,            // логическое время
    pub last_touch: u64,      // последний шаг доступа
    pub cold_start_done: bool,
}

impl QairoCursor {
    pub fn new(rows: usize) -> Self {
        Self {
            row_sigs: vec![0; rows],
            row_heat: vec![0; rows],
            layer_sig: 0,
            step: 0,
            last_touch: 0,
            cold_start_done: false,
        }
    }

    /// Зарегистрировать подписи строк (один раз при создании слоя).
    pub fn seed(&mut self, sigs: Vec<u64>) {
        self.layer_sig = sigs.iter().fold(0u64, |a, &s| a ^ s);
        self.row_sigs = sigs;
    }

    /// Отметить активность строки (энергия/когерентность выше порога).
    #[inline]
    pub fn observe(&mut self, i: usize, active: bool) {
        if active {
            self.row_heat[i] = self.row_heat[i].saturating_add(1);
        } else if self.row_heat[i] > 0 {
            self.row_heat[i] -= 1;
        }
    }

    /// Шаг времени: экспоненциальное затухание тепла.
    pub fn tick(&mut self) {
        self.step += 1;
        for h in self.row_heat.iter_mut() {
            *h >>= 1;
        }
    }

    /// Нужен ли холодный старт (слой создан или давно не трогали).
    pub fn needs_cold_start(&self) -> bool {
        !self.cold_start_done || (self.step - self.last_touch) > 64
    }

    /// XOR-прогрев весов: стримит данные в кэш/TLB до первого GEMV.
    /// data.len() — число f32; слов u64 = len/2.
    pub fn warm(&mut self, data: &[f32]) -> u64 {
        let mut acc: u64 = 0;
        let words = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u64, data.len() / 2)
        };
        for &w in words {
            acc ^= w;
        }
        std::hint::black_box(acc);
        self.cold_start_done = true;
        self.last_touch = self.step;
        acc
    }

    pub fn touch(&mut self) {
        self.last_touch = self.step;
    }

    #[inline]
    pub fn is_hot(&self, i: usize) -> bool {
        self.row_sigs[i].count_ones() > CURSOR_HOT_BITS && self.row_heat[i] > 0
    }

    #[inline]
    pub fn is_cold_row(&self, i: usize) -> bool {
        self.row_sigs[i].count_ones() < CURSOR_COLD_BITS
    }

    /// Доля горячих строк.
    pub fn hot_fraction(&self) -> f32 {
        if self.row_sigs.is_empty() {
            return 0.0;
        }
        let hot = self
            .row_sigs
            .iter()
            .filter(|&&s| s.count_ones() > CURSOR_HOT_BITS)
            .count();
        hot as f32 / self.row_sigs.len() as f32
    }

    /// Рекомендация ядра: 0=f32, 1=int4, 2=int4+skip-cold.
    pub fn recommend_kernel(&self) -> u8 {
        let hf = self.hot_fraction();
        if hf > 0.7 { 0 } else if hf > 0.3 { 1 } else { 2 }
    }

    /// Дистанция prefetch: чем холоднее слой, тем глубже греем.
    pub fn prefetch_distance(&self) -> usize {
        let hf = self.hot_fraction();
        if hf > 0.7 { 4 } else if hf > 0.3 { 8 } else { 12 }
    }
}