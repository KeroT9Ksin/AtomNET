// src/atom/flash/flash_arena.rs
//! FlashArena v2: точный f32 для активных чанков, освобождение мёртвых в int4.
//! RegionManager: бюджетный губернатор (IRON) поверх арены.

use crate::atom::flash::flash::Aligned64;
use crate::atom::flash::flash_compressed::FlashCompressed;
use crate::atom::flash::xor_alloc::XorAlloc;

pub const CHUNK_ROWS: usize = 256;
pub const RELEASE_HEAT: u32 = 2;

pub struct FlashArena {
    pub rows: usize,
    pub cols: usize,
    pub int4: FlashCompressed,
    pub hot: Vec<Option<Aligned64<f32>>>,
    pub heat: Vec<u32>,
    pub used_extra: usize,
}

impl FlashArena {
    pub fn from_dense(w: &[f32], rows: usize, cols: usize, bias: Vec<f32>) -> Self {
        let int4 = FlashCompressed::from_dense(w, rows, cols, bias);
        let nch = (rows + CHUNK_ROWS - 1) / CHUNK_ROWS;

        let mut hot = Vec::with_capacity(nch);
        let mut used = 0usize;
        for c in 0..nch {
            let r0 = c * CHUNK_ROWS;
            let r1 = (r0 + CHUNK_ROWS).min(rows);
            let n = (r1 - r0) * cols;
            let mut ov = Aligned64::<f32>::new_zeroed(n);
            ov.as_mut_slice().copy_from_slice(&w[r0 * cols..r1 * cols]);
            used += n * 4;
            hot.push(Some(ov));
        }

        Self { rows, cols, int4, hot, heat: vec![0; nch], used_extra: used }
    }

    #[inline]
    fn chunk_of(&self, row: usize) -> usize { row / CHUNK_ROWS }

    pub fn is_hot(&self, row: usize) -> bool { self.hot[self.chunk_of(row)].is_some() }

    pub fn note_energy(&mut self, row: usize, energy: f32) {
        if energy <= 0.0 {
            return;
        }
        let c = self.chunk_of(row);
        self.heat[c] = self.heat[c].saturating_add(1);
    }

    pub fn release_cold(&mut self) {
        for c in 0..self.hot.len() {
            if self.heat[c] < RELEASE_HEAT {
                self.drop_overlay(c);
            }
            self.heat[c] >>= 1;
        }
    }

    pub fn drop_overlay(&mut self, c: usize) {
        if let Some(ov) = self.hot[c].take() {
            self.used_extra -= ov.as_slice().len() * 4;
        }
    }

    pub fn promote(&mut self, c: usize) {
        if self.hot[c].is_some() { return; }
        let r0 = c * CHUNK_ROWS;
        let r1 = (r0 + CHUNK_ROWS).min(self.rows);
        let n = (r1 - r0) * self.cols;
        let mut ov = Aligned64::<f32>::new_zeroed(n);
        self.int4.dequant_rows(r0, r1, ov.as_mut_slice());
        self.hot[c] = Some(ov);
        self.used_extra += n * 4;
    }

    pub fn read_row(&self, row: usize) -> Vec<f32> {
        let c = self.chunk_of(row);
        if let Some(ov) = &self.hot[c] {
            let off = (row - c * CHUNK_ROWS) * self.cols;
            ov.as_slice()[off..off + self.cols].to_vec()
        } else {
            let mut out = vec![0.0f32; self.cols];
            self.int4.dequant_rows(row, row + 1, &mut out);
            out
        }
    }

    pub fn memory_bytes(&self) -> usize { self.int4.memory_bytes() + self.used_extra }
        /// Тир-план на битовой математике (читает подписи из int4-базы).
    pub fn xor_tiers(&self) -> Vec<u8> {
        XorAlloc::plan(&self.int4.row_sigs, &self.heat, CHUNK_ROWS)
    }

    /// XOR-управляемое освобождение: отпускаем только tier=0 (ice).
    pub fn release_cold_xor(&mut self) {
        let tiers = self.xor_tiers();
        for c in 0..self.hot.len() {
            if tiers[c] == 0 {
                self.drop_overlay(c);
            }
            self.heat[c] >>= 1;
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Region: бюджетный губернатор
// ═══════════════════════════════════════════════════════════

pub const REGION_CHUNKS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tier { Ice, Warm, Hot }

pub struct RegionManager {
    pub tiers: Vec<Tier>,
    pub ram_budget: usize,
}

impl RegionManager {
    pub fn new(budget: usize) -> Self {
        Self { tiers: Vec::new(), ram_budget: budget }
    }

    pub fn region_of(&self, chunk: usize) -> usize { chunk / REGION_CHUNKS }

    /// При превышении RAM вытесняет самый холодный живой чанк.
    pub fn rebalance(&mut self, arena: &mut FlashArena) {
        while arena.memory_bytes() > self.ram_budget {
            let mut best: Option<usize> = None;
            for c in 0..arena.hot.len() {
                if arena.hot[c].is_some() {
                    best = match best {
                        None => Some(c),
                        Some(b) => Some(if arena.heat[c] < arena.heat[b] { c } else { b }),
                    };
                }
            }
            match best {
                Some(c) => arena.drop_overlay(c),
                None => break,
            }
        }
    }
}
