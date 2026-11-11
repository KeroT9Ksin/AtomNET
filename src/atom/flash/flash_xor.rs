// src/atom/flash/flash_xor.rs
//! FlashXorDot — ЭКСПЕРИМЕНТ: бинарные веса (±1) + dot через
//! popcount(XOR) — настоящая XOR-математика в ядре.
//!
//! Память: 1 бит/вес (÷32 к f32). Скорость: 64 элемента за
//! popcnt-инструкцию. КАЧЕСТВО: знаки теряют величины — измеряем честно.

pub struct FlashXor {
    pub rows: usize,
    pub cols: usize,
    pub words: usize,
    /// Битовые плоскости: 1 => +1, 0 => -1.
    pub bits: Vec<u64>,
    /// Масштаб строки: s_i = mean|w_i| (восстанавливает величину частично).
    pub row_scale: Vec<f32>,
}

impl FlashXor {
    pub fn from_dense(w: &[f32], rows: usize, cols: usize) -> Self {
        let words = cols.div_ceil(64);
        let mut bits = vec![0u64; rows * words];
        let mut row_scale = vec![0.0f32; rows];
        for i in 0..rows {
            let row = &w[i * cols..(i + 1) * cols];
            let mut s = 0.0f32;
            for &v in row {
                s += v.abs();
            }
            row_scale[i] = s / cols.max(1) as f32;
            for wd in 0..words {
                let mut x: u64 = 0;
                for k in 0..64 {
                    let j = wd * 64 + k;
                    if j < cols && row[j] > 0.0 {
                        x |= 1u64 << k;
                    }
                }
                bits[i * words + wd] = x;
            }
        }
        Self {
            rows,
            cols,
            words,
            bits,
            row_scale,
        }
    }

    /// y_i = s_i · (M − 2·popcount(w_bits XOR x_bits)), x_bin = sign(x).
    /// Возвращает также «чистый» вариант без масштаба входа (s_x=1).
    pub fn gemv_binary(&self, x: &[f32]) -> Vec<f32> {
        let words = self.words;
        let mut xb = vec![0u64; words];
        let mut xsum_pos = 0i64;
        let mut x_scale = 0.0f32;
        for wd in 0..words {
            let mut v: u64 = 0;
            for k in 0..64 {
                let j = wd * 64 + k;
                if j < x.len() {
                    if x[j] > 0.0 {
                        v |= 1u64 << k;
                        xsum_pos += 1;
                    }
                    x_scale += x[j].abs();
                }
            }
            xb[wd] = v;
        }
        x_scale /= self.cols.max(1) as f32;
        let _ = xsum_pos;
        let mut y = vec![0.0f32; self.rows];
        for i in 0..self.rows {
            let base = i * words;
            let mut matches = 0i64;
            for wd in 0..words {
                matches += (64 - (self.bits[base + wd] ^ xb[wd]).count_ones() as i64);
            }
            // Σ b·b_x = matches − mismatches = matches − (cols − matches)
            let dot_b = 2 * matches - self.cols as i64;
            y[i] = self.row_scale[i] * x_scale * dot_b as f32;
        }
        y
    }


    /// «Стабилизатор канала»: быстрый XOR-проход даёт грубые ŷ;
    /// K строк с НАИБОЛЬШИМ |ŷ| доуточняются точным f32-произведением
    /// (эти строки доминируют в ошибке сигнала). Возвращает (y, refined).
    pub fn gemv_stabilized(&self, x: &[f32], w: &[f32], top_k: usize) -> (Vec<f32>, usize) {
        let rough = self.gemv_binary(x);
        if top_k == 0 || self.rows == 0 {
            return (rough, 0);
        }
        let mut order: Vec<usize> = (0..self.rows).collect();
        order.sort_by(|&a, &b| {
            rough[b]
                .abs()
                .total_cmp(&rough[a].abs())
        });
        let k = top_k.min(self.rows);
        let mut y = rough;
        for &i in order.iter().take(k) {
            let s: f32 = w[i * self.cols..(i + 1) * self.cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum();
            y[i] = s;
        }
        (y, k)
    }

    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * 8 + self.row_scale.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn xor_dot_speed_quality() {
        let rows = 4096usize;
        let cols = 2048usize;
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut nxt = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        };
        let w: Vec<f32> = (0..rows * cols).map(|_| nxt() * 0.02).collect();
        let x: Vec<f32> = (0..cols).map(|_| nxt() * 0.02).collect();

        let fx = FlashXor::from_dense(&w, rows, cols);
        let y = fx.gemv_binary(&x);

        // скорость
        let t0 = Instant::now();
        for _ in 0..20 {
            let _ = fx.gemv_binary(&x);
        }
        let per = t0.elapsed().as_secs_f64() / 20.0 * 1000.0;

        // качество против f32-эталона
        let mut max_err = 0.0f32;
        let mut ref_max = 0.0f32;
        for i in 0..rows {
            let s: f32 = w[i * cols..(i + 1) * cols]
                .iter()
                .zip(&x)
                .map(|(a, b)| a * b)
                .sum();
            ref_max = ref_max.max(s.abs());
            max_err = max_err.max((y[i] - s).abs());
        }
        eprintln!(
            "FlashXorDot: {rows}x{cols}, time={per:.3} ms, mem={:.2} MB, max_err={:.4} (ref_scale={:.4})",
            fx.memory_bytes() as f64 / 1e6,
            max_err,
            ref_max
        );
        eprintln!(
            "качество: знаковая проекция (бинарные веса) — относит. ошибка {:.0}%",
            max_err / ref_max.max(1e-9) * 100.0
        );
        // скорость сравнима с Q16 (~0.9 ms): печатаем и скорость, и цену качества
        assert!(y.len() == rows);
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(fx.memory_bytes() < 4 * 1024 * 1024, "память должна быть мизерной");

        // Стабилизатор канала: доуточнение top-K строк
        for frac in [0usize, 16, 64, 256, 1024, 4096] {
            let (ys, refined) = fx.gemv_stabilized(&x, &w, frac);
            let mut max_err = 0.0f32;
            for i in 0..rows {
                let s: f32 = w[i * cols..(i + 1) * cols]
                    .iter()
                    .zip(&x)
                    .map(|(a, b)| a * b)
                    .sum();
                max_err = max_err.max((ys[i] - s).abs());
            }
            eprintln!(
                "stabilizer K={frac:4}: refined={refined:4}, max_err={max_err:.5} (ref={ref_max:.4})"
            );
        }
    }
}
