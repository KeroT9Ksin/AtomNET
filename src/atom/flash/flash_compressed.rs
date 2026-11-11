// src/atom/flash/flash_compressed.rs
//! FLASH-COMPRESSED: int4 Q4xQ8 GEMV + int8-остаток (f32-точность), AVX2.

use crate::atom::math::simd;
use rayon::prelude::*;
use std::cell::RefCell;

pub const BLOCK: usize = 256;

thread_local! {
    static QX_BUF: RefCell<Vec<i8>> = RefCell::new(Vec::new());
}

pub struct FlashCompressed {
    pub rows: usize,
    pub cols: usize,
    pub pcols: usize,
    pub weights_i4: Vec<u8>,
    pub resid: Vec<u8>,
    pub scales: Vec<f32>, pub zeros: Vec<f32>, pub code_sums: Vec<i32>,
    pub rs: Vec<f32>, pub rz: Vec<f32>, pub rsum: Vec<i32>,
    pub bias: Vec<f32>,
    pub row_sigs: Vec<u64>,
    pub cold_start_done: bool,
}

impl FlashCompressed {
    pub fn from_dense(w: &[f32], rows: usize, cols: usize, bias: Vec<f32>) -> Self {
        assert_eq!(w.len(), rows * cols);
        let pcols = ((cols + BLOCK - 1) / BLOCK) * BLOCK;
        let blocks = pcols / BLOCK;
        let bpr = pcols / 2;

        let mut weights = vec![0u8; rows * bpr];
        let mut resid = vec![0u8; rows * pcols];
        let mut scales = Vec::with_capacity(rows * blocks);
        let mut zeros = Vec::with_capacity(rows * blocks);
        let mut code_sums = Vec::with_capacity(rows * blocks);
        let mut rs = Vec::with_capacity(rows * blocks);
        let mut rz = Vec::with_capacity(rows * blocks);
        let mut rsum = Vec::with_capacity(rows * blocks);
        let mut row_sigs = Vec::with_capacity(rows);
        let mut codes = vec![0i8; BLOCK];

        for i in 0..rows {
            let row = &w[i * cols..(i + 1) * cols];
            row_sigs.push(simd::xor_signature(row));
            let mut prow = Vec::with_capacity(pcols);
            prow.extend_from_slice(row);
            prow.resize(pcols, 0.0);

            for b in 0..blocks {
                let chunk = &prow[b * BLOCK..(b + 1) * BLOCK];
                let mean: f32 = chunk.iter().sum::<f32>() / BLOCK as f32;
                let m = chunk.iter().map(|&x| (x - mean).abs()).fold(0.0f32, f32::max);
                let sw = (m / 7.0).max(1e-12);
                let inv = 1.0 / sw;
                let mut sc8 = 0i32;
                for (j, &x) in chunk.iter().enumerate() {
                    let v = ((x - mean) * inv).round().clamp(-8.0, 7.0) as i8;
                    codes[j] = v; sc8 += v as i32;
                }
                let base = i * bpr + b * (BLOCK / 2);
                for k in 0..(BLOCK / 2) {
                    weights[base + k] = ((codes[2 * k] + 8) as u8)
                        | (((codes[2 * k + 1] + 8) as u8) << 4);
                }

                // Остаток int8 (f32-точность)
                let rbase = i * pcols + b * BLOCK;
                let mut rtmp = [0.0f32; BLOCK];
                for j in 0..BLOCK { rtmp[j] = chunk[j] - (codes[j] as f32 * sw + mean); }
                let rmean: f32 = rtmp.iter().sum::<f32>() / BLOCK as f32;
                let rM = rtmp.iter().map(|&v| (v - rmean).abs()).fold(0.0f32, f32::max);
                let rsc = (rM / 127.0).max(1e-12);
                let mut s8 = 0i32;
                for j in 0..BLOCK {
                    let c8 = ((rtmp[j] - rmean) / rsc).round().clamp(-127.0, 127.0) as i8;
                    resid[rbase + j] = (c8 as i32 + 128) as u8;
                    s8 += c8 as i32;
                }

                scales.push(sw); zeros.push(mean); code_sums.push(sc8);
                rs.push(rsc); rz.push(rmean); rsum.push(s8);
            }
        }

        Self {
            rows, cols, pcols, weights_i4: weights, resid,
            scales, zeros, code_sums, rs, rz, rsum,
            bias, row_sigs, cold_start_done: false,
        }
    }

    pub fn xor_cold_start(&mut self) {
        if self.cold_start_done { return; }
        let mut acc = 0u64;
        for &b in &self.weights_i4 { acc ^= b as u64; }
        std::hint::black_box(acc);
        self.cold_start_done = true;
    }

    pub fn row_liveness(&self, i: usize) -> u32 { self.row_sigs[i].count_ones() }
    pub fn hot_rows(&self) -> Vec<usize> {
        self.row_sigs.iter().enumerate().filter(|(_, &s)| s.count_ones() > 32).map(|(i, _)| i).collect()
    }
    pub fn dead_rows(&self) -> Vec<usize> {
        self.row_sigs.iter().enumerate().filter(|(_, &s)| s.count_ones() < 8).map(|(i, _)| i).collect()
    }
    pub fn row_similarity(&self, i: usize, j: usize) -> u32 {
        64 - (self.row_sigs[i] ^ self.row_sigs[j]).count_ones()
    }

    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let blocks = self.pcols / BLOCK;
        let mut xp = vec![0.0f32; self.pcols];
        xp[..self.cols].copy_from_slice(x);

        QX_BUF.with(|buf| {
            let mut qx = buf.borrow_mut();
            qx.resize(self.pcols, 0i8);
            let mut sx = vec![0.0f32; blocks];
            let mut mx = vec![0.0f32; blocks];
            let mut sq = vec![0i32; blocks];
            for b in 0..blocks {
                let xb = &xp[b * BLOCK..(b + 1) * BLOCK];
                let qo = &mut qx[b * BLOCK..(b + 1) * BLOCK];
                let mean: f32 = xb.iter().sum::<f32>() / BLOCK as f32;
                let m = xb.iter().map(|&v| (v - mean).abs()).fold(0.0f32, f32::max);
                let s = (m / 127.0).max(1e-12);
                let inv = 1.0 / s;
                let mut ssum = 0i32;
                for (qv, &xv) in qo.iter_mut().zip(xb.iter()) {
                    let q = ((xv - mean) * inv).round().clamp(-127.0, 127.0) as i8;
                    *qv = q; ssum += q as i32;
                }
                sx[b] = s; mx[b] = mean; sq[b] = ssum;
            }

            let mut y = vec![0.0f32; self.rows];
            #[cfg(target_arch = "x86_64")]
            {
                if simd::has_avx2() {
                    if let Some(jq) = crate::atom::flash::jit::jit_q4_gemv() {
                        // Q4×Q8 JIT: int4 + int8-остаток, всё в сгенерированном коде.
                        let ctx = crate::atom::flash::jit::Q4Ctx {
                            weights_i4: self.weights_i4.as_ptr() as *const u8,
                            resid: self.resid.as_ptr() as *const u8,
                            qx: qx.as_ptr() as *const i8,
                            scales: self.scales.as_ptr() as *const f32,
                            zeros: self.zeros.as_ptr() as *const f32,
                            code_sums: self.code_sums.as_ptr() as *const i32,
                            rs: self.rs.as_ptr() as *const f32,
                            rz: self.rz.as_ptr() as *const f32,
                            rsum: self.rsum.as_ptr() as *const i32,
                            sx: sx.as_ptr() as *const f32,
                            mx: mx.as_ptr() as *const f32,
                            sq: sq.as_ptr() as *const i32,
                            bias: self.bias.as_ptr() as *const f32,
                            y: y.as_mut_ptr() as *mut f32,
                            blocks: (self.pcols / BLOCK) as u64,
                            bpr: (self.pcols / 2) as u64,
                            rpr: self.pcols as u64,
                            ycell: (self.cols * 4) as u64,
                            rows: self.rows as u64,
                        };
                        unsafe { (jq.gemv)(&ctx) };
                        return y;
                    }
                    unsafe { self.kernel_fused_avx2(&qx, &sx, &mx, &sq, &mut y) };
                }
                else { self.kernel_scalar(&qx, &sx, &mx, &sq, &mut y); self.resid_scalar(&qx, &sx, &mx, &sq, &mut y); }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                self.kernel_scalar(&qx, &sx, &mx, &sq, &mut y);
                self.resid_scalar(&qx, &sx, &mx, &sq, &mut y);
            }
            y
        })
    }

    /// Слитый проход: int4-ядра + int8-остаток за один обход строк.
    /// Оба потока (веса ÷2 и остаток) prefetch-ятся одновременно —
    /// память читается последовательно, как по «каналам».
    #[cfg(target_arch = "x86_64")]
    unsafe fn kernel_fused_avx2(&self, qx: &[i8], sx: &[f32], mx: &[f32], sq: &[i32], y: &mut [f32]) {
        use std::arch::x86_64::*;
        let blocks = self.pcols / BLOCK;
        let bpr = self.pcols / 2;
        let rpr = self.pcols;
        let mask8 = _mm_set1_epi8(0x0F);
        let ones16 = _mm256_set1_epi16(1);
        let wp = self.weights_i4.as_ptr();
        let rp = self.resid.as_ptr();
        let qp = qx.as_ptr();
        for i in 0..self.rows {
            if i + 4 < self.rows {
                _mm_prefetch(wp.add((i + 4) * bpr) as *const i8, _MM_HINT_T0);
                _mm_prefetch(rp.add((i + 4) * rpr) as *const i8, _MM_HINT_T0);
            }
            let w0 = wp.add(i * bpr);
            let r0 = rp.add(i * rpr);
            let mut acc = 0.0f32;

            // ── int4 Q4×Q8 ──
            for b in 0..blocks {
                let bp = w0.add(b * (BLOCK / 2));
                let q = qp.add(b * BLOCK);
                let mut a0 = _mm256_setzero_si256();
                for c in 0..(BLOCK / 32) {
                    let vq = _mm256_loadu_si256(q.add(c * 32) as *const __m256i);
                    let wv = _mm_loadu_si128(bp.add(c * 16) as *const __m128i);
                    let lo = _mm_and_si128(wv, mask8);
                    let hi = _mm_and_si128(_mm_srli_epi16(wv, 4), mask8);
                    let vc = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_unpacklo_epi8(lo, hi)),
                        _mm_unpackhi_epi8(lo, hi), 1);
                    a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(vc, vq));
                }
                let m0 = _mm256_madd_epi16(a0, ones16);
                let mut t = [0i32; 8];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, m0);
                let d0: i32 = t.iter().sum();
                let base = i * blocks + b;
                let s = sx[b]; let m = mx[b]; let qsum = sq[b] as f32;
                acc += self.scales[base] * s * (d0 as f32 - 8.0 * qsum)
                    + self.scales[base] * m * self.code_sums[base] as f32
                    + self.zeros[base] * s * qsum
                    + self.zeros[base] * m * BLOCK as f32;
            }

            // ── int8-остаток (f32-точность) ──
            for b in 0..blocks {
                let bp = r0.add(b * BLOCK);
                let q = qp.add(b * BLOCK);
                let mut a32 = _mm256_setzero_si256();
                for c in 0..(BLOCK / 32) {
                    let vc8 = _mm256_loadu_si256(bp.add(c * 32) as *const __m256i);
                    let vq = _mm256_loadu_si256(q.add(c * 32) as *const __m256i);
                    let p16 = _mm256_maddubs_epi16(vc8, vq);
                    a32 = _mm256_add_epi32(a32, _mm256_madd_epi16(p16, ones16));
                }
                let mut t = [0i32; 8];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a32);
                let d8stored: i32 = t.iter().sum();
                let dot8 = d8stored - 128 * sq[b];
                let base = i * blocks + b;
                let s = sx[b]; let m = mx[b]; let qsum = sq[b] as f32;
                acc += self.rs[base] * (s * dot8 as f32 + m * self.rsum[base] as f32)
                    + self.rz[base] * (s * qsum + m * BLOCK as f32);
            }

            y[i] = self.bias[i] + acc;
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn kernel_scalar(&self, qx: &[i8], sx: &[f32], mx: &[f32], sq: &[i32], y: &mut [f32]) {
        // int4 Q4×Q8 поток scalar fallback (для не-x86_64 и scalar-третьих):
        // та же формула, что в kernel_fused_avx2, покинутая по cleanliness.
        let blocks = self.pcols / BLOCK;
        let bpr = self.pcols / 2;
        for i in 0..self.rows {
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let base = i * blocks + b;
                let bp = i * bpr + b * (BLOCK / 2);
                let q0 = b * BLOCK;
                let mut d0: i32 = 0;
                for c in 0..BLOCK / 32 {
                    let wv = &self.weights_i4[bp + c * 16..bp + c * 16 + 16];
                    let vq = &qx[q0 + c * 32..q0 + c * 32 + 32];
                    for j in 0..32 {
                        let pj = j / 2;
                        let rem = j % 2;
                        let byte = wv[pj];
                        let code = if rem == 0 { byte & 0x0F } else { byte >> 4 } as i8 - 8;
                        d0 += code as i32 * vq[j] as i32;
                    }
                }
                let s = sx[b]; let m = mx[b]; let qsum = sq[b] as f32;
                acc += self.scales[base] * s * (d0 as f32 - 8.0 * qsum)
                    + self.scales[base] * m * self.code_sums[base] as f32
                    + self.zeros[base] * s * qsum
                    + self.zeros[base] * m * BLOCK as f32;
            }
            y[i] = self.bias[i] + acc;
        }
    }

    fn resid_scalar(&self, qx: &[i8], sx: &[f32], mx: &[f32], sq: &[i32], y: &mut [f32]) {
        // int8-остативом мягкий проход (добавляет свой вклад к y[i]).
        let blocks = self.pcols / BLOCK;
        for i in 0..self.rows {
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let base = i * blocks + b;
                let bp = i * self.pcols + b * BLOCK;
                let q0 = b * BLOCK;
                // maddubs пари: saturating i16 (как _mm256_maddubs_epi16 в fused AVX2);
                // scalar-сумма без сатурации дала бы ±0.67% расхождения.
                let mut d8: i32 = 0;
                for k in 0..(BLOCK / 2) {
                    let r8 = self.resid[bp + 2 * k] as i32;
                    let r8n = self.resid[bp + 2 * k + 1] as i32;
                    let pair = (r8 * qx[q0 + 2 * k] as i32 + r8n * qx[q0 + 2 * k + 1] as i32)
                        .clamp(-32768, 32767);
                    d8 += pair;
                }
                let dot8 = d8 - 128 * sq[b];
                let s = sx[b]; let m = mx[b]; let qsum = sq[b] as f32;
                acc += self.rs[base] * (s * dot8 as f32 + m * self.rsum[base] as f32)
                    + self.rz[base] * (s * qsum + m * BLOCK as f32);
            }
            y[i] += acc;
        }
    }



    pub fn dequant_rows(&self, r0: usize, r1: usize, out: &mut [f32]) {
        let blocks = self.pcols / BLOCK;
        let bpr = self.pcols / 2;
        let rpr = self.pcols;
        for i in r0..r1 {
            let row = &self.weights_i4[i * bpr..(i + 1) * bpr];
            let rrow = &self.resid[i * rpr..(i + 1) * rpr];
            for b in 0..blocks {
                let base = i * blocks + b;
                let s = self.scales[base]; let z = self.zeros[base];
                let rs_ = self.rs[base]; let rz_ = self.rz[base];
                let bo = b * (BLOCK / 2); let rbo = b * BLOCK;
                let upto = BLOCK.min(self.cols);
                for j in 0..upto {
                    let byte = row[bo + j / 2];
                    let c4 = if j % 2 == 0 { (byte & 0x0F) as i32 } else { (byte >> 4) as i32 };
                    let c8 = rrow[rbo + j] as i32 - 128;
                    out[(i - r0) * self.cols + j] =
                        (c4 - 8) as f32 * s + z + c8 as f32 * rs_ + rz_;
                }
            }
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.weights_i4.len() + self.resid.len()
            + (self.scales.len() + self.zeros.len()) * 4
            + self.code_sums.len() * 4
            + (self.rs.len() + self.rz.len()) * 4 + self.rsum.len() * 4
            + self.bias.len() * 4 + self.row_sigs.len() * 8
    }

    pub fn compression_ratio(&self) -> f32 {
        (self.rows * self.cols * 4) as f32 / self.memory_bytes().max(1) as f32
    }

    /// WinRAR-подобная solid-упаковка всего контекста одним потоком:
    /// заголовок слоя + сжатые блоки в файл. Порядок — один конкат-стрим
    /// (dictionary), который читатель может стримить без предварительно
    /// распакованного верха: страницы LRU читают только записи, которые
    /// влезли в бюджет памяти.
    pub fn archive_begin(out: &mut Vec<u8>) {
        out.extend_from_slice(b"ATQ4AR1\0");
        out.extend_from_slice(&1u32.to_le_bytes()); // version
    }

    /// Чтение одной строки контекста (де-квантование: mean + codes + resid).
    /// Для embedding-lookup: строка = вектор эмбеддинга (точные байты 1:1
    /// с dequant_row из model.rs).
    pub fn row_values(&self, row: usize) -> Vec<f32> {
        assert!(row < self.rows, "row_values OOB");
        let bpr = self.pcols / 2;
        let blocks = (self.pcols / BLOCK).max(1);
        let mut out = Vec::with_capacity(self.pcols);
        for b in 0..blocks {
            let scale = self.scales[row * blocks + b];
            let mean = self.zeros[row * blocks + b];
            let rs = self.rs[row * blocks + b];
            let rz = self.rz[row * blocks + b];
            let base = row * bpr + b * (BLOCK / 2);
            let rbase = row * self.pcols + b * BLOCK;
            for k in 0..(BLOCK / 2) {
                let lo = (self.weights_i4[base + k] & 0x0F) as i32 - 8;
                let hi = ((self.weights_i4[base + k] >> 4) & 0x0F) as i32 - 8;
                let r_lo = (self.resid[rbase + 2*k] as i32 - 128) as f32 * rs + rz;
                let r_hi = (self.resid[rbase + 2*k + 1] as i32 - 128) as f32 * rs + rz;
                out.push(lo as f32 * scale + mean + r_lo);
                out.push(hi as f32 * scale + mean + r_hi);
            }
        }
        out.truncate(self.cols);
        out
    }

    pub fn write_archive(&self, out: &mut Vec<u8>) {
        let mut payload: Vec<u8> = Vec::new();
        for v in &[self.rows, self.cols, self.pcols] {
            payload.extend_from_slice(&(*v as u32).to_le_bytes());
        }
        for chunk in [&self.weights_i4, &self.resid] {
            payload.extend_from_slice(&(chunk.len() as u64).to_le_bytes());
            payload.extend_from_slice(chunk);
        }
        for v in [&self.scales, &self.zeros, &self.rs, &self.rz] {
            payload.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v.iter() {
                payload.extend_from_slice(&x.to_le_bytes());
            }
        }
        for v in [&self.code_sums, &self.rsum] {
            payload.extend_from_slice(&(v.len() as u64).to_le_bytes());
            for x in v.iter() {
                payload.extend_from_slice(&x.to_le_bytes());
            }
        }
        payload.extend_from_slice(&(self.row_sigs.len() as u64).to_le_bytes());
        for x in self.row_sigs.iter() {
            payload.extend_from_slice(&x.to_le_bytes());
        }
        payload.extend_from_slice(&(self.bias.len() as u64).to_le_bytes());
        for x in self.bias.iter() {
            payload.extend_from_slice(&x.to_le_bytes());
        }
        payload.extend_from_slice(&(self.cold_start_done as u32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&payload);
    }

    pub fn read_archive(payload: &[u8]) -> Result<Self, String> {
        let mut pos = 0usize;
        let mut rd_u32 = |pos: &mut usize| -> Result<u32, String> {
            if *pos + 4 > payload.len() { return Err("archive: eof u32".into()); }
            let v = u32::from_le_bytes(payload[*pos..*pos+4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        };
        let mut rd_u64 = |pos: &mut usize| -> Result<u64, String> {
            if *pos + 8 > payload.len() { return Err("archive: eof u64".into()); }
            let v = u64::from_le_bytes(payload[*pos..*pos+8].try_into().unwrap());
            *pos += 8;
            Ok(v)
        };
        let rows = rd_u32(&mut pos)? as usize;
        let cols = rd_u32(&mut pos)? as usize;
        let pcols = rd_u32(&mut pos)? as usize;
        let mut rd_bytes = |pos: &mut usize| -> Result<Vec<u8>, String> {
            let n = rd_u64(&mut *pos)? as usize;
            if *pos + n > payload.len() { return Err("archive: eof bytes".into()); }
            let v = payload[*pos..*pos+n].to_vec();
            *pos += n;
            Ok(v)
        };
        let weights_i4 = rd_bytes(&mut pos)?;
        let resid = rd_bytes(&mut pos)?;
        let mut rd_f32vec = |pos: &mut usize| -> Result<Vec<f32>, String> {
            let n = rd_u64(&mut *pos)? as usize;
            if *pos + 4*n > payload.len() { return Err("archive: eof f32".into()); }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(f32::from_le_bytes(payload[*pos+4*i..*pos+4*i+4].try_into().unwrap()));
            }
            *pos += 4*n;
            Ok(v)
        };
        let scales = rd_f32vec(&mut pos)?;
        let zeros = rd_f32vec(&mut pos)?;
        let rs = rd_f32vec(&mut pos)?;
        let rz = rd_f32vec(&mut pos)?;
        let mut rd_i32vec = |pos: &mut usize| -> Result<Vec<i32>, String> {
            let n = rd_u64(&mut *pos)? as usize;
            if *pos + 4*n > payload.len() { return Err("archive: eof i32".into()); }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(i32::from_le_bytes(payload[*pos+4*i..*pos+4*i+4].try_into().unwrap()));
            }
            *pos += 4*n;
            Ok(v)
        };
        let code_sums = rd_i32vec(&mut pos)?;
        let rsum = rd_i32vec(&mut pos)?;
        let mut rd_u64vec = |pos: &mut usize| -> Result<Vec<u64>, String> {
            let n = rd_u64(&mut *pos)? as usize;
            if *pos + 8*n > payload.len() { return Err("archive: eof u64vec".into()); }
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(u64::from_le_bytes(payload[*pos+8*i..*pos+8*i+8].try_into().unwrap()));
            }
            *pos += 8*n;
            Ok(v)
        };
        let row_sigs = rd_u64vec(&mut pos)?;
        let bias = rd_f32vec(&mut pos)?;
        let cold_start_done = rd_u32(&mut pos)? != 0;
        if pos != payload.len() {
            return Err("archive: хвост payload".into());
        }
        Ok(Self {
            rows, cols, pcols,
            weights_i4, resid, scales, zeros, code_sums, rs, rz, rsum,
            bias, row_sigs, cold_start_done,
        })
    }
}

// ═══════════════════════════════════════════════════════════
// FlashInt16: GEMV целиком на целочисленном SIMD (Q16×Q16).
//
// Идея: веса и вход квантуются в i16-коды с per-block (256) f32
// scale/zero (как Micron, гарантия |err| <= scale/2). Скалярное
// произведение считается через _mm256_madd_epi16 (i16×i16 → i32):
// 16 элементов за инструкцию против 8 у f32 FMA, при 2 байтах на
// элемент вместо 4 — полоса памяти делится на 2, элементная
// скорость умножается на 2. Блокировочные поправки — f32-скаляры.
// Это путь «перейти BLAS» на GEMV, где f32-ядро упирается в DRAM.
// ═══════════════════════════════════════════════════════════

/// Максимум кода: 7000 (≈13 бит). Произведение пары кодов ≤ 2·7000² ≈ 9.8e7;
/// за блок 256 элементов в i32-лан набегает ≤ 256·7000² ≈ 1.25e10?? нет:
/// лан накапливает по одному произведению пары за madd (2 элемента),
/// 16 madd на блок → 16·2·7000² ≈ 1.57e9 < i32 max — переполнения нет,
/// расширение в i64 нужно один раз на блок.
pub const I16_CODE_MAX: i32 = 7_000;

pub struct FlashInt16 {
    pub rows: usize,
    pub cols: usize,
    pub pcols: usize,
    pub wq: Vec<i16>,
    pub wscales: Vec<f32>,
    pub wzeros: Vec<f32>,
    pub wsum: Vec<i32>,
    pub bias: Vec<f32>,
    /// XOR-подписи строк (знак кодов) — «холодный старт» на уровне строк:
    /// мёртвые/холодные строки можно пропускать (дешевле шина) или
    /// обрабатывать низкой точностью.
    pub row_sigs: Vec<u64>,
    // ── Rineta-тиринг: холодные строки живут в Q8 (1 байт на элемент) ──
    /// Тир строки: 0 = горячая (Q16), 1 = холодная (Q8).
    pub tier: Vec<u8>,
    /// Индекс холодной строки в Q8-плане (для быстрого доступа в ядре).
    pub cold_idx: Vec<u32>,
    /// Индекс горячей строки в Q16-плане (компактный: только горячие).
    pub hot_idx: Vec<u32>,
    /// Холодный план: Q8-коды (сжато, без пропусков).
    pub wq8: Vec<i8>,
    pub w8scales: Vec<f32>,
    pub w8zeros: Vec<f32>,
    pub w8sum: Vec<i32>,
    // ── XOR-контур: подписи входа/выхода каждого вызова ──
    /// XOR-подпись последнего входа (активация предыдущего слоя).
    pub last_in_sig: u64,
    /// XOR-подпись последнего выхода (эта активация).
    pub last_out_sig: u64,
    /// hamming(выход сейчас, выход прошлого вызова) — «сколько изменилось».
    pub last_hamming: u32,
    /// Число вызовов GEMV (счётчик шагов контура).
    pub calls: u64,
}

/// Коды Q8-тира (±127).
pub const I8_CODE_MAX: i32 = 127;

/// Доля средней энергии строки: ниже — холодный тир Q8.
const COLD_ENERGY_FRAC: f32 = 0.05;

impl FlashInt16 {
    pub fn from_dense(w: &[f32], rows: usize, cols: usize, bias: Vec<f32>) -> Self {
        assert_eq!(w.len(), rows * cols);
        let pcols = ((cols + BLOCK - 1) / BLOCK) * BLOCK;
        let blocks = pcols / BLOCK;
        // Компактный Q16-план: только горячие строки.
        let mut wq: Vec<i16> = Vec::with_capacity(rows * pcols);
        let mut wscales = Vec::with_capacity(rows * blocks);
        let mut wzeros = Vec::with_capacity(rows * blocks);
        let mut wsum = Vec::with_capacity(rows * blocks);
        let mut row_sigs = Vec::with_capacity(rows);
        let mut hot_idx = vec![u32::MAX; rows];
        let mut n_hot = 0u32;
        // Rineta-тиринг: средняя энергия строки для порога холода.
        let mut l1s: Vec<f32> = Vec::with_capacity(rows);
        for i in 0..rows {
            l1s.push(
                w[i * cols..(i + 1) * cols]
                    .iter()
                    .map(|&v| v.abs())
                    .sum::<f32>(),
            );
        }
        let mean_l1 = if rows > 0 {
            l1s.iter().sum::<f32>() / rows as f32
        } else {
            1.0
        };
        let mut wq8: Vec<i8> = Vec::new();
        let mut w8scales = Vec::new();
        let mut w8zeros = Vec::new();
        let mut w8sum = Vec::new();
        let mut tier = vec![0u8; rows];
        let mut cold_idx = vec![u32::MAX; rows];
        let mut n_cold = 0u32;

        for i in 0..rows {
            let row = &w[i * cols..(i + 1) * cols];
            let sig = simd::xor_signature(row);
            row_sigs.push(sig);
            // Холодный тир: почти нулевые строки (энергия << средняя)
            // или вырожденная подпись (бит живости <= 8).
            let l1 = l1s[i];
            if l1 <= 1e-12 {
                // Тир 2: строка полностью нулевая — мёртвая, не хранится
                // нигде, ядра пропускают её (y = bias) в любом режиме.
                tier[i] = 2;
            } else if (l1 <= COLD_ENERGY_FRAC * mean_l1 || sig.count_ones() <= 8)
                && l1 > 1e-12
            {
                tier[i] = 1;
                cold_idx[i] = n_cold;
                n_cold += 1;
            }
            if tier[i] == 0 {
                hot_idx[i] = n_hot;
                n_hot += 1;
                for b in 0..blocks {
                    let j0 = b * BLOCK;
                    let j1 = ((b + 1) * BLOCK).min(cols);
                    let chunk = &row[j0..j1];
                    // Micron-схема: zero = среднее блока, scale = max|v-z|/CODE_MAX,
                    // гарантия |err| <= scale/2 (коды покрывают ±CODE_MAX).
                    let z: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                    let m = chunk
                        .iter()
                        .map(|&v| (v - z).abs())
                        .fold(0.0f32, f32::max);
                    let s = (m / I16_CODE_MAX as f32).max(1e-12);
                    let mut sum = 0i32;
                    for (_j, &v) in chunk.iter().enumerate() {
                        let c = ((v - z) / s)
                            .round()
                            .clamp(-I16_CODE_MAX as f32, I16_CODE_MAX as f32) as i16;
                        wq.push(c);
                        sum += c as i32;
                    }
                    for _ in j1 - j0..BLOCK {
                        wq.push(0);
                    }
                    wscales.push(s);
                    wzeros.push(z);
                    wsum.push(sum);
                }
            } else if tier[i] == 1 {
                // Холодный тир Q8: по одному байту на элемент.
                for b in 0..blocks {
                    let j0 = b * BLOCK;
                    let j1 = ((b + 1) * BLOCK).min(cols);
                    let chunk = &row[j0..j1];
                    let z: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                    let m = chunk
                        .iter()
                        .map(|&v| (v - z).abs())
                        .fold(0.0f32, f32::max);
                    let s = (m / I8_CODE_MAX as f32).max(1e-12);
                    let mut sum = 0i32;
                    for (_j, &v) in chunk.iter().enumerate() {
                        let c = ((v - z) / s)
                            .round()
                            .clamp(-I8_CODE_MAX as f32, I8_CODE_MAX as f32) as i8;
                        wq8.push(c);
                        sum += c as i32;
                    }
                    for _ in j1 - j0..BLOCK {
                        wq8.push(0);
                    }
                    w8scales.push(s);
                    w8zeros.push(z);
                    w8sum.push(sum);
                }
            }
        }
        Self {
            rows,
            cols,
            pcols,
            wq,
            wscales,
            wzeros,
            wsum,
            bias,
            row_sigs,
            tier,
            cold_idx,
            hot_idx,
            wq8,
            w8scales,
            w8zeros,
            w8sum,
            last_in_sig: 0,
            last_out_sig: 0,
            last_hamming: 0,
            calls: 0,
        }
    }

    /// XOR-контур: подпись входа/выхода, hamming между вызовами.
    fn track(&mut self, x: &[f32], y: &[f32]) {
        let in_sig = simd::xor_signature(x);
        let out_sig = simd::xor_signature(y);
        if self.calls > 0 {
            self.last_hamming = simd::hamming_distance(self.last_out_sig, out_sig);
        } else {
            self.last_hamming = 0;
        }
        self.last_in_sig = in_sig;
        self.last_out_sig = out_sig;
        self.calls += 1;
    }

    pub fn hot_rows(&self) -> Vec<usize> {
        self.tier
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == 0)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn cold_rows(&self) -> Vec<usize> {
        self.tier
            .iter()
            .enumerate()
            .filter(|(_, &t)| t == 1)
            .map(|(i, _)| i)
            .collect()
    }

    /// Живость строки по XOR-подписи (бит на знак элемента).
    pub fn row_liveness(&self, i: usize) -> u32 {
        self.row_sigs[i].count_ones()
    }

    /// Горячие/мёртвые по XOR-живости (не путать с тиром: тир = энергия).
    pub fn hot_rows_by_bits(&self, min_bits: u32) -> Vec<usize> {
        self.row_sigs
            .iter()
            .enumerate()
            .filter(|(_, &s)| s.count_ones() >= min_bits)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn dead_rows_by_bits(&self, max_bits: u32) -> Vec<usize> {
        self.row_sigs
            .iter()
            .enumerate()
            .filter(|(_, &s)| s.count_ones() <= max_bits)
            .map(|(i, _)| i)
            .collect()
    }

    /// GEMV со «скоростными шинами по холодному старту»: строки, чья
    /// XOR-живость <= min_bits, считаются мёртвыми — их веса не читаются
    /// из памяти (дешевле шина), ответ = bias. Возвращает (y, число
    /// пропущенных строк). Мёртвые строки в обученных сетях нередки:
    /// пропуск экономит ровно их долю трафика W.
    pub fn gemv_skip_cold(&mut self, x: &[f32], min_bits: u32) -> (Vec<f32>, usize) {
        let mut y = self.gemv_impl(x, min_bits);
        let mut skipped = 0usize;
        for i in 0..self.rows {
            if self.row_sigs[i].count_ones() <= min_bits {
                y[i] = self.bias[i];
                skipped += 1;
            }
        }
        self.track(x, &y);
        (y, skipped)
    }

    /// Внутренний GEMV: строки с XOR-живостью <= skip_bits пропускаются
    /// прямо в ядре (их веса не читаются — меньше трафика шины).
    fn gemv_impl(&self, x: &[f32], skip_bits: u32) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let blocks = self.pcols / BLOCK;
        let mut xp = vec![0.0f32; self.pcols];
        xp[..self.cols].copy_from_slice(x);

        let mut xq = vec![0i16; self.pcols];
        let mut xs = vec![0.0f32; blocks];
        let mut xz = vec![0.0f32; blocks];
        let mut xsum = vec![0i32; blocks];
        self.quant_block(xp.as_slice(), &mut xq, &mut xs, &mut xz, &mut xsum, 0, blocks);

        let mut y = vec![0.0f32; self.rows];
        #[cfg(target_arch = "x86_64")]
        {
            if simd::has_avx2() {
                unsafe { self.kernel_avx2(&xq, &xs, &xz, &xsum, skip_bits, &mut y) };
                return y;
            }
        }
        self.kernel_scalar(&xq, &xs, &xz, &xsum, skip_bits, &mut y);
        y
    }

    /// y = W x: вход квантуется в i16-коды (per-block), ядро — madd_epi16.
    pub fn gemv(&mut self, x: &[f32]) -> Vec<f32> {
        let y = self.gemv_impl(x, 0);
        self.track(x, &y);
        y
    }

    /// Квантование одной строки x (per-block mean/scale, коды ±I16_CODE_MAX).
    /// Хвост (cols..pcols) считается нулями — блоки никогда не выходят за pcols.
    fn quant_block(
        &self,
        x: &[f32],
        xq: &mut [i16],
        xs: &mut [f32],
        xz: &mut [f32],
        xsum: &mut [i32],
        row: usize,
        blocks: usize,
    ) {
        let xrow = &x[row * self.cols..row * self.cols + self.cols];
        let qo = row * self.pcols;
        for b in 0..blocks {
            let j0 = b * BLOCK;
            let mut z = 0.0f32;
            let mut n = 0usize;
            for j in 0..BLOCK {
                let idx = j0 + j;
                if idx < self.cols {
                    z += xrow[idx];
                    n += 1;
                }
            }
            if n == 0 {
                n = 1;
            }
            z /= n as f32;
            let mut m = 0.0f32;
            for j in 0..BLOCK {
                let idx = j0 + j;
                if idx < self.cols {
                    let d = (xrow[idx] - z).abs();
                    if d > m {
                        m = d;
                    }
                }
            }
            let s = (m / I16_CODE_MAX as f32).max(1e-12);
            let mut sum = 0i32;
            for j in 0..BLOCK {
                let idx = j0 + j;
                let v = if idx < self.cols { xrow[idx] } else { 0.0 };
                let c = ((v - z) / s).round().clamp(-I16_CODE_MAX as f32, I16_CODE_MAX as f32) as i16;
                xq[qo + j0 + j] = c;
                sum += c as i32;
            }
            xs[row * blocks + b] = s;
            xz[row * blocks + b] = z;
            xsum[row * blocks + b] = sum;
        }
    }

    /// Батч: out[b][i] = Σ_j W[i][j]·X[b][j] для B векторов X (b×cols).
    ///
    /// Структура: строка W читается ОДИН раз и переиспользуется всеми B
    /// векторами батча (в отличие от B вызовов gemv, где W читается B раз).
    /// Коды ±I16_CODE_MAX: i32-лан за блок не переполняется, i64-расширение
    /// раз на блок; поправки scale/zero — в f32 после целочисленного ядра.
    pub fn gemv_batch(&self, xb: &[f32], b: usize) -> Vec<f32> {
        assert_eq!(xb.len(), b * self.cols);
        let blocks = self.pcols / BLOCK;
        let mut xq = vec![0i16; b * self.pcols];
        let mut xs = vec![0.0f32; b * blocks];
        let mut xz = vec![0.0f32; b * blocks];
        let mut xsum = vec![0i32; b * blocks];
        for r in 0..b {
            self.quant_block(xb, &mut xq, &mut xs, &mut xz, &mut xsum, r, blocks);
        }

        let mut y = vec![0.0f32; b * self.rows];
        #[cfg(target_arch = "x86_64")]
        {
            if simd::has_avx2() {
                unsafe { self.kernel_batch_avx2(&xq, &xs, &xz, &xsum, b, &mut y) };
                return y;
            }
        }
        for r in 0..b {
            self.kernel_scalar(
                &xq[r * self.pcols..(r + 1) * self.pcols],
                &xs[r * blocks..(r + 1) * blocks],
                &xz[r * blocks..(r + 1) * blocks],
                &xsum[r * blocks..(r + 1) * blocks],
                0,
                &mut y[r * self.rows..(r + 1) * self.rows],
            );
        }
        y
    }

    /// Батч-ядро v2 (2D register blocking): тайлы строк × группы по 8
    /// векторов батча. W-чанк грузится 1 раз на 8 векторов, 8 i32-
    /// аккумуляторов живут в регистрах, поправки scale/zero считаются
    /// SIMD сразу по 8 векторам (в v1 они были скалярными на пару).
    #[cfg(target_arch = "x86_64")]
    unsafe fn kernel_batch_avx2(
        &self,
        xq: &[i16],
        xs: &[f32],
        xz: &[f32],
        xsum: &[i32],
        b: usize,
        y: &mut [f32],
    ) {
        use std::arch::x86_64::*;
        let blocks = self.pcols / BLOCK;
        let blk_f = BLOCK as f32;
        let wp = self.wq.as_ptr();
        let qp = xq.as_ptr();
        const GROUP: usize = 8;
        // Авто-тайл под L2: весовая плитка (TILE строк x pcols x 2B)
        // должна умещаться в половину L2 (кэш-иерархия как у CPU).
        let l2 = crate::atom::flash::flash_hw::l3_bytes();
        let l2_bytes = if l2 == 0 {
            512 * 1024
        } else {
            (l2 / 2).min(2 * 1024 * 1024).max(64 * 1024)
        };
        let row_bytes = self.pcols * 2;
        let tile_max = (l2_bytes / row_bytes.max(1)).max(8);
        let mut TILE = tile_max.min(self.rows).max(1);
        // кратность GROUP для ровного деления
        TILE = (TILE / GROUP).max(1) * GROUP;

        let mut i0 = 0usize;
        while i0 + TILE <= self.rows {
            let i1 = i0 + TILE;
            let mut g0 = 0usize;
            while g0 + GROUP <= b {
                for i in i0..i1 {
                    // Мёртвые (тир 2): только bias, веса не читаются.
                    if self.tier[i] == 2 {
                        for rr in 0..GROUP {
                            y[(g0 + rr) * self.rows + i] = self.bias[i];
                        }
                        continue;
                    }
                    // Rineta-тир: холодные строки в батче — Q8-скалярно
                    // (их мало; основной выигрыш — компактный Q16-план).
                    if self.tier[i] == 1 {
                        let c = self.cold_idx[i] as usize;
                        for rr in 0..GROUP {
                            let r_abs = g0 + rr;
                            let bb = r_abs * blocks;
                            y[r_abs * self.rows + i] = self.bias[i]
                                + self.cold_dot_scalar(
                                    c,
                                    &xq[r_abs * self.pcols..(r_abs + 1) * self.pcols],
                                    &xs[bb..bb + blocks],
                                    &xz[bb..bb + blocks],
                                    &xsum[bb..bb + blocks],
                                    blocks,
                                );
                        }
                        continue;
                    }
                    if i + 8 < i1 && self.hot_idx[i + 8] != u32::MAX {
                        _mm_prefetch(
                            wp.add(self.hot_idx[i + 8] as usize * self.pcols) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                    let wrow = wp.add(self.hot_idx[i] as usize * self.pcols);
                    let mut vy = _mm256_setzero_ps();

                    for blk in 0..blocks {
                        let wb = wrow.add(blk * BLOCK);
                        let mut a0 = _mm256_setzero_si256();
                        let mut a1 = _mm256_setzero_si256();
                        let mut a2 = _mm256_setzero_si256();
                        let mut a3 = _mm256_setzero_si256();
                        let mut a4 = _mm256_setzero_si256();
                        let mut a5 = _mm256_setzero_si256();
                        let mut a6 = _mm256_setzero_si256();
                        let mut a7 = _mm256_setzero_si256();
                        let xb0 = qp.add(g0 * self.pcols + blk * BLOCK);
                        for c in 0..(BLOCK / 16) {
                            let cw = _mm256_loadu_si256(wb.add(c * 16) as *const __m256i);
                            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(c * 16) as *const __m256i)));
                            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(1 * self.pcols + c * 16) as *const __m256i)));
                            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(2 * self.pcols + c * 16) as *const __m256i)));
                            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(3 * self.pcols + c * 16) as *const __m256i)));
                            a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(4 * self.pcols + c * 16) as *const __m256i)));
                            a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(5 * self.pcols + c * 16) as *const __m256i)));
                            a6 = _mm256_add_epi32(a6, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(6 * self.pcols + c * 16) as *const __m256i)));
                            a7 = _mm256_add_epi32(a7, _mm256_madd_epi16(cw, _mm256_loadu_si256(xb0.add(7 * self.pcols + c * 16) as *const __m256i)));
                        }
                        // Flush: dot каждого из 8 векторов (i64) + поправки SIMD.
                        let mut dt = [0i32; 8 * GROUP];
                        _mm256_storeu_si256(dt.as_mut_ptr() as *mut __m256i, a0);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(8) as *mut __m256i, a1);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(16) as *mut __m256i, a2);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(24) as *mut __m256i, a3);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(32) as *mut __m256i, a4);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(40) as *mut __m256i, a5);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(48) as *mut __m256i, a6);
                        _mm256_storeu_si256(dt.as_mut_ptr().add(56) as *mut __m256i, a7);

                        let wbase = self.hot_idx[i] as usize * blocks + blk;
                        let sw = self.wscales[wbase];
                        let zw = self.wzeros[wbase];
                        let wsumf = self.wsum[wbase] as f32;
                        let mut acc_f = [0.0f32; GROUP];
                        for r in 0..GROUP {
                            let mut dot: i64 = 0;
                            for k in 0..8 {
                                dot += dt[r * 8 + k] as i64;
                            }
                            let xbase = (g0 + r) * blocks + blk;
                            let s = xs[xbase];
                            let z = xz[xbase];
                            acc_f[r] = sw * s * dot as f32
                                + sw * z * wsumf
                                + zw * s * xsum[xbase] as f32
                                + zw * z * blk_f;
                        }
                        vy = _mm256_add_ps(vy, _mm256_loadu_ps(acc_f.as_ptr()));
                    }
                    let mut vt = [0.0f32; GROUP];
                    _mm256_storeu_ps(vt.as_mut_ptr(), vy);
                    for r in 0..GROUP {
                        y[(g0 + r) * self.rows + i] = self.bias[i] + vt[r];
                    }
                }
                g0 += GROUP;
            }
            i0 += TILE;
        }

        // Хвосты: остаток строк (r < r0tail) и остаток батча (r >= r0tail).
        let r0tail = (b / GROUP) * GROUP;
        for r in 0..b {
            let istart = if r < r0tail { i0 } else { 0 };
            let qrow = qp.add(r * self.pcols);
            for i in istart..self.rows {
                if self.tier[i] == 2 {
                    y[r * self.rows + i] = self.bias[i];
                    continue;
                }
                if self.tier[i] == 1 {
                    let c = self.cold_idx[i] as usize;
                    let bb = r * blocks;
                    y[r * self.rows + i] = self.bias[i]
                        + self.cold_dot_scalar(
                            c,
                            &xq[r * self.pcols..(r + 1) * self.pcols],
                            &xs[bb..bb + blocks],
                            &xz[bb..bb + blocks],
                            &xsum[bb..bb + blocks],
                            blocks,
                        );
                    continue;
                }
                let mut acc_f = 0.0f32;
                for blk in 0..blocks {
                    let wb = wp.add(self.hot_idx[i] as usize * self.pcols + blk * BLOCK);
                    let qb = qrow.add(blk * BLOCK);
                    let mut a0 = _mm256_setzero_si256();
                    for c in 0..(BLOCK / 16) {
                        let cw = _mm256_loadu_si256(wb.add(c * 16) as *const __m256i);
                        let cx = _mm256_loadu_si256(qb.add(c * 16) as *const __m256i);
                        a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(cw, cx));
                    }
                    let mut t = [0i32; 8];
                    _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a0);
                    let dot: i64 = t.iter().map(|&v| v as i64).sum();
                    let wbase = self.hot_idx[i] as usize * blocks + blk;
                    let xbase = r * blocks + blk;
                    acc_f += self.wscales[wbase] * xs[xbase] * dot as f32
                        + self.wscales[wbase] * xz[xbase] * self.wsum[wbase] as f32
                        + self.wzeros[wbase] * xs[xbase] * xsum[xbase] as f32
                        + self.wzeros[wbase] * xz[xbase] * blk_f;
                }
                y[r * self.rows + i] = self.bias[i] + acc_f;
            }
        }
    }

    /// Q8-тир: dot холодной строки (коды i8 расширяются до i16 на лету).
    #[cfg(target_arch = "x86_64")]
    unsafe fn cold_dot_avx2(
        &self,
        cold_row: usize,
        xq: &[i16],
        xs: &[f32],
        xz: &[f32],
        xsum: &[i32],
        blocks: usize,
    ) -> f32 {
        use std::arch::x86_64::*;
        let blk_f = BLOCK as f32;
        let p8 = self.wq8.as_ptr().add(cold_row * self.pcols);
        let qp = xq.as_ptr();
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let wb = p8.add(b * BLOCK);
            let qb = qp.add(b * BLOCK);
            let mut a0 = _mm256_setzero_si256();
            for c in 0..(BLOCK / 16) {
                // 16 кодов i8 (16 байт) -> 16 x i16, madd с Q16-входом.
                let cw8 = _mm_loadu_si128(wb.add(c * 16) as *const __m128i);
                let cw = _mm256_cvtepi8_epi16(cw8);
                let cx = _mm256_loadu_si256(qb.add(c * 16) as *const __m256i);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(cw, cx));
            }
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a0);
            let dot: i64 = t.iter().map(|&v| v as i64).sum();
            let base = cold_row * blocks + b;
            let sw = self.w8scales[base];
            let zw = self.w8zeros[base];
            let s = xs[b];
            let z = xz[b];
            acc += sw * s * dot as f32
                + sw * z * self.w8sum[base] as f32
                + zw * s * xsum[b] as f32
                + zw * z * blk_f;
        }
        acc
    }

    /// Q8-тир: скалярная версия (не-x86 и батч-хвосты).
    fn cold_dot_scalar(
        &self,
        cold_row: usize,
        xq: &[i16],
        xs: &[f32],
        xz: &[f32],
        xsum: &[i32],
        blocks: usize,
    ) -> f32 {
        let blk_f = BLOCK as f32;
        let p8 = &self.wq8[cold_row * self.pcols..(cold_row + 1) * self.pcols];
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let j0 = b * BLOCK;
            let mut dot = 0i64;
            for j in 0..BLOCK {
                dot += p8[j0 + j] as i64 * xq[j0 + j] as i64;
            }
            let base = cold_row * blocks + b;
            let sw = self.w8scales[base];
            let zw = self.w8zeros[base];
            let s = xs[b];
            let z = xz[b];
            acc += sw * s * dot as f32
                + sw * z * self.w8sum[base] as f32
                + zw * s * xsum[b] as f32
                + zw * z * blk_f;
        }
        acc
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn kernel_avx2(
        &self,
        xq: &[i16],
        xs: &[f32],
        xz: &[f32],
        xsum: &[i32],
        skip_bits: u32,
        y: &mut [f32],
    ) {
        let threads = crate::atom::veron::parallel_threads();
        let parallel = threads > 1 && self.rows >= 256 && self.pcols >= 512;
        if parallel {
            // Многопоточность по строкам: каждая строка считается тем же
            // кодом, что и последовательно, в отдельном буфере, затем
            // буферы склеиваются в исходном порядке — результат БИТ-В-БИТ
            // равен однопоточному (качество не теряется).
            let n = self.rows;
            let parts = threads.min(n);
            let per = n.div_ceil(parts);
            let ranges: Vec<(usize, usize)> = (0..parts)
                .map(|p| (p * per, ((p + 1) * per).min(n)))
                .filter(|(a, b)| a < b)
                .collect();
            let partials: Vec<Vec<f32>> = ranges
                .par_iter()
                .map(|&(r0, r1)| {
                    let mut out = vec![0.0f32; r1 - r0];
                    for (k, i) in (r0..r1).enumerate() {
                        out[k] = self.row_dot_avx2(i, xq, xs, xz, xsum, skip_bits);
                    }
                    out
                })
                .collect();
            for (part, &(r0, _)) in partials.iter().zip(&ranges) {
                y[r0..r0 + part.len()].copy_from_slice(part);
            }
            return;
        }
        for i in 0..self.rows {
            y[i] = self.row_dot_avx2(i, xq, xs, xz, xsum, skip_bits);
        }
    }

    /// Результат одной строки (общий код для 1 и N потоков — детерминизм).
    #[cfg(target_arch = "x86_64")]
    unsafe fn row_dot_avx2(
        &self,
        i: usize,
        xq: &[i16],
        xs: &[f32],
        xz: &[f32],
        xsum: &[i32],
        skip_bits: u32,
    ) -> f32 {
        use std::arch::x86_64::*;
        let blocks = self.pcols / BLOCK;
        let wp = self.wq.as_ptr();
        let qp = xq.as_ptr();
        let blk_f = BLOCK as f32;

        // «Скоростная шина» по холодному старту: мёртвая строка не
        // читается из памяти вовсе — сразу bias.
        if self.tier[i] == 2
            || (skip_bits > 0 && self.row_sigs[i].count_ones() <= skip_bits)
        {
            return self.bias[i];
        }
        // Rineta-тир: холодная строка живёт в Q8-плане (1 байт/элемент).
        if self.tier[i] == 1 {
            let c = self.cold_idx[i] as usize;
            return self.bias[i] + self.cold_dot_avx2(c, xq, xs, xz, xsum, blocks);
        }
        let p0 = wp.add(self.hot_idx[i] as usize * self.pcols);
        let mut acc = 0.0f32;

        for b in 0..blocks {
            let wb = p0.add(b * BLOCK);
            let qb = qp.add(b * BLOCK);
            // Коды ±7000: лан i32 за 16 madds не переполняется
            // (16·2·7000² ≈ 1.57e9 < 2.147e9), расширение — раз на блок.
            let mut a0 = _mm256_setzero_si256();
            for c in 0..(BLOCK / 16) {
                let cw = _mm256_loadu_si256(wb.add(c * 16) as *const __m256i);
                let cx = _mm256_loadu_si256(qb.add(c * 16) as *const __m256i);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(cw, cx));
            }
            let mut t = [0i32; 8];
            _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, a0);
            let dot: i64 = t.iter().map(|&v| v as i64).sum();

            let base = self.hot_idx[i] as usize * blocks + b;
            let sw = self.wscales[base];
            let zw = self.wzeros[base];
            let s = xs[b];
            let z = xz[b];
            acc += sw * s * dot as f32
                + sw * z * self.wsum[base] as f32
                + zw * s * xsum[b] as f32
                + zw * z * blk_f;
        }
        self.bias[i] + acc
    }

    fn kernel_scalar(
        &self,
        xq: &[i16],
        xs: &[f32],
        xz: &[f32],
        xsum: &[i32],
        skip_bits: u32,
        y: &mut [f32],
    ) {
        let blocks = self.pcols / BLOCK;
        let blk_f = BLOCK as f32;
        for i in 0..self.rows {
            if self.tier[i] == 2
                || (skip_bits > 0 && self.row_sigs[i].count_ones() <= skip_bits)
            {
                y[i] = self.bias[i];
                continue;
            }
            if self.tier[i] == 1 {
                let c = self.cold_idx[i] as usize;
                y[i] = self.bias[i] + self.cold_dot_scalar(c, xq, xs, xz, xsum, blocks);
                continue;
            }
            let row = &self.wq[self.hot_idx[i] as usize * self.pcols..(self.hot_idx[i] as usize + 1) * self.pcols];
            let mut acc = 0.0f32;
            for b in 0..blocks {
                let j0 = b * BLOCK;
                let mut dot = 0i64;
                for j in 0..BLOCK {
                    dot += row[j0 + j] as i64 * xq[j0 + j] as i64;
                }
                let base = self.hot_idx[i] as usize * blocks + b;
                let sw = self.wscales[base];
                let zw = self.wzeros[base];
                let s = xs[b];
                let z = xz[b];
                acc += sw * s * dot as f32
                    + sw * z * self.wsum[base] as f32
                    + zw * s * xsum[b] as f32
                    + zw * z * blk_f;
            }
            y[i] = self.bias[i] + acc;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.wq.len() * 2
            + (self.wscales.len() + self.wzeros.len()) * 4
            + self.wsum.len() * 4
            + self.wq8.len()
            + (self.w8scales.len() + self.w8zeros.len()) * 4
            + self.w8sum.len() * 4
            + self.tier.len()
            + self.cold_idx.len() * 4
            + self.bias.len() * 4
            + self.row_sigs.len() * 8
    }

    pub fn compression_ratio(&self) -> f32 {
        (self.rows * self.cols * 4) as f32 / self.memory_bytes().max(1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::veron;

    /// Многопоточность Veron не меняет результат: бит-в-бит равенство.
    #[test]
    fn parallel_rows_bitwise_equal() {
        let rows = 2048usize;
        let cols = 1024usize;
        let mut w = vec![0.0f32; rows * cols];
        for (k, v) in w.iter_mut().enumerate() {
            // Детерминированный псевдослучайный источник без внешних RNG.
            let x = (k as u32).wrapping_mul(2654435761) | 1;
            let u = (x >> 8) as f32 / (1u32 << 24) as f32;
            *v = (u - 0.5) * 0.04;
        }
        let bias = vec![0.0f32; rows];
        let mut layer = FlashInt16::from_dense(&w, rows, cols, bias);
        let x: Vec<f32> = (0..cols)
            .map(|k| {
                let u = (k.wrapping_mul(2246822519) >> 8) as f32 / (1u32 << 24) as f32;
                (u - 0.5) * 0.04
            })
            .collect();

        // Однопоточно
        veron::set_parallel_threads(1);
        let y1 = layer.gemv(&x);
        // Многопоточно
        veron::set_parallel_threads(4);
        let y2 = layer.gemv(&x);
        veron::set_parallel_threads(1);

        assert_eq!(y1.len(), y2.len());
        for (a, b) in y1.iter().zip(&y2) {
            assert_eq!(*a, *b, "многопоточный результат должен совпадать бит-в-бит");
        }
        // Санити: результат не нули и конечен.
        assert!(y1.iter().any(|&v| v.abs() > 1e-6));
        assert!(y1.iter().all(|v| v.is_finite()));
    }

    /// Q4×Q8: fused AVX2-ядро должно быть детерминированным и finite.
    #[cfg(all(test, target_arch = "x86_64"))]
    #[test]
    fn q4_fused_kernel_avx2_deterministic() {
        let rows = 3usize;
        let cols = 256usize;
        let mut w = vec![0.0f32; rows * cols];
        for (k, v) in w.iter_mut().enumerate() {
            let x = (k as u32).wrapping_mul(2654435761) | 1;
            let u = (x >> 8) as f32 / (1u32 << 24) as f32;
            *v = (u - 0.5) * 0.04;
        }
        let bias = vec![0.0f32; rows];
        let layer = FlashCompressed::from_dense(&w, rows, cols, bias);
        let x: Vec<f32> = (0..cols)
            .map(|k| {
                let u = (k.wrapping_mul(2246822519) >> 8) as f32 / (1u32 << 24) as f32;
                (u - 0.5) * 0.04
            })
            .collect();

        let blocks = layer.pcols / BLOCK;
        let bpr = layer.pcols / 2;
        let mut qx = vec![0i8; layer.pcols];
        let mut sx = vec![0.0f32; blocks];
        let mut mx = vec![0.0f32; blocks];
        let mut sq = vec![0i32; blocks];
        {
            let mut xp = vec![0.0f32; layer.pcols];
            xp[..layer.cols].copy_from_slice(&x);
            for b in 0..blocks {
                let xb = &xp[b * BLOCK..(b + 1) * BLOCK];
                let qo = &mut qx[b * BLOCK..(b + 1) * BLOCK];
                let mean: f32 = xb.iter().sum::<f32>() / BLOCK as f32;
                let m = xb.iter().map(|&v| (v - mean).abs()).fold(0.0f32, f32::max);
                let s = (m / 127.0).max(1e-12);
                let inv = 1.0 / s;
                let mut ssum = 0i32;
                for (qv, &xv) in qo.iter_mut().zip(xb.iter()) {
                    let q = ((xv - mean) * inv).round().clamp(-127.0, 127.0) as i8;
                    *qv = q;
                    ssum += q as i32;
                }
                sx[b] = s;
                mx[b] = mean;
                sq[b] = ssum;
            }
        }

        let run = |l: &FlashCompressed| -> Vec<f32> {
            let mut y = vec![0.0f32; rows];
            unsafe { l.kernel_fused_avx2(&qx, &sx, &mx, &sq, &mut y) };
            y
        };
        let y1 = run(&layer);
        let y2 = run(&layer);
        assert_eq!(y1.len(), rows);
        assert_eq!(y1, y2);
        for v in &y1 {
            assert!(v.is_finite(), "Q4 kernel produced non-finite value");
        }
    }

    /// Q4×Q8: DynASM-JIT должен совпадать с AVX2-эталоном.
    #[cfg(all(test, target_arch = "x86_64"))]
    #[test]
    fn q4_jit_matches_avx2_reference() {
        use crate::atom::flash::jit;
        let _jit = jit::TEST_JIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let rows = 16usize;
        let cols = 512usize;
        let mut w = vec![0.0f32; rows * cols];
        for (k, v) in w.iter_mut().enumerate() {
            let x = (k as u32).wrapping_mul(2654435761) | 1;
            let u = (x >> 8) as f32 / (1u32 << 24) as f32;
            *v = (u - 0.5) * 0.04;
        }
        let bias: Vec<f32> = (0..rows).map(|i| (i as f32 - 8.0) * 0.001).collect();
        let layer = FlashCompressed::from_dense(&w, rows, cols, bias);
        let x: Vec<f32> = (0..cols)
            .map(|k| {
                let u = (k.wrapping_mul(2246822519) >> 8) as f32 / (1u32 << 24) as f32;
                (u - 0.5) * 0.04
            })
            .collect();

        // Эталон: AVX2-ядро (JIT выключен).
        jit::set_jit_enabled(false);
        let y_ref = layer.gemv(&x);
        jit::set_jit_enabled(true);

        let has_jit = jit::jit_q4_gemv().is_some();
        let y_jit = layer.gemv(&x);
        jit::set_jit_enabled(true);

        assert!(y_ref.iter().all(|v| v.is_finite()));
        assert!(
            y_ref.iter().any(|&v| v.abs() > 1e-6),
            "эталон не должен быть нулевым"
        );
        if has_jit {
            assert!(
                y_jit.iter().any(|&v| v.abs() > 1e-6),
                "JIT-ядро вернуло нули (заглушка?)"
            );
            // JIT сливает int4+resid по блокам, AVX2 делает два прохода:
            // порядок сложения другой, поэтому допускаем ULP-уровень.
            for (i, (a, b)) in y_jit.iter().zip(y_ref.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-6 * (1.0 + b.abs()),
                    "ряд {i}: JIT {a} != AVX2 {b}"
                );
            }
        }
    }

    /// JIT с управляемым контекстом: точные значения по всем термам Q4×Q8.
    #[cfg(all(test, target_arch = "x86_64"))]
    #[test]
    fn q4_jit_direct_ctx_exact() {
        use crate::atom::flash::jit::{jit_q4_gemv, Q4Ctx};

        let rows = 2usize;
        let blocks = 1usize;
        let pcols = 256usize;

        let mut weights_i4 = vec![0u8; rows * pcols / 2];
        let mut resid = vec![128u8; rows * pcols];
        let mut qx = vec![0i8; pcols];
        let mut scales = vec![1.0f32; rows * blocks];
        let mut zeros = vec![0.0f32; rows * blocks];
        let mut code_sums = vec![0i32; rows * blocks];
        let mut rs = vec![0.0f32; rows * blocks];
        let mut rz = vec![0.0f32; rows * blocks];
        let mut rsum = vec![0i32; rows * blocks];
        let mut sx = vec![1.0f32; blocks];
        let mut mx = vec![0.0f32; blocks];
        let mut sq = vec![0i32; blocks];
        let bias = vec![0.0f32; rows];

        let jq = jit_q4_gemv().expect("JIT должен быть доступен");

        macro_rules! run {
            () => {{
                let mut y = vec![0.0f32; rows];
                let ctx = Q4Ctx {
                    weights_i4: weights_i4.as_ptr(),
                    resid: resid.as_ptr(),
                    qx: qx.as_ptr(),
                    scales: scales.as_ptr(),
                    zeros: zeros.as_ptr(),
                    code_sums: code_sums.as_ptr(),
                    rs: rs.as_ptr(),
                    rz: rz.as_ptr(),
                    rsum: rsum.as_ptr(),
                    sx: sx.as_ptr(),
                    mx: mx.as_ptr(),
                    sq: sq.as_ptr(),
                    bias: bias.as_ptr(),
                    y: y.as_mut_ptr(),
                    blocks: blocks as u64,
                    bpr: (pcols / 2) as u64,
                    rpr: pcols as u64,
                    ycell: 4,
                    rows: rows as u64,
                };
                unsafe { (jq.gemv)(&ctx) };
                y
            }};
        }
        macro_rules! expect {
            ($got:expr, $want:expr) => {{
                let want = $want as f32;
                for (i, v) in $got.iter().enumerate() {
                    assert!(
                        (v - want).abs() <= 1e-3 * (1.0 + want.abs()),
                        "ряд {i}: получили {v}, ждали {want}"
                    );
                }
            }};
        }

        // A: всё нулевое -> 0.
        expect!(run!(), 0.0);

        // B: qsum=5 -> sw*(0 - 8*5) = -40.
        sq[0] = 5;
        expect!(run!(), -40.0);
        sq[0] = 0;

        // C1: nib lo=15, hi=0 -> 15*128 = 1920.
        weights_i4.iter_mut().for_each(|b| *b = 0x0F);
        qx.iter_mut().for_each(|q| *q = 1);
        expect!(run!(), 1920.0);

        // C2: nib lo=0, hi=15 -> 1920.
        weights_i4.iter_mut().for_each(|b| *b = 0xF0);
        expect!(run!(), 1920.0);

        // C3: обе половины qx -> 15*(128*1 + 128*2) = 5760.
        weights_i4.iter_mut().for_each(|b| *b = 0xFF);
        for k in 0..pcols {
            qx[k] = if k < 128 { 1 } else { 2 };
        }
        expect!(run!(), 5760.0);

        // D: qsum=256, nib=15, qx=1 -> 3840 - 8*256 = 1792.
        qx.iter_mut().for_each(|q| *q = 1);
        sq[0] = 256;
        expect!(run!(), 1792.0);

        // E: resid=129 (c=1), rs=1 -> int4 1792 + resid 256 = 2048.
        resid.iter_mut().for_each(|b| *b = 129);
        rs.iter_mut().for_each(|v| *v = 1.0);
        expect!(run!(), 2048.0);

        // F: rz=2, rs=0 -> 1792 + rz*(s*qsum + m*256) = 1792 + 512.
        rs.iter_mut().for_each(|v| *v = 0.0);
        rz.iter_mut().for_each(|v| *v = 2.0);
        resid.iter_mut().for_each(|b| *b = 128);
        expect!(run!(), 1792.0 + 512.0);

        // G: zeros=1 -> int4 += z*(s*qsum + m*256) = 256.
        rz.iter_mut().for_each(|v| *v = 0.0);
        zeros.iter_mut().for_each(|v| *v = 1.0);
        expect!(run!(), 1792.0 + 256.0);

        // H: code_sums=10, mx=1 -> int4 += sw*m*cs = 10.
        zeros.iter_mut().for_each(|v| *v = 0.0);
        code_sums.iter_mut().for_each(|v| *v = 10);
        mx.iter_mut().for_each(|v| *v = 1.0);
        expect!(run!(), 1792.0 + 10.0);

        // I: rsum=10, rs=1 -> resid += rs*m*rsum = 10.
        code_sums.iter_mut().for_each(|v| *v = 0);
        rs.iter_mut().for_each(|v| *v = 1.0);
        rsum.iter_mut().for_each(|v| *v = 10);
        expect!(run!(), 1792.0 + 10.0);

        // J: sx=2 (d0 и qsum масштабируются) -> sw*s*(d0-8q) = 2*1792.
        rs.iter_mut().for_each(|v| *v = 0.0);
        sx[0] = 2.0;
        expect!(run!(), 3584.0);

        // K: scales=3 -> 3*3584.
        scales.iter_mut().for_each(|v| *v = 3.0);
        expect!(run!(), 3.0 * 3584.0);
    }
}

    #[cfg(all(test, target_arch = "x86_64"))]
    #[test]
    fn probe_scalar_single_row_block() {
        use crate::atom::math::simd;
        let mut s = 42424242u64;
        let mut x = vec![0.0f32; 256];
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % 2000) as f32 / 300.0 - 3.0;
        }
        let mut w = vec![0.0f32; 256];
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % 2000) as f32 / 300.0 - 3.0;
        }
        let fc = FlashCompressed::from_dense(&w, 1, 256, vec![0.0f32]);
        let blocks = fc.pcols / BLOCK;
        let mut qx = vec![0i8; fc.pcols];
        let mut sx = vec![0.0f32; blocks];
        let mut mx = vec![0.0f32; blocks];
        let mut sq = vec![0i32; blocks];
        let mut xp = vec![0.0f32; fc.pcols];
        xp.copy_from_slice(&x);
        for b in 0..blocks {
            let xb = &xp[b * BLOCK..(b + 1) * BLOCK];
            let qo = &mut qx[b * BLOCK..(b + 1) * BLOCK];
            let mean = xb.iter().sum::<f32>() / BLOCK as f32;
            let m = xb.iter().map(|&v| (v - mean).abs()).fold(0.0f32, f32::max);
            let ss = (m / 127.0).max(1e-12);
            let inv = 1.0 / ss;
            let mut ssum = 0i32;
            for (qv, &xv) in qo.iter_mut().zip(xb.iter()) {
                let q = ((xv - mean) * inv).round().clamp(-127.0, 127.0) as i8;
                *qv = q; ssum += q as i32;
            }
            sx[b] = ss; mx[b] = mean; sq[b] = ssum;
        }
        let mut y_s = vec![0.0f32; 1];
        fc.kernel_scalar(&qx, &sx, &mx, &sq, &mut y_s);
        // точная инвариант-проверка int4-части вручную:
        let mut manual = 0.0f32;
        for b in 0..blocks {
            let base = b;
            let bp = b * (BLOCK / 2);
            let q0 = b * BLOCK;
            let mut d0: i32 = 0;
            for c in 0..BLOCK / 32 {
                let wv = &fc.weights_i4[bp + c * 16..bp + c * 16 + 16];
                let vq = &qx[q0 + c * 32..q0 + c * 32 + 32];
                for j in 0..32 {
                    let pj = j / 2;
                    let rem = j % 2;
                    let byte = wv[pj];
                    let code = if rem == 0 { byte & 0x0F } else { byte >> 4 } as i8 - 8;
                    d0 += code as i32 * vq[j] as i32;
                }
            }
            let ss = sx[b]; let mm = mx[b]; let qsum = sq[b] as f32;
            manual += fc.scales[base] * ss * (d0 as f32 - 8.0 * qsum)
                + fc.scales[base] * mm * fc.code_sums[base] as f32
                + fc.zeros[base] * ss * qsum
                + fc.zeros[base] * mm * BLOCK as f32;
        }
        println!("y_s={} manual={}", y_s[0], manual);
        assert!((y_s[0] - manual).abs() < 1e-3, "scalar vs manual");
    }

// ═══════════════════════════════════════════════════════════════════
// FP8 block-scale хранения весов (DeepSeek-style, ue8m0)
//   scale_fmt : блок A = 128, clamp amax >= 1e-4, scale = amax/448 ->
//   (при ue8m0 оканчивается к степени 2) — то же, что v3 kernel.py.
//   коды     : E4M3FN (sign + 4exp + 3mant, bias 7, inf/NaN=1111_11x/res 0x7F/0xFF)
//   dequant  : code_float * scale_блока при чтении люми, за строку активации.
// ═══════════════════════════════════════════════════════════════════

/// FP8-блочное квантование (e4m3fn, блок 128 по строкам как в DeepSeekV3 kernel).
#[repr(C)]
#[derive(Clone, Debug)]
pub struct FlashFP8 {
    pub rows: usize,
    pub cols: usize,
    pub block: usize,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub bias: Vec<f32>,
}

/// E4M3FN float32 -> u8 bits: sign, exp(4 bits, bias 7), mantissa(3 bits).
/// NaN/Inf обработан (E4M3FN reserved = 0x7F, 0xFF; Inf saturate в максимальное
/// нормальное 0x7E/E7/0xFE? в FP8 E4M3FN inf вообще нет: sat → max 448).
pub fn f32_to_fp8_e4m3(v_in: f32) -> u8 {
    if v_in.is_nan() {
        return 0x7F;
    }
    // saturation по модулю (448 — максимальное E4M3FN), знаковый бит сохраняем!
    let v = v_in.clamp(-448.0f32, 448.0f32);
    let bits = v.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let exp = (((bits >> 23) & 0xFF) as i32) - 127;
    if exp < -6 {
        // Субнормальные E4M3: значение = mant/8 * 2^-6 = mant * 2^-9
        // (mant 1..=7 -> от 2^-9 и до 7*2^-9 ~= 0.0137). Округление к ближайшему.
        // Минимум (2^-9 = 0.001953125) — всё меньше округляется в 0 (как в torch).
        let mant_v = (v.abs() * 512.0).round() as u8; // = v * 2^9
        return sign | mant_v.min(7);
    }
    let mant = (bits >> 20) & 0x7;
    let exp_i = (exp + 7).min(15) as u8;
    sign | ((exp_i & 0x0F) << 3) | (mant as u8)
}

/// FP8 u8 bits -> float (E4M3FN decode).
pub fn fp8_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant_i = (b & 0x07) as f32;
    if exp == 0 {
        // E4M3 субнормальные: значение = mant_i * 2^-9 (max = 7*2^-9 ≈ 0.0137)
        sign * mant_i * (1.0f32 / 512.0)
    } else {
        // Нормали: (1 + mant_i/8) * 2^(exp - 7)
        sign * (1.0 + mant_i / 8.0) * ((exp - 7) as f32).exp2()
    }
}

impl FlashFP8 {
    pub fn from_dense(w: &[f32], rows: usize, cols: usize, block: usize, bias: Vec<f32>) -> Self {
        assert_eq!(w.len(), rows * cols);
        let blocks_per_row = cols.div_ceil(block);
        let mut codes = vec![0u8; rows * cols];
        let mut scales = vec![0.0f32; blocks_per_row * rows];
        for i in 0..rows {
            let row = &w[i * cols..(i + 1) * cols];
            for b in 0..blocks_per_row {
                let a0 = b * block;
                let a1 = (a0 + block).min(cols);
                let chunk = &row[a0..a1];
                let mut amax = 0.0f32;
                for &v in chunk {
                    let av = v.abs();
                    if av > amax {
                        amax = av;
                    }
                }
                let mut s = (amax / 448.0).clamp(1e-4, f32::MAX);
                // ue8m0: степ до степени 2
                s = s.log2().ceil().exp2();
                scales[i * blocks_per_row + b] = s;
                for (j, &v) in chunk.iter().enumerate() {
                    // E4M3FN packed float
                    codes[i * cols + a0 + j] = f32_to_fp8_e4m3(v / s);
                }
            }
        }
        Self { rows, cols, block, codes, scales, bias }
    }

    pub fn row_values(&self, row: usize) -> Vec<f32> {
        assert!(row < self.rows);
        let blocks_per_row = self.cols.div_ceil(self.block);
        let mut out = Vec::with_capacity(self.cols);
        for c in 0..self.cols {
            let code = self.codes[row * self.cols + c];
            let s = self.scales[row * blocks_per_row + c / self.block];
            out.push(fp8_to_f32(code) * s);
        }
        out
    }

    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.cols);
        let blocks_per_row = self.cols.div_ceil(self.block);
        let mut y = vec![0.0f32; self.rows];
        for i in 0..self.rows {
            let mut acc = 0.0f32;
            for c in 0..self.cols {
                let code = self.codes[i * self.cols + c];
                let s = self.scales[i * blocks_per_row + c / self.block];
                acc += fp8_to_f32(code) * s * x[c];
            }
            y[i] = self.bias[i] + acc;
        }
        y
    }

    /// FP8 GEMV с AVX2 + FMA: decode Фичи -> vec; acc fma по float-float;
    /// для больших FP8-квантованной упаковки выволяется большой.
    #[cfg(target_arch = "x86_64")]
    pub fn gemv_fast(&self, x: &[f32]) -> Vec<f32> {
        if crate::atom::math::simd::has_avx2() {
            unsafe { self.gemv_fast_avx2(x) }
        } else {
            self.gemv(x)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn gemv_fast_avx2(&self, x: &[f32]) -> Vec<f32> {
        use std::arch::x86_64::*;
        let blocks_per_row = self.cols.div_ceil(self.block);
        let mut y = vec![0.0f32; self.rows];
        for i in 0..self.rows {
            let mut acc_x = _mm256_setzero_ps();
            for c0 in (0..self.cols).step_by(8) {
                let c1 = (c0 + 8).min(self.cols);
                let lanes = {
                    let mut lanes_buf = [0.0f32; 8];
                    for j in 0..(c1 - c0) {
                        let code = self.codes[i * self.cols + c0 + j];
                        let s = self.scales[i * blocks_per_row + (c0 + j) / self.block];
                        lanes_buf[j] = fp8_to_f32(code) * s;
                    }
                    lanes_buf
                };
                let xs = _mm256_loadu_ps(x.as_ptr().add(c0));
                let w = _mm256_loadu_ps(lanes.as_ptr());
                acc_x = _mm256_fmadd_ps(w, xs, acc_x);
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc_x);
            let mut ssum = 0.0f32;
            for j in 0..((self.cols + 7) / 8).min(8) {
                let c1 = ((j + 1) * 8).min(self.cols + ((8 - self.cols % 8) % 8));
                let _ = c1;
                ssum += tmp[j];
            }
            y[i] = self.bias[i] + ssum;
        }
        y
    }

    pub fn memory_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4 + self.bias.len() * 4
    }
}

#[cfg(test)]
mod test_fp8 {
    use super::*;

    #[test]
    fn fp8_encode_decode_unit_roundtrip() {
        let cases = [0.5f32, 1.0, 2.0, 31.999, 0.1, -3.75, 448.0, 0.00390625, 100.0];
        for v in cases {
            let b = f32_to_fp8_e4m3(v);
            let d = fp8_to_f32(b);
            let rel = ((v - d).abs() / v.abs().max(1e-6)).min(1.0);
            assert!(rel < 0.135, "v={v} d={d} rel={rel}");
        }
    }

    #[test]
    fn fp8_quant_gemv_matches_reference() {
        let mut w = vec![0.0f32; 3 * 160];
        let mut s = 777u64;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % 2000) as f32 / 500.0 - 2.0;
        }
        let mut x = vec![0.0f32; 160];
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % 2000) as f32 / 400.0 - 2.5;
        }
        let fp8 = FlashFP8::from_dense(&w, 3, 160, 128, vec![0.0; 3]);
        let y = fp8.gemv(&x);
        // ручной reference: декодирование строк в float64
        for i in 0..3 {
            let wrow = fp8.row_values(i);
            let ref_y: f32 = wrow.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            assert!((y[i] - ref_y).abs() < 1e-3, "row {i}: {} vs {ref_y}", y[i]);
        }
    }
}

    #[cfg(all(test, target_arch = "x86_64"))]
    #[test]
    fn fp8_quant_gemv_fast_matches_scalar() {
        let mut w = vec![0.0f32; 5 * 640];
        let mut s = 777u64;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % 2000) as f32 / 500.0 - 2.0;
        }
        let fp8 = FlashFP8::from_dense(&w, 5, 640, 128, vec![0.0; 5]);
        let mut x = vec![0.0f32; 640];
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) % 2000) as f32 / 400.0 - 2.5;
        }
        let ya = fp8.gemv(&x);
        let yb = fp8.gemv_fast(&x);
        for (a, b) in ya.iter().zip(yb.iter()) {
            assert!((a - b).abs() < 1e-3 * a.abs().max(1.0), "{a} vs {b}");
        }
    }
