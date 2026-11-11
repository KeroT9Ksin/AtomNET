// src/atom/flash/flash.rs
//! FLASH — Fused Linear Algebra Subsystem for Hardware.
//! Собственный BLAS AtomNET + целочисленный Q4×Q8-движок.
//!
//! Уровень 0 (FLASH-CORE): FlashDense — f32 GEMV/GEMV^T, 2-строчный
//!   register blocking, AVX2+FMA, prefetch, выравнивание 64B, DAZ/FTZ.
//! Уровень 1 (FLASH-INT4): FlashCompressed — Q4 веса × Q8 вход,
//!   maddubs/madd, i16-аккумуляция, 8× меньше трафика → победа над BLAS.
//! Уровень 2 (FLASH-ATOM): FlashLayer — живой атомарный слой поверх Dense.
//!
//! Собственные компоненты: XOR cold start, hot/cold роутинг, Micron-гарантии.

use crate::atom::math::simd;
use std::alloc::Layout;
use std::cell::RefCell;
use std::sync::Once;

pub const BLOCK: usize = 256;          // супер-блок квантования (как Micron)
pub const PAD_F32: usize = 32;         // паддинг строк до 32×f32 = 128 байт

static FAST_MATH: Once = Once::new();

/// DAZ/FTZ: денормалы → 0. Один раз на поток-владельца. +5–15%.
fn ensure_fast_math() {
    FAST_MATH.call_once(|| {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            // FTZ (Flush-To-Zero) = 0x8000, DAZ (Denormals-Are-Zero) = 0x0040
            let csr = _mm_getcsr();
            _mm_setcsr(csr | 0x8040);
        }
    });
}

#[inline]
fn pad_cols(cols: usize) -> usize {
    (cols + PAD_F32 - 1) / PAD_F32 * PAD_F32
}

// ═══════════════════════════════════════════════════════════
// Выровненный буфер (64 байта) — требование AVX2/AtomCPU
// ═══════════════════════════════════════════════════════════

pub struct Aligned64<T> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    huge: bool,
}

impl<T> Aligned64<T> {
    pub fn new_zeroed(len: usize) -> Self {
        let size = len.max(1) * std::mem::size_of::<T>();

        // IRON R1: буферы >= 8 МБ пробуем на 2МБ-страницах (TLB)
        if size >= 8 * 1024 * 1024 {
            #[cfg(target_os = "linux")]
            {
                let p = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
                        -1,
                        0,
                    )
                };
                if p != libc::MAP_FAILED {
                    return Self {
                        ptr: std::ptr::NonNull::new(p as *mut T).unwrap(),
                        len,
                        huge: true,
                    };
                }
            }
        }

        let layout = Layout::from_size_align(size, 64).expect("Aligned64: bad layout");
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!raw.is_null(), "Aligned64: alloc failed");
        // Большие упакованные панели живут долго (TLS-кэш): 2 МБ страницы
        // убирают десятки тысяч page fault'ов и TLB-промахов при упаковке.
        if size >= 4 * 1024 * 1024 {
            crate::atom::flash::flash_hw::advise_huge_pages(raw, size);
        }
        Self {
            ptr: std::ptr::NonNull::new(raw as *mut T).unwrap(),
            len,
            huge: false,
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
}

impl<T> Drop for Aligned64<T> {
    fn drop(&mut self) {
        let size = self.len.max(1) * std::mem::size_of::<T>();
        if self.huge {
            unsafe { libc::munmap(self.ptr.as_ptr() as *mut libc::c_void, size) };
        } else {
            let layout = Layout::from_size_align(size, 64).unwrap();
            unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout) };
        }
    }
}

unsafe impl<T> Send for Aligned64<T> {}
unsafe impl<T> Sync for Aligned64<T> {}

// thread-local буфер квантованного входа (без аллокаций в цикле)
thread_local! {
    static QX_BUF: RefCell<Vec<i8>> = RefCell::new(Vec::new());
}

// ═══════════════════════════════════════════════════════════
// SIMD-хелперы
// ═══════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum4(
    a: std::arch::x86_64::__m256,
    b: std::arch::x86_64::__m256,
    c: std::arch::x86_64::__m256,
    d: std::arch::x86_64::__m256,
) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), s);
    t.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum2(a: std::arch::x86_64::__m256, b: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s = _mm256_add_ps(a, b);
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), s);
    t.iter().sum()
}

#[cfg(target_arch = "x86_64")]
unsafe fn axpy_avx2(y: &mut [f32], a: f32, x: &[f32]) {
    use std::arch::x86_64::*;
    let n = y.len().min(x.len());
    let chunks = n / 8;
    let va = _mm256_set1_ps(a);
    for c in 0..chunks {
        let o = c * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(o));
        let vy = _mm256_loadu_ps(y.as_ptr().add(o));
        _mm256_storeu_ps(y.as_mut_ptr().add(o), _mm256_fmadd_ps(vx, va, vy));
    }
    for i in chunks * 8..n {
        y[i] += a * x[i];
    }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn axpy_scalar(y: &mut [f32], a: f32, x: &[f32]) {
    for (yv, xv) in y.iter_mut().zip(x.iter()) {
        *yv += a * *xv;
    }
}

// ═══════════════════════════════════════════════════════════
// FlashDense: f32 GEMV (замена sgemv), 2-строчный blocking
// ═══════════════════════════════════════════════════════════

pub struct FlashDense {
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub data: Aligned64<f32>,
    pub bias: Vec<f32>,
    pub prefetch_ahead: usize,
}

impl FlashDense {
    pub fn new(rows: usize, cols: usize) -> Self {
        let stride = pad_cols(cols);
        Self {
            rows,
            cols,
            stride,
            data: Aligned64::<f32>::new_zeroed(rows * stride),
            bias: vec![0.0f32; rows],
            prefetch_ahead: 8,
        }
    }

    pub fn from_rows(rows_data: &[Vec<f32>], bias: Vec<f32>) -> Option<Self> {
        if rows_data.is_empty() || rows_data.len() != bias.len() {
            return None;
        }
        let cols = rows_data[0].len();
        for r in rows_data {
            if r.len() != cols {
                return None;
            }
        }
        let mut m = Self::new(rows_data.len(), cols);
        m.bias = bias;
        for (i, r) in rows_data.iter().enumerate() {
            m.set_row(i, r);
        }
        Some(m)
    }

    pub fn set_row(&mut self, i: usize, row: &[f32]) {
        let d = self.data.as_mut_slice();
        let o = i * self.stride;
        for j in 0..self.cols.min(row.len()) {
            d[o + j] = row[j];
        }
        for j in self.cols..self.stride {
            d[o + j] = 0.0;
        }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let d = self.data.as_slice();
        let o = i * self.stride;
        &d[o..o + self.cols]
    }

    fn pad_input(&self, x: &[f32]) -> Vec<f32> {
        let mut xp = vec![0.0f32; self.stride];
        for j in 0..self.cols.min(x.len()) {
            xp[j] = x[j];
        }
        xp
    }

    /// Public pad accessor для flash_gemm (GEMV/GEMM-ядра поверх pool'а).
    pub fn pad_input_pub(&self, x: &[f32]) -> Vec<f32> {
        self.pad_input(x)
    }

    /// y = W x
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        ensure_fast_math();
        let xp = self.pad_input(x);
        let mut y = vec![0.0f32; self.rows];

        #[cfg(target_arch = "x86_64")]
        {
            if simd::has_avx2() {
                unsafe { self.gemv_avx2_blocked(&xp, &mut y) };
            } else {
                self.gemv_scalar(&xp, &mut y);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.gemv_scalar(&xp, &mut y);

        y
    }

    /// y = W x + b
    pub fn gemv_bias(&self, x: &[f32]) -> Vec<f32> {
        let mut y = self.gemv(x);
        for i in 0..self.rows {
            y[i] += self.bias[i];
        }
        y
    }

    /// y = W^T g (backward по входу)
    pub fn gemv_t(&self, g: &[f32]) -> Vec<f32> {
        ensure_fast_math();
        let mut acc = vec![0.0f32; self.stride];
        let d = self.data.as_slice();

        #[cfg(target_arch = "x86_64")]
        {
            if simd::has_avx2() {
                unsafe { self.gemv_t_avx2_4row(g, &mut acc) };
            } else {
                for i in 0..self.rows.min(g.len()) {
                    let gi = g[i];
                    if gi == 0.0 {
                        continue;
                    }
                    let r = &d[i * self.stride..(i + 1) * self.stride];
                    axpy_scalar(&mut acc, gi, r);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        for i in 0..self.rows.min(g.len()) {
            let gi = g[i];
            if gi == 0.0 {
                continue;
            }
            let r = &d[i * self.stride..(i + 1) * self.stride];
            axpy_scalar(&mut acc, gi, r);
        }

        acc.truncate(self.cols);
        acc
    }

    /// y = W^T g, 4-строчный AVX2 register blocking.
    ///
    /// Ускорение против по-строчного axpy: аккумулятор читается/пишется
    /// 1 раз на 4 строки вместо 4 раз — трафик памяти по acc падает в 4x,
    /// ядро упирается только в чтение весов (как и должно быть).
    #[cfg(target_arch = "x86_64")]
    unsafe fn gemv_t_avx2_4row(&self, g: &[f32], acc: &mut [f32]) {
        use std::arch::x86_64::*;
        let stride = self.stride;
        let d = self.data.as_slice();
        let rows = self.rows.min(g.len());
        let groups = rows / 4;
        let chunks = stride / 8;

        let mut i = 0usize;
        for _ in 0..groups {
            let vg0 = _mm256_set1_ps(g[i]);
            let vg1 = _mm256_set1_ps(g[i + 1]);
            let vg2 = _mm256_set1_ps(g[i + 2]);
            let vg3 = _mm256_set1_ps(g[i + 3]);
            let p0 = d.as_ptr().add(i * stride);
            let p1 = p0.add(stride);
            let p2 = p1.add(stride);
            let p3 = p2.add(stride);
            let mut accp = acc.as_mut_ptr();
            let base = accp;
            for c in 0..chunks {
                let o = c * 8;
                accp = base.add(o);
                let a = _mm256_loadu_ps(accp);
                let t = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), vg0, a);
                let t = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o)), vg1, t);
                let t = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(o)), vg2, t);
                let t = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(o)), vg3, t);
                _mm256_storeu_ps(accp, t);
            }
            i += 4;
        }

        // Хвост: одиночные строки старым axpy-ядром.
        while i < rows {
            let gi = g[i];
            if gi != 0.0 {
                let r = &d[i * self.stride..(i + 1) * self.stride];
                axpy_avx2(acc, gi, r);
            }
            i += 1;
        }
    }

    fn gemv_scalar(&self, xp: &[f32], y: &mut [f32]) {
        let d = self.data.as_slice();
        for i in 0..self.rows {
            let o = i * self.stride;
            y[i] = dot_scalar(&d[o..o + self.stride], xp);
        }
    }

    /// 2-строчный AVX2 blocking: x грузится 1 раз на 2 строки,
    /// GEMV: 4-строчное AVX2 blocking для выходных строк [lo, hi),
    /// x — одна вектор-строка (длина stride, с паддингом). Выходные индексы
    /// — абсолютные (y индексируется по выходной размерности).
    pub(crate) unsafe fn gemv_avx2_range(&self, xp: &[f32], y: &mut [f32], lo: usize, hi: usize) {
        use std::arch::x86_64::*;
        let stride = self.stride;
        let chunks = stride / 16;
        let base = self.data.as_ptr();
        let px = xp.as_ptr();
        let mut i = lo;

        // 6-строчный блок: шесть независимых потоков весов дают больше
        // memory-level parallelism, чем 4 строки (12 аккумуляторов + 2 x).
        while i + 5 < hi {
            let pa = self.prefetch_ahead;
            if i + pa + 6 < hi {
                for k in pa..pa + 6 {
                    _mm_prefetch(base.add((i + k) * stride) as *const i8, _MM_HINT_T0);
                }
            }

            let p0 = base.add(i * stride);
            let p1 = p0.add(stride);
            let p2 = p1.add(stride);
            let p3 = p2.add(stride);
            let p4 = p3.add(stride);
            let p5 = p4.add(stride);

            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut b0 = _mm256_setzero_ps();
            let mut b1 = _mm256_setzero_ps();
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut d0 = _mm256_setzero_ps();
            let mut d1 = _mm256_setzero_ps();
            let mut e0 = _mm256_setzero_ps();
            let mut e1 = _mm256_setzero_ps();
            let mut f0 = _mm256_setzero_ps();
            let mut f1 = _mm256_setzero_ps();

            for c in 0..chunks {
                let o = c * 16;
                let x0 = _mm256_loadu_ps(px.add(o));
                let x1 = _mm256_loadu_ps(px.add(o + 8));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), x0, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 8)), x1, a1);
                b0 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o)), x0, b0);
                b1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o + 8)), x1, b1);
                c0 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(o)), x0, c0);
                c1 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(o + 8)), x1, c1);
                d0 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(o)), x0, d0);
                d1 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(o + 8)), x1, d1);
                e0 = _mm256_fmadd_ps(_mm256_loadu_ps(p4.add(o)), x0, e0);
                e1 = _mm256_fmadd_ps(_mm256_loadu_ps(p4.add(o + 8)), x1, e1);
                f0 = _mm256_fmadd_ps(_mm256_loadu_ps(p5.add(o)), x0, f0);
                f1 = _mm256_fmadd_ps(_mm256_loadu_ps(p5.add(o + 8)), x1, f1);
            }

            y[i] = hsum2(a0, a1);
            y[i + 1] = hsum2(b0, b1);
            y[i + 2] = hsum2(c0, c1);
            y[i + 3] = hsum2(d0, d1);
            y[i + 4] = hsum2(e0, e1);
            y[i + 5] = hsum2(f0, f1);
            i += 6;
        }

        while i + 3 < hi {
            let pa = self.prefetch_ahead;
            if i + pa + 4 < hi {
                for k in pa..pa + 4 {
                    _mm_prefetch(base.add((i + k) * stride) as *const i8, _MM_HINT_T0);
                }
            }

            let p0 = base.add(i * stride);
            let p1 = p0.add(stride);
            let p2 = p1.add(stride);
            let p3 = p2.add(stride);

            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut b0 = _mm256_setzero_ps();
            let mut b1 = _mm256_setzero_ps();
            let mut c0 = _mm256_setzero_ps();
            let mut c1 = _mm256_setzero_ps();
            let mut d0 = _mm256_setzero_ps();
            let mut d1 = _mm256_setzero_ps();

            for c in 0..chunks {
                let o = c * 16;
                let x0 = _mm256_loadu_ps(px.add(o));
                let x1 = _mm256_loadu_ps(px.add(o + 8));
                let x2 = _mm256_loadu_ps(px.add(o + 16));
                let x3 = _mm256_loadu_ps(px.add(o + 24));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), x0, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 8)), x1, a1);
                b0 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o)), x0, b0);
                b1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o + 8)), x1, b1);
                c0 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(o)), x0, c0);
                c1 = _mm256_fmadd_ps(_mm256_loadu_ps(p2.add(o + 8)), x1, c1);
                d0 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(o)), x0, d0);
                d1 = _mm256_fmadd_ps(_mm256_loadu_ps(p3.add(o + 8)), x1, d1);
            }

            y[i] = hsum2(a0, a1);
            y[i + 1] = hsum2(b0, b1);
            y[i + 2] = hsum2(c0, c1);
            y[i + 3] = hsum2(d0, d1);
            i += 4;
        }

        // Хвост строк < 4 — прежнее 1-строчное ядро (4 аккумулятора × 32 float).
        while i < hi {
            let p0 = base.add(i * stride);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            for c in 0..(stride / 32) {
                let o = c * 32;
                let x0 = _mm256_loadu_ps(px.add(o));
                let x1 = _mm256_loadu_ps(px.add(o + 8));
                let x2 = _mm256_loadu_ps(px.add(o + 16));
                let x3 = _mm256_loadu_ps(px.add(o + 24));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), x0, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 8)), x1, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 16)), x2, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 24)), x3, a3);
            }
            y[i] = hsum4(a0, a1, a2, a3);
            i += 1;
        }
    }

    unsafe fn gemv_avx2_blocked(&self, xp: &[f32], y: &mut [f32]) {
        unsafe { self.gemv_avx2_range(xp, y, 0, self.rows) }
    }

    pub fn bytes(&self) -> usize {
        self.rows * self.cols * 4 + self.bias.len() * 4
    }
    
    pub fn set_prefetch(&mut self, n: usize) {
        self.prefetch_ahead = n.clamp(2, 16);
    }
}

// ═══════════════════════════════════════════════════════════
// FlashCompressed: Q4×Q8 GEMV — оружие против BLAS
// ═══════════════════════════════════════════════════════════

pub struct FlashCompressed {
    pub rows: usize,
    pub cols: usize,
    /// Ниббл-упакованные коды 0..15 (2 на байт), выровнено на 64B
    pub weights_i4: Aligned64<u8>,
    /// s_w: масштаб весов блока
    pub scales: Vec<f32>,
    /// z_w: среднее весов блока
    pub zeros: Vec<f32>,
    /// S_c = Σ(code − 8) на блок
    pub code_sums: Vec<i32>,
    pub bias: Vec<f32>,
    pub row_sigs: Vec<u64>,
    pub cold_start_done: bool,
}

impl FlashCompressed {
    pub fn from_dense(w: &[f32], rows: usize, cols: usize, bias: Vec<f32>) -> Self {
        assert_eq!(w.len(), rows * cols, "FlashCompressed: w size mismatch");
        assert_eq!(cols % BLOCK, 0, "FlashCompressed: cols % 256 != 0");

        let bytes_per_row = cols / 2;
        let blocks = cols / BLOCK;
        let mut weights = Aligned64::<u8>::new_zeroed(rows * bytes_per_row);
        let wb = weights.as_mut_slice();

        let mut scales = Vec::with_capacity(rows * blocks);
        let mut zeros = Vec::with_capacity(rows * blocks);
        let mut code_sums = Vec::with_capacity(rows * blocks);
        let mut row_sigs = Vec::with_capacity(rows);
        let mut codes = [0i8; BLOCK];

        for i in 0..rows {
            let row = &w[i * cols..(i + 1) * cols];
            row_sigs.push(simd::xor_signature(row));

            for b in 0..blocks {
                let chunk = &row[b * BLOCK..(b + 1) * BLOCK];
                let mean: f32 = chunk.iter().sum::<f32>() / BLOCK as f32;
                let m = chunk
                    .iter()
                    .map(|&x| (x - mean).abs())
                    .fold(0.0f32, f32::max);
                let sw = (m / 7.0).max(1e-12);
                let inv = 1.0 / sw;

                let mut sc8: i32 = 0;
                for (j, &x) in chunk.iter().enumerate() {
                    let v = ((x - mean) * inv).round().clamp(-8.0, 7.0) as i8;
                    codes[j] = v;
                    sc8 += v as i32;
                }

                let base = i * bytes_per_row + b * (BLOCK / 2);
                for k in 0..(BLOCK / 2) {
                    let lo = (codes[2 * k] + 8) as u8;
                    let hi = (codes[2 * k + 1] + 8) as u8;
                    wb[base + k] = lo | (hi << 4);
                }

                scales.push(sw);
                zeros.push(mean);
                code_sums.push(sc8);
            }
        }

        Self {
            rows,
            cols,
            weights_i4: weights,
            scales,
            zeros,
            code_sums,
            bias,
            row_sigs,
            cold_start_done: false,
        }
    }

    /// XOR Cold Start: битовой прогрев кэша без тяжёлых вычислений.
    pub fn xor_cold_start(&mut self) {
        if self.cold_start_done {
            return;
        }
        let mut xor_acc: u64 = 0;
        for &b in self.weights_i4.as_slice() {
            xor_acc ^= b as u64;
        }
        let mut sum: f32 = 0.0;
        for (&s, &z) in self.scales.iter().zip(self.zeros.iter()) {
            sum += s * z;
        }
        std::hint::black_box(xor_acc);
        std::hint::black_box(sum);
        self.cold_start_done = true;
    }

    // ── XOR-роутинг (собственные компоненты) ────────────────

    pub fn row_liveness(&self, i: usize) -> u32 {
        self.row_sigs[i].count_ones()
    }

    pub fn hot_rows(&self) -> Vec<usize> {
        self.row_sigs
            .iter()
            .enumerate()
            .filter(|(_, &s)| s.count_ones() > 32)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn dead_rows(&self) -> Vec<usize> {
        self.row_sigs
            .iter()
            .enumerate()
            .filter(|(_, &s)| s.count_ones() < 8)
            .map(|(i, _)| i)
            .collect()
    }

    pub fn row_similarity(&self, i: usize, j: usize) -> u32 {
        64 - (self.row_sigs[i] ^ self.row_sigs[j]).count_ones()
    }

    // ── GEMV: y = W_i4 @ x + b ──────────────────────────────

    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        ensure_fast_math();
        assert_eq!(x.len(), self.cols, "FlashCompressed: x length mismatch");
        let blocks = self.cols / BLOCK;

        QX_BUF.with(|buf| {
            let mut qx = buf.borrow_mut();
            qx.resize(self.cols, 0i8);

            let mut sx = vec![0.0f32; blocks];
            let mut mx = vec![0.0f32; blocks];
            let mut sq = vec![0i32; blocks];

            // Квантуем вход в int8 один раз за вызов
            for b in 0..blocks {
                let xb = &x[b * BLOCK..(b + 1) * BLOCK];
                let qo = &mut qx[b * BLOCK..(b + 1) * BLOCK];
                let mean: f32 = xb.iter().sum::<f32>() / BLOCK as f32;
                let m = xb
                    .iter()
                    .map(|&v| (v - mean).abs())
                    .fold(0.0f32, f32::max);
                let s = (m / 127.0).max(1e-12);
                let inv = 1.0 / s;
                let mut ssum: i32 = 0;
                for (qv, &xv) in qo.iter_mut().zip(xb.iter()) {
                    let q = ((xv - mean) * inv).round().clamp(-127.0, 127.0) as i8;
                    *qv = q;
                    ssum += q as i32;
                }
                sx[b] = s;
                mx[b] = mean;
                sq[b] = ssum;
            }

            let mut y = vec![0.0f32; self.rows];

            #[cfg(target_arch = "x86_64")]
            {
                if simd::has_avx2() {
                    unsafe { self.gemv_q4xq8_avx2(&qx, &sx, &mx, &sq, &mut y) };
                } else {
                    self.gemv_scalar(&qx, &sx, &mx, &sq, &mut y);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            self.gemv_scalar(&qx, &sx, &mx, &sq, &mut y);

            y
        })
    }

    /// Целочисленное AVX2-ядро, 2-строчный blocking:
    /// q грузится 1 раз на 2 строки; maddubs/madd без float-конверсий в цикле.
    #[cfg(target_arch = "x86_64")]
    unsafe fn gemv_q4xq8_avx2(
        &self,
        qx: &[i8],
        sx: &[f32],
        mx: &[f32],
        sq: &[i32],
        y: &mut [f32],
    ) {
        use std::arch::x86_64::*;

        let blocks = self.cols / BLOCK;
        let mask8 = _mm_set1_epi8(0x0F);
        let ones16 = _mm256_set1_epi16(1);
        let bytes_per_row = self.cols / 2;
        let rows = self.rows;
        let rows2 = rows - (rows % 2);
        let wp = self.weights_i4.as_ptr();
        let qp_base = qx.as_ptr();
        let mut i = 0usize;

        while i < rows2 {
            // prefetch 4 строк вперёд
            if i + 12 < rows {
                for k in 8..12 {
                    _mm_prefetch(
                        wp.add((i + k) * bytes_per_row) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            let p0 = wp.add(i * bytes_per_row);
            let p1 = p0.add(bytes_per_row);

            let mut acc_f0 = 0.0f32;
            let mut acc_f1 = 0.0f32;

            for b in 0..blocks {
                let bp0 = p0.add(b * (BLOCK / 2));
                let bp1 = p1.add(b * (BLOCK / 2));
                let qp = qp_base.add(b * BLOCK);

                let mut a0 = _mm256_setzero_si256();
                let mut a1 = _mm256_setzero_si256();

                for c in 0..(BLOCK / 32) {
                    let vq = _mm256_loadu_si256(qp.add(c * 32) as *const __m256i);

                    // строка 0: распаковка нибблов
                    let w0 = _mm_loadu_si128(bp0.add(c * 16) as *const __m128i);
                    let lo0 = _mm_and_si128(w0, mask8);
                    let hi0 = _mm_and_si128(_mm_srli_epi16(w0, 4), mask8);
                    let vc0 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_unpacklo_epi8(lo0, hi0)),
                        _mm_unpackhi_epi8(lo0, hi0),
                        1,
                    );

                    // строка 1: распаковка нибблов
                    let w1 = _mm_loadu_si128(bp1.add(c * 16) as *const __m128i);
                    let lo1 = _mm_and_si128(w1, mask8);
                    let hi1 = _mm_and_si128(_mm_srli_epi16(w1, 4), mask8);
                    let vc1 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_unpacklo_epi8(lo1, hi1)),
                        _mm_unpackhi_epi8(lo1, hi1),
                        1,
                    );

                    // u8(codes) × i8(q) → i16, накопление в i16
                    a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(vc0, vq));
                    a1 = _mm256_add_epi16(a1, _mm256_maddubs_epi16(vc1, vq));
                }

                let m0 = _mm256_madd_epi16(a0, ones16);
                let m1 = _mm256_madd_epi16(a1, ones16);
                let mut t = [0i32; 8];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, m0);
                let d0: i32 = t.iter().sum();
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, m1);
                let d1: i32 = t.iter().sum();

                let base0 = i * blocks + b;
                let base1 = (i + 1) * blocks + b;
                let s = sx[b];
                let m = mx[b];
                let q = sq[b] as f32;

                // float-деквант ТОЛЬКО на границе блока (4 FMA)
                acc_f0 += self.scales[base0] * s * (d0 as f32 - 8.0 * q)
                    + self.scales[base0] * m * self.code_sums[base0] as f32
                    + self.zeros[base0] * s * q
                    + self.zeros[base0] * m * BLOCK as f32;

                acc_f1 += self.scales[base1] * s * (d1 as f32 - 8.0 * q)
                    + self.scales[base1] * m * self.code_sums[base1] as f32
                    + self.zeros[base1] * s * q
                    + self.zeros[base1] * m * BLOCK as f32;
            }

            y[i] = self.bias[i] + acc_f0;
            y[i + 1] = self.bias[i + 1] + acc_f1;
            i += 2;
        }

        // хвост: одиночная строка
        while i < rows {
            let p0 = wp.add(i * bytes_per_row);
            let mut acc_f = 0.0f32;

            for b in 0..blocks {
                let bp0 = p0.add(b * (BLOCK / 2));
                let qp = qp_base.add(b * BLOCK);
                let mut a0 = _mm256_setzero_si256();

                for c in 0..(BLOCK / 32) {
                    let vq = _mm256_loadu_si256(qp.add(c * 32) as *const __m256i);
                    let w0 = _mm_loadu_si128(bp0.add(c * 16) as *const __m128i);
                    let lo0 = _mm_and_si128(w0, mask8);
                    let hi0 = _mm_and_si128(_mm_srli_epi16(w0, 4), mask8);
                    let vc0 = _mm256_inserti128_si256(
                        _mm256_castsi128_si256(_mm_unpacklo_epi8(lo0, hi0)),
                        _mm_unpackhi_epi8(lo0, hi0),
                        1,
                    );
                    a0 = _mm256_add_epi16(a0, _mm256_maddubs_epi16(vc0, vq));
                }

                let m0 = _mm256_madd_epi16(a0, ones16);
                let mut t = [0i32; 8];
                _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, m0);
                let d0: i32 = t.iter().sum();

                let base0 = i * blocks + b;
                let s = sx[b];
                let m = mx[b];
                let q = sq[b] as f32;

                acc_f += self.scales[base0] * s * (d0 as f32 - 8.0 * q)
                    + self.scales[base0] * m * self.code_sums[base0] as f32
                    + self.zeros[base0] * s * q
                    + self.zeros[base0] * m * BLOCK as f32;
            }

            y[i] = self.bias[i] + acc_f;
            i += 1;
        }
    }

    fn gemv_scalar(
        &self,
        qx: &[i8],
        sx: &[f32],
        mx: &[f32],
        sq: &[i32],
        y: &mut [f32],
    ) {
        let blocks = self.cols / BLOCK;
        let bytes_per_row = self.cols / 2;
        let w = self.weights_i4.as_slice();

        for i in 0..self.rows {
            let row = &w[i * bytes_per_row..(i + 1) * bytes_per_row];
            let mut acc_f = 0.0f32;

            for b in 0..blocks {
                let mut dot_code: i32 = 0;
                let bo = b * (BLOCK / 2);
                let qo = b * BLOCK;
                for j in 0..BLOCK {
                    let byte = row[bo + j / 2];
                    let c = if j % 2 == 0 {
                        (byte & 0x0F) as i32
                    } else {
                        (byte >> 4) as i32
                    };
                    dot_code += c * qx[qo + j] as i32;
                }

                let base = i * blocks + b;
                let s = sx[b];
                let m = mx[b];
                let q = sq[b] as f32;

                acc_f += self.scales[base] * s * (dot_code as f32 - 8.0 * q)
                    + self.scales[base] * m * self.code_sums[base] as f32
                    + self.zeros[base] * s * q
                    + self.zeros[base] * m * BLOCK as f32;
            }

            y[i] = self.bias[i] + acc_f;
        }
    }

    pub fn memory_bytes(&self) -> usize {
        self.rows * self.cols / 2
            + (self.scales.len() + self.zeros.len()) * 4
            + self.code_sums.len() * 4
            + self.bias.len() * 4
            + self.row_sigs.len() * 8
    }

    pub fn compression_ratio(&self) -> f32 {
        let dense = (self.rows * self.cols * 4) as f32;
        dense / self.memory_bytes().max(1) as f32
    }
}

// ═══════════════════════════════════════════════════════════
// FlashLayer: живой атомарный слой (FLASH-ATOM)
// Возвращен для совместимости с python.rs и тестами.
// ═══════════════════════════════════════════════════════════

use crate::atom::life::life::{AtomLife, LifeParams, LifeState};

struct FlashRng { state: u64 }
impl FlashRng {
    fn new(seed: u64) -> Self { Self { state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed } } }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.state = x; x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 32;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    for c in 0..chunks {
        let o = c * 32;
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o)), _mm256_loadu_ps(b.as_ptr().add(o)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o + 8)), _mm256_loadu_ps(b.as_ptr().add(o + 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o + 16)), _mm256_loadu_ps(b.as_ptr().add(o + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(o + 24)), _mm256_loadu_ps(b.as_ptr().add(o + 24)), acc3);
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    let mut s: f32 = t.iter().sum();
    for i in chunks * 32..n { s += a[i] * b[i]; }
    s
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 { dot_scalar(a, b) }

pub struct FlashLayer {
    pub dense: FlashDense,
    pub nuclei: Vec<f32>,
    pub radii: Vec<f32>,
    pub sigs: Vec<u64>,
    pub lives: Vec<AtomLife>,
    pub pulse_gain: f32,
}

impl FlashLayer {
    pub fn random(out_features: usize, in_features: usize, std_dev: f32) -> Self {
        let mut rng = FlashRng::new(0xA70C_0DE5_F1A5_4C11);
        let mut rows = Vec::with_capacity(out_features);
        for _ in 0..out_features {
            let row: Vec<f32> = (0..in_features)
                .map(|_| rng.next_f32() * 2.0 * std_dev - std_dev)
                .collect();
            rows.push(row);
        }
        let bias = vec![0.0f32; out_features];
        let dense = FlashDense::from_rows(&rows, bias).expect("FlashLayer::random");
        Self::from_dense(dense)
    }

    pub fn from_dense(dense: FlashDense) -> Self {
        let rows = dense.rows;
        let dim_f = dense.cols.max(1) as f32;
        let mut nuclei = Vec::with_capacity(rows);
        let mut radii = Vec::with_capacity(rows);
        let mut sigs = Vec::with_capacity(rows);
        let mut lives = Vec::with_capacity(rows);

        for i in 0..rows {
            let row = dense.row(i);
            let nucleus: f32 = row.iter().sum::<f32>() / dim_f;
            let mut sumsq = 0.0f32;
            let mut residual = Vec::with_capacity(row.len());
            for v in row.iter() {
                let e = v - nucleus;
                residual.push(e);
                sumsq += e * e;
            }
            let radius = sumsq.sqrt();
            let sig = simd::xor_signature(&residual);
            nuclei.push(nucleus);
            radii.push(radius);
            sigs.push(sig);
            lives.push(AtomLife::new(LifeParams::default()));
        }
        Self { dense, nuclei, radii, sigs, lives, pulse_gain: 1.0 }
    }

    pub fn forward_linear(&self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.dense.cols { return Vec::new(); }
        self.dense.gemv_bias(x)
    }

    pub fn forward_living(&mut self, x: &[f32]) -> Vec<f32> {
        if x.len() != self.dense.cols { return Vec::new(); }
        ensure_fast_math();
        
        let sum_x: f32 = x.iter().sum();
        let xp = self.dense.pad_input(x);
        let rows = self.dense.rows;
        let stride = self.dense.stride;
        let mut y = vec![0.0f32; rows];

        #[cfg(target_arch = "x86_64")]
        let fast = simd::has_avx2();
        #[cfg(not(target_arch = "x86_64"))]
        let fast = false;

        for i in 0..rows {
            let o = i * stride;
            let r = &self.dense.data.as_slice()[o..o + stride];
            let raw = if fast {
                unsafe { dot_avx2(r, &xp) }
            } else {
                dot_scalar(r, &xp)
            };

            let n_i = self.nuclei[i];
            let r_i = self.radii[i];
            let dir_dot = if r_i > 1e-12 {
                ((raw - n_i * sum_x) / r_i).clamp(-1.0, 1.0)
            } else { 0.0 };

            self.lives[i].observe(self.sigs[i], raw.abs(), dir_dot);
            let gate = self.lives[i].gate();
            y[i] = self.dense.bias[i] + gate * raw;
        }

        let out_norm = simd::norm(&y).max(1e-8);
        let input_norm = simd::norm(x).max(1e-6);
        for v in y.iter_mut() { *v *= self.pulse_gain; }
        let new_norm = out_norm * self.pulse_gain;
        let raw_gain = (input_norm / new_norm).clamp(0.25, 8.0);
        self.pulse_gain = (0.7 * self.pulse_gain + 0.3 * raw_gain).clamp(0.1, 16.0);
        y
    }

    pub fn calibrate_gain(&mut self, x: &[f32]) {
        if x.len() != self.dense.cols { return; }
        let y = self.dense.gemv(x);
        let out_norm = simd::norm(&y).max(1e-8);
        let input_norm = simd::norm(x).max(1e-6);
        let raw_gain = (input_norm / (out_norm * self.pulse_gain).max(1e-8)).clamp(0.25, 8.0);
        self.pulse_gain = (0.7 * self.pulse_gain + 0.3 * raw_gain).clamp(0.1, 16.0);
    }

    pub fn backward_input(&self, grad_out: &[f32]) -> Vec<f32> {
        self.dense.gemv_t(grad_out)
    }

    pub fn distribute_grad(&mut self, grad: &[f32], input_norm: f32) {
        if grad.len() != self.dense.rows { return; }
        for i in 0..self.dense.rows {
            let grad_energy = grad[i].abs() * input_norm;
            self.lives[i].observe(self.sigs[i], grad_energy, 0.0);
        }
    }

    pub fn state_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mut dead = 0; let mut cold = 0; let mut born = 0;
        let mut alive = 0; let mut stable = 0; let mut decaying = 0;
        for life in &self.lives {
            match life.state {
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
}
