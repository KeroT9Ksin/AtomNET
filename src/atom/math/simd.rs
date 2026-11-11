// src/atom/math/simd.rs
//! SIMD-движок v3 «фиксированный чанк».
//!
//! Архитектура редукций:
//! - Все массивы делятся на чанки по CHUNK флоатов (256 К = 1 МБ на ось,
//!   2 МБ на пару). Чанк живёт в кэше целиком → ядро ts16 всегда работает
//!   в кэш-плане, даже когда массив стримится из DRAM.
//! - ts16 = два полупотока × 16 цепочек FMA, prefetch 512B. Пока оси
//!   кэш-резидентны, 4 потока загрузки занимают все порты чтения:
//!   50–57 ГБ/с на ядро.
//! - Частичные результаты лежат ПО НОМЕРУ ЧАНКА и суммируются по индексам
//!   → результат бит-в-бит одинаков при любом числе потоков Veron
//!   (потоки лишь меняют, кто какой чанк взял; сумма фиксированная).
//! - Own pool (vpool): постоянные потоки, спин-синхронизация, ноль
//!   аллокаций/спавна на fork/join (spawn-на-вызов стоит ~0.3–0.5 мс,
//!   fork — единицы микросекунд).
//!
//! Однопоточный режим по умолчанию (veron::parallel_threads == 1).

/// Виртуальный ISA — единый гейт AVX2/FMA для всех обрабатывающих
/// элементов (см. crate::atom::isa). Детект CPUID один раз на процесс.
#[inline]
pub fn has_avx2() -> bool {
    crate::atom::isa::has_avx2()
}

/// Размер фиксированного чанка: 256 К флоатов (1 МБ на ось).
pub const CHUNK: usize = 1 << 18;
/// Veron-fork включается с ≥ 2 чанков работы.
const FORK_LIMIT: usize = CHUNK;

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ═══════════════════════════════════════════════════════════
// ts-ядра: два полупотока × 16 цепочек FMA + prefetch 512B
// ═══════════════════════════════════════════════════════════

/// Свертка 16 векторов-аккумуляторов в один.
#[cfg(target_arch = "x86_64")]
unsafe fn hsum16(s: &[std::arch::x86_64::__m256; 16]) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let mut a0 = s[0];
    let mut a1 = s[8];
    for i in 1..8 {
        a0 = _mm256_add_ps(a0, s[i]);
        a1 = _mm256_add_ps(a1, s[i + 8]);
    }
    _mm256_add_ps(a0, a1)
}

/// Горизонтальная сумма AVX-вектора.
#[cfg(target_arch = "x86_64")]
unsafe fn scalar16(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    t.iter().sum()
}

/// dot: Σa·b (ts-план; вызывается на данных ≤ CHUNK).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_ts16(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let h = n / 2;
    let step = 64usize;
    let main = (h / step) * step;
    let mut s = [_mm256_setzero_ps(); 16];
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut o = 0usize;
    while o < main {
        let o2 = h + o;
        _mm_prefetch(pa.add(o + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pb.add(o + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pa.add(o2 + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pb.add(o2 + 512) as *const i8, _MM_HINT_T0);
        for i in 0..8 {
            s[i] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(o + i * 8)),
                _mm256_loadu_ps(pb.add(o + i * 8)),
                s[i]);
            s[i + 8] = _mm256_fmadd_ps(
                _mm256_loadu_ps(pa.add(o2 + i * 8)),
                _mm256_loadu_ps(pb.add(o2 + i * 8)),
                s[i + 8]);
        }
        o += step;
    }
    let mut acc = scalar16(hsum16(&s));
    for i in main..h { acc += a[i] * b[i] + a[h + i] * b[h + i]; }
    if h * 2 < n { acc += a[h * 2] * b[h * 2]; }
    acc
}

/// Σv² (ts-план).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sumsq_ts16(v: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = v.len();
    let h = n / 2;
    let step = 64usize;
    let main = (h / step) * step;
    let mut s = [_mm256_setzero_ps(); 16];
    let pv = v.as_ptr();
    let mut o = 0usize;
    while o < main {
        let o2 = h + o;
        _mm_prefetch(pv.add(o + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pv.add(o2 + 512) as *const i8, _MM_HINT_T0);
        for i in 0..8 {
            let x0 = _mm256_loadu_ps(pv.add(o + i * 8));
            let x1 = _mm256_loadu_ps(pv.add(o2 + i * 8));
            s[i] = _mm256_fmadd_ps(x0, x0, s[i]);
            s[i + 8] = _mm256_fmadd_ps(x1, x1, s[i + 8]);
        }
        o += step;
    }
    let mut acc = scalar16(hsum16(&s));
    for i in main..h { acc += v[i] * v[i] + v[h + i] * v[h + i]; }
    if h * 2 < n { acc += v[h * 2] * v[h * 2]; }
    acc
}

/// Σ(a−b)² (ts-план).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dsumsq_ts16(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let h = n / 2;
    let step = 64usize;
    let main = (h / step) * step;
    let mut s = [_mm256_setzero_ps(); 16];
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut o = 0usize;
    while o < main {
        let o2 = h + o;
        _mm_prefetch(pa.add(o + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pb.add(o + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pa.add(o2 + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pb.add(o2 + 512) as *const i8, _MM_HINT_T0);
        for i in 0..8 {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(o + i * 8)), _mm256_loadu_ps(pb.add(o + i * 8)));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(o2 + i * 8)), _mm256_loadu_ps(pb.add(o2 + i * 8)));
            s[i] = _mm256_fmadd_ps(d0, d0, s[i]);
            s[i + 8] = _mm256_fmadd_ps(d1, d1, s[i + 8]);
        }
        o += step;
    }
    let mut acc = scalar16(hsum16(&s));
    for i in main..h {
        let d0 = a[i] - b[i];
        let d1 = a[h + i] - b[h + i];
        acc += d0 * d0 + d1 * d1;
    }
    if h * 2 < n { let d = a[h * 2] - b[h * 2]; acc += d * d; }
    acc
}

/// (Σv, Σv²) одним проходом (ts-план).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sums2_ts16(v: &[f32]) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = v.len();
    let h = n / 2;
    let step = 64usize;
    let main = (h / step) * step;
    let mut s = [_mm256_setzero_ps(); 16];
    let mut q = [_mm256_setzero_ps(); 16];
    let pv = v.as_ptr();
    let mut o = 0usize;
    while o < main {
        let o2 = h + o;
        _mm_prefetch(pv.add(o + 512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(pv.add(o2 + 512) as *const i8, _MM_HINT_T0);
        for i in 0..8 {
            let x0 = _mm256_loadu_ps(pv.add(o + i * 8));
            let x1 = _mm256_loadu_ps(pv.add(o2 + i * 8));
            s[i] = _mm256_add_ps(s[i], x0);
            s[i + 8] = _mm256_add_ps(s[i + 8], x1);
            q[i] = _mm256_fmadd_ps(x0, x0, q[i]);
            q[i + 8] = _mm256_fmadd_ps(x1, x1, q[i + 8]);
        }
        o += step;
    }
    let (mut acc_s, mut acc_q) = (scalar16(hsum16(&s)), scalar16(hsum16(&q)));
    for i in main..h {
        acc_s += v[i] + v[h + i];
        acc_q += v[i] * v[i] + v[h + i] * v[h + i];
    }
    if h * 2 < n {
        acc_s += v[h * 2];
        acc_q += v[h * 2] * v[h * 2];
    }
    (acc_s, acc_q)
}

// ═══════════════════════════════════════════════════════════
// Публичные редукции (чанки + фиксированная сумма по индексам)
// ═══════════════════════════════════════════════════════════

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    if !has_avx2() {
        return dot_scalar(a, b);
    }
    let threads = crate::atom::veron::parallel_threads();
    let parts = if threads > 1 && n >= FORK_LIMIT { threads } else { 1 };
    thread_reduce(0, a, b, parts).0.into_iter().sum()
}

/// norm: sqrt(Σv²) — sqrt один раз на весь массив (частичные без sqrt).
pub fn norm(v: &[f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if !has_avx2() {
        return dot_scalar(v, v).sqrt();
    }
    let threads = crate::atom::veron::parallel_threads();
    let parts = if threads > 1 && n >= FORK_LIMIT { threads } else { 1 };
    thread_reduce(3, v, &[], parts).0.into_iter().sum::<f32>().sqrt()
}

/// l2: sqrt(Σ(a−b)²).
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    if !has_avx2() {
        return a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f32>().sqrt();
    }
    let threads = crate::atom::veron::parallel_threads();
    let parts = if threads > 1 && n >= FORK_LIMIT { threads } else { 1 };
    thread_reduce(2, a, &b[..n], parts).0.into_iter().sum::<f32>().sqrt()
}

/// MSE одним проходом: mean((a-b)²) без временного массива.
pub fn mse(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    if !has_avx2() {
        let s: f32 = a[..n].iter().zip(&b[..n]).map(|(x, y)| (x - y) * (x - y)).sum();
        return s / n as f32;
    }
    let threads = crate::atom::veron::parallel_threads();
    let parts = if threads > 1 && n >= FORK_LIMIT { threads } else { 1 };
    thread_reduce(2, a, &b[..n], parts).0.into_iter().sum::<f32>() / n as f32
}

/// (mean, var) одним проходом (numpy: 2-3 полных прохода + аллокации).
pub fn mean_var(v: &[f32]) -> (f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let n = v.len();
    if !has_avx2() {
        let mean = v.iter().sum::<f32>() / n as f32;
        let var = v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        return (mean, var);
    }
    let threads = crate::atom::veron::parallel_threads();
    let parts = if threads > 1 && n >= FORK_LIMIT { threads } else { 1 };
    let (p, q) = thread_reduce(7, v, &[], parts);
    let mean = p.into_iter().sum::<f32>() / n as f32;
    let var = (q.into_iter().sum::<f32>() / n as f32 - mean * mean).max(0.0);
    (mean, var)
}

/// XOR-подпись 64-бит: знак каждого из первых 64 элементов (>0 = 1).
/// Основа связи между Rineta/Micron/Enigma — проверка целостности состояния.
pub fn xor_signature(data: &[f32]) -> u64 {
    if data.len() >= 64 {
        if has_avx2() {
            return unsafe { xor_signature_avx2(data) };
        }
    }
    // Скалярная версия (fallback)
    data.iter().take(64).enumerate().fold(0u64, |acc, (i, &v)| {
        if v > 0.0 { acc | (1u64 << i) } else { acc }
    })
}

/// XOR-складка знаков всего массива: XOR 64-битных подписей чанков.
/// Позволяет проверить целостность без учёта порядка элементов (XOR — коммутативен).
/// Это основа XOR cold start и hashing в AtomNET (Enigma/Micron/Rineta).
pub fn xor_fold(data: &[f32]) -> u64 {
    // Чанки по 64 элементам: каждый чанк даёт 64-бит подпись, все XOR'ятся.
    let mut acc: u64 = 0;
    for chunk in data.chunks(64) {
        acc ^= xor_signature(chunk);
    }
    acc
}

/// Быстрая проверка равенства двух массивов через XOR-складку (без скалярного цикла).
/// True iff xor_fold(a) == xor_fold(b) для простых кейсов. O(n) в SIMD.
/// Это bitwise-операция, полностью заменяющая поэлементное сравнение в горячих путях.
pub fn xor_eq_check(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && xor_fold(a) == xor_fold(b)
}

/// Упаковка знаков в биты: 8 f32 → 1 байт (8x сжатие, чисто битовая операция).
/// Бит = 1 iff элемент > 0. Полезно для разреженности, XOR-подписей и сжатия.
/// AVX2-путь: movemask за 8 элементов + XOR аккумуляторов, без скалярного цикла.
pub fn pack_signs(data: &[f32]) -> Vec<u8> {
    let n = data.len();
    let mut out = vec![0u8; (n + 7) / 8];
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && n >= 8 {
        unsafe {
            use std::arch::x86_64::*;
            let vzero = _mm256_setzero_ps();
            let chunks = n / 8;
            for c in 0..chunks {
                let v = _mm256_loadu_ps(data.as_ptr().add(c * 8));
                out[c] = _mm256_movemask_ps(_mm256_cmp_ps(v, vzero, _CMP_GT_OS)) as u8;
            }
        }
        // Скалярный хвост (< 8 элементов)
        for i in (n / 8) * 8..n {
            if data[i] > 0.0 {
                out[i / 8] |= 1u8 << (i % 8);
            }
        }
        return out;
    }
    for (i, &v) in data.iter().enumerate() {
        if v > 0.0 {
            out[i / 8] |= 1u8 << (i % 8);
        }
    }
    out
}

/// popcount упакованных знаков: число положительных элементов (битовый подсчёт).
pub fn packed_popcount(packed: &[u8]) -> u32 {
    packed.iter().map(|b| b.count_ones()).sum()
}

/// Hamming distance между двумя XOR-подписями
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Изменение состояния: XOR текущей и предыдущей подписи.
/// 0 = нет изменений, 64 = полностью изменилось.
pub fn state_change(prev: u64, curr: u64) -> u32 {
    hamming_distance(prev, curr)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_signature_avx2(data: &[f32]) -> u64 {
    use std::arch::x86_64::*;
    let n = data.len().min(64);
    let chunks = n / 8;
    let vzero = _mm256_setzero_ps();
    let mut acc: u64 = 0;
    for c in 0..chunks {
        let v = _mm256_loadu_ps(data.as_ptr().add(c * 8));
        let m = _mm256_movemask_ps(_mm256_cmp_ps(v, vzero, _CMP_GT_OS)) as u32;
        acc |= (m as u64) << (c * 8);
    }
    // Скалярный хвост (если n < 64)
    for i in chunks * 8..n {
        if data[i] > 0.0 { acc |= 1u64 << i; }
    }
    acc
}

// ═══════════════════════════════════════════════════════════
// Слитые векторные функции «без временных массивов»: numpy на тех же
// операциях делает 2-3 прохода с аллокациями — здесь 1 проход в регистрах.
// ═══════════════════════════════════════════════════════════

/// y += k * x. Горячий путь — сгенерированное DynASM-ядро (AVX2, хвост
/// vmaskmovps), затем AVX2-intrinsics, затем скаляр.
pub fn axpy(y: &mut [f32], k: f32, x: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = y.len().min(x.len());
        if n > 0 {
            let sc = [k, 0.0f32];
            unsafe { (m.axpy)(y.as_mut_ptr(), x.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { axpy_avx2(y, k, x) };
        return;
    }
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += k * *xv;
    }
}

/// p = b1*p + ob*g (FMA; ob = 1-b1) — идёт через то же axpby-ядро.
pub fn momentum_update(p: &mut [f32], g: &[f32], b1: f32) {
    let ob = 1.0 - b1;
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = p.len().min(g.len());
        if n > 0 {
            let sc = [b1, ob];
            unsafe { (m.axpby)(p.as_mut_ptr(), g.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { momentum_avx2(p, g, b1, ob) };
        return;
    }
    for (pv, gv) in p.iter_mut().zip(g) {
        *pv = b1 * *pv + ob * *gv;
    }
}

#[inline]
pub fn scale(y: &mut [f32], k: f32) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        if !y.is_empty() {
            let sc = [k, 0.0f32];
            unsafe { (m.scale)(y.as_mut_ptr(), y.as_ptr(), y.len(), sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { scale_avx2(y, k) };
        return;
    }
    for v in y.iter_mut() { *v *= k; }
}

#[inline]
pub fn scale_to(y: &mut [f32], x: &[f32], k: f32) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = y.len().min(x.len());
        if n > 0 {
            let sc = [k, 0.0f32];
            unsafe { (m.scale_to)(y.as_mut_ptr(), x.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { scale_to_avx2(y, x, k) };
        return;
    }
    for (yv, xv) in y.iter_mut().zip(x) { *yv = *xv * k; }
}

/// y = a*y + b*x (слитое, FMA)
#[inline]
pub fn axpby(y: &mut [f32], a: f32, x: &[f32], b: f32) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = y.len().min(x.len());
        if n > 0 {
            let sc = [a, b];
            unsafe { (m.axpby)(y.as_mut_ptr(), x.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { axpby_avx2(y, a, x, b) };
        return;
    }
    for (yv, xv) in y.iter_mut().zip(x) { *yv = a * *yv + b * *xv; }
}

/// y += x (элиментарное, AVX2: 1 проход, без аллокаций — numpy делает
/// 2 прохода + временный массив).
#[inline]
pub fn add_elem(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = y.len().min(x.len());
        if n > 0 {
            let sc = [0.0f32; 2];
            unsafe { (m.add_elem)(y.as_mut_ptr(), x.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { add_elem_avx2(y, x) };
        return;
    }
    for (yv, xv) in y.iter_mut().zip(x) { *yv += *xv; }
}

/// y *= x (элиментарное, AVX2).
#[inline]
pub fn mul_elem(y: &mut [f32], x: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = y.len().min(x.len());
        if n > 0 {
            let sc = [0.0f32; 2];
            unsafe { (m.mul_elem)(y.as_mut_ptr(), x.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { mul_elem_avx2(y, x) };
        return;
    }
    for (yv, xv) in y.iter_mut().zip(x) { *yv *= *xv; }
}

/// y = max(0, y) in-place (AVX2).
#[inline]
pub fn relu(y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        if !y.is_empty() {
            let sc = [0.0f32; 2];
            unsafe { (m.relu)(y.as_mut_ptr(), y.as_ptr(), y.len(), sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { relu_avx2(y) };
        return;
    }
    for v in y.iter_mut() { *v = if *v > 0.0 { *v } else { 0.0 }; }
}

/// y = silu(y) = y / (1 + exp(-y)) in-place, точный float32 (std exp).
/// Для нейросетей точность важнее скорости: весь стек читает/считает
/// строго f32, без аппроксимаций и целочисленных квантов в этом пути.
#[inline]
pub fn silu(y: &mut [f32]) {
    silu_scalar(y);
}

/// Быстрый AVX2-вариант SiLU (fast-exp, погрешность ~2e-4) — только
/// для инференса, где допустима аппроксимация. Для обучения/точности
/// используйте [`silu_scalar`] / [`silu`].
#[inline]
pub fn silu_fast(y: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // Горячий путь: полностью развёрнутое DynASM-ядро (2×8 float за
        // итерацию, константы в ymm, хвост через vmaskmovps).
        if let Some(m) = crate::atom::flash::jit::jit_math() {
            if !y.is_empty() {
                unsafe { (m.silu)(y.as_mut_ptr(), std::ptr::null(), 1, y.len()) };
            }
            return;
        }
        unsafe { silu_avx2(y) };
        return;
    }
    silu_scalar(y);
}

/// p = softmax(x) in-place. ASM-ядро (max/exp/sum/normalize) при
/// доступном JIT; иначе точный скалярный путь.
#[inline]
pub fn softmax(y: &mut [f32]) {
    let n = y.len();
    if n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let p = y.as_mut_ptr();
        unsafe { (m.softmax)(p, p, 1, n, std::ptr::null_mut()) };
        return;
    }
    let max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in y.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-12);
    for v in y.iter_mut() {
        *v *= inv;
    }
}

/// y = softmax(x) (x и y могут совпадать). ASM-ядро или скаляр.
#[inline]
pub fn softmax_into(x: &[f32], y: &mut [f32]) {
    let n = x.len();
    assert!(y.len() >= n, "softmax_into: y.len() < x.len()");
    if n == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        unsafe {
            (m.softmax)(
                x.as_ptr(),
                y.as_mut_ptr(),
                1,
                n,
                std::ptr::null_mut(),
            )
        };
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (dst, &src) in y.iter_mut().zip(x) {
        let e = (src - max).exp();
        *dst = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-12);
    for dst in y.iter_mut().take(n) {
        *dst *= inv;
    }
}

/// p = softmax(y) in-place + (max, sum) для устойчивого CE.
#[inline]
pub fn softmax_stats(y: &mut [f32]) -> (f32, f32) {
    let n = y.len();
    if n == 0 {
        return (f32::NEG_INFINITY, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let mut st = [0.0f32; 2];
        let p = y.as_mut_ptr();
        unsafe { (m.softmax)(p, p, 1, n, st.as_mut_ptr()) };
        return (st[0], st[1]);
    }
    let max = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in y.iter_mut() {
        let e = (*v - max).exp();
        *v = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-12);
    for v in y.iter_mut() {
        *v *= inv;
    }
    (max, sum)
}

/// y = silu(y) точный (scalar exp).
#[inline]
pub fn silu_scalar(y: &mut [f32]) {
    for v in y.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// C[row*cols+col] = silu(C + bias[col]) in-place (rows×cols).
/// AVX2: 8-wide чанки по строкам (bias на колонку, без modulo на элемент),
/// fast-exp; хвост — скалярный точный. Пустой bias = 0.
pub fn fused_bias_silu(c: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // Полный bias: ASM-ядро без скалярного хвоста. Короткий bias
        // (редкий случай) остаётся на intrinsics-пути с нулевым дополнением.
        if bias.len() >= cols {
            if let Some(m) = crate::atom::flash::jit::jit_math() {
                if c.len() >= rows * cols {
                    unsafe { (m.silu_bias)(c.as_mut_ptr(), bias.as_ptr(), rows, cols) };
                    return;
                }
            }
        }
        unsafe { fused_bias_silu_avx2(c, bias, rows, cols) };
        return;
    }
    for r in 0..rows {
        let base = r * cols;
        for col in 0..cols {
            let v = c[base + col] + (if bias.len() > col { bias[col] } else { 0.0 });
            c[base + col] = v / (1.0 + (-v).exp());
        }
    }
}

/// Общий «32 флоата на итерацию» проход с одним входным массивом.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn for_each8(
    y: &mut [f32],
    op: impl Fn(usize, std::arch::x86_64::__m256) -> std::arch::x86_64::__m256,
    scalar: impl Fn(usize, f32) -> f32,
) {
    use std::arch::x86_64::*;
    let n = y.len();
    let chunks = n / 32;
    let py = y.as_mut_ptr();
    for c in 0..chunks {
        let o = c * 32;
        for i in 0..4 {
            let p = o + i * 8;
            let r = op(i, _mm256_loadu_ps(py.add(p)));
            _mm256_storeu_ps(py.add(p), r);
        }
    }
    for i in chunks * 32..n {
        *py.add(i) = scalar(i, *py.add(i));
    }
}

/// Общий «32 флоата на итерацию» проход: читает x, пишет y.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn for_each_load8(
    y: &mut [f32],
    x: &[f32],
    op: impl Fn(std::arch::x86_64::__m256, std::arch::x86_64::__m256) -> std::arch::x86_64::__m256,
    scalar: impl Fn(f32, f32) -> f32,
) {
    use std::arch::x86_64::*;
    let n = y.len().min(x.len());
    let chunks = n / 32;
    let py = y.as_mut_ptr();
    let px = x.as_ptr();
    for c in 0..chunks {
        let o = c * 32;
        for i in 0..4 {
            let p = o + i * 8;
            let r = op(_mm256_loadu_ps(px.add(p)), _mm256_loadu_ps(py.add(p)));
            _mm256_storeu_ps(py.add(p), r);
        }
    }
    for i in chunks * 32..n {
        *py.add(i) = scalar(*px.add(i), *py.add(i));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_avx2(y: &mut [f32], k: f32) {
    use std::arch::x86_64::*;
    let vk = _mm256_set1_ps(k);
    for_each8(y, |_i, y| _mm256_mul_ps(y, vk), |_i, v| v * k);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_to_avx2(y: &mut [f32], x: &[f32], k: f32) {
    use std::arch::x86_64::*;
    let vk = _mm256_set1_ps(k);
    for_each_load8(y, x, |xv, _yv| _mm256_mul_ps(xv, vk), |x, _y| x * k);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpby_avx2(y: &mut [f32], a: f32, x: &[f32], b: f32) {
    use std::arch::x86_64::*;
    let va = _mm256_set1_ps(a);
    let vb = _mm256_set1_ps(b);
    for_each_load8(y, x,
        |xv, yv| _mm256_fmadd_ps(va, yv, _mm256_mul_ps(vb, xv)),
        |x, y| a * y + b * x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_elem_avx2(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    for_each_load8(y, x, |xv, yv| _mm256_add_ps(yv, xv), |x, y| y + x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_elem_avx2(y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;
    for_each_load8(y, x, |xv, yv| _mm256_mul_ps(yv, xv), |x, y| y * x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(y: &mut [f32]) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    for_each8(y, |_i, v| _mm256_max_ps(v, zero), |_i, v| if v > 0.0 { v } else { 0.0 });
}

/// Быстрый AVX2 exp (Cephes-стиль, погрешность ~1.5e-4): clamp,
/// log2e-масштаб, 2^23-trick, полином степени 5.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_ps_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let cephes_exp_clamp = _mm256_set1_ps(88.3762626647949f32);
    let log2e = _mm256_set1_ps(1.44269504088896341f32);

    let x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f32)), cephes_exp_clamp);
    let x = _mm256_mul_ps(x, log2e);
    // fx = round-to-nearest (MXCSR), z = x - fx ∈ [-0.5, 0.5] (log2-пространство).
    // 2^z = e^{z·ln2}: u = z·ln2, полином аппроксимирует e^u.
    let fx_i = _mm256_cvtps_epi32(x);
    let fx = _mm256_cvtepi32_ps(fx_i);
    let z = _mm256_sub_ps(x, fx);
    let u = _mm256_mul_ps(z, _mm256_set1_ps(0.69314718056f32));
    // Horner для e^u: Q(u) = ((((p0*u+p1)*u+p2)*u+p3)*u+p4)*u+p5, mant = 1 + u*Q(u).
    // Коэффициенты Тейлора Q = 1 + u/2 + u²/6 + u³/24 + u⁴/120 + u⁵/720.
    let mut p = _mm256_set1_ps(0.00138888889f32); // 1/720
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(0.00833333333f32)); // 1/120
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(0.0416666667f32)); // 1/24
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(0.166666671f32)); // 1/6
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(0.5));
    p = _mm256_fmadd_ps(p, u, _mm256_set1_ps(1.0));
    let p = _mm256_fmadd_ps(u, p, _mm256_set1_ps(1.0));
    // 2^fx: (fx + 127) << 23 как float-биты.
    let ee_bits = _mm256_slli_epi32(_mm256_add_epi32(fx_i, _mm256_set1_epi32(127)), 23);
    _mm256_mul_ps(p, _mm256_castsi256_ps(ee_bits))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(y: &mut [f32]) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    for_each8(y, |_i, v| {
        let neg = _mm256_mul_ps(v, _mm256_set1_ps(-1.0));
        let e = exp_ps_avx2(neg);
        let s = _mm256_div_ps(v, _mm256_add_ps(one, e));
        s
    }, |_i, v| v / (1.0 + (-v).exp()));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_bias_silu_avx2(c: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg1 = _mm256_set1_ps(-1.0);
    for r in 0..rows {
        let base = r * cols;
        let mut col = 0usize;
        while col + 8 <= cols {
            let p = base + col;
            let mut v = _mm256_loadu_ps(c.as_ptr().add(p));
            let b = if bias.len() > col + 7 {
                _mm256_loadu_ps(bias.as_ptr().add(col))
            } else if !bias.is_empty() {
                let mut t = [0.0f32; 8];
                for i in 0..8 {
                    t[i] = if bias.len() > col + i { bias[col + i] } else { 0.0 };
                }
                _mm256_loadu_ps(t.as_ptr())
            } else {
                zero
            };
            v = _mm256_add_ps(v, b);
            let e = exp_ps_avx2(_mm256_mul_ps(v, neg1));
            v = _mm256_div_ps(v, _mm256_add_ps(one, e));
            _mm256_storeu_ps(c.as_mut_ptr().add(p), v);
            col += 8;
        }
        for cc in col..cols {
            let v = c[base + cc] + (if bias.len() > cc { bias[cc] } else { 0.0 });
            c[base + cc] = v / (1.0 + (-v).exp());
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(y: &mut [f32], k: f32, x: &[f32]) {
    use std::arch::x86_64::*;
    let vk = _mm256_set1_ps(k);
    for_each_load8(y, x, |xv, yv| _mm256_fmadd_ps(vk, xv, yv), |x, y| y + k * x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn momentum_avx2(p: &mut [f32], g: &[f32], b1: f32, ob: f32) {
    use std::arch::x86_64::*;
    let vb = _mm256_set1_ps(b1);
    let vo = _mm256_set1_ps(ob);
    for_each_load8(p, g, |gv, pv| _mm256_fmadd_ps(vo, gv, _mm256_mul_ps(vb, pv)),
                    |g, p| b1 * p + ob * g);
}

/// out = x·scale + shift: один FMA-проход, без временных массивов.
/// Горячий путь — DynASM-ядро, затем AVX2-intrinsics, затем скаляр.
pub fn affine_into(x: &[f32], scale: f32, shift: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let n = out.len().min(x.len());
        if n > 0 {
            let sc = [scale, shift];
            unsafe { (m.affine)(out.as_mut_ptr(), x.as_ptr(), n, sc.as_ptr()) };
        }
        return;
    }
    if has_avx2() {
        unsafe { affine_avx2(x, scale, shift, out) };
        return;
    }
    for (o, &xv) in out.iter_mut().zip(x) {
        *o = xv * scale + shift;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn affine_avx2(x: &[f32], scale: f32, shift: f32, out: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len().min(out.len());
    let chunks = n / 64;
    let vk = _mm256_set1_ps(scale);
    let vb = _mm256_set1_ps(shift);
    let px = x.as_ptr();
    let po = out.as_mut_ptr();
    for c in 0..chunks {
        let o = c * 64;
        _mm256_storeu_ps(po.add(o), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o)), vb));
        _mm256_storeu_ps(po.add(o + 8), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 8)), vb));
        _mm256_storeu_ps(po.add(o + 16), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 16)), vb));
        _mm256_storeu_ps(po.add(o + 24), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 24)), vb));
        _mm256_storeu_ps(po.add(o + 32), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 32)), vb));
        _mm256_storeu_ps(po.add(o + 40), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 40)), vb));
        _mm256_storeu_ps(po.add(o + 48), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 48)), vb));
        _mm256_storeu_ps(po.add(o + 56), _mm256_fmadd_ps(vk, _mm256_loadu_ps(px.add(o + 56)), vb));
    }
    for i in chunks * 64..n {
        out[i] = x[i] * scale + shift;
    }
}

// ═══════════════════════════════════════════════════════════
// Rineta: i16-квантование памяти и слитый pass1
// ═══════════════════════════════════════════════════════════

/// Слитый проход 1 для rin[]: декод i16→f32, p_new = b1*p_old + ob*g,
/// запись p_new в out; возврат (g2, p_old2, dot_gp, p_new2, dot_pg).
pub fn rin_pass1_block(
    g: &[f32],
    q: &[u16],
    scale: f32,
    b1: f32,
    out: &mut [f32],
) -> (f64, f64, f64, f64, f64) {
    if has_avx2() {
        return unsafe { rin_pass1_avx2(g, q, scale, b1, 1.0 - b1, out) };
    }
    let (mut a0, mut a1, mut a2, mut a3, mut a4) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    let n = g.len().min(q.len()).min(out.len());
    for i in 0..n {
        let gv = g[i] as f64;
        let p_old = (q[i] as i16) as f64 * scale as f64;
        let p_new = b1 as f64 * p_old + (1.0 - b1) as f64 * gv;
        out[i] = p_new as f32;
        a0 += gv * gv;
        a1 += p_old * p_old;
        a2 += gv * p_old;
        a3 += p_new * p_new;
        a4 += p_new * gv;
    }
    (a0, a1, a2, a3, a4)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rin_pass1_avx2(
    g: &[f32],
    q: &[u16],
    scale: f32,
    b1: f32,
    ob: f32,
    out: &mut [f32],
) -> (f64, f64, f64, f64, f64) {
    use std::arch::x86_64::*;
    let n = g.len().min(q.len()).min(out.len());
    let chunks = n / 8;
    let vb = _mm256_set1_ps(b1);
    let vo = _mm256_set1_ps(ob);
    let vs = _mm256_set1_ps(scale);
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut a4 = _mm256_setzero_ps();
    for c in 0..chunks {
        let o = c * 8;
        let vg = _mm256_loadu_ps(g.as_ptr().add(o));
        let qi = _mm_loadu_si128(q.as_ptr().add(o) as *const __m128i);
        let vp = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(qi)), vs);
        let vn = _mm256_fmadd_ps(vo, vg, _mm256_mul_ps(vb, vp));
        _mm256_storeu_ps(out.as_mut_ptr().add(o), vn);
        a0 = _mm256_fmadd_ps(vg, vg, a0);
        a1 = _mm256_fmadd_ps(vp, vp, a1);
        a2 = _mm256_fmadd_ps(vg, vp, a2);
        a3 = _mm256_fmadd_ps(vn, vn, a3);
        a4 = _mm256_fmadd_ps(vn, vg, a4);
    }
    let mut t = [0.0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), a0);
    let mut s0: f64 = t.iter().map(|&v| v as f64).sum();
    _mm256_storeu_ps(t.as_mut_ptr(), a1);
    let mut s1: f64 = t.iter().map(|&v| v as f64).sum();
    _mm256_storeu_ps(t.as_mut_ptr(), a2);
    let mut s2: f64 = t.iter().map(|&v| v as f64).sum();
    _mm256_storeu_ps(t.as_mut_ptr(), a3);
    let mut s3: f64 = t.iter().map(|&v| v as f64).sum();
    _mm256_storeu_ps(t.as_mut_ptr(), a4);
    let mut s4: f64 = t.iter().map(|&v| v as f64).sum();
    for i in chunks * 8..n {
        let gv = g[i] as f64;
        let p_old = (q[i] as i16) as f64 * scale as f64;
        let p_new = b1 as f64 * p_old + (ob as f64) * gv;
        out[i] = p_new as f32;
        s0 += gv * gv;
        s1 += p_old * p_old;
        s2 += gv * p_old;
        s3 += p_new * p_new;
        s4 += p_new * gv;
    }
    (s0, s1, s2, s3, s4)
}

/// out[i] = clamp(round(p[i]*inv), ±32767) как u16-биты i16.
pub fn rin_quantize_block(p: &[f32], inv: f32, out: &mut [u16]) {
    if has_avx2() {
        unsafe { rin_quantize_avx2(p, inv, out) };
        return;
    }
    for i in 0..p.len().min(out.len()) {
        let v = (p[i] * inv).round().clamp(-32767.0, 32767.0);
        out[i] = v as i16 as u16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rin_quantize_avx2(p: &[f32], inv: f32, out: &mut [u16]) {
    use std::arch::x86_64::*;
    let n = p.len().min(out.len());
    let chunks = n / 8;
    let vi = _mm256_set1_ps(inv);
    let vmin = _mm256_set1_ps(-32767.0);
    let vmax = _mm256_set1_ps(32767.0);
    let mut t = [0i32; 8];
    for c in 0..chunks {
        let o = c * 8;
        let v = _mm256_mul_ps(_mm256_loadu_ps(p.as_ptr().add(o)), vi);
        let r = _mm256_round_ps::<0x8>(v);
        let cl = _mm256_min_ps(_mm256_max_ps(r, vmin), vmax);
        _mm256_storeu_si256(t.as_mut_ptr() as *mut __m256i, _mm256_cvtps_epi32(cl));
        for i in 0..8 {
            out[o + i] = t[i] as i16 as u16;
        }
    }
    for i in chunks * 8..n {
        let v = (p[i] * inv).round().clamp(-32767.0, 32767.0);
        out[i] = v as i16 as u16;
    }
}

// ═══════════════════════════════════════════════════════════
// Veron-fork: собственный пул постоянных потоков (парковка на condvar,
// ноль аллокаций на fork/join). Чанки фиксированные → результаты
// бит-в-бит одинаковы при любом числе потоков: вилка меняет лишь то,
// какой поток какой чанк взял.
// ═══════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
mod vpool {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, OnceLock};

    type Job = unsafe fn(me: usize, arg: usize);

    struct Slot {
        job: AtomicUsize,
        arg: AtomicUsize,
    }

    struct Wake {
        gen: usize,
        done: usize,
        /// Сколько воркеров участвуют в текущем поколении. Воркеры с
        /// my >= active не должны выполнять job: их слот содержит job
        /// прошлого поколения, чей arg уже мог быть освобождён (UAF).
        active: usize,
        shutdown: bool,
    }

    pub const MAXW: usize = 12;
    static SLOT: [Slot; MAXW] = unsafe { std::mem::zeroed() };
    static STARTED: AtomicUsize = AtomicUsize::new(0);
    static READY: AtomicUsize = AtomicUsize::new(0);
    static WAKE: OnceLock<(Mutex<Wake>, Condvar)> = OnceLock::new();
    /// Пул — глобальный ресурс: два одновременных fork затирали бы слоты
    /// и поколение друг друга. Сериализуем fork (job не должен вызывать
    /// fork рекурсивно — иначе deadlock).
    static FORK_LOCK: Mutex<()> = Mutex::new(());
    /// Спавн воркеров тоже сериализуем: иначе два потока могут поднять
    /// дубликаты с одинаковым my.
    static SPAWN_LOCK: Mutex<()> = Mutex::new(());
    /// Паника в job не должна превращаться в вечный wait на condvar:
    /// воркер ловит panic, всё равно инкрементит done, а fork пробрасывает
    /// ошибку только после того, как все участники дочитали поколение.
    static PANIC: AtomicBool = AtomicBool::new(false);

    /// CPU для воркера my: сначала по одному на физическое ядро (без SMT),
    /// затем — оставшиеся логические CPU (SMT-соседи). Так 6 потоков
    /// садятся на 6 физических ядер, а не на 3 ядра × 2 гиперпотока
    /// (на Zen3 CPU 0/1 — одна SMT-пара).
    fn worker_cpu(my: usize) -> Option<usize> {
        #[cfg(target_os = "linux")]
        {
            let phys = crate::atom::flash::flash_hw::physical_cpu_ids();
            if my < phys.len() {
                return Some(phys[my]);
            }
            let all = crate::atom::flash::flash_hw::allowed_cpu_ids();
            let used: std::collections::HashSet<usize> = phys.iter().copied().collect();
            let sibs: Vec<usize> = all.iter().copied().filter(|c| !used.contains(c)).collect();
            return sibs
                .get(my - phys.len())
                .copied()
                .or_else(|| all.get(my).copied());
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = my;
            None
        }
    }

    /// Пришпилить текущий поток к cpu на время жизни guard'а и вернуть
    /// старую affinity-маску при выходе (в т.ч. при panic).
    #[cfg(target_os = "linux")]
    struct AffinityGuard(Option<libc::cpu_set_t>);

    #[cfg(target_os = "linux")]
    impl Drop for AffinityGuard {
        fn drop(&mut self) {
            if let Some(mask) = self.0.take() {
                unsafe {
                    libc::pthread_setaffinity_np(
                        libc::pthread_self(),
                        std::mem::size_of::<libc::cpu_set_t>(),
                        &mask,
                    );
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    fn pin_current(cpu: Option<usize>) -> AffinityGuard {
        let Some(cpu) = cpu else {
            return AffinityGuard(None);
        };
        unsafe {
            let mut old: libc::cpu_set_t = std::mem::zeroed();
            let sz = std::mem::size_of::<libc::cpu_set_t>();
            if libc::sched_getaffinity(0, sz, &mut old) != 0 {
                return AffinityGuard(None);
            }
            if crate::atom::flash::flash_hw::pin_thread_to(&[cpu]).is_err() {
                return AffinityGuard(None);
            }
            AffinityGuard(Some(old))
        }
    }

    fn wake() -> &'static (Mutex<Wake>, Condvar) {
        WAKE.get_or_init(|| {
            (
                Mutex::new(Wake {
                    gen: 0,
                    done: 0,
                    active: 0,
                    shutdown: false,
                }),
                Condvar::new(),
            )
        })
    }

    fn worker(my: usize) {
        // Воркер навсегда остаётся на своём физическом ядре: миграции
        // между SMT-соседями ломают масштабирование GEMM.
        #[cfg(target_os = "linux")]
        if let Some(cpu) = worker_cpu(my) {
            let _ = crate::atom::flash::flash_hw::pin_thread_to(&[cpu]);
        }
        let (lock, cv) = wake();
        READY.fetch_add(1, Ordering::Release);
        let mut seen = 0usize;
        loop {
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            while st.gen == seen && !st.shutdown {
                st = cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if st.shutdown {
                return;
            }
            let g = st.gen;
            let active = st.active;
            drop(st);

            // Воркер вне active текущего поколения обязан вернуться в wait:
            // его слот хранит job прошлого поколения с уже мёртвым arg.
            if my < active {
                let job = SLOT[my].job.load(Ordering::Acquire);
                let arg = SLOT[my].arg.load(Ordering::Acquire);
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    std::mem::transmute::<usize, Job>(job)(my, arg)
                }));
                if res.is_err() {
                    PANIC.store(true, Ordering::Release);
                }

                // done инкрементится ВСЕГДА: иначе паника воркера
                // оставит главный поток навсегда в cv.wait.
                let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
                st.done += 1;
                cv.notify_all();
            }
            seen = g;
        }
    }

    pub fn ensure_pool(threads: usize) {
        let _spawn = SPAWN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let want = threads.saturating_sub(1).min(MAXW);
        let started = STARTED.load(Ordering::Acquire);
        if started < want {
            for i in started..want {
                std::thread::Builder::new()
                    .name(format!("veron-w{}", i))
                    .spawn(move || worker(i))
                    .expect("veron: spawn worker");
            }
            STARTED.store(want, Ordering::Release);
        }
        // Не уходим до того, как новые воркеры дошли до wait: иначе
        // notify_all может потеряться и fork будет ждать вечно.
        let mut spins = 0usize;
        while READY.load(Ordering::Acquire) < want {
            std::thread::yield_now();
            spins += 1;
            if spins > 2_000_000 {
                panic!("veron: worker pool did not become ready");
            }
        }
    }

    /// job вызывается для me ∈ [0..parts): воркеры получают me = 0..workers,
    /// главный поток — me = workers (добирает остаток чанков).
    pub fn fork(parts: usize, job: Job, arg: usize) {
        let _fork = FORK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        ensure_pool(parts);
        let started = STARTED.load(Ordering::Acquire);
        let workers = parts.saturating_sub(1).min(started);
        if workers == 0 {
            unsafe { job(0, arg) };
            return;
        }

        PANIC.store(false, Ordering::Release);
        // Слоты заполняем до публикации нового поколения.
        for i in 0..workers {
            let s = &SLOT[i];
            s.arg.store(arg, Ordering::Release);
            s.job.store(job as usize, Ordering::Release);
        }

        let (lock, cv) = wake();
        {
            let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
            st.done = 0;
            st.active = workers;
            st.gen = st.gen.wrapping_add(1);
            cv.notify_all();
        }

        // Главный поток тоже сажаем на отдельное физическое ядро —
        // иначе шедулер может положить его на SMT-соседа воркера.
        #[cfg(target_os = "linux")]
        let _aff = pin_current(worker_cpu(workers));

        let main_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            job(workers, arg)
        }));

        // Даже при панике главного потока ОБЯЗАНЫ дождаться воркеров:
        // arg указывает на стек вызывающего и разрушается при unwind.
        let mut st = lock.lock().unwrap_or_else(|e| e.into_inner());
        while st.done < workers {
            st = cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        drop(st);

        let worker_panic = PANIC.load(Ordering::Acquire);
        if let Err(payload) = main_res {
            std::panic::resume_unwind(payload);
        }
        if worker_panic {
            panic!("veron: job panicked in worker thread");
        }
    }
}

/// Контекст чанковой job: оси + частичные результаты по НОМЕРУ ЧАНКА.
#[cfg(target_arch = "x86_64")]
struct ForkCtx {
    a: *const f32,
    b: *const f32,
    p: *mut f32,
    p2: *mut f32,
    n: usize,
    parts: usize,
    kind: u8,
}

#[cfg(target_arch = "x86_64")]
unsafe fn fork_job(me: usize, arg: usize) {
    let c = &*(arg as *const ForkCtx);
    let nch = c.n.div_ceil(CHUNK);
    // Чанки назначаются с шагом parts: поток me берёт me, me+parts, …
    let mut cid = me;
    while cid < nch {
        let s = cid * CHUNK;
        let e = (s + CHUNK).min(c.n).max(s);
        let ca = std::slice::from_raw_parts(c.a.add(s), e - s);
        let cb = std::slice::from_raw_parts(c.b.add(s), e - s);
        match c.kind {
            0 => *c.p.add(cid) = dot_ts16(ca, cb),
            2 => *c.p.add(cid) = dsumsq_ts16(ca, cb),
            3 => *c.p.add(cid) = sumsq_ts16(ca),
            _ => {
                let (x, y) = sums2_ts16(ca);
                *c.p.add(cid) = x;
                *c.p2.add(cid) = y;
            }
        }
        cid += c.parts;
    }
}

/// Чанковая редукция: каждое частичное считается ОДНИМ ts-ядром на
/// чанке ≤ CHUNK, итог = сумма частичных по индексам (детерминизм).
/// kind: 0 dot, 2 dsumsq, 3 sumsq, 7 sums2 (q = Σ² вместе с p = Σ).
fn thread_reduce(kind: u8, a: &[f32], b: &[f32], parts: usize) -> (Vec<f32>, Vec<f32>) {
    #[cfg(not(target_arch = "x86_64"))]
    {
        let n = a.len().min(b.len());
        let mut s = 0.0f32;
        let mut sq = 0.0f32;
        match kind {
            0 => s = dot_scalar(a, b),
            2 => s = a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum(),
            3 => { for &x in a { sq += x * x; } }
            _ => {
                for &x in a { s += x; sq += x * x; }
            }
        }
        (vec![s], vec![sq])
    }
    #[cfg(target_arch = "x86_64")]
    {
        let n = if b.is_empty() { a.len() } else { a.len().min(b.len()) };
        if !has_avx2() {
            let mut s = 0.0f32;
            let mut sq = 0.0f32;
            match kind {
                0 => s = dot_scalar(a, b),
                2 => s = a.iter().zip(b).map(|(x, y)| { let d = x - y; d * d }).sum(),
                3 => { for &x in a { sq += x * x; } }
                _ => { for &x in a { s += x; sq += x * x; } }
            }
            return (vec![s], vec![sq]);
        }
        let nch = n.div_ceil(CHUNK);
        let mut p = vec![0.0f32; nch];
        let mut q = vec![0.0f32; nch];
        if parts <= 1 {
            // Последовательный путь: те же чанки, то же ядро, тот же порядок
            // сумм — бит-в-бит равен многопоточному.
            for cid in 0..nch {
                let s = cid * CHUNK;
                let e = (s + CHUNK).min(n).max(s);
                unsafe {
                    let ca = a.get_unchecked(s..e);
                    let cb = b.get_unchecked(s..e);
                    match kind {
                        0 => p[cid] = dot_ts16(ca, cb),
                        2 => p[cid] = dsumsq_ts16(ca, cb),
                        3 => p[cid] = sumsq_ts16(ca),
                        _ => {
                            let (x, y) = sums2_ts16(ca);
                            p[cid] = x;
                            q[cid] = y;
                        }
                    }
                }
            }
            return (p, q);
        }
        vpool::ensure_pool(parts);
        let ctx = ForkCtx {
            a: a.as_ptr(),
            b: if b.is_empty() { a.as_ptr() } else { b.as_ptr() },
            p: p.as_mut_ptr(),
            p2: q.as_mut_ptr(),
            n,
            parts,
            kind,
        };
        vpool::fork(parts, fork_job, &ctx as *const ForkCtx as usize);
        (p, q)
    }
}

/// Прогрев Veron-пула (вызывается из veron::set_parallel_threads).
pub fn prewarm_fork(threads: usize) {
    #[cfg(target_arch = "x86_64")]
    if threads > 1 {
        vpool::ensure_pool(threads);
    }
}

/// Параллельный fork по строкам через Veron-пул: job(me, arg) для me в [0..parts).
/// `arg` — указатель на контекст (как в vpool::fork). Без аллокаций, без условных
/// волн: job получает всё в arg (ситтер, строки, output-указатель). Детерминированный
/// порядок чанков — результат не зависит от того, какой поток какой чанк взял.
pub fn fork_chunks(parts: usize, job: unsafe fn(usize, usize), arg: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        vpool::fork(parts, job, arg);
        return;
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        unsafe { job(0, arg) };
    }
}

/// Максимальное число участников fork (воркеры пула + главный поток).
pub fn max_fork_parts() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        vpool::MAXW + 1
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter(AtomicUsize);

    unsafe fn bad_job(me: usize, arg: usize) {
        let c = &*(arg as *const Counter);
        c.0.fetch_add(1, Ordering::Relaxed);
        // me==0 всегда воркер (главный получает me=workers>=1).
        if me == 0 {
            panic!("boom from worker {me}");
        }
    }

    unsafe fn good_job(_me: usize, arg: usize) {
        let c = &*(arg as *const Counter);
        c.0.fetch_add(1, Ordering::Relaxed);
    }

    /// Паника в воркере не должна вешать condvar-пул: fork обязан
    /// дождаться всех, пробросить panic и остаться рабочим.
    #[test]
    fn panic_in_worker_does_not_hang_pool() {
        let c = Counter(AtomicUsize::new(0));
        let arg = &c as *const Counter as usize;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            fork_chunks(4, bad_job, arg);
        }));
        assert!(res.is_err(), "panic must propagate to caller");

        let c2 = Counter(AtomicUsize::new(0));
        let arg2 = &c2 as *const Counter as usize;
        fork_chunks(4, good_job, arg2);
        assert_eq!(c2.0.load(Ordering::Relaxed), 4, "pool must stay usable");
    }

    #[test]
    fn elementwise_matches_scalar() {
        // Размеры: кратные 32, + хвост, + малые (<8).
        for &n in &[1usize, 7, 8, 31, 32, 33, 64, 100, 256, 1024] {
            let a: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.3).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();

            let mut y = a.clone();
            add_elem(&mut y, &b);
            for i in 0..n { assert!((y[i] - (a[i] + b[i])).abs() < 1e-6); }

            let mut y = a.clone();
            mul_elem(&mut y, &b);
            for i in 0..n { assert!((y[i] - (a[i] * b[i])).abs() < 1e-5); }

            let mut y = a.clone();
            relu(&mut y);
            for i in 0..n { assert!((y[i] - a[i].max(0.0)).abs() < 1e-6); }

            let mut y = a.clone();
            silu(&mut y);
            for i in 0..n {
                let want = a[i] / (1.0 + (-a[i]).exp());
                assert!((y[i] - want).abs() < 1e-6, "n={n} i={i}: {} vs {want}", y[i]);
            }

            let mut y = a.clone();
            silu_fast(&mut y);
            for i in 0..n {
                let want = a[i] / (1.0 + (-a[i]).exp());
                assert!((y[i] - want).abs() < 2e-4, "n={n} i={i}: {} vs {want}", y[i]);
            }
        }
    }
}

#[cfg(test)]
mod softmax_jit_tests {
    /// Регрессия: хвост длины 2 в JIT-softmax давал NaN (таблица масок
    /// перекрывалась константами Q4).
    #[test]
    fn softmax_jit_vs_scalar_all_len() {
        let _jit = crate::atom::flash::jit::TEST_JIT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for n in 1..=64usize {
            let x: Vec<f32> = (0..n).map(|k| (k as f32 * 0.37).sin() * 3.0).collect();
            let mut a = x.clone();
            let mut b = x.clone();
            crate::atom::flash::jit::set_jit_enabled(true);
            super::softmax(&mut a);
            crate::atom::flash::jit::set_jit_enabled(false);
            super::softmax(&mut b);
            crate::atom::flash::jit::set_jit_enabled(true);
            for i in 0..n {
                assert!(a[i].is_finite(), "JIT n={n}: NaN в позиции {i}");
                assert!(
                    (a[i] - b[i]).abs() < 1e-5,
                    "n={n} i={i}: JIT {} != scalar {}",
                    a[i],
                    b[i]
                );
            }
        }
    }

    /// Регрессия элементных ядер DynASM против intrinsics/scalar:
    /// все длины (хвосты 1..15 особенно), все 8 операций (+momentum).
    #[test]
    fn elem_jit_vs_intrinsics_all_len() {
        let _jit = crate::atom::flash::jit::TEST_JIT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for n in 1..=100usize {
            let base: Vec<f32> = (0..n)
                .map(|k| (k as f32 * 0.61).sin() * 2.0)
                .collect();
            let other: Vec<f32> = (0..n)
                .map(|k| (k as f32 * 0.29).cos() * 1.5 + 0.3)
                .collect();
            let (a, b) = (0.7f32, -0.35f32);
            // Эталон: скаляр прямо по формуле (intrinsics сверяются ниже
            // через отключённый JIT, но битовые различия mul vs fma здесь
            // допустимы до шкалы погрешности).
            crate::atom::flash::jit::set_jit_enabled(true);
            let mut r_add = base.clone(); super::add_elem(&mut r_add, &other);
            let mut r_mul = base.clone(); super::mul_elem(&mut r_mul, &other);
            let mut r_scale = base.clone(); super::scale(&mut r_scale, a);
            let mut r_sto = base.clone(); super::scale_to(&mut r_sto, &other, b);
            let mut r_axpy = base.clone(); super::axpy(&mut r_axpy, a, &other);
            let mut r_axpby = base.clone(); super::axpby(&mut r_axpby, a, &other, b);
            let mut r_relu = base.clone(); super::relu(&mut r_relu);
            let mut r_mom = base.clone(); super::momentum_update(&mut r_mom, &other, 0.9);
            let mut r_aff = vec![0.0f32; n]; super::affine_into(&other, a, b, &mut r_aff);

            crate::atom::flash::jit::set_jit_enabled(false);
            let mut s_add = base.clone(); super::add_elem(&mut s_add, &other);
            let mut s_mul = base.clone(); super::mul_elem(&mut s_mul, &other);
            let mut s_scale = base.clone(); super::scale(&mut s_scale, a);
            let mut s_sto = base.clone(); super::scale_to(&mut s_sto, &other, b);
            let mut s_axpy = base.clone(); super::axpy(&mut s_axpy, a, &other);
            let mut s_axpby = base.clone(); super::axpby(&mut s_axpby, a, &other, b);
            let mut s_relu = base.clone(); super::relu(&mut s_relu);
            let mut s_mom = base.clone(); super::momentum_update(&mut s_mom, &other, 0.9);
            let mut s_aff = vec![0.0f32; n]; super::affine_into(&other, a, b, &mut s_aff);
            crate::atom::flash::jit::set_jit_enabled(true);

            let check = |name: &str, r: &[f32], s: &[f32]| {
                for i in 0..n {
                    let (x, y) = (r[i], s[i]);
                    let tol = 1e-4 + 1e-3 * y.abs();
                    assert!(
                        (x - y).abs() < tol,
                        "{name} n={n} i={i}: JIT {x} != ref {y}"
                    );
                }
            };
            check("add_elem", &r_add, &s_add);
            check("mul_elem", &r_mul, &s_mul);
            check("scale", &r_scale, &s_scale);
            check("scale_to", &r_sto, &s_sto);
            check("axpy", &r_axpy, &s_axpy);
            check("axpby", &r_axpby, &s_axpby);
            check("relu", &r_relu, &s_relu);
            check("momentum_update", &r_mom, &s_mom);
            check("affine_into", &r_aff, &s_aff);
        }
    }

    /// Та же таблица масок используется SiLU-ядром: проверяем все длины
    /// хвостов, включая 2 (регрессия перекрытия MT_TAIL_OR с константами Q4).
    #[test]
    fn silu_fast_jit_finite_all_len() {
        let _jit = crate::atom::flash::jit::TEST_JIT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for n in 1..=64usize {
            let x: Vec<f32> = (0..n)
                .map(|k| (k as f32 * 0.53).sin() * 2.0)
                .collect();
            let mut fast = x.clone();
            let mut exact = x.clone();
            crate::atom::flash::jit::set_jit_enabled(true);
            super::silu_fast(&mut fast);
            super::silu(&mut exact);
            for i in 0..n {
                assert!(fast[i].is_finite(), "SiLU n={n}: NaN в позиции {i}");
                assert!(
                    (fast[i] - exact[i]).abs() < 2e-3,
                    "n={n} i={i}: fast {} != exact {}",
                    fast[i],
                    exact[i]
                );
            }
        }
    }
}
