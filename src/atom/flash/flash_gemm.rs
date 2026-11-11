// src/atom/flash/flash_gemm.rs
//! FLASH-GEMM: кэш-блочный GEMM + GEMV поверх pool-системы.
//!
//! Многоуровневая организация памяти (по возрастанию объёма):
//!
//! 1. **Регистры** — микроядро 4×16: 8 независимых FMA-цепочек, B грузится
//!    2× на k-шаг, A — 4 broadcast'а. 14 YMM из 16.
//! 2. **L1** — строка A и 16-колоночный блок B переиспользуются в k-цикле.
//! 3. **L2** — панель B KC×NC (128×256×4 = 128 КиБ) упакована в layout
//!    [kk][j] и читается из памяти ровно один раз на все строки-чанки.
//! 4. **L3** — панели общие для потоков (read-only), поэтому не копируются.
//! 5. **Потоки** — Veron-пул делит строки на чанки, кратные 6, а колонки —
//!    на jc-панели (2D-сетка div_m × div_n, как у OpenBLAS); пересечений
//!    нет, результат детерминирован. Каждый поток читает только свою
//!    долю B (1/div_n) вместо всей B.
//!
//! Операторы:
//!
//! - [`gemm_blocked`] / [`gemm_blocked_into`] — C = A×B: упаковка панелей B,
//!   микроядро 4×16 AVX2+FMA, fork по строкам. `_into` пишет сразу в готовый
//!   буфер (numpy-массив) без промежуточного Vec и копии.
//! - [`gemv_pooled`] — y = W×x: 4-строчный AVX2-блокинг (1.25 load/FMA
//!   против 1.5 у 2-строчного), pool по строкам для средних матриц.
//! - [`gemv_fused_silu`] — GEMV + SiLU-эпилог за один проход, без временного
//!   буфера (экономия прохода по памяти на больших слоях).
//! - [`gemv_default_parts`] — эмпирический выбор числа потоков по размеру:
//!   малые — последовательно (пробуждение пула дороже работы), средние
//!   (L2/L3) — несколько потоков, огромные — снова 1 (DRAM уже насыщен).
//!
//! Многослойность моделей: FlashDense (уровень 0) → FlashGemm (уровень 0.5,
//! dense GEMM/GEMV поверх pool) → FlashCompressed (уровень 1, Q4×Q8 GEMV) →
//! FlashLayer (уровень 2, живые слои). Каждый слой работает поверх
//! предыдущего, не дублируя ядра.

use crate::atom::flash::flash::FlashDense;
use crate::atom::math::simd;

#[allow(deprecated)] // _mm_getcsr/_mm_setcsr: тот же приём, что в flash.rs
fn ensure_fast_math() {
    use std::sync::Once;
    static FAST: Once = Once::new();
    FAST.call_once(|| {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let csr = _mm_getcsr();
            _mm_setcsr(csr | 0x8040);
        }
    });
}

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

// ═══════════════════════════════════════════════════════════
// Параллельный GEMV: y = W×x, строки делятся на parts частей,
// части — на потоки Veron-пула. Детерминированность: результат
// бит-идентичен последовательному gemv (те же 2-строчные блоки,
// тот же порядок FMA, hsum в том же порядке).
// ═══════════════════════════════════════════════════════════

struct GemvCtx {
    dense: *const FlashDense,
    xp: *const f32,
    y: *mut f32,
    stride: usize,
    rows: usize,
    parts: usize,
}

// Контекст целиком живёт на стеке caller'а fork'а; fork блокируется до конца
// джобы, поэтому данные не могут быть освобождены раньше (как в thread_reduce).
unsafe impl Send for GemvCtx {}
unsafe impl Sync for GemvCtx {}

unsafe fn gemv_fork_job(me: usize, arg: usize) {
    let c = &mut *(arg as *mut GemvCtx);
    // Чанки: с шагом parts (det order)
    let rows = c.rows;
    let parts = c.parts.max(1);
    let chunk = rows.div_ceil(parts);
    let start = me * chunk;
    let end = (start + chunk).min(rows);
    if start >= end {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let dense = &*c.dense;
        let stride = c.stride;
        let base = dense.data.as_ptr();
        let px = c.xp;
        let y_ptr = c.y;
        let chunks = stride / 32;
        // Пары строк считаем от start, а не от end: при нечётном start
        // иначе последняя пара писала бы y[i+1] за границу чанка (а на
        // последнем чанке — за границу вектора => порча heap).
        let rows2 = if (end - start) % 2 == 0 {
            end
        } else {
            end - 1
        };
        let mut i = start;
        while i < rows2 {
            // prefetch 4 строки вперёд
            if i + 8 < rows && (i + 8) < end {
                for k in 4..8 {
                    _mm_prefetch(base.add((i + k) * stride) as *const i8, _MM_HINT_T0);
                }
            }
            let p0 = base.add(i * stride);
            let p1 = p0.add(stride);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            let mut b0 = _mm256_setzero_ps();
            let mut b1 = _mm256_setzero_ps();
            let mut b2 = _mm256_setzero_ps();
            let mut b3 = _mm256_setzero_ps();
            for ch in 0..chunks {
                let o = ch * 32;
                let x0 = _mm256_loadu_ps(px.add(o));
                let x1 = _mm256_loadu_ps(px.add(o + 8));
                let x2 = _mm256_loadu_ps(px.add(o + 16));
                let x3 = _mm256_loadu_ps(px.add(o + 24));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), x0, a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 8)), x1, a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 16)), x2, a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 24)), x3, a3);
                b0 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o)), x0, b0);
                b1 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o + 8)), x1, b1);
                b2 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o + 16)), x2, b2);
                b3 = _mm256_fmadd_ps(_mm256_loadu_ps(p1.add(o + 24)), x3, b3);
            }
            *y_ptr.add(i) = hsum4(a0, a1, a2, a3);
            *y_ptr.add(i + 1) = hsum4(b0, b1, b2, b3);
            i += 2;
        }
        // Хвост нечётной строки.
        while i < end {
            let p0 = base.add(i * stride);
            let mut a0 = _mm256_setzero_ps();
            let mut a1 = _mm256_setzero_ps();
            let mut a2 = _mm256_setzero_ps();
            let mut a3 = _mm256_setzero_ps();
            for ch in 0..chunks {
                let o = ch * 32;
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o)), _mm256_loadu_ps(px.add(o)), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 8)), _mm256_loadu_ps(px.add(o + 8)), a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 16)), _mm256_loadu_ps(px.add(o + 16)), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(p0.add(o + 24)), _mm256_loadu_ps(px.add(o + 24)), a3);
            }
            *y_ptr.add(i) = hsum4(a0, a1, a2, a3);
            i += 1;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let dense = &*c.dense;
        let stride = c.stride;
        let d = dense.data.as_slice();
        let xp = std::slice::from_raw_parts(c.xp, stride);
        let y = std::slice::from_raw_parts_mut(c.y, c.rows);
        for i in start..end {
            let o = i * stride;
            y[i] = crate::atom::math::simd::dot(&d[o..o + stride], xp);
        }
    }
}

/// Параллельный GEMV через fork: строки делятся на parts частей.
pub fn gemv_pooled(dense: &FlashDense, x: &[f32], parts: usize) -> Vec<f32> {
    let rows = dense.rows;
    let mut y = vec![0.0f32; rows];
    if rows == 0 {
        return y;
    }
    ensure_fast_math();
    let xp = dense.pad_input_pub(x);
    let parts = parts.max(1).min(rows);
    if parts == 1 {
        return dense.gemv(x);
    }
    let parts = parts.min(simd::max_fork_parts());
    let mut ctx = GemvCtx {
        dense: dense as *const FlashDense,
        xp: xp.as_ptr(),
        y: y.as_mut_ptr(),
        stride: dense.stride,
        rows,
        parts,
    };
    simd::fork_chunks(parts, gemv_fork_job as unsafe fn(usize, usize), &mut ctx as *mut GemvCtx as usize);
    y
}

// ═══════════════════════════════════════════════════════════
// GEMV + SiLU за один проход (без временного буфера y):
// выход сразу silu(y_i), bias — прибавлен в эпилоге.
// ═══════════════════════════════════════════════════════════

/// y = silu(W×x + b) одним проходом без временных y-буферов.
/// `parts` — число потоков fork'а (1 = последовательный).
pub fn gemv_fused_silu(dense: &FlashDense, x: &[f32], parts: usize) -> Vec<f32> {
    let rows = dense.rows;
    let mut y = vec![0.0f32; rows];
    if rows == 0 {
        return y;
    }
    struct Ctx {
        dense: *const FlashDense,
        xp: *const f32,
        y: *mut f32,
        stride: usize,
        rows: usize,
        parts: usize,
    }
    unsafe impl Send for Ctx {}
    unsafe impl Sync for Ctx {}

    unsafe fn fork_job(me: usize, arg: usize) {
        let c = &*(arg as *const Ctx);
        let chunk = c.rows.div_ceil(c.parts.max(1));
        let start = me * chunk;
        let end = (start + chunk).min(c.rows);
        if start >= end {
            return;
        }
        let dense = &*c.dense;
        let xp = std::slice::from_raw_parts(c.xp, c.stride);
        let bias = &dense.bias;
        let y = std::slice::from_raw_parts_mut(c.y, c.rows);
        unsafe { dense.gemv_avx2_range(xp, y, start, end) };
        // SiLU-эпилог: ASM-ядро (2×8 float, FMA-полином, без libm в цикле).
        let yc = &mut y[start..end];
        let bc = bias.get(start..end).unwrap_or(&[]);
        crate::atom::math::simd::fused_bias_silu(yc, bc, 1, end - start);
    }

    let xp = dense.pad_input_pub(x);
    let parts = parts.max(1).min(rows).min(simd::max_fork_parts());
    if parts == 1 {
        let xp = &xp;
        let bias = &dense.bias;
        unsafe { (&*dense).gemv_avx2_range(xp, &mut y, 0, rows) };
        crate::atom::math::simd::fused_bias_silu(&mut y, bias, 1, rows);
        return y;
    }
    let ctx = Ctx {
        dense: dense as *const FlashDense,
        xp: xp.as_ptr(),
        y: y.as_mut_ptr(),
        stride: dense.stride,
        rows,
        parts,
    };
    simd::fork_chunks(parts, fork_job as unsafe fn(usize, usize), &ctx as *const Ctx as usize);
    y
}

// ═══════════════════════════════════════════════════════════
// Кэш-блочный GEMM: C(m×n) = A(m×k) × B(k×n).
//
// Упрощённая схема GotoBLAS:
//   1. B упаковывается в панели KC×NC (layout [kk][j]) — панель целиком
//      лежит в L2 и переиспользуется всеми строками, поэтому B читается
//      из памяти один раз, а не на каждый блок строк.
//   2. Микроядро 4×16: 8 независимых FMA-цепочек; на k-шаг — 2 загрузки
//      B (16 колонок), 4 broadcast'а A и 8 FMA. 14 YMM регистров.
//   3. Строки делятся на чанки между потоками Veron-пула. Панели B
//      read-only и общие, чанки строк не пересекаются → результат
//      детерминирован и не зависит от планировщика.
//
// Для 1024³ это в ~4 раза меньше трафика по B, чем у наивного 4×8-ядра.
// ═══════════════════════════════════════════════════════════

/// Высота k-панели B (128 × 256 × 4 = 128 КиБ — в L2).
const GEMM_KC: usize = 128;
/// Ширина n-панели B.
const GEMM_NC: usize = 256;

/// Размеры панелей (можно переопределить env-переменными для тюнинга:
/// ATOMNET_GEMM_KC / ATOMNET_GEMM_NC).
fn gemm_dims() -> (usize, usize) {
    use std::sync::OnceLock;
    static D: OnceLock<(usize, usize)> = OnceLock::new();
    *D.get_or_init(|| {
        let env_usize = |name: &str, def: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&v| (16..=1024).contains(&v))
                .unwrap_or(def)
        };
        (
            env_usize("ATOMNET_GEMM_KC", GEMM_KC),
            env_usize("ATOMNET_GEMM_NC", GEMM_NC),
        )
    })
}

/// Высота MC-блока строк (можно переопределить ATOMNET_GEMM_MC).
fn gemm_mc() -> usize {
    use std::sync::OnceLock;
    static D: OnceLock<usize> = OnceLock::new();
    *D.get_or_init(|| {
        std::env::var("ATOMNET_GEMM_MC")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| (24..=8192).contains(&v))
            .unwrap_or(96)
    })
}

/// Панель B: блок [pc, pc+kc) × [jc, jc+nc) лежащий в упаковочном буфере
/// как непрерывный kc×stride кусок (как у OpenBLAS: тайл ядра — одно
/// окно памяти без страйдов через страницу).
/// `off` — смещение панели в буфере (float), `stride` — шаг строки
/// панели (кратный 16, нечётное число 64Б-линий против конфликтов L1).
struct GemmPanel {
    pc: usize,
    jc: usize,
    kc: usize,
    nc: usize,
    off: usize,
    stride: usize,
}

/// Регион (регистровая регионная система): MC×NC блок, чьи C-данные
/// живут в L2 все k-панели. Регион делится на регистровые кластеры
/// 6×16 (12 YMM-аккумуляторов) — элементарные единицы JIT-ядер.
pub struct GemmRegion {
    pub r0: usize,
    pub r1: usize,
    pub jc0: usize,
    pub jc1: usize,
}

/// Регистровый кластер: JIT-тайл 6 строк × 16 колонок (12 YMM-аккумуляторов,
/// без спилов). Выполнение внутри региона — последовательные кластеры,
/// аккумуляторы живут в регистрах ровно один k-панельный вызов.
pub struct RegisterCluster {
    pub row_group: usize,
    pub col_tile: usize,
}

struct GemmCtx {
    a: *const f32,
    /// Исходный B: читается только в non-x86 fallback (там нет packed).
    #[allow(dead_code)]
    b: *const f32,
    c: *mut f32,
    packed: *const f32,
    /// Упакованный A (6-строчные группы) или null, если упаковка выключена.
    a_packed: *const f32,
    /// A-панель: шаг строки (kc_max), число 6-строчных групп
    /// (для указателя: (pc_idx*groups + g)*6*kc_max).
    a_kc: usize,
    groups: usize,
    panels: *const GemmPanel,
    n_panels: usize,
    rows: usize,
    k: usize,
    cols: usize,
    /// Всего воркеров fork (div_m × div_n).
    parts: usize,
    /// 2D-партиция потоков (OpenBLAS-зеркало, gemm_thread_mn divide_rule):
    /// регион (mi, ni) считает строки mi и колонки ni. Каждый поток читает
    /// только 1/div_n часть B-панелей вместо всей B (в разы меньше L3/DRAM
    /// трафика на больших GEMM). 1D (div_n=1) — частный случай.
    div_m: usize,
    div_n: usize,
    /// Регионная очередь: воркеры в пуле тянут регионы по атомарному
    /// счётчику (work-stealing на уровне регионов). При regions == parts
    /// порядок неважен (выполнение детерминировано: регион независим);
    /// при ragged-хвостах балансный воркёр подхватывает следующий регион.
    regions: *const GemmRegion,
    n_regions: usize,
    next_region: std::sync::atomic::AtomicUsize,
    /// DSR-регионы с живым скипом: битмаска строк (1 = живая, 0 = мёртвая),
    /// null = все живые. Мёртвые 6-строчные группы пропускают ядро
    /// (в C пишутся нули) — для гейтованных «живых» слоёв (QairoLivingLayer),
    /// где мёртвые строки всё равно умножаются гейтом 0. Dense — neutral
    /// (1 проверка на группу), sparse — выигрыш пропорционален доле мёртвых.
    row_alive: *const u8,
    /// Для JIT silu-эпилога: bias (длина ≥ cols) или null.
    #[allow(dead_code)]
    bias: *const f32,
    /// true — последняя k-панель каждого тайла пишет silu(A×B + bias)
    /// прямо из JIT; пост-проход по C не нужен.
    #[allow(dead_code)]
    do_silu: bool,
}
unsafe impl Send for GemmCtx {}
unsafe impl Sync for GemmCtx {}

/// Переиспользуемый буфер панелей B: панели [jc][pc] лежат как
/// непрерывные kc×NC-куски (порт OpenBLAS-подхода): тайл ядра 16×kc
/// читается из одного окна памяти. Раньше был padded row-major со
/// страйдом на всю строку B (4–16 КБ): каждый k-шаг ядра бил в новую
/// страницу → L1 DTLB не держал 128 страниц тайла, страницы перешагивались.
struct PackBuf {
    data: crate::atom::flash::flash::Aligned64<f32>,
    a_data: crate::atom::flash::flash::Aligned64<f32>,
    panels: Vec<GemmPanel>,
}

impl PackBuf {
    fn new() -> Self {
        Self {
            data: crate::atom::flash::flash::Aligned64::new_zeroed(0),
            a_data: crate::atom::flash::flash::Aligned64::new_zeroed(0),
            panels: Vec::new(),
        }
    }
}

thread_local! {
    static PACK_BUF: std::cell::RefCell<Option<PackBuf>> =
        const { std::cell::RefCell::new(None) };
}

fn with_pack_buf<R>(f: impl FnOnce(&mut PackBuf) -> R) -> R {
    PACK_BUF.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            let buf = slot.get_or_insert_with(PackBuf::new);
            return f(buf);
        }
        // Реентерабельный вызов (например, из job): свой буфер.
        let mut tmp = PackBuf::new();
        f(&mut tmp)
    })
}

/// Копирование одной панели B: kc строк по nc float, строка источника
/// с шагом n, строка назначения — шаг stride внутри панели.
#[inline]
unsafe fn copy_panel(src: *const f32, n: usize, dst: *mut f32, panel: &GemmPanel) {
    for kk in 0..panel.kc {
        std::ptr::copy_nonoverlapping(
            src.add((panel.pc + kk) * n + panel.jc),
            dst.add(panel.off + kk * panel.stride),
            panel.nc,
        );
    }
}

struct PackPanelsCtx {
    src: *const f32,
    n: usize,
    dst: *mut f32,
    panels: *const GemmPanel,
    n_panels: usize,
    parts: usize,
}
unsafe impl Send for PackPanelsCtx {}
unsafe impl Sync for PackPanelsCtx {}

unsafe fn pack_panels_job(me: usize, arg: usize) {
    let c = &*(arg as *const PackPanelsCtx);
    let per = c.n_panels.div_ceil(c.parts);
    let p0 = (me * per).min(c.n_panels);
    let p1 = (p0 + per).min(c.n_panels);
    for p in p0..p1 {
        copy_panel(c.src, c.n, c.dst, &*c.panels.add(p));
    }
}

/// Разложить B в неперерывные панели [jc][pc] (stride панели — нечётное
/// число 64Б-линий против конфликтов L1). Параллельно по панелям.
fn pack_b(b: &[f32], k: usize, n: usize, buf: &mut PackBuf, parts: usize) {
    use crate::atom::flash::flash::Aligned64;
    let (_kc_max, nc_max) = gemm_dims();

    buf.panels.clear();
    let mut total = 0usize;
    for jc in (0..n).step_by(nc_max) {
        let nc = nc_max.min(n - jc);
        // Шаг строки панели: округление до 16 float + исправление чётного
        // числа линий (чётное → все kk-строки на одних сетах L1).
        let mut stride = (nc + 15) & !15;
        if (stride / 16) % 2 == 0 {
            stride += 16;
        }
        for pc in (0..k).step_by(_kc_max) {
            let kc = _kc_max.min(k - pc);
            buf.panels.push(GemmPanel {
                pc,
                jc,
                kc,
                nc,
                off: total,
                stride,
            });
            total += kc * stride;
        }
    }

    if buf.data.as_slice().len() < total {
        buf.data = Aligned64::<f32>::new_zeroed(total);
    }
    let src = b.as_ptr();
    let dst = buf.data.as_mut_ptr();
    if parts > 1 && total >= 256 * 1024 && buf.panels.len() > 1 {
        let ctx = PackPanelsCtx {
            src,
            n,
            dst,
            panels: buf.panels.as_ptr(),
            n_panels: buf.panels.len(),
            parts,
        };
        crate::atom::math::simd::fork_chunks(
            parts.min(buf.panels.len().max(1)),
            pack_panels_job as unsafe fn(usize, usize),
            &ctx as *const PackPanelsCtx as usize,
        );
    } else {
        for p in 0..buf.panels.len() {
            unsafe { copy_panel(src, n, dst, &buf.panels[p]) };
        }
    }
}

/// Переопределение упаковки A: ATOMNET_GEMM_PACKA=0/1, по умолчанию None
/// (авто-режим по размеру).
fn packa_override() -> Option<bool> {
    use std::sync::OnceLock;
    static O: OnceLock<Option<bool>> = OnceLock::new();
    *O.get_or_init(|| {
        std::env::var("ATOMNET_GEMM_PACKA")
            .ok()
            .map(|v| v != "0")
    })
}

/// Контекст параллельной упаковки A (OpenBLAS-зеркало): группы строк
/// по 6; layout панели [pc][g][kk][8 floats] — на k-шаг 32 B: 6 строк
/// + 2 padding (kernel читает всё из ОДНОЙ кэш-линии вместо 6).
struct PackACtx {
    src: *const f32,
    rows: usize,
    k: usize,
    dst: *mut f32,
    n_pc: usize,
    kc_max: usize,
    groups: usize,
    parts: usize,
}
unsafe impl Send for PackACtx {}
unsafe impl Sync for PackACtx {}

unsafe fn pack_a_job(me: usize, arg: usize) {
    let c = &*(arg as *const PackACtx);
    let per = c.groups.div_ceil(c.parts);
    let g0 = (me * per).min(c.groups);
    let g1 = (g0 + per).min(c.groups);
    // 6 строк × kc_max — кольцевые займы L1 для scatter'а в [kk][8].
    let mut tmp = vec![0.0f32; 6 * c.kc_max];
    for g in g0..g1 {
        for pc_idx in 0..c.n_pc {
            let pc = pc_idx * c.kc_max;
            let kc = c.kc_max.min(c.k - pc);
            // Чтение: 6 связных строк подряд (быстрые memcpy по строкам).
            for r in 0..6usize {
                let gr = g * 6 + r;
                let row = &mut tmp[r * c.kc_max..r * c.kc_max + kc];
                if gr < c.rows {
                    std::ptr::copy_nonoverlapping(c.src.add(gr * c.k + pc), row.as_mut_ptr(), kc);
                } else {
                    row.fill(0.0);
                }
            }
            // Распределение в interleaved-записи [kk][r]: 32 B на шаг ядра.
            let panel = (pc_idx * c.groups + g) * c.kc_max * 8;
            for kk in 0..kc {
                for r in 0..6usize {
                    *c.dst.add(panel + kk * 8 + r) = tmp[r * c.kc_max + kk];
                }
            }
        }
    }
}

/// Упаковка A в interleaved панели [pc][g][kk][8 floats] (OpenBLAS-зеркало:
/// на k-шаг ядра — 32 B = 1 строка кэша вместо 6 разбросанных строк).
/// Возвращает kc_max.
fn pack_a(a: &[f32], rows: usize, k: usize, buf: &mut PackBuf, parts: usize) -> usize {
    use crate::atom::flash::flash::Aligned64;
    let (kc_max, _) = gemm_dims();
    let n_pc = k.div_ceil(kc_max);
    let groups = rows.div_ceil(6);
    let total = groups * n_pc * kc_max * 8;
    if buf.a_data.as_slice().len() < total {
        buf.a_data = Aligned64::<f32>::new_zeroed(total);
    }
    let ctx = PackACtx {
        src: a.as_ptr(),
        rows,
        k,
        dst: buf.a_data.as_mut_ptr(),
        n_pc,
        kc_max,
        groups,
        parts,
    };
    if parts > 1 && total >= 256 * 1024 && groups > 1 {
        crate::atom::math::simd::fork_chunks(
            parts.min(groups.max(1)),
            pack_a_job as unsafe fn(usize, usize),
            &ctx as *const PackACtx as usize,
        );
    } else {
        unsafe { pack_a_job(0, &ctx as *const PackACtx as usize) };
    }
    kc_max
}

/// Сетка потоков 2D (как OpenBLAS divide_rule): наибольший делитель
/// `parts`, дающий div_m × div_n с div_m ≈ sqrt(parts × rows/cols),
/// чтобы тайлы были квадратнее (баланс A/C-трафика). Ограничения:
/// div_m ≤ rows/6 (≥1 группа JIT на поток), div_n ≤ cols/16.
/// N-split (div_n>1) только для широких C (cols > rows): там B доминирует
/// в DRAM-трафике, и его дележ между потоками выигрывает. На квадратных
/// и высоких GEMM N-split даёт дисбаланс jc-панелей и пустые воркеры —
/// там чистый M-split (как у OpenBLAS при малых N/больших M).
fn gemm_grid(parts: usize, rows: usize, cols: usize) -> (usize, usize) {
    let parts = parts.max(1);
    if parts == 1 || rows < 64 || cols < 64 || cols <= rows {
        return (parts, 1);
    }
    let max_m = rows.div_ceil(6).max(1);
    let max_n = cols.div_ceil(16).max(1);
    let target = ((parts as f64) * (rows as f64 / cols.max(1) as f64)).sqrt();
    let mut best_m = 1usize;
    let mut best_err = f64::INFINITY;
    let mut d = 1usize;
    while d * d <= parts {
        if parts % d == 0 {
            for cand in [d, parts / d] {
                if cand <= max_m && parts / cand <= max_n {
                    let err = (cand as f64 - target).abs();
                    if err < best_err {
                        best_err = err;
                        best_m = cand;
                    }
                }
            }
        }
        d += 1;
    }
    (best_m, parts / best_m)
}

/// Построить регионы (регистровые блоки MC×NC) по сетке div_m × div_n.
/// Границы строк выровнены на 6 (ядро читает 6 строками с фикс. страйдом),
/// границы колонок — на JC-панели (кратные NC), чтобы панели B не
/// пересекались воркерами.
fn gemm_regions(rows: usize, cols: usize, div_m: usize, div_n: usize) -> Vec<GemmRegion> {
    let (_, nc_max) = gemm_dims();
    let align = 6;
    let chunk = rows.div_ceil(div_m.max(1)).div_ceil(align) * align;
    let n_jc = cols.div_ceil(nc_max.max(1));
    let jc_per = n_jc.div_ceil(div_n.max(1));
    let mut out = Vec::with_capacity(div_m * div_n);
    for mi in 0..div_m.max(1) {
        let r0 = mi * chunk;
        let r1 = (r0 + chunk).min(rows);
        if r0 >= rows {
            continue;
        }
        for ni in 0..div_n.max(1) {
            let jc0 = (ni * jc_per * nc_max).min(cols);
            let jc1 = (jc0 + jc_per * nc_max).min(cols);
            out.push(GemmRegion { r0, r1, jc0, jc1 });
        }
    }
    out
}

/// Переразбиение регионов по строкам (баланс хвостов + локальность C).
/// 1 = статично (regions == parts). Измерено на Zen (square/ragged/wide):
/// 2 ≈ 3 ≈ 4 быстрее 1 на 5–8% (меньшие C-регионы устойчивее в L2,
/// воркёр-оползень подхватывает следующий регион); дальше плато.
/// По умолчанию 2; перекрыть ATOMNET_GEMM_OVER (1..=8).
fn gemm_over() -> usize {
    use std::sync::OnceLock;
    static O: OnceLock<usize> = OnceLock::new();
    *O.get_or_init(|| {
        std::env::var("ATOMNET_GEMM_OVER")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| (1..=8).contains(&v))
            .unwrap_or(2)
    })
}

/// DSR-регион: 6-строчная группа полностью мёртвая → ядро пропускается.
#[inline(always)]
unsafe fn group_is_dead(mask: *const u8, i6: usize) -> bool {
    if mask.is_null() {
        return false;
    }
    let m = std::slice::from_raw_parts(mask, i6 + 6);
    m[i6] == 0 && m[i6 + 1] == 0 && m[i6 + 2] == 0 && m[i6 + 3] == 0 && m[i6 + 4] == 0 && m[i6 + 5] == 0
}

unsafe fn gemm_fork_job(_me: usize, arg: usize) {
    let ctx = &*(arg as *const GemmCtx);
    let rows = ctx.rows;
    let cols = ctx.cols;
    let k = ctx.k;
    let rm = ctx.row_alive;
    // Регионная очередь (ворк-стилинг): воркер тянет следующий регион,
    // а не «свой» me. При regions == parts эффект нулевой (выполнение
    // детерминировано: регионы независимы), при ragged — балансировка.
    loop {
        let idx = ctx.next_region.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if idx >= ctx.n_regions {
            break;
        }
        let region = unsafe { &*ctx.regions.add(idx) };
        // Чанки выравниваем по 6 строк: JIT-ядро (и A-панели 6-строчных групп)
        // читают шестью строками с фиксированным страйдом — их группы обязаны
        // быть 6-выровнены глобально, иначе строка уйдёт в соседнюю панель.
        // Хвосты (rows % 6) обрабатывают 4×16/1×16 и скалярные хвосты.
        let start = region.r0;
        let end = region.r1;
        if start >= end {
            continue;
        }
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let pa = ctx.a;
        let pc = ctx.c;
        // Панели B лежат как непрерывные куски (stride панели — в самом
        // GemmPanel): тайл ядра (kc×16) — одно окно памяти без прохода
        // по страницам, A-строки стримятся из L2.
        let pb = ctx.packed;
        // MC-блокировка строк: чанк потока при 2048/6 — 344 строки, и
        // A(344×128) + C(344×256) + B-панель(128 КБ) ≈ 528 КБ > 512 КБ L2.
        // C выдавливался и 16 раз за GEMM ходил в L3/DRAM. Блок 192 строк:
        // A(96 КБ) + C(192 КБ) + B(128 КБ) ≈ 416 КБ — всё в L2; вместе с
        // порядком панелей jc→pc C живёт в L2 все k-панели.
        let mc = gemm_mc();
        let mut ic0 = start;
        while ic0 < end {
            // В silu-режиме нефинальные MC-блоки ужмы — до кратности 6,
            // чтобы хвостовые строки всего GEMM легли в одну полосу
            // [rows - rows%6, rows) и их покрыл хвостовой проход.
            let mut ic1 = (ic0 + mc).min(end);
            if ctx.do_silu && ic1 < end {
                ic1 = ic0 + 6 * ((ic1 - ic0) / 6);
            }
            let start = ic0;
            let end = ic1;
            // Регион по N: jc-диапазон из очереди; каждый jc — ровно
            // одному региону, поэтому C не пишется двумя воркерами.
            let jc0 = region.jc0;
            let jc1 = region.jc1;
            for p in 0..ctx.n_panels {
            let panel = &*ctx.panels.add(p);
            if panel.jc < jc0 || panel.jc >= jc1 {
                continue;
            }
            let kc = panel.kc;
            let nc = panel.nc;
            let ncs = panel.stride;
            let poff = panel.off;
            // Внутри группы jc панели идут по k: pc=0 пишет C,
            // остальные — накапливают.
            let add = panel.pc != 0;

            // JIT-ядро 6×16 (DynASM): 12 аккумуляторов, без спилов.
            // Порядок обхода BLIS: j снаружи, i внутри. Тогда B-тайл
            // (kc × 16 float = 8 КБ при kc=128) остаётся в L1, а строки A
            // стримятся из L2. Обратный порядок (i снаружи) перечитывал
            // всю панель B (128 КБ) на каждый блок строк — это и было
            // главным тормозом против OpenBLAS.
            // Для остатка строк (0..5) ниже работает intrinsics 4×16 + хвост.
            let rows6 = if let Some(jit) = crate::atom::flash::jit::jit_gemm() {
                let rows6 = start + 6 * ((end - start) / 6);
                // Последняя k-панель тайла: если включён fused-режим, JIT
                // сам добавляет bias и считает SiLU в эпилоге (второй проход
                // по C не нужен).
                let silu_here = ctx.do_silu && panel.pc + panel.kc == k;
                // Interleaved-A (OpenBLAS-зеркало): B тоже есть [kk][n],
                // 1 строка кэша A на k-шаг вместо 6. Ядро другое — 8-ариантный
                // диспетчер: laid = (scattered|interleaved), silu, add.
                let packed_a = !ctx.a_packed.is_null();
                let kern = match (packed_a, silu_here, add) {
                    (false, true, true) => jit.add_silu,
                    (false, true, false) => jit.store_silu,
                    (false, false, true) => jit.add_any,
                    (false, false, false) => jit.store_any,
                    (true, true, true) => jit.add_silu_il,
                    (true, true, false) => jit.store_silu_il,
                    (true, false, true) => jit.add_any_il,
                    (true, false, false) => jit.store_any_il,
                };
                let mut j16 = 0usize;
                while j16 + 16 <= nc {
                    let bp = pb.add(poff + j16);
                    let bp_bias = if silu_here {
                        ctx.bias.add(panel.jc + j16)
                    } else {
                        std::ptr::null()
                    };
                    let mut i6 = start;
                    while i6 < rows6 {
                        // DSR-скип: мёртвая группа — нули в C (первый pc),
                        // дальше панель пропускается.
                        if group_is_dead(rm, i6) {
                            if !add {
                                let cp0 = pc.add(i6 * cols + panel.jc + j16);
                                for r in 0..6 {
                                    std::ptr::write_bytes(cp0.add(r * cols), 0, 16);
                                }
                            }
                            i6 += 6;
                            continue;
                        }
                        // A-панель: указатель в 6-строчную группу
                        // [g][pc][r][kk], шаг строки = kc_max; иначе —
                        // исходный A с шагом k. Ядро JIT одно и то же.
                        let (ap, a_str) = if ctx.a_packed.is_null() {
                            (pa.add(i6 * k + panel.pc), k)
                        } else {
                            let kc_max = ctx.a_kc;
                            let pc_idx = panel.pc / kc_max;
                            // Interleaved-панель: панель [pc][g][kk][8], 32 B
                            // на k-шаг; указатель на панель, a_stride=1 float
                            // (=4 B: строки записаны подряд внутри записи).
                            (
                                ctx.a_packed.add(
                                    (pc_idx * ctx.groups + (i6 / 6)) * kc_max * 8,
                                ),
                                1,
                            )
                        };
                        let cp = pc.add(i6 * cols + panel.jc + j16);
                        kern(ap, bp, cp, a_str, ncs, cols, kc, bp_bias);
                        i6 += 6;
                    }
                    j16 += 16;
                }
                // Хвост колонок (<16) для 6-строчных тайлов: скалярно.
                while j16 < nc {
                    let mut i6 = start;
                    while i6 < rows6 {
                        if group_is_dead(rm, i6) {
                            if !add {
                                for r in i6..i6 + 6 {
                                    let dst = pc.add(r * cols + panel.jc + j16);
                                    *dst = 0.0;
                                }
                            }
                            i6 += 6;
                            continue;
                        }
                        for r in i6..i6 + 6 {
                            let mut s = 0.0f32;
                            for kk in 0..kc {
                                s += *pa.add(r * k + panel.pc + kk)
                                    * *pb.add(poff + kk * ncs + j16);
                            }
                            let dst = pc.add(r * cols + panel.jc + j16);
                            *dst = if add { *dst + s } else { s };
                        }
                        i6 += 6;
                    }
                    j16 += 1;
                }
                rows6
            } else {
                start
            };
            let mut i = rows6;

            let rows4 = i + 4 * ((end - i) / 4);
            while i < rows4 {
                let mut j = 0usize;
                // 16 колонок: 8 аккумуляторов, 2 загрузки B на k-шаг.
                while j + 16 <= nc {
                    let mut c00 = _mm256_setzero_ps();
                    let mut c01 = _mm256_setzero_ps();
                    let mut c10 = _mm256_setzero_ps();
                    let mut c11 = _mm256_setzero_ps();
                    let mut c20 = _mm256_setzero_ps();
                    let mut c21 = _mm256_setzero_ps();
                    let mut c30 = _mm256_setzero_ps();
                    let mut c31 = _mm256_setzero_ps();
                    let bt = pb.add(poff + j);
                    for kk in 0..kc {
                        let b0 = _mm256_loadu_ps(bt.add(kk * ncs));
                        let b1 = _mm256_loadu_ps(bt.add(kk * ncs + 8));
                        let a0 = _mm256_set1_ps(*pa.add(i * k + panel.pc + kk));
                        let a1 = _mm256_set1_ps(*pa.add((i + 1) * k + panel.pc + kk));
                        let a2 = _mm256_set1_ps(*pa.add((i + 2) * k + panel.pc + kk));
                        let a3 = _mm256_set1_ps(*pa.add((i + 3) * k + panel.pc + kk));
                        c00 = _mm256_fmadd_ps(a0, b0, c00);
                        c01 = _mm256_fmadd_ps(a0, b1, c01);
                        c10 = _mm256_fmadd_ps(a1, b0, c10);
                        c11 = _mm256_fmadd_ps(a1, b1, c11);
                        c20 = _mm256_fmadd_ps(a2, b0, c20);
                        c21 = _mm256_fmadd_ps(a2, b1, c21);
                        c30 = _mm256_fmadd_ps(a3, b0, c30);
                        c31 = _mm256_fmadd_ps(a3, b1, c31);
                    }
                    let o = i * cols + panel.jc + j;
                    if add {
                        _mm256_storeu_ps(pc.add(o), _mm256_add_ps(_mm256_loadu_ps(pc.add(o)), c00));
                        _mm256_storeu_ps(pc.add(o + 8), _mm256_add_ps(_mm256_loadu_ps(pc.add(o + 8)), c01));
                        let o1 = o + cols;
                        _mm256_storeu_ps(pc.add(o1), _mm256_add_ps(_mm256_loadu_ps(pc.add(o1)), c10));
                        _mm256_storeu_ps(pc.add(o1 + 8), _mm256_add_ps(_mm256_loadu_ps(pc.add(o1 + 8)), c11));
                        let o2 = o1 + cols;
                        _mm256_storeu_ps(pc.add(o2), _mm256_add_ps(_mm256_loadu_ps(pc.add(o2)), c20));
                        _mm256_storeu_ps(pc.add(o2 + 8), _mm256_add_ps(_mm256_loadu_ps(pc.add(o2 + 8)), c21));
                        let o3 = o2 + cols;
                        _mm256_storeu_ps(pc.add(o3), _mm256_add_ps(_mm256_loadu_ps(pc.add(o3)), c30));
                        _mm256_storeu_ps(pc.add(o3 + 8), _mm256_add_ps(_mm256_loadu_ps(pc.add(o3 + 8)), c31));
                    } else {
                        _mm256_storeu_ps(pc.add(o), c00);
                        _mm256_storeu_ps(pc.add(o + 8), c01);
                        let o1 = o + cols;
                        _mm256_storeu_ps(pc.add(o1), c10);
                        _mm256_storeu_ps(pc.add(o1 + 8), c11);
                        let o2 = o1 + cols;
                        _mm256_storeu_ps(pc.add(o2), c20);
                        _mm256_storeu_ps(pc.add(o2 + 8), c21);
                        let o3 = o2 + cols;
                        _mm256_storeu_ps(pc.add(o3), c30);
                        _mm256_storeu_ps(pc.add(o3 + 8), c31);
                    }
                    j += 16;
                }
                // 8 колонок: 4 аккумулятора, 1 загрузка B на k-шаг.
                while j + 8 <= nc {
                    let mut c0 = _mm256_setzero_ps();
                    let mut c1 = _mm256_setzero_ps();
                    let mut c2 = _mm256_setzero_ps();
                    let mut c3 = _mm256_setzero_ps();
                    // 8-колоночная полоса: смещение 0 или 8 внутри 16-тайла.
                    let bt = pb.add(poff + j);
                    for kk in 0..kc {
                        let b0 = _mm256_loadu_ps(bt.add(kk * ncs));
                        let a0 = _mm256_set1_ps(*pa.add(i * k + panel.pc + kk));
                        let a1 = _mm256_set1_ps(*pa.add((i + 1) * k + panel.pc + kk));
                        let a2 = _mm256_set1_ps(*pa.add((i + 2) * k + panel.pc + kk));
                        let a3 = _mm256_set1_ps(*pa.add((i + 3) * k + panel.pc + kk));
                        c0 = _mm256_fmadd_ps(a0, b0, c0);
                        c1 = _mm256_fmadd_ps(a1, b0, c1);
                        c2 = _mm256_fmadd_ps(a2, b0, c2);
                        c3 = _mm256_fmadd_ps(a3, b0, c3);
                    }
                    let o = i * cols + panel.jc + j;
                    if add {
                        _mm256_storeu_ps(pc.add(o), _mm256_add_ps(_mm256_loadu_ps(pc.add(o)), c0));
                        _mm256_storeu_ps(pc.add(o + cols), _mm256_add_ps(_mm256_loadu_ps(pc.add(o + cols)), c1));
                        _mm256_storeu_ps(pc.add(o + 2 * cols), _mm256_add_ps(_mm256_loadu_ps(pc.add(o + 2 * cols)), c2));
                        _mm256_storeu_ps(pc.add(o + 3 * cols), _mm256_add_ps(_mm256_loadu_ps(pc.add(o + 3 * cols)), c3));
                    } else {
                        _mm256_storeu_ps(pc.add(o), c0);
                        _mm256_storeu_ps(pc.add(o + cols), c1);
                        _mm256_storeu_ps(pc.add(o + 2 * cols), c2);
                        _mm256_storeu_ps(pc.add(o + 3 * cols), c3);
                    }
                    j += 8;
                }
                // Хвост колонок < 8: скалярно по исходному B.
                while j < nc {
                    for r in i..i + 4 {
                        let mut s = 0.0f32;
                        for kk in 0..kc {
                            s += *pa.add(r * k + panel.pc + kk)
                                * *pb.add(poff + kk * ncs + j);
                        }
                        let dst = pc.add(r * cols + panel.jc + j);
                        *dst = if add { *dst + s } else { s };
                    }
                    j += 1;
                }
                i += 4;
            }
            // Хвост строк < 4: 1-строчное AVX2-ядро (16 + 8 колонок).
            // Раньше здесь был тройной скалярный цикл: 1–3 строки на поток
            // выполнялись в десятки раз медленнее и задавали время ВСЕГО
            // GEMM (parts=6 давал 205 GF/s против 359 у parts=5 без хвоста).
            while i < end {
                let mut j = 0usize;
                while j + 16 <= nc {
                    let mut c0 = _mm256_setzero_ps();
                    let mut c1 = _mm256_setzero_ps();
                    let bt = pb.add(poff + j);
                    for kk in 0..kc {
                        let b0 = _mm256_loadu_ps(bt.add(kk * ncs));
                        let b1 = _mm256_loadu_ps(bt.add(kk * ncs + 8));
                        let a = _mm256_set1_ps(*pa.add(i * k + panel.pc + kk));
                        c0 = _mm256_fmadd_ps(a, b0, c0);
                        c1 = _mm256_fmadd_ps(a, b1, c1);
                    }
                    let o = i * cols + panel.jc + j;
                    if add {
                        _mm256_storeu_ps(
                            pc.add(o),
                            _mm256_add_ps(_mm256_loadu_ps(pc.add(o)), c0),
                        );
                        _mm256_storeu_ps(
                            pc.add(o + 8),
                            _mm256_add_ps(_mm256_loadu_ps(pc.add(o + 8)), c1),
                        );
                    } else {
                        _mm256_storeu_ps(pc.add(o), c0);
                        _mm256_storeu_ps(pc.add(o + 8), c1);
                    }
                    j += 16;
                }
                while j + 8 <= nc {
                    let mut c0 = _mm256_setzero_ps();
                    let bt = pb.add(poff + j);
                    for kk in 0..kc {
                        let b0 = _mm256_loadu_ps(bt.add(kk * ncs));
                        let a = _mm256_set1_ps(*pa.add(i * k + panel.pc + kk));
                        c0 = _mm256_fmadd_ps(a, b0, c0);
                    }
                    let o = i * cols + panel.jc + j;
                    if add {
                        _mm256_storeu_ps(
                            pc.add(o),
                            _mm256_add_ps(_mm256_loadu_ps(pc.add(o)), c0),
                        );
                    } else {
                        _mm256_storeu_ps(pc.add(o), c0);
                    }
                    j += 8;
                }
                // < 8 колонок: по одной, но B уже в кэше (на строку).
                while j < nc {
                    let mut s = 0.0f32;
                    for kk in 0..kc {
                        s += *pa.add(i * k + panel.pc + kk)
                            * *pb.add(poff + kk * ncs + j);
                    }
                    let dst = pc.add(i * cols + panel.jc + j);
                    *dst = if add { *dst + s } else { s };
                    j += 1;
                }
                i += 1;
            }
            }
            ic0 = ic1;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Те же границы, что и у x86-пути: регион очереди (r0/r1/jc).
        let jc0 = region.jc0;
        let jc1 = region.jc1;
        let a = std::slice::from_raw_parts(ctx.a, rows * k);
        let b = std::slice::from_raw_parts(ctx.b, k * cols);
        let c = std::slice::from_raw_parts_mut(ctx.c, rows * cols);
        for r in start..end {
            let dead = !ctx.row_alive.is_null()
                && *std::slice::from_raw_parts(ctx.row_alive, rows)[r] == 0;
            for j in jc0..jc1 {
                if dead {
                    c[r * cols + j] = 0.0;
                    continue;
                }
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[r * k + kk] * b[kk * cols + j];
                }
                c[r * cols + j] = s;
            }
        }
    }
    }
}

/// C = A×B: упаковка панелей B + блочное AVX2-микроядро 4×16 (регистровые
/// кластеры 6×16 по 12 YMM). `parts` — число участников fork (1 = последовательно).
pub fn gemm_blocked(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    parts: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; a_rows * cols];
    gemm_blocked_into(a, a_rows, k, b, cols, parts, &mut c);
    c
}

/// Единый авто-выбор потоков GEMM (порог 8M MAC: мелкие — последовательно,
/// т.к. пробуждение пула дороже работы; крупные — default_parts).
pub fn gemm_auto_parts(parts: usize, rows: usize, k: usize, cols: usize) -> usize {
    if parts != 0 {
        return parts;
    }
    let work = rows.saturating_mul(k).saturating_mul(cols);
    if work < 8_000_000 {
        1
    } else {
        default_parts()
    }
}

/// C = A×B с авто-потоками (zero-copy: без промежуточных Tensor/копий).
pub fn gemm_blocked_auto(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    parts: usize,
) -> Vec<f32> {
    gemm_blocked(a, a_rows, k, b, cols, gemm_auto_parts(parts, a_rows, k, cols))
}

/// То же, но результат пишется в готовый буфер `c` (например, в numpy-массив):
/// без промежуточного Vec и без лишнего копирования. `c.len() == rows*cols`.
pub fn gemm_blocked_into(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    parts: usize,
    c: &mut [f32],
) {
    gemm_blocked_impl(a, a_rows, k, b, cols, parts, c, None, None);
}

/// DSR-регионы: то же, что `gemm_blocked_into`, но мёртвые 6-строчные
/// группы (по битмаске `row_alive`, 0 = мёртвая) пропускают ядро и
/// получают нули в C. Контракт: вызывающий гейтит мёртвые строки в 0
/// (как `QairoLivingLayer::forward_gated`), поэтому вклад их весов
/// в выход пренебрежим. Dense-маска (все живые) — полный паритет.
pub fn gemm_blocked_masked_into(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    parts: usize,
    c: &mut [f32],
    row_alive: &[u8],
) {
    assert_eq!(row_alive.len(), a_rows, "gemm_blocked_masked_into: mask len");
    gemm_blocked_impl(a, a_rows, k, b, cols, parts, c, None, Some(row_alive));
}

/// Общая реализация GEMM. `bias = Some(..)` включает fused-эпилог JIT:
/// последняя k-панель каждого полного 6×16-тайла сразу пишет
/// `silu(A×B + bias)`; возвращает true, если весь C покрыт этим эпилогом
/// (тогда пост-проход по C не нужен).
#[allow(clippy::too_many_arguments)]
fn gemm_blocked_impl(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    parts: usize,
    c: &mut [f32],
    bias: Option<&[f32]>,
    row_alive: Option<&[u8]>,
) -> bool {
    let rows = a_rows;
    assert_eq!(c.len(), rows * cols, "gemm_blocked_impl: bad c.len()");
    if rows == 0 || cols == 0 || k == 0 {
        return false;
    }
    // Без AVX2/FMA x86_64-путь выполнял бы AVX2-инструкции и падал SIGILL.
    // Уходим в честный скалярный fallback (старые CPU, эмуляторы).
    #[cfg(target_arch = "x86_64")]
    if !simd::has_avx2() {
        for r in 0..rows {
            for j in 0..cols {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[r * k + kk] * b[kk * cols + j];
                }
                c[r * cols + j] = s;
            }
        }
        return false;
    }
    // Fused-эпилог применяется на полной сетке 6×16-тайлов JIT, а хвосты
    // (rows % 6, cols % 16) — одним векторизованным bias+SiLU-проходом
    // по малым полосам ниже (4×16/1×16 и скалярные хвосты пишут сырой
    // результат). Отдельного полного прохода по C больше нет.
    let fuse = bias.is_some_and(|bs| bs.len() >= cols)
        && crate::atom::flash::jit::jit_gemm().is_some();
    let parts = parts.max(1).min(rows.max(1)).min(simd::max_fork_parts());
    // Однопанельный GEMM (k ≤ GEMM_KC и cols ≤ GEMM_NC): B целиком
    // помещается в одну панель — копирование не нужно, панель off=0
    // читает строки исходного B с шагом cols (убираем pack-overhead
    // для всех «мелких» слоёв LLM: FFN, attention-проекции и т.п.).
    let single_panel = k <= gemm_dims().0 && cols <= gemm_dims().1;
    let prof = std::env::var("ATOMNET_GEMM_PROFILE").is_ok();
    let t0 = std::time::Instant::now();
    with_pack_buf(|buf| {
        let (packed, n_panels) = if single_panel {
            buf.panels.clear();
            buf.panels.push(GemmPanel {
                pc: 0,
                jc: 0,
                kc: k,
                nc: cols,
                off: 0,
                stride: cols,
            });
            (b.as_ptr(), 1usize)
        } else {
            pack_b(b, k, cols, buf, parts);
            (buf.data.as_ptr(), buf.panels.len())
        };
        // A-панель ОТКЛЮЧЕНА по умолчанию. Измерения по вариантам:
        // (1) [g/pc][r][kk] — L2 set-конфликты (конгруэнтные тайлы 16-way при
        // 8-way) + двойной трафик в записи (4096³ compute: 277 против 268 мс);
        // (2) [pc][g]-slab — та же негативная картина;
        // (3) [pc][kk][6-подряд] (OpenBLAS-зеркало, 32 B на шаг, одна строка
        // кэша A на шаг вместо шести): compute нейтрален (4096³: 311 vs 321
        // мс), но упаковка в 2× дороже широкой (scatter по tmp) — выигрыш
        // схлопывается. Scattered-A (row-major исходной матрицы) пока лучший.
        // Включить её эксперимент: ATOMNET_GEMM_PACKA=1.
        let pack_a_on = match packa_override() {
            Some(b) => b && rows >= 6,
            None => false,
        };
        let (a_packed, a_kc) = if pack_a_on {
            let akc = pack_a(a, rows, k, buf, parts);
            (buf.a_data.as_ptr(), akc)
        } else {
            (std::ptr::null(), 0)
        };
        let groups = rows.div_ceil(6);
        let (div_m, div_n) = gemm_grid(parts, rows, cols);
        // Регистровые регионы: переразбиение по строкам для ворк-стилинга.
        // OVER=1 — регионах столько же, сколько воркеров (статично);
        // OVER=2 — двойные мелкие регионы для баланса хвостов.
        let over = gemm_over();
        let regions = gemm_regions(rows, cols, div_m * over, div_n);
        let t1 = std::time::Instant::now();
        let ctx = GemmCtx {
            a: a.as_ptr(),
            b: b.as_ptr(),
            c: c.as_mut_ptr(),
            packed,
            a_packed,
            a_kc,
            groups,
            panels: buf.panels.as_ptr(),
            n_panels,
            rows,
            k,
            cols,
            parts,
            div_m,
            div_n,
            regions: regions.as_ptr(),
            n_regions: regions.len(),
            next_region: std::sync::atomic::AtomicUsize::new(0),
            row_alive: row_alive.map_or(std::ptr::null(), |m| m.as_ptr()),
            bias: bias.map_or(std::ptr::null(), |s| s.as_ptr()),
            do_silu: fuse,
        };
        simd::fork_chunks(
            parts,
            gemm_fork_job as unsafe fn(usize, usize),
            &ctx as *const GemmCtx as usize,
        );
        if prof {
            let t2 = std::time::Instant::now();
            eprintln!(
                "gemm {rows}x{k}x{cols} parts={parts} panels={n_panels}: pack {:.3} ms, compute {:.3} ms",
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3
            );
        }
    });
    // Хвостовой bias+SiLU: JIT-эпилог покрыл 6-строчные × 16-колоночные
    // тайлы; строки rows6..rows и колонки cols16..cols посчитаны сырыми
    // (4×16/1×16, скалярные хвосты) — добавляем bias и SiLU по полосам.
    // f32-точность как в остальном стеке: fast-exp ядра silu_bias не идёт
    // на этот путь? — идёт тот же JIT-ядро silu_bias, что и в fused_bias_silu.
    if fuse {
        let bias = bias.expect("fuse implies bias");
        let rows6 = rows - rows % 6;
        let cols16 = cols - cols % 16;
        if rows6 < rows {
            simd::fused_bias_silu(&mut c[rows6 * cols..], bias, rows - rows6, cols);
        }
        if cols16 < cols {
            let tail = cols - cols16;
            let btail = &bias[cols16..];
            for r in 0..rows6 {
                let s = r * cols + cols16;
                simd::fused_bias_silu(&mut c[s..s + tail], btail, 1, tail);
            }
        }
    }
    fuse
}

/// C_out = silu(A×B + bias) — GEMM с fused-эпилогом: bias прибавляется и SiLU
/// применяются сразу к результату GEMM (in-place, без второй аллокации и без
/// отдельного прохода по памяти). Для паттерна `y = silu(x @ W^T + b)` это один
/// вызов вместо gemm + bias-add + silu. `bias` индексируется по выходной
/// размерности (столбецам C).
pub fn gemm_fused_silu(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    bias: &[f32],
    parts: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; a_rows * cols];
    gemm_fused_silu_into(a, a_rows, k, b, cols, bias, parts, &mut c);
    c
}

/// Вариант `gemm_fused_silu` с записью в caller-буфер `c` (in-place).
pub fn gemm_fused_silu_into(
    a: &[f32],
    a_rows: usize,
    k: usize,
    b: &[f32],
    cols: usize,
    bias: &[f32],
    parts: usize,
    c: &mut [f32],
) {
    assert_eq!(c.len(), a_rows * cols, "gemm_fused_silu_into: bad c.len()");
    // Если JIT покрыл весь C fused-эпилогом, второй проход не нужен.
    let fused = gemm_blocked_impl(a, a_rows, k, b, cols, parts, c, Some(bias), None);
    if !fused {
        // Запасной путь: bias (на колонку, без modulo) + SiLU, векторизованный
        // 8-wide проход по строкам (fast-exp для AVX2-части, точный хвост).
        crate::atom::math::simd::fused_bias_silu(c, bias, a_rows, cols);
    }
}

/// Дефолтное число участников GEMV по размеру задачи (эмпирика):
/// маленькие — последовательно (пробуждение пула дороже работы),
/// средние (L2/L3) — несколько потоков, огромные — снова 1
/// (DRAM-полосу одно ядро уже насыщает).
pub fn gemv_default_parts(elements: usize) -> usize {
    // Бюджет ATOMNET_THREADS/affinity — потолок и для GEMV (1 поток → 1).
    let budget = crate::atom::flash::flash_hw::thread_budget();
    let cap = simd::max_fork_parts()
        .saturating_sub(1)
        .min(budget)
        .max(1);
    match elements {
        // маленькие — последовательно (пробуждение пула дороже работы),
        0..500_000 => 1,
        // средние (L2/L3) — несколько потоков,
        500_000..4_000_000 => 4.min(cap),
        // огромные — по числу CPU,
        4_000_000..32_000_000 => cap.min(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        ),
        // совсем огромные — снова 1 (DRAM-полосу одно ядро уже насыщает).
        _ => 1,
    }
}

/// Число физических ядер в affinity-маске процесса, иначе logical/2.
/// Кэшируется: парсинг /proc/cpuinfo на каждый GEMM стоил дороже
/// маленьких матриц (256³). Топология за жизнь процесса не меняется.
pub fn default_parts() -> usize {
    // Бюджет на 1 поток и дальше: ATOMNET_THREADS=1 → всегда 1
    // (пул не спавнится, привязки нет — см. flash_hw::thread_budget).
    crate::atom::flash::flash_hw::thread_budget()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::flash::flash::FlashDense;

    #[test]
    fn gemv_pooled_matches_dense() {
        let mut fd = FlashDense::new(10, 8);
        for i in 0..10 {
            fd.set_row(i, &[(i as f32 + 1.0) * 0.5; 8]);
        }
        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let y = fd.gemv(&x);
        let yp = gemv_pooled(&fd, &x, 4);
        assert_eq!(yp.len(), y.len());
        for (a, b) in y.iter().zip(yp.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn gemm_blocked_matches_matmul() {
        // A (2×4) × B (4×5) → (2×5)
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![
            1.0f32, 0.0, 0.0, 1.0, 2.0, //
            0.0, 1.0, 0.0, 0.0, 2.0, //
            2.0, 0.0, 0.0, 2.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, 0.0,
        ];
        let c = gemm_blocked(&a, 2, 4, &b, 5, 2);
        let ta = crate::atom::math::tensor::Tensor::from_vec(a.clone(), 2, 4);
        let tb = crate::atom::math::tensor::Tensor::from_vec(b.clone(), 4, 5);
        let tc = crate::atom::math::tensor::matmul(&ta, &tb);
        for (x, y) in c.iter().zip(tc.data.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn gemm_blocked_parallel_matches_sequential() {
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..100).map(|i| i as f32 * 0.02).collect();
        let s = gemm_blocked(&a, 10, 10, &b, 10, 1);
        let p = gemm_blocked(&a, 10, 10, &b, 10, 4);
        for (x, y) in s.iter().zip(p.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn gemm_blocked_masked_matches_zeroed() {
        // DSR-регионы: мёртвые 6-строчные группы → нули; живые — как dense.
        // Маска по группам: честно зануляем 2 группы из 6 (12 строк).
        let (m, k, n) = (96usize, 64usize, 48usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let dead_groups = [1usize, 9]; // группы 1 (rows 6..12) и 9 (rows 54..60)
        let mut mask = vec![1u8; m];
        for &g in &dead_groups {
            for r in g * 6..g * 6 + 6 {
                mask[r] = 0;
            }
        }
        let mut masked = vec![0.0f32; m * n];
        gemm_blocked_masked_into(&a, m, k, &b, n, 3, &mut masked, &mask);
        let dense = gemm_blocked(&a, m, k, &b, n, 3);
        for r in 0..m {
            let base = r * n;
            if mask[r] == 0 {
                for j in 0..n {
                    assert_eq!(masked[base + j], 0.0, "dead row {r} must be zero");
                }
            } else {
                for j in 0..n {
                    assert_eq!(masked[base + j], dense[base + j], "live row {r} mismatch");
                }
            }
        }
        // Dense-маска — полный паритет с обычным GEMM.
        let ones = vec![1u8; m];
        let mut full = vec![0.0f32; m * n];
        gemm_blocked_masked_into(&a, m, k, &b, n, 3, &mut full, &ones);
        assert_eq!(full, dense);
    }

    #[test]
    fn gemm_blocked_2d_grid_matches_sequential() {
        // Широкий GEMM: сетка 2D (div_n>1) — бит-в-бит как 1D.
        // cols > rows => N-split активен (jc-панели делят B, C не гонка).
        let (m, k, n) = (64usize, 256usize, 1024usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        let s = gemm_blocked(&a, m, k, &b, n, 1);
        for parts in [2usize, 3, 4, 6] {
            let p = gemm_blocked(&a, m, k, &b, n, parts);
            assert_eq!(p.len(), s.len());
            for (x, y) in s.iter().zip(p.iter()) {
                assert_eq!(x, y, "2D grid parts={parts}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn gemm_fused_silu_matches_gemm_plus_bias_silu() {
        // C_out = silu(A×B + bias): fused-эпилог должен совпадать с
        // gemm_blocked + ручной bias-add + SiLU.
        // Допуск 1e-3: AVX2-часть fused-эпилога использует fast-exp
        // (ошибка ~2e-4), хвост и scalar-путь — точный exp.
        let (m, k, n) = (37usize, 64usize, 23usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();
        let bias: Vec<f32> = (0..n).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        let out = gemm_fused_silu(&a, m, k, &b, n, &bias, 1);
        let c = gemm_blocked(&a, m, k, &b, n, 1);
        let mut expect = c;
        for i in 0..expect.len() {
            let v = expect[i] + bias[i % n];
            expect[i] = crate::atom::math::tensor::silu(v);
        }
        for (x, y) in out.iter().zip(expect.iter()) {
            assert!((x - y).abs() < 1e-3, "{x} vs {y}");
        }
    }

    #[test]
    fn gemm_fused_silu_jit_fast_path_matches() {
        let _jit = crate::atom::flash::jit::TEST_JIT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Полная сетка 6×16-тайлов (m%6==0, n%16==0) + многопанельный k:
        // включается fused-эпилог JIT, пост-прохода по C быть не должно.
        for &(m, k, n, parts) in &[
            (72usize, 300usize, 64usize, 1usize),
            (72, 300, 64, 4),
            (120, 128, 128, 3),
            (48, 64, 32, 2),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.036)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.021)
                .collect();
            let bias: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.07).collect();
            let out = gemm_fused_silu(&a, m, k, &b, n, &bias, parts);
            let c = gemm_blocked(&a, m, k, &b, n, parts);
            for r in 0..m {
                for j in 0..n {
                    let v = c[r * n + j] + bias[j];
                    let want = v / (1.0 + (-v).exp());
                    let got = out[r * n + j];
                    assert!(
                        (got - want).abs() < 1e-3,
                        "m={m} k={k} n={n} parts={parts} r={r} j={j}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_fused_silu_odd_shapes_multipanel() {
        // Некратные размеры (m%6≠0, n%16≠0) + многопанельный k + parts>1:
        // раньше fused-эпилог отключался и шёл полный пост-проход по C.
        // Теперь JIT покрывает 6×16-тайлы, хвосты — хвостовым проходом.
        let _jit = crate::atom::flash::jit::TEST_JIT_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for &(m, k, n, parts) in &[
            (101usize, 300usize, 127usize, 4usize),
            (5, 300, 17, 3),
            (1, 300, 1, 1),
            (6, 300, 16, 2),
            (1024, 300, 1023, 6),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.036)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.021)
                .collect();
            let bias: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.07).collect();
            let out = gemm_fused_silu(&a, m, k, &b, n, &bias, parts);
            let c = gemm_blocked(&a, m, k, &b, n, parts);
            for r in 0..m {
                for j in 0..n {
                    let v = c[r * n + j] + bias[j];
                    let want = v / (1.0 + (-v).exp());
                    let got = out[r * n + j];
                    assert!(
                        (got - want).abs() < 1e-3,
                        "m={m} k={k} n={n} parts={parts} r={r} j={j}: {got} vs {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_blocked_unaligned_chunks_no_oob() {
        // m=128, parts=12 → chunk=11 (не кратен 4): именно этот случай
        // раньше писал за границу C и валил heap.
        let m = 128usize;
        let k = 19usize;
        let n = 37usize;
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.2).collect();
        let s = gemm_blocked(&a, m, k, &b, n, 1);
        let p = gemm_blocked(&a, m, k, &b, n, 12);
        for (x, y) in s.iter().zip(p.iter()) {
            assert!((x - y).abs() < 1e-3, "{x} vs {y}");
        }
        let ta = crate::atom::math::tensor::Tensor::from_vec(a, m, k);
        let tb = crate::atom::math::tensor::Tensor::from_vec(b, k, n);
        let tc = crate::atom::math::tensor::matmul(&ta, &tb);
        for (x, y) in p.iter().zip(tc.data.iter()) {
            assert!((x - y).abs() < 1e-3, "{x} vs {y}");
        }
    }

    #[test]
    fn gemm_blocked_multiple_k_panels_accumulate() {
        // k=300 > GEMM_KC(128): три k-панели. Раньше панели перезаписывали C,
        // и результат содержал только последний k-блок.
        let (m, k, n) = (67usize, 300usize, 200usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32 - 8.0) * 0.03).collect();
        let ta = crate::atom::math::tensor::Tensor::from_vec(a.clone(), m, k);
        let tb = crate::atom::math::tensor::Tensor::from_vec(b.clone(), k, n);
        let tc = crate::atom::math::tensor::matmul(&ta, &tb);
        for parts in [1usize, 4, 6] {
            let c = gemm_blocked(&a, m, k, &b, n, parts);
            for (x, y) in c.iter().zip(tc.data.iter()) {
                assert!((x - y).abs() < 1e-3, "parts={parts}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn gemv_fused_silu_matches_manual() {
        let mut fd = FlashDense::new(4, 4);
        for i in 0..4 {
            fd.set_row(i, &[1.0, 2.0, 3.0, 4.0]);
            fd.bias[i] = 0.1;
        }
        let x = [1.0f32, 0.0, 0.0, 0.0];
        let y = gemv_fused_silu(&fd, &x, 1);
        let raw = 1.0f32 + 0.1;
        let expect = raw / (1.0 + (-raw).exp());
        assert!((y[0] - expect).abs() < 0.05);
    }

    /// Один прогон с параметрами из env (для perf stat):
    /// BENCH_N=1024 BENCH_PARTS=6 BENCH_JIT=1 cargo test ... bench_gemm_one -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn bench_gemm_one() {
        use std::time::Instant;
        let env_usize = |name: &str, def: usize| {
            std::env::var(name)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(def)
        };
        let n = env_usize("BENCH_N", 1024);
        let parts = env_usize("BENCH_PARTS", 6);
        let reps = env_usize("BENCH_REPS", 5);
        let jit_on = env_usize("BENCH_JIT", 1) != 0;
        let a: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut c = vec![0.0f32; n * n];
        crate::atom::flash::jit::set_jit_enabled(jit_on);
        gemm_blocked_into(&a, n, n, &b, n, parts, &mut c);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = Instant::now();
            gemm_blocked_into(&a, n, n, &b, n, parts, &mut c);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!(
            "n={n} parts={parts} jit={jit_on}: {:.2} ms, {:.1} GF/s",
            best * 1e3,
            2.0 * (n as f64).powi(3) / best / 1e9
        );
    }

    /// Чистый Rust-бенчмарк конвейера (без Python, C переиспользуется):
    /// проверяем масштабирование JIT vs intrinsics на одинаковых данных.
    /// cargo test --release --lib atom::flash::flash_gemm::tests::bench_gemm_rust -- --ignored --nocapture
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn bench_gemm_rust() {
        use std::time::Instant;
        for &n in &[512usize, 1024, 2048] {
            let a: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32 * 0.01).collect();
            let b: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.01).collect();
            let mut c = vec![0.0f32; n * n];
            for parts in [1usize, 2, 3, 6] {
                for jit_on in [true, false] {
                    crate::atom::flash::jit::set_jit_enabled(jit_on);
                    gemm_blocked_into(&a, n, n, &b, n, parts, &mut c);
                    let mut best = f64::INFINITY;
                    for _ in 0..3 {
                        let t0 = Instant::now();
                        gemm_blocked_into(&a, n, n, &b, n, parts, &mut c);
                        best = best.min(t0.elapsed().as_secs_f64());
                    }
                    println!(
                        "{n}^3 parts={parts} jit={jit_on}: {:.2} ms, {:.1} GF/s",
                        best * 1e3,
                        2.0 * (n as f64).powi(3) / best / 1e9
                    );
                }
            }
            crate::atom::flash::jit::set_jit_enabled(true);
        }
    }
}

#[cfg(test)]
mod align_bench {
    use super::*;

    /// Проверка: даёт ли 64-байтовое выравнивание C выигрыш
    /// (storeu по 32 Б при base%64=16 пересекает cache line).
    #[test]
    #[ignore]
    fn bench_aligned_c() {
        use crate::atom::flash::flash::Aligned64;
        use std::time::Instant;
        for &n in &[1024usize, 2048, 4096] {
            let a: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32 * 0.01).collect();
            let b: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.01).collect();
            let mut cv = vec![0.0f32; n * n];
            let mut ca = Aligned64::<f32>::new_zeroed(n * n);
            let pv = cv.as_ptr() as usize % 64;
            let pa = ca.as_ptr() as usize % 64;
            let mut best_v = f64::INFINITY;
            let mut best_a = f64::INFINITY;
            for _ in 0..2 {
                gemm_blocked_into(&a, n, n, &b, n, 6, &mut cv);
                gemm_blocked_into(&a, n, n, &b, n, 6, ca.as_mut_slice());
            }
            for _ in 0..6 {
                let t = Instant::now();
                gemm_blocked_into(&a, n, n, &b, n, 6, &mut cv);
                best_v = best_v.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                gemm_blocked_into(&a, n, n, &b, n, 6, ca.as_mut_slice());
                best_a = best_a.min(t.elapsed().as_secs_f64());
            }
            let f = 2.0 * (n as f64).powi(3);
            println!(
                "n={n} vec%64={pv} {:.1} GF/s | aligned%64={pa} {:.1} GF/s | gain {:.1}%",
                f / best_v / 1e9,
                f / best_a / 1e9,
                (best_v / best_a - 1.0) * 100.0
            );
        }
    }
}

#[cfg(test)]
mod pack_bench {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn bench_pack() {
        for &n in &[1024usize, 2048, 4096] {
            let b: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.01).collect();
            with_pack_buf(|buf| {
                for &parts in &[1usize, 6] {
                    pack_b(&b, n, n, buf, parts);
                    let t = Instant::now();
                    for _ in 0..10 {
                        pack_b(&b, n, n, buf, parts);
                    }
                    let ms = t.elapsed().as_secs_f64() * 100.0;
                    println!(
                        "pack {n} parts={parts}: {ms:.2} ms/buf ({:.1} GB/s)",
                        (n * n * 4) as f64 / (ms / 1e3) / 1e9
                    );
                }
            });
        }
    }
}
