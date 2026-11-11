// src/python.rs
//! Python bindings for AtomNET.
//!
//! Новая структура без atom_math:
//!
//! atomnet.math
//! atomnet.life
//! atomnet.optim
//! atomnet.flash
//! atomnet.veron
//! atomnet.hexiron
//!
//! Плюс плоский совместимый API:
//!
//! atomnet.norm_np(...)
//! atomnet.dot_np(...)
//! atomnet.PyRinVec(...)
//! и так далее.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[allow(unused_imports)]
use crate::atom::{
    flash::flash_compressed,
    flash::flash as flash_mod,
    life::{life as atom_life, qairo_life, rin_vec, xor_cursor},
    math::{atom_codec, enigma, micron, simd},
    optim::{keralin, rineta},
    physics::hexiron,
    veron,
};
use crate::atom::flash::flash_arena;

// ═══════════════════════════════════════════════════════════
// Helper
// ═══════════════════════════════════════════════════════════

fn life_state_name(state: atom_life::LifeState) -> &'static str {
    match state {
        atom_life::LifeState::Dead => "dead",
        atom_life::LifeState::Cold => "cold",
        atom_life::LifeState::Born => "born",
        atom_life::LifeState::Alive => "alive",
        atom_life::LifeState::Stable => "stable",
        atom_life::LifeState::Decaying => "decaying",
    }
}

// ═══════════════════════════════════════════════════════════
// AtomMATH: base functions
// ═══════════════════════════════════════════════════════════

#[pyfunction]
fn version() -> String {
    crate::VERSION.to_string()
}

#[pyfunction]
fn enigma_signature(w: Vec<f32>) -> u64 {
    enigma::enigma_signature(&w)
}

#[pyfunction]
fn enigma_is_alive(w: Vec<f32>) -> bool {
    enigma::enigma_is_alive(&w)
}

#[pyfunction]
fn verify_with_micron(
    original: Vec<f32>,
    quantized: Vec<i8>,
    mu: f32,
    sc: f32,
) -> (bool, u32, f32) {
    enigma::verify_with_micron(&original, &quantized, mu, sc)
}

#[pyfunction]
fn dot(a: Vec<f32>, b: Vec<f32>) -> f32 {
    simd::dot(&a, &b)
}

#[pyfunction]
fn norm(x: Vec<f32>) -> f32 {
    simd::norm(&x)
}

#[pyfunction]
fn xor_signature(x: Vec<f32>) -> u64 {
    simd::xor_signature(&x)
}

#[pyfunction]
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[pyfunction]
fn dot_np<'py>(
    a: &Bound<'py, PyArray1<f32>>,
    b: &Bound<'py, PyArray1<f32>>,
) -> PyResult<f32> {
    let ra = a.readonly();
    let rb = b.readonly();

    let asl = ra.as_slice()?;
    let bsl = rb.as_slice()?;

    if asl.len() != bsl.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "dot_np: arrays must have same length",
        ));
    }

    Ok(simd::dot(asl, bsl))
}

#[pyfunction]
fn norm_np<'py>(x: &Bound<'py, PyArray1<f32>>) -> PyResult<f32> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    Ok(simd::norm(xs))
}

#[pyfunction]
fn xor_signature_np<'py>(x: &Bound<'py, PyArray1<f32>>) -> PyResult<u64> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    Ok(simd::xor_signature(xs))
}

/// XOR-складка всех знаков массива (uint64 fold через xor_signature чанков).
#[pyfunction]
fn xor_fold_np<'py>(x: &Bound<'py, PyArray1<f32>>) -> PyResult<u64> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    Ok(simd::xor_fold(xs))
}

/// Быстрая проверка равенства через XOR-складку (bitwise, SIMD).
#[pyfunction]
fn xor_eq_np<'py>(
    a: &Bound<'py, PyArray1<f32>>,
    b: &Bound<'py, PyArray1<f32>>,
) -> PyResult<bool> {
    let ra = a.readonly();
    let rb = b.readonly();
    let asl = ra.as_slice()?;
    let bsl = rb.as_slice()?;
    Ok(simd::xor_eq_check(asl, bsl))
}

/// Упаковка знаков в биты (8 f32 → 1 байт, 8x сжатие, bitwise).
#[pyfunction]
fn pack_signs_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    let packed = simd::pack_signs(xs);
    Ok(packed.into_pyarray(py))
}

/// popcount упакованных знаков (число положительных элементов).
#[pyfunction]
fn packed_popcount_np<'py>(
    x: &Bound<'py, PyArray1<u8>>,
) -> PyResult<u32> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    Ok(simd::packed_popcount(xs))
}

/// MSE одним проходом без временных массивов.
#[pyfunction]
fn mse_np<'py>(a: &Bound<'py, PyArray1<f32>>, b: &Bound<'py, PyArray1<f32>>) -> PyResult<f32> {
    let ra = a.readonly();
    let rb = b.readonly();
    let as_ = ra.as_slice()?;
    let bs = rb.as_slice()?;
    if as_.len() != bs.len() {
        return Err(PyErr::new::<PyValueError, _>("mse_np: length mismatch"));
    }
    Ok(simd::mse(as_, bs))
}

/// (mean, var) одним проходом.
#[pyfunction]
fn mean_var_np<'py>(x: &Bound<'py, PyArray1<f32>>) -> PyResult<(f32, f32)> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    Ok(simd::mean_var(xs))
}

/// out = x·scale + shift одним FMA-проходом (без временных массивов).
#[pyfunction]
fn affine_into_np<'py>(
    x: &Bound<'py, PyArray1<f32>>,
    scale: f32,
    shift: f32,
    out: &Bound<'py, PyArray1<f32>>,
) -> PyResult<()> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    let mut wo = out.readwrite();
    let os = wo.as_slice_mut()?;
    if os.len() != xs.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "affine_into_np: length mismatch",
        ));
    }
    simd::affine_into(xs, scale, shift, os);
    Ok(())
}

/// C = A × B: A (m×k), B (k×n) → C (m×n).
/// Быстрый нуль-копий путь: блочный FLASH-GEMM пишет прямо в numpy-массив
/// (как gemm_blocked_np), без промежуточных Vec и Tensor.
/// Результат бит-идентичен gemm_blocked_np. parts=0 → авто.
#[pyfunction]
#[pyo3(signature = (a, b, parts = 0))]
fn matmul_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
    parts: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let ra = a.readonly();
    let rb = b.readonly();
    let sa = ra.shape();
    let sb = rb.shape();
    if sa.len() != 2 || sb.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "matmul_np: 2D numpy-тензоры expected",
        ));
    }
    let (m, k) = (sa[0] as usize, sa[1] as usize);
    let (k2, n) = (sb[0] as usize, sb[1] as usize);
    if k != k2 {
        return Err(PyErr::new::<PyValueError, _>(
            "matmul_np: inner dims mismatch",
        ));
    }
    let fa = ra.as_slice()?;
    let fb = rb.as_slice()?;
    let parts = gemm_auto_parts(parts, m, k, n);
    let out = if k == 0 {
        PyArray2::<f32>::zeros(py, [m, n], false)
    } else {
        unsafe { PyArray2::<f32>::new(py, [m, n], false) }
    };
    let a_ptr = fa.as_ptr() as usize;
    let b_ptr = fb.as_ptr() as usize;
    let (a_len, b_len) = (fa.len(), fb.len());
    let c_ptr = {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        dst.as_mut_ptr() as usize
    };
    py.allow_threads(move || {
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        crate::atom::flash::flash_gemm::gemm_blocked_into(a, m, k, b, n, parts, c);
    });
    Ok(out)
}

/// DSR-регионы: C = A×B с живым скипом мёртвых 6-строчных групп.
/// `mask` (m,) 1=живая/0=мёртвая; мёртвые строки → 0 в C. Dense-маска —
/// полный паритет с gemm_blocked_np. parts=0 → авто.
#[pyfunction]
#[pyo3(signature = (a, b, mask, parts = 0))]
fn gemm_blocked_masked_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
    mask: &Bound<'py, PyArray1<u8>>,
    parts: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let ra = a.readonly();
    let rb = b.readonly();
    let rm = mask.readonly();
    let sa = ra.shape();
    let sb = rb.shape();
    if sa.len() != 2 || sb.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_masked_np: 2D numpy-тензоры expected",
        ));
    }
    let (m, k) = (sa[0] as usize, sa[1] as usize);
    let (k2, n) = (sb[0] as usize, sb[1] as usize);
    if k != k2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_masked_np: inner dims mismatch",
        ));
    }
    let ma = rm.as_slice()?;
    if ma.len() != m {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_masked_np: mask len != rows",
        ));
    }
    let fa = ra.as_slice()?;
    let fb = rb.as_slice()?;
    let parts = gemm_auto_parts(parts, m, k, n);
    let out = if k == 0 {
        PyArray2::<f32>::zeros(py, [m, n], false)
    } else {
        unsafe { PyArray2::<f32>::new(py, [m, n], false) }
    };
    let a_ptr = fa.as_ptr() as usize;
    let b_ptr = fb.as_ptr() as usize;
    let mask_ptr = ma.as_ptr() as usize;
    let (a_len, b_len, mask_len) = (fa.len(), fb.len(), ma.len());
    let c_ptr = {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        dst.as_mut_ptr() as usize
    };
    py.allow_threads(move || {
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        let pm = unsafe { std::slice::from_raw_parts(mask_ptr as *const u8, mask_len) };
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        crate::atom::flash::flash_gemm::gemm_blocked_masked_into(a, m, k, b, n, parts, c, pm);
    });
    Ok(out)
}

/// Авто-раскладка GEMM по потокам: части=0 → default_parts(), мелкие задачи
/// последовательно (пробуждение пула ~0.2 мс дороже работы; порог ~8M MAC).
/// Единый авто-выбор потоков (FLASH-GEMM): мелкие — последовательно,
/// крупные — default_parts. Единая точка правды, не 3 копии порога.
fn gemm_auto_parts(parts: usize, m: usize, k: usize, n: usize) -> usize {
    crate::atom::flash::flash_gemm::gemm_auto_parts(parts, m, k, n)
}

/// C = A × B блочным AVX2-ядром FLASH-GEMM (4×8 микро-блоки, fork по строкам).
/// parts=0 → авто (число доступных ядер). Результат детерминирован.
#[pyfunction]
#[pyo3(signature = (a, b, parts = 0))]
fn gemm_blocked_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
    parts: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let ra = a.readonly();
    let rb = b.readonly();
    let sa = ra.shape();
    let sb = rb.shape();
    if sa.len() != 2 || sb.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_np: 2D numpy-тензоры expected",
        ));
    }
    let (m, k) = (sa[0] as usize, sa[1] as usize);
    let (k2, n) = (sb[0] as usize, sb[1] as usize);
    if k != k2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_np: inner dims mismatch",
        ));
    }
    let fa = ra.as_slice()?;
    let fb = rb.as_slice()?;
    let parts = gemm_auto_parts(parts, m, k, n);
    // Пишем сразу в numpy-массив: без промежуточного Vec и копии.
    // zeros() делает однопоточный memset 16 МБ под GIL, хотя при k>0
    // GEMM (pc=0) пишет КАЖДЫЙ элемент C. Берём uninitialized new():
    // все элементы будут инициализированы до возврата в Python.
    // k==0 — вырожденный случай: C обязана быть нулевой, как у NumPy.
    let out = if k == 0 {
        PyArray2::<f32>::zeros(py, [m, n], false)
    } else {
        unsafe { PyArray2::<f32>::new(py, [m, n], false) }
    };
    // Сырые указатели + длины: GIL отпускаем на время счёта, чтобы
    // не блокировать другие Python/PyTorch-потоки (и OpenBLAS).
    // Массивы живы: Bound-ссылки a/b и out на стеке до конца функции.
    let a_ptr = fa.as_ptr() as usize;
    let b_ptr = fb.as_ptr() as usize;
    let (a_len, b_len) = (fa.len(), fb.len());
    let c_ptr = {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        // 2 МБ страницы для больших C: меньше page faults/TLB-промахов
        // на первой записи (THP включён в madvise-режиме).
        if dst.len() * std::mem::size_of::<f32>() >= 8 * 1024 * 1024 {
            crate::atom::flash::flash_hw::advise_huge_pages(
                dst.as_mut_ptr() as *mut u8,
                dst.len() * std::mem::size_of::<f32>(),
            );
        }
        dst.as_mut_ptr() as usize
    };
    py.allow_threads(move || {
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        crate::atom::flash::flash_gemm::gemm_blocked_into(a, m, k, b, n, parts, c);
    });
    Ok(out)
}

/// C_out = silu(A×B + bias) — GEMM с fused-эпилогом (bias + SiLU in-place).
/// Один вызов вместо gemm_blocked + bias-add + silu. `bias` — вектор длины n
/// (размер выходной размерности), индексируется по столбцам C.
#[pyfunction]
#[pyo3(signature = (a, b, bias, parts = 0))]
fn gemm_silu_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
    bias: &Bound<'py, PyArray1<f32>>,
    parts: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let ra = a.readonly();
    let rb = b.readonly();
    let rbias = bias.readonly();
    let sa = ra.shape();
    let sb = rb.shape();
    let sbias = rbias.shape();
    if sa.len() != 2 || sb.len() != 2 || sbias.len() != 1 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_silu_np: 2D A/B and 1D bias expected",
        ));
    }
    let (m, k) = (sa[0] as usize, sa[1] as usize);
    let (k2, n) = (sb[0] as usize, sb[1] as usize);
    if k != k2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_silu_np: inner dims mismatch",
        ));
    }
    if sbias[0] as usize != n {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_silu_np: bias size must equal output dim",
        ));
    }
    let fa = ra.as_slice()?;
    let fb = rb.as_slice()?;
    let fbias = rbias.as_slice()?;
    let out = if k == 0 {
        PyArray2::<f32>::zeros(py, [m, n], false)
    } else {
        unsafe { PyArray2::<f32>::new(py, [m, n], false) }
    };
    let parts = gemm_auto_parts(parts, m, k, n);
    let a_ptr = fa.as_ptr() as usize;
    let b_ptr = fb.as_ptr() as usize;
    let bias_ptr = fbias.as_ptr() as usize;
    let (a_len, b_len, bias_len) = (fa.len(), fb.len(), fbias.len());
    let c_ptr = {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        if dst.len() * std::mem::size_of::<f32>() >= 8 * 1024 * 1024 {
            crate::atom::flash::flash_hw::advise_huge_pages(
                dst.as_mut_ptr() as *mut u8,
                dst.len() * std::mem::size_of::<f32>(),
            );
        }
        dst.as_mut_ptr() as usize
    };
    py.allow_threads(move || {
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        let bias = unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, bias_len) };
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        crate::atom::flash::flash_gemm::gemm_fused_silu_into(a, m, k, b, n, bias, parts, c);
    });
    Ok(out)
}

/// Как gemm_blocked_np, но пишет в готовый numpy-массив `out`
/// (например, в буфер PyTorch-тензора — без аллокации и копии C).
#[pyfunction]
fn gemm_blocked_out_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
    out: &Bound<'py, PyArray2<f32>>,
    parts: usize,
) -> PyResult<()> {
    use numpy::PyUntypedArrayMethods;
    let ra = a.readonly();
    let rb = b.readonly();
    let sa = ra.shape();
    let sb = rb.shape();
    let so = out.shape();
    if sa.len() != 2 || sb.len() != 2 || so.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_out_np: 2D numpy-тензоры expected",
        ));
    }
    let (m, k) = (sa[0] as usize, sa[1] as usize);
    let (k2, n) = (sb[0] as usize, sb[1] as usize);
    if k != k2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_out_np: inner dims mismatch",
        ));
    }
    if so[0] as usize != m || so[1] as usize != n {
        return Err(PyErr::new::<PyValueError, _>(
            "gemm_blocked_out_np: out shape mismatch",
        ));
    }
    let fa = ra.as_slice()?;
    let fb = rb.as_slice()?;
    let parts = gemm_auto_parts(parts, m, k, n);
    let a_ptr = fa.as_ptr() as usize;
    let b_ptr = fb.as_ptr() as usize;
    let (a_len, b_len) = (fa.len(), fb.len());
    let c_ptr = {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        if dst.len() * std::mem::size_of::<f32>() >= 8 * 1024 * 1024 {
            crate::atom::flash::flash_hw::advise_huge_pages(
                dst.as_mut_ptr() as *mut u8,
                dst.len() * std::mem::size_of::<f32>(),
            );
        }
        dst.as_mut_ptr() as usize
    };
    py.allow_threads(move || {
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        crate::atom::flash::flash_gemm::gemm_blocked_into(a, m, k, b, n, parts, c);
    });
    Ok(())
}

/// y = W × x параллельным GEMV через Veron-пул (W: m×n, x: n).
/// Включить/выключить JIT-ядра (DynASM) на лету. Для бенчмарков.
#[pyfunction]
fn set_jit_enabled(enabled: bool) {
    #[cfg(target_arch = "x86_64")]
    crate::atom::flash::jit::set_jit_enabled(enabled);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = enabled;
}

/// Доступно ли JIT-ядро на этом CPU (x86_64 + AVX2 + FMA).
#[pyfunction]
fn jit_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::atom::flash::jit::jit_gemm().is_some()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// «Холодный старт»: DynASM-компиляция ядер, запуск и pinning пула,
/// прогрев упаковки/микроядер/аллокатора маленьким GEMM. Вызывается из
/// atomnet/__init__.py, чтобы первый шаг обучения не платил за init.
#[pyfunction]
#[pyo3(signature = (threads = 0))]
fn warmup(threads: usize) -> PyResult<String> {
    let parts = if threads == 0 {
        crate::atom::flash::flash_gemm::default_parts()
    } else {
        threads.max(1)
    };
    #[cfg(target_arch = "x86_64")]
    let jit = crate::atom::flash::jit::jit_gemm().is_some();
    #[cfg(not(target_arch = "x86_64"))]
    let jit = false;
    crate::atom::math::simd::prewarm_fork(parts);
    let n = 64usize;
    let a = vec![0.01f32; n * n];
    let b = vec![0.02f32; n * n];
    let mut c = vec![0.0f32; n * n];
    crate::atom::flash::flash_gemm::gemm_blocked_into(&a, n, n, &b, n, parts, &mut c);
    Ok(format!("warmup: parts={parts}, jit={jit}"))
}

/// parts=0 → авто. Результат бит-идентичен последовательному gemv.
#[pyfunction]
#[pyo3(signature = (w, x, parts = 0))]
fn gemv_pooled_np<'py>(
    py: Python<'py>,
    w: &Bound<'py, PyArray2<f32>>,
    x: &Bound<'py, PyArray1<f32>>,
    parts: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let rw = w.readonly();
    let rx = x.readonly();
    let sw = rw.shape();
    if sw.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "gemv_pooled_np: W должен быть 2D",
        ));
    }
    let (m, n) = (sw[0] as usize, sw[1] as usize);
    let xs = rx.as_slice()?;
    if xs.len() != n {
        return Err(PyErr::new::<PyValueError, _>(
            "gemv_pooled_np: len(x) != W.cols",
        ));
    }
    let ws = rw.as_slice()?;
    let w_ptr = ws.as_ptr() as usize;
    let w_len = ws.len();
    // x копируем (он маленький), W читаем по указателю: копирование
    // m×n весов и сам GEMV идут без GIL.
    let xv = xs.to_vec();
    let parts = if parts == 0 {
        crate::atom::flash::flash_gemm::gemv_default_parts(m * n)
    } else {
        parts
    };
    let y = py.allow_threads(move || {
        let ws = unsafe { std::slice::from_raw_parts(w_ptr as *const f32, w_len) };
        let mut dense = crate::atom::flash::flash::FlashDense::new(m, n);
        for i in 0..m {
            dense.set_row(i, &ws[i * n..(i + 1) * n]);
        }
        crate::atom::flash::flash_gemm::gemv_pooled(&dense, &xv, parts)
    });
    Ok(y.into_pyarray(py))
}

/// Предупакованная матрица для инференса: W (m×n) лежит в stride-выровненном
/// flash-буфере. Создаётся один раз; gemv больше не копирует веса.
#[pyclass(name = "FlashMatrix")]
struct PyFlashMatrix {
    inner: crate::atom::flash::flash::FlashDense,
}

#[pymethods]
impl PyFlashMatrix {
    #[new]
    fn new(w: &Bound<'_, PyArray2<f32>>) -> PyResult<Self> {
        use numpy::PyUntypedArrayMethods;
        let rw = w.readonly();
        let sw = rw.shape();
        if sw.len() != 2 {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashMatrix: W должен быть 2D",
            ));
        }
        let (m, n) = (sw[0] as usize, sw[1] as usize);
        let ws = rw.as_slice()?;
        let mut dense = crate::atom::flash::flash::FlashDense::new(m, n);
        for i in 0..m {
            dense.set_row(i, &ws[i * n..(i + 1) * n]);
        }
        Ok(Self { inner: dense })
    }

    fn rows(&self) -> usize {
        self.inner.rows
    }

    fn cols(&self) -> usize {
        self.inner.cols
    }

    /// y = W × x. parts=0 → последовательно (одно ядро уже насыщает
    /// пропускную способность памяти; fork добавляет ~0.2 мс на пробуждение).
    #[pyo3(signature = (x, parts = 0))]
    fn gemv<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
        parts: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let rx = x.readonly();
        let xs = rx.as_slice()?;
        if xs.len() != self.inner.cols {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashMatrix.gemv: len(x) != W.cols",
            ));
        }
        let parts = if parts == 0 {
            crate::atom::flash::flash_gemm::gemv_default_parts(
                self.inner.rows * self.inner.cols,
            )
        } else {
            parts
        };
        let y = crate::atom::flash::flash_gemm::gemv_pooled(&self.inner, xs, parts);
        Ok(y.into_pyarray(py))
    }

    /// y = silu(W × x + b) — fused-эпилог без временного буфера.
    #[pyo3(signature = (x, parts = 0))]
    fn gemv_silu<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
        parts: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let rx = x.readonly();
        let xs = rx.as_slice()?;
        if xs.len() != self.inner.cols {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashMatrix.gemv_silu: len(x) != W.cols",
            ));
        }
        let parts = if parts == 0 {
            crate::atom::flash::flash_gemm::gemv_default_parts(
                self.inner.rows * self.inner.cols,
            )
        } else {
            parts
        };
        let y = crate::atom::flash::flash_gemm::gemv_fused_silu(&self.inner, xs, parts);
        Ok(y.into_pyarray(py))
    }
}

/// Softmax по вектору: p = softmax(x).
#[pyfunction]
fn softmax_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    let p = crate::atom::math::tensor::softmax(xs);
    Ok(p.into_pyarray(py))
}

/// Elementwise GELU: y = gelu(x).
#[pyfunction]
fn gelu_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    let y: Vec<f32> = xs.iter().map(|&v| crate::atom::math::tensor::gelu(v)).collect();
    Ok(y.into_pyarray(py))
}

/// LayerNorm: y = xhat·gamma + beta.
#[pyfunction]
fn layer_norm_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
    gamma: &Bound<'py, PyArray1<f32>>,
    beta: &Bound<'py, PyArray1<f32>>,
    eps: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let rx = x.readonly();
    let rg = gamma.readonly();
    let rb = beta.readonly();
    let xs = rx.as_slice()?;
    let gs = rg.as_slice()?;
    let bs = rb.as_slice()?;
    let y = crate::atom::math::tensor::layer_norm(xs, gs, bs, eps);
    Ok(y.into_pyarray(py))
}

/// Batched cross-entropy: logits (B×V), targets (B,) → (mean_loss, dlogits (B×V)).
#[pyfunction]
fn cross_entropy_np<'py>(
    py: Python<'py>,
    logits: &Bound<'py, PyArray2<f32>>,
    targets: &Bound<'py, PyArray1<i32>>,
) -> PyResult<(f32, Bound<'py, PyArray2<f32>>)> {
    use numpy::PyUntypedArrayMethods;
    let rlog = logits.readonly();
    let rtar = targets.readonly();
    let sl = rlog.as_slice()?;
    let st = rtar.as_slice()?;
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>(
            "cross_entropy_np: 2D logits expected",
        ));
    }
    let vocab = shape[1] as usize;
    let batch = shape[0] as usize;
    let t: Vec<u32> = st.iter().map(|&t| t.max(0) as u32).collect();
    let (loss, d) = crate::atom::math::tensor::cross_entropy(sl, &t, vocab);
    let rows: Vec<Vec<f32>> = (0..batch)
        .map(|b| d[b * vocab..(b + 1) * vocab].to_vec())
        .collect();
    Ok((loss, PyArray2::from_vec2(py, &rows).map_err(|e| {
        PyErr::new::<PyValueError, _>(format!("cross_entropy_np: {e}"))
    })?))
}

/// PyTorch-мост: numpy → torch.Tensor (torch.from_numpy), без Rust-зависимости от libtorch.
#[pyfunction]
fn torch_from_numpy<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyAnyMethods;
    let torch = py.import("torch")?;
    let from_numpy = torch.getattr("from_numpy")?;
    Ok(from_numpy.call1((a.as_any(),))?)
}

/// PyTorch-мост: torch.Tensor → numpy-массив (t.numpy(), держит только CPU-тензоры).
#[pyfunction]
fn torch_to_numpy<'py>(t: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyAnyMethods;
    Ok(t.getattr("numpy")?.call0()?)
}

/// PyTorch-мост: torch.matmul(torch.from_numpy(a), torch.from_numpy(b)) → numpy-массив.
/// Демонстрация совместимости AtomNET ↔ PyTorch: вход numpy, выход numpy, вычисления делает torch.
#[pyfunction]
fn torch_matmul_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::PyAnyMethods;
    let torch = py.import("torch")?;
    let from_numpy = torch.getattr("from_numpy")?;
    let matmul = torch.getattr("matmul")?;
    let ta = from_numpy.call1((a.as_any(),))?;
    let tb = from_numpy.call1((b.as_any(),))?;
    let out = matmul.call1((&ta, &tb))?;
    Ok(out.getattr("numpy")?.call0()?)
}

#[pyfunction]
fn spherical_normalize_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;

    let mut y = xs.to_vec();
    enigma::project_sphere(&mut y);

    Ok(y.into_pyarray(py))
}

/// Сферическая нормализация в предоставленный буфер out (без аллокаций):
/// out = x/||x||. Возвращает норму до проекции.

/// Плоский list-алиас: старые тесты вызывают atomnet.spherical_normalize(list).
#[pyfunction]
fn spherical_normalize(x: Vec<f32>) -> Vec<f32> {
    let mut y = x;
    enigma::project_sphere(&mut y);
    y
}

/// Плоский numpy-алиас (явное имя).
#[pyfunction]
fn spherical_normalize_alias<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    spherical_normalize_np(py, x)
}

#[pyfunction]
fn spherical_normalize_into_np<'py>(
    x: &Bound<'py, PyArray1<f32>>,
    out: &Bound<'py, PyArray1<f32>>,
) -> PyResult<f32> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;
    let mut wo = out.readwrite();
    let os = wo.as_slice_mut()?;
    if os.len() != xs.len() {
        return Err(PyErr::new::<PyValueError, _>(
            "spherical_normalize_into_np: length mismatch",
        ));
    }
    Ok(enigma::project_into(xs, os))
}

// ═══════════════════════════════════════════════════════════
// AtomMATH: Micron
// ═══════════════════════════════════════════════════════════

#[pyfunction]
fn quantize_int8(x: Vec<f32>) -> PyResult<(Vec<i8>, f32, f32)> {
    let (q, mu, sc) = micron::quantize_int8(&x);
    Ok((q, mu, sc))
}

#[pyfunction]
fn dequantize_int8(q: Vec<i8>, mu: f32, sc: f32) -> Vec<f32> {
    micron::dequantize_int8(&q, mu, sc)
}

#[pyfunction]
fn verify_integrity(original: Vec<f32>, quantized: Vec<i8>, mu: f32, sc: f32) -> (bool, u32) {
    micron::verify_integrity(&original, &quantized, mu, sc)
}

#[pyfunction]
fn quantize_int8_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<(Bound<'py, PyArray1<i8>>, f32, f32)> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;

    let (q, mu, sc) = micron::quantize_int8(xs);

    Ok((q.into_pyarray(py), mu, sc))
}

#[pyfunction]
fn dequantize_int8_np<'py>(
    py: Python<'py>,
    q: &Bound<'py, PyArray1<i8>>,
    mu: f32,
    sc: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let ro = q.readonly();
    let qs = ro.as_slice()?;

    let out = micron::dequantize_int8(qs, mu, sc);

    Ok(out.into_pyarray(py))
}

#[pyfunction]
fn quantize_q4_resid_np(w: &Bound<'_, PyArray1<f32>>, block: usize) -> PyResult<PyQ4Resid> {
    let rw = w.readonly();
    Ok(PyQ4Resid { inner: micron::quantize_q4_resid(rw.as_slice()?, block) })
}
#[pyfunction]
fn dequantize_q4_resid_np(q: &PyQ4Resid, out: &Bound<'_, PyArray1<f32>>) -> PyResult<()> {
    let mut wo = out.readwrite();
    micron::dequantize_q4_resid(&q.inner, wo.as_slice_mut()?);
    Ok(())
}
#[pyfunction]
fn q4_resid_bytes_np(q: &PyQ4Resid) -> usize { micron::q4_resid_bytes(&q.inner) }

#[pyclass] struct PyQ4Resid { inner: micron::Q4Resid }

// ═══════════════════════════════════════════════════════════
// AtomMATH: AtomCodec
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyAtomCompressed {
    inner: atom_codec::AtomCompressed,
}

#[pymethods]
impl PyAtomCompressed {
    fn len(&self) -> usize {
        self.inner.len
    }

    fn nucleus(&self) -> f32 {
        self.inner.nucleus
    }

    fn radius(&self) -> f32 {
        self.inner.radius
    }

    fn signature(&self) -> u64 {
        self.inner.sig
    }

    fn is_alive(&self) -> bool {
        self.inner.sig != 0 && self.inner.radius > 1e-12
    }

    /// Байт на параметр в сжатом виде (nucleus + radius + sig + уровни).
    fn bytes_per_param(&self) -> f32 {
        let c = &self.inner;
        let mut bytes = 4.0 + 4.0 + 8.0;
        bytes += c.dir_q.len() as f32;
        for lvl in &c.residual_levels {
            bytes += 4.0; // scale
            bytes += lvl.q.len() as f32;
        }
        bytes / c.len.max(1) as f32
    }

    fn decompress_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let out = atom_codec::decompress_atom(&self.inner);
        out.into_pyarray(py)
    }

    fn max_error_np<'py>(
        &self,
        original: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<f32> {
        let ro = original.readonly();
        let xs = ro.as_slice()?;

        if xs.len() != self.inner.len {
            return Err(PyErr::new::<PyValueError, _>(
                "PyAtomCompressed.max_error_np: length mismatch",
            ));
        }

        Ok(atom_codec::atom_reconstruction_error(xs, &self.inner))
    }
}

#[pyfunction]
fn compress_atom_np<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f32>>,
    bits: Vec<u8>,
) -> PyResult<Bound<'py, PyAtomCompressed>> {
    let ro = x.readonly();
    let xs = ro.as_slice()?;

    let compressed = atom_codec::compress_atom(xs, &bits);

    Bound::new(
        py,
        PyAtomCompressed {
            inner: compressed,
        },
    )
}

// ═══════════════════════════════════════════════════════════
// AtomLIFE: AtomLife
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyAtomLife {
    inner: atom_life::AtomLife,
}

#[pymethods]
impl PyAtomLife {
    #[new]
    fn new() -> Self {
        Self {
            inner: atom_life::AtomLife::new(atom_life::LifeParams::default()),
        }
    }

    fn observe(&mut self, sig: u64, energy: f32, coherence: f32) {
        self.inner.observe(sig, energy, coherence);
    }

    fn observe_np<'py>(
        &mut self,
        arr: &Bound<'py, PyArray1<f32>>,
        energy: f32,
        coherence: f32,
    ) -> PyResult<()> {
        let ro = arr.readonly();
        let xs = ro.as_slice()?;

        let sig = simd::xor_signature(xs);
        self.inner.observe(sig, energy, coherence);

        Ok(())
    }

    fn gate(&self) -> f32 {
        self.inner.gate()
    }

    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    fn is_dead(&self) -> bool {
        self.inner.is_dead()
    }

    fn is_stable(&self) -> bool {
        self.inner.is_stable()
    }

    #[getter]
    fn state(&self) -> &'static str {
        life_state_name(self.inner.state)
    }

    #[getter]
    fn age(&self) -> u32 {
        self.inner.age
    }

    #[getter]
    fn usage(&self) -> u32 {
        self.inner.usage
    }

    #[getter]
    fn energy(&self) -> f32 {
        self.inner.energy
    }

    #[getter]
    fn coherence(&self) -> f32 {
        self.inner.coherence
    }

    #[getter]
    fn stability(&self) -> f32 {
        self.inner.stability
    }

    #[getter]
    fn sig(&self) -> u64 {
        self.inner.sig
    }

    #[getter]
    fn hamming(&self) -> u32 {
        self.inner.hamming
    }
}

// ═══════════════════════════════════════════════════════════
// AtomLIFE: RinVec
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyRinVec {
    inner: rin_vec::RinVec,
}

#[pymethods]
impl PyRinVec {
    #[new]
    fn new(data: Vec<f32>) -> Self {
        Self {
            inner: rin_vec::RinVec::new(data),
        }
    }

    #[staticmethod]
    fn from_numpy(arr: &Bound<'_, PyArray1<f32>>) -> PyResult<Self> {
        let ro = arr.readonly();
        let data = ro.as_slice()?.to_vec();

        Ok(Self {
            inner: rin_vec::RinVec::new(data),
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn data_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.data.clone().into_pyarray(py)
    }

    fn norm(&self) -> f32 {
        self.inner.norm()
    }

    fn dot<'py>(&self, py: Python<'py>, other: Py<PyRinVec>) -> PyResult<f32> {
        let other_ref = other.borrow(py);
        Ok(self.inner.dot(&other_ref.inner))
    }

    fn coherence_with<'py>(
        &mut self,
        py: Python<'py>,
        other: Py<PyRinVec>,
    ) -> PyResult<f32> {
        let other_ref = other.borrow(py);
        Ok(self.inner.coherence_with(&other_ref.inner))
    }

    fn normalize(&mut self) -> f32 {
        self.inner.normalize()
    }

    fn add_scaled<'py>(
        &mut self,
        py: Python<'py>,
        other: Py<PyRinVec>,
        alpha: f32,
    ) -> PyResult<bool> {
        let other_ref = other.borrow(py);
        Ok(self.inner.add_scaled(&other_ref.inner, alpha))
    }

    fn gated_dot<'py>(&self, py: Python<'py>, other: Py<PyRinVec>) -> PyResult<f32> {
        let other_ref = other.borrow(py);
        Ok(self.inner.gated_dot(&other_ref.inner))
    }

    fn gate(&self) -> f32 {
        self.inner.gate()
    }

    fn alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn dead(&self) -> bool {
        self.inner.is_dead()
    }

    #[getter]
    fn life_state(&self) -> &'static str {
        life_state_name(self.inner.life.state)
    }

    #[getter]
    fn life_energy(&self) -> f32 {
        self.inner.life.energy
    }

    #[getter]
    fn life_coherence(&self) -> f32 {
        self.inner.life.coherence
    }

    #[getter]
    fn life_stability(&self) -> f32 {
        self.inner.life.stability
    }
}

// ═══════════════════════════════════════════════════════════
// AtomLIFE: QairoLivingLayer
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyQairoLivingLayer {
    inner: qairo_life::QairoLivingLayer,
}

#[pymethods]
impl PyQairoLivingLayer {
    #[new]
    fn new(weight_rows: Vec<Vec<f32>>, bias: Vec<f32>) -> PyResult<Self> {
        qairo_life::QairoLivingLayer::new(weight_rows, bias)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer: invalid weight_rows or bias",
                )
            })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows.len()
    }

    /// Детерминированная случайная инициализация (для тестов/демо).
    #[staticmethod]
    fn random(rows: usize, cols: usize) -> PyResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "PyQairoLivingLayer.random: rows/cols must be > 0",
            ));
        }
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut weight_rows = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state >> 40) as f32 / (1u64 << 24) as f32;
                row.push((u - 0.5) * 0.04);
            }
            weight_rows.push(row);
        }
        let bias = vec![0.0f32; rows];
        qairo_life::QairoLivingLayer::new(weight_rows, bias)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.random: failed to build layer",
                )
            })
    }

    fn forward_np<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;

        if xs.len() != self.inner.dim {
            return Err(PyErr::new::<PyValueError, _>(
                "PyQairoLivingLayer.forward_np: x length mismatch",
            ));
        }

        let y = self.inner.forward(xs);
        Ok(y.into_pyarray(py))
    }

    /// Чистый аффинный forward для list-API: y = W x + b (без гейтов).
    /// Старые тесты сравнивают именно это с NumPy linear.
    fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        self.inner.forward_linear(&x)
    }

    /// forward_gate для старых list-тестов: это гейтованный живой forward.
    fn forward_gate(&mut self, x: Vec<f32>) -> Vec<f32> {
        self.inner.forward(&x)
    }

    fn shape(&self) -> (usize, usize) {
        (self.inner.dense.rows, self.inner.dim)
    }

    fn row_error(&self, idx: usize, original: Vec<f32>) -> PyResult<f32> {
        let row = self.inner.dense.row(idx);
        if row.len() != original.len() {
            return Err(PyErr::new::<PyValueError, _>("row_error: length mismatch"));
        }
        let mut max = 0.0f32;
        for (a, b) in row.iter().zip(original.iter()) {
            max = max.max((a - b).abs());
        }
        Ok(max)
    }

    fn forward_linear_np<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;

        if xs.len() != self.inner.dim {
            return Err(PyErr::new::<PyValueError, _>(
                "PyQairoLivingLayer.forward_linear_np: x length mismatch",
            ));
        }

        let y = self.inner.forward_linear(xs);
        Ok(y.into_pyarray(py))
    }
    
        /// grad_in = W^T grad_out (backward по входу, ядро FLASH gemv_t)
    fn backward_input_np<'py>(
        &self,
        py: Python<'py>,
        grad: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = grad.readonly();
        let gs = ro.as_slice()?;
        let y = self.inner.backward_input(gs);
        Ok(y.into_pyarray(py))
    }

    fn arena_read_row_np<'py>(
        &self,
        py: Python<'py>,
        row: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(self.inner.arena.read_row(row).into_pyarray(py))
    }
    fn forward_gated_frozen_np<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;

        if xs.len() != self.inner.dim {
            return Err(PyErr::new::<PyValueError, _>(
                "PyQairoLivingLayer.forward_gated_frozen_np: x length mismatch",
            ));
        }

        let y = self.inner.forward_gated_frozen(xs);
        Ok(y.into_pyarray(py))
    }

    fn distribute_grad_np<'py>(
        &mut self,
        grad: &Bound<'py, PyArray1<f32>>,
        input_norm: f32,
    ) -> PyResult<()> {
        let ro = grad.readonly();
        let gs = ro.as_slice()?;

        if gs.len() != self.inner.rows.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyQairoLivingLayer.distribute_grad_np: grad length mismatch",
            ));
        }

        self.inner.distribute_grad(gs, input_norm);
        Ok(())
    }
    
        fn epoch_tick(&mut self) {
        self.inner.epoch_tick();
    }

    fn arena_memory_bytes(&self) -> usize {
        self.inner.arena.memory_bytes()
    }
    
        fn forward_linear_batch_np<'py>(
        &self,
        py: Python<'py>,
        x_batch: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        use numpy::PyUntypedArrayMethods;
        let rx = x_batch.readonly();
        let sh = rx.shape();
        let (batch, cols) = (sh[0], sh[1]);
        if cols != self.inner.dim {
            return Err(PyErr::new::<PyValueError, _>("forward_linear_batch_np: cols mismatch"));
        }
        let xs = rx.as_slice()?;
        let rows = self.inner.dense.rows;
        let mut ys = vec![0.0f32; batch * rows];
        for b in 0..batch {
            let y = self.inner.forward_linear(&xs[b * cols..(b + 1) * cols]);
            ys[b * rows..(b + 1) * rows].copy_from_slice(&y);
        }
        Ok(ys.into_pyarray(py).reshape([batch, rows])?)
    }

    fn row_gate(&self, idx: usize) -> PyResult<f32> {
        self.inner
            .rows
            .get(idx)
            .map(|row| row.gate())
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.row_gate: index out of range",
                )
            })
    }

    fn row_alive(&self, idx: usize) -> PyResult<bool> {
        self.inner
            .rows
            .get(idx)
            .map(|row| row.is_alive())
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.row_alive: index out of range",
                )
            })
    }

    fn row_state(&self, idx: usize) -> PyResult<&'static str> {
        self.inner
            .rows
            .get(idx)
            .map(|row| life_state_name(row.life.state))
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.row_state: index out of range",
                )
            })
    }

    fn row_energy(&self, idx: usize) -> PyResult<f32> {
        self.inner
            .rows
            .get(idx)
            .map(|row| row.life.energy)
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.row_energy: index out of range",
                )
            })
    }

    fn row_coherence(&self, idx: usize) -> PyResult<f32> {
        self.inner
            .rows
            .get(idx)
            .map(|row| row.life.coherence)
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.row_coherence: index out of range",
                )
            })
    }

    fn row_stability(&self, idx: usize) -> PyResult<f32> {
        self.inner
            .rows
            .get(idx)
            .map(|row| row.life.stability)
            .ok_or_else(|| {
                PyErr::new::<PyValueError, _>(
                    "PyQairoLivingLayer.row_stability: index out of range",
                )
            })
    }

    fn set_ram_budget(&mut self, bytes: usize) {
        self.inner.region.ram_budget = bytes;
        self.inner.region.rebalance(&mut self.inner.arena);
    }

    fn arena_is_hot(&self, row: usize) -> bool {
        self.inner.arena.is_hot(row)
    }
    
        fn sgd_step_np<'py>(
        &mut self,
        grad: &Bound<'py, PyArray2<f32>>,
        grad_bias: &Bound<'py, PyArray1<f32>>,
        lr: f32,
    ) -> PyResult<()> {
        let rg = grad.readonly();
        let rg_slice = rg.as_slice()?;
        let rb = grad_bias.readonly();
        let rb_slice = rb.as_slice()?;

        if rg_slice.len() != self.inner.dense.rows * self.inner.dense.cols {
            return Err(PyErr::new::<PyValueError, _>(
                "PyQairoLivingLayer.sgd_step_np: grad shape mismatch",
            ));
        }

        self.inner.sgd_step(rg_slice, rb_slice, lr);
        Ok(())
    }
    
        fn quantize_to_int4(&mut self) {
        self.inner.quantize_to_int4();
    }

    fn is_int4(&self) -> bool {
        self.inner.is_int4()
    }

    fn drop_int4(&mut self) {
        self.inner.drop_int4();
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn cold_start(&mut self) -> u64 {
        self.inner.cold_start()
    }

    fn cursor_recommend_kernel(&self) -> u8 {
        self.inner.cursor.recommend_kernel()
    }

    fn cursor_hot_fraction(&self) -> f32 {
        self.inner.cursor.hot_fraction()
    }

    fn cursor_prefetch_distance(&self) -> usize {
        self.inner.cursor.prefetch_distance()
    }
}

#[pyclass]
struct PyFlashArena { inner: flash_arena::FlashArena }

#[pymethods]
impl PyFlashArena {
    #[staticmethod]
    fn from_numpy(w: &Bound<'_, PyArray2<f32>>) -> PyResult<Self> {
        use numpy::PyUntypedArrayMethods;
        let rw = w.readonly();
        let sh = rw.shape();
        let (rows, cols) = (sh[0], sh[1]);
        let ws = rw.as_slice()?;
        let bias = vec![0.0f32; rows];
        Ok(Self { inner: flash_arena::FlashArena::from_dense(ws, rows, cols, bias) })
    }
    fn note_energy(&mut self, row: usize, energy: f32) { self.inner.note_energy(row, energy); }
    fn is_hot(&self, row: usize) -> bool { self.inner.is_hot(row) }
        fn release_cold(&mut self) { self.inner.release_cold(); }
    fn memory_bytes(&self) -> usize { self.inner.memory_bytes() }
    fn read_row_np<'py>(&self, py: Python<'py>, row: usize) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(self.inner.read_row(row).into_pyarray(py))
    }
        fn release_cold_xor(&mut self) { self.inner.release_cold_xor(); }

    fn xor_tiers(&self) -> Vec<u8> { self.inner.xor_tiers() }
}

// ═══════════════════════════════════════════════════════════
// AtomLIFE: XorCursor
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyXorCursor {
    inner: xor_cursor::XorCursor,
}

#[pymethods]
impl PyXorCursor {
    #[new]
    fn new() -> Self {
        Self {
            inner: xor_cursor::XorCursor::new(),
        }
    }

    fn update_np<'py>(&mut self, arr: &Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let ro = arr.readonly();
        let xs = ro.as_slice()?;
        self.inner.update(xs);
        Ok(())
    }

    fn update_from_signature(&mut self, sig: u64) {
        self.inner.update_from_signature(sig);
    }

    fn is_cold(&self) -> bool {
        self.inner.is_cold()
    }

    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn is_stable(&self) -> bool {
        self.inner.is_stable()
    }

    fn is_changing_fast(&self) -> bool {
        self.inner.is_changing_fast()
    }

    fn needs_more_precision(&self) -> bool {
        self.inner.needs_more_precision()
    }

    fn can_compress_more(&self) -> bool {
        self.inner.can_compress_more()
    }

    fn adapt_compression(&mut self, max_level: u8) {
        self.inner.adapt_compression(max_level);
    }

    fn suggested_bits(&self) -> Vec<u8> {
        self.inner.suggested_bits()
    }

    fn suggested_bits_list(&self) -> Vec<u32> {
        self.inner
            .suggested_bits()
            .into_iter()
            .map(|b| b as u32)
            .collect()
    }

    #[getter]
    fn sig(&self) -> u64 {
        self.inner.sig
    }

    #[getter]
    fn prev_sig(&self) -> u64 {
        self.inner.prev_sig
    }

    #[getter]
    fn alive_bits(&self) -> u32 {
        self.inner.alive_bits
    }

    #[getter]
    fn hamming(&self) -> u32 {
        self.inner.hamming
    }

    #[getter]
    fn steps_alive(&self) -> u32 {
        self.inner.steps_alive
    }

    #[getter]
    fn compression_level(&self) -> u8 {
        self.inner.compression_level
    }
}

// ═══════════════════════════════════════════════════════════
// AtomOPTIM: Rin
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyRin {
    inner: rineta::Rin,
}

#[pymethods]
impl PyRin {
    #[new]
    fn new(data: Vec<f32>) -> Self {
        Self {
            inner: rineta::Rin::new(data),
        }
    }

    #[staticmethod]
    fn from_numpy(arr: &Bound<'_, PyArray1<f32>>) -> PyResult<Self> {
        let ro = arr.readonly();
        let data = ro.as_slice()?.to_vec();

        Ok(Self {
            inner: rineta::Rin::new(data),
        })
    }

    fn boost(&self) -> f32 {
        self.inner.boost()
    }

    #[getter]
    fn coherence(&self) -> f32 {
        self.inner.coherence
    }

    #[getter]
    fn flip_ema(&self) -> f32 {
        self.inner.flip_ema
    }

    #[getter]
    fn energy_ema(&self) -> f32 {
        self.inner.energy_ema
    }

    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn cold_start_hamming(&self) -> u32 {
        self.inner.cold_start_hamming()
    }

    fn warmup(&mut self, noise_scale: f32) {
        self.inner.warmup(noise_scale);
    }

    fn signature(&self) -> u64 {
        self.inner.signature()
    }

    fn state_changed(&self, prev_sig: u64) -> u32 {
        self.inner.state_changed(prev_sig)
    }

    fn apply_into_inplace_np<'py>(
        &mut self,
        w: &Bound<'py, PyArray1<f32>>,
        g: &Bound<'py, PyArray1<f32>>,
        lr: f32,
        beta1: f32,
    ) -> PyResult<f32> {
        let mut w_rw = w.readwrite();
        let w_slice = w_rw.as_slice_mut()?;

        let g_ro = g.readonly();
        let g_slice = g_ro.as_slice()?;

        if w_slice.len() != g_slice.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyRin.apply_into_inplace_np: w and g must have same length",
            ));
        }

        Ok(self.inner.apply_into(w_slice, g_slice, lr, beta1))
    }

    /// List-API для старых тестов: (новое w, coherence).
    fn apply_into(&mut self, w: Vec<f32>, g: Vec<f32>, lr: f32, beta1: f32) -> (Vec<f32>, f32) {
        let mut w = w;
        let coherence = self.inner.apply_into(&mut w, &g, lr, beta1);
        (w, coherence)
    }

    fn spherical_step_inplace_np<'py>(
        &mut self,
        w: &Bound<'py, PyArray1<f32>>,
        g: &Bound<'py, PyArray1<f32>>,
        lr: f32,
        beta1: f32,
    ) -> PyResult<f32> {
        let mut w_rw = w.readwrite();
        let w_slice = w_rw.as_slice_mut()?;

        let g_ro = g.readonly();
        let g_slice = g_ro.as_slice()?;

        if w_slice.len() != g_slice.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyRin.spherical_step_inplace_np: w and g must have same length",
            ));
        }

        Ok(self.inner.spherical_step(w_slice, g_slice, lr, beta1))
    }
}

// ═══════════════════════════════════════════════════════════
// AtomOPTIM: Keralin AtomWeight
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyAtomWeight {
    inner: keralin::AtomWeight,
    params: keralin::KeralinParams,
}

#[pymethods]
impl PyAtomWeight {
    #[new]
    fn new(w: &Bound<'_, PyArray1<f32>>) -> PyResult<Self> {
        let ro = w.readonly();
        let ws = ro.as_slice()?;

        Ok(Self {
            inner: keralin::AtomWeight::from_vec(ws),
            params: keralin::KeralinParams::default(),
        })
    }

    #[staticmethod]
    fn from_list(w: Vec<f32>) -> Self {
        Self {
            inner: keralin::AtomWeight::from_vec(&w),
            params: keralin::KeralinParams::default(),
        }
    }

    fn step_np<'py>(&mut self, grad: &Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let ro = grad.readonly();
        let gs = ro.as_slice()?;

        if gs.len() != self.inner.dir.len() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyAtomWeight.step_np: grad length mismatch",
            ));
        }

        keralin::keralin_step_atom(&mut self.inner, gs, &self.params);
        Ok(())
    }

    fn reconstruct_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let out = self.inner.reconstruct();
        out.into_pyarray(py)
    }

    #[getter]
    fn nucleus(&self) -> f32 {
        self.inner.nucleus
    }

    #[getter]
    fn radius(&self) -> f32 {
        self.inner.radius
    }

    #[getter]
    fn signature(&self) -> u64 {
        self.inner.sig
    }

    fn is_alive(&self) -> bool {
        self.inner.sig.count_ones() > 10 && self.inner.radius > 1e-12
    }
}

// ═══════════════════════════════════════════════════════════
// AtomOPTIM: Keralin Graph
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyKeralinGraph {
    inner: keralin::KeralinGraph,
}

#[pymethods]
impl PyKeralinGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: keralin::KeralinGraph::new(keralin::KeralinGraphParams::default()),
        }
    }

    #[pyo3(signature = (lr, wd, beta1, warmup_noise, auto_warm, adapt=true, beta2=0.99, eps=1e-8))]
    fn configure(
        &mut self,
        lr: f32,
        wd: f32,
        beta1: f32,
        warmup_noise: f32,
        auto_warm: bool,
        adapt: bool,
        beta2: f32,
        eps: f32,
    ) {
        self.inner.params.base_lr = lr;
        self.inner.params.wd = wd;
        self.inner.params.beta1 = beta1;
        self.inner.params.warmup_noise = warmup_noise;
        self.inner.params.auto_warm = auto_warm;
        self.inner.params.adapt = adapt;
        self.inner.params.beta2 = beta2;
        self.inner.params.eps = eps;
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.inner.params.base_lr
    }

    #[getter]
    fn step_count(&self) -> u64 {
        self.inner.step_count
    }

    #[getter]
    fn warmed(&self) -> u64 {
        self.inner.warmed_total
    }

    #[getter]
    fn global_coherence(&self) -> f32 {
        self.inner.global_coherence
    }

    #[getter]
    fn graph_signature(&self) -> u64 {
        self.inner.graph_sig
    }

    fn add_node<'py>(
        &mut self,
        _py: Python<'py>,
        w: &Bound<'py, PyArray1<f32>>,
        deps: Vec<usize>,
    ) -> PyResult<usize> {
        let ro = w.readonly();
        let ws = ro.as_slice()?;
        if ws.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyKeralinGraph.add_node: empty weight",
            ));
        }
        let atom = keralin::AtomWeight::from_vec(ws);
        self.inner
            .add_node(atom, &deps)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn connect(&mut self, from: usize, to: usize) -> PyResult<()> {
        self.inner
            .connect(from, to)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }

    fn topo(&mut self) -> Vec<usize> {
        self.inner.ensure_topo();
        self.inner.topo.clone()
    }

    fn has_cycle(&mut self) -> bool {
        self.inner.has_cycle()
    }

    /// XOR cold start: прогрев мёртвых/холодных узлов. Возвращает число прогревов.
    fn cold_start(&mut self) -> usize {
        self.inner.cold_start()
    }

    fn feed_grad<'py>(&mut self, id: usize, g: &Bound<'py, PyArray1<f32>>) -> PyResult<()> {
        let ro = g.readonly();
        let gs = ro.as_slice()?;
        self.inner
            .feed_grad(id, gs)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn accumulate_grad<'py>(
        &mut self,
        id: usize,
        g: &Bound<'py, PyArray1<f32>>,
        scale: f32,
    ) -> PyResult<()> {
        let ro = g.readonly();
        let gs = ro.as_slice()?;
        self.inner
            .accumulate_grad(id, gs, scale)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    /// Пакетная подача градиентов: одна строка матрицы на узел.
    fn feed_grads_np<'py>(
        &mut self,
        ids: Vec<usize>,
        g2d: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<()> {
        let ro = g2d.readonly();
        let shape = g2d.shape();
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        if ids.len() != rows {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "PyKeralinGraph.feed_grads_np: {} ids != {} rows",
                ids.len(),
                rows
            )));
        }
        let flat = ro.as_slice()?;
        for (k, &id) in ids.iter().enumerate() {
            let node = self
                .inner
                .nodes
                .get_mut(id)
                .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))?;
            if node.grad.len() != cols {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "node {} len {} != cols {}",
                    id,
                    node.grad.len(),
                    cols
                )));
            }
            let base = k * cols;
            node.grad.copy_from_slice(&flat[base..base + cols]);
        }
        Ok(())
    }

    /// Пакетная подача с накоплением: grad += scale * g (по строкам матрицы).
    fn accumulate_grads_np<'py>(
        &mut self,
        ids: Vec<usize>,
        g2d: &Bound<'py, PyArray2<f32>>,
        scale: f32,
    ) -> PyResult<()> {
        let ro = g2d.readonly();
        let shape = g2d.shape();
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        if ids.len() != rows {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "PyKeralinGraph.accumulate_grads_np: {} ids != {} rows",
                ids.len(),
                rows
            )));
        }
        let flat = ro.as_slice()?;
        for (k, &id) in ids.iter().enumerate() {
            let node = self
                .inner
                .nodes
                .get_mut(id)
                .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))?;
            if node.grad.len() != cols {
                return Err(PyErr::new::<PyValueError, _>(format!(
                    "node {} len {} != cols {}",
                    id,
                    node.grad.len(),
                    cols
                )));
            }
            let base = k * cols;
            for j in 0..cols {
                node.grad[j] += scale * flat[base + j];
            }
        }
        Ok(())
    }

    fn zero_grads(&mut self) {
        self.inner.zero_grads();
    }

    /// Один шаг обучения в топологическом порядке. Возвращает когерентность шага.
    fn step(&mut self) -> f32 {
        self.inner.step()
    }

    fn reconstruct_np<'py>(
        &self,
        py: Python<'py>,
        id: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let node = self
            .inner
            .nodes
            .get(id)
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))?;
        let out = node.atom.reconstruct();
        Ok(out.into_pyarray(py))
    }

    /// Пакетная реконструкция: узлы -> матрица (n x d), строки по порядку ids.
    fn reconstruct_many_np<'py>(
        &self,
        py: Python<'py>,
        ids: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if ids.is_empty() {
            return Err(PyErr::new::<PyValueError, _>(
                "PyKeralinGraph.reconstruct_many_np: empty ids",
            ));
        }
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        let mut d: Option<usize> = None;
        for id in ids {
            let node = self
                .inner
                .nodes
                .get(id)
                .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))?;
            let row = node.atom.reconstruct();
            match d {
                None => d = Some(row.len()),
                Some(prev) => {
                    if row.len() != prev {
                        return Err(PyErr::new::<PyValueError, _>(
                            "PyKeralinGraph.reconstruct_many_np: nodes have different lengths",
                        ));
                    }
                }
            }
            out.push(row);
        }
        let arr = PyArray2::from_vec2(py, &out)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("reconstruct_many_np: {}", e)))?;
        Ok(arr)
    }

    fn node_signature(&self, id: usize) -> PyResult<u64> {
        self.inner
            .nodes
            .get(id)
            .map(|node| node.atom.sig)
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))
    }

    fn node_nucleus(&self, id: usize) -> PyResult<f32> {
        self.inner
            .nodes
            .get(id)
            .map(|node| node.atom.nucleus)
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))
    }

    fn node_radius(&self, id: usize) -> PyResult<f32> {
        self.inner
            .nodes
            .get(id)
            .map(|node| node.atom.radius)
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))
    }

    fn node_state(&self, id: usize) -> PyResult<&'static str> {
        self.inner
            .nodes
            .get(id)
            .map(|node| node.state_name())
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))
    }

    fn node_alive(&self, id: usize) -> PyResult<bool> {
        self.inner
            .nodes
            .get(id)
            .map(|node| !keralin::keralin_sig_dead(node.atom.sig, node.atom.dir.len()))
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no node {}", id)))
    }

    /// (все узлы живы, мёртвых, холодных)
    fn integrity(&self) -> (bool, usize, usize) {
        self.inner.integrity()
    }

    fn alive_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mut dead = 0;
        let mut cold = 0;
        let mut born = 0;
        let mut alive = 0;
        let mut stable = 0;
        let mut decaying = 0;
        for node in &self.inner.nodes {
            match node.life.state {
                atom_life::LifeState::Dead => dead += 1,
                atom_life::LifeState::Cold => cold += 1,
                atom_life::LifeState::Born => born += 1,
                atom_life::LifeState::Alive => alive += 1,
                atom_life::LifeState::Stable => stable += 1,
                atom_life::LifeState::Decaying => decaying += 1,
            }
        }
        (dead, cold, born, alive, stable, decaying)
    }
}

// ═══════════════════════════════════════════════════════════
// AtomFLASH: FlashLayer
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyFlashLayer {
    inner: flash_mod::FlashLayer,
}

#[pymethods]
impl PyFlashLayer {
    #[staticmethod]
    fn random(out_features: usize, in_features: usize, std_dev: f32) -> Self {
        Self {
            inner: flash_mod::FlashLayer::random(out_features, in_features, std_dev),
        }
    }

    #[getter]
    fn out_features(&self) -> usize {
        self.inner.dense.rows
    }

    #[getter]
    fn in_features(&self) -> usize {
        self.inner.dense.cols
    }

    fn forward_linear_np<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;

        if xs.len() != self.inner.dense.cols {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashLayer.forward_linear_np: x length mismatch",
            ));
        }

        let y = self.inner.forward_linear(xs);
        Ok(y.into_pyarray(py))
    }

    fn forward_living_np<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;

        if xs.len() != self.inner.dense.cols {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashLayer.forward_living_np: x length mismatch",
            ));
        }

        let y = self.inner.forward_living(xs);
        Ok(y.into_pyarray(py))
    }

    fn distribute_grad_np<'py>(
        &mut self,
        grad: &Bound<'py, PyArray1<f32>>,
        input_norm: f32,
    ) -> PyResult<()> {
        let ro = grad.readonly();
        let gs = ro.as_slice()?;

        if gs.len() != self.inner.dense.rows {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashLayer.distribute_grad_np: grad length mismatch",
            ));
        }

        self.inner.distribute_grad(gs, input_norm);
        Ok(())
    }

    fn state_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        self.inner.state_counts()
    }
}

#[pyclass]
struct PyFlashCompressed {
    inner: flash_compressed::FlashCompressed,
}

#[pymethods]
impl PyFlashCompressed {
    #[staticmethod]
    fn from_numpy(
        w: &Bound<'_, PyArray2<f32>>,
        bias: &Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Self> {
        let rw = w.readonly();
        let rb = bias.readonly();
        let shape = rw.shape();
        let rows = shape[0];
        let cols = shape[1];
        
        let w_slice = rw.as_slice()?;
        let b_slice = rb.as_slice()?.to_vec();
        
        Ok(Self {
            inner: flash_compressed::FlashCompressed::from_dense(
                w_slice, rows, cols, b_slice
            ),
        })
    }
    
    fn xor_cold_start(&mut self) {
        self.inner.xor_cold_start();
    }
    
    fn gemv_np<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;
        let y = self.inner.gemv(xs);
        Ok(y.into_pyarray(py))
    }
    
    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }
    
    fn row_values(&self, i: usize) -> PyResult<Vec<f32>> {
        if i >= self.inner.rows {
            return Err(PyValueError::new_err("row_values OOB"));
        }
        Ok(self.inner.row_values(i))
    }

    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }
    
    fn hot_rows(&self) -> Vec<usize> {
        self.inner.hot_rows()
    }
}

// ═══════════════════════════════════════════════════════════
// FLASH-CORE: замена BLAS
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyFlashInt16 {
    inner: flash_compressed::FlashInt16,
}

#[pymethods]
impl PyFlashInt16 {
    #[staticmethod]
    fn from_numpy(
        w: &Bound<'_, PyArray2<f32>>,
        bias: &Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Self> {
        let rw = w.readonly();
        let rb = bias.readonly();
        let shape = rw.shape();
        let rows = shape[0];
        let cols = shape[1];
        let w_slice = rw.as_slice()?;
        let b_slice = rb.as_slice()?.to_vec();
        Ok(Self {
            inner: flash_compressed::FlashInt16::from_dense(w_slice, rows, cols, b_slice),
        })
    }

    fn gemv_np<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;
        let y = self.inner.gemv(xs);
        Ok(y.into_pyarray(py))
    }

    fn gemv_skip_cold_np<'py>(
        &mut self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
        min_bits: u32,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, usize)> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;
        let (y, skipped) = self.inner.gemv_skip_cold(xs, min_bits);
        Ok((y.into_pyarray(py), skipped))
    }

    /// XOR-контур: подписи и hamming последнего вызова.
    fn last_in_signature(&self) -> u64 {
        self.inner.last_in_sig
    }

    fn last_out_signature(&self) -> u64 {
        self.inner.last_out_sig
    }

    fn last_hamming(&self) -> u32 {
        self.inner.last_hamming
    }

    fn gemv_calls(&self) -> u64 {
        self.inner.calls
    }

    /// Батч: out (B x rows) = W @ X^T, X — (B x cols).
    /// W читается один раз на весь батч (против B вызовов gemv).
    fn gemv_batch_np<'py>(
        &self,
        py: Python<'py>,
        xb: &Bound<'py, PyArray2<f32>>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let ro = xb.readonly();
        let shape = xb.shape();
        let b = shape[0] as usize;
        let cols = shape[1] as usize;
        if cols != self.inner.cols {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "PyFlashInt16.gemv_batch_np: cols {} != layer cols {}",
                cols, self.inner.cols
            )));
        }
        let xs = ro.as_slice()?;
        let y = self.inner.gemv_batch(xs, b);
        let rows = self.inner.rows;
        let mat = y
            .chunks(rows)
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f32>>>();
        Ok(PyArray2::from_vec2(py, &mat)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("gemv_batch_np: {}", e)))?)
    }

    fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    fn compression_ratio(&self) -> f32 {
        self.inner.compression_ratio()
    }

    /// XOR-подпись всей матрицы (связывание слоёв: XOR подписей строк).
    fn signature(&self) -> u64 {
        self.inner
            .row_sigs
            .iter()
            .fold(0u64, |acc, &s| acc ^ s)
    }

    fn row_signature(&self, i: usize) -> PyResult<u64> {
        self.inner
            .row_sigs
            .get(i)
            .copied()
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no row {}", i)))
    }

    /// Число «живых» бит подписи строки (XOR-живость).
    fn row_bits(&self, i: usize) -> PyResult<u32> {
        self.inner
            .row_sigs
            .get(i)
            .map(|s| s.count_ones())
            .ok_or_else(|| PyErr::new::<PyValueError, _>(format!("no row {}", i)))
    }

    /// (горячие Q16, холодные Q8, мёртвые-нулевые) — срезы по тирам.
    fn tier_counts(&self) -> (usize, usize, usize) {
        let mut h = 0;
        let mut c = 0;
        let mut d = 0;
        for &t in &self.inner.tier {
            match t {
                0 => h += 1,
                1 => c += 1,
                _ => d += 1,
            }
        }
        (h, c, d)
    }
}

#[pyclass]
struct PyFlashDense {
    inner: flash_mod::FlashDense,
}

#[pymethods]
impl PyFlashDense {
    #[staticmethod]
    fn random(rows: usize, cols: usize, std_dev: f32) -> Self {
        let mut inner = flash_mod::FlashDense::new(rows, cols);
        // простая инициализация
        let mut rng_state: u64 = 0x1234_5678_9ABC_DEF0;
        for i in 0..rows {
            let row: Vec<f32> = (0..cols)
                .map(|_| {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let f = (rng_state >> 40) as f32 / (1u64 << 24) as f32;
                    f * 2.0 * std_dev - std_dev
                })
                .collect();
            inner.set_row(i, &row);
        }
        Self { inner }
    }

    #[staticmethod]
    fn from_numpy(
        w: &Bound<'_, PyArray2<f32>>,
        bias: &Bound<'_, PyArray1<f32>>,
    ) -> PyResult<Self> {
        let rw = w.readonly();
        let rb = bias.readonly();

        let shape = rw.shape();
        let rows = shape[0];
        let cols = shape[1];

        let w_slice = rw.as_slice()?;
        let b_slice = rb.as_slice()?;

        if b_slice.len() != rows {
            return Err(PyErr::new::<PyValueError, _>(
                "PyFlashDense.from_numpy: bias length != rows",
            ));
        }

        let mut inner = flash_mod::FlashDense::new(rows, cols);
        inner.bias = b_slice.to_vec();

        for i in 0..rows {
            let o = i * cols;
            inner.set_row(i, &w_slice[o..o + cols]);
        }

        Ok(Self { inner })
    }

    fn gemv_np<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;
        let y = self.inner.gemv(xs);
        Ok(y.into_pyarray(py))
    }

    fn gemv_bias_np<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = x.readonly();
        let xs = ro.as_slice()?;
        let y = self.inner.gemv_bias(xs);
        Ok(y.into_pyarray(py))
    }

    fn gemv_t_np<'py>(
        &self,
        py: Python<'py>,
        g: &Bound<'py, PyArray1<f32>>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let ro = g.readonly();
        let gs = ro.as_slice()?;
        let y = self.inner.gemv_t(gs);
        Ok(y.into_pyarray(py))
    }
    
        fn set_prefetch(&mut self, n: usize) {
        self.inner.set_prefetch(n);
    }
}

// ═══════════════════════════════════════════════════════════
// Veron
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyVeronRuntime {
    inner: veron::VeronRuntime,
}

/// Veron: установить число потоков параллелизма (глобально).
/// 0 = авто (физические ядра). Влияет на: Veron-fork редукции
/// (dot/norm/mse/l2), gemv по строкам. Детерминизм сохраняется.
#[pyfunction]
fn set_parallel_threads(n: usize) {
    veron::set_parallel_threads(n);
}

/// Veron: текущее число потоков параллелизма (авто, если не задано).
#[pyfunction]
fn parallel_threads() -> usize {
    veron::parallel_threads()
}

/// Автонастройка без ручных env: потоки = физм. ядра, Rayon-пул,
/// прогрев DynASM JIT + fork-пула + маленького GEMM. Вызывается
/// из python/atomnet/__init__.py при импорте, чтобы первый вызов
/// не платил за cold start. Никаких настроек не требуется.
#[pyfunction]
#[pyo3(signature = (threads = 0))]
fn configure(threads: usize) -> String {
    let t = if threads == 0 {
        crate::atom::flash::flash_hw::recommend_threads()
    } else {
        threads.max(1)
    };
    veron::set_parallel_threads(t);
    // Rayon global pool (строится один раз за процесс; ошибка "already
    // initialized" — нормально, пул уже есть).
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .build_global();
    #[cfg(target_arch = "x86_64")]
    let jit = crate::atom::flash::jit::jit_gemm().is_some();
    #[cfg(not(target_arch = "x86_64"))]
    let jit = false;
    crate::atom::math::simd::prewarm_fork(t);
    let n = 64usize;
    let a = vec![0.01f32; n * n];
    let b = vec![0.02f32; n * n];
    let mut c = vec![0.0f32; n * n];
    crate::atom::flash::flash_gemm::gemm_blocked_into(&a, n, n, &b, n, t, &mut c);
    format!(
        "configure: threads={}, jit={}, tier={}",
        t,
        jit,
        crate::atom::flash::flash_hw::kernel_tier()
    )
}

#[pymethods]
impl PyVeronRuntime {
    #[staticmethod]
    fn from_yaml(yaml: &str) -> PyResult<Self> {
        veron::VeronRuntime::from_yaml(yaml)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn to_yaml(&self) -> PyResult<String> {
        self.inner
            .to_yaml()
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn apply_thread_pool(&self) -> PyResult<()> {
        self.inner
            .apply_thread_pool()
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn summary(&self) -> String {
        self.inner.summary()
    }

    #[staticmethod]
    fn default_config(name: String) -> Self {
        Self {
            inner: veron::VeronRuntime::default_config(name),
        }
    }

    #[getter]
    fn model_name(&self) -> String {
        self.inner.model_name()
    }

    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    fn set_threads(&mut self, threads: usize) {
        self.inner.set_threads(threads);
    }

    #[getter]
    fn layer_count(&self) -> usize {
        self.inner.layer_count()
    }

    fn layers_info(&self) -> Vec<(String, String, usize, usize)> {
        self.inner.layers_info()
    }
}

// ═══════════════════════════════════════════════════════════
// Hexiron
// ═══════════════════════════════════════════════════════════

#[pyclass]
struct PyHexiron {
    inner: hexiron::Hexiron,
}

#[pymethods]
impl PyHexiron {
    #[new]
    fn new() -> Self {
        Self {
            inner: hexiron::Hexiron::new(),
        }
    }

    fn update(
        &mut self,
        energy: f32,
        coherence: f32,
        entropy: f32,
        temperature: f32,
    ) {
        self.inner.update(energy, coherence, entropy, temperature);
    }

    fn effective_lr(&self, base_lr: f32) -> f32 {
        self.inner.effective_lr(base_lr)
    }

    fn stability(&self) -> f32 {
        self.inner.stability()
    }

    fn vertices(&self) -> Vec<f32> {
        self.inner.vertices().to_vec()
    }

    fn state(&self) -> (f32, f32, f32, f32, f32, f32, bool) {
        (
            self.inner.state.energy,
            self.inner.state.coherence,
            self.inner.state.entropy,
            self.inner.state.temperature,
            self.inner.state.dissipation,
            self.inner.state.lifetime,
            self.inner.state.stable,
        )
    }

    #[getter]
    fn energy(&self) -> f32 {
        self.inner.state.energy
    }

    #[getter]
    fn coherence(&self) -> f32 {
        self.inner.state.coherence
    }

    #[getter]
    fn entropy(&self) -> f32 {
        self.inner.state.entropy
    }

    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.state.temperature
    }

    #[getter]
    fn dissipation(&self) -> f32 {
        self.inner.state.dissipation
    }

    #[getter]
    fn lifetime(&self) -> f32 {
        self.inner.state.lifetime
    }

    #[getter]
    fn stable(&self) -> bool {
        self.inner.state.stable
    }

    fn set_lifetime_bounds(&mut self, min: f32, max: f32) {
        self.inner.params.lifetime_min = min;
        self.inner.params.lifetime_max = max;
    }
}

// ═══════════════════════════════════════════════════════════
// Registration
// ═══════════════════════════════════════════════════════════


#[pyfunction]
fn sphere_normalize(v: Vec<f32>) -> Vec<f32> {
    crate::atom::math::atommath::normalize(&v)
}

#[pyfunction]
fn atommath_tangent_project(g: Vec<f32>, u: Vec<f32>) -> Vec<f32> {
    crate::atom::math::atommath::tangent_project(&g, &u)
}

#[pyfunction]
fn atommath_geodesic_step(u: Vec<f32>, g: Vec<f32>, lr: f64) -> Vec<f32> {
    crate::atom::math::atommath::geodesic_step(&u, &g, lr as f32)
}

#[pyfunction]
fn atommath_householder_reflect(x: Vec<f32>, u: Vec<f32>, v: Vec<f32>) -> Vec<f32> {
    crate::atom::math::atommath::householder_reflect(&x, &u, &v)
}

#[pyfunction]
fn atommath_shell_quantize(u: Vec<f32>, m: usize, k: usize) -> (usize, f64) {
    let (i, d) = crate::atom::math::atommath::shell_quantize(&u, m, k);
    (i, d as f64)
}

#[pyfunction]
fn atommath_decompose(w: Vec<f32>) -> (f64, f64, Vec<f32>) {
    let (n, r, d) = crate::atom::math::atommath::decompose(&w);
    (n as f64, r as f64, d)
}

#[pyfunction]
fn atommath_reconstruct(n: f64, r: f64, u: Vec<f32>) -> Vec<f32> {
    crate::atom::math::atommath::reconstruct(n as f32, r as f32, &u)
}


#[pyfunction]
fn atommath_dot_jit(a: Vec<f32>, b: Vec<f32>) -> f32 {
    crate::atom::math::atomjit::dot_jit(&a, &b)
}

#[pyfunction]
fn atommath_norm_jit(v: Vec<f32>) -> f32 {
    crate::atom::math::atomjit::norm_jit(&v)
}

#[pyfunction]
fn atommath_axpy_jit(coef: f64, x: Vec<f32>, y: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let t = any_to_tensor(y)?;
    let mut out = t.data.clone();
    crate::atom::math::atomjit::axpy_jit(coef as f32, &x, &mut out);
    Ok(PyTensor {
        inner: crate::atom::torchx::tx(out, t.shape.clone(), t.requires_grad.get()),
    })
}

#[pyfunction]
fn atommath_tangent_project_jit(g: Vec<f32>, u: Vec<f32>) -> Vec<f32> {
    crate::atom::math::atomjit::tangent_project_jit(&g, &u)
}

#[pyfunction]
fn atommath_geodesic_step_jit(u: Vec<f32>, g: Vec<f32>, lr: f64) -> Vec<f32> {
    crate::atom::math::atomjit::geodesic_step_jit(&u, &g, lr as f32)
}

#[pyfunction]
fn atommath_householder_jit(x: Vec<f32>, u: Vec<f32>, v: Vec<f32>) -> Vec<f32> {
    crate::atom::math::atomjit::householder_reflect_jit(&x, &u, &v)
}

#[pyfunction]
fn reg_pop(a: Vec<u8>) -> u32 {
    crate::atom::math::regnum::reg_pop(&a)
}

#[pyfunction]
fn reg_exchange(a: Vec<u8>, b: Vec<u8>) -> Vec<u8> {
    crate::atom::math::regnum::reg_exchange(&a, &b)
}

#[pyfunction]
fn reg_signature_nonzero(a: Vec<u8>) -> bool {
    crate::atom::math::regnum::reg_signature_nonzero(&a)
}

fn register_math(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(enigma_signature, m)?)?;
    m.add_function(wrap_pyfunction!(enigma_is_alive, m)?)?;
    m.add_function(wrap_pyfunction!(verify_with_micron, m)?)?;

    m.add_function(wrap_pyfunction!(dot, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(xor_signature, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance, m)?)?;

    m.add_function(wrap_pyfunction!(dot_np, m)?)?;
    m.add_function(wrap_pyfunction!(norm_np, m)?)?;
    m.add_function(wrap_pyfunction!(xor_signature_np, m)?)?;
    m.add_function(wrap_pyfunction!(xor_fold_np, m)?)?;
    m.add_function(wrap_pyfunction!(xor_eq_np, m)?)?;
    m.add_function(wrap_pyfunction!(pack_signs_np, m)?)?;
    m.add_function(wrap_pyfunction!(packed_popcount_np, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(reg_pop, m)?)?;
    m.add_function(wrap_pyfunction!(reg_exchange, m)?)?;
    m.add_function(wrap_pyfunction!(reg_signature_nonzero, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_normalize_alias, m)?)?;
    m.add_function(wrap_pyfunction!(spherical_normalize_np, m)?)?;
    m.add_function(wrap_pyfunction!(sphere_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_tangent_project, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_geodesic_step, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_householder_reflect, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_shell_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_decompose, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_reconstruct, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_dot_jit, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_norm_jit, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_axpy_jit, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_tangent_project_jit, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_geodesic_step_jit, m)?)?;
    m.add_function(wrap_pyfunction!(atommath_householder_jit, m)?)?;


    m.add_function(wrap_pyfunction!(spherical_normalize_into_np, m)?)?;
    m.add_function(wrap_pyfunction!(mse_np, m)?)?;
    m.add_function(wrap_pyfunction!(mean_var_np, m)?)?;
    m.add_function(wrap_pyfunction!(affine_into_np, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_np, m)?)?;
    m.add_function(wrap_pyfunction!(gemm_blocked_np, m)?)?;
    m.add_function(wrap_pyfunction!(gemm_blocked_masked_np, m)?)?;
    m.add_function(wrap_pyfunction!(gemm_silu_np, m)?)?;
    m.add_function(wrap_pyfunction!(gemm_blocked_out_np, m)?)?;
    m.add_function(wrap_pyfunction!(gemv_pooled_np, m)?)?;
    m.add_function(wrap_pyfunction!(set_jit_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(jit_available, m)?)?;
    m.add_function(wrap_pyfunction!(warmup, m)?)?;
    m.add_class::<PyFlashMatrix>()?;
    m.add_function(wrap_pyfunction!(softmax_np, m)?)?;
    m.add_function(wrap_pyfunction!(gelu_np, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_np, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_np, m)?)?;
    m.add_function(wrap_pyfunction!(torch_from_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(torch_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(torch_matmul_np, m)?)?;

    m.add_function(wrap_pyfunction!(quantize_int8, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8, m)?)?;
    m.add_function(wrap_pyfunction!(verify_integrity, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_int8_np, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_int8_np, m)?)?;

    m.add_function(wrap_pyfunction!(compress_atom_np, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_q4_resid_np, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_q4_resid_np, m)?)?;
    m.add_function(wrap_pyfunction!(q4_resid_bytes_np, m)?)?;
    m.add_class::<PyQ4Resid>()?;
    m.add_class::<PyAtomCompressed>()?;

    Ok(())
}

fn register_life(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAtomLife>()?;
    m.add_class::<PyRinVec>()?;
    m.add_class::<PyQairoLivingLayer>()?;
    m.add_class::<PyXorCursor>()?;

    Ok(())
}

fn register_optim(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRin>()?;
    m.add_class::<PyAtomWeight>()?;
    m.add_class::<PyKeralinGraph>()?;

    Ok(())
}

fn register_flash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlashLayer>()?;
    m.add_class::<PyFlashDense>()?;
    m.add_class::<PyFlashCompressed>()?;
    m.add_class::<PyFlashInt16>()?;
    m.add_class::<PyFlashArena>()?;// ← Э
    Ok(())
}

fn register_veron(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVeronRuntime>()?;
    m.add_function(wrap_pyfunction!(set_parallel_threads, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_threads, m)?)?;
    m.add_function(wrap_pyfunction!(configure, m)?)?;

    Ok(())
}

fn register_hexiron(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHexiron>()?;

    Ok(())
}

fn register_npu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(npu_dot, m)?)?;
    m.add_function(wrap_pyfunction!(npu_matvec, m)?)?;
    m.add_function(wrap_pyfunction!(npu_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(npu_dot_np, m)?)?;
    m.add_function(wrap_pyfunction!(npu_matvec_np, m)?)?;
    m.add_function(wrap_pyfunction!(npu_matmul_np, m)?)?;
    m.add_function(wrap_pyfunction!(npu_status, m)?)?;
    m.add_function(wrap_pyfunction!(npu_profile, m)?)?;
    m.add_function(wrap_pyfunction!(npu_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(virtual_isa, m)?)?;
    m.add_function(wrap_pyfunction!(npu_set_cores, m)?)?;
    m.add_function(wrap_pyfunction!(npu_distribute, m)?)?;
    Ok(())
}

/// TransformerLM как гражданин .so: модель создаётся и живёт в библиотеке.
#[pyclass(name = "Tensor", subclass, unsendable)]
struct PyTensor {
    inner: crate::atom::torchx::Tx,
}

impl Clone for PyTensor {
    fn clone(&self) -> Self {
        PyTensor {
            inner: self.inner.clone(),
        }
    }
}

fn pterralanae(x: &crate::atom::torchx::Tx) -> PyTensor {
    PyTensor { inner: x.clone() }
}

fn any_to_tensor(o: &Bound<'_, PyAny>) -> PyResult<crate::atom::torchx::Tx> {
    if let Ok(t) = o.downcast::<PyTensor>() {
        return Ok(t.borrow().inner.clone());
    }
    if let Ok(v) = o.extract::<f64>() {
        return Ok(crate::atom::torchx::tx(vec![v as f32], vec![], false));
    }
    Err(PyValueError::new_err("ожидался Tensor или число"))
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (data, shape=None, requires_grad=false))]
    fn new(data: Vec<f32>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let inner = crate::atom::torchx::tensor_from(data, shape, requires_grad)
            .map_err(PyValueError::new_err)?;
        Ok(PyTensor { inner })
    }

    fn __repr__(&self) -> String {
        let mut s = format!("Tensor(shape={:?}, dtype=float32", self.inner.shape);
        if self.inner.requires_grad.get() {
            s.push_str(", requires_grad=True");
        }
        if let Some(op) = self.inner.op_name.borrow().as_ref() {
            s.push_str(&format!(", grad_fn=<{}>", op));
        }
        s.push(')');
        s
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    #[getter]
    fn numel(&self) -> usize {
        self.inner.data.len()
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad.get()
    }

    #[getter]
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad.borrow().as_ref().map(|g| PyTensor {
            inner: crate::atom::torchx::tx(g.clone(), self.inner.shape.clone(), false),
        })
    }

    #[setter]
    fn set_grad(&self, g: Option<PyTensor>) {
        *self.inner.grad.borrow_mut() = g.map(|t| t.inner.data.clone());
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.data.clone()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_vec(py, self.inner.data.clone())
    }

    fn item(&self) -> PyResult<f64> {
        if self.inner.data.len() != 1 {
            return Err(PyValueError::new_err("только 1 элемент (tensor.item())"));
        }
        Ok(self.inner.data[0] as f64)
    }

    fn clone(&self) -> PyTensor {
        pterralanae(&self.inner)
    }

    fn detach(&self) -> PyTensor {
        PyTensor {
            inner: crate::atom::torchx::tx(self.inner.data.clone(), self.inner.shape.clone(), false),
        }
    }

    fn requires_grad_(&self, flag: bool) -> PyResult<()> {
        if self.inner.grad_fn.borrow().is_some() {
            return Err(PyValueError::new_err(
                "только leaf-тензоры могут менять requires_grad",
            ));
        }
        self.inner.requires_grad.set(flag);
        Ok(())
    }

    fn retain_grad(&self) {
        *self.inner.retained.borrow_mut() = true;
    }

    #[pyo3(signature = (gradient=None))]
    fn backward(&self, gradient: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let g: Option<Vec<f32>> = match gradient {
            None => None,
            Some(o) => Some(any_to_tensor(o)?.data.clone()),
        };
        crate::atom::torchx::backward(&self.inner, g.as_deref()).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (set_to_none=true))]
    fn zero_grad(&self, set_to_none: bool) {
        crate::atom::torchx::zero_grad(&self.inner, set_to_none);
    }

    fn add(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = any_to_tensor(other)?;
        Ok(pterralanae(&crate::atom::torchx::add_op(&self.inner, &b).map_err(PyValueError::new_err)?))
    }

    fn sub(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = any_to_tensor(other)?;
        Ok(pterralanae(&crate::atom::torchx::sub_op(&self.inner, &b).map_err(PyValueError::new_err)?))
    }

    fn mul(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = any_to_tensor(other)?;
        Ok(pterralanae(&crate::atom::torchx::mul_op(&self.inner, &b).map_err(PyValueError::new_err)?))
    }

    fn div(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = any_to_tensor(other)?;
        Ok(pterralanae(&crate::atom::torchx::div_op(&self.inner, &b).map_err(PyValueError::new_err)?))
    }

    fn __add__(&self, other: Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.add(&other)
    }

    fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.sub(&other)
    }

    fn __mul__(&self, other: Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.mul(&other)
    }

    fn __truediv__(&self, other: Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self.div(&other)
    }

    fn __neg__(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::neg_op(&self.inner))
    }

    fn abs(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::abs_op(&self.inner))
    }

    fn relu(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::relu_op(&self.inner))
    }

    fn sigmoid(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::sigmoid_op(&self.inner))
    }

    fn tanh(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::tanh_op(&self.inner))
    }

    fn silu(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::silu_op(&self.inner))
    }

    fn gelu(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::gelu_op(&self.inner))
    }

    fn exp(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::exp_op(&self.inner))
    }

    fn log(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::log_op(&self.inner))
    }

    fn sqrt(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::sqrt_op(&self.inner))
    }

    fn matmul(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = any_to_tensor(other)?;
        Ok(pterralanae(&crate::atom::torchx::matmul_op(&self.inner, &b).map_err(PyValueError::new_err)?))
    }

    fn t(&self) -> PyResult<PyTensor> {
        Ok(pterralanae(&crate::atom::torchx::transpose_op(&self.inner).map_err(PyValueError::new_err)?))
    }

    fn transpose(&self, d0: usize, d1: usize) -> PyResult<PyTensor> {
        if d0 == 0 && d1 == 1 {
            return self.t();
        }
        Err(PyValueError::new_err("transpose: только 2D (d0=0,d1=1) поддерживается"))
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        Ok(pterralanae(&crate::atom::torchx::reshape_op(&self.inner, shape).map_err(PyValueError::new_err)?))
    }

    fn view(&self, shape: Vec<usize>) -> PyResult<PyTensor> {
        self.reshape(shape)
    }

    fn unsqueeze(&self, dim: i64) -> PyResult<PyTensor> {
        Ok(pterralanae(&crate::atom::torchx::unsqueeze_op(&self.inner, dim).map_err(PyValueError::new_err)?))
    }

    fn squeeze(&self) -> PyResult<PyTensor> {
        Ok(pterralanae(&crate::atom::torchx::squeeze_op(&self.inner).map_err(PyValueError::new_err)?))
    }

    fn sum(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::sum_row_op(&self.inner))
    }

    fn mean(&self) -> PyTensor {
        pterralanae(&crate::atom::torchx::mean_row_op(&self.inner))
    }

    fn softmax(&self) -> PyResult<PyTensor> {
        Ok(pterralanae(&crate::atom::torchx::softmax_op(&self.inner).map_err(PyValueError::new_err)?))
    }

    // ── in-place (без autograd, как torch: на leaf с requires_grad — ошибка) ──
    fn _check_leaf(&self) -> PyResult<()> {
        if self.inner.requires_grad.get() && self.inner.grad_fn.borrow().is_none() {
            return Err(PyValueError::new_err(
                "a leaf Variable that requires grad is being used in an in-place operation.",
            ));
        }
        Ok(())
    }

    fn add_(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self._check_leaf()?;
        let b = any_to_tensor(other)?;
        let s = crate::atom::torchx::broadcast_shape(&self.inner.shape, &b.shape)
            .map_err(PyValueError::new_err)?;
        let be = crate::atom::torchx::expand(&b.data, &b.shape, &s);
        // in-place: сумма broadcast(b) + self (self данных уже в нужной форме при равном shape;
        // при разном shape torch делает результат в форме broadcast — упростим: равенство форм)
        if s != self.inner.shape {
            return Err(PyValueError::new_err("in-place: форма результата отличается"));
        }
        crate::atom::torchx::mutate_data(&self.inner, |d| {
            for (i, v) in d.iter_mut().enumerate() {
                *v += be[i];
            }
        });
        Ok(pterralanae(&self.inner))
    }

    fn mul_(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        self._check_leaf()?;
        let b = any_to_tensor(other)?;
        let be = crate::atom::torchx::expand(&b.data, &b.shape, &self.inner.shape);
        crate::atom::torchx::mutate_data(&self.inner, |d| {
            for (i, v) in d.iter_mut().enumerate() {
                *v *= be[i];
            }
        });
        Ok(pterralanae(&self.inner))
    }

    fn copy_(&self, other: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        let b = any_to_tensor(other)?;
        if b.data.len() != self.inner.data.len() {
            return Err(PyValueError::new_err(
                "copy_: число элементов не совпадает с целевым тензором",
            ));
        }
        let be = crate::atom::torchx::expand(&b.data, &b.shape, &self.inner.shape);
        crate::atom::torchx::mutate_data(&self.inner, |d| {
            d.copy_from_slice(&be);
        });
        Ok(pterralanae(&self.inner))
    }

    fn fill_(&self, val: f64) -> PyTensor {
        crate::atom::torchx::mutate_data(&self.inner, |d| {
            for v in d.iter_mut() {
                *v = val as f32;
            }
        });
        pterralanae(&self.inner)
    }

    fn zero_(&self) -> PyTensor {
        self.fill_(0.0)
    }
}

// ═══════════════════════════════════════════════════════════════
// torchx free functions
// ═══════════════════════════════════════════════════════════════

#[pyfunction]
fn manual_seed(seed: u64) {
    crate::atom::torchx::manual_seed(seed);
}

#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_zeros(shape))
}

#[pyfunction]
fn ones(shape: Vec<usize>) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_ones(shape))
}

#[pyfunction(name = "rand")]
fn rand_py(shape: Vec<usize>) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_rand(shape))
}

#[pyfunction]
fn randn(shape: Vec<usize>) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_randn(shape))
}

#[pyfunction]
fn randint(low: i64, high: i64, shape: Vec<usize>) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_randint(low, high, shape))
}

#[pyfunction]
fn arange(n: usize) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_arange(n))
}

#[pyfunction]
fn full(shape: Vec<usize>, val: f64) -> PyTensor {
    pterralanae(&crate::atom::torchx::factory_full(shape, val as f32))
}

#[pyfunction]
#[pyo3(signature = (data, shape=None, requires_grad=false))]
fn tensor(data: Vec<f32>, shape: Option<Vec<usize>>, requires_grad: bool) -> PyResult<PyTensor> {
    let shape = shape.unwrap_or_else(|| vec![data.len()]);
    Ok(PyTensor {
        inner: crate::atom::torchx::tensor_from(data, shape, requires_grad)
            .map_err(PyValueError::new_err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0))]
fn cross_entropy(
    input: &Bound<'_, PyAny>,
    target: Vec<i64>,
    weight: Option<Vec<f32>>,
    ignore_index: i64,
    reduction: &str,
    label_smoothing: f64,
) -> PyResult<PyTensor> {
    let inp = any_to_tensor(input)?;
    Ok(PyTensor {
        inner: crate::atom::torchx::cross_entropy_op(
            &inp,
            &target,
            ignore_index,
            weight.as_deref(),
            label_smoothing as f32,
            reduction,
        )
        .map_err(PyValueError::new_err)?,
    })
}

#[pyfunction]
fn mse_loss(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let ta = any_to_tensor(a)?;
    let tb = any_to_tensor(b)?;
    Ok(PyTensor {
        inner: crate::atom::torchx::mse_loss_op(&ta, &tb).map_err(PyValueError::new_err)?,
    })
}

#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, bias=None, eps=1e-5))]
fn layer_norm(
    input: &Bound<'_, PyAny>,
    normalized_shape: Vec<usize>,
    weight: Option<&Bound<'_, PyAny>>,
    bias: Option<&Bound<'_, PyAny>>,
    eps: f64,
) -> PyResult<PyTensor> {
    let x = any_to_tensor(input)?;
    let w = weight.map(any_to_tensor).transpose()?;
    let b = bias.map(any_to_tensor).transpose()?;
    Ok(PyTensor {
        inner: crate::atom::torchx::layer_norm_op(
            &x,
            &normalized_shape,
            w.as_ref(),
            b.as_ref(),
            eps as f32,
        )
        .map_err(PyValueError::new_err)?,
    })
}

#[pyfunction]
fn embedding(indices: Vec<usize>, weight: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let w = any_to_tensor(weight)?;
    Ok(PyTensor {
        inner: crate::atom::torchx::embedding_op(&indices, &w).map_err(PyValueError::new_err)?,
    })
}

#[pyfunction]
fn linear(input: &Bound<'_, PyAny>, weight: &Bound<'_, PyAny>, bias: Option<&Bound<'_, PyAny>>) -> PyResult<PyTensor> {
    let x = any_to_tensor(input)?;
    let w = any_to_tensor(weight)?;
    let wt = crate::atom::torchx::transpose_op(&w).map_err(PyValueError::new_err)?;
    let y = crate::atom::torchx::matmul_op(&x, &wt).map_err(PyValueError::new_err)?;
    match bias {
        Some(b) => {
            let bt = any_to_tensor(b)?;
            Ok(PyTensor {
                inner: crate::atom::torchx::add_op(&y, &bt).map_err(PyValueError::new_err)?,
            })
        }
        None => Ok(PyTensor { inner: y }),
    }
}

#[pyfunction]
fn is_grad_enabled() -> bool {
    crate::atom::torchx::grad_enabled()
}

#[pyfunction]
fn set_grad_enabled(enabled: bool) {
    crate::atom::torchx::set_grad_enabled(enabled);
}

#[pyfunction]
fn npu_emu_on() -> bool {
    crate::atom::torchx::npu_emu_on()
}

#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, eps=1e-6))]
fn rms_norm(
    input: &Bound<'_, PyAny>,
    normalized_shape: Vec<usize>,
    weight: Option<&Bound<'_, PyAny>>,
    eps: f64,
) -> PyResult<PyTensor> {
    let x = any_to_tensor(input)?;
    let w = weight.map(any_to_tensor).transpose()?;
    Ok(PyTensor {
        inner: crate::atom::torchx::rms_norm_op(
            &x,
            &normalized_shape,
            w.as_ref(),
            eps as f32,
        )
        .map_err(PyValueError::new_err)?,
    })
}


#[pyfunction]
fn rows_slice(input: &Bound<'_, PyAny>, start: usize, stop: usize) -> PyResult<PyTensor> {
    let t = any_to_tensor(input)?;
    Ok(PyTensor {
        inner: crate::atom::torchx::rows_slice_op(&t, start, stop).map_err(PyValueError::new_err)?,
    })
}


#[pyclass(name = "FlashArchive")]
struct PyFlashArchive {
    path: String,
    sections: Vec<(usize, usize, usize)>,
    buf: Vec<u8>,
}

#[pymethods]
impl PyFlashArchive {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let buf = std::fs::read(&path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if buf.len() < 16 || &buf[0..8] != b"ATQ4AR1\0" {
            return Err(PyValueError::new_err("неверный magic архива"));
        }
        let mut pos = 8usize;
        pos += 4; // version
        pos += 7 * 4 + 4; // cfg + rope_theta
        pos += 4; // legacy
        if pos + 4 > buf.len() { return Err(PyValueError::new_err("eof")); }
        let n_layers = u32::from_le_bytes(buf[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let mut sections = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            if pos + 8 > buf.len() { return Err(PyValueError::new_err("eof")); }
            let sz = u64::from_le_bytes(buf[pos..pos+8].try_into().unwrap()) as usize;
            if pos + 8 + 12 > buf.len() { return Err(PyValueError::new_err("eof payload")); }
            let rows = u32::from_le_bytes(buf[pos+8..pos+12].try_into().unwrap()) as usize;
            let cols = u32::from_le_bytes(buf[pos+12..pos+16].try_into().unwrap()) as usize;
            sections.push((rows, cols, sz));
            pos += 8 + sz;
        }
        if pos != buf.len() {
            return Err(PyValueError::new_err("хвост архива"));
        }
        Ok(Self { path, sections, buf })
    }

    fn section_count(&self) -> usize {
        self.sections.len()
    }

    fn section_info(&self, idx: usize) -> PyResult<(usize, usize, usize)> {
        self.sections.get(idx).cloned().ok_or_else(|| PyValueError::new_err("section out of range"))
    }

    fn layer(&self, idx: usize) -> PyResult<PyFlashCompressed> {
        let mut pos = 8usize + 4 + 32 + 4 + 4;
        for (si, (_, _, sz)) in self.sections.iter().enumerate() {
            let off = pos + 8;
            if si == idx {
                let fc = crate::atom::flash::flash_compressed::FlashCompressed::read_archive(
                    &self.buf[off..off+sz]).map_err(PyValueError::new_err)?;
                return Ok(PyFlashCompressed { inner: fc });
            }
            pos += 8 + sz;
        }
        Err(PyValueError::new_err("section out of range"))
    }

    fn archive_bytes(&self) -> usize {
        self.buf.len()
    }
}

#[pyfunction]
fn atom_fp8_block_encode(w: Vec<f32>, shape: Vec<usize>, block: usize) -> (Vec<u8>, Vec<f32>) {
    let rows = shape[0];
    let cols = shape[1];
    let fc = crate::atom::flash::flash_compressed::FlashFP8::from_dense(
        &w, rows, cols, block.max(1), vec![0.0; rows]);
    (fc.codes, fc.scales)
}

#[pyfunction]
fn atom_fp8_block_decode(codes: Vec<u8>, scales: Vec<f32>, shape: Vec<usize>, block: usize) -> Vec<f32> {
    let rows = shape[0];
    let cols = shape[1];
    let bpr = cols.div_ceil(block.max(1));
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for c in 0..cols {
            let code = codes[i * cols + c];
            let s = scales[i * bpr + c / block.max(1)];
            out[i * cols + c] = crate::atom::flash::flash_compressed::fp8_to_f32(code) * s;
        }
    }
    out
}

#[pyfunction]
fn aot_dump_kernels(path: String) -> PyResult<String> {
    crate::atom::math::atomjit::aot_dump_all(&path).map_err(PyValueError::new_err)
}

/// AOT v2: полный дамп FLASH-JIT (GEMM ×8, math, elem, Q4, atomjit) в файл.
/// При импорте с ATOMNET_AOT_PATH ядра маппятся без DynASM-сборки.
#[pyfunction]
fn aot_dump(path: String) -> PyResult<String> {
    crate::atom::flash::jit::aot_dump_all(&path).map_err(PyValueError::new_err)
}

/// AOT v2: проверка файла без установки в OnceLock (для тестов).
/// Возвращает [(имя, размер, table_old), ...].
#[pyfunction]
fn aot_check(path: String) -> PyResult<Vec<(String, usize, u64)>> {
    crate::atom::flash::jit::aot_import_check(&path).map_err(PyValueError::new_err)
}

/// Откуда взялся JIT после импорта: "none" / "assembled" / "aot".
#[pyfunction]
fn jit_source() -> String {
    crate::atom::flash::jit::jit_source().to_string()
}

#[pyfunction]
fn aot_import_kernels(path: String) -> PyResult<Vec<String>> {
    let parts = crate::atom::math::atomjit::aot_read(&path).map_err(PyValueError::new_err)?;
    Ok(parts.into_iter().map(|(n, b)| format!("{name}:{len}", name = n, len = b.len())).collect())
}

#[pyfunction]
fn bin_popcount(a: Vec<u8>) -> u32 {
    crate::atom::math::binjit::popcount_jit(&a)
}

#[pyfunction]
fn bin_hamming(a: Vec<u8>, b: Vec<u8>) -> u32 {
    crate::atom::math::binjit::hamming_jit(&a, &b)
}

#[pyfunction]
fn bin_xor(a: Vec<u8>, b: Vec<u8>) -> Vec<u8> {
    crate::atom::math::binjit::xor_bytes(&a, &b)
}

// ═══════════════════════════════════════════════════════════
// AtomArray: физический массив (замена numpy-буфера)
// ═══════════════════════════════════════════════════════════

/// Физический массив AtomNET: f32, row-major, с XOR-физикой.
/// Создаётся без numpy: фабрики zeros/ones/randn + методы-операции.
/// Numpy-мост — только на границе (numpy(), from_numpy).
#[pyclass(name = "AtomArray")]
struct PyAtomArray {
    inner: crate::atom::array::AtomArray,
}

fn atom_arr(inner: crate::atom::array::AtomArray) -> PyAtomArray {
    PyAtomArray { inner }
}

/// Borrow-доступ к AtomArray без копирования данных: вызывающий держит
/// PyRef на время операции в той же области. Ноль копий в горячем пути.
fn with_arr<R>(
    o: &Bound<'_, PyAny>,
    f: impl FnOnce(&crate::atom::array::AtomArray) -> PyResult<R>,
) -> PyResult<R> {
    let t = o
        .downcast::<PyAtomArray>()
        .map_err(|_| PyValueError::new_err("ожидался AtomArray"))?;
    let b = t.borrow();
    f(&b.inner)
}

#[pymethods]
impl PyAtomArray {
    #[new]
    #[pyo3(signature = (data, shape=None))]
    fn new(data: Vec<f32>, shape: Option<Vec<usize>>) -> Self {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        atom_arr(crate::atom::array::AtomArray::new(shape, data))
    }

    fn __repr__(&self) -> String {
        format!(
            "AtomArray(shape={:?}, dtype=float32, sig={:#x}, alive={})",
            self.inner.shape, self.inner.sig, self.inner.is_alive()
        )
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    #[getter]
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    #[getter]
    fn sig(&self) -> u64 {
        self.inner.sig
    }

    #[getter]
    fn energy(&self) -> f32 {
        self.inner.energy()
    }

    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn gate(&self) -> f32 {
        self.inner.gate()
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.data.clone()
    }

    fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.ndim() == 2 {
            let (m, n) = (self.inner.rows(), self.inner.cols());
            let rows: Vec<Vec<f32>> = (0..m)
                .map(|i| self.inner.data[i * n..(i + 1) * n].to_vec())
                .collect();
            let a = PyArray2::from_vec2(py, &rows)
                .map_err(|e| PyValueError::new_err(format!("AtomArray.numpy: {e}")))?;
            Ok(a.as_any().clone())
        } else {
            Ok(PyArray1::from_vec(py, self.inner.data.clone()).as_any().clone())
        }
    }

    /// Быстрое создание из вложенного списка [[..], [..]] — flatten в Rust
    /// без Python-цикла (в разы быстрее искусственного _flatten).
    #[staticmethod]
    fn from_rows(rows: Vec<Vec<f32>>) -> Self {
        let m = rows.len();
        let n = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(m * n);
        for r in rows {
            data.extend_from_slice(&r);
        }
        atom_arr(crate::atom::array::AtomArray::new(vec![m, n], data))
    }

    #[staticmethod]
    fn from_numpy(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Numpy-массив → AtomArray (копия на границе, дальше — без numpy).
        let py = arr.py();
        let np = py.import("numpy")?;
        let asarray = np.getattr("asarray")?;
        let a = asarray.call1((arr,))?;
        // shape как кортеж
        let shape_obj = a.getattr("shape")?;
        let shape: Vec<usize> = shape_obj.extract()?;
        // плоские данные f32
        let flat = a.getattr("reshape")?.call1((-1,))?;
        let contiguous = np.getattr("ascontiguousarray")?.call1((flat,))?;
        let data: Vec<f32> = contiguous.extract()?;
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(PyValueError::new_err("from_numpy: shape/data mismatch"));
        }
        Ok(atom_arr(crate::atom::array::AtomArray::new(shape, data)))
    }

    // ── фабрики ──────────────────────────────────────────────
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        atom_arr(crate::atom::array::AtomArray::zeros(shape))
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        atom_arr(crate::atom::array::AtomArray::ones(shape))
    }

    #[staticmethod]
    fn full(shape: Vec<usize>, val: f32) -> Self {
        atom_arr(crate::atom::array::AtomArray::full(shape, val))
    }

    #[staticmethod]
    fn arange(n: usize) -> Self {
        atom_arr(crate::atom::array::AtomArray::arange(n))
    }

    #[staticmethod]
    #[pyo3(signature = (start, stop, num))]
    fn linspace(start: f32, stop: f32, num: usize) -> Self {
        atom_arr(crate::atom::array::AtomArray::linspace(start, stop, num))
    }

    #[staticmethod]
    fn eye(n: usize) -> Self {
        atom_arr(crate::atom::array::AtomArray::eye(n))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, seed = 0))]
    fn rand(shape: Vec<usize>, seed: u64) -> Self {
        atom_arr(crate::atom::array::AtomArray::rand(shape, seed))
    }

    #[staticmethod]
    #[pyo3(signature = (shape, seed = 0))]
    fn randn(shape: Vec<usize>, seed: u64) -> Self {
        atom_arr(crate::atom::array::AtomArray::randn(shape, seed))
    }

    // ── elementwise / редукции (считаются в Rust без numpy) ─────
    fn sum(&self) -> f32 {
        self.inner.sum()
    }

    fn mean(&self) -> f32 {
        self.inner.mean()
    }

    fn var(&self) -> f32 {
        self.inner.var()
    }

    fn std(&self) -> f32 {
        self.inner.std()
    }

    fn max(&self) -> f32 {
        self.inner.max()
    }

    fn min(&self) -> f32 {
        self.inner.min()
    }

    fn argmax(&self) -> usize {
        self.inner.argmax()
    }

    fn argmin(&self) -> usize {
        self.inner.argmin()
    }

    fn norm(&self) -> f32 {
        self.inner.norm()
    }

    fn dot(&self, other: &Bound<'_, PyAny>) -> PyResult<f32> {
        with_arr(other, |b| Ok(self.inner.dot(b)))
    }

    fn matmul(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        with_arr(other, |b| Ok(atom_arr(self.inner.matmul(b))))
    }

    fn matvec(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        with_arr(other, |b| Ok(atom_arr(self.inner.matvec(b))))
    }

    fn __matmul__(&self, other: Bound<'_, PyAny>) -> PyResult<Self> {
        self.matmul(&other)
    }

    fn transpose(&self) -> PyResult<Self> {
        Ok(atom_arr(self.inner.transpose()))
    }

    fn t(&self) -> Self {
        atom_arr(self.inner.t())
    }

    fn reshape(&self, shape: Vec<usize>) -> Self {
        atom_arr(self.inner.reshape(shape))
    }

    fn add_arr(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        with_arr(other, |b| Ok(atom_arr(self.inner.add(b))))
    }

    fn sub_arr(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        with_arr(other, |b| Ok(atom_arr(self.inner.sub(b))))
    }

    fn mul_arr(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        with_arr(other, |b| Ok(atom_arr(self.inner.mul(b))))
    }

    fn add_scalar(&self, v: f32) -> Self {
        atom_arr(self.inner.add_scalar(v))
    }

    fn mul_scalar(&self, v: f32) -> Self {
        atom_arr(self.inner.mul_scalar(v))
    }

    fn relu_(&mut self) {
        self.inner.relu_();
    }

    fn silu_(&mut self) {
        self.inner.silu_();
    }

    fn softmax_(&mut self) {
        self.inner.softmax_();
    }

    fn gelu_(&mut self) {
        self.inner.gelu_();
    }

    fn sigmoid_(&mut self) {
        self.inner.sigmoid_();
    }

    fn tanh_(&mut self) {
        self.inner.tanh_();
    }

    fn exp_(&mut self) {
        self.inner.exp_();
    }

    fn project_sphere(&mut self) -> f32 {
        self.inner.project_sphere()
    }

    fn xor_signature(&self) -> u64 {
        self.inner.xor_signature()
    }

    fn xor_fold(&self) -> u64 {
        self.inner.xor_fold()
    }

    fn quantize_int8(&self) -> (Vec<i8>, f32, f32) {
        self.inner.quantize_int8()
    }

    fn allclose(&self, other: &Bound<'_, PyAny>, atol: f32) -> PyResult<bool> {
        with_arr(other, |b| Ok(self.inner.allclose(b, atol)))
    }

    fn abs_(&mut self) {
        self.inner.abs_();
    }

    fn sin_(&mut self) {
        self.inner.sin_();
    }

    fn cos_(&mut self) {
        self.inner.cos_();
    }

    fn sqrt_(&mut self) {
        self.inner.sqrt_();
    }

    fn log_(&mut self) {
        self.inner.log_();
    }

    fn clip_(&mut self, lo: f32, hi: f32) {
        self.inner.clip_(lo, hi);
    }

    fn affine_(&mut self, scale: f32, shift: f32) {
        self.inner.affine_(scale, shift);
    }
}

fn register_torchx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyAtomArray>()?;
    m.add_function(wrap_pyfunction!(manual_seed, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(rand_py, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(randint, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(embedding, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)?;
    m.add_function(wrap_pyfunction!(is_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(npu_emu_on, m)?)?;
    m.add_function(wrap_pyfunction!(rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(rows_slice, m)?)?;
    m.add_function(wrap_pyfunction!(atom_fp8_block_encode, m)?)?;
    m.add_function(wrap_pyfunction!(atom_fp8_block_decode, m)?)?;
    m.add_function(wrap_pyfunction!(aot_dump_kernels, m)?)?;
    m.add_function(wrap_pyfunction!(aot_dump, m)?)?;
    m.add_function(wrap_pyfunction!(aot_check, m)?)?;
    m.add_function(wrap_pyfunction!(jit_source, m)?)?;
    m.add_function(wrap_pyfunction!(aot_import_kernels, m)?)?;
    m.add_function(wrap_pyfunction!(bin_popcount, m)?)?;
    m.add_function(wrap_pyfunction!(bin_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(bin_xor, m)?)?;
    Ok(())
}

#[pyclass(name = "DynGraph")]
struct PyDynGraph {
    g: crate::atom::graph::DynGraph,
}

#[pymethods]
impl PyDynGraph {
    #[new]
    fn new() -> Self {
        Self { g: crate::atom::graph::DynGraph::new() }
    }

    fn add_const(&mut self, data: Vec<f32>) -> usize {
        self.g.add_const(data)
    }

    fn add_matmul(&mut self, a: usize, b: usize, m: usize, k: usize, n: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Gemm { m, k, n }, vec![a, b])
    }

    fn add_matmul_silu(&mut self, a: usize, b: usize, m: usize, k: usize, n: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::GemmSilu { m, k, n }, vec![a, b])
    }

    fn add_softmax(&mut self, src: usize, rows: usize, cols: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Softmax { rows, cols }, vec![src])
    }

    fn add_silu(&mut self, src: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Silu, vec![src])
    }

    fn add_sigmoid(&mut self, src: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Sigmoid, vec![src])
    }

    fn add_add(&mut self, a: usize, b: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Add, vec![a, b])
    }

    fn add_mul(&mut self, a: usize, b: usize) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Mul, vec![a, b])
    }

    fn add_scale(&mut self, src: usize, r: f32) -> usize {
        self.g.add_node(crate::atom::graph::OpKind::Scale { r }, vec![src])
    }

    fn set_outputs(&mut self, ids: Vec<usize>) {
        self.g.set_outputs(ids)
    }

    fn run(&self) -> PyResult<Vec<Vec<f32>>> {
        self.g.run().map_err(PyValueError::new_err)
    }
}

fn register_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTransformerLM>()?;
    m.add_class::<PyBpeTokenizer>()?;
    m.add_class::<PyFlashArchive>()?;
    m.add_class::<PyDynGraph>()?;
    m.add_class::<PyKvCache>()?;
    Ok(())
}

#[pyfunction]
fn npu_status() -> String {
    let backend = crate::atom::npu::NpuRouter::new()
        .backend()
        .map(|b| b.id())
        .unwrap_or("none");
    format!(
        "backend={} device={:?} thread_budget={} npu_emu={}",
        backend,
        crate::atom::npu::NpuDevice::detect(),
        crate::atom::flash::flash_hw::thread_budget(),
        crate::atom::torchx::npu_emu_on(),
    )
}

#[pyfunction]
fn npu_dot(a: Vec<f32>, b: Vec<f32>) -> f32 {
    crate::atom::npu::npu_dot(&a, &b)
}

#[pyfunction]
fn npu_matvec(w: Vec<f32>, rows: usize, cols: usize, x: Vec<f32>) -> Vec<f32> {
    crate::atom::npu::npu_matvec(&w, rows, cols, &x)
}

#[pyfunction]
fn npu_matmul(
    a: Vec<f32>,
    a_rows: usize,
    a_cols: usize,
    b: Vec<f32>,
    b_rows: usize,
    b_cols: usize,
) -> Vec<f32> {
    crate::atom::npu::npu_matmul(&a, a_rows, a_cols, &b, b_rows, b_cols)
}

// ── Zero-copy NPU (без конвертации list): numpy-массивы ──────────────────

/// Flat f32 для npy-множества: контигуозный — слайс без копии
/// (нужен вызывающиим, чтобы держать guard/владелец живым на время вызова),
/// strided — ravel-копия (редкий путь).
fn npy_flat1<'py>(
    a: &Bound<'py, PyArray1<f32>>,
) -> PyResult<(Vec<f32>, bool)> {
    use numpy::{PyUntypedArrayMethods, PyArrayMethods};
    // (owned_vec, is_owned): is_owned=false — данные лежат в самом массиве
    // (contiguous), вызывающий берёт слайс через as_slice под guard'ом.
    // Здесь только фолбэк для strided: .copy() делает C-порядок на C-массиве.
    if a.is_c_contiguous() {
        return Ok((Vec::new(), false));
    }
    let copied = a.as_any().call_method0("copy")?;
    let arr = copied.downcast::<PyArray1<f32>>()?;
    let vec = arr.readonly().as_slice()?.to_vec();
    Ok((vec, true))
}

fn npy_flat2<'py>(
    a: &Bound<'py, PyArray2<f32>>,
) -> PyResult<(Vec<f32>, bool)> {
    use numpy::{PyUntypedArrayMethods, PyArrayMethods};
    if a.is_c_contiguous() {
        return Ok((Vec::new(), false));
    }
    let copied = a.as_any().call_method0("copy")?;
    let arr = copied.downcast::<PyArray2<f32>>()?;
    let vec = arr.readonly().as_slice()?.to_vec();
    Ok((vec, true))
}

/// y = a·b через NPU-эмулятор (numpy).
#[pyfunction]
fn npu_dot_np<'py>(
    a: &Bound<'py, PyArray1<f32>>,
    b: &Bound<'py, PyArray1<f32>>,
) -> PyResult<f32> {
    let ra = a.readonly();
    let rb = b.readonly();
    let (oa, own_a) = npy_flat1(a)?;
    let (ob, own_b) = npy_flat1(b)?;
    let sa: &[f32] = if own_a { &oa } else { ra.as_slice()? };
    let sb: &[f32] = if own_b { &ob } else { rb.as_slice()? };
    Ok(crate::atom::npu::npu_dot(sa, sb))
}

/// y = W·x через NPU-эмулятор (numpy).
#[pyfunction]
fn npu_matvec_np<'py>(
    py: Python<'py>,
    w: &Bound<'py, PyArray2<f32>>,
    x: &Bound<'py, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let rw = w.readonly();
    let rx = x.readonly();
    let sw = rw.shape();
    if sw.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>("npu_matvec_np: W must be 2D"));
    }
    let (rows, cols) = (sw[0] as usize, sw[1] as usize);
    if rx.shape() != [cols] {
        return Err(PyErr::new::<PyValueError, _>("npu_matvec_np: len(x) != cols"));
    }
    let (ow, own_w) = npy_flat2(w)?;
    let (ox, own_x) = npy_flat1(x)?;
    let ws: &[f32] = if own_w { &ow } else { rw.as_slice()? };
    let xs: &[f32] = if own_x { &ox } else { rx.as_slice()? };
    let y = crate::atom::npu::npu_matvec(ws, rows, cols, xs);
    Ok(y.into_pyarray(py))
}

/// C = A×B через NPU-эмулятор (numpy).
#[pyfunction]
fn npu_matmul_np<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyArray2<f32>>,
    b: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use numpy::PyUntypedArrayMethods;
    let ra = a.readonly();
    let rb = b.readonly();
    let sa = ra.shape();
    let sb = rb.shape();
    if sa.len() != 2 || sb.len() != 2 {
        return Err(PyErr::new::<PyValueError, _>("npu_matmul_np: 2D expected"));
    }
    let (m, k) = (sa[0] as usize, sa[1] as usize);
    let (k2, n) = (sb[0] as usize, sb[1] as usize);
    if k != k2 {
        return Err(PyErr::new::<PyValueError, _>("npu_matmul_np: inner dims mismatch"));
    }
    let (oa, own_a) = npy_flat2(a)?;
    let (ob, own_b) = npy_flat2(b)?;
    let fa: &[f32] = if own_a { &oa } else { ra.as_slice()? };
    let fb: &[f32] = if own_b { &ob } else { rb.as_slice()? };
    // Zero-copy-out: пишем прямо в numpy-массив под allow_threads.
    // (Старый путь: Vec<f32> → Vec<Vec<f32>> → from_vec2 = 2 лишние копии.)
    let out = if k == 0 {
        PyArray2::<f32>::zeros(py, [m, n], false)
    } else {
        unsafe { PyArray2::<f32>::new(py, [m, n], false) }
    };
    let a_ptr = fa.as_ptr() as usize;
    let b_ptr = fb.as_ptr() as usize;
    let (a_len, b_len) = (fa.len(), fb.len());
    let c_ptr = {
        let mut rw = out.readwrite();
        let dst = rw.as_slice_mut()?;
        dst.as_mut_ptr() as usize
    };
    py.allow_threads(move || {
        let a = unsafe { std::slice::from_raw_parts(a_ptr as *const f32, a_len) };
        let b = unsafe { std::slice::from_raw_parts(b_ptr as *const f32, b_len) };
        let c = unsafe { std::slice::from_raw_parts_mut(c_ptr as *mut f32, m * n) };
        let parts = crate::atom::flash::flash_gemm::gemm_auto_parts(0, m, k, n);
        crate::atom::flash::flash_gemm::gemm_blocked_into(a, m, k, b, n, parts, c);
    });
    Ok(out)
}

/// Статус эмулятора: профиль, ядра, распределение.
#[pyfunction]
fn npu_profile() -> String {
    let p = crate::atom::npu::npu_profile();
    format!(
        "virtual_cores={} cores={} mac={}x{}",
        p.virtual_cores,
        p.cores(),
        p.mac_rows,
        p.mac_cols
    )
}

/// Циклы mac-массива для flops (модель реального NPU).
#[pyfunction]
fn npu_cycles(flops: u64) -> u64 {
    crate::atom::npu::npu_cycles(flops)
}

/// Виртуальный ISA: AVX2/FMA-регистровый файл на виртуальное ядро.
#[pyfunction]
fn virtual_isa() -> String {
    let isa = crate::atom::isa::virt_isa();
    format!(
        "tier={} avx2={} fma={} popcnt={} ymm_regs={} fma_ports={} ymm_core={}",
        isa.tier,
        isa.avx2,
        isa.fma,
        isa.popcnt,
        isa.ymm_regs,
        isa.fma_ports,
        crate::atom::isa::ymm_regs(0)
    )
}

/// Задать число виртуальных NPU-ядер (>=1, можно > потоков).
/// Изменения действуют на распределение работ (диагностика/зоны),
/// compute-ядра остаются детерминированными.
#[pyfunction]
fn npu_set_cores(virtual_cores: usize) -> String {
    std::env::set_var("ATOMNET_NPU_CORES", virtual_cores.max(1).to_string());
    format!(
        "cores={} (env ATOMNET_NPU_CORES; перечитайте профиль на старте)",
        virtual_cores.max(1)
    )
}

/// Распределение rows×cols по mac-ядрам (диагностика).
#[pyfunction]
fn npu_distribute(rows: usize, cols: usize) -> Vec<(usize, usize, usize, usize)> {
    crate::atom::npu::npu_distribute(rows, cols)
}

// ═══════════════════════════════════════════════════════════
// TransformerLM + KvCache как граждане .so (сервер = сама модель)
// ═══════════════════════════════════════════════════════════

/// Владелец нейросети: весовая память живёт как Rust-структуры внутри
/// динамической библиотеки; Python видит только методы.
#[pyclass(name = "TransformerLM")]
struct PyTransformerLM {
    inner: std::sync::Mutex<crate::atom::model::TransformerLM>,
}

/// Окарованная реализация KV-кэша инкремент-унктывенту деплоёны
/// (на том же языке, что и KvCache языка).
#[pyclass(name = "KvCache")]
struct PyKvCache {
    inner: std::sync::Mutex<crate::atom::model::KvCache>,
}

/// Byte-level BPE токенизатор (atom::tok) как гражданин .so.
#[pyclass(name = "BpeTokenizer")]
struct PyBpeTokenizer {
    inner: crate::atom::tok::BpeTokenizer,
}

#[pymethods]
impl PyBpeTokenizer {
    /// Обучить BPE на байтах: n_merges слияний (0 — только байт-фолбэк).
    #[staticmethod]
    fn train(data: Vec<u8>, n_merges: usize) -> Self {
        Self {
            inner: crate::atom::tok::train_bpe(&data, n_merges),
        }
    }

    /// Байты -> токены.
    fn encode(&self, data: Vec<u8>) -> Vec<usize> {
        self.inner.encode(&data)
    }

    /// Токены -> байты.
    fn decode(&self, ids: Vec<usize>) -> Vec<u8> {
        self.inner.decode(&ids)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[pymethods]
impl PyKvCache {
    #[new]
    fn new(lm: &PyTransformerLM) -> Self {
        Self {
            inner: std::sync::Mutex::new(lm.inner.lock().unwrap().new_cache()),
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    fn reset(&self) {
        self.inner.lock().unwrap().reset();
    }
}

#[pymethods]
impl PyTransformerLM {
    #[new]
    #[pyo3(signature = (vocab, d_model, heads=4, kv_heads=1, d_head=32, hidden=512, blocks=4, rope_theta=10000.0))]
    fn new(
        vocab: usize,
        d_model: usize,
        heads: usize,
        kv_heads: usize,
        d_head: usize,
        hidden: usize,
        blocks: usize,
        rope_theta: f32,
    ) -> Self {
        let cfg = crate::atom::model::LmConfig {
            vocab,
            d_model,
            q_heads: heads,
            kv_heads,
            d_head,
            hidden,
            blocks,
            rope_theta,
            tie_embeddings: false,
        };
        Self {
            inner: std::sync::Mutex::new(crate::atom::model::TransformerLM::new(&cfg)),
        }
    }

    fn param_count(&self) -> usize {
        self.inner.lock().unwrap().param_count()
    }

    fn set_lr(&self, lr: f32) {
        self.inner.lock().unwrap().set_lr(lr);
    }

    fn set_momentum(&self, beta: f32) {
        self.inner.lock().unwrap().set_momentum(beta);
    }

    fn clear_grads(&self) {
        self.inner.lock().unwrap().clear_grads();
    }

    /// Один sample: CE по позициям 0..l-1, накопление градиентов (loss).
    fn train_sample(&self, seq: Vec<usize>, scale: f32) -> PyResult<f32> {
        if seq.len() < 2 {
            return Err(PyErr::new::<PyValueError, _>("train_sample: seq>=2"));
        }
        Ok(self.inner.lock().unwrap().train_sample(&seq, scale))
    }

    /// Применить обновления по накопленным grad'ам (агменжение оптимизатора).
    fn step(&self) {
        self.inner.lock().unwrap().step();
    }

    /// Полный forward (пачка): logits (l × vocab) одним ответом.
    fn forward<'py>(
        &self,
        py: Python<'py>,
        seq: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let fwd = {
            py.allow_threads(|| self.inner.lock().unwrap().forward(&seq))
        };
        let vocab = self.inner.lock().unwrap().cfg.vocab.max(1);
        let rows: Vec<Vec<f32>> = fwd.logits.chunks(vocab).map(|c| c.to_vec()).collect();
        Ok(PyArray2::<f32>::from_vec2(py, &rows)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("forward: {}", e)))?)
    }

    /// Точечный forward на токен с KV-кэшем, возвращает logits (vocab,).
    fn forward_token_cached<'py>(
        &self,
        py: Python<'py>,
        token: usize,
        cache: PyRef<'_, PyKvCache>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // Без allow_threads: PyRef<T> не Send.
        let mut cache_guard = cache.inner.lock().unwrap();
        let out = self
            .inner
            .lock()
            .unwrap()
            .forward_token_cached(token, &mut *cache_guard);
        drop(cache_guard);
        Ok(out.into_pyarray(py))
    }

    /// Greedy-генерация: жадное расширение префикса с KV-кэшем.
    fn generate(&self, prefix: Vec<usize>, max_len: usize) -> PyResult<Vec<usize>> {
        let inner = self.inner.lock().unwrap();
        if prefix.is_empty() {
            return Err(PyErr::new::<PyValueError, _>("generate: пустой префикс"));
        }
        let vocab = inner.cfg.vocab;
        Ok(crate::atom::model::generate(&inner, &prefix, max_len, vocab))
    }

    /// Все веса флетом (читательная доводка для хранения данных).
    fn dump_weights(&self) -> Vec<f32> {
        self.inner.lock().unwrap().dump_weights()
    }

    /// Загрузить ранее разнесённые веса (флет-формат dump_weights).
    fn load_weights(&self, flat: Vec<f32>) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .load_weights(&flat)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn save_compressed(&self, path: String) -> PyResult<usize> {
        let lm = self.inner.lock().unwrap();
        lm.save_compressed(&path).map_err(PyValueError::new_err)
    }

    fn load_compressed(&self, path: String) -> PyResult<usize> {
        let mut lm = self.inner.lock().unwrap();
        lm.load_compressed(&path).map_err(PyValueError::new_err)
    }

    fn save_checkpoint(&self, path: String) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .save_checkpoint(&path)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }

    fn load_checkpoint(&self, path: String) -> PyResult<()> {
        self.inner
            .lock()
            .unwrap()
            .load_checkpoint(&path)
            .map_err(|e| PyErr::new::<PyValueError, _>(e))
    }
}

// ═══════════════════════════════════════════════════════════
// Module
// ═══════════════════════════════════════════════════════════

#[pymodule]
fn atomnet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    // New structured submodules.
    let math_mod = PyModule::new(py, "math")?;
    register_math(&math_mod)?;
    m.add_submodule(&math_mod)?;

    let life_mod = PyModule::new(py, "life")?;
    register_life(&life_mod)?;
    m.add_submodule(&life_mod)?;

    let optim_mod = PyModule::new(py, "optim")?;
    register_optim(&optim_mod)?;
    m.add_submodule(&optim_mod)?;

    let flash_mod = PyModule::new(py, "flash")?;
    register_flash(&flash_mod)?;
    m.add_submodule(&flash_mod)?;

    let veron_mod = PyModule::new(py, "veron")?;
    register_veron(&veron_mod)?;
    m.add_submodule(&veron_mod)?;

    let hexiron_mod = PyModule::new(py, "hexiron")?;
    register_hexiron(&hexiron_mod)?;
    m.add_submodule(&hexiron_mod)?;

    let npu_mod = PyModule::new(py, "npu")?;
    register_npu(&npu_mod)?;
    m.add_submodule(&npu_mod)?;

    // add_submodule делает атрибут, но не регистрирует имя в sys.modules;
    // без этого `import atomnet.math` падает.
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    for (name, sub) in [
        ("atomnet.math", &math_mod),
        ("atomnet.life", &life_mod),
        ("atomnet.optim", &optim_mod),
        ("atomnet.flash", &flash_mod),
        ("atomnet.veron", &veron_mod),
        ("atomnet.hexiron", &hexiron_mod),
        ("atomnet.npu", &npu_mod),
    ] {
        modules.set_item(name, sub)?;
    }

    // У PyO3-submodule нет реального __path__, поэтому import-стема
    // до сих пор не видит `atomnet.math` как submodule. Положить
    // пустой namespace-path и повторить регистрацию.
    for sub in [&math_mod, &life_mod, &optim_mod, &flash_mod, &veron_mod, &hexiron_mod, &npu_mod] {
        sub.setattr("__path__", Vec::<&str>::new())?;
    }

    // Flat compatibility API.
    register_math(m)?;
    register_life(m)?;
    // Старый класс жил как PyQairoLayer до переименования в PyQairoLivingLayer.
    {
        let qairo_cls = m.getattr("PyQairoLivingLayer")?;
        m.setattr("PyQairoLayer", qairo_cls)?;
        life_mod.setattr("PyQairoLayer", life_mod.getattr("PyQairoLivingLayer")?)?;
        // PyO3 вычисляет __all__ из зарегистрированных элементов до этого
        // setattr; без ручного добавления alias не попадал бы в пакетный
        // star-import (`from .atomnet import *` в __init__.py).
        let all: Vec<String> = m
            .getattr("__all__")
            .and_then(|a| a.extract())
            .unwrap_or_default();
        if !all.contains(&"PyQairoLayer".to_string()) {
            let mut all = all;
            all.push("PyQairoLayer".to_string());
            m.setattr("__all__", all)?;
        }
    }
    register_optim(m)?;
    register_flash(m)?;
    register_veron(m)?;
    register_hexiron(m)?;
    register_npu(m)?;
    register_model(m)?;
    register_torchx(m)?;

    Ok(())
}
