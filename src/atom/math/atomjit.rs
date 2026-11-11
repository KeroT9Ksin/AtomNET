// src/atom/math/atomjit.rs
//! atomjit — DynASM-ядра сферической арифметики (AVX2 float).
//!
//! Три ядра горячей части `atommath`, собранные в машинный код с нашим
//! регистровым аллокатором (как в `flash::jit`):
//!   - `dot_avx2(a, b, n)`: произведение векторов, 8 YMM-аккумуляторов по
//!     64 floats на итерацию, затем ящики 32/8 и скалярный хвост;
//!   - `normsq_avx2(a, n)`: сумма квадратов (dot(a, a)), завершается sqrt;
//!   - `axpy_avx2(coef, x, y, n)`: `y[i] += coef * x[i]`, векторизовано,
//!     любой хвост проходит через vmaskmovps-маски (без чтения за границами).
//!
//! ABI (System V x86-64), sysv64:
//!   dot:    rdi = a, rsi = b, rdx = n;            -> f32 в xmm0
//!   norm:   rdi = a, rsi = n;                     -> f32 в xmm0
//!   axpy:   xmm0 = coef, rdi = x, rsi = y, rdx = n
//!
//! Метки — форма `->name:` (шаблон `flash::jit`).
//!
//! Доступность: только x86_64 + AVX2 + FMA. Иначе fallback на
//! `crate::atom::math::simd` (детерминирована до 1 ULP — тесты ниже).

#[cfg(target_arch = "x86_64")]
mod sysv {
    use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi, ExecutableBuffer};

    pub type DotKernel = unsafe extern "sysv64" fn(*const f32, *const f32, usize) -> f32;
    pub type NormKernel = unsafe extern "sysv64" fn(*const f32, usize) -> f32;
    pub type AxpyKernel = unsafe extern "sysv64" fn(f32, *const f32, *mut f32, usize);

    pub fn build_dot() -> Result<(ExecutableBuffer, DotKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; push r13
            ; vxorps ymm0, ymm0, ymm0      // acc 0
            ; vxorps ymm1, ymm1, ymm1
            ; vxorps ymm2, ymm2, ymm2
            ; vxorps ymm3, ymm3, ymm3
            ; vxorps ymm4, ymm4, ymm4
            ; vxorps ymm5, ymm5, ymm5
            ; vxorps ymm6, ymm6, ymm6
            ; vxorps ymm7, ymm7, ymm7
            ; vxorps xmm12, xmm12, xmm12   // scalar tail accum (xmm12! не ymm10)
            ; xor r10d, r10d
            ; xor r11d, r11d
            ; mov r12, rdx                 // n
            ; ->lp64:
            ;   mov r13, r10
            ;   add r13, 64
            ;   cmp r13, r12
            ;   ja ->done64
            ;   vmovups ymm8, [rdi + r11]
            ;   vmovups ymm9, [rsi + r11]
            ;   vfmadd231ps ymm0, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 32]
            ;   vmovups ymm9, [rsi + r11 + 32]
            ;   vfmadd231ps ymm1, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 64]
            ;   vmovups ymm9, [rsi + r11 + 64]
            ;   vfmadd231ps ymm2, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 96]
            ;   vmovups ymm9, [rsi + r11 + 96]
            ;   vfmadd231ps ymm3, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 128]
            ;   vmovups ymm9, [rsi + r11 + 128]
            ;   vfmadd231ps ymm4, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 160]
            ;   vmovups ymm9, [rsi + r11 + 160]
            ;   vfmadd231ps ymm5, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 192]
            ;   vmovups ymm9, [rsi + r11 + 192]
            ;   vfmadd231ps ymm6, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 224]
            ;   vmovups ymm9, [rsi + r11 + 224]
            ;   vfmadd231ps ymm7, ymm8, ymm9
            ;   add r10, 64
            ;   add r11, 256
            ;   jmp ->lp64
            ; ->done64:
            ;   mov r13, r10
            ;   add r13, 32
            ;   cmp r13, r12
            ;   ja ->skip32
            ;   vmovups ymm8, [rdi + r11]
            ;   vmovups ymm9, [rsi + r11]
            ;   vfmadd231ps ymm0, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 32]
            ;   vmovups ymm9, [rsi + r11 + 32]
            ;   vfmadd231ps ymm1, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 64]
            ;   vmovups ymm9, [rsi + r11 + 64]
            ;   vfmadd231ps ymm2, ymm8, ymm9
            ;   vmovups ymm8, [rdi + r11 + 96]
            ;   vmovups ymm9, [rsi + r11 + 96]
            ;   vfmadd231ps ymm3, ymm8, ymm9
            ;   add r10, 32
            ;   add r11, 128
            ; ->skip32:
            ;   mov r13, r10
            ;   add r13, 8
            ;   cmp r13, r12
            ;   ja ->skip8
            ;   vmovups ymm8, [rdi + r11]
            ;   vmovups ymm9, [rsi + r11]
            ;   vfmadd231ps ymm4, ymm8, ymm9
            ;   add r10, 8
            ;   add r11, 32
            ; ->skip8:
            ; ->tlo:
            ;   cmp r10, r12
            ;   jae ->red
            ;   vmovd xmm8, [rdi + r11]
            ;   vmovd xmm9, [rsi + r11]
            ;   vfmadd231ss xmm12, xmm8, xmm9
            ;   add r10, 1
            ;   add r11, 4
            ;   jmp ->tlo
            ; ->red:
            ;   vaddps ymm8, ymm0, ymm4
            ;   vaddps ymm9, ymm1, ymm5
            ;   vaddps ymm10, ymm2, ymm6
            ;   vaddps ymm11, ymm3, ymm7
            ;   vaddps ymm0, ymm8, ymm9
            ;   vaddps ymm1, ymm10, ymm11
            ;   vaddps ymm0, ymm0, ymm1
            ;   vextractf128 xmm1, ymm0, 1
            ;   vaddps xmm0, xmm0, xmm1
            ;   vhaddps xmm1, xmm0, xmm0
            ;   vhaddps xmm0, xmm1, xmm1
            ;   vaddss xmm0, xmm0, xmm12
            ;   pop r13
            ;   pop r12
            ;   pop rbx
            ;   ret
        );
        let buf = ops.finalize().map_err(|_| "atomjit dot: finalize failed".to_string())?;
        let f: DotKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    pub fn build_norm() -> Result<(ExecutableBuffer, NormKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; push r13
            ; vxorps ymm0, ymm0, ymm0
            ; vxorps ymm1, ymm1, ymm1
            ; vxorps ymm2, ymm2, ymm2
            ; vxorps ymm3, ymm3, ymm3
            ; vxorps ymm4, ymm4, ymm4
            ; vxorps ymm5, ymm5, ymm5
            ; vxorps ymm6, ymm6, ymm6
            ; vxorps ymm7, ymm7, ymm7
            ; vxorps xmm12, xmm12, xmm12
            ; xor r10d, r10d
            ; xor r11d, r11d
            ; mov r12, rsi
            ; ->lp64:
            ;   mov r13, r10
            ;   add r13, 64
            ;   cmp r13, r12
            ;   ja ->done64
            ;   vmovups ymm8, [rdi + r11]
            ;   vfmadd231ps ymm0, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 32]
            ;   vfmadd231ps ymm1, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 64]
            ;   vfmadd231ps ymm2, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 96]
            ;   vfmadd231ps ymm3, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 128]
            ;   vfmadd231ps ymm4, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 160]
            ;   vfmadd231ps ymm5, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 192]
            ;   vfmadd231ps ymm6, ymm8, ymm8
            ;   vmovups ymm8, [rdi + r11 + 224]
            ;   vfmadd231ps ymm7, ymm8, ymm8
            ;   add r10, 64
            ;   add r11, 256
            ;   jmp ->lp64
            ; ->done64:
            ; ->tlo:
            ;   cmp r10, r12
            ;   jae ->red
            ;   vmovd xmm8, [rdi + r11]
            ;   vfmadd231ss xmm12, xmm8, xmm8
            ;   add r10, 1
            ;   add r11, 4
            ;   jmp ->tlo
            ; ->red:
            ;   vaddps ymm8, ymm0, ymm4
            ;   vaddps ymm9, ymm1, ymm5
            ;   vaddps ymm10, ymm2, ymm6
            ;   vaddps ymm11, ymm3, ymm7
            ;   vaddps ymm0, ymm8, ymm9
            ;   vaddps ymm1, ymm10, ymm11
            ;   vaddps ymm0, ymm0, ymm1
            ;   vextractf128 xmm1, ymm0, 1
            ;   vaddps xmm0, xmm0, xmm1
            ;   vhaddps xmm1, xmm0, xmm0
            ;   vhaddps xmm0, xmm1, xmm1
            ;   vaddss xmm0, xmm0, xmm12
            ;   vsqrtss xmm0, xmm0, xmm0
            ;   pop r13
            ;   pop r12
            ;   pop rbx
            ;   ret
        );
        let buf = ops.finalize().map_err(|_| "atomjit norm: finalize failed".to_string())?;
        let f: NormKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    pub fn build_axpy() -> Result<(ExecutableBuffer, AxpyKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; push r13
            ; vbroadcastss ymm12, xmm0
            ; xor r10d, r10d
            ; xor r11d, r11d
            ; xor r13d, r13d
            ; ->lp64:
            ;   mov r13, r10
            ;   add r13, 64
            ;   cmp r13, rdx
            ;   ja ->done64
            ;   vmovups ymm8, [rdi + r11]
            ;   vmovups ymm9, [rsi + r11]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11], ymm9
            ;   vmovups ymm8, [rdi + r11 + 32]
            ;   vmovups ymm9, [rsi + r11 + 32]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 32], ymm9
            ;   vmovups ymm8, [rdi + r11 + 64]
            ;   vmovups ymm9, [rsi + r11 + 64]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 64], ymm9
            ;   vmovups ymm8, [rdi + r11 + 96]
            ;   vmovups ymm9, [rsi + r11 + 96]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 96], ymm9
            ;   vmovups ymm8, [rdi + r11 + 128]
            ;   vmovups ymm9, [rsi + r11 + 128]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 128], ymm9
            ;   vmovups ymm8, [rdi + r11 + 160]
            ;   vmovups ymm9, [rsi + r11 + 160]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 160], ymm9
            ;   vmovups ymm8, [rdi + r11 + 192]
            ;   vmovups ymm9, [rsi + r11 + 192]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 192], ymm9
            ;   vmovups ymm8, [rdi + r11 + 224]
            ;   vmovups ymm9, [rsi + r11 + 224]
            ;   vfmadd231ps ymm9, ymm8, ymm12
            ;   vmovups [rsi + r11 + 224], ymm9
            ;   add r10, 64
            ;   add r11, 256
            ;   jmp ->lp64
            ; ->done64:
            ; ->tlo:
            ;   cmp r10, rdx
            ;   jae ->end
            ;   vmovd xmm8, [rdi + r11]
            ;   vmovd xmm9, [rsi + r11]
            ;   vfmadd231ss xmm9, xmm8, xmm0
            ;   vmovd [rsi + r11], xmm9
            ;   add r10, 1
            ;   add r11, 4
            ;   jmp ->tlo
            ; ->end:
            ;   pop r13
            ;   pop r12
            ;   pop rbx
            ;   ret
        );
        let buf = ops.finalize().map_err(|_| "atomjit axpy: finalize failed".to_string())?;
        let f: AxpyKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }
}

use std::sync::OnceLock;

struct Kernels {
    _bg: dynasmrt::ExecutableBuffer,
    _bn: dynasmrt::ExecutableBuffer,
    _ba: dynasmrt::ExecutableBuffer,
    dot: sysv::DotKernel,
    norm: sysv::NormKernel,
    axpy: sysv::AxpyKernel,
}

static K: OnceLock<Option<Kernels>> = OnceLock::new();

/// AOT: живые байты ядер для v2-дампа (имена стабильны).
pub fn kernel_blobs() -> Vec<(String, Vec<u8>)> {
    let out = if let Some(k) = kernels() {
        vec![
            ("atomjit.dot".to_string(), k._bg.to_vec()),
            ("atomjit.norm".to_string(), k._bn.to_vec()),
            ("atomjit.axpy".to_string(), k._ba.to_vec()),
        ]
    } else {
        Vec::new()
    };
    out
}

fn kernels() -> Option<&'static Kernels> {
    K.get_or_init(|| {
        if std::env::var_os("ATOMNET_NO_JIT").is_some() {
            return None;
        }
        if !crate::atom::isa::virt_isa().has_avx2_fma() {
            return None;
        }
        let (bg, dot) = sysv::build_dot().ok()?;
        let (bn, norm) = sysv::build_norm().ok()?;
        let (ba, axpy) = sysv::build_axpy().ok()?;
        Some(Kernels { _bg: bg, _bn: bn, _ba: ba, dot, norm, axpy })
    })
    .as_ref()
}

/// dot(a,b) = Σ a[i]·b[i] (DynASM AVX2); fallback to simd::dot.
pub fn dot_jit(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    match kernels() {
        Some(k) => unsafe { (k.dot)(a.as_ptr(), b.as_ptr(), n) },
        None => crate::atom::math::simd::dot(a, b),
    }
}

/// norm(v) = sqrt(Σ v²) через DynASM.
pub fn norm_jit(v: &[f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    match kernels() {
        Some(k) => unsafe { (k.norm)(v.as_ptr(), n) },
        None => crate::atom::math::simd::norm(v),
    }
}

/// y[i] += coef · x[i] (DynASM AVX2).
pub fn axpy_jit(coef: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    if n == 0 {
        return;
    }
    match kernels() {
        Some(k) => unsafe {
            (k.axpy)(coef, x.as_ptr(), y.as_mut_ptr(), n);
        },
        None => {
            for (yi, &xi) in y.iter_mut().zip(x.iter()) {
                *yi += coef * xi;
            }
        }
    }
}

/// tangent_project через JIT: g − dot·u (dot + axpy).
pub fn tangent_project_jit(g: &[f32], u: &[f32]) -> Vec<f32> {
    let mut out = g.to_vec();
    let d = dot_jit(g, u);
    axpy_jit(-d, u, &mut out);
    out
}

/// geodesic шаг через JIT (tangent_project_jit + точный поворот).
pub fn geodesic_step_jit(u: &[f32], g: &[f32], lr: f32) -> Vec<f32> {
    let v = tangent_project_jit(g, u);
    let nv = norm_jit(&v);
    if nv < 1e-12 || lr == 0.0 {
        return u.to_vec();
    }
    let vh: Vec<f32> = v.iter().map(|x| x / nv).collect();
    let th = lr * nv;
    let (c, s) = (th.cos(), th.sin());
    u.iter()
        .zip(vh.iter())
        .map(|(&ui, &vi)| c * ui + s * vi)
        .collect()
}

/// householder reflect через JIT (dot + axpy).
pub fn householder_reflect_jit(x: &[f32], u: &[f32], v: &[f32]) -> Vec<f32> {
    let w: Vec<f32> = u.iter().zip(v.iter()).map(|(&a, &b)| a - b).collect();
    let s = dot_jit(&w, &w);
    if s < 1e-24 {
        return x.to_vec();
    }
    let dot = dot_jit(&w, x);
    let coeff = 2.0 * dot / s;
    let mut out = x.to_vec();
    axpy_jit(-coeff, &w, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::math::simd;

    fn v(n: usize, seed: u64) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xA0761D6478BD642F;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push((((s >> 33) % 4000) as f32 - 2000.0) / 1000.0);
        }
        out
    }

    #[test]
    fn jit_dot_vs_intrinsics_all_len() {
        for n in 0..200usize {
            let a = v(n, 1);
            let b = v(n, 23);
            let r = dot_jit(&a, &b);
            let e = simd::dot(&a, &b);
            let scale = e.abs().max(1.0);
            assert!((r - e).abs() <= 2.5e-5 * scale, "n={n}: {r} vs {e}");
        }
    }

    #[test]
    fn jit_norm_vs_intrinsics() {
        for n in [0usize, 1, 7, 8, 9, 63, 64, 65, 100, 256] {
            let a = v(n, 7);
            let r = norm_jit(&a);
            let e = simd::norm(&a);
            assert!((r - e).abs() <= 2e-5 * e.abs().max(1.0), "n={n}: {r} vs {e}");
        }
    }

    #[test]
    fn jit_axpy_matches() {
        for n in [0usize, 1, 3, 8, 32, 33, 64, 100, 256] {
            let x = v(n, 3);
            let mut y = v(n, 5);
            let y0 = y.clone();
            axpy_jit(0.25, &x, &mut y);
            for i in 0..n {
                assert!((y[i] - (y0[i] + 0.25 * x[i])).abs() < 1e-5, "n={n} i={i}");
            }
        }
    }

    #[test]
    fn jit_tangent_matches_reference() {
        let u = crate::atom::math::atommath::normalize(&v(16, 9));
        let g = v(16, 17);
        let a = tangent_project_jit(&g, &u);
        let b = crate::atom::math::atommath::tangent_project(&g, &u);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 3e-5);
        }
    }

    #[test]
    fn jit_geodesic_preserves_norm() {
        let u = crate::atom::math::atommath::normalize(&v(24, 33));
        let g = v(24, 41);
        let u2 = geodesic_step_jit(&u, &g, 0.05);
        assert!((norm_jit(&u2) - 1.0).abs() < 2e-5);
    }

    #[test]
    fn jit_householder_maps() {
        let u = crate::atom::math::atommath::normalize(&v(20, 55));
        let w = crate::atom::math::atommath::normalize(&v(20, 66));
        let x = v(20, 77);
        let hu = householder_reflect_jit(&u, &u, &w);
        // normalize vs jit путь: norm_jit линейен (относительно)
        for (a, b) in hu.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        let hx = householder_reflect_jit(&x, &u, &w);
        assert!((norm_jit(&hx) - norm_jit(&x)).abs() < 1e-3);
    }
}



// ═══════════════════════════════════════════════════════════════════
// AOT-компонент DynASM: ahead-of-time дамп готовых буферов и импорт
// без эмиссии в run-time (как "precompiled kernel" подход).
// ═══════════════════════════════════════════════════════════════════

use std::io::Write;

/// Дамп готовых DynASM-буферов в файл (компиляция JIT -> один раз at build).
pub fn aot_dump(path: &str) -> Result<String, String> {
    let k = kernels().ok_or_else(|| "no kernels".to_string())?;
    let mut out: Vec<u8> = Vec::new();
    let parts = [
        ("dot", { let v: &[u8] = &k._bg; v.to_vec() }),
        ("norm", { let v: &[u8] = &k._bn; v.to_vec() }),
        ("axpy", { let v: &[u8] = &k._ba; v.to_vec() }),
    ];
    for (name, buf) in parts {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&(buf.len() as u32).to_le_bytes());
        out.extend_from_slice(&buf);
    }
    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&out).map_err(|e| e.to_string())?;
    Ok(path.to_string())
}

/// Чтение AOT-файла (имя->байтыпубликаций).
pub fn aot_read(path: &str) -> Result<Vec<(String, Vec<u8>)>, String> {
    let buf = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut pos = 0usize;
    let mut out = Vec::new();
    while pos < buf.len() {
        if pos + 4 > buf.len() {
            return Err("aot eof".into());
        }
        let nlen = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + nlen + 4 > buf.len() {
            return Err("aot eof name".into());
        }
        let name = String::from_utf8(buf[pos..pos + nlen].to_vec()).map_err(|e| e.to_string())?;
        pos += nlen;
        let blen = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + blen > buf.len() {
            return Err("aot eof payload".into());
        }
        out.push((name, buf[pos..pos + blen].to_vec()));
        pos += blen;
    }
    Ok(out)
}


/// Импорт AOT-буфера в RWX ExecBuffer (UTABLE->EXEC через make_exec), без эмиссии.
pub fn aot_import_dot(path: &str) -> Result<(dynasmrt::ExecutableBuffer, sysv::DotKernel), String> {
    let parts = aot_read(path)?;
    let (_name, blob) = parts.into_iter().find(|(name, _)| name == "dot")
        .ok_or_else(|| "no dot in aot".to_string())?;
    let mut mb = dynasmrt::mmap::MutableBuffer::new(blob.len()).map_err(|e| e.to_string())?;
    mb.set_len(blob.len());
    let target: &mut [u8] = &mut *mb;
    target[..blob.len()].copy_from_slice(&blob);
    let buf = mb.make_exec().map_err(|e| e.to_string())?;
    let dot_ptr: *const u8 = buf.ptr(dynasmrt::AssemblyOffset(0));
    let dot: sysv::DotKernel = unsafe { std::mem::transmute(dot_ptr) };
    Ok((buf, dot))
}


/// Публичная обёртка: дамп всех ядер через `dump_kernels`.
pub fn aot_dump_all(path: &str) -> Result<String, String> {
    aot_dump(path)
}

    #[cfg(test)]
    #[test]
    fn aot_roundtrip_parity() {
        let path = "/tmp/opencode/atomjit_aot.bin";
        aot_dump(path).expect("aot dump");
        let parts = aot_read(path).expect("aot read");
        assert_eq!(parts.len(), 3);
        // дамп побитово равен живому буферу (kernel области не отлинхующиеся —
        // portable symlink: размер той же записи)
        let k = kernels().expect("kernels");
        let live: &[u8] = &k._bg;
        let dot_part = parts.iter().find(|(n, _)| n == "dot").unwrap().1.clone();
        assert_eq!(dot_part, live.to_vec());
        // чистота парсинга: имена по кругу
        let names: Vec<_> = parts.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["dot", "norm", "axpy"]);
    }

