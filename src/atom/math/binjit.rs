// src/atom/math/binjit.rs
//! binjit — бинарная вычислительная машина на DynASM (byte-level SSE/AVX2).
//!
//! операций: hamming(a,b)-bytes, xor-bytes, and-fold, popcount8.
//! пакеты — numpy-like (x convolve через Python-бисты), и наши собственные
//! семплоны: byte-popcount (Hamming для atom-роутинг sig), xor по блокам.
//!
//! ABI: sysv64; доступности: POPCNT + AVX2 CPUID, иначе fallback на Rust.

#[cfg(target_arch = "x86_64")]
mod sysv {
    use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi, ExecutableBuffer};

    pub type PopcntFn = unsafe extern "sysv64" fn(*const u8, usize) -> u32;
    pub type HammFn = unsafe extern "sysv64" fn(*const u8, *const u8, usize) -> u32;
    pub type XorFn = unsafe extern "sysv64" fn(*const u8, *const u8, *mut u8, usize);

    pub fn build_popcnt() -> Result<(ExecutableBuffer, PopcntFn), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; xor r12d, r12d                     // acc (sum bytes)
            ; test rsi, rsi
            ; jz ->done
            ; mov rax, rsi                       // rax = length
            ; shr rax, 3                         // full qwords
            ; mov rcx, rax
            ; shl rcx, 3                         // bytes processed
            ; xor r10d, r10d                     // counter
            ; ->loop8:
            ;   mov rax, r10
            ;   add rax, 8
            ;   cmp rax, rsi
            ;   ja ->tail
            ;   mov rax, [rdi + r10]
            ;   popcnt rax, rax
            ;   add r12, rax
            ;   add r10, 8
            ;   jmp ->loop8
            ; ->tail:
            ;   mov rax, rsi
            ;   mov rcx, r10
            ;   cmp rcx, rax
            ;   jae ->ret
            ;   movzx rax, [rdi + rcx]
            ;   popcnt rax, rax
            ;   add r12, rax
            ;   inc r10
            ;   jmp ->tail
            ; ->done:
            ; ->ret:
            ;   mov eax, r12d
            ;   pop r12
            ;   pop rbx
            ;   ret
        );
        let buf = ops.finalize().map_err(|_| "popcnt: finalize".to_string())?;
        let f: PopcntFn = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    pub fn build_hamming() -> Result<(ExecutableBuffer, HammFn), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; xor r12d, r12d
            ; test rdx, rdx
            ; jz ->ret
            ; xor r10d, r10d
            ; ->loop8:
            ;   mov rax, r10
            ;   add rax, 8
            ;   cmp rax, rdx
            ;   ja ->tail
            ;   mov rax, [rdi + r10]
            ;   mov rcx, [rsi + r10]
            ;   xor rax, rcx
            ;   popcnt rax, rax
            ;   add r12, rax
            ;   add r10, 8
            ;   jmp ->loop8
            ; ->tail:
            ;   mov rax, r10
            ;   cmp rax, rdx
            ;   jae ->ret
            ;   movzx rcx, [rdi + rax]
            ;   movzx r8, [rsi + rax]
            ;   xor ecx, r8d
            ;   popcnt rcx, rcx
            ;   add r12, rcx
            ;   inc rax
            ;   mov r10, rax
            ;   jmp ->tail
            ; ->ret:
            ;   mov eax, r12d
            ;   pop r12
            ;   pop rbx
            ;   ret
        );
        let buf = ops.finalize().map_err(|_| "hamming: finalize".to_string())?;
        let f: HammFn = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    pub fn build_xor() -> Result<(ExecutableBuffer, XorFn), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        // ABI (a=rdi, b=rsi, y=rdx, n=rcx)
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; push r13
            ; mov r12, rcx
            ; xor r10d, r10d
            ; cmp r12, 32
            ; jb ->tail
            ; ->loop32:
            ;   mov r13, r10
            ;   add r13, 32
            ;   cmp r13, r12
            ;   ja ->tail
            ;   movdqu xmm0, [rdi + r10]
            ;   movdqu xmm1, [rsi + r10]
            ;   pxor xmm0, xmm1
            ;   movdqu [rdx + r10], xmm0
            ;   movdqu xmm0, [rdi + r10 + 16]
            ;   movdqu xmm1, [rsi + r10 + 16]
            ;   pxor xmm0, xmm1
            ;   movdqu [rdx + r10 + 16], xmm0
            ;   add r10, 32
            ;   mov r13, r10
            ;   add r13, 32
            ;   cmp r13, r12
            ;   jbe ->loop32
            ; ->tail:
            ;   mov r13, r10
            ;   cmp r13, r12
            ;   jae ->end
            ;   movzx rax, [rdi + r13]
            ;   movzx rcx, [rsi + r13]
            ;   xor eax, ecx
            ;   mov [rdx + r13], al
            ;   inc r13
            ;   jmp ->tail
            ; ->end:
            ;   pop r13
            ;   pop r12
            ;   pop rbx
            ;   ret
        );
        let buf = ops.finalize().map_err(|_| "xor: finalize".to_string())?;
        let f: XorFn = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }
}

use std::sync::OnceLock;

struct Bfk {
    _b0: dynasmrt::ExecutableBuffer,
    _b1: dynasmrt::ExecutableBuffer,
    _b2: dynasmrt::ExecutableBuffer,
    popcnt: sysv::PopcntFn,
    hamm: sysv::HammFn,
    xor: sysv::XorFn,
}

static B: OnceLock<Option<Bfk>> = OnceLock::new();

fn backend() -> Option<&'static Bfk> {
    B.get_or_init(|| {
        if std::env::var_os("ATOMNET_NO_JIT").is_some() {
            return None;
        }
        if !(crate::atom::isa::virt_isa().avx2
            && crate::atom::isa::virt_isa().popcnt)
        {
            return None;
        }
        let (b0, popcnt) = sysv::build_popcnt().ok()?;
        let (b1, hamm) = sysv::build_hamming().ok()?;
        let (b2, xor) = sysv::build_xor().ok()?;
        Some(Bfk { _b0: b0, _b1: b1, _b2: b2, popcnt, hamm, xor })
    })
    .as_ref()
}

fn scalar_hamming(a: &[u8], b: &[u8], n: usize) -> u32 {
    a.iter().take(n).zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

pub fn popcount_jit(a: &[u8]) -> u32 {
    if let Some(b) = backend() {
        unsafe { (b.popcnt)(a.as_ptr(), a.len()) }
    } else {
        a.iter().map(|x| x.count_ones()).sum()
    }
}

pub fn hamming_jit(a: &[u8], b: &[u8]) -> u32 {
    let n = a.len().min(b.len());
    if let Some(bk) = backend() {
        unsafe { (bk.hamm)(a.as_ptr(), b.as_ptr(), n) }
    } else {
        scalar_hamming(a, b, n)
    }
}

pub fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    // Простой и быстрый Rust-цикл по u64 (xorbytegpuh не влияет на внутренний
    // JIT-путь — popcnt/hamming — основные вычислительные ядра).
    let n = a.len().min(b.len());
    let mut out = vec![0u8; n];
    let ac = (n / 8) * 8;
    for i in (0..ac).step_by(8) {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        out[i..i + 8].copy_from_slice(&(x ^ y).to_le_bytes());
    }
    for i in ac..n {
        out[i] = a[i] ^ b[i];
    }
    out
}

pub fn xor_jit(a: &[u8], b: &[u8]) -> Vec<u8> {
    xor_bytes(a, b)
}

#[cfg(test)]
mod tests_binjit {
    use super::*;

    fn v(n: usize, seed: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xA0761D6478BD642F;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push((s >> 33) as u8);
        }
        out
    }

    #[test]
    fn popcnt_jit_vs_scalar() {
        for n in 0..300usize {
            let a = v(n, 1);
            let r = popcount_jit(&a);
            let e: u32 = a.iter().map(|x| x.count_ones()).sum();
            assert_eq!(r, e, "n={n}");
        }
    }

    #[test]
    fn hamming_jit_vs_scalar() {
        for n in 0..200usize {
            let a = v(n, 3);
            let b = v(n, 7);
            assert_eq!(hamming_jit(&a, &b), scalar_hamming(&a, &b, n), "n={n}");
        }
    }

    #[test]
    fn xor_jit_matches() {
        for n in [0usize, 1, 7, 8, 15, 31, 32, 33, 64, 100, 256] {
            let a = v(n, 11);
            let b = v(n, 13);
            let c = xor_jit(&a, &b);
            for i in 0..n {
                assert_eq!(c[i], a[i] ^ b[i]);
            }
        }
    }
}
