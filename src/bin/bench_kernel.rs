use std::time::Instant;

fn main() {
    let jit = atomnet::atom::flash::jit::jit_gemm().expect("JIT GEMM недоступен");
    let kc: usize = std::env::var("KC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    // L1-resident: A 6×kc, B kc×16, C 6×16.
    let a = vec![0.5f32; 6 * kc];
    let b = vec![0.25f32; kc * 16];
    let mut c = vec![0.0f32; 6 * 16];

    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let cp = c.as_mut_ptr();
    let reps = 2_000_000usize;

    for _ in 0..2000 {
        unsafe { (jit.store_any)(ap, bp, cp, kc, 16, 16, kc, std::ptr::null()) };
    }
    let t = Instant::now();
    for _ in 0..reps {
        unsafe { (jit.store_any)(ap, bp, cp, kc, 16, 16, kc, std::ptr::null()) };
    }
    let dt = t.elapsed().as_secs_f64();
    let flops = 2.0 * 6.0 * 16.0 * kc as f64 * reps as f64;
    println!(
        "kernel 6x16x{kc}: {:.1} GF/s, {:.1} ns/call, {:.1} FLOP/cycle @3.55GHz",
        flops / dt / 1e9,
        dt / reps as f64 * 1e9,
        flops / dt / 3.55e9
    );
}
