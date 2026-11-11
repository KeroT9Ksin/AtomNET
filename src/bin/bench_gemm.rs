use std::time::Instant;

fn env_usize(name: &str, def: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(def)
}

fn main() {
    let n: usize = env_usize("BENCH_N", 2048);
    let m: usize = env_usize("BENCH_M", n);
    let k: usize = env_usize("BENCH_K", n);
    let parts: usize = env_usize("BENCH_PARTS", 6);
    let reps: usize = env_usize("BENCH_REPS", 7);
    let jit = std::env::var("BENCH_JIT").map(|v| v != "0").unwrap_or(true);
    let silu = std::env::var("BENCH_SILU").map(|v| v != "0").unwrap_or(false);
    atomnet::atom::flash::jit::set_jit_enabled(jit);

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.01).collect();
    let bias: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let mut c = vec![0.0f32; m * n];

    let run = |c: &mut Vec<f32>| {
        if silu {
            atomnet::atom::flash::flash_gemm::gemm_fused_silu_into(
                &a, m, k, &b, n, &bias, parts, c,
            );
        } else {
            atomnet::atom::flash::flash_gemm::gemm_blocked_into(
                &a, m, k, &b, n, parts, c,
            );
        }
    };
    run(&mut c);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        run(&mut c);
        best = best.min(t.elapsed().as_secs_f64());
    }
    let flops = 2.0 * (m as f64) * (k as f64) * (n as f64);
    println!(
        "bin {m}x{k}x{n} parts={parts} jit={jit} silu={silu}: {:.2} ms, {:.1} GF/s",
        best * 1e3,
        flops / best / 1e9
    );
}
