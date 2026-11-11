"""FLASH-GEMM/GEMV vs NumPy/BLAS — честные замеры на CPU.

Методика:
  * отдельные фазы: сначала best-of-N для NumPy, затем пауза 0.4 c
    (OpenBLAS держит busy-wait потоки ~десятки мс и иначе душит наши
    воркеры), затем best-of-N для AtomNET;
  * warmup 3, лучший результат из N — шум планировщика не завышает AtomNET;
  * GEMM: parts = физические ядра; NumPy — свои потоки OpenBLAS;
  * GEMV: веса предупакованы в FlashMatrix (копирование W не входит в замер);
  * все матрицы float32, C-contiguous.

Запуск: ./venv/bin/python tests/bench_flash_gemm.py
"""
import time

import numpy as np

import atomnet

rng = np.random.default_rng(42)
PAUSE = 0.4


def best(fn, iters=15, warmup=3):
    for _ in range(warmup):
        fn()
    lo = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < lo:
            lo = dt
    return lo


def bench_gemm(m, k, n):
    a = np.ascontiguousarray(rng.standard_normal((m, k), dtype=np.float32))
    b = np.ascontiguousarray(rng.standard_normal((k, n), dtype=np.float32))

    r_np = a @ b
    r_at = atomnet.gemm_blocked_np(a, b)
    max_diff = float(np.max(np.abs(r_np - r_at)))

    t_np = best(lambda: a @ b)
    time.sleep(PAUSE)
    t_at = best(lambda: atomnet.gemm_blocked_np(a, b))
    time.sleep(PAUSE)
    t_old = best(lambda: atomnet.matmul_np(a, b), iters=5)

    flops = 2.0 * m * k * n
    print(
        f"GEMM {m:5d}x{k:5d}x{n:5d} | numpy {t_np*1e3:8.3f} ms ({flops/t_np/1e9:6.1f} GF/s)"
        f" | atomnet {t_at*1e3:8.3f} ms ({flops/t_at/1e9:6.1f} GF/s)"
        f" | ratio {t_np/t_at:5.2f}x | old matmul {t_np/t_old:5.2f}x"
        f" | max|diff| {max_diff:.1e}"
    )


def bench_gemv(m, n):
    w = np.ascontiguousarray(rng.standard_normal((m, n), dtype=np.float32))
    x = np.ascontiguousarray(rng.standard_normal(n, dtype=np.float32))
    fm = atomnet.FlashMatrix(w)

    r_np = w @ x
    r_at = fm.gemv(x)
    max_diff = float(np.max(np.abs(r_np - r_at)))

    t_np = best(lambda: w @ x, iters=25)
    time.sleep(PAUSE)
    t_at = best(lambda: fm.gemv(x), iters=25)

    gb = m * n * 4 / 1e9
    print(
        f"GEMV {m:5d}x{n:5d}       | numpy {t_np*1e6:8.2f} us ({gb/t_np:6.1f} GB/s)"
        f" | atomnet {t_at*1e6:8.2f} us ({gb/t_at:6.1f} GB/s)"
        f" | ratio {t_np/t_at:5.2f}x | max|diff| {max_diff:.1e}"
    )


def check_correctness():
    print("\n== Correctness ==")
    for m, k, n in [(1, 1, 1), (37, 53, 29), (64, 64, 64), (129, 257, 65)]:
        a = rng.standard_normal((m, k), dtype=np.float32)
        b = rng.standard_normal((k, n), dtype=np.float32)
        err = float(np.max(np.abs(a @ b - atomnet.gemm_blocked_np(a, b))))
        print(f"GEMM {m:4d}x{k:4d}x{n:4d}  max|diff| = {err:.2e}")
    for m, n in [(7, 5), (256, 1000)]:
        w = rng.standard_normal((m, n), dtype=np.float32)
        x = rng.standard_normal(n, dtype=np.float32)
        y = atomnet.FlashMatrix(w).gemv(x)
        err = float(np.max(np.abs(w @ x - y)))
        print(f"GEMV {m:4d}x{n:4d}       max|diff| = {err:.2e}")


if __name__ == "__main__":
    print("== GEMM: C = A @ B (parts = physical cores) ==")
    for m, k, n in [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]:
        bench_gemm(m, k, n)

    print("\n== GEMV: y = W @ x (prepacked weights) ==")
    for m, n in [(256, 256), (1024, 1024), (4096, 4096), (8192, 8192)]:
        bench_gemv(m, n)

    check_correctness()
