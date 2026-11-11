# tests/bench_speed_vs_numpy.py
# Сводный тест скорости: AtomNET vs NumPy (1 поток).
# Медиана из 3 замеров. Вывод: таблица + CSV.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import csv
import time

import numpy as np
import atomnet

OUT_CSV = "speed_vs_numpy.csv"


def median_ms(fn, reps, batches=3):
    for _ in range(3):
        fn()
    times = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        times.append((time.perf_counter() - t0) / reps * 1000.0)
    times.sort()
    return times[len(times) // 2]


def row(name, fn_np, fn_at, reps=30):
    t_np = median_ms(fn_np, reps)
    t_at = median_ms(fn_at, reps)
    ratio = t_np / max(t_at, 1e-9)
    print(f"{name:28s} numpy {t_np:9.3f} ms | atom {t_at:9.3f} ms | x{ratio:8.2f}")
    return [name, round(t_np, 4), round(t_at, 4), round(ratio, 3)]


def getc(flat, sub, mod):
    if hasattr(atomnet, flat):
        return getattr(atomnet, flat)
    return getattr(getattr(atomnet, mod), sub)


def main():
    np.random.seed(0)
    results = []

    print("=== AtomNET vs NumPy (1 поток, медиана 3x) ===\n")

    # ── SIMD-ядра ──
    n = 1 << 22
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    results.append(row("dot (4M)", lambda: np.dot(a, b), lambda: atomnet.dot_np(a, b), 30))
    results.append(row("norm (4M)", lambda: np.linalg.norm(a), lambda: atomnet.norm_np(a), 30))
    results.append(row("l2 (4M)", lambda: np.linalg.norm(a - b), lambda: atomnet.norm_np(a - b), 20))
    results.append(row("xor_signature (4M)", lambda: float(np.sum(a > 0)), lambda: atomnet.xor_signature_np(a), 30))
    results.append(row("hamming", lambda: int(np.random.randint(0, 2, 64).sum()), lambda: atomnet.hamming_distance(0xABCDEF0123456789, 0x123456789ABCDEF0), 2000))

    # ── Micron ──
    q, mu, sc = atomnet.quantize_int8_np(a)
    results.append(row("quantize_int8 (4M)", lambda: np.clip(np.round((a - a.mean()) / np.maximum(np.max(np.abs(a - a.mean())) / 127.0, 1e-12)), -127, 127).astype(np.int8), lambda: atomnet.quantize_int8_np(a), 10))
    results.append(row("dequantize_int8 (4M)", lambda: q.astype(np.float32) * sc + mu, lambda: atomnet.dequantize_int8_np(q, mu, sc), 10))

    # ── Rineta-слитый шаг (momentum + update за 1 проход) ──
    wa = np.zeros(n, dtype=np.float32)
    wn = np.zeros(n, dtype=np.float32)
    rin = atomnet.PyRin.from_numpy(np.zeros(n, dtype=np.float32))

    def np_step():
        nonlocal wn
        tmp = 0.9 * wn + 0.1 * a
        wn -= 0.01 * tmp

    results.append(row("rin fused step (4M)", np_step, lambda: rin.apply_into_inplace_np(wa, a, 0.01, 0.9), 20))

    # ── Слитые векторные функции (без временных массивов) ──
    outb = np.empty_like(a)
    results.append(row("mse (4M)", lambda: np.mean((a - b) ** 2), lambda: atomnet.mse_np(a, b), 20))
    results.append(row("mean+var (4M)", lambda: (a.mean(), a.var()), lambda: atomnet.mean_var_np(a), 20))
    results.append(row("affine_into (4M)", lambda: a * 1.5 - 0.7, lambda: atomnet.affine_into_np(a, 1.5, -0.7, outb), 20))

    # ── Enigma (проекция на сферу, в буфер без аллокаций) ──
    out_buf = np.empty_like(a)
    results.append(row("spherical_norm (4M)", lambda: a / max(float(np.linalg.norm(a)), 1e-12), lambda: atomnet.spherical_normalize_into_np(a, out_buf), 10))

    # ── FLASH dense f32 GEMV ──
    rows, cols = 4096, 2048
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    bb = np.zeros(rows, dtype=np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)
    g = (np.random.randn(rows) * 0.01).astype(np.float32)
    FD = getc("PyFlashDense", "PyFlashDense", "flash")
    fd = FD.from_numpy(W, bb)
    results.append(row("gemv f32 fwd 4096x2048", lambda: W @ x + bb, lambda: fd.gemv_bias_np(x), 50))
    results.append(row("gemv f32 bwd 4096x2048", lambda: W.T @ g, lambda: fd.gemv_t_np(g), 50))

    # ── FLASH int4+resid GEMV (f32-качество) ──
    FC = getc("PyFlashCompressed", "PyFlashCompressed", "flash")
    fc = FC.from_numpy(W, bb)
    err = float(np.abs(fc.gemv_np(x) - (W @ x + bb)).max())
    print(f"\nint4+resid max err vs f32: {err:.2e} (f32-качество)")
    results.append(row("gemv int4+resid fwd", lambda: W @ x + bb, lambda: fc.gemv_np(x), 50))

    # ── FLASH Int16: Q16×Q16 целочисленный SIMD GEMV (madd_epi16) ──
    FI16 = getc("PyFlashInt16", "PyFlashInt16", "flash")
    fi16 = FI16.from_numpy(W, bb)
    err16 = float(np.abs(fi16.gemv_np(x) - (W @ x + bb)).max())
    print(f"int16 max err vs f32: {err16:.2e} (ближе к f32, чем int4+resid)")
    results.append(row("gemv int16 fwd (Q16xQ16)", lambda: W @ x + bb, lambda: fi16.gemv_np(x), 50))

    # ── QairoLivingLayer int4 ──
    QL = getc("PyQairoLivingLayer", "PyQairoLivingLayer", "life")
    r2, c2 = 2048, 1024
    W2 = (np.random.randn(r2, c2) * 0.02).astype(np.float32)
    b2 = np.zeros(r2, dtype=np.float32)
    x2 = (np.random.randn(c2) * 0.02).astype(np.float32)
    ql = QL(W2.tolist(), b2.tolist())
    ql.quantize_to_int4()
    err2 = float(np.abs(ql.forward_linear_np(x2) - (W2 @ x2 + b2)).max())
    print(f"qairo int4 max err: {err2:.2e}")
    results.append(row("qairo int4 fwd 2048x1024", lambda: W2 @ x2 + b2, lambda: ql.forward_linear_np(x2), 50))

    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["bench", "numpy_ms", "atom_ms", "speedup_x"])
        wtr.writerows(results)
    print(f"\nCSV: {OUT_CSV}")

    # Итог: сколько операций быстрее numpy (без микробенчей <0.05 мс)
    real = [r for r in results if r[1] >= 0.05]
    faster = sum(1 for r in real if r[3] > 1.0)
    print(f"\nБыстрее NumPy: {faster}/{len(real)} операций (микробенчи исключены)")


if __name__ == "__main__":
    main()
