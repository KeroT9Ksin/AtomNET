import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet


def get_class(flat_name, sub_name):
    if hasattr(atomnet, flat_name):
        return getattr(atomnet, flat_name)
    if hasattr(atomnet, "flash") and hasattr(atomnet.flash, sub_name):
        return getattr(atomnet.flash, sub_name)
    return None


def bench(name, fn, reps=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    rows, cols = 4096, 2048

    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = np.zeros(rows, dtype=np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)
    g = (np.random.randn(rows) * 0.01).astype(np.float32)

    FlashDense = get_class("PyFlashDense", "PyFlashDense")
    FlashCompressed = get_class("PyFlashCompressed", "PyFlashCompressed")

    if FlashDense is None:
        raise RuntimeError("PyFlashDense not found")

    # ═══ Корректность dense ═══
    fd = FlashDense.from_numpy(W, b)

    y_np = W @ x + b
    y_fl = fd.gemv_bias_np(x)
    diff_fwd = float(np.abs(y_np - y_fl).max())

    gt_np = W.T @ g
    gt_fl = fd.gemv_t_np(g)
    diff_bwd = float(np.abs(gt_np - gt_fl).max())

    print("=== FLASH-CORE (dense f32) vs NumPy BLAS ===")
    print(f"forward max diff : {diff_fwd:.3e}")
    print(f"backward max diff: {diff_bwd:.3e}")

    # ═══ Скорость dense ═══
    t_np_fwd = bench("numpy fwd", lambda: W @ x + b)
    t_fl_fwd = bench("flash fwd", lambda: fd.gemv_bias_np(x))
    t_np_bwd = bench("numpy bwd", lambda: W.T @ g)
    t_fl_bwd = bench("flash bwd", lambda: fd.gemv_t_np(g))

    print(f"forward : numpy {t_np_fwd:8.3f} ms | flash {t_fl_fwd:8.3f} ms | x{t_np_fwd/max(t_fl_fwd,1e-9):5.2f}")
    print(f"backward: numpy {t_np_bwd:8.3f} ms | flash {t_fl_bwd:8.3f} ms | x{t_np_bwd/max(t_fl_bwd,1e-9):5.2f}")

    # ═══ FLASH int4 (настоящее оружие против BLAS) ═══
    if FlashCompressed is None:
        print("\nPyFlashCompressed not found; int4 section skipped.")
        return

    print("\n=== FLASH int4 (compressed) vs NumPy BLAS ===")

    fc = FlashCompressed.from_numpy(W, b)
    fc.xor_cold_start()

    y_fc = fc.gemv_np(x)
    diff_i4 = float(np.abs(y_np - y_fc).max())

    print(f"int4 max diff vs NumPy: {diff_i4:.3e}  (ожидаемо 1e-3..1e-2)")
    print(f"memory: {fc.memory_bytes()/1e6:.2f} MB "
          f"(сжатие {fc.compression_ratio():.1f}x)")
    print(f"hot rows: {len(fc.hot_rows())}/{rows}")

    t_fc = bench("flash int4", lambda: fc.gemv_np(x))

    print(f"NumPy BLAS : {t_np_fwd:8.3f} ms")
    print(f"FLASH int4 : {t_fc:8.3f} ms  (x{t_np_fwd/max(t_fc,1e-9):.2f} быстрее)")


if __name__ == "__main__":
    main()
