import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    np.random.seed(5)
    rows, cols = 2048, 1024
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = (np.random.randn(rows) * 0.001).astype(np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)

    layer = QL(W.tolist(), b.tolist())

    print("=== QairoCursor: состояние ===")
    print(f"hot_fraction      = {layer.cursor_hot_fraction():.3f}")
    print(f"recommend_kernel  = {layer.cursor_recommend_kernel()} (0=f32,1=int4,2=int4+skip)")
    print(f"prefetch_distance = {layer.cursor_prefetch_distance()}")

    print("\n=== Холодный старт: первый вызов ===")
    # без прогрева
    l1 = QL(W.tolist(), b.tolist())
    t0 = time.perf_counter(); l1.forward_linear_np(x); t_cold = (time.perf_counter() - t0) * 1e3

    # с прогревом через курсор
    l2 = QL(W.tolist(), b.tolist())
    l2.cold_start()
    t0 = time.perf_counter(); l2.forward_linear_np(x); t_warm = (time.perf_counter() - t0) * 1e3

    print(f"первый вызов БЕЗ cold_start : {t_cold:8.3f} ms")
    print(f"первый вызов С    cold_start : {t_warm:8.3f} ms  (x{t_cold / max(t_warm, 1e-9):.2f})")

    # корректность не пострадала
    y1 = np.asarray(l1.forward_linear_np(x))
    y2 = np.asarray(l2.forward_linear_np(x))
    d = float(np.abs(y1 - y2).max())
    print(f"\nmax |y_cold - y_warm| = {d:.3e} (должно быть ~0)")
    assert d < 1e-6

    print("\n✅ QAIRO CURSOR TEST PASSED")


if __name__ == "__main__":
    main()