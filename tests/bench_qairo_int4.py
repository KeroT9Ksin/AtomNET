import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def bench(name, fn, reps=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    np.random.seed(3)
    rows, cols = 2048, 1024

    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = (np.random.randn(rows) * 0.001).astype(np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)

    layer = QL(W.tolist(), b.tolist())
    y_np = W @ x + b

    # ── f32 путь ──
    y_dense = np.asarray(layer.forward_linear_np(x))
    d_dense = float(np.abs(y_np - y_dense).max())
    t_dense = bench("qairo f32 ", lambda: layer.forward_linear_np(x))

    # ── int4-снимок ──
    layer.quantize_to_int4()
    assert layer.is_int4()

    y_i4 = np.asarray(layer.forward_linear_np(x))
    d_i4 = float(np.abs(y_np - y_i4).max())
    rel = d_i4 / float(np.max(np.abs(y_np)) + 1e-12)
    t_i4 = bench("qairo int4", lambda: layer.forward_linear_np(x))

    t_np = bench("numpy affine", lambda: W @ x + b)

    dense_mb = rows * cols * 4 / 1e6
    int4_mb = (rows * cols / 2 + (rows * cols // 256) * 12) / 1e6

    print(f"max |f32  - numpy| = {d_dense:.3e}")
    print(f"max |int4 - numpy| = {d_i4:.3e}  (rel {rel:.2%})")
    print(f"numpy {t_np:8.3f} ms | f32 {t_dense:8.3f} ms | int4 {t_i4:8.3f} ms")
    print(f"int4 vs numpy: x{t_np / max(t_i4, 1e-9):.2f}")
    print(f"int4 vs f32  : x{t_dense / max(t_i4, 1e-9):.2f}")
    print(f"memory: dense {dense_mb:.1f} MB -> int4 ~{int4_mb:.2f} MB "
          f"({dense_mb / int4_mb:.1f}x)")

    # ── критерии ──
    assert d_dense < 1e-4, "f32 путь должен быть точным"
    assert rel < 1.0, "int4 деградировал сильнее допустимого для 4 бит"
    assert t_i4 < t_dense, "int4 должен быть быстрее f32"

    # ── инвалидация снимка при обучении ──
    g = np.random.randn(rows, cols).astype(np.float32) * 1e-3
    gb = np.random.randn(rows).astype(np.float32) * 1e-3
    layer.sgd_step_np(g, gb, 0.1)
    assert not layer.is_int4(), "sgd_step должен инвалидировать int4"

    print("\n✅ QAIRO INT4 TEST PASSED")


if __name__ == "__main__":
    main()