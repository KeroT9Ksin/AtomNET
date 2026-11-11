import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    np.random.seed(5)
    rows, cols = 2048, 1024
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = (np.random.randn(rows) * 0.001).astype(np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)

    # Эталон f32
    y_f32 = W @ x + b

    # int4 + int8-остаток
    ql = QL(W.tolist(), b.tolist())
    ql.quantize_to_int4()
    y_res = np.asarray(ql.forward_linear_np(x))

    abs_err = float(np.abs(y_res - y_f32).max())
    mean_abs = float(np.abs(y_res - y_f32).mean())
    max_y = float(np.abs(y_f32).max())
    mean_y = float(np.abs(y_f32).mean())
    rel = abs_err / max_y
    mean_rel = mean_abs / mean_y

    print(f"max  |int4+resid - f32| = {abs_err:.3e}")
    print(f"mean |int4+resid - f32| = {mean_abs:.3e}")
    print(f"rel err (max)  = {rel:.3e}")
    print(f"rel err (mean) = {mean_rel:.3e}")
    print(f"max |y_f32| = {max_y:.3e}, mean |y_f32| = {mean_y:.3e}")
    print(f"layer mem (dense+compressed) / f32 = {ql.memory_bytes()/(rows*cols*4):.2f}x (инфо)")

    # Абсолютная ошибка мала -> f32-подобие (выходы сами по себе ~1e-2)
    assert abs_err < 1e-3, "абсолютная ошибка должна быть < 1e-3"
    # Относительная <= 1% (порог для малых выходов)
    assert rel < 1e-2, "относительная ошибка должна быть < 1%"

    print("\n✅ FLASH RESID GEMV TEST PASSED")


if __name__ == "__main__":
    main()