import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import atomnet

FD = getattr(atomnet, "PyFlashDense", None) or atomnet.flash.PyFlashDense
QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    rows, cols = 4096, 2048
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = np.zeros(rows, dtype=np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)

    fd = FD.from_numpy(W, b)
    ql = QL(W.tolist(), b.tolist())
    ql.quantize_to_int4()

    # прогреть
    for _ in range(5):
        fd.gemv_bias_np(x)
        ql.forward_linear_np(x)

    N = 200
    print(f"== AFFINE f32  {rows}x{cols} ({N} iters) ==")
    for _ in range(N):
        fd.gemv_bias_np(x)
    print(f"== QAIRO int4 {rows}x{cols} ({N} iters) ==")
    for _ in range(N):
        ql.forward_linear_np(x)


if __name__ == "__main__":
    main()