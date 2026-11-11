import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet

FD = getattr(atomnet, "PyFlashDense", None) or atomnet.flash.PyFlashDense


def bench(fd, x, reps=30, warmup=3):
    for _ in range(warmup):
        fd.gemv_bias_np(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        fd.gemv_bias_np(x)
    return (time.perf_counter() - t0) / reps  # секунды


def main():
    rows, cols = 4096, 2048
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = np.zeros(rows, dtype=np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)
    bytes = rows * cols * 4 + rows * 4 + cols * 4

    fd = FD.from_numpy(W, b)

    print(f"shape {rows}x{cols}, bytes ~{bytes/1e6:.1f} MB")
    print("P (prefetch) | t (ms) | BW_eff (GB/s) | D (vs 40 GB/s)")
    print("-" * 55)

    for P in (2, 4, 6, 8, 10, 12, 16):
        fd.set_prefetch(P)
        t = bench(fd, x)
        bw = bytes / t / 1e9
        D = (t * 40e9) / bytes
        print(f"{P:12d} | {t*1e3:6.3f} | {bw:13.2f} | {D:.2f}")


if __name__ == "__main__":
    main()
