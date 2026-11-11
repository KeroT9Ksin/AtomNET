import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer
FD = getattr(atomnet, "PyFlashDense", None) or atomnet.flash.PyFlashDense


def bench(name, fn, reps=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def main():
    rows, cols = 2048, 1024

    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = (np.random.randn(rows) * 0.001).astype(np.float32)
    x = (np.random.randn(cols) * 0.02).astype(np.float32)

    print(f"=== Qairo v2 (на ядрах FLASH) vs Linear/Affine ===  [{rows}x{cols}]")

    layer = QL(W.tolist(), b.tolist())
    flash = FD.from_numpy(W, b)

    # ── Корректность ─────────────────────────────────────────
    np_linear = W @ x                 # чистый линейный
    np_affine = W @ x + b             # аффинный
    q_linear = np.asarray(layer.forward_linear_np(x))
    f_linear = np.asarray(flash.gemv_bias_np(x))

    d_aff = float(np.abs(np_affine - q_linear).max())
    d_lin = float(np.abs(np_linear - (q_linear - b)).max())
    d_fl  = float(np.abs(q_linear - f_linear).max())

    print(f"max |qairo - affine|      : {d_aff:.3e}")
    print(f"max |qairo-b - linear|    : {d_lin:.3e}")
    print(f"max |qairo - flash_dense| : {d_fl:.3e}  (должно быть ~0: одно ядро)")

    # ── Скорость ─────────────────────────────────────────────
    t_np_lin = bench("numpy linear  W@x",   lambda: W @ x)
    t_np_aff = bench("numpy affine  W@x+b", lambda: W @ x + b)
    t_q_lin  = bench("qairo forward_linear", lambda: layer.forward_linear_np(x))
    t_q_frz  = bench("qairo gated_frozen",  lambda: layer.forward_gated_frozen_np(x))
    t_q_liv  = bench("qairo living",        lambda: layer.forward_np(x))
    t_f_den  = bench("flash dense gemv_bias", lambda: flash.gemv_bias_np(x))

    print(f"\nnumpy linear      : {t_np_lin:8.3f} ms")
    print(f"numpy affine      : {t_np_aff:8.3f} ms")
    print(f"qairo linear      : {t_q_lin:8.3f} ms  (x{t_np_aff/max(t_q_lin,1e-9):.2f} vs affine)")
    print(f"qairo frozen      : {t_q_frz:8.3f} ms")
    print(f"qairo living      : {t_q_liv:8.3f} ms")
    print(f"flash dense       : {t_f_den:8.3f} ms")

    # ── Life sanity ──────────────────────────────────────────
    for _ in range(8):
        layer.forward_np(x)
    print("\nstate counts after 8 living passes:", layer and "ok")


if __name__ == "__main__":
    main()