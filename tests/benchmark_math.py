import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet


def bench(name, fn_np, fn_at, reps=20):
    for _ in range(3):
        fn_np(); fn_at()
    t0 = time.perf_counter()
    for _ in range(reps): fn_np()
    t_np = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps): fn_at()
    t_at = (time.perf_counter() - t0) / reps
    print(f"{name:16s} numpy {t_np*1e3:8.3f} ms | atom {t_at*1e3:8.3f} ms | x{t_np/max(t_at,1e-9):5.2f}")


def main():
    n = 1 << 22
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    print("=== SIMD-ядра (FLASH) vs NumPy ===")
    bench("dot",  lambda: np.dot(a, b),          lambda: atomnet.dot_np(a, b))
    bench("norm", lambda: np.linalg.norm(a),     lambda: atomnet.norm_np(a))
    bench("l2",   lambda: np.linalg.norm(a - b), lambda: atomnet.norm_np(a - b))

    print("\n=== Micron vs NumPy-ручное ===")
    def np_quant(x):
        mu = x.mean(); m = np.max(np.abs(x - mu))
        sc = np.maximum(m / 127.0, 1e-12)
        return np.clip(np.round((x - mu) / sc), -127, 127).astype(np.int8), mu, sc

    t0 = time.perf_counter(); q, mu, sc = atomnet.quantize_int8_np(a); tq = (time.perf_counter()-t0)*1e3
    t0 = time.perf_counter(); d = atomnet.dequantize_int8_np(q, mu, sc); td = (time.perf_counter()-t0)*1e3
    err = float(np.max(np.abs(a - d)))
    print(f"micron: quant {tq:.3f} ms, dequant {td:.3f} ms, "
          f"max_err {err:.6f} (bound {sc/2:.6f}) -> {'PASS' if err <= sc/2 + 1e-6 else 'FAIL'}")
    bench("quant", lambda: np_quant(a), lambda: atomnet.quantize_int8_np(a))

    print("\n=== XOR-подписи (связь компонентов) ===")
    bench("xor_sig", lambda: np.sum(a > 0), lambda: atomnet.xor_signature_np(a))
    sa = atomnet.xor_signature_np(a); sb = atomnet.xor_signature_np(b)
    print(f"hamming(a,b) = {atomnet.hamming_distance(sa, sb)} (ожидалось ~32)")

    print("\n=== Qairo/FLASH vs NumPy affine ===")
    FD = getattr(atomnet, "PyFlashDense", None) or atomnet.flash.PyFlashDense
    QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer
    for rows, cols in [(2048, 1024), (4096, 2048)]:
        W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
        bb = (np.random.randn(rows) * 0.001).astype(np.float32)
        x = (np.random.randn(cols) * 0.02).astype(np.float32)
        fd = FD.from_numpy(W, bb); ql = QL(W.tolist(), bb.tolist())
        ql.quantize_to_int4()
        print(f"[{rows}x{cols}]")
        bench("  affine", lambda: W @ x + bb, lambda: fd.gemv_bias_np(x))
        bench("  qairo4", lambda: W @ x + bb, lambda: ql.forward_linear_np(x))

    print("\n=== Rineta (информационно: делает БОЛЬШЕ работы) ===")
    Rin = getattr(atomnet, "PyRin", None) or atomnet.optim.PyRin
    m = 1 << 20
    w = (np.random.randn(m) * 0.02).astype(np.float32)
    g = (np.random.randn(m) * 0.02).astype(np.float32)
    p = np.zeros(m, dtype=np.float32)
    rin = Rin.from_numpy(np.zeros(m, dtype=np.float32))

    def np_mom():
        p2 = 0.9 * p + 0.1 * g
        return w - 1e-3 * p2

    bench("momentum", np_mom, lambda: rin.apply_into_inplace_np(w, g, 1e-3, 0.9))
    print("rineta добавляет когерентность/буст — сравнивай по ценности на шаг, не только по мс")


if __name__ == "__main__":
    main()