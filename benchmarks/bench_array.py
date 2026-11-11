"""Бенчмарк atomnet.array vs numpy: конструкция + операции.

Запуск:  venv/bin/python benchmarks/bench_array.py
"""
import time

import numpy as np

import atomnet
import atomnet.array as ara


def bench(name, fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"{name:44} {ms:10.3f} ms")
    return ms


def main():
    print("AtomNET", atomnet.version(), "| threads", atomnet.parallel_threads())
    n = 1_000_000

    print("\n=== Конструкция (массивы) ===")
    bench("zeros numpy", lambda: np.zeros(n, dtype=np.float32))
    bench("zeros atomnet", lambda: ara.zeros((n,)))
    bench("randn numpy", lambda: (np.random.randn(n) * 1).astype(np.float32))
    bench("randn atomnet", lambda: ara.randn((n,), seed=0))

    print("\n=== Elementwise + редукции (1 проход vs numpy) ===")
    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)
    xa = atomnet.AtomArray.from_numpy(x)
    ya = atomnet.AtomArray.from_numpy(y)
    bench("add numpy", lambda: x + y)
    bench("add atomnet", lambda: xa.add_arr(ya))
    bench("silu numpy", lambda: x / (1 + np.exp(-x)))
    xa2 = atomnet.AtomArray.from_numpy(x)
    bench("silu atomnet in-place", lambda: xa2.silu_())
    # in-place silu без копий:
    xt = xa.tolist()
    bench("norm numpy", lambda: float(np.linalg.norm(x)))
    bench("norm atomnet", lambda: xa.norm())
    bench("mean/var numpy", lambda: (float(x.mean()), float(x.var())))
    bench("mean/var atomnet", lambda: (xa.mean(), xa.var()))

    print("\n=== matmul / gemm ===")
    m = k = s = 512
    A = np.random.randn(m, k).astype(np.float32)
    B = np.random.randn(k, s).astype(np.float32)
    AA = atomnet.AtomArray.from_numpy(A)
    BB = atomnet.AtomArray.from_numpy(B)
    bench("numpy matmul", lambda: A @ B, iters=5)
    bench("atomnet.matmul_np", lambda: atomnet.matmul_np(A, B), iters=5)
    bench("atomnet.gemm_blocked_np", lambda: atomnet.gemm_blocked_np(A, B), iters=5)
    bench("AtomArray.matmul", lambda: AA.matmul(BB), iters=5)
    # fused: gemm+silu одним проходом
    bias = np.zeros(s, dtype=np.float32)
    bench("gemm+silu separate (numpy)", lambda: (A @ B) / (1 + np.exp(-(A @ B))), iters=3)
    bench("gemm_silu fused", lambda: atomnet.gemm_silu_np(A, B, bias), iters=5)

    print("\n=== Физика (этого нет в numpy) ===")
    a = ara.randn((1024,), seed=1)
    print("sig:", hex(a.sig), "alive:", a.is_alive(), "gate:", round(a.gate(), 3))
    a.project_sphere()
    print("norm after sphere:", round(a.norm(), 6))
    q, mu, sc = a.quantize_int8()
    print(f"int8 quant: codes={len(q)} mu={mu:.4f} sc={sc:.6f}")

    print("\n✅ bench_array done")


if __name__ == "__main__":
    main()
