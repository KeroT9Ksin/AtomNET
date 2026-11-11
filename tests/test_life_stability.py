import time

import numpy as np
import atomnet


def bench(name, fn, iters=20):
    fn()

    start = time.perf_counter()

    for _ in range(iters):
        fn()

    elapsed = (time.perf_counter() - start) / iters

    print(f"{name:42} {elapsed * 1000:10.4f} ms")


def main():
    print("AtomNET version:", atomnet.version())

    out_features = 64
    in_features = 256

    W = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    b = np.zeros(out_features, dtype=np.float32)

    layer = atomnet.PyQairoLivingLayer(W.tolist(), b.tolist())

    # Берём вход, согласованный с первой строкой веса.
    # Это должно давать высокую когерентность.
    x_aligned = W[0].copy()

    print("\n=== Before stability warmup ===")
    print("row 0 state:", layer.row_state(0))
    print("row 0 gate:", layer.row_gate(0))
    print("row 0 energy:", layer.row_energy(0))
    print("row 0 coherence:", layer.row_coherence(0))
    print("row 0 stability:", layer.row_stability(0))

    print("\n=== Stability warmup ===")

    for i in range(20):
        layer.forward_np(x_aligned)

        if i % 5 == 0:
            print(
                f"step {i:02d} | "
                f"state={layer.row_state(0):8s} | "
                f"gate={layer.row_gate(0):.4f} | "
                f"C={layer.row_coherence(0):+.4f} | "
                f"S={layer.row_stability(0):.4f} | "
                f"E={layer.row_energy(0):.6f}"
            )

    print("\n=== After stability warmup ===")
    print("row 0 state:", layer.row_state(0))
    print("row 0 gate:", layer.row_gate(0))
    print("row 0 energy:", layer.row_energy(0))
    print("row 0 coherence:", layer.row_coherence(0))
    print("row 0 stability:", layer.row_stability(0))

    state = layer.row_state(0)
    gate = layer.row_gate(0)

    if state == "stable" and gate >= 0.99:
        print("\nSTABILITY TEST PASSED")
    else:
        print("\nWARNING: row did not become stable")
        print("This can happen if input is not coherent enough.")

    # ============================================================
    # Мини-бенчмарк
    # ============================================================

    x = np.random.randn(in_features).astype(np.float32) * 0.02

    print("\n=== QairoLivingLayer benchmark ===")

    bench(
        "numpy_linear",
        lambda: W @ x + b,
    )

    bench(
        "qairo_linear",
        lambda: layer.forward_linear_np(x),
    )

    bench(
        "qairo_gated_frozen",
        lambda: layer.forward_gated_frozen_np(x),
    )

    bench(
        "qairo_living",
        lambda: layer.forward_np(x),
    )

    # ============================================================
    # Корректность линейного режима
    # ============================================================

    y_np = W @ x + b
    y_linear = layer.forward_linear_np(x)

    diff = np.abs(y_np - np.asarray(y_linear)).max()

    print("\nqairo_linear max diff:", float(diff))

    assert diff < 1e-3

    print("\nLIFE STABILITY TEST FINISHED")


if __name__ == "__main__":
    main()