# benchmarks/bench_atomnet.py
import time

import numpy as np
import atomnet


def bench(name, fn, iters=10):
    # warmup
    fn()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) / iters

    print(f"{name:42} {elapsed * 1000:10.4f} ms")


def main():
    print("AtomNET benchmark")
    print("version:", atomnet.version())

    n = 1_000_000

    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    # WARNING:
    # list conversion is done once here, but PyO3 still converts
    # Python list -> Vec<f32> on each call.
    x_list = x.tolist()
    y_list = y.tolist()

    print("\n=== Base math ===")

    bench(
        "numpy_norm",
        lambda: np.linalg.norm(x),
    )

    bench(
        "atomnet_norm",
        lambda: atomnet.norm(x_list),
    )

    bench(
        "numpy_dot",
        lambda: float(np.dot(x, y)),
    )

    bench(
        "atomnet_dot",
        lambda: atomnet.dot(x_list, y_list),
    )

    print("\n=== Enigma sphere ===")

    small_n = 262_144
    xs = np.random.randn(small_n).astype(np.float32)
    xs_list = xs.tolist()

    bench(
        "numpy_sphere_normalize",
        lambda: xs / (np.linalg.norm(xs) + 1e-12),
    )

    bench(
        "atomnet_sphere_normalize",
        lambda: atomnet.spherical_normalize(xs_list),
    )

    print("\n=== Micron int8 ===")

    bench(
        "atomnet_quantize_int8",
        lambda: atomnet.quantize_int8(xs_list),
    )

    q, mu, sc = atomnet.quantize_int8(xs_list)

    bench(
        "atomnet_dequantize_int8",
        lambda: atomnet.dequantize_int8(q, mu, sc),
    )

    print("\n=== Qairo layer ===")

    out_features = 256
    in_features = 1024

    W = np.random.randn(out_features, in_features).astype(np.float32)
    b = np.random.randn(out_features).astype(np.float32)
    xv = np.random.randn(in_features).astype(np.float32)

    W_list = W.tolist()
    b_list = b.tolist()
    xv_list = xv.tolist()

    layer = atomnet.PyQairoLayer(W_list, b_list)

    bench(
        "numpy_linear_Wx_b",
        lambda: W @ xv + b,
    )

    bench(
        "atomnet_qairo_forward",
        lambda: layer.forward(xv_list),
    )

    print("\n=== Qairo correctness ===")

    y_np = W @ xv + b
    y_q = np.array(layer.forward(xv_list), dtype=np.float32)

    max_err = float(np.abs(y_np - y_q).max())
    mean_err = float(np.abs(y_np - y_q).mean())

    print("qairo max error:", max_err)
    print("qairo mean error:", mean_err)

    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()