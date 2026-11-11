# benchmarks/benchmark_math.py
import time

import numpy as np
import atomnet


def bench(name, fn, iters=10):
    try:
        fn()
    except Exception as e:
        print(f"{name:46} SKIP: {e}")
        return None

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) / iters

    print(f"{name:46} {elapsed * 1000:10.4f} ms")
    return elapsed


def main():
    print("AtomNET benchmark")
    print("version:", atomnet.version())

    # ============================================================
    # Данные
    # ============================================================

    n = 1 << 20
    small_n = 1 << 16

    x = np.random.randn(n).astype(np.float32)
    y = np.random.randn(n).astype(np.float32)

    xs = np.random.randn(small_n).astype(np.float32) * 0.02

    # ============================================================
    # AtomMATH: SIMD / базовые операции
    # ============================================================

    print("\n=== AtomMATH: SIMD / base ===")

    bench(
        "numpy_norm",
        lambda: np.linalg.norm(x),
    )

    if hasattr(atomnet, "norm_np"):
        bench(
            "atomnet_norm_np",
            lambda: atomnet.norm_np(x),
        )
    else:
        bench(
            "atomnet_norm_list",
            lambda: atomnet.norm(x.tolist()),
        )

    bench(
        "numpy_dot",
        lambda: float(np.dot(x, y)),
    )

    if hasattr(atomnet, "dot_np"):
        bench(
            "atomnet_dot_np",
            lambda: atomnet.dot_np(x, y),
        )
    else:
        bench(
            "atomnet_dot_list",
            lambda: atomnet.dot(x.tolist(), y.tolist()),
        )

    # ============================================================
    # AtomMATH: Enigma sphere
    # ============================================================

    print("\n=== AtomMATH: Enigma sphere ===")

    bench(
        "numpy_sphere_normalize",
        lambda: xs / (np.linalg.norm(xs) + 1e-12),
    )

    if hasattr(atomnet, "spherical_normalize_np"):
        bench(
            "atomnet_sphere_normalize_np",
            lambda: atomnet.spherical_normalize_np(xs),
        )

    if hasattr(atomnet, "spherical_normalize_inplace_np"):
        tmp = xs.copy()
        bench(
            "atomnet_sphere_normalize_inplace_np",
            lambda: atomnet.spherical_normalize_inplace_np(tmp),
        )

    # ============================================================
    # AtomMATH: Micron int8
    # ============================================================

    print("\n=== AtomMATH: Micron int8 ===")

    if hasattr(atomnet, "quantize_int8_np"):
        bench(
            "atomnet_quantize_int8_np",
            lambda: atomnet.quantize_int8_np(xs),
        )

        q, mu, sc = atomnet.quantize_int8_np(xs)

        bench(
            "atomnet_dequantize_int8_np",
            lambda: atomnet.dequantize_int8_np(q, mu, sc),
        )

        x_hat = atomnet.dequantize_int8_np(q, mu, sc)
        err = float(np.abs(xs - x_hat).max())

        print(f"micron int8 max error: {err:.8f}")
        print(f"micron int8 bound:     {sc / 2.0:.8f}")
        print(f"micron int8 pass:      {err <= sc / 2.0 + 1e-6}")
    else:
        print("quantize_int8_np not found")

    # ============================================================
    # AtomMATH: AtomCodec multi-level
    # ============================================================

    print("\n=== AtomMATH: AtomCodec multi-level ===")

    if hasattr(atomnet, "compress_atom_np"):
        configs = [
            [8],
            [8, 4],
            [8, 4, 2],
            [4],
            [4, 2],
        ]

        for bits in configs:
            compressed = atomnet.compress_atom_np(xs, bits)

            bench(
                f"atom_codec_compress_{bits}",
                lambda: atomnet.compress_atom_np(xs, bits),
            )

            err = float(compressed.max_error_np(xs))
            bpp = float(compressed.bytes_per_param())

            print(f"  bits={bits}, error={err:.8f}, bytes/param={bpp:.4f}")

        compressed_84 = atomnet.compress_atom_np(xs, [8, 4])

        bench(
            "atom_codec_decompress_[8,4]",
            lambda: compressed_84.decompress_np(),
        )
    else:
        print("compress_atom_np not found")

    # ============================================================
    # AtomMATH: Qairo linear layer
    # ============================================================

    print("\n=== AtomMATH: Qairo layer ===")

    out_features = 256
    in_features = 1024

    W = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    b = np.random.randn(out_features).astype(np.float32) * 0.02
    xv = np.random.randn(in_features).astype(np.float32) * 0.02

    bench(
        "numpy_linear_Wx_b",
        lambda: W @ xv + b,
    )

    if hasattr(atomnet, "PyQairoLayer"):
        layer = atomnet.PyQairoLayer(W.tolist(), b.tolist())

        if hasattr(layer, "forward_np"):
            bench(
                "atomnet_qairo_forward_np",
                lambda: layer.forward_np(xv),
            )
        else:
            bench(
                "atomnet_qairo_forward_list",
                lambda: layer.forward(xv.tolist()),
            )

        y_np = W @ xv + b
        y_q = layer.forward_np(xv) if hasattr(layer, "forward_np") else np.array(layer.forward(xv.tolist()), dtype=np.float32)

        qairo_err = float(np.abs(y_np - y_q).max())
        print(f"qairo max error: {qairo_err:.8f}")

    # ============================================================
    # AtomOPTIM: Rineta
    # ============================================================

    print("\n=== AtomOPTIM: Rineta ===")

    optim_n = 1 << 18

    w = np.random.randn(optim_n).astype(np.float32) * 0.02
    g = np.random.randn(optim_n).astype(np.float32) * 0.02
    zeros = np.zeros(optim_n, dtype=np.float32)

    if hasattr(atomnet, "PyRin"):
        if hasattr(atomnet.PyRin, "from_numpy"):
            rin = atomnet.PyRin.from_numpy(zeros)
        else:
            rin = atomnet.PyRin(zeros.tolist())

        if hasattr(rin, "apply_into_inplace_np"):
            bench(
                "atomoptim_rin_apply_into_inplace_np",
                lambda: rin.apply_into_inplace_np(w, g, 1e-3, 0.9),
            )
        elif hasattr(rin, "apply_into_np"):
            bench(
                "atomoptim_rin_apply_into_np",
                lambda: rin.apply_into_np(w, g, 1e-3, 0.9),
            )

        print("rin coherence:", rin.coherence())
        print("rin boost:", rin.boost())
        print("rin alive:", rin.is_alive())

    # ============================================================
    # AtomOPTIM: Keralin
    # ============================================================

    print("\n=== AtomOPTIM: Keralin ===")

    if hasattr(atomnet, "PyAtomWeight"):
        atom = atomnet.PyAtomWeight(w)

        bench(
            "atomoptim_keralin_step_np",
            lambda: atom.step_np(g),
        )

        print("atom nucleus:", atom.nucleus())
        print("atom radius:", atom.radius())
        print("atom alive:", atom.is_alive())

    # ============================================================
    # AtomOPTIM: XorCursor
    # ============================================================

    print("\n=== AtomOPTIM: XorCursor ===")

    if hasattr(atomnet, "PyXorCursor"):
        cursor = atomnet.PyXorCursor()

        bench(
            "atomoptim_xor_cursor_update_np",
            lambda: cursor.update_np(xs),
        )

        cursor.update_np(xs)
        cursor.adapt_compression(3)

        print("cursor alive:", cursor.is_alive())
        print("cursor stable:", cursor.is_stable())
        print("cursor suggested bits:", list(cursor.suggested_bits()))

    print("\nBenchmark finished.")


if __name__ == "__main__":
    main()
