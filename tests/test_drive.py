# tests/test_drive.py
import numpy as np
import atomnet


def test_version():
    print("AtomNET version:", atomnet.version())


def test_sphere():
    print("\n=== TEST: Enigma sphere ===")

    x = np.random.randn(4096).astype(np.float32)
    x_list = x.tolist()

    y_list = atomnet.spherical_normalize(x_list)
    y = np.array(y_list, dtype=np.float32)

    norm_y = float(np.linalg.norm(y))
    print("norm after sphere:", norm_y)

    assert abs(norm_y - 1.0) < 1e-3

    alive = atomnet.enigma_is_alive(y_list)
    sig = atomnet.enigma_signature(x_list)

    print("enigma alive:", alive)
    print("enigma signature bits:", bin(sig).count("1"))


def test_micron():
    print("\n=== TEST: Micron int8 ===")

    x = np.random.randn(4096).astype(np.float32)
    x_list = x.tolist()

    q, mu, sc = atomnet.quantize_int8(x_list)
    x_hat_list = atomnet.dequantize_int8(q, mu, sc)
    x_hat = np.array(x_hat_list, dtype=np.float32)

    err = float(np.abs(x - x_hat).max())
    bound = sc / 2.0 + 1e-6

    print("mu:", mu)
    print("scale:", sc)
    print("max error:", err)
    print("theoretical bound:", bound)

    assert err <= bound

    ok, hamming = atomnet.verify_integrity(x_list, q, mu, sc)
    print("integrity ok:", ok)
    print("hamming:", hamming)

    ok_sphere, hamming_sphere, loss_pct = atomnet.verify_with_micron(
        x_list,
        q,
        mu,
        sc,
    )

    print("sphere+micron ok:", ok_sphere)
    print("sphere+micron hamming:", hamming_sphere)
    print("loss pct:", loss_pct)


def test_qairo_equivalence():
    print("\n=== TEST: Qairo vs NumPy Linear ===")

    out_features = 64
    in_features = 256

    W = np.random.randn(out_features, in_features).astype(np.float32)
    b = np.random.randn(out_features).astype(np.float32)
    x = np.random.randn(in_features).astype(np.float32)

    # NumPy linear
    y_np = W @ x + b

    # Qairo
    layer = atomnet.PyQairoLayer(W.tolist(), b.tolist())
    y_q_list = layer.forward(x.tolist())
    y_q = np.array(y_q_list, dtype=np.float32)

    diff = np.abs(y_np - y_q)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    print("shape:", layer.shape())
    print("max diff:", max_diff)
    print("mean diff:", mean_diff)

    assert np.allclose(y_np, y_q, rtol=1e-3, atol=1e-3)

    # Check one row reconstruction
    row_idx = 0
    row_original = W[row_idx]
    row_error = layer.row_error(row_idx, row_original.tolist())

    print("row reconstruction error:", row_error)

    assert row_error < 1e-3


def test_qairo_gate():
    print("\n=== TEST: Qairo gate ===")

    out_features = 32
    in_features = 128

    W = np.random.randn(out_features, in_features).astype(np.float32)
    b = np.random.randn(out_features).astype(np.float32)
    x = np.random.randn(in_features).astype(np.float32)

    layer = atomnet.PyQairoLayer(W.tolist(), b.tolist())

    y_linear = np.array(layer.forward(x.tolist()), dtype=np.float32)
    y_gate = np.array(layer.forward_gate(x.tolist()), dtype=np.float32)

    print("linear output norm:", float(np.linalg.norm(y_linear)))
    print("gate output norm:", float(np.linalg.norm(y_gate)))

    assert y_linear.shape == y_gate.shape


def test_rineta():
    print("\n=== TEST: Rineta ===")

    n = 1024
    w = np.random.randn(n).astype(np.float32)
    g = np.random.randn(n).astype(np.float32)

    rin = atomnet.PyRin(np.zeros(n, dtype=np.float32).tolist())

    print("before alive:", rin.is_alive())
    print("cold start hamming:", rin.cold_start_hamming())

    rin.warmup(1e-3)

    print("after warmup alive:", rin.is_alive())
    print("after warmup hamming:", rin.cold_start_hamming())

    w_new, coherence = rin.apply_into(
        w.tolist(),
        g.tolist(),
        1e-3,
        0.9,
    )

    print("coherence:", coherence)
    print("boost:", rin.boost())

    assert len(w_new) == n


if __name__ == "__main__":
    test_version()
    test_sphere()
    test_micron()
    test_qairo_equivalence()
    test_qairo_gate()
    test_rineta()

    print("\nALL TESTS PASSED")