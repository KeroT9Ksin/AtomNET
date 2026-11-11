import numpy as np
import atomnet


def test_atom_life():
    print("=== PyAtomLife ===")

    life = atomnet.PyAtomLife()

    x = np.random.randn(4096).astype(np.float32) * 0.02
    energy = float(np.linalg.norm(x))

    life.observe_np(x, energy, 0.65)

    print("state:", life.state)
    print("gate:", life.gate())
    print("energy:", life.energy)
    print("coherence:", life.coherence)
    print("stability:", life.stability)
    print("age:", life.age)
    print("alive:", life.is_active())

    assert life.state in [
        "dead",
        "cold",
        "born",
        "alive",
        "stable",
        "decaying",
    ]


def test_rin_vec():
    print("\n=== PyRinVec ===")

    n = 4096

    x = np.random.randn(n).astype(np.float32) * 0.02
    y = np.random.randn(n).astype(np.float32) * 0.02

    vx = atomnet.PyRinVec.from_numpy(x)
    vy = atomnet.PyRinVec.from_numpy(y)

    print("vx norm:", vx.norm())
    print("vy norm:", vy.norm())

    dot = vx.dot(vy)
    print("dot:", dot)

    coherence = vx.coherence_with(vy)
    print("coherence:", coherence)

    gated_dot = vx.gated_dot(vy)
    print("gated_dot:", gated_dot)

    print("vx life state:", vx.life_state)
    print("vx life gate:", vx.gate())
    print("vx alive:", vx.alive())

    old_norm = vx.normalize()
    print("old norm:", old_norm)
    print("new norm:", vx.norm())

    assert abs(vx.norm() - 1.0) < 1e-3


def test_qairo_living_layer():
    print("\n=== PyQairoLivingLayer ===")

    out_features = 64
    in_features = 256

    W = np.random.randn(out_features, in_features).astype(np.float32) * 0.02
    b = np.zeros(out_features, dtype=np.float32)

    layer = atomnet.PyQairoLivingLayer(W.tolist(), b.tolist())

    print("dim:", layer.dim)
    print("rows:", layer.rows)

    x = np.random.randn(in_features).astype(np.float32) * 0.02

    # Живый forward: обновляет LifeTime строк.
    y_living = layer.forward_np(x)

    # Чистый линейный forward: не обновляет LifeTime.
    y_linear = layer.forward_linear_np(x)

    # Детерминированный живой forward: использует текущие гейты,
    # но не обновляет состояние.
    y_frozen = layer.forward_gated_frozen_np(x)

    print("y_living shape:", y_living.shape)
    print("y_linear shape:", y_linear.shape)
    print("y_frozen shape:", y_frozen.shape)

    print("row 0 state:", layer.row_state(0))
    print("row 0 gate:", layer.row_gate(0))
    print("row 0 alive:", layer.row_alive(0))
    print("row 0 energy:", layer.row_energy(0))
    print("row 0 coherence:", layer.row_coherence(0))
    print("row 0 stability:", layer.row_stability(0))

    assert y_living.shape == (out_features,)
    assert y_linear.shape == (out_features,)
    assert y_frozen.shape == (out_features,)


if __name__ == "__main__":
    print("AtomNET version:", atomnet.version())

    test_atom_life()
    test_rin_vec()
    test_qairo_living_layer()

    print("\nLIFE BINDINGS TEST FINISHED")