"""Паритет тестов нашей атомной сферической математики (atommath)."""
import numpy as np

import atomnet as A


def _norm(v):
    return float(np.linalg.norm(v))


def test_normalize():
    u = A.sphere_normalize([3.0, 4.0])
    assert abs(_norm(np.array(u)) - 1.0) < 1e-6


def test_tangent_project():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(8).astype(np.float32)
    u = (u / np.linalg.norm(u)).tolist()
    g = rng.standard_normal(8).astype(np.float32).tolist()
    tg = np.array(A.atommath_tangent_project(g, u))
    assert abs(float(np.dot(tg, np.array(u)))) < 1e-5


def test_geodesic_step_norm_preserving():
    rng = np.random.default_rng(1)
    u = rng.standard_normal(16).astype(np.float32)
    u = (u / np.linalg.norm(u)).tolist()
    g = rng.standard_normal(16).astype(np.float32).tolist()
    gt = A.atommath_geodesic_step(u, g, 0.05)
    assert abs(_norm(np.array(gt)) - 1.0) < 2e-6
    # numpy-референс поворота
    uv = np.array(u); gv = np.array(g)
    v = gv - np.dot(gv, uv) * uv
    nv = np.linalg.norm(v)
    th = 0.05 * nv
    expect = np.cos(th) * uv + np.sin(th) * (v / nv)
    assert np.allclose(np.array(gt), expect, atol=3e-5)


def test_householder():
    rng = np.random.default_rng(2)
    u = rng.standard_normal(10).astype(np.float32)
    u = (u / np.linalg.norm(u)).tolist()
    v = rng.standard_normal(10).astype(np.float32)
    v = (v / np.linalg.norm(v)).tolist()
    x = rng.standard_normal(10).astype(np.float32).tolist()
    out = A.atommath_householder_reflect(x, u, v)
    # numpy-ref
    w = np.array(u) - np.array(v)
    denom = float(np.dot(w, w))
    expect = np.array(x) - 2.0 * float(np.dot(w, x)) / denom * w
    assert np.allclose(np.array(out), expect, atol=1e-5)
    assert abs(_norm(np.array(out)) - _norm(np.array(x))) < 1e-5
    hu = A.atommath_householder_reflect(u, u, v)
    assert np.allclose(np.array(hu), np.array(v), atol=1e-5)


def test_decompose_reconstruct_identity():
    w = np.random.default_rng(3).standard_normal(24).astype(np.float32)
    n, r, d = A.atommath_decompose(w.tolist())
    w2 = A.atommath_reconstruct(n, r, d)
    assert np.allclose(np.array(w2), w, atol=1e-5)


def test_shell_quantize():
    u = [1.0, 0.0, 0.0]
    idx, d = A.atommath_shell_quantize(u, 16, 16)
    assert d > 0.9


if __name__ == "__main__":
    for name, fn in sorted((k, v) for k, v in vars().items() if k.startswith("test_")):
        fn()
        print(f"OK {name}")


def test_jit_bindings_parity():
    rng = np.random.default_rng(11)
    for n in [1, 3, 8, 9, 64, 100]:
        a = rng.standard_normal(n).astype(np.float32).tolist()
        b = rng.standard_normal(n).astype(np.float32).tolist()
        g = np.array(a) / np.linalg.norm(a)
        dj = A.atommath_dot_jit(a, b)
        expect = float(np.dot(np.array(a), np.array(b)))
        tol = 3e-5 * max(abs(expect), 1.0)
        assert abs(dj - expect) < tol, (n, dj, expect)

    u = rng.standard_normal(16).astype(np.float32)
    u = (u / np.linalg.norm(u)).tolist()
    g = rng.standard_normal(16).astype(np.float32).tolist()
    pj = np.array(A.atommath_tangent_project_jit(g, u))
    # reference
    d = float(np.dot(np.array(g), np.array(u)))
    expect = np.array(g) - d * np.array(u)
    assert np.allclose(pj, expect, atol=3e-5)

    gj = np.array(A.atommath_geodesic_step_jit(u, g, 0.05))
    assert abs(np.linalg.norm(gj) - 1.0) < 1e-5

    x = rng.standard_normal(9).astype(np.float32).tolist()
    w = A.atommath_householder_jit(u, u, u)
    assert np.allclose(np.array(w), np.array(u), atol=1e-5)
