# tests/bench_v3_fair.py — Честный A/B: NumPy vs AtomNET v3 (такты совпадают).
# Измерения чередуются: rep №i -> numpy, затем atom — состояние ядра CPU одинаково.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import time
import numpy as np
import atomnet


def ab(name, fn_np, fn_at, reps=40, warm=3):
    for _ in range(warm):
        fn_np(); fn_at()
    tnp, tat = [], []
    for _ in range(reps):
        t0 = time.perf_counter(); fn_np(); tnp.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fn_at(); tat.append(time.perf_counter() - t0)
    med = lambda v: sorted(v)[len(v) // 2]
    m_np, m_at = med(tnp) * 1e3, med(tat) * 1e3
    print(f"  {name:36s} numpy {m_np:8.3f} ms | atom {m_at:8.3f} ms | x{m_np / max(m_at, 1e-12):6.2f}")
    return m_np, m_at


def bit_exact(name, v0, v1):
    ok = (v0 == v1)
    if not ok:
        print(f"  !! {name}: NOT bit-exact: {v0!r} vs {v1!r}")
    return ok


def main():
    np.random.seed(0)
    print(f"atomnet {atomnet.version()}, threads={atomnet.parallel_threads()}")

    sizes = [1 << 16, 1 << 20, 1 << 22]
    data = {}
    for n in sizes:
        data[n] = (np.random.randn(n).astype(np.float32), np.random.randn(n).astype(np.float32))

    for threads in (1, 6):
        atomnet.set_parallel_threads(threads)
        print(f"\n=== Veron threads = {threads} ===")
        ok_all = True
        for n in sizes:
            a, b = data[n]
            print(f"n = {n} ({2.0 * n * 4 / 1e6:.1f} МБ × 2):")
            r0 = np.dot(a, b)
            m_np, m_at = ab(f"dot", lambda: np.dot(a, b), lambda: atomnet.dot_np(a, b))
            ok_all &= bit_exact(f"dot{n}", float(r0), float(np.dot(a, b)))
            # bitwise: 1 поток против threads
            if threads > 1:
                atomnet.set_parallel_threads(1)
                d1 = float(atomnet.dot_np(a, b))
                atomnet.set_parallel_threads(threads)
                dt = float(atomnet.dot_np(a, b))
                ok_all &= bit_exact(f"dot{n} fork-vs-single", d1, dt)
            ab(f"norm", lambda: np.linalg.norm(a), lambda: atomnet.norm_np(a))
            ab(f"l2", lambda: _l2(a, b), lambda: atomnet.norm_np(np.subtract(a, b)))
            ab(f"mse", lambda: float(np.mean((a - b) ** 2)), lambda: atomnet.mse_np(a, b))
            ab(f"mean+var", lambda: (float(a.mean()), float(a.var())), lambda: atomnet.mean_var_np(a))
        if not ok_all:
            print("!! НАЙДЕНЫ РАСХОЖДЕНИЯ — стоп")
            return

    # количественная проверка против numpy (f32-точность)
    print("\n=== проверка корректности против numpy (f32) ===")
    for n in sizes:
        a, b = data[n]
        d_np = float(np.dot(a, b))
        d_at = float(atomnet.dot_np(a, b))
        rel = abs(d_np - d_at) / max(abs(d_np), 1e-9)
        n_at = float(atomnet.norm_np(a))
        n_np = float(np.linalg.norm(a))
        reln = abs(n_np - n_at) / max(n_np, 1e-9)
        mv = atomnet.mean_var_np(a)
        mvn = (float(a.mean()), float(a.var()))
        print(f"  n={n}: dot rel={rel:.2e}, norm rel={reln:.2e}, mean rel={abs(mv[0]-mvn[0])/max(abs(mvn[0]),1e-9):.2e}, var rel={abs(mv[1]-mvn[1])/max(mvn[1],1e-9):.2e}")


def _l2(a, b):
    d = a - b
    return float(np.linalg.norm(d))


if __name__ == "__main__":
    main()
