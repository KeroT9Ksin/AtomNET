"""Бенчмарк bitwise-опций AtomNET vs NumPy (PyTorch CPU).

Compare:
- xor_fold / xor_eq (bitwise fold) vs NumPy == / np.sum
- pack_signs / packed_popcount vs NumPy signs
- dot / norm / mse / mean_var vs numpy
- matmul vs numpy @ (для контекста)

Запуск:
    ./venv/bin/python tests/bench_bitwise.py
"""
import time

import numpy as np

import atomnet.math as M


def bench(fn, reps=200):
    """Покрылиговый median таймер."""
    best = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best.append(time.perf_counter() - t0)
    best.sort()
    return best[len(best) // 2] * 1e6  # microseconds


for n in [64, 1024, 65536, 1048576, 4194304]:
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    # dot
    t_np = bench(lambda: np.dot(a, b))
    t_at = bench(lambda: M.dot_np(a, b))
    print(f"n={n:>7}: dot        numpy {t_np:8.2f} us | atom {t_at:8.2f} us | x{t_np/max(t_at,1e-9):.2f}")

    # xor_fold (bitwise fold) — NumPy has no direct fold; use np.sum((x>0).astype)
    # Аналог: проверка знаков вручную
    def np_sign_count(x):
        return int(np.sum(x > 0))
    t_np2 = bench(lambda: np_sign_count(a))
    t_at2 = bench(lambda: M.xor_fold_np(a))
    print(f"n={n:>7}: xor_fold   numpy(sign count) {t_np2:8.2f} us | atom {t_at2:8.2f} us | x{t_np2/max(t_at2,1e-9):.2f}")

    # pack_signs
    def np_pack(x):
        return (x > 0).astype(np.uint8)
    t_np3 = bench(lambda: np_pack(a))
    t_at3 = bench(lambda: M.pack_signs_np(a))
    print(f"n={n:>7}: pack_signs numpy {t_np3:8.2f} us | atom {t_at3:8.2f} us | x{t_np3/max(t_at3,1e-9):.2f}")

    # packed_popcount — NumPy: bit_count via builtin
    packed = M.pack_signs_np(a.nbytes and a)
    def np_popcount(x):
        # Суммирование бит вручную
        pp = M.pack_signs_np(x)
        return sum(bin(int(v)).count("1") for v in pp)
    t_np4 = bench(lambda: np_popcount(a))
    t_at4 = bench(lambda: M.packed_popcount_np(M.pack_signs_np(a)))
    print(f"n={n:>7}: popcount   numpy {t_np4:8.2f} us | atom {t_at4:8.2f} us | x{t_np4/max(t_at4,1e-9):.2f}")

    # mse
    t_np5 = bench(lambda: np.mean((a - b) ** 2))
    t_at5 = bench(lambda: M.mse_np(a, b))
    print(f"n={n:>7}: mse        numpy {t_np5:8.2f} us | atom {t_at5:8.2f} us | x{t_np5/max(t_at5,1e-9):.2f}")
    print("")

# GEMM — для контекста (NumPy использует BLAS, мы честно сравниваем)
for sz in [64, 256, 1024]:
    a = np.random.randn(sz, sz).astype(np.float32)
    b = np.random.randn(sz, sz).astype(np.float32)
    t_np = bench(lambda: a @ b, reps=20)
    t_at = bench(lambda: M.matmul_np(a, b), reps=20)
    print(f"GEMM {sz}x{sz}: numpy {t_np:9.1f} us | atom {t_at:9.1f} us | x{t_np/max(t_at,1e-9):.2f}")

# xor_eq — bitwise equality
print("")
a = np.random.randn(65536).astype(np.float32)
b = a.copy()
t_np = bench(lambda: bool(np.amin(a == b)))
t_at = bench(lambda: M.xor_eq_np(a, b))
print(f"xor_eq n=65536: numpy {t_np:.1f} us | atom {t_at:.1f} us | x{t_np/max(t_at,1e-9):.2f}")
