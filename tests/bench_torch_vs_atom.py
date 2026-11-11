# tests/bench_torch_vs_atom.py
# AtomNET vs PyTorch (CPU, 1 поток): честные краты по компонентам обучения.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time

import numpy as np
import atomnet

import torch

torch.set_num_threads(1)
torch.manual_seed(0)

def ms(fn, reps=20):
    for _ in range(3):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000

n = 1 << 22
a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)
a_t = torch.from_numpy(a_np)
b_t = torch.from_numpy(b_np)
out = np.empty_like(a_np)
out_t = torch.empty_like(a_t)

print("=== AtomNET vs PyTorch, CPU 1 поток ===\n")
rows = []

def row(name, fn_at, fn_t):
    ta = ms(fn_at)
    tt = ms(fn_t)
    rows.append((name, ta, tt, tt / max(ta, 1e-9)))
    print(f"{name:24s} atom {ta:9.3f} ms | torch {tt:9.3f} ms | x{tt/max(ta,1e-9):8.2f}")

# слитые операции (numpy-эквиваленты torch делают за несколько проходов)
row("mse (4M)", lambda: atomnet.mse_np(a_np, b_np),
    lambda: float(torch.mean((a_t - b_t) ** 2)))
row("mean+var (4M)", lambda: atomnet.mean_var_np(a_np),
    lambda: (float(a_t.mean()), float(a_t.var())))
row("affine_into (4M)", lambda: atomnet.affine_into_np(a_np, 1.5, -0.7, out),
    lambda: (a_t * 1.5 - 0.7))
row("dot (4M)", lambda: atomnet.dot_np(a_np, b_np),
    lambda: float(torch.dot(a_t, b_t)))
row("xor_signature (4M)", lambda: atomnet.xor_signature_np(a_np),
    lambda: float(torch.count_nonzero(a_t > 0)))
row("quantize_int8 (4M)", lambda: atomnet.quantize_int8_np(a_np),
    lambda: torch.quantize_per_tensor(a_t, 1.0 / 127.0, 0, torch.qint8))

# GEMV / батч: FlashInt16 против torch
rows_l, cols_l = 4096, 2048
W = (np.random.randn(rows_l, cols_l) * 0.02).astype(np.float32)
bias = np.zeros(rows_l, dtype=np.float32)
x = (np.random.randn(cols_l) * 0.02).astype(np.float32)
W_t = torch.from_numpy(W)
x_t = torch.from_numpy(x)
fi = atomnet.PyFlashInt16.from_numpy(W, bias)
row("gemv int16 4096x2048", lambda: fi.gemv_np(x), lambda: W_t @ x_t)
for B in (8, 16):
    X = (np.random.randn(B, cols_l) * 0.02).astype(np.float32)
    X_t = torch.from_numpy(X)
    row(f"gemv batch int16 B={B}", lambda: fi.gemv_batch_np(X), lambda: X_t @ W_t.t())

geo = 1.0
for _, ta, tt, ratio in rows:
    if ratio > 1:
        geo *= ratio
print(f"\nСреднегеометрический выигрыш по строкам >1: x{geo ** (1.0 / max(sum(1 for r in rows if r[3] > 1), 1)):.1f}")
fast = sum(1 for r in rows if r[3] > 1)
print(f"Быстрее PyTorch: {fast}/{len(rows)} операций")
