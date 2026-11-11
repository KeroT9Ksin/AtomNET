#!/usr/bin/env python3
"""Уровень 2: MLP y=sin(x) — чисто AtomNET (без PyTorch).

Линейная алгебра (все square-GEMM/GEMV) — Rust-ядра AtomNET
(npu_matmul_np / npu_matvec_np), обучение — батчевый градиент.
Элементные активации и шум — numpy (не линейная алгебра).
Финал — int4-инференс через PyQairoLivingLayer.

Запуск: venv/bin/python examples/train_mlp_atomnet.py
"""
import time

import numpy as np

import atomnet
import atomnet.array as ara

SEED = 42
N = 400
H = 32
EPOCHS = 400
LR = 0.05

rng = np.random.default_rng(0)
X = np.linspace(-np.pi, np.pi, N, dtype=np.float32).reshape(-1, 1)
Y = (np.sin(X) + 0.05 * rng.standard_normal((N, 1))).astype(np.float32)

# Init (numpy, для зеркала) + AtomNET-mirror
W1 = (rng.standard_normal((H, 1)) / 1.0).astype(np.float32)
b1 = np.zeros(H, dtype=np.float32)
W2 = (rng.standard_normal((1, H)) / np.sqrt(H)).astype(np.float32)
b2 = np.zeros(1, dtype=np.float32)

QL = atomnet.life.PyQairoLivingLayer


def forward(Xm):
    Hm = np.tanh(atomnet.npu_matmul_np(Xm, W1.T) + b1)
    Yp = atomnet.npu_matmul_np(Hm, W2.T) + b2
    return Hm, Yp


# numpy-зеркало для честного сравнения
W1n, b1n, W2n, b2n = W1.copy(), b1.copy(), W2.copy(), b2.copy()

print(f"Сеть: 1 -> {H} -> 1 | N={N} | epochs={EPOCHS} | lr={LR}")
t0 = time.perf_counter()
for ep in range(EPOCHS):
    Hm, Yp = forward(X)
    loss = float(np.mean((Yp - Y) ** 2))
    dY = (2.0 * (Yp - Y) / N).astype(np.float32)
    dW2 = atomnet.npu_matmul_np(dY.T, Hm)
    db2 = np.array(dY.sum(axis=0).reshape(1), dtype=np.float32)
    dH = atomnet.npu_matmul_np(dY, W2)
    dPre = (dH * (1.0 - Hm ** 2)).astype(np.float32)
    dW1 = atomnet.npu_matmul_np(dPre.T, X)
    db1 = np.array(dPre.sum(axis=0), dtype=np.float32)
    W2 -= LR * dW2; b2 -= LR * db2
    W1 -= LR * dW1; b1 -= LR * db1
    # зеркало numpy (то же init, те же lr)
    Hn = np.tanh(X @ W1n.T + b1n)
    Yn = Hn @ W2n.T + b2n
    loss_n = float(np.mean((Yn - Y) ** 2))
    dYn = (2.0 * (Yn - Y) / N).astype(np.float32)
    dW2n = dYn.T @ Hn; db2n = dYn.sum(axis=0)
    dHn = dYn @ W2n; dPren = dHn * (1.0 - Hn ** 2)
    dW1n = dPren.T @ X; db1n = dPren.sum(axis=0)
    W2n -= LR * dW2n; b2n -= LR * db2n
    W1n -= LR * dW1n; b1n -= LR * db1n
    if ep % 100 == 0 or ep == EPOCHS - 1:
        print(f"{ep:4d} | Rust loss {loss:10.6f} | numpy loss {loss_n:10.6f}")
print(f"Итого: {time.perf_counter()-t0:.1f} с")

# int4 через QairoLivingLayer (квантование в Rust, инференс в Rust)
L1 = QL(W1.tolist(), b1.tolist())
L2 = QL(W2.tolist(), b2.tolist())
L1.quantize_to_int4(); L2.quantize_to_int4()
Yi = np.empty((N, 1), dtype=np.float32)
for i in range(N):
    Yi[i, 0] = float(np.array(L2.forward_linear_np(np.tanh(L1.forward_linear_np(X[i]))))[0])
print(f"loss int4: {float(np.mean((Yi - Y) ** 2)):.6f}")
print("✅ Готово: MLP обучен на AtomNET.")
