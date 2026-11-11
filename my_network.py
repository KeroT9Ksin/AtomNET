#!/usr/bin/env python3
"""
Моя нейросеть на AtomNET (v0.2.0).

Двухслойный перцептрон:
  Eingang  -> Linear(H) -> tanh -> Linear(out) -> выход

Обучается на простой регрессии y = sin(x) + шум,
сравнивается с NumPy-зеркалом (тот же init, тот же lr).
Все матричные операции forward/backward — в Rust (PyQairoLivingLayer).

Запуск:
    /home/keiron/atomnet/venv/bin/python3 my_network.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import atomnet

# ---------- Конфигурация ----------
SEED = 42
np.random.seed(SEED)

N_SAMPLES = 1000
D_IN = 1          # один вход x
HIDDEN = 32       # скрытый слой
D_OUT = 1         # один выход y
EPOCHS = 3000
LR = 0.05

# ---------- Данные: y = sin(x) + шум ----------
X = np.linspace(-np.pi, np.pi, N_SAMPLES).astype(np.float32).reshape(-1, 1)
Y = (np.sin(X) + 0.05 * np.random.randn(N_SAMPLES, 1)).astype(np.float32)

# ---------- Инициализация весов (одинаковая для Rust и NumPy) ----------
W1 = (np.random.randn(HIDDEN, D_IN) / np.sqrt(D_IN)).astype(np.float32)
b1 = np.zeros(HIDDEN, dtype=np.float32)
W2 = (np.random.randn(D_OUT, HIDDEN) / np.sqrt(HIDDEN)).astype(np.float32)
b2 = np.zeros(D_OUT, dtype=np.float32)

# ---------- AtomNET слои ----------
L1 = atomnet.life.PyQairoLivingLayer(W1.tolist(), b1.tolist())
L2 = atomnet.life.PyQairoLivingLayer(W2.tolist(), b2.tolist())

# Зеркало в NumPy для контроля сходимости
W1n = W1.copy(); b1n = b1.copy()
W2n = W2.copy(); b2n = b2.copy()

print(f"Сеть: {D_IN} -> {HIDDEN} -> {D_OUT}")
print(f"Образцов: {N_SAMPLES}, эпох: {EPOCHS}, lr: {LR}")
print(f"{'эпоха':>6} | {'loss_rust':>12} | {'loss_numpy':>12} | {'разница':>10}")
print("-" * 50)

loss0 = None
for ep in range(EPOCHS):
    # ---------- Forward (Rust) ----------
    Hq = np.empty((N_SAMPLES, HIDDEN), dtype=np.float32)
    Yq = np.empty((N_SAMPLES, D_OUT), dtype=np.float32)
    for i in range(N_SAMPLES):
        h = np.tanh(L1.forward_linear_np(X[i]))
        Hq[i] = h
        Yq[i] = L2.forward_linear_np(h)
    loss_q = float(np.mean((Yq - Y) ** 2))
    if loss0 is None:
        loss0 = loss_q

    # ---------- Backward (Rust) ----------
    dY = (2.0 * (Yq - Y) / N_SAMPLES).astype(np.float32)
    dW2 = dY.T @ Hq
    db2 = dY.sum(axis=0)

    dH = np.empty((N_SAMPLES, HIDDEN), dtype=np.float32)
    for i in range(N_SAMPLES):
        dH[i] = L2.backward_input_np(dY[i])          # W2^T @ dY
    dPre = dH * (1.0 - Hq ** 2)
    dW1 = dPre.T @ X
    db1 = dPre.sum(axis=0)

    L2.sgd_step_np(dW2, db2, LR)
    L1.sgd_step_np(dW1, db1, LR)

    # ---------- NumPyзеркало ----------
    Hn = np.tanh(X @ W1n.T + b1n)
    Yn = Hn @ W2n.T + b2n
    loss_n = float(np.mean((Yn - Y) ** 2))

    dYn = (2.0 * (Yn - Y) / N_SAMPLES).astype(np.float32)
    dW2n = dYn.T @ Hn
    db2n = dYn.sum(axis=0)
    dHn = dYn @ W2n
    dPren = dHn * (1.0 - Hn ** 2)
    dW1n = dPren.T @ X
    db1n = dPren.sum(axis=0)
    W2n -= LR * dW2n; b2n -= LR * db2n
    W1n -= LR * dW1n; b1n -= LR * db1n

    if ep % 300 == 0 or ep == EPOCHS - 1:
        print(f"{ep:6d} | {loss_q:12.6f} | {loss_n:12.6f} | {abs(loss_q - loss_n):10.2e}")

# ---------- int4-инференс (сжатие весов) ----------
L1.quantize_to_int4()
L2.quantize_to_int4()
Yi = np.empty((N_SAMPLES, D_OUT), dtype=np.float32)
for i in range(N_SAMPLES):
    Yi[i] = L2.forward_linear_np(np.tanh(L1.forward_linear_np(X[i])))
loss_i4 = float(np.mean((Yi - Y) ** 2))

print("-" * 50)
print(f"старт loss:    {loss0:.6f}")
print(f"loss Rust f32: {loss_q:.6f}")
print(f"loss NumPy:    {loss_n:.6f}")
print(f"loss int4:     {loss_i4:.6f}")
print(f"ускорение обучения (loss Rust < loss0): {'да' if loss_q < 0.2 * loss0 else 'нет'}")

# ---------- Проверка на нескольких точках ----------
print("\nПроверка на контрольных точках:")
for xv in [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]:
    x_arr = np.array([[xv]], dtype=np.float32)
    h = np.tanh(L1.forward_linear_np(x_arr[0]))
    y_pred = float(L2.forward_linear_np(h)[0])
    y_true = float(np.sin(xv))
    print(f"  x={xv:7.3f}  pred={y_pred:8.4f}  true={y_true:8.4f}  err={abs(y_pred-y_true):.4f}")

print("\n✅ Готово: нейросеть на AtomNET обучена и запущена.")
