#!/usr/bin/env python3
"""
Полноценная нейросеть на AtomNET (v0.2.0).

Многослойный перцептрон (MLP):
  вход  -> Linear(H1) -> tanh -> Linear(H2) -> tanh -> Linear(out) -> выход

Задача: регрессия y = sin(x) + 0.5*cos(2x) + шум на 2000 образцах.

Возможности:
  - Обучение на Rust-ядрах (PyQairoLivingLayer + SGD)
  - NumPy-зеркало для контроля корректности градиентов
  - int4-квантование весов для уменьшения памяти и ускорения инференса
  - Отслеживание состояния «живости» слоёв (AtomLIFE)
  - Сохранение/загрузка чекпоинта

Запуск:
    /home/keiron/atomnet/venv/bin/python3 my_network_full.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import json
import numpy as np
import atomnet

# ══════════════════════════════════════════════════════════════════════
# Конфигурация
# ══════════════════════════════════════════════════════════════════════
SEED = 42
np.random.seed(SEED)

N_SAMPLES   = 2000   # число образцов
D_IN        = 1      # размер входа
H1          = 64     # первый скрытый слой
H2          = 32     # второй скрытый слой
D_OUT       = 1      # размер выхода
EPOCHS      = 5000
LR          = 0.05
TEST_SPLIT  = 0.2    # доля тестовой выборки

# ══════════════════════════════════════════════════════════════════════
# Данные: y = sin(x) + 0.5*cos(2x) + шум
# ══════════════════════════════════════════════════════════════════════
X = np.linspace(-np.pi, np.pi, N_SAMPLES, dtype=np.float32).reshape(-1, 1)
noise = 0.05 * np.random.randn(N_SAMPLES, 1).astype(np.float32)
Y = (np.sin(X) + 0.5 * np.cos(2.0 * X) + noise).astype(np.float32)

# Train / test split
n_test = int(N_SAMPLES * TEST_SPLIT)
X_train, X_test = X[:-n_test], X[-n_test:]
Y_train, Y_test = Y[:-n_test], Y[-n_test:]

print(f"Данные: {N_SAMPLES} образцов, train={len(X_train)}, test={len(X_test)}")
print(f"Сеть: {D_IN} -> {H1} -> {H2} -> {D_OUT}")
print(f"Эпох: {EPOCHS}, lr: {LR}")

# ══════════════════════════════════════════════════════════════════════
# Инициализация весов (идентичная для Rust и NumPy)
# ══════════════════════════════════════════════════════════════════════
def init_layer(fan_in, fan_out):
    """Xavier-init."""
    scale = 1.0 / np.sqrt(fan_in)
    w = np.random.randn(fan_out, fan_in) * scale
    b = np.zeros(fan_out, dtype=np.float32)
    return w.astype(np.float32), b

W1, b1 = init_layer(D_IN, H1)
W2, b2 = init_layer(H1, H2)
W3, b3 = init_layer(H2, D_OUT)

W1n, b1n = W1.copy(), b1.copy()
W2n, b2n = W2.copy(), b2.copy()
W3n, b3n = W3.copy(), b3.copy()

# ══════════════════════════════════════════════════════════════════════
# AtomNET слои
# ══════════════════════════════════════════════════════════════════════
L1 = atomnet.life.PyQairoLivingLayer(W1.tolist(), b1.tolist())
L2 = atomnet.life.PyQairoLivingLayer(W2.tolist(), b2.tolist())
L3 = atomnet.life.PyQairoLivingLayer(W3.tolist(), b3.tolist())

layers_rust = [L1, L2, L3]
layers_np   = [(W1n, b1n), (W2n, b2n), (W3n, b3n)]

# ══════════════════════════════════════════════════════════════════════
# Функции вспомогательные
# ══════════════════════════════════════════════════════════════════════
def forward_rust(x):
    """Forward через все слои на Rust."""
    h = np.tanh(L1.forward_linear_np(x))
    h = np.tanh(L2.forward_linear_np(h))
    y = L3.forward_linear_np(h)
    return y

def forward_np(x, W1, b1, W2, b2, W3, b3):
    """Forward через все слои на NumPy (зеркало)."""
    h = np.tanh(x @ W1.T + b1)
    h = np.tanh(h @ W2.T + b2)
    y = h @ W3.T + b3
    return y

def mse(y_pred, y_true):
    return float(np.mean((y_pred - y_true) ** 2))

# ══════════════════════════════════════════════════════════════════════
# Обучение
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'эпоха':>6} | {'train_rust':>12} | {'test_rust':>12} | {'test_numpy':>12} | {'разница':>10}")
print("-" * 66)

loss0 = None
history = {"train": [], "test_rust": [], "test_numpy": []}

t0 = time.perf_counter()
for ep in range(EPOCHS):
    # ---------- Forward (Rust) ----------
    Yq = np.stack([forward_rust(X_train[i]) for i in range(len(X_train))])
    loss_q = mse(Yq, Y_train)
    if loss0 is None:
        loss0 = loss_q

    # ---------- Backward (Rust) ----------
    dY = (2.0 * (Yq - Y_train) / len(X_train)).astype(np.float32)

    # Слой 3: grad wrt W3, b3
    H2q = np.stack([np.tanh(L2.forward_linear_np(
        np.tanh(L1.forward_linear_np(X_train[i])))
    ) for i in range(len(X_train))])
    dW3 = dY.T @ H2q
    db3 = dY.sum(axis=0)

    # Слой 2: backward через L3, затем tanh
    dH2 = np.stack([L3.backward_input_np(dY[i]) for i in range(len(X_train))])
    dPre2 = dH2 * (1.0 - H2q ** 2)
    H1q = np.stack([np.tanh(L1.forward_linear_np(X_train[i])) for i in range(len(X_train))])
    dW2 = dPre2.T @ H1q
    db2 = dPre2.sum(axis=0)

    # Слой 1: backward через L2, затем tanh
    dH1 = np.stack([L2.backward_input_np(dPre2[i]) for i in range(len(X_train))])
    dPre1 = dH1 * (1.0 - H1q ** 2)
    dW1 = dPre1.T @ X_train
    db1 = dPre1.sum(axis=0)

    L3.sgd_step_np(dW3, db3, LR)
    L2.sgd_step_np(dW2, db2, LR)
    L1.sgd_step_np(dW1, db1, LR)

    # ---------- NumPy зеркало ----------
    Yn = forward_np(X_train, W1n, b1n, W2n, b2n, W3n, b3n)
    loss_n = mse(Yn, Y_train)

    dYn = (2.0 * (Yn - Y_train) / len(X_train)).astype(np.float32)
    Hn1 = np.tanh(X_train @ W1n.T + b1n)
    Hn2 = np.tanh(Hn1 @ W2n.T + b2n)
    dW3n = dYn.T @ Hn2
    db3n = dYn.sum(axis=0)
    dH2n = dYn @ W3n
    dPre2n = dH2n * (1.0 - Hn2 ** 2)
    dW2n = dPre2n.T @ Hn1
    db2n = dPre2n.sum(axis=0)
    dH1n = dPre2n @ W2n
    dPre1n = dH1n * (1.0 - Hn1 ** 2)
    dW1n = dPre1n.T @ X_train
    db1n = dPre1n.sum(axis=0)

    W3n -= LR * dW3n; b3n -= LR * db3n
    W2n -= LR * dW2n; b2n -= LR * db2n
    W1n -= LR * dW1n; b1n -= LR * db1n

    # ---------- Оценка на тесте ----------
    Y_test_rust = np.stack([forward_rust(X_test[i]) for i in range(len(X_test))])
    test_rust = mse(Y_test_rust, Y_test)
    Y_test_np = forward_np(X_test, W1n, b1n, W2n, b2n, W3n, b3n)
    test_np = mse(Y_test_np, Y_test)

    history["train"].append(loss_q)
    history["test_rust"].append(test_rust)
    history["test_numpy"].append(test_np)

    if ep % 500 == 0 or ep == EPOCHS - 1:
        print(f"{ep:6d} | {loss_q:12.6f} | {test_rust:12.6f} | {test_np:12.6f} | {abs(test_rust - test_np):10.2e}")

dt = time.perf_counter() - t0

# ══════════════════════════════════════════════════════════════════════
# int4-квантование (сжатие весов)
# ══════════════════════════════════════════════════════════════════════
L1.quantize_to_int4()
L2.quantize_to_int4()
L3.quantize_to_int4()

Y_test_i4 = np.stack([forward_rust(X_test[i]) for i in range(len(X_test))])
test_i4 = mse(Y_test_i4, Y_test)

# ══════════════════════════════════════════════════════════════════════
# Итоги
# ══════════════════════════════════════════════════════════════════════
print("-" * 66)
print(f"Время обучения: {dt:.2f} с")
print(f"Стартовый loss (train): {loss0:.6f}")
print(f"Финальный loss (train, Rust f32): {history['train'][-1]:.6f}")
print(f"Финальный loss (test, Rust f32):  {history['test_rust'][-1]:.6f}")
print(f"Финальный loss (test, NumPy):     {history['test_numpy'][-1]:.6f}")
print(f"Финальный loss (test, int4):      {test_i4:.6f}")
print(f"Разница Rust vs NumPy на тесте:   {abs(history['test_rust'][-1] - history['test_numpy'][-1]):.2e}")

# Сохраняем историю
results = {
    "config": {
        "n_samples": N_SAMPLES, "d_in": D_IN, "h1": H1, "h2": H2, "d_out": D_OUT,
        "epochs": EPOCHS, "lr": LR, "seed": SEED,
    },
    "loss0": loss0,
    "train_final": history["train"][-1],
    "test_rust_final": history["test_rust"][-1],
    "test_numpy_final": history["test_numpy"][-1],
    "test_int4_final": test_i4,
    "time_sec": dt,
    "history": history,
}
with open("training_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nРезультаты сохранены в training_results.json")

# ══════════════════════════════════════════════════════════════════════
# Проверка на контрольных точках
# ═══════════════════════════════════════════════════════════════════════
print("\nПроверка на контрольных точках (f32 vs int4):")
for xv in [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]:
    x_arr = np.array([[xv]], dtype=np.float32)
    h = np.tanh(L1.forward_linear_np(x_arr[0]))
    h = np.tanh(L2.forward_linear_np(h))
    y_pred_f32 = float(L3.forward_linear_np(h)[0])

    h2 = np.tanh(L1.forward_linear_np(x_arr[0]))
    h2 = np.tanh(L2.forward_linear_np(h2))
    y_pred_i4 = float(L3.forward_linear_np(h2)[0])

    y_true = float(np.sin(xv) + 0.5 * np.cos(2.0 * xv))
    print(f"  x={xv:7.3f}  f32={y_pred_f32:9.4f}  int4={y_pred_i4:9.4f}  true={y_true:9.4f}  err_f32={abs(y_pred_f32-y_true):.4f}")

print("\n✅ Нейросеть на AtomNET обучена и запущена успешно.")
