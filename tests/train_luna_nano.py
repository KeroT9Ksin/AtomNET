import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    np.random.seed(11)
    N, D, H, O = 256, 64, 128, 32

    X = (np.random.randn(N, D) * 0.5).astype(np.float32)

    # ── Teacher: 2-слойная tanh-сеть ──
    Wt1 = (np.random.randn(H, D) / np.sqrt(D)).astype(np.float32)
    Wt2 = (np.random.randn(O, H) / np.sqrt(H)).astype(np.float32)
    T = np.tanh(X @ Wt1.T) @ Wt2.T

    # ── Общий init для обоих студентов ──
    W1_0 = (np.random.randn(H, D) * 0.1).astype(np.float32)
    b1_0 = np.zeros(H, dtype=np.float32)
    W2_0 = (np.random.randn(O, H) * 0.1).astype(np.float32)
    b2_0 = np.zeros(O, dtype=np.float32)

    L1 = QL(W1_0.tolist(), b1_0.tolist())
    L2 = QL(W2_0.tolist(), b2_0.tolist())

    W1 = W1_0.copy(); b1 = b1_0.copy()
    W2 = W2_0.copy(); b2 = b2_0.copy()

    lr = 0.3
    epochs = 400

    print("epoch | qairo_loss | numpy_loss")
    t0 = time.perf_counter()
    loss_q = loss_n = float("nan")
    loss0 = None

    for ep in range(epochs):
        # ═══ Qairo forward + backward (Rust ядра) ═══
        Hq = np.empty((N, H), dtype=np.float32)
        Yq = np.empty((N, O), dtype=np.float32)
        for i in range(N):
            h = np.tanh(L1.forward_linear_np(X[i]))
            Hq[i] = h
            Yq[i] = L2.forward_linear_np(h)
        loss_q = float(np.mean((Yq - T) ** 2))
        if loss0 is None:
            loss0 = loss_q

        dY = (2.0 * (Yq - T) / N).astype(np.float32)
        dW2 = dY.T @ Hq
        db2 = dY.sum(axis=0)

        dH = np.empty((N, H), dtype=np.float32)
        for i in range(N):
            dH[i] = L2.backward_input_np(dY[i])   # ← Rust gemv_t
        dPre = dH * (1.0 - Hq ** 2)
        dW1 = dPre.T @ X
        db1 = dPre.sum(axis=0)

        L2.sgd_step_np(dW2, db2, lr)
        L1.sgd_step_np(dW1, db1, lr)

        # ═══ NumPy-зеркало (тот же init, тот же lr) ═══
        Hn = np.tanh(X @ W1.T + b1)
        Yn = Hn @ W2.T + b2
        loss_n = float(np.mean((Yn - T) ** 2))

        dYn = (2.0 * (Yn - T) / N).astype(np.float32)
        dW2n = dYn.T @ Hn
        db2n = dYn.sum(axis=0)
        dHn = dYn @ W2
        dPren = dHn * (1.0 - Hn ** 2)
        dW1n = dPren.T @ X
        db1n = dPren.sum(axis=0)
        W2 -= lr * dW2n; b2 -= lr * db2n
        W1 -= lr * dW1n; b1 -= lr * db1n

        if ep % 100 == 0 or ep == epochs - 1:
            print(f"{ep:5d} | {loss_q:12.6f} | {loss_n:12.6f}")

    dt = time.perf_counter() - t0
    print(f"\ntrain time: {dt:.2f} s")

    # ═══ int4-инференс снятой сети ═══
    L1.quantize_to_int4()
    L2.quantize_to_int4()
    Yi = np.empty((N, O), dtype=np.float32)
    for i in range(N):
        Yi[i] = L2.forward_linear_np(np.tanh(L1.forward_linear_np(X[i])))
    loss_i4 = float(np.mean((Yi - T) ** 2))
    print(f"int4 inference loss: {loss_i4:.6f}  (f32: {loss_q:.6f})")

    # ═══ Life после living-проходов ═══
    for i in range(8):
        L1.forward_np(X[i])
    print("L1 life states: ok")

    # ═══ Критерии успеха ═══
    assert loss_q < 0.2 * loss0, f"сеть не учится: {loss_q} vs старт {loss0}"
    assert abs(loss_q - loss_n) < 5e-3, f"траектории расходятся: {loss_q} vs {loss_n}"
    assert loss_i4 < 2.0 * loss_q + 0.02, "int4 деградировал сильнее допустимого"

    print("\n✅ LUNA-NANO MULTI-LAYER TEST PASSED")
    print(f"   старт {loss0:.4f} → f32 {loss_q:.6f} → int4 {loss_i4:.6f}")


if __name__ == "__main__":
    main()