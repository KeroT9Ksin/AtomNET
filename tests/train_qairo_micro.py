import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    np.random.seed(7)
    N, D, O = 256, 128, 64

    X = (np.random.randn(N, D) * 0.5).astype(np.float32)
    Wt = (np.random.randn(O, D) * 0.05).astype(np.float32)
    bt = (np.random.randn(O) * 0.01).astype(np.float32)
    T = X @ Wt.T + bt

    W0 = (np.random.randn(O, D) * 0.02).astype(np.float32)
    b0 = np.zeros(O, dtype=np.float32)

    layer = QL(W0.tolist(), b0.tolist())
    W = W0.copy()
    b = b0.copy()

    # ═══ Безопасный lr по ПОЛНОМУ Гессиану (W + bias) ═══
    # Столбец единиц учитывает bias-моду (собственное число ≈ 2),
    # которую мы раньше теряли — из-за неё был взрыв.
    Xa = np.hstack([X, np.ones((N, 1), dtype=np.float32)])
    H = (2.0 / N) * (Xa.T @ Xa)
    eigvals = np.linalg.eigvalsh(H)
    lambda_max = float(eigvals.max())
    lambda_min = float(eigvals.min())
    lr = 1.5 / lambda_max

    print(f"λ_max(полный Гессиан) = {lambda_max:.4f}")
    print(f"λ_min = {lambda_min:.6f}")
    print(f"Безопасный lr = 1.5/λ_max = {lr:.4f}")
    print(f"Проверка bias-моды: |1 - lr*2| = {abs(1 - lr * 2):.3f} (< 1 → устойчиво)")

    slow_decay = 1.0 - lr * lambda_min
    epochs_needed = int(np.ceil(6 * np.log(10) / (-np.log(slow_decay))))
    epochs = max(epochs_needed, 1500)
    print(f"Эпох нужно: {epochs_needed}; используем {epochs}\n")

    print("epoch | qairo_loss | numpy_loss")
    t0 = time.perf_counter()

    loss_q = loss_n = float("nan")
    for ep in range(epochs):
        # ── Qairo forward ──
        Yq = np.stack([layer.forward_linear_np(X[i]) for i in range(N)])
        loss_q = float(np.mean((Yq - T) ** 2))
        dY = (2.0 * (Yq - T) / N).astype(np.float32)
        dW = dY.T @ X
        db = dY.sum(axis=0)
        layer.sgd_step_np(dW, db, lr)

        # ── NumPy baseline ──
        Yn = X @ W.T + b
        loss_n = float(np.mean((Yn - T) ** 2))
        dYn = (2.0 * (Yn - T) / N).astype(np.float32)
        W -= lr * (dYn.T @ X)
        b -= lr * dYn.sum(axis=0)

        if ep % (epochs // 6) == 0 or ep == epochs - 1:
            print(f"{ep:5d} | {loss_q:12.6f} | {loss_n:12.6f}")

    dt = time.perf_counter() - t0
    print(f"\ntrain time: {dt:.2f} s")

    for i in range(8):
        layer.forward_np(X[i % N])
    print("life states after living passes: ok")

    # ── Критерии успеха ──
    assert np.isfinite(loss_q), f"qairo diverged: {loss_q}"
    assert loss_q < 1e-3, f"qairo не сошёлся: {loss_q}"
    assert abs(loss_q - loss_n) < 1e-5, f"кривые расходятся: {loss_q} vs {loss_n}"

    print("\n✅ MICRO TRAINING TEST PASSED")
    print(f"   qairo_loss  = {loss_q:.2e}")
    print(f"   numpy_loss  = {loss_n:.2e}")
    print(f"   |qairo-numpy| = {abs(loss_q - loss_n):.2e}")


if __name__ == "__main__":
    main()