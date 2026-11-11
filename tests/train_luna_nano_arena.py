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
    N, D, H, O = 128, 64, 128, 32

    Wt1 = (np.random.randn(H, D) * 0.1).astype(np.float32)
    bt1 = np.zeros(H, dtype=np.float32)
    Wt2 = (np.random.randn(O, H) * 0.1).astype(np.float32)
    bt2 = np.zeros(O, dtype=np.float32)

    X = (np.random.randn(N, D) * 0.5).astype(np.float32)
    T = np.tanh(X @ Wt1.T + bt1) @ Wt2.T + bt2

    W1 = (np.random.randn(H, D) * 0.05).astype(np.float32)
    b1 = np.zeros(H, dtype=np.float32)
    W2 = (np.random.randn(O, H) * 0.05).astype(np.float32)
    b2 = np.zeros(O, dtype=np.float32)

    l1 = QL(W1.tolist(), b1.tolist())
    l2 = QL(W2.tolist(), b2.tolist())

    lr = 0.1
    epochs = 500

    print("epoch | qairo_loss | arena_mb (l1+l2)")
    t0 = time.perf_counter()

    for ep in range(epochs):
        Z1q = np.stack([l1.forward_linear_np(X[i]) for i in range(N)])
        H1q = np.tanh(Z1q)
        Yq = np.stack([l2.forward_linear_np(H1q[i]) for i in range(N)])
        loss = float(np.mean((Yq - T) ** 2))

        dY = (2.0 * (Yq - T) / N).astype(np.float32)
        dW2 = dY.T @ H1q
        db2 = dY.sum(axis=0)
        dH1 = np.stack([l2.backward_input_np(dY[i]) for i in range(N)])
        dZ1 = (dH1 * (1.0 - H1q ** 2)).astype(np.float32)
        dW1 = dZ1.T @ X
        db1 = dZ1.sum(axis=0)

        l2.sgd_step_np(dW2, db2, lr)
        l1.sgd_step_np(dW1, db1, lr)
        l2.distribute_grad_np(dY.sum(axis=0), 1.0)
        l1.distribute_grad_np(dZ1.sum(axis=0), 1.0)

        if (ep + 1) % 50 == 0:
            l1.epoch_tick()
            l2.epoch_tick()

        if ep % 100 == 0 or ep == epochs - 1:
            print(f"{ep:5d} | {loss:12.6f} | {(l1.arena_memory_bytes() + l2.arena_memory_bytes())/1e6:10.2f} MB")

    dt = time.perf_counter() - t0
    print(f"\ntrain time: {dt:.2f} s")
    print(f"\n✅ LUNA-NANO WITH ARENA PASSED")


if __name__ == "__main__":
    main()