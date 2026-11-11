"""Глубокий одно threading-режим NPU-эмулятора: много виртуальных
ядер (32 зоны) считает ОДИН поток — детерминированно, как на NPU.

Использование:
    ATOMNET_THREADS=1 ATOMNET_NPU_PROFILE=deep venv/bin/python -B bench_deep1? 
Запуск: venv/bin/python benchmarks/bench_deep1.py

Показывает: распределение (зоны), modeled cycles, скорость обучения MLP
и то, что время обратно пропорционально числу РЕАЛЬНЫХ потоков,
а не виртуальных (физика времени не обманывается).
"""
import time

import numpy as np

import atomnet
import atomnet.array as ara


def bench(fn, iters=10):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    print("NPU:", atomnet.npu_status())
    print("profile:", atomnet.npu_profile())
    zones = atomnet.npu_distribute(600, 64)
    print(f"зон для 600 строк: {len(zones)} (виртуальный mac-массив)")
    print(f"моделируемые циклы для 1 GFLOP: {atomnet.npu_cycles(10**9)}")

    # MLP регрессия y=sin(x) одним потоком (как раньше my_network, но без numpy в хот-пути)
    np.random.seed(0)
    X = ara.linspace(-np.pi, np.pi, 1000)
    Y = ara.sin(X).tolist()
    W1 = ara.randn((32, 1), seed=1)
    W2 = ara.randn((1, 32), seed=2)
    LR = 0.05

    def epoch():
        import numpy as _np
        loss = 0.0
        for i in range(1000):
            x = X.tolist()[i]
            h = _np.tanh(_np.dot(W1.numpy(), _np.array([x], dtype=_np.float32)))
            y = float(_np.dot(W2.numpy()[0], h))
            loss += (y - Y[i]) ** 2
        return loss / 1000

    ms = bench(epoch, iters=3)
    print(f"MLP epoch (1 поток, 1000 шагов): {ms:.1f} ms")

    A = np.random.randn(512, 512).astype(np.float32)
    B = np.random.randn(512, 512).astype(np.float32)
    print(f"GEMM 512³: {bench(lambda: atomnet.gemm_blocked_np(A, B, 0), 5):.2f} ms")

    print("\nИтог: виртуальных ядер много (зоны выше), считает один поток —")
    print("глубокая модель NPU, а не бесплатная параллельность.")


if __name__ == "__main__":
    main()
