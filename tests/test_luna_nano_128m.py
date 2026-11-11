import time

import numpy as np
import psutil
import os
import atomnet


def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    print("=== Luna-nano 128M Prototype (AtomMATH vs NumPy) ===")
    print(f"Initial RAM: {get_ram_mb():.1f} MB")

    # =========================================================
    # Архитектура:
    # 8 блоков:
    #   2048 -> 4096
    #   4096 -> 2048
    #
    # Всего 16 слоёв.
    # Параметры:
    #   8 * (2048*4096 + 4096*2048) = 134,217,728
    # =========================================================

    num_blocks = 8
    in_dim = 2048
    hidden_dim = 4096
    num_layers = num_blocks * 2

    print(f"\n[1] Building ~128M model in Rust (Zero-Copy)...")

    t0 = time.perf_counter()

    layers = []

    for _ in range(num_blocks):
        # up: 2048 -> 4096
        up = atomnet.PyQairoLivingLayer.random(hidden_dim, in_dim)

        # down: 4096 -> 2048
        down = atomnet.PyQairoLivingLayer.random(in_dim, hidden_dim)

        layers.append(up)
        layers.append(down)

    build_time = time.perf_counter() - t0
    ram_after_build = get_ram_mb()

    # Проверка размерной цепочки
    for i in range(len(layers) - 1):
        if layers[i].rows != layers[i + 1].dim:
            raise ValueError(
                f"Layer chain mismatch at {i}: "
                f"layers[{i}].rows={layers[i].rows}, "
                f"layers[{i + 1}].dim={layers[i + 1].dim}"
            )

    total_params = sum(layer.rows * layer.dim for layer in layers)

    print(f"Build time: {build_time:.2f} s")
    print(f"RAM after build: {ram_after_build:.1f} MB")
    print(f"Estimated params: {total_params / 1e6:.1f} M")

    # Вход и цель имеют размерность входа/выхода сети.
    x = np.random.randn(in_dim).astype(np.float32) * 0.02
    target = np.random.randn(in_dim).astype(np.float32) * 0.02

    # =========================================================
    # AtomNET Forward (Living)
    # =========================================================

    print("\n[2] AtomNET Forward (Living)...")

    current_x = x
    input_norms = []

    t0 = time.perf_counter()

    for i, layer in enumerate(layers):
        if len(current_x) != layer.dim:
            raise ValueError(
                f"Forward shape mismatch at layer {i}: "
                f"x_len={len(current_x)}, expected_dim={layer.dim}"
            )

        input_norms.append(float(np.linalg.norm(current_x)))
        current_x = layer.forward_np(current_x)

    atom_forward_total = time.perf_counter() - t0
    atom_forward_avg_ms = atom_forward_total / num_layers * 1000

    print(f"Total forward time: {atom_forward_total * 1000:.3f} ms")
    print(f"Avg time per layer (Living): {atom_forward_avg_ms:.3f} ms")
    print(f"Output shape: {current_x.shape}")

    # =========================================================
    # AtomNET Backward (Life Distribution)
    #
    # Важно:
    # Это не настоящий backprop через Qairo.
    # Это синтетический градиент для нагрузки на LifeTime.
    # Настоящий атомарный backward будет позже, в AtomOPTIM.
    # =========================================================

    print("\n[3] AtomNET Backward (Life Distribution, synthetic grads)...")

    grads = []

    for layer in layers:
        g = np.random.randn(layer.rows).astype(np.float32) * 0.02
        grads.append(g)

    t0 = time.perf_counter()

    for i, layer in enumerate(layers):
        layer.distribute_grad_np(grads[i], input_norms[i])

    atom_backward_total = time.perf_counter() - t0
    atom_backward_avg_ms = atom_backward_total / num_layers * 1000

    print(f"Total life-backward time: {atom_backward_total * 1000:.3f} ms")
    print(f"Avg time per layer (Life Backward): {atom_backward_avg_ms:.3f} ms")

    # =========================================================
    # NumPy Baseline (Dense W @ x)
    # =========================================================

    print("\n[4] NumPy Baseline (one equivalent dense layer)...")

    W_np = np.random.randn(hidden_dim, in_dim).astype(np.float32) * 0.02
    b_np = np.zeros(hidden_dim, dtype=np.float32)

    def numpy_layer():
        return W_np @ x + b_np

    # warmup
    for _ in range(5):
        numpy_layer()

    iters = 50

    t0 = time.perf_counter()

    for _ in range(iters):
        numpy_layer()

    numpy_layer_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"NumPy one dense layer: {numpy_layer_ms:.3f} ms")
    print(f"Estimated NumPy 16-layer total: {numpy_layer_ms * num_layers:.3f} ms")

    # =========================================================
    # LifeTime Analysis
    # =========================================================

    print("\n[5] LifeTime Analysis ===")

    inspect_layers = [0, 1, len(layers) // 2, len(layers) - 1]

    for idx in inspect_layers:
        layer = layers[idx]

        states = {
            "dead": 0,
            "cold": 0,
            "born": 0,
            "alive": 0,
            "stable": 0,
            "decaying": 0,
        }

        for row_idx in range(layer.rows):
            st = layer.row_state(row_idx)
            states[st] = states.get(st, 0) + 1

        print(f"\nLayer {idx}: shape {layer.rows}x{layer.dim}")

        for state_name, count in states.items():
            if count > 0:
                print(f"  {state_name:10s}: {count:6d} ({count / layer.rows * 100:5.1f}%)")

    print("\n=== LUNA-NANO 128M TEST FINISHED ===")


if __name__ == "__main__":
    main()
