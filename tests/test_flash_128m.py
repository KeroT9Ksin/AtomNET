import time

import numpy as np
import psutil
import os
import atomnet


def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def main():
    print("=== FLASH 128M Prototype (AtomNET vs NumPy) ===")
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

    print(f"\n[1] Building ~128M FLASH model in Rust...")

    t0 = time.perf_counter()

    layers = []

    for _ in range(num_blocks):
        up = atomnet.PyFlashLayer.random(hidden_dim, in_dim, 0.02)
        down = atomnet.PyFlashLayer.random(in_dim, hidden_dim, 0.02)

        layers.append(up)
        layers.append(down)

    build_time = time.perf_counter() - t0
    ram_after_build = get_ram_mb()

    total_params = sum(layer.out_features * layer.in_features for layer in layers)

    print(f"Build time: {build_time:.2f} s")
    print(f"RAM after build: {ram_after_build:.1f} MB")
    print(f"Estimated params: {total_params / 1e6:.1f} M")

    x = np.random.randn(in_dim).astype(np.float32) * 0.02

    # =========================================================
    # FLASH Forward (Living)
    # =========================================================

    print("\n[2] FLASH Forward (Living)...")

    current_x = x
    input_norms = []

    t0 = time.perf_counter()

    for layer in layers:
        input_norms.append(float(np.linalg.norm(current_x)))
        current_x = layer.forward_living_np(current_x)

    flash_forward_total = time.perf_counter() - t0
    flash_forward_avg_ms = flash_forward_total / num_layers * 1000

    print(f"Total forward time: {flash_forward_total * 1000:.3f} ms")
    print(f"Avg time per layer (Living): {flash_forward_avg_ms:.3f} ms")
    print(f"Output shape: {current_x.shape}")

    # =========================================================
    # FLASH Life Backward (synthetic grads)
    # =========================================================

    print("\n[3] FLASH Backward (Life Distribution, synthetic grads)...")

    grads = []

    for layer in layers:
        g = np.random.randn(layer.out_features).astype(np.float32) * 0.02
        grads.append(g)

    t0 = time.perf_counter()

    for i, layer in enumerate(layers):
        layer.distribute_grad_np(grads[i], input_norms[i])

    flash_backward_total = time.perf_counter() - t0
    flash_backward_avg_ms = flash_backward_total / num_layers * 1000

    print(f"Total life-backward time: {flash_backward_total * 1000:.3f} ms")
    print(f"Avg time per layer (Life Backward): {flash_backward_avg_ms:.3f} ms")

    # =========================================================
    # NumPy Baseline
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

    print("\n[5] FLASH LifeTime Analysis ===")

    inspect_layers = [0, 1, len(layers) // 2, len(layers) - 1]

    for idx in inspect_layers:
        layer = layers[idx]
        dead, cold, born, alive, stable, decaying = layer.state_counts()
        total = layer.out_features

        print(f"\nLayer {idx}: shape {layer.out_features}x{layer.in_features}")

        states = {
            "dead": dead,
            "cold": cold,
            "born": born,
            "alive": alive,
            "stable": stable,
            "decaying": decaying,
        }

        for name, count in states.items():
            if count > 0:
                print(f"  {name:10s}: {count:6d} ({count / total * 100:5.1f}%)")

    print("\n=== FLASH 128M TEST FINISHED ===")


if __name__ == "__main__":
    main()