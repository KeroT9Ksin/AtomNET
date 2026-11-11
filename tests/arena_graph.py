import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
import atomnet

QL = getattr(atomnet, "PyQairoLivingLayer", None) or atomnet.life.PyQairoLivingLayer


def main():
    np.random.seed(7)
    rows, cols = 4096, 2048
    n_chunks = rows // 256

    print(f"Создаём слой {rows}×{cols} ({rows*cols*4/1e6:.1f} MB в f32)")
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = np.zeros(rows, dtype=np.float32)
    layer = QL(W.tolist(), b.tolist())

    x_test = (np.random.randn(cols) * 0.5).astype(np.float32)
    y_true = W @ x_test

    # Фиксированный активный набор: чанки 0 и 1 (как в реальном обучении)
    active_chunks = [0, 1]
    active_rows = []
    for c in active_chunks:
        active_rows.extend(range(c * 256, (c + 1) * 256))

    cold_probe = 5 * 256  # строка из чанка 5 (будет освобождён)

    n_steps = 20
    mem_log, hot_err_log, cold_err_log, fwd_log = [], [], [], []

    for step in range(n_steps):
        grad = np.zeros(rows, dtype=np.float32)
        grad[active_rows] = 1.0
        layer.distribute_grad_np(grad, 1.0)
        layer.epoch_tick()

        mem_log.append(layer.arena_memory_bytes() / 1e6)

        hot_errs = [float(np.abs(np.asarray(layer.arena_read_row_np(r)) - W[r]).max())
                    for r in active_rows[:32]]
        hot_err_log.append(np.mean(hot_errs))

        cold_errs = [float(np.abs(np.asarray(layer.arena_read_row_np(cold_probe + k)) - W[cold_probe + k]).max())
                     for k in range(32)]
        cold_err_log.append(np.mean(cold_errs))

        y_pred = np.asarray(layer.forward_linear_np(x_test))
        fwd_log.append(float(np.mean((y_pred - y_true) ** 2)))

        print(f"step {step:2d}: mem={mem_log[-1]:6.2f} MB | "
              f"hot={hot_err_log[-1]:.2e} | cold={cold_err_log[-1]:.2e} | fwd={fwd_log[-1]:.2e}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Memory (MB)", color="tab:blue")
    ax1.plot(mem_log, color="tab:blue", marker="o", label="Arena Memory")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Error", color="tab:red")
    ax2.plot(fwd_log, color="tab:red", marker="s", ls="--", label="Forward MSE")
    ax2.plot(hot_err_log, color="tab:green", marker="^", ls=":", label="Hot err (=0)")
    ax2.plot(cold_err_log, color="tab:orange", marker="v", ls="-.", label="Cold err (int4)")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("FlashArena: Memory vs Quality")
    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    plt.savefig("logs/arena_graph.png", dpi=150)
    print("\nГрафик: logs/arena_graph.png")

    print("\n=== ИТОГ ===")
    print(f"Пик   : {max(mem_log):.2f} MB")
    print(f"Мин   : {min(mem_log):.2f} MB")
    print(f"Экономия: x{max(mem_log)/min(mem_log):.1f}")
    print(f"Hot err : {max(hot_err_log):.2e} (=0, точный f32)")
    print(f"Cold err: {cold_err_log[-1]:.2e} (int4, ограничен)")


if __name__ == "__main__":
    main()
