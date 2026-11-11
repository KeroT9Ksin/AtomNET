import numpy as np

import atomnet

QL = atomnet.PyQairoLivingLayer


def main():
    np.random.seed(11)
    rows, cols = 512, 256
    W = (np.random.randn(rows, cols) * 0.02).astype(np.float32)
    b = np.zeros(rows, dtype=np.float32)
    layer = QL(W.tolist(), b.tolist())

    before = layer.arena_memory_bytes()
    layer.set_ram_budget(1 * 1024 * 1024)
    layer.epoch_tick()
    after = layer.arena_memory_bytes()

    print(f"arena before: {before / 1e6:.2f} MB")
    print(f"arena after : {after / 1e6:.2f} MB")
    assert after <= before

    # Горячие строки должны оставаться точными; холодные читаются из int4.
    for r in range(0, rows, 64):
        row = np.asarray(layer.arena_read_row_np(r))
        if layer.arena_is_hot(r):
            err = float(np.abs(row - W[r]).max())
            assert err < 1e-6, f"hot row {r} err={err}"
    print("region/arena test passed")


if __name__ == "__main__":
    main()
