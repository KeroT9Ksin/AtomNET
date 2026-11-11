import numpy as np
import atomnet

A = getattr(atomnet, "PyFlashArena", None) or atomnet.flash.PyFlashArena


def main():
    np.random.seed(3)
    W = (np.random.randn(4096, 1024) * 0.02).astype(np.float32)
    ar = A.from_numpy(W)

    m0 = ar.memory_bytes() / 1e6

    # греем чанк 0
    for r in range(256):
        ar.note_energy(r, 1.0)

    # XOR-управляемое освобождение
    ar.release_cold_xor()
    tiers = ar.xor_tiers()
    m1 = ar.memory_bytes() / 1e6

    print(f"до  : {m0:.2f} MB")
    print(f"после: {m1:.2f} MB  (экономия x{m0/m1:.1f})")
    print(f"tiers[:4] = {list(tiers[:4])}  (чанк0=2 hot, остальные=0 ice)")
    print(f"ice-чанков: {sum(1 for t in tiers if t == 0)} / {len(tiers)}")

    assert tiers[0] == 2, "активный чанк должен быть hot"
    assert sum(1 for t in tiers if t == 0) == 15, "15 холодных чанков = ice"
    assert m1 < m0 / 2
    print("\n✅ XOR ALLOC TEST PASSED")


if __name__ == "__main__":
    main()
