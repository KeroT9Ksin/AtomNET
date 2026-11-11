import numpy as np
import atomnet

A = getattr(atomnet, "PyFlashArena", None) or atomnet.flash.PyFlashArena


def main():
    np.random.seed(3)
    W = (np.random.randn(2048, 1024) * 0.02).astype(np.float32)
    ar = A.from_numpy(W)
    full = W.nbytes

    m0 = ar.memory_bytes()             # int4 + все f32 (точное)
    for r in range(256):               # греем чанк 0
        ar.note_energy(r, 1.0)
    ar.release_cold()                  # освобождаем мёртвые
    m1 = ar.memory_bytes()             # int4 + только чанк 0 в f32

    hot = np.asarray(ar.read_row_np(5))
    err_hot = float(np.abs(hot - W[5]).max())

    print(f"full f32               : {full/1e6:6.2f} MB")
    print(f"до release (всё точное): {m0/1e6:6.2f} MB")
    print(f"после release (1 чанк) : {m1/1e6:6.2f} MB  (экономия x{m0/m1:.1f})")
    print(f"hot err (точный f32)   : {err_hot:.2e}")

    assert err_hot < 1e-6, "активный чанк должен быть точным как f32"
    # int4-база остаётся на месте, поэтому максимум ~2.7x, а не 3x:
    # 1 горячий чанк + int4 всех весов.
    assert m1 < m0 / 2, f"ожидалось >2x экономии, получено x{m0/m1:.1f}"
    print("\n✅ FLASH ARENA (EXACT MODE) TEST PASSED")


if __name__ == "__main__":
    main()