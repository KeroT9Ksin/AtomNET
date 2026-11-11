import time
import numpy as np
import atomnet

def main():
    n = 1 << 20
    w = (np.random.randn(n) * 0.5).astype(np.float32)

    # Детерминизм: два прогона битово равны
    a = np.asarray(atomnet.spherical_normalize_np(w.copy()))
    b = np.asarray(atomnet.spherical_normalize_np(w.copy()))
    print("bitwise equal:", np.array_equal(a, b))
    assert np.array_equal(a, b)

    # Норма после проекции = 1
    print("norm after:", float(np.linalg.norm(a)))

    # Скорость
    t0 = time.perf_counter()
    for _ in range(50): atomnet.spherical_normalize_np(w.copy())
    print(f"spherical: {(time.perf_counter()-t0)/50*1e3:.3f} ms")

    print("✅ ENIGMA FUSED TEST PASSED")

if __name__ == "__main__":
    main()