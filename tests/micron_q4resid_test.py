import numpy as np
import atomnet

def main():
    np.random.seed(3)
    w = (np.random.randn(1 << 20) * 0.02).astype(np.float32)

    # чистый int4 (для сравнения)
    q4, mu, sc = atomnet.quantize_int8_np(w)  # int8 как прокси; ниже int4-ошибка из арены
    d4 = np.asarray(atomnet.dequantize_int8_np(q4, mu, sc))
    err_int8 = float(np.abs(d4 - w).max())

    # двухуровневый q4+resid
    qr = atomnet.quantize_q4_resid_np(w, 32)
    out = np.zeros_like(w)
    atomnet.dequantize_q4_resid_np(qr, out)
    err_res = float(np.abs(out - w).max())

    full = w.nbytes
    print(f"f32          : {full/1e6:.2f} MB")
    print(f"int8 err     : {err_int8:.2e}")
    print(f"q4+resid err : {err_res:.2e}  (f32-подобие)")
    print(f"q4+resid bytes: {atomnet.q4_resid_bytes_np(qr)/1e6:.2f} MB (x{full/atomnet.q4_resid_bytes_np(qr):.1f})")

    assert err_res < err_int8, "остаток должен улучшить точность"
    print("\n✅ MICRON Q4-RESID TEST PASSED")

if __name__ == "__main__":
    main()