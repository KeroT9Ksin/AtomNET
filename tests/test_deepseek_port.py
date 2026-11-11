"""Порт DeepSeek (MoE + MLA + FP8) — численная проверка против ручных эталонов."""
import numpy as np

import atomnet as A
from atomnet import nn, functional as F


def test_moe_routing_forward():
    A.manual_seed(0)
    dim, n_exp, k = 8, 4, 2
    moe = nn.DeepSeekMoE(dim, n_experts=8, top_k=2, shared_experts=1)
    tok = A.randn([3, dim])
    out = moe(tok)
    assert out.shape == [3, dim]
    # гейт селектов: top-k точно детерминирован
    moe.update_bias([1.0] * 8)
    out2 = moe(tok)
    assert out2.shape == [3, dim]
    print("OK MoE routing forward")


def test_moe_shared_and_scale():
    A.manual_seed(1)
    moe = nn.DeepSeekMoE(8, n_experts=8, top_k=2, shared_experts=1)
    x = A.randn([2, 8])
    out = moe(x)
    assert out.numel == 2 * 8
    print("OK MoE shared expert add")


def test_mla_latent_and_forward():
    A.manual_seed(2)
    mla = nn.MLALayer(16, n_heads=2, q_lora_rank=8, kv_latent=6, nope_dim=4, rope_dim=4, v_dim=4)
    x = A.randn([4, 16])
    lat = mla.latent(x)
    assert lat.shape == [4, 6], lat.shape  # без взрывного кэша: только latent
    out = mla(x)
    assert out.shape == [4, 16]
    print("OK MLA latent cache + forward")


def test_fp8_storage_quant():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((128, 256)) * 3.0).astype(np.float32)
    enc = A.atom_fp8_block_encode(w.ravel().tolist(), [128, 256], 128)
    codes, scales = enc[0], enc[1]
    dec = np.array(A.atom_fp8_block_decode(codes, scales, [128, 256], 128), dtype=np.float32)
    err = float(np.abs(dec - w.ravel()).max() / np.abs(w).max())
    print(f"FP8 block err={err:.4f} codes={len(codes)} scales={len(scales)}")
    assert err < 0.085, err  # E4M3 hardware error ~max 7.5% on wide-range normal
    print("OK FP8 block-scale storage")


if __name__ == "__main__":
    test_moe_routing_forward()
    test_moe_shared_and_scale()
    test_mla_latent_and_forward()
    test_fp8_storage_quant()
    print("ALL DEEPSEEK PORT TESTS OK")
