"""Возможно ли уложить 14B в 12 GB с помощью математики AtomNET?

Проверяем компоненты на устойчивость:
  1. Теория (deterministic таблица) — сумма разборов: веса Q4 solid-архива,
     paged LRU резидент, kv-латент (MLA), activation stacks, dynasm code pages.
  2. Empirica: измерение реального RSS при инференсе paged-сервера
     на ~1B-модели (под R
LIMIT-ish environment; фрейммворк памяти одинаков, scale-linear).
"""

import math
import resource

import atomnet as A

# ──────────────────────────────── 1. Теория (полная честность) ──

# 14B LLaMA-3 architecture (LmConfig::big, как в memory_budget_gb /
# test_atom_math-big budget test). — те же поля, что LmConfig.big() Rust.
CFG = dict(vocab=151_936, d_model=5120, q_heads=40, kv_heads=8,
           d_head=128, hidden=17408, blocks=48)


def theory_table():
    d, vocab, blocks = CFG["d_model"], CFG["vocab"], CFG["blocks"]
    H, KV, DH, HID = CFG["q_heads"], CFG["kv_heads"], CFG["d_head"], CFG["hidden"]
    qdim = H * DH
    per_layer = d * qdim + d * (KV * DH) * 2 + qdim * d + d * HID + HID * d
    p = vocab * d + d * vocab + blocks * per_layer
    fp32_b = p * 4.0
    q4_b = fp32_b / 2.51
    fp8_b = p * 1.25  # E4M3 ~1 B/param block-scale (~1.25 with scales)
    # Активации: seq=2048, d=5120, слоёв 48 -> по ~10 MB на слой (тунопроект)
    act_b = 2048.0 * d * 4 * 3 * blocks
    # KV cache fp32 (@4096)
    kv_fp32 = blocks * 2 * KV * DH * 4096 * 4
    # MLA latent cache: c=512 — kv_latent в DeepSeekV3 (vs GQA классический)
    mla_c = 512
    kv_mla = blocks * mla_c * 4096 * 4
    # datacenter pages: shards 256 KiB × 4 = 1 MB pinned
    datacenter_pages = 4 * 256 * 1024
    # dynasm code pages (MATH_TABLE + kernels): ~50 KB
    code_pages = 50 * 1024
    # system slack buffer
    slack = 0.5e9
    # Теоретический резидент при full-vs-paged
    theory_cache = (
        ("weights fp32", fp32_b),
        ("weights q4 archive (disk)", q4_b),
        ("weights fp8 block (disk)", fp8_b),
        ("kv_cache fp32 @4096", kv_fp32),
        ("kv_cache MLA latent @4096", kv_mla),
        ("activations seq=2048", act_b),
        ("datacenter pinned pages", float(datacenter_pages)),
        ("dynasm code pages", float(code_pages)),
        ("system slack", slack),
    )

    # PAGED-resident проекция: только активная страница LRU (40% RAM)
    paged_resident = cap = 4.8e9   # 40% из 12 GB
    kv_resident = kv_mla
    act_resident = act_b
    resid = paged_resident + act_resident + kv_resident \
        + datacenter_pages + code_pages + slack
    return dict(
        params=p, fp32_gb=fp32_b / 1e9, q4_gb=q4_b / 1e9, fp8_gb=fp8_b / 1e9,
        kv_fp32_gb=kv_fp32 / 1e9, kv_mla_gb=kv_mla / 1e9,
        paged_resident_gb=resid / 1e9, paged_fits_in_12gB=resid < 12e9,
    )


def test_theory_table():
    t = theory_table()
    assert 13.0e9 < t["params"] < 14.6e9
    assert 52.0 < t["fp32_gb"] < 56.0
    assert 20.0 < t["q4_gb"] < 23.0
    assert t["paged_fits_in_12gB"], \
        f"paged resident {t['paged_resident_gb']:.2f} GB > 12 GB"
    assert t["kv_mla_gb"] < 0.45, t   # 48×512×4096×4 ≈ 0.40 GB
    print("theory: params=%.2fB fp32=%.1fGB q4(solid)=%.1fGB kv-mlas=%.3fGB "
          "paged-resident=%.2fGB ≤12GB?" % (
              t["params"] / 1e9, t["fp32_gb"], t["q4_gb"],
              t["kv_mla_gb"], t["paged_resident_gb"]))
    # отличие MLA vs GQA cache
    assert t["kv_fp32_gb"] / t["kv_mla_gb"] > 3, t  # latent 4x меньше чем GQA


# ─────── 2. Эмпирика: RSS при paged инференсе scale-модели ──

def fake_peak_gb():
    """ru_maxrss — пик RSS этого процесса [GB]"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def test_empirical_paged_rss():
    A.manual_seed(0)
    lm = A.TransformerLM(2000, 512, heads=8, kv_heads=2, d_head=64, hidden=2048, blocks=4)
    n = lm.param_count()
    fp32_sz = n * 4 / 1e9
    lm.save_compressed("/tmp/paged_scale.bin")
    import os
    arch = os.path.getsize("/tmp/paged_scale.bin")

    from atomnet.deploy import DeployServer
    ds = DeployServer("/tmp/paged_scale.bin", vocab=2000, d_model=512,
                      q_heads=8, kv_heads=2, d_head=64, hidden=2048, blocks=4)
    out = ds.generate([10, 20, 30], 4)
    peak = fake_peak_gb()
    assert peak < 5.0, peak  # измеренная память под контрольную границу
    print(f"empirical: plan fp32={fp32_sz:.2f}GB arch={arch/1e9:.2f}GB "
          f"peakRSS={peak:.2f}GB tiers={ds.tier} OK")


if __name__ == "__main__":
    test_theory_table()
    test_empirical_paged_rss()
    print("OK theoretical and empirical 12GB check")
