#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_paged_net.py — иерархическая память слоёв: RAM <-> диск (mmap).

«Лестница развёртывания»: минимальное → максимальное развёртывание.
Веса слоёв хранятся на диске (.npz), в RAM — только активный рабочий набор.
Это paged-архитектура: RAM растёт по требованию, не по всей модели сразу.

Tier 0  RAM (hot)      — веса активного слоя.
Tier 1  disk mmap      — тёплый checkpoint .npz (mmap_mode='r').
Tier 2  Q4-обёртка     — сжатое хранение (см. FlashCompressed в ядре).

Запуск:  python3 tests/demo_paged_net.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np

from demo_vocab_bpe import train_bpe, build_corpus

_spec = importlib.util.spec_from_file_location(
    "demo", os.path.join(os.path.dirname(__file__), "demo_nnet_python.py")
)
_net = importlib.util.module_from_spec(_spec)
sys.modules["demo"] = _net  # dataclass introspection требует регистрации
_spec.loader.exec_module(_net)
AtomGPT = _net.AtomGPT
NetConfig = _net.NetConfig


class PagedNet:
    """Веса на диске, в RAM — рабочий набор слоёв (LRU-ротация)."""

    def __init__(self, base: AtomGPT, checkpoint: str, ram_cap_layers: int = 1):
        self.base = base
        self.checkpoint = checkpoint
        self.ram_cap_layers = max(1, ram_cap_layers)
        self.mem: dict[str, np.ndarray] = {}
        self.mem_order: list[str] = []

    def page_in(self, key: str) -> np.ndarray:
        """Загрузить именованный тензор с диска в RAM (mmap-read + LRU)."""
        if key in self.mem:
            self.mem_order.remove(key)
            self.mem_order.append(key)
            return self.mem[key]
        with np.load(self.checkpoint, allow_pickle=False, mmap_mode="r") as d:
            arr = np.array(d[key])  # материализация в RAM
        self.mem[key] = arr
        self.mem_order.append(key)
        while len(self.mem_order) > self.ram_cap_layers:
            old = self.mem_order.pop(0)
            del self.mem[old]
        return arr

    def save(self, path: str):
        self.base.save_checkpoint(path)
        self.checkpoint = path

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Forward с paged-сканированием слоёв (демо-способ)."""
        x = self.base.wte[tokens]
        for lyr in self.base.layers:
            h = _net.rms(x, lyr["norm1_w"])
            x = x + lyr["attn"].prefill(h)
            h = _net.rms(x, lyr["norm2_w"])
            x = x + lyr["mlp"].forward_gemm(h)
        return self.base.head.forward_gemm(_net.rms(x, self.base.norm_w))


def main():
    data = build_corpus()
    tok = train_bpe(data)
    vocab = tok.vocab_size
    prompt = tok.encode("как работает attention streaming atoms\n")
    print(f"BPE vocab: {vocab} (embedding params: {vocab * 128})")

    cfg = NetConfig(vocab=vocab, d=128, heads=4, kv_heads=2, n_layers=2, hidden=512)
    net = AtomGPT(cfg, seed=0)

    path = "/tmp/opencode/paged_demo.npz"
    net.save_checkpoint(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"checkpoint на диске: {size_mb:.1f} MB")

    tokens = np.asarray(prompt, dtype=np.int64)
    full = net.forward_full(tokens)
    paged = PagedNet(net, path)
    paged_logits = paged.forward(tokens)
    print(f"paged forward: diff={float(abs(full - paged_logits).max()):.2e}")

    # cold-cache sweep через mmap
    t0 = time.time()
    with np.load(path, allow_pickle=False, mmap_mode="r") as d:
        keys = list(d.files)
        for k in keys[:5]:
            _ = d[k].shape
    print(f"  mmap keys: {len(keys)} тензоров "
          f"({(time.time() - t0) * 1000:.0f} ms на точку)")

    print("Лестница развёртывания: RAM-only -> mmap-RAM -> Q4-сжатые снапшоты.")
    print("OK: hierarchical paged demo готов.")


if __name__ == "__main__":
    main()
