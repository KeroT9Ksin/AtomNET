#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_nnet_python.py — AtomGPT: концепт-нейросеть на ядрах AtomNET (Python).

nano-GPT架构 на примитивах FLASH (GEMM/GEMV/softmax/SiLU/GEMM+SiLU):

    tok ─▶ embedding ─▶ [RoPE позиции] ─▶
        для каждого слоя:
            attn: h = RMSNorm(x) ─▶ Q,K,V ─▶ causal-attn ─▶ @Wo ─▶ x += …
            mlp:  h = RMSNorm(x) ─▶ SiLU(x@Wgate+b)·(x@Wup+b) ─▶ @Wdown + x
        final RMS ─▶ head ─▶ logits

Генерация: префикс целиком — через GEMM, дальше — шаг за шагом
через GEMV + KV-cache (последний токен против всех прошлых).

Демо-режим: веса случайны,ලා. продолжение — из шумовых токенов,
просто показывает связку скоростей.

Запуск:  python3 tests/demo_nnet_python.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

import atomnet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NetConfig:
    """Архитектура nano-GPT (d=128, 2 слоя — компактная наглядность)."""
    vocab: int = 2048
    d: int = 128
    heads: int = 4          # query heads
    kv_heads: int = 2       # GQA
    n_layers: int = 2
    hidden: int = 512       # SwiGLU hidden (4×d)
    rope_base: float = 10000.0
    max_seq: int = 2048

    @property
    def head_dim(self) -> int:
        return self.d // self.heads

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def params_count(self) -> int:
        d, h, L, V = self.d, self.hidden, self.n_layers, self.vocab
        body = L * (d * d + 2 * d * self.kv_dim + d * d + 2 * d * h)
        return V * d + body + d * V + 2 * L * d * 4


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


def rms(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ms = (x * x).mean(axis=-1, keepdims=True)
    normed = x / np.sqrt(ms + eps)
    return (normed * w).astype(np.float32)


def rope_tables(seq_len: int, head_dim: int, base: float):
    inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(seq_len, dtype=np.float32)[:, None]
    ang = np.outer(pos[:, 0], inv)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


# ---------------------------------------------------------------------------
# Linear-аналоги (PyTorch layout: W[out × in])
# ---------------------------------------------------------------------------


class Linear:
    def __init__(self, out_dim: int, in_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.w = (rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward_gemm(self, x: np.ndarray) -> np.ndarray:
        """x (..., in) @ W^T + b — батч GEMM через FLASH."""
        single = x.ndim == 1
        if single:
            x = x[None, :]
        out = atomnet.gemm_blocked_np(x, np.ascontiguousarray(self.w.T))
        out += self.b
        return out[0] if single else out

    def forward_gemv(self, x: np.ndarray) -> np.ndarray:
        """Один вектор: y = W @ x + b — GEMV через FLASH pool."""
        return atomnet.gemv_pooled_np(self.w, x) + self.b


# ---------------------------------------------------------------------------
# Attention (causal, GQA, KV-cache для декодирования)
# ---------------------------------------------------------------------------


class Attention:
    def __init__(self, cfg: NetConfig, seed: int):
        self.cfg = cfg
        self.wq = Linear(cfg.heads * cfg.head_dim, cfg.d, seed + 1)
        self.wk = Linear(cfg.kv_heads * cfg.head_dim, cfg.d, seed + 2)
        self.wv = Linear(cfg.kv_heads * cfg.head_dim, cfg.d, seed + 3)
        self.wo = Linear(cfg.d, cfg.heads * cfg.head_dim, seed + 4)
        self.cache_k: list[np.ndarray] = []
        self.cache_v: list[np.ndarray] = []

    def clear_cache(self):
        self.cache_k.clear()
        self.cache_v.clear()

    def prefill(self, x: np.ndarray) -> np.ndarray:
        """x: (T, d) — полная обработка префикса (non-cached)."""
        cfg = self.cfg
        T = x.shape[0]
        q = self.wq.forward_gemm(x).reshape(T, cfg.heads, cfg.head_dim)
        k = self.wk.forward_gemm(x).reshape(T, cfg.kv_heads, cfg.head_dim)
        v = self.wv.forward_gemm(x).reshape(T, cfg.kv_heads, cfg.head_dim)

        cos_t, sin_t = rope_tables(T, cfg.head_dim, cfg.rope_base)
        cos_t = cos_t[:, None, :]                                     # [T, 1, half]
        sin_t = sin_t[:, None, :]

        def rot(y: np.ndarray) -> np.ndarray:                         # y: [T, H, Dh]
            half = cfg.head_dim // 2
            a, b = y[..., :half], y[..., half:]
            r = np.empty_like(y)
            r[..., :half] = a * cos_t - b * sin_t
            r[..., half:] = a * sin_t + b * cos_t
            return r

        q = rot(q)
        k = rot(k)

        repeat = cfg.heads // cfg.kv_heads
        k_rep = np.transpose(np.repeat(k[None, :, :, :], repeat, axis=2)[:, :, :, :], (0, 2, 1, 3))
        v_rep = np.transpose(np.repeat(v[None, :, :, :], repeat, axis=2)[:, :, :, :], (0, 2, 1, 3))

        q_h = np.transpose(q[None], (0, 2, 1, 3))                     # [1, H, T, Dh]
        scores = (q_h @ np.transpose(k_rep, (0, 1, 3, 2))) * np.float32(1.0 / np.sqrt(cfg.head_dim))
        scores = scores.astype(np.float32)
        mask = np.triu(np.ones((T, T), dtype=bool), 1)[None, None, :, :]
        scores = np.where(mask, -1e9, scores)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        out = (probs @ v_rep).astype(np.float32)                      # [1, H, T, Dh]
        out_t = np.transpose(out[0], (1, 2, 0)).reshape(T, cfg.d)
        return self.wo.forward_gemm(out_t)

    def decode_step(self, x: np.ndarray, pos: int) -> np.ndarray:
        """Инкрементальный шаг: один токен (d,) против KV-кэша."""
        cfg = self.cfg
        q = self.wq.forward_gemv(x)
        k = self.wk.forward_gemv(x)
        v = self.wv.forward_gemv(x)
        self.cache_k.append(k.copy())
        self.cache_v.append(v.copy())

        half = cfg.head_dim // 2
        inv = 1.0 / (cfg.rope_base ** (np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim))
        ang = pos * inv
        qh = q.reshape(cfg.heads, cfg.head_dim)
        ct, st = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        x0, x1 = qh[:, :half].copy(), qh[:, half:].copy()
        qh[:, :half] = x0 * ct - x1 * st
        qh[:, half:] = x0 * st + x1 * ct

        K_all = np.stack(self.cache_k)                                # [T, kv_dim]
        V_all = np.stack(self.cache_v)
        T = K_all.shape[0]
        Kh = K_all.reshape(T, cfg.kv_heads, cfg.head_dim)
        Vh = V_all.reshape(T, cfg.kv_heads, cfg.head_dim)

        pos_ar = np.arange(T, dtype=np.float32)
        ang_all = np.outer(pos_ar, inv)                               # [T, half]
        ct_all, st_all = np.cos(ang_all).astype(np.float32), np.sin(ang_all).astype(np.float32)
        Ka = np.empty_like(Kh)
        a, b = Kh[..., :half], Kh[..., half:]
        Ka[..., :half] = a * ct_all[:, None, :] - b * st_all[:, None, :]
        Ka[..., half:] = a * st_all[:, None, :] + b * ct_all[:, None, :]

        out_d = np.zeros(cfg.d, dtype=np.float32)
        for h in range(cfg.heads):
            kv_i = h // (cfg.heads // cfg.kv_heads)
            s = (Ka[:, kv_i, :] @ qh[h]) * np.float32(1.0 / np.sqrt(cfg.head_dim))
            s = (s - s.max()).astype(np.float32)
            w = np.exp(s)
            w = w / w.sum()
            out_d[h * cfg.head_dim:(h + 1) * cfg.head_dim] = (w @ Vh[:, kv_i, :]).astype(np.float32)
        return self.wo.forward_gemv(out_d)


# ---------------------------------------------------------------------------
# SwiGLU-MLP
# ---------------------------------------------------------------------------


class SwiGLU:
    def __init__(self, cfg: NetConfig, seed: int):
        self.gate = Linear(cfg.hidden, cfg.d, seed + 10)
        self.up = Linear(cfg.hidden, cfg.d, seed + 11)
        self.down = Linear(cfg.d, cfg.hidden, seed + 12)

    def forward_gemm(self, x: np.ndarray) -> np.ndarray:                # (..., d)
        g = atomnet.gemm_silu_np(x, self.gate.w.T, self.gate.b)         # SiLU в JIT-эпилоге
        u = atomnet.gemm_blocked_np(x, self.up.w.T) + self.up.b
        return atomnet.gemm_blocked_np(g * u, self.down.w.T) + self.down.b

    def forward_gemv(self, x: np.ndarray) -> np.ndarray:
        g = atomnet.gemm_silu_np(x[None, :], self.gate.w.T, self.gate.b)[0]
        u = atomnet.gemv_pooled_np(self.up.w, x) + self.up.b
        h = g * u
        return atomnet.gemv_pooled_np(self.down.w, h) + self.down.b


# ---------------------------------------------------------------------------
# Полная модель
# ---------------------------------------------------------------------------


class AtomGPT:
    def __init__(self, cfg: NetConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.wte = (rng.standard_normal((cfg.vocab, cfg.d)) / np.sqrt(cfg.d)).astype(np.float32)
        self.layers = [
            {
                "attn": Attention(cfg, seed + l * 20),
                "mlp": SwiGLU(cfg, seed + l * 40),
                "norm1_w": np.ones(cfg.d, dtype=np.float32),
                "norm2_w": np.ones(cfg.d, dtype=np.float32),
            }
            for l in range(cfg.n_layers)
        ]
        self.norm_w = np.ones(cfg.d, dtype=np.float32)
        self.head = Linear(cfg.vocab, cfg.d, seed + 999)

    # --- полный forward ---
    def forward_full(self, tokens: np.ndarray) -> np.ndarray:
        x = self.wte[tokens]
        for lyr in self.layers:
            h = rms(x, lyr["norm1_w"])
            x = x + lyr["attn"].prefill(h)
            h = rms(x, lyr["norm2_w"])
            x = x + lyr["mlp"].forward_gemm(h)
        return self.head.forward_gemm(rms(x, self.norm_w))

    # --- инкрементальная генерация с KV-cache ---
    def generate_cached(self, prompt: list[int], max_new: int, top_k: int = 8) -> list[int]:
        for lyr in self.layers:
            lyr["attn"].clear_cache()
        x = self.wte[np.asarray(prompt, dtype=np.int64)]
        for lyr in self.layers:
            h = rms(x, lyr["norm1_w"])
            x = x + lyr["attn"].prefill(h)
            h = rms(x, lyr["norm2_w"])
            x = x + lyr["mlp"].forward_gemm(h)
        logits = self.head.forward_gemm(rms(x, self.norm_w))[0]
        tokens = list(prompt)
        last = None
        for _ in range(max_new):
            p = atomnet.softmax_np(logits)
            if top_k <= 0:
                next_tok = int(np.argmax(p))
            else:
                tk = np.argpartition(p, -top_k)[-top_k:]                  # top-k индексы
                next_tok = int(tk[np.argmax(p[tk])])
            tokens.append(next_tok)
            x1 = self.wte[next_tok]
            pos = len(tokens) - 1
            for lyr in self.layers:
                h = rms(x1, lyr["norm1_w"])
                x1 = x1 + lyr["attn"].decode_step(h, pos)
                h = rms(x1, lyr["norm2_w"])
                x1 = x1 + lyr["mlp"].forward_gemv(h)
            logits = self.head.forward_gemv(rms(x1, self.norm_w))
        return tokens

    def save_checkpoint(self, path: str):
        np.savez(
            path,
            wte=self.wte,
            norm=self.norm_w,
            **{f"L{l}_wq": lyr["attn"].wq.w for l, lyr in enumerate(self.layers)},
            **{f"L{l}_wk": lyr["attn"].wk.w for l, lyr in enumerate(self.layers)},
            **{f"L{l}_wv": lyr["attn"].wv.w for l, lyr in enumerate(self.layers)},
            **{f"L{l}_wo": lyr["attn"].wo.w for l, lyr in enumerate(self.layers)},
            **{f"L{l}_gate": lyr["mlp"].gate.w for l, lyr in enumerate(self.layers)},
            **{f"L{l}_up": lyr["mlp"].up.w for l, lyr in enumerate(self.layers)},
            **{f"L{l}_down": lyr["mlp"].down.w for l, lyr in enumerate(self.layers)},
            head_w=self.head.w,
        )

    def load_checkpoint(self, path: str) -> None:
        d = np.load(path)
        self.wte[:] = d["wte"]
        self.norm_w[:] = d["norm"]
        for l, lyr in enumerate(self.layers):
            lyr["attn"].wq.w[:] = d[f"L{l}_wq"]
            lyr["attn"].wk.w[:] = d[f"L{l}_wk"]
            lyr["attn"].wv.w[:] = d[f"L{l}_wv"]
            lyr["attn"].wo.w[:] = d[f"L{l}_wo"]
            lyr["mlp"].gate.w[:] = d[f"L{l}_gate"]
            lyr["mlp"].up.w[:] = d[f"L{l}_up"]
            lyr["mlp"].down.w[:] = d[f"L{l}_down"]
        self.head.w[:] = d["head_w"]


# ---------------------------------------------------------------------------
# SFEROM-память (концепт из Rineto: tri-score routing)
# ---------------------------------------------------------------------------


def sferom_demo():
    def classify(text: str) -> tuple[str, dict]:
        t = text.lower()
        score = {"data_os": 0.0, "memory": 0.0, "query": 0.0}
        for kw in ("data", "os", "kernel", "linux", "процесс", "поток"):
            if kw in t:
                score["data_os"] += 1
        for kw in ("памят", "memory", "cache", "кэш", "storage"):
            if kw in t:
                score["memory"] += 1
        for kw in ("запрос", "query", "вопрос", "how", "what"):
            if kw in t:
                score["query"] += 1
        return max(score, key=score.get), score

    tests = [
        "как работает kernel поток",
        "сколько памяти занимает кэш",
        "что делает запрос",
    ]
    routed = [classify(t)[0] for t in tests]
    print("  SFEROM demo routed:", list(zip(tests, routed)))


def cascade_demo():
    """Концепт RinetoCascade: жёсткие запросы → большой (think),
    простые → компактный (chat). Здесь маленькие модели-примеры, указатели
    на реальные веса поднимаются снаружи (см. transplant)."""
    small = AtomGPT(NetConfig(vocab=2048, d=64, heads=2, kv_heads=2, n_layers=1, hidden=256), seed=1)
    big = AtomGPT(NetConfig(vocab=2048, d=256, heads=8, kv_heads=4, n_layers=4, hidden=1024), seed=2)

    def pick(net, prompt: list[int]):
        return "think" if len(prompt) > 6 else "chat"

    for prompt, _ in [([1, 2, 3], "chat"), ([1, 2, 3, 4, 5, 6, 7, 8, 9], "think")]:
        route = pick(None, prompt)
        net = big if route == "think" else small
        t0 = time.time()
        gen = net.generate_cached(prompt, max_new=4, top_k=6)
        print(f"  cascade → {route} ({'big' if route=='think' else 'small'} net, "
              f"{net.cfg.params_count()/1e6:.1f}M): {gen}  ({(time.time()-t0)*1000:.0f} ms)")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main():
    cfg = NetConfig()
    print(f"AtomGPT: params ≈ {cfg.params_count() / 1e6:.1f}M "
          f"(vocab={cfg.vocab}, d={cfg.d}, layers={cfg.n_layers}, "
          f"heads={cfg.heads}/{cfg.kv_heads} kv, hidden={cfg.hidden})")
    print("  ATOMNET_THREADS:", os.environ.get("ATOMNET_THREADS", "auto"))
    print("  jit_available:", atomnet.jit_available() if hasattr(atomnet, "jit_available") else "n/a")

    net = AtomGPT(cfg, seed=0)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    logits = net.forward_full(np.asarray(prompt, dtype=np.int64))
    print(f"  forward_full ok: logits {logits.shape}")

    t0 = time.time()
    gen = net.generate_cached(prompt, max_new=6)
    dt = time.time() - t0
    print(f"  generate_cached: {gen}  ({dt * 1000:.0f} ms)")

    # roundtrip чекпоинта
    net.save_checkpoint("/tmp/opencode/atomgpt_toy.npz")
    before = net.forward_full(np.asarray(prompt, dtype=np.int64))
    net2 = AtomGPT(cfg, seed=123)
    net2.load_checkpoint("/tmp/opencode/atomgpt_toy.npz")
    after = net2.forward_full(np.asarray(prompt, dtype=np.int64))
    print(f"  checkpoint roundtrip: maxdiff={float(abs(before - after).max()):.2e}")

    sferom_demo()
    print("  --- каскад chat↔think demo ---")
    cascade_demo()
    print("OK: AtomGPT demo готов.")


if __name__ == "__main__":
    main()
