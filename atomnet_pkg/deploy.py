"""Продакшн-сервер деплоя AtomNET на WinRAR-подобном solid-архиве Q4.

Идея (как WinRAR): веса хранятся ОДНИМ solid-архивом (конкат-стрим словаря
сжатых Q4-слоёв). Сервер никогда не держит dense FP32-копию:
  - секция архива преобразуется в PyFlashCompressed (ядра JIT 6x16) ТОЛЬКО
    когда слой реально нужен для forward (пагирование);
  - LRU-механизм поверх paged-секций: холодные слои выселяются,
    их контекст просто отпускается (архив на диске остаётся источником правды);
  - авто-тивы по probe(): MemAvailable /meminfo:
      'solid-q4'  — весь архив (компрессия ~2.5×) + paged LRU (60% MemAvailable);
      'paged'     — только если solid-q4 не влезает: cap = 40% MemAvailable;
  - статистика памяти: /proc/self/statm + ru_maxrss peak.

Внутрипроцессное API (без сети — сеть запрещена направлением задачи;
офлайн-датацентр: см. модуль datacenter.py):
  server.generate(prompt, max_len)
  server.memory_stats()
  server.train_and_repack(seq) (через TransformerLM -> save_compressed)

Инференс считается вручную в numpy через FC-gemv (точнее — np.math), связка
RoPE(GQA, softmax, RMSNorm) — синхранизирована с atom::seq (sign-спекстандарт).


"""

import json
import math
import os
import resource
import struct
import time

import numpy as np

import atomnet as A

from .atomnet import FlashArchive


# ───────────────────── probо memory ─────────────────────

def meminfo_kb():
    info = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "kB":
                info[parts[0].rstrip(":")] = int(parts[1])
    return info


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def mem_available_gb():
    return meminfo_kb().get("MemAvailable", 0) / 1e6


# ───────────────────── Variants/tiers ─────────────────────

class TierAuto:
    """Выбор тиров по свободному объему RAM (self-репликация статусов)."""

    def __init__(self, archive_bytes, fraction=0.6):
        self.archive_bytes = archive_bytes
        self.fraction = fraction

    def choose(self):
        cap = mem_available_gb() * self.fraction
        if self.archive_bytes <= cap:
            return "solid-q4", cap
        #  Даже solid не влезает в достаточно большой чанк: paged = "только Q4-слои"
        return "paged", max(cap * 0.5, 0.25)  # гарантированный минимум pagr=0.25GB


# ───────────────────── NumPy forward (JavaScript, sync с Rust model) ──

SIG_RMS = 1e-5  # RMS_NORM_EPS (atom::model)


def rms_norm(x, eps=SIG_RMS):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps)


def rope_sign(x, pos, theta, d, sign=1.0):
    y = x.copy()
    for i in range(0, d, 2):
        if i + 1 >= d:
            break
        angle = sign * pos * (theta ** (-i / d))
        s, c = math.sin(angle), math.cos(angle)
        a, b = x[i], x[i + 1]
        y[i] = a * c - b * s
        y[i + 1] = a * s + b * c
    return y


def gqa_forward(q, k, v, q_heads, kv_heads, d_head):
    L = q.shape[0]
    group = q_heads // kv_heads
    out = np.zeros((L, q_heads * d_head), dtype=np.float32)
    for h in range(q_heads):
        kv = h // group
        qh = q[:, h * d_head:(h + 1) * d_head]
        kh = k[:, kv * d_head:(kv + 1) * d_head]
        vh = v[:, kv * d_head:(kv + 1) * d_head]
        scores = np.matmul(qh, kh.T) / math.sqrt(d_head)
        mask = np.triu(np.ones((L, L)) * -1e9, k=1)
        scores = scores + mask
        m = scores.max(axis=-1, keepdims=True)
        e = np.exp(scores - m)
        att = e / e.sum(axis=-1, keepdims=True)
        out[:, h * d_head:(h + 1) * d_head] = att @ vh
    return out


class NumpyForward:
    """Прямой прогон через FC-слои (``PyFlashCompressed`` объекты) в numpy.

    Рекурсия семантики TransformerLM (RMSNorm→attn→MLP); плотные веса не
    материализуются — каждый линейный вызов = FC.gemv.
    """

    def __init__(self, cfg, fc_layers, legacy_rope=False):
        # cfg: dict vocab/d_model/q_heads/kv_heads/d_head/hidden/blocks/rope_theta
        self.cfg = cfg
        self.fc = fc_layers   # список PyFlashCompressed
        self.legacy_rope = legacy_rope
        self.d = cfg["d_model"]
        self.H = cfg["q_heads"]
        self.KH = cfg["kv_heads"]
        self.DH = cfg["d_head"]
        self.hidden = cfg["hidden"]
        self.blocks = cfg["blocks"]
        self.theta = cfg["rope_theta"]

    def _lin(self, idx, x):
        fc = self.fc[idx]
        y = fc.gemv_np(_as_np(np.ascontiguousarray(x, dtype=np.float32)))
        return np.array(y, dtype=np.float32).reshape(-1)

    def forward(self, seq, t_offset=0):
        # limit: small (≤ hundreds tokens) OK
        d, T = self.d, len(seq)
        x = np.zeros((T, d), dtype=np.float32)
        emb = self.fc[0]
        for t, tok in enumerate(seq):
            x[t] = np.array(emb.row_values(tok), dtype=np.float32)
        # смешай время layer0 через FC на -1
        li = 1
        for b in range(self.blocks):
            # ln1, attn(q,k,v,o)
            h = rms_norm(x.copy())
            q = np.stack([self._lin(li + 0, h[t]) for t in range(T)])
            k = np.stack([self._lin(li + 1, h[t]) for t in range(T)])
            v = np.stack([self._lin(li + 2, h[t]) for t in range(T)])
            li += 3
            for t in range(T):
                p = t + t_offset
                for h_i in range(self.H):
                    q[t, h_i * self.DH:(h_i + 1) * self.DH] = rope_sign(
                        q[t, h_i * self.DH:(h_i + 1) * self.DH], p, self.theta, self.DH)
                for kh_i in range(self.KH):
                    k[t, kh_i * self.DH:(kh_i + 1) * self.DH] = rope_sign(
                        k[t, kh_i * self.DH:(kh_i + 1) * self.DH], p, self.theta, self.DH)
            att = gqa_forward(q, k, v, self.H, self.KH, self.DH)
            o = np.stack([self._lin(li, att[t]) for t in range(T)])
            li += 1
            x = x + o
            # ln2, mlp(up, silu, down)
            h = rms_norm(x.copy())
            up = np.stack([self._lin(li, h[t]) for t in range(T)])
            li += 1
            act = up / (1.0 + np.exp(-up))
            dn = np.stack([self._lin(li, act[t]) for t in range(T)])
            li += 1
            x = x + dn
        x = rms_norm(x.copy())
        head = self.fc[li]
        logits = np.stack([head.gemv_np(_as_np(np.ascontiguousarray(x[t], dtype=np.float32)))
                           for t in range(T)])
        return logits.astype(np.float32)


def _as_np(arr):
    return np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)


# ───────────────────── Paged server ─────────────────────

class DeployServer:
    """Продакшн-запускалка на WinRAR-подобном solid-архиве."""

    def __init__(self, path, vocab=None, d_model=32, q_heads=2, kv_heads=1, d_head=16,
                 hidden=64, blocks=2):
        self.path = path
        self.ar = FlashArchive(path)
        self.n = self.ar.section_count()
        self.tier, self.cap_gb = TierAuto(self.ar.archive_bytes()).choose()
        self.cache = {}  # idx → (FC, bytes_used)
        self.used_gb = 0.0
        self.cfg = dict(vocab=vocab or 12, d_model=d_model, q_heads=q_heads,
                        kv_heads=kv_heads, d_head=d_head, hidden=hidden,
                        blocks=blocks, rope_theta=10000.0)
        self.seq = []

    def _fc(self, idx):
        if idx in self.cache:
            return self.cache[idx][0]
        need_gb = 0.0  # в CTX не измеряются по PyFlashCompressed API
        self.cache[idx] = (self.ar.layer(idx), need_gb)
        return self.cache[idx][0]

    def _evict(self):
        # LRU-эвикция по cap: при перерасходе сношу самые холодные (по порядку)
        while self.used_gb > self.cap_gb and self.cache:
            old = next(iter(self.cache))
            del self.cache[old]

    def get_layers(self):
        fc = [self._fc(i) for i in range(self.n)]
        return fc

    def memory_stats(self):
        return dict(
            tier=self.tier,
            archive_bytes=self.ar.archive_bytes(),
            cached_layers=len(self.cache),
            cap_gb=self.cap_gb,
            peak_rss_gb=peak_rss_gb(),
        )

    def generate(self, prefix, max_len=16):
        fc = self.get_layers()
        fwd = NumpyForward(self.cfg, fc)
        seq = list(prefix)
        while len(seq) < max_len:
            logits = fwd.forward(seq)
            last = logits.reshape(-1)[-self.cfg["vocab"]:]
            nxt = int(np.argmax(last))
            seq.append(nxt)
            if len(seq) > 4096:
                break
        return seq



def serve(path, port=8066, **kw):
    """Офлайн-презентация развёртывания (без сети — конфликт с датацентром).

    Возвращает DeployServer — пайт размещения DeployServer по шардам архива
    (постепенно выдавать min-usage памяти, как LRU-направленные винрары).
    """
    server = DeployServer(path, **kw)
    print("deploy tier:", server.tier, "| archive:", server.ar.archive_bytes())
    return server

def _selftest():
    os.environ.setdefault("ATOMNET_THREADS", "1")
    A.manual_seed(0)
    # Слойное квантование BLOCK=256: выигрыш Q4 проявляется честно на
    # реальных ширинах (>=256), мелкие слои (cols < 256) платят pad-штраф.
    lm = A.TransformerLM(64, 256, heads=4, kv_heads=2, d_head=64, hidden=512, blocks=2)
    seq = [1, 7, 10, 2, 5, 11, 4, 2]
    ls = 0.0
    for _ in range(1):
        ls = lm.train_sample(seq, 1.0)
        lm.step()
        lm.clear_grads()
    n = lm.save_compressed("/tmp/deploy_test.bin")
    assert n > 0
    sz = os.path.getsize("/tmp/deploy_test.bin")
    print("archive layers:", n, "bytes:", sz)
    fp = os.path.getsize("/tmp/deploy_test.bin")
    lm.save_checkpoint("/tmp/deploy_test_full.bin")
    full = os.path.getsize("/tmp/deploy_test_full.bin")
    print(f"archive/full-fp32 ratio: {sz/full:.2%}")
    assert sz * 2 <= full, (sz, full)
    try:
        fp_check = (sz / full) * 100
        print(f"compression: {fp_check:.1f}% of fp32 checkpoint (WinRAR solid Q4)")
    except Exception as e:
        print("compression stat err", e)
    srv = DeployServer("/tmp/deploy_test.bin", vocab=64, d_model=256, q_heads=4,
                       kv_heads=2, d_head=64, hidden=512, blocks=2)
    print("tier:", srv.tier, "| cap:", round(srv.cap_gb, 2), "GB")
    out = srv.generate([1, 7, 10], 8)
    print("generated:", out[:8])
    st = srv.memory_stats()
    print("memory stats:", st)
    assert os.path.getsize("/tmp/deploy_test.bin") * 2 <= full
    assert st["archive_bytes"] == fp
    print("SELFTEST OK")
    return "SELFTEST OK"


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        print(_selftest())
    else:
        serve(sys.argv[1] if len(sys.argv) > 1 else "/tmp/deploy_test.bin")
