"""Luna-mini — первая нейросеть на AtomNET + PyTorch.

Миниатюрный блок-Transformer (decoder-only) на Rust-ядрах AtomNET и фреймворке PyTorch:

- **PyTorch** отвечает за тренировки (autograd, оптимизатор AdamW, checkpoint).
- **AtomNET** выполняет тяжёлые тензорные ядра: ``matmul`` (SIMD AVX2), а опционально
  ``softmax/gelu/layer_norm`` — через NumPy-мост ``atomnet.math`` (``matmul_np`` и т.д.).

Архитектура ``LunaMini`` повторяет Rust-``LmConfig::nano`` (``atom::model``):
vocab=12, d_model=32, q_heads=2 (GQA: kv_heads=1), d_head=16, hidden=64, blocks=2,
RoPE (современный, не legacy) + RMSNorm + SiLU MLP. Начинает как модель сложения
в одно знакомое представление (как Rust-куррикулум ``luna``).

Хвосты бэкенда:

- ``backend="torch"``  — все Linear на чистом ``torch.nn.Linear``.
- ``backend="atomnet"`` — все Linear исполняются через ``AtomNetLinear``:
  forward/backend используют ``atomnet.math.matmul_np`` (NumPy-совместимый мост,
  SIMD AVX2 внутри), backward тоже через AtomNET-ядра. Bit-identical с PyTorch
  для тех же входов (см. ``tests/test_luna_mini.py``).

Использование:

>>> from atomnet import LunaMini, LunaMiniConfig, encode_sum
>>> cfg = LunaMiniConfig(backend="atomnet")
>>> model = LunaMini(cfg)
>>> logits = model(torch.tensor([[1, 10, 2, 11]]))     # [1, 4, 12]
>>> # Научить сложению одноразрядных чисел: training demo ниже.
>>> from atomnet.luna_mini import train_addition, generate
>>> model, hist = train_addition(cfg, steps=400, lr=3e-3)
>>> generate(model, [1, 10, 2, 11], n_new=2)           # [1, 10, 2, 11, 0, 3]

CLI:

    python -m atomnet.luna_mini train --backend atomnet --steps 400 --save luna-mini.pt
    python -m atomnet.luna_mini generate --ckpt luna-mini.pt -- "1+2="
    python -m atomnet.luna_mini info
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import atomnet.math as _math

# FLASH-GEMM (кэш-блочное 4×16 AVX2-ядро); fallback на старое matmul_np.
_GEMM = getattr(_math, "gemm_blocked_np", _math.matmul_np)


# ═══════════════════════════════════════════════════════════
# Токенизатор сложения (совпадает с Rust `encode1`)
# ═══════════════════════════════════════════════════════════

DIGITS = list(range(10))          # 0..9 → токены 0..9
PLUS = 10
EQUALS = 11
VOCAB = 12


def encode_sum(a: int, b: int) -> list[int]:
    """Кодирует ``a+b=a+b=s`` как [a, '+', b, '=', s//10, s%10] (сложение одноразрядных).

    Совпадает с Rust ``encode1(a, b)`` в ``atom::prod``.
    """
    s = a + b
    return [a, PLUS, b, EQUALS, s // 10, s % 10]


def decode_tokens(tokens: list[int]) -> str:
    """Обратно в строку (для логов)."""
    out = []
    for t in tokens:
        if 0 <= t <= 9:
            out.append(str(t))
        elif t == PLUS:
            out.append("+")
        elif t == EQUALS:
            out.append("=")
        else:
            out.append(f"<{t}>")
    return "".join(out)


# ═══════════════════════════════════════════════════════════
# AtomNET Linear (autograd поверх AtomMATH-ядра)
# ═══════════════════════════════════════════════════════════

class _AtomNetLinearFn(torch.autograd.Function):
    """torch.autograd.Function, который выполняет GEMM через AtomNET.

    forward: X(F32) @ Wᵀ через ``atomnet.math.matmul_np`` (SIMD AVX2 внутри AtomNET);
    backward: grad_X = g @ W, grad_W = gᵀ @ X — тоже через AtomNET-ядра.
    Численно эквивалентен ``torch.nn.Linear`` (bitclose до порядка f32-аккумуляции).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor):
        # x: (..., in), w: (out, in). Flatten → (BT, in).
        lead = x.shape[:-1]
        d_in = x.shape[-1]
        xf = np.ascontiguousarray(
            x.detach().cpu().reshape(-1, d_in).numpy().astype(np.float32)
        )
        wf = np.ascontiguousarray(w.detach().cpu().numpy().astype(np.float32))
        if xf.size > 0:
            out_np = _GEMM(xf, np.ascontiguousarray(wf.T))
            out = torch.from_numpy(np.ascontiguousarray(out_np)).reshape(*lead, -1)
        else:
            out = torch.zeros(*lead, w.shape[0], dtype=torch.float32)
        ctx.save_for_backward(x.detach(), w.detach())
        ctx.lead = lead
        ctx.d_in = d_in
        return out

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        x, w = ctx.saved_tensors
        bt = x.numel() // x.shape[-1]
        gf = np.ascontiguousarray(g.detach().cpu().reshape(bt, -1).numpy().astype(np.float32))
        xf = np.ascontiguousarray(x.cpu().reshape(bt, -1).numpy().astype(np.float32))
        wf = np.ascontiguousarray(w.cpu().numpy().astype(np.float32))
        # grad_W = gᵀ @ x ; grad_x = g @ W — оба через AtomNET-ядра.
        grad_w_np = _GEMM(np.ascontiguousarray(gf.T), xf)
        grad_x_np = _GEMM(gf, wf)
        grad_x = torch.from_numpy(np.ascontiguousarray(grad_x_np)).reshape(*ctx.lead, ctx.d_in)
        grad_w = torch.from_numpy(np.ascontiguousarray(grad_w_np))
        return grad_x, grad_w


class AtomNetLinear(nn.Module):
    """nn.Linear, исполняемый на AtomNET SIMD-ядрах (matmul_np).

    Сигнатура совместима с :class:`torch.nn.Linear`: параметры ``weight`` (out,in),
    ``bias`` (out,) — они остаются обычными torch.Tensor, так что checkpoint
    (state_dict) совместим с ``backend="torch"``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _AtomNetLinearFn.apply(x, self.weight)
        return y + self.bias if self.bias is not None else y


def _make_linear(in_features: int, out_features: int, backend: str) -> nn.Module:
    if backend == "atomnet":
        return AtomNetLinear(in_features, out_features)
    return nn.Linear(in_features, out_features)


# ═══════════════════════════════════════════════════════════
# Конфиг + блоки Luna-mini
# ═══════════════════════════════════════════════════════════

@dataclass
class LunaMiniConfig:
    """Конфиг Luna-mini (совпадает с Rust `LmConfig::nano`)."""

    vocab: int = 12
    d_model: int = 32
    q_heads: int = 2
    kv_heads: int = 1
    d_head: int = 16
    hidden: int = 64
    blocks: int = 2
    rope_theta: float = 10000.0
    backend: str = "atomnet"  # "atomnet" или "torch"

    def assert_consistent(self):
        assert self.d_model % self.q_heads == 0
        assert self.d_head * self.q_heads == self.d_model, "d_head*q_heads must equal d_model"
        assert self.q_heads % self.kv_heads == 0


class RMSNorm(nn.Module):
    """RMSNorm (как в Llama), считается на fp32 для стабильности."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        return (x * self.weight).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE (стандартный): x: (B, H, T, D). cos/sin: (T, D) с полной шириной D."""
    return (x * cos) + (_rotate_half(x) * sin)


def _rope_tables(T: int, D: int, theta: float, device: torch.device):
    """cos/sin таблички RoPE: (T, D) в dtype float32, ширина D (полная)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, device=device).float() / D))
    pos = torch.arange(T, device=device).float()
    angles = torch.outer(pos, inv_freq)         # (T, D/2)
    cos = angles.cos()
    sin = angles.sin()
    return (
        torch.cat((cos, cos), dim=-1),          # (T, D) — для rotate_half пары
        torch.cat((sin, sin), dim=-1),
    )


class AttnBlock(nn.Module):
    """GQA-attention с RoPE (как в Rust `atom::seq::gqa_fwd`)."""

    def __init__(self, cfg: LunaMiniConfig):
        super().__init__()
        self.q_heads = cfg.q_heads
        self.kv_heads = cfg.kv_heads
        self.d_head = cfg.d_head
        self.backend = cfg.backend
        kv_dim = cfg.kv_heads * cfg.d_head
        self.wq = _make_linear(cfg.d_model, cfg.q_heads * cfg.d_head, cfg.backend)
        self.wk = _make_linear(cfg.d_model, kv_dim, cfg.backend)
        self.wv = _make_linear(cfg.d_model, kv_dim, cfg.backend)
        self.wo = _make_linear(cfg.q_heads * cfg.d_head, cfg.d_model, cfg.backend)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.q_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv_heads, self.d_head).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv_heads, self.d_head).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA: расширяем KV на q_heads.
        rep = self.q_heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.q_heads * self.d_head)
        return self.wo(out)


class MlpBlock(nn.Module):
    """SiLU-MLP (как Rust: ``down(silu(up(x)))``)."""

    def __init__(self, cfg: LunaMiniConfig):
        super().__init__()
        self.up = _make_linear(cfg.d_model, cfg.hidden, cfg.backend)
        self.down = _make_linear(cfg.hidden, cfg.d_model, cfg.backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LunaMiniConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = AttnBlock(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = MlpBlock(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LunaMini(nn.Module):
    """Luna-mini: decoder-only Transformer на AtomNET + PyTorch."""

    def __init__(self, cfg: LunaMiniConfig | None = None):
        super().__init__()
        self.cfg = cfg or LunaMiniConfig()
        self.cfg.assert_consistent()
        cfg = self.cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.blocks))
        self.norm = RMSNorm(cfg.d_model)
        # Tied LM-head (вес = embed): меньше параметров, как у Llama/GPT.
        self.head_weight = self.tok_emb.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) long → logits: (B, T, vocab)."""
        x = self.tok_emb(input_ids)  # (B, T, D)
        T = input_ids.shape[1]
        cos, sin = _rope_tables(T, self.cfg.d_head, self.cfg.rope_theta, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = F.linear(self.norm(x), self.head_weight)  # tied
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ─── Checkpoint ──────────────────────────────────────────────────
    def save(self, path: str):
        torch.save(
            {
                "type": "luna-mini",
                "config": self.cfg,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LunaMini":
        blob = torch.load(path, map_location=device, weights_only=False)
        model = cls(blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


# ═══════════════════════════════════════════════════════════
# Генерация и обучение
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model: LunaMini, prefix: list[int], n_new: int = 2, device: str = "cpu") -> list[int]:
    """Жадная генерация: argmax после префикса (как Rust ``generate``)."""
    model.eval()
    ids = list(prefix)
    for _ in range(n_new):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1]
        ids.append(int(logits.argmax().item()))
    return ids


def addition_dataset() -> list[tuple[list[int], list[int]]]:
    """Все одноразрядные пары: (inputs, targets) для next-token обучения."""
    data = []
    for a in range(10):
        for b in range(10):
            seq = encode_sum(a, b)
            data.append((seq[:-1], seq[1:]))
    return data


def exact_match(model: LunaMini, pairs: list[tuple[int, int]], device: str = "cpu") -> float:
    """Точность: модель даёт оба выходных токена суммы (s//10, s%10) жадной генерацией."""
    model.eval()
    ok = 0
    for a, b in pairs:
        seq = encode_sum(a, b)
        pred = generate(model, seq[:-2], n_new=2, device=device)
        if pred[-2:] == seq[-2:]:
            ok += 1
    return ok / max(1, len(pairs))


def train_addition(
    cfg: LunaMiniConfig | None = None,
    steps: int = 900,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = True,
    device: str = "cpu",
) -> tuple[LunaMini, list[tuple[int, float]]]:
    """Полный обучающий цикл Luna-mini: CE loss, AdamW, жадная оценка точного совпадения.

    Returns: (model, history) где history = [(step, loss), ...] на каждом шаге.
    """
    torch.manual_seed(seed)
    model = LunaMini(cfg or LunaMiniConfig())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    data = addition_dataset()
    X = torch.tensor([d[0] for d in data], dtype=torch.long, device=device)
    Y = torch.tensor([d[1] for d in data], dtype=torch.long, device=device)
    model.train()
    hist: list[tuple[int, float]] = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1))
        loss.backward()
        opt.step()
        hist.append((step, float(loss.item())))
        if verbose and (step % 100 == 0 or step == 1):
            em = exact_match(model, [(a, b) for a in range(10) for b in range(10,)], device)
            print(
                f"step {step:4d} | loss {loss.item():.4f} | exact-match {em:.2f} | "
                f"params {model.num_parameters()}"
            )
    return model, hist


# ═══════════════════════════════════════════════════════════
# CLI: python -m atomnet.luna_mini ...
# ═══════════════════════════════════════════════════════════

def _cli_train(args):
    cfg = LunaMiniConfig(backend=args.backend)
    model, _ = train_addition(cfg, steps=args.steps, lr=args.lr, seed=args.seed)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    em = exact_match(model, pairs)
    print(f"[luna-mini] train done: backend={args.backend} steps={args.steps} exact-match={em:.3f} params={model.num_parameters()}")
    if args.save:
        model.save(args.save)
        print(f"[luna-mini] saved → {args.save}")


def _cli_generate(args):
    model = LunaMini.load(args.ckpt)
    prefix = _parse_expr(args.expr)
    out = generate(model, prefix, n_new=2)
    print(f"{args.expr} = {decode_tokens(out[-2:])}  (full: {decode_tokens(out)})")
    # Печать всей последовательности: "a+b=s…"
    seq = out
    print("tokens:", seq)


def _parse_expr(expr: str) -> list[int]:
    """«2+7=» → [2, 10, 7, 11] (допустимы многозначные числа по цифрам)."""
    expr = expr.strip()
    toks: list[int] = []
    for ch in expr:
        if ch.isdigit():
            toks.append(ord(ch) - 48)
        elif ch == "+":
            toks.append(PLUS)
        elif ch == "=":
            toks.append(EQUALS)
        elif ch in " \t":
            continue
        else:
            raise SystemExit(f"не поддерживаемый символ: {ch!r}")
    return toks


def main():
    parser = argparse.ArgumentParser(prog="luna-mini", description="Luna-mini: AtomNET + PyTorch")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train", help="обучить Luna-mini на сложении")
    p_tr.add_argument("--backend", choices=["atomnet", "torch"], default="atomnet")
    p_tr.add_argument("--steps", type=int, default=500)
    p_tr.add_argument("--lr", type=float, default=3e-3)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--save", type=str, default="luna-mini.pt")
    p_tr.set_defaults(func=_cli_train)

    p_gen = sub.add_parser("generate", help="жадная генерация из чекпойнта")
    p_gen.add_argument("--ckpt", type=str, default="luna-mini.pt")
    p_gen.add_argument("expr", help='префикс: "2+7=" (цифры, +, =)')
    p_gen.set_defaults(func=_cli_generate)

    p_info = sub.add_parser("info", help="инфо о модели")
    p_info.set_defaults(func=lambda args: print(
        f"Luna-mini | vocab=12 d=32 h=16 blocks=2 hidden=64 | "
        f"params={LunaMini().num_parameters()} | backends: torch / atomnet (matmul_np)"
    ))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
