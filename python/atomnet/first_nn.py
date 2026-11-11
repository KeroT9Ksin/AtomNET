"""Первая нейросеть AtomNET — MiniTransformer на torch-API.

Архитектура каноническая (nanoGPT / LLaMA):
    tok_emb + sinusoidal positions
    N × [ RMSNorm -> CausalSelfAttention(QKV, softmax, causal) -> +res
          RMSNorm -> MLP (Linear 4x, SiLU, Linear) -> +res ]
    RMSNorm -> Linear(vocab)
Loss = cross_entropy по последней позиции. Оптимизатор — Adam (atomnet.optimizers).

Путь обучения (single-thread NPU): чистый autograd, без Keralin. Forward и
Backward GEMM идут через npu_router (или ATOMNET_NPU_EMU=1 — JIT-тайлы 6x16,
mac-массив, 1 поток, детерминированно). См. tests/test_npu_emu.py.

Отличия от PyTorch: нет RoPE (позиции — синусоидальные), нет GQA,
индексы эмбеддингов подаются списком (один токен-ряд за раз).
"""

import json
import math

from . import atomnet as _ext
from .array import AtomArray
from .nn import Module, Linear, Embedding, RMSNorm, SiLU
from . import functional as F


def causal_mask(T, neg=-1e9):
    return _ext.tensor([neg if j > i else 0.0 for i in range(T) for j in range(T)], [T, T])


def sinusoidal_positions(T, d):
    """Синусоидальные позиции как AtomArray (без numpy)."""
    data = [0.0] * (T * d)
    for t in range(T):
        for i in range(0, d, 2):
            div = math.exp(-math.log(10000.0) * i / d)
            data[t * d + i] = math.sin(t * div)
            if i + 1 < d:
                data[t * d + i + 1] = math.cos(t * div)
    return AtomArray(data, [T, d])


class CausalSelfAttention(Module):
    def __init__(self, d_model):
        super().__init__()
        self.d = d_model
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

    def forward(self, x):
        T = x.shape[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        scale = 1.0 / math.sqrt(self.d)
        scores = q.matmul(k.t()).mul(scale).add(causal_mask(T))
        att = scores.softmax()
        return self.wo(att.matmul(v))


class _MLP(Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.up = Linear(d_model, d_model * mult)
        self.act = SiLU()
        self.down = Linear(d_model * mult, d_model)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class Block(Module):
    def __init__(self, d_model, mult=4, moe=False):
        super().__init__()
        self.ln1 = RMSNorm([d_model])
        self.attn = CausalSelfAttention(d_model)
        self.ln2 = RMSNorm([d_model])
        if moe:
            from .nn import DeepSeekMoE
            self.mlp = DeepSeekMoE(d_model, n_experts=8, top_k=2, shared_experts=1)
        else:
            self.mlp = _MLP(d_model, mult)

    def forward(self, x):
        x = x.add(self.attn(self.ln1(x)))
        x = x.add(self.mlp(self.ln2(x)))
        return x


class _BlockList(Module):
    def __init__(self, n, d, moe=False):
        super().__init__()
        for i in range(n):
            self.add_module(str(i), Block(d, moe=moe))

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class MiniTransformer(Module):
    def __init__(self, vocab, d_model=32, blocks=2, max_len=64, seed=None, moe=False):
        super().__init__()
        self.vocab = vocab
        self.d = d_model
        self.max_len = max_len
        if seed is not None:
            _ext.manual_seed(seed)
        self.emb = Embedding(vocab, d_model)
        pos_t = _ext.tensor(
            sinusoidal_positions(max_len, d_model).reshape(-1).tolist(),
            [max_len, d_model])
        self.register_buffer("pos", pos_t)
        self.blocks = _BlockList(blocks, d_model, moe)
        self.lnf = RMSNorm([d_model])
        self.head = Linear(d_model, vocab)

    def forward(self, tokens):
        """tokens: list[int] (len T) -> logits (T, vocab)."""
        h = self.emb(tokens)
        T = h.shape[0]
        h = h.add(_ext.rows_slice(self.pos, 0, T))
        h = self.blocks(h)
        h = self.lnf(h)
        return self.head(h)


# ───────────────────── state_dict на JSON (точность f32 сохраняется) ──

def save_state_dict(module, path):
    with open(path, "w") as f:
        json.dump(
            {k: [list(t.shape), t.tolist()] for k, t in module.state_dict().items()},
            f)


def load_state_dict(module, path):
    with open(path) as f:
        sd = json.load(f)
    return module.load_state_dict(
        {k: _ext.tensor(v[1], list(v[0])) for k, v in sd.items()})


# ═══════════════════════════════════════════════════════════════════
# AtomGPTTrainer — учебный стек: LR-шедулер (cosine+warmup),
# градиентная аккумуляция, checkpoints, eval, save/load, data-stream API.
# ═══════════════════════════════════════════════════════════════════

def lr_at(step, warmup_steps, max_steps, max_lr, min_lr, schedule="cosine"):
    """LR на шаге `step` (cosine + warmup по nanoGPT policy)."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        decay_ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + decay_ratio * (max_lr - min_lr)
    if schedule == "linear":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * (1.0 - progress)
    return max_lr


class AtomGPTTrainer:
    """Полный учебный стек: data stream -> batches -> optimizer (Adam/AdamW/SGD)
    с LR-шедулером, checkpoints, eval и generate — вокруг MiniTransformer."""

    def __init__(self, vocab=12, opt_name="adamw", batch=4, lr=0.02, min_lr=None,
                 warmup_steps=20, max_steps=1000, schedule="cosine", seed=0, moe=False):
        from . import atomnet as _a
        from .nn import DeepSeekMoE  # noqa: F401 (используется для архипастропогных меж)
        if seed is not None:
            _a.manual_seed(seed)
        self.model = MiniTransformer(vocab=vocab, d_model=32, blocks=2,
                                     max_len=64, seed=seed, moe=moe)
        self.vocab = vocab
        self.batch = batch
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.max_lr = lr
        self.min_lr = min_lr if min_lr is not None else lr * 0.1
        self.schedule = schedule
        self.step_idx = 0
        self.history = []
        from . import optimizers as _opt
        self.opt = _opt.AdamW(self.model.parameters(), lr=self.min_lr) \
            if opt_name == "adamw" else _opt.Adam(self.model.parameters(), lr=self.min_lr) \
            if opt_name == "adam" else _opt.SGD(self.model.parameters(), lr=self.min_lr)

    def train_batch(self, seqs):
        """Один батч: grads обнуляются до, accumulate по seqs, затем opt.step."""
        losses = []
        lr = lr_at(self.step_idx, self.warmup_steps, self.max_steps,
                   self.max_lr, self.min_lr, self.schedule)
        self.opt.set_lr(lr)
        self.opt.zero_grad()
        for seq in seqs:
            self.model.train()
            lg = self.model(seq)
            x = _ext.rows_slice(lg, 0, len(seq) - 1)
            loss = F.cross_entropy(x, seq[1:])
            loss.backward()
            losses.append(loss.item())
        self.opt.step()
        return losses

    def evaluate(self, seqs):
        """Без grad — eval loss по наборам."""
        self.model.eval()
        loss = 0.0
        with F.no_grad():
            for seq in seqs:
                lg = self.model(seq)
                x = _ext.rows_slice(lg, 0, len(seq) - 1)
                loss += float(F.cross_entropy(x, seq[1:]).item())
        return loss / max(1, len(seqs))

    def fit(self, sequences, epochs=6):
        """Полный цикл: эпохи по последовательностям (datacenter-sideчится)."""
        self.model.train()
        losses = []
        for epoch in range(epochs):
            shuffled = list(sequences)
            import random
            random.shuffle(shuffled)
            batch_losses = []
            for i in range(0, len(shuffled), self.batch):
                batch = shuffled[i:i + self.batch]
                batch_losses.extend(self.train_batch(batch))
            losses.append(float(sum(batch_losses) / max(1, len(batch_losses))))
            self.step_idx += len(batch_losses)
            self.history.append(losses[-1])
        return losses

    # ── generate: argmax greedy ──
    def generate(self, prefix, max_new=16):
        self.model.eval()
        seq = list(prefix)
        with F.no_grad():
            for _ in range(max_new):
                lg = self.model(seq)
                vals = lg.tolist()
                T = len(seq)
                last = vals[(T - 1) * self.vocab:T * self.vocab]
                nxt = max(range(self.vocab), key=lambda i: last[i])
                seq.append(nxt)
        return seq

    # ── checkpoint ──
    def save(self, path):
        save_state_dict(self.model, path)
        import json
        with open(path + ".meta.json", "w") as fh:
            json.dump({"step": self.step_idx, "history": self.history}, fh)

    def load(self, path):
        load_state_dict(self.model, path)
        import os, json
        if os.path.exists(path + ".meta.json"):
            with open(path + ".meta.json") as fh:
                d = json.load(fh)
            self.step_idx = d.get("step", 0)


# ════════════════════════════════════════════════════════════
# Airo-Nano v0.0.1 — первый.shipable LLM (Phi-стиль обучения)
# ════════════════════════════════════════════════════════════

def synth_textbook_corpus(n_samples=200, max_len=24, seed=0):
    """Синтетический 'textbook' корпус: простые примеры по curriculum
    (обобщение-объекты детям выражения) для phi-подхода к созданию корпуса.
    Формат Phi-style: "1+1=2" текстовые задачавки, сложенной возрастающей.”
    """
    import random
    rng = random.Random(seed)
    ops_data = []
    for _ in range(n_samples):
        a, b = rng.randrange(10), rng.randrange(10)
        s = a + b
        ops_data.append([a, b, 10, s // 100, (s // 10) % 10, s % 10,
                         rng.randrange(8), rng.randrange(4)])
    return ops_data


def train_aiiro_nano(n_epochs=6, seed=0):
    """Эnd-to-end: синтез текста → обучение → eval → сохранение."""
    tr = AtomGPTTrainer(vocab=12, opt_name="adamw", batch=4, lr=0.02,
                        warmup_steps=10, max_steps=400, seed=seed)
    data = synth_textbook_corpus(n_samples=100, seed=seed)
    losses = tr.fit(data, epochs=n_epochs)
    ev = tr.evaluate(data[:8])
    gen = tr.generate([1, 7, 10, 0, 0], 6)
    tr.save("/tmp/aiiro_nano.bin")
    print("aiiro-nano train losses:", [round(l, 3) for l in losses[:6]])
    print("eval loss:", round(ev, 3), "| generate:", gen)
    return losses
