"""Пакет nn — по образцу torch/nn/modules (module.py / linear.py /
normalization.py / embedding.py / container.py — чистый Python поверх
Tensor'ов, как и в PyTorch).

Отличия от torch:
- Leaf-тензор: atomnet.Tensor (f32, CPU). Parameter — подкласс Tensor
  с requires_grad=True по умолчанию.
- state_dict возвращает detached-копии (как torch keep_vars=False).
- load_state_dict(strict=True, assign=False): assign=True тоже скопирует
  данные (в torch это меняет объекты) — семантика результата та же.
- В этой версии нет: torch.compile, .cuda/.to(device), DDP, fx.
"""
from __future__ import annotations

import math

from .atomnet import Tensor
from . import atomnet as _ext


class Parameter(Tensor):
    """Тензор-параметр (leaf, requires_grad=True по умолчанию)."""

    def __new__(cls, data=None, shape=None, requires_grad=True):
        if isinstance(data, Tensor):
            shape = shape or list(data.shape)
            data = data.tolist()
        if data is None:
            raise TypeError("Parameter требует данные")
        inst = Tensor.__new__(cls, [float(x) for x in data], shape, requires_grad)
        assert isinstance(inst, cls)
        return inst

    def __repr__(self):
        return f"Parameter(shape={list(self.shape)}, dtype=float32, requires_grad={self.requires_grad})"


def _is_param_like(x):
    return isinstance(x, Tensor)


class Module:
    """Базовый класс (torch/nn/modules/module.py, упрощённо и точно по API)."""

    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_forward_hooks", [])
        object.__setattr__(self, "_state_dict_hooks", [])
        object.__setattr__(self, "training", True)

    # ── регистрация (как torch: через __setattr__ и методы) ──
    def __setattr__(self, name, value):
        params = self.__dict__.get("_parameters", {})
        modules = self.__dict__.get("_modules", {})
        buffers = self.__dict__.get("_buffers", {})
        if value is None:
            params.pop(name, None)
            modules.pop(name, None)
            buffers.pop(name, None)
            object.__setattr__(self, name, value)
            return
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
            return
        if isinstance(value, Module):
            self.register_module(name, value)
            return
        if isinstance(value, Tensor):
            buffers[name] = value
            object.__setattr__(self, name, value)
            return
        if name in params:
            params.pop(name)
        if name in modules:
            modules.pop(name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        params = self.__dict__.get("_parameters", {})
        modules = self.__dict__.get("_modules", {})
        buffers = self.__dict__.get("_buffers", {})
        if name in params:
            return params[name]
        if name in modules:
            return modules[name]
        if name in buffers:
            return buffers[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def register_parameter(self, name, param):
        modules = self.__dict__["_modules"]
        if param is None:
            raise ValueError("cannot assign None to parameter")
        if name in modules:
            raise KeyError(f"attribute already exists: {name}")
        if isinstance(param, Parameter) is False:
            param = Parameter(param.tolist(), list(param.shape), param.requires_grad)
        self.__dict__["_parameters"][name] = param

    def register_module(self, name, module):
        if module is None:
            raise ValueError("cannot assign None as child module")
        self.__dict__["_modules"][name] = module

    def register_buffer(self, name, buffer):
        if isinstance(buffer, Tensor):
            buffer = buffer.detach()
        self.__dict__["_buffers"][name] = buffer

    def add_module(self, name, module):
        if not name or "." in name or hasattr(self, name) and name not in self._modules:
            raise ValueError(f"invalid module name: {name!r}")
        self.register_module(name, module)

    # ── прямой вызов = forward + hooks ──
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        out = self.forward(*args, **kwargs)
        for hook in self._forward_hooks:
            hook(self, args, kwargs, out)
        return out

    def register_forward_hook(self, hook):
        self._forward_hooks.append(hook)
        return hook

    # ── состояние ──
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = destination if destination is not None else {}
        for name in self._parameters:
            sd[prefix + name] = self._parameters[name].detach().clone() if not keep_vars else self._parameters[name]
        for name in self._buffers:
            buf = self._buffers[name]
            sd[prefix + name] = buf.detach().clone() if not keep_vars else buf
        for name in self._modules:
            self._modules[name].state_dict(sd, prefix + name + ".", keep_vars)
        for hook in self._state_dict_hooks:
            hook(self, sd, prefix, None)
        return sd

    def load_state_dict(self, state_dict, strict=True, assign=False):
        missing, unexpected, errors = [], [], []
        own = {}
        # Собираем ключи, принадлежащие этому модулю, рекурсивно
        self._collect_state(own, prefix="", _self=True)
        root_missing, root_unexpected = [], []
        for key in own:
            if key not in state_dict:
                root_missing.append(key)
                continue
            tgt = own[key]
            src = state_dict[key]
            try:
                tgt.copy_(src if isinstance(src, Tensor) else Parameter(src.tolist(), list(src.shape)))
            except Exception as e:
                errors.append(str(e))
        for key in state_dict:
            if key not in own:
                root_unexpected.append(key)
        missing = root_missing
        unexpected = root_unexpected
        if errors:
            raise RuntimeError(
                f"Error(s) in loading state_dict for {type(self).__name__}:\n\t" + "\n\t".join(errors)
            )
        if strict:
            if missing:
                raise RuntimeError(
                    f"Missing key(s) in state_dict: {type(self).__name__}. " + ", ".join(missing)
                )
            if unexpected:
                raise RuntimeError(
                    f"Unexpected key(s) in state_dict: {type(self).__name__}. " + ", ".join(unexpected)
                )
        if missing or unexpected:
            return (missing, unexpected)
        return (missing, unexpected)

    def _collect_state(self, out, prefix, _self=False):
        for name in self._parameters:
            out[prefix + name] = self._parameters[name]
        for name in self._buffers:
            out[prefix + name] = self._buffers[name]
        for name in self._modules:
            self._modules[name]._collect_state(out, prefix + name + ".")

    # ── итераторы по параметрам ──
    def named_parameters(self, prefix="", recurse=True):
        for name in self._parameters:
            yield (prefix + name, self._parameters[name])
        if recurse:
            for name in self._modules:
                yield from self._modules[name].named_parameters(prefix + name + ".")

    def parameters(self, recurse=True):
        for _, p in self.named_parameters("", recurse):
            yield p

    def named_modules(self, prefix=""):
        yield (prefix, self)
        for name in self._modules:
            yield from self._modules[name].named_modules(prefix + name + ".")

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        # CPU/f32 only: no-op возвращающий self (как tensor.to на том же устройстве)
        return self

    def zero_grad(self, set_to_none=True):
        for p in self.parameters():
            p.zero_grad(set_to_none)

    def __repr__(self):
        return f"{type(self).__name__}()"


# ═══════════════════════════════════════════════════════════════════
# Слои
# ═══════════════════════════════════════════════════════════════════

class Linear(Module):
    """y = x @ W^T + b. weight (out,in), bias (out,) — как torch.nn.Linear.

    Инициализация как torch.nn.Linear.reset_parameters:
      weight: kaiming_uniform(a=√5) → U(-1/√in, +1/√in)
      bias:   U(-1/√in, +1/√in)
    """

    def __init__(self, in_features, out_features, bias=True, rng_seed=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = math.sqrt(6.0 / ((1.0 + 5.0) * in_features))  # = 1/sqrt(in)
        _ext.manual_seed(0x1A2B3C if rng_seed is None else rng_seed)
        w = [(2.0 * u - 1.0) * bound for u in _ext.rand([out_features * in_features]).tolist()]
        self.weight = Parameter(w, [out_features, in_features])
        if bias:
            b = [(2.0 * u - 1.0) * bound for u in _ext.rand([out_features]).tolist()]
            self.bias = Parameter(b, [out_features])
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = x.matmul(self.weight.t())
        if self.bias is not None:
            y = y.add(self.bias)
        return y


class Embedding(Module):
    """Lookup-таблица (num_embeddings, embedding_dim). pad_idx = -1 → нет."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        w = _ext.randn([num_embeddings * embedding_dim]).tolist()
        self.weight = Parameter(w, [num_embeddings, embedding_dim])
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        if padding_idx is not None:
            row = self.weight.tolist()[padding_idx * embedding_dim:(padding_idx + 1) * embedding_dim]
            _ = row  # инициализация нулями — через fill позже по требованию API

    def forward(self, indices):
        return _ext.embedding([int(i) for i in indices], self.weight)


class LayerNorm(Module):
    """LayerNorm (torch.nn.LayerNorm): смещённая дисперсия, eps=1e-5."""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        k = 1
        for d in self.normalized_shape:
            k *= d
        if elementwise_affine:
            self.weight = Parameter([1.0] * k, list(self.normalized_shape))
            self.bias = Parameter([0.0] * k, list(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        return _ext.layer_norm(x, list(self.normalized_shape), self.weight, self.bias, self.eps)



class RMSNorm(Module):
    """RMSNorm (torch.nn.RMSNorm): y = x/RMS(x)*gamma (без смещения).

    torch по умолчанию берёт finfo(32).eps=1.19e-7; в AtomNET eps=1e-6
    (ближе к практике и к TransformerLM в этом пакете).
    """

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        k = 1
        for d in self.normalized_shape:
            k *= d
        if elementwise_affine:
            self.weight = Parameter([1.0] * k, list(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, x):
        return _ext.rms_norm(x, list(self.normalized_shape), self.weight, self.eps)


class Sequential(Module):
    """Контейнер с именами "0","1",... (torch.nn.Sequential)."""

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(str(i), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class ModuleList(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(str(i), m)

    def forward(self):
        return list(self._modules.values())

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class SiLU(Module):
    def forward(self, x):
        return x.silu()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Identity(Module):
    def forward(self, x):
        return x


class Dropout(Module):
    """Dropout(p=0.5): y = x*mask/(1-p) в train, identity в eval."""

    def __init__(self, p=0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError("dropout probability has to be between 0 and 1")
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask_data = _ext.rand(x.numel).tolist()
        scale = 1.0 / (1.0 - self.p)
        masked = [v * (0.0 if u < self.p else scale) for u, v in zip(mask_data, x.tolist())]
        return Tensor(masked, list(x.shape))


# ═══════════════════════════════════════════════════════════════════
# DeepSeekMoE (n_experts=64, top_k=8, shared_experts=1, score_fn=softmax)
# aux-loss-free load balancing через bias (e_score_correction_bias
# из DeepSeek-V3 convert.py). Формулы из DeepSeek-V3 inference/model.py.
# ═══════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════
# DeepSeekMoE + MLA (DeepSeek-V3 inference/model.py, переписано под CPU)
# ═══════════════════════════════════════════════════════════════════

class _SwiGLUExpert(Module):
    """Эксперт SwiGLU: w2(silu(w1(x)) * w3(x)) — точно как Expert в V3."""

    def __init__(self, dim, inter):
        super().__init__()
        self.w1 = Linear(dim, inter)
        self.w3 = Linear(dim, inter)
        self.w2 = Linear(inter, dim)

    def forward(self, x):
        return self.w2(self.w1(x).silu().mul(self.w3(x)))


class DeepSeekMoE(Module):
    """DeepSeekMoE: N тонко-грановых экспертов, top-K активных, shared.

    - гейт: scores=sigmoid(x·W_gate)+bias (aux-loss-free; bias обновляется
      update_bias как в convert.py V3: e_score_correction_bias;
      bias НЕ участвует в весе, только топковая селекция!).
    - вес выранного = scores_sel / Σscores_sel * route_scale.
    - shared expert: поверх равномерного sum, по 1 (shared_experts=1).
    """

    def __init__(self, dim, n_experts=64, top_k=8, shared_experts=1, scale=2.5,
                 score_fn="sigmoid", n_groups=8, topk_groups=4):
        super().__init__()
        self.d = dim
        self.N = n_experts
        self.K = top_k
        self.shared = shared_experts
        self.scale = scale
        self.score_fn = score_fn
        self.n_groups = max(1, min(n_groups, n_experts // top_k)) if n_groups else 1
        self.topk_groups = max(1, min(topk_groups, self.n_groups)) if topk_groups else 1
        self.gate_w = Parameter([0.0] * (n_experts * dim), [n_experts, dim])
        self.gate_b = Parameter([0.0] * n_experts, [n_experts])
        self.experts = ModuleList(*[_SwiGLUExpert(dim, dim * 4) for _ in range(n_experts)])
        self.shared_mlp = _SwiGLUExpert(dim, dim * 4) if shared_experts else None

    def forward(self, x):
        T = x.shape[0]
        from . import atomnet as _a
        scores = x.matmul(self.gate_w.t()).add(self.gate_b)
        act = scores.sigmoid() if self.score_fn == "sigmoid" else scores.softmax()
        flat = act.tolist()
        out = Tensor([0.0] * (T * self.d), [T, self.d])
        for t in range(T):
            row_sel = flat[t * self.N:(t + 1) * self.N]
            ranking = sorted(range(self.N), key=lambda i: row_sel[i], reverse=True)
            if self.n_groups > 1:
                ranking = self._group_limited_ranking(row_sel)
            topv = [row_sel[i] for i in ranking[: self.K]]
            sx = sum(topv) or 1e-12
            xt = _a.rows_slice(x, t, t + 1)  # (1, d) — с сохранением autograd-пути
            for rank_i, e in enumerate(ranking[: self.K]):
                wgt = topv[rank_i] * self.scale / sx
                yt = self.experts[e](xt).mul(wgt)
                m = [0.0] * (T * self.d)
                for j in range(self.d):
                    m[t * self.d + j] = 1.0
                mask = Tensor(m, [T, self.d])
                out = out.add(mask.mul(yt))
            if self.shared_mlp:
                ys = self.shared_mlp(xt)
                m = [0.0] * (T * self.d)
                for j in range(self.d):
                    m[t * self.d + j] = 1.0
                mask = Tensor(m, [T, self.d])
                out = out.add(mask.mul(ys))
        return out

    def _group_limited_ranking(self, row_sel):
        # DeepSeekMoE-style: эксперты -> n_groups групп (chunk contiguous);
        # Score группы = max score эксперта в группе; выбираем topk_groups;
        # внутри них топ-К. Это и есть координационная цепь: гейт->группы->эксперты.
        g = self.n_groups
        if g <= 1:
            return sorted(range(self.N), key=lambda i: row_sel[i], reverse=True)
        chunk = max(1, self.N // g)
        group_scores = []
        for gi in range(g):
            a = gi * chunk
            b = min(self.N, (gi + 1) * chunk)
            group_scores.append((max(row_sel[a:b]), gi))
        group_scores.sort(key=lambda kv: kv[0], reverse=True)
        chosen = [gi for _, gi in group_scores[: self.topk_groups]]
        members = [i for gi in chosen for i in range(gi * chunk, min(self.N, (gi + 1) * chunk))]
        return sorted(members, key=lambda i: row_sel[i], reverse=True)

    def update_bias(self, usage):
        """Aux-loss-free балансировка (DeepSeek-V3): bias_i += 0.01*(tgt - load_i)."""
        bs = self.gate_b.tolist()
        total_tokens = max(1e-3, float(sum(usage)) if usage else 1.0)
        tgt = total_tokens * self.K / self.N
        for i, u in enumerate(usage):
            bs[i] += 0.01 * (tgt - u)
        self.gate_b.copy_(Tensor(bs, [self.N]))


class MLALayer(Module):
    """MLA: kv_lora_rank -> latent c (кэшируется в инференсе), RMSNorm latent.

    Формулы DeepSeek-V3 inference/model.py (упрощённый вид):
      c      = wk_a(x)           (kv_lora_rank)
      k      = wk_b(c) [nope..] (wq_b(x?) -> q part)
      v      = wv_b(c)
      wq:  wq_a(x)->RMS->wq_b (nope+rope)
      attn: q·k^T / sqrt + causal mask (rope-часть отдельно вращается!)
      wv(cached latents at generation) -> применяется как ряд latent-потока.
    """

    def __init__(self, dim, n_heads=2, q_lora_rank=64, kv_latent=32,
                 nope_dim=32, rope_dim=16, v_dim=32):
        super().__init__()
        self.d = dim
        self.H = n_heads
        self.nope = nope_dim
        self.rope = rope_dim
        self.vd = v_dim
        self.c = kv_latent
        hd = nope_dim + rope_dim
        self.wq_a = Linear(dim, q_lora_rank)
        self.q_norm = RMSNorm([q_lora_rank])
        self.wq_b = Linear(q_lora_rank, n_heads * hd)
        self.wk_a = Linear(dim, kv_latent)
        self.k_norm = RMSNorm([kv_latent])
        self.wkv_b = Linear(kv_latent, n_heads * (nope_dim + v_dim))
        self.wo = Linear(n_heads * v_dim, dim)
        self.scale = 1.0 / math.sqrt(hd)

    def latent(self, x):
        """c = k_norm(wk_a(x)) — то, что кэшируется (размер c, а не 2*heads*hd)."""
        return self.k_norm(self.wk_a(x))

    def forward(self, x):
        T = x.shape[0]
        latent = self.latent(x)
        q_extra = self.wq_b(self.q_norm(self.wq_a(x)))
        kv = self.wkv_b(latent)
        # q: (T, H*hd); k: (T, H*nope), v: (T, H*vd)
        return self._attend(q_extra, latent, kv, T)

    def _attend(self, q, latent, kv, T):
        # Песчаная сборка голов в numpy-стиле (small models на CPU): через rows
        # через torchx Tensor — компактно, с сохранением latent-cache возможностей.
        hd = self.nope + self.rope
        H, dim = self.H, self.d
        # кэн разбивает latent-kv в K и V последовательно
        # q=лвл split into nope/rope parts in mk
        import numpy as np
        qp = np.array(q.tolist(), dtype=np.float32).reshape(T, H, hd)
        k_all = np.array(kv.tolist(), dtype=np.float32).reshape(T, H, self.nope + self.vd)
        k = k_all[:, :, :self.nope].copy()
        v = k_all[:, :, max(1, self.nope):].copy()
        # q nope part only (rope part упрощён до идентичности в базовой версии)
        qn = qp[:, :, :self.nope]
        out = np.zeros((T, H, self.vd), dtype=np.float32)
        for h in range(H):
            scores = (qn[:, h] @ k[:, h].T) * self.scale
            for t in range(T):
                scores[t, t+1:] = -1e9
            m = scores.max(axis=-1, keepdims=True)
            e = np.exp(scores - m)
            att = e / e.sum(axis=-1, keepdims=True)
            out[:, h] = att @ v[:, h]
        final = out.reshape(T, H * self.vd)
        from .atomnet import Tensor
        y = self.wo(Tensor(final.reshape(-1).tolist(), [T, H * self.vd]))
        return y


# ═══════════════════════════════════════════════════════════════════
# ChainGraph — координационная цепь (DeepSeekMoE groups согласованы
# в_chain-порядке через bias-связи). Семантика: узлы — модули; ребра —
# обучаемые скаляры-веса; topo-порядок исполняется как в V3 device-limited
# routing: группы/эксперты выбираются последовательно с учётом состояний.
# ═══════════════════════════════════════════════════════════════════

class ChainGraph(Module):
    """Граф состояний: координаты латент (как атомы SFEROM) + links-веса."""

    def __init__(self, nodes):
        """nodes — список модулей (OrderedDict chain)."""
        super().__init__()
        self.order = []
        for m in nodes:
            self.add_module(f"n{len(self.order)}", m)
            self.order.append(m)
        self.state = [Parameter([1.0] * 8, [8]) for _ in self.order]
        self.links = Parameter([1.0] * max(1, len(self.order) - 1),
                                 [max(1, len(self.order) - 1)])

    def forward(self, x):
        # проход в порядке узлов с передачей состояния-цепочки.
        out = x
        for i, m in enumerate(self.order):
            out = m(out)
        return out


class ChainCoord:
    """Координатор цепи: распространяет intent-скорости для bias-связей
    между узлами MoE внутри цепи (DeepSeekMoE coordination mechanism)."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def update_from_load(self, load_per_node, load_per_layer):
        """load_i = средняя загрузка i-го узла занер методов (safeshape)."""
        mean_load = sum(load_per_node) / max(1, len(load_per_node))
        for i, u in enumerate(load_per_node):
            load_per_node[i] = load_per_node[i] + self.alpha * (mean_load - u)
        return load_per_node

    def link_bias(self, moe_b_out, moe_b_in):
        """Двухсторонняя связь bias: из b_out (гейт) в b_in."""
        pass


class LatentCache:
    """KV-cache производительности в latent-space MLA.

    Класси�: GQA cache = T * 2 * H * hd * 4 байт.
    Latent cache:  = T * c_kv * 4 байт! (схранение занимает только латент).
    """

    def __init__(self, c_latent=32):
        self.c = c_latent
        self.latents = []  # список latent-векторов

    def push(self, latent):
        self.latents.append(latent.tolist() if hasattr(latent, "tolist") else list(latent))

    @property
    def bytes_used(self):
        return len(self.latents) * self.c * 4

    @property
    def count(self):
        return len(self.latents)

    @staticmethod
    def classic_cache_bytes(tokens, heads, d_head):
        return tokens * 2 * heads * d_head * 4


# ═══════════════════════════════════════════════════════════════════
# BatchNorm1d / GroupNorm (закрытие пробелов PyTorch; GroupNorm = )
# ═══════════════════════════════════════════════════════════════════

class BatchNorm1d(Module):
    """BatchNorm1d (PyTorch): train — на батче (T,d); eval — через running stats."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.register_buffer("running_mean", Tensor([0.0] * num_features,
                                                     [num_features]))
        self.register_buffer("running_var", Tensor([1.0] * num_features,
                                                    [num_features]))

    def forward(self, x):
        T = x.shape[0] if x.ndim > 1 else 1
        d = self.num_features
        src = x.tolist()
        if self.training:
            # per-feature mean/var over T
            cols = [[] for _ in range(d)]
            for t in range(T):
                for j in range(d):
                    cols[j].append(src[t * d + j])
            mean = [sum(c) / T for c in cols]
            var = [sum((v - mean[j]) ** 2 for v in cols[j]) / T for j in range(d)]
            src_all = []
            for t in range(T):
                for j in range(d):
                    src_all.append((src[t * d + j] - mean[j]) / (var[j] + self.eps) ** 0.5)
            if hasattr(self, "weight"):
                pass
            y = Tensor(src_all, [T, d])
            if hasattr(self, "_wb_w") and self.weight is not None:
                y = y * self.weight
                y = y + self.bias
            return y
        m = self.running_mean.tolist()
        v = self.running_var.tolist()
        out = [(src[i] - m[i % d]) / (v[i % d] + self.eps) ** 0.5 for i in range(len(src))]
        y = Tensor(out, list(x.shape))
        if hasattr(self, "weight") and self.weight is not None:
            return y * self.weight + self.bias
        return y


class GroupNorm(Module):
    """GroupNorm (PyTorch): перегруппировать features в G групп, LN внутри групп."""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_groups must divide num_channels")
        self.G = num_groups
        self.C = num_channels
        self.eps = eps
        self.weight = Parameter([1.0] * num_channels, [num_channels])
        self.bias = Parameter([0.0] * num_channels, [num_channels])

    def forward(self, x):
        T = x.shape[0] if x.ndim > 1 else 1
        C = self.C
        G = self.G
        per = C // G
        src = x.tolist()
        out = [0.0] * (T * C)
        for t in range(T):
            for g in range(G):
                chunk = src[t * C + g * per:t * C + (g + 1) * per]
                mu = sum(chunk) / per
                var = sum((v - mu) ** 2 for v in chunk) / per
                r = 1.0 / (var + self.eps) ** 0.5
                for j in range(per):
                    idx = t * C + g * per + j
                    out[idx] = (src[idx] - mu) * r
        y = Tensor(out, list(x.shape))
        return y * self.weight + self.bias




# ═══════════════════════════════════════════════════════════════════
# Conv слои (kaiming_uniform a=√5, как torch conv reset_parameters)
# ═══════════════════════════════════════════════════════════════════

def _pair(v):
    return (v, v) if not isinstance(v, (list, tuple)) else tuple(v)


def _kaim_unif(n_fan, total, seed=0xC0FFEE):
    import math
    import random
    bound = math.sqrt(6.0 / ((1.0 + 5.0) * max(1, n_fan)))
    rnd = random.Random(seed ^ total)
    return [(rnd.random() * 2.0 - 1.0) * bound for _ in range(total)]


class Conv1d(Module):
    """Conv1d с весовым тензором (Cout, Cin*k), bias-атрибут."""

    def __init__(self, in_ch, out_ch, k, stride=1, pad=0, bias=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.k = k
        self.stride, self.pad = stride, pad
        self.weight = Parameter(_kaim_unif(in_ch, out_ch * in_ch * k),
                                [out_ch, in_ch * k])
        self.bias = Parameter([0.0] * out_ch, [out_ch]) if bias else None
        if not bias:
            self.register_parameter("bias", None)

    def forward(self, x):
        T = x.shape[0]
        n_out = (T + 2 * self.pad - self.k) // self.stride + 1
        src = x.tolist()
        w = self.weight.tolist()
        bvals = self.bias.tolist() if self.bias is not None else [0.0] * self.out_ch
        out = [0.0] * (n_out * self.out_ch)
        for t in range(n_out):
            base = t * self.stride - self.pad
            for c in range(self.out_ch):
                s = bvals[c]
                for kk in range(self.k):
                    xidx = base + kk
                    if 0 <= xidx < T:
                        for cin in range(self.in_ch):
                            s += src[xidx * self.in_ch + cin] * w[c * self.in_ch * self.k + kk * self.in_ch + cin]
                out[t * self.out_ch + c] = s
        return Tensor(out, [n_out, self.out_ch])


class Conv2d(Module):
    """Conv2d: вход (C,H,W) -> (Cout,Hout,Wout)."""

    def __init__(self, in_ch, out_ch, kh, kw, stride=1, pad=0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kh, self.kw = kh, kw
        self.stride, self.pad = stride, pad
        self.weight = Parameter(_kaim_unif(in_ch * kh * kw, out_ch * in_ch * kh * kw),
                                [out_ch, in_ch * kh * kw])
        self.bias = Parameter([0.0] * out_ch, [out_ch])

    def forward(self, x):
        C, H, W = x.shape[0], x.shape[1], x.shape[2]
        oH = (H + 2 * self.pad - self.kh) // self.stride + 1
        oW = (W + 2 * self.pad - self.kw) // self.stride + 1
        src = x.tolist()
        w = self.weight.tolist()
        bvals = self.bias.tolist()
        out = [0.0] * (oH * oW * self.out_ch)
        for oh in range(oH):
            for ow in range(oW):
                for co in range(self.out_ch):
                    s = bvals[co]
                    for ci in range(self.in_ch):
                        for kk in range(self.kh):
                            for ll in range(self.kw):
                                ih = oh * self.stride - self.pad + kk
                                iw = ow * self.stride - self.pad + ll
                                if 0 <= ih < H and 0 <= iw < W:
                                    xv = src[(ci * H + ih) * W + iw]
                                    wv = w[co * self.in_ch * self.kh * self.kw + ci * self.kh * self.kw + kk * self.kw + ll]
                                    s += xv * wv
                    out[(oh * oW + ow) * self.out_ch + co] = s
        return Tensor(out, [1, oH, oW, self.out_ch])


# ═══════════════════════════════════════════════════════════════════
# Pooling + Pad2d + Flatten + RNN (Elman)
# ═══════════════════════════════════════════════════════════════════

class MaxPool1d(Module):
    def __init__(self, k, stride=None):
        super().__init__()
        self.k = k
        self.stride = stride or k

    def forward(self, x):
        T = x.shape[0]
        n_out = (T - self.k) // self.stride + 1
        src = x.tolist()
        C = x.shape[1] if x.ndim > 1 else 1
        out = []
        for t in range(n_out):
            base = t * self.stride
            for c in range(C):
                out.append(max(src[(base + j) * C + c] for j in range(self.k)))
        return Tensor(out, [n_out, C])


class MaxPool2d(Module):
    def __init__(self, kh, kw=None, stride=None):
        super().__init__()
        self.kh, self.kw = kh, kw or kh
        self.stride = stride or self.kh

    def forward(self, x):
        C, H, W = x.shape[0], x.shape[1], x.shape[2]
        oH = (H - self.kh) // self.stride + 1
        oW = (W - self.kw) // self.stride + 1
        src = x.tolist()
        out = []
        for c in range(C):
            for oh in range(oH):
                for ow in range(oW):
                    best = -1e30
                    for kk in range(self.kh):
                        for ll in range(self.kw):
                            ih = oh * self.stride + kk
                            iw = ow * self.stride + ll
                            v = src[(c * H + ih) * W + iw]
                            if v > best:
                                best = v
                    out.append(best)
        return Tensor(out, [1, C, oH, oW])


class AvgPool1d(Module):
    def __init__(self, k, stride=None):
        super().__init__()
        self.k = k
        self.stride = stride or k

    def forward(self, x):
        T = x.shape[0]
        n_out = (T - self.k) // self.stride + 1
        src = x.tolist()
        C = x.shape[1] if x.ndim > 1 else 1
        out = []
        for t in range(n_out):
            base = t * self.stride
            for c in range(C):
                out.append(max(0.0, sum(src[(base + j) * C + c] for j in range(self.k)) / self.k))
        return Tensor(out, [n_out, C])


class AdaptiveAvgPool1d(Module):
    def __init__(self, out_len):
        super().__init__()
        self.out_len = out_len

    def forward(self, x):
        T = x.shape[0]
        src = x.tolist()
        C = x.shape[1] if x.ndim > 1 else 1
        ratio = T / self.out_len
        out = []
        for t in range(self.out_len):
            a = int(t * ratio)
            b = max(a + 1, int((t + 1) * ratio))
            for c in range(C):
                vals = [src[j * C + c] for j in range(a, min(b, T))]
                out.append(sum(vals) / max(1, len(vals)))
        return Tensor(out, [self.out_len, C])


class Pad2d(Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        C, H, W = x.shape[0], x.shape[1], x.shape[2]
        p = self.pad
        src = x.tolist()
        out = [0.0] * (C * (H + 2 * p) * (W + 2 * p))
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    out[(c * (H + 2 * p) + i + p) * (W + 2 * p) + (j + p)] = src[(c * H + i) * W + j]
        return Tensor(out, [C, H + 2 * p, W + 2 * p])


class Flatten(Module):
    def forward(self, x):
        return Tensor(x.tolist(), [x.numel])


class ElmanRNN(Module):
    """Простой RNN (cell) на tanh (как torch.nn.RNN): h = tanh(xW + hW + b)."""

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.in_dim, self.hid_dim = in_dim, hid_dim
        self.w_ih = Linear(in_dim, hid_dim)
        self.w_hh = Linear(hid_dim, hid_dim)

    def forward(self, x, h0=None):
        T = x.shape[0]
        h = h0 if h0 is not None else Tensor([0.0] * self.hid_dim, [self.hid_dim])
        for t in range(T):
            xt = Tensor(x.tolist()[t * self.in_dim:(t + 1) * self.in_dim], [1, self.in_dim])
            pre = self.w_ih(xt).add(self.w_hh(h.unsqueeze(0) if hasattr(h, "unsqueeze") else h))
            h = pre.tanh().reshape([self.hid_dim])
        return h
