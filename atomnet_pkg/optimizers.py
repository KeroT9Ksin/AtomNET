"""Оптимизаторы — по исходникам torch/optim/{optimizer,adam,sgd,adamw}.py.

Точные формулы шага (single-tensor путь, математика совпадает с multi-tensor):

Adam (с coupled decay по умолчанию):
    m ← lerp(g, 1-β1)          # m*(β1) + g*(1-β1)
    v ← β2*v + (1-β2)*g²       # addcmul
    t ← t+1
    step_size ← lr / (1-β1^t)
    denom ← √v / √(1-β2^t) + ε
    θ ← θ - step_size * m / denom
    перед этим при wd≠0: θ ← θ - lr*wd*θ   (coupled, как torch Adam)

AdamW (decoupled_weight_decay=True):
    θ ← θ * (1 - lr*wd)  — decay ПЕРЕД шагом Adam.

SGD:
    d_p ← g (+ wd*θ при wd≠0)
    momentum>0:
        buf ← (first step: d_p; иначе momentum*buf + (1-dampening)*d_p)
        nesterov ? d_p ← d_p + momentum*buf : d_p ← buf
    θ ← θ - lr*d_p

state_dict: {'state': {param_index: state}, 'param_groups': [...]} —
  инты по сквозной нумерации параметров, как id() в torch; load
  сопоставляет по порядку (zip), без дополнительной верификации.
"""
import copy

from .atomnet import Tensor


def _flat_add(a, b, alpha):
    return [x + alpha * y for x, y in zip(a, b)]


class Optimizer:
    def __init__(self, params, defaults):
        self.defaults = copy.deepcopy(defaults)
        self.param_groups = []
        if isinstance(params, (list, tuple)):
            # список Tensor или список dict'ов param_group
            first = params[0] if params else None
            if isinstance(first, dict):
                for g in params:
                    self.add_param_group(g)
            else:
                self.add_param_group({"params": list(params)})
        else:
            self.add_param_group({"params": list(params)})

    def add_param_group(self, group):
        params = group["params"]
        if not isinstance(params, (list, tuple)):
            raise TypeError("params должен быть списком Tensor или param_group")
        merged = dict(self.defaults)
        merged.update({k: v for k, v in group.items() if k != "params"})
        merged["params"] = list(params)
        if len(merged["params"]) == 0:
            raise ValueError(" список параметров param_group пуст")
        self.param_groups.append(merged)

    # ── отключённая цепочка состояний: мировой индекс параметра ──
    def _ordered_params(self):
        out = []
        for g in self.param_groups:
            out.extend(g["params"])
        return out

    def _state_for(self, index, template):
        if index not in self.state:
            self.state[index] = template()
        return self.state[index]

    def state_dict(self):
        return {
            "state": copy.deepcopy(self.state),
            "param_groups": [
                {k: v for k, v in g.items() if k != "params"} | {"params": list(range(len(self._ordered_params())))}
                for g in self.param_groups
            ],
        }

    def load_state_dict(self, sd):
        self.state = copy.deepcopy(sd["state"])
        # zip по порядку, как torch (без верификации соответствия)
        assert len(sd["param_groups"]) == len(self.param_groups)
        for gm, gs in zip(self.param_groups, sd["param_groups"]):
            for key in gs:
                if key != "params":
                    gm[key] = gs[key]

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g["params"]:
                p.zero_grad(set_to_none=set_to_none)

    def step(self):
        raise NotImplementedError

    def set_lr(self, lr):
        for g in self.param_groups:
            g["lr"] = lr


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.state = {}
        super().__init__(params, {
            "lr": lr, "momentum": momentum, "dampening": dampening,
            "weight_decay": weight_decay, "nesterov": nesterov,
        })
        if nesterov and momentum == 0:
            raise ValueError("Nesterov momentum требует momentum > 0")

    def step(self):
        idx = 0
        for g in self.param_groups:
            lr = g["lr"]
            wd = g["weight_decay"]
            momentum = g["momentum"]
            dampening = g["dampening"]
            nesterov = g["nesterov"]
            for p in g["params"]:
                if p.grad is None:
                    idx += 1
                    continue
                st = self._state_for(idx, lambda: {"step": 0, "momentum_buffer": None})
                grad = p.grad.tolist()
                d_p = list(grad)
                if wd != 0.0:
                    w = p.tolist()
                    d_p = [dp + wd * wi for dp, wi in zip(d_p, w)]
                if momentum != 0:
                    buf = st["momentum_buffer"]
                    if buf is None:
                        buf = list(d_p)
                    else:
                        buf = [momentum * bi + (1.0 - dampening) * dp for bi, dp in zip(buf, d_p)]
                    st["momentum_buffer"] = buf
                    d_p = [dp + momentum * bi for dp, bi in zip(d_p, buf)] if nesterov else buf
                d = p.tolist()
                new = [di - lr * dpi for di, dpi in zip(d, d_p)]
                p.copy_(Tensor(new, list(p.shape)))
                st["step"] += 1
                idx += 1


class _AdamBase(Optimizer):
    """Общая логика Adam / AdamW (флаг decoupled различает поведение)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 amsgrad=False, decoupled_weight_decay=False):
        self.state = {}
        super().__init__(params, {
            "lr": lr, "betas": betas, "eps": eps,
            "weight_decay": weight_decay, "amsgrad": amsgrad,
            "decoupled_weight_decay": decoupled_weight_decay,
        })

    def step(self):
        idx = 0
        for g in self.param_groups:
            lr = g["lr"]
            beta1, beta2 = g["betas"]
            eps = g["eps"]
            wd = g["weight_decay"]
            decoupled = g["decoupled_weight_decay"]
            amsgrad = g["amsgrad"]
            for p in g["params"]:
                if p.grad is None:
                    idx += 1
                    continue
                st = self._state_for(idx, lambda: {
                    "step": 0,
                    "exp_avg": [0.0] * p.numel,
                    "exp_avg_sq": [0.0] * p.numel,
                    "max_exp_avg_sq": [0.0] * p.numel if amsgrad else None,
                })
                grad = p.grad.tolist()
                theta = p.tolist()
                # decay
                if wd != 0.0:
                    if decoupled:
                        theta = [t * (1.0 - lr * wd) for t in theta]
                    else:
                        theta = [t - lr * wd * t for t in theta]
                # moments
                m = [mi + (1.0 - beta1) * (gi - mi) for mi, gi in zip(st["exp_avg"], grad)]
                v = [vi * beta2 + (1.0 - beta2) * gi * gi for vi, gi in zip(st["exp_avg_sq"], grad)]
                st["exp_avg"] = m
                st["exp_avg_sq"] = v
                if amsgrad:
                    mx = st["max_exp_avg_sq"]
                    st["max_exp_avg_sq"] = [max(a, b) for a, b in zip(mx, v)]
                    veff = st["max_exp_avg_sq"]
                else:
                    veff = v
                t = st["step"] + 1
                st["step"] = t
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                step_size = lr / bc1
                bc2_sqrt = bc2 ** 0.5
                new = []
                for th, mi, vi in zip(theta, m, veff):
                    denom = (vi ** 0.5) / bc2_sqrt + eps
                    new.append(th - step_size * mi / denom)
                p.copy_(Tensor(new, list(p.shape)))
                idx += 1


class Adam(_AdamBase):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                         decoupled_weight_decay=False)


class AdamW(_AdamBase):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad,
                         decoupled_weight_decay=True)
