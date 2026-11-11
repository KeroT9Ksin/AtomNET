"""Functional API — по форме torch/nn/functional.py (точечно, для ядра phi):

cross_entropy: log_softmax + nll fused, reduction mean/sum/none,
ignore_index=-100, class weight, label_smoothing — как в PyTorch.
Только class-indices target (soft target не поддерживается в чистом ядре
Tensor; вероятностный target тоже допустим в torch, но не здесь).
"""
from . import atomnet as _ext


def cross_entropy(input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    return _ext.cross_entropy(
        input,
        [int(t) for t in target],
        weight=[float(w) for w in weight] if weight is not None else None,
        ignore_index=int(ignore_index),
        reduction=reduction,
        label_smoothing=float(label_smoothing),
    )


def linear(input, weight, bias=None):
    return _ext.linear(input, weight, bias)


def embedding(indices, weight):
    return _ext.embedding([int(i) for i in indices], weight)


def mse_loss(a, b):
    return _ext.mse_loss(a, b)


def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return _ext.layer_norm(input, list(normalized_shape), weight, bias, eps)


def softmax(x, dim=-1):
    # dim только последнего измерения поддерживается в ядре
    if dim != -1:
        raise ValueError("softmax: только dim=-1 поддерживается")
    return x.softmax()


def gelu(x):
    return x.gelu()


def silu(x):
    return x.silu()


def relu(x):
    return x.relu()


def tanh(x):
    return x.tanh()


def sigmoid(x):
    return x.sigmoid()


def rms_norm(input, normalized_shape, weight=None, eps=1e-6):
    return _ext.rms_norm(input, list(normalized_shape), weight, eps)


class _no_grad_ctx:
    """torch.no_grad: отключает построение графа (eval-проходы)."""

    def __enter__(self):
        self._prev = _ext.is_grad_enabled()
        _ext.set_grad_enabled(False)
        return self

    def __exit__(self, *a):
        _ext.set_grad_enabled(self._prev)
        return False


def no_grad():
    """Контекстный менеджер torch.no_grad (отключает autograd)."""
    return _no_grad_ctx()


def is_grad_enabled():
    return _ext.is_grad_enabled()
