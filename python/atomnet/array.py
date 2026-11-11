"""atomnet.array — физический массив AtomNET: замена NumPy-буфера.

Всё считается в Rust (SIMD/FLASH-ядра), numpy нужен только на границе
(``from_numpy`` / ``to_numpy``). Никаких env-настроек: потоки,
JIT и прогрев делаются автоматически при ``import atomnet``.

    import atomnet.array as ara

    a = ara.randn((4, 8), seed=0)
    b = ara.ones((4, 8))
    c = (a * b).silu()          # fused-ядра без копий и без numpy
    print(c.mean(), c.sig, c.is_alive())

Поддерживается подмножество numpy API, реально нужное для ML:
фабрики, elementwise, редукции, matmul, stack/concat —
всё на Rust-ядрах AtomNET.
"""
from __future__ import annotations

from .atomnet import AtomArray

__all__ = [
    "AtomArray",
    "zeros", "ones", "empty", "full", "arange", "linspace", "eye",
    "rand", "randn", "asarray", "array", "ascontiguousarray",
    "zeros_like", "ones_like", "stack", "concatenate", "reshape",
    "matmul", "dot", "transpose",
    "sum", "mean", "var", "std", "max", "min", "argmax", "argmin",
    "norm", "mse", "sqrt", "exp", "log", "sin", "cos", "tanh",
    "abs", "clip", "allclose",
]


def _shape(shape) -> list[int]:
    if isinstance(shape, int):
        return [shape]
    return [int(s) for s in shape]


def zeros(shape) -> AtomArray:
    return AtomArray.zeros(_shape(shape))


def ones(shape) -> AtomArray:
    return AtomArray.ones(_shape(shape))


def empty(shape) -> AtomArray:
    return AtomArray.zeros(_shape(shape))


def full(shape, val: float) -> AtomArray:
    return AtomArray.full(_shape(shape), float(val))


def arange(n: int) -> AtomArray:
    return AtomArray.arange(int(n))


def linspace(start: float, stop: float, num: int) -> AtomArray:
    return AtomArray.linspace(float(start), float(stop), int(num))


def eye(n: int) -> AtomArray:
    return AtomArray.eye(int(n))


def rand(shape, seed: int = 0) -> AtomArray:
    return AtomArray.rand(_shape(shape), int(seed))


def randn(shape, seed: int = 0) -> AtomArray:
    return AtomArray.randn(_shape(shape), int(seed))


def _infer_shape(data) -> list[int]:
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return [0]
        inner = _infer_shape(data[0])
        # проверка прямоугольности
        for row in data:
            if _infer_shape(row) != inner:
                raise ValueError("asarray: ragged sequence")
        return [len(data)] + inner
    return []


def _flatten(data, out: list[float]):
    if isinstance(data, (list, tuple)):
        for v in data:
            _flatten(v, out)
    elif isinstance(data, AtomArray):
        out.extend(data.tolist())
    else:
        # numpy-массив на границе (единственное место с numpy)
        try:
            import numpy as _np
            if isinstance(data, _np.ndarray):
                out.extend([float(x) for x in data.ravel().tolist()])
                return
        except ImportError:
            pass
        out.append(float(data))  # type: ignore[arg-type]


def asarray(data, shape=None) -> AtomArray:
    if isinstance(data, AtomArray):
        return data
    # numpy-мост: только конвертация на входе
    try:
        import numpy as _np
        if isinstance(data, _np.ndarray):
            return AtomArray.from_numpy(data)
    except ImportError:
        pass
    if shape is not None:
        # Явная shape: плоский/вложенный — flatten в Rust где возможно.
        if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
            # 2D вложенный → Rust flatten (без цикла Python)
            sh = _shape(shape)
            if len(sh) == 2 and len(data) == sh[0] and data and len(data[0]) == sh[1]:
                return AtomArray.from_rows(data)  # type: ignore[arg-type]
            flat = []
            _flatten(data, flat)
            return AtomArray(flat, sh)
        return AtomArray([float(x) for x in data], _shape(shape))  # type: ignore[arg-type]
    sh = _infer_shape(data)
    if not sh:
        sh = [1]
    if len(sh) == 2 and isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        # Быстрый путь 2D: flatten в Rust, без поэлементного цикла Python.
        return AtomArray.from_rows(data)  # type: ignore[arg-type]
    flat: list[float] = []
    _flatten(data, flat)
    return AtomArray(flat, sh)


def array(data, shape=None) -> AtomArray:
    return asarray(data, shape)


def ascontiguousarray(data) -> AtomArray:
    # AtomArray всегда contiguous — конвертация только для совместимости.
    return asarray(data)


def zeros_like(a: AtomArray) -> AtomArray:
    return AtomArray.zeros(list(a.shape))


def ones_like(a: AtomArray) -> AtomArray:
    return AtomArray.ones(list(a.shape))


def stack(arrays, axis: int = 0) -> AtomArray:
    arrs = [asarray(a) for a in arrays]
    if not arrs:
        raise ValueError("stack: empty")
    # Только axis=0 для 1D/2D (самый частый путь в проекте).
    if axis != 0:
        raise ValueError("stack: only axis=0 in M0")
    shape0 = list(arrs[0].shape)
    for a in arrs:
        if list(a.shape) != shape0:
            raise ValueError("stack: shape mismatch")
    out: list[float] = []
    for a in arrs:
        out.extend(a.tolist())
    return AtomArray(out, [len(arrs)] + shape0)


def concatenate(arrays, axis: int = 0) -> AtomArray:
    arrs = [asarray(a) for a in arrays]
    if not arrs:
        raise ValueError("concatenate: empty")
    if len(arrs) == 1:
        return arrs[0]
    # axis=0 для 2D (конкатенация строк) и 1D.
    if axis != 0:
        raise ValueError("concatenate: only axis=0 in M0")
    if arrs[0].ndim == 1:
        out: list[float] = []
        for a in arrs:
            out.extend(a.tolist())
        return AtomArray(out, [len(out)])
    if arrs[0].ndim == 2:
        cols = arrs[0].shape[1]
        for a in arrs:
            if a.shape[1] != cols:
                raise ValueError("concatenate: cols mismatch")
        out = []
        for a in arrs:
            out.extend(a.tolist())
        total_rows = 0
        for a in arrs:
            total_rows += a.shape[0]
        return AtomArray(out, [total_rows, cols])
    raise ValueError("concatenate: ND not supported in M0")


def reshape(a: AtomArray, shape) -> AtomArray:
    return a.reshape(_shape(shape))


def matmul(a: AtomArray, b: AtomArray) -> AtomArray:
    return a.matmul(b)


def dot(a: AtomArray, b: AtomArray) -> float:
    return float(a.dot(b))


def transpose(a: AtomArray) -> AtomArray:
    return a.transpose()


# ── редукции ──────────────────────────────────────────────────

def sum(a: AtomArray) -> float:  # noqa: A001 (numpy-имя)
    return float(a.sum())


def mean(a: AtomArray) -> float:
    return float(a.mean())


def var(a: AtomArray) -> float:
    return float(a.var())


def std(a: AtomArray) -> float:
    return float(a.std())


def max(a: AtomArray) -> float:  # noqa: A001
    return float(a.max())


def min(a: AtomArray) -> float:  # noqa: A001
    return float(a.min())


def argmax(a: AtomArray) -> int:
    return int(a.argmax())


def argmin(a: AtomArray) -> int:
    return int(a.argmin())


def norm(a: AtomArray) -> float:
    return float(a.norm())


def mse(a: AtomArray, b: AtomArray) -> float:
    # mse через numpy-формулу на Rust-ядре: mean((a-b)^2)
    d = a.sub_arr(b)
    return float(d.mul_arr(d).mean())


def sqrt(a: AtomArray) -> AtomArray:
    import copy
    b = AtomArray(a.tolist(), list(a.shape))
    b.sqrt_()
    return b


def exp(a: AtomArray) -> AtomArray:
    b = AtomArray(a.tolist(), list(a.shape))
    b.exp_()
    return b


def log(a: AtomArray) -> AtomArray:
    b = AtomArray(a.tolist(), list(a.shape))
    b.log_()
    return b


def sin(a: AtomArray) -> AtomArray:
    b = AtomArray(a.tolist(), list(a.shape))
    b.sin_()
    return b


def cos(a: AtomArray) -> AtomArray:
    b = AtomArray(a.tolist(), list(a.shape))
    b.cos_()
    return b


def tanh(a: AtomArray) -> AtomArray:
    b = AtomArray(a.tolist(), list(a.shape))
    b.tanh_()
    return b


def abs(a: AtomArray) -> AtomArray:  # noqa: A001
    b = AtomArray(a.tolist(), list(a.shape))
    b.abs_()
    return b


def clip(a: AtomArray, lo: float, hi: float) -> AtomArray:
    b = AtomArray(a.tolist(), list(a.shape))
    b.clip_(float(lo), float(hi))
    return b


def allclose(a: AtomArray, b: AtomArray, atol: float = 1e-5) -> bool:
    return bool(a.allclose(asarray(b), float(atol)))
