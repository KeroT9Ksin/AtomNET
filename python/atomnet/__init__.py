"""AtomNET Python package — native Rust extension + pure-Python models.

Быстрый старт без настроек: при импорте автоматически подбираются
потоки (физические ядра), JIT-ядра и прогрев пайплайна. Никаких
env-переменных не нужно (ATOMNET_NO_WARMUP=1 отключает прогрев
только для замеров cold start).
"""
from .atomnet import *

from . import array as array
from .array import AtomArray

# Автонастройка: потоки = физм. ядра, Rayon-пул, JIT + прогрев GEMM.
import os as _os
if not _os.environ.get("ATOMNET_NO_WARMUP"):
    try:
        from . import atomnet as _native
        _native.configure()
    except Exception:
        try:
            _native.warmup()
        except Exception:
            pass

__doc__ = atomnet.__doc__
if hasattr(atomnet, "__all__"):
    __all__ = list(atomnet.__all__) + ["array", "AtomArray"]
else:
    __all__ = ["array", "AtomArray"]

# Путь к первой работающей сети AtomNET (чистый Python + PyTorch).
try:
    from .luna_mini import LunaMini, LunaMiniConfig, encode_sum
except Exception:  # pragma: no cover - torch может отсутствовать
    pass

# Объединённые модули (перенесены из старого atomnet_pkg).
for _mod in ("nn", "functional", "optimizers", "first_nn", "deploy", "datacenter", "dc"):
    try:
        __import__(f"atomnet.{_mod}", fromlist=[_mod])
    except Exception:
        pass
