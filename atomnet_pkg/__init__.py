"""atomnet_pkg — устаревшая shim-обёртка.

Старый пакет перенесён в единый ``atomnet`` (см. ``python/atomnet/``).
Этот модуль оставлен только для обратной совместимости импортов::

    from atomnet_pkg.nn import Linear   # всё ещё работает
    from atomnet.nn import Linear       # предпочтительно
"""
import sys as _sys

try:
    import atomnet as _unified

    for _name in list(getattr(_unified, "__all__", [])):
        try:
            _sys.modules[__name__ + "." + _name] = getattr(_unified, _name)
        except AttributeError:
            pass
    for _mod in ("nn", "functional", "optimizers", "first_nn", "deploy", "datacenter", "dc"):
        try:
            _sys.modules[f"{__name__}.{_mod}"] = __import__(f"atomnet.{_mod}", fromlist=[_mod])
        except Exception:
            pass
    _sys.modules[__name__ + ".atomnet"] = __import__("atomnet.atomnet", fromlist=["x"])
except Exception as _e:  # pragma: no cover
    raise ImportError("atomnet_pkg: единый пакет atomnet не загрузился") from _e
