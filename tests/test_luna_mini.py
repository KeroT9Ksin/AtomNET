"""Тесты Luna-mini: PyTorch training + AtomNET SIMD-ядра."""

import os
import sys

import numpy as np
import torch

# tests запускаются из корня репозитория / venv — venv в path уже есть
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from atomnet.luna_mini import (  # noqa: E402
    AtomNetLinear,
    LunaMini,
    LunaMiniConfig,
    decode_tokens,
    encode_sum,
    exact_match,
    generate,
    train_addition,
)


def _set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def test_forward_shapes():
    cfg = LunaMiniConfig(backend="atomnet")
    model = LunaMini(cfg)
    ids = torch.tensor([encode_sum(2, 3)[:-1]])
    logits = model(ids)
    assert logits.shape == (1, len(ids[0]), cfg.vocab), logits.shape
    assert torch.isfinite(logits).all()
    print("OK forward_shapes:", tuple(logits.shape))


def test_atomlinear_matches_torch():
    """AtomNetLinear forward и grad совпадают с чистым torch.nn.Linear."""
    _set_seed(42)
    atom = AtomNetLinear(16, 24, bias=True)
    ref = torch.nn.Linear(16, 24, bias=True)
    with torch.no_grad():
        ref.weight.copy_(atom.weight)
        ref.bias.copy_(atom.bias)
    atom.eval()
    ref.eval()
    x = torch.randn(5, 16)
    with torch.no_grad():
        ya = atom(x)
        yr = ref(x)
    assert torch.allclose(ya, yr, atol=1e-5, rtol=1e-4), (ya - yr).abs().max()

    # Backward-совпадение (num grad на одном примере через .backward())
    xa = x.clone().requires_grad_(True)
    ya = atom(xa)
    g = torch.randn_like(ya)
    ya.backward(g)
    xr = x.clone().requires_grad_(True)
    yr = ref(xr)
    yr.backward(g)
    # Градиенты по весу и входу должны совпасть с точностью f32-аккумуляции.
    assert (xa.grad - xr.grad).abs().max().item() < 1e-4
    ga_w = atom.weight.grad
    gr_w = ref.weight.grad
    assert (ga_w - gr_w).abs().max().item() < 1e-4
    print("OK atomlinear_matches_torch: max diff", (ya - yr).abs().max().item())


def test_torch_backend_still_trains():
    """backend='torch' тоже сходится (Санity check бэкенд-совместимости)."""
    cfg = LunaMiniConfig(backend="torch")
    model, hist = train_addition(cfg, steps=300, lr=3e-3, verbose=False)
    em = exact_match(model, [(a, b) for a in range(10) for b in range(10)])
    assert em > 0.5, f"exact-match={em}"
    print(f"OK torch_backend trains: em={em:.2f} loss {hist[0][1]:.3f}→{hist[-1][1]:.3f}")


def test_training_atomnet_converges():
    cfg = LunaMiniConfig(backend="atomnet")
    model, hist = train_addition(cfg, steps=900, lr=3e-3, verbose=False)
    pairs = [(a, b) for a in range(10) for b in range(10)]
    em = exact_match(model, pairs)
    print(f"em={em:.2f} loss {hist[0][1]:.3f}→{hist[-1][1]:.3f}")
    assert em >= 0.97, f"exact-match={em}"


def test_generate_simple():
    cfg = LunaMiniConfig(backend="atomnet")
    model, _ = train_addition(cfg, steps=400, lr=3e-3, verbose=False)
    out = generate(model, [1, 10, 2, 11], n_new=2)
    assert out[-2:] == encode_sum(1, 2)[-2:], out
    assert decode_tokens(encode_sum(7, 8)) == "7+8=15"
    print("OK generate:", decode_tokens(out))


def test_checkpoint_roundtrip(tmp_path=None):
    cfg = LunaMiniConfig(backend="atomnet", d_model=32)
    model, _ = train_addition(cfg, steps=50, lr=5e-3, verbose=False)
    path = "/tmp/opencode/luna-mini-test.pt"
    model.save(path)
    loaded = LunaMini.load(path)
    assert loaded.cfg.d_model == cfg.d_model
    ids = torch.tensor([encode_sum(4, 5)[:-1]])
    with torch.no_grad():
        a = model(ids)
        b = loaded(ids)
    assert torch.allclose(a, b, atol=1e-6)
    # generate from loaded
    out = generate(loaded, [4, 10, 5, 11], n_new=2)
    assert len(out) == 6
    os.remove(path)
    print("OK checkpoint_roundtrip")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
        except Exception as e:
            print("FAIL", fn.__name__, repr(e))
            raise
    print("ALL LUNA-MINI TESTS OK")


if __name__ == "__main__":
    main()
