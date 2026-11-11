"""PyTorch-паритетные тесты ядра torchx (Tensor+autograd, nn, optim, F).

Эталоны — ручные numpy-реализации канонических формул из torch/nn/functional.py,
torch/optim/{adam,sgd,adamw}.py (studied), module.py.
"""
import numpy as np

import atomnet as A
from atomnet import nn, functional
from atomnet import optimizers as optim
from atomnet import Tensor


def mk(arr, requires=True):
    arr = np.asarray(arr, dtype=np.float32)
    return Tensor(arr.reshape(-1).tolist(), list(arr.shape), requires)


def g(t):
    return np.array(t.grad.tolist(), dtype=np.float32).reshape(t.grad.shape)


# ───────────────────── Tensor/autograd: база ─────────────────────

def test_mul_add_chain_backward():
    x = mk([3.0], True)
    wm = x * 2.0
    wm.backward()
    assert np.allclose(g(x), [2.0]), g(x)
    # накопление через второй backward
    y = x * 4.0
    y.sum().backward()
    assert np.allclose(g(x), [6.0]), g(x)


def test_matmul_backward():
    A.manual_seed(0)
    xv = np.random.randn(2, 4).astype(np.float32)
    wv = np.random.randn(3, 4).astype(np.float32)
    x, w = mk(xv), mk(wv)
    y = x.matmul(w.t())
    assert np.allclose(np.array(y.tolist()).reshape(2, 3), xv @ wv.T, atol=1e-4)
    y.sum().backward()
    assert np.allclose(g(x), np.ones((3, 4)).sum(0)[None, :] * 0 + (wv.sum(0)[None, :]), atol=1e-4)
    assert np.allclose(g(w), np.broadcast_to(xv.sum(0)[None, :], (3, 4)), atol=1e-4)


def test_broadcast_backward():
    a = mk(np.ones((1, 4)), True)
    b = mk(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), True)
    c = a + b
    c.sum().backward()
    # grad a: сумма по строкам b (2 строки) → [2,2,2,2]
    assert np.allclose(g(a), 2.0 * np.ones((1, 4)), atol=1e-5)
    assert np.allclose(g(b), np.ones((2, 4)), atol=1e-5)


def test_matmul_1d_paths():
    x = mk([1.0, 2.0, 3.0, 4.0])
    # linear: x (in=4) с весом (out,in)=(2,4) => y = W @ x = [12, 5]
    B = mk(np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 0.0]]))
    y = functional.linear(x, B)
    assert np.allclose(y.tolist(), [12.0, 5.0])
    y.sum().backward()
    # d(loss)/dB = ones(out,1) @ x^T = 1 outer x (2,4) = [[1,2,3,4],[1,2,3,4]]
    expect_B = np.ones((2, 4), dtype=np.float32) * np.array([1.0, 2.0, 3.0, 4.0])[None, :]
    assert np.allclose(g(B), expect_B, atol=1e-5)
    # dx = B^T @ ones(out) = [1,1,2,2]
    assert np.allclose(g(x), np.array([1.0, 1.0, 2.0, 2.0], np.float32), atol=1e-5)


# ───────────────────── CE (exact, как torch) ─────────────────────

def _logsumexp(x, axis=-1):
    mx = x.max(axis=axis, keepdims=True)
    return (mx + np.log(np.exp(x - mx).sum(axis=axis, keepdims=True))).squeeze(axis) if False else mx.squeeze(axis) + np.log(np.exp(x - mx).sum(axis=axis))


def test_ce_forward_exact():
    A.manual_seed(0)
    logits = np.random.randn(5, 4).astype(np.float32)
    target = [0, 2, 1, 3, 2]
    l = functional.cross_entropy(mk(logits, False), target)
    ex = _logsumexp(logits, axis=1) - logits[np.arange(5), target]
    assert np.allclose(l.item(), float(ex.mean()), atol=1e-5), (l.item(), float(ex.mean()))


def test_ce_backward_exact():
    A.manual_seed(1)
    logits = np.random.randn(4, 3).astype(np.float32)
    target = [1, 0, 2, 1]
    x = mk(logits, True)
    functional.cross_entropy(x, target).backward()
    p = np.exp(logits - _logsumexp(logits, axis=1, keepdims=True) if False else logits - _lse(logits))
    one = np.eye(3, dtype=np.float32)[target]
    assert np.allclose(g(x), (p - one) / 4.0, atol=1e-5)


def _lse(z):
    mx = z.max(axis=1, keepdims=True)
    return mx + np.log(np.exp(z - mx).sum(axis=1, keepdims=True))


def test_ce_ignore_and_weights():
    A.manual_seed(2)
    logits = np.random.randn(4, 3).astype(np.float32)
    # ignore_index: 2-я строка игнорируется
    l = functional.cross_entropy(mk(logits, False), [0, 1, -100, 2], ignore_index=-100)
    ex = (_logsumexp(logits[[0, 1, 3]], axis=1) - logits[[0, 1, 3], [0, 1, 2]]).mean()
    assert np.allclose(l.item(), float(ex), atol=1e-5)
    # weight: mean = sum(w*l)/sum(w)
    w = [1.0, 2.0, 5.0]
    lw = functional.cross_entropy(mk(logits, False), [0, 1, 2, 0], weight=w)
    per = (_logsumexp(logits, axis=1) - logits[np.arange(4), [0, 1, 2, 0]]).astype(np.float64) * np.array(w)[[0, 1, 2, 0]]
    expect = per.sum() / np.array(w)[[0, 1, 2, 0]].sum()
    assert np.allclose(lw.item(), float(expect), atol=1e-5)


def test_ce_label_smoothing():
    A.manual_seed(3)
    logits = np.random.randn(3, 4).astype(np.float32)
    target = [1, 2, 0]
    eps = 0.1
    l = functional.cross_entropy(mk(logits, False), target, label_smoothing=eps)
    logp = logits - _lse(logits)
    one = np.eye(4)[target]
    dist = (1 - eps) * one + eps / 4.0
    expect = float((-(dist * logp).sum(1)).mean())
    assert np.allclose(l.item(), expect, atol=1e-5)


# ───────────────────── LayerNorm (biased var!) ─────────────────────

def test_ln_forward_exact():
    A.manual_seed(4)
    x = np.random.randn(3, 5).astype(np.float32) * 3.0
    gm = np.random.randn(5).astype(np.float32)
    bt = np.random.randn(5).astype(np.float32)
    y = functional.layer_norm(mk(x, False), [5], mk(gm, False), mk(bt, False), 1e-5)
    mu = x.mean(1, keepdims=True)
    var = ((x - mu) ** 2).mean(1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + 1e-5)
    expect = xh * gm + bt
    assert np.allclose(np.array(y.tolist()).reshape(3, 5), expect, atol=1e-4)


def test_ln_backward_numeric():
    A.manual_seed(5)
    x = (np.random.randn(4, 3).astype(np.float32) + 0.5)
    gm = np.random.randn(3).astype(np.float32)
    xt = mk(x, True)
    yt = mk(gm, True)
    y = functional.layer_norm(xt, [3], yt, None, 1e-5)
    y.sum().backward()
    gx = g(xt)
    eps = 1e-3
    num = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xp = x.copy(); xp[i, j] += eps
            xm = x.copy(); xm[i, j] -= eps
            fp = float(functional.layer_norm(mk(xp, False), [3], mk(gm, False), None, 1e-5).sum().item())
            fm = float(functional.layer_norm(mk(xm, False), [3], mk(gm, False), None, 1e-5).sum().item())
            num[i, j] = (fp - fm) / (2 * eps)
    assert np.allclose(gx, num, atol=2e-3), (gx, num)


# ───────────────────── Embedding ─────────────────────

def test_embedding_scatter():
    W = np.random.randn(6, 3).astype(np.float32)
    w = mk(W, True)
    idx = [1, 4, 1, 2, 4]
    # forward
    emb = functional.embedding(idx, w)
    assert np.allclose(np.array(emb.tolist()).reshape(5, 3), W[idx], atol=1e-4)
    emb.sum().backward()
    gw = g(w)
    for i, cnt in [(1, 2), (4, 2), (2, 1)]:
        assert np.allclose(gw[i], cnt * np.ones(3), atol=1e-5), (i, gw[i])
    assert np.allclose(gw[0], 0.0)


# ───────────────────── optimizers: эталоны ─────────────────────

def test_sgd_momentum_vs_reference():
    # 3 шага ручного SGD с momentum (torch формула)
    A.manual_seed(0)
    th0 = np.array([1.0, -2.0], np.float64)
    grads = [np.array([0.5, 0.0]), np.array([0.0, -0.5]), np.array([0.25, 0.25])]
    lr, momentum, damp, wd = 0.1, 0.9, 0.0, 0.01
    fwd = th0.copy()
    p = mk(th0.astype(np.float32))
    opt = optim.SGD([p], lr=lr, momentum=momentum, dampening=damp, weight_decay=wd)
    buf = None
    for gv in grads:
        opt.zero_grad()
        p.grad = mk(gv.astype(np.float32))
        opt.step()
        d = gv + wd * fwd
        if buf is None:
            buf = d.copy()
        else:
            buf = momentum * buf + (1 - damp) * d
        fwd = fwd - lr * buf
    assert np.allclose(np.array(p.tolist()), fwd.astype(np.float32), atol=1e-5)


def test_adam_two_steps_vs_reference():
    A.manual_seed(0)
    th0 = np.array([1.0, -0.5], np.float64)
    grads = [np.array([0.2, 0.1]), np.array([-0.1, 0.3])]
    lr, b1, b2, eps, wd = 0.01, 0.9, 0.999, 1e-8, 0.01
    fwd = th0.copy()
    m = v = np.zeros(2)
    p = mk(th0.astype(np.float32))
    opt = optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    for t, gv in enumerate(grads, 1):
        opt.zero_grad()
        p.grad = mk(gv.astype(np.float32))
        opt.step()
        # coupled weight decay (torch Adam)
        th = fwd - lr * wd * fwd
        m = b1 * m + (1 - b1) * gv
        v = b2 * v + (1 - b2) * gv ** 2
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        step = lr / bc1
        denom = np.sqrt(v) / np.sqrt(bc2) + eps
        fwd = th - step * m / denom
    assert np.allclose(np.array(p.tolist()), fwd.astype(np.float32), atol=1e-5), (
        np.array(p.tolist()), fwd)


def test_adamw_decoupled_decay():
    p = mk(np.array([2.0], np.float32))
    lr, wd = 0.1, 0.5
    p.grad = mk(np.array([0.0], np.float32))
    opt = optim.AdamW([p], lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    opt.step()
    # gradient=0 → m=0 → adam-step 0; decoupled decay: θ*(1-lr*wd)
    assert np.allclose(p.tolist(), [2.0 * (1 - lr * wd)], atol=1e-6)


def test_state_dict_roundtrip_and_optim_state():
    A.manual_seed(7)
    net = nn.Sequential(nn.Linear(4, 5), nn.SiLU(), nn.Linear(5, 2))
    x = A.randn([2, 4])
    net(x).sum().backward()
    opt = optim.Adam(net.parameters(), lr=0.01)
    opt.step()
    sd = net.state_dict()
    net2 = nn.Sequential(nn.Linear(4, 5), nn.SiLU(), nn.Linear(5, 2))
    net2.load_state_dict(sd)
    y1 = net(x).tolist()
    y2 = net2(x).tolist()
    assert np.allclose(np.array(y1) - np.array(y2), 0.0, atol=1e-6), (y1, y2)
    # missing key check
    try:
        net2.load_state_dict({"0.weight": sd["0.weight"], "0.bias": sd["0.bias"]}, strict=True)
        raise AssertionError("ожидался RuntimeError про missing")
    except RuntimeError as e:
        assert "Missing key" in str(e)
    # optimizer state save/load
    osd = opt.state_dict()
    opt2 = optim.Adam(net.parameters(), lr=0.01)
    opt2.load_state_dict(osd)
    assert opt2.state[0]["step"] == 1


def test_zero_grad_none_semantics():
    p = mk([1.0], True)
    x = p * 3.0
    x.sum().backward()
    assert p.grad is not None
    p.zero_grad(set_to_none=True)
    assert p.grad is None


def test_mlp_train_step_numpy_gradcheck():
    # MLP gradcheck против central diff (точная проверка autograd-цепочек)
    A.manual_seed(8)
    xv = np.random.randn(8, 4).astype(np.float32)
    W1 = np.random.randn(4, 6).astype(np.float32) * 0.5
    b1 = np.random.randn(6).astype(np.float32)
    W2 = np.random.randn(6, 3).astype(np.float32) * 0.5
    b2 = np.random.randn(3).astype(np.float32)

    x = mk(xv, True); w1 = mk(W1, True); bb1 = mk(b1); w2 = mk(W2, True); bb2 = mk(b2)
    h1 = x.matmul(w1).add(bb1).silu()
    out = h1.matmul(w2).add(bb2)
    out.sum().backward()
    num_eps = 1e-3
    def loss_w(wv):
        x_ = mk(xv, False)
        h_ = x_.matmul(mk(wv, False)).add(bb1).silu()
        z_ = h_.matmul(w2).add(bb2)
        return float(z_.sum().item())
    num = np.zeros_like(W1)
    for i in range(4):
        for j in range(6):
            wp = W1.copy(); wp[i, j] += num_eps
            wm = W1.copy(); wm[i, j] -= num_eps
            num[i, j] = (loss_w(wp) - loss_w(wm)) / (2 * num_eps)
    assert np.allclose(g(w1), num, atol=3e-3), (g(w1), num)
    loss0 = float(out.sum().item())
    opt = optim.SGD([w1, w2], lr=5e-3, momentum=0.0)
    opt.step()
    assert float(out.sum().item()) <= loss0 + 1e-3


def test_task_end_to_end_mini_lm():
    """Mini LM (Embedding→Linear→CE) обучается: loss убывает, всё в torch API."""
    A.manual_seed(42)
    net = nn.Sequential(nn.Linear(20, 20), nn.LayerNorm([20]), nn.SiLU(), nn.Linear(20, 3))
    opt = optim.Adam(net.parameters(), lr=0.03)
    X = A.tensor(np.random.randn(8 * 20).tolist(), [8, 20], False)
    y = A.tensor([0, 1, 2, 1, 0, 2, 2, 1])
    losses = []
    for step in range(12):
        opt.zero_grad()
        logits = net(X)
        loss = functional.cross_entropy(logits, [0, 1, 2, 1, 0, 2, 2, 1])
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], losses


if __name__ == "__main__":
    for fn in [v for k, v in sorted(vars().items()) if k.startswith("test_")]:
        fn()
        print(f"OK {fn.__name__}")
