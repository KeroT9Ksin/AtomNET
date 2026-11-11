"""Тесты недостающих слоёв: Conv1d/Conv2d/Pooling/Pad2d/Flatten/RNN + BatchNorm."""
import numpy as np

import atomnet as A
from atomnet import nn


def _T(x, shape):
    return A.Tensor(np.asarray(x, dtype=np.float32).reshape(-1).tolist(), list(shape))


def test_conv1d_matches_numpy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 3)).astype(np.float32)
    w = rng.standard_normal((4, 3 * 3)).astype(np.float32)
    c = nn.Conv1d(3, 4, 3)
    c.weight.copy_(_T(w, [4, 9]))
    out = c(_T(X, [6, 3]))
    ref = np.zeros((4, 4), dtype=np.float32)
    for t in range(4):
        patch = X[t:t + 3].ravel()
        for co in range(4):
            ref[t, co] = float(np.dot(patch, w[co]))
    assert np.allclose(np.array(out.tolist(), dtype=np.float32).reshape(4, 4), ref, atol=2e-5)
    print("OK Conv1d numpy-match")


def test_conv2d_matches_numpy():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2, 5, 5)).astype(np.float32)
    w = rng.standard_normal((1, 18)).astype(np.float32)
    c = nn.Conv2d(2, 1, 3, 3)
    c.weight.copy_(_T(w, [1, 18]))
    out = c(_T(X, [2, 5, 5]))
    ref = []
    for oh in range(3):
        for ow in range(3):
            patch = X[0:, oh:oh + 3, ow:ow + 3].ravel()
            ref.append(float(np.dot(patch, w[0])))
    assert np.allclose(np.array(out.tolist(), dtype=np.float32), ref, atol=2e-5)
    print("OK Conv2d numpy-match")


def test_pool_pad():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2, 5, 5)).astype(np.float32)
    pool = nn.MaxPool2d(2, 2)
    out = pool(_T(X, [2, 5, 5]))
    assert list(out.shape) == [1, 2, 2, 2]
    v = np.array(out.tolist(), dtype=np.float32)
    assert not np.isnan(v).any()
    ap = nn.AdaptiveAvgPool1d(3)
    out2 = ap(_T(rng.standard_normal((9, 2)), [9, 2]))
    assert list(out2.shape) == [3, 2]
    p = nn.Pad2d(1)
    out3 = p(_T(X, [2, 5, 5]))
    assert list(out3.shape) == [2, 7, 7]
    f = nn.Flatten()
    assert list(f(out3).shape) == [2 * 49]
    print("OK pooling/pad/flatten")


def test_rnn():
    rng = np.random.default_rng(3)
    r = nn.ElmanRNN(8, 16)
    h = r(_T(rng.standard_normal((5, 8)), [5, 8]))
    assert list(h.shape) == [16]
    print("OK RNN")


def test_bn_gn():
    A.manual_seed(0)
    x = A.randn([4, 8])
    bn = nn.BatchNorm1d(8)
    bn.training = True
    out = bn(x)
    assert list(out.shape) == [4, 8]
    out2 = bn(x)
    assert list(out2.shape) == [4, 8]
    gn = nn.GroupNorm(4, 8)
    out3 = gn(x.reshape([4, 8]) if False else x)
    assert list(out3.shape) == [4, 8]
    print("OK BatchNorm1d / GroupNorm")


if __name__ == "__main__":
    test_conv1d_matches_numpy()
    test_conv2d_matches_numpy()
    test_pool_pad()
    test_rnn()
    test_bn_gn()
    print("ALL NEW LAYER TESTS OK")
