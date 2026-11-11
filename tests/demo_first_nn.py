"""Первая нейросеть AtomNET — MiniTransformer на torch-API.

Обучение на задаче сложения (цифровые токены): "07+02=" -> "009".
Все параметры — Module/Parameter, оптимизатор — Adam, loss — F.cross_entropy,
инференс и eval — через no_grad, чекпоинт — save/load_state_dict (JSON).

Проверка single-thread NPU: запустите с ATOMNET_THREADS=1 / taskset -c 0 —
результат не меняется (tests/test_single_thread.py), а с ATOMNET_NPU_EMU=1
GEMM идёт через JIT-mac-тайлы (6x16) в 1 поток — траектория обучения та же
до допуска (tests/test_npu_emu.py).
"""
import math
import sys

import atomnet as A
from atomnet import nn, optimizers, functional as F, first_nn

VOCAB = 12  # 0..9 цифры, +(10), =(11)
PLUS, EQ = 10, 11

sys.setrecursionlimit(10000)


def encode(a, b):
    s = a + b
    return [a // 10, a % 10, PLUS, b // 10, b % 10, EQ, s // 100, (s // 10) % 10, s % 10]


def argmax_row(logits_tensor):
    vals = logits_tensor.tolist()
    return max(range(len(vals)), key=lambda i: vals[i])


def greedy_next(net, prefix):
    """argmax next-token generation (последняя строка логитов)."""
    with F.no_grad():
        assert not F.is_grad_enabled(), "no_grad не вступил в силу"
        logits = net(prefix)
        vals = logits.tolist()
        T, voc = len(prefix), net.vocab
        row = vals[(T - 1) * voc: T * voc]
        return max(range(voc), key=lambda i: row[i])


def demo():
    A.manual_seed(0)
    net = first_nn.MiniTransformer(vocab=VOCAB, d_model=32, blocks=2, max_len=16, seed=0)
    net.train()
    opt = optimizers.Adam(net.parameters(), lr=0.02, weight_decay=0.01)

    # forward hooks (как torch.register_forward_hook)
    hook_calls = [0]

    def counter(module, args, kwargs, out):
        hook_calls[0] += 1

    for m in net.named_modules():
        if m[1] is not net:
            m[1].register_forward_hook(counter)
            break

    # данные: мини-пакеты сложения по 8 примеров
    import random
    rng = random.Random(0)
    pairs = [(rng.randrange(10), rng.randrange(10)) for _ in range(8)]
    losses = []
    steps = 24
    for step in range(steps):
        opt.zero_grad()
        L = 0.0
        for a, b in pairs:
            seq = encode(a, b)
            logits = net(seq)
            x = A.rows_slice(logits, 0, 8)
            loss = F.cross_entropy(x, seq[1:])
            loss.backward()
            L += loss.item()
        L /= len(pairs)
        losses.append(L)
        opt.step()
        if step % 6 == 0 or step == steps - 1:
            print(f"step {step:3d}  loss {L:.4f}")

    # eval-режим + no_grad + greedy
    net.eval()
    exact = 0
    for a, b in [(1, 2), (4, 5), (9, 0), (6, 7)]:
        seq = encode(a, b)
        pred = greedy_next(net, seq[:6])
        exact += (pred == seq[6])
        print(f"  {a}+{b} greedy next (сотни): pred {pred} true {seq[6]}")
    print(f"hook calls: {hook_calls[0]}")
    assert losses[-1] < losses[0], "потеря должна убывать"

    # state_dict roundtrip: файл -> обратный load -> тот же forward
    first_nn.save_state_dict(net, "/tmp/first_nn.bin")
    net2 = first_nn.MiniTransformer(vocab=VOCAB, d_model=32, blocks=2, max_len=16)
    first_nn.load_state_dict(net2, "/tmp/first_nn.bin")
    with F.no_grad():
        y1 = net(encode(2, 3)).tolist()
        y2 = net2(encode(2, 3)).tolist()
    assert all(abs(a - b) < 1e-6 for a, b in zip(y1, y2)), "state_dict roundtrip"
    print("state_dict roundtrip ok")
    return "SELFTEST OK"


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        print(demo())
