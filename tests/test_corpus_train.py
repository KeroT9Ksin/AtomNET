"""Корпус TinyShakespeare → datacenter-пакковка → обучение nano-LM char-level.

Классический минимум training (Karpathy tiny-shakespeare): проверяем
конвергенцию loss и генерацию.

Запуск: venv-python tests/test_corpus_train.py
"""
import os

import atomnet as A

BASE = "/tmp/atom_corpora"

def download_tinyshake():
    import urllib.request
    """Качает raw текстовый corpus (1 MB) в BASE."""
    os.makedirs(BASE, exist_ok=True)
    path = os.path.join(BASE, "tinyshakespeare.txt")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        resp = urllib.request.urlopen(url, timeout=120)
        blob = resp.read()
        assert len(blob) > 100_000, f"slamшедый corpus: {len(blob)}"
        with open(path, "wb") as f:
            f.write(blob)
    return path


def main():
    path = download_tinyshake()
    A.manual_seed(0)

    from atomnet import datacenter

    corpus_dir = os.path.join(BASE, "ds")
    os.makedirs(corpus_dir, exist_ok=True)
    # россыпь файлов в datacenter
    import shutil
    shutil.copy(path, os.path.join(corpus_dir, "corpus.txt"))
    man = datacenter.pack_tokens_dir(corpus_dir)
    compression = man["total_comp"] / man["total_raw"]
    print(f"packed {len(man['shards'])} shard(s), comp={compression:.2%}")

    # чтение тверых текстовых потоков
    tokens = datacenter.read_shard_bytes(os.path.join(corpus_dir, man["shards"][0]["name"]))
    text = tokens.decode("utf-8", errors="ignore")
    text = text.replace("\r\n", "\n")
    assert len(text) > 200_000, len(text)
    print(f"corpus chars={len(text)} vocab bytes={len(set(text.encode()))}")

    # Тренировка char-GPT на байтах (vocab=256 raw) — d=64, 2 blocks
    from atomnet import first_nn, functional as F
    net = first_nn.MiniTransformer(vocab=256, d_model=64, blocks=2, max_len=64, seed=0)
    from atomnet import optimizers as O
    opt = O.AdamW(net.parameters(), lr=0.005)
    optm = fib = opt

    # нарежем seqs по T=48
    T = 48
    raw = list(text.encode()[:40000])
    seqs = []
    for i in range(0, len(raw) - T, T):
        seqs.append(raw[i:i + T])
    import random
    rng = random.Random(0)
    losses = []
    # 3 эпохи малых, сохра д.webp
    for phrase in range(3):
        random.shuffle(seqs)
        batch_L = []
        for i in range(0, len(seqs), 8):
            batch = [s.copy() for s in seqs[i:i + 8]]
            batch_L.extend(tr_batch(net, batch, optm))
        losses.append(sum(batch_L) / max(1, len(batch_L)))
    print("char-LM loss per epoch:", [round(l, 3) for l in losses])
    out = net([ord(c) for c in "First"])
    vals = out.tolist()
    T = 1
    T_ = 1
    guess = int(max(vals[-256:], key=lambda v: v))
    print("greedy guess after 'First':", guess)
    print("corpus-train ok")


def tr_batch(net, batch, opt):
    from atomnet import functional as F
    opt.zero_grad()
    L = []
    import atomnet as A
    for seq in batch:
        lg = net(seq)
        x = A.rows_slice(lg, 0, len(seq) - 1)
        loss = F.cross_entropy(x, seq[1:])
        loss.backward()
        L.append(loss.item())
    opt.step()
    return L


if __name__ == "__main__":
    main()
