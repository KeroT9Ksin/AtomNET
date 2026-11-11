#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_vocab_bpe.py — «умное словокупотребление»: byte-BPE tokenizer.

Byte-pair encoding в чистом Python (~120 строк): учит 255 типов слияний
на пробном корпусе, достигает сжатия по словарю; для модели vocab
снижается от 2048 до 256-512 => embedding Params сокращается в 4-8 раз.

byte-level: первые 256 токенов = все байты 0..255 (байт-покрытие гарантирует
OOV-fallback БЕЗ <unk>).

Запуск: python3 tests/demo_vocab_bpe.py
"""

VOCAB_SIZE = 256
N_MERGES = 80

CORPUS = """
"""

def build_corpus() -> bytes:
    # Представительный микро-корпус: текст + цифры + псевдокодли (без kaggle).
    text = (
        "как работает паралельнвый gemm на atomnet\n"
        "the kernel runs on zon 5 7430u cpu with avx2 and fma\n"
        "softmax exp sum normalize for attention heads\n"
        "кэш l1 l2 l3 память блок матричный умножене жа\n"
        "def think(x): return silu(x) @ w + bias\n"
        "for token in prompt stream if tree elinfraed\n"
        "1234567890 abcdefghijklmnop qrst uvwxyz\n"
        "как выстроить лесу троичного поиска\n"
        "интернет neural network transformer layers\n"
        "the data is the water maybe not data os kernel\n"
    ) * 4
    return text.encode("utf-8")


def byte_pairs(data: bytes) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(data, data[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge_tokens(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


class BpeTokenizer:
    """Byte-BPE с каноническим byte-fallback (всегда покрывает всю таблицу)."""

    def __init__(self, merges: list[tuple[tuple[int, int], int]]):
        self.merges = merges
        self.vocab_size = 256 + len(merges)
        self.token_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (pair, new_id) in merges:
            self.token_bytes[new_id] = self.token_bytes[pair[0]] + self.token_bytes[pair[1]]

    def encode(self, text: str) -> list[int]:
        toks = list(text.encode("utf-8"))
        # Применяем слияния по порядке убывания приоритета обучения.
        for pair, new_id in self.merges:
            if len(toks) < 2:
                break
            toks = merge_tokens(toks, pair, new_id)
        return toks

    def decode(self, ids: list[int]) -> str:
        out = b"".join(self.token_bytes[i] for i in ids)
        return out.decode("utf-8", errors="replace")


def train_bpe(data: bytes, n_merges: int = N_MERGES) -> BpeTokenizer:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    tokens = list(data)
    merges = []
    next_id = 256
    for _ in range(n_merges):
        pairs = byte_pairs(tokens)
        if not pairs:
            break
        pair = max(pairs, key=lambda p: pairs[p])
        if pairs[pair] < 2:
            break
        tokens = merge_tokens(tokens, pair, next_id)
        vocab[next_id] = vocab[pair[0]] + vocab[pair[1]]
        merges.append((pair, next_id))
        next_id += 1
    tok = BpeTokenizer(merges)
    tok.token_bytes = vocab
    return tok


def main():
    data = build_corpus()
    tok = train_bpe(data)
    sample = "how does the attention kernel work on atomnet stream\n"
    ids = tok.encode(sample)
    dec = tok.decode(ids)

    bytes0 = len(sample.encode())
    # сжатие по токенам: байтовые куски склеены => меньше токенов на символ
    ratio = bytes0 / max(1, len(ids))
    vocab = tok.vocab_size
    embed_ratio = 2048 / vocab

    print(f"corpus: {len(data)} bytes")
    print(f"vocab size: {vocab} (from byte-floor 256; embedding saved x{embed_ratio:.1f} vs 2048)")
    print(f"sample: {sample.strip()!r}")
    print(f"  tokens: {len(ids)} (bytes {bytes0}, compression ratio {ratio:.2f}x)")
    print(f"  decode ok: {dec == sample}")
    # грубая модель из demo_nnet: vocab 256 vs 2048 по параметрам:
    d = 128
    print(f"  embedding params: {vocab * d} (vs {2048 * d} at vocab=2048)")
    print("OK: smart-vocab byte-BPE готов.")


if __name__ == "__main__":
    main()
