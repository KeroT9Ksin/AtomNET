# tests/test_atom_codec.py
import time

import numpy as np
import atomnet


def bench(name, fn, iters=10):
    fn()
    start = time.perf_counter()

    for _ in range(iters):
        fn()

    elapsed = (time.perf_counter() - start) / iters
    print(f"{name:42} {elapsed * 1000:10.4f} ms")


def to_bits(bits_raw):
    # PyO3 может возвращать Vec<u8> как bytes.
    # list(...) превращает это в обычный список [8, 4, 2].
    return list(bits_raw)


def test_cursor():
    print("=== XorCursor ===")

    x = np.random.randn(4096).astype(np.float32) * 0.02

    cursor = atomnet.PyXorCursor()
    cursor.update_np(x)

    print("alive:", cursor.is_alive())
    print("cold:", cursor.is_cold())
    print("stable:", cursor.is_stable())
    print("changing fast:", cursor.is_changing_fast())
    print("alive bits:", cursor.alive_bits)
    print("hamming:", cursor.hamming)
    print("suggested bits:", to_bits(cursor.suggested_bits()))

    assert cursor.is_alive()
    assert not cursor.is_changing_fast()


def test_atom_codec_quality():
    print("\n=== AtomCodec quality ===")

    # Важно:
    # не используем голый randn со std=1.
    # Для весов реальных сетей масштаб обычно намного меньше.
    x = (np.random.randn(8192) * 0.02).astype(np.float32)

    configs = [
        [8],
        [8, 4],
        [8, 4, 2],
        [4],
        [4, 2],
    ]

    errors = {}

    print("config".ljust(16), "error".ljust(16), "bytes/param")

    for bits in configs:
        compressed = atomnet.compress_atom_np(x, bits)
        err = float(compressed.max_error_np(x))
        bpp = float(compressed.bytes_per_param())

        errors[tuple(bits)] = err

        print(str(bits).ljust(16), f"{err:.8f}".ljust(16), f"{bpp:.4f}")

        assert compressed.is_alive()

    # Проверка логики многоуровневости:
    # каждый следующий остаточный уровень должен уменьшать ошибку.
    assert errors[(8, 4)] < errors[(8,)] + 1e-9
    assert errors[(8, 4, 2)] < errors[(8, 4)] + 1e-9
    assert errors[(4, 2)] < errors[(4,)] + 1e-9

    # Для [8,4] на маленьких весах ошибка должна быть разумной.
    assert errors[(8, 4)] < 0.1


def test_atom_codec_vs_micron():
    print("\n=== AtomCodec vs Micron int8 ===")

    x = (np.random.randn(8192) * 0.02).astype(np.float32)

    # Micron int8: прямое блочное квантование.
    q, mu, sc = atomnet.quantize_int8_np(x)
    x_micron = atomnet.dequantize_int8_np(q, mu, sc)
    micron_err = float(np.abs(x - x_micron).max())

    # AtomCodec [8]
    atom8 = atomnet.compress_atom_np(x, [8])
    atom8_err = float(atom8.max_error_np(x))

    # AtomCodec [8, 4]
    atom84 = atomnet.compress_atom_np(x, [8, 4])
    atom84_err = float(atom84.max_error_np(x))

    # AtomCodec [8, 4, 2]
    atom842 = atomnet.compress_atom_np(x, [8, 4, 2])
    atom842_err = float(atom842.max_error_np(x))

    print("micron int8 error:        ", micron_err)
    print("atom codec [8] error:     ", atom8_err)
    print("atom codec [8,4] error:   ", atom84_err)
    print("atom codec [8,4,2] error: ", atom842_err)

    print("micron bytes/param:       ", 1.0)
    print("atom [8] bytes/param:     ", float(atom8.bytes_per_param()))
    print("atom [8,4] bytes/param:   ", float(atom84.bytes_per_param()))
    print("atom [8,4,2] bytes/param: ", float(atom842.bytes_per_param()))

    # AtomCodec [8] может быть хуже прямого Micron int8.
    # Это нормально: [8] — сферическое направление, а не прямое квантование.
    # Но многоуровневый AtomCodec должен уменьшать ошибку.
    assert atom84_err < atom8_err + 1e-9
    assert atom842_err < atom84_err + 1e-9


def test_atom_codec_speed():
    print("\n=== AtomCodec speed ===")

    x = (np.random.randn(262_144) * 0.02).astype(np.float32)

    bench(
        "micron_quantize_int8_np",
        lambda: atomnet.quantize_int8_np(x),
    )

    bench(
        "atom_codec_compress_[8]",
        lambda: atomnet.compress_atom_np(x, [8]),
    )

    bench(
        "atom_codec_compress_[8,4]",
        lambda: atomnet.compress_atom_np(x, [8, 4]),
    )

    bench(
        "atom_codec_compress_[8,4,2]",
        lambda: atomnet.compress_atom_np(x, [8, 4, 2]),
    )

    compressed = atomnet.compress_atom_np(x, [8, 4])

    bench(
        "atom_codec_decompress_[8,4]",
        lambda: compressed.decompress_np(),
    )


def test_cursor_adaptive():
    print("\n=== XorCursor adaptive compression ===")

    cursor = atomnet.PyXorCursor()

    x = (np.random.randn(4096) * 0.02).astype(np.float32)

    for _ in range(20):
        cursor.update_np(x)
        cursor.adapt_compression(3)

    print("steps alive:", cursor.steps_alive)
    print("compression level:", cursor.compression_level)
    print("suggested bits:", to_bits(cursor.suggested_bits()))

    assert cursor.steps_alive > 0
    assert cursor.is_alive()
    assert not cursor.is_cold()


if __name__ == "__main__":
    print("AtomNET version:", atomnet.version())

    test_cursor()
    test_atom_codec_quality()
    test_atom_codec_vs_micron()
    test_atom_codec_speed()
    test_cursor_adaptive()

    print("\nATOM CODEC TEST FINISHED")
