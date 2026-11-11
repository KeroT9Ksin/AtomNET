# atomnet.array — физический массив (замена NumPy)

## Быстрый старт (никаких настроек)

```python
import atomnet
import atomnet.array as ara

a = ara.randn((512, 512), seed=0)   # конструкция в Rust, без numpy
b = ara.ones((512, 512))
c = a.matmul(b)                     # FLASH-GEMM (блочный AVX2/JIT)
c.silu_()                           # in-place активация, 1 проход
print(c.mean(), c.norm(), hex(c.sig), c.is_alive())
```

При `import atomnet` автоматически: потоки = физические ядра,
Rayon-пул поднят, JIT-ядра скомпилированы, GEMM прогрет (`configure()`).
Отключить прогрев для замеров cold start: `ATOMNET_NO_WARMUP=1`.

## Что умеет

- Фабрики: `zeros, ones, empty, full, arange, linspace, eye, rand, randn`,
  `asarray, array, ascontiguousarray, zeros_like, ones_like, stack, concatenate`.
- Elementwise: `add_arr, sub_arr, mul_arr, add_scalar, mul_scalar`,
  in-place `relu_, silu_, softmax_, gelu_, sigmoid_, tanh_, exp_, log_, sin_, cos_, abs_, clip_, affine_`.
- Редукции за 1 проход: `sum, mean, var, std, max, min, argmax, argmin, norm, dot, mse`.
- Лин. алгебра 2D: `matmul` (`@`), `matvec`, `transpose` (`t()`), `reshape`.
- Физика (нет в numpy): `sig`/`xor_signature`, `xor_fold`, `is_alive`, `gate`,
  `project_sphere`, `quantize_int8`/`dequantize_int8`, `pack_signs`.

Numpy — только на границе: `AtomArray.from_numpy(x)`, `a.numpy()`, `a.tolist()`.

## Скорость (Zen, 1M элементов, см. benchmarks/bench_array.py)

- `zeros`: ~2× быстрее numpy, `randn`: ~1.5× быстрее (Rust RNG vs mt19937).
- `norm`: на уровне numpy; `add`: ~2× медленнее (цена физики: XOR-подпись).
- `matmul` 2048³: паритет с OpenBLAS (~50 мс); на 512³ numpy выигрывает
  за счёт multithread SGEMM — мелкие задачи лучше вести через `gemm_silu`
  fused (один проход вместо gemm+bias+silu).
- `matmul_np` теперь zero-copy (пишет прямо в numpy-массив через
  `gemm_blocked_into`), бит-идентичен `gemm_blocked_np`.

## Ограничения M0

`f32` + `i64` для индексов; broadcasting — только скаляры и совпадение форм;
`matmul` — 2D; `stack/concatenate` — `axis=0`. Полный numpy (fft, dtype-зоопарк,
продвинутая индексация) намеренно вне скоупа.
