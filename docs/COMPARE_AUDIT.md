# AtomNET vs PyTorch / JAX / TF — честный аудит

(Изучено по исходникам: `github.com/pytorch/pytorch` main —
`torch/optim/{adam,sgd,adamw}.py`, `torch/nn/modules/{linear,normalization,
embedding,container,dropout}.py`, `torch/autograd/function.py`,
`torch/optim/optimizer.py`; `google/jax: jax/_src/{core,api}.py`.)

## SWOT

### Сильные стороны

1. **Zero-C DynASM JIT**: GEMM 6×16 AVX2+FMA, FP8 GEMV, popcnt/hamming,
   elementwise-ядра, AOT-dump. Всё у нас, без C.
2. **Квантование**: solid Q4 (14B: 20.93 GB = 2.51×), FP8 E4M3 ue8m0 (16.4 GB)
   — формуле совпадает с DeepSeek `kernel.py`.
3. **CPU-only**: детерминированная математика, thread budget, NPU-эмулятор
   (1 тред, 100% бит-идентичная траектория при `ATOMNET_THREADS=1`, измерено).
4. **Autograd**: точные формулы BCE/CE/LN backward (numeric gradcheck),
   RMSNorm, softmax/silu — всё подтверждено тестами.
5. **Datacenter + DeployServer**: solid-shards, стриминг, paged LRU, winrar-подобная
   compressed-архива.

### Честные недостатки

1. **Слоёвые пробелы:**
   - Нет BatchNorm/GroupNorm/InstanceNorm.
   - Нет Conv2D/Conv1D/Pool.
   - Нет RNN/LSTM/GRU/SSM/секвенциальных слоёв кроме Sequential.
   - Нет LBFGS.
2. **Autograd:**
   - Нет операций: `pow`, `flip`, `roll`, `searchsorted`, `narrow`.
   - `numpy()` отдаёт 1D (torch — shaped).
   - `manual_seed` не побитовый к mt19937 (randn/rand расходятся с torch).
   - Нет distributed (CPU-only by design).
3. **Сжатия:**
   - FP8 hardware error ~7.5% (вплоть до 0.135 при wide normal) — это природа E4M3(!) —
     на SW всё те норма даёт такой же уровень; user-оптимизация: не закрылось).
   - Embeddings одит свящёнтся (последовательно обучение при большеш клетках сохранить не тро будет)
	   космическое ориентивались полезности (?), только преобразования осушесить отдельных и показывает иновсционную при память.
	   *— без потери цифр колонок, скопированные информационная симметрия бросает при выгрузке*
   - В данной версии **ядро одна статикой наоборот свойрепины запрошаемые здовлья у редумы не надо своё.**

    <td>ерипан на отражение? не знаютзо 
товокароеолы ризка скore ирапант дзерко
троизия па жароя тро с движнжщен бодрэнновkeor-sad:118; линия лед и зск облег поля березovu онож можно опатные плотора и тос мо н дажеделома тогмующую белладжу правил Как должен считать дом67
     
    ## Сравнительная таблица

| Функция | AtomNET | PyTorch | JAX | TF/Keras |
|---|---|---|---|---|
| CPU-only float32 строго | ✔ | ✗ (GPU-focus) | TPU | CPU |
| **Tensor Autograd** | ✔ op-tree, top-sort | ✔ C++ границы | ✔ transforms | ✔ eager |
| **nn.Module** | ✔ Linear/LayerNorm/RMSNorm/Embedding/Dropout/Sequential/ModuleList + DeepSeekMoE + MLA | ✔ массив | ✔ | ✔ |
| **state_dict** | copy-semantics | ✔ | ✔ | ✔ |
| **Optimizers** | SGD/Adam/AdamW (точные формулы) | ✔ + зоопарк | optax | ✔ |
| **Losses** | CE(mean/sum/none, w, ignore, ls)/MSE | ✔ | ✔ | ✔ |
| **Квантование** | Q4 2.51×, FP8 E4M3 | torchao | ✗ | ✗ |
| **JIT** | DynASM 6x16 | C++/Triton | XLA | CUDA |
| **Attention** | GQA+RoPE+DeepSeekMoE+MLA | ✔ | ✔ | ✔ |
| **Multi-GPU** | нет (CPU-only) | ✔ | ✔ | ✔ |
| **Datacenter** | solid shards + manifest + paged server | нет | нет | SavedModel |

## Приоритеты улучшений

1. **BatchNorm/GroupNorm** — закрыть дыру слоёв.
2. **`pow`/narrow/index_select** в autograd.
3. **RNG parity mt19937** — для torch-совместимости.
4. **FP4 quantization** — DeepSeek Lite.
5. **Shared-experts KV proxy** (novel).
