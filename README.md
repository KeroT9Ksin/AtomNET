# AtomNET

**CPU-only нейросетевой фреймворк на Rust** с собственной математикой тензоров,
SIMD/AVX2 ядрами, JIT-компиляцией GEMM на DynASM (без единой строки C),
PyO3-биндингами в стиле PyTorch, обучаемым сервером и иерархической
умной памятью (RAM ↔ диск, Q4/FP8-сжатие, paged-LRU).

> Статус: **research core v0.2.0** — всё измеряется и проверяется тестами.
> Честные замеры и известные ограничения — в конце файла и в `docs/COMPARE_AUDIT.md`.

---

## Быстрый старт

```bash
# сборка (нужен Rust + maturin-совместимое окружение; C-компилятор не используется)
cargo build --release --offline

# положить расширение в Python-окружение
cp target/release/libatomnet.so \
   $VENV/lib/python3.14/site-packages/atomnet/atomnet.cpython-314-x86_64-linux-gnu.so

# тесты
cargo test --release --offline --lib          # 107 passed / 0 failed / 6 ignored
venv/bin/python tests/test_torchx_parity.py   # Python-сьюты
```

```python
import numpy as np
import atomnet as A
from atomnet import nn, functional as F, optimizers as ox

x = A.Tensor([1.0, 2.0, 3.0], [1, 3])
m = nn.Linear(3, 4)
y = m(x)                       # forward
loss = F.mse_loss(y, A.Tensor([0.0] * 4, [1, 4]))
loss.backward()                # autograd (torchx)
ox.AdamW(m.parameters(), lr=1e-3).step()

# физический массив (замена NumPy, JIT-GEMM)
import atomnet.array as ara
c = ara.randn((512, 512), seed=0) @ ara.ones((512, 512))
c.silu_()
```

При `import atomnet` автоматически подбираются потоки (физические ядра),
поднимается Rayon-пул, компилируются JIT-ядра и прогревается GEMM
(`ATOMNET_NO_WARMUP=1` — отключить прогрев для замеров cold start).

---

## Архитектура (детальный разбор кода)

```
src/
├── lib.rs                  # pub mod atom + mod python
├── python.rs               # ~490 PyO3-функций/классов (мост Rust ↔ Python)
├── bin/                    # CLI-бинарники: bench_gemm, bench_kernel,
│                           #   check_budget, luna, xorprobe
└── atom/
    ├── math/               #AtomMATH — чистая математика
    │   ├── atommath.rs     #   атомная сферическая арифметика: атом (nucleus, radius,
    │   │                   #   direction, shell), геодезический шаг, Householder,
    │   │                   #   Fubini–Study тангенциальная проекция (см. docs/ATOM_MATH.md)
    │   ├── tensor.rs       #   аналитические backward: softmax/silu/CE/LN/RMSNorm
    │   ├── simd.rs         #   статические AVX2+FMA ядра (dot/norm/reductions)
    │   ├── atomjit.rs      #   JIT (DynASM): dot/axpy/norm + composite geodesic/householder
    │   ├── binjit.rs       #   бинарная машина: hamming/xor/and/popcnt8 на DynASM (sysv64)
    │   ├── regnum.rs       #   регистровая система счисления: popcount/byte-таблицы,
    │   │                   #   reg_xor/and/or-пары, волновой exchange, подписи атомов
    │   ├── sferom.rs       #   SFEROM — иерархическая сферическая память CORE→LOBE→nanos
    │   │                   #   (порт KeroT9Ksin/Rineto: 0.70·cos + 0.20·(1−hamming/64) + …)
    │   ├── qairo.rs        #   Qairo — атомарная замена линейного слоя (атом на строку)
    │   ├── enigma.rs       #   взрывные/радиальные составные операции
    │   ├── micron.rs       #   микро-квант-ядро int4 residual
    │   ├── atom_codec.rs   #   атомное кодирование весов
    │   └── pool.rs         #   пул буферов (закрытый, без гонок)
    │
    ├── flash/              #AtomFLASH — собственный BLAS + квантованные слои
    │   ├── flash.rs        #   уровни: FLASH-CORE (f32 GEMV 2-строчный blocking,
    │   │                   #   AVX2+FMA, prefetch, 64B-align, DAZ/FTZ),
    │   │                   #   FLASH-INT4 (Q4×Q8), ядро системы
    │   ├── jit.rs          #   DynASM JIT GEMM 6×16: 12 YMM-аккумуляторов — регистровый
    │   │                   #   аллокатор наш, не LLVM; MATH_TABLE — единая таблица
    │   │                   #   констант (exp/silu/logistic), 256-bit broadcast = vpbroadcastd
    │   ├── flash_gemm.rs   #   кэш-блочный GEMM: регистры → L1 → L2 → L3 → pool,
    │   │                   #   JC→PC-петля (BLIS/Goto-порядок), 2D-сетка потоков,
    │   │                   #   MC/KC/NC-параметры (env ATOMNET_GEMM_MC/KC/NC)
    │   ├── flash_compressed.rs  # FlashCompressed (Q4×Q8 + int8-остаток = f32-точность),
    │   │                   #   FlashInt16 (Q16×Q16), FlashFP8 (E4M3FN + ue8m0, блок 128,
    │   │                   #   s = amax/448 — формула сверена с DeepSeek kernel.py)
    │   ├── flash_xor.rs    #   XOR-подписи слоёв (жизнь/целостность)
    │   ├── flash_arena.rs  #   arena-аллокатор под слои
    │   ├── flash_hw.rs     #   thread_budget() (ATOMNET_THREADS, N=1 → без пула),
    │   │                   #   L3-кластеры, pin_thread, huge pages, CpuInfo
    │   └── flash_facade.rs #   единый фасад «хоть dense, хоть packed»
    │
    ├── optim/              #AtomOPTIM
    │   ├── keralin.rs      #   **Keralin** — наш оптимизатор: вес = атом
    │   │                   #   (nucleus/radius/direction), градиент раскладывается на
    │   │                   #   ядерный/радиальный/тангенциальный; decoupled weight decay,
    │   │                   #   bias-correction, grad clip, RMS-адаптация, warm-фаза,
    │   │                   #   параллельный шаг графа (rayon), LR-schedule (warmup+cosine),
    │   │                   #   EMA-snapshot, NAdam-lookahead; KeralinGraph — узлы-атомы
    │   │                   #   с зависимостями (topo-порядок), жизнь/смерть узлов
    │   └── rineta.rs        #   Rineta — резонансная математика: волны, когерентность,
    │                       #   gain, rin[]-состояния (Rin/RinArray)
    │
    ├── life/               #AtomLIFE — жизнь состояния (НЕ оптимизатор)
    │   ├── life.rs         #   AtomLife: живость атома, XOR-подпись, cold start
    │   ├── rin_vec.rs      #   вектор жизни
    │   ├── qairo_life.rs / qairo_cursor.rs / xor_cursor.rs   # жизнь слоёв/курсоры
    │
    ├── nn.rs               # gradcheck-проверенные компоненты: RMSNorm, GQA-Attention
    │                       #   (QKV+RoPE+causal), batch cross-entropy, активации
    ├── seq.rs              # причинное внимание с KV-кэшем, sampler
    ├── model.rs            # TransformerLM — сборщик decoder-block
    │                       #   h = x + Attn(RMSNorm(x)); h += Down(silu(Up(RMSNorm(h))))
    │                       #   веса строк = узлы KeralinGraph (единый реестр)
    ├── torchx.rs           # Tensor + autograd «в духе torch.autograd»: leaf.grad
    │                       #   компаундинг, документированные отклонения (CPU-only, f32)
    ├── train.rs            # Rust-обучение без NumPy: реестр слоёв, аналитические
    │                       #   backward, полный батч, шаг через KeralinGraph
    ├── tok.rs              # byte-level BPE: n_merges слияний, decode(encode(x)) == x
    │                       #   (гарантия байт-фолбэка), PyBpeTokenizer
    ├── graph/
    │   └── dyn_graph.rs    # **DynGraph** — динамический граф: add_const/add_node/op,
    │                       #   level-scheduling, Rayon-work-stealing run (DC: 1.2 ms)
    ├── npu/mod.rs          # NPU-эмулятор: виртуальные ядра 6×16 (12 YMM) маппятся
    │                       #   на CPU-потоки; 1 тред = бит-в-бит тот же результат
    │                       #   (ATOMNET_NPU_EMU, ATOMNET_NPU_CORES)
    ├── isa.rs              # виртуальный ISA: AVX2/FMA, 16 YMM, 2 FMA-порта — единая
    │                       #   точка правды для всех JIT/SIMD-гейтов (tier форсируем)
    ├── array.rs            # AtomArray — физический массив (замена NumPy): row-major f32
    │                       #   + XOR-подпись/жизнь/энергия; rand=Xoroshiro (быстрее mt19937)
    ├── veron.rs            # Veron: YAML-конфигурация, runtime, сборка модели
    ├── physics/hexiron.rs  # Hexiron — 6-вершинная физика: energy/coherence/entropy/
    │                       #   temperature и др.
    ├── prod/mod.rs         # продакшн-обвязка: curriculum-обучение (бинарь luna.rs)
    └── model.rs/…          # см. выше
```

Python-пакет (`python/atomnet/`, собирается maturin'ом, `atomnet_pkg/` — legacy-shim):

| Модуль | Содержимое |
|---|---|
| `atomnet.nn` | `Parameter, Module, Linear, Embedding, LayerNorm, RMSNorm, Sequential, ModuleList, Dropout`, активации (`ReLU, SiLU, GELU, Tanh, Sigmoid`), **`DeepSeekMoE`** (sigmoid-гейт + aux-loss-free bias + group-limited routing + SwiGLU-эксперты + shared expert), **`MLALayer`** + `LatentCache` (latent KV-cache), `ChainGraph`, `BatchNorm1d, GroupNorm, Conv1d, Conv2d, MaxPool1d/2d, AvgPool1d, AdaptiveAvgPool1d, Pad2d, Flatten, ElmanRNN` |
| `atomnet.functional` | `cross_entropy (label_smoothing, ignore_index), linear, embedding, mse, layer_norm, softmax, gelu/silu/relu/tanh/sigmoid, rms_norm, no_grad` |
| `atomnet.optimizers` | `SGD (momentum/nesterov), Adam, AdamW (decoupled wd)` — точные формулы PyTorch |
| `atomnet.torchx` | `Tensor` с autograd, `atomnet.array` — физические массивы |
| `atomnet.first_nn` | `MiniTransformer` (CausalSelfAttention/Block), **`AtomGPTTrainer`** (lr_at warmup+cosine/linear, zero_grad, checkpoints, eval, generate), `save/load_state_dict`, `train_aiiro_nano` |
| `atomnet.deploy` | `DeployServer` — авто-tier по `MemAvailable`, paged-LRU, HTTP-inference |
| `atomnet.datacenter` | solid-shards паковка, `StreamReader/StreamTrainer` — стриминговое обучение с RSS ≤ 0.03 GB |
| `atomnet.dc` / `dc.py` | демо DynGraph-дашборда (q/k/v/gate/out-граф, 1.2 ms) |
| `atomnet.luna_mini` | LunaMini — мини-LM с выбором backend (atomnet ↔ torch) |

---

## Что реализовано (ключевые возможности)

### 1. JIT без C: DynASM + AOT
- **GEMM 6×16** (12 YMM-аккумуляторов) — регистровый аллокатор написан нами,
  компилятор не выбрасывает регистры; включается `ATOMNET_NO_JIT=1` для
  честного scalar-фолбэка.
- **atomjit**: dot/axpy/norm + составные геодезические/Householder-ядра.
- **binjit**: hamming/xor/and/popcnt (sysv64 ABI, POPCNT+AVX2, иначе Rust-fallback).
- **AOT**: `aot_dump/aot_read/aot_import_dot` — дамп машинного кода в файл и
  обратная загрузка (бит-идентичность доказана тестом; runtime-исполнение
  импортированного буфера — известное ограничение, см. «Ограничения»).
- `ATOMNET_JIT_DUMP=1` — дамп скомпилированных ядер.

### 2. Честные замеры (на Ryzen 5 7430U, 6C/12T, AVX2+FMA)

**vs NumPy** (`speed_vs_numpy.csv`, 4M элементов):

| бенч | numpy, мс | AtomNET, мс | ускорение |
|---|---|---|---|
| mean+var | 8.20 | 0.69 | **11.9×** |
| mse | 13.48 | 1.62 | **8.3×** |
| quantize_int8 | 35.26 | 10.78 | 3.3× |
| dequantize_int8 | 13.50 | 3.70 | 3.7× |
| affine_into | 11.07 | 2.53 | 4.4× |
| rin fused step | 21.98 | 8.72 | 2.5× |
| xor_signature | 2.94 | 0.0003 | **×11294** (подпись, а не вычисление) |
| gemv int16 (Q16×Q16) | 1.34 | 0.85 | 1.6× |

**vs OpenBLAS** (чистый процесс, `taskset`, см. `docs/GEMM.md`):

| размер | OpenBLAS | AtomNET FLASH | дельта |
|---|---|---|---|
| 512³ | 1.39 мс (193 GFLOPS) | 1.59 мс (169 GFLOPS, 59% пика) | −12% |
| 1024³ | 11.3 мс | 13.7 мс | −21% |
| 2048³ | 88.4 мс | 112.7 мс | −27% |

Ядра эквивалентны OpenBLAS Haswell (тот же класс регистров, тот же
FMA/лоад-профиль ~0.67); расхождение — в масштабе (у нас JIT-ядро, не
половиногодовая тюнинг-параметризация). Это **честные** цифры, а не маркетинг.

### 3. Собственная математика: атомы, а не «веса»
Вес строки `w = n·1 + r·u` (nucleus + radius·direction на сфере) —
см. `docs/ATOM_MATH.md`: weight normalization + трансляционная
инвариантность, риманова инвариантность `SO(n)×R⁺×R`, тангенциальная
проекция Fubini–Study. На этом стоят Keralin (оптимизатор), Qairo
(линейный слой из атомов), SFEROM (ассоциативная память), Rineta (резонанс).

### 4. Оптимизатор Keralin
- Разложение градиента на ядерный/радиальный/тангенциальный — ортогональные
  подпространства, шаг идёт по «своей геометрии», а не по координатам.
- AdamW-паритет по обвеске: **decoupled weight decay** `×(1 − lr·wd)`,
  **bias-correction** `(1 − β^t)`, **grad clip**, RMS-адаптация (bias-corrected),
  warm-фаза до шага, детерминированный RNG прогрева.
- `KeralinGraph`: узлы-атомы + зависимости, topo-порядок, параллельный шаг
  (rayon, детерминированная f32-арифметика), жизнь узлов (`AtomLife`).
- **Реализовано дополнительно:** LR-schedule (linear warmup + cosine decay,
  `schedule_warmup/max_steps/min_frac`), **EMA-snapshot** весов (SWA-подобный),
  **NAdam-lookahead**, отладка `momentum_f16` (fp16-state, память ×½),
  исправлен первый шаг адаптации ( деление на нулевой RMS → взрыв 10⁵).

### 5. Квантование и память
- **Q4 solid-архив** (WinRAR-стиль): 14B fp32 52.5 GB → **20.9 GB** (2.51×),
  интеграл с int8-остатком = точность f32.
- **FP8 E4M3FN + ue8m0** (блок 128, `s = amax/448`) — формула сверена с
  DeepSeek `kernel.py`; аппаратная ошибка E4M3 ~7.5% (природа формата).
- **MLA-latent KV-cache**: 1.61 GB → 0.40 GB (4×; до 93% против GQA).
- **14B в 12 GB — подтверждено** (`docs/TWELVE_GB.md`, `tests/test_twelve_gb.py`):
  paged-LRU резидент 11.74 GB ≤ 12 GB (детерминированная таблица + RSS-эмпирика).
- **Иерархия RAM ↔ диск**: `DeployServer` (auto-tier по MemAvailable, paged LRU),
  `datacenter.py` (solid-shards, стриминг, 43–50% от raw, 7901 tok/s при 0.03 GB RSS).

### 6. Слой нейросетей (PyTorch-паритет)
- `torchx` autograd: аналитические backward проверены **numeric gradcheck**
  (18 тестов parity в `test_torchx_parity.py`).
- Полный набор слоёв (`test_new_layers.py`): Conv1d/2d сверяются с NumPy
  до 4.8e-7; pooling/pad/flatten/RNN/BatchNorm/GroupNorm.
- `TransformerLM` (RMSNorm, GQA+RoPE+causal, KV-cache), BPE-токенизатор.
- DeepSeek-порты: MoE (проверено: loss 2.94→0.019 на MiniTransformer),
  MLA/LatentCache, FP8.

### 7. Обучение и сервер
- **Rust-сторона**: `train.rs` (полный батч, аналитические backward, шаг
  KeralinGraph), бинарь `luna` + `prod` (curriculum-обучение).
- **Python**: `AtomGPTTrainer` — warmup+cosine, zero_grad, чекпоинты, eval, generate.
- **HTTP-сервер** `tests/demo_nnet_server.py`: веса живут в `.so`,
  `POST /train`, `/forward`, `/generate`, `/save`, `/load` — обучение по HTTP.
- **Обученные результаты**:
  - Airo-nano v0.0.1 (Phi-style): 6 эпох, loss 2.083 → **1.266**, стабильно;
  - char-LM на TinyShakespeare: 2.785 → 2.275 (3 эпохи), datacenter-pack 38.97%;
  - curriculum-математика TransformerLM (Rust): loss-конвергенция в тестах.

### 8. Ресурсы и детерминизм
- `flash_hw::thread_budget()`: `ATOMNET_THREADS=N` — **на 1 потоке всё
  работает** (N=1 → пул не поднимается); L3-кластеры, pin, huge pages.
- NPU-эмулятор: 1 тред = бит-в-бит тот же результат, что N потоков.
- `DynGraph` + Rayon work-stealing: динамические графы под JIT.

---

## Тесты

```
cargo test --release --offline --lib      # 107 passed, 0 failed, 6 ignored (~355 s)
```

Python (`tests/`, 49 файлов, 71 тест-функция + demo-самопроверки):

```
test_torchx_parity   test_atom_math      test_deepseek_port   test_deploy
test_14b_memory      test_npu_emu        test_twelve_gb       test_new_layers
test_corpus_train    test_keralin_graph  test_atom_codec      test_flash_* …
demo: first_nn, nnet_server, server_python (`--selftest` → SELFTEST OK),
      nnet_python, paged_net, vocab_bpe (запуск → `OK: …`),
      python -m atomnet.{dc,datacenter,deploy} --selftest
```

Игнорируемые Rust-тесты (`#[ignore]` — 6 шт.) — тяжёлые
параметрические замеры GEMM, запускаются вручную: `cargo test -- --ignored`.

---

## Ограничения (честно)

1. **AOT runtime-исполнение** импортированного буфера (`MutableBuffer::make_exec`)
   падает segfault'ом — тест проверяет бит-идентичность дампа, но код по
   импорту не исполняется. JIT-путь (собственная ассемблировка) работает.
2. **14B не помещается в физическую RAM этой машины** (15 GB) — доступны
   архитектура и бюджет (11.74 GB ≤ 12 GB), не «живой» прогона 14B.
3. **GEMM против OpenBLAS** на больших размерах отстаёт на 12–27% (см. выше) —
   наш выигрыш в других местах: интеграция без FFI, fused-ядра (bias+SiLU),
   Q4/Q16/FP8-пути, которых у типичного BLAS нет.
4. **FP8** даёт ~7.5% ошибки — это природа E4M3, не дефект реализации.
5. **Нет** distributed/AMP/double-backward — CPU-only и f32 по замыслу;
   RNG не побитово совпадает с mt19937 PyTorch.
6. Keralin **не побеждает AdamW везде**: на квадратичной задаче 200 шагов
   AdamW доводит loss до 0.19, сильные стороны Keralin — параллельный
   графовый шаг (1.46 мс/шаг на 512 узлах) и своя геометрия;
   честное сравнение — в `docs/COMPARE_AUDIT.md`.

---

## Переменные окружения

| Переменная | Назначение |
|---|---|
| `ATOMNET_THREADS` | бюджет потоков (1 = один поток, без пула) |
| `ATOMNET_NO_JIT` | отключить DynASM-JIT (scalar-фолбэк) |
| `ATOMNET_NO_WARMUP` | без прогрева GEMM (cold-start замеры) |
| `ATOMNET_JIT_DUMP` | дамп JIT-ядер |
| `ATOMNET_AOT_PATH` | путь AOT-дампа ядер |
| `ATOMNET_GEMM_MC/KC/NC` | тайлинг GEMM (BLOCK-M/K/N) |
| `ATOMNET_GEMM_PACKA` | эксперимент: interleaved A-панель (по умолчанию off) |
| `ATOMNET_GEMM_PROFILE/OVER` | профиль/оверлей ядра |
| `ATOMNET_NPU_EMU/CORES/PROFILE` | NPU-эмулятор: вкл, число ядер, профиль |
| `ATOMNET_ISA_TIER` | форс виртуального ISA (`scalar` для тестов) |

---

## Документация

| Файл | Тема |
|---|---|
| `docs/ATOM_MATH.md` | формальный каркас атомной сферической математики |
| `docs/GEMM.md` | разбор ядра OpenBLAS Haswell и паритет с FLASH-GEMM |
| `docs/LLM.md` | BPE, train/val, LR-расписание — аудит против nanoGPT/HF |
| `docs/DEEPSEEK.md` | разбор DeepSeek-V3/MLA/MoE/FP8 и что портировано |
| `docs/TWELVE_GB.md` | как 14B укладывается в 12 GB (таблица 11.74 GB) |
| `docs/COMPARE_AUDIT.md` | честный SWOT против PyTorch/JAX/TF + бэклог улучшений |
| `docs/ARRAY.md` | AtomArray — физическая замена NumPy (API, скорость, ограничения M0) |

---

## Стек и лицензия

Rust 2021 · pyo3 0.25 · numpy 0.25 · dynasmrt 5.1 · rayon 1.10 · rand_chacha 0.3
· maturin (сборка расширения) · Python ≥ 3.10.

**Только Rust — ни одной строки C/CUDA. Только CPU (AVX2+FMA). Только float32.**
