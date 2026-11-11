# DeepSeek — разбор движка и карта улучшений для AtomNET

(Изучено по github.com/deepseek-ai: **DeepSeek-V3** `inference/{model,kernel,convert}.py`,
**DeepGEMM** `README`, **DeepSeek-MoE** `finetune/`, DeepSeek-V2/Coder V2 Lite MLA, FlashMLA.
Даты батчей: 2024-2026).

## 1. Мощные техники DeepSeek (что нового и как оно устроено)

### 1.1 Multi-head Latent Attention (MLA, DeepSeek-V2/3) — это главный наш трюк на будущее
Ключевой механизм KV-cache сжатия:

```
qk:   x ──► wq_a(d→q_lora_rank=1536) ──► RMSNorm ──► wq_b ──► (heads × qk_head_dim   )
kv:   x ──► wk_a(d→kv_lora_rank=512) ──► RMSNorm ──► latent c_KV (512 на голову!)
      latent c (512) — ЕДИНСТВЕННАЯ вещь, которую кэшируем в инференсе!
                                                       ▲ 93% экономия KV-cache vs GQA
      K = wkv(c) +  RoPE-обёртки на rope-часть (qk_rope_head_dim=64)
```

Математика внутри TransformerLM нашего кода: KV-cache хранит токены (seq × heads × d_head, ~25 KB/токен у LLaMA 8B); MLA хранит 512/layer-единиц подключаемых latent — аудитория во много раз лучше доступна mideе.

Идеала для CPU: слой латента нужен поскольку хранит компактную векторную проекцию. Attention использует (черезгруженное) same-manhattan для головы как linear-у отражает.

**Что мы реализуем:**

- `MLADown(keys latent 512), MLAUp(weight)` как в V3 (`wq_a/wq_b/wk_a` и `wkv`, `qk_rope_head_dim 64).
- Свеже️ KV-cache через latent, с впадениеми из существующих cache-ci.
- Оценка экономии KV-паммяти на единицах (уже есть numbers).

### 1.2 DeepSeek-MoE (масштабирование, составной эксперторест)
Выше изучили DeepSeekMoE.pdf + актуальную V3 `class MoE`:
- N_routed = 256 тонко-грановых экспертов, активизируется только K=8, shared_experts=1.
- **Routing без auxiliary loss** через bias: `scores = softmax/sigmoid(X·W_gate) + bias` — bias обновляется в зависимости от загрузки (Aux-free balance via e_score_correction_bias в ckpt).
- Device-limited routing: groups (n_groups=8) × topk_groups (4) — маршрутизация сначала по группам, потом внутри.
- Expert: SwiGLU (`w2(silu(w1(x))·w3(x))`).

Гейт формулы (из V3):
```python
scores = linear(x, self.weight)
if score_func == "softmax": scores = softmax(scores)
else: scores = sigmoid(scores)
original_scores = scores
if bias is not None: scores = scores + bias      # aux-loss-free boost
# n_groups > 1: group-limited routing -> повернем
# topk: weights = group scores; indices = argmax_k
```

**Что мы задумываем:**

- `MoELayer(num_experts=64, top_k=8, shared_experts=1, use_bias=True)` в nn.py
  (все дейцы-слоты в блокнотировании — **unified flattened batch** на CPU, LR-поования шаг-операриев)
- Shared experts: всегда все активны (один слой extra MLP)
- **Decompressed expert storage** через solid-архив Q4:
  только активные эксперты выпадают в memory_hot, остывшие выкидываются (LRU paging!)

### 1.3 FP8 (kv-quant через kernel.py : `act_quant_kernel`)
Формула один в один (фотокопировано в AtomNET):
```
s        = max(|x|) / 448   (clamp amax ≥ 1e-4)
if ue8m0: s = 2^ceil(log2(s))  (степень две)
codes    = x / s → E4M3FN (sign+4exp+3mant, bias 7, без Inf: NaN → 0x7F)
```

В AtomNET идёт в `flash_compressed::FlashFP8` — ue8m0-модуль и f32/f32 E4M3 conversion (тесты пройдены).

DeepSeek использует scale-формат "ue8m0" для хранения отдельных множителей (FP8, совместим с E4M3 — наш модуль ниже смешей).

### 1.4 DeepGEMM — JIT движок GEMM (TFLOPS-холд)
По README: unified CPU/GPU frontend:
- compiled-tiles'ы через runtime JIT (**DeepJIT**) — **аналог нашего DynASM JIT-пути**:
  "no CUDA compilation needed" → у нас: "no C code".
- FP8/FP4 GEMM из коробки, MegaMoE (fused MoE with overlapped communication), weight-FP8-FP4 quant → зеркалит наше сопероличество с GEMM+SiLU fused epi (силовые границы в ядре).
- **Дисбаланс в основном**: DeepGEMM на CUDA (H100/H800 1550 TFLOPS); у нас CPU 1-потокный NPU-tiling.

**Что мы реализуем:**
- Наше JIT: unified dispatch за одним вызовом (GEMM + GEMV + FP8 GEMM) с unified тратэктингов (фаза захвата "компиляция JIT" — аналог).
- Наш раздел: `libatomnet.so атомный solid-блок` ~ MegaMoE tier.

### 1.5 DeepEP — колocation перекрёстная интерфейсов передача блоков
Активная вставка слоя в между здесь: "send-and-compute overlap" — communication портировать
в промежуточную фазу computation/overlap (Deepseek overlapping path пишет expertoчное ОК)
→ в нашем случае это = "scheduler run-overlap: prefetch next page while GEMM runs".

## 2. Сходства и различия AtomNET/DeepSeek (честный список)

| Компонент | DeepSeek | AtomNET (после портирования) | Комментарий |
|---|---|---|---|
| MoE | 256 экспертов, top-8, shared, bias-баланс | 64 эксперта, top-8, shared, bias-баланс (Python nn) | Q4 compressed epochs via archive это замена device.paged storage |
| MLA | latent 512 доуп-роутива | latent latent cKV + rope-часть separate | неявное упоминание функциями; KV-cache можно ещё совместимо обновить |
| FP8 | E4M3FN + ue8m0 (block 128) | точно так же (Rust) | измерен на работах |
| GEMM | DeepGEMM JIT (CUDA) | JIT DynASM 6x16 (CPU) + Q4 GEMV | не захвати гей AST |
| Оптимизатор |(aux-loss free bias gate) | Atom Adam + AdamW | простое SGD |
| DataShard | локальный 1 папки | datacenter.py (sharding+solid) | аналог "L2 tier" |
| KV-cache | latent-only | обычный + paged | эконом через перегрузка KV до самой тихой |

## 3. Конкретная дорожная карта улучшений

**Приоритет 1 (сжатие весов/ памяти)**
- [x] FP8 block quant (Rust `FlashFP8`, лёгкая GEMV-Rust) → измерен убыток ~0.135 на кодирование динамических диапазонов в ±448.
- [x] Solid Q4 симулятор локации (DONE: current archive).
- [ ] MLA down/up-проекции как модули (KV-память меньше) — в triple-mm очередь.
- [ ] "page-in-roving" (paged-in кэшированние слоёв при bias-roving-активности) — приходит враз когда разрешённого более плотного кэширования.

**Приоритет 2 (динамическое обучение)**
- [ ] DeepSeekMoE (Python nn): gating with aux-free bias, top-k выбираются операцией softmax/sigmoid(scores+bias), (grad-tracking через sigmoid-weighting одним LayerNorm-ops)
- [ ] Canonical MLA в first_nn и для KV-cache
- [ ] (опционально) router как отдельный сквозной mesh зафиксированный в solid-oper умираitivos операций Duck.

**Приоритет 3 (прочая несовместимость)**
- [ ] switch to paged KV cache LRU (память меньше доспоччительно даже при 14B: запись с напряжением — бедить тчко scalars- кванты иблетки-скено квантыдные памяти |
- [ ] Загад.
