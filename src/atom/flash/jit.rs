// src/atom/flash/jit.rs
//! JIT-ядра FLASH на DynASM (`dynasmrt`).
//!
//! Идея: горячее микроядро GEMM генерируется в машинный код в рантайме,
//! поэтому регистровый аллокатор — наш, а не LLVM: 6 строк × 16 колонок =
//! 12 YMM-аккумуляторов + 2 регистра под B + 1 под broadcast A. Всё
//! помещается в 16 YMM без спилов (у intrinsics-версии 6×16 компилятор
//! спиливал регистры на стек и терял 30–40%).
//!
//! ABI (System V, Linux x86-64):
//!
//! ```text
//! rdi = A (указатель на строку 0, колонку pc)
//! rsi = B (упакованная панель, [kk][j])
//! rdx = C (строка 0, колонка jc+j)
//! rcx = a_stride (в float; шаг строки A = исходный k)
//! r8  = b_stride (в float; шаг строки панели B = nc_pad)
//! r9  = c_stride (в float; шаг строки C = cols)
//! [rsp+8] = kc (число k-шагов)
//! ```
//!
//! Ядро считает ровно один тайл 6×16 и пишет (store), накапливает (add)
//! или сразу считает `silu(C + bias)` (fused-эпилог) в C. Внешние циклы по
//! строкам/колонкам/панелям остаются в Rust — JIT-код остаётся маленьким и
//! не содержит ветвлений, кроме k-цикла. Константы e^x fused-эпилога лежат
//! в статической таблице (`MATH_TABLE`), эпилог читает их одной
//! vpbroadcastd [r11+off] на константу. Там же — константы elementwise-ядер
//! SiLU, softmax и 8 элементарных ядер (add/mul/scale/scale_to/axpy/axpby/
//! relu/affine): все они генерируются в машинный код (AVX2), разворачиваются
//! по 16/8 float, хвосты всегда через vmaskmovps. В DynASM форм
//! vbroadcastss ymm из памяти зернит верхние 128 бит — здесь только
//! vpbroadcastd (отловлено на регрессе, см. elem_jit_vs_intrinsics_all_len).
//!
//! Доступность: только x86_64 + AVX2 + FMA. Иначе — обычный intrinsics-путь.
//! Отключить JIT можно переменной окружения `ATOMNET_NO_JIT=1`.
//!
//! ## Архитектурная карта OpenBLAS (изучено 2026-11-08, на что сверены решения)
//!
//! Ядро Haswell (`KERNEL.HASWELL` → `sgemm_kernel_8x4_haswell_2.c`):
//! микротайл 8m×4n; на k-шаг — **один** A-vector 32 B (`vmovups(%0)` на
//! упакованную панель [k][m-группы×8]) + `vmovsldup`/`vmovshdup` (2 шафла) +
//! 2× `vbroadcastsd` из B-панели `[kk][4-n-группы]` + 4× `vfmadd231ps`.
//! Аккумуляторы interleaved (m×n пары в регистрах), раскладка затем
//! исправляется shuffle-SAVE (`vunpcklps/...`), k-unroll×2, и это всё
//! **2 строки кэша на шаг** против наших 7 (6 A-broadcast'ов + B-строка).
//! A-паккер (`gemm_ncopy_8.c`): панели [k-группы][m-группы·8] — 32 B
//! при операции pack-входа. B-паккер (`sgemm_ncopy_4_skylakex.c`):
//! 4×4 транспозы → панель [k][4n] — **эквивалентна нашей** [kk][n].
//! `sgemm_beta_skylakex.c`: BETA-патежка C через Memset при β=0 (у нас —
//! Store-экземплляр эпилога). SAVE-эпилог fused `alpha·C` (у нас то же
//! SiluAdd-фузия, но silu+bias вместо alpha-счёта).
//!
//! Драйвер (`driver/level3/level3.c`): цикл Jc→Ls→Is со значением canonical
//! GotoBLAS blocking (весь порядок, ICOPY/OCOPY/KERNEL). OpenBLAS MC/KC/NC
//! на Haswell — эмпирические числа; мы подобрали MC=96/KC=128/NC=256 под
//! Zen3 (они — аналог, не оригинал).
//!
//! Threading (`driver/level3/gemm_thread_mn.c`): два-dimensional partition
//! по `divide_rule[nthreads]`: 6 тредов → (divM=2, divN=3), 4 → (2,2),
//! 12 → (3,4). КООРДИНАТЫ ± равномерно убывают. Наш пул делит только rows
//! (1×6), потому что DRAM-трафик B инвариантен к разделению
//! (rows/MC·B_total независимо отAllocation при row-major); выиграш 2D
//! ограничивается локальностью L3. Не реализован: измеренный выигрыш на
//! AMD с нашим MC-block = не показывать в пользу 1D.
//!
//! Решены перенесённые уроки:
//! - B-панели [kk][n] (из их `sgemm_ncopy_4_skylakex`) — принята; она и
//!   закрыла её нам 25% плюс разрыв abspath-агоди, и вручила
//!   `mc_block3 · kcbatch = 8-KiB L1-тайл чтобы он лежал на плотной
//!   1-KiB сетке (`gemm_ncopy_4_skylakex`» (odd-line stride)») — то же
//!   результата передача");
//! - Ин supervisor

#[cfg(target_arch = "x86_64")]
mod imp {
    use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi, ExecutableBuffer};
    use std::sync::OnceLock;

    /// Единая таблица float-констант для всей ASM-математики.
    ///
    /// Раскладка (байтовые смещения):
    ///   0..24   — broadcast-константы (sign, clamp_lo/hi, log2e, one, 127, -inf);
    ///   32..216 — 6 векторов коэффициентов 2^z (по 8 копий, 32 байта);
    ///   224..480 — 8 векторов-масок для хвоста (lane < r = все единицы);
    ///   480..736 — 8 векторов «OR-хвост» (lane >= r = -inf).
    ///
    /// Эпилог GEMM и elementwise-ядра грузят константы одной
    /// `vpbroadcastd [r11+off]` / `vmovups [r11+off]`, без mov imm32.
    #[repr(align(64))]
    struct MathTable([i32; 192]);

    // Байтовые смещения в MathTable (чтобы dynasm писал [r11 + MT_*]).
    const MT_SIGN: i32 = 0;
    const MT_CLAMP_LO: i32 = 4;
    const MT_CLAMP_HI: i32 = 8;
    const MT_LOG2E: i32 = 12;
    const MT_ONE: i32 = 16;
    const MT_I127: i32 = 20;
    const MT_NEG_INF: i32 = 24;
    const MT_C1: i32 = 32;
    const MT_C2: i32 = 64;
    const MT_C3: i32 = 96;
    const MT_C4: i32 = 128;
    const MT_C5: i32 = 160;
    const MT_C6: i32 = 192;
    const MT_MASK: i32 = 224;
    const MT_TAIL_OR: i32 = 480;
    // Дополнительные f32-константы (736..744) — для Q4×Q8 поправок.
    // ВАЖНО: 480..736 занято таблицей MT_TAIL_OR (не перезаписывать!).
    const MT_M256: i32 = 736;
    const MT_M8: i32 = 740;
    const MT_M128: i32 = 744;

    const fn build_math_table() -> MathTable {
        let mut a = [0i32; 192];
        a[MT_SIGN as usize / 4] = i32::MIN; // 0x80000000: x xor sign = -x
        a[MT_CLAMP_LO as usize / 4] = (-88.0f32).to_bits() as i32;
        a[MT_CLAMP_HI as usize / 4] = (88.0f32).to_bits() as i32;
        a[MT_LOG2E as usize / 4] = (1.4426950408889634f32).to_bits() as i32;
        a[MT_ONE as usize / 4] = (1.0f32).to_bits() as i32;
        a[MT_I127 as usize / 4] = 127;
        a[MT_NEG_INF as usize / 4] = f32::NEG_INFINITY.to_bits() as i32;

        // 2^z = Σ (z·ln2)^k/k!; z ∈ [-0.5, 0.5] после range reduction.
        let c = [
            0.6931471805599453f32,  // ln2
            0.2402265069591007,     // ln2²/2!
            0.05550410866482158,    // ln2³/3!
            0.009618129107628477,   // ln2⁴/4!
            0.0013333558146428443,  // ln2⁵/5!
            0.00015403530393381608, // ln2⁶/6!
        ];
        let mut k = 0;
        while k < 6 {
            let bits = c[k].to_bits() as i32;
            let mut lane = 0;
            while lane < 8 {
                a[(MT_C1 as usize + 32 * k) / 4 + lane] = bits;
                lane += 1;
            }
            k += 1;
        }

        let mut r = 0;
        while r < 8 {
            let mut lane = 0;
            while lane < 8 {
                a[MT_MASK as usize / 4 + r * 8 + lane] = if lane < r { -1 } else { 0 };
                a[MT_TAIL_OR as usize / 4 + r * 8 + lane] = if lane < r {
                    0
                } else {
                    f32::NEG_INFINITY.to_bits() as i32
                };
                lane += 1;
            }
            r += 1;
        }
        a[MT_M256 as usize / 4] = (256.0f32).to_bits() as i32;
        a[MT_M8 as usize / 4] = (8.0f32).to_bits() as i32;
        a[MT_M128 as usize / 4] = (128.0f32).to_bits() as i32;
        MathTable(a)
    }

    static MATH_TABLE: MathTable = build_math_table();

    /// Вид эпилога JIT-ядра (match → jump table в сгенерированном коде).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Epilogue {
        Store,
        Add,
        Silu,
        SiluAdd,
    }

    /// Сигнатура сгенерированного микроядра (System V AMD64).
    pub type GemmKernel = unsafe extern "sysv64" fn(
        *const f32, // A
        *const f32, // B (packed)
        *mut f32,   // C
        usize,      // a_stride (float)
        usize,      // b_stride (float)
        usize,      // c_stride (float)
        usize,      // kc
    );

    pub struct JitGemm {
        // Буферы обязаны жить, пока живы указатели на код.
        _store_buf: ExecutableBuffer,
        _add_buf: ExecutableBuffer,
        _silu_buf: ExecutableBuffer,
        _siluadd_buf: ExecutableBuffer,
        /// Варианты под interleaved-A (A-панель [kk][6 строк] = 32 B на шаг,
        /// ближе к раскладке ядер OpenBLAS: 1 строка кэша на k-шаг вместо 6).
        _store_il_buf: ExecutableBuffer,
        _add_il_buf: ExecutableBuffer,
        _silu_il_buf: ExecutableBuffer,
        _siluadd_il_buf: ExecutableBuffer,
        pub store: GemmKernel,
        /// C += A×B (для k-панелей после первой).
        pub add: GemmKernel,
        pub store_silu: GemmKernelSilu,
        pub add_silu: GemmKernelSilu,
        /// Те же store/add, приведённые к 8-аргументной сигнатуре: лишний
        /// bias-arg игнорируется. Нужны для табличного диспетчера без
        /// ветвлений в горячем цикле тайлов.
        pub store_any: GemmKernelSilu,
        pub add_any: GemmKernelSilu,
        /// Interleaved-A варианты (B идёт туда же, epilogue один).
        pub store_any_il: GemmKernelSilu,
        pub add_any_il: GemmKernelSilu,
        pub store_silu_il: GemmKernelSilu,
        pub add_silu_il: GemmKernelSilu,
    }

    /// Ядро с fused-эпилогом: C = silu(A×B + bias) (bias — 16 float на тайл,
    /// extra-`None` не предусмотрен: вызывающий обеспечивает буфер bias).
    pub type GemmKernelSilu = unsafe extern "sysv64" fn(
        *const f32, // A
        *const f32, // B (packed)
        *mut f32,   // C
        usize,      // a_stride (float)
        usize,      // b_stride (float)
        usize,      // c_stride (float)
        usize,      // kc
        *const f32, // bias (≥16 float)
    );

    // Код неизменяем после finalize; буферы не содержат внутренней мутабельности.
    unsafe impl Send for JitGemm {}
    unsafe impl Sync for JitGemm {}

    static GEMM: OnceLock<Option<JitGemm>> = OnceLock::new();
    static ENABLED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(true);

    /// Включить/выключить JIT на лету (бенчмарки, отладка).
    pub fn set_jit_enabled(v: bool) {
        ENABLED.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn jit_enabled() -> bool {
        ENABLED.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// JIT-ядро GEMM или None (не x86_64/нет AVX2+FMA/отключено/env).
    pub fn jit_gemm() -> Option<&'static JitGemm> {
        if !jit_enabled() {
            return None;
        }
        GEMM
            .get_or_init(|| {
                if std::env::var_os("ATOMNET_NO_JIT").is_some() {
                    return None;
                }
                if !crate::atom::isa::virt_isa().has_avx2_fma() {
                    // Единый виртуальный ISA-гейт (ATOMNET_ISA_TIER для тестов).
                    return None;
                }
                // AOT: маппим машинный код с патчингом таблицы вместо сборки.
                if let Some(path) = aot_path_from_env() {
                    if let Some(j) = try_import_gemm(&path) {
                        set_source(2);
                        return Some(j);
                    }
                }
                set_source(1);
                // mode: 0 = store, 1 = add, 2 = fused bias+silu, 3 = add fused bias+silu.
                // Интерливинг ((_il) = A-панель [kk][6 строк]): A читается
                // одной 32-байтной строкой на k-шаг (как ядро Haswell у
                // OpenBLAS), 6 строк идёт.broadcast'ами из неё.
                let store = build_variant(Epilogue::Store, false);
                let add = build_variant(Epilogue::Add, false);
                let silu = build_variant(Epilogue::Silu, false);
                let siluadd = build_variant(Epilogue::SiluAdd, false);
                let store_il = build_variant(Epilogue::Store, true);
                let add_il = build_variant(Epilogue::Add, true);
                let silu_il = build_variant(Epilogue::Silu, true);
                let siluadd_il = build_variant(Epilogue::SiluAdd, true);
                match (
                    store,
                    add,
                    silu,
                    siluadd,
                    store_il,
                    add_il,
                    silu_il,
                    siluadd_il,
                ) {
                    (
                        Ok((sb, sf)),
                        Ok((ab, af)),
                        Ok((lb, lf)),
                        Ok((qb, qf)),
                        Ok((sb2, sf2)),
                        Ok((ab2, af2)),
                        Ok((lb2, lf2)),
                        Ok((qb2, qf2)),
                    ) => Some(JitGemm {
                        _store_buf: sb,
                        _add_buf: ab,
                        _silu_buf: lb,
                        _siluadd_buf: qb,
                        _store_il_buf: sb2,
                        _add_il_buf: ab2,
                        _silu_il_buf: lb2,
                        _siluadd_il_buf: qb2,
                        // store/add собраны как тот же код (8-й стековый arg
                        // игнорируется) — безопасно смотреть как 7/8-аргументные.
                        store: unsafe { std::mem::transmute::<_, GemmKernel>(sf) },
                        add: unsafe { std::mem::transmute::<_, GemmKernel>(af) },
                        store_silu: lf,
                        add_silu: qf,
                        store_any: sf,
                        add_any: af,
                        store_any_il: sf2,
                        add_any_il: af2,
                        store_silu_il: lf2,
                        add_silu_il: qf2,
                    }),
                    _ => None,
                }
            })
            .as_ref()
    }

    /// mode: 0 = store, 1 = add, 2 = fused bias+silu, 3 = add fused bias+silu.
    /// `il`: interleaved-A (A-панель [kk][6 строк], шаг k = 32 B).
    fn build_variant(kind: Epilogue, il: bool) -> Result<(ExecutableBuffer, GemmKernelSilu), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();

        // ── Пролог: сохранить callee-saved, разложить аргументы ──
        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push rbp
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            // 7-й аргумент (kc) лежит на стеке: [rsp+8] на входе,
            // после 6 push'ей — [rsp+56].
            ; mov r10, [rsp + 56]
            // Шаги из float в байты.
            ; shl rcx, 2
            ; shl r8, 2
            ; shl r9, 2
            // Указатели на 6 строк A (rdi — строка 0).
            ; mov r12, rdi
            ; add r12, rcx
            ; mov r13, r12
            ; add r13, rcx
            ; mov r14, r13
            ; add r14, rcx
            ; mov r15, r14
            ; add r15, rcx
            ; mov rbx, r15
            ; add rbx, rcx
            // r11 = смещение по k в байтах (0).
            ; xor r11d, r11d
            // Счётчики: rbp = main_iters = kc>>3, r10 = tail = kc&7.
            // Аккумуляторы НЕ zero'ятся: vfmadd231 сначала читает их как
            // источник, но первые 4 шага перезаписывают? Нет: FMA читает
            // acc как вход (acc = acc + a*b), поэтому zero-init обязателен
            // только один раз — оставляем, но НЕ делаем в epilogue.
            // NOTE: vxorps на регистрах нужен, т.к. FMA читает acc.
            // Оставляем zero-init (12 инструкций, один раз на тайл).
            ; vxorps ymm0, ymm0, ymm0
            ; vxorps ymm1, ymm1, ymm1
            ; vxorps ymm2, ymm2, ymm2
            ; vxorps ymm3, ymm3, ymm3
            ; vxorps ymm4, ymm4, ymm4
            ; vxorps ymm5, ymm5, ymm5
            ; vxorps ymm6, ymm6, ymm6
            ; vxorps ymm7, ymm7, ymm7
            ; vxorps ymm8, ymm8, ymm8
            ; vxorps ymm9, ymm9, ymm9
            ; vxorps ymm10, ymm10, ymm10
            ; vxorps ymm11, ymm11, ymm11
            ; mov rbp, r10
            ; shr rbp, 3
            ; and r10, 7
            ; test rbp, rbp
            ; jz ->tailloop
        );

        // ── Основной цикл: unroll×8, безопасный prefetch ----------
        // Prefetch B на 16 k-шагов вперёд ТОЛЬКО если гарантированно
        // внутри панели: при rbp > 2 остаётся ≥ 24 строк — prefetch
        // целит в +16, что < kc. Иначе (последние группы) — без prefetch,
        // исключает чтение за концом буфера (overread-баг доработки).
        // Unroll×8: двойная глубина конвейерирования L2-латентности
        // A-broadcasts (6 L1-miss на k-шаг, MSHR-лимит) — FMA не ждёт.
        // Шаг k в ядре (байты). Scattered A (row-major): 4 (след. float
        // строки). Interleaved A (OpenBLAS-зеркало, запись [kk][6 строк]):
        // 32 (след. запись = 6 строк + 2 padding). Строки A берутся
        // broadcast'ами из [rdi + r11 + 4r] — одна кэш-линия вместо 6.
        macro_rules! kstep {
            ($s:literal) => {
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm12, [rsi]
                    ; vmovups ymm13, [rsi + 32]
                    ; vpbroadcastd ymm14, [rdi + r11]
                    ; vfmadd231ps ymm0, ymm14, ymm12
                    ; vfmadd231ps ymm1, ymm14, ymm13
                    ; vpbroadcastd ymm14, [r12 + r11]
                    ; vfmadd231ps ymm2, ymm14, ymm12
                    ; vfmadd231ps ymm3, ymm14, ymm13
                    ; vpbroadcastd ymm14, [r13 + r11]
                    ; vfmadd231ps ymm4, ymm14, ymm12
                    ; vfmadd231ps ymm5, ymm14, ymm13
                    ; vpbroadcastd ymm14, [r14 + r11]
                    ; vfmadd231ps ymm6, ymm14, ymm12
                    ; vfmadd231ps ymm7, ymm14, ymm13
                    ; vpbroadcastd ymm14, [r15 + r11]
                    ; vfmadd231ps ymm8, ymm14, ymm12
                    ; vfmadd231ps ymm9, ymm14, ymm13
                    ; vpbroadcastd ymm14, [rbx + r11]
                    ; vfmadd231ps ymm10, ymm14, ymm12
                    ; vfmadd231ps ymm11, ymm14, ymm13
                    ; add rsi, r8
                    ; add r11, $s
                );
            };
        }
        // Помощник-селектор: выбор шага по il в точке эмиссии (это
        // runtime-ветвление строителя, не ядра).
        macro_rules! kstep_v {
            () => {{
                if il { kstep!(32); } else { kstep!(4); }
            }};
        }
        dynasm!(ops
            ; .arch x64
            ; ->mainloop:
            ; cmp rbp, 2
            ; jbe ->noprefb
            ; prefetcht0 [rsi + r8*8]
            ; ->noprefb:
        );
        // Префетч A на 512 B вперёд только для interleaved (Haswell-ядро
        // OpenBLAS: prefetcht0 512(%0)); prefetch не фейлится, поэтому
        // за концом панели не страшно. Отдельным блоком (стейтмент
        // эмиссии — runtime if строителя допустим).
        if il {
            dynasm!(ops
                ; .arch x64
                ; cmp rbp, 2
                ; jbe ->noprefa
                ; prefetcht0 [rdi + r11 + 512]
                ; ->noprefa:
            );
        }
        kstep_v!();
        kstep_v!();
        kstep_v!();
        kstep_v!();
        kstep_v!();
        kstep_v!();
        kstep_v!();
        kstep_v!();
        dynasm!(ops
            ; .arch x64
            ; dec rbp
            ; jnz ->mainloop
            ; ->tailloop:
            ; test r10, r10
            ; jz ->done
            ; ->tailinner:
        );
        kstep_v!();
        dynasm!(ops
            ; .arch x64
            ; dec r10
            ; jnz ->tailinner
            ; ->done:
        );

        // ── Эпилог: store / add / bias+silu ──
        // Диспетчеризация через match (компилятор делает jump table), а
        // float-константы e^x лежат в общей MathTable:
        // одна vpbroadcastd [r11+off] вместо mov+vmovd+vbroadcastss.
        // r10 = &bias (16 float тайла), r11 = база таблицы констант.
        //
        // Полином 2^z (z ∈ [-0.5,0.5]) считается FMA-цепочкой: 6
        // vfmadd213ps вместо 12 mul+add и без отдельного умножения на ln2.
        // Все шаги — в развёрнутом виде, без вызовов и ветвлений.
        macro_rules! silu_acc {
            ($acc:ident, $boff:expr) => {
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [r10 + $boff]
                    ; vaddps $acc, $acc, ymm15
                    ; vpbroadcastd ymm15, [r11 + MT_SIGN]
                    ; vxorps ymm15, $acc, ymm15
                    ; vpbroadcastd ymm14, [r11 + MT_CLAMP_LO]
                    ; vmaxps ymm15, ymm15, ymm14
                    ; vpbroadcastd ymm14, [r11 + MT_CLAMP_HI]
                    ; vminps ymm15, ymm15, ymm14
                    ; vpbroadcastd ymm14, [r11 + MT_LOG2E]
                    ; vmulps ymm15, ymm15, ymm14
                    ; vcvtps2dq ymm14, ymm15
                    ; vcvtdq2ps ymm13, ymm14
                    ; vpbroadcastd ymm12, [r11 + MT_I127]
                    ; vpaddd ymm12, ymm14, ymm12
                    ; vpslld ymm12, ymm12, 23
                    ; vsubps ymm13, ymm15, ymm13
                    ; vmovups ymm14, [r11 + MT_C6]
                    ; vfmadd213ps ymm14, ymm13, [r11 + MT_C5]
                    ; vfmadd213ps ymm14, ymm13, [r11 + MT_C4]
                    ; vfmadd213ps ymm14, ymm13, [r11 + MT_C3]
                    ; vfmadd213ps ymm14, ymm13, [r11 + MT_C2]
                    ; vfmadd213ps ymm14, ymm13, [r11 + MT_C1]
                    ; vpbroadcastd ymm15, [r11 + MT_ONE]
                    ; vfmadd213ps ymm14, ymm13, ymm15
                    ; vmulps ymm14, ymm14, ymm12
                    ; vaddps ymm15, ymm14, ymm15
                    ; vdivps $acc, $acc, ymm15
                );
            };
        }
        match kind {
            Epilogue::Silu => {
                let tab = MATH_TABLE.0.as_ptr() as i64;
                dynasm!(ops
                    ; .arch x64
                    ; mov r10, [rsp + 64]
                    ; mov r11, QWORD tab
                );
                silu_acc!(ymm0, 0);
                silu_acc!(ymm1, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rdx], ymm0
                    ; vmovups [rdx + 32], ymm1
                    ; lea rax, [rdx + r9]
                );
                silu_acc!(ymm2, 0);
                silu_acc!(ymm3, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm2
                    ; vmovups [rax + 32], ymm3
                    ; add rax, r9
                );
                silu_acc!(ymm4, 0);
                silu_acc!(ymm5, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm4
                    ; vmovups [rax + 32], ymm5
                    ; add rax, r9
                );
                silu_acc!(ymm6, 0);
                silu_acc!(ymm7, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm6
                    ; vmovups [rax + 32], ymm7
                    ; add rax, r9
                );
                silu_acc!(ymm8, 0);
                silu_acc!(ymm9, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm8
                    ; vmovups [rax + 32], ymm9
                    ; add rax, r9
                );
                silu_acc!(ymm10, 0);
                silu_acc!(ymm11, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm10
                    ; vmovups [rax + 32], ymm11
                );
            }
            Epilogue::SiluAdd => {
                let tab = MATH_TABLE.0.as_ptr() as i64;
                dynasm!(ops
                    ; .arch x64
                    ; mov r10, [rsp + 64]
                    ; mov r11, QWORD tab
                    ; vmovups ymm15, [rdx]
                    ; vaddps ymm0, ymm0, ymm15
                );
                silu_acc!(ymm0, 0);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [rdx + 32]
                    ; vaddps ymm1, ymm1, ymm15
                );
                silu_acc!(ymm1, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rdx], ymm0
                    ; vmovups [rdx + 32], ymm1
                    ; lea rax, [rdx + r9]
                    ; vmovups ymm15, [rax]
                    ; vaddps ymm2, ymm2, ymm15
                );
                silu_acc!(ymm2, 0);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [rax + 32]
                    ; vaddps ymm3, ymm3, ymm15
                );
                silu_acc!(ymm3, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm2
                    ; vmovups [rax + 32], ymm3
                    ; add rax, r9
                    ; vmovups ymm15, [rax]
                    ; vaddps ymm4, ymm4, ymm15
                );
                silu_acc!(ymm4, 0);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [rax + 32]
                    ; vaddps ymm5, ymm5, ymm15
                );
                silu_acc!(ymm5, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm4
                    ; vmovups [rax + 32], ymm5
                    ; add rax, r9
                    ; vmovups ymm15, [rax]
                    ; vaddps ymm6, ymm6, ymm15
                );
                silu_acc!(ymm6, 0);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [rax + 32]
                    ; vaddps ymm7, ymm7, ymm15
                );
                silu_acc!(ymm7, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm6
                    ; vmovups [rax + 32], ymm7
                    ; add rax, r9
                    ; vmovups ymm15, [rax]
                    ; vaddps ymm8, ymm8, ymm15
                );
                silu_acc!(ymm8, 0);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [rax + 32]
                    ; vaddps ymm9, ymm9, ymm15
                );
                silu_acc!(ymm9, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm8
                    ; vmovups [rax + 32], ymm9
                    ; add rax, r9
                    ; vmovups ymm15, [rax]
                    ; vaddps ymm10, ymm10, ymm15
                );
                silu_acc!(ymm10, 0);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm15, [rax + 32]
                    ; vaddps ymm11, ymm11, ymm15
                );
                silu_acc!(ymm11, 32);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rax], ymm10
                    ; vmovups [rax + 32], ymm11
                );
            }
            Epilogue::Add => {
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm14, [rdx]
                    ; vaddps ymm0, ymm0, ymm14
                    ; vmovups [rdx], ymm0
                    ; vmovups ymm14, [rdx + 32]
                    ; vaddps ymm1, ymm1, ymm14
                    ; vmovups [rdx + 32], ymm1
                    ; lea rax, [rdx + r9]
                    ; vmovups ymm14, [rax]
                    ; vaddps ymm2, ymm2, ymm14
                    ; vmovups [rax], ymm2
                    ; vmovups ymm14, [rax + 32]
                    ; vaddps ymm3, ymm3, ymm14
                    ; vmovups [rax + 32], ymm3
                    ; add rax, r9
                    ; vmovups ymm14, [rax]
                    ; vaddps ymm4, ymm4, ymm14
                    ; vmovups [rax], ymm4
                    ; vmovups ymm14, [rax + 32]
                    ; vaddps ymm5, ymm5, ymm14
                    ; vmovups [rax + 32], ymm5
                    ; add rax, r9
                    ; vmovups ymm14, [rax]
                    ; vaddps ymm6, ymm6, ymm14
                    ; vmovups [rax], ymm6
                    ; vmovups ymm14, [rax + 32]
                    ; vaddps ymm7, ymm7, ymm14
                    ; vmovups [rax + 32], ymm7
                    ; add rax, r9
                    ; vmovups ymm14, [rax]
                    ; vaddps ymm8, ymm8, ymm14
                    ; vmovups [rax], ymm8
                    ; vmovups ymm14, [rax + 32]
                    ; vaddps ymm9, ymm9, ymm14
                    ; vmovups [rax + 32], ymm9
                    ; add rax, r9
                    ; vmovups ymm14, [rax]
                    ; vaddps ymm10, ymm10, ymm14
                    ; vmovups [rax], ymm10
                    ; vmovups ymm14, [rax + 32]
                    ; vaddps ymm11, ymm11, ymm14
                    ; vmovups [rax + 32], ymm11
                );
            }
            Epilogue::Store => {
                dynasm!(ops
                    ; .arch x64
                    ; vmovups [rdx], ymm0
                    ; vmovups [rdx + 32], ymm1
                    ; lea rax, [rdx + r9]
                    ; vmovups [rax], ymm2
                    ; vmovups [rax + 32], ymm3
                    ; add rax, r9
                    ; vmovups [rax], ymm4
                    ; vmovups [rax + 32], ymm5
                    ; add rax, r9
                    ; vmovups [rax], ymm6
                    ; vmovups [rax + 32], ymm7
                    ; add rax, r9
                    ; vmovups [rax], ymm8
                    ; vmovups [rax + 32], ymm9
                    ; add rax, r9
                    ; vmovups [rax], ymm10
                    ; vmovups [rax + 32], ymm11
                );
            }
        }
        dynasm!(ops
            ; .arch x64
            ; vzeroupper
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbp
            ; pop rbx
            ; ret
        );

        let buf = ops
            .finalize()
            .map_err(|_| "dynasm: finalize failed".to_string())?;
        if std::env::var_os("ATOMNET_JIT_DUMP").is_some() {
            let name = match kind {
                Epilogue::Add => "add",
                Epilogue::Silu => "silu",
                Epilogue::SiluAdd => "siluadd",
                Epilogue::Store => "store",
            };
            let _ = std::fs::write(format!("/tmp/opencode/jit_{name}.bin"), &buf[..]);
        }
        let f: GemmKernelSilu = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    // ═══════════════════════════════════════════════════════════
    // ASM-математика: elementwise SiLU по плоскому массиву/строкам.
    //
    // Это не «вызов libm на элемент» и не цикл по скалярам: ядро делает
    // 16 float за итерацию (2 независимых 8-lane блока), полином e^x
    // развёрнут FMA-цепочкой, константы один раз загружены в ymm0..ymm5,
    // хвост (<8) закрывается vmaskmovps без скалярных итераций.
    // ═══════════════════════════════════════════════════════════

    /// y[r*cols + j] = silu(y[r*cols + j] + bias[j]) (bias может быть null
    /// для варианта без bias). Полностью векторизовано, включая хвост.
    pub type MathKernel =
        unsafe extern "sysv64" fn(*mut f32, *const f32, usize, usize);

    /// y[r*cols + j] = softmax(x[r*cols ..]) — max/exp/sum/normalize
    /// целиком в сгенерированном ASM, 2×8 float за итерацию.
    pub type SoftmaxKernel = unsafe extern "sysv64" fn(
        *const f32, // x
        *mut f32,   // y
        usize,      // rows
        usize,      // cols
        *mut f32,   // stats: [max, sum] на строку (может быть null)
    );

    pub struct JitMath {
        _silu_buf: ExecutableBuffer,
        _silu_bias_buf: ExecutableBuffer,
        _softmax_buf: ExecutableBuffer,
        /// Буферы 8 элементных ядер (живут, пока живы указатели).
        _elem_bufs: Vec<ExecutableBuffer>,
        /// y = silu(y)
        pub silu: MathKernel,
        /// y = silu(y + bias), bias — на колонку (cols значений).
        pub silu_bias: MathKernel,
        /// y = softmax(x) по каждой строке.
        pub softmax: SoftmaxKernel,
        /// y += x
        pub add_elem: ElemKernel,
        /// y *= x
        pub mul_elem: ElemKernel,
        /// y *= scal[0]
        pub scale: ElemKernel,
        /// y = x * scal[0]
        pub scale_to: ElemKernel,
        /// y += scal[0] * x
        pub axpy: ElemKernel,
        /// y = scal[0]*y + scal[1]*x
        pub axpby: ElemKernel,
        /// y = max(0, y)
        pub relu: ElemKernel,
        /// y = scal[0]*x + scal[1] (affine_into)
        pub affine: ElemKernel,
    }

    unsafe impl Send for JitMath {}
    unsafe impl Sync for JitMath {}

    static MATH: OnceLock<Option<JitMath>> = OnceLock::new();

    /// ASM-математика или None (JIT выключен / нет AVX2+FMA).
    pub fn jit_math() -> Option<&'static JitMath> {
        if !jit_enabled() {
            return None;
        }
        MATH.get_or_init(|| {
            if std::env::var_os("ATOMNET_NO_JIT").is_some() {
                return None;
            }
            if !crate::atom::isa::virt_isa().has_avx2_fma() {
                return None;
            }
            // AOT: маппим машинный код с патчингом таблицы вместо сборки.
            if let Some(path) = aot_path_from_env() {
                if let Some(j) = try_import_math(&path) {
                    return Some(j);
                }
            }
            let plain = build_math(false);
            let bias = build_math(true);
            let softmax = build_softmax();
            let elem_ops = [
                ElemOp::Add,
                ElemOp::Mul,
                ElemOp::Scale,
                ElemOp::ScaleTo,
                ElemOp::Axpy,
                ElemOp::Axpby,
                ElemOp::Relu,
                ElemOp::Affine,
            ];
            let elems: Result<Vec<_>, _> = elem_ops
                .iter()
                .map(|op| build_elem(*op))
                .collect();
            match (plain, bias, softmax, elems) {
                (Ok((pb, pf)), Ok((bb, bf)), Ok((sb, sf)), Ok(elems)) => {
                    let mut bufs = Vec::new();
                    let mut fns = Vec::new();
                    for (buf, f) in elems {
                        bufs.push(buf);
                        fns.push(f);
                    }
                    Some(JitMath {
                        _silu_buf: pb,
                        _silu_bias_buf: bb,
                        _softmax_buf: sb,
                        _elem_bufs: bufs,
                        silu: pf,
                        silu_bias: bf,
                        softmax: sf,
                        add_elem: fns[0],
                        mul_elem: fns[1],
                        scale: fns[2],
                        scale_to: fns[3],
                        axpy: fns[4],
                        axpby: fns[5],
                        relu: fns[6],
                        affine: fns[7],
                    })
                }
                _ => None,
            }
        })
        .as_ref()
    }

    fn build_math(with_bias: bool) -> Result<(ExecutableBuffer, MathKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        let tab = MATH_TABLE.0.as_ptr() as i64;

        // Один 8-lane блок: x = load(y); x += bias; x = silu(x).
        // Регистры блока: $x (значение), $t (z), $q (int/2^fx), $p (полином).
        macro_rules! silu_block {
            ($x:ident, $t:ident, $q:ident, $p:ident, $y:tt, $b:tt) => {{
                dynasm!(ops
                    ; .arch x64
                    ; vmovups $x, $y
                );
                if with_bias {
                    dynasm!(ops
                        ; .arch x64
                        ; vaddps $x, $x, $b
                    );
                }
                dynasm!(ops
                    ; .arch x64
                    ; vxorps $t, $x, ymm0          // t = -x
                    ; vmaxps $t, $t, ymm1          // clamp lo
                    ; vminps $t, $t, ymm2          // clamp hi
                    ; vmulps $t, $t, ymm3          // q = t·log2e
                    ; vcvtps2dq $q, $t             // fx_i = round(q)
                    ; vcvtdq2ps $p, $q             // fx = float(fx_i)
                    ; vsubps $t, $t, $p            // z = q - fx
                    ; vpaddd $q, $q, ymm5          // fx_i + 127
                    ; vpslld $q, $q, 23            // 2^fx bits
                    ; vmovups $p, [r11 + MT_C6]
                    ; vfmadd213ps $p, $t, [r11 + MT_C5]
                    ; vfmadd213ps $p, $t, [r11 + MT_C4]
                    ; vfmadd213ps $p, $t, [r11 + MT_C3]
                    ; vfmadd213ps $p, $t, [r11 + MT_C2]
                    ; vfmadd213ps $p, $t, [r11 + MT_C1]
                    ; vfmadd213ps $p, $t, ymm4      // + 1.0
                    ; vmulps $p, $p, $q             // exp(-x)
                    ; vaddps $p, $p, ymm4          // 1 + exp(-x)
                    ; vdivps $x, $x, $p            // silu(x)
                    ; vmovups $y, $x
                );
            }};
        }

        // Хвост 1..7 float: maskload/maskstore; невалидные lane не пишутся.
        macro_rules! silu_tail {
            ($y:tt, $b:tt) => {{
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm14, [r11 + rax + MT_MASK]
                    ; vmaskmovps ymm6, ymm14, $y
                );
                if with_bias {
                    dynasm!(ops
                        ; .arch x64
                        ; vmaskmovps ymm7, ymm14, $b
                        ; vaddps ymm6, ymm6, ymm7
                    );
                }
                dynasm!(ops
                    ; .arch x64
                    ; vxorps ymm7, ymm6, ymm0
                    ; vmaxps ymm7, ymm7, ymm1
                    ; vminps ymm7, ymm7, ymm2
                    ; vmulps ymm7, ymm7, ymm3
                    ; vcvtps2dq ymm8, ymm7
                    ; vcvtdq2ps ymm9, ymm8
                    ; vsubps ymm7, ymm7, ymm9
                    ; vpaddd ymm8, ymm8, ymm5
                    ; vpslld ymm8, ymm8, 23
                    ; vmovups ymm9, [r11 + MT_C6]
                    ; vfmadd213ps ymm9, ymm7, [r11 + MT_C5]
                    ; vfmadd213ps ymm9, ymm7, [r11 + MT_C4]
                    ; vfmadd213ps ymm9, ymm7, [r11 + MT_C3]
                    ; vfmadd213ps ymm9, ymm7, [r11 + MT_C2]
                    ; vfmadd213ps ymm9, ymm7, [r11 + MT_C1]
                    ; vfmadd213ps ymm9, ymm7, ymm4
                    ; vmulps ymm9, ymm9, ymm8
                    ; vaddps ymm9, ymm9, ymm4
                    ; vdivps ymm6, ymm6, ymm9
                    ; vmovups ymm14, [r11 + rax + MT_MASK]
                    ; vmaskmovps $y, ymm14, ymm6
                );
            }};
        }

        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; mov r11, QWORD tab
            // Константы живут в регистрах весь проход — в цикле нет
            // ни одной загрузки-константы, только данные.
            ; vpbroadcastd ymm0, [r11 + MT_SIGN]
            ; vpbroadcastd ymm1, [r11 + MT_CLAMP_LO]
            ; vpbroadcastd ymm2, [r11 + MT_CLAMP_HI]
            ; vpbroadcastd ymm3, [r11 + MT_LOG2E]
            ; vpbroadcastd ymm4, [r11 + MT_ONE]
            ; vpbroadcastd ymm5, [r11 + MT_I127]
            ; mov rbx, rcx
            ; shl rbx, 2                 // шаг строки в байтах
            ; test rdx, rdx              // rows == 0?
            ; jz ->mdone
            ; ->mrow:
            ; mov r8, rdi                // y текущей строки
            ; mov r9, rsi                // bias текущей строки
            ; mov r10, rcx
            ; shr r10, 4                 // пар 8-lane блоков (16 float)
            ; mov r12, rcx
            ; and r12, 15                // остаток: бит 3 = блок 8, биты 0..2 = хвост
            ; test r10, r10
            ; jz ->mnopairs
            ; ->mpair:
        );
        silu_block!(ymm6, ymm7, ymm8, ymm9, [r8], [r9]);
        silu_block!(ymm10, ymm11, ymm12, ymm13, [r8 + 32], [r9 + 32]);
        dynasm!(ops
            ; .arch x64
            ; add r8, 64
            ; add r9, 64
            ; dec r10
            ; jnz ->mpair
            ; ->mnopairs:
            ; test r12, 8
            ; jz ->mnoodd
        );
        silu_block!(ymm6, ymm7, ymm8, ymm9, [r8], [r9]);
        dynasm!(ops
            ; .arch x64
            ; add r8, 32
            ; add r9, 32
            ; ->mnoodd:
            ; and r12, 7
            ; jz ->mrowdone
            ; mov rax, r12
            ; shl rax, 5                 // смещение маски = r*32
        );
        silu_tail!([r8], [r9]);
        dynasm!(ops
            ; .arch x64
            ; ->mrowdone:
            ; add rdi, rbx
            ; dec rdx
            ; jnz ->mrow
            ; ->mdone:
            ; vzeroupper
            ; pop r12
            ; pop rbx
            ; ret
        );

        let buf = ops
            .finalize()
            .map_err(|_| "dynasm math: finalize failed".to_string())?;
        if std::env::var_os("ATOMNET_JIT_DUMP").is_some() {
            let name = if with_bias { "silu_bias" } else { "silu" };
            let _ = std::fs::write(format!("/tmp/opencode/jit_math_{name}.bin"), &buf[..]);
        }
        let f: MathKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    /// Сигнатура элементного ядра (System V AMD64): rdi=y, rsi=x, rdx=n,
    /// rcx=scal[2] (собранные на стеке вызывающего константы).
    pub type ElemKernel =
        unsafe extern "sysv64" fn(*mut f32, *const f32, usize, *const f32);

    /// Вид элементной операции (match → последовательность в сгенерированном
    /// коде); ymm14=scal[0], ymm15=scal[1], ymm13=0.0 исключены из блока.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum ElemOp {
        Add,
        Mul,
        Scale,
        ScaleTo,
        Axpy,
        Axpby,
        Relu,
        Affine,
    }

    // ═══════════════════════════════════════════════════════════
    // ASM-элементные ядра: y (и x) одним развёрнутым AVX2-проходом,
    // 2×8 float за итерацию + хвост vmaskmovps. Константы scal — в
    // ymm14/ymm15 один раз; в DynASM только vpbroadcastd из памяти
    // (форм vbroadcastss ymm из памяти зернит верхние 128 бит!).
    fn build_elem(op: ElemOp) -> Result<(ExecutableBuffer, ElemKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        let tab = MATH_TABLE.0.as_ptr() as i64;

        // Вычисление на регистрах: $v — y-значение, $t — x-значение.
        macro_rules! compute_e {
            ($v:ident, $t:ident) => {
                match op {
                    ElemOp::Add => dynasm!(ops
                        ; .arch x64
                        ; vaddps $v, $v, $t
                    ),
                    ElemOp::Mul => dynasm!(ops
                        ; .arch x64
                        ; vmulps $v, $v, $t
                    ),
                    ElemOp::Scale => dynasm!(ops
                        ; .arch x64
                        ; vmulps $v, $v, ymm14
                    ),
                    ElemOp::ScaleTo => dynasm!(ops
                        ; .arch x64
                        ; vmovaps $v, $t
                        ; vmulps $v, $v, ymm14
                    ),
                    ElemOp::Axpy => dynasm!(ops
                        ; .arch x64
                        ; vfmadd231ps $v, ymm14, $t
                    ),
                    ElemOp::Axpby => dynasm!(ops
                        ; .arch x64
                        ; vmulps $v, $v, ymm14
                        ; vfmadd231ps $v, ymm15, $t
                    ),
                    ElemOp::Relu => dynasm!(ops
                        ; .arch x64
                        ; vmaxps $v, $v, ymm13
                    ),
                    ElemOp::Affine => dynasm!(ops
                        ; .arch x64
                        ; vmovaps $v, ymm15
                        ; vfmadd231ps $v, ymm14, $t
                    ),
                }
            };
        }
        // Полный 8-lane блок с обычными загрузками/сохранением.
        macro_rules! block_e {
            ($v:ident, $t:ident, $y:tt, $x:tt) => {
                dynasm!(ops
                    ; .arch x64
                    ; vmovups $v, $y
                    ; vmovups $t, $x
                );
                compute_e!($v, $t);
                dynasm!(ops
                    ; .arch x64
                    ; vmovups $y, $v
                );
            };
        }
        // Маскированный хвост: читаются только свои колонки обоих массивов,
        // запись маскированная — без OOB.
        macro_rules! block_masked_e {
            ($v:ident, $t:ident, $y:tt, $x:tt) => {{
                dynasm!(ops
                    ; .arch x64
                    ; vmovups ymm12, [r11 + rax + MT_MASK]
                );
                match op {
                    ElemOp::Scale | ElemOp::Relu => {
                        dynasm!(ops
                            ; .arch x64
                            ; vmaskmovps $v, ymm12, $y
                        );
                    }
                    ElemOp::ScaleTo | ElemOp::Affine => {
                        dynasm!(ops
                            ; .arch x64
                            ; vmaskmovps $t, ymm12, $x
                        );
                    }
                    _ => {
                        dynasm!(ops
                            ; .arch x64
                            ; vmaskmovps $v, ymm12, $y
                            ; vmaskmovps $t, ymm12, $x
                        );
                    }
                }
                compute_e!($v, $t);
                dynasm!(ops
                    ; .arch x64
                    ; vmaskmovps $y, ymm12, $v
                );
            }};
        }

        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; mov r11, QWORD tab
            ; vpbroadcastd ymm14, [rcx]
            ; vpbroadcastd ymm15, [rcx + 4]
            ; vxorps ymm13, ymm13, ymm13
            ; mov r12, rdi     // y
            ; mov rbx, rsi     // x
            ; mov r10, rdx
            ; shr r10, 4       // пары 8-lane блоков (16 float)
            ; mov rax, rdx
            ; and rax, 15
            ; test r10, r10
            ; jz ->enopairs
            ; ->epair:
        );
        block_e!(ymm0, ymm1, [r12], [rbx]);
        block_e!(ymm2, ymm3, [r12 + 32], [rbx + 32]);
        dynasm!(ops
            ; .arch x64
            ; add r12, 64
            ; add rbx, 64
            ; dec r10
            ; jnz ->epair
            ; ->enopairs:
            ; test rax, 8
            ; jz ->enoodd
        );
        block_e!(ymm0, ymm1, [r12], [rbx]);
        dynasm!(ops
            ; .arch x64
            ; add r12, 32
            ; add rbx, 32
            ; ->enoodd:
            ; and rax, 7
            ; jz ->edone
            ; shl rax, 5
        );
        block_masked_e!(ymm0, ymm1, [r12], [rbx]);
        dynasm!(ops
            ; .arch x64
            ; ->edone:
            ; vzeroupper
            ; pop r12
            ; pop rbx
            ; ret
        );

        let buf = ops
            .finalize()
            .map_err(|_| "dynasm elem: finalize failed".to_string())?;
        if std::env::var_os("ATOMNET_JIT_DUMP").is_some() {
            let name = match op {
                ElemOp::Add => "add_elem",
                ElemOp::Mul => "mul_elem",
                ElemOp::Scale => "scale",
                ElemOp::ScaleTo => "scale_to",
                ElemOp::Axpy => "axpy",
                ElemOp::Axpby => "axpby",
                ElemOp::Relu => "relu",
                ElemOp::Affine => "affine",
            };
            let _ = std::fs::write(format!("/tmp/opencode/jit_elem_{name}.bin"), &buf[..]);
        }
        let f: ElemKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    // ═══════════════════════════════════════════════════════════
    // ASM-softmax: max → exp+sum → normalize, всё в машинном коде.
    // Один проход по строке на фазу, 16 float за итерацию (2×8),
    // хвост через vmaskmovps; в горячем цикле нет ни вызовов, ни
    // скалярных итераций по элементам.
    // ═══════════════════════════════════════════════════════════
    fn build_softmax() -> Result<(ExecutableBuffer, SoftmaxKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        let tab = MATH_TABLE.0.as_ptr() as i64;

        // exp(x - max) с накоплением суммы; max в ymm4, one в ymm2.
        macro_rules! exp_block {
            ($x:ident, $t:ident, $q:ident, $p:ident, $xo:tt, $yo:tt, $sum:ident) => {
                dynasm!(ops
                    ; .arch x64
                    ; vmovups $x, $xo
                    ; vsubps $t, $x, ymm4       // t = x - max ≤ 0
                    ; vmaxps $t, $t, ymm0       // clamp lo (-88)
                    ; vmulps $t, $t, ymm1       // q = t·log2e
                    ; vcvtps2dq $q, $t
                    ; vcvtdq2ps $p, $q
                    ; vsubps $t, $t, $p         // z
                    ; vpaddd $q, $q, ymm3
                    ; vpslld $q, $q, 23
                    ; vmovups $p, [r11 + MT_C6]
                    ; vfmadd213ps $p, $t, [r11 + MT_C5]
                    ; vfmadd213ps $p, $t, [r11 + MT_C4]
                    ; vfmadd213ps $p, $t, [r11 + MT_C3]
                    ; vfmadd213ps $p, $t, [r11 + MT_C2]
                    ; vfmadd213ps $p, $t, [r11 + MT_C1]
                    ; vfmadd213ps $p, $t, ymm2
                    ; vmulps $p, $p, $q
                    ; vmovups $yo, $p
                    ; vaddps $sum, $sum, $p
                );
            };
        }

        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push r12
            ; push r13
            ; push r14
            ; mov r13, r8                // stats (5-й аргумент)
            ; xor r14, r14
            ; test r13, r13
            ; jz ->snostats
            ; mov r14, 8                 // шаг stats: 2 float на строку
            ; ->snostats:
            ; mov r11, QWORD tab
            ; mov rbx, rcx
            ; shl rbx, 2
            ; test rdx, rdx
            ; jz ->sdone
            ; ->srow:
            // ── 1. максимум по строке ──
            ; mov r8, rdi
            ; mov r10, rcx
            ; shr r10, 4
            ; mov r12, rcx
            ; and r12, 15
            ; vpbroadcastd ymm0, [r11 + MT_NEG_INF]
            ; vmovaps ymm1, ymm0
            ; test r10, r10
            ; jz ->smaxnopairs
            ; ->smaxpair:
            ; vmovups ymm4, [r8]
            ; vmaxps ymm0, ymm0, ymm4
            ; vmovups ymm4, [r8 + 32]
            ; vmaxps ymm1, ymm1, ymm4
            ; add r8, 64
            ; dec r10
            ; jnz ->smaxpair
            ; ->smaxnopairs:
            ; test r12, 8
            ; jz ->smaxnoodd
            ; vmovups ymm4, [r8]
            ; vmaxps ymm0, ymm0, ymm4
            ; add r8, 32
            ; ->smaxnoodd:
            ; and r12, 7
            ; jz ->smaxdone
            ; mov rax, r12
            ; shl rax, 5
            ; vmovups ymm5, [r11 + rax + MT_MASK]
            ; vmaskmovps ymm4, ymm5, [r8]
            ; vpor ymm4, ymm4, [r11 + rax + MT_TAIL_OR]
            ; vmaxps ymm0, ymm0, ymm4
            ; ->smaxdone:
            // горизонтальный max: ymm0 = max(ymm0, ymm1) -> все lane
            ; vmaxps ymm0, ymm0, ymm1
            ; vshufps ymm1, ymm0, ymm0, 0x4e
            ; vmaxps ymm0, ymm0, ymm1
            ; vshufps ymm1, ymm0, ymm0, -79
            ; vmaxps ymm0, ymm0, ymm1
            ; vextractf128 xmm1, ymm0, 1
            ; vmaxps xmm0, xmm0, xmm1
            ; vperm2f128 ymm4, ymm0, ymm0, 0x00
            ; vshufps ymm4, ymm4, ymm4, 0x00
            ; test r13, r13
            ; jz ->smaxnostats
            ; vmovss [r13], xmm4
            ; ->smaxnostats:
            // ── 2. exp + сумма ──
            ; vpbroadcastd ymm0, [r11 + MT_CLAMP_LO]
            ; vpbroadcastd ymm1, [r11 + MT_LOG2E]
            ; vpbroadcastd ymm2, [r11 + MT_ONE]
            ; vpbroadcastd ymm3, [r11 + MT_I127]
            ; mov r8, rdi
            ; mov r9, rsi
            ; mov r10, rcx
            ; shr r10, 4
            ; mov r12, rcx
            ; and r12, 15
            ; vxorps ymm6, ymm6, ymm6
            ; vxorps ymm7, ymm7, ymm7
            ; test r10, r10
            ; jz ->sexpnopairs
            ; ->sexppair:
        );
        exp_block!(ymm8, ymm9, ymm10, ymm11, [r8], [r9], ymm6);
        exp_block!(ymm12, ymm13, ymm14, ymm15, [r8 + 32], [r9 + 32], ymm7);
        dynasm!(ops
            ; .arch x64
            ; add r8, 64
            ; add r9, 64
            ; dec r10
            ; jnz ->sexppair
            ; ->sexpnopairs:
            ; test r12, 8
            ; jz ->sexpnoodd
        );
        exp_block!(ymm8, ymm9, ymm10, ymm11, [r8], [r9], ymm6);
        dynasm!(ops
            ; .arch x64
            ; add r8, 32
            ; add r9, 32
            ; ->sexpnoodd:
            ; and r12, 7
            ; jz ->sexpdone
            ; mov rax, r12
            ; shl rax, 5
            ; vmovups ymm5, [r11 + rax + MT_MASK]
            ; vmaskmovps ymm8, ymm5, [r8]
            ; vsubps ymm9, ymm8, ymm4
            ; vmaxps ymm9, ymm9, ymm0
            ; vmulps ymm9, ymm9, ymm1
            ; vcvtps2dq ymm10, ymm9
            ; vcvtdq2ps ymm11, ymm10
            ; vsubps ymm9, ymm9, ymm11
            ; vpaddd ymm10, ymm10, ymm3
            ; vpslld ymm10, ymm10, 23
            ; vmovups ymm11, [r11 + MT_C6]
            ; vfmadd213ps ymm11, ymm9, [r11 + MT_C5]
            ; vfmadd213ps ymm11, ymm9, [r11 + MT_C4]
            ; vfmadd213ps ymm11, ymm9, [r11 + MT_C3]
            ; vfmadd213ps ymm11, ymm9, [r11 + MT_C2]
            ; vfmadd213ps ymm11, ymm9, [r11 + MT_C1]
            ; vfmadd213ps ymm11, ymm9, ymm2
            ; vmulps ymm11, ymm11, ymm10
            ; vandps ymm11, ymm11, ymm5
            ; vaddps ymm6, ymm6, ymm11
            ; vmaskmovps [r9], ymm5, ymm11
            ; ->sexpdone:
            // ── 3. сумма -> 1/sum, broadcast ──
            ; vaddps ymm6, ymm6, ymm7
            ; vhaddps ymm6, ymm6, ymm6
            ; vhaddps ymm6, ymm6, ymm6
            ; vextractf128 xmm7, ymm6, 1
            ; vaddps xmm6, xmm6, xmm7
            ; test r13, r13
            ; jz ->ssumnostats
            ; vmovss [r13 + 4], xmm6
            ; ->ssumnostats:
            ; vdivss xmm6, xmm2, xmm6
            ; vperm2f128 ymm5, ymm6, ymm6, 0x00
            ; vshufps ymm5, ymm5, ymm5, 0x00
            // ── 4. normalize ──
            ; mov r9, rsi
            ; mov r10, rcx
            ; shr r10, 4
            ; mov r12, rcx
            ; and r12, 15
            ; test r10, r10
            ; jz ->snormnopairs
            ; ->snormpair:
            ; vmovups ymm0, [r9]
            ; vmulps ymm0, ymm0, ymm5
            ; vmovups [r9], ymm0
            ; vmovups ymm0, [r9 + 32]
            ; vmulps ymm0, ymm0, ymm5
            ; vmovups [r9 + 32], ymm0
            ; add r9, 64
            ; dec r10
            ; jnz ->snormpair
            ; ->snormnopairs:
            ; test r12, 8
            ; jz ->snormnoodd
            ; vmovups ymm0, [r9]
            ; vmulps ymm0, ymm0, ymm5
            ; vmovups [r9], ymm0
            ; add r9, 32
            ; ->snormnoodd:
            ; and r12, 7
            ; jz ->snormdone
            ; mov rax, r12
            ; shl rax, 5
            ; vmovups ymm6, [r11 + rax + MT_MASK]
            ; vmaskmovps ymm0, ymm6, [r9]
            ; vmulps ymm0, ymm0, ymm5
            ; vmaskmovps [r9], ymm6, ymm0
            ; ->snormdone:
            ; add rdi, rbx
            ; add rsi, rbx
            ; add r13, r14
            ; dec rdx
            ; jnz ->srow
            ; ->sdone:
            ; vzeroupper
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbx
            ; ret
        );

        let buf = ops
            .finalize()
            .map_err(|_| "dynasm softmax: finalize failed".to_string())?;
        if std::env::var_os("ATOMNET_JIT_DUMP").is_some() {
            let _ = std::fs::write("/tmp/opencode/jit_math_softmax.bin", &buf[..]);
        }
        let f: SoftmaxKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }

    // ═══════════════════════════════════════════════════════════
    // ASM-ядро Q4×Q8 GEMV: y = W_q4 @ x + b (int4 + int8-остаток).
    //
    // Полный контекст (System V AMD64, rdi = &ctx):
    //   +0   weights_i4 ptr, +8  resid ptr, +16 qx ptr,
    //   +24  scales ptr, +32 zeros, +40 code_sums, +48 rs, +56 rz,
    //   +64  rsum, +72 sx, +80 mx, +88 sq, +96 bias, +104 y,
    //   +112 blocks(u64), +120 bpr(u64), +128 rpr(u64), +136 ycell(u64),
    //   +144 rows(u64)
    // ═══════════════════════════════════════════════════════════

    #[repr(C)]
    pub struct Q4Ctx {
        pub weights_i4: *const u8,
        pub resid: *const u8,
        pub qx: *const i8,
        pub scales: *const f32,
        pub zeros: *const f32,
        pub code_sums: *const i32,
        pub rs: *const f32,
        pub rz: *const f32,
        pub rsum: *const i32,
        pub sx: *const f32,
        pub mx: *const f32,
        pub sq: *const i32,
        pub bias: *const f32,
        pub y: *mut f32,
        pub blocks: u64,
        pub bpr: u64,
        pub rpr: u64,
        pub ycell: u64,
        pub rows: u64,
    }

    pub type Q4GemvKernel = unsafe extern "sysv64" fn(*const Q4Ctx) -> ();

    pub struct JitQ4 {
        _buf: ExecutableBuffer,
        pub gemv: Q4GemvKernel,
    }

    unsafe impl Send for JitQ4 {}
    unsafe impl Sync for JitQ4 {}

    static Q4: OnceLock<Option<JitQ4>> = OnceLock::new();

    /// ASM-ядро Q4×Q8 GEMV или None (JIT выключен / нет AVX2+FMA).
    pub fn jit_q4_gemv() -> Option<&'static JitQ4> {
        if !jit_enabled() {
            return None;
        }
        Q4.get_or_init(|| {
            if std::env::var_os("ATOMNET_NO_JIT").is_some() {
                return None;
            }
            if !crate::atom::isa::virt_isa().has_avx2_fma() {
                return None;
            }
            // AOT: маппим машинный код с патчингом таблицы вместо сборки.
            if let Some(path) = aot_path_from_env() {
                if let Some(j) = try_import_q4(&path) {
                    return Some(j);
                }
            }
            let (buf, k) = build_q4_gemv().ok()?;
            Some(JitQ4 {
                _buf: buf,
                gemv: k,
            })
        })
        .as_ref()
    }

    // ═══════════════ AOT v2: полный дамп/импорт DynASM-ядер ═══════════════
    // Один файл на все ядра (GEMM ×8, math, elem, Q4, atomjit): при повторных
    // коротких процессах не пересобираем DynASM, а маппим байты. CPU-гейт
    // (AVX2+FMA) — обязателен: чужие машинные коды на без-AVX2 валится SIGILL.
    // Ядра с абсолютной адресацией MathTable патчатся при импорте
    // (mov r11, imm64 → новый адрес таблицы).

    pub const AOT_MAGIC: &[u8; 10] = b"ATOMJIT2\0\0";
    const AOT_VERSION: u32 = 2;

    /// Прочитать v2-карту (имя → (байты, table_old)).
    pub fn aot_read_v2(path: &str) -> Result<Vec<(String, Vec<u8>, u64)>, String> {
        let buf = std::fs::read(path).map_err(|e| e.to_string())?;
        if buf.len() < 10 + 8 || &buf[0..10] != AOT_MAGIC {
            return Err("aot2: bad magic".into());
        }
        let ver = u32::from_le_bytes(buf[10..14].try_into().unwrap());
        if ver != AOT_VERSION {
            return Err(format!("aot2: version {ver}"));
        }
        let n = u32::from_le_bytes(buf[14..18].try_into().unwrap()) as usize;
        let mut pos = 18usize;
        let mut out = Vec::with_capacity(n);
        let mut rd32 = |pos: &mut usize| -> Result<u32, String> {
            if *pos + 4 > buf.len() {
                return Err("aot2 eof".into());
            }
            let v = u32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            Ok(v)
        };
        let mut rd64 = |pos: &mut usize| -> Result<u64, String> {
            if *pos + 8 > buf.len() {
                return Err("aot2 eof".into());
            }
            let v = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
            *pos += 8;
            Ok(v)
        };
        for _ in 0..n {
            let nl = rd32(&mut pos)? as usize;
            if pos + nl > buf.len() {
                return Err("aot2 eof name".into());
            }
            let name =
                String::from_utf8(buf[pos..pos + nl].to_vec()).map_err(|e| e.to_string())?;
            pos += nl;
            let bl = rd64(&mut pos)? as usize;
            let _off = rd64(&mut pos)?;
            let table_old = rd64(&mut pos)?;
            if pos + bl > buf.len() {
                return Err("aot2 eof blob".into());
            }
            out.push((name, buf[pos..pos + bl].to_vec(), table_old));
            pos += bl;
        }
        if pos != buf.len() {
            return Err("aot2: trailing bytes".into());
        }
        Ok(out)
    }

    /// Импортироватьblob в исполняемый буфер, пропатчив MathTable.
    /// Возвращает буфер и адрес входа (offset 0).
    pub fn aot_exec_import(
        bytes: &[u8],
        table_old: u64,
    ) -> Result<(ExecutableBuffer, *const u8), String> {
        if bytes.is_empty() {
            return Err("aot2: empty kernel".into());
        }
        let mut mb =
            dynasmrt::mmap::MutableBuffer::new(bytes.len()).map_err(|e| e.to_string())?;
        mb.set_len(bytes.len());
        {
            let target = &mut *mb;
            target.copy_from_slice(bytes);
            if table_old != 0 {
                let new_table = MATH_TABLE.0.as_ptr() as u64;
                let pat = table_old.to_le_bytes();
                let nw = new_table.to_le_bytes();
                let mut patched = 0usize;
                let len = target.len();
                for i in 0..len.saturating_sub(8) {
                    if target[i..i + 8] == pat {
                        target[i..i + 8].copy_from_slice(&nw);
                        patched += 1;
                    }
                }
                if patched == 0 {
                    return Err("aot2: table pattern not found".into());
                }
            }
        }
        let buf = mb.make_exec().map_err(|e| e.to_string())?;
        let entry = buf.ptr(dynasmrt::AssemblyOffset(0));
        Ok((buf, entry))
    }

    /// Путь AOT-файла из окружения (нет — AOT не используется).
    pub fn aot_path_from_env() -> Option<String> {
        std::env::var("ATOMNET_AOT_PATH").ok().filter(|s| !s.is_empty())
    }

    static SOURCE: std::sync::atomic::AtomicU8 =
        std::sync::atomic::AtomicU8::new(0);

    fn set_source(v: u8) {
        SOURCE.store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Откуда взяли JIT: "none" / "assembled" / "aot".
    pub fn jit_source() -> &'static str {
        match SOURCE.load(std::sync::atomic::Ordering::Relaxed) {
            1 => "assembled",
            2 => "aot",
            _ => "none",
        }
    }

    fn aot_get(
        map: &[(String, Vec<u8>, u64)],
        name: &str,
    ) -> Option<(Vec<u8>, u64)> {
        map.iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, b, t)| (b.clone(), *t))
    }

    fn try_import_gemm(path: &str) -> Option<JitGemm> {
        let map = aot_read_v2(path).ok()?;
        let names = [
            "gemm.store",
            "gemm.add",
            "gemm.silu",
            "gemm.siluadd",
            "gemm.store_il",
            "gemm.add_il",
            "gemm.silu_il",
            "gemm.siluadd_il",
        ];
        let mut bufs = Vec::new();
        let mut fns: Vec<GemmKernelSilu> = Vec::new();
        for name in names {
            let (bytes, table) = aot_get(&map, name)?;
            let (buf, entry) = aot_exec_import(&bytes, table).ok()?;
            bufs.push(buf);
            fns.push(unsafe { std::mem::transmute::<*const u8, GemmKernelSilu>(entry) });
        }
        Some(JitGemm {
            _store_buf: bufs.remove(0),
            _add_buf: bufs.remove(0),
            _silu_buf: bufs.remove(0),
            _siluadd_buf: bufs.remove(0),
            _store_il_buf: bufs.remove(0),
            _add_il_buf: bufs.remove(0),
            _silu_il_buf: bufs.remove(0),
            _siluadd_il_buf: bufs.remove(0),
            store: unsafe { std::mem::transmute::<_, GemmKernel>(fns[0]) },
            add: unsafe { std::mem::transmute::<_, GemmKernel>(fns[1]) },
            store_silu: fns[2],
            add_silu: fns[3],
            store_any: fns[0],
            add_any: fns[1],
            store_any_il: fns[4],
            add_any_il: fns[5],
            store_silu_il: fns[6],
            add_silu_il: fns[7],
        })
    }

    fn try_import_math(path: &str) -> Option<JitMath> {
        let map = aot_read_v2(path).ok()?;
        let (sb, se) = aot_exec_import(
            &aot_get(&map, "math.silu")?.0,
            aot_get(&map, "math.silu")?.1,
        )
        .ok()?;
        let (bb, be) = aot_exec_import(
            &aot_get(&map, "math.silu_bias")?.0,
            aot_get(&map, "math.silu_bias")?.1,
        )
        .ok()?;
        let (xb, xe) = aot_exec_import(
            &aot_get(&map, "math.softmax")?.0,
            aot_get(&map, "math.softmax")?.1,
        )
        .ok()?;
        let names = ["add", "mul", "scale", "scale_to", "axpy", "axpby", "relu", "affine"];
        let mut bufs = Vec::new();
        let mut fns: Vec<ElemKernel> = Vec::new();
        for nm in names {
            let (bytes, table) = aot_get(&map, &format!("elem.{nm}"))?;
            let (buf, entry) = aot_exec_import(&bytes, table).ok()?;
            bufs.push(buf);
            fns.push(unsafe { std::mem::transmute::<*const u8, ElemKernel>(entry) });
        }
        Some(JitMath {
            _silu_buf: sb,
            _silu_bias_buf: bb,
            _softmax_buf: xb,
            _elem_bufs: bufs,
            silu: unsafe { std::mem::transmute::<*const u8, MathKernel>(se) },
            silu_bias: unsafe { std::mem::transmute::<*const u8, MathKernel>(be) },
            softmax: unsafe { std::mem::transmute::<*const u8, SoftmaxKernel>(xe) },
            add_elem: fns[0],
            mul_elem: fns[1],
            scale: fns[2],
            scale_to: fns[3],
            axpy: fns[4],
            axpby: fns[5],
            relu: fns[6],
            affine: fns[7],
        })
    }

    fn try_import_q4(path: &str) -> Option<JitQ4> {
        let map = aot_read_v2(path).ok()?;
        let (bytes, table) = aot_get(&map, "q4.gemv")?;
        let (buf, entry) = aot_exec_import(&bytes, table).ok()?;
        Some(JitQ4 {
            _buf: buf,
            gemv: unsafe { std::mem::transmute::<*const u8, Q4GemvKernel>(entry) },
        })
    }

    /// Полный дамп всех ядер в v2-файл. Принудительно собирает живые
    /// JIT-ядра (OnceLock), поэтому файл получается с текущей машины.
    pub fn aot_dump_all(path: &str) -> Result<String, String> {
        use std::io::Write;
        let table = MATH_TABLE.0.as_ptr() as u64;
        let mut parts: Vec<(String, Vec<u8>, u64)> = Vec::new();
        let mut push = |name: &str, buf: &[u8], parts: &mut Vec<(String, Vec<u8>, u64)>| {
            let has_table = {
                let pat = table.to_le_bytes();
                buf.windows(8).any(|w| w == pat)
            };
            parts.push((
                name.to_string(),
                buf.to_vec(),
                if has_table { table } else { 0 },
            ));
        };
        if let Some(g) = jit_gemm() {
            push("gemm.store", &g._store_buf, &mut parts);
            push("gemm.add", &g._add_buf, &mut parts);
            push("gemm.silu", &g._silu_buf, &mut parts);
            push("gemm.siluadd", &g._siluadd_buf, &mut parts);
            push("gemm.store_il", &g._store_il_buf, &mut parts);
            push("gemm.add_il", &g._add_il_buf, &mut parts);
            push("gemm.silu_il", &g._silu_il_buf, &mut parts);
            push("gemm.siluadd_il", &g._siluadd_il_buf, &mut parts);
        }
        if let Some(m) = jit_math() {
            push("math.silu", &m._silu_buf, &mut parts);
            push("math.silu_bias", &m._silu_bias_buf, &mut parts);
            push("math.softmax", &m._softmax_buf, &mut parts);
            let names = ["add", "mul", "scale", "scale_to", "axpy", "axpby", "relu", "affine"];
            for (i, nm) in names.iter().enumerate() {
                if i < m._elem_bufs.len() {
                    push(&format!("elem.{nm}"), &m._elem_bufs[i], &mut parts);
                }
            }
        }
        if let Some(q) = jit_q4_gemv() {
            push("q4.gemv", &q._buf, &mut parts);
        }
        for (name, bytes) in crate::atom::math::atomjit::kernel_blobs() {
            push(&name, &bytes, &mut parts);
        }
        if parts.is_empty() {
            return Err("aot2: no kernels (jit off?)".into());
        }
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(AOT_MAGIC);
        out.extend_from_slice(&AOT_VERSION.to_le_bytes());
        out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for (name, bytes, table_old) in &parts {
            out.extend_from_slice(&(name.len() as u32).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes()); // entry offset
            out.extend_from_slice(&table_old.to_le_bytes());
            out.extend_from_slice(bytes);
        }
        let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
        f.write_all(&out).map_err(|e| e.to_string())?;
        Ok(path.to_string())
    }

    /// Проверка импорта без установки в OnceLock: вёрнет список ядер и
    /// число патчей таблицы (для Python/тестов).
    pub fn aot_import_check(path: &str) -> Result<Vec<(String, usize, u64)>, String> {
        let parts = aot_read_v2(path)?;
        let mut out = Vec::new();
        for (name, bytes, table_old) in &parts {
            let (_buf, _entry) = aot_exec_import(bytes, *table_old)?;
            out.push((name.clone(), bytes.len(), *table_old));
        }
        Ok(out)
    }

    fn build_q4_gemv() -> Result<(ExecutableBuffer, Q4GemvKernel), String> {
        let mut ops = dynasmrt::x64::Assembler::new().map_err(|e| e.to_string())?;
        let start = ops.offset();
        let tab = MATH_TABLE.0.as_ptr() as i64;

        // int4-чанк: 32 qx-байта (ymm2) + 16 байт весов -> 32 nibble-байта (ymm6).
        macro_rules! int4_chunk {
            ($qoff:literal, $woff:literal) => {
                dynasm!(ops
                    ; vmovdqu ymm2, [r10 + $qoff]
                    ; vmovdqu xmm3, [r8 + $woff]
                    ; vpand xmm4, xmm3, xmm14
                    ; vpsrlw xmm5, xmm3, 4
                    ; vpand xmm5, xmm5, xmm14
                    ; vpunpcklbw xmm6, xmm4, xmm5
                    ; vpunpckhbw xmm7, xmm4, xmm5
                    ; vinserti128 ymm6, ymm6, xmm7, 1
                    ; vpmaddubsw ymm8, ymm6, ymm2
                    ; vpaddw ymm0, ymm0, ymm8
                );
            };
        }
        // int8-резидуал: 32 байта остатка (ymm3) + 32 байта qx (ymm2).
        macro_rules! resid_chunk {
            ($off:literal) => {
                dynasm!(ops
                    ; vmovdqu ymm2, [r10 + $off]
                    ; vmovdqu ymm3, [r9 + $off]
                    ; vpmaddubsw ymm8, ymm3, ymm2
                    ; vpmaddwd ymm8, ymm8, ymm15
                    ; vpaddd ymm0, ymm0, ymm8
                );
            };
        }

        dynasm!(ops
            ; .arch x64
            ; push rbx
            ; push rbp
            ; push r12
            ; push r13
            ; push r14
            ; push r15
            ; sub rsp, 96
            ; mov r15, rdi
            ; mov r11, QWORD tab
            ; mov eax, 0x0F0F0F0F
            ; vmovd xmm14, eax
            ; vpbroadcastd ymm14, xmm14
            ; mov eax, 0x00010001
            ; vmovd xmm15, eax
            ; vpbroadcastd ymm15, xmm15
            ; mov r12, [r15 + 144]
            ; mov r8, [r15 + 0]
            ; mov rbx, [r15 + 24]
            ; mov rsi, [r15 + 40]
            ; mov rdi, [r15 + 48]
            ; mov rbp, [r15 + 104]
            ; mov r14, [r15 + 96]
            ; mov r9, [r15 + 32]
            ; mov [rsp + 0], r9
            ; mov r9, [r15 + 64]
            ; mov [rsp + 8], r9
            ; mov r9, [r15 + 8]
            ; mov [rsp + 16], r9
            ; mov r9, [r15 + 56]
            ; mov [rsp + 24], r9
            ; jmp ->q4row
            ; ->q4row:
            ;   test r12, r12
            ;   jz ->q4done
            ;   vxorps xmm1, xmm1, xmm1
            ;   mov r13, [r15 + 112]
            ;   mov r10, [r15 + 16]
            ;   mov rcx, [r15 + 72]
            ;   mov rdx, [r15 + 80]
            ;   mov rax, [r15 + 88]
            ; ->q4blk:
            ;   vpxor ymm0, ymm0, ymm0
        );
        int4_chunk!(0, 0);
        int4_chunk!(32, 16);
        int4_chunk!(64, 32);
        int4_chunk!(96, 48);
        int4_chunk!(128, 64);
        int4_chunk!(160, 80);
        int4_chunk!(192, 96);
        int4_chunk!(224, 112);
        dynasm!(ops
            ; .arch x64
            ; vpmaddwd ymm0, ymm0, ymm15
            ; vextracti128 xmm2, ymm0, 1
            ; vpaddd xmm0, xmm0, xmm2
            ; vphaddd xmm0, xmm0, xmm0
            ; vphaddd xmm0, xmm0, xmm0
            ; vcvtdq2ps xmm0, xmm0
            ; vmovd xmm6, [rax]
            ; vcvtdq2ps xmm6, xmm6
            ; vmulss xmm7, xmm6, [r11 + MT_M8]
            ; vsubss xmm0, xmm0, xmm7
            ; vmovss xmm9, [rbx]
            ; vmovss xmm10, [rcx]
            ; vmovss xmm11, [rdx]
            ; mov r9, [rsp + 0]
            ; vmovss xmm13, [r9]
            ; vmovd xmm4, [rsi]
            ; vcvtdq2ps xmm4, xmm4
            ; vmulss xmm12, xmm9, xmm10
            ; vmulss xmm12, xmm12, xmm0
            ; vmulss xmm5, xmm9, xmm11
            ; vmulss xmm5, xmm5, xmm4
            ; vaddss xmm12, xmm12, xmm5
            ; vmulss xmm5, xmm13, xmm10
            ; vmulss xmm5, xmm5, xmm6
            ; vaddss xmm12, xmm12, xmm5
            ; vmulss xmm5, xmm13, xmm11
            ; vmulss xmm5, xmm5, [r11 + MT_M256]
            ; vaddss xmm12, xmm12, xmm5
            ; vaddss xmm1, xmm1, xmm12
            ; vpxor ymm0, ymm0, ymm0
            ; mov r9, [rsp + 16]
        );
        resid_chunk!(0);
        resid_chunk!(32);
        resid_chunk!(64);
        resid_chunk!(96);
        resid_chunk!(128);
        resid_chunk!(160);
        resid_chunk!(192);
        resid_chunk!(224);
        dynasm!(ops
            ; .arch x64
            ; vextracti128 xmm2, ymm0, 1
            ; vpaddd xmm0, xmm0, xmm2
            ; vphaddd xmm0, xmm0, xmm0
            ; vphaddd xmm0, xmm0, xmm0
            ; vcvtdq2ps xmm0, xmm0
            ; vmulss xmm7, xmm6, [r11 + MT_M128]
            ; vsubss xmm0, xmm0, xmm7
            ; vmovss xmm9, [rdi]
            ; mov r9, [rsp + 24]
            ; vmovss xmm5, [r9]
            ; mov r9, [rsp + 8]
            ; vmovd xmm4, [r9]
            ; vcvtdq2ps xmm4, xmm4
            ; vmulss xmm12, xmm10, xmm0
            ; vmulss xmm4, xmm11, xmm4
            ; vaddss xmm12, xmm12, xmm4
            ; vmulss xmm12, xmm9, xmm12
            ; vmulss xmm7, xmm10, xmm6
            ; vmulss xmm4, xmm11, [r11 + MT_M256]
            ; vaddss xmm7, xmm7, xmm4
            ; vmulss xmm7, xmm5, xmm7
            ; vaddss xmm12, xmm12, xmm7
            ; vaddss xmm1, xmm1, xmm12
            ; add r8, 128
            ; add rbx, 4
            ; add rsi, 4
            ; add rdi, 4
            ; add rax, 4
            ; add rcx, 4
            ; add rdx, 4
            ; add r10, 256
            ; mov r9, [rsp + 0]
            ; add r9, 4
            ; mov [rsp + 0], r9
            ; mov r9, [rsp + 8]
            ; add r9, 4
            ; mov [rsp + 8], r9
            ; mov r9, [rsp + 16]
            ; add r9, 256
            ; mov [rsp + 16], r9
            ; mov r9, [rsp + 24]
            ; add r9, 4
            ; mov [rsp + 24], r9
            ; sub r13, 1
            ; jnz ->q4blk
            ; vmovss xmm0, [r14]
            ; vaddss xmm1, xmm1, xmm0
            ; vmovss [rbp], xmm1
            ; add r14, 4
            ; add rbp, 4
            ; dec r12
            ; jnz ->q4row
            ; ->q4done:
            ; add rsp, 96
            ; pop r15
            ; pop r14
            ; pop r13
            ; pop r12
            ; pop rbp
            ; pop rbx
            ; ret
        );

        let buf = ops
            .finalize()
            .map_err(|_| "dynasm q4 gemv: finalize failed".to_string())?;
        if std::env::var_os("ATOMNET_JIT_DUMP").is_some() {
            let _ = std::fs::write("/tmp/opencode/jit_math_q4.bin", &buf[..]);
        }
        let f: Q4GemvKernel = unsafe { std::mem::transmute(buf.ptr(start)) };
        Ok((buf, f))
    }
}

/// Тесты, переключающие глобальный JIT через `set_jit_enabled`, обязаны
/// держать этот замок: иначе параллельный тест увидит чужое состояние.
#[cfg(test)]
pub static TEST_JIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_arch = "x86_64")]
pub use imp::{
    aot_dump_all, aot_import_check, aot_path_from_env, aot_read_v2, jit_enabled, jit_gemm,
    jit_math, jit_q4_gemv, jit_source, set_jit_enabled, GemmKernel, GemmKernelSilu, JitGemm,
    JitMath, JitQ4, MathKernel, ElemKernel, Q4Ctx, Q4GemvKernel, SoftmaxKernel,
};
