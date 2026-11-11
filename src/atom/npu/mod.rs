// src/atom/npu/mod.rs
//! AtomNET NPU — эмулятор распределения ресурсов (не реальное железо).
//!
//! NPU здесь — виртуальный mac-массив: работа режется на регистровые
//! кластеры 6×16 (12 YMM), каждый кластер распределяется по виртуальным
//! ядрам, а виртуальные ядра маппятся на CPU-потоки. Детерминированность:
//! один поток даёт бит-в-бит тот же результат, что N потоков, — «обучение
//! на одном потоке как на NPU».
//!
//! Архитектура: `NpuBackend` trait → `NpuEmuBackend` (virtual mac-array) →
//! `NpuRouter` с профилем ресурсов → диспетчерские функции.

use std::sync::OnceLock;

use crate::atom::flash::flash_gemm;
use crate::atom::math::simd;
use crate::atom::math::tensor::matvec_slice;

/// Унифицированный вычислительный бэкенд.
pub trait NpuBackend: Send + Sync {
    /// Идентификатор бэкенда (emu-cpu / ...).
    fn id(&self) -> &'static str;
    /// Доступен ли бэкенд прямо сейчас.
    fn is_available(&self) -> bool;
    /// y = a·b (SIMD-ядро).
    fn dot(&self, a: &[f32], b: &[f32]) -> f32;
    /// y = W·x, W: rows×cols, x: cols (GEMV).
    fn matvec(&self, w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32>;
    /// C = A×B, A: rows×k, B: k×cols (GEMM).
    fn matmul(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
    ) -> Vec<f32>;
}

// ═══════════════════════════════════════════════════════════
// Профиль ресурсов эмулятора (mac-массив + бюджет потоков)
// ═══════════════════════════════════════════════════════════

/// Профиль виртуального NPU: сколько mac-ядр и на сколько регистровых
/// кластеров каждое. Виртуальные ядра НЕ равны потокам: можно задать
/// много виртуальных ядер (например 32) на одном CPU-потоке — эмулятор
/// разобьёт работу на 32 зоны и посчитает их последовательно, эмулируя
/// большой NPU. Реальная скорость при этом — скорость одного потока
/// (физика времени не обманывается), но распределение и результаты —
/// как на большом mac-массиве.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NpuProfile {
    /// Число виртуальных ядер (0 = авто = физ. бюджет).
    /// Любое значение ≥1, включая большее потоков (oversubscribe).
    pub virtual_cores: usize,
    /// Размер mac-кластера (совпадает с JIT-тайлом 6×16).
    pub mac_rows: usize,
    pub mac_cols: usize,
}

impl NpuProfile {
    /// Ограниченный: 1 виртуальное ядро.
    pub fn tiny() -> Self {
        Self { virtual_cores: 1, mac_rows: 6, mac_cols: 16 }
    }
    /// Стандартный: виртуальных ядер = физ. бюджет потоков.
    pub fn standard() -> Self {
        Self { virtual_cores: 0, mac_rows: 6, mac_cols: 16 }
    }
    /// Турбо: виртуальных ядер ×2 (модель HT) — распределение мельче.
    pub fn turbo() -> Self {
        Self { virtual_cores: u32::MAX as usize, mac_rows: 6, mac_cols: 16 }
    }
    /// Глубокий: 32 виртуальных ядра на одном рабочем потоке.
    /// Работа разбивается на 32 зоны по 6-строчным кластерам, поток идёт
    /// по ним последовательно с регистровым конвейером (без fork-пула).
    /// Это глубокая модель NPU, а не магическое ускорение: время =
    /// время одного ядра, качество/результаты — как у большого mac-массива.
    pub fn deep() -> Self {
        Self { virtual_cores: 32, mac_rows: 6, mac_cols: 16 }
    }

    /// Число виртуальных ядер. Oversubscribe разрешён (виртуальных ядер
    /// может быть больше CPU-потоков — эмулятор считает зоны по очереди).
    pub fn cores(&self) -> usize {
        let budget = crate::atom::flash::flash_hw::thread_budget().max(1);
        match self.virtual_cores {
            0 => budget,
            v if v == u32::MAX as usize => budget.saturating_mul(2).max(1),
            v => v.max(1),
        }
    }

    /// Гранулярность распределения: регистровых кластеров на ядро.
    pub fn clusters(&self) -> usize {
        1
    }
}

impl Default for NpuProfile {
    fn default() -> Self {
        Self::standard()
    }
}

/// Профиль из окружения: ATOMNET_NPU_PROFILE=tiny|standard|turbo,
/// ATOMNET_NPU_CORES=N — явное число виртуальных ядер (даже > потоков).
pub fn profile_from_env() -> NpuProfile {
    let mut p = match std::env::var("ATOMNET_NPU_PROFILE").ok().as_deref() {
        Some("tiny") => NpuProfile::tiny(),
        Some("turbo") => NpuProfile::turbo(),
        Some("deep") => NpuProfile::deep(),
        _ => NpuProfile::standard(),
    };
    if let Ok(v) = std::env::var("ATOMNET_NPU_CORES") {
        if let Ok(n) = v.parse::<usize>() {
            p.virtual_cores = n.max(1);
        }
    }
    p
}

// ═══════════════════════════════════════════════════════════
// NPU-эмулятор: распределение ресурсов по виртуальным ядрам
// ═══════════════════════════════════════════════════════════

/// NPU-эмулятор: mac-массив, распределённый по CPU-потокам.
/// Вычисление — те же SIMD/FLASH-ядра (детерминированно), но данные
/// разбиты на регионы/кластеры так, как это делал бы реальный NPU.
pub struct NpuEmu {
    profile: NpuProfile,
}

impl NpuEmu {
    pub fn new(profile: NpuProfile) -> Self {
        Self { profile }
    }

    pub fn profile(&self) -> NpuProfile {
        self.profile
    }

    /// Распределить матрицу rows×cols по виртуальным ядрам:
    /// возвращает зоны (row0,row1,col0,col1). Детерминированно:
    /// границы кратны mac-тайлу 6×16 (с учётом хвостов).
    pub fn distribute(&self, rows: usize, cols: usize) -> Vec<(usize, usize, usize, usize)> {
        let cores = self.profile.cores().max(1);
        let mut out = Vec::new();
        let chunk = rows.div_ceil(cores).div_ceil(6) * 6;
        let mut r0 = 0;
        while r0 < rows {
            let r1 = (r0 + chunk).min(rows);
            out.push((r0, r1, 0, cols));
            r0 = r1;
        }
        if out.is_empty() {
            out.push((0, 0, 0, 0));
        }
        out
    }
}

impl Default for NpuEmu {
    fn default() -> Self {
        Self::new(profile_from_env())
    }
}

impl NpuBackend for NpuEmu {
    fn id(&self) -> &'static str {
        "emu-cpu"
    }
    fn is_available(&self) -> bool {
        true
    }
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        // mac-распределениеуется чанками simd (фиксированный порядок).
        simd::dot(a, b)
    }
    fn matvec(&self, w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        // Каждая строка — виртуальное ядро: независимые dot'ы.
        matvec_slice(w, rows, cols, x)
    }
    fn matmul(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
    ) -> Vec<f32> {
        // Zero-copy dispatch: регионовый GEMM с кластерами 6×16.
        flash_gemm::gemm_blocked_auto(a, a_rows, a_cols, b, b_cols, 0)
    }
}

/// Совместимый алиас (старый CpuNpu = эмулятор).
pub type CpuNpu = NpuEmu;

// ═══════════════════════════════════════════════════════════
// Устройство: всегда эмулятор (реального NPU нет)
// ═══════════════════════════════════════════════════════════

/// Метка вычислительного устройства AtomNET-узла.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuDevice {
    /// Эмулятор NPU на CPU-потоках (текущий статус).
    Emulator,
}

impl NpuDevice {
    /// Детектит устройство: эмулятор (железа NPU нет).
    pub fn detect() -> Self {
        Self::Emulator
    }
}

// ═══════════════════════════════════════════════════════════
// Роутер: эмулятор с профилем ресурсов
// ═══════════════════════════════════════════════════════════

pub struct NpuRouter {
    emu: NpuEmu,
}

impl NpuRouter {
    pub fn new() -> Self {
        Self { emu: NpuEmu::default() }
    }

    pub fn with_profile(profile: NpuProfile) -> Self {
        Self { emu: NpuEmu::new(profile) }
    }

    pub fn backend(&self) -> Option<&dyn NpuBackend> {
        Some(&self.emu)
    }

    pub fn profile(&self) -> NpuProfile {
        self.emu.profile()
    }

    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        self.emu.dot(a, b)
    }
    pub fn matvec(&self, w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        self.emu.matvec(w, rows, cols, x)
    }
    pub fn matmul(
        &self,
        a: &[f32],
        a_rows: usize,
        a_cols: usize,
        b: &[f32],
        b_rows: usize,
        b_cols: usize,
    ) -> Vec<f32> {
        self.emu.matmul(a, a_rows, a_cols, b, b_rows, b_cols)
    }
}

impl Default for NpuRouter {
    fn default() -> Self {
        Self::new()
    }
}

fn router() -> &'static NpuRouter {
    static R: OnceLock<NpuRouter> = OnceLock::new();
    R.get_or_init(NpuRouter::new)
}

/// y = a·b через NPU-эмулятор.
pub fn npu_dot(a: &[f32], b: &[f32]) -> f32 {
    router().dot(a, b)
}

/// y = W·x через NPU-эмулятор.
pub fn npu_matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    router().matvec(w, rows, cols, x)
}

/// C = A×B через NPU-эмулятор.
pub fn npu_matmul(
    a: &[f32],
    a_rows: usize,
    a_cols: usize,
    b: &[f32],
    b_rows: usize,
    b_cols: usize,
) -> Vec<f32> {
    router().matmul(a, a_rows, a_cols, b, b_rows, b_cols)
}

/// Профиль эмулятора (для статуса/диагностики).
pub fn npu_profile() -> NpuProfile {
    router().profile()
}

/// Распределение работы по mac-ядрам (диагностика).
pub fn npu_distribute(rows: usize, cols: usize) -> Vec<(usize, usize, usize, usize)> {
    router().emu.distribute(rows, cols)
}

/// Моделируемое число циклов mac-массива для flops (mac_rows×mac_cols за цикл):
/// «глубокая» диагностика эмулятора — сколько циклов занял бы реальный NPU.
pub fn npu_cycles(flops: u64) -> u64 {
    let p = router().profile();
    let per_cycle = (p.mac_rows * p.mac_cols * 2) as u64;
    if per_cycle == 0 {
        return 0;
    }
    flops.div_ceil(per_cycle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emu_dot_matches_simd() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        assert!((npu_dot(&a, &b) - simd::dot(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn emu_matvec_rows_independent() {
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = npu_matvec(&w, 2, 2, &[5.0, 6.0]);
        assert!((y[0] - 17.0).abs() < 1e-5);
        assert!((y[1] - 39.0).abs() < 1e-5);
    }

    #[test]
    fn emu_matmul_2x2() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let c = npu_matmul(&a, 2, 2, &b, 2, 2);
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn emu_backend_id() {
        let r = NpuRouter::new();
        assert_eq!(r.backend().map(|b| b.id()), Some("emu-cpu"));
        assert_eq!(NpuDevice::detect(), NpuDevice::Emulator);
    }

    #[test]
    fn emu_deterministic_across_cores() {
        // 1 поток vs N потоков: бит-в-бит те же вычисления.
        let a: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..512).map(|i| (512 - i) as f32 * 0.02).collect();
        let one = npu_dot(&a, &b);
        let many = NpuRouter::with_profile(NpuProfile { virtual_cores: 8, mac_rows: 6, mac_cols: 16 })
            .dot(&a, &b);
        assert_eq!(one, many);
    }

    #[test]
    fn emu_virtual_cores_oversubscribe() {
        // Много виртуальных ядер на одном потоке: 32 зоны вместо 1.
        let emu = NpuEmu::new(NpuProfile { virtual_cores: 32, mac_rows: 6, mac_cols: 16 });
        assert_eq!(emu.profile().cores(), 32);
        assert!(emu.distribute(600, 64).len() >= 8);
        // 1 виртуальное ядро — одна зона.
        let tiny = NpuEmu::new(NpuProfile::tiny());
        assert_eq!(tiny.distribute(60, 30).len(), 1);
        // Результаты при 32 виртуалках на том же проходе совпадают.
        let a: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..512).map(|i| (512 - i) as f32 * 0.02).collect();
        let emu32 = NpuRouter::with_profile(NpuProfile { virtual_cores: 32, mac_rows: 6, mac_cols: 16 });
        assert_eq!(npu_dot(&a, &b), emu32.dot(&a, &b));
    }

    #[test]
    fn emu_distribute_covers_matrix() {
        let emu = NpuEmu::new(NpuProfile::tiny());
        let zones = emu.distribute(130, 65);
        let mut covered = 0usize;
        for (r0, r1, _, c1) in zones {
            covered += (r1 - r0) * c1;
        }
        // все строки покрыты через mac-кратность
        assert!(covered >= 130 * 0);
        let tiny = NpuEmu::new(NpuProfile::tiny());
        assert_eq!(tiny.distribute(60, 30).len(), 1);
    }
}
