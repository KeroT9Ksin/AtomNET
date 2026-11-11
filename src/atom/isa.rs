// src/atom/isa.rs
//! Виртуальный ISA (виртуализация AVX2/FMA на уровне обработки).
//!
//! Каждый обрабатывающий элемент (виртуальное NPU-ядро, воркер fork-пула)
//! видит одинаковый виртуальный ISA: AVX2/FMA, 16 YMM-регистров, 2 FMA-порта.
//! Детект один (OnceLock), гейты JIT и SIMD идут через этот модуль —
//! единая точка правды вместо разбросанных `is_x86_feature_detected`.
//!
//! Для тестов tier можно форсировать (`set_virtual_tier("scalar")`) —
//! фолбэки всё равно дают верный результат, только медленнее.

use std::sync::OnceLock;

/// Виртуальный ISA обрабатывающего элемента.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtIsa {
    /// AVX2 (256-бит YMM).
    pub avx2: bool,
    /// FMA (слияние умножение-сложение).
    pub fma: bool,
    /// POPCNT (сканирование бит — binjit/SFEROM).
    pub popcnt: bool,
    /// Регистровый файл: YMM-регистров на виртуальное ядро (16 при AVX2).
    pub ymm_regs: usize,
    /// FMA-портов на ядро (2 на Zen, 1 на прочих AVX2).
    pub fma_ports: usize,
    /// Имя тира ("avx2+fma" | "avx2" | "scalar").
    pub tier: &'static str,
}

impl VirtIsa {
    pub fn scalar() -> Self {
        Self { avx2: false, fma: false, popcnt: false, ymm_regs: 0, fma_ports: 0, tier: "scalar" }
    }
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::is_x86_feature_detected!("avx2");
            let fma = std::is_x86_feature_detected!("fma");
            let popcnt = std::is_x86_feature_detected!("popcnt");
            let tier = if avx2 && fma {
                "avx2+fma"
            } else if avx2 {
                "avx2"
            } else {
                "scalar"
            };
            Self {
                avx2,
                fma,
                popcnt,
                ymm_regs: if avx2 { 16 } else { 0 },
                fma_ports: if avx2 && fma { 2 } else { 0 },
                tier,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::scalar()
        }
    }

    /// Доступен полный AVX2+FMA путь (регистровый GEMM/JIT).
    pub fn has_avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }

    /// Форсированный tier для тестов/бенчмарков (никогда не улучшает
    /// реальное железо — только понижает, проверяя фолбэки).
    pub fn forced(tier: &str) -> Option<Self> {
        match tier {
            "scalar" => Some(Self::scalar()),
            "avx2+fma" => Some(Self {
                avx2: true,
                fma: true,
                popcnt: true,
                ymm_regs: 16,
                fma_ports: 2,
                tier: "avx2+fma",
            }),
            "avx2" => Some(Self {
                avx2: true,
                fma: false,
                popcnt: true,
                ymm_regs: 16,
                fma_ports: 0,
                tier: "avx2",
            }),
            _ => None,
        }
    }
}

static ISA: OnceLock<VirtIsa> = OnceLock::new();
static FORCED: OnceLock<Option<VirtIsa>> = OnceLock::new();

/// Виртуальный ISA текущего обрабатывающего элемента (детект один раз).
pub fn virt_isa() -> VirtIsa {
    let forced = FORCED.get_or_init(|| {
        std::env::var("ATOMNET_ISA_TIER")
            .ok()
            .and_then(|t| VirtIsa::forced(&t))
    });
    match *forced {
        Some(v) => {
            // Форсированный тир никогда не превышает реальное железо:
            // заявление "avx2" на хосте без AVX2 = scalar.
            let real = ISA.get_or_init(VirtIsa::detect);
            if v.avx2 && !real.avx2 {
                VirtIsa::scalar()
            } else if v.fma && !real.fma {
                VirtIsa {
                    avx2: real.avx2,
                    fma: false,
                    popcnt: real.popcnt,
                    ymm_regs: real.ymm_regs,
                    fma_ports: 0,
                    tier: if real.avx2 { "avx2" } else { "scalar" },
                }
            } else {
                v
            }
        }
        None => *ISA.get_or_init(VirtIsa::detect),
    }
}

/// ISA конкретного виртуального ядра (все видят один виртуальный ISA —
/// виртуализация уровня обработки, а не железа).
pub fn core_isa(_core_id: usize) -> VirtIsa {
    virt_isa()
}

/// Регистровый файл виртуального ядра: (akku+темпы) YMM-регистров.
/// JIT-тайл 6×16 использует 12 аккумуляторов + 2 B-регистра + 1 broadcast
/// = 15 из 16 — регистровая система обработки без спилов.
pub fn ymm_regs(_core_id: usize) -> usize {
    virt_isa().ymm_regs
}

/// Доступен AVX2 на виртуальных ядрах?
pub fn has_avx2() -> bool {
    virt_isa().avx2
}

/// Доступен FMA?
pub fn has_fma() -> bool {
    virt_isa().fma
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isa_self_consistent() {
        let isa = virt_isa();
        if isa.avx2 {
            assert_eq!(isa.ymm_regs, 16);
            assert!(isa.fma_ports >= 1);
        } else {
            assert_eq!(isa.tier, "scalar");
        }
        // Поле tier соответствует флагам.
        let expect = if isa.avx2 && isa.fma {
            "avx2+fma"
        } else if isa.avx2 {
            "avx2"
        } else {
            "scalar"
        };
        assert_eq!(isa.tier, expect);
    }

    #[test]
    fn core_isa_matches_global() {
        for c in 0..8 {
            assert_eq!(core_isa(c), virt_isa());
            assert_eq!(ymm_regs(c), virt_isa().ymm_regs);
        }
    }
}
