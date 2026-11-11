// src/atom/veron.rs
//! Veron: конфигурация и runtime модели (однопоточный режим).
//!
//! Задачи:
//! - читать модель из YAML;
//! - задавать имя модели;
//! - задавать слои;
//! - быть гибким каркасом для будущей сборки нейросети.

use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Глобальное число потоков обучения.
/// 0 = авто (физические ядра из flash_hw::recommend_threads, у respect к
/// ATOMNET_THREADS и affinity). Авто = быстро из коробки без настроек:
/// детерминизм сохраняется, т.к. ядра параллелятся только по строкам
/// с фиксированным порядком суммирования.
static PARALLEL_THREADS: AtomicUsize = AtomicUsize::new(0);

pub fn set_parallel_threads(n: usize) {
    let t = if n == 0 {
        crate::atom::flash::flash_hw::recommend_threads()
    } else {
        n.max(1)
    };
    PARALLEL_THREADS.store(t, Ordering::Relaxed);
    // Прогреваем Veron-пул: постоянные потоки готовы к первому fork.
    crate::atom::math::simd::prewarm_fork(t);
}

pub fn parallel_threads() -> usize {
    let t = PARALLEL_THREADS.load(Ordering::Relaxed);
    if t == 0 {
        crate::atom::flash::flash_hw::recommend_threads()
    } else {
        t
    }
}

fn default_version() -> String {
    "0.1.0".to_string()
}

fn default_precision() -> String {
    "f32".to_string()
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn default_std_dev() -> f32 {
    0.02
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VeronLayerConfig {
    pub name: String,

    /// Тип слоя.
    ///
    /// Например:
    /// - qairo
    /// - qairo_living
    /// - flash
    /// - linear
    /// - embedding
    pub kind: String,

    pub in_features: usize,
    pub out_features: usize,

    #[serde(default)]
    pub bias: bool,

    #[serde(default = "default_std_dev")]
    pub std_dev: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VeronModelConfig {
    pub name: String,

    #[serde(default = "default_version")]
    pub version: String,

    #[serde(default)]
    pub seed: u64,

    #[serde(default = "default_threads")]
    pub threads: usize,

    #[serde(default = "default_precision")]
    pub precision: String,

    #[serde(default)]
    pub layers: Vec<VeronLayerConfig>,

    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

pub struct VeronRuntime {
    pub config: VeronModelConfig,
}

impl VeronRuntime {
    /// Загрузить конфигурацию из YAML-строки.
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let config: VeronModelConfig =
            serde_yaml::from_str(yaml).map_err(|e| e.to_string())?;

        Ok(Self { config })
    }

    /// Сериализовать конфигурацию обратно в YAML.
    pub fn to_yaml(&self) -> Result<String, String> {
        serde_yaml::to_string(&self.config).map_err(|e| e.to_string())
    }

    /// Применить настройки потоков: строит глобальный пул Rayon
    /// (можно один раз за процесс) и публикует число потоков для ядер.
    ///
    /// Ядра AtomNET однопоточны; многопоточность применяется ТОЛЬКО по
    /// строкам/сэмплам (результат элемента не меняется) — качество
    /// обучения сохраняется бит-в-бит.
    pub fn apply_thread_pool(&self) -> Result<(), String> {
        // threads == 0 => авто: физические ядра по топологии (flash_hw).
        let threads = if self.config.threads == 0 {
            crate::atom::flash::flash_hw::recommend_threads()
        } else {
            self.config.threads.max(1)
        };
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|e| e.to_string())?;
        set_parallel_threads(threads);
        Ok(())
    }

    pub fn model_name(&self) -> String {
        self.config.name.clone()
    }

    pub fn threads(&self) -> usize {
        self.config.threads
    }

    pub fn set_threads(&mut self, threads: usize) {
        self.config.threads = threads.max(1);
    }

    pub fn layer_count(&self) -> usize {
        self.config.layers.len()
    }

    pub fn layers_info(&self) -> Vec<(String, String, usize, usize)> {
        self.config
            .layers
            .iter()
            .map(|l| {
                (
                    l.name.clone(),
                    l.kind.clone(),
                    l.in_features,
                    l.out_features,
                )
            })
            .collect()
    }

    pub fn summary(&self) -> String {
        let mut out = String::new();

        out.push_str(&format!("model: {}\n", self.config.name));
        out.push_str(&format!("version: {}\n", self.config.version));
        out.push_str(&format!("precision: {}\n", self.config.precision));
        out.push_str(&format!("seed: {}\n", self.config.seed));
        out.push_str(&format!("threads: {}\n", self.config.threads));
        out.push_str(&format!("layers: {}\n", self.config.layers.len()));

        for (i, layer) in self.config.layers.iter().enumerate() {
            out.push_str(&format!(
                "  layer {}: name={} kind={} in={} out={} bias={}\n",
                i,
                layer.name,
                layer.kind,
                layer.in_features,
                layer.out_features,
                layer.bias,
            ));
        }

        out
    }

    /// Пример конфигурации по умолчанию.
    pub fn default_config(name: String) -> Self {
        let layers = vec![
            VeronLayerConfig {
                name: "up".to_string(),
                kind: "qairo_living".to_string(),
                in_features: 2048,
                out_features: 4096,
                bias: true,
                std_dev: 0.02,
            },
            VeronLayerConfig {
                name: "down".to_string(),
                kind: "qairo_living".to_string(),
                in_features: 4096,
                out_features: 2048,
                bias: true,
                std_dev: 0.02,
            },
        ];

        let mut metadata = BTreeMap::new();
        metadata.insert(
            "description".to_string(),
            "Default Veron AtomNET configuration".to_string(),
        );

        Self {
            config: VeronModelConfig {
                name,
                version: default_version(),
                seed: 42,
                threads: default_threads(),
                precision: default_precision(),
                layers,
                metadata,
            },
        }
    }
}