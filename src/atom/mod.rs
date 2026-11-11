// src/atom/mod.rs
//! AtomNET core architecture.
//!
//! AtomMATH   — чистая математика.
//! AtomLIFE   — жизнь, состояние, lifetime, cold start.
//! AtomOPTIM  — оптимизаторы и обучение.
//! AtomFLASH  — упакованные слои, страницы, память, hot/cold.
//! Veron      — конфигурация, runtime, потоки, YAML, сборка модели.
//! Physics    — физический контроль, Hexiron.
//! ISA        — виртуальный ISA: AVX2/FMA-регистровый файл на виртуальное ядро.

pub mod isa;
pub mod array;
pub mod math;
pub mod npu;
pub mod life;
pub mod optim;
pub mod flash;
pub mod veron;
pub mod physics;
pub mod train;
pub mod tok;
pub mod seq;
pub mod graph;
pub mod nn;
pub mod model;
pub mod prod;
pub mod torchx;
