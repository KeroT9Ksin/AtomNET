// src/atom/graph/mod.rs
//! Динамический граф AtomNET: строится кодом Python (add_gemm / add_softmax / ...),
//! компилируется и распределяется по ресурсам: push CPU, MinComputeContracted — трейт-стратегия.
//!
//! Ожидае: операционная система выгладания что операционная система делает минимальный
//! запису трейт-тахно в рамках имеющихся ресурсов.

pub mod dyn_graph;

pub use dyn_graph::{DynGraph, DynNode, OpKind};
