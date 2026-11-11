// src/atom/math/mod.rs
//! AtomMATH: чистая математика.
//!
//! Сюда входит только детерминированная математика:
//! - [`crate::atom::math::simd`] — базовые SIMD-операции (dot/norm/mse/axpy/xor);
//! - [`crate::atom::math::enigma`] — сферическая геометрия Enigma;
//! - [`crate::atom::math::micron`] — сжатие и целостность Micron;
//! - [`crate::atom::math::qairo`] — атомарный линейный слой Qairo;
//! - [`crate::atom::math::atom_codec`] — многоуровневое атомарное сжатие AtomCodec;
//! - [`crate::atom::math::tensor`] — тензорный слой: Tensor, GEMM/GEMV, активации,
//!   нормализация, softmax, cross-entropy, linear backward;
//! - [`crate::atom::math::pool`] — kernel pool (очередь SIMD-заданий).
//!
//! Здесь не должно быть:
//! - обучения;
//! - lr / momentum;
//! - управления потоками (кроме kernel pool);
//! - выбора уровня сжатия как политики;
//! - температуры;
//! - маршрутизации экспертов.

pub mod simd;
pub mod enigma;
pub mod micron;
pub mod qairo;
pub mod atom_codec;
pub mod tensor;
pub mod pool;
pub mod sferom;
pub mod atommath;
pub mod atomjit;
pub mod regnum;
pub mod binjit;