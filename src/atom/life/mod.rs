// src/atom/life/mod.rs
//! AtomLIFE: жизнь, состояние, cold start, гейты.
//!
//! Сюда входят компоненты, которые хранят состояние жизни,
//! но ещё не являются оптимизаторами весов.
//!
//! Примеры:
//! - AtomLife;
//! - RinVec как живой вектор;
//! - QairoLivingLayer;
//! - XorCursor для наблюдения за живостью и стабильностью.

pub mod life;
pub mod rin_vec;
pub mod qairo_life;
pub mod xor_cursor;
pub mod qairo_cursor;
