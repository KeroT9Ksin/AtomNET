// src/atom/flash/mod.rs
//! AtomFLASH: упакованная память, слои, страницы, hot/cold, multi-SIMD.
//!
//! Здесь будут жить:
//! - FlashLayer;
//! - FlashPage;
//! - FlashController;
//! - packed rows;
//! - streaming experts;
//! - hot/cold memory policy.

pub mod flash;
pub mod flash_compressed;
pub mod flash_arena;
pub mod flash_gemm;
#[cfg(target_arch = "x86_64")]
pub mod jit;
pub mod xor_alloc;
pub mod flash_facade;
pub mod flash_hw;
pub mod flash_xor;
