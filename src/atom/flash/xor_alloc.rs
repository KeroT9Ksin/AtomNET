// src/atom/flash/xor_alloc.rs
//! XorAlloc: аллокация и тиринг на чисто битовой математике (XOR/popcnt).

pub struct XorAlloc;

impl XorAlloc {
    #[inline]
    pub fn liveness(sig: u64) -> u32 { sig.count_ones() }

    #[inline]
    pub fn sim(a: u64, b: u64) -> u32 { 64 - (a ^ b).count_ones() }

    #[inline]
    pub fn bucket(sig: u64) -> u32 { (sig >> 58) as u32 } // топ-6 бит
    
   #[inline]
    pub fn changed(old: u64, new: u64) -> u32 { (old ^ new).count_ones() }

    /// Агрегатная подпись чанка = XOR подписей строк (без float).
    pub fn chunk_sig(row_sigs: &[u64]) -> u64 {
        row_sigs.iter().fold(0u64, |a, &s| a ^ s)
    }

    /// Tier чисто по битам: 2=hot, 1=warm, 0=ice (match → компактный код).
    pub fn decide(_chunk_sig: u64, heat: u32) -> u8 {
        match heat {
            16..=u32::MAX => 2,
            1..=15 => 1,
            _ => 0,
        }
    }

    /// План тиров для всех чанков (битовая математика).
    pub fn plan(row_sigs: &[u64], heat: &[u32], chunk_rows: usize) -> Vec<u8> {
        let nch = (row_sigs.len() + chunk_rows - 1) / chunk_rows;
        (0..nch)
            .map(|c| {
                let r0 = c * chunk_rows;
                let r1 = (r0 + chunk_rows).min(row_sigs.len());
                let cs = Self::chunk_sig(&row_sigs[r0..r1]);
                Self::decide(cs, heat.get(c).copied().unwrap_or(0))
            })
            .collect()
    }
}
