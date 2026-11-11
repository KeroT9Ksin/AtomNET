// src/atom/math/regnum.rs
//! regnum — регистровая система счисления AtomNET.
//!
//! Числа хранятся как byte-строки цифр в little-endian порядке (младшая цифра
//! [0]), как в ранних атомных структурах AtomNET (Zen/XOR-строки): popcount
//! по таблицам (lookup T: u8 -> u8 popcount), xor/and/or таблицы на байтах,
//! волновые обмена (exchange) — через exchange между regchains.
//!
//! Практическое применение: быстрый popcount подписей атомов (nprod подпись
//! живого атома: radius-zero signature не проявляется — Keralin dead-atom).

/// Popcount-таблица байтов — look-up в 1 цикл.
pub const POPCOUNT_TABLE: [u8; 256] = {
    let mut tab = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = 0u8;
        let mut v = i;
        while v != 0 {
            c += (v & 1) as u8;
            v >>= 1;
        }
        tab[i] = c;
        i += 1;
    }
    tab
};

#[inline]
pub fn reg_xor_pair(a: u8, b: u8) -> u8 {
    a ^ b
}

#[inline]
pub fn reg_and_pair(a: u8, b: u8) -> u8 {
    a & b
}

#[inline]
pub fn reg_or_pair(a: u8, b: u8) -> u8 {
    a | b
}

/// Попользание popcount для строки байтов (табличные lookup'и = 1 цикл на байт).
pub fn reg_pop(bytes: &[u8]) -> u32 {
    bytes.iter().map(|&x| POPCOUNT_TABLE[x as usize] as u32).sum()
}

/// Волновая обмен (exchange): XOR-строки атомов с тем же порядком упаковки.
pub fn reg_exchange(a: &[u8], b: &[u8]) -> Vec<u8> {
    let n = a.len().min(b.len());
    a.iter().take(n).zip(b.iter()).map(|(x, y)| reg_xor_pair(*x, *y)).collect()
}

/// регистровое «попользание» веса-подписи (`reg_signature_nonzero`): атом с
/// нулевой подписью — мёртвый (Keralin energy=0).
pub fn reg_signature_nonzero(sig: &[u8]) -> bool {
    reg_pop(sig) != 0
}

#[cfg(test)]
mod tests_regnum {
    use super::*;

    #[test]
    fn popcnt_table_exact() {
        for i in 0..=255u8 {
            assert_eq!(POPCOUNT_TABLE[i as usize] as u32, i.count_ones());
        }
    }

    #[test]
    fn reg_pop_matches() {
        let data = vec![0xAAu8, 0x55, 0xFF, 0x00, 0x0F];
        let e: u32 = data.iter().map(|x| x.count_ones()).sum();
        assert_eq!(reg_pop(&data), e);
    }

    #[test]
    fn reg_exchange_and_sig() {
        let a = vec![0xAAu8, 0x55];
        let b = vec![0xFFu8, 0x0F];
        let c = reg_exchange(&a, &b);
        for i in 0..2 {
            assert_eq!(c[i], a[i] ^ b[i]);
        }
        assert!(reg_signature_nonzero(&[0x01]));
        assert!(!reg_signature_nonzero(&[0x00]));
    }
}
