// src/atom/tok.rs
//! Byte-level BPE токенизатор (словарное «умное словокупотребление»).
//!
//! Замена байт-фолбэк (vocab=256, каждый токен = один байт) на BPE:
//! обученные слияния пар токенов -> слова/конструкции корпуса
//! (например, "the" = токен, "ing" = токен). Декодер гарантирует
//! байт-фолбэк: каждый из 256 байт покрывается одиночным токеном,
//! а слияния — конкатенацией байт, значит decode(encode(x)) == x.
//!
//! Алгоритм: частотный подсчёт пар (left,right) -> count, выбор пары с
//! максимумом (минимум 2 вхождения), линейный merge по всей последовательности.

use std::collections::HashMap;

/// Токенизатор с обученными слияниями и таблицей id -> байты.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// token_bytes[id] = байты, которые кодирует токен id.
    /// id < 256 — одиночные байты; id >= 256 — результат слияния.
    pub token_bytes: Vec<Vec<u8>>,
    /// Слияния в порядке обучения: (left_id, right_id, new_id).
    pub merges: Vec<(usize, usize, usize)>,
}

impl BpeTokenizer {
    pub fn new(merges: Vec<(usize, usize, usize)>, token_bytes: Vec<Vec<u8>>) -> Self {
        Self { token_bytes, merges }
    }

    /// Длина словаря: 256 байт + число слияний.
    pub fn vocab_size(&self) -> usize {
        256 + self.merges.len()
    }

    /// Кодировать байты -> последовательность токенов (BPE).
    /// Применяет слияния в порядке обучения (инкрементальный BPE).
    pub fn encode(&self, data: &[u8]) -> Vec<usize> {
        let mut toks: Vec<usize> = data.iter().map(|&b| b as usize).collect();
        for &(l, r, new) in &self.merges {
            if toks.len() < 2 {
                break;
            }
            let mut out = Vec::with_capacity(toks.len());
            let mut i = 0;
            while i < toks.len() {
                if i + 1 < toks.len() && toks[i] == l && toks[i + 1] == r {
                    out.push(new);
                    i += 2;
                } else {
                    out.push(toks[i]);
                    i += 1;
                }
            }
            toks = out;
        }
        toks
    }

    /// Декодировать токены -> байты (конкатенация по token_bytes).
    pub fn decode(&self, ids: &[usize]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(&self.token_bytes[id]);
        }
        out
    }
}

/// Обучить BPE на байтах: n_merges максимум, пока лучшая пара встречается ≥ 2 раз.
pub fn train_bpe(data: &[u8], n_merges: usize) -> BpeTokenizer {
    let mut token_bytes: Vec<Vec<u8>> = (0..256).map(|i| vec![i as u8]).collect();
    let mut tokens: Vec<usize> = data.iter().map(|&b| b as usize).collect();
    let mut merges: Vec<(usize, usize, usize)> = Vec::new();
    let mut next_id: usize = 256;
    for _ in 0..n_merges {
        if tokens.len() < 2 {
            break;
        }
        let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
        for w in tokens.windows(2) {
            *counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
        // Детерминированный выбор: максимум частоты; при равенстве —
        // меньшая пара (иначе порядок HashMap делает слияния
        // недетерминированными между запусками).
        let (pair, cnt) = match counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| a.0.cmp(b.0)))
            .map(|(p, c)| (*p, *c))
        {
            Some(x) => x,
            None => break,
        };
        if cnt < 2 {
            break;
        }
        let new_id = next_id;
        next_id += 1;
        let mut out = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                out.push(new_id);
                i += 2;
            } else {
                out.push(tokens[i]);
                i += 1;
            }
        }
        merges.push((pair.0, pair.1, new_id));
        let mut v = token_bytes[pair.0].clone();
        v.extend_from_slice(&token_bytes[pair.1]);
        token_bytes.push(v);
        tokens = out;
    }
    BpeTokenizer {
        token_bytes,
        merges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_roundtrip_bytes() {
        let data = b"hello world hello world";
        let tok = train_bpe(data, 4);
        let ids = tok.encode(data);
        assert_eq!(tok.decode(&ids), data);
        assert!(tok.vocab_size() > 256);
    }

    #[test]
    fn bpe_single_byte_fallback() {
        // Пустой ввод -> пустая последовательность (без паники).
        let tok = train_bpe(b"", 8);
        assert!(tok.encode(b"").is_empty());
    }

    #[test]
    fn bpe_deterministic() {
        // Одинаковый корпус -> одинаковые слияния (детерминизм tie-break).
        let data = b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps again and again.";
        let a = train_bpe(data, 32);
        let b = train_bpe(data, 32);
        assert_eq!(a.merges, b.merges, "BPE должен быть детерминированным");
        assert_eq!(a.encode(data), b.encode(data));
    }

    #[test]
    fn bpe_inspect_merges() {
        let path = "data/persona_v3_clean.txt";
        let Ok(data) = std::fs::read(path) else {
            return;
        };
        let tok = train_bpe(&data, 256);
        for (i, &(l, r, new)) in tok.merges.iter().take(20).enumerate() {
            println!(
                "merge {i:3}: [{:>12}] + [{:>12}] -> [{:>12}]",
                String::from_utf8_lossy(&tok.token_bytes[l]),
                String::from_utf8_lossy(&tok.token_bytes[r]),
                String::from_utf8_lossy(&tok.token_bytes[new])
            );
        }
        let sample = "привет, как дела? я работаю с atomnet";
        let ids = tok.encode(sample.as_bytes());
        println!("sample {:?} -> {:?}", sample, ids);
        println!("decoded: {:?}", String::from_utf8_lossy(&tok.decode(&ids)));
        // Слияния должны быть «осмысленными»: хотя бы часть — буквы/слова.
        let meaningful = tok
            .merges
            .iter()
            .filter(|(_, _, new)| {
                tok.token_bytes[*new]
                    .iter()
                    .all(|&b| b.is_ascii_alphanumeric() || b == b' ' || (b >= 0xC0))
            })
            .count();
        println!("meaningful merges: {meaningful}/{}", tok.merges.len());
    }

    #[test]
    fn bpe_real_corpus_roundtrip_and_compression() {
        let path = "data/persona_v3_clean.txt";
        let Ok(data) = std::fs::read(path) else {
            return; // корпус не в репозитории — тест пропускается
        };
        let tok = train_bpe(&data, 256);
        let ids = tok.encode(&data);
        // Критичное свойство: decode(encode(x)) == x на реальном корпусе.
        assert_eq!(tok.decode(&ids), data, "roundtrip на реальном корпусе");
        let ratio = data.len() as f64 / ids.len().max(1) as f64;
        println!(
            "BPE: {} байт -> {} токенов ({:.2} байт/токен), vocab={}",
            data.len(),
            ids.len(),
            ratio,
            tok.vocab_size()
        );
        assert!(ids.len() < data.len(), "BPE должен сжимать: токенов меньше байт");
    }
}
