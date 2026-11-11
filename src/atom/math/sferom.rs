// src/atom_math/sferom.rs
//! SFEROM 0.2 (портировано из KeroT9Ksin/Rineto `sferom.rs`): иерархическая
//! сферическая ассоциативная память CORE → LOBE → nanos.
//!
//!-triple-score маршрутизации (0..1):
//! ```text
//! score = 0.70 · cosine(query, centroid)
//!       + 0.20 · (1 − hamming/64)
//!       + 0.10 · jaccard(query, nano_best)
//! ```
//!
//! Центроиды обновляются EMA на нормированной сфере, recall ищет по ВСЕМ
//! долям (без предфильтра), dim — из модели.

use crate::atom::math::simd;

const WAVE_PHI: u64 = 0x9E3779B97F4A7C15; // золотое сечение
const WAVE_PSI: u64 = 0xC2B2AE3D27D4F4F;

/// Битовый хэш байтов (SimHash-алхимия Ринето): XOR-свёртка с вращением,
/// умножением на нечётную константу и xorshift-перемешиванием.
pub fn rineto_hash_bytes(data: &[u8]) -> u64 {
    let mut state: u64 = WAVE_PHI ^ (data.len() as u64).wrapping_mul(WAVE_PSI);
    for (idx, &byte) in data.iter().enumerate() {
        let mixed = (byte as u64)
            .wrapping_mul(WAVE_PSI)
            .wrapping_add(idx as u64)
            .rotate_left(((idx as u64) % 64) as u32);
        state ^= mixed;
        state = state.rotate_left(13).wrapping_mul(WAVE_PHI);
        state ^= state >> 7;
    }
    state
}

/// SimHash-подпись текста по n-граммам (uni/bi/tri) на хэше Ринето.
pub fn simhash_rineto(text: &str, gram1: bool, gram2: bool, gram3: bool) -> u64 {
    let bytes = text.as_bytes();
    let mut acc = 0u64;
    // Позитивное-отрицательное влияние: каждый бит n-граммы складывается
    // через взвешенный xor в 64-позиционную таблицу.
    let mut table = [0u64; 64];
    if gram1 {
        for (i, &b) in bytes.iter().enumerate() {
            let h = rineto_hash_bytes(&bytes[i..i + 1]);
            for bit in 0..64 {
                if (h >> bit) & 1 == 1 {
                    table[bit] += 1;
                } else {
                    table[bit] = table[bit].wrapping_sub(1);
                }
            }
        }
    }
    if gram2 && bytes.len() >= 2 {
        for i in 0..bytes.len() - 1 {
            let h = rineto_hash_bytes(&bytes[i..i + 2]);
            for bit in 0..64 {
                if (h >> bit) & 1 == 1 {
                    table[bit] += 1;
                } else {
                    table[bit] = table[bit].wrapping_sub(1);
                }
            }
        }
    }
    if gram3 && bytes.len() >= 3 {
        for i in 0..bytes.len() - 2 {
            let h = rineto_hash_bytes(&bytes[i..i + 3]);
            for bit in 0..64 {
                if (h >> bit) & 1 == 1 {
                    table[bit] += 1;
                } else {
                    table[bit] = table[bit].wrapping_sub(1);
                }
            }
        }
    }
    for (i, &v) in table.iter().enumerate() {
        if v > 0 {
            acc |= 1u64 << i;
        }
    }
    acc
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.iter().filter(|x| b.contains(x)).count() as f32;
    let union = (a.len() + b.len()) as f32 - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn sig_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '+' && c != '-')
                .to_lowercase()
        })
        .filter(|word| word.chars().count() > 2)
        .collect()
}

fn sign_vec(text: &str, dim: usize) -> Vec<f32> {
    let sig = simhash_rineto(text, true, true, true);
    (0..dim)
        .map(|i| if ((sig >> (i % 64)) & 1) == 1 { 1.0 } else { -1.0 })
        .collect()
}

/// Иерархическая сферическая ассоциативная память.
pub struct Sferom {
    pub num_lobes: usize,
    pub total_nanos: u64,
    pub dim: usize,
    lobe_names: Vec<String>,
    nanos: Vec<Vec<(u64, String, String)>>,
    centroids: Vec<Vec<f32>>,
    lobe_counts: Vec<u64>,
    radial: Vec<f32>,
    core: Vec<f32>,
}

impl Sferom {
    pub fn new(names: Vec<String>, dim: usize) -> Option<Self> {
        if names.is_empty() || dim == 0 {
            return None;
        }
        let n = names.len();
        Some(Self {
            num_lobes: n,
            total_nanos: 0,
            dim,
            lobe_names: names,
            nanos: vec![Vec::new(); n],
            centroids: vec![vec![0.0; dim]; n],
            lobe_counts: vec![0; n],
            radial: vec![0.0; n],
            core: vec![0.0; dim],
        })
    }

    /// Добавить знание в долю; центроид — EMA (первый пример = центр).
    pub fn learn(&mut self, lobe: usize, text: &str, label: &str) -> Result<(), &'static str> {
        if lobe >= self.num_lobes {
            return Err("SFEROM: доля вне диапазона");
        }
        if text.trim().is_empty() {
            return Err("SFEROM: пустой текст нельзя обучить");
        }
        let dim = self.dim;
        let signature = simhash_rineto(text, true, true, true);
        let mut vector = sign_vec(text, dim);
        normalize(&mut vector);

        self.lobe_counts[lobe] += 1;
        let count = self.lobe_counts[lobe];

        if count == 1 {
            self.centroids[lobe] = vector.clone();
        } else {
            let alpha = if count < 10 { 0.20 } else { 0.10 };
            for i in 0..dim {
                self.centroids[lobe][i] =
                    (1.0 - alpha) * self.centroids[lobe][i] + alpha * vector[i];
            }
            normalize(&mut self.centroids[lobe]);
        }

        if self.total_nanos == 0 {
            self.core = vector.clone();
        } else {
            let alpha = 1.0 / ((self.total_nanos + 1) as f32);
            for i in 0..dim {
                self.core[i] = (1.0 - alpha) * self.core[i] + alpha * vector[i];
            }
            normalize(&mut self.core);
        }

        self.nanos[lobe].push((signature, text.to_string(), label.to_string()));
        self.total_nanos += 1;
        self.radial[lobe] = (self.lobe_counts[lobe] as f32).sqrt();
        Ok(())
    }

    /// Лучшая доля (обёртка route_topk(1)).
    pub fn route(&self, text: &str) -> Option<(usize, String, f32)> {
        self.route_topk(text, 1).first().cloned()
    }

    /// Топ-K долей по три-проверочному score.
    pub fn route_topk(&self, text: &str, k: usize) -> Vec<(usize, String, f32)> {
        if k == 0 || self.dim == 0 {
            return Vec::new();
        }
        let dim = self.dim;
        let query = sign_vec(text, dim);
        let query_words = sig_words(text);
        let query_sig = simhash_rineto(text, true, true, true);

        let mut scores: Vec<(usize, f32)> = Vec::new();
        for lobe in 0..self.num_lobes {
            if self.lobe_counts[lobe] == 0 {
                continue;
            }
            let cosine = simd::dot(&query, &self.centroids[lobe]).clamp(-1.0, 1.0);
            let semantic = (cosine + 1.0) * 0.5;

            let lexical = self.nanos[lobe]
                .iter()
                .map(|(_, stored, _)| jaccard(&query_words, &sig_words(stored)))
                .fold(0.0f32, f32::max);

            let best_hamming = self.nanos[lobe]
                .iter()
                .map(|(s, _, _)| (query_sig ^ *s).count_ones())
                .min()
                .unwrap_or(64);
            let hamming_similarity = 1.0 - (best_hamming as f32 / 64.0);

            let score = 0.70 * semantic + 0.20 * hamming_similarity + 0.10 * lexical;
            scores.push((lobe, score));
        }

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k.min(scores.len()));
        scores
            .into_iter()
            .map(|(id, s)| (id, self.lobe_names[id].clone(), s))
            .collect()
    }

    /// Recall: ВСЕ доли, по Hamming-у сигнатур.
    pub fn recall(&self, text: &str, k: usize) -> Vec<(String, u32)> {
        if k == 0 {
            return Vec::new();
        }
        let signature = simhash_rineto(text, true, true, true);
        let mut hits: Vec<(String, u32)> = Vec::new();
        for lobe in &self.nanos {
            for (stored_sig, _, label) in lobe {
                let dist = (signature ^ stored_sig).count_ones();
                hits.push((label.clone(), dist));
            }
        }
        hits.sort_by_key(|x| x.1);
        hits.truncate(k);
        hits
    }

    pub fn lobe_size(&self, lobe: usize) -> Option<u64> {
        self.lobe_counts.get(lobe).copied()
    }

    pub fn lobe_names(&self) -> Vec<String> {
        self.lobe_names.clone()
    }

    pub fn stats(&self) -> Vec<(String, u64, f32)> {
        self.lobe_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), self.lobe_counts[i], self.radial[i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Sferom {
        let names: Vec<String> = vec!["code".into(), "math".into(), "chat".into()];
        let mut s = Sferom::new(names, 64).expect("sferom new");
        s.learn(0, "как работает Rust iterator", "code").expect("learn code 1");
        s.learn(0, "for loop в Rust", "code").expect("learn code 2");
        s.learn(1, "производная sin x + cos x", "math").expect("learn math 1");
        s.learn(1, "интеграл от x^2", "math").expect("learn math 2");
        s.learn(2, "привет как дела", "chat").expect("learn chat 1");
        s
    }

    #[test]
    fn route_and_recall_smoke() {
        let s = fixture();
        assert_eq!(s.num_lobes, 3);
        assert_eq!(s.total_nanos, 5);
        let (lobe, name, score) = s.route("как работает Rust iterator").expect("route");
        assert_eq!(name, "code", "трещво-трек score lobe={lobe} score={score}");
        assert!(score > 0.0 && score <= 1.05, "score={score}");
        let r = s.recall("Rust iterator", 3);
        assert!(!r.is_empty());
        assert!(r.iter().all(|(_, d)| *d < 64));
    }

    #[test]
    fn empty_state() {
        let s = Sferom::new(vec!["x".to_string()], 32).expect("new");
        assert!(s.route("anything").is_none());
        assert!(s.route_topk("anything", 3).is_empty());
        assert!(s.recall("anything", 3).is_empty());
    }

    #[test]
    fn new_rejects_bad_inputs() {
        assert!(Sferom::new(Vec::new(), 64).is_none());
        assert!(Sferom::new(vec!["a".into()], 0).is_none());
    }

    #[test]
    fn learn_rejects_bad_input() {
        let mut s = Sferom::new(vec!["a".into()], 16).expect("new");
        assert!(s.learn(5, "abc", "lbl").is_err());
        assert!(s.learn(0, "   ", "lbl").is_err());
    }
}
