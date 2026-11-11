//! src/atom/prod — продакшн-обвязка обучения (без тестов).
//!
//! Публичные сценарии, собираемые из публичного API TransformerLM:
//! curriculum-обучение и вывод метрик. Бинарь `src/bin/luna.rs`
//! вызывает этот модуль — «запускаемый продукт» поверх AtomNET.

use crate::atom::model::{decode_with_verifier, generate, LmConfig, TransformerLM};

fn encode2(a: usize, b: usize) -> Vec<usize> {
    let s = a + b;
    vec![a / 10, a % 10, 10, b / 10, b % 10, 11, s / 100, (s / 10) % 10, s % 10]
}

fn encode1(a: usize, b: usize) -> Vec<usize> {
    let s = a + b;
    vec![a, 10, b, 11, s / 10, s % 10]
}

fn rng_next(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 40) as f32) / (1u64 << 24) as f32
}


/// Настроить многопоточность обучения (пул + флаг ядер), как Veron.
pub fn setup_threads(threads: usize) {
    let t = threads.max(1);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(t).build_global();
    crate::atom::veron::set_parallel_threads(t);
    let info = crate::atom::flash::flash_hw::probe();
    println!(
        "[luna] threads={t} tier={} l2={}KB l3={}KB cores_allowed={}",
        crate::atom::flash::flash_hw::kernel_tier(),
        info.l2_bytes / 1024,
        info.l3_bytes / 1024,
        info.allowed_cores
    );
}

fn train_stage2(lm: &mut TransformerLM, epochs: usize, static_dataset: bool) -> usize {
    const VOCAB: usize = 12;
    let temps = [0.03f32, 0.1, 0.25, 0.5, 0.9];
    // статичный датасет: 1500 фиксированных пар; саморазвитие: свежие пары
    let mut fixed: Vec<(usize, usize)> = Vec::new();
    let mut st: u64 = 9;
    for _ in 0..1500 {
        let a = ((rng_next(&mut st).abs() * 90.0) as usize % 90) + 10;
        let b = ((rng_next(&mut st).abs() * 90.0) as usize % 90) + 10;
        fixed.push((a, b));
    }
    let mut state: u64 = 0x55AA_1122_3344_5566;
    for _ in 0..epochs {
        let batch: Vec<(usize, usize)> = if static_dataset {
            fixed.clone()
        } else {
            (0..400)
                .map(|_| {
                    let a = ((rng_next(&mut state).abs() * 90.0) as usize % 90) + 10;
                    let b = ((rng_next(&mut state).abs() * 90.0) as usize % 90) + 10;
                    (a, b)
                })
                .collect()
        };
        for chunk in batch.chunks(42) {
            lm.clear_grads();
            let sc = 1.0 / (chunk.len() as f32 * 8.0);
            for (a, b) in chunk {
                lm.train_sample(&encode2(*a, *b), sc);
            }
            lm.step();
        }
    }
    // оценка на НЕвиданных парах
    let mut ok = 0usize;
    let mut st2: u64 = 0xABCD_EF01_2345_6789;
    for i in 0..120usize {
        let a = ((rng_next(&mut st2).abs() * 90.0) as usize % 90) + 10;
        let b = ((rng_next(&mut st2).abs() * 90.0) as usize % 90) + 10;
        let mut pref = encode2(a, b);
        pref.truncate(6);
        let cand = decode_with_verifier(
            lm,
            &pref,
            VOCAB,
            9,
            5,
            &temps,
            3000 + i as u64,
            |sq| crate::atom::model::verify_sum(sq, 10, 11),
        );
        if cand.len() == 9 && cand[6] * 100 + cand[7] * 10 + cand[8] == a + b {
            ok += 1;
        }
    }
    ok
}


/// Byte-LM «писать»: файл -> BPE-токены -> next-token.
/// Обучение с val-сплитом, LR-warmup + cosine, генерация через тот же BPE.
pub fn run_text_lm(path: &str, epochs: usize, max_bytes: usize, temp: f32, d: usize, blocks: usize, window: usize) {
    // Строгая валидация: неверные параметры -> понятная ошибка сразу,
    // а не тихая поломка глубоко в ядре.
    assert!(epochs >= 1, "run_text_lm: epochs должен быть >= 1");
    assert!(
        temp > 0.0 && temp <= 2.0,
        "run_text_lm: temp должен быть в интервале (0, 2]"
    );
    assert!(
        d >= 16 && d % 8 == 0,
        "run_text_lm: d должен быть >= 16 и кратен 8 (d_head = d/2)"
    );
    assert!(blocks >= 1, "run_text_lm: blocks должен быть >= 1");
    assert!(window >= 16, "run_text_lm: window должен быть >= 16");
    let data = std::fs::read(path).expect("read corpus");
    let data: Vec<u8> = data.into_iter().take(max_bytes).collect();
    if data.len() < 64 {
        panic!("корпус слишком мал: {} байт", data.len());
    }
    // train/val split: последние 10% корпуса — валидация, не видна обучению.
    let split = ((data.len() as f64) * 0.90) as usize;
    // ── BPE: словарь = 256 байт + n_merges слияний; обучаем ТОЛЬКО на train ──
    let n_merges = 256usize;
    let tok = crate::atom::tok::train_bpe(&data[..split], n_merges);
    let vocab = tok.vocab_size();
    println!("[luna] bpe: merges={n_merges} vocab={vocab}");
    let mut cfg = LmConfig::nano(vocab);
    cfg.d_model = d;
    cfg.d_head = d / 2;
    cfg.hidden = d * 2;
    cfg.blocks = blocks;
    let mut lm = TransformerLM::new(&cfg);
    println!("[luna] params: {}", lm.param_count());
    let base_lr = 0.1f32;
    lm.set_lr(base_lr);
    lm.set_momentum(0.9);
    let window = window.max(16);

    let train_tokens = tok.encode(&data[..split]);
    let val_tokens = tok.encode(&data[split..]);
    println!(
        "[luna] text corpus: {} bytes, window={window} tok, train={} val={}",
        data.len(),
        train_tokens.len(),
        val_tokens.len()
    );

    let warmup_epochs = (epochs / 10).max(1);
    let mut best_val = f32::INFINITY;
    let mut best_ep = 0usize;
    let mut state: u64 = 0xCAFE_F00D_1234_5678;
    for ep in 0..epochs {
        // LR: линейный warmup до base_lr, далее cosine-охлаждение к ~0.
        let lr = if (ep as f32) < warmup_epochs as f32 {
            base_lr * ((ep as f32 + 1.0) / (warmup_epochs as f32))
        } else {
            let prog = (ep - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32;
            base_lr * 0.5 * (1.0 + (-prog * std::f32::consts::PI).cos())
        };
        lm.set_lr(lr);

        // 32 случайных окна на эпоху, мини-батчи по 8.
        let mut windows: Vec<Vec<usize>> = Vec::new();
        for _ in 0..32 {
            if train_tokens.len() < window + 1 {
                break;
            }
            let max_start = train_tokens.len() - window - 1;
            let start = ((rng_next(&mut state).abs() * (max_start as f32)) as usize).min(max_start);
            let end = (start + window + 1).min(train_tokens.len());
            windows.push(train_tokens[start..end].to_vec());
        }
        if ep % 25 == 0 {
            // Средний CE по валидационным окнам (avg_loss уже усреднён
            // по позициям окна — нельзя делить ещё раз на число позиций).
            let mut lv = 0.0f32;
            let mut nw = 0usize;
            for s in val_tokens.windows(window + 1).take(8) {
                let seq: Vec<usize> = s.to_vec();
                let fwd = lm.forward(&seq);
                lv += crate::atom::model::avg_loss(&fwd.logits, &seq, vocab);
                nw += 1;
            }
            let vloss = if nw > 0 { lv / nw as f32 } else { f32::INFINITY };
            if vloss < best_val {
                best_val = vloss;
                best_ep = ep;
            }
            println!("[luna] epoch {ep}/{epochs} val_loss={vloss:.2}");
        }
        for chunk in windows.chunks(8) {
            lm.clear_grads();
            let sc = 1.0 / (chunk.len() as f32 * (window as f32 - 1.0));
            for seq in chunk {
                lm.train_sample(seq, sc);
            }
            lm.step();
        }
    }
    println!("[luna] best val loss: {best_val:.2} at epoch {best_ep}");

    // ── Генерация ответа Луны с температурой + валидатор-гейт качества ──
    let q = "<user>иронично и умно: привет, кто ты, какой у тебя характер и что ты умеешь?</user><assistant>";
    let seed_ids: Vec<usize> = tok.encode(q.as_bytes());
    let mut seq: Vec<usize> = seed_ids.clone();
    let mut st: u64 = 0x5EED_0001;
    let mut last = None;
    let mut cache = lm.new_cache();
    for &tid in &seed_ids {
        last = Some(lm.forward_token_cached(tid, &mut cache));
    }
    while seq.len() < seed_ids.len() + 220 {
        let row = last.as_ref().expect("text generation: seed processed");
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        let u = ((st >> 40) as f32) / (1u64 << 24) as f32;
        // repetition penalty: токены из хвоста ответа штрафуются (анти-«эхо»)
        let mut row_pen: Vec<f32> = row.clone();
        let gen = &seq[seed_ids.len()..];
        let tail0: &[usize] = if gen.len() > 64 { &gen[gen.len() - 64..] } else { gen };
        for t in tail0 {
            row_pen[*t] -= 1.3;
        }
        let pred = crate::atom::seq::sample_topk(&row_pen, temp, 8, u);
        seq.push(pred);
        last = Some(lm.forward_token_cached(pred, &mut cache));
    }
    let raw: Vec<u8> = tok.decode(&seq[seed_ids.len()..]);
    let text = String::from_utf8_lossy(&raw).into_owned();
    let close = text.find("</assistant>").unwrap_or(text.len());
    let reply = &text[..close.min(text.len())];
    println!("[luna] LUNA (temp={temp}): {:?}", reply);

    // ── Диагностика: продолжение сида из корпуса (in-distribution) ──
    // Greedy-генерация: модель должна продолжить связным текстом.
    let mid = train_tokens.len() / 2;
    let cseed: Vec<usize> = train_tokens[mid..mid + 24].to_vec();
    let cgen = generate(&lm, &cseed, cseed.len() + 48, vocab);
    println!(
        "[luna] corpus seed: {:?}",
        String::from_utf8_lossy(&tok.decode(&cseed))
    );
    println!(
        "[luna] corpus gen : {:?}",
        String::from_utf8_lossy(&tok.decode(&cgen[cseed.len()..]))
    );

    // ── Валидатор-гейт качества ──
    let valid_tag = text.contains("</assistant>");
    let gen_tokens = &seq[seed_ids.len()..];
    let tail = if gen_tokens.len() > 96 { &gen_tokens[gen_tokens.len() - 96..] } else { gen_tokens };
    let mut best_rep = 0.0f32;
    for p in 1..=8usize {
        if tail.len() > p {
            let mut same = 0usize;
            let mut n = 0usize;
            for i in 0..tail.len() - p {
                if tail[i] == tail[i + p] {
                    same += 1;
                }
                n += 1;
            }
            let r = same as f32 / n as f32;
            if r > best_rep {
                best_rep = r;
            }
        }
    }
    let rep = best_rep;
    let non_empty = !reply.is_empty();
    let passed = valid_tag && non_empty && rep < 0.45;
    println!(
        "[luna] gates: valid_tag={valid_tag}, non_empty={non_empty}, repetition={rep:.2} -> {}",
        if passed { "PASSED" } else { "GARBAGE" }
    );
    let ckpt = "luna_text.atomnet";
    lm.save_checkpoint(ckpt).expect("save text ckpt");
    println!("[luna] text checkpoint saved: {ckpt}");
}
/// Сравнение: саморазвитие (свежие пары+верификатор) против статичного
/// датасета. Стартует обе модели одинаково (этап 1 таблицы).
pub fn run_compare(epochs1: usize, epochs2: usize) {
    const VOCAB: usize = 12;
    let mk = |seed: u64| -> TransformerLM {
        let mut lm = TransformerLM::new(&LmConfig::nano(VOCAB));
        lm.set_lr(0.05);
        let pairs1: Vec<(usize, usize)> =
            (0..10).flat_map(|a| (0..10).map(move |b| (a, b))).collect();
        let mut st = seed;
        for _ in 0..epochs1 {
            let order: Vec<(usize, usize)> = if seed % 2 == 0 { pairs1.clone() } else { pairs1.clone() };
            let _ = rng_next(&mut st);
            for chunk in order.chunks(25) {
                lm.clear_grads();
                let sc = 1.0 / (chunk.len() as f32 * 5.0);
                for (a, b) in chunk {
                    lm.train_sample(&encode1(*a, *b), sc);
                }
                lm.step();
            }
        }
        lm.set_lr(0.08);
        lm.set_momentum(0.9);
        lm
    };
    let mut a = mk(111);
    let mut b = mk(222);
    let ok_static = train_stage2(&mut a, epochs2, true);
    let ok_selfdev = train_stage2(&mut b, epochs2, false);
    println!(
        "[luna] compare (epochs2={epochs2}): static-dataset verifier {ok_static}/120, self-development {ok_selfdev}/120"
    );
    println!(
        "[luna] verdict: self-development > static: {}",
        ok_selfdev > ok_static
    );
}

/// Curriculum-обучение математике (этап 1 + этап 2) с печатью метрик.
pub fn run_math_curriculum(epochs1: usize, epochs2: usize, ckpt: Option<&str>) {
    const VOCAB: usize = 12;
    let temps = [0.03f32, 0.1, 0.25, 0.5, 0.9];
    println!("[luna] cfg: blocks=2 d=32 h=64");
    let mut lm = TransformerLM::new(&LmConfig::nano(VOCAB));
    if let Some(path) = ckpt {
        if std::path::Path::new(path).exists() {
            lm.load_checkpoint(path).expect("resume");
            println!("[luna] resumed from {path}");
        }
    }
    lm.set_lr(0.05);

    // Этап 1: вся таблица однозначного сложения (с расписанием lr)
    let pairs1: Vec<(usize, usize)> = (0..10).flat_map(|a| (0..10).map(move |b| (a, b))).collect();
    for ep in 0..epochs1 {
        if ep == (epochs1 * 7 / 10).max(1) {
            lm.set_lr(0.01);
        }
        for chunk in pairs1.chunks(25) {
            lm.clear_grads();
            let sc = 1.0 / (chunk.len() as f32 * 5.0);
            for (a, b) in chunk {
                lm.train_sample(&encode1(*a, *b), sc);
            }
            lm.step();
        }
    }
    let mut ok1 = 0usize;
    for (a, b) in &pairs1 {
        let prefix = vec![*a, 10, *b, 11];
        let g = generate(&lm, &prefix, 6, VOCAB);
        if g.len() == 6 && g[4] * 10 + g[5] == a + b {
            ok1 += 1;
        }
    }
    println!("[luna] stage1 (table): {ok1}/100 greedy");
    if let Some(path) = ckpt {
        lm.save_checkpoint(path).expect("save stage1");
        println!("[luna] checkpoint saved: {path}");
    }

    // Этап 2: двузначное с переносом (мини-батчи + моментум)
    lm.set_lr(0.08);
    lm.set_momentum(0.9);
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for epoch in 0..epochs2 {
        if epoch == epochs2 / 2 {
            lm.set_lr(0.02);
        }
        if epoch == (epochs2 * 9) / 13 {
            lm.set_lr(0.008);
        }
        let mut batch: Vec<Vec<usize>> = Vec::with_capacity(425);
        for _ in 0..400 {
            let a = ((rng_next(&mut state).abs() * 90.0) as usize % 90) + 10;
            let b = ((rng_next(&mut state).abs() * 90.0) as usize % 90) + 10;
            batch.push(encode2(a, b));
        }
        for (a, b) in pairs1.iter().take(25) {
            batch.push(encode1(*a, *b));
        }
        for chunk in batch.chunks(42) {
            lm.clear_grads();
            let sc = 1.0 / (chunk.len() as f32 * 8.0);
            for seq in chunk.iter() {
                lm.train_sample(seq, sc);
            }
            lm.step();
        }
    }
    let mut ok2 = 0usize;
    let mut g_ok = 0usize;
    let mut seed2: u64 = 0x0FED_CBA9_8765_4321;
    for i in 0..150usize {
        let a = ((rng_next(&mut seed2).abs() * 90.0) as usize % 90) + 10;
        let b = ((rng_next(&mut seed2).abs() * 90.0) as usize % 90) + 10;
        let mut pref = encode2(a, b);
        pref.truncate(6);
        let cand = decode_with_verifier(&lm, &pref, VOCAB, 9, 5, &temps, 2000 + i as u64, |sq| {
            crate::atom::model::verify_sum(sq, 10, 11)
        });
        let g = generate(&lm, &pref, 9, VOCAB);
        if cand.len() == 9 && cand[6] * 100 + cand[7] * 10 + cand[8] == a + b {
            ok2 += 1;
        }
        if g.len() == 9 && g[6] * 100 + g[7] * 10 + g[8] == a + b {
            g_ok += 1;
        }
    }
    println!("[luna] stage2 (two-digit): verifier {ok2}/150, greedy {g_ok}/150");
    if let Some(path) = ckpt {
        lm.save_checkpoint(path).expect("save final");
        println!("[luna] final checkpoint saved: {path}");
    }
}

/// Компактная память диалога фиксированного размера (из «белой книги»):
/// байтовое кольцо последних реплик; выборка обучающих окон смещена
/// к свежим фрагментам (recency-взвешивание) — «долгое эхо» без роста KV.
pub struct RingMemory {
    buf: Vec<u8>,
    cap: usize,
}

impl RingMemory {
    pub fn new(cap: usize) -> Self {
        Self { buf: Vec::new(), cap: cap.max(64) }
    }

    pub fn push_text(&mut self, text: &str) {
        for b in text.as_bytes() {
            if self.buf.len() >= self.cap {
                let drop = self.cap / 4;
                self.buf.drain(..drop);
            }
            self.buf.push(*b);
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Старт окна: смещение из [0..=max], свежесть усиливает частоту.
    pub fn weighted_start(&self, rng: f32, window: usize) -> usize {
        if self.buf.len() <= window + 1 {
            return 0;
        }
        let max_start = self.buf.len() - window - 1;
        // recency: u близко к 1 -> свежий хвост; u близко к 0 -> старое начало
        let u = rng.clamp(0.0, 1.0);
        // recency: u близко к 1 -> свежий хвост; u близко к 0 -> старое начало
        let v = 1.0 - (1.0 - u) * (1.0 - u) * (1.0 - u);
        ((v * max_start as f32) as usize).min(max_start)
    }

    pub fn window(&self, start: usize, window: usize) -> &[u8] {
        let end = (start + window + 1).min(self.buf.len());
        &self.buf[start..end]
    }
}

#[cfg(test)]
mod ring_tests {
    use super::*;

    #[test]
    fn ring_caps_and_keeps_recent() {
        let mut m = RingMemory::new(256);
        m.push_text("привет");
        assert_eq!(m.len(), 12); // utf8 bytes "привет"
        for i in 0..100 {
            m.push_text(&format!("фраза номер {} для заполнения буфера, ", i));
        }
        assert!(m.len() <= 256);
        // свежие данные сохранились (последняя фраза), старые вытеснены
        assert!(String::from_utf8_lossy(&m.buf).contains("99 для заполнения"));
        assert!(!String::from_utf8_lossy(&m.buf).contains("фраза номер 0 для"));
        // окна в границах
        let w = m.weighted_start(1.0, 32);
        assert!(w + 33 <= m.len());
    }

    /// Диагностика: учим text-LM на BPE-токенах и генерируем с сидом
    /// из самого корпуса (in-distribution), чтобы понять, училась ли модель.
    #[test]
    fn diag_text_lm_generation() {
        let path = "data/persona_v3_clean.txt";
        let Ok(data) = std::fs::read(path) else {
            return;
        };
        let split = (data.len() as f64 * 0.9) as usize;
        let tok = crate::atom::tok::train_bpe(&data[..split], 256);
        let vocab = tok.vocab_size();
        let mut cfg = LmConfig::nano(vocab);
        cfg.d_model = 96;
        cfg.d_head = 48;
        cfg.hidden = 192;
        cfg.blocks = 2;
        let mut lm = TransformerLM::new(&cfg);
        lm.set_lr(0.1);
        lm.set_momentum(0.9);
        let train_tokens = tok.encode(&data[..split]);
        let window = 128usize;
        let mut state = 0xCAFE_F00D_1234_5678u64;
        for _ep in 0..40 {
            let mut windows: Vec<Vec<usize>> = Vec::new();
            for _ in 0..32 {
                let max_start = train_tokens.len() - window - 1;
                let start =
                    ((rng_next(&mut state).abs() * max_start as f32) as usize).min(max_start);
                windows.push(train_tokens[start..start + window + 1].to_vec());
            }
            for chunk in windows.chunks(8) {
                lm.clear_grads();
                let sc = 1.0 / (chunk.len() as f32 * (window as f32 - 1.0));
                for seq in chunk {
                    lm.train_sample(seq, sc);
                }
                lm.step();
            }
        }
        // Сид из корпуса (in-distribution): модель должна продолжить связно.
        let seed: Vec<usize> = train_tokens[1000..1032].to_vec();
        let out = generate(&lm, &seed, seed.len() + 64, vocab);
        println!(
            "seed: {:?}",
            String::from_utf8_lossy(&tok.decode(&seed))
        );
        println!(
            "gen : {:?}",
            String::from_utf8_lossy(&tok.decode(&out[seed.len()..]))
        );
        // Мета-промпт (out-of-distribution) — для сравнения.
        let q = "<user>иронично и умно: привет, кто ты?</user><assistant>";
        let qids = tok.encode(q.as_bytes());
        let qout = generate(&lm, &qids, qids.len() + 64, vocab);
        println!(
            "meta : {:?}",
            String::from_utf8_lossy(&tok.decode(&qout[qids.len()..]))
        );
    }
}
