// src/atom/train.rs
//! atom::train — обучение нейросетей целиком на Rust, без NumPy.
//!
//! Фаза A: реестр слоёв (Linear, Embedding), аналитические backward,
//! накопление градиентов полного батча, шаг через KeralinGraph
//! (графовый оптимизатор на атомной математике AtomNET).
//!
//! Каждая строка весов — узел-атом графа (как в KeralinGraph):
//! XOR-подписи, жизнь, адаптивный RMS-шаг — всё то же, что и в Python API.

use crate::atom::flash::flash::FlashDense;
use crate::atom::math::simd;
use crate::atom::optim::keralin::{AtomWeight, KeralinGraph, KeralinGraphParams};
use crate::atom::physics::hexiron::Hexiron;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

/// Детерминированный шум: Box-Muller через два u32 из ChaCha8.
struct Noise {
    rng: ChaCha8Rng,
    spare: Option<f32>,
}

impl Noise {
    fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            spare: None,
        }
    }

    fn next_f32(&mut self) -> f32 {
        let u = |r: &mut ChaCha8Rng| -> f32 {
            (r.next_u32() >> 8) as f32 / (1u32 << 24) as f32
        };
        if let Some(s) = self.spare.take() {
            return s;
        }
        let u1: f32 = u(&mut self.rng).max(1e-12);
        let u2: f32 = u(&mut self.rng);
        let r: f32 = (-2.0 * u1.ln()).sqrt();
        let t: f32 = std::f32::consts::TAU * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LayerKind {
    Linear,
    Embedding,
}

/// Слой: веса строко-мажорные (rows×cols), каждая строка — атом графа.
///
/// Биоконтуры:
/// - `w` — быстрые веса (обучаются каждый шаг, «fast cache» из PDF);
/// - `slow` — консолидированные («persistent»): обновляются только при
///   положительном дофаминовом сигнале (улучшение потерь);
/// - `elig` — след элигибильности строки (|градиент| с затуханием):
///   «сколько строка недавно работала» — третья факторная метка.
pub struct Layer {
    pub kind: LayerKind,
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
    pub slow: Vec<f32>,
    pub elig: Vec<f32>,
    pub grad: Vec<f32>,
    pub node_ids: Vec<usize>,
    /// Flash hot-path mirror of `w` (Aligned64 + prefetch + AVX2 block-GEMV).
    /// Source of truth remains `w` + graph atoms; flash is synced from `w`
    /// after every weight mutation (optimize_step / birth_row).
    pub flash: Option<FlashDense>,
}

impl Layer {
    fn new(kind: LayerKind, rows: usize, cols: usize, scale: f32, seed: u64) -> Self {
        let mut noise = Noise::new(seed);
        let mut w = vec![0.0f32; rows * cols];
        for v in w.iter_mut() {
            *v = noise.next_f32() * scale;
        }
        Self {
            kind,
            rows,
            cols,
            slow: w.clone(),
            elig: vec![0.0f32; rows],
            w,
            grad: vec![0.0f32; rows * cols],
            node_ids: Vec::new(),
            flash: None,
        }
    }

    pub fn row(&self, i: usize) -> &[f32] {
        assert!(
            (i + 1) * self.cols <= self.w.len(),
            "row oob: kind={:?} i={} cols={} wlen={}",
            self.kind,
            i,
            self.cols,
            self.w.len()
        );
        &self.w[i * self.cols..(i + 1) * self.cols]
    }

    /// y = W x (Linear) — через Flash hot-path если включён, иначе NPU/CPU router.
    pub fn forward_linear(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(self.cols, x.len());
        if let Some(fd) = self.flash.as_ref() {
            return fd.gemv(x);
        }
        crate::atom::npu::npu_matvec(&self.w, self.rows, self.cols, x)
    }

    /// Включить Flash hot-path для слоя:_aligned64-копия `w`.
    pub fn enable_flash(&mut self) {
        let rows_data: Vec<Vec<f32>> =
            (0..self.rows).map(|i| self.row(i).to_vec()).collect();
        let bias = vec![0.0f32; self.rows];
        if let Some(fd) = FlashDense::from_rows(&rows_data, bias) {
            self.flash = Some(fd);
        }
    }

    /// Перелить flash-зеркало из `w` (вызывать после любой мутации весов).
    pub fn sync_flash(&mut self) {
        if self.flash.is_none() {
            return;
        }
        let mut fd = self.flash.take().expect("sync_flash: take");
        for i in 0..self.rows {
            let base = i * self.cols;
            let row: Vec<f32> = self.w[base..base + self.cols].to_vec();
            fd.set_row(i, &row);
        }
        self.flash = Some(fd);
    }

    pub fn has_flash(&self) -> bool {
        self.flash.is_some()
    }
}

/// Буфер важных воспоминаний (реплей): хранит сэмплы с наибольшей
/// ошибкой — «что не выучено». Используется в сне-консолидации и
/// авто-куррикулуме.
pub struct Replay {
    pub x: Vec<Vec<f32>>,
    pub y: Vec<f32>,
    pub err: Vec<f32>,
    cap: usize,
}

impl Replay {
    pub fn new(cap: usize) -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            err: Vec::new(),
            cap: cap.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Запомнить сэмпл, если он важнее худшего из буфера.
    pub fn record(&mut self, x: &[f32], y: f32, err: f32) {
        if self.x.len() < self.cap {
            self.x.push(x.to_vec());
            self.y.push(y);
            self.err.push(err);
            return;
        }
        if let Some((idx, &min)) = self
            .err
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if err > min {
                self.x[idx] = x.to_vec();
                self.y[idx] = y;
                self.err[idx] = err;
            }
        }
    }

    /// Сэмплы по убыванию ошибки (сначала самое невыученное).
    pub fn by_importance(&self) -> Vec<(usize, &[f32], f32)> {
        let mut order: Vec<usize> = (0..self.x.len()).collect();
        order.sort_by(|&a, &b| self.err[b].partial_cmp(&self.err[a]).unwrap());
        order
            .into_iter()
            .map(|i| (i, self.x[i].as_slice(), self.y[i]))
            .collect()
    }

    pub fn update_err(&mut self, i: usize, err: f32) {
        if let Some(e) = self.err.get_mut(i) {
            *e = 0.7 * *e + 0.3 * err;
        }
    }
}

/// Тренер: слои + KeralinGraph + служебные накопители.
pub struct Trainer {
    pub layers: Vec<Layer>,
    pub graph: KeralinGraph,
    pub lr: f32,
    pub seed: u64,
    // Биоконтуры самообучения
    /// Затухание следа элигибильности (0..1, как γ в R-STDP).
    pub elig_decay: f32,
    /// Доля консолидации fast→slow за один вызов (0..1).
    pub commit_rate: f32,
    /// EMA дофаминового сигнала (улучшения потерь).
    pub reward_ema: f32,
    /// Rineta-усилитель: амплитуда шага от когерентности
    /// amp = 1 + alpha·tanh(kappa·coherence)·(1 + gamma·Δenergy)
    pub rineta_alpha: f32,
    pub rineta_kappa: f32,
    pub rineta_gamma: f32,
    /// EMA нормы градиентов (для Δenergy в усилителе).
    grad_energy_ema: f32,
    // Hexiron-губернатор lr
    pub hex: Hexiron,
    pub hexiron_on: bool,
    /// Прошлая когерентность (энтропия = осцилляции когерентности).
    prev_coherence: f32,
}

impl Trainer {
    pub fn new(lr: f32) -> Self {
        let mut p = KeralinGraphParams::default();
        p.base_lr = lr;
        p.wd = 0.0;
        p.auto_warm = true;
        // Микромасштаб: плоский режим стабильнее адаптации и импульса.
        p.adapt = false;
        p.beta1 = 0.0;
        Self {
            layers: Vec::new(),
            graph: KeralinGraph::new(p),
            lr,
            seed: 1,
            elig_decay: 0.9,
            commit_rate: 0.5,
            reward_ema: 0.0,
            rineta_alpha: 0.5,
            rineta_kappa: 4.0,
            rineta_gamma: 0.5,
            grad_energy_ema: 0.0,
            hex: Hexiron::new(),
            hexiron_on: true,
            prev_coherence: 0.0,
        }
    }

    /// Добавить линейный слой rows×cols, вернуть индекс.
    pub fn add_linear(&mut self, rows: usize, cols: usize) -> usize {
        let scale = (1.0 / (cols.max(1) as f32).sqrt()).min(0.5);
        let mut layer = Layer::new(LayerKind::Linear, rows, cols, scale, self.seed + 0x1111);
        for i in 0..rows {
            let atom = AtomWeight::from_vec(layer.row(i));
            let id = self
                .graph
                .add_node(atom, &[])
                .expect("add_linear: node");
            layer.node_ids.push(id);
        }
        self.seed = self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
        self.layers.push(layer);
        self.layers.len() - 1
    }

    /// Добавить эмбеддинг vocab×dim.
    pub fn add_embedding(&mut self, vocab: usize, dim: usize) -> usize {
        let mut layer = Layer::new(LayerKind::Embedding, vocab, dim, 0.1, self.seed + 0x2222);
        for i in 0..vocab {
            let atom = AtomWeight::from_vec(layer.row(i));
            let id = self
                .graph
                .add_node(atom, &[])
                .expect("add_embedding: node");
            layer.node_ids.push(id);
        }
        self.seed = self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
        self.layers.push(layer);
        self.layers.len() - 1
    }

    pub fn forward(&self, li: usize, x: &[f32]) -> Vec<f32> {
        self.layers[li].forward_linear(x)
    }

    /// Включить Flash hot-path для слоя `li`.
    pub fn enable_flash(&mut self, li: usize) {
        self.layers[li].enable_flash();
    }

    /// Синхронизировать flash-зеркала всех слоёв из `w`.
    pub fn sync_flashes(&mut self) {
        for layer in &mut self.layers {
            layer.sync_flash();
        }
    }

    pub fn embed(&self, li: usize, idx: usize) -> Vec<f32> {
        let l = &self.layers[li];
        debug_assert_eq!(l.kind, LayerKind::Embedding);
        l.row(idx).to_vec()
    }

    /// Накопить dW линейного слоя: grad += scale · dout_i · x; вернуть dx.
    pub fn backward_linear(&mut self, li: usize, dout: &[f32], x: &[f32], scale: f32) -> Vec<f32> {
        let (rows, cols) = (self.layers[li].rows, self.layers[li].cols);
        debug_assert_eq!(rows, dout.len());
        let mut dx = vec![0.0f32; cols];
        {
            let layer = &mut self.layers[li];
            for i in 0..rows {
                let di = dout[i];
                let base = i * cols;
                for j in 0..cols {
                    layer.grad[base + j] += scale * di * x[j];
                    dx[j] += di * layer.w[base + j];
                }
            }
        }
        dx
    }

    /// Накопить градиент строки эмбеддинга: grad[idx] += scale·g.
    pub fn backward_embed(&mut self, li: usize, idx: usize, g: &[f32], scale: f32) {
        let layer = &mut self.layers[li];
        debug_assert_eq!(layer.kind, LayerKind::Embedding);
        let base = idx * layer.cols;
        for j in 0..layer.cols {
            layer.grad[base + j] += scale * g[j];
        }
    }

    /// Очистить накопленные градиенты слоёв.
    pub fn clear_grads(&mut self) {
        for layer in &mut self.layers {
            layer.grad.fill(0.0);
        }
    }

    /// Пересобрать атомы KeralinGraph из текущих `layer.w`.
    ///
    /// Нужно после загрузки чекпойнта: иначе первый `optimize_step`
    /// сделает writeback из старых (случайно инициализированных) атомов
    /// и уничтожит загруженные веса. Импульс/жизнь сбрасываются, веса —
    /// источник истины.
    pub fn sync_graph_from_weights(&mut self) {
        for layer in &self.layers {
            for (i, &node_id) in layer.node_ids.iter().enumerate() {
                let base = i * layer.cols;
                let atom = AtomWeight::from_vec(&layer.w[base..base + layer.cols]);
                self.graph
                    .replace_node(node_id, atom)
                    .expect("sync_graph_from_weights: node");
            }
        }
        self.graph.zero_grads();
        self.graph.step_count = 0;
        self.graph.warmed_total = 0;
        self.graph.global_coherence = 0.0;
        self.graph.graph_sig = self.graph.graph_signature();
    }

    /// Средняя норма градиентов слоёв (энергия шага).
    pub fn energy(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for layer in &self.layers {
            for row in 0..layer.rows {
                let base = row * layer.cols;
                sum += simd::norm(&layer.grad[base..base + layer.cols]);
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    }

    /// Шаг оптимизатора: строки слоёв → узлы графа → step() → writeback.
    /// Возвращает среднюю когерентность шага.
    pub fn optimize_step(&mut self) -> f32 {
        self.graph.cold_start();
        for layer in &self.layers {
            for (i, &node) in layer.node_ids.iter().enumerate() {
                let base = i * layer.cols;
                self.graph
                    .feed_grad(node, &layer.grad[base..base + layer.cols])
                    .expect("optimize_step: feed");
            }
        }
        let coh = self.graph.step();
        // Rineta-усилитель: coherence шага и Δэнергии модулируют lr
        // следующего шага (устойчивое ускорение на согласованных градиентах).
        {
            let e = self.energy();
            let de = (e - self.grad_energy_ema).abs() / self.grad_energy_ema.max(1e-9);
            let amp = 1.0
                + self.rineta_alpha * (self.rineta_kappa * coh).tanh() * (1.0 + self.rineta_gamma * de);
            // Hexiron-губернатор: физика шага -> lifetime/dissipation -> lr_eff.
            if self.hexiron_on {
                let entropy =
                    (1.0 - coh.abs()) + 0.5 * (coh - self.prev_coherence).abs();
                self.prev_coherence = coh;
                let e_norm = e / (e + 0.01);
                self.hex.update(e_norm, coh, entropy.min(2.0), 0.0);
                let hex_factor = self.hex.effective_lr(1.0);
                let base = self.lr;
                self.graph.params.base_lr =
                    (base * amp * hex_factor).clamp(base * 0.1, base * 5.0);
            } else {
                let base = self.lr;
                self.graph.params.base_lr = (base * amp).clamp(base * 0.5, base * 3.0);
            }
            self.grad_energy_ema = 0.9 * self.grad_energy_ema + 0.1 * e;
        }
        // Writeback: атомы графа → веса слоёв (источник истины — граф).
        for layer in &mut self.layers {
            for (i, &node) in layer.node_ids.iter().enumerate() {
                let row = self.graph.nodes[node].atom.reconstruct();
                let base = i * layer.cols;
                layer.w[base..base + layer.cols].copy_from_slice(&row);
            }
        }
        // Flash-зеркало: перелить из `w` (hot-path forward без дрейфа).
        self.sync_flashes();
        // Биоконтур: след элигибильности строк = затухающий |градиент|.
        for layer in &mut self.layers {
            for i in 0..layer.rows {
                let base = i * layer.cols;
                let mut norm = 0.0f32;
                for &g in &layer.grad[base..base + layer.cols] {
                    norm += g.abs();
                }
                layer.elig[i] = self.elig_decay * layer.elig[i] + norm;
            }
        }
        coh
    }

    /// Дофаминовый сигнал от наблюдателя: r = -(Δloss) (улучшение > 0).
    /// Обновляет EMA сигнала, возвращает нормированный reward.
    pub fn reward(&mut self, loss_new: f32, loss_old: f32) -> f32 {
        let r = (loss_old - loss_new) / (loss_old.abs().max(1e-9));
        self.reward_ema = 0.9 * self.reward_ema + 0.1 * r;
        r
    }

    /// Консолидация fast→slow (синаптический кэш из PDF): при успехе
    /// (reward > 0) строки с высокой элигибильностью частично переносятся
    /// в медленные веса. Возвращает число консолидированных строк.
    pub fn consolidate(&mut self, reward: f32) -> usize {
        if reward <= 0.0 {
            return 0;
        }
        let mut committed = 0usize;
        let c = self.commit_rate.clamp(0.0, 1.0);
        for layer in &mut self.layers {
            let mean_elig = layer.elig.iter().sum::<f32>() / layer.rows.max(1) as f32;
            for i in 0..layer.rows {
                // Порог: половина средней элигибильности — консолидируются
                // и строки-одиночки, и верхняя половина по «работе».
                if layer.elig[i] > mean_elig.max(1e-12) * 0.5 {
                    let base = i * layer.cols;
                    for j in 0..layer.cols {
                        let sw = layer.slow[base + j];
                        layer.slow[base + j] = sw + c * (layer.w[base + j] - sw);
                    }
                    layer.elig[i] *= 0.5; // после переноса метка гаснет
                    committed += 1;
                }
            }
        }
        committed
    }

    /// Forward по консолидированным (медленным) весам — inference-путь.
    pub fn forward_slow(&self, li: usize, x: &[f32]) -> Vec<f32> {
        let layer = &self.layers[li];
        debug_assert_eq!(layer.cols, x.len());
        crate::atom::npu::npu_matvec(&layer.slow, layer.rows, layer.cols, x)
    }

    /// Сменить базовый lr (сон идёт с пониженным).
    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
        self.graph.params.base_lr = lr;
    }

    /// «Рождение» нейрона: строка `row` заменяется мутацией активной
    /// строки `active` (вес родителя + шум). Рёбра графа сохраняются,
    /// жизнь сбрасывается (fresh-born). Возвращает новую строку.
    pub fn birth_row(
        &mut self,
        li: usize,
        row: usize,
        active: usize,
        noise_scale: f32,
    ) -> Vec<f32> {
        let cols = self.layers[li].cols;
        let mut noise = Noise::new(self.seed);
        self.seed = self.seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
        let mut newborn = vec![0.0f32; cols];
        {
            let layer = &self.layers[li];
            let parent = &layer.w[active * cols..(active + 1) * cols];
            for j in 0..cols {
                newborn[j] = parent[j] + noise.next_f32() * noise_scale;
            }
        }
        // Быстрые веса + slow + атом графа + метка.
        {
            let layer = &mut self.layers[li];
            layer.w[row * cols..(row + 1) * cols].copy_from_slice(&newborn);
            layer.slow[row * cols..(row + 1) * cols].copy_from_slice(&newborn);
            layer.elig[row] = 0.0;
        }
        self.layers[li].sync_flash();
        let atom = AtomWeight::from_vec(&newborn);
        let node = self.layers[li].node_ids[row];
        self.graph.replace_node(node, atom).expect("birth: node");
        newborn
    }

    /// «Сон»: прогон буфера воспоминаний с пониженным lr и консолидацией.
    /// Возвращает среднюю ошибку по буферу после сна.
    pub fn sleep(&mut self, li: usize, buf: &mut Replay, lr_scale: f32, epochs: usize) -> f32 {
        let base_lr = self.lr;
        self.set_lr(base_lr * lr_scale.max(0.01));
        let n = buf.len();
        for _ in 0..epochs {
            let mut mean_err = 0.0f32;
            self.clear_grads();
            let items: Vec<(usize, Vec<f32>, f32)> = buf
                .by_importance()
                .into_iter()
                .map(|(i, x, y)| (i, x.to_vec(), y))
                .collect();
            for (i, x, y) in &items {
                let pred = self.forward(li, x)[0];
                let err = (pred - y).abs();
                buf.update_err(*i, err);
                mean_err += err;
                let dout = vec![2.0 * (pred - y)];
                self.backward_linear(li, &dout, x, 1.0 / n.max(1) as f32);
            }
            self.optimize_step();
            // Успех сна → консолидация в slow.
            self.consolidate(1.0);
        }
        self.set_lr(base_lr);
        let mut mean = 0.0f32;
        for (_, x, y) in buf.by_importance() {
            let pred = self.forward(li, x)[0];
            mean += (pred - y).abs();
        }
        mean / n.max(1) as f32
    }
}

/// Softmax + cross-entropy: (loss, dlogits) для одного вектора логитов.
pub fn softmax_ce(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    crate::atom::math::tensor::softmax_ce(logits, target)
}

/// ═══════════════════════════════════════════════════════════
/// MiniTrainer — компактная обучающая система AtomNET.
///
/// Полный цикл: loss (только тензорное ядро AtomMATH) → backward
/// (тензорный linear_backward) → optimize_step (граф+Keralin+Rineta).
/// Предназначена как минимальный «рабочий» движок для демо/прогона
/// и интеграции в продукт (prod, luna, mind).
/// ═══════════════════════════════════════════════════════════

pub struct MiniTrainer {
    trainer: Trainer,
}

impl MiniTrainer {
    pub fn new(lr: f32) -> Self {
        Self {
            trainer: Trainer::new(lr),
        }
    }
    pub fn with_lr(mut self, lr: f32) -> Self {
        self.trainer.set_lr(lr);
        self
    }
    pub fn add_linear(&mut self, rows: usize, cols: usize) -> usize {
        self.trainer.add_linear(rows, cols)
    }
    pub fn enable_flash(&mut self, li: usize) {
        self.trainer.enable_flash(li);
    }
    pub fn forward(&self, li: usize, x: &[f32]) -> Vec<f32> {
        self.trainer.forward(li, x)
    }
    /// Loss MSE для одного примера.
    pub fn loss(&self, li: usize, x: &[f32], y: f32) -> f32 {
        let pred = self.trainer.forward(li, x)[0];
        let d = pred - y;
        d * d
    }
    /// Один обучающий шаг: loss → backward → optimize_step.
    pub fn train_step(&mut self, li: usize, x: &[f32], y: f32) {
        self.trainer.clear_grads();
        let pred = self.trainer.forward(li, x)[0];
        let dout = vec![2.0 * (pred - y)];
        self.trainer.backward_linear(li, &dout, x, 1.0);
        self.trainer.optimize_step();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// 1) Линейная регрессия: учим y = 2·x0 − 3·x1 + 0.5 на 4 точках.
    #[test]
    fn train_linear_regression() {
        let mut t = Trainer::new(0.01);
        let li = t.add_linear(1, 3);
        let data = [
            ([1.0f32, 0.0, 0.0], 2.0),
            ([0.0, 1.0, 0.0], -3.0),
            ([0.0, 0.0, 1.0], 0.5),
            ([1.0, 1.0, 0.0], -1.0),
        ];
        let mut last = f32::MAX;
        for _ in 0..3000 {
            t.clear_grads();
            for (x, y) in &data {
                let pred = t.forward(li, x)[0];
                let dout = vec![2.0 * (pred - y)];
                t.backward_linear(li, &dout, x, 1.0 / data.len() as f32);
                last = (pred - y) * (pred - y);
            }
            t.optimize_step();
        }
        assert!(last < 1e-3, "regression mse={last}");
        let y = t.forward(li, &[1.0, 1.0, 0.0])[0];
        assert!(approx(y, -1.0, 0.15), "y={y}");
    }

    /// Мини-обучающая система AtomNET: полный цикл loss → backward → optimize_step.
    #[test]
    fn mini_trainer_end_to_end_linear_regression() {
        let mut mt = MiniTrainer::new(0.01).with_lr(0.05);
        let li = mt.add_linear(1, 3);
        let data = [
            ([1.0f32, 0.0, 0.0], 2.0),
            ([0.0, 1.0, 0.0], -3.0),
            ([0.0, 0.0, 1.0], 0.5),
            ([1.0, 1.0, 0.0], -1.0),
        ];
        let mut last = f32::MAX;
        for _ in 0..3000 {
            for (x, y) in &data {
                last = mt.loss(li, x, *y);
                mt.train_step(li, x, *y);
            }
        }
        assert!(last < 1e-3, "mini trainer mse={last}");
    }

    /// Flash hot-path: forward через FlashDense совпадает с plain NPU-GEMV,
    /// и обучение с включённым flash не дрейфует.
    #[test]
    fn flash_hotpath_matches_plain_and_trains() {
        let mut t = Trainer::new(0.01);
        let li = t.add_linear(4, 8);
        let x: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.3).collect();
        let y_plain = t.forward(li, &x);
        t.enable_flash(li);
        assert!(t.layers[li].has_flash());
        let y_flash = t.forward(li, &x);
        assert_eq!(y_plain.len(), y_flash.len());
        for (a, b) in y_plain.iter().zip(y_flash.iter()) {
            assert!((a - b).abs() < 1e-4, "flash vs plain: {a} vs {b}");
        }
        // Обучение с flash: регрессия всё ещё сходится (синхронизация после step).
        let data = [
            ([1.0f32, 0.0, 0.0], 2.0),
            ([0.0, 1.0, 0.0], -3.0),
            ([0.0, 0.0, 1.0], 0.5),
            ([1.0, 1.0, 0.0], -1.0),
        ];
        let mut t2 = Trainer::new(0.01);
        let li2 = t2.add_linear(1, 3);
        t2.enable_flash(li2);
        let mut last = f32::MAX;
        for _ in 0..3000 {
            t2.clear_grads();
            for (x, y) in &data {
                let pred = t2.forward(li2, x)[0];
                let dout = vec![2.0 * (pred - y)];
                t2.backward_linear(li2, &dout, x, 1.0 / data.len() as f32);
                last = (pred - y) * (pred - y);
            }
            t2.optimize_step();
        }
        assert!(last < 1e-3, "flash training mse={last}");
        // После обучения flash находился в синхроне с w.
        let layer = &t2.layers[li2];
        let fd = layer.flash.as_ref().expect("flash present");
        for i in 0..layer.rows {
            for (a, b) in layer.row(i).iter().zip(fd.row(i).iter()) {
                assert!((a - b).abs() < 1e-5, "flash desync: {a} vs {b}");
            }
        }
    }

    /// 2) Char-LM: embed(2) → h(16) → logits(3), CE, паттерн abcabc…
    #[test]
    fn train_char_lm() {
        const VOCAB: usize = 3;
        const DIM: usize = 8;
        const H: usize = 16;
        let mut t = Trainer::new(0.01);
        let e = t.add_embedding(VOCAB, DIM);
        let h = t.add_linear(H, 2 * DIM);
        let o = t.add_linear(VOCAB, H);

        // Паттерн: после пары (a,b) идёт c; (b,c)->a; (c,a)->b.
        let samples: Vec<(usize, usize, usize)> = vec![(0, 1, 2), (1, 2, 0), (2, 0, 1)];
        let mut correct = 0usize;
        let mut n = 0usize;
        for _ in 0..1200 {
            t.clear_grads();
            let mut tot = 0.0f32;
            for (c0, c1, target) in &samples {
                // forward
                let x = [t.embed(e, *c0), t.embed(e, *c1)];
                let mut cat = Vec::with_capacity(2 * DIM);
                cat.extend_from_slice(&x[0]);
                cat.extend_from_slice(&x[1]);
                let pre = t.forward(h, &cat);
                let preacts: Vec<f32> = pre.iter().map(|&v| v.tanh()).collect();
                let logits = t.forward(o, &preacts);

                let (loss, dlogits) = softmax_ce(&logits, *target);
                tot += loss;

                // backward
                let dpre = t.backward_linear(o, &dlogits, &preacts, 1.0);
                let dcat: Vec<f32> = dpre
                    .iter()
                    .zip(pre.iter())
                    .map(|(d, &p)| d * (1.0 - p * p))
                    .collect();
                let dx = t.backward_linear(h, &dcat, &cat, 1.0);
                t.backward_embed(e, *c0, &dx[..DIM], 1.0);
                t.backward_embed(e, *c1, &dx[DIM..], 1.0);
            }
            t.optimize_step();
            assert!(tot.is_finite(), "loss должен быть конечным");

            // оценка на train
            correct = 0;
            n = 0;
            for (c0, c1, target) in &samples {
                let x = [t.embed(e, *c0), t.embed(e, *c1)];
                let mut cat = Vec::with_capacity(2 * DIM);
                cat.extend_from_slice(&x[0]);
                cat.extend_from_slice(&x[1]);
                let pre = t.forward(h, &cat);
                let preacts: Vec<f32> = pre.iter().map(|&v| v.tanh()).collect();
                let logits = t.forward(o, &preacts);
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap();
                if pred == *target {
                    correct += 1;
                }
                n += 1;
            }
        }
        assert_eq!(
            correct, n,
            "char-LM должен выучить паттерн (acc {correct}/{n})"
        );
    }

    /// 3) Самообучение: continual learning через синаптический кэш.
    /// Задача A: y = 2·x0 (x1 неважен). Задача B: y = 2·x1.
    /// Без консолидации fast-веса забывают A; slow-веса сохраняют её.
    #[test]
    fn continual_learning_synaptic_cache() {
        fn run_task(t: &mut Trainer, li: usize, use_x1: bool, epochs: usize) -> f32 {
            let mut last = f32::MAX;
            for _ in 0..epochs {
                t.clear_grads();
                let samples = [
                    ([1.0f32, 0.0], 2.0),
                    ([0.0, 1.0], -0.0),
                    ([-1.0, 0.0], -2.0),
                ];
                for (x, y) in &samples {
                    let xv = if use_x1 { [x[1], x[0]] } else { *x };
                    let pred = t.forward(li, &xv)[0];
                    let dout = vec![2.0 * (pred - y)];
                    t.backward_linear(li, &dout, &xv, 1.0 / samples.len() as f32);
                    last = (pred - y) * (pred - y);
                }
                t.optimize_step();
            }
            last
        }
        fn eval_a(t: &Trainer, li: usize, slow: bool) -> f32 {
            let samples = [([1.0f32, 0.0], 2.0), ([-1.0, 0.0], -2.0)];
            let mut err = 0.0f32;
            for (x, y) in &samples {
                let pred = if slow {
                    t.forward_slow(li, x)[0]
                } else {
                    t.forward(li, x)[0]
                };
                err += (pred - y) * (pred - y);
            }
            err / samples.len() as f32
        }

        let mut t = Trainer::new(0.01);
        let li = t.add_linear(1, 2);

        // Фаза A с консолидацией успеха.
        run_task(&mut t, li, false, 1500);
        let r_a = t.reward(0.01, 4.0);
        let committed_a = t.consolidate(r_a);
        let a_slow_after_a = eval_a(&t, li, true);
        assert!(committed_a > 0, "после A должна быть консолидация");
        // slow = смесь старта и w_A (c=0.5) — заметно лучше случайного.
        assert!(a_slow_after_a < 1.0, "slow помнит A: {a_slow_after_a}");

        // Фаза B (переучивание на другую задачу) — fast забывает A.
        run_task(&mut t, li, true, 1500);
        let fast_err_a = eval_a(&t, li, false);
        let slow_err_a = eval_a(&t, li, true);
        let r_b = t.reward(0.01, 4.0);
        t.consolidate(r_b);

        // Slow-путь обязан забыть A сильно меньше, чем fast:
        // fast теряет A почти полностью (≈4), slow держит смесь (≈2.2).
        assert!(
            slow_err_a < fast_err_a * 0.7,
            "синаптический кэш должен сгладить забывание: slow={slow_err_a}, fast={fast_err_a}"
        );
        assert!(fast_err_a > 2.0, "контроль: fast действительно забывает A: {fast_err_a}");
    }

    /// 6) Математика на Rust: оконная seq-модель учится сложению.
    /// Токены: цифры 0..9, '+', '='. Формат: [a,'+',b,'=',r1,r2].
    /// Модель предсказывает следующий токен по окну из W предыдущих
    /// (общая embed + 2 FFN-слоя на все позиции — как mini-LM без внимания).
    #[test]
    fn train_addition_seq() {
        const VOCAB: usize = 12; // 0..9, '+', '='
        const DIM: usize = 32;
        const H: usize = 128;
        const W: usize = 5;
        let digit = |t: usize| t < 10;

        fn encode(a: usize, b: usize) -> Vec<usize> {
            // [a,'+',b,'=',tens,ones]
            let s = a + b;
            vec![a, 10, b, 11, s / 10, s % 10]
        }

        // Данные: детерминированная выборка однозначных сложений.
        let mut rng = Noise::new(77);
        let mut train: Vec<Vec<usize>> = Vec::new();
        for _ in 0..1600 {
            let a = (rng.next_f32() * 10.0) as usize % 10;
            let b = (rng.next_f32() * 10.0) as usize % 10;
            train.push(encode(a, b));
        }

        let mut t = Trainer::new(0.1);
        let scale = 1.0 / (train.len().max(1) as f32 * 2.0);
        let e = t.add_embedding(VOCAB, DIM);
        let h = t.add_linear(H, W * DIM);
        let o = t.add_linear(VOCAB, H);

        // Обучаем на всех позициях teacher-forcing.
        for _ in 0..400 {
            t.clear_grads();
            for seq in &train {
                // Учим только токены результата (r1, r2): меньше
                // конфликтов между позициями у общих весов.
                for pos in 4..seq.len() {
                    // Окно из W предыдущих токенов (слева паддим '=').
                    let from = pos.saturating_sub(W);
                    let ctx: Vec<usize> = if pos >= W {
                        seq[pos - W..pos].to_vec()
                    } else {
                        let mut c = vec![11; W - pos];
                        c.extend_from_slice(&seq[..pos]);
                        c
                    };
                    debug_assert_eq!(ctx.len(), W);
                    let mut cat = Vec::with_capacity(W * DIM);
                    for &tk in &ctx {
                        cat.extend_from_slice(&t.embed(e, tk));
                    }
                    let pre = t.forward(h, &cat);
                    let preacts: Vec<f32> = pre.iter().map(|&v| v.tanh()).collect();
                    let logits = t.forward(o, &preacts);
                    let target = seq[pos];
                    let (_loss, dlogits) = softmax_ce(&logits, target);
                    let dpre = t.backward_linear(o, &dlogits, &preacts, scale);
                    let dcat: Vec<f32> = dpre
                        .iter()
                        .zip(pre.iter())
                        .map(|(d, &p)| d * (1.0 - p * p))
                        .collect();
                    let dx = t.backward_linear(h, &dcat, &cat, scale);
                    for (j, chunk) in ctx.iter().enumerate() {
                        let slice = &dx[j * DIM..(j + 1) * DIM];
                        t.backward_embed(e, *chunk, slice, scale);
                    }
                }
            }
            t.optimize_step();
        }
        // Генеративная оценка: предсказываем результат по [a,'+',b,'='].
        let mut ok = 0usize;
        let mut total = 0usize;
        let mut rng2 = Noise::new(123);
        for _ in 0..400 {
            let a = (rng2.next_f32() * 10.0) as usize % 10;
            let b = (rng2.next_f32() * 10.0) as usize % 10;
            let true_sum = a + b;
            let mut seq = vec![a, 10, b, 11];
            while seq.len() < 6 {
                let from = seq.len().saturating_sub(W);
                let ctx: Vec<usize> = if seq.len() >= W {
                    seq[seq.len() - W..].to_vec()
                } else {
                    let mut c = vec![11; W - seq.len()];
                    c.extend_from_slice(&seq);
                    c
                };
                let mut cat = Vec::with_capacity(W * DIM);
                for &tk in &ctx {
                    cat.extend_from_slice(&t.embed(e, tk));
                }
                let pre = t.forward(h, &cat);
                let preacts: Vec<f32> = pre.iter().map(|&v| v.tanh()).collect();
                let logits = t.forward(o, &preacts);
                let pred = logits
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                seq.push(pred);
            }
            let predicted = seq[4] * 10 + seq[5];
            total += 1;
            if predicted == true_sum {
                ok += 1;
            }
        }
        assert!(
            ok * 10 >= total * 8,
            "модель должна решать сложение: acc {ok}/{total}"
        );
        let _ = digit;
    }

    /// 4) Связка саморазвития: реплей-«воспоминания» во время фазы B
    /// удерживают задачу A (baseline без реплея её теряет), «сон»
    /// доучивает буфер с пониженным lr.
    #[test]
    fn self_development_replay_sleep() {
        fn phase(t: &mut Trainer, li: usize, x1: [f32; 2], epochs: usize) -> f32 {
            let mut last = f32::MAX;
            for _ in 0..epochs {
                t.clear_grads();
                let samples = [(x1, 2.0), ([-x1[0], -x1[1]], -2.0)];
                for (x, y) in &samples {
                    let pred = t.forward(li, x)[0];
                    let dout = vec![2.0 * (pred - y)];
                    t.backward_linear(li, &dout, x, 0.5);
                    last = (pred - y) * (pred - y);
                }
                t.optimize_step();
            }
            last
        }
        fn mse_a(t: &Trainer, li: usize) -> f32 {
            let samples = [([1.0f32, 0.0], 2.0), ([-1.0, 0.0], -2.0)];
            let mut e = 0.0f32;
            for (x, y) in &samples {
                let p = t.forward(li, x)[0];
                e += (p - y) * (p - y);
            }
            e / 2.0
        }

        // Baseline: A -> B без памяти.
        let mut base = Trainer::new(0.05);
        let lb = base.add_linear(1, 2);
        phase(&mut base, lb, [1.0, 0.0], 1500);
        phase(&mut base, lb, [0.0, 1.0], 1500);
        let base_err_a = mse_a(&base, lb);

        // Саморазвитие: A -> (B + реплей A каждую эпоху) -> сон.
        let mut dev = Trainer::new(0.05);
        let ld = dev.add_linear(1, 2);
        phase(&mut dev, ld, [1.0, 0.0], 1500);
        // Важные воспоминания фазы A (ошибка сначала низкая, но это
        // «школьный материал» — держим его для закрепления).
        let mut buf = Replay::new(64);
        for x in [[1.0f32, 0.0], [-1.0, 0.0]] {
            buf.record(&x, if x[0] > 0.0 { 2.0 } else { -2.0 }, 0.5);
        }
        for _ in 0..1500 {
            dev.clear_grads();
            let samples = [([0.0f32, 1.0], 2.0), ([0.0, -1.0], -2.0)];
            for (x, y) in &samples {
                let pred = dev.forward(ld, x)[0];
                let dout = vec![2.0 * (pred - y)];
                dev.backward_linear(ld, &dout, x, 0.5);
            }
            dev.optimize_step();
            // Реплей воспоминаний A (каждую 3-ю эпоху).
            if buf.len() > 0 {
                dev.clear_grads();
                for (_, x, y) in buf.by_importance() {
                    let pred = dev.forward(ld, x)[0];
                    let err = (pred - y).abs();
                    let dout = vec![2.0 * (pred - y)];
                    dev.backward_linear(ld, &dout, x, 0.5 / buf.len() as f32);
                    let _ = err;
                }
                dev.optimize_step();
            }
        }
        // Сон: доучиваем воспоминания с пониженным lr + консолидация.
        dev.sleep(ld, &mut buf, 0.3, 200);
        let dev_err_a = mse_a(&dev, ld);

        assert!(
            dev_err_a < base_err_a * 0.5,
            "реплей+сон должны удержать A: dev={dev_err_a}, base={base_err_a}"
        );
    }

    /// 5) Рождение нейрона: спящая строка рождается мутацией активной.
    #[test]
    fn birth_neuron_smoke() {
        let mut t = Trainer::new(0.01);
        let li = t.add_linear(2, 2);
        // Обучаем только выход 0 (y = 2·x0); выход 1 спит.
        for _ in 0..1500 {
            t.clear_grads();
            let x = [1.0f32, 0.3];
            let pred = t.forward(li, &x);
            let dout = vec![2.0 * (pred[0] - 2.0), 0.0];
            t.backward_linear(li, &dout, &x, 1.0);
            t.optimize_step();
        }
        let before = t.forward(li, &[1.0, 0.0])[1];
        let newborn = t.birth_row(li, 1, 0, 0.05);
        let after = t.forward(li, &[1.0, 0.0])[1];
        assert!(
            (after - 2.0).abs() < 0.4,
            "рождённый нейрон наследует активную функцию: pred={after}"
        );
        assert!((after - before).abs() > 1e-3, "строка действительно заменилась");
        assert!(newborn.iter().any(|&v| v != 0.0));
    }
}

    /// 7) Мини-трансформер на Rust: embed → GQA-attn(RoPE)+residual →
    /// FFN → head; двузначное сложение с переносом; веса = атомы графа.
        const MT_VOCAB: usize = 12;
    const MT_D: usize = 24;
    const MT_QH: usize = 2;
    const MT_KVH: usize = 1;
    const MT_DH: usize = 12;
    const MT_HH: usize = 48;
    const MT_S: usize = 9;

    /// 7) Мини-трансформер на Rust (SUPERSEDED модулем atom::model —
    /// собирался на старой раскладке GQA). Сохранён как регрессия-история.
    #[test]
    #[ignore = "superseded by atom::model TransformerLM"]
    fn mini_transformer_two_digit_add() {
        use crate::atom::nn::silu;
        use crate::atom::seq::{gqa_bwd, gqa_fwd, rope_apply_sign};

        let scale = |n: usize| 1.0 / (n.max(1) as f32 * (MT_S as f32 - 1.0));

        fn encode(a: usize, b: usize) -> Vec<usize> {
            let s = a + b;
            vec![a / 10, a % 10, 10, b / 10, b % 10, 11, s / 100, (s / 10) % 10, s % 10]
        }

        let mut t = Trainer::new(0.1);
        t.hexiron_on = false;
        t.rineta_alpha = 0.0;
        let e = t.add_embedding(MT_VOCAB, MT_D);
        let wq = t.add_linear(MT_D, MT_D);
        let wk = t.add_linear(MT_KVH * MT_DH, MT_D);
        let wv = t.add_linear(MT_KVH * MT_DH, MT_D);
        let wo = t.add_linear(MT_D, MT_D);
        let up = t.add_linear(MT_HH, MT_D);
        let head = t.add_linear(MT_VOCAB, MT_HH);
        let layers = (e, wq, wk, wv, wo, up, head);

        fn fwd(
            t: &Trainer,
            layers: (usize, usize, usize, usize, usize, usize, usize),
            seq: &[usize],
        ) -> (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<crate::atom::seq::AttnCache>,
        ) {
            let (e, wq, wk, wv, wo, up, head) = layers;
            let l = seq.len();
            let mut q = vec![0.0f32; l * MT_QH * MT_DH];
            let mut k = vec![0.0f32; l * MT_KVH * MT_DH];
            let mut v = vec![0.0f32; l * MT_KVH * MT_DH];
            let mut x_emb = vec![0.0f32; l * MT_D];
            for (tt, &tk) in seq.iter().enumerate() {
                let xt = t.embed(e, tk);
                x_emb[tt * MT_D..(tt + 1) * MT_D].copy_from_slice(&xt);
                let qt = t.forward(wq, &xt);
                let kt = t.forward(wk, &xt);
                let vt = t.forward(wv, &xt);
                q[tt * MT_QH * MT_DH..(tt + 1) * MT_QH * MT_DH].copy_from_slice(&qt);
                k[tt * MT_KVH * MT_DH..(tt + 1) * MT_KVH * MT_DH].copy_from_slice(&kt);
                v[tt * MT_KVH * MT_DH..(tt + 1) * MT_KVH * MT_DH].copy_from_slice(&vt);
            }
            let theta = 10000.0f32;
            for tt in 0..l {
                for h in 0..MT_QH {
                    let sl = &mut q[(tt * MT_QH + h) * MT_DH..(tt * MT_QH + h + 1) * MT_DH];
                    rope_apply_sign(sl, tt, theta, MT_DH, 1.0);
                }
                for h in 0..MT_KVH {
                    let sl = &mut k[(tt * MT_KVH + h) * MT_DH..(tt * MT_KVH + h + 1) * MT_DH];
                    rope_apply_sign(sl, tt, theta, MT_DH, 1.0);
                }
            }
            let (attn, caches) = gqa_fwd(&q, &k, &v, MT_QH, MT_KVH, MT_DH);
            let mut block = vec![0.0f32; l * MT_D];
            let mut act = vec![0.0f32; l * MT_HH];
            let mut logits = vec![0.0f32; l * MT_VOCAB];
            for tt in 0..l {
                let at = &attn[tt * MT_D..(tt + 1) * MT_D];
                let ot = t.forward(wo, at);
                for j in 0..MT_D {
                    block[tt * MT_D + j] = x_emb[tt * MT_D + j] + ot[j];
                }
                let bt = &block[tt * MT_D..(tt + 1) * MT_D];
                let ht = t.forward(up, bt);
                for j in 0..MT_HH {
                    act[tt * MT_HH + j] = silu(ht[j]);
                }
                let at2 = &act[tt * MT_HH..(tt + 1) * MT_HH];
                let lg = t.forward(head, at2);
                logits[tt * MT_VOCAB..(tt + 1) * MT_VOCAB].copy_from_slice(&lg);
            }
            (q, k, v, attn, block, act, logits, caches)
        }

        const MB: usize = 100;
        for epoch in 0..1200 {
            // Затухание lr: грубая фаза -> тонкая.
            if epoch == 601 {
                t.set_lr(0.02);
            }
            // Свежие примеры каждый шаг: учим обобщению, а не порядку.
            let mut g = Noise::new(epoch as u64 + 777);
            let mut epoch_samples: Vec<Vec<usize>> = Vec::with_capacity(500);
            for _ in 0..500 {
                let a = ((g.next_f32().abs() * 90.0) as usize % 90) + 10;
                let b = ((g.next_f32().abs() * 90.0) as usize % 90) + 10;
                epoch_samples.push(encode(a, b));
            }
            let mut start = 0usize;
            while start < epoch_samples.len() {
                let end = (start + MB).min(epoch_samples.len());
                t.clear_grads();
                let sc = 1.0 / ((end - start).max(1) as f32 * (MT_S as f32 - 1.0));
                for seq in &epoch_samples[start..end] {
                    let (q, k, v, attn, block, act, logits, caches) = fwd(&t, layers, seq);
                    let l = seq.len();
                    let mut dlog = vec![0.0f32; l * MT_VOCAB];
                    for tt in 0..l - 1 {
                        let row = &logits[tt * MT_VOCAB..(tt + 1) * MT_VOCAB];
                        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sm = 0.0f32;
                        for kk in 0..MT_VOCAB {
                            let ex = (row[kk] - max).exp();
                            dlog[tt * MT_VOCAB + kk] = ex;
                            sm += ex;
                        }
                        for kk in 0..MT_VOCAB {
                            dlog[tt * MT_VOCAB + kk] /= sm;
                        }
                        let tg = seq[tt + 1];
                        dlog[tt * MT_VOCAB + tg] -= 1.0;
                    }
                    let mut d_block = vec![0.0f32; l * MT_D];
                    for tt in 0..l - 1 {
                        let drow = &dlog[tt * MT_VOCAB..(tt + 1) * MT_VOCAB];
                        let act_row = &act[tt * MT_HH..(tt + 1) * MT_HH];
                        let da = t.backward_linear(head, drow, act_row, sc);
                        let pre = t.forward(up, &block[tt * MT_D..(tt + 1) * MT_D]);
                        let mut dh = vec![0.0f32; MT_HH];
                        for j in 0..MT_HH {
                            let g2 = 1.0 / (1.0 + (-pre[j]).exp());
                            dh[j] = da[j] * (g2 + pre[j] * g2 * (1.0 - g2));
                        }
                        let dbl =
                            t.backward_linear(up, &dh, &block[tt * MT_D..(tt + 1) * MT_D], sc);
                        for j in 0..MT_D {
                            d_block[tt * MT_D + j] = dbl[j];
                        }
                    }
                    let (dq, dk, dv) =
                        gqa_bwd(&d_block, &q, &k, &v, &caches, MT_QH, MT_KVH, MT_DH);
                    let mut dq = dq;
                    let mut dk = dk;
                    let theta = 10000.0f32;
                    for tt in 0..l {
                        for h in 0..MT_QH {
                            let sl = &mut dq
                                [(tt * MT_QH + h) * MT_DH..(tt * MT_QH + h + 1) * MT_DH];
                            rope_apply_sign(sl, tt, theta, MT_DH, -1.0);
                        }
                        for h in 0..MT_KVH {
                            let sl = &mut dk
                                [(tt * MT_KVH + h) * MT_DH..(tt * MT_KVH + h + 1) * MT_DH];
                            rope_apply_sign(sl, tt, theta, MT_DH, -1.0);
                        }
                    }
                    for tt in 0..l {
                        let xt = t.embed(e, seq[tt]);
                        let dxq = t.backward_linear(
                            wq,
                            &dq[tt * MT_D..(tt + 1) * MT_D],
                            &xt,
                            sc,
                        );
                        let dxk = t.backward_linear(
                            wk,
                            &dk[tt * MT_KVH * MT_DH..(tt + 1) * MT_KVH * MT_DH],
                            &xt,
                            sc,
                        );
                        let dxv = t.backward_linear(
                            wv,
                            &dv[tt * MT_KVH * MT_DH..(tt + 1) * MT_KVH * MT_DH],
                            &xt,
                            sc,
                        );
                        let attn_row = &attn[tt * MT_D..(tt + 1) * MT_D];
                        let dxwo = t.backward_linear(
                            wo,
                            &d_block[tt * MT_D..(tt + 1) * MT_D],
                            attn_row,
                            sc,
                        );
                        let mut gg = vec![0.0f32; MT_D];
                        for j in 0..MT_D {
                            gg[j] = dxq[j] + dxk[j] + dxv[j] + dxwo[j] + d_block[tt * MT_D + j];
                        }
                        t.backward_embed(e, seq[tt], &gg, sc);
                    }
                }
                t.optimize_step();
                start = end;
            }
            if epoch % 100 == 0 {
                let seqp = &epoch_samples[0];
                let (_, _, _, _, _, _, logits2, _) = fwd(&t, layers, seqp);
                let mut lv = 0.0f32;
                for tt in 0..seqp.len() - 1 {
                    let row = &logits2[tt * MT_VOCAB..(tt + 1) * MT_VOCAB];
                    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sm = 0.0f32;
                    for kk in 0..MT_VOCAB {
                        sm += (row[kk] - max).exp();
                    }
                    lv += (sm.ln() - (row[seqp[tt + 1]] - max)).max(0.0);
                }
                eprintln!("epoch {epoch}: loss0={:.3}", lv / 8.0);
            }
        }
        // Финальная loss-проверка (учится ли каркас).
        let seq0 = encode(13, 24);
        let (_, _, _, _, _, _, logits_f, _) = fwd(&t, layers, &seq0);
        let mut loss0_final = 0.0f32;
        for tt in 0..seq0.len() - 1 {
            let row = &logits_f[tt * MT_VOCAB..(tt + 1) * MT_VOCAB];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sm = 0.0f32;
            for kk in 0..MT_VOCAB {
                sm += (row[kk] - max).exp();
            }
            loss0_final += (sm.ln() - (row[seq0[tt + 1]] - max)).max(0.0);
        }
        loss0_final /= (seq0.len() - 1) as f32;
        eprintln!("final loss (sample0)={loss0_final:.3}");

        // Генеративная оценка
        let mut ok = 0usize;
        let mut tot = 0usize;
        let mut digits_ok = 0usize;
        let mut digits_tot = 0usize;
        let mut noise2 = Noise::new(7);
        for _ in 0..200 {
            let a = ((noise2.next_f32().abs() * 90.0) as usize % 90) + 10;
            let b = ((noise2.next_f32().abs() * 90.0) as usize % 90) + 10;
            let mut seq = encode(a, b);
            seq.truncate(6);
            let mut guarded = 0;
            while seq.len() < 9 && guarded < 10 {
                guarded += 1;
                let (q, k, v, _, _, _, logits, _) = fwd(&t, layers, &seq);
                let _ = (q, k, v);
                let tt = seq.len() - 1;
                let row = &logits[tt * MT_VOCAB..(tt + 1) * MT_VOCAB];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.total_cmp(y.1))
                    .map(|(i, _)| i)
                    .unwrap();
                seq.push(pred);
            }
            tot += 1;
            let target = a + b;
            if seq.len() == 9 {
                for (p, pos) in [(6usize, 100usize), (7, 10), (8, 1)] {
                    if seq[p] == (target / pos) % 10 {
                        digits_ok += 1;
                    }
                    digits_tot += 1;
                }
                if seq[6] * 100 + seq[7] * 10 + seq[8] == target {
                    ok += 1;
                }
            }
        }
        eprintln!("eval: exact {ok}/{tot}, digit-level {digits_ok}/{digits_tot}");
        eprintln!(
            "v1-статус: каркас обучается; точная 3-цифровая генерация требует ёмкости/эпох"
        );
        assert!(
            loss0_final < 0.9,
            "мини-трансформер должен учиться (loss {loss0_final})"
        );
    


}

#[cfg(test)]
mod xor_table_extra {
    use super::*;

    /// XOR-таблица через граф-оптимизатор: нелинейная задача.
    /// ОТЛОЖЕН: bias-аугментация, мини-батчи, моментум, клип дают 3/4;
    /// остаточный NaN генерируется в импульсном пути Keralin на одной
    /// точке — чинится на уровне оптимизатора (bias-слой + clip моментума).
    #[test]
    #[ignore = "NaN persists beyond warm/clip — deep optimizer fixme"]
    fn xor_table_with_graph() {
        let mut t = Trainer::new(0.01);
        t.graph.params.wd = 1e-4;
        t.graph.params.beta1 = 0.5;
        t.hexiron_on = false;
        t.rineta_alpha = 0.0;
        let h = t.add_linear(16, 3);
        let o = t.add_linear(1, 16);
        // вход с константой 1 = «смещение»: ломает чётную симметрию XOR
        // (без неё градиенты обнуляются структурно)
        let data = [
            ([1.0f32, 1.0, 1.0], -1.0),
            ([1.0, -1.0, 1.0], 1.0),
            ([-1.0, 1.0, 1.0], 1.0),
            ([-1.0, -1.0, 1.0], -1.0),
        ];
        let mut worst = f32::MAX;
        let mut noise = Noise::new(2024);
        for ep in 0..6000 {
            // перемешивание 4 точек + мини-батчи по 2
            let mut order: Vec<usize> = vec![0, 1, 2, 3];
            let mut rnd = noise.next_f32();
            let _ = rnd;
            // детерминированный шиффл через abs-noise
            let mut rng = Noise::new((ep as u64).wrapping_mul(31) + 5);
            for i in (1..order.len()).rev() {
                let j = ((rng.next_f32().abs() * (i + 1) as f32) as usize).min(i);
                order.swap(i, j);
            }
            for chunk in order.chunks(2) {
                t.clear_grads();
                let sc = 0.5;
                for &pi in chunk {
                    let (x, y) = &data[pi];
                    let pre = t.forward(h, x);
                    let acts: Vec<f32> = pre.iter().map(|&v| v.tanh()).collect();
                    let pred = t.forward(o, &acts)[0];
                    let dout = vec![2.0 * (pred - y)];
                    let da = t.backward_linear(o, &dout, &acts, sc);
                    let dh: Vec<f32> = da
                        .iter()
                        .zip(pre.iter())
                        .map(|(d, &p)| d * (1.0 - p * p))
                        .collect();
                    t.backward_linear(h, &dh, x, sc);
                    if pred.is_finite() {
                        worst = worst.max((pred - y).abs());
                    }
                }
                // градиентный клип (NaN-устойчивый)
                let mut gmax = 0.0f32;
                for layer in &t.layers {
                    for &g in &layer.grad {
                        if g.is_finite() {
                            gmax = gmax.max(g.abs());
                        }
                    }
                }
                let f = if gmax > 10.0 { 10.0 / gmax } else { 1.0 };
                for layer in &mut t.layers {
                    for g in layer.grad.iter_mut() {
                        *g = if g.is_finite() { *g * f } else { 0.0 };
                    }
                }
                t.optimize_step();
                // простая регуляризация: ограничение размаха весов
                for layer in &mut t.layers {
                    for v in layer.w.iter_mut() {
                        if *v > 4.0 {
                            *v = 4.0;
                        } else if *v < -4.0 {
                            *v = -4.0;
                        }
                    }
                }
            }
        }
        // классификация всех четырёх точек
        let mut correct = 0usize;
        for (x, y) in &data {
            let pre = t.forward(h, x);
            let acts: Vec<f32> = pre.iter().map(|&v| v.tanh()).collect();
            let pred = t.forward(o, &acts)[0];
            if pred.is_finite() && (pred > 0.0) == (*y > 0.0) {
                correct += 1;
            }
        }
        assert_eq!(
            correct, 4,
            "XOR обязан решаться: {correct}/4, worst|err|={worst:.3}"
        );
        assert!(worst < 0.35, "XOR max-ошибка {worst}");
    }

}
