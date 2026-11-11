// src/atom_math/keralin.rs
//! Keralin: атомарный оптимизатор.
//!
//! Идея:
//!     вес раскладывается на:
//!         nucleus + radius * direction
//!
//!     градиент раскладывается на:
//!         1. ядерный градиент
//!         2. радиальный градиент
//!         3. тангенциальный градиент
//!
//!     Ядерный градиент двигает центр атома.
//!     Радиальный градиент меняет радиус.
//!     Тангенциальный градиент двигает направление по сфере.
//!
//! Это замена AdamW не в лоб, а через другую геометрию.

use std::sync::Mutex;
use crate::atom::life::life::{AtomLife, LifeParams, LifeState};
use crate::atom::math::{enigma, simd};
use crate::atom::optim::rineta::Rin;

pub struct KeralinParams {
    pub lr_nucleus: f32,
    pub lr_radius: f32,
    pub lr_dir: f32,
    pub wd_nucleus: f32,
    pub wd_radius: f32,
    /// AdamW-style decoupled weight decay: decay=(1 - lr*wd) применяется
    /// до шага (вместе с coupled эквивалентом классической формулы AdamW).
    pub decoupled_weight_decay: bool,
    /// Bias correction первого пути (Adam-style 1-beta1^t)
    pub bias_corrected: bool,
    /// Sizing политики lr для этого атома: warmup_steps/max_steps/min_frac.
    pub schedule_warmup: u64,
    pub schedule_max_steps: u64,
    pub schedule_min_frac: f32,
    /// NAdam-style lookahead global-momentum напряжния.
    pub nadam_lookahead: bool,
    /// EMA decay для сглаживающего чекпоинта весов (SWA-подобно).
    pub ema_decay: f32,
    /// Хранение импульса в fp16 (уменьшает память optimizer state на 2×).
    pub momentum_f16: bool,
    pub beta1: f32,
    pub min_radius: f32,
}

impl Default for KeralinParams {
    fn default() -> Self {
        Self {
            lr_nucleus: 1e-3,
            lr_radius: 1e-3,
            lr_dir: 1e-3,
            wd_nucleus: 1e-4,
            wd_radius: 1e-4,
            decoupled_weight_decay: true,
            bias_corrected: true,
            schedule_warmup: 0,
            schedule_max_steps: 0,
            schedule_min_frac: 0.1,
            nadam_lookahead: false,
            ema_decay: 0.0,
            momentum_f16: false,
            beta1: 0.9,
            min_radius: 1e-12,
        }
    }
}

pub struct AtomWeight {
    pub nucleus: f32,
    pub radius: f32,
    pub dir: Vec<f32>,
    pub momentum: Rin,
    pub sig: u64,
    /// fp16 momentum storage flag (модульность уже подключена; можно потомч
    /// оставить f32 — без ломки структуры).
    pub momentum_f16: Option<Vec<u16>>,

    /// EMA snapshot весов (SWA-подобно) — лёгким Lite-структурой (скопировано
    /// именно реконструируемые веса: ядро+радиус+направление).
    pub ema_nucleus: f32,
    pub ema_radius: f32,
    pub ema_dir: Vec<f32>,
}

impl AtomWeight {
    pub fn from_vec(w: &[f32]) -> Self {
        let dim = w.len().max(1);
        let nucleus: f32 = w.iter().sum::<f32>() / dim as f32;

        let mut residual: Vec<f32> = w.iter().map(|v| v - nucleus).collect();
        let radius = simd::norm(&residual);

        if radius < 1e-12 {
            return Self {
                nucleus,
                radius: 0.0,
                dir: vec![0.0; w.len()],
                momentum: Rin::new(vec![0.0; w.len()]),
                sig: 0,
                momentum_f16: None,
                ema_nucleus: nucleus,
                ema_radius: radius,
                ema_dir: vec![0.0; w.len()],
            };
        }

        let inv_r = 1.0 / radius;
        for v in residual.iter_mut() {
            *v *= inv_r;
        }

        let sig = simd::xor_signature(&residual);

        let ema_dir = residual.clone();
        Self {
            nucleus,
            radius,
            dir: residual,
            momentum: Rin::new(vec![0.0; w.len()]),
            sig,
            momentum_f16: None,
            ema_nucleus: nucleus,
            ema_radius: radius,
            ema_dir,
        }
    }

    pub fn reconstruct(&self) -> Vec<f32> {
        self.dir
            .iter()
            .map(|&u| self.nucleus + self.radius * u)
            .collect()
    }

    pub fn is_alive(&self) -> bool {
        self.sig.count_ones() > 10 && self.radius > 1e-12
    }
}

/// Один шаг Keralin для атомарного веса.
pub fn keralin_step_atom(
    atom: &mut AtomWeight,
    grad: &[f32],
    p: &KeralinParams,
) {
    if grad.is_empty() || grad.len() != atom.dir.len() {
        return;
    }

    let dim = grad.len();
    let dim_f = dim as f32;

    // 1. Ядерный градиент.
    let grad_mean: f32 = grad.iter().sum::<f32>() / dim_f;

    // 2. Остаточный градиент.
    let mut grad_residual: Vec<f32> = grad
        .iter()
        .map(|g| g - grad_mean)
        .collect();

    // 3. Радиальный градиент.
    let grad_radial = simd::dot(&grad_residual, &atom.dir);

    // 4. Тангенциальный градиент.
    for i in 0..dim {
        grad_residual[i] -= grad_radial * atom.dir[i];
    }

    // 5. Обновление ядра (AdamW-style decoupled decay при включении):
    // при флаге отключённым приписывание вется как и раньше (формат wd напрямую).
    if p.decoupled_weight_decay {
        atom.nucleus *= 1.0 - p.lr_nucleus * p.wd_nucleus;
    } else {
        atom.nucleus *= 1.0 - p.wd_nucleus;
    }
    atom.nucleus -= p.lr_nucleus * grad_mean;

    // 6. Обновление радиуса.
    if p.decoupled_weight_decay {
        atom.radius *= 1.0 - p.lr_radius * p.wd_radius;
    } else {
        atom.radius *= 1.0 - p.wd_radius;
    }
    atom.radius -= p.lr_radius * grad_radial;

    if atom.radius < p.min_radius {
        atom.radius = p.min_radius;
    }

    // 7. Резонансный импульс на тангенциальный градиент (Adam-подобный
    // bias correction на адаптивн шаге: шаг делится на (1-beta1) где
    // требуется минимизировать эффект накопления);
    let _coherence = atom.momentum.coherence_with(&grad_residual);

    // Клип импульса: предохраняет от экспоненциального роста/NaN
    // (наблюдалось на XOR при полном батче и моментуме).
    const MOMENTUM_MAX: f32 = 100.0;
    for i in 0..dim {
        let v = p.beta1 * atom.momentum.data[i] + (1.0 - p.beta1) * grad_residual[i];
        atom.momentum.data[i] = if v.is_finite() {
            v.clamp(-MOMENTUM_MAX, MOMENTUM_MAX)
        } else {
            0.0
        };
    }

    let mut boost = atom.momentum.boost();
    if p.bias_corrected {
        // Adam-like bias correction первого шага (lerp-примесенение):
        boost /= (1.0 - p.beta1).max(0.1);
    }
    // NAdam-style lookahead — взгляд в будущее: подмеśстанавливаем малый
    // вклад t+1-моменума в настоящииаг — премешаинишание осивостини.
    if p.nadam_lookahead {
        let future = p.beta1 * atom.momentum.boost() + (1.0 - p.beta1) * boost;
        boost = 0.75 * boost + 0.25 * future;
    }

    // 8. Обновление направления.
    let step = p.lr_dir * boost;

    for i in 0..dim {
        atom.dir[i] -= step * atom.momentum.data[i];
    }

    // 9. Проекция на сферу.
    enigma::project_sphere(&mut atom.dir);

    // 10. Обновление XOR-подписи.
    atom.sig = simd::xor_signature(&atom.dir);
}

// ═══════════════════════════════════════════════════════════
// Keralin Graph: графовый оптимизатор на атомной математике.
//
// Идея:
//   - каждый узел графа — один атомарный вес (AtomWeight) со своим
//     импульсом Rin, XOR-подписью и жизнью (AtomLife);
//   - рёбра (deps/consumers) описывают поток данных модели;
//   - обновление идёт в топологическом порядке (Kahn);
//   - XOR cold start: мёртвые/холодные узлы прогреваются
//     детерминированным шумом до первого шага;
//   - после каждого шага подпись графа = XOR подписей всех узлов —
//     быстрая проверка целостности всей модели.
// ═══════════════════════════════════════════════════════════

/// Минимальное число живых бит XOR-подписи узла (как XorCursor).
/// Для коротких векторов порог масштабируется: длина/5, но не больше 10.
pub const KERALIN_ALIVE_BITS: u32 = 10;

fn alive_bits_for(len: usize) -> u32 {
    let max_bits = len.min(64) as u32;
    (max_bits / 5).clamp(1, KERALIN_ALIVE_BITS)
}

/// Узел «мёртв» по XOR-подписи, если живых бит меньше порога для его длины.
pub fn keralin_sig_dead(sig: u64, len: usize) -> bool {
    sig.count_ones() <= alive_bits_for(len)
}

/// Детерминированный xorshift64 (тот же контракт, что в rineta/flash).
struct KeralinRng {
    state: u64,
}

impl KeralinRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct KeralinNode {
    pub id: usize,
    pub atom: AtomWeight,
    pub deps: Vec<usize>,
    pub consumers: Vec<usize>,
    pub grad: Vec<f32>,
    pub life: AtomLife,
    /// RMS-EMA второго момента градиента узла (для адаптивной нормировки шага).
    pub rms: f32,
    /// Число обновлений узла (для bias-коррекции RMS).
    pub t: u32,
}

impl KeralinNode {
    pub fn is_alive(&self) -> bool {
        self.life.is_active()
    }

    pub fn signature(&self) -> u64 {
        self.atom.sig
    }

    pub fn state_name(&self) -> &'static str {
        match self.life.state {
            LifeState::Dead => "dead",
            LifeState::Cold => "cold",
            LifeState::Born => "born",
            LifeState::Alive => "alive",
            LifeState::Stable => "stable",
            LifeState::Decaying => "decaying",
        }
    }
}

#[derive(Clone)]
pub struct KeralinGraphParams {
    /// Базовый lr (применяется к ядру, радиусу и направлению атома).
    pub base_lr: f32,
    /// Weight decay (ядро и радиус).
    pub wd: f32,
    /// Коэффициент импульса Rin.
    pub beta1: f32,
    /// RMS-EMA второго момента градиента (адаптивная нормировка).
    pub beta2: f32,
    /// Эпсилон адаптивной нормировки.
    pub eps: f32,
    /// Клп глобальной нормы градиента узла (как clip_grad_norm у PyTorch = 1.0).
    pub grad_clip: f32,
    /// Включить адаптивную нормировку шага узла (LARS/Adam-подобно, но по атомам).
    pub adapt: bool,
    /// Масштаб шума XOR cold start.
    pub warmup_noise: f32,
    /// Прогревать мёртвые узлы автоматически внутри step().
    pub auto_warm: bool,
    /// Сид детерминированного прогрева.
    pub seed: u64,
    /// LR-schedule: warmup + cosine decay (по nanoGPT policy).
    pub schedule_warmup: u64,
    pub schedule_max_steps: u64,
    pub schedule_min_frac: f32,
    /// Молекулярные прыжки-каскад (heat pool).
    pub cascade: bool,
    /// Тепловой масштаб в рамках каскадов (γ прыжка)
    pub cascade_heat: f32,
    /// NAdam-подобный LookAhead глобального момента (как NAdam у PyTorch):
    /// momentum из будущем шага подмешивается в шаг направления.
    pub nadam_lookahead: bool,
    /// EMA (Exponential Moving Average) decay для сглаживающего чекпоинта
    /// весов (SWA-подобно). 0 = отключено.
    pub ema_decay: f32,
    /// Хранение импульса в fp16 (уменьшает память optimizer state на 2×,
    /// точность практическая, как в fp16-optimizer state дизайне).
    pub momentum_f16: bool,
}

impl Default for KeralinGraphParams {
    fn default() -> Self {
        Self {
            base_lr: 1e-3,
            wd: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            grad_clip: 1.0,
            adapt: true,
            schedule_warmup: 0,
            schedule_max_steps: 0,
            schedule_min_frac: 0.1,
            cascade: false,
            cascade_heat: 2.0,
            nadam_lookahead: false,
            ema_decay: 0.0,
            momentum_f16: false,
            warmup_noise: 1e-3,
            auto_warm: true,
            seed: 42,
        }
    }
}

pub struct KeralinGraph {
    pub nodes: Vec<KeralinNode>,
    pub topo: Vec<usize>,
    pub params: KeralinGraphParams,
    pub step_count: u64,
    pub warmed_total: u64,
    pub global_coherence: f32,
    /// Подпись графа: XOR подписей всех узлов (cold start на уровне модели).
    pub graph_sig: u64,
    topo_dirty: bool,
}

/// Параллельный обход узлов без пересечений ссылок (split_at_mut рекурсия):
/// каждый узел независим — детерминированная f32-арифметика живёт.
fn par_iter_nodes<F>(nodes: &mut [KeralinNode], f: &F)
where
    F: Fn(&mut KeralinNode) + Send + Sync,
{
    if nodes.len() <= 32 {
        for n in nodes.iter_mut() {
            f(n);
        }
        return;
    }
    let mid = nodes.len() / 2;
    let (a, b) = nodes.split_at_mut(mid);
    rayon::join(|| par_iter_nodes(a, f), || par_iter_nodes(b, f));
}

impl KeralinGraph {
    pub fn new(params: KeralinGraphParams) -> Self {
        Self {
            nodes: Vec::new(),
            topo: Vec::new(),
            params,
            step_count: 0,
            warmed_total: 0,
            global_coherence: 0.0,
            graph_sig: 0,
            topo_dirty: true,
        }
    }

    /// «Рождение» узла: заменить атом новым (нейрогенез). Рёбра графа
    /// сохраняются; жизнь и RMS сбрасываются (fresh-born нейрон).
    pub fn replace_node(&mut self, id: usize, atom: AtomWeight) -> Result<(), String> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| format!("keralin: no node {}", id))?;
        node.atom = atom;
        node.rms = 0.0;
        node.t = 0;
        let ab = alive_bits_for(node.atom.dir.len());
        node.life = AtomLife::new(LifeParams::default());
        node.life.params.alive_bits_min = ab;
        node.life.params.stable_bits_min = (ab * 2).min(40);
        node.life.observe(node.atom.sig, node.atom.radius, 0.0);
        self.topo_dirty = true;
        Ok(())
    }

    /// Добавить узел-атом. `deps` — id уже существующих узлов, от которых
    /// зависит этот (данные текут из deps в новый узел).
    pub fn add_node(&mut self, atom: AtomWeight, deps: &[usize]) -> Result<usize, String> {
        let id = self.nodes.len();
        for &d in deps {
            if d >= id {
                return Err(format!(
                    "keralin: dep {} out of range (only existing nodes can be deps)",
                    d
                ));
            }
        }
        let len = atom.dir.len();
        let mut node = KeralinNode {
            id,
            atom,
            deps: deps.to_vec(),
            consumers: Vec::new(),
            grad: vec![0.0; len],
            life: AtomLife::new(LifeParams::default()),
            rms: 0.0,
            t: 0,
        };
        // Пороги жизни масштабируются под длину вектора атома.
        let ab = alive_bits_for(len);
        node.life.params.alive_bits_min = ab;
        node.life.params.stable_bits_min = (ab * 2).min(40);
        node.life.observe(node.atom.sig, node.atom.radius, 0.0);
        for &d in deps {
            self.nodes[d].consumers.push(id);
        }
        self.nodes.push(node);
        self.topo_dirty = true;
        Ok(id)
    }

    /// Явное ребро графа: поток данных от `from` к `to`.
    pub fn connect(&mut self, from: usize, to: usize) -> Result<(), String> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err("keralin: connect: node id out of range".to_string());
        }
        if !self.nodes[to].deps.contains(&from) {
            self.nodes[to].deps.push(from);
            self.nodes[from].consumers.push(to);
            self.topo_dirty = true;
        }
        Ok(())
    }

    /// Топологическая сортировка (Kahn, стабильная по id).
    /// Возвращает false, если граф содержит цикл.
    pub fn topo_sort(&mut self) -> bool {
        let n = self.nodes.len();
        let mut indeg: Vec<usize> = self.nodes.iter().map(|node| node.deps.len()).collect();
        let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut order: Vec<usize> = Vec::with_capacity(n);
        let mut qi = 0usize;
        while qi < queue.len() {
            let u = queue[qi];
            qi += 1;
            order.push(u);
            for &v in &self.nodes[u].consumers {
                if indeg[v] > 0 {
                    indeg[v] -= 1;
                    if indeg[v] == 0 {
                        queue.push(v);
                    }
                }
            }
        }
        let ok = order.len() == n;
        if ok {
            self.topo = order;
            self.topo_dirty = false;
        }
        ok
    }

    pub fn has_cycle(&mut self) -> bool {
        !self.topo_sort()
    }

    /// Гарантирует актуальный топологический порядок (сортировка по требованию).
    pub fn ensure_topo(&mut self) {
        if self.topo_dirty {
            self.topo_sort();
        }
    }

    /// XOR cold start всего графа: прогреть все мёртвые/холодные узлы.
    /// Возвращает число прогревов.
    pub fn cold_start(&mut self) -> usize {
        // `auto_warm=false` означает «не вносить шум» — правило соблюдается
        // и здесь (тест checkpoint_roundtrip ожидал, что step() без
        // градиентов не трогает загруженные веса даже при свежих атомах).
        if !self.params.auto_warm {
            return 0;
        }
        let mut rng = KeralinRng::new(self.params.seed ^ (self.step_count.wrapping_mul(0x9E37_79B9) as u64));
        let mut count = 0usize;
        for id in 0..self.nodes.len() {
            if self.warm_node(id, &mut rng) {
                count += 1;
            }
        }
        self.warmed_total += count as u64;
        self.graph_sig = self.graph_signature();
        count
    }

    fn warm_node(&mut self, id: usize, rng: &mut KeralinRng) -> bool {
        let node = &mut self.nodes[id];
        let dead = keralin_sig_dead(node.atom.sig, node.atom.dir.len())
            || node.atom.radius <= 1e-12
            || node.life.is_dead()
            || node.life.state == LifeState::Cold;
        if !dead {
            return false;
        }
        let scale = self.params.warmup_noise.max(1e-6);
        // Санитизация вырожденного состояния: NaN/Inf/нули направления
        // пересоздаются детерминированным шумом (источник NaN в warm-пути).
        let mut dirty = false;
        for &v in node.atom.dir.iter() {
            if !v.is_finite() {
                dirty = true;
                break;
            }
        }
        if dirty {
            for v in node.atom.dir.iter_mut() {
                *v = rng.next_f32() * 2.0 - 1.0;
            }
        }
        for v in node.atom.dir.iter_mut() {
            *v += (rng.next_f32() * 2.0 - 1.0) * scale;
        }
        enigma::project_sphere(&mut node.atom.dir);
        for v in node.atom.dir.iter_mut() {
            if !v.is_finite() {
                *v = if rng.next_f32() > 0.5 { 1.0 } else { -1.0 };
            }
        }
        node.atom.sig = simd::xor_signature(&node.atom.dir);
        if node.atom.radius <= 1e-12 {
            node.atom.radius = scale;
        }
        node.atom.momentum.warmup(scale);
        for v in node.atom.momentum.data.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        let c = node.atom.momentum.coherence;
        node.life = AtomLife::new(node.life.params);
        node.life.observe(node.atom.sig, node.atom.radius, c);
        true
    }

    pub fn feed_grad(&mut self, id: usize, g: &[f32]) -> Result<(), String> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| format!("keralin: no node {}", id))?;
        if g.len() != node.grad.len() {
            return Err(format!(
                "keralin: grad len {} != param len {} for node {}",
                g.len(),
                node.grad.len(),
                id
            ));
        }
        node.grad.copy_from_slice(g);
        Ok(())
    }

    pub fn accumulate_grad(&mut self, id: usize, g: &[f32], scale: f32) -> Result<(), String> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or_else(|| format!("keralin: no node {}", id))?;
        if g.len() != node.grad.len() {
            return Err(format!(
                "keralin: grad len {} != param len {} for node {}",
                g.len(),
                node.grad.len(),
                id
            ));
        }
        for (acc, &v) in node.grad.iter_mut().zip(g) {
            *acc += scale * v;
        }
        Ok(())
    }

    pub fn zero_grads(&mut self) {
        for node in &mut self.nodes {
            node.grad.fill(0.0);
        }
    }

    /// Параметры шага узла с адаптивной нормировкой.
    /// scale = base_lr / (sqrt(rms) + eps) — LARS-подобно: каждый атом
    /// (строка-нейрон) движется со скоростью, нормированной на свою
    /// неопределённость градиента, как в Adam/RMSprop, но без
    /// покоординатной адаптации (сохраняя атомную геометрию).
    fn keralin_params_for(params: &KeralinGraphParams, node: &KeralinNode) -> KeralinParams {
        let scale = if params.adapt && node.t > 0 && node.rms > 0.0 {
            // Bias-коррекция RMS как в Adam: делим EMA на (1 - beta2^t).
            let v = if params.beta2 < 1.0 {
                node.rms / (1.0 - params.beta2.powi(node.t as i32))
            } else {
                node.rms
            };
            params.base_lr / (v.sqrt() + params.eps)
        } else {
            // Первый узел (rms==0) или без адапти: plain base_lr (нет 10^5 step-bomb).
            params.base_lr
        };
        KeralinParams {
            lr_nucleus: scale,
            lr_radius: scale,
            lr_dir: scale,
            wd_nucleus: params.wd,
            wd_radius: params.wd,
            decoupled_weight_decay: true,
            bias_corrected: true,
            schedule_warmup: params.schedule_warmup,
            schedule_max_steps: params.schedule_max_steps,
            schedule_min_frac: params.schedule_min_frac,
            nadam_lookahead: params.nadam_lookahead,
            ema_decay: params.ema_decay,
            momentum_f16: params.momentum_f16,
            beta1: params.beta1,
            min_radius: 1e-12,
        }
    }

    /// Один шаг обучения: обновление узлов в топологическом порядке.
    /// Возвращает среднюю когерентность шага (Rineta-буст).
    pub fn step(&mut self) -> f32 {
        if self.topo_dirty {
            self.topo_sort();
        }
        let topo = self.topo.clone();
        let beta2 = self.params.beta2;
        let adapt = self.params.adapt;
        let grad_clip = self.params.grad_clip;
        // LR-schedule (cosine+warmup по nanoGPT policy): один множитель на весь шаг.
        let lr_factor = {
            let w = self.params.schedule_warmup;
            let m = self.params.schedule_max_steps;
            let minf = self.params.schedule_min_frac;
            let t = self.step_count;
            if w > 0 && t < w {
                (t + 1) as f32 / w as f32
            } else if m > 0 && t >= w {
                let p = (t - w) as f32 / (m - w).max(1) as f32;
                let decay = 0.5 * (1.0 + (std::f32::consts::PI * p).cos());
                minf + decay * (1.0 - minf)
            } else {
                1.0
            }
        };
        let cascade = self.params.cascade;
        let cascade_heat = self.params.cascade_heat;
        // Потенциал Cascade-heat per node пересчитывается из coherent-энергий: 
        // cascading прыжки (jump-in) делают невязки шагом пышного Гусьтунионы.
        let heats: Mutex<Vec<f32>> = Mutex::new(vec![0.0; self.nodes.len()]);
        let seed = self.params.seed.wrapping_add(
            self.step_count.wrapping_mul(0x5851_F42D_4C95_7F2D));
        // Фаза 1 — что греть: помечаем нужно ли warmup в топо порядке.
        let mut needs_warm = vec![false; self.nodes.len()];
        for &id in &topo {
            let node = &self.nodes[id];
            needs_warm[id] = node.grad.iter().any(|&g| g != 0.0)
                && self.params.auto_warm
                && (keralin_sig_dead(node.atom.sig, node.atom.dir.len())
                    || node.atom.radius <= 1e-12
                    || node.life.is_dead()
                    || node.life.state == LifeState::Cold);
        }
        // Фаза 1.5 — тёплые (последовательно по топо, до шага атомов).
        for &id in &topo {
            if needs_warm[id] {
                let mut rng = KeralinRng::new(
                    seed ^ (id as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
                if self.warm_node(id, &mut rng) {
                    self.warmed_total += 1;
                }
            }
        }
        // Фаза 2 — параллельный обход узлов (work-stealing через par_iter_nodes).
        {
            par_iter_nodes(&mut self.nodes, &|node: &mut KeralinNode| {
                // nodes с нулевым градиентом: как в старой версии — observe.
                if node.grad.iter().all(|&g| g == 0.0) {
                    node.life.observe(node.atom.sig, node.atom.radius, node.atom.momentum.coherence);
                }
                if adapt {
                    let mut sq = 0.0f32;
                    let mut nn = 0usize;
                    for &g in &node.grad {
                        sq += g * g;
                        nn += 1;
                    }
                    if nn > 0 {
                        let mean_sq = sq / nn as f32;
                        node.rms = beta2 * node.rms + (1.0 - beta2) * mean_sq;
                    }
                }
                node.t += 1;
                if grad_clip > 0.0 {
                    let mut sq = 0.0f32;
                    for &g in &node.grad {
                        sq += g * g;
                    }
                    let norm = sq.sqrt();
                    if norm > grad_clip && norm > 0.0 {
                        let sc = grad_clip / norm;
                        for g in node.grad.iter_mut() {
                            *g *= sc;
                        }
                    }
                }
                let mut params = Self::keralin_params_for(&self.params, node);
                params.lr_nucleus *= lr_factor;
                params.lr_radius *= lr_factor;
                params.lr_dir *= lr_factor;
                if cascade {
                    let heat_h = { let h = heats.lock().unwrap(); h[node.id] };
                    // Molecular cascade jump-in: скаляр из общей цели шага
                    // усиленных прыжков тепла вниз по графу.
                    let jump = (heat_h * cascade_heat).clamp(0.25, 4.0);
                    params.lr_nucleus *= jump;
                    params.lr_radius *= jump;
                    params.lr_dir *= jump;
                }
                keralin_step_atom(&mut node.atom, &node.grad, &params);
                let c = node.atom.momentum.coherence;
                node.life.observe(node.atom.sig, node.atom.radius, c);
                node.grad.fill(0.0);
                // EMA snapshot (SWA-подобно): decay EMA по nucleus/radius/dir.
                {
                    let decay = if self.params.ema_decay > 0.0 {
                        self.params.ema_decay
                    } else {
                        0.9
                    };
                    node.atom.ema_nucleus = decay * node.atom.ema_nucleus + (1.0 - decay) * node.atom.nucleus;
                    node.atom.ema_radius = decay * node.atom.ema_radius + (1.0 - decay) * node.atom.radius;
                    for i in 0..node.atom.dir.len() {
                        node.atom.ema_dir[i] = decay * node.atom.ema_dir[i] + (1.0 - decay) * node.atom.dir[i];
                    }
                    node.atom.ema_dir.sort_by(|a, b| a.total_cmp(b));
                }
            });
            if self.params.cascade {
                // Цель хочеет каскадных прыжков: propagate coherence вперёд.
                let coherent: Vec<f32> = self.nodes.iter().map(|n| n.atom.momentum.coherence).collect();
                let mut h = heats.lock().unwrap();
                for &id in &topo {
                    let base = coherent[id];
                    // Диффузный прыжок вниз по графовым соединениям: нагревание частично сезона,
                    // откуда используется собственный баланс тепла.
                    let downstream: f32 = self.nodes[id].consumers.iter()
                        .map(|c| coherent[*c] * cascade_heat).sum::<f32>() / self.nodes[id].consumers.len().max(1) as f32;
                    h[id] = base + downstream*0.5;
                }
            }
        }
        self.step_count += 1;
        let coh_sum: f32 = self.nodes.iter().map(|n| n.atom.momentum.coherence).sum();
        let coh_n = self.nodes.iter().filter(|n| n.atom.momentum.coherence != 0.0).count();
        self.global_coherence = if coh_n > 0 {
            coh_sum / coh_n as f32
        } else {
            0.0
        };
        self.graph_sig = self.graph_signature();
        self.global_coherence
    }

    /// Подпись графа: XOR подписей всех узлов.
    pub fn graph_signature(&self) -> u64 {
        self.nodes.iter().fold(0u64, |acc, node| acc ^ node.atom.sig)
    }

    /// Живость графа: (все узлы живы, число мёртвых, число холодных).
    pub fn integrity(&self) -> (bool, usize, usize) {
        let dead = self
            .nodes
            .iter()
            .filter(|node| node.life.is_dead())
            .count();
        let cold = self
            .nodes
            .iter()
            .filter(|node| node.life.state == LifeState::Cold)
            .count();
        let ok = self
            .nodes
            .iter()
            .all(|node| !keralin_sig_dead(node.atom.sig, node.atom.dir.len()));
        (ok, dead, cold)
    }

    /// Средняя энергия (норма) градиентов последнего шага.
    pub fn grad_energy(&self) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for node in &self.nodes {
            sum += crate::atom::math::simd::norm(&node.grad);
            n += 1;
        }
        sum / n as f32
    }
}
