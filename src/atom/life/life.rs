// src/atom_math/life.rs
//! AtomLife: чистая математика жизни/состояния атома.
//!
//! Это НЕ оптимизатор.
//! Здесь нет обучения, нет lr, нет обновления весов.
//!
//! Задача:
//! - отслеживать живость атома;
//! - отслеживать стабильность;
//! - отслеживать энергию и когерентность;
//! - возвращать lifetime-гейт для математических операций.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LifeState {
    Dead,
    Cold,
    Born,
    Alive,
    Stable,
    Decaying,
}

#[derive(Clone, Copy, Debug)]
pub struct LifeParams {
    pub alive_bits_min: u32,
    pub stable_bits_min: u32,
    pub energy_min: f32,
    pub coherence_min: f32,
    pub hamming_stable_max: u32,
    pub hamming_decay_min: u32,
    pub stability_gain: f32,
    pub stability_decay: f32,
}

impl Default for LifeParams {
    fn default() -> Self {
        Self {
            alive_bits_min: 10,
            stable_bits_min: 20,
            energy_min: 1e-6,
            coherence_min: 0.05,
            hamming_stable_max: 4,
            hamming_decay_min: 16,
            stability_gain: 0.15,
            stability_decay: 0.10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AtomLife {
    pub state: LifeState,
    pub age: u32,
    pub usage: u32,
    pub energy: f32,
    pub coherence: f32,
    pub stability: f32,
    pub sig: u64,
    pub prev_sig: u64,
    pub hamming: u32,
    pub initialized: bool,
    pub params: LifeParams,
}

impl AtomLife {
    pub fn new(params: LifeParams) -> Self {
        Self {
            state: LifeState::Dead,
            age: 0,
            usage: 0,
            energy: 0.0,
            coherence: 0.0,
            stability: 0.0,
            sig: 0,
            prev_sig: 0,
            hamming: 0,
            initialized: false,
            params,
        }
    }

    /// Обновить состояние жизни по подписи, энергии и когерентности.
    ///
    /// sig:
    ///     XOR-подпись текущего блока/вектора.
    ///
    /// energy:
    ///     текущая энергия, например норма или активация.
    ///
    /// coherence:
    ///     направленная согласованность, например cos(x, y).
    pub fn observe(&mut self, sig: u64, energy: f32, coherence: f32) {
        let bits = sig.count_ones();

        if self.initialized {
            self.hamming = (sig ^ self.sig).count_ones();
        } else {
            self.hamming = 0;
            self.initialized = true;
        }

        self.prev_sig = self.sig;
        self.sig = sig;

        self.age += 1;

        if energy > self.params.energy_min {
            self.usage += 1;
        }

        // EMA-сглаживание энергии и когерентности.
        self.energy = 0.9 * self.energy + 0.1 * energy.max(0.0);
        self.coherence = 0.9 * self.coherence + 0.1 * coherence.clamp(-1.0, 1.0);

        let energy_active = self.energy >= self.params.energy_min;

        let stable_now = bits >= self.params.stable_bits_min
            && self.hamming <= self.params.hamming_stable_max
            && self.coherence >= self.params.coherence_min
            && energy_active;

        if stable_now {
            self.stability = (self.stability + self.params.stability_gain).min(1.0);
        } else {
            self.stability = (self.stability - self.params.stability_decay).max(0.0);
        }

        let dead = bits <= 2 || self.energy < self.params.energy_min * 0.05;

        let cold = !dead && bits < self.params.alive_bits_min;

        let decaying = self.hamming >= self.params.hamming_decay_min
            || self.energy < self.params.energy_min * 0.25
            || self.coherence < -0.25;

        self.state = if dead {
            LifeState::Dead
        } else if cold {
            LifeState::Cold
        } else if decaying && self.stability < 0.35 {
            LifeState::Decaying
        } else if self.stability >= 0.75 {
            LifeState::Stable
        } else if self.age < 8 {
            LifeState::Born
        } else {
            LifeState::Alive
        };
    }

    /// Плавный гейт без дискретных скачков.
    ///
    /// Старая проблема:
    ///     born  -> 0.8
    ///     alive -> 1.0
    /// При age = 8 ВСЕ атомы переключались одновременно,
    /// и 8 слоёв давали скачок (1/0.8)^8 ≈ 6x.
    ///
    /// Теперь гейт — непрерывная функция возраста и стабильности.
    pub fn gate(&self) -> f32 {
        // Плавное «созревание» за первые 8 шагов: 0..1
        let t = (self.age as f32 / 8.0).clamp(0.0, 1.0);

        // smoothstep: 3t^2 - 2t^3 — без излома
        let maturity = t * t * (3.0 - 2.0 * t);

        // 0.8 -> 1.0 без скачка
        let base = 0.8 + 0.2 * maturity;

        // Мягкая надбавка за стабильность: 0.95..1.0
        let soft = 0.95 + 0.05 * self.stability.clamp(0.0, 1.0);

        let full = (base * soft).clamp(0.0, 1.0);

        match self.state {
            LifeState::Dead => 0.0,
            LifeState::Cold => 0.2 * full,
            LifeState::Decaying => 0.5 * full,
            _ => full,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            LifeState::Born | LifeState::Alive | LifeState::Stable
        )
    }

    pub fn is_dead(&self) -> bool {
        self.state == LifeState::Dead
    }

    pub fn is_stable(&self) -> bool {
        self.state == LifeState::Stable
    }
}
