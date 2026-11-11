// src/atom/physics/hexiron.rs
//! Hexiron: шестиугольная физика стабильности.
//!
//! Шесть точек / вершин:
//! 1. energy       — энергия сигнала / градиента.
//! 2. coherence    — согласованность направления.
//! 3. entropy      — шум / нестабильность.
//! 4. temperature  — физическая температура CPU.
//! 5. dissipation  — коэффициент трения / потерь.
//! 6. lifetime     — ресурс жизни состояния.
//!
//! Задача:
//! не давать системе уходить за пределы устойчивости.

#[derive(Clone, Copy, Debug)]
pub struct HexironParams {
    pub lifetime_min: f32,
    pub lifetime_max: f32,

    pub max_energy: f32,
    pub max_entropy: f32,
    pub max_temperature: f32,

    pub dissipation_base: f32,
    pub temperature_coeff: f32,
    pub entropy_coeff: f32,
    pub incoherence_coeff: f32,

    /// Насколько быстро lifetime возвращается к целевому значению.
    pub lifetime_recovery: f32,
}

impl Default for HexironParams {
    fn default() -> Self {
        Self {
            lifetime_min: 0.0,
            lifetime_max: 1.0,

            max_energy: 100.0,
            max_entropy: 10.0,
            max_temperature: 95.0,

            dissipation_base: 0.05,
            temperature_coeff: 0.60,
            entropy_coeff: 0.35,
            incoherence_coeff: 0.25,

            lifetime_recovery: 0.20,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HexironState {
    pub energy: f32,
    pub coherence: f32,
    pub entropy: f32,
    pub temperature: f32,
    pub dissipation: f32,
    pub lifetime: f32,
    pub stable: bool,
}

impl Default for HexironState {
    fn default() -> Self {
        Self {
            energy: 0.0,
            coherence: 0.0,
            entropy: 0.0,
            temperature: 0.0,
            dissipation: 0.0,
            lifetime: 1.0,
            stable: true,
        }
    }
}

pub struct Hexiron {
    pub state: HexironState,
    pub params: HexironParams,
}

impl Hexiron {
    pub fn new() -> Self {
        Self {
            state: HexironState::default(),
            params: HexironParams::default(),
        }
    }

    /// Обновить физику по четырём входным сигналам.
    ///
    /// energy:
    ///     энергия градиента / активации.
    ///
    /// coherence:
    ///     направленная когерентность, обычно [-1, 1].
    ///
    /// entropy:
    ///     шум, нестабильность, частота осцилляций.
    ///
    /// temperature:
    ///     температура CPU в градусах.
    pub fn update(
        &mut self,
        energy: f32,
        coherence: f32,
        entropy: f32,
        temperature: f32,
    ) {
        let e_norm = (energy / self.params.max_energy).clamp(0.0, 1.0);

        let c = coherence.clamp(-1.0, 1.0);
        let c_norm = (c + 1.0) * 0.5;

        let s_norm = (entropy / self.params.max_entropy).clamp(0.0, 1.0);
        let t_norm = (temperature / self.params.max_temperature).clamp(0.0, 1.0);

        // Целевой уровень жизни.
        //
        // Высокая когерентность повышает жизнь.
        // Высокая энтропия понижает жизнь.
        // Высокая температура понижает жизнь.
        let target_lifetime = (0.25 + 0.75 * c_norm)
            * (1.0 - 0.45 * s_norm)
            * (1.0 - 0.35 * t_norm);

        // Iron-коррекция: возвращаем lifetime к цели.
        let lifetime_error = target_lifetime - self.state.lifetime;
        self.state.lifetime += self.params.lifetime_recovery * lifetime_error;

        // Жёсткие пределы.
        self.state.lifetime = self
            .state
            .lifetime
            .clamp(self.params.lifetime_min, self.params.lifetime_max);

        // Диссипация / трение.
        self.state.dissipation = (self.params.dissipation_base
            + self.params.temperature_coeff * t_norm
            + self.params.entropy_coeff * s_norm
            + self.params.incoherence_coeff * (1.0 - c_norm))
            .max(0.0);

        self.state.energy = energy;
        self.state.coherence = c;
        self.state.entropy = entropy;
        self.state.temperature = temperature;

        self.state.stable = self.state.lifetime >= 0.30
            && self.state.dissipation <= 2.0
            && s_norm <= 0.85
            && e_norm <= 0.98;
    }

    /// Эффективный learning rate с учётом диссипации.
    ///
    /// Формула:
    ///     lr_eff = lr_base / (1 + ζ)
    pub fn effective_lr(&self, base_lr: f32) -> f32 {
        base_lr / (1.0 + self.state.dissipation.max(0.0))
    }

    /// Общая устойчивость от 0 до 1.
    pub fn stability(&self) -> f32 {
        let dissipation_norm = (self.state.dissipation / 3.0).clamp(0.0, 1.0);
        let lifetime_part = self.state.lifetime.clamp(0.0, 1.0);
        let dissipation_part = 1.0 - dissipation_norm;

        (lifetime_part * dissipation_part).clamp(0.0, 1.0)
    }

    /// Шесть нормализованных вершин шестиугольника.
    ///
    /// Порядок:
    /// [
    ///     energy,
    ///     coherence,
    ///     entropy,
    ///     temperature,
    ///     dissipation,
    ///     lifetime,
    /// ]
    pub fn vertices(&self) -> [f32; 6] {
        let e_norm = (self.state.energy / self.params.max_energy).clamp(0.0, 1.0);

        let c_norm = (self.state.coherence.clamp(-1.0, 1.0) + 1.0) * 0.5;

        let s_norm = (self.state.entropy / self.params.max_entropy).clamp(0.0, 1.0);

        let t_norm = (self.state.temperature / self.params.max_temperature).clamp(0.0, 1.0);

        let d_norm = (self.state.dissipation / 3.0).clamp(0.0, 1.0);

        let l_norm = self.state.lifetime.clamp(0.0, 1.0);

        [e_norm, c_norm, s_norm, t_norm, d_norm, l_norm]
    }
}

impl Default for Hexiron {
    fn default() -> Self {
        Self::new()
    }
}
