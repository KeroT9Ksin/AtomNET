// src/atom/math/tensor.rs
//! AtomMATH: тензорный слой — Tensor, GEMM/GEMV, активации, норма, softmax, CE, backward.
//!
//! Это «чистая» математика нового поколения AtomNET:
//! - [`Tensor`] — компактный row-major f32 тензор с явной формой;
//! - [`matmul`] / [`matvec`] — GEMM/GEMV на ts-ядрах AtomMATH (AVX2+FMA,
//!   чанковая редукция, детерминированный порядок сумм);
//! - активации (silu/gelu/tanh/sigmoid/relu) с аналитическими backward;
//! - нормализация (RMSNorm/LayerNorm) и batched cross-entropy;
//! - [`linear_backward`] — dW = Σ dout⊗x, dx = Wᵀ·dout.
//!
//! Все функции — детерминированные чистые функции над `&[f32]` / `&Tensor`:
//! без глобального состояния, без градиентного трафика, без обучения.

use crate::atom::math::simd;
use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════
// Tensor
// ═══════════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0f32; rows * cols],
            rows,
            cols,
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0f32; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols, "tensor: len must equal rows*cols");
        Self { data, rows, cols }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn numel(&self) -> usize {
        self.rows * self.cols
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

// ═══════════════════════════════════════════════════════════
// GEMM / GEMV
// ═══════════════════════════════════════════════════════════

/// C = A × B (row-major). C[i][j] = Σ_k A[i][k] · B[k][j].
/// Блочный FLASH-GEMM (4×16 AVX2 + JIT, панели B, fork по строкам) —
/// в разы быстрее наивного axpy-цикла; малые задачи (<8M MAC) идут
/// последовательно в том же ядре. Детерминирован бит-в-бит.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    debug_assert_eq!(a.cols, b.rows, "matmul: inner dims mismatch");
    let mut data = vec![0.0f32; a.rows * b.cols];
    let parts =
        crate::atom::flash::flash_gemm::gemm_auto_parts(0, a.rows, a.cols, b.cols);
    crate::atom::flash::flash_gemm::gemm_blocked_into(
        &a.data, a.rows, a.cols, &b.data, b.cols, parts, &mut data,
    );
    Tensor {
        data,
        rows: a.rows,
        cols: b.cols,
    }
}

/// C = A × B с записью в готовый буфер (без промежуточного Vec).
pub fn matmul_into(a: &Tensor, b: &Tensor, out: &mut [f32]) {
    debug_assert_eq!(a.cols, b.rows, "matmul_into: inner dims mismatch");
    debug_assert_eq!(out.len(), a.rows * b.cols, "matmul_into: out len mismatch");
    let parts =
        crate::atom::flash::flash_gemm::gemm_auto_parts(0, a.rows, a.cols, b.cols);
    crate::atom::flash::flash_gemm::gemm_blocked_into(
        &a.data, a.rows, a.cols, &b.data, b.cols, parts, out,
    );
}

/// y = A × x (GEMV). A: rows×cols, x: cols.
pub fn matvec(a: &Tensor, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; a.rows];
    if a.rows > 0 {
        let want_par = crate::atom::veron::parallel_threads() > 1 && a.rows >= 8;
        let out: Vec<f32> = if want_par {
            (0..a.rows)
                .into_par_iter()
                .map(|i| simd::dot(a.row(i), x))
                .collect::<Vec<f32>>()
        } else {
            (0..a.rows).map(|i| simd::dot(a.row(i), x)).collect()
        };
        y = out;
    }
    y
}

/// y = A × x для row-major буфера A (rows×cols): GEMV без аллокаций Tensor.
pub fn matvec_slice(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    if rows > 0 {
        let want_par = crate::atom::veron::parallel_threads() > 1 && rows >= 8;
        let out: Vec<f32> = if want_par {
            (0..rows)
                .into_par_iter()
                .map(|i| simd::dot(&w[i * cols..(i + 1) * cols], x))
                .collect::<Vec<f32>>()
        } else {
            (0..rows).map(|i| simd::dot(&w[i * cols..(i + 1) * cols], x)).collect()
        };
        y = out;
    }
    y
}

// ═══════════════════════════════════════════════════════════
// Активации + backward
// ═══════════════════════════════════════════════════════════

/// silu(x) = x·σ(x).
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// d(silu)/dx.
pub fn silu_b(d: f32, x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    d * (s + x * s * (1.0 - s))
}

/// gelu(x) = ½x(1 + tanh(√(2/π)·(x + 0.044715·x³))).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608028654_f32 * (x + 0.044715 * x * x * x)).tanh())
}

/// d(gelu)/dx.
pub fn gelu_b(d: f32, x: f32) -> f32 {
    let c = 0.7978845608028654_f32;
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    let du = c * (1.0 + 3.0 * 0.044715 * x * x);
    d * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)
}

pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

pub fn tanh_b(d: f32, x: f32) -> f32 {
    d * (1.0 - x.tanh() * x.tanh())
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_b(d: f32, x: f32) -> f32 {
    let s = sigmoid(x);
    d * s * (1.0 - s)
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn relu_b(d: f32, x: f32) -> f32 {
    d * if x > 0.0 { 1.0 } else { 0.0 }
}

// ═══════════════════════════════════════════════════════════
// Нормализация
// ═══════════════════════════════════════════════════════════

/// RMSNorm: y = x / √(mean(x²)+eps) · gamma.
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n.max(1) as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter()
        .zip(gamma)
        .map(|(&xv, &g)| xv * inv * g)
        .collect()
}

/// Backward RMSNorm: (dx, dgamma).
pub fn rms_norm_bwd(dout: &[f32], x: &[f32], gamma: &[f32], eps: f32) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n.max(1) as f32;
    let s = (mean_sq + eps).sqrt();
    let inv = 1.0 / s;
    let mut dx = vec![0.0f32; n];
    let mut dgamma = vec![0.0f32; n];
    let common = dout
        .iter()
        .zip(x)
        .zip(gamma)
        .map(|((&d, &xv), &g)| d * xv * g)
        .sum::<f32>();
    let inv3 = inv * inv * inv;
    for i in 0..n {
        dgamma[i] = dout[i] * x[i] * inv;
        dx[i] = dout[i] * gamma[i] * inv - x[i] * inv3 * common / n.max(1) as f32;
    }
    (dx, dgamma)
}

/// LayerNorm: y = xhat·gamma + beta, xhat = (x−mean)/√(var+eps).
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n.max(1) as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n.max(1) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(gamma.iter().zip(beta))
        .map(|(&xv, (&g, &b))| (xv - mean) * inv * g + b)
        .collect()
}

/// Backward LayerNorm: (dx, dgamma, dbeta).
pub fn layer_norm_bwd(
    dout: &[f32],
    x: &[f32],
    gamma: &[f32],
    _beta: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = x.len();
    let mean = x.iter().sum::<f32>() / n.max(1) as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n.max(1) as f32;
    let inv = 1.0 / (var + eps).sqrt();
    let mut dx = vec![0.0f32; n];
    let mut dgamma = vec![0.0f32; n];
    let mut dbeta = vec![0.0f32; n];
    // Восстановление batch-структуры: здесь x/dout — одного образца, поэтому
    // mean по оси нормализации считается по длине n.
        for i in 0..n {
        let xhat = (x[i] - mean) * inv;
        let dxhat = dout[i] * gamma[i];
        let dsum = dxhat / n.max(1) as f32;
        let mut dsum_xhat = 0.0f32;
        for j in 0..n {
            dsum_xhat += dxhat * ((x[j] - mean) * inv);
        }
        dsum_xhat /= n.max(1) as f32;
        dx[i] = inv * (dxhat - dsum - xhat * dsum_xhat);
        dgamma[i] = dout[i] * ((x[i] - mean) * inv);
        dbeta[i] = dout[i];
    }
    (dx, dgamma, dbeta)
}

// ═══════════════════════════════════════════════════════════
// Softmax + cross-entropy
// ═══════════════════════════════════════════════════════════

/// Softmax по последней оси: p = softmax(x).
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        let mut out = vec![0.0f32; n];
        unsafe {
            (m.softmax)(
                x.as_ptr(),
                out.as_mut_ptr(),
                1,
                n,
                std::ptr::null_mut(),
            )
        };
        return out;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = vec![0.0f32; n];
    let mut sum = 0.0f32;
    for (e, &l) in exps.iter_mut().zip(x) {
        let v = (l - max).exp();
        *e = v;
        sum += v;
    }
    let inv = 1.0 / sum.max(1e-12);
    for e in &mut exps {
        *e *= inv;
    }
    exps
}

/// Softmax + cross-entropy для одного образца: (loss, dlogits).
pub fn softmax_ce(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    #[cfg(target_arch = "x86_64")]
    if let Some(m) = crate::atom::flash::jit::jit_math() {
        if !logits.is_empty() {
            let mut d = vec![0.0f32; logits.len()];
            let mut stats = [0.0f32; 2];
            unsafe {
                (m.softmax)(
                    logits.as_ptr(),
                    d.as_mut_ptr(),
                    1,
                    logits.len(),
                    stats.as_mut_ptr(),
                )
            };
            // loss = ln(sum) − (logit[target] − max): та же устойчивая
            // формула, что и в скалярном пути, но max/sum уже посчитаны ASM.
            let loss = (stats[1].ln() - (logits[target] - stats[0])).max(0.0);
            d[target] -= 1.0;
            return (loss, d);
        }
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut exps = vec![0.0f32; logits.len()];
    for (e, &l) in exps.iter_mut().zip(logits) {
        let v = (l - max).exp();
        *e = v;
        sum += v;
    }
    let loss = (sum.ln() - (logits[target] - max)).max(0.0);
    let inv = 1.0 / sum.max(1e-12);
    let mut d = exps;
    for v in d.iter_mut() {
        *v *= inv;
    }
    d[target] -= 1.0;
    (loss, d)
}

/// Энтропия распределения логитов (для регуляризации/уверенности).
pub fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    // Один ASM-проход exp вместо двух; ln остаётся скалярным (нужен для
    // энтропии, не для вероятностей).
    let p = softmax(logits);
    let mut h = 0.0f32;
    for &v in &p {
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    h
}

/// Batched cross-entropy: logits B×V (flat), targets B → (mean_loss, dlogits B×V).
pub fn cross_entropy(logits: &[f32], targets: &[u32], vocab: usize) -> (f32, Vec<f32>) {
    let b = targets.len();
    // Батчевый ASM-softmax: один вызов на B×V, stats на строку.
    #[cfg(target_arch = "x86_64")]
    if b > 0 && vocab > 0 && logits.len() >= b * vocab {
        if let Some(m) = crate::atom::flash::jit::jit_math() {
            let mut d = vec![0.0f32; logits.len()];
            let mut stats = vec![0.0f32; b * 2];
            unsafe {
                (m.softmax)(
                    logits.as_ptr(),
                    d.as_mut_ptr(),
                    b,
                    vocab,
                    stats.as_mut_ptr(),
                )
            };
            let mut loss = 0.0f32;
            for t in 0..b {
                let tg = targets[t] as usize;
                let max = stats[2 * t];
                let sum = stats[2 * t + 1];
                loss += (sum.ln() - (logits[t * vocab + tg] - max)).max(0.0);
                d[t * vocab + tg] -= 1.0;
            }
            return (loss / b.max(1) as f32, d);
        }
    }
    let mut loss = 0.0f32;
    let mut d = vec![0.0f32; logits.len()];
    for t in 0..b {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let (l, dd) = softmax_ce(row, targets[t] as usize);
        loss += l;
        for (k, v) in dd.iter().enumerate() {
            d[t * vocab + k] = *v;
        }
    }
    (loss / b.max(1) as f32, d)
}

// ═══════════════════════════════════════════════════════════
// Linear backward
// ═══════════════════════════════════════════════════════════

/// linear_backward: dw (rows×cols) = Σ_t dout_t ⊗ x_t; dx = Wᵀ·dout.
/// `dout` имеет длину rows·l (l = batch), `x` — l·cols.
pub fn linear_backward(
    w: &Tensor,
    rows: usize,
    cols: usize,
    dout: &[f32],
    x: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let l = dout.len() / rows;
    let mut dw = vec![0.0f32; rows * cols];
    let mut dx = vec![0.0f32; l * cols];
    for t in 0..l {
        let drow = &dout[t * rows..(t + 1) * rows];
        let xrow = &x[t * cols..(t + 1) * cols];
        for i in 0..rows {
            let di = drow[i];
            let base = i * cols;
            for j in 0..cols {
                dw[base + j] += di * xrow[j];
                dx[t * cols + j] += di * w.row(i)[j];
            }
        }
    }
    (dw, dx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(rows: usize, cols: usize, fill: f32) -> Tensor {
        let mut t = Tensor::new(rows, cols);
        for v in t.data.iter_mut() {
            *v = fill;
        }
        t
    }

    #[test]
    fn matmul_identity_and_values() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), (2, 2));
        assert!((c.row(0)[0] - 19.0).abs() < 1e-5);
        assert!((c.row(0)[1] - 22.0).abs() < 1e-5);
        assert!((c.row(1)[0] - 43.0).abs() < 1e-5);
        assert!((c.row(1)[1] - 50.0).abs() < 1e-5);

        let e = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let ce = matmul(&e, &a);
        for i in 0..2 {
            for j in 0..2 {
                assert!((ce.row(i)[j] - a.row(i)[j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn matmul_batched_shape() {
        // A (2×2) × B (2×4) → C (2×4): C[i] = Σ_k A[i][k]·B[k][:]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = Tensor::from_vec(vec![1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 2.0], 2, 4);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), (2, 4));
        // C[0] = 1*b0 + 2*b1 = [1,2,2,4]
        assert!((c.row(0)[0] - 1.0).abs() < 1e-6);
        assert!((c.row(0)[1] - 2.0).abs() < 1e-6);
        assert!((c.row(0)[2] - 2.0).abs() < 1e-6);
        assert!((c.row(0)[3] - 4.0).abs() < 1e-6);
        // C[1] = 3*b0 + 4*b1 = [3,4,6,8]
        assert!((c.row(1)[0] - 3.0).abs() < 1e-6);
        assert!((c.row(1)[1] - 4.0).abs() < 1e-6);
        assert!((c.row(1)[2] - 6.0).abs() < 1e-6);
        assert!((c.row(1)[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn matvec_matches_matmul() {
        let a = tensor(3, 2, 0.5);
        let x = vec![2.0, 3.0];
        let y = matvec(&a, &x);
        // y_i = Σ_j 0.5 * x_j = 0.5 * (x0 + x1) = 2.5 для всех строк
        assert!((y[0] - 2.5).abs() < 1e-6);
        assert!((y[2] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_ce_gradcheck() {
        let logits = vec![0.1f32, -0.2, 0.4, 1.5];
        let target = 2usize;
        let (loss, d) = softmax_ce(&logits, target);
        assert!(loss.is_finite() && loss >= 0.0);
        assert_eq!(d.len(), logits.len());
        assert!(d[target] < 0.0);
        assert!((d.iter().sum::<f32>()).abs() < 1e-5);

        let eps = 1e-3f32;
        let mut hi = logits.clone();
        hi[target] += eps;
        let (l1, _) = softmax_ce(&hi, target);
        let mut lo = logits.clone();
        lo[target] -= eps;
        let (l0, _) = softmax_ce(&lo, target);
        let num = (l1 - l0) / (2.0 * eps);
        // d(loss)/d(logit[target]) = p_t - 1
        let probs = softmax(&logits);
        let expected = probs[target] - 1.0;
        assert!(
            (num - expected).abs() < 1e-2,
            "softmax_ce grad: numeric={}, expected={expected}",
            num
        );
    }

    #[test]
    fn gelu_gradcheck() {
        let x = 0.7f32;
        let d = 1.3f32;
        let analytic = gelu_b(d, x) / d; // d(gelu)/dx
        let eps = 1e-3f32;
        let mut hp = x;
        let mut lp = x;
        hp += eps;
        lp -= eps;
        let num = (gelu(hp) - gelu(lp)) / (2.0 * eps);
        assert!((num - analytic).abs() < 1e-2, "gelu bwd: analytic={analytic}, num={num}");
    }

    #[test]
    fn layer_norm_bwd_shape() {
        let x = vec![0.2f32, -0.4, 0.9, 0.1, -0.3, 0.5];
        let gamma = vec![1.0f32, 0.8, 1.2, 0.9, 1.1, 1.0];
        let beta = vec![0.05f32; 6];
        let eps = 1e-5f32;
        let y = layer_norm(&x, &gamma, &beta, eps);
        assert_eq!(y.len(), x.len());
        let (dx, dg, db) = layer_norm_bwd(&y, &x, &gamma, &beta, eps);
        assert_eq!(dx.len(), x.len());
        assert_eq!(dg.len(), x.len());
        assert_eq!(db.len(), x.len());
    }
}
