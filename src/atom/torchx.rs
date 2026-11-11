// src/atom/torchx.rs
//! torchx — Tensor с автодифференцированием в духе torch.autograd.
//!
//! Документированные отклонения от PyTorch (CPU-only, f32):
//! - нет double backward (create_graph=False — дефолт PyTorch),
//! - нет распределённого/accelerator, нет foreach/fused путей,
//! - RNG не побитово совпадает с mt19937 PyTorch.
//!
//! В остальном семантика = PyTorch: leaf.grad компауналится через
//! backward (accumulateция между вызовами), граф освобождается после
//! backward (кроме retain_grad), in-place на leaf с requires_grad — ошибка,
//! softmax/CE/LayerNorm — точные формулы из torch/nn/functional.py.

use crate::atom::math::simd;
use crate::atom::nn::silu;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Узел графа автодифференцирования.
pub struct TxNode {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: std::cell::Cell<bool>,
    /// leaf.grad: None = "градиента нет" (set_to_none семантика).
    pub grad: RefCell<Option<Vec<f32>>>,
    pub grad_fn: RefCell<Option<Box<GradFn>>>,
    pub retained: RefCell<bool>,
    /// имя следующей операции (для repr, как grad_fn в PyTorch).
    pub op_name: RefCell<Option<String>>,
}

pub type Tx = Rc<TxNode>;

pub struct GradFn {
    pub inputs: Vec<Tx>,
    /// (grad_output, input_index) -> grad wrt inputs[i].
    pub bwd: Rc<dyn Fn(&[f32], usize) -> Vec<f32> + 'static>,
}

/// Xoshiro256** — детерминированный RNG для factories.
#[derive(Clone)]
pub struct Xoroshiro {
    s: [u64; 4],
}

impl Xoroshiro {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed;
        for slot in s.iter_mut() {
            // splitmix64 warmup
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut t = z;
            t = (t ^ (t >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            t = (t ^ (t >> 27)).wrapping_mul(0x94D049BB133111EB);
            *slot = t ^ (t >> 31);
        }
        Self { s }
    }

    pub fn next_f64(&mut self) -> f64 {
        let result = ((self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9)) as f64;
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        (result / (u64::MAX as f64)) as f64
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    pub fn normal(&mut self) -> f32 {
        // Box-Muller
        let u1 = self.next_f32().max(1e-12);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

thread_local! {
    static RNG: RefCell<Xoroshiro> = RefCell::new(Xoroshiro::new(0x853c49e6748fea9b));
    /// torch.no_grad: счётчик вложенности; 0 => grad выключен (eval-проходы).
    static GRAD_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

pub fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

pub fn set_grad_enabled(on: bool) {
    GRAD_ENABLED.with(|g| g.set(on));
}

/// NPU-симулятор обучения: ATOMNET_NPU_EMU=1 -> GEMM идёт через JIT-тайловый
/// блок (6x16 AVX2+FMA mac-массив), один поток, независимо от бюджета —
/// какпосторочный («посторочный» = каждый вывод по строкам-массивам) mac-
/// тракт NPU. Одинаковый нетвурк и результат при ATOMNET_THREADS=1.
pub fn npu_emu_on() -> bool {
    std::env::var("ATOMNET_NPU_EMU").map(|v| v == "1").unwrap_or(false)
}

pub fn manual_seed(seed: u64) {
    RNG.with(|r| *r.borrow_mut() = Xoroshiro::new(seed));
}

fn next_rand() -> f32 {
    RNG.with(|r| r.borrow_mut().next_f32())
}

fn next_randn() -> f32 {
    RNG.with(|r| r.borrow_mut().normal())
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

pub fn tx(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Tx {
    Rc::new(TxNode {
        data,
        shape,
        requires_grad: std::cell::Cell::new(requires_grad),
        grad: RefCell::new(None),
        grad_fn: RefCell::new(None),
        retained: RefCell::new(false),
        op_name: RefCell::new(None),
    })
}

/// Конструктор с проверкой размера (как torch.tensor по списку).
pub fn tensor_from(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Result<Tx, String> {
    let need = numel(&shape);
    if data.len() != need && !shape.is_empty() {
        return Err(format!(
            "число элементов ({}) не совпадает с shape {:?} ({})",
            data.len(),
            shape,
            need
        ));
    }
    let data = if shape.is_empty() {
        data[..1].to_vec()
    } else {
        data
    };
    Ok(tx(data, shape, requires_grad))
}

// ═══════════════════════════════════════════════════════════════════
// Broadcasting (правила PyTorch: выравнивание справа, 1 => expand)
// ═══════════════════════════════════════════════════════════════════

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let na = a.len();
    let nb = b.len();
    let n = na.max(nb);
    let mut out = Vec::with_capacity(n);
    for dim in 0..n {
        let da = if dim < n - na { 1 } else { a[dim - (n - na)] };
        let db = if dim < n - nb { 1 } else { b[dim - (n - nb)] };
        match (da, db) {
            (x, y) if x == y => out.push(x),
            (1, y) => out.push(y),
            (x, 1) => out.push(x),
            _ => {
                return Err(format!(
                    "несовместимые формы для broadcast: {:?} vs {:?}",
                    a, b
                ))
            }
        }
    }
    Ok(out)
}

fn index_map(src_shape: &[usize], tgt_shape: &[usize], flat: usize) -> usize {
    // Разворачиваем flat-индекс в координате tgt, сворачивая src (dim==1 -> 0).
    let mut rem = flat;
    let mut coords = vec![0usize; tgt_shape.len()];
    for d in (0..tgt_shape.len()).rev() {
        coords[d] = rem % tgt_shape[d];
        rem /= tgt_shape[d];
    }
    let lead = tgt_shape.len() - src_shape.len();
    let mut idx = 0usize;
    let mut stride = 1usize;
    for d in (0..src_shape.len()).rev() {
        let c = if src_shape[d] == 1 {
            0
        } else {
            coords[d + lead]
        };
        idx += c * stride;
        stride *= src_shape[d];
    }
    idx
}

/// Материализует данные src_shape -> tgt_shape (expand).
pub fn expand(data: &[f32], src_shape: &[usize], tgt_shape: &[usize]) -> Vec<f32> {
    if src_shape == tgt_shape {
        return data.to_vec();
    }
    let need = numel(tgt_shape);
    let mut out = vec![0.0f32; need];
    for i in 0..need {
        out[i] = data[index_map(src_shape, tgt_shape, i)];
    }
    out
}

/// Суммирование градиента по broadcast-осям (обратная сторона expand).
pub fn unexpand(grad: &[f32], tgt_shape: &[usize], src_shape: &[usize]) -> Vec<f32> {
    if src_shape == tgt_shape {
        return grad.to_vec();
    }
    let mut out = vec![0.0f32; numel(src_shape)];
    let n = numel(tgt_shape);
    for i in 0..n {
        out[index_map(src_shape, tgt_shape, i)] += grad[i];
    }
    out
}

// ═══════════════════════════════════════════════════════════════════
// Autograd core (топологическая сортировка, накопление .grad листьев)
// ═══════════════════════════════════════════════════════════════════

fn accum(slot: &RefCell<Option<Vec<f32>>>, gi: Vec<f32>) {
    let mut g = slot.borrow_mut();
    match g.as_mut() {
        None => *g = Some(gi),
        Some(cur) => {
            for (c, v) in cur.iter_mut().zip(gi.iter()) {
                *c += *v;
            }
        }
    }
}

/// PyTorch Tensor.backward(): grad опционально (для скаляра — ones).
pub fn backward(t: &Tx, grad: Option<&[f32]>) -> Result<(), String> {
    let shape = &t.shape;
    let seed: Vec<f32> = match grad {
        Some(g) => {
            if g.len() != numel(shape) {
                return Err("grad: размер не совпадает с shape тензора".into());
            }
            g.to_vec()
        }
        None => {
            if t.data.len() != 1 {
                return Err(
                    "backward: gradient должен быть задан для не-скалярного выхода".into(),
                );
            }
            vec![1.0]
        }
    };
    if !t.requires_grad.get() && t.grad_fn.borrow().is_none() {
        return Err(
            "element 0 of tensors does not require grad and does not have a grad_fn".into(),
        );
    }
    // Топологическая сортировка: TRUE post-order (узел после своих входов),
    // затем reverse -> выход (корень) обрабатывается ПЕРВЫМ.
    // (pre-order + reverse даёт неверный порядок, из-за чего градиенты
    // потерь оседают на узлах, уже прошедших обработку.)
    let mut topo: Vec<Tx> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(Tx, bool)> = vec![(t.clone(), false)];
    while let Some((node, processed)) = stack.pop() {
        if processed {
            topo.push(node);
            continue;
        }
        let id = Rc::as_ptr(&node) as usize;
        if !visited.insert(id) {
            continue;
        }
        let inputs: Vec<Tx> = match node.grad_fn.borrow().as_ref() {
            Some(gf) => gf.inputs.clone(),
            None => Vec::new(),
        };
        stack.push((node.clone(), true));
        for inp in inputs {
            stack.push((inp, false));
        }
    }
    topo.reverse(); // topo[0] = выход (корень)

    let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
    grads.insert(Rc::as_ptr(t) as usize, seed);
    for node in topo.iter() {
        let id = Rc::as_ptr(node) as usize;
        let Some(g) = grads.get(&id).cloned() else {
            continue;
        };
        let gf_opt: Option<(Vec<Tx>, Rc<dyn Fn(&[f32], usize) -> Vec<f32>>)> =
            node.grad_fn
                .borrow()
                .as_ref()
                .map(|gf| (gf.inputs.clone(), gf.bwd.clone()));
        if let Some((inputs, bwd)) = gf_opt {
            for (i, inp) in inputs.iter().enumerate() {
                let gi = bwd(&g, i);
                accum(&inp.grad, gi.clone());
                grads
                    .entry(Rc::as_ptr(inp) as usize)
                    .and_modify(|cur| {
                        for (c, v) in cur.iter_mut().zip(gi.iter()) {
                            *c += *v;
                        }
                    })
                    .or_insert(gi);
                // Храним именно накопленный grad для цепочки: grads хранит
                // САМОГО градиента узла; запишем заново последний gi:
                // (чтобы forward-дедупликация не теряла суммирование)
            }
        }
    }
    // Освобождение графа (PyTorch: next_edges отбрасываются после backward),
    // кроме retain_grad-узлов.
    for node in topo.iter() {
        if !*node.retained.borrow() {
            *node.grad_fn.borrow_mut() = None;
        }
    }
    Ok(())
}

/// leaf.zero_ точный аналог (set_to_none=True по умолчанию).
pub fn zero_grad(t: &Tx, set_to_none: bool) {
    let mut g = t.grad.borrow_mut();
    if set_to_none {
        *g = None;
    } else if let Some(cur) = g.as_mut() {
        for v in cur.iter_mut() {
            *v = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Операции: элементарные + broadcasting
// ═══════════════════════════════════════════════════════════════════

fn node(
    data: Vec<f32>,
    shape: Vec<usize>,
    requires: bool,
    inputs: Vec<Tx>,
    bwd: Rc<dyn Fn(&[f32], usize) -> Vec<f32>>,
    name: &str,
) -> Tx {
    // no_grad: не наблюдаем граф, requires_grad результата = False
    // (torch: output.requires_grad == False под no_grad).
    let req = requires && grad_enabled();
    let t = tx(data, shape, req);
    if req {
        *t.grad_fn.borrow_mut() = Some(Box::new(GradFn { inputs, bwd }));
        *t.op_name.borrow_mut() = Some(name.to_string());
    }
    t
}

pub fn add_op(a: &Tx, b: &Tx) -> Result<Tx, String> {
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let ae = expand(&a.data, &a.shape, &shape);
    let be = expand(&b.data, &b.shape, &shape);
    let mut out = vec![0.0f32; ae.len()];
    for i in 0..out.len() {
        out[i] = ae[i] + be[i];
    }
    let req = a.requires_grad.get() || b.requires_grad.get();
    let (as_, bs, shape_c) = (a.shape.clone(), b.shape.clone(), shape.clone());
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        if i == 0 {
            unexpand(g, &shape_c, &as_)
        } else {
            unexpand(g, &shape_c, &bs)
        }
    });
    Ok(node(out, shape, req, vec![a.clone(), b.clone()], f, "AddBackward0"))
}

pub fn sub_op(a: &Tx, b: &Tx) -> Result<Tx, String> {
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let ae = expand(&a.data, &a.shape, &shape);
    let be = expand(&b.data, &b.shape, &shape);
    let mut out = vec![0.0f32; ae.len()];
    for i in 0..out.len() {
        out[i] = ae[i] - be[i];
    }
    let req = a.requires_grad.get() || b.requires_grad.get();
    let (as_, bs) = (a.shape.clone(), b.shape.clone());
    let sh = shape.clone();
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        if i == 0 {
            unexpand(g, &sh, &as_)
        } else {
            unexpand(&g.iter().map(|v| -v).collect::<Vec<_>>(), &sh, &bs)
        }
    });
    Ok(node(out, shape, req, vec![a.clone(), b.clone()], f, "SubBackward0"))
}

pub fn mul_op(a: &Tx, b: &Tx) -> Result<Tx, String> {
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let ae = expand(&a.data, &a.shape, &shape);
    let be = expand(&b.data, &b.shape, &shape);
    let mut out = vec![0.0f32; ae.len()];
    for i in 0..out.len() {
        out[i] = ae[i] * be[i];
    }
    let req = a.requires_grad.get() || b.requires_grad.get();
    let (ae2, be2, as_, bs, sh) = (
        ae.clone(),
        be.clone(),
        a.shape.clone(),
        b.shape.clone(),
        shape.clone(),
    );
    // dA = g*B, dB = g*A (произведение по ДРУГОМУ операнду!)
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        let (m, orig) = if i == 0 { (&be2, &as_) } else { (&ae2, &bs) };
        let mut gi = vec![0.0f32; g.len()];
        for k in 0..g.len() {
            gi[k] = g[k] * m[k];
        }
        unexpand(&gi, &sh, orig)
    });
    Ok(node(out, shape, req, vec![a.clone(), b.clone()], f, "MulBackward0"))
}

pub fn div_op(a: &Tx, b: &Tx) -> Result<Tx, String> {
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let ae = expand(&a.data, &a.shape, &shape);
    let be = expand(&b.data, &b.shape, &shape);
    let mut out = vec![0.0f32; ae.len()];
    for i in 0..out.len() {
        out[i] = ae[i] / be[i];
    }
    let req = a.requires_grad.get() || b.requires_grad.get();
    let (ae2, be2, as_, bs, sh) = (
        ae.clone(),
        be.clone(),
        a.shape.clone(),
        b.shape.clone(),
        shape.clone(),
    );
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        let mut gi = vec![0.0f32; g.len()];
        for k in 0..g.len() {
            gi[k] = if i == 0 {
                g[k] / be2[k]
            } else {
                -g[k] * ae2[k] / (be2[k] * be2[k])
            };
        }
        unexpand(&gi, &sh, if i == 0 { &as_ } else { &bs })
    });
    Ok(node(out, shape, req, vec![a.clone(), b.clone()], f, "DivBackward0"))
}

pub fn neg_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = -*v;
    }
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().map(|v| -v).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "NegBackward0")
}

pub fn abs_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = v.abs();
    }
    let s = a.data.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter()
            .zip(s.iter())
            .map(|(gv, sv)| gv * sv.signum())
            .collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "AbsBackward0")
}

pub fn relu_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    let s = a.data.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().zip(s.iter()).map(|(gv, sv)| if *sv > 0.0 { *gv } else { 0.0 }).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "ReluBackward0")
}

pub fn sigmoid_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
    let s = out.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().zip(s.iter()).map(|(gv, sv)| gv * sv * (1.0 - sv)).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "SigmoidBackward0")
}

pub fn tanh_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = v.tanh();
    }
    let s = out.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().zip(s.iter()).map(|(gv, sv)| gv * (1.0 - sv * sv)).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "TanhBackward0")
}

pub fn silu_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = silu(*v);
    }
    let s = a.data.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        // derivative of x*sigmoid(x): sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        g.iter()
            .zip(s.iter())
            .map(|(gv, &x)| {
                let sg = 1.0 / (1.0 + (-x).exp());
                gv * (sg + x * sg * (1.0 - sg))
            })
            .collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "SiluBackward0")
}

/// Локальная erf (A&S 7.1.26, погрешность ~1e-7) для точного gelu.
pub fn erf(x: f32) -> f32 {
    let sgn = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sgn * y
}

pub fn gelu_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
    }
    let s = a.data.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        let c = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
        g.iter()
            .zip(s.iter())
            .map(|(gv, &x)| {
                let phi = (-0.5 * x * x).exp() * c;
                let cdf = 0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2));
                gv * (cdf + x * phi)
            })
            .collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "GeluBackward0")
}

pub fn exp_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = v.exp();
    }
    let s = out.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().zip(s.iter()).map(|(gv, sv)| gv * sv).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "ExpBackward0")
}

pub fn log_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = v.ln();
    }
    let s = a.data.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().zip(s.iter()).map(|(gv, sv)| gv / sv).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "LogBackward0")
}

pub fn sqrt_op(a: &Tx) -> Tx {
    let mut out = a.data.clone();
    for v in out.iter_mut() {
        *v = v.sqrt();
    }
    let s = out.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.iter().zip(s.iter()).map(|(gv, sv)| gv / (2.0 * sv)).collect()
    });
    node(out, a.shape.clone(), a.requires_grad.get(), vec![a.clone()], f, "SqrtBackward0")
}

// ═══════════════════════════════════════════════════════════════════
// reductions / reshape / transpose / matmul
// ═══════════════════════════════════════════════════════════════════

pub fn sum_row_op(a: &Tx) -> Tx {
    let s: f32 = a.data.iter().sum();
    let len = a.data.len();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        vec![g[0]; len]
    });
    node(vec![s], vec![], a.requires_grad.get(), vec![a.clone()], f, "SumBackward0")
}

pub fn mean_row_op(a: &Tx) -> Tx {
    let s: f32 = a.data.iter().sum();
    let n = a.data.len() as f32;
    let len = a.data.len();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        let v = g[0] / n;
        vec![v; len]
    });
    node(vec![s / n], vec![], a.requires_grad.get(), vec![a.clone()], f, "MeanBackward0")
}

pub fn reshape_op(a: &Tx, new_shape: Vec<usize>) -> Result<Tx, String> {
    if numel(&new_shape) != a.data.len() {
        return Err(format!(
            "shape {} не совпадает с numel {}",
            new_shape.len(),
            a.data.len()
        ));
    }
    let old = a.shape.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        g.to_vec()
    });
    let _ = &old;
    Ok(node(a.data.clone(), new_shape, a.requires_grad.get(), vec![a.clone()], f, "ViewBackward0"))
}

fn transpose2d_data(data: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = data[i * n + j];
        }
    }
    out
}

pub fn transpose_op(a: &Tx) -> Result<Tx, String> {
    if a.shape.len() != 2 {
        return Err("transpose: поддерживаются только 2D-тензоры".into());
    }
    let (m, n) = (a.shape[0], a.shape[1]);
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        transpose2d_data(g, n, m)
    });
    Ok(node(
        transpose2d_data(&a.data, m, n),
        vec![n, m],
        a.requires_grad.get(),
        vec![a.clone()],
        f,
        "TransposeBackward0",
    ))
}

/// gemm2: (m,k)x(k,n) = m*n; входные данные CONTIGUOUS, A (m,k) строки, B (k,n).
/// Реализвация через транспонированную копию B и построчные матвэки роутера.
/// В режиме ATOMNET_NPU_EMU=1 — через JIT-блоки 6x16 mac-массива, 1 поток
/// (симулятор NPU: детерминировано и независимо от числа потоков).
pub fn gemm2(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    if npu_emu_on() {
        return crate::atom::flash::flash_gemm::gemm_blocked(a, m, k, b, n, 1);
    }
    let bt = transpose2d_data(b, k, n);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        let row = &a[i * k..(i + 1) * k];
        let y = crate::atom::npu::npu_matvec(&bt, n, k, row);
        out[i * n..(i + 1) * n].copy_from_slice(&y);
    }
    out
}

pub fn matmul_op(a: &Tx, b: &Tx) -> Result<Tx, String> {
    let (a2d, b2d) = match (a.shape.len(), b.shape.len()) {
        (2, 2) => (false, false),
        (1, 2) => (true, false),
        (2, 1) => (false, true),
        (1, 1) => (true, true),
        _ => return Err("matmul: поддерживаются только 1D/2D".into()),
    };
    let a_mat = if a2d { vec![1, a.shape[0]] } else { a.shape.clone() };
    let b_mat = if b2d { vec![b.shape[0], 1] } else { b.shape.clone() };
    let (m, k) = (a_mat[0], a_mat[1]);
    let (k2, n) = (b_mat[0], b_mat[1]);
    if k != k2 {
        return Err(format!("matmul: внутренние размеры не совпадают: {} vs {}", k, k2));
    }
    let out = gemm2(&a.data, m, k, &b.data, n);
    let out_shape = match (a2d, b2d) {
        (true, true) => Vec::<usize>::new(),
        (true, false) => vec![n],
        (false, true) => vec![m],
        (false, false) => vec![m, n],
    };
    let req = a.requires_grad.get() || b.requires_grad.get();
    let (a_save, b_save) = (a.data.clone(), b.data.clone());
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        // Разворачиваем grad_out в 2D-форму перед матричными произведениями.
        let gmat = g.to_vec();
        if i == 0 {
            // dA = dO * B^T : npu_matvec(w=B (k,n)-> need (k,n)?? B is (k,n): dA row len k = npu_matvec(b_save -> (k,n) weight, rows=n...)... проще: bt=b^T (n,k); dO (m,n) row @ bt (n,k) = dO_row . bt_row_j??? используем gemm2(dO_(m,n), b_save_(k,n)) с b_save^T... gemm2 оптимальна: dA = gmat(m,n) @ B^T(n,k) = gemm2(gmat, n=k_b... точно: gemm2(x, m, n, y, n, k) где y = B^T = (n,k)
            let bt = transpose2d_data(&b_save, k, n);
            let d_a = gemm2(&gmat, m, n, &bt, k);
            if a2d {
                d_a
            } else {
                d_a
            }
        } else {
            // dB = A^T(k,m) @ dO(m,n) = (n,m)... транспониуем A и скармливаем
            let at = transpose2d_data(&a_save, m, k);
            gemm2(&at, k, m, &gmat, n)
        }
    });
    Ok(node(out, out_shape, req, vec![a.clone(), b.clone()], f, "MmBackward0"))
}

pub fn unsqueeze_op(a: &Tx, dim: i64) -> Result<Tx, String> {
    let mut shape = a.shape.clone();
    let dim = if dim < 0 {
        (shape.len() as i64 + dim) as usize
    } else {
        dim as usize
    };
    if dim > shape.len() {
        return Err("unsqueeze: размерность вне диапазона".into());
    }
    shape.insert(dim, 1);
    reshape_op(a, shape)
}

pub fn squeeze_op(a: &Tx) -> Result<Tx, String> {
    let shape: Vec<usize> = a.shape.iter().cloned().filter(|&d| d != 1).collect();
    reshape_op(a, shape)
}

// ═══════════════════════════════════════════════════════════════════
// softmax / log_softmax (последняя ось)
// ═══════════════════════════════════════════════════════════════════

fn softmax_rows(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (row[c] - mx).exp();
            out[r * cols + c] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for c in 0..cols {
            out[r * cols + c] *= inv;
        }
    }
    out
}

fn log_softmax_rows(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += (row[c] - mx).exp();
        }
        let lse = mx + sum.ln();
        for c in 0..cols {
            out[r * cols + c] = row[c] - lse;
        }
    }
    out
}

pub fn softmax_op(a: &Tx) -> Result<Tx, String> {
    let shape = a.shape.clone();
    if shape.is_empty() {
        return Err("softmax: пустая форма".into());
    }
    let cols = *shape.last().unwrap();
    let rows = numel(&shape) / cols.max(1);
    let probs = softmax_rows(&a.data, rows, cols);
    let req = a.requires_grad.get();
    let saved = probs.clone();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        // dx = s * (g - sum(g*s))
        let mut out = vec![0.0f32; g.len()];
        for r in 0..rows {
            let mut ssum = 0.0f32;
            for c in 0..cols {
                ssum += g[r * cols + c] * saved[r * cols + c];
            }
            for c in 0..cols {
                let k = r * cols + c;
                out[k] = saved[k] * (g[k] - ssum);
            }
        }
        out
    });
    Ok(node(probs, shape, req, vec![a.clone()], f, "SoftmaxBackward0"))
}

// ═══════════════════════════════════════════════════════════════════
// cross_entropy (log_softmax + nll, fused — как в torch/nn/functional.py)
// targets: индексы классов (N,), вход (N,C) или (C,) -> скаляр.
// ═══════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
pub fn cross_entropy_op(
    input: &Tx,
    targets: &[i64],
    ignore_index: i64,
    weight: Option<&[f32]>,
    label_smoothing: f32,
    reduction: &str, // "mean" | "sum" | "none"
) -> Result<Tx, String> {
    let (rows, cols) = match input.shape.len() {
        1 => (1usize, input.shape[0]),
        2 => (input.shape[0], input.shape[1]),
        _ => return Err("cross_entropy: вход должен быть (C,) или (N,C)".into()),
    };
    let n_target = if input.shape.len() == 1 { 1 } else { rows };
    if targets.len() != n_target {
        return Err("cross_entropy: targets не соответствует входу".into());
    }
    let nll_only = label_smoothing <= 0.0;
    let ls = label_smoothing.clamp(0.0, 1.0);
    let logp = log_softmax_rows(&input.data, rows, cols);
    let probs = softmax_rows(&input.data, rows, cols);
    let mut losses = vec![0.0f32; rows];
    let mut denom = 0.0f32;
    let mut count = 0usize;
    let mut scales = vec![1.0f32; rows];
    for r in 0..rows {
        let yi = if targets[r] == ignore_index {
            continue;
        } else if targets[r] < 0 || targets[r] as usize >= cols {
            return Err("cross_entropy: target вне диапазона классов".into());
        } else {
            targets[r] as usize
        };
        let w = weight.map(|ws| ws[yi]).unwrap_or(1.0);
        denom += w;
        count += 1;
        let mut l = -logp[r * cols + yi];
        if !nll_only {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += -logp[r * cols + c];
            }
            let uniform = acc / cols as f32;
            l = (1.0 - ls) * l + ls * uniform;
        }
        losses[r] = l * w;
        scales[r] = w;
    }
    let (out_data, out_shape, out_req) = match reduction {
        "mean" => {
            let s: f32 = losses.iter().sum();
            let d = if count == 0 { 1.0 } else { denom };
            (vec![s / d], Vec::<usize>::new(), input.requires_grad.get())
        }
        "sum" => {
            let s: f32 = losses.iter().sum();
            (vec![s], Vec::<usize>::new(), input.requires_grad.get())
        }
        _ => (losses.clone(), vec![rows], input.requires_grad.get()),
    };
    // ignore-маска делается в момент forward через targets==ignore_index.
    let ignored: Vec<bool> = (0..rows)
        .map(|r| targets[r] == ignore_index)
        .collect();
    let saved_probs = probs.clone();
    let saved_targets = targets.to_vec();
    let saved_scales = scales.clone();
    let saved_reduction = reduction.to_string();
    let eps = 1e-12f32;
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; saved_probs.len()];
        for r in 0..rows {
            if ignored[r] {
                continue;
            }
            let yi = saved_targets[r] as usize;
            let w = saved_scales[r];
            let scaled_g = match saved_reduction.as_str() {
                "mean" => {
                    let d = if count == 0 { 1.0 } else { denom };
                    g[0] / d.max(eps)
                }
                _ => g[r],
            };
            let coeff = scaled_g * w;
            for c in 0..cols {
                let dist = if nll_only {
                    if c == yi { 1.0 } else { 0.0 }
                } else if c == yi {
                    1.0 - ls + ls / cols as f32
                } else {
                    ls / cols as f32
                };
                out[r * cols + c] = coeff * (saved_probs[r * cols + c] - dist);
            }
        }
        out
    });
    Ok(node(
        out_data,
        out_shape,
        out_req,
        vec![input.clone()],
        f,
        "CrossEntropyBackward0",
    ))
}

// ═══════════════════════════════════════════════════════════════════
// layer_norm (torch: смещённая дисперсия, eps ВНУТРИ sqrt, γ·xh+β)
// ═══════════════════════════════════════════════════════════════════

pub fn layer_norm_op(
    x: &Tx,
    normalized_shape: &[usize],
    weight: Option<&Tx>,
    bias: Option<&Tx>,
    eps: f32,
) -> Result<Tx, String> {
    let nd = x.shape.len();
    let ks = normalized_shape.len();
    if ks > nd {
        return Err("layer_norm: normalized_shape длиннее формы входа".into());
    }
    let lead = nd - ks;
    if x.shape[lead..] != *normalized_shape {
        return Err("layer_norm: normalized_shape не совпадает с хвостом формы".into());
    }
    let k: usize = normalized_shape.iter().product();
    let rows = numel(&x.shape) / k.max(1);
    let mut y = vec![0.0f32; x.data.len()];
    let mut xhat = vec![0.0f32; x.data.len()];
    for r in 0..rows {
        let row = &x.data[r * k..(r + 1) * k];
        let mean: f32 = row.iter().sum::<f32>() / k as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / k as f32;
        let rr = 1.0 / (var + eps).sqrt();
        for c in 0..k {
            let h = (row[c] - mean) * rr;
            xhat[r * k + c] = h;
            let g = weight.map(|w| w.data[c]).unwrap_or(1.0);
            let b = bias.map(|bb| bb.data[c]).unwrap_or(0.0);
            y[r * k + c] = h * g + b;
        }
    }
    let req = x.requires_grad.get() || weight.map(|w| w.requires_grad.get()).unwrap_or(false) || bias.map(|b| b.requires_grad.get()).unwrap_or(false);
    layer_norm_impl(x, normalized_shape, weight, bias, eps, rows, k, y, xhat, req)
}

/// Точная реализация (сохраняет rr для backward — корректнее выше).
#[allow(clippy::too_many_arguments)]
fn layer_norm_impl(
    x: &Tx,
    normalized_shape: &[usize],
    weight: Option<&Tx>,
    bias: Option<&Tx>,
    eps: f32,
    rows: usize,
    k: usize,
    y: Vec<f32>,
    xhat: Vec<f32>,
    req: bool,
) -> Result<Tx, String> {
    // rr per row
    let mut rrs = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x.data[r * k..(r + 1) * k];
        let mean: f32 = row.iter().sum::<f32>() / k as f32;
        let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / k as f32;
        rrs[r] = 1.0 / (var + eps).sqrt();
    }
    let gamma = weight.map(|w| w.data.clone()).unwrap_or_else(|| vec![1.0; k]);
    let mut inputs = vec![x.clone()];
    if let Some(w) = weight {
        inputs.push(w.clone());
    }
    if let Some(b) = bias {
        inputs.push(b.clone());
    }
    let n_inputs = inputs.len();
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        match i {
            0 => {
                let mut out = vec![0.0f32; g.len()];
                for r in 0..rows {
                    let rr = rrs[r];
                    // s = g*gamma; mean_s; mean_sx = mean(s*xhat)
                    let mut mean_s = 0.0f32;
                    let mut mean_sx = 0.0f32;
                    for c in 0..k {
                        let s = g[r * k + c] * gamma[c];
                        mean_s += s;
                        mean_sx += s * xhat[r * k + c];
                    }
                    mean_s /= k as f32;
                    mean_sx /= k as f32;
                    for c in 0..k {
                        let s = g[r * k + c] * gamma[c];
                        out[r * k + c] = rr * (s - mean_s - xhat[r * k + c] * mean_sx);
                    }
                }
                out
            }
            j if n_inputs > 2 && j == 1 => {
                let mut gg = vec![0.0f32; k];
                for r in 0..rows {
                    for c in 0..k {
                        gg[c] += g[r * k + c] * xhat[r * k + c];
                    }
                }
                gg
            }
            _ => {
                // bias grad (индекс: если weight None, то bias идёт как 1)
                let mut gb = vec![0.0f32; k];
                for r in 0..rows {
                    for c in 0..k {
                        gb[c] += g[r * k + c];
                    }
                }
                gb
            }
        }
    });
    Ok(node(y, x.shape.clone(), req, inputs, f, "NativeLayerNormBackward0"))
}

// ═══════════════════════════════════════════════════════════════════
// embedding (lookup, backward scatter-add)
// ═══════════════════════════════════════════════════════════════════

pub fn embedding_op(indices: &[usize], weight: &Tx) -> Result<Tx, String> {
    let (v, e) = match weight.shape.len() {
        2 => (weight.shape[0], weight.shape[1]),
        _ => return Err("embedding: weight должен быть 2D (V,E)".into()),
    };
    for &idx in indices {
        if idx >= v {
            return Err("embedding: индекс вне словаря".into());
        }
    }
    let mut out = vec![0.0f32; indices.len() * e];
    for (ri, &idx) in indices.iter().enumerate() {
        out[ri * e..(ri + 1) * e].copy_from_slice(&weight.data[idx * e..(idx + 1) * e]);
    }
    let req = weight.requires_grad.get();
    let idxs = indices.to_vec();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        let mut gw = vec![0.0f32; v * e];
        for (ri, &idx) in idxs.iter().enumerate() {
            for c in 0..e {
                gw[idx * e + c] += g[ri * e + c];
            }
        }
        gw
    });
    Ok(node(out, vec![indices.len(), e], req, vec![weight.clone()], f, "EmbeddingBackward0"))
}

// ═══════════════════════════════════════════════════════════════════
// mse_loss (mean of squares — как torch.nn.MSELoss reduction='mean')
// ═══════════════════════════════════════════════════════════════════

pub fn mse_loss_op(a: &Tx, b: &Tx) -> Result<Tx, String> {
    let shape = broadcast_shape(&a.shape, &b.shape)?;
    let ae = expand(&a.data, &a.shape, &shape);
    let be = expand(&b.data, &b.shape, &shape);
    let n = ae.len() as f32;
    let mut s = 0.0f32;
    for i in 0..ae.len() {
        let d = ae[i] - be[i];
        s += d * d;
    }
    let loss = s / n;
    let (as_, bs, sh) = (a.shape.clone(), b.shape.clone(), shape.clone());
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        let mut gi = vec![0.0f32; ae.len()];
        for k in 0..ae.len() {
            let d = ae[k] - be[k];
            gi[k] = g[0] * (2.0 / n) * if i == 0 { d } else { -d };
        }
        unexpand(&gi, &sh, if i == 0 { &as_ } else { &bs })
    });
    let req = a.requires_grad.get() || b.requires_grad.get();
    Ok(node(vec![loss], vec![], req, vec![a.clone(), b.clone()], f, "MseBackward0"))
}

// ═══════════════════════════════════════════════════════════════════
// factories (детерминирована manual_seed)
// ═══════════════════════════════════════════════════════════════════

pub fn factory_rand(shape: Vec<usize>) -> Tx {
    let n = numel(&shape);
    let mut d = vec![0.0f32; n];
    for v in d.iter_mut() {
        *v = next_rand();
    }
    tx(d, shape, false)
}

pub fn factory_randn(shape: Vec<usize>) -> Tx {
    let n = numel(&shape);
    let mut d = vec![0.0f32; n];
    for v in d.iter_mut() {
        *v = next_randn();
    }
    tx(d, shape, false)
}

pub fn factory_randint(low: i64, high: i64, shape: Vec<usize>) -> Tx {
    let n = numel(&shape);
    let span = (high - low).max(1) as f64;
    let mut d = vec![0.0f32; n];
    for v in d.iter_mut() {
        *v = (low as f64 + (next_rand() as f64) * span) as i64 as f32;
    }
    tx(d, shape, false)
}

pub fn factory_zeros(shape: Vec<usize>) -> Tx {
    tx(vec![0.0; numel(&shape)], shape, false)
}

pub fn factory_ones(shape: Vec<usize>) -> Tx {
    tx(vec![1.0; numel(&shape)], shape, false)
}

pub fn factory_full(shape: Vec<usize>, val: f32) -> Tx {
    tx(vec![val; numel(&shape)], shape, false)
}

pub fn factory_arange(n: usize) -> Tx {
    let mut d = vec![0.0f32; n];
    for (i, v) in d.iter_mut().enumerate() {
        *v = i as f32;
    }
    tx(d, vec![n], false)
}

/// In-place мутация данных тензора (Tensor.add_/fill_/copy_/zero_ в PyTorch).
/// Вызывающий гарантирует: на этот тензор не висит активных autograd-цепей
/// (как в torch, in-place на leaf с requires_grad запрещён — проверка снаружи).
pub fn mutate_data(t: &Tx, f: impl FnOnce(&mut Vec<f32>)) {
    let ptr = Rc::as_ptr(t) as *mut TxNode;
    let d = unsafe { &mut (*ptr).data };
    f(d);
}

/// simd convenience re-export for callers in this crate.
#[allow(dead_code)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    simd::dot(a, b)
}

// ═══════════════════════════════════════════════════════════════════
// rms_norm (torch: y = x/RMS(x)*γ; eps -> opmath_eps, default здесь 1e-6)
// ═══════════════════════════════════════════════════════════════════

pub fn rms_norm_op(
    x: &Tx,
    normalized_shape: &[usize],
    weight: Option<&Tx>,
    eps: f32,
) -> Result<Tx, String> {
    let nd = x.shape.len();
    let ks = normalized_shape.len();
    if ks > nd {
        return Err("rms_norm: normalized_shape длиннее формы входа".into());
    }
    let lead = nd - ks;
    if x.shape[lead..] != *normalized_shape {
        return Err("rms_norm: normalized_shape не совпадает с хвостом формы".into());
    }
    let k: usize = normalized_shape.iter().product();
    let rows = numel(&x.shape) / k.max(1);
    let mut y = vec![0.0f32; x.data.len()];
    let mut rrs = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x.data[r * k..(r + 1) * k];
        let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / k as f32;
        let rr = 1.0 / (ms + eps).sqrt();
        rrs[r] = rr;
        for c in 0..k {
            let g = weight.map(|w| w.data[c]).unwrap_or(1.0);
            y[r * k + c] = row[c] * rr * g;
        }
    }
    let req = (x.requires_grad.get() || weight.map(|w| w.requires_grad.get()).unwrap_or(false)) && grad_enabled();
    let gamma = weight.map(|w| w.data.clone()).unwrap_or_else(|| vec![1.0; k]);
    let xs = x.data.clone();
    let f = Rc::new(move |g: &[f32], i: usize| -> Vec<f32> {
        match i {
            0 => {
                // dx_i = gamma_i*r*(dy_i - x_i * mean(x * gamma * dy) * r^2)
                let mut out = vec![0.0f32; g.len()];
                for r in 0..rows {
                    let rr = rrs[r];
                    let mut acc = 0.0f32;
                    for c in 0..k {
                        acc += xs[r * k + c] * gamma[c] * g[r * k + c];
                    }
                    let coeff = (acc / k as f32) * rr * rr;
                    for c in 0..k {
                        out[r * k + c] = gamma[c] * rr * (g[r * k + c] - xs[r * k + c] * coeff);
                    }
                }
                out
            }
            _ => {
                let mut gg = vec![0.0f32; k];
                for r in 0..rows {
                    for c in 0..k {
                        gg[c] += g[r * k + c] * xs[r * k + c] * rrs[r];
                    }
                }
                gg
            }
        }
    });
    let mut inputs = vec![x.clone()];
    if let Some(w) = weight {
        inputs.push(w.clone());
    }
    Ok(node(y, x.shape.clone(), req, inputs, f, "NativeRMSNormBackward0"))
}

// ═══════════════════════════════════════════════════════════════════
// rows_slice: строки [start..stop] первой оси (narrow по dim0)
// ═══════════════════════════════════════════════════════════════════

pub fn rows_slice_op(a: &Tx, start: usize, stop: usize) -> Result<Tx, String> {
    if a.shape.is_empty() {
        return Err("rows_slice: пустая форма".into());
    }
    let rows = a.shape[0];
    if start >= stop || stop > rows {
        return Err(format!("rows_slice: диапазон [{start},{stop}) вне [0,{rows})"));
    }
    let row_sz = a.data.len() / rows;
    let mut out = vec![0.0f32; (stop - start) * row_sz];
    out.copy_from_slice(&a.data[start * row_sz..stop * row_sz]);
    let new_shape = {
        let mut s = a.shape.clone();
        s[0] = stop - start;
        s
    };
    let orig_len = a.data.len();
    let f = Rc::new(move |g: &[f32], _i: usize| -> Vec<f32> {
        let mut full = vec![0.0f32; orig_len];
        full[start * row_sz..stop * row_sz].copy_from_slice(g);
        full
    });
    Ok(node(out, new_shape, a.requires_grad.get(), vec![a.clone()], f, "NarrowBackward0"))
}
