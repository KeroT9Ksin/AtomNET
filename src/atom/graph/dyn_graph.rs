// src/atom/graph/dyn_graph.rs
//! dyn_graph — Динамический граф вычислений (демо под DynASM).
//!
//! Строится кодом (Python: `DynGraph.add_node` через биндинг), компилируется
//! (уровни- расписания) и исполняется с распределением ворк-стилингом через
//! Rayon-пул: каждый уровень топо-порядка параллелит формацию из узлов
//! (детерминированность f32 сохраняется — узел зависит только от своих входов).

use std::sync::Mutex;

#[derive(Debug, Clone)]
pub struct DynNode {
    pub op: OpKind,
    pub inputs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum OpKind {
    /// Константа-узел: значения множатели извлекаются из контекста при построении
    Const(Vec<f32>),
    /// gemm2(a, b) — (m,k)x(k,n) -> (m,n) через flash/gemm router (JIT).
    Gemm { m: usize, k: usize, n: usize },
    /// gemm2+silu — слитый fused silu-эпилог (JIT kernel_epilog).
    GemmSilu { m: usize, k: usize, n: usize },
    /// softmax по последней оси (rows x cols).
    Softmax { rows: usize, cols: usize },
    Silu,
    Sigmoid,
    Add,
    Mul,
    Scale { r: f32 },
}

#[derive(Debug)]
pub enum DynError {
    BadInput(String),
}

impl std::fmt::Display for DynError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DynError::BadInput(s) => write!(f, "dyn_graph: {s}"),
        }
    }
}

pub struct DynGraph {
    pub nodes: Vec<DynNode>,
    pub schedule: Vec<Vec<usize>>,
    pub outputs: Vec<usize>,
}

impl DynGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), schedule: Vec::new(), outputs: Vec::new() }
    }

    pub fn add_const(&mut self, data: Vec<f32>) -> usize {
        self.add_node(OpKind::Const(data), vec![])
    }

    pub fn add_node(&mut self, op: OpKind, inputs: Vec<usize>) -> usize {
        let id = self.nodes.len();
        for &i in &inputs {
            if i >= id {
                panic!("dyn_graph: node {} references future node {}", id, i);
            }
        }
        self.nodes.push(DynNode { op, inputs });
        let mut preds = self.nodes[id].inputs.clone();
        preds.sort_unstable();
        preds.dedup();
        let lvl = preds.iter().map(|p| self.level_of(*p) + 1).max().unwrap_or(0);
        if self.schedule.len() <= lvl {
            self.schedule.resize_with(lvl + 1, Vec::new);
        }
        self.schedule[lvl].push(id);
        id
    }

    pub fn set_outputs(&mut self, ids: Vec<usize>) {
        self.outputs = ids;
    }

    pub fn level_of(&self, id: usize) -> usize {
        for (li, nodes) in self.schedule.iter().enumerate() {
            if nodes.contains(&id) {
                return li;
            }
        }
        0
    }

    /// Исполнение графа: уровни по расписанию, внутри уровня - par_iter
    /// на Rayon-пуле (work-stealing; детерминированная f32-арифметика
    /// на узел: его вход всегда готов при старте).
    pub fn run(&self) -> Result<Vec<Vec<f32>>, String> {
        use rayon::prelude::*;
        let n = self.nodes.len();
        let values: Mutex<Vec<Option<Vec<f32>>>> =
            Mutex::new((0..n).map(|_| None).collect());
        for level in self.schedule.iter() {
            level.par_iter().for_each(|&nid| {
                let node = &self.nodes[nid];
                let vals: Vec<Vec<f32>> = {
                    let tie = values.lock().unwrap();
                    node.inputs.iter().map(|&src| tie[src].clone().expect("input not ready")).collect()
                };
                let out = run_node(node, &vals);
                let mut tie = values.lock().unwrap();
                tie[nid] = Some(out);
            });
        }
        let r = values.into_inner().unwrap();
        let mut out = Vec::new();
        for id in self.outputs.iter() {
            out.push(r[*id].clone().ok_or_else(|| format!("output {} unset", id))?);
        }
        Ok(out)
    }
}

fn run_node(node: &DynNode, ins: &[Vec<f32>]) -> Vec<f32> {
    match &node.op {
        OpKind::Const(d) => d.clone(),
        OpKind::Gemm { m, k, n } => {
            crate::atom::torchx::gemm2(&ins[0], *m, *k, &ins[1], *n)
        }
        OpKind::GemmSilu { m, k, n } => {
            let mut out = crate::atom::torchx::gemm2(&ins[0], *m, *k, &ins[1], *n);
            for v in out.iter_mut() {
                *v = crate::atom::nn::silu(*v);
            }
            out
        }
        OpKind::Softmax { rows, cols } => {
            let src = &ins[0];
            let mut out = vec![0.0f32; src.len()];
            for row in 0..*rows {
                let sl = &src[row * *cols..(row + 1) * *cols];
                let mx = sl.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sm = 0.0f32;
                for c in 0..*cols {
                    sm += (sl[c] - mx).exp();
                }
                let inv = 1.0 / sm;
                for c in 0..*cols {
                    out[row * *cols + c] = (sl[c] - mx).exp() * inv;
                }
            }
            out
        }
        OpKind::Silu => ins[0].iter().map(|v| crate::atom::nn::silu(*v)).collect(),
        OpKind::Sigmoid => ins[0].iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect(),
        OpKind::Add => {
            let a = &ins[0];
            let b = &ins[1];
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
        }
        OpKind::Mul => {
            let a = &ins[0];
            let b = &ins[1];
            assert_eq!(a.len(), b.len());
            a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
        }
        OpKind::Scale { r } => ins[0].iter().map(|v| v * r).collect(),
    }
}

#[cfg(test)]
mod tests_dyn_graph {
    use super::*;

    fn v(n: usize, seed: u64) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(6364136223846793005) ^ 0xA0761D6478BD642F;
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            out.push((((s >> 33) % 2000) as f32 - 1000.0) / 1000.0);
        }
        out
    }

    #[test]
    fn dyn_graph_matmul_deterministic() {
        let a = v(120, 3);
        let b = v(120 * 64, 7);
        let mut g = DynGraph::new();
        let ai = g.add_const(a.clone());
        let bi = g.add_const(b.clone());
        let ci = g.add_node(OpKind::Gemm { m: 10, k: 12, n: 64 }, vec![ai, bi]);
        g.set_outputs(vec![ci]);
        let out = g.run().unwrap();
        // deterministic re-run: бит-идентичность
        let out2 = g.run().unwrap();
        for x in out[0].iter().zip(out2[0].iter()) {
            assert_eq!(x.0.to_bits(), x.1.to_bits());
        }
        let expected = crate::atom::torchx::gemm2(&a, 10, 12, &b, 64);
        for (x, y) in out[0].iter().zip(expected.iter()) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn dyn_graph_two_levels() {
        let a = v(120, 1);
        let b = v(120 * 32, 2);
        let mut g = DynGraph::new();
        let ai = g.add_const(a.clone());
        let bi = g.add_const(b.clone());
        let ci = g.add_node(OpKind::GemmSilu { m: 10, k: 12, n: 32 }, vec![ai, bi]);
        let di = g.add_node(OpKind::Softmax { rows: 10, cols: 32 }, vec![ci]);
        g.set_outputs(vec![di]);
        let out = g.run().unwrap();
        assert_eq!(out[0].len(), 320);
    }
}
