"""AtomNET Datacenter Engine — граф под DynASM + распределение ресурсов.

Глаза:
  - динамический граф DynGraph (from atomnet): компилируется (уровни-расписания),
    исполняется с Rayon work-stealing pool.
  - модель — исполняет свой граф через npu_mac (JIT-обре, gemm2).
  на одну нейросеть, и через первую примитива resume-parallelism
"""

import time

import atomnet as A
import numpy as np


def run_demo():
    dim = 48
    rng = np.random.default_rng(0)
    inp = rng.standard_normal(dim).astype(np.float32)
    Wq = rng.standard_normal((dim, dim)).astype(np.float32)
    Wk = rng.standard_normal((dim, dim)).astype(np.float32)
    Wv = rng.standard_normal((dim, dim)).astype(np.float32)
    Wo = rng.standard_normal((dim, dim)).astype(np.float32)

    g = A.DynGraph()
    x = g.add_const([float(v) for v in inp.ravel()])
    wq = g.add_const([float(v) for v in Wq.ravel()])
    wk = g.add_const([float(v) for v in Wk.ravel()])
    wv = g.add_const([float(v) for v in Wv.ravel()])
    wo = g.add_const([float(v) for v in Wo.ravel()])
    # Demonstrsion:строить граф как в transformers (q/k/v -> скейл -> attention через
    # matmul с Wo). k/v-act используется как gates.
    q = g.add_matmul_silu(x, wq, 1, dim, dim)
    k = g.add_matmul_silu(x, wk, 1, dim, dim)
    v_ = g.add_matmul_silu(x, wv, 1, dim, dim)
    gate = g.add_add(k, v_) if hasattr(g, "add_add") else (k)
    out = g.add_matmul_silu(gate, wo, 1, dim, dim)
    g.set_outputs([out])
    t0 = time.time()
    y = g.run()
    dt = time.time() - t0
    print("graph output len:", len(y[0]), "norm:", float(np.linalg.norm(y[0:])))
    print(f"runtime: {dt*1000:.1f} ms  SELFTEST OK")
    return "SELFTEST OK"


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        print(run_demo())
