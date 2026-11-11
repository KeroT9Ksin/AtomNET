# tests/test_keralin_heavy.py
# Тяжёлый тест: KeralinGraph vs AdamW на 3-слойной MLP (~400K параметров).
# Сходимость, стабильность, время; график + CSV + JSON.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import csv
import json
import time

import numpy as np
import atomnet

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

KG = getattr(atomnet, "PyKeralinGraph", None) or atomnet.optim.PyKeralinGraph

OUT_PNG = "keralin_heavy_graph.png"
OUT_CSV = "keralin_heavy_records.csv"
OUT_JSON = "keralin_heavy_metrics.json"

# ── Модель: D → H1 → H2 → O (3 слоя tanh, без bias) ──
N, D, H1, H2, O = 1024, 384, 512, 384, 32
EPOCHS = 400
LR0 = 0.01
LR_END = 0.01 * 0.05  # косинус-распад до 5%
SEED = 17

as_f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)


def tanh(x):
    return np.tanh(x)


def cos_lr(t, T):
    frac = (1.0 + np.cos(np.pi * min(t, T) / T)) / 2.0
    return LR_END + (LR0 - LR_END) * frac


def build_teacher():
    np.random.seed(SEED)
    X = (np.random.randn(N, D) * 0.5).astype(np.float32)
    Wt1 = (np.random.randn(H1, D) / np.sqrt(D)).astype(np.float32)
    Wt2 = (np.random.randn(H2, H1) / np.sqrt(H1)).astype(np.float32)
    Wt3 = (np.random.randn(O, H2) / np.sqrt(H2)).astype(np.float32)
    h1 = tanh(X @ Wt1.T)
    h2 = tanh(h1 @ Wt2.T)
    T = (h2 @ Wt3.T).astype(np.float32)
    np.random.seed(SEED + 1)
    W1_0 = (np.random.randn(H1, D) * 0.1).astype(np.float32)
    W2_0 = (np.random.randn(H2, H1) * 0.1).astype(np.float32)
    W3_0 = (np.random.randn(O, H2) * 0.1).astype(np.float32)
    return X, T, W1_0, W2_0, W3_0


def fwd(W1, W2, W3, X):
    h1 = tanh(X @ W1.T)
    h2 = tanh(h1 @ W2.T)
    y = h2 @ W3.T
    return h1, h2, y


def grads(X, T, W1, W2, W3):
    """Полный батч: аналитический backward, возвращает dW1..dW3."""
    Nn = X.shape[0]
    h1, h2, y = fwd(W1, W2, W3, X)
    loss = float(np.mean((y - T) ** 2))
    dy = (2.0 * (y - T) / Nn).astype(np.float32)
    dW3 = (dy.T @ h2).astype(np.float32)
    dh2 = dy @ W3
    dpre2 = (dh2 * (1.0 - h2**2)).astype(np.float32)
    dW2 = (dpre2.T @ h1).astype(np.float32)
    dh1 = dpre2 @ W2
    dpre1 = (dh1 * (1.0 - h1**2)).astype(np.float32)
    dW1 = (dpre1.T @ X).astype(np.float32)
    return loss, dW1, dW2, dW3


def build_graph(W1, W2, W3):
    g = KG()
    g.configure(lr=LR0, wd=0.0, beta1=0.9, warmup_noise=1e-3,
                auto_warm=True, adapt=True, beta2=0.99, eps=1e-8)
    ids1 = [g.add_node(as_f32(W1[i]), []) for i in range(H1)]
    ids2 = [g.add_node(as_f32(W2[i]), ids1) for i in range(H2)]
    ids3 = [g.add_node(as_f32(W3[i]), ids2) for i in range(O)]
    g.cold_start()
    return g, ids1, ids2, ids3


def keralin_epochs(X, T, W1_0, W2_0, W3_0):
    g, ids1, ids2, ids3 = build_graph(W1_0, W2_0, W3_0)
    losses = []
    times = []
    t0 = time.perf_counter()
    for t in range(1, EPOCHS + 1):
        lr = cos_lr(t, EPOCHS)
        g.configure(lr=lr, wd=0.0, beta1=0.9, warmup_noise=1e-3,
                    auto_warm=True, adapt=True, beta2=0.99, eps=1e-8)
        W1 = np.asarray(g.reconstruct_many_np(ids1), dtype=np.float32)
        W2 = np.asarray(g.reconstruct_many_np(ids2), dtype=np.float32)
        W3 = np.asarray(g.reconstruct_many_np(ids3), dtype=np.float32)
        loss, dW1, dW2, dW3 = grads(X, T, W1, W2, W3)
        losses.append(loss)
        g.feed_grads_np(ids1, dW1)
        g.feed_grads_np(ids2, dW2)
        g.feed_grads_np(ids3, dW3)
        g.step()
        times.append(time.perf_counter() - t0)
    wall = time.perf_counter() - t0
    stats = {
        "final_coherence": float(g.global_coherence),
        "integrity_ok": g.integrity()[0],
        "dead": g.integrity()[1],
        "alive_counts": g.alive_counts(),
        "steps": int(g.step_count),
        "warmed": int(g.warmed),
        "wall_s": round(wall, 2),
    }
    return losses, stats


def adamw_epochs(X, T, W1_0, W2_0, W3_0):
    W1 = W1_0.copy()
    W2 = W2_0.copy()
    W3 = W3_0.copy()
    m1, v1 = np.zeros_like(W1), np.zeros_like(W1)
    m2, v2 = np.zeros_like(W2), np.zeros_like(W2)
    m3, v3 = np.zeros_like(W3), np.zeros_like(W3)
    b1, b2, eps = 0.9, 0.999, 1e-8
    losses = []
    t0 = time.perf_counter()
    for t in range(1, EPOCHS + 1):
        lr = cos_lr(t, EPOCHS)
        loss, dW1, dW2, dW3 = grads(X, T, W1, W2, W3)
        losses.append(loss)
        for m, v, g, w in ((m1, v1, dW1, W1), (m2, v2, dW2, W2), (m3, v3, dW3, W3)):
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1**t)
            vh = v / (1 - b2**t)
            w -= lr * mh / (np.sqrt(vh) + eps)
    wall = time.perf_counter() - t0
    return losses, {"wall_s": round(wall, 2)}


def epochs_to(losses, target):
    for i, l in enumerate(losses):
        if l < target:
            return i + 1
    return None


def tail_stats(losses, frac=0.2):
    tail = losses[int(len(losses) * (1 - frac)):]
    mean = float(np.mean(tail))
    std = float(np.std(tail))
    return mean, std, std / max(mean, 1e-12)


def main():
    print("=== KERALIN HEAVY TEST: KeralinGraph vs AdamW ===")
    print(f"model: {N}x{D} -> {H1} -> {H2} -> {O}  "
          f"params={N*0 + H1*D + H2*H1 + O*H2/1e6:.2f}M  epochs={EPOCHS}")
    print(f"matplotlib: {'yes' if HAS_PLOT else 'no'}")

    X, T, W1_0, W2_0, W3_0 = build_teacher()
    loss0, _, _, _ = grads(X, T, W1_0, W2_0, W3_0)
    print(f"initial loss: {loss0:.6f}\n")

    print("[1] KeralinGraph (Rust)...")
    losses_k, stats_k = keralin_epochs(X, T, W1_0, W2_0, W3_0)
    print(f"    loss {loss0:.6f} -> {losses_k[-1]:.6f}  "
          f"wall {stats_k['wall_s']} s  coherence {stats_k['final_coherence']:.3f}")
    print(f"    integrity ok={stats_k['integrity_ok']}, dead={stats_k['dead']}, "
          f"alive_counts={stats_k['alive_counts']}")

    print("[2] AdamW (NumPy)...")
    losses_a, stats_a = adamw_epochs(X, T, W1_0, W2_0, W3_0)
    print(f"    loss {loss0:.6f} -> {losses_a[-1]:.6f}  wall {stats_a['wall_s']} s")

    k_final, k_tail_std, k_cv = tail_stats(losses_k)
    a_final, a_tail_std, a_cv = tail_stats(losses_a)
    e_k = epochs_to(losses_k, 1e-3)
    e_a = epochs_to(losses_a, 1e-3)
    e_k5 = epochs_to(losses_k, 5e-4)
    e_a5 = epochs_to(losses_a, 5e-4)

    print("\n[3] Metrics ===")
    print(f"final loss      : keralin {losses_k[-1]:.6f} | adamw {losses_a[-1]:.6f}")
    print(f"tail mean (20%) : keralin {k_final:.6f} | adamw {a_final:.6f}")
    print(f"tail CV         : keralin {k_cv:.4f} | adamw {a_cv:.4f}   (стабильность)")
    print(f"epochs to 1e-3  : keralin {e_k} | adamw {e_a}")
    print(f"epochs to 5e-4  : keralin {e_k5} | adamw {e_a5}")
    print(f"wall            : keralin {stats_k['wall_s']} s | adamw {stats_a['wall_s']} s")

    metrics = {
        "loss0": loss0,
        "final": {"keralin": losses_k[-1], "adamw": losses_a[-1]},
        "tail_mean": {"keralin": k_final, "adamw": a_final},
        "tail_cv": {"keralin": k_cv, "adamw": a_cv},
        "epochs_to_1e-3": {"keralin": e_k, "adamw": e_a},
        "epochs_to_5e-4": {"keralin": e_k5, "adamw": e_a5},
        "keralin_stats": stats_k,
        "adamw_stats": stats_a,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nJSON: {OUT_JSON}")

    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["epoch", "keralin_loss", "adamw_loss"])
        for i in range(EPOCHS):
            wtr.writerow([i + 1, losses_k[i], losses_a[i]])
    print(f"CSV: {OUT_CSV}")

    if HAS_PLOT:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        ax = axes[0]
        ax.plot(range(1, EPOCHS + 1), losses_k, label="KeralinGraph", lw=1.6)
        ax.plot(range(1, EPOCHS + 1), losses_a, label="AdamW (numpy)", lw=1.6)
        ax.axhline(1e-3, color="gray", ls="--", lw=0.8, label="target 1e-3")
        for lab, e in (("keralin", e_k), ("adamw", e_a)):
            if e:
                ax.axvline(e, ls=":", lw=0.8, alpha=0.6)
                ax.annotate(f"{lab}: ep {e}", (e, 1e-3), xytext=(e + 5, 1e-3 * 1.4),
                            fontsize=8)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (log)")
        ax.set_title("KeralinGraph vs AdamW: convergence")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(range(1, EPOCHS + 1), losses_k, label="KeralinGraph", lw=1.4)
        ax.plot(range(1, EPOCHS + 1), losses_a, label="AdamW (numpy)", lw=1.4)
        tail_from = int(EPOCHS * 0.8)
        ax.axvspan(tail_from, EPOCHS, color="green", alpha=0.08, label="tail (stability)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (linear)")
        ax.set_title(f"Stability: tail CV keralin={k_cv:.4f}, adamw={a_cv:.4f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=140)
        print(f"PNG: {OUT_PNG}")

    # ═══ Критерии ═══
    print("\n[4] Asserts ===")
    assert np.isfinite(losses_k[-1]), "keralin diverged"
    assert np.isfinite(losses_a[-1]), "adamw diverged"
    assert losses_k[-1] < 0.05 * loss0, (
        f"keralin не сошёлся: {losses_k[-1]} vs 5% старта {0.05 * loss0}"
    )
    assert k_cv < 1.0, f"keralin нестабилен в хвосте: CV={k_cv:.3f}"
    assert a_cv < 1.0, f"adamw нестабилен в хвосте: CV={a_cv:.3f}"
    assert stats_k["integrity_ok"] and stats_k["dead"] == 0, "keralin: мёртвые узлы"
    assert losses_k[-1] < 5e-3, f"keralin: абсолютное качество {losses_k[-1]}"
    if e_k is not None and e_a is not None:
        assert e_k <= 2 * e_a + 20, (
            f"keralin сильно медленнее по сходимости: {e_k} vs adamw {e_a}"
        )
    print(f"keralin final {losses_k[-1]:.6f} < adamw final {losses_a[-1]:.6f} "
          f"* 100 + 2e-4: {losses_k[-1] < 100 * losses_a[-1] + 2e-4}")
    assert losses_k[-1] < 100 * losses_a[-1] + 2e-4, "keralin сильно хуже adamw"

    print("\n✅ KERALIN HEAVY TEST PASSED")


if __name__ == "__main__":
    main()
