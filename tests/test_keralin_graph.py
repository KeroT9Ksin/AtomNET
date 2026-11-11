# tests/test_keralin_graph.py
# Keralin Graph Optimizer: XOR cold start, topo-порядок, циклы, обучение vs AdamW.
import time

import numpy as np
import atomnet

KG = getattr(atomnet, "PyKeralinGraph", None) or atomnet.optim.PyKeralinGraph


def as_f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def add_row_nodes(graph, W, deps=None):
    """Каждая строка матрицы W становится отдельным узлом-атомом."""
    deps = deps if deps is not None else []
    ids = []
    for row in W:
        ids.append(graph.add_node(as_f32(row), deps))
    return ids


def collect_rows(graph, ids):
    rows = [np.asarray(graph.reconstruct_np(i), dtype=np.float32) for i in ids]
    return np.stack(rows)


def test_structure_and_cold_start():
    print("=== 1. Структура графа + XOR cold start ===")
    np.random.seed(1)

    g = KG()
    D, H = 16, 8
    W = (np.random.randn(H, D) * 0.02).astype(np.float32)

    ids = add_row_nodes(g, W)          # независимые входы
    out = g.add_node(as_f32(np.random.randn(D) * 0.02), ids)   # зависит от всех

    assert g.node_count == H + 1
    assert len(set(g.topo())) == H + 1
    topo = g.topo()
    assert topo[-1] == out, f"выходной узел должен обновляться последним: {topo}"

    # Случайная инициализация уже «жива» (случайное направление = живая подпись)
    ok, dead, cold = g.integrity()
    print(f"случайные узлы: ok={ok}, dead={dead}, cold={cold}")
    assert ok and dead == 0

    # Нулевой/мёртвый узел: подпись 0 -> требует XOR cold start
    zero_id = g.add_node(as_f32(np.zeros(D, dtype=np.float32)), ids)
    ok, dead, cold = g.integrity()
    print(f"после нулевого узла: ok={ok}, dead={dead}, cold={cold}")
    assert not ok and dead + cold > 0
    assert not g.node_alive(zero_id)

    n = g.cold_start()
    print(f"cold_start прогревов: {n} (нулевой узел должен прогреться)")
    assert n >= 1 and g.warmed >= 1

    ok, dead, cold = g.integrity()
    st = g.node_state(zero_id)
    print(f"после cold_start: ok={ok}, dead={dead}, cold={cold}, "
          f"zero_state={st}, zero_alive={g.node_alive(zero_id)}, "
          f"sig={g.graph_signature:016x}")
    assert ok and dead == 0 and cold == 0
    assert g.node_alive(zero_id) and g.node_radius(zero_id) > 0
    assert st in ("born", "alive", "stable")
    assert g.graph_signature != 0

    # Шаг без градиентов не должен ничего ломать
    c = g.step()
    ok, dead, cold = g.integrity()
    assert ok and dead == 0
    print(f"step без градиентов: coherence={c:.4f}, ok={ok}")
    print("✅ structure/cold start passed\n")


def test_cycle_detection():
    print("=== 2. Детекция циклов ===")

    g = KG()
    a = g.add_node(as_f32(np.random.randn(16).astype(np.float32) * 0.02), [])
    b = g.add_node(as_f32(np.random.randn(16).astype(np.float32) * 0.02), [a])
    assert not g.has_cycle(), "простая цепь не должна иметь цикл"

    g.connect(b, a)  # назад: a -> b -> a = цикл
    assert g.has_cycle(), "обратное ребро должно дать цикл"

    # Отдельный ацикличный merge-граф: 0,1 -> 2 -> 3
    g2 = KG()
    n0 = g2.add_node(as_f32(np.random.randn(16).astype(np.float32) * 0.02), [])
    n1 = g2.add_node(as_f32(np.random.randn(16).astype(np.float32) * 0.02), [])
    n2 = g2.add_node(as_f32(np.random.randn(16).astype(np.float32) * 0.02), [n0, n1])
    n3 = g2.add_node(as_f32(np.random.randn(16).astype(np.float32) * 0.02), [n2])
    topo = g2.topo()
    assert not g2.has_cycle()
    assert topo.index(n0) < topo.index(n2) < topo.index(n3)
    assert topo.index(n1) < topo.index(n2)
    assert topo[-1] == n3
    print(f"topo merge-графа: {topo}")
    print("✅ cycle detection passed\n")


def train_step(X, T, W1, W2, H):
    """Полный батч: forward + аналитический backward. Возвращает (loss, dW1, dW2)."""
    N = X.shape[0]
    Hq = np.tanh(X @ W1.T)
    Yq = Hq @ W2.T
    loss = float(np.mean((Yq - T) ** 2))
    dY = (2.0 * (Yq - T) / N).astype(np.float32)
    dW2 = (dY.T @ Hq).astype(np.float32)
    dH = dY @ W2
    dPre = (dH * (1.0 - Hq**2)).astype(np.float32)
    dW1 = (dPre.T @ X).astype(np.float32)
    return loss, dW1, dW2


def adamw_step(W1, W2, m1, v1, m2, v2, dW1, dW2, lr, b1, b2, eps, t):
    g1 = dW1
    g2 = dW2
    m1 = b1 * m1 + (1 - b1) * g1
    v1 = b2 * v1 + (1 - b2) * g1 * g1
    m2 = b1 * m2 + (1 - b1) * g2
    v2 = b2 * v2 + (1 - b2) * g2 * g2
    m1h = m1 / (1 - b1**t)
    v1h = v1 / (1 - b2**t)
    m2h = m2 / (1 - b1**t)
    v2h = v2 / (1 - b2**t)
    W1 -= lr * m1h / (np.sqrt(v1h) + eps)
    W2 -= lr * m2h / (np.sqrt(v2h) + eps)
    return W1, W2, m1, v1, m2, v2


def test_train_mlp_vs_adamw():
    print("=== 3. Обучение 2-слойной сети: KeralinGraph vs AdamW ===")
    np.random.seed(7)
    N, D, H = 256, 64, 48
    epochs = 1500
    lr = 0.02

    X = (np.random.randn(N, D) * 0.5).astype(np.float32)
    Wt1 = (np.random.randn(H, D) / np.sqrt(D)).astype(np.float32)
    Wt2 = (np.random.randn(1, H) / np.sqrt(H)).astype(np.float32)
    T = (np.tanh(X @ Wt1.T) @ Wt2.T).astype(np.float32)

    W1_0 = (np.random.randn(H, D) * 0.1).astype(np.float32)
    W2_0 = (np.random.randn(1, H) * 0.1).astype(np.float32)

    # ── Графовый оптимизатор Keralin ──
    graph = KG()
    graph.configure(lr=lr, wd=0.0, beta1=0.9, warmup_noise=1e-3, auto_warm=True)

    ids1 = add_row_nodes(graph, W1_0)          # H узлов (строки W1)
    id2 = graph.add_node(as_f32(W2_0[0]), ids1)  # 1 узел (строка W2)

    t0 = time.perf_counter()
    loss0 = float("nan")
    loss_g = float("nan")
    for ep in range(epochs):
        W1 = collect_rows(graph, ids1)
        W2 = collect_rows(graph, [id2])

        loss_g, dW1, dW2 = train_step(X, T, W1, W2, H)
        if ep == 0:
            loss0 = loss_g

        for i in range(H):
            graph.feed_grad(ids1[i], as_f32(dW1[i]))
        graph.feed_grad(id2, as_f32(dW2[0]))

        graph.step()

    dt_g = time.perf_counter() - t0

    # ── AdamW (NumPy-зеркало, тот же init/lr) ──
    W1 = W1_0.copy()
    W2 = W2_0.copy()
    m1 = np.zeros_like(W1)
    v1 = np.zeros_like(W1)
    m2 = np.zeros_like(W2)
    v2 = np.zeros_like(W2)
    loss_n = float("nan")
    for t in range(1, epochs + 1):
        loss_n, dW1, dW2 = train_step(X, T, W1, W2, H)
        W1, W2, m1, v1, m2, v2 = adamw_step(
            W1, W2, m1, v1, m2, v2, dW1, dW2, lr, 0.9, 0.999, 1e-8, t
        )

    print(f"graph : loss {loss0:.6f} -> {loss_g:.6f}  ({dt_g:.2f} s)")
    print(f"adamw : loss {loss0:.6f} -> {loss_n:.6f}")
    print(f"steps={graph.step_count}, coherence={graph.global_coherence:.4f}, "
          f"graph_sig={graph.graph_signature:016x}")

    ok, dead, cold = graph.integrity()
    counts = graph.alive_counts()
    print(f"integrity: ok={ok}, dead={dead}, cold={cold}")
    print(f"life counts (dead,cold,born,alive,stable,decaying)={counts}")

    assert graph.step_count == epochs
    assert np.isfinite(loss_g), "граф разошёлся"
    assert loss_g < 0.4 * loss0, f"граф не учится: {loss_g} vs старт {loss0}"
    assert loss_g < 0.01, f"граф не достиг хорошего качества: {loss_g}"
    assert ok and dead == 0, "после обучения все узлы должны быть живы"
    assert graph.graph_signature != 0

    # ── Проверка accumulate_grad ──
    g2 = KG()
    g2.configure(lr=1e-3, wd=0.0, beta1=0.0, warmup_noise=1e-3, auto_warm=True)
    nid = g2.add_node(as_f32(W1_0[0]), [])
    g2.cold_start()
    half = as_f32((dW1[0] * 0.5).astype(np.float32))
    g2.accumulate_grad(nid, half, 1.0)
    g2.accumulate_grad(nid, half, 1.0)
    assert g2.step_count == 0
    g2.step()
    assert g2.step_count == 1
    print("✅ accumulate_grad: две половинки = один градиент")
    print("\n✅ KERALIN GRAPH TEST PASSED")


if __name__ == "__main__":
    print("AtomNET version:", atomnet.version())
    test_structure_and_cold_start()
    test_cycle_detection()
    test_train_mlp_vs_adamw()
    print("\nALL KERALIN GRAPH TESTS PASSED")
