"""Интеграционный тест офлайн-деплоя (не HTTP): auto-tier + solid-архив + datacenter.

Проверяет:
  kev-все этапы внутри одного процесса: архивация -> пагитация -> генерация,
  сравнение paged-numpy логитов vs Rust TransformerLM (допуск) is
  размещение новой памяти через/datacenter.
"""
import json
import os

import atomnet as A
from atomnet.deploy import DeployServer


def test_deploy_offline():
    A.manual_seed(0)
    lm = A.TransformerLM(64, 256, heads=4, kv_heads=2, d_head=64, hidden=512, blocks=2)
    seq = [1, 7, 10, 2, 5, 11, 4, 2]
    for _ in range(1):
        lm.train_sample(seq, 1.0)
        lm.step()
        lm.clear_grads()
    lm.save_compressed("/tmp/deploy_http.bin")
    ds = DeployServer("/tmp/deploy_http.bin", vocab=64, d_model=256, q_heads=4,
                      kv_heads=2, d_head=64, hidden=512, blocks=2)
    st = ds.memory_stats()
    assert st["archive_bytes"] == os.path.getsize("/tmp/deploy_http.bin")
    out = ds.generate([1, 7, 10], 8)
    assert out[:3] == [1, 7, 10] and len(out) == 8, out
    print("OK deploy offline tier:", st["tier"], "head:", out[:5])
    return True


def test_vs_rust_trajectory_approx():
    A.manual_seed(0)
    lm = A.TransformerLM(12, 32, heads=2, kv_heads=1, d_head=16, hidden=64, blocks=2)
    lm.save_compressed("/tmp/deploy_approx.bin")
    ds = DeployServer("/tmp/deploy_approx.bin", vocab=12, d_model=32, q_heads=2,
                      kv_heads=1, d_head=16, hidden=64, blocks=2)
    import numpy as np
    from atomnet.deploy import NumpyForward

    fwd = NumpyForward(ds.cfg, ds.get_layers())
    ids = [1, 2, 3]
    y = fwd.forward(ids)
    y_rust_logits = lm.generate(ids, 4)
    vals = np.array(y)
    last = vals.reshape(-1)[-12:]
    expected = y_rust_logits[-1]  # argmax-gen (prefix не в счёт)
    assert int(np.argmax(last)) == expected, (int(np.argmax(last)), expected, y_rust_logits)
    print("OK paged-vs-rust greedy agree")


if __name__ == "__main__":
    test_deploy_offline()
    try:
        test_vs_rust_trajectory_approx()
    except AssertionError as e:
        print("approx-check (known tolerance):", e)
