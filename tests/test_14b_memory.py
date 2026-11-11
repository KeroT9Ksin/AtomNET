"""Память для 14B на основе AtomNET (чистая проверка чисел бюджет-теста).

Независимый пересчёт архитектуры (LmConfig::big: LLaMA-3-8B-класс):
    d=5120, q_heads=40, kv_heads=8, d_head=128, hidden=17408, blocks=48,
    vocab=151_936 (не tied).
И сверка с таблицей Rust-теста atom_budget_14b_detailed.
"""
import subprocess
import re
import sys


def python_budget():
    d, heads_v, kv, dh, hid, blocks, vocab = 5120, 40, 8, 128, 17408, 48, 151_936
    qdim = heads_v * dh
    per_layer = d * qdim + d * (kv * dh) + d * (kv * dh) + qdim * d + d * hid + hid * d
    p = vocab * d + d * vocab + blocks * per_layer
    fp32 = p * 4.0 / 1e9
    q4 = fp32 / 2.51
    q8 = p * 1.25 / 1e9
    train = fp32 * 5.0
    return p, fp32, q4, q8, train


def test_14b_memory_numbers():
    p, fp32, q4, q8, train = python_budget()
    assert 13.0e9 < p < 14.6e9, p
    assert 52.0 < fp32 < 56.0, fp32
    assert 20.0 < q4 < 23.0, q4
    assert train > 250.0, train
    print(f"python budget: params={p/1e9:.2f}B fp32={fp32:.2f}GB Q4={q4:.2f}GB Q8={q8:.2f}GB train~{train:.0f}GB")


def test_vs_rust_table():
    out = subprocess.run(
        ["cargo", "test", "--release", "--offline", "--lib", "atom_budget_14b_detailed",
         "--", "--nocapture"],
        capture_output=True, text=True, cwd="/home/keiron/atomnet")
    assert out.returncode == 0, out.stderr[-1500:]
    m = re.search(r"weights fp32\s+:\s+([\d.]+) GB", out.stdout)
    assert m, out.stdout[-2000:]
    rust_fp32 = float(m.group(1))
    _, py_fp32, _, _, _ = python_budget()
    assert abs(rust_fp32 - py_fp32) / py_fp32 < 0.005, (rust_fp32, py_fp32)
    print(f"rust==python fp32 {rust_fp32:.2f}GB")
    m2 = re.search(r"weights Q4 \(FLASH\)\s+:\s+([\d.]+) GB", out.stdout)
    assert m2
    assert abs(float(m2.group(1)) - 20.5) < 1.5
    print(f"rust Q4 {float(m2.group(1)):.2f}GB ok")


def test_machine_verdict():
    mem_kb = 0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                break
    avail_gb = mem_kb / 1e6
    _, _, q4, _, _ = python_budget()
    fits = q4 <= avail_gb
    print(f"this machine RAM={avail_gb:.1f}GB; 14B Q4 fits here? {fits} (hosting need {'yes' if fits else 'external (>=64GB)'})")
    return fits


if __name__ == "__main__":
    test_14b_memory_numbers()
    test_vs_rust_table()
    test_machine_verdict()
    print("OK: 14B memory budget verified (python formula == rust table)")
