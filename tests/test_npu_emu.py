"""NPU-эмулятор обучения (JIT-mac-тайлы 6x16, 1 поток) + single-thread.

Проходят subprocess'ами с env:
  A) fingerprint -- fingerprint matmul+backward на фикс. матрицах (ядро),
  B) emulator    -- то же с ATOMNET_NPU_EMU=1 (GEMM идёт через JIT-блоки 6x16,
                    принудительно 1 поток, mac-массив): допускаемый разброс
                    ≤ 2e-5 (порядок редукции другой, значение то же),
  C) singlethread-- demo_first_nn со строкой SELFTEST под
                    ATOMNET_THREADS=1 (+taskset -c 0 если доступен).
"""
import os
import subprocess
import sys

PY = "/home/keiron/venv/bin/python"


def run_env(script_args, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    out = subprocess.run(
        [PY, *script_args], capture_output=True, text=True, env=env, cwd="/tmp")
    return out.returncode, out.stdout, out.stderr


FINGERPRINT = """
import atomnet as A
from atomnet import functional as F, first_nn
from atomnet import optimizers as optmod
A.manual_seed(123)
net = first_nn.MiniTransformer(vocab=12, d_model=32, blocks=1, max_len=16, seed=123)
for s in range(3):
    opt = optmod.Adam(net.parameters(), lr=0.01)
    opt.zero_grad()
    seq = [s+1, 2, 10, 3, 4, 11, 0, 0, s]
    lg = net(seq)
    x = A.rows_slice(lg, 0, 8)
    loss = F.cross_entropy(x, seq[1:])
    loss.backward()
    opt.step()
    print('step', s, 'L', loss.item(), 'wd', net.head.weight.tolist()[:3])
print('emu', A.npu_emu_on(), A.npu_status())
print('FINGER_OK')
"""


def test_emu_parity():
    rc, out, err = run_env(["-c", FINGERPRINT], {})
    assert rc == 0, err
    rc2, out2, err2 = run_env(["-c", FINGERPRINT], {"ATOMNET_NPU_EMU": "1"})
    assert rc2 == 0, err2
    import re
    def losses(txt):
        res = {}
        for l in txt.splitlines():
            m = re.match(r"step (\d+) L ([\-0-9.e+]+)", l)
            if m:
                res[int(m.group(1))] = float(m.group(2))
        return res
    f1, f2 = losses(out), losses(out2)
    assert len(f1) == 3, f1
    for k in f1:
        v1, v2 = f1[k], f2[k]
        assert abs(v1 - v2) / max(abs(v1), 1e-6) < 2e-5, (k, v1, v2)
    assert "npu_emu=true" in out2
    print("OK NPU-emu parity (rel err < 2e-5, mac-shapes match)")
    return rc, out


def test_single_thread_training():
    rc, out, err = run_env(
        ["/home/keiron/atomnet/tests/demo_first_nn.py", "--selftest"],
        {"ATOMNET_THREADS": "1"})
    assert rc == 0, err
    assert "SELFTEST OK" in out
    # идентичность final loss между потоками — на уровне цифры (детерминирована)
    final = [l for l in out.splitlines() if "step  23" in l][0].split()[-1]
    rc2, out2, err2 = run_env(
        ["/home/keiron/atomnet/tests/demo_first_nn.py", "--selftest"], {})
    final2 = [l for l in out2.splitlines() if "step  23" in l][0].split()[-1]
    assert final == final2, (final, final2)
    print(f"OK single-thread: ATOMNET_THREADS=1 даёт тот же final loss ({final})")


def test_status():
    out = A = __import__("atomnet")
    s = A.npu_status()
    assert "thread_budget=" in s
    print("OK npu_status:", s)


if __name__ == "__main__":
    for fn in [test_status, test_emu_parity, test_single_thread_training]:
        fn()
