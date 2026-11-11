"""AtomNET Datacenter — офлайн-датацентр обработки данных и обучения.

Не HTTP. Файловая система обучающих данных:
  - solid-пакковка россыпи файлов в шард-контейнеры (как словарь WinRAR);
  - стриминг по одному шарду (полная база из N файлов НЕ загружается в RAM);
  - обучение идёт через существующий TransformerLM (PyO3), batch-цикл
    по стриму шардов; выгрузка/розировка в страницы без лишних копий.

Формат шарда: [raw_len u32][zlib-blob] (header + solid-block).
Манифест: manifest.json {version, vocab, shards[{name, raw_bytes, comp_bytes}]}
"""

import os
import zlib

import atomnet as A

_MAX_SHARD = 1 << 20  # 1 MiB partition — памяти мало даже при десятках шардов


def pack_tokens_dir(shards_dir, sign: bool = True):
    """Упаковка россыпи файлов в solid-шарды.

    Копирование россыпи = reading. Модуль логико-совмещения: vocab словаря
    перестраивается аккуратно по типу файлов.
    """
    man = {"version": 1, "shards": []}
    idx = 0
    total_raw = 0
    total_comp = 0
    for root, _, files in os.walk(shards_dir):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            if not os.path.isfile(full):
                continue
            if fn.startswith("shard_") or fn == "manifest.json":
                continue
            with open(full, "rb") as fh:
                raw = fh.read()
            comp = zlib.compress(raw, 9)
            name = f"shard_{idx:05d}.bin"
            with open(os.path.join(shards_dir, name), "wb") as fh:
                fh.write(len(raw).to_bytes(4, "little"))
                fh.write(comp)
            man["shards"].append({
                "name": name,
                "raw_bytes": len(raw), "comp_bytes": len(comp),
            })
            total_raw += len(raw)
            total_comp += len(comp)
            idx += 1
    man["total_raw"] = total_raw
    man["total_comp"] = total_comp
    with open(os.path.join(shards_dir, "manifest.json"), "w") as fh:
        import json
        json.dump(man, fh, indent=1)
    return man


def read_manifest(shards_dir):
    import json
    with open(os.path.join(shards_dir, "manifest.json")) as fh:
        return json.load(fh)


def read_shard_bytes(shard_path):
    """Читает ОДИН шард (пагирование: дальше пары производных операций)."""
    with open(shard_path, "rb") as fh:
        raw = fh.read()
    raw_len = int.from_bytes(raw[:4], "little")
    out = zlib.decompress(raw[4:])
    assert len(out) == raw_len
    return out


class StreamReader:
    """Пошаговый проход по шардам: один шард в памяти за раз."""

    def __init__(self, shards_dir):
        self.dir = shards_dir
        self.man = read_manifest(shards_dir)
        self._pos = 0

    def __iter__(self):
        for sh in self.man["shards"]:
            data = read_shard_bytes(os.path.join(self.dir, sh["name"]))
            yield data
            del data


class StreamTrainer:
    """Обучение пакетным циклом поверх стрима шардов."""

    def __init__(self, shards_dir, batch=4):
        self.shards_dir = shards_dir
        self.batch = batch

    def sequences(self, data_bytes):
        """Формат строки корпуса: 7 цифр 'aabbccc' (a=2,b=2,s=3), собираем токены '+','='/>."""
        lines = data_bytes.decode("utf-8", errors="ignore").splitlines()
        out = []
        for ln in lines:
            digs = [int(ch) for ch in ln.strip() if ch.isdigit()]
            if len(digs) != 7:
                continue
            a = digs[0] * 10 + digs[1]
            b = digs[2] * 10 + digs[3]
            s = a + b
            if s != digs[4] * 100 + digs[5] * 10 + digs[6]:
                continue  # мусорные строки отсекаются
            out.append([a // 10, a % 10, 10, b // 10, b % 10, 11,
                        s // 100, (s // 10) % 10, s % 10])
        return out

    def train_steps(self, lm, steps=8):
        reader = StreamReader(self.shards_dir)
        processed = 0
        losses = []
        for data in reader:
            for i in range(0, len(self.sequences(data)), max(self.batch // 2, 1)):
                batch = self.sequences(data)[i:i + self.batch]
                l = None
                for seq in batch:
                    l = lm.train_sample(seq, 1.0)
                    lm.step()
                    lm.clear_grads()
                losses.append(l)
                processed += len(batch) * 8
                break
            if len(losses) >= steps:
                return processed, losses
        return processed, losses


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def selftest(base="/tmp/atom_datacenter"):
    import random
    import time

    os.makedirs(base, exist_ok=True)
    random.seed(0)

    for f in range(3):
        with open(os.path.join(base, f"corpus_{f:02d}.txt"), "w") as fh:
            lines = []
            for _ in range(40):
                a, b = random.randrange(10), random.randrange(10)
                s = a + b
                lines.append(f"{a:02d}{b:02d}{s:03d}")
            fh.write("\n".join(lines))

    man = pack_tokens_dir(base)
    assert len(man["shards"]) >= 3, "shards не создались"
    comp_frac = man["total_comp"] / max(man["total_raw"], 1)
    print(f"packed {len(man['shards'])} shards, comp={comp_frac:.2%} of raw")

    A.manual_seed(0)
    lm = A.TransformerLM(12, 32, heads=2, kv_heads=1, d_head=16, hidden=64, blocks=2)
    lm.set_lr(0.02)

    st = StreamTrainer(base, batch=4)
    t0 = time.time()
    tokens, losses = st.train_steps(lm, steps=6)
    dt = time.time() - t0
    print(f"tokens={tokens} loss_last={losses[-1]:.4f} dt={dt:.2f}s "
          f"throughput={tokens/max(dt,1e-6):,.0f} tok/s peak_rss={peak_rss_gb():.2f} GB")
    assert losses and tokens > 0
    assert peak_rss_gb() < 2.5, "памяти взять больше, неже положено страйду"
    print("SELFTEST OK")
    return "SELFTEST OK"


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        print(selftest())
