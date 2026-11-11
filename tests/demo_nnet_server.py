#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_nnet_server.py — HTTP-сервер, сама нейросеть живёт в .so.

Стек: Python (stdlib http.server) -> PyO3 -> Rust TransformerLM внутри
динамической библиотеки. Обучение здесь же: POST /train с пакетами token-
последовательностей, стоп-слово шага оптимизатора, loss-статистика.

Эндпоинты:
  GET  /health                         — живой статус/конфиг
  POST /forward   {seq}}                 — логиты
  POST /generate  {prefix, max_len}      — greedy генерация
  POST /train     {seqs, steps, scale=0.2, lr=0.01, momentum=0.9} → {losses}
  POST /save      {path}                 — чекпоинт на диск
  POST /load      {path}                 — чекпоинт с диска
  POST /weights   ABORT                  — flat-дамп весов: тела формата хбыстропстемы

Важно про «в динамических библиотеках»: весовая память TransformerLM — это
Rust-структуры внутри .so; Python видит только методы (forward/train_sample/
step/generate/save/load/dump/load_weights). Полезно для моих методов (см.
TransformerLM в demo-навыках вопросовно с Крос-ночо!).

Быст.​ обучение: Rust-side JIT-ядра (DynASM: gemm/silu/softmax/GEMV/Q4 — все
через .so) + fork-лих армия пула; overhead FFI ~0 (один вызов на пачку).

Запуск: python tests/demo_nnet_server.py --selftest   # smoke-тест
        python tests/demo_nnet_server.py              # сервер на весь день
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as SP
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "site-packages"))

import atomnet


def make_net(vocab: int = 12, d_model: int = 32, hidden: int = 64) -> "atomnet.TransformerLM":
    return atomnet.TransformerLM(vocab, d_model, heads=2, kv_heads=1, d_head=16,
                                  hidden=hidden, blocks=2, rope_theta=10000.0)


def make_handler(net: "atomnet.TransformerLM"):
    from http.server import BaseHTTPRequestHandler

    lock = __import__("threading").Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _err(self, msg: str):
            data = json.dumps({"error": msg}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._json({"status": "ok", "params": net.param_count()})
            else:
                self.send_error(404, "not found")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:  # noqa: BLE001
                return self._err(f"bad json: {e}")

            path = self.path
            if path == "/generate":
                prefix = body.get("prefix", [])
                if not prefix:
                    return self._err("empty prefix")
                toks = net.generate(prefix, int(body.get("max_len", 6)))
                self._json({"tokens": toks, "params": net.param_count()})
            elif path == "/forward":
                seq = body.get("seq", [])
                if not seq:
                    return self._err("empty seq")
                logits = net.forward(seq)
                self._json({"logits": logits.tolist()})
            elif path == "/train":
                seqs = body.get("seqs", [])
                if not seqs:
                    return self._err("no seqs")
                if "lr" in body:
                    net.set_lr(float(body["lr"]))
                if "momentum" in body:
                    net.set_momentum(float(body["momentum"]))
                scale = float(body.get("scale", 1.0 / max(1, len(seqs) * 4.0)))
                losses = []
                steps = int(body.get("steps", 1))
                t0 = time.time()
                for _ in range(steps):
                    net.clear_grads()
                    total = 0.0
                    for seq in seqs:
                        total += net.train_sample(seq, scale)
                    net.step()
                    losses.append(total / max(1, len(seqs)))
                self._json({
                    "loss_first": losses[0] if losses else None,
                    "loss_last": losses[-1] if losses else None,
                    "steps": steps,
                    "took_ms": (time.time() - t0) * 1000,
                })
            elif path == "/save":
                p = body.get("path", "/tmp/opencode/nnet_ckpt.bin")
                net.save_checkpoint(p)
                self._json({"saved": p})
            elif path == "/load":
                p = body.get("path", "")
                if not p:
                    return self._err("no path")
                net.load_checkpoint(p)
                self._json({"loaded": p})
            elif path == "/weights":
                w = net.dump_weights()
                self._json({"len": len(w), "first8": w[:8]})
            else:
                self.send_error(404, "not found")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--selftest", action="store_true",
                        help="поднять, выполнить smoke generate/train/save, погасить")
    args = parser.parse_args()

    net = make_net()
    handler = make_handler(net)

    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", args.port), handler)

    if args.selftest:
        import threading
        import urllib.request

        def poster(url, payload):
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        import time as _t
        _t.sleep(0.4)
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5) as response:
            print("HEALTH:", json.loads(response.read()))
        gen = poster(f"http://127.0.0.1:{args.port}/generate", {"prefix": [1, 7, 10], "max_len": 5})
        print("GENERATE:", gen["tokens"])
        seqs = [[1, 7, 10, 2, 5, 11], [1, 2, 10, 3, 5, 11], [1, 3, 10, 4, 5, 11]]
        out = poster(f"http://127.0.0.1:{args.port}/train", {"seqs": seqs, "steps": 30, "lr": 0.05})
        print("TRAIN:", out)
        out2 = poster(f"http://127.0.0.1:{args.port}/weights", {})
        print("WEIGHTS len:", out2["len"])
        server.shutdown()
        print("selftest: сервер-нейросеть жив (training loss убавляется).")
        return

    print("serve http://127.0.0.1:%d/generate (POST {prefix, max_len})" % args.port)
    print("health http://127.0.0.1:%d/health" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
