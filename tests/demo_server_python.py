#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_server_python.py — HTTP-сервер генерации поверх AtomNET (stdlib).

Минимальный сервер: POST /generate {prompt, max_new, top_k} -> {tokens}.
GET /health -> {status}. Запуск: python tests/demo_server_python.py [--selftest].
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "site-packages"))

import numpy as np  # noqa: F401
import atomnet  # noqa: F401

from demo_vocab_bpe import train_bpe, build_corpus
from demo_nnet_python import AtomGPT, NetConfig


class GenHandler:
    def __init__(self, model, tok):
        self.net = model
        self.tok = tok

    def generate(self, prompt: str, max_new: int = 8, top_k: int = 8) -> list[int]:
        ids = self.tok.encode(prompt)
        return self.net.generate_cached(ids, max_new, top_k=top_k)


def make_handler(model, tok):
    from http.server import BaseHTTPRequestHandler

    gen = GenHandler(model, tok)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # тихий
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404, "not found")
                return
            self._json({"status": "ok", "layers": model.cfg.n_layers})

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404, "not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            ids = gen.generate(
                body.get("prompt", ""),
                max_new=int(body.get("max_new", 8)),
                top_k=int(body.get("top_k", 8)),
            )
            self._json({"tokens": ids})

        def _json(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--selftest", action="store_true",
                        help="поднять, прогнать generate, погасить и выйти")
    args = parser.parse_args()

    tok = train_bpe(build_corpus())
    cfg = NetConfig(vocab=tok.vocab_size, d=128, heads=4, kv_heads=2, n_layers=2, hidden=512)
    model = AtomGPT(cfg, seed=0)
    handler = make_handler(model, tok)

    from http.server import HTTPServer

    server = HTTPServer(("127.0.0.1", args.port), handler)

    if args.selftest:
        import threading
        import urllib.request

        t = threading.Thread(
            target=lambda: (
                server.handle_request(),
                server.handle_request(),
                server.handle_request(),
            ),
            daemon=True,
        )
        t.start()
        import time

        for _ in range(200):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                    print("SELF-HEALTH", json.loads(response.read()))
                    break
            except Exception:
                time.sleep(0.05)
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/generate",
            data=json.dumps({"prompt": "как работает поток", "max_new": 4}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            print("SELF-GENERATE", json.loads(response.read()))
        return

    print("serve: http://127.0.0.1:%d/generate (POST {prompt, max_new, top_k})" % args.port)
    print("health: http://127.0.0.1:%d/health" % args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
