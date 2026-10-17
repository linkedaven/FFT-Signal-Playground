"""FFT & Signal Playground backend: signal generation and DSP over HTTP.

Run:  python server.py   (opens http://127.0.0.1:8003 in your browser)
"""
import logging
import os
import socket
import sys
import threading
import time
import webbrowser

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

import signals

BASE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024
FS_RANGE = (100.0, 48000.0)


def _num(src, key, default, lo, hi):
    try:
        v = float(src.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number.")
    if not (np.isfinite(v) and lo <= v <= hi):
        raise ValueError(f"{key} must be between {lo:g} and {hi:g}.")
    return v


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/preset")
def api_preset():
    a = request.args
    try:
        fs = _num(a, "fs", 8000, *FS_RANGE)
        n = int(_num(a, "n", 4096, signals.MIN_N, signals.MAX_N))
        freq = _num(a, "freq", 100, 0.1, fs * 2)
        noise = _num(a, "noise", 0, 0, 5)
        wpm = _num(a, "wpm", 20, 5, 40)
        x = signals.preset(a.get("name", "sine"), fs, n, freq, noise,
                           text=a.get("text", "SOS")[:40], wpm=wpm)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(x=x.tolist())


@app.post("/api/analyze")
def api_analyze():
    d = request.get_json(silent=True) or {}
    try:
        fs = _num(d, "fs", 8000, *FS_RANGE)
        x = d.get("x")
        if not isinstance(x, list):
            raise ValueError("x must be a list of samples.")
        out = signals.analyze(x, fs, d.get("window", "hann"), d.get("filter"))
    except (ValueError, TypeError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(out)


# ----------------------------------------------------------------------
# Launcher: open the browser, and quit when the browser tab is closed
# ----------------------------------------------------------------------
HOST, PORT = "127.0.0.1", 8003
URL = f"http://{HOST}:{PORT}"
_seen = {"last": time.time(), "bye": None}


@app.post("/api/ping")
def ping():
    _seen["last"] = time.time()
    _seen["bye"] = None
    return jsonify(ok=True)


@app.post("/api/bye")
def bye():
    _seen["bye"] = time.time() + 5
    return jsonify(ok=True)


def _watchdog():
    while True:
        time.sleep(1)
        now = time.time()
        if (_seen["bye"] and now > _seen["bye"]) or now - _seen["last"] > 120:
            os._exit(0)


def _already_running():
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def main():
    if _already_running():
        webbrowser.open(URL)
        return
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(target=_watchdog, daemon=True).start()
    threading.Timer(1.0, webbrowser.open, args=[URL]).start()
    app.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    print(f"FFT & Signal Playground running at {URL}  (Ctrl+C to stop)")
    main()
