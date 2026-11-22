"""Live mirror server: serves web/live.html and steps the twin against a fake reality.

    python web/live_server.py [--port 8765] [--policy Habit]

Open http://localhost:8765. No dependencies beyond the project's own (numpy). The page polls
/api/state; its buttons POST to /api/control.
"""
import argparse, json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dairytwin.live import LiveMirror, available_policies

LOCK = threading.Lock()
S = {"mirror": None, "playing": False, "rate": 2.0, "auto_forecast": True, "forecasting": False}


def new_mirror(policy):
    S["mirror"] = LiveMirror(policy=policy)


def run_forecast():
    with LOCK:
        if S["forecasting"]:
            return
        S["forecasting"] = True
        import copy
        m = S["mirror"]
        snap = copy.copy(m)                         # forecast works on a deep copy of the twin
        snap.twin = copy.deepcopy(m.twin)
    try:
        res = snap.forecast()
        with LOCK:
            if S["mirror"] is m:
                m.forecast_result = res
    finally:
        S["forecasting"] = False


def loop():
    last = time.time()
    while True:
        time.sleep(0.05)
        if not S["playing"]:
            last = time.time(); continue
        if time.time() - last >= 1.0 / S["rate"]:
            last = time.time()
            with LOCK:
                m = S["mirror"]; m.step()
                monday = m.twin.day % 7 == 0
            if S["auto_forecast"] and monday:
                threading.Thread(target=run_forecast, daemon=True).start()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send((ROOT / "web" / "live.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/init":
            with LOCK:
                self._send(json.dumps(S["mirror"].static()))
        elif self.path.startswith("/api/state"):
            with LOCK:
                st = S["mirror"].state()
                st.update(playing=S["playing"], rate=S["rate"], auto_forecast=S["auto_forecast"],
                          forecasting=S["forecasting"])
                self._send(json.dumps(st))
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        c = json.loads(self.rfile.read(n) or b"{}")
        a = c.get("action")
        if a == "play": S["playing"] = True
        elif a == "pause": S["playing"] = False
        elif a == "rate": S["rate"] = float(c["value"])
        elif a == "auto_forecast": S["auto_forecast"] = bool(c["value"])
        elif a == "step":
            with LOCK: S["mirror"].step()
        elif a == "sync":
            with LOCK: S["mirror"].sync = c["value"]
        elif a == "inject":
            with LOCK:
                S["mirror"].inject(c["kind"], c.get("value"), int(c.get("days", 7)), c.get("locality"))
        elif a == "reset":
            S["playing"] = False
            with LOCK: new_mirror(c.get("policy") or S["mirror"].policy_name)
        elif a == "forecast":
            threading.Thread(target=run_forecast, daemon=True).start()
        self._send(json.dumps({"ok": True}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--policy", default="Habit", choices=available_policies())
    a = ap.parse_args()
    new_mirror(a.policy)
    threading.Thread(target=loop, daemon=True).start()
    print(f"live mirror on http://localhost:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
