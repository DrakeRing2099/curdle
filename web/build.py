"""Inline the engine, config, learned policy and Monte Carlo summary into one self-contained page."""
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
ROOT = W.parent
sys.path.insert(0, str(ROOT))


def policy_json(path=ROOT / "results" / "q_table.npz"):
    """Greedy action per state (-1 = never visited), plus the action menu, for engine.js."""
    if not Path(path).exists():
        return "null"
    from dairytwin.mdp import ACTIONS, DEFAULT_HEAT, DEFAULT_NORMAL
    z = np.load(path)
    Q, N = z["Q"], z["N"]
    table = [int(np.argmax(np.where(N[s] > 0, Q[s], -np.inf))) if N[s].sum() > 0 else -1 for s in range(Q.shape[0])]
    return json.dumps({"table": table, "default_heat": DEFAULT_HEAT, "default_normal": DEFAULT_NORMAL,
                       "actions": [{**a.__dict__, "label": a.label()} for a in ACTIONS]}, separators=(",", ":"))


(W / "policy.json").write_text(policy_json())     # for parity_check.js
html = (W / "app.html").read_text()
html = html.replace("/*ENGINE*/", "globalThis.DAIRY_POLICY = " + policy_json() + ";\n" + (W / "engine.js").read_text())
html = html.replace("/*CONFIG*/", json.dumps(json.loads((W / "config.json").read_text()), separators=(",", ":")))
summ = ROOT / "results" / "summary.json"
html = html.replace("/*SUMMARY*/", json.dumps(json.loads(summ.read_text()), separators=(",", ":")) if summ.exists() else "null")
if not html.lstrip().lower().startswith("<!doctype"):
    html = '<!DOCTYPE html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n' + html
out = W / "dairy_twin_control_room.html"
out.write_text(html)
print(out, round(len(html) / 1024), "KB")
