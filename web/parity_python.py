"""Python side of the parity check: same KPIs as parity_check.js.
    python web/parity_python.py 30 Habit
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dairytwin import DairyTwin, POLICIES
from dairytwin.config import Scenario, TwinConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
pol = sys.argv[2] if len(sys.argv) > 2 else "Habit"
cfg = TwinConfig()
rows = []
for s in range(N):
    o = DairyTwin(policy=POLICIES[pol](), seed=1000 + s).run()
    d, sa, h = o["demand"], o["sales"], o["weather"] == 1
    waste = o["waste_kirana"].sum() + o["waste_plant"].sum() + o["waste_mt"].sum() + o["waste_qc"].sum()
    W = cfg.days // 7
    wo = o["kirana_orders"][:, 0][: W * 7].reshape(W, 7).sum(1)
    wd = d[:, 0, 0][: W * 7].reshape(W, 7).sum(1)
    rows.append({"fill": sa.sum() / d.sum(), "fill_heat": sa[h].sum() / d[h].sum() if h.any() else np.nan,
                 "waste_share": waste / o["made"].sum(), "lost": int((d - sa).sum()),
                 "bullwhip": wo.var() / wd.var(), "migrated": int(cfg.city.households[0] - o["customers"][-1, 0]),
                 "profit": float(o["cash"][-1] - cfg.money.opening_cash)})
out = {}
for k in rows[0]:
    a = np.array([r[k] for r in rows], float); a = a[~np.isnan(a)]
    out[k] = {"mean": a.mean(), "sd": a.std(ddof=1), "p5": np.percentile(a, 5), "p95": np.percentile(a, 95)}
print(json.dumps({"engine": "python", "policy": pol, "N": N, "kpis": out}, indent=1, default=float))
