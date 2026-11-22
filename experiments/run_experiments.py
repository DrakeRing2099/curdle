"""Monte Carlo experiment grid.

Every experiment runs the same N seeds (common random numbers), so year i of every
experiment faces the same weather, milk noise, power cuts and late vans.

    python experiments/run_experiments.py --n 100
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dairytwin.config import Scenario  # noqa: E402
from dairytwin.montecarlo import run_batch  # noqa: E402

EXPERIMENTS = [
    # key, policy, scenario, label
    ("habit", "Habit", Scenario(name="Baseline"), "Habit"),
    ("heat", "Heat-aware", Scenario(name="Baseline"), "Heat-aware"),
    ("habit_batt_risk", "Habit", Scenario(name="Battery fridges, risk-targeted", battery_fridge_share=0.3),
     "Habit + 30% battery fridges (riskiest feeders)"),
    ("habit_batt_rand", "Habit", Scenario(name="Battery fridges, random", battery_fridge_share=0.3,
                                          battery_targeting="random"),
     "Habit + 30% battery fridges (random shops)"),
    ("habit_returns", "Habit", Scenario(name="Kirana returns", kirana_returns=True), "Habit + kirana returns"),
    ("heat_batt", "Heat-aware", Scenario(name="Battery fridges, risk-targeted", battery_fridge_share=0.3),
     "Heat-aware + 30% battery fridges"),
    ("habit_nocut", "Habit", Scenario(name="No power cuts", outages_enabled=False), "Habit, no power cuts"),
    ("habit_severe", "Habit", Scenario(name="Severe summer").severe_summer(), "Habit, severe summer"),
    ("heat_severe", "Heat-aware", Scenario(name="Severe summer").severe_summer(), "Heat-aware, severe summer"),
    ("heat_batt_severe", "Heat-aware",
     Scenario(name="Severe summer + batteries", battery_fridge_share=0.3).severe_summer(),
     "Heat-aware + batteries, severe summer"),
    ("habit_severe_nocut", "Habit", Scenario(name="Severe summer, no power cuts",
                                             outages_enabled=False).severe_summer(),
     "Habit, severe summer, no power cuts"),
    # learned policies (train first: experiments/train_policy.py)
    ("learned", "Learned", Scenario(name="Baseline"), "Learned (Q-learning, λ = ₹10)"),
    ("learned_severe", "Learned", Scenario(name="Severe summer").severe_summer(), "Learned, severe summer"),
    ("learned_lam0", "Learned:results/q_table_lam0.npz", Scenario(name="Baseline"), "Learned, λ = ₹0 (profit only)"),
    ("learned_lam30", "Learned:results/q_table_lam30.npz", Scenario(name="Baseline"), "Learned, λ = ₹30"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/mc.pkl")
    ap.add_argument("--only", nargs="*", help="run only these experiment keys, keeping the rest of --out")
    a = ap.parse_args()
    results = {}
    if a.only and Path(a.out).exists():
        with open(a.out, "rb") as f:
            results = pickle.load(f)["experiments"]
    t0 = time.time()
    for key, pol, scen, label in EXPERIMENTS:
        if a.only and key not in a.only:
            continue
        if pol.startswith("Learned") and not Path(pol.split(":", 1)[1] if ":" in pol else "results/q_table.npz").exists():
            print(f"{key:22s} skipped (no trained Q table yet)", flush=True)
            continue
        t = time.time()
        recs = run_batch(pol, scen, a.n, workers=a.workers)
        results[key] = {"policy": pol, "scenario": scen, "label": label, "records": recs}
        print(f"{key:22s} {time.time() - t:6.1f}s", flush=True)
        with open(a.out, "wb") as f:
            pickle.dump({"n": a.n, "experiments": results}, f)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
