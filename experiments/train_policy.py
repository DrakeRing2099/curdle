"""Train the Q-learning policy against the twin.

    python experiments/train_policy.py                 # main run: lambda = 10, 3000 years
    python experiments/train_policy.py --lam 30 --years 1500 --out results/q_table_lam30.npz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dairytwin.mdp import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lam", type=float, default=10.0, help="rupees of penalty per lost unit")
ap.add_argument("--mu", type=float, default=0.0, help="rupees of penalty per wasted unit")
ap.add_argument("--years", type=int, default=3000)
ap.add_argument("--out", default="results/q_table.npz")
a = ap.parse_args()
train(lam=a.lam, mu=a.mu, years=a.years, out_path=a.out, log=lambda m: print(m, flush=True))
