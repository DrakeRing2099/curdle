"""The decision layer as a Markov decision process, solved with tabular Q-learning.

MDP element        In the twin
-----------        -----------
State              compressed observation (72 combinations, see `encode_state`)
Action             one of 24 weekly decisions (see ACTIONS)
Transition         run the stochastic twin for 7 days
Reward             weekly profit - lambda x lost sales - mu x waste, in lakh rupees
Policy             greedy action from the learned Q table

Training runs on seeds 10_000+ so it never sees the 100 evaluation years (seeds 0..99).
"""
from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import BUTTER, CURD, KIR, MILK, MT, N_PRODUCTS, PANEER, QC, Scenario, TwinConfig
from .policies import WeeklyDecision
from .twin import DairyTwin

# ------------------------------------------------------------------ state
FILL_BANDS = (0.92, 0.96)          # last week's kirana availability: low / ok / high
N_STATES = 2 * 2 * 2 * 3 * 3


def encode_state(obs: dict) -> int:
    season = 0 if obs["flush"] else (1 if obs["lean"] else 2)    # flush / lean / normal
    f = obs["kirana_fill_7d"]
    band = 0 if f < FILL_BANDS[0] else (1 if f < FILL_BANDS[1] else 2)
    s = obs["weather"]
    s = s * 2 + obs["festival"]
    s = s * 2 + obs["supply"]
    s = s * 3 + season
    s = s * 3 + band
    return int(s)


def describe_state(i: int) -> dict:
    band = i % 3; i //= 3
    season = i % 3; i //= 3
    supply = i % 2; i //= 2
    festival = i % 2; i //= 2
    weather = i
    return {"weather": ["Normal", "Heat wave"][weather], "festival": bool(festival),
            "supply": ["Normal", "Stressed"][supply], "season": ["Flush", "Lean", "Normal"][season],
            "kirana_fill": ["< 92%", "92-96%", "> 96%"][band]}


# ------------------------------------------------------------------ actions
@dataclass(frozen=True)
class Action:
    weather_forecast: bool        # scale last week's orders by the change in demand conditions
    margin: float                 # production margin over the forecast
    kiranas_first: bool           # on shortage: kiranas, then apps, then supermarkets; staples first
    twice_daily: bool             # kirana noon drop

    def label(self) -> str:
        return (f"{'weather' if self.weather_forecast else 'habit'} forecast +{int(self.margin * 100)}%, "
                f"{'kiranas & staples first' if self.kiranas_first else 'pro-rata'}, "
                f"{'2' if self.twice_daily else '1'} drop{'s' if self.twice_daily else ''}/day")


ACTIONS = [Action(*a) for a in itertools.product((False, True), (0.0, 0.08, 0.16), (False, True), (False, True))]
N_ACTIONS = len(ACTIONS)
# used in states the learner never visited: the hand-written heat-aware rule's choice
DEFAULT_NORMAL = ACTIONS.index(Action(True, 0.08, False, False))
DEFAULT_HEAT = ACTIONS.index(Action(True, 0.08, True, True))


def decision(a: Action, obs: dict) -> WeeklyDecision:
    plan = obs["orders_7d"].astype(float)
    if a.weather_forecast:
        plan = plan * obs["demand_mult_now"] / np.maximum(obs["demand_mult_last"], 1e-9)
    plan = plan * (1 + a.margin)
    return WeeklyDecision(plan=plan,
                          channel_priority=[KIR, QC, MT] if a.kiranas_first else None,
                          twice_daily=a.twice_daily,
                          milk_priority=[MILK, CURD, PANEER, BUTTER] if a.kiranas_first else None)


# ------------------------------------------------------------------ policies
class EpsilonGreedy:
    """Used during training: random action with probability eps, else greedy. Records the
    (state, action) of every week so rewards can be attached after the year is run."""
    name = "Training"

    def __init__(self, Q: np.ndarray, eps: float, seed: int):
        self.Q, self.eps = Q, eps
        self.rng = np.random.default_rng(seed + 7_000_000)
        self.trace: list[tuple[int, int]] = []

    def weekly(self, obs: dict) -> WeeklyDecision:
        s = encode_state(obs)
        if self.rng.random() < self.eps:
            a = int(self.rng.integers(N_ACTIONS))
        else:
            a = int(np.argmax(self.Q[s]))
        self.trace.append((s, a))
        return decision(ACTIONS[a], obs)


class LearnedPolicy:
    """Greedy policy from a saved Q table."""
    name = "Learned"

    def __init__(self, path: str | Path = "results/q_table.npz"):
        z = np.load(path)
        self.Q, self.N = z["Q"], z["N"]
        self.unvisited_weeks = 0

    def action_index(self, obs: dict) -> int:
        s = encode_state(obs)
        if self.N[s].sum() == 0:
            self.unvisited_weeks += 1
            return DEFAULT_HEAT if obs["weather"] == 1 else DEFAULT_NORMAL
        q = np.where(self.N[s] > 0, self.Q[s], -np.inf)           # only trust tried actions
        return int(np.argmax(q))

    def weekly(self, obs: dict) -> WeeklyDecision:
        return decision(ACTIONS[self.action_index(obs)], obs)


# ------------------------------------------------------------------ reward
def weekly_rewards(out: dict, cfg: TwinConfig, lam: float, mu: float) -> np.ndarray:
    """Reward of each week, in lakh rupees: profit - lam x lost units - mu x wasted units."""
    W = out["demand"].shape[0] // 7
    cash = np.concatenate([[cfg.money.opening_cash], out["cash"]])
    profit = cash[7::7][:W] - cash[0:-1:7][:W]
    lost = (out["demand"] - out["sales"]).sum(axis=(1, 2))[: W * 7].reshape(W, 7).sum(1)
    waste = (out["waste_kirana"].sum(1) + out["waste_plant"].sum(1) + out["waste_mt"].sum(1)
             + out["waste_qc"].sum(1))[: W * 7].reshape(W, 7).sum(1)
    return (profit - lam * lost - mu * waste) / 1e5


# ------------------------------------------------------------------ training
def _episode(args):
    Q, eps, seed, lam, mu = args
    pol = EpsilonGreedy(Q, eps, seed)
    tw = DairyTwin(policy=pol, seed=seed)
    out = tw.run()
    r = weekly_rewards(out, tw.cfg, lam, mu)
    return pol.trace, r


def train(lam: float = 10.0, mu: float = 0.0, years: int = 3000, batch: int = 20, gamma: float = 0.95,
          eps_start: float = 0.5, eps_end: float = 0.05, workers: int = 2, seed0: int = 10_000,
          out_path: str | Path = "results/q_table.npz", log=print) -> dict:
    """Batched Q-learning: each round runs `batch` years in parallel with the current
    Q table (epsilon-greedy), then applies the Q-learning update to every transition."""
    Q = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros((N_STATES, N_ACTIONS), dtype=np.int64)
    rounds = years // batch
    curve = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k in range(rounds):
            eps = eps_start + (eps_end - eps_start) * k / max(rounds - 1, 1)
            jobs = [(Q, eps, seed0 + k * batch + i, lam, mu) for i in range(batch)]
            returns = []
            for trace, r in ex.map(_episode, jobs):
                returns.append(r.sum())
                for t, (s, a) in enumerate(trace):
                    N[s, a] += 1
                    alpha = max(1.0 / N[s, a], 0.02)
                    target = r[t] + (gamma * Q[trace[t + 1][0]].max() if t + 1 < len(trace) else 0.0)
                    Q[s, a] += alpha * (target - Q[s, a])
            curve.append(float(np.mean(returns)))
            if (k + 1) % 10 == 0:
                log(f"lambda={lam:g} round {k + 1}/{rounds} eps={eps:.2f} "
                    f"mean annual reward {np.mean(curve[-10:]):.1f} lakh  ({time.time() - t0:.0f}s)")
    np.savez(out_path, Q=Q, N=N, curve=np.array(curve), lam=lam, mu=mu, gamma=gamma, years=years)
    return {"Q": Q, "N": N, "curve": curve}


def policy_table(path: str | Path = "results/q_table.npz", min_visits: int = 20) -> list[dict]:
    """What the learned policy does in each state it saw often enough."""
    z = np.load(path)
    Q, N = z["Q"], z["N"]
    rows = []
    for s in range(N_STATES):
        n = int(N[s].sum())
        if n < min_visits:
            continue
        q = np.where(N[s] > 0, Q[s], -np.inf)
        a = int(np.argmax(q))
        rows.append({"state": describe_state(s), "visits": n, "action": ACTIONS[a].label(),
                     "action_fields": ACTIONS[a].__dict__})
    return rows
