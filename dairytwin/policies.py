"""Fixed decision policies (the weekly control knobs).

The proposal's knobs, set once a week from the fully observed state:

* production plan per product (units/day for the coming week)
* allocation rule when the plant cannot fill every channel's order
* kirana delivery frequency (1 or 2 drops a day)
* milk priority when supply is short

No learning yet: policies are fixed rules compared as scenarios under common random
numbers. A learned policy (MDP) can later replace `weekly()` without touching the twin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CURD, KIR, MILK, MT, N_PRODUCTS, PANEER, BUTTER, QC


@dataclass
class WeeklyDecision:
    plan: np.ndarray                  # (P,) plant stock to hold after production each day (units)
    channel_priority: list | None     # None = pro rata to orders; else serve in this order
    twice_daily: bool                 # kirana noon drop
    milk_priority: list | None        # None = scale all down; else staples first


def allocate(avail: np.ndarray, orders: np.ndarray, priority: list | None) -> np.ndarray:
    """Split avail (P,) among channel orders (C, P). Returns shipped (C, P)."""
    C = orders.shape[0]
    out = np.zeros_like(orders)
    tot = orders.sum(axis=0)
    for p in range(orders.shape[1]):
        if tot[p] <= avail[p]:
            out[:, p] = orders[:, p]
        elif priority is None:
            share = np.floor(orders[:, p] * avail[p] / max(tot[p], 1)).astype(np.int64)
            out[:, p] = share
        else:
            left = avail[p]
            for c in priority:
                g = min(orders[c, p], left)
                out[c, p] = g
                left -= g
    return out


class HabitPolicy:
    """Rule of thumb: produce last week's average orders plus a margin, share shortages
    pro rata, deliver to kiranas once a day, scale every product down when milk is short."""
    name = "Habit"

    def __init__(self, margin: float = 0.05):
        self.margin = margin

    def weekly(self, obs: dict) -> WeeklyDecision:
        plan = obs["orders_7d"] * (1 + self.margin)
        return WeeklyDecision(plan=plan, channel_priority=None, twice_daily=False,
                              milk_priority=None)


class HeatAwarePolicy:
    """Reads the weather regime (fully observed) and acts on it the week it starts:

    * scales the forecast by the demand multipliers of the new regime instead of
      waiting for orders to catch up,
    * in a heat wave: noon drop to kiranas, kiranas served first on shortage
      (they hold most customers and spill disappointed buyers onto tiny dark stores),
    * when milk is short: staples first (milk, curd, then paneer, butter)."""
    name = "Heat-aware"

    def __init__(self, margin: float = 0.08):
        self.margin = margin

    def weekly(self, obs: dict) -> WeeklyDecision:
        mult_now, mult_last = obs["demand_mult_now"], obs["demand_mult_last"]
        plan = obs["orders_7d"] * (mult_now / np.maximum(mult_last, 1e-9)) * (1 + self.margin)
        heat = obs["weather"] == 1
        return WeeklyDecision(
            plan=plan,
            channel_priority=[KIR, QC, MT] if heat else None,
            twice_daily=heat,
            milk_priority=[MILK, CURD, PANEER, BUTTER],
        )


POLICIES = {"Habit": HabitPolicy, "Heat-aware": HeatAwarePolicy}
