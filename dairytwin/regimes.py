"""Slow, persistent randomness: weekly Markov regime chains.

Three chains step once a week:

* weather   Normal <-> Heat wave   (shared cause: lowers milk supply AND raises curd demand)
* supply    Normal <-> Stressed    (disease, fodder shortage)
* festival  None   <-> Festival    (only likely inside calendar windows: Diwali, weddings)

Their transition probabilities depend on the calendar week, so a heat wave is likely
in April-June and almost impossible in December. Persistence is the point: a three-week
heat wave hurts far more than three scattered hot days, and that is a property of the
chain, not of any average.
"""
from __future__ import annotations

import numpy as np

from .config import RegimeParams


def _in_range(week: int, lo: int, hi: int) -> bool:
    return lo <= week <= hi


class RegimeChains:
    def __init__(self, params: RegimeParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self.weather = 0
        self.supply = 0
        self.festival = 0

    # --- transition probabilities (one row of each matrix P_k) -------------
    def p_heat_next(self, week: int) -> float:
        lo, hi = self.p.summer_weeks
        if self.weather == 1:
            return self.p.heat_persist if _in_range(week, lo, hi + 2) else 0.25
        return self.p.heat_onset_summer if _in_range(week, lo, hi) else self.p.heat_onset_other

    def p_stress_next(self) -> float:
        return self.p.stress_persist if self.supply == 1 else self.p.stress_onset

    def in_festival_window(self, week: int) -> bool:
        return any(_in_range(week, lo, hi) for lo, hi in self.p.festival_windows)

    def p_festival_next(self, week: int) -> float:
        if not self.in_festival_window(week):
            return 0.0
        if self.festival == 1:
            return self.p.festival_persist_in_window
        return self.p.festival_onset_in_window

    def is_flush(self, week: int) -> bool:
        a, b, c, d = self.p.flush_weeks
        return week >= a or week <= d

    def is_lean(self, week: int) -> bool:
        lo, hi = self.p.summer_weeks
        return lo <= week <= hi

    # --- one weekly step ---------------------------------------------------
    def step(self, week: int, forced_weather: int | None = None) -> None:
        """Draw next week's regimes. Uniform draws are always consumed in the same
        order, so forcing the weather leaves the other chains on the same random path
        (common random numbers across scenarios)."""
        u = self.rng.random(3)
        heat = int(u[0] < self.p_heat_next(week))
        self.weather = heat if forced_weather is None else int(forced_weather)
        self.supply = int(u[1] < self.p_stress_next())
        self.festival = int(u[2] < self.p_festival_next(week))

    def as_tuple(self):
        return self.weather, self.supply, self.festival
