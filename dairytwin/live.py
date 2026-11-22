"""The twin as a mirror of a running system.

A *reality* (here a second simulated city that we can poke, standing in for the real
dairy network) runs day by day and emits a feed: observed weather and supply state,
power-cut hours per locality, footfall (demand) per channel, and stock readings from a
subset of shops with sensors. The *twin* is a separate DairyTwin with its own random
future. Each day it takes the feed in, steps, and is then corrected toward the readings.

    sync = "off"      the twin ignores the feed (an ordinary simulation)
    sync = "regimes"  it takes the observed regimes and power cuts, nothing else
    sync = "full"     plus demand level estimated from footfall, plus stock corrected
                      at the sensor shops and at the plant

From the synced state, `forecast` copies the twin and runs many futures for the next
days under each candidate policy: "what happens if I switch to X now?".

Everything is fake data for now. Replace `Reality` with a real feed reader and nothing
else changes.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .config import KIR, MT, QC, N_PRODUCTS
from .twin import DairyTwin, DayLog
from . import POLICIES
from . import mdp

KEEP_LOG = 30


def make_policy(name: str):
    if name == "Learned":
        return mdp.LearnedPolicy(str(Path(__file__).resolve().parents[1] / "results" / "q_table.npz"))
    return POLICIES[name]()


def available_policies() -> list[str]:
    names = list(POLICIES)
    if (Path(__file__).resolve().parents[1] / "results" / "q_table.npz").exists():
        names.append("Learned")
    return names


def trim_log(tw: DairyTwin, keep: int = KEEP_LOG):
    for k, v in tw.log.fields.items():
        if len(v) > keep:
            tw.log.fields[k] = v[-keep:]
    if len(tw.order_hist) > 40:
        tw.order_hist = tw.order_hist[-40:]


def jump_rngs(tw: DairyTwin, k: int):
    """Move every random stream of a copied twin to a different, reproducible future."""
    for g in tw.rng.values():
        g.bit_generator.state = g.bit_generator.jumped(k).state


class LiveMirror:
    def __init__(self, policy: str = "Habit", reality_seed: int = 101, twin_seed: int = 7,
                 city_seed: int = 0, sensor_share: float = 0.3, reading_noise: float = 0.05,
                 horizon: int = 14, futures: int = 8):
        self.policy_name, self.city_seed = policy, city_seed
        self.reality_seed, self.twin_seed = reality_seed, twin_seed
        self.sensor_share, self.reading_noise = sensor_share, reading_noise
        self.horizon, self.futures = horizon, futures
        self.sync = "full"
        self.reset()

    # ---------------------------------------------------------------- setup
    def reset(self):
        self.reality = DairyTwin(policy=make_policy(self.policy_name), seed=self.reality_seed,
                                 city_seed=self.city_seed)
        self.twin = DairyTwin(policy=make_policy(self.policy_name), seed=self.twin_seed,
                              city_seed=self.city_seed)
        self.reality.feed = {}
        self.twin.feed = {}
        n = self.reality.cfg.city.n_kirana
        r = np.random.default_rng(5)
        self.sensors = np.zeros(n, dtype=bool)
        self.sensors[r.choice(n, int(round(n * self.sensor_share)), replace=False)] = True
        self.noise_rng = np.random.default_rng(9)
        self.scale_true, self.scale_est = 1.0, 1.0
        self.timers: dict[str, int] = {}          # injection name -> days left
        self.hist: list[dict] = []
        self.last_shops: dict = {}
        self.forecast_result: dict | None = None

    # ---------------------------------------------------------------- injections into reality
    def inject(self, kind: str, value=None, days: int = 7, locality: int | None = None):
        R = self.reality
        if kind == "heat":
            R.feed["weather"] = 1
            self.timers["heat"] = days
        elif kind == "stress":
            R.feed["supply"] = 1
            self.timers["stress"] = days
        elif kind == "festival":
            R.feed["festival"] = 1
            self.timers["festival"] = days
        elif kind == "demand":
            self.scale_true = float(value)
            R.feed["demand_scale"] = self.scale_true
            self.timers["demand"] = days
        elif kind == "cut":
            loc = int(locality if locality is not None else R.rng["power"].integers(0, R.n_loc))
            for d in range(R.day, R.day + max(1, days)):
                R.scen.forced_outages.setdefault(d, {})[loc] = float(value or 6)
        elif kind == "clear":
            self.timers = {k: 0 for k in self.timers}
            self._expire()

    def _expire(self):
        R = self.reality
        names = {"heat": "weather", "stress": "supply", "festival": "festival"}
        for name, left in list(self.timers.items()):
            if left > 0:
                continue
            del self.timers[name]
            if name in names:
                R.feed.pop(names[name], None)
                setattr(R.regimes, names[name], 0)      # the shock ends now, not at the next draw
            elif name == "demand":
                R.feed.pop("demand_scale", None)
                self.scale_true = 1.0

    # ---------------------------------------------------------------- the daily loop
    def step(self) -> dict:
        R, T, K = self.reality, self.twin, self.twin.kirana
        R.step()
        for name in self.timers:
            self.timers[name] -= 1
        self._expire()
        Rk = R.kirana
        dem_r = R.log.fields["demand"][-1]
        sal_r = R.log.fields["sales"][-1]
        hours = R.last_outage_hours.copy()

        feed = {}
        if self.sync in ("regimes", "full"):
            feed.update(weather=R.regimes.weather, supply=R.regimes.supply,
                        festival=R.regimes.festival, outage_hours=hours)
            if R.day % 7 == 1:                        # Monday's plan is known to the operator
                feed["decision"] = R.decision
        if self.sync == "full":
            feed["demand_scale"] = self.scale_est
        T.feed = feed
        T.step()
        dem_t = T.log.fields["demand"][-1]
        sal_t = T.log.fields["sales"][-1]
        pred_total, pred_scale = float(dem_t.sum()), self.scale_est

        # assimilation: footfall -> demand level, readings -> stock
        if self.sync == "full":
            a = 0.25
            ratio = float(dem_r.sum()) / max(pred_total, 1.0)
            self.scale_est = float(np.clip(pred_scale * ((1 - a) + a * ratio), 0.3, 3.0))
            self._assimilate_stock()
        trim_log(R)
        trim_log(T)

        sr, st = Rk.stock.sum(-1), K.stock.sum(-1)                  # (n, P) units on shelf
        err = float(np.abs(st - sr).sum() / max((sr.sum() + st.sum()) / 2, 1))
        fill = lambda s, d: float(s[KIR].sum() / max(d[KIR].sum(), 1))
        rec = {
            "day": R.day, "weather_r": int(R.regimes.weather), "weather_t": int(T.regimes.weather),
            "supply_r": int(R.regimes.supply), "supply_t": int(T.regimes.supply),
            "festival_r": int(R.regimes.festival), "festival_t": int(T.regimes.festival),
            "cuts": int((hours > 0).sum()),
            "fill_r": fill(sal_r, dem_r), "fill_t": fill(sal_t, dem_t),
            "demand_r": dem_r.sum(1).astype(int).tolist(), "demand_t": dem_t.sum(1).astype(int).tolist(),
            "sales_r": sal_r.sum(1).astype(int).tolist(), "sales_t": sal_t.sum(1).astype(int).tolist(),
            "stock_err": err, "scale_true": self.scale_true, "scale_est": self.scale_est,
            "plant_r": int(R.plant.stock.sum()), "plant_t": int(T.plant.stock.sum()),
            "cash_r": round(float(R.cash)), "cash_t": round(float(T.cash)),
            "timers": dict(self.timers),
        }
        self.hist.append(rec)
        if len(self.hist) > 400:
            self.hist = self.hist[-400:]

        dr, ds = Rk.today_demand.sum(1), Rk.today_sales.sum(1)
        dt, dst = K.today_demand.sum(1), K.today_sales.sum(1)
        self.last_shops = {
            "fill_r": np.where(dr > 0, np.round(100 * ds / np.maximum(dr, 1)), -1).astype(int).tolist(),
            "fill_t": np.where(dt > 0, np.round(100 * dst / np.maximum(dt, 1)), -1).astype(int).tolist(),
            "stock_r": sr.sum(1).astype(int).tolist(), "stock_t": st.sum(1).astype(int).tolist(),
            "hours": np.round(hours, 1).tolist(),
        }
        return rec

    def _assimilate_stock(self):
        R, T = self.reality, self.twin
        P = N_PRODUCTS
        read = R.kirana.stock.sum(-1)                               # true stock, (n, P)
        noisy = np.maximum(0, np.round(read * (1 + self.noise_rng.normal(0, self.reading_noise, read.shape))))
        cur = T.kirana.stock.sum(-1)
        for i in np.where(self.sensors)[0]:
            for p in range(P):
                tgt = int(noisy[i, p])
                if cur[i, p] > 0:
                    prof = T.kirana.stock[i, p].astype(float)
                    new = np.floor(prof * tgt / cur[i, p]).astype(np.int64)
                    new[int(np.argmax(prof))] += tgt - int(new.sum())
                    T.kirana.stock[i, p] = np.maximum(new, 0)
                else:
                    T.kirana.stock[i, p] = 0
                    T.kirana.stock[i, p, 0] = tgt
        pr = R.plant.stock.sum(-1)
        pc = T.plant.stock.sum(-1)
        for p in range(P):
            tgt = int(round(pr[p] * (1 + self.noise_rng.normal(0, self.reading_noise))))
            if pc[p] > 0:
                prof = T.plant.stock[p].astype(float)
                new = np.floor(prof * tgt / pc[p]).astype(np.int64)
                new[int(np.argmax(prof))] += tgt - int(new.sum())
                T.plant.stock[p] = np.maximum(new, 0)

    # ---------------------------------------------------------------- forecast
    def forecast(self, policies: list[str] | None = None) -> dict:
        """From today's synced twin, run `futures` futures for `horizon` days under each policy.
        Every policy meets the same futures (common random numbers)."""
        policies = policies or available_policies()
        out = {"day": self.twin.day, "horizon": self.horizon, "futures": self.futures, "policies": {}}
        base = copy.deepcopy(self.twin)
        trim_log(base, 7)
        for name in policies:
            fills, lost, waste, cash = [], [], [], []
            for k in range(self.futures):
                c = copy.deepcopy(base)
                jump_rngs(c, k + 1 + self.twin.day * 17)
                c.feed = {"demand_scale": self.scale_est} if self.sync == "full" else {}
                c.policy = make_policy(name)
                c.decision = c.policy.weekly(c.observe())      # switch takes effect today
                c0 = c.cash
                f = []
                for _ in range(self.horizon):
                    c.step()
                    d, s = c.log.fields["demand"][-1], c.log.fields["sales"][-1]
                    f.append(float(s[KIR].sum() / max(d[KIR].sum(), 1)))
                dd = sum(x.sum() for x in c.log.fields["demand"][-self.horizon:])
                ss = sum(x.sum() for x in c.log.fields["sales"][-self.horizon:])
                w = sum(c.log.fields[k][-self.horizon:][i].sum() for k in
                        ("waste_kirana", "waste_plant", "waste_mt", "waste_qc") for i in range(self.horizon))
                fills.append(f); lost.append(float(dd - ss)); waste.append(float(w)); cash.append(c.cash - c0)
            F = np.array(fills)
            out["policies"][name] = {
                "fill_mean": np.round(F.mean(0), 4).tolist(),
                "fill_lo": np.round(np.percentile(F, 10, axis=0), 4).tolist(),
                "fill_hi": np.round(np.percentile(F, 90, axis=0), 4).tolist(),
                "lost": round(float(np.mean(lost))), "waste": round(float(np.mean(waste))),
                "profit": round(float(np.mean(cash))),
            }
        self.forecast_result = out
        return out

    # ---------------------------------------------------------------- views
    def static(self) -> dict:
        R = self.reality
        cfg = R.cfg
        return {
            "shop_pos": np.round(R.kirana.pos, 3).tolist(),
            "shop_locality": R.kirana.locality.astype(int).tolist(),
            "grid": list(cfg.city.grid), "n_loc": R.n_loc,
            "sensors": self.sensors.astype(int).tolist(),
            "policies": available_policies(),
        }

    def state(self, tail: int = 90) -> dict:
        return {
            "day": self.reality.day, "sync": self.sync, "policy": self.policy_name,
            "timers": dict(self.timers), "scale_true": self.scale_true, "scale_est": self.scale_est,
            "hist": self.hist[-tail:], "shops": self.last_shops, "forecast": self.forecast_result,
        }
