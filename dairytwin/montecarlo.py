"""Monte Carlo: every result is a distribution over simulated years.

`run_batch` runs one (policy, scenario) for seeds 0..N-1. Seeds are shared across
experiments, so experiment A and B face the same weather, milk noise, outages and late
vans in year i (common random numbers): differences come from the decision, not luck.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

from .config import KIR, MT, N_PRODUCTS, QC, Scenario, TwinConfig
from .policies import POLICIES
from .mdp import weekly_rewards
from .twin import DairyTwin


def weekly(x: np.ndarray, days: int) -> np.ndarray:
    w = days // 7
    return x[: w * 7].reshape(w, 7, *x.shape[1:]).sum(axis=1)


def predicate_weeks(out: dict, cfg: TwinConfig) -> dict:
    """Evaluate the weekly goal predicate (slide 26) on one simulated year.

    P_w = AND_{p,ch} rolling-4-week fill >= phi_ch  AND  AND_p waste share <= cap_p  AND  cash >= 0
    Returns which conjuncts held each week (weeks with a full 4-week window only)."""
    T = cfg.targets
    days = out["demand"].shape[0]
    d, s = weekly(out["demand"], days), weekly(out["sales"], days)     # (W, C, P)
    waste = (weekly(out["waste_kirana"], days) + weekly(out["waste_plant"], days)
             + weekly(out["waste_mt"], days) + weekly(out["waste_qc"], days))
    made = weekly(out["made"], days)
    W, k = d.shape[0], T.window_weeks
    fill_ok, waste_ok, cash_ok = [], [], []
    cash_week = out["cash"][6::7][:W]
    for w in range(k - 1, W):
        dd, ss = d[w - k + 1: w + 1].sum(0), s[w - k + 1: w + 1].sum(0)
        fr = ss / np.maximum(dd, 1)
        fill_ok.append(fr >= np.asarray(T.fill_rate)[:, None])              # (C, P)
        ws = waste[w - k + 1: w + 1].sum(0) / np.maximum(made[w - k + 1: w + 1].sum(0), 1)
        waste_ok.append(ws <= np.asarray(T.waste_cap))
        cash_ok.append(cash_week[w] >= 0)
    fill_ok, waste_ok, cash_ok = np.array(fill_ok), np.array(waste_ok), np.array(cash_ok)
    holds = fill_ok.all(axis=(1, 2)) & waste_ok.all(axis=1) & cash_ok
    return {"holds": holds, "fill_ok": fill_ok, "waste_ok": waste_ok, "cash_ok": cash_ok}


def behaviour_goals(out: dict, cfg: TwinConfig) -> dict:
    """Value-and-behaviour goals: conditions on the pattern of the weekly trace, not on a
    single week's value.

    recovery   after every heat-wave spell ends, kirana weekly fill is back at its target
               within `recovery_weeks` weeks
    no_collapse no channel's weekly fill stays below `collapse_fill` for more than
               `collapse_max_run` weeks in a row
    fair_waste over the year, no city sector carries more than `max_sector_waste_share`
               of all kirana waste
    """
    T = cfg.targets
    days = out["demand"].shape[0]
    wd, ws = weekly(out["demand"], days), weekly(out["sales"], days)
    fill_ch = ws.sum(axis=2) / np.maximum(wd.sum(axis=2), 1)                 # (W, C)
    heat = weekly((out["weather"] == 1).astype(int), days) > 0
    W = heat.size

    # recovery after each heat spell
    ends = [w for w in range(W - 1) if heat[w] and not heat[w + 1]]
    rec_weeks = []
    for w in ends:
        k = next((j for j in range(1, W - w) if fill_ch[w + j, KIR] >= T.fill_rate[KIR]), None)
        rec_weeks.append(np.inf if k is None else k)
    recovery = all(k <= T.recovery_weeks for k in rec_weeks)

    # longest run of weeks below the collapse line, per channel
    longest = 0
    for c in range(fill_ch.shape[1]):
        run = 0
        for v in fill_ch[:, c]:
            run = run + 1 if v < T.collapse_fill else 0
            longest = max(longest, run)
    no_collapse = longest <= T.collapse_max_run

    # waste fairness across sectors
    loc = out["shop_locality"]
    sector_waste = np.bincount(loc, weights=out["shop_waste"].sum(1), minlength=loc.max() + 1)
    max_share = float(sector_waste.max() / max(sector_waste.sum(), 1))
    fair_waste = max_share <= T.max_sector_waste_share

    return {"recovery": bool(recovery), "recovery_weeks": [float(k) for k in rec_weeks],
            "no_collapse": bool(no_collapse), "longest_collapse_run": int(longest),
            "fair_waste": bool(fair_waste), "max_sector_waste_share": max_share,
            "all": bool(recovery and no_collapse and fair_waste)}


def summarise(out: dict, cfg: TwinConfig) -> dict:
    """Compact per-year record: weekly series + annual scalars."""
    days = out["demand"].shape[0]
    d, s = out["demand"], out["sales"]
    heat = out["weather"] == 1
    waste_k = out["waste_kirana"].sum(1)
    waste_all = waste_k + out["waste_plant"].sum(1) + out["waste_mt"].sum(1) + out["waste_qc"].sum(1)
    r = cfg.arr("milk_per_unit")
    pred = predicate_weeks(out, cfg)
    # bullwhip: variance of weekly kirana orders over variance of weekly kirana demand (milk)
    wo, wdm = weekly(out["kirana_orders"][:, 0], days).astype(float), weekly(d[:, KIR, 0], days).astype(float)
    bullwhip = wo.var() / max(wdm.var(), 1e-9)
    wd, ws = weekly(d, days), weekly(s, days)
    return {
        # weekly series (W, ...)
        "w_fill": ws.sum(axis=(1, 2)) / np.maximum(wd.sum(axis=(1, 2)), 1),
        "w_fill_ch": ws.sum(axis=2) / np.maximum(wd.sum(axis=2), 1),            # (W, C)
        "w_fill_curd_kir": ws[:, KIR, 1] / np.maximum(wd[:, KIR, 1], 1),
        "w_waste": weekly(waste_all, days),
        "w_waste_kirana": weekly(waste_k, days),
        "w_lost": (wd - ws).sum(axis=(1, 2)),
        "w_milk_supply": weekly(out["milk"], days),
        "w_milk_demand": (wd.sum(axis=1) * r[None, :]).sum(axis=1),
        "w_heat": weekly(heat.astype(int), days) > 0,
        "w_festival": weekly(out["festival"], days) > 0,
        "w_outage_hours": weekly(out["outage_shop_hours"], days),
        "w_customers_k": out["customers"][6::7, KIR][: days // 7],
        "w_kirana_orders_milk": weekly(out["kirana_orders"][:, 0], days),
        "w_kirana_demand_milk": weekly(d[:, KIR, 0], days),
        "w_predicate": pred["holds"],
        # annual scalars
        "fill": s.sum() / d.sum(),
        "fill_cp": s.sum(0) / np.maximum(d.sum(0), 1),                           # (C, P)
        "fill_heat": s[heat].sum() / max(d[heat].sum(), 1) if heat.any() else np.nan,
        "worst_week_fill": (ws.sum(axis=(1, 2)) / np.maximum(wd.sum(axis=(1, 2)), 1)).min(),
        "lost": int((d - s).sum()),
        "waste_share": waste_all.sum() / max(out["made"].sum(), 1),
        "waste_units": int(waste_all.sum()),
        "waste_by_tier": np.array([out["waste_kirana"].sum() - out["waste_kirana_returned"].sum(),
                                   out["waste_kirana_returned"].sum(), out["waste_plant"].sum(),
                                   out["waste_mt"].sum(), out["waste_qc"].sum()]),
        "waste_by_product": waste_all_by_product(out),
        "kirana_waste_heat_share": waste_k[heat].sum() / max(waste_k.sum(), 1),
        "heat_weeks": int(weekly(heat.astype(int), days).astype(bool).sum()),
        "outage_shop_hours": float(out["outage_shop_hours"].sum()),
        "bullwhip": float(bullwhip),
        "migrated_net": int(cfg.city.households[KIR] - out["customers"][-1, KIR]),
        "profit": float(out["cash"][-1] - cfg.money.opening_cash),
        "predicate_share": float(pred["holds"].mean()),
        "predicate_year": bool(pred["holds"].all()),
        "behaviour": behaviour_goals(out, cfg),
        "reward_lakh": {lam: float(weekly_rewards(out, cfg, lam, 0.0).sum()) for lam in (0.0, 10.0, 30.0)},
        "fill_conjunct_fail": 1 - pred["fill_ok"].mean(axis=0),                   # (C, P) share of weeks failing
        "waste_conjunct_fail": 1 - pred["waste_ok"].mean(axis=0),                 # (P,)
        "shop_waste_rate": out["shop_waste"].sum(1) / np.maximum(out["shop_demand"].sum(1), 1),
        "shop_lost_rate": out["shop_lost"].sum(1) / np.maximum(out["shop_demand"].sum(1), 1),
        "shop_outage_hours": out["shop_outage_hours"],
    }


def waste_all_by_product(out):
    return (out["waste_kirana"].sum(0) + out["waste_plant"].sum(0)
            + out["waste_mt"].sum(0) + out["waste_qc"].sum(0))


def _one(args):
    policy_name, scen_dict, seed, cfg = args
    scen = Scenario(**scen_dict)
    if policy_name.startswith("Learned"):
        from .mdp import LearnedPolicy
        path = policy_name.split(":", 1)[1] if ":" in policy_name else "results/q_table.npz"
        policy = LearnedPolicy(path)
    else:
        policy = POLICIES[policy_name]()
    tw = DairyTwin(cfg=cfg, scenario=scen, policy=policy, seed=seed)
    out = tw.run()
    rec = summarise(out, cfg)
    if hasattr(policy, "unvisited_weeks"):
        rec["unvisited_weeks"] = policy.unvisited_weeks
    if seed == 0:
        rec["map"] = {"pos": out["shop_pos"], "locality": out["shop_locality"],
                      "battery": out["shop_battery"], "feeder_risk": out["feeder_risk"]}
    return rec


def run_batch(policy_name: str, scen: Scenario, n: int, cfg: TwinConfig | None = None,
              workers: int = 2) -> list[dict]:
    cfg = cfg or TwinConfig()
    jobs = [(policy_name, asdict(scen), s, cfg) for s in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, jobs, chunksize=4))


def stack(records: list[dict], key: str) -> np.ndarray:
    return np.stack([np.asarray(r[key]) for r in records])
