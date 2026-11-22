"""Export one simulated year, day by day, for the twin viewer (web/twin_viewer.html).

    python experiments/export_run.py --policy Learned --scenario severe --seed 3 --out web/runs/learned_severe.json

The file holds everything the viewer draws: the fixed city, each day's regimes, flows between
tiers, stock held at every tier, every shop's stock and fill, power cuts by locality, waste by
tier, customers, cash, and the plan the policy chose each Monday. Drag the file onto the viewer
to replay it. Change the twin, re-export, and the picture follows.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dairytwin import DairyTwin, POLICIES
from dairytwin.config import Scenario, TwinConfig, PRODUCTS, N_PRODUCTS, KIR, MT, QC
from dairytwin import mdp

SCENARIOS = {
    "baseline": lambda: Scenario(name="Baseline"),
    "severe": lambda: Scenario(name="Severe summer").severe_summer(),
    "batteries": lambda: Scenario(name="Battery fridges, risk-targeted", battery_fridge_share=0.3),
    "batteries_random": lambda: Scenario(name="Battery fridges, random", battery_fridge_share=0.3,
                                         battery_targeting="random"),
    "returns": lambda: Scenario(name="Kirana returns", kirana_returns=True),
    "nocut": lambda: Scenario(name="No power cuts", outages_enabled=False),
    "severe_batteries": lambda: Scenario(name="Severe summer + batteries", battery_fridge_share=0.3).severe_summer(),
    "severe_nocut": lambda: Scenario(name="Severe summer, no power cuts", outages_enabled=False).severe_summer(),
}


def make_policy(name: str):
    if name.startswith("Learned"):
        path = name.split(":", 1)[1] if ":" in name else "results/q_table.npz"
        return mdp.LearnedPolicy(path)
    return POLICIES[name]()


def decision_label(policy, obs, dec) -> tuple[str, dict | None]:
    if isinstance(policy, mdp.LearnedPolicy):
        a = mdp.ACTIONS[policy.action_index(obs)]
        return a.label(), mdp.describe_state(mdp.encode_state(obs))
    forecast = "habit" if policy.name == "Habit" else "weather"
    margin = int(round(getattr(policy, "margin", 0.05) * 100))
    prio = "kiranas & staples first" if dec.channel_priority else "pro-rata"
    drops = "2 drops/day" if dec.twice_daily else "1 drop/day"
    return f"{forecast} forecast +{margin}%, {prio}, {drops}", None


def export(policy_name: str, scenario_key: str, seed: int, days: int | None, city_seed: int = 0) -> dict:
    cfg = TwinConfig()
    scen = SCENARIOS[scenario_key]()
    policy = make_policy(policy_name)
    tw = DairyTwin(cfg=cfg, scenario=scen, policy=policy, seed=seed, city_seed=city_seed)
    days = days or cfg.days
    n_loc = tw.n_loc

    # capture the hours of power cut per locality, which the twin uses but does not log
    captured = {}
    orig_outages = tw.outages
    def outages():
        h = orig_outages(); captured["hours"] = h; return h
    tw.outages = outages

    # capture each Monday's decision with the observation it was made from
    weeks = []
    orig_weekly = policy.weekly
    def weekly(obs):
        dec = orig_weekly(obs)
        label, state = decision_label(policy, obs, dec)
        weeks.append({"week": int(obs["week"]), "label": label, "state": state,
                      "plan": [int(round(x)) for x in dec.plan],
                      "kiranas_first": bool(dec.channel_priority), "twice_daily": bool(dec.twice_daily),
                      "staples_first": bool(dec.milk_priority),
                      "kirana_fill_7d": round(float(obs["kirana_fill_7d"]), 4)})
        return dec
    policy.weekly = weekly

    K = tw.kirana
    daily = []
    for _ in range(days):
        tw.step()
        f = {k: v[-1] for k, v in tw.log.fields.items()}
        plant = tw.plant.stock
        shop_stock = K.stock.sum(axis=(1, 2))
        dem, sal = K.today_demand.sum(1), K.today_sales.sum(1)
        fill = np.where(dem > 0, np.round(100 * sal / np.maximum(dem, 1)), -1).astype(int)
        daily.append({
            "weather": int(f["weather"]), "supply": int(f["supply"]), "festival": int(f["festival"]),
            "flush": bool(tw.regimes.is_flush(tw.day // 7)), "lean": bool(tw.regimes.is_lean(tw.day // 7)),
            "milk": int(f["milk"]), "milk_used": int(f["milk_used"]),
            "made": f["made"].astype(int).tolist(),
            "orders": f["orders"].astype(int).tolist(), "shipped": f["shipped"].astype(int).tolist(),
            "demand": f["demand"].astype(int).tolist(), "sales": f["sales"].astype(int).tolist(),
            "spill": f["spill"].astype(int).tolist(), "refused": f["refused"].astype(int).tolist(),
            "late_shops": int(f["late_shops"]),
            "waste": {"plant": f["waste_plant"].astype(int).tolist(), "kirana": f["waste_kirana"].astype(int).tolist(),
                      "returned": f["waste_kirana_returned"].astype(int).tolist(),
                      "mt": f["waste_mt"].astype(int).tolist(), "qc": f["waste_qc"].astype(int).tolist()},
            "plant_stock": plant.sum(-1).astype(int).tolist(),
            "plant_fresh": plant[:, :2].sum(-1).astype(int).tolist(),
            "mt_dc": tw.mt.dc.sum(-1).astype(int).tolist(), "mt_shelf": tw.mt.shelf.sum(-1).astype(int).tolist(),
            "qc_store": tw.qc.store.sum(axis=(1, 2)).astype(int).tolist(),
            "qc_store_p": tw.qc.store.sum(axis=(0, 2)).astype(int).tolist(),
            "shop_stock": shop_stock.astype(int).tolist(), "shop_fill": fill.tolist(),
            "shop_spoiled": K.today_spoiled.sum(1).astype(int).tolist(),
            "shop_k": np.round(K.k * 100).astype(int).tolist(),
            "loc_hours": np.round(captured.get("hours", np.zeros(n_loc)), 1).tolist(),
            "customers": f["customers"].astype(int).tolist(), "migrated": int(f["migrated"]),
            "cash": round(float(f["cash"])), "revenue": round(float(f["revenue"])), "cost": round(float(f["cost"])),
            "twice_daily": bool(f["twice_daily"]),
        })

    gx, gy = cfg.city.grid
    return {
        "meta": {"policy": policy_name, "scenario": scen.name, "scenario_key": scenario_key, "seed": seed,
                 "city_seed": city_seed, "days": days,
                 "battery_share": scen.battery_fridge_share, "battery_targeting": scen.battery_targeting,
                 "kirana_returns": scen.kirana_returns, "outages_enabled": scen.outages_enabled,
                 "forced_heat_weeks": sorted(int(w) for w, s in scen.forced_weather.items() if s == 1)},
        "city": {"products": PRODUCTS, "shelf_life": list(cfg.products.shelf_life), "grid": [gx, gy],
                 "n_routes": cfg.city.n_routes, "n_dark_stores": cfg.city.n_dark_stores,
                 "shop_pos": np.round(K.pos, 3).tolist(), "shop_locality": K.locality.astype(int).tolist(),
                 "shop_route": K.route.astype(int).tolist(), "shop_battery": K.battery.astype(bool).tolist(),
                 "shop_fridge_cap": np.round(K.fridge_cap).astype(int).tolist(),
                 "feeder_risk": np.round(tw.feeder_risk, 2).tolist(),
                 "plant_capacity": tw.plant.capacity.astype(int).tolist(),
                 "mt_shelf_cap": tw.mt.shelf_cap.astype(int).tolist(),
                 "qc_cap": tw.qc.cap.sum(1).astype(int).tolist(),
                 "households0": list(cfg.city.households), "opening_cash": cfg.money.opening_cash,
                 "retail_price": list(cfg.products.retail_price)},
        "weeks": weeks, "days": daily,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="Heat-aware", help="Habit | Heat-aware | Learned | Learned:path.npz")
    ap.add_argument("--scenario", default="baseline", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = export(a.policy, a.scenario, a.seed, a.days)
    out = Path(a.out or f"web/runs/{a.policy.split(':')[0].lower().replace('-', '')}_{a.scenario}_s{a.seed}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, separators=(",", ":")))
    print(out, round(out.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
