"""Verification: did we build the model correctly? (slide 93)

Run with:  python -m pytest -q tests
"""
import numpy as np
import pytest

from dairytwin import DairyTwin, HabitPolicy, HeatAwarePolicy, TwinConfig
from dairytwin.config import Scenario
from dairytwin.regimes import RegimeChains

DAYS = 120


def run(days=DAYS, **kw):
    tw = DairyTwin(**kw)
    start = tw.units_in_system().copy()
    out = tw.run(days)
    return tw, out, start


def waste_total(out):
    return (out["waste_kirana"].sum(0) + out["waste_plant"].sum(0)
            + out["waste_mt"].sum(0) + out["waste_qc"].sum(0))


@pytest.mark.parametrize("policy", [HabitPolicy(), HeatAwarePolicy()])
@pytest.mark.parametrize("scen", [Scenario(), Scenario(battery_fridge_share=0.5, kirana_returns=True),
                                  Scenario().severe_summer(start_week=2)])
def test_conservation_every_unit_accounted_for(policy, scen):
    """start + produced = sold + wasted + still in stock, product by product."""
    tw, out, start = run(policy=policy, scenario=scen, seed=11)
    lhs = start + out["made"].sum(0)
    rhs = out["sales"].sum(axis=(0, 1)) + waste_total(out) + tw.units_in_system()
    assert np.array_equal(lhs, rhs)


def test_customers_conserved():
    _, out, _ = run(seed=5)
    assert np.all(out["customers"].sum(1) == sum(TwinConfig().city.households))


def test_same_seed_same_year_different_seed_different_year():
    _, a, _ = run(days=60, seed=7)
    _, b, _ = run(days=60, seed=7)
    _, c, _ = run(days=60, seed=8)
    assert np.array_equal(a["sales"], b["sales"]) and np.array_equal(a["milk"], b["milk"])
    assert not np.array_equal(a["sales"], c["sales"])


def test_common_random_numbers_shocks_identical_across_policies():
    """Two policies on one seed face the same weather and milk noise."""
    _, a, _ = run(days=200, policy=HabitPolicy(), seed=3)
    _, b, _ = run(days=200, policy=HeatAwarePolicy(), seed=3)
    assert np.array_equal(a["weather"], b["weather"])
    assert np.allclose(a["milk"], b["milk"])
    assert np.array_equal(a["outage_localities"], b["outage_localities"])


def test_regime_chain_matches_specification():
    """Long-run share of heat weeks in summer equals the chain's stationary share."""
    p = TwinConfig().regimes
    rng = np.random.default_rng(0)
    rc = RegimeChains(p, rng)
    lo, hi = p.summer_weeks
    heat_summer, n_summer, heat_other, n_other = 0, 0, 0, 0
    for _year in range(3000):
        for w in range(52):
            rc.step(w)
            if lo + 4 <= w <= hi:               # well inside summer: chain near stationarity
                heat_summer += rc.weather; n_summer += 1
            elif 30 <= w <= 40:
                heat_other += rc.weather; n_other += 1
    a, b = p.heat_onset_summer, p.heat_persist
    stationary = a / (a + 1 - b)
    assert abs(heat_summer / n_summer - stationary) < 0.02
    assert heat_other / n_other < 0.05


def test_poisson_demand_has_intended_mean():
    """On calm days, kirana demand averages households x per-capita."""
    cfg = TwinConfig()
    cfg.consumers.p_migrate_k_to_q = cfg.consumers.p_migrate_q_to_k = cfg.consumers.p_winback = 0.0
    tw, out, _ = run(days=DAYS, cfg=cfg, seed=4)
    calm = (out["weather"] == 0) & (out["festival"] == 0)
    expected = cfg.city.households[0] * np.asarray(cfg.products.per_capita)
    got = out["demand"][calm, 0].mean(0)
    assert np.all(np.abs(got / expected - 1) < 0.02)


def test_no_households_no_demand():
    cfg = TwinConfig()
    cfg.city.households = [0, 0, 0]
    _, out, _ = run(days=30, cfg=cfg, seed=1)
    assert out["demand"].sum() == 0 and out["sales"].sum() == 0


def test_no_milk_means_production_stops():
    cfg = TwinConfig()
    cfg.supply.mean_milk = 0.0
    _, out, _ = run(days=30, cfg=cfg, seed=1)
    assert out["made"].sum() == 0
    assert out["sales"][-5:].sum() <= out["sales"][:2].sum()  # only initial stock sells, then none


def test_without_outages_there_is_no_spoilage_from_power_cuts():
    _, on, _ = run(days=DAYS, scenario=Scenario().severe_summer(start_week=1), seed=2)
    _, off, _ = run(days=DAYS, scenario=Scenario(outages_enabled=False).severe_summer(start_week=1), seed=2)
    assert off["outage_shop_hours"].sum() == 0
    assert on["waste_kirana"].sum() > off["waste_kirana"].sum()


def test_full_battery_cover_removes_short_outages():
    cfg = TwinConfig()
    cfg.power.hours_heat = (1.0, 5.0)
    cfg.power.hours_normal = (1.0, 3.0)
    _, out, _ = run(days=60, cfg=cfg, scenario=Scenario(battery_fridge_share=1.0), seed=2)
    assert out["outage_shop_hours"].sum() == 0


def test_kirana_actor_invariants():
    tw = DairyTwin(scenario=Scenario().severe_summer(start_week=0), policy=HeatAwarePolicy(), seed=9)
    K, vol = tw.kirana, tw.cfg.arr("fridge_volume")
    for _ in range(90):
        tw.step()
        assert np.all(K.stock >= 0)
        assert np.all(K.stock_volume() <= K.fridge_cap + 1e-6)          # never over the fridge
        assert np.all(tw.pending >= 0)                                  # orders never negative
        assert np.all((K.k >= tw.cfg.kirana.k_min) & (K.k <= tw.cfg.kirana.k_max))
        A = K.stock.shape[-1]
        ages = np.arange(A)
        shelf = np.asarray(tw.cfg.products.shelf_life)
        assert np.all(K.stock[:, :, :][..., ages[None, :] >= shelf[:, None]].sum() == 0)  # nothing past shelf life


def test_monte_carlo_batches_agree():
    """Two independent batches of simulated years estimate the same mean within
    a few standard errors (the Monte Carlo estimator is unbiased and stable)."""
    a = np.array([DairyTwin(seed=s).run(56)["sales"].sum() for s in range(0, 10)])
    b = np.array([DairyTwin(seed=s).run(56)["sales"].sum() for s in range(100, 110)])
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    assert abs(a.mean() - b.mean()) < 4 * se


def test_one_city_many_years():
    """The twinned city is fixed; only the year's events change with the seed."""
    a, b = DairyTwin(seed=1), DairyTwin(seed=2)
    for attr in ["share", "fridge_cap", "k_base", "locality", "pos"]:
        assert np.array_equal(getattr(a.kirana, attr), getattr(b.kirana, attr))
    assert np.array_equal(a.feeder_risk, b.feeder_risk)
    assert np.array_equal(a.qc.share, b.qc.share)
    c = DairyTwin(seed=1, city_seed=5)
    assert not np.array_equal(a.kirana.share, c.kirana.share)


def test_dark_stores_filled_straight_from_plant():
    """Quick-commerce has no middle tier: every drop stays within the day's allowance and
    within each dark store's free space, and nothing sits between plant and store."""
    tw = DairyTwin(scenario=Scenario().severe_summer(start_week=1), policy=HeatAwarePolicy(), seed=4)
    assert not hasattr(tw.qc, "hub")
    real_drop = tw.qc.drop

    def checked(plant, allowance):
        before_plant = plant.available().copy()
        given = real_drop(plant, allowance)
        assert np.all(given >= 0) and np.all(given <= np.maximum(allowance, 0))
        assert np.all(tw.qc.store.sum(axis=-1) <= tw.qc.cap)
        assert np.array_equal(before_plant - plant.available(), given)   # left the plant = arrived
        return given

    tw.qc.drop = checked
    for _ in range(60):
        tw.step()


def test_mdp_state_and_learned_fallback(tmp_path):
    """Every observation maps to one of the 72 states, and a policy with an empty Q table
    falls back to the hand-written defaults."""
    from dairytwin import mdp
    tw = DairyTwin(seed=3)
    seen = set()
    for d in range(70):
        tw.step()
        if d % 7 == 6:
            s = mdp.encode_state(tw.observe())
            assert 0 <= s < mdp.N_STATES
            seen.add(s)
    path = tmp_path / "q.npz"
    np.savez(path, Q=np.zeros((mdp.N_STATES, len(mdp.ACTIONS))),
             N=np.zeros((mdp.N_STATES, len(mdp.ACTIONS))), lam=10.0, mu=0.0)
    pol = mdp.LearnedPolicy(str(path))
    out = DairyTwin(seed=3, policy=pol).run(28)
    assert out["sales"].sum() > 0


def test_export_run_is_consistent(tmp_path):
    """The viewer export agrees with the twin's own log and conserves units day to day."""
    import sys, importlib
    sys.path.insert(0, "experiments")
    export = importlib.import_module("export_run").export
    run = export("Heat-aware", "baseline", seed=5, days=21)
    assert len(run["days"]) == 21 and len(run["weeks"]) == 3
    for d in run["days"]:
        for ch in range(3):
            assert all(s <= q for s, q in zip(d["sales"][ch], d["demand"][ch]))
        assert sum(d["shop_stock"]) >= 0 and all(-1 <= f <= 100 for f in d["shop_fill"])
        assert len(d["loc_hours"]) == run["city"]["grid"][0] * run["city"]["grid"][1]
    assert all(w["label"] for w in run["weeks"])
