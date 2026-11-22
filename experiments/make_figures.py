"""Figures and summary tables from the Monte Carlo results.

    python experiments/make_figures.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dairytwin.config import CHANNELS, PRODUCTS, TwinConfig  # noqa: E402
from dairytwin.montecarlo import stack  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# reference palette (dataviz skill): categorical slots in fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
POLICY_COLOR = {"habit": BLUE, "heat": VIOLET, "learned": AQUA}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def load():
    with open(ROOT / "results" / "mc.pkl", "rb") as f:
        return pickle.load(f)


def q(x, axis=0):
    return np.nanpercentile(x, [5, 50, 95], axis=axis)


def fan(ax, x, data, color, label):
    lo, med, hi = q(data)
    ax.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0)
    ax.plot(x, med, color=color, label=label)


def shade_heat(ax, recs):
    p = stack(recs, "w_heat").mean(0)
    for w in range(p.size):
        if p[w] > 0.02:
            ax.axvspan(w - 0.5, w + 0.5, color=ORANGE, alpha=0.10 * min(1, p[w] / 0.6), linewidth=0)


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def beh(R, key):
    return np.array([r["behaviour"][key] for r in R])


# ------------------------------------------------------------------ figures
def fig_year(E):
    """Weekly fan charts over the year: milk squeeze, fill, waste."""
    h, a = E["habit"]["records"], E["heat"]["records"]
    W = stack(h, "w_fill").shape[1]
    x = np.arange(W)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)

    ax = axes[0]
    shade_heat(ax, h)
    ratio = stack(h, "w_milk_supply") / stack(h, "w_milk_demand")
    fan(ax, x, ratio, BLUE, "Milk reaching the plant ÷ milk needed by consumer demand")
    ax.axhline(1, color=INK2, linewidth=1)
    ax.set_title("Milk runs short in the lean summer weeks, and heat waves deepen the gap")
    ax.set_ylabel("Supply ÷ demand")
    ax.legend(loc="lower left")

    ax = axes[1]
    shade_heat(ax, h)
    fan(ax, x, stack(h, "w_fill"), BLUE, "Habit")
    fan(ax, x, stack(a, "w_fill"), VIOLET, "Heat-aware")
    if "learned" in E:
        fan(ax, x, stack(E["learned"]["records"], "w_fill"), AQUA, "Learned")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Weekly fill rate: weather-aware policies lift the summer trough")
    ax.set_ylabel("Sales ÷ demand")
    ax.legend(loc="lower left", ncol=3)

    ax = axes[2]
    shade_heat(ax, h)
    fan(ax, x, stack(h, "w_waste") / 1000, BLUE, "Habit")
    fan(ax, x, stack(E["habit_nocut"]["records"], "w_waste") / 1000, MUTED, "Habit, no power cuts")
    ax.set_title("Power-cut waste lives in the tail: a calm median, large spikes in the worst years")
    ax.set_ylabel("Units wasted per week (thousands)")
    ax.set_xlabel("Week of year")
    ax.legend(loc="upper right", ncol=2)
    fig.text(0.01, 0.005, "Lines: median of 100 simulated years. Bands: 5th-95th percentile. "
             "Orange shading: share of years with a heat wave that week.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, "fig1_year_fan.png")


def dot_rows(axes, E, order, specs):
    for ax, (key, title, scale, unit) in zip(axes, specs):
        for i, k in enumerate(order):
            v = stack(E[k]["records"], key) / scale
            lo, med, hi = q(v)
            y = len(order) - 1 - i
            ax.plot([lo, hi], [y, y], color=SEQ[1], linewidth=3, solid_capstyle="round", zorder=1)
            jitter = (np.random.default_rng(i).random(v.size) - 0.5) * 0.35
            ax.scatter(v, y + jitter, s=6, color=BLUE, alpha=0.35, linewidths=0)
            ax.plot([med], [y], "o", color=INK, markersize=5)
        ax.set_title(f"{title} ({unit})")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([E[k]["label"] for k in order][::-1])


def fig_distributions(E, order):
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6), sharey=True)
    dot_rows(axes, E, order, [("lost", "Lost sales per year", 1e6, "million units"),
                              ("waste_units", "Waste per year", 1e3, "thousand units")])
    fig.text(0.01, -0.01, "Each dot is one simulated year; black dot = median; bar = 5th-95th percentile.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig2_distributions.png")


def paired(E, a, b, key):
    return stack(E[a]["records"], key) - stack(E[b]["records"], key)


def fig_power_cut_attribution(E):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, scale, title in [
        (axes[0], "waste_units", 1e3, "Extra waste caused by power cuts (thousand units / year)"),
        (axes[1], "lost", 1e3, "Extra lost sales caused by power cuts (thousand units / year)"),
    ]:
        d1 = paired(E, "habit", "habit_nocut", key) / scale
        d2 = paired(E, "habit_severe", "habit_severe_nocut", key) / scale
        bins = np.linspace(min(d1.min(), d2.min()), max(d1.max(), d2.max()), 28)
        ax.hist(d1, bins=bins, color=BLUE, alpha=0.75, label="Typical years", rwidth=0.9)
        ax.hist(d2, bins=bins, color=ORANGE, alpha=0.75, label="Severe summer forced", rwidth=0.9)
        ax.set_title(title)
        ax.set_ylabel("Simulated years")
        ax.legend()
    fig.tight_layout()
    save(fig, "fig3_power_cut_attribution.png")


def fig_heat_dose(E):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    keys = [("habit", "Habit"), ("heat", "Heat-aware")] + ([("learned", "Learned")] if "learned" in E else [])
    for j, (k, lab) in enumerate(keys):
        c = POLICY_COLOR[k]
        hw = stack(E[k]["records"], "heat_weeks")
        lost = stack(E[k]["records"], "lost") / 1e6
        jit = (np.random.default_rng(j).random(hw.size) - 0.5) * 0.25
        ax.scatter(hw + jit + (j - 1) * 0.2, lost, s=16, color=c, alpha=0.7, label=lab,
                   edgecolors=SURFACE, linewidths=1)
        b = np.polyfit(hw, lost, 1)
        xs = np.array([hw.min(), hw.max()])
        ax.plot(xs, np.polyval(b, xs), color=c, linewidth=2)
    ax.set_xlabel("Heat-wave weeks in the simulated year")
    ax.set_ylabel("Lost sales (million units)")
    ax.set_title("Each heat-wave week costs sales; weather-aware policies flatten the slope")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig4_heat_dose.png")


def fig_map(E):
    cfg = TwinConfig()
    gx, gy = cfg.city.grid
    m = E["habit"]["records"][0]["map"]
    risk = m["feeder_risk"]
    base = np.mean([r["shop_waste_rate"] for r in E["habit"]["records"]], axis=0) * 100
    batt = np.mean([r["shop_waste_rate"] for r in E["habit_batt_risk"]["records"]], axis=0) * 100
    has_batt = E["habit_batt_risk"]["records"][0]["map"]["battery"]
    cut = base - batt
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    panels = [(axes[0], base, "Kirana waste, % of demand", "Waste concentrates on the weak feeders",
               np.percentile(base, 2), np.percentile(base, 98)),
              (axes[1], cut, "Reduction in waste, percentage points", "Battery fridges remove it where they sit",
               0, np.percentile(cut, 98))]
    for ax, val, cblab, title, lo, hi in panels:
        for loc in range(gx * gy):
            x, y = loc % gx, gy - 1 - loc // gx
            ax.add_patch(plt.Rectangle((x + 0.02, y + 0.02), 0.96, 0.96, color=YELLOW,
                                       alpha=0.06 + 0.32 * (risk[loc] - risk.min()) / np.ptp(risk), linewidth=0))
            ax.text(x + 0.06, y + 0.9, f"Sector {loc + 1}", fontsize=7.5, color=INK2, va="top")
        px, py = m["pos"][:, 0], gy - m["pos"][:, 1]
        sc = ax.scatter(px, py, c=val, cmap=cmap, vmin=lo, vmax=hi, s=24, edgecolors=SURFACE, linewidths=0.8)
        if ax is axes[1]:
            ax.scatter(px[has_batt], py[has_batt], s=80, facecolors="none", edgecolors=ORANGE, linewidths=1.2)
        ax.set_xlim(0, gx); ax.set_ylim(0, gy); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, shrink=0.8)
        cb.set_label(cblab, color=INK2); cb.outline.set_visible(False)
    fig.text(0.01, 0.01, "Dots: 240 kiranas, mean of 100 simulated years. Yellow tint: weaker power feeder (more outages). "
             "Orange ring: battery-backed brand fridge (30% of shops, riskiest feeders first).", color=MUTED, fontsize=8.5)
    save(fig, "fig5_city_map.png")


def fig_predicate(E, order):
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for i, k in enumerate(order):
        v = stack(E[k]["records"], "predicate_share") * 100
        lo, med, hi = q(v)
        y = len(order) - 1 - i
        ax.plot([lo, hi], [y, y], color=SEQ[1], linewidth=3, solid_capstyle="round")
        ax.plot([med], [y], "o", color=INK, markersize=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([E[k]["label"] for k in order][::-1])
    ax.set_xlabel("% of weeks the goal predicate holds")
    ax.set_title("How often the goal is met")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    fail = stack(E["habit"]["records"], "fill_conjunct_fail").mean(0) * 100
    ax.imshow(fail, cmap=matplotlib.colors.ListedColormap(SEQ), vmin=0, vmax=max(50, fail.max()))
    for c in range(fail.shape[0]):
        for p in range(fail.shape[1]):
            ax.text(p, c, f"{fail[c, p]:.0f}%", ha="center", va="center",
                    color=SURFACE if fail[c, p] > 30 else INK, fontsize=9)
    ax.set_xticks(range(len(PRODUCTS))); ax.set_xticklabels(PRODUCTS, rotation=20)
    ax.set_yticks(range(len(CHANNELS))); ax.set_yticklabels(CHANNELS)
    ax.grid(False)
    ax.set_title("Habit: share of weeks each fill target fails")
    fig.tight_layout()
    save(fig, "fig6_predicate.png")


def fig_convergence(E):
    v = stack(E["habit"]["records"], "fill") * 100
    n = np.arange(1, v.size + 1)
    mean = np.cumsum(v) / n
    sd = np.array([v[:i].std(ddof=1) if i > 1 else np.nan for i in n])
    half = 1.96 * sd / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.fill_between(n, mean - half, mean + half, color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(n, mean, color=BLUE)
    ax.set_xlabel("Simulated years (N)")
    ax.set_ylabel("Mean annual fill rate (%)")
    ax.set_title("Monte Carlo convergence: the estimate settles well before N = 100")
    fig.tight_layout()
    save(fig, "fig7_convergence.png")


def fig_bullwhip(E):
    h = E["habit"]["records"]
    o = stack(h, "w_kirana_orders_milk")[0] / 1000
    d = stack(h, "w_kirana_demand_milk")[0] / 1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    shade_heat(ax, h[:1])
    ax.plot(d, color=MUTED, label="Consumer demand at kiranas")
    ax.plot(o, color=BLUE, label="Kirana orders to the distributor")
    ax.set_title("One simulated year: shop orders against real demand")
    ax.set_ylabel("Pouch milk, thousand units / week")
    ax.set_xlabel("Week of year")
    ax.legend(loc="lower left")
    ax = axes[1]
    for k, lab in [("habit", "Habit"), ("heat", "Heat-aware")]:
        ax.hist(stack(E[k]["records"], "bullwhip"), bins=20, color=POLICY_COLOR[k], alpha=0.7, label=lab, rwidth=0.9)
    ax.axvline(1, color=INK2, linewidth=1)
    ax.set_title("Bullwhip ratio across 100 years")
    ax.set_xlabel("var(weekly orders) ÷ var(weekly demand)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig8_bullwhip.png")


def fig_learning(E):
    """Training curve and the three policies compared on the same 100 evaluation years."""
    z = np.load(ROOT / "results" / "q_table.npz")
    curve = z["curve"]
    batch = int(z["years"]) // curve.size
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    yrs = (np.arange(curve.size) + 1) * batch
    ax.plot(yrs, curve, color=SEQ[1], linewidth=1, label="Each round of 20 training years")
    k = 10
    roll = np.convolve(curve, np.ones(k) / k, mode="valid")
    ax.plot(yrs[k - 1:], roll, color=AQUA, label="Rolling mean")
    for key, lab in [("habit", "Habit"), ("heat", "Heat-aware")]:
        ax.axhline(np.mean([r["reward_lakh"][10.0] for r in E[key]["records"]]), color=POLICY_COLOR[key],
                   linewidth=1.2, linestyle="--", label=f"{lab} (evaluation years)")
    ax.set_title("Training: annual reward climbs as exploration falls")
    ax.set_xlabel("Training years simulated")
    ax.set_ylabel("Annual reward (lakh ₹)")
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    keys = [("habit", "Habit"), ("heat", "Heat-aware"), ("learned", "Learned")]
    for i, (key, lab) in enumerate(keys):
        v = np.array([r["reward_lakh"][10.0] for r in E[key]["records"]])
        lo, med, hi = q(v)
        ax.plot([lo, hi], [i, i], color=SEQ[1], linewidth=3, solid_capstyle="round")
        jitter = (np.random.default_rng(i).random(v.size) - 0.5) * 0.3
        ax.scatter(v, i + jitter, s=7, color=POLICY_COLOR[key], alpha=0.45, linewidths=0)
        ax.plot([med], [i], "o", color=INK, markersize=5)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([lab for _, lab in keys])
    ax.grid(axis="y", visible=False)
    ax.set_title("Reward on the 100 evaluation years")
    ax.set_xlabel("Annual reward: profit − ₹10 × lost units (lakh ₹)")
    fig.tight_layout()
    save(fig, "fig9_learning.png")


def fig_lambda(E):
    keys = [k for k in ("learned_lam0", "learned", "learned_lam30") if k in E]
    lams = {"learned_lam0": 0, "learned": 10, "learned_lam30": 30}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, key, scale, title in [(axes[0], "profit", 1e7, "Profit (crore ₹)"),
                                  (axes[1], "lost", 1e6, "Lost sales (million units)"),
                                  (axes[2], "waste_units", 1e3, "Waste (thousand units)")]:
        for k in keys:
            v = stack(E[k]["records"], key) / scale
            lo, med, hi = q(v)
            x = lams[k]
            ax.plot([x, x], [lo, hi], color=SEQ[1], linewidth=3, solid_capstyle="round")
            ax.plot([x], [med], "o", color=AQUA, markersize=6)
        ax.set_xticks([lams[k] for k in keys])
        ax.set_xlabel("Penalty per lost unit, λ (₹)")
        ax.set_title(title)
    fig.suptitle("What the learned policy trades off as lost sales get more expensive", x=0.01, ha="left",
                 fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    save(fig, "fig10_lambda_tradeoff.png")


def fig_behaviour(E, order):
    goals = [("recovery", "Back to target\nwithin 2 weeks\nof a heat wave"),
             ("no_collapse", "No channel below\n85% for over\n2 weeks running"),
             ("fair_waste", "No sector over\n20% of\nkirana waste"),
             ("all", "All three")]
    M = np.array([[beh(E[k]["records"], g).mean() * 100 for g, _ in goals] for k in order])
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(order) + 1.8))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(SEQ), vmin=0, vmax=100, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center",
                    color=SURFACE if M[i, j] > 55 else INK, fontsize=9)
    ax.set_xticks(range(len(goals))); ax.set_xticklabels([t for _, t in goals], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(order))); ax.set_yticklabels([E[k]["label"] for k in order])
    ax.grid(False)
    ax.set_title("Behaviour-based goals: share of the 100 simulated years in which each holds", pad=34)
    fig.tight_layout()
    save(fig, "fig11_behaviour_goals.png")


# ------------------------------------------------------------------ tables
def summary_table(E, order):
    rows = []
    for k in order:
        R = E[k]["records"]

        def stat(key, scale=1.0):
            v = stack(R, key) * scale
            lo, med, hi = q(v)
            return {"mean": float(np.nanmean(v)), "p5": float(lo), "p50": float(med), "p95": float(hi)}

        rew = np.array([r["reward_lakh"][10.0] for r in R])
        rows.append({
            "key": k, "label": E[k]["label"], "policy": E[k]["policy"],
            "fill_pct": stat("fill", 100), "fill_heat_pct": stat("fill_heat", 100),
            "worst_week_fill_pct": stat("worst_week_fill", 100),
            "lost_units": stat("lost"), "waste_units": stat("waste_units"),
            "waste_pct": stat("waste_share", 100),
            "kirana_customers_lost": stat("migrated_net"),
            "profit_crore": stat("profit", 1e-7),
            "predicate_weeks_pct": stat("predicate_share", 100),
            "bullwhip": stat("bullwhip"),
            "reward_lam10_lakh": {"mean": float(rew.mean()), "p5": float(np.percentile(rew, 5)),
                                  "p50": float(np.median(rew)), "p95": float(np.percentile(rew, 95))},
            "behaviour_pct": {g: float(beh(R, g).mean() * 100)
                              for g in ("recovery", "no_collapse", "fair_waste", "all")},
            "unvisited_weeks_mean": float(np.mean([r.get("unvisited_weeks", 0) for r in R])),
        })
    return rows


def paired_effects(E):
    """CRN paired differences: effect of a change, year by year, with a 95% interval."""
    pairs = [
        ("Heat-aware vs Habit", "heat", "habit"),
        ("Learned vs Habit", "learned", "habit"),
        ("Learned vs Heat-aware", "learned", "heat"),
        ("Learned vs Heat-aware (severe summer)", "learned_severe", "heat_severe"),
        ("Battery fridges (risk-targeted) vs none", "habit_batt_risk", "habit"),
        ("Battery fridges: risk-targeted vs random", "habit_batt_risk", "habit_batt_rand"),
        ("Kirana returns vs none", "habit_returns", "habit"),
        ("Heat-aware + batteries vs Habit", "heat_batt", "habit"),
        ("Power cuts vs none (typical years)", "habit", "habit_nocut"),
        ("Power cuts vs none (severe summer)", "habit_severe", "habit_severe_nocut"),
        ("Heat-aware vs Habit (severe summer)", "heat_severe", "habit_severe"),
        ("Heat-aware + batteries vs Habit (severe summer)", "heat_batt_severe", "habit_severe"),
    ]
    out = []
    for name, a, b in pairs:
        if a not in E or b not in E:
            continue
        row = {"comparison": name}
        for key, scale in [("lost", 1), ("waste_units", 1), ("fill", 100), ("profit", 1e-7),
                           ("migrated_net", 1), ("predicate_share", 100), ("reward10", 1)]:
            if key == "reward10":
                d = (np.array([r["reward_lakh"][10.0] for r in E[a]["records"]])
                     - np.array([r["reward_lakh"][10.0] for r in E[b]["records"]]))
            else:
                d = paired(E, a, b, key) * scale
            se = d.std(ddof=1) / np.sqrt(d.size)
            better = (d < 0).mean() if key in ("lost", "waste_units", "migrated_net") else (d > 0).mean()
            row[key] = {"mean": float(d.mean()), "ci95": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
                        "share_years_better": float(better)}
        out.append(row)
    return out


def fans(E):
    """Weekly percentile bands for the web page."""
    def band(k, key, scale=1.0):
        lo, med, hi = q(stack(E[k]["records"], key) * scale)
        return {"p5": lo.round(3).tolist(), "p50": med.round(3).tolist(), "p95": hi.round(3).tolist()}
    f = {
        "habit": {"fill": band("habit", "w_fill", 100), "waste": band("habit", "w_waste")},
        "heat": {"fill": band("heat", "w_fill", 100), "waste": band("heat", "w_waste")},
        "habit_nocut": {"waste": band("habit_nocut", "w_waste")},
        "heat_share": stack(E["habit"]["records"], "w_heat").mean(0).round(3).tolist(),
    }
    if "learned" in E:
        f["learned"] = {"fill": band("learned", "w_fill", 100), "waste": band("learned", "w_waste")}
    return f


def main():
    R = load()
    E = R["experiments"]
    all_keys = ["habit", "heat", "learned", "habit_batt_risk", "habit_batt_rand", "habit_returns", "heat_batt",
                "habit_nocut", "habit_severe", "heat_severe", "learned_severe", "heat_batt_severe",
                "habit_severe_nocut", "learned_lam0", "learned_lam30"]
    order = [k for k in all_keys if k in E]
    fig_year(E)
    fig_distributions(E, order)
    fig_power_cut_attribution(E)
    fig_heat_dose(E)
    fig_map(E)
    fig_predicate(E, order)
    fig_convergence(E)
    fig_bullwhip(E)
    fig_behaviour(E, order)
    summary = {"n_years": R["n"], "experiments": summary_table(E, order), "paired_effects": paired_effects(E),
               "fans": fans(E)}
    if "learned" in E and (ROOT / "results" / "q_table.npz").exists():
        fig_learning(E)
        fig_lambda(E)
        from dairytwin.mdp import policy_table
        summary["policy_table"] = policy_table(ROOT / "results" / "q_table.npz")
    with open(ROOT / "results" / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("figures and results/summary.json written")


if __name__ == "__main__":
    main()
