# Dairy Twin

A stochastic digital twin of how one dairy brand's products reach consumers in an Indian
city. Built for the DAIICT Digital Twins course (IT433), following the course's system
definition, System Dynamics, actor modelling and simulation-based decision-making
material. All data is synthetic.

## What is in the city

| Part | Model type | What makes it distinct |
|---|---|---|
| Plant | stock and flow | random daily milk supply (season x heat wave x supply stress x noise); production capacity; base-stock planning |
| 240 kiranas | **actors** | each shop has its own fridge, size and caution; orders by habit from noisy sales; reacts to late vans and waste; loses stock in power cuts |
| Distributor | actor | rations short shipments, runs 6 morning routes, each may miss the 6 am window |
| Modern trade | stock and flow | brand warehouse with random dwell (binomial release) -> 4 supermarkets' shelves; forecast orders; expired stock returned to the brand |
| Quick-commerce | stock and flow | 6 tiny dark stores filled straight from the plant twice a day; each drop only fills free space, the rest stays at the plant; absorbs kirana customers who find an empty shelf |
| Consumers | stock and flow, binomial | households per channel; customers who meet stockouts drift between channels one decision at a time |
| Power grid | stochastic | 12 feeders with different reliability; cuts are far more frequent in a heat wave; unrefrigerated stock loses shelf life |

**Uncertainty is structural.** Three weekly Markov regimes (weather, supply, festival) give slow,
persistent shocks; every flow is a random draw each day (Poisson demand, lognormal milk,
binomial DC release, Bernoulli late vans, binomial churn, random power cuts). Stock is held
by age, so waste is real expiry. The weather regime is shared: a heat wave lowers milk yield
*and* raises curd demand in the same weeks, and makes power cuts more likely.

## Layout

```
dairytwin/          the twin (Python)
  config.py         every parameter, with units
  regimes.py        weekly Markov chains
  batches.py        age-structured stock (FIFO sales, ageing, expiry)
  kirana.py         kirana actors (rules R1-R10) and the distributor actor
  channels.py       plant, modern trade, quick-commerce
  policies.py       weekly decision rules: Habit, Heat-aware
  mdp.py            the MDP: state, action menu, reward, Q-learning, learned policy
  twin.py           one simulated day, x(t+1) = f(x(t), a(t), xi(t))
  montecarlo.py     batches of years, KPIs, value goal (predicate) and behaviour goals
tests/              verification (slide 93): conservation, reproducibility, regime and
                    demand distributions, extreme conditions, actor invariants, Monte Carlo
experiments/        train_policy.py (Q-learning), run_experiments.py (100 years each),
                    make_figures.py, export_run.py (one year, day by day, for the viewer)
results/            q_table*.npz (learned policies), mc.pkl (raw),
                    summary.json (tables, paired effects, fan data, policy table)
figures/            report figures
web/                engine.js (JavaScript port), app.html, build.py,
                    dairy_twin_control_room.html (self-contained live page),
                    parity_check.js / parity_python.py (JS vs Python agreement)
```

## Run it

```bash
pip install numpy matplotlib pytest
python -m pytest -q tests                         # verification, ~25 s
python experiments/train_policy.py                # lambda=10, 3000 years, ~40 min on 2 cores
python experiments/train_policy.py --lam 0 --years 1500 --out results/q_table_lam0.npz
python experiments/train_policy.py --lam 30 --years 1500 --out results/q_table_lam30.npz
python experiments/run_experiments.py --n 100     # ~25 min on 2 cores
python experiments/make_figures.py                # figures + results/summary.json
python web/build.py                               # rebuild the live page
```

One year in code:

```python
from dairytwin import DairyTwin, HeatAwarePolicy
from dairytwin.config import Scenario

out = DairyTwin(policy=HeatAwarePolicy(),
                scenario=Scenario(battery_fridge_share=0.3).severe_summer(),
                seed=4).run()
print(out["sales"].sum() / out["demand"].sum())
```

## Kirana actor rules

Written in the slide 132 grammar, `on event && condition do action [@ probability]`:

| Rule | Event && condition | Action |
|---|---|---|
| R1 | delivery | accept what fits in the fridge, hand the rest back |
| R2 | delivery && after 6 am | raise caution k @ 0.6 |
| R3 | customer && in stock | sell the oldest unit |
| R4 | customer && out of stock | customer tries the app @ 0.3, else goes without |
| R5 | evening close | age stock, discard expired, k drifts back to normal |
| R6 | close && heavy waste | lower caution k @ 0.5 |
| R7 | order time | order k x forecast x cover + sqrt buffer - stock, capped by fridge space |
| R8 | brand fridge installed | more capacity, battery backup |
| R9 | close && returns policy | hand expired units back to the brand |
| R10 | power cut && no battery cover | stock loses shelf life; sour units thrown out |

## The decision layer (MDP)

Every Monday a policy reads the twin and picks the week's plan. Three policies are compared on
the same 100 simulated years (seeds 0-99, common random numbers):

* **Habit**: last week's orders + 5%, pro-rata when stock is short, one drop a day.
* **Heat-aware**: scales the forecast by this week's weather and festival, +8%; in a heat
  wave it serves kiranas and staples first and runs two drops a day.
* **Learned**: tabular Q-learning on the twin itself.

| MDP part | In the twin |
|---|---|
| State (72) | weather (normal / heat) x festival week x village supply (normal / stressed) x season (flush / lean / normal) x last week's kirana fill (<92%, 92-96%, >96%) |
| Action (24) | forecast (habit / weather) x margin (0, +8%, +16%) x when short (pro-rata / kiranas & staples first) x drops a day (1 / 2) |
| Transition | one simulated week of the twin |
| Reward | weekly profit - lambda x lost units - mu x wasted units (lambda = Rs 10, mu = 0 by default) |
| Discount | 0.95 per week |
| Training | 3000 years on seeds 10000+ (never the evaluation years), epsilon 0.5 -> 0.05, step size max(1/visits, 0.02) |

The learned policy acts greedily over actions it tried; in a state it never met in training it
falls back to the Heat-aware choice. Training again with lambda = 0 and lambda = 30 shows the
trade-off between profit and lost sales.

## Goals

* **Value goal (predicate)**, checked every week: rolling 4-week fill at or above 92% kirana,
  95% modern trade, 92% quick-commerce for every product, waste under its cap, cash above zero.
* **Behaviour goals**, checked over a year:
  1. after each heat wave ends, kirana fill is back to 95% within 2 weeks;
  2. no channel stays below 85% fill for more than 2 weeks in a row;
  3. no sector carries more than 20% of kirana waste.

## The system view (web/twin_viewer.html)

A standalone page that replays one simulated year day by day: every tier's stock, the flows
between them, power cuts by locality, all 240 shops, customers, waste, cash and the plan the
policy chose each Monday. It needs no server and no network: open the file in a browser.

```
python experiments/export_run.py --policy Learned --scenario severe --seed 7   # -> web/runs/*.json
python web/build_viewer.py                                                   # embeds web/runs/*.json
```

Any exported run can also be dropped onto the open page. Loading several runs with the same
seed (for example Habit, Heat-aware and Learned in the same severe summer) puts them on one
timeline and one year-to-date table. Scenarios: baseline, severe, batteries, batteries_random,
returns, nocut, severe_batteries, severe_nocut. Because the export reads the twin's own state,
any change to the model shows up in the viewer after a re-export.

## Results (100 years, medians)

| Policy | Fill | Lost sales | Waste | Profit | Reward (lambda=10) | Goal-met weeks | All behaviour goals |
|---|---|---|---|---|---|---|---|
| Habit | 94.6% | 1.64M | 241k | Rs 11.47 cr | 987 lakh | 47% | 14% of years |
| Heat-aware | 95.8% | 1.25M | 161k | Rs 11.33 cr | 1,006 lakh | 45% | 31% |
| Learned (lambda=10) | 96.7% | 0.99M | 77k | Rs 11.31 cr | 1,035 lakh | 11% | 23% |
| Learned (lambda=0) | 96.5% | 1.06M | 118k | Rs 11.14 cr | 1,007 lakh | 12% | 31% |
| Learned (lambda=30) | 96.8% | 0.97M | 74k | Rs 11.34 cr | 1,039 lakh | 39% | 28% |

Paired against Heat-aware on the same years, the learned policy loses 266k fewer units
(95% CI 251k-280k), wastes 83k fewer, earns the same profit (difference within +-Rs 3 lakh) and
scores Rs 26 lakh more reward, better in 99 of 100 years. In a severe summer it also earns
Rs 17 lakh more profit. The cost: the weekly goal predicate holds in 11% of weeks against 45%.
The reward counts every lost unit the same and has no per-channel targets, so the policy lets
supermarket butter and some milk weeks sit just under target while total lost sales fall.
Its weekly orders also swing harder (bullwhip 3.0 vs 0.47), because it switches margin between
0% and +16% as last week's fill crosses a band.

## Notes

* The JavaScript engine implements the same rules with a different random number generator,
  so it agrees with Python in distribution, not run for run. Parity over 100 years each
  (Habit policy, `web/parity_check.js` vs `web/parity_python.py`): fill 94.7% vs 94.6%, waste share
  0.86% vs 0.85%, lost sales 1.61M vs 1.64M, profit Rs 11.45 vs 11.45 crore, households lost
  2,875 vs 2,975 (2.0 SE). **Bullwhip does not match:** JS reads 1.35 against Python's 1.15
  (2.8 SE). Samplers, regime frequencies, caution dynamics and mean weekly order shapes match;
  the gap is extra week-scale order variance and is not yet isolated. The learned policy also
  agrees across engines (40 JS years: fill 96.6% vs 96.7%, lost 1.01M vs 0.99M). Treat Python as
  the reference; the live page's bullwhip trace is illustrative.

## Live mirror

The twin can follow a running system instead of generating its own year. `dairytwin/live.py`
runs a fake *reality* (a second simulated city we can poke) that emits a daily feed: weather,
supply and festival state, power-cut hours per area, footfall per channel, and stock readings
from the 30% of shops that have sensors. The twin takes the feed in, steps, and is corrected toward
the readings. From its synced state it runs eight futures for 14 days under each policy.

    python web/live_server.py        # then open http://localhost:8765

Sync modes: off (plain simulation), weather/supply/power cuts, full (plus demand level estimated
from footfall, plus stock correction). The page lets you inject a heat wave, supply stress, a
festival, a demand shift or a power cut into the fake reality and shows how fast the twin follows
and how the 14-day forecast changes. To use real data, replace the reality object with a feed
reader; the twin side does not change. All data here is synthetic.
