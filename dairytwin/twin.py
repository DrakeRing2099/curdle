"""The digital twin: one stochastic simulated year of the dairy distribution network.

    x_{t+1} = f(x_t, a_t, xi_t)

x_t   full state (every stock by age, every shop's caution, customer pools, regimes, cash)
      The city itself (shops, fridges, habits, power feeders) is fixed by `city_seed`;
      `seed` only changes what happens to it during the year.
a_t   the weekly decision from a policy (see policies.py)
xi_t  the day's random draws, from independent per-component streams so that two
      policies run on the same seed face the same weather, milk, outages and late vans
      (common random numbers).

A day, in order
---------------
 1. (Mondays) regime chains step; the policy observes the state and sets the week's knobs
 2. random milk arrives; the plant produces within milk and capacity
 3. plant ships against last night's channel orders (allocation rule on shortage)
 4. kirana channel: distributor rations, routes run (some late), morning customers,
    late vans and noon drops arrive, afternoon customers; stockouts spill onto the app
 5. modern trade: DC receives, random dwell, store customers
 6. quick-commerce: plant drops straight into the dark stores at dawn, customers, noon
    drop, afternoon customers + kirana spill
 7. power grid (inside step 4, after the morning): feeders fail at random, far more in a
    heat wave; stock in a dead fridge loses shelf life unless a battery covers the cut
 8. evening close everywhere: age, expire, count waste; kiranas update caution
 9. orders for tomorrow from every channel
10. customers who met stockouts drift between channels (binomial, customer by customer)
11. cash
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .channels import ModernTrade, Plant, QuickCommerce
from .config import KIR, MT, N_PRODUCTS, QC, Scenario, TwinConfig
from .kirana import DeliveryMessage, Distributor, KiranaActors, build_city
from .policies import HabitPolicy, WeeklyDecision, allocate
from .regimes import RegimeChains

STREAMS = ["regimes", "supply", "kirana", "distributor", "mt", "qc", "power", "consumers"]


@dataclass
class DayLog:
    """Everything recorded per day (arrays of shape (days, ...))."""
    fields: dict = field(default_factory=dict)

    def add(self, **kw):
        for k, v in kw.items():
            self.fields.setdefault(k, []).append(np.asarray(v))

    def stack(self) -> dict:
        return {k: np.stack(v) for k, v in self.fields.items()}


class DairyTwin:
    def __init__(self, cfg: TwinConfig | None = None, scenario: Scenario | None = None,
                 policy=None, seed: int = 0, city_seed: int = 0):
        self.cfg = cfg or TwinConfig()
        self.scen = scenario or Scenario()
        self.policy = policy or HabitPolicy()
        self.seed = seed
        ss = np.random.SeedSequence(seed)
        self.rng = {name: np.random.default_rng(s) for name, s in zip(STREAMS, ss.spawn(len(STREAMS)))}
        # the city itself (layout, shops, fridges, habits, feeders) is fixed: one city, many years
        self.city_seed = city_seed
        city = np.random.default_rng(np.random.SeedSequence([city_seed, 2026]))

        cfg = self.cfg
        self.pc = cfg.arr("per_capita")
        self.C = np.array(cfg.city.households, dtype=np.int64)        # customers per channel
        self.migrated = 0                                              # left kirana after stockouts
        base = self.C[:, None] * self.pc[None, :]                      # (C, P) mean daily units

        loc, route, pos, risk = build_city(cfg, city)
        self.feeder_risk = risk
        self.n_loc = cfg.city.grid[0] * cfg.city.grid[1]
        self.kirana = KiranaActors(cfg, self.scen, self.rng["kirana"], loc, route, pos, risk, city)
        self.dist = Distributor(cfg, self.rng["distributor"], route)
        self.plant = Plant(cfg, self.rng["supply"], base.sum(axis=0))
        self.plant.stock[:, 0] = np.round(base.sum(axis=0) * 0.5).astype(np.int64)
        self.mt = ModernTrade(cfg, self.rng["mt"], base[MT])
        self.qc = QuickCommerce(cfg, self.rng["qc"], base[QC], city)
        self.regimes = RegimeChains(cfg.regimes, self.rng["regimes"])

        self.pending = np.round(base).astype(np.int64)                 # last night's orders (C, P)
        shop_orders = np.round(self.kirana.share[:, None] * base[KIR][None, :]).astype(np.int64)
        self.dist.pending = shop_orders                                # opening per-shop orders
        self.pending[KIR] = shop_orders.sum(axis=0)
        self.order_hist = [self.pending.sum(axis=0)] * 7
        self.cash = cfg.money.opening_cash
        self.day = 0
        self.decision: WeeklyDecision | None = None
        # live feed: None for an ordinary simulation. A live driver (live.py) sets a dict that
        # may hold weather / supply / festival (observed regimes), outage_hours (per locality),
        # demand_scale (a level shift on every channel's demand) and decision (the plan in force).
        self.feed: dict | None = None
        self.last_outage_hours = np.zeros(self.n_loc)
        self.last_mult = np.ones(N_PRODUCTS)
        self.log = DayLog()
        # per-shop annual tallies (for maps)
        n = cfg.city.n_kirana
        self.shop_waste = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.shop_lost = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.shop_demand = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.shop_outage_hours = np.zeros(n)

    # ------------------------------------------------------------ bookkeeping
    def units_in_system(self) -> np.ndarray:
        """Every unit held anywhere, by product (for the conservation check)."""
        return (self.plant.stock.sum(-1) + self.kirana.stock.sum(axis=(0, 2))
                + self.mt.dc.sum(-1) + self.mt.shelf.sum(-1)
                + self.qc.store.sum(axis=(0, 2)))

    # ------------------------------------------------------------ observations
    def demand_mult(self) -> np.ndarray:
        p = self.cfg.products
        m = np.ones(N_PRODUCTS)
        if self.regimes.weather == 1:
            m = m * np.asarray(p.heat_mult)
        if self.regimes.festival == 1:
            m = m * np.asarray(p.festival_mult)
        if self.feed and self.feed.get("demand_scale") is not None:
            m = m * float(self.feed["demand_scale"])
        return m

    def supply_mult(self, week: int) -> float:
        s = self.cfg.supply
        m = 1.0
        if self.regimes.is_flush(week):
            m *= s.flush_mult
        elif self.regimes.is_lean(week):
            m *= s.lean_mult
        if self.regimes.weather == 1:
            m *= s.heat_mult
        if self.regimes.supply == 1:
            m *= s.stressed_mult
        return m

    def observe(self) -> dict:
        return {
            "day": self.day, "week": self.day // 7,
            "weather": self.regimes.weather, "supply": self.regimes.supply,
            "festival": self.regimes.festival,
            "orders_7d": np.mean(self.order_hist[-7:], axis=0),
            "demand_mult_now": self.demand_mult(), "demand_mult_last": self.last_mult,
            "plant_stock": self.plant.available(), "customers": self.C.copy(),
            "cash": self.cash, "kirana_fill_7d": self.recent_kirana_fill(),
            "flush": self.regimes.is_flush(self.day // 7), "lean": self.regimes.is_lean(self.day // 7),
        }

    def recent_kirana_fill(self) -> float:
        """Share of kirana demand met over the last 7 days (1.0 before any history)."""
        f = self.log.fields
        if not f.get("demand"):
            return 1.0
        d = sum(x[KIR].sum() for x in f["demand"][-7:])
        s = sum(x[KIR].sum() for x in f["sales"][-7:])
        return float(s / max(d, 1))

    # ------------------------------------------------------------ power grid
    def outages(self) -> np.ndarray:
        """Hours without power per locality today. Uniform draws are always consumed,
        so switching outages off (the counterfactual) keeps every other draw aligned."""
        pw = self.cfg.power
        r = self.rng["power"]
        heat = self.regimes.weather == 1
        p = (pw.p_outage_heat if heat else pw.p_outage_normal) * self.feeder_risk
        hit = r.random(self.n_loc) < np.minimum(p, 0.95)
        lo, hi = pw.hours_heat if heat else pw.hours_normal
        hours = np.where(hit, r.uniform(lo, hi, self.n_loc), 0.0)
        for loc, h in self.scen.forced_outages.get(self.day, {}).items():
            hours[loc] = max(hours[loc], h)
        if not self.scen.outages_enabled:
            hours = np.zeros_like(hours)
        if self.feed and self.feed.get("outage_hours") is not None:     # observed, replaces the draw
            hours = np.asarray(self.feed["outage_hours"], dtype=float)
        self.last_outage_hours = hours
        return hours

    # ------------------------------------------------------------ one day
    def step(self):
        cfg, K = self.cfg, self.kirana
        t = self.day
        week = t // 7
        if t % 7 == 0:
            self.last_mult = self.demand_mult()
            self.regimes.step(week, self.scen.forced_weather.get(week))
        if self.feed:                                   # observed regimes overwrite the chain
            for key in ("weather", "supply", "festival"):
                if self.feed.get(key) is not None:
                    setattr(self.regimes, key, int(self.feed[key]))
        if t % 7 == 0:
            if self.feed and self.feed.get("decision") is not None:
                self.decision = self.feed["decision"]
            else:
                self.decision = self.policy.weekly(self.observe())
        dec = self.decision
        mult = self.demand_mult()
        heat = self.regimes.weather == 1

        # 2. milk and production
        milk = self.plant.milk_today(self.supply_mult(week))
        # produce up to the plan net of what is still in the plant cold store (base-stock)
        to_make = np.maximum(0, np.ceil(dec.plan - self.plant.available()))
        made, milk_used = self.plant.produce(to_make, milk, dec.milk_priority)

        # 3. ship against last night's orders
        orders = self.pending
        shipped = allocate(self.plant.available(), orders, dec.channel_priority)
        # kiranas and modern trade are loaded now; quick-commerce is dropped later in the
        # day, straight from the plant, so its allocation is only an allowance for now
        prof = [self.plant.ship(shipped[c]) for c in (KIR, MT)]
        qc_allowance = shipped[QC].copy()

        # 4. kirana channel
        K.begin_day()
        alloc = self.dist.ration(shipped[KIR])
        shop_prof = self.dist.split_profile(alloc, prof[KIR])
        if dec.twice_daily:
            drop1 = np.floor(shop_prof * 0.55).astype(np.int64)
            drop2 = shop_prof - drop1
        else:
            drop1, drop2 = shop_prof, None
        late = self.dist.late_flags(heat)
        mean_K = (K.share[:, None] * self.C[KIR] * self.pc[None, :] * mult[None, :])
        ms = cfg.arr("morning_share")
        refused = K.on_delivery(DeliveryMessage(np.where(late[:, None, None], 0, drop1), late, drop=1))
        spill = K.on_customers(mean_K, ms)
        refused += K.on_delivery(DeliveryMessage(np.where(late[:, None, None], drop1, 0), late, drop=0))
        # power cuts hit in the hot afternoon, while the fridge still holds the morning stock
        hours_loc = self.outages()
        hours = hours_loc[K.locality]
        unprot = np.where(K.battery, np.maximum(0.0, hours - cfg.power.battery_hours), hours)
        K.on_power_cut(unprot)
        if drop2 is not None:
            refused += K.on_delivery(DeliveryMessage(drop2, late, drop=2))   # fresh stock after the cut
        self.plant.stock += refused.sum(axis=0)                               # van brings it back
        spill = spill + K.on_customers(mean_K, 1 - ms)

        # 5. modern trade
        mean_M = self.C[MT] * self.pc * mult
        self.mt.receive(prof[MT])
        self.mt.release()
        dM, sM = self.mt.serve(mean_M)

        # 6. quick-commerce (two drops a day, absorbs kirana spill in the afternoon)
        mean_Q = self.C[QC] * self.pc * mult
        dropped = self.qc.drop(self.plant, qc_allowance)
        dQ1, sQ1 = self.qc.serve(mean_Q, ms)
        dropped = dropped + self.qc.drop(self.plant, qc_allowance - dropped)
        shipped[QC] = dropped
        dQ2, sQ2 = self.qc.serve(mean_Q, 1 - ms, extra=spill)
        dQ, sQ = dQ1 + dQ2, sQ1 + sQ2

        # 8. evening close
        k_discard, k_return = K.on_close()
        w_plant = self.plant.close()
        w_mt = self.mt.close()
        w_qc = self.qc.close()

        # 9. tomorrow's orders
        o_K = self.dist.on_orders(K.on_order_time(dec.twice_daily))
        o_M = self.mt.order()
        o_Q = self.qc.order(dQ, sQ)
        self.pending = np.stack([o_K, o_M, o_Q])
        self.order_hist.append(self.pending.sum(axis=0))

        # 10. customers drift between channels, one binomial decision per customer
        cp, r = cfg.consumers, self.rng["consumers"]
        dK, sK, lK = K.today_demand.sum(0), K.today_sales.sum(0), K.today_lost.sum(0)
        lQ = dQ - sQ
        dis_K = int(lK.sum() / cp.units_per_visit)
        dis_Q = int(lQ.sum() / cp.units_per_visit)
        k_to_q = min(r.binomial(dis_K, cp.p_migrate_k_to_q), self.C[KIR])
        q_to_k = min(r.binomial(dis_Q, cp.p_migrate_q_to_k), self.C[QC])
        fill_K = sK.sum() / max(dK.sum(), 1)
        back = r.binomial(self.migrated, cp.p_winback) if fill_K >= cp.winback_fill else 0
        back = min(back, self.C[QC] - q_to_k)
        self.C[KIR] += q_to_k + back - k_to_q
        self.C[QC] += k_to_q - q_to_k - back
        self.migrated = max(0, min(self.migrated + k_to_q - back, self.C[QC]))

        # 11. cash
        mo, price = cfg.money, cfg.arr("retail_price")
        wp = mo.wholesale_share * price
        kept = shipped[KIR] - refused.sum(axis=(0, 2))
        revenue = (wp * kept).sum() - (wp * k_return.sum(0)).sum() + (wp * (sM + sQ)).sum()
        drops = 2 if dec.twice_daily else 1
        cost = (milk_used * cfg.supply.milk_cost
                + (made * cfg.arr("processing_cost")).sum()
                + cfg.city.n_routes * drops * mo.trip_cost + mo.mt_trip_cost
                + cfg.city.n_dark_stores * 2 * mo.qc_trip_cost
                + K.battery.sum() * (mo.fridge_cost_per_day + mo.battery_cost_per_day)
                + self.plant.available().sum() * mo.holding_cost)
        self.cash += revenue - cost

        # tallies and log
        self.shop_waste += k_discard + k_return
        self.shop_lost += K.today_lost
        self.shop_demand += K.today_demand
        self.shop_outage_hours += unprot
        demand = np.stack([dK, dM, dQ])
        sales = np.stack([sK, sM, sQ])
        self.log.add(
            weather=self.regimes.weather, supply=self.regimes.supply,
            festival=self.regimes.festival, milk=milk, milk_used=milk_used,
            made=made, shipped=shipped, orders=orders, demand=demand, sales=sales,
            lost=demand - sales, spill=spill,
            waste_kirana=(k_discard + k_return).sum(0), waste_kirana_returned=k_return.sum(0),
            waste_plant=w_plant, waste_mt=w_mt, waste_qc=w_qc,
            outage_localities=(hours_loc > 0).sum(), outage_shop_hours=unprot.sum(),
            late_shops=late.sum(), refused=refused.sum(axis=(0, 2)), mean_k=K.k.mean(),
            customers=self.C.copy(),
            migrated=self.migrated, cash=self.cash, revenue=revenue, cost=cost,
            kirana_orders=o_K, twice_daily=dec.twice_daily,
        )
        self.day += 1

    def run(self, days: int | None = None) -> dict:
        for _ in range(days or self.cfg.days):
            self.step()
        out = self.log.stack()
        out["shop_waste"] = self.shop_waste
        out["shop_lost"] = self.shop_lost
        out["shop_demand"] = self.shop_demand
        out["shop_outage_hours"] = self.shop_outage_hours
        out["shop_pos"] = self.kirana.pos
        out["shop_locality"] = self.kirana.locality
        out["shop_battery"] = self.kirana.battery
        out["feeder_risk"] = self.feeder_risk
        return out
