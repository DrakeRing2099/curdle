"""Kirana retailers as actors, plus the distributor actor that serves them.

Actor model (slides 131-132)
----------------------------
Each kirana is an actor with its own **state** (fridge, stock by age, caution) and
**behaviour** written as rules of the form ``on event && condition do action``, some of
them stochastic (``action1 @ p1, action2 @ p2``). Actors interact only by
**messages** ``<event, relevant data>``.

Implementation note: the twin simulates hundreds of shops for hundreds of runs, so the
actors are stored "array-of-actors" style: row ``i`` of every state array *is* actor
``i``, and each rule is applied to every actor whose event fired, in one vectorised
step. The semantics are exactly per-actor: no rule ever reads another shop's state, and
every random draw is per shop.

Rules (one simulated day)
-------------------------
R1  on delivery                              do  add units to stock (oldest-first sales later)
R2  on delivery && arrived after 6 am        do  raise caution k by step  @ p_k_late
R3  on customer asks for p && stock > 0      do  sell the oldest unit
R4  on customer asks for p && stock == 0     do  customer tries the app (spill to QC) @ p_spill,
                                                 or goes without           @ 1 - p_spill
R5  on evening close                         do  age stock one day, discard expired units,
                                                 k drifts back to k_base
R6  on evening close && heavy waste today    do  lower caution k by step  @ p_k_waste
R7  on order time                            do  order = k x f x cover + b x sqrt(f x cover) - stock,
                                                 f = half today's (sales + w x noticed lost), half memory,
                                                 capped by free fridge space; send <order>
R8  on brand fridge installed                do  fridge capacity up, battery backup on
R9  on returns pickup && returns policy      do  hand expired units back instead of discarding
R10 on power cut && no battery cover         do  stock in the dead fridge loses shelf life;
                                                 units that turn sour are thrown out
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .batches import age_and_expire, fifo_take
from .config import N_PRODUCTS, Scenario, TwinConfig

P_SPILL = 0.30   # R4: share of disappointed kirana customers who try the quick-commerce app


# ---------------------------------------------------------------- messages
@dataclass
class OrderMessage:
    """<order, quantities> from each kirana to the distributor."""
    qty: np.ndarray                  # (n, P) units


@dataclass
class DeliveryMessage:
    """<delivery, quantities, arrival time> from the distributor to each kirana."""
    profile: np.ndarray              # (n, P, A) units by age
    late: np.ndarray                 # (n,) bool: arrived after the 6 am window
    drop: int                        # 1 = morning drop, 2 = noon drop


@dataclass
class DaySummaryMessage:
    """<day summary, sales, lost> from the kirana population to the consumer model."""
    demand: np.ndarray               # (n, P)
    sales: np.ndarray                # (n, P)
    lost: np.ndarray                 # (n, P)
    spill: np.ndarray                # (P,) units of demand that went to the QC app


# ---------------------------------------------------------------- city layout
def build_city(cfg: TwinConfig, rng: np.random.Generator):
    """Place localities (power feeders) on a grid and shops inside them."""
    gx, gy = cfg.city.grid
    n_loc = gx * gy
    # uneven shop counts per locality
    weights = rng.dirichlet(np.full(n_loc, 6.0))
    locality = rng.choice(n_loc, size=cfg.city.n_kirana, p=weights)
    locality.sort()
    lx, ly = locality % gx, locality // gx
    pos = np.stack([lx + rng.uniform(0.08, 0.92, locality.size),
                    ly + rng.uniform(0.30, 0.94, locality.size)], axis=1)
    route = locality * cfg.city.n_routes // n_loc
    # some feeders are weaker than others: per-locality outage risk multiplier
    feeder_risk = rng.lognormal(0.0, 0.35, n_loc)
    feeder_risk /= feeder_risk.mean()
    return locality, route, pos, feeder_risk


# ---------------------------------------------------------------- kirana actors
class KiranaActors:
    def __init__(self, cfg: TwinConfig, scen: Scenario, rng: np.random.Generator,
                 locality: np.ndarray, route: np.ndarray, pos: np.ndarray,
                 feeder_risk: np.ndarray, city_rng: np.random.Generator):
        """rng: this year's random events. city_rng: the fixed city (shop sizes, fridges,
        habits), identical in every simulated year."""
        kp = cfg.kirana
        self.cfg, self.kp, self.rng = cfg, kp, rng
        n, A = cfg.city.n_kirana, cfg.max_age
        self.n = n
        self.locality, self.route, self.pos = locality, route, pos

        # ---- fixed state, drawn once per shop (heterogeneity)
        share = city_rng.lognormal(0.0, kp.size_sigma, n)
        self.share = share / share.sum()
        vol = cfg.arr("fridge_volume")
        pc = cfg.arr("per_capita")
        exp_daily_vol = self.share * cfg.city.households[0] * float((pc * vol).sum())
        self.fridge_cap = exp_daily_vol * kp.fridge_cover_mean * city_rng.lognormal(0.0, kp.fridge_sigma, n)
        self.k_base = city_rng.uniform(kp.k_base_low, kp.k_base_high, n)
        if scen.kirana_returns:
            self.k_base = self.k_base + kp.returns_k_bonus

        # ---- R8: brand fridges with battery backup (strategic knob)
        self.battery = np.zeros(n, dtype=bool)
        m = int(round(scen.battery_fridge_share * n))
        if m > 0:
            if scen.battery_targeting == "risk":
                order = np.lexsort((city_rng.random(n), -feeder_risk[locality]))
            else:
                order = city_rng.permutation(n)
            chosen = order[:m]
            self.battery[chosen] = True
            self.fridge_cap[chosen] *= 1.25
        self.returns = scen.kirana_returns

        # ---- changing state
        self.k = self.k_base.copy()
        self.stock = np.zeros((n, N_PRODUCTS, A), dtype=np.int64)
        exp_daily = self.share[:, None] * cfg.city.households[0] * pc[None, :]
        self.stock[:, :, 1] = np.round(exp_daily).astype(np.int64)
        self.last_sales = np.round(exp_daily).astype(np.int64)
        self.memory = exp_daily.copy()          # shopkeeper's feel for normal daily sales
        self.last_lost = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.today_demand = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.today_sales = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.today_lost = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.today_spill = np.zeros(N_PRODUCTS, dtype=np.int64)
        self.last_late = np.zeros(n, dtype=bool)
        self.last_waste = np.zeros((n, N_PRODUCTS), dtype=np.int64)
        self.last_outage_hours = np.zeros(n)
        self.event_log: list[tuple] = []     # optional, for the shop inspector

    # ------------------------------------------------------------ helpers
    def stock_volume(self) -> np.ndarray:
        return (self.stock.sum(axis=-1) * self.cfg.arr("fridge_volume")[None, :]).sum(axis=1)

    def begin_day(self):
        self.today_demand[:] = 0
        self.today_sales[:] = 0
        self.today_lost[:] = 0
        self.today_spill[:] = 0
        self.today_spoiled = np.zeros((self.n, N_PRODUCTS), dtype=np.int64)

    # ------------------------------------------------------------ R1, R2
    def on_delivery(self, msg: DeliveryMessage) -> np.ndarray:
        """Accept what fits in the fridge; hand the rest back to the van (returned)."""
        vol = self.cfg.arr("fridge_volume")
        free = np.maximum(0.0, self.fridge_cap - self.stock_volume())
        incoming_vol = (msg.profile.sum(axis=-1) * vol[None, :]).sum(axis=1)
        frac = np.where(incoming_vol > free, free / np.maximum(incoming_vol, 1e-9), 1.0)
        accepted = np.floor(msg.profile * frac[:, None, None]).astype(np.int64)
        refused = msg.profile - accepted
        self.stock += accepted
        if msg.drop == 1:
            late = msg.late
            self.last_late = late.copy()
            bump = late & (self.rng.random(self.n) < self.kp.p_k_late)
            self.k = np.where(bump, self.k + self.kp.k_late_step, self.k)
        return refused

    # ------------------------------------------------------------ R3, R4
    def on_customers(self, mean_demand: np.ndarray, part: np.ndarray):
        """Customers arrive: Poisson per shop and product. `part` (P,) is the share of
        the day's demand arriving in this half of the day."""
        lam = mean_demand * part[None, :]
        d = self.rng.poisson(lam)
        on_hand = self.stock.sum(axis=-1)
        sold = np.minimum(d, on_hand)
        _, self.stock = fifo_take(self.stock, sold)
        lost = d - sold
        spill = self.rng.binomial(lost, P_SPILL).sum(axis=0)
        self.today_demand += d
        self.today_sales += sold
        self.today_lost += lost
        self.today_spill += spill
        return spill

    # ------------------------------------------------------------ R10
    def on_power_cut(self, unprotected_hours: np.ndarray):
        """on power cut && no battery cover  do  stock in the dead fridge loses shelf life;
        units that turn sour are thrown out (customers will not buy them).
        unprotected_hours (n,): hours without refrigeration today."""
        self.last_outage_hours = unprotected_hours
        per4 = self.cfg.arr("outage_age_per_4h")
        extra_f = unprotected_hours[:, None] / 4.0 * per4[None, :]
        whole = np.floor(extra_f)
        extra = (whole + (self.rng.random(extra_f.shape) < (extra_f - whole))).astype(np.int64)
        shelf = np.asarray(self.cfg.products.shelf_life)
        spoiled, self.stock = age_and_expire(self.stock, shelf, extra, base=0)
        self.today_spoiled = spoiled
        return spoiled

    # ------------------------------------------------------------ R5, R6, R9
    def on_close(self):
        """Age stock one day, discard expired units, update caution k."""
        shelf = np.asarray(self.cfg.products.shelf_life)
        expired, self.stock = age_and_expire(self.stock, shelf)
        expired = expired + self.today_spoiled
        self.last_waste = expired

        held = self.today_sales + expired
        heavy = (expired.sum(axis=1) > self.kp.heavy_waste_share * np.maximum(held.sum(axis=1), 1))
        cut = heavy & (self.rng.random(self.n) < self.kp.p_k_waste)
        self.k = np.where(cut, self.k - self.kp.k_waste_step, self.k)
        self.k = self.k_base + (self.k - self.k_base) * (1 - self.kp.k_decay)
        self.k = np.clip(self.k, self.kp.k_min, self.kp.k_max)

        self.last_sales = self.today_sales.copy()
        self.last_lost = self.today_lost.copy()
        returned = expired if self.returns else np.zeros_like(expired)
        discarded = expired - returned
        return discarded, returned

    # ------------------------------------------------------------ R7
    def on_order_time(self, twice_daily: bool) -> OrderMessage:
        seen = self.today_sales + self.kp.lost_weight * self.today_lost
        m = self.kp.memory
        self.memory = m * self.memory + (1 - m) * seen
        f = 0.5 * seen + 0.5 * self.memory              # yesterday weighs heavily, memory steadies it
        cover = np.asarray(self.kp.cover_days)[None, :]
        target = np.ceil(self.k[:, None] * f * cover + self.kp.buffer_sd * np.sqrt(f * cover))
        on_hand = self.stock.sum(axis=-1)
        want = np.maximum(0, target - on_hand)
        vol = self.cfg.arr("fridge_volume")
        free = np.maximum(0.0, self.fridge_cap - self.stock_volume())
        if twice_daily:
            # a noon drop is coming, so the morning's sales free space before it arrives
            ms = self.cfg.arr("morning_share")
            free = free + (self.today_sales * ms[None, :] * vol[None, :]).sum(axis=1)
        need = (want * vol[None, :]).sum(axis=1)
        scale = np.where(need > free, free / np.maximum(need, 1e-9), 1.0)
        qty = np.floor(want * scale[:, None]).astype(np.int64)
        return OrderMessage(qty=qty)


# ---------------------------------------------------------------- distributor actor
class Distributor:
    """Aggregates kirana orders, rations short shipments, and runs the morning routes.

    on <order> from shops        do  sum by product; send one order to the plant
    on <shipment> from the plant do  ration pro rata to orders; split into drops;
                                     each route misses 6 am @ p_late (one draw per route)
    """

    def __init__(self, cfg: TwinConfig, rng: np.random.Generator, route: np.ndarray):
        self.cfg, self.rng, self.route = cfg, rng, route
        self.pending = np.zeros((route.size, N_PRODUCTS), dtype=np.int64)

    def on_orders(self, msg: OrderMessage) -> np.ndarray:
        self.pending = msg.qty.copy()
        return self.pending.sum(axis=0)

    def ration(self, shipped: np.ndarray) -> np.ndarray:
        """Integer pro-rata allocation of shipped (P,) units to shops (n, P)."""
        orders = self.pending
        tot = orders.sum(axis=0)
        alloc = np.zeros_like(orders)
        for p in range(N_PRODUCTS):
            if tot[p] == 0:
                continue
            if shipped[p] >= tot[p]:
                alloc[:, p] = orders[:, p]
                continue
            exact = orders[:, p] * shipped[p] / tot[p]
            base = np.floor(exact).astype(np.int64)
            rem = int(shipped[p] - base.sum())
            if rem > 0:
                frac = exact - base
                pick = self.rng.choice(orders.shape[0], size=rem, replace=False,
                                       p=frac / frac.sum()) if frac.sum() > 0 else []
                base[pick] += 1
            alloc[:, p] = base
        return alloc

    def split_profile(self, alloc: np.ndarray, profile: np.ndarray) -> np.ndarray:
        """Hand each shop its units with ages taken from the shipment's age profile.
        Shops are served in a random order each day so nobody always gets the oldest."""
        n, P = alloc.shape
        A = profile.shape[-1]
        out = np.zeros((n, P, A), dtype=np.int64)
        perm = self.rng.permutation(n)
        for p in range(P):
            q = alloc[perm, p]
            ends = np.cumsum(q)
            starts = ends - q
            # shipment laid out oldest -> newest
            layer = profile[p, ::-1]
            bend = np.cumsum(layer)
            bstart = bend - layer
            ov = np.clip(np.minimum(ends[:, None], bend[None, :]) -
                         np.maximum(starts[:, None], bstart[None, :]), 0, None)
            out[perm, p, :] = ov[:, ::-1]
        return out

    def late_flags(self, heat: bool) -> np.ndarray:
        kp = self.cfg.kirana
        p = kp.p_late_heat if heat else kp.p_late_normal
        late_route = self.rng.random(self.cfg.city.n_routes) < p
        return late_route[self.route]
