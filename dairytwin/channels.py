"""Plant, modern trade and quick-commerce: the stock-and-flow parts of the twin.

They are written as stocks (age-structured arrays) with random flows, as in the
proposal's stochastic stock-and-flow model. Each channel is structurally different:

* Plant          one cold store; production limited by the day's random milk supply.
* Modern trade   two tiers (brand DC -> store shelves), random DC dwell
                 (Binomial release per age batch), planogram shelf cap, forecast orders,
                 expired units returned to the brand.
* Quick-commerce no middle tier; six tiny dark stores filled straight from the plant twice
                 a day, algorithmic orders, absorbs kirana spill-over demand.
"""
from __future__ import annotations

import numpy as np

from .batches import age_and_expire, fifo_take
from .config import N_PRODUCTS, TwinConfig


class Plant:
    def __init__(self, cfg: TwinConfig, rng: np.random.Generator, mean_demand: np.ndarray):
        self.cfg, self.rng = cfg, rng
        self.stock = np.zeros((N_PRODUCTS, cfg.max_age), dtype=np.int64)
        self.capacity = np.ceil(mean_demand * cfg.supply.plant_capacity_mult).astype(np.int64)
        self.shelf = np.asarray(cfg.products.shelf_life)

    def milk_today(self, mean_mult: float) -> float:
        eps = self.rng.lognormal(0.0, self.cfg.supply.noise_sigma)
        return self.cfg.supply.mean_milk * mean_mult * eps

    def produce(self, plan: np.ndarray, milk: float, priority: np.ndarray | None):
        """Make up to `plan` units within capacity and the milk available.
        priority=None scales every product down by the same factor when milk is short;
        otherwise products are filled in the given order (staples first)."""
        r = self.cfg.arr("milk_per_unit")
        want = np.minimum(plan, self.capacity).astype(float)
        need = float((want * r).sum())
        if need <= milk:
            made = want
        elif priority is None:
            made = want * (milk / need)
        else:
            made = np.zeros(N_PRODUCTS)
            left = milk
            for p in priority:
                take = max(0.0, min(want[p], left / r[p]))
                made[p] = take
                left -= take * r[p]
        made = np.maximum(0, np.floor(made)).astype(np.int64)
        self.stock[:, 0] += made
        return made, float((made * r).sum())

    def ship(self, qty: np.ndarray) -> np.ndarray:
        """Take qty (P,) units oldest first; returns their age profile (P, A)."""
        taken, self.stock = fifo_take(self.stock, np.minimum(qty, self.stock.sum(axis=-1)))
        return taken

    def available(self) -> np.ndarray:
        return self.stock.sum(axis=-1)

    def close(self):
        expired, self.stock = age_and_expire(self.stock[None], self.shelf)
        self.stock = self.stock[0]
        return expired[0]


class ModernTrade:
    def __init__(self, cfg: TwinConfig, rng: np.random.Generator, mean_demand: np.ndarray):
        self.cfg, self.rng = cfg, rng
        A = cfg.max_age
        self.dc = np.zeros((N_PRODUCTS, A), dtype=np.int64)
        self.shelf = np.zeros((N_PRODUCTS, A), dtype=np.int64)
        self.shelf[:, 1] = np.round(mean_demand).astype(np.int64)
        self.shelf_cap = np.ceil(mean_demand * cfg.mt.shelf_cover).astype(np.int64)
        self.smoothed = mean_demand.astype(float).copy()
        self.shelf_life = np.asarray(cfg.products.shelf_life)

    def receive(self, profile: np.ndarray):
        self.dc += profile

    def release(self):
        """Random DC dwell: each unit leaves the DC today with probability p, as long as
        the store shelf has room (oldest batches first)."""
        moving = self.rng.binomial(self.dc, self.cfg.mt.dc_release_p)
        room = np.maximum(0, self.shelf_cap - self.shelf.sum(axis=-1))
        go, _ = fifo_take(moving, np.minimum(room, moving.sum(axis=-1)))
        self.dc -= go
        self.shelf += go

    def serve(self, mean_demand: np.ndarray):
        d = self.rng.poisson(mean_demand)
        sold = np.minimum(d, self.shelf.sum(axis=-1))
        _, self.shelf = fifo_take(self.shelf, sold)
        self._d, self._s = d, sold
        return d, sold

    def close(self):
        e1, dc = age_and_expire(self.dc[None], self.shelf_life)
        e2, sh = age_and_expire(self.shelf[None], self.shelf_life)
        self.dc, self.shelf = dc[0], sh[0]
        return e1[0] + e2[0]           # all returned to the brand

    def order(self) -> np.ndarray:
        a = self.cfg.mt.smoothing
        seen = self._s + 0.8 * (self._d - self._s)        # POS sees sales + most lost
        self.smoothed = a * seen + (1 - a) * self.smoothed
        pipeline = self.dc.sum(axis=-1) + self.shelf.sum(axis=-1)
        target = self.smoothed * (1 + self.cfg.mt.order_cover)       # tomorrow + cover
        return np.maximum(0, np.ceil(target - pipeline)).astype(np.int64)


class QuickCommerce:
    """Six tiny dark stores with no middle tier: the plant drops straight into them twice
    a day. Each drop fills a store only up to its free space; whatever does not fit never
    leaves the plant."""

    def __init__(self, cfg: TwinConfig, rng: np.random.Generator, mean_demand: np.ndarray,
                 city_rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        S, A = cfg.city.n_dark_stores, cfg.max_age
        self.S = S
        share = city_rng.dirichlet(np.full(S, 8.0))
        self.share = share
        self.store = np.zeros((S, N_PRODUCTS, A), dtype=np.int64)
        self.cap = np.ceil(share[:, None] * mean_demand[None, :] * cfg.qc.store_cover).astype(np.int64)
        self.store[:, :, 1] = self.cap // 2
        self.smoothed = mean_demand.astype(float).copy()
        self.shelf_life = np.asarray(cfg.products.shelf_life)

    def drop(self, plant: "Plant", allowance: np.ndarray) -> np.ndarray:
        """One drop from the plant: top every dark store up to its capacity, oldest stock
        first, within today's remaining allowance (P,). Returns units dropped (P,)."""
        room = np.maximum(0, self.cap - self.store.sum(axis=-1))           # (S, P)
        avail = np.minimum(allowance, plant.available())                   # (P,)
        tot = room.sum(axis=0)
        give = np.where(tot[None, :] > avail[None, :],
                        np.floor(room * avail[None, :] / np.maximum(tot[None, :], 1)),
                        room).astype(np.int64)
        for s in range(self.S):
            self.store[s] += plant.ship(give[s])
        return give.sum(axis=0)

    def serve(self, mean_demand: np.ndarray, part: np.ndarray, extra: np.ndarray | None = None):
        lam = self.share[:, None] * mean_demand[None, :] * part[None, :]
        d = self.rng.poisson(lam)
        if extra is not None:
            # spill-over customers from kiranas land on random dark stores
            for p in range(N_PRODUCTS):
                if extra[p] > 0:
                    d[:, p] += self.rng.multinomial(int(extra[p]), self.share)
        sold = np.minimum(d, self.store.sum(axis=-1))
        _, self.store = fifo_take(self.store, sold)
        return d.sum(axis=0), sold.sum(axis=0)

    def close(self):
        expired, self.store = age_and_expire(self.store, self.shelf_life)
        return expired.sum(axis=0)

    def order(self, demand_today: np.ndarray, sales_today: np.ndarray) -> np.ndarray:
        """Tomorrow's request to the plant: smoothed true demand plus cover, net of the
        stock already in the dark stores."""
        a = self.cfg.qc.smoothing
        self.smoothed = a * demand_today + (1 - a) * self.smoothed       # sees true demand
        on_hand = self.store.sum(axis=(0, 2))
        return np.maximum(0, np.ceil(self.smoothed * (1 + self.cfg.qc.order_cover) - on_hand)
                          ).astype(np.int64)
