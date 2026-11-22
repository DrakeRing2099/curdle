"""Parameters of the synthetic city.

Everything here is a *calibration constant* in the sense of slide 93: fixed during a
run, chosen to be plausible for an Indian city, and tuned so the baseline shows the
stylised facts we expect (summer curd spike, bullwhip ratio > 1, outage-driven waste).

Units
-----
* product quantities: units (a 500 ml pouch, a 400 g curd cup, a 200 g paneer pack,
  a 100 g butter pack)
* milk: litres
* money: rupees
* time: days (the twin steps once per day; decisions are taken weekly)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

PRODUCTS = ["Pouch milk", "Curd", "Paneer", "Butter"]
MILK, CURD, PANEER, BUTTER = range(4)
N_PRODUCTS = 4

CHANNELS = ["Kirana", "Modern trade", "Quick-commerce"]
KIR, MT, QC = range(3)

WEATHER = ["Normal", "Heat wave"]
SUPPLY = ["Normal", "Stressed"]
FESTIVAL = ["None", "Festival"]


@dataclass
class ProductParams:
    # a unit expires when its age (days since production) reaches shelf_life
    shelf_life: List[int] = field(default_factory=lambda: [3, 5, 7, 60])
    # litres of raw milk needed per unit
    milk_per_unit: List[float] = field(default_factory=lambda: [0.5, 0.42, 1.1, 0.6])
    retail_price: List[float] = field(default_factory=lambda: [30.0, 35.0, 90.0, 58.0])
    processing_cost: List[float] = field(default_factory=lambda: [3.0, 5.0, 12.0, 6.0])
    # fridge space taken by one unit (a milk pouch = 1)
    fridge_volume: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.25])
    # units bought per household per day, normal weather, no festival
    per_capita: List[float] = field(default_factory=lambda: [0.75, 0.12, 0.035, 0.02])
    # demand multipliers in a heat wave and in a festival week
    heat_mult: List[float] = field(default_factory=lambda: [0.95, 1.6, 0.85, 0.9])
    festival_mult: List[float] = field(default_factory=lambda: [1.08, 1.1, 2.2, 1.7])
    # share of the day's demand that arrives before ~10 am
    morning_share: List[float] = field(default_factory=lambda: [0.6, 0.35, 0.3, 0.3])
    # extra days of shelf life lost per 4 hours without refrigeration
    outage_age_per_4h: List[float] = field(default_factory=lambda: [2.0, 1.5, 1.2, 0.1])


@dataclass
class SupplyParams:
    mean_milk: float = 50_000.0            # litres/day reaching the plant, average year
    flush_mult: float = 1.15               # Nov-Feb
    lean_mult: float = 0.86                # Apr-Jun
    heat_mult: float = 0.88                # cows yield less in a heat wave
    stressed_mult: float = 0.90            # disease, fodder shortage
    noise_sigma: float = 0.05              # day-to-day lognormal noise
    plant_capacity_mult: float = 2.0       # max production per product vs mean demand
    milk_cost: float = 40.0                # rupees per litre


@dataclass
class RegimeParams:
    # weather: P(Normal -> Heat) in summer weeks and other weeks; P(Heat -> Heat)
    heat_onset_summer: float = 0.28
    heat_onset_other: float = 0.015
    heat_persist: float = 0.70
    summer_weeks: tuple = (13, 25)          # roughly April to June
    # supply regime
    stress_onset: float = 0.04
    stress_persist: float = 0.70
    # festival: high chance of being in a festival week inside the windows
    festival_windows: tuple = ((42, 44), (47, 49), (2, 3))  # Diwali, wedding season, Jan
    festival_onset_in_window: float = 0.55
    festival_persist_in_window: float = 0.65
    flush_weeks: tuple = (44, 52, 0, 8)     # Nov-Feb (two ranges)


@dataclass
class CityParams:
    n_kirana: int = 240
    grid: tuple = (4, 3)                    # localities = power feeders, 4 x 3 grid
    n_routes: int = 6                       # each distributor route covers 2 localities
    n_dark_stores: int = 6
    households: List[int] = field(default_factory=lambda: [60_000, 15_000, 12_000])


@dataclass
class KiranaParams:
    # each shop's share of kirana demand is lognormal, then normalised
    size_sigma: float = 0.40
    # fridge capacity = expected daily fridge volume x this x lognormal noise
    fridge_cover_mean: float = 1.30
    fridge_sigma: float = 0.25
    k_base_low: float = 1.05
    k_base_high: float = 1.35
    k_late_step: float = 0.15               # raise k after a late delivery ...
    p_k_late: float = 0.60                  # ... with this probability
    k_waste_step: float = 0.05              # lower k after heavy waste ...
    p_k_waste: float = 0.50                 # ... with this probability
    heavy_waste_share: float = 0.10
    k_decay: float = 0.10                   # daily drift back toward k_base
    k_min: float = 0.9
    k_max: float = 2.2
    lost_weight: float = 0.5                # how much of noticed lost demand enters the order
    buffer_sd: float = 1.0                  # extra units = buffer_sd x sqrt(forecast) (keeps slow items stocked)
    memory: float = 0.8                     # weight on the shopkeeper's running memory of sales
    cover_days: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.3, 3.0])  # days stocked per product
    p_late_normal: float = 0.06             # route misses the 6 am window
    p_late_heat: float = 0.10
    returns_k_bonus: float = 0.15           # shops order bolder when unsold stock is taken back


@dataclass
class PowerParams:
    p_outage_normal: float = 0.04           # per locality per day
    p_outage_heat: float = 0.35
    hours_normal: tuple = (1.0, 3.0)
    hours_heat: tuple = (2.0, 8.0)
    battery_hours: float = 6.0              # a battery-backed brand fridge rides out this long


@dataclass
class ModernTradeParams:
    shelf_cover: float = 1.8                # shelf capacity in days of mean demand
    dc_release_p: float = 0.85              # chance a unit leaves the DC on a given day
    order_cover: float = 0.3
    smoothing: float = 0.3


@dataclass
class QuickCommerceParams:
    store_cover: float = 0.9                # tiny: capacity per drop in days of demand
    smoothing: float = 0.5
    order_cover: float = 0.3


@dataclass
class ConsumerParams:
    units_per_visit: float = 1.1
    p_migrate_k_to_q: float = 0.004         # a disappointed kirana customer moves to QC
    p_migrate_q_to_k: float = 0.004         # a disappointed QC customer moves to kirana
    p_winback: float = 0.004                # migrated customer returns on a good kirana day
    winback_fill: float = 0.97


@dataclass
class MoneyParams:
    wholesale_share: float = 0.85           # brand sells to retailers at 85% of MRP
    trip_cost: float = 1_800.0              # per route per drop
    mt_trip_cost: float = 3_000.0
    qc_trip_cost: float = 1_200.0           # per dark store per drop
    fridge_cost_per_day: float = 9.0        # amortised plain brand fridge
    battery_cost_per_day: float = 14.0      # amortised battery backup
    holding_cost: float = 0.05              # per unit per day at plant
    opening_cash: float = 5_000_000.0


@dataclass
class TargetParams:
    """The goal predicate (slide 26) and its chance constraint."""
    fill_rate: List[float] = field(default_factory=lambda: [0.92, 0.95, 0.92])  # phi per channel, rolling 4 weeks
    waste_cap: List[float] = field(default_factory=lambda: [0.06, 0.08, 0.10, 0.03])
    window_weeks: int = 4
    delta: float = 0.10                     # predicate must hold all year in >= 90% of years
    # behaviour-based goals (patterns over the weekly trace, slide 211)
    recovery_weeks: int = 2                 # after a heat wave, kirana fill back to target within this
    collapse_fill: float = 0.85             # no channel's weekly fill below this ...
    collapse_max_run: int = 2               # ... for more than this many weeks in a row
    max_sector_waste_share: float = 0.20    # no sector carries more than this share of kirana waste


@dataclass
class TwinConfig:
    products: ProductParams = field(default_factory=ProductParams)
    supply: SupplyParams = field(default_factory=SupplyParams)
    regimes: RegimeParams = field(default_factory=RegimeParams)
    city: CityParams = field(default_factory=CityParams)
    kirana: KiranaParams = field(default_factory=KiranaParams)
    power: PowerParams = field(default_factory=PowerParams)
    mt: ModernTradeParams = field(default_factory=ModernTradeParams)
    qc: QuickCommerceParams = field(default_factory=QuickCommerceParams)
    consumers: ConsumerParams = field(default_factory=ConsumerParams)
    money: MoneyParams = field(default_factory=MoneyParams)
    targets: TargetParams = field(default_factory=TargetParams)
    days: int = 364                         # 52 weeks

    # ---- derived helpers -------------------------------------------------
    def arr(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self.products, name), dtype=float)

    @property
    def max_age(self) -> int:
        return max(self.products.shelf_life) + 4


@dataclass
class Scenario:
    """Strategic knobs and forced shocks. Compared as what-if scenarios."""
    name: str = "Baseline"
    battery_fridge_share: float = 0.0       # share of kiranas given battery-backed brand fridges
    battery_targeting: str = "risk"         # "risk": highest-outage localities first; "random"
    kirana_returns: bool = False            # brand takes back expired kirana stock
    outages_enabled: bool = True            # False = counterfactual world without power cuts
    forced_weather: Dict[int, int] = field(default_factory=dict)  # week -> weather state
    forced_outages: Dict[int, Dict[int, float]] = field(default_factory=dict)  # day -> {locality: hours}

    def severe_summer(self, start_week: int = 16, length: int = 6) -> "Scenario":
        for w in range(start_week, start_week + length):
            self.forced_weather[w] = 1
        return self
