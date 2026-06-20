"""
environment.py — the inventory simulator (Step 1).

This is the "world." It does NOT score anything and knows nothing about
the agent. It only models: demand arriving, shipments arriving, stock
being consumed, and the per-day cost being recorded.

Determinism is the whole point: given the same seed + config, the demand
sequence is identical every run. That is what makes grading reproducible
and un-gameable by memorization.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SKUConfig:
    """Per-SKU settings the environment uses to run the season."""
    name: str
    base_demand: float          # average units/day before seasonality
    holding_rate: float         # cost per unit held, per day
    stockout_penalty: float     # cost per unit of unmet demand
    unit_cost: float            # cost per unit ordered (discourages over-ordering)
    lead_time: int              # days between ordering and arrival
    init_stock: int             # starting on-hand inventory
    spike_days: list = field(default_factory=list)   # days with a demand spike
    spike_mult: float = 3.0     # how much bigger demand is on a spike day


@dataclass
class SeasonConfig:
    """A whole task: a set of SKUs, a length, and a seed."""
    skus: list                  # list[SKUConfig]
    season_length: int = 60     # days in the episode
    seasonality_amp: float = 0.3  # +/- fraction swing over the season
    seed: int = 0


class InventoryEnv:
    """
    A small multi-SKU inventory simulator.

    Usage:
        env = InventoryEnv(season_config)
        obs = env.reset()
        while not env.done:
            action = {sku_name: reorder_qty, ...}
            obs, day_cost = env.step(action)
    """

    def __init__(self, config: SeasonConfig):
        self.config = config
        self.reset()

    # ---- demand generation (the hidden dynamics the agent never sees) ----
    def _generate_demand(self):
        """
        Build the full demand table up front: shape (season_length, n_skus).
        Poisson around a base rate, modulated by a smooth seasonal curve,
        with planted spikes on specific days. Seeded => reproducible.
        """
        rng = np.random.default_rng(self.config.seed)
        T = self.config.season_length
        demand = np.zeros((T, len(self.config.skus)), dtype=int)

        for j, sku in enumerate(self.config.skus):
            # smooth seasonal multiplier over the season (one gentle wave)
            t = np.arange(T)
            seasonal = 1.0 + self.config.seasonality_amp * np.sin(2 * np.pi * t / T)
            rate = sku.base_demand * seasonal
            # planted spikes on chosen days
            for d in sku.spike_days:
                if 0 <= d < T:
                    rate[d] *= sku.spike_mult
            demand[:, j] = rng.poisson(rate)
        return demand

    # ---- episode lifecycle ----
    def reset(self):
        self.t = 0
        self.done = False
        self.demand = self._generate_demand()
        self.skus = self.config.skus
        self.on_hand = np.array([s.init_stock for s in self.skus], dtype=float)
        # pending[j] = list of (arrival_day, qty) shipments in transit for SKU j
        self.pending = [[] for _ in self.skus]
        # rolling record of recent sales for the observation
        self.sales_history = []
        return self._observe()

    def _observe(self, k: int = 5):
        """What the agent sees on day t, per SKU."""
        recent = self.sales_history[-k:]
        obs = {"day": self.t, "skus": {}}
        for j, sku in enumerate(self.skus):
            sold_recent = sum(day[j] for day in recent) if recent else 0
            in_transit = [
                {"arrives_in": arr - self.t, "qty": q}
                for (arr, q) in self.pending[j] if arr > self.t
            ]
            obs["skus"][sku.name] = {
                "on_hand": int(self.on_hand[j]),
                "sold_last_k_days": int(sold_recent),
                "in_transit": in_transit,
                "holding_rate": sku.holding_rate,
                "stockout_penalty": sku.stockout_penalty,
                "unit_cost": sku.unit_cost,
                "lead_time": sku.lead_time,
            }
        return obs

    def step(self, action: dict):
        """
        Advance one day. `action` maps sku_name -> reorder qty (>=0).
        Returns (next_observation, total_cost_for_this_day).
        Order of operations matters and mirrors a real warehouse day:
          1. receive shipments due today
          2. place new orders (become in-transit, arrive after lead time)
          3. demand hits; fill from stock; unmet = stockout
          4. tally holding + stockout + order cost
        """
        if self.done:
            raise RuntimeError("Episode is over; call reset().")

        day_cost = 0.0
        sold_today = np.zeros(len(self.skus), dtype=int)

        for j, sku in enumerate(self.skus):
            # 1. receive arrivals due today
            arrived = sum(q for (arr, q) in self.pending[j] if arr == self.t)
            self.on_hand[j] += arrived
            self.pending[j] = [(arr, q) for (arr, q) in self.pending[j] if arr > self.t]

            # 2. place new order
            order_qty = max(0, int(action.get(sku.name, 0)))
            if order_qty > 0:
                self.pending[j].append((self.t + sku.lead_time, order_qty))
                day_cost += order_qty * sku.unit_cost

            # 3. demand
            d = int(self.demand[self.t, j])
            filled = min(self.on_hand[j], d)
            unmet = d - filled
            self.on_hand[j] -= filled
            sold_today[j] = filled

            # 4. costs
            day_cost += self.on_hand[j] * sku.holding_rate     # holding (end of day)
            day_cost += unmet * sku.stockout_penalty           # stockout

        self.sales_history.append(sold_today)
        self.t += 1
        if self.t >= self.config.season_length:
            self.done = True
        obs = None if self.done else self._observe()
        return obs, float(day_cost)
