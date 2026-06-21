"""
sequential.py — a day-by-day sequential wrapper over the inventory simulator.

The one-shot version has the agent submit static order-up-to-S targets once.
This version puts the agent INSIDE the season: it observes state at each decision
point, places orders for that point, and the simulator runs forward `interval`
days before the next decision. Cost accumulates across the whole season.

This is what makes the environment adaptive — the agent can react to a demand
spike it observes, instead of guessing the whole policy blind up front.

Decision interval is configurable: interval=7 -> weekly decisions (~9 per
60-day season), interval=1 -> daily (~60). The agent's order is placed on the
decision day; the intervening days run with no new orders (the in-transit
pipeline plays out), which mirrors a real periodic-review inventory system.
"""

from environment import InventoryEnv, SeasonConfig


class SequentialSeason:
    """Drives an InventoryEnv in decision-interval chunks.

    Usage:
        seq = SequentialSeason(config, interval=7)
        state = seq.reset()
        while not seq.done:
            orders = {sku_name: qty, ...}   # agent's decision this period
            state, period_cost = seq.step(orders)
        total = seq.total_cost
    """

    def __init__(self, config: SeasonConfig, interval: int = 7):
        self.config = config
        self.interval = max(1, int(interval))
        self.reset()

    def reset(self):
        self.env = InventoryEnv(self.config)
        self._obs = self.env.reset()
        self.done = False
        self.total_cost = 0.0
        self.decisions_made = 0
        return self._state()

    def _state(self):
        """What the agent sees at a decision point — the sim's observation plus
        how far through the season we are."""
        if self.done:
            return None
        obs = dict(self._obs)
        obs["days_remaining"] = self.config.season_length - self.env.t
        obs["decision_interval"] = self.interval
        return obs

    def step(self, orders: dict):
        """Place `orders` on the current decision day, then run `interval` days
        (or to season end). Returns (next_state, cost_incurred_this_period)."""
        if self.done:
            raise RuntimeError("Season is over; call reset().")

        period_cost = 0.0
        # day 0 of the period: apply the agent's order
        action = {k: max(0, int(v)) for k, v in orders.items()}
        _, c = self.env.step(action)
        period_cost += c
        # remaining days of the period: no new orders (periodic review)
        for _ in range(self.interval - 1):
            if self.env.done:
                break
            _, c = self.env.step({})
            period_cost += c

        self.decisions_made += 1
        self.total_cost += period_cost
        if self.env.done:
            self.done = True
            self._obs = None
        else:
            self._obs = self.env._observe()
        return self._state(), float(period_cost)


def run_sequential(config: SeasonConfig, policy_fn, interval: int = 7) -> float:
    """Drive a policy callable through a sequential season. Returns total cost.

    policy_fn(state) -> {sku_name: order_qty}. Used for baselines and the
    optimal reference (a base-stock policy adapts naturally here: it reorders
    up to target at each decision point using the state it sees)."""
    seq = SequentialSeason(config, interval=interval)
    state = seq.reset()
    while not seq.done:
        orders = policy_fn(state)
        state, _ = seq.step(orders)
    return seq.total_cost
