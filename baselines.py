"""
baselines.py — reference policies (Step 2).

A policy is just a function: given the day's observation, return an
action dict {sku_name: reorder_qty}.

We need two:
  * do_nothing  -> the floor (never reorders). Should score badly.
  * order_up_to_s -> the "decent human" heuristic. Should score well.

If these two DON'T separate when run through the sim, the reward
function / cost ratios are broken and any agent number is meaningless.
That separation check is the real purpose of this file.
"""


def do_nothing_policy(obs):
    """Never reorder. The lower-bound baseline."""
    return {name: 0 for name in obs["skus"]}


def make_order_up_to_s(S_by_sku):
    """
    Order-up-to-S (a.k.a. base-stock) policy.

    Each day, for each SKU, look at the 'inventory position' =
    on_hand + everything already in transit. Order enough to bring
    that position up to the target S. This naturally accounts for
    lead time: stuff already ordered counts toward the target, so we
    don't double-order while a shipment is en route.

    S_by_sku: dict mapping sku_name -> target level S.
    """
    def policy(obs):
        action = {}
        for name, s in obs["skus"].items():
            in_transit = sum(o["qty"] for o in s["in_transit"])
            position = s["on_hand"] + in_transit
            target = S_by_sku[name]
            action[name] = max(0, target - position)
        return action
    return policy
