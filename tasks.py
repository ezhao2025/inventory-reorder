"""The three inventory-reorder tasks.

    hud eval tasks.py claude --task-ids inventory_easy   -y --max-steps 8
    hud eval tasks.py claude --task-ids inventory_medium -y --max-steps 8
    hud eval tasks.py claude --task-ids inventory_hard   -y --max-steps 8

Each task is a seeded season. The agent reads the catalog and an early-demand
sample, then submits order-up-to-S targets. Reward is a normalized cost score:
0 = no better than never ordering, 1 = matches the reference heuristic, >1 = beats it.
"""

# re-export env so `hud eval tasks.py` can find the Environment; import the templates.
from env import (env, inventory_easy, inventory_medium, inventory_hard,  # noqa: F401
                 inventory_sequential, inventory_sequential_hard)

_PROMPT = (
    "You manage inventory reordering for a small catalog of products over a 60-day season. "
    "Call read_catalog to see each product's holding cost, stockout penalty, unit cost, lead "
    "time, and starting stock. Call peek_recent_demand to sample early demand and estimate how "
    "much each product sells per day and how variable it is. Then call submit_policy with an "
    "order-up-to-S target for each product: each day the system tops (on-hand + in-transit) up "
    "to your target. Holding too much wastes money; holding too little causes costly stockouts -- "
    "and there are demand spikes later in the season you cannot see. Choose targets that balance "
    "the two given each product's lead time and demand."
)

# Produce a runnable Task from each template by CALLING it with the prompt,
# then label it with a stable slug (mirrors `_one_deal.slug = ...` in the template).
_easy = inventory_easy(prompt=_PROMPT)
_easy.slug = "inventory_easy"

_medium = inventory_medium(prompt=_PROMPT)
_medium.slug = "inventory_medium"

_hard = inventory_hard(prompt=_PROMPT)
_hard.slug = "inventory_hard"

_SEQ_PROMPT = (
    "You manage inventory for a few products over a 60-day season, reviewing stock once a week. "
    "Each week: call get_state to see current stock, recent sales, what's in transit, and days "
    "remaining; then call place_order with a quantity per product for that week. Ordering advances "
    "the simulation one week and returns the result, including the cost incurred. Repeat until the "
    "season ends. Order enough to cover demand over the coming weeks given each product's lead time "
    "— too much wastes holding cost, too little causes stockouts. Demand can spike; when you see "
    "sales jump, react. Minimize total cost across the whole season."
)
_sequential = inventory_sequential(prompt=_SEQ_PROMPT)
_sequential.slug = "inventory_sequential"

_sequential_hard = inventory_sequential_hard(prompt=_SEQ_PROMPT)
_sequential_hard.slug = "inventory_sequential_hard"

tasks = [_easy, _medium, _hard, _sequential, _sequential_hard]
