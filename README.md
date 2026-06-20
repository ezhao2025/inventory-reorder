# inventory-reorder

A HUD v6 environment for a demand-driven inventory loop: **an agent manages reordering for a
small catalog of SKUs over a simulated season, scored on a true reward function** — holding cost
vs. stockout cost — not pass/fail.

## What the agent does

Each task is a seeded `season`. The agent has three tools:

- `read_catalog()` — per-SKU holding cost, stockout penalty, unit cost, lead time, starting stock.
- `peek_recent_demand(days)` — a sample of early-season demand to calibrate against.
- `submit_policy(targets)` — order-up-to-S target levels, one per SKU.

The verifier then runs the full season deterministically: each day it tops (on-hand + in-transit)
up to the submitted target, demand hits (Poisson, seasonal, with hidden spikes the agent can't see),
and per-day holding + stockout + order costs accrue.

## Reward

```
episode cost = sum over days, SKUs of (holding + stockout + order cost)
reward       = (agent - do_nothing) / (optimal - do_nothing),  clamped to [0, 1]
```

- **0.0** = no better than never ordering (or worse — over-ordering can go below the floor).
- **1.0** = matches the best achievable order-up-to-S policy (found by a per-SKU sweep at grade time).

So the reward reads as **fraction of optimal**. The unclamped `raw_score` is kept in the result
`info` (a negative value means the policy is actively worse than doing nothing — e.g. over-ordering
on a high-holding-cost SKU).

## Tasks

| task | SKUs | difficulty |
|------|------|-----------|
| `inventory_easy`   | 1 | stable demand, short lead time |
| `inventory_medium` | 3 | mild seasonality, one demand spike |
| `inventory_hard`   | 5 | long lead times, two spikes, tight holding/stockout ratio |

The environment discriminates: a fixed naive policy scores ~0.83 on easy but ~0.40 on hard.

## Run

```bash
uv sync
uv run hud eval tasks.py claude --task-ids inventory_easy   -y --max-steps 8
uv run hud eval tasks.py claude --task-ids inventory_medium -y --max-steps 8
uv run hud eval tasks.py claude --task-ids inventory_hard   -y --max-steps 10 --group-size 5
```

## Reading the results

**Use mean reward, not success rate.** This is a continuous-reward environment; HUD's "success rate"
applies a binary threshold and will show 0% for a perfectly valid score like 0.53. The reward is the
metric.

**Single runs are noisy.** The demand is seeded (the environment is deterministic), but the agent's
policy choice is not — run with `--group-size 5` and compare means before drawing conclusions, e.g.
across models.

## Known characteristic (honest note)

This is a **one-shot estimation** task: the agent reads costs + a demand sample, then commits static
targets with no day-by-day feedback. Both frontier models can do the arithmetic, so the variance
between them comes mostly from how cautious each is about safety stock — meaning the environment does
not cleanly *rank* models. To make it model-discriminating, the next design step would be a
**sequential** version: let the agent place orders day-by-day and observe demand as it unfolds, so a
better model can adapt to spikes rather than committing blind.

## Tests

```bash
uv run pytest tests/ -q
```

Offline, deterministic, no keys: determinism, the do-nothing/optimal endpoints, bounded reward, the
difficulty gradient, and JSON-serializability of the result (numpy types must not leak over RPC).
