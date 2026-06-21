# inventory-reorder

A HUD v6 environment for a demand-driven inventory loop: **an agent manages reordering for a
small catalog of SKUs over a simulated season, scored on a true reward function** — holding cost
vs. stockout cost — not pass/fail. It ships in two flavors: one-shot (commit a static policy) and
sequential (reorder week-by-week, adapting to demand as it unfolds).

## What the agent does

Each task is a seeded `season` (deterministic demand: Poisson, seasonal, with hidden spikes the
agent never sees in full).

**One-shot tasks** — three tools:
- `read_catalog()` — per-SKU holding cost, stockout penalty, unit cost, lead time, starting stock.
- `peek_recent_demand(days)` — a sample of early-season demand to calibrate against.
- `submit_policy(targets)` — order-up-to-S target levels, one per SKU. Scored over the full season.

**Sequential tasks** — the agent lives inside the season:
- `get_state()` — current on-hand, recent sales, in-transit orders, days remaining.
- `place_order(orders)` — order per SKU for this week, advance one week, return new state + cost.
It repeats (~9 weekly decisions over a 60-day season) and can react to demand spikes as they appear.

## Reward

```
episode cost = sum over days, SKUs of (holding + stockout + order cost)
reward       = (agent - do_nothing) / (optimal - do_nothing),  clamped to [0, 1]
```

- **0.0** = no better than never ordering (or worse — over-ordering can go below the floor).
- **1.0** = matches the best achievable order-up-to-S policy, found by a per-SKU sweep at grade time
  (a weekly-tuned sweep for the sequential tasks).

The reward reads as **fraction of optimal**. The unclamped `raw_score` is kept in the result `info`
(a negative value means the policy is actively worse than doing nothing).

## Tasks

| task | SKUs | mode | difficulty |
|------|------|------|-----------|
| `inventory_easy`              | 1 | one-shot | stable demand, short lead time |
| `inventory_medium`            | 3 | one-shot | mild seasonality, one demand spike |
| `inventory_hard`              | 5 | one-shot | long lead times, two spikes, tight ratio |
| `inventory_sequential`        | 3 | weekly   | adapt to demand week-by-week |
| `inventory_sequential_hard`   | 5 | weekly   | long leads, multiple spikes, weekly review |

## Results (Claude Sonnet 4.6, grouped rollouts)

The environment produces a clean difficulty gradient and discriminates between policies:

| task | mean reward | n |
|------|------|---|
| `inventory_easy`       | 0.82 ± 0.13 | 10 |
| `inventory_medium`     | 0.72 ± 0.09 | 10 |
| `inventory_hard`       | 0.09 ± 0.11 | 10 |
| `inventory_sequential` | 0.78 ± 0.09 | 5  |

**Static vs. adaptive:** on the same medium season, a static one-shot policy scores ~0.72 while the
weekly-adaptive agent scores ~0.78–0.85 — adapting to demand beats committing blind.

**Model comparison (sequential task):** the environment ranks models. Opus scores 0.84 ± 0.02 vs.
Sonnet's 0.78 ± 0.09 — higher *and* far more consistent.

## Run

```bash
uv sync
uv run hud eval tasks.py claude --task-ids inventory_easy        -y --max-steps 8
uv run hud eval tasks.py claude --task-ids inventory_medium      -y --max-steps 8  --group-size 5
uv run hud eval tasks.py claude --task-ids inventory_hard        -y --max-steps 10 --group-size 5
uv run hud eval tasks.py claude --task-ids inventory_sequential  -y --max-steps 30 --group-size 5
```

## Reading the results

**Use mean reward, not success rate.** This is a continuous-reward environment; HUD's "success rate"
applies a binary threshold and will show 0% for a perfectly valid score like 0.53. The reward is the
metric.

**Single runs are noisy.** Demand is seeded (the environment is deterministic), but the agent's policy
choice is not — run with `--group-size 5+` and compare means before drawing conclusions.

## Layout

```
environment.py   the seeded day-by-day simulator (state, step, demand)
sequential.py    week-by-week wrapper for the sequential tasks
baselines.py     do-nothing + order-up-to-S reference policies
grader.py        one-shot episode runner + normalized scoring
env.py           HUD environment: tools, verifiers, the five @env.template tasks
tasks.py         task definitions (prompts + slugs)
tests/           offline verifier tests (no keys, no network)
```

## Tests

```bash
uv run pytest tests/ -q
```

Offline and deterministic: determinism, the do-nothing/optimal endpoints, bounded reward, the
difficulty gradient, and JSON-serializability of results (numpy types must not leak over RPC).
