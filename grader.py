"""
grader.py — runs a policy through an episode and scores it (Step 3).

run_episode: drives one policy through the sim, returns total cost.
normalized_score: turns raw cost into the interpretable headline number
  by placing the agent between the do-nothing floor and the
  order-up-to-S reference:

      score = (agent - donothing) / (reference - donothing)

  where each term is the EPISODE REWARD = -total_cost (less negative
  is better). Read it as:
      ~0.0  -> no better than doing nothing
      ~1.0  -> matched your heuristic
      >1.0  -> beat your heuristic
"""

from environment import InventoryEnv


def run_episode(config, policy):
    """Drive one policy through one season. Returns total_cost (>=0)."""
    env = InventoryEnv(config)
    obs = env.reset()
    total_cost = 0.0
    while not env.done:
        action = policy(obs)
        obs, day_cost = env.step(action)
        total_cost += day_cost
    return total_cost


def normalized_score(config, agent_policy, donothing_policy, reference_policy):
    """Place the agent between the floor and the reference baseline."""
    agent_cost = run_episode(config, agent_policy)
    floor_cost = run_episode(config, donothing_policy)
    ref_cost = run_episode(config, reference_policy)

    # convert to rewards (negative cost): higher is better
    agent_r, floor_r, ref_r = -agent_cost, -floor_cost, -ref_cost
    denom = (ref_r - floor_r)
    if denom == 0:
        return 0.0, agent_cost, floor_cost, ref_cost
    score = (agent_r - floor_r) / denom
    return score, agent_cost, floor_cost, ref_cost
