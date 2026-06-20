"""Offline tests for the inventory-reorder verifier — no keys, no network.

    uv run pytest tests/ -q

Mirrors the template's tests/: deterministic checks that the simulator and the
reward function behave correctly, so you can validate changes without burning
eval credits.
"""
import env as E


def _bind(season):
    E._SEASON = season
    E._POLICY = None


def test_simulator_is_deterministic():
    """Same seed -> identical demand; different seed -> different."""
    from environment import InventoryEnv
    a = InventoryEnv(E._EASY_SEASON)
    b = InventoryEnv(E._EASY_SEASON)
    assert (a.demand == b.demand).all()


def test_no_policy_scores_zero():
    _bind(E._EASY_SEASON)
    E._POLICY = None
    reward, info = E._score()
    assert reward == 0.0


def test_donothing_scores_zero():
    _bind(E._EASY_SEASON)
    E._POLICY = {s.name: 0 for s in E._EASY_SEASON.skus}
    reward, _ = E._score()
    assert reward == 0.0


def test_optimal_scores_near_one():
    """The cached optimal policy should score ~1.0 by construction."""
    for season in (E._EASY_SEASON, E._MEDIUM_SEASON, E._HARD_SEASON):
        _bind(season)
        E._POLICY = dict(E._optimal_targets(season))
        reward, _ = E._score()
        assert reward >= 0.98, f"optimal scored {reward} on seed {season.seed}"


def test_reward_is_bounded():
    """Reward always in [0, 1] regardless of policy, including wild over-ordering."""
    _bind(E._HARD_SEASON)
    for mult in (0, 1, 5, 50, 500):
        E._POLICY = {s.name: s.base_demand * mult for s in E._HARD_SEASON.skus}
        reward, _ = E._score()
        assert 0.0 <= reward <= 1.0


def test_difficulty_gradient():
    """A fixed naive policy should score lower as difficulty rises (env discriminates)."""
    scores = {}
    for label, season in [("easy", E._EASY_SEASON), ("hard", E._HARD_SEASON)]:
        _bind(season)
        E._POLICY = {s.name: s.base_demand * (s.lead_time + 2) for s in season.skus}
        scores[label], _ = E._score()
    assert scores["easy"] > scores["hard"], scores


def test_result_is_json_serializable():
    """HUD sends the result over RPC — no numpy types allowed."""
    import json
    _bind(E._MEDIUM_SEASON)
    E._POLICY = {s.name: 30 for s in E._MEDIUM_SEASON.skus}
    reward, info = E._score()
    json.dumps({"reward": reward, "info": info})  # must not raise
