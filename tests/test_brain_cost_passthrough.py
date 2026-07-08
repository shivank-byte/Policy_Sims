"""
Regression tests for brain.py's heuristic firm pricing.

These exist specifically to prevent the second bug found during development:
the firm was applying a flat +0.05 price bump every single round that
subsidy_cut stayed active, rather than a bounded, one-time cost pass-through
the round the cost actually changes.
"""
import pytest
from policysim.brain import AgentBrain


def _firm_state(subsidy_rate, last_effective_cost, active_policies=(), unit_cost=10.0):
    return {
        "role": "firm", "kind": "small_shop", "price": 12.0, "wage": 11.0,
        "min_wage": 11.0, "unit_cost": unit_cost, "employees": 3, "max_employees": 6,
        "last_demand": 60.0,  # utilization exactly 1.0 at capacity_per_employee=20 -> neutral demand pricing
        "capacity_per_employee": 20.0, "subsidy_rate": subsidy_rate,
        "last_effective_cost": last_effective_cost, "active_policies": list(active_policies),
    }


def test_no_cost_change_means_no_passthrough_bump():
    """If effective cost hasn't changed since last round, price shouldn't
    drift just because a policy label is still 'active'."""
    brain = AgentBrain(backend="heuristic")
    unit_cost = 10.0
    subsidy_rate = 0.08
    effective_cost = unit_cost * (1 - subsidy_rate)
    state = _firm_state(subsidy_rate, last_effective_cost=effective_cost, active_policies=["subsidy_cut"])
    decision = brain._heuristic_firm(state)
    # price should stay close to the input price (12.0) -- only noise, no cost-shock term
    assert abs(decision["price"] - 12.0) < 0.5


def test_cost_increase_produces_one_time_bounded_bump():
    """A cost increase from last round should shift price up ONCE, bounded,
    not runaway."""
    brain = AgentBrain(backend="heuristic")
    unit_cost = 10.0
    old_subsidy, new_subsidy = 0.10, 0.08  # cost went up because subsidy fell
    old_cost = unit_cost * (1 - old_subsidy)
    new_cost = unit_cost * (1 - new_subsidy)
    state = _firm_state(new_subsidy, last_effective_cost=old_cost, active_policies=["subsidy_cut"])
    decision = brain._heuristic_firm(state)
    assert decision["price"] > 12.0, "a cost increase should push price up"
    # clamp in brain.py bounds any single round's price_delta to +/-10%
    assert decision["price"] <= 12.0 * 1.15  # generous ceiling incl. demand term + noise


def test_repeated_rounds_at_same_cost_stabilize_price():
    """Simulates 5 consecutive rounds where the cost shock happened once (round
    1) and then stays flat -- price should stabilize, not keep climbing."""
    brain = AgentBrain(backend="heuristic")
    unit_cost = 10.0
    subsidy_rate = 0.08
    effective_cost = unit_cost * (1 - subsidy_rate)

    prices = []
    price = 12.0
    last_cost = None
    for round_num in range(5):
        state = _firm_state(subsidy_rate, last_effective_cost=last_cost, active_policies=["subsidy_cut"])
        state["price"] = price
        decision = brain._heuristic_firm(state)
        price = decision["price"]
        prices.append(price)
        last_cost = effective_cost  # cost shock only happens once, at round 0

    # after the first round's one-time shift, later rounds should not keep
    # compounding a further cost-driven increase -- price changes should shrink,
    # not grow, across the tail of the window
    early_move = abs(prices[1] - prices[0])
    late_move = abs(prices[-1] - prices[-2])
    assert late_move <= early_move + 0.5, "price appears to keep climbing instead of stabilizing"


def test_subsidy_restoration_produces_price_relief_not_bump():
    """If the subsidy is restored (cost falls), price should trend down, not
    still get the old 'subsidy_cut pushes prices up' treatment."""
    brain = AgentBrain(backend="heuristic")
    unit_cost = 10.0
    old_subsidy, new_subsidy = 0.05, 0.10  # subsidy restored, cost falls
    old_cost = unit_cost * (1 - old_subsidy)
    new_cost = unit_cost * (1 - new_subsidy)
    state = _firm_state(new_subsidy, last_effective_cost=old_cost, active_policies=[])
    decision = brain._heuristic_firm(state)
    assert decision["price"] < 12.0, "a cost decrease should ease price down"
