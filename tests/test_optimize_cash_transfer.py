import pytest

from experiments.optimize_cash_transfer import outlay, N_LOW, N_MID, ACTIVE_ROUNDS, grid_search


def test_outlay_zero_at_zero_magnitude():
    assert outlay(0.0) == 0.0


def test_outlay_scales_linearly_with_magnitude():
    assert outlay(1000.0) == pytest.approx(2 * outlay(500.0))


def test_outlay_matches_the_policy_apply_formula():
    """Cross-check against the actual formula in policies.py's cash_transfer
    apply function: cash_transfer_low += m, cash_transfer_mid += 0.4*m,
    applied every active round."""
    m = 500.0
    expected_per_round = m * N_LOW + 0.4 * m * N_MID
    assert outlay(m) == pytest.approx(expected_per_round * ACTIVE_ROUNDS)


def test_grid_search_returns_a_feasible_optimum_within_constraints():
    # Small grid (coarse step, low m_max) so this runs fast as a regression test.
    baseline, best, rows = grid_search(budget=10_000_000, epsilon=0.05, m_max=500, step=250)
    assert best is not None, "m=0 should always be feasible under a generous budget/epsilon"
    m_star, result_star, spend_star = best
    assert spend_star <= 10_000_000
    assert result_star["gini"] - baseline["gini"] <= 0.05 + 1e-9
    assert len(rows) >= 1


def test_grid_search_is_infeasible_under_a_near_zero_budget():
    baseline, best, rows = grid_search(budget=1.0, epsilon=0.05, m_max=500, step=250)
    # Only m=0 costs nothing, so it should be the only feasible point.
    m_star, result_star, spend_star = best
    assert m_star == 0.0
