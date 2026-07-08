import pytest

from experiments.cash_transfer_effect import run_once, run_experiment, cohens_d
from policysim.simulation import DEFAULT_HOUSEHOLDS


def test_run_once_returns_expected_metric_keys():
    result = run_once(seed=1, apply_transfer=False)
    assert set(result.keys()) == {"gini", "gini_income", "unemployment_rate", "price_level", "total_spending"}


def test_run_once_is_deterministic_given_a_seed():
    a = run_once(seed=42, apply_transfer=False)
    b = run_once(seed=42, apply_transfer=False)
    assert a == pytest.approx(b)


def test_default_config_flags_gini_income_and_unemployment_as_degenerate():
    """Regression test for a real bug: at the default calibration (0%
    unemployment always), income has zero within-group variance across
    seeds in both conditions, so a Welch's t-test on it is mathematically
    degenerate (was previously reported as t=-inf, Cohen's d=0.000 next to
    p=0.0000 -- misleading, since that combination looks like a computed
    result rather than a broken one). Both should now be explicitly
    flagged rather than silently producing nonsense statistics."""
    results = run_experiment(n_runs=8, n_rounds=10, seed_offset=1000)
    assert results["gini_income"]["degenerate"] is True
    assert results["gini_income"]["t_stat"] is None
    assert results["gini_income"]["p_value"] is None
    assert results["unemployment_rate"]["degenerate"] is True
    # gini (wealth) has real per-round spend noise even when income doesn't -- must NOT be degenerate
    assert results["gini"]["degenerate"] is False
    assert results["gini"]["t_stat"] is not None


def test_one_deterministic_group_with_one_real_group_is_not_flagged_degenerate():
    """Regression test for the *second*, subtler bug found while fixing the
    first one: an earlier fix flagged 'degenerate' whenever EITHER group
    had near-zero variance, which incorrectly suppressed a legitimate
    result (unemployment_slack.py: cash_transfer pushes every seed to
    exactly 0% unemployment, while the control group genuinely varies
    0/10/20% across seeds -- that's a normal, well-posed Welch's test,
    not a degenerate one). Only BOTH groups being near-zero variance
    should trigger the degenerate flag."""
    slack_firms = [
        dict(name="Corner Shop", kind="small_shop", price=200.0, wage=220.0,
             unit_cost=120.0, employees=2, max_employees=6, capacity_per_employee=200.0),
        dict(name="Metro Industries", kind="large_firm", price=500.0, wage=280.0,
             unit_cost=300.0, employees=5, max_employees=15, capacity_per_employee=18.0),
    ]
    results = run_experiment(n_runs=15, n_rounds=20, firm_spec=slack_firms, seed_offset=9000)
    assert results["unemployment_rate"]["degenerate"] is False
    assert results["unemployment_rate"]["t_stat"] is not None
    assert results["unemployment_rate"]["p_value"] is not None


def test_cohens_d_is_zero_for_identical_distributions():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert cohens_d(a, a) == pytest.approx(0.0)


def test_cohens_d_sign_matches_direction_of_difference():
    higher = [10.0, 11.0, 12.0, 13.0]
    lower = [1.0, 2.0, 3.0, 4.0]
    assert cohens_d(higher, lower) > 0
    assert cohens_d(lower, higher) < 0


def test_run_experiment_supports_doubled_population_for_robustness_checks():
    doubled = [dict(g, n=g["n"] * 2) for g in DEFAULT_HOUSEHOLDS]
    results = run_experiment(n_runs=6, n_rounds=8, household_spec=doubled, seed_offset=2000)
    assert set(results.keys()) == {"gini", "gini_income", "unemployment_rate", "price_level", "total_spending"}
