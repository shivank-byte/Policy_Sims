import math
import numpy as np
import pandas as pd
import pytest

from policysim.stats_engine import (
    gini, simple_linear_trend, welch_ttest, bootstrap_mean_diff_ci,
    before_after_comparison, price_elasticity_of_demand, magnitude_sensitivity_sweep,
)


def test_gini_perfect_equality_is_zero():
    assert gini([10, 10, 10, 10]) == pytest.approx(0.0, abs=1e-9)


def test_gini_maximal_inequality_approaches_one():
    # one household holds everything, rest hold nothing
    g = gini([0, 0, 0, 100])
    assert g > 0.7  # not exactly 1 with finite n, but clearly high


def test_gini_handles_negative_wealth_by_shifting():
    # should not raise, and should still be bounded in [0, 1]
    g = gini([-50, 0, 50, 100])
    assert 0.0 <= g <= 1.0


def test_simple_linear_trend_recovers_known_slope():
    y = [1, 3, 5, 7, 9]  # perfect line, slope=2, intercept=1
    trend = simple_linear_trend(y)
    assert trend["slope"] == pytest.approx(2.0)
    assert trend["intercept"] == pytest.approx(1.0)
    assert trend["r2"] == pytest.approx(1.0)


def test_welch_ttest_matches_known_textbook_case():
    # classic separated-means example: t should be exactly 9.0, df=8
    a = [1, 2, 3, 4, 5]
    b = [10, 11, 12, 13, 14]
    result = welch_ttest(a, b)
    assert result["t_stat"] == pytest.approx(9.0, abs=1e-6)
    assert result["df"] == pytest.approx(8.0, abs=1e-6)
    assert result["p_value"] < 0.001


def test_welch_ttest_identical_samples_gives_high_p_value():
    a = [5.0, 5.1, 4.9, 5.05, 4.95]
    b = [5.0, 5.1, 4.9, 5.05, 4.95]
    result = welch_ttest(a, b)
    assert result["t_stat"] == pytest.approx(0.0, abs=1e-9)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_welch_ttest_p_value_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(0, 1, 8)
        b = rng.normal(rng.uniform(-3, 3), 1, 8)
        p = welch_ttest(a, b)["p_value"]
        assert 0.0 <= p <= 1.0


def test_bootstrap_ci_contains_true_difference_most_of_the_time():
    """Not a flaky-proof test of coverage, but a sanity check that a big,
    obvious mean difference is reliably captured by the CI."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 30)
    b = rng.normal(5, 1, 30)  # true difference = 5
    ci = bootstrap_mean_diff_ci(a, b, n_boot=1000, seed=2)
    assert ci["ci_low"] < 5 < ci["ci_high"]


def test_before_after_comparison_flags_significant_and_nonsignificant_correctly():
    rounds = list(range(1, 11))
    # price_level: big, consistent jump after round 5 -> should be significant
    # gini: identical before/after -> should NOT be significant
    price = [1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5]
    gini_vals = [0.4] * 10
    df = pd.DataFrame({
        "round": rounds,
        "price_level": price,
        "gini": gini_vals,
        "unemployment_rate": [0.1] * 10,
        "total_spending": [1000.0] * 10,
    })
    comp = before_after_comparison(df, trigger_round=5, window=4, seed=3)
    price_row = comp[comp["metric"] == "price_level"].iloc[0]
    gini_row = comp[comp["metric"] == "gini"].iloc[0]
    assert price_row["significant"] == True  # noqa: E712
    assert gini_row["significant"] == False  # noqa: E712


def test_price_elasticity_of_demand_sign_makes_sense():
    """Price up + real quantity down should give a negative elasticity
    (standard downward-sloping demand), not a sign error."""
    rounds = list(range(1, 11))
    price = [1.0] * 5 + [1.2] * 5          # price rises 20%
    spending = [1000.0] * 5 + [1080.0] * 5  # nominal spending rises less than price
    df = pd.DataFrame({
        "round": rounds, "price_level": price, "total_spending": spending,
    })
    result = price_elasticity_of_demand(df, trigger_round=5, window=4)
    assert result["elasticity"] < 0, "demand should fall (or rise less than price) when price rises"


def test_magnitude_sensitivity_sweep_distinguishes_magnitudes():
    """Regression test at the integration level for the subsidy_cut
    saturation bug: different magnitudes must not collapse to one outcome."""
    sweep = magnitude_sensitivity_sweep(
        "subsidy_cut", [0.1, 0.2, 0.3, 0.4], rounds=12, trigger_round=5, firm_kind="small_shop",
    )
    assert sweep["final_price_level"].nunique() == len(sweep), (
        "magnitudes collapsed to the same outcome -- possible regression of the saturation bug"
    )
    assert "warning" not in sweep.attrs
