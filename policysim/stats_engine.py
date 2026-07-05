"""
stats_engine.py
----------------
All the "real statistics" backing PolicySim: Gini coefficient, before/after
policy-shock comparisons, and a light trend regression. Deliberately built on
numpy/pandas only (no statsmodels requirement) so the analysis layer never
breaks even on a machine with a minimal install.
"""

from __future__ import annotations
import math
from typing import Optional, Sequence
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Significance testing (Welch's t-test + bootstrap CI), implemented on plain
# numpy/math only — no scipy/statsmodels dependency, so the analysis layer
# never breaks on a minimal install.
# --------------------------------------------------------------------------- #
def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a,b) via continued fraction
    (Numerical Recipes 6.4 / Lentz's algorithm). Used to get an exact
    Student-t p-value without needing scipy."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    front = math.exp(ln_beta) / a

    # Lentz's continued fraction for the incomplete beta
    def _cf(x, a, b):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1, a - 1
        c = 1.0
        d = 1 - qab * x / qap
        if abs(d) < FPMIN:
            d = FPMIN
        d = 1 / d
        h = d
        for m in range(1, MAXIT):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d
            if abs(d) < FPMIN:
                d = FPMIN
            c = 1 + aa / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1 / d
            delta = d * c
            h *= delta
            if abs(delta - 1) < EPS:
                break
        return h

    if x < (a + 1) / (a + b + 2):
        return front * _cf(x, a, b)
    else:
        return 1.0 - (math.exp(ln_beta) / b) * _cf(1 - x, b, a)


def _student_t_sf(t: float, df: float) -> float:
    """Two-tailed survival function (p-value) of Student's t at |t| with `df`
    degrees of freedom, i.e. P(|T| > |t|)."""
    t = abs(t)
    x = df / (df + t * t)
    p_one_tail = 0.5 * _incomplete_beta(x, df / 2, 0.5)
    return max(0.0, min(1.0, 2 * p_one_tail))


def welch_ttest(a: "Sequence[float]", b: "Sequence[float]") -> dict:
    """Welch's t-test (unequal variances) for the difference in means of two
    samples. Preferred over Student's pooled-variance t-test here because the
    'before' and 'after' windows can have different variance (a policy shock
    often changes volatility, not just the mean)."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"t_stat": float("nan"), "df": float("nan"), "p_value": float("nan")}

    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        # Both samples are internally constant. If the constants are equal,
        # there's genuinely no difference (p=1). If they differ, the samples
        # are perfectly separated -- the limiting case of infinite t, p -> 0 --
        # not "no evidence of a difference" (which p=1 would wrongly imply).
        if m1 == m2:
            return {"t_stat": 0.0, "df": float(n1 + n2 - 2), "p_value": 1.0}
        return {"t_stat": float("inf") if m2 > m1 else float("-inf"),
                "df": float(n1 + n2 - 2), "p_value": 0.0}

    t_stat = (m2 - m1) / se
    df = (v1 / n1 + v2 / n2) ** 2 / (((v1 / n1) ** 2 / (n1 - 1)) + ((v2 / n2) ** 2 / (n2 - 1)))
    p_value = _student_t_sf(t_stat, df)
    return {"t_stat": float(t_stat), "df": float(df), "p_value": float(p_value)}


def bootstrap_mean_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 2000,
                            ci: float = 0.95, seed: Optional[int] = None) -> dict:
    """Bootstrap confidence interval for the difference in means (after - before),
    resampling each group independently with replacement. Distribution-free —
    a useful cross-check on the Welch t-test's normal-theory assumption,
    especially with small per-window sample sizes (n≈5 rounds)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"ci_low": float("nan"), "ci_high": float("nan")}

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        boot_a = rng.choice(a, size=len(a), replace=True)
        boot_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = boot_b.mean() - boot_a.mean()

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(diffs, [alpha, 1 - alpha])
    return {"ci_low": float(lo), "ci_high": float(hi)}


def gini(values: Sequence[float]) -> float:
    """Standard Gini coefficient (0 = perfect equality, 1 = maximal inequality)."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0
    if np.amin(arr) < 0:
        arr = arr - np.amin(arr)  # shift so Gini is defined for negative wealth values
    arr = np.sort(arr)
    n = len(arr)
    cum = np.cumsum(arr)
    if cum[-1] == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * arr) - (n + 1) * cum[-1]) / (n * cum[-1]))


def simple_linear_trend(y: np.ndarray) -> dict:
    """OLS trend of y against round index using plain numpy (no statsmodels dependency)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return {"slope": 0.0, "intercept": float(y[0]) if n else 0.0, "r2": 0.0}
    x = np.arange(n)
    x_mean, y_mean = x.mean(), y.mean()
    denom = np.sum((x - x_mean) ** 2)
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom) if denom else 0.0
    intercept = float(y_mean - slope * x_mean)
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2}


def before_after_comparison(history_df: pd.DataFrame, trigger_round: int, window: int = 5,
                             alpha: float = 0.05, n_boot: int = 2000, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Compare key metrics in the `window` rounds before vs after a trigger round,
    with statistical backing (not just a raw mean difference):

      - Welch's t-test p-value for the difference in means (unequal variances)
      - a bootstrap 95% CI on the mean difference, as a distribution-free cross-check
      - a `significant` flag (p < alpha AND the bootstrap CI excludes 0)

    Returns a tidy DataFrame: metric | before | after | pct_change | p_value |
    ci_low | ci_high | significant | n_before | n_after
    """
    metrics = ["price_level", "gini", "unemployment_rate", "total_spending"]
    before = history_df[(history_df["round"] >= trigger_round - window) & (history_df["round"] < trigger_round)]
    after = history_df[(history_df["round"] > trigger_round) & (history_df["round"] <= trigger_round + window)]

    rows = []
    for m in metrics:
        if m not in history_df.columns:
            continue
        b_vals = before[m].to_numpy(dtype=float) if len(before) else np.array([])
        a_vals = after[m].to_numpy(dtype=float) if len(after) else np.array([])
        b_mean = b_vals.mean() if len(b_vals) else float("nan")
        a_mean = a_vals.mean() if len(a_vals) else float("nan")
        pct = ((a_mean - b_mean) / b_mean * 100) if b_mean not in (0, None) and not pd.isna(b_mean) and b_mean != 0 else float("nan")

        tt = welch_ttest(b_vals, a_vals)
        boot = bootstrap_mean_diff_ci(b_vals, a_vals, n_boot=n_boot, seed=seed)
        p_value = tt["p_value"]
        ci_low, ci_high = boot["ci_low"], boot["ci_high"]
        ci_excludes_zero = (not pd.isna(ci_low)) and (ci_low > 0 or ci_high < 0)
        significant = bool((not pd.isna(p_value)) and p_value < alpha and ci_excludes_zero)

        rows.append({
            "metric": m,
            "before": round(b_mean, 4) if not pd.isna(b_mean) else b_mean,
            "after": round(a_mean, 4) if not pd.isna(a_mean) else a_mean,
            "pct_change": round(pct, 2) if not pd.isna(pct) else pct,
            "p_value": round(p_value, 4) if not pd.isna(p_value) else p_value,
            "ci_low": round(ci_low, 4) if not pd.isna(ci_low) else ci_low,
            "ci_high": round(ci_high, 4) if not pd.isna(ci_high) else ci_high,
            "significant": significant,
            "n_before": len(b_vals),
            "n_after": len(a_vals),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Elasticity sanity check: does the simulated economy's demand response fall
# within textbook price-elasticity ranges, or is something in the pricing/
# demand logic behaving unrealistically?
# --------------------------------------------------------------------------- #
# Rough textbook price-elasticity-of-demand ranges (own-price elasticity,
# i.e. %change in real quantity demanded / %change in price). These are
# widely-cited ballpark figures for teaching purposes, not precise estimates
# for any specific economy.
TEXTBOOK_ELASTICITY_RANGES = {
    "small_shop": (-1.2, -0.3),   # everyday/necessity-like goods: fairly inelastic
    "large_firm": (-2.5, -0.8),   # more discretionary/branded goods: more elastic
    "aggregate": (-2.0, -0.2),    # whole-economy blend used when not split by firm
}


def price_elasticity_of_demand(history_df: pd.DataFrame, trigger_round: int, window: int = 5,
                                firm_kind: str = "aggregate") -> dict:
    """
    Estimates the simulated own-price elasticity of demand around a policy
    shock: (%change in REAL quantity demanded) / (%change in price level).

    Real quantity is approximated as total_spending / price_level (nominal
    spending deflated by the price level), since total_spending alone is a
    nominal (P*Q) figure and would conflate price and volume effects.

    Flags whether the estimate falls inside a textbook-plausible range for
    `firm_kind` — this is a sanity check on the simulation's pricing/demand
    logic, not a claim that the "true" elasticity of a toy economy is
    well-defined.
    """
    before = history_df[(history_df["round"] >= trigger_round - window) & (history_df["round"] < trigger_round)]
    after = history_df[(history_df["round"] > trigger_round) & (history_df["round"] <= trigger_round + window)]
    if len(before) == 0 or len(after) == 0:
        return {"elasticity": float("nan"), "in_textbook_range": None, "textbook_range": TEXTBOOK_ELASTICITY_RANGES.get(firm_kind)}

    price_b, price_a = before["price_level"].mean(), after["price_level"].mean()
    real_qty_b = (before["total_spending"] / before["price_level"]).mean()
    real_qty_a = (after["total_spending"] / after["price_level"]).mean()

    pct_price = (price_a - price_b) / price_b if price_b else float("nan")
    pct_qty = (real_qty_a - real_qty_b) / real_qty_b if real_qty_b else float("nan")

    if pct_price in (0, None) or pd.isna(pct_price) or abs(pct_price) < 1e-6:
        elasticity = float("nan")
    else:
        elasticity = pct_qty / pct_price

    lo, hi = TEXTBOOK_ELASTICITY_RANGES.get(firm_kind, TEXTBOOK_ELASTICITY_RANGES["aggregate"])
    in_range = (lo <= elasticity <= hi) if not pd.isna(elasticity) else None

    return {
        "elasticity": round(elasticity, 3) if not pd.isna(elasticity) else elasticity,
        "pct_price_change": round(pct_price * 100, 2) if not pd.isna(pct_price) else pct_price,
        "pct_quantity_change": round(pct_qty * 100, 2) if not pd.isna(pct_qty) else pct_qty,
        "textbook_range": (lo, hi),
        "in_textbook_range": in_range,
    }


def magnitude_sensitivity_sweep(policy_id: str, magnitudes: list, rounds: int = 15, trigger_round: int = 5,
                                 backend: str = "heuristic", seed: int = 42, firm_kind: str = "aggregate") -> pd.DataFrame:
    """
    Runs the simulation once per magnitude in `magnitudes` (same seed, same
    policy, everything else held fixed) and reports the resulting price
    change and implied elasticity for each. This is the tool that catches
    bugs like a policy saturating (all magnitudes producing an identical
    outcome) or blowing up (magnitude and outcome moving in lockstep with no
    diminishing/bounded response).
    """
    # local import to avoid a circular import (simulation.py imports stats_engine)
    from .simulation import Simulation
    from .brain import AgentBrain

    rows = []
    for mag in magnitudes:
        sim = Simulation(brain=AgentBrain(backend=backend), seed=seed)
        for r in range(1, rounds + 1):
            if r == trigger_round:
                sim.apply_policy_event(policy_id, mag)
            sim.run_round()
        df = sim.history_df()
        window = min(trigger_round - 1, 5) if trigger_round > 1 else 1
        elas = price_elasticity_of_demand(df, trigger_round=trigger_round, window=window, firm_kind=firm_kind)
        final_price = df.iloc[-1]["price_level"]
        rows.append({
            "magnitude": mag,
            "final_price_level": round(float(final_price), 4),
            "pct_price_change": elas["pct_price_change"],
            "elasticity": elas["elasticity"],
            "in_textbook_range": elas["in_textbook_range"],
        })

    result = pd.DataFrame(rows)
    # flag if magnitudes are being distinguished at all (catches "saturation" bugs)
    if result["final_price_level"].nunique() == 1 and len(result) > 1:
        result.attrs["warning"] = (
            "All magnitudes produced an identical final price level — the policy "
            "may be saturating (e.g. an absolute cut hitting a floor/ceiling) "
            "rather than scaling with magnitude as expected."
        )
    return result


def multi_run_consistency(runs_df: pd.DataFrame, group_col: str = "run_id") -> pd.DataFrame:
    """
    Given a long-format DataFrame of several simulation runs stacked together,
    compute mean & std across runs for the final-round metrics -- i.e. "how
    consistent are the outcomes across repeated runs of the same policy?"
    """
    final_rows = runs_df.sort_values("round").groupby(group_col).tail(1)
    summary = final_rows[["price_level", "gini", "unemployment_rate", "total_spending"]].agg(["mean", "std"])
    return summary
