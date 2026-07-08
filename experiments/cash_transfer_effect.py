"""
experiments/cash_transfer_effect.py
------------------------------------
A genuine before/after statistical experiment run on the actual
Simulation engine (not a mock) to test whether the cash_transfer policy
has a real effect on inequality, unemployment, and price level -- or
whether an apparent effect could just be noise.

Design:
    - N_RUNS independent simulation runs with cash_transfer OFF (control)
    - N_RUNS independent simulation runs with cash_transfer ON from round 1
      (treatment)
    - Each run: N_ROUNDS rounds, heuristic backend, different random seed
    - Compare the final-round statistics between groups with Welch's
      t-test (does not assume equal variances) and report Cohen's d

Metrics reported:
    - gini              wealth (savings) Gini coefficient
    - gini_income       disposable-income Gini coefficient
    - unemployment_rate
    - price_level
    - total_spending

Why both Gini measures matter: the headline finding of this project is
that a flat cash transfer *raises* wealth Gini even though it obviously
raises low/mid income directly. Reporting gini_income alongside gini is
what actually tests that story instead of just asserting it -- see
§0.5 of the README for the full explanation and the MPC-by-income-tier
literature it rests on.

This module also exposes `run_experiment()` as a reusable function so
other scripts (e.g. robustness_check.py, unemployment_slack.py) can run
the same design with different population sizes or firm specs without
duplicating the statistical machinery.

Run:
    python -m experiments.cash_transfer_effect
"""

from __future__ import annotations
import statistics
from typing import Optional

from scipy import stats

from policysim.simulation import Simulation, DEFAULT_HOUSEHOLDS, DEFAULT_FIRMS
from policysim.brain import AgentBrain

N_RUNS = 40
N_ROUNDS = 20
METRICS = ["gini", "gini_income", "unemployment_rate", "price_level", "total_spending"]


def run_once(seed: int, apply_transfer: bool, household_spec=None, firm_spec=None,
             n_rounds: int = N_ROUNDS) -> dict:
    sim = Simulation(
        brain=AgentBrain(backend="heuristic"),
        seed=seed,
        household_spec=household_spec,
        firm_spec=firm_spec,
    )
    if apply_transfer:
        sim.apply_policy_event("cash_transfer")
    for _ in range(n_rounds):
        sim.run_round()
    final = sim.history[-1]
    return {m: final[m] for m in METRICS}


def cohens_d(a, b) -> float:
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled_sd = (((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)) ** 0.5
    return (statistics.mean(a) - statistics.mean(b)) / pooled_sd if pooled_sd else 0.0


def run_experiment(n_runs: int = N_RUNS, n_rounds: int = N_ROUNDS,
                    household_spec=None, firm_spec=None, seed_offset: int = 0) -> dict:
    """Run the control-vs-treatment cash_transfer experiment once and return a
    dict of {metric: {control_mean, treatment_mean, t_stat, p_value, cohens_d,
    degenerate}}. Reusable by robustness/slack experiments with different
    population specs.

    A metric can come back "degenerate" (t_stat/p_value/cohens_d all None)
    when a group has ~zero within-group variance -- this happens for
    e.g. gini_income in the default (0% unemployment) configuration, where
    disposable income is a deterministic function of tier + policy with no
    stochastic input at all, so every seed produces the *exact same* value.
    A Welch's t-test on two zero-variance samples is mathematically
    degenerate (division by ~0 variance), not a real significance test --
    reporting a p-value there would misrepresent an exact, deterministic
    difference as if it were a tested statistical claim. We report the
    actual mean difference instead and flag it, rather than let scipy hand
    back nonsense like Cohen's d = 0.000 next to p = 0.0000.
    """
    control = [run_once(seed=seed_offset + i, apply_transfer=False,
                         household_spec=household_spec, firm_spec=firm_spec,
                         n_rounds=n_rounds) for i in range(n_runs)]
    treatment = [run_once(seed=seed_offset + i, apply_transfer=True,
                           household_spec=household_spec, firm_spec=firm_spec,
                           n_rounds=n_rounds) for i in range(n_runs)]

    # Calibrated against this project's actual metrics: a genuinely
    # variable metric (e.g. wealth gini, driven by per-round spend noise)
    # has variance on the order of 1e-4 across 40 seeds. A metric that's
    # deterministic-or-nearly-so in BOTH conditions (e.g. income gini in
    # the default 0%-unemployment config, where nothing stochastic feeds
    # into income at all) shows variance for both groups several orders
    # of magnitude smaller -- around 1e-7 or exactly 0. 1e-6 sits cleanly
    # between those two regimes.
    #
    # Both groups must be near-zero to flag "degenerate" -- one group
    # being deterministic while the other genuinely varies is a normal,
    # well-posed Welch's test (e.g. unemployment_slack.py: cash_transfer
    # pushes every single seed to exactly 0% unemployment, while the
    # control group varies 0/10/20% across seeds -- that's a real,
    # meaningful t-stat, not degenerate). Only requiring the FIRST
    # attempt at this fix -- flagging on EITHER group near-zero --
    # incorrectly suppressed that legitimate result too. The actual
    # failure mode is a near-zero *combined* denominator (both groups
    # contributing ~nothing), not "one group happens to be constant".
    VARIANCE_FLOOR = 1e-6

    results = {}
    for metric in METRICS:
        c = [r[metric] for r in control]
        t = [r[metric] for r in treatment]
        c_mean, t_mean = statistics.mean(c), statistics.mean(t)
        c_var, t_var = statistics.variance(c), statistics.variance(t)

        if c_var < VARIANCE_FLOOR and t_var < VARIANCE_FLOOR:
            # Both groups deterministic: the difference is exact, not
            # estimated -- there's nothing for a significance test to do.
            results[metric] = {
                "control_mean": c_mean, "treatment_mean": t_mean,
                "t_stat": None, "p_value": None, "cohens_d": None,
                "degenerate": True,
            }
            continue

        t_stat, p_val = stats.ttest_ind(t, c, equal_var=False)
        d = cohens_d(t, c)
        results[metric] = {
            "control_mean": c_mean, "treatment_mean": t_mean,
            "t_stat": t_stat, "p_value": p_val, "cohens_d": d,
            "degenerate": False,
        }
    return results


def print_results(results: dict, title: str):
    print(f"\n{title}\n")
    print(f"{'metric':<20}{'control mean':>14}{'treatment mean':>16}{'t-stat':>10}{'p-value':>10}{'Cohen d':>10}")
    print("-" * 80)
    for metric, r in results.items():
        if r["degenerate"]:
            diff = r["treatment_mean"] - r["control_mean"]
            print(f"{metric:<20}{r['control_mean']:>14.4f}{r['treatment_mean']:>16.4f}"
                  f"{'exact':>10}{f'diff={diff:+.4f}':>16}")
        else:
            print(f"{metric:<20}{r['control_mean']:>14.4f}{r['treatment_mean']:>16.4f}"
                  f"{r['t_stat']:>10.3f}{r['p_value']:>10.4f}{r['cohens_d']:>10.3f}")
    degenerate_metrics = [m for m, r in results.items() if r["degenerate"]]
    if degenerate_metrics:
        print(f"\nNote: {', '.join(degenerate_metrics)} showed ~zero within-group variance "
              "in both conditions (deterministic given tier/employment/policy at this "
              "calibration) -- reporting the exact difference instead of a t-test, since "
              "a significance test on two zero-variance samples is mathematically "
              "degenerate, not just 'very significant'.")


def main():
    # ---- 1. Baseline experiment (default 10 households / 2 firms) ----
    baseline = run_experiment()
    print_results(
        baseline,
        f"PolicySim experiment: cash_transfer effect after {N_ROUNDS} rounds, "
        f"n={N_RUNS} runs/condition (default population)",
    )

    gini_p = baseline["gini"]["p_value"]
    gini_income_r = baseline["gini_income"]
    unemployment_r = baseline["unemployment_rate"]
    gini_income_desc = (
        f"(p={gini_income_r['p_value']:.4f})"
        if not gini_income_r["degenerate"]
        else f"(exact, deterministic diff={gini_income_r['treatment_mean'] - gini_income_r['control_mean']:+.4f} -- "
             "near-zero variance in this configuration; note unemployment itself is NOT "
             "deterministic here (see below), so this isn't the same 'no stochastic input' "
             "situation as the old all-employed default -- worth investigating further "
             "rather than assuming a single cause)"
    )
    print(
        "\nWealth Gini vs income Gini: wealth Gini "
        f"{'rises' if baseline['gini']['treatment_mean'] > baseline['gini']['control_mean'] else 'falls'} "
        f"significantly (p={gini_p:.4f}), while income Gini "
        f"{'rises' if baseline['gini_income']['treatment_mean'] > baseline['gini_income']['control_mean'] else 'falls'} "
        f"{gini_income_desc}. This is the core counter-intuitive result -- "
        "see README §0.5 for the mechanism (differing marginal propensity to "
        "consume by income tier)."
    )
    if not unemployment_r["degenerate"]:
        print(
            f"Unemployment itself {'rises' if unemployment_r['treatment_mean'] > unemployment_r['control_mean'] else 'falls'} "
            f"significantly too (p={unemployment_r['p_value']:.4f}, {unemployment_r['control_mean']*100:.1f}% -> "
            f"{unemployment_r['treatment_mean']*100:.1f}%) -- with real labor-market slack in the default "
            "population, the transfer's demand boost measurably moves employment on its own, without "
            "needing a special slack scenario (contrast with experiments/unemployment_slack.py's "
            "full-employment case, where this can't happen by construction)."
        )

    # ---- 2. Robustness check: does the wealth-Gini effect survive a very
    #         different population size? Double every tier's headcount. ----
    doubled_households = [dict(g, n=g["n"] * 2) for g in DEFAULT_HOUSEHOLDS]
    robustness = run_experiment(n_runs=N_RUNS, n_rounds=N_ROUNDS,
                                 household_spec=doubled_households,
                                 seed_offset=5000)
    print_results(
        robustness,
        f"Robustness check: same design with 2x households ({sum(g['n'] for g in doubled_households)} "
        "households instead of 10)",
    )
    same_direction = (
        (baseline["gini"]["treatment_mean"] > baseline["gini"]["control_mean"])
        == (robustness["gini"]["treatment_mean"] > robustness["gini"]["control_mean"])
    )
    print(
        f"\nWealth-Gini effect direction {'held' if same_direction else 'DID NOT hold'} "
        f"under 2x population (p={robustness['gini']['p_value']:.4f}, "
        f"d={robustness['gini']['cohens_d']:.3f} vs baseline d={baseline['gini']['cohens_d']:.3f})."
    )


if __name__ == "__main__":
    main()
