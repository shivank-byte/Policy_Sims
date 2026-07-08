"""
experiments/unemployment_slack.py
----------------------------------
NOTE: this experiment's original premise is now historical. It used to
demonstrate that the *default* firm spec (11 jobs vs. 10 households) left
no labor-market slack, so unemployment was pinned at exactly 0% regardless
of policy -- a real bug, documented in README §6.3 (bug #3). That's fixed
now: DEFAULT_FIRMS/DEFAULT_HOUSEHOLDS (see simulation.py) were expanded to
24 households and 4 firms offering only 18 starting jobs, so the *default*
population already has ~25-33% unemployment and genuine labor-market
slack, without needing a special override.

This experiment is kept, repurposed to test the opposite extreme: does
the wealth-Gini finding (README §0.5) still hold when the labor market is
fully saturated (jobs >= households, unemployment structurally pinned at
0%, same as the old default used to be)? If yes, that confirms the
wealth-Gini mechanism doesn't depend on employment risk being present --
if the effect vanished, it would mean unemployment risk was secretly
driving the whole result.

Run:
    python -m experiments.unemployment_slack
"""

from __future__ import annotations

from policysim.simulation import DEFAULT_FIRMS, DEFAULT_HOUSEHOLDS
from experiments.cash_transfer_effect import run_experiment, print_results, N_RUNS, N_ROUNDS

N_HOUSEHOLDS = sum(g["n"] for g in DEFAULT_HOUSEHOLDS)
DEFAULT_JOBS = sum(f["employees"] for f in DEFAULT_FIRMS)

# Same four firms, headcount raised enough that total jobs comfortably
# exceed the household count -- i.e. deliberately recreating the
# "no slack" condition the old default used to have, but now as a
# controlled comparison case rather than an accidental default.
FULL_EMPLOYMENT_FIRMS = [
    dict(DEFAULT_FIRMS[0], employees=6),   # Corner Shop: 3 -> 6
    dict(DEFAULT_FIRMS[1], employees=12),  # Metro Industries: 6 -> 12
    dict(DEFAULT_FIRMS[2], employees=8),   # Farm Fresh Co-op: 6 -> 8
    dict(DEFAULT_FIRMS[3], employees=4),   # TechServe Solutions: 3 -> 4
]
FULL_EMPLOYMENT_JOBS = sum(f["employees"] for f in FULL_EMPLOYMENT_FIRMS)


def main():
    print(f"Default firm spec jobs: {DEFAULT_JOBS} vs. {N_HOUSEHOLDS} households "
          f"-> real slack ({(1 - DEFAULT_JOBS/N_HOUSEHOLDS)*100:.0f}% starting unemployment).")
    print(f"Full-employment firm spec jobs: {FULL_EMPLOYMENT_JOBS} vs. {N_HOUSEHOLDS} households "
          "-> no slack, unemployment structurally pinned at 0% (deliberately, for comparison).\n")

    results = run_experiment(n_runs=N_RUNS, n_rounds=N_ROUNDS, firm_spec=FULL_EMPLOYMENT_FIRMS,
                              seed_offset=9000)
    print_results(
        results,
        f"cash_transfer effect after {N_ROUNDS} rounds, n={N_RUNS} runs/condition "
        f"(full employment: {FULL_EMPLOYMENT_JOBS} jobs / {N_HOUSEHOLDS} households, no slack)",
    )

    g = results["gini"]
    if not g["degenerate"] and g["p_value"] < 0.05 and g["treatment_mean"] > g["control_mean"]:
        print(
            "\nWealth Gini still rises significantly even with zero unemployment risk "
            f"(p={g['p_value']:.4f}, Cohen's d={g['cohens_d']:.2f}) -- confirming the mechanism "
            "in README §0.5 (differing marginal propensity to consume by tier) drives this "
            "result, not employment risk. Compare against the default-population run in "
            "§0.5/experiments.cash_transfer_effect, which has real unemployment and shows a "
            "similar wealth-Gini effect."
        )
    else:
        print(
            "\nThe wealth-Gini effect did not replicate under full employment in this run -- "
            "worth investigating whether employment risk was doing more work in the original "
            "result than §0.5 assumed."
        )


if __name__ == "__main__":
    main()
