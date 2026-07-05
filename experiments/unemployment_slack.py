"""
experiments/unemployment_slack.py
----------------------------------
The default firm spec (employees=3 + 8 = 11 jobs) offers more jobs than
there are households (10), so unemployment is pinned at exactly 0% in
every run regardless of policy -- a genuine model-exposed limitation
documented in README §0.5, not a bug.

This experiment re-runs the same cash_transfer control-vs-treatment
design with fewer starting jobs (employees=2 + 5 = 7 jobs vs. 10
households), so there is real slack in the labor market, and reports
whether a demand-side shock (cash_transfer) measurably closes any of
that slack.

Run:
    python -m experiments.unemployment_slack
"""

from __future__ import annotations

from policysim.simulation import DEFAULT_FIRMS
from experiments.cash_transfer_effect import run_experiment, print_results, N_RUNS, N_ROUNDS

# Same two firms as the default spec, with fewer starting jobs (so the
# labor market isn't already saturated before any policy runs) AND lower
# per-employee capacity (so utilization can actually cross the brain's
# hiring threshold within a 20-round run -- with the *default* capacity
# figures, demand under this reduced headcount never gets firms above the
# 1.15-utilization hiring trigger in brain.py, so unemployment sits flat
# regardless of policy; that's a second, more subtle version of the same
# "the labor margin never binds" issue documented in README §0.5).
SLACK_FIRMS = [
    dict(DEFAULT_FIRMS[0], employees=2, capacity_per_employee=200.0),   # Corner Shop: 3 -> 2 jobs
    dict(DEFAULT_FIRMS[1], employees=5, capacity_per_employee=18.0),    # Metro Industries: 8 -> 5 jobs
]


def main():
    print(f"Default firm spec jobs: {sum(f['employees'] for f in DEFAULT_FIRMS)} "
          f"vs. 10 households -> unemployment always 0% (no slack).")
    print(f"Slack firm spec jobs:   {sum(f['employees'] for f in SLACK_FIRMS)} "
          f"vs. 10 households -> real slack in the labor market.\n")

    results = run_experiment(n_runs=N_RUNS, n_rounds=N_ROUNDS, firm_spec=SLACK_FIRMS,
                              seed_offset=9000)
    print_results(
        results,
        f"cash_transfer effect after {N_ROUNDS} rounds, n={N_RUNS} runs/condition "
        "(labor-market slack: 7 jobs / 10 households)",
    )

    u = results["unemployment_rate"]
    print(
        f"\nUnemployment: control mean={u['control_mean']:.4f}, "
        f"treatment mean={u['treatment_mean']:.4f}, p={u['p_value']:.4f}. "
        + ("Cash transfer measurably moves unemployment once there's slack "
           "for a demand shock to close -- confirming §0.5's diagnosis that "
           "0% unemployment in the default run was a labor-supply ceiling, "
           "not a broken policy effect."
           if u["p_value"] == u["p_value"] and u["p_value"] < 0.05 else  # not NaN and significant
           "Even with slack, the effect is not statistically significant at "
           "this population size/round count -- worth a larger sweep before "
           "drawing a firm conclusion.")
    )


if __name__ == "__main__":
    main()
