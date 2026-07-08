"""
experiments/optimize_cash_transfer.py
---------------------------------------
A small constrained-optimization example built on top of the existing
magnitude sweep (magnitude_sweep_cash_transfer.py). See README's
"Operations Research framing" section for the formal statement -- short
version:

    minimize     income_gini(m)
    subject to   wealth_gini(m) - wealth_gini(0) <= epsilon   (inequality tolerance)
                 outlay(m) <= Budget                           (fiscal constraint)
                 m >= 0

Why not just "minimize wealth_gini subject to budget"? Because in this
model *any* cash transfer > 0 raises wealth Gini (see README §0.5) --
that objective alone always has a trivial corner solution (m* = 0, spend
nothing), which isn't an interesting constrained-optimization example.
Framing it as "get as much income-equality benefit as an inequality
tolerance allows, without blowing the budget" creates a genuine trade-off
with a non-trivial solution, and matches how a real policymaker would
actually pose this (some wealth-Gini drift is tolerated in exchange for
lower income inequality, up to a point).

income_gini(m) and wealth_gini(m) are black-box, stochastic outputs of
the actual agent-based simulation (no closed form), so this uses grid
search rather than a gradient method -- standard practice for noisy
simulation-optimization problems where you can't differentiate the
objective.

Run:
    python -m experiments.optimize_cash_transfer --budget 300000 --epsilon 0.03
"""

from __future__ import annotations
import argparse

from policysim.simulation import DEFAULT_HOUSEHOLDS
from experiments.magnitude_sweep_cash_transfer import (
    run_at_magnitude_averaged, ROUNDS, TRIGGER_ROUND,
)

N_LOW = next(g["n"] for g in DEFAULT_HOUSEHOLDS if g["tier"] == "low")
N_MID = next(g["n"] for g in DEFAULT_HOUSEHOLDS if g["tier"] == "mid")
ACTIVE_ROUNDS = ROUNDS - TRIGGER_ROUND + 1  # transfer is active from TRIGGER_ROUND through ROUNDS


def outlay(magnitude: float) -> float:
    """Total government spend on the transfer across all active rounds.
    cash_transfer_low += m, cash_transfer_mid += 0.4m each active round
    (see policies.py POLICY_LIBRARY['cash_transfer']['apply'])."""
    per_round = magnitude * N_LOW + 0.4 * magnitude * N_MID
    return per_round * ACTIVE_ROUNDS


def grid_search(budget: float, epsilon: float, m_max: float = 4000.0, step: float = 100.0):
    baseline = run_at_magnitude_averaged(0.0)
    wealth_gini_floor = baseline["gini"]

    rows = []
    best = None
    m = 0.0
    while m <= m_max + 1e-9:
        spend = outlay(m)
        budget_ok = spend <= budget
        result = run_at_magnitude_averaged(m) if budget_ok else None
        tolerance_ok = (result is not None) and (result["gini"] - wealth_gini_floor <= epsilon)
        feasible = budget_ok and tolerance_ok
        rows.append((m, spend, feasible, result))
        if feasible and (best is None or result["gini_income"] < best[1]["gini_income"]):
            best = (m, result, spend)
        m += step
    return baseline, best, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=300000.0,
                         help="Total government outlay ceiling over the transfer's active rounds")
    parser.add_argument("--epsilon", type=float, default=0.03,
                         help="Max tolerated increase in wealth Gini above the m=0 baseline")
    parser.add_argument("--step", type=float, default=100.0, help="Grid step size for magnitude")
    args = parser.parse_args()

    print(
        f"Optimizing cash_transfer magnitude:\n"
        f"  minimize    income_gini(m)\n"
        f"  subject to  wealth_gini(m) - wealth_gini(0) <= {args.epsilon}\n"
        f"              outlay(m) <= Rs.{args.budget:,.0f}\n"
        f"({N_LOW} low-tier + {N_MID} mid-tier households, transfer active for "
        f"{ACTIVE_ROUNDS} rounds)\n"
    )

    baseline, best, rows = grid_search(args.budget, args.epsilon, step=args.step)
    print(f"Baseline (m=0): wealth Gini={baseline['gini']:.4f}, income Gini={baseline['gini_income']:.4f}\n")

    print(f"{'magnitude':>10}{'outlay':>14}{'feasible':>10}{'wealth gini':>14}{'income gini':>14}")
    for m, spend, feasible, result in rows:
        gini_str = f"{result['gini']:.4f}" if result else "-"
        income_str = f"{result['gini_income']:.4f}" if result else "-"
        print(f"{m:>10.0f}{spend:>14,.0f}{str(feasible):>10}{gini_str:>14}{income_str:>14}")

    if best is None:
        print(
            "\nNo feasible magnitude found -- either the budget is too tight or epsilon is "
            "too strict for any m > 0 (m=0 is always technically feasible under these "
            "constraints by construction; if even that's not showing as best, check inputs)."
        )
        return

    m_star, result_star, spend_star = best
    print(
        f"\nOptimal magnitude: m* = {m_star:.0f}\n"
        f"  income Gini = {result_star['gini_income']:.4f} "
        f"(down from {baseline['gini_income']:.4f} at m=0)\n"
        f"  wealth Gini = {result_star['gini']:.4f} "
        f"(+{result_star['gini'] - baseline['gini']:.4f} vs. m=0, within epsilon={args.epsilon})\n"
        f"  outlay = Rs.{spend_star:,.0f} (<= budget Rs.{args.budget:,.0f})"
    )


if __name__ == "__main__":
    main()
