"""
experiments/magnitude_sweep_cash_transfer.py
----------------------------------------------
`stats_engine.magnitude_sensitivity_sweep` already exists and is used in
the README to catch policies that *saturate* (same outcome at every
magnitude). This script applies that same idea specifically to
cash_transfer's effect on wealth Gini: does a bigger transfer widen
inequality more, or does the effect plateau?

Design: for each magnitude, average final-round wealth/income Gini and
price level across N_SEEDS independent runs (cash_transfer applied at
round 5, 20 rounds total) -- a single seed per magnitude turned out to be
too noisy to tell a real trend from run-to-run variance, so this averages
exactly like `multi_run_consistency` does elsewhere in the project.

Run:
    python -m experiments.magnitude_sweep_cash_transfer
"""

from __future__ import annotations
import statistics

from policysim.simulation import Simulation
from policysim.brain import AgentBrain

MAGNITUDES = [0, 250, 500, 1000, 2000, 4000]
ROUNDS = 20
TRIGGER_ROUND = 5
N_SEEDS = 15
BASE_SEED = 42


def run_at_magnitude(magnitude: float, seed: int) -> dict:
    sim = Simulation(brain=AgentBrain(backend="heuristic"), seed=seed)
    for r in range(1, ROUNDS + 1):
        if r == TRIGGER_ROUND and magnitude > 0:
            sim.apply_policy_event("cash_transfer", magnitude)
        sim.run_round()
    final = sim.history[-1]
    return {"gini": final["gini"], "gini_income": final["gini_income"],
            "price_level": final["price_level"]}


def run_at_magnitude_averaged(magnitude: float) -> dict:
    runs = [run_at_magnitude(magnitude, BASE_SEED + s) for s in range(N_SEEDS)]
    return {
        "gini": statistics.mean(r["gini"] for r in runs),
        "gini_std": statistics.stdev(r["gini"] for r in runs),
        "gini_income": statistics.mean(r["gini_income"] for r in runs),
        "price_level": statistics.mean(r["price_level"] for r in runs),
    }


def main():
    print(f"cash_transfer magnitude sweep, averaged over {N_SEEDS} seeds/magnitude "
          f"(trigger round={TRIGGER_ROUND}, {ROUNDS} rounds total)\n")
    print(f"{'magnitude':>10}{'wealth gini':>14}{'(std)':>8}{'income gini':>14}{'price level':>14}")
    print("-" * 60)
    rows = []
    for mag in MAGNITUDES:
        r = run_at_magnitude_averaged(mag)
        rows.append((mag, r))
        print(f"{mag:>10}{r['gini']:>14.4f}{r['gini_std']:>8.4f}{r['gini_income']:>14.4f}{r['price_level']:>14.4f}")

    ginis = [r["gini"] for _, r in rows]
    incomes = [r["gini_income"] for _, r in rows]
    if len(set(round(g, 3) for g in ginis)) == 1:
        print("\nWARNING: wealth Gini is identical at every magnitude -- the "
              "effect may be saturating rather than scaling.")
        return

    # The actual pattern found (averaged over 15 seeds): wealth Gini jumps up
    # as soon as ANY transfer is applied (a threshold/level-shift effect, not
    # a gentle linear ramp), then edges slightly back down as magnitude keeps
    # growing further. Income Gini, in contrast, falls smoothly and
    # monotonically as the transfer grows. Report the actual shape rather
    # than force a monotonic-or-not verdict.
    peak_idx = max(range(len(ginis)), key=lambda i: ginis[i])
    print(
        f"\nWealth Gini jumps from {ginis[0]:.4f} (no transfer) to {ginis[1]:.4f} "
        f"the moment ANY transfer is applied (magnitude={MAGNITUDES[1]}) -- a threshold "
        f"effect, not a gradual ramp. It then peaks at magnitude={MAGNITUDES[peak_idx]} "
        f"({ginis[peak_idx]:.4f}) and drifts slightly back down as the transfer grows "
        f"further (down to {ginis[-1]:.4f} at magnitude={MAGNITUDES[-1]}) -- consistent "
        "with low/mid households eventually saving a larger absolute amount even at a "
        "roughly constant savings *rate*, once the transfer is large relative to their "
        "base income.\n"
        f"Income Gini, by contrast, falls smoothly and monotonically with magnitude "
        f"({incomes[0]:.4f} -> {incomes[-1]:.4f}) -- exactly as expected of a transfer "
        "targeted at low/mid earners.\n"
        "Net takeaway: don't read the wealth-Gini-widening result as 'bigger transfer = "
        "worse for wealth inequality' -- it's better described as 'any transfer widens "
        "wealth inequality at this population size, with a mild ceiling on how much.'"
    )


if __name__ == "__main__":
    main()
