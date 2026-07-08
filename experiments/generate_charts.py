"""
experiments/generate_charts.py
--------------------------------
Generates round-by-round time-series charts (not just before/after
snapshots) for the cash_transfer control-vs-treatment comparison, so the
mechanism is visible over time rather than asserted from two numbers.

Averages `N_RUNS` independent runs per condition per round (mean +/- 1
std band) and saves PNGs to assets/charts/.

Run:
    python -m experiments.generate_charts
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policysim.simulation import Simulation
from policysim.brain import AgentBrain

N_RUNS = 40
N_ROUNDS = 20
OUT_DIR = "assets/charts"


def collect(apply_transfer: bool) -> dict:
    """Returns {metric: array of shape (N_RUNS, N_ROUNDS)}"""
    metrics = ["gini", "gini_income", "price_level", "unemployment_rate"]
    data = {m: np.zeros((N_RUNS, N_ROUNDS)) for m in metrics}
    for run in range(N_RUNS):
        sim = Simulation(brain=AgentBrain(backend="heuristic"), seed=run)
        if apply_transfer:
            sim.apply_policy_event("cash_transfer")
        for r in range(N_ROUNDS):
            stats = sim.run_round()
            for m in metrics:
                data[m][run, r] = stats[m]
    return data


def plot_metric(control: np.ndarray, treatment: np.ndarray, title: str, ylabel: str, filename: str):
    rounds = np.arange(1, N_ROUNDS + 1)
    c_mean, c_std = control.mean(axis=0), control.std(axis=0)
    t_mean, t_std = treatment.mean(axis=0), treatment.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rounds, c_mean, label="Control (no transfer)", color="#4C72B0", linewidth=2)
    ax.fill_between(rounds, c_mean - c_std, c_mean + c_std, color="#4C72B0", alpha=0.15)
    ax.plot(rounds, t_mean, label="Treatment (cash transfer, round 1+)", color="#DD8452", linewidth=2)
    ax.fill_between(rounds, t_mean - t_std, t_mean + t_std, color="#DD8452", alpha=0.15)
    ax.axvline(1, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{filename}", dpi=150)
    plt.close(fig)
    print(f"saved {OUT_DIR}/{filename}")


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Running {N_RUNS} control + {N_RUNS} treatment simulations, "
          f"{N_ROUNDS} rounds each, to build round-by-round charts...")
    control = collect(apply_transfer=False)
    treatment = collect(apply_transfer=True)

    plot_metric(control["gini"], treatment["gini"],
                "Wealth (Savings) Gini Over Time", "Gini coefficient",
                "wealth_gini_over_time.png")
    plot_metric(control["gini_income"], treatment["gini_income"],
                "Income Gini Over Time", "Gini coefficient",
                "income_gini_over_time.png")
    plot_metric(control["price_level"], treatment["price_level"],
                "Price Level Over Time (round 1 = 1.0)", "Price level",
                "price_level_over_time.png")
    plot_metric(control["unemployment_rate"], treatment["unemployment_rate"],
                "Unemployment Rate Over Time (default population -- pinned at 0%)",
                "Unemployment rate", "unemployment_over_time.png")

    print("\nDone. These make the wealth-Gini-up / income-Gini-down divergence "
          "visible as a trend across all 20 rounds, not just a single before/after "
          "snapshot.")


if __name__ == "__main__":
    main()
