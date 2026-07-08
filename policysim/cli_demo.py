"""
cli_demo.py
-----------
A terminal-only run of the full simulation -- no Streamlit, no webcam, no
LLM required. Useful for: quick sanity checks, CI, or presenting the
research/statistics side of the project without the live show.

Usage:
    python -m policysim.cli_demo --rounds 30 --policy cash_transfer --at 10 --backend heuristic
    python -m policysim.cli_demo --multi-run 8 --rounds 20 --policy subsidy_cut --at 8
"""

from __future__ import annotations
import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .simulation import Simulation, run_multiple
from .brain import AgentBrain
from .stats_engine import before_after_comparison, multi_run_consistency, price_elasticity_of_demand
from .policies import POLICY_LIBRARY


def single_run(args):
    sim = Simulation(brain=AgentBrain(backend=args.backend), seed=args.seed)
    for r in range(1, args.rounds + 1):
        if args.policy and r == args.at:
            msg = sim.apply_policy_event(args.policy, args.magnitude)
            print(f"\n>>> Round {r}: {msg}\n")
        stats = sim.run_round()
        print(f"R{stats['round']:>3} | price={stats['price_level']:.3f} "
              f"gini={stats['gini']:.3f} unemp={stats['unemployment_rate']*100:5.1f}% "
              f"spend={stats['total_spending']:9.0f} | {stats['active_policies']}")

    print("\n--- Sample agent reasoning (last round) ---")
    for t in sim.thought_feed[-5:]:
        print(f"  [{t['agent']}] {t['thought']}")

    if args.policy:
        df = sim.history_df()
        window = min(args.at - 1, 5)
        print("\n--- Before / After comparison (Welch's t-test + bootstrap 95% CI) ---")
        print(before_after_comparison(df, trigger_round=args.at, window=window).to_string(index=False))

        elas = price_elasticity_of_demand(df, trigger_round=args.at, window=window)
        print("\n--- Price-elasticity-of-demand sanity check ---")
        if elas["in_textbook_range"] is None:
            print("  Not enough data to estimate elasticity for this window.")
        else:
            lo, hi = elas["textbook_range"]
            verdict = "within" if elas["in_textbook_range"] else "OUTSIDE"
            print(f"  elasticity={elas['elasticity']}  price_change={elas['pct_price_change']:+.1f}%  "
                  f"qty_change={elas['pct_quantity_change']:+.1f}%  textbook_range=({lo},{hi})  -> {verdict}")

    os.makedirs(args.output_dir, exist_ok=True)
    df = sim.history_df()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(df["round"], df["price_level"]); axes[0, 0].set_title("Price Level")
    axes[0, 1].plot(df["round"], df["gini"], color="darkred"); axes[0, 1].set_title("Wealth Gini Coefficient")
    axes[1, 0].plot(df["round"], df["unemployment_rate"] * 100, color="orange"); axes[1, 0].set_title("Unemployment Rate (%)")
    axes[1, 1].plot(df["round"], df["spend_low"], label="low income")
    axes[1, 1].plot(df["round"], df["spend_mid"], label="mid income")
    axes[1, 1].plot(df["round"], df["spend_high"], label="high income")
    axes[1, 1].set_title("Spending by Income Tier"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Round"); ax.grid(alpha=0.3)
        if args.policy:
            ax.axvline(args.at, color="gray", linestyle="--", alpha=0.6)
    fig.tight_layout()
    chart_path = os.path.join(args.output_dir, "simulation_charts.png")
    fig.savefig(chart_path, dpi=130)
    csv_path = os.path.join(args.output_dir, "simulation_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved charts -> {chart_path}")
    print(f"Saved history -> {csv_path}")


def multi_run(args):
    policy_event = (args.policy, args.magnitude) if args.policy else None
    df = run_multiple(args.multi_run, args.rounds, backend=args.backend,
                       policy_event=policy_event, policy_round=args.at)
    print(f"\n--- Consistency across {args.multi_run} independent runs ---")
    print(multi_run_consistency(df).to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, g in df.groupby("run_id"):
        ax.plot(g["round"], g["price_level"], alpha=0.5, label=f"run {run_id}")
    if args.policy:
        ax.axvline(args.at, color="gray", linestyle="--")
    ax.set_title(f"Price level across {args.multi_run} runs ({args.policy or 'no policy'})")
    ax.set_xlabel("Round"); ax.grid(alpha=0.3)
    path = os.path.join(args.output_dir, "multi_run_price_level.png")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"Saved multi-run chart -> {path}")


def main():
    p = argparse.ArgumentParser(description="PolicySim CLI demo")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--policy", choices=list(POLICY_LIBRARY.keys()), default=None)
    p.add_argument("--at", type=int, default=8, help="round at which to trigger the policy")
    p.add_argument("--magnitude", type=float, default=None)
    p.add_argument("--backend", choices=["heuristic", "ollama", "groq"], default="heuristic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multi-run", type=int, default=0, help="if >0, run this many independent simulations")
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    if args.multi_run and args.multi_run > 0:
        multi_run(args)
    else:
        single_run(args)


if __name__ == "__main__":
    main()
