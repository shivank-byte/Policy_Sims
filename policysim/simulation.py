"""
simulation.py
-------------
The round-based engine. Each call to `run_round()`:

  1. Firms observe last round's demand & reason about price/wage/staffing.
  2. Households observe prices & their own situation, reason about spend vs save.
  3. Market clears: household spending is allocated across firms (weighted by
     inverse price -- cheaper firms capture more demand), revenue/employment
     outcomes feed back into each firm's `last_demand`.
  4. Aggregate statistics are computed and appended to history.

The engine is agent-count agnostic (you can add more households/firms) and
completely decoupled from *how* agents reason -- that's all in brain.py.
"""

from __future__ import annotations
import random
import pandas as pd

from .agents import Household, Firm, Government
from .brain import AgentBrain
from .stats_engine import gini
from .policies import trigger_policy


# Consumption propensities fall from low -> high tier, mirroring the
# well-documented drop in propensity to consume as income/wealth rises
# (PSID-based estimates: MPC ~0.15 lowest wealth quintile vs ~0.06 highest,
# Fisher/Johnson/Smeeding/Thompson; Penn Wharton Budget Model recession-
# calibrated MPC by income quintile: 0.55 -> 0.12). See brain.py for the
# full citation and the caveat that these are average- not marginal-
# propensity levels tuned for a repeated-round heuristic, not a fitted
# model of any single dataset.
DEFAULT_HOUSEHOLDS = [
    dict(tier="low", base_income=24000, consumption_propensity=0.85, n=12),
    dict(tier="mid", base_income=50000, consumption_propensity=0.65, n=9),
    dict(tier="high", base_income=120000, consumption_propensity=0.45, n=3),
]
# 24 households in a 50% / 37.5% / 12.5% low/mid/high split -- a rough
# income-pyramid shape, rather than the earlier flat 4/4/2 split.

DEFAULT_FIRMS = [
    dict(name="Corner Shop", kind="small_shop", price=200.0, wage=220.0,
         unit_cost=120.0, employees=3, max_employees=8, capacity_per_employee=300.0),
    dict(name="Metro Industries", kind="large_firm", price=500.0, wage=280.0,
         unit_cost=300.0, employees=6, max_employees=18, capacity_per_employee=20.0),
    dict(name="Farm Fresh Co-op", kind="agriculture", price=80.0, wage=180.0,
         unit_cost=50.0, employees=6, max_employees=12, capacity_per_employee=350.0),
    dict(name="TechServe Solutions", kind="services", price=900.0, wage=450.0,
         unit_cost=300.0, employees=3, max_employees=8, capacity_per_employee=15.0),
]
# Four sectors instead of two, for more realistic diversity: a small
# retail shop, a large manufacturer, a labor-intensive low-margin
# agriculture co-op (cheap staple goods, high volume per worker), and a
# high-wage/low-headcount services firm. Total starting jobs =
# 3+6+6+3 = 18 vs. 24 households -> 25% unemployment at round 0. This
# also finally resolves README §6.3 bug #3 (unemployment pinned at 0% by
# default) for the *shipped* default, not just the separate
# experiments/unemployment_slack.py scenario -- the live demo and the
# statistical experiments now use the same, more realistic labor market.


class Simulation:
    def __init__(self, brain: AgentBrain = None, seed: int = None,
                 household_spec: list = None, firm_spec: list = None):
        if seed is not None:
            random.seed(seed)

        self.brain = brain or AgentBrain(backend="heuristic")
        self.government = Government()
        self.households = self._build_households(household_spec or DEFAULT_HOUSEHOLDS)
        self.firms = self._build_firms(firm_spec or DEFAULT_FIRMS)

        self.round_num = 0
        self.history: list[dict] = []
        self.thought_feed: list[dict] = []   # rolling log of agent reasoning, newest last
        self.event_log: list[dict] = []       # policy trigger events

    @staticmethod
    def _build_households(spec):
        out = []
        for group in spec:
            for _ in range(group["n"]):
                out.append(Household(
                    tier=group["tier"],
                    base_income=group["base_income"],
                    consumption_propensity=group["consumption_propensity"],
                    savings=group["base_income"] * random.uniform(0.5, 2.0),
                ))
        return out

    @staticmethod
    def _build_firms(spec):
        return [Firm(**f) for f in spec]

    # ------------------------------------------------------------------ #
    # Policy triggers (called by trigger_system.py or dashboard buttons)
    # ------------------------------------------------------------------ #
    def apply_policy_event(self, policy_id: str, magnitude: float = None):
        msg = trigger_policy(self.government, policy_id, magnitude, round_num=self.round_num)
        self.event_log.append({"round": self.round_num, "policy_id": policy_id, "message": msg})
        return msg

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run_round(self) -> dict:
        self.round_num += 1

        # ---- 1. Firms reason & act -------------------------------------------------
        for firm in self.firms:
            state = firm.state(self.government)
            decision = self.brain.decide(state)
            firm.apply_decision(decision, self.government, self.round_num)
            self.thought_feed.append({
                "round": self.round_num, "agent": f"Firm:{firm.name}",
                "thought": firm.last_reasoning,
            })

        price_level = self._price_level()

        # ---- 2. Households reason & act --------------------------------------------
        total_spending = 0.0
        for hh in self.households:
            state = hh.state(self.government, price_level)
            decision = self.brain.decide(state)
            hh.apply_decision(decision, self.government, self.round_num)
            total_spending += hh.last_spend
            self.thought_feed.append({
                "round": self.round_num, "agent": f"Household:{hh.tier}#{hh.id}",
                "thought": hh.last_reasoning,
            })

        # trim thought feed so it doesn't grow unbounded across a long demo
        if len(self.thought_feed) > 400:
            self.thought_feed = self.thought_feed[-400:]

        # ---- 3. Market clearing: allocate spending across firms by inverse price ---
        weights = [1 / max(f.price, 0.01) for f in self.firms]
        wsum = sum(weights) or 1
        for firm, w in zip(self.firms, weights):
            demand_share = total_spending * (w / wsum)
            firm.last_demand = demand_share / max(firm.price, 0.01)  # units demanded
            firm.last_revenue = demand_share
            firm.capacity = max(firm.employees, 1) * firm.capacity_per_employee

        # ---- 4. Employment & aggregate stats ---------------------------------------
        total_jobs = sum(f.employees for f in self.firms)
        total_slots = sum(f.max_employees for f in self.firms)
        employed_households = min(len(self.households), total_jobs)
        for i, hh in enumerate(self.households):
            hh.employed = i < employed_households
        unemployment_rate = 1 - (employed_households / len(self.households)) if self.households else 0.0

        incomes = [hh.disposable_income(self.government) for hh in self.households]
        wealth = [hh.savings for hh in self.households]

        round_stats = {
            "round": self.round_num,
            "price_level": price_level,
            "total_spending": total_spending,
            "gini": gini(wealth),
            "gini_income": gini(incomes),
            "unemployment_rate": unemployment_rate,
            "avg_wage": sum(f.wage for f in self.firms) / len(self.firms),
            "spend_low": sum(hh.last_spend for hh in self.households if hh.tier == "low"),
            "spend_mid": sum(hh.last_spend for hh in self.households if hh.tier == "mid"),
            "spend_high": sum(hh.last_spend for hh in self.households if hh.tier == "high"),
            "active_policies": ",".join(self.government.active_policy_labels()) or "none",
        }
        self.history.append(round_stats)
        return round_stats

    def run_n_rounds(self, n: int) -> list[dict]:
        return [self.run_round() for _ in range(n)]

    def _price_level(self) -> float:
        """Weighted average price, normalized so round-1 price level = 1.0."""
        avg = sum(f.price for f in self.firms) / len(self.firms)
        if not hasattr(self, "_base_price"):
            self._base_price = avg
        return avg / self._base_price if self._base_price else 1.0

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def thought_feed_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.thought_feed)


def run_multiple(num_runs: int, rounds_per_run: int, backend: str = "heuristic",
                  policy_event: tuple = None, policy_round: int = 5) -> pd.DataFrame:
    """
    Run the simulation multiple independent times (different random seeds) to
    check how *consistent* outcomes are for the same policy shock -- this is
    what makes the project a genuine mini research tool rather than a single
    anecdotal run.

    policy_event: (policy_id, magnitude) applied at `policy_round` in every run.
    Returns a long-format DataFrame with a `run_id` column.
    """
    frames = []
    for run_id in range(num_runs):
        sim = Simulation(brain=AgentBrain(backend=backend), seed=1000 + run_id)
        for r in range(1, rounds_per_run + 1):
            if policy_event and r == policy_round:
                sim.apply_policy_event(*policy_event)
            sim.run_round()
        df = sim.history_df()
        df["run_id"] = run_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
