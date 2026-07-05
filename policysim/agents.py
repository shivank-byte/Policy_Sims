"""
agents.py
---------
Defines the three agent types that populate the PolicySim economy:

  - Household  : consumes, saves, works, receives income & transfers
  - Firm       : sets prices & wages, hires/fires, produces goods
  - Government : holds the current policy state (taxes, subsidies, wage floor, transfers)

Each agent exposes a `state()` dict (fed to the reasoning brain) and an
`apply_decision()` method (applies whatever the brain decided back onto the
agent). This keeps "thinking" (brain.py) cleanly separated from "being"
(agents.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import itertools

_id_counter = itertools.count(1)


@dataclass
class Household:
    tier: str                      # "low", "mid", "high"
    base_income: float             # baseline wage/salary before policy effects
    savings: float = 0.0
    consumption_propensity: float = 0.7   # baseline marginal propensity to consume
    employed: bool = True
    id: int = field(default_factory=lambda: next(_id_counter))

    # last-round outcomes, used as context for next round's reasoning
    last_spend: float = 0.0
    last_income: float = 0.0
    last_reasoning: str = ""
    memory: list = field(default_factory=list)  # short rolling memory of past rounds

    def disposable_income(self, government: "Government") -> float:
        """Income after tax, wage-floor effects, plus any cash transfer."""
        income = self.base_income if self.employed else self.base_income * 0.25  # unemployment cushion
        # progressive-ish tax: only 'high' tier pays the luxury/income tax bump
        tax_rate = government.tax_rate
        if self.tier == "high":
            tax_rate += government.luxury_tax_surcharge
        income *= (1 - tax_rate)
        if self.tier == "low":
            income += government.cash_transfer_low
        elif self.tier == "mid":
            income += government.cash_transfer_mid
        return max(income, 0.0)

    def remember(self, round_num: int, note: str):
        self.memory.append(f"R{round_num}: {note}")
        if len(self.memory) > 4:
            self.memory.pop(0)

    def state(self, government: "Government", price_level: float) -> dict:
        return {
            "role": "household",
            "id": self.id,
            "tier": self.tier,
            "disposable_income": round(self.disposable_income(government), 2),
            "savings": round(self.savings, 2),
            "employed": self.employed,
            "price_level": round(price_level, 3),
            "consumption_propensity": self.consumption_propensity,
            "active_policies": government.active_policy_labels(),
            "memory": self.memory,
        }

    def apply_decision(self, decision: dict, government: "Government", round_num: int):
        income = self.disposable_income(government)
        spend_fraction = float(decision.get("spend_fraction", self.consumption_propensity))
        spend_fraction = min(max(spend_fraction, 0.0), 1.0)
        spend = income * spend_fraction
        save = income - spend
        self.savings += save
        self.last_spend = spend
        self.last_income = income
        self.last_reasoning = decision.get("reasoning", "")
        self.remember(round_num, f"spent {spend:.0f}/{income:.0f} income")


@dataclass
class Firm:
    name: str
    kind: str                 # "small_shop" or "large_firm"
    price: float
    wage: float
    unit_cost: float
    employees: int
    max_employees: int
    capacity_per_employee: float = 20.0
    inventory: float = 100.0
    id: int = field(default_factory=lambda: next(_id_counter))

    last_revenue: float = 0.0
    last_demand: float = 0.0
    last_reasoning: str = ""
    memory: list = field(default_factory=list)
    # Tracks the effective unit cost (post-subsidy) as of the last round, so
    # brain.py can pass through a COST CHANGE exactly once (the round it
    # happens) rather than re-applying a bump every round the policy stays
    # active. None on round 1 means "no shift yet, nothing to pass through".
    last_effective_cost: Optional[float] = None

    def effective_wage_floor(self, government: "Government") -> float:
        return max(self.wage, government.minimum_wage)

    def remember(self, round_num: int, note: str):
        self.memory.append(f"R{round_num}: {note}")
        if len(self.memory) > 4:
            self.memory.pop(0)

    def state(self, government: "Government") -> dict:
        return {
            "role": "firm",
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "price": round(self.price, 2),
            "wage": round(self.wage, 2),
            "min_wage": government.minimum_wage,
            "unit_cost": round(self.unit_cost, 2),
            "employees": self.employees,
            "max_employees": self.max_employees,
            "last_demand": round(self.last_demand, 2),
            "last_revenue": round(self.last_revenue, 2),
            "capacity_per_employee": self.capacity_per_employee,
            "subsidy_rate": government.firm_subsidy_rate,
            "last_effective_cost": self.last_effective_cost,
            "active_policies": government.active_policy_labels(),
            "memory": self.memory,
        }

    def apply_decision(self, decision: dict, government: "Government", round_num: int):
        new_price = float(decision.get("price", self.price))
        new_wage = float(decision.get("wage", self.wage))
        new_wage = max(new_wage, government.minimum_wage)  # cannot legally undercut wage floor
        employees_delta = int(decision.get("employees_delta", 0))

        self.price = max(new_price, self.unit_cost * 0.5)  # floor so price can't go absurd/negative
        self.wage = new_wage
        self.employees = min(max(self.employees + employees_delta, 0), self.max_employees)
        self.last_reasoning = decision.get("reasoning", "")
        self.remember(round_num, f"price={self.price:.2f} wage={self.wage:.2f} staff={self.employees}")
        # record this round's effective cost so next round can detect any
        # NEW change (subsidy cut, subsidy restored, etc.) and pass it
        # through exactly once, instead of re-adding a bump every round.
        self.last_effective_cost = self.unit_cost * (1 - government.firm_subsidy_rate)


@dataclass
class Government:
    tax_rate: float = 0.15
    luxury_tax_surcharge: float = 0.0
    minimum_wage: float = 240.0
    cash_transfer_low: float = 0.0
    cash_transfer_mid: float = 0.0
    firm_subsidy_rate: float = 0.10   # baseline subsidy that lowers firm unit costs
    active_policies: dict = field(default_factory=dict)  # policy_id -> {"magnitude":..., "round_applied":...}

    def active_policy_labels(self) -> list:
        return list(self.active_policies.keys())

    def state(self) -> dict:
        return {
            "role": "government",
            "tax_rate": self.tax_rate,
            "luxury_tax_surcharge": self.luxury_tax_surcharge,
            "minimum_wage": self.minimum_wage,
            "cash_transfer_low": self.cash_transfer_low,
            "cash_transfer_mid": self.cash_transfer_mid,
            "firm_subsidy_rate": self.firm_subsidy_rate,
            "active_policies": self.active_policies,
        }
