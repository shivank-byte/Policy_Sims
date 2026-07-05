"""
brain.py
--------
The "reasoning" layer. Every agent, every round, calls
`AgentBrain.decide(state)` and gets back `(decision: dict, reasoning: str)`.

Three backends, tried in this order (configurable):
  1. "ollama"    -> local Llama 3.1 8B via http://localhost:11434 (free, offline, demo-safe)
  2. "groq"      -> Groq free-tier hosted Llama models (free, needs internet, very fast)
  3. "heuristic" -> pure-python bounded-rationality rules (zero dependencies, zero cost,
                    ALWAYS available — guarantees the demo never breaks on stage)

This means the simulation runs perfectly well with no LLM installed at all
(heuristic mode), but drops in *real* generative reasoning the moment Ollama
or Groq is reachable — no code changes required, just set the backend.
"""

from __future__ import annotations
import json
import random
import re
from typing import Optional

try:
    import requests
except ImportError:  # requests should be present, but degrade gracefully
    requests = None


SYSTEM_PROMPT = {
    "household": (
        "You are a household in a small simulated economy, income tier: {tier}. "
        "You must decide what fraction of your disposable income to spend this round "
        "(the rest is saved). Think briefly like a real person reacting to prices, "
        "your income, and any government policy in effect, then respond ONLY with JSON: "
        '{{"spend_fraction": <0..1 float>, "reasoning": "<one short sentence, plain language>"}}'
    ),
    "firm": (
        "You are the manager of a {kind} named {name} in a small simulated economy. "
        "Decide this round's price, wage offered to workers, and how many staff to "
        "hire (+) or lay off (-). React like a real manager to costs, demand, wage "
        "floors and subsidies. Respond ONLY with JSON: "
        '{{"price": <float>, "wage": <float>, "employees_delta": <int>, '
        '"reasoning": "<one short sentence, plain language>"}}'
    ),
}


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction from an LLM completion."""
    text = text.strip()
    text = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class AgentBrain:
    def __init__(self, backend: str = "heuristic", ollama_model: str = "llama3.1:8b",
                 ollama_url: str = "http://localhost:11434/api/generate",
                 groq_api_key: Optional[str] = None,
                 groq_model: str = "llama-3.1-8b-instant"):
        """
        backend: "heuristic" | "ollama" | "groq"
        Falls back to "heuristic" automatically if the chosen backend errors out,
        so a flaky wifi connection never kills a live demo.
        """
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.fallback_count = 0  # tracked so the dashboard can show backend health

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def decide(self, state: dict) -> dict:
        role = state["role"]
        prompt = self._build_prompt(state)

        decision = None
        if self.backend == "ollama":
            decision = self._call_ollama(prompt)
        elif self.backend == "groq":
            decision = self._call_groq(prompt)

        if decision is None:
            self.fallback_count += 1
            decision = self._heuristic(state)

        return decision

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_prompt(self, state: dict) -> str:
        role = state["role"]
        sys = SYSTEM_PROMPT[role].format(**state)
        context = {k: v for k, v in state.items() if k not in ("role",)}
        return f"{sys}\n\nCurrent situation (JSON):\n{json.dumps(context)}\n\nYour JSON response:"

    # ------------------------------------------------------------------ #
    # Backend: Ollama (local, offline)
    # ------------------------------------------------------------------ #
    def _call_ollama(self, prompt: str) -> Optional[dict]:
        if requests is None:
            return None
        try:
            resp = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=15,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            return _extract_json(text)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Backend: Groq (hosted, needs internet, very fast free tier)
    # ------------------------------------------------------------------ #
    def _call_groq(self, prompt: str) -> Optional[dict]:
        if requests is None or not self.groq_api_key:
            return None
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                json={
                    "model": self.groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 200,
                },
                timeout=15,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return _extract_json(text)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Backend: heuristic (zero-dependency bounded rationality)
    # ------------------------------------------------------------------ #
    def _heuristic(self, state: dict) -> dict:
        role = state["role"]
        if role == "household":
            return self._heuristic_household(state)
        return self._heuristic_firm(state)

    def _heuristic_household(self, state: dict) -> dict:
        """
        Coefficients below are hand-tuned but deliberately kept within
        published empirical ranges rather than picked arbitrarily:

        - Base consumption propensities (low/mid/high = 0.85/0.65/0.45,
          set in simulation.py) track the well-documented fall in average
          propensity to consume as income rises. Studies using PSID data
          find MPC roughly 0.15 for the lowest wealth quintile vs. 0.06
          for the highest in normal times, and the Penn Wharton Budget
          Model's recession-calibrated MPCs by income quintile run
          0.55 -> 0.12 top to bottom (Fisher, Johnson, Smeeding & Thompson
          2016/2020, PSID 1999-2013; Penn Wharton Budget Model 2021).
          This model's tiers are coarser (3 tiers, not 5 quintiles) and
          use average- rather than marginal-propensity levels, since
          households here re-decide total spend each round rather than
          spend out of a one-off shock, so the absolute levels sit higher
          -- but the *gradient* (low >> mid >> high) mirrors the same
          finding: poorer households have far less room to save.
        - The price-sensitivity swing (max 8 percentage points of spend
          fraction for a >=25% price rise) reflects that a mixed
          household basket dominated by necessities is price-inelastic
          (typical estimates cluster around -0.5 to -1.0, well short of
          the elastic >1 seen for pure luxury/discretionary goods)
          (Wikipedia: Price elasticity of demand; Saylor Principles of
          Economics 5.1).
        These are still simplifications of a single aggregate reasoning
        rule, not a solved household optimization problem -- treat them
        as a documented, literature-consistent heuristic, not a citation
        of a specific paper's point estimate.
        """
        base = state["consumption_propensity"]
        price_level = state["price_level"]
        income = state["disposable_income"]
        tier = state["tier"]
        policies = state["active_policies"]

        adj = 0.0
        reasons = []

        # price sensitivity: higher prices -> save a bit more (real-income squeeze)
        if price_level > 1.05:
            adj -= 0.08 * min((price_level - 1.0) * 4, 1.0)
            reasons.append(f"prices are up {((price_level-1)*100):.0f}%, so I'm cutting back a little")
        elif price_level < 0.95:
            adj += 0.05
            reasons.append("prices have dropped, so I can afford to spend a bit more")

        # unemployment fear
        if not state["employed"]:
            adj -= 0.25
            reasons.append("I lost my job so I'm spending only on essentials")

        # policy reactions
        if "cash_transfer" in policies:
            adj += 0.06 if tier in ("low", "mid") else 0.0
            reasons.append("the new cash transfer gives me a bit more room to spend")
        if "luxury_tax" in policies and tier == "high":
            adj -= 0.07
            reasons.append("the luxury tax makes me hold back on discretionary purchases")
        if "minimum_wage_increase" in policies and tier == "low":
            adj += 0.04
            reasons.append("the higher minimum wage means more take-home pay")
        if "subsidy_cut" in policies:
            adj -= 0.03
            reasons.append("with subsidies cut, everyday costs feel tighter")

        # low income households have less slack to save; high income more slack to cut
        # (same low->high gradient as the MPC-by-quintile literature cited above)
        if tier == "low":
            adj += 0.02
        elif tier == "high":
            adj -= 0.02

        noise = random.uniform(-0.02, 0.02)
        spend_fraction = min(max(base + adj + noise, 0.15), 0.98)

        reasoning = "; ".join(reasons) if reasons else f"steady income of {income:.0f}, spending as usual"
        return {"spend_fraction": round(spend_fraction, 3), "reasoning": reasoning}

    def _heuristic_firm(self, state: dict) -> dict:
        price = state["price"]
        wage = state["wage"]
        min_wage = state["min_wage"]
        unit_cost = state["unit_cost"]
        demand = state["last_demand"]
        employees = state["employees"]
        max_employees = state["max_employees"]
        subsidy_rate = state["subsidy_rate"]
        policies = state["active_policies"]

        reasons = []
        effective_cost = unit_cost * (1 - subsidy_rate)

        # demand-driven pricing: capacity proxy calibrated per firm
        cap_per_emp = state.get("capacity_per_employee", 20.0)
        capacity = max(employees, 1) * cap_per_emp
        utilization = demand / capacity if capacity else 1.0

        price_delta = 0.0
        if utilization > 1.1:
            price_delta = 0.06
            reasons.append("demand is outstripping what we can supply, so we're raising prices")
        elif utilization < 0.7:
            price_delta = -0.04
            reasons.append("demand is soft, so we're trimming prices to move inventory")

        new_wage = max(wage, min_wage)
        if min_wage > wage:
            reasons.append(f"minimum wage rose to {min_wage:.2f}, so we must raise pay")

        # cost pass-through: if wage floor rose, some cost gets passed into price
        if new_wage > wage:
            price_delta += (new_wage - wage) / max(unit_cost, 1) * 0.3

        # --- Real one-time cost pass-through (replaces old flat +0.05/round bump) ---
        # Compares THIS round's effective (post-subsidy) unit cost against last
        # round's, and passes through only the fraction of the CHANGE that's
        # new. A subsidy cut (or restoration) therefore produces one bounded
        # price shift the round it happens, then price stabilizes — instead of
        # compounding every single round the policy happens to still be active.
        COST_PASS_THROUGH_RATE = 0.6  # firms absorb 40% of a cost shock, pass through 60%
        last_effective_cost = state.get("last_effective_cost")
        if last_effective_cost and last_effective_cost > 0:
            cost_change_pct = (effective_cost - last_effective_cost) / last_effective_cost
            if abs(cost_change_pct) > 1e-6:
                price_delta += cost_change_pct * COST_PASS_THROUGH_RATE
                if cost_change_pct > 0:
                    reasons.append(
                        f"input costs rose {cost_change_pct*100:.1f}% (e.g. subsidy cut), "
                        f"passing through {cost_change_pct*COST_PASS_THROUGH_RATE*100:.1f}% to price"
                    )
                else:
                    reasons.append(
                        f"input costs fell {abs(cost_change_pct)*100:.1f}%, "
                        f"passing some of that saving on to price"
                    )
        if "luxury_tax" in policies and state["kind"] == "large_firm":
            reasons.append("luxury tax is softening demand for premium goods")

        price_delta = min(max(price_delta, -0.10), 0.10)  # clamp so no single round can spike/crash price >10%
        new_price = max(price * (1 + price_delta), effective_cost * 1.05)

        # hiring/firing based on utilization vs margin health
        employees_delta = 0
        margin = (new_price - effective_cost) / max(new_price, 0.01)
        if utilization > 1.15 and employees < max_employees:
            employees_delta = 1
            reasons.append("we're hiring to keep up with demand")
        elif (utilization < 0.6 or margin < 0.05) and employees > 0:
            employees_delta = -1
            reasons.append("weak demand/margins mean we have to cut staff")

        noise = random.uniform(-0.01, 0.01)
        new_price = round(max(new_price * (1 + noise), 0.5), 2)

        reasoning = "; ".join(reasons) if reasons else "holding price and staffing steady"
        return {
            "price": new_price,
            "wage": round(new_wage, 2),
            "employees_delta": employees_delta,
            "reasoning": reasoning,
        }
