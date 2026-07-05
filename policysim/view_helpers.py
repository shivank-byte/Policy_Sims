"""
view_helpers.py
----------------
Turns Simulation/Household/Firm/Government objects into plain dicts.
Used by both server.py (FastAPI, optional standalone 3D server) and
streamlit_app.py (the single-deployment Streamlit Cloud entrypoint) so
there's one source of truth for "what does the simulation state look
like as JSON" instead of two copies drifting apart.
"""

from .policies import POLICY_LIBRARY


def household_payload(hh) -> dict:
    return {
        "id": hh.id,
        "tier": hh.tier,
        "employed": hh.employed,
        "base_income": hh.base_income,
        "savings": round(hh.savings, 2),
        "last_spend": round(hh.last_spend, 2),
        "last_income": round(hh.last_income, 2),
        "last_reasoning": hh.last_reasoning,
    }


def firm_payload(firm) -> dict:
    return {
        "id": firm.id,
        "name": firm.name,
        "kind": firm.kind,
        "price": round(firm.price, 2),
        "wage": round(firm.wage, 2),
        "employees": firm.employees,
        "max_employees": firm.max_employees,
        "last_demand": round(firm.last_demand, 2),
        "last_revenue": round(firm.last_revenue, 2),
        "capacity_per_employee": firm.capacity_per_employee,
        "last_reasoning": firm.last_reasoning,
    }


def government_payload(gov) -> dict:
    return {
        "tax_rate": gov.tax_rate,
        "luxury_tax_surcharge": gov.luxury_tax_surcharge,
        "minimum_wage": gov.minimum_wage,
        "cash_transfer_low": gov.cash_transfer_low,
        "cash_transfer_mid": gov.cash_transfer_mid,
        "firm_subsidy_rate": gov.firm_subsidy_rate,
        "active_policies": gov.active_policies,
    }


def policy_library_payload() -> dict:
    return {
        k: {"label": v["label"], "description": v["description"], "color": v["card_color"]}
        for k, v in POLICY_LIBRARY.items()
    }


def snapshot(sim) -> dict:
    latest = sim.history[-1] if sim.history else None
    return {
        "round": sim.round_num,
        "households": [household_payload(h) for h in sim.households],
        "firms": [firm_payload(f) for f in sim.firms],
        "government": government_payload(sim.government),
        "latest_stats": latest,
        "policy_library": policy_library_payload(),
    }
