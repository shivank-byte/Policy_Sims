"""
policies.py
-----------
Defines the catalogue of policy shocks that can be triggered live (by color
card or QR code) and exactly how each one mutates the Government's state.

Each entry in POLICY_LIBRARY has:
  label        - human-readable name shown on the dashboard
  description  - one-line explanation for the audience
  card_color   - which color card triggers it (for the OpenCV color detector)
  default_magnitude - used if no magnitude is encoded (e.g. plain color card)
  apply(gov, magnitude) - mutates the Government dataclass in place
  revert(gov, magnitude) - undoes the effect (used when a policy is toggled off)
"""

from __future__ import annotations
from .agents import Government


def _subsidy_cut(gov: Government, magnitude: float):
    """Multiplicative cut: magnitude=0.20 means 'cut the subsidy rate by 20%',
    e.g. 0.10 -> 0.08. This lets 20%/30%/40% stay distinguishable at any
    starting rate, unlike an absolute percentage-point subtraction which
    saturates to 0 once magnitude >= the baseline rate."""
    gov.firm_subsidy_rate = max(0.0, gov.firm_subsidy_rate * (1 - magnitude))


def _subsidy_cut_undo(gov: Government, magnitude: float):
    denom = max(1e-6, 1 - magnitude)
    gov.firm_subsidy_rate = min(1.0, gov.firm_subsidy_rate / denom)


def _min_wage_increase(gov: Government, magnitude: float):
    gov.minimum_wage *= (1 + magnitude)


def _min_wage_increase_undo(gov: Government, magnitude: float):
    gov.minimum_wage /= (1 + magnitude)


def _luxury_tax(gov: Government, magnitude: float):
    gov.luxury_tax_surcharge += magnitude


def _luxury_tax_undo(gov: Government, magnitude: float):
    gov.luxury_tax_surcharge = max(0.0, gov.luxury_tax_surcharge - magnitude)


def _cash_transfer(gov: Government, magnitude: float):
    gov.cash_transfer_low += magnitude
    gov.cash_transfer_mid += magnitude * 0.4


def _cash_transfer_undo(gov: Government, magnitude: float):
    gov.cash_transfer_low = max(0.0, gov.cash_transfer_low - magnitude)
    gov.cash_transfer_mid = max(0.0, gov.cash_transfer_mid - magnitude * 0.4)


POLICY_LIBRARY = {
    "subsidy_cut": {
        "label": "Fuel/Input Subsidy Cut",
        "description": "Cuts the subsidy that lowers firms' input costs.",
        "card_color": "red",
        "default_magnitude": 0.20,
        "apply": _subsidy_cut,
        "revert": _subsidy_cut_undo,
    },
    "minimum_wage_increase": {
        "label": "Minimum Wage Increase",
        "description": "Raises the legal wage floor firms must pay.",
        "card_color": "yellow",
        "default_magnitude": 0.15,
        "apply": _min_wage_increase,
        "revert": _min_wage_increase_undo,
    },
    "luxury_tax": {
        "label": "Luxury Tax Introduced",
        "description": "Adds a surcharge on high-income households' tax rate.",
        "card_color": "blue",
        "default_magnitude": 0.10,
        "apply": _luxury_tax,
        "revert": _luxury_tax_undo,
    },
    "cash_transfer": {
        "label": "Cash Transfer to Low-Income Households",
        "description": "Direct payment boosting low/mid income each round.",
        "card_color": "green",
        "default_magnitude": 1000.0,
        "apply": _cash_transfer,
        "revert": _cash_transfer_undo,
    },
}

COLOR_TO_POLICY = {v["card_color"]: k for k, v in POLICY_LIBRARY.items()}


def trigger_policy(gov: Government, policy_id: str, magnitude: float = None, round_num: int = 0) -> str:
    """Apply a policy shock to the government, toggling it off if already active
    (so showing the same card twice reverts the policy — nice for live demos)."""
    if policy_id not in POLICY_LIBRARY:
        raise ValueError(f"Unknown policy '{policy_id}'. Known: {list(POLICY_LIBRARY)}")

    spec = POLICY_LIBRARY[policy_id]
    mag = magnitude if magnitude is not None else spec["default_magnitude"]

    if policy_id in gov.active_policies:
        spec["revert"](gov, gov.active_policies[policy_id]["magnitude"])
        del gov.active_policies[policy_id]
        return f"REVERTED: {spec['label']}"
    else:
        spec["apply"](gov, mag)
        gov.active_policies[policy_id] = {"magnitude": mag, "round_applied": round_num}
        return f"TRIGGERED: {spec['label']} (magnitude={mag})"


def parse_qr_payload(payload: str):
    """Parse a QR string like 'subsidy_cut:20%' or 'cash_transfer:50' into (policy_id, magnitude)."""
    payload = payload.strip()
    if ":" not in payload:
        return payload, None
    policy_id, raw_mag = payload.split(":", 1)
    raw_mag = raw_mag.strip().rstrip("%")
    try:
        val = float(raw_mag)
        magnitude = val / 100 if "%" in payload else val
    except ValueError:
        magnitude = None
    return policy_id.strip(), magnitude
