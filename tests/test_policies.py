"""
Regression tests for policies.py.

Two of these tests exist specifically to prevent the exact bugs found during
development from silently coming back:
  - test_subsidy_cut_is_multiplicative_not_saturating
  - test_subsidy_cut_distinguishes_magnitudes
"""
import pytest
from policysim.agents import Government
from policysim.policies import (
    POLICY_LIBRARY, trigger_policy, parse_qr_payload, COLOR_TO_POLICY,
)


def test_subsidy_cut_is_multiplicative_not_saturating():
    """A subsidy cut should scale the EXISTING rate down, not subtract an
    absolute amount that can hit zero and stay there regardless of magnitude."""
    gov = Government(firm_subsidy_rate=0.10)
    trigger_policy(gov, "subsidy_cut", magnitude=0.20)
    assert gov.firm_subsidy_rate == pytest.approx(0.08)
    assert gov.firm_subsidy_rate > 0, "a 20% cut of a 10% rate must not saturate to 0"


def test_subsidy_cut_distinguishes_magnitudes():
    """20%, 30%, and 40% cuts from the same baseline must produce three
    DIFFERENT resulting rates. Regression test for the original bug where an
    absolute percentage-point cut made every magnitude >= baseline identical."""
    results = {}
    for mag in (0.20, 0.30, 0.40):
        gov = Government(firm_subsidy_rate=0.10)
        trigger_policy(gov, "subsidy_cut", magnitude=mag)
        results[mag] = gov.firm_subsidy_rate
    assert len(set(results.values())) == 3, f"magnitudes did not stay distinguishable: {results}"


def test_subsidy_cut_revert_round_trips():
    """Triggering the same policy twice (as when a card is shown twice live)
    should revert it back to (approximately) the original rate."""
    gov = Government(firm_subsidy_rate=0.10)
    msg1 = trigger_policy(gov, "subsidy_cut", magnitude=0.25)
    assert msg1.startswith("TRIGGERED")
    assert "subsidy_cut" in gov.active_policies
    msg2 = trigger_policy(gov, "subsidy_cut", magnitude=0.25)
    assert msg2.startswith("REVERTED")
    assert "subsidy_cut" not in gov.active_policies
    assert gov.firm_subsidy_rate == pytest.approx(0.10, abs=1e-6)


@pytest.mark.parametrize("policy_id", list(POLICY_LIBRARY.keys()))
def test_every_policy_round_trips_to_original_state(policy_id):
    """Every policy in the library should be fully reversible: apply then
    revert returns the government to its original state (within float tol)."""
    gov_before = Government()
    gov = Government()
    trigger_policy(gov, policy_id)  # apply with default magnitude
    trigger_policy(gov, policy_id)  # revert
    for field_name in ("tax_rate", "luxury_tax_surcharge", "minimum_wage",
                       "cash_transfer_low", "cash_transfer_mid", "firm_subsidy_rate"):
        assert getattr(gov, field_name) == pytest.approx(getattr(gov_before, field_name), abs=1e-6), (
            f"{policy_id} did not cleanly revert field '{field_name}'"
        )
    assert gov.active_policies == {}


@pytest.mark.parametrize("policy_id,spec", POLICY_LIBRARY.items())
def test_policy_has_required_fields(policy_id, spec):
    for key in ("label", "description", "card_color", "default_magnitude", "apply", "revert"):
        assert key in spec, f"{policy_id} is missing required field '{key}'"


def test_color_to_policy_mapping_is_unique():
    """Each color card should map to exactly one policy (no collisions)."""
    colors = [spec["card_color"] for spec in POLICY_LIBRARY.values()]
    assert len(colors) == len(set(colors)), "two policies share the same trigger card color"
    assert len(COLOR_TO_POLICY) == len(POLICY_LIBRARY)


@pytest.mark.parametrize("payload,expected_id,expected_mag", [
    ("subsidy_cut:20%", "subsidy_cut", 0.20),
    ("subsidy_cut:20", "subsidy_cut", 20.0),   # no '%' -> raw value, not divided by 100
    ("cash_transfer:50", "cash_transfer", 50.0),
    ("minimum_wage_increase:15%", "minimum_wage_increase", 0.15),
    ("luxury_tax", "luxury_tax", None),
])
def test_parse_qr_payload(payload, expected_id, expected_mag):
    policy_id, magnitude = parse_qr_payload(payload)
    assert policy_id == expected_id
    if expected_mag is None:
        assert magnitude is None
    else:
        assert magnitude == pytest.approx(expected_mag)


def test_unknown_policy_raises():
    gov = Government()
    with pytest.raises(ValueError):
        trigger_policy(gov, "not_a_real_policy")
