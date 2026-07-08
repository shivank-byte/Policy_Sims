import numpy as np
import pytest

from policysim.trigger_system import ColorCardDetector
from policysim.policies import COLOR_TO_POLICY, POLICY_LIBRARY


def _synthetic_frame(bgr_color):
    """A neutral-gray frame with a solid color rectangle in the middle,
    standing in for a real camera photo of someone holding up a color
    card -- big enough to clear MIN_CARD_AREA."""
    frame = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    frame[100:380, 150:490] = bgr_color
    return frame


@pytest.mark.parametrize("color_name,bgr", [
    ("red", (30, 30, 210)),
    ("yellow", (30, 220, 220)),
    ("blue", (210, 60, 30)),
    ("green", (30, 160, 30)),
])
def test_color_card_detector_identifies_each_card_color(color_name, bgr):
    detected = ColorCardDetector().detect(_synthetic_frame(bgr))
    assert detected == color_name


def test_every_detectable_color_maps_to_a_real_policy():
    """This is what streamlit_app.py's camera trigger actually relies on --
    every color the detector can return must resolve to a valid policy."""
    for color_name, bgr in [("red", (30, 30, 210)), ("yellow", (30, 220, 220)),
                             ("blue", (210, 60, 30)), ("green", (30, 160, 30))]:
        detected = ColorCardDetector().detect(_synthetic_frame(bgr))
        policy_id = COLOR_TO_POLICY.get(detected)
        assert policy_id is not None, f"{color_name} did not map to a policy"
        assert policy_id in POLICY_LIBRARY


def test_no_card_present_returns_none():
    blank = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    assert ColorCardDetector().detect(blank) is None


def test_small_color_patch_below_min_area_is_ignored():
    """A tiny fleck of red in the background (e.g. a red logo) shouldn't
    falsely trigger a policy -- only a card filling a meaningful area of
    the frame should count."""
    frame = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    frame[10:30, 10:30] = (30, 30, 210)  # small red patch, well under MIN_CARD_AREA
    assert ColorCardDetector().detect(frame) is None
