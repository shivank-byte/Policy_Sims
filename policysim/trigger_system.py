"""
trigger_system.py
------------------
Turns a physical color card or QR code, held up to a webcam, into a policy
event inside the simulation. Two independent detectors, both pure OpenCV
(no model training, no extra ML dependency):

  ColorCardDetector - HSV thresholding + largest-contour detection for
                      Red / Yellow / Blue / Green cards.
  QRTriggerDetector  - cv2.QRCodeDetector reads a 'policy_id:magnitude%' payload
                      straight off a printed or on-screen QR code.

Both can run against a live cv2.VideoCapture(0) feed OR against a single
still image/frame (handy for Streamlit, where you grab one frame at a time
instead of owning the capture loop yourself).
"""

from __future__ import annotations
import time
from typing import Optional
import numpy as np
import cv2

from .policies import COLOR_TO_POLICY, parse_qr_payload


# HSV ranges tuned for standard printer-paper color cards under normal indoor light.
# (Red wraps around the hue wheel, so it needs two ranges.)
HSV_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
    "blue": [((100, 100, 70), (130, 255, 255))],
    "green": [((40, 70, 70), (85, 255, 255))],
}

MIN_CARD_AREA = 4000  # pixels; ignore small color specks/noise


class ColorCardDetector:
    """Detects which single-color card (if any) dominates the current frame."""

    def __init__(self, min_area: int = MIN_CARD_AREA):
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> Optional[str]:
        """Returns a color name ('red'/'yellow'/'blue'/'green') or None."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        best_color, best_area = None, self.min_area

        for color, ranges in HSV_RANGES.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            area = max(cv2.contourArea(c) for c in contours)
            if area > best_area:
                best_area = area
                best_color = color

        return best_color

    def detect_policy(self, frame: np.ndarray) -> Optional[str]:
        color = self.detect(frame)
        return COLOR_TO_POLICY.get(color) if color else None


class QRTriggerDetector:
    """Decodes a QR code payload like 'subsidy_cut:20%' from a frame or image file."""

    def __init__(self):
        self._detector = cv2.QRCodeDetector()

    def decode(self, frame: np.ndarray) -> Optional[str]:
        data, points, _ = self._detector.detectAndDecode(frame)
        return data if data else None

    def decode_policy(self, frame: np.ndarray):
        payload = self.decode(frame)
        if not payload:
            return None, None
        return parse_qr_payload(payload)

    def decode_from_file(self, path: str):
        img = cv2.imread(path)
        if img is None:
            return None, None
        return self.decode_policy(img)


class PolicyTriggerSystem:
    """
    High-level convenience wrapper used by the dashboard: owns both detectors
    and a debounce timer so holding up one card doesn't fire 30 events/second.
    """

    def __init__(self, debounce_seconds: float = 3.0):
        self.color_detector = ColorCardDetector()
        self.qr_detector = QRTriggerDetector()
        self.debounce_seconds = debounce_seconds
        self._last_trigger_time = 0.0
        self._last_policy = None

    def process_frame(self, frame: np.ndarray):
        """
        Returns (policy_id, magnitude) if a NEW trigger is detected and the
        debounce window has passed, otherwise (None, None).
        """
        now = time.time()
        if now - self._last_trigger_time < self.debounce_seconds:
            return None, None

        # QR codes take priority (they carry an explicit magnitude)
        policy_id, magnitude = self.qr_detector.decode_policy(frame)
        if policy_id:
            self._last_trigger_time = now
            return policy_id, magnitude

        policy_id = self.color_detector.detect_policy(frame)
        if policy_id:
            self._last_trigger_time = now
            return policy_id, None

        return None, None

    def run_webcam_loop(self, on_trigger, camera_index: int = 0, show_window: bool = True):
        """
        Blocking loop for local (non-Streamlit) use: opens the webcam, calls
        `on_trigger(policy_id, magnitude)` whenever a new card/QR is detected,
        and shows a debug window. Press 'q' to quit.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                policy_id, magnitude = self.process_frame(frame)
                if policy_id:
                    on_trigger(policy_id, magnitude)
                if show_window:
                    cv2.imshow("PolicySim - Trigger Camera (q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
