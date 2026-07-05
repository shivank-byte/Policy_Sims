"""
calibrate_camera.py
--------------------
Interactive webcam calibration tool for the color-card trigger system.

The HSV_RANGES baked into trigger_system.py are tuned for "standard printer
paper under normal indoor light" -- real venues (stage lighting, a dim
conference room, a sunlit window behind the presenter) will need different
thresholds, or the color detector will miss cards or misfire on the wrong
color. This tool lets you tune those ranges live against YOUR camera and
YOUR lighting, instead of guessing.

Usage:
    python -m policysim.calibrate_camera                 # tune all 4 colors in turn
    python -m policysim.calibrate_camera --color red      # tune just one color
    python -m policysim.calibrate_camera --camera 1       # use a different camera index

Controls while running:
  - Hold up the color card you're calibrating in front of the camera.
  - Drag the trackbars in the "Calibration" window until the "Mask" window
    shows a clean white blob over the card (and black everywhere else).
  - Press 's' to save the current color's tuned range and move to the next.
  - Press 'n' to skip a color without saving (keep the built-in default).
  - Press 'q' to quit early.

At the end, the tool prints a ready-to-paste HSV_RANGES dict AND writes it
to `assets/calibrated_hsv_ranges.json` -- copy either into
trigger_system.py's HSV_RANGES before a live demo in a new venue.
"""

from __future__ import annotations
import argparse
import json
import os

import cv2
import numpy as np

from .trigger_system import HSV_RANGES as DEFAULT_HSV_RANGES, MIN_CARD_AREA

WINDOW = "Calibration"
MASK_WINDOW = "Mask (tune until this is a clean white blob on the card)"

COLORS_IN_ORDER = ["red", "yellow", "blue", "green"]


def _nothing(_):
    pass


def _make_trackbars(initial_lo, initial_hi):
    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("H min", WINDOW, initial_lo[0], 179, _nothing)
    cv2.createTrackbar("S min", WINDOW, initial_lo[1], 255, _nothing)
    cv2.createTrackbar("V min", WINDOW, initial_lo[2], 255, _nothing)
    cv2.createTrackbar("H max", WINDOW, initial_hi[0], 179, _nothing)
    cv2.createTrackbar("S max", WINDOW, initial_hi[1], 255, _nothing)
    cv2.createTrackbar("V max", WINDOW, initial_hi[2], 255, _nothing)


def _read_trackbars():
    lo = (cv2.getTrackbarPos("H min", WINDOW), cv2.getTrackbarPos("S min", WINDOW), cv2.getTrackbarPos("V min", WINDOW))
    hi = (cv2.getTrackbarPos("H max", WINDOW), cv2.getTrackbarPos("S max", WINDOW), cv2.getTrackbarPos("V max", WINDOW))
    return lo, hi


def calibrate_color(cap, color: str, existing_ranges: dict) -> list:
    """Live-tunes one color's HSV range against the camera feed.
    Returns the tuned [(lo, hi), ...] range list for this color (red keeps
    its two-range wraparound shape; others get a single range)."""
    # seed the trackbars from whichever range is widest for this color (or a
    # sane default if the color isn't in the current ranges at all)
    ranges = existing_ranges.get(color, [((0, 100, 70), (10, 255, 255))])
    lo0, hi0 = ranges[0]
    _make_trackbars(lo0, hi0)

    print(f"\n=== Calibrating '{color}' ===")
    print("Hold up the card. Drag trackbars until the mask window shows a clean "
          "white blob over the card. Press 's' to save, 'n' to skip, 'q' to quit.")

    saved_range = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("  (camera read failed)")
            break

        lo, hi = _read_trackbars()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = max((cv2.contourArea(c) for c in contours), default=0)
        detected = area > MIN_CARD_AREA

        preview = frame.copy()
        status = f"{color.upper()}  area={int(area):>6}  {'DETECTED' if detected else 'no card'}"
        cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if detected else (0, 0, 255), 2)
        cv2.imshow(WINDOW, preview)
        cv2.imshow(MASK_WINDOW, mask)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("s"):
            saved_range = [(lo, hi)]
            if color == "red":
                # red wraps the hue wheel; keep a second high-hue band so the
                # calibrated range still covers wraparound reds
                saved_range.append(((170, lo[1], lo[2]), (180, hi[1], hi[2])))
            print(f"  Saved '{color}': {saved_range}")
            break
        elif key == ord("n"):
            print(f"  Skipped '{color}' (keeping existing range).")
            break
        elif key == ord("q"):
            print("Quitting calibration early.")
            return None

    cv2.destroyWindow(WINDOW)
    cv2.destroyWindow(MASK_WINDOW)
    return saved_range if saved_range is not None else ranges


def run(camera_index: int = 0, only_color: str = None, output_path: str = "assets/calibrated_hsv_ranges.json"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Check the camera is connected, "
            f"not in use by another app, and that this process has camera permission."
        )

    colors = [only_color] if only_color else COLORS_IN_ORDER
    tuned = dict(DEFAULT_HSV_RANGES)

    try:
        for color in colors:
            result = calibrate_color(cap, color, tuned)
            if result is None:  # user pressed 'q'
                break
            tuned[color] = result
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n=== Calibrated HSV_RANGES (paste into trigger_system.py) ===\n")
    print("HSV_RANGES = {")
    for color, ranges in tuned.items():
        print(f"    {color!r}: {ranges},")
    print("}\n")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tuned, f, indent=2)
    print(f"Also saved to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Calibrate PolicySim's color-card HSV thresholds to your camera/lighting")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--color", choices=COLORS_IN_ORDER, default=None,
                   help="calibrate just one color instead of all four")
    p.add_argument("--output", default="assets/calibrated_hsv_ranges.json")
    args = p.parse_args()
    run(camera_index=args.camera, only_color=args.color, output_path=args.output)


if __name__ == "__main__":
    main()
