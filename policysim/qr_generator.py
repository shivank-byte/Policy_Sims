"""
qr_generator.py
---------------
Generates printable QR-code trigger cards, one per policy in POLICY_LIBRARY,
encoding a payload like 'subsidy_cut:20%'. Requires the `qrcode` package
(pip install qrcode[pil]) -- imported lazily so the rest of PolicySim works
fine even if this optional package isn't installed.

Usage:
    python -m policysim.qr_generator            # writes PNGs to assets/qr_codes/
"""

from __future__ import annotations
import os
from .policies import POLICY_LIBRARY


def generate_all(output_dir: str = "assets/qr_codes"):
    try:
        import qrcode
    except ImportError as e:
        raise ImportError(
            "The 'qrcode' package is required for QR generation. "
            "Install it with: pip install qrcode[pil]"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    written = []
    for policy_id, spec in POLICY_LIBRARY.items():
        magnitude = spec["default_magnitude"]
        # cash_transfer's magnitude is a currency amount, others are percentages
        payload = f"{policy_id}:{magnitude}" if policy_id == "cash_transfer" else f"{policy_id}:{magnitude*100:.0f}%"
        img = qrcode.make(payload)
        path = os.path.join(output_dir, f"{policy_id}.png")
        img.save(path)
        written.append((path, payload, spec["label"]))
    return written


if __name__ == "__main__":
    results = generate_all()
    print(f"Generated {len(results)} QR trigger cards:")
    for path, payload, label in results:
        print(f"  {path:<35} payload='{payload}'  ({label})")
