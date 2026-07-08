"""
dashboard.py
------------
The live, audience-facing Streamlit dashboard.

Run it with:
    streamlit run policysim/dashboard.py

Panels:
  - Sidebar: reasoning backend picker, playback controls, manual policy
    trigger buttons (color-card equivalent), QR image upload, optional
    live webcam trigger loop.
  - Main: real-time charts (price level, inequality, spending by tier),
    a scrolling "agent thoughts" feed for transparency, an event log, and
    an automatic before/after table whenever a policy has been triggered.

Design note: Streamlit reruns the whole script on every interaction, so all
simulation state lives in `st.session_state` and is never rebuilt unless the
user explicitly resets it.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

# Make sure the project root (parent of the `policysim` package folder) is on
# sys.path, so `streamlit run policysim/dashboard.py` works no matter which
# directory it's launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policysim.simulation import Simulation
from policysim.brain import AgentBrain
from policysim.policies import POLICY_LIBRARY
from policysim.stats_engine import before_after_comparison, price_elasticity_of_demand
from policysim.trigger_system import PolicyTriggerSystem, QRTriggerDetector, ColorCardDetector

st.set_page_config(page_title="PolicySim", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_state():
    if "sim" not in st.session_state:
        st.session_state.sim = None
    if "backend" not in st.session_state:
        st.session_state.backend = "heuristic"
    if "trigger_round_marks" not in st.session_state:
        st.session_state.trigger_round_marks = []  # (round, label) for chart annotations
    if "trigger_system" not in st.session_state:
        st.session_state.trigger_system = PolicyTriggerSystem()
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False


def build_sim(backend: str, seed: int):
    st.session_state.sim = Simulation(brain=AgentBrain(backend=backend), seed=seed)
    st.session_state.trigger_round_marks = []


init_state()


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 PolicySim Controls")

    st.subheader("1. Reasoning engine")
    backend = st.selectbox(
        "Agent reasoning backend",
        ["heuristic", "ollama", "groq"],
        help=("heuristic = offline, zero-dependency bounded-rationality rules (always works). "
              "ollama = local Llama 3.1 8B via http://localhost:11434 (free, offline, needs Ollama running). "
              "groq = hosted free-tier Llama (needs internet + API key)."),
    )
    groq_key = None
    if backend == "groq":
        try:
            default_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            default_key = ""  # no secrets configured at all -- fine, just start blank
        groq_key = st.text_input("Groq API key", type="password", value=default_key,
                                  help="Auto-filled from Streamlit secrets (GROQ_API_KEY) if set, "
                                       "so you don't have to type it in on mobile each session.")
    if st.button("🔄 (Re)start simulation", type="primary"):
        b = AgentBrain(backend=backend, groq_api_key=groq_key) if backend == "groq" else AgentBrain(backend=backend)
        st.session_state.sim = Simulation(brain=b, seed=np.random.randint(0, 100000))
        st.session_state.trigger_round_marks = []

    st.divider()
    st.subheader("2. Playback")
    col1, col2 = st.columns(2)
    step_1 = col1.button("▶ Step 1 round")
    step_5 = col2.button("⏩ Step 5 rounds")
    st.session_state.auto_play = st.toggle("Auto-play (1 round/sec)", value=st.session_state.auto_play)

    st.divider()
    st.subheader("3. Trigger a policy live")
    st.caption("Equivalent of holding up a color card in front of the audience.")
    for policy_id, spec in POLICY_LIBRARY.items():
        if st.button(f"🟥🟨🟦🟩  {spec['label']}", key=f"btn_{policy_id}", help=spec["description"]):
            if st.session_state.sim:
                msg = st.session_state.sim.apply_policy_event(policy_id)
                st.session_state.trigger_round_marks.append((st.session_state.sim.round_num, spec["label"]))
                st.toast(msg)

    st.divider()
    st.subheader("4. 📱 Trigger via camera (phone or laptop — works anywhere)")
    st.caption(
        "Uses your browser's own camera picker, so this works from a phone even when the "
        "app is hosted on Streamlit Cloud — no webcam or laptop required. Point at a color "
        "card or QR trigger card and tap the capture button."
    )
    cam_photo = st.camera_input("Point at a card, then capture", label_visibility="collapsed")
    if cam_photo is not None and st.session_state.sim is not None:
        # st.camera_input keeps returning the same object across reruns until the user
        # retakes a photo -- only act on a genuinely NEW capture, not every rerun.
        if st.session_state.get("last_camera_photo_id") != cam_photo.file_id:
            st.session_state.last_camera_photo_id = cam_photo.file_id
            import cv2
            file_bytes = np.frombuffer(cam_photo.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            policy_id, magnitude = QRTriggerDetector().decode_policy(img)  # QR checked first (carries magnitude)
            if not policy_id:
                policy_id, magnitude = ColorCardDetector().detect_policy(img), None

            if policy_id and policy_id in POLICY_LIBRARY:
                msg = st.session_state.sim.apply_policy_event(policy_id, magnitude)
                st.session_state.trigger_round_marks.append(
                    (st.session_state.sim.round_num, POLICY_LIBRARY[policy_id]["label"])
                )
                st.success(msg)
            else:
                st.warning(
                    "No recognized card or QR code found in that photo. Try moving closer, "
                    "better lighting, or holding the card flatter. If color cards keep "
                    "missing, run `python -m policysim.calibrate_camera` to retune the "
                    "detector to your lighting."
                )

    with st.expander("Alternative: upload a photo instead of using the live camera picker"):
        qr_file = st.file_uploader("Upload a photo of a QR or color trigger card", type=["png", "jpg", "jpeg"])
        if qr_file is not None and st.session_state.sim is not None:
            import cv2
            file_bytes = np.frombuffer(qr_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            policy_id, magnitude = QRTriggerDetector().decode_policy(img)
            if not policy_id:
                policy_id, magnitude = ColorCardDetector().detect_policy(img), None
            if policy_id and policy_id in POLICY_LIBRARY:
                msg = st.session_state.sim.apply_policy_event(policy_id, magnitude)
                st.session_state.trigger_round_marks.append((st.session_state.sim.round_num, POLICY_LIBRARY[policy_id]["label"]))
                st.success(msg)
            else:
                st.warning("No recognized policy QR code or color card found in that image.")

    st.divider()
    st.subheader("5. Continuous local webcam loop (optional, laptop only)")
    st.caption(
        "For a hands-free live demo where cards are held up continuously without tapping "
        "capture each time. Requires running `streamlit run` on a laptop with a webcam — "
        "not available on Streamlit Cloud or from a phone."
    )
    cam_on = st.toggle("Enable continuous webcam trigger loop")
    cam_slot_sidebar = st.empty()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("PolicySim — Live Generative Agent-Based Economy")
st.caption("Households, firms, and government agents reason each round. Trigger a real policy shock "
           "live and watch the AI economy react.")

sim: Simulation = st.session_state.sim
if sim is None:
    st.info("👈 Click **(Re)start simulation** in the sidebar to begin.")
    st.stop()

# ---- advance simulation based on controls ----
if step_1:
    sim.run_round()
if step_5:
    sim.run_n_rounds(5)
if cam_on:
    import cv2
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cam_slot_sidebar.image(frame, channels="BGR", caption="Live camera frame", width=200)
        policy_id, magnitude = st.session_state.trigger_system.process_frame(frame)
        if policy_id and policy_id in POLICY_LIBRARY:
            msg = sim.apply_policy_event(policy_id, magnitude)
            st.session_state.trigger_round_marks.append((sim.round_num, POLICY_LIBRARY[policy_id]["label"]))
            st.toast(msg)

df = sim.history_df()

# ---- top-line metrics ----
if len(df):
    latest = df.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Round", int(latest["round"]))
    m2.metric("Price Level", f"{latest['price_level']:.3f}",
              f"{(latest['price_level'] - df.iloc[0]['price_level'])*100:+.1f}% vs R1")
    m3.metric("Wealth Gini", f"{latest['gini']:.3f}")
    m4.metric("Unemployment", f"{latest['unemployment_rate']*100:.1f}%")
    m5.metric("Total Spending", f"{latest['total_spending']:,.0f}")

    st.subheader("Active policies")
    st.write(", ".join(sim.government.active_policy_labels()) or "_none currently active_")

    # ---- charts ----
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Price Level**")
        st.line_chart(df.set_index("round")[["price_level"]])
        st.markdown("**Wealth Inequality (Gini coefficient)**")
        st.line_chart(df.set_index("round")[["gini"]])
    with c2:
        st.markdown("**Unemployment Rate**")
        st.line_chart(df.set_index("round")[["unemployment_rate"]])
        st.markdown("**Spending by Income Tier**")
        st.line_chart(df.set_index("round")[["spend_low", "spend_mid", "spend_high"]])

    if st.session_state.trigger_round_marks:
        st.caption("📍 Policy triggers so far: " +
                   "; ".join(f"R{r}: {label}" for r, label in st.session_state.trigger_round_marks))

    # ---- before/after analysis ----
    st.subheader("📈 Before / After Policy Analysis")
    if st.session_state.trigger_round_marks:
        last_round, last_label = st.session_state.trigger_round_marks[-1]
        window = min(last_round - 1, 5) if last_round > 1 else 1
        if window >= 1 and last_round + 1 <= df["round"].max():
            comp = before_after_comparison(df, trigger_round=last_round, window=window)
            st.write(f"Comparing {window} rounds before vs after **{last_label}** (triggered at round {last_round}):")
            st.dataframe(
                comp.style.apply(
                    lambda row: ["background-color: #d4f7d4" if row["significant"] else "" for _ in row],
                    axis=1,
                ),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "`p_value` is from Welch's t-test (unequal variances); `ci_low`/`ci_high` is a "
                "bootstrap 95% CI on the mean difference. `significant` requires both p < 0.05 "
                "AND the bootstrap CI to exclude zero — with only a handful of rounds per window, "
                "treat this as a directional signal, not a definitive result."
            )

            st.markdown("**Price-elasticity-of-demand sanity check**")
            elas = price_elasticity_of_demand(df, trigger_round=last_round, window=window)
            if elas["in_textbook_range"] is None:
                st.info("Not enough data yet to estimate elasticity for this window.")
            else:
                lo, hi = elas["textbook_range"]
                badge = "✅ within typical textbook range" if elas["in_textbook_range"] else "⚠️ outside typical textbook range"
                st.write(
                    f"Estimated elasticity: **{elas['elasticity']}** "
                    f"(price {elas['pct_price_change']:+.1f}%, real quantity {elas['pct_quantity_change']:+.1f}%) "
                    f"— textbook range ≈ ({lo}, {hi}) — {badge}"
                )
                if not elas["in_textbook_range"]:
                    st.caption(
                        "An out-of-range estimate is a flag to investigate the demand/pricing logic "
                        "(or that this toy economy's parameters just don't map onto real-world goods) "
                        "— not something to silently adjust away."
                    )
        else:
            st.info("Step forward a few more rounds to build up an 'after' window for comparison.")
    else:
        st.info("Trigger a policy from the sidebar to see a before/after comparison here.")

    # ---- agent thought feed ----
    st.subheader("🧠 Live Agent Thought Feed")
    feed = sim.thought_feed[-25:][::-1]
    for entry in feed:
        st.markdown(f"`R{entry['round']}` **{entry['agent']}** — {entry['thought']}")

    # ---- event log ----
    with st.expander("📜 Full policy event log"):
        if sim.event_log:
            st.dataframe(pd.DataFrame(sim.event_log), use_container_width=True, hide_index=True)
        else:
            st.write("No policy events triggered yet.")

    # ---- raw data / download ----
    with st.expander("🔢 Raw round-by-round data"):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="policysim_history.csv")

else:
    st.info("No rounds run yet — click **Step 1 round** in the sidebar.")

# ---- auto-play rerun ----
if st.session_state.auto_play:
    time.sleep(1)
    sim.run_round()
    st.rerun()
