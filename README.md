# PolicySim
### Generative Agent-Based Economic Simulation with Live Policy Triggers

[![tests](https://github.com/USERNAME/policysim/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/policysim/actions/workflows/tests.yml)

> Replace `USERNAME` above with your GitHub username/org once this is pushed to a repo — the badge is wired to `.github/workflows/tests.yml`, which runs the 52-test regression suite on every push/PR.


A miniature economy run by AI agents — households, firms, and a government —
that reason in plain language and act each round. An audience can trigger a
real policy shock live, by holding up a color card or QR code to a webcam,
and watch the simulated economy react in real time, with the statistics
(inflation, inequality, spending shifts) to back it up.

Built for an MSc Economics portfolio: it works completely offline, costs
nothing to run, and generates its own data for genuine before/after
statistical analysis rather than just calling a pretrained model.

---

## 0. Deploying — one single interactive app (Streamlit Cloud)

`streamlit_app.py` (repo root) is the one file to point Streamlit Cloud at.
It's a single deployment with everything working together:

- The sidebar has real controls — Step round, Restart, and the 4 policy
  toggles — wired straight to the actual `Simulation` object.
- The main panel embeds the 3D Three.js village, fed the live simulation
  state as JSON each time the sidebar triggers a rerun.
- Click any house/firm/Town Hall in the 3D view for a read-only stats
  panel with that agent's actual last-round reasoning.

**Streamlit Cloud setup:** New app → pick this repo → main file path
`streamlit_app.py` → deploy. No FastAPI server, no separate front-end —
one URL, one process.

One trade-off worth knowing: Streamlit reruns the whole script on every
button click, which rebuilds the embedded 3D view each time — so the
camera angle resets on each step/toggle. Just re-orbit after; it doesn't
affect the underlying data.

(`server.py` + `policysim/static/index.html` also still exist if you'd
rather run a standalone FastAPI + 3D setup locally — see section 6 — but
for Streamlit Cloud, `streamlit_app.py` is the one you want.)

### 0.1 Camera-based color card trigger — important correction

The original `trigger_system.py`/`calibrate_camera.py` use
`cv2.VideoCapture(0)`, which opens a camera physically attached to the
*machine running the Python process*. On Streamlit Cloud, that machine is
a remote server with no camera at all — this code can **never** access
your own laptop/phone camera when deployed to the cloud, no matter what
settings you change. That's a hard platform limitation, not a bug to
work around.

The correct fix, already built into `streamlit_app.py`: **`st.camera_input()`**,
Streamlit's own widget that asks your *browser* for camera permission
(via the same API any website uses to request your webcam) and uploads a
single photo to the app. In the sidebar, under "📷 Or trigger with a color
card," take a photo of a plain red/yellow/blue/green card and it runs the
same `ColorCardDetector` HSV logic from `trigger_system.py` on that one
frame, then triggers the matching policy exactly like the live-demo
color-card system does. This works identically on a phone or a laptop,
because the browser (not the server) owns the camera.

**Streamlit Cloud setup requirement:** the full `opencv-python` package
(used elsewhere for the local live-demo/calibration tools) needs system
graphics libraries that Streamlit Cloud's base container doesn't ship by
default, and importing it without them fails with
`ImportError: libGL.so.1: cannot open shared object file`. This repo
includes a **`packages.txt`** at the root (`libgl1`, `libglib2.0-0`) —
Streamlit Cloud automatically `apt-get install`s anything listed there
before running `pip install`, which resolves this without needing a
separate headless opencv build. If you ever see that ImportError anyway,
double check `packages.txt` actually made it into your repo (mobile
GitHub uploads sometimes skip dotfile-adjacent files — this one doesn't
start with a dot, so it should upload normally, but it's easy to miss in
a file picker).

If opencv fails to import for any reason, `streamlit_app.py` catches it
and disables just the camera section with a clear message — the rest of
the app (3D view, manual policy buttons, Groq) keeps working regardless.

### 0.2 What goes in Streamlit secrets

Only one thing, and only if you want live LLM reasoning instead of the
default heuristic engine: on Streamlit Cloud, go to your app → **Settings
→ Secrets** and add exactly this —

```toml
GROQ_API_KEY = "your-actual-groq-api-key"
```

That's the only secret this app uses. Nothing else needs to go in
Secrets — not a database URL, not another model's key, nothing. If
`GROQ_API_KEY` is present, `streamlit_app.py` automatically uses the Groq
backend (see `_make_simulation()`); if it's absent or a Groq call fails
mid-session, it falls back to the heuristic backend automatically (see
`AgentBrain`'s `fallback_count` shown in the sidebar). Never put the key
directly in `streamlit_app.py` or any file you commit to GitHub — Streamlit's
Secrets store is the only place it should live.

---

## 0.5 Results — does the cash transfer policy actually do anything?

**Full writeup with charts: [`policysim_results_report.pdf`](policysim_results_report.pdf).**
This section is the short version. Every number below is computed live
by `build_report.py`/the experiment scripts, not hand-typed — see §0.6.

`experiments/cash_transfer_effect.py` runs the real engine 40 times with
`cash_transfer` off and 40 times with it on from round 1, for 20 rounds
each, on the default population (24 households, 4 firms), and compares
round-20 outcomes with Welch's t-test:

```bash
python -m experiments.cash_transfer_effect
```

```
metric                control mean  treatment mean    t-stat   p-value   Cohen d
--------------------------------------------------------------------------------
gini                        0.4172          0.4661    18.175    0.0000     4.064
gini_income                 0.2215          0.2128     exact    diff=-0.0087
unemployment_rate           0.3344          0.3219    -3.961    0.0002    -0.886
price_level                 1.0232          1.0249     0.578    0.5646     0.129
total_spending         358042.3125     406896.6394    25.354    0.0000     5.669
```

**Total spending rises significantly (p < 0.0001)** — expected: more
disposable income for low/mid households means more demand. Price level
does *not* move significantly here (p=0.56) at this population/magnitude
— worth noting since an earlier, smaller population did show a
significant price effect; scale changes what's detectable.

**Wealth Gini rises significantly (Cohen's d = 4.06) while income Gini
falls (exact, deterministic difference)** — this is the headline,
counter-intuitive result: a flat cash transfer *widens* wealth (savings)
inequality even though it *narrows* income inequality in the same runs.
The transfer is a fixed rupee amount, small relative to high-tier
households' base income, so higher-income households keep compounding
savings at their usual pace while low/mid tiers spend a larger share of
their (now slightly bigger) income rather than saving it. See
`assets/charts/wealth_gini_over_time.png` and `income_gini_over_time.png`
for the full 20-round trend (not just the before/after snapshot) — the
two measures diverge almost immediately and stay apart.

**Unemployment itself falls significantly too (p=0.0002, 33.4% → 32.2%)**
— with real labor-market slack in the default population (18 starting
jobs vs. 24 households), the transfer's demand boost measurably moves
employment on its own. This used to require a special override scenario
to demonstrate — the *default* population was small enough (10
households, 2 firms offering 11 jobs) that labor demand always exceeded
headcount and unemployment was pinned at exactly 0%, a real bug (see
§6.3, bug #3). Expanding to 24 households / 4 firms with only 18
starting jobs fixed this for the shipped default, not just a workaround
experiment.

This mirrors real research on marginal propensity to consume (MPC) by
income/wealth tier: PSID-based estimates put MPC around 0.15 for the
lowest wealth quintile vs. ~0.06 for the highest (Fisher, Johnson,
Smeeding & Thompson); Penn Wharton Budget Model recession-calibrated
estimates show a similar gradient by income quintile (~0.55 down to
~0.12). `simulation.py`'s household consumption propensities are loosely
tuned to that gradient — the wealth/income Gini split isn't an arbitrary
quirk of this toy model.

**Does a bigger transfer make it worse?** Not linearly.
`experiments/magnitude_sweep_cash_transfer.py` averages 15 seeds per
magnitude (a single seed per magnitude was too noisy to trust) and finds
wealth Gini jumps as soon as *any* transfer is applied, then slightly
recedes as the transfer keeps growing — a threshold effect with a mild
ceiling, not a dial that keeps making inequality worse the harder you
push it. Income Gini, by contrast, falls smoothly and monotonically with
magnitude, exactly as expected:

```bash
python -m experiments.magnitude_sweep_cash_transfer
```

```
 magnitude   wealth gini   income gini   price level
------------------------------------------------------
         0        0.4191        0.2216        1.0226
       250        0.4601        0.2195        1.0226
       500        0.4593        0.2173        1.0226
      1000        0.4576        0.2132        1.0226
      2000        0.4542        0.2052        1.0226
      4000        0.4475        0.1906        1.0331
```

**Does the wealth-Gini effect depend on employment risk?** Tested
directly with the opposite extreme: `experiments/unemployment_slack.py`
(kept its filename, repurposed) raises firm headcount enough that jobs
comfortably exceed households (30 jobs vs. 24), pinning unemployment at
exactly 0% *by construction* — the same situation the old default used
to have by accident. If the wealth-Gini effect vanished here, that would
mean employment risk was secretly driving the whole result:

```bash
python -m experiments.unemployment_slack
```

```
metric                control mean  treatment mean    t-stat   p-value   Cohen d
--------------------------------------------------------------------------------
gini                        0.5179          0.5642    34.444    0.0000     7.702
gini_income                 0.3094          0.2994     exact    diff=-0.0100
unemployment_rate           0.0000          0.0000     exact    diff=+0.0000
price_level                 0.9785          1.0041     8.149    0.0000     1.822
```

Wealth Gini still rises significantly (p<0.0001, d=7.70) with zero
employment risk possible — confirming the mechanism really is differing
marginal propensity to consume by tier, not employment risk in disguise.

**Robustness check:** does the wealth-Gini finding survive a different
population size? `run_experiment()` in `cash_transfer_effect.py` was
re-run with double the household count (48 instead of 24, same firms) —
direction and significance held (d=3.67 vs. baseline d=4.06), so this
looks like a stable emergent property of the mechanism rather than a
fluke of one specific population size.

**Regenerating everything:**
```bash
python -m experiments.generate_charts       # writes assets/charts/*.png
python build_report.py                       # regenerates policysim_results_report.pdf
```

**Caveats:** this is a 24–48 household / 4-firm toy economy on a 20-round
horizon with the heuristic reasoning backend only — see §7 of the PDF
report for the full limitations list (population size, single-backend
results, multiple-comparisons across 5 metrics per experiment, etc).

## 0.6 A note on reproducibility discipline

Every number in §0.5 and in `policysim_results_report.pdf` is computed at
run/build time by importing and calling the real experiment functions —
`build_report.py` used to hardcode all of these as literal strings, which
meant a real fix (like the degenerate-stats bug below) could get fixed in
code while the report kept silently showing the old, wrong numbers
forever. That's fixed now (§6.3, bug #6) — if you change `brain.py`,
`simulation.py`'s defaults, or anything else that affects these numbers,
re-running `python build_report.py` will always reflect the *current*
code, not a stale snapshot.

---

## 1. What's actually in this repo

```
policysim/
├── policysim/
│   ├── agents.py             Household, Firm, Government dataclasses
│   ├── brain.py              Reasoning engine: heuristic / Ollama / Groq backends
│   ├── policies.py           Policy catalogue + card/QR → policy mapping
│   ├── simulation.py         Round-based engine, market clearing, multi-run harness
│   ├── stats_engine.py       Gini, before/after significance testing, elasticity check
│   ├── trigger_system.py     OpenCV color-card + QR-code detectors
│   ├── calibrate_camera.py   Interactive HSV calibration tool for your venue/lighting
│   ├── qr_generator.py       Generates printable QR trigger cards
│   ├── dashboard.py          Streamlit live dashboard (2D charts)
│   ├── cli_demo.py           Terminal-only runner (no Streamlit/webcam needed)
│   ├── server.py             Optional standalone FastAPI server for the 3D view
│   ├── view_helpers.py       Shared state → JSON serialization (server.py + streamlit_app.py)
│   └── static/index.html     3D Three.js village (used by server.py)
├── streamlit_app.py          Single-deployment entrypoint: 2D controls + live 3D view
├── experiments/
│   ├── cash_transfer_effect.py       40-vs-40 run stat test + 2x-population robustness check (§0.5)
│   ├── magnitude_sweep_cash_transfer.py  Does a bigger transfer widen wealth Gini more? (§0.5)
│   ├── unemployment_slack.py         Confirms the unemployment-ceiling diagnosis with real slack (§0.5)
│   └── generate_charts.py            Round-by-round Gini/price charts -> assets/charts/*.png
├── assets/charts/                    Generated PNGs used in the results report
├── build_report.py                   Generates policysim_results_report.pdf
├── policysim_results_report.pdf      Full results writeup with charts, limitations, literature
├── .github/workflows/tests.yml       CI: runs the regression suite on push/PR
├── tests/                    Regression test suite (pytest)
├── requirements.txt
└── README.md
```

Everything is genuinely wired together and has been tested end-to-end in
**heuristic mode** (see §3) — the simulation, the statistics, and the color
card detector all run correctly out of the box with zero external services.
The regression suite (§7) locks in six real bugs found and fixed during
development (see §6.3).

---

## 2. Quick start

```bash
pip install -r requirements.txt

# Terminal-only run, no webcam or LLM needed:
python -m policysim.cli_demo --rounds 24 --policy cash_transfer --at 10

# Live dashboard (the real demo):
streamlit run policysim/dashboard.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). Click
**"(Re)start simulation"**, then **Step 1 round** a few times, then trigger a
policy from the sidebar and watch the charts and agent "thoughts" react.

---

## 3. The three reasoning backends — and why this matters

Every agent decision goes through `AgentBrain.decide()`, which supports three
interchangeable backends:

| Backend | What it is | Cost | Needs internet? |
|---|---|---|---|
| `heuristic` **(default)** | Pure-Python bounded-rationality rules — price sensitivity, unemployment fear, policy-specific reactions, with small random noise | Free | No |
| `ollama` | Local Llama 3.1 8B via `http://localhost:11434` | Free | No |
| `groq` | Hosted free-tier Llama models | Free tier | Yes |

**Why a heuristic fallback exists at all:** a live demo in front of an
audience cannot depend on venue wifi or a model finishing warm-up mid-round.
`AgentBrain` tries your chosen backend and **automatically falls back to the
heuristic** if it times out or errors — so the demo literally cannot break on
stage. It also means you can develop, test, and present the entire
statistical/analytical side of the project without installing an LLM at all.

To use real generative reasoning:
```bash
# Option A — local, offline, free
ollama pull llama3.1:8b
ollama serve
# then in the dashboard sidebar, pick backend = "ollama"

# Option B — hosted, needs internet, very fast
# get a free key at https://console.groq.com
# pick backend = "groq" in the sidebar and paste the key
```

---

## 4. The policy catalogue

| Policy | Card color | Effect |
|---|---|---|
| Fuel/Input Subsidy Cut | 🟥 Red | Raises firms' effective input costs |
| Minimum Wage Increase | 🟨 Yellow | Raises the legal wage floor |
| Luxury Tax | 🟦 Blue | Adds a tax surcharge on high-income households |
| Cash Transfer | 🟩 Green | Direct payment to low/mid income households each round |

Showing the **same** card twice reverts the policy — useful for a live demo
where you want to show cause *and* effect reversibility.

QR codes encode `policy_id:magnitude`, e.g. `subsidy_cut:20%` or
`cash_transfer:50`. Generate a printable set:
```bash
python -m policysim.qr_generator
# writes assets/qr_codes/*.png
```

---

## 5. The trigger system (OpenCV, zero model training)

`trigger_system.py` has two independent, tested detectors:

- **`ColorCardDetector`** — HSV thresholding + largest-contour detection,
  tuned for standard printer-paper cards under normal indoor light.
- **`QRTriggerDetector`** — `cv2.QRCodeDetector` decodes the payload directly.

`PolicyTriggerSystem` wraps both with a 3-second debounce so holding up one
card doesn't fire dozens of duplicate events.

**You do NOT need a physical webcam or a laptop to trigger policies live.**
The dashboard's sidebar has `st.camera_input`, which opens your **phone's own
camera through the browser** and snaps a photo — this works identically
whether the app is running locally or hosted free on Streamlit Cloud, since
the capture happens client-side in the browser and gets uploaded. Just open
the dashboard URL on your phone, tap the capture button, and point at a
color card or QR code.

Two other options exist for different situations:
- **Upload a photo** (in the sidebar's expander) if you'd rather pick an
  existing image than use the live camera picker.
- **Continuous local webcam loop** (`run_webcam_loop()` / the dashboard's
  "Enable continuous webcam trigger loop" toggle) — for a hands-free demo
  where cards are held up continuously without tapping capture each time.
  This one genuinely does need `streamlit run` on a laptop with a webcam
  attached; it has no camera access on Streamlit Cloud.

---

## 6. The statistics layer

`stats_engine.py` computes, from data the simulation itself generates:

- **Gini coefficient** on both wealth (savings) and income, every round
- **Before/after comparison with statistical significance** — mean of each
  metric before vs after a policy trigger, backed by:
  - **Welch's t-test** (unequal variances), implemented dependency-free on
    numpy/math (regularized incomplete beta function for the exact p-value —
    no scipy/statsmodels required)
  - a **bootstrap 95% CI** on the mean difference, as a distribution-free
    cross-check (useful given the small per-window sample sizes — a handful
    of rounds — where t-test normality assumptions are shakiest)
  - a `significant` flag requiring **both** p < 0.05 **and** the bootstrap CI
    to exclude zero, shown in the dashboard's before/after table
- **Price-elasticity-of-demand sanity check** — estimates
  `%Δ(real quantity) / %Δ(price)` around a shock and flags whether it falls
  inside a textbook-plausible range for the affected firm type. This is a
  check on the simulation's own pricing/demand logic, not a claim about the
  "true" elasticity of a toy economy — see §6.3 for what it caught.
- **Magnitude sensitivity sweep** — runs the same policy at several
  magnitudes (same seed, everything else fixed) and reports the resulting
  price change and elasticity for each. This is the tool that catches a
  policy **saturating** (all magnitudes producing an identical outcome) —
  see §6.3.
- **Multi-run consistency check** — run the same policy shock across several
  independent random seeds and report mean ± std of the final outcomes, so
  you can say *"this result is a stable emergent property, not a fluke of one
  run"* — the difference between a demo and a research tool.

### 6.1 Reading the significance columns

```
    metric   before    after  pct_change  p_value  ci_low  ci_high  significant
price_level   1.0389   1.0644        2.45   0.1453  0.0059   0.0532        False
      gini   0.4122   0.4807       16.63   0.0052  0.0477   0.0909         True
```
With only ~5 rounds per window, treat `significant=True` as a directional
signal worth narrating live, not a definitive causal claim — say so if asked
during a demo/defense.

### 6.2 Reading the elasticity check

```python
from policysim.stats_engine import price_elasticity_of_demand
price_elasticity_of_demand(df, trigger_round=5, window=4)
# {'elasticity': -2.27, 'pct_price_change': 3.2, 'pct_quantity_change': -7.3,
#  'textbook_range': (-1.2, -0.3), 'in_textbook_range': False}
```
An out-of-range result is a flag to *investigate* the pricing/demand
mechanism — not something to quietly tune away.

### 6.3 Real bugs this tooling caught (and fixed)

1. **`subsidy_cut` saturated at any magnitude ≥ the baseline rate.** It was
   originally an *absolute* percentage-point subtraction
   (`subsidy_rate - magnitude`), so a 10% baseline rate hit its floor of 0 at
   magnitude=10% and stayed there — meaning 20%, 30%, and 40% "cuts" were all
   *identical*. The magnitude sensitivity sweep surfaced this immediately
   (all four magnitudes produced the same final price level). Fixed by
   making the cut **multiplicative** (`subsidy_rate * (1 - magnitude)`), like
   every other policy in the catalogue — now each magnitude produces a
   distinguishable outcome (see `tests/test_policies.py`).
2. **Firms re-applied a flat `+0.05` price bump every single round** the
   `subsidy_cut` policy stayed active — not scaled by magnitude, and never
   decaying — which produced runaway-looking inflation regardless of the
   shock's actual size. Replaced with a proper **one-time cost pass-through**:
   `brain.py` now compares each firm's *effective* (post-subsidy) unit cost
   round-over-round and passes through 60% of any genuine change, exactly
   once, the round it happens (see `tests/test_brain_cost_passthrough.py`).
   Price now shifts once and stabilizes, instead of climbing every round the
   policy label happens to still be "active."
3. **Unemployment pinned at 0% in every run, making the cash_transfer
   experiment's unemployment column statistically meaningless.** The default
   firm spec offered more jobs (11) than households (10), so the employment
   margin never bound. Diagnosed by running the actual experiment and
   noticing a `nan`/degenerate result, not by code review — see §0.5 and
   `experiments/unemployment_slack.py`.
4. **A zero-variance metric produced a nonsensical `t-stat = -inf`,
   `Cohen's d = 0.000` sitting right next to `p = 0.0000`.** Income Gini is
   fully deterministic given tier/employment/policy when unemployment is
   pinned at 0%, so every one of 40 seeds produces the *exact same* value —
   a Welch's t-test on two zero-variance samples is mathematically
   degenerate, not "very significant." `run_experiment()` now detects this
   (both groups' variance below a calibrated floor) and reports the exact
   deterministic difference instead of a fabricated p-value (see
   `tests/test_experiments.py`).
5. **The fix for #4, on its first attempt, broke a *different*, legitimate
   result.** Flagging "degenerate" whenever *either* group had near-zero
   variance incorrectly suppressed `unemployment_slack.py`'s real finding
   (cash_transfer pushes every seed to exactly 0% unemployment, while the
   control group genuinely varies 0/10/20% across seeds — a normal,
   well-posed Welch's test, not a degenerate one). Corrected to require
   *both* groups near-zero variance, not either — caught by re-running the
   experiment after the first fix and noticing a real, previously-reported
   t-stat (t=-11.72) had silently vanished.
6. **`build_report.py` originally hardcoded every number in the PDF as a
   literal string**, disconnected from the code that actually produced
   them — meaning bugs #3-#5 above would have gone on being reported as
   fixed in the report while the report itself kept showing the old, wrong
   numbers forever. Rewritten to import and call the real experiment
   functions at report-generation time, so the PDF can never drift out of
   sync with the simulation code again.

---

## 7. Regression test suite

```bash
pip install pytest
pytest tests/ -v
```
52 tests across six files, covering:

| File | Covers |
|---|---|
| `test_policies.py` | Policy round-trip/revert correctness, the subsidy_cut multiplicative fix, QR payload parsing, card-color uniqueness |
| `test_brain_cost_passthrough.py` | The one-time cost pass-through fix — no bump when cost is unchanged, bounded bump on a real change, price stabilizing over repeated rounds, symmetric relief on a subsidy restoration |
| `test_stats_engine.py` | Gini edge cases, Welch's t-test against a known textbook case (t=9.0 exactly), bootstrap CI coverage, before/after significance flagging, elasticity sign correctness, the sensitivity-sweep saturation regression check |
| `test_simulation_integration.py` | Full round loop runs LLM-free, policy events land in the event log, same-seed determinism, multi-run harness, face-validity of a cash transfer raising low-income spending |
| `test_experiments.py` | The degenerate-variance detection fix (bugs #4 and #5 in §6.3) — confirms a truly deterministic metric is flagged, confirms a legitimately-varying-vs-deterministic pair is *not* falsely flagged, plus basic sanity checks on `run_once`/`cohens_d` |
| `test_trigger_system.py` | The camera color-card detector — each of the four card colors is correctly identified from a synthetic frame and maps to a real policy, a blank frame returns no detection, and a small background color fleck (below the area threshold) is correctly ignored rather than false-triggering a policy |

---

## 8. Calibrating the color-card detector to your venue

The built-in `HSV_RANGES` in `trigger_system.py` are tuned for standard
printer paper under normal indoor light — a different room (stage lighting,
a dim hall, a sunlit window) can make cards get missed or misclassified.
Rather than guessing new thresholds, tune them live against your actual
camera and lighting:

```bash
python -m policysim.calibrate_camera             # walks through all 4 colors
python -m policysim.calibrate_camera --color red # tune just one
```

Hold up each card, drag the HSV trackbars until the mask window shows a
clean white blob over the card and nothing else, press `s` to save (or `n`
to skip and keep the default). At the end it prints a ready-to-paste
`HSV_RANGES` dict and also writes it to
`assets/calibrated_hsv_ranges.json` — copy either into `trigger_system.py`
before a live demo in a new room. **Do this the day of the demo** if
possible; lighting is the single biggest cause of color-card misfires.

---

## 9. Live demo script (suggested flow)

1. Start the dashboard, click **(Re)start simulation**, step forward ~8
   rounds so the audience sees a stable baseline economy.
2. Hold up the **red card** (subsidy cut). Narrate: *"the government just
   cut fuel subsidies — watch what firms do."*
3. Step forward a few rounds. Point at the price-level chart ticking up and
   the live agent-thought feed explaining *why* ("input subsidies were cut,
   pushing our costs up").
4. Open the **before/after table** — real numbers, not vibes.
5. Show the **green card** (cash transfer) to demonstrate the opposite kind
   of shock, and how low-income spending responds differently than high
   income.

---

## 10. Design decisions worth knowing about

- **Market clearing** allocates household spending across firms weighted by
  inverse price (cheaper firms capture more demand) — a simple but genuine
  supply/demand mechanism, not scripted numbers.
- **Firm capacity is calibrated per firm** (`capacity_per_employee`) so
  utilization starts near equilibrium; price changes are clamped to ±10%
  per round so a single mis-calibrated round can't spiral into runaway
  inflation while still allowing sustained multi-round trends from a real
  policy shock.
- **Everything is agent-count agnostic** — add more households or firms via
  `household_spec` / `firm_spec` passed into `Simulation()`.

## 11. Extending it

- Add more policy cards by adding an entry to `POLICY_LIBRARY` in
  `policies.py` — the trigger system and dashboard pick it up automatically.
- Add more firms/households of different tiers/kinds via the `*_spec`
  arguments to `Simulation()`.
- The bootstrap CI in `before_after_comparison` defaults to 2000 resamples;
  bump `n_boot` if you want tighter CIs for a written report (at the cost of
  runtime), or pass a fixed `seed` for reproducible numbers in a paper.
- `magnitude_sensitivity_sweep` and `price_elasticity_of_demand` are plain
  functions independent of the dashboard — call them directly from a
  notebook for deeper analysis than the live demo needs.

---

## 12. Operations Research framing

Everything above describes the project in plain language. This section
states the same model formally — notation, the exact decision rules as
equations, the round loop as pseudocode, and one constrained-optimization
example built on top of it. Nothing here changes the model; it's the
same heuristic rules from `brain.py`/`simulation.py`/`agents.py`,
written down precisely instead of just described in prose.

### 12.1 Notation

| Symbol | Meaning |
|---|---|
| $H$ | set of households, indexed $i = 1, \dots, N$ |
| $F$ | set of firms, indexed $j = 1, \dots, K$ |
| $t$ | round index, $t = 1, 2, \dots$ |
| $\text{tier}(i)$ | $\in \{\text{low}, \text{mid}, \text{high}\}$ |
| $B_i$ | household $i$'s base income |
| $c_i$ | household $i$'s base consumption propensity (by tier) |
| $Y_i(t)$ | household $i$'s disposable income in round $t$ |
| $s_i(t)$ | household $i$'s spend fraction in round $t$ |
| $X_i(t)$ | household $i$'s spend, $= s_i(t)\,Y_i(t)$ |
| $\text{Sav}_i(t)$ | household $i$'s savings (wealth) after round $t$ |
| $e_i(t) \in \{0,1\}$ | household $i$'s employment status |
| $P_j(t)$ | firm $j$'s price |
| $W_j(t)$ | firm $j$'s wage |
| $E_j(t)$ | firm $j$'s headcount |
| $\kappa_j$ | firm $j$'s output capacity per employee |
| $U_j(t)$ | firm $j$'s capacity utilization |
| $\tau, \tau_{\text{lux}}$ | base tax rate, luxury-tax surcharge (high tier only) |
| $\sigma$ | firm subsidy rate on input costs |
| $m$ | a policy's magnitude (e.g. cash transfer size in ₹) |

### 12.2 Household decision rule

Disposable income (`agents.py::Household.disposable_income`):

$$Y_i(t) = \big[e_i(t)\, B_i + (1-e_i(t))\, 0.25 B_i\big]\,\big(1 - \tau - \tau_{\text{lux}}\mathbb{1}[\text{tier}(i)=\text{high}]\big) + T(\text{tier}(i))$$

where $T(\text{low}) = $ `cash_transfer_low`, $T(\text{mid}) = 0.4\times$ that value, $T(\text{high}) = 0$ (see `policies.py`'s `cash_transfer` policy).

Spend fraction (`brain.py::_heuristic_household`) — a bounded-rationality
rule, not a solved optimization, deliberately (see §12.5 for why this
matters for how to read the results):

$$s_i(t) = \text{clip}\Big(c_i + \Delta_i(t) + \varepsilon_i(t),\; 0.15,\; 0.98\Big), \qquad \varepsilon_i(t) \sim \mathcal{U}(-0.02, 0.02)$$

$\Delta_i(t)$ sums several additive terms activated by conditions (price
level vs. 1.0, employment status, active policies, tier) — see the code
for the exact eight terms and their citations (§0.5 already covers the
empirical grounding for the tier gradient and price sensitivity, so it's
not repeated here). Spend and savings update as:

$$X_i(t) = s_i(t)\,Y_i(t), \qquad \text{Sav}_i(t) = \text{Sav}_i(t-1) + Y_i(t) - X_i(t)$$

### 12.3 Firm decision rule and market clearing

Effective (post-subsidy) unit cost and capacity utilization:

$$\text{EffCost}_j(t) = \text{unit\_cost}_j\,(1-\sigma), \qquad U_j(t) = \frac{\text{Demand}_j(t)}{\max(E_j(t-1),1)\,\kappa_j}$$

Price update (`brain.py::_heuristic_firm`), clamped to a $\pm10\%$ move per round:

$$\Delta P_j(t) = \text{clip}\Big(\underbrace{\delta^{\text{demand}}_j}_{+0.06 \text{ if } U_j>1.1,\ -0.04 \text{ if } U_j<0.7} + \underbrace{\delta^{\text{wage}}_j}_{\text{wage-floor pass-through}} + \underbrace{0.6\cdot\frac{\text{EffCost}_j(t)-\text{EffCost}_j(t-1)}{\text{EffCost}_j(t-1)}}_{\text{one-time cost pass-through, rate }0.6},\; -0.10,\; 0.10\Big)$$

$$P_j(t) = \max\big(P_j(t-1)(1+\Delta P_j(t)),\; 1.05\,\text{EffCost}_j(t)\big)\cdot(1+\eta_j(t)), \qquad \eta_j(t)\sim\mathcal{U}(-0.01,0.01)$$

Employment update, with margin $\mu_j(t) = \frac{P_j(t)-\text{EffCost}_j(t)}{P_j(t)}$:

$$E_j(t) = \text{clip}\Big(E_j(t-1) + \Delta E_j(t),\; 0,\; E_j^{\max}\Big), \qquad \Delta E_j(t) = \begin{cases} +1 & U_j(t) > 1.15 \\ -1 & U_j(t) < 0.6 \ \text{or}\ \mu_j(t) < 0.05 \\ 0 & \text{otherwise} \end{cases}$$

Market clearing allocates total household spending across firms by
**inverse-price weighting** (cheaper firms capture more demand share) —
this is the whole model's "market", and it's intentionally this simple
rather than a solved general-equilibrium clearing price:

$$\Theta(t) = \sum_{i=1}^N X_i(t), \qquad w_j(t) = \frac{1}{\max(P_j(t), 0.01)}, \qquad \text{Demand}_j(t) = \frac{\Theta(t)\cdot w_j(t)/\sum_k w_k(t)}{P_j(t)}$$

Employment allocation is **index-order rationing**, not wage-based
matching — worth being explicit about, since it's a real simplification:
$\text{TotalJobs}(t) = \sum_j E_j(t)$, and households $1,\dots,\text{TotalJobs}(t)$
(by index, not by any merit/wage criterion) are marked employed. A more
realistic labor market would match on reservation wages; this project
uses simple rationing because the households don't currently have a
reservation-wage concept to match on.

### 12.4 Round-loop pseudocode

```
procedure RUN_ROUND(households, firms, government):
    for each firm j in firms:
        state_j <- firm_j.state(government)
        decision_j <- BRAIN.decide(state_j)          # heuristic / ollama / groq
        firm_j.apply(decision_j)                       # updates P_j, W_j, E_j

    price_level <- mean(P_j for j in firms) / base_price

    total_spend <- 0
    for each household i in households:
        state_i <- household_i.state(government, price_level)
        decision_i <- BRAIN.decide(state_i)
        household_i.apply(decision_i)                  # updates Sav_i, sets X_i
        total_spend <- total_spend + X_i

    for each firm j in firms:                           # market clearing
        w_j <- 1 / max(P_j, 0.01)
        demand_share_j <- total_spend * w_j / sum(w_k for k in firms)
        Demand_j <- demand_share_j / P_j

    total_jobs <- sum(E_j for j in firms)
    for i in 1..N:                                       # index-order rationing
        employed_i <- (i <= total_jobs)

    record {price_level, gini(savings), gini(incomes), unemployment_rate, total_spend}
    return round_stats
```

### 12.5 Reading the heuristic as bounded rationality, not optimization

Worth being explicit: $s_i(t)$ and the firm's pricing/hiring rule are
**heuristics that react to the current round's state**, not the solution
to a formal household utility-maximization or firm profit-maximization
problem. That's a deliberate modeling choice (a live demo needs agents
that visibly "reason" in plain language each round — see §3 — which is
easier to narrate for a heuristic rule than for the first-order
conditions of a solved optimization), but it means results here describe
**"what a population of boundedly-rational, reactive agents does"**, not
**"what a population of fully-rational optimizing agents would do."**
Section 0.5's findings (e.g. wealth Gini rising under a cash transfer)
are a property of *this* rule set, not a universal economic law — a
different, more sophisticated agent brain could plausibly show a
different result under the same policy.

### 12.6 A constrained optimization example

`experiments/optimize_cash_transfer.py` poses one concrete constrained
optimization problem on top of the simulation and solves it by grid
search (the objective is a black-box, stochastic output of the actual
agent-based simulation — no closed form, no gradient — so grid search
over the one-dimensional decision variable is the standard, appropriate
tool here, not a simplification taken for convenience):

$$\min_{m \,\geq\, 0} \quad \text{income\_gini}(m)$$
$$\text{s.t.} \quad \text{wealth\_gini}(m) - \text{wealth\_gini}(0) \;\leq\; \epsilon, \qquad \text{outlay}(m) \;\leq\; \text{Budget}$$

where $\text{outlay}(m) = m\,(N_{\text{low}} + 0.4\,N_{\text{mid}})\times(\text{active rounds})$
is the government's total spend on the transfer, matching the exact
`cash_transfer_low`/`cash_transfer_mid` update in `policies.py`.

**Why not just "minimize wealth Gini subject to budget"?** Because in
this model *any* transfer $m>0$ raises wealth Gini (§0.5) — that
objective alone always has the trivial corner solution $m^*=0$ (spend
nothing), which isn't an interesting constrained-optimization example.
Framing it as "get as much income-equality benefit as an inequality
tolerance allows, without blowing the budget" creates a genuine
trade-off, and is closer to how a real policymaker would actually pose
the question.

```bash
python -m experiments.optimize_cash_transfer --budget 3000000 --epsilon 0.03
```

A real run at those defaults (24-household population, see §0.5) finds
$m^*=4000$ (the grid's upper bound) — income Gini falls from 0.2216 to
0.1906, wealth Gini rises by only +0.0284 (within $\epsilon=0.03$),
costing ≈₹9.98 lakh in outlay, well under the ₹30 lakh budget. **Worth
noting honestly:** the epsilon constraint, not the budget, is what binds
here, and the optimum sits at the search grid's edge rather than a clean
interior point — because wealth Gini jumps immediately at any $m>0$ then
gently *recedes* as $m$ grows further (§0.5's magnitude sweep), the
tolerance constraint is infeasible at small-to-medium $m$ and becomes
feasible again at large $m$. Widen `--m_max` in the script if you want
to check whether an even larger transfer keeps improving income equality
within tolerance, or tighten `--epsilon` to see the trade-off bind
somewhere more interior.

