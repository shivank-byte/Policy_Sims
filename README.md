# PolicySim
### Generative Agent-Based Economic Simulation with Live Policy Triggers

[![tests](https://github.com/USERNAME/policysim/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/policysim/actions/workflows/tests.yml)

> Replace `USERNAME` above with your GitHub username/org once this is pushed to a repo — the badge is wired to `.github/workflows/tests.yml`, which runs the 38-test regression suite on every push/PR.


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

---

## 0.5 Results — does the cash transfer policy actually do anything?

**Full writeup with charts: [`policysim_results_report.pdf`](policysim_results_report.pdf).**
This section is the short version.

`experiments/cash_transfer_effect.py` runs the real engine 40 times with
`cash_transfer` off and 40 times with it on from round 1, for 20 rounds
each, and compares round-20 outcomes with Welch's t-test:

```bash
python -m experiments.cash_transfer_effect
```

```
metric                control mean  treatment mean    t-stat   p-value   Cohen d
--------------------------------------------------------------------------------
gini                        0.4948          0.5309    18.447    0.0000     4.125
gini_income                 0.3254          0.3169      -inf    0.0000     0.000
unemployment_rate           0.0000          0.0000       nan       nan     0.000
price_level                 1.0615          1.0700     2.159    0.0340     0.483
total_spending         261363.8650     278934.6375    17.577    0.0000     3.930
```

**Total spending and price level both rise significantly (p < 0.05)** —
expected: more disposable income for low/mid households means more
demand, which pushes prices up under the market-clearing rule.

**Wealth Gini rises significantly (Cohen's d = 4.1) while income Gini
falls** — this is the headline, counter-intuitive result, and it's now
tested directly rather than just asserted: a flat cash transfer *widens*
wealth (savings) inequality even though it *narrows* income inequality in
the same runs. The transfer is a fixed rupee amount, small relative to
high-tier households' base income, so higher-income households keep
compounding savings at their usual pace while low/mid tiers spend a
larger share of their (now slightly bigger) income rather than saving
it. See `assets/charts/wealth_gini_over_time.png` and
`income_gini_over_time.png` for the full 20-round trend (not just the
before/after snapshot) — the two measures diverge almost immediately and
stay apart.

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
 magnitude   wealth gini   (std)   income gini   price level
------------------------------------------------------------
         0        0.4974  0.0093        0.3254        1.0622
       250        0.5282  0.0094        0.3232        1.0663
       500        0.5275  0.0095        0.3211        1.0705
      1000        0.5265  0.0095        0.3169        1.0705
      2000        0.5225  0.0099        0.3087        1.0886
      4000        0.5152  0.0093        0.2927        1.1097
```

**Unemployment shows exactly zero variance in the default setup** — this
is a genuine model-exposed limitation, not a bug: at the default firm
settings (`employees=3` and `8`, total 11 jobs vs. 10 households),
labor demand already exceeds the household count before any policy
runs, so the employment margin never binds.

That diagnosis is now confirmed rather than just claimed:
`experiments/unemployment_slack.py` reruns the same design with fewer
starting jobs *and* lower per-employee capacity (`employees=2` and `5`,
lower `capacity_per_employee`, so utilization can actually cross the
firm brain's hiring threshold) — with real slack in the labor market, the
cash transfer's demand boost measurably closes it:

```bash
python -m experiments.unemployment_slack
```

```
metric                control mean  treatment mean    t-stat   p-value   Cohen d
--------------------------------------------------------------------------------
unemployment_rate           0.1125          0.0000   -11.720    0.0000    -2.621
gini                        0.4066          0.4945    13.545    0.0000     3.029
gini_income                 0.2630          0.3169     6.440    0.0000     1.440
price_level                 1.1800          1.2352     8.371    0.0000     1.872
```

**Robustness check:** does the wealth-Gini finding survive a different
population size? `run_experiment()` in `cash_transfer_effect.py` was
re-run with double the household count (20 instead of 10) — direction
and significance held (d=3.74 vs. baseline d=4.13), so this looks like a
stable emergent property of the mechanism rather than a fluke of one
specific population size.

**Regenerating the charts:**
```bash
python -m experiments.generate_charts       # writes assets/charts/*.png
python build_report.py                       # regenerates policysim_results_report.pdf
```

**Caveats:** this is a 10–20 household / 2-firm toy economy on a 20-round
horizon with the heuristic reasoning backend only — see §7 of the PDF
report for the full limitations list (population size, single-backend
results, multiple-comparisons across 5 metrics per experiment, etc).

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
The regression suite (§7) locks in two real bugs found and fixed during
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

### 6.3 Two real bugs this tooling caught (and fixed)

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

---

## 7. Regression test suite

```bash
pip install pytest
pytest tests/ -v
```
38 tests across four files, covering:

| File | Covers |
|---|---|
| `test_policies.py` | Policy round-trip/revert correctness, the subsidy_cut multiplicative fix, QR payload parsing, card-color uniqueness |
| `test_brain_cost_passthrough.py` | The one-time cost pass-through fix — no bump when cost is unchanged, bounded bump on a real change, price stabilizing over repeated rounds, symmetric relief on a subsidy restoration |
| `test_stats_engine.py` | Gini edge cases, Welch's t-test against a known textbook case (t=9.0 exactly), bootstrap CI coverage, before/after significance flagging, elasticity sign correctness, the sensitivity-sweep saturation regression check |
| `test_simulation_integration.py` | Full round loop runs LLM-free, policy events land in the event log, same-seed determinism, multi-run harness, face-validity of a cash transfer raising low-income spending |

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
