from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, PageBreak, HRFlowable)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#2a4d69")))
styles.add(ParagraphStyle(name="Bodyc", parent=styles["BodyText"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, textColor=colors.grey, leading=11))
styles.add(ParagraphStyle(name="TitleC", parent=styles["Title"], spaceAfter=4))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=24))

story = []

def h1(t): story.append(Paragraph(t, styles["H1c"]))
def h2(t): story.append(Paragraph(t, styles["H2c"]))
def p(t): story.append(Paragraph(t, styles["Bodyc"]))
def small(t): story.append(Paragraph(t, styles["Small"]))
def sp(h=10): story.append(Spacer(1, h))
def rule(): story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cccccc"), spaceBefore=6, spaceAfter=10))

def make_table(header, rows, col_widths=None):
    data = [header] + rows
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a4d69")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f7fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

# ---------------------------------------------------------------- TITLE
story.append(Spacer(1, 60))
story.append(Paragraph("PolicySim", styles["TitleC"]))
story.append(Paragraph("Does a Cash Transfer Policy Actually Work?", styles["Subtitle"]))
story.append(Paragraph(
    "A statistical evaluation of a generative agent-based economic simulation<br/>"
    "MSc Economics portfolio project &mdash; results report",
    ParagraphStyle(name="sub2", parent=styles["Normal"], alignment=1, fontSize=10, textColor=colors.grey)
))
sp(40)
rule()

# ---------------------------------------------------------------- 1. OVERVIEW
h1("1. What this is")
p("PolicySim is a small agent-based economy (households, firms, a government) that reasons in "
  "plain language each round and reacts to policy shocks a user can trigger live. This report "
  "focuses on one question the project set out to answer with real statistics rather than "
  "narrated vibes: <b>does a flat cash transfer to low/mid-income households actually change the "
  "simulated economy, and if so, how?</b>")
p("All numbers below come from running the real simulation engine end-to-end (heuristic reasoning "
  "backend, no LLM needed), not from a mock or a pre-written table. Every experiment can be "
  "reproduced with <font face='Courier'>python -m experiments.&lt;name&gt;</font>.")

h2("1.1 Method")
p("40 independent simulation runs per condition (different random seed each run), 20 rounds per "
  "run, cash_transfer applied from round 1 in the treatment group and never in the control group. "
  "Final-round outcomes are compared with Welch's t-test (unequal variances) and Cohen's d for "
  "effect size.")

sp()
rule()

# ---------------------------------------------------------------- 2. HEADLINE RESULT
h1("2. Headline result: wealth inequality up, income inequality down")
p("The core, counter-intuitive finding: a cash transfer that directly boosts low/mid-income "
  "households' <i>income</i> makes income inequality better, but makes wealth (savings) "
  "inequality <i>worse</i>.")

make_table(
    ["Metric", "Control mean", "Treatment mean", "t-stat", "p-value", "Cohen's d"],
    [
        ["Wealth Gini", "0.4948", "0.5309", "18.447", "<0.0001", "4.13"],
        ["Income Gini", "0.3254", "0.3169", "\u2013", "<0.0001", "\u2013"],
        ["Price level", "1.0615", "1.0700", "2.159", "0.034", "0.48"],
        ["Total spending", "261,363.9", "278,934.6", "17.577", "<0.0001", "3.93"],
        ["Unemployment", "0.0%", "0.0%", "n/a", "n/a", "0 (pinned, see \u00a74)"],
    ],
    col_widths=[95, 80, 90, 55, 60, 65]
)
sp()
p("<b>Why:</b> the transfer is a fixed rupee amount, small relative to high-tier households' base "
  "income. High earners keep compounding savings at their usual pace. Low/mid households spend a "
  "larger share of their (now slightly bigger) income rather than saving it, so nominal wealth "
  "inequality widens even though the transfer helps low/mid <i>income</i> directly &mdash; visible "
  "in income Gini falling in the same runs.")

sp(6)
story.append(Image("assets/charts/wealth_gini_over_time.png", width=5.4*inch, height=3.47*inch))
sp(4)
story.append(Image("assets/charts/income_gini_over_time.png", width=5.4*inch, height=3.47*inch))
small("Mean &plusmn; 1 std across 40 runs per condition, round by round. The two Gini measures "
      "diverge almost immediately after the transfer starts and stay apart for the full 20 rounds "
      "&mdash; this is a stable trend, not a single lucky snapshot.")

story.append(PageBreak())

# ---------------------------------------------------------------- 3. MAGNITUDE SWEEP
h1("3. Does a bigger transfer make wealth inequality worse?")
p("Not linearly. Averaging 15 seeds per magnitude removes enough noise to see the actual shape: "
  "wealth Gini jumps as soon as <i>any</i> transfer is applied, then slightly recedes as the "
  "transfer keeps growing &mdash; a threshold effect with a mild ceiling, not a dial that keeps "
  "making things worse the harder you push it.")

make_table(
    ["Magnitude", "Wealth Gini", "Income Gini", "Price level"],
    [
        ["0 (none)", "0.4974", "0.3254", "1.0622"],
        ["250", "0.5282", "0.3232", "1.0663"],
        ["500", "0.5275", "0.3211", "1.0705"],
        ["1000 (default)", "0.5265", "0.3169", "1.0705"],
        ["2000", "0.5225", "0.3087", "1.0886"],
        ["4000", "0.5152", "0.2927", "1.1097"],
    ],
    col_widths=[90, 90, 90, 90]
)
sp()
p("Income Gini, in contrast, falls smoothly and monotonically as the transfer grows &mdash; "
  "exactly what you'd expect from a transfer targeted at low/mid earners. "
  "<b>Takeaway:</b> don't read the wealth-Gini result as \u201cbigger transfer = worse for wealth "
  "inequality\u201d; it's closer to \u201cany transfer at all widens wealth inequality at this "
  "population size, with a ceiling on how much.\u201d")

sp()
rule()

# ---------------------------------------------------------------- 4. UNEMPLOYMENT / LABOR SLACK
h1("4. Unemployment: a genuine model-exposed limitation, tested and confirmed")
p("At the default firm settings (11 jobs available vs. 10 households), labor demand already "
  "exceeds the household count before any policy runs &mdash; so the employment margin never "
  "binds, and unemployment sits at exactly 0.0% in every run regardless of policy. That's a real "
  "limitation of the default calibration, not a broken policy effect.")
p("To confirm this diagnosis rather than just assert it, the same control-vs-treatment design was "
  "re-run with fewer starting jobs (7 vs. 10 households) and lower per-employee capacity, so there "
  "is genuine slack for a demand shock to close:")

make_table(
    ["Metric", "Control mean", "Treatment mean", "t-stat", "p-value", "Cohen's d"],
    [
        ["Unemployment rate", "11.25%", "0.00%", "\u201311.72", "<0.0001", "\u20132.62"],
        ["Wealth Gini", "0.4066", "0.4945", "13.545", "<0.0001", "3.03"],
        ["Income Gini", "0.2630", "0.3169", "6.440", "<0.0001", "1.44"],
        ["Price level", "1.1800", "1.2352", "8.371", "<0.0001", "1.87"],
    ],
    col_widths=[95, 80, 90, 65, 65, 65]
)
sp()
p("With real slack in the labor market, the cash transfer's demand boost measurably closes it "
  "(unemployment falls from 11.25% to 0.00%, p&lt;0.0001) &mdash; confirming that the flat 0% "
  "unemployment in the default configuration was a labor-supply ceiling, not evidence the policy "
  "has no effect on jobs.")

sp()
rule()

# ---------------------------------------------------------------- 5. ROBUSTNESS
h1("5. Robustness check: does the headline result survive a different population size?")
p("The wealth-Gini finding was re-tested with double the household count (20 households instead "
  "of 10, same firms, same design) to check it isn't an artifact of a specific small population.")

make_table(
    ["Metric", "Control mean", "Treatment mean", "p-value", "Cohen's d"],
    [
        ["Wealth Gini (baseline, n=10 hh)", "0.4948", "0.5309", "<0.0001", "4.13"],
        ["Wealth Gini (2x households, n=20 hh)", "0.4036", "0.4393", "<0.0001", "3.74"],
    ],
    col_widths=[190, 90, 55, 55]
)
sp()
p("The direction and significance of the effect held under 2x population (effect size similar: "
  "d=3.74 vs. d=4.13). Absolute Gini levels differ because a larger, evenly-tiered population "
  "changes the wealth distribution's baseline shape, but the treatment-vs-control gap is the same "
  "story: this is a stable emergent property of the mechanism, not a fluke of one specific "
  "population size.")

sp()
rule()

# ---------------------------------------------------------------- 6. LITERATURE
h1("6. Why this matches real economics")
p("The mechanism behind the divergence &mdash; poorer households spend a larger share of "
  "additional income while richer households save more of theirs &mdash; mirrors well-documented "
  "differences in marginal propensity to consume (MPC) by income/wealth level. PSID-based "
  "estimates put MPC around 0.15 for the lowest wealth quintile vs. roughly 0.06 for the highest "
  "(Fisher, Johnson, Smeeding &amp; Thompson); recession-calibrated estimates from the Penn "
  "Wharton Budget Model show a similar gradient by income quintile (roughly 0.55 down to 0.12). "
  "PolicySim's heuristic household brain is loosely tuned to that gradient (see "
  "<font face='Courier'>brain.py</font> and <font face='Courier'>simulation.py</font>), so the "
  "wealth-Gini-up / income-Gini-down split isn't an arbitrary quirk of this toy model &mdash; it's "
  "the same distributional mechanic that shows up in real transfer-payment research, reproduced "
  "here in miniature.")

sp()
rule()

# ---------------------------------------------------------------- 7. LIMITATIONS
h1("7. Limitations")
p("&bull; <b>Population size.</b> 10&ndash;20 households and 2 firms is a toy economy. Directional "
  "results are consistent under a 2x robustness check, but this is not a claim about real-world "
  "magnitudes.")
p("&bull; <b>Heuristic reasoning only.</b> All results here use the zero-dependency heuristic "
  "brain, not an LLM backend (ollama/groq); those backends introduce their own variance not "
  "characterized here.")
p("&bull; <b>Single policy studied in depth.</b> The subsidy cut, minimum wage, and luxury tax "
  "policies exist and are unit-tested, but this report focuses statistical depth on cash_transfer "
  "specifically.")
p("&bull; <b>20-round horizon.</b> Longer-horizon dynamics (e.g. compounding wealth effects over "
  "100+ rounds) are untested here.")
p("&bull; <b>Multiple comparisons.</b> Five metrics are reported per experiment; treat each "
  "p-value as part of a larger comparison set rather than a single pre-registered test.")

sp()
rule()

# ---------------------------------------------------------------- 8. REPRODUCIBILITY
h1("8. Reproducing these results")
p("<font face='Courier' size=8.5>"
  "python -m experiments.cash_transfer_effect &nbsp;&nbsp;# &sect;2 and &sect;5<br/>"
  "python -m experiments.magnitude_sweep_cash_transfer &nbsp;&nbsp;# &sect;3<br/>"
  "python -m experiments.unemployment_slack &nbsp;&nbsp;# &sect;4<br/>"
  "python -m experiments.generate_charts &nbsp;&nbsp;# regenerates the charts above<br/>"
  "pytest tests/ -v &nbsp;&nbsp;# 38-test regression suite"
  "</font>")

sp(20)
small("Generated from the PolicySim repository. See README.md for the full project writeup, "
      "the policy catalogue, and the live-demo trigger system.")

doc = SimpleDocTemplate("policysim_results_report.pdf", pagesize=letter,
                         topMargin=54, bottomMargin=54, leftMargin=56, rightMargin=56,
                         title="PolicySim Results Report")
doc.build(story)
print("wrote policysim_results_report.pdf")
