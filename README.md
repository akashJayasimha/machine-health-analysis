# Machine Health & Failure Pattern Analysis
*A concept analytical project built for DATATRONiQ GmbH*

---

## Overview

DATATRONiQ builds industrial IoT platforms that help manufacturers monitor machines in real time. The core challenge their customers face isn't data collection — it's the analytical layer between raw sensor readings and an actual maintenance decision.

This project demonstrates that analytical layer — a step-by-step machine health analysis in Python, built on synthetic sensor data that mirrors what a real manufacturing environment produces. The output is a self-contained HTML report a maintenance engineer could open and act on immediately.

---

## What the Analysis Does

The workflow runs in 5 deliberate steps:

**Step 1 — Data overview**
Load and understand the dataset: 540 sensor readings across 6 machines over 90 days. Fleet health score, failure counts, first-pass observations.

**Step 2 — Sensor trend analysis**
Identify which signals carry predictive weight. Temperature drift over time, vibration distributions, and the relationship between machine age and degradation.

**Step 3 — Failure patterns & downtime cost**
Separate failure *frequency* from failure *cost*. A machine that fails often but recovers quickly is a very different problem from one that fails rarely but takes 12 hours to repair.

**Step 4 — Risk classifier**
A transparent, rule-based classifier that labels each asset as `Critical`, `At Risk`, or `Stable`. Deliberately not a black-box model — each rule maps to something a maintenance engineer can validate and trust.

**Step 5 — Recommendations**
Four ranked actions tied directly back to the data. Analysis that doesn't end in a decision is just noise.

---

## Key Findings

| Finding | Value |
|---|---|
| Average fleet health | 70.7% *(below 75% target)* |
| Correlation: machine age → vibration | **r = 0.923** |
| Correlation: temperature → vibration | **r = 0.947** |
| Critical assets identified | MCH-004, MCH-006 |
| Total failure events (90 days) | 8 |

> The strongest insight: **machine age alone is a near-perfect predictor of vibration levels** (r = 0.923). This means age — a data point every facility already has — is an underused early-warning signal.

---

## Why Synthetic Data?

Real manufacturing sensor data is confidential. Synthetic data is standard practice in industrial analytics for exactly this reason — it lets you prototype and validate an analytical framework before deploying it to real equipment.

The synthetic dataset was designed to exhibit realistic patterns:
- Stable machines with low drift (MCH-001, MCH-003)
- Degrading machines with progressive temperature and vibration rise (MCH-004, MCH-006)
- Mid-range machines showing early warning signs (MCH-002, MCH-005)

---

## How to Run

**Prerequisites**
```bash
pip install numpy pandas matplotlib
```

**Run the analysis**
```bash
python3 analysis.py
```

**Output**
```
[1/5] Generating synthetic sensor dataset...
    → 540 rows  ·  6 machines  ·  8 failure events
[2/5] Computing fleet-level statistics...
    → avg fleet health : 70.7%
    → corr(age, vib)   : 0.923
    → corr(temp, vib)  : 0.947
[3/5] Generating charts...
    → 5 charts rendered
[4/5] Applying threshold-based risk classifier...
[5/5] Compiling HTML report...

✓ Report written to datatroniq_analysis.html
```

Open `datatroniq_analysis.html` in any browser. Fully self-contained — no server, no extra dependencies.

---

## Stack

- **Python 3** — pandas, NumPy, matplotlib
- **Output** — standalone HTML report (charts embedded as base64 images)
- **Runtime** — ~3 seconds

---

## Repository Structure

```
├── analysis.py                  # Main script — runs the full analysis
├── datatroniq_analysis.html     # Pre-generated report (open directly in browser)
└── README.md                    # This file
```

---

## What I'd Do Differently with Real Data

- **Failure type labelling** — distinguishing mechanical from electrical failures would sharpen the classifier significantly
- **Rolling window analysis** — a 7-day rolling average would surface trends much earlier than 90-day means
- **Actual MTTR data** — real maintenance logs would make the downtime cost analysis concrete rather than estimated
- **Anomaly detection layer** — a z-score or IQR-based flagging system on top of the threshold classifier would catch edge cases

---

## About This Project

Built as part of an unsolicited application to DATATRONiQ GmbH for an analyst role. Rather than write a cover letter, I wanted to show how I think — starting from a business problem, designing an analysis around it, and delivering something immediately usable.

I used AI tools to accelerate the build. The analytical framework — what to investigate, why a transparent classifier beats a complex model here, how to make findings legible for non-technical readers — came from my background in data analytics.

---

*Built by Akash Jayasimha · February 2025*
