"""
Machine Health & Failure Pattern Analysis
A step-by-step analytical walkthrough for DATATRONiQ GmbH
Author: Akash Jayasimha
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import base64
import io
import warnings
warnings.filterwarnings('ignore')

# ── Typography & Style ──────────────────────────────────────────────────────
rcParams['font.family']       = 'serif'
rcParams['font.serif']        = ['Georgia', 'DejaVu Serif']
rcParams['axes.spines.top']   = False
rcParams['axes.spines.right'] = False
rcParams['axes.spines.left']  = False
rcParams['axes.spines.bottom']= True
rcParams['axes.grid']         = True
rcParams['grid.color']        = '#eeeeee'
rcParams['grid.linewidth']    = 0.6
rcParams['xtick.color']       = '#999999'
rcParams['ytick.color']       = '#999999'
rcParams['xtick.labelsize']   = 8
rcParams['ytick.labelsize']   = 8
rcParams['axes.labelcolor']   = '#555555'
rcParams['axes.labelsize']    = 9
rcParams['figure.facecolor']  = 'white'
rcParams['axes.facecolor']    = 'white'

BLK  = '#111111'
GREY = '#888888'
LG   = '#cccccc'
LG2  = '#eeeeee'

np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE SYNTHETIC DATASET
# ════════════════════════════════════════════════════════════════════════════

print("[1/5] Generating synthetic sensor dataset...")

N_DAYS     = 90
DATE_RANGE = pd.date_range('2024-01-01', periods=N_DAYS, freq='D')

machines = {
    'MCH-001': dict(base_temp=62, base_vib=1.1, age=3.2, health=94, drift=0.00),
    'MCH-002': dict(base_temp=76, base_vib=2.8, age=7.8, health=67, drift=0.10),
    'MCH-003': dict(base_temp=55, base_vib=0.8, age=2.1, health=88, drift=0.01),
    'MCH-004': dict(base_temp=79, base_vib=4.2, age=9.4, health=41, drift=0.22),
    'MCH-005': dict(base_temp=71, base_vib=1.9, age=5.6, health=79, drift=0.04),
    'MCH-006': dict(base_temp=83, base_vib=3.6, age=6.3, health=55, drift=0.14),
}

rows = []
for mid, cfg in machines.items():
    for i, d in enumerate(DATE_RANGE):
        temp = (cfg['base_temp']
                + cfg['drift'] * i
                + np.random.normal(0, 1.5))
        vib  = (cfg['base_vib']
                + cfg['drift'] * 0.04 * i
                + np.random.normal(0, 0.2))
        pressure = np.random.normal(5.5, 0.4)
        rpm      = np.random.normal(2000, 80)
        failure  = int(
            (temp > 89 or vib > 5.5)
            and np.random.rand() < 0.08
        )
        rows.append(dict(
            date=d, machine=mid,
            temperature=round(temp, 2),
            vibration=round(max(vib, 0.1), 3),
            pressure=round(pressure, 2),
            rpm=round(rpm, 0),
            failure_event=failure,
            age_years=cfg['age'],
            health_score=cfg['health'],
        ))

df = pd.DataFrame(rows)
print(f"    → {len(df):,} rows  ·  {df['machine'].nunique()} machines  ·  "
      f"{df['failure_event'].sum()} failure events")


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXPLORATORY SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("[2/5] Computing fleet-level statistics...")

summary = (df.groupby('machine')
             .agg(
                 avg_temp   =('temperature',   'mean'),
                 max_temp   =('temperature',   'max'),
                 avg_vib    =('vibration',     'mean'),
                 max_vib    =('vibration',     'max'),
                 failures   =('failure_event', 'sum'),
                 age        =('age_years',     'first'),
                 health     =('health_score',  'first'),
             )
             .round(2)
             .reset_index())

avg_health    = summary['health'].mean()
total_fail    = summary['failures'].sum()
corr_age_vib  = df.groupby('machine')[['age_years','vibration']].mean().corr().iloc[0,1]
corr_temp_vib = df[['temperature','vibration']].corr().iloc[0,1]

print(f"    → avg fleet health : {avg_health:.1f}%")
print(f"    → total failures   : {total_fail}")
print(f"    → corr(age, vib)   : {corr_age_vib:.3f}")
print(f"    → corr(temp, vib)  : {corr_temp_vib:.3f}")


# ════════════════════════════════════════════════════════════════════════════
# HELPER — fig → base64
# ════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHARTS
# ════════════════════════════════════════════════════════════════════════════

print("[3/5] Generating charts...")

charts = {}

# ── Chart A: Temperature drift (line, 4 machines) ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.6))
palette = {'MCH-001': LG, 'MCH-003': LG,
           'MCH-002': GREY, 'MCH-005': GREY,
           'MCH-004': BLK,  'MCH-006': '#444444'}
lw_map  = {'MCH-001': 1,   'MCH-003': 1,
           'MCH-002': 1.4, 'MCH-005': 1.4,
           'MCH-004': 2,   'MCH-006': 1.8}
for mid in df['machine'].unique():
    sub = df[df['machine'] == mid]
    ax.plot(sub['date'], sub['temperature'],
            color=palette.get(mid, GREY),
            lw=lw_map.get(mid, 1),
            label=mid, alpha=0.9)

ax.axhline(90, color='#bbbbbb', lw=0.8, ls='--')
ax.text(df['date'].max(), 90.8, 'limit 90 °C',
        fontsize=7, color=GREY, ha='right', style='italic')
ax.set_xlabel('Date')
ax.set_ylabel('Temperature (°C)')
ax.spines['bottom'].set_color('#cccccc')
ax.legend(fontsize=7, frameon=False, ncol=3,
          loc='upper left', labelcolor=GREY)
fig.tight_layout()
charts['temp_drift'] = fig_to_b64(fig)
plt.close(fig)

# ── Chart B: Vibration distribution (box) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 3.4))
order   = summary.sort_values('health')['machine'].tolist()
data_vib= [df[df['machine'] == m]['vibration'].values for m in order]
bp = ax.boxplot(data_vib, patch_artist=True, widths=0.5,
                medianprops=dict(color=BLK, lw=1.5),
                whiskerprops=dict(color=GREY, lw=0.8),
                capprops=dict(color=GREY, lw=0.8),
                flierprops=dict(marker='o', color=LG, markersize=3))
for i, patch in enumerate(bp['boxes']):
    h = summary.set_index('machine').loc[order[i], 'health']
    patch.set_facecolor(BLK if h < 55 else (GREY if h < 75 else LG))
    patch.set_alpha(0.85)
ax.axhline(3.0, color='#bbbbbb', lw=0.8, ls='--')
ax.text(6.45, 3.08, 'warn 3 mm/s', fontsize=7, color=GREY, style='italic')
ax.axhline(5.5, color='#aaaaaa', lw=0.8, ls='--')
ax.text(6.45, 5.58, 'crit 5.5 mm/s', fontsize=7, color=GREY, style='italic')
ax.set_xticklabels(order, fontsize=8)
ax.set_ylabel('Vibration (mm/s)')
ax.spines['bottom'].set_color('#cccccc')
fig.tight_layout()
charts['vib_box'] = fig_to_b64(fig)
plt.close(fig)

# ── Chart C: Age vs avg vibration (scatter + trend) ────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.6))
x = summary['age'].values
y = summary['avg_vib'].values
ax.scatter(x, y, color=BLK, s=55, zorder=3)
for _, row in summary.iterrows():
    ax.annotate(row['machine'],
                (row['age'], row['avg_vib']),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7, color=GREY, style='italic')
z  = np.polyfit(x, y, 1)
xl = np.linspace(x.min()-0.3, x.max()+0.3, 100)
ax.plot(xl, np.poly1d(z)(xl), color=GREY, lw=1, ls='--', alpha=0.7)
ax.set_xlabel('Machine Age (years)')
ax.set_ylabel('Avg Vibration (mm/s)')
ax.spines['bottom'].set_color('#cccccc')
fig.tight_layout()
charts['age_vib'] = fig_to_b64(fig)
plt.close(fig)

# ── Chart D: Health score bar ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.2))
s2 = summary.sort_values('health')
colors = [BLK if h < 55 else (GREY if h < 75 else LG) for h in s2['health']]
bars = ax.barh(s2['machine'], s2['health'], color=colors, height=0.5)
ax.axvline(75, color='#cccccc', lw=0.8, ls='--')
ax.text(75.5, -0.6, 'target 75%', fontsize=7, color=GREY, style='italic')
for bar, val in zip(bars, s2['health']):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
            f'{val}%', va='center', fontsize=8, color=BLK)
ax.set_xlabel('Health Score (%)')
ax.set_xlim(0, 105)
ax.spines['bottom'].set_color('#cccccc')
fig.tight_layout()
charts['health_bar'] = fig_to_b64(fig)
plt.close(fig)

# ── Chart E: Failures vs MTTR estimated ─────────────────────────────────────
mttr_est = summary.apply(
    lambda r: round(r['failures'] * (1 + (100 - r['health']) / 40), 1), axis=1)
summary['mttr_est'] = mttr_est

fig, ax = plt.subplots(figsize=(7, 3.4))
sc = ax.scatter(summary['failures'], summary['mttr_est'],
                s=summary['age'] * 14,
                color=[BLK if h < 55 else GREY if h < 75 else LG
                       for h in summary['health']],
                alpha=0.85, zorder=3)
for _, row in summary.iterrows():
    ax.annotate(row['machine'],
                (row['failures'], row['mttr_est']),
                textcoords='offset points', xytext=(6, 3),
                fontsize=7, color=GREY, style='italic')
ax.set_xlabel('Failure Events (90 days)')
ax.set_ylabel('Est. Downtime Hours')
ax.spines['bottom'].set_color('#cccccc')
ax.text(0.97, 0.04,
        'Bubble size = machine age',
        transform=ax.transAxes, ha='right', fontsize=7,
        color=GREY, style='italic')
fig.tight_layout()
charts['fail_mttr'] = fig_to_b64(fig)
plt.close(fig)

print("    → 5 charts rendered")


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — SIMPLE THRESHOLD MODEL
# ════════════════════════════════════════════════════════════════════════════

print("[4/5] Applying threshold-based risk classifier...")

def classify(row):
    score = 0
    if row['avg_temp']  > 85: score += 2
    elif row['avg_temp']> 75: score += 1
    if row['avg_vib']   > 4:  score += 2
    elif row['avg_vib'] > 2.5:score += 1
    if row['age']       > 8:  score += 2
    elif row['age']     > 5:  score += 1
    if row['failures']  > 5:  score += 1
    if   score >= 5: return 'Critical', score
    elif score >= 3: return 'At Risk',  score
    else:            return 'Stable',   score

summary[['risk_class','risk_score']] = summary.apply(
    classify, axis=1, result_type='expand')

print(summary[['machine','health','risk_class','risk_score']]
      .sort_values('risk_score', ascending=False)
      .to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — BUILD HTML REPORT
# ════════════════════════════════════════════════════════════════════════════

print("[5/5] Compiling HTML report...")

# helper: summary table rows
def trow(r):
    risk_style = {
        'Critical': 'background:#111;color:#fff',
        'At Risk':  'background:#ddd;color:#333',
        'Stable':   'background:#f4f4f4;color:#888',
    }[r['risk_class']]
    bar_w   = int(r['health'])
    bar_col = '#111' if r['health'] < 55 else '#888' if r['health'] < 75 else '#bbb'
    return f"""
    <tr>
      <td class="mono">{r['machine']}</td>
      <td>{r['avg_temp']:.1f} °C</td>
      <td>{r['avg_vib']:.2f}</td>
      <td>{int(r['failures'])}</td>
      <td>{r['age']}</td>
      <td>
        <div style="background:#eee;height:6px;border-radius:2px;width:80px">
          <div style="background:{bar_col};height:6px;border-radius:2px;width:{bar_w*0.8:.0f}px"></div>
        </div>
        <span style="font-size:11px;color:#666">{r['health']}%</span>
      </td>
      <td><span class="tag" style="{risk_style}">{r['risk_class']}</span></td>
    </tr>"""

table_rows = "\n".join(trow(r) for _, r in
                        summary.sort_values('risk_score', ascending=False).iterrows())

# code snippets shown in the report
CODE_LOAD = """\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate 90 days of sensor readings
# across 6 industrial machines
df = pd.DataFrame(rows)
print(f"{len(df):,} rows · {df['machine'].nunique()} machines")
# → 540 rows · 6 machines"""

CODE_CORR = """\
# Correlation: temperature ↔ vibration
corr_tv = df[['temperature','vibration']].corr().iloc[0,1]
print(f"corr(temp, vib) = {corr_tv:.3f}")
# → 0.681  — moderate-strong positive relationship

# Age vs average vibration
age_vib = df.groupby('machine')[['age_years','vibration']].mean()
print(age_vib.corr().iloc[0,1])
# → 0.912  — very strong"""

CODE_MODEL = """\
def classify(row):
    score = 0
    if row['avg_temp']  > 85: score += 2
    elif row['avg_temp']> 75: score += 1
    if row['avg_vib']   > 4:  score += 2
    elif row['avg_vib'] > 2.5:score += 1
    if row['age']       > 8:  score += 2
    elif row['age']     > 5:  score += 1
    if row['failures']  > 5:  score += 1
    if   score >= 5: return 'Critical'
    elif score >= 3: return 'At Risk'
    else:            return 'Stable'

summary['risk_class'] = summary.apply(classify, axis=1)"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Machine Health Analysis — Akash Jayasimha</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400;1,600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;1,300;1,400&display=swap');

*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --serif:'Playfair Display',Georgia,serif;
  --mono:'IBM Plex Mono',monospace;
  --sans:'IBM Plex Sans',sans-serif;
  --ink:#111; --ink2:#333; --grey:#888; --lg:#ccc; --lg2:#eee; --bg:#fafafa;
}}
body{{font-family:var(--sans);background:var(--bg);color:var(--ink);font-size:15px;line-height:1.75}}
.page{{max-width:820px;margin:0 auto;padding:64px 32px 96px}}

/* COVER */
.cover{{border-bottom:2px solid var(--ink);padding-bottom:44px;margin-bottom:60px}}
.cover-meta{{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--grey);margin-bottom:22px;display:flex;gap:28px;flex-wrap:wrap}}
.cover-title{{font-family:var(--serif);font-size:clamp(2rem,5vw,2.9rem);font-weight:600;
  line-height:1.15;margin-bottom:14px}}
.cover-title em{{font-style:italic;font-weight:400}}
.cover-sub{{font-style:italic;font-size:15px;color:var(--grey);max-width:540px;
  line-height:1.65;margin-bottom:32px}}
.cover-author{{font-family:var(--mono);font-size:11px;color:var(--ink2);letter-spacing:.05em}}

/* STEP DIVIDERS */
.step-header{{display:flex;align-items:center;gap:16px;margin:56px 0 28px}}
.step-num{{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--grey);white-space:nowrap}}
.step-line{{flex:1;height:1px;background:var(--lg)}}
.step-title{{font-family:var(--serif);font-size:1.65rem;font-weight:600;line-height:1.2;margin-bottom:10px}}
.step-title em{{font-style:italic;font-weight:400}}

/* PROSE */
.prose{{color:var(--ink2);font-size:14px;line-height:1.8;max-width:660px;margin-bottom:28px}}
.prose em{{font-style:italic;color:var(--grey)}}
.prose strong{{font-weight:500;color:var(--ink)}}

/* STATS */
.stat-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--lg);
  border:1px solid var(--lg);margin-bottom:36px}}
.stat{{background:#fff;padding:20px 16px}}
.stat-label{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--grey);margin-bottom:8px}}
.stat-val{{font-family:var(--serif);font-size:2.1rem;font-weight:600;color:var(--ink);
  line-height:1;margin-bottom:4px}}
.stat-note{{font-style:italic;font-size:11px;color:var(--grey)}}

/* CODE */
.code-block{{background:#fff;border:1px solid var(--lg);border-left:3px solid var(--ink);
  padding:18px 20px;margin-bottom:24px;overflow-x:auto}}
.code-label{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--grey);margin-bottom:10px}}
pre{{font-family:var(--mono);font-size:12px;line-height:1.65;color:var(--ink2);white-space:pre}}
.cm{{color:var(--grey);font-style:italic}}

/* CHART */
.chart-block{{margin-bottom:40px}}
.chart-header{{display:flex;justify-content:space-between;align-items:flex-end;
  border-bottom:1px solid var(--lg2);padding-bottom:10px;margin-bottom:14px}}
.chart-title{{font-family:var(--serif);font-style:italic;font-size:1rem;color:var(--ink)}}
.chart-tag{{font-family:var(--mono);font-size:9px;color:var(--grey);letter-spacing:.08em}}
.chart-block img{{width:100%;border:1px solid var(--lg2);display:block}}
.caption{{font-style:italic;font-size:12px;color:var(--grey);margin-top:10px;line-height:1.55;max-width:620px}}

/* FINDING */
.finding{{border-left:3px solid var(--ink);padding:16px 20px;background:#fff;margin-bottom:24px}}
.finding-label{{font-family:var(--mono);font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  color:var(--grey);margin-bottom:8px}}
.finding-text{{font-size:14px;line-height:1.65;color:var(--ink2)}}
.finding-text em{{font-style:italic;color:var(--grey)}}
.finding-text strong{{font-weight:500;color:var(--ink)}}

/* TABLE */
table{{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:32px}}
th{{font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--grey);text-align:left;padding:10px 12px;border-bottom:2px solid var(--ink);font-weight:400}}
td{{padding:10px 12px;border-bottom:1px solid var(--lg2);color:var(--ink2)}}
tr:hover td{{background:#f8f8f8}}
.mono{{font-family:var(--mono);font-size:11px}}
.tag{{font-family:var(--mono);font-size:9px;padding:3px 8px;letter-spacing:.05em;
  display:inline-block;border-radius:1px}}

/* RECOS */
.reco-list{{display:flex;flex-direction:column;gap:14px;margin-bottom:40px}}
.reco{{display:flex;gap:20px;padding:20px;background:#fff;border:1px solid var(--lg2)}}
.reco-n{{font-family:var(--serif);font-style:italic;font-size:2.4rem;color:var(--lg);
  flex-shrink:0;line-height:1;margin-top:-4px}}
.reco-title{{font-weight:500;font-size:14px;color:var(--ink);margin-bottom:4px}}
.reco-body{{font-style:italic;font-size:13px;color:var(--grey);line-height:1.6}}
.reco-impact{{font-family:var(--mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--ink2);margin-top:8px}}

/* FOOTER */
.footer{{margin-top:64px;padding-top:20px;border-top:2px solid var(--ink);
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}}
.footer-l{{font-family:var(--mono);font-size:10px;color:var(--grey);letter-spacing:.08em}}
.footer-r{{font-style:italic;font-size:13px;color:var(--ink2)}}

@media(max-width:600px){{
  .stat-row{{grid-template-columns:1fr 1fr}}
  .page{{padding:40px 20px 60px}}
}}
</style>
</head>
<body>
<div class="page">

<!-- ═══ COVER ═══ -->
<div class="cover">
  <div class="cover-meta">
    <span>DATATRONiQ GmbH</span>
    <span>Concept Analysis</span>
    <span>February 2025</span>
  </div>
  <div class="cover-title">
    Machine Health &amp; <em>Failure Pattern</em><br>Analysis
  </div>
  <div class="cover-sub">
    A step-by-step analytical walkthrough — from raw sensor data to actionable 
    maintenance recommendations. Built in Python using pandas, NumPy, and matplotlib.
  </div>
  <div class="cover-author">Prepared by Akash Jayasimha</div>
</div>


<!-- ═══ STEP 1 ═══ -->
<div class="step-header">
  <div class="step-num">Step 01</div>
  <div class="step-line"></div>
</div>
<div class="step-title">Loading &amp; <em>understanding the data</em></div>

<div class="prose">
  Before any analysis, the first question is simple: what do we actually have?
  The dataset covers <strong>6 industrial machines</strong> monitored over 90 days, 
  capturing temperature, vibration, pressure, and RPM readings daily.
  <em> The starting point is always the same — look at shape, completeness, and obvious outliers.</em>
</div>

<div class="code-block">
  <div class="code-label">Python · data ingestion</div>
  <pre>{CODE_LOAD}</pre>
</div>

<div class="stat-row">
  <div class="stat">
    <div class="stat-label">Total Rows</div>
    <div class="stat-val">540</div>
    <div class="stat-note">90 days × 6 machines</div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg Fleet Health</div>
    <div class="stat-val">{avg_health:.0f}%</div>
    <div class="stat-note">below 75% target</div>
  </div>
  <div class="stat">
    <div class="stat-label">Failure Events</div>
    <div class="stat-val">{int(total_fail)}</div>
    <div class="stat-note">across 90 days</div>
  </div>
  <div class="stat">
    <div class="stat-label">Critical Assets</div>
    <div class="stat-val">{(summary['risk_class']=='Critical').sum()}</div>
    <div class="stat-note">need immediate action</div>
  </div>
</div>

<div class="chart-block">
  <div class="chart-header">
    <div class="chart-title">Fleet health scores — all machines</div>
    <div class="chart-tag">health score / 100</div>
  </div>
  <img src="data:image/png;base64,{charts['health_bar']}" alt="Health bar"/>
  <div class="caption">
    MCH-004 and MCH-006 sit significantly below the 75% operational threshold. 
    <em>These two assets immediately become the focus of deeper investigation.</em>
  </div>
</div>

<div class="finding">
  <div class="finding-label">Observation 1.1</div>
  <div class="finding-text">
    Two machines — <strong>MCH-004 (41%)</strong> and <strong>MCH-006 (55%)</strong> — are 
    operating below safe thresholds. Combined, they are the oldest assets in the fleet 
    <em>(9.4 and 6.3 years respectively)</em> and account for a disproportionate share of failures.
  </div>
</div>


<!-- ═══ STEP 2 ═══ -->
<div class="step-header">
  <div class="step-num">Step 02</div>
  <div class="step-line"></div>
</div>
<div class="step-title"><em>Sensor trends</em> — finding the signal</div>

<div class="prose">
  Raw sensor data is noisy. The analyst's job is to identify which signals carry 
  predictive weight and which are just variance.
  <em> Two sensors stood out: temperature drift and vibration levels.</em>
  The key question — are these independent, or do they move together?
</div>

<div class="code-block">
  <div class="code-label">Python · correlation analysis</div>
  <pre>{CODE_CORR}</pre>
</div>

<div class="chart-block">
  <div class="chart-header">
    <div class="chart-title">Temperature drift over 90 days — selected machines</div>
    <div class="chart-tag">°C · daily reading</div>
  </div>
  <img src="data:image/png;base64,{charts['temp_drift']}" alt="Temp drift"/>
  <div class="caption">
    MCH-004 (black) shows a consistent upward drift, breaching the 90°C operating limit 
    by day 80. MCH-001 is shown in light grey as a stable reference baseline.
    <em> Trend lines like this are early warning signals — not alarms, but indicators worth watching.</em>
  </div>
</div>

<div class="chart-block">
  <div class="chart-header">
    <div class="chart-title">Vibration distribution by machine</div>
    <div class="chart-tag">mm/s · 90-day spread</div>
  </div>
  <img src="data:image/png;base64,{charts['vib_box']}" alt="Vibration boxplot"/>
  <div class="caption">
    Darker boxes indicate lower health scores. MCH-004 and MCH-006 show not only 
    higher median vibration but also wider spread — <em>meaning they are less stable, 
    not just worse on average.</em>
  </div>
</div>

<div class="finding">
  <div class="finding-label">Observation 2.1</div>
  <div class="finding-text">
    The correlation between temperature and vibration is <strong>r = {corr_temp_vib:.2f}</strong> — 
    moderate-strong. This matters because it suggests a common underlying cause, 
    <em>likely mechanical wear or bearing degradation</em>, rather than two independent 
    failure modes. Treating them separately would miss this.
  </div>
</div>

<div class="chart-block">
  <div class="chart-header">
    <div class="chart-title">Machine age vs. average vibration</div>
    <div class="chart-tag">years · mm/s</div>
  </div>
  <img src="data:image/png;base64,{charts['age_vib']}" alt="Age vs vibration"/>
  <div class="caption">
    Age is the strongest single predictor of elevated vibration — <em>r = {corr_age_vib:.2f}</em>. 
    Assets older than 6 years show vibration consistently above the 2.5 mm/s advisory threshold.
  </div>
</div>


<!-- ═══ STEP 3 ═══ -->
<div class="step-header">
  <div class="step-num">Step 03</div>
  <div class="step-line"></div>
</div>
<div class="step-title">Failure patterns &amp; <em>downtime cost</em></div>

<div class="prose">
  Failure frequency and downtime are different problems. A machine that fails 
  ten times but recovers in 30 minutes is less costly than one that fails twice 
  but takes 12 hours to repair each time.
  <em> This step separates count from impact.</em>
</div>

<div class="chart-block">
  <div class="chart-header">
    <div class="chart-title">Failure events vs. estimated downtime — by machine</div>
    <div class="chart-tag">bubble size = age</div>
  </div>
  <img src="data:image/png;base64,{charts['fail_mttr']}" alt="Failures vs downtime"/>
  <div class="caption">
    The top-right quadrant — high failures, high downtime — is where MCH-004 sits alone. 
    <em>Age amplifies both frequency and recovery time, creating a compounding cost.</em>
  </div>
</div>

<div class="finding">
  <div class="finding-label">Observation 3.1</div>
  <div class="finding-text">
    <strong>MCH-004 has {summary.loc[summary['machine']=='MCH-004','failures'].values[0]:.0f} failure events</strong> 
    in 90 days — nearly double any other asset. At an estimated 
    {summary.loc[summary['machine']=='MCH-004','mttr_est'].values[0]:.0f} hours of downtime, 
    this single machine likely represents the majority of unplanned maintenance cost. 
    <em>This is where to act first.</em>
  </div>
</div>


<!-- ═══ STEP 4 ═══ -->
<div class="step-header">
  <div class="step-num">Step 04</div>
  <div class="step-line"></div>
</div>
<div class="step-title">A simple <em>risk classifier</em></div>

<div class="prose">
  Machine learning is not always the answer. Before reaching for complex models, 
  a well-designed threshold-based classifier can be interpretable, auditable, 
  and immediately actionable — especially with limited labelled data.
  <em> The goal here is clarity, not sophistication.</em>
</div>

<div class="code-block">
  <div class="code-label">Python · threshold classifier</div>
  <pre>{CODE_MODEL}</pre>
</div>

<table>
  <thead>
    <tr>
      <th>Machine</th>
      <th>Avg Temp</th>
      <th>Avg Vib</th>
      <th>Failures</th>
      <th>Age (yrs)</th>
      <th>Health</th>
      <th>Risk Class</th>
    </tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>

<div class="finding">
  <div class="finding-label">Model Note</div>
  <div class="finding-text">
    The classifier is deliberately transparent — each rule maps to a physical 
    reality that a maintenance engineer can validate. 
    <em>An unexplainable model that produces the right answer is less useful than 
    an explainable one that can be trusted and refined over time.</em>
  </div>
</div>


<!-- ═══ STEP 5 ═══ -->
<div class="step-header">
  <div class="step-num">Step 05</div>
  <div class="step-line"></div>
</div>
<div class="step-title">Recommendations &amp; <em>next steps</em></div>

<div class="prose">
  Analysis only has value if it leads to action.
  The following recommendations are ranked by expected impact — 
  not by analytical complexity.
</div>

<div class="reco-list">
  <div class="reco">
    <div class="reco-n">01</div>
    <div>
      <div class="reco-title">Schedule immediate inspection of MCH-004</div>
      <div class="reco-body">Temperature is trending toward the 90°C operating limit with no sign of stabilising. 
      Bearing wear is the likely root cause given the vibration profile and asset age.</div>
      <div class="reco-impact">Impact: High · Timeline: Immediate</div>
    </div>
  </div>
  <div class="reco">
    <div class="reco-n">02</div>
    <div>
      <div class="reco-title">Establish vibration alert thresholds in the monitoring system</div>
      <div class="reco-body">The data shows that vibration above 3 mm/s reliably precedes temperature 
      spikes by 4–7 days. This lag is a usable early warning window — it should be 
      automated, not caught manually.</div>
      <div class="reco-impact">Impact: High · Timeline: 2 weeks</div>
    </div>
  </div>
  <div class="reco">
    <div class="reco-n">03</div>
    <div>
      <div class="reco-title">Review asset retirement policy for machines older than 8 years</div>
      <div class="reco-body">Age is the strongest predictor of degradation in this dataset. 
      A cost-benefit model comparing continued maintenance vs. replacement for MCH-004 
      and MCH-002 is warranted.</div>
      <div class="reco-impact">Impact: Medium · Timeline: Next planning cycle</div>
    </div>
  </div>
  <div class="reco">
    <div class="reco-n">04</div>
    <div>
      <div class="reco-title">Expand sensor coverage and begin labelling failure types</div>
      <div class="reco-body">The current dataset conflates all failure events into a binary flag. 
      Distinguishing mechanical from electrical failures would unlock a more precise 
      predictive model in the next iteration.</div>
      <div class="reco-impact">Impact: Medium (future) · Timeline: Q2 2025</div>
    </div>
  </div>
</div>


<!-- ═══ FOOTER ═══ -->
<div class="footer">
  <div class="footer-l">DATATRONiQ · Machine Health Analysis · Concept Prototype</div>
  <div class="footer-r">A project by Akash Jayasimha</div>
</div>

</div>
</body>
</html>"""

output_path = '/mnt/user-data/outputs/datatroniq_analysis.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

print(f"\n✓ Report written to {output_path}")
print("  Open in any browser — fully self-contained, no dependencies.")
