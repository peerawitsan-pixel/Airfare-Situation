#!/usr/bin/env python3
"""
Q1: How do geopolitical oil shocks translate into airline ticket price increases,
    and with what time lag?

Datasets used:
  oil_jet_fuel_prices.csv    – monthly Brent crude & jet fuel (2019-2026)
  conflict_oil_events.csv    – 39 key geopolitical events with oil & airfare impact
  airline_ticket_prices.csv  – 14 850 monthly ticket records (base fare + surcharge)
  airline_financial_impact.csv / fuel_surcharges.csv / route_cost_impact.csv – supplementary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from scipy import stats
import warnings
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path as pth
warnings.filterwarnings("ignore")

DATA_DIR = pth.PATH

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load all datasets
# ─────────────────────────────────────────────────────────────────────────────
oil        = pd.read_csv(f"{DATA_DIR}/oil_jet_fuel_prices.csv",      parse_dates=["month"])
events     = pd.read_csv(f"{DATA_DIR}/conflict_oil_events.csv",       parse_dates=["event_date"])
tickets    = pd.read_csv(f"{DATA_DIR}/airline_ticket_prices.csv",     parse_dates=["month"])
financial  = pd.read_csv(f"{DATA_DIR}/airline_financial_impact.csv",  parse_dates=["month"])
surcharges = pd.read_csv(f"{DATA_DIR}/fuel_surcharges.csv",           parse_dates=["month"])
routes     = pd.read_csv(f"{DATA_DIR}/route_cost_impact.csv",         parse_dates=["month"])

print("=" * 60)
print("1. DATA OVERVIEW")
print("=" * 60)
for name, df in [("oil_jet_fuel", oil), ("conflict_events", events),
                 ("ticket_prices", tickets), ("airline_financials", financial),
                 ("fuel_surcharges", surcharges), ("route_costs", routes)]:
    print(f"  {name:22s}: {df.shape[0]:7,} rows x {df.shape[1]} cols")

# 2. EDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. EDA SUMMARY")
print("=" * 60)

print("\n-- Oil Prices --")
print(f"  Date range  : {oil.month.min().strftime('%Y-%m')} to {oil.month.max().strftime('%Y-%m')}")
print(f"  Brent range : ${oil.brent_crude_usd_barrel.min():.2f} – ${oil.brent_crude_usd_barrel.max():.2f} /bbl")
print(f"  Jet fuel    : ${oil.jet_fuel_usd_barrel.min():.2f} – ${oil.jet_fuel_usd_barrel.max():.2f} /bbl")
print(f"  Hormuz disrupted: {(oil.strait_hormuz_disrupted == 'Yes').sum()} months")

print("\n-- Ticket Prices --")
print(f"  Airlines    : {tickets.airline.nunique()}")
print(f"  Route types : {list(tickets.route_class.unique())}")
print(f"  Total fare  : ${tickets.total_fare_usd.mean():.2f} avg  "
      f"(min ${tickets.total_fare_usd.min():.2f}, max ${tickets.total_fare_usd.max():.2f})")
print(f"  Surcharge   : ${tickets.fuel_surcharge_usd.mean():.2f} avg")

print("\n-- Geopolitical Events --")
print(f"  Total events: {len(events)}")
print(f"  By severity:")
sev_stats = events.groupby("severity")[["oil_price_change_pct", "airfare_impact_pct"]].agg(["count", "mean"])
sev_stats.columns = ["_".join(c) for c in sev_stats.columns]
print(sev_stats.round(2).to_string())

print("\n-- Conflict Phases in oil dataset --")
print(oil.groupby("conflict_phase")["brent_crude_usd_barrel"].agg(["count", "mean", "max"]).round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build monthly merged table
# ─────────────────────────────────────────────────────────────────────────────
monthly_fare = (
    tickets.groupby("month")
    .agg(
        avg_total_fare     = ("total_fare_usd",       "mean"),
        avg_base_fare      = ("base_fare_usd",         "mean"),
        avg_fuel_surcharge = ("fuel_surcharge_usd",    "mean"),
        avg_load_factor    = ("load_factor_pct",       "mean"),
        avg_fuel_pct_opex  = ("fuel_cost_pct_opex",    "mean"),
    )
    .reset_index()
)

monthly = oil.merge(monthly_fare, on="month", how="inner").sort_values("month").reset_index(drop=True)
monthly["brent_mom_pct"] = monthly["brent_crude_usd_barrel"].pct_change() * 100
monthly["jet_mom_pct"]   = monthly["jet_fuel_usd_barrel"].pct_change() * 100
monthly["fare_mom_pct"]  = monthly["avg_total_fare"].pct_change() * 100

print(f"\n-- Monthly merged table --")
print(f"  Rows: {len(monthly)}  |  Date range: "
      f"{monthly.month.min().strftime('%Y-%m')} to {monthly.month.max().strftime('%Y-%m')}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cross-Correlation (lag analysis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CROSS-CORRELATION: Oil % Change -> Ticket % Change")
print("=" * 60)

clean = monthly.dropna(subset=["brent_mom_pct", "fare_mom_pct"]).copy()
oil_ch  = clean["brent_mom_pct"].values
fare_ch = clean["fare_mom_pct"].values

MAX_LAG = 8
lags, corrs, pvals = [], [], []
for lag in range(0, MAX_LAG + 1):
    x = oil_ch[:-lag] if lag > 0 else oil_ch
    y = fare_ch[lag:]  if lag > 0 else fare_ch
    r, p = stats.pearsonr(x, y)
    lags.append(lag);  corrs.append(r);  pvals.append(p)
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"  Lag {lag:2d} months: r = {r:+.4f}  p = {p:.4f} {sig}")

best_lag = int(np.argmax(np.abs(corrs)))
print(f"\n  ==> Best lag: {best_lag} month(s)  (r = {corrs[best_lag]:+.4f}, p = {pvals[best_lag]:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Event-window summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. EVENT IMPACT BY SEVERITY")
print("=" * 60)
print(events.groupby("severity")[["oil_price_change_pct", "airfare_impact_pct"]].mean().round(2).to_string())

print("\n" + "=" * 60)
print("5. EVENT IMPACT BY CONFLICT PHASE")
print("=" * 60)
print(events.groupby("conflict_phase")[["oil_price_change_pct", "airfare_impact_pct"]].mean().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# Color palettes
# ─────────────────────────────────────────────────────────────────────────────
PHASE_COLORS = {
    "Pre-Pandemic Baseline":        "#4CAF50",
    "COVID-19 Collapse":            "#9C27B0",
    "Recovery & Surge":             "#2196F3",
    "Ukraine War Shock":            "#FF5722",
    "Stabilisation":                "#607D8B",
    "Gaza-Israel Conflict":         "#FF9800",
    "Pre-Iran Escalation":          "#795548",
    "US-Iran War Conflict":         "#F44336",
    "De-escalation & Negotiations": "#00BCD4",
}

SEV_COLORS = {
    "Medium":    "#FFC107",
    "High":      "#FF9800",
    "Very High": "#F44336",
    "Extreme":   "#9C27B0",
}

PHASE_ORDER = [
    "Pre-Pandemic Baseline", "COVID-19 Collapse", "Recovery & Surge",
    "Ukraine War Shock", "Stabilisation", "Gaza-Israel Conflict",
    "Pre-Iran Escalation", "US-Iran War Conflict", "De-escalation & Negotiations",
]

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – Full Timeline Overview (3 panels)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(17, 12.5), sharex=True,
                         gridspec_kw={"hspace": 0.12, "height_ratios": [2, 2, 1.5]})
fig.suptitle("Geopolitical Oil Shocks → Airline Ticket Prices  (2019 – 2026)\nFull Timeline Overview",
             fontsize=20, fontweight="bold", y=0.998)

# Phase shading helper
def shade_phases(ax, monthly):
    for phase, grp in monthly.groupby("conflict_phase"):
        c = PHASE_COLORS.get(phase, "#CCCCCC")
        ax.axvspan(grp.month.min(), grp.month.max(), alpha=0.10, color=c, lw=0)

# Event lines helper
def draw_events(ax, events, ymax_frac=1.0):
    ylim = ax.get_ylim()
    for _, ev in events.iterrows():
        c = SEV_COLORS.get(ev.severity, "#888888")
        ax.axvline(ev.event_date, color=c, lw=0.8, alpha=0.65, zorder=2)

# Short display names for the shock/situation labels drawn on the timeline
SHORT_PHASE_LABEL = {
    "Pre-Pandemic Baseline":        "Pre-Pandemic\nBaseline",
    "COVID-19 Collapse":            "COVID-19\nCollapse",
    "Recovery & Surge":             "Recovery\n& Surge",
    "Ukraine War Shock":            "Ukraine\nWar",
    "Stabilisation":                "Stabilisation",
    "Gaza-Israel Conflict":         "Gaza-Israel\nConflict",
    "Pre-Iran Escalation":          "Pre-Iran\nEscalation",
    "US-Iran War Conflict":         "US-Iran\nWar",
    "De-escalation & Negotiations": "De-escalation\n& Talks",
}

# Phase/shock name labels — placed at the top of a panel so the audience can
# immediately tie the shaded timeline region to the name of the situation.
def label_phases(ax, monthly):
    y0, y1 = ax.get_ylim()
    y_text = y1 - (y1 - y0) * 0.03
    for phase in PHASE_ORDER:
        grp = monthly[monthly.conflict_phase == phase]
        if grp.empty:
            continue
        c = PHASE_COLORS.get(phase, "#555555")
        x_center = grp.month.min() + (grp.month.max() - grp.month.min()) / 2
        ax.text(x_center, y_text, SHORT_PHASE_LABEL.get(phase, phase),
                rotation=90, va="top", ha="center", fontsize=10.5,
                color=c, fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))

# Panel 1: Oil prices
ax1 = axes[0]
shade_phases(ax1, monthly)
ax1.plot(monthly.month, monthly.brent_crude_usd_barrel, color="#1565C0", lw=2.2, label="Brent Crude ($/bbl)")
ax1.plot(monthly.month, monthly.jet_fuel_usd_barrel,    color="#E65100", lw=2.2, label="Jet Fuel ($/bbl)")
ax1.fill_between(monthly.month, monthly.brent_crude_usd_barrel, alpha=0.08, color="#1565C0")
ax1.set_ylabel("Price  (USD / barrel)", fontsize=11.5, fontweight="bold", labelpad=8)
ax1.set_ylim(0)
ax1.tick_params(axis="both", labelsize=10.5)
ax1.legend(fontsize=10.5, loc="lower left", framealpha=0.9)
ax1.set_title("Crude Oil & Jet Fuel Prices", fontsize=15, fontweight="bold", pad=10)
draw_events(ax1, events)
label_phases(ax1, monthly)

# Panel 2: Ticket prices
ax2 = axes[1]
shade_phases(ax2, monthly)
ax2.plot(monthly.month, monthly.avg_total_fare,     color="#2E7D32", lw=2.2,   label="Total Fare (avg)")
ax2.plot(monthly.month, monthly.avg_base_fare,      color="#00796B", lw=1.8, ls="--", label="Base Fare (avg)")
ax2.plot(monthly.month, monthly.avg_fuel_surcharge, color="#C62828", lw=1.8, ls=":",  label="Fuel Surcharge (avg)")
ax2.fill_between(monthly.month, monthly.avg_total_fare, alpha=0.07, color="#2E7D32")
ax2.set_ylabel("USD ($)", fontsize=11.5, fontweight="bold", labelpad=8)
ax2.set_ylim(0)
ax2.tick_params(axis="both", labelsize=10.5)
ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.9)
ax2.set_title("Average Airline Ticket Price", fontsize=15, fontweight="bold", pad=10)
draw_events(ax2, events)

# Panel 3: MoM % changes
ax3 = axes[2]
shade_phases(ax3, monthly)
clp = monthly.dropna(subset=["brent_mom_pct", "fare_mom_pct"])
ax3.bar(clp.month, clp.brent_mom_pct, width=20, color="#1565C0", alpha=0.45, label="Brent Δ% MoM")
ax3.plot(clp.month, clp.fare_mom_pct, color="#2E7D32", lw=2.0, marker="o", ms=3.0, label="Avg Fare Δ% MoM")
ax3.axhline(0, color="black", lw=0.6)
ax3.set_ylabel("MoM % Change", fontsize=11.5, fontweight="bold", labelpad=8)
ax3.tick_params(axis="both", labelsize=11)
ax3.legend(fontsize=10.5, loc="lower left", framealpha=0.9)
ax3.set_title("Month-over-Month % Change: Oil vs Ticket Price", fontsize=15, fontweight="bold", pad=10)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.xaxis.set_major_locator(mdates.YearLocator())

# Severity legend & Conflict Phase legend (bottom declarations)
sev_patches = [mpatches.Patch(color=v, label=k) for k, v in SEV_COLORS.items()]
phase_patches = [mpatches.Patch(color=PHASE_COLORS[p], alpha=0.7, label=p) for p in PHASE_ORDER if p in PHASE_COLORS]

fig.legend(handles=sev_patches, loc="lower left", ncol=4, fontsize=11,
           title="Event Severity (vertical lines)", title_fontsize=12,
           frameon=True, facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
           bbox_to_anchor=(0.01, -0.06))

fig.legend(handles=phase_patches, loc="lower right", ncol=3, fontsize=10.5,
           title="Conflict Phase (shading)", title_fontsize=12,
           frameon=True, facecolor="white", edgecolor="#CCCCCC", framealpha=0.95,
           bbox_to_anchor=(0.99, -0.06))

plt.savefig(f"{DATA_DIR}/fig1_timeline_overview.png", dpi=200, bbox_inches="tight")
plt.close()
print("[Saved] fig1_timeline_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – Cross-Correlation Lag Analysis
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Time-Lag Analysis: How Long Does an Oil Shock Take to Hit Ticket Prices?",
             fontsize=12, fontweight="bold")

# Left: correlation bars
ax = axes[0]
bar_colors = ["#43A047" if i == best_lag else "#90A4AE" for i in lags]
bars = ax.bar(lags, corrs, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("Oil price leads ticket price by N months", fontsize=9)
ax.set_ylabel("Pearson r", fontsize=9)
ax.set_title("Cross-Correlation at Each Lag", fontsize=10, fontweight="bold")
ax.set_xticks(lags)
ax.set_xticklabels([f"Lag {l}" for l in lags], rotation=30, ha="right")
for bar, r, p in zip(bars, corrs, pvals):
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    ypos = bar.get_height() + 0.005 if r >= 0 else bar.get_height() - 0.018
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"{r:.3f}{sig}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(min(corrs) - 0.05, max(corrs) + 0.08)
ax.text(0.97, 0.97,
        f"Best lag: {best_lag} month(s)\nr = {corrs[best_lag]:.3f}  p = {pvals[best_lag]:.4f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.9))

# Right: scatter at best lag
ax2 = axes[1]
if best_lag == 0:
    sx = oil_ch;         sy = fare_ch;        phases_s = clean["conflict_phase"]
else:
    sx = oil_ch[:-best_lag];  sy = fare_ch[best_lag:];  phases_s = clean["conflict_phase"].iloc[:-best_lag]

sc_colors = [PHASE_COLORS.get(p, "#888") for p in phases_s]
ax2.scatter(sx, sy, c=sc_colors, alpha=0.75, s=50, edgecolors="white", linewidth=0.5, zorder=3)

sl, ic, rv, pv, _ = stats.linregress(sx, sy)
xl = np.linspace(sx.min(), sx.max(), 100)
ax2.plot(xl, sl * xl + ic, "r--", lw=2, label=f"OLS: y = {sl:.2f}x + {ic:.2f}")
ax2.axhline(0, color="grey", lw=0.5, ls="--")
ax2.axvline(0, color="grey", lw=0.5, ls="--")
ax2.set_xlabel("Brent % Change (month t)", fontsize=9)
ax2.set_ylabel(f"Avg Ticket % Change (month t + {best_lag})", fontsize=9)
ax2.set_title(f"Scatter: Oil Shock vs Ticket Price  (Lag = {best_lag} month)", fontsize=10, fontweight="bold")
ax2.legend(fontsize=9)
ax2.text(0.03, 0.97, f"r = {rv:.3f}   p = {pv:.4f}", transform=ax2.transAxes,
         va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

# Phase color legend
phase_patches = [mpatches.Patch(color=PHASE_COLORS[p], label=p)
                 for p in PHASE_ORDER if p in monthly.conflict_phase.values]
ax2.legend(handles=phase_patches, fontsize=7, ncol=1, loc="lower right", title="Phase")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/fig2_cross_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig2_cross_correlation.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – Event Impact Analysis (Avg Oil & Airfare Impact by Conflict Phase)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
fig.suptitle("Geopolitical Event Impact: Avg Oil & Airfare Impact by Conflict Phase",
             fontsize=14, fontweight="bold", y=0.97)

ep = events.groupby("conflict_phase")[["oil_price_change_pct", "airfare_impact_pct"]].mean()
ep = ep.reindex([p for p in PHASE_ORDER if p in ep.index])
x = np.arange(len(ep))
w = 0.36

# Phase colors for bars
bar_colors = [PHASE_COLORS.get(p, "#4C72B0") for p in ep.index]

rects1 = ax.bar(x - w / 2, ep.oil_price_change_pct, w, label="Oil Price Change (%)",
               color=bar_colors, alpha=0.9, edgecolor="white", linewidth=1.2)
rects2 = ax.bar(x + w / 2, ep.airfare_impact_pct, w, label="Airfare Impact (%)",
               color=bar_colors, alpha=0.45, edgecolor="white", linewidth=1.2, hatch="//")

# Add numerical labels on top/bottom of bars
for rect in rects1:
    h = rect.get_height()
    va = "bottom" if h >= 0 else "top"
    offset = 0.4 if h >= 0 else -0.8
    ax.annotate(f"{h:+.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=va, fontsize=8.5, fontweight="bold", color="#333333")

for rect in rects2:
    h = rect.get_height()
    va = "bottom" if h >= 0 else "top"
    offset = 0.4 if h >= 0 else -0.8
    ax.annotate(f"{h:+.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=va, fontsize=8.5, fontweight="bold", color="#333333")

# Formatted phase labels
formatted_labels = [
    p.replace(" Baseline", "\nBaseline")
     .replace(" Collapse", "\nCollapse")
     .replace(" & Surge", "\n& Surge")
     .replace(" War Shock", "\nWar Shock")
     .replace(" Conflict", "\nConflict")
     .replace(" Escalation", "\nEscalation")
     .replace(" & Negotiations", "\n& Negotiations")
    for p in ep.index
]

ax.set_xticks(x)
ax.set_xticklabels(formatted_labels, fontsize=9.5, fontweight="medium")
ax.axhline(0, color="black", lw=0.9, linestyle="-")
ax.grid(True, axis="y", linestyle=":", alpha=0.4, color="gray")

ax.set_ylabel("Average Impact (%)", fontsize=11, fontweight="bold")
ax.set_title("Average Impact of Oil Price Shocks and Immediate Airfare Changes per Conflict Phase",
             fontsize=11, pad=15)

# Custom legend
solid_patch = mpatches.Patch(facecolor="#607D8B", alpha=0.9, label="Oil Price Change (%)")
hatch_patch = mpatches.Patch(facecolor="#607D8B", alpha=0.45, hatch="//", label="Airfare Impact (%)")
ax.legend(handles=[solid_patch, hatch_patch], fontsize=10, loc="upper left", framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{DATA_DIR}/fig3_event_impact.png", dpi=200, bbox_inches="tight")
plt.close()
print("[Saved] fig3_event_impact.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 – Phase-Level Summary Dashboard (2×2)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Oil-to-Airfare Transmission — Phase Dashboard", fontsize=13, fontweight="bold")

phase_monthly = monthly.groupby("conflict_phase").agg(
    avg_brent    = ("brent_crude_usd_barrel", "mean"),
    avg_jet      = ("jet_fuel_usd_barrel",    "mean"),
    avg_fare     = ("avg_total_fare",          "mean"),
    avg_surcharge= ("avg_fuel_surcharge",      "mean"),
).reindex([p for p in PHASE_ORDER if p in monthly.conflict_phase.values])
pcolors = [PHASE_COLORS.get(p, "#888") for p in phase_monthly.index]
xlabels = [p.replace(" & ", "\n& ") for p in phase_monthly.index]

# (0,0) Avg Brent by phase
ax = axes[0, 0]
bars = ax.bar(range(len(phase_monthly)), phase_monthly.avg_brent, color=pcolors, edgecolor="white")
ax.set_xticks(range(len(phase_monthly)))
ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=7.5)
ax.set_ylabel("USD / barrel", fontsize=9)
ax.set_title("Avg Brent Crude Price by Phase", fontsize=10, fontweight="bold")
for bar, v in zip(bars, phase_monthly.avg_brent):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"${v:.0f}", ha="center", va="bottom", fontsize=7.5)

# (0,1) Avg ticket fare by phase
ax = axes[0, 1]
bars = ax.bar(range(len(phase_monthly)), phase_monthly.avg_fare, color=pcolors, edgecolor="white")
ax.set_xticks(range(len(phase_monthly)))
ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=7.5)
ax.set_ylabel("USD", fontsize=9)
ax.set_title("Avg Total Ticket Fare by Phase", fontsize=10, fontweight="bold")
for bar, v in zip(bars, phase_monthly.avg_fare):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"${v:.0f}", ha="center", va="bottom", fontsize=7.5)

# (1,0) Avg fuel surcharge by phase
ax = axes[1, 0]
bars = ax.bar(range(len(phase_monthly)), phase_monthly.avg_surcharge, color=pcolors, edgecolor="white")
ax.set_xticks(range(len(phase_monthly)))
ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=7.5)
ax.set_ylabel("USD", fontsize=9)
ax.set_title("Avg Fuel Surcharge by Phase", fontsize=10, fontweight="bold")
for bar, v in zip(bars, phase_monthly.avg_surcharge):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"${v:.0f}", ha="center", va="bottom", fontsize=7.5)

# (1,1) Oil price vs ticket fare scatter (all months, phase-colored)
ax = axes[1, 1]
for phase in PHASE_ORDER:
    sub = monthly[monthly.conflict_phase == phase]
    if len(sub):
        ax.scatter(sub.brent_crude_usd_barrel, sub.avg_total_fare,
                   c=PHASE_COLORS.get(phase, "#888"), label=phase,
                   alpha=0.7, s=35, edgecolors="none")

sl, ic, rv, pv, _ = stats.linregress(monthly.brent_crude_usd_barrel, monthly.avg_total_fare)
xl = np.linspace(monthly.brent_crude_usd_barrel.min(), monthly.brent_crude_usd_barrel.max(), 100)
ax.plot(xl, sl * xl + ic, "k--", lw=1.8)
ax.set_xlabel("Brent Crude ($/bbl)", fontsize=9)
ax.set_ylabel("Avg Ticket Fare (USD)", fontsize=9)
ax.set_title(f"Oil Price vs Ticket Fare (all months)  r = {rv:.3f}", fontsize=10, fontweight="bold")
phase_patches = [mpatches.Patch(color=PHASE_COLORS[p], label=p)
                 for p in PHASE_ORDER if p in monthly.conflict_phase.values]
ax.legend(handles=phase_patches, fontsize=6.5, ncol=1, loc="upper left")

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/fig4_phase_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] fig4_phase_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)
print(f"  1. Best lag (oil shock -> ticket price): {best_lag} month(s)  r = {corrs[best_lag]:.3f}")
print(f"  2. Extreme events - avg oil change: {events[events.severity=='Extreme'].oil_price_change_pct.mean():.1f}%  |  avg airfare change: {events[events.severity=='Extreme'].airfare_impact_pct.mean():.1f}%")
print(f"  3. US-Iran War phase - avg Brent: ${oil[oil.conflict_phase=='US-Iran War Conflict'].brent_crude_usd_barrel.mean():.1f}/bbl")
print(f"  4. US-Iran War phase - avg ticket fare: ${monthly[monthly.conflict_phase=='US-Iran War Conflict'].avg_total_fare.mean():.0f}")
print(f"\n  Figures saved to: {DATA_DIR}")
print("    fig1_timeline_overview.png")
print("    fig2_cross_correlation.png")
print("    fig3_event_impact.png")
print("    fig4_phase_dashboard.png")
