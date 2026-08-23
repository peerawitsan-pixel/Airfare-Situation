#!/usr/bin/env python3
"""
Q2: Do airlines pass through geopolitical oil-price shocks into airfares at a
    consistent rate and speed, and how does the 2026 Iran crisis differ from
    COVID-19 (2020) and Russia-Ukraine War (2022)?

Approach:
  - Define shock windows for the 3 crises
  - Indexed trajectory comparison (oil & fare both = 100 at T0)
  - Pass-through rate = (fare % change from base) / (oil % change from base)
  - Cross-correlation per shock to measure transmission speed
  - Airline-level heterogeneity: Flag Carrier vs LCC, by region
  - Iran crisis deep-dive vs prior shocks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path as pth

DATA_DIR = pth.PATH

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
oil        = pd.read_csv(f"{DATA_DIR}/oil_jet_fuel_prices.csv",      parse_dates=["month"])
events     = pd.read_csv(f"{DATA_DIR}/conflict_oil_events.csv",       parse_dates=["event_date"])
tickets    = pd.read_csv(f"{DATA_DIR}/airline_ticket_prices.csv",     parse_dates=["month"])
financial  = pd.read_csv(f"{DATA_DIR}/airline_financial_impact.csv",  parse_dates=["month"])

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build monthly aggregates (same as analysis1)
# ─────────────────────────────────────────────────────────────────────────────
monthly_fare = (
    tickets.groupby("month")
    .agg(
        avg_total_fare     = ("total_fare_usd",       "mean"),
        avg_base_fare      = ("base_fare_usd",         "mean"),
        avg_fuel_surcharge = ("fuel_surcharge_usd",    "mean"),
        avg_load_factor    = ("load_factor_pct",       "mean"),
    )
    .reset_index()
)

monthly = (
    oil.merge(monthly_fare, on="month", how="inner")
    .sort_values("month")
    .reset_index(drop=True)
)
monthly["brent_mom_pct"] = monthly["brent_crude_usd_barrel"].pct_change() * 100
monthly["fare_mom_pct"]  = monthly["avg_total_fare"].pct_change() * 100

# ─────────────────────────────────────────────────────────────────────────────
# 3. Define three shock windows
#    T0 = baseline month (just before shock)
#    Shock window = T0 onwards
# ─────────────────────────────────────────────────────────────────────────────
SHOCKS = {
    "COVID-19 Collapse\n(Mar 2020)": {
        "t0":      "2020-02",    # Brent = $61.7, last normal month
        "start":   "2020-03",    # first collapse month
        "end":     "2021-02",    # 12 months post-shock
        "color":   "#9C27B0",
        "hatch":   "",
        "direction": -1,         # oil went DOWN
        "label":   "COVID-19",
    },
    "Russia-Ukraine War\n(Mar 2022)": {
        "t0":      "2022-02",    # Brent = $67.3
        "start":   "2022-03",    # invasion & price spike
        "end":     "2023-02",
        "color":   "#FF5722",
        "hatch":   "//",
        "direction": +1,
        "label":   "Ukraine War",
    },
    "US-Iran War\n(Dec 2025)": {
        "t0":      "2025-11",    # Brent = $92.6
        "start":   "2025-12",    # strikes begin
        "end":     "2026-06",    # last available month
        "color":   "#F44336",
        "hatch":   "xx",
        "direction": +1,
        "label":   "Iran Crisis",
    },
}

def get_shock_df(shock_key, shock_cfg, monthly):
    """Return monthly rows for a shock window, indexed to T0=100."""
    t0   = pd.Timestamp(shock_cfg["t0"])
    end  = pd.Timestamp(shock_cfg["end"])
    sub  = monthly[monthly.month >= t0].copy()
    sub  = sub[sub.month <= end].copy()
    sub  = sub.reset_index(drop=True)

    base_oil  = sub.loc[0, "brent_crude_usd_barrel"]
    base_fare = sub.loc[0, "avg_total_fare"]
    base_jet  = sub.loc[0, "jet_fuel_usd_barrel"]
    base_surch= sub.loc[0, "avg_fuel_surcharge"]

    sub["t_offset"]    = (sub["month"].dt.to_period("M") - pd.Period(shock_cfg["t0"], "M")).apply(lambda x: x.n)
    sub["oil_idx"]     = sub["brent_crude_usd_barrel"] / base_oil  * 100
    sub["jet_idx"]     = sub["jet_fuel_usd_barrel"]    / base_jet  * 100
    sub["fare_idx"]    = sub["avg_total_fare"]          / base_fare * 100
    sub["surch_idx"]   = sub["avg_fuel_surcharge"]      / base_surch * 100
    sub["oil_chg_pct"] = (sub["brent_crude_usd_barrel"] - base_oil)  / base_oil  * 100
    sub["fare_chg_pct"]= (sub["avg_total_fare"]          - base_fare) / base_fare * 100

    # Pass-through rate: avoid division by zero / near-zero
    sub["pass_through"] = np.where(
        sub["oil_chg_pct"].abs() > 1.0,
        sub["fare_chg_pct"] / sub["oil_chg_pct"],
        np.nan,
    )
    sub["shock"] = shock_cfg["label"]
    sub["color"] = shock_cfg["color"]
    return sub

shock_dfs = {k: get_shock_df(k, v, monthly) for k, v in SHOCKS.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print pass-through statistics
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PASS-THROUGH RATE ANALYSIS")
print("  (fare % change from T0) / (oil % change from T0)")
print("=" * 65)

pt_summary = []
for shock_name, cfg in SHOCKS.items():
    sdf = shock_dfs[shock_name]
    label = cfg["label"]

    # Peak oil change moment
    if cfg["direction"] == +1:
        peak_idx = sdf["oil_chg_pct"].idxmax()
    else:
        peak_idx = sdf["oil_chg_pct"].idxmin()

    peak_row = sdf.loc[peak_idx]
    final_row = sdf.iloc[-1]

    # Speed: first month where fare change exceeds 5% of T0 base (positive or negative)
    if cfg["direction"] == +1:
        thresh = sdf[sdf["fare_chg_pct"] > 5]
    else:
        thresh = sdf[sdf["fare_chg_pct"] < -5]
    speed = thresh.iloc[0]["t_offset"] if len(thresh) else np.nan

    pt_summary.append({
        "Shock": label,
        "T0 Brent ($/bbl)":  round(sdf.iloc[0]["brent_crude_usd_barrel"], 1),
        "Peak Brent ($/bbl)": round(peak_row["brent_crude_usd_barrel"], 1),
        "Peak Oil Chg%":     round(peak_row["oil_chg_pct"], 1),
        "Peak Fare Chg%":    round(peak_row["fare_chg_pct"], 1),
        "Pass-Through (peak)": round(peak_row["pass_through"], 3) if pd.notna(peak_row["pass_through"]) else "N/A",
        "Final Fare Chg%":   round(final_row["fare_chg_pct"], 1),
        "PT (final)":        round(final_row["pass_through"], 3) if pd.notna(final_row["pass_through"]) else "N/A",
        "Months to +5% fare": int(speed) if pd.notna(speed) else ">window",
    })
    print(f"\n  {label}")
    print(f"    T0 Brent         : ${sdf.iloc[0]['brent_crude_usd_barrel']:.1f}/bbl")
    print(f"    Peak oil change  : {peak_row['oil_chg_pct']:+.1f}%  (month T+{int(peak_row['t_offset'])})")
    print(f"    Fare chg at peak : {peak_row['fare_chg_pct']:+.1f}%")
    print(f"    Pass-through     : {peak_row['pass_through']:.3f}" if pd.notna(peak_row['pass_through']) else "    Pass-through     : N/A")
    print(f"    Final fare chg   : {final_row['fare_chg_pct']:+.1f}%  (month T+{int(final_row['t_offset'])})")
    print(f"    Months to +-5%   : {int(speed) if pd.notna(speed) else '>window'}")

pt_df = pd.DataFrame(pt_summary)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-airline pass-through during Iran & Ukraine shocks
# ─────────────────────────────────────────────────────────────────────────────
def airline_passthrough(shock_cfg, tickets, oil):
    t0  = pd.Timestamp(shock_cfg["t0"])
    end = pd.Timestamp(shock_cfg["end"])

    oil_base = oil.loc[oil.month == t0, "brent_crude_usd_barrel"].values
    oil_peak = oil.loc[(oil.month >= t0) & (oil.month <= end), "brent_crude_usd_barrel"].max()
    if len(oil_base) == 0:
        return pd.DataFrame()
    oil_base = oil_base[0]
    oil_chg  = (oil_peak - oil_base) / oil_base * 100

    # Per-airline avg fare at T0 and at peak
    sub = tickets[(tickets.month >= t0) & (tickets.month <= end)].copy()
    fare_t0   = tickets[tickets.month == t0].groupby("airline")["total_fare_usd"].mean()
    fare_peak = sub.groupby("airline")["total_fare_usd"].max()  # max monthly avg per airline

    df = pd.DataFrame({"fare_t0": fare_t0, "fare_peak": fare_peak}).dropna()
    df["fare_chg_pct"] = (df["fare_peak"] - df["fare_t0"]) / df["fare_t0"] * 100
    df["pass_through"] = df["fare_chg_pct"] / oil_chg
    df["oil_chg_pct"]  = oil_chg

    # Merge airline metadata
    meta = tickets[["airline", "airline_type", "region"]].drop_duplicates("airline")
    df = df.reset_index().merge(meta, on="airline")
    df["shock"] = shock_cfg["label"]
    return df

airline_pt = pd.concat([
    airline_passthrough(SHOCKS["Russia-Ukraine War\n(Mar 2022)"], tickets, oil),
    airline_passthrough(SHOCKS["US-Iran War\n(Dec 2025)"],        tickets, oil),
], ignore_index=True)

print("\n" + "=" * 65)
print("AIRLINE-LEVEL PASS-THROUGH (Ukraine vs Iran)")
print("=" * 65)
pivot = airline_pt.pivot(index="airline", columns="shock", values="pass_through").round(3)
print(pivot.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-correlation per shock (transmission speed)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CROSS-CORRELATION PER SHOCK (oil MoM% -> fare MoM%)")
print("=" * 65)

speed_results = {}
for shock_name, cfg in SHOCKS.items():
    sdf = shock_dfs[shock_name].dropna(subset=["brent_mom_pct", "fare_mom_pct"]).copy()
    if len(sdf) < 5:
        continue
    oil_ch  = sdf["brent_mom_pct"].values
    fare_ch = sdf["fare_mom_pct"].values
    max_lag = min(4, len(sdf) - 2)
    best_r, best_lag = 0, 0
    lag_corrs = {}
    for lag in range(0, max_lag + 1):
        x = oil_ch[:-lag] if lag > 0 else oil_ch
        y = fare_ch[lag:]  if lag > 0 else fare_ch
        if len(x) < 3:
            continue
        r, p = stats.pearsonr(x, y)
        lag_corrs[lag] = (r, p)
        if abs(r) > abs(best_r):
            best_r, best_lag = r, lag
    speed_results[cfg["label"]] = {"best_lag": best_lag, "best_r": best_r, "lag_corrs": lag_corrs}
    print(f"\n  {cfg['label']}")
    for lag, (r, p) in lag_corrs.items():
        marker = " <<" if lag == best_lag else ""
        print(f"    Lag {lag}: r = {r:+.3f}  p = {p:.3f}{marker}")

# ─────────────────────────────────────────────────────────────────────────────
# COLOR / STYLE setup
# ─────────────────────────────────────────────────────────────────────────────
SHOCK_COLORS = {cfg["label"]: cfg["color"] for cfg in SHOCKS.values()}
SHOCK_KEYS   = list(SHOCKS.keys())
SHOCK_LABELS = [SHOCKS[k]["label"] for k in SHOCK_KEYS]

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 – Three-Shock Indexed Trajectory (3 cols x 2 rows)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey="row")
fig.suptitle(
    "Three-Shock Comparison: Indexed Trajectories (T0 = 100)\n"
    "How fast and how far did oil prices and airfares move?",
    fontsize=13, fontweight="bold",
)

for col, (shock_name, cfg) in enumerate(SHOCKS.items()):
    sdf   = shock_dfs[shock_name]
    color = cfg["color"]
    label = cfg["label"]

    # Row 0: Oil & Jet fuel index
    ax = axes[0, col]
    ax.plot(sdf.t_offset, sdf.oil_idx, color=color,   lw=2.5, marker="o", ms=4, label="Brent Crude")
    ax.plot(sdf.t_offset, sdf.jet_idx, color=color,   lw=1.5, ls="--", ms=3, marker="s", label="Jet Fuel")
    ax.axhline(100, color="black", lw=0.7, ls=":")
    ax.axvline(0,   color="black", lw=0.7, ls="--", alpha=0.5)
    ax.fill_between(sdf.t_offset, 100, sdf.oil_idx, alpha=0.12, color=color)
    ax.set_title(f"{shock_name}", fontsize=10, fontweight="bold")
    if col == 0:
        ax.set_ylabel("Price Index (T0 = 100)", fontsize=9)
    ax.legend(fontsize=7.5, loc="best")
    # Annotate peak/trough
    if cfg["direction"] == +1:
        pk = sdf.loc[sdf.oil_idx.idxmax()]
    else:
        pk = sdf.loc[sdf.oil_idx.idxmin()]
    ax.annotate(f"{pk.oil_idx:.0f}", (pk.t_offset, pk.oil_idx),
                xytext=(3, 4), textcoords="offset points", fontsize=8, color=color, fontweight="bold")

    # Row 1: Fare & Surcharge index
    ax2 = axes[1, col]
    ax2.plot(sdf.t_offset, sdf.fare_idx,  color=color,   lw=2.5, marker="o", ms=4, label="Total Fare")
    ax2.plot(sdf.t_offset, sdf.surch_idx, color=color,   lw=1.5, ls=":", ms=3, marker="^", label="Fuel Surcharge")
    ax2.axhline(100, color="black", lw=0.7, ls=":")
    ax2.axvline(0,   color="black", lw=0.7, ls="--", alpha=0.5)
    ax2.fill_between(sdf.t_offset, 100, sdf.fare_idx, alpha=0.12, color=color)
    if col == 0:
        ax2.set_ylabel("Fare Index (T0 = 100)", fontsize=9)
    ax2.set_xlabel("Months from shock start (T0)", fontsize=9)
    ax2.legend(fontsize=7.5, loc="best")
    # Annotate peak/trough fare
    if cfg["direction"] == +1:
        pk2 = sdf.loc[sdf.fare_idx.idxmax()]
    else:
        pk2 = sdf.loc[sdf.fare_idx.idxmin()]
    ax2.annotate(f"{pk2.fare_idx:.0f}", (pk2.t_offset, pk2.fare_idx),
                 xytext=(3, 4), textcoords="offset points", fontsize=8, color=color, fontweight="bold")

    # Add pass-through text box
    valid_pt = sdf["pass_through"].dropna()
    if len(valid_pt):
        mean_pt = valid_pt.mean()
        ax2.text(0.97, 0.05, f"Avg PT: {mean_pt:.2f}",
                 transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

plt.tight_layout()
plt.savefig(f"{DATA_DIR}/fig5_three_shock_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] fig5_three_shock_trajectories.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 – Pass-Through Rate: Inconsistency Across Shocks
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5))
fig.suptitle(
    "Pass-Through Rate: Is It Consistent Across Shocks?\n"
    "(Pass-through = Fare % change / Oil % change from T0)",
    fontsize=13, fontweight="bold", y=0.98
)

# Left Panel: Oil Δ% vs Fare Δ% scatter (each dot = one month within shock window)
ax1 = axes[0]
for shock_name, cfg in SHOCKS.items():
    sdf = shock_dfs[shock_name]
    ax1.scatter(sdf.oil_chg_pct, sdf.fare_chg_pct,
                c=cfg["color"], s=70, alpha=0.8, label=cfg["label"],
                edgecolors="white", linewidth=0.7, zorder=3)
    # OLS per shock
    sub = sdf.dropna(subset=["oil_chg_pct", "fare_chg_pct"])
    if len(sub) >= 3:
        sl, ic, r, _, _ = stats.linregress(sub.oil_chg_pct, sub.fare_chg_pct)
        xl = np.linspace(sub.oil_chg_pct.min() - 2, sub.oil_chg_pct.max() + 2, 50)
        ax1.plot(xl, sl * xl + ic, color=cfg["color"], lw=1.8, ls="--")
        # Annotate slope
        mid_x = sub.oil_chg_pct.mean()
        mid_y = sl * mid_x + ic
        ax1.text(mid_x, mid_y + 2.0, f"slope={sl:.2f}\nr={r:.2f}",
                 fontsize=8.5, color=cfg["color"], ha="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=cfg["color"], alpha=0.85))

ax1.axhline(0, color="black", lw=0.7)
ax1.axvline(0, color="black", lw=0.7)
ax1.grid(True, linestyle=":", alpha=0.4, color="gray")
ax1.set_xlabel("Brent Crude % change from T0", fontsize=10, fontweight="bold")
ax1.set_ylabel("Avg Ticket Fare % change from T0", fontsize=10, fontweight="bold")
ax1.set_title("Oil Shock vs Fare Response (per shock period)\nEach dot = one month", fontsize=11, fontweight="bold", pad=10)
ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)

# Right Panel: Bar chart - summary pass-through at peak and final
ax2 = axes[1]
labels = [r["Shock"] for r in pt_summary]
pt_peak  = []
pt_final = []
for r in pt_summary:
    try:
        pt_peak.append(float(r["Pass-Through (peak)"]))
    except:
        pt_peak.append(0.0)
    try:
        pt_final.append(float(r["PT (final)"]))
    except:
        pt_final.append(0.0)

x = np.arange(len(labels))
w = 0.35
bars1 = ax2.bar(x - w/2, pt_peak,  w, label="At oil-price peak",
                color=[SHOCK_COLORS[l] for l in labels], alpha=0.9, edgecolor="white", linewidth=1.0)
bars2 = ax2.bar(x + w/2, pt_final, w, label="At end of window",
                color=[SHOCK_COLORS[l] for l in labels], alpha=0.45, edgecolor="white", linewidth=1.0, hatch="//")
ax2.axhline(0,   color="black", lw=0.7)
ax2.axhline(1.0, color="grey",  lw=0.9, ls=":", label="100% pass-through")
ax2.grid(True, axis="y", linestyle=":", alpha=0.4, color="gray")

for bar, v in zip(bars1, pt_peak):
    va = "bottom" if v >= 0 else "top"
    offset = 0.02 if v >= 0 else -0.05
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
             f"{v:.2f}", ha="center", va=va, fontsize=9, fontweight="bold", color="#222222")
for bar, v in zip(bars2, pt_final):
    va = "bottom" if v >= 0 else "top"
    offset = 0.02 if v >= 0 else -0.05
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
             f"{v:.2f}", ha="center", va=va, fontsize=9, fontweight="bold", color="#333333")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10, fontweight="medium")
ax2.set_ylabel("Pass-Through Rate", fontsize=10, fontweight="bold")
ax2.set_title("Cumulative Pass-Through at Peak vs End\n(1.0 = 100% of oil shock passed to fare)", fontsize=11, fontweight="bold", pad=10)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{DATA_DIR}/fig6_pass_through_rates.png", dpi=200, bbox_inches="tight")
plt.close()
print("[Saved] fig6_pass_through_rates.png")





# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 – Iran Crisis Deep-Dive vs Prior Shocks
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.5, 10.5))
gs  = gridspec.GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.50)
fig.suptitle(
    "US-Iran War (2025-26): Deep-Dive vs COVID-19 & Russia-Ukraine War\n"
    "What makes this crisis structurally different?",
    fontsize=14, fontweight="bold", y=0.98
)

# Shared legend handles for top row
top_handles = []

# (0,0-2) Absolute Brent crude levels
ax1 = fig.add_subplot(gs[0, 0:2])
for shock_name, cfg in SHOCKS.items():
    sdf = shock_dfs[shock_name]
    line = ax1.plot(sdf.t_offset, sdf["brent_crude_usd_barrel"],
                    color=cfg["color"], lw=2.5, marker="o", ms=4, label=cfg["label"])
    if len(top_handles) < len(SHOCKS):
        top_handles.append(line[0])

ax1.axhline(0, color="black", lw=0.3)
ax1.grid(True, linestyle=":", alpha=0.4, color="gray")
ax1.set_xlabel("Months from T0", fontsize=9.5, fontweight="bold")
ax1.set_ylabel("Brent Crude ($/bbl)", fontsize=9.5, fontweight="bold", labelpad=8)
ax1.set_title("Absolute Oil Price Level", fontsize=10.5, fontweight="bold", pad=8)
ax1.set_ylim(-5, 185)

# (0,2-4) Absolute total fare
ax2 = fig.add_subplot(gs[0, 2:4])
for shock_name, cfg in SHOCKS.items():
    sdf = shock_dfs[shock_name]
    ax2.plot(sdf.t_offset, sdf["avg_total_fare"],
             color=cfg["color"], lw=2.5, marker="o", ms=4, label=cfg["label"])
ax2.grid(True, linestyle=":", alpha=0.4, color="gray")
ax2.set_xlabel("Months from T0", fontsize=9.5, fontweight="bold")
ax2.set_ylabel("Avg Total Ticket Fare (USD)", fontsize=9.5, fontweight="bold", labelpad=8)
ax2.set_title("Absolute Ticket Price Level", fontsize=10.5, fontweight="bold", pad=8)
ax2.set_ylim(200, 3200)

# (0,4-6) Fuel surcharge (absolute)
ax3 = fig.add_subplot(gs[0, 4:6])
for shock_name, cfg in SHOCKS.items():
    sdf = shock_dfs[shock_name]
    ax3.plot(sdf.t_offset, sdf["avg_fuel_surcharge"],
             color=cfg["color"], lw=2.5, marker="o", ms=4, label=cfg["label"])
ax3.grid(True, linestyle=":", alpha=0.4, color="gray")
ax3.set_xlabel("Months from T0", fontsize=9.5, fontweight="bold")
ax3.set_ylabel("Avg Fuel Surcharge (USD)", fontsize=9.5, fontweight="bold", labelpad=8)
ax3.set_title("Absolute Fuel Surcharge Level", fontsize=10.5, fontweight="bold", pad=8)
ax3.set_ylim(-10, 560)

# Add single clean shared legend above top row subplots
fig.legend(handles=top_handles, loc="upper center", bbox_to_anchor=(0.5, 0.93),
           ncol=3, fontsize=10, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

# Bottom Row: Financial impact (0:3 and 3:6)
shock_fin = {
    "COVID-19":     {"start": "2020-03", "end": "2020-12"},
    "Ukraine War":  {"start": "2022-03", "end": "2022-12"},
    "Iran Crisis":  {"start": "2025-12", "end": "2026-06"},
}
baseline_fin = financial[
    (financial.month >= "2019-01") & (financial.month <= "2019-12")
]["fuel_cost_pct_revenue"].mean()

fin_vals = {"Baseline\n(2019)": baseline_fin}
for label2, rng in shock_fin.items():
    sub = financial[(financial.month >= rng["start"]) & (financial.month <= rng["end"])]
    fin_vals[label2] = sub["fuel_cost_pct_revenue"].mean()

# (1, 0:3) Financial impact: fuel cost % of revenue during each shock
ax5 = fig.add_subplot(gs[1, 0:3])
cols5 = ["#4CAF50", "#9C27B0", "#FF5722", "#F44336"]
bars5 = ax5.bar(list(fin_vals.keys()), list(fin_vals.values()), color=cols5, alpha=0.88, edgecolor="white", linewidth=1.2)
for bar, v in zip(bars5, fin_vals.values()):
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222222")
ax5.grid(True, axis="y", linestyle=":", alpha=0.4, color="gray")
ax5.set_ylabel("Avg Fuel Cost % of Revenue", fontsize=9.5, fontweight="bold", labelpad=8)
ax5.set_title("Airline Cost Pressure (Fuel Cost as % of Revenue)", fontsize=11, fontweight="bold", pad=10)
ax5.set_ylim(0, 42)

# (1, 3:6) Profit margin during each shock
ax6 = fig.add_subplot(gs[1, 3:6])
margin_vals = {"Baseline\n(2019)": financial[(financial.month >= "2019-01") & (financial.month <= "2019-12")]["profit_margin_pct"].mean()}
for label2, rng in shock_fin.items():
    sub = financial[(financial.month >= rng["start"]) & (financial.month <= rng["end"])]
    margin_vals[label2] = sub["profit_margin_pct"].mean()

cols6 = ["#4CAF50", "#9C27B0", "#FF5722", "#F44336"]
bars6 = ax6.bar(list(margin_vals.keys()), list(margin_vals.values()), color=cols6, alpha=0.88, edgecolor="white", linewidth=1.2)
for bar, v in zip(bars6, margin_vals.values()):
    if v >= 0:
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222222")
    else:
        # Place label inside negative bar near top edge to avoid overlapping x-axis labels
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="white")

ax6.axhline(0, color="black", lw=0.8, ls="--")
ax6.grid(True, axis="y", linestyle=":", alpha=0.4, color="gray")
ax6.set_ylabel("Avg Profit Margin (%)", fontsize=9.5, fontweight="bold", labelpad=8)
ax6.set_title("Airline Profitability (Net Profit Margin by Shock)", fontsize=11, fontweight="bold", pad=10)
ax6.set_ylim(-36, 14)

plt.tight_layout(rect=[0.01, 0, 0.99, 0.91])
plt.savefig(f"{DATA_DIR}/fig7_iran_deepdive.png", dpi=200, bbox_inches="tight")
plt.close()
print("[Saved] fig7_iran_deepdive.png")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("KEY FINDINGS — QUESTION 2")
print("=" * 65)
print("  CONSISTENCY OF PASS-THROUGH:")
covid_pt  = [r for r in pt_summary if r["Shock"] == "COVID-19"][0]
ukr_pt    = [r for r in pt_summary if r["Shock"] == "Ukraine War"][0]
iran_pt   = [r for r in pt_summary if r["Shock"] == "Iran Crisis"][0]
def fmt_pt(v):
    try: return f"{float(v):.3f}"
    except: return str(v)
print(f"    COVID-19   : {fmt_pt(covid_pt['Pass-Through (peak)'])} at peak  |  {fmt_pt(covid_pt['PT (final)'])} at end")
print(f"    Ukraine War: {fmt_pt(ukr_pt['Pass-Through (peak)'])} at peak  |  {fmt_pt(ukr_pt['PT (final)'])} at end")
print(f"    Iran Crisis: {fmt_pt(iran_pt['Pass-Through (peak)'])} at peak  |  {fmt_pt(iran_pt['PT (final)'])} at end")
print()
print("  SPEED (best lag in months):")
for label, res in speed_results.items():
    print(f"    {label:15s}: {res['best_lag']} month(s)  (r = {res['best_r']:.3f})")
print()
print("  IRAN CRISIS DIFFERENCES vs PRIOR SHOCKS:")
print(f"    - Oil spike magnitude: +{iran_pt['Peak Oil Chg%']:.1f}%  (Ukraine: {ukr_pt['Peak Oil Chg%']:+.1f}%)")
print(f"    - Fare spike magnitude: +{iran_pt['Peak Fare Chg%']:.1f}%  (Ukraine: {ukr_pt['Peak Fare Chg%']:+.1f}%)")
print(f"    - Pass-through at peak: {fmt_pt(iran_pt['Pass-Through (peak)'])}  (Ukraine: {fmt_pt(ukr_pt['Pass-Through (peak)'])})")
print(f"    - Strait of Hormuz disrupted: 7 months in dataset")
print()
print("  Figures saved:")
for f in ["fig5_three_shock_trajectories.png", "fig6_pass_through_rates.png",
          "fig7_iran_deepdive.png"]:
    print(f"    {f}")
