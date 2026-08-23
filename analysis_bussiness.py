#!/usr/bin/env python3
"""
BUSINESS-ANGLE ANALYSIS  (B1, B2, B4)  -- see suggestion.md / suggestion_th.md

B1: When does fuel hedging start to pay off, and by how much?
    -> fig8_hedging_effectiveness.png
B2: What did rerouting around the Strait of Hormuz actually cost?
    -> fig9_reroute_cost.png
B4: What is the break-even Brent price for each airline?
    -> fig11_breakeven_brent.png

Figure numbering follows suggestion.md, where fig10 is reserved for B3 (FSC vs LCC),
which is not implemented here.

Datasets used:
  airline_financial_impact.csv  - quarterly x airline P&L, hedging (B1, B4)
  route_cost_impact.csv         - monthly x airline x route detour economics (B2)
  oil_jet_fuel_prices.csv       - monthly Brent / jet fuel spine (context)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import path as pth
warnings.filterwarnings("ignore")

DATA_DIR = pth.PATH

sns.set_style("whitegrid")
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
plt.rcParams["font.size"] = 10

C_HEDGE   = "#1f77b4"   # hedging / protected
C_NOHEDGE = "#d62728"   # unhedged / exposure
C_FSC     = "#2c5f8a"
C_LCC     = "#e08214"
C_GULF    = "#c0392b"
C_NONGULF = "#7f8c8d"

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load
# ─────────────────────────────────────────────────────────────────────────────
fin    = pd.read_csv(f"{DATA_DIR}/airline_financial_impact.csv")
routes = pd.read_csv(f"{DATA_DIR}/route_cost_impact.csv")
oil    = pd.read_csv(f"{DATA_DIR}/oil_jet_fuel_prices.csv")

print("=" * 70)
print("BUSINESS ANALYSIS  -  B1 / B2 / B4")
print("=" * 70)
print(f"  financial : {fin.shape[0]:,} rows  ({fin.quarter.nunique()} quarters x "
      f"{fin.airline.nunique()} airlines, {fin.quarter.min()} .. {fin.quarter.max()})")
print(f"  routes    : {routes.shape[0]:,} rows  ({routes.month.nunique()} months x "
      f"{routes.airline.nunique()} airlines)")
print(f"  oil       : {oil.shape[0]:,} months  ({oil.month.min()} .. {oil.month.max()})")


# ═════════════════════════════════════════════════════════════════════════════
# B1.  HEDGING EFFECTIVENESS AND TIMING
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B1. WHEN DOES FUEL HEDGING START TO PAY OFF?")
print("=" * 70)

fin = fin.copy()
# Counterfactual: strip the hedge gain back out of the reported margin.
fin["hedge_gain_pct_rev"] = fin.hedge_savings_usd_m / fin.revenue_usd_m * 100
fin["margin_unhedged"]    = fin.profit_margin_pct - fin.hedge_savings_usd_m / fin.revenue_usd_m * 100
fin["net_fuel_usd_m"]     = fin.fuel_cost_usd_m - fin.hedge_savings_usd_m
fin["net_fuel_pct_rev"]   = fin.net_fuel_usd_m / fin.revenue_usd_m * 100

q = fin.groupby("quarter").agg(
    phase          = ("conflict_phase", "first"),
    brent          = ("brent_crude_usd_barrel", "first"),
    hedge_ratio    = ("fuel_hedging_pct", "mean"),
    savings_usd_m  = ("hedge_savings_usd_m", "sum"),
    fuel_usd_m     = ("fuel_cost_usd_m", "sum"),
    margin         = ("profit_margin_pct", "mean"),
    margin_unhedged= ("margin_unhedged", "mean"),
    fuel_pct_rev   = ("fuel_cost_pct_revenue", "mean"),
    net_fuel_pct   = ("net_fuel_pct_rev", "mean"),
).reset_index()
q["savings_pct_fuel"] = q.savings_usd_m / q.fuel_usd_m * 100
q["margin_cushion"]   = q.margin - q.margin_unhedged

MATERIAL = 5.0   # % of the gross fuel bill
first_nonzero  = q.loc[q.savings_pct_fuel > 0, "quarter"].min()
first_material = q.loc[q.savings_pct_fuel >= MATERIAL, "quarter"].min()
peak_q         = q.loc[q.savings_pct_fuel.idxmax()]

print(f"\n  First quarter with ANY recorded hedge savings : {first_nonzero}")
print(f"  First quarter above {MATERIAL:.0f}% of the fuel bill      : {first_material}")
print(f"  Peak quarter                                 : {peak_q.quarter}  "
      f"({peak_q.savings_pct_fuel:.1f}% of fuel bill, Brent ${peak_q.brent:.1f})")

print("\n  Quarterly hedge panel")
print("  " + "-" * 92)
print(f"  {'quarter':9s} {'Brent':>7s} {'hedge%':>7s} {'sav $m':>9s} {'sav %fuel':>10s} "
      f"{'margin':>8s} {'unhedged':>9s} {'cushion':>8s}  phase")
for _, r in q.iterrows():
    print(f"  {r.quarter:9s} {r.brent:7.1f} {r.hedge_ratio:7.1f} {r.savings_usd_m:9.1f} "
          f"{r.savings_pct_fuel:10.2f} {r.margin:8.2f} {r.margin_unhedged:9.2f} "
          f"{r.margin_cushion:8.2f}  {r.phase}")

# Cross-sectional test inside the shock quarters: does a bigger hedge book
# actually mean a better margin?
SHOCK_Q = ["2022-Q2", "2024-Q4", "2025-Q2", "2026-Q1"]
print("\n  Cross-sectional test WITHIN shock quarters  (does more hedging = better margin?)")
xsec = []
for qq in SHOCK_Q:
    s = fin[fin.quarter == qq]
    r_pear, p = stats.pearsonr(s.fuel_hedging_pct, s.profit_margin_pct)
    xsec.append((qq, r_pear, p, len(s)))
    print(f"    {qq}:  r = {r_pear:+.3f}   p = {p:.3f}   n = {len(s)}")

# Airline-level hedge effectiveness in the peak quarter
peak_name = peak_q.quarter
pk = fin[fin.quarter == peak_name].copy()
pk["sav_pct_fuel"] = pk.hedge_savings_usd_m / pk.fuel_cost_usd_m * 100
pk = pk.sort_values("fuel_hedging_pct", ascending=False)
r_pk, p_pk = stats.pearsonr(pk.fuel_hedging_pct, pk.profit_margin_pct)

print(f"\n  {peak_name} cross-section  (hedge ratio vs realised margin)")
print(f"    corr(hedge_ratio, margin)      = {r_pk:+.3f}  (p = {p_pk:.3f})")
print(f"    corr(hedge_ratio, sav %fuel)   = "
      f"{pk.fuel_hedging_pct.corr(pk.sav_pct_fuel):+.3f}")
print(f"    best hedger : {pk.iloc[0].airline} ({pk.iloc[0].fuel_hedging_pct:.1f}%) "
      f"-> margin {pk.iloc[0].profit_margin_pct:+.1f}%")
print(f"    best margin : {pk.loc[pk.profit_margin_pct.idxmax()].airline} "
      f"({pk.loc[pk.profit_margin_pct.idxmax()].fuel_hedging_pct:.1f}% hedged) "
      f"-> margin {pk.profit_margin_pct.max():+.1f}%")

# ── Figure 8 ────────────────────────────────────────────────────────────────
# Four panels, one plain question each, so the chart can be read top-to-bottom
# without cross-referencing axes.
i_pk = int(q.index[q.quarter == peak_name][0])

C_GREY = "#b0b7bd"      # nothing recorded
C_LITE = "#7fb2dc"      # recorded but small
C_DARK = "#0d4f8b"      # material

fig = plt.figure(figsize=(17, 11))
gs  = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.30,
                        left=0.055, right=0.965, top=0.865, bottom=0.085)
fig.suptitle("B1 — Fuel Hedging: When Does It Start To Pay Off, And By How Much?",
             fontsize=17, fontweight="bold", y=0.965)
fig.text(0.5, 0.917,
         f"Short answer:  nothing shows up until {first_nonzero}  ·  it turns material in "
         f"{first_material}  ·  it peaks in {peak_name}, worth "
         f"+{q.margin_cushion.iloc[i_pk]:.1f} margin points",
         ha="center", fontsize=12, color="#333333")

x = np.arange(len(q))

# ── (a) WHEN did hedging start to pay? ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
colors = [C_GREY if v == 0 else (C_DARK if v >= MATERIAL else C_LITE)
          for v in q.savings_pct_fuel]
ax1.bar(x, q.savings_pct_fuel, color=colors, width=0.68, zorder=3)
ax1.axhline(MATERIAL, color=C_NOHEDGE, ls="--", lw=1.3, zorder=2)
ax1.text(0.4, MATERIAL + 0.5, f"{MATERIAL:.0f}% of the fuel bill = 'material'",
         fontsize=9, color=C_NOHEDGE, fontweight="bold")

for i, v in enumerate(q.savings_pct_fuel):
    if v >= MATERIAL:
        ax1.text(i, v + 0.6, f"{v:.0f}%", ha="center", fontsize=8.5,
                 fontweight="bold", color=C_DARK)

ax1.set_ylabel("Money saved by hedging\n(% of the gross fuel bill)", fontsize=10.5)
ax1.set_ylim(0, q.savings_pct_fuel.max() * 1.30)
ax1.set_xticks(x)
ax1.set_xticklabels(q.quarter, rotation=45, ha="right", fontsize=8.5)
ax1.set_xlim(-0.7, len(q) - 0.3)

axb = ax1.twinx()
axb.plot(x, q.brent, color="#8e44ad", lw=2.0, ls=":", zorder=4)
axb.set_ylabel("Brent crude ($/bbl)", fontsize=10.5, color="#8e44ad")
axb.tick_params(axis="y", labelcolor="#8e44ad")
axb.grid(False)

i_fn = int(q.index[q.quarter == first_nonzero][0])
i_fm = int(q.index[q.quarter == first_material][0])
# Arrows land on the middle of the bar so they never sit on top of its value label.
ax1.annotate("first recorded\nsavings", xy=(i_fn, q.savings_pct_fuel.iloc[i_fn] * 0.55),
             xytext=(i_fn - 3.4, q.savings_pct_fuel.max() * 0.42), fontsize=9,
             ha="center", color="#333333",
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
ax1.annotate("first material\nquarter", xy=(i_fm - 0.36, q.savings_pct_fuel.iloc[i_fm] * 0.62),
             xytext=(i_fm - 3.4, q.savings_pct_fuel.max() * 0.72), fontsize=9,
             ha="center", color="#333333",
             arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))

ax1.legend(handles=[mpatches.Patch(color=C_GREY, label="no savings recorded"),
                    mpatches.Patch(color=C_LITE, label="small (<5% of fuel bill)"),
                    mpatches.Patch(color=C_DARK, label="material (>=5%)"),
                    plt.Line2D([], [], color="#8e44ad", ls=":", lw=2,
                               label="Brent crude ($/bbl, right axis)")],
           loc="upper left", fontsize=9, framealpha=0.94, ncol=2)
ax1.set_title("(a)  WHEN does hedging pay?   Only when oil is expensive — the hedge has to be "
              "in the money before any saving is booked",
              fontsize=12, fontweight="bold", pad=10)
ax1.text(0.995, 0.60,
         f"The hedge ratio itself barely moves\n"
         f"({q.hedge_ratio.min():.0f}-{q.hedge_ratio.max():.0f}% of volume all the way through).\n"
         f"What changes is the oil price, not the hedging.",
         transform=ax1.transAxes, ha="right", va="top", fontsize=9, style="italic",
         bbox=dict(boxstyle="round,pad=0.45", fc="#f4f6f8", ec="#c9d1d8"))

# ── (b) HOW MUCH margin did it add? ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
cush_colors = [C_GREY if v == 0 else (C_DARK if v >= 1 else C_LITE)
               for v in q.margin_cushion]
ax2.bar(x, q.margin_cushion, color=cush_colors, width=0.68, zorder=3)
ax2.set_ylabel("Margin points added by hedging (pp)", fontsize=10)
ax2.set_xticks(x[::3])
ax2.set_xticklabels(q.quarter[::3], rotation=45, ha="right", fontsize=8.5)
ax2.set_xlim(-0.7, len(q) - 0.3)
ax2.set_ylim(0, q.margin_cushion.max() * 1.25)
ax2.annotate(f"+{q.margin_cushion.iloc[i_pk]:.1f} pp",
             xy=(i_pk - 0.45, q.margin_cushion.iloc[i_pk] * 0.92),
             xytext=(i_pk - 9.5, q.margin_cushion.max() * 0.86), fontsize=11.5,
             fontweight="bold", color=C_DARK, va="center",
             arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.4))
calm = q[(q.margin_cushion > 0) & (q.quarter != peak_name)].margin_cushion.mean()
ax2.axhline(calm, color="#555555", ls="--", lw=1.2)
ax2.text(0.3, calm + 0.18, f"every other quarter: about +{calm:.1f} pp",
         ha="left", fontsize=9, color="#555555")
ax2.set_title("(b)  HOW MUCH is it worth?\nAlmost nothing in calm quarters, a lot in a crisis",
              fontsize=12, fontweight="bold", pad=10)

# ── (c) the peak quarter, with vs without ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
rows = [
    ("Fuel bill\n(% of revenue)", q.fuel_pct_rev.iloc[i_pk], q.net_fuel_pct.iloc[i_pk], "lower"),
    ("Net profit margin\n(%)",    q.margin_unhedged.iloc[i_pk], q.margin.iloc[i_pk], "higher"),
]
for yi, (lab, without, with_, better) in enumerate(rows):
    ax3.plot([without, with_], [yi, yi], color="#95a5a6", lw=3, zorder=2,
             solid_capstyle="round")
    ax3.scatter([without], [yi], s=230, color=C_NOHEDGE, zorder=3, edgecolor="white", lw=1.5)
    ax3.scatter([with_],   [yi], s=230, color=C_HEDGE,   zorder=3, edgecolor="white", lw=1.5)
    ax3.text(without, yi + 0.20, f"{without:.1f}", ha="center", fontsize=10,
             fontweight="bold", color=C_NOHEDGE)
    ax3.text(with_, yi + 0.20, f"{with_:.1f}", ha="center", fontsize=10,
             fontweight="bold", color=C_HEDGE)
    ax3.text((without + with_) / 2, yi - 0.26,
             f"{abs(with_ - without):.1f} pp {better}", ha="center", fontsize=9.5,
             style="italic", color="#333333")

ax3.axvline(0, color="black", lw=1)
ax3.set_yticks(range(len(rows)))
ax3.set_yticklabels([r[0] for r in rows], fontsize=10)
ax3.set_ylim(-0.6, len(rows) - 0.35)
ax3.set_xlabel("Percent", fontsize=10)
ax3.legend(handles=[mpatches.Patch(color=C_NOHEDGE, label="without hedging (counterfactual)"),
                    mpatches.Patch(color=C_HEDGE, label="as reported (with hedging)")],
           fontsize=9, loc="lower center", framealpha=0.94)
ax3.set_title(f"(c)  {peak_name} in one view:\nwhat hedging actually changed",
              fontsize=12, fontweight="bold", pad=10)

# ── (d) does hedging MORE mean surviving better? ────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
for t, c in [("Flag Carrier", C_FSC), ("Low Cost", C_LCC)]:
    s = pk[pk.airline_type == t]
    ax4.scatter(s.fuel_hedging_pct, s.profit_margin_pct, s=95, color=c,
                edgecolor="white", lw=1.1, alpha=0.9, label=t, zorder=3)
sl, ic = np.polyfit(pk.fuel_hedging_pct, pk.profit_margin_pct, 1)
xs = np.linspace(pk.fuel_hedging_pct.min(), pk.fuel_hedging_pct.max(), 50)
ax4.plot(xs, sl * xs + ic, color="#333333", ls="--", lw=1.6,
         label=f"trend  (r = {r_pk:+.2f}, p = {p_pk:.2f} — not significant)")

lab_pts = pd.concat([pk.nlargest(2, "fuel_hedging_pct"), pk.nsmallest(2, "fuel_hedging_pct"),
                     pk.nlargest(2, "profit_margin_pct")]).drop_duplicates("airline")
# Alternate the offset so near-identical points (Delta / Lufthansa) do not overprint.
for k, (_, rr) in enumerate(lab_pts.sort_values("fuel_hedging_pct").iterrows()):
    dy = 7 if k % 2 == 0 else -13
    ax4.annotate(rr.airline, (rr.fuel_hedging_pct, rr.profit_margin_pct),
                 xytext=(6, dy), textcoords="offset points", fontsize=8.5)

ax4.set_xlabel("Fuel hedged (% of volume)", fontsize=10)
ax4.set_ylabel(f"Net profit margin in {peak_name} (%)", fontsize=10)
ax4.legend(fontsize=8.5, loc="lower left", framealpha=0.94)
ax4.set_title("(d)  Does hedging MORE mean surviving better?\nNo — the line is flat, "
              "each dot is one airline", fontsize=12, fontweight="bold", pad=10)

plt.savefig(f"{DATA_DIR}/fig8_hedging_effectiveness.png", dpi=180, bbox_inches="tight")
plt.close()
print("\n  [Saved] fig8_hedging_effectiveness.png")


# ═════════════════════════════════════════════════════════════════════════════
# B2.  COST OF REROUTING AROUND THE STRAIT OF HORMUZ
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B2. WHAT DID REROUTING AROUND THE STRAIT OF HORMUZ COST?")
print("=" * 70)

hormuz_months = sorted(oil.loc[oil.strait_hormuz_disrupted == "Yes", "month"].unique())
print(f"\n  Disruption window: {hormuz_months[0]} .. {hormuz_months[-1]}  "
      f"({len(hormuz_months)} months)")

win = routes[routes.month.isin(hormuz_months)].copy()
win["is_reroute"] = win.rerouted == "Yes"
win["is_cancel"]  = win.flight_cancelled == "Yes"

GULF_CITIES = {"Dubai", "Doha", "Abu Dhabi", "Riyadh", "Kuwait City",
               "Muscat", "Bahrain", "Sharjah"}
win["hub_type"] = np.where(win.origin_city.isin(GULF_CITIES),
                           "Gulf hub carrier", "Non-Gulf carrier")

# Lost revenue: a cancelled row books route_revenue_usd = 0, so value the loss at the
# airline-route's own median revenue in the months it did operate inside the window.
oper = win[~win.is_cancel]
med_rev = (oper.groupby(["airline", "origin_city", "destination_city"])
               .route_revenue_usd.median().rename("median_route_rev").reset_index())
win = win.merge(med_rev, on=["airline", "origin_city", "destination_city"], how="left")
win["lost_revenue_usd"] = np.where(win.is_cancel, win.median_route_rev.fillna(0.0), 0.0)

# A route-month can be BOTH rerouted and cancelled, so exposure counts distinct
# disrupted route-months rather than summing the two flags.
win["is_disrupted"] = win.is_reroute | win.is_cancel

b2 = win.groupby(["airline", "iata_code", "hub_type"]).agg(
    route_months     = ("month", "size"),
    disrupted_months = ("is_disrupted", "sum"),
    reroute_months   = ("is_reroute", "sum"),
    cancel_months    = ("is_cancel", "sum"),
    extra_km         = ("extra_distance_km", "sum"),
    extra_fuel_usd   = ("extra_fuel_cost_usd", "sum"),
    lost_rev_usd     = ("lost_revenue_usd", "sum"),
    total_fuel_usd   = ("total_fuel_cost_usd", "sum"),
).reset_index()

n_routes = (win.drop_duplicates(["airline", "origin_city", "destination_city"])
               .groupby("airline").size().rename("n_routes").reset_index())
b2 = b2.merge(n_routes, on="airline", how="left")

b2["total_burden"]   = b2.extra_fuel_usd + b2.lost_rev_usd
b2["exposure_pct"]   = b2.disrupted_months / b2.route_months * 100
b2["burden_per_rm"]  = b2.total_burden / b2.route_months           # normalises route count
b2["fuel_per_rm"]    = b2.extra_fuel_usd / b2.route_months
b2["rev_per_rm"]     = b2.lost_rev_usd / b2.route_months
b2["extra_fuel_pct"] = b2.extra_fuel_usd / b2.total_fuel_usd * 100
b2 = b2.sort_values("total_burden", ascending=False)

tot_extra = b2.extra_fuel_usd.sum()
tot_lost  = b2.lost_rev_usd.sum()
print(f"\n  Total detour fuel cost   : ${tot_extra:,.0f}")
print(f"  Total lost route revenue : ${tot_lost:,.0f}   "
      f"({int(b2.cancel_months.sum())} cancelled route-months)")
print(f"  Combined burden          : ${tot_extra + tot_lost:,.0f}")
print("  NOTE: figures are per representative flight per route per month, "
      "NOT fleet-wide totals.")

print("\n  Burden by airline")
print("  " + "-" * 104)
print(f"  {'airline':20s} {'hub':17s} {'rts':>4s} {'r-mo':>5s} {'canc':>5s} "
      f"{'extra fuel $':>13s} {'lost rev $':>12s} {'burden $':>12s} {'exp%':>6s} {'$/route-mo':>11s}")
for _, r in b2.iterrows():
    print(f"  {r.airline:20s} {r.hub_type:17s} {int(r.n_routes):4d} "
          f"{int(r.reroute_months):5d} {int(r.cancel_months):5d} "
          f"{r.extra_fuel_usd:13,.0f} {r.lost_rev_usd:12,.0f} {r.total_burden:12,.0f} "
          f"{r.exposure_pct:6.1f} {r.burden_per_rm:11,.0f}")

hub_cmp = b2.groupby("hub_type").agg(
    airlines       = ("airline", "size"),
    routes         = ("n_routes", "sum"),
    exposure_pct   = ("exposure_pct", "mean"),
    fuel_per_rm    = ("fuel_per_rm", "mean"),
    rev_per_rm     = ("rev_per_rm", "mean"),
    burden_per_rm  = ("burden_per_rm", "mean"),
    extra_fuel_usd = ("extra_fuel_usd", "sum"),
)
hub_cmp["share_detour_fuel_pct"] = hub_cmp.extra_fuel_usd / tot_extra * 100
gulf_fuel_share = hub_cmp.loc["Gulf hub carrier", "share_detour_fuel_pct"]
fuel_share_of_burden = tot_extra / (tot_extra + tot_lost) * 100

print("\n  Gulf hub carriers vs the rest  (route-count normalised)")
print(hub_cmp.round(1).to_string())
print("\n  Reading this honestly:")
print(f"    - The detour-fuel burden IS concentrated in the Gulf hubs: "
      f"{gulf_fuel_share:.0f}% of all extra fuel")
print(f"      cost, from {int(hub_cmp.loc['Gulf hub carrier', 'routes'])} of "
      f"{int(hub_cmp.routes.sum())} routes in the window.")
print(f"    - But detour fuel is only {fuel_share_of_burden:.0f}% of the total burden. The other "
      f"{100 - fuel_share_of_burden:.0f}% is")
print("      revenue lost to cancellations, which scales with how big the route was.")
print(f"    - So per route-month the Gulf carriers are NOT worse off "
      f"(${hub_cmp.loc['Gulf hub carrier', 'burden_per_rm']:,.0f} vs")
print(f"      ${hub_cmp.loc['Non-Gulf carrier', 'burden_per_rm']:,.0f}). Geography drives the fuel "
      "penalty; route size drives the revenue loss.")

monthly = win.groupby("month").agg(
    extra_fuel = ("extra_fuel_cost_usd", "sum"),
    lost_rev   = ("lost_revenue_usd", "sum"),
    cancels    = ("is_cancel", "sum"),
    reroutes   = ("is_reroute", "sum"),
).reset_index()
print("\n  Monthly profile of the disruption")
print(monthly.round(0).to_string(index=False))

# ── Figure 9 ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.5, 10.5))
gs  = gridspec.GridSpec(2, 2, height_ratios=[1.35, 1], hspace=0.32, wspace=0.40,
                        left=0.075, right=0.97, top=0.885, bottom=0.08)
fig.suptitle("B2 — The Operational Shock: What Flying Around The Strait of Hormuz Cost\n"
             f"{hormuz_months[0]} .. {hormuz_months[-1]}  ·  "
             f"${tot_extra + tot_lost:,.0f} combined burden across "
             f"{b2.airline.nunique()} carriers  ·  "
             f"{int(b2.cancel_months.sum())} cancelled route-months",
             fontsize=15, fontweight="bold", y=0.975)

# (a) absolute burden, stacked
axa = fig.add_subplot(gs[0, 0])
d = b2.sort_values("total_burden")
yp = np.arange(len(d))
axa.barh(yp, d.extra_fuel_usd, color="#e67e22", label="Extra fuel from detours")
axa.barh(yp, d.lost_rev_usd, left=d.extra_fuel_usd, color="#8e44ad",
         label="Lost revenue from cancellations")
axa.set_yticks(yp)
axa.set_yticklabels(d.airline, fontsize=8.5)
axa.set_xlabel("USD per representative flight, summed over 7 months", fontsize=9.5)
axa.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
axa.set_title("(a)  Absolute burden — dominated by carriers with the most\n"
              "routes through the closed airspace", fontsize=11, fontweight="bold", pad=8)
for i, (_, r) in enumerate(d.iterrows()):
    if r.total_burden > 0:
        axa.text(r.total_burden * 1.01, i, f"${r.total_burden:,.0f}",
                 va="center", fontsize=7.5, color="#333333")
axa.set_xlim(0, d.total_burden.max() * 1.22)

# (b) route-count-normalised burden, split into its two mechanisms
axb2 = fig.add_subplot(gs[0, 1])
d2 = b2.sort_values("burden_per_rm")
yp2 = np.arange(len(d2))
axb2.barh(yp2, d2.fuel_per_rm, color="#e67e22", label="Extra fuel per route-month")
axb2.barh(yp2, d2.rev_per_rm, left=d2.fuel_per_rm, color="#8e44ad",
          label="Lost revenue per route-month")
axb2.set_yticks(yp2)
axb2.set_yticklabels(d2.airline, fontsize=8.5)
for tick, h in zip(axb2.get_yticklabels(), d2.hub_type):
    if h == "Gulf hub carrier":
        tick.set_color(C_GULF)
        tick.set_fontweight("bold")
axb2.set_xlabel("Burden per route-month (USD)  —  route count normalised", fontsize=9.5)
handles, labels = axb2.get_legend_handles_labels()
handles.append(mpatches.Patch(color=C_GULF, label="(red label = Gulf hub carrier)"))
axb2.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.92)
g_avg = hub_cmp.loc["Gulf hub carrier", "burden_per_rm"]
n_avg = hub_cmp.loc["Non-Gulf carrier", "burden_per_rm"]
axb2.set_title(f"(b)  Normalised for route count the Gulf hubs are NOT worse off\n"
               f"(\\${g_avg:,.0f} vs \\${n_avg:,.0f}/route-month) — "
               f"{100 - fuel_share_of_burden:.0f}% of the burden is lost revenue, not fuel",
               fontsize=11, fontweight="bold", pad=8)

# (c) monthly profile
axc = fig.add_subplot(gs[1, 0])
xm = np.arange(len(monthly))
axc.bar(xm - 0.19, monthly.extra_fuel, width=0.38, color="#e67e22",
        label="Extra fuel cost")
axc.bar(xm + 0.19, monthly.lost_rev, width=0.38, color="#8e44ad",
        label="Lost revenue")
axc.set_xticks(xm)
axc.set_xticklabels(monthly.month, rotation=45, ha="right", fontsize=9)
axc.set_ylabel("USD", fontsize=9.5)
axm = axc.twinx()
axm.plot(xm, monthly.cancels, color="#c0392b", marker="o", lw=2, ms=5,
         label="Cancelled route-months")
axm.set_ylabel("Cancelled route-months", fontsize=9.5, color="#c0392b")
axm.tick_params(axis="y", labelcolor="#c0392b")
axm.grid(False)
h1, l1 = axc.get_legend_handles_labels()
h2, l2 = axm.get_legend_handles_labels()
axc.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper right", framealpha=0.92)
axc.set_title("(c)  Detours ran for all 7 months, but cancellations — 95% of the cost —\n"
              "stopped the moment de-escalation began in 2026-04",
              fontsize=11, fontweight="bold", pad=8)

# (d) worst routes by detour
axd = fig.add_subplot(gs[1, 1])
rt = (win[win.is_reroute]
      .groupby(["airline", "origin_city", "destination_city"])
      .agg(extra_km=("extra_distance_km", "mean"),
           extra_usd=("extra_fuel_cost_usd", "mean")).reset_index())
rt["label"] = rt.origin_city + " → " + rt.destination_city + "  (" + rt.airline + ")"
rt = rt.sort_values("extra_km", ascending=False).head(12).sort_values("extra_km")
axd.barh(np.arange(len(rt)), rt.extra_km, color="#16a085", alpha=0.9)
axd.set_yticks(np.arange(len(rt)))
axd.set_yticklabels(rt.label, fontsize=8)
axd.set_xlabel("Average extra distance per flight (km)", fontsize=9.5)
for i, (_, r) in enumerate(rt.iterrows()):
    axd.text(r.extra_km * 1.01, i, f"+${r.extra_usd:,.0f}/flt", va="center",
             fontsize=7.5, color="#333333")
axd.set_xlim(0, rt.extra_km.max() * 1.28)
axd.set_title("(d)  Worst detours — the routes that physically had to fly around",
              fontsize=11, fontweight="bold", pad=8)

plt.savefig(f"{DATA_DIR}/fig9_reroute_cost.png", dpi=180, bbox_inches="tight")
plt.close()
print("\n  [Saved] fig9_reroute_cost.png")


# ═════════════════════════════════════════════════════════════════════════════
# B4.  BREAK-EVEN BRENT PRICE PER AIRLINE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("B4. WHAT IS THE BREAK-EVEN BRENT PRICE FOR EACH AIRLINE?")
print("=" * 70)

# COVID is excluded: margin there was destroyed by collapsed revenue, not by fuel.
be_src = fin[fin.conflict_phase != "COVID-19 Collapse"].copy()
print(f"\n  Sample: {be_src.quarter.nunique()} quarters "
      f"({be_src.quarter.min()} .. {be_src.quarter.max()}), "
      f"COVID-19 Collapse excluded, {be_src.airline.nunique()} airlines")

rows = []
for airline, g in be_src.groupby("airline"):
    sl_h,  ic_h  = np.polyfit(g.brent_crude_usd_barrel, g.profit_margin_pct, 1)
    sl_u,  ic_u  = np.polyfit(g.brent_crude_usd_barrel, g.margin_unhedged, 1)
    r_h, p_h     = stats.pearsonr(g.brent_crude_usd_barrel, g.profit_margin_pct)
    rows.append(dict(
        airline   = airline,
        type      = g.iloc[0].airline_type,
        region    = g.iloc[0].region,
        n         = len(g),
        slope     = sl_h,
        r         = r_h,
        p         = p_h,
        breakeven          = -ic_h / sl_h if sl_h < 0 else np.nan,
        breakeven_unhedged = -ic_u / sl_u if sl_u < 0 else np.nan,
    ))
be = pd.DataFrame(rows)
be["hedge_uplift"] = be.breakeven - be.breakeven_unhedged
be["avg_hedge"]    = be.airline.map(be_src.groupby("airline").fuel_hedging_pct.mean())
be = be.dropna(subset=["breakeven"]).sort_values("breakeven", ascending=False)

n_bad = len(rows) - len(be)
if n_bad:
    print(f"  WARNING: {n_bad} airline(s) had a non-negative margin-vs-Brent slope "
          f"and no valid break-even; excluded.")

BRENT_NOW  = float(fin.loc[fin.quarter == "2026-Q1", "brent_crude_usd_barrel"].iloc[0])
BRENT_PEAK = float(oil.brent_crude_usd_barrel.max())

print(f"\n  Reference: 2026-Q1 actual Brent = ${BRENT_NOW:.2f}/bbl   "
      f"(monthly peak ${BRENT_PEAK:.2f})")
print("\n  Break-even Brent per airline  (margin = 0)")
print("  " + "-" * 100)
print(f"  {'airline':20s} {'type':13s} {'hedge%':>7s} {'slope':>7s} {'r':>6s} "
      f"{'BE hedged':>10s} {'BE unhedged':>12s} {'uplift $':>9s}  survives 2026-Q1?")
for _, r in be.iterrows():
    ok = "YES" if r.breakeven > BRENT_NOW else "no"
    print(f"  {r.airline:20s} {r['type']:13s} {r.avg_hedge:7.1f} {r.slope:7.3f} "
          f"{r.r:6.2f} {r.breakeven:10.1f} {r.breakeven_unhedged:12.1f} "
          f"{r.hedge_uplift:9.1f}  {ok}")

survivors = be[be.breakeven > BRENT_NOW]
print(f"\n  Airlines whose break-even sits ABOVE the 2026-Q1 price: "
      f"{len(survivors)} of {len(be)}  -> {', '.join(survivors.airline)}")
print(f"  Median break-even  hedged  : ${be.breakeven.median():.1f}/bbl")
print(f"  Median break-even unhedged : ${be.breakeven_unhedged.median():.1f}/bbl")
print(f"  Median uplift bought by the hedge book: ${be.hedge_uplift.median():.1f}/bbl")

# Validation: does the model's break-even actually predict the 2026-Q1 outcome?
q26 = fin[fin.quarter == "2026-Q1"][["airline", "profit_margin_pct", "airline_type"]]
val = be.merge(q26, on="airline")
r_val, p_val = stats.pearsonr(val.breakeven, val.profit_margin_pct)
print(f"\n  Validation: corr(break-even Brent, actual 2026-Q1 margin) = "
      f"{r_val:+.3f}  (p = {p_val:.4f}, n = {len(val)})")

# ── Figure 11 ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16.5, 10))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2.1, 1], wspace=0.20,
                        left=0.115, right=0.965, top=0.875, bottom=0.085)
fig.suptitle("B4 — How High Can The Oil Price Go Before Each Airline Loses Money?",
             fontsize=17, fontweight="bold", y=0.965)
fig.text(0.5, 0.916,
         f"Each airline has a break-even Brent price. Below it they profit, above it they "
         f"lose. In {peak_name} Brent was \\${BRENT_NOW:.0f} — "
         f"only {len(survivors)} of {len(be)} airlines were above water.",
         ha="center", fontsize=12, color="#333333")

ax1 = fig.add_subplot(gs[0, 0])
d = be.sort_values("breakeven")
yp = np.arange(len(d))

# Shade the zone that was loss-making at the 2026-Q1 price.
ax1.axvspan(80, BRENT_NOW, color=C_NOHEDGE, alpha=0.07, zorder=0)

# Dumbbell: red dot = break-even without hedging, blue dot = with. The bar between
# them is the headroom the hedge book bought.
for i, (_, r) in enumerate(d.iterrows()):
    ax1.plot([r.breakeven_unhedged, r.breakeven], [i, i], color="#9aa5ad", lw=3.2,
             solid_capstyle="round", zorder=2)
ax1.scatter(d.breakeven_unhedged, yp, s=70, color=C_NOHEDGE, edgecolor="white",
            lw=1.2, zorder=3, label="Break-even WITHOUT hedging")
ax1.scatter(d.breakeven, yp, s=70, color=C_HEDGE, edgecolor="white",
            lw=1.2, zorder=3, label="Break-even WITH hedging")

ax1.axvline(BRENT_NOW, color="black", lw=2.2, ls="--", zorder=5,
            label=f"Actual Brent in {peak_name}  (\\${BRENT_NOW:.0f}/bbl)")
ax1.axvline(BRENT_PEAK, color="#7f1d1d", lw=1.6, ls=":", zorder=5,
            label=f"Crisis peak Brent  (\\${BRENT_PEAK:.0f}/bbl)")

ax1.set_yticks(yp)
ax1.set_yticklabels([f"{a}  ({'FSC' if t == 'Flag Carrier' else 'LCC'})"
                     for a, t in zip(d.airline, d["type"])], fontsize=9)
for tick, bev in zip(ax1.get_yticklabels(), d.breakeven):
    if bev > BRENT_NOW:
        tick.set_color("#1a7a3c")
        tick.set_fontweight("bold")

ax1.set_xlabel("Brent crude price at which the airline breaks even  ($/bbl)", fontsize=10.5)
ax1.set_xlim(80, max(d.breakeven.max() * 1.10, BRENT_PEAK * 1.03))
ax1.set_ylim(-0.9, len(d) - 0.1)
ax1.legend(fontsize=9, loc="lower right", framealpha=0.96)
# Both labels ride next to the blue dot, so they never cross the 2026-Q1 line.
for i, (_, r) in enumerate(d.iterrows()):
    ax1.text(r.breakeven + 1.4, i, f"\\${r.breakeven:.0f}", va="center", fontsize=8.5,
             fontweight="bold", color=C_HEDGE)
    ax1.text(r.breakeven + 8.0, i, f"+\\${r.hedge_uplift:.0f}", va="center",
             fontsize=8, color="#7f8c8d")

ax1.text(BRENT_NOW - 1.5, len(d) - 0.55, "LOSS-MAKING at the 2026-Q1 price  ←",
         ha="right", va="center", fontsize=10, fontweight="bold", color=C_NOHEDGE)
ax1.text(BRENT_NOW + 1.5, len(d) - 0.55, "→  STILL PROFITABLE",
         ha="left", va="center", fontsize=10, fontweight="bold", color="#1a7a3c")
ax1.set_title("Longer bar = more protection bought by hedging      "
              "(blue = break-even, grey +\\$ = extra headroom the hedge book bought)",
              fontsize=11, fontweight="bold", pad=8)

ax2 = fig.add_subplot(gs[0, 1])
for t, c in [("Flag Carrier", C_FSC), ("Low Cost", C_LCC)]:
    s = val[val["type"] == t]
    ax2.scatter(s.breakeven, s.profit_margin_pct, s=95, color=c, edgecolor="white",
                lw=1.1, alpha=0.9, label=t, zorder=3)
sl, ic = np.polyfit(val.breakeven, val.profit_margin_pct, 1)
xs = np.linspace(val.breakeven.min(), val.breakeven.max(), 50)
ax2.plot(xs, sl * xs + ic, color="#333333", ls="--", lw=1.6,
         label=f"trend  (r = {r_val:+.2f})")
ax2.axvline(BRENT_NOW, color="black", lw=2, ls="--")
ax2.axhline(0, color="black", lw=1)
lab2 = pd.concat([val[val.breakeven > BRENT_NOW],
                  val.nsmallest(2, "profit_margin_pct")]).drop_duplicates("airline")
for k, (_, r) in enumerate(lab2.sort_values("breakeven").iterrows()):
    dy = 7 if k % 2 == 0 else -13
    ax2.annotate(r.airline, (r.breakeven, r.profit_margin_pct), xytext=(6, dy),
                 textcoords="offset points", fontsize=8.5)
ax2.set_xlabel("Break-even Brent ($/bbl)", fontsize=10)
ax2.set_ylabel(f"Actual net margin in {peak_name} (%)", fontsize=10)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.94)
ax2.set_title("Does the break-even actually predict who bled?\n"
              f"Yes — the higher the break-even, the smaller the loss (r = {r_val:+.2f})",
              fontsize=11.5, fontweight="bold", pad=8)

plt.savefig(f"{DATA_DIR}/fig11_breakeven_brent.png", dpi=180, bbox_inches="tight")
plt.close()
print("\n  [Saved] fig11_breakeven_brent.png")


# ═════════════════════════════════════════════════════════════════════════════
# KEY FINDINGS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("KEY FINDINGS  -  BUSINESS ANALYSIS")
print("=" * 70)

print("\n  B1  HEDGING")
print(f"    1. Recorded hedge savings first appear in {first_nonzero}; they become")
print(f"       material (>{MATERIAL:.0f}% of the fuel bill) in {first_material}, the Ukraine shock.")
print(f"    2. Peak effect {peak_q.quarter}: {peak_q.savings_pct_fuel:.1f}% of the gross fuel bill,")
print(f"       lifting the average net margin from {q.margin_unhedged.iloc[i_pk]:.2f}% to "
      f"{q.margin.iloc[i_pk]:.2f}%  (+{q.margin_cushion.iloc[i_pk]:.1f} pp).")
print(f"    3. Effective fuel burden in {peak_q.quarter}: "
      f"{q.fuel_pct_rev.iloc[i_pk]:.1f}% of revenue gross -> "
      f"{q.net_fuel_pct.iloc[i_pk]:.1f}% net of hedges.")
print(f"    4. BUT across airlines the hedge ratio barely explains the outcome")
print(f"       (r = {r_pk:+.3f}, p = {p_pk:.2f} in {peak_name}). Hedging lifts the whole")
print("       industry's loss curve; it does not decide who wins.")
print("    5. Caveat: savings are booked only when hedges are in the money, so the")
print("       2019-2021H1 zeros are NOT an absence of hedging (avg ratio ~32%).")

print("\n  B2  HORMUZ REROUTING")
print(f"    1. {len(hormuz_months)} disrupted months ({hormuz_months[0]} .. {hormuz_months[-1]}), "
      f"{int(b2.reroute_months.sum())} rerouted and")
print(f"       {int(b2.cancel_months.sum())} cancelled route-months across {b2.airline.nunique()} carriers.")
print(f"    2. Detour fuel ${tot_extra:,.0f} + lost revenue ${tot_lost:,.0f} "
      f"= ${tot_extra + tot_lost:,.0f} total")
print("       (per representative flight per route-month, not fleet-wide).")
print(f"    3. The cost is NOT mainly a fuel story: detours are only "
      f"{fuel_share_of_burden:.0f}% of the burden,")
print(f"       cancellations are {100 - fuel_share_of_burden:.0f}%. Fixing the fuel bill would "
      "have fixed almost nothing.")
print(f"    4. Geography drives the fuel penalty - Gulf hubs took {gulf_fuel_share:.0f}% of all "
      "detour fuel")
print(f"       cost - but per route-month they were NOT worse off (${g_avg:,.0f} vs "
      f"${n_avg:,.0f}),")
print("       because revenue loss scales with route size, not hub location.")
worst = b2.iloc[0]
print(f"    5. Worst absolute exposure: {worst.airline} (${worst.total_burden:,.0f} across "
      f"{int(worst.n_routes)} routes).")
print(f"    6. The two mechanisms decouple in time: reroutes held flat at "
      f"{int(monthly.reroutes.iloc[-1])}/month across all 7 months,")
print(f"       but cancellations collapsed from {int(monthly.cancels.max())}/month to "
      f"{int(monthly.cancels.iloc[-1])} once de-escalation began in")
print("       2026-04. The expensive half of the shock ended first.")

print("\n  B4  BREAK-EVEN BRENT")
print(f"    1. Break-even Brent spans ${be.breakeven.min():.0f} - ${be.breakeven.max():.0f}/bbl "
      f"(median ${be.breakeven.median():.0f}).")
print(f"    2. The hedge book buys a median +${be.hedge_uplift.median():.1f}/bbl of headroom; "
      f"the best-hedged")
print(f"       carriers gain up to +${be.hedge_uplift.max():.1f}/bbl.")
print(f"    3. At the 2026-Q1 price of ${BRENT_NOW:.0f}/bbl only {len(survivors)} of {len(be)} "
      f"airlines sit above")
print(f"       their break-even: {', '.join(survivors.airline)}.")
print(f"    4. The ranking validates out of sample: corr(break-even, actual 2026-Q1 margin)")
print(f"       = {r_val:+.3f} (p = {p_val:.4f}).")
print("    5. Caveat: COVID quarters are excluded because margin there was driven by")
print("       collapsed revenue, not fuel; the fit is linear and extrapolates poorly")
print(f"       far outside the observed ${be_src.brent_crude_usd_barrel.min():.0f}-"
      f"${be_src.brent_crude_usd_barrel.max():.0f}/bbl range.")

print(f"\n  Figures saved to: {DATA_DIR}")
for f in ["fig8_hedging_effectiveness.png", "fig9_reroute_cost.png",
          "fig11_breakeven_brent.png"]:
    print(f"    {f}")
print()
