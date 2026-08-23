# CLAUDE.md

Project context for the **Geopolitical Oil Shocks -> Airline Ticket Price Dynamics (2019-2026)** analysis.

---

## 1. What this project is

An empirical time-series study of how geopolitical conflict events and crude-oil shocks
transmit into global airfares, fuel surcharges, route economics and airline profitability.

Coverage: **90 months (2019-01 .. 2026-06)**, 33 airlines, 5 regions, 39 geopolitical events,
4 major shocks (COVID-19, Russia-Ukraine, Gaza-Israel, US-Iran / Strait of Hormuz).

Two research questions are already implemented:

| Script | Question |
|---|---|
| `Airfare-Situation/analysis.py` | **Q1** - How do oil shocks translate into ticket-price increases, and with what lag? |
| `Airfare-Situation/analysis2.py` | **Q2** - Is pass-through consistent in rate and speed? How does the 2025-26 Iran crisis differ from 2020 and 2022? |
| `analysis_bussiness.py` | **B1 / B2 / B4** - hedging effectiveness, Hormuz rerouting cost, break-even Brent. See `suggestion.md`. |

---

## 2. Repository layout

```
C:\5001_FINAL\
├── path.py                       # single source of truth for the data directory
├── analysis_bussiness.py         # B1/B2/B4 -> fig8, fig9, fig11
├── CLAUDE.md                     # this file
├── suggestion.md                 # proposed business questions (Q3+), EN
├── suggestion_th.md              # same, Thai
└── Airfare-Situation\
    ├── analysis.py               # Q1  -> fig1..fig4
    ├── analysis2.py              # Q2  -> fig5..fig7
    ├── README.md                 # public write-up of Q1/Q2
    ├── *.csv                     # 6 datasets (see section 4)
    └── fig1..fig7 *.png          # generated figures
```

### Path convention (important)

`path.py` lives in the **repo root**, but the analysis scripts live in `Airfare-Situation/`.
Python puts the *script's* folder on `sys.path`, not the repo root, so a bare `import path`
fails with `ModuleNotFoundError`. Every script must bootstrap the parent directory first:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path as pth

DATA_DIR = pth.PATH        # currently r'C:\5001_FINAL\Airfare-Situation'
```

New scripts placed in `Airfare-Situation/` must copy this block. Scripts in the repo root
can `import path as pth` directly. All CSV reads and figure writes go through `DATA_DIR`;
never hard-code an absolute path.

### Running

```powershell
python .\Airfare-Situation\analysis.py      # regenerates fig1..fig4
python .\Airfare-Situation\analysis2.py     # regenerates fig5..fig7
python .\analysis_bussiness.py              # regenerates fig8, fig9, fig11
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.
Each script prints a console report and overwrites its PNGs in `DATA_DIR`.

---

## 3. The conflict-phase timeline

`conflict_phase` is the shared categorical spine across every dataset. It is the primary
grouping variable for phase-level comparisons.

| Phase | Months | n | Avg Brent | Peak Brent |
|---|---|---|---|---|
| Pre-Pandemic Baseline | 2019-01 .. 2020-02 | 14 | $64.1 | $67.8 |
| COVID-19 Collapse | 2020-03 .. 2021-05 | 15 | $50.0 | $83.1 |
| Recovery & Surge | 2021-06 .. 2022-02 | 9 | $75.3 | $87.2 |
| Ukraine War Shock | 2022-03 .. 2022-12 | 10 | $97.1 | $117.6 |
| Stabilisation | 2023-01 .. 2023-09 | 9 | $86.2 | $91.5 |
| Gaza-Israel Conflict | 2023-10 .. 2024-11 | 14 | $84.7 | $94.8 |
| Pre-Iran Escalation | 2024-12 .. 2025-11 | 12 | $95.0 | $104.8 |
| US-Iran War Conflict | 2025-12 .. 2026-03 | 4 | $144.5 | $169.1 |
| De-escalation & Negotiations | 2026-04 .. 2026-06 | 3 | $103.0 | $111.4 |

Strait of Hormuz is flagged disrupted for **2025-12 .. 2026-06** (7 months) - the same
window in which every rerouting event in `route_cost_impact.csv` occurs.

---

## 4. Dataset reference

All CSVs live in `Airfare-Situation/`. `month` is a `YYYY-MM` string; parse with
`parse_dates=["month"]`. Every table carries `conflict_phase`, so any table can be
grouped or joined on `month` or `(month, airline)` plus phase.

### 4.1 `oil_jet_fuel_prices.csv` - 90 rows x 12 cols

**Grain: one row per month.** The macro spine of the project.

| Column | Notes |
|---|---|
| `month`, `conflict_phase` | key + phase label |
| `brent_crude_usd_barrel` | $20.55 - $169.10 |
| `jet_fuel_usd_barrel` | $23.55 - $201.10 |
| `refinery_margin_usd` | crack spread, $2.98 - $31.97 |
| `jet_fuel_usd_per_gallon` | $0.55 - $4.88 |
| `wti_crude_usd_barrel` | $18.97 - $155.80 |
| `opec_production_mbd` | 25.57 - 30.44 m bbl/day |
| `us_strategic_reserve_mbl` | 373 - 640 m bbl (SPR) |
| `strait_hormuz_disrupted` | `Yes` / `No` |
| `yoy_brent_change_pct` | 12 nulls (first 12 months) |
| `data_source` | constant: EIA / Platts / IATA |

`refinery_margin_usd`, `wti_crude_usd_barrel`, `opec_production_mbd` and
`us_strategic_reserve_mbl` are **not used by analysis.py or analysis2.py** - unexploited.

### 4.2 `airline_ticket_prices.csv` - 14,850 rows x 18 cols

**Grain: month x airline x route_class** (90 x 33 x 5).

| Column | Notes |
|---|---|
| `airline`, `iata_code`, `country`, `region` | 33 airlines, 22 countries, 5 regions |
| `airline_type` | `Flag Carrier` / `Low Cost` |
| `route_class` | Short-Haul Domestic / Short-Haul Regional / Medium-Haul / Long-Haul / Ultra-Long-Haul |
| `avg_route_km` | 800 / 2,200 / 4,500 / 9,000 / 14,000 (one value per route_class) |
| `base_fare_usd` + `fuel_surcharge_usd` + `taxes_fees_usd` | = `total_fare_usd` |
| `brent_crude_usd`, `jet_fuel_usd_barrel` | denormalised copies of the oil table |
| `load_factor_pct` | 31 - 92 |
| `fuel_cost_pct_opex` | **stored as a fraction 0.052 - 0.763, not a percent** |
| `yoy_price_change_pct` | 1,980 nulls (first 12 months of each series) |

### 4.3 `fuel_surcharges.csv` - 10,440 rows x 13 cols

**Grain: month x airline x surcharge_band** (90 x 29 x 4). A *different* airline set (29)
and a band-based instead of route_class-based cut - do not assume it aligns 1:1 with the
ticket table.

| Column | Notes |
|---|---|
| `surcharge_band` | Band 1 Short Haul / Band 2 Medium / Band 3 Long / Band 4 Ultra Long |
| `km_range` | up to 1500 / 1501-4500 / 4501-9000 / over 9000 km |
| `fuel_surcharge_usd` | $0 - $916.80 |
| `surcharge_as_pct_base` | 0 - 153 % |
| `yoy_surcharge_change_pct` | **1,392 nulls and `inf` values** (divide-by-zero when the prior-year surcharge was 0). Must be cleaned before use. |
| `region` | 6 values here - `Asia` and `Asia-Pacific` are split, unlike the 5-region tables |

### 4.4 `airline_financial_impact.csv` - 725 rows x 20 cols

**Grain: quarter x airline** (29 quarters x 25 airlines). The only table with hedging and
P&L data - the backbone of any business-side analysis.

| Column | Notes |
|---|---|
| `quarter` | `2019-Q1` .. `2026-Q1` (29) |
| `month` | first month of the quarter; **quarterly grain, not monthly** |
| `revenue_usd_m`, `fuel_cost_usd_m`, `net_profit_usd_m` | USD millions |
| `fuel_cost_pct_revenue` | 5.1 - 153.3 %; **gross, before hedge savings** |
| `profit_margin_pct` | -44.9 .. +14.0 |
| `passengers_carried_m`, `fleet_size` | |
| `fuel_hedging_pct` | 0.2 - 64.8 % of fuel volume hedged |
| `hedge_savings_usd_m` | $0 - $798.9 m |
| `daily_fuel_consumption_bbl`, `quarterly_fuel_bbl` | volume basis for hedge math |
| `brent_crude_usd_barrel`, `jet_fuel_usd_barrel` | quarter-level copies |
| `region` | uses `N. America` (the ticket table uses `North America`) |

### 4.5 `route_cost_impact.csv` - 3,240 rows x 23 cols

**Grain: month x airline x route** (90 x 22 airlines x 36 routes). The only table with
rerouting and cancellation economics.

| Column | Notes |
|---|---|
| `origin_city` -> `destination_city`, `aircraft_type` | 21 origins, 17 destinations, 14 types |
| `original_distance_km`, `actual_distance_km`, `extra_distance_km` | detour math; extra > 0 only during Hormuz disruption |
| `rerouted`, `flight_cancelled` | `Yes` / `No` |
| `fuel_consumption_bbl`, `total_fuel_cost_usd`, `extra_fuel_cost_usd` | |
| `base_ticket_price_usd` + `fuel_surcharge_usd` | = `total_ticket_price_usd` |
| `estimated_passengers`, `route_revenue_usd` | 0 when the flight is cancelled |
| `fuel_pct_of_cost` | 89.9 - 99.1 % - implausibly high; see caveats |

Cancellations appear in two windows only: 2022-03..2022-12 (Ukraine) and 2025-12..2026-06 (Iran).

### 4.6 `conflict_oil_events.csv` - 39 rows x 14 cols

**Grain: one row per geopolitical event.**

| Column | Notes |
|---|---|
| `event_date` | 2019-04-11 .. 2026 |
| `event_type` | Political / Military / Economic / Operational / Diplomatic |
| `severity` | Medium / High / Very High / Extreme |
| `brent_before_usd`, `brent_after_usd`, `oil_price_change_pct` | -39 % .. +31.3 % |
| `airfare_impact_pct` | -29.5 % .. +25 % |
| `days_since_prev_event` | 1 null (first event) |
| `flight_cancellations_est`, `airspace_closures_countries` | 678 - 13,410; 0 - 9 |
| `location`, `event_description`, `data_source` | free text |

---

## 5. Existing figures

All PNGs are written to `DATA_DIR` (`Airfare-Situation/`).

**From `analysis.py` (Q1 - transmission speed and lag):**

| File | Content |
|---|---|
| `fig1_timeline_overview.png` | 3 stacked panels: Brent & jet-fuel prices; average total ticket price; month-over-month % change oil vs fare. Conflict phases shaded. |
| `fig2_cross_correlation.png` | Cross-correlation of oil vs fare at lags 0-6 months, plus a scatter at the best lag. **Best lag = 0 months, r = 0.400.** |
| `fig3_event_impact.png` | Average oil change vs immediate airfare change, grouped by conflict phase and event severity. |
| `fig4_phase_dashboard.png` | 4-panel phase dashboard: avg Brent, avg total fare, avg fuel surcharge, and an oil-vs-fare scatter across all months. |

**From `analysis2.py` (Q2 - pass-through rate and crisis comparison):**

| File | Content |
|---|---|
| `fig5_three_shock_trajectories.png` | Indexed trajectories (T0 = 100) for COVID-19, Ukraine and Iran shocks, oil vs fare side by side. |
| `fig6_pass_through_rates.png` | Oil shock vs fare response scatter per shock period, and cumulative pass-through at peak vs end. **Peak PTR: COVID 0.987, Ukraine 0.460, Iran 1.869.** |
| `fig7_iran_deepdive.png` | 6-panel Iran deep dive: absolute oil, ticket and surcharge levels; fuel cost as % of revenue; net profit margin by shock. |

**From `analysis_bussiness.py` (B1 / B2 / B4 - business angles, see `suggestion.md`):**

| File | Content |
|---|---|
| `fig8_hedging_effectiveness.png` | Four panels, one question each: (a) WHEN does hedging pay (savings as % of the gross fuel bill per quarter, against Brent); (b) HOW MUCH margin it adds (cushion in pp); (c) 2026-Q1 with vs without hedging as a dumbbell; (d) does hedging more mean surviving better. **Savings start 2021-Q3, turn material 2022-Q2, peak 2026-Q1 at 25.3 % of the fuel bill (+8.1 pp of margin) — but the cross-airline correlation is ~zero.** |
| `fig9_reroute_cost.png` | (a) absolute detour-fuel + lost-revenue burden by airline; (b) the same normalised per route-month, Gulf hubs marked; (c) monthly profile of detours vs cancellations; (d) worst detour routes. **$12.6 m combined burden, 95 % of it lost revenue, not fuel.** |
| `fig11_breakeven_brent.png` | Dumbbell ranking: red dot = break-even without hedging, blue dot = with, bar length = headroom the hedge book bought; the loss-making zone at the 2026-Q1 price is shaded. Right panel validates the ranking against realised margin. **Break-even $107-$144/bbl, hedging buys a median +$12.8/bbl; only 4 of 25 airlines sat above the 2026-Q1 price (validation r = +0.87).** |

`fig10` is deliberately unused - it is reserved for B3 (FSC vs LCC), which is not implemented.

**Not yet visualised:** load factor, refinery margin / crack spread, OPEC production,
the SPR series, and the surcharge-band recovery slopes (B5).

---

## 6. Data caveats

Known quirks. Check these before building a narrative on top of a number.

1. **`hedge_savings_usd_m` is 0 for every airline before 2021-Q3, and again in 2022-Q1.**
   Savings are only recorded when the hedge is in the money, so 2019-2021H1 is not "no
   hedging" - `fuel_hedging_pct` averages ~32 % in that period. Any "hedging starts to
   work in quarter X" statement must say *recorded savings*, not *hedging activity*.
2. **Quarterly data ends at 2026-Q1; monthly data runs to 2026-06.** The financial table
   does not cover the De-escalation phase at all. Never compare a full-monthly aggregate
   against a financial aggregate without truncating to 2026-Q1.
3. **2021-Q1/Q2 are still labelled `COVID-19 Collapse`** while Brent is already $75-77 and
   `fuel_cost_pct_revenue` hits 62 % - that ratio is driven by collapsed revenue, not
   expensive fuel. Treat `fuel_cost_pct_revenue` during COVID as a revenue artefact.
4. **`fuel_cost_pct_opex` in the ticket table is a fraction (0.28), not a percent (28).**
   Multiply by 100 before labelling an axis.
5. **`yoy_surcharge_change_pct` contains `inf`.** Replace with NaN before any mean.
6. **Region labels are inconsistent** across tables: `N. America` (financial) vs
   `North America` (tickets), and `Asia` / `Asia-Pacific` split only in surcharges.
   Normalise before joining.
7. **Airline sets differ per table**: 33 (tickets), 29 (surcharges), 25 (financial),
   22 (routes). Use inner joins and state the surviving airline count.
8. **`fuel_pct_of_cost` in `route_cost_impact.csv` is 90-99 %**, i.e. the "total cost"
   denominator is fuel-only, not full route cost. Do not read it as a real cost structure.
9. **The Iran-crisis lag in `analysis2.py` reports 4 months at r = -0.938.** A strong
   *negative* correlation means the window extends past the shock's turning point; this is
   a windowing artefact, not a 4-month transmission lag. Do not cite it as speed.
10. `data_source` is a constant string in two tables - drop it from any groupby.

---

## 7. Conventions for new work

- Put new analyses in the repo root next to `analysis_bussiness.py`, or in
  `Airfare-Situation/` with the `sys.path` bootstrap from section 2.
- Number new figures continuing from `fig8_`.
- Follow the existing house style: a print banner per section (`"=" * 60`), a final
  `KEY FINDINGS` block, and
  `plt.savefig(f"{DATA_DIR}/figN_name.png", dpi=150-200, bbox_inches="tight")`.
- Console output is part of the deliverable - findings are reported both as figures and as
  printed numbers.
- Clean `inf`/null values before aggregating, and state the caveat in the printed output.
