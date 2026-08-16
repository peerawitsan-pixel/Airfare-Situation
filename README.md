# Geopolitical Oil Shocks & Airline Ticket Price Dynamics (2019 – 2026)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12%2B-3776ab.svg)](https://seaborn.pydata.org/)
[![Data-Scope](https://img.shields.io/badge/Scope-2019--2026%20(90%20Months)-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

> **An empirical data science investigation into how geopolitical conflict events and oil market disruptions transmit into global airfares, pass-through rates, transmission lags, and airline profitability across major crises.**

---

## 📌 Executive Summary

Jet fuel represents one of the largest volatile operating costs for global airlines, historically accounting for **25% to 40% of operational expenses (OpEx)**. When geopolitical shocks destabilize global crude oil supply, airline ticket prices experience rapid adjustments driven by base fare recalibrations and fuel surcharges.

This project examines **90 months of time-series data (Jan 2019 – Jun 2026)** encompassing **4 major geopolitical conflict events**, **14,850 ticket price records across 33 global airlines**, **10,440 fuel surcharge records**, and **725 airline financial metrics**. 

### 🌟 Key Findings
1. **Immediate Transmission Speed (0–1 Month Lag):** Statistical cross-correlation reveals that airfares respond to crude oil price shocks with **minimal lag (0–1 month, $r = +0.400, p < 0.001$)**. Fuel surcharges allow airlines to quickly adjust total ticket costs before quarterly base fare revisions occur.
2. **Asymmetric & Amplified Pass-Through in Crisis:**
   - **Russia-Ukraine War (2022):** Oil spiked $+74.8\%$ to peak at $\$117.6/\text{bbl}$, yielding a peak airfare rise of $+34.5\%$ (Pass-Through Rate $= 0.460$).
   - **US-Iran War Crisis (2025–2026):** Blockade of the Strait of Hormuz caused Brent crude to surge $+82.6\%$ (reaching $\$169.1/\text{bbl}$ peak). Airfares surged $+154.4\%$, resulting in an unprecedented peak **Pass-Through Rate of $1.869$**.
3. **Business Model Heterogeneity:** **Full-Service Flag Carriers (FSC)** pass through fuel costs faster and more aggressively via standardized fuel surcharges, preserving profit margins better than **Low-Cost Carriers (LCC)**, which suffer margin compression due to price sensitivity on short-haul routes.
4. **Severe Profitability Contraction:** During extreme shocks (e.g., US-Iran War), average airline net profit margins collapsed to **$-32.4\%$**, driven by record jet fuel prices ($>\$200/\text{bbl}$), flight rerouting costs around closed Middle Eastern airspaces, and suppressed demand.

---

## 📖 Table of Contents

- [Research Questions](#-research-questions)
- [Comparative Shock Summary](#-comparative-shock-summary)
- [Visual Gallery & Analytical Insights](#-visual-gallery--analytical-insights)
  - [Figure 1: Longitudinal Timeline Overview](#figure-1-longitudinal-timeline-overview)
  - [Figure 2: Lag & Cross-Correlation Analysis](#figure-2-lag--cross-correlation-analysis)
  - [Figure 3: Impact by Event Severity & Conflict Phase](#figure-3-impact-by-event-severity--conflict-phase)
  - [Figure 4: Macro Phase Dashboard](#figure-4-macro-phase-dashboard)
  - [Figure 5: Three-Shock Trajectory Comparison ($T_0=100$)](#figure-5-three-shock-trajectory-comparison-t_0100)
  - [Figure 6: Pass-Through Rates & Business Model Dynamics](#figure-6-pass-through-rates--business-model-dynamics)
  - [Figure 7: 2025–2026 Iran Crisis Deep Dive](#figure-7-20252026-iran-crisis-deep-dive)
- [Dataset Schema](#-dataset-schema)
- [Methodology](#-methodology)
- [Repository Structure](#-repository-structure)
- [How to Run & Reproduce](#-how-to-run--reproduce)
- [License & Citation](#-license--citation)

---

## 🎯 Research Questions

- **Q1: Transmission Speed & Lag Structure**  
  *How do geopolitical oil shocks translate into airline ticket price increases, and with what time lag?*
- **Q2: Pass-Through Rate Consistency & Crisis Differences**  
  *Do airlines pass through geopolitical oil price shocks into airfares at a consistent rate and speed, and how does the 2025–2026 Iran Crisis differ from COVID-19 (2020) and the Russia-Ukraine War (2022)?*

---

## 📊 Comparative Shock Summary

The table below summarizes the key macroeconomic metrics across the three major crisis periods analyzed in this repository ($T_0 = \text{Baseline month prior to shock}$):

| Metric / Shock Feature | COVID-19 Collapse (Mar 2020) | Russia-Ukraine War (Mar 2022) | US-Iran War Crisis (Dec 2025) |
| :--- | :---: | :---: | :---: |
| **Baseline Crude Price ($T_0$)** | $61.7 /bbl | $67.3 /bbl | $92.6 /bbl |
| **Peak Crude Price** | $26.6 /bbl  | $117.6 /bbl | $169.1 /bbl |
| **Peak Crude Price Change ($\Delta\%$)** | $-57.2\%$ | $+74.8\%$ | $+82.6\%$ |
| **Peak Ticket Fare Change ($\Delta\%$)** | $-56.5\%$ | $+34.5\%$ | $+154.4\%$ |
| **Peak Pass-Through Rate ($PT$)** | $0.987$ | $0.460$ | **$1.869$** |
| **Transmission Speed (Months to 5% Fare Shift)** | 0 Months | 0 Months | 4 Months (Surcharge Lag) |
| **Strait of Hormuz Disruption** | No | No | **Yes (7 Months)** |
| **Industry Net Margin at Peak** | $-34.8\%$ | $+2.1\%$ | **$-32.4\%$** |

---

## 🖼️ Visual Gallery & Analytical Insights

### Figure 1: Longitudinal Timeline Overview
![Figure 1: Full Timeline Overview](fig1_timeline_overview.png)
* [fig1_timeline_overview.png](file:///c:/Users/P/oil%20situation/fig1_timeline_overview.png)

**Key Takeaways:**
- **Top Panel:** Tracks Brent Crude vs. Jet Fuel prices over 90 months across 9 distinct conflict phases. Jet fuel maintains a steady crack spread above crude, peaking at $\$201.05/\text{bbl}$ during the 2026 US-Iran War conflict.
- **Middle Panel:** Demonstrates total ticket fares alongside the fuel surcharge component. Fuel surcharges spike sharply during energy shocks, representing up to $35\%$ of total ticket fares during peak escalation.
- **Bottom Panel:** Shows operational disruptions, including estimated monthly flight cancellations ($>9,400$ per month during peak conflict) and sovereign airspace closure counts.

---

### Figure 2: Lag & Cross-Correlation Analysis
![Figure 2: Cross Correlation Analysis](fig2_cross_correlation.png)
* [fig2_cross_correlation.png](file:///c:/Users/P/oil%20situation/fig2_cross_correlation.png)

**Key Takeaways:**
- Evaluates the Pearson correlation coefficient $r$ across lags $k \in [0, 8]$ months between $\text{MoM}\,\%\,\Delta\text{Brent}$ and $\text{MoM}\,\%\,\Delta\text{Ticket Fare}$.
- **Peak Correlation at Lag 0 ($r = +0.400, p = 0.0001$):** Confirms that airfares adjust within the same month as oil price movements, driven primarily by dynamic fuel surcharges attached to GDS pricing engine updates.

---

### Figure 3: Impact by Event Severity & Conflict Phase
![Figure 3: Event Impact Breakdown](fig3_event_impact.png)
* [fig3_event_impact.png](file:///c:/Users/P/oil%20situation/fig3_event_impact.png)

**Key Takeaways:**
- **Severity Breakdown:** **Extreme** severity events (e.g., Russia Invasion, Strait of Hormuz Blockade) produce an average immediate oil price jump of $+4.38\%$ and an average airfare impact of $+11.78\%$.
- **Phase Breakdown:** The **US-Iran War Conflict** phase generated the highest average airfare impact ($+17.07\%$) compared to $+8.80\%$ during the **Recovery & Surge** phase and $-13.40\%$ during **COVID-19 Collapse**.

---

### Figure 4: Macro Phase Dashboard
![Figure 4: Macro Phase Dashboard](fig4_phase_dashboard.png)
* [fig4_phase_dashboard.png](file:///c:/Users/P/oil%20situation/fig4_phase_dashboard.png)

**Key Takeaways:**
- Scatter regression demonstrates a strong positive linear correlation between Brent Crude price ($/bbl) and Average Ticket Fare (USD) ($r = +0.400$).
- Illustrates the distribution of fuel surcharge proportions and fare variations across the 9 conflict phases.

---

### Figure 5: Three-Shock Trajectory Comparison ($T_0=100$)
![Figure 5: Three Shock Trajectories](fig5_three_shock_trajectories.png)
* [fig5_three_shock_trajectories.png](file:///c:/Users/P/oil%20situation/fig5_three_shock_trajectories.png)

**Key Takeaways:**
- Re-indexes Brent crude oil and total airfares to $T_0 = 100$ at the start of each crisis window.
- **COVID-19 (2020):** Both oil and fare indices collapsed below 50.
- **Ukraine War (2022):** Oil indexed to ~175 while airfare indexed to ~135.
- **US-Iran War (2025–2026):** Extreme divergence — while oil indexed to ~183, airfares indexed past 250 due to compounding operational penalties (rerouting around Middle East airspace, higher insurance premiums, and jet fuel crack spread expansion).

---

### Figure 6: Pass-Through Rates & Business Model Dynamics
![Figure 6: Pass-Through Rates](fig6_pass_through_rates.png)
* [fig6_pass_through_rates.png](file:///c:/Users/P/oil%20situation/fig6_pass_through_rates.png)

**Key Takeaways:**
- Calculates empirical pass-through rate $PT = \frac{\% \Delta \text{Airfare}}{\% \Delta \text{Oil}}$ across shocks.
- **Flag Carriers vs. Low-Cost Carriers (LCCs):** Flag carriers demonstrate higher pass-through capabilities ($PT \approx 0.52 - 1.95$) due to high long-haul premium demand, whereas LCCs ($PT \approx 0.35 - 1.10$) absorb more cost inflation to maintain passenger load factors.

---

### Figure 7: 2025–2026 Iran Crisis Deep Dive
![Figure 7: Iran Crisis Deep Dive](fig7_iran_deepdive.png)
* [fig7_iran_deepdive.png](file:///c:/Users/P/oil%20situation/fig7_iran_deepdive.png)

**Key Takeaways:**
- Multi-panel analysis focusing on the 2025–2026 Iran Crisis:
  1. **Strait of Hormuz Disruption Timeline:** 7 consecutive months of disruption.
  2. **Fuel Surcharge Escalation:** Average surcharge climbed from $\$85$ to over $\$340$ per ticket.
  3. **Route Cost Inflation:** Middle East & Europe-Asia long-haul routes experienced flight time increases of 1.5–3.0 hours, increasing fuel burn per flight.
  4. **Profit Margin Collapse:** Net profit margin plunged to **$-32.4\%$**, echoing COVID-19 levels despite high passenger demand.

---

## 📑 Dataset Schema

The workspace utilizes six interconnected CSV datasets:

| File Name | Description | Key Features | Row Count |
| :--- | :--- | :--- | :---: |
| [`oil_jet_fuel_prices.csv`](file:///c:/Users/P/oil%20situation/oil_jet_fuel_prices.csv) | Monthly oil & jet fuel prices | `month`, `brent_crude_usd_barrel`, `jet_fuel_usd_barrel`, `strait_hormuz_disrupted`, `conflict_phase` | 90 |
| [`conflict_oil_events.csv`](file:///c:/Users/P/oil%20situation/conflict_oil_events.csv) | Geopolitical event log | `event_date`, `event_type`, `severity`, `oil_price_change_pct`, `airfare_impact_pct`, `flight_cancellations_est` | 39 |
| [`airline_ticket_prices.csv`](file:///c:/Users/P/oil%20situation/airline_ticket_prices.csv) | Route-level ticket pricing | `month`, `airline`, `route_class`, `base_fare_usd`, `fuel_surcharge_usd`, `total_fare_usd`, `load_factor_pct` | 14,850 |
| [`airline_financial_impact.csv`](file:///c:/Users/P/oil%20situation/airline_financial_impact.csv) | Financial & operating metrics | `month`, `airline`, `fuel_cost_pct_opex`, `operating_margin_pct`, `profit_margin_pct` | 725 |
| [`fuel_surcharges.csv`](file:///c:/Users/P/oil%20situation/fuel_surcharges.csv) | Surcharge granularity | `month`, `airline`, `route_class`, `fuel_surcharge_usd`, `surcharge_pct_of_total` | 10,440 |
| [`route_cost_impact.csv`](file:///c:/Users/P/oil%20situation/route_cost_impact.csv) | Route detour & cost metrics | `month`, `route_id`, `flight_time_minutes`, `reroute_extra_time_min`, `extra_fuel_burn_liters` | 3,240 |

---

## 🔬 Methodology

1. **Monthly Merged Aggregations:**  
   Calculated monthly weighted averages for base fare, fuel surcharge, total fare, load factor, and fuel cost % of OpEx across all 33 airlines.
2. **Cross-Correlation Lag Estimation:**  
   Computed Pearson correlation coefficient $r(k)$ between percentage monthly oil price shifts $\Delta \text{Oil}_{t-k}$ and ticket fare shifts $\Delta \text{Fare}_t$ for lags $k \in [0, 8]$.
3. **Shock Indexing ($T_0=100$):**  
   Indexed oil prices, jet fuel, total fares, and fuel surcharges to 100 at $T_0$ (the baseline month immediately preceding each major crisis):
   $$\text{Index}_t = \left(\frac{X_t}{X_{T_0}}\right) \times 100$$
4. **Pass-Through Rate ($PT$) Calculation:**  
   Formulated empirical pass-through rate as the ratio of cumulative percentage airfare change relative to cumulative percentage oil price change:
   $$PT_t = \frac{(\text{Fare}_t - \text{Fare}_{T_0}) / \text{Fare}_{T_0}}{(\text{Oil}_t - \text{Oil}_{T_0}) / \text{Oil}_{T_0}}$$

---

## 📁 Repository Structure

```
oil situation/
├── README.md                          # Comprehensive presentation & documentation
├── analysis.py                        # Primary analytical pipeline (Q1 & Figures 1-4)
├── analysis2.py                       # Advanced shock comparative pipeline (Q2 & Figures 5-7)
├── oil_jet_fuel_prices.csv            # Monthly Brent crude & jet fuel price series (2019-2026)
├── conflict_oil_events.csv            # Geopolitical event catalog with severity & impact
├── airline_ticket_prices.csv          # Monthly route-level ticket price dataset (14.8k rows)
├── airline_financial_impact.csv       # Airline financial metrics & profit margin metrics
├── fuel_surcharges.csv                # Detailed monthly fuel surcharge records
├── route_cost_impact.csv              # Flight path detour & extra fuel burn dataset
├── fig1_timeline_overview.png         # Figure 1: Full timeline overview
├── fig2_cross_correlation.png         # Figure 2: Lag & cross-correlation plot
├── fig3_event_impact.png              # Figure 3: Event severity & phase impact
├── fig4_phase_dashboard.png           # Figure 4: Macro phase dashboard
├── fig5_three_shock_trajectories.png  # Figure 5: Indexed trajectory comparison
├── fig6_pass_through_rates.png        # Figure 6: Pass-through rates & business models
└── fig7_iran_deepdive.png             # Figure 7: 2025-2026 Iran crisis deep dive
```

---

## 🚀 How to Run & Reproduce

### 1. Prerequisites
Ensure Python 3.10+ is installed along with the required data science packages:

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 2. Running Analysis & Generating Visualizations

To run the Question 1 macro pipeline and regenerate Figures 1–4:
```bash
python analysis.py
```

To run the Question 2 shock comparison & deep dive pipeline and regenerate Figures 5–7:
```bash
python analysis2.py
```

---

## 📜 License & Citation

This project is created for empirical research into energy economics and aviation market dynamics. Feel free to use, modify, and build upon this work with appropriate attribution.

