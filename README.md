# E-commerce Sales Forecasting

Forecast the next quarter (Oct–Dec 2025) of revenue for a synthetic
Indian e-commerce business — and, more importantly, find out honestly
**what is actually forecastable** in the data. The headline result is humbling in the best
way: the simple "same month last year" baseline beats every fancier
model, so the forecast uses it — and says so.

Built on the dataset from the companion project
[ecommerce-sales-dashboard](https://github.com/Pirate9678/ecommerce-sales-dashboard)
(54,831 cleaned orders): that project describes what happened; this one
predicts what happens next.

## The forecast (Oct–Dec 2025)

Best model by backtest: **seasonal-naive** (same month last year),
mean MAPE **3.62%** on the overall series across three 3-month test
windows. Forecast from the full history (Jan 2023 – Sep 2025):

| Month | Forecast | 95% interval |
|---|---|---|
| Oct 2025 | **₹27,267,265 (₹2.73 Cr)** | ₹24.61 – 29.92 M |
| Nov 2025 | **₹24,626,824 (₹2.46 Cr)** | ₹21.97 – 27.28 M |
| Dec 2025 | **₹23,782,082 (₹2.38 Cr)** | ₹21.13 – 26.44 M |
| **Oct–Dec 2025 total** | **₹75,676,171 (₹7.57 Cr)** | — |

By category (90-day totals): Electronics **₹50.59 M (₹5.06 Cr, 66.9%)**,
Home & Kitchen ₹13.87 M, Fashion ₹7.34 M, Sports ₹2.92 M, Beauty ₹0.96 M.
By region: South ₹19.83 M, West ₹19.12 M, North ₹16.78 M, East ₹10.83 M,
Central ₹9.11 M. Full breakdowns in `results/forecast_monthly.csv`.
(The "90-day" daily file covers 1 Oct – 31 Dec 2025, which is 92 days —
the full quarter, kept whole so its totals reconcile with the monthly
forecast; the "30-day" file is 1–30 Oct.)

## The key finding: forecast monthly, not daily

The daily series was tested first — and it is essentially unpredictable
noise around a monthly level:

- Daily revenue: coefficient of variation 0.37, lag-1 autocorrelation
  **0.13**, lag-7 **0.14** — no memory from one day to the next.
- Day-of-week means are flat (₹614k–₹648k): there is **no weekly
  pattern** in this data, so weekly seasonality buys nothing.
- Best daily model still misses by **~29% MAPE** (calendar regression);
  the daily seasonal-naive baseline is worst at ~45%.

The monthly series is the opposite: lag-12 autocorrelation **0.90**,
and the same-month-last-year baseline already scores ~5% MAPE on its
own. A good analyst forecasts at the granularity where the signal is.
So the headline forecast is monthly, and the daily files
(`results/forecast_30d.csv`, `forecast_90d.csv`) spread each forecast
month evenly across its days — the honest daily shape when day-to-day
variation cannot be predicted. The daily analysis is reported as a
finding, not hidden.

## Data

- Source: `data/ecommerce_sales_clean.csv` — the cleaned dataset from
  the dashboard project (copied in, so this repo runs standalone).
- 54,831 orders, 49,227 Delivered, total revenue **₹637,611,185
  (₹63.76 Cr)**, 0 nulls. Only Delivered orders count as revenue.
- Daily series: **1,004 days, 2023-01-01 → 2025-09-30**, no missing days.
  Daily mean ₹635,071; max day ₹1,645,795 on 13 Oct 2024 (Diwali season).
- Monthly totals range ₹13.96 M (Feb 2023) to ₹27.27 M (Oct 2024).
- Categories: Electronics 65.7% of revenue, Home & Kitchen 19.6%,
  Fashion 9.7%, Sports 3.7%, Beauty 1.3%.
- Regions: South ₹16.11 Cr, West ₹15.78 Cr, North ₹15.10 Cr,
  East ₹9.36 Cr, Central ₹7.41 Cr.
- Diwali fell on 12 Nov 2023 and 31 Oct 2024; the festive lift shows up
  as a whole-month October/November effect, not a single-day spike —
  which is exactly why the monthly forecast captures it and a daily one
  would miss it. Diwali 2025 (20 Oct) sits inside the forecast window.

## Method

Daily and monthly series built from Delivered orders only
(`src/build_series.py`). Five models compared on rolling-origin
backtests — no model is hidden when it loses:

| Model | What it does |
|---|---|
| `naive` | Next period = last period |
| `seasonal_naive` | Next period = same period last cycle (7 days / 12 months) |
| `holt_winters` | statsmodels ExponentialSmoothing, damped additive trend + additive seasonality |
| `sarima` | statsmodels SARIMA, order picked by lowest AIC from a small grid |
| `cal_regression` | OLS on trend + month/day-of-week dummies (numpy `lstsq`) |

- **Monthly backtest:** 3 rolling windows of 3 months — Oct–Dec 2024
  (festive quarter), Apr–Jun 2025, Jul–Sep 2025.
- **Daily backtest:** 3 rolling windows — Oct 2024 (Diwali), Aug 2025,
  Sep 2025 (30–31 days each).
- Metrics: MAPE, RMSE, MAE, bias (mean signed error; positive bias =
  the model under-forecasts). Full tables in `results/metrics_monthly.csv`
  and `results/metrics.csv`.

**Prophet was not used:** it is not installed here and needs a Stan
backend; Holt-Winters and SARIMA cover the same ground (trend +
seasonality) with far lighter dependencies. That is a documented
choice, not an oversight.

**Two real bugs caught during backtesting, both fixed and documented
in the code:**
1. SARIMA's AIC is not comparable across different differencing orders
   — with 21 monthly points, seasonal differencing (D=1) leaves ~7
   effective observations, so its AIC looks artificially tiny and the
   grid always picked it. The picked model then forecast **₹91 M
   against ~₹17 M actuals**. The grid now uses D=0 only, where AIC
   comparisons are valid.
2. Holt-Winters cannot initialise with fewer than two full seasonal
   cycles (24 months) — it fails on the first fold (21 months) even
   with `initialization_method="estimated"`. It is reported as
   unavailable for that fold rather than silently dropped.

## Results

### Monthly backtest — mean MAPE across the 3 windows (lower is better)

| Model | Overall | Electronics | South |
|---|---|---|---|
| **seasonal_naive** | **3.62%** | **9.40%** | 9.71% |
| holt_winters | 4.44% * | 12.60% * | 10.25% * |
| cal_regression | 4.58% | 12.17% | **7.89%** |
| sarima | 12.40% | 14.78% | 19.02% |
| naive | 15.95% | 17.20% | 15.58% |

\* Holt-Winters has no score for the Oct–Dec 2024 window (needs ≥24
months to initialise); its mean is over the other two windows.

Overall per window: Oct–Dec 2024 — seasonal-naive 4.83%, cal_regression
5.18%, SARIMA 17.76%, naive 25.48%; Apr–Jun 2025 — seasonal-naive
2.34%, Holt-Winters 2.43%, cal_regression 4.80%; Jul–Sep 2025 —
seasonal-naive 3.68%, cal_regression 3.75%, Holt-Winters 6.45%.

### Daily backtest — mean MAPE, overall series

| Model | Mean MAPE |
|---|---|
| cal_regression | 28.73% |
| sarima | 30.48% |
| holt_winters | 31.49% |
| naive | 33.30% |
| seasonal_naive | 45.24% |

Per-series (Sep 2025 only): Electronics best is SARIMA at 59.40%;
South best is naive at 73.47%. Daily sub-series are not forecastable
at all in this data — reported, not hidden.

## How to reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. Build the daily series from the cleaned orders
.venv/bin/python src/build_series.py

# 2. Backtest daily, then monthly (monthly is the headline)
.venv/bin/python src/backtest.py
.venv/bin/python src/backtest_monthly.py

# 3. Forecast Oct–Dec 2025 from the best monthly model
.venv/bin/python src/forecast.py

# 4. Dashboard
.venv/bin/streamlit run app.py
```

Environment: Python 3.12, pandas 3.1.1, numpy 2.5.4, statsmodels
0.15.0, matplotlib 3.11.2, streamlit 1.67.0 (pinned in
`requirements.txt`).

## Repo structure

```
├── app.py                    # Streamlit dashboard
├── data/
│   ├── ecommerce_sales_clean.csv   # source orders (copied in)
│   ├── daily_overall.csv     # daily revenue, overall
│   ├── daily_category.csv    # daily revenue by category
│   ├── daily_region.csv      # daily revenue by region
│   └── series_profile.json   # date range, shares, Diwali notes
├── src/
│   ├── build_series.py       # orders -> daily series
│   ├── backtest.py           # daily backtest (5 models, 3 windows)
│   ├── backtest_monthly.py   # monthly backtest (headline)
│   └── forecast.py           # Oct–Dec 2025 forecast + intervals
├── results/
│   ├── metrics.csv / metrics.json               # daily backtest
│   ├── metrics_monthly.csv / metrics_monthly.json
│   ├── forecast_monthly.csv  # Oct/Nov/Dec totals + intervals
│   ├── forecast_30d.csv      # daily rows, Oct 2025
│   ├── forecast_90d.csv      # daily rows, Oct–Dec 2025
│   └── forecast_summary.json
├── charts/                   # backtest + forecast charts
├── requirements.txt
└── README.md
```

## Limitations (read before quoting the forecast)

- **Short history:** 33 months is barely 2¾ seasonal cycles. Monthly
  models are estimated on very little data; one unusual quarter would
  move the numbers.
- **Synthetic data:** the orders were generated, not observed. The
  "no weekly pattern / no daily memory" finding is a property of this
  generator, not of real e-commerce — real data usually has both.
- **No external regressors:** no prices, promotions, marketing spend,
  or competitor data. The forecast is purely "what the calendar says".
- **Diwali moves:** the festival shifts between October and November
  each year. The monthly seasonal-naive forecast handles this because
  it reuses whole festive months; a daily model would need an explicit
  festive regressor (only two Diwalis in the data — too few to estimate
  one reliably, so none was added).
- **Intervals are approximate:** ±1.96 × in-sample one-step error
  standard deviation for every model except SARIMA (model-based).
  They describe typical error, not worst-case festive error.
- **The forecast is a point of view, not a promise:** seasonal-naive
  assumes next October looks like last October. If the business runs a
  bigger sale, or the market turns, the actuals will differ — the
  backtest MAPE of ~3.6% is the honest scale of "usually", not a
  guarantee.

## Interview cheat sheet

**The 30-second answer:**
"I forecast e-commerce revenue from two and a half years of order data.
First I tested whether daily revenue is even predictable — it isn't,
there's no weekly pattern and no day-to-day memory, so the best daily
model still misses by about 30%. Monthly revenue is predictable, because
the same month last year repeats with 90% correlation. I backtested five
models on three rolling windows, and the simple same-month-last-year
baseline won at 3.6% average error, beating Holt-Winters, SARIMA and a
regression — so the forecast uses the baseline. October to December 2025
comes out at about ₹7.6 crore, with October the biggest month because
of Diwali."

**Likely questions:**
- *Why didn't you use Prophet?* — "It needs a Stan backend and wasn't
  installed; Holt-Winters and SARIMA do the same job — trend plus
  seasonality — with lighter dependencies, and I wanted the repo to run
  anywhere."
- *Why does a naive baseline beat SARIMA?* — "Only 33 monthly points.
  SARIMA has to estimate many parameters from very little data, and with
  seasonal differencing it actually exploded — it once forecast ₹9.1 crore
  against ₹1.7 crore actuals, because AIC isn't comparable across
  differencing orders. The seasonal pattern here is so stable that last
  year is simply the best predictor."
- *Why forecast monthly instead of daily?* — "I checked the daily data
  first: no day-of-week effect, autocorrelation near zero. Daily is
  noise around a monthly level, so I forecast the month and spread it
  across days rather than pretending I can predict individual days."
- *What would make the forecast better?* — "More history — a third and
  fourth Diwali would let me estimate a festive effect properly — and
  real drivers like promotions and prices as regressors. Right now it's
  a calendar-only forecast, and the README says that."
- *How do you know the numbers are real?* — "Everything is backtested,
  not fitted in-sample. The test windows were never used for training,
  and the tables show all five models including the ones that lost."

## What was verified vs. what wasn't

- ✅ Series build: 1,004 days, no missing days, Delivered-only,
  totals reconcile with the dashboard project (₹63.76 Cr)
- ✅ Daily + monthly backtests executed end to end; all metrics in this
  README come from `results/metrics*.csv` — none invented
- ✅ SARIMA explosion reproduced, diagnosed (AIC across differencing
  orders) and fixed; Holt-Winters initialisation limit documented
- ✅ Forecast files generated from the best backtested model
  (seasonal-naive); intervals computed from measured one-step errors
- ✅ Streamlit dashboard runs and serves (HTTP 200)
- ✅ Charts rendered from the real series and forecasts
- ⚠️ Figures describe a synthetic dataset, not a real business
- ⚠️ The forecast has no out-of-sample check yet — Oct–Dec 2025 had
  not happened when it was made
