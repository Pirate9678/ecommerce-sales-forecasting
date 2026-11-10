"""
E-commerce Sales Forecasting — Streamlit dashboard.

Shows the monthly revenue series, the backtest model comparison and the
Oct–Dec 2025 forecast. Reads only the CSV/JSON files produced by
src/build_series.py, src/backtest.py, src/backtest_monthly.py and
src/forecast.py — run those first if the files are missing.

Run:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

st.set_page_config(page_title="E-commerce Sales Forecasting",
                   layout="wide")
st.title("E-commerce Sales Forecasting")
st.write("Monthly Delivered revenue, Jan 2023 – Sep 2025, forecast to "
         "Dec 2025. Data is synthetic (from the ecommerce-sales-dashboard "
         "project). Every number here is a measured backtest result — "
         "see the README for the method and the limits.")

overall_d = pd.read_csv(DATA / "daily_overall.csv",
                        parse_dates=["order_date"])
monthly = (overall_d.set_index("order_date")["revenue"]
           .resample("ME").sum())
metrics_m = pd.read_csv(RESULTS / "metrics_monthly.csv")
fc_m = pd.read_csv(RESULTS / "forecast_monthly.csv")
fc90 = pd.read_csv(RESULTS / "forecast_90d.csv", parse_dates=["date"])

# --- headline numbers ---------------------------------------------------
tot90 = fc_m[fc_m["series"] == "overall"]["forecast"].sum()
oct_f = fc_m[(fc_m["series"] == "overall")
             & (fc_m["month"] == "2025-10")]["forecast"].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Mean monthly revenue", f"₹{monthly.mean() / 1e7:.2f} Cr")
c2.metric("Forecast, Oct 2025", f"₹{oct_f / 1e7:.2f} Cr")
c3.metric("Forecast, Oct–Dec 2025", f"₹{tot90 / 1e7:.2f} Cr")

# --- monthly series + forecast -------------------------------------------
st.subheader("Monthly revenue and Oct–Dec 2025 forecast")
plot_m = monthly.rename("actual").to_frame()
fc = fc_m[fc_m["series"] == "overall"].copy()
fc["month"] = pd.to_datetime(fc["month"])
fc = fc.set_index("month")["forecast"].rename("forecast")
plot_m = pd.concat([plot_m, fc], axis=1)
st.line_chart(plot_m.tail(18))
st.caption("The forecast reuses the same months of 2024 (seasonal-naive "
           "— the best backtested model at 3.6% mean MAPE). Diwali 2025 "
           "falls on 20 Oct, inside the forecast October, as it did in "
           "Oct 2024. The daily chart below covers the full quarter "
           "1 Oct – 31 Dec 2025 (92 days).")

# --- backtest ------------------------------------------------------------
st.subheader("Backtest — which model earned the forecast?")
st.write("Mean MAPE across three 3-month rolling windows (Oct–Dec 2024, "
         "Apr–Jun 2025, Jul–Sep 2025). Lower is better. The seasonal-naive "
         "baseline — 'same month last year' — beats every fancier model, "
         "so the forecast uses it. That is the honest result, not a "
         "failure: with 33 months of very stable year-over-year "
         "seasonality, last year is the best predictor.")
mean_mape = (metrics_m.groupby(["model", "series"])["mape"].mean()
             .unstack())
st.bar_chart(mean_mape)
st.dataframe(metrics_m.sort_values(["series", "window", "mape"]),
             use_container_width=True)

# --- daily finding ---------------------------------------------------------
st.subheader("Why monthly, not daily?")
st.write("Daily revenue was tested first and is essentially unpredictable "
         "noise around the monthly level: no day-of-week pattern, lag-1 "
         "autocorrelation 0.13, and the best daily model still misses by "
         "~29% MAPE. The 90-day daily forecast below therefore spreads "
         "each forecast month evenly across its days — the honest daily "
         "shape when day-to-day variation cannot be predicted.")
fc_d = fc90[fc90["series"] == "overall"].set_index("date")[["forecast"]]
plot_d = pd.concat([overall_d.set_index("order_date")[["revenue"]]
                    .rename(columns={"revenue": "actual"}), fc_d], axis=1)
st.line_chart(plot_d.tail(210))

# --- per-category forecast ----------------------------------------------
st.subheader("Oct–Dec 2025 forecast by category")
cats = fc_m[fc_m["series"].str.startswith("category:")].copy()
cats["series"] = cats["series"].str.replace("category:", "")
tot = cats.groupby("series")["forecast"].sum().sort_values(ascending=False)
st.bar_chart(tot)
st.caption("Electronics dominates the forecast, as in the history "
           "(65.7% of all revenue).")

st.caption("Data: synthetic ecommerce orders (54,831 cleaned rows). "
           "Method and limits: see README.md.")
