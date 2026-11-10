"""
Forecast Oct–Dec 2025 from the best backtested MONTHLY model.

The best model is picked from results/metrics_monthly.json (lowest mean
MAPE on the overall series across the three rolling windows). It is refit
on the FULL monthly series (Jan 2023 – Sep 2025, 33 months) and forecast
3 months ahead: Oct, Nov, Dec 2025.

Daily files: the daily backtest showed day-to-day revenue is
unpredictable noise (no weekly pattern, lag-1 autocorr 0.13), so the
30/90-day daily files spread each forecast month evenly across its days
(monthly total / days in month). The honest statement is "October will
total about ₹X", not "Oct 14 will be ₹Y". The daily files exist so the
forecast is usable at daily granularity; their within-month shape is
flat by design, and the README says so. Note on day counts: the
"30-day" file is 1–30 Oct 2025 (30 rows); the "90-day" file is the
full quarter 1 Oct – 31 Dec 2025, which is 92 days, not 90 — it is
kept whole so its totals reconcile exactly with the monthly forecast.

Prediction intervals: SARIMA uses its own model-based 95% intervals.
Every other model uses ±1.96 × the standard deviation of its in-sample
one-step errors — an honest approximation, labelled as such.

Honest caveat: the 90-day window covers Diwali 2025 (20 Oct 2025).
Diwali 2024 fell on 31 Oct, so the seasonal-naive forecast for Oct 2025
reuses Oct 2024 — the festive month as a whole, which is the right
comparison at monthly granularity. If the best model were a daily one,
the festive spike would be missed; at monthly granularity it is
captured by the seasonal pattern itself.

Run:  python3 src/forecast.py   (after src/backtest_monthly.py)
Outputs: results/forecast_monthly.csv, results/forecast_30d.csv,
         results/forecast_90d.csv, results/forecast_summary.json,
         charts/forecast_90d.png, charts/forecast_category.png,
         charts/forecast_monthly.png
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CHARTS = ROOT / "charts"

FORECAST_MONTHS = pd.period_range("2025-10", "2025-12", freq="M")


def pick_best_model():
    metrics = json.loads((RESULTS / "metrics_monthly.json").read_text())
    overall = metrics["summary"]["overall"]
    best = min(overall, key=lambda m: overall[m]["mape_mean"])
    return best, overall[best]["mape_mean"]


def _one_step_errors(model, s):
    """In-sample one-step errors for interval width (monthly)."""
    errs = []
    for i in range(12, len(s)):
        train = s.iloc[:i]
        try:
            if model == "seasonal_naive":
                pred = train.iloc[-12]
            elif model == "naive":
                pred = train.iloc[-1]
            elif model == "holt_winters":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                fit = ExponentialSmoothing(
                    train, trend="add", damped_trend=True, seasonal="add",
                    seasonal_periods=12,
                    initialization_method="estimated").fit(optimized=True)
                pred = fit.forecast(1).iloc[0]
            elif model == "sarima":
                from backtest_monthly import fit_sarima_m
                fit = fit_sarima_m(train)
                pred = (fit.forecast(1).iloc[0] if fit is not None
                        else train.iloc[-12])
            else:  # cal_regression
                from backtest_monthly import _design_m
                X = _design_m(train.index)
                beta, *_ = np.linalg.lstsq(X, train.to_numpy(), rcond=None)
                pred = _design_m(s.index[i:i + 1], n_before=i) @ beta
                pred = float(pred[0])
            errs.append(float(s.iloc[i] - pred))
        except Exception:
            continue
    return np.array(errs)


def forecast_monthly(model, s, horizon=3):
    """Return (point forecast, lower 95%, upper 95%) for the next months."""
    if model == "seasonal_naive":
        fc = s.iloc[-12:].iloc[:horizon].to_numpy().astype(float)
        sigma = float(np.std(_one_step_errors(model, s)))
        return fc, np.maximum(fc - 1.96 * sigma, 0), fc + 1.96 * sigma
    if model == "naive":
        fc = np.full(horizon, s.iloc[-1])
        sigma = float(np.std(np.diff(s.to_numpy())))
        return fc, np.maximum(fc - 1.96 * sigma, 0), fc + 1.96 * sigma
    if model == "holt_winters":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(
            s, trend="add", damped_trend=True, seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated").fit(optimized=True)
        fc = fit.forecast(horizon).to_numpy()
        sigma = float(np.std(fit.resid))
        return fc, np.maximum(fc - 1.96 * sigma, 0), fc + 1.96 * sigma
    if model == "sarima":
        from backtest_monthly import fit_sarima_m
        fit = fit_sarima_m(s)
        if fit is None:
            return forecast_monthly("seasonal_naive", s, horizon)
        pred = fit.get_prediction(start=len(s), end=len(s) + horizon - 1)
        mean = pred.predicted_mean.to_numpy()
        ci = pred.conf_int(alpha=0.05).to_numpy()
        return mean, np.maximum(ci[:, 0], 0), ci[:, 1]
    # cal_regression
    from backtest_monthly import _design_m
    X = _design_m(s.index)
    beta, *_ = np.linalg.lstsq(X, s.to_numpy(), rcond=None)
    resid = s.to_numpy() - X @ beta
    sigma = float(np.std(resid))
    future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1),
                               periods=horizon, freq="ME")
    fc = _design_m(future_idx, n_before=len(s)) @ beta
    fc = np.maximum(fc, 0)
    return fc, np.maximum(fc - 1.96 * sigma, 0), fc + 1.96 * sigma


def load_monthly_all():
    overall = pd.read_csv(DATA / "daily_overall.csv",
                          parse_dates=["order_date"])
    series = {"overall": (overall.set_index("order_date")["revenue"]
                          .resample("ME").sum())}
    cat = pd.read_csv(DATA / "daily_category.csv", parse_dates=["order_date"])
    for c, g in cat.groupby("product_category"):
        series[f"category:{c}"] = (g.set_index("order_date")["revenue"]
                                   .resample("ME").sum())
    reg = pd.read_csv(DATA / "daily_region.csv", parse_dates=["order_date"])
    for r, g in reg.groupby("region"):
        series[f"region:{r}"] = (g.set_index("order_date")["revenue"]
                                 .resample("ME").sum())
    return series


def main():
    best, best_mape = pick_best_model()
    print(f"best monthly model from backtest: {best} "
          f"(overall mean MAPE {best_mape:.2f}%)")
    series = load_monthly_all()
    m_rows, d30_rows, d90_rows = [], [], []
    summary = {"best_model": best, "best_mape_mean_pct": best_mape,
               "forecast_months": ["2025-10", "2025-11", "2025-12"],
               "note": "Daily rows spread each forecast month evenly "
                       "across its days (see module docstring). The "
                       "90-day file is the full quarter 1 Oct – 31 Dec "
                       "2025 (92 days), so its totals reconcile with "
                       "the monthly forecast.",
               "series": {}}
    for name, s in series.items():
        fc, lo, hi = forecast_monthly(best, s)
        months = [p.to_timestamp("M") for p in FORECAST_MONTHS]
        for m, f, l, h in zip(months, fc, lo, hi):
            m_rows.append({"month": m.strftime("%Y-%m"), "series": name,
                           "forecast": round(float(f), 2),
                           "lower_95": round(float(l), 2),
                           "upper_95": round(float(h), 2)})
        # daily rows: monthly total spread evenly over the month
        for m, f, l, h in zip(months, fc, lo, hi):
            days = pd.date_range(m.replace(day=1),
                                 m + pd.offsets.MonthEnd(0), freq="D")
            for d in days:
                row = {"date": d.date(), "series": name,
                       "forecast": round(float(f / len(days)), 2),
                       "lower_95": round(float(l / len(days)), 2),
                       "upper_95": round(float(h / len(days)), 2)}
                d90_rows.append(row)
                if m.month == 10 and d.day <= 30:
                    d30_rows.append(row)
        summary["series"][name] = {
            "total_90d": round(float(fc.sum()), 2),
            "oct_2025": round(float(fc[0]), 2),
            "nov_2025": round(float(fc[1]), 2),
            "dec_2025": round(float(fc[2]), 2),
        }
        print(f"{name:28s} Oct ₹{fc[0]:>12,.0f}  Nov ₹{fc[1]:>12,.0f}  "
              f"Dec ₹{fc[2]:>12,.0f}  (90d ₹{fc.sum():,.0f})")
    pd.DataFrame(m_rows).to_csv(RESULTS / "forecast_monthly.csv",
                                index=False)
    pd.DataFrame(d30_rows).to_csv(RESULTS / "forecast_30d.csv", index=False)
    pd.DataFrame(d90_rows).to_csv(RESULTS / "forecast_90d.csv", index=False)
    (RESULTS / "forecast_summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote results/forecast_monthly.csv, forecast_30d.csv, "
          "forecast_90d.csv, forecast_summary.json")
    plot_forecast_90d()
    plot_category_forecast()
    plot_forecast_monthly()


def plot_forecast_90d():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    daily = pd.read_csv(DATA / "daily_overall.csv", parse_dates=["order_date"])
    fc = pd.read_csv(RESULTS / "forecast_90d.csv", parse_dates=["date"])
    fc = fc[fc["series"] == "overall"].set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(daily["order_date"].iloc[-120:], daily["revenue"].iloc[-120:],
            color="black", lw=1, label="actual daily (last 120 days)")
    ax.plot(fc.index, fc["forecast"], color="#1f77b4", lw=1.6,
            label="forecast (90 days, monthly level spread evenly)")
    ax.fill_between(fc.index, fc["lower_95"], fc["upper_95"],
                    color="#1f77b4", alpha=0.2, label="95% interval")
    ax.axvline(pd.Timestamp("2025-10-20"), color="#d62728", ls="--", lw=1)
    ax.text(pd.Timestamp("2025-10-21"), ax.get_ylim()[1] * 0.97,
            "Diwali 2025 (20 Oct)", color="#d62728", fontsize=8, va="top")
    ax.set_title("Overall daily revenue — 90-day forecast from 1 Oct 2025")
    ax.set_ylabel("₹ per day")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "forecast_90d.png", dpi=150)
    plt.close(fig)


def plot_category_forecast():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc = pd.read_csv(RESULTS / "forecast_monthly.csv")
    cats = ["category:Electronics", "category:Home & Kitchen",
            "category:Fashion"]
    totals = (fc[fc["series"].isin(cats)].groupby("series")["forecast"]
              .sum().reindex(cats))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    totals.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Forecast revenue, Oct–Dec 2025 — top 3 categories")
    ax.set_ylabel("₹ (90-day total)")
    ax.set_xticklabels(["Electronics", "Home & Kitchen", "Fashion"],
                       rotation=0)
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, labels=[f"₹{v / 1e7:.2f} Cr" for v in totals],
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS / "forecast_category.png", dpi=150)
    plt.close(fig)


def plot_forecast_monthly():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fc = pd.read_csv(RESULTS / "forecast_monthly.csv")
    fc = fc[fc["series"] == "overall"].set_index("month")
    daily = pd.read_csv(DATA / "daily_overall.csv", parse_dates=["order_date"])
    monthly = daily.set_index("order_date")["revenue"].resample("ME").sum()
    last_year = [float(monthly.loc[f"2024-{m}"].iloc[0])
                 for m in ("10", "11", "12")]
    months = ["Oct 2025", "Nov 2025", "Dec 2025"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, [v / 1e7 for v in last_year], 0.4,
           label="same month 2024 (actual)", color="#888888")
    ax.bar(x + 0.2, [v / 1e7 for v in fc["forecast"]], 0.4,
           label="forecast 2025", color="#1f77b4")
    ax.errorbar(x + 0.2, [v / 1e7 for v in fc["forecast"]],
                yerr=[(fc["forecast"] - fc["lower_95"]) / 1e7,
                      (fc["upper_95"] - fc["forecast"]) / 1e7],
                fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("₹ crore per month")
    ax.set_title("Overall monthly revenue — forecast vs same month last year")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "forecast_monthly.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
