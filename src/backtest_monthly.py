"""
Monthly backtest — the headline evaluation.

Why monthly: the daily series was backtested first (src/backtest.py) and
the diagnostics showed daily revenue is essentially unpredictable noise
around a monthly level — day-of-week means are flat (₹614k–₹648k),
lag-1 autocorrelation is 0.13 and lag-7 is 0.14, so every daily model
lands at 25–40% MAPE and even the seasonal-naive baseline loses to plain
naive. The predictable signal lives at the monthly level: monthly totals
have lag-12 autocorrelation 0.90 and the same-month-last-year baseline
already scores ~5% MAPE. A good analyst forecasts at the granularity
where the signal is — so the monthly backtest below is what the final
forecast is built on, and the daily result is reported as a finding,
not hidden.

Design
------
* Series: monthly Delivered revenue — overall, Electronics (top
  category), South (top region), aggregated from the daily series.
* Test windows (rolling origin, 3 months each): Oct–Dec 2024 (festive
  quarter, Diwali 31 Oct 2024), Apr–Jun 2025, Jul–Sep 2025.
* Models: naive, seasonal_naive (same month last year, s=12),
  holt_winters (additive damped trend + additive annual seasonality),
  sarima (order by lowest AIC from a small grid), cal_regression
  (OLS: trend + month dummies).
* Metrics per window: MAPE, RMSE, MAE, bias.

Run:  python3 src/backtest_monthly.py
Outputs: results/metrics_monthly.csv, results/metrics_monthly.json,
         charts/backtest_monthly.png, charts/model_comparison_monthly.png
"""

import itertools
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

MODELS_M = ["naive", "seasonal_naive", "holt_winters", "sarima",
            "cal_regression"]
SARIMA_GRID_M = list(itertools.product(
    [(1, 1, 1), (0, 1, 1), (1, 1, 0)],
    [(1, 0, 1, 12), (0, 0, 1, 12)],
))
# NOTE: seasonal differencing (D=1) is deliberately NOT in the grid.
# With only 21–30 monthly observations, D=1 leaves ~7 effective points:
# its AIC looks artificially tiny (likelihood on fewer observations) so
# AIC always picks it, and the fitted seasonal terms then explode —
# in testing, (0,1,1)(1,1,1,12) forecast ₹91M against ~₹17M actuals.
# All candidates here share D=0, so their AICs are comparable.


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    return {
        "mape": float(np.mean(np.abs(err) / np.maximum(actual, 1e-9)) * 100),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def f_naive_m(train, h):
    return np.full(h, train.iloc[-1])


def f_seasonal_naive_m(train, h, period=12):
    last_year = train.iloc[-period:].to_numpy()
    return np.tile(last_year, int(np.ceil(h / period)))[:h]


def f_holt_winters_m(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # NOTE: with fewer than two full seasonal cycles (24 months) this
    # raises "Cannot compute initial seasonals using heuristic method" —
    # even with initialization_method="estimated". The caller catches it
    # and the model is simply unavailable for that fold, which is itself
    # a finding: Holt-Winters needs 2 years of history to start.
    fit = ExponentialSmoothing(
        train, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=12,
        initialization_method="estimated").fit(optimized=True)
    return fit.forecast(h).to_numpy()


def fit_sarima_m(train):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best, best_aic = None, np.inf
    for order, seasonal in SARIMA_GRID_M:
        try:
            fit = SARIMAX(train, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            if fit.aic < best_aic:
                best_aic, best = fit.aic, fit
        except Exception:
            continue
    return best


def f_sarima_m(train, h):
    fit = fit_sarima_m(train)
    if fit is None:
        return f_seasonal_naive_m(train, h)
    return fit.forecast(h).to_numpy()


def _design_m(idx, n_before=0):
    """Intercept + trend + month dummies (Jan reference)."""
    n = len(idx)
    t = np.arange(n_before, n_before + n, dtype=float)
    mon = np.zeros((n, 11))
    for i, m in enumerate(idx.month):
        if m > 1:
            mon[i, m - 2] = 1.0
    return np.column_stack([np.ones(n), t, mon])


def f_cal_regression_m(train, h, train_idx, future_idx):
    X = _design_m(train_idx)
    beta, *_ = np.linalg.lstsq(X, train.to_numpy(), rcond=None)
    return _design_m(future_idx, n_before=len(train_idx)) @ beta


def forecast_m(model, train, h, train_idx, future_idx):
    if model == "naive":
        return f_naive_m(train, h)
    if model == "seasonal_naive":
        return f_seasonal_naive_m(train, h)
    if model == "holt_winters":
        return f_holt_winters_m(train, h)
    if model == "sarima":
        return f_sarima_m(train, h)
    return f_cal_regression_m(train, h, train_idx, future_idx)


def load_monthly():
    overall = pd.read_csv(DATA / "daily_overall.csv",
                          parse_dates=["order_date"])
    overall = (overall.set_index("order_date")["revenue"]
               .resample("ME").sum())
    cat = pd.read_csv(DATA / "daily_category.csv", parse_dates=["order_date"])
    elec = (cat[cat["product_category"] == "Electronics"]
            .set_index("order_date")["revenue"].resample("ME").sum())
    reg = pd.read_csv(DATA / "daily_region.csv", parse_dates=["order_date"])
    south = (reg[reg["region"] == "South"]
             .set_index("order_date")["revenue"].resample("ME").sum())
    return {"overall": overall, "electronics": elec, "south": south}


def main():
    series = load_monthly()
    folds = [
        ("2024-10-01", "2024-12-31", "Oct–Dec 2024 (Diwali)"),
        ("2025-04-01", "2025-06-30", "Apr–Jun 2025"),
        ("2025-07-01", "2025-09-30", "Jul–Sep 2025"),
    ]
    rows, last = [], {}
    for name, s in series.items():
        for start, end, label in folds:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            train = s[s.index < start]
            test = s.loc[start:end]
            for model in MODELS_M:
                try:
                    pred = forecast_m(model, train, len(test),
                                      train.index, test.index)
                except Exception as e:
                    print(f"{name} {label} {model}: FAILED ({e})")
                    continue
                r = metrics(test.to_numpy(), pred)
                rows.append({"series": name, "window": label,
                             "test_start": str(start.date()),
                             "test_end": str(end.date()),
                             "test_months": len(test), "model": model,
                             "mape": round(r["mape"], 2),
                             "rmse": round(r["rmse"], 2),
                             "mae": round(r["mae"], 2),
                             "bias": round(r["bias"], 2)})
                print(f"{name:12s} {label:22s} {model:15s} "
                      f"MAPE {r['mape']:5.2f}%  RMSE ₹{r['rmse']:,.0f}  "
                      f"MAE ₹{r['mae']:,.0f}")
            if name == "overall" and label == "Jul–Sep 2025":
                last = {"train": train, "test": test,
                        "preds": {m: forecast_m(m, train, len(test),
                                                train.index, test.index)
                                  for m in MODELS_M}}
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics_monthly.csv", index=False)
    summary = {}
    for (name, model), g in df.groupby(["series", "model"]):
        summary.setdefault(name, {})[model] = {
            "mape_mean": round(float(g["mape"].mean()), 2),
            "mape_by_window": {r["window"]: r["mape"]
                               for _, r in g.iterrows()},
            "rmse_mean": round(float(g["rmse"].mean()), 2),
            "mae_mean": round(float(g["mae"].mean()), 2),
        }
    (RESULTS / "metrics_monthly.json").write_text(json.dumps(
        {"rows": rows, "summary": summary}, indent=2))
    print("\nwrote results/metrics_monthly.csv and metrics_monthly.json")
    plot_monthly_backtest(last)
    plot_monthly_comparison(df)


def plot_monthly_backtest(last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, test = last["train"], last["test"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train.index, train, color="#888888", lw=1.2, label="train")
    ax.plot(test.index, test, "ko-", lw=1.8, label="actual (Jul–Sep 2025)")
    colors = {"naive": "#d62728", "seasonal_naive": "#ff7f0e",
              "holt_winters": "#1f77b4", "sarima": "#2ca02c",
              "cal_regression": "#9467bd"}
    for model, pred in last["preds"].items():
        ax.plot(test.index, pred, lw=1.2, color=colors[model],
                marker="o", ms=4, label=model)
    ax.set_title("Overall monthly revenue — backtest, Jul–Sep 2025")
    ax.set_ylabel("₹ per month")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "backtest_monthly.png", dpi=150)
    plt.close(fig)


def plot_monthly_comparison(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = (df.groupby(["model", "series"])["mape"].mean().unstack()
           .reindex(MODELS_M))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Mean test MAPE by model — monthly, 3 rolling windows "
                 "(lower is better)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("")
    ax.legend(title="series")
    ax.grid(axis="y", alpha=0.3)
    # Holt-Winters has no Oct–Dec 2024 score (needs >=24 months to
    # initialise), so its mean is over 2 windows — say so on the chart.
    fig.text(0.12, 0.01, "Note: Holt-Winters mean is over 2 windows "
             "(unavailable Oct–Dec 2024); all other means are over 3.",
             fontsize=8, color="#555555")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "model_comparison_monthly.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
