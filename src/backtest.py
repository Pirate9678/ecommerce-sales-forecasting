"""
Backtest revenue forecasts on the daily series.

Design
------
* Series: daily Delivered revenue — overall, Electronics (top category),
  South (top region). Built by src/build_series.py.
* Test windows (rolling origin, overall series): Oct 2024 (31 days —
  includes the Diwali festive spike), Aug 2025 (31 days), Sep 2025
  (30 days). Category / region series are evaluated on Sep 2025 only.
* Models:
    naive          — tomorrow = today (flat forecast)
    seasonal_naive — tomorrow = same weekday last week (period 7)
    holt_winters   — statsmodels ExponentialSmoothing, additive trend
                     (damped) + additive weekly seasonality
    sarima         — statsmodels SARIMA, order picked by lowest AIC from a
                     small candidate grid on each training window
    cal_regression — OLS on trend + day-of-week + month dummies
                     (numpy lstsq, no extra dependencies)
* Metrics on each test window: MAPE, RMSE, MAE, bias (mean signed error).
  No model is hidden if it loses to a baseline — the table shows all.

Prophet was NOT used: it is not installed in this environment and needs a
Stan backend; statsmodels Holt-Winters / SARIMA cover the same ground
(trend + seasonality) with far lighter dependencies.

Run:  python3 src/backtest.py
Outputs: results/metrics.json, results/metrics.csv, charts/backtest_overall.png,
         charts/model_comparison.png
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
RESULTS.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------- models
def f_naive(train, horizon):
    return np.full(horizon, train.iloc[-1])


def f_seasonal_naive(train, horizon, period=7):
    last_week = train.iloc[-period:].to_numpy()
    return np.tile(last_week, int(np.ceil(horizon / period)))[:horizon]


def f_holt_winters(train, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    fit = ExponentialSmoothing(
        train, trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=7,
    ).fit(optimized=True)
    return fit.forecast(horizon).to_numpy()


SARIMA_GRID = list(itertools.product(
    [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 1)],
    [(1, 0, 1, 7), (0, 0, 1, 7)],
))
# NOTE: seasonal differencing (D=1) is deliberately NOT in the grid — AIC
# is not comparable across different differencing orders (the likelihood
# is computed on fewer observations after differencing), and with a short
# series the D=1 fits can explode. See src/backtest_monthly.py.


def f_sarima(train, horizon):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best, best_aic = None, np.inf
    for order, seasonal in SARIMA_GRID:
        try:
            fit = SARIMAX(train, order=order, seasonal_order=seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            if fit.aic < best_aic:
                best_aic, best = fit.aic, fit
        except Exception:
            continue
    if best is None:
        return f_seasonal_naive(train, horizon)
    return best.forecast(horizon).to_numpy()


def _design(idx, n_before=0):
    """Trend + day-of-week + month dummies for a DatetimeIndex.

    Columns are built explicitly (not via get_dummies) so the train and
    forecast design matrices always have the same 19 columns:
    intercept, trend, dow 1..6 (Mon=0 is the reference), month 2..12
    (Jan is the reference). ``n_before`` shifts the trend forward for
    future dates.
    """
    n = len(idx)
    t = np.arange(n_before, n_before + n, dtype=float)
    dow = np.zeros((n, 6))
    for i, d in enumerate(idx.dayofweek):
        if d > 0:
            dow[i, d - 1] = 1.0
    mon = np.zeros((n, 11))
    for i, m in enumerate(idx.month):
        if m > 1:
            mon[i, m - 2] = 1.0
    return np.column_stack([np.ones(n), t, dow, mon])


def f_cal_regression(train, horizon, train_idx, future_idx):
    Xtr = _design(train_idx)
    beta, *_ = np.linalg.lstsq(Xtr, train.to_numpy(), rcond=None)
    Xf = _design(future_idx, n_before=len(train_idx))
    return Xf @ beta


MODELS = ["naive", "seasonal_naive", "holt_winters", "sarima",
          "cal_regression"]


def forecast(model, train, horizon, train_idx, future_idx):
    if model == "naive":
        return f_naive(train, horizon)
    if model == "seasonal_naive":
        return f_seasonal_naive(train, horizon)
    if model == "holt_winters":
        return f_holt_winters(train, horizon)
    if model == "sarima":
        return f_sarima(train, horizon)
    if model == "cal_regression":
        return f_cal_regression(train, horizon, train_idx, future_idx)
    raise ValueError(model)


def metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    return {
        "mape": float(np.mean(np.abs(err) / np.maximum(actual, 1e-9)) * 100),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),  # >0 means the model under-forecasts
    }


# ---------------------------------------------------------------- series
def load_series():
    overall = pd.read_csv(DATA / "daily_overall.csv", parse_dates=["order_date"])
    overall = overall.set_index("order_date")["revenue"]
    cat = pd.read_csv(DATA / "daily_category.csv", parse_dates=["order_date"])
    elec = (cat[cat["product_category"] == "Electronics"]
            .set_index("order_date")["revenue"])
    reg = pd.read_csv(DATA / "daily_region.csv", parse_dates=["order_date"])
    south = (reg[reg["region"] == "South"]
             .set_index("order_date")["revenue"])
    return {"overall": overall, "electronics": elec, "south": south}


# ---------------------------------------------------------------- run
def evaluate(series, name, test_start, test_end):
    train = series.loc[:test_start - pd.Timedelta(days=1)]
    test = series.loc[test_start:test_end]
    out = {}
    for model in MODELS:
        pred = forecast(model, train, len(test), train.index, test.index)
        out[model] = {"pred": pred, **metrics(test.to_numpy(), pred)}
    return train, test, out


def main():
    series = load_series()
    folds = [
        ("2024-10-01", "2024-10-31", "Oct 2024 (Diwali)"),
        ("2025-08-01", "2025-08-31", "Aug 2025"),
        ("2025-09-01", "2025-09-30", "Sep 2025"),
    ]
    all_rows, last_fold = [], {}
    for name, s in series.items():
        windows = folds if name == "overall" else [folds[-1]]
        for start, end, label in windows:
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            train, test, res = evaluate(s, name, start, end)
            for model, r in res.items():
                row = {"series": name, "window": label,
                       "test_start": str(start.date()),
                       "test_end": str(end.date()),
                       "test_days": len(test), "model": model,
                       "mape": round(r["mape"], 2),
                       "rmse": round(r["rmse"], 2),
                       "mae": round(r["mae"], 2),
                       "bias": round(r["bias"], 2)}
                all_rows.append(row)
                print(f"{name:12s} {label:20s} {model:15s} "
                      f"MAPE {r['mape']:5.2f}%  RMSE ₹{r['rmse']:,.0f}  "
                      f"MAE ₹{r['mae']:,.0f}")
            if (name, label) == ("overall", "Sep 2025"):
                last_fold = {"train": train, "test": test, "res": res}
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "metrics.csv", index=False)

    summary = {}
    for (name, model), g in df.groupby(["series", "model"]):
        summary.setdefault(name, {})[model] = {
            "mape_mean": round(float(g["mape"].mean()), 2),
            "mape_by_window": {r["window"]: r["mape"]
                               for _, r in g.iterrows()},
            "rmse_mean": round(float(g["rmse"].mean()), 2),
            "mae_mean": round(float(g["mae"].mean()), 2),
        }
    (RESULTS / "metrics.json").write_text(json.dumps(
        {"rows": all_rows, "summary": summary}, indent=2))
    print("\nwrote results/metrics.csv and results/metrics.json")
    plot_backtest(last_fold)
    plot_comparison(df)


def plot_backtest(fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train, test = fold["train"], fold["test"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(train.index[-120:], train.iloc[-120:], color="#888888",
            lw=1, label="train (last 120 days)")
    ax.plot(test.index, test, color="black", lw=1.8, label="actual (Sep 2025)")
    colors = {"naive": "#d62728", "seasonal_naive": "#ff7f0e",
              "holt_winters": "#1f77b4", "sarima": "#2ca02c",
              "cal_regression": "#9467bd"}
    for model, r in fold["res"].items():
        ax.plot(test.index, r["pred"], lw=1.2, color=colors[model],
                label=f"{model} (MAPE {r['mape']:.1f}%)")
    ax.set_title("Overall daily revenue — backtest, Sep 2025 (30-day holdout)")
    ax.set_ylabel("₹ per day")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CHARTS / "backtest_overall.png", dpi=150)
    plt.close(fig)


def plot_comparison(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = (df[df["window"] == "Sep 2025"]
           .pivot(index="model", columns="series", values="mape")
           .reindex(MODELS))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    piv.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Test MAPE by model — Sep 2025 holdout (lower is better)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("")
    ax.legend(title="series")
    ax.grid(axis="y", alpha=0.3)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
