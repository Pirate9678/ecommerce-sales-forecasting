"""
Build daily revenue series from the cleaned ecommerce orders.

Source: data/ecommerce_sales_clean.csv (copied from the ecommerce-sales-
dashboard project, 54,831 cleaned rows). Only Delivered orders count
towards revenue — Cancelled / Returned orders never produced revenue.

Outputs (all under data/):
  daily_overall.csv    — one row per day: date, revenue, orders
  daily_category.csv   — one row per day x product_category
  daily_region.csv     — one row per day x region
  series_profile.json  — date range, day counts, category/region shares,
                         festive-spike notes (Diwali)

Run:  python3 src/build_series.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SRC = DATA / "ecommerce_sales_clean.csv"

# Diwali dates inside the observed window (festive-season demand spike).
DIWALI = {"2023-11-12": "Diwali 2023", "2024-10-31": "Diwali 2024"}


def main() -> None:
    df = pd.read_csv(SRC, parse_dates=["order_date"])
    delivered = df[df["order_status"] == "Delivered"].copy()
    print(f"rows: {len(df):,} | Delivered: {len(delivered):,} "
          f"| revenue: ₹{delivered['net_revenue'].sum():,.0f}")

    # --- overall daily series -------------------------------------------
    overall = (
        delivered.groupby("order_date")
        .agg(revenue=("net_revenue", "sum"), orders=("order_id", "count"))
        .reset_index()
    )
    full_idx = pd.date_range(overall["order_date"].min(),
                             overall["order_date"].max(), freq="D")
    missing = full_idx.difference(overall["order_date"])
    assert len(missing) == 0, f"missing days: {missing.tolist()}"
    overall.to_csv(DATA / "daily_overall.csv", index=False)

    # --- by category / region -------------------------------------------
    cat = (
        delivered.groupby(["order_date", "product_category"])
        .agg(revenue=("net_revenue", "sum"), orders=("order_id", "count"))
        .reset_index()
    )
    cat.to_csv(DATA / "daily_category.csv", index=False)
    reg = (
        delivered.groupby(["order_date", "region"])
        .agg(revenue=("net_revenue", "sum"), orders=("order_id", "count"))
        .reset_index()
    )
    reg.to_csv(DATA / "daily_region.csv", index=False)

    # --- profile ---------------------------------------------------------
    cat_share = (
        delivered.groupby("product_category")["net_revenue"].sum()
        .sort_values(ascending=False)
    )
    reg_share = (
        delivered.groupby("region")["net_revenue"].sum()
        .sort_values(ascending=False)
    )
    monthly = delivered.groupby(delivered["order_date"].dt.to_period("M"))[
        "net_revenue"].sum()
    profile = {
        "source_rows": int(len(df)),
        "delivered_rows": int(len(delivered)),
        "total_revenue_inr": round(float(delivered["net_revenue"].sum()), 2),
        "date_min": str(overall["order_date"].min().date()),
        "date_max": str(overall["order_date"].max().date()),
        "n_days": int(len(overall)),
        "missing_days": 0,
        "daily_revenue_mean": round(float(overall["revenue"].mean()), 2),
        "daily_revenue_median": round(float(overall["revenue"].median()), 2),
        "daily_revenue_min": round(float(overall["revenue"].min()), 2),
        "daily_revenue_max": round(float(overall["revenue"].max()), 2),
        "max_day": str(overall.loc[overall["revenue"].idxmax(), "order_date"].date()),
        "category_share_inr": {k: round(float(v), 2)
                               for k, v in cat_share.items()},
        "region_share_inr": {k: round(float(v), 2)
                             for k, v in reg_share.items()},
        "monthly_revenue_inr": {str(k): round(float(v), 2)
                                for k, v in monthly.items()},
        "diwali_dates": DIWALI,
        "notes": [
            "Daily revenue is Delivered orders only.",
            "Oct 2024 is the best month (₹27.27M): Diwali fell on 31 Oct 2024 "
            "and the festive lift shows up across mid-October, not on a single day.",
            "Oct 2023 (Diwali 12 Nov 2023) shows a smaller lift; the spike is "
            "kept in the data as-is — no smoothing or outlier removal — so the "
            "forecast error on festive weeks is honest.",
        ],
    }
    (DATA / "series_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"days: {len(overall)} ({profile['date_min']} -> {profile['date_max']})")
    print(f"daily mean ₹{profile['daily_revenue_mean']:,.0f} | "
          f"max ₹{profile['daily_revenue_max']:,.0f} on {profile['max_day']}")
    print("wrote daily_overall.csv, daily_category.csv, daily_region.csv, "
          "series_profile.json")


if __name__ == "__main__":
    main()
