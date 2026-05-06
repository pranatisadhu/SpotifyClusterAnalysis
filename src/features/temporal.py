"""
temporal.py
-----------
Computes temporal diversity and listening pattern stability features per user.
Uses vectorised groupby operations instead of a per-user loop to avoid
scanning 19 M rows 992 times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(scipy_entropy(probs, base=2))


def compute_temporal_features(
    scrobbles: pd.DataFrame,
    pca_components: int = 5,
) -> pd.DataFrame:
    df = scrobbles[["userid", "timestamp"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"]  = df["timestamp"].dt.hour
    df["dow"]   = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["date"]  = df["timestamp"].dt.date

    # ── Vectorised aggregations ───────────────────────────────────────────────
    totals = df.groupby("userid").size().rename("n")

    # Hour distribution (24 columns)
    hour_pivot = (
        df.groupby(["userid", "hour"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    hour_pivot.columns = [f"_hour_{h}" for h in range(24)]

    # Entropy calculations
    hour_arr   = hour_pivot.values.astype(float)
    dow_pivot  = (
        df.groupby(["userid", "dow"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(7), fill_value=0)
    )
    month_pivot = (
        df.groupby(["userid", "month"]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )

    hour_entropy  = pd.Series(
        [_entropy(r) for r in hour_arr], index=hour_pivot.index, name="temporal_hour_entropy"
    )
    dow_entropy   = pd.Series(
        [_entropy(r) for r in dow_pivot.values], index=dow_pivot.index, name="temporal_dow_entropy"
    )
    month_entropy = pd.Series(
        [_entropy(r) for r in month_pivot.values], index=month_pivot.index, name="temporal_month_entropy"
    )

    # Ratios (vectorised)
    n = totals.reindex(hour_pivot.index)
    morning_ratio = hour_pivot[[f"_hour_{h}" for h in range(6, 12)]].sum(axis=1) / n
    evening_ratio = hour_pivot[[f"_hour_{h}" for h in range(18, 24)]].sum(axis=1) / n
    weekend_ratio = (
        dow_pivot.get(5, 0) + dow_pivot.get(6, 0)
    ) / n

    # Listening days and avg daily plays
    listening_days = df.groupby("userid")["date"].nunique().rename("listening_days")
    date_range     = (
        df.groupby("userid")["timestamp"]
        .agg(lambda s: (s.max() - s.min()).days + 1)
        .rename("date_range")
    )
    avg_daily = (n / date_range.clip(lower=1)).rename("avg_daily_plays")

    # ── Combine ───────────────────────────────────────────────────────────────
    result = pd.concat([
        hour_entropy, dow_entropy, month_entropy,
        morning_ratio.rename("morning_ratio"),
        evening_ratio.rename("evening_ratio"),
        weekend_ratio.rename("weekend_ratio"),
        listening_days, avg_daily,
        hour_pivot,
    ], axis=1)
    result.index.name = "userid"

    # ── PCA on normalised hourly profiles ─────────────────────────────────────
    hour_cols = [f"_hour_{h}" for h in range(24)]
    if len(result) >= pca_components:
        scaled = StandardScaler().fit_transform(result[hour_cols].values)
        pca_scores = PCA(n_components=pca_components, random_state=42).fit_transform(scaled)
        for i in range(pca_components):
            result[f"temporal_stability_pc{i+1}"] = pca_scores[:, i]
    else:
        for i in range(pca_components):
            result[f"temporal_stability_pc{i+1}"] = 0.0

    result = result.drop(columns=hour_cols)
    return result
