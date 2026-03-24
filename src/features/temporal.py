"""
temporal.py
-----------
Computes temporal diversity and listening pattern stability features per user.

Features produced:
  - temporal_hour_entropy      : entropy of play distribution across 24 hours
  - temporal_dow_entropy       : entropy of play distribution across 7 days of week
  - temporal_month_entropy     : entropy of play distribution across 12 months
  - morning_ratio              : fraction of plays between 06:00–12:00
  - evening_ratio              : fraction of plays between 18:00–00:00
  - weekend_ratio              : fraction of plays on Sat/Sun
  - listening_days             : number of distinct calendar days with activity
  - avg_daily_plays            : mean plays per active day
  - temporal_stability_pc1..N  : principal components of the hourly play profile
                                  (derived via PCA — captures pattern stability
                                  while reducing hour-level multicollinearity)
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
    """
    Compute temporal diversity and stability features for each user.

    Parameters
    ----------
    scrobbles : DataFrame with columns [userid, timestamp (datetime64, UTC), ...]
    pca_components : number of PCA components for hourly profile compression

    Returns
    -------
    pd.DataFrame indexed by userid
    """
    df = scrobbles.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month
    df["date"] = df["timestamp"].dt.date

    records: list[dict] = []
    users = df["userid"].unique()

    for uid in users:
        user_df = df[df["userid"] == uid]
        n = len(user_df)

        # Hour distribution
        hour_counts = np.zeros(24)
        hc = user_df["hour"].value_counts()
        for h, c in hc.items():
            hour_counts[int(h)] = c

        # Day-of-week distribution
        dow_counts = np.zeros(7)
        dc = user_df["dow"].value_counts()
        for d, c in dc.items():
            dow_counts[int(d)] = c

        # Month distribution
        month_counts = np.zeros(12)
        mc = user_df["month"].value_counts()
        for m, c in mc.items():
            month_counts[int(m) - 1] = c

        morning = hour_counts[6:12].sum() / n if n else 0.0
        evening = hour_counts[18:24].sum() / n if n else 0.0
        weekend = (dow_counts[5] + dow_counts[6]) / n if n else 0.0

        listening_days = user_df["date"].nunique()
        date_range = (user_df["timestamp"].max() - user_df["timestamp"].min()).days + 1
        avg_daily = n / date_range if date_range > 0 else n

        records.append(
            {
                "userid": uid,
                "temporal_hour_entropy": _entropy(hour_counts),
                "temporal_dow_entropy": _entropy(dow_counts),
                "temporal_month_entropy": _entropy(month_counts),
                "morning_ratio": morning,
                "evening_ratio": evening,
                "weekend_ratio": weekend,
                "listening_days": listening_days,
                "avg_daily_plays": avg_daily,
                # Raw hourly profile stored for PCA (will be extracted later)
                **{f"_hour_{h}": hour_counts[h] / n if n else 0.0 for h in range(24)},
            }
        )

    result = pd.DataFrame(records).set_index("userid")

    # PCA on normalised hourly profiles to capture listening pattern stability
    hour_cols = [f"_hour_{h}" for h in range(24)]
    if len(result) >= pca_components:
        hour_matrix = result[hour_cols].values
        scaler = StandardScaler()
        hour_scaled = scaler.fit_transform(hour_matrix)
        pca = PCA(n_components=pca_components, random_state=42)
        pca_scores = pca.fit_transform(hour_scaled)
        for i in range(pca_components):
            result[f"temporal_stability_pc{i+1}"] = pca_scores[:, i]
    else:
        for i in range(pca_components):
            result[f"temporal_stability_pc{i+1}"] = 0.0

    result = result.drop(columns=hour_cols)
    return result
