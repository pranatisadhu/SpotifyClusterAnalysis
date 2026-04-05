"""
evaluation.py
-------------
Tools for evaluating and interpreting cluster quality and profiles.

Provides:
  - Cluster profile summaries (mean feature values per cluster)
  - Feature importance (which features best separate clusters)
  - Stability analysis via bootstrapped silhouette scores
  - Elbow/silhouette plot data for k selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def summarise_clusters(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-cluster mean and median for all numeric features.

    Parameters
    ----------
    feature_matrix : raw (unscaled) feature matrix indexed by userid
    labels : cluster label per user (same order as feature_matrix.index)

    Returns
    -------
    pd.DataFrame with MultiIndex columns (feature, statistic) and
    cluster IDs as index
    """
    exclude_cols = exclude_cols or ["gender", "age", "country"]
    df = feature_matrix.copy()
    df["cluster"] = labels

    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols and c != "cluster"
    ]

    agg = df.groupby("cluster")[numeric_cols].agg(["mean", "median", "std"])
    return agg


def label_clusters(
    cluster_summary: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
) -> dict[int, str]:
    """
    Generate a human-readable persona label for each cluster.

    Uses Cohen's d style effect size: (cluster_mean - global_mean) / global_std
    so clusters near the average on all features still get a sensible label.
    """
    # Map (primary_dimension, direction) → persona name
    # Priority order: novelty/discovery first (most thesis-relevant), then engagement, then temporal
    _PERSONA_NAMES = [
        # (feature, high_threshold, low_threshold, high_name, low_name)
        ("novelty_ratio",           0.25, -0.25, "The Explorer",          "The Loyalist"),
        ("discovery_velocity_30d",  0.25, -0.25, "The Discoverer",        "The Habitual Listener"),
        ("track_replay_rate",       0.30, -0.30, "The Replayer",          "The Variety Seeker"),
        ("artist_entropy",          0.25, -0.25, "The Eclectic Listener", "The Focused Listener"),
        ("genre_entropy",           0.25, -0.25, "The Genre Explorer",    "The Genre Purist"),
        ("avg_tracks_per_session",  0.30, -0.30, "The Binge Listener",    "The Casual Listener"),
        ("weekend_ratio",           0.20, -0.20, "The Weekend Listener",  "The Weekday Listener"),
        ("temporal_hour_entropy",   0.25, -0.25, "The All-Day Listener",  "The Scheduled Listener"),
    ]

    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    global_means = feature_matrix[numeric_cols].mean()
    global_stds  = feature_matrix[numeric_cols].std().replace(0, 1)

    labels_map: dict[int, str] = {}
    df = feature_matrix.copy()
    df["cluster"] = labels

    for cid in sorted(set(labels)):
        if cid == -1:
            labels_map[cid] = "Noise"
            continue

        cluster_means = df[df["cluster"] == cid][numeric_cols].mean()

        # Find the single most distinctive feature by effect size
        best_name = None
        best_effect = 0.0

        for feat, hi_thresh, lo_thresh, hi_name, lo_name in _PERSONA_NAMES:
            if feat not in cluster_means or feat not in global_means:
                continue
            effect = (cluster_means[feat] - global_means[feat]) / global_stds[feat]
            if effect > hi_thresh and abs(effect) > best_effect:
                best_effect = abs(effect)
                best_name = hi_name
            elif effect < lo_thresh and abs(effect) > best_effect:
                best_effect = abs(effect)
                best_name = lo_name

        labels_map[cid] = best_name if best_name else f"Balanced Listener (Segment {cid})"

    # Deduplicate: if two clusters get the same name, append the cluster id
    seen: dict[str, int] = {}
    for cid, name in labels_map.items():
        if cid == -1:
            continue
        if name in seen:
            labels_map[seen[name]] = f"{name} (A)"
            labels_map[cid] = f"{name} (B)"
        else:
            seen[name] = cid

    return labels_map


def feature_importance(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    exclude_cols: list[str] | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Use a Random Forest classifier to rank features by how well they
    distinguish the discovered clusters.

    Returns
    -------
    pd.DataFrame with columns [feature, importance] sorted descending
    """
    exclude_cols = exclude_cols or ["gender", "age", "country"]
    mask = labels != -1  # Exclude HDBSCAN noise
    X = feature_matrix.loc[
        feature_matrix.index[mask]
    ].select_dtypes(include=[np.number])
    X = X[[c for c in X.columns if c not in exclude_cols]]
    y = labels[mask]

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    rf.fit(X_imputed, y)

    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance_df


def bootstrap_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 50,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Estimate silhouette score stability via bootstrapped sub-sampling.

    Returns
    -------
    (mean_silhouette, std_silhouette)
    """
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(random_state)
    scores = []
    mask = labels != -1
    X_clean = X[mask]
    y_clean = labels[mask]
    n = len(X_clean)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=int(n * sample_fraction), replace=False)
        if len(set(y_clean[idx])) < 2:
            continue
        try:
            s = silhouette_score(X_clean[idx], y_clean[idx])
            scores.append(s)
        except Exception:
            continue

    return float(np.mean(scores)), float(np.std(scores))


def elbow_data(inertias: dict[int, float], silhouette_scores: dict[int, float]) -> pd.DataFrame:
    """
    Package KMeans sweep results into a tidy DataFrame for plotting.
    """
    ks = sorted(inertias.keys())
    return pd.DataFrame(
        {
            "k": ks,
            "inertia": [inertias[k] for k in ks],
            "silhouette": [silhouette_scores.get(k, None) for k in ks],
        }
    )
