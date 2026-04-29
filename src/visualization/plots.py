"""
plots.py
--------
Interactive Plotly visualisations for the cluster analysis.

All functions return a plotly Figure object — call .show() or .write_html()
as needed.

Functions:
  - plot_umap_clusters      : 2D UMAP scatter coloured by cluster
  - plot_cluster_radar      : radar/spider chart of cluster feature profiles
  - plot_feature_importance : horizontal bar chart of RF feature importance
  - plot_elbow              : KMeans elbow + silhouette dual-axis chart
  - plot_cluster_heatmap    : heatmap of normalised feature means per cluster
  - plot_genre_distribution : stacked bar of top genres per cluster
  - plot_temporal_heatmap   : listening activity by hour × day-of-week per cluster
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _cluster_color_map(labels: np.ndarray) -> dict[int, str]:
    unique = sorted(set(labels))
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(unique)}


def plot_umap_clusters(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    userids: list[str] | None = None,
    title: str = "UMAP Projection of User Listening Profiles",
) -> go.Figure:
    """
    2D UMAP scatter plot coloured by cluster label.

    Parameters
    ----------
    umap_coords : (n_users, 2) array of UMAP coordinates
    labels : cluster label per user
    cluster_names : optional mapping {cluster_id: human_readable_name}
    userids : optional list of user IDs for hover tooltips
    """
    df = pd.DataFrame(
        {"umap_x": umap_coords[:, 0], "umap_y": umap_coords[:, 1], "cluster": labels}
    )
    if userids:
        df["userid"] = userids
    cluster_names = cluster_names or {}
    df["cluster_label"] = df["cluster"].map(
        lambda c: cluster_names.get(c, f"Cluster {c}" if c != -1 else "Noise")
    )

    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster_label",
        hover_data=["userid"] if "userid" in df.columns else None,
        color_discrete_sequence=_PALETTE,
        title=title,
        labels={"umap_x": "UMAP-1", "umap_y": "UMAP-2", "cluster_label": "Segment"},
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    return fig


def plot_cluster_radar(
    cluster_summary: pd.DataFrame,
    features: list[str],
    cluster_names: dict[int, str] | None = None,
    title: str = "Cluster Feature Profiles (Normalised)",
) -> go.Figure:
    """
    Radar chart comparing cluster profiles across selected features.
    Each feature is min-max normalised across clusters so all axes share
    the same 0–1 scale and no single high-magnitude feature collapses
    the others to the centre.

    Parameters
    ----------
    cluster_summary : output of evaluation.summarise_clusters (mean values)
    features : list of feature names to include on the radar
    """
    cluster_names = cluster_names or {}

    # Collect raw means per (cluster, feature)
    valid_cids = [cid for cid in cluster_summary.index if cid != -1]
    raw: dict[int, list[float]] = {}
    for cid in valid_cids:
        raw[cid] = [
            float(cluster_summary.loc[cid, (feat, "mean")])
            if (feat, "mean") in cluster_summary.columns else 0.0
            for feat in features
        ]

    # Per-feature min-max normalisation across clusters so every axis is 0–1
    feat_min = [min(raw[c][i] for c in valid_cids) for i in range(len(features))]
    feat_max = [max(raw[c][i] for c in valid_cids) for i in range(len(features))]

    def normalise(vals: list[float]) -> list[float]:
        return [
            (v - feat_min[i]) / (feat_max[i] - feat_min[i] + 1e-9)
            for i, v in enumerate(vals)
        ]

    fig = go.Figure()
    for cid in valid_cids:
        norm_vals = normalise(raw[cid])
        label = cluster_names.get(cid, f"Cluster {cid}")
        fig.add_trace(
            go.Scatterpolar(
                r=norm_vals + [norm_vals[0]],   # close the polygon
                theta=features + [features[0]],
                name=label,
                fill="toself",
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=title,
        template="plotly_white",
        showlegend=True,
    )
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Features Distinguishing Listener Segments",
) -> go.Figure:
    """
    Horizontal bar chart of Random Forest feature importances.
    """
    df = importance_df.head(top_n).copy()
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Blues",
        template="plotly_white",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    return fig


def plot_elbow(
    elbow_df: pd.DataFrame,
    title: str = "KMeans Model Selection: Elbow + Silhouette",
) -> go.Figure:
    """
    Dual-axis chart: inertia (left axis) and silhouette score (right axis) vs k.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=elbow_df["k"],
            y=elbow_df["inertia"],
            name="Inertia (Elbow)",
            mode="lines+markers",
            marker=dict(size=8),
            line=dict(color="#636EFA"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=elbow_df["k"],
            y=elbow_df["silhouette"],
            name="Silhouette Score",
            mode="lines+markers",
            marker=dict(size=8, symbol="diamond"),
            line=dict(color="#EF553B"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title=title,
        xaxis_title="Number of Clusters (k)",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return fig


def plot_cluster_heatmap(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    features: list[str] | None = None,
    cluster_names: dict[int, str] | None = None,
    title: str = "Normalised Feature Means by Cluster",
) -> go.Figure:
    """
    Heatmap where rows = clusters, columns = features, values = z-scored means.
    """
    from sklearn.preprocessing import StandardScaler

    cluster_names = cluster_names or {}
    df = feature_matrix.copy()
    df["cluster"] = labels

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "cluster" and c not in ["age"]]
    if features:
        numeric_cols = [c for c in features if c in numeric_cols]

    cluster_means = df.groupby("cluster")[numeric_cols].mean()
    cluster_means = cluster_means[cluster_means.index != -1]

    scaler = StandardScaler()
    z_scores = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns,
    )

    y_labels = [
        cluster_names.get(int(cid), f"Cluster {cid}") for cid in z_scores.index
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_scores.values,
            x=z_scores.columns.tolist(),
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(z_scores.values, 2),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        template="plotly_white",
        height=max(400, 80 * len(z_scores)),
    )
    return fig


def plot_genre_distribution(
    scrobbles: pd.DataFrame,
    artist_genres: pd.DataFrame,
    labels: np.ndarray,
    userids: list[str],
    top_n_genres: int = 10,
    cluster_names: dict[int, str] | None = None,
    title: str = "Top Genre Distribution by Listener Segment",
) -> go.Figure:
    """
    Stacked bar chart: proportion of each top genre per cluster.
    """
    cluster_names = cluster_names or {}

    # Coerce both sides to str to avoid int/str key mismatches
    user_cluster = {str(uid): int(cid) for uid, cid in zip(userids, labels)}
    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].astype(str).map(user_cluster)
    sc["artist_name_normalized"] = sc["artist_name"].str.strip().str.lower()

    merged = sc.merge(
        artist_genres[["artist_name_normalized", "genres"]],
        on="artist_name_normalized",
        how="left",
    )
    merged["genres"] = merged["genres"].apply(lambda x: x if isinstance(x, list) else [])
    exploded = merged.explode("genres").dropna(subset=["genres"])
    exploded = exploded[exploded["genres"] != ""]

    top_genres = exploded["genres"].value_counts().head(top_n_genres).index.tolist()
    filtered = exploded[exploded["genres"].isin(top_genres)]

    pivot = (
        filtered.groupby(["cluster", "genres"])
        .size()
        .reset_index(name="count")
    )
    pivot_wide = pivot.pivot(index="cluster", columns="genres", values="count").fillna(0)
    pivot_wide = pivot_wide.div(pivot_wide.sum(axis=1), axis=0)  # normalise rows
    pivot_wide = pivot_wide[pivot_wide.index != -1]
    pivot_wide.index = [
        cluster_names.get(int(c), f"Cluster {c}") for c in pivot_wide.index
    ]

    fig = px.bar(
        pivot_wide.reset_index().melt(id_vars="cluster", var_name="genre", value_name="proportion"),
        x="cluster",
        y="proportion",
        color="genre",
        title=title,
        labels={"cluster": "Listener Segment", "proportion": "Proportion of Plays"},
        color_discrete_sequence=_PALETTE,
        template="plotly_white",
    )
    fig.update_layout(barmode="stack")
    return fig


def plot_temporal_heatmap(
    scrobbles: pd.DataFrame,
    labels: np.ndarray,
    userids: list[str],
    cluster_id: int,
    cluster_name: str | None = None,
) -> go.Figure:
    """
    Heatmap of mean listening activity: hours (x) × days-of-week (y) for one cluster.
    Values are average plays per user per hour-day cell so patterns are
    comparable across clusters of different sizes.
    """
    # Coerce both sides to str to avoid int/str key mismatches
    user_cluster = {str(uid): int(cid) for uid, cid in zip(userids, labels)}
    n_cluster_users = sum(1 for v in user_cluster.values() if v == cluster_id)

    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].astype(str).map(user_cluster)
    cluster_sc = sc[sc["cluster"] == cluster_id].copy()

    cluster_sc["hour"] = pd.to_datetime(cluster_sc["timestamp"]).dt.hour
    cluster_sc["dow"] = pd.to_datetime(cluster_sc["timestamp"]).dt.dayofweek

    raw_counts = (
        cluster_sc.groupby(["dow", "hour"])
        .size()
        .reset_index(name="plays")
        .pivot(index="dow", columns="hour", values="plays")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )

    # Normalise: average plays per user so all clusters are on the same scale
    heatmap_data = raw_counts / max(n_cluster_users, 1)

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    name = cluster_name or f"Cluster {cluster_id}"

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=dow_labels,
            colorscale="Blues",
            colorbar=dict(title="Avg plays<br>per user"),
        )
    )
    fig.update_layout(
        title=f"Listening Activity Pattern: {name}",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_white",
    )
    return fig


def save_figure(fig: go.Figure, path: str | Path, formats: list[str] | None = None) -> None:
    """Save a Plotly figure to HTML and optionally to static formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    for fmt in (formats or []):
        fig.write_image(str(path.with_suffix(f".{fmt}")))
