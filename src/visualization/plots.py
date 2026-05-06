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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _cluster_color_map(labels: np.ndarray) -> dict[int, str]:
    unique = sorted(set(labels))
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(unique)}


def _uid_str(uid) -> str:
    """Normalise a user ID to a plain integer string regardless of dtype.
    Handles int64 (1000002), float64 (1000002.0), and str ('1000002') equally."""
    s = str(uid)
    return s[:-2] if s.endswith(".0") else s


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

    user_cluster = {_uid_str(uid): int(cid) for uid, cid in zip(userids, labels)}
    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].apply(_uid_str).map(user_cluster)
    sc["artist_name_normalized"] = sc["artist_name"].str.strip().str.lower()

    # Mapping coverage check — printed so mismatches are immediately visible
    mapped = sc["cluster"].notna().sum()
    logger.info(
        "Genre chart — %d / %d scrobbles mapped to a cluster; per-cluster: %s",
        mapped, len(sc),
        sc["cluster"].value_counts(dropna=False).sort_index().to_dict(),
    )

    merged = sc.merge(
        artist_genres[["artist_name_normalized", "genres"]],
        on="artist_name_normalized",
        how="left",
    )

    def _parse_genres(x) -> list:
        """Accept list, numpy array, JSON string, or comma-separated string."""
        if x is None:
            return []
        if isinstance(x, (list, np.ndarray)):
            return [g for g in x if g and str(g).strip()]
        if isinstance(x, str) and x.strip():
            import json
            try:
                parsed = json.loads(x)
                if isinstance(parsed, list):
                    return [g for g in parsed if g]
            except (json.JSONDecodeError, ValueError):
                pass
            return [g.strip() for g in x.split(",") if g.strip()]
        return []

    merged["genres"] = merged["genres"].apply(_parse_genres)
    exploded = merged.explode("genres")
    exploded = exploded[exploded["genres"].notna() & (exploded["genres"] != "")]

    if exploded.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            annotations=[dict(text="No genre data available for these artists",
                              showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")],
        )
        return fig

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
    pivot_wide.index.name = "cluster"

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


def plot_genre_features(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    title: str = "Genre Diversity Features By Listener Segment",
) -> go.Figure:
    """
    Grouped bar chart of genre diversity metrics (unique_genres, genre_entropy,
    genre_concentration_5, avg_genre_tags_per_play) per cluster.

    Uses pre-computed columns in the feature matrix — no raw scrobbles needed.
    Each metric is min-max normalised so all four bars share a 0–1 y-axis.
    """
    cluster_names = cluster_names or {}
    genre_cols = [c for c in [
        "unique_genres", "genre_entropy",
        "genre_concentration_5", "avg_genre_tags_per_play",
    ] if c in feature_matrix.columns]

    if not genre_cols:
        fig = go.Figure()
        fig.update_layout(
            title=title, template="plotly_white",
            annotations=[dict(text="No genre feature columns found in feature matrix.",
                              showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")],
        )
        return fig

    df = feature_matrix.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]

    means = df.groupby("cluster")[genre_cols].mean()

    # Min-max normalise each metric across clusters so all fit on 0-1 axis
    normed = (means - means.min()) / (means.max() - means.min() + 1e-9)

    fig = go.Figure()
    for col in genre_cols:
        x_labels = [
            cluster_names.get(int(cid), f"Cluster {cid}") for cid in normed.index
        ]
        fig.add_trace(go.Bar(
            name=col.replace("_", " ").title(),
            x=x_labels,
            y=normed[col].values,
        ))

    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Listener Segment",
        yaxis_title="Normalised Score (0–1)",
        template="plotly_white",
        legend_title="Feature",
    )
    return fig


def plot_temporal_heatmap(
    scrobbles: pd.DataFrame,
    labels: np.ndarray,
    userids: list[str],
    cluster_id: int,
    cluster_name: str | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """
    Heatmap of mean listening activity: hours (x) × days-of-week (y) for one cluster.
    Values are average plays per user per hour-day cell so patterns are
    comparable across clusters of different sizes.

    Parameters
    ----------
    zmax : shared colour-scale ceiling; pass the same value to all clusters so
           their heatmaps are directly comparable.  If None, defaults to the
           maximum value in this cluster's data (minimum floor 0.01).
    """
    user_cluster = {_uid_str(uid): int(cid) for uid, cid in zip(userids, labels)}
    n_cluster_users = sum(1 for v in user_cluster.values() if v == cluster_id)

    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].apply(_uid_str).map(user_cluster)
    cluster_sc = sc[sc["cluster"] == cluster_id].copy()

    name = cluster_name or f"Cluster {cluster_id}"

    # Guard: no scrobbles found for this cluster's users
    if cluster_sc.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Listening Activity Pattern: {name}",
            template="plotly_white",
            annotations=[dict(
                text=(
                    f"No scrobbles found for {name}.<br>"
                    "Restart the kernel and re-run all cells, then check<br>"
                    "the genre diagnostic cell (Section 5) for userid mapping info."
                ),
                showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(size=13),
            )],
        )
        return fig

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

    # Colour-scale ceiling: use provided zmax or derive from this cluster's data
    _zmax = zmax if zmax is not None else max(float(heatmap_data.values.max()), 0.01)

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=dow_labels,
            colorscale="Blues",
            zmin=0,
            zmax=_zmax,
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


def save_figure(
    fig: go.Figure,
    path: str | Path,
    formats: list[str] | None = None,
) -> None:
    """Save a Plotly figure to HTML + static formats (default: EPS).

    Requires kaleido for static export: pip install kaleido
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    for fmt in (formats if formats is not None else ["eps"]):
        fig.write_image(str(path.with_suffix(f".{fmt}")))
