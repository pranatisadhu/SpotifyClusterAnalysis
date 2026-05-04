"""
plots.py
--------
Interactive Plotly visualisations for the cluster analysis.

All functions return a plotly Figure object — call .show() or .write_html()
as needed.

Functions:
  - plot_umap_clusters        : 2D UMAP scatter coloured by cluster
  - plot_cluster_radar        : radar/spider chart of cluster feature profiles
  - plot_feature_importance   : horizontal bar chart of RF feature importance
  - plot_elbow                : KMeans elbow + silhouette dual-axis chart
  - plot_cluster_heatmap      : heatmap of normalised feature means per cluster
  - plot_genre_distribution   : stacked bar of top genres per cluster
  - plot_genre_features       : grouped bar of per-cluster genre metric ratios
  - plot_temporal_heatmap     : listening activity by hour × day-of-week per cluster
  - plot_temporal_ratios      : grouped bar of morning/evening/weekend ratios per cluster
  - plot_hour_distribution    : normalised hour-of-day listening curves per cluster
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

    Parameters
    ----------
    cluster_summary : output of evaluation.summarise_clusters (mean values)
    features : list of feature names to include on the radar
    """
    # Extract mean values for selected features
    cluster_names = cluster_names or {}
    fig = go.Figure()

    for cid in cluster_summary.index:
        if cid == -1:
            continue
        means = []
        for feat in features:
            val = cluster_summary.loc[cid, (feat, "mean")] if (feat, "mean") in cluster_summary.columns else 0
            means.append(val)

        # Normalise to 0–1 range across clusters for comparability
        label = cluster_names.get(cid, f"Cluster {cid}")
        fig.add_trace(
            go.Scatterpolar(
                r=means + [means[0]],  # close the polygon
                theta=features + [features[0]],
                name=label,
                fill="toself",
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
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

    # Build user → cluster mapping
    user_cluster = dict(zip(userids, labels))
    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].map(user_cluster)
    sc = sc[sc["cluster"].notna()].copy()
    sc["cluster"] = sc["cluster"].astype(int)
    sc["artist_name_normalized"] = sc["artist_name"].str.strip().str.lower()

    merged = sc.merge(
        artist_genres[["artist_name_normalized", "genres"]],
        on="artist_name_normalized",
        how="left",
    )
    merged["genres"] = merged["genres"].apply(lambda x: x if isinstance(x, list) else [])
    exploded = merged.explode("genres").dropna(subset=["genres"])
    exploded = exploded[exploded["genres"] != ""]

    if exploded.empty:
        match_rate = merged["genres"].apply(len).gt(0).mean()
        raise ValueError(
            f"No genre data found after joining scrobbles to artist_genres "
            f"(artist name match rate: {match_rate:.1%}). "
            "Check that artist_genres contains an 'artist_name_normalized' column "
            "whose values are lowercase-stripped and match the scrobbles 'artist_name' field."
        )

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
    """
    user_cluster = dict(zip(userids, labels))
    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].map(user_cluster)
    sc = sc[sc["cluster"].notna()].copy()
    sc["cluster"] = sc["cluster"].astype(int)
    cluster_sc = sc[sc["cluster"] == cluster_id].copy()

    cluster_sc["hour"] = cluster_sc["timestamp"].dt.hour
    cluster_sc["dow"] = cluster_sc["timestamp"].dt.dayofweek

    heatmap_data = (
        cluster_sc.groupby(["dow", "hour"])
        .size()
        .reset_index(name="plays")
        .pivot(index="dow", columns="hour", values="plays")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    name = cluster_name or f"Cluster {cluster_id}"

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=dow_labels,
            colorscale="Viridis",
        )
    )
    fig.update_layout(
        title=f"Listening Activity Pattern: {name}",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        template="plotly_white",
    )
    return fig


def plot_genre_features(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    title: str = "Genre Feature Breakdown by Segment",
) -> go.Figure:
    """
    Grouped bar chart comparing per-cluster means of the four genre-level features:
    unique_genres, genre_entropy, genre_concentration_5, and avg_genre_tags_per_play.

    Each metric is normalised to its global mean so all four axes are comparable
    (ratio-to-global-mean, where 1.0 = average).

    Parameters
    ----------
    feature_matrix : user-level feature DataFrame containing genre columns
    labels         : cluster label per user (same order as feature_matrix rows)
    cluster_names  : optional mapping {cluster_id: human_readable_name}
    """
    cluster_names = cluster_names or {}
    genre_cols = [
        c for c in [
            "unique_genres", "genre_entropy", "genre_concentration_5", "avg_genre_tags_per_play"
        ]
        if c in feature_matrix.columns
    ]
    if not genre_cols:
        raise ValueError(
            "feature_matrix contains none of: unique_genres, genre_entropy, "
            "genre_concentration_5, avg_genre_tags_per_play. "
            "Run compute_genre_diversity() to produce these columns."
        )

    df = feature_matrix[genre_cols].copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]

    global_means = df[genre_cols].mean()
    cluster_means = df.groupby("cluster")[genre_cols].mean()

    # Normalise: each value expressed as ratio to global mean
    normed = cluster_means.div(global_means.replace(0, np.nan)).fillna(0)
    normed = normed.reset_index()
    normed["cluster_label"] = normed["cluster"].map(
        lambda c: cluster_names.get(int(c), f"Cluster {c}")
    )

    label_map = {
        "unique_genres": "Unique Genres",
        "genre_entropy": "Genre Entropy",
        "genre_concentration_5": "Top-5 Genre Conc.",
        "avg_genre_tags_per_play": "Avg Tags / Play",
    }
    bar_colors = {
        "unique_genres": "#AB63FA",
        "genre_entropy": "#00CC96",
        "genre_concentration_5": "#EF553B",
        "avg_genre_tags_per_play": "#FFA15A",
    }

    fig = go.Figure()
    for col in genre_cols:
        fig.add_trace(
            go.Bar(
                name=label_map.get(col, col),
                x=normed["cluster_label"],
                y=normed[col],
                marker_color=bar_colors.get(col),
                text=normed[col].map(lambda v: f"{v:.2f}×"),
                textposition="outside",
            )
        )

    fig.add_hline(
        y=1.0,
        line_dash="dot",
        line_color="grey",
        annotation_text="Global mean",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Listener Segment",
        yaxis_title="Ratio to Global Mean",
        barmode="group",
        template="plotly_white",
        legend_title="Genre Metric",
    )
    return fig


def plot_temporal_ratios(
    feature_matrix: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    title: str = "Temporal Listening Ratios by Segment",
) -> go.Figure:
    """
    Grouped bar chart comparing morning_ratio, evening_ratio, and weekend_ratio
    across all clusters.

    Parameters
    ----------
    feature_matrix : user-level feature DataFrame (must contain the ratio columns)
    labels         : cluster label per user (same order as feature_matrix rows)
    cluster_names  : optional mapping {cluster_id: human_readable_name}
    """
    cluster_names = cluster_names or {}
    ratio_cols = [c for c in ["morning_ratio", "evening_ratio", "weekend_ratio"] if c in feature_matrix.columns]
    if not ratio_cols:
        raise ValueError("feature_matrix must contain at least one of: morning_ratio, evening_ratio, weekend_ratio")

    df = feature_matrix[ratio_cols].copy()
    df["cluster"] = labels
    means = df[df["cluster"] != -1].groupby("cluster")[ratio_cols].mean().reset_index()
    means["cluster_label"] = means["cluster"].map(
        lambda c: cluster_names.get(int(c), f"Cluster {c}")
    )

    label_map = {
        "morning_ratio": "Morning (06–12h)",
        "evening_ratio": "Evening (18–00h)",
        "weekend_ratio": "Weekend",
    }

    fig = go.Figure()
    bar_colors = {"morning_ratio": "#FFA15A", "evening_ratio": "#636EFA", "weekend_ratio": "#00CC96"}

    for col in ratio_cols:
        fig.add_trace(
            go.Bar(
                name=label_map.get(col, col),
                x=means["cluster_label"],
                y=means[col],
                marker_color=bar_colors.get(col),
                text=means[col].map(lambda v: f"{v:.1%}"),
                textposition="outside",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Listener Segment",
        yaxis_title="Fraction of Plays",
        yaxis_tickformat=".0%",
        barmode="group",
        template="plotly_white",
        legend_title="Time Period",
    )
    return fig


def plot_hour_distribution(
    scrobbles: pd.DataFrame,
    labels: np.ndarray,
    userids: list[str],
    cluster_names: dict[int, str] | None = None,
    title: str = "Hour-of-Day Listening Distribution by Segment",
) -> go.Figure:
    """
    Normalised hour-of-day distribution (0–23) with one line per cluster, so
    different segments' peak listening times can be compared directly.

    Parameters
    ----------
    scrobbles     : DataFrame with columns [userid, timestamp (datetime64)]
    labels        : cluster label per user (same order as userids)
    userids       : list of user IDs matching the labels array
    cluster_names : optional mapping {cluster_id: human_readable_name}
    """
    cluster_names = cluster_names or {}
    user_cluster = dict(zip(userids, labels))

    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].map(user_cluster)
    sc["hour"] = sc["timestamp"].dt.hour
    sc = sc[sc["cluster"].notna() & (sc["cluster"] != -1)]
    sc["cluster"] = sc["cluster"].astype(int)

    fig = go.Figure()

    for i, cid in enumerate(sorted(sc["cluster"].unique())):
        cluster_sc = sc[sc["cluster"] == cid]
        hour_counts = cluster_sc["hour"].value_counts().reindex(range(24), fill_value=0)
        hour_norm = hour_counts / hour_counts.sum()

        label = cluster_names.get(int(cid), f"Cluster {cid}")
        fig.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=hour_norm.values,
                mode="lines+markers",
                name=label,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
                marker=dict(size=6),
                hovertemplate="Hour %{x}:00 — %{y:.1%}<extra>" + label + "</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Hour of Day",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)],
            tickangle=-45,
        ),
        yaxis=dict(title="Fraction of Plays", tickformat=".0%"),
        template="plotly_white",
        legend_title="Segment",
    )
    return fig


def save_figure(fig: go.Figure, path: str | Path, formats: list[str] | None = None) -> None:
    """Save a Plotly figure to HTML and optionally to static formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path.with_suffix(".html")))
    for fmt in (formats or []):
        fig.write_image(str(path.with_suffix(f".{fmt}")))
