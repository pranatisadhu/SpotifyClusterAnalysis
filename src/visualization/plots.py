"""
plots.py
--------
Visualisations for the cluster analysis.

Plotly figures are saved as both interactive HTML and high-quality EPS
(via kaleido).  Matplotlib helpers return Axes so callers can save with
plt.savefig(..., format='eps').

Functions:
  - plot_umap_clusters         : 2D UMAP scatter coloured by cluster
  - plot_cluster_radar         : radar/spider chart of cluster feature profiles
  - plot_feature_importance    : horizontal bar chart of RF feature importance
  - plot_elbow                 : KMeans elbow + silhouette dual-axis chart
  - plot_cluster_heatmap       : heatmap of normalised feature means per cluster
  - plot_genre_distribution    : stacked bar of top genres per cluster
  - plot_temporal_heatmap      : listening activity heatmap (hour × dow) per cluster
  - plot_demographic_breakdown : cluster × demographic bar/box charts (matplotlib)
  - save_figure                : save Plotly figure to HTML + EPS
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


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def plot_umap_clusters(
    umap_coords: np.ndarray,
    labels: np.ndarray,
    cluster_names: dict[int, str] | None = None,
    userids: list[str] | None = None,
    title: str = "UMAP Projection Of User Listening Profiles",
) -> go.Figure:
    """2D UMAP scatter plot coloured by cluster label."""
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
    fig.update_layout(
        title_font_size=16,
        legend_title_text="Segment",
    )
    return fig


def plot_cluster_radar(
    cluster_summary: pd.DataFrame,
    features: list[str],
    cluster_names: dict[int, str] | None = None,
    title: str = "Cluster Feature Profiles (Normalised)",
) -> go.Figure:
    """Radar chart comparing cluster profiles across selected features."""
    cluster_names = cluster_names or {}
    fig = go.Figure()

    for cid in cluster_summary.index:
        if cid == -1:
            continue
        means = []
        for feat in features:
            val = (
                cluster_summary.loc[cid, (feat, "mean")]
                if (feat, "mean") in cluster_summary.columns
                else 0
            )
            means.append(val)

        label = cluster_names.get(cid, f"Cluster {cid}")
        fig.add_trace(
            go.Scatterpolar(
                r=means + [means[0]],
                theta=features + [features[0]],
                name=label,
                fill="toself",
                opacity=0.6,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        title_font_size=16,
        template="plotly_white",
        showlegend=True,
    )
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Top Features Distinguishing Listener Segments",
) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
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
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        title_font_size=16,
    )
    return fig


def plot_elbow(
    elbow_df: pd.DataFrame,
    title: str = "KMeans Model Selection: Elbow And Silhouette",
) -> go.Figure:
    """Dual-axis chart: inertia (left) and silhouette score (right) vs k."""
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
        title_font_size=16,
        xaxis_title="Number Of Clusters (k)",
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
    title: str = "Normalised Feature Means By Cluster (Z-Score)",
) -> go.Figure:
    """Heatmap where rows = clusters, columns = features, values = z-scored means."""
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
            colorbar=dict(title="Z-Score"),
        )
    )
    fig.update_layout(
        title=title,
        title_font_size=16,
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
    title: str = "Top Genre Distribution By Listener Segment",
) -> go.Figure:
    """Stacked bar chart: proportion of each top genre per cluster."""
    cluster_names = cluster_names or {}

    user_cluster = dict(zip(userids, labels))
    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].map(user_cluster)
    # Drop rows where cluster is not mapped (users not in clustering result)
    sc = sc.dropna(subset=["cluster"])
    sc["cluster"] = sc["cluster"].astype(int)
    sc["artist_name_normalized"] = sc["artist_name"].str.strip().str.lower()

    merged = sc.merge(
        artist_genres[["artist_name_normalized", "genres"]],
        on="artist_name_normalized",
        how="left",
    )
    merged["genres"] = merged["genres"].apply(lambda x: x if isinstance(x, list) else [])
    exploded = merged.explode("genres").dropna(subset=["genres"])
    exploded = exploded[exploded["genres"].astype(str).str.strip() != ""]

    if exploded.empty:
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white")
        return fig

    top_genres = exploded["genres"].value_counts().head(top_n_genres).index.tolist()
    filtered = exploded[exploded["genres"].isin(top_genres)]

    pivot = (
        filtered.groupby(["cluster", "genres"])
        .size()
        .reset_index(name="count")
    )
    pivot_wide = pivot.pivot(index="cluster", columns="genres", values="count").fillna(0)
    pivot_wide = pivot_wide.div(pivot_wide.sum(axis=1), axis=0)
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
        labels={"cluster": "Listener Segment", "proportion": "Proportion Of Plays", "genre": "Genre"},
        color_discrete_sequence=_PALETTE,
        template="plotly_white",
    )
    fig.update_layout(
        barmode="stack",
        title_font_size=16,
        xaxis_title="Listener Segment",
        yaxis_title="Proportion Of Genre Plays",
    )
    return fig


def plot_temporal_heatmap(
    scrobbles: pd.DataFrame,
    labels: np.ndarray,
    userids: list[str],
    cluster_id: int,
    cluster_name: str | None = None,
) -> go.Figure:
    """
    Heatmap of mean listening activity (avg plays per user) for one cluster.

    Values are normalised by the number of users in the cluster so that
    heatmaps are directly comparable across segments of different sizes.
    Rows = days of week (Mon–Sun), columns = hour of day (00:00–23:00).
    """
    user_cluster = dict(zip(userids, labels))
    sc = scrobbles.copy()
    sc["cluster"] = sc["userid"].map(user_cluster)
    cluster_sc = sc[sc["cluster"] == cluster_id].copy()

    if cluster_sc.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Listening Activity Pattern: {cluster_name or f'Cluster {cluster_id}'} (No Data)",
            template="plotly_white",
        )
        return fig

    cluster_sc["hour"] = cluster_sc["timestamp"].dt.hour
    cluster_sc["dow"] = cluster_sc["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun

    # Total plays per (day, hour)
    heatmap_data = (
        cluster_sc.groupby(["dow", "hour"])
        .size()
        .reset_index(name="plays")
        .pivot(index="dow", columns="hour", values="plays")
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )

    # Normalise by user count → avg plays per user per slot
    n_users = cluster_sc["userid"].nunique()
    heatmap_data = heatmap_data / max(n_users, 1)

    dow_labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    name = cluster_name or f"Cluster {cluster_id}"

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=dow_labels,
            colorscale="Viridis",
            colorbar=dict(title="Avg Plays Per User"),
        )
    )
    fig.update_layout(
        title=f"Listening Activity Pattern: {name}  (n={n_users} Users)",
        title_font_size=16,
        xaxis_title="Hour Of Day",
        yaxis_title="Day Of Week",
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Matplotlib demographic breakdown (returns matplotlib Figure for EPS export)
# ---------------------------------------------------------------------------

def plot_demographic_breakdown(
    labels_df: pd.DataFrame,
    profiles: pd.DataFrame,
) -> dict[str, "matplotlib.figure.Figure"]:
    """
    Produce matplotlib figures showing how clusters split across demographic
    dimensions (gender, age, country).  Returns a dict of {name: Figure} so
    callers can save each individually as EPS.

    Parameters
    ----------
    labels_df : DataFrame with columns [userid, cluster_name]
    profiles  : DataFrame with columns [userid, gender, age, country]
    """
    import matplotlib.pyplot as plt

    demo = profiles.set_index("userid")[["gender", "age", "country"]].copy()
    labeled = labels_df.set_index("userid").join(demo, how="left")

    figs: dict[str, plt.Figure] = {}
    palette = _PALETTE

    # --- Gender Distribution Per Segment ---
    gender_data = labeled.dropna(subset=["gender"]).copy()
    gender_data = gender_data[gender_data["gender"].str.strip() != ""]
    if not gender_data.empty:
        gender_ct = (
            gender_data.groupby(["cluster_name", "gender"])
            .size()
            .unstack(fill_value=0)
        )
        gender_pct = gender_ct.div(gender_ct.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = palette[: len(gender_pct.columns)]
        gender_pct.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.7)
        ax.set_title("Gender Distribution By Listener Segment", fontsize=13, fontweight="bold")
        ax.set_xlabel("Listener Segment")
        ax.set_ylabel("Percentage Of Users (%)")
        ax.legend(title="Gender", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        figs["segment_gender_distribution"] = fig

    # --- Median Age Per Segment (bar) ---
    age_data = labeled.dropna(subset=["age"]).copy()
    age_data = age_data[(age_data["age"] >= 10) & (age_data["age"] <= 100)]
    if not age_data.empty:
        median_age = age_data.groupby("cluster_name")["age"].median().sort_values()

        fig, ax = plt.subplots(figsize=(8, 5))
        median_age.plot(kind="barh", ax=ax, color=palette[0], edgecolor="white")
        ax.set_title("Median Age Per Listener Segment", fontsize=13, fontweight="bold")
        ax.set_xlabel("Median Age (Years)")
        ax.set_ylabel("Listener Segment")
        plt.tight_layout()
        figs["segment_median_age"] = fig

    # --- Age Distribution Per Segment (overlapping histograms) ---
    if not age_data.empty:
        segments = age_data["cluster_name"].unique()
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, seg in enumerate(sorted(segments)):
            grp = age_data[age_data["cluster_name"] == seg]["age"]
            grp.hist(ax=ax, bins=15, alpha=0.55, label=seg,
                     color=palette[i % len(palette)], edgecolor="white")
        ax.set_title("Age Distribution By Listener Segment", fontsize=13, fontweight="bold")
        ax.set_xlabel("Age (Years)")
        ax.set_ylabel("Number Of Users")
        ax.legend(title="Segment", bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        figs["segment_age_distribution"] = fig

    # --- Top Country Distribution Per Segment ---
    country_data = labeled.dropna(subset=["country"]).copy()
    country_data = country_data[country_data["country"].str.strip() != ""]
    if not country_data.empty:
        top_countries = (
            country_data["country"].value_counts().head(6).index.tolist()
        )
        c_seg = country_data[country_data["country"].isin(top_countries)]
        if not c_seg.empty:
            country_ct = (
                c_seg.groupby(["cluster_name", "country"])
                .size()
                .unstack(fill_value=0)
            )
            country_pct = country_ct.div(country_ct.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(12, 5))
            country_pct.plot(kind="bar", ax=ax, edgecolor="white", width=0.7)
            ax.set_title(
                "Top Country Distribution By Listener Segment",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("Listener Segment")
            ax.set_ylabel("Percentage Of Users (%)")
            ax.legend(title="Country", bbox_to_anchor=(1.01, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=30)
            plt.tight_layout()
            figs["segment_country_distribution"] = fig

    return figs


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(
    fig: go.Figure,
    path: str | Path,
    formats: list[str] | None = None,
) -> None:
    """
    Save a Plotly figure to HTML (interactive) and EPS (publication quality).

    EPS export requires the kaleido package.  If kaleido is unavailable the
    function warns and falls back to SVG, which is also a lossless vector format.

    Parameters
    ----------
    fig     : Plotly Figure
    path    : base path without extension (extensions are appended automatically)
    formats : list of static formats to export, defaults to ["eps"]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Always save interactive HTML
    fig.write_html(str(path.with_suffix(".html")))

    # Static vector export — default to EPS
    for fmt in formats or ["eps"]:
        out_path = str(path.with_suffix(f".{fmt}"))
        try:
            fig.write_image(out_path)
            logger.debug("Saved static figure: %s", out_path)
        except Exception as exc:
            # kaleido may not support eps in all build variants; fall back to svg
            logger.warning(
                "Could not export figure as %s (%s). Trying SVG fallback.", fmt, exc
            )
            try:
                svg_path = str(path.with_suffix(".svg"))
                fig.write_image(svg_path)
                logger.info("Saved SVG fallback: %s", svg_path)
            except Exception as exc2:
                logger.warning("SVG export also failed: %s", exc2)
