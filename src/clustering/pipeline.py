"""
pipeline.py
-----------
Full clustering pipeline:
  1. Preprocessing: impute missing values, scale features, remove demographic cols
  2. Dimensionality reduction: PCA for multicollinearity, optional UMAP for viz
  3. Clustering: KMeans (sweep k) and HDBSCAN
  4. Model selection: best k via silhouette + elbow (KMeans), auto-select HDBSCAN
  5. Returns annotated feature matrix with cluster labels and UMAP coordinates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan as hdbscan_lib
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Columns that are descriptive only — excluded from clustering features
_DESCRIPTIVE_COLS = ["gender", "age", "country"]


@dataclass
class ClusterResult:
    """Container for all outputs from the clustering pipeline."""
    labels: np.ndarray                  # -1 = noise (HDBSCAN only)
    algorithm: str
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    feature_matrix_scaled: np.ndarray
    feature_names: list[str]
    umap_coords: np.ndarray | None = None  # (n_users, 2)
    pca_variance_explained: np.ndarray | None = None
    model: Any = field(default=None, repr=False)


def _select_features(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Select numeric clustering features; exclude descriptive demographic columns.
    """
    drop = [c for c in _DESCRIPTIVE_COLS if c in feature_matrix.columns]
    X = feature_matrix.drop(columns=drop).select_dtypes(include=[np.number])
    return X, X.columns.tolist()


def preprocess(
    feature_matrix: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Impute NaNs and standardise all clustering features.

    Returns
    -------
    X_scaled : (n_users, n_features) numpy array
    feature_names : list of column names
    """
    X, feature_names = _select_features(feature_matrix)

    # Median imputation for sparse features
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, feature_names


def reduce_with_pca(
    X_scaled: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, PCA]:
    """
    PCA to remove multicollinearity; retain components explaining
    `variance_threshold` of total variance.
    """
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(
        "PCA: reduced %d features → %d components (%.1f%% variance retained).",
        X_scaled.shape[1],
        X_pca.shape[1],
        pca.explained_variance_ratio_.sum() * 100,
    )
    return X_pca, pca


def embed_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """UMAP embedding for 2D visualisation."""
    if not _UMAP_AVAILABLE:
        logger.warning("umap-learn not installed; skipping UMAP embedding.")
        return np.zeros((X.shape[0], n_components))
    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def cluster_kmeans(
    X: np.ndarray,
    k_range: tuple[int, int] = (3, 10),
    n_init: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, int, dict[int, float], dict[int, float]]:
    """
    KMeans sweep over k_range; select k via silhouette score.

    Returns
    -------
    best_labels, best_k, silhouette_scores, inertias
    """
    silhouette_scores: dict[int, float] = {}
    inertias: dict[int, float] = {}
    best_labels = None
    best_k = k_range[0]
    best_sil = -1.0

    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        silhouette_scores[k] = sil
        inertias[k] = km.inertia_
        logger.info("  KMeans k=%d → silhouette=%.4f, inertia=%.1f", k, sil, km.inertia_)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    logger.info("KMeans selected k=%d (silhouette=%.4f).", best_k, best_sil)
    return best_labels, best_k, silhouette_scores, inertias


def cluster_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """HDBSCAN clustering (density-based; automatically determines cluster count)."""
    if not _HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Run: pip install hdbscan")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    return clusterer.fit_predict(X)


def run_clustering_pipeline(
    feature_matrix: pd.DataFrame,
    algorithm: str = "kmeans",
    use_pca: bool = True,
    use_umap_viz: bool = True,
    config: dict | None = None,
) -> ClusterResult:
    """
    End-to-end clustering pipeline.

    Parameters
    ----------
    feature_matrix : user-level feature matrix (from features/builder.py)
    algorithm : "kmeans" or "hdbscan"
    use_pca : reduce dimensions with PCA before clustering
    use_umap_viz : compute UMAP 2D embedding for visualisation
    config : optional dict from config.yaml['clustering']

    Returns
    -------
    ClusterResult with labels, metrics, and UMAP coords
    """
    cfg = config or {}
    km_cfg = cfg.get("kmeans", {})
    hdb_cfg = cfg.get("hdbscan", {})
    umap_cfg = cfg.get("umap", {})
    k_range = tuple(cfg.get("n_clusters_range", (3, 10)))

    logger.info("Preprocessing feature matrix (%d users)...", len(feature_matrix))
    X_scaled, feature_names = preprocess(feature_matrix)

    pca_obj = None
    if use_pca:
        X_input, pca_obj = reduce_with_pca(X_scaled)
    else:
        X_input = X_scaled

    # UMAP embedding — computed before HDBSCAN so the low-dimensional
    # representation can be used as clustering input (avoids curse of
    # dimensionality that causes all points to be labelled noise in ≥10-D).
    umap_coords = None
    if use_umap_viz:
        logger.info("Computing UMAP 2D embedding...")
        umap_coords = embed_umap(
            X_input,
            n_components=umap_cfg.get("n_components", 2),
            n_neighbors=umap_cfg.get("n_neighbors", 15),
            min_dist=umap_cfg.get("min_dist", 0.1),
            random_state=umap_cfg.get("random_state", 42),
        )

    # Cluster
    if algorithm == "kmeans":
        k_range_val = (
            km_cfg.get("n_clusters_range", [3, 10])[0],
            km_cfg.get("n_clusters_range", [3, 10])[1],
        )
        labels, n_clusters, sil_scores, inertias = cluster_kmeans(
            X_input,
            k_range=k_range_val,
            n_init=km_cfg.get("n_init", 20),
            random_state=km_cfg.get("random_state", 42),
        )
    elif algorithm == "hdbscan":
        # Prefer the 2-D UMAP embedding as HDBSCAN input; fall back to
        # PCA-reduced features when UMAP is unavailable.
        hdb_input = umap_coords if umap_coords is not None else X_input
        labels = cluster_hdbscan(
            hdb_input,
            min_cluster_size=hdb_cfg.get("min_cluster_size", 10),
            min_samples=hdb_cfg.get("min_samples", 5),
            metric=hdb_cfg.get("metric", "euclidean"),
        )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("HDBSCAN found %d clusters (%d noise points).",
                    n_clusters, (labels == -1).sum())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose 'kmeans' or 'hdbscan'.")

    # Compute evaluation metrics (excluding noise for HDBSCAN)
    eval_input = umap_coords if (algorithm == "hdbscan" and umap_coords is not None) else X_input
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil = silhouette_score(eval_input[mask], labels[mask])
        db = davies_bouldin_score(eval_input[mask], labels[mask])
        ch = calinski_harabasz_score(eval_input[mask], labels[mask])
    else:
        sil, db, ch = 0.0, 0.0, 0.0

    logger.info(
        "Cluster metrics — Silhouette: %.4f | Davies-Bouldin: %.4f | Calinski-Harabasz: %.1f",
        sil, db, ch,
    )

    return ClusterResult(
        labels=labels,
        algorithm=algorithm,
        n_clusters=n_clusters,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        feature_matrix_scaled=X_scaled,
        feature_names=feature_names,
        umap_coords=umap_coords,
        pca_variance_explained=(
            pca_obj.explained_variance_ratio_ if pca_obj else None
        ),
    )
