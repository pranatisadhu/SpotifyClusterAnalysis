"""
builder.py
----------
Combines all feature modules into a single user-level feature matrix.
Saves the result to data/processed/user_features.parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features.diversity import compute_artist_diversity, compute_genre_diversity
from src.features.temporal import compute_temporal_features
from src.features.engagement import compute_engagement_features, compute_audio_feature_profile
from src.data.loader import load_config

logger = logging.getLogger(__name__)


def build_feature_matrix(
    scrobbles: pd.DataFrame,
    artist_genres: pd.DataFrame,
    audio_features: pd.DataFrame,
    profiles: pd.DataFrame | None = None,
    config_path: str = "configs/config.yaml",
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the full user-level feature matrix by running all feature modules.

    Parameters
    ----------
    scrobbles : enriched scrobble DataFrame
    artist_genres : artist → genres lookup from Spotify
    audio_features : track → audio features lookup from Spotify
    profiles : optional user profile DataFrame (gender, age, country)
    config_path : path to config.yaml
    save_path : if given, save the feature matrix to this Parquet path

    Returns
    -------
    pd.DataFrame indexed by userid — one row per user, one column per feature
    """
    cfg = load_config(config_path)
    top_n = cfg["feature_engineering"]["top_artists_n"]
    session_gap = cfg["session_detection"]["session_gap_minutes"]
    pca_n = cfg["feature_engineering"]["temporal_pca_components"]

    logger.info("Computing artist diversity features...")
    artist_div = compute_artist_diversity(scrobbles, top_n=top_n)

    logger.info("Computing genre diversity features...")
    genre_div = compute_genre_diversity(scrobbles, artist_genres, top_n=5)

    logger.info("Computing temporal features...")
    temporal = compute_temporal_features(scrobbles, pca_components=pca_n)

    logger.info("Computing engagement and discovery features...")
    engagement = compute_engagement_features(scrobbles, session_gap_minutes=session_gap)

    logger.info("Computing audio feature profile...")
    audio_profile = compute_audio_feature_profile(scrobbles, audio_features)

    # Combine all feature blocks
    feature_matrix = (
        artist_div
        .join(genre_div, how="outer")
        .join(temporal, how="outer")
        .join(engagement, how="outer")
        .join(audio_profile, how="outer")
    )

    # Optionally join demographic features (used as descriptive, not clustering inputs)
    if profiles is not None:
        profiles_indexed = profiles.set_index("userid")[["gender", "age", "country"]]
        feature_matrix = feature_matrix.join(profiles_indexed, how="left")

    feature_matrix = feature_matrix.sort_index()
    logger.info(
        "Feature matrix: %d users × %d features", *feature_matrix.shape
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        feature_matrix.to_parquet(save_path)
        logger.info("Feature matrix saved to %s", save_path)

    return feature_matrix
