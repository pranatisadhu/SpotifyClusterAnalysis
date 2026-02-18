"""
enrichment_pipeline.py
----------------------
Orchestrates the full data enrichment workflow:
  1. Load baseline scrobbles (existing TSV or already-fetched Parquet)
  2. Fetch updated scrobbles via Last.fm API for the last N years
  3. Merge baseline + updated, deduplicate
  4. Fetch artist genres from Spotify
  5. Fetch audio features from Spotify for a representative track sample
  6. Save enriched scrobbles + lookup tables to Parquet

Run this script once; subsequent analysis reads from data/processed/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.loader import load_config, load_scrobbles, load_profiles
from src.data.lastfm_client import build_network, fetch_all_users
from src.data.spotify_client import build_client, fetch_artist_genres, fetch_audio_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_enrichment(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    dc = cfg["data_collection"]

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load baseline scrobbles from the original 2009 TSV (optional)
    # ------------------------------------------------------------------
    baseline_path = Path(paths["raw_scrobbles"])
    if baseline_path.exists():
        logger.info("Loading baseline scrobbles from %s...", baseline_path)
        baseline = load_scrobbles(
            baseline_path,
            save_parquet=processed_dir / "scrobbles_baseline.parquet",
        )
    else:
        logger.warning(
            "Baseline TSV not found at %s — skipping. Will use API data only.", baseline_path
        )
        baseline = pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. Load user profiles
    # ------------------------------------------------------------------
    profile_path = Path(paths["raw_profiles"])
    if profile_path.exists():
        profiles = load_profiles(
            profile_path,
            save_parquet=processed_dir / "profiles.parquet",
        )
        userids = profiles["userid"].dropna().unique().tolist()
        logger.info("Found %d users in profile file.", len(userids))
    elif not baseline.empty:
        userids = baseline["userid"].unique().tolist()
        logger.info("No profile file; using %d users from baseline scrobbles.", len(userids))
    else:
        raise RuntimeError(
            "Neither profile TSV nor baseline scrobbles found. "
            "Place the raw data in data/raw/ before running enrichment."
        )

    # ------------------------------------------------------------------
    # 3. Fetch updated Last.fm scrobbles (last N years)
    # ------------------------------------------------------------------
    logger.info(
        "Fetching Last.fm scrobbles for last %d years for %d users...",
        dc["lookback_years"],
        len(userids),
    )
    lfm_network = build_network()
    updated = fetch_all_users(
        network=lfm_network,
        userids=userids,
        lookback_years=dc["lookback_years"],
        page_size=dc["lastfm_page_size"],
        request_delay=dc["lastfm_request_delay"],
        min_scrobbles=dc["min_scrobbles_threshold"],
        save_dir=processed_dir / "lastfm_user_cache",
        resume=True,
    )

    # ------------------------------------------------------------------
    # 4. Merge baseline + updated, deduplicate
    # ------------------------------------------------------------------
    frames = [f for f in [baseline, updated] if not f.empty]
    scrobbles = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    scrobbles = scrobbles.drop_duplicates(
        subset=["userid", "timestamp", "artist_name", "track_name"]
    ).sort_values(["userid", "timestamp"]).reset_index(drop=True)

    scrobbles.to_parquet(paths["processed_scrobbles"], index=False)
    logger.info(
        "Merged scrobbles: %d rows, %d users → %s",
        len(scrobbles),
        scrobbles["userid"].nunique(),
        paths["processed_scrobbles"],
    )

    # ------------------------------------------------------------------
    # 5. Fetch artist genres from Spotify
    # ------------------------------------------------------------------
    unique_artists = scrobbles["artist_name"].dropna().unique().tolist()
    logger.info("Fetching Spotify genres for %d unique artists...", len(unique_artists))
    sp = build_client()
    artist_genres = fetch_artist_genres(
        sp,
        unique_artists,
        cache_path=processed_dir / "spotify_artist_cache.parquet",
        request_delay=0.1,
    )
    artist_genres.to_parquet(processed_dir / "artist_genres.parquet", index=False)
    logger.info("Artist genres saved.")

    # ------------------------------------------------------------------
    # 6. Sample tracks for audio feature enrichment
    #    (fetching features for every play in 19M+ rows is prohibitive;
    #     sample the top N most-played unique tracks per user)
    # ------------------------------------------------------------------
    TOP_TRACKS_PER_USER = 20
    track_sample = (
        scrobbles.groupby(["userid", "artist_name", "track_name"])
        .size()
        .reset_index(name="play_count")
        .sort_values(["userid", "play_count"], ascending=[True, False])
        .groupby("userid")
        .head(TOP_TRACKS_PER_USER)
    )
    track_pairs = list(
        zip(track_sample["artist_name"], track_sample["track_name"])
    )
    # Deduplicate (same track can appear for multiple users)
    track_pairs = list(set(track_pairs))
    logger.info(
        "Fetching Spotify audio features for %d unique (artist, track) pairs...",
        len(track_pairs),
    )
    audio_features = fetch_audio_features(
        sp,
        track_pairs,
        cache_path=processed_dir / "spotify_audio_features_cache.parquet",
        request_delay=0.1,
    )
    audio_features.to_parquet(processed_dir / "audio_features.parquet", index=False)
    logger.info("Audio features saved.")

    logger.info("Enrichment pipeline complete. All outputs in %s/", processed_dir)


if __name__ == "__main__":
    run_enrichment()
