"""
enrichment_pipeline.py
----------------------
Orchestrates the full data enrichment workflow:
  1. Discover a random sample of users via the Last.fm tag→artist→fan graph
     (or fall back to the baseline TSV if a raw file is present)
  2. Fetch updated scrobbles via Last.fm API for the last N years
  3. Merge baseline + updated, deduplicate
  4. Fetch artist genres from Spotify
  5. Fetch audio features from Spotify for a representative track sample
  6. Save enriched scrobbles + lookup tables to CSV

Run this script once; subsequent analysis reads from data/processed/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.data.loader import load_config, load_scrobbles, load_profiles
from src.data.lastfm_client import build_network, fetch_all_users, discover_random_users
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
    # 1. Determine which users to fetch scrobbles for
    #    Priority:
    #      a) Random discovery via Last.fm API (sample_users is set)
    #      b) Fall back to baseline TSV if it exists
    # ------------------------------------------------------------------
    sample_n = dc.get("sample_users")  # e.g. 10_000, or None to use full baseline

    baseline_path = Path(paths["raw_scrobbles"])
    profile_path = Path(paths["raw_profiles"])

    baseline = pd.DataFrame()
    userids: list[str] = []

    lfm_network = build_network()

    if sample_n:
        # ── Random API-based sampling ────────────────────────────────
        logger.info(
            "Random user discovery requested (target: %d users). "
            "Skipping baseline TSV.",
            sample_n,
        )
        userids = discover_random_users(
            network=lfm_network,
            target_n=sample_n,
            n_tags=dc.get("discovery_n_tags", 50),
            n_artists_per_tag=dc.get("discovery_n_artists_per_tag", 5),
            n_fans_per_artist=dc.get("discovery_n_fans_per_artist", 100),
            seed=dc.get("discovery_seed", 42),
            request_delay=dc["lastfm_request_delay"],
        )
    else:
        # ── Fall back to baseline TSV ────────────────────────────────
        if baseline_path.exists():
            logger.info("Loading baseline scrobbles from %s...", baseline_path)
            baseline = load_scrobbles(
                baseline_path,
                save_csv=processed_dir / "scrobbles_baseline.csv",
            )
        else:
            logger.warning(
                "Baseline TSV not found at %s — will use profile file for user list.",
                baseline_path,
            )

        if profile_path.exists():
            profiles_df = load_profiles(
                profile_path,
                save_csv=processed_dir / "profiles.csv",
            )
            userids = profiles_df["userid"].dropna().unique().tolist()
            logger.info("Found %d users in profile file.", len(userids))
        elif not baseline.empty:
            userids = baseline["userid"].unique().tolist()
            logger.info(
                "No profile file; using %d users from baseline scrobbles.", len(userids)
            )
        else:
            raise RuntimeError(
                "Neither profile TSV nor baseline scrobbles found, and sample_users "
                "is not set. Place raw data in data/raw/ or set sample_users in config."
            )

    # ------------------------------------------------------------------
    # 2. Save profiles for API-discovered users (stub when no TSV)
    # ------------------------------------------------------------------
    profiles_csv = processed_dir / "profiles.csv"
    if sample_n and not profiles_csv.exists():
        stub = pd.DataFrame({"userid": userids})
        for col in ["gender", "age", "country", "signup"]:
            stub[col] = None
        stub.to_csv(profiles_csv, index=False)
        logger.info("Stub profiles.csv saved for %d API-discovered users.", len(userids))

    # ------------------------------------------------------------------
    # 3. Fetch updated Last.fm scrobbles (last N years)
    # ------------------------------------------------------------------
    logger.info(
        "Fetching Last.fm scrobbles for last %d years for %d users...",
        dc["lookback_years"],
        len(userids),
    )
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

    scrobbles.to_csv(paths["processed_scrobbles"], index=False)
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
        cache_path=processed_dir / "spotify_artist_cache.csv",
        request_delay=0.1,
    )
    # Serialise genres list → JSON string for CSV output
    df_genres_out = artist_genres.copy()
    df_genres_out["genres"] = df_genres_out["genres"].apply(json.dumps)
    df_genres_out.to_csv(processed_dir / "artist_genres.csv", index=False)
    logger.info("Artist genres saved.")

    # ------------------------------------------------------------------
    # 6. Sample tracks for audio feature enrichment
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
    track_pairs = list(set(zip(track_sample["artist_name"], track_sample["track_name"])))
    logger.info(
        "Fetching Spotify audio features for %d unique (artist, track) pairs...",
        len(track_pairs),
    )
    audio_features = fetch_audio_features(
        sp,
        track_pairs,
        cache_path=processed_dir / "spotify_audio_features_cache.csv",
        request_delay=0.1,
    )
    audio_features.to_csv(processed_dir / "audio_features.csv", index=False)
    logger.info("Audio features saved.")

    logger.info("Enrichment pipeline complete. All outputs in %s/", processed_dir)


if __name__ == "__main__":
    run_enrichment()
