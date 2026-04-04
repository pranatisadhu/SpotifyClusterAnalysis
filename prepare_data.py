"""
prepare_data.py
---------------
Run this ONCE from the project root to generate all parquet files needed
for notebooks 02, 03, and 04.

    python prepare_data.py

What it does:
  1. Loads the raw scrobble TSV → data/processed/scrobbles_baseline.parquet
                                → data/processed/scrobbles_updated.parquet  (same file)
  2. Loads the user profile TSV  → data/processed/profiles.parquet
  3. Fetches Spotify artist genres (still works) → data/processed/artist_genres.parquet
  4. Creates an empty audio_features stub → data/processed/audio_features.parquet
     (Spotify removed the audio-features endpoint for apps created after Nov 27 2024)

NOTE: Last.fm API calls are intentionally skipped. The baseline TSV contains
everything needed. If you want to add recent scrobbles later, that can be done
separately — it is not required to run the clustering analysis.

BEFORE RUNNING:
  - Copy .env.example to .env and fill in your SPOTIFY_CLIENT_ID and
    SPOTIFY_CLIENT_SECRET (from https://developer.spotify.com/dashboard)
  - Make sure your raw TSV files are in data/raw/:
      data/raw/userid-timestamp-artid-artname-traid-traname.tsv  (~2.5 GB)
      data/raw/userid-profile.tsv                                 (~38 KB)
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_scrobbles, load_profiles, load_config
from src.data.spotify_client import build_client, check_credentials, fetch_artist_genres

PROCESSED = Path("data/processed")
RAW = Path("data/raw")

SCROBBLES_TSV = RAW / "userid-timestamp-artid-artname-traid-traname.tsv"
PROFILES_TSV = RAW / "userid-profile.tsv"


def step1_load_scrobbles() -> pd.DataFrame:
    """Load raw TSV → parquet. Returns the scrobble DataFrame."""
    baseline_path = PROCESSED / "scrobbles_baseline.parquet"
    updated_path = PROCESSED / "scrobbles_updated.parquet"

    # Fast path: already done
    if updated_path.exists():
        logger.info("scrobbles_updated.parquet already exists — loading from cache.")
        return pd.read_parquet(updated_path)

    if not SCROBBLES_TSV.exists():
        logger.error(
            "Raw scrobble file not found: %s\n"
            "Download the Last.fm Dataset 1K and place the TSV in data/raw/",
            SCROBBLES_TSV,
        )
        sys.exit(1)

    logger.info("Step 1/4 — Loading scrobbles from TSV (~2.5 GB, takes a few minutes)...")
    scrobbles = load_scrobbles(
        SCROBBLES_TSV,
        chunksize=500_000,
        save_parquet=str(baseline_path),
    )
    # Use baseline as scrobbles_updated (no Last.fm API needed)
    scrobbles.to_parquet(updated_path, index=False)
    logger.info(
        "Scrobbles saved: %s rows across %s users.",
        f"{len(scrobbles):,}",
        scrobbles["userid"].nunique(),
    )
    return scrobbles


def step2_load_profiles() -> pd.DataFrame:
    """Load user profile TSV → parquet."""
    profiles_path = PROCESSED / "profiles.parquet"

    if profiles_path.exists():
        logger.info("profiles.parquet already exists — loading from cache.")
        return pd.read_parquet(profiles_path)

    if not PROFILES_TSV.exists():
        logger.error("Raw profiles file not found: %s", PROFILES_TSV)
        sys.exit(1)

    logger.info("Step 2/4 — Loading user profiles...")
    profiles = load_profiles(PROFILES_TSV, save_parquet=str(profiles_path))
    logger.info("Profiles saved: %s users.", len(profiles))
    return profiles


def step3_fetch_artist_genres(scrobbles: pd.DataFrame) -> pd.DataFrame:
    """Fetch Spotify genre tags for all unique artists."""
    genres_path = PROCESSED / "artist_genres.parquet"

    if genres_path.exists():
        df = pd.read_parquet(genres_path)
        logger.info(
            "artist_genres.parquet already exists (%s artists) — skipping fetch.",
            len(df),
        )
        return df

    logger.info("Step 3/4 — Checking Spotify credentials...")
    status = check_credentials()
    print(f"  env vars set:     {status['env_vars_set']}")
    print(f"  API reachable:    {status['api_reachable']}")
    print(f"  audio features:   {status['audio_features_available']} (expected False for new apps)")

    if not status["api_reachable"]:
        logger.warning(
            "Spotify API not reachable. Saving empty artist_genres.parquet.\n"
            "Genre diversity features will be zeroed out but clustering will still run."
        )
        empty = pd.DataFrame(
            columns=["artist_name_normalized", "spotify_artist_id", "genres", "popularity"]
        )
        empty.to_parquet(genres_path, index=False)
        return empty

    sp = build_client()
    unique_artists = scrobbles["artist_name"].dropna().unique().tolist()
    logger.info(
        "Step 3/4 — Fetching genres for %s unique artists (this takes a while — "
        "progress shown below)...",
        f"{len(unique_artists):,}",
    )
    artist_genres = fetch_artist_genres(
        sp,
        unique_artists,
        cache_path=str(genres_path),
        request_delay=0.1,
    )
    matched = artist_genres["spotify_artist_id"].notna().sum()
    logger.info("Artist genres saved: %s / %s matched on Spotify.", matched, len(artist_genres))
    return artist_genres


def step4_stub_audio_features() -> pd.DataFrame:
    """
    Create an empty audio_features parquet with the right columns.
    The Spotify audio-features endpoint is unavailable for apps created
    after Nov 27 2024, so we stub it out rather than hang indefinitely.
    """
    audio_path = PROCESSED / "audio_features.parquet"

    if audio_path.exists():
        df = pd.read_parquet(audio_path)
        logger.info("audio_features.parquet already exists (%s tracks).", len(df))
        return df

    logger.info(
        "Step 4/4 — Spotify audio-features endpoint unavailable for new apps.\n"
        "           Creating empty stub so notebooks 02-04 run without errors."
    )
    stub = pd.DataFrame(
        columns=[
            "track_key", "track_spotify_id",
            "danceability", "energy", "valence", "tempo",
            "acousticness", "instrumentalness", "liveness", "speechiness",
            "loudness", "mode", "time_signature",
        ]
    )
    stub.to_parquet(audio_path, index=False)
    logger.info("Empty audio_features.parquet saved.")
    return stub


def main():
    print("\n" + "=" * 60)
    print("  Spotify Cluster Analysis — Data Preparation")
    print("=" * 60 + "\n")

    PROCESSED.mkdir(parents=True, exist_ok=True)

    scrobbles = step1_load_scrobbles()
    profiles = step2_load_profiles()
    artist_genres = step3_fetch_artist_genres(scrobbles)
    audio_features = step4_stub_audio_features()

    print("\n" + "=" * 60)
    print("  All done. Files in data/processed/:")
    for f in sorted(PROCESSED.glob("*.parquet")):
        size_mb = f.stat().st_size / 1_048_576
        print(f"    {f.name:<45} {size_mb:>6.1f} MB")
    print("\n  You can now open and run notebooks 02, 03, and 04.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
