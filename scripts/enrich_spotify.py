"""
enrich_spotify.py
-----------------
Standalone script to enrich existing scrobble data with Spotify genre and
audio-feature information.

Run from the project root:
    python scripts/enrich_spotify.py

Reads  : data/processed/scrobbles_updated.csv  (or .parquet if it exists)
Writes : data/processed/artist_genres.parquet
         data/processed/audio_features.parquet

Credentials are loaded from .env in the project root:
    SPOTIFY_CLIENT_ID=...
    SPOTIFY_CLIENT_SECRET=...

The script uses persistent caches so it is safe to interrupt and re-run —
it will only fetch what is not yet cached.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import pandas as pd

from src.data.spotify_client import build_client, fetch_artist_genres, fetch_audio_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED = PROJECT_ROOT / "data" / "processed"

SCROBBLES_PARQUET = PROCESSED / "scrobbles_updated.parquet"
SCROBBLES_CSV = PROCESSED / "scrobbles_updated.csv"

ARTIST_GENRES_OUT = PROCESSED / "artist_genres.parquet"
AUDIO_FEATURES_OUT = PROCESSED / "audio_features.parquet"

ARTIST_CACHE = PROCESSED / "spotify_artist_cache.parquet"
AUDIO_CACHE = PROCESSED / "spotify_audio_features_cache.parquet"

TOP_TRACKS_PER_USER = 20


# ---------------------------------------------------------------------------
# Load scrobbles
# ---------------------------------------------------------------------------
def load_scrobbles() -> pd.DataFrame:
    if SCROBBLES_PARQUET.exists():
        logger.info("Loading scrobbles from %s", SCROBBLES_PARQUET)
        return pd.read_parquet(SCROBBLES_PARQUET)
    if SCROBBLES_CSV.exists():
        logger.info("Loading scrobbles from %s (CSV — this may take a minute)…", SCROBBLES_CSV)
        df = pd.read_csv(SCROBBLES_CSV, low_memory=False)
        # Normalise column names in case the CSV uses different casing
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    raise FileNotFoundError(
        "Neither scrobbles_updated.parquet nor scrobbles_updated.csv found in "
        f"{PROCESSED}. Please check your data/processed/ directory."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== Spotify Enrichment Script ===")
    logger.info("Project root: %s", PROJECT_ROOT)

    # 1. Load scrobbles
    scrobbles = load_scrobbles()
    logger.info(
        "Scrobbles loaded: %d rows, %d users",
        len(scrobbles),
        scrobbles["userid"].nunique(),
    )

    required_cols = {"userid", "artist_name", "track_name"}
    missing = required_cols - set(scrobbles.columns)
    if missing:
        raise ValueError(
            f"Scrobbles data is missing expected columns: {missing}. "
            f"Found columns: {list(scrobbles.columns)}"
        )

    # 2. Build Spotify client
    logger.info("Initialising Spotify client…")
    sp = build_client()

    # ---------------------------------------------------------------------------
    # Section 4 — Artist genres
    # ---------------------------------------------------------------------------
    if ARTIST_GENRES_OUT.exists():
        logger.info(
            "artist_genres.parquet already exists — skipping genre fetch. "
            "Delete %s to re-fetch.", ARTIST_GENRES_OUT
        )
    else:
        unique_artists = scrobbles["artist_name"].dropna().unique().tolist()
        logger.info("Unique artists to look up: %d", len(unique_artists))

        artist_genres = fetch_artist_genres(
            sp,
            unique_artists,
            cache_path=ARTIST_CACHE,
            request_delay=0.1,
        )
        artist_genres.to_parquet(ARTIST_GENRES_OUT, index=False)
        found = artist_genres["spotify_artist_id"].notna().sum()
        logger.info(
            "Artist genres saved → %s  (%d / %d matched on Spotify)",
            ARTIST_GENRES_OUT, found, len(artist_genres),
        )

    # ---------------------------------------------------------------------------
    # Section 5 — Audio features
    # ---------------------------------------------------------------------------
    if AUDIO_FEATURES_OUT.exists():
        logger.info(
            "audio_features.parquet already exists — skipping audio feature fetch. "
            "Delete %s to re-fetch.", AUDIO_FEATURES_OUT
        )
    else:
        track_sample = (
            scrobbles.groupby(["userid", "artist_name", "track_name"])
            .size()
            .reset_index(name="play_count")
            .sort_values(["userid", "play_count"], ascending=[True, False])
            .groupby("userid")
            .head(TOP_TRACKS_PER_USER)
        )
        track_pairs = list(
            set(zip(track_sample["artist_name"], track_sample["track_name"]))
        )
        logger.info(
            "Unique (artist, track) pairs to enrich: %d  (top %d tracks per user)",
            len(track_pairs), TOP_TRACKS_PER_USER,
        )

        audio_features = fetch_audio_features(
            sp,
            track_pairs,
            cache_path=AUDIO_CACHE,
            request_delay=0.1,
        )
        audio_features.to_parquet(AUDIO_FEATURES_OUT, index=False)
        logger.info(
            "Audio features saved → %s  (%d tracks)",
            AUDIO_FEATURES_OUT, len(audio_features),
        )

    logger.info("=== Done. All Spotify enrichment files are in %s ===", PROCESSED)


if __name__ == "__main__":
    main()
