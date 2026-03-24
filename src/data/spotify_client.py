"""
spotify_client.py
-----------------
Spotify API client using spotipy (Client Credentials flow — no user login needed).

Provides:
  - Artist genre lookup (by artist name search)
  - Audio feature enrichment for tracks (danceability, energy, valence, etc.)
  - Batched calls to stay within Spotify rate limits
  - Persistent cache (Parquet) to avoid redundant API calls across runs
"""

from __future__ import annotations

import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

# Spotify allows up to 50 IDs per batch request for audio features
_SPOTIFY_BATCH_SIZE = 50

_AUDIO_FEATURE_COLS = [
    "track_spotify_id",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "loudness",
    "mode",
    "time_signature",
]

_ARTIST_GENRE_COLS = [
    "artist_name_normalized",
    "spotify_artist_id",
    "genres",
    "popularity",
]


def build_client() -> spotipy.Spotify:
    """Build a Spotify client using Client Credentials (no user login required)."""
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=os.environ["SPOTIFY_CLIENT_ID"],
            client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        ),
        requests_timeout=10,
    )


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _search_artist(sp: spotipy.Spotify, artist_name: str) -> dict | None:
    """Search Spotify for an artist by name; return best match or None."""
    results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    items = results.get("artists", {}).get("items", [])
    return items[0] if items else None


def fetch_artist_genres(
    sp: spotipy.Spotify,
    artist_names: list[str],
    cache_path: str | Path | None = None,
    request_delay: float = 0.1,
    max_workers: int = 10,
    cache_save_interval: int = 500,
) -> pd.DataFrame:
    """
    Fetch Spotify genre tags and popularity for each artist in `artist_names`.
    Uses a persistent Parquet cache to avoid repeated lookups.
    Fetches in parallel using a thread pool; saves cache periodically so
    progress is not lost on interruption.

    Returns
    -------
    pd.DataFrame with columns: artist_name_normalized, spotify_artist_id,
    genres (list[str]), popularity
    """
    def _norm(name: str) -> str:
        return name.strip().lower()

    cache: dict[str, dict] = {}
    cache_lock = threading.Lock()

    if cache_path and Path(cache_path).exists():
        cached_df = pd.read_parquet(cache_path)
        for _, row in cached_df.iterrows():
            cache[row["artist_name_normalized"]] = row.to_dict()
        logger.info("Loaded %d cached artist lookups from %s.", len(cache), cache_path)

    names_to_fetch = [n for n in artist_names if _norm(n) not in cache]
    logger.info(
        "Artist genre lookup: %d new artists to fetch (cache has %d).",
        len(names_to_fetch),
        len(cache),
    )

    completed_count = 0

    def _fetch_one(name: str) -> None:
        nonlocal completed_count
        # Each thread gets its own client to avoid shared session state
        thread_sp = build_client()
        norm_name = _norm(name)
        try:
            artist = _search_artist(thread_sp, name)
            record = {
                "artist_name_normalized": norm_name,
                "spotify_artist_id": artist["id"] if artist else None,
                "genres": artist.get("genres", []) if artist else [],
                "popularity": artist.get("popularity", None) if artist else None,
            }
        except Exception as exc:
            logger.warning("Failed to fetch artist '%s': %s", name, exc)
            record = {
                "artist_name_normalized": norm_name,
                "spotify_artist_id": None,
                "genres": [],
                "popularity": None,
            }

        with cache_lock:
            cache[norm_name] = record
            completed_count += 1
            # Periodically save cache so progress survives interruption
            if cache_path and completed_count % cache_save_interval == 0:
                _save_cache(cache, cache_path)
                logger.info(
                    "Progress saved: %d / %d fetched.", completed_count, len(names_to_fetch)
                )

    def _save_cache(c: dict, path: str | Path) -> None:
        df_tmp = pd.DataFrame(list(c.values()), columns=_ARTIST_GENRE_COLS)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df_tmp.to_parquet(path, index=False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_one, name): name for name in names_to_fetch}
        with tqdm(total=len(names_to_fetch), desc="Fetching artist genres") as pbar:
            for future in as_completed(futures):
                future.result()  # re-raise any uncaught exception
                pbar.update(1)

    all_records = list(cache.values())
    df = pd.DataFrame(all_records, columns=_ARTIST_GENRE_COLS)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Artist genre cache saved to %s (%d entries).", cache_path, len(df))

    return df


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _search_track(
    sp: spotipy.Spotify, artist_name: str, track_name: str
) -> str | None:
    """Return Spotify track ID for an (artist, track) pair or None."""
    query = f"artist:{artist_name} track:{track_name}"
    results = sp.search(q=query, type="track", limit=1)
    items = results.get("tracks", {}).get("items", [])
    return items[0]["id"] if items else None


def fetch_audio_features(
    sp: spotipy.Spotify,
    track_pairs: list[tuple[str, str]],
    cache_path: str | Path | None = None,
    request_delay: float = 0.1,
    max_workers: int = 10,
) -> pd.DataFrame:
    """
    Fetch Spotify audio features for a list of (artist_name, track_name) pairs.
    Resolves track IDs in parallel, then batch-fetches audio features (50/batch).

    Returns
    -------
    pd.DataFrame with columns: track_key (artist|track), danceability, energy,
    valence, tempo, acousticness, instrumentalness, liveness, speechiness,
    loudness, mode, time_signature
    """
    def _key(artist: str, track: str) -> str:
        return f"{artist.strip().lower()}|||{track.strip().lower()}"

    cache: dict[str, dict] = {}
    if cache_path and Path(cache_path).exists():
        cached_df = pd.read_parquet(cache_path)
        for _, row in cached_df.iterrows():
            cache[row["track_key"]] = row.to_dict()
        logger.info("Loaded %d cached track audio features.", len(cache))

    # Identify which pairs still need fetching
    pairs_to_fetch = [(a, t) for a, t in track_pairs if _key(a, t) not in cache]
    logger.info(
        "Audio features: %d new tracks to fetch (cache has %d).",
        len(pairs_to_fetch),
        len(cache),
    )

    # Step 1: Resolve track IDs in parallel
    track_id_map: dict[str, str] = {}
    id_map_lock = threading.Lock()

    def _resolve_one(pair: tuple[str, str]) -> None:
        artist, track = pair
        k = _key(artist, track)
        thread_sp = build_client()
        try:
            tid = _search_track(thread_sp, artist, track)
            if tid:
                with id_map_lock:
                    track_id_map[k] = tid
        except Exception as exc:
            logger.warning("Could not resolve track '%s - %s': %s", artist, track, exc)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_resolve_one, pair) for pair in pairs_to_fetch]
        with tqdm(total=len(pairs_to_fetch), desc="Resolving track IDs") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    # Step 2: Batch-fetch audio features
    keys = list(track_id_map.keys())
    ids = list(track_id_map.values())

    for batch_start in tqdm(
        range(0, len(ids), _SPOTIFY_BATCH_SIZE), desc="Fetching audio features"
    ):
        batch_keys = keys[batch_start : batch_start + _SPOTIFY_BATCH_SIZE]
        batch_ids = ids[batch_start : batch_start + _SPOTIFY_BATCH_SIZE]
        try:
            features_list = sp.audio_features(batch_ids)
        except Exception as exc:
            logger.warning("Batch audio features failed: %s", exc)
            features_list = [None] * len(batch_ids)

        for k, features in zip(batch_keys, features_list):
            if features:
                cache[k] = {
                    "track_key": k,
                    "track_spotify_id": features.get("id"),
                    "danceability": features.get("danceability"),
                    "energy": features.get("energy"),
                    "valence": features.get("valence"),
                    "tempo": features.get("tempo"),
                    "acousticness": features.get("acousticness"),
                    "instrumentalness": features.get("instrumentalness"),
                    "liveness": features.get("liveness"),
                    "speechiness": features.get("speechiness"),
                    "loudness": features.get("loudness"),
                    "mode": features.get("mode"),
                    "time_signature": features.get("time_signature"),
                }
            else:
                cache[k] = {"track_key": k, "track_spotify_id": None}

        time.sleep(request_delay)

    df = pd.DataFrame(list(cache.values()))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Audio features cache saved to %s.", cache_path)

    return df
