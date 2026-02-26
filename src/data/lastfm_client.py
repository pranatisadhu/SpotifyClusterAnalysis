"""
lastfm_client.py
----------------
Last.fm API client using pylast.
Fetches updated scrobble history for users from the Last.fm Dataset 1K.

Handles:
  - Authentication
  - Paginated user.getRecentTracks (last N years)
  - Rate-limit-aware retries via tenacity
  - Incremental saving to avoid data loss on long runs
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import pylast
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

_SCROBBLE_COLS = [
    "userid",
    "timestamp",
    "artist_mbid",
    "artist_name",
    "track_mbid",
    "track_name",
]


def build_network() -> pylast.LastFMNetwork:
    """Build an authenticated pylast LastFMNetwork from environment variables."""
    api_key = os.environ["LASTFM_API_KEY"]
    api_secret = os.environ["LASTFM_API_SECRET"]
    username = os.environ.get("LASTFM_USERNAME", "")
    password_hash = os.environ.get("LASTFM_PASSWORD_HASH", "")

    return pylast.LastFMNetwork(
        api_key=api_key,
        api_secret=api_secret,
        username=username or None,
        password_hash=password_hash or None,
    )


@retry(
    retry=retry_if_exception_type((pylast.NetworkError, pylast.MalformedResponseError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch_recent_tracks_page(
    network: pylast.LastFMNetwork,
    username: str,
    from_ts: int,
    to_ts: int,
    page: int,
    page_size: int,
) -> list[pylast.PlayedTrack]:
    """Fetch one page of recent tracks for a user. Retries on transient errors."""
    user = network.get_user(username)
    return user.get_recent_tracks(
        limit=page_size,
        time_from=from_ts,
        time_to=to_ts,
        page=page,
    )


def fetch_user_scrobbles(
    network: pylast.LastFMNetwork,
    username: str,
    lookback_years: int = 5,
    page_size: int = 200,
    request_delay: float = 0.25,
) -> pd.DataFrame:
    """
    Fetch all scrobbles for *username* over the last `lookback_years` years.

    Returns
    -------
    pd.DataFrame with columns matching _SCROBBLE_COLS, or empty DataFrame
    if the user has no recent activity or does not exist.
    """
    now = datetime.now(tz=timezone.utc)
    from_dt = now - timedelta(days=365 * lookback_years)
    from_ts = int(from_dt.timestamp())
    to_ts = int(now.timestamp())

    records: list[dict] = []
    page = 1

    logger.debug("Fetching scrobbles for %s (pages)...", username)

    while True:
        try:
            tracks = _fetch_recent_tracks_page(
                network, username, from_ts, to_ts, page, page_size
            )
        except pylast.WSError as exc:
            if "User not found" in str(exc) or "Invalid user" in str(exc):
                logger.warning("User %s not found on Last.fm — skipping.", username)
                return pd.DataFrame(columns=_SCROBBLE_COLS)
            raise

        if not tracks:
            break

        for played in tracks:
            track = played.track
            records.append(
                {
                    "userid": username,
                    "timestamp": pd.to_datetime(int(played.timestamp), unit="s", utc=True),
                    "artist_mbid": "",
                    "artist_name": track.artist.name if track.artist else "",
                    "track_mbid": "",
                    "track_name": track.title,
                }
            )

        if len(tracks) < page_size:
            break

        page += 1
        time.sleep(request_delay)

    if not records:
        logger.info("  %s: no scrobbles in lookback window.", username)
        return pd.DataFrame(columns=_SCROBBLE_COLS)

    df = pd.DataFrame(records)
    logger.debug("  %s: fetched %d scrobbles.", username, len(df))
    return df


def fetch_all_users(
    network: pylast.LastFMNetwork,
    userids: list[str],
    lookback_years: int = 5,
    page_size: int = 200,
    request_delay: float = 0.25,
    min_scrobbles: int = 50,
    save_dir: str | Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Fetch updated scrobbles for all users in `userids`.

    Parameters
    ----------
    network : authenticated pylast network
    userids : list of Last.fm usernames (from the original dataset)
    lookback_years : how many years back to pull
    page_size : Last.fm API page size (max 200)
    request_delay : seconds between requests per page
    min_scrobbles : drop users with fewer scrobbles than this in the window
    save_dir : directory for incremental per-user Parquet saves
    resume : if True, skip users whose save file already exists in save_dir

    Returns
    -------
    Combined pd.DataFrame of all users' scrobbles.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for uid in tqdm(userids, desc="Fetching Last.fm scrobbles"):
        user_path = (save_dir / f"{uid}.parquet") if save_dir else None

        if resume and user_path and user_path.exists():
            df_user = pd.read_parquet(user_path)
            logger.debug("Resumed %s from cache (%d rows).", uid, len(df_user))
        else:
            df_user = fetch_user_scrobbles(
                network,
                uid,
                lookback_years=lookback_years,
                page_size=page_size,
                request_delay=request_delay,
            )
            if len(df_user) >= min_scrobbles and user_path:
                df_user.to_parquet(user_path, index=False)

        if len(df_user) >= min_scrobbles:
            all_frames.append(df_user)
        else:
            logger.info("Dropping %s: only %d scrobbles (< %d).", uid, len(df_user), min_scrobbles)

    if not all_frames:
        logger.warning("No users passed the minimum scrobble threshold.")
        return pd.DataFrame(columns=_SCROBBLE_COLS)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["userid", "timestamp"]).reset_index(drop=True)
    logger.info(
        "Fetch complete: %d scrobbles across %d users.",
        len(combined),
        combined["userid"].nunique(),
    )
    return combined


def discover_random_users(
    network: pylast.LastFMNetwork,
    target_n: int = 1000,
    n_tags: int = 20,
    n_artists_per_tag: int = 10,
    n_fans_per_artist: int = 50,
    seed: int = 42,
    request_delay: float = 0.25,
) -> list[str]:
    """
    Discover a pool of Last.fm usernames by crawling tags → top artists → top listeners.

    Strategy
    --------
    1. Fetch the global top tags from Last.fm.
    2. For each of the top `n_tags` tags, fetch `n_artists_per_tag` top artists.
    3. For each artist, fetch up to `n_fans_per_artist` top listeners (fans).
    4. Deduplicate and sample `target_n` usernames using `seed` for reproducibility.

    Returns
    -------
    List of Last.fm usernames (strings).
    """
    import random

    rng = random.Random(seed)
    user_pool: set[str] = set()

    try:
        top_tags = network.get_top_tags(limit=n_tags)
    except Exception as exc:
        logger.error("Failed to fetch top tags: %s", exc)
        return []

    for tag_item in top_tags:
        tag_name = tag_item.item.get_name() if hasattr(tag_item.item, "get_name") else str(tag_item.item)
        try:
            top_artists = network.get_tag(tag_name).get_top_artists(limit=n_artists_per_tag)
        except Exception as exc:
            logger.warning("Could not fetch artists for tag '%s': %s", tag_name, exc)
            continue

        for artist_item in top_artists:
            artist_name = artist_item.item.get_name() if hasattr(artist_item.item, "get_name") else str(artist_item.item)
            try:
                fans = network.get_artist(artist_name).get_top_listeners(limit=n_fans_per_artist)
                for fan in fans:
                    username = fan.item.get_name() if hasattr(fan.item, "get_name") else str(fan.item)
                    if username:
                        user_pool.add(username)
            except Exception as exc:
                logger.warning("Could not fetch fans for artist '%s': %s", artist_name, exc)
                continue

            time.sleep(request_delay)

    logger.info("User pool: %d unique usernames discovered.", len(user_pool))

    pool_list = sorted(user_pool)
    if len(pool_list) < target_n:
        logger.warning(
            "Pool size (%d) is smaller than target (%d). Using all discovered users.",
            len(pool_list),
            target_n,
        )
        sampled = pool_list
    else:
        sampled = rng.sample(pool_list, target_n)

    logger.info("Sampled %d users for enrichment.", len(sampled))
    return sampled


def get_user_top_artists(
    network: pylast.LastFMNetwork,
    username: str,
    period: str = pylast.PERIOD_OVERALL,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch top artists for a user (used as a cross-check / supplementary signal).

    Parameters
    ----------
    period : one of pylast.PERIOD_* constants
             (PERIOD_7DAYS, PERIOD_1MONTH, PERIOD_3MONTHS,
              PERIOD_6MONTHS, PERIOD_12MONTHS, PERIOD_OVERALL)
    """
    try:
        user = network.get_user(username)
        top = user.get_top_artists(period=period, limit=limit)
        return [
            {"userid": username, "artist_name": ta.item.name, "playcount": int(ta.weight)}
            for ta in top
        ]
    except pylast.WSError as exc:
        logger.warning("Could not fetch top artists for %s: %s", username, exc)
        return []
