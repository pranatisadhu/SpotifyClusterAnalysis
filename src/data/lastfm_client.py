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
  - Random user discovery via tag → artist → fan graph
"""

from __future__ import annotations

import logging
import os
import random
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


def discover_random_users(
    network: pylast.LastFMNetwork,
    target_n: int = 10_000,
    n_tags: int = 50,
    n_artists_per_tag: int = 5,
    n_fans_per_artist: int = 100,
    seed: int = 42,
    request_delay: float = 0.5,
) -> list[str]:
    """
    Discover a diverse random sample of Last.fm usernames by walking the
    tag → artist → fan graph.

    Steps:
      1. Fetch the top ``n_tags`` global tags (genres / moods).
      2. For each tag fetch the top ``n_artists_per_tag`` artists.
      3. For each artist fetch up to ``n_fans_per_artist`` top fans (users).
      4. Deduplicate the union, then randomly sample ``target_n`` usernames.

    The three-hop walk ensures diversity: users are spread across genres
    and listening styles rather than being clustered around one community.

    Parameters
    ----------
    network           : authenticated pylast LastFMNetwork
    target_n          : desired number of unique users to return
    n_tags            : number of global top tags to seed the walk from
    n_artists_per_tag : top artists to pull per tag
    n_fans_per_artist : top fans to pull per artist
    seed              : random seed for reproducible sampling
    request_delay     : seconds between API calls (respect rate limits)

    Returns
    -------
    List of up to ``target_n`` unique Last.fm usernames, randomly sampled
    from the discovered pool.
    """
    rng = random.Random(seed)
    user_pool: set[str] = set()

    # ── Step 1: Fetch top global tags ────────────────────────────────────────
    logger.info("Fetching top %d Last.fm tags for user discovery...", n_tags)
    try:
        top_tag_items = network.get_top_tags(limit=n_tags)
        tags = [t.item for t in top_tag_items[:n_tags]]
    except Exception as exc:
        logger.error("Failed to fetch top tags: %s", exc)
        return []

    logger.info("Got %d tags. Walking tag → artist → fan graph...", len(tags))

    # ── Steps 2 & 3: Walk to fans ────────────────────────────────────────────
    for tag in tqdm(tags, desc="Discovering users (tag→artist→fan)"):
        if len(user_pool) >= target_n * 3:
            logger.info("Pool (%d users) is 3× target — stopping early.", len(user_pool))
            break

        try:
            top_artists = tag.get_top_artists(limit=n_artists_per_tag)
            time.sleep(request_delay)
        except Exception as exc:
            logger.debug("Could not get artists for tag '%s': %s", tag, exc)
            continue

        for artist_item in top_artists[:n_artists_per_tag]:
            try:
                fans = artist_item.item.get_top_fans(limit=n_fans_per_artist)
                time.sleep(request_delay)
                for fan in fans:
                    # pylast User objects expose .get_name() or .name
                    username = (
                        fan.item.get_name()
                        if hasattr(fan.item, "get_name")
                        else str(fan.item)
                    )
                    if username:
                        user_pool.add(username)
            except Exception as exc:
                logger.debug(
                    "Could not get fans for artist '%s': %s", artist_item.item, exc
                )

    # ── Step 4: Sample ───────────────────────────────────────────────────────
    user_pool_list = sorted(user_pool)
    logger.info("User pool: %d unique usernames discovered.", len(user_pool_list))

    if len(user_pool_list) >= target_n:
        sampled = rng.sample(user_pool_list, target_n)
    else:
        logger.warning(
            "Pool size (%d) is smaller than target (%d). Using all discovered users.",
            len(user_pool_list),
            target_n,
        )
        sampled = user_pool_list
        rng.shuffle(sampled)

    logger.info("Sampled %d users for enrichment.", len(sampled))
    return sampled


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
    userids : list of Last.fm usernames
    lookback_years : how many years back to pull
    page_size : Last.fm API page size (max 200)
    request_delay : seconds between requests per page
    min_scrobbles : drop users with fewer scrobbles than this in the window
    save_dir : directory for incremental per-user CSV saves (allows resuming)
    resume : if True, skip users whose save file already exists in save_dir

    Returns
    -------
    Combined pd.DataFrame of all users' scrobbles.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Track qualifying CSV paths separately from in-memory frames.
    # Files are written only when a user meets min_scrobbles, so existence
    # implies the threshold was already passed — no need to re-read during
    # the fetch loop.  This avoids the 2× memory spike that occurs when all
    # user DataFrames are held in a list before pd.concat.
    qualifying_paths: list[Path] = []
    fallback_frames: list[pd.DataFrame] = []   # used only when save_dir is None

    for uid in tqdm(userids, desc="Fetching Last.fm scrobbles"):
        user_path = (save_dir / f"{uid}.csv") if save_dir else None

        if resume and user_path and user_path.exists():
            # File exists ↔ user already passed min_scrobbles threshold
            qualifying_paths.append(user_path)
            logger.debug("Queued %s from cache.", uid)
            continue

        df_user = fetch_user_scrobbles(
            network,
            uid,
            lookback_years=lookback_years,
            page_size=page_size,
            request_delay=request_delay,
        )

        if len(df_user) >= min_scrobbles:
            if user_path:
                df_user.to_csv(user_path, index=False)
                qualifying_paths.append(user_path)
            else:
                fallback_frames.append(df_user)
        else:
            logger.info("Dropping %s: only %d scrobbles (< %d).", uid, len(df_user), min_scrobbles)

    if not qualifying_paths and not fallback_frames:
        logger.warning("No users passed the minimum scrobble threshold.")
        return pd.DataFrame(columns=_SCROBBLE_COLS)

    # Read qualifying users from disk one at a time into the concat call —
    # peak memory is now just the final combined DataFrame, not 2× that.
    frames: list[pd.DataFrame] = [
        pd.read_csv(p, parse_dates=["timestamp"])
        for p in tqdm(qualifying_paths, desc="Reading user CSVs")
    ]
    frames.extend(fallback_frames)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["userid", "timestamp"]).reset_index(drop=True)
    logger.info(
        "Fetch complete: %d scrobbles across %d users.",
        len(combined),
        combined["userid"].nunique(),
    )
    return combined


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
