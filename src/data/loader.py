"""
loader.py
---------
Load and parse the Last.fm Dataset 1K TSV files into pandas DataFrames.
Handles the large scrobbles file efficiently via chunked reading and
saves to CSV for downstream access.

All persistent outputs use CSV format for consistency and human-readability.
The ``genres`` column in artist_genres CSV is stored as a JSON string and
must be parsed back to a list with ``_parse_genres`` (or the helper
``load_artist_genres_csv``) before feature engineering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Iterator

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_SCROBBLE_COLS = [
    "userid",
    "timestamp",
    "artist_mbid",
    "artist_name",
    "track_mbid",
    "track_name",
]

_PROFILE_COLS = [
    "userid",
    "gender",
    "age",
    "country",
    "signup",
]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_scrobbles(
    tsv_path: str | Path,
    chunksize: int = 500_000,
    save_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the large scrobbles TSV (userid-timestamp-artid-artname-traid-traname.tsv).
    Reads in chunks to manage memory for the ~19M-row file.

    Parameters
    ----------
    tsv_path  : path to the raw TSV file
    chunksize : rows per chunk (tune based on available RAM)
    save_csv  : if given, save the result as CSV at this path

    Returns
    -------
    pd.DataFrame with columns: userid, timestamp (datetime64), artist_mbid,
    artist_name, track_mbid, track_name
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Scrobbles TSV not found: {tsv_path}")

    chunks = []
    logger.info("Loading scrobbles from %s (chunked at %d rows)...", tsv_path, chunksize)

    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        names=_SCROBBLE_COLS,
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
        chunksize=chunksize,
    )

    for i, chunk in enumerate(reader):
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"], utc=True, errors="coerce"
        )
        chunk = chunk.dropna(subset=["userid", "timestamp", "artist_name"])
        chunk["userid"] = chunk["userid"].str.strip()
        chunk["artist_name"] = chunk["artist_name"].str.strip()
        chunk["track_name"] = chunk["track_name"].str.strip()
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            logger.info("  Loaded %d chunks (~%d rows so far)...", i + 1, (i + 1) * chunksize)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["userid", "timestamp"]).reset_index(drop=True)
    logger.info("Scrobbles loaded: %d rows, %d users", len(df), df["userid"].nunique())

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        logger.info("Saved scrobbles to %s", save_csv)

    return df


def load_profiles(tsv_path: str | Path, save_csv: str | Path | None = None) -> pd.DataFrame:
    """
    Load the user profile TSV (userid-profile.tsv).

    Returns
    -------
    pd.DataFrame with columns: userid, gender, age, country, signup
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Profiles TSV not found: {tsv_path}")

    df = pd.read_csv(
        tsv_path,
        sep="\t",
        names=_PROFILE_COLS,
        header=None,
        encoding="utf-8",
        on_bad_lines="skip",
    )
    df["userid"] = df["userid"].str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["signup"] = pd.to_datetime(df["signup"], errors="coerce")

    logger.info("Profiles loaded: %d users", len(df))

    if save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        logger.info("Saved profiles to %s", save_csv)

    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a saved CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def load_artist_genres_csv(path: str | Path) -> pd.DataFrame:
    """
    Load artist_genres.csv and deserialise the ``genres`` column from its
    JSON string representation back to a Python list[str].

    This is the correct loader for artist_genres — use it instead of
    ``pd.read_csv`` directly so that ``compute_genre_diversity`` receives
    proper lists.
    """
    df = load_csv(path)
    df["genres"] = df["genres"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    return df


def make_sample(
    scrobbles: pd.DataFrame,
    n_users: int = 50,
    save_path: str | Path | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Randomly sample n_users from the scrobble data for fast development iteration.
    """
    sampled_users = (
        scrobbles["userid"]
        .drop_duplicates()
        .sample(n=min(n_users, scrobbles["userid"].nunique()), random_state=random_state)
    )
    sample = scrobbles[scrobbles["userid"].isin(sampled_users)].copy()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        sample.to_csv(save_path, index=False)
        logger.info("Sample saved: %d rows, %d users → %s", len(sample), n_users, save_path)

    return sample


# ---------------------------------------------------------------------------
# Memory-efficient / chunked CSV helpers
# (useful when scrobbles_updated.csv is still large)
# ---------------------------------------------------------------------------

def get_csv_userids(path: str | Path) -> list[str]:
    """
    Read only the 'userid' column from a CSV file and return all unique
    user IDs.  Uses chunked reading so only the userid column is held in RAM.

    Parameters
    ----------
    path : path to the CSV file (e.g. scrobbles_updated.csv)

    Returns
    -------
    Sorted list of unique user ID strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    user_set: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["userid"], chunksize=500_000):
        user_set.update(chunk["userid"].dropna().astype(str).tolist())

    unique = sorted(user_set)
    logger.info("Found %d unique users in %s", len(unique), path)
    return unique


def load_csv_chunked(
    path: str | Path,
    chunksize: int = 100_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream a CSV file in chunks to avoid loading the whole file into RAM.

    Parameters
    ----------
    path      : path to the CSV file
    chunksize : approximate number of rows per yielded DataFrame
    usecols   : if given, only these columns are read (column pruning)

    Yields
    ------
    pd.DataFrame with up to `chunksize` rows each.

    Example
    -------
    >>> for chunk in load_csv_chunked('data/processed/scrobbles_updated.csv',
    ...                               usecols=['userid', 'timestamp']):
    ...     process(chunk)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        yield chunk


def load_csv_for_users(
    path: str | Path,
    userids: list[str],
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load rows for a specific subset of users from a CSV file.
    Streams the file in chunks to keep peak memory low.

    Parameters
    ----------
    path    : path to the CSV file
    userids : list of user IDs to include
    usecols : optional column subset

    Returns
    -------
    pd.DataFrame containing only the requested users' rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    uid_set = set(userids)
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        filtered = chunk[chunk["userid"].isin(uid_set)]
        if not filtered.empty:
            frames.append(filtered)

    if not frames:
        cols = usecols or []
        return pd.DataFrame(columns=cols)

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d rows for %d/%d requested users from %s",
        len(df), df["userid"].nunique(), len(userids), path
    )
    return df


def stream_scrobbles_by_user_batch(
    path: str | Path,
    all_userids: list[str],
    users_per_batch: int = 100,
    usecols: list[str] | None = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield scrobble DataFrames in batches of ``users_per_batch`` users.

    This is the recommended pattern for feature engineering on large CSV files:
    process one user-batch at a time, concatenate feature results at the end.

    Parameters
    ----------
    path            : path to the CSV file
    all_userids     : full list of user IDs (get via ``get_csv_userids``)
    users_per_batch : how many users to load at once (tune to your RAM)
    usecols         : optional column subset (column pruning)

    Yields
    ------
    pd.DataFrame — scrobbles for ``users_per_batch`` users.

    Example
    -------
    >>> userids = get_csv_userids('data/processed/scrobbles_updated.csv')
    >>> results = []
    >>> for chunk in stream_scrobbles_by_user_batch(
    ...         'data/processed/scrobbles_updated.csv', userids,
    ...         users_per_batch=100,
    ...         usecols=['userid', 'timestamp', 'artist_name', 'track_name']):
    ...     results.append(compute_engagement_features(chunk))
    >>> engagement = pd.concat(results)
    """
    path = Path(path)
    n_batches = (len(all_userids) + users_per_batch - 1) // users_per_batch
    logger.info(
        "Streaming %d users in %d batches of %d from %s",
        len(all_userids), n_batches, users_per_batch, path
    )
    for i in range(0, len(all_userids), users_per_batch):
        batch_users = all_userids[i : i + users_per_batch]
        batch_num = i // users_per_batch + 1
        logger.info(
            "  Batch %d/%d — users %d–%d", batch_num, n_batches, i + 1, i + len(batch_users)
        )
        yield load_csv_for_users(path, batch_users, usecols=usecols)
