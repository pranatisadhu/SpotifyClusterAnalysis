"""
loader.py
---------
Load and parse the Last.fm Dataset 1K TSV files into pandas DataFrames.
Handles the large scrobbles file efficiently via chunked reading and
saves to Parquet for fast downstream access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Iterator

import pandas as pd
import pyarrow.parquet as pq
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
    save_parquet: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load the large scrobbles TSV (userid-timestamp-artid-artname-traid-traname.tsv).
    Reads in chunks to manage memory for the ~19M-row file.

    Parameters
    ----------
    tsv_path : path to the raw TSV file
    chunksize : rows per chunk (tune based on available RAM)
    save_parquet : if given, save the result as Parquet at this path

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

    if save_parquet:
        Path(save_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_parquet, index=False)
        logger.info("Saved scrobbles to %s", save_parquet)

    return df


def load_profiles(tsv_path: str | Path, save_parquet: str | Path | None = None) -> pd.DataFrame:
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

    if save_parquet:
        Path(save_parquet).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_parquet, index=False)
        logger.info("Saved profiles to %s", save_parquet)

    return df


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a saved Parquet file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


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
        sample.to_parquet(save_path, index=False)
        logger.info("Sample saved: %d rows, %d users → %s", len(sample), n_users, save_path)

    return sample


# ---------------------------------------------------------------------------
# Memory-efficient / chunked Parquet helpers
# ---------------------------------------------------------------------------

def get_parquet_userids(path: str | Path) -> list[str]:
    """
    Read only the 'userid' column from a Parquet file and return all unique
    user IDs.  Much cheaper than loading the entire file.

    Parameters
    ----------
    path : path to the Parquet file (e.g. scrobbles_updated.parquet)

    Returns
    -------
    Sorted list of unique user ID strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    table = pq.read_table(path, columns=["userid"])
    userids = table.column("userid").to_pylist()
    unique = sorted(set(u for u in userids if u is not None))
    logger.info("Found %d unique users in %s", len(unique), path)
    return unique


def load_parquet_chunked(
    path: str | Path,
    batch_size: int = 100_000,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream a Parquet file in row-group batches to avoid loading the whole
    file into RAM at once.

    Parameters
    ----------
    path       : path to the Parquet file
    batch_size : approximate number of rows per yielded DataFrame
    columns    : if given, only these columns are read (column pruning)

    Yields
    ------
    pd.DataFrame with up to `batch_size` rows each.

    Example
    -------
    >>> feature_chunks = []
    >>> for chunk in load_parquet_chunked('data/processed/scrobbles_updated.parquet',
    ...                                   columns=['userid', 'timestamp']):
    ...     feature_chunks.append(compute_temporal_features(chunk))
    >>> temporal = pd.concat(feature_chunks)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def load_parquet_for_users(
    path: str | Path,
    userids: list[str],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load scrobbles for a specific subset of users using Parquet predicate
    pushdown — far cheaper than reading the whole file when you only need
    a few users.

    Parameters
    ----------
    path    : path to the Parquet file
    userids : list of user IDs to include
    columns : if given, only these columns are loaded (column pruning)

    Returns
    -------
    pd.DataFrame containing only the requested users' rows.

    Example
    -------
    >>> chunk = load_parquet_for_users(
    ...     'data/processed/scrobbles_updated.parquet',
    ...     userids=['user_001', 'user_002'],
    ...     columns=['userid', 'timestamp', 'artist_name', 'track_name'],
    ... )
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    filters = [("userid", "in", set(userids))]
    table = pq.read_table(path, columns=columns, filters=filters)
    df = table.to_pandas()
    logger.info(
        "Loaded %d rows for %d/%d requested users from %s",
        len(df), df["userid"].nunique() if len(df) else 0, len(userids), path
    )
    return df


def stream_scrobbles_by_user_batch(
    path: str | Path,
    all_userids: list[str],
    users_per_batch: int = 100,
    columns: list[str] | None = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield scrobble DataFrames in batches of `users_per_batch` users.

    This is the recommended pattern for feature engineering on large files:
    process one user-batch at a time, concatenate results at the end.

    Parameters
    ----------
    path            : path to the Parquet file
    all_userids     : full list of user IDs (get via ``get_parquet_userids``)
    users_per_batch : how many users to load at once (tune to your RAM)
    columns         : optional column subset (column pruning)

    Yields
    ------
    pd.DataFrame — scrobbles for `users_per_batch` users.

    Example
    -------
    >>> userids = get_parquet_userids('data/processed/scrobbles_updated.parquet')
    >>> results = []
    >>> for chunk in stream_scrobbles_by_user_batch(
    ...         'data/processed/scrobbles_updated.parquet', userids,
    ...         users_per_batch=100,
    ...         columns=['userid', 'timestamp', 'artist_name', 'track_name']):
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
        logger.info("  Batch %d/%d — users %d–%d", batch_num, n_batches, i + 1, i + len(batch_users))
        yield load_parquet_for_users(path, batch_users, columns=columns)
