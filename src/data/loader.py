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
    # Drop any row where the header was read as data (userid == "userid")
    df = df[df["userid"] != "userid"].copy()
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
