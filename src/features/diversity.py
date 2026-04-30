"""
diversity.py
------------
Computes genre and artist diversity features per user.

Features produced:
  - unique_artists          : number of distinct artists
  - unique_genres           : number of distinct Spotify genre tags
  - artist_entropy          : Shannon entropy of artist play distribution
  - genre_entropy           : Shannon entropy of genre play distribution
  - artist_concentration_20 : % of plays attributed to the top 20 artists
  - genre_concentration_5   : % of plays attributed to top 5 genres
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


def _shannon_entropy(counts: pd.Series) -> float:
    """Normalised Shannon entropy (bits). Returns 0 for a single-element series."""
    probs = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def compute_artist_diversity(scrobbles: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Compute artist-level diversity features per user.

    Parameters
    ----------
    scrobbles : DataFrame with columns [userid, artist_name, ...]
    top_n : number of top artists for concentration metric

    Returns
    -------
    pd.DataFrame indexed by userid with artist diversity columns
    """
    play_counts = (
        scrobbles.groupby(["userid", "artist_name"])
        .size()
        .reset_index(name="plays")
    )

    # Unique artist count
    unique_artists = (
        play_counts.groupby("userid")["artist_name"].nunique().rename("unique_artists")
    )

    # Shannon entropy of artist play distribution
    artist_entropy = (
        play_counts.groupby("userid")["plays"]
        .apply(_shannon_entropy)
        .rename("artist_entropy")
    )

    # Concentration: % plays from top N artists
    def _concentration(group: pd.DataFrame) -> float:
        total = group["plays"].sum()
        top_plays = group.nlargest(top_n, "plays")["plays"].sum()
        return top_plays / total if total > 0 else 0.0

    artist_concentration = (
        play_counts.groupby("userid")
        .apply(_concentration)
        .rename(f"artist_concentration_{top_n}")
    )

    result = pd.concat(
        [unique_artists, artist_entropy, artist_concentration], axis=1
    ).fillna(0)
    return result


def compute_genre_diversity(
    scrobbles: pd.DataFrame,
    artist_genres: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Compute genre-level diversity features per user by joining artist genre tags.

    Parameters
    ----------
    scrobbles : DataFrame with columns [userid, artist_name, ...]
    artist_genres : DataFrame with columns [artist_name_normalized, genres (list)]
    top_n : top genres for concentration metric

    Returns
    -------
    pd.DataFrame indexed by userid with genre diversity columns
    """
    # Normalise artist name for join
    scrobbles = scrobbles.copy()
    scrobbles["artist_name_normalized"] = scrobbles["artist_name"].str.strip().str.lower()

    # Explode genres: one row per (userid, play, genre)
    merged = scrobbles.merge(
        artist_genres[["artist_name_normalized", "genres"]],
        on="artist_name_normalized",
        how="left",
    )
    merged["genres"] = merged["genres"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Explode to (userid, genre) level
    merged_exploded = merged.explode("genres").dropna(subset=["genres"])
    merged_exploded = merged_exploded[merged_exploded["genres"] != ""]

    if merged_exploded.empty:
        return pd.DataFrame(
            {
                "unique_genres": 0,
                "genre_entropy": 0.0,
                f"genre_concentration_{top_n}": 0.0,
            }
        )

    genre_counts = (
        merged_exploded.groupby(["userid", "genres"])
        .size()
        .reset_index(name="plays")
    )

    unique_genres = (
        genre_counts.groupby("userid")["genres"].nunique().rename("unique_genres")
    )

    genre_entropy = (
        genre_counts.groupby("userid")["plays"]
        .apply(_shannon_entropy)
        .rename("genre_entropy")
    )

    def _conc(group: pd.DataFrame) -> float:
        total = group["plays"].sum()
        top_plays = group.nlargest(top_n, "plays")["plays"].sum()
        return top_plays / total if total > 0 else 0.0

    genre_concentration = (
        genre_counts.groupby("userid").apply(_conc).rename(f"genre_concentration_{top_n}")
    )

    result = pd.concat(
        [unique_genres, genre_entropy, genre_concentration], axis=1
    ).fillna(0)
    return result
