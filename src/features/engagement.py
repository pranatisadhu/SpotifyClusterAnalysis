"""
engagement.py
-------------
Computes engagement and discovery features per user.

Features produced:
  - total_scrobbles          : total play events
  - unique_tracks            : distinct (artist, track) pairs
  - track_replay_rate        : total plays / unique tracks (>1 means replaying)
  - avg_tracks_per_session   : mean tracks listened per session
  - session_count            : total number of detected sessions
  - avg_session_length_min   : mean session duration in minutes
  - discovery_velocity_30d   : new artists per day in most recent 30 days
  - discovery_velocity_90d   : new artists per day in most recent 90 days
  - novelty_ratio            : % of recent plays (last 90d) from artists first
                               encountered in the last 90 days
  - top_artist_play_share    : fraction of plays by the single most-played artist

Session definition: consecutive plays with gap < session_gap_minutes constitute
one session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm


def _detect_sessions(
    user_df: pd.DataFrame, session_gap_minutes: int = 30
) -> pd.Series:
    """
    Assign a session ID to each row.  A new session starts whenever the
    gap between consecutive plays exceeds `session_gap_minutes`.
    Caller must ensure user_df is already sorted by timestamp.
    """
    deltas = user_df["timestamp"].diff()
    gap = pd.Timedelta(minutes=session_gap_minutes)
    session_starts = (deltas > gap) | deltas.isna()
    return session_starts.cumsum()


def compute_engagement_features(
    scrobbles: pd.DataFrame,
    session_gap_minutes: int = 30,
) -> pd.DataFrame:
    """
    Compute engagement metrics for each user.

    Parameters
    ----------
    scrobbles : DataFrame with [userid, timestamp (datetime64 UTC),
                artist_name, track_name, ...]
    session_gap_minutes : inactivity gap that defines a new session

    Returns
    -------
    pd.DataFrame indexed by userid
    """
    records: list[dict] = []

    # Sort once upfront so each groupby group arrives pre-sorted by timestamp.
    # groupby then yields each user's slice directly without a repeated full-table
    # boolean scan — ~100× faster than filtering inside a plain for-loop.
    scrobbles_sorted = scrobbles.sort_values(["userid", "timestamp"])
    n_users = scrobbles_sorted["userid"].nunique()

    for uid, df in tqdm(
        scrobbles_sorted.groupby("userid", sort=False),
        desc="Computing engagement features",
        total=n_users,
    ):
        df = df.reset_index(drop=True)

        total_scrobbles = len(df)
        unique_tracks = df.groupby(["artist_name", "track_name"]).ngroups

        track_replay_rate = total_scrobbles / unique_tracks if unique_tracks > 0 else 0.0

        top_artist_plays = df["artist_name"].value_counts().iloc[0] if total_scrobbles > 0 else 0
        top_artist_share = top_artist_plays / total_scrobbles if total_scrobbles > 0 else 0.0

        # Sessions
        df["session_id"] = _detect_sessions(df, session_gap_minutes).values
        session_stats = df.groupby("session_id").agg(
            track_count=("track_name", "count"),
            duration=(
                "timestamp",
                lambda ts: (ts.max() - ts.min()).total_seconds() / 60,
            ),
        )
        session_count = len(session_stats)
        avg_tracks_per_session = session_stats["track_count"].mean() if session_count > 0 else 0.0
        avg_session_length_min = session_stats["duration"].mean() if session_count > 0 else 0.0

        # Discovery velocity: new artists introduced per day in recent windows
        now = df["timestamp"].max()

        def _discovery_velocity(df: pd.DataFrame, days: int) -> float:
            cutoff = now - timedelta(days=days)
            past = df[df["timestamp"] < cutoff]["artist_name"].unique()
            recent = df[df["timestamp"] >= cutoff]
            if recent.empty:
                return 0.0
            new_in_window = recent[~recent["artist_name"].isin(past)]["artist_name"].nunique()
            return new_in_window / days

        vel_30 = _discovery_velocity(df, 30)
        vel_90 = _discovery_velocity(df, 90)

        # Novelty ratio: % plays in last 90d that are from artists first encountered in last 90d
        cutoff_90 = now - timedelta(days=90)
        prior_artists = df[df["timestamp"] < cutoff_90]["artist_name"].unique()
        recent_90 = df[df["timestamp"] >= cutoff_90]
        if not recent_90.empty:
            novel_plays = recent_90[~recent_90["artist_name"].isin(prior_artists)]
            novelty_ratio = len(novel_plays) / len(recent_90)
        else:
            novelty_ratio = 0.0

        records.append(
            {
                "userid": uid,
                "total_scrobbles": total_scrobbles,
                "unique_tracks": unique_tracks,
                "track_replay_rate": track_replay_rate,
                "avg_tracks_per_session": avg_tracks_per_session,
                "session_count": session_count,
                "avg_session_length_min": avg_session_length_min,
                "discovery_velocity_30d": vel_30,
                "discovery_velocity_90d": vel_90,
                "novelty_ratio": novelty_ratio,
                "top_artist_play_share": top_artist_share,
            }
        )

    return pd.DataFrame(records).set_index("userid")


def compute_audio_feature_profile(
    scrobbles: pd.DataFrame,
    audio_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Spotify audio features to user level (weighted mean by play count).

    Parameters
    ----------
    scrobbles : scrobble DataFrame with [userid, artist_name, track_name]
    audio_features : DataFrame with [track_key, danceability, energy, valence, ...]
                     where track_key = "artist|||track" (lowercase)

    Returns
    -------
    pd.DataFrame indexed by userid with mean audio feature columns
    """
    audio_cols = [
        "danceability", "energy", "valence", "tempo",
        "acousticness", "instrumentalness", "liveness", "speechiness",
    ]

    # Build track_key in scrobbles
    sc = scrobbles.copy()
    sc["track_key"] = (
        sc["artist_name"].str.strip().str.lower()
        + "|||"
        + sc["track_name"].str.strip().str.lower()
    )

    # Merge with audio features
    merged = sc.merge(
        audio_features[["track_key"] + audio_cols].dropna(subset=audio_cols),
        on="track_key",
        how="left",
    )

    # Mean audio features per user (user-level listening profile)
    user_audio = merged.groupby("userid")[audio_cols].mean()
    user_audio.columns = [f"mean_{c}" for c in user_audio.columns]
    return user_audio
