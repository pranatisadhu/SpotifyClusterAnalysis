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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import timedelta


def _session_ids(timestamps: pd.Series, gap_minutes: int) -> pd.Series:
    """Return a session-ID series for an already-sorted timestamp series."""
    gap = pd.Timedelta(minutes=gap_minutes)
    starts = timestamps.diff().gt(gap) | timestamps.diff().isna()
    return starts.cumsum()


def compute_engagement_features(
    scrobbles: pd.DataFrame,
    session_gap_minutes: int = 30,
) -> pd.DataFrame:
    """
    Compute engagement metrics for each user using a single groupby pass —
    avoids 992 individual full-table scans of the 19 M-row scrobble file.
    """
    # Sort once; all per-user operations then work on contiguous slices
    sc = scrobbles[["userid", "timestamp", "artist_name", "track_name"]].copy()
    sc["timestamp"] = pd.to_datetime(sc["timestamp"], utc=True)
    sc = sc.sort_values(["userid", "timestamp"]).reset_index(drop=True)

    records: list[dict] = []

    for uid, df in sc.groupby("userid", sort=False):
        total_scrobbles = len(df)
        unique_tracks = df.groupby(["artist_name", "track_name"]).ngroups
        track_replay_rate = total_scrobbles / unique_tracks if unique_tracks else 0.0

        top_artist_plays = df["artist_name"].value_counts().iloc[0] if total_scrobbles else 0
        top_artist_share = top_artist_plays / total_scrobbles if total_scrobbles else 0.0

        # Sessions — timestamps already sorted
        session_ids = _session_ids(df["timestamp"], session_gap_minutes)
        session_stats = df.groupby(session_ids).agg(
            track_count=("track_name", "count"),
            duration=("timestamp", lambda ts: (ts.max() - ts.min()).total_seconds() / 60),
        )
        session_count = len(session_stats)
        avg_tracks_per_session = float(session_stats["track_count"].mean()) if session_count else 0.0
        avg_session_length_min = float(session_stats["duration"].mean()) if session_count else 0.0

        # Discovery velocity
        now = df["timestamp"].max()

        def _vel(days: int) -> float:
            cutoff = now - timedelta(days=days)
            past = df.loc[df["timestamp"] < cutoff, "artist_name"].unique()
            recent = df[df["timestamp"] >= cutoff]
            if recent.empty:
                return 0.0
            return recent[~recent["artist_name"].isin(past)]["artist_name"].nunique() / days

        vel_30 = _vel(30)
        vel_90 = _vel(90)

        # Novelty ratio
        cutoff_90 = now - timedelta(days=90)
        prior = df.loc[df["timestamp"] < cutoff_90, "artist_name"].unique()
        recent_90 = df[df["timestamp"] >= cutoff_90]
        novelty_ratio = (
            len(recent_90[~recent_90["artist_name"].isin(prior)]) / len(recent_90)
            if not recent_90.empty else 0.0
        )

        records.append({
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
        })

    return pd.DataFrame(records).set_index("userid")


def compute_audio_feature_profile(
    scrobbles: pd.DataFrame,
    audio_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate Spotify audio features to user level (weighted mean by play count).
    """
    audio_cols = [
        "danceability", "energy", "valence", "tempo",
        "acousticness", "instrumentalness", "liveness", "speechiness",
    ]

    sc = scrobbles[["userid", "artist_name", "track_name"]].copy()
    sc["track_key"] = (
        sc["artist_name"].str.strip().str.lower()
        + "|||"
        + sc["track_name"].str.strip().str.lower()
    )

    merged = sc.merge(
        audio_features[["track_key"] + audio_cols].dropna(subset=audio_cols),
        on="track_key",
        how="left",
    )

    user_audio = merged.groupby("userid")[audio_cols].mean()
    user_audio.columns = [f"mean_{c}" for c in user_audio.columns]
    return user_audio
