"""
generate_sample_data.py
-----------------------
Generates synthetic sample data for development and testing.
Produces all the files that Notebooks 01 and 02 need, without
requiring real Last.fm / Spotify API credentials or the large raw TSV.

Run from the project root:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_USERS = 50
MIN_SCROBBLES = 200
MAX_SCROBBLES = 1500
N_ARTISTS = 120
N_TRACKS_PER_ARTIST = 8

GENDERS = ["m", "f", "n"]
COUNTRIES = ["United States", "United Kingdom", "Germany", "Brazil", "Canada",
             "France", "Australia", "Sweden", "Netherlands", "Poland"]
GENRES_POOL = [
    "rock", "indie rock", "alternative rock", "pop", "indie pop",
    "electronic", "ambient", "techno", "house", "deep house",
    "hip hop", "rap", "trap", "r&b", "soul",
    "jazz", "jazz fusion", "blues",
    "classical", "orchestral",
    "metal", "heavy metal", "death metal",
    "folk", "country", "americana",
    "reggae", "ska", "punk", "post-punk",
]

ARTIST_NAMES = [
    "The Velvet Underground", "Radiohead", "Portishead", "Massive Attack",
    "Björk", "PJ Harvey", "Nick Cave", "Tom Waits", "Leonard Cohen",
    "David Bowie", "Lou Reed", "Iggy Pop", "Patti Smith", "Television",
    "Talking Heads", "Wire", "Joy Division", "New Order", "The Cure",
    "Siouxsie and the Banshees", "Bauhaus", "Sisters of Mercy",
    "The Smiths", "Morrissey", "The Fall", "Gang of Four", "Magazine",
    "Can", "Neu!", "Kraftwerk", "Faust", "Tangerine Dream",
    "Brian Eno", "Harold Budd", "Klaus Schulze", "Cluster",
    "Aphex Twin", "Autechre", "Squarepusher", "Boards of Canada",
    "The Orb", "Orbital", "Underworld", "Leftfield", "Portishead",
    "Tricky", "Goldie", "LTJ Bukem", "4Hero",
    "Miles Davis", "John Coltrane", "Bill Evans", "Thelonious Monk",
    "Charles Mingus", "Ornette Coleman", "Sun Ra",
    "Bob Dylan", "Neil Young", "Joni Mitchell", "Nick Drake",
    "Richard Thompson", "Sandy Denny", "John Martyn",
    "The Beatles", "The Rolling Stones", "The Kinks", "The Who",
    "Led Zeppelin", "Pink Floyd", "Yes", "Genesis", "King Crimson",
    "Van der Graaf Generator", "Soft Machine", "Hatfield and the North",
    "Nico", "John Cale", "Sterling Morrison",
    "Suicide", "Chrome", "Pere Ubu", "Devo",
    "Captain Beefheart", "Frank Zappa",
    "Scott Walker", "Current 93", "Death in June", "Coil",
    "Skinny Puppy", "Front 242", "Einstürzende Neubauten",
    "Swans", "Sonic Youth", "My Bloody Valentine", "Ride",
    "Slowdive", "Lush", "Cocteau Twins", "This Mortal Coil",
    "Dead Can Dance", "4AD", "Mark Hollis", "Talk Talk",
    "The Blue Nile", "Lloyd Cole", "The Triffids",
    "AR Kane", "A.R. Kane", "Felt", "The Durutti Column",
    "Cabaret Voltaire", "Throbbing Gristle", "SPK",
    "Nurse with Wound", "Current 93", "Psychic TV",
    "Hüsker Dü", "The Replacements", "Minutemen",
    "Pixies", "Dinosaur Jr.", "Pavement", "Guided by Voices",
    "Built to Spill", "Neutral Milk Hotel", "Sebadoh",
    "Modest Mouse", "Elliott Smith", "Bright Eyes",
][:N_ARTISTS]

# Build track lists per artist
ARTIST_TRACKS: dict[str, list[str]] = {}
for art in ARTIST_NAMES:
    ARTIST_TRACKS[art] = [f"{art} - Track {i+1}" for i in range(N_TRACKS_PER_ARTIST)]

# Assign genres to artists (2-4 genres each)
ARTIST_GENRE_MAP: dict[str, list[str]] = {}
for art in ARTIST_NAMES:
    n_genres = rng.integers(2, 5)
    ARTIST_GENRE_MAP[art] = list(rng.choice(GENRES_POOL, size=n_genres, replace=False))

# Assign audio features to each (artist, track) pair
TRACK_AUDIO: dict[tuple[str, str], dict] = {}
for art in ARTIST_NAMES:
    for track in ARTIST_TRACKS[art]:
        TRACK_AUDIO[(art, track)] = {
            "danceability":     float(rng.uniform(0.1, 0.9)),
            "energy":           float(rng.uniform(0.1, 0.95)),
            "valence":          float(rng.uniform(0.05, 0.95)),
            "tempo":            float(rng.uniform(60, 180)),
            "acousticness":     float(rng.uniform(0.0, 0.98)),
            "instrumentalness": float(rng.uniform(0.0, 0.9)),
            "liveness":         float(rng.uniform(0.05, 0.5)),
            "speechiness":      float(rng.uniform(0.02, 0.35)),
            "loudness":         float(rng.uniform(-20, -3)),
            "mode":             int(rng.integers(0, 2)),
            "time_signature":   int(rng.choice([3, 4, 4, 4, 4])),
        }


# ---------------------------------------------------------------------------
# Generate scrobbles
# ---------------------------------------------------------------------------

def _make_scrobbles() -> pd.DataFrame:
    records = []
    now = pd.Timestamp("2024-01-01", tz="UTC")
    five_years_ago = now - pd.Timedelta(days=5 * 365)

    users = [f"user_{i:04d}" for i in range(N_USERS)]

    # Give each user a preferred set of artists (biased listening)
    user_artist_weights: dict[str, np.ndarray] = {}
    for uid in users:
        weights = rng.exponential(scale=1.0, size=N_ARTISTS)
        weights /= weights.sum()
        user_artist_weights[uid] = weights

    for uid in users:
        n_scrobbles = int(rng.integers(MIN_SCROBBLES, MAX_SCROBBLES + 1))
        # Random timestamps spread over 5 years
        timestamps = sorted(
            pd.to_datetime(
                rng.integers(
                    int(five_years_ago.timestamp()),
                    int(now.timestamp()),
                    size=n_scrobbles,
                ),
                unit="s",
                utc=True,
            )
        )
        artist_indices = rng.choice(N_ARTISTS, size=n_scrobbles,
                                     p=user_artist_weights[uid])
        for ts, ai in zip(timestamps, artist_indices):
            art = ARTIST_NAMES[ai]
            track = random.choice(ARTIST_TRACKS[art])
            records.append({
                "userid": uid,
                "timestamp": ts,
                "artist_mbid": "",
                "artist_name": art,
                "track_mbid": "",
                "track_name": track,
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["userid", "timestamp"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generate profiles
# ---------------------------------------------------------------------------

def _make_profiles(users: list[str]) -> pd.DataFrame:
    records = []
    for uid in users:
        records.append({
            "userid": uid,
            "gender": random.choice(GENDERS),
            "age": int(rng.integers(16, 60)) if rng.random() > 0.1 else None,
            "country": random.choice(COUNTRIES),
            "signup": pd.Timestamp("2005-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 365 * 10))
            ),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Generate artist_genres
# ---------------------------------------------------------------------------

def _make_artist_genres() -> pd.DataFrame:
    records = []
    for art in ARTIST_NAMES:
        records.append({
            "artist_name_normalized": art.strip().lower(),
            "spotify_artist_id": f"sp_{abs(hash(art)) % 10**18:018d}",
            "genres": ARTIST_GENRE_MAP[art],
            "popularity": int(rng.integers(20, 90)),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Generate audio_features
# ---------------------------------------------------------------------------

def _make_audio_features() -> pd.DataFrame:
    records = []
    for (art, track), feats in TRACK_AUDIO.items():
        key = f"{art.strip().lower()}|||{track.strip().lower()}"
        records.append({
            "track_key": key,
            "track_spotify_id": f"tr_{abs(hash(key)) % 10**18:018d}",
            **feats,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out = Path(__file__).parent.parent / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    print("Generating scrobbles...")
    scrobbles = _make_scrobbles()
    users = scrobbles["userid"].unique().tolist()
    scrobbles.to_parquet(out / "scrobbles_updated.parquet", index=False)
    scrobbles.to_parquet(out / "scrobbles_baseline.parquet", index=False)
    print(f"  {len(scrobbles):,} scrobbles, {len(users)} users → scrobbles_updated.parquet + scrobbles_baseline.parquet")

    print("Generating profiles...")
    profiles = _make_profiles(users)
    profiles.to_parquet(out / "profiles.parquet", index=False)
    print(f"  {len(profiles)} users → profiles.parquet")

    print("Generating artist genres...")
    artist_genres = _make_artist_genres()
    artist_genres.to_parquet(out / "artist_genres.parquet", index=False)
    print(f"  {len(artist_genres)} artists → artist_genres.parquet")

    print("Generating audio features...")
    audio_features = _make_audio_features()
    audio_features.to_parquet(out / "audio_features.parquet", index=False)
    print(f"  {len(audio_features)} tracks → audio_features.parquet")

    # Also write small raw TSVs so Notebook 01 can load them without error
    raw = Path(__file__).parent.parent / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    print("Writing raw sample TSVs...")
    scrobbles_tsv = scrobbles.copy()
    scrobbles_tsv["timestamp"] = scrobbles_tsv["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    scrobbles_tsv[["userid", "timestamp", "artist_mbid", "artist_name",
                   "track_mbid", "track_name"]].to_csv(
        raw / "userid-timestamp-artid-artname-traid-traname.tsv",
        sep="\t", header=False, index=False,
    )

    profiles_tsv = profiles.copy()
    profiles_tsv["signup"] = pd.to_datetime(profiles_tsv["signup"]).dt.strftime("%b %d, %Y")
    profiles_tsv["age"] = profiles_tsv["age"].fillna("").astype(str)
    profiles_tsv[["userid", "gender", "age", "country", "signup"]].to_csv(
        raw / "userid-profile.tsv",
        sep="\t", header=False, index=False,
    )
    print("  Done.")

    print("\nSample data ready in data/processed/ and data/raw/")


if __name__ == "__main__":
    main()
