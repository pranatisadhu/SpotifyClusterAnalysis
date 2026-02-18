# Spotify × Last.fm Listener Cluster Analysis

**Research question:** What user segments exist based on listening diversity patterns, and how can streaming platforms use these insights to optimise personalisation strategies for different listener types?

---

## Project Overview

This project combines the **Last.fm Dataset 1K** (992 users, ~19M historical scrobbles) with live data fetched via the **Last.fm API** (last 5 years) and enriched with **Spotify API** genre and audio features. The resulting user-level behavioural feature matrix is then clustered to identify distinct listener archetypes.

---

## Setup

### 1. Clone and install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add API credentials

```bash
cp .env.example .env
# Edit .env with your Last.fm and Spotify API keys
```

- **Last.fm API:** Register at https://www.last.fm/api/account/create
- **Spotify API:** Create an app at https://developer.spotify.com/dashboard

### 3. Place raw data files

```
data/raw/
├── userid-timestamp-artid-artname-traid-traname.tsv   # 2.53 GB scrobble history
└── userid-profile.tsv                                  # 38 KB user demographics
```

---

## Running the Analysis

Run the notebooks in order:

| Notebook | Purpose |
|---|---|
| `01_data_collection.ipynb` | Load baseline TSV, fetch 5-year API update, Spotify enrichment |
| `02_feature_engineering.ipynb` | Compute all behavioural features, build feature matrix |
| `03_clustering.ipynb` | KMeans + HDBSCAN, model selection, cluster profiling |
| `04_insights_visualization.ipynb` | Interactive Plotly charts + personalisation recommendations |

Or run the full enrichment pipeline as a script:

```bash
python -m src.data.enrichment_pipeline
```

---

## Feature Matrix

All features are computed at the **user level** (one row per user):

### Diversity Features
| Feature | Description |
|---|---|
| `unique_artists` | Count of distinct artists listened to |
| `artist_entropy` | Shannon entropy of artist play distribution |
| `artist_concentration_20` | % of plays from top 20 artists |
| `unique_genres` | Count of distinct Spotify genre tags |
| `genre_entropy` | Shannon entropy of genre distribution |
| `genre_concentration_5` | % of plays from top 5 genres |
| `avg_genre_tags_per_play` | Mean genre diversity per play event |

### Engagement & Discovery Features
| Feature | Description |
|---|---|
| `track_replay_rate` | Total plays / unique tracks (>1 = re-listener) |
| `avg_tracks_per_session` | Mean tracks per detected listening session |
| `discovery_velocity_30d` | New artists per day in most recent 30 days |
| `discovery_velocity_90d` | New artists per day in most recent 90 days |
| `novelty_ratio` | % of recent plays from newly discovered artists |
| `top_artist_play_share` | Fraction of plays by single top artist |

### Temporal Features
| Feature | Description |
|---|---|
| `temporal_hour_entropy` | Entropy of play distribution across hours of day |
| `temporal_dow_entropy` | Entropy across days of week |
| `morning_ratio` | Fraction of plays 06:00–12:00 |
| `evening_ratio` | Fraction of plays 18:00–00:00 |
| `weekend_ratio` | Fraction of plays on weekends |
| `temporal_stability_pc1..5` | PCA components of hourly profile (pattern stability) |

### Spotify Audio Profile
| Feature | Description |
|---|---|
| `mean_energy` | Mean track energy (0–1) |
| `mean_valence` | Mean track positivity/mood (0–1) |
| `mean_danceability` | Mean danceability score (0–1) |
| `mean_acousticness` | Mean acousticness (0–1) |
| `mean_instrumentalness` | Mean instrumentalness (0–1) |

---

## Clustering Approach

1. **Preprocessing:** Median imputation + StandardScaler
2. **Dimensionality reduction:** PCA (retain 95% variance) — reduces multicollinearity across correlated features
3. **Algorithms:** KMeans (k swept 3–10, selected by silhouette) and HDBSCAN
4. **Visualisation:** UMAP 2D embedding for cluster scatter plots
5. **Evaluation:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, bootstrap stability

---

## Project Structure

```
SpotifyClusterAnalysis/
├── configs/
│   └── config.yaml              # All parameters (API settings, feature params, cluster config)
├── data/
│   ├── raw/                     # .gitignored — place TSV files here
│   ├── processed/               # .gitignored — generated Parquet files
│   └── sample/                  # Small sample data for development
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_insights_visualization.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py            # TSV loading with chunked reading
│   │   ├── lastfm_client.py     # Last.fm API — paginated scrobble fetch
│   │   ├── spotify_client.py    # Spotify API — genre + audio features
│   │   └── enrichment_pipeline.py  # Orchestration script
│   ├── features/
│   │   ├── diversity.py         # Artist & genre diversity metrics
│   │   ├── temporal.py          # Temporal entropy + PCA stability
│   │   ├── engagement.py        # Sessions, replay, discovery
│   │   └── builder.py           # Assemble full feature matrix
│   ├── clustering/
│   │   ├── pipeline.py          # Preprocessing → PCA → cluster → UMAP
│   │   └── evaluation.py        # Metrics, profiling, feature importance
│   └── visualization/
│       └── plots.py             # All Plotly chart functions
├── outputs/
│   └── figures/                 # HTML and PNG chart outputs
├── .env.example                 # API credential template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Attribution

The baseline dataset is the **Last.fm Dataset 1K** (Oscar Celma, 2010), distributed with permission of Last.fm for non-commercial use. Updated listening data is fetched via the Last.fm API under their Terms of Service.
