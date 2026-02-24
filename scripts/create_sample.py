"""
create_sample.py
----------------
Creates a memory-safe sample of scrobbles_updated.csv for machines with ≤ 8 GB RAM.

Uses DuckDB to query the 2.4 GB CSV directly — the full file is NEVER loaded
into Python memory. Peak RAM usage is ~300 MB regardless of input file size.

Usage (run from project root):
    python scripts/create_sample.py

Output:
    data/processed/scrobbles_sampled.csv   (~200-400 MB, 500 users)

After running this, notebook 02 will automatically detect and use this
smaller file instead of the full 2.4 GB one.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent

SOURCE  = PROJECT_ROOT / 'data' / 'processed' / 'scrobbles_updated.csv'
OUTPUT  = PROJECT_ROOT / 'data' / 'processed' / 'scrobbles_sampled.csv'
PROFILE_SRC = PROJECT_ROOT / 'data' / 'processed' / 'profiles.csv'
PROFILE_OUT = PROJECT_ROOT / 'data' / 'processed' / 'profiles_sampled.csv'

N_USERS = 500   # enough for k=3-10 clusters (~50-160 users per cluster)
SEED    = 42    # fixed seed → reproducible sample

# ── Checks ───────────────────────────────────────────────────────────────────
if not SOURCE.exists():
    print(f'ERROR: {SOURCE} not found. Run notebook 01 first.')
    sys.exit(1)

try:
    import duckdb
except ImportError:
    print('ERROR: duckdb not installed. Run:  pip install duckdb')
    sys.exit(1)

# ── Sample users ─────────────────────────────────────────────────────────────
src_gb = SOURCE.stat().st_size / 1e9
print(f'Source : {SOURCE.name}  ({src_gb:.1f} GB)')
print(f'Sampling {N_USERS} users (seed={SEED}) — DuckDB peak RAM ~300 MB ...')

duckdb.sql(f"""
    COPY (
        WITH sampled_users AS (
            SELECT DISTINCT userid
            FROM read_csv_auto('{SOURCE}', header=True)
            USING SAMPLE {N_USERS} ROWS (bernoulli, {SEED})
        )
        SELECT s.*
        FROM read_csv_auto('{SOURCE}', header=True) s
        INNER JOIN sampled_users u USING (userid)
        ORDER BY s.userid, s.timestamp
    )
    TO '{OUTPUT}' (HEADER, DELIMITER ',')
""")

out_mb = OUTPUT.stat().st_size / 1e6
print(f'Scrobbles sample written: {OUTPUT.name}  ({out_mb:.0f} MB)')

# Verify row / user counts
result = duckdb.sql(f"""
    SELECT COUNT(*) AS rows, COUNT(DISTINCT userid) AS users
    FROM read_csv_auto('{OUTPUT}', header=True)
""").fetchone()
print(f'  → {result[0]:,} rows, {result[1]:,} users')

# ── Filter profiles to match sampled users ────────────────────────────────────
if PROFILE_SRC.exists():
    print(f'\nFiltering profiles to sampled users ...')
    duckdb.sql(f"""
        COPY (
            WITH sampled_users AS (
                SELECT DISTINCT userid
                FROM read_csv_auto('{OUTPUT}', header=True)
            )
            SELECT p.*
            FROM read_csv_auto('{PROFILE_SRC}', header=True) p
            INNER JOIN sampled_users u USING (userid)
        )
        TO '{PROFILE_OUT}' (HEADER, DELIMITER ',')
    """)
    n_profiles = duckdb.sql(
        f"SELECT COUNT(*) FROM read_csv_auto('{PROFILE_OUT}', header=True)"
    ).fetchone()[0]
    print(f'Profiles sample written: {PROFILE_OUT.name}  ({n_profiles:,} users)')

print('\nDone. You can now run notebook 02.')
print(f'It will automatically use the sampled files.')
