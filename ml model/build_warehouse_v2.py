import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import re
import sys

# Configuration
BASE_DIR = Path("/Users/akashc/my-trankcopy/ml model")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "warehouse_v2"
EPM_DIR = DATA_DIR / "epm "  # Note the space in the directory name if preserved from previous steps
BE_DIR = DATA_DIR / "basketball_excel"
RAPM_PATH = DATA_DIR / "nba_six_factor_rapm_clean.csv"
WHITELIST_PATH = BASE_DIR / "nba_aux_whitelist_v2.yaml"

def load_config():
    with open(WHITELIST_PATH, "r") as f:
        return yaml.safe_load(f)

def clean_id(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return -1

def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan

# --- Step A: Raw Ingest ---

def load_raw_epm(config):
    """Load and union all EPM CSVs, filtering columns."""
    print("Loading Raw EPM...")
    epm_conf = config['sources']['dunks_threes_epm']
    keep_cols = epm_conf['include_columns']
    
    parts = []
    # Find all years matching pattern
    # The file pattern in yaml is: dunks_threes_stats_{season}.csv
    # But files on disk are: "Dunks & Threes Stats {year}.csv"
    # We will trust the disk pattern we observed earlier.
    
    for f in EPM_DIR.glob("Dunks & Threes Stats *.csv"):
        match = re.search(r"(\d{4})", f.name)
        if not match:
            continue
        year = int(match.group(1))
        
        df = pd.read_csv(f)
        
        # Normalize columns to lowercase for consistent checking
        # But we must map back to the whitelist names or just rename?
        # The whitelist has 'player_id', 'season', etc. The CSV usually has them.
        
        # Ensure season is present
        if 'season' not in df.columns:
            df['season'] = year
            
        # Select whitelisted columns (intersection)
        available_cols = [c for c in keep_cols if c in df.columns]
        df = df[available_cols].copy()
        
        # Type enforcement for IDs
        if 'player_id' in df.columns:
            df['player_id'] = df['player_id'].apply(clean_id)
            
        parts.append(df)
        
    if not parts:
        raise ValueError("No EPM files found!")
        
    raw_epm = pd.concat(parts, ignore_index=True)
    print(f"  Loaded {len(raw_epm)} rows from {len(parts)} seasons.")
    return raw_epm

def load_raw_be(config):
    """Load Basketball-Excel parquet, filtering columns."""
    print("Loading Raw Basketball-Excel...")
    be_conf = config['sources']['basketball_excel']
    path = BE_DIR / be_conf['table']
    
    if not path.exists():
        path_sub = BE_DIR / "players" / "all_players.parquet"
        if path_sub.exists():
            path = path_sub
        else:
            raise FileNotFoundError(f"Could not find BE parquet at {path}")
            
    df = pd.read_parquet(path)
    
    # Filter columns
    exact_cols = be_conf.get('include_columns', [])
    prefixes = be_conf.get('include_prefixes', [])
    
    keep = []
    # Always keep 'nid' and 'pid' for join keys if present, even if not in whitelist?
    # Whitelist has 'nid' and 'pid'.
    
    for c in df.columns:
        if c in exact_cols:
            keep.append(c)
        elif any(c.startswith(p) for p in prefixes):
            keep.append(c)
            
    df = df[keep].copy()
    
    # Standardize IDs
    # 'nid' is the numeric NBA ID. 'pid' is the slug.
    if 'nid' in df.columns:
        df['nba_id'] = df['nid'].apply(clean_id)
    else:
        # Should not happen based on inspection, but fallback:
        print("WARNING: 'nid' column missing in BE data! merge will fail.")
        df['nba_id'] = -1

    if 'bbr_pid' in df.columns:
        df['bbr_pid'] = df['bbr_pid'].astype(str)
        
    print(f"  Loaded {len(df)} rows. Kept {len(keep)}/{len(df.columns)} columns.")
    return df

def load_raw_rapm():
    """Load cleaned RAPM file."""
    print("Loading Raw RAPM...")
    df = pd.read_csv(RAPM_PATH)
    # df has nba_id, player_name, etc.
    if 'nba_id' in df.columns:
        df['nba_id'] = df['nba_id'].apply(clean_id)
    print(f"  Loaded {len(df)} rows.")
    return df

# --- Step B: Crosswalk ---

def build_dim_player_crosswalk(raw_be, raw_epm, raw_rapm):
    """Build unique player list from all sources."""
    print("Building dim_player_crosswalk...")
    
    # 1. From Basketball-Excel (Primary source for metadata)
    # Use 'nba_id' derived from 'nid'. 'pid' is slug.
    be_players = raw_be[['nba_id', 'pid', 'bbr_pid', 'nm']].drop_duplicates()
    be_players = be_players.rename(columns={'nm': 'player_name', 'bbr_pid': 'bbr_id'})
    # pid (slug) kept as pid
    
    # 2. From EPM
    epm_players = raw_epm[['player_id', 'player_name']].drop_duplicates().rename(columns={'player_id': 'nba_id'})
    
    # 3. From RAPM
    rapm_players = raw_rapm[['nba_id', 'player_name']].drop_duplicates()
    
    # Merge strategy: BE is the spine, but we must include everyone.
    all_names = pd.concat([
        be_players[['nba_id', 'player_name']],
        epm_players[['nba_id', 'player_name']],
        rapm_players[['nba_id', 'player_name']]
    ]).drop_duplicates(subset=['nba_id'])
    
    # Join back aux IDs from BE
    aux_ids = be_players[['nba_id', 'pid', 'bbr_id']].drop_duplicates(subset=['nba_id']).dropna(subset=['nba_id'])
    
    crosswalk = pd.merge(all_names, aux_ids, on='nba_id', how='left')
    crosswalk = crosswalk[crosswalk['nba_id'] > 0]
    
    print(f"  Crosswalk: {len(crosswalk)} unique players.")
    return crosswalk

# --- Step C: Dimensions & Anthro ---

def build_dim_player_nba(raw_be, crosswalk, config):
    """Build player metadata and anthropometrics from NBA records."""
    print("Building dim_player_nba (Anthro)...")
    
    # 1. Rookie Year Logic (Strict)
    # Rule: If draft_year exists -> rookie_season = draft_year + 1
    #       Else -> first NBA season with mp > 200
    
    be_sorted = raw_be.sort_values('season_year')
    
    # Helper for MP
    be_sorted['mp'] = pd.to_numeric(be_sorted['mp'], errors='coerce').fillna(0)
    
    # Draft Info
    # Check if d_y exists
    if 'd_y' in be_sorted.columns:
        # Get first non-null draft year per player
        draft_info = be_sorted.dropna(subset=['d_y']).groupby('nba_id')['d_y'].first().reset_index()
        draft_info.columns = ['nba_id', 'draft_year']
        # Convert to int safe
        draft_info['draft_year'] = pd.to_numeric(draft_info['draft_year'], errors='coerce')
    else:
        draft_info = pd.DataFrame(columns=['nba_id', 'draft_year'])
        
    # Merge draft info onto base IDs
    player_base = pd.DataFrame({'nba_id': be_sorted['nba_id'].unique()})
    player_base = pd.merge(player_base, draft_info, on='nba_id', how='left')
    
    # Calculate Rookie Season
    def get_rookie_year(row):
        nba_id = row['nba_id']
        dy = row['draft_year']
        
        if pd.notna(dy) and dy > 0:
            return int(dy) + 1
        
        # Fallback: First season > 200 MP
        player_rows = be_sorted[be_sorted['nba_id'] == nba_id]
        qualified = player_rows[player_rows['mp'] > 200]
        if not qualified.empty:
            return qualified['season_year'].min()
        else:
            # Fallback to absolute first season if never played > 200?
            if not player_rows.empty:
                return player_rows['season_year'].min()
            return np.nan

    # Vectorized approach or apply? Apply is safer for logic mix
    # But apply per row on 2000 players is fine.
    # To optimize, pre-compute "first > 200" map
    
    qual_map = be_sorted[be_sorted['mp'] > 200].groupby('nba_id')['season_year'].min()
    abs_map = be_sorted.groupby('nba_id')['season_year'].min()
    
    def derive_rookie(row):
        if pd.notna(row['draft_year']) and row['draft_year'] > 0:
            return row['draft_year'] + 1
        
        nid = row['nba_id']
        if nid in qual_map:
            return qual_map[nid]
        if nid in abs_map:
            return abs_map[nid]
        return np.nan

    player_base['rookie_season_year'] = player_base.apply(derive_rookie, axis=1)
    
    # Cast to Int64 (nullable int) to avoid float format in parquet
    player_base['rookie_season_year'] = player_base['rookie_season_year'].astype('Int64')
    
    # 3. Anthropometrics (Growth)
    def clean_val(df, col):
        return pd.to_numeric(df[col], errors='coerce')

    anthro_df = be_sorted[['nba_id', 'season_year', 'ht', 'wt']].copy()
    anthro_df['ht'] = clean_val(anthro_df, 'ht')
    anthro_df['wt'] = clean_val(anthro_df, 'wt')

    anthro_df = anthro_df.sort_values('season_year')

    ht_first = anthro_df.dropna(subset=['ht']).groupby('nba_id')['ht'].first()
    ht_max = anthro_df.groupby('nba_id')['ht'].max()
    
    wt_first = anthro_df.dropna(subset=['wt']).groupby('nba_id')['wt'].first()
    wt_max = anthro_df.groupby('nba_id')['wt'].max()
    
    anthro_stats = pd.DataFrame({
        'ht_first': ht_first,
        'ht_max': ht_max,
        'wt_first': wt_first,
        'wt_max': wt_max
    }).reset_index()
    
    anthro_stats['ht_peak_delta'] = anthro_stats['ht_max'] - anthro_stats['ht_first']
    anthro_stats['wt_peak_delta'] = anthro_stats['wt_max'] - anthro_stats['wt_first']
    
    counts = anthro_df.dropna(subset=['ht']).groupby('nba_id')['season_year'].nunique()
    anthro_stats['has_ht_multiseason'] = anthro_stats['nba_id'].map(counts).fillna(0) >= 2
    
    counts_wt = anthro_df.dropna(subset=['wt']).groupby('nba_id')['season_year'].nunique()
    anthro_stats['has_wt_multiseason'] = anthro_stats['nba_id'].map(counts_wt).fillna(0) >= 2

    anthro_stats['wingspan_const'] = np.nan
    anthro_stats['has_wingspan'] = 0
    
    # Merge
    dim = pd.merge(crosswalk, player_base, on='nba_id', how='left')
    dim = pd.merge(dim, anthro_stats, on='nba_id', how='left')
    
    print(f"  Dim Player NBA: {len(dim)} rows.")
    return dim

# --- Step D: Facts ---

def build_fact_year1_epm(raw_epm, dim_player):
    """Build Year-1 EPM stats."""
    print("Building fact_player_year1_epm...")
    
    target_players = dim_player[['nba_id', 'rookie_season_year']].dropna()
    merged = pd.merge(raw_epm, target_players, left_on='player_id', right_on='nba_id', how='inner')
    
    # Filter: season match + regular season
    # EPM 'seasontype' usually 'Regular', 'Playoffs'
    # Whitelist says 'seasontype' column exists. dunks_threes_stats usually has it?
    # Checking load_raw_epm: it loads it.
    
    # If seasontype col exists, use it. Else assume file is regular?
    # Files are named "Dunks & Threes Stats {year}.csv". Usually regular season.
    # But let's check col.
    
    y1_data = merged[merged['season'] == merged['rookie_season_year']].copy()
    
    # Dedup: if multiple rows (e.g. traded), take total or weighted?
    # EPM usually has 'team_id' or 'Tm'. If multiple, usually a 'TOT' row exists?
    # EPM export usually has one row per player-season if it's the main leaderboard.
    # If traded, it might have separate.
    # We'll take the row with max MP if duplicates exist.
    
    if 'mp' in y1_data.columns:
        y1_data = y1_data.sort_values('mp', ascending=False)
        y1_data = y1_data.drop_duplicates(subset=['nba_id'])
    else:
        y1_data = y1_data.drop_duplicates(subset=['nba_id'])
    
    fact = dim_player[['nba_id']].copy()
    fact = pd.merge(fact, y1_data, on='nba_id', how='left')
    
    target_cols = {
        'tot': 'year1_epm_tot',
        'off': 'year1_epm_off',
        'def': 'year1_epm_def',
        'ewins': 'year1_epm_ewins',
        'usg': 'year1_usg',
        'tspct': 'year1_tspct',
        'mp': 'year1_mp'
    }
    
    for src, dst in target_cols.items():
        if src in fact.columns:
            fact[dst] = pd.to_numeric(fact[src], errors='coerce')
        else:
            fact[dst] = np.nan
            
    fact['missing_year1'] = fact['year1_epm_tot'].isna().astype(int)
    
    print(f"  Year 1 Facts: {len(fact)} rows. Missing Year 1: {fact['missing_year1'].sum()}")
    return fact

def build_fact_peak_rapm(raw_rapm, dim_player):
    """Build Peak RAPM targets."""
    print("Building fact_player_peak_rapm...")
    
    df = raw_rapm.copy()
    # Contract: peak_start = Latest - 2, peak_end = Latest
    df['peak_end_year'] = df['Latest_Year']
    df['peak_start_year'] = df['Latest_Year'] - 2
    
    renames = {
        'OVR_RAPM': 'y_peak_ovr',
        'Off_RAPM': 'y_peak_off',
        'Def_RAPM': 'y_peak_def',
        'Off_Poss': 'peak_poss'
    }
    df = df.rename(columns={k:v for k,v in renames.items() if k in df.columns})
    
    fact = pd.merge(dim_player[['nba_id']], df, on='nba_id', how='inner')
    
    cols = ['nba_id', 'y_peak_ovr', 'y_peak_off', 'y_peak_def', 'peak_poss', 'peak_end_year', 'peak_start_year']
    # Ensure all exist
    for c in cols:
        if c not in fact.columns:
            fact[c] = np.nan
            
    fact = fact[cols]
    
    print(f"  Peak RAPM Facts: {len(fact)} rows.")
    return fact

# --- Validate ---

def validate_warehouse(dim, fact_y1, fact_rapm):
    print("\n--- Validation ---")
    print(f"Dim Rows: {len(dim)}")
    print(f"Fact Y1 Rows: {len(fact_y1)}")
    print(f"Fact RAPM Rows: {len(fact_rapm)}")
    
    if len(dim) != len(fact_y1):
        print("WARNING: Dim and Fact Y1 row counts differ!")
        
    growth_fill = dim['ht_peak_delta'].notna().mean()
    print(f"Height Growth Fill Rate: {growth_fill:.2%}")
    
    missing_y1 = fact_y1['missing_year1'].mean()
    print(f"Missing Year 1 Rate: {missing_y1:.2%}")

# --- Main ---

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    
    # Ingest
    raw_epm = load_raw_epm(config)
    raw_be = load_raw_be(config)
    raw_rapm = load_raw_rapm()
    
    # Crosswalk
    crosswalk = build_dim_player_crosswalk(raw_be, raw_epm, raw_rapm)
    crosswalk.to_parquet(OUT_DIR / "dim_player_crosswalk.parquet", index=False)
    
    # Dim
    dim_player = build_dim_player_nba(raw_be, crosswalk, config)
    dim_player.to_parquet(OUT_DIR / "dim_player_nba.parquet", index=False)
    
    # Facts
    fact_y1 = build_fact_year1_epm(raw_epm, dim_player)
    fact_y1.to_parquet(OUT_DIR / "fact_player_year1_epm.parquet", index=False)
    
    fact_rapm = build_fact_peak_rapm(raw_rapm, dim_player)
    fact_rapm.to_parquet(OUT_DIR / "fact_player_peak_rapm.parquet", index=False)
    
    # Validate
    validate_warehouse(dim_player, fact_y1, fact_rapm)
    print("\nWarehouse v2 Build Complete.")

if __name__ == "__main__":
    main()
