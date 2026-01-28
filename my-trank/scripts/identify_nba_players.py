import pandas as pd
import os
import re

def identify_nba_players():
    # Paths
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    SEASON_CSV = os.path.join(BASE_DIR, 'public', 'data', 'season.csv')
    CAREER_CSV = os.path.join(BASE_DIR, 'public', 'data', 'career.csv')
    INTL_CSV = os.path.join(BASE_DIR, 'public', 'data', 'international_stat_history', 'internationalplayerarchive.csv')
    UNDRAFTED_CSV = os.path.join(BASE_DIR, 'data', 'undrafted_nba_players.csv')
    TANKATHON_CSV = os.path.join(BASE_DIR, 'data', 'tankathon_draft_picks.csv')

    print("Loading data files...")
    # Load with low_memory=False to avoid DtypeWarning
    df_season = pd.read_csv(SEASON_CSV, low_memory=False)
    df_intl = pd.read_csv(INTL_CSV, low_memory=False)
    df_undrafted = pd.read_csv(UNDRAFTED_CSV)
    
    # Get unique player names from our database
    db_players = set(df_season['key'].dropna().unique())
    db_players.update(set(df_intl['key'].dropna().unique()))
    print(f"Total players in DB: {len(db_players)}")

    # Get undrafted names
    undrafted_names = set(df_undrafted['player_name_lower'].unique())
    print(f"Total undrafted players from Wikipedia: {len(undrafted_names)}")

    # Identify "Made NBA"
    nba_players = []
    
    # 1. Check players with draft picks in our DB
    for df in [df_season, df_intl]:
        drafted = df[df['pick'].notna() & (df['pick'] != 'NA') & (df['pick'] != '')]
        for _, row in drafted.iterrows():
            nba_players.append({
                'name': row['key'],
                'name_lower': str(row['key']).lower(),
                'status': 'drafted',
                'pick': row['pick'],
                'college': row['team'] if 'team' in row else ''
            })

    # 2. Check players in undrafted list
    for name in db_players:
        name_lower = str(name).lower()
        if name_lower in undrafted_names:
            nba_players.append({
                'name': name,
                'name_lower': name_lower,
                'status': 'undrafted',
                'pick': 'NA',
                'college': '' # Will try to populate later
            })

    # Deduplicate by name_lower
    nba_df = pd.DataFrame(nba_players).drop_duplicates(subset=['name_lower'])
    print(f"Identified {len(nba_df)} unique players who made the NBA")
    
    output_path = os.path.join(BASE_DIR, 'data', 'nba_player_lookup.csv')
    nba_df.to_csv(output_path, index=False)
    print(f"Saved lookup to {output_path}")

if __name__ == "__main__":
    identify_nba_players()
