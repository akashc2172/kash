"""
College Data Explorer - FULL CATALOG
====================================
Explore EVERY table in the warehouse to show the full breadth of data.

Tables to Inspect:
1. PBP & Shots (Action)
2. Lineups & Participants (Context)
3. Recruiting & Transfers (Talent)
4. Conference & Team Info (Meta)
5. Any derived stats (if present)
"""

import duckdb
import pandas as pd

# Connect to warehouse
con = duckdb.connect('data/warehouse.duckdb')

def get_table_list():
    """Get list of all tables in the database."""
    return [t[0] for t in con.execute("SHOW TABLES").fetchall()]

def sample_table(table_name: str, limit: int = 3):
    """Show a sample of a table with its columns."""
    print(f"\n{'='*50}")
    print(f"TABLE: {table_name}")
    print(f"{'='*50}")
    
    # Get row count
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"Total Rows: {count:,}")
    except:
        print("Total Rows: Unknown")
        
    # Get columns
    try:
        schema = con.execute(f"DESCRIBE {table_name}").fetchall()
        cols = [s[0] for s in schema]
        print(f"Columns ({len(cols)}): {', '.join(cols[:10])} ...")
    except:
        pass

    # Get sample
    try:
        df = con.query(f"SELECT * FROM {table_name} LIMIT {limit}").to_df()
        if not df.empty:
            print("\nSample Data:")
            print(df.to_string(index=False))
        else:
            print("\n(Table is empty)")
    except Exception as e:
        print(f"\nError sampling: {e}")

def main():
    print("--- FULL COLLEGE DATA INVENTORY ---\n")
    
    tables = get_table_list()
    print(f"Found {len(tables)} tables: {', '.join(tables)}\n")
    
    # Prioritize interesting tables
    priority_tables = [
        'fact_recruiting_players',  # Recruiting rankings
        'fact_transfers',           # (If exists)
        'dim_players',              # (If exists)
        'stg_sys_efficiency',       # (If exists)
        'dim_venues',               # Venues
        'dim_conferences',          # Conferences
        'fact_box_scores'           # (If exists)
    ]
    
    # 1. Show priority tables first
    for t in priority_tables:
        if t in tables:
            sample_table(t)
            
    # 2. Show all others (briefly)
    for t in tables:
        if t not in priority_tables and not t.startswith('stg_'):
            sample_table(t)

if __name__ == "__main__":
    main()
