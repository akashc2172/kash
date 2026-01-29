import duckdb
import pandas as pd
import csv

# Connect to warehouse
con = duckdb.connect('data/warehouse.duckdb')

def get_table_schema(table_name):
    """Get columns and types for a table."""
    try:
        return con.execute(f"DESCRIBE {table_name}").fetchall()
    except:
        return []

def get_sample_value(table_name, column_name, season_year=2025):
    """Get a sample value for a column specifically from the 2025 season."""
    try:
        # Filter by season = 2025
        query = f"SELECT {column_name} FROM {table_name} WHERE season = {season_year} AND {column_name} IS NOT NULL LIMIT 1"
        val = con.execute(query).fetchone()
        return val[0] if val else None
    except Exception as e:
        return None

def main():
    print("Generating 2025 Data Dictionary...")
    
    target_table = "fact_play_raw"
    output_file = "data/debug/full_data_dictionary_2025.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Table', 'Column', 'Type', 'Sample Value (2025)'])
        
        print(f"Processing {target_table}...")
        schema = get_table_schema(target_table)
        
        for col_info in schema:
            col_name = col_info[0]
            col_type = col_info[1]
            sample = get_sample_value(target_table, col_name, 2025)
            
            # Format the sample value if it's complex (struct/list)
            if sample is not None:
                sample_str = str(sample)
                # Truncate if insanely long
                if len(sample_str) > 200:
                    sample_str = sample_str[:197] + "..."
            else:
                sample_str = "NULL"
                
            writer.writerow([target_table, col_name, col_type, sample_str])
            print(f"  - {col_name}: {sample_str}")
                
    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    main()
