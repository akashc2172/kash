"""
Export Data Dictionary to CSV
=============================
Exports a comprehensive catalog of all tables and columns in the warehouse to a CSV file.

Output:
data_dictionary.csv
- Table Name
- Column Name
- Data Type
- Sample Values (first non-null value)
"""

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

def get_sample_value(table_name, column_name):
    """Get a sample value for a column."""
    try:
        val = con.execute(f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 1").fetchone()
        return val[0] if val else None
    except:
        return None

def main():
    print("Generating Dictionary CSV...")
    
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    
    with open('config/warehouse_data_dictionary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Table', 'Column', 'Type', 'Sample Value'])
        
        for table in tables:
            print(f"Processing {table}...")
            schema = get_table_schema(table)
            
            for col_info in schema:
                col_name = col_info[0]
                col_type = col_info[1]
                sample = get_sample_value(table, col_name)
                
                writer.writerow([table, col_name, col_type, sample])
                
    print("\nDone! Saved to config/warehouse_data_dictionary.csv")

if __name__ == "__main__":
    main()
