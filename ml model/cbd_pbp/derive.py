from __future__ import annotations
import os
import duckdb
from .warehouse import Warehouse

def execute_staging_layer(wh: Warehouse, sql_path: str = "cbd_pbp/staging_layer.sql"):
    """
    Executes the SQL Staging Layer to transform Raw JSON into Fact Tables.
    
    Architecture:
    [Raw Tables] -> [Staging Views] -> [Materialized Facts]
    
    This replaces the old Python-based derivation logic with pure DuckDB SQL.
    """
    print(f"Executing Staging Layer from {sql_path}...")
    
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"Staging SQL file not found: {sql_path}")
        
    with open(sql_path, "r") as f:
        sql_content = f.read()
        
    # Split by semicolon to execute statement by statement (better error reporting)
    # Filter out empty statements
    statements = [s.strip() for s in sql_content.split(";") if s.strip()]
    
    with wh.con.cursor() as cur:
        for i, stmt in enumerate(statements):
            if stmt.startswith("--"): 
                continue
            try:
                # Log progress for big tables
                if "CREATE OR REPLACE TABLE" in stmt:
                    # Extract table name for logging
                    tbl = stmt.split("TABLE")[1].split("AS")[0].strip()
                    print(f"  Building {tbl}...")
                elif "CREATE OR REPLACE VIEW" in stmt:
                     tbl = stmt.split("VIEW")[1].split("AS")[0].strip()
                     # specific logging for views if needed, or silent
                     pass
                
                cur.execute(stmt)
            except Exception as e:
                print(f"Error executing statement #{i+1}:\n{stmt[:100]}...\nError: {e}")
                raise e
    
    print("Staging Layer Execution Complete.")

def build_derived_sql(wh: Warehouse):
    """Wrapper to call execute_staging_layer with default path."""
    # distinct from old functions to avoid confusion
    execute_staging_layer(wh)
