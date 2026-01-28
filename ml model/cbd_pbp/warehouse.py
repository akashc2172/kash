from __future__ import annotations
import duckdb
import pandas as pd
from typing import Optional, Dict, Any, List

class Warehouse:
    def __init__(self, path: str):
        self.path = path
        self.con = duckdb.connect(path)

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def ensure_table(self, name: str, df: pd.DataFrame, pk: Optional[List[str]]=None):
        """
        Create table if not exists and insert data.
        Uses EXPLICIT COLUMN LISTS to prevent column-order drift.
        
        Args:
            name: Table name
            df: DataFrame to insert
            pk: Optional primary key columns for upsert behavior
        """
        if df.empty:
            return
            
        # Get incoming column names
        cols = list(df.columns)
        col_list = ", ".join([f'"{c}"' for c in cols])
        
        # Register the DataFrame
        self.con.register("incoming", df)
        
        # Check if table exists
        table_exists = self.con.execute(
            f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
        ).fetchone()[0] > 0
        
        if not table_exists:
            # Create table from DataFrame (first time)
            self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming LIMIT 0")
            self.con.execute(f"INSERT INTO {name} SELECT * FROM incoming")
        else:
            # Table exists - get its columns
            existing_cols = [row[0] for row in self.con.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{name}'"
            ).fetchall()]
            
            # Find common columns (intersection)
            common_cols = [c for c in cols if c in existing_cols]
            
            if not common_cols:
                print(f"Warning: No common columns between incoming data and {name}")
                self.con.unregister("incoming")
                return
            
            common_col_list = ", ".join([f'"{c}"' for c in common_cols])
            
            if pk:
                # Upsert: delete matching keys, then insert
                cond = " AND ".join([f'{name}."{c}" = incoming."{c}"' for c in pk if c in common_cols])
                if cond:
                    self.con.execute(f"DELETE FROM {name} USING incoming WHERE {cond}")
                self.con.execute(f"INSERT INTO {name}({common_col_list}) SELECT {common_col_list} FROM incoming")
            else:
                # Append only - use explicit column list
                self.con.execute(f"INSERT INTO {name}({common_col_list}) SELECT {common_col_list} FROM incoming")
        
        self.con.unregister("incoming")

    def query_df(self, sql: str, params: Optional[Dict[str,Any]]=None) -> pd.DataFrame:
        return self.con.execute(sql, params or {}).df()

    def exec(self, sql: str, params: Optional[Dict[str,Any]]=None):
        self.con.execute(sql, params or {})
