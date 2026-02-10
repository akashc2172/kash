from __future__ import annotations
import duckdb
import pandas as pd
from typing import Optional, Dict, Any, List, Set
import re


class Warehouse:
    """
    DuckDB warehouse wrapper with schema caching for performance.
    """
    
    def __init__(self, path: str):
        self.path = path
        self.con = duckdb.connect(path)
        # Schema cache to avoid redundant information_schema queries
        self._known_tables: Set[str] = set()
        self._known_columns: Dict[str, Set[str]] = {}
        self._load_schema_cache()

    def _load_schema_cache(self):
        """Populate cache from existing schema on startup."""
        try:
            tables = self.con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            self._known_tables = {t[0] for t in tables}
            
            for table in self._known_tables:
                cols = self.con.execute(
                    f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
                ).fetchall()
                self._known_columns[table] = {c[0] for c in cols}
        except Exception:
            # Fresh DB, no tables yet
            pass

    def init_schema(self, ddl_dict: Dict[str, str]):
        """
        Initialize tables from explicit DDL definitions.
        
        Args:
            ddl_dict: Mapping of table_name -> CREATE TABLE statement
        """
        for table_name, ddl in ddl_dict.items():
            if table_name not in self._known_tables:
                self.con.execute(ddl)
                self._known_tables.add(table_name)
                # Refresh column cache for new table
                cols = self.con.execute(
                    f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
                ).fetchall()
                self._known_columns[table_name] = {c[0] for c in cols}

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def _get_columns(self, table_name: str) -> Set[str]:
        """Get columns for a table, using cache when possible."""
        if table_name in self._known_columns:
            return self._known_columns[table_name]
        
        # Cache miss - query and cache
        cols = self.con.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
        ).fetchall()
        self._known_columns[table_name] = {c[0] for c in cols}
        return self._known_columns[table_name]

    def ensure_table(self, name: str, df: pd.DataFrame, pk: Optional[List[str]] = None):
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
        
        # Register the DataFrame
        self.con.register("incoming", df)
        
        # Check cache first (fast path)
        table_exists = name in self._known_tables
        
        if not table_exists:
            # Fallback to DB check (handles race conditions)
            table_exists = self.con.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}'"
            ).fetchone()[0] > 0
            if table_exists:
                self._known_tables.add(name)
        
        if not table_exists:
            # Create table from DataFrame (first time)
            self.con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming LIMIT 0")
            self.con.execute(f"INSERT INTO {name} SELECT * FROM incoming")
            self._known_tables.add(name)
            self._known_columns[name] = set(cols)
        else:
            # Table exists - get its columns (cached)
            existing_cols = self._get_columns(name)
            
            # Find common columns (intersection)
            common_cols = [c for c in cols if c in existing_cols]
            
            if not common_cols:
                print(f"Warning: No common columns between incoming data and {name}")
                self.con.unregister("incoming")
                return
            
            def _insert_with_column_fallback(cols_to_insert: List[str]):
                """
                Insert using explicit columns. If one column has a type-mismatch conversion
                (common with nested map/struct drift), drop that column and retry.
                """
                active_cols = list(cols_to_insert)
                while active_cols:
                    col_list = ", ".join([f'"{c}"' for c in active_cols])
                    try:
                        self.con.execute(
                            f"INSERT INTO {name}({col_list}) SELECT {col_list} FROM incoming"
                        )
                        return active_cols
                    except Exception as e:
                        msg = str(e)
                        dropped = None
                        # Prefer explicit source-column hint from DuckDB error text.
                        m = re.search(r"source column\s+([A-Za-z0-9_]+)", msg)
                        if m:
                            col = m.group(1)
                            if col in active_cols:
                                dropped = col
                        if dropped is None:
                            raise
                        active_cols.remove(dropped)
                        print(f"Warning: dropped column '{dropped}' while inserting into {name} due to conversion mismatch.")
                raise RuntimeError(f"Failed to insert any compatible columns into {name}.")

            if pk:
                # Upsert: delete matching keys, then insert
                cond = " AND ".join([f'{name}."{c}" = incoming."{c}"' for c in pk if c in existing_cols])
                if cond:
                    self.con.execute(f"DELETE FROM {name} USING incoming WHERE {cond}")
                _insert_with_column_fallback(common_cols)
            else:
                # Append only - use explicit column list with fallback
                _insert_with_column_fallback(common_cols)
        
        self.con.unregister("incoming")

    def query_df(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        return self.con.execute(sql, params or {}).df()

    def exec(self, sql: str, params: Optional[Dict[str, Any]] = None):
        self.con.execute(sql, params or {})
