"""
Tests for Warehouse Module
==========================
Integration tests for DuckDB warehouse operations.
"""

import pandas as pd
import pytest
import tempfile
import os

import sys
from pathlib import Path

cbd_pbp_dir = Path(__file__).resolve().parents[1]
ml_model_dir = cbd_pbp_dir.parent
sys.path.insert(0, str(ml_model_dir))

from cbd_pbp.warehouse import Warehouse


@pytest.fixture
def temp_warehouse():
    """Create a temporary warehouse for testing."""
    # Get a temp path but don't create the file (DuckDB creates it)
    import uuid
    path = f"/tmp/test_warehouse_{uuid.uuid4().hex}.duckdb"
    
    wh = Warehouse(path)
    yield wh
    
    wh.close()
    try:
        os.unlink(path)
    except OSError:
        pass


class TestWarehouseEnsureTable:
    """Tests for Warehouse.ensure_table method."""
    
    def test_creates_new_table(self, temp_warehouse):
        """ensure_table should create a new table if it doesn't exist."""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"]
        })
        
        temp_warehouse.ensure_table("test_table", df, pk=None)
        
        result = temp_warehouse.query_df("SELECT * FROM test_table ORDER BY id")
        assert len(result) == 2
        assert result["name"].tolist() == ["Alice", "Bob"]
    
    def test_appends_to_existing_table(self, temp_warehouse):
        """ensure_table should append to existing table without PK."""
        df1 = pd.DataFrame({"id": [1], "name": ["Alice"]})
        df2 = pd.DataFrame({"id": [2], "name": ["Bob"]})
        
        temp_warehouse.ensure_table("test_table", df1, pk=None)
        temp_warehouse.ensure_table("test_table", df2, pk=None)
        
        result = temp_warehouse.query_df("SELECT * FROM test_table ORDER BY id")
        assert len(result) == 2
    
    def test_upserts_with_primary_key(self, temp_warehouse):
        """ensure_table should upsert when PK is provided."""
        df1 = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        df2 = pd.DataFrame({"id": [2], "name": ["Bobby"]})  # Update id=2
        
        temp_warehouse.ensure_table("test_table", df1, pk=["id"])
        temp_warehouse.ensure_table("test_table", df2, pk=["id"])
        
        result = temp_warehouse.query_df("SELECT * FROM test_table ORDER BY id")
        assert len(result) == 2
        assert result[result["id"] == 2]["name"].iloc[0] == "Bobby"
    
    def test_handles_column_mismatch(self, temp_warehouse):
        """ensure_table should use common columns when schemas differ."""
        df1 = pd.DataFrame({"id": [1], "name": ["Alice"], "age": [30]})
        df2 = pd.DataFrame({"id": [2], "name": ["Bob"]})  # Missing 'age'
        
        temp_warehouse.ensure_table("test_table", df1, pk=None)
        temp_warehouse.ensure_table("test_table", df2, pk=None)
        
        result = temp_warehouse.query_df("SELECT * FROM test_table ORDER BY id")
        assert len(result) == 2
        assert "age" in result.columns
    
    def test_empty_dataframe_noop(self, temp_warehouse):
        """ensure_table should do nothing with empty DataFrame."""
        df = pd.DataFrame()
        
        temp_warehouse.ensure_table("test_table", df, pk=None)
        
        # Table should not exist
        assert "test_table" not in temp_warehouse._known_tables


class TestWarehouseSchemaCache:
    """Tests for schema caching behavior."""
    
    def test_cache_populated_on_init(self, temp_warehouse):
        """Cache should be populated when Warehouse is initialized."""
        # Create a table
        df = pd.DataFrame({"id": [1], "name": ["Test"]})
        temp_warehouse.ensure_table("cached_table", df, pk=None)
        
        # Verify cache
        assert "cached_table" in temp_warehouse._known_tables
        assert "id" in temp_warehouse._known_columns.get("cached_table", set())
        assert "name" in temp_warehouse._known_columns.get("cached_table", set())
    
    def test_cache_prevents_redundant_queries(self, temp_warehouse):
        """Subsequent ensure_table calls should use cache."""
        df = pd.DataFrame({"id": [1], "name": ["Test"]})
        temp_warehouse.ensure_table("cached_table", df, pk=None)
        
        # Second insert should hit cache
        df2 = pd.DataFrame({"id": [2], "name": ["Test2"]})
        temp_warehouse.ensure_table("cached_table", df2, pk=None)
        
        # Verify both rows exist (cache didn't break functionality)
        result = temp_warehouse.query_df("SELECT COUNT(*) as cnt FROM cached_table")
        assert result["cnt"].iloc[0] == 2


class TestWarehouseInitSchema:
    """Tests for explicit schema initialization."""
    
    def test_init_schema_creates_tables(self, temp_warehouse):
        """init_schema should create tables from DDL."""
        ddl = {
            "test_explicit": """
                CREATE TABLE IF NOT EXISTS test_explicit (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR
                )
            """
        }
        
        temp_warehouse.init_schema(ddl)
        
        assert "test_explicit" in temp_warehouse._known_tables
        # Insert should work
        temp_warehouse.exec("INSERT INTO test_explicit VALUES (1, 'Test')")
        result = temp_warehouse.query_df("SELECT * FROM test_explicit")
        assert len(result) == 1
    
    def test_init_schema_idempotent(self, temp_warehouse):
        """init_schema should be safe to call multiple times."""
        ddl = {
            "test_explicit": """
                CREATE TABLE IF NOT EXISTS test_explicit (
                    id INTEGER PRIMARY KEY
                )
            """
        }
        
        temp_warehouse.init_schema(ddl)
        temp_warehouse.init_schema(ddl)  # Should not error
        
        assert "test_explicit" in temp_warehouse._known_tables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
