"""
Pytest Configuration for CBD Tests
===================================
Shared fixtures for testing.
"""

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
def memory_warehouse():
    """In-memory DuckDB warehouse for fast tests."""
    wh = Warehouse(":memory:")
    yield wh
    wh.close()


@pytest.fixture
def temp_db_path():
    """Temporary file path for DuckDB."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass
