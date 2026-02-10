"""
Tests for CBD Ingest Logic
==========================
Unit tests for transformation functions and game processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytest

import sys
from pathlib import Path

# Add parent to path for imports
cbd_pbp_dir = Path(__file__).resolve().parents[1]
ml_model_dir = cbd_pbp_dir.parent
sys.path.insert(0, str(ml_model_dir))

from cbd_pbp.ingest import _sanitize_plays_df, GameResult


class TestSanitizePlaysDF:
    """Tests for the _sanitize_plays_df transformation function."""
    
    def test_converts_datetime_wallclock_to_unix(self):
        """Wallclock datetime should be converted to Unix timestamp (seconds)."""
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "id": [1],
            "wallclock": [dt],
            "playText": ["Made shot"]
        })
        
        result = _sanitize_plays_df(df)
        
        assert result["wallclock"].iloc[0] == int(dt.timestamp())
        assert result["wallclock"].dtype == "Int64"
    
    def test_handles_null_wallclock(self):
        """NaT/None values should remain None (nullable Int64)."""
        df = pd.DataFrame({
            "id": [1, 2],
            "wallclock": [pd.NaT, None],
            "playText": ["Made shot", "Missed shot"]
        })
        
        result = _sanitize_plays_df(df)
        
        assert pd.isna(result["wallclock"].iloc[0])
        assert pd.isna(result["wallclock"].iloc[1])
        assert result["wallclock"].dtype == "Int64"
    
    def test_handles_mixed_wallclock_values(self):
        """Mix of valid datetimes and NaT should work correctly."""
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "wallclock": [dt, pd.NaT, dt],
            "playText": ["A", "B", "C"]
        })
        
        result = _sanitize_plays_df(df)
        
        assert result["wallclock"].iloc[0] == int(dt.timestamp())
        assert pd.isna(result["wallclock"].iloc[1])
        assert result["wallclock"].iloc[2] == int(dt.timestamp())
    
    def test_empty_dataframe_returns_empty(self):
        """Empty DataFrame should return empty DataFrame."""
        df = pd.DataFrame()
        result = _sanitize_plays_df(df)
        assert result.empty
    
    def test_missing_wallclock_column_unchanged(self):
        """DataFrame without wallclock column should pass through unchanged."""
        df = pd.DataFrame({
            "id": [1, 2],
            "playText": ["Made shot", "Missed shot"]
        })
        
        result = _sanitize_plays_df(df)
        
        assert "wallclock" not in result.columns
        assert list(result.columns) == ["id", "playText"]


class TestGameResult:
    """Tests for the GameResult dataclass structure."""
    
    def test_game_result_initialization(self):
        """GameResult should initialize with empty lists."""
        result = GameResult(game_id="123", lineups=[], subs=[], plays=[], errors=[])
        
        assert result.game_id == "123"
        assert result.lineups == []
        assert result.subs == []
        assert result.plays == []
        assert result.errors == []
    
    def test_game_result_with_data(self):
        """GameResult should hold data correctly."""
        result = GameResult(
            game_id="456",
            lineups=[{"teamId": 1}],
            subs=[{"playerId": 10}],
            plays=[{"playText": "Made shot"}],
            errors=["subs: timeout"]
        )
        
        assert len(result.lineups) == 1
        assert len(result.subs) == 1
        assert len(result.plays) == 1
        assert len(result.errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
