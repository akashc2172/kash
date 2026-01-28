from __future__ import annotations
import typer
from typing import List
from .warehouse import Warehouse
from .ingest import ingest_static, ingest_season, ingest_play_types, ingest_games_only
from .derive import build_derived_sql
from .windows import build_windows_player, build_windows_team
from .export import export_player_asof_wide

app = typer.Typer(add_completion=False)

@app.command()
def ingest_season_cmd(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason/tournament depending on API"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    wh = Warehouse(out)
    ingest_static(wh)
    ingest_play_types(wh)
    ingest_season(wh, season=season, season_type=season_type)
    wh.close()
    typer.echo("OK: ingested season")

@app.command()
def resume_ingest(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason/tournament depending on API"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """Resume per-game ingest (skips static dims and bulk tables)."""
    wh = Warehouse(out)
    ingest_games_only(wh, season=season, season_type=season_type)
    wh.close()
    typer.echo("OK: resumed ingest")

@app.command()
def build_derived(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """Build derived tables using SQL Staging Layer."""
    wh = Warehouse(out)
    build_derived_sql(wh)
    wh.close()
    typer.echo("OK: derived tables built via SQL")

@app.command()
def build_windows(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular"),
    out: str = typer.Option("data/warehouse.duckdb"),
    windows: str = typer.Option("season_to_date,rolling10"),
):
    wh = Warehouse(out)
    window_ids = [w.strip() for w in windows.split(",") if w.strip()]
    build_windows_player(wh, season=season, season_type=season_type, window_ids=window_ids)
    build_windows_team(wh, season=season, season_type=season_type, window_ids=window_ids)
    wh.close()
    typer.echo("OK: windows built")

@app.command()
def export_wide(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular"),
    out: str = typer.Option("data/warehouse.duckdb"),
    window_ids: str = typer.Option("season_to_date,rolling10"),
    dest: str = typer.Option("data/player_asof_wide.parquet"),
):
    wh = Warehouse(out)
    wids = [w.strip() for w in window_ids.split(",") if w.strip()]
    export_player_asof_wide(wh, season=season, season_type=season_type, window_ids=wids, dest=dest)
    wh.close()
    typer.echo(f"OK: exported {dest}")

def main():
    app()

if __name__ == "__main__":
    main()
