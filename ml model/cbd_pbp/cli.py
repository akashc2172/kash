from __future__ import annotations
import typer
from typing import List
from .warehouse import Warehouse
from .ingest import ingest_static, ingest_play_types, ingest_games_only, fetch_and_ingest_season
from .ingest import ingest_player_season_stats
from .ingest import ingest_games_only_endpoints, load_game_ids_from_file
from .derive import build_derived_sql
from .windows import build_windows_player, build_windows_team
from .export import export_player_asof_wide
from .bridges import build_scrape_bridges

app = typer.Typer(add_completion=False)

@app.command()
def ingest_season_cmd(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason/tournament depending on API"),
    include_lineups: bool = typer.Option(True, help="Fetch /lineups/game/{gameId} for missing games"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    wh = Warehouse(out)
    ingest_static(wh)
    ingest_play_types(wh)
    fetch_and_ingest_season(wh, season=season, season_type=season_type, include_lineups=include_lineups)
    wh.close()
    typer.echo("OK: ingested season")

@app.command()
def resume_ingest(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason/tournament depending on API"),
    include_lineups: bool = typer.Option(False, help="Enable lineup backfill (can be expensive on older seasons)"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """Resume per-game ingest (skips static dims and bulk tables)."""
    wh = Warehouse(out)
    ingest_games_only(wh, season=season, season_type=season_type, include_lineups=include_lineups)
    wh.close()
    typer.echo("OK: resumed ingest")


@app.command()
def resume_ingest_endpoints(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason/tournament depending on API"),
    endpoints: str = typer.Option("plays,subs,lineups", help="Comma-separated: plays,subs,lineups"),
    only_game_ids_file: str = typer.Option(None, help="Optional path to game IDs to include"),
    skip_game_ids_file: str = typer.Option(None, help="Optional path to game IDs to exclude"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """
    Resume per-game ingest with endpoint-level controls.
    Use this for strict staged backfills (plays -> subs -> lineups).
    """
    wanted = {x.strip().lower() for x in endpoints.split(",") if x.strip()}
    allowed = {"plays", "subs", "lineups"}
    invalid = sorted(wanted - allowed)
    if invalid:
        raise typer.BadParameter(f"Invalid endpoint(s): {invalid}. Allowed: plays,subs,lineups")

    only_ids = load_game_ids_from_file(only_game_ids_file)
    skip_ids = load_game_ids_from_file(skip_game_ids_file)

    wh = Warehouse(out)
    ingest_games_only_endpoints(
        wh,
        season=season,
        season_type=season_type,
        include_plays="plays" in wanted,
        include_subs="subs" in wanted,
        include_lineups="lineups" in wanted,
        only_game_ids=only_ids,
        skip_game_ids=skip_ids,
    )
    wh.close()
    typer.echo("OK: resumed endpoint-scoped ingest")

@app.command()
def fetch_ingest(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason/tournament"),
    include_lineups: bool = typer.Option(False, help="Enable lineup backfill"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """
    Fetch games for a season/type and ingest per-game data.
    Unlike ingest-season-cmd, this does NOT re-ingest static dimension tables,
    avoiding schema mismatch issues. Use this for adding new seasons/postseasons.
    """
    wh = Warehouse(out)
    fetch_and_ingest_season(wh, season=season, season_type=season_type, include_lineups=include_lineups)
    wh.close()
    typer.echo("OK: fetched games and ingested per-game data")


@app.command()
def ingest_player_season(
    season: int = typer.Option(...),
    season_type: str = typer.Option("regular", help="regular/postseason"),
    team: str = typer.Option(None, help="Optional team filter"),
    conference: str = typer.Option(None, help="Optional conference filter"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """Ingest player season stats with explicit seasonType + normalized table output."""
    wh = Warehouse(out)
    ingest_player_season_stats(
        wh,
        season=season,
        season_type=season_type,
        team=team,
        conference=conference,
    )
    wh.close()
    typer.echo("OK: ingested player season stats")


@app.command()
def build_bridges(
    scrape_root: str = typer.Option("data/manual_scrapes", help="Root folder for manual NCAA scrape CSVs"),
    max_files: int = typer.Option(None, help="Optional limit for debug runs"),
    out: str = typer.Option("data/warehouse.duckdb"),
):
    """Build persistent game/player bridge tables linking CBD ids to manual scrape ids/names."""
    wh = Warehouse(out)
    build_scrape_bridges(wh, scrape_root=scrape_root, max_files=max_files)
    wh.close()
    typer.echo("OK: built scrape bridges")

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
