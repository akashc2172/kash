#!/usr/bin/env python3
"""
Build site season/career exports directly from the ml-model CBB warehouse.

This script is READ-ONLY against:
  ml model/data/warehouse.duckdb

It writes to:
  new-trank/exports/*.csv

No writes are made to ml model/.
"""

from __future__ import annotations

import pathlib
import duckdb


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WAREHOUSE = REPO_ROOT / "ml model" / "data" / "warehouse.duckdb"
OUT_DIR = REPO_ROOT / "new-trank" / "exports"


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE), read_only=True)

    # Team ranking by adjusted net rating for opponent quality splits.
    con.execute(
        """
        create temp table team_rank as
        select
          season,
          teamId,
          rank() over (
            partition by season
            order by netRating desc nulls last
          ) as opp_rank
        from fact_ratings_adjusted
        """
    )

    # Player-game rows enriched with season + opponent rank.
    con.execute(
        """
        create temp table game_rows as
        with base as (
          select
            cast(pg.gameId as int) as game_id,
            pg.athleteId as athlete_id,
            pg.teamId as team_id,
            dg.season,
            case
              when pg.teamId = dg.homeTeamId then dg.awayTeamId
              else dg.homeTeamId
            end as opp_team_id,
            pg.fga,
            pg.fgm,
            pg.pts,
            pg.assisted_att,
            pg.seconds_on,
            pg.on_ortg,
            pg.on_drtg,
            pg.on_net_rating
          from fact_player_game pg
          join dim_games dg on cast(pg.gameId as int) = dg.id
          where dg.seasonType = 'regular'
        )
        select
          b.*,
          tr.opp_rank
        from base b
        left join team_rank tr
          on tr.season = b.season and tr.teamId = b.opp_team_id
        """
    )

    # Split rows (ALL / VS_TOP50 / VS_TOP100) for games.
    con.execute(
        """
        create temp table game_rows_split as
        select *, 'ALL' as split_id from game_rows
        union all
        select *, 'VS_TOP50' as split_id from game_rows where opp_rank <= 50
        union all
        select *, 'VS_TOP100' as split_id from game_rows where opp_rank <= 100
        """
    )

    # Aggregate game-level signals by split.
    con.execute(
        """
        create temp table game_agg as
        select
          season,
          athlete_id,
          split_id,
          count(distinct game_id) as g,
          sum(seconds_on) / 60.0 as minutes_total,
          sum(fga) as fga_total,
          sum(fgm) as fgm_total,
          sum(pts) as pts_total,
          sum(assisted_att) as assisted_att_total,
          sum(seconds_on * on_ortg) / nullif(sum(seconds_on), 0) as on_ortg,
          sum(seconds_on * on_drtg) / nullif(sum(seconds_on), 0) as on_drtg,
          sum(seconds_on * on_net_rating) / nullif(sum(seconds_on), 0) as on_net_rating
        from game_rows_split
        group by 1,2,3
        """
    )

    # Shot bucket aggregation by split.
    con.execute(
        """
        create temp table shots_agg as
        with shot_rows as (
          select
            cast(s.gameId as int) as game_id,
            s.athleteId as athlete_id,
            s.teamId as team_id,
            grs.season,
            grs.split_id,
            s.range_bucket,
            s.att,
            s.made,
            s.assisted_att
          from fact_player_game_shots_bucketed s
          join game_rows_split grs
            on cast(s.gameId as int) = grs.game_id
           and s.athleteId = grs.athlete_id
           and s.teamId = grs.team_id
        )
        select
          season,
          athlete_id,
          split_id,
          sum(case when range_bucket = 'rim' then att else 0 end) as rim_attempts,
          sum(case when range_bucket = 'rim' then made else 0 end) as rim_m,
          sum(case when range_bucket = 'mid' then att else 0 end) as middy_attempts,
          sum(case when range_bucket = 'mid' then made else 0 end) as middy_m,
          sum(case when range_bucket = 'three' then att else 0 end) as three_a,
          sum(case when range_bucket = 'three' then made else 0 end) as three_m,
          sum(case when range_bucket = 'ft' then att else 0 end) as fta,
          sum(case when range_bucket = 'ft' then made else 0 end) as ftm,
          sum(case when range_bucket = 'rim' then assisted_att else 0 end) as assisted_rim_att,
          sum(case when range_bucket = 'mid' then assisted_att else 0 end) as assisted_mid_att,
          sum(case when range_bucket = 'three' then assisted_att else 0 end) as assisted_three_att
        from shot_rows
        group by 1,2,3
        """
    )

    # Draft picks (if available).
    con.execute(
        """
        create temp table draft_min as
        select
          athleteId as athlete_id,
          min(pick) as pick
        from fact_draft_picks
        group by 1
        """
    )

    # Base player-season metadata from season stats.
    con.execute(
        """
        create temp table player_meta as
        select
          season,
          athleteId as athlete_id,
          name as key,
          team,
          conference as conf,
          position as pos,
          games as season_g,
          minutes as season_minutes,
          points as points_total,
          assists as ast_total,
          turnovers as tov_total,
          steals as stl_total,
          blocks as blk_total,
          usage as usg,
          PORPAG as porpag,
          offensiveRating as ortg,
          defensiveRating as drtg,
          trueShootingPct as ts,
          effectiveFieldGoalPct as efg,
          freeThrowRate as ftr,
          assistsTurnoverRatio as ast_to,
          offensiveReboundPct as oreb_rate,
          fieldGoals.made as fgm,
          fieldGoals.attempted as fga,
          twoPointFieldGoals.made as two_m,
          twoPointFieldGoals.attempted as two_a,
          threePointFieldGoals.made as three_m_season,
          threePointFieldGoals.attempted as three_a_season,
          freeThrows.made as ftm_season,
          freeThrows.attempted as fta_season,
          rebounds.offensive as oreb,
          rebounds.defensive as dreb
        from fact_player_season_stats
        """
    )

    # Final season export with split rows.
    season_df = con.execute(
        """
        select
          pm.season as torvik_year,
          pm.athlete_id as torvik_id,
          pm.key,
          pm.team,
          pm.conf,
          pm.pos,
          coalesce(ga.g, pm.season_g) as g,
          coalesce(ga.minutes_total, pm.season_minutes) as minutes_total,
          case
            when coalesce(ga.g, 0) > 0 then coalesce(ga.minutes_total, 0) / ga.g
            when pm.season_g > 0 then pm.season_minutes / pm.season_g
            else null
          end as mpg,
          dm.pick,
          case when dm.pick is null then 0 else 1 end as drafted,
          cast(null as double) as recruit_rank,
          cast(null as double) as height,
          cast(null as double) as weight,
          cast(null as double) as bmi,
          pm.porpag,
          cast(null as double) as dporpag,
          cast(null as double) as obpm,
          cast(null as double) as dbpm,
          cast(null as double) as gbpm,
          pm.ts,
          pm.efg,
          pm.ftr,
          cast(coalesce(sa.fgm_total, pm.fgm) as double) / nullif(coalesce(sa.fga_total, pm.fga), 0) as fg_pct,
          cast(coalesce(sa.ftm, pm.ftm_season) as double) / nullif(coalesce(sa.fta, pm.fta_season), 0) as ft_pct,
          pm.ortg,
          pm.drtg,
          cast(null as double) as adj_oe,
          cast(null as double) as adj_de,
          cast(null as double) as per,
          pm.usg,
          pm.ast_to,
          cast(pm.ast_total as double) / nullif(coalesce(ga.g, pm.season_g), 0) as ast,
          cast(pm.tov_total as double) / nullif(coalesce(ga.g, pm.season_g), 0) as tov,
          cast(pm.stl_total as double) / nullif(coalesce(ga.g, pm.season_g), 0) as stl,
          cast(pm.blk_total as double) / nullif(coalesce(ga.g, pm.season_g), 0) as blk,
          pm.oreb_rate,
          cast(pm.dreb as double) / nullif(pm.dreb + pm.oreb, 0) as dreb_rate,
          coalesce(sa.fga_total, pm.fga) as fga,
          coalesce(sa.fgm_total, pm.fgm) as fgm,
          coalesce(sa.two_a, pm.two_a) as two_a,
          coalesce(sa.two_m, pm.two_m) as two_m,
          coalesce(sa.three_a, pm.three_a_season) as three_a,
          coalesce(sa.three_m, pm.three_m_season) as three_m,
          coalesce(sa.fta, pm.fta_season) as fta,
          coalesce(sa.ftm, pm.ftm_season) as ftm,
          cast(coalesce(sa.two_m, pm.two_m) as double) / nullif(coalesce(sa.two_a, pm.two_a), 0) as "2p%",
          cast(coalesce(sa.three_m, pm.three_m_season) as double) / nullif(coalesce(sa.three_a, pm.three_a_season), 0) as "3p%",
          cast(coalesce(sa.rim_m, 0) as double) / nullif(coalesce(sa.rim_attempts, 0), 0) as "rim%",
          cast(coalesce(sa.middy_m, 0) as double) / nullif(coalesce(sa.middy_attempts, 0), 0) as "middy fg%",
          coalesce(sa.rim_attempts, 0) as "rim attempts",
          coalesce(sa.rim_m, 0) as rim_m,
          coalesce(sa.middy_attempts, 0) as "middy attempts",
          coalesce(sa.middy_m, 0) as middy_m,
          cast(coalesce(sa.three_a, pm.three_a_season) as double) / nullif(coalesce(sa.fga_total, pm.fga), 0) as "3pr",
          cast(coalesce(sa.two_a, pm.two_a) as double) / nullif(coalesce(sa.fga_total, pm.fga), 0) as "2pr",
          cast(coalesce(sa.middy_attempts, 0) as double) / nullif(coalesce(sa.fga_total, pm.fga), 0) as "midr",
          cast(coalesce(sa.rim_attempts, 0) as double) / nullif(coalesce(sa.fga_total, pm.fga), 0) as "rimr",
          cast(coalesce(sa.assisted_rim_att, 0) as double) / nullif(coalesce(sa.rim_attempts, 0), 0) as "assisted rim fg%",
          cast(coalesce(sa.assisted_mid_att, 0) as double) / nullif(coalesce(sa.middy_attempts, 0), 0) as "assisted middy fg%",
          cast(coalesce(sa.assisted_three_att, 0) as double) / nullif(coalesce(sa.three_a, pm.three_a_season), 0) as "assisted 3p %",
          cast(coalesce(sa.assisted_rim_att, 0) as double) / nullif(coalesce(sa.assisted_rim_att, 0) + coalesce(sa.assisted_mid_att, 0) + coalesce(sa.assisted_three_att, 0), 0) as "% of assists end w/ rim",
          cast(coalesce(sa.assisted_mid_att, 0) as double) / nullif(coalesce(sa.assisted_rim_att, 0) + coalesce(sa.assisted_mid_att, 0) + coalesce(sa.assisted_three_att, 0), 0) as "% of assists end w/ middy",
          cast(coalesce(sa.assisted_three_att, 0) as double) / nullif(coalesce(sa.assisted_rim_att, 0) + coalesce(sa.assisted_mid_att, 0) + coalesce(sa.assisted_three_att, 0), 0) as "% of assists end w/ 3p",
          ga.on_ortg as on_ortg,
          ga.on_drtg as on_drtg,
          ga.on_net_rating as "total RAPM",
          ga.on_ortg as "offensive rapm",
          ga.on_drtg as "defensive rapm",
          cast(coalesce(sa.rim_attempts, 0) as double) / nullif(coalesce(ga.minutes_total, pm.season_minutes), 0) * 40 as "rimfga/100",
          cast(coalesce(sa.middy_attempts, 0) as double) / nullif(coalesce(ga.minutes_total, pm.season_minutes), 0) * 40 as "midfga/100",
          cast(coalesce(sa.three_a, pm.three_a_season) as double) / nullif(coalesce(ga.minutes_total, pm.season_minutes), 0) * 40 as "3pa/100",
          cast(coalesce(sa.two_a, pm.two_a) as double) / nullif(coalesce(ga.minutes_total, pm.season_minutes), 0) * 40 as "2pa/100",
          cast(coalesce(ga.on_net_rating, 0) as double) as "g-score",
          cast(null as varchar) as exp,
          coalesce(ga.split_id, 'ALL') as split_id,
          'NCAA D1' as league
        from player_meta pm
        left join game_agg ga
          on ga.season = pm.season and ga.athlete_id = pm.athlete_id
        left join (
          select
            season,
            athlete_id,
            split_id,
            sum(case when range_bucket = 'rim' then att else 0 end) as rim_attempts,
            sum(case when range_bucket = 'rim' then made else 0 end) as rim_m,
            sum(case when range_bucket = 'mid' then att else 0 end) as middy_attempts,
            sum(case when range_bucket = 'mid' then made else 0 end) as middy_m,
            sum(case when range_bucket = 'three' then att else 0 end) as three_a,
            sum(case when range_bucket = 'three' then made else 0 end) as three_m,
            sum(case when range_bucket = 'ft' then att else 0 end) as fta,
            sum(case when range_bucket = 'ft' then made else 0 end) as ftm,
            sum(case when range_bucket = 'rim' then assisted_att else 0 end) as assisted_rim_att,
            sum(case when range_bucket = 'mid' then assisted_att else 0 end) as assisted_mid_att,
            sum(case when range_bucket = 'three' then assisted_att else 0 end) as assisted_three_att,
            sum(case when range_bucket in ('rim','mid') then att else 0 end) as two_a,
            sum(case when range_bucket in ('rim','mid') then made else 0 end) as two_m,
            sum(case when range_bucket in ('rim','mid','three') then att else 0 end) as fga_total,
            sum(case when range_bucket in ('rim','mid','three') then made else 0 end) as fgm_total
          from (
            select
              season, athlete_id, split_id, range_bucket, att, made, assisted_att
            from (
              select
                cast(s.gameId as int) as game_id,
                s.athleteId as athlete_id,
                grs.season,
                grs.split_id,
                s.range_bucket,
                s.att,
                s.made,
                s.assisted_att
              from fact_player_game_shots_bucketed s
              join game_rows_split grs
                on cast(s.gameId as int) = grs.game_id
               and s.athleteId = grs.athlete_id
               and s.teamId = grs.team_id
            ) z
          ) y
          group by 1,2,3
        ) sa
          on sa.season = pm.season and sa.athlete_id = pm.athlete_id and sa.split_id = ga.split_id
        left join draft_min dm
          on dm.athlete_id = pm.athlete_id
        where pm.season >= 2010
        """
    ).fetchdf()

    # Clean a couple columns names for app compatibility.
    season_df.rename(columns={"recruit_rank": "recruit rank"}, inplace=True)

    season_path = OUT_DIR / "season.csv"
    season_df.to_csv(season_path, index=False)

    career_df = con.execute(
        """
        with s as (
          select * from read_csv_auto(?)
          where split_id = 'ALL'
        )
        select
          key,
          any_value(team) as team,
          any_value(pos) as pos,
          min(torvik_year) as first_year,
          max(torvik_year) as last_year,
          count(*) as length,
          sum(g) as g,
          sum(minutes_total) / nullif(sum(g), 0) as mpg,
          avg("total RAPM") as "total RAPM",
          avg("offensive rapm") as "offensive rapm",
          avg("defensive rapm") as "defensive rapm",
          avg(porpag) as porpag,
          avg(ortg) as ortg,
          avg(drtg) as drtg,
          avg(ts) as ts,
          avg(efg) as efg,
          avg("2p%") as "2p%",
          avg("3p%") as "3p%",
          avg("rim%") as "rim%",
          avg("middy fg%") as "middy fg%",
          avg("3pr") as "3pr",
          avg("2pr") as "2pr",
          avg("midr") as "midr",
          avg("rimr") as "rimr",
          avg(ast) as ast,
          avg(tov) as tov,
          avg(stl) as stl,
          avg(blk) as blk,
          avg(oreb_rate) as oreb_rate,
          avg(dreb_rate) as dreb_rate,
          min(pick) as pick,
          max(drafted) as drafted,
          'NCAA D1' as league
        from s
        group by 1
        """,
        [str(season_path)],
    ).fetchdf()

    career_path = OUT_DIR / "career.csv"
    career_df.to_csv(career_path, index=False)

    # Keep these passthroughs for the app contract.
    for name in ("weights.csv", "nba_lookup.csv", "br_advanced_stats.csv"):
        src = REPO_ROOT / "my-trank" / "public" / "data" / name
        if src.exists():
            (OUT_DIR / name).write_bytes(src.read_bytes())

    con.close()
    print(f"[ok] wrote {season_path}")
    print(f"[ok] wrote {career_path}")


if __name__ == "__main__":
    build()
