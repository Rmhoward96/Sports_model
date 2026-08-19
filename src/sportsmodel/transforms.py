"""Raw Statcast -> per-PA rate profiles (docs/methodology.md Part A, step 1 inputs).

Reads the partitioned Statcast Parquet, classifies each plate appearance into one of
the 7 terminal outcomes, and aggregates to batter/pitcher per-PA rates split by
opposing handedness, for both 'career' (all seasons) and 'season' (latest year)
windows. Also computes league baseline rates used by the odds-ratio blend.

For MLB the MLBAM player id is our canonical player_id, so no crosswalk is needed
within this pipeline (crosswalks come later, when external projections are joined).
"""
from __future__ import annotations

from . import config

# events (terminal PA outcome) -> one of the 7 model outcomes. HBP grouped with walks;
# reached-on-error / fielder's choice / sac count as outs (not hits).
_OUTCOME_CASE = """
CASE
  WHEN events IN ('walk','intent_walk','hit_by_pitch') THEN 'bb'
  WHEN events IN ('strikeout','strikeout_double_play')  THEN 'k'
  WHEN events = 'single'    THEN '1b'
  WHEN events = 'double'    THEN '2b'
  WHEN events = 'triple'    THEN '3b'
  WHEN events = 'home_run'  THEN 'hr'
  ELSE 'out'
END
"""

# statcast files land here (partitioned season=YYYY/); recursive glob picks all up.
_PARQUET_GLOB = str(config.RAW_DIR / "statcast" / "**" / "*.parquet")


def _rate_cols() -> str:
    """SQL fragment: outcome-rate columns computed from a grouped set of PAs."""
    outs = ["bb", "k", "1b", "2b", "3b", "hr", "out"]
    lines = [f"SUM(CASE WHEN outcome='{o}' THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) AS p_{o}"
             for o in outs]
    lines.append(
        "SUM(CASE WHEN outcome IN ('1b','2b','3b','hr') THEN 1 ELSE 0 END)::DOUBLE "
        "/ COUNT(*) AS p_hit"
    )
    lines.append("AVG(xwoba) AS xwoba")
    return ",\n           ".join(lines)


def _base_cte(id_col: str, opp_col: str) -> str:
    """One row per PA: player_id, opposing hand, year, outcome, xwoba."""
    return f"""
    WITH base AS (
        SELECT CAST({id_col} AS BIGINT)                            AS player_id,
               {opp_col}                                           AS opp_hand,
               CAST(SUBSTR(CAST(game_date AS VARCHAR), 1, 4) AS INT) AS yr,
               {_OUTCOME_CASE}                                     AS outcome,
               estimated_woba_using_speedangle                     AS xwoba
        FROM read_parquet('{_PARQUET_GLOB}')
        WHERE events IS NOT NULL
          AND {id_col} IS NOT NULL
          AND {opp_col} IN ('L','R')
    ),
    maxyr AS (SELECT max(yr) AS y FROM base)
    """


def _profile_query(id_col: str, opp_col: str) -> str:
    """career + season windows, each with per-hand (L/R) and ALL rollup rows."""
    rates = _rate_cols()
    grouping = (
        "GROUP BY GROUPING SETS ((player_id, opp_hand), (player_id))"
    )
    vs_hand = "CASE WHEN GROUPING(opp_hand)=1 THEN 'ALL' ELSE opp_hand END AS vs_hand"
    return f"""
    {_base_cte(id_col, opp_col)}
    SELECT player_id,
           {vs_hand},
           'career' AS window_name,
           COUNT(*) AS pa,
           {rates}
    FROM base
    {grouping}
    UNION ALL
    SELECT player_id,
           {vs_hand},
           'season' AS window_name,
           COUNT(*) AS pa,
           {rates}
    FROM base
    WHERE yr = (SELECT y FROM maxyr)
    {grouping}
    """


def build_batter_profiles(con) -> int:
    """feat_batter_profile: batter rates split by opposing PITCHER hand (p_throws)."""
    con.execute(
        f"CREATE OR REPLACE TABLE feat_batter_profile AS {_profile_query('batter', 'p_throws')}"
    )
    return con.execute("SELECT count(*) FROM feat_batter_profile").fetchone()[0]


def build_pitcher_profiles(con) -> int:
    """feat_pitcher_profile: pitcher rates-allowed split by opposing BATTER hand (stand)."""
    con.execute(
        f"CREATE OR REPLACE TABLE feat_pitcher_profile AS {_profile_query('pitcher', 'stand')}"
    )
    return con.execute("SELECT count(*) FROM feat_pitcher_profile").fetchone()[0]


def build_team_offense_profiles(con) -> int:
    """feat_team_offense: each team's offensive per-PA rates, by opposing pitcher hand.

    Batting team derived from inning half: Top => away bats, Bottom => home bats.
    """
    rates = _rate_cols()
    base = f"""
    WITH base AS (
        SELECT CASE WHEN inning_topbot='Top' THEN away_team ELSE home_team END AS team,
               p_throws AS opp_hand,
               CAST(SUBSTR(CAST(game_date AS VARCHAR), 1, 4) AS INT) AS yr,
               {_OUTCOME_CASE} AS outcome,
               estimated_woba_using_speedangle AS xwoba
        FROM read_parquet('{_PARQUET_GLOB}')
        WHERE events IS NOT NULL AND inning_topbot IS NOT NULL AND p_throws IN ('L','R')
    ),
    maxyr AS (SELECT max(yr) AS y FROM base)
    """
    vs_hand = "CASE WHEN GROUPING(opp_hand)=1 THEN 'ALL' ELSE opp_hand END AS vs_hand"
    grouping = "GROUP BY GROUPING SETS ((team, opp_hand), (team))"
    query = f"""
    {base}
    SELECT team, {vs_hand}, 'career' AS window_name, COUNT(*) AS pa, {rates}
    FROM base {grouping}
    UNION ALL
    SELECT team, {vs_hand}, 'season' AS window_name, COUNT(*) AS pa, {rates}
    FROM base WHERE yr = (SELECT y FROM maxyr) {grouping}
    """
    con.execute(f"CREATE OR REPLACE TABLE feat_team_offense AS {query}")
    return con.execute("SELECT count(*) FROM feat_team_offense").fetchone()[0]


def build_team_bullpen_profiles(con) -> int:
    """feat_team_bullpen: each team's relief run-prevention (innings >= 7 as a proxy).

    Pitching team = home when Top (away bats), away when Bottom. Late innings are
    reliever-dominated, so this approximates the team bullpen without an SP/RP label.
    """
    rates = _rate_cols()
    base = f"""
    WITH base AS (
        SELECT CASE WHEN inning_topbot='Top' THEN home_team ELSE away_team END AS team,
               stand AS opp_hand,
               CAST(SUBSTR(CAST(game_date AS VARCHAR), 1, 4) AS INT) AS yr,
               {_OUTCOME_CASE} AS outcome,
               estimated_woba_using_speedangle AS xwoba
        FROM read_parquet('{_PARQUET_GLOB}')
        WHERE events IS NOT NULL AND inning_topbot IS NOT NULL
          AND inning >= 7 AND stand IN ('L','R')
    ),
    maxyr AS (SELECT max(yr) AS y FROM base)
    """
    vs_hand = "CASE WHEN GROUPING(opp_hand)=1 THEN 'ALL' ELSE opp_hand END AS vs_hand"
    grouping = "GROUP BY GROUPING SETS ((team, opp_hand), (team))"
    query = f"""
    {base}
    SELECT team, {vs_hand}, 'career' AS window_name, COUNT(*) AS pa, {rates}
    FROM base {grouping}
    UNION ALL
    SELECT team, {vs_hand}, 'season' AS window_name, COUNT(*) AS pa, {rates}
    FROM base WHERE yr = (SELECT y FROM maxyr) {grouping}
    """
    con.execute(f"CREATE OR REPLACE TABLE feat_team_bullpen AS {query}")
    return con.execute("SELECT count(*) FROM feat_team_bullpen").fetchone()[0]


# Per-PA wOBA value of each outcome (2023-era weights; matches model/game.py).
_WOBA_CASE = """
CASE
  WHEN events IN ('walk','intent_walk','hit_by_pitch') THEN 0.69
  WHEN events = 'single'   THEN 0.89
  WHEN events = 'double'   THEN 1.27
  WHEN events = 'triple'   THEN 1.62
  WHEN events = 'home_run' THEN 2.10
  ELSE 0.0
END
"""


def build_park_factors(con) -> int:
    """park_factors(team, pf_runs): each home park's run environment.

    Standard home/road method — a team's wOBA in its own park vs the same team's
    wOBA on the road controls for team quality. Ratio clamped to a sane band.
    """
    query = f"""
    WITH pas AS (
        SELECT home_team, away_team, inning_topbot, {_WOBA_CASE} AS w
        FROM read_parquet('{_PARQUET_GLOB}')
        WHERE events IS NOT NULL AND inning_topbot IS NOT NULL
    ),
    home AS (SELECT home_team AS team, avg(w) AS hw FROM pas WHERE inning_topbot='Bot' GROUP BY home_team),
    road AS (SELECT away_team AS team, avg(w) AS rw FROM pas WHERE inning_topbot='Top' GROUP BY away_team)
    SELECT h.team, greatest(0.85, least(1.25, hw / rw)) AS pf_runs
    FROM home h JOIN road r USING (team)
    """
    con.execute(f"CREATE OR REPLACE TABLE park_factors AS {query}")
    return con.execute("SELECT count(*) FROM park_factors").fetchone()[0]


def build_league_rates(con) -> int:
    """ref_league_rates: all-player baseline per outcome, by hand and window."""
    rates = _rate_cols()
    grouping = "GROUP BY GROUPING SETS ((opp_hand), ())"
    vs_hand = "CASE WHEN GROUPING(opp_hand)=1 THEN 'ALL' ELSE opp_hand END AS vs_hand"
    # Use the batter view of PAs (every PA appears once); hand = pitcher hand.
    query = f"""
    {_base_cte('batter', 'p_throws')}
    SELECT {vs_hand}, 'career' AS window_name, COUNT(*) AS pa, {rates}
    FROM base {grouping}
    UNION ALL
    SELECT {vs_hand}, 'season' AS window_name, COUNT(*) AS pa, {rates}
    FROM base WHERE yr = (SELECT y FROM maxyr) {grouping}
    """
    con.execute(f"CREATE OR REPLACE TABLE ref_league_rates AS {query}")
    return con.execute("SELECT count(*) FROM ref_league_rates").fetchone()[0]


def build_all(con) -> dict[str, int]:
    return {
        "feat_batter_profile": build_batter_profiles(con),
        "feat_pitcher_profile": build_pitcher_profiles(con),
        "feat_team_offense": build_team_offense_profiles(con),
        "feat_team_bullpen": build_team_bullpen_profiles(con),
        "park_factors": build_park_factors(con),
        "ref_league_rates": build_league_rates(con),
    }
