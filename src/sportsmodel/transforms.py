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
        "ref_league_rates": build_league_rates(con),
    }
