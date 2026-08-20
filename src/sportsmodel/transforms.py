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

# Optional point-in-time cutoff for backtesting: when set (YYYY-MM-DD), every profile
# is built using only games strictly before this date (no look-ahead leakage). None in
# production. Set via set_cutoff(); every transform WHERE injects _date_filter().
_CUTOFF: str | None = None


def set_cutoff(cutoff: str | None) -> None:
    global _CUTOFF
    _CUTOFF = cutoff


def _date_filter() -> str:
    return f"AND CAST(game_date AS DATE) < DATE '{_CUTOFF}'" if _CUTOFF else ""


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
          {_date_filter()}
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
          {_date_filter()}
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


def build_team_defense(con) -> int:
    """feat_team_defense(team, def_factor): fielding quality on balls in play.

    On BIP (contact, excluding K/BB/HBP/HR), compare a team's ACTUAL hits allowed to
    the EXPECTED hits from contact quality (estimated_ba_using_speedangle). A team that
    allows fewer hits than its contact quality predicts is fielding well. Normalized so
    league-average = 1.0; def_factor < 1 = better defense. Clamped to a sane band.
    """
    query = f"""
    WITH bip AS (
        SELECT CASE WHEN inning_topbot='Top' THEN home_team ELSE away_team END AS team,
               CASE WHEN events IN ('single','double','triple') THEN 1.0 ELSE 0.0 END AS actual_hit,
               estimated_ba_using_speedangle AS xhit
        FROM read_parquet('{_PARQUET_GLOB}')
        WHERE events IS NOT NULL AND inning_topbot IS NOT NULL
          AND events NOT IN ('strikeout','strikeout_double_play','walk','intent_walk','hit_by_pitch','home_run')
          AND estimated_ba_using_speedangle IS NOT NULL
          {_date_filter()}
    ),
    lg AS (SELECT avg(actual_hit) / avg(xhit) AS lgr FROM bip)
    SELECT team,
           greatest(0.92, least(1.08, (avg(actual_hit) / avg(xhit)) / (SELECT lgr FROM lg))) AS def_factor
    FROM bip GROUP BY team
    """
    con.execute(f"CREATE OR REPLACE TABLE feat_team_defense AS {query}")
    return con.execute("SELECT count(*) FROM feat_team_defense").fetchone()[0]


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
          {_date_filter()}
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
          {_date_filter()}
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


# Outs recorded by a PA outcome (double plays count 2, triple plays 3).
_OUTS_CASE = """
CASE
  WHEN events IN ('grounded_into_double_play','double_play','strikeout_double_play','sac_fly_double_play') THEN 2
  WHEN events = 'triple_play' THEN 3
  WHEN events IN ('strikeout','field_out','force_out','fielders_choice_out','other_out','sac_bunt','sac_fly') THEN 1
  ELSE 0
END
"""


def build_pitcher_workload(con) -> int:
    """feat_pitcher_workload: a starter's avg batters faced & outs recorded per start.

    A 'start' = a game in which the pitcher threw in the 1st inning. Feeds the pitcher
    props (BF -> Ks / Hits Allowed; outs -> Outs Recorded). Requires >= 3 starts.
    """
    query = f"""
    WITH pas AS (
        SELECT game_pk, CAST(pitcher AS BIGINT) AS player_id, inning, {_OUTS_CASE} AS outs
        FROM read_parquet('{_PARQUET_GLOB}')
        WHERE events IS NOT NULL AND pitcher IS NOT NULL
          {_date_filter()}
    ),
    starts AS (SELECT DISTINCT game_pk, player_id FROM pas WHERE inning = 1),
    per_start AS (
        SELECT p.game_pk, p.player_id, COUNT(*) AS bf, SUM(p.outs) AS outs
        FROM pas p JOIN starts s USING (game_pk, player_id)
        GROUP BY p.game_pk, p.player_id
    )
    SELECT player_id,
           avg(bf)                          AS avg_bf,
           coalesce(stddev_samp(bf), 3.0)   AS sd_bf,
           avg(outs)                        AS avg_outs,
           coalesce(stddev_samp(outs), 6.0) AS sd_outs,
           count(*)                         AS starts
    FROM per_start GROUP BY player_id HAVING count(*) >= 3
    """
    con.execute(f"CREATE OR REPLACE TABLE feat_pitcher_workload AS {query}")
    return con.execute("SELECT count(*) FROM feat_pitcher_workload").fetchone()[0]


def shrink_toward_league(con, table: str) -> None:
    """Empirical-Bayes regression of a player profile's rates toward league average.

    Small samples are noise (a 20-PA hitter with 3 HR is not a 15% HR bat). Each rate
    becomes (rate*pa + k*league_rate)/(pa + k) with per-outcome k (stabilization PA),
    then the 7 outcomes are renormalized. See docs/methodology.md §A.1.
    """
    from .model.rates import DEFAULT_K, OUTCOMES

    league = dict(zip(OUTCOMES, con.execute(
        f"SELECT {', '.join(OUTCOMES)} FROM ref_league_rates "
        f"WHERE vs_hand='ALL' AND window_name='career' LIMIT 1"
    ).fetchone()))

    df = con.execute(f"SELECT * FROM {table}").df()
    for o in OUTCOMES:
        k = DEFAULT_K[o]
        df[o] = (df[o] * df["pa"] + k * league[o]) / (df["pa"] + k)
    total = df[list(OUTCOMES)].sum(axis=1)
    for o in OUTCOMES:
        df[o] = df[o] / total
    df["p_hit"] = df["p_1b"] + df["p_2b"] + df["p_3b"] + df["p_hr"]
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")


def build_all(con) -> dict[str, int]:
    return {
        "feat_batter_profile": build_batter_profiles(con),
        "feat_pitcher_profile": build_pitcher_profiles(con),
        "feat_team_offense": build_team_offense_profiles(con),
        "feat_team_bullpen": build_team_bullpen_profiles(con),
        "feat_team_defense": build_team_defense(con),
        "feat_pitcher_workload": build_pitcher_workload(con),
        "park_factors": build_park_factors(con),
        "ref_league_rates": build_league_rates(con),
    }
