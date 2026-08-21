"""Loaders for the committed rate-profile snapshots (assets/profiles/*.parquet).

Shared by the props generator. Each returns plain dicts so callers stay pure.
"""
from __future__ import annotations

import duckdb
import polars as pl

from . import config

OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
PROFILE_DIR = config.PROJECT_ROOT / "assets" / "profiles"

# Recency-weighted blend of the three windows. Validated in the walk-forward backtest
# (2024 + 2025): gentle recency beats no-recency and aggressive recency on win prob.
WINDOW_WEIGHTS = {"30d": 0.2, "season": 0.4, "career": 0.4}


def _blend_rows(rows, min_season: int, min_recent: int) -> dict:
    """Blend 30d/season/career per entity, gating thin windows and renormalizing."""
    per: dict = {}
    for r in rows:
        key, win, pa = r[0], r[1], r[2]
        vec = {o: r[3 + i] for i, o in enumerate(OUTCOMES)}
        per.setdefault(key, {})[win] = (pa, vec)
    gate = {"30d": min_recent, "season": min_season, "career": 1}
    out = {}
    for key, wins in per.items():
        parts = [(WINDOW_WEIGHTS[w], vec) for w, (pa, vec) in wins.items()
                 if w in WINDOW_WEIGHTS and pa >= gate[w]]
        if not parts:
            if "career" in wins:
                out[key] = wins["career"][1]
            continue
        tw = sum(w for w, _ in parts)
        out[key] = {o: sum(w * vec[o] for w, vec in parts) / tw for o in OUTCOMES}
    return out


def _query_rows(table: str, id_col: str, ids=None):
    path = PROFILE_DIR / f"{table}.parquet"
    where = "vs_hand = 'ALL'"
    if ids is not None:
        keys = sorted({i for i in ids if i is not None})
        if not keys:
            return []
        key_list = ",".join(f"'{k}'" if isinstance(k, str) else str(int(k)) for k in keys)
        where += f" AND {id_col} IN ({key_list})"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT {id_col}, window_name, pa, {', '.join(OUTCOMES)} "
        f"FROM read_parquet('{path}') WHERE {where}"
    ).fetchall()
    con.close()
    return rows


def load_pitcher_vectors(ids) -> dict[int, dict]:
    return _blend_rows(_query_rows("feat_pitcher_profile", "player_id", ids), 150, 40)


def load_batter_vectors(ids) -> dict[int, dict]:
    return _blend_rows(_query_rows("feat_batter_profile", "player_id", ids), 150, 40)


def load_team_offense_vectors() -> dict[str, dict]:
    return _blend_rows(_query_rows("feat_team_offense", "team"), 1000, 300)


def load_team_bullpen_vectors() -> dict[str, dict]:
    return _blend_rows(_query_rows("feat_team_bullpen", "team"), 1000, 300)


def load_league_vector() -> dict:
    path = PROFILE_DIR / "ref_league_rates.parquet"
    con = duckdb.connect(":memory:")
    r = con.execute(
        f"SELECT {', '.join(OUTCOMES)} FROM read_parquet('{path}') "
        f"WHERE vs_hand = 'ALL' AND window_name = 'career' LIMIT 1"
    ).fetchone()
    con.close()
    return {o: r[i] for i, o in enumerate(OUTCOMES)}


def load_park_factors() -> dict[str, float]:
    path = PROFILE_DIR / "park_factors.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(f"SELECT team, pf_runs FROM read_parquet('{path}')").fetchall()
    con.close()
    return {t: pf for t, pf in rows}


def load_team_defense() -> dict[str, float]:
    path = PROFILE_DIR / "feat_team_defense.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(f"SELECT team, def_factor FROM read_parquet('{path}')").fetchall()
    con.close()
    return {t: f for t, f in rows}


def load_pitcher_workload(ids) -> dict[int, tuple]:
    """{player_id: (avg_bf, sd_bf, avg_outs, sd_outs)} for starters with >= 3 starts."""
    keys = sorted({int(i) for i in ids if i is not None})
    if not keys:
        return {}
    path = PROFILE_DIR / "feat_pitcher_workload.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT player_id, avg_bf, sd_bf, avg_outs, sd_outs FROM read_parquet('{path}') "
        f"WHERE player_id IN ({','.join(map(str, keys))})"
    ).fetchall()
    con.close()
    return {int(r[0]): (r[1], r[2], r[3], r[4]) for r in rows}


def load_advancement_rows() -> list[dict]:
    """Load committed advancement table rows (outcome, occ, end_occ, runs, prob) as dicts."""
    path = config.PROJECT_ROOT / "assets" / "advancement" / "mlb_advancement.parquet"
    return pl.read_parquet(path).to_dicts()
