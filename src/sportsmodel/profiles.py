"""Loaders for the committed rate-profile snapshots (assets/profiles/*.parquet).

Shared by the props generator. Each returns plain dicts so callers stay pure.
"""
from __future__ import annotations

import duckdb

from . import config

OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
PROFILE_DIR = config.PROJECT_ROOT / "assets" / "profiles"
_MIN_PA = 150  # season window needs this many PA/BF, else fall back to career


def _best_window_vectors(table: str, id_col: str, ids, min_pa: int = _MIN_PA):
    """{id: outcome-vector} choosing season if it has >= min_pa, else career."""
    keys = sorted({i for i in ids if i is not None})
    if not keys:
        return {}
    path = PROFILE_DIR / f"{table}.parquet"
    key_list = ",".join(f"'{k}'" if isinstance(k, str) else str(int(k)) for k in keys)
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT {id_col}, window_name, pa, {', '.join(OUTCOMES)} "
        f"FROM read_parquet('{path}') "
        f"WHERE vs_hand = 'ALL' AND {id_col} IN ({key_list})"
    ).fetchall()
    con.close()
    best: dict = {}
    for r in rows:
        key, win, pa = r[0], r[1], r[2]
        vec = {o: r[3 + i] for i, o in enumerate(OUTCOMES)}
        rank = 0 if (win == "season" and pa >= min_pa) else (1 if win == "career" else 2)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, vec)
    return {k: v for k, (_, v) in best.items()}


def load_pitcher_vectors(ids) -> dict[int, dict]:
    return _best_window_vectors("feat_pitcher_profile", "player_id", ids)


def load_batter_vectors(ids) -> dict[int, dict]:
    return _best_window_vectors("feat_batter_profile", "player_id", ids)


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
