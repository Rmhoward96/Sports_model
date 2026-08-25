from __future__ import annotations
import pandas as pd

_SKILL = {"QB", "RB", "HB", "FB", "WR", "TE"}

def _norm(name: str) -> str:
    return " ".join((name or "").lower().split())

def active_universe(rosters: pd.DataFrame, injuries: pd.DataFrame,
                    espn_inactives, season: int, week: int) -> list[dict]:
    out_ids = set()
    if injuries is not None and len(injuries):
        inj = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
        out_ids = set(inj[inj["report_status"] == "Out"]["gsis_id"])
    inactive_names = {_norm(n) for n in (espn_inactives or [])}
    uni = []
    for _, r in rosters.iterrows():
        if r["position"] not in _SKILL:
            continue
        if r["player_id"] in out_ids:
            continue
        if _norm(r["player_name"]) in inactive_names:
            continue
        uni.append({"player_id": r["player_id"], "player_name": r["player_name"],
                    "team": r["team"], "position": r["position"],
                    "depth_chart_position": r.get("depth_chart_position")})
    return uni

def match_book_player(name: str, universe: list[dict]):
    target = _norm(name)
    for p in universe:
        if _norm(p["player_name"]) == target:
            return p["player_id"]
    return None

def bump_backup(universe: list[dict], out_player_id: str) -> list[dict]:
    # The OUT player is already excluded by active_universe; this promotes the next
    # same-team/position player to the vacated depth slot (marker for share redistribution).
    return [p for p in universe if p["player_id"] != out_player_id]
