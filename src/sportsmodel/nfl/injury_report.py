"""One NFL injury report for the sim AND the desk: nflverse's official weekly
report, verified per player against ESPN's live injury list, and never trusted
when stale.

nflverse's newest report is LAST week's until the Wed-Fri report posts. A player
Out last week (Penix, Week 2) would otherwise stay Out in this week's sim/desk.
So: if the report's week is older than the target week and ESPN is reachable,
the report is dropped entirely and ESPN's current designations are used; ESPN
also overrides the current report per player. PURE merge + thin IO wrapper.
"""
from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from sportsmodel.nfl import espn

_OUT_WORDS = ("out", "injured reserve", "reserve", "suspension", "suspended", "physically unable", "non football")
# ASCII apostrophe + U+2019 right single quotation mark for player name normalization
_APOS = "'" + "’"
_DROP_RE = re.compile("[." + re.escape(_APOS) + "]")


def normalize_status(s) -> str | None:
    st = str(s or "").strip().lower()
    # Treat statuses starting with "active" as None (e.g., "Active/PUP" → None)
    if st.startswith("active"):
        return None
    # Replace punctuation with spaces to normalize hyphenated/slashed status strings
    st = re.sub(r"[-/_]", " ", st)
    st = re.sub(r"\s+", " ", st).strip()
    # Match whole words for doubtful/questionable
    if re.search(r"\bdoubtful\b", st):
        return "Doubtful"
    if re.search(r"\bquestionable\b", st):
        return "Questionable"
    # Match whole words for pup/nfi
    if re.search(r"\bpup\b", st) or re.search(r"\bnfi\b", st):
        return "Out"
    # Match substring for other out words
    if any(w in st for w in _OUT_WORDS):
        return "Out"
    return None


def _key(name) -> str:
    s = _DROP_RE.sub("", str(name or "").strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)


def merge_report(nflverse_by_abbr: dict, report_week, target_week, espn_rows: list[dict],
                 name_to_abbr: dict[str, str], espn_available: bool = True) -> dict:
    stale = report_week is None or (target_week is not None and int(report_week) < int(target_week))
    use_nflverse = not (stale and espn_available)
    by_team: dict[str, dict[str, dict]] = {}
    nfl_status: dict[tuple, str] = {}
    for abbr, rows in (nflverse_by_abbr or {}).items():
        for r in rows or []:
            st = normalize_status(r.get("status"))
            nfl_status[(abbr, _key(r.get("player")))] = st
            if use_nflverse and st:
                by_team.setdefault(abbr, {})[_key(r.get("player"))] = {
                    "player": r.get("player"), "position": r.get("position"), "status": st,
                    "note": r.get("note"), "source": "nflverse"}
    conflicts = []
    for e in espn_rows or []:
        abbr = name_to_abbr.get(e.get("team"))
        k = _key(e.get("player"))
        if not abbr or not k:
            continue
        st = normalize_status(e.get("status"))
        before = nfl_status.get((abbr, k))
        prior_row = by_team.get(abbr, {}).pop(k, None)
        name = prior_row["player"] if prior_row else e.get("player")
        if st:
            by_team.setdefault(abbr, {})[k] = {"player": name, "position": prior_row.get("position") if prior_row else None,
                                               "status": st, "note": None, "source": "espn"}
        # A conflict is a real flip of a player nflverse DID designate (e.g. Out ->
        # Active/Questionable, Questionable -> Out); ESPN-only additions (IR etc.)
        # are not disagreements.
        if before is not None and (before in ("Out", "Doubtful")) != (st in ("Out", "Doubtful")):
            conflicts.append({"team": abbr, "player": name, "nflverse": before, "espn": st})
    return {"by_team": {a: list(v.values()) for a, v in by_team.items()}, "stale": bool(stale),
            "report_week": report_week, "target_week": target_week,
            "espn_available": espn_available, "conflicts": conflicts}


def latest_report_week(df) -> int | None:
    """Latest week with a real designation in the nflverse injuries frame."""
    if df is None or len(df) == 0 or "report_status" not in df.columns:
        return None
    rep = df[df["report_status"].isin(["Out", "Doubtful", "Questionable"])]
    return int(rep["week"].max()) if len(rep) else None


def current_report(now: datetime, target_week, name_to_abbr: dict[str, str]) -> dict:
    """IO: nflverse season report + ESPN live injuries, merged."""
    import nfl_data_py as nfl

    from sportsmodel.nfl.injuries_nflverse import nfl_season, parse_injuries
    from sportsmodel.nfl.nflverse import import_by_season

    df = import_by_season(nfl.import_injuries, [nfl_season(now)], "injuries", required=False)
    week = latest_report_week(df)
    by_abbr = parse_injuries(df, week=week) if week is not None else {}
    try:
        espn_rows, ok = espn.fetch_injuries(), True
    except Exception as exc:  # noqa: BLE001 -- ESPN down: fall back to nflverse, flagged
        print(f"WARN espn injuries unavailable ({exc!r})")
        espn_rows, ok = [], False
    return merge_report(by_abbr, week, target_week, espn_rows, name_to_abbr, espn_available=ok)


_ET = ZoneInfo("America/New_York")


def _schedule_target_week(now: datetime, schedules_df) -> int | None:
    """Earliest REG-season week of `nfl_season(now)` with a game on/after today (ET)."""
    from sportsmodel.nfl.injuries_nflverse import nfl_season

    df = schedules_df
    if df is None or len(df) == 0 or not {"season", "game_type", "week", "gameday"} <= set(df.columns):
        return None
    today = now.astimezone(_ET).date().isoformat()
    rows = df[(df["season"] == nfl_season(now)) & (df["game_type"] == "REG")
              & (df["gameday"].astype(str).str[:10] >= today)]
    return int(rows["week"].min()) if len(rows) else None


def resolve_target_week(now: datetime, schedules_df=None) -> int | None:
    """The NFL week being picked/simmed. ESPN first; on failure, the local
    nflverse schedule (assets/nfl/schedules.parquet unless `schedules_df` is
    given); None only if both fail (merge_report then treats it as not stale)."""
    try:
        return int(espn.resolve_target_week()["week"])
    except Exception as exc:  # noqa: BLE001 -- fall back to the local schedule
        print(f"WARN espn target week unavailable ({exc!r}); using local schedule")
    try:
        if schedules_df is None:
            import pandas as pd

            from sportsmodel import config
            schedules_df = pd.read_parquet(config.PROJECT_ROOT / "assets" / "nfl" / "schedules.parquet")
        week = _schedule_target_week(now, schedules_df)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN schedule target week unavailable ({exc!r})")
        week = None
    if week is None:
        print("WARN target week unresolved; injury staleness unchecked")
    return week
