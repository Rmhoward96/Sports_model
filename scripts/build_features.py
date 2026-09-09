"""Per-game feature assembler -> assets/<sport>/features.parquet.

`assemble` is PURE: it consumes an Elo-augmented per-game schedule frame
(the shape `run_elo(schedule_df, EloConfig()).games` produces -- each row
already carries pre-game `elo_home`/`elo_away`) plus a team -> EPA/PPA dict,
and emits one feature row per game. `main()` is the thin live wrapper that
loads the committed schedule parquet, runs Elo over it, loads EPA (NFL) or
PPA (CFB, via CFBD), calls `assemble`, and writes the output parquet.

No-leakage contract
--------------------
For a target game at (season, week), the schedule-derived features (L10,
SOS, SOV, rest) for BOTH teams are computed only over the same-season slice
of games strictly before that week (`season == target_season and
week < target_week`). A team's first game of a season therefore has no
prior slice at all, so its L10/SOS/SOV/rest come back NaN -- this is the
scoping the whole task is built to get right, not an edge-case afterthought.
Elo values are not recomputed here: `elo_home`/`elo_away` on each input row
are already the PRE-GAME ratings `run_elo` attached, so they're read
straight off the row for the target game itself.

CFB has no reliable per-game dates in its schedule (see Task-3 notes), so
CFB rest is `rest_weeks_cfb`'s week-gap/bye proxy rather than day counts;
NFL rest is exact day counts via `rest_days_nfl`.

Season scope for team-level EPA/PPA (documented, keep-it-simple choice):
main() aggregates EPA/PPA over the single most-recently-completed season in
the schedule (not the full historical span) for BOTH sports -- nflverse
play-by-play for the full multi-decade span is hundreds of MB, and CFBD's
`/ppa/teams` is inherently a single-`year` query. `assemble` itself is
agnostic to how many seasons went into the dict; a caller wanting a
multi-season blend can pass one in.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel.features import schedule as sf
from sportsmodel.nfl.elo import EloConfig, run_elo

_ASSETS = Path(__file__).resolve().parents[1] / "assets"

# Game-identity columns to carry through when present on the input frame,
# in addition to the always-present season/week/home_team/away_team.
_ID_COLS = ["game_id", "espn", "game_pk"]

_NAN = float("nan")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _num(value: Any) -> Any:
    """None -> NaN so every feature column is numeric/NaN, never None."""
    return _NAN if value is None else value


def _diff(home_value: Any, away_value: Any) -> float:
    """home - away, NaN (never a crash) if either side is missing."""
    if _is_missing(home_value) or _is_missing(away_value):
        return _NAN
    return home_value - away_value


def assemble(
    sport: str,
    elo_games_df: pd.DataFrame,
    epa_or_ppa_by_team: dict[str, dict] | None,
    upcoming: bool = False,
) -> pd.DataFrame:
    """Pure: Elo-augmented schedule + team EPA/PPA dict -> one feature row/game.

    `sport` is "nfl" or "cfb" -- it selects the rest-feature shape (NFL exact
    day rest vs CFB week-gap/bye) and the EPA/PPA column names
    (`*_off_epa`/`*_def_epa` vs `*_off_ppa`/`*_def_ppa`).

    With `upcoming=False` (default): only games with BOTH scores present are
    emitted, each carrying `margin`/`total` targets. With `upcoming=True`:
    only games with NO scores present are emitted (the current slate), with
    no target columns at all.

    Every home/away feature pair also gets a `*_diff` = home - away column;
    a diff is NaN whenever either side's underlying value is None/NaN --
    it never raises.
    """
    if sport not in ("nfl", "cfb"):
        raise ValueError(f"sport must be 'nfl' or 'cfb', got {sport!r}")

    epa_or_ppa_by_team = epa_or_ppa_by_team or {}
    off_key, def_key = ("off_epa", "def_epa") if sport == "nfl" else ("off_ppa", "def_ppa")

    df = elo_games_df.sort_values(["season", "week"]).reset_index(drop=True)
    id_cols = [c for c in _ID_COLS if c in df.columns]

    rows: list[dict[str, Any]] = []
    for _, g in df.iterrows():
        season = g["season"]
        week = g["week"]
        home = g["home_team"]
        away = g["away_team"]
        home_score = g.get("home_score")
        away_score = g.get("away_score")
        completed = pd.notna(home_score) and pd.notna(away_score)

        if upcoming and completed:
            continue
        if not upcoming and not completed:
            continue

        # No-leakage slice: same season, strictly earlier week than this game.
        prior = df[(df["season"] == season) & (df["week"] < week)]

        row: dict[str, Any] = {
            "season": season,
            "week": week,
            "home_team": home,
            "away_team": away,
        }
        for c in id_cols:
            row[c] = g[c]

        elo_home = g["elo_home"]
        elo_away = g["elo_away"]
        row["home_elo"] = elo_home
        row["away_elo"] = elo_away
        row["elo_diff"] = _diff(elo_home, elo_away)

        home_l10 = sf.last10_win_pct(prior, home)
        away_l10 = sf.last10_win_pct(prior, away)
        row["home_last10"] = _num(home_l10)
        row["away_last10"] = _num(away_l10)
        row["last10_diff"] = _diff(home_l10, away_l10)

        home_sos = sf.sos(prior, home)
        away_sos = sf.sos(prior, away)
        row["home_sos"] = _num(home_sos)
        row["away_sos"] = _num(away_sos)
        row["sos_diff"] = _diff(home_sos, away_sos)

        home_sov = sf.sov(prior, home)
        away_sov = sf.sov(prior, away)
        row["home_sov"] = _num(home_sov)
        row["away_sov"] = _num(away_sov)
        row["sov_diff"] = _diff(home_sov, away_sov)

        if sport == "nfl":
            gameday = g["gameday"]
            home_rest = sf.rest_days_nfl(prior, home, gameday)
            away_rest = sf.rest_days_nfl(prior, away, gameday)
            row["home_rest"] = _num(home_rest)
            row["away_rest"] = _num(away_rest)
            row["rest_diff"] = _diff(home_rest, away_rest)
        else:  # cfb
            home_rw = sf.rest_weeks_cfb(prior, home, week)
            away_rw = sf.rest_weeks_cfb(prior, away, week)
            row["home_rest_weeks"] = _num(home_rw["weeks_since_prev"])
            row["away_rest_weeks"] = _num(away_rw["weeks_since_prev"])
            row["home_off_bye"] = home_rw["off_bye"]
            row["away_off_bye"] = away_rw["off_bye"]

        home_epa = epa_or_ppa_by_team.get(home) or {}
        away_epa = epa_or_ppa_by_team.get(away) or {}
        home_off = home_epa.get(off_key)
        home_def = home_epa.get(def_key)
        away_off = away_epa.get(off_key)
        away_def = away_epa.get(def_key)
        row[f"home_{off_key}"] = _num(home_off)
        row[f"home_{def_key}"] = _num(home_def)
        row[f"away_{off_key}"] = _num(away_off)
        row[f"away_{def_key}"] = _num(away_def)
        row[f"{off_key}_diff"] = _diff(home_off, away_off)
        row[f"{def_key}_diff"] = _diff(home_def, away_def)

        if not upcoming:
            row["margin"] = home_score - away_score
            row["total"] = home_score + away_score

        rows.append(row)

    return pd.DataFrame(rows)


def _latest_completed_season(sched: pd.DataFrame) -> int:
    completed = sched.dropna(subset=["home_score", "away_score"])
    source = completed if len(completed) else sched
    return int(source["season"].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    args = parser.parse_args()

    assets_dir = _ASSETS / args.sport
    sched = pd.read_parquet(assets_dir / "schedules.parquet")

    elo_games = run_elo(sched, EloConfig()).games
    season = _latest_completed_season(sched)

    if args.sport == "nfl":
        from sportsmodel.nfl.epa import load_team_epa

        epa_by_team = load_team_epa([season])
    else:
        from sportsmodel.cfb import cfbd

        api_key = os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError("CFBD_API_KEY must be set to fetch CFB PPA")
        payload = cfbd._get(
            "/ppa/teams", api_key, {"year": season, "excludeGarbageTime": "true"}
        )
        epa_by_team = cfbd.parse_team_ppa(payload)

    features = assemble(args.sport, elo_games, epa_by_team, upcoming=False)

    out_path = assets_dir / "features.parquet"
    features.to_parquet(out_path, index=False)

    print(f"rows: {len(features)}")
    print(f"columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
