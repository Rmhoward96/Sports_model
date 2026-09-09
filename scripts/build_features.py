"""Per-game feature assembler -> assets/<sport>/features.parquet.

`assemble` is PURE: it consumes an Elo-augmented per-game schedule frame
(the shape `run_elo(schedule_df, EloConfig()).games` produces -- each row
already carries pre-game `elo_home`/`elo_away`) plus a (season, team) ->
EPA/PPA dict, and emits one feature row per game. `main()` is the thin live
wrapper that loads the committed schedule parquet, runs Elo over it, loads
EPA (NFL) or PPA (CFB, via CFBD), calls `assemble`, and writes the output
parquet.

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

Season scope for team-level EPA/PPA (leakage-free, prior-season):
a game in season S uses each team's EPA/PPA computed over season S-1 only
-- by the time season S kicks off, season S-1 is fully complete and known,
so there is no way for a game to see EPA/PPA derived from its own season's
plays (which would leak the game's own future/in-progress plays into its
features). `assemble` itself does not know about "prior season" at all: it
just looks up `(game_season, team)` in the dict it's given. The season
shift is entirely `main()`'s responsibility -- it aggregates EPA/PPA per
season and stores season S-1's values under key `(S, team)` before calling
`assemble`. The earliest season in the schedule has no S-1 to draw from, so
those games' EPA/PPA come back NaN (correct: there is nothing else to use
without leaking).
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
    epa_by_season_team: dict[tuple[int, str], dict] | None,
    upcoming: bool = False,
) -> pd.DataFrame:
    """Pure: Elo-augmented schedule + (season, team) EPA/PPA dict -> one
    feature row per game.

    `sport` is "nfl" or "cfb" -- it selects the rest-feature shape (NFL exact
    day rest vs CFB week-gap/bye) and the EPA/PPA column names
    (`*_off_epa`/`*_def_epa` vs `*_off_ppa`/`*_def_ppa`).

    `epa_by_season_team` is keyed `(season, team) -> {off/def value}`. For
    each game, the home/away EPA/PPA join looks up `(game_season, home_team)`
    / `(game_season, away_team)` -- i.e. whatever value the caller stored
    under the CURRENT game's season key. `assemble` has no notion of "prior
    season" itself; it is the caller's job (see `main()`) to have already
    shifted prior-season aggregates onto the current season's key so no game
    ever sees EPA/PPA computed from its own season. A missing key (e.g. no
    entry at all for that season) yields NaN, same as a missing team.

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

    epa_by_season_team = epa_by_season_team or {}
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

        home_epa = epa_by_season_team.get((season, home)) or {}
        away_epa = epa_by_season_team.get((season, away)) or {}
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    args = parser.parse_args()

    assets_dir = _ASSETS / args.sport
    sched = pd.read_parquet(assets_dir / "schedules.parquet")

    elo_games = run_elo(sched, EloConfig()).games
    seasons = sorted(int(s) for s in sched["season"].dropna().unique())

    if args.sport == "nfl":
        from sportsmodel.nfl.epa import team_epa_by_season
        import nfl_data_py as nfl

        # Need pbp for every season present in the schedule PLUS each of
        # their season-1's, so every season S has its prior season's EPA
        # available to shift forward onto key (S, team).
        needed = set(seasons) | {s - 1 for s in seasons}
        pbp = nfl.import_pbp_data(sorted(needed))
        by_season = team_epa_by_season(pbp)

        # Leakage-free shift: season S's key holds season (S-1)'s EPA.
        epa_by_season_team: dict[tuple[int, str], dict] = {}
        for (prev_season, team), values in by_season.items():
            epa_by_season_team[(prev_season + 1, team)] = values
    else:
        from sportsmodel.cfb import cfbd

        api_key = os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise RuntimeError("CFBD_API_KEY must be set to fetch CFB PPA")

        # One CFBD call per distinct prior season (dedupe requested years).
        prior_seasons = sorted({s - 1 for s in seasons})
        ppa_by_prior_season = {}
        for prior in prior_seasons:
            payload = cfbd._get(
                "/ppa/teams", api_key, {"year": prior, "excludeGarbageTime": "true"}
            )
            ppa_by_prior_season[prior] = cfbd.parse_team_ppa(payload)

        # Leakage-free shift: season S's key holds season (S-1)'s PPA.
        epa_by_season_team = {}
        for s in seasons:
            for team, values in ppa_by_prior_season[s - 1].items():
                epa_by_season_team[(s, team)] = values

    features = assemble(args.sport, elo_games, epa_by_season_team, upcoming=False)

    out_path = assets_dir / "features.parquet"
    features.to_parquet(out_path, index=False)

    print(f"rows: {len(features)}")
    print(f"columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
