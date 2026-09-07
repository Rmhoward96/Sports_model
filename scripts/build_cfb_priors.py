"""Build forward-looking CFB preseason priors -> assets/cfb/priors.parquet.

Pulls, per season in `--seasons START END`, the six CFBD signals that predict
a team's *coming* season before any games are played: SP+ overall rating,
returning production, recruiting class strength, transfer-portal net rating,
first-year-head-coach flag, and (for strength-of-schedule) the season's game
schedule. `build_priors_rows` is the pure per-season assembly step -- it
takes already-parsed dicts (the shapes cfbd.parse_* return) for the season
plus the prior season's SP+ ratings/schedule (SoS needs both), and returns
one row per team. main() is the thin network/IO wrapper: it resolves the
season range, calls the retried CFBD client per endpoint per season, runs
the parsers, calls build_priors_rows, maps team names -> ESPN ids (dropping
unmapped/non-FBS names), concatenates every season, and writes the parquet.

Strength-of-schedule convention (raw, not z-scored -- z-scoring is a later
task):
  - prior_sos: average SP+ rating of the team's PRIOR-season opponents
    (using the PRIOR season's SP+ ratings, since that's when they played).
  - forward_sos_shift: this season's average opponent SP+ rating (using this
    season's schedule + this season's SP+ ratings, i.e. opponents' current
    preseason strength) minus prior_sos. Positive => this year's schedule
    projects tougher than last year's.
  Opponents absent from the relevant SP+ dict (e.g. FCS teams, which SP+
  doesn't rate) are simply excluded from the average, not treated as 0.

Market-independent: this script never reads odds/lines, only CFBD's
team/roster/schedule signals.

Usage:
    CFBD_API_KEY=... PYTHONPATH=src uv run --no-sync python scripts/build_cfb_priors.py \
        --seasons 2015 2025
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel.cfb import cfbd
from sportsmodel.cfb.teams import cfbd_to_espn

OUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "cfb" / "priors.parquet"

COLUMNS = [
    "season", "team_espn_id", "team_name", "sp_rating", "returning_pct",
    "returning_starters", "qb_returning", "recruiting_points", "portal_net",
    "coach_first_year", "prior_sos", "forward_sos_shift",
]


def _avg_opponent_sp(team: str, games: list[dict], sp_ratings: dict[str, float]) -> float | None:
    """Average SP+ rating of `team`'s opponents in `games`, using `sp_ratings`.

    Opponents missing from `sp_ratings` (e.g. FCS teams, which SP+ doesn't
    cover) are excluded from the average rather than treated as 0. None if
    `team` has no rated opponents in `games` (no games recorded, or every
    opponent is unrated)."""
    opponents = []
    for g in games:
        if g["home_team"] == team:
            opponents.append(g["away_team"])
        elif g["away_team"] == team:
            opponents.append(g["home_team"])
    ratings = [sp_ratings[o] for o in opponents if o in sp_ratings]
    return sum(ratings) / len(ratings) if ratings else None


def build_priors_rows(parsed: dict, season: int) -> list[dict]:
    """Pure per-season row assembly. No network/DB/file access.

    `parsed` holds the already-parsed per-endpoint data for `season`:
      - "sp": dict[team, SP+ rating] for `season` (defines the team universe)
      - "sp_prior": dict[team, SP+ rating] for `season - 1`
      - "returning": dict[team, {returning_pct, returning_starters, qb_returning}]
      - "recruiting": dict[team, points]
      - "portal": dict[team, {"in", "out", "net"}]
      - "coaches": dict[team, bool] (first-year HC this season)
      - "games": list[{"home_team", "away_team", "season"}] for `season`
      - "games_prior": same shape, for `season - 1`

    Returns one dict per team in `sp` (the SP+ universe is the team-season
    universe: SP+ rates every FBS team every season). Missing per-team data
    from the other endpoints becomes None (returning/recruiting -- genuinely
    unknown) or a documented default (portal_net 0.0 = no net transfer
    activity recorded; coach_first_year False = no coaching-change signal).
    `team_espn_id` is NOT included here -- name -> ESPN-id mapping is main()'s
    job, since it needs the cfb.teams asset (an IO-adjacent lookup), not pure
    parser output.
    """
    sp = parsed["sp"]
    sp_prior = parsed.get("sp_prior", {})
    returning = parsed.get("returning", {})
    recruiting = parsed.get("recruiting", {})
    portal = parsed.get("portal", {})
    coaches = parsed.get("coaches", {})
    games = parsed.get("games", [])
    games_prior = parsed.get("games_prior", [])

    rows = []
    for team in sorted(sp):
        ret = returning.get(team, {})
        prior_sos = _avg_opponent_sp(team, games_prior, sp_prior)
        this_sos = _avg_opponent_sp(team, games, sp)
        forward_shift = (
            this_sos - prior_sos if (this_sos is not None and prior_sos is not None) else None
        )
        rows.append({
            "season": season,
            "team_name": team,
            "sp_rating": sp[team],
            "returning_pct": ret.get("returning_pct"),
            "returning_starters": ret.get("returning_starters"),
            "qb_returning": ret.get("qb_returning"),
            "recruiting_points": recruiting.get(team),
            "portal_net": portal.get(team, {}).get("net", 0.0),
            "coach_first_year": coaches.get(team, False),
            "prior_sos": prior_sos,
            "forward_sos_shift": forward_shift,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CFB preseason priors parquet from CFBD.")
    ap.add_argument("--seasons", type=int, nargs=2, metavar=("START", "END"), required=True,
                     help="Inclusive season range, e.g. --seasons 2015 2025")
    args = ap.parse_args()
    start, end = args.seasons

    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        sys.exit("CFBD_API_KEY not set in environment (add it as a secret / export it).")

    # SP+ + schedules are needed both as "this season" and (for the next
    # season in the loop) "prior season" -- fetch the whole range including
    # one extra season before `start` so every season in [start, end] has a
    # prior-season SP+/schedule to compute SoS against.
    sp_by_season: dict[int, dict[str, float]] = {}
    games_by_season: dict[int, list[dict]] = {}
    for s in range(start - 1, end + 1):
        sp_by_season[s] = cfbd.parse_sp(cfbd._get("/ratings/sp", api_key, params={"year": s}))
        games_by_season[s] = cfbd.parse_games(
            cfbd._get("/games", api_key, params={"year": s, "seasonType": "regular"})
        )
        print(f"  fetched SP+/schedule for {s} ({len(sp_by_season[s])} teams, "
              f"{len(games_by_season[s])} games)", flush=True)

    all_rows: list[dict] = []
    dropped = 0
    for season in range(start, end + 1):
        returning = cfbd.parse_returning(
            cfbd._get("/player/returning", api_key, params={"year": season})
        )
        recruiting = cfbd.parse_recruiting(
            cfbd._get("/recruiting/teams", api_key, params={"year": season})
        )
        portal = cfbd.parse_portal(
            cfbd._get("/player/portal", api_key, params={"year": season})
        )
        coaches = cfbd.parse_coaches(
            cfbd._get("/coaches", api_key, params={"year": season}), season
        )

        parsed = {
            "sp": sp_by_season[season],
            "sp_prior": sp_by_season[season - 1],
            "returning": returning,
            "recruiting": recruiting,
            "portal": portal,
            "coaches": coaches,
            "games": games_by_season[season],
            "games_prior": games_by_season[season - 1],
        }
        rows = build_priors_rows(parsed, season)
        for row in rows:
            espn_id = cfbd_to_espn(row["team_name"])
            if espn_id is None:
                dropped += 1
                continue
            row["team_espn_id"] = espn_id
            all_rows.append(row)
        print(f"{season}: {len(rows)} teams", flush=True)

    print(f"Dropped {dropped} unmapped/non-FBS team-seasons")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df = df.sort_values(["season", "team_espn_id"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"\nWrote {OUT_PATH} ({len(df)} rows)")
    if len(df):
        print(f"seasons: {sorted(df.season.unique())}")


if __name__ == "__main__":
    main()
