"""Build the per-slate CFB decision-desk input bundle -> JSON.

Assembles one entry per UPCOMING FBS game for the (later, in-session) desk
agents (statistics/analyst/news) to read: a model block (margin/total/
win_prob from `predictions_current`), a recent-form block (last-N results +
ATS/pace summary from `assets/cfb/schedules.parquet`), a news block
(injuries for both teams, from SportsDataIO via `cfb.sportsdata`; weather
stays None -- see below), and the pick-time market line.

SportsDataIO's CFB API has no News endpoint and no usable weather endpoint
(verified against their published OpenAPI swagger -- the prior
implementation guessed at both and 404'd live), so neither is fetched here.
Injuries come from the real `InjuredPlayers` endpoint, keyed by team
ABBREVIATION; `Teams` supplies the abbreviation -> school crosswalk needed
to rekey onto ESPN's display names (see `_rekey_by_espn_name` below). Both
SportsDataIO calls are non-fatal: a failure there logs a warning and yields
empty injuries rather than aborting the whole bundle -- the model/form
blocks are more valuable than a hard dependency on a third-party feed.

Line convention (per the Task 2 ruling): market_spread/market_total are the
ESPN pickcenter line (sportsbook convention -- home favored -> negative
spread), the SAME source Task 5's grader uses for the closing line, so CLV
comparisons are apples-to-apples. `predictions_current.market_spread` /
`market_total` are themselves ESPN pickcenter values already (see
scripts/generate_cfb.py, which stamps game_predictions.market_spread/total
straight from `cfb.espn.fetch_schedule`'s embedded odds) -- so `main()`
REUSES those columns rather than re-fetching ESPN, per the ruling's explicit
allowance. We do NOT touch `assets/cfb/lines.parquet` (CFBD home-margin
convention) anywhere in this script.

`build_bundle` is pure (no network/DB/file access) -- see its docstring for
the exact input/output shapes. `main()` is the thin IO wrapper.

Usage:
    SPORTSDATA_API_KEY=... DATABASE_URL=... \\
        PYTHONPATH=src uv run python scripts/desk_inputs.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel import config
from sportsmodel.cfb import sportsdata
from sportsmodel.db import get_postgres

OUT_PATH = config.DATA_DIR / "cfb" / "desk_bundle.json"

RECENT_FORM_N = 5

log = logging.getLogger(__name__)


# =============================================================================
# Pure assembly
# =============================================================================

def build_bundle(
    games: list[dict],
    model_rows: list[dict],
    form_rows: dict[str, dict],
    injuries: dict[str, list[dict]],
    weather: dict[int, dict],
    now: datetime,
) -> list[dict]:
    """Assemble one bundle entry per UPCOMING game. PURE -- no network/DB/file.

    Args:
      games: one dict per game in the slate: {"game_pk", "home_team",
        "away_team", "commence_time"}. `home_team`/`away_team` are the team
        identifiers used to key into `form_rows`/`injuries` (and are also
        used, verbatim, as the display names in `matchup`) -- main() is
        responsible for putting every input into the same team-identifier
        space before calling this function.
      model_rows: one dict per game the model has predicted:
        {"game_pk", "home_win_prob", "pred_home_score", "pred_away_score",
        "market_spread", "market_total"}. A game_pk absent here (model
        hasn't run for that game yet) gets an all-None model block and None
        market fields -- it still appears in the output.
      form_rows: {team -> recent-form summary dict}, passed through as-is
        (last-N results + ATS/pace summary -- shape is main()'s to define).
        A team absent here yields `None` for that side's form.
      injuries: {team -> list of injury dicts}, as `sportsdata.parse_injuries`
        returns (already rekeyed by main() onto the same team-identifier
        space as `games`). A team absent here yields `[]` for that side's
        injuries.
      weather: {game_pk -> weather dict}, keyed here by the SAME game_pk as
        `games`. Currently always empty -- SportsDataIO's CFB API has no
        usable weather endpoint (see module docstring) -- so every game's
        weather is None; the parameter is kept so a future weather source
        can be plugged in without changing this function's shape.
      now: only games with `commence_time` strictly after `now` are
        included (already-started/finished games are dropped). Compared as
        ISO-8601 strings via `datetime.fromisoformat` (with a trailing "Z"
        normalized to "+00:00").

    Returns one dict per upcoming game:
      {"game_pk", "matchup" ("{away} @ {home}"), "commence_time",
       "market_spread", "market_total",
       "model": {"margin", "total", "win_prob"},
       "form": {"home", "away"},
       "news": {"injuries": {"home", "away"}, "weather"}}
    """
    model_by_pk = {row["game_pk"]: row for row in model_rows}

    out = []
    for game in games:
        if _parse_iso(game["commence_time"]) <= now:
            continue

        game_pk = game["game_pk"]
        home_team, away_team = game["home_team"], game["away_team"]
        model_row = model_by_pk.get(game_pk)

        pred_home = model_row.get("pred_home_score") if model_row else None
        pred_away = model_row.get("pred_away_score") if model_row else None
        has_pred = pred_home is not None and pred_away is not None

        out.append({
            "game_pk": game_pk,
            "matchup": f"{away_team} @ {home_team}",
            "commence_time": game["commence_time"],
            "market_spread": model_row.get("market_spread") if model_row else None,
            "market_total": model_row.get("market_total") if model_row else None,
            "model": {
                "margin": (pred_home - pred_away) if has_pred else None,
                "total": (pred_home + pred_away) if has_pred else None,
                "win_prob": model_row.get("home_win_prob") if model_row else None,
            },
            "form": {
                "home": form_rows.get(home_team),
                "away": form_rows.get(away_team),
            },
            "news": {
                "injuries": {
                    "home": injuries.get(home_team, []),
                    "away": injuries.get(away_team, []),
                },
                "weather": weather.get(game_pk),
            },
        })
    return out


def _parse_iso(ts: str) -> datetime:
    """ISO-8601 -> aware datetime, tolerating a trailing 'Z' (ESPN's format)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# =============================================================================
# Recent-form computation (pure over an already-loaded schedule; no file IO)
# =============================================================================

def compute_recent_form(schedule: pd.DataFrame, teams: set[str], n: int = RECENT_FORM_N) -> dict[str, dict]:
    """Last-N results + a simple pace/ATS-adjacent summary per team, from the
    CFBD-home-margin-convention `schedules.parquet` (season, week, home_team,
    away_team, home_score, away_score, game_type).

    Only `game_type == "REG"` games with both scores recorded are considered.
    Rows are ordered by (season, week) and the trailing `n` games per team
    are kept. Note: schedules.parquet has no per-game market line, so this is
    a scoring/pace trend (points-for/against, W-L, avg margin), NOT a true
    ATS record -- that would need historical lines joined in, which is out
    of scope here (see task-2-report.md).

    Returns {team -> {"record": "W-L", "last_n": ["W"/"L", ...] most-recent
    first, "avg_margin": float, "pace": float (avg combined points)}}, one
    entry per team in `teams` that has at least one recorded game. Teams
    with zero recorded games are simply absent (build_bundle already treats
    a missing team as `None` form).
    """
    df = schedule[
        (schedule["game_type"] == "REG")
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    ].sort_values(["season", "week"])

    out: dict[str, dict] = {}
    for team in teams:
        mask_home = df["home_team"] == team
        mask_away = df["away_team"] == team
        games = df[mask_home | mask_away]
        if games.empty:
            continue
        recent = games.tail(n)

        results, margins, totals = [], [], []
        wins = losses = 0
        for _, row in recent.iterrows():
            is_home = row["home_team"] == team
            pf = row["home_score"] if is_home else row["away_score"]
            pa = row["away_score"] if is_home else row["home_score"]
            margin = pf - pa
            margins.append(margin)
            totals.append(pf + pa)
            if margin > 0:
                wins += 1
                results.append("W")
            elif margin < 0:
                losses += 1
                results.append("L")
            else:
                results.append("T")

        out[team] = {
            "record": f"{wins}-{losses}",
            "last_n": list(reversed(results)),  # most-recent first
            "avg_margin": sum(margins) / len(margins),
            "pace": sum(totals) / len(totals),
        }
    return out


# =============================================================================
# Best-effort team-name reconciliation (SportsDataIO <-> ESPN display names)
# =============================================================================

def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return "".join(c for c in s if c.isalnum() or c.isspace()).strip()


def _rekey_by_espn_name(source: dict[str, object], espn_names: list[str]) -> dict[str, object]:
    """Best-effort remap of a SportsDataIO team-keyed dict onto ESPN display
    names, matching a SportsDataIO school name (e.g. "Alabama") as a
    normalized PREFIX of the ESPN displayName (e.g. "Alabama Crimson Tide").

    SportsDataIO's CFB payloads key rows by bare school name while
    predictions_current carries ESPN's full displayName -- there's no
    published id crosswalk between the two providers, so this is a
    heuristic, not a guaranteed join (see task-2-report.md "concerns").
    Keys that don't uniquely match an ESPN name are dropped rather than
    risk attaching a team's injuries to the wrong game.
    """
    norm_espn = {_norm_name(n): n for n in espn_names}
    out: dict[str, object] = {}
    for key, value in source.items():
        nk = _norm_name(key)
        matches = [espn for norm, espn in norm_espn.items() if norm.startswith(nk) or nk.startswith(norm)]
        if len(matches) == 1:
            out[matches[0]] = value
    return out


# =============================================================================
# main()
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the CFB decision-desk input bundle.")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--days-ahead", type=int, default=7,
                     help="Only include games within this many days of now (default 7).")
    args = ap.parse_args()

    api_key = os.environ.get("SPORTSDATA_API_KEY")
    if not api_key:
        sys.exit("SPORTSDATA_API_KEY not set in environment (add it as a secret / export it).")

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=args.days_ahead)

    # -- model rows + game identity, both from predictions_current (CFB) --
    # predictions_current.market_spread/market_total are ESPN pickcenter
    # values already (see module docstring) -- reused here rather than
    # re-fetched, per the Task 2 ruling.
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT game_pk, game_date, home_team_name, away_team_name,
                   home_win_prob, pred_home_score, pred_away_score,
                   commence_time, market_spread, market_total
            FROM predictions_current
            WHERE sport = 'cfb'
        """)
        cols = ["game_pk", "game_date", "home_team_name", "away_team_name",
                "home_win_prob", "pred_home_score", "pred_away_score",
                "commence_time", "market_spread", "market_total"]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

    games = [
        {"game_pk": r["game_pk"], "home_team": r["home_team_name"],
         "away_team": r["away_team_name"], "commence_time": r["commence_time"].isoformat()}
        for r in rows
        if r["commence_time"] is not None and now < r["commence_time"] <= horizon
    ]
    model_rows = [
        {"game_pk": r["game_pk"], "home_win_prob": r["home_win_prob"],
         "pred_home_score": r["pred_home_score"], "pred_away_score": r["pred_away_score"],
         "market_spread": r["market_spread"], "market_total": r["market_total"]}
        for r in rows
    ]
    espn_names = sorted({g["home_team"] for g in games} | {g["away_team"] for g in games})

    if not games:
        print("No upcoming CFB games in predictions_current within the window; writing empty bundle.")

    # -- recent form, from schedules.parquet --
    # schedules.parquet keys teams by ESPN id/"FCS" (not display name), while
    # predictions_current (and thus `games`/`espn_names` above) is keyed by
    # ESPN displayName -- translate ids -> names via the same fbs_teams.json
    # asset cfb.teams reads, so compute_recent_form can join on team name.
    schedule_path = config.PROJECT_ROOT / "assets" / "cfb" / "schedules.parquet"
    schedule = pd.read_parquet(schedule_path)
    fbs_names_path = config.PROJECT_ROOT / "assets" / "cfb" / "fbs_teams.json"
    id_to_name = json.loads(fbs_names_path.read_text()) if fbs_names_path.exists() else {}
    schedule_named = schedule.assign(
        home_team=schedule["home_team"].astype(str).map(lambda t: id_to_name.get(t, t)),
        away_team=schedule["away_team"].astype(str).map(lambda t: id_to_name.get(t, t)),
    )
    form_rows = compute_recent_form(schedule_named, set(espn_names))

    # -- injuries, from SportsDataIO's real InjuredPlayers/Teams endpoints --
    # InjuredPlayers keys rows by team ABBREVIATION (e.g. "SMU"); Teams
    # supplies the abbreviation -> School crosswalk, and _rekey_by_espn_name
    # then prefix-matches School (e.g. "Florida State") onto ESPN's
    # displayName (e.g. "Florida State Seminoles"). Both calls are
    # non-fatal: a SportsDataIO outage/error logs a warning and yields empty
    # injuries rather than aborting the whole bundle -- the model/form
    # blocks are still worth writing on their own.
    injuries: dict[str, list[dict]] = {}
    try:
        injuries_by_abbrev = sportsdata.parse_injuries(
            sportsdata._get("/scores/json/InjuredPlayers", api_key)
        )
        teams_by_abbrev = sportsdata.parse_teams(
            sportsdata._get("/scores/json/Teams", api_key)
        )
        injuries_by_school = {
            teams_by_abbrev[abbrev]: rows
            for abbrev, rows in injuries_by_abbrev.items()
            if abbrev in teams_by_abbrev
        }
        injuries = _rekey_by_espn_name(injuries_by_school, espn_names)
    except Exception:
        log.warning("SportsDataIO injuries/teams fetch failed; continuing with no injury data", exc_info=True)

    # Weather is intentionally NOT fetched here: SportsDataIO's CFB API has
    # no usable weather endpoint (verified against their OpenAPI swagger --
    # see module docstring), so the news block's weather is left None for
    # every game.
    weather: dict[int, dict] = {}

    bundle = build_bundle(games, model_rows, form_rows, injuries, weather, now)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"Wrote {args.out} ({len(bundle)} games)")


if __name__ == "__main__":
    main()
