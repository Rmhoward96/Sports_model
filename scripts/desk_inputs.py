"""Build the per-slate decision-desk input bundle -> JSON, for CFB or NFL.

Assembles one entry per UPCOMING game (`--sport {cfb,nfl}`, default cfb) for
the (later, in-session) desk agents (statistics/analyst/news) to read: a
model block (margin/total/win_prob from `predictions_current`), a
recent-form block (last-N results + ATS/pace summary from
`assets/<sport>/schedules.parquet`), a news block (injuries for both teams,
from SportsDataIO via the sport's `sportsdata` adapter; weather stays None --
see below), and the pick-time market line. All sport-specific inputs (adapter,
schedules/crosswalk paths, DB filter, output path) are resolved by
`_sport_config`; everything else here is sport-generic.

SportsDataIO's CFB/NFL APIs have no News endpoint and no usable weather endpoint
(verified against their published OpenAPI swagger -- the prior
implementation guessed at both and 404'd live), so neither is fetched here.
Injuries come from the real `InjuredPlayers` endpoint, keyed by team
ABBREVIATION; `Teams` supplies the abbreviation -> full-name crosswalk (CFB
School / NFL FullName) needed to rekey onto ESPN's display names (see
`_rekey_by_espn_name` below). Both
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
        PYTHONPATH=src uv run python scripts/desk_inputs.py [--sport cfb|nfl] [--out PATH]
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
from sportsmodel.cfb import sportsdata as cfb_sportsdata
from sportsmodel.nfl import sportsdata as nfl_sportsdata
from sportsmodel.nfl import espn as nfl_espn
from sportsmodel.nfl import injury_report
from sportsmodel.db import get_postgres

RECENT_FORM_N = 5

log = logging.getLogger(__name__)


# =============================================================================
# Pure assembly
# =============================================================================

def _rec(r: dict | None, a: str = "w", b: str = "l") -> str | None:
    """Format a record dict {a, b, p} as 'a-b-p' string, or None if empty."""
    if not r or (r.get(a) is None and r.get(b) is None):
        return None
    return f"{r.get(a) or 0}-{r.get(b) or 0}-{r.get('p') or 0}"


def trend_block(records: dict | None, situational: list[dict], *, is_home: bool, is_fav: bool | None) -> dict | None:
    """PURE. One team's trends for the desk: Action Network season records
    relevant to this game (ATS overall / this venue / this role / last 5, O/U
    overall / this venue, units) + NFL situational trend lines. None when empty."""
    out: dict = {}
    if records:
        venue = "home" if is_home else "road"
        venue_lbl = "at home" if is_home else "on the road"
        picks = [("ATS", _rec(records.get("ats"))), (f"ATS {venue_lbl}", _rec(records.get(f"ats_{venue}"))),
                 ("ATS last 5", _rec(records.get("ats_last_5"))),
                 ("O/U", _rec(records.get("over_under"), "o", "u")),
                 (f"O/U {venue_lbl}", _rec(records.get(f"over_under_{venue}"), "o", "u"))]
        if is_fav is not None:
            role, role_lbl = ("fav", "as favorite") if is_fav else ("dog", "as underdog")
            picks.insert(2, (f"ATS {role_lbl}", _rec(records.get(f"ats_{role}"))))
        units = records.get("units")
        if units and units.get("w") is not None:
            # Action Network stores units lost as a negative number (0-2 SU ->
            # l = -2); subtract the magnitude so either sign convention works.
            net = float(units.get("w") or 0) - abs(float(units.get("l") or 0))
            picks.append(("Units", f"{net:+.1f}"))
        out["records"] = {k: v for k, v in picks if v is not None}
    sit = [f"{t['ats_w']}-{t['ats_l']}-{t['ats_p']} ATS, O/U {t['ou_o']}-{t['ou_u']}-{t['ou_p']} "
           f"{t['label']} since {t['since_season']}" for t in (situational or [])]
    if sit:
        out["situational"] = sit
    if not out:
        return None
    out.setdefault("records", {})
    out.setdefault("situational", [])
    return out


def game_trends(
    home_team: str,
    away_team: str,
    market_spread: float | None,
    records_by_team: dict[str, dict],
    sit_by_game_team: dict[tuple[int, str], list[dict]],
    game_pk: int,
) -> dict:
    """PURE. Assemble trends for both sides of a single game.

    Args:
      home_team, away_team: team names (keys for records_by_team)
      market_spread: ESPN home-referenced spread (negative = home favored);
        None or 0 -> is_fav = None for both teams (no role trends).
      records_by_team: {team -> records dict}
      sit_by_game_team: {(game_pk, team) -> [situational trends]}
      game_pk: game identifier

    Returns: {"home": trend_block(...) | None, "away": trend_block(...) | None}
    """
    # Determine is_fav from market_spread: None or 0 (pick'em) -> None (no role);
    # negative = home favored (True); positive = away favored (home False).
    is_fav_home = None if market_spread is None or market_spread == 0 else (market_spread < 0)
    is_fav_away = None if is_fav_home is None else (not is_fav_home)

    home_trends = trend_block(
        records_by_team.get(home_team),
        sit_by_game_team.get((game_pk, home_team), []),
        is_home=True,
        is_fav=is_fav_home,
    )
    away_trends = trend_block(
        records_by_team.get(away_team),
        sit_by_game_team.get((game_pk, away_team), []),
        is_home=False,
        is_fav=is_fav_away,
    )

    return {"home": home_trends, "away": away_trends}


def build_bundle(
    games: list[dict],
    model_rows: list[dict],
    form_rows: dict[str, dict],
    injuries: dict[str, list[dict]],
    weather: dict[int, dict],
    now: datetime,
    trends: dict[int, dict] | None = None,
    injury_meta: dict | None = None,
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
        `games`. Currently always empty -- no usable SportsDataIO weather
        endpoint is wired in (see module docstring) -- so every game's
        weather is None; the parameter is kept so a future weather source
        can be plugged in without changing this function's shape.
      now: only games with `commence_time` strictly after `now` are
        included (already-started/finished games are dropped). Compared as
        ISO-8601 strings via `datetime.fromisoformat` (with a trailing "Z"
        normalized to "+00:00").
      trends: {game_pk -> {"home": trend_block, "away": trend_block}}, passed
        through as-is (season records + situational trends). A game_pk absent
        here yields {"home": None, "away": None} for that game's trends.
      injury_meta: the injury source's freshness metadata (see
        `_nfl_injuries_by_name` / `_cfb_injuries_by_name`): {"source",
        "stale", "report_week"?, "target_week"?, "conflicts": [{"team"
        (display name), "player", "nflverse", "espn"}]}. Each game gets
        `news.injury_report` = {"source", "stale", "report_week",
        "target_week", "conflicts"} with conflicts filtered to that game's
        home/away teams; None for every game when injury_meta is None.

    Returns one dict per upcoming game:
      {"game_pk", "matchup" ("{away} @ {home}"), "commence_time",
       "market_spread", "market_total",
       "model": {"margin", "total", "win_prob"},
       "form": {"home", "away"},
       "news": {"injuries": {"home", "away"}, "injury_report", "weather"},
       "trends": {"home": ..., "away": ...}}
    """
    model_by_pk = {row["game_pk"]: row for row in model_rows}
    trends_by_pk = trends or {}

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
                "injury_report": _injury_report_block(injury_meta, home_team, away_team),
                "weather": weather.get(game_pk),
            },
            "trends": trends_by_pk.get(game_pk) or {"home": None, "away": None},
        })
    return out


def _injury_report_block(meta: dict | None, home_team: str, away_team: str) -> dict | None:
    """Per-game injury freshness block; conflicts limited to this game's teams."""
    if meta is None:
        return None
    teams = {home_team, away_team}
    return {
        "source": meta.get("source"),
        "stale": meta.get("stale"),
        "report_week": meta.get("report_week"),
        "target_week": meta.get("target_week"),
        "conflicts": [c for c in meta.get("conflicts") or [] if c.get("team") in teams],
    }


def _parse_iso(ts: str) -> datetime:
    """ISO-8601 -> aware datetime, tolerating a trailing 'Z' (ESPN's format)."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


# =============================================================================
# Recent-form computation (pure over an already-loaded schedule; no file IO)
# =============================================================================

def _current_season(now: datetime) -> int:
    """The season year in progress at `now`.

    NFL and CFB seasons both start ~August and run through the following
    January/February (bowls / Super Bowl), so games in Jan/Feb belong to the
    PRIOR calendar year's season, and the spring/summer offseason belongs to
    the last completed season. Aug onward is the new season. This matches the
    `season` convention in schedules.parquet (the year the season started).
    """
    return now.year if now.month >= 8 else now.year - 1


def compute_recent_form(
    schedule: pd.DataFrame, teams: set[str], current_season: int, n: int = RECENT_FORM_N
) -> dict[str, dict]:
    """Current-season W-L record + a recent scoring/pace summary per team, from
    the CFBD-home-margin-convention `schedules.parquet` (season, week,
    home_team, away_team, home_score, away_score, game_type).

    ONLY games in `current_season` are considered (`game_type == "REG"` with
    both scores recorded). This is deliberate: the trailing games must never
    span a season boundary, or a team that has played 2 games this season
    would show a record built from last season's games (the desk once showed
    CFB "Ole Miss 5-0 / LSU 2-3" for teams with only 2 current-season games,
    because the schedule asset lagged the season and the trailing window
    reached back into the prior year). A team with NO current-season games is
    absent (build_bundle treats missing form as `None`) -- honest "no data
    yet", never a prior-season record.

    `record` is the FULL current-season W-L (all of the team's current-season
    games). The recent-form metrics (`last_n`, `avg_margin`, `pace`) use the
    trailing `n` current-season games. schedules.parquet has no per-game
    market line, so this is a scoring/pace trend, NOT a true ATS record.

    Returns {team -> {"record": "W-L", "last_n": ["W"/"L", ...] most-recent
    first, "avg_margin": float, "pace": float (avg combined points)}}.
    """
    df = schedule[
        (schedule["game_type"] == "REG")
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
        & (schedule["season"] == current_season)
    ].sort_values(["season", "week"])

    out: dict[str, dict] = {}
    for team in teams:
        mask_home = df["home_team"] == team
        mask_away = df["away_team"] == team
        games = df[mask_home | mask_away]
        if games.empty:
            continue

        # record: full current-season W-L (every current-season game).
        wins = losses = 0
        for _, row in games.iterrows():
            is_home = row["home_team"] == team
            margin = (row["home_score"] - row["away_score"]) if is_home else (row["away_score"] - row["home_score"])
            if margin > 0:
                wins += 1
            elif margin < 0:
                losses += 1

        # recent-form metrics: trailing n current-season games.
        recent = games.tail(n)
        results, margins, totals = [], [], []
        for _, row in recent.iterrows():
            is_home = row["home_team"] == team
            pf = row["home_score"] if is_home else row["away_score"]
            pa = row["away_score"] if is_home else row["home_score"]
            margin = pf - pa
            margins.append(margin)
            totals.append(pf + pa)
            results.append("W" if margin > 0 else "L" if margin < 0 else "T")

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
    names, matching a SportsDataIO team name as a normalized PREFIX (either
    direction) of the ESPN displayName -- CFB "Alabama" -> "Alabama Crimson
    Tide"; NFL "Philadelphia Eagles" -> "Philadelphia Eagles" (an exact
    match, which prefix-matching subsumes).

    SportsDataIO keys rows by its own team name (CFB School / NFL FullName)
    while predictions_current carries ESPN's full displayName -- there's no
    published id crosswalk between the two providers, so this is a
    heuristic, not a guaranteed join. Keys that don't uniquely match an ESPN
    name are dropped rather than risk attaching a team's injuries to the
    wrong game.
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
# injury sources -- one per sport, each returning ({full_team_name -> [rows]}, meta)
# =============================================================================
#
# Both return injuries keyed by FULL team name (ready for _rekey_by_espn_name)
# plus freshness metadata for build_bundle's `news.injury_report`.
# They differ only in where the data comes from and how abbreviations resolve:
#   - CFB: SportsDataIO's real InjuredPlayers + Teams endpoints (its CFB feed
#     is genuine). abbrev -> School name via parse_teams (a live call). Needs
#     the api_key; ignores the crosswalk.
#   - NFL: the shared injury report (sportsmodel.nfl.injury_report, same as
#     the sim): nflverse's official report verified per player against ESPN's
#     live list, dropped when stale (SportsDataIO's NFL feed is SCRAMBLED).
#     abbrev -> full name via the nfl_teams.json crosswalk (the SAME
#     abbrev->name map already loaded for form). Needs no api_key.


def _cfb_injuries_by_name(adapter, api_key, crosswalk, now) -> tuple[dict[str, list[dict]], dict]:
    """CFB injuries, ({School name -> rows}, meta), from SportsDataIO. Its CFB
    feed is live (no weekly report), so there's no stale concept."""
    injuries_by_abbrev = adapter.parse_injuries(
        adapter._get(adapter.INJURED_PLAYERS_PATH, api_key)
    )
    teams_by_abbrev = adapter.parse_teams(
        adapter._get(adapter.TEAMS_PATH, api_key)
    )
    by_name = {
        teams_by_abbrev[abbrev]: rows
        for abbrev, rows in injuries_by_abbrev.items()
        if abbrev in teams_by_abbrev
    }
    return by_name, {"source": "sportsdata", "stale": False, "conflicts": []}


def _nfl_target_week() -> int | None:
    """The NFL week the desk is picking, from ESPN; None on any error (the
    injury report then treats nflverse's newest week as not stale)."""
    try:
        return int(nfl_espn.resolve_target_week()["week"])
    except Exception:
        log.warning("target week unavailable; injury staleness unchecked", exc_info=True)
        return None


def _nfl_injuries_by_name(adapter, api_key, crosswalk, now) -> tuple[dict[str, list[dict]], dict]:
    """NFL injuries, ({full team name -> rows}, meta), from the shared injury
    report (`injury_report.current_report`, the same one the sim uses).

    The report keys by nflverse abbreviation (ARI, PHI, LA, ...) -- exactly
    the keys in `nfl_teams.json` -- so the form crosswalk resolves them to ESPN
    full names; its inverse is the report's name_to_abbr for ESPN's rows.
    meta = the report minus by_team, plus "source", with each conflict's team
    mapped abbr -> display name (so build_bundle can filter per game).
    `adapter`/`api_key` are unused; kept so both sources share one shape."""
    name_to_abbr = {name: abbr for abbr, name in crosswalk.items()}
    report = injury_report.current_report(now, _nfl_target_week(), name_to_abbr)
    by_name = {
        crosswalk[abbrev]: rows
        for abbrev, rows in report["by_team"].items()
        if abbrev in crosswalk
    }
    meta = {k: v for k, v in report.items() if k != "by_team"}
    meta["source"] = "nflverse+espn"
    meta["conflicts"] = [
        {**c, "team": crosswalk.get(c.get("team"), c.get("team"))}
        for c in report.get("conflicts") or []
    ]
    return by_name, meta


def _fetch_nfl_sim(conn) -> dict[int, dict]:
    """{game_pk -> {"home_win_prob","margin","total","disagreement"}} from
    `nfl_sim_current`. NFL-only signal (the sim engine has no CFB analog).
    Caller wraps this in try/except -- a missing table (migration not yet
    run) or any other DB hiccup should degrade to no sim data rather than
    abort the bundle, mirroring the injury-fetch non-fatal pattern above."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT game_pk, sim_home_win_prob, sim_margin, sim_total, disagreement
            FROM nfl_sim_current
        """)
        cols = ["game_pk", "home_win_prob", "margin", "total", "disagreement"]
        return {
            row[0]: dict(zip(cols[1:], row[1:]))
            for row in cur.fetchall()
        }


def _sport_config(sport: str) -> dict:
    """Per-sport paths/adapter for main(). Everything else in this module is
    sport-generic. Raises SystemExit on an unsupported sport."""
    root = config.PROJECT_ROOT
    configs = {
        "cfb": {
            "adapter": cfb_sportsdata,
            "injury_source": _cfb_injuries_by_name,
            "schedules_path": root / "assets" / "cfb" / "schedules.parquet",
            "crosswalk_path": root / "assets" / "cfb" / "fbs_teams.json",
            "out_default": config.DATA_DIR / "cfb" / "desk_bundle.json",
        },
        "nfl": {
            "adapter": nfl_sportsdata,
            "injury_source": _nfl_injuries_by_name,
            "schedules_path": root / "assets" / "nfl" / "schedules.parquet",
            "crosswalk_path": root / "assets" / "nfl" / "nfl_teams.json",
            "out_default": config.DATA_DIR / "nfl" / "desk_bundle.json",
        },
    }
    if sport not in configs:
        raise SystemExit(f"unsupported --sport {sport!r} (expected one of {sorted(configs)})")
    return configs[sport]


# =============================================================================
# main()
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build the decision-desk input bundle.")
    ap.add_argument("--sport", choices=["cfb", "nfl"], default="cfb")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output path (defaults to data/<sport>/desk_bundle.json).")
    ap.add_argument("--days-ahead", type=int, default=7,
                     help="Only include games within this many days of now (default 7).")
    args = ap.parse_args()

    cfg = _sport_config(args.sport)
    out_path = args.out if args.out is not None else cfg["out_default"]
    adapter = cfg["adapter"]

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
            WHERE sport = %(sport)s
        """, {"sport": args.sport})
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
        print(f"No upcoming {args.sport} games in predictions_current within the window; writing empty bundle.")

    # -- recent form, from schedules.parquet --
    # schedules.parquet keys teams by the schedule's own team key (CFB: ESPN
    # id/"FCS"; NFL: SportsDataIO abbreviation), while predictions_current
    # (and thus `games`/`espn_names` above) is keyed by ESPN displayName --
    # translate the schedule's key -> name via the sport's crosswalk asset,
    # so compute_recent_form can join on team name. A team key absent from
    # the crosswalk falls through unchanged (degrades to no-form, not a
    # crash).
    schedule = pd.read_parquet(cfg["schedules_path"])
    crosswalk_path = cfg["crosswalk_path"]
    crosswalk = json.loads(crosswalk_path.read_text()) if crosswalk_path.exists() else {}
    schedule_named = schedule.assign(
        home_team=schedule["home_team"].astype(str).map(lambda t: crosswalk.get(t, t)),
        away_team=schedule["away_team"].astype(str).map(lambda t: crosswalk.get(t, t)),
    )
    form_rows = compute_recent_form(schedule_named, set(espn_names), _current_season(now))

    # -- injuries, from the sport's injury_source (see the functions above) --
    # Each source returns {full team name -> rows} (CFB School / NFL FullName,
    # e.g. "Florida State" / "Philadelphia Eagles"); _rekey_by_espn_name then
    # prefix-matches that onto ESPN's displayName ("Florida State Seminoles" /
    # "Philadelphia Eagles"). CFB uses SportsDataIO's real feed; NFL uses the
    # shared nflverse+ESPN report (SportsDataIO's NFL injury feed is
    # scrambled). Each source also returns freshness meta for
    # news.injury_report. The fetch is non-fatal: any outage/error logs a
    # warning and yields empty injuries (and injury_report None) rather than
    # aborting the whole bundle -- the model/form blocks are still worth
    # writing on their own.
    injuries: dict[str, list[dict]] = {}
    injury_meta: dict | None = None
    try:
        injuries_by_name, injury_meta = cfg["injury_source"](adapter, api_key, crosswalk, now)
        injuries = _rekey_by_espn_name(injuries_by_name, espn_names)
    except Exception:
        log.warning("injury fetch failed; continuing with no injury data", exc_info=True)

    # Weather is intentionally NOT fetched here: SportsDataIO's CFB/NFL feeds
    # have no usable weather endpoint wired in (verified against their OpenAPI
    # swagger -- see module docstring), so the news block's weather is left
    # None for every game.
    weather: dict[int, dict] = {}

    # -- trends, from team_betting_records + nfl_game_trends (NFL only) --
    # team_betting_records stores {category: {w, l, p}} season records per team.
    # nfl_game_trends (NFL only) stores situational ATS/O-U trends per game/team.
    # The fetch is non-fatal: missing tables log warnings and yield no trends
    # (desk still runs); the desk methodology treats trends as supporting evidence
    # only, never required for a pick.
    trends: dict[int, dict] = {}
    records_by_team: dict[str, dict] = {}
    try:
        with get_postgres() as conn, conn.cursor() as cur:
            # Load team betting records for the current season
            cur.execute("""
                SELECT team_name, records
                FROM team_betting_records
                WHERE sport = %(sport)s AND season = %(season)s
            """, {"sport": args.sport, "season": _current_season(now)})
            for team_name, records_json in cur.fetchall():
                # Handle JSONB that might come back as dict or str
                recs = records_json if isinstance(records_json, dict) else json.loads(records_json or "{}")
                records_by_team[team_name] = recs
    except Exception:
        log.warning("team_betting_records fetch failed; continuing with no records data", exc_info=True)

    # Load NFL situational trends if applicable
    sit_by_game_team: dict[tuple[int, str], list[dict]] = {}
    if args.sport == "nfl":
        try:
            game_pks = [g["game_pk"] for g in games]
            if game_pks:
                with get_postgres() as conn, conn.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(game_pks))
                    cur.execute(f"""
                        SELECT game_pk, team_name, label, ats_w, ats_l, ats_p,
                               ou_o, ou_u, ou_p, since_season
                        FROM nfl_game_trends
                        WHERE game_pk IN ({placeholders})
                    """, game_pks)
                    for row in cur.fetchall():
                        game_pk, team_name, label, ats_w, ats_l, ats_p, ou_o, ou_u, ou_p, since_season = row
                        sit_by_game_team.setdefault((game_pk, team_name), []).append({
                            "label": label,
                            "ats_w": ats_w,
                            "ats_l": ats_l,
                            "ats_p": ats_p,
                            "ou_o": ou_o,
                            "ou_u": ou_u,
                            "ou_p": ou_p,
                            "since_season": since_season,
                        })
        except Exception:
            log.warning("nfl_game_trends fetch failed; continuing with no situational trends", exc_info=True)

    # Build trends dict keyed by game_pk using the pure game_trends helper
    model_by_pk_trends = {row["game_pk"]: row for row in model_rows}
    for game in games:
        game_pk = game["game_pk"]
        model_row = model_by_pk_trends.get(game_pk)
        mkt_spread = model_row.get("market_spread") if model_row else None
        trends[game_pk] = game_trends(
            game["home_team"],
            game["away_team"],
            mkt_spread,
            records_by_team,
            sit_by_game_team,
            game_pk,
        )

    bundle = build_bundle(games, model_rows, form_rows, injuries, weather, now,
                          trends=trends, injury_meta=injury_meta)

    # -- sim disagreement signal, NFL only, from nfl_sim_current --
    # LEFT-join by game_pk: a game with no sim row yet (or the table itself
    # missing -- migration not applied) gets sim=None rather than aborting
    # the bundle, same as the injury fetch above. CFB games get no "sim" key
    # at all (the sim engine is NFL-only).
    if args.sport == "nfl":
        sim_by_pk: dict[int, dict] = {}
        try:
            with get_postgres() as conn:
                sim_by_pk = _fetch_nfl_sim(conn)
        except Exception:
            log.warning("sim fetch failed; continuing with no sim data", exc_info=True)
        for game in bundle:
            game["sim"] = sim_by_pk.get(game["game_pk"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, default=str))
    print(f"Wrote {out_path} ({len(bundle)} games)")


if __name__ == "__main__":
    main()
