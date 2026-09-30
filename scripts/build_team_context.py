"""Daily team-context job (NFL + CFB): team game log, history and matchup grades for
the upcoming games, and power rankings for the upcoming week. Descriptive only --
not a pick.

Per sport:

* sources
    NFL: nflverse release schedules (``location`` / ``espn`` included) and pbp for
         seasons S-4..S (the grade cutoffs and the market scale read S-3..S-1, whose
         early-season blend needs S-4).
    CFB: assets/cfb/schedules.parquet + the ESPN live schedule (previous, current and
         next week of ESPN's current week: games the asset lacks are appended --
         upcoming ones unplayed, finished ones with their score), assets/cfb/lines.parquet,
         the ``odds_snapshot`` closing consensus for the current season
         (``live_closing_consensus``; skipped with a warning without DATABASE_URL), and
         assets/cfb/advanced_games.parquet (missing -> matchup grades skipped with a
         warning, everything else still runs).
* S = the season of the next unplayed game (kickoff >= now); the log covers S-2..S so
  L20 windows are complete.
* upcoming games = unplayed, kickoff in [now, now + 8 days): history (both sides) and
  matchup grades (per offense side; ratings as of the game's week, cutoffs frozen per
  season -- computed once per season per run).
* rankings: (S, W) = the next unplayed game's season/week; previous week's table for
  the move. NFL uses the market scale fitted once for S (``nfl_market_scale``) for both.
* writes: ``db.upsert_team_context`` (one transaction per sport); ``--dry-run`` prints
  counts, the runtime and a sample instead and performs no DB writes (it may
  read ``odds_snapshot`` for CFB closing lines when DATABASE_URL is set).

Usage:
    uv run python scripts/build_team_context.py --sport all [--dry-run] [--now ISO]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel import config, db  # noqa: E402
from sportsmodel.context.game_log import (  # noqa: E402
    cfb_game_log, live_closing_consensus, nfl_game_log)
from sportsmodel.context.history import history_for_games  # noqa: E402
from sportsmodel.context.matchup import cutoffs_for_season, grades_for_games  # noqa: E402
from sportsmodel.cfb.priors import load_weights, season_priors  # noqa: E402
from sportsmodel.cfb.teams import load_fbs_ids  # noqa: E402
from sportsmodel.context.power import rankings  # noqa: E402
from sportsmodel.context.results_power import (  # noqa: E402
    ResultsParams, load_params, played_games, power_asof)
from sportsmodel.context.units import (  # noqa: E402
    BLEND_K, POWER_BLEND_K, cfb_unit_games, nfl_unit_games, unit_ratings_asof,
    window_ratings)
from sportsmodel.nfl.elo import EloConfig  # noqa: E402
from sportsmodel.nfl.ratings import BlendConfig  # noqa: E402
from sportsmodel.nfl.teams import normalize_team  # noqa: E402

CFB_ASSETS = ROOT / "assets" / "cfb"
ADVANCED_PATH = CFB_ASSETS / "advanced_games.parquet"
UPCOMING_DAYS = 8
LOG_PRIOR_SEASONS = 2        # log = S-2..S -> L20 complete
NFL_PBP_PRIOR_SEASONS = 4    # pbp S-4..S: cutoffs / market scale use S-3..S-1 (+ S-4 blend)
TABLES = tuple(db.TEAM_CONTEXT_COLUMNS)
TABLE_COLUMNS = db.TEAM_CONTEXT_COLUMNS


def warn(msg: str) -> None:
    print(f"::warning::team-context: {msg}", flush=True)


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _empty(table: str) -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS[table])


# ------------------------------------------------------------------ shared core
class _Ratings:
    """Per-run caches: previous-season ratings, as-of ratings and grade cutoffs."""

    def __init__(self, unit_games: pd.DataFrame | None, blend_k: float = BLEND_K):
        self.ug = unit_games
        self.blend_k = blend_k
        self._prev: dict[int, pd.DataFrame] = {}
        self._asof: dict[tuple[int, int], pd.DataFrame] = {}
        self._cut: dict[int, dict | None] = {}

    def asof(self, season: int, week: int) -> pd.DataFrame:
        k = (int(season), int(week))
        if k not in self._asof:
            if k[0] not in self._prev:
                self._prev[k[0]] = window_ratings(self.ug, k[0] - 1, 99)
            self._asof[k] = unit_ratings_asof(self.ug, k[0], k[1], prev=self._prev[k[0]],
                                              blend_k=self.blend_k)
        return self._asof[k]

    def cutoffs(self, season: int) -> dict | None:
        if season not in self._cut:
            self._cut[season] = cutoffs_for_season(self.ug, int(season))
            if self._cut[season] is None:
                warn(f"no grading history for season {season}: units only, no letters")
        return self._cut[season]


def _next_unplayed(log: pd.DataFrame, now: pd.Timestamp) -> pd.Series | None:
    fut = log[log["su"].isna() & (log["kickoff"] >= now)]
    return None if fut.empty else fut.sort_values("kickoff", kind="stable").iloc[0]


def current_season(log: pd.DataFrame, now: pd.Timestamp) -> int:
    nxt = _next_unplayed(log, now)
    return int(nxt["season"]) if nxt is not None else int(log["season"].max())


def upcoming(log: pd.DataFrame, now: pd.Timestamp, days: int = UPCOMING_DAYS) -> pd.DataFrame:
    k = log["kickoff"]
    return log[log["su"].isna() & (k >= now) & (k < now + pd.Timedelta(days=days))]


def _sided(frame: pd.DataFrame, games: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Attach game_pk + side (home/away from the source schedule, not the venue)."""
    m = frame.merge(games[["game_key", "game_pk", "home_team"]], on="game_key", how="inner")
    m["side"] = np.where(m["team"] == m["home_team"], "home", "away")
    m["sport"] = sport
    return m


def _history(log: pd.DataFrame, up: pd.DataFrame, games: pd.DataFrame, sport: str):
    if up.empty:
        return _empty("team_history")
    h = history_for_games(log, up)
    if h.empty:
        return _empty("team_history")
    return _sided(h, games, sport)[TABLE_COLUMNS["team_history"]].reset_index(drop=True)


def _grades(up: pd.DataFrame, games: pd.DataFrame, cache: _Ratings, sport: str):
    if up.empty:
        return _empty("matchup_grades")
    g = games[games["game_key"].isin(set(up["game_key"]))]
    frames = []
    for (season, week), gw in g.groupby(["season", "week"], sort=True):
        cols = ["game_pk", "home_team", "away_team", "home_is_fbs", "away_is_fbs",
                "season", "week", "kickoff"]
        frames.append(grades_for_games(gw[cols], cache.asof(season, week),
                                       cache.cutoffs(int(season))))
    if not frames:
        warn(f"{sport}: no upcoming games with ids to grade; matchup_grades empty")
        return _empty("matchup_grades")
    gr = pd.concat(frames, ignore_index=True)
    gr["sport"] = sport
    return gr[TABLE_COLUMNS["matchup_grades"]].reset_index(drop=True)


def _rank_frame(rk: pd.DataFrame, sport: str, conf: dict | None = None) -> pd.DataFrame:
    if rk.empty:
        return _empty("power_rankings")
    rk = rk.copy()
    rk["sport"] = sport
    rk["conf"] = rk["team"].map(conf) if conf else None
    if "games" not in rk.columns:
        rk["games"] = None
    return rk[TABLE_COLUMNS["power_rankings"]].reset_index(drop=True)


def _log_frame(log: pd.DataFrame) -> pd.DataFrame:
    return log[TABLE_COLUMNS["team_game_log"]].reset_index(drop=True)


# ------------------------------------------------------------------------- NFL
def load_nfl_sources(now: pd.Timestamp) -> dict:
    """nflverse release schedules + pbp -> unit games (network)."""
    from sportsmodel.nfl.nflverse import load_release

    season = now.year if now.month >= 3 else now.year - 1
    seasons = list(range(season - NFL_PBP_PRIOR_SEASONS, season + 1))
    sched = load_release("schedules", seasons)
    frames = []
    for yr in seasons:   # one season at a time: pbp is wide, keep only the unit games
        pbp = load_release("pbp", [yr], required=False)
        if len(pbp):
            frames.append(nfl_unit_games(pbp))
        del pbp
    if not frames:
        raise RuntimeError(f"nflverse pbp: no seasons available from {seasons}")
    return {"schedules": sched, "unit_games": pd.concat(frames, ignore_index=True)}


def nfl_results_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """nflverse schedule -> REG played games for the results rating."""
    s = schedules[schedules["game_type"] == "REG"]
    return played_games(s.assign(
        home_team=s["home_team"].map(normalize_team), away_team=s["away_team"].map(normalize_team),
        neutral=s["location"].astype(str).str.lower().eq("neutral")))


def _nfl_games(schedules: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame({
        "game_key": schedules["game_id"].astype(str),
        "game_pk": pd.to_numeric(schedules["espn"], errors="coerce").astype("Int64"),
        "home_team": schedules["home_team"].map(normalize_team),
        "away_team": schedules["away_team"].map(normalize_team),
        "season": schedules["season"].astype(int), "week": schedules["week"].astype(int),
    })
    kick = log.drop_duplicates("game_key").set_index("game_key")["kickoff"]
    g["kickoff"] = g["game_key"].map(kick)
    g["home_is_fbs"], g["away_is_fbs"] = True, True
    return g[g["game_key"].isin(set(log["game_key"]))].reset_index(drop=True)


def build_nfl(schedules: pd.DataFrame, unit_games: pd.DataFrame,
              now: pd.Timestamp, rp: ResultsParams | None = None) -> dict[str, pd.DataFrame]:
    now = _utc(now)
    full = nfl_game_log(schedules)
    season = current_season(full, now)
    log = full[full["season"] >= season - LOG_PRIOR_SEASONS].reset_index(drop=True)
    games = _nfl_games(schedules, log)
    missing = games["game_pk"].isna()
    if missing.any():
        warn(f"nfl: {int(missing.sum())} games without an ESPN id skipped from history/grades")
        games = games[~missing]
    log = log.copy()
    log["game_pk"] = log["game_key"].map(games.set_index("game_key")["game_pk"]).astype("Int64")

    up = upcoming(log, now)
    cache = _Ratings(unit_games)
    hist = _history(log, up, games, "nfl")
    grades = _grades(up, games, cache, "nfl")

    rk = _empty("power_rankings")
    nxt = _next_unplayed(log, now)
    if nxt is not None:
        s, w = int(nxt["season"]), int(nxt["week"])
        # results-based rating (results_power); unit ranks from the season-weighted
        # unit ratings (POWER_BLEND_K) are shown alongside, not rated
        power, prev = power_asof(nfl_results_games(schedules), rp or load_params("nfl"), s, w)
        ur = _Ratings(unit_games, POWER_BLEND_K).asof(s, w)
        rk = _rank_frame(rankings(power, prev, ur, log), "nfl")
    return {"team_game_log": _log_frame(log), "team_history": hist,
            "matchup_grades": grades, "power_rankings": rk}


# ------------------------------------------------------------------------- CFB
_SCHED_COLS = ["season", "week", "home_team", "away_team", "home_score", "away_score",
               "game_type", "game_pk", "start_date", "neutral_site", "conference_game",
               "home_conf", "away_conf"]


def merge_espn_schedule(asset: pd.DataFrame, espn_games: list[dict],
                        season_type: int) -> pd.DataFrame:
    """Append ESPN scoreboard games (``cfb.espn.parse_schedule`` rows) the asset lacks.

    Only STATUS_FINAL games keep their score (others are unplayed: NaN). Postseason
    (season_type 3) games are ``game_type`` POST and their week is offset past the
    season's last regular-season week (ESPN restarts postseason weeks at 1).
    """
    have = set(pd.to_numeric(asset["game_pk"], errors="coerce").dropna().astype("int64"))
    rows, seen = [], set()
    for g in espn_games:
        pk = int(g["game_pk"])
        if pk in have or pk in seen:
            continue
        seen.add(pk)
        final = g.get("status") == "STATUS_FINAL"
        rows.append({
            "season": int(g["season"]), "week": int(g["week"]),
            "home_team": str(g["home_team"]), "away_team": str(g["away_team"]),
            "home_score": float(g["home_score"]) if final and g.get("home_score") is not None else np.nan,
            "away_score": float(g["away_score"]) if final and g.get("away_score") is not None else np.nan,
            "game_type": "POST" if season_type == 3 else "REG", "game_pk": pk,
            "start_date": g["start_date"], "neutral_site": bool(g.get("neutral_site")),
            "conference_game": bool(g.get("conference_game")),
            "home_conf": g.get("home_conf"), "away_conf": g.get("away_conf"),
        })
    base = asset[_SCHED_COLS].copy()
    base["home_score"] = base["home_score"].astype(float)
    base["away_score"] = base["away_score"].astype(float)
    if not rows:
        return base.reset_index(drop=True)
    new = pd.DataFrame(rows, columns=_SCHED_COLS)
    if season_type == 3:
        both = pd.concat([base, new[new["game_type"] == "REG"]])
        last_reg = both[both["game_type"] == "REG"].groupby("season")["week"].max()
        new["week"] = new["week"] + new["season"].map(last_reg).fillna(0).astype(int)
    return pd.concat([base, new], ignore_index=True)


def _espn_games(now: pd.Timestamp) -> tuple[list[dict], int]:
    from sportsmodel.cfb import espn

    try:
        cur = espn.fetch_current_week()
    except Exception as exc:  # noqa: BLE001 -- ESPN down: run on the asset alone
        warn(f"cfb: ESPN current week unavailable ({type(exc).__name__}); asset only")
        return [], 2
    season, week, st = int(cur["season"]), int(cur["week"]), int(cur["season_type"])
    if st not in (2, 3):
        return [], st
    games: list[dict] = []
    for w in (week - 1, week, week + 1):
        if w < 1:
            continue
        try:
            games += espn.fetch_schedule(season, w, season_type=st)
        except Exception as exc:  # noqa: BLE001
            warn(f"cfb: ESPN week {w} unavailable ({type(exc).__name__})")
    return games, st


def load_cfb_odds(game_pks: list[int]) -> pd.DataFrame | None:
    """Last pre-kickoff spread/total capture per (game, market, side, book)."""
    if not config.DATABASE_URL:
        warn("cfb: DATABASE_URL unset -- no odds_snapshot closing lines "
             "(current-season games get SU only)")
        return None
    if not game_pks:
        return None
    with db.get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (game_pk, market, side, book)
                   game_pk, market, side, book, line, commence_time, captured_at
            FROM odds_snapshot
            WHERE game_pk = ANY(%s) AND market IN ('spread', 'total')
              AND player_name = '' AND captured_at < commence_time
            ORDER BY game_pk, market, side, book, captured_at DESC
            """,
            (list(game_pks),),
        )
        rows = cur.fetchall()
    cols = ["game_pk", "market", "side", "book", "line", "commence_time", "captured_at"]
    return live_closing_consensus(pd.DataFrame(rows, columns=cols))


def load_rating() -> tuple[EloConfig, BlendConfig]:
    j = json.loads((CFB_ASSETS / "rating.json").read_text())
    return (EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"],
                      base=j.get("base", 1500.0)),
            BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"]))


def load_cfb_priors(season: int) -> dict:
    """{team: preseason rating (Elo scale)} for ``season`` -- the leak-free v2 prior
    (``cfb.priors.season_priors``: previous-season SP+ + returning / recruiting /
    portal / prior SOS). {} when the priors asset or the season's rows are missing
    (rankings then fall back to carried-over Elo)."""
    path = CFB_ASSETS / "priors.parquet"
    wpath = CFB_ASSETS / "priors_weights.json"
    if not path.exists() or not wpath.exists():
        warn(f"cfb: {path.name} / {wpath.name} missing -- rankings use Elo as the prior")
        return {}
    df = pd.read_parquet(path)
    df = df[df["season"].isin([season - 1, season])]
    rows = {int(y): g.to_dict("records") for y, g in df.groupby("season")}
    if season not in rows:
        return {}
    return season_priors(rows, season, load_weights(wpath))


def _points_from_elo_scale(pri: dict, members) -> dict[str, float]:
    """Elo-scale preseason ratings -> points vs the FBS average (/ 25)."""
    fb = [v for t, v in pri.items() if str(t) in members]
    mu = float(np.mean(fb)) if fb else 0.0
    return {str(t): (float(v) - mu) / 25.0 for t, v in pri.items()}


def cfb_prior_points(season: int, members) -> dict[str, float]:
    """The v2 preseason prior for ``season`` in points vs the FBS average."""
    return _points_from_elo_scale(load_cfb_priors(season), members)


def cfb_results_games(schedules: pd.DataFrame) -> pd.DataFrame:
    s = schedules
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    return played_games(s.assign(neutral=s["neutral_site"].fillna(False).astype(bool)))


def load_cfb_sources(now: pd.Timestamp) -> dict:
    asset = pd.read_parquet(CFB_ASSETS / "schedules.parquet")
    espn_games, st = _espn_games(now)
    sched = merge_espn_schedule(asset, espn_games, st)
    print(f"cfb: schedule {len(asset)} asset rows + {len(sched) - len(asset)} from ESPN")
    season = int(sched["season"].max())
    pks = sched.loc[sched["season"] == season, "game_pk"].astype("int64").tolist()
    advanced = pd.read_parquet(ADVANCED_PATH) if ADVANCED_PATH.exists() else None
    return {"schedules": sched, "lines": pd.read_parquet(CFB_ASSETS / "lines.parquet"),
            "live_close": load_cfb_odds(pks), "advanced": advanced}


def _cfb_season(schedules: pd.DataFrame, now: pd.Timestamp) -> int:
    """Season of the next unplayed game (kickoff >= now), else the latest season."""
    kick = pd.to_datetime(schedules["start_date"], utc=True)
    fut = schedules[schedules["home_score"].isna() & (kick >= now)]
    if fut.empty:
        return int(schedules["season"].max())
    return int(fut.loc[kick[fut.index].idxmin(), "season"])


def _cfb_conf(sched: pd.DataFrame) -> dict:
    s = sched.sort_values("start_date", kind="stable")
    pairs = pd.concat([
        s[["home_team", "home_conf"]].set_axis(["team", "conf"], axis=1),
        s[["away_team", "away_conf"]].set_axis(["team", "conf"], axis=1)])
    pairs = pairs[pairs["conf"].notna() & (pairs["team"] != "FCS")]
    return pairs.drop_duplicates("team", keep="last").set_index("team")["conf"].astype(str).to_dict()


def build_cfb(schedules: pd.DataFrame, lines: pd.DataFrame, live_close: pd.DataFrame | None,
              advanced: pd.DataFrame | None, now: pd.Timestamp,
              fbs=None, priors: dict | None = None,
              rp: ResultsParams | None = None) -> dict[str, pd.DataFrame]:
    now = _utc(now)
    season = _cfb_season(schedules, now)
    sched = schedules[schedules["season"] >= season - LOG_PRIOR_SEASONS]
    log = cfb_game_log(sched, lines, live_close)
    games = pd.DataFrame({
        "game_key": sched["game_pk"].astype(str),
        "game_pk": sched["game_pk"].astype("int64"),
        "home_team": sched["home_team"].astype(str), "away_team": sched["away_team"].astype(str),
        "season": sched["season"].astype(int), "week": sched["week"].astype(int),
    })
    games = games[games["game_key"].isin(set(log["game_key"]))].drop_duplicates("game_key")
    games["kickoff"] = games["game_key"].map(
        log.drop_duplicates("game_key").set_index("game_key")["kickoff"])
    games["home_is_fbs"] = games["home_team"] != "FCS"
    games["away_is_fbs"] = games["away_team"] != "FCS"

    up = upcoming(log, now)
    hist = _history(log, up, games, "cfb")
    if advanced is None:
        warn(f"cfb: {ADVANCED_PATH.relative_to(ROOT)} missing -- matchup grades skipped "
             "and no unit ranks (history, game log and rankings still written)")
        ug, grades = None, _empty("matchup_grades")
    else:
        ug = cfb_unit_games(advanced)
        grades = _grades(up, games, _Ratings(ug), "cfb")

    rk = _empty("power_rankings")
    nxt = _next_unplayed(log, now)
    if nxt is not None:
        s, w = int(nxt["season"]), int(nxt["week"])
        members = {str(t) for t in (fbs if fbs is not None else load_fbs_ids())}
        pre = {y: cfb_prior_points(y, members) for y in range(s - 3, s)}
        pre[s] = (cfb_prior_points(s, members) if priors is None
                  else _points_from_elo_scale(priors, members))
        power, prev = power_asof(cfb_results_games(schedules), rp or load_params("cfb"), s, w,
                                 preseason=pre, members=members)
        ur = unit_ratings_asof(ug, s, w, blend_k=POWER_BLEND_K) if ug is not None else None
        rk = _rank_frame(rankings(power, prev, ur, log), "cfb", _cfb_conf(sched))
    return {"team_game_log": _log_frame(log), "team_history": hist,
            "matchup_grades": grades, "power_rankings": rk}


# --------------------------------------------------------------------- records
def _py(v):
    """numpy / pandas scalar -> DB-ready python value (NaN / NA -> None)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    return v


def table_records(frames: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    out = {}
    for t in TABLES:
        cols = TABLE_COLUMNS[t]
        df = frames[t]
        rows = []
        for rec in df[cols].to_dict("records"):
            r = {c: _py(rec[c]) for c in cols}
            if t == "team_game_log" and r["date_et"] is not None:
                r["date_et"] = pd.Timestamp(r["date_et"]).date()
            rows.append(r)
        out[t] = rows
    return out


# ------------------------------------------------------------------------- CLI
def _rec(w: dict) -> str:
    return f"SU {w['su']} ATS {w['ats']} O/U {w['ou']} (n={w['n']})"


def print_sample(sport: str, frames: dict[str, pd.DataFrame]) -> None:
    hist, gr, rk = frames["team_history"], frames["matchup_grades"], frames["power_rankings"]
    if len(hist):
        first = hist.sort_values(["kickoff", "game_pk"], kind="stable").iloc[0]
        pk = first["game_pk"]
        print(f"[{sport}] sample game {pk} (season {first['season']} week {first['week']}, "
              f"kickoff {first['kickoff']}):")
        for _, h in hist[hist["game_pk"] == pk].sort_values("side").iterrows():
            print(f"  {h['side']:4s} {h['team']} vs {h['opponent']}")
            for k in ("L5", "L10", "L20", "season"):
                print(f"    {k:6s} {_rec(h['windows'][k])}")
            st = h["streaks"]
            print(f"    streaks SU {st['su']} ATS {st['ats']} O/U {st['ou']} notes {st['notes']}")
            g = gr[(gr["game_pk"] == pk) & (gr["side"] == h["side"])]
            if len(g):
                g = g.iloc[0]
                print(f"    grade (offense) overall {g['overall']} pass {g['pass']} run {g['run']} "
                      f"(pct {g['overall_pct']}/{g['pass_pct']}/{g['run_pct']}) early={g['early']}")
    if len(rk):
        print(f"[{sport}] top 5 rankings (season {rk['season'].iloc[0]} week {rk['week'].iloc[0]}):")
        cols = ["rank", "team", "rating", "prev_rank", "move", "sos", "su", "ats"]
        print(rk[cols].head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def run_sport(sport: str, now: pd.Timestamp, dry_run: bool) -> dict[str, int]:
    t0 = time.perf_counter()
    if sport == "nfl":
        frames = build_nfl(**load_nfl_sources(now), now=now)
    else:
        frames = build_cfb(**load_cfb_sources(now), now=now)
    secs = time.perf_counter() - t0
    counts = {t: len(frames[t]) for t in TABLES}
    print(f"[{sport}] runtime {secs:.1f}s; rows: "
          + ", ".join(f"{t}={n}" for t, n in counts.items()))
    if dry_run:
        print_sample(sport, frames)
        return counts
    written = db.upsert_team_context(table_records(frames))
    print(f"[{sport}] upserted: " + ", ".join(f"{t}={n}" for t, n in written.items()))
    return counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sport", choices=["nfl", "cfb", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; no DB writes")
    ap.add_argument("--now", default=None, help="as-of time (ISO, default: now UTC)")
    args = ap.parse_args(argv)
    if not args.dry_run and not config.DATABASE_URL:
        sys.exit("DATABASE_URL is not set (use --dry-run to compute without writing)")
    now = _utc(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    failed = []
    for sport in (["nfl", "cfb"] if args.sport == "all" else [args.sport]):
        try:   # one sport failing must not block the other
            run_sport(sport, now, args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"::error::team-context: {sport} failed: {type(exc).__name__}: {exc}",
                  flush=True)
            failed.append(sport)
    if failed:
        sys.exit(f"team-context failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
