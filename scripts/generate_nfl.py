"""Generate game + prop predictions for the current NFL slate (LIVE producer).

Analog of `generate_sim.py` for NFL: assembles per-game predictions from the
P1 Elo/SoS margin model + P2 opponent-adjusted points/gameline model + the P3
usage/efficiency prop engine, and writes `game_predictions`/`prop_predictions`
rows tagged `sport='nfl'`.

    P1 (elo/srs/ratings):     pre-game Elo + season-to-date SRS -> model_margin
    P2 (points/gameline):     opponent-adjusted points -> model_total; shrink
                              toward the market line -> serving dists
    P3 (gamescript/usage/
        efficiency/props):    model margin/total -> team play volume -> per-
                              player share of that volume -> prop dists

`build_game_row`/`build_prop_rows` are pure (all inputs injected) and unit
tested directly; `main()`'s ESPN/DB I/O is the thin live wrapper that feeds
them from the committed P1-P3 assets + a live ESPN schedule/inactives pull.

Data sources (all committed snapshots so the pure assembly is testable/CI-safe):
  - ratings/gameline/props configs: assets/nfl/{rating,gameline,props}.json
  - historical schedule/weekly/rosters/injuries: assets/nfl/*.parquet
  - live schedule + inactives: ESPN scoreboard/summary (espn.py)
  - market line: Supabase odds_snapshot (falls back to model-only if unset/empty)
  - output: Supabase game_predictions/prop_predictions (sport='nfl'), only if
            DATABASE_URL is set (mirrors generate_sim.py)

Usage:
    uv run python scripts/generate_nfl.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel import config
from sportsmodel.db import upsert_game_predictions, upsert_prop_predictions
from sportsmodel.nfl import config as nfl_config
from sportsmodel.nfl import (
    efficiency, espn, gameline, gamescript, points, props, ratings, srs,
    universe, usage,
)
from sportsmodel.nfl.elo import run_elo

GAME_MODEL_VERSION = "nfl-elo-v1"
PROP_MODEL_VERSION = "nfl-props-v1"

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "nfl"


def _load_committed(name: str) -> pd.DataFrame:
    return pd.read_parquet(_ASSETS / name)


def _norm_name(name: str) -> str:
    return " ".join((name or "").lower().split())


def _game_date_from_commence(commence_iso: str) -> str:
    """US game date from an ESPN UTC `commence_time`.

    NFL kickoffs run from ~13:00 UTC (early Sunday/int'l window) to ~00:20-
    01:20 UTC the NEXT day (SNF/MNF). An 8h shift maps every real kickoff
    time back into its true US game day without needing a timezone lookup
    (mirrors `ingest.odds.resolved_game_date`'s MLB-tuned version of the same
    trick, retuned for NFL's kickoff window instead of MLB's).
    """
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    return (dt - timedelta(hours=8)).date().isoformat()


def build_game_row(game: dict, ctx: dict, gl_cfg: "gameline.GameLineConfig") -> dict:
    """Pure: model margin/total + market line -> a `game_predictions`-shaped row.

    `ctx` = {"model_margin", "model_total", "market": {"spread_line","total_line"},
    "week"}. `margin_dist`/`total_dist` are returned as plain dicts (not JSON) so
    callers can inspect them directly (see the unit test); the live `main()`
    JSON-encodes them right before the DB upsert, same boundary generate_sim.py
    uses.
    """
    row = gameline.build_gameline(ctx["model_margin"], ctx["model_total"],
                                  ctx["market"], ctx["week"], gl_cfg)
    return {
        **row,
        "sport": "nfl",
        "model_version": GAME_MODEL_VERSION,
        "game_pk": game["game_pk"],
        "game_date": game["game_date"],
        "home_team_name": game["home_name"],
        "away_team_name": game["away_name"],
    }


def _markets_for(position: str, volume: dict) -> list[str]:
    """Position -> applicable prop markets, gated on actually-projected volume
    (no point serving a reception market to a player projected for 0 targets).

    QB: pass_yds/pass_tds only (rushing QB props are out of scope for v1).
    RB/HB/FB: the full rushing+receiving mix, including the combined
    rush_reception_yds market real books offer running backs.
    WR/TE: receiving markets only (rush_yds/rush_reception_yds are not
    normally offered for these positions).
    """
    markets: list[str] = []
    if position == "QB":
        if volume["pass_att"] > 0:
            markets += ["pass_yds", "pass_tds"]
        return markets
    if volume["targets"] > 0:
        markets += ["reception_yds", "receptions"]
    if position in ("RB", "HB", "FB") and volume["carries"] > 0:
        markets.append("rush_yds")
    if position in ("RB", "HB", "FB") and (volume["carries"] > 0 or volume["targets"] > 0):
        markets.append("rush_reception_yds")
    if volume["carries"] > 0 or volume["targets"] > 0:
        markets.append("anytime_td")
    return markets


def _gsis_to_espn_crosswalk(rosters_df: pd.DataFrame) -> dict[str, int]:
    """{gsis player_id (str) -> int ESPN athlete id} from the committed
    rosters snapshot, deduped to the LATEST season a given gsis id appears in
    (a player can recur across seasons; the newest row is the freshest link).

    `prop_predictions`/`board_picks`/`picks`/`prediction_results` all store
    `player_id` as BIGINT, but the internal player id used everywhere else in
    the NFL pipeline (usage shares, efficiency, universe) is the nflverse/GSIS
    id -- a string like "00-0034473" that can't be written to a BIGINT column.
    The ESPN athlete id is numeric and fits BIGINT, so it's what actually gets
    served; this crosswalk is how a gsis-keyed player picks up the int id it
    needs to be written. `nfl_results.fetch_results` keys its graded results
    by that same int ESPN id directly (no crosswalk needed on that side), so
    the written `player_id` and the graded lookup agree without any
    remapping at grade time.

    Rows with a missing/non-numeric espn_id are dropped from the map -- a
    player with no numeric ESPN id has no valid BIGINT to write, so
    `build_prop_rows` skips them entirely rather than crash or write garbage.
    """
    df = rosters_df.dropna(subset=["espn_id"])
    df = df.sort_values("season").drop_duplicates("player_id", keep="last")
    out: dict[str, int] = {}
    for pid, espn_id in zip(df["player_id"], df["espn_id"]):
        try:
            out[str(pid)] = int(espn_id)
        except (TypeError, ValueError):
            continue
    return out


def build_prop_rows(game: dict, universe_list: list[dict], usage_shares: dict,
                    eff: dict, team_volume: dict, props_cfg,
                    gsis_to_espn: dict) -> list[dict]:
    """Pure: for each ACTIVE player in `universe_list` with a usage share and
    efficiency rates, allocate their share of their team's projected volume
    and build every applicable market's distribution.

    Players not in `universe_list` (inactive/OUT, already filtered by
    `universe.active_universe`) never get a row here -- this is the only
    place that decides which players get served, and it only ever iterates
    the given universe.

    `gsis_to_espn` (from `_gsis_to_espn_crosswalk`) maps the internal gsis
    `player_id` to the int ESPN athlete id that actually gets WRITTEN as each
    row's `player_id` -- the serving columns are BIGINT and can't hold the
    gsis string. A player missing from the crosswalk (no numeric espn_id) is
    SKIPPED entirely: there's no numeric id to write for them.
    """
    rows = []
    for player in universe_list:
        pid = player["player_id"]
        share = usage_shares.get(pid)
        if share is None:
            continue
        e = eff.get(pid)
        if e is None:
            continue
        espn_id = gsis_to_espn.get(pid)
        if espn_id is None:
            continue
        team = share.get("team") or player.get("team")
        tv = team_volume.get(team)
        if tv is None:
            continue
        vol = usage.allocate(share, tv)
        position = share.get("position") or player.get("position")
        player_name = player.get("player_name") or share.get("player_name")
        for market in _markets_for(position, vol):
            built = props.build_prop(market, vol, e, props_cfg)
            rows.append({
                "sport": "nfl",
                "model_version": PROP_MODEL_VERSION,
                "game_pk": game["game_pk"],
                "game_date": game["game_date"],
                "player_id": espn_id,
                "player_name": player_name,
                "team_name": team,
                "market": market,
                "projected_mean": built["projected_mean"],
                "dist": built["dist"],
                "line": None,
            })
    return rows


def _out_player_ids(rosters_df: pd.DataFrame, injuries_df: pd.DataFrame,
                    espn_inactives, season: int, week: int) -> set:
    """The player_ids `universe.active_universe` excluded this week -- i.e.
    the mirror image of that function's filtering, needed here so their
    usage share can be redistributed instead of just vanishing.

    Sourced the same two ways active_universe checks: the committed
    injuries snapshot's `report_status == "Out"` for (season, week), and the
    live ESPN inactives list (matched by normalized name against the full
    roster, since an OUT player is by definition not in the active universe
    and so can't be looked up there).
    """
    out_ids = set()
    if injuries_df is not None and len(injuries_df):
        inj = injuries_df[(injuries_df["season"] == season) & (injuries_df["week"] == week)]
        out_ids |= set(inj[inj["report_status"] == "Out"]["gsis_id"])
    inactive_names = {_norm_name(n) for n in (espn_inactives or [])}
    if inactive_names:
        for _, r in rosters_df.iterrows():
            if _norm_name(r["player_name"]) in inactive_names:
                out_ids.add(r["player_id"])
    return out_ids


def redistribute_out_shares(usage_shares: dict, out_ids: set,
                            universe_list: list[dict]) -> dict:
    """Real backup-bump: give an OUT starter's usage share to the remaining
    same-team/same-position ACTIVE players, split PROPORTIONALLY to each
    survivor's OWN existing share -- not by depth-chart rank. The committed
    `assets/nfl/rosters.parquet`'s `depth_chart_position` column stores the
    position CODE (e.g. every KC WR row reads `"WR"`, not a numeric depth
    rank), so a depth-rank sort ties every same-position player and its pick
    is arbitrary row order -- not a real depth chart. `universe.bump_backup`
    only REMOVES the OUT player from the roster list (a marker); this is the
    parked share-transfer seam it leaves for the producer to implement.

    Redistribution is done independently PER SHARE DIMENSION
    (target_share/carry_share/pass_att_share), so e.g. a receiving-back
    specialist backup picks up proportionally more of an OUT bruiser's
    TARGET share than a pure between-the-tackles backup would, and vice
    versa for CARRY share. If no same-team/same-position survivor has ANY
    existing share in a given dimension, that dimension of the OUT player's
    share is DROPPED (conservative -- no legitimate player to route it to)
    rather than guessed at.
    """
    shares = {pid: dict(s) for pid, s in usage_shares.items()}
    by_slot: dict[tuple, list[str]] = {}
    for p in universe_list:
        by_slot.setdefault((p["team"], p["position"]), []).append(p["player_id"])

    fields = ("target_share", "carry_share", "pass_att_share")
    for pid in out_ids:
        s = shares.get(pid)
        if s is None:
            continue
        survivors = [sid for sid in by_slot.get((s.get("team"), s.get("position")), [])
                    if sid != pid and sid in shares]
        if not survivors:
            continue
        for f in fields:
            out_val = s.get(f, 0.0)
            if out_val <= 0:
                continue
            weights = {sid: shares[sid].get(f, 0.0) for sid in survivors}
            total = sum(weights.values())
            if total <= 0:
                continue  # no survivor has any existing share in this dimension -> drop it
            for sid, w in weights.items():
                shares[sid][f] = shares[sid].get(f, 0.0) + out_val * (w / total)
    return shares


def _latest_market_line(game_pk: int) -> dict:
    """Latest captured spread/total line for `game_pk` from `odds_snapshot`,
    or `{"spread_line": None, "total_line": None}` if nothing has been
    captured yet -- `build_gameline` treats `None` as model-only.

    SIGN CONVENTION: `odds_snapshot` stores the raw sportsbook (Odds API)
    convention -- side='home' `line` is NEGATIVE when the home team is
    favored (e.g. -1.5; see grade_results.py's own "home_line is the home
    team's spread point" comment). `build_gameline`/`shrink` and the whole
    P1/P2 pipeline instead use nflverse's `spread_line` convention --
    POSITIVE `spread_line` = home favored, i.e. the SAME sign as
    `model_margin` (positive = home favored). These are opposite, so the
    raw sportsbook home line must be negated before it's usable as
    `market["spread_line"]` here. Totals have no such sign ambiguity.
    """
    from sportsmodel.db import get_postgres
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT market, side, line FROM odds_snapshot "
            "WHERE game_pk = %s AND market IN ('spread', 'total') "
            "ORDER BY captured_at DESC",
            [game_pk],
        )
        rows = cur.fetchall()
    spread_line = total_line = None
    for market, side, line in rows:
        if line is None:
            continue
        if market == "spread" and side == "home" and spread_line is None:
            spread_line = -float(line)   # sportsbook home-favored-negative -> nflverse home-favored-positive
        elif market == "total" and total_line is None:
            total_line = float(line)
    return {"spread_line": spread_line, "total_line": total_line}


def _resolve_season_week() -> tuple[int, int, int]:
    tw = espn.resolve_target_week()
    return int(tw["season"]), int(tw["week"]), int(tw["season_type"])


def main() -> None:
    season, week, season_type = _resolve_season_week()

    elo_cfg, blend_cfg = nfl_config.load_rating()
    gl_cfg = nfl_config.load_gameline()
    props_cfg = nfl_config.load_props()

    sched = _load_committed("schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    played = reg[(reg["season"] < season) | ((reg["season"] == season) & (reg["week"] < week))].copy()
    played = played.dropna(subset=["home_score", "away_score"])

    elo_final = run_elo(played, elo_cfg).final if len(played) else {}
    srs_now = srs.compute_srs(played) if len(played) else {}
    points_ratings, lg_avg = points.compute_points_ratings(played) if len(played) else ({}, 44.0)
    games_played: dict[str, int] = {}
    for _, g in played.iterrows():
        games_played[g["home_team"]] = games_played.get(g["home_team"], 0) + 1
        games_played[g["away_team"]] = games_played.get(g["away_team"], 0) + 1

    weekly_all = _load_committed("weekly.parquet")
    gs_model = gamescript.fit_gamescript(gamescript.team_game_volume(weekly_all, sched))

    latest_weekly_season = int(weekly_all["season"].max())
    weekly_latest = weekly_all[weekly_all["season"] == latest_weekly_season]
    usage_shares = usage.compute_usage_shares(weekly_latest)
    eff = efficiency.compute_efficiency(weekly_latest)

    rosters_all = _load_committed("rosters.parquet")
    latest_roster_season = int(rosters_all["season"].max())
    roster_now = rosters_all[rosters_all["season"] == latest_roster_season]
    if "week" in roster_now.columns and len(roster_now):
        # Per-TEAM latest week, not a single global latest week: nflverse's
        # weekly roster snapshot tracks each team only through the weeks it
        # actually played, so a team eliminated in the Wild Card round has no
        # rows past week ~19 while the Super Bowl teams run through week 22.
        # A single global-max week would silently keep only the last 2-8
        # teams still alive that deep into the playoffs.
        latest_week = roster_now.groupby("team")["week"].transform("max")
        roster_now = roster_now[roster_now["week"] == latest_week]
    injuries_all = _load_committed("injuries.parquet")
    gsis_to_espn = _gsis_to_espn_crosswalk(rosters_all)

    espn_games = espn.fetch_schedule(season, week, season_type=season_type)

    game_rows, prop_rows = [], []
    for g in espn_games:
        inactives = espn.fetch_inactives(g["game_pk"])
        uni = universe.active_universe(roster_now, injuries_all, inactives, season, week)
        out_ids = _out_player_ids(roster_now, injuries_all, inactives, season, week)
        adj_shares = redistribute_out_shares(usage_shares, out_ids, uni)

        elo_h = elo_final.get(g["home_team"], elo_cfg.base)
        elo_a = elo_final.get(g["away_team"], elo_cfg.base)
        model_margin = ratings.expected_margin(
            elo_h, elo_a, srs_now.get(g["home_team"]), srs_now.get(g["away_team"]),
            games_played.get(g["home_team"], 0), games_played.get(g["away_team"], 0),
            elo_cfg, blend_cfg)
        model_total = points.expected_total(points_ratings, lg_avg, g["home_team"], g["away_team"])
        market = _latest_market_line(g["game_pk"]) if config.DATABASE_URL else {
            "spread_line": None, "total_line": None}
        ctx = {"model_margin": model_margin, "model_total": model_total,
              "market": market, "week": week}

        game_for_row = {**g, "game_date": _game_date_from_commence(g["commence_time"])}
        row = build_game_row(game_for_row, ctx, gl_cfg)

        home_tv = gamescript.project_team_volume(gs_model, row["pred_margin"], row["pred_home_score"])
        away_tv = gamescript.project_team_volume(gs_model, -row["pred_margin"], row["pred_away_score"])
        team_volume = {g["home_team"]: home_tv, g["away_team"]: away_tv}

        game_rows.append({
            **row,
            "margin_dist": json.dumps(row["margin_dist"]),
            "total_dist": json.dumps(row["total_dist"]),
        })

        game_prop_rows = build_prop_rows(game_for_row, uni, adj_shares, eff, team_volume,
                                         props_cfg, gsis_to_espn)
        for r in game_prop_rows:
            r["dist"] = json.dumps(r["dist"])
        prop_rows.extend(game_prop_rows)

    if config.DATABASE_URL:
        upsert_game_predictions(game_rows)
        upsert_prop_predictions(prop_rows)

    print(f"predicted {len(game_rows)} games, {len(prop_rows)} prop rows")


if __name__ == "__main__":
    main()
