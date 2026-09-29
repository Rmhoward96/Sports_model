"""Build the props-ML feature tables (Task 6 of
docs/superpowers/plans/2026-09-24-nfl-props-ml-p1.md).

Fetches nflverse sources for SEASONS and writes
  data/props_ml/player_week_features.parquet  (one row per player-week:
      every played REG skill-player game + active-but-no-snap stubs)
  data/props_ml/team_week_features.parquet    (one row per team-week)
The tables are regenerable and never committed (data/ is gitignored).

Pure seams (unit tested in tests/scripts/test_build_player_features.py):
  active_stubs            -- active depth-chart skill players with no played row
                             (same exact-week-else-latest-earlier chart rule as
                             usage.active_usage)
  chart_coverage          -- REG team-weeks by chart used (exact / fallback / none)
  depth_rank_distribution -- per-season p_depth_rank shares for one position
  dropped_snap_mappings   -- snap rows lost to a missing pfr -> gsis id mapping
  nan_share_by_group      -- NaN share per feature prefix group
  qb_params               -- (H, k) of the QB profiles from qb_profile_params.json
build_tables (no IO beyond the committed params asset) is shared with live
serving (generate_sim_nfl). It adds the mx_/di_/qb_ features
(`player_features.extra_features`) to both tables.
fetch_sources()/build_and_write()/main() are IO (network + parquet writes)
and not unit tested.

Usage:
    uv run python scripts/build_player_features.py
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.nfl.teams import normalize_team  # noqa: E402

SEASONS = list(range(2016, 2027))
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "props_ml"
QB_PARAMS_PATH = Path(__file__).resolve().parents[1] / "assets" / "nfl" / "props_ml" / "qb_profile_params.json"
QB_FIRST_SEASON = 1999     # QB profiles use every QB game from 1999 on
SKILL = ("QB", "RB", "WR", "TE")
FEATURE_GROUPS = ("p_", "ngs_", "tm_", "op_", "st_", "cx_", "mk_", "mx_", "di_", "qb_")
_STUB_COLS = ["player_id", "season", "week", "team", "opponent", "position"]
# pbp columns read by team_games / player_redzone / team_game_epa / unit_games
# (the full release has ~370 columns; trimming per season keeps the 11-season
# frame small).
_PBP_COLS = ["season", "week", "season_type", "play_type", "posteam", "defteam", "sack", "qb_hit",
             "wp", "down", "qtr", "yardline_100", "receiver_player_id", "rusher_player_id", "epa",
             "yards_gained", "success", "pass_touchdown", "rush_touchdown"]


def _norm(code) -> str | None:
    if pd.isna(code):
        return None
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def _reg_team_weeks(schedules: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team, opponent) for both sides of every REG game."""
    g = schedules[schedules["game_type"] == "REG"]
    home, away = g["home_team"].map(_norm), g["away_team"].map(_norm)
    sw = g[["season", "week"]].astype(int)
    tw = pd.concat([sw.assign(team=home, opponent=away), sw.assign(team=away, opponent=home)], ignore_index=True)
    return tw.dropna(subset=["team", "opponent"]).drop_duplicates(["season", "week", "team"])


def active_stubs(depth: pd.DataFrame, injuries: pd.DataFrame, pg: pd.DataFrame,
                 schedules: pd.DataFrame) -> pd.DataFrame:
    """Active-but-no-snap skill players per REG team-week. PURE.

    Stubs = QB/RB/WR/TE rows of the depth chart `usage.active_usage` uses
    for each REG team-week in `schedules` (which supplies the opponent): the
    exact week's chart, else the team's latest earlier chart
    (`usage.chart_weeks_asof`, the same fallback rule), minus players whose
    injury `report_status` is Out or Doubtful that (season, week) (matched by
    gsis_id), minus players already in `pg` for that (season, week). Team
    codes are normalized. Columns: player_id (gsis), season, week, team,
    opponent, position; one row per player-week.
    """
    from sportsmodel.sim.nfl.usage import chart_weeks_asof

    if depth is None or not len(depth):
        return pd.DataFrame(columns=_STUB_COLS)
    dn = depth.assign(club_code=depth["club_code"].map(_norm)).dropna(subset=["club_code", "season", "week"])
    dn = dn.astype({"season": int, "week": int})
    tw = _reg_team_weeks(schedules).reset_index(drop=True)
    cw = chart_weeks_asof(dn, tw).assign(opponent=tw["opponent"].to_numpy()).dropna(subset=["chart_season"])
    cw = cw.astype({"chart_season": int, "chart_week": int})
    d = dn[dn["position"].isin(SKILL) & dn["gsis_id"].notna()]
    d = pd.DataFrame({"player_id": d["gsis_id"], "chart_season": d["season"], "chart_week": d["week"],
                      "team": d["club_code"], "position": d["position"]})
    d = cw.merge(d, on=["team", "chart_season", "chart_week"], how="inner")
    d = d[["player_id", "season", "week", "team", "opponent", "position"]].astype({"season": int, "week": int})
    if injuries is not None and len(injuries):
        out = injuries[injuries["report_status"].isin(["Out", "Doubtful"])].dropna(subset=["gsis_id", "season", "week"])
        out = out.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "week"]]
        out = out.astype({"season": int, "week": int}).drop_duplicates()
        d = d.merge(out.assign(_out=1), on=["player_id", "season", "week"], how="left")
        d = d[d["_out"].isna()].drop(columns="_out")
    if pg is not None and len(pg):
        played = pg[["player_id", "season", "week"]].astype({"season": int, "week": int}).drop_duplicates()
        d = d.merge(played.assign(_pl=1), on=["player_id", "season", "week"], how="left")
        d = d[d["_pl"].isna()].drop(columns="_pl")
    return d.drop_duplicates(["player_id", "season", "week"])[_STUB_COLS].reset_index(drop=True)


def chart_coverage(depth: pd.DataFrame, schedules: pd.DataFrame) -> dict[str, int]:
    """REG team-weeks by which depth chart active_usage/active_stubs use:
    `exact` week, `fallback` (latest earlier chart), or `none`. PURE."""
    from sportsmodel.sim.nfl.usage import chart_weeks_asof

    tw = _reg_team_weeks(schedules)
    dn = depth.assign(club_code=depth["club_code"].map(_norm)).dropna(subset=["club_code", "season", "week"])
    cw = chart_weeks_asof(dn, tw)
    none = cw["chart_season"].isna()
    exact = (cw["chart_season"] == cw["season"]) & (cw["chart_week"] == cw["week"])
    return {"exact": int(exact.sum()), "fallback": int((~exact & ~none).sum()), "none": int(none.sum())}


def depth_rank_distribution(feats: pd.DataFrame, position: str = "WR") -> pd.DataFrame:
    """Per season: share of `position` rows with p_depth_rank 1, 2, 3, 4, 5+
    or NaN, plus row count and mean rank. PURE."""
    f = feats[feats["position"] == position]
    r = f["p_depth_rank"]
    bucket = r.clip(upper=5).map(lambda v: "nan" if pd.isna(v) else ("5+" if v >= 5 else str(int(v))))
    tab = pd.crosstab(f["season"], bucket, normalize="index")
    tab = tab.reindex(columns=["1", "2", "3", "4", "5+", "nan"], fill_value=0.0).round(3)
    tab.insert(0, "rows", f.groupby("season").size())
    tab["mean"] = r.groupby(f["season"]).mean().round(2)
    tab.columns.name = None
    return tab


def dropped_snap_mappings(snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> dict[int, int]:
    """Per season: REG skill-player snap rows with offense_snaps > 0 (the rows
    player_games keeps) whose pfr_player_id has no gsis mapping. PURE."""
    s = snaps[(snaps["game_type"] == "REG") & (snaps["offense_snaps"] > 0) & snaps["position"].isin(SKILL)]
    unmapped = s["pfr_player_id"].map(pfr2gsis).isna()
    counts = unmapped.groupby(s["season"].astype(int)).sum()
    seasons = sorted(snaps["season"].dropna().astype(int).unique())
    return {int(yr): int(counts.get(yr, 0)) for yr in seasons}


def nan_share_by_group(df: pd.DataFrame) -> dict[str, float]:
    """NaN share over all cells of each feature prefix group (NaN if the
    group has no columns). PURE."""
    res: dict[str, float] = {}
    for pre in FEATURE_GROUPS:
        cols = [c for c in df.columns if c.startswith(pre)]
        if not cols or not len(df):
            res[pre] = float("nan")
            continue
        sub = df[cols]
        res[pre] = float(sub.isna().to_numpy().sum()) / sub.size
    return res


def qb_params(mode: str = "gate", path: Path | None = None) -> tuple[float, float]:
    """(H, k) of the QB profiles: the `gate` block (fit on 2021-2023) or the
    `serving` block (fit on 2021 -> latest) of qb_profile_params.json."""
    if mode not in ("gate", "serving"):
        raise ValueError(f"qb_params_mode must be 'gate' or 'serving', got {mode!r}")
    path = QB_PARAMS_PATH if path is None else Path(path)
    block = json.loads(Path(path).read_text()).get(mode)
    if block is None:
        raise RuntimeError(f"{path} has no {mode!r} block -- run "
                           f"`uv run python scripts/tune_qb_profile.py --mode {mode}` first")
    return float(block["H"]), float(block["k"])


def _load_pbp(seasons: list[int]) -> pd.DataFrame:
    """pbp season by season, trimmed to _PBP_COLS (memory)."""
    from sportsmodel.nfl.nflverse import load_release

    frames = []
    for yr in seasons:
        f = load_release("pbp", [yr], required=False)
        if len(f):
            frames.append(f[[c for c in _PBP_COLS if c in f.columns]])
    if not frames:
        raise RuntimeError(f"nflverse pbp: no seasons available from {seasons}")
    return pd.concat(frames, ignore_index=True)


def fetch_sources(seasons: list[int] | None = None) -> dict:
    """IO: every nflverse input for `seasons` (default SEASONS; network).
    Live serving (generate_sim_nfl's SIM_ML_MODE path) passes
    SEASONS[0]..current season. `weekly_qb` = the QB rows of weekly stats
    1999..max(seasons) (the requested seasons' frame is reused; only the
    earlier history is downloaded again); `qb_params_mode` defaults to
    "gate" (set "serving" to use the serving (H, k))."""
    import nfl_data_py as nfl

    from sportsmodel.nfl.nflverse import import_by_season, load_release
    from sportsmodel.sim.nfl.usage import build_pfr_to_gsis, depth_charts_asof

    seasons = SEASONS if seasons is None else list(seasons)
    sched = load_release("schedules", seasons)
    weekly = load_release("weekly", seasons)
    hist = [y for y in range(QB_FIRST_SEASON, max(seasons) + 1) if y not in seasons]
    qb_frames = [weekly[weekly["position"] == "QB"]]
    if hist:
        h = load_release("weekly", hist)
        qb_frames.insert(0, h[h["position"] == "QB"])
    return {
        "sched": sched,
        "pbp": _load_pbp(seasons),
        "weekly": weekly,
        "weekly_qb": pd.concat(qb_frames, ignore_index=True),
        "qb_params_mode": "gate",
        "snaps": load_release("snaps", seasons),
        "depth": depth_charts_asof(load_release("depth", seasons), sched),
        "injuries": import_by_season(nfl.import_injuries, seasons, "injuries", required=False),
        "ngs": {k: load_release(f"ngs_{v}", seasons, required=False)
                for k, v in (("rec", "receiving"), ("rush", "rushing"), ("pass", "passing"))},
        "pfr2gsis": build_pfr_to_gsis(nfl.import_ids()),
    }


def build_tables(src: dict, ctx_fill: Callable[[pd.DataFrame, dict, pd.DataFrame], pd.DataFrame] | None = None,
                 qb1_override: dict | None = None) -> dict:
    """Both feature tables from fetched sources (no network). The one builder
    shared by the parquet build (training) and live serving: serving passes
    `ctx_fill(ctx, stadiums, sched)` (e.g. context.fill_forecast_weather with
    a fetcher) to fill upcoming games' forecast weather before both tables
    are built. The mx_/di_/qb_ features (`player_features.extra_features`)
    use the QB-profile (H, k) of the `src["qb_params_mode"]` block ("gate"
    by default) and `qb1_override` {(season, week, team): gsis_id} for QB1.
    Returns {"feats", "team", "pg", "tg", "stubs", "extra"}."""
    from sportsmodel.nfl import context, efficiency, player_features

    sched, pbp, injuries, depth = src["sched"], src["pbp"], src["injuries"], src["depth"]
    pg = player_features.player_games(src["weekly"], src["snaps"], src["pfr2gsis"])
    tg = player_features.team_games(pbp)
    rz = player_features.player_redzone(pbp)
    stadiums = context.load_stadiums()
    ctx = context.team_game_context(sched, stadiums)
    if ctx_fill is not None:
        ctx = ctx_fill(ctx, stadiums, sched)
    game_epa = efficiency.team_game_epa(pbp)
    stubs = active_stubs(depth, injuries, pg, sched)
    H, k = qb_params(src.get("qb_params_mode", "gate"), QB_PARAMS_PATH)
    extra = player_features.extra_features(player_features.team_week_keys(tg, ctx), pbp, src["snaps"],
                                           src["pfr2gsis"], injuries, depth, src["weekly_qb"], H, k,
                                           qb1_override=qb1_override)
    feats = player_features.build_feature_table(pg, tg, rz, ctx, src["ngs"], injuries, depth, game_epa,
                                                stubs=stubs, extra=extra)
    team = player_features.build_team_table(tg, ctx, game_epa, extra=extra)
    return {"feats": feats, "team": team, "pg": pg, "tg": tg, "stubs": stubs, "extra": extra}


def build_and_write(src: dict, t0: float, t_fetch: float) -> None:
    """Build both tables from fetched sources, write the parquets, print the summary."""
    sched, depth = src["sched"], src["depth"]
    dropped = dropped_snap_mappings(src["snaps"], src["pfr2gsis"])
    print(f"sources ready ({t_fetch:.1f}s); building feature tables...", flush=True)
    built = build_tables(src)
    feats, team, stubs = built["feats"], built["team"], built["stubs"]
    print(f"{len(built['pg'])} player-games, {len(built['tg'])} team-games, {len(stubs)} stubs", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT_DIR / "player_week_features.parquet", index=False)
    team.to_parquet(OUT_DIR / "team_week_features.parquet", index=False)
    elapsed = time.monotonic() - t0

    played = feats["y_targets"].notna()
    print("\n=== props-ML feature build ===")
    print(f"snap rows dropped (pfr_player_id with no gsis mapping), per season: {dropped} "
          f"(total {sum(dropped.values())})")
    print(f"player-week rows: {len(feats)} (played {int(played.sum())}, stubs {int((~played).sum())}); "
          f"active_stubs produced {len(stubs)}")
    per_season = feats.assign(_played=played).groupby("season")["_played"].agg(rows="size", played="sum")
    per_season["stubs"] = per_season["rows"] - per_season["played"]
    print("rows per season:\n" + per_season.astype(int).to_string())
    print(f"team-week rows: {len(team)} ({team['y_team_pass_att'].notna().sum()} played)")
    shares = nan_share_by_group(feats)
    print("NaN share per feature group (player table): "
          + ", ".join(f"{k}={v:.3f}" for k, v in shares.items()))
    tshares = nan_share_by_group(team)
    print("NaN share, new groups (team table): "
          + ", ".join(f"{k}={tshares[k]:.3f}" for k in ("mx_", "di_", "qb_")))
    cov = chart_coverage(depth, sched)
    print(f"REG team-weeks by depth chart used (active_usage rule): exact {cov['exact']}, "
          f"fallback to an earlier chart {cov['fallback']}, none {cov['none']}")
    print("WR p_depth_rank (rank within team-week-position) share by season:\n"
          + depth_rank_distribution(feats, "WR").to_string())
    print(f"feature columns: {sum(c.startswith(FEATURE_GROUPS) for c in feats.columns)}; "
          f"written to {OUT_DIR}")
    print(f"build time: {elapsed:.1f}s (fetch {t_fetch:.1f}s, features {elapsed - t_fetch:.1f}s)")


def main() -> None:
    t0 = time.monotonic()
    src = fetch_sources()
    build_and_write(src, t0, time.monotonic() - t0)


if __name__ == "__main__":
    main()
