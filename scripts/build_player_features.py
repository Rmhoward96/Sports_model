"""Build the props-ML feature tables (Task 6 of
docs/superpowers/plans/2026-09-24-nfl-props-ml-p1.md).

Fetches nflverse sources for SEASONS and writes
  data/props_ml/player_week_features.parquet  (one row per player-week:
      every played REG skill-player game + active-but-no-snap stubs)
  data/props_ml/team_week_features.parquet    (one row per team-week)
The tables are regenerable and never committed (data/ is gitignored).

Pure seams (unit tested in tests/scripts/test_build_player_features.py):
  active_stubs          -- active depth-chart skill players with no played row
  dropped_snap_mappings -- snap rows lost to a missing pfr -> gsis id mapping
  nan_share_by_group    -- NaN share per feature prefix group
fetch_sources()/build_and_write()/main() are IO (network + parquet writes)
and not unit tested.

Usage:
    uv run python scripts/build_player_features.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.nfl.teams import normalize_team  # noqa: E402

SEASONS = list(range(2016, 2027))
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "props_ml"
SKILL = ("QB", "RB", "WR", "TE")
FEATURE_GROUPS = ("p_", "ngs_", "tm_", "op_", "st_", "cx_", "mk_")
_STUB_COLS = ["player_id", "season", "week", "team", "opponent", "position"]
# pbp columns read by team_games / player_redzone / team_game_epa (the full
# release has ~370 columns; trimming per season keeps the 11-season frame small).
_PBP_COLS = ["season", "week", "season_type", "play_type", "posteam", "defteam", "sack", "qb_hit",
             "wp", "down", "qtr", "yardline_100", "receiver_player_id", "rusher_player_id", "epa"]


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

    Stubs = as-of depth-chart QB/RB/WR/TE rows for (season, week, club_code)
    on a REG team-week in `schedules` (which supplies the opponent), minus
    players whose injury `report_status` is Out or Doubtful that (season,
    week) (matched by gsis_id), minus players already in `pg` for that
    (season, week). Team codes are normalized. Columns: player_id (gsis),
    season, week, team, opponent, position; one row per player-week.
    """
    if depth is None or not len(depth):
        return pd.DataFrame(columns=_STUB_COLS)
    d = depth[depth["position"].isin(SKILL)]
    d = d.assign(team=d["club_code"].map(_norm), player_id=d["gsis_id"])
    d = d[d["team"].notna() & d["player_id"].notna()]
    d = d[["player_id", "season", "week", "team", "position"]].astype({"season": int, "week": int})
    d = d.merge(_reg_team_weeks(schedules), on=["season", "week", "team"], how="inner")
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


def fetch_sources() -> dict:
    """IO: every nflverse input for SEASONS (network)."""
    import nfl_data_py as nfl

    from sportsmodel.nfl.nflverse import import_by_season, load_release
    from sportsmodel.sim.nfl.usage import build_pfr_to_gsis, depth_charts_asof

    sched = load_release("schedules", SEASONS)
    return {
        "sched": sched,
        "pbp": _load_pbp(SEASONS),
        "weekly": load_release("weekly", SEASONS),
        "snaps": load_release("snaps", SEASONS),
        "depth": depth_charts_asof(load_release("depth", SEASONS), sched),
        "injuries": import_by_season(nfl.import_injuries, SEASONS, "injuries", required=False),
        "ngs": {k: load_release(f"ngs_{v}", SEASONS, required=False)
                for k, v in (("rec", "receiving"), ("rush", "rushing"), ("pass", "passing"))},
        "pfr2gsis": build_pfr_to_gsis(nfl.import_ids()),
    }


def build_and_write(src: dict, t0: float, t_fetch: float) -> None:
    """Build both tables from fetched sources, write the parquets, print the summary."""
    from sportsmodel.nfl.context import load_stadiums, team_game_context
    from sportsmodel.nfl.efficiency import team_game_epa
    from sportsmodel.nfl.player_features import (
        build_feature_table, build_team_table, player_games, player_redzone, team_games,
    )

    sched, pbp, injuries, depth = src["sched"], src["pbp"], src["injuries"], src["depth"]
    dropped = dropped_snap_mappings(src["snaps"], src["pfr2gsis"])
    pg = player_games(src["weekly"], src["snaps"], src["pfr2gsis"])
    tg = team_games(pbp)
    rz = player_redzone(pbp)
    ctx = team_game_context(sched, load_stadiums())
    game_epa = team_game_epa(pbp)
    stubs = active_stubs(depth, injuries, pg, sched)
    print(f"sources ready ({t_fetch:.1f}s): {len(pg)} player-games, {len(tg)} team-games, {len(stubs)} stubs; "
          "building feature tables...", flush=True)

    feats = build_feature_table(pg, tg, rz, ctx, src["ngs"], injuries, depth, game_epa, stubs=stubs)
    team = build_team_table(tg, ctx, game_epa)

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
    print(f"feature columns: {sum(c.startswith(FEATURE_GROUPS) for c in feats.columns)}; "
          f"written to {OUT_DIR}")
    print(f"build time: {elapsed:.1f}s (fetch {t_fetch:.1f}s, features {elapsed - t_fetch:.1f}s)")


def main() -> None:
    t0 = time.monotonic()
    src = fetch_sources()
    build_and_write(src, t0, time.monotonic() - t0)


if __name__ == "__main__":
    main()
