"""Tests for sportsmodel.cfb.walkforward.raw_model_predictions.

(a) parity: every key the old `backtest_cfb_gameline._raw_model_predictions`
    returned is unchanged (a frozen verbatim copy of the old function lives
    below as the reference), and the backtest script now uses the shared one;
(b) leak test: appending a later game never changes an earlier row;
plus the new pass-through / pre-game rating fields.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sportsmodel.cfb import walkforward
from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.srs import compute_srs

ELO_CFG = EloConfig(k=40, hfa_elo=70, carryover=0.9, base=1500.0)
BLEND_CFG = BlendConfig(w_sos=0.45, srs_min_games=3)
TEAMS = ["1", "2", "3", "4", "5", "6"]


def make_schedule(seasons=(2021, 2022), weeks=5, seed=7) -> pd.DataFrame:
    """Round-robin-ish synthetic CFB schedule with the context + line columns."""
    rng = np.random.default_rng(seed)
    rows, pk = [], 1000
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(rng.permutation(TEAMS))
            for i in range(0, len(order), 2):
                h, a = order[i], order[i + 1]
                hs, as_ = int(rng.integers(0, 50)), int(rng.integers(0, 50))
                pk += 1
                spread = float(rng.integers(-14, 15)) + 0.5
                rows.append({
                    "season": season, "week": week, "home_team": h, "away_team": a,
                    "home_score": hs, "away_score": as_, "game_type": "REG",
                    "game_pk": pk,
                    "start_date": f"{season}-09-{week * 7 - 3:02d}T17:00Z",
                    "neutral_site": bool(i == 2), "conference_game": bool(i == 0),
                    "home_conf": "8", "away_conf": "5",
                    "market_spread": spread if i != 4 else np.nan,
                    "market_total": 50.5 + week,
                    "spread_open": spread - 1.0 if i != 2 else np.nan,
                    "total_open": 49.5 + week,
                    "ml_home": -150 if i == 0 else pd.NA,
                    "ml_away": 130 if i == 0 else pd.NA,
                })
    df = pd.DataFrame(rows)
    df["ml_home"] = df["ml_home"].astype("Int64")
    df["ml_away"] = df["ml_away"].astype("Int64")
    return df


# --- frozen verbatim copy of the pre-move backtest_cfb_gameline function ---
def _old_clean_market(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _old_raw_model_predictions(schedule_df, elo_cfg, blend_cfg):
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)
    games = res.games
    out = []
    for season, sdf in games.groupby("season"):
        sdf = sdf.sort_values("week")
        counts, srs_cache, pts_cache, lg_cache = {}, {}, {}, 0.0
        srs_hist = sdf.iloc[0:0]
        pts_hist = sdf.iloc[0:0]
        for week, wdf in sdf.groupby("week"):
            wdf = wdf.dropna(subset=["home_score", "away_score"])
            if wdf.empty:
                continue
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                gh, ga = counts.get(h, 0), counts.get(a, 0)
                model_margin = expected_margin(g["elo_home"], g["elo_away"],
                                               srs_cache.get(h), srs_cache.get(a),
                                               gh, ga, elo_cfg, blend_cfg)
                model_total = ((expected_total(pts_cache, lg_cache, h, a) if pts_cache
                               else 2 * lg_cache) if lg_cache else 55.0)
                out.append({
                    "season": int(season), "week": int(week),
                    "home_team": h, "away_team": a,
                    "model_margin": model_margin, "model_total": model_total,
                    "market_spread": _old_clean_market(g.get("market_spread")),
                    "market_total": _old_clean_market(g.get("market_total")),
                    "actual_margin": float(g["home_score"] - g["away_score"]),
                    "actual_total": float(g["home_score"] + g["away_score"]),
                })
            for _, g in wdf.iterrows():
                h, a = g["home_team"], g["away_team"]
                counts[h] = counts.get(h, 0) + 1
                counts[a] = counts.get(a, 0) + 1
            srs_hist = pd.concat([srs_hist, wdf], ignore_index=True)
            pts_hist = pd.concat([pts_hist, wdf], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
            pts_cache, lg_cache = compute_points_ratings(pts_hist, k_points=4.0)
    return out


def _load_backtest():
    path = Path(__file__).resolve().parents[2] / "scripts" / "backtest_cfb_gameline.py"
    spec = importlib.util.spec_from_file_location("backtest_cfb_gameline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def test_parity_with_old_function_on_every_old_key():
    sched = make_schedule()
    old = _old_raw_model_predictions(sched, ELO_CFG, BLEND_CFG)
    new = walkforward.raw_model_predictions(sched, ELO_CFG, BLEND_CFG)
    assert len(old) == len(new) > 0
    for o, n in zip(old, new):
        for k, v in o.items():
            assert _same(n[k], v), (k, n[k], v)


def test_parity_without_optional_columns():
    """A bare schedule (no context/line columns) still matches exactly."""
    sched = make_schedule()[["season", "week", "home_team", "away_team",
                             "home_score", "away_score", "market_spread", "market_total"]]
    old = _old_raw_model_predictions(sched, ELO_CFG, BLEND_CFG)
    new = walkforward.raw_model_predictions(sched, ELO_CFG, BLEND_CFG)
    assert [{k: n[k] for k in o} for o, n in zip(old, new)] == old
    assert "spread_open" not in new[0] and "game_pk" not in new[0]


def test_backtest_script_uses_shared_walkforward():
    bt = _load_backtest()
    assert bt._raw_model_predictions is walkforward.raw_model_predictions


def test_passthrough_and_pregame_ratings():
    sched = make_schedule()
    new = walkforward.raw_model_predictions(sched, ELO_CFG, BLEND_CFG)
    elo = run_elo(sched.sort_values(["season", "week"]).reset_index(drop=True), ELO_CFG).games
    by_pk = {int(r.game_pk): r for r in elo.itertuples()}
    src = {int(r.game_pk): r for r in sched.itertuples()}
    for r in new:
        g = by_pk[r["game_pk"]]
        assert r["elo_home"] == float(g.elo_home) and r["elo_away"] == float(g.elo_away)
        s = src[r["game_pk"]]
        assert r["start_date"] == s.start_date
        assert r["neutral_site"] is bool(s.neutral_site)
        assert r["home_conf"] == "8" and r["away_conf"] == "5"
        assert (r["spread_open"] is None) == pd.isna(s.spread_open)
        assert (r["ml_home"] is None) == pd.isna(s.ml_home)
        if r["ml_home"] is not None:
            assert r["ml_home"] == -150 and isinstance(r["ml_home"], int)
    week1 = [r for r in new if r["week"] == 1]
    assert all(r["srs_home"] is None and r["srs_away"] is None for r in week1)
    week3 = [r for r in new if r["season"] == 2021 and r["week"] == 3]
    assert all(isinstance(r["srs_home"], float) for r in week3)
    # SRS is the season-to-date solve over strictly-earlier weeks
    hist = sched[(sched.season == 2021) & (sched.week < 3)]
    srs = compute_srs(hist)
    for r in week3:
        assert r["srs_home"] == srs[r["home_team"]]


def test_appending_later_game_never_changes_earlier_rows():
    sched = make_schedule()
    base = walkforward.raw_model_predictions(sched, ELO_CFG, BLEND_CFG)
    later_week = pd.DataFrame([{**sched.iloc[-1].to_dict(), "week": 6, "game_pk": 99001,
                                "home_score": 70, "away_score": 0,
                                "start_date": "2022-10-10T17:00Z"}])
    later_season = pd.DataFrame([{**sched.iloc[0].to_dict(), "season": 2023, "week": 1,
                                  "game_pk": 99002, "start_date": "2023-09-01T17:00Z"}])
    for extra in (later_week, later_season, pd.concat([later_week, later_season])):
        more = walkforward.raw_model_predictions(pd.concat([sched, extra], ignore_index=True),
                                                 ELO_CFG, BLEND_CFG)
        assert len(more) == len(base) + len(extra)
        assert more[:len(base)] == base
