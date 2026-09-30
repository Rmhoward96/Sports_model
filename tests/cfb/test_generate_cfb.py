import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_cfb.py"
_s = importlib.util.spec_from_file_location("generate_cfb", _p)
gc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gc)

from sportsmodel.cfb.priors import DecayConfig, PriorWeights, season_priors
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.ratings import BlendConfig

ELO_CFG = EloConfig()
BLEND_CFG = BlendConfig(w_sos=0.0, srs_min_games=3)  # w_sos=0 -> pure elo margin, no SRS needed
GL_CFG = GameLineConfig()


def _ratings(elo_final=None, srs_now=None, points_ratings=None, lg_avg=55.0,
            games_played=None, r_pre=None, decay_cfg=None) -> dict:
    return {
        "elo_final": elo_final or {},
        "srs_now": srs_now or {},
        "points_ratings": points_ratings or {},
        "lg_avg": lg_avg,
        "games_played": games_played or {},
        "elo_cfg": ELO_CFG,
        "blend_cfg": BLEND_CFG,
        "r_pre": r_pre or {},
        "decay_cfg": decay_cfg or DecayConfig(half_life_games=4.0),
    }


def test_build_game_row_is_serving_shaped():
    game = {"game_pk": 401628354, "game_date": "2026-09-05",
            "commence_time": "2026-09-05T23:30Z",
            "market_spread": -3.5, "market_total": 55.5,
            "home_team": "96", "away_team": "61",
            "home_name": "Kentucky Wildcats", "away_name": "Georgia Bulldogs"}
    ctx = {"model_margin": -7.0, "model_total": 58.0, "week": 3}
    row = gc.build_game_row(game, ctx, GL_CFG)
    assert row["market_spread"] == -3.5 and row["market_total"] == 55.5  # Vegas line for the lean
    assert row["sport"] == "cfb"
    assert row["model_version"] == "cfb-ratings-v2"   # v2: walk-forward state + leak-free prior
    assert row["game_pk"] == 401628354
    assert row["game_date"] == "2026-09-05"
    assert row["commence_time"] == "2026-09-05T23:30Z"  # kickoff carried through for time sort
    assert row["home_team_name"] == "Kentucky Wildcats"
    assert row["away_team_name"] == "Georgia Bulldogs"
    assert 0 < row["home_win_prob"] < 1
    assert isinstance(row["pred_home_score"], float)
    assert isinstance(row["pred_away_score"], float)
    assert row["margin_dist"]["kind"] == "margin"
    assert row["total_dist"]["kind"] == "pmf"
    # model-only: no market was ever passed in, so pred_margin/pred_total
    # must equal the raw model inputs untouched by any shrink
    assert row["pred_margin"] == ctx["model_margin"]
    assert row["pred_total"] == ctx["model_total"]


def test_build_game_rows_skips_fcs_games():
    games = [
        {"game_pk": 1, "home_team": "96", "away_team": "61",
         "home_name": "Kentucky Wildcats", "away_name": "Georgia Bulldogs",
         "game_date": "2026-09-05"},
        {"game_pk": 2, "home_team": "158", "away_team": "FCS",
         "home_name": "Nebraska Cornhuskers", "away_name": "Northern Iowa Panthers",
         "game_date": "2026-09-05"},
        {"game_pk": 3, "home_team": "FCS", "away_team": "12",
         "home_name": "Some FCS School", "away_name": "Arizona Wildcats",
         "game_date": "2026-09-05"},
    ]
    ratings = _ratings(elo_final={"96": 1550.0, "61": 1620.0})
    rows = gc.build_game_rows(games, ratings, week=3, gl_cfg=GL_CFG)
    assert len(rows) == 1
    assert rows[0]["game_pk"] == 1
    assert all(r["sport"] == "cfb" for r in rows)


def test_build_game_rows_uses_ratings_to_derive_margin_and_total():
    games = [
        {"game_pk": 10, "home_team": "H", "away_team": "A",
         "home_name": "Home Team", "away_name": "Away Team", "game_date": "2026-09-05"},
    ]
    ratings = _ratings(
        elo_final={"H": 1700.0, "A": 1500.0},
        points_ratings={"H": {"off": 5.0, "def": -2.0}, "A": {"off": -3.0, "def": 1.0}},
        lg_avg=55.0,
        games_played={"H": 5, "A": 5},
    )
    rows = gc.build_game_rows(games, ratings, week=3, gl_cfg=GL_CFG)
    assert len(rows) == 1
    row = rows[0]
    # home is much stronger (elo 1700 vs 1500 + HFA) -> favored, positive margin
    assert row["pred_margin"] > 0
    assert row["home_win_prob"] > 0.5
    assert row["pred_home_score"] > 0 and row["pred_away_score"] > 0


def test_build_game_rows_defaults_missing_ratings_to_base_elo():
    # Teams absent from elo_final/srs_now/points_ratings (e.g. first game of
    # the historical window) must fall back gracefully rather than KeyError.
    games = [
        {"game_pk": 20, "home_team": "H", "away_team": "A",
         "home_name": "Home Team", "away_name": "Away Team", "game_date": "2026-09-05"},
    ]
    ratings = _ratings()  # everything empty
    rows = gc.build_game_rows(games, ratings, week=1, gl_cfg=GL_CFG)
    assert len(rows) == 1
    assert 0 < rows[0]["home_win_prob"] < 1


# --------------------------------------------------------------------------
# Forward-looking priors (Task 6): build_game_rows blends each team's R_pre
# (from priors.parquet + PriorWeights, via preseason_rating/season_features_z)
# into the pre-game Elo rating with decay-by-games_played, mirroring
# backtest_cfb_priors.py's prior_seeded_margin design (design decision #1:
# blend at the Elo level, feed the blended value into expected_margin).
# --------------------------------------------------------------------------

_PRIOR_WEIGHTS = PriorWeights(sp_scale=25.0)   # R_pre = 1500 + 25 * prev_sp
_NAN = float("nan")


def _prior_row(team, season, sp, **kw):
    return {"season": season, "team_espn_id": team, "team_name": team, "sp_rating": sp,
            "returning_pct": 0.5, "returning_starters": 0.5, "qb_returning": True,
            "recruiting_points": 200.0, "portal_net": _NAN, "coach_first_year": False,
            "prior_sos": 0.0, "forward_sos_shift": 0.0, **kw}


# H: previous-season SP+ +10 -> R_pre = 1500 + 25*10 = 1750 (the only team
# with a row in 2026, so every z-score is 0)
_R_PRE_H = season_priors({2025: [_prior_row("H", 2025, 10.0)],
                          2026: [_prior_row("H", 2026, -3.0)]}, 2026, _PRIOR_WEIGHTS)["H"]
assert _R_PRE_H == 1750.0


def _games_h_vs_a():
    return [{"game_pk": 30, "home_team": "H", "away_team": "A",
             "home_name": "Home Team", "away_name": "Away Team", "game_date": "2026-09-05"}]


def test_build_game_rows_margin_driven_by_prior_at_zero_games_played():
    # Team H has a strong R_pre (1750) but no in-season Elo history yet
    # (games_played=0 for both sides, elo_final empty -> both default to
    # elo_cfg.base=1500). At games_played=0 the decay weight is 1.0, so the
    # blended rating for H should equal R_pre exactly, producing a
    # noticeably larger home margin than the no-prior baseline (a 250 elo
    # gap is a 10-point margin swing at this engine's 25-elo-per-point scale).
    baseline = gc.build_game_rows(_games_h_vs_a(), _ratings(games_played={"H": 0, "A": 0}),
                                  week=1, gl_cfg=GL_CFG)
    with_prior = gc.build_game_rows(
        _games_h_vs_a(),
        _ratings(games_played={"H": 0, "A": 0}, r_pre={"H": _R_PRE_H}),
        week=1, gl_cfg=GL_CFG)
    assert with_prior[0]["pred_margin"] - baseline[0]["pred_margin"] > 9.9


def test_build_game_rows_prior_influence_negligible_at_high_games_played():
    # Once a team has played many games, the decay weight collapses toward
    # the DecayConfig floor, so the prior-seeded margin should be within a
    # hair of the no-prior baseline (not identical, since the weight never
    # hits exactly 0, but negligible).
    baseline = gc.build_game_rows(_games_h_vs_a(), _ratings(games_played={"H": 100, "A": 100}),
                                  week=1, gl_cfg=GL_CFG)
    with_prior = gc.build_game_rows(
        _games_h_vs_a(),
        _ratings(games_played={"H": 100, "A": 100}, r_pre={"H": _R_PRE_H}),
        week=1, gl_cfg=GL_CFG)
    assert abs(with_prior[0]["pred_margin"] - baseline[0]["pred_margin"]) < 0.01


def test_build_game_rows_missing_priors_falls_back_to_todays_behavior():
    # No "r_pre"/"decay_cfg" injected at all (mirrors main() when
    # assets/cfb/priors.parquet doesn't exist -- `_ratings()`'s own defaults
    # of r_pre={} apply) -- must be byte-for-byte the pre-Task-6 behavior.
    games = _games_h_vs_a()
    ratings = _ratings(elo_final={"H": 1700.0, "A": 1500.0}, games_played={"H": 5, "A": 5})
    rows = gc.build_game_rows(games, ratings, week=3, gl_cfg=GL_CFG)
    assert len(rows) == 1
    assert rows[0]["pred_margin"] > 0
    assert rows[0]["home_win_prob"] > 0.5


# --------------------------------------------------------------------------
# Live state == backtested walk-forward state (fix/cfb-live-ratings, part A).
#
# The live producer must serve exactly what backtest_cfb_gameline.py /
# walkforward.raw_model_predictions scored: Elo continuous across seasons,
# SRS / points ratings / games_played SEASON-TO-DATE (current season, weeks
# < W), and the same total seed/fallback. The old `_season_to_date_ratings`
# pooled SRS/points/counts over every season since 2015 (games_played ~140,
# so the preseason prior decayed to 0 and the SRS gate was always open).
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import walkforward

_WF_ELO = EloConfig(k=40, hfa_elo=70, carryover=0.9, base=1500.0)
_WF_BLEND = BlendConfig(w_sos=0.45, srs_min_games=3)
_TEAMS = [str(t) for t in range(1, 11)]


def _synthetic_schedule(seasons=(2021, 2022), weeks=6, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, pk = [], 5000
    for season in seasons:
        for week in range(1, weeks + 1):
            order = list(rng.permutation(_TEAMS))
            for i in range(0, len(order), 2):
                pk += 1
                rows.append({"season": season, "week": week, "game_type": "REG",
                             "home_team": order[i], "away_team": order[i + 1],
                             "home_score": int(rng.integers(0, 56)),
                             "away_score": int(rng.integers(0, 56)), "game_pk": pk})
    return pd.DataFrame(rows)


def _games_of_week(sched, season, week):
    wk = sched[(sched["season"] == season) & (sched["week"] == week)]
    return [{"game_pk": int(g.game_pk), "home_team": g.home_team, "away_team": g.away_team,
             "home_name": g.home_team, "away_name": g.away_team, "game_date": "2000-01-01"}
            for g in wk.itertuples()]


def _live_preds(sched, season, week, elo_cfg=_WF_ELO, blend_cfg=_WF_BLEND):
    """generate_cfb's model margin/total for (season, week), prior disabled.
    `sched` is passed WHOLE (incl. week W's own results and later weeks) --
    the live state must ignore everything at or after (season, week)."""
    games = _games_of_week(sched, season, week)
    ratings = gc._season_to_date_ratings(sched, season, week, elo_cfg, blend_cfg, games)
    ratings["r_pre"] = {}                      # prior blend disabled
    rows = gc.build_game_rows(games, ratings, week, GameLineConfig())  # bias 0 -> pred == model
    return {r["game_pk"]: (r["pred_margin"], r["pred_total"]) for r in rows}


def _wf_upcoming_preds(sched, season, week, elo_cfg=_WF_ELO, blend_cfg=_WF_BLEND):
    """walkforward rows for week W when week W is the live, unplayed week:
    history through (season, week) with week W's scores blanked."""
    frame = sched[(sched["season"] < season)
                  | ((sched["season"] == season) & (sched["week"] <= week))].copy()
    frame["home_score"] = frame["home_score"].astype("Float64")
    frame["away_score"] = frame["away_score"].astype("Float64")
    wk = (frame["season"] == season) & (frame["week"] == week)
    frame.loc[wk, ["home_score", "away_score"]] = pd.NA
    rows = walkforward.raw_model_predictions(frame, elo_cfg, blend_cfg, include_unscored=True)
    return {r["game_pk"]: (r["model_margin"], r["model_total"])
            for r in rows if r["season"] == season and r["week"] == week}


def _wf_backtest_preds(sched, season, week, elo_cfg=_WF_ELO, blend_cfg=_WF_BLEND):
    """walkforward rows exactly as the backtest scored them (full schedule)."""
    rows = walkforward.raw_model_predictions(sched, elo_cfg, blend_cfg)
    return {r["game_pk"]: (r["model_margin"], r["model_total"])
            for r in rows if r["season"] == season and r["week"] == week}


@pytest.mark.parametrize("season,week", [(2022, 1), (2022, 2), (2022, 4), (2022, 6)])
def test_live_parity_with_walkforward_synthetic(season, week):
    sched = _synthetic_schedule()
    live = _live_preds(sched, season, week)
    assert live and live == _wf_upcoming_preds(sched, season, week)
    # no team plays twice in a synthetic week -> also equal to the backtest rows
    bt = _wf_backtest_preds(sched, season, week)
    assert set(live) == set(bt)
    for pk, (m, t) in live.items():
        assert m == pytest.approx(bt[pk][0], abs=1e-9)
        assert t == pytest.approx(bt[pk][1], abs=1e-9)


def test_live_state_is_season_to_date_not_pooled():
    sched = _synthetic_schedule()
    games = _games_of_week(sched, 2022, 1)
    st = gc._season_to_date_ratings(sched, 2022, 1, _WF_ELO, _WF_BLEND, games)
    # week 1: a whole prior season of history exists, but none of it this season
    assert st["games_played"] == {} and st["srs_now"] == {} and st["points_ratings"] == {}
    rows = gc.build_game_rows(games, {**st, "r_pre": {}}, 1, GameLineConfig())
    assert all(r["pred_total"] == walkforward._DEFAULT_TOTAL_SEED for r in rows)
    st4 = gc._season_to_date_ratings(sched, 2022, 4, _WF_ELO, _WF_BLEND,
                                     _games_of_week(sched, 2022, 4))
    assert max(st4["games_played"].values()) == 3          # weeks 1-3 of 2022 only
    # Elo, by contrast, is continuous: week-1 ratings carry 2021 in (with carryover)
    assert st["elo_final"] and any(v != _WF_ELO.base for v in st["elo_final"].values())


def test_live_state_ignores_results_at_or_after_the_week():
    sched = _synthetic_schedule()
    base = _live_preds(sched, 2022, 3)
    later = sched.copy()
    later.loc[(later.season == 2022) & (later.week >= 3), "home_score"] += 30
    assert _live_preds(later, 2022, 3) == base


def _real_merged():
    import importlib.util as _ilu
    p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_cfb_gameline.py"
    spec = _ilu.spec_from_file_location("backtest_cfb_gameline_parity", p)
    bt = _ilu.module_from_spec(spec)
    spec.loader.exec_module(bt)
    root = p.parents[1]
    return bt.load_merged_schedule(str(root / "assets/cfb/schedules.parquet"),
                                   str(root / "assets/cfb/lines.parquet"))


@pytest.mark.parametrize("week", [1, 6])
def test_live_parity_with_walkforward_real_history(week):
    """Real CFBD history (2021-2023), committed rating.json config: the live
    state for a historical 2023 week reproduces the walk-forward rows."""
    elo_cfg, blend_cfg = gc.load_rating()
    merged = _real_merged()
    sched = merged[merged["season"].between(2021, 2023)].reset_index(drop=True)
    live = _live_preds(sched, 2023, week, elo_cfg, blend_cfg)
    up = _wf_upcoming_preds(sched, 2023, week, elo_cfg, blend_cfg)
    assert len(live) > 40
    assert live == {pk: up[pk] for pk in live}           # exact, same engine
    # vs the backtest's scored rows: equal except a team's 2nd game in the
    # same CFBD week (week 0 is folded into week 1), whose pre-game Elo in
    # the backtest already includes that week's first game.
    wk = sched[(sched.season == 2023) & (sched.week == week)]
    seen, single = set(), set()
    for g in wk.sort_values(["season", "week"], kind="stable").itertuples():
        if g.home_team not in seen and g.away_team not in seen:
            single.add(int(g.game_pk))
        seen.update({g.home_team, g.away_team})
    bt = _wf_backtest_preds(sched, 2023, week, elo_cfg, blend_cfg)
    checked = 0
    for pk, (m, t) in live.items():
        if pk in single:
            assert m == pytest.approx(bt[pk][0], abs=1e-6)
            assert t == pytest.approx(bt[pk][1], abs=1e-6)
            checked += 1
    assert checked > 40


def test_load_priors_for_season_is_leak_free(tmp_path):
    """Live R_pre uses the PREVIOUS season's final SP+; planting a same-season
    SP+ / new-coach flag / forward-SoS (post-season info) changes nothing."""
    prev = [_prior_row("A", 2025, 10.0), _prior_row("B", 2025, -10.0)]
    cur = [_prior_row("A", 2026, 0.0, returning_pct=0.8), _prior_row("B", 2026, 0.0)]
    w = PriorWeights(sp_scale=20.0, w_returning=10.0)
    path = tmp_path / "priors.parquet"
    pd.DataFrame(prev + cur).to_parquet(path)
    clean = gc.load_priors_for_season(2026, w, path)
    assert clean["A"] == pytest.approx(1500.0 + 200.0 + 10.0)
    assert clean["B"] == pytest.approx(1500.0 - 200.0 - 10.0)
    planted = [{**r, "sp_rating": 99.0, "coach_first_year": True, "forward_sos_shift": 9.0}
               for r in cur]
    pd.DataFrame(prev + planted).to_parquet(path)
    assert gc.load_priors_for_season(2026, w, path) == clean
    assert gc.load_priors_for_season(2030, w, path) == {}
    assert gc.load_priors_for_season(2026, w, tmp_path / "missing.parquet") == {}


# --------------------------------------------------------------------------
# Fix round 1: postseason state week (R2) + strict prior weights (R4)
# --------------------------------------------------------------------------

def test_state_week_regular_season_is_the_espn_week():
    sched = _synthetic_schedule()
    assert gc.state_week(sched, 2022, 4, season_type=2) == 4


def test_state_week_postseason_uses_the_full_regular_season():
    # ESPN reports bowls as season_type 3, week 1: the state must be the
    # whole REG season (max REG week + 1), not week 1's empty season-to-date.
    sched = _synthetic_schedule()                       # 2022 REG weeks 1-6
    assert gc.state_week(sched, 2022, 1, season_type=3) == 7
    games = _games_of_week(sched, 2022, 6)
    st = gc._season_to_date_ratings(sched, 2022, gc.state_week(sched, 2022, 1, 3),
                                    _WF_ELO, _WF_BLEND, games)
    assert max(st["games_played"].values()) == 6 and st["srs_now"] and st["lg_avg"] > 0
    # season absent from the schedule -> fall back to the ESPN week
    assert gc.state_week(sched, 2030, 1, season_type=3) == 1


def _write_json(path, data):
    path.write_text(__import__("json").dumps(data))
    return path


def test_live_prior_weights_strict_when_priors_exist(tmp_path):
    priors = tmp_path / "priors.parquet"
    pd.DataFrame([_prior_row("A", 2026, 1.0)]).to_parquet(priors)
    full = {"sp_scale": 17.5, "sp_offset": 1500.0, "w_returning": 50.0, "w_recruiting": 80.0,
            "w_portal": 50.0, "w_qb": 0.0, "w_sos_prior": 50.0}
    ok = _write_json(tmp_path / "w.json", full)
    assert gc.load_live_prior_weights(ok, priors) == PriorWeights(**full)
    with pytest.raises(FileNotFoundError, match="priors_weights"):
        gc.load_live_prior_weights(tmp_path / "missing.json", priors)
    old = _write_json(tmp_path / "old.json", {"sp_scale": 1.0, "sp_offset": 1500.0,
                                              "w_portal": 0.0, "w_coach": 0.0, "w_qb": 0.0,
                                              "w_starters": 0.0, "w_sos_prior": 0.0,
                                              "w_sos_shift": 0.0})
    with pytest.raises(ValueError, match="w_recruiting"):
        gc.load_live_prior_weights(old, priors)
    # no priors asset -> no prior is served, so no weights are required
    assert gc.load_live_prior_weights(tmp_path / "missing.json",
                                      tmp_path / "no_priors.parquet") == PriorWeights()
