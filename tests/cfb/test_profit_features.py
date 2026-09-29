"""Tests for sportsmodel.cfb.profit_features.build_rows / FEATURE_COLS.

(c) label + push logic for spread and total at open and close, f_move sign,
    moneyline rows, rest days, class mapping;
(d) priors join by (season, team) with NaN when missing;
plus a leak test on the feature table (appending a later game never changes
earlier rows).
"""
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import profit_features as pf

FBS = {"333", "2", "194", "2006", "87", "349", "2426", "256", "8", "12", "2309"}
# 333 Alabama (SEC 8), 2 Auburn (SEC 8), 194 Ohio State (B1G 5), 2006 Akron (MAC 15),
# 87 Notre Dame (Ind 18), 349 Army (Ind 18), 2426 Navy (AAC 151), 256 JMU,
# 8 Arkansas (SEC), 12 Arizona (Pac-12 9 -> Big 12 4), 2309 Kent State (MAC 15)


def raw_game(**kw) -> dict:
    base = {
        "season": 2022, "week": 2, "game_pk": 1, "home_team": "333", "away_team": "2",
        "model_margin": 10.0, "model_total": 50.0,
        "market_spread": None, "market_total": None,
        "actual_margin": 7.0, "actual_total": 60.0,
        "start_date": "2022-09-03T19:00Z", "neutral_site": False, "conference_game": True,
        "home_conf": "8", "away_conf": "8",
        "spread_open": None, "total_open": None, "ml_home": None, "ml_away": None,
        "elo_home": 1600.0, "elo_away": 1550.0, "srs_home": 5.0, "srs_away": 2.0,
    }
    base.update(kw)
    return base


EMPTY_PRIORS = pd.DataFrame(columns=["season", "team_espn_id", "team_name", "sp_rating",
                                     "returning_pct", "returning_starters", "qb_returning",
                                     "recruiting_points", "portal_net", "coach_first_year",
                                     "prior_sos", "forward_sos_shift"])


def _row(df, market, price_point, game_pk=1):
    sel = df[(df.market == market) & (df.price_point == price_point) & (df.game_pk == game_pk)]
    assert len(sel) <= 1
    return None if sel.empty else sel.iloc[0]


def test_spread_and_total_labels_pushes_and_move():
    g1 = raw_game(spread_open=6.5, market_spread=7.0,      # open: 7 > 6.5 cover; close: push
                  total_open=58.5, market_total=61.5,       # open: over; close: under
                  ml_home=-250, ml_away=200)
    g2 = raw_game(game_pk=2, week=3, start_date="2022-09-10T19:00Z",
                  actual_margin=-3.0, actual_total=50.0,
                  spread_open=None, market_spread=-2.5,     # -3 > -2.5 false -> 0
                  total_open=50.0, market_total=None)        # open push -> dropped
    df = pf.build_rows([g1, g2], EMPTY_PRIORS, FBS)

    so = _row(df, "spread", "open")
    assert so["y"] == 1 and so["line"] == 6.5 and so["f_move"] == 0.0
    assert so["f_edge_pts"] == pytest.approx(10.0 - 6.5)
    assert _row(df, "spread", "close") is None               # push dropped

    to = _row(df, "total", "open")
    assert to["y"] == 1 and to["line"] == 58.5 and to["f_move"] == 0.0
    assert to["f_edge_pts"] == pytest.approx(50.0 - 58.5)
    tc = _row(df, "total", "close")
    assert tc["y"] == 0 and tc["line"] == 61.5
    assert tc["f_move"] == pytest.approx(61.5 - 58.5)        # line rose -> positive move

    ml = _row(df, "moneyline", "close")
    assert ml["y"] == 1 and math.isnan(ml["line"])
    assert ml["ml_home"] == -250 and ml["ml_away"] == 200
    assert ml["f_edge_pts"] == 10.0
    assert _row(df, "moneyline", "open") is None
    assert math.isnan(so["ml_home"]) and math.isnan(so["ml_away"])

    s2 = _row(df, "spread", "close", 2)
    assert s2["y"] == 0 and math.isnan(s2["f_move"])          # no opener -> NaN move
    assert _row(df, "spread", "open", 2) is None
    assert _row(df, "total", "open", 2) is None               # push dropped
    assert _row(df, "total", "close", 2) is None
    assert _row(df, "moneyline", "close", 2) is None

    # spread move sign: close below the opener (home line fell) -> negative
    g3 = raw_game(game_pk=3, spread_open=-3.0, market_spread=-6.0, actual_margin=-10.0)
    s3 = _row(pf.build_rows([g3], EMPTY_PRIORS, FBS), "spread", "close", 3)
    assert s3["f_move"] == -3.0 and s3["y"] == 0


def test_moneyline_tie_dropped_and_common_features():
    g = raw_game(actual_margin=0.0, ml_home=-110, ml_away=-110, market_spread=1.0,
                 neutral_site=True, conference_game=False)
    df = pf.build_rows([g], EMPTY_PRIORS, FBS)
    assert _row(df, "moneyline", "close") is None
    r = _row(df, "spread", "close")
    assert r["y"] == 0                                         # 0 > 1 false
    assert r["f_model_margin"] == 10.0 and r["f_model_total"] == 50.0
    assert r["f_elo_diff"] == 50.0 and r["f_srs_diff"] == 3.0
    assert r["f_neutral"] == 1.0 and r["f_conf_game"] == 0.0 and r["f_week"] == 2
    assert r["season"] == 2022 and r["home_team"] == "333" and r["away_team"] == "2"
    df2 = pf.build_rows([raw_game(srs_home=None, market_spread=1.0)], EMPTY_PRIORS, FBS)
    assert math.isnan(df2.iloc[0]["f_srs_diff"])


def test_rest_days_same_season_only():
    games = [
        raw_game(game_pk=1, season=2021, week=14, start_date="2021-12-04T20:00Z",
                 home_team="333", away_team="2", market_spread=3.0),
        raw_game(game_pk=2, week=1, start_date="2022-08-27T23:00Z",
                 home_team="333", away_team="FCS", away_conf="48", market_spread=30.0),
        raw_game(game_pk=3, week=2, start_date="2022-09-03T19:00Z",
                 home_team="2", away_team="333", market_spread=3.0),
        # Saturday-night US kickoff stored as next-day UTC -> still a 7-day rest
        raw_game(game_pk=4, week=3, start_date="2022-09-11T03:30Z",
                 home_team="333", away_team="FCS", away_conf="48", market_spread=30.0),
    ]
    df = pf.build_rows(games, EMPTY_PRIORS, FBS)
    r = lambda pk: _row(df, "spread", "close", pk)
    assert math.isnan(r(1)["f_rest_home"]) and math.isnan(r(1)["f_rest_away"])
    assert math.isnan(r(2)["f_rest_home"])                    # no 2022 game yet (2021 ignored)
    assert math.isnan(r(2)["f_rest_away"])                    # FCS pseudo-team: never rested
    assert math.isnan(r(3)["f_rest_home"])                    # Auburn's first 2022 game
    assert r(3)["f_rest_away"] == 7.0                         # Alabama: 8/27 -> 9/3
    assert r(4)["f_rest_home"] == 7.0                         # 9/3 -> 9/10 (US date)
    assert math.isnan(r(4)["f_rest_away"])


@pytest.mark.parametrize("season,home,away,hc,ac,expected", [
    (2022, "333", "194", "8", "5", 0),        # SEC vs Big Ten
    (2022, "333", "2006", "8", "15", 1),      # SEC vs MAC
    (2022, "2006", "2309", "15", "15", 2),    # MAC vs MAC
    (2022, "333", "FCS", "8", "48", 3),       # FBS vs FCS pseudo-team
    (2022, "2006", "FCS", "15", "48", 3),
    (2023, "12", "333", "9", "8", 0),         # Pac-12 is P4 through 2023
    (2024, "12", "333", "9", "8", 1),         # ...and G5 from 2024 (remnant Pac-12)
    (2022, "87", "333", "18", "8", 0),        # Notre Dame (independent) counted P4
    (2022, "349", "2426", "18", "151", 2),    # Army (independent) vs Navy (AAC)
    (2019, "256", "8", "48", "8", 3),         # JMU in its FCS (CAA) era
    (2022, "999", "333", "8", "8", 3),        # not in the FBS set -> FCS
    (2022, "333", "194", None, "5", None),    # unknown conference -> NaN
])
def test_class_mapping(season, home, away, hc, ac, expected):
    got = pf.game_class(season, home, away, hc, ac, FBS)
    if expected is None:
        assert math.isnan(got)
    else:
        assert got == expected


def test_class_in_feature_table():
    df = pf.build_rows([raw_game(away_team="2006", away_conf="15", market_spread=20.0)],
                       EMPTY_PRIORS, FBS)
    assert df.iloc[0]["f_class"] == 1


def _prior(season, team, **kw):
    row = {"season": season, "team_espn_id": team, "team_name": team, "sp_rating": np.nan,
           "returning_pct": np.nan, "returning_starters": np.nan, "qb_returning": None,
           "recruiting_points": np.nan, "portal_net": np.nan, "coach_first_year": False,
           "prior_sos": np.nan, "forward_sos_shift": np.nan}
    row.update(kw)
    return row


def test_priors_join_by_season_and_team_with_nan_when_missing():
    priors = pd.DataFrame([
        _prior(2022, "333", returning_pct=0.6, qb_returning=True, recruiting_points=300.0,
               portal_net=2.0),
        _prior(2021, "333", sp_rating=25.0, coach_first_year=False),
        # the away team has only a 2021 row: its SAME-season fields are missing
        # for 2022, its previous-season SP+/coach flag are used
        _prior(2021, "2", sp_rating=10.0, returning_pct=0.9, coach_first_year=True),
        _prior(2022, "2006", returning_pct=0.4, recruiting_points=90.0),
        # Akron has no 2021 row -> previous-season fields NaN
    ])
    g1 = raw_game(market_spread=7.5)
    g2 = raw_game(game_pk=2, away_team="2006", away_conf="15", market_spread=30.5)
    df = pf.build_rows([g1, g2], priors, FBS)
    r1 = _row(df, "spread", "close", 1)
    assert r1["f_sp_prev_home"] == 25.0 and r1["f_sp_prev_away"] == 10.0
    assert r1["f_sp_prev_diff"] == 15.0
    assert r1["f_coach_new_prev_home"] == 0.0 and r1["f_coach_new_prev_away"] == 1.0
    assert r1["f_ret_home"] == 0.6 and math.isnan(r1["f_ret_away"])   # 2021 row not used
    assert math.isnan(r1["f_ret_diff"])
    assert r1["f_qb_ret_home"] == 1.0 and math.isnan(r1["f_qb_ret_away"])
    r2 = _row(df, "spread", "close", 2)
    assert math.isnan(r2["f_sp_prev_away"]) and math.isnan(r2["f_sp_prev_diff"])
    assert math.isnan(r2["f_coach_new_prev_away"])
    assert r2["f_ret_away"] == 0.4 and r2["f_ret_diff"] == pytest.approx(0.2)
    assert r2["f_rec_diff"] == 210.0
    assert r2["f_portal_home"] == 2.0 and math.isnan(r2["f_portal_away"])


def test_same_season_sp_and_coach_flag_never_reach_a_feature():
    planted = 12345.0
    priors = pd.DataFrame([
        _prior(2022, "333", sp_rating=planted, coach_first_year=True, forward_sos_shift=planted),
        _prior(2022, "2", sp_rating=-planted, coach_first_year=True, forward_sos_shift=planted),
        _prior(2021, "333", sp_rating=20.0, coach_first_year=False),
        _prior(2021, "2", sp_rating=5.0, coach_first_year=False),
    ])
    g = raw_game(spread_open=6.5, market_spread=7.5, total_open=58.5, market_total=61.5,
                 ml_home=-250, ml_away=200)
    df = pf.build_rows([g], priors, FBS)
    feats = sorted({c for cols in pf.FEATURE_COLS.values() for c in cols})
    vals = df[feats].to_numpy(dtype=float)
    assert not np.isin(np.abs(vals), [planted, 2 * planted]).any()
    assert (df["f_sp_prev_diff"] == 15.0).all()
    assert (df["f_coach_new_prev_home"] == 0.0).all()
    assert not any(c.startswith("f_sp_") and "prev" not in c for c in df.columns)
    assert not any("coach" in c and "prev" not in c for c in df.columns)
    # no priors column other than the audited ones is used at all
    assert not any("sos" in c for c in df.columns)


def test_unscored_rows_kept_with_nan_label():
    # an upcoming game whose closing line equals what would be a push: kept, y NaN
    g = raw_game(actual_margin=None, actual_total=None, spread_open=7.0, market_spread=7.0,
                 total_open=60.0, market_total=60.0, ml_home=-250, ml_away=200)
    df = pf.build_rows([g], EMPTY_PRIORS, FBS)
    assert len(df) == 5
    assert df["y"].isna().all()
    assert set(zip(df.market, df.price_point)) == {
        ("spread", "open"), ("spread", "close"), ("total", "open"), ("total", "close"),
        ("moneyline", "close")}


def test_rest_days_count_unplayed_scheduled_games():
    games = [
        raw_game(game_pk=1, week=1, start_date="2022-09-01T23:00Z", market_spread=3.0),
        # unplayed (e.g. upcoming / no score yet) but scheduled -> still counts
        raw_game(game_pk=2, week=2, start_date="2022-09-08T23:00Z", market_spread=3.0,
                 actual_margin=None, actual_total=None),
        raw_game(game_pk=3, week=3, start_date="2022-09-17T19:00Z", market_spread=3.0,
                 actual_margin=None, actual_total=None),
    ]
    df = pf.build_rows(games, EMPTY_PRIORS, FBS)
    r = lambda pk: _row(df, "spread", "close", pk)
    assert math.isnan(r(1)["f_rest_home"])
    assert r(2)["f_rest_home"] == 7.0
    assert r(3)["f_rest_home"] == 9.0          # from game 2 (unplayed), not game 1
    # strictly-before only: the later unplayed game never changes earlier rows
    base = pf.build_rows(games[:2], EMPTY_PRIORS, FBS)
    pd.testing.assert_frame_equal(df[df.game_pk != 3].reset_index(drop=True), base)


def test_feature_cols_per_market():
    assert set(pf.FEATURE_COLS) == {"spread", "total", "moneyline"}
    for m in ("spread", "total"):
        assert "line" in pf.FEATURE_COLS[m] and "f_move" in pf.FEATURE_COLS[m]
    ml = pf.FEATURE_COLS["moneyline"]
    assert "line" not in ml and "f_move" not in ml
    assert "f_edge_pts" not in ml and "f_model_margin" in ml   # duplicate dropped
    assert "f_edge_pts" in pf.FEATURE_COLS["spread"]
    for c in ("f_class", "f_rest_home", "f_sp_prev_diff", "f_qb_ret_away",
              "f_coach_new_prev_home"):
        assert c in ml
    g = raw_game(spread_open=6.5, market_spread=7.5, total_open=58.5, market_total=61.5,
                 ml_home=-250, ml_away=200)
    df = pf.build_rows([g], EMPTY_PRIORS, FBS)
    for m, cols in pf.FEATURE_COLS.items():
        assert set(cols) <= set(df.columns)
        assert (df.market == m).any()


def test_appending_later_game_never_changes_earlier_feature_rows():
    games = [
        raw_game(game_pk=1, week=1, start_date="2022-08-27T23:00Z",
                 spread_open=6.5, market_spread=7.5, total_open=50.5, market_total=52.5),
        raw_game(game_pk=2, week=2, start_date="2022-09-03T19:00Z", home_team="2",
                 away_team="333", spread_open=-3.5, market_spread=-2.5, ml_home=120, ml_away=-140),
    ]
    later = raw_game(game_pk=3, week=3, start_date="2022-09-10T19:00Z",
                     spread_open=1.5, market_spread=2.5, total_open=40.5, market_total=41.5)
    base = pf.build_rows(games, EMPTY_PRIORS, FBS)
    more = pf.build_rows(games + [later], EMPTY_PRIORS, FBS)
    earlier = more[more.game_pk != 3].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier, base.reset_index(drop=True))
