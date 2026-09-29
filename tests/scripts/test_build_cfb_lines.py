"""Tests for the pure seams in build_cfb_lines.py (`parse_game_lines`, `coverage`).

Spread convention: CFBD `spread` is book convention (home favored => negative);
we store home-margin convention (home favored => positive) = -spread.
"""
import importlib.util
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_lines.py"
_spec = importlib.util.spec_from_file_location("build_cfb_lines", _p)
bcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcl)


def _ln(provider, **kw):
    base = {"provider": provider, "spread": None, "spreadOpen": None, "overUnder": None,
            "overUnderOpen": None, "homeMoneyline": None, "awayMoneyline": None}
    return {**base, **kw}


def _game(lines, home="Alabama", away="Georgia"):
    return {"season": 2023, "week": 5, "homeTeam": home, "awayTeam": away, "lines": lines}


def test_parse_medians_sign_flip_and_missing_fields():
    g = _game([
        _ln("A", spread=-7.0, spreadOpen=-6.0, overUnder=50.0, overUnderOpen=49.0,
            homeMoneyline=-280, awayMoneyline=230),
        _ln("B", spread=-6.5, spreadOpen=None, overUnder=51.0, overUnderOpen=49.5,
            homeMoneyline=-300, awayMoneyline=240),           # missing spreadOpen
        _ln("C", spread=-7.5, spreadOpen=-6.5, overUnder=50.5, overUnderOpen=50.5,
            homeMoneyline=None, awayMoneyline=None),          # missing moneylines
    ])
    r = bcl.parse_game_lines(g)
    assert r["season"] == 2023 and r["week"] == 5
    assert r["home_team"] and r["away_team"] and r["home_team"] != r["away_team"]
    assert r["market_spread"] == 7.0          # -median(-7, -6.5, -7.5)
    assert r["spread_open"] == 6.25           # -median(-6, -6.5) home-margin
    assert r["market_total"] == 50.5
    assert r["total_open"] == 49.5
    assert r["ml_home"] == -290               # median(-280, -300)
    assert r["ml_away"] == 235                # median(230, 240) rounded to int
    assert isinstance(r["ml_home"], int) and isinstance(r["ml_away"], int)
    assert r["n_providers"] == 3


def test_parse_home_underdog_positive_cfbd_spread_flips_negative():
    r = bcl.parse_game_lines(_game([_ln("A", spread=3.0, spreadOpen=2.5)]))
    assert r["market_spread"] == -3.0
    assert r["spread_open"] == -2.5
    assert r["market_total"] is None and r["ml_home"] is None and r["ml_away"] is None


def test_parse_no_price_fields_is_none():
    assert bcl.parse_game_lines(_game([_ln("A")])) is None
    assert bcl.parse_game_lines(_game([])) is None
    assert bcl.parse_game_lines({"season": 2023, "week": 1, "homeTeam": "Alabama",
                                 "awayTeam": "Georgia", "lines": None}) is None


def test_parse_unmapped_team_is_none():
    assert bcl.parse_game_lines(_game([_ln("A", spread=-3.0)], away="Nowhere Tech")) is None


def test_coverage_two_season_frame():
    df = pd.DataFrame([
        {"season": 2022, "market_spread": 3.0, "spread_open": 2.5, "market_total": 50.0,
         "total_open": None, "ml_home": -150, "ml_away": 130},
        {"season": 2022, "market_spread": 1.0, "spread_open": None, "market_total": None,
         "total_open": None, "ml_home": -150, "ml_away": None},
        {"season": 2023, "market_spread": None, "spread_open": None, "market_total": 48.0,
         "total_open": 47.5, "ml_home": None, "ml_away": None},
    ])
    cov = bcl.coverage(df)
    assert list(cov.index) == [2022, 2023]
    assert list(cov.columns) == ["market_spread", "spread_open", "market_total",
                                 "total_open", "moneyline"]
    assert cov.loc[2022, "market_spread"] == 1.0
    assert cov.loc[2022, "spread_open"] == 0.5
    assert cov.loc[2022, "market_total"] == 0.5
    assert cov.loc[2022, "total_open"] == 0.0
    assert cov.loc[2022, "moneyline"] == 0.5     # needs BOTH sides
    assert cov.loc[2023, "market_spread"] == 0.0
    assert cov.loc[2023, "total_open"] == 1.0
    assert cov.loc[2023, "moneyline"] == 0.0


def _ml(*pairs):
    return bcl.parse_game_lines(_game([_ln(f"P{i}", spread=-3.0, homeMoneyline=h, awayMoneyline=a)
                                       for i, (h, a) in enumerate(pairs)]))


def test_ml_mixed_sign_providers_never_invalid():
    r = _ml((-105, 105), (105, -105))
    assert r["ml_home"] == 100 and r["ml_away"] == 100  # decimal median ~2.0012 -> +100
    assert abs(r["ml_home"]) >= 100 and abs(r["ml_away"]) >= 100


def test_ml_straddling_pair_is_valid_price():
    r = _ml((-150, -150), (110, 110))  # decimals 1.667, 2.1 -> 1.883 -> -113
    assert r["ml_home"] == -113 and r["ml_away"] == -113


def test_ml_malformed_provider_value_ignored():
    r = _ml((-150, 130), (50, 0), (-150, 130))
    assert r["ml_home"] == -150 and r["ml_away"] == 130
    assert bcl.parse_game_lines(_game([_ln("A", homeMoneyline=50, awayMoneyline=-99)])) is None


def test_ml_odd_count_is_plain_median():
    r = _ml((-300, 240), (-280, 230), (-320, 250))
    assert r["ml_home"] == -300 and r["ml_away"] == 240


def test_finalize_frame_dtypes():
    row = bcl.parse_game_lines(_game([_ln("A", spread=-7.0, homeMoneyline=-280, awayMoneyline=230)]))
    df = bcl.finalize_frame([row])
    assert str(df["ml_home"].dtype) == "Int64" and str(df["ml_away"].dtype) == "Int64"
    for c in ("market_spread", "market_total", "spread_open", "total_open"):
        assert str(df[c].dtype) == "float64"   # even all-null columns
