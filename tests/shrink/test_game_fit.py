"""Game-level shrink fit (profit step 2 Part B): closed forms, sign conventions, walk-forward, the 300 minimum, the documents.
No database, no network; two committed assets are read (lines/schedules parquet, gameline.json)."""
import json
import random
from pathlib import Path

import pytest

from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
from sportsmodel.shrink import game_fit
from sportsmodel.shrink.fit import closed_form_k

ROOT = Path(__file__).resolve().parents[2]
NO_BIAS = GameLineConfig()               # bias 0 and no market: the served number IS the model's number


def american(p):
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def raw_row(season=2018, week=3, **kw):
    base = {"season": season, "week": week, "home_team": "H", "away_team": "A", "model_margin": 10.0, "model_total": 52.0,
            "market_spread": 7.0, "market_total": 50.0, "actual_margin": 8.5, "actual_total": 51.0, "ml_home": None,
            "ml_away": None}
    return base | kw


def spread_rows(k_true, seasons=(2016, 2017, 2018, 2019), n=400, seed=3):
    """Walk-forward-shaped rows whose actual margin is close + k_true * (model - close): the fit must recover k_true."""
    rng = random.Random(seed)
    rows = []
    for s in seasons:
        for i in range(n):
            close = rng.uniform(-20, 20)
            model = close + rng.uniform(-8, 8)
            rows.append(raw_row(s, 1 + i % 14, model_margin=model, market_spread=close,
                                actual_margin=close + k_true * (model - close)))
    return rows


def served(raw):
    return game_fit.served_rows(raw, NO_BIAS)


def pleg(x, y, season=2018):
    """A point leg with close = 0 so model = x and actual = y."""
    return {"season": season, "model": float(x), "close": 0.0, "actual": float(y), "x": float(x), "y": float(y)}


def ml_rows(k_true, n=40_000, seed=11):
    """Served-shaped moneyline rows: the home team wins with probability m + k_true * (p - m)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        m = rng.uniform(0.35, 0.65)
        p = min(max(m + rng.uniform(-0.15, 0.15), 0.02), 0.98)
        home_won = rng.random() < m + k_true * (p - m)
        rows.append({"season": 2022, "p_home": p, "actual_margin": 3.0 if home_won else -3.0,
                     "ml_home": american(m), "ml_away": american(1 - m)})
    return rows


def test_the_constants_are_the_specs():
    assert game_fit.MIN_GAMES == 300 and game_fit.FIRST_GRADED_SEASON == 2016 and game_fit.FIRST_TEST_SEASON == 2018
    assert game_fit.MARKETS == ("moneyline", "spread", "total") and game_fit.CFB_MODEL_VERSION == "cfb-ratings-v2"
    assert game_fit.LAST_SEASON == 2025 and game_fit.FIT_SPORT == "cfb"


# ---- points: spread and total ----------------------------------------------------------------------------------

def test_point_k_recovers_a_known_k_exactly():
    legs = game_fit.point_legs(served(spread_rows(0.4)), "spread")
    assert len(legs) == 1600
    assert game_fit.k_points(legs) == pytest.approx(0.4)


def test_point_k_clamps_and_zero_variance_gives_zero():
    assert game_fit.k_points([pleg(1, -1), pleg(-2, 2)]) == 0.0            # the model points the wrong way
    assert game_fit.k_points([pleg(1, 3), pleg(-2, -6)]) == 1.0            # the model under-states its own disagreement
    assert game_fit.k_points([pleg(0, 5), pleg(0, -3)]) == 0.0             # model == close everywhere: nothing to learn
    assert game_fit.k_points([]) == 0.0


def test_spread_x_and_y_are_in_expected_home_margin_space_with_positive_meaning_home_favoured():
    # the closing line is +7 (home favoured by 7: lines.parquet stores minus the book's home line); the model says +10;
    # the home team won by 8.5
    [leg] = game_fit.point_legs(served([raw_row(model_margin=10.0, market_spread=7.0, actual_margin=8.5)]), "spread")
    assert (leg["x"], leg["y"]) == (3.0, 1.5)
    [away] = game_fit.point_legs(served([raw_row(model_margin=-10.0, market_spread=-7.0, actual_margin=-8.5)]), "spread")
    assert (away["x"], away["y"]) == (-3.0, -1.5)                           # the mirror image gives the same k


def test_total_legs_use_the_served_total_the_closing_total_and_the_actual_total():
    [leg] = game_fit.point_legs(served([raw_row(model_total=52.0, market_total=50.0, actual_total=51.0)]), "total")
    assert (leg["x"], leg["y"]) == (2.0, 1.0)
    assert game_fit.k_points([leg]) == 0.5


@pytest.mark.parametrize("bad", [dict(market_spread=None), dict(market_spread=float("nan")), dict(actual_margin=None),
                                 dict(model_margin=float("nan"))])
def test_rows_missing_a_number_are_not_spread_legs(bad):
    assert game_fit.point_legs(served([raw_row(**bad)]), "spread") == []


# ---- moneyline --------------------------------------------------------------------------------------------------

def test_moneyline_k_recovers_a_known_k_on_simulated_outcomes():
    legs = game_fit.moneyline_legs(ml_rows(0.30))
    assert len(legs) == 40_000
    assert game_fit.fit_k(legs, "moneyline") == pytest.approx(0.30, abs=0.08)
    assert game_fit.fit_k(legs, "moneyline") == closed_form_k(legs)         # the Part A closed form, not a copy of it


def test_moneyline_legs_exclude_ties_and_unusable_prices():
    ok = {"season": 2022, "p_home": 0.6, "actual_margin": 3.0, "ml_home": -150, "ml_away": 130}
    assert len(game_fit.moneyline_legs([ok])) == 1
    for bad in (dict(actual_margin=0.0), dict(ml_home=None), dict(ml_away=None), dict(ml_home=0), dict(ml_home=-99),
                dict(ml_away=50), dict(ml_home=float("nan")), dict(p_home=None), dict(actual_margin=None)):
        assert game_fit.moneyline_legs([ok | bad]) == [], bad
    [leg] = game_fit.moneyline_legs([ok | {"actual_margin": -1.0}])
    assert leg["y"] == 0 and leg["p"] == 0.6 and 0.5 < leg["m"] < 0.6        # the no-vig home probability of -150 / +130


# ---- the 300-game minimum ---------------------------------------------------------------------------------------

def test_a_market_with_fewer_than_300_graded_games_gets_k_zero():
    strong = [pleg(1.0, 1.0) for _ in range(299)]
    assert game_fit.fit_k(strong, "spread") == 0.0
    assert game_fit.fit_k(strong + [pleg(1.0, 1.0)], "spread") == 1.0
    perfect = {"season": 2022, "p": 0.7, "m": 0.5, "y": 1}
    assert game_fit.fit_k([perfect] * 299, "moneyline") == 0.0
    assert game_fit.fit_k([perfect] * 300, "moneyline") == 1.0


# ---- walk-forward ------------------------------------------------------------------------------------------------

def test_each_season_is_scored_with_a_k_fitted_on_earlier_seasons_only():
    rows = spread_rows(0.0, seasons=(2016, 2017, 2018)) + spread_rows(1.0, seasons=(2019,), seed=4)
    rep = game_fit.season_report(game_fit.point_legs(served(rows), "spread"), "spread")
    assert [r["season"] for r in rep] == [2018, 2019]                       # 2016 and 2017 are history, not test seasons
    by = {r["season"]: r for r in rep}
    assert by[2018]["k"] == 0.0 and by[2018]["n_train"] == 800 and by[2018]["n_test"] == 400
    assert by[2019]["k"] == 0.0 and by[2019]["n_train"] == 1200             # 2019 itself (k_true = 1) never trains 2019
    more = rows + spread_rows(1.0, seasons=(2020,), seed=5)
    later = game_fit.season_report(game_fit.point_legs(served(more), "spread"), "spread")
    assert [(r["season"], r["k"]) for r in later[:2]] == [(r["season"], r["k"]) for r in rep]    # a later season changes nothing earlier
    assert later[2]["season"] == 2020 and 0.1 < later[2]["k"] < 0.45       # 2020 sees 2016-2019


def test_out_of_sample_metrics_for_raw_shrunk_and_market():
    legs = [pleg(2.0, 1.0, season=2018) for _ in range(400)] + [pleg(2.0, 1.0, season=2019) for _ in range(10)]
    r = {x["season"]: x for x in game_fit.season_report(legs, "spread")}[2019]
    assert (r["metric"], r["n_test"], r["n_train"]) == ("mae", 10, 400) and r["k"] == pytest.approx(0.5)
    assert r["raw"] == pytest.approx(1.0) and r["shrunk"] == pytest.approx(0.0) and r["market"] == pytest.approx(1.0)
    ml = [{"season": 2021, "p": 0.8, "m": 0.6, "y": 1}] * 400 + [{"season": 2022, "p": 0.8, "m": 0.6, "y": 1}] * 10
    q = {x["season"]: x for x in game_fit.season_report(ml, "moneyline")}[2022]
    assert q["metric"] == "brier" and q["k"] == 1.0
    assert q["raw"] == pytest.approx(0.04) and q["shrunk"] == pytest.approx(0.04) and q["market"] == pytest.approx(0.16)


# ---- the served conversion --------------------------------------------------------------------------------------

def test_served_rows_use_the_served_conversion_not_the_raw_model():
    cfg = GameLineConfig(sigma_margin=16.0, offset=110, total_max=150, bias_margin=-0.5, bias_total=0.25)
    [row] = game_fit.served_rows([raw_row(model_margin=6.0, model_total=55.0)], cfg)
    want = build_gameline(6.0, 55.0, {"spread_line": None, "total_line": None}, 3, cfg)
    assert row["pred_margin"] == want["pred_margin"] == 6.5 and row["pred_total"] == want["pred_total"] == 54.75
    assert row["p_home"] == want["home_win_prob"] and 0.5 < row["p_home"] < 1


def test_served_rows_carry_the_closing_numbers_and_skip_rows_with_no_model_number():
    rows = game_fit.served_rows([raw_row(ml_home=-150, ml_away=130), raw_row(model_margin=None),
                                 raw_row(model_total=float("nan"))], NO_BIAS)
    assert len(rows) == 1
    r = rows[0]
    assert (r["season"], r["market_spread"], r["market_total"], r["actual_margin"], r["actual_total"], r["ml_home"],
            r["ml_away"]) == (2018, 7.0, 50.0, 8.5, 51.0, -150, 130)


def test_gameline_config_reads_the_committed_cfb_file():
    j = json.loads((ROOT / "assets" / "cfb" / "gameline.json").read_text())
    cfg = game_fit.gameline_config(j)
    assert (cfg.sigma_margin, cfg.offset, cfg.bias_margin, cfg.bias_total) == (j["sigma_margin"], j["offset"],
                                                                               j["bias_margin"], j["bias_total"])


def test_the_committed_cfb_lines_are_in_home_margin_convention():
    import pandas as pd
    lines = pd.read_parquet(ROOT / "assets" / "cfb" / "lines.parquet")
    sched = pd.read_parquet(ROOT / "assets" / "cfb" / "schedules.parquet")
    m = sched.merge(lines, on=["season", "week", "home_team", "away_team"]).dropna(subset=["market_spread"])
    m = m[(m.season >= 2016) & (m.season <= 2025)]
    assert len(m) > 5000
    corr = (m.home_score - m.away_score).corr(m.market_spread)
    assert corr > 0.5, "a positive market_spread must mean the home team is favoured (the expected home margin)"


# ---- the documents ----------------------------------------------------------------------------------------------

def doc_for(rows, **kw):
    return game_fit.build_document(rows, "2026-10-14T12:00:00+00:00", **kw)


def test_the_document_has_fitted_cfb_entries_and_explicit_zero_nfl_and_mlb():
    rows = served(spread_rows(0.4, seasons=(2015, 2016, 2017, 2018, 2019, 2026)))
    doc = doc_for(rows, last_season=2025)
    assert doc["schema"] == 1 and doc["min_games"] == 300 and doc["last_season"] == 2025
    assert doc["model_version"] == {"cfb": "cfb-ratings-v2"} and doc["fitted_at"] == "2026-10-14T12:00:00+00:00"
    sp = doc["markets"]["cfb"]["spread"]
    assert sp["k"] == pytest.approx(0.4) and sp["n"] == 1600 and sp["seasons"] == [2016, 2019]   # 2015 is warm-up, 2026 is past last_season
    assert [r["season"] for r in sp["per_season"]] == [2018, 2019]
    for sport in ("nfl", "mlb"):
        for m in ("moneyline", "spread", "total"):
            e = doc["markets"][sport][m]
            assert e["k"] == 0.0 and e["n"] == 0 and "fewer than 300 graded games" in e["reason"]
    ml = doc["markets"]["cfb"]["moneyline"]
    assert ml["k"] == 0.0 and ml["n"] == 0 and "fewer than 300" in ml["reason"]       # these rows carry no prices
    assert json.loads(json.dumps(doc)) == doc                                          # plain JSON types only


def test_the_site_module_is_a_frozen_table_of_exactly_the_documents_k_values():
    doc = doc_for(served(spread_rows(0.4)))
    js = game_fit.render_js(doc)
    assert js.startswith("/* GENERATED") and "const SHRINK_K = Object.freeze(" in js and js.endswith(");\n")
    table = game_fit.parse_js(js)
    assert set(table) == {"nfl", "cfb", "mlb"}
    for sport in table:
        assert set(table[sport]) == {"moneyline", "spread", "total"}
        for m, k in table[sport].items():
            assert k == doc["markets"][sport][m]["k"]
    assert table["nfl"] == {"moneyline": 0.0, "spread": 0.0, "total": 0.0} == table["mlb"]
    assert game_fit.render_js(doc) == js                                               # deterministic


def test_the_stability_table_lists_every_held_out_season():
    text = game_fit.format_tables(doc_for(served(spread_rows(0.4))))
    assert "cfb spread" in text and "2018" in text and "2019" in text and "nfl spread" in text


def test_the_committed_assets_agree_and_nfl_mlb_are_explicit_zeros():
    doc = json.loads((ROOT / "assets" / "game_shrink_k.json").read_text())
    text = (ROOT / "site" / "js" / "shrink-k.js").read_text()
    assert text == game_fit.render_js(doc), "site/js/shrink-k.js is generated from assets/game_shrink_k.json: re-run scripts/fit_game_shrink.py"
    table = game_fit.parse_js(text)
    for sport in ("nfl", "cfb", "mlb"):
        for m in ("moneyline", "spread", "total"):
            assert 0.0 <= table[sport][m] <= 1.0 and table[sport][m] == doc["markets"][sport][m]["k"]
    for sport in ("nfl", "mlb"):
        assert all(e["k"] == 0.0 and e["n"] == 0 and e["reason"] for e in doc["markets"][sport].values())
    assert doc["model_version"]["cfb"] == "cfb-ratings-v2" and doc["min_games"] == 300
    for m, e in doc["markets"]["cfb"].items():
        assert e["n"] >= 300 or e["k"] == 0.0, m
        assert e["seasons"] == [] or e["seasons"][0] >= 2016, m
