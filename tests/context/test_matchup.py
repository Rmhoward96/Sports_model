import numpy as np
import pandas as pd
import pytest

from sportsmodel.context import matchup
from sportsmodel.context.matchup import (
    COMPONENTS, WEIGHTS, cutoffs_for_season, grade_table, grades_for_games,
    letter, matchup_history, side_scores,
)
from sportsmodel.context.units import METRICS, cfb_unit_games, unit_ratings_asof
from tests.context.synth import league

PROFILE = {"A": {"pass_o": 3, "pass_d": 3}, "B": {"pass_o": 3, "pass_d": 3},
           "C": {"run_o": 3, "run_d": 3}, "D": {"run_o": 3, "run_d": 3}}


def _row(prefix, **kw):
    r = {f"{prefix}_{m}": 0.0 for m in METRICS}
    r.update({f"{prefix}_{k}": v for k, v in kw.items()})
    return r


def test_side_scores_weights_and_pass_rate():
    off = {**_row("off", pass_epa=0.1, pass_success=0.02, pass_explosive=0.01,
                  run_epa=-0.05), "pass_rate": 0.6}
    de = _row("def", pass_epa=0.05, run_success=0.04)
    s = side_scores(off, de)
    assert s["pass"] == pytest.approx(0.6 * 0.15 + 0.25 * 0.02 + 0.15 * 0.01)
    assert s["run"] == pytest.approx(0.6 * -0.05 + 0.25 * 0.04)
    assert s["overall"] == pytest.approx(0.6 * s["pass"] + 0.4 * s["run"])
    scales = {c: 2.0 for c in COMPONENTS}
    assert side_scores(off, de, scales)["pass"] == pytest.approx(s["pass"] / 2)
    assert WEIGHTS == {"epa": 0.6, "success": 0.25, "explosive": 0.15}


def test_strong_pass_o_vs_weak_pass_d_grades_high_both_ways():
    ug = league(range(2020, 2025), profile=PROFILE, seed=5)
    cut = cutoffs_for_season(ug, 2024)
    ratings = unit_ratings_asof(ug, 2024, 8)
    g = grades_for_games(pd.DataFrame([{"game_pk": 1, "home_team": "A", "away_team": "B"},
                                       {"game_pk": 2, "home_team": "E", "away_team": "F"}]),
                         ratings, cut)
    ab = g[g["game_pk"] == 1].set_index("side")
    assert list(ab["pass"]) == ["A", "A"]
    assert ab.loc["home", "team"] == "A" and ab.loc["home", "opponent"] == "B"
    assert ab.loc["away", "team"] == "B"
    assert (ab["overall"] <= "B").all()             # A or B overall
    ef = g[g["game_pk"] == 2]
    assert set(ef["pass"]) <= {"B", "C", "D"}       # average vs average -> middle


def test_symmetric_run_case():
    ug = league(range(2020, 2025), profile=PROFILE, seed=5)
    cut = cutoffs_for_season(ug, 2024)
    ratings = unit_ratings_asof(ug, 2024, 8)
    g = grades_for_games(pd.DataFrame([{"game_pk": 9, "home_team": "C", "away_team": "D"}]),
                         ratings, cut)
    assert list(g["run"]) == ["A", "A"]
    assert "A" not in set(g["pass"])


def test_history_uses_previous_three_seasons_walk_forward():
    ug = league(range(2019, 2025), n_weeks=6, seed=2)
    h = matchup_history(ug, [2021, 2022, 2023])
    assert sorted(h["season"].unique()) == [2021, 2022, 2023]
    assert len(h) == 3 * 6 * 8
    # the row for a week-3 game equals components built from ratings as of week 3
    row = h[(h["season"] == 2022) & (h["week"] == 3) & (h["team"] == "A")].iloc[0]
    r = unit_ratings_asof(ug, 2022, 3).set_index("team")
    opp = row["opponent"]
    assert row["pass_epa"] == pytest.approx(r.loc["A", "off_pass_epa"] + r.loc[opp, "def_pass_epa"])
    assert row["games"] == 2


def test_cutoffs_from_prior_seasons_only():
    ug = league(range(2019, 2026), n_weeks=6, seed=4)
    cut = cutoffs_for_season(ug, 2025)
    assert cut["seasons"] == [2022, 2023, 2024]
    # current and future seasons never move the frozen table
    other = ug.copy()
    m = other["season"] >= 2025
    other.loc[m, list(METRICS)] += 1.0
    assert cutoffs_for_season(other, 2025) == cut
    assert cutoffs_for_season(ug[ug["season"] < 2025], 2025) == cut
    # ...but prior seasons do
    other = ug.copy()
    m = other["season"] == 2023
    other.loc[m, "pass_epa"] = other.loc[m, "pass_epa"] * 3
    assert cutoffs_for_season(other, 2025) != cut


def test_grade_table_percentiles_and_letters():
    h = pd.DataFrame({c: np.linspace(-1, 1, 101) for c in COMPONENTS})
    h["pass_rate"] = 0.5
    h["season"] = 2023
    t = grade_table(h)
    assert set(t["cutoffs"]) == {"overall", "pass", "run"}
    c = t["cutoffs"]["pass"]
    assert c["p10"] < c["p30"] < c["p70"] < c["p90"]
    assert letter(c["p90"] + 1e-9, c) == "A"
    assert letter(c["p90"] - 1e-9, c) == "B"
    assert letter(c["p30"], c) == "C"
    assert letter(c["p10"] - 1e-9, c) == "F"
    assert letter(None, c) is None
    # letters over the history reproduce ~10/20/40/20/10
    sc = matchup.score_frame(h, t["scales"])
    counts = pd.Series([letter(v, t["cutoffs"]["overall"]) for v in sc["overall"]]).value_counts(normalize=True)
    assert counts["A"] == pytest.approx(0.10, abs=0.02)
    assert counts["C"] == pytest.approx(0.40, abs=0.02)


def test_empty_history_gives_no_cutoffs():
    ug = league([2024], n_weeks=4)
    assert cutoffs_for_season(ug, 2024) is None
    g = grades_for_games(pd.DataFrame([{"game_pk": 1, "home_team": "A", "away_team": "B"}]),
                         unit_ratings_asof(ug, 2024, 3), None)
    assert g["overall"].isna().all() and g["overall_score"].isna().all()
    assert g["units"].notna().all()


def test_early_flag():
    ug = league(range(2020, 2025), seed=5)
    cut = cutoffs_for_season(ug, 2024)
    games = pd.DataFrame([{"game_pk": 1, "home_team": "A", "away_team": "B"}])
    assert grades_for_games(games, unit_ratings_asof(ug, 2024, 3), cut)["early"].all()
    assert not grades_for_games(games, unit_ratings_asof(ug, 2024, 4), cut)["early"].any()


def test_grades_leak_free():
    ug = league(range(2020, 2025), seed=5)
    games = pd.DataFrame([{"game_pk": 1, "home_team": "A", "away_team": "B"}])
    base = grades_for_games(games, unit_ratings_asof(ug, 2024, 7), cutoffs_for_season(ug, 2024))
    other = ug.copy()
    m = (other["season"] == 2024) & (other["week"] >= 7)
    other.loc[m, list(METRICS)] = -5.0
    alt = grades_for_games(games, unit_ratings_asof(other, 2024, 7), cutoffs_for_season(other, 2024))
    pd.testing.assert_frame_equal(base, alt)


def test_unknown_or_fcs_opponent_gets_no_grade():
    ug = league(range(2020, 2025), seed=5)
    cut = cutoffs_for_season(ug, 2024)
    ratings = unit_ratings_asof(ug, 2024, 6)
    games = pd.DataFrame([
        {"game_pk": 1, "home_team": "A", "away_team": "FCS"},
        {"game_pk": 2, "home_team": "A", "away_team": "B", "home_is_fbs": True, "away_is_fbs": False},
        {"game_pk": 3, "home_team": "A", "away_team": "B", "home_is_fbs": True, "away_is_fbs": True},
        {"game_pk": 4, "home_team": "A", "away_team": "B", "home_is_fbs": np.True_,
         "away_is_fbs": np.False_},
    ]).astype({"home_is_fbs": object, "away_is_fbs": object})
    g = grades_for_games(games, ratings, cut)
    none = g[g["game_pk"].isin([1, 2, 4])]
    assert len(none) == 6
    for c in ("overall", "pass", "run", "overall_score", "pass_score", "run_score"):
        assert none[c].isna().all()
    assert g[g["game_pk"] == 3]["overall"].notna().all()


def test_cfb_end_to_end_fcs_none():
    rows = []
    teams = ["1", "2", "3", "4"]
    rng = np.random.default_rng(0)
    gid = 0
    for s in (2021, 2022, 2023, 2024, 2025):
        for w in range(1, 7):
            for h, a in (("1", "2"), ("3", "4")) if w % 2 else (("1", "3"), ("2", "4")):
                gid += 1
                for t, o in ((h, a), (a, h)):
                    r = dict(season=s, week=w, game_id=gid, team=t, opponent=o)
                    for u in ("off", "def"):
                        for k in ("ppa", "success", "explosiveness", "pass_ppa", "pass_success",
                                  "pass_explosiveness", "rush_ppa", "rush_success",
                                  "rush_explosiveness", "plays"):
                            r[f"{u}_{k}"] = float(rng.normal(0.3, 0.1))
                    r["off_pass_plays"], r["off_rush_plays"] = 30.0, 35.0
                    rows.append(r)
    ug = cfb_unit_games(pd.DataFrame(rows))
    cut = cutoffs_for_season(ug, 2025)
    ratings = unit_ratings_asof(ug, 2025, 4)
    # "99" is an FCS school: no advanced rows ever -> no rating -> no grade
    g = grades_for_games(pd.DataFrame([{"game_pk": 7, "home_team": "1", "away_team": "99"},
                                       {"game_pk": 8, "home_team": "1", "away_team": "2"}]),
                         ratings, cut)
    assert g[g["game_pk"] == 7]["overall"].isna().all()
    assert g[g["game_pk"] == 8]["overall"].notna().all()
    assert g[g["game_pk"] == 8]["units"].iloc[0]["pass_rate"] == pytest.approx(30 / 65, abs=1e-4)
