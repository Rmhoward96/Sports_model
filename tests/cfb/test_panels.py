import math

import pandas as pd
import pytest

from sportsmodel.cfb import panels


def _game(season, home, away, hq, aq):
    return {"season": season, "season_type": "regular", "home_team": home, "away_team": away,
            **{f"home_q{i + 1}": float(v) for i, v in enumerate(hq)},
            **{f"away_q{i + 1}": float(v) for i, v in enumerate(aq)}}


def test_quarter_shares_sum_to_one_and_follow_the_pattern():
    games = pd.DataFrame([_game(2025, "A", "B", [14, 0, 0, 0], [0, 7, 0, 0]),
                          _game(2025, "A", "C", [7, 0, 0, 0], [0, 0, 7, 0]),
                          _game(2026, "B", "C", [0, 0, 0, 7], [0, 0, 0, 7])])
    sh = panels.quarter_shares(games, 2026).set_index("team")
    assert list(sh.index) == ["A", "B", "C"] and list(sh["games_used"]) == [2, 2, 2]
    for t in sh.index:
        assert sum(sh.loc[t, f"scored_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
        assert sum(sh.loc[t, f"allowed_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
    lq = [21 / 49, 7 / 49, 7 / 49, 14 / 49]          # league scored points per quarter over all six team-games: 21 / 7 / 7 / 14 of 49
    w = 2 / 14                                                           # n / (n + 12)
    assert sh.loc["A", "scored_q1"] == pytest.approx(w * 1.0 + (1 - w) * lq[0])
    assert sh.loc["A", "scored_q2"] == pytest.approx((1 - w) * lq[1])
    assert sh.loc["A", "allowed_q2"] == pytest.approx(w * 0.5 + (1 - w) * lq[1])


def test_quarter_shares_small_n_is_close_to_the_league_and_large_n_to_the_team():
    big = [_game(2026, "A", "B", [10, 0, 0, 0], [0, 0, 0, 10]) for _ in range(120)]
    one = [_game(2026, "C", "D", [0, 10, 0, 0], [0, 0, 10, 0])]
    sh = panels.quarter_shares(pd.DataFrame(big + one), 2026).set_index("team")
    lg = 10 / 2420                                   # league share of q2: only C's single game scored there (10 of 2,420 points)
    assert sh.loc["A", "scored_q1"] == pytest.approx(120 / 132 * 1.0 + 12 / 132 * (1200 / 2420))      # 120 games: w = 120 / 132
    assert sh.loc["C", "scored_q2"] == pytest.approx(1 / 13 + 12 / 13 * lg)                          # one game: w = 1 / 13, so mostly league
    assert sh.loc["C", "games_used"] == 1


def test_quarter_shares_window_missing_quarters_fbs_filter_and_empty():
    games = pd.DataFrame([_game(2023, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]),          # too old (2026 - 2 = 2024)
                          _game(2025, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]),
                          {**_game(2025, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]), "home_q3": float("nan")}])
    sh = panels.quarter_shares(games, 2026)
    assert list(sh["games_used"]) == [1, 1]
    assert list(panels.quarter_shares(games, 2026, fbs={"A"})["team"]) == ["A"]
    assert panels.quarter_shares(games.iloc[0:1], 2026).empty
    assert list(panels.quarter_shares(games.iloc[0:0], 2026).columns) == panels.SHARE_COLUMNS


def test_quarter_shares_a_team_that_never_scored_falls_back_to_the_league():
    games = pd.DataFrame([_game(2026, "A", "B", [0, 0, 0, 0], [7, 7, 7, 7]), _game(2026, "A", "C", [0, 0, 0, 0], [3, 3, 3, 3])])
    sh = panels.quarter_shares(games, 2026).set_index("team")
    assert sum(sh.loc["A", f"scored_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
    assert sh.loc["A", "scored_q1"] == pytest.approx(0.25)


def test_to_rows_python_scalars_and_none():
    rows = panels.to_rows(panels.quarter_shares(pd.DataFrame([_game(2026, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7])]), 2026))
    assert type(rows[0]["games_used"]) is int and type(rows[0]["scored_q1"]) is float and rows[0]["team"] == "A"
    assert panels.to_rows(pd.DataFrame({"x": [float("nan"), 1.5], "n_ranked": [3.0, None]})) == [{"x": None, "n_ranked": 3}, {"x": 1.5, "n_ranked": None}]


def _h(team, opp, week, gid, def_ev, def_pl, off_ev, off_pl):
    return {"season": 2026, "week": week, "season_type": "regular", "game_id": gid, "team": team, "opponent": opp,
            "def_havoc_events": def_ev, "def_plays": def_pl, "off_havoc_events": off_ev, "off_plays": off_pl}


def _adv(team, week, gid, off_x, off_pl, def_x, def_pl):
    return {"season": 2026, "week": week, "season_type": "regular", "game_id": gid, "team": team,
            "off_explosiveness": off_x, "off_plays": off_pl, "def_explosiveness": def_x, "def_plays": def_pl}


def _fixture():
    """Teams 1..4, three weeks each (weeks 1-3), plus team 5 with two games. Havoc created: 1 > 2 > 3 > 4."""
    pairs = {1: [(1, 2), (3, 4)], 2: [(1, 3), (2, 4)], 3: [(1, 4), (2, 3)]}
    created = {"1": 20, "2": 15, "3": 10, "4": 5}          # per 100 plays
    allowed = {"1": 5, "2": 10, "3": 15, "4": 20}
    hv, ad, ts, gid = [], [], [], 0
    for w, ps in pairs.items():
        for a, b in ps:
            gid += 1
            for t, o in ((str(a), str(b)), (str(b), str(a))):
                hv.append(_h(t, o, w, gid, created[t], 100, allowed[t], 100))
                ad.append(_adv(t, w, gid, 0.1 * int(t), 50 + int(t), 0.5 - 0.1 * int(t), 60))
                ts.append({"season": 2026, "week": w, "season_type": "regular", "game_id": gid, "team": t,
                           "opponent": o, "giveaways": float(int(t)), "takeaways": float(5 - int(t))})
    for w in (1, 2):                                          # team 5 plays itself... a second opponent pool
        gid += 1
        hv.append(_h("5", "6", w, gid, 99, 100, 1, 100)); hv.append(_h("6", "5", w, gid, 1, 100, 99, 100))
    return pd.DataFrame(hv), pd.DataFrame(ad), pd.DataFrame(ts)


def test_team_insights_rates_ranks_and_n_ranked():
    hv, ad, ts = _fixture()
    ins = panels.team_insights(hv, ad, ts, 2026).set_index("team")
    assert ins.loc["1", "games"] == 3 and ins.loc["1", "through_week"] == 3
    assert ins.loc["1", "def_havoc_rate"] == pytest.approx(0.20)
    assert ins.loc["1", "off_havoc_allowed_rate"] == pytest.approx(0.05)
    assert [int(ins.loc[t, "def_havoc_rank"]) for t in "1234"] == [1, 2, 3, 4]             # higher created = better
    assert [int(ins.loc[t, "off_havoc_allowed_rank"]) for t in "1234"] == [1, 2, 3, 4]     # lower allowed = better
    assert ins.loc["1", "turnover_margin"] == pytest.approx(3 * (4 - 1))                  # takeaways 4, giveaways 1 per game
    assert ins.loc["1", "turnover_margin_per_game"] == pytest.approx(3.0)
    assert [int(ins.loc[t, "turnover_margin_rank"]) for t in "1234"] == [1, 2, 3, 4]
    assert [int(ins.loc[t, "off_explosiveness_rank"]) for t in "1234"] == [4, 3, 2, 1]     # 0.1 * team id
    assert [int(ins.loc[t, "def_explosiveness_allowed_rank"]) for t in "1234"] == [4, 3, 2, 1]   # 0.5 - 0.1 * id: lower allowed = better
    assert set(ins["n_ranked"]) == {4}
    for t in ("5", "6"):                                                                    # two games: values yes, ranks no
        assert ins.loc[t, "games"] == 2 and pd.isna(ins.loc[t, "def_havoc_rank"]) and pd.isna(ins.loc[t, "turnover_margin_rank"])
    assert ins.loc["5", "def_havoc_rate"] == pytest.approx(0.99)
    assert math.isnan(ins.loc["5", "turnover_margin"])                                      # no team_stats rows -> NaN, never 0
    assert list(panels.team_insights(hv, ad, None, 2026).loc[:, "turnover_margin"].isna()) == [True] * 6


def test_team_insights_filters_season_postseason_and_weeks_beyond_havoc():
    hv, ad, ts = _fixture()
    post = hv.iloc[:2].assign(season_type="postseason", week=9)
    old = hv.iloc[:2].assign(season=2025)
    ad2 = pd.concat([ad, ad.iloc[:1].assign(week=8)], ignore_index=True)        # advanced is ahead of havoc: ignored
    got = panels.team_insights(pd.concat([hv, post, old]), ad2, ts, 2026)
    assert got["through_week"].eq(3).all() and got.set_index("team").loc["1", "games"] == 3
    assert panels.team_insights(hv, ad, ts, 2030).empty
    assert list(panels.team_insights(hv.iloc[0:0], ad, ts, 2026).columns) == panels.INSIGHT_COLUMNS
    assert set(panels.team_insights(hv, ad, ts, 2026, fbs={"1", "2"})["team"]) == {"1", "2"}


def test_to_rows_insight_ranks_are_ints_and_missing_values_are_none():
    hv, ad, ts = _fixture()
    rows = panels.to_rows(panels.team_insights(hv, ad, ts, 2026))
    r5 = next(r for r in rows if r["team"] == "5")
    assert r5["def_havoc_rank"] is None and r5["turnover_margin"] is None and r5["games"] == 2
    r1 = next(r for r in rows if r["team"] == "1")
    assert type(r1["def_havoc_rank"]) is int and r1["def_havoc_rank"] == 1 and type(r1["def_havoc_rate"]) is float
