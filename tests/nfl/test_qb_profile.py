import numpy as np
import pandas as pd

from sportsmodel.nfl import qb_profile as qp


def _w(pid, season, week, team, opp, att, yds, tds=1, ints=0, sacks=2):
    return {"player_id": pid, "season": season, "week": week, "season_type": "REG",
            "recent_team": team, "opponent_team": opp, "position": "QB", "attempts": att,
            "passing_yards": yds, "passing_tds": tds, "passing_interceptions": ints,
            "sacks_suffered": sacks}


def _weekly():
    rows = []
    for yr in (2016, 2017, 2018):                       # veteran starter, then benched
        for wk in range(1, 17):
            rows.append(_w("VET", yr, wk, "MIN", "GB", 30, 210))
    for wk in range(1, 17):                              # 2019-2024: other starters, league avg 7.0
        for yr in range(2019, 2025):
            rows.append(_w(f"S{yr}", yr, wk, "MIN", "GB", 30, 210))
    rows.append(_w("ROOK", 2024, 16, "MIN", "GB", 5, 20, tds=0))   # thin backup
    return pd.DataFrame(rows)


def test_qb_games_rates():
    g = qp.qb_games(_weekly())
    r = g[g.player_id == "VET"].iloc[0]
    assert r.ypa == 7.0 and np.isclose(r.sack_rate, 2 / 32)


def test_profile_weights_long_career_and_shrinks_thin_one():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    keys = pd.DataFrame({"player_id": ["VET", "ROOK", "NEW"], "season": [2025] * 3, "week": [4] * 3})
    p = qp.profile_asof(qga, keys, H=2.0, k=200.0, repl=repl).set_index("player_id")
    # VET: 1,440 attempts but 7-9 seasons old -> n_eff ~ 90 at H=2 -> (90*7 + 200*5)/290 ~ 5.6
    assert 5.4 < p.loc["VET", "qb_ypa"] < 7.0
    # ROOK: 5 attempts at 4.0 ypa -> (5*4 + 200*5)/205 ~ 4.98: essentially replacement
    assert 4.9 < p.loc["ROOK", "qb_ypa"] < 5.0
    # a longer half-life keeps more of the veteran's own record
    p4 = qp.profile_asof(qga, keys, H=4.0, k=200.0, repl=repl).set_index("player_id")
    assert p4.loc["VET", "qb_ypa"] > p.loc["VET", "qb_ypa"]
    assert p.loc["NEW", "qb_ypa"] == 5.0 and p.loc["NEW", "qb_n_eff"] == 0.0


def test_profile_is_strictly_prior():
    w = _weekly()
    qga = qp.opponent_adjust(qp.qb_games(w))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    keys = pd.DataFrame({"player_id": ["VET"], "season": [2017], "week": [5]})
    a = qp.profile_asof(qga, keys, 2.0, 200.0, repl)
    w2 = pd.concat([w, pd.DataFrame([_w("VET", 2017, 5, "MIN", "GB", 40, 900)])], ignore_index=True)
    b = qp.profile_asof(qp.opponent_adjust(qp.qb_games(w2)), keys, 2.0, 200.0, repl)
    pd.testing.assert_frame_equal(a, b)


def test_qb1_skips_out_and_honors_override():
    depth = pd.DataFrame({"season": [2024] * 3, "week": [5] * 3, "club_code": ["CHI"] * 3,
                          "position": ["QB"] * 3, "gsis_id": ["CW", "TB", "CK"], "depth_team": [1, 2, 3]})
    inj = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"], "gsis_id": ["CW"],
                        "report_status": ["Out"]})
    tw = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"]})
    assert qp.qb1_by_team_week(depth, inj, tw).iloc[0].qb1_id == "TB"
    assert qp.qb1_by_team_week(depth, inj, tw, override={(2024, 5, "CHI"): "CK"}).iloc[0].qb1_id == "CK"


def test_ratio_is_one_for_the_usual_starter_and_below_one_for_a_weaker_backup():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    tw = pd.DataFrame({"season": [2024, 2024], "week": [10, 10], "team": ["MIN", "MIN"],
                       "opponent": ["GB", "GB"]})
    qb1 = pd.DataFrame({"season": [2024, 2024], "week": [10, 10], "team": ["MIN", "MIN"],
                        "qb1_id": ["S2024", "ROOK"]})
    same = qp.team_qb_features(tw.iloc[:1], qb1.iloc[:1], qga, 2.0, 200.0, repl).iloc[0]
    back = qp.team_qb_features(tw.iloc[1:], qb1.iloc[1:], qga, 2.0, 200.0, repl).iloc[0]
    assert np.isclose(same.qb_ratio_ypa, 1.0, atol=0.02) and same.qb_changed == 0.0
    assert back.qb_ratio_ypa < 0.9 and back.qb_changed == 1.0



def test_tune_returns_a_grid_point_and_ties_keep_the_first():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    res = qp.tune(qga, [2023, 2024])
    grid_pts = {(h, k) for h in (0.5, 1, 1.5, 2, 3, 4) for k in (25, 50, 100, 200, 400, 800)}
    assert (res["H"], res["k"]) in grid_pts and len(res["grid"]) == 36
    assert set(res) >= {"H", "k", "mse", "grid"} and "se" not in res and "best" not in res
    assert np.isfinite(res["mse"])
    # the plain argmin is chosen: the first grid point with the minimum mse
    lo = min(g["mse"] for g in res["grid"])
    first = next(g for g in res["grid"] if g["mse"] == lo)
    assert (res["H"], res["k"], res["mse"]) == (first["H"], first["k"], lo)
    # a one-point grid passed twice: identical scores -> the first grid point is kept
    tie = qp.tune(qga, [2024], grid_h=(2.0, 2.0), grid_k=(200.0,))
    assert len(tie["grid"]) == 2 and tie["grid"][0]["mse"] == tie["grid"][1]["mse"]
    assert (tie["H"], tie["k"], tie["mse"]) == (2.0, 200.0, tie["grid"][0]["mse"])
