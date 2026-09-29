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


# --- opponent adjustment / replacement on a multi-defense fixture ----------------------------

def _two_defenses(prev=False, extra=()):
    """2020 weeks 1-4: MIN's QB M faces a strong pass D (GB, 4.0 ypa allowed), CHI's QB C a weak
    one (DET, 10.0). 60 attempts a game, so a defense's window reaches 100 attempts in week 3.
    prev=True adds 2019 (GB allowed 5.0, DET 9.0 on 600 attempts each; league 7.0)."""
    rows = []
    for wk in range(1, 5):
        rows += [_w("M", 2020, wk, "MIN", "GB", 60, 240), _w("C", 2020, wk, "CHI", "DET", 60, 600)]
    if prev:
        for wk in range(1, 11):
            rows += [_w("M19", 2019, wk, "MIN", "GB", 60, 300), _w("C19", 2019, wk, "CHI", "DET", 60, 540)]
    return pd.DataFrame(rows + list(extra))


def _shift(qga):
    s = qga.assign(shift=qga["ypa"] - qga["ypa_adj"])
    return s.set_index(["player_id", "season", "week"])["shift"].sort_index()


def test_opponent_shift_uses_only_that_seasons_earlier_weeks():
    s = _shift(qp.opponent_adjust(qp.qb_games(_two_defenses())))
    # weeks 1-2: < 100 attempts faced and no previous season -> no adjustment
    for wk in (1, 2):
        assert s[("M", 2020, wk)] == 0.0 and s[("C", 2020, wk)] == 0.0
    # week 3: GB allowed 4.0 over weeks 1-2, league 7.0 -> M's game is shifted by -3 (DET: +3)
    assert np.isclose(s[("M", 2020, 3)], -3.0) and np.isclose(s[("C", 2020, 3)], 3.0)
    # GB is even stronger in week 3 (1.0 ypa): week 3's own shift is unchanged, week 4's moves
    w = _two_defenses()
    w.loc[(w.player_id == "M") & (w.week == 3), "passing_yards"] = 60
    s2 = _shift(qp.opponent_adjust(qp.qb_games(w)))
    assert np.isclose(s2[("M", 2020, 3)], -3.0)
    assert np.isclose(s2[("M", 2020, 4)], 540 / 180 - (540 + 1800) / 360)   # 3.0 - 6.5


def test_changing_a_week_w_game_changes_adj_only_after_week_w():
    base = qp.opponent_adjust(qp.qb_games(_two_defenses()))
    for wk in (1, 2, 3):
        w = _two_defenses()
        w.loc[(w.player_id == "C") & (w.week == wk), "passing_yards"] = 0
        new = qp.opponent_adjust(qp.qb_games(w))
        same = (base.week <= wk) & ~((base.player_id == "C") & (base.week == wk))
        np.testing.assert_allclose(new.loc[same, "ypa_adj"], base.loc[same, "ypa_adj"], rtol=0, atol=1e-12)
        # at least one later game (from week 3 on, when windows reach 100 attempts) moves
        later = base.week > max(wk, 2)
        assert (np.abs(new.loc[later, "ypa_adj"] - base.loc[later, "ypa_adj"]) > 1e-6).any()


def test_previous_season_allowed_rate_used_under_100_attempts():
    s = _shift(qp.opponent_adjust(qp.qb_games(_two_defenses(prev=True))))
    for wk in (1, 2):          # < 100 attempts faced this season: 2019's GB 5.0 vs league 7.0
        assert np.isclose(s[("M", 2020, wk)], -2.0) and np.isclose(s[("C", 2020, wk)], 2.0)
    assert np.isclose(s[("M", 2020, 3)], -3.0)          # >= 100 attempts: this season's window
    assert s[("M19", 2019, 1)] == 0.0                     # 2019 week 1: nothing known


def test_replacement_excludes_team_leaders_and_later_seasons():
    backups = [_w("MB", 2020, 4, "MIN", "GB", 10, 50), _w("CB", 2020, 4, "CHI", "DET", 10, 80)]
    later = [_w("Z", 2021, 1, "MIN", "GB", 60, 420), _w("ZB", 2021, 1, "MIN", "GB", 10, 300)]
    qga = qp.opponent_adjust(qp.qb_games(_two_defenses(extra=backups + later)))
    r = qp.replacement(qga, 2021)
    b = qga[qga.player_id.isin(["MB", "CB"])]
    assert np.isclose(r["ypa_adj"], np.average(b.ypa_adj, weights=b.att))
    allq = qga[qga.season < 2021]
    assert not np.isclose(r["ypa_adj"], np.average(allq.ypa_adj, weights=allq.att))
    # seasons >= before_season are ignored; 2022 picks up 2021's backup
    qga0 = qp.opponent_adjust(qp.qb_games(_two_defenses(extra=backups)))
    assert qp.replacement(qga0, 2021) == r
    assert qp.replacement(qga, 2022)["ypa_adj"] > r["ypa_adj"] + 1.0


def test_vectorized_matches_brute_force():
    rng = np.random.default_rng(0)
    teams = [("MIN", "GB"), ("GB", "MIN"), ("CHI", "DET"), ("DET", "CHI"), ("MIN", "CHI"), ("DET", "GB")]
    rows = []
    for yr in range(2017, 2021):
        for wk in range(1, 9):
            for i, (t, o) in enumerate(teams[:4] if wk % 2 else teams[2:]):
                rows.append(_w(f"{t}{(yr + i) % 3}", yr, wk, t, o, int(rng.integers(0, 45)),
                               int(rng.integers(0, 400)), tds=int(rng.integers(0, 4)),
                               ints=int(rng.integers(0, 3)), sacks=int(rng.integers(0, 5))))
    qga = qp.opponent_adjust(qp.qb_games(pd.DataFrame(rows)))

    def rate(df, c):
        ok = df[c].notna()
        return (df.att[ok] * df[c][ok]).sum() / df.att[ok].sum()

    for _, r in qga.iterrows():
        for c in qp.RATES:
            if np.isnan(r[c]):
                continue
            win = qga[(qga.season == r.season) & (qga.week < r.week)]
            ow = win[win.opponent == r.opponent]
            pv = qga[qga.season == r.season - 1]
            po = pv[pv.opponent == r.opponent]
            if ow.att.sum() >= 100:
                sh = rate(ow, c) - rate(win, c)
            elif po.att.sum() > 0:
                sh = rate(po, c) - rate(pv, c)
            else:
                sh = 0.0
            assert np.isclose(r[c] - sh, r[f"{c}_adj"], atol=1e-12)
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    keys = qga[["player_id", "season", "week"]].drop_duplicates().sample(40, random_state=1)
    keys = keys.reset_index(drop=True)
    H, k = 1.5, 100.0
    p = qp.profile_asof(qga, keys, H, k, repl)
    for i, kr in keys.iterrows():
        g = qga[(qga.player_id == kr.player_id) & (qga.season * 100 + qga.week < kr.season * 100 + kr.week)]
        wt = g.att * 0.5 ** (((kr.season + (kr.week - 1) / 18) - (g.season + (g.week - 1) / 18)) / H)
        assert np.isclose(p.loc[i, "qb_n_eff"], wt.sum())
        for c in qp.RATES:
            ok = g[f"{c}_adj"].notna()
            v = ((wt[ok] * g[f"{c}_adj"][ok]).sum() + k * repl[f"{c}_adj"]) / (wt[ok].sum() + k)
            assert np.isclose(p.loc[i, f"qb_{c}"], v)


# --- QB1 selection and team features: edge cases ---------------------------------------------

def _chi_depth(weeks_starters):
    rows = []
    for wk, order in weeks_starters.items():
        for d, gid in enumerate(order, start=1):
            rows.append({"season": 2024, "week": wk, "club_code": "CHI", "position": "QB",
                         "gsis_id": gid, "depth_team": str(d)})
    return pd.DataFrame(rows)


def test_qb1_skips_doubtful():
    inj = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"], "gsis_id": ["CW"],
                        "report_status": ["Doubtful"]})
    tw = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"]})
    assert qp.qb1_by_team_week(_chi_depth({5: ["CW", "TB"]}), inj, tw).iloc[0].qb1_id == "TB"
    q = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"], "gsis_id": ["CW"],
                      "report_status": ["Questionable"]})
    assert qp.qb1_by_team_week(_chi_depth({5: ["CW", "TB"]}), q, tw).iloc[0].qb1_id == "CW"


def test_qb1_falls_back_to_the_latest_earlier_chart():
    depth = _chi_depth({3: ["CW", "TB"], 5: ["TB", "CW"]})
    tw = pd.DataFrame({"season": [2024] * 4, "week": [2, 3, 4, 5], "team": ["CHI"] * 4})
    q = qp.qb1_by_team_week(depth, None, tw)
    assert pd.isna(q.qb1_id.iloc[0])                  # no chart yet
    assert list(q.qb1_id.iloc[1:]) == ["CW", "CW", "TB"]   # week 4 uses week 3's chart


def test_ratio_is_clamped():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    tw = pd.DataFrame({"season": [2024], "week": [10], "team": ["MIN"]})
    qb1 = tw.assign(qb1_id=["NEW"])
    hi = {"ypa_adj": 50.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    lo = {"ypa_adj": 0.5, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    assert qp.team_qb_features(tw, qb1, qga, 2.0, 200.0, hi).iloc[0].qb_ratio_ypa == 1.3
    assert qp.team_qb_features(tw, qb1, qga, 2.0, 200.0, lo).iloc[0].qb_ratio_ypa == 0.6


def test_no_prior_team_game_gives_nan_ratio_and_qb_changed():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    tw = pd.DataFrame({"season": [2016, 2024], "week": [1, 10], "team": ["MIN", "NYJ"]})
    f = qp.team_qb_features(tw, tw.assign(qb1_id=["VET", "X"]), qga, 2.0, 200.0, repl)
    assert f[["qb_ratio_ypa", "qb_ratio_td", "qb_ratio_sack", "qb_changed"]].isna().all().all()
    assert (f.qb_ypa == 5.0).all() and (f.qb_n_eff == 0.0).all()   # no prior games -> replacement
