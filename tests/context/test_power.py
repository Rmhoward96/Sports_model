import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb.teams import load_fbs_ids
from sportsmodel.context.power import cfb_power, nfl_market_scale, nfl_power, rankings
from sportsmodel.context.units import unit_ratings_asof
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from tests.context.synth import TEAMS, league, round_robin

ELO = EloConfig(k=40, hfa_elo=70, carryover=0.9, base=1500.0)
BLEND = BlendConfig(w_sos=0.45, srs_min_games=3)
N_CFB = 16
CFB_TEAMS = [f"T{i:02d}" for i in range(N_CFB)]


# ---------------------------------------------------------------- CFB fixtures
def _cfb_schedule(seasons=(2023, 2024), n_weeks=12, strong="T00", edge=10.0,
                  home_edge=3.0, fcs_weeks=(), seed=3, noise=0.0):
    """Round-robin FBS league; ``strong`` beats everyone by ``edge`` more than
    the home edge would say. ``fcs_weeks``: weeks where T01 hosts the pooled
    FCS pseudo-team (win by 30)."""
    rng = np.random.default_rng(seed)
    rr = round_robin(CFB_TEAMS)
    rows, pk = [], 1
    for s in seasons:
        for w in range(1, n_weeks + 1):
            for h, a in rr[(w - 1) % len(rr)]:
                if w % 2 == 0:                      # balance home / away
                    h, a = a, h
                m = home_edge + edge * ((h == strong) - (a == strong)) + noise * rng.standard_normal()
                m = float(np.round(m))
                rows.append(dict(season=s, week=w, home_team=h, away_team=a,
                                 home_score=30 + m, away_score=30.0, game_type="REG",
                                 game_pk=pk, neutral_site=False))
                pk += 1
            if w in fcs_weeks:
                rows.append(dict(season=s, week=w, home_team="T01", away_team="FCS",
                                 home_score=44.0, away_score=14.0, game_type="REG",
                                 game_pk=pk, neutral_site=False))
                pk += 1
    return pd.DataFrame(rows)


FBS = set(CFB_TEAMS)


def test_cfb_plus_ten_team_ranks_first_rating_about_ten_srs_side():
    # pure SRS blend: the strong team is +10 vs each other team, so vs the
    # league average (which includes it) it is 10*(n-1)/n on a neutral field.
    sched = _cfb_schedule(home_edge=0.0)     # SRS has no home term
    p = cfb_power(sched, ELO, BlendConfig(w_sos=1.0, srs_min_games=3), (2024, 11), fbs=FBS)
    p = p.set_index("team")
    assert p["rating"].idxmax() == "T00"
    assert p.loc["T00", "rating"] == pytest.approx(10 * (N_CFB - 1) / N_CFB, abs=1e-6)
    assert p.drop("T00")["rating"].abs().max() < 1.0


def test_cfb_plus_ten_team_ranks_first_real_blend():
    sched = _cfb_schedule(seasons=(2021, 2022, 2023, 2024), noise=12.0)
    p = cfb_power(sched, ELO, BLEND, (2024, 11), fbs=FBS)
    r = rankings(p, None, None, None).set_index("team")
    assert r.loc["T00", "rank"] == 1
    assert 7.0 < r.loc["T00", "rating"] < 14.0


def test_cfb_neutral_field_removes_hfa():
    sched = _cfb_schedule(noise=10.0)
    for hfa in (0.0, 70.0, 140.0):
        cfg = EloConfig(k=40, hfa_elo=hfa, carryover=0.9, base=1500.0)
        p = cfb_power(sched, cfg, BLEND, (2024, 11), fbs=FBS)
        # average FBS team vs itself on a neutral field = 0: the FBS-mean rating is 0
        assert p["rating"].mean() == pytest.approx(0.0, abs=1e-9)
        # rating = expected_margin (home = team) minus the home edge it adds
        mean_elo = p["elo"].mean()
        for r in p.itertuples():
            em = expected_margin(r.elo, mean_elo, r.srs, 0.0, r.games,
                                 BLEND.srs_min_games, cfg, BLEND)
            assert r.rating == pytest.approx(em - hfa / 25.0, abs=1e-9)
    # pure-SRS side: hfa cannot move a rating at all
    a = cfb_power(sched, EloConfig(hfa_elo=0.0), BlendConfig(w_sos=1.0, srs_min_games=3), (2024, 11), fbs=FBS)
    b = cfb_power(sched, EloConfig(hfa_elo=90.0), BlendConfig(w_sos=1.0, srs_min_games=3), (2024, 11), fbs=FBS)
    assert np.allclose(a.sort_values("team")["rating"], b.sort_values("team")["rating"])


def test_cfb_preseason_uses_carried_over_elo_only():
    sched = _cfb_schedule()
    p = cfb_power(sched, ELO, BLEND, (2025, 1), fbs=FBS).set_index("team")
    end = cfb_power(sched, ELO, BLEND, (2024, 99), fbs=FBS).set_index("team")
    assert (p["games"] == 0).all() and p["srs"].isna().all()
    exp = ELO.base + ELO.carryover * (end["elo"] - ELO.base)
    assert np.allclose(p["elo"], exp.reindex(p.index))
    assert p.loc["T00", "rating"] == pytest.approx(
        (p.loc["T00", "elo"] - p["elo"].mean()) / 25.0)


def test_cfb_fcs_pseudo_team_flagged_and_excluded_from_average():
    sched = _cfb_schedule(fcs_weeks=(2, 5))
    p = cfb_power(sched, ELO, BLEND, (2024, 11), fbs=FBS).set_index("team")
    assert not p.loc["FCS", "is_fbs"] and p.drop("FCS")["is_fbs"].all()
    assert p.drop("FCS")["rating"].mean() == pytest.approx(0.0, abs=1e-9)
    assert p.loc["FCS", "rating"] < p.drop("FCS")["rating"].min()
    r = rankings(p.reset_index(), None, None, None)
    assert "FCS" not in set(r["team"]) and sorted(r["rank"]) == list(range(1, N_CFB + 1))


def test_cfb_power_leak_free():
    sched = _cfb_schedule(noise=10.0)
    base = cfb_power(sched, ELO, BLEND, (2024, 6), fbs=FBS)
    later = sched.copy()
    fut = later["season"].eq(2024) & later["week"].ge(6)
    later.loc[fut, "home_score"] = later.loc[fut, "home_score"] + 40      # rewrite the future
    extra = sched[sched["season"] == 2024].assign(season=2025)
    later = pd.concat([later, extra], ignore_index=True)
    pd.testing.assert_frame_equal(base, cfb_power(later, ELO, BLEND, (2024, 6), fbs=FBS))
    # ...and a past change does move it (the test has teeth)
    past = sched.copy()
    m = past["season"].eq(2024) & past["week"].eq(5) & past["home_team"].eq("T03")
    past.loc[m, "home_score"] += 30
    assert not base.equals(cfb_power(past, ELO, BLEND, (2024, 6), fbs=FBS))


# ---------------------------------------------------------------- NFL
def _ratings_frame(teams, **over):
    rows = []
    for t in teams:
        r = dict(season=2024, week=10, team=t, games=9, weight=0.75,
                 off_pass_epa=0.0, off_run_epa=0.0, def_pass_epa=0.0, def_run_epa=0.0,
                 off_pass_plays=35.0, off_run_plays=25.0,
                 def_pass_plays=35.0, def_run_plays=25.0)
        r.update(over.get(t, {}))
        rows.append(r)
    return pd.DataFrame(rows)


def test_nfl_power_formula_plus_ten():
    teams = list("ABCD")
    ur = _ratings_frame(teams, A=dict(off_pass_epa=0.1, off_run_epa=0.06,
                                      def_pass_epa=-0.1, def_run_epa=-0.06,
                                      def_pass_plays=50.0))   # own def plays are NOT used
    p = nfl_power(ur).set_index("team")
    # offense: 0.1*35 + 0.06*25 = 5; defense: -( -0.1*35 + -0.06*25 ) = +5
    assert p.loc["A", "rating"] == pytest.approx(10.0)
    assert p.loc["A", "off_rating"] == pytest.approx(5.0)
    assert p.loc["A", "def_rating"] == pytest.approx(5.0)
    assert p.drop("A")["rating"].abs().max() == 0.0
    r = rankings(p.reset_index(), None, ur, None).set_index("team")
    assert r.loc["A", "rank"] == 1


def test_nfl_power_team_plays_scale_offense_league_average_scales_defense():
    ur = _ratings_frame(list("AB"), A=dict(off_pass_epa=0.1, off_pass_plays=45.0,
                                           def_run_epa=0.1),
                        B=dict(off_run_plays=35.0))
    p = nfl_power(ur).set_index("team")
    lg_run = (25.0 + 35.0) / 2
    assert p.loc["A", "rating"] == pytest.approx(0.1 * 45 - 0.1 * lg_run)


def test_nfl_power_league_plays_from_unit_games_window():
    ug = league([2024], n_weeks=6, noise=0.0)
    ug.loc[ug["week"] >= 3, "run_plays"] = 99.0          # outside the window: ignored
    ug.loc[(ug["week"] < 3) & (ug["team"] == "B"), "run_plays"] = 41.0
    ur = _ratings_frame(list("AB"), A=dict(def_run_epa=0.1)).assign(week=3)
    lg_run = ug[(ug["season"] == 2024) & (ug["week"] < 3)]["run_plays"].mean()
    p = nfl_power(ur, ug).set_index("team")
    assert p.loc["A", "rating"] == pytest.approx(-0.1 * lg_run)


def test_nfl_synthetic_league_plus_ten_end_to_end():
    # A is +5 on offense (3 pass + 2 run pts/game) and +5 on defense vs every
    # other team; synth scales profile by 0.5*SD(epa)=0.075 EPA/play.
    x, y = (3 / 35) / 0.075, (2 / 25) / 0.075
    prof = {"A": {"pass_o": x, "run_o": y, "pass_d": -x, "run_d": -y}}
    ug = league([2023, 2024], n_weeks=14, profile=prof, noise=0.0)
    ur = unit_ratings_asof(ug, 2024, 10)
    p = nfl_power(ur, ug)
    r = rankings(p, None, ur, None).set_index("team")
    assert r.loc["A", "rank"] == 1
    assert r.loc["A", "rating"] == pytest.approx(10 * 7 / 8, abs=0.5)
    assert r.loc["A", ["pass_off_rank", "run_off_rank", "pass_def_rank", "run_def_rank"]].tolist() == [1, 1, 1, 1]


def test_nfl_power_leak_free():
    ug = league([2023, 2024], n_weeks=14, noise=0.5)
    p = nfl_power(unit_ratings_asof(ug, 2024, 8), ug)
    ug2 = ug.copy()
    fut = ug2["season"].eq(2024) & ug2["week"].ge(8)
    ug2.loc[fut, ["pass_epa", "run_epa"]] += 1.0
    ug2.loc[fut, "run_plays"] = 80.0
    pd.testing.assert_frame_equal(p, nfl_power(unit_ratings_asof(ug2, 2024, 8), ug2))


# ---------------------------------------------------------------- rankings
def _power(ratings: dict, season=2024, week=5):
    return pd.DataFrame({"season": season, "week": week, "team": list(ratings),
                         "rating": list(ratings.values())})


def _log_row(team, opp, week, margin, line, season=2024):
    played = margin is not None
    su = None if not played else ("W" if margin > 0 else "L" if margin < 0 else "T")
    ats = (None if not played or line is None else
           "W" if margin > line else "L" if margin < line else "P")
    return dict(sport="nfl", season=season, week=week, team=team, opponent=opp,
                margin=np.nan if margin is None else float(margin),
                team_line=np.nan if line is None else float(line), su=su, ats=ats,
                team_is_fbs=True, opp_is_fbs=True)


def _log():
    rows = [
        _log_row("A", "B", 1, 7, 3),     # SU W, ATS W
        _log_row("A", "C", 2, -3, -6),   # SU L, ATS W
        _log_row("A", "D", 3, 3, 3),     # SU W, ATS P
        _log_row("A", "C", 4, 10, None),  # SU W, no line
        _log_row("A", "D", 5, -20, 1),   # week 5 = the ranking week: excluded
        _log_row("A", "B", 6, None, 2),  # upcoming
        _log_row("A", "D", 17, 1, 1, season=2023),  # last season: excluded
        _log_row("B", "A", 1, -7, -3),
    ]
    return pd.DataFrame(rows)


def test_rankings_rank_move_and_records():
    cur = _power({"A": 6.0, "B": 2.0, "C": -1.0, "D": -7.0})
    prev = _power({"A": 1.0, "B": 3.0, "C": -1.5, "D": 0.5}, week=4)   # B, A, D, C
    r = rankings(cur, prev, None, _log()).set_index("team")
    assert r["rank"].to_dict() == {"A": 1, "B": 2, "C": 3, "D": 4}
    assert r["prev_rank"].to_dict() == {"A": 2, "B": 1, "C": 4, "D": 3}
    assert r["move"].to_dict() == {"A": 1, "B": -1, "C": 1, "D": -1}   # + = moved up
    assert r.loc["A", "su"] == "3-1" and r.loc["A", "ats"] == "2-0-1"
    assert r.loc["C", "su"] == "0-0" and r.loc["C", "ats"] == "0-0-0"
    assert r.loc["A", "rating"] == 6.0
    assert list(r.index) == ["A", "B", "C", "D"]


def test_rankings_without_previous_week_or_new_team():
    cur = _power({"A": 1.0, "B": 2.0})
    prev = _power({"B": 0.0}, week=4)
    r = rankings(cur, prev, None, None).set_index("team")
    assert pd.isna(r.loc["A", "prev_rank"]) and pd.isna(r.loc["A", "move"])
    assert r.loc["B", "prev_rank"] == 1 and r.loc["B", "move"] == 0
    r0 = rankings(cur, None, None, None)
    assert r0["prev_rank"].isna().all() and r0["move"].isna().all()
    assert r0["sos"].isna().all() and r0["su"].isna().all()


def test_rankings_sos_uses_current_ratings_of_opponents_played():
    cur = _power({"A": 6.0, "B": 2.0, "C": -1.0, "D": -7.0})
    log = _log()
    log = pd.concat([log, pd.DataFrame([_log_row("A", "FCS", 4, 30, None)])])
    r = rankings(cur, None, None, log).set_index("team")
    # A played B (2.0), C (-1.0), D (-7.0), C (-1.0) before week 5; FCS unrated -> skipped
    assert r.loc["A", "sos"] == pytest.approx((2.0 - 1.0 - 7.0 - 1.0) / 4)
    assert r.loc["B", "sos"] == pytest.approx(6.0)
    assert pd.isna(r.loc["C", "sos"])


def test_rankings_sos_counts_rated_non_fbs_opponent():
    cur = pd.concat([_power({"A": 3.0, "B": -3.0}).assign(is_fbs=True),
                     _power({"FCS": -30.0}).assign(is_fbs=False)])
    log = pd.DataFrame([_log_row("A", "B", 1, 3, 1), _log_row("A", "FCS", 2, 40, None)])
    r = rankings(cur, None, None, log).set_index("team")
    assert "FCS" not in r.index
    assert r.loc["A", "sos"] == pytest.approx((-3.0 - 30.0) / 2)


def test_rankings_unit_ranks_defense_best_is_lowest_allowed():
    cur = _power({"A": 1.0, "B": 0.0, "C": -1.0})
    ur = _ratings_frame(list("ABCX"),
                        A=dict(off_pass_epa=0.2, off_run_epa=-0.1, def_pass_epa=0.1, def_run_epa=-0.2),
                        B=dict(off_pass_epa=0.1, off_run_epa=0.1, def_pass_epa=-0.1, def_run_epa=0.0),
                        C=dict(off_pass_epa=0.0, off_run_epa=0.0, def_pass_epa=0.0, def_run_epa=0.1),
                        X=dict(off_pass_epa=9.0))    # not in the power table: ignored
    r = rankings(cur, None, ur, None).set_index("team")
    assert r["pass_off_rank"].to_dict() == {"A": 1, "B": 2, "C": 3}
    assert r["run_off_rank"].to_dict() == {"A": 3, "B": 1, "C": 2}
    assert r["pass_def_rank"].to_dict() == {"A": 3, "B": 1, "C": 2}
    assert r["run_def_rank"].to_dict() == {"A": 1, "B": 2, "C": 3}
    u = r.loc["A", "units"]
    assert u["pass_off"] == {"rank": 1, "epa": pytest.approx(0.2)}
    assert u["run_def"] == {"rank": 1, "epa": pytest.approx(-0.2)}


def test_rankings_without_unit_ratings_have_no_unit_ranks():
    r = rankings(_power({"A": 1.0, "B": 0.0}), None, None, None)
    assert r["pass_off_rank"].isna().all() and r["units"].isna().all()


def test_rankings_leak_free_log():
    cur = _power({"A": 6.0, "B": 2.0, "C": -1.0, "D": -7.0})
    log = _log()
    base = rankings(cur, None, None, log)
    later = log.copy()
    later.loc[later["week"] >= 5, ["su", "ats"]] = "L"
    later = pd.concat([later, pd.DataFrame([_log_row("A", "D", 7, 50, 1)])])
    pd.testing.assert_frame_equal(base, rankings(cur, None, None, later))


def test_cfb_default_fbs_membership_from_asset():
    ids = sorted(load_fbs_ids())[:4]
    rows = []
    for w, (h, a) in enumerate([(ids[0], ids[1]), (ids[2], ids[3]), (ids[0], "FCS"),
                                (ids[1], ids[2]), (ids[3], ids[0])], start=1):
        rows.append(dict(season=2024, week=w, home_team=h, away_team=a, home_score=28.0,
                         away_score=20.0, game_type="REG"))
    p = cfb_power(pd.DataFrame(rows), ELO, BLEND, (2024, 9)).set_index("team")
    assert p.loc[ids, "is_fbs"].all() and not p.loc["FCS", "is_fbs"]
    assert np.isfinite(p["rating"]).all()


# ------------------------------------------------------- NFL market scale (P3)
def _raw_diffs(ug, season, week):
    raw = nfl_power(unit_ratings_asof(ug, season, week), ug)
    return raw.set_index("team")["raw_rating"] if len(raw) else None


def _market_schedule(ug, seasons, slope=0.5, hfa=2.0, n_weeks=14):
    """Each week every round-robin pair plays twice (home and away swapped) plus
    one neutral game; spread_line = slope * raw_diff (+ hfa when not neutral)."""
    rr = round_robin(TEAMS)
    rows = []
    for s in seasons:
        for w in range(1, n_weeks + 1):
            r = _raw_diffs(ug, s, w)
            if r is None:
                continue
            pairs = rr[(w - 1) % len(rr)]
            for h, a in pairs:
                for hh, aa in ((h, a), (a, h)):
                    rows.append(dict(season=s, week=w, game_type="REG", home_team=hh,
                                     away_team=aa, location="Home",
                                     spread_line=slope * (r[hh] - r[aa]) + hfa))
            h, a = pairs[0]
            rows.append(dict(season=s, week=w, game_type="REG", home_team=h, away_team=a,
                             location="Neutral", spread_line=slope * (r[h] - r[a])))
            rows.append(dict(season=s, week=w, game_type="WC", home_team=h, away_team=a,
                             location="Home", spread_line=99.0))       # POST: ignored
    return pd.DataFrame(rows)


def test_nfl_market_scale_recovers_slope_and_hfa():
    ug = league([2020, 2021, 2022, 2023], n_weeks=10, noise=0.6, seed=11)
    sched = _market_schedule(ug, [2020, 2021, 2022], n_weeks=10)
    sc = nfl_market_scale(ug, sched, 2023)
    assert sc["seasons"] == [2020, 2021, 2022] and sc["n"] > 100
    assert sc["slope"] == pytest.approx(0.5, abs=1e-9)
    assert sc["hfa"] == pytest.approx(2.0, abs=1e-9)


def test_nfl_market_scale_never_reads_season_s():
    ug = league([2020, 2021, 2022, 2023], n_weeks=10, noise=0.6, seed=11)
    sched = _market_schedule(ug, [2020, 2021, 2022], n_weeks=10)
    base = nfl_market_scale(ug, sched, 2023)
    ug2 = ug.copy()
    ug2.loc[ug2["season"] >= 2023, ["pass_epa", "run_epa"]] += 3.0
    extra = sched[sched["season"] == 2022].assign(season=2023, spread_line=-50.0)
    assert nfl_market_scale(ug2, pd.concat([sched, extra]), 2023) == base
    # ...but it does read S-1 (the test has teeth)
    moved = sched.copy()
    moved.loc[moved["season"] == 2022, "spread_line"] *= 2
    assert nfl_market_scale(ug, moved, 2023)["slope"] != base["slope"]


def test_nfl_power_applies_scale_and_stores_it():
    ur = _ratings_frame(list("ABCD"), A=dict(off_pass_epa=0.1, off_run_epa=0.06,
                                             def_pass_epa=-0.1, def_run_epa=-0.06))
    sc = {"season": 2024, "slope": 0.5, "hfa": 1.8, "n": 800, "seasons": [2021, 2022, 2023]}
    p = nfl_power(ur, scale=sc)
    a = p.set_index("team").loc["A"]
    assert a["raw_rating"] == pytest.approx(10.0) and a["rating"] == pytest.approx(5.0)
    assert a["off_rating"] == pytest.approx(2.5) and a["def_rating"] == pytest.approx(2.5)
    assert (p["scale"] == 0.5).all() and (p["hfa"] == 1.8).all()
    assert p.attrs["market_scale"]["slope"] == 0.5 and p.attrs["market_scale"]["hfa"] == 1.8
    raw = nfl_power(ur)
    assert (raw["scale"] == 1.0).all() and raw["hfa"].isna().all()


def test_rankings_records_and_sos_are_regular_season_only():
    cur = _power({"A": 6.0, "B": 2.0, "C": -1.0, "D": -7.0}, week=20)
    rows = [_log_row("A", "B", 1, 7, 3), _log_row("A", "D", 19, -10, 3)]
    log = pd.DataFrame(rows).assign(is_post=[False, True])
    r = rankings(cur, None, None, log).set_index("team")
    assert r.loc["A", "su"] == "1-0" and r.loc["A", "ats"] == "1-0-0"
    assert r.loc["A", "sos"] == pytest.approx(2.0)
    assert r.loc["D", "su"] == "0-0"


# ------------------------------------------------- season-weighted CFB power
from sportsmodel.context.power import cfb_power_current  # noqa: E402


def test_cfb_power_current_weights_this_season_by_games_over_games_plus_one():
    sched = _cfb_schedule(seasons=(2023, 2024), n_weeks=12, strong="T00", edge=10.0)
    base = cfb_power(sched, ELO, BLEND, (2024, 5), fbs=FBS).set_index("team")
    priors = {t: 1500.0 for t in CFB_TEAMS}
    priors["T05"] = 1500.0 + 25 * 16          # preseason says T05 is +16 better
    p = cfb_power_current(sched, ELO, BLEND, (2024, 5), priors=priors, fbs=FBS).set_index("team")
    g = int(p.loc["T00", "games"])
    assert g == 4
    w = g / (g + 1)
    assert p.loc["T00", "weight"] == pytest.approx(w)
    fbs_srs = base.loc[CFB_TEAMS, "srs"]
    pre_mean = np.mean(list(priors.values()))
    for t in ("T00", "T05", "T09"):
        cur = base.loc[t, "srs"] - fbs_srs.mean()
        pre = (priors[t] - pre_mean) / 25.0
        assert p.loc[t, "rating"] == pytest.approx(w * cur + (1 - w) * pre)
        assert p.loc[t, "prior"] == pytest.approx(pre)
    assert p.index[0] == "T00"


def test_cfb_power_current_falls_back_to_elo_without_a_prior_and_uses_prior_before_games():
    sched = _cfb_schedule(seasons=(2023, 2024), n_weeks=12, strong="T00", edge=10.0)
    base = cfb_power(sched, ELO, BLEND, (2024, 1), fbs=FBS).set_index("team")
    p = cfb_power_current(sched, ELO, BLEND, (2024, 1), priors={}, fbs=FBS).set_index("team")
    avg_elo = base.loc[CFB_TEAMS, "elo"].mean()
    for t in ("T00", "T07"):
        assert p.loc[t, "weight"] == 0.0
        assert p.loc[t, "rating"] == pytest.approx((base.loc[t, "elo"] - avg_elo) / 25.0)


def test_cfb_power_current_leak_free():
    sched = _cfb_schedule(seasons=(2023, 2024), n_weeks=12, noise=6.0)
    a = cfb_power_current(sched, ELO, BLEND, (2024, 6), priors={}, fbs=FBS)
    later = sched.copy()
    later.loc[(later["season"] == 2024) & (later["week"] >= 6), "home_score"] += 40
    b = cfb_power_current(later, ELO, BLEND, (2024, 6), priors={}, fbs=FBS)
    pd.testing.assert_frame_equal(a, b)


def test_nfl_market_scale_accepts_blend_k():
    ug = league([2020, 2021, 2022, 2023], n_weeks=10, noise=0.6, seed=11)
    sched = _market_schedule(ug, [2020, 2021, 2022], n_weeks=10)
    a = nfl_market_scale(ug, sched, 2023)
    b = nfl_market_scale(ug, sched, 2023, blend_k=1.0)
    assert a["n"] == b["n"] and b["slope"] is not None and b["slope"] != a["slope"]
