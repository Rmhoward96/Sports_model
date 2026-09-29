"""Tests for sportsmodel.model.props_ml.v2_checks (the v2-vs-v1 gate sub-checks; pure)."""
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.model import props_eval
from sportsmodel.model.props_ml import v2_checks as vc


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch):
    real = props_eval.cluster_bootstrap
    monkeypatch.setattr(props_eval, "cluster_bootstrap",
                        lambda df, stat, n_boot=1000, seed=0: real(df, stat, n_boot=60, seed=seed))


# ---- qb_change_mask ------------------------------------------------------------------------

def _qb_feats():
    rows = []
    for w in (3, 4, 5):
        for team in ("A", "B", "C"):
            for k in range(2):
                rows.append({"player_id": f"{team}{k}", "season": 2024, "week": w, "team": team,
                             "qb_changed": 0, "qb_ratio_ypa": 1.02})
    f = pd.DataFrame(rows)
    f.loc[(f.team == "A") & (f.week == 3), "qb_changed"] = 1            # new starter
    f.loc[(f.team == "B") & (f.week == 5), "qb_ratio_ypa"] = 0.90       # weaker QB profile
    f.loc[(f.team == "C") & (f.week == 4), "qb_ratio_ypa"] = 1.05       # boundary: not a change
    f.loc[(f.team == "C") & (f.week == 5), "qb_ratio_ypa"] = np.nan     # unknown: not a change
    return f


def test_qb_change_mask_picks_exactly_the_planted_team_weeks():
    f = _qb_feats()
    recs = pd.DataFrame([{"season": 2024, "week": w, "player_id": f"{t}{k}", "market": m}
                         for w in (3, 4, 5) for t in ("A", "B", "C") for k in range(2)
                         for m in ("pass_yds", "rec_yds")])
    recs.index = recs.index + 100  # the mask follows the records' index
    recs = pd.concat([recs, pd.DataFrame([{"season": 2024, "week": 3, "player_id": "ZZ",
                                           "market": "pass_yds"}], index=[999])])
    mask = vc.qb_change_mask(recs, f)
    assert list(mask.index) == list(recs.index) and mask.dtype == bool
    got = {(int(r.week), r.player_id[0]) for r, m in zip(recs.itertuples(), mask) if m}
    assert got == {(3, "A"), (5, "B")}
    # every record of a planted team-week is flagged (the flag is team-level)
    assert mask[(recs.week == 3) & recs.player_id.str.startswith("A")].all()
    assert not mask.loc[999]  # no feature row -> not a QB-change record


def test_qb_change_mask_uses_the_team_week_when_only_one_row_carries_the_flag():
    f = _qb_feats()
    f.loc[(f.team == "A") & (f.week == 3) & (f.player_id == "A1"), "qb_changed"] = 0
    recs = pd.DataFrame([{"season": 2024, "week": 3, "player_id": "A1", "market": "rec_yds"}])
    assert vc.qb_change_mask(recs, f).tolist() == [True]


# ---- player_baseline -----------------------------------------------------------------------

def test_player_baseline_previous_n_played_games_nan_below_three():
    rows = [{"player_id": "p", "season": 2023, "week": w, "is_stub": False,
             "y_rush_yds": float(w), "y_rec_yds": 10.0} for w in range(1, 11)]
    rows += [{"player_id": "p", "season": 2024, "week": 1, "is_stub": True,     # stub: not played
              "y_rush_yds": 0.0, "y_rec_yds": 0.0},
             {"player_id": "p", "season": 2024, "week": 2, "is_stub": False,    # the record's own game
              "y_rush_yds": 999.0, "y_rec_yds": 999.0},
             {"player_id": "q", "season": 2024, "week": 1, "is_stub": False,
              "y_rush_yds": 50.0, "y_rec_yds": np.nan},
             {"player_id": "q", "season": 2024, "week": 2, "is_stub": False,
              "y_rush_yds": 70.0, "y_rec_yds": np.nan}]
    f = pd.DataFrame(rows)
    recs = pd.DataFrame([
        {"season": 2024, "week": 2, "player_id": "p", "market": "rush_yds"},  # 2023 wks 3..10
        {"season": 2023, "week": 5, "player_id": "p", "market": "rush_yds"},  # wks 1..4
        {"season": 2023, "week": 3, "player_id": "p", "market": "rush_yds"},  # 2 games -> NaN
        {"season": 2024, "week": 3, "player_id": "q", "market": "rush_yds"},  # 2 games -> NaN
        {"season": 2024, "week": 2, "player_id": "p", "market": "rec_yds"},
        {"season": 2024, "week": 2, "player_id": "nobody", "market": "rec_yds"},
    ], index=[7, 3, 5, 1, 2, 9])
    got = vc.player_baseline(recs, f, n=8)
    assert list(got.index) == [7, 3, 5, 1, 2, 9]
    assert got.loc[7] == pytest.approx(np.mean(range(3, 11)))
    assert got.loc[3] == pytest.approx(2.5)
    assert math.isnan(got.loc[5]) and math.isnan(got.loc[1]) and math.isnan(got.loc[9])
    assert got.loc[2] == pytest.approx(10.0)
    assert vc.player_baseline(recs.loc[[3]], f, n=3).loc[3] == pytest.approx(3.0)  # wks 2..4


# ---- matchup_response ----------------------------------------------------------------------

GROUPS = {"RB": ("rush_yds", "y_rush_yds", 100.0), "QB": ("pass_yds", "y_pass_yds", 250.0),
          "WR": ("rec_yds", "y_rec_yds", 80.0), "TE": ("rec_yds", "y_rec_yds", 50.0)}
MX = [-2.0, -1.0, 0.0, 1.0, 2.0]  # team Ti faces defense Di -> quintile i + 1
# ACTUAL deviation from baseline per quintile. Top (strong run D / weak pass D):
# RB 20 % under, passing game over; bottom: the mirror; linear in between.
TOP = {"RB": -0.20, "QB": 0.10, "WR": 0.12, "TE": 0.08}
ACT = {g: [-v, -v / 2, 0.0, v / 2, v] for g, v in TOP.items()}


def _mx_feats():
    rows = []
    for i in range(5):
        for pos, (market, label, base) in GROUPS.items():
            pid = f"T{i}{pos}"
            for w in range(1, 9):
                rows.append({"player_id": pid, "season": 2024, "week": w, "team": f"T{i}",
                             "opponent": "X", "position": pos, "is_stub": False,
                             "mx_pass_minus_rush": np.nan, "y_rush_yds": np.nan,
                             "y_pass_yds": np.nan, "y_rec_yds": np.nan, label: base})
            rows.append({"player_id": pid, "season": 2024, "week": 9, "team": f"T{i}",
                         "opponent": f"D{i}", "position": pos, "is_stub": False,
                         "mx_pass_minus_rush": MX[i], "y_rush_yds": np.nan, "y_pass_yds": np.nan,
                         "y_rec_yds": np.nan})
    return pd.DataFrame(rows)


def _served(dev):
    """Served records at week 9: mean = baseline x (1 + dev[pos][quintile])."""
    rows = []
    for i in range(5):
        for pos, (market, label, base) in GROUPS.items():
            rows.append({"season": 2024, "week": 9, "home": f"T{i}", "player_id": f"T{i}{pos}",
                         "market": market, "mean": base * (1 + dev[pos][i]), "rps": 1.0,
                         "pit": 0.5, "actual": base * (1 + ACT[pos][i])})
    # a market outside the four groups is ignored
    rows.append({"season": 2024, "week": 9, "home": "T0", "player_id": "T0RB",
                 "market": "receptions", "mean": 3.0, "rps": 1.0, "pit": 0.5, "actual": 2.0})
    return pd.DataFrame(rows)


V1 = {g: [0.0] * 5 for g in GROUPS}
V2 = {g: [0.75 * a for a in ACT[g]] for g in GROUPS}  # top RB -15 %


def test_matchup_response_both_directions_pass_and_v2_tracks_actuals_better():
    got = vc.matchup_response(_served(V1), _served(V2), _mx_feats())
    assert got["pass_top"] is True and got["pass_bottom"] is True
    assert got["corr_v2"] == pytest.approx(1.0)
    assert got["corr_v1"] == 0.0            # constant v1 deviations: no correlation
    assert got["corr_v2"] > got["corr_v1"] and got["pass"] is True
    cells = {(c["group"], c["quintile"]): c for c in got["table"]}
    assert len(cells) == 20
    top_rb = cells[("RB", 5)]
    assert top_rb["n"] == 1
    assert top_rb["act_dev"] == pytest.approx(-0.20)
    assert top_rb["pred_dev_v2"] == pytest.approx(-0.15)
    assert top_rb["pred_dev_v1"] == pytest.approx(0.0)
    assert cells[("QB", 1)]["act_dev"] == pytest.approx(-0.10)
    assert cells[("TE", 5)]["pred_dev_v2"] == pytest.approx(0.06)
    assert got["n_records"] == 20


def test_matchup_response_flipped_v2_rb_sign_fails():
    flipped = {**V2, "RB": [-x for x in V2["RB"]]}
    got = vc.matchup_response(_served(V1), _served(flipped), _mx_feats())
    assert got["pass_top"] is False and got["pass_bottom"] is False
    assert got["pass"] is False


def test_matchup_response_bottom_only_failure_and_sign_must_match_actual():
    # passing game right in the bottom quintile's direction but wrong for QB
    bad = {**V2, "QB": V2["QB"][:4] + [-0.05]}   # top QB predicted down, actual up
    got = vc.matchup_response(_served(V1), _served(bad), _mx_feats())
    assert got["pass_top"] is False and got["pass_bottom"] is True and got["pass"] is False


def test_matchup_response_needs_v2_to_beat_v1_correlation():
    got = vc.matchup_response(_served(V2), _served(V2), _mx_feats())
    assert got["pass_top"] and got["pass_bottom"]
    assert got["corr_v2"] == pytest.approx(got["corr_v1"]) and got["pass"] is False


# ---- qb_change_check -----------------------------------------------------------------------

def _paired(n_weeks=12, improve=0.1, seasons=(2024, 2025)):
    rows = []
    for s in seasons:
        for w in range(1, n_weeks + 1):
            for k in range(3):
                b = 10.0 + k + (w % 3)
                for m in ("pass_yds", "rec_yds"):
                    rows.append({"season": s, "week": w, "player_id": f"p{k}", "market": m,
                                 "home": "H", "rps_b": b, "rps_c": b * (1 - improve),
                                 "pit_b": ((w * 7 + k) % 10) / 10 + 0.05,
                                 "pit_c": ((w * 7 + k) % 10) / 10 + 0.05,
                                 "cluster": f"{s}-{w}-H"})
    return pd.DataFrame(rows)


def test_qb_change_check_pass_yds_only_and_ci_must_exclude_zero():
    p = _paired()
    mask = pd.Series(True, index=p.index)
    got = vc.qb_change_check(p, mask)
    assert got["n"] == 2 * 12 * 3 and got["pass"] is True
    assert got["diff"] < 0 and got["hi"] < 0
    assert got["n_clusters"] == 24   # season-week clusters
    worse = p.assign(rps_c=p["rps_b"] * 1.05)
    assert vc.qb_change_check(worse, mask)["pass"] is False
    none = vc.qb_change_check(p, pd.Series(False, index=p.index))
    assert none["n"] == 0 and none["pass"] is False


# ---- paired_served -------------------------------------------------------------------------

def test_paired_served_joins_v1_v2_and_requires_identical_keys():
    v1 = _served(V1)
    v2 = _served(V2).assign(rps=0.5, pit=0.4)
    p = vc.paired_served(v1, v2)
    assert len(p) == len(v1)
    assert set(p.columns) >= {"season", "week", "home", "player_id", "market", "rps_b", "rps_c",
                              "pit_b", "pit_c", "cluster"}
    assert (p["rps_b"] == 1.0).all() and (p["rps_c"] == 0.5).all()
    assert p["cluster"].iloc[0].startswith("2024-9-")
    with pytest.raises(ValueError, match="key"):
        vc.paired_served(v1, v2.iloc[1:])
