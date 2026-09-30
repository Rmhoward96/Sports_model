import numpy as np
import pandas as pd
import pytest

from sportsmodel.context.units import (
    CFB_DEFAULT_PASS_RATE, METRICS, blend_weight, cfb_unit_games, nfl_unit_games,
    unit_ratings_asof, window_ratings,
)
from sportsmodel.nfl.efficiency import adjusted_efficiency
from tests.context.synth import league


# --------------------------------------------------------------------- NFL pbp
def _play(pos, de, play_type, yds, epa, success, *, scramble=0, sack=0,
          season_type="REG", week=1, two_pt=0):
    return dict(season=2024, week=week, season_type=season_type, game_id=f"2024_0{week}_KC_BAL",
                posteam=pos, defteam=de, play_type=play_type, yards_gained=yds, epa=epa,
                success=success, qb_scramble=scramble, sack=sack,
                two_point_attempt=two_pt, **{"pass": int(play_type == "pass" or scramble),
                                             "rush": int(play_type == "run" and not scramble)})


def _pbp():
    return pd.DataFrame([
        _play("KC", "BAL", "pass", 25, 1.0, 1),              # explosive dropback
        _play("KC", "BAL", "pass", -7, -1.5, 0, sack=1),     # sack = dropback
        _play("KC", "BAL", "run", 12, 0.8, 1, scramble=1),   # scramble = dropback, not explosive (<20)
        _play("KC", "BAL", "run", 10, 0.6, 1),               # explosive run (>=10)
        _play("KC", "BAL", "run", 3, -0.2, 0),
        _play("KC", "BAL", "punt", 0, 0.1, 0),               # not a unit play
        _play("KC", "BAL", "pass", 5, np.nan, 0),            # no epa -> dropped
        _play("KC", "BAL", "pass", 30, 3.0, 1, two_pt=1),    # 2pt try -> dropped
        _play("KC", "BAL", "pass", 30, 3.0, 1, season_type="PRE"),
        _play("BAL", "KC", "pass", 8, 0.3, 1),
        _play("BAL", "KC", "run", 2, -0.4, 0),
    ])


def test_nfl_unit_games_metrics():
    ug = nfl_unit_games(_pbp())
    kc = ug[ug["team"] == "KC"].iloc[0]
    assert kc["opponent"] == "BAL" and kc["game_id"] == "2024_01_KC_BAL"
    assert kc["pass_plays"] == 3 and kc["run_plays"] == 2
    assert kc["pass_rate"] == pytest.approx(0.6)
    assert kc["pass_epa"] == pytest.approx((1.0 - 1.5 + 0.8) / 3)
    assert kc["pass_success"] == pytest.approx(2 / 3)
    assert kc["pass_explosive"] == pytest.approx(1 / 3)
    assert kc["run_epa"] == pytest.approx(0.2)
    assert kc["run_success"] == pytest.approx(0.5)
    assert kc["run_explosive"] == pytest.approx(0.5)
    bal = ug[ug["team"] == "BAL"].iloc[0]
    assert bal["pass_rate"] == pytest.approx(0.5) and bal["opponent"] == "KC"


def test_nfl_unit_games_keeps_postseason():
    pbp = pd.concat([_pbp(), pd.DataFrame([
        _play("KC", "BAL", "pass", 5, 0.1, 1, season_type="POST", week=19),
        _play("BAL", "KC", "run", 5, 0.1, 1, season_type="POST", week=19)])])
    ug = nfl_unit_games(pbp)
    assert sorted(ug["week"].unique()) == [1, 19]


# ------------------------------------------------------------------ CFB parquet
_ADV_COLS = ["season", "week", "game_id", "team", "opponent"] + [
    f"{u}_{s}" for u in ("off", "def") for s in (
        "plays", "ppa", "success", "explosiveness", "pass_ppa", "pass_success",
        "pass_explosiveness", "rush_ppa", "rush_success", "rush_explosiveness")
] + ["off_pass_plays", "off_rush_plays"]


def _adv_row(season, week, gid, team, opp, pe=0.2, re=0.1, pxp=1.5, rxp=1.0,
             pp=np.nan, rp=np.nan, d_pe=0.1, d_pxp=1.4):
    r = dict(season=season, week=week, game_id=gid, team=team, opponent=opp)
    for u in ("off", "def"):
        r.update({f"{u}_plays": 70.0, f"{u}_ppa": 0.15, f"{u}_success": 0.42,
                  f"{u}_explosiveness": 1.2, f"{u}_pass_success": 0.45,
                  f"{u}_rush_success": 0.40, f"{u}_rush_ppa": 0.05,
                  f"{u}_rush_explosiveness": 1.0})
    r.update(off_pass_ppa=pe, off_rush_ppa=re, off_pass_explosiveness=pxp,
             off_rush_explosiveness=rxp, off_pass_plays=pp, off_rush_plays=rp,
             def_pass_ppa=d_pe, def_pass_explosiveness=d_pxp)
    return r


def _adv(rows):
    return pd.DataFrame(rows, columns=_ADV_COLS)


def test_cfb_unit_games_columns_and_pass_rate_fallback():
    adv = _adv([_adv_row(2024, 1, 1, "1", "2", pp=30, rp=20),
                _adv_row(2024, 1, 1, "2", "1")])
    ug = cfb_unit_games(adv)
    one = ug[ug["team"] == "1"].iloc[0]
    two = ug[ug["team"] == "2"].iloc[0]
    assert one["pass_epa"] == pytest.approx(0.2) and one["run_epa"] == pytest.approx(0.1)
    assert one["pass_success"] == pytest.approx(0.45)
    assert one["pass_rate"] == pytest.approx(0.6)
    assert two["pass_rate"] == CFB_DEFAULT_PASS_RATE  # no split play counts -> 0.5
    assert set(METRICS) <= set(ug.columns)


def test_cfb_mirror_row_from_defense_when_opponent_row_missing():
    adv = _adv([_adv_row(2024, 1, 1, "1", "2", d_pe=-0.3),
                _adv_row(2024, 1, 2, "3", "4"), _adv_row(2024, 1, 2, "4", "3")])
    ug = cfb_unit_games(adv)
    two = ug[(ug["team"] == "2") & (ug["game_id"] == 1)]
    assert len(two) == 1
    assert two.iloc[0]["opponent"] == "1"
    assert two.iloc[0]["pass_epa"] == pytest.approx(-0.3)
    assert len(ug) == 4


def test_cfb_explosiveness_z_scored_within_season_to_date():
    w1 = [_adv_row(2024, 1, 1, "1", "2", pxp=1.0), _adv_row(2024, 1, 1, "2", "1", pxp=2.0),
          _adv_row(2024, 1, 2, "3", "4", pxp=3.0), _adv_row(2024, 1, 2, "4", "3", pxp=4.0)]
    w2 = [_adv_row(2024, 2, 3, "1", "3", pxp=10.0), _adv_row(2024, 2, 3, "3", "1", pxp=20.0)]
    ug = cfb_unit_games(_adv(w1 + w2))
    wk1 = ug[ug["week"] == 1].set_index("team")["pass_explosive"]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    exp = (x - x.mean()) / x.std(ddof=1)
    assert wk1.loc[["1", "2", "3", "4"]].to_numpy() == pytest.approx(exp)
    # week-1 z-scores do not see week-2 values (leak-free standardization)
    ug_only1 = cfb_unit_games(_adv(w1))
    assert ug_only1.set_index("team")["pass_explosive"].loc[["1", "2", "3", "4"]].to_numpy() \
        == pytest.approx(exp)
    # a new season restarts the standardization
    ug2 = cfb_unit_games(_adv(w1 + [dict(r, season=2025) for r in w1]))
    a = ug2[ug2["season"] == 2025].set_index("team")["pass_explosive"]
    assert a.loc[["1", "2", "3", "4"]].to_numpy() == pytest.approx(exp)


# -------------------------------------------------------------------- ratings
def test_window_matches_efficiency_adjusted_efficiency():
    ug = league([2024], n_weeks=6)
    for m in ("pass_epa", "run_success"):
        mine = window_ratings(ug, 2024, 5)
        off = {(int(s), int(w), t): (v, o) for s, w, t, o, v in
               ug[["season", "week", "team", "opponent", m]].itertuples(index=False)}
        gd = {k: {"off": v, "def": off[(k[0], k[1], o)][0], "opp": o} for k, (v, o) in off.items()}
        ref = adjusted_efficiency(gd, 2024, 5)
        lo = np.mean([v["off_adj"] for v in ref.values()])
        ld = np.mean([v["def_adj"] for v in ref.values()])
        for t, v in ref.items():
            assert mine.loc[t, f"off_{m}"] == pytest.approx(v["off_adj"] - lo)
            assert mine.loc[t, f"def_{m}"] == pytest.approx(v["def_adj"] - ld)
        assert mine[f"off_{m}"].mean() == pytest.approx(0.0, abs=1e-12)


def test_opponent_adjustment_direction():
    # A's offense is average but faced only B's elite pass defense early -> adjusts up.
    ug = league([2024], n_weeks=7, profile={"B": {"pass_d": -4}}, noise=0.0)
    r = window_ratings(ug, 2024, 8)
    assert r.loc["B", "def_pass_epa"] < -0.1          # strong D allows less (negative)
    # every other defense is average -> all equal (B's strength drags the mean down)
    others = r.drop("B")["def_pass_epa"]
    assert others.max() - others.min() < 0.005
    assert r.loc["B", "def_pass_epa"] - others.mean() < -0.25
    assert r["off_pass_epa"].abs().max() < 0.02       # opponents of B adjusted back to avg


def test_blend_weight_values():
    assert blend_weight(0) == 0.0
    assert blend_weight(3) == pytest.approx(0.5)
    assert blend_weight(9) == pytest.approx(0.75)


@pytest.mark.parametrize("week,games,w", [(1, 0, 0.0), (4, 3, 0.5), (10, 9, 0.75)])
def test_early_season_blend(week, games, w):
    ug = league([2023, 2024], n_weeks=12, profile={"A": {"pass_o": 3}}, seed=3)
    prev = unit_ratings_asof(ug[ug["season"] == 2023], 2023, 99).set_index("team")
    both = unit_ratings_asof(ug, 2024, week).set_index("team")
    assert (both["games"] == games).all()
    assert both["weight"].to_numpy() == pytest.approx(np.full(8, w))
    cols = [f"{s}_{m}" for s in ("off", "def") for m in METRICS] + ["pass_rate"]
    if games == 0:
        exp = prev[cols]
    else:
        cur = unit_ratings_asof(ug[ug["season"] == 2024], 2024, week).set_index("team")
        exp = w * cur[cols] + (1 - w) * prev[cols]
    pd.testing.assert_frame_equal(both[cols].sort_index(), exp.sort_index(),
                                  check_exact=False, atol=1e-12)


def test_no_previous_season_uses_current_only():
    ug = league([2024], n_weeks=6)
    r = unit_ratings_asof(ug, 2024, 4).set_index("team")
    w = window_ratings(ug, 2024, 4)
    assert r.loc["A", "off_pass_epa"] == pytest.approx(w.loc["A", "off_pass_epa"])


def test_ratings_leak_free():
    ug = league([2023, 2024], n_weeks=12, seed=11)
    base = unit_ratings_asof(ug, 2024, 6)
    later = ug.copy()
    mask = (later["season"] == 2024) & (later["week"] >= 6)
    later.loc[mask, list(METRICS)] = later.loc[mask, list(METRICS)] * 5 + 3
    later.loc[mask, "pass_rate"] = 0.99
    pd.testing.assert_frame_equal(base, unit_ratings_asof(later, 2024, 6))
    dropped = ug[~mask]
    pd.testing.assert_frame_equal(base, unit_ratings_asof(dropped, 2024, 6))


def test_ratings_columns():
    r = unit_ratings_asof(league([2024], n_weeks=4), 2024, 3)
    for c in ["season", "week", "team", "games", "weight", "pass_rate",
              "off_pass_plays", "off_run_plays", "def_pass_plays", "def_run_plays"]:
        assert c in r.columns
    assert (r["season"] == 2024).all() and (r["week"] == 3).all()
    assert r["off_pass_plays"].iloc[0] == pytest.approx(35.0)


def test_nfl_unit_games_without_game_id_and_unknown_team():
    pbp = _pbp().drop(columns=["game_id"])
    pbp = pd.concat([pbp, pd.DataFrame([_play("XXX", "KC", "pass", 5, 0.1, 1)]).drop(columns=["game_id"])])
    ug = nfl_unit_games(pbp)
    assert set(ug["team"]) == {"KC", "BAL"}
    assert set(ug["game_id"]) == {"2024_01_BAL_KC"}


@pytest.mark.parametrize("week,games,w", [(2, 1, 0.5), (4, 3, 0.75), (10, 9, 0.9)])
def test_power_blend_k_weights_this_season(week, games, w):
    from sportsmodel.context.units import POWER_BLEND_K
    assert POWER_BLEND_K == 1.0
    ug = league([2023, 2024], n_weeks=12, profile={"A": {"pass_o": 3}}, seed=3)
    r = unit_ratings_asof(ug, 2024, week, blend_k=POWER_BLEND_K).set_index("team")
    assert (r["games"] == games).all()
    assert r["weight"].to_numpy() == pytest.approx(np.full(8, w))
    default = unit_ratings_asof(ug, 2024, week).set_index("team")
    assert default["weight"].to_numpy() == pytest.approx(np.full(8, games / (games + 3)))
