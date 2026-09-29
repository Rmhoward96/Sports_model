import numpy as np
import pandas as pd

from sportsmodel.nfl.unit_efficiency import UNIT_METRICS, unit_features, unit_games


def _play(season, week, off, de, kind, yds, epa, success, sack=0):
    return {"season": season, "week": week, "season_type": "REG", "posteam": off, "defteam": de,
            "play_type": kind, "yards_gained": yds, "epa": epa, "success": success, "sack": sack}


def _pbp():
    rows = []
    # week 1: KC offense vs BUF D, BAL offense vs PIT D (and reverse)
    for wk, pairs in ((1, [("KC", "BUF"), ("BUF", "KC"), ("BAL", "PIT"), ("PIT", "BAL")]),
                      (2, [("KC", "BAL"), ("BAL", "KC"), ("BUF", "PIT"), ("PIT", "BUF")])):
        for off, de in pairs:
            # BUF's defense: stout vs the run (-1 yd carries), leaky vs the pass (+12 yd dropbacks)
            rush_yds = -1 if de == "BUF" else 5
            pass_yds = 12 if de == "BUF" else 6
            rows += [_play(2024, wk, off, de, "run", rush_yds, 0.1 if rush_yds > 0 else -0.5, int(rush_yds > 3))
                     for _ in range(10)]
            rows += [_play(2024, wk, off, de, "pass", pass_yds, 0.4 if pass_yds > 8 else 0.0, 1)
                     for _ in range(10)]
            rows.append(_play(2024, wk, off, de, "pass", -7, -1.5, 0, sack=1))
    return pd.DataFrame(rows)


def test_unit_games_rates():
    ug = unit_games(_pbp())
    kc1 = ug[(ug.week == 1) & (ug.team == "KC")].iloc[0]
    assert kc1.opponent == "BUF"
    assert kc1.ypc == -1.0 and kc1.stuff_rate == 1.0
    assert np.isclose(kc1.nypd, (12 * 10 - 7) / 11) and np.isclose(kc1.sack_rate, 1 / 11)
    assert set(UNIT_METRICS) <= set(ug.columns)


def test_features_are_strictly_prior_and_relative_to_league():
    ug = unit_games(_pbp())
    tw = pd.DataFrame({"season": [2024, 2024, 2024], "week": [1, 3, 3],
                       "team": ["KC", "KC", "PIT"], "opponent": ["BUF", "BUF", "BUF"]})
    f = unit_features(ug, tw)
    w1 = f[f.week == 1].iloc[0]
    assert np.isnan(w1.mx_op_ypc_allowed_adj)          # nothing before week 1 of 2024
    w3 = f[(f.week == 3) & (f.team == "KC")].iloc[0]
    # BUF's run D is the league's best (negative allowed), its pass D the worst (positive allowed)
    assert w3.mx_op_ypc_allowed_adj < 0 < w3.mx_op_nypd_allowed_adj
    assert w3.mx_pass_minus_rush > 0                    # R1: + means pass D weak vs run D
    assert w3.mx_pass_edge == w3.mx_tm_pass_epa_adj + w3.mx_op_pass_epa_allowed_adj


def test_changing_week_3_plays_does_not_move_week_3_features():
    pbp = _pbp()
    tw = pd.DataFrame({"season": [2024], "week": [3], "team": ["KC"], "opponent": ["BUF"]})
    base = unit_features(unit_games(pbp), tw)
    extra = pd.DataFrame([_play(2024, 3, "KC", "BUF", "run", 80, 5.0, 1)] * 50)
    moved = unit_features(unit_games(pd.concat([pbp, extra], ignore_index=True)), tw)
    cols = [c for c in base.columns if c.startswith("mx_")]
    pd.testing.assert_frame_equal(base[cols], moved[cols])


def test_prev_uses_last_season_full():
    pbp = _pbp()
    tw = pd.DataFrame({"season": [2025], "week": [1], "team": ["KC"], "opponent": ["BUF"]})
    f = unit_features(unit_games(pbp), tw).iloc[0]
    assert f.mx_op_ypc_allowed_prev < 0 and np.isnan(f.mx_op_ypc_allowed_adj)


def test_qb_scramble_counts_as_dropback_not_designed_run():
    """A scramble (play_type 'run', qb_scramble == 1) is a dropback: it must
    move nypd/pass_epa but leave ypc/stuff_rate untouched."""
    pbp = _pbp()
    base = unit_games(pbp)
    kc1_base = base[(base.week == 1) & (base.team == "KC")].iloc[0]

    scramble = _play(2024, 1, "KC", "BUF", "run", 9, 0.3, 1)
    scramble["qb_scramble"] = 1
    pbp_with_scramble = pd.concat([pbp, pd.DataFrame([scramble])], ignore_index=True)
    ug = unit_games(pbp_with_scramble)
    kc1 = ug[(ug.week == 1) & (ug.team == "KC")].iloc[0]

    # dropbacks go from 11 to 12, gaining the scramble's 9 yards
    assert np.isclose(kc1.nypd, (12 * 10 - 7 + 9) / 12)
    assert not np.isclose(kc1.nypd, kc1_base.nypd)
    # designed-run metrics are unaffected by the scramble
    assert kc1.ypc == kc1_base.ypc == -1.0
    assert kc1.stuff_rate == kc1_base.stuff_rate == 1.0
