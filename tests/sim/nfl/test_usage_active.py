"""Tests for the per-week active-roster usage model (`active_usage`).

Fixtures mirror the nflverse shapes documented in usage.py / rates.py:
- depth_df   : season, week, club_code, depth_team, position, gsis_id,
               full_name, football_name
- weekly_df  : player_id (== gsis_id), player_display_name, position,
               recent_team, season, week, targets, carries, receptions,
               receiving_yards, rushing_yards, receiving_tds, rushing_tds
               (optionally `attempts` for the QB tiebreak)
- snaps_df   : pfr_player_id, offense_pct, season, week (reserved; v1 does
               not consume snaps)
"""
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.sim.nfl.spec import PlayerInput
from sportsmodel.sim.nfl.usage import active_usage


def _depth(rows):
    cols = ["season", "week", "club_code", "depth_team", "position",
            "gsis_id", "full_name", "football_name"]
    return pd.DataFrame(rows, columns=cols)


def _weekly(rows):
    cols = ["player_id", "player_display_name", "position", "recent_team",
            "season", "week", "targets", "carries", "receptions",
            "receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds"]
    return pd.DataFrame(rows, columns=cols)


def _wrow(pid, name, pos, team, season, week, *, targets=0, carries=0,
          receptions=0, rec_yds=0, rush_yds=0, rec_tds=0, rush_tds=0):
    return dict(player_id=pid, player_display_name=name, position=pos,
                recent_team=team, season=season, week=week, targets=targets,
                carries=carries, receptions=receptions, receiving_yards=rec_yds,
                rushing_yards=rush_yds, receiving_tds=rec_tds, rushing_tds=rush_tds)


_EMPTY_SNAPS = pd.DataFrame(
    columns=["pfr_player_id", "offense_pct", "season", "week"]
)


def test_out_player_excluded_from_active_set():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha Star", football_name="A.Star"),
        dict(season=2023, week=5, club_code="KC", depth_team="2", position="WR",
             gsis_id="gB", full_name="Beta Backup", football_name="B.Backup"),
    ])
    weekly = _weekly([
        _wrow("gA", "Alpha Star", "WR", "KC", 2023, 4, targets=8, receptions=6, rec_yds=90),
        _wrow("gB", "Beta Backup", "WR", "KC", 2023, 4, targets=5, receptions=3, rec_yds=40),
    ])
    players, _qb = active_usage(
        "KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {},
        injuries_out_names={"beta backup"},
    )
    ids = {p.player_id for p in players}
    assert "gA" in ids
    assert "gB" not in ids  # ruled OUT by name (case-insensitive)


def test_shares_renormalize_over_active_set_not_full_roster():
    """A high-usage player who is NOT active must not dilute active shares."""
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha Star", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="RB",
             gsis_id="gB", full_name="Bravo Back", football_name=None),
    ])
    weekly = _weekly([
        _wrow("gA", "Alpha Star", "WR", "KC", 2023, 4, targets=8, receptions=6, rec_yds=90),
        _wrow("gB", "Bravo Back", "RB", "KC", 2023, 4, targets=2, carries=15, rush_yds=60),
        # High-usage player present in weekly but NOT on the depth chart (inactive).
        _wrow("gX", "Xavier Ghost", "WR", "KC", 2023, 4, targets=12, receptions=9, rec_yds=140),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    by_id = {p.player_id: p for p in players}
    assert set(by_id) == {"gA", "gB"}
    # Active-only denominator = 8 + 2 = 10, NOT 8 + 2 + 12 = 22.
    assert by_id["gA"].target_share == pytest.approx(0.8)
    assert by_id["gA"].target_share != pytest.approx(8 / 22)
    assert sum(p.target_share for p in players) == pytest.approx(1.0)
    assert sum(p.carry_share for p in players) == pytest.approx(1.0)


def test_high_recent_usage_gets_proportionally_high_share():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="2", position="WR",
             gsis_id="gB", full_name="Bravo", football_name=None),
    ])
    weekly = _weekly([
        _wrow("gA", "Alpha", "WR", "KC", 2023, 4, targets=10, receptions=7, rec_yds=100),
        _wrow("gB", "Bravo", "WR", "KC", 2023, 4, targets=2, receptions=1, rec_yds=12),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    by_id = {p.player_id: p for p in players}
    assert by_id["gA"].target_share == pytest.approx(10 / 12)
    assert by_id["gB"].target_share == pytest.approx(2 / 12)


def test_recency_weighting_favors_recent_games():
    """Two players with equal raw totals: the one who ramped up recently
    gets the higher share under the recency-weighted scheme."""
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gFront", full_name="Front Loaded", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gBack", full_name="Back Loaded", football_name=None),
    ])
    weekly = _weekly([
        # Front-loaded: high early, low recent (raw total 20).
        _wrow("gFront", "Front Loaded", "WR", "KC", 2023, 2, targets=12),
        _wrow("gFront", "Front Loaded", "WR", "KC", 2023, 3, targets=6),
        _wrow("gFront", "Front Loaded", "WR", "KC", 2023, 4, targets=2),
        # Back-loaded: low early, high recent (raw total 20).
        _wrow("gBack", "Back Loaded", "WR", "KC", 2023, 2, targets=2),
        _wrow("gBack", "Back Loaded", "WR", "KC", 2023, 3, targets=6),
        _wrow("gBack", "Back Loaded", "WR", "KC", 2023, 4, targets=12),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    by_id = {p.player_id: p for p in players}
    # Equal raw totals => equal shares if unweighted; recency weighting must
    # push the back-loaded player above the front-loaded one.
    assert by_id["gBack"].target_share > by_id["gFront"].target_share


def test_qb1_returned_and_exactly_one_qb_in_active_set():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="QB",
             gsis_id="qbA", full_name="Starter QB", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="2", position="QB",
             gsis_id="qbB", full_name="Backup QB", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gW", full_name="Wide Out", football_name=None),
    ])
    weekly = _weekly([
        _wrow("qbA", "Starter QB", "QB", "KC", 2023, 4, carries=3, rush_yds=15),
        _wrow("qbB", "Backup QB", "QB", "KC", 2023, 4, carries=1, rush_yds=2),
        _wrow("gW", "Wide Out", "WR", "KC", 2023, 4, targets=8, receptions=6, rec_yds=90),
    ])
    players, qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    assert qb == "qbA"
    qbs = [p for p in players if p.pos == "QB"]
    assert len(qbs) == 1
    assert qbs[0].player_id == "qbA"


def test_qb1_tiebreak_by_recent_attempts_when_depth_team_equal():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="QB",
             gsis_id="qbA", full_name="QB A", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="QB",
             gsis_id="qbB", full_name="QB B", football_name=None),
    ])
    weekly = _weekly([
        _wrow("qbA", "QB A", "QB", "KC", 2023, 4),
        _wrow("qbB", "QB B", "QB", "KC", 2023, 4),
    ])
    weekly["attempts"] = [10, 35]  # qbB threw far more recently
    _players, qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    assert qb == "qbB"


def test_no_active_qb_returns_none_and_no_qb_input():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gW", full_name="Wide Out", football_name=None),
    ])
    weekly = _weekly([
        _wrow("gW", "Wide Out", "WR", "KC", 2023, 4, targets=8, receptions=6, rec_yds=90),
    ])
    players, qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    assert qb is None
    assert all(p.pos != "QB" for p in players)


def test_leakage_guard_discriminates_the_compound_boundary():
    """The leakage guard MUST be the compound compare
    ``season < S OR (season == S AND week < W)`` — not a week-only or a
    season-only shortcut. This fixture is built so ONLY the compound compare
    yields gA=0.75 / gB=0.25; every plausible buggy compare yields something
    else, so a regression to any of them fails here.

    Cutoff = (2023, week 5). Legit before-cutoff usage: gA=(2023,3):6 targets,
    gB=(2023,3):2 targets  =>  compound shares 6/8 and 2/8.

    Two trap rows on gA that the compound compare EXCLUDES:
    - (2023, week 5): same season, AT the cutoff week. A ``season <= S``
      (week-ignoring) or a ``week <= W`` compare would wrongly include it.
    - (2024, week 1): a LATER season but EARLIER week. A ``week < W`` compare
      would wrongly include it (1 < 5); the compound compare excludes it
      (2024 is not < 2023).

    Each buggy compare pulls a 100-target trap into gA and blows the split away
    from 0.75, so asserting exactly 0.75 / 0.25 discriminates the boundary.
    """
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gB", full_name="Bravo", football_name=None),
    ])
    weekly = _weekly([
        # Legit before-cutoff rows (included by the compound compare).
        _wrow("gA", "Alpha", "WR", "KC", 2023, 3, targets=6, receptions=4, rec_yds=60),
        _wrow("gB", "Bravo", "WR", "KC", 2023, 3, targets=2, receptions=1, rec_yds=12),
        # Trap 1 — same season, AT cutoff week: excluded only by "week < W".
        _wrow("gA", "Alpha", "WR", "KC", 2023, 5, targets=100, receptions=90, rec_yds=1200),
        # Trap 2 — LATER season, EARLIER week: a week-only compare wrongly keeps it.
        _wrow("gA", "Alpha", "WR", "KC", 2024, 1, targets=100, receptions=90, rec_yds=1200),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    by_id = {p.player_id: p for p in players}
    # Compound-correct split. A week-only compare -> gA ~0.97; a season-strict
    # compare -> both cold-start 0.5/0.5; a season-inclusive/week-<= compare ->
    # gA ~0.97. Only the compound compare lands exactly here.
    assert by_id["gA"].target_share == pytest.approx(0.75)
    assert by_id["gB"].target_share == pytest.approx(0.25)


def test_cold_start_active_player_gets_small_nonzero_share_no_crash():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha Star", football_name=None),
        # Cold-start rookie: on the depth chart, no weekly rows at all.
        dict(season=2023, week=5, club_code="KC", depth_team="2", position="WR",
             gsis_id="gRookie", full_name="Rook Ie", football_name=None),
    ])
    weekly = _weekly([
        _wrow("gA", "Alpha Star", "WR", "KC", 2023, 4, targets=10, receptions=8, rec_yds=120),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    by_id = {p.player_id: p for p in players}
    assert "gRookie" in by_id
    rookie = by_id["gRookie"]
    assert rookie.target_share > 0.0                     # nonzero prior
    assert rookie.target_share < by_id["gA"].target_share  # but small vs the starter
    # Cold-start efficiency defaults are populated (documented positional values).
    assert rookie.ypr > 0.0
    assert rookie.catch_rate > 0.0


def test_weighted_usage_guards_nan_stat_column_no_crash_finite_shares():
    """A NaN in a recent weekly stat column (nflverse occasionally has one)
    must not propagate to a NaN share -- it should be treated as 0 for that
    game, same as `_actual_player_stats` in backtest_sim_nfl.py does."""
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha Star", football_name=None),
        dict(season=2023, week=5, club_code="KC", depth_team="2", position="WR",
             gsis_id="gB", full_name="Beta Backup", football_name=None),
    ])
    weekly = _weekly([
        # gA's most recent game has a NaN targets value (missing stat).
        _wrow("gA", "Alpha Star", "WR", "KC", 2023, 3, targets=8, receptions=6, rec_yds=90),
        _wrow("gA", "Alpha Star", "WR", "KC", 2023, 4, targets=np.nan, receptions=5, rec_yds=70),
        _wrow("gB", "Beta Backup", "WR", "KC", 2023, 4, targets=4, receptions=3, rec_yds=40),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    assert len(players) == 2
    for p in players:
        for field in ("target_share", "carry_share", "td_share", "ypt", "ypc", "ypr", "catch_rate"):
            value = getattr(p, field)
            assert math.isfinite(value), f"{p.player_id}.{field} is not finite: {value!r}"
    assert sum(p.target_share for p in players) == pytest.approx(1.0)


def test_returns_playerinput_instances():
    depth = _depth([
        dict(season=2023, week=5, club_code="KC", depth_team="1", position="WR",
             gsis_id="gA", full_name="Alpha", football_name=None),
    ])
    weekly = _weekly([
        _wrow("gA", "Alpha", "WR", "KC", 2023, 4, targets=8, receptions=6, rec_yds=90),
    ])
    players, _qb = active_usage("KC", 2023, 5, depth, weekly, _EMPTY_SNAPS, {}, set())
    assert all(isinstance(p, PlayerInput) for p in players)
