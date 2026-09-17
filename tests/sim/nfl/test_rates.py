"""Tests for leakage-free nflverse rate aggregation."""
import pandas as pd
import pytest

from sportsmodel.sim.nfl.rates import player_inputs_from_weekly, team_rates_from_pbp


def _pbp_rows():
    """Small synthetic play-by-play frame spanning two weeks for two teams.

    Columns mirror nflverse's `import_pbp_data` output: season, week,
    posteam, defteam, play_type, fixed_drive_result, drive, game_id, yardline_100.
    """
    rows = [
        # Week 1 (2023): KC offense, 2 drives -> 1 TD, 1 punt.
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             fixed_drive_result="Touchdown", drive=1, game_id="g1", yardline_100=15),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="run",
             fixed_drive_result="Touchdown", drive=1, game_id="g1", yardline_100=10),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             fixed_drive_result="Punt", drive=2, game_id="g1", yardline_100=60),
        # Week 1 (2023): BUF offense, 1 drive -> field goal.
        dict(season=2023, week=1, posteam="BUF", defteam="KC", play_type="run",
             fixed_drive_result="Field Goal", drive=3, game_id="g1", yardline_100=12),
        # Week 2 (2023): KC offense, 1 drive -> turnover. Included (week=2 < upto_week=3).
        dict(season=2023, week=2, posteam="KC", defteam="NE", play_type="pass",
             fixed_drive_result="Turnover", drive=4, game_id="g2", yardline_100=55),
        # Week 3 (2023): KC offense -> should be EXCLUDED (at upto_week boundary).
        dict(season=2023, week=3, posteam="KC", defteam="NE", play_type="pass",
             fixed_drive_result="Touchdown", drive=5, game_id="g3", yardline_100=8),
        # Season 2024 week 1: should be EXCLUDED (at/after upto_season boundary... wait
        # upto_season=2023 so season=2024 is after; excluded).
        dict(season=2024, week=1, posteam="KC", defteam="NE", play_type="pass",
             fixed_drive_result="Touchdown", drive=6, game_id="g4", yardline_100=5),
    ]
    return pd.DataFrame(rows)


def test_team_rates_excludes_rows_at_or_after_cutoff():
    """A drive in week 3 (== upto_week) and season 2024 (> upto_season) must not
    affect the aggregate — only weeks 1-2 of 2023 should be counted for KC."""
    pbp = _pbp_rows()
    rates = team_rates_from_pbp(pbp, upto_season=2023, upto_week=3)

    kc = rates["KC"]
    # KC drives strictly before (2023, 3): drive 1 (td), drive 2 (punt), drive 4 (turnover).
    assert kc.drive_outcomes["td"] == pytest.approx(1 / 3)
    assert kc.drive_outcomes["punt"] == pytest.approx(1 / 3)
    assert kc.drive_outcomes["turnover"] == pytest.approx(1 / 3)
    assert kc.drive_outcomes["fg"] == pytest.approx(0.0)
    # The week-3 and season-2024 touchdowns must not leak in.
    assert kc.drives_per_game == pytest.approx(3 / 2)  # 3 drives over 2 games (g1, g2)


def test_team_rates_drive_outcomes_sum_to_one():
    pbp = _pbp_rows()
    rates = team_rates_from_pbp(pbp, upto_season=2023, upto_week=3)
    for team_rates in rates.values():
        assert sum(team_rates.drive_outcomes.values()) == pytest.approx(1.0)
        assert set(team_rates.drive_outcomes.keys()) == {
            "td", "fg", "punt", "turnover", "downs", "end",
        }


def test_team_rates_pass_rate_and_buf_fg():
    pbp = _pbp_rows()
    rates = team_rates_from_pbp(pbp, upto_season=2023, upto_week=3)
    buf = rates["BUF"]
    # BUF has exactly 1 drive, ending in FG, and 1 run play (no pass plays).
    assert buf.drive_outcomes["fg"] == pytest.approx(1.0)
    assert buf.pass_rate == pytest.approx(0.0)

    kc = rates["KC"]
    # KC has 3 pass plays + 1 run play among the leaked-in rows.
    assert kc.pass_rate == pytest.approx(3 / 4)


def _weekly_rows():
    """Small synthetic weekly frame mirroring nflverse's `import_weekly_data`.

    Columns: player_id, player_display_name, position, recent_team, season, week,
    targets, carries, receptions, receiving_yards, rushing_yards, receiving_tds,
    rushing_tds.
    """
    rows = [
        # Week 1: WR1 targets=8 rec=6 rec_yds=90 td=1; RB1 carries=15 rush_yds=60 td=1.
        dict(player_id="p1", player_display_name="WR One", position="WR",
             recent_team="KC", season=2023, week=1, targets=8, carries=0,
             receptions=6, receiving_yards=90, rushing_yards=0,
             receiving_tds=1, rushing_tds=0),
        dict(player_id="p2", player_display_name="RB One", position="RB",
             recent_team="KC", season=2023, week=1, targets=2, carries=15,
             receptions=1, receiving_yards=5, rushing_yards=60,
             receiving_tds=0, rushing_tds=1),
        # Week 3: should be excluded (upto_week=3).
        dict(player_id="p1", player_display_name="WR One", position="WR",
             recent_team="KC", season=2023, week=3, targets=20, carries=0,
             receptions=20, receiving_yards=500, rushing_yards=0,
             receiving_tds=5, rushing_tds=0),
        # A player on a different team with zero targets/carries -- divide-by-zero guard.
        dict(player_id="p3", player_display_name="No Usage", position="WR",
             recent_team="NE", season=2023, week=1, targets=0, carries=0,
             receptions=0, receiving_yards=0, rushing_yards=0,
             receiving_tds=0, rushing_tds=0),
    ]
    return pd.DataFrame(rows)


def test_player_inputs_excludes_rows_at_or_after_cutoff_and_shares_sum_to_one():
    weekly = _weekly_rows()
    snaps = pd.DataFrame(columns=["player_id", "season", "week", "offense_pct"])

    inputs = player_inputs_from_weekly(weekly, snaps, upto_season=2023, upto_week=3)

    kc_players = {p.player_id: p for p in inputs["KC"]}
    assert "p1" in kc_players and "p2" in kc_players
    wr1 = kc_players["p1"]
    rb1 = kc_players["p2"]

    # Week-3 blowout stats for p1 must not leak in: target_share computed only
    # from week-1 targets (8 vs team total 10).
    assert wr1.target_share == pytest.approx(8 / 10)
    assert rb1.carry_share == pytest.approx(15 / 15)

    target_shares = sum(p.target_share for p in inputs["KC"])
    carry_shares = sum(p.carry_share for p in inputs["KC"])
    assert target_shares == pytest.approx(1.0)
    assert carry_shares == pytest.approx(1.0)

    assert wr1.ypt == pytest.approx(90 / 8)
    assert wr1.catch_rate == pytest.approx(6 / 8)
    assert rb1.ypc == pytest.approx(60 / 15)
    # Team TDs in window = 1 (p1) + 1 (p2) = 2.
    assert wr1.td_share == pytest.approx(0.5)
    assert rb1.td_share == pytest.approx(0.5)


def test_player_inputs_divide_by_zero_guarded():
    weekly = _weekly_rows()
    snaps = pd.DataFrame(columns=["player_id", "season", "week", "offense_pct"])

    inputs = player_inputs_from_weekly(weekly, snaps, upto_season=2023, upto_week=3)

    ne_players = {p.player_id: p for p in inputs["NE"]}
    no_usage = ne_players["p3"]
    assert no_usage.target_share == 0.0
    assert no_usage.carry_share == 0.0
    assert no_usage.ypt == 0.0
    assert no_usage.ypc == 0.0
    assert no_usage.catch_rate == 0.0
    assert no_usage.td_share == 0.0
