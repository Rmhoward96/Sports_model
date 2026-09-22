"""Tests for leakage-free nflverse rate aggregation."""
import pandas as pd
import pytest

from sportsmodel.sim.nfl.rates import (player_inputs_from_weekly, ratings_tilt,
                                       team_rates_from_pbp, team_defense_rates_from_pbp)


def _pbp_rows():
    """Small synthetic play-by-play frame spanning two weeks for two teams.

    Columns mirror nflverse's `import_pbp_data` output: season, week,
    posteam, defteam, play_type, fixed_drive_result, drive, game_id,
    yardline_100, sack, complete_pass. (sack/complete_pass are not exercised
    by the assertions in this fixture's tests; see `_volume_pbp_rows` for
    those.)
    """
    rows = [
        # Week 1 (2023): KC offense, 2 drives -> 1 TD, 1 punt.
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=1,
             fixed_drive_result="Touchdown", drive=1, game_id="g1", yardline_100=15),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0,
             fixed_drive_result="Touchdown", drive=1, game_id="g1", yardline_100=10),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=0,
             fixed_drive_result="Punt", drive=2, game_id="g1", yardline_100=60),
        # Week 1 (2023): BUF offense, 1 drive -> field goal.
        dict(season=2023, week=1, posteam="BUF", defteam="KC", play_type="run",
             sack=0, complete_pass=0,
             fixed_drive_result="Field Goal", drive=3, game_id="g1", yardline_100=12),
        # Week 2 (2023): KC offense, 1 drive -> turnover. Included (week=2 < upto_week=3).
        dict(season=2023, week=2, posteam="KC", defteam="NE", play_type="pass",
             sack=0, complete_pass=0,
             fixed_drive_result="Turnover", drive=4, game_id="g2", yardline_100=55),
        # Week 3 (2023): KC offense -> should be EXCLUDED (at upto_week boundary).
        dict(season=2023, week=3, posteam="KC", defteam="NE", play_type="pass",
             sack=0, complete_pass=1,
             fixed_drive_result="Touchdown", drive=5, game_id="g3", yardline_100=8),
        # Season 2024 week 1: should be EXCLUDED (at/after upto_season boundary... wait
        # upto_season=2023 so season=2024 is after; excluded).
        dict(season=2024, week=1, posteam="KC", defteam="NE", play_type="pass",
             sack=0, complete_pass=1,
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


def test_drive_result_map_pins_real_nflverse_labels():
    """Pin the exact fixed_drive_result -> bucket mapping against real nflverse
    2024 labels, including the v1 simplification that opponent-scoring drives
    (safety, opp touchdown) are treated as 0-point offensive turnovers."""
    rows = [
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=1,
             fixed_drive_result="Touchdown", drive=1, game_id="g1", yardline_100=10),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0,
             fixed_drive_result="Field Goal", drive=2, game_id="g1", yardline_100=15),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0,
             fixed_drive_result="Missed Field Goal", drive=3, game_id="g1", yardline_100=25),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=0,
             fixed_drive_result="Punt", drive=4, game_id="g1", yardline_100=60),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=0,
             fixed_drive_result="Turnover", drive=5, game_id="g1", yardline_100=50),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0,
             fixed_drive_result="Turnover on Downs", drive=6, game_id="g1", yardline_100=40),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0,
             fixed_drive_result="Safety", drive=7, game_id="g1", yardline_100=2),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=0,
             fixed_drive_result="Opp touchdown", drive=8, game_id="g1", yardline_100=95),
        dict(season=2024, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=0,
             fixed_drive_result="End of Half", drive=9, game_id="g1", yardline_100=70),
    ]
    pbp = pd.DataFrame(rows)
    rates = team_rates_from_pbp(pbp, upto_season=2024, upto_week=2)
    kc = rates["KC"]

    # 9 drives total, one each: td, fg, downs (missed FG), punt, turnover,
    # downs (turnover on downs), turnover (safety), turnover (opp td), end.
    assert kc.drive_outcomes["td"] == pytest.approx(1 / 9)
    assert kc.drive_outcomes["fg"] == pytest.approx(1 / 9)
    assert kc.drive_outcomes["downs"] == pytest.approx(2 / 9)  # missed FG + turnover on downs
    assert kc.drive_outcomes["punt"] == pytest.approx(1 / 9)
    assert kc.drive_outcomes["turnover"] == pytest.approx(3 / 9)  # turnover + safety + opp td
    assert kc.drive_outcomes["end"] == pytest.approx(1 / 9)
    assert sum(kc.drive_outcomes.values()) == pytest.approx(1.0)


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


def _volume_pbp_rows():
    """Synthetic pbp for one posteam (KC) across 2 games, covering pass/run/sack/
    completion combinations, plus a leaked-in row at/after the cutoff with big
    counts that must NOT affect the rates.

    upto_season=2023, upto_week=3 -> strictly-before rows are weeks 1-2.
    Game 1 (g1, week 1): 3 pass attempts (2 complete, 1 incomplete), 1 sack
        (pass play, sack==1, must NOT count as an attempt), 2 runs.
    Game 2 (g2, week 2): 1 pass attempt (complete), 1 sack, 1 run.
    Leaked row (week 3, g3): huge counts, must be excluded entirely.
    """
    rows = [
        # --- Game 1 (week 1, 2023) ---
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=1, fixed_drive_result="Touchdown", drive=1,
             game_id="g1", yardline_100=50),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=1, fixed_drive_result="Touchdown", drive=1,
             game_id="g1", yardline_100=40),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=0, complete_pass=0, fixed_drive_result="Touchdown", drive=1,
             game_id="g1", yardline_100=30),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="pass",
             sack=1, complete_pass=0, fixed_drive_result="Punt", drive=2,
             game_id="g1", yardline_100=60),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0, fixed_drive_result="Touchdown", drive=1,
             game_id="g1", yardline_100=45),
        dict(season=2023, week=1, posteam="KC", defteam="BUF", play_type="run",
             sack=0, complete_pass=0, fixed_drive_result="Punt", drive=2,
             game_id="g1", yardline_100=55),
        # --- Game 2 (week 2, 2023) ---
        dict(season=2023, week=2, posteam="KC", defteam="NE", play_type="pass",
             sack=0, complete_pass=1, fixed_drive_result="Turnover", drive=4,
             game_id="g2", yardline_100=55),
        dict(season=2023, week=2, posteam="KC", defteam="NE", play_type="pass",
             sack=1, complete_pass=0, fixed_drive_result="Turnover", drive=4,
             game_id="g2", yardline_100=55),
        dict(season=2023, week=2, posteam="KC", defteam="NE", play_type="run",
             sack=0, complete_pass=0, fixed_drive_result="Turnover", drive=4,
             game_id="g2", yardline_100=55),
        # --- Leaked row: week 3 (== upto_week boundary), huge counts ---
        dict(season=2023, week=3, posteam="KC", defteam="NE", play_type="pass",
             sack=0, complete_pass=1, fixed_drive_result="Touchdown", drive=5,
             game_id="g3", yardline_100=8),
    ]
    return pd.DataFrame(rows)


def test_team_rates_volume_fields():
    """pass_att_pg excludes sacks, rush_att_pg counts runs, sack_rate and
    completion_pct are correct ratios, all per-game and leakage-free."""
    pbp = _volume_pbp_rows()
    rates = team_rates_from_pbp(pbp, upto_season=2023, upto_week=3)
    kc = rates["KC"]

    # n_games = 2 (g1, g2). Leaked g3 row excluded entirely.
    # Pass plays (play_type=="pass") strictly-before-cutoff: g1 has 4 (3 non-sack
    # + 1 sack), g2 has 2 (1 non-sack + 1 sack) -> 6 pass plays total.
    # Attempts (pass & sack!=1): g1 has 3, g2 has 1 -> 4 attempts total.
    assert kc.pass_att_pg == pytest.approx(4 / 2)
    # Runs: g1 has 2, g2 has 1 -> 3 runs total.
    assert kc.rush_att_pg == pytest.approx(3 / 2)
    # Sacks: g1 has 1, g2 has 1 -> 2 sacks / 6 pass plays.
    assert kc.sack_rate == pytest.approx(2 / 6)
    # Completions: g1 has 2, g2 has 1 -> 3 completions / 4 attempts.
    assert kc.completion_pct == pytest.approx(3 / 4)


def test_team_rates_volume_fields_divide_by_zero_guarded():
    """A team with no pass plays at all must not raise and must report 0.0
    for sack_rate/completion_pct (denominators are zero)."""
    rows = [
        dict(season=2023, week=1, posteam="BUF", defteam="KC", play_type="run",
             sack=0, complete_pass=0, fixed_drive_result="Field Goal", drive=1,
             game_id="g1", yardline_100=12),
    ]
    pbp = pd.DataFrame(rows)
    rates = team_rates_from_pbp(pbp, upto_season=2023, upto_week=3)
    buf = rates["BUF"]

    assert buf.pass_att_pg == pytest.approx(0.0)
    assert buf.rush_att_pg == pytest.approx(1.0)
    assert buf.sack_rate == pytest.approx(0.0)
    assert buf.completion_pct == pytest.approx(0.0)


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

    # ypr = receiving_yards / receptions, distinct from ypt = receiving_yards
    # / targets (FIX 1: the kernel attributes reception yardage using ypr,
    # not ypt, so this needs its own correct computation here).
    assert wr1.ypr == pytest.approx(90 / 6)
    assert wr1.ypr != pytest.approx(wr1.ypt)
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
    assert no_usage.ypr == 0.0
    assert no_usage.catch_rate == 0.0
    assert no_usage.td_share == 0.0


# --- season weighting (current season heavier than prior) ---

def test_ratings_tilt_sign_scale_and_off_switch():
    # Stronger home -> positive tilt; symmetric when teams swap; weight 0 = off.
    assert ratings_tilt(1600, 1400, weight=1.0) > 0
    assert ratings_tilt(1400, 1600, weight=1.0) == pytest.approx(
        -ratings_tilt(1600, 1400, weight=1.0))
    assert ratings_tilt(1600, 1400, weight=0.0) == 0.0
    assert ratings_tilt(1500, 1500, weight=1.0) == 0.0
    # Bigger Elo gap -> bigger tilt; heavier weight -> bigger tilt.
    assert ratings_tilt(1700, 1400, 1.0) > ratings_tilt(1600, 1400, 1.0)
    assert ratings_tilt(1600, 1400, 2.0) == pytest.approx(2 * ratings_tilt(1600, 1400, 1.0))


def test_team_defense_rates_group_by_defteam():
    # DEN is on defense for two drives it faced: one TD, one punt -> allowed
    # drive_outcomes should be td=0.5, punt=0.5 (grouped by defteam, not posteam).
    pbp = pd.DataFrame([
        dict(season=2023, week=1, posteam="KC", defteam="DEN", play_type="pass",
             sack=0, complete_pass=1, fixed_drive_result="Touchdown",
             drive=1, game_id="g1", yardline_100=15),
        dict(season=2023, week=1, posteam="KC", defteam="DEN", play_type="run",
             sack=0, complete_pass=0, fixed_drive_result="Punt",
             drive=2, game_id="g1", yardline_100=60),
    ])
    d = team_defense_rates_from_pbp(pbp, upto_season=2023, upto_week=2)
    assert "DEN" in d
    assert d["DEN"].drive_outcomes["td"] == pytest.approx(0.5)
    assert d["DEN"].drive_outcomes["punt"] == pytest.approx(0.5)


def test_season_weights_decay():
    from sportsmodel.sim.nfl.rates import season_weights
    s = pd.Series([2022, 2023, 2024])
    w = season_weights(s, upto_season=2024, decay=0.5)
    assert list(w) == [0.25, 0.5, 1.0]           # current=1, each prior *0.5
    w1 = season_weights(s, upto_season=2024, decay=1.0)
    assert list(w1) == [1.0, 1.0, 1.0]           # uniform


def _two_season_pbp():
    """KC: prior season (2023) all runs/punts, current (2024) all passes/TDs."""
    common = dict(sack=0, defteam="X")
    return pd.DataFrame([
        dict(season=2023, week=1, posteam="KC", play_type="run", complete_pass=0,
             fixed_drive_result="Punt", drive=1, game_id="a", yardline_100=60, **common),
        dict(season=2023, week=1, posteam="KC", play_type="run", complete_pass=0,
             fixed_drive_result="Punt", drive=2, game_id="a", yardline_100=55, **common),
        dict(season=2024, week=1, posteam="KC", play_type="pass", complete_pass=1,
             fixed_drive_result="Touchdown", drive=1, game_id="b", yardline_100=15, **common),
        dict(season=2024, week=1, posteam="KC", play_type="pass", complete_pass=1,
             fixed_drive_result="Touchdown", drive=2, game_id="b", yardline_100=10, **common),
    ])


def test_team_rates_weighting_pulls_toward_current_season():
    pbp = _two_season_pbp()
    uniform = team_rates_from_pbp(pbp, 2024, 2, season_decay=1.0)["KC"]
    weighted = team_rates_from_pbp(pbp, 2024, 2, season_decay=0.5)["KC"]
    # Uniform: equal pass/run -> 0.5. Weighted: 2024(pass) w=1 vs 2023(run) w=0.5 -> 2/3.
    assert uniform.pass_rate == pytest.approx(0.5)
    assert weighted.pass_rate == pytest.approx(2 / 3)
    # Current season is all TD drives -> weighting raises the TD-drive share.
    assert weighted.drive_outcomes["td"] > uniform.drive_outcomes["td"]


def _two_season_weekly():
    """P1 breaks out in the current season (2024); P2 fades."""
    def r(season, pid, name, tgt, tds):
        return dict(season=season, week=1, recent_team="KC", player_id=pid,
                    player_display_name=name, position="WR", targets=tgt, carries=0,
                    receptions=tgt, receiving_yards=tgt * 10, rushing_yards=0,
                    receiving_tds=tds, rushing_tds=0)
    return pd.DataFrame([
        r(2023, "P1", "P One", 2, 0), r(2023, "P2", "P Two", 8, 1),
        r(2024, "P1", "P One", 10, 1), r(2024, "P2", "P Two", 2, 0),
    ])


def test_player_inputs_weighting_emphasizes_recent_usage():
    weekly = _two_season_weekly()
    snaps = pd.DataFrame(columns=["player_id", "season", "week", "offense_pct"])
    uniform = {p.player_id: p for p in
               player_inputs_from_weekly(weekly, snaps, 2024, 2, season_decay=1.0)["KC"]}
    weighted = {p.player_id: p for p in
                player_inputs_from_weekly(weekly, snaps, 2024, 2, season_decay=0.5)["KC"]}
    # P1's current-season breakout gets more weight -> higher target share.
    assert weighted["P1"].target_share > uniform["P1"].target_share
    assert weighted["P1"].target_share == pytest.approx(11 / 17)  # (2*.5+10)/((2+8)*.5+(10+2))
