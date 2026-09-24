import pandas as pd

from sportsmodel.nfl.player_features import player_games, player_redzone, team_games


def _weekly():
    return pd.DataFrame({
        "player_id": ["w1", "q1"], "season": [2024, 2024], "week": [1, 1], "season_type": ["REG", "REG"],
        "team": ["KC", "KC"], "opponent_team": ["BAL", "BAL"], "position": ["WR", "QB"],
        "targets": [8, 0], "carries": [1, 3], "attempts": [0, 30], "completions": [0, 20], "receptions": [6, 0],
        "receiving_yards": [90, 0], "rushing_yards": [4, 12], "passing_yards": [0, 250], "passing_tds": [0, 2],
        "receiving_tds": [1, 0], "rushing_tds": [0, 0], "receiving_air_yards": [70, 0],
        "receiving_yards_after_catch": [30, 0], "target_share": [0.27, 0.0], "air_yards_share": [0.3, 0.0],
    })


def _snaps():
    return pd.DataFrame({
        "season": [2024] * 4, "week": [1] * 4, "game_type": ["REG"] * 4,
        "pfr_player_id": ["W1", "Q1", "T1", "K1"], "team": ["KC"] * 4, "opponent": ["BAL"] * 4,
        "position": ["WR", "QB", "TE", "K"], "offense_snaps": [60, 65, 10, 0], "offense_pct": [0.92, 1.0, 0.15, 0.0],
    })


def test_population_is_skill_players_with_offensive_snaps_and_zero_filled_labels():
    pg = player_games(_weekly(), _snaps(), {"W1": "w1", "Q1": "q1", "T1": "t1", "K1": "k1"})
    assert sorted(pg["player_id"]) == ["q1", "t1", "w1"]          # kicker + 0-snap dropped
    t1 = pg.set_index("player_id").loc["t1"]
    assert t1["y_targets"] == 0 and t1["y_rec_yds"] == 0            # played, no stats -> 0 not NaN
    w1 = pg.set_index("player_id").loc["w1"]
    assert w1["y_anytime_td"] == 1 and w1["y_rec_yds"] == 90 and w1["snap_pct"] == 0.92


def _pbp():
    return pd.DataFrame({
        "season": [2024] * 6, "week": [1] * 6, "season_type": ["REG"] * 6, "play_id": range(6),
        "posteam": ["KC"] * 6, "defteam": ["BAL"] * 6,
        "play_type": ["pass", "pass", "run", "run", "pass", "no_play"],
        "sack": [0, 1, 0, 0, 0, 0], "qb_hit": [1, 0, 0, 0, 0, 0],
        "wp": [0.5, 0.5, 0.5, 0.9, 0.5, 0.5], "down": [1, 2, 1, 1, 3, 1], "qtr": [1, 1, 2, 4, 1, 1],
        "yardline_100": [15, 40, 4, 30, 60, 50],
        "receiver_player_id": ["w1", None, None, None, "w1", None],
        "rusher_player_id": [None, None, "r1", "r1", None, None],
    })


def test_team_games_counts_attempts_excluding_sacks():
    tg = team_games(_pbp()).iloc[0]
    assert tg["pass_att"] == 2 and tg["rush_att"] == 2 and tg["dropbacks"] == 3 and tg["plays"] == 5
    assert tg["pressures_allowed"] == 2                     # qb_hit + sack
    assert tg["neutral_pass_rate"] == 2 / 3                 # neutral: wp .2-.8, downs 1-2, qtr<=3


def test_player_redzone():
    rz = player_redzone(_pbp()).set_index("player_id")
    assert rz.loc["w1", "rz_targets"] == 1 and rz.loc["r1", "rz_carries"] == 1 and rz.loc["r1", "gl_carries"] == 1
