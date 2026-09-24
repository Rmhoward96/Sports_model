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


def test_player_redzone_ignores_non_plays():
    """Ruling: only pass/run plays count -- a QB kneel inside the 5 is not a carry."""
    kneel = pd.DataFrame({
        "season": [2024], "week": [1], "season_type": ["REG"], "play_id": [6], "posteam": ["KC"], "defteam": ["BAL"],
        "play_type": ["qb_kneel"], "sack": [0], "qb_hit": [0], "wp": [0.99], "down": [1], "qtr": [4],
        "yardline_100": [2], "receiver_player_id": [None], "rusher_player_id": ["r1"],
    })
    rz = player_redzone(pd.concat([_pbp(), kneel], ignore_index=True)).set_index("player_id")
    assert rz.loc["r1", "rz_carries"] == 1 and rz.loc["r1", "gl_carries"] == 1


import numpy as np

from sportsmodel.nfl.player_features import roll_features


def _series_df():
    return pd.DataFrame({"player_id": ["a"] * 5, "season": [2023, 2023, 2024, 2024, 2024],
                         "week": [1, 2, 1, 2, 3], "x": [1.0, 3.0, 5.0, 7.0, 9.0]})


def test_roll_features_are_strictly_prior():
    out = roll_features(_series_df(), "player_id", ["x"], "p_").set_index(["season", "week"])
    assert np.isnan(out.loc[(2023, 1), "p_x_r3"])                 # no history
    assert out.loc[(2024, 3), "p_x_r3"] == np.mean([3.0, 5.0, 7.0])  # excludes own game (9)
    assert out.loc[(2024, 3), "p_x_std"] == 6.0                   # season-to-date: 5, 7
    assert out.loc[(2024, 1), "p_x_prev"] == 2.0                  # 2023 mean
    assert out.loc[(2024, 3), "p_n_season"] == 2


def test_perturbing_target_week_and_later_does_not_change_features():
    """Leakage guard: garbage box scores at/after (2024, 3) must leave the
    (2024, 3) feature row identical."""
    from tests.nfl.fixtures_props import feature_inputs     # synthetic 2-season league (see Step 3)
    base = feature_inputs()
    t0 = _build(base).set_index(["player_id", "season", "week"])
    bad = feature_inputs(perturb_from=(2024, 3))
    t1 = _build(bad).set_index(["player_id", "season", "week"])
    feats = [c for c in t0.columns if not c.startswith("y_")]
    key = [k for k in t0.index if k[1:] == (2024, 3)]
    pd.testing.assert_frame_equal(t0.loc[key, feats], t1.loc[key, feats])
    assert not t0.loc[key, "y_targets"].equals(t1.loc[key, "y_targets"])  # labels did change


def _build(inp):
    from sportsmodel.nfl.player_features import build_feature_table
    return build_feature_table(**inp)


def test_feature_table_scripted_events():
    from tests.nfl.fixtures_props import feature_inputs
    inp = feature_inputs()
    t = _build(inp).set_index(["player_id", "season", "week"])
    assert not t.index.duplicated().any()
    raw = {"snap_pct", "target_share", "air_yards_share", "rec_air_yards", "yac", "ypt", "catch_rate",
           "rz_targets", "rz_carries", "gl_carries", "carry_share", "avg_separation", "efficiency"}
    assert raw.isdisjoint(t.columns)                                 # same-game box score never a feature
    assert {c.split("_")[0] for c in t.columns} >= {"p", "ngs", "tm", "op", "st", "cx", "mk", "y"}
    k = (2024, 3)
    # stub: labels NaN, history-only features
    stub = t.loc[("KC_RB2", *k)]
    pg = inp["pg"]
    rb2 = pg[pg["player_id"] == "KC_RB2"]
    assert np.isnan(stub["y_carries"]) and stub["p_career_games"] == 4
    assert stub["p_y_carries_prev"] == rb2["y_carries"].mean() and stub["p_depth_rank"] == 2
    # status: KC_WR + BAL_QB Out, BAL_QB2 starts, MIA_TE Questionable
    wr = pg[(pg["player_id"] == "KC_WR") & (pg["season"] * 100 + pg["week"] < 202403)]
    exp = wr["target_share"].ewm(halflife=4.0).mean().iloc[-1]
    assert np.isclose(t.loc[("KC_QB", *k), "st_vacated_tgt"], exp)
    assert np.isclose(t.loc[("KC_WR", 2024, 2), "st_vacated_tgt"], 0.0)
    assert t.loc[("BAL_WR", *k), "st_vacated_car"] > 0
    assert t.loc[("BAL_QB2", *k), "st_qb_changed"] == 1 and t.loc[("KC_QB", *k), "st_qb_changed"] == 0
    assert t.loc[("BAL_QB2", *k), "p_depth_rank"] == 1
    assert t.loc[("MIA_TE", *k), "st_questionable"] == 1 and t.loc[("MIA_WR", *k), "st_questionable"] == 0
    # team efficiency: nothing before week 1
    assert np.isnan(t.loc[("KC_QB", 2024, 1), "tm_off_adj"]) and not np.isnan(t.loc[("KC_QB", 2024, 2), "tm_off_adj"])
    assert not np.isnan(t.loc[("KC_QB", 2024, 1), "tm_off_prev"])
    assert str(t["p_pos"].dtype) == "category"


def test_team_table_labels_and_opponent_view():
    from sportsmodel.nfl.player_features import build_team_table
    from tests.nfl.fixtures_props import feature_inputs
    inp = feature_inputs()
    tt = build_team_table(inp["tg"], inp["ctx"], inp["game_epa"]).set_index(["team", "season", "week"])
    assert len(tt) == 32 and not tt.index.duplicated().any()
    tg = inp["tg"].set_index(["team", "season", "week"])
    assert tt.loc[("KC", 2023, 2), "y_team_pass_att"] == tg.loc[("KC", 2023, 2), "pass_att"]
    # KC faces BUF in wk2; BUF's defense faced MIA's offense in wk1
    assert tt.loc[("KC", 2023, 2), "op_pass_att_r3"] == tg.loc[("MIA", 2023, 1), "pass_att"]
    mia = tg.loc[("MIA", 2023, 1)]
    assert np.isclose(tt.loc[("KC", 2023, 2), "op_press_rate_r3"], mia["pressures_allowed"] / mia["dropbacks"])
    assert tt.loc[("KC", 2023, 2), "tm_pass_att_r3"] == tg.loc[("KC", 2023, 1), "pass_att"]
    assert {"cx_rest", "cx_home", "mk_implied", "tm_off_adj", "op_def_adj", "op_def_prev"} <= set(tt.columns)
    assert not any(c in tt.columns for c in ("pass_att", "rush_att", "plays", "neutral_pass_rate"))


def test_leakage_guard_also_moves_ratio_features():
    """x10 leaves per-game ratios unchanged (10y/10t); add an offset so ratio
    features (p_ypt_*, p_carry_share_*, op_press_rate_*, ...) are guarded too."""
    from tests.nfl.fixtures_props import feature_inputs
    t0 = _build(feature_inputs()).set_index(["player_id", "season", "week"])
    t1 = _build(feature_inputs(perturb_from=(2024, 3), offset=3.7)).set_index(["player_id", "season", "week"])
    feats = [c for c in t0.columns if not c.startswith("y_")]
    key = [k for k in t0.index if k[1:] == (2024, 3)]
    pd.testing.assert_frame_equal(t0.loc[key, feats], t1.loc[key, feats])
    later = [k for k in t0.index if k[1:] == (2024, 4)]
    assert not t0.loc[later, "p_ypt_r3"].equals(t1.loc[later, "p_ypt_r3"])   # the perturbation reaches ratios


def test_feature_table_accepts_string_dtype_ids_from_real_sources():
    """Real nflverse frames under pandas 3 carry ids as StringDtype (injuries,
    depth) while pg's player_id (pfr->gsis .map) is object; merge_asof in the
    status features must not reject the mixed key dtypes."""
    from tests.nfl.fixtures_props import feature_inputs
    base = _build(feature_inputs()).set_index(["player_id", "season", "week"])
    inp = feature_inputs()
    for k in ("injuries", "depth"):
        f = inp[k].copy()
        for c in ("gsis_id", "team", "club_code", "report_status", "position"):
            if c in f.columns:
                f[c] = f[c].astype("string")
        inp[k] = f
    inp["stubs"] = inp["stubs"].assign(player_id=inp["stubs"]["player_id"].astype("string"))
    got = _build(inp).set_index(["player_id", "season", "week"])
    pd.testing.assert_frame_equal(base, got, check_dtype=False, check_index_type=False)
