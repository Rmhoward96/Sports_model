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
    from tests.nfl.fixtures_props import feature_inputs, leaked_features   # synthetic 2-season league
    assert leaked_features(_build) == []
    t0 = _build(feature_inputs()).set_index(["player_id", "season", "week"])
    t1 = _build(feature_inputs(perturb_from=(2024, 3))).set_index(["player_id", "season", "week"])
    key = [k for k in t0.index if k[1:] == (2024, 3)]
    assert not t0.loc[key, "y_targets"].equals(t1.loc[key, "y_targets"])  # labels did change


def test_planted_leaky_feature_is_caught():
    """The perturbation check must detect a feature that reads the target
    week's own label (spec: 'a planted leaky feature is caught')."""
    from tests.nfl.fixtures_props import leaked_features

    def leaky_build(inp):
        t = _build(inp)
        return t.assign(p_leak_targets=t["y_targets"],                # the target week's label
                        p_leak_next=t.groupby("player_id")["y_targets"].shift(-1))  # a later week's
    assert leaked_features(leaky_build) == ["p_leak_targets", "p_leak_next"]
    assert leaked_features(leaky_build, offset=3.7) == ["p_leak_targets", "p_leak_next"]


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
    from tests.nfl.fixtures_props import feature_inputs, leaked_features
    assert leaked_features(_build, offset=3.7) == []
    t0 = _build(feature_inputs()).set_index(["player_id", "season", "week"])
    t1 = _build(feature_inputs(perturb_from=(2024, 3), offset=3.7)).set_index(["player_id", "season", "week"])
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


# ---- p_depth_rank: rank within (team, season, week, position), both eras -----------

def _snap_pg(rows):
    """Minimal played rows (player_id, season, week, snap_pct) for _depth_rank."""
    return pd.DataFrame(rows, columns=["player_id", "season", "week", "snap_pct"])


def _depth_rows(rows):
    return pd.DataFrame(rows, columns=["season", "week", "club_code", "depth_team", "position", "gsis_id"])


def test_depth_rank_old_schema_slot_ties_broken_by_prior_snap_share():
    """Old schema (<= 2024): WR1/WR2/WR3 are each depth_team 1 (LWR/RWR/SWR).
    The rank orders them by strictly-prior snap-share EWM."""
    from sportsmodel.nfl.player_features import _depth_rank
    pg = _snap_pg([("wa", 2024, 1, 0.50), ("wb", 2024, 1, 0.95), ("wc", 2024, 1, 0.70)])
    depth = _depth_rows([(2024, 2, "KC", 1.0, "WR", "wa"), (2024, 2, "KC", 1.0, "WR", "wb"),
                         (2024, 2, "KC", 1.0, "WR", "wc"), (2024, 2, "KC", 2.0, "WR", "wd"),
                         (2024, 2, "KC", 1.0, "RB", "rb")])
    r = _depth_rank(depth, pg).set_index("player_id")["p_depth_rank"]
    assert r.to_dict() == {"wb": 1, "wc": 2, "wa": 3, "wd": 4, "rb": 1}


def test_depth_rank_snapshot_schema_equals_pos_rank_order():
    """Snapshot schema (2025+): depth_team is pos_rank (already a within-position
    order) -> the rank reproduces it, whatever the prior snap shares."""
    from sportsmodel.nfl.player_features import _depth_rank
    pg = _snap_pg([("w1", 2025, 1, 0.40), ("w2", 2025, 1, 0.90), ("w3", 2025, 1, 0.99), ("w4", 2025, 1, 1.0)])
    depth = _depth_rows([(2025, 2, "KC", 3.0, "WR", "w3"), (2025, 2, "KC", 1.0, "WR", "w1"),
                         (2025, 2, "KC", 4.0, "WR", "w4"), (2025, 2, "KC", 2.0, "WR", "w2")])
    r = _depth_rank(depth, pg).set_index("player_id")["p_depth_rank"]
    assert r.to_dict() == {"w1": 1, "w2": 2, "w3": 3, "w4": 4}


def test_depth_rank_tie_break_uses_only_strictly_prior_snaps_and_nan_last():
    """Leakage: the target week's own snap share must not move the rank; a
    player with no prior snaps sorts after tied players with history, then
    gsis_id breaks remaining ties; a double listing counts once (best slot)."""
    from sportsmodel.nfl.player_features import _depth_rank
    prior = [("wa", 2024, 1, 0.9), ("wb", 2024, 1, 0.5)]
    depth = _depth_rows([(2024, 2, "KC", 1.0, "WR", "wa"), (2024, 2, "KC", 1.0, "WR", "wb"),
                         (2024, 2, "KC", 1.0, "WR", "wz"), (2024, 2, "KC", 1.0, "WR", "wy"),
                         (2024, 2, "KC", 2.0, "WR", "wa")])                     # wa listed twice
    base = _depth_rank(depth, _snap_pg(prior)).set_index("player_id")["p_depth_rank"]
    assert base.to_dict() == {"wa": 1, "wb": 2, "wy": 3, "wz": 4}
    # the target week's snaps (reversed order) must not change anything
    leaky = _snap_pg(prior + [("wa", 2024, 2, 0.1), ("wb", 2024, 2, 1.0), ("wz", 2024, 2, 1.0)])
    pd.testing.assert_series_equal(base, _depth_rank(depth, leaky).set_index("player_id")["p_depth_rank"])


def test_depth_rank_is_comparable_across_eras_in_the_feature_table():
    """build_feature_table's p_depth_rank for the fixture league: every
    (team, season, week, position) group ranks 1..n."""
    from tests.nfl.fixtures_props import feature_inputs
    t = _build(feature_inputs())
    ranked = t.dropna(subset=["p_depth_rank"])
    assert (ranked["p_depth_rank"] >= 1).all()
    assert ranked["p_depth_rank"].max() == 2


def test_depth_rank_ignores_special_teams_listings_of_the_old_schema():
    """Old-schema KR/PR slots are listed under the player's own position (a
    backup WR who returns kicks is 'WR', depth_team 1, formation 'Special
    Teams'); the snapshot schema files them under pos_abb KR/PR instead. Only
    offensive listings (or snapshot rows, formation NaN) rank."""
    from sportsmodel.nfl.player_features import _depth_rank
    pg = _snap_pg([("w1", 2024, 1, 0.9), ("w2", 2024, 1, 0.8), ("kr", 2024, 1, 0.1)])
    depth = _depth_rows([(2024, 2, "KC", 1.0, "WR", "w1"), (2024, 2, "KC", 2.0, "WR", "w2"),
                         (2024, 2, "KC", 3.0, "WR", "kr"), (2024, 2, "KC", 1.0, "WR", "kr")])
    depth["formation"] = ["Offense", "Offense", "Offense", "Special Teams"]
    r = _depth_rank(depth, pg).set_index("player_id")["p_depth_rank"]
    assert r.to_dict() == {"w1": 1, "w2": 2, "kr": 3}


def test_feature_table_flags_stub_rows_explicitly():
    """is_stub marks active-but-no-snap rows (the learned count models train
    them as 0); it is a bool, never a feature (no feature prefix)."""
    from tests.nfl.fixtures_props import feature_inputs
    t = _build(feature_inputs())
    assert t["is_stub"].dtype == bool
    stubs = t[t["is_stub"]]
    assert set(stubs["player_id"]) == {"KC_RB2"} and len(stubs) == 4
    assert stubs["y_targets"].isna().all() and t.loc[~t["is_stub"], "y_targets"].notna().all()


def test_career_games_exclude_the_own_game_on_played_rows():
    """p_career_games on a PLAYED row counts prior played games only."""
    from tests.nfl.fixtures_props import feature_inputs
    inp = feature_inputs()
    t = _build(inp).set_index(["player_id", "season", "week"])
    pg = inp["pg"]
    for pid, key in (("KC_QB", (2024, 3)), ("KC_QB", (2023, 1)), ("BUF_WR", (2024, 4)), ("BAL_QB2", (2024, 3))):
        assert not t.loc[(pid, *key), "is_stub"]                          # a played row
        mine = pg[pg["player_id"] == pid]
        prior = int((mine["season"] * 100 + mine["week"] < key[0] * 100 + key[1]).sum())
        assert t.loc[(pid, *key), "p_career_games"] == prior
    assert t.loc[("KC_QB", 2024, 3), "p_career_games"] == 6 and t.loc[("BAL_QB2", 2024, 3), "p_career_games"] == 0
