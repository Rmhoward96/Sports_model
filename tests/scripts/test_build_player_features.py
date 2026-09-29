"""Tests for the pure seams in build_player_features.py (`active_stubs`,
`dropped_snap_mappings`, `nan_share_by_group`). main()'s nflverse IO and
parquet writes are not unit tested. Synthetic frames only -- no network."""
import importlib.util
import math
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_player_features.py"
_spec = importlib.util.spec_from_file_location("build_player_features", _p)
bpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpf)


def _depth(rows):
    return pd.DataFrame(rows, columns=["season", "week", "club_code", "depth_team", "position", "gsis_id"])


def _sched():
    return pd.DataFrame({
        "season": [2024, 2024], "week": [1, 1], "game_type": ["REG", "REG"],
        "home_team": ["KC", "LAR"], "away_team": ["BAL", "DET"],
    })


def _injuries(rows):
    return pd.DataFrame(rows, columns=["season", "week", "team", "gsis_id", "report_status"])


def _pg(rows):
    return pd.DataFrame(rows, columns=["player_id", "season", "week"])


def test_active_stubs_excludes_played_and_out_players():
    depth = _depth([
        (2024, 1, "KC", 1, "QB", "QB1"),
        (2024, 1, "KC", 2, "QB", "QB2"),
        (2024, 1, "KC", 1, "WR", "WR1"),
        (2024, 1, "KC", 2, "WR", "WR2"),
    ])
    inj = _injuries([(2024, 1, "KC", "WR1", "Out")])
    pg = _pg([("QB1", 2024, 1)])
    out = bpf.active_stubs(depth, inj, pg, _sched())
    assert list(out.columns) == ["player_id", "season", "week", "team", "opponent", "position"]
    got = out.sort_values("player_id").reset_index(drop=True)
    assert got["player_id"].tolist() == ["QB2", "WR2"]
    assert got["team"].tolist() == ["KC", "KC"]
    assert got["opponent"].tolist() == ["BAL", "BAL"]
    assert got["position"].tolist() == ["QB", "WR"]
    assert got["season"].tolist() == [2024, 2024] and got["week"].tolist() == [1, 1]


def test_active_stubs_doubtful_excluded_questionable_kept_and_non_skill_dropped():
    depth = _depth([
        (2024, 1, "KC", 1, "RB", "RB1"),
        (2024, 1, "KC", 1, "TE", "TE1"),
        (2024, 1, "KC", 1, "LT", "OL1"),        # not a skill position
        (2024, 1, "KC", 1, "WR", None),         # no gsis id
    ])
    inj = _injuries([(2024, 1, "KC", "RB1", "Doubtful"), (2024, 1, "KC", "TE1", "Questionable")])
    out = bpf.active_stubs(depth, inj, _pg([]), _sched())
    assert out["player_id"].tolist() == ["TE1"]


def test_active_stubs_normalizes_codes_and_uses_opponent_from_schedule():
    # depth lists LAR (alias of LA); schedule's home team is LAR too
    depth = _depth([(2024, 1, "LAR", 1, "WR", "W"), (2024, 1, "DET", 1, "TE", "T")])
    out = bpf.active_stubs(depth, _injuries([]), _pg([]), _sched()).set_index("player_id")
    assert out.loc["W", "team"] == "LA" and out.loc["W", "opponent"] == "DET"
    assert out.loc["T", "team"] == "DET" and out.loc["T", "opponent"] == "LA"


def test_active_stubs_injury_only_matches_its_own_week_and_dedupes():
    depth = _depth([
        (2024, 1, "KC", 1, "WR", "WR1"),
        (2024, 1, "KC", 1, "WR", "WR1"),        # listed twice (two formations)
        (2024, 2, "KC", 1, "WR", "WR1"),        # week 2 has no REG game in schedule
    ])
    inj = _injuries([(2024, 2, "KC", "WR1", "Out")])   # a different week
    out = bpf.active_stubs(depth, inj, _pg([("WR1", 2023, 1)]), _sched())
    assert len(out) == 1 and out.iloc[0]["week"] == 1 and out.iloc[0]["player_id"] == "WR1"


def test_active_stubs_ignores_non_reg_games_and_handles_empty_injuries():
    sched = _sched().assign(game_type=["REG", "POST"])
    depth = _depth([(2024, 1, "LA", 1, "QB", "Q"), (2024, 1, "KC", 1, "QB", "K")])
    out = bpf.active_stubs(depth, pd.DataFrame(), _pg([]), sched)
    assert out["player_id"].tolist() == ["K"]


def test_dropped_snap_mappings_counts_unmapped_skill_rows_per_season():
    snaps = pd.DataFrame({
        "season": [2023, 2023, 2023, 2024, 2024],
        "game_type": ["REG"] * 5,
        "position": ["WR", "QB", "LT", "RB", "TE"],
        "offense_snaps": [10, 20, 30, 5, 0],
        "pfr_player_id": ["a", "zz", "yy", "b", "xx"],
    })
    got = bpf.dropped_snap_mappings(snaps, {"a": "G1", "b": "G2"})
    # 2023: QB "zz" unmapped (LT is not skill); 2024: TE with 0 snaps doesn't count
    assert got == {2023: 1, 2024: 0}


def test_nan_share_by_group():
    df = pd.DataFrame({"p_a": [1.0, math.nan], "p_b": [math.nan, math.nan], "cx_x": [1.0, 2.0],
                       "y_targets": [math.nan, 1.0]})
    got = bpf.nan_share_by_group(df)
    assert got["p_"] == 0.75 and got["cx_"] == 0.0
    assert "y_" not in got and math.isnan(got["mk_"])


def test_active_stubs_fall_back_to_latest_earlier_chart_like_active_usage():
    """A team-week with no exact chart uses the team's latest earlier chart
    (usage.active_usage's fallback), so an active non-player still gets a row."""
    sched = pd.DataFrame({"season": [2024, 2024, 2024], "week": [1, 2, 3], "game_type": ["REG"] * 3,
                          "home_team": ["KC", "KC", "BAL"], "away_team": ["BAL", "DET", "KC"]})
    depth = _depth([
        (2024, 1, "KC", 1, "WR", "WR1"),
        (2024, 1, "KC", 2, "WR", "WR2"),
        (2024, 3, "KC", 1, "WR", "WR1"),        # week 3: exact chart, WR2 no longer listed
    ])
    inj = _injuries([(2024, 2, "KC", "WR1", "Out")])
    out = bpf.active_stubs(depth, inj, _pg([]), sched)
    got = sorted(zip(out["player_id"], out["week"], out["opponent"]))
    # wk1 exact chart; wk2 has no chart -> wk1's (WR1 Out that week); wk3 exact chart
    assert got == [("WR1", 1, "BAL"), ("WR1", 3, "BAL"), ("WR2", 1, "BAL"), ("WR2", 2, "DET")]
    assert set(out.loc[out["team"] == "BAL", "player_id"]) == set()   # BAL never had a chart


def test_chart_coverage_counts_exact_fallback_and_none():
    sched = pd.DataFrame({"season": [2024, 2024], "week": [1, 2], "game_type": ["REG"] * 2,
                          "home_team": ["KC", "KC"], "away_team": ["BAL", "BAL"]})
    depth = _depth([(2024, 1, "KC", 1, "WR", "W")])
    # KC wk1 exact, KC wk2 fallback to wk1, BAL both weeks none
    assert bpf.chart_coverage(depth, sched) == {"exact": 1, "fallback": 1, "none": 2}


def test_depth_rank_distribution_shares_by_season():
    feats = pd.DataFrame({"season": [2024] * 4 + [2025] * 2, "position": ["WR"] * 5 + ["RB"],
                          "p_depth_rank": [1.0, 2.0, 6.0, math.nan, 1.0, 1.0]})
    got = bpf.depth_rank_distribution(feats, "WR")
    assert got.loc[2024, "rows"] == 4 and got.loc[2025, "rows"] == 1
    assert got.loc[2024, ["1", "2", "5+", "nan"]].tolist() == [0.25, 0.25, 0.25, 0.25]
    assert got.loc[2024, "mean"] == 3.0 and got.loc[2025, "1"] == 1.0


def test_build_tables_applies_ctx_fill_before_both_builders(monkeypatch):
    """build_tables (shared by the parquet build and live serving) passes
    ctx_fill's output -- called as ctx_fill(ctx, stadiums, sched) -- to both
    table builders; without ctx_fill the raw context goes through."""
    import sportsmodel.nfl.context as ctx_mod
    import sportsmodel.nfl.efficiency as eff_mod
    import sportsmodel.nfl.player_features as pf_mod

    raw_ctx, filled = pd.DataFrame({"c": [1]}), pd.DataFrame({"c": [2]})
    seen: dict = {}
    monkeypatch.setattr(ctx_mod, "load_stadiums", lambda: {"S": {}})
    monkeypatch.setattr(ctx_mod, "team_game_context", lambda sched, st: raw_ctx)
    monkeypatch.setattr(eff_mod, "team_game_epa", lambda pbp: {})
    for name in ("player_games", "team_games", "player_redzone"):
        monkeypatch.setattr(pf_mod, name, lambda *a: pd.DataFrame())
    monkeypatch.setattr(bpf, "active_stubs", lambda *a: pd.DataFrame())
    monkeypatch.setattr(pf_mod, "build_feature_table",
                        lambda pg, tg, rz, ctx, *a, **k: seen.setdefault("feats_ctx", ctx))
    monkeypatch.setattr(pf_mod, "build_team_table",
                        lambda tg, ctx, epa, extra=None: seen.setdefault("team_ctx", ctx))
    monkeypatch.setattr(pf_mod, "team_week_keys", lambda tg, ctx: pd.DataFrame())
    monkeypatch.setattr(pf_mod, "extra_features", lambda *a, **k: pd.DataFrame())
    src = {"sched": pd.DataFrame({"s": [1]}), "pbp": None, "injuries": None, "depth": None,
           "weekly": None, "snaps": None, "pfr2gsis": {}, "ngs": {}, "weekly_qb": None}

    def fill(ctx, stadiums, sched):
        seen["fill_args"] = (ctx, stadiums, sched)
        return filled

    built = bpf.build_tables(src, ctx_fill=fill)
    assert seen["fill_args"][0] is raw_ctx and seen["fill_args"][1] == {"S": {}}
    assert seen["fill_args"][2] is src["sched"]
    assert seen["feats_ctx"] is filled and seen["team_ctx"] is filled
    assert built["feats"] is filled and built["team"] is filled

    seen.clear()
    bpf.build_tables(src)
    assert seen["feats_ctx"] is raw_ctx and seen["team_ctx"] is raw_ctx


def test_fetch_sources_passes_seasons_to_every_loader(monkeypatch):
    import nfl_data_py

    import sportsmodel.nfl.nflverse as nv
    import sportsmodel.sim.nfl.usage as usage

    calls: list[tuple[str, list[int]]] = []

    def fake_load_release(dataset, seasons, **kw):
        calls.append((dataset, list(seasons)))
        return pd.DataFrame({"season": [s for s in seasons for _ in range(2)],
                             "position": ["QB", "WR"] * len(seasons)})

    def fake_import_by_season(fn, seasons, name, **kw):
        calls.append((name, list(seasons)))
        return pd.DataFrame()

    monkeypatch.setattr(nv, "load_release", fake_load_release)
    monkeypatch.setattr(nv, "import_by_season", fake_import_by_season)
    monkeypatch.setattr(usage, "depth_charts_asof", lambda raw, sched: raw)
    monkeypatch.setattr(usage, "build_pfr_to_gsis", lambda ids: {})
    monkeypatch.setattr(nfl_data_py, "import_ids", lambda: pd.DataFrame())

    src = bpf.fetch_sources([2020, 2021])
    datasets = {d for d, _ in calls}
    assert {"schedules", "weekly", "snaps", "depth", "injuries", "ngs_receiving"} <= datasets
    # pbp is read season by season
    assert ("pbp", [2020]) in calls and ("pbp", [2021]) in calls
    # QB history for the qb_ profiles: 1999 up to the first requested season (the
    # requested seasons' weekly frame is reused, not downloaded twice)
    hist = list(range(1999, 2020))
    assert ("weekly", hist) in calls and ("weekly", [2020, 2021]) in calls
    assert all(s == [2020, 2021] for d, s in calls if d != "pbp" and s != hist)
    wq = src["weekly_qb"]
    assert (wq["position"] == "QB").all() and sorted(wq["season"].unique()) == list(range(1999, 2022))
    assert src["qb_params_mode"] == "gate"

    calls.clear()
    bpf.fetch_sources()
    assert all(s == bpf.SEASONS for d, s in calls if d != "pbp" and s != list(range(1999, bpf.SEASONS[0])))


# ---- mx_/di_/qb_ wiring (build_tables with a stubbed src) ----------------------------------

def _stub_src():
    """A tiny but complete `src` built from the props fixture's raw extra sources
    (4 teams x 2 seasons x 4 weeks): pbp, offense + defense snaps, weekly, weekly_qb."""
    from tests.nfl import fixtures_props as fp
    raw = fp.extra_sources()
    games = fp._games()
    home = games[games["is_home"] == 1]
    sched = pd.DataFrame({"season": home["season"], "week": home["week"], "game_type": "REG",
                          "home_team": home["team"], "away_team": home["opponent"]})
    pbp = raw["pbp"].assign(qb_hit=0.0, wp=0.5, down=1.0, qtr=1.0, yardline_100=50.0,
                            receiver_player_id=None, rusher_player_id=None)
    wq = raw["weekly_qb"]
    stat0 = ["targets", "carries", "receptions", "receiving_yards", "rushing_yards", "receiving_air_yards",
             "receiving_yards_after_catch", "target_share", "air_yards_share", "receiving_tds", "rushing_tds"]
    weekly = wq.assign(**{c: 0.0 for c in stat0})
    qsnaps = pd.DataFrame({"season": wq["season"], "week": wq["week"], "game_type": "REG",
                           "pfr_player_id": "P" + wq["player_id"], "team": wq["recent_team"],
                           "opponent": wq["opponent_team"], "position": "QB", "offense_snaps": 60.0,
                           "offense_pct": 1.0, "defense_snaps": 0.0, "defense_pct": 0.0})
    snaps = pd.concat([raw["snaps"].assign(offense_pct=0.0), qsnaps], ignore_index=True)
    pfr2gsis = {p: p[1:] for p in snaps["pfr_player_id"].unique()}
    return {"sched": sched, "pbp": pbp, "weekly": weekly, "snaps": snaps, "depth": fp._depth(games),
            "injuries": fp._injuries(), "ngs": {}, "pfr2gsis": pfr2gsis, "weekly_qb": wq}


def _stub_context(monkeypatch):
    import sportsmodel.nfl.context as ctx_mod
    from tests.nfl import fixtures_props as fp
    monkeypatch.setattr(ctx_mod, "load_stadiums", lambda: {})
    monkeypatch.setattr(ctx_mod, "team_game_context", lambda sched, st: fp._games())


def test_build_tables_adds_mx_di_qb_and_qb1_override_moves_only_that_team_week(monkeypatch):
    _stub_context(monkeypatch)
    built = bpf.build_tables(_stub_src())
    team, feats = built["team"], built["feats"]
    for t in (team, feats):
        assert {"mx_pass_edge", "mx_op_ypc_allowed_adj", "di_op_vacated_cov", "qb_ypa", "qb_changed"} <= set(t.columns)
    assert "y_team_off_tds" in team.columns and team["y_team_off_tds"].notna().all()
    tt = team.set_index(["season", "week", "team"])
    assert tt.loc[(2024, 3, "BAL"), "di_op_vacated_cov"] > 0          # BUF_CB Out, BAL faces BUF
    assert tt.xs((2024, 3), level=["season", "week"])["qb_ypa"].notna().all()

    over = bpf.build_tables(_stub_src(), qb1_override={(2024, 3, "BAL"): "BAL_QB"})
    ot = over["team"].set_index(["season", "week", "team"])
    diff = ~pd.Series(ot["qb_ypa"].to_numpy() == tt["qb_ypa"].to_numpy(), index=tt.index) & tt["qb_ypa"].notna()
    assert list(tt.index[diff]) == [(2024, 3, "BAL")]


def test_build_tables_new_columns_use_only_the_new_prefixes(monkeypatch):
    """Global constraint: the build adds only mx_/di_/qb_ features (+ the
    y_team_off_tds label) -- the v1 prefixes see exactly v1's columns."""
    import sportsmodel.nfl.player_features as pf_mod
    _stub_context(monkeypatch)
    with_extra = bpf.build_tables(_stub_src())
    monkeypatch.setattr(pf_mod, "extra_features",
                        lambda tw, *a, **k: tw[["season", "week", "team"]].copy())
    without = bpf.build_tables(_stub_src())
    for name in ("feats", "team"):
        added = set(with_extra[name].columns) - set(without[name].columns)
        assert added and all(c.startswith(("mx_", "di_", "qb_")) for c in added), name
        assert set(without[name].columns) <= set(with_extra[name].columns)


def test_qb_params_gate_by_default_serving_on_request_and_missing_serving_is_clear(tmp_path, monkeypatch):
    import json

    import pytest

    p = tmp_path / "qb_profile_params.json"
    p.write_text(json.dumps({"gate": {"H": 1.0, "k": 100.0}}))
    assert bpf.qb_params("gate", p) == (1.0, 100.0)
    with pytest.raises(RuntimeError, match="serving.*tune_qb_profile.py --mode serving"):
        bpf.qb_params("serving", p)
    p.write_text(json.dumps({"gate": {"H": 1.0, "k": 100.0}, "serving": {"H": 2.0, "k": 50.0}}))
    assert bpf.qb_params("serving", p) == (2.0, 50.0)
    with pytest.raises(ValueError):
        bpf.qb_params("bogus", p)
    # build_tables reads the block named by src["qb_params_mode"]
    seen = {}
    import sportsmodel.nfl.player_features as pf_mod
    _stub_context(monkeypatch)
    monkeypatch.setattr(bpf, "QB_PARAMS_PATH", p)

    def spy(tw, *a, **k):
        seen["Hk"] = a[6:8]
        return tw[["season", "week", "team"]].copy()
    monkeypatch.setattr(pf_mod, "extra_features", spy)
    bpf.build_tables({**_stub_src(), "qb_params_mode": "serving"})
    assert seen["Hk"] == (2.0, 50.0)
    bpf.build_tables(_stub_src())
    assert seen["Hk"] == (1.0, 100.0)
    # an explicit src["qb_params"] (live serving: the served artifacts' own block) wins
    bpf.build_tables({**_stub_src(), "qb_params_mode": "serving", "qb_params": (1.5, 200.0)})
    assert seen["Hk"] == (1.5, 200.0)


# ---- Fix round 1 (Ruling S1): --qb-params flag + the tables' build record ----------------

def _params_file(tmp_path, serving=True):
    import json
    p = tmp_path / "qb_profile_params.json"
    doc = {"gate": {"H": 1.0, "k": 100.0}}
    if serving:
        doc["serving"] = {"H": 1.5, "k": 200.0}
    p.write_text(json.dumps(doc))
    return p


def test_qb_params_flag_defaults_to_gate_and_accepts_serving():
    import pytest

    assert bpf.parse_args([]).qb_params == "gate"
    assert bpf.parse_args(["--qb-params", "serving"]).qb_params == "serving"
    with pytest.raises(SystemExit):
        bpf.parse_args(["--qb-params", "bogus"])


def test_main_sets_the_mode_and_refuses_a_missing_block_before_fetching(tmp_path, monkeypatch):
    import pytest

    seen = {}
    monkeypatch.setattr(bpf, "QB_PARAMS_PATH", _params_file(tmp_path))
    monkeypatch.setattr(bpf, "fetch_sources", lambda: {"qb_params_mode": "gate", "x": 1})
    monkeypatch.setattr(bpf, "build_and_write", lambda src, t0, tf: seen.setdefault("modes", []).append(
        src["qb_params_mode"]))
    bpf.main(["--qb-params", "serving"])
    bpf.main([])
    assert seen["modes"] == ["serving", "gate"]
    monkeypatch.setattr(bpf, "QB_PARAMS_PATH", _params_file(tmp_path, serving=False))
    monkeypatch.setattr(bpf, "fetch_sources", lambda: (_ for _ in ()).throw(AssertionError("fetched")))
    with pytest.raises(RuntimeError, match="serving"):
        bpf.main(["--qb-params", "serving"])


def test_resolved_qb_params_and_build_record_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(bpf, "QB_PARAMS_PATH", _params_file(tmp_path))
    assert bpf.resolved_qb_params({}) == {"mode": "gate", "H": 1.0, "k": 100.0}
    assert bpf.resolved_qb_params({"qb_params_mode": "serving"}) == {"mode": "serving", "H": 1.5, "k": 200.0}
    assert bpf.resolved_qb_params({"qb_params_mode": "serving", "qb_params": (2, 50)}) == {
        "mode": "explicit", "H": 2.0, "k": 50.0}
    assert bpf.read_build_info(tmp_path) is None
    bpf.write_build_info({"mode": "serving", "H": 1.5, "k": 200.0}, tmp_path)
    got = bpf.read_build_info(tmp_path)
    assert got["qb_params"] == {"mode": "serving", "H": 1.5, "k": 200.0} and "created_at" in got
    assert bpf.build_info_path(tmp_path) == tmp_path / "feature_build.json"


def test_build_and_write_records_the_qb_params_next_to_the_parquets(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(bpf, "QB_PARAMS_PATH", _params_file(tmp_path))
    feats = pd.DataFrame({"season": [2024], "week": [1], "player_id": ["a"], "y_targets": [1.0],
                          "position": ["WR"], "p_depth_rank": [1.0]})
    team = pd.DataFrame({"season": [2024], "week": [1], "team": ["KC"], "y_team_pass_att": [30.0],
                         "mx_x": [0.1], "di_x": [0.0], "qb_x": [7.0]})
    monkeypatch.setattr(bpf, "build_tables", lambda src: {"feats": feats, "team": team, "pg": feats,
                                                          "tg": team, "stubs": feats.iloc[:0]})
    monkeypatch.setattr(bpf, "dropped_snap_mappings", lambda snaps, m: {})
    monkeypatch.setattr(bpf, "chart_coverage", lambda d, s: {"exact": 0, "fallback": 0, "none": 0})
    monkeypatch.setattr(bpf, "depth_rank_distribution", lambda f, pos: pd.DataFrame())
    src = {"sched": None, "depth": None, "snaps": None, "pfr2gsis": {}, "qb_params_mode": "serving"}
    out = tmp_path / "tables"
    bpf.build_and_write(src, 0.0, 0.0, out_dir=out)
    assert (out / "player_week_features.parquet").is_file()
    assert bpf.read_build_info(out)["qb_params"] == {"mode": "serving", "H": 1.5, "k": 200.0}
    # a build that fails after the old record existed leaves NO record behind
    monkeypatch.setattr(bpf, "build_tables", lambda src: {"feats": feats, "team": team, "pg": feats,
                                                          "tg": team, "stubs": feats.iloc[:0]})
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        bpf.build_and_write({**src, "qb_params_mode": "gate"}, 0.0, 0.0, out_dir=out)
    assert bpf.read_build_info(out) is None


# ---- final review: v1-safe live build (I3) + complete QB history (M1) ----------------------

def _fake_fetch_io(monkeypatch, weekly_missing=()):
    """Fake every loader fetch_sources uses; `weekly_missing`: seasons the
    weekly release lacks (a failed / 404 download that load_release skips)."""
    import nfl_data_py

    import sportsmodel.nfl.nflverse as nv
    import sportsmodel.sim.nfl.usage as usage

    calls: list[tuple[str, list[int]]] = []

    def fake_load_release(dataset, seasons, **kw):
        calls.append((dataset, list(seasons)))
        got = [s for s in seasons if not (dataset == "weekly" and s in weekly_missing)]
        return pd.DataFrame({"season": [s for s in got for _ in range(2)],
                             "position": ["QB", "WR"] * len(got)})

    monkeypatch.setattr(nv, "load_release", fake_load_release)
    monkeypatch.setattr(nv, "import_by_season", lambda fn, seasons, name, **kw: pd.DataFrame())
    monkeypatch.setattr(usage, "depth_charts_asof", lambda raw, sched: raw)
    monkeypatch.setattr(usage, "build_pfr_to_gsis", lambda ids: {})
    monkeypatch.setattr(nfl_data_py, "import_ids", lambda: pd.DataFrame())
    return calls


def test_fetch_sources_without_qb_history_downloads_no_history(monkeypatch):
    calls = _fake_fetch_io(monkeypatch)
    src = bpf.fetch_sources([2016, 2017], qb_history=False)
    assert [s for d, s in calls if d == "weekly"] == [[2016, 2017]]      # no 1999-2015 download
    assert sorted(src["weekly_qb"]["season"].unique()) == [2016, 2017]
    assert (src["weekly_qb"]["position"] == "QB").all()
    calls.clear()
    bpf.fetch_sources([2016, 2017])                                      # default: full history
    assert ("weekly", list(range(1999, 2016))) in calls
    assert bpf.qb_history_seasons([2016, 2017]) == list(range(1999, 2016))


def test_fetch_sources_raises_when_a_qb_history_season_is_missing(monkeypatch):
    import pytest

    _fake_fetch_io(monkeypatch, weekly_missing=(2003, 2011))
    with pytest.raises(RuntimeError, match=r"season\(s\) \[2003, 2011\] unavailable"):
        bpf.fetch_sources([2016, 2017])
    # a missing REQUESTED season (e.g. the current one not yet published) is not history
    _fake_fetch_io(monkeypatch, weekly_missing=(2017,))
    assert sorted(bpf.fetch_sources([2016, 2017])["weekly_qb"]["season"].unique()) == \
        list(range(1999, 2017))


def test_extra_feature_columns_are_the_builders_columns(monkeypatch):
    _stub_context(monkeypatch)
    built = bpf.build_tables(_stub_src())
    for name in ("feats", "team"):
        got = [c for c in built[name].columns if c.startswith(("mx_", "di_", "qb_"))]
        assert sorted(got) == sorted(bpf.extra_feature_columns()), name
    assert built["extra_error"] is None


def test_build_tables_extra_failure_is_nan_plus_warning_only_with_fallback(monkeypatch, capsys):
    import numpy as np
    import pytest

    import sportsmodel.nfl.player_features as pf_mod
    _stub_context(monkeypatch)
    good = bpf.build_tables(_stub_src())

    def boom(*a, **k):
        raise KeyError("qb1_id")

    monkeypatch.setattr(pf_mod, "extra_features", boom)
    with pytest.raises(KeyError):                         # default (training, v2 serving): fatal
        bpf.build_tables(_stub_src())
    got = bpf.build_tables(_stub_src(), extra_fallback=True)
    out = capsys.readouterr().out
    assert "::warning::props-ml: mx_/di_/qb_ feature build failed (KeyError: 'qb1_id')" in out
    assert got["extra_error"] == "KeyError: 'qb1_id'"
    for name in ("feats", "team"):
        extra = bpf.extra_feature_columns()
        assert set(extra) <= set(got[name].columns) and got[name][extra].isna().all().all()
        # every other column is exactly the normal build's
        rest = [c for c in good[name].columns if c not in extra]
        pd.testing.assert_frame_equal(got[name][rest], good[name][rest])
    assert np.isnan(got["team"]["qb_ypa"]).all()
