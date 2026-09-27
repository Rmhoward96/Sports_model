"""Tests for generate_sim_nfl.py (the NFL sim-engine slate runner): the pure
assembly seam, and main()'s SIM_ML_MODE off/shadow/live behavior with every
IO dependency (DB upserts, nflverse, injury report, feature build, props-ML
artifacts) monkeypatched -- no network, no DB."""
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "generate_sim_nfl.py"
_spec = importlib.util.spec_from_file_location("generate_sim_nfl", _p)
gsn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gsn)

from sportsmodel.sim.nfl.spec import NflGameSims, NflGameSpec, PlayerInput, TeamRates


def _team_rates():
    return TeamRates(
        drive_outcomes={"td": 0.25, "fg": 0.15, "punt": 0.4, "turnover": 0.1, "downs": 0.05, "end": 0.05},
        pass_rate=0.58,
        drives_per_game=11.0,
        rz_td_rate=0.6,
    )


def _player(pid, name, pos="WR"):
    return PlayerInput(
        player_id=pid, name=name, pos=pos,
        target_share=0.5, carry_share=0.0, ypt=8.0, ypc=0.0, ypr=12.3,
        catch_rate=0.65, td_share=0.5,
    )


def _spec(home_team="KC", away_team="BAL"):
    return NflGameSpec(
        home_team=home_team,
        away_team=away_team,
        home=_team_rates(),
        away=_team_rates(),
        home_players=[_player("qb_home", "Home QB", "QB"), _player("wr_home", "Home WR", "WR")],
        away_players=[_player("qb_away", "Away QB", "QB")],
    )


def _sims(home_score, away_score, player_ids):
    n = len(home_score)
    player_stats = {
        pid: {
            "pass_yds": np.array([200] * n),
            "rush_yds": np.array([10] * n),
            "rec_yds": np.array([50] * n),
            "receptions": np.array([4] * n),
            "td": np.array([0, 1] * (n // 2) + [0] * (n % 2)),
        }
        for pid in player_ids
    }
    return NflGameSims(
        home_score=np.array(home_score),
        away_score=np.array(away_score),
        player_stats=player_stats,
    )


def _game(game_pk=1, home_team="Kansas City Chiefs", away_team="Baltimore Ravens"):
    return {
        "game_pk": game_pk,
        "matchup": f"{away_team} @ {home_team}",
        "commence_time": "2026-09-21T17:00:00+00:00",
        "home_team": home_team,
        "away_team": away_team,
    }


def test_assemble_sim_rows_returns_one_sim_row_per_game():
    games = [_game()]
    sims = _sims([24, 27, 20, 17], [17, 20, 24, 27], ["qb_home", "wr_home", "qb_away"])
    sim_rows, player_rows = gsn.assemble_sim_rows(
        games, {1: sims}, {1: _spec()}, {1: 0.5}
    )
    assert len(sim_rows) == 1
    row = sim_rows[0]
    assert row["game_pk"] == 1
    assert row["model_version"] == "sim-nfl-v1"
    assert row["matchup"] == "Baltimore Ravens @ Kansas City Chiefs"
    assert row["commence_time"] == "2026-09-21T17:00:00+00:00"


def test_assemble_sim_rows_carries_game_distributions():
    games = [_game()]
    sims = _sims([24, 27, 20, 17], [17, 20, 24, 27], ["qb_home", "wr_home", "qb_away"])
    sim_rows, _ = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.5})
    row = sim_rows[0]
    assert row["margin_dist"]["kind"] == "margin"
    assert "offset" in row["margin_dist"] and len(row["margin_dist"]["pmf"]) > 0
    assert row["total_dist"]["kind"] == "pmf" and len(row["total_dist"]["pmf"]) > 0
    assert row["away_score_dist"]["kind"] == "pmf" and len(row["away_score_dist"]["pmf"]) > 0
    assert row["home_score_dist"]["kind"] == "pmf" and len(row["home_score_dist"]["pmf"]) > 0
    # margin pmf sums to 1 (it is a probability mass function)
    assert sum(row["margin_dist"]["pmf"]) == pytest.approx(1.0)


def test_assemble_sim_rows_sim_fields_match_pred_scores():
    games = [_game()]
    sims = _sims([24, 27, 20, 17], [17, 20, 24, 27], ["qb_home", "wr_home", "qb_away"])
    sim_rows, _ = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.5})
    row = sim_rows[0]
    home_mean = float(np.mean([24, 27, 20, 17]))
    away_mean = float(np.mean([17, 20, 24, 27]))
    assert row["sim_home_win_prob"] == pytest.approx(0.5)  # 2 home wins, 2 away wins
    assert row["sim_margin"] == pytest.approx(home_mean - away_mean)
    assert row["sim_total"] == pytest.approx(home_mean + away_mean)


def test_assemble_sim_rows_disagreement_matches_formula():
    games = [_game()]
    # All 4 sims have the home team winning -> sim_home_win_prob == 1.0
    sims = _sims([30, 28, 24, 21], [10, 14, 17, 20], ["qb_home", "wr_home", "qb_away"])
    analytic = 0.65
    sim_rows, _ = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: analytic})
    row = sim_rows[0]
    assert row["sim_home_win_prob"] == pytest.approx(1.0)
    assert row["disagreement"] == pytest.approx(abs(analytic - 1.0))


def test_assemble_sim_rows_player_row_count_is_players_times_markets():
    games = [_game()]
    sims = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away"])
    _, player_rows = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.5})
    # 3 players (2 home + 1 away) x 5 markets (pass_yds, rush_yds, rec_yds,
    # receptions, anytime_td) from nfl_player_prop_dists.
    assert len(player_rows) == 3 * 5


def test_assemble_sim_rows_player_row_fields_and_team_attribution():
    games = [_game()]
    sims = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away"])
    _, player_rows = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.5})

    by_player_market = {(r["player_id"], r["market"]): r for r in player_rows}
    row = by_player_market[("qb_home", "pass_yds")]
    assert row["name"] == "Home QB"
    assert row["pos"] == "QB"
    assert row["team"] == "Kansas City Chiefs"
    assert row["game_pk"] == 1
    assert row["model_version"] == "sim-nfl-v1"
    assert row["mean"] == pytest.approx(200.0)
    assert row["dist"]["kind"] == "pmf"
    assert row["commence_time"] == "2026-09-21T17:00:00+00:00"

    away_row = by_player_market[("qb_away", "rec_yds")]
    assert away_row["team"] == "Baltimore Ravens"

    anytime_td_row = by_player_market[("qb_home", "anytime_td")]
    assert len(anytime_td_row["dist"]["pmf"]) == 2


def test_assemble_sim_rows_multiple_games():
    game1 = _game(game_pk=1, home_team="Kansas City Chiefs", away_team="Baltimore Ravens")
    game2 = _game(game_pk=2, home_team="Buffalo Bills", away_team="Miami Dolphins")
    sims1 = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away"])
    sims2 = _sims([14, 17], [21, 24], ["qb_home2"])
    spec2 = NflGameSpec(
        home_team="BUF", away_team="MIA", home=_team_rates(), away=_team_rates(),
        home_players=[_player("qb_home2", "Bills QB", "QB")], away_players=[],
    )
    sim_rows, player_rows = gsn.assemble_sim_rows(
        [game1, game2],
        {1: sims1, 2: sims2},
        {1: _spec(), 2: spec2},
        {1: 0.5, 2: 0.4},
    )
    assert len(sim_rows) == 2
    assert {r["game_pk"] for r in sim_rows} == {1, 2}
    assert len(player_rows) == 3 * 5 + 1 * 5


def test_assemble_sim_rows_skips_game_missing_from_sims():
    games = [_game(game_pk=1), _game(game_pk=2)]
    sims = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away"])
    sim_rows, player_rows = gsn.assemble_sim_rows(
        games, {1: sims}, {1: _spec(), 2: _spec()}, {1: 0.5, 2: 0.5}
    )
    # game_pk=2 has no entry in sims_by_game -- skipped, not crashed.
    assert len(sim_rows) == 1
    assert sim_rows[0]["game_pk"] == 1


def test_assemble_sim_rows_skips_game_missing_from_analytic():
    games = [_game(game_pk=1)]
    sims = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away"])
    sim_rows, player_rows = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {})
    assert sim_rows == []
    assert player_rows == []


def test_assemble_sim_rows_skips_player_not_in_spec_roster():
    games = [_game()]
    # "ghost" player_id appears in sims but not in the spec's rosters.
    sims = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away", "ghost"])
    _, player_rows = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.5})
    assert all(r["player_id"] != "ghost" for r in player_rows)
    assert len(player_rows) == 3 * 5


def test_assemble_sim_rows_empty_games_list():
    sim_rows, player_rows = gsn.assemble_sim_rows([], {}, {}, {})
    assert sim_rows == []
    assert player_rows == []


def test_assemble_sim_rows_analytic_zero_is_not_treated_as_missing():
    """analytic_by_game.get(game_pk) == 0.0 must NOT be treated as 'missing'
    (a naive falsy check would wrongly skip a legitimate 0.0 win prob)."""
    games = [_game()]
    sims = _sims([10, 14], [24, 27], ["qb_home", "wr_home", "qb_away"])
    sim_rows, _ = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.0})
    assert len(sim_rows) == 1
    assert sim_rows[0]["disagreement"] == pytest.approx(sim_rows[0]["sim_home_win_prob"])


REPORT = {
    "by_team": {
        "ATL": [
            {"player": "Michael Penix Jr.", "position": "QB", "status": "Questionable", "note": None, "source": "espn"},
            {"player": "Samson Ebukam", "position": "DE", "status": "Out", "note": None, "source": "nflverse"},
        ],
        "NYG": [{"player": "Jaxson Dart", "position": "QB", "status": "Doubtful", "note": None, "source": "espn"}],
    },
    "stale": True, "report_week": 2, "target_week": 3, "espn_available": True,
    "conflicts": [{"team": "ATL", "player": "Michael Penix Jr.", "nflverse": "Out", "espn": "Questionable"}],
}


def test_out_and_questionable_names_from_normalized_report():
    # current_report's statuses are normalized Out/Doubtful/Questionable; the
    # sim's lowercase status sets must still sort them into OUT vs down-weight.
    out = gsn._out_names_by_team(REPORT["by_team"])
    q = gsn._questionable_names_by_team(REPORT["by_team"])
    assert out == {"ATL": {"samson ebukam"}, "NYG": {"jaxson dart"}}
    assert q == {"ATL": {"michael penix jr."}, "NYG": set()}


def test_injury_summary_line():
    assert gsn._injury_summary(REPORT) == (
        "injuries: report_week=2 target_week=3 stale=True espn=OK conflicts=1")
    down = dict(REPORT, espn_available=False, conflicts=[], stale=False, target_week=None)
    assert gsn._injury_summary(down) == (
        "injuries: report_week=2 target_week=None stale=False espn=UNAVAILABLE conflicts=0")


def test_sim_uses_shared_target_week_resolver():
    # One resolver (injury_report.resolve_target_week) for sim + desk.
    assert not hasattr(gsn, "_target_week")


# =============================================================================
# main(): SIM_ML_MODE off / shadow / live (all IO monkeypatched -- no network,
# no DB). Tiny synthetic sims; the ML module's functions are faked.
# =============================================================================

from sportsmodel.sim.nfl import ml_serving  # noqa: E402 -- faked per test via monkeypatch

_GAMES = [
    {"game_pk": 11, "matchup": "Baltimore Ravens @ Kansas City Chiefs",
     "commence_time": "2026-09-27T17:00:00+00:00", "home_team": "Kansas City Chiefs",
     "away_team": "Baltimore Ravens", "home_win_prob": 0.55},
    {"game_pk": 12, "matchup": "Miami Dolphins @ Buffalo Bills",
     "commence_time": "2026-09-27T20:25:00+00:00", "home_team": "Buffalo Bills",
     "away_team": "Miami Dolphins", "home_win_prob": 0.6},
]
_XWALK = {"Kansas City Chiefs": "KC", "Baltimore Ravens": "BAL",
          "Buffalo Bills": "BUF", "Miami Dolphins": "MIA"}
# nfl_player_prop_dists markets from _sims_with: pass_yds, rush_yds, rec_yds,
# receptions, anytime_td. rec_yds/anytime_td are ML-served; the rest baseline.
_ML_MARKETS = {"rec_yds": {"source": "ml"}, "anytime_td": {"source": "ml"},
               "pass_yds": {"source": "baseline"}, "rush_yds": {"source": "baseline"},
               "receptions": {"source": "baseline"}}


def _sims_with(home_score, away_score, player_ids, pass_yds):
    n = len(home_score)
    stats = {pid: {"pass_yds": np.array([pass_yds] * n), "rush_yds": np.array([10] * n),
                   "rec_yds": np.array([50] * n), "receptions": np.array([4] * n),
                   "td": np.array([0, 1] * (n // 2))}
             for pid in player_ids}
    return NflGameSims(home_score=np.array(home_score), away_score=np.array(away_score),
                       player_stats=stats)


class _Rec:
    """Records every faked IO call in order."""

    def __init__(self):
        self.upserts: list[tuple[str, list[dict]]] = []
        self.active_kwargs: list[dict] = []
        self.load_release: list[tuple] = []
        self.depth_asof_args: list[tuple] = []
        self.ml_sim_rngs: list[dict] = []
        self.ml_dists_calls = 0
        self.ml_gates: list = []   # the `gate` each ml_player_dists call received
        self.load_artifacts_args: list[tuple] = []
        self.slates: list[tuple[list[dict], list[dict]]] = []   # upsert_nfl_sim_slate calls


_ML_SPEC_TAG = "ML"


def _install_io(monkeypatch, tmp_path, mode, served="sim-nfl-v1"):
    """Fake every IO dependency of gsn.main(); returns the call recorder.
    `served`: what nfl_sim_serving says (None = table missing; an Exception
    instance = the read raises)."""
    rec = _Rec()

    def fake_served():
        if isinstance(served, Exception):
            raise served
        return served

    monkeypatch.setattr(gsn, "served_nfl_sim_version", fake_served)
    if mode is None:
        monkeypatch.delenv("SIM_ML_MODE", raising=False)
    else:
        monkeypatch.setenv("SIM_ML_MODE", mode)
    monkeypatch.setenv("DESK_SIM_N", "4")
    monkeypatch.setattr(gsn, "_load_upcoming_games", lambda: [dict(g) for g in _GAMES])
    monkeypatch.setattr(gsn, "_load_crosswalk", lambda: dict(_XWALK))
    monkeypatch.setattr(gsn, "TEAMS_CROSSWALK_PATH", tmp_path / "missing" / "nfl_teams.json")
    monkeypatch.setattr(gsn, "nfl_season", lambda now: 2026)
    monkeypatch.setattr(gsn, "fetch_nflverse", lambda seasons: {
        "pbp": pd.DataFrame({"season": [2026, 2026], "week": [1, 2]}), "weekly": pd.DataFrame()})
    monkeypatch.setattr(gsn, "team_rates_from_pbp", lambda *a, **k: {})
    monkeypatch.setattr(gsn, "team_defense_rates_from_pbp", lambda *a, **k: {})
    monkeypatch.setattr(gsn, "resolve_target_week", lambda now: 3)
    monkeypatch.setattr(gsn, "current_report", lambda now, wk, xw: {
        "by_team": {"KC": [{"player": "D.J. Moore Jr.", "status": "Out"}]},
        "report_week": 3, "target_week": 3, "stale": False, "espn_available": True, "conflicts": []})
    monkeypatch.setattr(gsn, "fetch_usage_sources", lambda seasons: {
        "ids": pd.DataFrame(), "snaps": pd.DataFrame(), "depth": pd.DataFrame()})
    monkeypatch.setattr(gsn, "build_pfr_to_gsis", lambda ids: {})

    def fake_load_release(dataset, seasons, **kw):
        rec.load_release.append((dataset, list(seasons)))
        return pd.DataFrame({"dataset": [dataset]})

    def fake_depth_asof(raw, schedules):
        rec.depth_asof_args.append((raw["dataset"].iloc[0], schedules["dataset"].iloc[0]))
        return pd.DataFrame({"club_code": ["KC", "BAL", "BUF", "MIA"]})

    monkeypatch.setattr(gsn, "load_release", fake_load_release)
    monkeypatch.setattr(gsn, "depth_charts_asof", fake_depth_asof)
    monkeypatch.setattr(gsn, "abbrev_alignment", lambda *a: {
        "depth_unknown": [], "injuries_unknown": [], "games_unknown": []})

    def fake_active_usage(team, *args, **kwargs):
        rec.active_kwargs.append(kwargs)
        return [_player(f"{team}_p", f"{team} P")], f"{team}_p"

    monkeypatch.setattr(gsn, "active_usage", fake_active_usage)
    monkeypatch.setattr(gsn, "build_spec_from_usage", lambda home, away, *a, **k: NflGameSpec(
        home_team=home, away_team=away, home=_team_rates(), away=_team_rates(),
        home_players=[_player(f"{home}_p", f"{home} P")],
        away_players=[_player(f"{away}_p", f"{away} P")]))

    def fake_simulate(spec, n_sims, rng, **kwargs):
        pids = [p.player_id for p in (*spec.home_players, *spec.away_players)]
        if spec.home_team.startswith(_ML_SPEC_TAG):   # the ML spec
            rec.ml_sim_rngs.append(rng.bit_generator.state)
            return _sims_with([30, 31, 32, 33], [10, 10, 10, 10], pids, pass_yds=300)
        return _sims_with([24, 27, 20, 17], [17, 20, 24, 27], pids, pass_yds=200)

    monkeypatch.setattr(gsn, "simulate_game", fake_simulate)
    monkeypatch.setattr(gsn, "upsert_nfl_sim", lambda rows: rec.upserts.append(("sim", list(rows))))
    monkeypatch.setattr(gsn, "upsert_nfl_player_sim",
                        lambda rows: rec.upserts.append(("player", list(rows))))

    def fake_slate(sim_rows, player_rows):
        rec.slates.append((list(sim_rows), list(player_rows)))
        return len(sim_rows), len(player_rows)

    monkeypatch.setattr(gsn, "upsert_nfl_sim_slate", fake_slate)
    return rec


_ML_MARKET_MAX = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 200, "receptions": 15, "anytime_td": 1}
_ALL_PLAYERS = ["KC_p", "BAL_p", "BUF_p", "MIA_p"]


def _install_ml(monkeypatch, rec, tmp_path, *, artifacts="ok", dists_raise=None, config_file=True,
                sched=None, feats=None, team=None, share_fallback_games=(), market_max=None):
    """Fake the ML branch: model dir, feature-table builder + ml_serving functions.
    `dists_raise`: exception ml_player_dists raises (or {home_team: exc});
    `share_fallback_games`: home abbrevs whose build_ml_spec bumps
    learned.share_fallbacks."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    if config_file:
        (model_dir / "props_ml_config.json").write_text("{}")
    monkeypatch.setattr(gsn, "ML_MODEL_DIR", model_dir)
    if feats is None:
        feats = pd.DataFrame({"player_id": _ALL_PLAYERS, "season": 2026, "week": 3,
                              "team": [p.split("_")[0] for p in _ALL_PLAYERS]})
    if team is None:
        team = pd.DataFrame({"team": ["KC", "BAL", "BUF", "MIA"], "season": 2026, "week": 3})
    if sched is None:
        sched = pd.DataFrame({"season": [2026, 2026], "week": [3, 3], "game_type": ["REG", "REG"],
                              "home_team": ["KC", "BUF"], "away_team": ["BAL", "MIA"]})
    monkeypatch.setattr(gsn, "_build_ml_tables", lambda upto_season, now: (feats, team, sched))
    learned = types.SimpleNamespace(share_fallbacks=0)
    cfg = {"markets": _ML_MARKETS, "market_max": dict(_ML_MARKET_MAX if market_max is None else market_max)}
    arts = None if artifacts is None else types.SimpleNamespace(config=cfg, learned=learned)

    def fake_load(model_dir, player_tbl=None, team_tbl=None):
        rec.load_artifacts_args.append((model_dir, player_tbl, team_tbl))
        return arts

    def fake_build_ml_spec(spec, artifacts, feature_rows, team_rows):
        if spec.home_team in share_fallback_games:
            artifacts.learned.share_fallbacks += 1
        return NflGameSpec(home_team=_ML_SPEC_TAG + spec.home_team, away_team=spec.away_team,
                           home=spec.home, away=spec.away, home_players=spec.home_players,
                           away_players=spec.away_players)

    def fake_ml_dists(spec, sims_ml, feature_rows, team_rows, artifacts, rng, gate=None):
        rec.ml_dists_calls += 1
        rec.ml_gates.append(gate)
        exc = dists_raise.get(spec.home_team[len(_ML_SPEC_TAG):]) if isinstance(dists_raise, dict) else dists_raise
        if exc is not None:
            raise exc
        return {p.player_id: {"rec_yds": {"kind": "pmf", "pmf": [0.0, 1.0], "mean": 99.0},
                              "anytime_td": {"kind": "pmf", "pmf": [0.3, 0.7], "mean": 0.7}}
                for p in (*spec.home_players, *spec.away_players)}

    monkeypatch.setattr(ml_serving, "load_artifacts", fake_load)
    monkeypatch.setattr(ml_serving, "build_ml_spec", fake_build_ml_spec)
    monkeypatch.setattr(ml_serving, "ml_player_dists", fake_ml_dists)
    return feats, team


def _written(rec, kind, version):
    """Rows of `kind` ("sim" | "player") written under `version`, via the two
    upserts or the one-transaction slate write."""
    slate_rows = [r for sim, player in rec.slates for r in (sim if kind == "sim" else player)]
    return [r for rows in [rows for k, rows in rec.upserts if k == kind] + [slate_rows]
            for r in rows if r["model_version"] == version]


def _ml_summary_lines(out):
    return [ln for ln in out.splitlines() if ln.startswith("ml_mode=")]


def test_main_off_never_touches_ml(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, None)   # unset -> default off
    monkeypatch.setitem(sys.modules, "sportsmodel.sim.nfl.ml_serving", None)  # import raises

    def boom(*a, **k):
        raise AssertionError("ML path touched in off mode")

    monkeypatch.setattr(gsn, "_build_ml_tables", boom)
    monkeypatch.setattr(gsn, "_load_script", boom)
    gsn.main()
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert {r["model_version"] for _, rows in rec.upserts for r in rows} == {"sim-nfl-v1"}
    assert "ML:" not in out and _ml_summary_lines(out) == []
    assert "games=2 players=" in out


def test_main_depth_asof_and_name_keys_in_all_modes(monkeypatch, tmp_path):
    rec = _install_io(monkeypatch, tmp_path, "off")
    gsn.main()
    assert ("depth", [2025, 2026]) in rec.load_release
    assert ("schedules", [2025, 2026]) in rec.load_release
    assert rec.depth_asof_args == [("depth", "schedules")]
    assert rec.active_kwargs and all(k.get("match_name_keys") is True for k in rec.active_kwargs)


def test_main_shadow_writes_both_versions(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    feats, team = _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    out = capsys.readouterr().out
    # current sim first (two upserts), then the ML version in ONE slate write
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert len(rec.slates) == 1
    assert {r["game_pk"] for r in _written(rec, "sim", "sim-nfl-v1")} == {11, 12}
    ml_games = _written(rec, "sim", "nfl-sim-ml-v1")
    assert {r["game_pk"] for r in ml_games} == {11, 12}
    # I3: game-level outputs stay on the CURRENT sim (the ML sims -- home
    # 30..33 vs 10 -- only feed player dists): identical to sim-nfl-v1's rows
    assert all(r["sim_margin"] == pytest.approx(0.0) for r in ml_games)
    _assert_game_rows_equal_current(rec)
    ml_players = _written(rec, "player", "nfl-sim-ml-v1")
    assert len(ml_players) == 4 * 5   # 2 games x 2 players x 5 markets: a complete slate
    # feature tables always go to load_artifacts; ml_player_dists once per game
    (_, p_arg, t_arg), = rec.load_artifacts_args
    assert p_arg is feats and t_arg is team
    assert rec.ml_dists_calls == 2
    assert _ml_summary_lines(out) == ["ml_mode=shadow ml_games=2 ml_players=20 ml_fallback_games=0 ml_status=ok"]


def test_main_ml_uses_per_game_seed(monkeypatch, tmp_path):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    bsn = gsn._load_script("backtest_sim_nfl")
    expected = [np.random.default_rng(bsn.game_seed(gsn.SIM_SEED, 2026, 3, h, a)).bit_generator.state
                for h, a in (("KC", "BAL"), ("BUF", "MIA"))]
    assert rec.ml_sim_rngs == expected


def test_main_ml_slate_baseline_markets_come_from_current_sim(monkeypatch, tmp_path):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    ml = {(r["game_pk"], r["player_id"], r["market"]): r for r in _written(rec, "player", "nfl-sim-ml-v1")}
    cur = {(r["game_pk"], r["player_id"], r["market"]): r for r in _written(rec, "player", "sim-nfl-v1")}
    key = (11, "KC_p", "pass_yds")
    # baseline market: the CURRENT sim's dist (200), not the ML sims' unmapped draws (300)
    assert ml[key]["mean"] == pytest.approx(200.0)
    assert ml[key]["dist"] == cur[key]["dist"]
    assert ml[(11, "KC_p", "receptions")]["dist"] == cur[(11, "KC_p", "receptions")]["dist"]
    # ML market: from ml_player_dists
    assert ml[(11, "KC_p", "rec_yds")]["mean"] == pytest.approx(99.0)
    assert ml[(12, "MIA_p", "anytime_td")]["dist"]["pmf"] == [0.3, 0.7]
    # identity from the current spec
    assert ml[(11, "BAL_p", "rec_yds")]["team"] == "Baltimore Ravens"


def test_main_live_missing_artifacts_writes_current_then_exits_1(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live")
    _install_ml(monkeypatch, rec, tmp_path, artifacts=None)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert {r["model_version"] for _, rows in rec.upserts for r in rows} == {"sim-nfl-v1"}
    assert "ML: FAILED" in out
    assert any(ln.startswith("::warning::props-ml: ") for ln in out.splitlines())
    assert _ml_summary_lines(out) == ["ml_mode=live ml_games=0 ml_players=0 ml_fallback_games=0 ml_status=failed"]


def test_main_shadow_missing_artifacts_exits_0(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path, artifacts=None)
    gsn.main()   # no SystemExit
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert "ML: FAILED" in out
    assert _ml_summary_lines(out) == ["ml_mode=shadow ml_games=0 ml_players=0 ml_fallback_games=0 ml_status=failed"]


def _assert_game_rows_equal_current(rec):
    """Every ML-version nfl_sim row equals the current sim's row (I3)."""
    def strip(rows):
        return sorted(tuple((k, repr(v)) for k, v in sorted(r.items()) if k != "model_version")
                      for r in rows)
    ml = _written(rec, "sim", "nfl-sim-ml-v1")
    cur = [r for r in _written(rec, "sim", "sim-nfl-v1") if r["game_pk"] in {m["game_pk"] for m in ml}]
    assert ml and strip(ml) == strip(cur)


def _assert_game_copied_from_current(rec, game_pk):
    """The ML version of `game_pk` is the current sim's rows, re-labelled."""
    def strip(rows):
        return sorted((tuple((k, repr(v)) for k, v in sorted(r.items()) if k != "model_version"))
                      for r in rows if r["game_pk"] == game_pk)
    for kind in ("sim", "player"):
        cur, ml = _written(rec, kind, "sim-nfl-v1"), _written(rec, kind, "nfl-sim-ml-v1")
        assert strip(ml) and strip(ml) == strip(cur)


def _fallback_warnings(out):
    return [ln for ln in out.splitlines()
            if ln.startswith("::warning::props-ml: ") and "served from the current sim" in ln]


def test_main_per_game_exception_falls_back_and_run_stays_ok(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live")
    _install_ml(monkeypatch, rec, tmp_path, dists_raise={"BUF": RuntimeError("boom\nsecond line")})
    gsn.main()   # live, but a per-game failure is not a global failure: no SystemExit
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"] and len(rec.slates) == 1
    assert {r["game_pk"] for r in _written(rec, "sim", "nfl-sim-ml-v1")} == {11, 12}
    _assert_game_copied_from_current(rec, 12)
    ml11 = [r for r in _written(rec, "player", "nfl-sim-ml-v1") if r["game_pk"] == 11]
    assert any(r["market"] == "rec_yds" and r["mean"] == pytest.approx(99.0) for r in ml11)
    assert "ML: FAILED" not in out
    (warn,) = _fallback_warnings(out)
    assert warn.startswith("::warning::props-ml: 1 game(s) served from the current sim: 12: ")
    assert "boom second line" in warn
    # ml_games counts ML-SERVED games only (ml_players = their player rows)
    assert _ml_summary_lines(out) == ["ml_mode=live ml_games=1 ml_players=10 ml_fallback_games=1 ml_status=ok"]


def test_main_postseason_game_falls_back_while_reg_game_gets_ml(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    sched = pd.DataFrame({"season": [2026, 2026], "week": [3, 19], "game_type": ["REG", "WC"],
                          "home_team": ["KC", "BUF"], "away_team": ["BAL", "MIA"]})
    _install_ml(monkeypatch, rec, tmp_path, sched=sched)
    gsn.main()
    out = capsys.readouterr().out
    assert rec.ml_dists_calls == 1   # only the REG game ran the ML pipeline
    _assert_game_copied_from_current(rec, 12)
    ml_sims = {r["game_pk"]: r for r in _written(rec, "sim", "nfl-sim-ml-v1")}
    assert set(ml_sims) == {11, 12}                       # the ML slate covers every game
    assert ml_sims[11]["sim_margin"] == pytest.approx(0.0)    # I3: current sim's game row
    (warn,) = _fallback_warnings(out)
    assert "12: " in warn and "REG" in warn
    assert _ml_summary_lines(out) == [
        "ml_mode=shadow ml_games=1 ml_players=10 ml_fallback_games=1 ml_status=ok"]


def _missing_rows_tables():
    feats = pd.DataFrame({"player_id": ["KC_p", "BAL_p", "BUF_p"], "season": 2026, "week": 3,
                          "team": ["KC", "BAL", "BUF"]})            # MIA_p has no row
    team = pd.DataFrame({"team": ["KC", "BUF", "MIA"], "season": 2026, "week": 3})  # BAL has none
    return feats, team


def test_main_missing_feature_rows_every_reg_game_falls_back_is_global_failure(
        monkeypatch, tmp_path, capsys):
    # I6: every REG game fell back -> a GLOBAL ML failure; served sim-nfl-v1
    # -> nothing is written under the ML version; shadow exits 0.
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    feats, team = _missing_rows_tables()
    _install_ml(monkeypatch, rec, tmp_path, feats=feats, team=team)
    gsn.main()
    out = capsys.readouterr().out
    assert rec.ml_dists_calls == 0
    assert rec.slates == [] and _written(rec, "sim", "nfl-sim-ml-v1") == []
    (failed,) = [ln for ln in out.splitlines() if ln.startswith("ML: FAILED")]
    assert "every REG game fell back" in failed
    assert "11: missing feature rows (team rows: BAL)" in failed
    assert "12: missing feature rows (players: MIA_p)" in failed
    assert _fallback_warnings(out) == []
    assert _ml_summary_lines(out) == [
        "ml_mode=shadow ml_games=0 ml_players=0 ml_fallback_games=0 ml_status=failed"]


def test_main_every_reg_game_falls_back_served_ml_writes_copy_and_live_exits_1(
        monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live", served="nfl-sim-ml-v1")
    feats, team = _missing_rows_tables()
    _install_ml(monkeypatch, rec, tmp_path, feats=feats, team=team)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert len(rec.slates) == 1
    _assert_game_copied_from_current(rec, 11)
    _assert_game_copied_from_current(rec, 12)
    assert "ML: FAILED" in out and "every REG game fell back" in out
    assert _ml_summary_lines(out) == [
        "ml_mode=live ml_games=0 ml_players=20 ml_fallback_games=2 ml_status=failed"]


def test_main_postseason_only_slate_is_not_a_global_failure(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live")
    sched = pd.DataFrame({"season": [2026, 2026], "week": [19, 19], "game_type": ["WC", "WC"],
                          "home_team": ["KC", "BUF"], "away_team": ["BAL", "MIA"]})
    _install_ml(monkeypatch, rec, tmp_path, sched=sched)
    gsn.main()   # no SystemExit
    out = capsys.readouterr().out
    assert rec.ml_dists_calls == 0
    _assert_game_copied_from_current(rec, 11)
    _assert_game_copied_from_current(rec, 12)
    assert "ML: FAILED" not in out
    assert _ml_summary_lines(out) == [
        "ml_mode=live ml_games=0 ml_players=0 ml_fallback_games=2 ml_status=ok"]


def test_main_share_fallback_routes_game_to_current_sim(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path, share_fallback_games=("KC",))
    gsn.main()
    out = capsys.readouterr().out
    assert rec.ml_dists_calls == 1
    _assert_game_copied_from_current(rec, 11)
    (warn,) = _fallback_warnings(out)
    assert "11: share fallback" in warn


def test_main_shadow_global_db_failure_keeps_current_and_exits_0(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path)

    def slate(sim_rows, player_rows):
        raise ConnectionError("db gone\nretry later")

    monkeypatch.setattr(gsn, "upsert_nfl_sim_slate", slate)
    gsn.main()
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert {r["model_version"] for _, rows in rec.upserts for r in rows} == {"sim-nfl-v1"}
    failed = [ln for ln in out.splitlines() if ln.startswith("ML: FAILED")]
    assert len(failed) == 1 and "db gone" in failed[0]
    warn = [ln for ln in out.splitlines() if ln.startswith("::warning::props-ml: ")]
    assert len(warn) == 1 and "retry later" in warn[0]   # one annotation line
    assert _ml_summary_lines(out) == ["ml_mode=shadow ml_games=0 ml_players=0 ml_fallback_games=0 ml_status=failed"]


def test_main_live_market_max_mismatch_is_global_failure(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live")
    _install_ml(monkeypatch, rec, tmp_path, market_max=dict(_ML_MARKET_MAX, rec_yds=250))
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert rec.ml_dists_calls == 0
    assert "market_max mismatch" in out and "rec_yds" in out


def test_check_market_max():
    assert gsn._market_max_mismatches({"markets": {"rec_yds": {"source": "ml"},
                                                   "anytime_td": {"source": "ml"},
                                                   "pass_yds": {"source": "baseline"}},
                                       "market_max": {"rec_yds": 200, "anytime_td": 1, "pass_yds": 999}}) == []
    bad = gsn._market_max_mismatches({"markets": {"rec_yds": {"source": "ml"}, "odd": {"source": "ml"}},
                                      "market_max": {"rec_yds": 150, "odd": 3}})
    assert bad == ["odd: config 3 vs sim None", "rec_yds: config 150 vs sim 200"]


def test_missing_feature_rows_helper():
    spec = NflGameSpec(home_team="KC", away_team="BAL", home=_team_rates(), away=_team_rates(),
                       home_players=[_player("a", "A")], away_players=[_player("b", "B")])
    p = pd.DataFrame({"player_id": ["a"]})
    t = pd.DataFrame({"team": ["KC"]})
    assert gsn._missing_feature_rows(spec, "KC", "BAL", p, t) == "team rows: BAL; players: b"
    assert gsn._missing_feature_rows(spec, "KC", "BAL", pd.DataFrame({"player_id": ["a", "b"]}),
                                     pd.DataFrame({"team": ["KC", "BAL"]})) is None


def test_main_live_exception_in_feature_build_exits_1(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live")
    _install_ml(monkeypatch, rec, tmp_path)

    def broken(upto_season, now):
        raise OSError("nflverse down")

    monkeypatch.setattr(gsn, "_build_ml_tables", broken)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert "nflverse down" in capsys.readouterr().out


def test_ml_mode_parsing(monkeypatch, capsys):
    monkeypatch.delenv("SIM_ML_MODE", raising=False)
    assert gsn._ml_mode() == "off"
    monkeypatch.setenv("SIM_ML_MODE", " Shadow ")
    assert gsn._ml_mode() == "shadow"
    monkeypatch.setenv("SIM_ML_MODE", "live")
    assert gsn._ml_mode() == "live"
    monkeypatch.setenv("SIM_ML_MODE", "on")
    with pytest.raises(ValueError, match="SIM_ML_MODE"):
        gsn._ml_mode()


def test_main_invalid_mode_writes_current_then_exits_1(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "lvie")
    monkeypatch.setitem(sys.modules, "sportsmodel.sim.nfl.ml_serving", None)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert "ML: FAILED" in out and "lvie" in out
    assert _ml_summary_lines(out) == ["ml_mode=invalid ml_games=0 ml_players=0 ml_fallback_games=0 ml_status=failed"]


def test_schedule_week_resolves_reg_game_by_normalized_teams():
    sched = pd.DataFrame({"season": [2026, 2026, 2025], "week": [3, 4, 3],
                          "game_type": ["REG", "REG", "REG"],
                          "home_team": ["LAR", "KC", "LA"], "away_team": ["BAL", "LA", "BAL"]})
    assert gsn._schedule_week(sched, 2026, "LA", "BAL") == 3
    assert gsn._schedule_week(sched, 2026, "KC", "LA") == 4
    with pytest.raises(ValueError):
        gsn._schedule_week(sched, 2026, "BAL", "LA")


def test_merge_ml_dists_prefers_ml_for_ml_markets_else_current():
    cur = {"p1": {"pass_yds": {"mean": 200.0}, "rec_yds": {"mean": 50.0}},
           "p2": {"rec_yds": {"mean": 40.0}}}
    ml = {"p1": {"rec_yds": {"mean": 99.0}}}
    gate = {("p1", "pass_yds"), ("p1", "rec_yds"), ("p2", "rec_yds")}
    merged, n_fallback = gsn.merge_ml_dists(cur, ml, {"rec_yds": "ml", "pass_yds": "baseline"}, gate)
    assert merged == {"p1": {"pass_yds": {"mean": 200.0}, "rec_yds": {"mean": 99.0}},
                      "p2": {"rec_yds": {"mean": 40.0}}}
    assert n_fallback == 1   # p2 rec_yds had no ML dist


def test_assemble_sim_rows_dists_override():
    games = [_game()]
    sims = _sims([24, 27], [17, 20], ["qb_home", "wr_home", "qb_away"])
    dists = {1: {"qb_home": {"rec_yds": {"kind": "pmf", "pmf": [1.0], "mean": 7.0}}}}
    _, player_rows = gsn.assemble_sim_rows(games, {1: sims}, {1: _spec()}, {1: 0.5},
                                           dists_by_game=dists)
    assert [(r["player_id"], r["market"], r["mean"]) for r in player_rows] == [("qb_home", "rec_yds", 7.0)]


def test_main_live_no_config_fails_before_feature_build(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "live")
    _install_ml(monkeypatch, rec, tmp_path, config_file=False)

    def must_not_build(upto_season, now):
        raise AssertionError("feature build ran without artifacts")

    monkeypatch.setattr(gsn, "_build_ml_tables", must_not_build)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert "ML: FAILED RuntimeError: props-ML artifacts missing" in out
    assert rec.load_artifacts_args == []


# =============================================================================
# C1/C2: the SERVED version (nfl_sim_serving) decides what must be written.
# =============================================================================

_SERVING_MISSING = "nfl_sim_serving missing — run db/migration_nfl_sim_serving.sql"
_OFF_SERVED_ML_WARNING = (
    "::warning::props-ml: SIM_ML_MODE=off but the site serves nfl-sim-ml-v1 — served the current "
    "sim under the ML version; UPDATE nfl_sim_serving back to sim-nfl-v1 to roll back")


def _ml_path_must_not_run(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("ML path ran")

    monkeypatch.setattr(gsn, "_build_ml_tables", boom)
    monkeypatch.setattr(gsn, "run_ml", boom)


def test_main_served_missing_off_mode_unchanged(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "off", served=None)
    _ml_path_must_not_run(monkeypatch)
    gsn.main()
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"] and rec.slates == []
    assert "ML:" not in out and "::warning::" not in out and _ml_summary_lines(out) == []


@pytest.mark.parametrize("mode,code", [("shadow", None), ("live", 1)])
def test_main_served_missing_ml_modes_are_global_failures_and_write_nothing(
        monkeypatch, tmp_path, capsys, mode, code):
    rec = _install_io(monkeypatch, tmp_path, mode, served=None)
    _ml_path_must_not_run(monkeypatch)
    if code is None:
        gsn.main()
    else:
        with pytest.raises(SystemExit) as exc:
            gsn.main()
        assert exc.value.code == code
    out = capsys.readouterr().out
    # nothing under nfl-sim-ml-v1: the unfiltered views would serve it live
    assert rec.slates == [] and _written(rec, "player", "nfl-sim-ml-v1") == []
    assert f"ML: FAILED {_SERVING_MISSING}" in out.splitlines()
    assert f"::warning::props-ml: {_SERVING_MISSING}" in out.splitlines()
    assert _ml_summary_lines(out) == [
        f"ml_mode={mode} ml_games=0 ml_players=0 ml_fallback_games=0 ml_status=failed"]


def test_main_served_v1_off_writes_only_current(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "off", served="sim-nfl-v1")
    _ml_path_must_not_run(monkeypatch)
    gsn.main()
    out = capsys.readouterr().out
    assert rec.slates == [] and "::warning::" not in out


def test_main_served_ml_off_writes_copy_slate_with_warning(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "off", served="nfl-sim-ml-v1")
    monkeypatch.setitem(sys.modules, "sportsmodel.sim.nfl.ml_serving", None)  # import raises
    _ml_path_must_not_run(monkeypatch)
    gsn.main()   # exit 0
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"]
    assert len(rec.slates) == 1                       # ONE transaction for the ML-version slate
    _assert_game_copied_from_current(rec, 11)
    _assert_game_copied_from_current(rec, 12)
    assert _OFF_SERVED_ML_WARNING in out.splitlines()
    assert "ML: FAILED" not in out


@pytest.mark.parametrize("served", ["sim-nfl-v1", "nfl-sim-ml-v1"])
def test_main_shadow_success_writes_ml_slate_for_either_served_version(
        monkeypatch, tmp_path, capsys, served):
    rec = _install_io(monkeypatch, tmp_path, "shadow", served=served)
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    out = capsys.readouterr().out
    assert len(rec.slates) == 1
    ml = {(r["game_pk"], r["player_id"], r["market"]): r for r in _written(rec, "player", "nfl-sim-ml-v1")}
    assert ml[(11, "KC_p", "rec_yds")]["mean"] == pytest.approx(99.0)
    assert _ml_summary_lines(out) == [
        "ml_mode=shadow ml_games=2 ml_players=20 ml_fallback_games=0 ml_status=ok"]


@pytest.mark.parametrize("mode,code", [("shadow", None), ("live", 1)])
def test_main_served_ml_global_failure_writes_copy_slate(monkeypatch, tmp_path, capsys, mode, code):
    rec = _install_io(monkeypatch, tmp_path, mode, served="nfl-sim-ml-v1")
    _install_ml(monkeypatch, rec, tmp_path, artifacts=None)   # global failure
    if code is None:
        gsn.main()
    else:
        with pytest.raises(SystemExit) as exc:
            gsn.main()
        assert exc.value.code == code
    out = capsys.readouterr().out
    assert len(rec.slates) == 1
    _assert_game_copied_from_current(rec, 11)
    _assert_game_copied_from_current(rec, 12)
    assert len([ln for ln in out.splitlines() if ln.startswith("ML: FAILED")]) == 1
    assert any(ln.startswith("::warning::props-ml: ") for ln in out.splitlines())
    assert _ml_summary_lines(out) == [
        f"ml_mode={mode} ml_games=0 ml_players=20 ml_fallback_games=2 ml_status=failed"]


@pytest.mark.parametrize("mode", ["shadow", "live"])
def test_main_served_v1_global_failure_writes_nothing_under_ml(monkeypatch, tmp_path, capsys, mode):
    rec = _install_io(monkeypatch, tmp_path, mode, served="sim-nfl-v1")
    _install_ml(monkeypatch, rec, tmp_path, artifacts=None)
    if mode == "live":
        with pytest.raises(SystemExit):
            gsn.main()
    else:
        gsn.main()
    assert rec.slates == []


def test_main_served_ml_ml_write_fails_then_copy_slate_written(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1")
    _install_ml(monkeypatch, rec, tmp_path)
    calls = []

    def slate(sim_rows, player_rows):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("ml write lost")
        rec.slates.append((list(sim_rows), list(player_rows)))
        return len(sim_rows), len(player_rows)

    monkeypatch.setattr(gsn, "upsert_nfl_sim_slate", slate)
    gsn.main()
    out = capsys.readouterr().out
    assert len(calls) == 2
    _assert_game_copied_from_current(rec, 11)
    _assert_game_copied_from_current(rec, 12)
    assert "ml write lost" in out
    assert _ml_summary_lines(out)[0].endswith("ml_status=failed")


def test_main_served_ml_invalid_mode_writes_copy_then_exits_1(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "lvie", served="nfl-sim-ml-v1")
    monkeypatch.setitem(sys.modules, "sportsmodel.sim.nfl.ml_serving", None)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    assert len(rec.slates) == 1
    _assert_game_copied_from_current(rec, 11)


def test_main_served_read_error_shadow_fails_without_writing(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow", served=ConnectionError("pg down"))
    _ml_path_must_not_run(monkeypatch)
    gsn.main()
    out = capsys.readouterr().out
    assert rec.slates == []
    assert "ML: FAILED could not read nfl_sim_serving (ConnectionError: pg down)" in out


def test_main_served_read_error_off_warns_and_exits_1(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "off", served=ConnectionError("pg down"))
    _ml_path_must_not_run(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert [k for k, _ in rec.upserts] == ["sim", "player"] and rec.slates == []
    assert any(ln.startswith("::warning::props-ml: could not read nfl_sim_serving") for ln in out.splitlines())



# =============================================================================
# I4: ML dists only inside the gated population (current sim's dist means)
# =============================================================================

def test_merge_ml_dists_only_gated_player_markets_take_ml():
    cur = {"star": {"rec_yds": {"mean": 60.0}, "anytime_td": {"mean": 0.4}},
           "fringe": {"rec_yds": {"mean": 3.0}, "anytime_td": {"mean": 0.02}}}
    ml = {"star": {"rec_yds": {"mean": 70.0}, "anytime_td": {"mean": 0.5}},
          "fringe": {"rec_yds": {"mean": 9.0}, "anytime_td": {"mean": 0.1}},
          "ghost": {"rec_yds": {"mean": 1.0}}}                  # absent from the current sim
    gate = {("star", "rec_yds"), ("star", "anytime_td")}
    merged, n_fallback = gsn.merge_ml_dists(cur, ml, {"rec_yds": "ml", "anytime_td": "ml"}, gate)
    assert merged == {"star": {"rec_yds": {"mean": 70.0}, "anytime_td": {"mean": 0.5}},
                      "fringe": {"rec_yds": {"mean": 3.0}, "anytime_td": {"mean": 0.02}}}
    assert n_fallback == 0      # an ungated player-market is not a fallback


def test_main_ungated_player_keeps_current_dists_and_gate_reaches_ml_serving(
        monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path)
    real = gsn.simulate_game

    def low_bal(spec, n_sims, rng, **kw):
        sims = real(spec, n_sims, rng, **kw)
        if not spec.home_team.startswith(_ML_SPEC_TAG) and "BAL_p" in sims.player_stats:
            sims.player_stats["BAL_p"]["rec_yds"] = np.array([5] * n_sims)   # below the gate
        return sims

    monkeypatch.setattr(gsn, "simulate_game", low_bal)
    gsn.main()
    ml = {(r["game_pk"], r["player_id"], r["market"]): r for r in _written(rec, "player", "nfl-sim-ml-v1")}
    cur = {(r["game_pk"], r["player_id"], r["market"]): r for r in _written(rec, "player", "sim-nfl-v1")}
    # gated: ML dists
    assert ml[(11, "KC_p", "rec_yds")]["mean"] == pytest.approx(99.0)
    assert ml[(11, "KC_p", "anytime_td")]["dist"]["pmf"] == [0.3, 0.7]
    # ungated (rec_yds 5 < 25; anytime_td has no gated parent): the CURRENT sim's dists
    for m in ("rec_yds", "anytime_td"):
        assert ml[(11, "BAL_p", m)]["dist"] == cur[(11, "BAL_p", m)]["dist"], m
    gate11 = rec.ml_gates[0]
    assert ("KC_p", "rec_yds") in gate11 and ("KC_p", "anytime_td") in gate11
    assert ("BAL_p", "rec_yds") not in gate11 and ("BAL_p", "anytime_td") not in gate11
    assert ("KC_p", "pass_yds") in gate11        # 200 >= 150 (the gate is market-agnostic of source)
