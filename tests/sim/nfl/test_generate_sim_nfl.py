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
# The ESPN target-week schedule (espn.parse_schedule rows) for the same games:
# the market lines ML_GAME_LINES=on copies onto game_predictions (generate_nfl's
# source for them).
_ESPN = [
    {"game_pk": 11, "commence_time": "2026-09-27T17:00Z", "home_team": "KC", "away_team": "BAL",
     "home_name": "Kansas City Chiefs", "away_name": "Baltimore Ravens", "status": "STATUS_SCHEDULED",
     "market_spread": -3.5, "market_total": 47.5},
    {"game_pk": 12, "commence_time": "2026-09-27T20:25Z", "home_team": "BUF", "away_team": "MIA",
     "home_name": "Buffalo Bills", "away_name": "Miami Dolphins", "status": "STATUS_SCHEDULED",
     "market_spread": None, "market_total": None},
]
_NOW = gsn.datetime(2026, 9, 27, 12, tzinfo=gsn.timezone.utc)   # main()'s pinned clock (_install_io)
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
        self.usage_include_depth: list = []
        self.load_artifacts_args: list[tuple] = []
        self.slates: list[tuple[list[dict], list[dict]]] = []   # upsert_nfl_sim_slate calls
        self.game_preds: list[list[dict]] = []   # upsert_game_predictions calls
        self.espn_calls = 0


_ML_SPEC_TAG = "ML"


def _install_io(monkeypatch, tmp_path, mode, served="sim-nfl-v1", lines=None, espn=None):
    """Fake every IO dependency of gsn.main(); returns the call recorder.
    `served`: what nfl_sim_serving says (None = table missing; an Exception
    instance = the read raises). `lines`: ML_GAME_LINES (None = unset).
    `espn`: the ESPN target-week schedule `_load_espn_slate` returns (default
    `_ESPN`; an Exception instance = the fetch raises)."""
    rec = _Rec()
    # the fixture slate's Sunday morning: _GAMES/_ESPN kick off later today
    monkeypatch.setattr(gsn, "_utcnow", lambda: _NOW)
    if lines is None:
        monkeypatch.delenv("ML_GAME_LINES", raising=False)
    else:
        monkeypatch.setenv("ML_GAME_LINES", lines)

    def fake_espn():
        rec.espn_calls += 1
        src = _ESPN if espn is None else espn
        if isinstance(src, Exception):
            raise src
        return [dict(e) for e in src]

    monkeypatch.setattr(gsn, "_load_espn_slate", fake_espn)
    monkeypatch.setattr(gsn, "upsert_game_predictions",
                        lambda rows: rec.game_preds.append(list(rows)) or len(rows))

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
    monkeypatch.setattr(gsn, "_load_pass_yds_odds", lambda game_pks: [])
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
    def fake_usage_sources(seasons, include_depth=True):
        rec.usage_include_depth.append(include_depth)
        out = {"ids": pd.DataFrame(), "snaps": pd.DataFrame()}
        return {**out, "depth": pd.DataFrame()} if include_depth else out

    monkeypatch.setattr(gsn, "fetch_usage_sources", fake_usage_sources)
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
                              "team": [p.split("_")[0] for p in _ALL_PLAYERS],
                              "st_questionable": [0.0, np.nan, 1.0, 0.0],
                              "st_vacated_tgt": [0.0, 0.1, 0.0, 0.0],
                              "st_vacated_car": [np.nan] * 4})
    if team is None:
        team = pd.DataFrame({"team": ["KC", "BAL", "BUF", "MIA"], "season": 2026, "week": 3})
    if sched is None:
        sched = pd.DataFrame({"season": [2026, 2026], "week": [3, 3], "game_type": ["REG", "REG"],
                              "home_team": ["KC", "BUF"], "away_team": ["BAL", "MIA"]})
    monkeypatch.setattr(gsn, "_build_ml_tables",
                        lambda upto_season, now, report=None: (feats, team, sched))
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
    """The ml_mode= summary lines WITHOUT the trailing st_nan_* fields (those
    are asserted separately -- see _st_nan)."""
    return [ln.split(" st_nan_")[0] for ln in out.splitlines() if ln.startswith("ml_mode=")]


def _st_nan(out):
    (line,) = [ln for ln in out.splitlines() if ln.startswith("ml_mode=")]
    return dict(kv.split("=", 1) for kv in line.split() if kv.startswith("st_nan_"))


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

    def broken(upto_season, now, report=None):
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

    def must_not_build(upto_season, now, report=None):
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



# =============================================================================
# I5: serve-time injury features
# =============================================================================

def test_summary_line_carries_target_week_st_nan_shares(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    assert _st_nan(capsys.readouterr().out) == {
        "st_nan_questionable": "0.25", "st_nan_vacated_tgt": "0.00", "st_nan_vacated_car": "1.00"}


def test_summary_line_st_nan_na_when_tables_never_built(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    _install_ml(monkeypatch, rec, tmp_path, artifacts=None, config_file=False)
    gsn.main()
    assert _st_nan(capsys.readouterr().out) == {
        "st_nan_questionable": "na", "st_nan_vacated_tgt": "na", "st_nan_vacated_car": "na"}


def test_st_nan_shares_only_the_target_weeks():
    feats = pd.DataFrame({"season": [2026, 2026, 2026, 2025], "week": [3, 3, 2, 3],
                          "st_questionable": [np.nan, 0.0, np.nan, np.nan],
                          "st_vacated_tgt": [0.0, 0.0, np.nan, np.nan]})
    got = gsn._st_nan_shares(feats, 2026, {3})
    assert got == {"st_questionable": 0.5, "st_vacated_tgt": 0.0, "st_vacated_car": None}
    assert gsn._st_nan_shares(feats, 2026, set()) == {
        "st_questionable": None, "st_vacated_tgt": None, "st_vacated_car": None}


_DEPTH = pd.DataFrame({"season": [2026, 2026], "week": [3, 3], "club_code": ["KC", "KC"],
                       "gsis_id": ["00-1", "00-2"], "full_name": ["Travis Kelce", "Xavier Worthy"],
                       "football_name": ["Travis", "Xavier"]})
_LIVE = {"by_team": {"KC": [{"player": "Travis Kelce", "status": "Questionable"},
                            {"player": "Xavier Worthy", "status": "Out"},
                            {"player": "Mystery Man", "status": "Out"}]},
         "target_week": 3}


def test_inject_live_injuries_when_nflverse_has_no_target_week_rows():
    nflv = pd.DataFrame({"season": [2026], "week": [2], "team": ["KC"], "gsis_id": ["00-1"],
                         "report_status": ["Out"]})
    out, msg = gsn._inject_live_injuries(nflv, _DEPTH, _LIVE, 2026)
    wk3 = out[(out["season"] == 2026) & (out["week"] == 3)]
    assert set(zip(wk3["gsis_id"], wk3["report_status"])) == {("00-1", "Questionable"), ("00-2", "Out")}
    assert len(out[out["week"] == 2]) == 1                  # nflverse rows kept
    assert "injected 2 live statuses" in msg and "unmapped 1" in msg and "KC Mystery Man" in msg


def test_inject_live_injuries_none_frame():
    out, msg = gsn._inject_live_injuries(None, _DEPTH, _LIVE, 2026)
    assert len(out) == 2 and "injected 2" in msg


def test_inject_live_injuries_leaves_a_posted_nflverse_week_alone():
    nflv = pd.DataFrame({"season": [2026], "week": [3], "team": ["KC"], "gsis_id": ["00-9"],
                         "report_status": ["Questionable"]})
    out, msg = gsn._inject_live_injuries(nflv, _DEPTH, _LIVE, 2026)
    assert out is nflv and "nflverse report present" in msg
    out, msg = gsn._inject_live_injuries(nflv, _DEPTH, {**_LIVE, "target_week": None}, 2026)
    assert out is nflv and "no target week" in msg
    out, msg = gsn._inject_live_injuries(nflv, _DEPTH, None, 2026)
    assert out is nflv


def test_build_ml_tables_feeds_injected_injuries_to_the_builder(monkeypatch, capsys):
    seen = {}

    class FakeBpf:
        SEASONS = [2016]

        @staticmethod
        def fetch_sources(seasons):
            seen["seasons"] = list(seasons)
            return {"injuries": pd.DataFrame({"season": [2026], "week": [2], "team": ["KC"],
                                              "gsis_id": ["00-1"], "report_status": ["Out"]}),
                    "depth": _DEPTH, "sched": pd.DataFrame({"x": [1]})}

        @staticmethod
        def build_tables(src, ctx_fill=None):
            seen["injuries"] = src["injuries"]
            return {"feats": pd.DataFrame(), "team": pd.DataFrame()}

    monkeypatch.setattr(gsn, "_load_script", lambda name: FakeBpf)
    gsn._build_ml_tables(2026, gsn.datetime(2026, 9, 27, tzinfo=gsn.timezone.utc), report=_LIVE)
    inj = seen["injuries"]
    assert set(inj.loc[inj["week"] == 3, "gsis_id"]) == {"00-1", "00-2"}
    out = capsys.readouterr().out
    assert "props-ML injuries: nflverse has no 2026 week 3 rows: injected 2 live statuses" in out


def test_run_ml_passes_the_live_report_to_the_feature_build(monkeypatch, tmp_path):
    rec = _install_io(monkeypatch, tmp_path, "shadow")
    feats, team = _install_ml(monkeypatch, rec, tmp_path)
    sched = pd.DataFrame({"season": [2026, 2026], "week": [3, 3], "game_type": ["REG", "REG"],
                          "home_team": ["KC", "BUF"], "away_team": ["BAL", "MIA"]})
    got = {}

    def build(upto_season, now, report=None):
        got["report"] = report
        return feats, team, sched

    monkeypatch.setattr(gsn, "_build_ml_tables", build)
    gsn.main()
    assert got["report"]["target_week"] == 3 and "KC" in got["report"]["by_team"]



def test_main_never_downloads_the_old_depth_charts(monkeypatch, tmp_path):
    # the sim reads depth only via load_release("depth") + depth_charts_asof;
    # fetch_usage_sources' old nfl_data_py depth import is skipped
    rec = _install_io(monkeypatch, tmp_path, "off")
    gsn.main()
    assert rec.usage_include_depth == [False]


# =============================================================================
# ML_GAME_LINES (ML-only NFL Task 4): the ML sim writes NFL game_predictions.
# =============================================================================

import json  # noqa: E402

from sportsmodel.nfl import config as nfl_config  # noqa: E402

_gn_p = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "generate_nfl.py"
_gn_spec = importlib.util.spec_from_file_location("generate_nfl_for_parity", _gn_p)
gnfl = importlib.util.module_from_spec(_gn_spec)
_gn_spec.loader.exec_module(gnfl)

_LINES_GAME = {"game_pk": 11, "commence_time": "2026-09-27T17:00:00+00:00",
               "home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
               "game_date": "2026-09-27", "market_spread": -3.5, "market_total": 47.5}
_GP_LINES_WARN = "::warning::ml-game-lines: "


def _game_preds(rec):
    """{game_pk: row} over every upsert_game_predictions call, dists decoded."""
    out = {}
    for rows in rec.game_preds:
        for r in rows:
            out[r["game_pk"]] = {**r, "margin_dist": json.loads(r["margin_dist"]),
                                 "total_dist": json.loads(r["total_dist"])}
    return out


def _lines_warnings(out):
    return [ln for ln in out.splitlines() if ln.startswith(_GP_LINES_WARN)]


def test_game_prediction_row_shape_matches_generate_nfl_build_game_row():
    """Ruling M2: same keys, same dist keys/offset/support/length as a REAL
    generate_nfl.build_game_row output (gameline.build_gameline), pmfs sum to 1."""
    gl_cfg = nfl_config.load_gameline()
    ref = gnfl.build_game_row(
        {"game_pk": 11, "game_date": "2026-09-27", "commence_time": "2026-09-27T17:00Z",
         "home_name": "Kansas City Chiefs", "away_name": "Baltimore Ravens",
         "market_spread": -3.5, "market_total": 47.5},
        {"model_margin": 3.0, "model_total": 45.0, "week": 3}, gl_cfg)
    sims = _sims_with([30, 31, 32, 33], [10, 10, 10, 10], ["KC_p"], pass_yds=300)
    row = gsn.game_prediction_row(_LINES_GAME, sims, gl_cfg)
    assert set(row) == set(ref)
    for key in ("margin_dist", "total_dist"):
        assert set(row[key]) == set(ref[key])
        assert row[key]["kind"] == ref[key]["kind"]
        assert len(row[key]["pmf"]) == len(ref[key]["pmf"])
        assert sum(row[key]["pmf"]) == pytest.approx(1.0)
        assert sum(ref[key]["pmf"]) == pytest.approx(1.0)
        assert all(isinstance(p, float) for p in row[key]["pmf"])
    assert row["margin_dist"]["offset"] == ref["margin_dist"]["offset"] == gl_cfg.offset
    assert len(row["total_dist"]["pmf"]) == gl_cfg.total_max + 1
    assert row["sport"] == "nfl" and row["model_version"] == "nfl-sim-ml-v1"
    json.dumps(row)   # plain JSON-able values (main encodes the dists at the DB boundary)


def test_game_prediction_row_values_come_from_the_sims():
    gl_cfg = nfl_config.load_gameline()
    # margins 0, 3, 4, -7 (one tie); totals 40, 37, 44, 41
    sims = _sims_with([20, 20, 24, 17], [20, 17, 20, 24], ["KC_p"], pass_yds=300)
    row = gsn.game_prediction_row(_LINES_GAME, sims, gl_cfg)
    o = row["margin_dist"]["offset"]
    pmf = row["margin_dist"]["pmf"]
    # the gameline convention: pmf[i] = P(margin == i - offset)
    assert pmf[o + 0] == pytest.approx(0.25) and pmf[o + 3] == pytest.approx(0.25)
    assert pmf[o + 4] == pytest.approx(0.25) and pmf[o - 7] == pytest.approx(0.25)
    assert row["total_dist"]["pmf"][37] == pytest.approx(0.25)
    # home win among sims with a winner: margins +3, +4 win, -7 loses, the
    # 0 (tie) is left out -> 2/3 (not 0.625, which would split the tie)
    assert row["home_win_prob"] == pytest.approx(2 / 3)
    assert row["pred_margin"] == pytest.approx(0.0)
    assert row["pred_total"] == pytest.approx(40.5)
    assert row["pred_home_score"] == pytest.approx(20.25)
    assert row["pred_away_score"] == pytest.approx(20.25)
    assert {k: row[k] for k in ("game_pk", "game_date", "commence_time", "market_spread",
                                "market_total", "home_team_name", "away_team_name")} == {
        "game_pk": 11, "game_date": "2026-09-27", "commence_time": "2026-09-27T17:00:00+00:00",
        "market_spread": -3.5, "market_total": 47.5,
        "home_team_name": "Kansas City Chiefs", "away_team_name": "Baltimore Ravens"}


def test_merge_espn_slate_enriches_db_games_and_adds_upcoming_espn_games():
    now = gsn.datetime(2026, 9, 27, 12, tzinfo=gsn.timezone.utc)
    db_games = [dict(_GAMES[0]), {**_GAMES[1], "commence_time": gsn.datetime(
        2026, 9, 28, 0, 20, tzinfo=gsn.timezone.utc)}]   # the DB hands back datetimes
    espn = [dict(_ESPN[0]),
            {"game_pk": 13, "commence_time": "2026-10-04T17:00Z", "home_team": "KC",
             "away_team": "BUF", "home_name": "Kansas City Chiefs", "away_name": "Buffalo Bills",
             "status": "STATUS_SCHEDULED", "market_spread": 1.5, "market_total": 50.0},
            {"game_pk": 14, "commence_time": "2026-09-27T11:00Z", "home_team": "MIA",
             "away_team": "BAL", "home_name": "Miami Dolphins", "away_name": "Baltimore Ravens",
             "status": "STATUS_IN_PROGRESS", "market_spread": None, "market_total": None}]
    out = gsn.merge_espn_slate(db_games, espn, now)
    by_pk = {g["game_pk"]: g for g in out}
    assert [g["game_pk"] for g in out] == [11, 12, 13]   # 14 already kicked off: dropped
    assert by_pk[11] == {**_GAMES[0], "commence_time": "2026-09-27T17:00Z",   # ESPN's kickoff
                         "game_date": "2026-09-27", "market_spread": -3.5, "market_total": 47.5}
    # SNF 00:20 UTC is the previous US day (generate_nfl's 8h shift); not on ESPN -> no lines
    assert by_pk[12]["game_date"] == "2026-09-27"
    assert by_pk[12]["market_spread"] is None and by_pk[12]["market_total"] is None
    assert by_pk[13] == {"game_pk": 13, "matchup": "Buffalo Bills @ Kansas City Chiefs",
                         "commence_time": "2026-10-04T17:00Z", "home_team": "Kansas City Chiefs",
                         "away_team": "Buffalo Bills", "home_win_prob": None,
                         "game_date": "2026-10-04", "market_spread": 1.5, "market_total": 50.0}


def test_merge_espn_slate_carries_market_lines_from_predictions_current():
    """Ruling (a): ESPN's line wins when present; otherwise (no ESPN row, or
    a null line) each field is carried from predictions_current independently."""
    now = gsn.datetime(2026, 9, 27, 12, tzinfo=gsn.timezone.utc)
    db_games = [{**_GAMES[0], "market_spread": -3.0, "market_total": 46.5},
                {**_GAMES[1], "market_spread": 2.5, "market_total": 44.0},
                {**_GAMES[1], "game_pk": 15, "market_spread": 1.0, "market_total": 40.0}]
    espn = [{**_ESPN[0], "market_spread": -3.5, "market_total": None},    # spread only
            {**_ESPN[1], "market_spread": None, "market_total": None}]    # no line at all
    by_pk = {g["game_pk"]: g for g in gsn.merge_espn_slate(db_games, espn, now)}
    assert (by_pk[11]["market_spread"], by_pk[11]["market_total"]) == (-3.5, 46.5)
    assert (by_pk[12]["market_spread"], by_pk[12]["market_total"]) == (2.5, 44.0)
    assert (by_pk[15]["market_spread"], by_pk[15]["market_total"]) == (1.0, 40.0)   # not on ESPN
    # ESPN down: every served line is kept
    down = {g["game_pk"]: g for g in gsn.merge_espn_slate(db_games, [], now)}
    assert [(g["market_spread"], g["market_total"]) for g in down.values()] == [
        (-3.0, 46.5), (2.5, 44.0), (1.0, 40.0)]


def test_merge_espn_slate_refreshes_kickoff_and_drops_started_or_unscheduled_games():
    """I1: a listed predictions_current game takes ESPN's kickoff (and
    game_date from it); one ESPN no longer reports STATUS_SCHEDULED after
    `now` is dropped, so no post-kickoff re-sim row is written. Games ESPN
    does not list keep their predictions_current kickoff."""
    now = gsn.datetime(2026, 9, 27, 12, tzinfo=gsn.timezone.utc)
    db = [{**_GAMES[0], "game_pk": pk, "commence_time": gsn.datetime(
              2026, 9, 27, 17, tzinfo=gsn.timezone.utc)} for pk in (21, 22, 23, 24, 25, 26)]
    espn_row = {**_ESPN[0], "status": "STATUS_SCHEDULED"}
    espn = [{**espn_row, "game_pk": 21, "commence_time": "2026-09-28T00:20Z"},   # flexed to SNF
            {**espn_row, "game_pk": 22, "commence_time": "2026-09-27T11:00Z"},   # moved up: kicked off
            {**espn_row, "game_pk": 23, "status": "STATUS_IN_PROGRESS"},
            {**espn_row, "game_pk": 24, "status": "STATUS_POSTPONED",
             "commence_time": "2026-09-29T17:00Z"},
            {**espn_row, "game_pk": 25, "status": "STATUS_FINAL"},
            # a new ESPN game that is not scheduled is not added either
            {**espn_row, "game_pk": 31, "status": "STATUS_POSTPONED",
             "commence_time": "2026-10-04T17:00Z"}]
    out = gsn.merge_espn_slate(db, espn, now)
    assert [g["game_pk"] for g in out] == [21, 26]
    moved, unlisted = out
    assert moved["commence_time"] == "2026-09-28T00:20Z"
    assert moved["game_date"] == "2026-09-27"   # SNF: the UTC-8h rule, from ESPN's kickoff
    assert unlisted["commence_time"] == db[5]["commence_time"]
    assert unlisted["game_date"] == "2026-09-27"
    # a moved kickoff also moves game_date (generate_nfl's rule)
    tnf = gsn.merge_espn_slate([db[0]], [{**espn_row, "game_pk": 21,
                                          "commence_time": "2026-10-02T00:15Z"}], now)
    assert tnf[0]["game_date"] == "2026-10-01"


@pytest.mark.parametrize("raw,expected,warns", [
    (None, "off", False), ("off", "off", False), ("on", "on", False), (" ON ", "on", False),
    ("", "off", False), ("yes", "off", True), ("true", "off", True)])
def test_ml_game_lines_parsing(monkeypatch, capsys, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("ML_GAME_LINES", raising=False)
    else:
        monkeypatch.setenv("ML_GAME_LINES", raw)
    assert gsn._ml_game_lines() == expected
    out = capsys.readouterr().out
    if warns:
        assert _lines_warnings(out) == [
            f"::warning::ml-game-lines: unknown ML_GAME_LINES={raw!r} (expected on|off); treated as off"]
    else:
        assert out == ""


def _run(monkeypatch, tmp_path, capsys, mode, served, lines, *, exit_code=None, **ml):
    rec = _install_io(monkeypatch, tmp_path, mode, served=served, lines=lines)
    if mode != "off":
        _install_ml(monkeypatch, rec, tmp_path, **ml)
    if exit_code is None:
        gsn.main()
    else:
        with pytest.raises(SystemExit) as exc:
            gsn.main()
        assert exc.value.code == exit_code
    return rec, capsys.readouterr().out


@pytest.mark.parametrize("mode,served", [("shadow", "nfl-sim-ml-v1"), ("off", "nfl-sim-ml-v1"),
                                         ("shadow", "sim-nfl-v1"), ("off", "sim-nfl-v1")])
def test_main_lines_off_is_todays_behavior(monkeypatch, tmp_path, capsys, mode, served):
    """ML_GAME_LINES=off writes exactly what unset writes (today): no
    game_predictions, no ESPN call, ML-version game rows from the CURRENT sim."""
    runs_ = {}
    for lines in (None, "off"):
        (tmp_path / str(lines)).mkdir()
        rec, out = _run(monkeypatch, tmp_path / str(lines), capsys, mode, served, lines)
        assert rec.game_preds == [] and rec.espn_calls == 0
        assert "ml-game-lines" not in out
        runs_[lines] = (repr(rec.upserts), repr(rec.slates), out)
    assert runs_[None] == runs_["off"]
    if served == "nfl-sim-ml-v1":
        _assert_game_rows_equal_current(rec)


def test_main_lines_invalid_value_warns_and_runs_as_off(monkeypatch, tmp_path, capsys):
    rec, out = _run(monkeypatch, tmp_path, capsys, "shadow", "nfl-sim-ml-v1", "yes")
    assert _lines_warnings(out) == [
        "::warning::ml-game-lines: unknown ML_GAME_LINES='yes' (expected on|off); treated as off"]
    assert rec.game_preds == [] and rec.espn_calls == 0
    _assert_game_rows_equal_current(rec)


def test_main_lines_on_served_ml_writes_ml_sim_game_predictions(monkeypatch, tmp_path, capsys):
    rec, out = _run(monkeypatch, tmp_path, capsys, "shadow", "nfl-sim-ml-v1", "on")
    assert rec.espn_calls == 1
    assert len(rec.game_preds) == 1   # one upsert for the whole slate
    gp = _game_preds(rec)
    assert set(gp) == {11, 12}
    gl_cfg = nfl_config.load_gameline()
    for r in gp.values():
        assert r["model_version"] == "nfl-sim-ml-v1" and r["sport"] == "nfl"
        # ML sims: home 30..33 vs 10
        assert r["pred_margin"] == pytest.approx(21.5) and r["pred_total"] == pytest.approx(41.5)
        assert r["home_win_prob"] == pytest.approx(1.0)
        assert r["margin_dist"]["kind"] == "margin" and r["margin_dist"]["offset"] == gl_cfg.offset
        assert len(r["margin_dist"]["pmf"]) == 2 * gl_cfg.offset + 1
        assert len(r["total_dist"]["pmf"]) == gl_cfg.total_max + 1
        assert r["game_date"] == "2026-09-27"
    assert gp[11]["market_spread"] == -3.5 and gp[11]["market_total"] == 47.5
    assert gp[12]["market_spread"] is None and gp[12]["market_total"] is None
    assert (gp[11]["home_team_name"], gp[11]["away_team_name"]) == ("Kansas City Chiefs", "Baltimore Ravens")
    assert gp[11]["commence_time"] == "2026-09-27T17:00Z"   # ESPN's kickoff (I1)
    # (a) the ML-version nfl_sim game rows now come from the ML sims ...
    ml_rows = {r["game_pk"]: r for r in _written(rec, "sim", "nfl-sim-ml-v1")}
    assert set(ml_rows) == {11, 12}
    assert all(r["sim_margin"] == pytest.approx(21.5) for r in ml_rows.values())
    # ... while the current sim's own rows are untouched
    assert all(r["sim_margin"] == pytest.approx(0.0) for r in _written(rec, "sim", "sim-nfl-v1"))
    # player rows are the same merged slate as before
    assert len(_written(rec, "player", "nfl-sim-ml-v1")) == 4 * 5
    assert "game_lines: wrote 2 game_predictions rows under nfl-sim-ml-v1 (ml_sim=2 current_sim=0)" in out
    assert _lines_warnings(out) == []
    assert _ml_summary_lines(out) == ["ml_mode=shadow ml_games=2 ml_players=20 ml_fallback_games=0 ml_status=ok"]


@pytest.mark.parametrize("mode", ["shadow", "off"])
@pytest.mark.parametrize("served", ["sim-nfl-v1", None])
def test_main_lines_on_but_served_not_ml_writes_no_game_predictions(
        monkeypatch, tmp_path, capsys, mode, served):
    rec, out = _run(monkeypatch, tmp_path, capsys, mode, served, "on",
                    exit_code=None)
    assert rec.game_preds == []
    (warn,) = _lines_warnings(out)
    assert warn == (f"::warning::ml-game-lines: ML_GAME_LINES=on but the site serves {served} "
                    "(not nfl-sim-ml-v1); no game_predictions written")
    # the ML-version nfl_sim rows (shadow, served v1) keep the current sim's game outputs
    if rec.slates:
        _assert_game_rows_equal_current(rec)


def test_main_lines_on_fallback_game_gets_current_sim_game_prediction(monkeypatch, tmp_path, capsys):
    rec, out = _run(monkeypatch, tmp_path, capsys, "live", "nfl-sim-ml-v1", "on",
                    dists_raise={"BUF": RuntimeError("boom")})
    gp = _game_preds(rec)
    assert set(gp) == {11, 12}   # the slate stays complete
    assert gp[11]["pred_margin"] == pytest.approx(21.5)      # ML sims
    assert gp[12]["pred_margin"] == pytest.approx(0.0)       # current sim (fallback)
    assert gp[12]["pred_total"] == pytest.approx(44.0)
    assert gp[12]["home_win_prob"] == pytest.approx(0.5)
    assert gp[12]["model_version"] == "nfl-sim-ml-v1"
    _assert_game_copied_from_current(rec, 12)     # nfl_sim + player rows of the fallback game
    ml11 = [r for r in _written(rec, "sim", "nfl-sim-ml-v1") if r["game_pk"] == 11]
    assert ml11[0]["sim_margin"] == pytest.approx(21.5)
    assert "game_lines: wrote 2 game_predictions rows under nfl-sim-ml-v1 (ml_sim=1 current_sim=1)" in out
    assert _ml_summary_lines(out) == ["ml_mode=live ml_games=1 ml_players=10 ml_fallback_games=1 ml_status=ok"]


def test_main_lines_on_sim_ml_mode_off_served_ml_writes_current_sim_game_predictions(
        monkeypatch, tmp_path, capsys):
    rec, out = _run(monkeypatch, tmp_path, capsys, "off", "nfl-sim-ml-v1", "on")
    assert len(rec.slates) == 1
    _assert_game_copied_from_current(rec, 11)
    gp = _game_preds(rec)
    assert set(gp) == {11, 12}
    assert all(r["pred_margin"] == pytest.approx(0.0) for r in gp.values())
    assert _OFF_SERVED_ML_WARNING in out.splitlines()
    assert "game_lines: wrote 2 game_predictions rows under nfl-sim-ml-v1 (ml_sim=0 current_sim=2)" in out


@pytest.mark.parametrize("mode,code", [("shadow", None), ("live", 1)])
def test_main_lines_on_global_failure_served_ml_writes_current_sim_game_predictions(
        monkeypatch, tmp_path, capsys, mode, code):
    rec, out = _run(monkeypatch, tmp_path, capsys, mode, "nfl-sim-ml-v1", "on",
                    exit_code=code, artifacts=None)
    _assert_game_copied_from_current(rec, 11)
    _assert_game_copied_from_current(rec, 12)
    gp = _game_preds(rec)
    assert set(gp) == {11, 12}
    assert all(r["pred_margin"] == pytest.approx(0.0) for r in gp.values())
    assert "ML: FAILED" in out
    assert "game_lines: wrote 2 game_predictions rows under nfl-sim-ml-v1 (ml_sim=0 current_sim=2)" in out


def test_main_lines_on_ml_slate_write_fails_game_predictions_follow_the_copy(
        monkeypatch, tmp_path, capsys):
    """The ML slate write fails after the ML games ran: the copy slate is
    written, and game_predictions come from the current sim (what nfl_sim now
    holds), never from the unwritten ML sims."""
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on")
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
    gp = _game_preds(rec)
    assert set(gp) == {11, 12} and all(r["pred_margin"] == pytest.approx(0.0) for r in gp.values())
    _assert_game_copied_from_current(rec, 11)


def test_main_lines_on_adds_espn_games_missing_from_predictions_current(monkeypatch, tmp_path, capsys):
    """With generate-nfl disabled nothing else adds a new week's games to
    predictions_current, so ML_GAME_LINES=on also takes the slate from ESPN's
    target week (generate_nfl's source)."""
    espn = [*_ESPN, {"game_pk": 13, "commence_time": "2099-10-04T17:00Z", "home_team": "KC",
                     "away_team": "BUF", "home_name": "Kansas City Chiefs",
                     "away_name": "Buffalo Bills", "status": "STATUS_SCHEDULED",
                     "market_spread": 1.5, "market_total": 50.0}]
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on", espn=espn)
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    out = capsys.readouterr().out
    cur = {r["game_pk"]: r for r in _written(rec, "sim", "sim-nfl-v1")}
    assert set(cur) == {11, 12, 13}
    assert cur[13]["disagreement"] is None   # no analytic prediction on file for a new game
    assert cur[13]["matchup"] == "Buffalo Bills @ Kansas City Chiefs"
    gp = _game_preds(rec)
    assert set(gp) == {11, 12, 13}
    assert gp[13]["market_spread"] == 1.5 and gp[13]["game_date"] == "2099-10-04"
    assert gp[13]["pred_margin"] == pytest.approx(0.0)   # no REG schedule row -> current sim
    assert "games=3 players=" in out and "mean_disagreement=nan" not in out


def test_main_lines_on_espn_failure_warns_and_keeps_the_db_slate(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on",
                      espn=ConnectionError("espn down"))
    _install_ml(monkeypatch, rec, tmp_path)
    db = [{**_GAMES[0], "market_spread": -3.0, "market_total": 46.5},
          {**_GAMES[1], "market_spread": None, "market_total": None}]
    monkeypatch.setattr(gsn, "_load_upcoming_games", lambda: [dict(g) for g in db])
    gsn.main()
    out = capsys.readouterr().out
    assert _lines_warnings(out) == [
        "::warning::ml-game-lines: ESPN target-week schedule unavailable (ConnectionError: espn down); "
        "slate = predictions_current only, market lines carried from it"]
    gp = _game_preds(rec)
    assert set(gp) == {11, 12}
    # ruling (a): the served lines survive an ESPN outage (they used to be nulled)
    assert (gp[11]["market_spread"], gp[11]["market_total"]) == (-3.0, 46.5)
    assert gp[12]["market_spread"] is None and gp[12]["market_total"] is None
    assert gp[11]["game_date"] == "2026-09-27"
    assert gp[11]["commence_time"] == _GAMES[0]["commence_time"]   # no ESPN row: DB kickoff kept


def test_main_lines_on_drops_a_db_game_espn_reports_started(monkeypatch, tmp_path, capsys):
    """I1 end to end: a predictions_current game ESPN reports in progress is
    not re-simulated, so it gets no post-kickoff game_predictions row."""
    espn = [{**_ESPN[0], "status": "STATUS_IN_PROGRESS"}, dict(_ESPN[1])]
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on", espn=espn)
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    out = capsys.readouterr().out
    assert set(_game_preds(rec)) == {12}
    assert {r["game_pk"] for r in _written(rec, "sim", "sim-nfl-v1")} == {12}
    assert "game_lines: slate 1 games (0 added from ESPN's target week, 1 dropped" in out


def test_main_lines_on_espn_only_game_sim_failure_warns(monkeypatch, tmp_path, capsys):
    """M4: a game only ESPN supplied has no Elo row to fall back on, so its
    failed sim is a `::warning::ml-game-lines:` line (it is off the site)."""
    espn = [*_ESPN, {"game_pk": 13, "commence_time": "2099-10-04T17:00Z", "home_team": "LV",
                     "away_team": "BUF", "home_name": "Las Vegas Raiders",   # not in _XWALK
                     "away_name": "Buffalo Bills", "status": "STATUS_SCHEDULED",
                     "market_spread": 1.5, "market_total": 50.0}]
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on", espn=espn)
    _install_ml(monkeypatch, rec, tmp_path)
    gsn.main()
    out = capsys.readouterr().out
    assert set(_game_preds(rec)) == {11, 12}
    assert _lines_warnings(out) == [
        "::warning::ml-game-lines: skipping game_pk=13 (Buffalo Bills @ Las Vegas Raiders), added "
        "from ESPN's target week: sim failed (KeyError: 'Las Vegas Raiders'); it has no "
        "game_predictions row"]
    assert "\nskipping game_pk=13" not in out


def test_main_lines_on_db_game_sim_failure_keeps_the_plain_skip_line(monkeypatch, tmp_path, capsys):
    """M4 is only for ESPN-only games: a predictions_current game that fails
    still has its served row, so it keeps the plain `skipping` line."""
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on")
    _install_ml(monkeypatch, rec, tmp_path)
    monkeypatch.setattr(gsn, "_load_crosswalk",
                        lambda: {k: v for k, v in _XWALK.items() if k != "Miami Dolphins"})
    gsn.main()
    out = capsys.readouterr().out
    assert "skipping game_pk=12 (Miami Dolphins @ Buffalo Bills): 'Miami Dolphins'" in out
    assert _lines_warnings(out) == []


def test_main_lines_on_game_predictions_write_failure_exits_1(monkeypatch, tmp_path, capsys):
    rec = _install_io(monkeypatch, tmp_path, "shadow", served="nfl-sim-ml-v1", lines="on")
    _install_ml(monkeypatch, rec, tmp_path)

    def boom(rows):
        raise ConnectionError("gp write lost")

    monkeypatch.setattr(gsn, "upsert_game_predictions", boom)
    with pytest.raises(SystemExit) as exc:
        gsn.main()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert len(rec.slates) == 1   # the nfl_sim ML slate is written first
    assert _lines_warnings(out) == [
        "::warning::ml-game-lines: game_predictions write failed (ConnectionError: gp write lost); "
        "the served NFL game lines were not refreshed"]
    assert _ml_summary_lines(out) == ["ml_mode=shadow ml_games=2 ml_players=20 ml_fallback_games=0 ml_status=ok"]
