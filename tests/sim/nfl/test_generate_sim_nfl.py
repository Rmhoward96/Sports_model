"""Tests for the pure assembly seam in generate_sim_nfl.py (the NFL sim-engine
slate runner). main()'s DB/nflverse IO is thin and not unit-tested here --
see the module docstring in scripts/generate_sim_nfl.py."""
import importlib.util
import pathlib

import numpy as np
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


def test_espn_injury_names_categorizes_and_maps_teams():
    crosswalk = {"New York Giants": "NYG", "Denver Broncos": "DEN"}
    espn = [
        {"team": "New York Giants", "player": "Jaxson Dart", "status": "Doubtful"},
        {"team": "New York Giants", "player": "Malik Nabers", "status": "Questionable"},
        {"team": "New York Giants", "player": "Someone Active", "status": "Active"},
        {"team": "Denver Broncos", "player": "Injured Guy", "status": "Injured Reserve"},
        {"team": "Unknown Team", "player": "Nobody", "status": "Out"},  # no crosswalk -> skipped
    ]
    out, q = gsn._espn_injury_names(espn, crosswalk)
    assert out["NYG"] == {"jaxson dart"}          # Doubtful -> OUT
    assert out["DEN"] == {"injured guy"}          # Injured Reserve -> OUT
    assert q["NYG"] == {"malik nabers"}           # Questionable -> down-weight
    assert "Unknown Team" not in out and all(k in ("NYG", "DEN") for k in out)  # unknown skipped
    # Active is neither dropped nor down-weighted.
    assert not any("someone active" in s for s in out.get("NYG", set()) | q.get("NYG", set()))


def test_merge_espn_injuries_overrides_stale_nflverse_status():
    # nflverse's newest report is LAST week's mid-week (Tue): a QB listed Out
    # last week but Active now must not stay benched; ESPN is fresher per player.
    crosswalk = {"Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL"}
    out = {"ATL": {"michael penix jr."}, "BAL": {"zay flowers", "stale only"}}
    q = {"ATL": set(), "BAL": set()}
    espn = [
        {"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Active"},
        {"team": "Baltimore Ravens", "player": "Zay Flowers", "status": "Questionable"},
        {"team": "Baltimore Ravens", "player": "New Injury", "status": "Out"},
    ]
    gsn._merge_espn_injuries(out, q, espn, crosswalk)
    assert "michael penix jr." not in out["ATL"] | q["ATL"]   # Active -> cleared
    assert "zay flowers" not in out["BAL"] and "zay flowers" in q["BAL"]  # Doubtful -> Questionable
    assert "new injury" in out["BAL"]                         # ESPN-only Out -> added
    assert "stale only" in out["BAL"]                         # nflverse-only entry kept
