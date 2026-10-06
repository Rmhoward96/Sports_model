"""MLB StatsAPI helpers: schedule parsing (postseason + first-pitch time) and finals (network-free)."""
from sportsmodel.ingest import mlb_statsapi


def _game(state="Final", abstract="Final", home=4, away=3, game_type="D", pk=849825):
    return {
        "gamePk": pk, "officialDate": "2026-10-04", "gameDate": "2026-10-04T20:00:00Z", "gameType": game_type,
        "status": {"abstractGameState": abstract, "detailedState": state},
        "venue": {"id": 32, "name": "American Family Field"},
        "teams": {
            "home": {"team": {"id": 158, "name": "Milwaukee Brewers"}, "score": home,
                     "probablePitcher": {"id": 1, "fullName": "A Pitcher"}},
            "away": {"team": {"id": 135, "name": "San Diego Padres"}, "score": away},
        },
    }


def test_parse_game_carries_first_pitch_and_game_type_and_keeps_postseason():
    rec = mlb_statsapi._parse_game(_game(game_type="D"))
    assert rec["commence_time"] == "2026-10-04T20:00:00Z"
    assert rec["game_type"] == "D"
    assert rec["game_pk"] == 849825 and rec["game_date"] == "2026-10-04"
    assert rec["home_probable_pitcher_name"] == "A Pitcher" and rec["away_probable_pitcher_id"] is None


def test_fetch_schedule_does_not_filter_on_game_type(monkeypatch):
    payload = {"dates": [{"games": [_game(game_type=t, pk=i) for i, t in enumerate("RFDLWSE", start=1)]}]}
    monkeypatch.setattr(mlb_statsapi, "_get", lambda path, params: payload)
    recs = mlb_statsapi.fetch_schedule("2026-10-04")
    assert [r["game_type"] for r in recs] == list("RFDLWSE")


def test_daily_schedule_columns_unaffected_by_new_keys():
    # upsert_daily_schedule writes an explicit column list, so the extra keys are inert.
    import inspect
    from sportsmodel import db
    src = inspect.getsource(db.upsert_daily_schedule)
    assert "commence_time" not in src and "game_type" not in src


def test_parse_final_for_a_played_game():
    assert mlb_statsapi.parse_final(_game()) == {
        "home_score": 4, "away_score": 3, "final": True, "market_spread": None, "market_total": None}
    assert mlb_statsapi.parse_final(_game(state="Game Over", abstract="Final"))["home_score"] == 4
    assert mlb_statsapi.parse_final(_game(state="Completed Early: Rain"))["away_score"] == 3


def test_parse_final_is_none_unless_played_to_completion():
    assert mlb_statsapi.parse_final(_game(state="In Progress", abstract="Live")) is None
    assert mlb_statsapi.parse_final(_game(state="Pre-Game", abstract="Preview")) is None
    # A postponed / cancelled game ALSO reports abstractGameState "Final".
    assert mlb_statsapi.parse_final(_game(state="Postponed", abstract="Final", home=None, away=None)) is None
    assert mlb_statsapi.parse_final(_game(state="Cancelled", abstract="Final")) is None
    assert mlb_statsapi.parse_final(_game(home=None)) is None


def test_fetch_final_picks_the_requested_game(monkeypatch):
    seen = {}

    def fake_get(path, params):
        seen.update(path=path, params=params)
        return {"dates": [{"games": [_game(pk=1, home=9, away=0), _game(pk=849825, home=4, away=3)]}]}

    monkeypatch.setattr(mlb_statsapi, "_get", fake_get)
    final = mlb_statsapi.fetch_final(849825)
    assert (final["home_score"], final["away_score"]) == (4, 3)
    assert seen["path"] == "/schedule" and seen["params"]["gamePk"] == 849825
    monkeypatch.setattr(mlb_statsapi, "_get", lambda path, params: {"dates": []})
    assert mlb_statsapi.fetch_final(849825) is None
