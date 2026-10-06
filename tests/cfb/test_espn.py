import copy
import json
import pathlib

from sportsmodel.cfb import espn
from sportsmodel.cfb.espn import parse_schedule, parse_final

FIX = json.loads((pathlib.Path(__file__).parent.parent
                  / "fixtures/cfb/espn_scoreboard.json").read_text())


def test_parse_schedule_normalizes_fbs_and_fcs_teams():
    games = parse_schedule(FIX)
    assert len(games) == 2

    g0 = games[0]  # Georgia @ Kentucky -- FBS vs FBS
    assert g0["game_pk"] == 401628354 and isinstance(g0["game_pk"], int)
    assert g0["home_team"] == "96"   # Kentucky ESPN id, FBS passthrough
    assert g0["away_team"] == "61"   # Georgia ESPN id, FBS passthrough
    assert g0["home_name"] == "Kentucky Wildcats"
    assert g0["away_name"] == "Georgia Bulldogs"
    assert g0["status"] == "STATUS_FINAL"

    g1 = games[1]  # Northern Iowa @ Nebraska -- FBS vs FCS
    assert g1["home_team"] == "158"   # Nebraska ESPN id, FBS passthrough
    assert g1["away_team"] == "FCS"   # Northern Iowa collapses to FCS anchor
    assert g1["away_name"] == "Northern Iowa Panthers"


def test_parse_schedule_types_scores_and_ids():
    g0 = parse_schedule(FIX)[0]
    assert g0["home_score"] == 12 and isinstance(g0["home_score"], int)
    assert g0["away_score"] == 13 and isinstance(g0["away_score"], int)


def test_parse_schedule_populates_week_and_season():
    for g in parse_schedule(FIX):
        assert g["week"] == 3
        assert g["season"] == 2024


def test_parse_final_gates_on_status():
    assert parse_final(FIX["events"][0]) == {"home_score": 12, "away_score": 13, "final": True}
    assert parse_final(FIX["events"][1]) == {"home_score": 34, "away_score": 3, "final": True}

    not_final = copy.deepcopy(FIX["events"][0])
    not_final["status"]["type"]["name"] = "STATUS_SCHEDULED"
    assert parse_final(not_final) is None


def _ev(**comp_overrides):
    ev = copy.deepcopy(FIX["events"][0])
    ev["competitions"][0].update(comp_overrides)
    return ev


def test_parse_schedule_context_columns_present_and_default_when_absent():
    # Fixture events carry no neutralSite/conferenceCompetition/conferenceId
    # (neutralSite is False, the rest absent) -> False / False / None.
    g0 = parse_schedule(FIX)[0]
    assert g0["start_date"] == "2024-09-14T23:30Z"
    assert g0["neutral_site"] is False
    assert g0["conference_game"] is False
    assert g0["home_conf"] is None and g0["away_conf"] is None


def test_parse_schedule_neutral_site_conference_game_and_conf_ids():
    ev = _ev(neutralSite=True, conferenceCompetition=True)
    for c in ev["competitions"][0]["competitors"]:
        c["team"]["conferenceId"] = "8" if c["homeAway"] == "home" else "9"
    payload = {**FIX, "events": [ev]}
    g = parse_schedule(payload)[0]
    assert g["neutral_site"] is True
    assert g["conference_game"] is True
    assert g["home_conf"] == "8" and g["away_conf"] == "9"
    assert g["game_pk"] == 401628354 and isinstance(g["game_pk"], int)
    # existing keys unchanged
    assert g["home_team"] == "96" and g["home_score"] == 12 and g["week"] == 3


def test_parse_schedule_conference_game_false_when_flag_false():
    g = parse_schedule({**FIX, "events": [_ev(conferenceCompetition=False)]})[0]
    assert g["conference_game"] is False


# ------------------------------------------------------------- line scores --

def _final_event(pk, home, away, status="STATUS_FINAL", hs=None, as_=None):
    ls = lambda xs: [{"value": float(x), "displayValue": str(x)} for x in xs]  # noqa: E731
    hs, as_ = sum(home) if hs is None else hs, sum(away) if as_ is None else as_
    return {"id": str(pk), "status": {"type": {"name": status}},
            "competitions": [{"competitors": [{"homeAway": "home", "score": str(hs), "linescores": ls(home)},
                                              {"homeAway": "away", "score": str(as_), "linescores": ls(away)}]}]}


def test_parse_line_scores_final_games_with_overtime_total():
    from sportsmodel.cfb.espn import parse_line_scores
    payload = {"events": [_final_event(1, [0, 10, 3, 24], [0, 3, 7, 0]),
                          _final_event(2, [7, 7, 7, 3, 7], [7, 7, 7, 3, 3]),
                          _final_event(3, [7, 7, 7, 3], [7, 7, 7, 3], status="STATUS_IN_PROGRESS"),
                          _final_event(4, [7, 7], [3, 3]),
                          {"id": "5"}]}
    assert parse_line_scores(payload) == {
        1: {"home": [0, 10, 3, 24], "away": [0, 3, 7, 0], "home_ot": 0, "away_ot": 0},
        2: {"home": [7, 7, 7, 3], "away": [7, 7, 7, 3], "home_ot": 7, "away_ot": 3}}
    assert parse_line_scores({}) == {}


def test_parse_line_scores_skips_games_whose_periods_do_not_add_up_to_the_score():
    from sportsmodel.cfb.espn import parse_line_scores
    ok = _final_event(1, [7, 7, 7, 3], [0, 3, 7, 0])
    off_by_one = _final_event(2, [7, 7, 7, 3], [0, 3, 7, 0], hs=25)           # home periods sum to 24, score 25
    missing_ot = _final_event(3, [7, 7, 7, 3, 7], [7, 7, 7, 3, 3], hs=24)     # OT ignored would match 24
    no_score = _final_event(4, [7, 7, 7, 3], [0, 3, 7, 0]); del no_score["competitions"][0]["competitors"][0]["score"]
    bad_ot = _final_event(5, [7, 7, 7, 3, 7], [7, 7, 7, 3, 3])
    bad_ot["competitions"][0]["competitors"][0]["linescores"][4]["value"] = None
    assert set(parse_line_scores({"events": [ok, off_by_one, missing_ot, no_score, bad_ot]})) == {1}


def test_fetch_scoreboard_asks_for_the_fbs_group_and_returns_the_raw_payload(monkeypatch):
    from sportsmodel.cfb import espn
    seen = {}
    monkeypatch.setattr(espn, "_get", lambda path, params=None: seen.update(path=path, params=params) or {"events": []})
    assert espn.fetch_scoreboard(2026, 6, 2) == {"events": []}
    assert seen == {"path": "/scoreboard", "params": {"dates": 2026, "seasontype": 2, "week": 6, "groups": 80}}


_FPI_FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "cfb" / "espn_fpi.json"


def test_parse_fpi_maps_team_id_to_fpi_value_and_skips_items_without_one():
    payload = json.loads(_FPI_FIXTURE.read_text())
    assert espn.parse_fpi(payload) == {"194": 28.837, "61": 28.2}
    assert espn.parse_fpi({}) == {} and espn.parse_fpi(None) == {}


def test_parse_fpi_ignores_non_numeric_values():
    item = {"team": {"$ref": ".../teams/5?lang=en"},
            "predictives": [{"name": "fpi", "value": None}]}
    bad = {"team": {"$ref": ".../teams/6?lang=en"},
           "predictives": [{"name": "fpi", "value": "n/a"}]}
    assert espn.parse_fpi({"items": [item, bad]}) == {}


def test_fetch_fpi_follows_every_page(monkeypatch):
    pages = {
        1: {"pageIndex": 1, "pageCount": 2, "items": [
            {"team": {"$ref": ".../teams/194?x"}, "predictives": [{"name": "fpi", "value": 28.8}]}]},
        2: {"pageIndex": 2, "pageCount": 2, "items": [
            {"team": {"$ref": ".../teams/61?x"}, "predictives": [{"name": "fpi", "value": 28.2}]}]},
    }
    calls = []

    def fake(path, params=None):
        calls.append((path, dict(params)))
        return pages[params["page"]]
    monkeypatch.setattr(espn, "_get_core", fake)
    assert espn.fetch_fpi(2026) == {"194": 28.8, "61": 28.2}
    assert calls == [("/seasons/2026/powerindex", {"limit": 200, "page": 1}),
                     ("/seasons/2026/powerindex", {"limit": 200, "page": 2})]
