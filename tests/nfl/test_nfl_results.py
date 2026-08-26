import json, pathlib
from sportsmodel.ingest.nfl_results import parse_results

FIX = json.loads((pathlib.Path(__file__).parent.parent
                  / "fixtures/nfl/espn_summary.json").read_text())


def test_parse_results_scores_and_player_actuals():
    res = parse_results(FIX)
    assert isinstance(res["home_score"], int) and isinstance(res["away_score"], int)
    assert res["home_score"] == 27 and res["away_score"] == 20
    assert res["final"] is True
    # per-player actuals keyed by player_id -> {market: value}
    pid = next(iter(res["players"]))
    assert set(res["players"][pid]).issuperset(
        {"pass_yds", "reception_yds", "rush_yds", "receptions", "pass_tds", "anytime_td"})


def test_parse_results_qb_stat_line():
    res = parse_results(FIX)
    mahomes = res["players"]["3139477"]
    assert mahomes["pass_yds"] == 291
    assert mahomes["pass_tds"] == 2


def test_parse_results_combines_rushing_and_receiving_for_same_player():
    res = parse_results(FIX)
    pacheco = res["players"]["4241389"]
    assert pacheco["rush_yds"] == 93
    assert pacheco["reception_yds"] == 11
    assert pacheco["receptions"] == 2
    assert pacheco["rush_reception_yds"] == 93 + 11
    assert pacheco["anytime_td"] == 1  # 1 rush TD + 0 rec TD


def test_parse_results_anytime_td_zero_when_no_touchdowns():
    res = parse_results(FIX)
    flowers = res["players"]["4243167"]
    assert flowers["anytime_td"] == 0


def test_parse_results_not_final_gates_false():
    not_final = json.loads(json.dumps(FIX))
    not_final["header"]["competitions"][0]["status"]["type"]["name"] = "STATUS_IN_PROGRESS"
    res = parse_results(not_final)
    assert res["final"] is False


def test_parse_results_remaps_to_gsis_ids_via_id_map():
    # Mahomes + Pacheco crosswalked to gsis ids; Kelce and BAL's Lamar Jackson /
    # Zay Flowers deliberately left OUT of the map to exercise the fallback.
    id_map = {"3139477": "00-0033873", "4241389": "00-0036389"}
    res = parse_results(FIX, id_map=id_map)
    # mapped athletes now keyed by gsis id, with stats intact
    assert res["players"]["00-0033873"]["pass_yds"] == 291
    assert res["players"]["00-0036389"]["rush_yds"] == 93
    assert res["players"]["00-0036389"]["reception_yds"] == 11
    # mapped ids replace the raw ESPN ids -- they're gone from the dict
    assert "3139477" not in res["players"]
    assert "4241389" not in res["players"]
    # unmapped athletes fall back to keeping their raw ESPN id (not dropped)
    assert res["players"]["15847"]["reception_yds"] == 97
    assert res["players"]["3916387"]["pass_tds"] == 1
    assert res["players"]["4243167"]["anytime_td"] == 0


def test_parse_results_default_id_map_none_keeps_espn_ids():
    res = parse_results(FIX)
    assert "3139477" in res["players"]  # untouched when id_map is not given
