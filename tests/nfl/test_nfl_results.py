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
