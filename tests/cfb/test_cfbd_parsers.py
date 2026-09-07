"""Pure-parser tests for the CFBD adapter. No network; all fixtures are
committed sample JSON under tests/fixtures/cfb/, shaped to match the real
CFBD OpenAPI schema (SP+ ratings, returning production, recruiting team
totals, transfer portal, coaches, games)."""
import json
import pathlib

from sportsmodel.cfb import cfbd

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "cfb"


def _load(name: str):
    return json.loads((FIX / name).read_text())


# ---------------------------------------------------------------- parse_sp --

def test_parse_sp_maps_team_to_rating():
    payload = _load("cfbd_sp.json")
    out = cfbd.parse_sp(payload)
    assert out["Alabama"] == 28.4          # 'rating' field
    assert "Kent State" in out
    assert out["Kent State"] == -18.6


# --------------------------------------------------------- parse_returning --

def test_parse_returning_reports_pct_starters_and_qb_proxy():
    payload = _load("cfbd_returning.json")
    out = cfbd.parse_returning(payload)

    bama = out["Alabama"]
    assert bama["returning_pct"] == 0.71          # percentPPA
    assert bama["returning_starters"] == 0.66     # usage
    assert bama["qb_returning"] is True           # percentPassingPPA 0.62 >= 0.5

    kent = out["Kent State"]
    assert kent["returning_pct"] == 0.40
    assert kent["returning_starters"] == 0.38
    assert kent["qb_returning"] is False          # percentPassingPPA 0.35 < 0.5


# --------------------------------------------------------- parse_recruiting --

def test_parse_recruiting_maps_team_to_points():
    payload = _load("cfbd_recruiting.json")
    out = cfbd.parse_recruiting(payload)
    assert out["Georgia"] == 315.62
    assert out["Kent State"] == 120.05


# -------------------------------------------------------------- parse_portal --

def test_parse_portal_nets_incoming_minus_outgoing():
    payload = _load("cfbd_portal.json")
    out = cfbd.parse_portal(payload)

    # incoming ratings summed for destination, outgoing for origin
    assert out["Ole Miss"]["in"] == 1.50
    assert out["Ole Miss"]["out"] == 0.40
    assert out["Ole Miss"]["net"] == round(out["Ole Miss"]["in"] - out["Ole Miss"]["out"], 4)
    assert out["Ole Miss"]["net"] == 1.10

    assert out["Alabama"]["out"] == 0.95
    assert out["Alabama"]["in"] == 0
    assert out["Alabama"]["net"] == -0.95

    assert out["Miami"]["in"] == 0.40
    assert out["Kent State"]["out"] == 0.55


# ------------------------------------------------------------- parse_coaches --

def test_parse_coaches_flags_first_year_hc():
    payload = _load("cfbd_coaches.json")
    out = cfbd.parse_coaches(payload, season=2024)
    assert out["Washington"] is True       # new HC in 2024 (Fisch's 1st year there)
    assert out["Georgia"] is False         # returning HC (Smart since 2016)


# --------------------------------------------------------------- parse_games --

def test_parse_games_extracts_matchups_for_sos():
    payload = _load("cfbd_games.json")
    out = cfbd.parse_games(payload)
    assert len(out) == 2
    assert out[0] == {"home_team": "Georgia", "away_team": "Clemson", "season": 2024}
    assert out[1] == {"home_team": "Alabama", "away_team": "Western Kentucky", "season": 2024}
