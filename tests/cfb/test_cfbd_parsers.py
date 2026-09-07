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


# ------------------------------------------------- null handling (live data) --
# Real CFBD responses carry nulls the hand-written happy-path fixtures did not
# (a null portal `rating`/`destination`, a null `percentPassingPPA`, an SP+
# "nationalAverages" row with no team, an unscored recruiting class). These
# used to crash the ingest (TypeError on `int += None`); guard each parser.

def test_parse_portal_handles_null_rating_and_destination():
    out = cfbd.parse_portal(_load("cfbd_portal.json"))
    assert None not in out                       # null destination never becomes a key
    assert out["Miami"]["in"] == 0.40            # unrated (null) incoming adds 0, no crash
    assert out["Ole Miss"]["out"] == 0.40        # uncommitted null-rating outbound adds 0


def test_parse_returning_null_passing_ppa_is_neutral_qb():
    out = cfbd.parse_returning(_load("cfbd_returning.json"))
    np = out["New Program"]
    assert np["qb_returning"] is None            # null passing data -> neutral, not a penalty
    assert np["returning_pct"] is None
    assert np["returning_starters"] is None


def test_parse_sp_skips_national_averages_and_null_rating():
    out = cfbd.parse_sp(_load("cfbd_sp.json"))
    assert None not in out                        # the team:null nationalAverages row is dropped
    assert set(out) == {"Alabama", "Kent State"}


def test_parse_recruiting_skips_null_points():
    out = cfbd.parse_recruiting(_load("cfbd_recruiting.json"))
    assert "No Class Yet" not in out
    assert set(out) == {"Georgia", "Kent State"}
