"""Pure-assembly tests for scripts/build_cfb_priors.py's build_priors_rows.

No network: build_priors_rows takes already-parsed per-endpoint dicts (the
shape cfbd.parse_* returns) for one season, plus the prior/current season's
SP+ ratings and schedules needed for strength-of-schedule, and returns plain
per-team-season row dicts. team_name -> team_espn_id mapping and parquet I/O
both live in main(), not here.
"""
import importlib.util
import pathlib

import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_priors.py"
_spec = importlib.util.spec_from_file_location("build_cfb_priors", _p)
bcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bcp)

build_priors_rows = bcp.build_priors_rows

SEASON = 2025
PRIOR = 2024

# -- tiny hand-checked fixture -------------------------------------------
# 2025 (this season) schedule: Ames-Boone, Coralville-Ames
# 2025 SP+: Ames 20.0, Boone 10.0, Coralville -5.0
#   -> Ames' 2025 opponents = Boone, Coralville -> avg = (10.0 + -5.0)/2 = 2.5
#   -> Boone's 2025 opponents = Ames -> avg = 20.0
#   -> Coralville's 2025 opponents = Ames -> avg = 20.0
#
# 2024 (prior season) schedule: Ames-Boone only (Coralville has no recorded
# 2024 game -> its prior_sos/forward_sos_shift should come back None)
# 2024 SP+: Ames 15.0, Boone 5.0, Coralville -10.0
#   -> Ames' 2024 opponents = Boone -> avg = 5.0  == prior_sos(Ames)
#   -> Boone's 2024 opponents = Ames -> avg = 15.0 == prior_sos(Boone)
#
# forward_sos_shift = this-season avg opponent SP+ - prior-season avg opponent SP+
#   Ames:  2.5 - 5.0  = -2.5
#   Boone: 20.0 - 15.0 = 5.0
#   Coralville: prior_sos is None -> forward_sos_shift is None too

PARSED = {
    "sp": {"Ames": 20.0, "Boone": 10.0, "Coralville": -5.0},
    "sp_prior": {"Ames": 15.0, "Boone": 5.0, "Coralville": -10.0},
    "returning": {
        "Ames": {"returning_pct": 0.65, "returning_starters": 0.60, "qb_returning": True},
        "Boone": {"returning_pct": 0.40, "returning_starters": 0.35, "qb_returning": False},
        # Coralville deliberately absent -> None fields
    },
    "recruiting": {"Ames": 250.0, "Boone": 150.0},  # Coralville absent -> None
    "portal": {
        "Ames": {"in": 5.0, "out": 2.0, "net": 3.0},
        "Boone": {"in": 1.0, "out": 1.0, "net": 0.0},
        # Coralville absent -> defaults to net 0.0
    },
    "coaches": {"Ames": True, "Boone": False},  # Coralville absent -> defaults False
    "games": [
        {"home_team": "Ames", "away_team": "Boone", "season": SEASON},
        {"home_team": "Coralville", "away_team": "Ames", "season": SEASON},
    ],
    "games_prior": [
        {"home_team": "Ames", "away_team": "Boone", "season": PRIOR},
    ],
}

EXPECTED_KEYS = {
    "season", "team_name", "sp_rating", "returning_pct", "returning_starters",
    "qb_returning", "recruiting_points", "portal_net", "coach_first_year",
    "prior_sos", "forward_sos_shift",
}


def _by_name(rows, name):
    return next(r for r in rows if r["team_name"] == name)


def test_build_priors_rows_row_shape_and_count():
    rows = build_priors_rows(PARSED, SEASON)
    assert len(rows) == 3
    for row in rows:
        assert set(row.keys()) == EXPECTED_KEYS
        assert row["season"] == SEASON


def test_build_priors_rows_hand_checked_sos():
    rows = build_priors_rows(PARSED, SEASON)
    ames = _by_name(rows, "Ames")
    assert ames["prior_sos"] == 5.0
    assert ames["forward_sos_shift"] == -2.5

    boone = _by_name(rows, "Boone")
    assert boone["prior_sos"] == 15.0
    assert boone["forward_sos_shift"] == 5.0


def test_build_priors_rows_missing_prior_schedule_is_none():
    rows = build_priors_rows(PARSED, SEASON)
    coralville = _by_name(rows, "Coralville")
    assert coralville["prior_sos"] is None
    assert coralville["forward_sos_shift"] is None


def test_build_priors_rows_passthrough_fields_and_defaults():
    rows = build_priors_rows(PARSED, SEASON)
    ames = _by_name(rows, "Ames")
    assert ames["sp_rating"] == 20.0
    assert ames["returning_pct"] == 0.65
    assert ames["returning_starters"] == 0.60
    assert ames["qb_returning"] is True
    assert ames["recruiting_points"] == 250.0
    assert ames["portal_net"] == 3.0
    assert ames["coach_first_year"] is True

    coralville = _by_name(rows, "Coralville")
    assert coralville["returning_pct"] is None
    assert coralville["returning_starters"] is None
    assert coralville["qb_returning"] is None
    assert coralville["recruiting_points"] is None
    assert coralville["portal_net"] == 0.0  # no portal entry -> defaults to 0
    assert coralville["coach_first_year"] is False  # no coach entry -> defaults False


# -- backfill tolerance: early seasons where an endpoint returns nothing --------

def test_build_priors_rows_empty_portal_payload_gives_nan_portal_net():
    """Portal data starts in 2021: for 2016 the endpoint returns nothing, so
    portal_net is unknown (NaN in the parquet), NOT a fabricated 0.0."""
    parsed = {**PARSED, "portal": {}}
    rows = build_priors_rows(parsed, 2016)
    assert len(rows) == 3
    for r in rows:
        assert r["portal_net"] is None
    # mixed with a covered season, the column is float64 and 2016 is NaN
    df = pd.DataFrame(rows + build_priors_rows(PARSED, 2022))
    assert df["portal_net"].dtype == "float64"
    assert df.loc[df.season == 2016, "portal_net"].isna().all()
    assert df.loc[df.season == 2022, "portal_net"].notna().all()
    # everything else is still populated
    assert _by_name(rows, "Ames")["sp_rating"] == 20.0


def _boom(path, key, params=None):
    raise RuntimeError("404")


def _fetch(path, season, current=2026):
    parser = bcp.cfbd.parse_portal if path == "/player/portal" else bcp.cfbd.parse_returning
    return bcp._fetch_parsed(parser, path, "k", {"year": season},
                             bcp._tolerates_missing(path, season, current))


def test_early_season_error_and_empty_are_tolerated(monkeypatch, capsys):
    monkeypatch.setattr(bcp.cfbd, "_get", _boom)
    assert _fetch("/player/returning", 2016) == {}
    assert "WARN" in capsys.readouterr().out
    monkeypatch.setattr(bcp.cfbd, "_get", lambda path, key, params=None: [])
    assert _fetch("/player/returning", 2016) == {}


def test_current_season_error_and_empty_raise(monkeypatch):
    monkeypatch.setattr(bcp.cfbd, "_get", _boom)
    for path in ("/player/returning", "/recruiting/teams", "/player/portal"):
        with pytest.raises(RuntimeError):
            _fetch(path, 2026)
    monkeypatch.setattr(bcp.cfbd, "_get", lambda path, key, params=None: [])
    with pytest.raises(RuntimeError):
        _fetch("/player/returning", 2026)


def test_portal_empty_pre_2021_tolerated_but_2023_raises(monkeypatch):
    monkeypatch.setattr(bcp.cfbd, "_get", lambda path, key, params=None: [])
    assert _fetch("/player/portal", 2020) == {}
    with pytest.raises(RuntimeError):
        _fetch("/player/portal", 2023)   # historical but portal data exists
    monkeypatch.setattr(bcp.cfbd, "_get", _boom)
    with pytest.raises(RuntimeError):
        _fetch("/player/portal", 2023)
    assert _fetch("/player/portal", 2019) == {}


def test_tolerates_missing_rule():
    t = bcp._tolerates_missing
    assert t("/player/returning", 2025, 2026) and not t("/player/returning", 2026, 2026)
    assert t("/player/portal", 2020, 2026) and not t("/player/portal", 2021, 2026)


def test_safe_fetch_passes_through_data(monkeypatch):
    payload = [{"origin": "Ames", "destination": "Boone", "rating": 0.9}]
    monkeypatch.setattr(bcp.cfbd, "_get", lambda path, key, params=None: payload)
    out = bcp._fetch_parsed(bcp.cfbd.parse_portal, "/player/portal", "k", {"year": 2022})
    assert set(out) == {"Ames", "Boone"}


def test_committed_priors_cover_every_current_fbs_team():
    """The live v2 prior reads seasons S-1 and S of assets/cfb/priors.parquet; a team with no row
    silently gets no prior. Arkansas/Missouri/Virginia were absent from every season until the
    CFBD->ESPN mapping fix (their bare names were ambiguous with the State/Tech siblings)."""
    from sportsmodel.cfb.teams import load_fbs_ids

    df = pd.read_parquet(_p.parents[1] / "assets" / "cfb" / "priors.parquet")
    latest = int(df["season"].max())
    fbs = load_fbs_ids()
    for season in (latest - 1, latest):
        missing = sorted(fbs - set(df.loc[df["season"] == season, "team_espn_id"]))
        assert not missing, f"priors.parquet {season} lacks FBS teams {missing}"
    for team in ("8", "142", "258"):   # Arkansas, Missouri, Virginia: every season
        assert set(df.loc[df["team_espn_id"] == team, "season"]) == set(df["season"])
