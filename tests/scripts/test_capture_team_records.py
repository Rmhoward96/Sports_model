"""Test scripts/capture_team_records.py pure functions using importlib load."""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".." / "scripts" / "capture_team_records.py"
_spec = importlib.util.spec_from_file_location("capture_team_records", _SCRIPT_PATH)
capture_team_records = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capture_team_records)


def test_resolve_names_matched_name_replaced():
    """Matched AN names are replaced with our names; unmatched stay as AN names."""
    rows = [
        {"an_team_name": "Buffalo Bills", "records": {}},
        {"an_team_name": "Alabama Crimson Tide", "records": {}},
        {"an_team_name": "Nowhere Goblins", "records": {}},
    ]
    ours = ["Buffalo Bills", "Alabama Crimson Tide"]
    result_rows, unmatched = capture_team_records.resolve_names(rows, ours)

    assert len(result_rows) == 3
    assert result_rows[0]["team_name"] == "Buffalo Bills"
    assert result_rows[1]["team_name"] == "Alabama Crimson Tide"
    assert result_rows[2]["team_name"] == "Nowhere Goblins"  # unmatched, stays as AN name
    assert unmatched == ["Nowhere Goblins"]


def test_resolve_names_only_unmatched_reported():
    """Only AN names not in our list are reported as unmatched."""
    rows = [
        {"an_team_name": "Stanford Cardinal", "records": {}},
        {"an_team_name": "UCLA Bruins", "records": {}},
    ]
    ours = ["UCLA Bruins"]
    result_rows, unmatched = capture_team_records.resolve_names(rows, ours)

    assert unmatched == ["Stanford Cardinal"]
    assert result_rows[1]["team_name"] == "UCLA Bruins"


# --- F2: store the season we requested -------------------------------------

def test_stamp_season_overrides_none_and_other_years():
    rows = [
        {"an_team_name": "Buffalo Bills", "season": None, "records": {}},
        {"an_team_name": "Miami Dolphins", "season": 2025, "records": {}},
        {"an_team_name": "New York Jets", "records": {}},
    ]
    out = capture_team_records.stamp_season(rows, 2026)
    assert [r["season"] for r in out] == [2026, 2026, 2026]
    assert rows[0]["season"] is None  # PURE: input untouched


# --- F4: NFL abbreviation fallback -----------------------------------------

def test_resolve_names_nfl_abbr_fallback():
    rows = [
        {"sport": "nfl", "an_team_name": "Bills", "abbr": "BUF", "records": {}},
        {"sport": "nfl", "an_team_name": "Rams", "abbr": "LAR", "records": {}},
    ]
    ours = capture_team_records.our_team_names("nfl")
    result_rows, unmatched = capture_team_records.resolve_names(rows, ours)
    assert result_rows[0]["team_name"] == "Buffalo Bills"
    assert result_rows[1]["team_name"] == "Los Angeles Rams"
    assert unmatched == []


def test_resolve_names_nfl_bad_abbr_keeps_an_name():
    rows = [
        {"sport": "nfl", "an_team_name": "Goblins", "abbr": "ZZZ", "records": {}},
        {"sport": "nfl", "an_team_name": "Nobody", "abbr": None, "records": {}},
    ]
    result_rows, unmatched = capture_team_records.resolve_names(
        rows, capture_team_records.our_team_names("nfl"))
    assert [r["team_name"] for r in result_rows] == ["Goblins", "Nobody"]
    assert unmatched == ["Goblins", "Nobody"]


def test_resolve_names_cfb_does_not_use_nfl_abbr_fallback():
    rows = [{"sport": "cfb", "an_team_name": "Bills U", "abbr": "BUF", "records": {}}]
    result_rows, unmatched = capture_team_records.resolve_names(rows, ["Alabama Crimson Tide"])
    assert result_rows[0]["team_name"] == "Bills U"
    assert unmatched == ["Bills U"]


# --- F3: fail loudly --------------------------------------------------------

import pytest  # noqa: E402


def _patch_main(monkeypatch, fail_leagues=(), zero_leagues=()):
    upserted = []
    monkeypatch.setenv("APIFY_TOKEN", "test-token")
    monkeypatch.setattr(capture_team_records, "current_season", lambda sport: 2026)
    monkeypatch.setattr(capture_team_records, "our_team_names", lambda sport: ["Team A"])

    def fake_fetch(token, leagues, game_status, extra_input):
        if leagues[0] in fail_leagues:
            raise RuntimeError(f"actor blew up for {leagues[0]}")
        return [leagues[0]]

    def fake_parse(items):
        league = items[0]
        if league in zero_leagues:
            return []
        return [{"sport": league, "season": 2025, "an_team_name": "Team A", "abbr": None,
                 "records": {}}]

    def fake_upsert(rows):
        upserted.append(rows)
        return len(rows)

    monkeypatch.setattr(capture_team_records, "fetch_splits", fake_fetch)
    monkeypatch.setattr(capture_team_records, "parse_team_records", fake_parse)
    monkeypatch.setattr(capture_team_records, "upsert_team_betting_records", fake_upsert)
    return upserted


def test_main_one_league_raises_exits_1_after_other_league_upserted(monkeypatch, capsys):
    upserted = _patch_main(monkeypatch, fail_leagues=("nfl",))
    with pytest.raises(SystemExit) as exc:
        capture_team_records.main()
    assert exc.value.code == 1
    assert len(upserted) == 1 and upserted[0][0]["sport"] == "ncaaf"
    assert "nfl" in capsys.readouterr().out


def test_main_league_with_zero_rows_exits_1(monkeypatch):
    upserted = _patch_main(monkeypatch, zero_leagues=("ncaaf",))
    with pytest.raises(SystemExit) as exc:
        capture_team_records.main()
    assert exc.value.code == 1
    assert len(upserted) == 2  # both leagues still processed


def test_main_both_ok_no_exit_and_rows_stamped_with_requested_season(monkeypatch):
    upserted = _patch_main(monkeypatch)
    capture_team_records.main()  # must not raise
    assert len(upserted) == 2
    assert all(r["season"] == 2026 for rows in upserted for r in rows)


def test_resolve_names_cfb_aliases_from_first_live_run():
    # Action Network names seen unmatched on the first live capture (2026-09-23)
    # that are the same schools under a different name in fbs_teams.json.
    ours = ["Miami Hurricanes", "Miami (OH) RedHawks", "NC State Wolfpack", "Massachusetts Minutemen",
            "Hawai'i Rainbow Warriors", "UL Monroe Warhawks", "App State Mountaineers",
            "Delaware Blue Hens", "Sam Houston Bearkats"]
    an = ["Miami (FL) Hurricanes", "North Carolina State Wolfpack", "UMass Minutemen",
          "Hawai'i Warriors", "UL-Monroe Warhawks", "Appalachian State Mountaineers",
          "Delaware Fightin Blue Hens", "Sam Houston State Bearkats", "Sacramento State Hornets"]
    rows = [{"sport": "cfb", "an_team_name": n, "abbr": None} for n in an]
    out, unmatched = capture_team_records.resolve_names(rows, ours)
    got = {r["an_team_name"]: r["team_name"] for r in out}
    assert got["Miami (FL) Hurricanes"] == "Miami Hurricanes"
    assert got["North Carolina State Wolfpack"] == "NC State Wolfpack"
    assert got["UMass Minutemen"] == "Massachusetts Minutemen"
    assert got["Hawai'i Warriors"] == "Hawai'i Rainbow Warriors"
    assert got["UL-Monroe Warhawks"] == "UL Monroe Warhawks"
    assert got["Appalachian State Mountaineers"] == "App State Mountaineers"
    assert got["Delaware Fightin Blue Hens"] == "Delaware Blue Hens"
    assert got["Sam Houston State Bearkats"] == "Sam Houston Bearkats"
    assert unmatched == ["Sacramento State Hornets"]   # not an FBS team in our list
