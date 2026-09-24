"""Pure seams of scripts/injury_watch.py: plan_check, snapshot_rows,
write_github_output. main()'s DB/injury IO is thin and not unit-tested."""
import importlib.util
import pathlib

from sportsmodel.serving.injury_watch import fingerprint, game_statuses

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "injury_watch.py"
_spec = importlib.util.spec_from_file_location("injury_watch_script", _p)
iw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(iw)

ATL, CAR, DAL, NYG = "Atlanta Falcons", "Carolina Panthers", "Dallas Cowboys", "New York Giants"
GAMES = [
    {"game_pk": 1, "home_team": ATL, "away_team": CAR},
    {"game_pk": 2, "home_team": DAL, "away_team": NYG},
]


def _stored_for(game, inj):
    st = game_statuses(game["home_team"], game["away_team"], inj)
    return {"fingerprint": fingerprint(st), "statuses": st}


def test_plan_check_nothing_changed():
    inj = {ATL: [{"player": "Michael Penix Jr.", "status": "Out"}]}
    stored = {g["game_pk"]: _stored_for(g, inj) for g in GAMES}
    # a new Questionable designation never triggers
    inj_now = {ATL: [{"player": "Michael Penix Jr.", "status": "Out"}],
               DAL: [{"player": "Q", "status": "Questionable"}]}
    changed, lines = iw.plan_check(GAMES, inj_now, stored)
    assert changed == []
    assert lines == []


def test_plan_check_flip_reports_diff():
    old = {ATL: [{"player": "Michael Penix Jr.", "status": "Out"}]}
    stored = {g["game_pk"]: _stored_for(g, old) for g in GAMES}
    changed, lines = iw.plan_check(GAMES, {}, stored)
    assert changed == [1]
    assert lines == [f"{CAR} @ {ATL} (game_pk 1)",
                     f"  {ATL}: Michael Penix Jr. Out -> (none)"]


def test_plan_check_missing_snapshot_is_changed():
    inj = {}
    stored = {1: _stored_for(GAMES[0], inj)}
    changed, lines = iw.plan_check(GAMES, inj, stored)
    assert changed == [2]
    assert lines == [f"{NYG} @ {DAL} (game_pk 2)", "  no stored snapshot"]


def test_snapshot_rows():
    inj = {NYG: [{"player": "B", "status": "Doubtful"}]}
    rows = iw.snapshot_rows("nfl", GAMES, inj)
    assert [r["game_pk"] for r in rows] == [1, 2]
    assert all(r["sport"] == "nfl" for r in rows)
    assert rows[1]["statuses"] == [{"team": NYG, "player": "B", "status": "Doubtful"}]
    assert rows[1]["fingerprint"] == fingerprint(rows[1]["statuses"])
    assert rows[0]["statuses"] == [] and rows[0]["fingerprint"] == fingerprint([])


def test_write_github_output_appends(tmp_path, monkeypatch):
    out = tmp_path / "gh_out"
    out.write_text("prior=1\n")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    iw.write_github_output(True)
    iw.write_github_output(False)
    assert out.read_text() == "prior=1\nchanged=true\nchanged=false\n"


def test_write_github_output_unset_is_noop(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    iw.write_github_output(True)  # no error


# ---- fix round 1 ----
from datetime import datetime, timedelta, timezone  # noqa: E402

_dp = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "desk_inputs.py"
_dspec = importlib.util.spec_from_file_location("desk_inputs_for_iw", _dp)
desk = importlib.util.module_from_spec(_dspec)
_dspec.loader.exec_module(desk)

NOW = datetime(2026, 9, 25, 12, tzinfo=timezone.utc)


def test_check_inputs_rekeys_over_7day_names_then_narrows():
    # "Iowa" prefix-matches both ESPN names; Iowa State's game is outside 30h.
    games_7d = [
        {"game_pk": 10, "home_team": "Iowa Hawkeyes", "away_team": "Rutgers Scarlet Knights",
         "commence_time": NOW + timedelta(hours=20)},
        {"game_pk": 11, "home_team": "Iowa State Cyclones", "away_team": "Baylor Bears",
         "commence_time": NOW + timedelta(days=3)},
    ]
    by_name = {"Iowa": [{"player": "Hawk QB", "status": "Out"}],
               "Baylor": [{"player": "Bear RB", "status": "Out"}]}
    games, injuries = iw.check_inputs(games_7d, by_name, desk._rekey_by_espn_name, NOW)
    assert [g["game_pk"] for g in games] == [10]
    # the desk (7-day names) drops ambiguous "Iowa" -> check must too
    desk_view = desk._rekey_by_espn_name(by_name, sorted(
        {g["home_team"] for g in games_7d} | {g["away_team"] for g in games_7d}))
    assert injuries == desk_view
    assert "Iowa Hawkeyes" not in injuries
    assert iw.snapshot_rows("cfb", games, injuries)[0]["statuses"] == []
    # a 30h-only rekey would (wrongly) have attached Iowa's injury
    naive = desk._rekey_by_espn_name(by_name, ["Iowa Hawkeyes", "Rutgers Scarlet Knights"])
    assert "Iowa Hawkeyes" in naive


def test_record_from_bundle_matches_check_fingerprint():
    inj = {ATL: [{"player": "Michael Penix Jr.", "status": "Out", "position": "QB"},
                 {"player": "Q", "status": "Questionable"}],
           CAR: [{"player": "B", "status": "Doubtful"}]}
    bundle = [{"game_pk": 1, "matchup": f"{CAR} @ {ATL}",
               "news": {"injuries": {"home": inj[ATL], "away": inj[CAR]}}},
              {"game_pk": 2, "matchup": f"{NYG} @ {DAL}",
               "news": {"injuries": {"home": [], "away": []}}}]
    rows = iw.bundle_snapshot_rows("nfl", bundle)
    assert rows == iw.snapshot_rows("nfl", GAMES, inj)
    stored = {r["game_pk"]: r for r in rows}
    changed, _ = iw.plan_check(GAMES, inj, stored)
    assert changed == []


def test_bundle_snapshot_rows_skips_malformed_matchup():
    bundle = [{"game_pk": 3, "matchup": "no separator", "news": {"injuries": {"home": [], "away": []}}}]
    assert iw.bundle_snapshot_rows("nfl", bundle) == []


def _boom(*a, **k):
    raise RuntimeError("boom")


def test_record_fetch_error_warns_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(iw, "fetch_games", _boom)
    assert iw.main(["--sport", "nfl", "--record"]) == 0
    assert "WARN" in capsys.readouterr().out


def test_record_bad_bundle_warns_and_exits_zero(tmp_path, capsys):
    assert iw.main(["--sport", "nfl", "--record", "--bundle", str(tmp_path / "missing.json")]) == 0
    assert "WARN" in capsys.readouterr().out


def test_record_upsert_error_warns_and_exits_zero(tmp_path, monkeypatch, capsys):
    from sportsmodel import db
    p = tmp_path / "b.json"
    p.write_text('[{"game_pk": 1, "matchup": "A @ B", "news": {"injuries": {"home": [], "away": []}}}]')
    monkeypatch.setattr(db, "upsert_injury_snapshots", _boom)
    assert iw.main(["--sport", "cfb", "--record", "--bundle", str(p)]) == 0
    assert "WARN" in capsys.readouterr().out


def test_record_from_bundle_needs_no_api_key(tmp_path, monkeypatch):
    from sportsmodel import db
    monkeypatch.setenv("SPORTSDATA_API_KEY", "")
    got = {}
    monkeypatch.setattr(db, "upsert_injury_snapshots", lambda rows: got.setdefault("rows", rows) and len(rows))
    p = tmp_path / "b.json"
    p.write_text('[{"game_pk": 7, "matchup": "A @ B", "news": {"injuries": '
                 '{"home": [{"player": "P", "status": "Out"}], "away": []}}}]')
    assert iw.main(["--sport", "cfb", "--record", "--bundle", str(p)]) == 0
    assert got["rows"] == [{"sport": "cfb", "game_pk": 7, "fingerprint": fingerprint(
        [{"team": "B", "player": "P", "status": "Out"}]),
        "statuses": [{"team": "B", "player": "P", "status": "Out"}]}]


def test_check_nfl_espn_unavailable_is_unknown(tmp_path, monkeypatch, capsys):
    from sportsmodel import db
    out = tmp_path / "gh"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setattr(iw, "fetch_games", lambda sport, now, horizon: [
        {"game_pk": 1, "home_team": ATL, "away_team": CAR, "commence_time": datetime.now(timezone.utc) + timedelta(hours=5)}])
    monkeypatch.setattr(iw, "fetch_injuries", lambda sport, now, api_key: (
        {ATL: [{"player": "P", "status": "Out"}]}, {"espn_available": False}))
    monkeypatch.setattr(db, "load_injury_snapshots", _boom)
    assert iw.main(["--sport", "nfl", "--check"]) == 0
    assert out.read_text() == "changed=false\n"
    assert "WARN" in capsys.readouterr().out


def test_check_flags_change_end_to_end(tmp_path, monkeypatch, capsys):
    from sportsmodel import db
    out = tmp_path / "gh"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    soon = datetime.now(timezone.utc) + timedelta(hours=5)
    later = datetime.now(timezone.utc) + timedelta(days=3)
    monkeypatch.setattr(iw, "fetch_games", lambda sport, now, horizon: [
        {"game_pk": 1, "home_team": ATL, "away_team": CAR, "commence_time": soon},
        {"game_pk": 2, "home_team": DAL, "away_team": NYG, "commence_time": later}])
    monkeypatch.setattr(iw, "fetch_injuries", lambda sport, now, api_key: (
        {ATL: [{"player": "P", "status": "Out"}]}, {"espn_available": True}))
    seen = {}
    monkeypatch.setattr(db, "load_injury_snapshots", lambda sport, pks: seen.setdefault("pks", pks) and {})
    assert iw.main(["--sport", "nfl", "--check"]) == 0
    assert seen["pks"] == [1]  # only the 30h game is checked
    assert out.read_text() == "changed=true\n"
    assert "no stored snapshot" in capsys.readouterr().out
