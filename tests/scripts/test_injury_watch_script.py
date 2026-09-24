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
