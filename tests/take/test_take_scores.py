"""scripts/take_scores.py: fetch final scores for leagues with unsettled picks, settle them. Fake Deps only."""
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sportsmodel.take import constants

_SPEC = importlib.util.spec_from_file_location(
    "take_scores", Path(__file__).resolve().parents[2] / "scripts" / "take_scores.py")
take_scores = importlib.util.module_from_spec(_SPEC)
sys.modules["take_scores"] = take_scores          # @dataclass resolves its module through sys.modules
_SPEC.loader.exec_module(take_scores)
run_scores, Deps = take_scores.run_scores, take_scores.Deps

NOW = datetime(2026, 11, 3, 9, 30, tzinfo=UTC)


def score_event(event_id="e1", home=24, away=17, completed=True):
    return {"id": event_id, "home_team": "Home", "away_team": "Away", "completed": completed,
            "scores": [{"name": "Home", "score": str(home)}, {"name": "Away", "score": str(away)}] if completed else None}


def pick(pick_id=7, event_id="e1", market="spread", side="home", line=-3.5, price=-110, league="nfl"):
    return {"pick_id": pick_id, "event_id": event_id, "market": market, "side": side, "line": line, "price": price,
            "league": league}


class Rec:
    def __init__(self, *, remaining="5000", cost="2", tables=(), leagues=("nfl",), picks=None, scores=None, fail=(),
                 fail_write=()):
        self.order, self.args = [], {}
        self.stored, self.outcomes = {}, []
        self.remaining_value, self.cost, self.tables = remaining, cost, list(tables)
        self.leagues, self.picks = list(leagues), picks if picks is not None else {}
        self.scores, self.fail, self.fail_write = scores if scores is not None else {}, set(fail), set(fail_write)

    def n(self, name):
        return self.order.count(name)

    def _note(self, name, *a):
        self.order.append(name)
        self.args.setdefault(name, []).append(a)

    def deps(self) -> Deps:
        def fetch_scores(cfg, days):
            self._note("fetch_scores", cfg, days)
            if cfg.key in self.fail:
                raise RuntimeError("boom")
            return self.scores.get(cfg.key, [])

        def store_scores(finals):
            self._note("store_scores", finals)
            if "store_scores" in self.fail_write:
                raise RuntimeError("db down")
            self.stored.update(finals)
            return len(finals)

        def set_outcomes(rows):
            self._note("set_outcomes", rows)
            self.outcomes += rows
            return len(rows)

        return Deps(
            missing_tables=lambda: self.tables, prime=lambda: self._note("prime") or [], fetch_scores=fetch_scores,
            remaining=lambda: self.remaining_value, last_cost=lambda: self.cost,
            unsettled_leagues=lambda: self._note("unsettled_leagues") or self.leagues,
            unsettled=lambda lg: self._note("unsettled", lg) or self.picks.get(lg, []),
            store_scores=store_scores, set_outcomes=set_outcomes, log_call=lambda *a: self._note("log_call", *a))


def test_scores_settles_final_events_and_logs_the_call():
    f = Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event(home=24, away=17)]}, remaining="4998")
    out = run_scores(NOW, f.deps(), run_id="run9")
    cfg, days = f.args["fetch_scores"][0]
    assert cfg.odds_sport == "americanfootball_nfl" and days == 3
    assert f.stored == {"e1": (24, 17)}
    assert [(o["pick_id"], o["outcome"]) for o in f.outcomes] == [(7, "win")]        # home -3.5, won by 7
    assert f.outcomes[0]["units"] == pytest.approx(100 / 110)
    assert f.args["log_call"][0] == ("scores", "nfl", "scores", 2, 4998, [], "run9")
    assert f.order.index("fetch_scores") < f.order.index("store_scores") < f.order.index("set_outcomes") < f.order.index("log_call")
    assert out["settled"] == 1 and out["credits"] == 2 and out["leagues"]["nfl"] == {"finals": 1, "settled": 1, "credits": 2}


def test_scores_settles_losses_and_pushes():
    picks = [pick(1, "e1", "spread", "home", -7.0), pick(2, "e1", "total", "over", 41.0, 100), pick(3, "e1", "ml", "away", None, 150)]
    f = Rec(picks={"nfl": picks}, scores={"nfl": [score_event(home=24, away=17)]})
    run_scores(NOW, f.deps())
    assert {o["pick_id"]: (o["outcome"], o["units"]) for o in f.outcomes} == {
        1: ("push", 0.0), 2: ("push", 0.0), 3: ("loss", -1.0)}                          # 24-17: margin 7, total 41


def test_scores_only_settles_events_that_are_final():
    f = Rec(picks={"nfl": [pick(1, "e1"), pick(2, "e2"), pick(3, "e3")]},
            scores={"nfl": [score_event("e1"), score_event("e2", completed=False)]})        # e2 live, e3 not returned
    out = run_scores(NOW, f.deps())
    assert [o["pick_id"] for o in f.outcomes] == [1] and out["settled"] == 1
    assert list(f.stored) == ["e1"]


def test_scores_with_nothing_final_writes_no_outcomes_but_still_logs_the_spend():
    f = Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event(completed=False)]})
    run_scores(NOW, f.deps())
    assert f.n("set_outcomes") == 0 and f.n("store_scores") == 0 and f.n("log_call") == 1


def test_scores_does_nothing_when_nothing_is_unsettled():
    f = Rec(leagues=[])
    out = run_scores(NOW, f.deps())
    assert f.n("prime") == 0 and f.n("fetch_scores") == 0 and out["credits"] == 0 and out["settled"] == 0


def test_scores_primes_then_fetches_each_league_in_order():
    f = Rec(leagues=["nfl", "mlb"], picks={}, scores={})
    run_scores(NOW, f.deps())
    assert f.order[:2] == ["unsettled_leagues", "prime"]
    assert [a[0].key for a in f.args["fetch_scores"]] == ["nfl", "mlb"]


def test_scores_skips_below_floor_and_stops_the_remaining_leagues():
    f = Rec(remaining="299", leagues=["nfl", "mlb"], picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    out = run_scores(NOW, f.deps())
    assert f.n("fetch_scores") == 0 and f.n("set_outcomes") == 0 and out["skipped"] == ["nfl", "mlb"]
    f = Rec(remaining="300", picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    run_scores(NOW, f.deps())
    assert f.n("fetch_scores") == 1 and constants.SCORES_FLOOR == 300


def test_scores_floor_is_rechecked_per_league():
    f = Rec(leagues=["nfl", "mlb", "nhl"], picks={})
    deps = f.deps()
    deps.remaining = lambda: "400" if f.n("fetch_scores") == 0 else "250"      # the first league drains the balance
    out = run_scores(NOW, deps)
    assert [a[0].key for a in f.args["fetch_scores"]] == ["nfl"] and out["skipped"] == ["mlb", "nhl"]


def test_scores_unknown_balance_runs():
    f = Rec(remaining=None, picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    run_scores(NOW, f.deps())
    assert f.n("fetch_scores") == 1


def test_scores_one_failing_league_warns_and_the_next_goes_on(capsys):
    f = Rec(leagues=["nfl", "mlb"], picks={"mlb": [pick(2, "m1", "ml", "home", None, -120, "mlb")]},
            scores={"mlb": [score_event("m1", 5, 2)]}, fail={"nfl"})
    out = run_scores(NOW, f.deps())
    assert "failed" in out["leagues"]["nfl"] and [o["pick_id"] for o in f.outcomes] == [2]
    assert "::warning::take-scores: nfl scores failed" in capsys.readouterr().out


def test_scores_write_failure_after_the_paid_call_is_a_failed_league_but_keeps_the_credits(capsys):
    f = Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event()]}, fail_write={"store_scores"})
    out = run_scores(NOW, f.deps())
    assert set(out["leagues"]["nfl"]) == {"failed"} and out["credits"] == 2 and f.n("set_outcomes") == 0
    assert "::warning::take-scores: nfl scores failed" in capsys.readouterr().out


def test_scores_credit_cost_falls_back_to_two_when_the_header_is_unreadable():
    f = Rec(cost=None, picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    assert run_scores(NOW, f.deps())["credits"] == 2


def test_scores_dry_run_writes_nothing(capsys):
    f = Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    out = run_scores(NOW, f.deps(), dry_run=True)
    for w in ("store_scores", "set_outcomes", "log_call"):
        assert f.n(w) == 0, w
    assert out["settled"] == 1 and f.n("fetch_scores") == 1 and "DRY RUN" in capsys.readouterr().out


def test_scores_missing_tables_warn_and_do_nothing(capsys):
    f = Rec(tables=["take_results"])
    assert run_scores(NOW, f.deps())["status"] == "no_table" and f.order == []
    assert "migration_take_these.sql" in capsys.readouterr().out


def _main_env(monkeypatch, rec):
    monkeypatch.setenv("DATABASE_URL", "x")
    monkeypatch.setenv("ODDS_API_KEY", "y")
    monkeypatch.setattr(take_scores, "_now", lambda: NOW)
    monkeypatch.setattr(take_scores, "writes_disabled", lambda: False)
    monkeypatch.setattr(take_scores, "_db_deps", lambda: rec.deps())


def test_main_exits_1_when_every_attempted_league_fails(monkeypatch):
    _main_env(monkeypatch, Rec(leagues=["nfl", "mlb"], fail={"nfl", "mlb"}))
    assert take_scores.main([]) == 1
    _main_env(monkeypatch, Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event()]}, fail_write={"store_scores"}))
    assert take_scores.main([]) == 1                                     # a write failure is a failed league too


def test_main_exit_codes(monkeypatch):
    _main_env(monkeypatch, Rec(leagues=["nfl", "mlb"], fail={"nfl"}))
    assert take_scores.main([]) == 0                                     # one league survived
    _main_env(monkeypatch, Rec(leagues=[]))
    assert take_scores.main([]) == 0                                     # nothing attempted
    _main_env(monkeypatch, Rec(tables=["take_results"]))
    assert take_scores.main([]) == 0                                     # migration not run yet
    monkeypatch.delenv("ODDS_API_KEY")
    assert take_scores.main([]) == 1


def test_main_dry_run_flag_and_ledger_off_both_mean_dry_run(monkeypatch):
    f = Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    _main_env(monkeypatch, f)
    assert take_scores.main(["--dry-run"]) == 0 and f.n("set_outcomes") == 0
    f = Rec(picks={"nfl": [pick()]}, scores={"nfl": [score_event()]})
    _main_env(monkeypatch, f)
    monkeypatch.setattr(take_scores, "writes_disabled", lambda: True)
    assert take_scores.main([]) == 0 and f.n("set_outcomes") == 0
