"""scripts/backfill_accuracy_pick_probs.py: the pure planning step (no network, no DB) and its --dry-run plumbing."""
import importlib.util
import math
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backfill_accuracy_pick_probs.py"
_spec = importlib.util.spec_from_file_location("backfill_accuracy_pick_probs", _PATH)
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

MARGIN = {"kind": "margin", "offset": 2, "pmf": [0.1, 0.2, 0.3, 0.25, 0.15]}
TOTAL = {"kind": "pmf", "pmf": [0, 0.1, 0.2, 0.4, 0.2, 0.1]}


def acc(pk, **o):
    row = {"sport": "cfb", "game_pk": pk, "pred_margin": 7.0, "market_spread": -1.5, "pred_total": 47.0, "market_total": 2.5,
           "spread_pick_prob": None, "total_pick_prob": None}
    row.update(o)
    return row


def test_plan_fills_both_columns_from_the_prediction_distributions():
    plan = bf.plan_updates([acc(1)], {1: {"margin_dist": MARGIN, "total_dist": TOTAL}}, {})
    (u,) = plan
    assert (u["sport"], u["game_pk"]) == ("cfb", 1)
    assert math.isclose(u["spread_pick_prob"], 0.15) and math.isclose(u["total_pick_prob"], 0.7)


def test_plan_uses_the_nfl_sim_row_when_the_prediction_has_no_distribution():
    plan = bf.plan_updates([acc(2, sport="nfl")], {2: {"margin_dist": None, "total_dist": None}},
                           {2: {"margin_dist": MARGIN, "total_dist": TOTAL}})
    assert math.isclose(plan[0]["spread_pick_prob"], 0.15) and math.isclose(plan[0]["total_pick_prob"], 0.7)


def test_plan_leaves_null_without_distribution_and_skips_empty_updates():
    assert bf.plan_updates([acc(3)], {}, {}) == []                      # no dist anywhere: nothing to write, never invented
    only_total = bf.plan_updates([acc(4)], {4: {"margin_dist": None, "total_dist": TOTAL}}, {})
    assert only_total[0]["spread_pick_prob"] is None and math.isclose(only_total[0]["total_pick_prob"], 0.7)
    assert bf.plan_updates([acc(5, market_spread=None, market_total=None)], {5: {"margin_dist": MARGIN, "total_dist": TOTAL}}, {}) == []


def test_plan_never_overwrites_a_value_already_stored():
    done = acc(6, spread_pick_prob=0.5, total_pick_prob=0.5)
    assert bf.plan_updates([done], {6: {"margin_dist": MARGIN, "total_dist": TOTAL}}, {}) == []
    half = acc(7, spread_pick_prob=0.5)
    (u,) = bf.plan_updates([half], {7: {"margin_dist": MARGIN, "total_dist": TOTAL}}, {})
    assert u["spread_pick_prob"] is None and math.isclose(u["total_pick_prob"], 0.7)


def test_plan_pick_side_follows_the_edge_sign():
    away = acc(8, pred_margin=0.4)        # edge = 0.4 - 1.5 < 0 -> away pick
    (u,) = bf.plan_updates([away], {8: {"margin_dist": MARGIN, "total_dist": None}}, {})
    assert math.isclose(u["spread_pick_prob"], 0.85)


def test_update_sql_only_touches_the_two_columns_and_keeps_stored_values():
    sql = bf.UPDATE_SQL
    assert "UPDATE prediction_accuracy" in sql and "COALESCE(%s, spread_pick_prob)" in sql and "COALESCE(%s, total_pick_prob)" in sql
    assert "graded_at" not in sql
    assert "WHERE sport = %s AND game_pk = %s" in sql


def test_main_dry_run_never_writes(monkeypatch, capsys):
    calls = {"write": 0}

    class Cur:
        def __init__(self):
            self.last = ""

        def execute(self, sql, params=None):
            self.last = sql
            if sql.lstrip().upper().startswith("UPDATE"):
                calls["write"] += 1

        def executemany(self, sql, rows):
            calls["write"] += 1

        def fetchall(self):
            if "information_schema" in self.last:
                return [("spread_pick_prob",), ("total_pick_prob",)]
            if "FROM prediction_accuracy" in self.last:
                return [("cfb", 1, 7.0, -1.5, 47.0, 2.5, None, None)]
            if "FROM game_predictions" in self.last:
                return [(1, MARGIN, TOTAL)]
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn:
        def cursor(self):
            return Cur()

        def commit(self):
            calls["write"] += 1

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bf, "get_postgres", lambda: Conn())
    monkeypatch.setattr(bf, "_served_version", lambda: None)   # never touch a real DB
    monkeypatch.setattr(bf.config, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr("sys.argv", ["backfill", "--dry-run"])
    bf.main()
    out = capsys.readouterr().out
    assert calls["write"] == 0
    assert "dry-run: would update 3 row(s); nothing written." in out   # the fake returns one fillable row per sport



def test_backfill_survives_a_failing_nfl_sim_lookup_and_passes_the_served_version(monkeypatch, capsys):
    seen = {"rollbacks": 0, "served": []}

    class Cur:
        def __init__(self):
            self.last = ""

        def execute(self, sql, params=None):
            self.last = sql
            if "FROM nfl_sim" in sql:
                seen["served"].append(params[0])
                raise RuntimeError("permission denied for table nfl_sim")

        def fetchall(self):
            if "information_schema" in self.last:
                return [("spread_pick_prob",), ("total_pick_prob",)]
            if "FROM prediction_accuracy" in self.last:
                return [("nfl", 1, 7.0, -1.5, 47.0, 2.5, None, None)]
            if "FROM game_predictions" in self.last:
                return [(1, MARGIN, TOTAL)]
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn:
        def cursor(self):
            return Cur()

        def rollback(self):
            seen["rollbacks"] += 1

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(bf, "get_postgres", lambda: Conn())
    monkeypatch.setattr(bf, "_served_version", lambda: "nfl-sim-ml-v2")
    monkeypatch.setattr(bf.config, "DATABASE_URL", "postgres://test")
    monkeypatch.setattr("sys.argv", ["backfill", "--dry-run", "--sport", "nfl"])
    bf.main()                                                     # must not raise
    out = capsys.readouterr().out
    assert seen["served"] == ["nfl-sim-ml-v2"] and seen["rollbacks"] == 1
    assert "WARN" in out and "dry-run: would update 1 row(s)" in out   # the prediction's own distributions still count
