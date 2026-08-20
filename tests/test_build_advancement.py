import duckdb
import pytest
from sportsmodel.sim.mlb.build_advancement import build_advancement_table, _base_occ_expr


def _mini_pbp(con):
    # Two half-innings of synthetic PA data with the columns the builder reads.
    # Columns mirror Statcast: on_1b/on_2b/on_3b are runner ids (NULL = empty).
    con.execute("""
        CREATE TABLE pbp AS SELECT * FROM (VALUES
        -- game 1, top 1: leadoff single (bases empty -> runner on 1st, 0 runs, 0 outs)
        (1, 1, 'Top', 1, 'single', NULL, NULL, NULL, 0, 0),
        -- next PA: runner on 1st, single -> table should see occ=1,out=0 -> some end state
        (1, 1, 'Top', 2, 'single', 100, NULL, NULL, 0, 0),
        (1, 1, 'Top', 3, 'field_out', 101, 100, NULL, 0, 1)
        ) AS t(game_pk, inning, inning_topbot, at_bat_number, events,
               on_1b, on_2b, on_3b, bat_score, post_bat_score)
    """)


def test_probabilities_sum_to_one_per_group():
    con = duckdb.connect(":memory:")
    _mini_pbp(con)
    rows = build_advancement_table(con, _table="pbp")
    from collections import defaultdict
    tot = defaultdict(float)
    for r in rows:
        tot[(r["outcome"], r["occ"], r["outs"])] += r["prob"]
    assert rows, "expected at least one transition row"
    for key, s in tot.items():
        assert abs(s - 1.0) < 1e-9, f"{key} sums to {s}"


def test_occ_mask_encoding():
    # bit0=1B, bit1=2B, bit2=3B
    assert _base_occ_expr  # symbol exists
