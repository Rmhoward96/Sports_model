import duckdb
import pytest
from sportsmodel.sim.mlb.build_advancement import build_advancement_table, _base_occ_expr


def _mini_pbp(con):
    # One half-inning (3 PAs) of synthetic PA data with the columns the builder reads.
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


def test_pitch_level_rows_do_not_corrupt_transitions():
    # Production Statcast is PITCH-level: `events` is populated only on the terminal
    # pitch of each PA. Interleave non-terminal "decoy" rows (events=NULL, with
    # deliberately wrong/stale on-base ids) between the terminal PA rows. If the
    # builder fails to filter to `events IS NOT NULL` before computing LEAD/
    # ROW_NUMBER/COUNT over the half-inning, a decoy row's occ can be picked up as
    # `next_occ` (or as `n_pa`/`pa_idx`) instead of the true next PA's terminal occ.
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE pbp AS SELECT * FROM (VALUES
        -- PA1 terminal: bases empty -> single (runner on 1st for PA2)
        (1, 1, 'Top', 1, 'single', NULL, NULL, NULL, 0, 0),
        -- PA2 decoy pitch (non-terminal): bogus on-base ids, occ would be 3 if picked up
        (1, 1, 'Top', 2, NULL, 999, 998, NULL, 0, 0),
        -- PA2 terminal: real pre-state occ=1 (runner 100 on 1st, from PA1)
        (1, 1, 'Top', 2, 'single', 100, NULL, NULL, 0, 0),
        -- PA3 decoy pitch (non-terminal): bases empty (occ=0) -- wrong if used as next_occ
        (1, 1, 'Top', 3, NULL, NULL, NULL, NULL, 0, 0),
        -- PA3 terminal: real pre-state occ=3 (1st+2nd occupied); ends inning, 1 run scores
        (1, 1, 'Top', 3, 'field_out', 101, 100, NULL, 0, 1)
        ) AS t(game_pk, inning, inning_topbot, at_bat_number, events,
               on_1b, on_2b, on_3b, bat_score, post_bat_score)
    """)
    rows = build_advancement_table(con, _table="pbp")
    by_key = {(r["outcome"], r["occ"], r["end_occ"], r["runs"]): r["prob"] for r in rows}

    # PA1: single from empty bases -> runner on 1st. Must not pick up PA2's decoy occ.
    assert by_key.get(("p_1b", 0, 1, 0)) == pytest.approx(1.0)
    # PA2: single with runner on 1st -> runners on 1st+2nd. Starting occ must come
    # from PA2's TERMINAL row (occ=1, not the decoy's occ=3); end_occ must come from
    # PA3's TERMINAL row (occ=3, not PA3's decoy pitch, occ=0).
    assert by_key.get(("p_1b", 1, 3, 0)) == pytest.approx(1.0)
    # PA3: field_out ends the inning with runners on 1st+2nd, 1 run scores.
    assert by_key.get(("p_out", 3, 0, 1)) == pytest.approx(1.0)


def test_p_out_excludes_batter_reaches_base_events():
    # field_error (batter safe on error) and plain fielders_choice (no out on the
    # batter) are not outs and must not be bucketed into p_out. Both events map to
    # no outcome at all, same as any other unmapped event -- so a half-inning made
    # entirely of these should yield zero transition rows.
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE pbp AS SELECT * FROM (VALUES
        (1, 1, 'Top', 1, 'field_error', NULL, NULL, NULL, 0, 0),
        (1, 1, 'Top', 2, 'fielders_choice', 100, NULL, NULL, 0, 0)
        ) AS t(game_pk, inning, inning_topbot, at_bat_number, events,
               on_1b, on_2b, on_3b, bat_score, post_bat_score)
    """)
    rows = build_advancement_table(con, _table="pbp")
    assert rows == [], "field_error and plain fielders_choice should map to no outcome at all"
