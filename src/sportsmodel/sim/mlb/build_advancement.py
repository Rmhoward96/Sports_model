"""Derive empirical base-out advancement tables from Statcast play-by-play.

For each (outcome, starting base occupancy, outs) we tally the resulting
(end occupancy, runs scored, outs added) by comparing a PA's pre-state to the
NEXT PA's pre-state within the same half-inning, and the batting-team score delta.
Only in-play outcomes need a table; BB/K/HR are deterministic in the kernel.
"""
from __future__ import annotations

import duckdb

from sportsmodel import transforms

# Statcast `events` -> our outcome code (only the in-play ones we tabulate).
_EVENT_TO_OUTCOME = {
    "single": "p_1b",
    "double": "p_2b",
    "triple": "p_3b",
    # every other batted-ball out / fielders choice / DP / sac counts as a generic out
}
_OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "force_out", "sac_fly", "sac_bunt",
    "fielders_choice", "fielders_choice_out", "double_play", "field_error",
    "sac_fly_double_play", "triple_play",
}

_base_occ_expr = (
    "((CASE WHEN on_1b IS NOT NULL THEN 1 ELSE 0 END) "
    "+ (CASE WHEN on_2b IS NOT NULL THEN 2 ELSE 0 END) "
    "+ (CASE WHEN on_3b IS NOT NULL THEN 4 ELSE 0 END))"
)


def build_advancement_table(con: duckdb.DuckDBPyConnection, _table: str | None = None) -> list[dict]:
    """Transition rows respecting the active cutoff. `_table` overrides the source
    (tests pass a small in-memory table); production reads the Statcast parquet."""
    src = _table or f"read_parquet('{transforms._PARQUET_GLOB}')"
    cutoff = ""
    if _table is None and transforms._CUTOFF:  # same gate the other builders use
        cutoff = f"WHERE CAST(game_date AS DATE) < DATE '{transforms._CUTOFF}'"

    # 1) order PAs within each half-inning; read this PA's pre-state + the NEXT PA's
    #    pre-state (LEAD) + the batting-team score delta.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _adv_seq AS
        WITH pa AS (
            SELECT game_pk, inning, inning_topbot, at_bat_number, events,
                   {_base_occ_expr} AS occ,
                   COALESCE(post_bat_score - bat_score, 0) AS runs,
                   on_1b, on_2b, on_3b
            FROM {src} {cutoff}
        ),
        seq AS (
            SELECT *,
                   LEAD(occ) OVER w AS next_occ,
                   ROW_NUMBER() OVER w AS pa_idx,
                   COUNT(*) OVER (PARTITION BY game_pk, inning, inning_topbot) AS n_pa
            FROM pa
            WINDOW w AS (PARTITION BY game_pk, inning, inning_topbot ORDER BY at_bat_number)
        )
        SELECT * FROM seq
    """)

    # 2) map events -> outcome; compute end_occ (0 if this PA ended the half-inning),
    #    outs_added from runners-lost accounting isn't reliable, so derive outs_added
    #    from base+score bookkeeping: outs_added = (runners_before + 1) - runners_after - runs.
    #    runners_before = popcount(occ), runners_after = popcount(end_occ).
    rows = con.execute("""
        WITH mapped AS (
            SELECT
                CASE
                    WHEN events IN ('single') THEN 'p_1b'
                    WHEN events IN ('double') THEN 'p_2b'
                    WHEN events IN ('triple') THEN 'p_3b'
                    WHEN events IN ('field_out','grounded_into_double_play','force_out',
                                    'sac_fly','sac_bunt','fielders_choice','fielders_choice_out',
                                    'double_play','field_error','sac_fly_double_play','triple_play')
                        THEN 'p_out'
                    ELSE NULL
                END AS outcome,
                occ,
                runs,
                CASE WHEN pa_idx = n_pa THEN 0 ELSE COALESCE(next_occ, 0) END AS end_occ
            FROM _adv_seq
        ),
        counted AS (
            SELECT outcome, occ, end_occ, runs, count(*) AS c
            FROM mapped
            WHERE outcome IS NOT NULL
            GROUP BY outcome, occ, end_occ, runs
        ),
        tot AS (
            SELECT outcome, occ, sum(c) AS n FROM counted GROUP BY outcome, occ
        )
        SELECT c.outcome, c.occ, c.end_occ, c.runs, c.c::DOUBLE / t.n AS prob
        FROM counted c JOIN tot t USING (outcome, occ)
        ORDER BY c.outcome, c.occ, c.end_occ, c.runs
    """).fetchall()

    # NOTE: outs are collapsed across out-states in v1 (occ ignores out count) to keep the
    # table dense; `outs` fixed to a nominal 0 and `outs_added` derived in the kernel from
    # runner bookkeeping. This keeps every (outcome, occ) group well-populated.
    out = []
    for outcome, occ, end_occ, runs, prob in rows:
        out.append({"outcome": outcome, "occ": int(occ), "outs": 0,
                    "end_occ": int(end_occ), "runs": int(runs),
                    "outs_added": None, "prob": float(prob)})
    return out
