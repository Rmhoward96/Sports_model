"""db/migration_ev_record_ledger.sql: the +EV game-line track record is sourced from the append-only pick
ledger. Text/regex assertions only (no Postgres harness in this repo)."""
import re
from pathlib import Path

DB_DIR = Path(__file__).parent.parent / "db"
MIGRATION = DB_DIR / "migration_ev_record_ledger.sql"
PARLAYS = DB_DIR / "migration_ev_best_parlays.sql"
CUTOFF = "2026-10-14T18:00:00Z"

VIEW_COLS = ["sport", "game_pk", "market", "side", "matchup", "commence_time", "best_price", "best_book",
             "ev_best", "base_prob", "won", "clv", "result", "pnl", "graded_at", "first_shown_at"]


def sql():
    return MIGRATION.read_text()


def view_body(text, name):
    m = re.search(rf"CREATE OR REPLACE VIEW {name} AS(.*?);\s*(?:\n--|\nGRANT|\nCREATE|\Z)", text, re.DOTALL)
    assert m, f"no CREATE OR REPLACE VIEW {name}"
    return m.group(1)


def tail_from(text, marker):
    """Everything from `marker` to the end of that CREATE VIEW statement."""
    i = text.index(marker)
    return text[i:text.index(";", i)]


def test_file_exists_and_is_idempotent():
    assert MIGRATION.exists()
    t = sql()
    assert "CREATE OR REPLACE VIEW ev_game_record" in t
    assert not re.search(r"\bDROP\s+(VIEW|TABLE)\b", t, re.I)
    assert not re.search(r"\b(INSERT|UPDATE|DELETE)\b\s+(INTO|FROM)?\s*(pick_ledger|ev_)", t, re.I)


def test_view_columns_in_order():
    body = view_body(sql(), "ev_game_record")
    first = body.split("UNION ALL")[0]
    aliases = re.findall(r"\bAS\s+(\w+)\s*,?\s*$", first, re.M)
    assert aliases == VIEW_COLS, aliases


def test_cutoff_and_ledger_filters():
    t = sql()
    assert t.count(CUTOFF) >= 4, "cutoff on the legacy branch and on observed_at / commence_time in the ledger branch"
    assert re.search(r"p\.commence_time\s*<\s*'2026-10-14T18:00:00Z'", t)
    assert re.search(r"plr\.observed_at\s*>=\s*'2026-10-14T18:00:00Z'", t)
    assert re.search(r"plr\.commence_time\s*>=\s*'2026-10-14T18:00:00Z'", t)
    assert "plr.source = 'ev_game'" in t
    assert "plr.basis = 'first'" in t
    assert "plr.units_at_pick IS NOT NULL" in t
    assert "FROM pick_ledger_results plr" in t
    assert "JOIN pick_ledger pl ON pl.id = f.ledger_id" in t


def test_one_bet_per_market_earliest_side_wins():
    t = sql()
    assert "DISTINCT ON (plr.sport, plr.game_pk, plr.market)" in t
    assert re.search(r"ORDER BY plr\.sport, plr\.game_pk, plr\.market, plr\.observed_at ASC, plr\.side", t)


def test_ledger_branch_mapping():
    t = sql()
    assert "CASE f.result WHEN 'W' THEN true WHEN 'L' THEN false ELSE NULL END" in t
    assert "CASE f.result WHEN 'W' THEN 'win' WHEN 'L' THEN 'loss' ELSE 'push' END" in t
    assert "10 * f.units_at_pick" in t
    assert "pl.ev_pct / 100.0" in t and "pl.ref_prob" in t and "pl.best_book" in t
    assert "f.clv_prob" in t and "f.observed_at" in t and "f.graded_at" in t
    # matchup: latest ev_picks row for the game by created_at, any side / any is_pick
    m = re.search(r"LEFT JOIN LATERAL \(\s*SELECT e\.matchup FROM ev_picks e(.*?)\) m ON true", t, re.DOTALL)
    assert m and "is_pick" not in m.group(1) and "ORDER BY e.created_at DESC LIMIT 1" in m.group(1)


def test_legacy_branch_is_the_old_ev_pnl_logic():
    t = sql()
    assert "FROM ev_results r" in t
    assert "e.game_pk = r.game_pk AND e.market = r.market AND e.side = r.side AND e.is_pick" in t
    assert "ORDER BY e.created_at DESC LIMIT 1" in t
    assert "WHERE (r.won IS NOT NULL OR p.best_price IS NOT NULL)" in t
    assert "WHEN r.won THEN 10 * (american_to_decimal(p.best_price) - 1)" in t
    assert "ELSE -10.0 END" in t


def test_ev_pnl_redefined_with_identical_prop_and_parlay_branches():
    new = tail_from(sql(), "CREATE OR REPLACE VIEW ev_pnl AS")
    old = tail_from(PARLAYS.read_text(), "CREATE OR REPLACE VIEW ev_pnl AS")
    assert "FROM ev_game_record" in new
    assert "ev_results" not in new

    def rest(s):
        return s[s.index("UNION ALL"):]
    # the prop + parlay UNION ALL branches stay byte-identical to the previous definition
    assert rest(old).split("UNION ALL", 1)[1] == rest(new).split("UNION ALL", 1)[1]


def test_ev_pnl_column_names_and_types_match():
    new = tail_from(sql(), "CREATE OR REPLACE VIEW ev_pnl AS")
    head = new.split("FROM ev_game_record")[0]
    for frag in ("sport::text AS sport", "game_pk::bigint AS game_pk", "market::text AS market", "side::text AS side",
                 "commence_time::timestamptz AS commence_time", "result::text AS result", "pnl::double precision AS pnl"):
        assert frag in head, frag


def test_dependent_views_unchanged_columns():
    # ev_pnl_daily / ev_pnl_daily_all read only these ev_pnl columns, all still present
    track = (DB_DIR / "migration_track_record_start.sql").read_text()
    for name in ("ev_pnl_daily", "ev_pnl_daily_all"):
        body = view_body(track, name)
        assert "FROM ev_pnl" in body
        used = set(re.findall(r"\b(commence_time|sport|market|result|pnl|side|game_pk)\b", body))
        assert used <= {"commence_time", "sport", "market", "result", "pnl", "side", "game_pk"}


def test_grants_to_anon_and_authenticated():
    t = sql()
    m = re.search(r"GRANT SELECT ON ([\w, ]+) TO anon, authenticated;", t)
    assert m and "ev_game_record" in m.group(1)
