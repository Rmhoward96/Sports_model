-- =============================================================================
-- +EV desk-driven pilot: per-market edge/EV board + graded results, plus the
-- view the front-end reads (upcoming board). See
-- .superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.

-- One row per (sport, game_pk, market, side, model_version): the pilot's
-- computed edge/EV for one market side of one game, as produced by
-- scripts/build_ev_board.py (sportsmodel.serving.ev_pilot). Re-running the
-- same model_version for the same (game, market, side) overwrites the row's
-- fields in place (ON CONFLICT DO UPDATE in db.upsert_ev_picks); created_at is
-- set once on first insert and is never reassigned, so it reflects when the
-- row was FIRST posted, not last recomputed.
CREATE TABLE IF NOT EXISTS ev_picks (
    sport             TEXT NOT NULL,
    game_pk           BIGINT NOT NULL,
    market            TEXT NOT NULL,
    side              TEXT NOT NULL,
    model_version     TEXT NOT NULL,
    matchup           TEXT,
    commence_time     TIMESTAMPTZ,
    base_prob         DOUBLE PRECISION,
    true_prob         DOUBLE PRECISION,
    edge              DOUBLE PRECISION,
    desk_delta        DOUBLE PRECISION,
    conviction_tier   TEXT,
    pinnacle_price    INTEGER,
    ev_pinnacle       DOUBLE PRECISION,
    best_book         TEXT,
    best_price        INTEGER,
    ev_best           DOUBLE PRECISION,
    is_pick           BOOLEAN,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk, market, side, model_version)
);

-- Line-shopping columns, added after the initial CREATE TABLE -- IF NOT EXISTS
-- keeps this idempotent on an already-existing ev_picks table.
-- best_line_implied: raw implied prob of the best soft price actually bettable.
-- soft_vs_sharp_gap: base_prob (Pinnacle no-vig) minus best_line_implied;
-- positive means the best soft price is a bargain vs the sharp fair line.
ALTER TABLE ev_picks ADD COLUMN IF NOT EXISTS best_line_implied DOUBLE PRECISION;
ALTER TABLE ev_picks ADD COLUMN IF NOT EXISTS soft_vs_sharp_gap DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_ev_picks_commence ON ev_picks (sport, commence_time);

-- One row per (sport, game_pk, market, side): the graded outcome of that
-- board row, written by the grading job. Idempotent on
-- (sport, game_pk, market, side) -- a re-grade overwrites in place and bumps
-- graded_at.
CREATE TABLE IF NOT EXISTS ev_results (
    sport             TEXT NOT NULL,
    game_pk           BIGINT NOT NULL,
    market            TEXT NOT NULL,
    side              TEXT NOT NULL,
    won               BOOLEAN,
    clv               DOUBLE PRECISION,
    graded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk, market, side)
);

-- Upcoming board with the pilot's latest row per (game, market, side), for
-- the front-end. ev_picks carries one row per (game_pk, market, side,
-- model_version); DISTINCT ON picks the most-recently-created version per
-- (sport, game_pk, market, side) so a re-run doesn't produce duplicate rows
-- for the same market. Only games that haven't kicked off yet. is_pick is
-- included so the page can show picks vs. passes.
--
-- best_line_implied/soft_vs_sharp_gap are appended at the END of the select
-- list (not interleaved next to best_price/ev_best) so CREATE OR REPLACE VIEW
-- can add them in place -- Postgres allows appending output columns via
-- CREATE OR REPLACE, but errors 42P16 ("cannot change name/type of an
-- existing column") if you try to insert or reorder columns mid-list (see the
-- DROP VIEW comment in migration_prediction_tool.sql for the case where that
-- DOES require a DROP first).
CREATE OR REPLACE VIEW ev_current AS
  SELECT DISTINCT ON (sport, game_pk, market, side)
    sport,
    game_pk,
    market,
    side,
    model_version,
    matchup,
    commence_time,
    base_prob,
    true_prob,
    edge,
    desk_delta,
    conviction_tier,
    pinnacle_price,
    ev_pinnacle,
    best_book,
    best_price,
    ev_best,
    is_pick,
    best_line_implied,
    soft_vs_sharp_gap
  FROM ev_picks
  WHERE commence_time > now()
  ORDER BY sport, game_pk, market, side, created_at DESC, commence_time ASC;

-- Public read-only access for the browser (anon key), mirroring
-- migration_decision_desk.sql.
ALTER TABLE ev_picks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_picks" ON ev_picks;
CREATE POLICY "public read ev_picks" ON ev_picks FOR SELECT USING (true);

ALTER TABLE ev_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_results" ON ev_results;
CREATE POLICY "public read ev_results" ON ev_results FOR SELECT USING (true);

GRANT SELECT ON ev_picks, ev_current, ev_results TO anon, authenticated;
