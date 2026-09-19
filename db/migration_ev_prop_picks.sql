-- =============================================================================
-- NFL player-props +EV board: per-player-market edge/EV board + graded
-- results, plus the view the front-end reads (upcoming board). Mirrors
-- migration_ev_pilot.sql (ev_picks/ev_results) but keyed on
-- (game_pk, player_id, market, line, model_version). See
-- .superpowers/sdd/2026-09-19-nfl-props-productization-C/.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.

-- One row per (game_pk, player_id, market, line, model_version): the sim's
-- computed edge/EV for one player prop side, as produced by the props +EV
-- board build. Re-running the same model_version for the same
-- (game, player, market, line) overwrites the row's fields in place
-- (ON CONFLICT DO UPDATE in db.upsert_ev_prop_picks); created_at is set once
-- on first insert and is never reassigned, so it reflects when the row was
-- FIRST posted, not last recomputed. open_pinnacle_price is likewise frozen
-- on first insert (see below) so CLV grading measures the pick-time price.
CREATE TABLE IF NOT EXISTS ev_prop_picks (
    sport                TEXT,
    game_pk              BIGINT NOT NULL,
    player_id            TEXT NOT NULL,
    player_name          TEXT,
    market               TEXT NOT NULL,
    side                 TEXT,
    line                 DOUBLE PRECISION NOT NULL,
    model_version        TEXT NOT NULL,
    matchup              TEXT,
    commence_time        TIMESTAMPTZ,
    model_prob           DOUBLE PRECISION,
    market_prob          DOUBLE PRECISION,
    edge                 DOUBLE PRECISION,
    ev_best              DOUBLE PRECISION,
    best_book            TEXT,
    best_price           INTEGER,
    pinnacle_price       INTEGER,
    open_pinnacle_price  INTEGER,
    is_pick              BOOLEAN,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market, line, model_version)
);

CREATE INDEX IF NOT EXISTS idx_ev_prop_picks_commence ON ev_prop_picks (sport, commence_time);

-- Upcoming board with the latest row per (game_pk, player_id, market, line),
-- for the front-end. ev_prop_picks carries one row per (game_pk, player_id,
-- market, line, model_version); DISTINCT ON picks the most-recently-created
-- version per (game_pk, player_id, market, line) so a re-run doesn't produce
-- duplicate rows for the same player/market/line. Only games that haven't
-- kicked off yet. is_pick is included so the page can show picks vs. passes
-- (mirrors ev_current in migration_ev_pilot.sql).
CREATE OR REPLACE VIEW ev_prop_picks_current AS
  SELECT DISTINCT ON (game_pk, player_id, market, line)
    sport,
    game_pk,
    player_id,
    player_name,
    market,
    side,
    line,
    model_version,
    matchup,
    commence_time,
    model_prob,
    market_prob,
    edge,
    ev_best,
    best_book,
    best_price,
    pinnacle_price,
    open_pinnacle_price,
    is_pick
  FROM ev_prop_picks
  WHERE commence_time > now()
  ORDER BY game_pk, player_id, market, line, created_at DESC, commence_time ASC;

-- One row per (game_pk, player_id, market, line, model_version): the graded
-- outcome of that board row, written by the grading job. Idempotent on the
-- same key -- a re-grade overwrites in place (ON CONFLICT DO UPDATE in
-- db.upsert_ev_prop_results). created_at is set once on first insert and
-- never reassigned (same treatment as ev_prop_picks.created_at).
CREATE TABLE IF NOT EXISTS ev_prop_results (
    sport             TEXT,
    game_pk           BIGINT NOT NULL,
    player_id         TEXT NOT NULL,
    player_name       TEXT,
    market            TEXT NOT NULL,
    side              TEXT,
    line              DOUBLE PRECISION NOT NULL,
    model_version     TEXT NOT NULL,
    commence_time     TIMESTAMPTZ,
    model_prob        DOUBLE PRECISION,
    novig_close       DOUBLE PRECISION,
    actual            DOUBLE PRECISION,
    result            TEXT,
    clv               DOUBLE PRECISION,
    profit            DOUBLE PRECISION,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market, line, model_version)
);

-- Public read-only access for the browser (anon key), mirroring
-- migration_ev_pilot.sql / migration_decision_desk.sql.
ALTER TABLE ev_prop_picks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_prop_picks" ON ev_prop_picks;
CREATE POLICY "public read ev_prop_picks" ON ev_prop_picks FOR SELECT USING (true);

ALTER TABLE ev_prop_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_prop_results" ON ev_prop_results;
CREATE POLICY "public read ev_prop_results" ON ev_prop_results FOR SELECT USING (true);

GRANT SELECT ON ev_prop_picks, ev_prop_picks_current, ev_prop_results TO anon, authenticated;
