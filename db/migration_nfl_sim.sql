-- =============================================================================
-- NFL sim engine: game-level and player-level simulation outputs, plus the
-- views the front-end reads (upcoming slate). See
-- .superpowers/sdd/2026-09-17-nfl-sim-engine/. NFL only.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.

-- One row per (game_pk, model_version): the sim engine's Monte Carlo output
-- for one game, written by the (future) sim build job. Re-running the same
-- model_version for the same game overwrites the row's fields in place
-- (ON CONFLICT DO UPDATE in db.upsert_nfl_sim); created_at is set once on
-- first insert and is never reassigned, so it reflects when the row was
-- FIRST posted, not last recomputed.
CREATE TABLE IF NOT EXISTS nfl_sim (
    game_pk             BIGINT NOT NULL,
    model_version       TEXT NOT NULL,
    matchup             TEXT,
    commence_time       TIMESTAMPTZ,
    sim_home_win_prob   DOUBLE PRECISION,
    sim_margin          DOUBLE PRECISION,
    sim_total           DOUBLE PRECISION,
    disagreement        DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, model_version)
);

CREATE INDEX IF NOT EXISTS idx_nfl_sim_commence ON nfl_sim (commence_time);

-- Upcoming slate with the sim engine's latest row per game_pk. nfl_sim
-- carries one row per (game_pk, model_version); DISTINCT ON picks the
-- most-recently-created version per game_pk so a re-run doesn't produce
-- duplicate rows for the same matchup. Only games that haven't kicked off yet.
CREATE OR REPLACE VIEW nfl_sim_current AS
  SELECT DISTINCT ON (game_pk)
    game_pk,
    model_version,
    matchup,
    commence_time,
    sim_home_win_prob,
    sim_margin,
    sim_total,
    disagreement
  FROM nfl_sim
  WHERE commence_time > now()
  ORDER BY game_pk, created_at DESC;

-- One row per (game_pk, player_id, market, model_version): the sim engine's
-- per-player prop output, written by the same build job. Re-running the same
-- model_version for the same (game, player, market) overwrites the row's
-- fields in place (ON CONFLICT DO UPDATE in db.upsert_nfl_player_sim);
-- created_at is set once on first insert and is never reassigned.
CREATE TABLE IF NOT EXISTS nfl_player_sim (
    game_pk         BIGINT NOT NULL,
    player_id       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    name            TEXT,
    pos             TEXT,
    team            TEXT,
    market          TEXT NOT NULL,
    mean            DOUBLE PRECISION,
    dist            JSONB,
    commence_time   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market, model_version)
);

CREATE INDEX IF NOT EXISTS idx_nfl_player_sim_commence ON nfl_player_sim (commence_time);

-- Upcoming player-prop slate with the sim engine's latest row per
-- (game_pk, player_id, market). Only games that haven't kicked off yet.
CREATE OR REPLACE VIEW nfl_player_sim_current AS
  SELECT DISTINCT ON (game_pk, player_id, market)
    game_pk,
    player_id,
    model_version,
    name,
    pos,
    team,
    market,
    mean,
    dist,
    commence_time
  FROM nfl_player_sim
  WHERE commence_time > now()
  ORDER BY game_pk, player_id, market, created_at DESC;

-- Public read-only access for the browser (anon key), mirroring
-- migration_ev_parlays.sql.
ALTER TABLE nfl_sim ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_sim" ON nfl_sim;
CREATE POLICY "public read nfl_sim" ON nfl_sim FOR SELECT USING (true);

ALTER TABLE nfl_player_sim ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_player_sim" ON nfl_player_sim;
CREATE POLICY "public read nfl_player_sim" ON nfl_player_sim FOR SELECT USING (true);

GRANT SELECT ON nfl_sim, nfl_sim_current, nfl_player_sim, nfl_player_sim_current
  TO anon, authenticated;
