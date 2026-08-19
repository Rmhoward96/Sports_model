-- =============================================================================
-- Serving bootstrap — the small, always-on tables browsed in Supabase.
-- =============================================================================
-- Run this in the Supabase SQL Editor to get a hosted, self-updating result you
-- can look at immediately, BEFORE the full conformed schema (db/schema.sql) and
-- the modeling layer are in place. Everything here is tiny (KB-MB) — safe on the
-- Supabase free tier. Raw Statcast pitch data never lands here.

-- Denormalized daily slate written by scripts/daily_ingest.py on each cron run.
-- game_pk is MLB's native id; the daily job upserts on it (idempotent).
CREATE TABLE IF NOT EXISTS daily_schedule (
    game_pk                     BIGINT PRIMARY KEY,
    game_date                   DATE,
    status                      TEXT,
    venue_id                    INTEGER,
    venue_name                  TEXT,
    home_team_id                INTEGER,
    home_team_name              TEXT,
    away_team_id                INTEGER,
    away_team_name              TEXT,
    home_probable_pitcher_id    INTEGER,
    home_probable_pitcher_name  TEXT,
    away_probable_pitcher_id    INTEGER,
    away_probable_pitcher_name  TEXT,
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_daily_schedule_date ON daily_schedule (game_date);

-- Later: gold_game_predictions and gold_player_prop_predictions (from db/schema.sql)
-- land alongside this and become the tables you actually browse for edges.
