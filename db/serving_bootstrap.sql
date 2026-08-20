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

-- Game-level predictions written by scripts/generate_predictions.py. Denormalized
-- (carries team + pitcher names) so it's readable directly in the Table Editor.
CREATE TABLE IF NOT EXISTS game_predictions (
    game_pk                     BIGINT NOT NULL,
    model_version               TEXT NOT NULL,
    generated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    game_date                   DATE,
    home_team_name              TEXT,
    away_team_name              TEXT,
    home_probable_pitcher_name  TEXT,
    away_probable_pitcher_name  TEXT,
    pred_home_score             REAL,
    pred_away_score             REAL,
    pred_total                  REAL,
    pred_margin                 REAL,
    home_win_prob               REAL,
    PRIMARY KEY (game_pk, model_version)
);

CREATE INDEX IF NOT EXISTS idx_game_predictions_date ON game_predictions (game_date);

-- Player-prop projections written by scripts/generate_props.py. One row per
-- (game, player, market). projected_mean is the number to compare to the book line;
-- prob_over is P(over the standard line). lineup_source = confirmed|projected.
CREATE TABLE IF NOT EXISTS prop_predictions (
    game_pk        BIGINT NOT NULL,
    player_id      BIGINT NOT NULL,
    market         TEXT NOT NULL,          -- hits | total_bases | home_run
    model_version  TEXT NOT NULL,
    generated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    game_date      DATE,
    player_name    TEXT,
    team_name      TEXT,
    batting_slot   INTEGER,
    projected_pa   REAL,
    lineup_source  TEXT,
    projected_mean REAL,
    line           REAL,
    prob_over      REAL,
    PRIMARY KEY (game_pk, player_id, market, model_version)
);

CREATE INDEX IF NOT EXISTS idx_prop_predictions_date ON prop_predictions (game_date);

-- Odds snapshots from The Odds API (scripts/ingest_odds.py). captured_at is part of
-- the key so repeated pulls accumulate line movement; the CLOSING line is the last
-- snapshot before commence_time. Game lines have player_name = '' (empty). Props carry
-- the player name (joined to our player_id / prop_predictions by name at analysis time).
CREATE TABLE IF NOT EXISTS odds_snapshot (
    game_pk       BIGINT NOT NULL,
    market        TEXT NOT NULL,          -- moneyline | total | spread | (prop codes)
    side          TEXT NOT NULL,          -- home | away | over | under
    player_name   TEXT NOT NULL DEFAULT '',
    book          TEXT NOT NULL,
    line          REAL,                   -- total/spread/prop line (NULL for moneyline)
    price         INTEGER,                -- American odds
    commence_time TIMESTAMPTZ,
    captured_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (game_pk, market, side, player_name, book, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_odds_snapshot_game ON odds_snapshot (game_pk, market);
