-- =============================================================================
-- Migration: put `line` in the odds_snapshot uniqueness (run once in Supabase).
-- =============================================================================
-- The odds_snapshot key omitted `line`, so a book posting alternate lines for the
-- same (game, market, side, player) — e.g. home runs at over 0.5 / 1.5 / 2.5 — all
-- collided on the key and `ON CONFLICT DO NOTHING` kept only one, dropping the rest.
-- Adding `line` (COALESCE for game-line NULLs) lets every distinct line survive.
-- Safe to re-run.

ALTER TABLE odds_snapshot DROP CONSTRAINT IF EXISTS odds_snapshot_pkey;

CREATE UNIQUE INDEX IF NOT EXISTS odds_snapshot_uniq
    ON odds_snapshot (game_pk, market, side, player_name, book, captured_at,
                      (COALESCE(line, -99999::real)));
