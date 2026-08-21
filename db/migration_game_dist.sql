-- =============================================================================
-- Migration: probability + EV framing for totals + spread (run once in Supabase
-- SQL Editor).
-- =============================================================================
-- Adds the model's raw total-runs and home-away-margin distributions to
-- game_predictions so P(over) and P(cover) can be evaluated at the BOOK's actual
-- line/spread (not a fixed default). Safe to re-run (IF NOT EXISTS).

ALTER TABLE game_predictions ADD COLUMN IF NOT EXISTS total_dist  JSONB;
ALTER TABLE game_predictions ADD COLUMN IF NOT EXISTS margin_dist JSONB;
