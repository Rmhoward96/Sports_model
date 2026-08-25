-- =============================================================================
-- NFL P4: sport column migration for game_predictions / prop_predictions.
-- =============================================================================
-- Idempotent — safe to re-run. Run this in the Supabase SQL Editor BEFORE the
-- first NFL producer run so the board can filter NFL rows in these tables
-- (board_picks/picks already carry a `sport` column from the serving layer).

-- NFL P4: sport column on the prediction tables (board_picks/picks already have it).
ALTER TABLE game_predictions ADD COLUMN IF NOT EXISTS sport TEXT DEFAULT 'mlb';
ALTER TABLE prop_predictions  ADD COLUMN IF NOT EXISTS sport TEXT DEFAULT 'mlb';
UPDATE game_predictions SET sport = 'mlb' WHERE sport IS NULL;
UPDATE prop_predictions  SET sport = 'mlb' WHERE sport IS NULL;
CREATE INDEX IF NOT EXISTS idx_game_predictions_sport ON game_predictions (sport, game_date);
CREATE INDEX IF NOT EXISTS idx_prop_predictions_sport ON prop_predictions (sport, game_date);
