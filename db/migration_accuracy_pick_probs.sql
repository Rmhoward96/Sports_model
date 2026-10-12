-- =============================================================================
-- Track Record "Model Prob." for spread and total rows.
-- =============================================================================
-- prediction_accuracy stored only pred_margin / pred_total, so the Track Record showed "—" for the model's
-- probability on spread and total picks. These two columns hold the model's probability that the PICK WINS at the
-- closing line, computed at grading time from the stored margin / total distribution (the game page's maths:
-- pushes excluded, the rest renormalised):
--   spread_pick_prob  home pick (pred_margin + market_spread > 0): P(margin > -market_spread); away pick: P(margin < -market_spread)
--   total_pick_prob   Over pick (pred_total > market_total): P(total > market_total); Under pick: P(total < market_total)
-- NULL when the game has no stored distribution, no market line, or the model sat exactly on the line.
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor, THEN run
-- scripts/backfill_accuracy_pick_probs.py for the rows that are already graded.
-- Grading and the site both work before this runs (they skip / fall back to the old column list).
ALTER TABLE prediction_accuracy ADD COLUMN IF NOT EXISTS spread_pick_prob DOUBLE PRECISION;
ALTER TABLE prediction_accuracy ADD COLUMN IF NOT EXISTS total_pick_prob  DOUBLE PRECISION;

-- Make PostgREST (the site's API) pick the new columns up immediately.
NOTIFY pgrst, 'reload schema';
