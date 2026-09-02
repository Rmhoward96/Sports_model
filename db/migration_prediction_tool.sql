-- =============================================================================
-- Prediction-tool reframe: accuracy tracking (winner-correct, margin error)
-- instead of betting ROI. See .superpowers/sdd/2026-09-01-prediction-tool/.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.

-- One row per graded game: the model's pre-game prediction alongside the
-- actual outcome, plus derived accuracy fields. Written by the (future)
-- grading job; idempotent on (sport, game_pk) so a re-grade overwrites in place.
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    sport             TEXT NOT NULL,
    game_pk           BIGINT NOT NULL,
    game_date         DATE,
    home_team_name    TEXT,
    away_team_name    TEXT,
    win_prob          DOUBLE PRECISION,  -- model's HOME win probability, pre-game
    predicted_winner  TEXT,
    actual_winner     TEXT,
    winner_correct    BOOLEAN,
    pred_margin       DOUBLE PRECISION,  -- predicted home - away
    actual_margin     DOUBLE PRECISION,  -- actual home - away
    margin_error      DOUBLE PRECISION,  -- abs(pred_margin - actual_margin)
    pred_total        DOUBLE PRECISION,
    actual_total      DOUBLE PRECISION,
    total_error       DOUBLE PRECISION,  -- abs(pred_total - actual_total)
    spread_covered    BOOLEAN,           -- model favorite covered its predicted spread
    total_over        BOOLEAN,           -- actual total went over the model's total
    graded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk)
);

CREATE INDEX IF NOT EXISTS idx_prediction_accuracy_date ON prediction_accuracy (sport, game_date);

-- Added after the initial deploy (safe to re-run): market-style grade columns.
ALTER TABLE prediction_accuracy ADD COLUMN IF NOT EXISTS spread_covered BOOLEAN;
ALTER TABLE prediction_accuracy ADD COLUMN IF NOT EXISTS total_over BOOLEAN;

-- Calibration check: among games where the model was X% confident, was it
-- actually right X% of the time? win_prob is the HOME win prob, so confidence
-- is whichever side the model favored -- greatest(win_prob, 1 - win_prob).
CREATE OR REPLACE VIEW accuracy_by_confidence AS
  SELECT
    sport,
    CASE
      WHEN greatest(win_prob, 1 - win_prob) >= 0.8 THEN '80-100'
      WHEN greatest(win_prob, 1 - win_prob) >= 0.7 THEN '70-80'
      WHEN greatest(win_prob, 1 - win_prob) >= 0.6 THEN '60-70'
      ELSE '50-60'
    END AS conf_tier,
    count(*) AS games,
    round((avg(winner_correct::int) * 100)::numeric, 1) AS winner_pct,       -- moneyline accuracy
    round(avg(margin_error)::numeric, 1) AS avg_margin_error,
    round((avg(spread_covered::int) * 100)::numeric, 1) AS spread_cover_pct,  -- favorite covered model spread
    round(avg(total_error)::numeric, 1) AS avg_total_error,
    round((avg(total_over::int) * 100)::numeric, 1) AS total_over_pct         -- went over model total
  FROM prediction_accuracy
  WHERE win_prob IS NOT NULL AND winner_correct IS NOT NULL
  GROUP BY sport, conf_tier
  ORDER BY sport, conf_tier;

-- Upcoming slate with the model's current pick, for the front-end "today's
-- predictions" view. game_predictions carries one row per (game_pk,
-- model_version); DISTINCT ON picks the most-recently-generated version per
-- game so a re-run doesn't produce duplicate rows for the same matchup.
CREATE OR REPLACE VIEW predictions_current AS
  SELECT DISTINCT ON (sport, game_pk)
    sport,
    game_pk,
    game_date,
    home_team_name,
    away_team_name,
    home_win_prob,
    pred_home_score,
    pred_away_score
  FROM game_predictions
  WHERE game_date >= current_date
  ORDER BY sport, game_pk, generated_at DESC;

-- Public read-only access for the browser (anon key), mirroring board_picks/picks.
ALTER TABLE prediction_accuracy ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read prediction_accuracy" ON prediction_accuracy;
CREATE POLICY "public read prediction_accuracy" ON prediction_accuracy FOR SELECT USING (true);
GRANT SELECT ON prediction_accuracy, accuracy_by_confidence, predictions_current TO anon, authenticated;
