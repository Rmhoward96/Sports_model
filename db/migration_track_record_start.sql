-- =============================================================================
-- Track record restart per sport (NFL restarts with nfl-sim-ml-v2).
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- `track_record_start` holds one row per sport whose public track record
-- starts fresh: games kicking off before `starts_at` are left out of the
-- site's record views (and filtered client-side from the raw tables the site
-- reads). NOTHING is deleted: every graded row stays in its base table, and
-- the `*_all` views keep the full history (the pre-restart definitions) for
-- comparing the old models with the new one. Restarting again later is one
-- UPDATE of starts_at.

CREATE TABLE IF NOT EXISTS track_record_start (
  sport         text PRIMARY KEY,
  starts_at     timestamptz NOT NULL,
  model_version text,
  note          text,
  updated_at    timestamptz NOT NULL DEFAULT now()
);

INSERT INTO track_record_start (sport, starts_at, model_version, note)
VALUES ('nfl', '2026-09-29 17:27:10+00', 'nfl-sim-ml-v2',
        'NFL record restarted with ML v2 (week 4 onward); earlier games (Elo weeks 1-3, one ML v1 game) are in the *_all views')
ON CONFLICT (sport) DO NOTHING;

ALTER TABLE track_record_start ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read track_record_start" ON track_record_start;
CREATE POLICY "public read track_record_start" ON track_record_start FOR SELECT USING (true);
GRANT SELECT ON track_record_start TO anon, authenticated;

-- In the current record? (timestamp form: kickoff; date form: ET game date)
CREATE OR REPLACE FUNCTION in_track_record(p_sport text, p_ts timestamptz)
RETURNS boolean LANGUAGE sql STABLE AS $$
  SELECT p_ts IS NULL OR NOT EXISTS (
    SELECT 1 FROM track_record_start s WHERE s.sport = p_sport AND p_ts < s.starts_at)
$$;

CREATE OR REPLACE FUNCTION in_track_record(p_sport text, p_date date)
RETURNS boolean LANGUAGE sql STABLE AS $$
  SELECT p_date IS NULL OR NOT EXISTS (
    SELECT 1 FROM track_record_start s
    WHERE s.sport = p_sport AND p_date < (s.starts_at AT TIME ZONE 'America/New_York')::date)
$$;

-- ---------------------------------------------------------------------------
-- Full-history copies (the definitions before this migration), for comparison
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW accuracy_by_confidence_all AS
  SELECT sport,
    CASE
      WHEN GREATEST(win_prob, 1 - win_prob) >= 0.8 THEN '80-100'
      WHEN GREATEST(win_prob, 1 - win_prob) >= 0.7 THEN '70-80'
      WHEN GREATEST(win_prob, 1 - win_prob) >= 0.6 THEN '60-70'
      ELSE '50-60'
    END AS conf_tier,
    count(*) AS games,
    round(avg(winner_correct::integer) * 100, 1) AS winner_pct,
    round(avg(margin_error)::numeric, 1) AS avg_margin_error,
    round(avg(spread_pick_correct::integer) * 100, 1) AS spread_ats_pct,
    round(avg(total_error)::numeric, 1) AS avg_total_error,
    round(avg(total_pick_correct::integer) * 100, 1) AS total_pick_pct
  FROM prediction_accuracy
  WHERE win_prob IS NOT NULL AND winner_correct IS NOT NULL
  GROUP BY 1, 2
  ORDER BY 1, 2;

CREATE OR REPLACE VIEW prediction_pnl_daily_all AS
  SELECT game_date, sport, market, count(*) AS n,
    count(*) FILTER (WHERE result = 'win') AS wins,
    count(*) FILTER (WHERE result = 'loss') AS losses,
    count(*) FILTER (WHERE result = 'push') AS pushes,
    sum(pnl) AS pnl
  FROM prediction_pnl
  GROUP BY game_date, sport, market;

CREATE OR REPLACE VIEW ev_pnl_daily_all AS
  SELECT (commence_time AT TIME ZONE 'America/New_York')::date AS game_date, sport, market, count(*) AS n,
    count(*) FILTER (WHERE result = 'win') AS wins,
    count(*) FILTER (WHERE result = 'loss') AS losses,
    count(*) FILTER (WHERE result = 'push') AS pushes,
    sum(pnl) AS pnl
  FROM ev_pnl
  GROUP BY 1, sport, market;

CREATE OR REPLACE VIEW nfl_prop_accuracy_all AS
  SELECT COALESCE(market, 'all') AS market, count(*) AS n,
    count(*) FILTER (WHERE result = 'hit') AS hits,
    count(*) FILTER (WHERE result = ANY (ARRAY['hit', 'miss'])) AS decided,
    avg(sim_err) AS sim_mae, avg(line_err) AS line_mae, avg(sim_bias) AS bias
  FROM nfl_prop_grades
  GROUP BY ROLLUP(market);

CREATE OR REPLACE VIEW nfl_prop_pnl_by_game_all AS
  WITH lean AS (
    SELECT game_pk, max(commence_time) AS commence_time, max(season) AS season, max(week) AS week,
      count(*) AS n,
      count(*) FILTER (WHERE result = 'hit') AS hits,
      count(*) FILTER (WHERE result = 'miss') AS misses,
      count(*) FILTER (WHERE result = 'push') AS pushes,
      count(pnl) AS n_priced, sum(pnl) AS pnl
    FROM nfl_prop_pnl GROUP BY game_pk
  ), ev AS (
    SELECT game_pk, count(*) AS ev_n,
      count(*) FILTER (WHERE result = 'win') AS ev_wins,
      count(*) FILTER (WHERE result = 'loss') AS ev_losses,
      sum(10::double precision * profit) AS ev_pnl
    FROM ev_prop_results WHERE result IS NOT NULL AND profit IS NOT NULL GROUP BY game_pk
  )
  SELECT l.game_pk, l.commence_time, l.season, l.week, l.n, l.hits, l.misses, l.pushes, l.n_priced, l.pnl,
    (SELECT s.matchup FROM nfl_sim s WHERE s.game_pk = l.game_pk ORDER BY s.created_at DESC LIMIT 1) AS matchup,
    COALESCE(ev.ev_n, 0) AS ev_n, COALESCE(ev.ev_wins, 0) AS ev_wins, COALESCE(ev.ev_losses, 0) AS ev_losses,
    ev.ev_pnl
  FROM lean l LEFT JOIN ev ON ev.game_pk = l.game_pk;

-- ---------------------------------------------------------------------------
-- Site views: same columns, current record only
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW accuracy_by_confidence AS
  SELECT * FROM (
    SELECT sport,
      CASE
        WHEN GREATEST(win_prob, 1 - win_prob) >= 0.8 THEN '80-100'
        WHEN GREATEST(win_prob, 1 - win_prob) >= 0.7 THEN '70-80'
        WHEN GREATEST(win_prob, 1 - win_prob) >= 0.6 THEN '60-70'
        ELSE '50-60'
      END AS conf_tier,
      count(*) AS games,
      round(avg(winner_correct::integer) * 100, 1) AS winner_pct,
      round(avg(margin_error)::numeric, 1) AS avg_margin_error,
      round(avg(spread_pick_correct::integer) * 100, 1) AS spread_ats_pct,
      round(avg(total_error)::numeric, 1) AS avg_total_error,
      round(avg(total_pick_correct::integer) * 100, 1) AS total_pick_pct
    FROM prediction_accuracy
    WHERE win_prob IS NOT NULL AND winner_correct IS NOT NULL
      AND in_track_record(sport, game_date)
    GROUP BY 1, 2
  ) t
  ORDER BY sport, conf_tier;

-- prediction_pnl_daily is MATERIALIZED (db/migration_pnl_materialize.sql: the
-- anon read of prediction_pnl times out) and refreshed by grade_predictions.py
-- after each run. Rebuilt here with the record filter; it takes effect for a
-- changed starts_at at the next refresh (or run REFRESH MATERIALIZED VIEW).
-- prediction_pnl_daily_all above is a plain view (slow, full history, for
-- comparisons -- not read by the browser).
DROP MATERIALIZED VIEW IF EXISTS prediction_pnl_daily;
CREATE MATERIALIZED VIEW prediction_pnl_daily AS
  SELECT game_date, sport, market, count(*) AS n,
    count(*) FILTER (WHERE result = 'win') AS wins,
    count(*) FILTER (WHERE result = 'loss') AS losses,
    count(*) FILTER (WHERE result = 'push') AS pushes,
    sum(pnl) AS pnl
  FROM prediction_pnl
  WHERE in_track_record(sport, game_date)
  GROUP BY game_date, sport, market;
REFRESH MATERIALIZED VIEW prediction_pnl_daily;

CREATE OR REPLACE VIEW ev_pnl_daily AS
  SELECT (commence_time AT TIME ZONE 'America/New_York')::date AS game_date, sport, market, count(*) AS n,
    count(*) FILTER (WHERE result = 'win') AS wins,
    count(*) FILTER (WHERE result = 'loss') AS losses,
    count(*) FILTER (WHERE result = 'push') AS pushes,
    sum(pnl) AS pnl
  FROM ev_pnl
  WHERE in_track_record(sport, commence_time)
  GROUP BY 1, sport, market;

CREATE OR REPLACE VIEW nfl_prop_accuracy AS
  SELECT COALESCE(market, 'all') AS market, count(*) AS n,
    count(*) FILTER (WHERE result = 'hit') AS hits,
    count(*) FILTER (WHERE result = ANY (ARRAY['hit', 'miss'])) AS decided,
    avg(sim_err) AS sim_mae, avg(line_err) AS line_mae, avg(sim_bias) AS bias
  FROM nfl_prop_grades
  WHERE in_track_record('nfl', commence_time)
  GROUP BY ROLLUP(market);

CREATE OR REPLACE VIEW nfl_prop_pnl_by_game AS
  SELECT * FROM nfl_prop_pnl_by_game_all WHERE in_track_record('nfl', commence_time);

GRANT SELECT ON accuracy_by_confidence, accuracy_by_confidence_all, prediction_pnl_daily, prediction_pnl_daily_all,
  ev_pnl_daily, ev_pnl_daily_all, nfl_prop_accuracy, nfl_prop_accuracy_all,
  nfl_prop_pnl_by_game, nfl_prop_pnl_by_game_all TO anon, authenticated;
