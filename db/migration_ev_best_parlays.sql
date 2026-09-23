-- +EV Parlays: up to 3 three-leg tickets per build, each at one book, and their
-- graded results. Idempotent. Run in the Supabase SQL Editor.
CREATE TABLE IF NOT EXISTS ev_best_parlays (
    parlay_id       TEXT PRIMARY KEY,         -- book|sorted leg keys
    sport           TEXT,                     -- nfl | cfb | mixed
    book            TEXT NOT NULL,
    legs            JSONB NOT NULL,           -- [{key,kind,sport,game_pk,market,side,line,prob,price,label,matchup,commence_time,player_id,player_name}]
    n_legs          INTEGER,
    parlay_dec      DOUBLE PRECISION,
    parlay_price    INTEGER,
    true_prob       DOUBLE PRECISION,
    ev              DOUBLE PRECISION,
    first_commence  TIMESTAMPTZ,              -- locks at the first leg's kickoff
    last_commence   TIMESTAMPTZ,
    model_version   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ev_best_parlays_first ON ev_best_parlays (first_commence);

CREATE TABLE IF NOT EXISTS ev_best_parlay_results (
    parlay_id       TEXT PRIMARY KEY,
    sport           TEXT,
    book            TEXT,
    parlay_price    INTEGER,
    first_commence  TIMESTAMPTZ,
    result          TEXT,                     -- win | loss | push
    pnl             DOUBLE PRECISION,         -- $10 stake
    payout_dec      DOUBLE PRECISION,
    legs            JSONB,                    -- ticket legs, each with "result"
    graded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW ev_best_parlays_current AS
  SELECT parlay_id, sport, book, legs, n_legs, parlay_dec, parlay_price, true_prob, ev,
         first_commence, last_commence, created_at
  FROM ev_best_parlays
  WHERE first_commence > now();

-- ev_pnl (migration_pnl_views.sql) + graded parlays, market 'parlay'.
CREATE OR REPLACE VIEW ev_pnl AS
  SELECT r.sport, r.game_pk, r.market, r.side, p.commence_time,
         CASE WHEN r.won IS NULL THEN 'push' WHEN r.won THEN 'win' ELSE 'loss' END AS result,
         CASE WHEN r.won IS NULL THEN 0.0
              WHEN r.won THEN 10 * (american_to_decimal(p.best_price) - 1)
              ELSE -10.0 END AS pnl
  FROM ev_results r
  JOIN LATERAL (
    SELECT commence_time, best_price FROM ev_picks e
    WHERE e.game_pk = r.game_pk AND e.market = r.market AND e.side = r.side AND e.is_pick
    ORDER BY e.created_at DESC LIMIT 1
  ) p ON true
  WHERE r.won IS NOT NULL OR p.best_price IS NOT NULL
  UNION ALL
  SELECT sport, game_pk, 'prop', side, commence_time,
         CASE result WHEN 'win' THEN 'win' WHEN 'loss' THEN 'loss' ELSE 'push' END,
         10 * profit
  FROM ev_prop_results
  WHERE result IS NOT NULL AND profit IS NOT NULL
  UNION ALL
  SELECT sport, NULL::bigint, 'parlay', NULL::text, first_commence, result, pnl
  FROM ev_best_parlay_results;

ALTER TABLE ev_best_parlays ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_best_parlays" ON ev_best_parlays;
CREATE POLICY "public read ev_best_parlays" ON ev_best_parlays FOR SELECT USING (true);
ALTER TABLE ev_best_parlay_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_best_parlay_results" ON ev_best_parlay_results;
CREATE POLICY "public read ev_best_parlay_results" ON ev_best_parlay_results FOR SELECT USING (true);
GRANT SELECT ON ev_best_parlays, ev_best_parlays_current, ev_best_parlay_results, ev_pnl, ev_pnl_daily
  TO anon, authenticated;
