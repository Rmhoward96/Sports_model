-- =============================================================================
-- +EV auto-parlays: juiced-favorite legs (straight price <= -250) recombined
-- into a single +EV parlay ticket. See sportsmodel.serving.parlay and
-- scripts/build_ev_board.py. Idempotent -- safe to re-run in the Supabase SQL Editor.
-- =============================================================================

-- One row per (sport, parlay_id, model_version). parlay_id is a stable hash of
-- the leg set (sorted "game_pk:market:side"), so a rebuild with the same legs
-- overwrites in place; a changed leg set is a new row (the old one is demoted
-- to is_pick=false by db.demote_stale_parlays). legs is the full ticket as
-- JSONB: [{game_pk, market, side, matchup, price, true_prob, commence_time}].
CREATE TABLE IF NOT EXISTS ev_parlays (
    sport           TEXT NOT NULL,
    parlay_id       TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    legs            JSONB,
    book            TEXT,
    parlay_price    INTEGER,          -- combined American price at `book`
    true_prob       DOUBLE PRECISION, -- product of leg true probs
    ev              DOUBLE PRECISION, -- true_prob * parlay_decimal - 1
    n_legs          INTEGER,
    commence_time   TIMESTAMPTZ,      -- latest leg kickoff (parlay is live until then)
    is_pick         BOOLEAN,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, parlay_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_ev_parlays_commence ON ev_parlays (sport, commence_time);

-- Graded parlay outcome: won only if EVERY leg won. Idempotent on
-- (sport, parlay_id); a re-grade overwrites and bumps graded_at.
CREATE TABLE IF NOT EXISTS ev_parlay_results (
    sport           TEXT NOT NULL,
    parlay_id       TEXT NOT NULL,
    won             BOOLEAN,
    graded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, parlay_id)
);

-- Upcoming parlay board: the latest is_pick parlay per (sport, parlay_id) that
-- hasn't fully kicked off yet.
CREATE OR REPLACE VIEW ev_parlays_current AS
  SELECT DISTINCT ON (sport, parlay_id)
    sport, parlay_id, model_version, legs, book, parlay_price,
    true_prob, ev, n_legs, commence_time, is_pick
  FROM ev_parlays
  WHERE is_pick = true AND commence_time > now()
  ORDER BY sport, parlay_id, created_at DESC;

-- Public read-only access for the browser (anon key).
ALTER TABLE ev_parlays ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_parlays" ON ev_parlays;
CREATE POLICY "public read ev_parlays" ON ev_parlays FOR SELECT USING (true);

ALTER TABLE ev_parlay_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_parlay_results" ON ev_parlay_results;
CREATE POLICY "public read ev_parlay_results" ON ev_parlay_results FOR SELECT USING (true);

GRANT SELECT ON ev_parlays, ev_parlays_current, ev_parlay_results TO anon, authenticated;
