-- =============================================================================
-- "Take these" list: best +EV prices at the list books vs Pinnacle's de-vigged fair, plus the CLV test that
-- decides whether the list has an edge.
-- =============================================================================
-- Idempotent and additive only -- safe to re-run. Run this in the Supabase SQL Editor BEFORE setting the repo variable
-- TAKE_THESE=on (until then every take job warns and exits 0; nothing existing changes).
--
--   take_events      one row per upcoming/played game across the 8 leagues (teams, start, MLB/NFL/CFB game_pk when
--                    matched, final score once /scores reports it).
--   take_prices      raw prices from every pull: (pulled_at, event, market, side, book). Pruned after 7 days.
--   take_picks       APPEND-ONLY ledger: one row per (event, market, side), written the FIRST time a list book beats
--                    Pinnacle's fair by >= 1.5% EV. A trigger refuses UPDATE / DELETE / TRUNCATE and the write
--                    privileges are revoked from every role except the table owner.
--   take_pick_state  the live list: one row per (event, market, side) with the current best book/price, the other
--                    books that also qualify, and whether the pick is still active (dropped_at when it vanished).
--   take_closes      the latest pre-start Pinnacle snapshot per (event, market) used to grade CLV.
--   take_results     one row per graded pick: CLV or line direction vs the close, then W/L/P and units after /scores.
--   take_api_calls   credit log for every paid Odds API call (job, league, endpoint, credits).
--   take_verdict     APPEND-ONLY single row (id = 1): the one-time CLV test verdict. Judged once.
--   take_list_v / take_test_v   read-only views the site and the report use.
--
-- Rollback (comment only; run by hand if ever needed, dependency order):
--   DROP VIEW take_list_v, take_test_v;
--   DROP TABLE take_verdict, take_results, take_closes, take_pick_state, take_picks, take_prices, take_events, take_api_calls;
--   DROP FUNCTION take_append_only();

CREATE TABLE IF NOT EXISTS take_events (
  event_id       text PRIMARY KEY,                -- The Odds API event id
  league         text NOT NULL CHECK (league IN ('nfl', 'cfb', 'mlb', 'nhl', 'nba', 'wnba', 'epl', 'mls')),
  home_team      text NOT NULL,
  away_team      text NOT NULL,
  commence_time  timestamptz NOT NULL,
  game_pk        bigint,                          -- the model's game id when the event matched one (NULL otherwise)
  home_score     integer,
  away_score     integer,
  completed      boolean NOT NULL DEFAULT false,
  updated_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS take_events_commence_idx ON take_events (commence_time);

CREATE TABLE IF NOT EXISTS take_prices (
  pulled_at    timestamptz NOT NULL,
  event_id     text NOT NULL,
  market       text NOT NULL,
  side         text NOT NULL,
  book         text NOT NULL,
  league       text NOT NULL,
  line         double precision,                  -- the SIDE's own line (NULL for moneyline)
  price        integer NOT NULL,                  -- American price
  last_update  timestamptz,                       -- the book's own last_update for this market
  PRIMARY KEY (pulled_at, event_id, market, side, book)
);

CREATE INDEX IF NOT EXISTS take_prices_pulled_idx ON take_prices (pulled_at);

CREATE TABLE IF NOT EXISTS take_picks (
  id           bigserial PRIMARY KEY,
  event_id     text NOT NULL,
  market       text NOT NULL CHECK (market IN ('ml', 'spread', 'total')),
  side         text NOT NULL CHECK (side IN ('home', 'away', 'draw', 'over', 'under')),
  league       text NOT NULL CHECK (league IN ('nfl', 'cfb', 'mlb', 'nhl', 'nba', 'wnba', 'epl', 'mls')),
  book         text NOT NULL,                     -- best list book at the first sighting
  line         double precision,                  -- the SIDE's own line at that book (NULL for moneyline)
  price        integer NOT NULL,                  -- American price at that book
  ev           double precision NOT NULL,         -- fair_prob * decimal(price) - 1
  fair_prob    double precision NOT NULL,         -- Pinnacle's no-vig probability for this side
  good_until   integer NOT NULL,                  -- the worst American price still at +1.5% EV
  better_line  boolean NOT NULL,                  -- the book's line is better than Pinnacle's (not the same line)
  ref_line     double precision,                  -- Pinnacle's own line for this side (NULL for moneyline)
  pin_prices   jsonb NOT NULL,                    -- Pinnacle's side -> {line, price} at the first sighting
  first_seen   timestamptz NOT NULL,              -- the pull time
  run_id       text NOT NULL,                     -- GitHub run id, or 'local'
  UNIQUE (event_id, market, side)
);

CREATE OR REPLACE FUNCTION take_append_only() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION '% is append-only: % is not allowed', TG_TABLE_NAME, TG_OP
    USING ERRCODE = 'restrict_violation';
END;
$$;

DROP TRIGGER IF EXISTS take_picks_no_update_delete ON take_picks;
CREATE TRIGGER take_picks_no_update_delete
  BEFORE UPDATE OR DELETE ON take_picks
  FOR EACH ROW EXECUTE FUNCTION take_append_only();

DROP TRIGGER IF EXISTS take_picks_no_truncate ON take_picks;
CREATE TRIGGER take_picks_no_truncate
  BEFORE TRUNCATE ON take_picks
  FOR EACH STATEMENT EXECUTE FUNCTION take_append_only();

CREATE TABLE IF NOT EXISTS take_pick_state (
  event_id     text NOT NULL,
  market       text NOT NULL CHECK (market IN ('ml', 'spread', 'total')),
  side         text NOT NULL CHECK (side IN ('home', 'away', 'draw', 'over', 'under')),
  league       text NOT NULL,
  book         text NOT NULL,                     -- current best list book
  line         double precision,
  price        integer NOT NULL,
  ev           double precision NOT NULL,
  fair_prob    double precision NOT NULL,
  good_until   integer NOT NULL,
  better_line  boolean NOT NULL,
  ref_line     double precision,                  -- Pinnacle's own line for this side (NULL for moneyline)
  others       jsonb NOT NULL DEFAULT '[]'::jsonb,  -- the other qualifying books: [{book, line, price, ev}], ev desc
  first_seen   timestamptz NOT NULL,
  last_seen    timestamptz NOT NULL,
  active       boolean NOT NULL DEFAULT true,
  dropped_at   timestamptz,
  PRIMARY KEY (event_id, market, side)
);

CREATE TABLE IF NOT EXISTS take_closes (
  event_id         text NOT NULL,
  market           text NOT NULL,
  prices           jsonb NOT NULL,                -- side -> {line, price} (Pinnacle)
  captured_at      timestamptz NOT NULL,
  minutes_to_start double precision NOT NULL,
  PRIMARY KEY (event_id, market)
);

CREATE TABLE IF NOT EXISTS take_results (
  pick_id       bigint PRIMARY KEY REFERENCES take_picks (id),
  graded_by     text NOT NULL CHECK (graded_by IN ('clv', 'direction', 'no_close')),
  close_fair    double precision,                 -- Pinnacle's closing no-vig probability for the pick's side
  clv           double precision,                 -- close_fair * decimal(price) - 1 (graded_by = 'clv' only)
  beat          boolean,                          -- clv > 0, or the line moved toward the pick (direction)
  close_line    double precision,
  close_minutes double precision,
  graded_at     timestamptz NOT NULL DEFAULT now(),
  outcome       text CHECK (outcome IN ('win', 'loss', 'push')),
  units         double precision,                 -- 1 unit staked at the pick's price
  settled_at    timestamptz
);

CREATE TABLE IF NOT EXISTS take_api_calls (
  id                bigserial PRIMARY KEY,
  called_at         timestamptz NOT NULL DEFAULT now(),
  job               text NOT NULL CHECK (job IN ('pull', 'close', 'scores')),
  league            text,
  endpoint          text,
  credits_used      integer,
  credits_remaining integer,
  books_missing     text[],
  run_id            text
);

CREATE TABLE IF NOT EXISTS take_verdict (
  id         smallint PRIMARY KEY CHECK (id = 1),
  judged_at  timestamptz NOT NULL,
  n          integer NOT NULL,
  beat_rate  double precision NOT NULL,
  mean_clv   double precision NOT NULL,
  lower      double precision NOT NULL,           -- cluster-bootstrap 95% lower bound on mean CLV
  passed     boolean NOT NULL,
  criteria   jsonb NOT NULL,
  pick_ids   bigint[] NOT NULL
);

DROP TRIGGER IF EXISTS take_verdict_no_update_delete ON take_verdict;
CREATE TRIGGER take_verdict_no_update_delete
  BEFORE UPDATE OR DELETE ON take_verdict
  FOR EACH ROW EXECUTE FUNCTION take_append_only();

DROP TRIGGER IF EXISTS take_verdict_no_truncate ON take_verdict;
CREATE TRIGGER take_verdict_no_truncate
  BEFORE TRUNCATE ON take_verdict
  FOR EACH STATEMENT EXECUTE FUNCTION take_append_only();

-- The live list: current state joined to the event; a dropped pick stays for 24 h after it dropped (or, if it
-- never dropped, after it started).
CREATE OR REPLACE VIEW take_list_v WITH (security_invoker = true) AS
SELECT s.event_id, s.market, s.side, s.league, s.book, s.line, s.price, s.ev, s.fair_prob, s.good_until,
       s.better_line, s.ref_line, s.others, s.first_seen, s.last_seen, s.active, s.dropped_at,
       e.home_team, e.away_team, e.commence_time, e.game_pk
FROM take_pick_state s
JOIN take_events e ON e.event_id = s.event_id
WHERE s.active OR coalesce(s.dropped_at, e.commence_time) > now() - interval '24 hours';

-- Everything the CLV report and the site's test page read: the pick, its event, its (optional) grade.
CREATE OR REPLACE VIEW take_test_v WITH (security_invoker = true) AS
SELECT p.id, p.event_id, p.league, p.market, p.side, p.book, p.line, p.price, p.ev, p.fair_prob, p.better_line,
       p.ref_line, p.first_seen, e.home_team, e.away_team, e.commence_time,
       r.graded_by, r.clv, r.beat, r.close_fair, r.close_line, r.close_minutes, r.graded_at, r.outcome, r.units
FROM take_picks p
JOIN take_events e ON e.event_id = p.event_id
LEFT JOIN take_results r ON r.pick_id = p.id;

ALTER TABLE take_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_events" ON take_events;
CREATE POLICY "public read take_events" ON take_events FOR SELECT USING (true);
ALTER TABLE take_prices ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_prices" ON take_prices;
CREATE POLICY "public read take_prices" ON take_prices FOR SELECT USING (true);
ALTER TABLE take_picks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_picks" ON take_picks;
CREATE POLICY "public read take_picks" ON take_picks FOR SELECT USING (true);
ALTER TABLE take_pick_state ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_pick_state" ON take_pick_state;
CREATE POLICY "public read take_pick_state" ON take_pick_state FOR SELECT USING (true);
ALTER TABLE take_closes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_closes" ON take_closes;
CREATE POLICY "public read take_closes" ON take_closes FOR SELECT USING (true);
ALTER TABLE take_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_results" ON take_results;
CREATE POLICY "public read take_results" ON take_results FOR SELECT USING (true);
ALTER TABLE take_api_calls ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_api_calls" ON take_api_calls;
CREATE POLICY "public read take_api_calls" ON take_api_calls FOR SELECT USING (true);
ALTER TABLE take_verdict ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read take_verdict" ON take_verdict;
CREATE POLICY "public read take_verdict" ON take_verdict FOR SELECT USING (true);

-- Defence in depth: the API roles can read these tables and nothing else (the jobs write as the DB owner).
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON take_picks, take_events, take_prices, take_pick_state, take_closes,
  take_results, take_api_calls, take_verdict FROM anon, authenticated;
GRANT SELECT ON take_picks, take_events, take_prices, take_pick_state, take_closes, take_results, take_api_calls,
  take_verdict TO anon, authenticated;
GRANT SELECT ON take_list_v, take_test_v TO anon, authenticated;

-- The picks and the verdict are append-only for EVERY non-owner role: also strip UPDATE / DELETE / TRUNCATE from
-- PUBLIC and from the other Supabase API role (service_role) when it exists. The owner is held back by the trigger.
REVOKE UPDATE, DELETE, TRUNCATE ON take_picks, take_verdict FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON take_picks, take_verdict FROM service_role';
  END IF;
END;
$$;
