-- =============================================================================
-- Early-week stale-number monitor (profit project, step 3).
-- =============================================================================
-- Idempotent and additive only -- safe to re-run. Run this in the Supabase SQL Editor BEFORE the new jobs first run
-- (until then every early-monitor job warns and exits 0; nothing existing changes).
--
--   early_tickets         APPEND-ONLY paper tickets: one row per (sport, game_pk, market, side), logged the FIRST time
--                         a US book's number beats Pinnacle's same-poll fair by +1.5% EV with kickoff >= 24 h away.
--                         Written hourly by scripts/early_monitor.py. A trigger refuses UPDATE / DELETE / TRUNCATE and
--                         the write privileges are revoked from every role except the table owner.
--   early_ticket_results  one row per ticket, graded daily by scripts/grade_early_tickets.py against Pinnacle's
--                         pre-kick close (close EV, beat-close, line CLV, W/L/P, units, availability one poll later).
--                         An ordinary upsert table (a re-grade overwrites in place).

CREATE TABLE IF NOT EXISTS early_tickets (
  id              bigserial PRIMARY KEY,
  sport           text NOT NULL CHECK (sport IN ('cfb', 'nfl')),
  game_pk         bigint NOT NULL,
  market          text NOT NULL CHECK (market IN ('spread', 'total')),
  side            text NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
  book            text NOT NULL,                  -- The Odds API bookmaker key of the best US book at the first sighting
  book_line       double precision NOT NULL,      -- the SIDE's own line at that book (home spread, away spread, total)
  price           integer NOT NULL,               -- American price at that book
  pinnacle_line   double precision NOT NULL,      -- Pinnacle's line for the same side, same poll
  pinnacle_fair   double precision NOT NULL,      -- Pinnacle's two-way no-vig probability for this side at its own line
  p_win           double precision NOT NULL,      -- at the BOOK's line, from the landing-table line adjustment
  p_push          double precision NOT NULL,
  p_loss          double precision NOT NULL,
  ev              double precision NOT NULL,      -- p_win * (decimal - 1) - p_loss, per unit staked
  observed_at     timestamptz NOT NULL,           -- the poll's capture time
  commence_time   timestamptz NOT NULL,
  hours_to_kick   double precision NOT NULL,
  poll_run_id     text NOT NULL,                  -- GitHub run id, or 'local'
  recorded_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sport, game_pk, market, side)
);

CREATE INDEX IF NOT EXISTS early_tickets_commence_idx ON early_tickets (commence_time);

CREATE OR REPLACE FUNCTION early_tickets_append_only() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION 'early_tickets is append-only: % is not allowed', TG_OP
    USING ERRCODE = 'restrict_violation';
END;
$$;

DROP TRIGGER IF EXISTS early_tickets_no_update_delete ON early_tickets;
CREATE TRIGGER early_tickets_no_update_delete
  BEFORE UPDATE OR DELETE ON early_tickets
  FOR EACH ROW EXECUTE FUNCTION early_tickets_append_only();

DROP TRIGGER IF EXISTS early_tickets_no_truncate ON early_tickets;
CREATE TRIGGER early_tickets_no_truncate
  BEFORE TRUNCATE ON early_tickets
  FOR EACH STATEMENT EXECUTE FUNCTION early_tickets_append_only();

CREATE TABLE IF NOT EXISTS early_ticket_results (
  ticket_id              bigint PRIMARY KEY REFERENCES early_tickets (id),
  sport                  text NOT NULL,
  game_pk                bigint NOT NULL,
  market                 text NOT NULL,
  side                   text NOT NULL,
  book                   text NOT NULL,
  book_line              double precision NOT NULL,
  price                  integer NOT NULL,
  observed_at            timestamptz NOT NULL,
  commence_time          timestamptz NOT NULL,
  result                 text NOT NULL CHECK (result IN ('W', 'L', 'P')),
  units                  double precision,         -- 1 unit staked at the ticket's price
  close_kind             text NOT NULL CHECK (close_kind IN ('pinnacle_novig', 'missing')),
  close_fair             double precision,         -- Pinnacle's closing no-vig probability for the ticket's side
  close_line             double precision,         -- Pinnacle's closing line for the ticket's side
  close_minutes_to_kick  double precision,
  close_ev               double precision,         -- the ticket's line and price valued at Pinnacle's closing fair (primary)
  beat_close             boolean,                  -- close_ev > 0
  clv_pts                double precision,         -- line points gained vs Pinnacle's closing line, in the ticket side's favour
  available_next_poll    boolean,                  -- same book, same line, same or better price 30 min to 3 h later; NULL outside
  graded_at              timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS early_ticket_results_game_idx ON early_ticket_results (sport, game_pk);

ALTER TABLE early_tickets ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read early_tickets" ON early_tickets;
CREATE POLICY "public read early_tickets" ON early_tickets FOR SELECT USING (true);
ALTER TABLE early_ticket_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read early_ticket_results" ON early_ticket_results;
CREATE POLICY "public read early_ticket_results" ON early_ticket_results FOR SELECT USING (true);

-- Defence in depth: the API roles can read these tables and nothing else (the jobs write as the DB owner).
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.early_tickets, public.early_ticket_results FROM anon, authenticated;
GRANT SELECT ON early_tickets, early_ticket_results TO anon, authenticated;

-- The tickets are append-only for EVERY non-owner role: also strip UPDATE / DELETE / TRUNCATE from PUBLIC and from the
-- other Supabase API role (service_role) when it exists. The owner is held back by the trigger.
REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM service_role';
  END IF;
END;
$$;
