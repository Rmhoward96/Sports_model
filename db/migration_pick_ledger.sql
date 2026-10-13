-- =============================================================================
-- Pick ledger + pre-kick close capture (profit project, step 1).
-- =============================================================================
-- Idempotent and additive only -- safe to re-run. Run this in the Supabase SQL Editor BEFORE the new jobs
-- first run (until then every ledger job warns and exits 0; nothing existing changes).
--
--   pick_ledger          APPEND-ONLY log: one row per OBSERVATION of a pick (first sighting, then again only
--                        when the price / line moves or a probability moves >= 1 percentage point). Written
--                        best-effort by the pick writers. A trigger refuses UPDATE / DELETE / TRUNCATE and
--                        the write privileges are revoked from every role except the table owner.
--   close_capture        one row per (sport, game_pk, kind) recording a pre-kick closing-price capture
--                        (kind = lines | props). Written by scripts/capture_close.py. A 'failed' attempt may be
--                        superseded by the next one; an 'ok' or 'empty' row is final and never replaced.
--   pick_ledger_results  one row per (pick_key, basis): the pick graded from its FIRST ledger row (the
--                        earliest bettable observation) and from its LAST pre-kick row. Written nightly by
--                        scripts/grade_ledger.py.

CREATE TABLE IF NOT EXISTS pick_ledger (
  id               bigserial PRIMARY KEY,
  pick_key         text NOT NULL,                 -- deterministic key of the bet slot, never the price or line
  source           text NOT NULL CHECK (source IN ('model_game', 'ev_game', 'ev_prop', 'parlay_leg')),
  sport            text NOT NULL,
  game_pk          bigint NOT NULL,
  market           text NOT NULL,
  side             text NOT NULL,
  line             double precision,              -- the SIDE's own line (home spread, total, prop line); NULL for moneyline
  player_id        text,
  player_name      text,
  parlay_id        text,
  model_version    text,
  model_prob       double precision,              -- model P(this side wins at `line`); NULL when the source has none
  best_price       integer,                       -- American price at the best book, as seen at observed_at
  best_book        text,
  ref_prob         double precision,              -- fair probability of this side at `line` per the writer
  ref_kind         text CHECK (ref_kind IS NULL OR ref_kind IN ('pinnacle_novig', 'us_consensus_novig')),
  ref_books        integer,
  ev_pct           double precision,              -- EV in percent, as the source computed it
  observed_at      timestamptz NOT NULL,
  commence_time    timestamptz NOT NULL,
  minutes_to_kick  double precision NOT NULL,
  writer           text NOT NULL,                 -- script name
  writer_run       text NOT NULL,                 -- GitHub run id, or 'local'
  recorded_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pick_ledger_game_idx ON pick_ledger (sport, game_pk);
-- UNIQUE so a retried insert of the same observation is a no-op (ON CONFLICT DO NOTHING).
CREATE UNIQUE INDEX IF NOT EXISTS pick_ledger_key_observed_idx ON pick_ledger (pick_key, observed_at);

CREATE OR REPLACE FUNCTION pick_ledger_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'pick_ledger is append-only: % is not allowed', TG_OP
    USING ERRCODE = 'restrict_violation';
END;
$$;

DROP TRIGGER IF EXISTS pick_ledger_no_update_delete ON pick_ledger;
CREATE TRIGGER pick_ledger_no_update_delete
  BEFORE UPDATE OR DELETE ON pick_ledger
  FOR EACH ROW EXECUTE FUNCTION pick_ledger_append_only();

DROP TRIGGER IF EXISTS pick_ledger_no_truncate ON pick_ledger;
CREATE TRIGGER pick_ledger_no_truncate
  BEFORE TRUNCATE ON pick_ledger
  FOR EACH STATEMENT EXECUTE FUNCTION pick_ledger_append_only();

CREATE TABLE IF NOT EXISTS close_capture (
  sport            text NOT NULL,
  game_pk          bigint NOT NULL,
  event_id         text,                          -- The Odds API event id
  kind             text NOT NULL CHECK (kind IN ('lines', 'props')),
  captured_at      timestamptz NOT NULL,          -- equals captured_at of the odds_snapshot rows it stored
  commence_time    timestamptz NOT NULL,
  minutes_to_kick  double precision NOT NULL,
  credits_used     integer NOT NULL DEFAULT 0,
  status           text NOT NULL CHECK (status IN ('ok', 'empty', 'failed')),
  regions          text,                          -- e.g. 'us,eu'
  markets          text,
  updated_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_pk, kind)
);

CREATE TABLE IF NOT EXISTS pick_ledger_results (
  pick_key               text NOT NULL,
  basis                  text NOT NULL CHECK (basis IN ('first', 'last')),
  ledger_id              bigint NOT NULL,
  source                 text NOT NULL,
  sport                  text NOT NULL,
  game_pk                bigint NOT NULL,
  market                 text NOT NULL,
  side                   text NOT NULL,
  line                   double precision,
  player_id              text,
  player_name            text,
  best_price             integer,
  observed_at            timestamptz NOT NULL,
  commence_time          timestamptz NOT NULL,
  minutes_to_kick        double precision NOT NULL,
  result                 text NOT NULL CHECK (result IN ('W', 'L', 'P')),
  units_at_pick          double precision,        -- 1 unit staked at best_price (-110 for model spread/total picks)
  close_kind             text NOT NULL CHECK (close_kind IN ('pinnacle_novig', 'us_consensus_novig', 'missing')),
  close_fair_prob        double precision,
  close_line             double precision,
  close_same_line        boolean,
  close_books            integer,
  close_minutes_to_kick  double precision,
  clv_prob               double precision,        -- close fair prob - implied prob of best_price, same line only
  clv_pts                double precision,        -- line points gained vs the close (spreads / totals / props)
  beat_close             boolean,
  graded_at              timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (pick_key, basis)
);

CREATE INDEX IF NOT EXISTS pick_ledger_results_game_idx ON pick_ledger_results (sport, game_pk);
CREATE INDEX IF NOT EXISTS pick_ledger_results_commence_idx ON pick_ledger_results (commence_time);

ALTER TABLE pick_ledger ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read pick_ledger" ON pick_ledger;
CREATE POLICY "public read pick_ledger" ON pick_ledger FOR SELECT USING (true);
ALTER TABLE close_capture ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read close_capture" ON close_capture;
CREATE POLICY "public read close_capture" ON close_capture FOR SELECT USING (true);
ALTER TABLE pick_ledger_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read pick_ledger_results" ON pick_ledger_results;
CREATE POLICY "public read pick_ledger_results" ON pick_ledger_results FOR SELECT USING (true);

-- Defence in depth: the API roles can read these tables and nothing else (the jobs write as the DB owner).
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.pick_ledger, public.close_capture, public.pick_ledger_results FROM anon, authenticated;
GRANT SELECT ON pick_ledger, close_capture, pick_ledger_results TO anon, authenticated;

-- The ledger is append-only for EVERY non-owner role: also strip UPDATE / DELETE / TRUNCATE from PUBLIC and
-- from the other Supabase API role (service_role) when it exists. The owner is held back by the trigger.
REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM service_role';
  END IF;
END;
$$;
