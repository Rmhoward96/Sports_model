-- =============================================================================
-- CFB site panels: game hero venue/weather line + weather insight, CFB havoc / turnover insights,
-- and the CFB Projected Game Flow. Descriptive data only -- not picks.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor BEFORE the jobs first run (the site
-- hides each panel while its table is missing or empty, so it can deploy before or after this).
--
--   game_info           one row per (sport, game_pk) for games from 2 days ago to 10 days ahead (CFB + NFL).
--                       Written by scripts/build_game_info.py (.github/workflows/build-game-info.yml).
--                       Weather columns are NULL for indoor games and whenever the source has no reading;
--                       weather_kind: forecast | observed. line_score (CFB, finished games only):
--                       {"home": [q1..q4], "away": [q1..q4]}, overtime excluded. The primary key is the
--                       (sport, game_pk) lookup index the site uses, so no second index is created.
--   cfb_team_insights   one row per (season, team): season-to-date regular-season FBS-vs-FBS havoc,
--                       turnover margin and explosiveness with national ranks (1 = best) among teams with
--                       >= 3 games (n_ranked = size of that field). Written weekly by
--                       scripts/build_cfb_panels.py (build-cfb-advanced.yml, Monday).
--   cfb_quarter_shares  one row per team: share of points scored / allowed in Q1-Q4 over the last two
--                       completed seasons plus this one, shrunk toward the league average; each set of
--                       four sums to 1. Written weekly by the same job.
-- Team ids are ESPN team ids as text (the same ids as team_history.team).

CREATE TABLE IF NOT EXISTS game_info (
  sport          text NOT NULL,              -- nfl | cfb
  game_pk        bigint NOT NULL,            -- ESPN event id (= CFBD game id)
  venue_name     text,
  city           text,
  state          text,
  indoor         boolean,
  temp_f         double precision,
  wind_mph       double precision,
  precip_chance  double precision,           -- 0-100; NULL for CFB (CFBD sends no probability)
  precip_in      double precision,           -- inches; NULL for NFL (ESPN sends none)
  conditions     text,
  weather_kind   text CHECK (weather_kind IN ('forecast', 'observed')),
  source         text NOT NULL CHECK (source IN ('cfbd', 'espn')),
  line_score     jsonb,
  captured_at    timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_pk)
);

CREATE TABLE IF NOT EXISTS cfb_team_insights (
  season                          integer NOT NULL,
  team                            text NOT NULL,
  games                           integer NOT NULL,
  def_havoc_rate                  double precision,
  def_havoc_rank                  integer,
  off_havoc_allowed_rate          double precision,
  off_havoc_allowed_rank          integer,
  turnover_margin                 double precision,
  turnover_margin_per_game        double precision,
  turnover_margin_rank            integer,
  off_explosiveness               double precision,
  off_explosiveness_rank          integer,
  def_explosiveness_allowed       double precision,
  def_explosiveness_allowed_rank  integer,
  n_ranked                        integer,
  through_week                    integer,
  updated_at                      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS cfb_quarter_shares (
  team         text NOT NULL PRIMARY KEY,
  scored_q1    double precision NOT NULL,
  scored_q2    double precision NOT NULL,
  scored_q3    double precision NOT NULL,
  scored_q4    double precision NOT NULL,
  allowed_q1   double precision NOT NULL,
  allowed_q2   double precision NOT NULL,
  allowed_q3   double precision NOT NULL,
  allowed_q4   double precision NOT NULL,
  games_used   integer NOT NULL,
  updated_at   timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE game_info ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read game_info" ON game_info;
CREATE POLICY "public read game_info" ON game_info FOR SELECT USING (true);
ALTER TABLE cfb_team_insights ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read cfb_team_insights" ON cfb_team_insights;
CREATE POLICY "public read cfb_team_insights" ON cfb_team_insights FOR SELECT USING (true);
ALTER TABLE cfb_quarter_shares ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read cfb_quarter_shares" ON cfb_quarter_shares;
CREATE POLICY "public read cfb_quarter_shares" ON cfb_quarter_shares FOR SELECT USING (true);

GRANT SELECT ON game_info, cfb_team_insights, cfb_quarter_shares TO anon, authenticated;
