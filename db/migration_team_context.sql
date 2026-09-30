-- =============================================================================
-- Team context (NFL + CFB): game log, history, matchup grades, power rankings.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- Written daily by scripts/build_team_context.py (.github/workflows/
-- build-team-context.yml) with idempotent upserts on each primary key;
-- computed_at is bumped on every rewrite. Everything here is DESCRIPTIVE --
-- not a pick. Every value for a game uses only games that kicked off before it.
--
--   team_game_log   one row per team per game (current + 2 prior seasons), played
--                   or upcoming. team_line = the team's closing spread from its own
--                   side, + = favored by that many; covers iff margin > team_line.
--                   CFB non-FBS opponents are the pooled team 'FCS'.
--   team_history    per upcoming game (next 8 days) and side: L5/L10/L20/season
--                   windows, streaks, home/away + fav/dog splits (JSON).
--   matchup_grades  per upcoming game and OFFENSE side: A-F overall / pass / run
--                   vs the opponent's defense, scores, percentiles, early flag
--                   (either team < 3 games), unit components (JSON).
--   power_rankings  per (sport, season, week): rating = expected margin vs an
--                   average team on a neutral field; past weeks are kept for the
--                   weekly move. power_rankings_current = the latest week per sport.

CREATE TABLE IF NOT EXISTS team_game_log (
  sport        text NOT NULL,             -- nfl | cfb
  season       integer NOT NULL,
  week         integer NOT NULL,
  game_key     text NOT NULL,             -- NFL: nflverse game_id; CFB: ESPN game id
  game_pk      bigint,                    -- ESPN game id (NFL: nflverse `espn`)
  kickoff      timestamptz NOT NULL,
  date_et      date,
  team         text NOT NULL,
  opponent     text NOT NULL,
  venue        text,                      -- home | away | neutral
  pf           double precision,          -- NULL until played
  pa           double precision,
  margin       double precision,
  team_line    double precision,          -- + = team favored by that many
  total_line   double precision,
  role         text,                      -- fav | dog | pick | NULL (no line)
  su           text,                      -- W | L | T
  ats          text,                      -- W | L | P
  ou           text,                      -- O | U | P
  game_type    text,                      -- NFL REG/WC/DIV/CON/SB; CFB REG/POST
  is_post      boolean,
  team_is_fbs  boolean,
  opp_is_fbs   boolean,
  computed_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_key, team)
);
CREATE INDEX IF NOT EXISTS team_game_log_team_idx ON team_game_log (sport, team, kickoff);

CREATE TABLE IF NOT EXISTS team_history (
  sport        text NOT NULL,
  game_pk      bigint NOT NULL,
  side         text NOT NULL CHECK (side IN ('home', 'away')),
  team         text NOT NULL,
  opponent     text,
  season       integer,
  week         integer,
  kickoff      timestamptz,
  windows      jsonb NOT NULL,            -- {L5, L10, L20, season: {su, ats, ou, n, n_ats, n_ou, avg_*}}
  streaks      jsonb,                     -- {su, ats, ou, notes: {k: "U6 of 7"}}
  splits       jsonb,                     -- {home, away, fav, dog: {su, ats, n}}
  computed_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_pk, side)
);

CREATE TABLE IF NOT EXISTS matchup_grades (
  sport          text NOT NULL,
  game_pk        bigint NOT NULL,
  side           text NOT NULL CHECK (side IN ('home', 'away')),   -- the offense's side
  team           text NOT NULL,                                    -- the offense
  opponent       text,                                             -- the defense
  season         integer,
  week           integer,
  kickoff        timestamptz,
  overall        text,                    -- A-F; NULL when ungraded (FCS / no ratings)
  pass           text,
  run            text,
  overall_score  double precision,
  pass_score     double precision,
  run_score      double precision,
  overall_pct    double precision,        -- percentile vs the previous 3 seasons
  pass_pct       double precision,
  run_pct        double precision,
  early          boolean,                 -- either team < 3 games this season
  units          jsonb,                   -- {games, opp_games, pass_rate, off: {...}, def: {...}}
  computed_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_pk, side)
);

CREATE TABLE IF NOT EXISTS power_rankings (
  sport        text NOT NULL,
  season       integer NOT NULL,
  week         integer NOT NULL,          -- the upcoming week the ranking is for
  team         text NOT NULL,
  rank         integer NOT NULL,
  rating       double precision,          -- points vs an average team, neutral field
  prev_rank    integer,                   -- NULL: no previous week / new team
  move         integer,                   -- prev_rank - rank (+ = moved up)
  units        jsonb,                     -- {pass_off|run_off|pass_def|run_def: {rank, epa}}
  sos          double precision,
  su           text,
  ats          text,
  games        integer,
  conf         text,                      -- CFB: ESPN conference id
  computed_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, season, week, team)
);

CREATE OR REPLACE VIEW power_rankings_current WITH (security_invoker = true) AS
  SELECT p.*
  FROM power_rankings p
  JOIN (SELECT DISTINCT ON (sport) sport, season, week
        FROM power_rankings
        ORDER BY sport, season DESC, week DESC) l
    ON l.sport = p.sport AND l.season = p.season AND l.week = p.week;

ALTER TABLE team_game_log ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read team_game_log" ON team_game_log;
CREATE POLICY "public read team_game_log" ON team_game_log FOR SELECT USING (true);
ALTER TABLE team_history ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read team_history" ON team_history;
CREATE POLICY "public read team_history" ON team_history FOR SELECT USING (true);
ALTER TABLE matchup_grades ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read matchup_grades" ON matchup_grades;
CREATE POLICY "public read matchup_grades" ON matchup_grades FOR SELECT USING (true);
ALTER TABLE power_rankings ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read power_rankings" ON power_rankings;
CREATE POLICY "public read power_rankings" ON power_rankings FOR SELECT USING (true);

GRANT SELECT ON team_game_log, team_history, matchup_grades, power_rankings,
  power_rankings_current TO anon, authenticated;
