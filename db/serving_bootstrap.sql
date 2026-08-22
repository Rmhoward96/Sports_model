-- Serving layer for the Blue Edge front-end (spec 2026-08-21).
-- Idempotent: safe to re-run. Run in the Supabase SQL Editor.
-- Writers (GitHub Actions via the Session pooler) connect as the table owner and
-- bypass RLS; the anon/authenticated roles get read-only access for the front-end.

-- Live board: one row per game x market (game lines) and per player x market (props),
-- fully refreshed each run with the current best-book price for the model's chosen side.
CREATE TABLE IF NOT EXISTS board_picks (
  sport TEXT, game_pk BIGINT, game_date DATE, commence_time TIMESTAMPTZ, matchup TEXT,
  market TEXT, market_label TEXT, player_id BIGINT, player_name TEXT, team TEXT,
  pick_label TEXT, side TEXT, line REAL,
  odds INT, book TEXT, model_prob REAL, implied_prob REAL, ev REAL, is_pick BOOLEAN,
  generated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (game_pk, market, player_id)
);

-- Bet log / track record: inserted once when a pick first turns +EV (bet price locked),
-- then filled with the outcome + CLV by the grader.
CREATE TABLE IF NOT EXISTS picks (
  game_pk BIGINT, market TEXT, player_id BIGINT,
  sport TEXT, game_date DATE, commence_time TIMESTAMPTZ, matchup TEXT,
  market_label TEXT, player_name TEXT, team TEXT, pick_label TEXT, side TEXT, line REAL,
  bet_odds INT, bet_book TEXT, model_prob REAL, novig_bet REAL, ev_bet REAL,
  bet_at TIMESTAMPTZ DEFAULT now(),
  status TEXT DEFAULT 'pending',
  actual REAL, result TEXT, profit REAL,
  novig_close REAL, clv REAL, graded_at TIMESTAMPTZ,
  PRIMARY KEY (game_pk, market, player_id)
);

-- Track-record segment summary (the "By league & market" table).
CREATE OR REPLACE VIEW track_record_segments AS
  SELECT sport, market,
    count(*) FILTER (WHERE result = 'win')  AS wins,
    count(*) FILTER (WHERE result = 'loss') AS losses,
    count(*) FILTER (WHERE result = 'push') AS pushes,
    round((avg((result = 'win')::int) FILTER (WHERE result IN ('win', 'loss')) * 100)::numeric, 1) AS win_pct,
    round(sum(profit)::numeric, 2) AS units,
    round((sum(profit) / nullif(count(*), 0))::numeric * 100, 1) AS roi,
    round(avg(ev_bet)::numeric * 100, 1) AS avg_ev,
    round(avg(clv)::numeric * 100, 1) AS avg_clv
  FROM picks WHERE status = 'graded'
  GROUP BY sport, market;

-- Cumulative units by week (the track-record chart).
CREATE OR REPLACE VIEW cumulative_units_weekly AS
  SELECT week, units,
         sum(units) OVER (ORDER BY week) AS cumulative_units
  FROM (SELECT date_trunc('week', game_date)::date AS week, sum(profit) AS units
        FROM picks WHERE status = 'graded'
        GROUP BY 1) w
  ORDER BY week;

-- Public read-only access for the browser (anon key).
ALTER TABLE board_picks ENABLE ROW LEVEL SECURITY;
ALTER TABLE picks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read board_picks" ON board_picks;
CREATE POLICY "public read board_picks" ON board_picks FOR SELECT USING (true);
DROP POLICY IF EXISTS "public read picks" ON picks;
CREATE POLICY "public read picks" ON picks FOR SELECT USING (true);
GRANT SELECT ON track_record_segments, cumulative_units_weekly TO anon, authenticated;
