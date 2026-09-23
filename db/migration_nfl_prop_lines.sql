-- nfl_prop_lines: the book line + best over/under price for EVERY sim player
-- projection that has a line (no usage gate, no +EV filter), and
-- nfl_prop_grades: those rows graded vs nfl_player_actuals. Idempotent.
CREATE TABLE IF NOT EXISTS nfl_prop_lines (
    game_pk        BIGINT NOT NULL,
    player_id      TEXT   NOT NULL,
    player_name    TEXT,
    team           TEXT,
    market         TEXT   NOT NULL,   -- pass_yds|rush_yds|rec_yds|receptions|rush_att
    line           DOUBLE PRECISION NOT NULL,
    over_price     INTEGER,
    over_book      TEXT,
    under_price    INTEGER,
    under_book     TEXT,
    projection     DOUBLE PRECISION,
    p_over         DOUBLE PRECISION,
    lean           TEXT,
    n_books        INTEGER,
    commence_time  TIMESTAMPTZ,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market)
);
CREATE INDEX IF NOT EXISTS idx_nfl_prop_lines_commence ON nfl_prop_lines (commence_time);

CREATE OR REPLACE VIEW nfl_prop_grades AS
  SELECT l.game_pk, l.player_id, l.player_name, l.team, l.market, l.line,
         l.over_price, l.under_price, l.projection, l.p_over, l.lean,
         l.commence_time, a.actual, a.season, a.week,
         CASE WHEN a.actual = l.line THEN 'push'
              WHEN (l.lean = 'over'  AND a.actual > l.line)
                OR (l.lean = 'under' AND a.actual < l.line) THEN 'hit'
              ELSE 'miss' END            AS result,
         abs(l.projection - a.actual)    AS sim_err,
         abs(l.line - a.actual)          AS line_err,
         l.projection - a.actual         AS sim_bias
  FROM nfl_prop_lines l
  JOIN nfl_player_actuals a
    ON a.game_pk = l.game_pk AND a.player_id = l.player_id AND a.market = l.market
  WHERE l.commence_time <= now();

ALTER TABLE nfl_prop_lines ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_prop_lines" ON nfl_prop_lines;
CREATE POLICY "public read nfl_prop_lines" ON nfl_prop_lines FOR SELECT USING (true);
GRANT SELECT ON nfl_prop_lines, nfl_prop_grades TO anon, authenticated;

-- Pre-aggregated per-market (+ overall) accuracy counts, so the front-end
-- doesn't have to page through nfl_prop_grades itself -- PostgREST caps
-- unordered/limited reads at 1000 rows, which silently truncated the
-- accuracy section to an arbitrary subset once grades passed ~2 weeks.
CREATE OR REPLACE VIEW nfl_prop_accuracy AS
  SELECT COALESCE(market, 'all')                                AS market,
         count(*)                                               AS n,
         count(*) FILTER (WHERE result = 'hit')                 AS hits,
         count(*) FILTER (WHERE result IN ('hit','miss'))       AS decided,
         avg(sim_err)                                           AS sim_mae,
         avg(line_err)                                          AS line_mae,
         avg(sim_bias)                                          AS bias
  FROM nfl_prop_grades
  GROUP BY ROLLUP (market);
GRANT SELECT ON nfl_prop_accuracy TO anon, authenticated;
