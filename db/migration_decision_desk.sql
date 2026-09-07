-- =============================================================================
-- CFB decision desk: model picks + graded results, plus the two views the
-- front-end reads (upcoming slate, historical record). See
-- .superpowers/sdd/2026-09-07-cfb-decision-desk/.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.

-- One row per (sport, game_pk, model_version): the desk's pre-game pick for a
-- game, as produced by the writer job. Re-running the same model_version for
-- the same game overwrites the pick fields in place (ON CONFLICT DO UPDATE in
-- db.upsert_desk_picks); created_at is set once on first insert and is never
-- reassigned, so it reflects when the pick was FIRST posted, not last edited.
CREATE TABLE IF NOT EXISTS desk_picks (
    sport             TEXT NOT NULL,
    game_pk           BIGINT NOT NULL,
    model_version     TEXT NOT NULL,
    game_date         DATE,
    commence_time     TIMESTAMPTZ,
    matchup           TEXT,
    ml_pick           TEXT,
    spread_side       TEXT,
    spread_line       DOUBLE PRECISION,
    total_side        TEXT,
    total_line        DOUBLE PRECISION,
    confidence        DOUBLE PRECISION,
    conviction_tier   TEXT,
    rationale         TEXT,
    agent_notes       JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk, model_version)
);

CREATE INDEX IF NOT EXISTS idx_desk_picks_date ON desk_picks (sport, game_date);

-- One row per (sport, game_pk): the graded outcome of that game's pick(s),
-- written by the grading job. Idempotent on (sport, game_pk) -- a re-grade
-- overwrites in place and bumps graded_at.
CREATE TABLE IF NOT EXISTS desk_pick_results (
    sport             TEXT NOT NULL,
    game_pk           BIGINT NOT NULL,
    ml_correct        BOOLEAN,
    spread_cover      BOOLEAN,
    total_result      BOOLEAN,
    clv_spread        DOUBLE PRECISION,
    clv_total         DOUBLE PRECISION,
    graded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk)
);

-- Upcoming slate with the desk's current pick per game, for the front-end.
-- desk_picks carries one row per (game_pk, model_version); DISTINCT ON picks
-- the most-recently-created version per game so a re-run doesn't produce
-- duplicate rows for the same matchup. Only games that haven't kicked off yet.
CREATE OR REPLACE VIEW desk_current AS
  SELECT DISTINCT ON (sport, game_pk)
    sport,
    game_pk,
    model_version,
    game_date,
    commence_time,
    matchup,
    ml_pick,
    spread_side,
    spread_line,
    total_side,
    total_line,
    confidence,
    conviction_tier,
    rationale
  FROM desk_picks
  WHERE commence_time > now()
  ORDER BY sport, game_pk, created_at DESC, commence_time ASC;

-- Historical record of the desk's picks, grouped by sport + conviction tier:
-- how many games have been graded, moneyline accuracy, ATS-vs-line rate,
-- total-vs-line rate, and mean closing-line-value on both markets. avg()
-- ignores NULLs, so ungraded fields (e.g. no spread pick that game) drop out
-- of that column's average rather than dragging it toward zero.
CREATE OR REPLACE VIEW desk_record AS
  SELECT
    p.sport,
    p.conviction_tier,
    count(*) AS games_graded,
    round((avg(r.ml_correct::int) * 100)::numeric, 1) AS ml_accuracy_pct,
    round((avg(r.spread_cover::int) * 100)::numeric, 1) AS ats_pct,
    round((avg(r.total_result::int) * 100)::numeric, 1) AS total_pct,
    round(avg(r.clv_spread)::numeric, 2) AS mean_clv_spread,
    round(avg(r.clv_total)::numeric, 2) AS mean_clv_total
  FROM desk_picks p
  JOIN desk_pick_results r ON r.sport = p.sport AND r.game_pk = p.game_pk
  GROUP BY p.sport, p.conviction_tier
  ORDER BY p.sport, p.conviction_tier;

-- Public read-only access for the browser (anon key), mirroring
-- prediction_accuracy/game_predictions.
ALTER TABLE desk_picks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read desk_picks" ON desk_picks;
CREATE POLICY "public read desk_picks" ON desk_picks FOR SELECT USING (true);

ALTER TABLE desk_pick_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read desk_pick_results" ON desk_pick_results;
CREATE POLICY "public read desk_pick_results" ON desk_pick_results FOR SELECT USING (true);

GRANT SELECT ON desk_picks, desk_pick_results, desk_current, desk_record TO anon, authenticated;
