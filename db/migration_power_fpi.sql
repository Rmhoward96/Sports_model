-- CFB power rankings = 60% ESPN FPI + 40% our results-based rating (2026-10-06).
-- Run BEFORE merging feat/cfb-fpi-rankings (the daily build-team-context job writes and
-- reads these columns).
--   fpi           ESPN Football Power Index used in the blend (CFB; NULL: NFL, or no FPI)
--   model_rating  our results-based rating before the blend (CFB; NULL: NFL)
--   rating        CFB: 0.6 * fpi + 0.4 * model_rating (model_rating alone when fpi is NULL)

ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS fpi double precision;
ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS model_rating double precision;

-- p.* is expanded when a view is created: recreate so the view carries the new
-- columns (they are appended at the end, which CREATE OR REPLACE allows).
CREATE OR REPLACE VIEW power_rankings_current WITH (security_invoker = true) AS
  SELECT p.*
  FROM power_rankings p
  JOIN (SELECT DISTINCT ON (sport) sport, season, week
        FROM power_rankings
        ORDER BY sport, season DESC, week DESC) l
    ON l.sport = p.sport AND l.season = p.season AND l.week = p.week;

GRANT SELECT ON power_rankings_current TO anon, authenticated;
