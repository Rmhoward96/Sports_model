-- Results-based power rankings (2026-09-30): strength of victory and home / road
-- records on power_rankings. Run BEFORE merging the results-power PR (the daily
-- build-team-context job writes these columns).
--   sov          mean current rating of the teams beaten this season (NULL: no win)
--   home_record  home W-L[-T] this season (neutral-site games in neither)
--   road_record  road W-L[-T] this season

ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS sov double precision;
ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS home_record text;
ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS road_record text;

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
