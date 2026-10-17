-- =============================================================================
-- Scoreboard restart: the published record starts Oct 14, 2026 for NFL, CFB, MLB.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- From Oct 14 the site recommends only what survives the shrink (spread / total
-- leans at least a quarter point off the line after shrinking toward the market)
-- and the +EV board's picks (9 books, power de-vig vs Pinnacle, same-pull
-- prices). The record restarts on that date so it counts only results under
-- that method. NOTHING is deleted: the `*_all` views keep the full history,
-- and the site's Track Record keeps it under "Archive (before Oct 14)".
--
-- Rollback (the previous fences):
--   UPDATE track_record_start SET starts_at = '2026-09-29 17:27:10+00', model_version = 'nfl-sim-ml-v2',
--     note = 'NFL record restarted with ML v2 (week 4 onward); earlier games (Elo weeks 1-3, one ML v1 game) are in the *_all views',
--     updated_at = now() WHERE sport = 'nfl';
--   UPDATE track_record_start SET starts_at = '2026-10-06 04:00:00+00', model_version = 'mlb-hybrid-v1',
--     note = 'MLB reactivated for the 2026 postseason', updated_at = now() WHERE sport = 'mlb';
--   DELETE FROM track_record_start WHERE sport = 'cfb';      -- CFB had no fence before
--   REFRESH MATERIALIZED VIEW prediction_pnl_daily;

INSERT INTO track_record_start (sport, starts_at, model_version, note)
VALUES
  ('nfl', '2026-10-14 04:00:00+00', 'nfl-sim-ml-v2',
   'Record restarted Oct 14: only shrunk spread / total leans and +EV picks (9 books, power de-vig) count; earlier results are in the *_all views'),
  ('cfb', '2026-10-14 04:00:00+00', 'cfb-ratings-v2',
   'Record restarted Oct 14: only shrunk spread / total leans and +EV picks (9 books, power de-vig) count; earlier results are in the *_all views'),
  ('mlb', '2026-10-14 04:00:00+00', 'mlb-hybrid-v1',
   'Record restarted Oct 14: only shrunk spread / total leans and +EV picks (9 books, power de-vig) count; earlier results are in the *_all views')
ON CONFLICT (sport) DO UPDATE
  SET starts_at = EXCLUDED.starts_at, model_version = EXCLUDED.model_version, note = EXCLUDED.note, updated_at = now();

-- prediction_pnl_daily is a materialized view filtered by in_track_record: rebuild it on the new fence.
REFRESH MATERIALIZED VIEW prediction_pnl_daily;

SELECT sport, starts_at, model_version FROM track_record_start ORDER BY sport;
