-- =============================================================================
-- ev_prop_picks_current: only each game's LATEST board build.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- The view used to take the newest row per (game_pk, player_id, market, line)
-- across EVERY build since the lines opened, so a pick a later build no longer
-- produced (e.g. a benched QB's, once his prop was pulled and the sim moved to
-- the new starter) stayed on the board until kickoff. Each build stamps all of
-- a game's rows with one created_at, so the view now keeps only rows from the
-- game's newest created_at. Columns are unchanged.
CREATE OR REPLACE VIEW ev_prop_picks_current AS
  SELECT DISTINCT ON (game_pk, player_id, market, line)
    sport,
    game_pk,
    player_id,
    player_name,
    market,
    side,
    line,
    model_version,
    matchup,
    commence_time,
    model_prob,
    market_prob,
    edge,
    ev_best,
    best_book,
    best_price,
    pinnacle_price,
    open_pinnacle_price,
    is_pick
  FROM ev_prop_picks p
  WHERE commence_time > now()
    AND created_at = (SELECT max(created_at) FROM ev_prop_picks b
                      WHERE b.sport = p.sport AND b.game_pk = p.game_pk)
  ORDER BY game_pk, player_id, market, line, created_at DESC, commence_time ASC;

GRANT SELECT ON ev_prop_picks_current TO anon, authenticated;
