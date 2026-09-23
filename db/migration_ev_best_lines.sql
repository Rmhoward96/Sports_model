-- =============================================================================
-- ev_best_lines: the spread/total NUMBER at the best book for each current
-- +EV game-line pick, for the site's +EV page.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- ev_current carries best_book + best_price but not the line, and odds_snapshot
-- is RLS-locked from the browser (no anon policy). This view exposes only the
-- one line per current pick -- the best book's latest captured row for that
-- side, preferring the row at the pick's price -- not the odds table itself.
CREATE OR REPLACE VIEW ev_best_lines AS
  SELECT e.sport, e.game_pk, e.market, e.side, e.best_book, l.line
  FROM ev_current e
  JOIN LATERAL (
    SELECT o.line FROM odds_snapshot o
    WHERE o.game_pk = e.game_pk AND o.market = e.market AND o.side = e.side
      AND o.book = e.best_book AND o.line IS NOT NULL
    ORDER BY (o.price = e.best_price) DESC, o.captured_at DESC
    LIMIT 1
  ) l ON true
  WHERE e.is_pick AND e.market <> 'moneyline';

GRANT SELECT ON ev_best_lines TO anon, authenticated;
