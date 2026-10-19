-- =============================================================================
-- +EV game-line track record from the append-only pick ledger.
-- =============================================================================
-- Idempotent (CREATE OR REPLACE VIEW) -- safe to re-run. Run this in the Supabase SQL Editor AFTER
-- migration_pick_ledger.sql and migration_ev_best_parlays.sql (it redefines ev_pnl, which that file last defined).
--
-- WHY: scripts/build_ev_board.py re-writes both sides of every market on every build and recomputes
-- ev_picks.is_pick each time, and the grader / ev_pnl only look at is_pick = true rows. A pick that was +EV at
-- noon and lost its edge by 3pm was flipped back to is_pick = false and vanished from the track record, even
-- though the user saw it on the site (since 2026-10-14 the ledger holds 32 distinct ev_game picks; only 3 were
-- in the record).
--
-- FIX: every +EV game-line pick, once shown, counts -- graded at its FIRST-shown price from the append-only
-- ledger (pick_ledger_results, basis 'first'). One bet per (sport, game_pk, market): the side shown FIRST.
--   * Picks first shown before the power de-vig fix (2026-10-14T18:00:00Z) are NOT taken from the ledger.
--   * Games that commenced before that cutoff keep their existing ev_results grading (legacy branch), so the
--     history does not change.
--
--   ev_game_record   one row per graded game-line pick (legacy + ledger branches). The site reads it directly
--                    for the +EV pick performance table; first_shown_at is when the pick was first surfaced.
--   ev_pnl           redefined: the game-line branch now reads ev_game_record; the prop and parlay branches are
--                    unchanged. ev_pnl_daily / ev_pnl_daily_all sit on ev_pnl and need no change (same columns).

CREATE OR REPLACE VIEW ev_game_record AS
  -- Legacy branch: exactly the old ev_pnl game-line logic, for games that commenced before the de-vig fix.
  SELECT r.sport::text                       AS sport,
         r.game_pk::bigint                   AS game_pk,
         r.market::text                      AS market,
         r.side::text                        AS side,
         p.matchup::text                     AS matchup,
         p.commence_time::timestamptz        AS commence_time,
         p.best_price::integer               AS best_price,
         p.best_book::text                   AS best_book,
         p.ev_best::double precision         AS ev_best,
         p.base_prob::double precision       AS base_prob,
         r.won::boolean                      AS won,
         r.clv::double precision             AS clv,
         (CASE WHEN r.won IS NULL THEN 'push' WHEN r.won THEN 'win' ELSE 'loss' END)::text AS result,
         (CASE WHEN r.won IS NULL THEN 0.0
               WHEN r.won THEN 10 * (american_to_decimal(p.best_price) - 1)
               ELSE -10.0 END)::double precision AS pnl,
         r.graded_at::timestamptz            AS graded_at,
         p.created_at::timestamptz           AS first_shown_at
  FROM ev_results r
  JOIN LATERAL (
    SELECT matchup, commence_time, best_price, best_book, ev_best, base_prob, created_at
    FROM ev_picks e
    WHERE e.game_pk = r.game_pk AND e.market = r.market AND e.side = r.side AND e.is_pick
    ORDER BY e.created_at DESC LIMIT 1
  ) p ON true
  WHERE (r.won IS NOT NULL OR p.best_price IS NOT NULL)
    AND p.commence_time < '2026-10-14T18:00:00Z'::timestamptz
  UNION ALL
  -- Ledger branch: the first-shown observation of each surfaced pick, graded at the first-shown price.
  -- units_at_pick is SIGNED profit for 1 unit (W: decimal - 1, L: -1, P: 0), so a $10 bet's pnl is 10 * units.
  SELECT f.sport::text,
         f.game_pk::bigint,
         f.market::text,
         f.side::text,
         m.matchup::text,
         f.commence_time::timestamptz,
         f.best_price::integer,
         pl.best_book::text,
         (pl.ev_pct / 100.0)::double precision,
         pl.ref_prob::double precision,
         (CASE f.result WHEN 'W' THEN true WHEN 'L' THEN false ELSE NULL END)::boolean,
         f.clv_prob::double precision,
         (CASE f.result WHEN 'W' THEN 'win' WHEN 'L' THEN 'loss' ELSE 'push' END)::text,
         (10 * f.units_at_pick)::double precision,
         f.graded_at::timestamptz,
         f.observed_at::timestamptz
  FROM (
    -- ONE row per (sport, game_pk, market): the side shown FIRST wins. A pick whose first observation is before
    -- the cutoff is excluded (its basis = 'first' observed_at IS that first observation); never fall back to a
    -- later observation.
    SELECT DISTINCT ON (plr.sport, plr.game_pk, plr.market)
           plr.ledger_id, plr.sport, plr.game_pk, plr.market, plr.side, plr.best_price, plr.commence_time,
           plr.result, plr.units_at_pick, plr.clv_prob, plr.graded_at, plr.observed_at
    FROM pick_ledger_results plr
    WHERE plr.source = 'ev_game'
      AND plr.basis = 'first'
      AND plr.observed_at >= '2026-10-14T18:00:00Z'::timestamptz
      AND plr.commence_time >= '2026-10-14T18:00:00Z'::timestamptz
      AND plr.units_at_pick IS NOT NULL
    ORDER BY plr.sport, plr.game_pk, plr.market, plr.observed_at ASC, plr.side
  ) f
  JOIN pick_ledger pl ON pl.id = f.ledger_id
  LEFT JOIN LATERAL (
    SELECT e.matchup FROM ev_picks e
    WHERE e.sport = f.sport AND e.game_pk = f.game_pk
    ORDER BY e.created_at DESC LIMIT 1
  ) m ON true;

-- ev_pnl: game lines now come from ev_game_record; props and parlays are unchanged (verbatim from
-- migration_ev_best_parlays.sql). Column names, order and types match the previous definition.
CREATE OR REPLACE VIEW ev_pnl AS
  SELECT sport::text AS sport, game_pk::bigint AS game_pk, market::text AS market, side::text AS side,
         commence_time::timestamptz AS commence_time, result::text AS result, pnl::double precision AS pnl
  FROM ev_game_record
  UNION ALL
  SELECT sport, game_pk, 'prop', side, commence_time,
         CASE result WHEN 'win' THEN 'win' WHEN 'loss' THEN 'loss' ELSE 'push' END,
         10 * profit
  FROM ev_prop_results
  WHERE result IS NOT NULL AND profit IS NOT NULL
  UNION ALL
  SELECT sport, NULL::bigint, 'parlay', NULL::text, first_commence, result, pnl
  FROM ev_best_parlay_results;

GRANT SELECT ON ev_game_record, ev_pnl TO anon, authenticated;
