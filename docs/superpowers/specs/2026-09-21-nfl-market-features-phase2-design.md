# NFL Market-Microstructure Features (Phase 2) — Design

**Date:** 2026-09-21
**Status:** Draft for review
**Author:** Ryan + Claude
**Builds on:** the Phase-1 cover/total ensemble infra on main (`serving/ensemble.py`, `nfl/efficiency.py`, `scripts/build_cover_dataset.py`, `scripts/train_cover_ensemble.py` + its walk-forward ship gate). Phase 1's gate FAILED: team-quality features don't beat the NFL closing line. This phase adds the features the thesis (Levine, Duke 2019) actually found predictive — **market microstructure** (what sharp money is doing), not team quality.

## Goal

Add two market-microstructure feature families to the cover/total GBM, capture them going forward, and re-run the existing walk-forward ship gate once enough labeled games accrue:
1. **Betting splits** (SportsDataIO Betting tier): public cash% vs ticket% per game/market and their divergence (the sharp-money signal).
2. **Line movement** (from the existing `odds_snapshot`): opening→closing line delta, reverse-line-movement, and sharp(Pinnacle)-vs-soft-consensus divergence.

Ship nothing user-facing until the gate passes (Phase-1 discipline).

## The hard reality (state it plainly)

Both families are **live-first**: SportsDataIO has no historical splits, and `odds_snapshot` is currently **empty** (no line-movement history either). So there is **no near-term gate-testable signal** — this phase is *build the capture + features, accumulate forward, re-gate later*. Realistically **a full season+ of captured market data** is needed before the walk-forward gate can meaningfully test these features (it must train on seasons/weeks that have them and test on others that do too). There is no shortcut; this is inherent to market data.

**Design consequence that makes this clean:** the GBM is `HistGradientBoostingRegressor`, which **handles NaN natively**. So the market features are added to the dataset as **NaN for the historical tail** (2015–pre-capture) and populated only for games captured from now on. No backfill is required or attempted; the GBM uses the features where present, and the periodic gate re-run tells us if/when they add out-of-sample skill. This is the correct architecture for live-first features.

## Scope

- Spread + total (matching Phase 1). NFL only.
- **Capture + wait to re-gate** — no serving/UI change until the gate passes out-of-sample.
- Out: no user-facing surface now; no moneyline/props; no Sportradar (staying on SportsDataIO).

## Components

### 1. Betting-splits ingest (SportsDataIO Betting tier)
- Extend `src/sportsmodel/nfl/sportsdata.py` with a splits fetch + parser (pure `parse_betting_splits`). Endpoint/fields confirmed at implementation once the Betting tier is active (SportsDataIO "Betting" product — e.g. the BettingSplits/consensus family); design against the expected shape: per (game, market, side) consensus **cash %** and **ticket %**.
- New table `nfl_betting_splits` (migration, user-run): `game_pk, market (spread|total|moneyline), side, cash_pct, ticket_pct, captured_at` (+ index on game_pk). Training-side only — no anon read needed.
- `db.upsert_nfl_betting_splits` (idempotent per game/market/side/captured_at).
- Capture workflow `.github/workflows/capture-betting-splits.yml`: near-close windowing (splits are meaningful post-lineup), reusing the `capture-odds` PROP_WINDOW pattern; NFL only; gated on `SPORTSDATA_API_KEY` + a `INGEST_SPLITS` var so it's a no-op until the tier is live.

### 2. Line-movement features (from `odds_snapshot`, pure)
`odds_snapshot` columns: `game_pk, market (moneyline|total|spread), side (home|away|over|under), book, line, price, commence_time, captured_at`. New pure module `src/sportsmodel/nfl/market_features.py`:
- `line_movement(snapshots, game_pk, market, decision_ts) -> dict`: from snapshots strictly at/before `decision_ts` — opening line (earliest), decision/closing line (latest ≤ ts), `dline` (close−open), `abs_dline`, movement velocity.
- `sharp_vs_soft(snapshots, ...) -> float`: Pinnacle no-vig prob minus soft-book consensus no-vig prob for the side (reusing `serving/board.novig`) — the sharp-disagreement signal available without %-splits.
- `reverse_line_movement(line_move, splits) -> float|None`: line moving toward the side the public %-money is *away* from (needs splits; None when splits absent).
- All leakage-safe: only snapshots before the decision timestamp.

### 3. Splits features (pure)
In `market_features.py`: `split_features(splits, game_pk, market, side) -> dict` → `cash_pct`, `ticket_pct`, `cash_minus_ticket` (sharp divergence). NaN when splits absent for that game.

### 4. Feature integration
- Extend `build_cover_dataset.assemble_rows` to add the market features (line movement, sharp-vs-soft, splits divergence) to each row — **NaN where the underlying data is absent** (the whole historical tail, initially).
- Add these columns to the GBM feature lists in `train_cover_ensemble.py` (`MARGIN_FEATURES`/`TOTAL_FEATURES`). HistGradientBoosting consumes NaN directly — no imputation.
- **Prerequisite fix (from Phase-1 follow-up), do FIRST:** `assemble_rows` currently looks up `adj[home]`/`sim_lookup` with raw `schedule_df` team codes vs `normalize_team`'d keys. Normalize `home`/`away` before all lookups (efficiency + sim + the new market joins) so a future aliased code can't silently zero features. This is the one latent correctness item Phase 1 deferred.

### 5. Re-gate (unchanged harness)
- Re-run `scripts/train_cover_ensemble.py` walk-forward periodically. It already: fits the GBM on prior seasons, blends via the meta, evaluates OOS Brier/log-loss + calibration vs each base learner and p=0.5, and **writes artifacts only if the gate passes**. Ship (wire into `board`/`ev_picks`, the deferred Phase-1 Tasks 7–8) **only** when the gate passes with the market features present.

## Testing
- Pure feature builders (`line_movement`, `sharp_vs_soft`, `reverse_line_movement`, `split_features`) unit-tested with synthetic snapshot/splits frames incl. leakage cases (a post-decision snapshot must not affect the result) and NaN-absence cases.
- `parse_betting_splits` tested against a mock payload; the fetch IO is not unit-tested (matches the module pattern).
- `db.upsert_nfl_betting_splits` FakeCursor test.
- Full suite green.

## Deployment (user, in order)
1. Confirm the SportsDataIO **Betting tier** is active; confirm the exact splits endpoint/fields.
2. Run the `nfl_betting_splits` migration.
3. Merge; enable `capture-betting-splits.yml` (+ `INGEST_SPLITS` var) and confirm odds capture is actually populating `odds_snapshot` (it's currently empty — line-movement features need it flowing).
4. **Accumulate** — let splits + odds capture run across the season.
5. Re-run `build_cover_dataset` + `train_cover_ensemble` periodically; wire into serving only if/when the gate passes.

## Non-goals / honest caveats
- No user-facing change and no proven edge in this phase — it is capture + accumulate + re-gate.
- Payoff is delayed a season+; the gate may still fail (market efficiency is a strong prior even for microstructure features). The value is that we'll *know*, honestly, via the same leakage-tested gate.
- Requires the paid Betting tier and that `odds_snapshot` capture is actually running (verify — currently empty).
