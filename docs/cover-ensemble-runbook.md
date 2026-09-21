# Cover/Total Ensemble — Runbook

The NFL spread-cover / total ensemble and its walk-forward **ship gate**. Phase 1
(team-quality features) shipped the infra but **failed the gate** — team quality
doesn't beat the NFL closing line. Phase 2 adds **market-microstructure** features
(betting splits + line movement) and re-tests with the same gate. Nothing is
wired into serving until the gate passes.

## Pieces
- `scripts/build_cover_dataset.py` → `assets/nfl/cover_dataset.parquet` (2015–2025 labels + features). Features: opponent-adjusted EPA efficiency, ratings/sim base probs, and the Phase-2 market columns (`mkt_*`, NaN for the historical tail).
- `scripts/train_cover_ensemble.py` — walk-forward GBM + logistic meta, calibration diagnostic (ECE), leakage regression test, and the **ship gate**: writes artifacts (`assets/nfl/cover_ensemble.json`, `cover_gbm_*.joblib`, extends `assets/calibration.json`) **only if** the ensemble beats each base learner AND p=0.5 out-of-sample (Brier + log-loss) on both markets.
- `serving/ensemble.py` — applies the artifact at serve time (`load_ensemble()` → None when absent → graceful no-op).
- Market data capture: `scripts/capture_betting_splits.py` + `capture-betting-splits.yml` (splits); `odds_snapshot` via `capture-odds.yml` (line movement).

## The accumulate-then-re-gate loop (Phase 2)
Market features are **live-first** — no history, so the gate can't validate them until data accrues (realistically a season+).

1. **Enable capture.**
   - Splits: confirm the SportsDataIO **Betting tier** is active; verify the splits endpoint/fields in `nfl/sportsdata.py` (`BETTING_SPLITS_PATH`, `parse_betting_splits`) against a live payload, and the SportsDataIO→ESPN game-id resolution in `capture_betting_splits._fetch_and_parse_splits`. Then set repo var `INGEST_SPLITS=true` (and run `db/migration_nfl_betting_splits.sql`).
   - Line movement: confirm `capture-odds.yml` is actually populating `odds_snapshot` (it was empty as of 2026-09-21).
2. **Accumulate** across the season (both feeds run near close on game days).
3. **Re-run** periodically: `uv run python scripts/build_cover_dataset.py` then `uv run python scripts/train_cover_ensemble.py`. Read the printed OOS metrics table (Brier/log-loss + ECE per market vs each base learner and p=0.5).
4. **Ship only if the gate passes.** Then wire the ensemble into serving — the deferred Phase-1 tasks: `board.spread_row`/`total_row` accept an ensemble prob, and the +EV producer computes base probs → `ensemble_prob` → `agreement`-gated `ev_picks`. Until then, serving is unchanged.

## Honest caveats
- Payoff is delayed and not guaranteed — market efficiency is a strong prior even for microstructure features. The gate tells you honestly if/when there's edge.
- Sim base probs are backfilled for 2024 only (`_SIM_SEASONS`); the shipped meta is 2-way `[ratings, gbm]` with sim via the agreement gate.
- Team codes are normalized (`normalize_team`) before every feature lookup; keep new joins consistent.
