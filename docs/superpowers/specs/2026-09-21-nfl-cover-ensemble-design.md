# NFL Cover/Total Stacked-Ensemble — Design

**Date:** 2026-09-21
**Status:** Draft for review
**Author:** Ryan + Claude
**Inspiration:** Levine, *Beating Vegas: Creating a Dynamic Sports Betting Model* (Duke, 2019) — its cover-probability-from-posterior-predictive-ECDF method, betting-market features (cash%/ticket%, line movement), DVOA efficiency features, and "agreement" strategy.

## Goal

Improve the NFL **spread-cover** and **total over/under** probabilities (the markets currently ~tying the naive baseline) by replacing the single-source cover/over probability with a **stacked ensemble** that blends three base learners — the ratings model, the Monte-Carlo sim, and a gradient-boosted feature model — into one **calibrated** probability, and only surfacing a pick when the learners **agree**. Two new inputs drive the feature model: an **opponent-adjusted EPA efficiency differential** (a DVOA analog) and **public betting splits** (cash% vs ticket%) + line movement.

## Non-goals / honest caveats

- **Not** reproducing the thesis's betting-return simulation. Its returns are optimistic (2 seasons, leaky k-fold, Martingale, Kelly capped at 0.2). The scorecard here stays **calibration + CLV**, consistent with the rest of the project.
- NFL is small-N (~272 games/season). Everything is regularized and validated walk-forward; we pool 2015–2025 and lean on shrinkage, not a deep model.
- **Betting splits and intra-week line movement have no history** in our data (odds capture just started; nflverse gives only the closing line). They are **live-first** features whose learned weight comes online only as labeled games accrue (Phase 2). The efficiency variable is trainable today (Phase 1).
- Strict **as-of-decision** feature construction — no post-game or same-week leakage — mirroring `epa.team_epa_by_season`'s existing discipline.

## What we already have (reused, not rebuilt)

- `nfl/gameline.build_gameline` → `margin_dist` / `total_dist` from the ratings model. Base learner #1.
- Monte-Carlo sim `nfl_sim.margin_dist` / `total_dist` (just shipped). Base learner #2.
- `model/distributions.prob_cover` / `prob_over_dist` / `apply_affine` — turn a dist + line into a probability. The thesis's ECDF step.
- `serving/board.spread_row` / `total_row` — compute the cover/over prob, no-vig market prob, EV, best price → board rows. **This is the single seam the ensemble plugs into.**
- `model/calibration` (Platt `fit`/`apply`/`calibrate`, `assets/calibration.json`) — keep as the final calibration.
- `nfl/epa.py` — leakage-free per-(season, team) off/def EPA back to 2015. Basis for the DVOA analog.
- `nfl/sportsdata.py` — SportsDataIO adapter (key, backoff) already used for injuries; extend for betting splits.
- `schedules.parquet` (2002–2025: `spread_line`, `total_line`, `result`, scores) — labels for cover/total outcomes. `backtest_nfl_gameline.py` — walk-forward harness to reuse.

## Architecture

```
ratings margin/total ─► prob_cover / prob_over ─┐
sim margin/total    ─► prob_cover / prob_over ─┤
                                               ├─► META-LEARNER ─► Platt calibrate ─► board (spread_row/total_row) ─► ev_picks
GBM(features)        ─► prob_cover / prob_over ─┘        │
                                                 agreement gate (direction + edge margin)
```

### Base learners (each emits P(cover) for spread and P(over) for total)
1. **Ratings** — `prob_cover(gameline.margin_dist, home_line)`, `prob_over_dist(gameline.total_dist, total_line)`.
2. **Sim** — same functions on the sim's `margin_dist` / `total_dist`.
3. **GBM** — `sklearn.ensemble.HistGradientBoostingRegressor` (predicts margin and total) over the feature set below, converted to a probability against the line via a residual-sigma Normal (same `normal_to_*_pmf` path the ratings model uses). Regularized: shallow depth, high `min_samples_leaf`, early stopping. **scikit-learn is added as a project dependency** (also provides the meta-learner's logistic regression).

### Feature set (as-of decision time, leakage-safe)
- **Efficiency (variable #1, trainable now):** opponent-adjusted off/def EPA per team → home/away **efficiency differential** (`(home_off_adj − away_def_adj) − (away_off_adj − home_def_adj)`), plus pace/total-oriented EPA sums for the total model. Built by extending `epa.py` with a schedule-strength adjustment (ridge/iterative opponent adjustment), computed from games strictly before the target week.
- **Context (have):** rest days, home field, division, travel, weather (`venues`/`weather`), injury severity (nflverse/SportsDataIO injuries).
- **The line itself:** decision-point `spread_line` / `total_line` (the number to beat).
- **Betting market (variable #2, live-first):** cash% vs ticket% split and their divergence (sharp signal), + line movement / reverse-line-movement from `odds_snapshot`. Sourced from **SportsDataIO Betting tier** via a new `sportsdata` adapter method. Fed to the GBM once available; until a labeled history exists, the GBM is trained without them and they enter via a small prior/Phase-2 retrain.

### Meta-learner (the stack)
- A **regularized logistic regression** per market (spread, total) over the three base probabilities (+ optionally a couple of context features: week, |edge| dispersion). Trained **out-of-fold, walk-forward** (fold = later season predicted from earlier) so it never sees its own game — the correct way to stack without leakage.
- Output → existing **Platt calibration** (`calibration.calibrate`), extending `assets/calibration.json` with `cover`/`total` ensemble targets.

### Agreement gate
- Surface a spread/total pick to `ev_picks` only when: base learners agree on direction (all lean the same side) **and** the calibrated edge clears break-even by a configurable margin. A learned, quantified version of the thesis's safest ("Agreement") strategy. Non-agreeing games still show probabilities but aren't flagged as +EV picks.

### Serving integration
- New `serving/ensemble.py` (pure): `ensemble_cover_prob(ratings_p, sim_p, gbm_p) -> float` and `ensemble_over_prob(...)`, loading the meta-learner + calibration params from an asset.
- `board.spread_row` / `total_row` gain an optional ensemble probability: when present, it replaces the single-source `p_home`/`p_over`; otherwise they behave exactly as today (graceful).
- Everything downstream (novig, EV, best price, `ev_picks`, grading, CLV) is unchanged.

## Training & validation
- Labels from `schedules.parquet`: home cover vs `spread_line`, over vs `total_line`, 2015–2025.
- Reconstruct base-learner probabilities historically: ratings via `backtest_nfl_gameline`'s point-in-time path; sim via the sim backtest; GBM via as-of features.
- **Walk-forward** by season; the meta-learner and GBM are fit only on prior seasons for each test season.
- Metrics: Brier / log-loss + **calibration curves** per market, and **CLV** on the forward slate (the real bar). Ship only if the ensemble beats each base learner and the naive baseline out-of-sample.
- Artifacts committed to `assets/` (GBM model, meta-learner coefficients, calibration additions), refreshed by an offline training script (not the daily job).

## Phasing
- **Phase 1 (now, both markets):** ratings + sim + GBM(efficiency + context + line) → stacked, calibrated, agreement-gated; walk-forward + CLV. Ships the fix using variable #1.
- **Phase 2 (data-gated):** add SportsDataIO betting splits + live line-movement to the GBM; retrain as labeled history with splits accrues. Requires the Betting product tier on the SportsDataIO subscription.

## Decisions locked
- **Provider:** staying on **SportsDataIO** (subscription upgraded to the **Betting tier** for splits); no Sportradar switch. CFB desk injuries keep flowing from SportsDataIO.
- **Modeling library:** **scikit-learn** added as a dependency (HistGradientBoosting GBM + logistic-regression meta-learner).
- **Scope:** spread **and** total, stacked ensemble, agreement-gated. Variables: opponent-adjusted EPA differential (Phase 1) + SportsDataIO betting splits & line movement (Phase 2, live-first).

## Open items (implementation-time)
- **SportsDataIO Betting endpoint** — confirm the exact path (`/v3/nfl/odds/json/BettingSplitsByGameID` family) and CFB equivalent once the Betting tier is active (Phase 2 only; does not block Phase 1).
- Training-data reconstruction cost: regenerating historical ratings + sim probabilities for 2015–2025 is the heaviest task; the sim/gameline backtests already exist to do it.
