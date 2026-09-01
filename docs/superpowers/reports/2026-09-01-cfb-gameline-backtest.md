# CFB P2 Task 3: Walk-forward game-line backtest -- the market-beat GATE

Script: `scripts/backtest_cfb_gameline.py`
Output: `assets/cfb/gameline.json`
Inputs: `assets/cfb/schedules.parquet` (2015-2025, 9062 REG games), `assets/cfb/lines.parquet`
(2015-2024 CFBD closing lines, 6825 rows), `assets/cfb/rating.json` (P1's tuned
Elo/SoS config).

## What this does

Wires P1's Elo + SRS + `ratings.expected_margin` SoS blend (loaded from the shipped
`assets/cfb/rating.json`: `k=40, hfa_elo=70, carryover=0.9, w_sos=0.45, srs_min_games=3`)
+ `points.compute_points_ratings`/`expected_total` + `shrink.ShrinkParams`/`shrink` +
`gameline.build_gameline` into a single leak-free, per-game walk-forward over the
continuous 2015-2024 span. At each game, model margin/total come from information
strictly available before that game (pre-game Elo, season-to-date SRS, season-to-date
opponent-adjusted points); the CFBD closing spread/total are shrunk in via `shrink()`;
both are scored against the actual outcome.

Reused untouched from the NFL P2 engine: `nfl.elo`, `nfl.srs`, `nfl.ratings`,
`nfl.points`, `nfl.shrink`, `nfl.gameline`, `model.distributions`.

TRAIN span: 2015-2022 (shrink-curve + sigma fitting). OOS span: 2023-2024 (two complete,
held-out seasons with full CFBD market coverage -- the gate). Season 2025 is excluded
entirely (no CFBD lines exist for it, and it is CFB P1's "current/partial season"
convention).

## Three adaptations from `backtest_nfl_gameline.py`

1. **Separate market asset, joined in.** NFL's nflverse schedules already embed
   `spread_line`/`total_line`; CFB's CFBD lines live in a separate parquet keyed on
   `(season, week, home_team, away_team)`. `load_merged_schedule` left-joins schedules
   -> lines, and `market_spread`/`market_total` (CFBD's naming, home-margin convention)
   are mapped into the `"spread_line"`/`"total_line"` keys `build_gameline` expects, with
   the same NaN -> `None` sanitization `_clean_market` performs in the NFL script. CFBD
   only prices ~76% of the FBS schedule in this holdout (buy games vs. FCS, etc. are
   frequently unpriced) -- every market-referencing metric (blend/market-only MAE,
   cover/O-U accuracy) is therefore scored on the subset of games carrying a valid
   market number for **that specific metric**, so model-only/blend/market-only MAE are
   always compared on the exact same game set.
2. **Bad CFBD self-matches dropped.** A handful of rows have `home_team == away_team`
   (two different unresolved/FCS opponents both collapsing to the same placeholder, or
   an unmatched name on both sides) -- 3 of 9062 schedule rows, 2 of 6825 line rows.
   Dropped from both sides before anything else runs, so a team is never made to face
   itself inside Elo/SRS/points.
3. **Weekly-batched SRS+points refresh, continuous full-span walk-forward.** Following
   `backtest_cfb_ratings.py`'s precedent (CFB's ~830 games/season across ~137 teams makes
   a per-game Gauss-Seidel SRS+points recompute intractable), both caches refresh once
   per completed week, not per game. The walk-forward runs **once** over the continuous
   2015-2024 span (not separately per train/OOS span the way the NFL script does) so
   that OOS (2023-2024) ratings carry real multi-season history in, consistent with the
   brief's "ratings use only prior seasons + season-to-date" and with how P1's own gate
   script evaluates CFB OOS.

## Fitted parameters (`assets/cfb/gameline.json`)

```json
{
  "sigma_margin": 16.665,
  "sigma_total": 16.889,
  "offset": 110,
  "total_max": 150,
  "w_margin": {"start": 0.95, "floor": 0.3, "decay": 0.1},
  "w_total":  {"start": 0.95, "floor": 0.3, "decay": 0.1}
}
```

- `sigma_margin`/`sigma_total`: RMSE of `(pred - actual)` on the TRAIN span at the
  fitted shrink curves -- both noticeably wider than NFL's (~13.4/13.6), matching CFB's
  much larger talent gaps and higher-variance scoring.
- `w_margin`/`w_total`: the coordinate search converged to the same grid-boundary curve
  NFL did (`start=0.95` top of grid, `floor=0.3` top of grid, `decay=0.1` bottom of
  grid) -- maximal, flattest market weight within the searched range, for the same
  reason: the closing line minimizes TRAIN MAE better than the model at nearly every
  week, so an MAE-driven search leans on it as hard as the grid allows.
- `offset=110`, `total_max=150` are fixed CFB-scale constants (not tuned by this
  backtest's MAE objective), sized against the schedule's own extremes: max observed
  margin is 79 (p99.9 = 74), max observed total is 146 (median 55). NFL's defaults
  (75/120) would clip real CFB blowouts and shootouts.

## Search strategy + runtime

Coordinate (per-axis) search over `ShrinkParams(start, floor, decay)`, independently for
margin and total, minimizing TRAIN-span MAE -- identical method and grid to the NFL
script (`start in [0.5,0.65,0.75,0.85,0.95]`, `floor in [0.05,0.15,0.2,0.3]`,
`decay in [0.1,0.2,0.25,0.35,0.5]`). The expensive walk-forward
(`_raw_model_predictions`) is run exactly once over the full 2015-2024 span and cached;
every shrink trial re-scores those cached rows via `_apply_gl` (O(n), no Elo/SRS/points
recomputation).

Measured wall-clock (`PYTHONPATH=src uv run --no-sync python scripts/backtest_cfb_gameline.py`):
- Walk-forward (Elo+SRS+points, full 2015-2024 span, weekly-batched refresh): **9.3s**.
- Shrink coordinate search (3 passes x 3 axes x ~5 values, 2 curves): **38.4s**.
- **Total: 49.5s.**

## OOS (2023-2024) results -- THE GATE

All three configs share the fitted `sigma_margin`/`sigma_total` (`offset=110`,
`total_max=150`) and are scored from the same cached raw walk-forward rows, so the only
thing that differs between rows is the shrink weight.

### Margin MAE (games with a valid `market_spread`, n=1440)

| model-only (w=0) | **blend (fitted w-curve)** | market-only (w=1) |
|---|---|---|
| 12.971 (SE 0.263) | **12.147 (SE 0.246)** | 12.067 (SE 0.246) |

### Total MAE (games with a valid `market_total`, n=1440)

| model-only (w=0) | **blend (fitted w-curve)** | market-only (w=1) |
|---|---|---|
| 13.208 (SE 0.268) | **12.835 (SE 0.259)** | 12.816 (SE 0.259) |

Ordering is exactly as expected on both: model-only < blend < market-only, with the
blend landing very close to market-only (consistent with the market-heavy fitted
curve). The model-only vs. market-only gap (~0.9 margin pts, ~0.4 total pts) says the
closing CFBD line out-predicts this Elo+SoS+points model on raw point-estimate error,
same qualitative result P1's NFL gate found for NFL.

### Beats-the-closing-line accuracy (decided/non-push games only, vs. 52.4% breakeven)

| | spread cover acc | over/under acc |
|---|---|---|
| **model-only** | **49.72%** (n=1410) | **51.62%** (n=1416) |
| **blend** | **49.72%** (n=1410) | **51.62%** (n=1416) |

Both are below the ~52.4% breakeven a -110/-110 spread market requires. Spread cover
accuracy sits essentially at coin-flip (49.72%, i.e. slightly *worse* than random);
over/under accuracy is closer to breakeven (51.62%) but still under it.

**Why model-only and blend are numerically identical here -- not a bug.** For any game,
`blend_pred - market = (1 - w(week)) * (model_pred - market)`, and the fitted curve
never lets `w` reach 1 (`floor=0.3`), so `(1 - w)` is always strictly positive. Shrinking
toward the market rescales *how far* the served line sits from market but can never flip
*which side* of the market number it's on. So "the model's own pick" and "the blend's
pick" are the same pick on every single game, by construction, whenever `w < 1`
everywhere on the curve (true here). ATS/O-U accuracy is therefore entirely a property
of the raw model's own disagreement with the market -- shrinking cannot change it, only
MAE (which does care about magnitude, not just sign) can differ between the two.

## GATE verdict (numbers only, no pass/fail declared here per the task contract)

- **CFB model does not currently beat the closing CFBD line on point-estimate error**
  (margin MAE and total MAE both worse for model-only than market-only, same direction
  NFL P2 found).
- **CFB model's own side-of-the-market pick is at or below breakeven** on this 2023-2024
  holdout: 49.72% spread cover accuracy (below 52.4%), 51.62% total accuracy (below
  52.4%). Both n are in the 1400+ range, so this is not a small-sample artifact -- a 2.7pp
  shortfall on n=1410 has a binomial SE of ~1.3pp, so cover accuracy is within ~2 SE of
  breakeven either way; it is not evidence of a strong signal in either direction, but it
  is not evidence of an edge either.

## Concerns

1. **No standalone accuracy edge found in this holdout.** Same shape of result as the
   NFL P2 gate (market wins on MAE), but here the *directional* pick accuracy also sits
   at/under breakeven rather than comfortably above it -- this task's brief was explicit
   that reporting these numbers, not judging pass/fail, is Task 3's job, so this is
   flagged for the ledger/stakeholder review, not treated as a defect to silently fix.
2. **CFBD market coverage gap (~24% of the schedule has no line at all)** means the
   margin/total MAE and cover/O-U numbers above describe the ~76% of games CFBD prices
   (generally the more "mainstream" FBS matchups), not the full schedule. Model-only
   performance on the unpriced 24% (mostly lopsided buy games and small-conference
   matchups) is not evaluated here.
3. **Continuous full-span walk-forward (unlike the NFL script's separate train/valid
   passes)** means the fitted `sigma_margin`/`sigma_total`/shrink-curve TRAIN games and
   the OOS-eval games share one continuous Elo/SRS/points trajectory rather than two
   independently-initialized ones. This was a deliberate choice (matching
   `backtest_cfb_ratings.py`'s precedent, to avoid an unfair Elo cold-start at the
   OOS boundary) but is a real behavioral difference from the NFL P2 reference script,
   noted here for anyone comparing methodology across sports.
4. **Shrink search saturates at the grid boundary** for both curves (`start=0.95`,
   `floor=0.3` at the top of their grids, `decay=0.1` at the bottom) -- same finding as
   NFL P2's Concern #1. Not re-probed with a wider grid here; a follow-up stakeholder
   who wants the unconstrained MAE-optimal curve should expect it to lean even harder on
   the market than what shipped.

## Commands used

- `cd "/Users/ryan/Desktop/Sports Model" && PYTHONPATH=src uv run --no-sync python scripts/backtest_cfb_gameline.py`
  (real fit, writes `assets/cfb/gameline.json`)
- `PYTHONPATH=src uv run --no-sync python -m pytest -q` (full suite, 256 passed)
