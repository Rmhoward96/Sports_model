# CFB live projection fix: held-out comparison (2026-09-30)

Branch `fix/cfb-live-ratings`. Produced by `scripts/compare_cfb_live_fix.py`
(committed assets as of this branch; nothing written to any database).

## What was wrong
1. **Pooled live state.** `generate_cfb._season_to_date_ratings` computed SRS,
   opponent-adjusted points and games_played over *every* REG game since 2015.
   The tuned/backtested engine (`walkforward.raw_model_predictions`) uses Elo
   continuous across seasons but SRS/points/games_played season-to-date. With
   games_played around 140, the preseason prior decayed to nothing and the
   SRS gate was always open, so live served a model that was never backtested.
2. **Leaky prior.** R_pre = 1500 + the *same* season's `sp_rating`, which is
   CFBD's end-of-season SP+. Its weights (and coach_first_year /
   forward_sos_shift) were fitted on post-season information.

## What changed
- **A:** live state = walk-forward state (shared `walkforward.walk` /
  `live_week_state` / `model_margin_total`). Parity tests pin generate_cfb's
  margin/total (prior off) to the walk-forward rows, synthetic and real
  2023 weeks 1 and 6. Walk-forward output is byte-identical on 2015–2026.
- **B:** leak-free prior: previous season's final SP+ (missing → league mean)
  + z(returning_pct, recruiting_points, portal_net, prior_sos) + signed QB
  flag. The feature weights were fit on TRAIN 2016–2022 weeks 1–5 (FBS margin
  MAE) with the fixed engine: sp_scale 17.5, sp_offset 1500, w_returning 50,
  w_recruiting 80, w_portal 50, w_qb 0, w_sos_prior 50. The decay was then
  re-chosen with those weights held fixed, on **all** TRAIN weeks (fix round
  1: the floor governs the whole season): half_life_games 3, prior_floor
  0.35. The weeks 1–5 fit had chosen a floor of 0.65, which hurt weeks 6+.
- **Fix round 1:**
  - Postseason runs (ESPN season_type 3, week 1) now use state week =
    max REG week + 1, so bowls get the full-season state.
  - `GAME_MODEL_VERSION` is now `cfb-ratings-v2`.
  - The live producer refuses to run on a missing or pre-fix
    `priors_weights.json` when `priors.parquet` exists.

## Held-out comparison (2023–2025)
Games are REG FBS-vs-FBS games with a closing spread (the games live serves),
scored week by week as the producer would have run before each week. Both
methods go through the same `build_game_rows` + gameline.json (served
`pred_margin` / `pred_total`, bias applied). ATS is the model's side vs the
closing spread; pushes are excluded from the percentage.
- **today:** a faithful frozen copy of the pre-fix live code (pooled
  Elo/SRS/points/counts + 1500 + same-season SP+, half-life 4, floor 0).
- **fixed:** the current live code (A + B).

### All held-out seasons

| method | n | margin MAE (all) | margin MAE wk 1-5 | margin MAE wk 6+ | total MAE (all) | total MAE wk 1-5 | ATS vs close (W-L-P) |
|---|---|---|---|---|---|---|---|
| today (pooled + same-season SP+ prior) | 2169 | 13.092 | 13.064 (n=731) | 13.107 (n=1438) | 13.733 | 13.874 | 50.5% (1073-1051-45) |
| fixed (walk-forward state + leak-free prior) | 2169 | 12.607 | 12.864 (n=731) | 12.476 (n=1438) | 13.098 | 13.447 | 49.0% (1040-1084-45) |
| closing spread (reference) | 2169 | 12.016 | 11.719 | 12.167 | - | - | - |

Paired margin abs-error difference (fixed - today): -0.485 ± 0.116 (SE)

Paired total abs-error difference (fixed - today): -0.635 ± 0.156 (SE)
Paired margin difference, weeks 1-5 only: -0.200 ± 0.227 (SE)
Paired margin difference, weeks 6+ only: -0.630 ± 0.131 (SE)

### By season

| season / method | n | margin MAE (all) | margin MAE wk 1-5 | margin MAE wk 6+ | total MAE (all) | total MAE wk 1-5 | ATS vs close (W-L-P) |
|---|---|---|---|---|---|---|---|
| 2023 today | 719 | 12.688 | 12.051 (n=260) | 13.049 (n=459) | 13.945 | 14.169 | 51.7% (364-340-15) |
| 2023 fixed | 719 | 12.384 | 12.035 (n=260) | 12.582 (n=459) | 13.274 | 14.001 | 50.0% (352-352-15) |
| 2024 today | 721 | 13.600 | 14.413 (n=237) | 13.201 (n=484) | 13.444 | 13.154 | 48.4% (342-364-15) |
| 2024 fixed | 721 | 13.081 | 14.556 (n=237) | 12.359 (n=484) | 13.137 | 12.661 | 47.9% (338-368-15) |
| 2025 today | 729 | 12.990 | 12.824 (n=234) | 13.068 (n=495) | 13.808 | 14.276 | 51.4% (367-347-15) |
| 2025 fixed | 729 | 12.358 | 12.071 (n=234) | 12.493 (n=495) | 12.886 | 13.627 | 49.0% (350-364-15) |

The fixed path was cross-checked against the backtest's prior-seeded
walk-forward (`backtest_cfb_priors.prior_seeded_margin` − bias). All 2169
games are identical (max |diff| 0.0), re-checked with the refit decay
(half-life 3, floor 0.35).

## Reading
- **Margin:** fixed beats today over the full season, −0.49 ± 0.12 MAE
  (paired), and in every held-out season.
  - Most of the gain is in weeks 6+ (−0.63 ± 0.13).
  - In weeks 1–5 the gain is small and not significant (−0.20 ± 0.23;
    2024 is slightly worse). Today's pooled SRS/points carried last seasons'
    strength into September, which acts as a crude prior.
  - Today's leaky same-season SP+ prior had no effect: games_played ≈ 140
    decayed it to 0.
- **Total:** fixed is better, −0.64 ± 0.16 MAE, even with the 55.0 week-1 seed
  (the backtested behaviour; the early-season totals change is deferred).
- **ATS vs close:** fixed 49.0% vs today 50.5% (n = 2124 decisions each).
  Neither is distinguishable from 50% (SE ≈ 1.1 pp). These are predictions,
  not bets. The closing spread's own margin MAE (12.02) is still better than
  either model.
- **Prior refit (backtest_cfb_priors, HOLDOUT 2023–25, FBS, all games):**
  weeks 1–5: 13.495 without the prior → 12.842 with it; weeks 6+: 12.580 →
  12.497; all weeks: 12.890 → 12.614.

## Caveats
- Holdout ablation (informational, not used for selection): zeroing
  `w_portal` would improve holdout weeks 1–5 (12.842 → 12.744). Portal data
  exists only for 2021–22 in TRAIN.
- A team's second game in the same CFBD week (week 0 folds into week 1)
  gets the week-start Elo live; the backtest updates Elo within the week.
  No FBS-vs-FBS closing-line game in 2023–25 was affected.
- Deferred (controller ruling R3): week-1 totals still use the flat 55.0 seed.
  Week-1 total MAE is worse than today's pooled points (15.05 vs 14.36).
