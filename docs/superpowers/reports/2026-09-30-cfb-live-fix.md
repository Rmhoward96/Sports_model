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
  flag. It was refit on TRAIN 2016–2022 (weeks 1–5 FBS margin MAE) with the
  fixed engine: sp_scale 17.5, sp_offset 1500, w_returning 50, w_recruiting 80,
  w_portal 50, w_qb 0, w_sos_prior 50, half_life_games 3, prior_floor 0.65.

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

| method | n | margin MAE (all) | margin MAE wk 1-5 | total MAE (all) | total MAE wk 1-5 | ATS vs close (W-L-P) |
|---|---|---|---|---|---|---|
| today (pooled + same-season SP+ prior) | 2169 | 13.092 | 13.064 (n=731) | 13.733 | 13.874 | 50.5% (1073-1051-45) |
| fixed (walk-forward state + leak-free prior) | 2169 | 12.632 | 12.900 (n=731) | 13.098 | 13.447 | 50.2% (1067-1057-45) |
| closing spread (reference) | 2169 | 12.016 | 11.719 | - | - | - |

Paired margin abs-error difference (fixed - today): -0.460 ± 0.116 (SE)

Paired total abs-error difference (fixed - today): -0.635 ± 0.156 (SE)
Paired margin difference, weeks 1-5 only: -0.164 ± 0.229 (SE)

### By season

| season / method | n | margin MAE (all) | margin MAE wk 1-5 | total MAE (all) | total MAE wk 1-5 | ATS vs close (W-L-P) |
|---|---|---|---|---|---|---|
| 2023 today | 719 | 12.688 | 12.051 (n=260) | 13.945 | 14.169 | 51.7% (364-340-15) |
| 2023 fixed | 719 | 12.405 | 12.102 (n=260) | 13.274 | 14.001 | 51.3% (361-343-15) |
| 2024 today | 721 | 13.600 | 14.413 (n=237) | 13.444 | 13.154 | 48.4% (342-364-15) |
| 2024 fixed | 721 | 13.108 | 14.607 (n=237) | 13.137 | 12.661 | 48.9% (345-361-15) |
| 2025 today | 729 | 12.990 | 12.824 (n=234) | 13.808 | 14.276 | 51.4% (367-347-15) |
| 2025 fixed | 729 | 12.385 | 12.057 (n=234) | 12.886 | 13.627 | 50.6% (361-353-15) |

The fixed path was cross-checked against the backtest's prior-seeded
walk-forward (`backtest_cfb_priors.prior_seeded_margin` − bias). All 2169
games are identical (max |diff| 0.0).

## Reading
- **Margin:** fixed beats today over the full season, −0.46 ± 0.12 MAE
  (paired), and in every held-out season. In weeks 1–5 the gain is small and
  not significant (−0.16 ± 0.23; 2024 is slightly worse). Today's pooled
  SRS/points carried last seasons' strength into September, which acts as a
  crude prior. Its leaky same-season SP+ prior had no effect: games_played ≈
  140 decayed it to 0.
- **Total:** fixed is better, −0.64 ± 0.16 MAE, even with the 55.0 week-1 seed
  (the backtested behaviour).
- **ATS vs close:** both are about 50% (today 50.5%, fixed 50.2%). Neither beats
  the closing line; these are predictions, not bets. The closing spread's own
  margin MAE (12.02) is still better than either model.
- **Prior refit (backtest_cfb_priors, HOLDOUT 2023–25, weeks 1–5 FBS, all
  games):** 13.495 without the prior → 12.874 with it.

## Caveats
- The floor of 0.65 (prior keeps at least 65% weight all season) was chosen
  on weeks 1–5 only. The full-season held-out numbers above include its effect
  on weeks 6+ (net positive).
- Holdout ablation (informational, not used for selection): zeroing
  `w_portal` would improve holdout weeks 1–5 (12.874 → 12.760). Portal data
  exists only for 2021–22 in TRAIN.
- A team's second game in the same CFBD week (week 0 folds into week 1)
  gets the week-start Elo live; the backtest updates Elo within the week.
  No FBS-vs-FBS closing-line game in 2023–25 was affected.
