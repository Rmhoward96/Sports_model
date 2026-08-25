# NFL Elo/SoS walk-forward backtest — findings

Date: 2026-08-24 (revised, Final-review fix wave)
Script: `scripts/backtest_nfl_elo.py`
Data: `assets/nfl/schedules.parquet` (seasons 2002–2025, REG games only: 6,223 rows)
Train span: 2002–2019 (4,608 games) · Validation span: 2020–2025 (1,615 games)

Walk-forward is leak-free throughout: `run_elo` supplies PRE-game Elo per
game, and SRS is recomputed each game from only the games already played
that season (before the game being scored) — see `run_backtest`.

## Metric choice — why margin MAE, not Brier, is the SoS comparison metric

`run_backtest`'s `brier`/`win_acc` are computed from `e_home = g["e_home"]`,
which is `run_elo`'s pure-Elo win probability (`expected_home`) — it is never
touched by `blend_cfg`/`w_sos`. Only `margin_mae`/`margin_rmse` (fed by
`ratings.expected_margin`, which the blend does move) respond to the SoS
weight. Verified directly: at a fixed `(k, hfa_elo, carryover)`, Brier and
win_acc are **bit-for-bit identical across every `w_sos`** — the validation
run below shows `brier=0.22712` and `win_acc=0.62539` for both the blended
and pure-Elo configs, to 5 decimal places, exactly. This is structural (how
`e_home` is wired in `run_backtest`), not an empirical finding, so:

- `tune()` selects by **train-span margin MAE**, not Brier.
- The "does SoS beat pure Elo out-of-sample" verdict is made on
  **validation-span margin MAE**, not Brier.

## Enforced-fair comparison (Final-review fix wave)

The pure-Elo counterfactual is constructed by taking the selected best
BLENDED config's exact `(k, hfa_elo, carryover)` and only zeroing `w_sos`
(`srs_min_games` carried over too, though it's inert when `w_sos=0`). This
makes "does adding SoS help, holding Elo fixed" a **causal**, same-Elo-params
comparison rather than two independently-tuned configs whose Elo params
might coincidentally match (an earlier draft of this report selected the two
configs independently — see "What changed" below). The decision rule is also
strict: `blend_wins = blend_margin_mae < pure_margin_mae` (an exact tie does
**not** ship the blend).

## Search strategy

A single `run_backtest` call over the 2020–2025 validation span costs ~6.0s;
over the 2002–2019 train span (2.85x more games), ~17.6s. The brief's full
Cartesian grid (4×4×4×4×3 = 768 configs) evaluated on train would cost
≈768 × 17.6s ≈ 3.75 hours — far past budget — so `main()` uses a
**coordinate search** (`tune(train, valid, grid)` itself stays full-product,
as the unit test exercises it): 3 passes, one parameter swept at a time over
its listed values holding the others at the current best, starting from the
middle value of each parameter's list, selecting on **train-span margin
MAE**, with a result cache to skip already-evaluated combos.

Actual wall-clock time for this run: **10m 38.6s** (626.3s user CPU).

## Tuned parameters (`assets/nfl/rating.json`)

```json
{
  "k": 16,
  "hfa_elo": 55,
  "carryover": 0.6,
  "base": 1500.0,
  "w_sos": 0.3,
  "srs_min_games": 6
}
```

## Validation-span metrics (2020–2025, n=1615)

| config | Brier | Win acc | Margin MAE | Margin RMSE |
|---|---|---|---|---|
| **Blended (shipped, w_sos=0.3, srs_min_games=6)** | 0.22712 | 62.54% | **10.2076** | 13.1550 |
| Pure Elo (w_sos=0, identical k=16/hfa_elo=55/carryover=0.6) | 0.22712 | 62.54% | 10.2587 | 13.2192 |

Brier and win accuracy are identical between the two rows, exactly as
expected from the structural note above. Margin MAE and RMSE are the metrics
that actually differ, and are the basis for the verdict.

## Naive baselines (validation span, n=1615) — computed in code (`naive_baselines()`)

| baseline | Brier | Win acc | Margin MAE | Margin RMSE |
|---|---|---|---|---|
| Home-always (p=1.0 for home) | 0.4675 | 53.25% | — | — |
| Prior-season win% (p=0.5+(home_prior_wp−away_prior_wp)/2, clipped) | 0.2392 | 59.01% | — | — |
| Naive margin (constant = train mean home margin, +2.35 pts) | — | — | 11.0210 | 14.2182 |

`naive_baselines(reg_df, train_df, valid_df)` in `scripts/backtest_nfl_elo.py`
computes all three directly from the committed data (no hand computation),
so this table is reproducible by re-running the script.

The tuned Elo/SoS model beats every baseline on both axes the acceptance bar
calls for:
- **Brier:** 0.227 vs 0.4675 (home-always) / 0.239 (prior-season win%) —
  model wins clearly.
- **Win accuracy:** 62.5% vs 53.3% / 59.0% — model wins clearly.
- **Margin MAE:** 10.208–10.259 (model, either config) vs 11.021 (naive
  constant-margin baseline) — model wins clearly, by ~0.76–0.81 points.

## Does the SoS blend beat pure Elo out-of-sample? (honest verdict)

**Directionally yes, on the one metric it can move (margin MAE/RMSE) — but
the margin is not statistically distinguishable from noise.**

At identical Elo hyperparameters (`k=16, hfa_elo=55, carryover=0.6`):

- Blended margin MAE: **10.2076** vs pure-Elo margin MAE: **10.2587** —
  a 0.0511-point improvement (~0.5%).
- Blended margin RMSE: **13.1550** vs pure-Elo: **13.2192** — a similar-sized
  improvement (~0.5%).
- Both differences favor the blend, at every rounding level checked.

**But this is a small effect relative to sampling noise.** The standard
error of the mean absolute margin error on n≈1615 validation games is
≈0.207 points (computed directly: `std(|error|)/sqrt(n)` ≈ 8.30/√1615 for
each config) — the 0.051-point improvement is well within **one standard
error** of that noise floor. A paired comparison (same games, both configs)
narrows the SE to ≈0.030 (errors are correlated since both configs share the
same Elo trajectory), giving a paired t-statistic of ≈1.68 (p≈0.09,
two-tailed) — suggestive of a real, small, positive effect, but not
significant at the conventional 0.05 threshold either.

**Verdict:** the blend meets the spec's literal bar (`blend_margin_mae <
pure_margin_mae` out-of-sample, strict inequality) and degrades gracefully
via the cold-start guard (`srs_min_games`, `w_sos<=0` fallback to pure Elo
when SRS is unavailable), so shipping `w_sos=0.3` is **defensible**. But the
improvement is **weakly supported** — not clearly distinguishable from noise
on this sample size — and should be revisited once real odds/CLV data exist
to test whether the blend's margin edge translates into any pricing/betting
value, or whether it's within the model's inherent variance.

## What changed across review rounds

1. **Original submission:** selected `w_sos` by validation-span **Brier**,
   which is structurally blind to the blend (a tie by construction) —
   undecidable comparison.
2. **Fix round 1:** switched selection to **train-span margin MAE** (the
   metric the blend actually moves), but selected the "best pure-Elo"
   config **independently** from the blended config — their matching Elo
   params (`k=16, hfa_elo=55, carryover=0.6` in both) was a coincidence of
   the search, not enforced.
3. **Final-review fix wave (this revision):** the pure-Elo counterfactual now
   **reuses the selected blend's exact Elo params** and only zeroes
   `w_sos` — a causal, same-Elo comparison — with a **strict** `<` decision
   rule (ties don't ship the blend), plus the three naive baselines computed
   in code and an honest statistical hedge on the verdict (above), replacing
   earlier "clearly"/"real improvement" language.

Numerically, this revision's tuned params and metrics happen to match Fix
round 1's exactly (`k=16, hfa_elo=55, carryover=0.6, w_sos=0.3,
srs_min_games=6`; validation Brier/win_acc/margin_mae/margin_rmse all
identical to 5+ decimal places) — the coordinate search converges to the
same point either way in this case, but the enforcement now *guarantees*
that outcome rather than relying on coincidence, and the comparison is
reported honestly rather than overstated.

## Concerns / follow-ups for the controller

- The margin-MAE improvement from SoS blending, while directionally
  consistent and meeting the literal spec bar, is small relative to sampling
  noise (SE ≈0.2 pts unpaired, ≈0.03 pts paired) on ~1615 validation games.
  Recommend re-evaluating `w_sos` once more seasons of data accumulate or
  once the model is checked against real market lines (CLV), rather than
  treating `w_sos=0.3` as settled.
- Brier/win_acc remaining permanently blend-invariant in this backtest means
  any future win-probability quality claim for the SoS feature (as opposed
  to margin/point-spread quality) would need `expected_home`-level wiring of
  the blend, which is out of this task's scope (P2 territory per the plan).
