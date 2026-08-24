# NFL Elo/SoS walk-forward backtest — findings

Date: 2026-08-24 (revised, Fix round 1)
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
win_acc are **bit-for-bit identical across every `w_sos`** in the grid — the
validation run below shows `brier=0.22712` for both the blended and pure-Elo
configs, to 5 decimal places, exactly. This is structural (how `e_home` is
wired in `run_backtest`), not an empirical finding, so:

- `tune()` now selects by **train-span margin MAE**, not Brier.
- The "does SoS beat pure Elo out-of-sample" verdict below is made on
  **validation-span margin MAE**, not Brier.

## Search strategy

A single `run_backtest` call over the full 2020–2025 validation span costs
~6.0s; over the 2002–2019 train span (2.85x more games), ~17.6s. The brief's
full Cartesian grid (4×4×4×4×3 = 768 configs) evaluated on train would cost
≈768 × 17.6s ≈ 3.75 hours — far past budget — so `main()` uses a
**coordinate search** (`tune(train, valid, grid)` itself stays full-product,
as the unit test exercises it): 3 passes, one parameter swept at a time over
its listed values holding the others at the current best, starting from the
middle value of each parameter's list, selecting on **train-span margin
MAE**, with a result cache to skip already-evaluated combos.

Actual wall-clock time for the real tuning run: **10m 24.6s** (614.0s user
CPU). This is somewhat longer than the original valid-span-selected search
(2m 45s) because each evaluation now runs over the larger train span, and
because two additional validation-span backtests are run at the end (for the
final OOS comparison) — but it stayed within a workable single run.

## Tuned parameters (`assets/nfl/rating.json`)

Selected by train-span margin MAE via coordinate search, then confirmed to
win out-of-sample on validation-span margin MAE (see verdict below), so the
blended config ships:

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

The best pure-Elo config found by the same train-selected coordinate search
(restricted to `w_sos=0`, same objective) had identical `(k, hfa_elo,
carryover) = (16, 55, 0.6)` — the search converged to the same Elo
hyperparameters whether or not SoS blending was allowed, so the OOS
comparison below is a clean apples-to-apples test of `w_sos=0` vs `w_sos=0.3`
at the same Elo settings.

## Validation-span metrics (2020–2025, n=1615)

| config | Brier | Win acc | Margin MAE | Margin RMSE |
|---|---|---|---|---|
| **Blended (shipped, w_sos=0.3, srs_min_games=6)** | 0.22712 | 62.54% | **10.2076** | 13.1550 |
| Pure Elo (w_sos=0, same k/hfa/carryover) | 0.22712 | 62.54% | 10.2587 | 13.2192 |

Brier and win accuracy are identical between the two rows, exactly as
expected from the structural note above — they cannot move with `w_sos`.
Margin MAE and RMSE are the metrics that differ.

## Naive baselines (validation span, n=1615)

| baseline | Brier | Win acc |
|---|---|---|
| Home-always (p=1.0 for home) | 0.4675 | 53.25% |
| Prior-season win% (p=0.5+(home_prior_wp−away_prior_wp)/2, clipped) | 0.2392 | 59.01% |

(A third reference point, a constant home-rate baseline fit on the train
span's overall home-win rate of 0.5692 and applied flat to validation, scores
Brier 0.2503 / win acc 53.25% — included for completeness though the spec
only calls for the two above.)

The tuned Elo/SoS model clearly beats both naive baselines on Brier (0.227 vs
0.250 / 0.239) and win accuracy (62.5% vs 53.3% / 59.0%).

## Does the SoS blend beat pure Elo out-of-sample?

**Yes — on validation-span margin MAE, the metric the blend actually
affects.** Blended margin MAE 10.2076 vs pure-Elo margin MAE 10.2587 at the
same `(k=16, hfa_elo=55, carryover=0.6)` — a real, out-of-sample improvement
of 0.051 points (~0.5%), and margin RMSE improves similarly (13.155 vs
13.219, ~0.5%). Since both configs were reached by the identical train-span
selection process (only differing in whether `w_sos` was allowed off zero),
this is a fair OOS test, and the blend wins it. `assets/nfl/rating.json`
therefore ships the blended config (`w_sos=0.3`, `srs_min_games=6`).

Brier/win_acc cannot register this improvement at all (see the metric-choice
note above) — that is a property of how `run_backtest` computes those two
metrics, not evidence against the blend.

## Fix round 1 — what changed from the original submission

The original submission selected hyperparameters (including `w_sos`) by
**validation-span Brier**, which is structurally blind to the SoS blend (see
above), and `tune()`'s `train_df` argument was unused. That made "does SoS
beat pure Elo" undecidable from the reported metric. This revision:

1. `tune()` now selects by **train-span margin MAE** (uses `train_df` for
   real; each result entry carries both `train` and `valid` metrics).
2. `main()`'s coordinate search selects on **train-span margin MAE**, then
   computes **validation-span** metrics for both the selected blended config
   and the best pure-Elo config (`w_sos=0`) found by the same search.
3. The verdict is now made on **validation-span margin MAE**, which the blend
   can actually move, and comes out **in favor of the blend** (0.051-point
   OOS improvement) — `rating.json` was regenerated accordingly (previous
   run had picked `w_sos=0.3` too, but for the wrong/uninformative reason —
   tied Brier — and with different `k/hfa_elo/carryover/srs_min_games`
   values since it searched a different objective).

## Concerns / follow-ups for the controller

- Brier/win_acc remaining permanently blend-invariant in this backtest means
  any future win-probability quality claim for the SoS feature (as opposed
  to margin/point-spread quality) would need `expected_home`-level wiring of
  the blend, which is out of this task's scope (P2 territory per the plan).
- The two "pure-Elo" configs found across the two tuning runs
  (`k=20,hfa=40,carry=0.6` under Brier-selection vs `k=16,hfa=55,carry=0.6`
  under margin-MAE-selection) differ, confirming Brier and margin MAE were
  genuinely selecting on different signal — margin MAE is the correct
  objective for this feature per the fix.
