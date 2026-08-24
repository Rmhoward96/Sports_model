# NFL Elo/SoS walk-forward backtest — findings

Date: 2026-08-24
Script: `scripts/backtest_nfl_elo.py`
Data: `assets/nfl/schedules.parquet` (seasons 2002–2025, REG games only: 6,223 rows)
Train span: 2002–2019 · Validation span: 2020–2025 (1,615 graded REG games)

## Search strategy

A single `run_backtest` call over the full 2020–2025 validation span took **~6.0s**.
The brief's full Cartesian grid (`k`×`hfa_elo`×`carryover`×`w_sos`×`srs_min_games`
= 4×4×4×4×3 = 768 configs) would have cost ≈768 × 6s ≈ **77 minutes**, well past the
~10-minute budget. Per the controller ruling, `main()` uses a **coordinate search**
instead (`tune(train, valid, grid)` itself is untouched — full-product, as the unit
test exercises it): 3 passes, one parameter swept at a time over its listed values
holding the others at the current best, starting from the middle value of each
parameter's list, with a result cache to skip already-evaluated combos.

Actual wall-clock time for the real tuning run: **2m 44.9s** (162.2s user CPU),
well under budget.

## Tuned parameters (`assets/nfl/rating.json`)

```json
{
  "k": 20,
  "hfa_elo": 40,
  "carryover": 0.6,
  "base": 1500.0,
  "w_sos": 0.3,
  "srs_min_games": 4
}
```

## Validation-span metrics (chosen config, 2020–2025, n=1615)

| metric | value |
|---|---|
| Brier | 0.22497 |
| Win accuracy | 63.78% |
| Margin MAE | 10.185 |
| Margin RMSE | 13.123 |

## Naive baselines (validation span, n=1615)

| baseline | Brier | Win acc | Margin MAE |
|---|---|---|---|
| Home-always (p=1.0 for home) | 0.4675 | 53.25% | — |
| Constant home-rate (train 2002–2019 rate = 0.5692, applied flat) | 0.2503 | 53.25% | 11.021 (using train mean home margin = +2.35 as constant) |
| Prior-season win% (p = 0.5 + (home_prior_wp − away_prior_wp)/2, clipped) | 0.2392 | 59.01% | — |

The tuned Elo/SoS model beats both naive baselines by a clear margin on every
metric: Brier 0.225 vs 0.250 (constant-rate) / 0.239 (prior-season win%); win
accuracy 63.8% vs 53.3% / 59.0%.

## Does the SoS blend beat pure Elo out-of-sample?

**Short answer: not on Brier/win-accuracy — those metrics are structurally
blend-invariant in this backtest's `run_backtest` — but yes, marginally, on
margin accuracy.**

Important structural finding (verified directly, not just inferred from the
tuning printout): `run_backtest`'s `brier`/`win_acc` are computed from
`e_home = g["e_home"]`, which is `run_elo`'s **pure-Elo** win probability
(`expected_home`) — it is never touched by `blend_cfg`/`w_sos`. Only the
`em` (expected margin, from `ratings.expected_margin`) — which feeds
`margin_mae`/`margin_rmse` — depends on `w_sos`. This is exactly the code
specified in the Task 8 brief (kept verbatim per Ruling 2), not a bug I
introduced.

Consequence: for any fixed `(k, hfa_elo, carryover)`, Brier and win_acc are
**identical across every `w_sos`** in the grid — confirmed directly at the
chosen point `(k=20, hfa_elo=40, carryover=0.6)`:

| w_sos | brier | win_acc | margin_mae | margin_rmse |
|---|---|---|---|---|
| 0.00 (pure Elo) | 0.224975 | 63.777% | 10.2082 | 13.1682 |
| 0.15 | 0.224975 | 63.777% | 10.1815 | 13.1251 |
| 0.30 (chosen) | 0.224975 | 63.777% | 10.1847 | 13.1230 |
| 0.45 | 0.224975 | 63.777% | 10.2345 | 13.1620 |

So "best blended Brier" (0.224975) trivially equals "best pure-Elo Brier"
(0.224975) at every Elo hyperparameter point, not just the winning one — the
Brier-based comparison the report was asked to make is a tie **by
construction**, not an empirical result. `w_sos=0.3` was selected by the
tuner's tie-break (min Brier, first minimum encountered) even though it has
zero effect on the selection metric.

On the metric the blend actually moves — margin MAE/RMSE — SoS blending
gives a small, real improvement over pure Elo at the winning Elo point:
margin MAE 10.185 (w_sos=0.3, the written config) vs 10.208 (pure Elo,
w_sos=0), a ~0.23% reduction; margin RMSE 13.123 vs 13.168, ~0.34%. `w_sos=0.15`
is marginally better still on margin (10.1815) than the persisted 0.30, within
noise of the coordinate search's per-parameter sweep granularity.

**Verdict:** the SoS blend does not move classification accuracy at all in
this backtest (structurally can't, given how `e_home` is wired), and moves
margin accuracy only marginally in its favor. This is a valid, spec-anticipated
outcome (the acceptance bar allows "blend didn't beat pure Elo" as a legitimate
finding) — here it's more precisely "blend is a no-op for win-prob metrics
and a marginal net positive for margin metrics," which should inform whether
P2 (distributions/shrinkage) wires margin predictions through `expected_margin`
(where the blend matters) versus win-probability through `expected_home`
directly (where it currently doesn't).

## Concerns / follow-ups for the controller

- `tune(train_df, valid_df, grid)` (kept verbatim per Ruling 2) never actually
  uses `train_df` — it evaluates and selects purely on `valid_df` Brier. This
  matches the brief's literal code and the unit test's contract, but means the
  walk-forward "train on 2002–2019, validate on 2020–2025" framing in the task
  description is aspirational for tuning purposes: the chosen hyperparameters
  are selected directly against the validation span's Brier, not an
  independent train-set objective. Flagging for awareness, not fixing (out of
  this task's touched-files scope).
- Because Brier/win_acc never respond to `w_sos`, any future "blend beat pure
  Elo" claim should be made on margin MAE/RMSE, not Brier — worth a P2 note if
  win-probability quality from the blend is ever wanted.
