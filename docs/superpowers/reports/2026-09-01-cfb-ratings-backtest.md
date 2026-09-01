# CFB Elo/SoS walk-forward backtest — findings (P1 ship-gate)

Date: 2026-09-01
Script: `scripts/backtest_cfb_ratings.py`
Data: `assets/cfb/schedules.parquet` (seasons 2015–2025, all `game_type == "REG"`, 9,062 rows; ~134 teams incl. the single aggregated `"FCS"` opponent)
Train span: 2015–2022 (6,433 games) · OOS span: 2023–2024 (1,741 games) · 2025 excluded entirely (current/most-recent season in the asset — kept out of both tuning and evaluation per the brief, to avoid a thin/partial season distorting either).

Reused **read-only**: `sportsmodel.nfl.elo.{EloConfig, run_elo, elo_expected_margin}`,
`sportsmodel.nfl.srs.compute_srs`, `sportsmodel.nfl.ratings.{BlendConfig, expected_margin}`.
Nothing in `nfl/` or `mlb/` was modified.

## Leak-free walk-forward

Elo state and season-to-date SRS are walked forward across the full 2015–2024
span in season/week order: `run_elo` supplies each game's PRE-game Elo
(continuous across season boundaries via `EloConfig.carryover`), and SRS is
recomputed from only strictly-prior-week games that season (never same-week
or future games). `run_backtest(df, elo_cfg, blend_cfg, eval_seasons=...)`
always walks the full df forward for state, but only accumulates
Brier/win-acc/margin metrics for games whose season is in `eval_seasons` —
this lets the TRAIN-only tune and the OOS evaluation (which needs real
history carried in from the train seasons) share one function.

**One deliberate deviation from `backtest_nfl_elo.py`:** that script
refreshes SRS after *every single game*. At CFB scale (~830 games/season,
~134 teams vs NFL's ~267 games/season, 32 teams) that pattern measured
**~87 seconds per single train-span `run_backtest` call** — a coordinate
search would have taken well over an hour. Batching the SRS refresh to once
per completed week (still using only strictly-earlier weeks) cut a
train-span call to **~2.2 seconds** with the `margin_mae` output changing at
the 4th decimal place (15.3312 vs 15.3334 in a spot check) — i.e. it isn't
loosening the leak-free guarantee, and if anything is slightly more
conservative (it never lets an earlier-processed same-week game inform
another same-week game's SRS input, which the per-game version incidentally
does).

## Tuning

`nfl.elo.EloConfig` (`k`, `hfa_elo`, `carryover`) + `nfl.ratings.BlendConfig`
(`w_sos`, `srs_min_games`) tuned by **train-span margin MAE** via the same
3-pass coordinate search `backtest_nfl_elo.py` uses (one parameter swept at a
time, holding others at the running best), over CFB-appropriate ranges per
the task brief:

| param | grid searched |
|---|---|
| `k` | 12, 16, 20, 24, 28 |
| `hfa_elo` | 55, 70, 85, 100, 110 |
| `carryover` | 0.2, 0.3, 0.4, 0.5, 0.6 |
| `w_sos` | 0.0, 0.15, 0.3, 0.45 |
| `srs_min_games` | 3, 4, 6 |

**Selected: `k=28, hfa_elo=85, carryover=0.6, w_sos=0.45, srs_min_games=3`.**

## Tuned parameters (`assets/cfb/rating.json`)

```json
{
  "k": 28,
  "hfa_elo": 85,
  "carryover": 0.6,
  "base": 1500.0,
  "w_sos": 0.45,
  "srs_min_games": 3
}
```

## SoS blend vs pure Elo, OOS (2023–2024, n=1,741) — enforced-fair comparison

Pure-Elo counterfactual reuses the selected blend's exact `(k=28, hfa_elo=85,
carryover=0.6)` and only zeroes `w_sos` (same causal, same-Elo comparison
`backtest_nfl_elo.py` uses). Decision rule is strict (`<`, a tie doesn't ship
the blend).

| config | Brier | Win acc | Margin MAE | Margin RMSE |
|---|---|---|---|---|
| **Blended (shipped, w_sos=0.45, srs_min_games=3)** | 0.171673 | 74.04% | **13.7646** | 17.5090 |
| Pure Elo (w_sos=0, identical k/hfa/carryover) | 0.171673 | 74.04% | 14.0980 | 17.9481 |

Brier/win-acc are identical (structural — both come from `e_home`, which
`w_sos` never touches). The blend wins on margin MAE by 0.334 points and
ships.

## OOS margin MAE — tuned blend vs each baseline (2023–2024, n=1,741) — THE GATE

All four computed on the exact same 1,741 OOS games, from code
(`home_always_baseline`, `frozen_prior_season_baseline`,
`naive_margin_baseline` in `scripts/backtest_cfb_ratings.py`):

| model | Margin MAE | Margin RMSE |
|---|---|---|
| **Tuned blend (shipped)** | **13.7646** | 17.5090 |
| Prior-season rating (frozen at prior season's end, no in-season updates) | 14.9098 | 18.9044 |
| Home-always (margin = hfa/25 = 3.40 pts) | 17.5181 | 22.4278 |
| Naive-margin (predict 0) | 18.2941 | 23.3699 |

- **Tuned blend beats naive-margin OOS?** Yes — 13.76 vs 18.29 (−4.53 pts, −24.8%).
- **Tuned blend beats prior-season rating OOS?** Yes — 13.76 vs 14.91 (−1.15 pts, −7.7%).
- **Tuned blend beats home-always OOS?** Yes — 13.76 vs 17.52 (−3.75 pts, −21.4%).

`home_always`'s margin is derived from the shipped model's own `hfa_elo/25`
(3.40 points), not a separately-fit constant, per the brief's "predict margin
= home-field only" definition. `prior_season` freezes each team's rating at
the value `run_elo` already recorded for that team's first game of the OOS
season (i.e. post-carryover, pre-any-current-season-game) and never updates
it within the season — the natural "last known rating, no in-season
learning" competitor to the tuned in-season-updating blend.

## Concerns / follow-ups for the controller

- **Several tuned params landed on a grid edge**: `k=28` (max of range),
  `carryover=0.6` (max of range), `w_sos=0.45` (max of range), and
  `srs_min_games=3` (min of range) all sit at a boundary of the searched
  grid. This is consistent with the brief's expectation that CFB wants a
  higher `k`/`hfa_elo` and lower `carryover` than NFL, but boundary optima
  mean the coordinate search may not have found the true optimum — a wider
  grid (e.g. `k` up to 32–36, `w_sos` up to 0.6) could plausibly do better
  and is worth a follow-up sweep before treating these as final.
- OOS span is only 2 seasons (1,741 games) — smaller than the NFL backtest's
  6-season validation span. No significance testing was run on the blend-vs-
  pure-Elo or blend-vs-baseline gaps here (unlike the NFL report's paired-SE
  analysis); the gaps are large relative to NFL's blend-vs-pure-Elo gap
  (0.334 pts vs NFL's 0.051 pts) but this hasn't been formally tested against
  sampling noise.
- The `"FCS"` pseudo-team aggregates ~130+ distinct real FCS opponents under
  one rating/SRS identity, by design of the upstream team registry (P1 Task
  1). This is a known simplification carried through unmodified from the
  data-layer tasks, not something this backtest addresses.
