# ML sim vs Elo — NFL game-level gate — 2026-09-28

**Verdict: FAIL.** The ML sim does not meet the game-level gate against the Elo game model on seasons 2021, 2022, 2023, 2024, 2025 (1359 paired games). Failing: win_brier: ML 0.22577 > Elo 0.21492; cover_brier: ML 0.26028 > Elo 0.25229; over_brier: ML 0.25532 > Elo 0.25390; margin_mae: ML 10.3484 > 1.01 x Elo 9.8578; total_mae: ML 10.7742 > 1.01 x Elo 10.4584. The Elo model stays the NFL game-line source.

## Paired comparison

ML sim (walk-forward Props-1 A configuration) vs the Elo comparator (`backtest_nfl_gameline.per_game_predictions`, committed configs) on the same games; ML − Elo < 0 favours the ML sim. Gate: Brier point estimates ML ≤ Elo; MAE ML ≤ 1.01 × Elo.

| metric | paired n | ML | Elo | ML − Elo | 95% CI (cluster bootstrap) | rule | result |
|---|---:|---:|---:|---:|---|---|---|
| win_brier | 1359 | 0.2258 | 0.2149 | +0.0108 | [+0.0071, +0.0143] | ML ≤ Elo | FAIL |
| cover_brier | 1326 | 0.2603 | 0.2523 | +0.0080 | [+0.0043, +0.0115] | ML ≤ Elo | FAIL |
| over_brier | 1348 | 0.2553 | 0.2539 | +0.0014 | [-0.0023, +0.0052] | ML ≤ Elo | FAIL |
| margin_mae | 1359 | 10.3484 | 9.8578 | +0.4906 | [+0.3532, +0.6247] | ML ≤ 1.01 × Elo | FAIL |
| total_mae | 1359 | 10.7742 | 10.4584 | +0.3158 | [+0.1759, +0.4651] | ML ≤ 1.01 × Elo | FAIL |

## Caveats

- **In-sample edge for Elo.** Elo's configuration (`assets/nfl/rating.json`, `assets/nfl/gameline.json`: K/HFA/carryover, SoS blend, sigmas 13.424/13.576 and the closing-line shrink curves) was fit by `backtest_nfl_elo.py` / `backtest_nfl_gameline.py` on seasons ≤ 2019 and selected against 2020+ as its validation span, so these gate seasons informed Elo's configuration. Treat Elo's numbers here as in-sample; this biases the gate toward NOT switching.
- **The Elo comparator uses the closing line.** `per_game_predictions` shrinks Elo's margin/total toward each game's closing spread/total by week (margin w(week): start 0.95, floor 0.3, decay 0.1; total w(week): start 0.95, floor 0.3, decay 0.1); production `generate_nfl.py` serves Elo model-only (no market line). The gated comparator is therefore stronger than what is served; the as-served (no-shrink) comparison is below for information only.
- The ML A configuration (kept toggles, per-season tuned decay/max_iter) was selected in the props-ML A gate on these same seasons (player props, not game lines).
- ML probabilities are Monte Carlo estimates from 1000 sims per game.

## Informational: Elo as served (no closing-line shrink)

| metric | paired n | ML | Elo | ML − Elo | 95% CI (cluster bootstrap) | rule | result |
|---|---:|---:|---:|---:|---|---|---|
| win_brier | 1359 | 0.2258 | 0.2236 | +0.0022 | [-0.0006, +0.0049] | ML ≤ Elo | FAIL |
| cover_brier | 1326 | 0.2603 | 0.2593 | +0.0010 | [-0.0016, +0.0036] | ML ≤ Elo | FAIL |
| over_brier | 1348 | 0.2553 | 0.2665 | -0.0112 | [-0.0183, -0.0040] | ML ≤ Elo | pass |
| margin_mae | 1359 | 10.3484 | 10.1826 | +0.1658 | [+0.0629, +0.2733] | ML ≤ 1.01 × Elo | FAIL |
| total_mae | 1359 | 10.7742 | 10.9149 | -0.1407 | [-0.3855, +0.1180] | ML ≤ 1.01 × Elo | pass |

(pass under the gate rules: False; not the verdict)

## Coverage

- Scheduled completed REG games: 1359; ML sim: 1359; Elo: 1359; paired: 1359.
- Paired n per metric (games where both models are scorable): win_brier 1359, cover_brier 1326, over_brier 1348, margin_mae 1359, total_mae 1359.
- ML-missing (0): none.
- Elo-missing (0): none.
- Missing games are excluded from the paired comparison. Both models take their closing lines and actuals from the same nflverse schedule rows (checked per game).

## Standalone metrics (each model's own games)

| model | games | win Brier | cover Brier (n) | over Brier (n) | margin MAE | total MAE |
|---|---:|---:|---|---|---:|---:|
| ML sim | 1359 | 0.2258 | 0.2603 (1326) | 0.2553 (1348) | 10.348 | 10.774 |
| Elo | 1359 | 0.2149 | 0.2523 (1326) | 0.2539 (1348) | 9.858 | 10.458 |

## Run and identity

- Test seasons: 2021, 2022, 2023, 2024, 2025; sims per game: 1000; base seed 42 (per-game seeded streams).
- Refit weeks 1, 5, 9, 13, 17: models for (S, r) train only on rows strictly before (S, r).
- A configuration: toggles context, efficiency, market, volume; tuned (decay, max_iter) 2021: 1.0/300, 2022: 1.0/150, 2023: 0.8/150, 2024: 1.0/150, 2025: 0.8/300; learned-share questionable multiplier 1.0.
- Production sim settings: season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5.
- Code: git ccf94b493a54a386e65e71352ee8fb7b49a58a2d.
- Features: player sha256 fef0aa42d927 (25818364 B), team sha256 7f193a118c22 (1220518 B); same files a_gate.json was tuned on: False.
- Backtest sources: depth 226096 rows (46e6b3f1a214), injuries 34812 rows (7cc1f5342829), pbp 294989 rows (6cc159bc67fd), pfr2gsis 7821 rows (2b9b92822af4), schedules 1693 rows (baddabe5a701), snaps 157616 rows (228d0c2502c5), weekly 112450 rows (aee6a2c931ef).
- Elo configs: rating {"base": 1500.0, "carryover": 0.6, "hfa_elo": 55, "k": 16, "srs_min_games": 6, "w_sos": 0.3}; gameline {"bias_margin": 0.0, "bias_total": 0.0, "offset": 75, "sigma_margin": 13.424279251482753, "sigma_total": 13.575562615604422, "total_max": 120, "w_margin": {"decay": 0.1, "floor": 0.3, "start": 0.95}, "w_total": {"decay": 0.1, "floor": 0.3, "start": 0.95}}.
- ML run: 1359 games in 41.4 min; team-sides with share fallback 19; fresh (`game_records__s2021-2025__every4.parquet`).
- Elapsed: 81.0 min.

pass = False
