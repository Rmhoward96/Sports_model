# Props-ML fixed configuration volume, efficiency, context, market — 2026-09-25

**Verdict: PASS** for toggles {volume, efficiency, context, market} against the current sim (one fixed configuration; no ladder).

## Run

- Test seasons: 2025; sims per game: 1000; base seed 42 with per-game seeded random streams (game-level alignment across runs).
- Refit schedule: weekly (refit weeks 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18).
- Games: 272; records: 23276; sides with share fallback: 0.
- Code: git a52e8390b06e095aea542f23dd6e90b6a479aa24.
- Run tag: s2025__weekly (checkpoints records_<name>__s2025__weekly.parquet).
- questionable multiplier on learned shares: 1.0 (count models train on active-but-no-snap rows as 0 and see st_questionable; baseline shares keep production's 0.75).
- The final gate reuses the test seasons the ladder selected rungs on; it is not an independent holdout (rung selection and the final comparison share data).
- The baseline's cold-start prior uses is_starter = depth_team == 1, which carries the depth-chart era drift the learned p_depth_rank corrects (slot depth <= 2024 vs within-position pos_rank 2025+). This is baseline behavior, unchanged.

## Versus the current sim

- Pooled relative RPS skill: **+0.0364** (95% clustered-bootstrap CI [+0.0199, +0.0536])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 531 | 44.0732 → 43.5417 | 0.0394 → 0.0382 |
| rec_yds | 1585 | 18.1426 → 17.2212 | 0.0195 → 0.0204 |
| receptions | 1433 | 1.2993 → 1.2333 | 0.0176 → 0.0141 |
| rush_yds | 760 | 19.1182 → 18.5091 | 0.0179 → 0.0121 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 531 | 0.0394 → 0.0382 |
| rec_yds | 1585 | 0.0195 → 0.0204 |
| receptions | 1433 | 0.0176 → 0.0141 |
| rush_yds | 760 | 0.0179 → 0.0121 |

Fail reasons: none.

pass = True
