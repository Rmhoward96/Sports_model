# Props-ML A gate — 2026-09-24

**Verdict: PASS.** The props-ML A gate passes. Kept rungs: context, efficiency, market, volume. The kept configuration beats the current sim on pooled seasons 2021, 2022, 2023, 2024, 2025 and on 2025 alone.

## Run

- Test seasons: 2021, 2022, 2023, 2024, 2025; sims per game: 1000; base seed 42 with per-game seeded random streams (game-level alignment across runs).
- Refit schedule: every 4 weeks (refit weeks 1, 5, 9, 13, 17); models for (S, r) train only on rows strictly before (S, r).
- Baseline: current sim, production config (season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5); 1359 games. Every candidate covered exactly these games.
- Population: baseline projected-usage gate (fixed for every candidate).
- A rung passes iff it beats the currently kept configuration (rung_decision) AND no market's ECE exceeds the baseline's by more than 0.005.
- Code: git a976e8822afa0906e394ec320cc355b5a7c0b0b5; features: player sha256 6ba250c35b60 (25767614 B), team sha256 5b7afd3a670b (1216370 B).
- Run tag: s2021-2025__every4 (checkpoints records_<name>__s2021-2025__every4.parquet).
- questionable multiplier on learned shares: 1.0 (count models train on active-but-no-snap rows as 0 and see st_questionable; baseline shares keep production's 0.75).
- The final gate reuses the test seasons the ladder selected rungs on; it is not an independent holdout (rung selection and the final comparison share data).
- The baseline's cold-start prior uses is_starter = depth_team == 1, which carries the depth-chart era drift the learned p_depth_rank corrects (slot depth <= 2024 vs within-position pos_rank 2025+). This is baseline behavior, unchanged.
- Elapsed: 341.5 min.

Tuned (decay, max_iter) per test season (tuned once with toggles {volume}, reused for every rung):

| season | decay | max_iter |
|---|---:|---:|
| 2021 | 1.0 | 300 |
| 2022 | 1.0 | 150 |
| 2023 | 0.8 | 150 |
| 2024 | 1.0 | 150 |
| 2025 | 0.8 | 300 |

## Ladder

### volume

- Candidate toggles: volume
- Compared against: baseline (current sim)
- Games: 1359; records: 107276; sides with share fallback: 19; run time 39.4 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0111** (95% clustered-bootstrap CI [+0.0058, +0.0164])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 2511 | 46.2721 → 46.9642 | 0.0332 → 0.0315 |
| rec_yds | 9038 | 18.5861 → 18.3440 | 0.0091 → 0.0118 |
| receptions | 8642 | 1.3224 → 1.2885 | 0.0046 → 0.0070 |
| rush_yds | 4123 | 18.6447 → 18.2618 | 0.0064 → 0.0070 |

Fail reasons:
- market pass_yds: rps_c (46.9642) > 1.01 * rps_b (46.7348)

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 2511 | 0.0332 → 0.0315 |
| rec_yds | 9038 | 0.0091 → 0.0118 |
| receptions | 8642 | 0.0046 → 0.0070 |
| rush_yds | 4123 | 0.0064 → 0.0070 |

Fail reasons: none.

Rung result: **FAIL**. Kept after this rung: none

### efficiency

- Candidate toggles: efficiency, volume
- Compared against: baseline (current sim)
- Games: 1359; records: 107276; sides with share fallback: 19; run time 41.8 min
- Note: volume was not kept; efficiency evaluated as {efficiency, volume} anyway

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0365** (95% clustered-bootstrap CI [+0.0303, +0.0433])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 2511 | 46.2721 → 45.8576 | 0.0332 → 0.0358 |
| rec_yds | 9038 | 18.5861 → 17.4580 | 0.0091 → 0.0131 |
| receptions | 8642 | 1.3224 → 1.2658 | 0.0046 → 0.0068 |
| rush_yds | 4123 | 18.6447 → 18.0156 | 0.0064 → 0.0091 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 2511 | 0.0332 → 0.0358 |
| rec_yds | 9038 | 0.0091 → 0.0131 |
| receptions | 8642 | 0.0046 → 0.0068 |
| rush_yds | 4123 | 0.0064 → 0.0091 |

Fail reasons: none.

Rung result: **PASS**. Kept after this rung: efficiency, volume

### context

- Candidate toggles: context, efficiency, volume
- Compared against: efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 42.0 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0034** (95% clustered-bootstrap CI [+0.0010, +0.0060])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 2511 | 45.8576 → 45.5317 | 0.0358 → 0.0363 |
| rec_yds | 9038 | 17.4580 → 17.4485 | 0.0131 → 0.0134 |
| receptions | 8642 | 1.2658 → 1.2627 | 0.0068 → 0.0070 |
| rush_yds | 4123 | 18.0156 → 17.9507 | 0.0091 → 0.0097 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 2511 | 0.0332 → 0.0363 |
| rec_yds | 9038 | 0.0091 → 0.0134 |
| receptions | 8642 | 0.0046 → 0.0070 |
| rush_yds | 4123 | 0.0064 → 0.0097 |

Fail reasons: none.

Rung result: **PASS**. Kept after this rung: context, efficiency, volume

### market

- Candidate toggles: context, efficiency, market, volume
- Compared against: context, efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 42.1 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0059** (95% clustered-bootstrap CI [+0.0031, +0.0088])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 2511 | 45.5317 → 44.7413 | 0.0363 → 0.0379 |
| rec_yds | 9038 | 17.4485 → 17.3770 | 0.0134 → 0.0139 |
| receptions | 8642 | 1.2627 → 1.2598 | 0.0070 → 0.0069 |
| rush_yds | 4123 | 17.9507 → 17.9557 | 0.0097 → 0.0094 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 2511 | 0.0332 → 0.0379 |
| rec_yds | 9038 | 0.0091 → 0.0139 |
| receptions | 8642 | 0.0046 → 0.0069 |
| rush_yds | 4123 | 0.0064 → 0.0094 |

Fail reasons: none.

Rung result: **PASS**. Kept after this rung: context, efficiency, market, volume

## Final gate (kept vs current sim)

### Pooled seasons 2021, 2022, 2023, 2024, 2025

- Pooled relative RPS skill: **+0.0456** (95% clustered-bootstrap CI [+0.0385, +0.0523])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 2511 | 46.2721 → 44.7413 | 0.0332 → 0.0379 |
| rec_yds | 9038 | 18.5861 → 17.3770 | 0.0091 → 0.0139 |
| receptions | 8642 | 1.3224 → 1.2598 | 0.0046 → 0.0069 |
| rush_yds | 4123 | 18.6447 → 17.9557 | 0.0064 → 0.0094 |

Fail reasons: none.

### 2025 alone

- Pooled relative RPS skill: **+0.0356** (95% clustered-bootstrap CI [+0.0185, +0.0529])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 531 | 44.2068 → 43.3965 | 0.0382 → 0.0378 |
| rec_yds | 1567 | 18.2046 → 17.2606 | 0.0197 → 0.0187 |
| receptions | 1446 | 1.2938 → 1.2319 | 0.0162 → 0.0130 |
| rush_yds | 752 | 19.1427 → 18.6754 | 0.0180 → 0.0151 |

Fail reasons: none.

final_pass = True
