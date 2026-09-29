# Props-ML A gate — 2026-09-28

**Verdict: PASS.** The props-ML A gate passes. Kept rungs: context, efficiency, volume. The kept configuration beats the current sim on pooled seasons 2021, 2022, 2023, 2024, 2025 and on 2025 alone.

## Run

- Test seasons: 2021, 2022, 2023, 2024, 2025; sims per game: 1000; base seed 42 with per-game seeded random streams (game-level alignment across runs).
- Refit schedule: every 4 weeks (refit weeks 1, 5, 9, 13, 17); models for (S, r) train only on rows strictly before (S, r).
- Baseline: current sim, production config (season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5); 1359 games. Every candidate covered exactly these games.
- Population: baseline projected-usage gate (fixed for every candidate).
- A rung passes iff it beats the currently kept configuration (rung_decision) AND no market's ECE exceeds the baseline's by more than 0.005.
- Rung decisions (both checks above) use seasons 2021, 2022, 2023 only; seasons 2024, 2025 are reported in the final section but never decide a rung.
- Code: git 3696f7907af6689c939f936f5a71b4da5e5535f9; features: player sha256 9428d7eec11a (31829909 B), team sha256 d6d11d5b6c1d (2705762 B).
- Run tag: s2021-2025__every4 (checkpoints records_<name>__s2021-2025__every4_v2.parquet). Gate name: _v2.
- questionable multiplier on learned shares: 1.0 (count models train on active-but-no-snap rows as 0 and see st_questionable; baseline shares keep production's 0.75).
- The final gate reuses the test seasons the ladder selected rungs on; it is not an independent holdout (rung selection and the final comparison share data).
- The baseline's cold-start prior uses is_starter = depth_team == 1, which carries the depth-chart era drift the learned p_depth_rank corrects (slot depth <= 2024 vs within-position pos_rank 2025+). This is baseline behavior, unchanged.
- Elapsed: 560.1 min.

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
- Games: 1359; records: 107276; sides with share fallback: 19; run time 38.0 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0122** (95% clustered-bootstrap CI [+0.0060, +0.0186])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.7163 → 47.2472 | 0.0336 → 0.0326 |
| rec_yds | 5581 | 18.6748 → 18.4220 | 0.0078 → 0.0097 |
| receptions | 5391 | 1.3284 → 1.2963 | 0.0054 → 0.0074 |
| rush_yds | 2498 | 18.5601 → 18.1414 | 0.0074 → 0.0091 |

Fail reasons:
- market pass_yds: rps_c (47.2472) > 1.01 * rps_b (47.1835)

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0326 |
| rec_yds | 5581 | 0.0078 → 0.0097 |
| receptions | 5391 | 0.0054 → 0.0074 |
| rush_yds | 2498 | 0.0074 → 0.0091 |

Fail reasons: none.

Rung result: **FAIL**. Kept after this rung: none

### efficiency

- Candidate toggles: efficiency, volume
- Compared against: baseline (current sim)
- Games: 1359; records: 107276; sides with share fallback: 19; run time 40.0 min
- Note: volume was not kept; efficiency evaluated as {efficiency, volume} anyway

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0352** (95% clustered-bootstrap CI [+0.0268, +0.0432])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.7163 → 46.4392 | 0.0336 → 0.0368 |
| rec_yds | 5581 | 18.6748 → 17.5461 | 0.0078 → 0.0124 |
| receptions | 5391 | 1.3284 → 1.2748 | 0.0054 → 0.0068 |
| rush_yds | 2498 | 18.5601 → 17.9293 | 0.0074 → 0.0106 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0368 |
| rec_yds | 5581 | 0.0078 → 0.0124 |
| receptions | 5391 | 0.0054 → 0.0068 |
| rush_yds | 2498 | 0.0074 → 0.0106 |

Fail reasons: none.

Rung result: **PASS**. Kept after this rung: efficiency, volume

### context

- Candidate toggles: context, efficiency, volume
- Compared against: efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 42.2 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0035** (95% clustered-bootstrap CI [+0.0002, +0.0072])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.4392 → 46.0982 | 0.0368 → 0.0375 |
| rec_yds | 5581 | 17.5461 → 17.5396 | 0.0124 → 0.0125 |
| receptions | 5391 | 1.2748 → 1.2723 | 0.0068 → 0.0059 |
| rush_yds | 2498 | 17.9293 → 17.8493 | 0.0106 → 0.0121 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0375 |
| rec_yds | 5581 | 0.0078 → 0.0125 |
| receptions | 5391 | 0.0054 → 0.0059 |
| rush_yds | 2498 | 0.0074 → 0.0121 |

Fail reasons: none.

Rung result: **PASS**. Kept after this rung: context, efficiency, volume

### market

- Candidate toggles: context, efficiency, market, volume
- Compared against: context, efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 39.9 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0058** (95% clustered-bootstrap CI [+0.0019, +0.0102])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.0982 → 45.3405 | 0.0375 → 0.0391 |
| rec_yds | 5581 | 17.5396 → 17.4524 | 0.0125 → 0.0132 |
| receptions | 5391 | 1.2723 → 1.2688 | 0.0059 → 0.0054 |
| rush_yds | 2498 | 17.8493 → 17.8677 | 0.0121 → 0.0103 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **FAIL**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0391 |
| rec_yds | 5581 | 0.0078 → 0.0132 |
| receptions | 5391 | 0.0054 → 0.0054 |
| rush_yds | 2498 | 0.0074 → 0.0103 |

Fail reasons:
- market pass_yds: ece_cand (0.0391) > baseline ece (0.0336) + 0.005
- market rec_yds: ece_cand (0.0132) > baseline ece (0.0078) + 0.005

Rung result: **FAIL**. Kept after this rung: context, efficiency, volume

### matchup

- Candidate toggles: context, efficiency, matchup, volume
- Compared against: context, efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 40.0 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0027** (95% clustered-bootstrap CI [-0.0017, +0.0067])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.0982 → 45.9860 | 0.0375 → 0.0370 |
| rec_yds | 5581 | 17.5396 → 17.4610 | 0.0125 → 0.0119 |
| receptions | 5391 | 1.2723 → 1.2684 | 0.0059 → 0.0062 |
| rush_yds | 2498 | 17.8493 → 17.8353 | 0.0121 → 0.0128 |

Fail reasons:
- bootstrap 2.5% bound (-0.0017) <= 0

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **FAIL**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0370 |
| rec_yds | 5581 | 0.0078 → 0.0119 |
| receptions | 5391 | 0.0054 → 0.0062 |
| rush_yds | 2498 | 0.0074 → 0.0128 |

Fail reasons:
- market rush_yds: ece_cand (0.0128) > baseline ece (0.0074) + 0.005

Rung result: **FAIL**. Kept after this rung: context, efficiency, volume

### def_injuries

- Candidate toggles: context, def_injuries, efficiency, volume
- Compared against: context, efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 40.2 min

Versus the kept configuration:

- Pooled relative RPS skill: **-0.0005** (95% clustered-bootstrap CI [-0.0036, +0.0020])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.0982 → 46.2526 | 0.0375 → 0.0360 |
| rec_yds | 5581 | 17.5396 → 17.5014 | 0.0125 → 0.0124 |
| receptions | 5391 | 1.2723 → 1.2708 | 0.0059 → 0.0064 |
| rush_yds | 2498 | 17.8493 → 17.8887 | 0.0121 → 0.0105 |

Fail reasons:
- bootstrap 2.5% bound (-0.0036) <= 0

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0360 |
| rec_yds | 5581 | 0.0078 → 0.0124 |
| receptions | 5391 | 0.0054 → 0.0064 |
| rush_yds | 2498 | 0.0074 → 0.0105 |

Fail reasons: none.

Rung result: **FAIL**. Kept after this rung: context, efficiency, volume

### qb_profile

- Candidate toggles: context, efficiency, qb_profile, volume
- Compared against: context, efficiency, volume
- Games: 1359; records: 107276; sides with share fallback: 19; run time 40.3 min

Versus the kept configuration:

- Pooled relative RPS skill: **+0.0030** (95% clustered-bootstrap CI [-0.0013, +0.0071])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.0982 → 45.5554 | 0.0375 → 0.0388 |
| rec_yds | 5581 | 17.5396 → 17.4803 | 0.0125 → 0.0131 |
| receptions | 5391 | 1.2723 → 1.2678 | 0.0059 → 0.0065 |
| rush_yds | 2498 | 17.8493 → 17.9688 | 0.0121 → 0.0110 |

Fail reasons:
- bootstrap 2.5% bound (-0.0013) <= 0

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **FAIL**

| market | n | ECE baseline → cand |
|---|---:|---|
| pass_yds | 1491 | 0.0336 → 0.0388 |
| rec_yds | 5581 | 0.0078 → 0.0131 |
| receptions | 5391 | 0.0054 → 0.0065 |
| rush_yds | 2498 | 0.0074 → 0.0110 |

Fail reasons:
- market pass_yds: ece_cand (0.0388) > baseline ece (0.0336) + 0.005
- market rec_yds: ece_cand (0.0131) > baseline ece (0.0078) + 0.005

Rung result: **FAIL**. Kept after this rung: context, efficiency, volume

## Final gate (kept vs current sim)

### Pooled seasons 2021, 2022, 2023, 2024, 2025

- Pooled relative RPS skill: **+0.0396** (95% clustered-bootstrap CI [+0.0332, +0.0462])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 2511 | 46.2721 → 45.5450 | 0.0332 → 0.0363 |
| rec_yds | 9038 | 18.5861 → 17.4517 | 0.0091 → 0.0134 |
| receptions | 8642 | 1.3224 → 1.2629 | 0.0046 → 0.0071 |
| rush_yds | 4123 | 18.6447 → 17.9596 | 0.0064 → 0.0098 |

Fail reasons: none.

### 2025 alone

- Pooled relative RPS skill: **+0.0273** (95% clustered-bootstrap CI [+0.0112, +0.0437])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 531 | 44.2068 → 44.3312 | 0.0382 → 0.0394 |
| rec_yds | 1567 | 18.2046 → 17.3720 | 0.0197 → 0.0176 |
| receptions | 1446 | 1.2938 → 1.2375 | 0.0162 → 0.0137 |
| rush_yds | 752 | 19.1427 → 18.7090 | 0.0180 → 0.0151 |

Fail reasons: none.

### Decide seasons 2021, 2022, 2023 (the seasons rung decisions used)

- Pooled relative RPS skill: **+0.0386** (95% clustered-bootstrap CI [+0.0300, +0.0464])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 1491 | 46.7163 → 46.0982 | 0.0336 → 0.0375 |
| rec_yds | 5581 | 18.6748 → 17.5396 | 0.0078 → 0.0125 |
| receptions | 5391 | 1.3284 → 1.2723 | 0.0054 → 0.0059 |
| rush_yds | 2498 | 18.5601 → 17.8493 | 0.0074 → 0.0121 |

Fail reasons: none.

### Season 2024 alone (not used for rung decisions)

- Pooled relative RPS skill: **+0.0539** (95% clustered-bootstrap CI [+0.0400, +0.0677])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 489 | 47.1601 → 45.1762 | 0.0301 → 0.0383 |
| rec_yds | 1890 | 18.6407 → 17.2584 | 0.0095 → 0.0140 |
| receptions | 1805 | 1.3274 → 1.2552 | 0.0078 → 0.0091 |
| rush_yds | 873 | 18.4579 → 17.6296 | 0.0100 → 0.0065 |

Fail reasons:
- market pass_yds: ece_c (0.0383) > ece_b (0.0301) + 0.005

### Season 2025 alone (not used for rung decisions)

- Pooled relative RPS skill: **+0.0273** (95% clustered-bootstrap CI [+0.0112, +0.0437])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| pass_yds | 531 | 44.2068 → 44.3312 | 0.0382 → 0.0394 |
| rec_yds | 1567 | 18.2046 → 17.3720 | 0.0197 → 0.0176 |
| receptions | 1446 | 1.2938 → 1.2375 | 0.0162 → 0.0137 |
| rush_yds | 752 | 19.1427 → 18.7090 | 0.0180 → 0.0151 |

Fail reasons: none.

final_pass = True
