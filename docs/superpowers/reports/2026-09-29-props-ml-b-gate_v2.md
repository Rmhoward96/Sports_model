# Props-ML B gate — 2026-09-29

**Verdict: PASS.** The props-ML B gate passes. Kept B-ladder rungs: blend; markets served from ML: pass_yds, rush_yds, rec_yds, receptions, rush_att, pass_tds, anytime_td. The served pipeline beats the current sim on all test seasons pooled and on 2025 alone.

## Run

- Test seasons: 2021, 2022, 2023, 2024, 2025; sims per game: 1000; base seed 42 with per-game seeded random streams (game-level alignment across runs).
- Refit weeks 1, 5, 9, 13, 17 (A and B models for (S, r) train only on rows strictly before (S, r)). Run tag: s2021-2025__every4__b7; gate name _v2 (a_gate_v2.json).
- Decide seasons (A re-check sources, rung decisions, final per-market sources): 2021, 2022, 2023, 2024, 2025; blend weights and calibration maps stay walk-forward per season.
- Kept A toggles (a_gate_v2.json): volume, efficiency, context, market, matchup, def_injuries, qb_profile.
- Baseline: current sim (season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5); 1359 games; the kept-A run covered exactly these.
- Population: baseline projected-usage gate, 7 markets: pass_yds 2511, rush_yds 4123, rec_yds 9038, receptions 8642, rush_att 4544, pass_tds 2423, anytime_td 12451.
- Blend weights and calibration maps for season S use only out-of-fold records of test seasons < S; the first test season is pure A with identity calibration.
- anytime_td is a 2-bin pmf, so its RPS column is its Brier score.
- Code: git a3d9c9f622ece184558ee59105ee0f9cd7091515; features: player sha256 9428d7eec11a, team sha256 d6d11d5b6c1d.
- Elapsed: 199.7 min.

## A re-check (kept A vs current sim, 7 markets)

- Pooled relative RPS skill: **+0.0398** (95% clustered-bootstrap CI [+0.0348, +0.0447])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2024 → 0.2005 | 0.0047 → 0.0038 |
| pass_tds | 2423 | 0.6153 → 0.6130 | 0.0107 → 0.0110 |
| pass_yds | 2511 | 46.2721 → 44.5276 | 0.0332 → 0.0370 |
| rec_yds | 9038 | 18.5861 → 17.3403 | 0.0091 → 0.0137 |
| receptions | 8642 | 1.3224 → 1.2533 | 0.0046 → 0.0072 |
| rush_att | 4544 | 3.1861 → 2.9643 | 0.0207 → 0.0108 |
| rush_yds | 4123 | 18.6447 → 17.9234 | 0.0064 → 0.0101 |

Fail reasons: none.

Starting sources: pass_yds ml, rush_yds ml, rec_yds ml, receptions ml, rush_att ml, pass_tds ml, anytime_td ml

## B out-of-fold

- 175 fits (decay 1.0, max_iter 150) in 76.0 min; records without a B feature row (kept A): pass_yds 0, rush_yds 0, rec_yds 0, receptions 0, rush_att 0, pass_tds 0, anytime_td 0.
- B trains and predicts only on each market's pre-game role subset (pass: QB with pass-att EWM >= 10; ROLE_SUBSETS_V2: also a pass attempt in the last 10 games; rush: carries EWM >= 3; receiving: WR/TE/RB with targets EWM >= 2; anytime_td: all). Out-of-role records get B := A: pass_yds 58, rush_yds 145, rec_yds 432, receptions 184, rush_att 142, pass_tds 55, anytime_td 0.

## +B blend

Weight on A per test season:

| season | pass_yds | rush_yds | rec_yds | receptions | rush_att | pass_tds | anytime_td |
|---|---|---|---|---|---|---|---|
| 2021 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2022 | 0.3 | 0.3 | 0.3 | 0.3 | 0.5 | 0.7 | 0.3 |
| 2023 | 0.2 | 0.3 | 0.3 | 0.3 | 0.5 | 0.6 | 0.2 |
| 2024 | 0.3 | 0.4 | 0.4 | 0.3 | 0.5 | 0.6 | 0.2 |
| 2025 | 0.2 | 0.4 | 0.4 | 0.3 | 0.4 | 0.6 | 0.2 |

Versus the kept composite (blend):

- Pooled relative RPS skill: **+0.0247** (95% clustered-bootstrap CI [+0.0220, +0.0271])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2005 → 0.1945 | 0.0038 → 0.0022 |
| pass_tds | 2423 | 0.6130 → 0.6054 | 0.0110 → 0.0113 |
| pass_yds | 2511 | 44.5276 → 42.2069 | 0.0370 → 0.0101 |
| rec_yds | 9038 | 17.3403 → 17.1389 | 0.0137 → 0.0083 |
| receptions | 8642 | 1.2533 → 1.2299 | 0.0072 → 0.0021 |
| rush_att | 4544 | 2.9643 → 2.8757 | 0.0108 → 0.0111 |
| rush_yds | 4123 | 17.9234 → 17.5941 | 0.0101 → 0.0067 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| anytime_td | 12451 | 0.0047 → 0.0022 |
| pass_tds | 2423 | 0.0107 → 0.0113 |
| pass_yds | 2511 | 0.0332 → 0.0101 |
| rec_yds | 9038 | 0.0091 → 0.0083 |
| receptions | 8642 | 0.0046 → 0.0021 |
| rush_att | 4544 | 0.0207 → 0.0111 |
| rush_yds | 4123 | 0.0064 → 0.0067 |

Fail reasons: none.

Rung result: **PASS**.

## +calibration

Map per test season (PIT isotonic; Platt [a, b] for anytime_td):

| season | pass_yds | rush_yds | rec_yds | receptions | rush_att | pass_tds | anytime_td |
|---|---|---|---|---|---|---|---|
| 2021 | identity | identity | identity | identity | identity | identity | identity |
| 2022 | pit | pit | pit | pit | pit | pit | [0.56, -0.41] |
| 2023 | pit | pit | pit | pit | pit | pit | [0.68, -0.31] |
| 2024 | pit | pit | pit | pit | pit | pit | [0.76, -0.20] |
| 2025 | pit | pit | pit | pit | pit | pit | [0.82, -0.13] |

Versus the kept composite (calibration):

- Pooled relative RPS skill: **-0.0008** (95% clustered-bootstrap CI [-0.0023, +0.0006])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.1945 → 0.1952 | 0.0022 → 0.0019 |
| pass_tds | 2423 | 0.6054 → 0.6050 | 0.0113 → 0.0058 |
| pass_yds | 2511 | 42.2069 → 42.3849 | 0.0101 → 0.0098 |
| rec_yds | 9038 | 17.1389 → 17.1005 | 0.0083 → 0.0047 |
| receptions | 8642 | 1.2299 → 1.2310 | 0.0021 → 0.0020 |
| rush_att | 4544 | 2.8757 → 2.8790 | 0.0111 → 0.0043 |
| rush_yds | 4123 | 17.5941 → 17.5757 | 0.0067 → 0.0032 |

Fail reasons:
- bootstrap 2.5% bound (-0.0023) <= 0

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| anytime_td | 12451 | 0.0047 → 0.0019 |
| pass_tds | 2423 | 0.0107 → 0.0058 |
| pass_yds | 2511 | 0.0332 → 0.0098 |
| rec_yds | 9038 | 0.0091 → 0.0047 |
| receptions | 8642 | 0.0046 → 0.0020 |
| rush_att | 4544 | 0.0207 → 0.0043 |
| rush_yds | 4123 | 0.0064 → 0.0032 |

Fail reasons: none.

Rung result: **FAIL**.

## Final gate (served pipeline vs current sim)

A market is served from ML iff its RPS <= the baseline's AND its ECE <= the baseline's + 0.005 on the decide seasons (2021, 2022, 2023, 2024, 2025; the table shows those scores).

| market | n | RPS base | RPS ML | ECE base | ECE ML | source | w_final | calibrate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| pass_yds | 2511 | 46.2721 | 42.2069 | 0.0332 | 0.0101 | ml | 0.2 | False |
| rush_yds | 4123 | 18.6447 | 17.5941 | 0.0064 | 0.0067 | ml | 0.4 | False |
| rec_yds | 9038 | 18.5861 | 17.1389 | 0.0091 | 0.0083 | ml | 0.4 | False |
| receptions | 8642 | 1.3224 | 1.2299 | 0.0046 | 0.0021 | ml | 0.3 | False |
| rush_att | 4544 | 3.1861 | 2.8757 | 0.0207 | 0.0111 | ml | 0.4 | False |
| pass_tds | 2423 | 0.6153 | 0.6054 | 0.0107 | 0.0113 | ml | 0.5 | False |
| anytime_td | 12451 | 0.2024 | 0.1945 | 0.0047 | 0.0022 | ml | 0.2 | False |

### Selected (per-market sources above)

All seasons:

- Pooled relative RPS skill: **+0.0635** (95% clustered-bootstrap CI [+0.0584, +0.0683])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2024 → 0.1945 | 0.0047 → 0.0022 |
| pass_tds | 2423 | 0.6153 → 0.6054 | 0.0107 → 0.0113 |
| pass_yds | 2511 | 46.2721 → 42.2069 | 0.0332 → 0.0101 |
| rec_yds | 9038 | 18.5861 → 17.1389 | 0.0091 → 0.0083 |
| receptions | 8642 | 1.3224 → 1.2299 | 0.0046 → 0.0021 |
| rush_att | 4544 | 3.1861 → 2.8757 | 0.0207 → 0.0111 |
| rush_yds | 4123 | 18.6447 → 17.5941 | 0.0064 → 0.0067 |

Fail reasons: none.

2025 alone:

- Pooled relative RPS skill: **+0.0700** (95% clustered-bootstrap CI [+0.0584, +0.0818])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 2223 | 0.2114 → 0.2036 | 0.0112 → 0.0081 |
| pass_tds | 521 | 0.6313 → 0.6113 | 0.0121 → 0.0121 |
| pass_yds | 531 | 44.2068 → 40.0179 | 0.0382 → 0.0168 |
| rec_yds | 1567 | 18.2046 → 16.9800 | 0.0197 → 0.0169 |
| receptions | 1446 | 1.2938 → 1.1931 | 0.0162 → 0.0074 |
| rush_att | 844 | 3.2025 → 2.7895 | 0.0336 → 0.0141 |
| rush_yds | 752 | 19.1427 → 18.1435 | 0.0180 → 0.0157 |

Fail reasons: none.

pass = True

### Unselected (ML wherever the ladder kept it; no per-market swap)

All seasons:

- Pooled relative RPS skill: **+0.0635** (95% clustered-bootstrap CI [+0.0584, +0.0683])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2024 → 0.1945 | 0.0047 → 0.0022 |
| pass_tds | 2423 | 0.6153 → 0.6054 | 0.0107 → 0.0113 |
| pass_yds | 2511 | 46.2721 → 42.2069 | 0.0332 → 0.0101 |
| rec_yds | 9038 | 18.5861 → 17.1389 | 0.0091 → 0.0083 |
| receptions | 8642 | 1.3224 → 1.2299 | 0.0046 → 0.0021 |
| rush_att | 4544 | 3.1861 → 2.8757 | 0.0207 → 0.0111 |
| rush_yds | 4123 | 18.6447 → 17.5941 | 0.0064 → 0.0067 |

Fail reasons: none.

2025 alone:

- Pooled relative RPS skill: **+0.0700** (95% clustered-bootstrap CI [+0.0584, +0.0818])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 2223 | 0.2114 → 0.2036 | 0.0112 → 0.0081 |
| pass_tds | 521 | 0.6313 → 0.6113 | 0.0121 → 0.0121 |
| pass_yds | 531 | 44.2068 → 40.0179 | 0.0382 → 0.0168 |
| rec_yds | 1567 | 18.2046 → 16.9800 | 0.0197 → 0.0169 |
| receptions | 1446 | 1.2938 → 1.1931 | 0.0162 → 0.0074 |
| rush_att | 844 | 3.2025 → 2.7895 | 0.0336 → 0.0141 |
| rush_yds | 752 | 19.1427 → 18.1435 | 0.0180 → 0.0157 |

Fail reasons: none.

pass = True

The selected verdict is optimistic by construction where its per-market sources were chosen on seasons it is scored on (the decide seasons above); the unselected decision is not.

final_pass = True (unselected: True)

## Caveats

- The final gate reuses the test seasons the ladder selected rungs on; it is not an independent holdout (rung selection and the final comparison share data).
- The baseline's cold-start prior uses is_starter = depth_team == 1, which carries the depth-chart era drift the learned p_depth_rank corrects (slot depth <= 2024 vs within-position pos_rank 2025+). This is baseline behavior, unchanged.
- Per-market sources are chosen on the decide seasons (a market is served from ML only if its own RPS there beats the baseline's); when those include the final-gate seasons the final gate is not an independent check of that choice.
- B models are fit unweighted with max_iter 150 (runtime rulings); A keeps the per-season tuned values from a_gate.json.
- The B role thresholds (dist_models.ROLE_SUBSETS) were chosen after a 2025 smoke run: that design freedom was exercised on the final-gate season, so the 2025 result is not fully independent of it.
- Serving simulates DEFAULT_N_SIMS (10,000) sims per game (generate_sim_nfl.py) where this gate scored 1000 per game; the served pmfs are smoother than the gated ones (same pipeline, less Monte Carlo noise).
