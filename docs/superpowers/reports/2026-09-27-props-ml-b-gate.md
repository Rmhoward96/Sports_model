# Props-ML B gate — 2026-09-27

**Verdict: PASS.** The props-ML B gate passes. Kept B-ladder rungs: blend; markets served from ML: pass_yds, rush_yds, rec_yds, receptions, rush_att, pass_tds, anytime_td. The served pipeline beats the current sim on all test seasons pooled and on 2025 alone.

## Run

- Test seasons: 2021, 2022, 2023, 2024, 2025; sims per game: 1000; base seed 42 with per-game seeded random streams (game-level alignment across runs).
- Refit weeks 1, 5, 9, 13, 17 (A and B models for (S, r) train only on rows strictly before (S, r)). Run tag: s2021-2025__every4__b7.
- Kept A toggles (a_gate.json): volume, efficiency, context, market.
- Baseline: current sim (season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5); 1359 games; the kept-A run covered exactly these.
- Population: baseline projected-usage gate, 7 markets: pass_yds 2511, rush_yds 4123, rec_yds 9038, receptions 8642, rush_att 4544, pass_tds 2423, anytime_td 12451.
- Blend weights and calibration maps for season S use only out-of-fold records of test seasons < S; the first test season is pure A with identity calibration.
- anytime_td is a 2-bin pmf, so its RPS column is its Brier score.
- Code: git 05a43388eabaf30a578847e99d4ed2c5d4152af8; features: player sha256 fef0aa42d927, team sha256 7f193a118c22.
- Elapsed: 192.4 min.

## A re-check (kept A vs current sim, 7 markets)

- Pooled relative RPS skill: **+0.0378** (95% clustered-bootstrap CI [+0.0328, +0.0427])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2024 → 0.2004 | 0.0047 → 0.0040 |
| pass_tds | 2423 | 0.6153 → 0.6155 | 0.0107 → 0.0106 |
| pass_yds | 2511 | 46.2721 → 44.7403 | 0.0332 → 0.0380 |
| rec_yds | 9038 | 18.5861 → 17.3787 | 0.0091 → 0.0140 |
| receptions | 8642 | 1.3224 → 1.2598 | 0.0046 → 0.0069 |
| rush_att | 4544 | 3.1861 → 2.9551 | 0.0207 → 0.0098 |
| rush_yds | 4123 | 18.6447 → 17.9538 | 0.0064 → 0.0096 |

Fail reasons: none.

Starting sources: pass_yds ml, rush_yds ml, rec_yds ml, receptions ml, rush_att ml, pass_tds ml, anytime_td ml

## B out-of-fold

- 175 fits (decay 1.0, max_iter 150) in 72.4 min; records without a B feature row (kept A): pass_yds 0, rush_yds 0, rec_yds 0, receptions 0, rush_att 0, pass_tds 0, anytime_td 0.
- B trains and predicts only on each market's pre-game role subset (pass: QB with pass-att EWM >= 10; rush: carries EWM >= 3; receiving: WR/TE/RB with targets EWM >= 2; anytime_td: all). Out-of-role records get B := A: pass_yds 54, rush_yds 145, rec_yds 432, receptions 184, rush_att 142, pass_tds 51, anytime_td 0.

## +B blend

Weight on A per test season:

| season | pass_yds | rush_yds | rec_yds | receptions | rush_att | pass_tds | anytime_td |
|---|---|---|---|---|---|---|---|
| 2021 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 2022 | 0.3 | 0.4 | 0.3 | 0.3 | 0.5 | 0.6 | 0.4 |
| 2023 | 0.2 | 0.4 | 0.3 | 0.3 | 0.5 | 0.6 | 0.2 |
| 2024 | 0.3 | 0.4 | 0.4 | 0.3 | 0.5 | 0.6 | 0.3 |
| 2025 | 0.3 | 0.4 | 0.4 | 0.3 | 0.5 | 0.6 | 0.2 |

Versus the kept composite (blend):

- Pooled relative RPS skill: **+0.0249** (95% clustered-bootstrap CI [+0.0224, +0.0275])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2004 → 0.1948 | 0.0040 → 0.0024 |
| pass_tds | 2423 | 0.6155 → 0.6068 | 0.0106 → 0.0101 |
| pass_yds | 2511 | 44.7403 → 42.3386 | 0.0380 → 0.0091 |
| rec_yds | 9038 | 17.3787 → 17.1602 | 0.0140 → 0.0082 |
| receptions | 8642 | 1.2598 → 1.2369 | 0.0069 → 0.0010 |
| rush_att | 4544 | 2.9551 → 2.8748 | 0.0098 → 0.0094 |
| rush_yds | 4123 | 17.9538 → 17.5829 | 0.0096 → 0.0071 |

Fail reasons: none.

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| anytime_td | 12451 | 0.0047 → 0.0024 |
| pass_tds | 2423 | 0.0107 → 0.0101 |
| pass_yds | 2511 | 0.0332 → 0.0091 |
| rec_yds | 9038 | 0.0091 → 0.0082 |
| receptions | 8642 | 0.0046 → 0.0010 |
| rush_att | 4544 | 0.0207 → 0.0094 |
| rush_yds | 4123 | 0.0064 → 0.0071 |

Fail reasons: none.

Rung result: **PASS**.

## +calibration

Map per test season (PIT isotonic; Platt [a, b] for anytime_td):

| season | pass_yds | rush_yds | rec_yds | receptions | rush_att | pass_tds | anytime_td |
|---|---|---|---|---|---|---|---|
| 2021 | identity | identity | identity | identity | identity | identity | identity |
| 2022 | pit | pit | pit | pit | pit | pit | [0.57, -0.41] |
| 2023 | pit | pit | pit | pit | pit | pit | [0.68, -0.32] |
| 2024 | pit | pit | pit | pit | pit | pit | [0.76, -0.20] |
| 2025 | pit | pit | pit | pit | pit | pit | [0.81, -0.14] |

Versus the kept composite (calibration):

- Pooled relative RPS skill: **-0.0007** (95% clustered-bootstrap CI [-0.0021, +0.0006])
- Decision: **FAIL**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.1948 → 0.1954 | 0.0024 → 0.0016 |
| pass_tds | 2423 | 0.6068 → 0.6065 | 0.0101 → 0.0051 |
| pass_yds | 2511 | 42.3386 → 42.4919 | 0.0091 → 0.0094 |
| rec_yds | 9038 | 17.1602 → 17.1247 | 0.0082 → 0.0053 |
| receptions | 8642 | 1.2369 → 1.2379 | 0.0010 → 0.0013 |
| rush_att | 4544 | 2.8748 → 2.8779 | 0.0094 → 0.0028 |
| rush_yds | 4123 | 17.5829 → 17.5670 | 0.0071 → 0.0039 |

Fail reasons:
- bootstrap 2.5% bound (-0.0021) <= 0

Calibration vs baseline (ECE may not exceed the current sim's by more than 0.005): **PASS**

| market | n | ECE baseline → cand |
|---|---:|---|
| anytime_td | 12451 | 0.0047 → 0.0016 |
| pass_tds | 2423 | 0.0107 → 0.0051 |
| pass_yds | 2511 | 0.0332 → 0.0094 |
| rec_yds | 9038 | 0.0091 → 0.0053 |
| receptions | 8642 | 0.0046 → 0.0013 |
| rush_att | 4544 | 0.0207 → 0.0028 |
| rush_yds | 4123 | 0.0064 → 0.0039 |

Fail reasons: none.

Rung result: **FAIL**.

## Final gate (served pipeline vs current sim)

A market is served from ML iff its RPS <= the baseline's AND its ECE <= the baseline's + 0.005 (all seasons).

| market | n | RPS base | RPS ML | ECE base | ECE ML | source | w_final | calibrate |
|---|---:|---:|---:|---:|---:|---|---:|---|
| pass_yds | 2511 | 46.2721 | 42.3386 | 0.0332 | 0.0091 | ml | 0.3 | False |
| rush_yds | 4123 | 18.6447 | 17.5829 | 0.0064 | 0.0071 | ml | 0.4 | False |
| rec_yds | 9038 | 18.5861 | 17.1602 | 0.0091 | 0.0082 | ml | 0.4 | False |
| receptions | 8642 | 1.3224 | 1.2369 | 0.0046 | 0.0010 | ml | 0.3 | False |
| rush_att | 4544 | 3.1861 | 2.8748 | 0.0207 | 0.0094 | ml | 0.4 | False |
| pass_tds | 2423 | 0.6153 | 0.6068 | 0.0107 | 0.0101 | ml | 0.5 | False |
| anytime_td | 12451 | 0.2024 | 0.1948 | 0.0047 | 0.0024 | ml | 0.2 | False |

### Selected (per-market sources above)

All seasons:

- Pooled relative RPS skill: **+0.0618** (95% clustered-bootstrap CI [+0.0567, +0.0666])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2024 → 0.1948 | 0.0047 → 0.0024 |
| pass_tds | 2423 | 0.6153 → 0.6068 | 0.0107 → 0.0101 |
| pass_yds | 2511 | 46.2721 → 42.3386 | 0.0332 → 0.0091 |
| rec_yds | 9038 | 18.5861 → 17.1602 | 0.0091 → 0.0082 |
| receptions | 8642 | 1.3224 → 1.2369 | 0.0046 → 0.0010 |
| rush_att | 4544 | 3.1861 → 2.8748 | 0.0207 → 0.0094 |
| rush_yds | 4123 | 18.6447 → 17.5829 | 0.0064 → 0.0071 |

Fail reasons: none.

2025 alone:

- Pooled relative RPS skill: **+0.0696** (95% clustered-bootstrap CI [+0.0580, +0.0810])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 2223 | 0.2114 → 0.2037 | 0.0112 → 0.0086 |
| pass_tds | 521 | 0.6313 → 0.6099 | 0.0121 → 0.0128 |
| pass_yds | 531 | 44.2068 → 40.0620 | 0.0382 → 0.0149 |
| rec_yds | 1567 | 18.2046 → 17.0029 | 0.0197 → 0.0157 |
| receptions | 1446 | 1.2938 → 1.1967 | 0.0162 → 0.0069 |
| rush_att | 844 | 3.2025 → 2.7833 | 0.0336 → 0.0146 |
| rush_yds | 752 | 19.1427 → 18.1613 | 0.0180 → 0.0146 |

Fail reasons: none.

pass = True

### Unselected (ML wherever the ladder kept it; no per-market swap)

All seasons:

- Pooled relative RPS skill: **+0.0618** (95% clustered-bootstrap CI [+0.0567, +0.0666])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 12451 | 0.2024 → 0.1948 | 0.0047 → 0.0024 |
| pass_tds | 2423 | 0.6153 → 0.6068 | 0.0107 → 0.0101 |
| pass_yds | 2511 | 46.2721 → 42.3386 | 0.0332 → 0.0091 |
| rec_yds | 9038 | 18.5861 → 17.1602 | 0.0091 → 0.0082 |
| receptions | 8642 | 1.3224 → 1.2369 | 0.0046 → 0.0010 |
| rush_att | 4544 | 3.1861 → 2.8748 | 0.0207 → 0.0094 |
| rush_yds | 4123 | 18.6447 → 17.5829 | 0.0064 → 0.0071 |

Fail reasons: none.

2025 alone:

- Pooled relative RPS skill: **+0.0696** (95% clustered-bootstrap CI [+0.0580, +0.0810])
- Decision: **PASS**

| market | n | RPS base → cand | ECE base → cand |
|---|---:|---|---|
| anytime_td | 2223 | 0.2114 → 0.2037 | 0.0112 → 0.0086 |
| pass_tds | 521 | 0.6313 → 0.6099 | 0.0121 → 0.0128 |
| pass_yds | 531 | 44.2068 → 40.0620 | 0.0382 → 0.0149 |
| rec_yds | 1567 | 18.2046 → 17.0029 | 0.0197 → 0.0157 |
| receptions | 1446 | 1.2938 → 1.1967 | 0.0162 → 0.0069 |
| rush_att | 844 | 3.2025 → 2.7833 | 0.0336 → 0.0146 |
| rush_yds | 752 | 19.1427 → 18.1613 | 0.0180 → 0.0146 |

Fail reasons: none.

pass = True

The selected verdict is optimistic by construction: its per-market sources are chosen on the same seasons it is scored on; the unselected decision is not.

final_pass = True (unselected: True)

## Caveats

- The final gate reuses the test seasons the ladder selected rungs on; it is not an independent holdout (rung selection and the final comparison share data).
- The baseline's cold-start prior uses is_starter = depth_team == 1, which carries the depth-chart era drift the learned p_depth_rank corrects (slot depth <= 2024 vs within-position pos_rank 2025+). This is baseline behavior, unchanged.
- Per-market sources are chosen on the same test seasons the final gate scores (a market is served from ML only if its own pooled RPS beats the baseline's).
- B models are fit unweighted with max_iter 150 (runtime rulings); A keeps the per-season tuned values from a_gate.json.
- The B role thresholds (dist_models.ROLE_SUBSETS) were chosen after a 2025 smoke run: that design freedom was exercised on the final-gate season, so the 2025 result is not fully independent of it.
- Serving simulates DEFAULT_N_SIMS (10,000) sims per game (generate_sim_nfl.py) where this gate scored 1000 per game; the served pmfs are smoother than the gated ones (same pipeline, less Monte Carlo noise).
