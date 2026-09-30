# Results-based power rating — fit and holdout

## CFB

Fitted on 2016–2023: cap 35.0, home edge 2.5, win-credit weight 0.0, last-season shrink 0.8 (sigma 16.7, prior weight k 1.0).

Holdout [2024, 2025] next-week margin MAE (n=1761):

| Rating | MAE |
|---|---|
| **New results rating** | **13.052** |
| Point differential + SOS only | 13.011 |
| Today's rankings rating (same games, n=1760) | 13.461 (new 13.057; diff -0.404, 95% CI [-0.657, -0.150]) |
| Closing line (reference, n=1450) | 12.077 (new 12.702) |

By phase: weeks 1–4 new 14.566 vs point-diff 14.193; weeks 5+ new 12.238 vs 12.376.

Win-credit weight vs accuracy (best other params per weight):

| Win credit (pts) | Fit MAE | Holdout MAE |
|---|---|---|
| 0 | 13.448 | 13.052 |
| 1 | 13.457 | 13.057 |
| 2 | 13.468 | 13.064 |
| 4 | 13.494 | 13.083 |
| 7 | 13.541 | 13.118 |
| 10 | 13.597 | 13.163 |

## NFL

Fitted on 2010–2023: cap 14.0, home edge 1.5, win-credit weight 0.0, last-season shrink 0.75 (sigma 13.5, prior weight k 1.0).

Holdout [2024, 2025] next-week margin MAE (n=544):

| Rating | MAE |
|---|---|
| **New results rating** | **10.354** |
| Point differential + SOS only | 10.687 |
| Today's rankings rating (same games, n=544) | 10.322 (new 10.354; diff +0.032, 95% CI [-0.158, +0.211]) |
| Closing line (reference, n=544) | 9.666 (new 10.354) |

By phase: weeks 1–4 new 10.739 vs point-diff 11.135; weeks 5+ new 10.235 vs 10.549.

Win-credit weight vs accuracy (best other params per weight):

| Win credit (pts) | Fit MAE | Holdout MAE |
|---|---|---|
| 0 | 10.607 | 10.354 |
| 1 | 10.630 | 10.356 |
| 2 | 10.654 | 10.363 |
| 4 | 10.705 | 10.408 |
| 6 | 10.761 | 10.432 |
| 9 | 10.861 | 10.476 |

## Served parameters (user decision 2026-09-30)

The fit's best win-credit weight was 0 in both sports (margin already carries the
information). The user chose **4 points** so wins themselves count (road upsets earn,
bad home losses cost), at a holdout cost of ~0.03 (CFB) / ~0.05 (NFL) points of MAE;
the other parameters are the best fit at that weight (`assets/context/results_power.json`,
`fit.served`). Holdout 2024–25 vs today's rankings rating at the served params:
CFB 13.088 vs 13.461 (−0.373, 95% CI [−0.629, −0.125], n=1760 — better); NFL 10.408 vs
10.322 (+0.086, 95% CI [−0.137, +0.304], n=544 — within noise, slightly behind the EPA
rating).

## Update — win bonus lightened 33% (user, 2026-09-30)

Win credit 4 → **2.67** points; cap / home edge / shrink refit at that weight
(`fit_results_power.py --beta 2.67`). Holdout 2024–25 vs today's rankings rating:

| | Served (2.67) | Today's rating | Diff (95% CI) |
|---|---|---|---|
| CFB (cap 35, hfa 2.5, rho 0.8) | 13.075 | 13.461 | −0.386 [−0.642, −0.136] |
| NFL (cap 14, hfa 1.5, rho 0.75) | 10.369 | 10.322 | +0.047 [−0.163, +0.248] |
