# CFB priors: Arkansas / Missouri / Virginia backfill (2026-10-06)

`assets/cfb/priors.parquet` had 0 rows for ESPN ids 8 (Arkansas), 142 (Missouri), 258 (Virginia)
in every season 2015-2026 (the CFBD->ESPN resolver bug fixed in b81a103), so the live
cfb-ratings-v2 model served these teams with no preseason prior.

## Asset

`scripts/build_cfb_priors.py --seasons 2015 2026` re-run with the fixed resolver: 1535 -> 1571 rows
(+36 = 3 teams x 12 seasons). Every other team's 2015-2025 rows came back byte-identical. 2026 rows
had drifted at CFBD since the last build (132 `sp_rating` + 133 `forward_sos_shift`, neither
read by the leak-free prior for 2026; 1 `portal_net`, Rutgers 0.72 -> 1.58). The committed asset is
the old 1535 rows unchanged plus the 36 new rows, so the diff is only the fix. Every current FBS team
now has 2025 and 2026 rows (test `test_committed_priors_cover_every_current_fbs_team`).

## Held-out 2023-25 (gate harness, v2, gameline.json unchanged)

Configs: A = old priors + committed weights (live today); B = new priors + committed weights;
C = new priors + refit weights (`backtest_cfb_priors.py` on 2016-22 now selects sp_offset 1700,
w_recruiting 90, w_portal 60; decay unchanged 3.0 / 0.35. On the old asset it reproduces the
committed weights exactly).

Gate eval set (2169 games; `lines.parquet` also lacks the three teams, so it excludes them):

| config | margin MAE | total MAE | ATS | O/U | log-loss | ECE |
|---|---|---|---|---|---|---|
| A (live) | 12.607 | 13.098 | 48.96% | 51.00% | 0.5357 | 0.0157 |
| B | 12.607 | 13.098 | 48.96% | 51.00% | 0.5357 | 0.0156 |
| C | 12.603 | 13.098 | 48.92% | 51.00% | 0.5359 | 0.0167 |

Expanded set (2266 games; CFBD lines re-pulled to scratch, +349 line rows, all involving the three
teams, every existing line identical):

| config | margin MAE | total MAE | ATS | ECE | 97 three-team games: margin MAE / ATS |
|---|---|---|---|---|---|
| A (live) | 12.618 | 13.092 | 48.72% | 0.0155 | 12.88 / 43.2% |
| B | 12.608 | 13.092 | 49.12% | 0.0148 | 12.61 / 52.6% |
| C | 12.604 | 13.092 | 48.85% | 0.0165 | 12.62 / 47.4% |

Paired: A->B dMAE -0.011 (SE 0.012), dATS +0.41pp (SE 0.17); B->C dMAE -0.004 (SE 0.009),
dATS -0.27pp (SE 0.26), ECE +0.0017. The refit is noise on MAE, and its ATS and ECE are worse.

## Live 2026 week 6 (generate_cfb replay, no DB)

| game | A margin | B margin | C margin |
|---|---|---|---|
| Texas A&M @ Missouri | +0.60 | +1.47 | +1.40 |
| Tennessee @ Arkansas | -15.78 | -13.49 | -13.48 |
| Syracuse @ Virginia | +8.63 | +9.77 | +9.62 |

The other 53 games under B move at most 0.03 pts of margin (0.006 mean), and totals are unchanged.
The cause is the season z-score and league-mean shift from adding three teams. Under C the other
games move up to 0.34 pts. 2026 games already played (12 involving the three teams): margin MAE
A 18.64 -> B 19.38 (n=12; Arkansas has underperformed its prior).

Decision: ship B (this asset with the committed weights and decay; no refit, no version change).

## lines.parquet (same bug)

`lines.parquet` (CFBD closing/opening lines, 2015-2025) also lacked the three teams. Re-pulled with
the fixed resolver: 7553 -> 7902 rows (+349, all involving 8/142/258). Every existing row is
identical in every column. It is not read by the live game-line producer. It is read by:
- the gates and backtests. The gate eval set grows 2169 -> 2266 games, so
  `v3_gate.V2_EXPECTED` moves to 12.61 / 13.09 / 0.491 (measured 12.6075 / 13.0919 / 49.12% on
  the committed assets). The v3 / v3.1 gate JSONs stay as historical records on the old set.
- the daily team-context job (game log ATS/O-U history, rankings). The three teams go from 0 to
  about 120 lined games each, and every game-log row not involving them is identical.
