# Historical lean backtest — NFL / CFB / MLB (2026-10-16)

Question: if the site's lean rule had been applied to past games, how would it have done?

**Rule (copied from `site/js/shrink.js` shrinkLean):**
- shrunk = close + k·(model − close);
- round to the half point, unless the number equals the close;
- lean when the result is ≥ 0.25 off the closing number.

**Grading:**
- CFB and NFL: −110 against the closing number.
- MLB: the consensus closing line, at the median closing price of the side bet.

Script: `scripts/backtest_leans.py [cfb|nfl|mlb]`. Caches go in `data/backtest_leans/`, which is git-ignored.

## Data

| Sport | Model numbers | Games |
|---|---|---|
| CFB | `cfb-ratings` walk-forward, served numbers (the same rows `scripts/fit_game_shrink.py` fits on), closes from `assets/cfb/lines.parquet` | 5,719 (2018–2025 graded; 2016–17 used only to fit k) |
| NFL | ML-sim walk-forward records `data/props_ml/game_records__s2021-2025__every4.parquet`, nflverse closes | 1,359 (2021–2025) |
| MLB | `mlb-hybrid-v1` / `mlb-game-v2-defense` from `game_predictions`, last pre-game `odds_snapshot`, finals from MLB StatsAPI | 135 (Aug–Oct 2026) |

**Two ways of choosing k:**
- **Walk-forward:** each season uses k fitted only on the seasons before it, with the site's 300-game minimum. This is what a live bettor would actually have had.
- **Fixed-k sweep:** the same k every season. This is hindsight; it shows whether any k would have worked.

## Results

### Walk-forward (the honest number)

| Sport · market | k each season | Leans | Record | ROI | 95% CI |
|---|---|---|---|---|---|
| CFB spread | 0 every season (slope −0.016) | 0 | — | — | — |
| CFB total | 0.002–0.097 | 1,367 | 676-688-3 | **−5.4%** | [−10.5%, −0.3%] |
| NFL spread | 0 every season (slope −0.11) | 0 | — | — | — |
| NFL total | 0 every season (slope −0.08) | 0 | — | — | — |

CFB totals by season: −8.8% (2018), −5.2%, −6.1%, −7.6%, −14.4%, −0.1%, +3.7% (2024), +4.4% (2025). The last two seasons are positive, but every single-season CI spans zero.

### Fixed-k sweep (best case with hindsight)

| Sport · market | Best k | Leans | ROI at best k | ROI range across k = 0.05–1 |
|---|---|---|---|---|
| CFB spread | 0.05 | 2,076 | −6.1% | −6.1% to −7.1% |
| CFB total | 0.1 | 3,827 | −4.7% | −4.7% to −6.0% |
| NFL spread | 0.05 | 207 | +0.2% | +0.2% to −10.8% |
| NFL total | 0.3 | 1,079 | −7.0% | −7.0% to −14.0% |
| MLB run line | 0.75 | 126 | −6.8% | −6.8% to −13.3% |
| MLB total | 0.2 | 10 | −33% | −27% to −33% |

Restricting to larger edges (≥ 1, 2 or 3 points) does not help anywhere. The only sample with a positive point estimate is NFL spread, k = 0.3, edge ≥ 2: 139 leans, +0.3%, CI [−15%, +17%]. That is a handful of bets, picked after the fact.

## What it means

- **The models carry almost no information beyond the closing line.** The fitted slope of (actual − close) on (model − close) is:
  - CFB spread: −0.016
  - CFB total: +0.043
  - NFL spread: −0.11
  - NFL total: −0.08

  NFL per-season correlation is ≈ 0 or negative in every season, 2021–2025. The model's MAE is worse than the market's every season.
- **That is why the site's k is 0 for spreads.** The shrink is working as designed: it suppresses leans the history says would have lost about 7% at −110.
- **The current CFB totals leans have not shown an edge yet:**
  - k = 0.043 is the only nonzero point-market k.
  - Walk-forward, they are −5.4% over 8 seasons, though slightly positive in 2024–25.
  - The live 2026 record (63-49, 56.3%) is out-of-sample, since k was fitted through 2025. It is about 0.8 SD above the 52.4% break-even, which is well within noise.
- **MLB is too small to judge (135 games),** and nothing in it is positive. Totals are clearly negative: 36-61-8 at k = 1.
- **Closing-line caveat:** grading against the close is a high bar. Earlier, softer numbers (the early-week monitor, step 3) are the remaining place a stale-number edge could exist. This backtest says nothing about those.
