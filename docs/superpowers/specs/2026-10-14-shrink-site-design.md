# Shrink game-level model numbers on the site (profit project, step 2 Part B) — design spec

Date: 2026-10-14 · Status: design approved in chat (two sections); awaiting user review of this file

**Evidence.** From `docs/superpowers/reports/2026-10-13-edge-audit.md`:
- **§3.** Against the close, the model-minus-market coefficient is +0.04 (SE 0.16) for CFB margin and +0.00 for NFL. The CFB moneyline is compressed relative to the market (slope 0.75), and CFB value-side moneylines lose 18.9%.
- **§7 item 2.** The audit recommends shrinking every model probability toward the market.

**Related work.** Part A (props), `docs/superpowers/specs/2026-10-13-shrink-props-design.md`, is merged as f4b8899.

## Goal

Every game-level model number the site shows as a fair price or an edge becomes market-weighted. That covers the CappingAlpha fair moneyline, spread and total, Edge, Alpha and the game-page Read. The shrink factor k is fitted walk-forward on history. As a result the site never presents a model edge larger than the evidence supports. The raw model projection stays visible, clearly labelled as raw.

## User decisions (2026-10-13/14)

- **Switch now, no shadow period.** This is carried over from Part A.
- **Shrunk everywhere.** The CappingAlpha columns, Edge, Alpha and the Read all use shrunk numbers, so they always agree. The raw projection stays on the game page, labelled.
- **Minimum sample.** A sport/market with fewer than 300 graded games of the current model gets k = 0.

## Current state (from code exploration)

**Where game-level model numbers enter the site:**
- They come from `game_predictions` through the `predictions_current` and `predictions_any` views: `home_win_prob`, `margin_dist`, `total_dist`, plus `market_spread` and `market_total`.
- They are computed against the market in the browser:
  - `site/js/pages/board.js` `boardAlpha`, using the moneyline against `game_moneylines_current`;
  - the sport-page table's CappingAlpha and Edge columns;
  - `site/js/pages/game.js` (`gmOdds`, `gmCandidates`, `gameRead`, the Alpha card);
  - `alphaScore()` in `site/js/metrics.js`.
- The +EV board and Top Alpha Edges use Pinnacle no-vig only, and props shrink server-side as of Part A.

**History available for fitting:**
- **CFB:** the v2 model replayed walk-forward (`src/sportsmodel/cfb/walkforward.py` `raw_model_predictions`; 2015 is warm-up) against `assets/cfb/lines.parquet`. That file has closing `market_spread` and `market_total` from 2015, and `ml_home` / `ml_away` from 2021. Results come from `assets/cfb/schedules.parquet`.
- **NFL:** the current game model (the ML sim, `ML_GAME_LINES=on` since 2026-09-28) has about 6 weeks of graded games. Older-model history does not describe it.
- **MLB:** the game model has negligible graded history.

## 1. Fitting k

**Spread and total, in points.** For each graded game:
- x = model − close (model expected home margin minus the closing line expressed as expected home margin; for totals, model total minus closing total);
- y = actual − close.

Then:

> `k_pts = clamp( Σxy / Σx², 0, 1 )`

This is the no-intercept least-squares slope: how much of the model's disagreement with the close showed up in results. If Σx² = 0, k = 0.

**Moneyline, in probability.** The same closed form as Part A:

> `k_ml = clamp( Σ(y − m)(p − m) / Σ(p − m)², 0, 1 )`

- p is the model's home win probability, using the same conversion the site and served model use.
- m is the two-way no-vig probability from the closing `ml_home` / `ml_away`.
- y = 1 if the home team won.
- Ties are excluded, and so are rows with an invalid American price (|price| < 100).

**CFB sample and walk-forward.**
- Model numbers come from `raw_model_predictions` for v2 as served (`GAME_MODEL_VERSION = "cfb-ratings-v2"`), with seasons 2016–2025 graded (2015 is warm-up).
- The sign conventions must match those verified in step 3: CFB `market_spread` is positive when the home team is favoured, i.e. it is the expected home margin.
- **Per-season stability report:** for each held-out season S from 2018 to 2025, fit on seasons before S and report k_S. Also report the S out-of-sample MAE (spreads and totals) or Brier score (moneyline) for raw, shrunk and market.
- **Served k:** fitted on all graded seasons, 2016–2025 for spread and total and 2021–2025 for moneyline.
- **Minimum:** fewer than 300 games for a market means k = 0.

**NFL and MLB.** k = 0 for spread, total and moneyline, because the current model has fewer than 300 graded games. The fit script writes these zeros explicitly, along with the reason.

**Outputs, from `scripts/fit_game_shrink.py`.** The script reads only committed assets. It touches no database and no network.
- `assets/game_shrink_k.json`: per `(sport, market)` it records `k`, `n`, the seasons used, the per-season k and out-of-sample metrics, `model_version` and `fitted_at`.
- `site/js/shrink-k.js`: the same k values as a frozen ES module. A test asserts that it matches the JSON.

**Refit.** The script is run manually: at season's end, or when NFL reaches 300 graded games of its current model. There is no scheduled job.

## 2. The site

**A shared helper, `site/js/shrink.js`.** It provides:
- `shrinkProb(p, m, k)`, which is `clip(m + k(p − m), 0.01, 0.99)`;
- `shrinkLine(model, market, k)`, which is `market + k(model − market)`;
- `fairAmerican(p)`.

Each takes k from `shrink-k.js` by `(sport, market)`. An unknown sport or market gives k = 0.

**Fair moneyline when there is no moneyline market.** The shrunk fair moneyline needs a no-vig market probability. When a game has a market spread but no moneyline, the moneyline column shows the market-derived value if the existing code already derives one. Otherwise it shows "—". The plan decides from the existing code.

**Rule.** Every place that compares a model number with a market number, or shows a model fair price beside a market, goes through the helper. That includes Edge, Alpha, sorting, the Read and the dashboard.

**No market line.** The model value is shown labelled "projection". No edge or Alpha is shown.

**Per page:**
- **Sport pages** (`board.js` and friends):
  - Upcoming rows show the shrunk CappingAlpha ML, Spread and Total. Edge is shrunk minus market.
  - Alpha Score, and sorting by Alpha or Edge, use the shrunk edge.
  - Past-week result rows keep the recorded model picks and W/L. That is history, and it is unchanged.
  - Top Alpha Edges and Market Pulse are unchanged.
- **Game page** (`game.js`):
  - The CappingAlpha Read, the Alpha card and the odds and candidate edges use shrunk numbers.
  - A new small line, `Raw model projection: margin, total, win %`, is clearly labelled.
- **Dashboard:** any game-level model edge it shows uses the helper.
- **Unchanged:**
  - Track Record and the historical panels (Model vs Market, Model Projections bands), which stay raw;
  - the props board;
  - the +EV board;
  - early tickets.

**House rules.**
- Escaping uses `ctxEsc`.
- Desktop first.
- The cache key is bumped once at the end. `smoke.test.mjs` pins it.

## Testing

**Python, network-free and DB-free:**
- The point and probability closed forms on synthetic data, including recovering a known k, the clamp, and zero variance giving 0.
- The CFB sign conventions on a tiny fixture.
- The walk-forward per-season fit uses only earlier seasons.
- The 300 minimum.
- NFL and MLB are written as zeros.
- Ties and invalid prices are excluded.
- The JSON and the site module match.

**Site** (`cd site && npm test`):
- The helper maths: `shrinkProb`, `shrinkLine`, `fairAmerican`, the clip, and an unknown key giving k = 0.
- The sport page uses shrunk values for the CappingAlpha columns, Edge and Alpha.
- Past rows are unchanged.
- The no-market case shows "projection" and no edge.
- The game-page Read and Alpha use shrunk values.
- The raw-projection line is present.
- The cache key is bumped.

**Visual check:** NFL, CFB and a game page at 1280, 1440 and 1920 wide, with a hard reload.

## Out of scope

- Props (Part A).
- Any change to the models or the server-side predictions.
- The track record.
- A weekly refit job.
- An NFL or MLB fit before 300 graded games.
- Pick thresholds or removing products.

## Risks

- **Model-only game numbers vanish.** NFL and MLB game-level model numbers effectively become the market until there is evidence, so their edges will show as 0. This is intended.
- **Mostly no edge.** CFB edges will mostly be tiny, so the Read will usually say there is no meaningful model edge.
- **Fitted against the close.** k is fitted on the closing line, while the site compares against the current line. This is conservative: the model gets no credit for beating stale numbers it has not been shown to beat.
- **The served model changes.** If it changes (for example, CFB v3 or a new NFL game model), k must be refit before it is trusted. `model_version` in the JSON records which model k was fitted for.
