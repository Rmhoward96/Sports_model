# Shrink Game-Level Model Numbers on the Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every game-level model number the site shows as a fair price or an edge (the CappingAlpha ML / spread / total columns, Edge, the moneyline Alpha Score, sorting by Edge or Alpha, the game-page Read, Model Projection table and cover probabilities, the dashboard's model-vs-market gaps) equal `market + k × (model − market)`, with k fitted walk-forward on committed CFB history, k = 0 for NFL and MLB, and the raw projection kept visible and labelled (profit roadmap, step 2 Part B).

**Architecture:** A new pure module `src/sportsmodel/shrink/game_fit.py` holds the fit (closed forms, walk-forward stability report, the 300-game minimum, the two output documents). One IO script, `scripts/fit_game_shrink.py`, reads only committed assets, runs the existing leak-free CFB walk-forward once, and writes `assets/game_shrink_k.json` plus the generated site table `site/js/shrink-k.js`; a test pins that the two agree. On the site, one new classic script `site/js/shrink.js` (`shrinkK`, `shrinkProb`, `shrinkLine`, `fairAmerican`, `shrinkHalf`) is the only place a model number is pulled toward a market; the sport board (`board.js`), the game page (`game.js`) and the dashboard route every model-versus-market comparison through it. Graded rows, the Track Record, the historical panels, the props board, the +EV board and early tickets are untouched.

**Tech Stack:** Python 3.11+ with pandas, numpy and pytest (via `uv run`); vanilla JavaScript classic scripts tested with `node --test` (`cd site && npm test`). No new dependencies, no database, no network.

**Spec:** `docs/superpowers/specs/2026-10-14-shrink-site-design.md` (the authority; read it with this plan). Evidence: `docs/superpowers/reports/2026-10-13-edge-audit.md` section 3 (CFB model-minus-market coefficient +0.04, SE 0.16; NFL +0.00; CFB value-side moneylines lose 18.9%) and section 7 item 2. Sibling feature already merged: Part A, `docs/superpowers/plans/2026-10-13-shrink-props.md` (`src/sportsmodel/shrink/{formula,fit,store,...}.py`); this plan reuses its `formula.market_prob_over`, `formula.shrink_prob` and `fit.closed_form_k`.

**Baselines before this plan** (branch `feat/shrink-site`, HEAD `62efbb0`): `cd site && npm test` = 374 passed. `uv run pytest -q` = 3461 passed, 1 failed (about 4 minutes). The one failure, `tests/test_db_ev_prop_shrink_cols.py::test_the_migration_is_the_specs_idempotent_additive_alter`, is caused by the user's unrelated uncommitted edit to `db/migration_shrink_props.sql`. Do not touch that file; treat that single failure as the baseline, not as yours.

**Facts established while planning** (read-only runs against this repo; the plan depends on them):

- `raw_model_predictions(schedule_df, elo_cfg, blend_cfg, *, include_unscored=False)` in `src/sportsmodel/cfb/walkforward.py` returns a list of dicts with `season, week, home_team, away_team, model_margin, model_total, market_spread, market_total, actual_margin, actual_total` plus the pass-through `game_pk, start_date, neutral_site, conference_game, home_conf, away_conf, spread_open, total_open, ml_home, ml_away` and `elo_home, elo_away, srs_home, srs_away`. It takes about 10.5 seconds for all 9,504 merged games 2015-2026, and `build_gameline` over them about 1.4 seconds. The real fit therefore runs once, in a task step; every unit test uses tiny fixtures.
- Served CFB v2 (`scripts/generate_cfb.py`, `GAME_MODEL_VERSION = "cfb-ratings-v2"`) calls `build_gameline(model_margin, model_total, {"spread_line": None, "total_line": None}, week, gl_cfg)`. With an empty market, `shrink()` returns the model value, so the stored `pred_margin = model_margin − bias_margin`, `pred_total = model_total − bias_total`, `margin_dist = normal_to_margin_pmf(pred_margin, sigma_margin, offset)` and **`home_win_prob = prob_cover(margin_dist, 0.0)` = P(margin > 0)**. The fit calls the same `build_gameline` so its p is, by construction, what the site would have been given. The config is `assets/cfb/gameline.json` (sigma 16.665, offset 110, bias_margin −0.582, bias_total 0.0756).
- **Sign convention (two conventions coexist).** `assets/cfb/lines.parquet` `market_spread` is stored in expected-home-margin convention: `scripts/build_cfb_lines.py` stores `−spread`, so positive means the home team is favoured. Verified on the merged data: corr(`market_spread`, actual home margin) = +0.670 over the 7,163 graded 2016-2025 games with a line, and corr(`model_margin`, `market_spread`) = +0.90. **The site is the opposite.** `game_predictions.market_spread` (ESPN pickcenter, `cfb/espn.py::_market_from_odds`) is the HOME book line, negative when home is favoured; `site/js/pages/board.js` (`fav = mktSpread < 0 ? "home" : "away"`), `game.js` (`gmLean`: `edge = r05(h − a) + ls`) and `distributions.spread_pick_prob` all use that. The fit works in expected-home-margin space (x = pred_margin − market_spread); the site converts with `−market_spread` (or equivalently shrinks in home-line space, `shrinkLine` being linear).
- The site is made of **classic `<script>` tags that share one global scope**, not ES modules; tests concatenate the files (`site/tests/load.mjs`). So `shrink-k.js` is `const SHRINK_K = Object.freeze({...})` and `shrink.js` defines top-level functions, loaded after `js/metrics.js`. `smoke.test.mjs` pins the script order and the cache key.
- Existing code derives **no** fair moneyline from a spread. A moneyline edge needs the two-sided no-vig probability from `game_moneylines_current` (board) or the consensus / best prices (game page).
- Planning-time measurement of the same pipeline the script runs (informational, not asserted by any test): CFB spread n = 7,163 with clamped k = 0 (unclamped slope negative: the model is worse than the close); total n = 7,154, k ≈ 0.043; moneyline n = 3,561 (2021-2025), k ≈ 0.033. So after go-live the CFB site is nearly all market, consistent with the audit.

## Global Constraints

Every task's requirements include this section. Values are copied from the spec.

- **Formula:** `shrinkProb(p, m, k) = clip(m + k(p − m), 0.01, 0.99)`; `shrinkLine(model, market, k) = market + k(model − market)`; `fairAmerican(p)`.
- **k for points (spread, total):** `k_pts = clamp( Σxy / Σx², 0, 1 )`, x = model − close, y = actual − close; "If Σx² = 0, k = 0."
- **k for the moneyline:** `k_ml = clamp( Σ(y − m)(p − m) / Σ(p − m)², 0, 1 )` (the Part A closed form); p = the model's home win probability "using the same conversion the site and served model use"; m = the two-way no-vig probability from the closing `ml_home` / `ml_away`; y = 1 if the home team won; "Ties are excluded, and so are rows with an invalid American price (|price| < 100)."
- **CFB sample:** the v2 model as served (`GAME_MODEL_VERSION = "cfb-ratings-v2"`) via `raw_model_predictions`; seasons 2016-2025 graded (2015 is warm-up). Per-season stability: for each held-out season S from 2018 to 2025, fit on seasons before S and report k_S plus out-of-sample MAE (spread, total) or Brier (moneyline) for raw, shrunk and market. Served k: all graded seasons, 2016-2025 for spread and total and 2021-2025 for moneyline.
- **Minimum:** "A sport/market with fewer than 300 graded games of the current model gets k = 0."
- **NFL and MLB: k = 0 for spread, total and moneyline**, because the current NFL game model (the ML sim, `ML_GAME_LINES=on` since 2026-09-28) has about 6 weeks of graded games and the MLB game model has negligible graded history, both fewer than 300. The fit script writes these zeros explicitly, with the reason.
- **Outputs:** `assets/game_shrink_k.json` (per `(sport, market)`: `k`, `n`, the seasons used, per-season k and out-of-sample metrics, `model_version`, `fitted_at`) and `site/js/shrink-k.js` (the same k values, frozen); a test asserts they agree.
- **Unknown key → k = 0.** An unknown sport or market, or a missing or malformed table, is k = 0 and never the raw model.
- **Rule:** every place that compares a model number with a market number, or shows a model fair price beside a market, goes through the helper (Edge, Alpha, sorting, the Read, the dashboard).
- **No market line:** the model value is shown labelled "projection"; no edge or Alpha. Moneyline: with a market spread but no moneyline the column shows "—" (the existing code derives nothing from a spread).
- **Unchanged:** past-week result rows and recorded model picks, Track Record and the historical panels (Model vs Market, Model Projections bands), the props board, the +EV board, Top Alpha Edges, Market Pulse, early tickets. Raw projection cards (Projected Score / Spread / Total, Win Probability) stay raw.
- **House rules:** escaping through `ctxEsc` (`gmEsc` on the game page); desktop first; the cache key is bumped **exactly once, in the last site task**, and `smoke.test.mjs` pins it.
- **Fit script:** reads only committed assets; no database, no network; never print or source `.env`. Python tests are network-free and DB-free and run with `uv run pytest ...`; site tests run with `cd site && npm test`.
- **Out of scope:** props (Part A), any change to the models or server-side predictions, the track record, a weekly refit job, an NFL or MLB fit before 300 graded games, pick thresholds, removing products.
- **Repository hygiene:** stay on branch `feat/shrink-site`; never touch `db/migration_shrink_props.sql`; every commit message ends with the Co-Authored-By line the executing harness provides. In the session that wrote this plan it is `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; every commit command below passes it as a second `-m`. If your harness provides a different line, use yours. Project convention: implementers and reviewers run on Sonnet (pass `model: "sonnet"` explicitly).

## Review Focus

The inputs and failure modes the spec implies but does not spell out, most likely to bite first. Each has a pinning test in the task that owns the code.

1. **Spread sign: the site's `market_spread` is the home BOOK line, the fit's is the expected home margin.** Getting it backwards turns "shrink toward the market" into "double the gap", and only for one side. Behavior expected: a home favourite and an away favourite are both pulled toward the market, never past it. Task 1: `test_spread_x_and_y_are_in_expected_home_margin_space_with_positive_meaning_home_favoured`, `test_the_committed_cfb_lines_are_in_home_margin_convention`; Task 3: `boardLines: an away favourite is shrunk in the favourite's own direction`; Task 4: `gmOdds (CFB k = 0.5)`.
2. **A missing, malformed or hand-edited k table** (`SHRINK_K` not loaded, `NaN`, 1.5, −0.1, `"0.5"`, `null`, an unknown sport such as `nba`, a prototype key such as `constructor`). A reasonable person expects the page to show the market, never the raw model. Task 2: `shrinkK: ...` and `shrinkK: a malformed or missing table is k = 0`; Task 1: `test_the_committed_assets_agree_and_nfl_mlb_are_explicit_zeros`.
3. **Quarter-point and odd market lines at k = 0** (CFB totals such as 51.75). The old code rounded the model to the half point, so a model equal to the market would have shown a phantom ±0.25 edge. Expected: edge exactly 0 and the model column equal to the market number. Task 3: `boardLines: a quarter-point total at k = 0 gives edge exactly 0`; Task 4: `gmLean uses the shrunk model`; Task 4 dashboard test.
4. **Partial markets:** no spread or total, no moneyline, one side's moneyline price only, no market at all. Expected: projection labelled, no edge, no Alpha Score, "—" for a moneyline that cannot be shrunk, no `NaN` or `undefined` in the HTML. Task 3: `boardLines: no market line ...`, `boardLines: a spread but no moneyline ...`, `boardUpcomingTable: a game with no market line ...`; Task 4: `gmOdds: without a posted spread or total ...`.
5. **History must not move:** graded rows, their recorded model picks and the Model Projections bands must be identical whatever k is, and team names with markup must stay escaped in the new "Raw model projection" line. Task 3: `boardLines: a graded game keeps its recorded model numbers whatever k is`; Task 4: `the Read card shows the raw model projection, labelled, and escapes team names`.

Also pinned: the fit never uses a later season to score an earlier one and the 299/300 line is exact (Task 1: `test_each_season_is_scored_with_a_k_fitted_on_earlier_seasons_only`, `test_a_market_with_fewer_than_300_graded_games_gets_k_zero`); ties, invalid prices, missing numbers, 2015 warm-up rows and seasons after `--last-season` are excluded, never made up; the fit script touches no database, network or env file (Task 1: `test_the_script_touches_no_database_network_or_env_file`).

## Decisions on spec ambiguities (resolved here; the plan follows these)

| # | Spec text | Resolution | Why |
|---|---|---|---|
| 1 | "`site/js/shrink-k.js`: the same k values as a frozen ES module" | A classic script: `const SHRINK_K = Object.freeze({...})`, loaded after `js/metrics.js` on every page | The site has no module loader; every file shares one global scope (`tests/load.mjs`) |
| 2 | "The sign conventions must match those verified in step 3: CFB `market_spread` is positive when the home team is favoured" | True for `lines.parquet` (the fit). The site's `game_predictions.market_spread` is the opposite (home book line). The fit uses x = pred_margin − market_spread; the site shrinks `−market_spread` (margin space) or the home line itself (line space) | See Facts. Both are pinned by tests |
| 3 | "Model numbers come from `raw_model_predictions` for v2 as served" | The p, pred_margin and pred_total are produced by `build_gameline` with an empty market, exactly as `generate_cfb.py` serves them | Same bias correction and same `prob_cover` normal-pmf win probability as the served rows |
| 4 | Same sentence | **Known limit, recorded in the go-live doc:** `raw_model_predictions` is the plain walk-forward. Served v2 also blends a leak-free preseason prior into each team's Elo (`build_game_rows` with `r_pre`), which the walk-forward omits, so k is fitted on a slightly weaker model than the one served in the early weeks of a season. A weaker model gives a smaller k, which is the conservative direction. Rebuilding the prior-blended walk-forward would need the prior weights, which were themselves fitted on 2016-2022 and so would be in-sample | The spec names `raw_model_predictions`; this is an accepted approximation, not a silent one |
| 5 | "the moneyline column shows the market-derived value if the existing code already derives one. Otherwise '—'. The plan decides from the existing code." | Existing code derives none. Rule: a two-sided no-vig probability exists → the shrunk fair ML; no such probability but the game has any market number (spread, total, or one ML price) → "—"; the game has no market number at all → the raw fair ML labelled "proj" | Literal reading of the spec's two clauses; one-sided prices no longer produce a vigged ML edge |
| 6 | Game page: "the Alpha card ... use shrunk numbers" | `gmAlphaCard` shows the best +EV opportunity (`ev_picks`, Pinnacle no-vig `true_prob`), which holds no CappingAlpha game-level model number, and the spec keeps the +EV board unchanged. It is left alone. The only model-based Alpha Score is `boardAlpha`'s moneyline fallback on the sport page, which is shrunk (Task 3) | Shrinking something that is not model output would be wrong; flagged to the user in the go-live doc |
| 7 | "the odds and candidate edges use shrunk numbers" for spread and total on the game page | Shrink the expected margin / total in points with `shrinkLine` (toward the posted line), then evaluate the model's own distributions shifted by the same amount: P(M + δ > t) = P(M > t − δ) is `gmSplit(dist, t − δ)`. The moneyline uses `shrinkProb` toward the no-vig probability of the consensus (else best) prices of both sides | The fit's k for spread and total is in points, so shrinking in points is the consistent use; the pmf shift needs no new distribution code |
| 8 | "Edge is shrunk minus market" with the existing half-point rounding | `shrinkHalf(model, market)`: round the shrunk value to the half point, except when it equals the market (then return the market number untouched). At k = 1 this is exactly the old `r05` display, at k = 0 it is the market itself | Gives edge exactly 0 at k = 0 even for a 51.75 total, and leaves every k = 1 behavior byte-identical |
| 9 | "Past-week result rows keep the recorded model picks" | `boardLines` shrinks only when the game has no graded row (`!g.acc`); the graded branch is the old code | Board-right's Model Projections bands call `boardLines` with graded rows; they must not move |
| 10 | How the tests keep their raw expectations | New fixture tables `site/tests/fixtures/shrink-k-identity.js` (k = 1 everywhere), `shrink-k-mixed.js` (NFL 0, CFB 0.5) and `shrink-k-bad.js`; existing suites load the identity fixture in place of `js/shrink-k.js`, new tests load the mixed one. `tests/` is excluded from deployment by `sync_site.sh` | A test seam with no production hook |
| 11 | "Hero odds boxes lean highlight" | `gmLean`'s spread and total leans compare the shrunk model with the market, so they go through the helper too; the moneyline lean is the model's favourite, not a comparison, and stays raw | The spec's rule covers every model-versus-market comparison |
| 12 | Refit | Manual: `uv run python scripts/fit_game_shrink.py --last-season <year>` at season's end. The script fits CFB only; NFL and MLB stay explicit zeros until a follow-up adds their fit path from graded game predictions | Matches "no scheduled job" and "no NFL or MLB fit before 300 graded games" |

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/sportsmodel/shrink/game_fit.py` | Create | Served-number conversion, point and probability legs, k, walk-forward report, document, JS rendering |
| `scripts/fit_game_shrink.py` | Create | The IO wrapper: walk-forward over committed assets, print the stability table, write the two files |
| `assets/game_shrink_k.json` | Create (by running the script) | The committed fit |
| `site/js/shrink-k.js` | Create (by running the script) | Generated frozen table `SHRINK_K` |
| `site/js/shrink.js` | Create | `shrinkK`, `shrinkProb`, `shrinkLine`, `fairAmerican`, `shrinkHalf` |
| `site/js/pages/board.js` | Modify | `boardShrink`, `boardLines` (upcoming branch), `boardAlpha`, projection labels |
| `site/js/pages/game.js` | Modify | `gmShrunkLines`, `gmLean`, `gmOdds`, `modelProjectionRows`, `gmRawLine`, `gmReadCard` |
| `site/js/pages/dashboard.js` | Modify | `dashModelVsMarket` |
| `site/css/theme.css` | Modify | One rule for the raw-projection line |
| `site/*.html` (10 pages) | Modify | Two script tags (Task 2), then the cache key (Task 5) |
| `site/tests/fixtures/shrink-k-{identity,mixed,bad}.js` | Create | Test tables |
| `site/tests/shrink.test.mjs` | Create | Helper maths, table validation, JSON agreement |
| `site/tests/{board,game,dashboard,smoke}.test.mjs`, other suites' `FILES` lists | Modify | New and updated tests |
| `tests/shrink/test_game_fit.py`, `tests/scripts/test_fit_game_shrink.py` | Create | Python tests |
| `docs/superpowers/reports/2026-10-14-shrink-site-go-live.md` | Create | The user's go-live checklist |

Not touched: the models, `generate_cfb.py`, `game_predictions`, Part A files, `track.js`, `ev.js`, `board-left.js`, `board-right.js` (they call `boardLines` only for graded rows), `scripts/sync_site.sh`, every workflow.

---

### Task 1: The game-level fit, the script, and the committed k assets

**Files:**
- Create: `src/sportsmodel/shrink/game_fit.py`
- Create: `scripts/fit_game_shrink.py`
- Create: `tests/shrink/test_game_fit.py`, `tests/scripts/test_fit_game_shrink.py`
- Create (by running the script in Step 8): `assets/game_shrink_k.json`, `site/js/shrink-k.js`

**Interfaces:**
- Consumes (existing): `sportsmodel.nfl.gameline.{GameLineConfig, build_gameline}`, `sportsmodel.nfl.shrink.ShrinkParams`, `sportsmodel.shrink.fit.closed_form_k(legs)` (legs with `y`, `m`, `p`), `sportsmodel.shrink.formula.{market_prob_over, shrink_prob}`, `sportsmodel.cfb.walkforward.raw_model_predictions`, `scripts/backtest_cfb_gameline.py::load_merged_schedule`.
- Produces (later tasks and the go-live doc rely on these exact names):
  - `game_fit.MIN_GAMES = 300`, `MARKETS = ("moneyline", "spread", "total")`, `FIT_SPORT = "cfb"`, `CFB_MODEL_VERSION = "cfb-ratings-v2"`, `FIRST_GRADED_SEASON = 2016`, `FIRST_TEST_SEASON = 2018`, `LAST_SEASON = 2025`
  - `game_fit.gameline_config(j: dict) -> GameLineConfig`
  - `game_fit.served_rows(raw: list[dict], gl_cfg) -> list[dict]` (keys `season, week, pred_margin, pred_total, p_home, market_spread, market_total, ml_home, ml_away, actual_margin, actual_total`)
  - `game_fit.point_legs(rows, market) -> list[dict]` (`market` is `"spread"` or `"total"`; leg keys `season, model, close, actual, x, y`), `game_fit.moneyline_legs(rows) -> list[dict]` (leg keys `season, p, m, y`)
  - `game_fit.k_points(legs) -> float`, `game_fit.fit_k(legs, market) -> float` (applies the 300 minimum)
  - `game_fit.season_report(legs, market, first_test=2018) -> list[dict]` (`season, k, n_train, n_test, metric, raw, shrunk, market`)
  - `game_fit.build_document(rows, fitted_at, *, last_season=2025, model_version="cfb-ratings-v2") -> dict`
  - `game_fit.render_js(doc) -> str`, `game_fit.parse_js(text) -> dict`, `game_fit.format_tables(doc) -> str`
  - `scripts/fit_game_shrink.py::main(argv=None, *, raw_fn=None) -> int` (`raw_fn(last_season) -> list[dict]` is the injection point for tests)
  - `assets/game_shrink_k.json`: `{"schema": 1, "fitted_at", "min_games": 300, "last_season", "model_version": {"cfb": "cfb-ratings-v2"}, "markets": {sport: {market: {"k", "n", "k_raw", "seasons", "per_season", ["reason"]}}}}` for sports `cfb`, `mlb`, `nfl`
  - `site/js/shrink-k.js`: `const SHRINK_K = Object.freeze({"cfb": {"moneyline": ..., "spread": ..., "total": ...}, "mlb": {...}, "nfl": {...}});`

- [ ] **Step 1: Write the failing tests for the fit module**

Create `tests/shrink/test_game_fit.py`:

````python
"""Game-level shrink fit (profit step 2 Part B): closed forms, sign conventions, walk-forward, the 300 minimum, the documents.
No database, no network; two committed assets are read (lines/schedules parquet, gameline.json)."""
import json
import random
from pathlib import Path

import pytest

from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
from sportsmodel.shrink import game_fit
from sportsmodel.shrink.fit import closed_form_k

ROOT = Path(__file__).resolve().parents[2]
NO_BIAS = GameLineConfig()               # bias 0 and no market: the served number IS the model's number


def american(p):
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def raw_row(season=2018, week=3, **kw):
    base = {"season": season, "week": week, "home_team": "H", "away_team": "A", "model_margin": 10.0, "model_total": 52.0,
            "market_spread": 7.0, "market_total": 50.0, "actual_margin": 8.5, "actual_total": 51.0, "ml_home": None,
            "ml_away": None}
    return base | kw


def spread_rows(k_true, seasons=(2016, 2017, 2018, 2019), n=400, seed=3):
    """Walk-forward-shaped rows whose actual margin is close + k_true * (model - close): the fit must recover k_true."""
    rng = random.Random(seed)
    rows = []
    for s in seasons:
        for i in range(n):
            close = rng.uniform(-20, 20)
            model = close + rng.uniform(-8, 8)
            rows.append(raw_row(s, 1 + i % 14, model_margin=model, market_spread=close,
                                actual_margin=close + k_true * (model - close)))
    return rows


def served(raw):
    return game_fit.served_rows(raw, NO_BIAS)


def pleg(x, y, season=2018):
    """A point leg with close = 0 so model = x and actual = y."""
    return {"season": season, "model": float(x), "close": 0.0, "actual": float(y), "x": float(x), "y": float(y)}


def ml_rows(k_true, n=40_000, seed=11):
    """Served-shaped moneyline rows: the home team wins with probability m + k_true * (p - m)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        m = rng.uniform(0.35, 0.65)
        p = min(max(m + rng.uniform(-0.15, 0.15), 0.02), 0.98)
        home_won = rng.random() < m + k_true * (p - m)
        rows.append({"season": 2022, "p_home": p, "actual_margin": 3.0 if home_won else -3.0,
                     "ml_home": american(m), "ml_away": american(1 - m)})
    return rows


def test_the_constants_are_the_specs():
    assert game_fit.MIN_GAMES == 300 and game_fit.FIRST_GRADED_SEASON == 2016 and game_fit.FIRST_TEST_SEASON == 2018
    assert game_fit.MARKETS == ("moneyline", "spread", "total") and game_fit.CFB_MODEL_VERSION == "cfb-ratings-v2"
    assert game_fit.LAST_SEASON == 2025 and game_fit.FIT_SPORT == "cfb"


# ---- points: spread and total ----------------------------------------------------------------------------------

def test_point_k_recovers_a_known_k_exactly():
    legs = game_fit.point_legs(served(spread_rows(0.4)), "spread")
    assert len(legs) == 1600
    assert game_fit.k_points(legs) == pytest.approx(0.4)


def test_point_k_clamps_and_zero_variance_gives_zero():
    assert game_fit.k_points([pleg(1, -1), pleg(-2, 2)]) == 0.0            # the model points the wrong way
    assert game_fit.k_points([pleg(1, 3), pleg(-2, -6)]) == 1.0            # the model under-states its own disagreement
    assert game_fit.k_points([pleg(0, 5), pleg(0, -3)]) == 0.0             # model == close everywhere: nothing to learn
    assert game_fit.k_points([]) == 0.0


def test_spread_x_and_y_are_in_expected_home_margin_space_with_positive_meaning_home_favoured():
    # the closing line is +7 (home favoured by 7: lines.parquet stores minus the book's home line); the model says +10;
    # the home team won by 8.5
    [leg] = game_fit.point_legs(served([raw_row(model_margin=10.0, market_spread=7.0, actual_margin=8.5)]), "spread")
    assert (leg["x"], leg["y"]) == (3.0, 1.5)
    [away] = game_fit.point_legs(served([raw_row(model_margin=-10.0, market_spread=-7.0, actual_margin=-8.5)]), "spread")
    assert (away["x"], away["y"]) == (-3.0, -1.5)                           # the mirror image gives the same k


def test_total_legs_use_the_served_total_the_closing_total_and_the_actual_total():
    [leg] = game_fit.point_legs(served([raw_row(model_total=52.0, market_total=50.0, actual_total=51.0)]), "total")
    assert (leg["x"], leg["y"]) == (2.0, 1.0)
    assert game_fit.k_points([leg]) == 0.5


@pytest.mark.parametrize("bad", [dict(market_spread=None), dict(market_spread=float("nan")), dict(actual_margin=None),
                                 dict(model_margin=float("nan"))])
def test_rows_missing_a_number_are_not_spread_legs(bad):
    assert game_fit.point_legs(served([raw_row(**bad)]), "spread") == []


# ---- moneyline --------------------------------------------------------------------------------------------------

def test_moneyline_k_recovers_a_known_k_on_simulated_outcomes():
    legs = game_fit.moneyline_legs(ml_rows(0.30))
    assert len(legs) == 40_000
    assert game_fit.fit_k(legs, "moneyline") == pytest.approx(0.30, abs=0.08)
    assert game_fit.fit_k(legs, "moneyline") == closed_form_k(legs)         # the Part A closed form, not a copy of it


def test_moneyline_legs_exclude_ties_and_unusable_prices():
    ok = {"season": 2022, "p_home": 0.6, "actual_margin": 3.0, "ml_home": -150, "ml_away": 130}
    assert len(game_fit.moneyline_legs([ok])) == 1
    for bad in (dict(actual_margin=0.0), dict(ml_home=None), dict(ml_away=None), dict(ml_home=0), dict(ml_home=-99),
                dict(ml_away=50), dict(ml_home=float("nan")), dict(p_home=None), dict(actual_margin=None)):
        assert game_fit.moneyline_legs([ok | bad]) == [], bad
    [leg] = game_fit.moneyline_legs([ok | {"actual_margin": -1.0}])
    assert leg["y"] == 0 and leg["p"] == 0.6 and 0.5 < leg["m"] < 0.6        # the no-vig home probability of -150 / +130


# ---- the 300-game minimum ---------------------------------------------------------------------------------------

def test_a_market_with_fewer_than_300_graded_games_gets_k_zero():
    strong = [pleg(1.0, 1.0) for _ in range(299)]
    assert game_fit.fit_k(strong, "spread") == 0.0
    assert game_fit.fit_k(strong + [pleg(1.0, 1.0)], "spread") == 1.0
    perfect = {"season": 2022, "p": 0.7, "m": 0.5, "y": 1}
    assert game_fit.fit_k([perfect] * 299, "moneyline") == 0.0
    assert game_fit.fit_k([perfect] * 300, "moneyline") == 1.0


# ---- walk-forward ------------------------------------------------------------------------------------------------

def test_each_season_is_scored_with_a_k_fitted_on_earlier_seasons_only():
    rows = spread_rows(0.0, seasons=(2016, 2017, 2018)) + spread_rows(1.0, seasons=(2019,), seed=4)
    rep = game_fit.season_report(game_fit.point_legs(served(rows), "spread"), "spread")
    assert [r["season"] for r in rep] == [2018, 2019]                       # 2016 and 2017 are history, not test seasons
    by = {r["season"]: r for r in rep}
    assert by[2018]["k"] == 0.0 and by[2018]["n_train"] == 800 and by[2018]["n_test"] == 400
    assert by[2019]["k"] == 0.0 and by[2019]["n_train"] == 1200             # 2019 itself (k_true = 1) never trains 2019
    more = rows + spread_rows(1.0, seasons=(2020,), seed=5)
    later = game_fit.season_report(game_fit.point_legs(served(more), "spread"), "spread")
    assert [(r["season"], r["k"]) for r in later[:2]] == [(r["season"], r["k"]) for r in rep]    # a later season changes nothing earlier
    assert later[2]["season"] == 2020 and 0.1 < later[2]["k"] < 0.45       # 2020 sees 2016-2019


def test_out_of_sample_metrics_for_raw_shrunk_and_market():
    legs = [pleg(2.0, 1.0, season=2018) for _ in range(400)] + [pleg(2.0, 1.0, season=2019) for _ in range(10)]
    r = {x["season"]: x for x in game_fit.season_report(legs, "spread")}[2019]
    assert (r["metric"], r["n_test"], r["n_train"]) == ("mae", 10, 400) and r["k"] == pytest.approx(0.5)
    assert r["raw"] == pytest.approx(1.0) and r["shrunk"] == pytest.approx(0.0) and r["market"] == pytest.approx(1.0)
    ml = [{"season": 2021, "p": 0.8, "m": 0.6, "y": 1}] * 400 + [{"season": 2022, "p": 0.8, "m": 0.6, "y": 1}] * 10
    q = {x["season"]: x for x in game_fit.season_report(ml, "moneyline")}[2022]
    assert q["metric"] == "brier" and q["k"] == 1.0
    assert q["raw"] == pytest.approx(0.04) and q["shrunk"] == pytest.approx(0.04) and q["market"] == pytest.approx(0.16)


# ---- the served conversion --------------------------------------------------------------------------------------

def test_served_rows_use_the_served_conversion_not_the_raw_model():
    cfg = GameLineConfig(sigma_margin=16.0, offset=110, total_max=150, bias_margin=-0.5, bias_total=0.25)
    [row] = game_fit.served_rows([raw_row(model_margin=6.0, model_total=55.0)], cfg)
    want = build_gameline(6.0, 55.0, {"spread_line": None, "total_line": None}, 3, cfg)
    assert row["pred_margin"] == want["pred_margin"] == 6.5 and row["pred_total"] == want["pred_total"] == 54.75
    assert row["p_home"] == want["home_win_prob"] and 0.5 < row["p_home"] < 1


def test_served_rows_carry_the_closing_numbers_and_skip_rows_with_no_model_number():
    rows = game_fit.served_rows([raw_row(ml_home=-150, ml_away=130), raw_row(model_margin=None),
                                 raw_row(model_total=float("nan"))], NO_BIAS)
    assert len(rows) == 1
    r = rows[0]
    assert (r["season"], r["market_spread"], r["market_total"], r["actual_margin"], r["actual_total"], r["ml_home"],
            r["ml_away"]) == (2018, 7.0, 50.0, 8.5, 51.0, -150, 130)


def test_gameline_config_reads_the_committed_cfb_file():
    j = json.loads((ROOT / "assets" / "cfb" / "gameline.json").read_text())
    cfg = game_fit.gameline_config(j)
    assert (cfg.sigma_margin, cfg.offset, cfg.bias_margin, cfg.bias_total) == (j["sigma_margin"], j["offset"],
                                                                               j["bias_margin"], j["bias_total"])


def test_the_committed_cfb_lines_are_in_home_margin_convention():
    import pandas as pd
    lines = pd.read_parquet(ROOT / "assets" / "cfb" / "lines.parquet")
    sched = pd.read_parquet(ROOT / "assets" / "cfb" / "schedules.parquet")
    m = sched.merge(lines, on=["season", "week", "home_team", "away_team"]).dropna(subset=["market_spread"])
    m = m[(m.season >= 2016) & (m.season <= 2025)]
    assert len(m) > 5000
    corr = (m.home_score - m.away_score).corr(m.market_spread)
    assert corr > 0.5, "a positive market_spread must mean the home team is favoured (the expected home margin)"


# ---- the documents ----------------------------------------------------------------------------------------------

def doc_for(rows, **kw):
    return game_fit.build_document(rows, "2026-10-14T12:00:00+00:00", **kw)


def test_the_document_has_fitted_cfb_entries_and_explicit_zero_nfl_and_mlb():
    rows = served(spread_rows(0.4, seasons=(2015, 2016, 2017, 2018, 2019, 2026)))
    doc = doc_for(rows, last_season=2025)
    assert doc["schema"] == 1 and doc["min_games"] == 300 and doc["last_season"] == 2025
    assert doc["model_version"] == {"cfb": "cfb-ratings-v2"} and doc["fitted_at"] == "2026-10-14T12:00:00+00:00"
    sp = doc["markets"]["cfb"]["spread"]
    assert sp["k"] == pytest.approx(0.4) and sp["n"] == 1600 and sp["seasons"] == [2016, 2019]   # 2015 is warm-up, 2026 is past last_season
    assert [r["season"] for r in sp["per_season"]] == [2018, 2019]
    for sport in ("nfl", "mlb"):
        for m in ("moneyline", "spread", "total"):
            e = doc["markets"][sport][m]
            assert e["k"] == 0.0 and e["n"] == 0 and "fewer than 300 graded games" in e["reason"]
    ml = doc["markets"]["cfb"]["moneyline"]
    assert ml["k"] == 0.0 and ml["n"] == 0 and "fewer than 300" in ml["reason"]       # these rows carry no prices
    assert json.loads(json.dumps(doc)) == doc                                          # plain JSON types only


def test_the_site_module_is_a_frozen_table_of_exactly_the_documents_k_values():
    doc = doc_for(served(spread_rows(0.4)))
    js = game_fit.render_js(doc)
    assert js.startswith("/* GENERATED") and "const SHRINK_K = Object.freeze(" in js and js.endswith(");\n")
    table = game_fit.parse_js(js)
    assert set(table) == {"nfl", "cfb", "mlb"}
    for sport in table:
        assert set(table[sport]) == {"moneyline", "spread", "total"}
        for m, k in table[sport].items():
            assert k == doc["markets"][sport][m]["k"]
    assert table["nfl"] == {"moneyline": 0.0, "spread": 0.0, "total": 0.0} == table["mlb"]
    assert game_fit.render_js(doc) == js                                               # deterministic


def test_the_stability_table_lists_every_held_out_season():
    text = game_fit.format_tables(doc_for(served(spread_rows(0.4))))
    assert "cfb spread" in text and "2018" in text and "2019" in text and "nfl spread" in text
````

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/shrink/test_game_fit.py`
Expected: collection error, `ImportError: cannot import name 'game_fit' from 'sportsmodel.shrink'`.

- [ ] **Step 3: Write the fit module**

Create `src/sportsmodel/shrink/game_fit.py`:

````python
"""Fitting k for the game-level shrink (profit step 2 Part B). PURE -- no IO.

Spec: docs/superpowers/specs/2026-10-14-shrink-site-design.md section 1.

    spread, total (points):  k = clamp( sum(x*y) / sum(x*x), 0, 1 )      x = model - close, y = actual - close
                             The spread lives in expected-HOME-MARGIN space (positive = home favoured), which is how
                             assets/cfb/lines.parquet stores market_spread. (The site's live market_spread is the home BOOK
                             line, the opposite sign; the page code converts.)
    moneyline (probability): k = clamp( sum(y - m)(p - m) / sum(p - m)^2, 0, 1 )   -- the Part A closed form, fit.closed_form_k
                             p = the served home win probability, m = the two-way no-vig closing probability of the home
                             team, y = 1 when the home team won

The model numbers are the ones served: build_gameline with an empty market (what scripts/generate_cfb.py writes), so
pred_margin = model_margin - bias_margin and home_win_prob = P(margin > 0) of the normal margin pmf.
A market with fewer than 300 graded games gets k = 0. NFL and MLB are written as explicit zeros with the reason.
"""
from __future__ import annotations

import json
import math
import re

from ..nfl.gameline import GameLineConfig, build_gameline
from ..nfl.shrink import ShrinkParams
from .fit import closed_form_k
from .formula import market_prob_over, shrink_prob

SCHEMA = 1
MIN_GAMES = 300
MARKETS = ("moneyline", "spread", "total")
SPORTS = ("cfb", "mlb", "nfl")
FIT_SPORT = "cfb"
CFB_MODEL_VERSION = "cfb-ratings-v2"
FIRST_GRADED_SEASON = 2016          # 2015 is warm-up for the walk-forward
FIRST_TEST_SEASON = 2018            # first held-out season of the stability report
LAST_SEASON = 2025
ZERO_REASON = {
    "nfl": f"k = 0: the current NFL game model (the ML sim, ML_GAME_LINES=on since 2026-09-28) has fewer than {MIN_GAMES} graded games",
    "mlb": f"k = 0: the MLB game model has fewer than {MIN_GAMES} graded games",
}
_NO_MARKET = {"spread_line": None, "total_line": None}

JS_HEADER = """/* GENERATED by scripts/fit_game_shrink.py from assets/game_shrink_k.json -- do not edit by hand.
   Shrink factors for the game-level model numbers (profit step 2 Part B): shown = market + k * (model - market).
   k = 0 shows the market, k = 1 the raw model. An unknown sport or market is k = 0 (js/shrink.js, shrinkK). */
"""


def _num(x) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def gameline_config(j: dict) -> GameLineConfig:
    """The served GameLineConfig from the parsed assets/cfb/gameline.json (same fields scripts/generate_cfb.py reads)."""
    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"], offset=j["offset"],
                          total_max=j["total_max"], w_margin=ShrinkParams(**j["w_margin"]),
                          w_total=ShrinkParams(**j["w_total"]), bias_margin=j.get("bias_margin", 0.0),
                          bias_total=j.get("bias_total", 0.0))


def served_rows(raw: list[dict], gl_cfg: GameLineConfig) -> list[dict]:
    """Walk-forward rows (cfb.walkforward.raw_model_predictions) -> the numbers the site would have shown: the served
    pred_margin, pred_total and home win probability, plus the closing numbers and the actuals. Rows with no usable model
    margin or total are skipped."""
    out = []
    for r in raw:
        mm, mt = _num(r.get("model_margin")), _num(r.get("model_total"))
        if mm is None or mt is None:
            continue
        g = build_gameline(mm, mt, _NO_MARKET, int(r["week"]), gl_cfg)
        out.append({
            "season": int(r["season"]), "week": int(r["week"]),
            "pred_margin": g["pred_margin"], "pred_total": g["pred_total"], "p_home": _num(g["home_win_prob"]),
            "market_spread": _num(r.get("market_spread")), "market_total": _num(r.get("market_total")),
            "ml_home": r.get("ml_home"), "ml_away": r.get("ml_away"),
            "actual_margin": _num(r.get("actual_margin")), "actual_total": _num(r.get("actual_total")),
        })
    return out


def point_legs(rows: list[dict], market: str) -> list[dict]:
    """Spread or total legs: x = model - close, y = actual - close. A row missing any of the three numbers is dropped."""
    legs = []
    for r in rows:
        if market == "spread":
            model, close, actual = r["pred_margin"], r["market_spread"], r["actual_margin"]
        else:
            model, close, actual = r["pred_total"], r["market_total"], r["actual_total"]
        if model is None or close is None or actual is None:
            continue
        legs.append({"season": r["season"], "model": model, "close": close, "actual": actual,
                     "x": model - close, "y": actual - close})
    return legs


def moneyline_legs(rows: list[dict]) -> list[dict]:
    """Moneyline legs {season, p, m, y}. Excluded: ties, a missing probability or result, and any row whose closing
    home or away price is missing or not a valid American price (|price| < 100)."""
    legs = []
    for r in rows:
        p, am = _num(r.get("p_home")), _num(r.get("actual_margin"))
        m = market_prob_over(r.get("ml_home"), r.get("ml_away"))
        if p is None or am is None or m is None or am == 0:
            continue
        legs.append({"season": r["season"], "p": p, "m": m, "y": 1 if am > 0 else 0})
    return legs


def legs_for(rows: list[dict], market: str) -> list[dict]:
    return moneyline_legs(rows) if market == "moneyline" else point_legs(rows, market)


def _slope(legs: list[dict], market: str) -> float | None:
    """The unclamped slope (num / den); None when there is no variance. Kept in the document as k_raw for audit."""
    if market == "moneyline":
        num = sum((l["y"] - l["m"]) * (l["p"] - l["m"]) for l in legs)
        den = sum((l["p"] - l["m"]) ** 2 for l in legs)
    else:
        num = sum(l["x"] * l["y"] for l in legs)
        den = sum(l["x"] ** 2 for l in legs)
    return num / den if den > 0 else None


def k_points(legs: list[dict]) -> float:
    """clamp( sum(x*y) / sum(x*x), 0, 1 ); 0.0 when there are no legs or no variance in x."""
    s = _slope(legs, "spread")
    return 0.0 if s is None else min(1.0, max(0.0, s))


def fit_k(legs: list[dict], market: str) -> float:
    """k for one market: 0.0 below the 300-game minimum, else the clamped closed form (Part A's for the moneyline)."""
    if len(legs) < MIN_GAMES:
        return 0.0
    return closed_form_k(legs) if market == "moneyline" else k_points(legs)


def oos_metrics(test: list[dict], market: str, k: float) -> dict:
    """Out-of-sample error of the raw model, the shrunk model and the market on `test` legs: MAE in points for spread
    and total, Brier score for the moneyline."""
    if market == "moneyline":
        return {"metric": "brier",
                "raw": _mean([(l["p"] - l["y"]) ** 2 for l in test]),
                "shrunk": _mean([(shrink_prob(l["p"], l["m"], k) - l["y"]) ** 2 for l in test]),
                "market": _mean([(l["m"] - l["y"]) ** 2 for l in test])}
    return {"metric": "mae",
            "raw": _mean([abs(l["actual"] - l["model"]) for l in test]),
            "shrunk": _mean([abs(l["actual"] - (l["close"] + k * (l["model"] - l["close"]))) for l in test]),
            "market": _mean([abs(l["actual"] - l["close"]) for l in test])}


def season_report(legs: list[dict], market: str, first_test: int = FIRST_TEST_SEASON) -> list[dict]:
    """Per held-out season S >= first_test: k fitted on the legs of seasons BEFORE S only, and the out-of-sample metrics
    of S. A season with no legs is omitted."""
    out = []
    for s in sorted({l["season"] for l in legs}):
        if s < first_test:
            continue
        train = [l for l in legs if l["season"] < s]
        test = [l for l in legs if l["season"] == s]
        k = fit_k(train, market)
        out.append({"season": s, "k": k, "n_train": len(train), "n_test": len(test), **oos_metrics(test, market, k)})
    return out


def fit_market(rows: list[dict], market: str, *, first_season: int = FIRST_GRADED_SEASON,
               last_season: int = LAST_SEASON, first_test: int = FIRST_TEST_SEASON) -> dict:
    """One market's served entry: k on every graded season in [first_season, last_season], n, the unclamped slope, the
    seasons used, the per-season stability report, and a reason when the 300-game minimum forced k = 0."""
    legs = [l for l in legs_for(rows, market) if first_season <= l["season"] <= last_season]
    seasons = sorted({l["season"] for l in legs})
    entry = {"k": fit_k(legs, market), "n": len(legs), "k_raw": _slope(legs, market),
             "seasons": [seasons[0], seasons[-1]] if seasons else [],
             "per_season": season_report(legs, market, first_test)}
    if len(legs) < MIN_GAMES:
        entry["reason"] = f"k = 0: fewer than {MIN_GAMES} graded games"
    return entry


def build_document(rows: list[dict], fitted_at: str, *, last_season: int = LAST_SEASON,
                   model_version: str = CFB_MODEL_VERSION) -> dict:
    """The contents of assets/game_shrink_k.json: CFB fitted, NFL and MLB explicit zeros with the reason."""
    cfb = {m: fit_market(rows, m, last_season=last_season) for m in MARKETS}
    zero = lambda reason: {m: {"k": 0.0, "n": 0, "k_raw": None, "seasons": [], "per_season": [], "reason": reason}  # noqa: E731
                           for m in MARKETS}
    return {"schema": SCHEMA, "fitted_at": fitted_at, "min_games": MIN_GAMES, "last_season": last_season,
            "model_version": {FIT_SPORT: model_version},
            "markets": {FIT_SPORT: cfb, "nfl": zero(ZERO_REASON["nfl"]), "mlb": zero(ZERO_REASON["mlb"])}}


def render_js(doc: dict) -> str:
    """The generated classic script site/js/shrink-k.js: a frozen SHRINK_K[sport][market] table of the document's k values."""
    ks = {sport: {m: doc["markets"][sport][m]["k"] for m in MARKETS} for sport in SPORTS}
    return JS_HEADER + "const SHRINK_K = Object.freeze(" + json.dumps(ks, indent=2, sort_keys=True) + ");\n"


def parse_js(text: str) -> dict:
    """The {sport: {market: k}} table back out of a generated shrink-k.js (the inside of Object.freeze is plain JSON)."""
    found = re.search(r"Object\.freeze\((\{.*\})\);", text, re.S)
    if not found:
        raise ValueError("shrink-k.js has no Object.freeze({...}) table")
    return json.loads(found.group(1))


def format_tables(doc: dict) -> str:
    """The text the script prints: one summary line per sport and market, then the per-season stability table."""
    lines = []
    for sport in SPORTS:
        for market in MARKETS:
            e = doc["markets"][sport][market]
            slope = "n/a" if e.get("k_raw") is None else f"{e['k_raw']:+.4f}"
            span = f"seasons {e['seasons'][0]}-{e['seasons'][1]}" if e["seasons"] else "no graded games"
            note = f"  [{e['reason']}]" if e.get("reason") else ""
            lines.append(f"{sport} {market}: k={e['k']:.4f} (unclamped slope {slope}) on n={e['n']} games, {span}{note}")
            if e["per_season"]:
                lines.append("  season  k_train  n_train  n_test      raw   shrunk   market  (metric)")
                for r in e["per_season"]:
                    lines.append(f"  {r['season']}    {r['k']:.4f}  {r['n_train']:>7}  {r['n_test']:>6}  {r['raw']:>7.4f}  "
                                 f"{r['shrunk']:>7.4f}  {r['market']:>7.4f}  {r['metric']}")
    return "\n".join(lines)
````

- [ ] **Step 4: Run the fit tests to verify they pass**

Run: `uv run pytest -q tests/shrink/test_game_fit.py`
Expected: every test in this file passes (together with the script tests of Step 6 the count is `27 passed`).

- [ ] **Step 5: Write the failing tests for the script**

Create `tests/scripts/test_fit_game_shrink.py`:

````python
"""scripts/fit_game_shrink.py: injected walk-forward rows in, two generated files out, nothing else. No database, no network."""
import importlib.util
import json
import random
import sys
from pathlib import Path

from sportsmodel.shrink import game_fit

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fit_game_shrink", ROOT / "scripts" / "fit_game_shrink.py")
fgs = importlib.util.module_from_spec(_spec)
sys.modules["fit_game_shrink"] = fgs
_spec.loader.exec_module(fgs)

AS_OF = "2026-10-14T12:00:00+00:00"


def raw_rows(k_true=0.4, seasons=(2016, 2017, 2018, 2019), n=400):
    rng = random.Random(5)
    out = []
    for s in seasons:
        for i in range(n):
            close = rng.uniform(-20, 20)
            model = close + rng.uniform(-8, 8)
            out.append({"season": s, "week": 1 + i % 14, "home_team": "H", "away_team": "A", "model_margin": model,
                        "model_total": 52.0, "market_spread": close, "market_total": 50.0,
                        "actual_margin": close + k_true * (model - close), "actual_total": 51.0, "ml_home": None, "ml_away": None})
    return out


def run(tmp_path, rows, *extra, seen=None):
    out, js = tmp_path / "game_shrink_k.json", tmp_path / "shrink-k.js"

    def raw_fn(last_season):
        if seen is not None:
            seen.append(last_season)
        return rows

    code = fgs.main(["--as-of", AS_OF, "--out", str(out), "--site-out", str(js), *extra], raw_fn=raw_fn)
    return code, out, js


def test_it_writes_the_document_and_the_site_table_and_they_agree(tmp_path, capsys):
    code, out, js = run(tmp_path, raw_rows())
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["fitted_at"] == AS_OF and doc["model_version"] == {"cfb": "cfb-ratings-v2"}
    assert 0.3 < doc["markets"]["cfb"]["spread"]["k"] < 0.5         # the served bias (-0.58) nudges x slightly; k_true is 0.4
    assert all(e["k"] == 0.0 and e["n"] == 0 for s in ("nfl", "mlb") for e in doc["markets"][s].values())
    assert js.read_text() == game_fit.render_js(doc)
    shown = capsys.readouterr().out
    assert "cfb spread" in shown and "2018" in shown and "nfl moneyline" in shown     # the stability table is printed


def test_two_runs_with_the_same_inputs_write_identical_files(tmp_path):
    _, out, js = run(tmp_path, raw_rows())
    first = (out.read_text(), js.read_text())
    run(tmp_path, raw_rows())
    assert (out.read_text(), js.read_text()) == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["game_shrink_k.json", "shrink-k.js"]     # no temp file left behind


def test_dry_run_prints_but_writes_nothing(tmp_path, capsys):
    code, out, js = run(tmp_path, raw_rows(), "--dry-run")
    assert code == 0 and not out.exists() and not js.exists()
    assert "cfb spread" in capsys.readouterr().out


def test_no_walk_forward_rows_exits_non_zero_and_leaves_existing_files_alone(tmp_path):
    out, js = tmp_path / "game_shrink_k.json", tmp_path / "shrink-k.js"
    out.write_text("keep")
    js.write_text("keep")
    code, _, _ = run(tmp_path, [])
    assert code == 1 and out.read_text() == "keep" and js.read_text() == "keep"


def test_the_last_season_is_passed_to_the_walk_forward(tmp_path):
    seen = []
    run(tmp_path, raw_rows(), seen=seen)
    run(tmp_path, raw_rows(), "--last-season", "2026", seen=seen)
    assert seen == [2025, 2026]


def test_the_script_touches_no_database_network_or_env_file():
    text = (ROOT / "scripts" / "fit_game_shrink.py").read_text()
    for banned in ("DATABASE_URL", "get_postgres", "sportsmodel.db", "import db", "httpx", "requests", "urllib", "dotenv",
                   ".env", "psycopg"):
        assert banned not in text, banned
````

- [ ] **Step 6: Run them to verify they fail, then write the script**

Run: `uv run pytest -q tests/scripts/test_fit_game_shrink.py`
Expected: collection error, `FileNotFoundError` for `scripts/fit_game_shrink.py`.

Create `scripts/fit_game_shrink.py`:

````python
"""Fit the game-level shrink factors from the committed CFB history (profit step 2 Part B) and write assets/game_shrink_k.json and the
generated site table site/js/shrink-k.js.

Runs the leak-free CFB walk-forward (cfb.walkforward.raw_model_predictions, about 10 seconds) once over the committed schedules and
closing lines, converts every game to the numbers the site would have shown (build_gameline with an empty market, as scripts/generate_cfb.py
serves them), fits k for the moneyline, spread and total (300-game minimum, per-season walk-forward stability report), prints the stability
table, and writes the two files. NFL and MLB are written as explicit zeros. It reads only committed assets: nothing else is touched.

    uv run python scripts/fit_game_shrink.py [--last-season 2025] [--out assets/game_shrink_k.json]
                                             [--site-out site/js/shrink-k.js] [--as-of ISO8601] [--dry-run]

Run by hand at season's end (pass --last-season <year>) or when the served game model changes; there is no scheduled job.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.shrink import game_fit  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
OUT_JSON = ROOT / "assets" / "game_shrink_k.json"
OUT_JS = ROOT / "site" / "js" / "shrink-k.js"


def _load_backtest():
    """scripts/backtest_cfb_gameline.py, for its load_merged_schedule (the same merge the gates and the live fix use)."""
    spec = importlib.util.spec_from_file_location("backtest_cfb_gameline", ROOT / "scripts" / "backtest_cfb_gameline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_raw(last_season: int) -> list[dict]:
    """The walk-forward rows for every merged game up to and including `last_season` (Elo is continuous across seasons)."""
    from sportsmodel.cfb.walkforward import raw_model_predictions
    from sportsmodel.nfl.elo import EloConfig
    from sportsmodel.nfl.ratings import BlendConfig

    merged = _load_backtest().load_merged_schedule(str(ASSETS / "schedules.parquet"), str(ASSETS / "lines.parquet"))
    span = merged[merged["season"] <= last_season]
    j = json.loads((ASSETS / "rating.json").read_text())
    elo_cfg = EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"], base=j.get("base", 1500.0))
    blend_cfg = BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"])
    return raw_model_predictions(span, elo_cfg, blend_cfg)


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to a temp file next to `path`, then replace: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main(argv: list[str] | None = None, *, raw_fn=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--last-season", type=int, default=game_fit.LAST_SEASON)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--site-out", default=str(OUT_JS))
    ap.add_argument("--as-of", default=None, help="ISO 8601 timestamp recorded as fitted_at (default: now, UTC)")
    ap.add_argument("--dry-run", action="store_true", help="print the tables, write nothing")
    args = ap.parse_args(argv)

    fitted_at = args.as_of or datetime.now(timezone.utc).isoformat()
    raw = (raw_fn or load_raw)(args.last_season)
    gl_cfg = game_fit.gameline_config(json.loads((ASSETS / "gameline.json").read_text()))
    rows = game_fit.served_rows(raw, gl_cfg)
    if not rows:
        print("fit_game_shrink: the walk-forward produced no rows; nothing written", file=sys.stderr)
        return 1
    doc = game_fit.build_document(rows, fitted_at, last_season=args.last_season)
    print(game_fit.format_tables(doc))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    write_atomic(Path(args.out), json.dumps(doc, indent=2, sort_keys=True) + "\n")
    write_atomic(Path(args.site_out), game_fit.render_js(doc))
    print(f"wrote {args.out} and {args.site_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

- [ ] **Step 7: Run both Python test files to verify they pass**

Run: `uv run pytest -q tests/shrink/test_game_fit.py tests/scripts/test_fit_game_shrink.py`
Expected: `27 passed`.

- [ ] **Step 8: Run the real fit once and read the output**

Run (about 15 seconds; reads only `assets/cfb/*`, no database, no network):

```bash
uv run python scripts/fit_game_shrink.py
```

Expected: the printed summary lines for `cfb moneyline`, `cfb spread`, `cfb total` each followed by a per-season stability table (held-out seasons 2018-2025 for spread and total; for the moneyline 2021-2025, where 2021 has nothing earlier to train on so its k is 0), then `nfl ...` and `mlb ...` lines with `k=0.0000 ... [k = 0: ... fewer than 300 graded games]`, then `wrote ...game_shrink_k.json and ...shrink-k.js`. When this plan was checked (the same code run from a scratch copy of the repository) the output was: spread n = 7163, k = 0.0000 (unclamped slope -0.0158); total n = 7154, k = 0.0429; moneyline n = 3561, k = 0.0326; the run took about 12 seconds. **Sanity gate:** if any CFB k comes out above 0.2, stop and re-check the sign convention (Review Focus item 1) before committing anything.

Check the generated table:

```bash
cat site/js/shrink-k.js
```

Expected: the `/* GENERATED ... */` header and `const SHRINK_K = Object.freeze({` with `cfb`, `mlb`, `nfl` blocks, `mlb` and `nfl` all `0.0`.

**In your report, paste: (a) the three CFB `k` values with their `n`, and (b) the full per-season stability table the script printed.**

- [ ] **Step 9: Add the test that pins the committed assets, and run everything**

Append to `tests/shrink/test_game_fit.py`:

````python
def test_the_committed_assets_agree_and_nfl_mlb_are_explicit_zeros():
    doc = json.loads((ROOT / "assets" / "game_shrink_k.json").read_text())
    text = (ROOT / "site" / "js" / "shrink-k.js").read_text()
    assert text == game_fit.render_js(doc), "site/js/shrink-k.js is generated from assets/game_shrink_k.json: re-run scripts/fit_game_shrink.py"
    table = game_fit.parse_js(text)
    for sport in ("nfl", "cfb", "mlb"):
        for m in ("moneyline", "spread", "total"):
            assert 0.0 <= table[sport][m] <= 1.0 and table[sport][m] == doc["markets"][sport][m]["k"]
    for sport in ("nfl", "mlb"):
        assert all(e["k"] == 0.0 and e["n"] == 0 and e["reason"] for e in doc["markets"][sport].values())
    assert doc["model_version"]["cfb"] == "cfb-ratings-v2" and doc["min_games"] == 300
    for m, e in doc["markets"]["cfb"].items():
        assert e["n"] >= 300 or e["k"] == 0.0, m
        assert e["seasons"] == [] or e["seasons"][0] >= 2016, m
````

Run: `uv run pytest -q tests/shrink tests/scripts/test_fit_game_shrink.py`
Expected: all pass (the two new files plus the Part A tests in `tests/shrink`, which must stay green; `171 passed` when this plan was checked).

- [ ] **Step 10: Commit**

```bash
git add src/sportsmodel/shrink/game_fit.py scripts/fit_game_shrink.py tests/shrink/test_game_fit.py tests/scripts/test_fit_game_shrink.py assets/game_shrink_k.json site/js/shrink-k.js
git commit -m "feat(shrink): fit game-level shrink k on CFB history and generate the site table" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The site helper, the fixture tables, and the script wiring

**Files:**
- Create: `site/js/shrink.js`
- Create: `site/tests/fixtures/shrink-k-identity.js`, `site/tests/fixtures/shrink-k-mixed.js`, `site/tests/fixtures/shrink-k-bad.js`
- Create: `site/tests/shrink.test.mjs`
- Modify: `site/*.html` (all 10 pages: two new `<script>` tags after `js/metrics.js`, same cache key as today)
- Modify: `site/tests/smoke.test.mjs` (the script order list only)
- Modify: every other `site/tests/*.test.mjs` whose `loadScripts` list names `"js/metrics.js"` (swap in the identity fixture and `js/shrink.js`)

**Interfaces:**
- Consumes (Task 1): `site/js/shrink-k.js` (`const SHRINK_K`), `assets/game_shrink_k.json`. Existing: `r05` (`site/app.js`), `probToAmerican` (`site/js/metrics.js`).
- Produces (Tasks 3 and 4 rely on these exact names):
  - `shrinkK(sport: string, market: "moneyline" | "spread" | "total") -> number` in [0, 1]; unknown or unusable gives 0
  - `shrinkProb(p: number, m: number, k: number) -> number | null` (`clip(m + k(p − m), 0.01, 0.99)`; `null` for a non-number)
  - `shrinkLine(model: number, market: number, k: number) -> number | null` (`market + k(model − market)`)
  - `fairAmerican(p: number) -> number | null` (American price of a probability strictly between 0 and 1)
  - `shrinkHalf(model: number, market: number) -> number | null` (`r05(model)`, except the market number itself when `|model − market| < 1e-9`)
  - Test tables: `tests/fixtures/shrink-k-identity.js` (every sport k = 1), `shrink-k-mixed.js` (NFL 0, CFB 0.5, nothing for MLB), `shrink-k-bad.js` (junk values)

- [ ] **Step 1: Create the fixture tables**

Create `site/tests/fixtures/shrink-k-identity.js`:

````js
/* Test table: k = 1 everywhere, so page code shows the raw model exactly as it did before shrinking existed. Never deployed (tests/ is excluded by sync_site.sh). */
const SHRINK_K = Object.freeze({ nfl: { moneyline: 1, spread: 1, total: 1 }, cfb: { moneyline: 1, spread: 1, total: 1 }, mlb: { moneyline: 1, spread: 1, total: 1 } });
````

Create `site/tests/fixtures/shrink-k-mixed.js`:

````js
/* Test table: NFL k = 0 (the real situation), CFB k = 0.5, no MLB entry (unknown means k = 0). Never deployed. */
const SHRINK_K = Object.freeze({ nfl: { moneyline: 0, spread: 0, total: 0 }, cfb: { moneyline: 0.5, spread: 0.5, total: 0.5 } });
````

Create `site/tests/fixtures/shrink-k-bad.js`:

````js
/* Test table of junk: every lookup must come back 0. Never deployed. */
const SHRINK_K = Object.freeze({ cfb: { moneyline: NaN, spread: 1.5, total: "0.5" }, mlb: { moneyline: -0.1, spread: null, total: Infinity }, nfl: null });
````

- [ ] **Step 2: Write the failing helper tests**

Create `site/tests/shrink.test.mjs`:

````js
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";

const BASE = ["app.js", "js/metrics.js"];
const live = loadScripts([...BASE, "js/shrink-k.js", "js/shrink.js", "js/boot.js"]);                              // the committed table
const mixed = loadScripts([...BASE, "tests/fixtures/shrink-k-mixed.js", "js/shrink.js", "js/boot.js"]);          // NFL 0, CFB 0.5
const bad = loadScripts([...BASE, "tests/fixtures/shrink-k-bad.js", "js/shrink.js", "js/boot.js"]);
const none = loadScripts([...BASE, "js/shrink.js", "js/boot.js"]);                                                // no table loaded at all
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);

test("shrinkProb: market + k (model - market), clipped to [0.01, 0.99]", () => {
  near(mixed.shrinkProb(0.70, 0.50, 0), 0.50); near(mixed.shrinkProb(0.70, 0.50, 1), 0.70);
  near(mixed.shrinkProb(0.70, 0.50, 0.25), 0.55); near(mixed.shrinkProb(0.30, 0.50, 0.25), 0.45);
  assert.equal(mixed.shrinkProb(1, 0.5, 1), 0.99); assert.equal(mixed.shrinkProb(0, 0.5, 1), 0.01);
  assert.equal(mixed.shrinkProb(0.5, 0.995, 0), 0.99, "a market at 99.5% is clipped too"); assert.equal(mixed.shrinkProb(0.5, 0.004, 0), 0.01);
});

test("shrinkProb / shrinkLine: anything that is not a finite number gives null, never NaN", () => {
  for (const bad of [NaN, null, undefined, "0.5", Infinity]) {
    assert.equal(mixed.shrinkProb(bad, 0.5, 0.5), null); assert.equal(mixed.shrinkProb(0.5, bad, 0.5), null); assert.equal(mixed.shrinkProb(0.5, 0.5, bad), null);
    assert.equal(mixed.shrinkLine(bad, 7, 0.5), null); assert.equal(mixed.shrinkLine(7, bad, 0.5), null); assert.equal(mixed.shrinkLine(7, 7, bad), null);
  }
});

test("shrinkLine: market + k (model - market), in either sign convention", () => {
  near(mixed.shrinkLine(11, 7, 0.5), 9); near(mixed.shrinkLine(11, 7, 0), 7); near(mixed.shrinkLine(11, 7, 1), 11);
  near(mixed.shrinkLine(-11, -7, 0.5), -9, 1e-9);                    // the same gap in home-line space
  near(mixed.shrinkLine(55, 50, 0.25), 51.25);
});

test("fairAmerican: the American price of a probability; null outside (0, 1)", () => {
  assert.equal(mixed.fairAmerican(0.75), -300); assert.equal(mixed.fairAmerican(0.4), 150); assert.equal(mixed.fairAmerican(0.5), -100);
  for (const bad of [0, 1, -0.2, 1.2, NaN, null, undefined]) assert.equal(mixed.fairAmerican(bad), null, String(bad));
});

test("shrinkHalf: the half point, except that a model equal to the market IS the market (no rounding artefact)", () => {
  assert.equal(mixed.shrinkHalf(53.2, 50), 53); assert.equal(mixed.shrinkHalf(-8.8, -4.5), -9); assert.equal(mixed.shrinkHalf(52.9, 52.6), 53);
  assert.equal(mixed.shrinkHalf(51.75, 51.75), 51.75, "a quarter-point market at k = 0"); assert.equal(mixed.shrinkHalf(51.75 + 1e-12, 51.75), 51.75);
  assert.equal(mixed.shrinkHalf(NaN, 50), null); assert.equal(mixed.shrinkHalf(7, NaN), 7, "no market to equal: just the half point");
});

test("shrinkK: the committed table; an unknown sport or market is k = 0, never the raw model", () => {
  for (const sport of ["nfl", "mlb"]) for (const m of ["moneyline", "spread", "total"]) assert.equal(live.shrinkK(sport, m), 0, `${sport} ${m} is explicitly 0`);
  for (const [s, m] of [["nba", "spread"], ["cfb", "puckline"], ["", ""], [undefined, undefined], ["constructor", "spread"], ["__proto__", "total"]]) {
    assert.equal(live.shrinkK(s, m), 0, `${s}/${m}`);
  }
  assert.equal(mixed.shrinkK("cfb", "spread"), 0.5); assert.equal(mixed.shrinkK("nfl", "moneyline"), 0); assert.equal(mixed.shrinkK("mlb", "total"), 0, "no entry");
});

test("shrinkK: a malformed or missing table is k = 0", () => {
  for (const [s, m] of [["cfb", "moneyline"], ["cfb", "spread"], ["cfb", "total"], ["mlb", "moneyline"], ["mlb", "spread"], ["mlb", "total"], ["nfl", "spread"]]) {
    assert.equal(bad.shrinkK(s, m), 0, `${s}/${m}`);
  }
  assert.equal(none.shrinkK("cfb", "spread"), 0, "js/shrink-k.js not loaded: the lookup still answers 0");
});

test("js/shrink-k.js agrees with assets/game_shrink_k.json for every sport and market; NFL and MLB are explicit zeros", () => {
  const doc = JSON.parse(fs.readFileSync(new URL("../../assets/game_shrink_k.json", import.meta.url), "utf8"));
  for (const sport of ["nfl", "cfb", "mlb"]) for (const m of ["moneyline", "spread", "total"]) {
    const k = doc.markets[sport][m].k;
    assert.ok(k >= 0 && k <= 1, `${sport} ${m}`);
    assert.equal(live.shrinkK(sport, m), k, `${sport} ${m}`);
  }
  for (const sport of ["nfl", "mlb"]) for (const m of ["moneyline", "spread", "total"]) assert.equal(doc.markets[sport][m].k, 0);
  assert.ok(Object.isFrozen(live.SHRINK_K), "the table is frozen");
});
````

- [ ] **Step 3: Run it to verify it fails**

Run: `cd site && node --test tests/shrink.test.mjs`
Expected: FAIL with `ENOENT ... js/shrink.js` (the helper does not exist yet).

- [ ] **Step 4: Write the helper**

Create `site/js/shrink.js`:

````js
/* Shrinking a game-level model number toward the market (profit step 2 Part B, docs/superpowers/specs/2026-10-14-shrink-site-design.md).
   shown = market + k (model - market). k comes from SHRINK_K[sport][market] (js/shrink-k.js, generated by scripts/fit_game_shrink.py from
   assets/game_shrink_k.json; market = "moneyline" | "spread" | "total"): k = 0 shows the market, k = 1 the raw model. An unknown sport or
   market, or a table that is missing or malformed, is k = 0 -- never the raw model. Pure: no DOM, no fetch.
   Depends on app.js (r05) and metrics.js (probToAmerican). js/shrink-k.js is only read inside shrinkK, so it just has to be loaded before the
   first call. Every page that compares a model number with a market number goes through these functions (board.js, game.js, dashboard.js). */
const SHRINK_P_LO = 0.01, SHRINK_P_HI = 0.99;      // a shrunk probability is clipped to [0.01, 0.99]
const shrinkNum = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);
function shrinkK(sport, market) {
  let k;
  try { k = SHRINK_K[sport][market]; } catch { k = undefined; }
  return typeof k === "number" && Number.isFinite(k) && k >= 0 && k <= 1 ? k : 0;
}
// clip(market + k (model - market), 0.01, 0.99); null when any input is not a finite number.
function shrinkProb(p, m, k) {
  const pn = shrinkNum(p), mn = shrinkNum(m), kn = shrinkNum(k);
  if (pn == null || mn == null || kn == null) return null;
  return Math.min(SHRINK_P_HI, Math.max(SHRINK_P_LO, mn + kn * (pn - mn)));
}
// market + k (model - market), in whatever line space both numbers share (home margin, the home book line, a total); null on a bad input.
function shrinkLine(model, market, k) {
  const mo = shrinkNum(model), ma = shrinkNum(market), kn = shrinkNum(k);
  if (mo == null || ma == null || kn == null) return null;
  return ma + kn * (mo - ma);
}
// The American price of a probability strictly between 0 and 1, else null.
const fairAmerican = (p) => (shrinkNum(p) != null && p > 0 && p < 1 ? probToAmerican(p) : null);
// The model number as quoted on the page: the half point (r05), except that a value equal to its market IS the market number, so a model
// at k = 0 shows (and has an edge of exactly 0 against) a quarter-point line such as a 51.75 total.
const shrinkHalf = (model, market) => (shrinkNum(model) == null ? null : shrinkNum(market) != null && Math.abs(model - market) < 1e-9 ? market : r05(model));
````


- [ ] **Step 5: Run the helper tests to verify they pass**

Run: `cd site && node --test tests/shrink.test.mjs`
Expected: all pass.

- [ ] **Step 6: Wire the two scripts into every page and the smoke test**

Insert the two tags right after `js/metrics.js` on all ten pages (this task keeps the current cache key; Task 5 bumps it once):

```bash
cd site && sed -i '' 's#<script src="js/metrics.js?v=20261012b"></script>#&<script src="js/shrink-k.js?v=20261012b"></script><script src="js/shrink.js?v=20261012b"></script>#' *.html && grep -c "js/shrink.js" *.html
```

Expected: ten lines, each `...html:1` (every page is a single line, so `grep -c` counts lines).

Edit `site/tests/smoke.test.mjs`: in the `order` array change `"js/metrics.js", "js/ui.js"` to `"js/metrics.js", "js/shrink-k.js", "js/shrink.js", "js/ui.js"`. Nothing else in that file changes in this task.

- [ ] **Step 7: Make every other suite load the identity table and the helper**

Every existing suite that concatenates the page scripts must now also load a table and the helper (page code in Tasks 3 and 4 calls them). The identity table keeps their raw expectations valid. From `site/`:

```bash
for f in $(ls tests/*.test.mjs | grep -v -e smoke.test.mjs -e shrink.test.mjs); do sed -i '' 's#"js/metrics.js", #"js/metrics.js", "tests/fixtures/shrink-k-identity.js", "js/shrink.js", #g' "$f"; done
grep -c "shrink-k-identity" tests/*.test.mjs
npm test
```

Expected: the `grep -c` shows a positive count for every suite that listed `js/metrics.js` (app, board-left, board, board-right, data, dashboard, game, game_panels, ev, mlb_live, metrics, legacy_css, ui, track) and `0` for `shell`, `hero_stadium`, `team_color`, `smoke`, `shrink`; `npm test` passes with `tests 382`: the 374 existing tests plus the 8 new ones in `shrink.test.mjs`, no failures (no page code uses the helper yet).

- [ ] **Step 8: Commit**

```bash
git add site/js/shrink.js site/tests/fixtures site/tests/shrink.test.mjs site/tests/*.test.mjs site/*.html
git commit -m "feat(site): shrink helper (shrinkK/shrinkProb/shrinkLine/fairAmerican) and test tables, wired into every page" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Sport pages: shrunk fair prices, edges, Alpha and sorting

**Files:**
- Modify: `site/js/pages/board.js` (`boardShrink` new, `boardLines`, `boardAlpha`, `boardModelCells`, doc comment)
- Modify: `site/css/theme.css` (one rule: the board's CSS contract test requires every emitted `ca-*` class to be defined)
- Test: `site/tests/board.test.mjs`

**Interfaces:**
- Consumes (Task 2): `shrinkK(sport, market)`, `shrinkProb(p, m, k)`, `shrinkLine(model, market, k)`, `fairAmerican(p)`, `shrinkHalf(model, market)`. Existing: `r05`, `probToAmerican`, `americanToProb`, `boardNum`, `alphaScore`, `alphaTier`.
- Produces:
  - `boardShrink(sport, i) -> { ml, spread, total, edge: { ml, spread, total }, proj: { ml, spread, total } }`; `i = { pFav, mktProb, devig, anyMarket, margin, mktSpread, sign, total, mktTotal }` (see the code comment)
  - `boardLines(g, D)` keeps its shape and gains `proj: { ml, spread, total }` (booleans: that model number is an unshrunk projection because there is no market to shrink toward). Upcoming games are shrunk; graded games (`g.acc`) are exactly as before.
  - `boardAlpha(g, D)` unchanged signature; the moneyline fallback uses the shrunk probability and needs both prices.
  - `boardProj(html, on)` wraps a model cell with the muted " proj" label.

- [ ] **Step 1: Write the failing tests**

At the top of `site/tests/board.test.mjs`, directly after the existing `const g = loadScripts([...]);` line, add the mixed-table instance:

````js
// NFL k = 0, CFB k = 0.5, anything else (MLB, NBA) unknown = 0: the profit step 2 Part B tests below use this instance (h); g keeps k = 1 (the raw model).
const h = loadScripts(["app.js", "js/metrics.js", "tests/fixtures/shrink-k-mixed.js", "js/shrink.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/board.js", "js/boot.js"]);
````

Append to the end of `site/tests/board.test.mjs`:

````js
// ---- shrink: upcoming games show the model after shrinking toward the market (profit step 2 Part B) ----
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);
const SG = (o) => ({ sport: "cfb", game_pk: 7, away: "Away U", home: "Home U", ...o });
const UP = { home_win_prob: 0.8, pred_home_score: 33, pred_away_score: 22, market_spread: -7, market_total: 50 };    // model: home by 11, total 55, 80%; market: home -7 (book line), total 50
const PRICES = { mlBy: new Map([["7", { home_price: -233, away_price: 190 }]]) };
const NOVIG = (233 / 333) / ((233 / 333) + (100 / 290));                                                           // the home no-vig probability of -233 / +190

test("boardLines (upcoming, CFB k = 0.5): the fair ML, spread and total sit halfway to the market and the edges are half the raw gaps", () => {
  const L = h.boardLines(SG({ pred: UP }), PRICES), s = 0.5 * 0.8 + 0.5 * NOVIG;
  assert.equal(L.fav, "home");
  assert.equal(L.model.ml, g.probToAmerican(s)); near(L.edge.ml, (s - NOVIG) * 100);
  assert.equal(L.model.spread, -9); near(L.edge.spread, 2);            // raw: -11 and 4
  assert.equal(L.model.total, 52.5); near(L.edge.total, 2.5);          // raw: 55 and 5
  assert.deepEqual(L.proj, { ml: false, spread: false, total: false });
  const raw = g.boardLines(SG({ pred: UP }), PRICES);                  // the identity table is the old behaviour
  assert.equal(raw.model.spread, -11); near(raw.edge.spread, 4); assert.equal(raw.model.total, 55); near(raw.edge.total, 5);
});

test("boardLines (upcoming, NFL k = 0): the model IS the market, so every edge is exactly 0 however far the raw model sits from it", () => {
  const L = h.boardLines(SG({ sport: "nfl", pred: UP }), PRICES);
  assert.equal(L.model.spread, -7); assert.equal(L.edge.spread, 0);
  assert.equal(L.model.total, 50); assert.equal(L.edge.total, 0);
  near(L.edge.ml, 0); assert.equal(L.model.ml, g.probToAmerican(NOVIG));
});

test("boardLines: a quarter-point total at k = 0 gives edge exactly 0 and shows the market's own number", () => {
  const L = h.boardLines(SG({ sport: "nfl", pred: { ...UP, market_total: 51.75 } }), PRICES);
  assert.equal(L.model.total, 51.75); assert.equal(L.edge.total, 0);
});

test("boardLines: an away favourite is shrunk in the favourite's own direction (home line +3, model has the away side by 10)", () => {
  // the market has away -3 (home book line +3); the model's home margin is -10; at k = 0.5 the shrunk away margin is 6.5: the favourite's line -6.5, edge +3.5
  const L = h.boardLines(SG({ pred: { home_win_prob: 0.2, pred_home_score: 20, pred_away_score: 30, market_spread: 3, market_total: 50 } }),
    { mlBy: new Map([["7", { home_price: 140, away_price: -160 }]]) });
  assert.equal(L.fav, "away"); assert.equal(L.mkt.spread, -3); assert.equal(L.model.spread, -6.5); near(L.edge.spread, 3.5);
});

test("boardLines: no market line means the raw projection, labelled, with no edge", () => {
  const L = h.boardLines(SG({ pred: { home_win_prob: 0.8, pred_home_score: 33, pred_away_score: 22 } }), PRICES);     // two prices, no spread or total
  assert.equal(L.model.spread, -11); assert.equal(L.edge.spread, null); assert.equal(L.proj.spread, true);
  assert.equal(L.model.total, 55); assert.equal(L.edge.total, null); assert.equal(L.proj.total, true);
  assert.equal(typeof L.model.ml, "number"); assert.equal(L.proj.ml, false);                                         // the moneyline has both prices: shrunk as usual
  const none = h.boardLines(SG({ pred: { home_win_prob: 0.8, pred_home_score: 33, pred_away_score: 22 } }), {});    // no market at all
  assert.equal(none.model.ml, g.probToAmerican(0.8)); assert.equal(none.proj.ml, true); assert.equal(none.edge.ml, null);
});

test("boardLines: a spread but no moneyline shows no fair ML and no ML edge; one price only is the same (nothing to shrink toward)", () => {
  const a = h.boardLines(SG({ pred: UP }), {});
  assert.equal(a.model.ml, null); assert.equal(a.edge.ml, null); assert.equal(a.proj.ml, false); near(a.edge.spread, 2);
  const b = h.boardLines(SG({ pred: UP }), { mlBy: new Map([["7", { home_price: -210, away_price: null }]]) });
  assert.equal(b.model.ml, null); assert.equal(b.edge.ml, null); assert.equal(b.mlVig, true);
});

test("boardLines: a graded game keeps its recorded model numbers whatever k is (history is never shrunk)", () => {
  const F = SG({ pred: null, acc: { win_prob: 0.64, predicted_winner: "Home U", home_team_name: "Home U", away_team_name: "Away U", pred_margin: 11, pred_total: 55, market_spread: -7, market_total: 50 } });
  const closing = { closing: new Map([["7|home", -233], ["7|away", 190]]) };
  const kept = h.boardLines(F, closing);
  assert.deepEqual(kept, g.boardLines(F, closing), "k = 0.5 and k = 1 agree on a graded game");
  assert.equal(kept.model.spread, -11); near(kept.edge.spread, 4); assert.equal(kept.model.total, 55); near(kept.edge.total, 5);
  assert.deepEqual(kept.proj, { ml: false, spread: false, total: false });
});

test("boardAlpha: the moneyline Alpha uses the shrunk probability and needs both prices; a tiered opportunity keeps its score", () => {
  const D = (m) => ({ mlBy: new Map([["7", m]]) }), p = { home_win_prob: 0.53 }, prices = { home_price: 100, away_price: -120 };
  assert.equal(g.boardAlpha(SG({ pred: p }), D(prices)).score, 75, "unshrunk: a 3-point edge at +100");
  assert.equal(h.boardAlpha(SG({ sport: "nfl", pred: p }), D(prices)).score, 45, "NFL k = 0: the shrunk probability is the no-vig price, no edge, the floor score");
  assert.equal(h.boardAlpha(SG({ pred: p }), D(prices)).score, 49, "CFB k = 0.5 sits between");
  assert.equal(h.boardAlpha(SG({ pred: p }), D({ home_price: 100, away_price: null })), null, "one price: no Alpha Score");
  assert.equal(h.boardAlpha(SG({ pred: p }), {}), null);
  assert.equal(h.boardAlpha(SG({ opp: { alpha: 81, tier: "STRONG", kind: "line", market: "total", side: "over", sport: "cfb" } }), {}).score, 81);
});

test("boardView: sorting by Edge or Alpha uses the shrunk numbers (a big raw NFL gap no longer outranks a smaller CFB one)", () => {
  const nfl = { sport: "nfl", game_pk: 1, away: "A", home: "B", commence: null, date: "2026-10-04", final: false, acc: null, opp: null,
    pred: { sport: "nfl", home_win_prob: 0.8, pred_home_score: 33, pred_away_score: 22, market_spread: -7, market_total: 50 } };
  const cfb = { ...nfl, sport: "cfb", game_pk: 2, pred: { ...nfl.pred, sport: "cfb", home_win_prob: 0.7, pred_home_score: 28, pred_away_score: 22 } };
  const prices = { home_price: 120, away_price: -140 }, D = { mlBy: new Map([["1", prices], ["2", prices]]) };
  const order = (inst, sort) => inst.boardView([nfl, cfb], D, { team: "", time: "", market: "all", sort }).map((x) => x.game_pk);
  assert.deepEqual(order(h, "edge"), [2, 1]); assert.deepEqual(order(h, "alpha"), [2, 1]);
  assert.deepEqual(order(g, "edge"), [1, 2], "unshrunk, the bigger raw gap wins");
});

test("boardUpcomingTable: a game with no market line labels its projections and shows no edge or Alpha; a game with one shows the shrunk edge", () => {
  const D = { mlBy: new Map([["7", { home_price: -233, away_price: 190 }], ["8", { home_price: -233, away_price: 190 }]]) };
  const full = SG({ game_pk: 7, pred: UP, commence: "2026-10-11T17:00:00Z" });
  const bare = SG({ game_pk: 8, pred: { home_win_prob: 0.8, pred_home_score: 33, pred_away_score: 22 }, commence: "2026-10-11T17:00:00Z" });
  const html = h.boardUpcomingTable([full, bare], D, { market: "all" });
  const rows = html.split("<tr data-href").slice(1);
  assert.match(rows[0], /\+2\.0/, "the spread edge is half of the raw +4");
  assert.ok(!rows[0].includes("ca-bd-proj"));
  assert.equal((rows[1].match(/ca-bd-proj/g) || []).length, 2, "the spread and total projections are labelled");
  assert.ok(!/NaN|undefined|Infinity/.test(html));
});
````

- [ ] **Step 2: Run them to verify they fail**

Run: `cd site && node --test tests/board.test.mjs`
Expected: the new tests FAIL (for example `h.boardLines(...).proj` is undefined and `L.model.spread` is `-11`, not `-9`); the existing tests still pass.

- [ ] **Step 3: Add `boardShrink` and rewrite the upcoming branch of `boardLines`**

In `site/js/pages/board.js`, insert this just above the `// The market, the model and the edge of one game` comment (above `function boardLines`):

````js
// The model numbers of an UPCOMING game after shrinking toward the market (profit step 2 Part B), all from the favourite's side. Every number goes through
// js/shrink.js: market + k (model - market), k = SHRINK_K[sport][market] (0 for NFL and MLB, so they show the market).
//   i = { pFav (the model's favourite probability, raw), mktProb (the favourite's market probability), devig (mktProb is the no-vig one: both ML prices exist),
//         anyMarket (the game has any market number), margin (the model's HOME margin, raw), mktSpread (the market's HOME book line: negative = home favoured),
//         sign (+1 home favourite, -1 away), total (raw), mktTotal }
// Returns { ml (fair American), spread (the favourite's line), total, edge: { ml (prob pts), spread (pts), total (pts) }, proj: { ml, spread, total } }.
//   * a number with a market to shrink toward is shrunk and has an edge (k = 0: the market itself, edge exactly 0);
//   * a number with NO market line is the raw projection (proj = true) and has no edge;
//   * the fair ML needs a two-sided no-vig price: without one it is null ("—"), except for a game with no market number at all (the raw projection, proj = true).
function boardShrink(sport, i) {
  const out = { ml: null, spread: null, total: null, edge: { ml: null, spread: null, total: null }, proj: { ml: false, spread: false, total: false } };
  if (i.pFav != null) {
    if (i.devig && i.mktProb != null) {
      const s = shrinkProb(i.pFav, i.mktProb, shrinkK(sport, "moneyline"));
      if (s != null) { out.ml = fairAmerican(s); out.edge.ml = (s - i.mktProb) * 100; }
    } else if (!i.anyMarket) { out.ml = fairAmerican(i.pFav); out.proj.ml = true; }
  }
  if (i.margin != null) {
    if (i.mktSpread != null) {
      const sMargin = shrinkLine(i.margin, -i.mktSpread, shrinkK(sport, "spread"));         // the market's expected home margin is minus its home book line
      out.spread = shrinkHalf(-i.sign * sMargin, i.sign * i.mktSpread) || 0;               // the favourite's line, to the half point (the market's own at k = 0)
      out.edge.spread = i.sign * i.mktSpread - out.spread;
    } else { out.spread = r05(-i.sign * i.margin) || 0; out.proj.spread = true; }
  }
  if (i.total != null) {
    if (i.mktTotal != null) { out.total = shrinkHalf(shrinkLine(i.total, i.mktTotal, shrinkK(sport, "total")), i.mktTotal); out.edge.total = out.total - i.mktTotal; }
    else { out.total = r05(i.total); out.proj.total = true; }
  }
  return out;
}

````

Then replace this exact block in `boardLines` (from `const modLine =` through the `return` line):

````js
  const modLine = fav && margin != null ? r05(-sign * margin) || 0 : null;     // the model's numbers are quoted to the half point everywhere (modelLineStr, game page), so the edge is exactly model - market as shown
  if (total != null) total = r05(total);
  const modMl = pFav != null && pFav > 0 && pFav < 1 ? probToAmerican(pFav) : null;
  // The ML edge is against the NO-VIG probability of the favourite's price: normalised with the other side's price (proportional) when both exist, else the
  // vigged implied probability (mlVig = true; the Model Projections ML tooltip says so).
  const pf = mktMl == null ? NaN : americanToProb(mktMl), po = fav ? americanToProb(fav === "home" ? prices.away : prices.home) : NaN, devig = Number.isFinite(pf) && Number.isFinite(po) && pf + po > 0;
  const mktProb = Number.isFinite(pf) ? (devig ? pf / (pf + po) : pf) : null;
  const edge = { ml: pFav != null && mktProb != null ? (pFav - mktProb) * 100 : null,
    spread: mktLine != null && modLine != null ? mktLine - modLine : null, total: mktTotal != null && total != null ? total - mktTotal : null };
  return { fav, favName, favAbbr: fav ? teamShort(favName, g.sport) : "", mkt: { ml: mktMl, spread: mktLine, total: mktTotal }, model: { ml: modMl, spread: modLine, total }, edge, mlVig: mktProb != null && !devig };
````

with:

````js
  // The market's probability for the favourite: NO-VIG (the price normalised with the other side's price) when both prices exist, else the vigged implied probability (mlVig = true).
  const pf = mktMl == null ? NaN : americanToProb(mktMl), po = fav ? americanToProb(fav === "home" ? prices.away : prices.home) : NaN, devig = Number.isFinite(pf) && Number.isFinite(po) && pf + po > 0;
  const mktProb = Number.isFinite(pf) ? (devig ? pf / (pf + po) : pf) : null;
  let modLine, modMl, edge, proj = { ml: false, spread: false, total: false };
  if (a) {
    // A graded game: the model's RECORDED numbers against the closing prices, exactly as stored. History (the Final table, the Model Projections bands) is never shrunk.
    modLine = fav && margin != null ? r05(-sign * margin) || 0 : null;     // the model's numbers are quoted to the half point everywhere (modelLineStr, game page), so the edge is exactly model - market as shown
    if (total != null) total = r05(total);
    modMl = pFav != null && pFav > 0 && pFav < 1 ? probToAmerican(pFav) : null;
    edge = { ml: pFav != null && mktProb != null ? (pFav - mktProb) * 100 : null,
      spread: mktLine != null && modLine != null ? mktLine - modLine : null, total: mktTotal != null && total != null ? total - mktTotal : null };
  } else {
    // An upcoming game: every model number is pulled toward the market by the fitted k (boardShrink).
    const S = boardShrink(g.sport || p.sport, { pFav, mktProb, devig, anyMarket: mktSpread != null || mktTotal != null || prices.home != null || prices.away != null,
      margin: fav ? margin : null, mktSpread, sign, total, mktTotal });
    modLine = S.spread; modMl = S.ml; total = S.total; edge = S.edge; proj = S.proj;
  }
  return { fav, favName, favAbbr: fav ? teamShort(favName, g.sport) : "", mkt: { ml: mktMl, spread: mktLine, total: mktTotal }, model: { ml: modMl, spread: modLine, total }, edge, mlVig: mktProb != null && !devig, proj };
````

Also update the file's header doc comment: replace the line

`   rounded to the half point (modelLineStr / r05, the same as the game page), so the edge is exactly what the two columns show.`

with

`   rounded to the half point (modelLineStr / r05, the same as the game page), so the edge is exactly what the two columns show.\n   UPCOMING games show the model after shrinking toward the market (js/shrink.js: market + k x (model - market), k per sport and market from js/shrink-k.js; NFL and\n   MLB k = 0, so they show the market and an edge of 0). A number with no market line is the raw projection, labelled "proj", with no edge; the fair ML needs a two-sided\n   no-vig price ("—" without one). Graded games keep their recorded model numbers.`

- [ ] **Step 4: Shrink the moneyline Alpha Score and label projections**

In `boardAlpha`, replace this block:

````js
  const pHome = boardNum(g.pred && g.pred.home_win_prob), m = D.mlBy && D.mlBy.get(String(g.game_pk));
  if (pHome == null || !m) return null;
  let best = null;
  for (const [p, price, team] of [[pHome, boardNum(m.home_price), g.home], [1 - pHome, boardNum(m.away_price), g.away]]) {
    const imp = price == null ? NaN : americanToProb(price);
    if (!Number.isFinite(imp)) continue;
````

with:

````js
  const pHome = boardNum(g.pred && g.pred.home_win_prob), m = D.mlBy && D.mlBy.get(String(g.game_pk));
  if (pHome == null || !m) return null;
  // The model's win probability is shrunk toward the market's NO-VIG probability (js/shrink.js), which needs both prices: with one side missing there is no Alpha Score.
  const hp = boardNum(m.home_price), ap = boardNum(m.away_price), ih = hp == null ? NaN : americanToProb(hp), ia = ap == null ? NaN : americanToProb(ap);
  if (!Number.isFinite(ih) || !Number.isFinite(ia) || ih + ia <= 0) return null;
  const sHome = shrinkProb(pHome, ih / (ih + ia), shrinkK(g.sport, "moneyline"));
  if (sHome == null) return null;
  let best = null;
  for (const [p, price, team] of [[sHome, hp, g.home], [1 - sHome, ap, g.away]]) {
    const imp = americanToProb(price);
````

and in the comment above `boardAlpha`, change `the MODEL's win probability vs the best` to `the MODEL's win probability (shrunk toward the market's no-vig probability, js/shrink.js) vs the best`.

Replace the `boardModelCells` line:

````js
const boardModelCells = (L) => `<td data-m="moneyline" class="ca-bd-gs">${boardOdds(L.model.ml)}</td><td data-m="spread">${boardModSpread(L)}</td><td data-m="total">${boardTotal(L.model.total)}</td>`;
````

with:

````js
// A model number that is only a projection (no market line to shrink toward) carries a muted "proj" label.
const boardProj = (html, on) => (on ? `${html}<small class="muted ca-bd-proj" title="Projection only: no market line"> proj</small>` : html);
const boardModelCells = (L) => `<td data-m="moneyline" class="ca-bd-gs">${boardProj(boardOdds(L.model.ml), L.proj.ml)}</td><td data-m="spread">${boardProj(boardModSpread(L), L.proj.spread)}</td><td data-m="total">${boardProj(boardTotal(L.model.total), L.proj.total)}</td>`;
````

- [ ] **Step 5: Style the label and update the one existing test whose behavior changes on purpose**

In `site/css/theme.css`, directly after the line `.ca-bd-sp i{font-style:normal;color:var(--muted)}` add on a new line (without it the existing test `CSS contract: every ca-* class the board ... exists in theme.css` fails on `ca-bd-proj`):

````css
.ca-bd-proj{font-size:.78em;font-weight:400;white-space:nowrap}
````

In `site/tests/board.test.mjs`, in the test `boardLines: every line from the market favourite's side; ...`, replace

````js
  assert.ok(Math.abs(V.edge.ml - (0.7 - 210 / 310) * 100) < 1e-9 && V.mlVig === true, "no price for the other side: vigged fallback, flagged");
````

with

````js
  assert.equal(V.edge.ml, null); assert.equal(V.model.ml, null); assert.equal(V.mlVig, true, "no price for the other side: a shrunk fair price needs the no-vig probability, so no ML edge and no fair ML (the flag stays for graded rows)");
````

- [ ] **Step 6: Run the whole site suite**

Run: `cd site && npm test`
Expected: all pass, `tests 392` (382 plus the 10 new board tests). The existing board, board-left and board-right suites run on the identity table, so every other raw expectation holds. If another existing test fails, the cause must be one of these deliberate rule changes (a one-sided ML price now gives no ML edge or Alpha; a game with a spread but no moneyline now shows "—" for the fair ML); update that assertion to the new rule and say so in your report. Do not weaken any test of graded rows.

- [ ] **Step 7: Commit**

```bash
git add site/js/pages/board.js site/css/theme.css site/tests/board.test.mjs
git commit -m "feat(site): sport pages show shrunk fair prices, edges and Alpha for upcoming games" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Game page and dashboard

**Files:**
- Modify: `site/js/pages/game.js` (`gmShrunkLines` new, `gmLean`, `gmOdds`, `modelProjectionRows`, `gmRawLine` new, `gmReadCard`, header comment)
- Modify: `site/js/pages/dashboard.js` (`dashModelVsMarket`)
- Modify: `site/css/theme.css` (one rule)
- Test: `site/tests/game.test.mjs`, `site/tests/dashboard.test.mjs`

**Interfaces:**
- Consumes (Task 2): `shrinkK`, `shrinkProb`, `shrinkLine`, `shrinkHalf`. Existing: `gmSplit(dist, threshold)`, `gmDists`, `consensusAmerican`, `numOrNull`, `teamShort`, `modelLineStr`, `lineStr`, `pct1`, `gmEsc` (= `ctxEsc`).
- Produces:
  - `gmShrunkLines(pred) -> { margin, total, sMargin, sTotal, hasSpread, hasTotal }` (margin and total are the raw model numbers from `pred_home_score` and `pred_away_score`; `sMargin` and `sTotal` are shrunk toward `−market_spread` and `market_total`; `hasSpread` and `hasTotal` say a posted line exists)
  - `gmOdds(...)` keeps its shape; every side's `modelProb` is now the shrunk probability (`null` when there is nothing to shrink toward) and each moneyline side also carries `rawModelProb`
  - `gmRawLine(pred) -> string` (the escaped paragraph `Raw model projection: ...`, or `""`)
  - `dashModelVsMarket(preds)` returns the same shape from shrunk numbers

- [ ] **Step 1: Write the failing game-page tests**

At the top of `site/tests/game.test.mjs`, after the existing `const g = loadScripts(FILES);` line, add:

````js
// NFL k = 0, CFB k = 0.5: the profit step 2 Part B tests below use this instance (mixed); g keeps k = 1 (the raw model).
const mixed = loadScripts(["app.js", "js/metrics.js", "tests/fixtures/shrink-k-mixed.js", "js/shrink.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/game.js", "js/boot.js"]);
````

Append to the end of `site/tests/game.test.mjs`:

````js
// ---- shrink: the game page's model numbers are the model after shrinking toward the market (profit step 2 Part B) ----
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);
const normPmf = (mean, sd, lo, hi) => { const w = []; for (let v = lo; v <= hi; v++) w.push(Math.exp(-0.5 * ((v - mean) / sd) ** 2)); const s = w.reduce((a, b) => a + b, 0); return w.map((x) => x / s); };
const MARGIN_N = { kind: "margin", offset: 60, pmf: normPmf(11, 13, -60, 60) }, TOTAL_N = { kind: "pmf", pmf: normPmf(55, 10, 0, 150) };
const PM = { sport: "cfb", game_pk: 9, away_team_name: "Away U", home_team_name: "Home U", home_win_prob: 0.8, pred_home_score: 33, pred_away_score: 22, market_spread: -7, market_total: 50,
  margin_dist: MARGIN_N, total_dist: TOTAL_N, commence_time: "2099-01-01T00:00:00Z" };    // model: home by 11, total 55, 80%; market: home -7 (book line), total 50
const MLROW = { home_price: -233, away_price: 190 };
const M_HOME = (233 / 333) / ((233 / 333) + (100 / 290));                                  // the home no-vig probability of -233 / +190
const split = (dist, t) => { const { under, over } = g.distProbs(dist, t); return { over: over / (over + under), under: under / (over + under) }; };
const EVN = [evRow({ market: "spread", side: "home", best_line_implied: 0.52 }), evRow({ market: "spread", side: "away", best_line_implied: 0.52 }),
  evRow({ market: "total", side: "over", best_line_implied: 0.52 }), evRow({ market: "total", side: "under", best_line_implied: 0.52 })];

test("gmOdds (CFB k = 0.5): the win, cover and total probabilities are the model halfway to the market", () => {
  const o = mixed.gmOdds(PM, [], MLROW, null, null), s = 0.5 * 0.8 + 0.5 * M_HOME;
  near(o.moneyline.home.modelProb, s); near(o.moneyline.away.modelProb, 1 - s); assert.equal(o.moneyline.home.rawModelProb, 0.8);
  // margin 11 -> 9 (market 7): the shifted margin beats 7 when the raw margin beats 9; away covers +7 when the shifted margin is below 7
  near(o.spread.home.modelProb, split(MARGIN_N, 9).over); near(o.spread.away.modelProb, split(MARGIN_N, 9).under);
  // total 55 -> 52.5 (market 50): over 50 when the raw total beats 52.5
  near(o.total.over.modelProb, split(TOTAL_N, 52.5).over); near(o.total.under.modelProb, split(TOTAL_N, 52.5).under);
  const raw = g.gmOdds(PM, [], MLROW, null, null);                     // the identity table is the old behaviour
  near(raw.spread.home.modelProb, split(MARGIN_N, 7).over); near(raw.total.over.modelProb, split(TOTAL_N, 50).over);
});

test("the Read at k = 0 (NFL): a model far from the market still says no edge; at CFB k = 0.5 the same game has a smaller divergence than the raw model", () => {
  const nfl = { ...PM, sport: "nfl" };
  assert.equal(mixed.gameRead(mixed.gmCandidates(nfl, mixed.gmOdds(nfl, EVN, MLROW, null, null))), null);
  const half = mixed.gameRead(mixed.gmCandidates(PM, mixed.gmOdds(PM, EVN, MLROW, null, null)));
  const raw = g.gameRead(g.gmCandidates(PM, g.gmOdds(PM, EVN, MLROW, null, null)));
  assert.ok(half && raw && half.edgePp > 0 && half.edgePp < raw.edgePp, `${half && half.edgePp} < ${raw && raw.edgePp}`);
});

test("gmOdds: without a posted spread or total the model has no spread / total probability (projection only); one moneyline price leaves no win probability", () => {
  const bare = { ...PM, market_spread: null, market_total: null };
  const o = mixed.gmOdds(bare, [], MLROW, null, null);
  assert.equal(o.spread.home.modelProb, null); assert.equal(o.spread.away.modelProb, null); assert.equal(o.total.over.modelProb, null); assert.equal(o.total.under.modelProb, null);
  assert.ok(o.moneyline.home.modelProb != null);
  const one = mixed.gmOdds(PM, [], { home_price: -233, away_price: null }, null, null);
  assert.equal(one.moneyline.home.modelProb, null); assert.equal(one.moneyline.home.rawModelProb, 0.8);
  const rows = mixed.modelProjectionRows(bare, o);
  assert.match(rows[1].model, /\(projection\)$/); assert.match(rows[2].model, /\(projection\)$/); assert.equal(rows[1].pts, null); assert.equal(rows[2].diff, null);
  const noMkt = mixed.modelProjectionRows(bare, mixed.gmOdds(bare, [], null, null, null));
  assert.equal(noMkt[0].model, `${g.pct1(0.8)} (projection)`, "no market number at all: the raw win probability, labelled");
  const spreadOnly = mixed.modelProjectionRows({ ...PM, market_total: null }, mixed.gmOdds({ ...PM, market_total: null }, [], null, null, null));
  assert.equal(spreadOnly[0].model, "—", "a market but no moneyline: nothing to show");
});

test("modelProjectionRows (CFB k = 0.5): the CappingAlpha column is the shrunk model at the market's own line", () => {
  const rows = mixed.modelProjectionRows(PM, mixed.gmOdds(PM, [], MLROW, null, null)), s = 0.5 * 0.8 + 0.5 * M_HOME;
  assert.equal(rows[0].model, g.pct1(s));
  assert.match(rows[1].model, / -9 \(\d+\.\d%\)$/); near(rows[1].pts, 2);          // raw: -11 and 4
  assert.match(rows[2].model, /^52\.5 \(\d+\.\d%\)$/); near(rows[2].pts, 2.5);      // raw: 55.0 and 5
  const nfl = { ...PM, sport: "nfl" }, n = mixed.modelProjectionRows(nfl, mixed.gmOdds(nfl, [], MLROW, null, null));
  assert.match(n[1].model, / -7 \(/); near(n[1].pts, 0); assert.match(n[2].model, /^50\.0 \(/); near(n[2].pts, 0);     // k = 0: the market's own numbers
});

test("gmLean uses the shrunk model: NFL (k = 0) never leans on spread or total, CFB (k = 0.5) still does", () => {
  assert.deepEqual({ ...mixed.gmLean({ ...PM, sport: "nfl" }) }, { ml: "home", spread: null, total: null });
  assert.deepEqual({ ...mixed.gmLean(PM) }, { ml: "home", spread: "home", total: "over" });
  assert.deepEqual({ ...mixed.gmLean({ ...PM, sport: "nfl", market_total: 51.75 }) }, { ml: "home", spread: null, total: null }, "a quarter-point market at k = 0: no lean");
});

test("the Read card shows the raw model projection, labelled, and escapes team names", () => {
  const evil = { ...PM, home_team_name: 'Home <i>x</i> & "U"' };
  const html = mixed.gmReadCard({ r: evil, sport: "cfb", evRows: [], mlRow: MLROW, simRows: [], lineBy: null });
  assert.match(html, /Raw model projection: /); assert.ok(html.includes("total 55.0") && html.includes("80.0% win"), "the raw margin side, total and win probability");
  assert.ok(!html.includes("<i>x</i>") && !html.includes('"U"'), "team names are escaped");
  assert.ok(!/NaN|undefined/.test(html));
  const empty = mixed.gmReadCard({ r: { sport: "cfb", home_team_name: "A", away_team_name: "B", commence_time: "2099-01-01T00:00:00Z" }, sport: "cfb", evRows: [], mlRow: null, simRows: [], lineBy: null });
  assert.ok(!empty.includes("Raw model projection"), "no projection numbers, no line");
});
````

Update the two existing tests whose expectations change on purpose.

First, `gmLean: moneyline favorite, spread / total lean at half-point rounding, none when on the line` builds its prediction without a sport, and an unknown sport is now k = 0 (the market), so it would see no lean. In that test replace

````js
  const p = { home_win_prob: 0.7, pred_home_score: 30, pred_away_score: 20, market_spread: -3, market_total: 45.5 };
````

with

````js
  const p = { sport: "cfb", home_win_prob: 0.7, pred_home_score: 30, pred_away_score: 20, market_spread: -3, market_total: 45.5 };
````

Second, in `modelProjectionRows: away favorite, under lean, missing odds ...` replace

````js
  assert.equal(ml.market, "—"); assert.equal(ml.model, "70.0%"); assert.equal(ml.diff, null);
````

with

````js
  assert.equal(ml.market, "—"); assert.equal(ml.model, "—", "a market (spread, total) but no moneyline price: no win probability to show"); assert.equal(ml.diff, null);
````

- [ ] **Step 2: Write the failing dashboard test**

At the top of `site/tests/dashboard.test.mjs`, after the existing `const g = loadScripts([...]);` line, add:

````js
const mixed = loadScripts(["app.js", "js/metrics.js", "tests/fixtures/shrink-k-mixed.js", "js/shrink.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/boot.js"]);    // NFL k = 0, CFB k = 0.5
````

Append to the end of `site/tests/dashboard.test.mjs`:

````js
test("model vs market (shrunk): NFL (k = 0) shows no gap, CFB (k = 0.5) shows half of the raw gap, a quarter-point total at k = 0 shows none", () => {
  const base = { game_pk: 1, home_team_name: "H", away_team_name: "A", pred_home_score: 33, pred_away_score: 22, market_spread: -7, market_total: 50 };
  assert.deepEqual(mixed.dashModelVsMarket([{ ...base, sport: "nfl" }]), []);
  assert.deepEqual(mixed.dashModelVsMarket([{ ...base, sport: "nfl", market_total: 51.75 }]), []);
  const m = mixed.dashModelVsMarket([{ ...base, sport: "cfb" }]);
  assert.deepEqual(m.map((x) => [x.market, x.from, x.to, x.diff]), [["total", 50, 52.5, 2.5], ["spread", -7, -9, -2]]);
});
````

- [ ] **Step 3: Run them to verify they fail**

Run: `cd site && node --test tests/game.test.mjs tests/dashboard.test.mjs`
Expected: the new tests FAIL (`gmOdds` still returns raw probabilities, `gmShrunkLines`/`gmRawLine` do not exist, `dashModelVsMarket` shows raw gaps); the other tests still pass except the one whose expectation you just changed, which fails until Step 4.

- [ ] **Step 4: Implement the game page**

In `site/js/pages/game.js`:

(a) Header comment: replace the line

`   (home_win_prob; margin / total distributions at the market line), never ev_picks.true_prob, which is Pinnacle's no-vig line.`

with

`   (home_win_prob; margin / total distributions at the market line), never ev_picks.true_prob, which is Pinnacle's no-vig line.\n   Those model numbers are first shrunk toward the market (js/shrink.js: market + k x (model - market), k per sport and market from js/shrink-k.js; NFL and MLB\n   k = 0), so the Read, the Model Projection table, the cover probability and the leans compare the shrunk model; the raw projection stays on the Projected cards\n   and in the "Raw model projection" line of the Read card. With no posted line the model value is a labelled projection with no probability or edge.`

(b) Add `gmShrunkLines` immediately above `// The side the model favors in each market` (above `function gmLean`) and replace `gmLean`:

````js
// The model's expected home margin and total (raw, from the projected score) and the same numbers after shrinking toward the posted market line
// (js/shrink.js). The posted spread is the HOME BOOK line (negative = home favoured), so the market's expected home margin is its negative.
// hasSpread / hasTotal: a posted line exists to shrink toward; without one the shrunk value is just the raw projection and nothing is compared.
function gmShrunkLines(pred) {
  const p = pred || {}, sport = p.sport, h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score), ls = numOrNull(p.market_spread), lt = numOrNull(p.market_total);
  const margin = h != null && a != null ? h - a : null, total = h != null && a != null ? h + a : null;
  const hasSpread = margin != null && ls != null, hasTotal = total != null && lt != null;
  return { margin, total, hasSpread, hasTotal,
    sMargin: hasSpread ? shrinkLine(margin, -ls, shrinkK(sport, "spread")) : margin, sTotal: hasTotal ? shrinkLine(total, lt, shrinkK(sport, "total")) : total };
}

// The side the model favors in each market: moneyline favorite; spread / total only when the SHRUNK model's half-point
// line is off the market line (same rule as the legacy lean, on the shrunk number); null when unknowable or on the line.
function gmLean(pred) {
  const p = pred || {}, wp = numOrNull(p.home_win_prob), h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score);
  const ls = numOrNull(p.market_spread), lt = numOrNull(p.market_total), S = gmShrunkLines(p);
  const out = { ml: wp == null ? null : wp >= 0.5 ? "home" : "away", spread: null, total: null };
  if (ls != null && h != null && a != null) {
    const edge = shrinkHalf(S.sMargin, -ls) + ls;      // the shrunk model margin (half point) minus the market's expected home margin (-ls); exactly 0 at k = 0
    out.spread = Math.abs(edge) < 0.25 ? null : edge > 0 ? "home" : "away";
  }
  if (lt != null && h != null && a != null) out.total = S.sTotal > lt ? "over" : S.sTotal < lt ? "under" : null;
  return out;
}
````

Delete the old `function gmLean(pred) {...}` (the block from `// The side the model favors in each market:` through its closing `}`), since the version above replaces it.

(c) Replace `gmOdds` (the whole function including its leading comment block, which starts `// Per-market view of the game's odds`) with:

````js
// Per-market view of the game's odds: consensus moneyline prices, and per side the line it is priced at, the implied
// probability of the price you would take (latest ev_picks row; moneyline falls back to the moneylines view's best price)
// and the model probability at that same line AFTER SHRINKING toward the market (js/shrink.js). modelProb is null whenever there is nothing to
// shrink toward: no distribution, no two-sided moneyline price (the win probability is shrunk toward the NO-VIG probability of both sides' prices),
// no posted spread / total. Spread and total are shrunk in points (gmShrunkLines) and the model's own distribution is shifted by the same amount:
// P(M + d > t) = P(M > t - d). The moneyline sides also carry rawModelProb, the unshrunk win probability.
function gmOdds(pred, evRows, mlRow, sim, lineBy) {
  const p = pred || {}, by = new Map(gmLatest(evRows).map((r) => [`${r.market}|${r.side}`, r])), d = gmDists(p, sim);
  const wp = numOrNull(p.home_win_prob), S = gmShrunkLines(p);
  const mlSide = (side) => {
    const r = by.get(`moneyline|${side}`), cons = mlRow ? consensusAmerican(mlRow[`${side}_prices`]) : null;
    const bestPrice = r && numOrNull(r.best_price) != null ? +r.best_price : mlRow ? numOrNull(mlRow[`${side}_price`]) : null;
    const price = cons != null ? cons : bestPrice;
    const prob = (x) => (x != null && Number.isFinite(americanToProb(x)) ? americanToProb(x) : null);
    return { price, implied: prob(price), bestPrice, bestImplied: r && gmImplied(r) != null ? gmImplied(r) : prob(bestPrice),
      rawModelProb: wp == null ? null : side === "home" ? wp : 1 - wp, modelProb: null };
  };
  const ml = { away: mlSide("away"), home: mlSide("home") };
  const mHome = ml.home.implied != null && ml.away.implied != null ? ml.home.implied / (ml.home.implied + ml.away.implied) : null;
  const wpS = wp == null || mHome == null ? null : shrinkProb(wp, mHome, shrinkK(p.sport, "moneyline"));
  if (wpS != null) { ml.home.modelProb = wpS; ml.away.modelProb = 1 - wpS; }
  const one = (market, side, shrinkable, modelAt) => {
    const r = by.get(`${market}|${side}`), line = gmSideLine(p, market, side, lineBy), imp = r ? gmImplied(r) : null, price = r ? numOrNull(r.best_price) : null;
    return { line, price, implied: imp, bestPrice: price, bestImplied: imp, modelProb: line == null || !shrinkable ? null : modelAt(line) };
  };
  const m = d.margin, t = d.total, dM = S.hasSpread ? S.sMargin - S.margin : 0, dT = S.hasTotal ? S.sTotal - S.total : 0;
  return {
    moneyline: ml,
    spread: { line: numOrNull(p.market_spread),
      away: one("spread", "away", S.hasSpread, (L) => { const s = gmSplit(m, L - dM); return s && s.under; }),     // away covers when the (shifted) home margin < its line
      home: one("spread", "home", S.hasSpread, (L) => { const s = gmSplit(m, -L - dM); return s && s.over; }) },   // home covers when the (shifted) home margin > -its line
    total: { line: numOrNull(p.market_total),
      over: one("total", "over", S.hasTotal, (L) => { const s = gmSplit(t, L - dT); return s && s.over; }),
      under: one("total", "under", S.hasTotal, (L) => { const s = gmSplit(t, L - dT); return s && s.under; }) },
  };
}
````

(d) Replace `modelProjectionRows` (the whole function including its leading comment block) with:

````js
// Model Projection table: Market / CappingAlpha / Difference for moneyline (the model's favorite), spread (the model's
// side) and total (the model's lean). The CappingAlpha cell is the model AFTER SHRINKING toward the market (gmOdds), at the same line the Market
// cell shows. diff = shrunk model minus market implied probability, in points (null when either is unknown); pts = how many more points the
// shrunk model asks for than the market, from the same side (null when unknown). With no posted line the cell is the raw projection, labelled
// "(projection)", with no probability, diff or pts. The moneyline needs a two-sided no-vig price: with a market but no moneyline it is "—"; with
// no market number at all it is the raw win probability, labelled.
function modelProjectionRows(pred, odds) {
  const p = pred || {}, o = odds || {}, sport = p.sport;
  const wp = numOrNull(p.home_win_prob), S = gmShrunkLines(p);
  const blank = (label) => ({ label, market: "—", model: "—", diff: null, pts: null });
  const ab = (side) => teamShort(side === "home" ? p.home_team_name : p.away_team_name, sport);
  const anyMkt = [o.moneyline && o.moneyline.home.price, o.moneyline && o.moneyline.away.price, o.spread && o.spread.line, o.total && o.total.line].some((x) => x != null);
  const rows = [];
  const ml = blank("Moneyline");
  if (wp != null) {
    const side = wp >= 0.5 ? "home" : "away", e = o.moneyline && o.moneyline[side];
    ml.model = e && e.modelProb != null ? pct1(e.modelProb) : anyMkt ? "—" : `${pct1(Math.max(wp, 1 - wp))} (projection)`;
    if (e && e.price != null) { ml.market = `${ab(side)} ${oddsStr(e.price)}${gmPctP(e.implied)}`; if (e.implied != null && e.modelProb != null) ml.diff = (e.modelProb - e.implied) * 100; }
  }
  rows.push(ml);
  const sp = blank("Spread");
  if (S.margin != null) {
    const side = S.margin >= 0 ? "home" : "away", e = o.spread && o.spread[side], mktHome = numOrNull(p.market_spread);
    // The half-point line the cell shows, in the chosen side's line space (the home line is minus the margin); the market's own line when k = 0
    const modelLine = (S.hasSpread ? shrinkHalf(side === "home" ? -S.sMargin : S.sMargin, side === "home" ? mktHome : -mktHome) : r05(side === "home" ? -S.margin : S.margin)) || 0;
    sp.model = `${ab(side)} ${modelLineStr(modelLine)}${S.hasSpread ? gmPctP(e ? e.modelProb : null) : " (projection)"}`;
    if (e && e.line != null) {
      sp.market = `${ab(side)} ${lineStr(e.line)}${gmPctP(e.implied)}`;
      if (S.hasSpread) {
        sp.pts = e.line - modelLine;
        if (e.modelProb != null && e.implied != null) sp.diff = (e.modelProb - e.implied) * 100;
      }
    }
  }
  rows.push(sp);
  const tt = blank("Total");
  if (S.total != null) {
    const T0 = o.total ? o.total.line : null, lean = T0 == null ? null : S.total >= T0 ? "over" : "under", shown = S.hasTotal ? S.sTotal : S.total;
    const e = lean && o.total[lean], T = e && e.line != null ? e.line : T0;
    tt.model = `${shown.toFixed(1)}${S.hasTotal ? gmPctP(e ? e.modelProb : null) : " (projection)"}`;
    if (T != null) {
      tt.market = `${lean === "under" ? "U" : "O"} ${T}${gmPctP(e ? e.implied : null)}`;
      if (S.hasTotal) {
        tt.pts = lean === "over" ? shown - T : T - shown;
        if (e && e.modelProb != null && e.implied != null) tt.diff = (e.modelProb - e.implied) * 100;
      }
    }
  }
  rows.push(tt);
  return rows;
}
````

(e) Replace `gmReadCard` (the whole function) and add `gmRawLine` just above it:

````js
// "Raw model projection: ATL -9 · total 49.0 · ATL 70.0% win": the model's own numbers before any shrinking, always shown with the Read so the
// shrunk read can be checked against them (the Projected cards and the Win Probability card are the same raw numbers). Escaped; "" when the row has none.
function gmRawLine(pred) {
  const p = pred || {}, sport = p.sport, wp = numOrNull(p.home_win_prob), h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score), bits = [];
  const ab = (side) => teamShort(side === "home" ? p.home_team_name : p.away_team_name, sport);
  if (h != null && a != null) { const m = h - a, side = m >= 0 ? "home" : "away"; bits.push(`${ab(side)} ${modelLineStr(side === "home" ? -m : m)}`, `total ${(h + a).toFixed(1)}`); }
  if (wp != null) { const side = wp >= 0.5 ? "home" : "away"; bits.push(`${ab(side)} ${pct1(Math.max(wp, 1 - wp))} win`); }
  return bits.length ? `<p class="ca-gm-raw muted">Raw model projection: ${gmEsc(bits.join(" · "))}</p>` : "";
}
function gmReadCard(D) {
  const started = gmStarted(D.r), m = started ? null : gameRead(gmCandidates(D.r, gmOddsOf(D))), raw = gmRawLine(D.r);
  const noun = m && { moneyline: "win", spread: "cover", total: "hit" }[m.market];
  const body = started
    ? `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>Pre-game read — this game has started</h2>
        <p>The model read is shown before kickoff.</p>${raw}</div>`
    : m
    ? `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>${gmEsc(m.label)} shows the strongest model divergence</h2>
        <p>The market implies a ${pct1(m.impliedProb)} ${noun} probability. CappingAlpha estimates <b>${pct1(m.modelProb)}</b>.</p>${raw}</div>
       <div class="ca-gm-edge"><b>${pStr(m.edgePp)}</b><span>MODEL EDGE</span></div>`
    : `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>No model edge on this game</h2>
        <p>No game-line market prices above the model's estimate right now.</p>${raw}</div>`;
  return `<section class="ca-card ca-gm-read" id="gm-read"><div class="ca-gm-mark" aria-hidden="true">C<i>α</i></div>${body}</section>`;
}
````

Delete the old `function gmReadCard(D) {...}` block that these replace.

(f) In `site/css/theme.css`, after the line `.ca-gm-read p b{color:var(--ink)}` add on a new line:

````css
.ca-gm-read p.ca-gm-raw{margin-top:8px;font-size:13px;color:var(--muted)}
````

- [ ] **Step 5: Implement the dashboard**

In `site/js/pages/dashboard.js`, replace the body of `dashModelVsMarket` (keep the comment line above it, extended) with:

````js
// Model line vs market line per game (spread on the home side, total), biggest gap first. The model number is shrunk toward the market first
// (js/shrink.js, k per sport and market): the gap shown is the evidence-weighted one, and 0 (so absent) for a sport with no fitted k.
function dashModelVsMarket(preds) {
  const out = [];
  for (const r of preds || []) {
    if (!r || !finite(r.pred_home_score) || !finite(r.pred_away_score)) continue;
    const margin = +r.pred_home_score - +r.pred_away_score, total = +r.pred_home_score + +r.pred_away_score;
    if (finite(r.market_spread)) { const ms = +r.market_spread, to = shrinkHalf(shrinkLine(-margin, ms, shrinkK(r.sport, "spread")), ms) || 0; out.push({ pred: r, market: "spread", from: ms, to, diff: to - ms }); }
    if (finite(r.market_total)) { const mt = +r.market_total, to = shrinkHalf(shrinkLine(total, mt, shrinkK(r.sport, "total")), mt); out.push({ pred: r, market: "total", from: mt, to, diff: to - mt }); }
  }
  return out.filter((x) => x.diff !== 0).sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
}
````

(The spread is shrunk in home-line space, `−margin` against the home book line, which is the same arithmetic as shrinking the margin toward `−market_spread`.)

- [ ] **Step 6: Run the whole site suite**

Run: `cd site && npm test`
Expected: all pass, `tests 399` (392 plus 6 new game tests and 1 new dashboard test). The existing game and dashboard suites run on the identity table, so their raw expectations hold. If another existing game test fails, the cause must be one of the deliberate rules (a model probability now needs the pred's projected score and a posted line for spread and total, and both sides' moneyline prices for the win probability): add the missing fixture field (`pred_home_score`, `pred_away_score`, a market line, or the second ML price) or update the assertion, and say which in your report. Do not weaken any assertion about graded or historical data.

- [ ] **Step 7: Commit**

```bash
git add site/js/pages/game.js site/js/pages/dashboard.js site/css/theme.css site/tests/game.test.mjs site/tests/dashboard.test.mjs
git commit -m "feat(site): game page and dashboard compare the shrunk model with the market; raw projection line" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Bump the cache key (once) and check the pages

**Files:**
- Modify: `site/*.html` (the cache key on every script and on `theme.css`)
- Modify: `site/tests/smoke.test.mjs`

**Interfaces:**
- Consumes: Tasks 2 to 4 (all page changes are in).
- Produces: cache key `20261014a` on every page; `smoke.test.mjs` pins it and checks that nothing in `site/` carries the previous key `20261012b`.

- [ ] **Step 1: Bump the key everywhere and update the smoke test**

```bash
cd site && sed -i '' 's/20261012b/20261014a/g' *.html tests/smoke.test.mjs && grep -c "20261014a" *.html
```

Expected: ten lines, each `...html:1` (every page is a single line; the key now sits on every script and on `theme.css`).

Then in `site/tests/smoke.test.mjs` change the obfuscated previous-key line

`const old = ["2026", "1007", "d"].join(""), seen = [];`

to

`const old = ["2026", "1012", "b"].join(""), seen = [];`

so the test now proves that no file in `site/` still carries `20261012b`.

- [ ] **Step 2: Run the site suite and confirm the key changed exactly once**

Run: `cd site && npm test`
Expected: all pass, including the two cache-key tests in `smoke.test.mjs`.

Run (from the repository root): `git log --oneline -S"20261014a" -- site/nfl.html`
Expected: nothing yet (the bump is not committed). After the commit in Step 4 the same command must list exactly one commit.

- [ ] **Step 3: Check the pages in a browser, desktop first**

Serve the site and load the three pages at 1280, 1440 and 1920 pixels wide, each with a hard reload (the pages read the public API with the anon key already in `app.js`; no credentials are needed from you):

```bash
cd site && python3 -m http.server 4173
```

Open `http://localhost:4173/nfl.html`, `http://localhost:4173/cfb.html` and one game page linked from the CFB board. Check: the NFL Edge columns read 0.0 / 0.0 / 0.0% (or a dash) and the Alpha Score column is low or a dash; CFB edges are small; nothing reads `NaN` or `undefined`; the game page Read card carries the grey "Raw model projection" line and says "No model edge" for NFL games; a past-week Final table is unchanged. If the Browser tools are not available, write "visual check not run" in the report; the go-live checklist (Task 6) lists the same check for the user. Stop the server when done.

- [ ] **Step 4: Commit**

```bash
git add site/*.html site/tests/smoke.test.mjs
git commit -m "chore(site): bump cache key to 20261014a" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git log --oneline -S"20261014a" -- site/nfl.html
```

Expected: the `git log` line lists exactly one commit, the one you just made.

---

### Task 6: The go-live checklist

**Files:**
- Create: `docs/superpowers/reports/2026-10-14-shrink-site-go-live.md`

**Interfaces:**
- Consumes: Task 1's `assets/game_shrink_k.json` (the current k values), the finished site.
- Produces: the user's deploy and refit instructions.

- [ ] **Step 1: Print the current k table to paste into the document**

```bash
uv run python - <<'EOF'
import json
d = json.load(open("assets/game_shrink_k.json"))
print("| Sport | Moneyline | Spread | Total |\n|---|---|---|---|")
for s in ("nfl", "cfb", "mlb"):
    print(f"| {s.upper()} | " + " | ".join(f"{d['markets'][s][m]['k']:.3f}" for m in ("moneyline", "spread", "total")) + " |")
print()
for m in ("moneyline", "spread", "total"):
    e = d["markets"]["cfb"][m]
    print(f"cfb {m}: n={e['n']}, seasons {e['seasons']}, unclamped slope {e['k_raw']}")
EOF
```

- [ ] **Step 2: Write the document**

Create `docs/superpowers/reports/2026-10-14-shrink-site-go-live.md` with the content below. In the section "The k values committed with this change", replace the sentence "Printed from `assets/game_shrink_k.json` when this document was written:" and the two lines after it with the table and the three `cfb ...` lines that Step 1 printed (the numbers must be the real ones; do not leave the sentence in place).

````markdown
# Shrink game-level model numbers on the site: go-live checklist (2026-10-14)

Branch `feat/shrink-site`. Everything below is for you to run; nothing has been deployed, no workflow was changed and no code connected to the database. The fit used only committed files.

## What this changes, in plain words

Until now the sport pages and the game page showed the model's own fair moneyline, spread and total, and measured every edge against the market from that raw number. The edge audit showed that was too bold: against the closing line the CFB model adds almost nothing (coefficient +0.04, standard error 0.16) and the NFL model nothing at all. From the first load of the new pages, every game-level model number for an **upcoming** game is `market + k x (model - market)`, where k is how much of its disagreement with the close the model has earned in graded history. The same shrunk number drives the CappingAlpha columns, Edge, the moneyline Alpha Score, the Edge and Alpha sorts, the game-page Read, the Model Projection table, the Cover Probability and the dashboard's model-vs-market gaps, so they always agree. The raw projection stays on the Projected Score / Spread / Total and Win Probability cards and in a grey "Raw model projection" line under the Read.

## The k values committed with this change

Printed from `assets/game_shrink_k.json` when this document was written:

(replace with the table and lines from Step 1)

NFL and MLB are **0** for all three markets: the current NFL game model (the ML sim, in place since 2026-09-28) has about six weeks of graded games and the MLB game model almost none, both under the 300-game minimum. k = 0 means the site shows the market, so NFL and MLB edges are 0 until a fit exists. `assets/game_shrink_k.json` also holds the per-season stability table (k and out-of-sample error for raw, shrunk and market for each held-out season 2018-2025).

## What you will notice

| Where | Before | After |
|---|---|---|
| Sport page, upcoming games: CappingAlpha ML / Spread / Total | the model's own numbers | the shrunk numbers (NFL and MLB: equal to the market) |
| Edge columns, Edge sort | model minus market, often several points | shrunk minus market: tiny for CFB, exactly 0.0 for NFL and MLB |
| Alpha Score (moneyline fallback), Alpha sort | from the raw win probability | from the shrunk probability; a game with no spread or only one moneyline price has none |
| Game page Read | the strongest raw divergence | usually "No model edge on this game" |
| Game page Model Projection, Cover Probability | raw model at the market line | shrunk model at the market line |
| A game with no market line | model numbers with an edge | the raw projection labelled "proj" / "(projection)", no edge |
| A game with a spread but no moneyline | a fair ML | "—" |
| Past-week results, Track Record, Model vs Market and Model Projections bands, props board, +EV board, early tickets | as before | unchanged |

The Alpha card on the game page shows the best +EV opportunity (Pinnacle's no-vig line against the soft books). It never held a CappingAlpha game-level model number, so it is unchanged.

## 1. Merge, then deploy the site yourself

1. Merge the PR into `main`.
2. From the repository root run `scripts/sync_site.sh` (it backs up your CappingAlpha folder, then copies `site/` there without the tests), then deploy that folder to Cloudflare Pages as usual.
3. Hard reload (Cmd-Shift-R). The scripts carry the cache key `20261014a`, so a normal reload also picks the new files up.

There is no migration, no workflow and no secret involved.

## 2. Check it by eye (desktop first)

At 1280, 1440 and 1920 pixels wide, with a hard reload, open `nfl.html`, `cfb.html` and one game page. NFL Edge cells should read 0.0 and the Alpha Score column should be low or a dash. CFB edges should be small. The Read card should carry the grey "Raw model projection" line. A past week's Final table should look exactly as it did. Nothing should say `NaN` or `undefined`.

## 3. Refitting k

The fit is manual: there is no scheduled job. Refit at the end of each season, or when the served game model changes.

```bash
uv run python scripts/fit_game_shrink.py --last-season 2026
```

It reads only committed files (about 15 seconds), prints the per-season stability table, and rewrites `assets/game_shrink_k.json` and `site/js/shrink-k.js` together; a test fails if the two ever disagree. Commit both, run `cd site && npm test`, `scripts/sync_site.sh`, and redeploy. Bump the cache key (the `?v=` on every script and on `theme.css` in the ten `site/*.html` pages, and the two constants in `site/tests/smoke.test.mjs`) so browsers load the new table.

The script fits CFB only. NFL and MLB stay explicit zeros until a follow-up adds their fit from graded game predictions, once the current NFL game model reaches 300 graded games.

## 4. If the served model changes

k is fitted for one model. `assets/game_shrink_k.json` records which (`model_version.cfb`, now `cfb-ratings-v2`). If the CFB model changes (v3, a new prior, a new `gameline.json`) the old k no longer describes it: refit before trusting the edges, or set the CFB k values to 0 in `assets/game_shrink_k.json` and rerun the script so the site shows the market until the new fit exists. The same holds for a new NFL game model.

## 5. Known limits

- **The fit is against the closing line; the site compares with the current line.** Conservative: the model gets no credit for beating stale numbers it has not been shown to beat.
- **The fitted model is the plain walk-forward, not exactly what is served.** Served CFB v2 also blends a preseason prior into each team's Elo; the walk-forward (`raw_model_predictions`) does not. The fit therefore sees a slightly weaker model in the early weeks of each season, which can only make k smaller.
- **Quarter-point and odd lines.** At k = 0 the page shows the market's own number, so the edge is exactly 0 even on a 51.75 total.
- **Edges come from the shrink, not from line shopping.** On the game page the shrunk win probability is compared with the best available price, so a moneyline can still show a small positive read when one book is out of line with the consensus. That is a price gap, not a model edge.
- **Expect almost no model edges.** With k near 0 for CFB and 0 for NFL and MLB, the Read will almost always say "No model edge on this game". That is the intended result of the audit, not a fault.

## 6. If something looks wrong

Any page problem: revert the merge commit (`git revert -m 1 <merge sha>`), run `scripts/sync_site.sh` and redeploy; the data and the models were never touched. To show the raw model again without reverting, set every k to 1 in both `assets/game_shrink_k.json` and `site/js/shrink-k.js` (the agreement test needs both) and redeploy. This is not recommended: it removes the protection the audit asked for.
````

- [ ] **Step 3: Run the full verification**

Run, from the repository root:

```bash
uv run pytest -q
(cd site && npm test)
git status --short
```

Expected: `uv run pytest -q` shows the baseline plus the new tests (3461 passed before this work, plus every test added in Task 1) and the single baseline failure `test_the_migration_is_the_specs_idempotent_additive_alter` caused by the unrelated local edit; no other failure. `npm test` passes in full. `git status --short` shows only ` M db/migration_shrink_props.sql` (the user's file, untouched) and the new report file.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/reports/2026-10-14-shrink-site-go-live.md
git commit -m "docs: shrink-site go-live checklist" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.**

| Spec requirement | Task |
|---|---|
| §1 closed forms for points and probability, clamp, Σx² = 0 → 0, ties and invalid prices excluded | 1 (`k_points`, `moneyline_legs`, `closed_form_k` reuse) |
| §1 CFB sample: v2 as served, 2016-2025, 2015 warm-up, p from the served conversion | 1 (`served_rows` via `build_gameline`, `load_raw`, Decision 3 and 4) |
| §1 per-season stability 2018-2025 with raw / shrunk / market MAE or Brier; served k on all graded seasons | 1 (`season_report`, `fit_market`, printed table, JSON `per_season`) |
| §1 300-game minimum | 1 (`fit_k`, test at 299/300) |
| §1 NFL and MLB explicit zeros with reason | 1 (`ZERO_REASON`, `build_document`) |
| §1 outputs JSON + frozen site table + agreement test | 1 (`render_js`, `parse_js`, committed-assets test), 2 (node agreement test) |
| §1 refit is manual | 1 (script flags), 6 (procedure) |
| §2 helper `shrinkProb` / `shrinkLine` / `fairAmerican`, unknown key → 0 | 2 |
| §2 fair ML without a moneyline market (decided from existing code) | Decision 5; 3 (`boardShrink`), 4 (`modelProjectionRows`) |
| §2 rule: Edge, Alpha, sorting, Read, dashboard through the helper | 3 (`boardLines`, `boardAlpha`, `boardView` unchanged but fed shrunk numbers), 4 (`gmOdds`, `gmLean`, `gameRead` via candidates, `dashModelVsMarket`) |
| §2 no market line → "projection", no edge or Alpha | 3, 4 |
| §2 sport pages: upcoming shrunk; past rows, Top Alpha Edges, Market Pulse unchanged | 3 (graded branch untouched; `board-left` / `board-right` not edited) |
| §2 game page: Read, odds and candidate edges shrunk; raw projection line | 4 |
| §2 game page Alpha card | Decision 6 (it holds no model game-level number; left alone; stated in the go-live doc) |
| §2 dashboard | 4 |
| §2 unchanged list (Track Record, historical panels, props board, +EV board, early tickets) | Global Constraints; no task edits those files |
| House rules: `ctxEsc`, desktop first, cache key once and last | 4 (`gmEsc`), 5 |
| Testing: Python closed forms, sign convention, walk-forward, 300 minimum, zeros, ties and prices, JSON/JS match | 1 |
| Testing: site helper maths, sport page, past rows, no-market case, game Read / raw line, cache key | 2, 3, 4, 5 |
| Testing: visual check at 1280 / 1440 / 1920 with hard reload | 5 Step 3 and 6 section 2 |
| Risks recorded (model-only game numbers vanish, mostly no edge, fitted against the close, served model changes) | 6 sections 4 and 5 |

**Placeholder scan.** No "TBD", "similar to Task N" or empty steps. The only fill-in is Task 6 Step 2's k table, which Step 1 prints from the committed file and says to paste in verbatim.

**Type consistency.** `shrinkK(sport, market)`, `shrinkProb(p, m, k)`, `shrinkLine(model, market, k)`, `fairAmerican(p)`, `shrinkHalf(model, market)` are defined in Task 2 and used with those exact signatures in Tasks 3 and 4. `boardShrink(sport, i)` returns `{ml, spread, total, edge, proj}` and `boardLines` returns `proj` alongside the old fields. `gmShrunkLines(pred)` returns `{margin, total, hasSpread, hasTotal, sMargin, sTotal}` and is used by `gmLean`, `gmOdds` and `modelProjectionRows`. `game_fit` names (`served_rows`, `point_legs`, `moneyline_legs`, `fit_k`, `season_report`, `build_document`, `render_js`, `parse_js`, `format_tables`) are the same in the module, its tests and the script.

**Review Focus.** Each of the five lines names the tests that pin it, and all five test groups exist in the owning task.
