# NFL P2 — Game-line probability model (distributions + total model + market shrinkage)

**Date:** 2026-08-24
**Status:** Approved design (pre-plan)
**Parent spec:** `docs/superpowers/specs/2026-08-23-nfl-model.md` (P2 phase)
**Depends on:** P1 (merged, commit dc5b04e) — `nfl/{teams,data,elo,srs,ratings}.py`, committed
`assets/nfl/*.parquet`, tuned `assets/nfl/rating.json`.

## Problem & goal

P1 produces an opponent-adjusted **expected margin** (Elo × SoS blend). P2 turns that plus a new
**total** model into a full, bettable game line: wrap the expected margin and total in probability
**distributions** (the serving layer's `margin_dist`/`total_dist`/`win_prob`), and **shrink both
toward the current market line** by a weight that decays over the season so early weeks (little
2026 data) stay sane and the model grows independent as data accumulates. A walk-forward
backtest — using nflverse's historical closing lines as the market — fits the distribution σ's
and the shrinkage curves.

Output must match the **existing MLB serving contract** exactly (`game_predictions.margin_dist` /
`total_dist` / `home_win_prob` / `pred_*`), so P4's board/grade passes (sport-parameterized in
P0) consume NFL rows with no format work.

## Scope / non-goals

**In (P2):**
- Extend the committed `schedules.parquet` snapshot with the market/result columns nflverse
  already carries (`spread_line`, `total_line`, `result`, `total`, moneylines).
- `points.py` — opponent-adjusted offense/defense **points** ratings (SRS-style), season-to-date,
  shrunk to league average early → expected home/away points → expected total.
- `shrink.py` — market shrinkage of expected margin and total toward the current line by
  `w(week)`.
- `gameline.py` — orchestrator: P1 margin + points total + shrinkage → Normal distributions →
  `margin_dist`/`total_dist`/`win_prob`/`pred_home_score`/`pred_away_score`, in the serving format.
- `scripts/backtest_nfl_gameline.py` — walk-forward fit of σ_margin, σ_total, and the `w(week)`
  curves against nflverse historical lines; committed params.

**Out (later phases):**
- Player props → **P3**.
- The `generate_nfl` producer, NFL board/grade passes, front-end → **P4** (P2 only guarantees
  format compatibility).
- **Key-number-aware** margin distribution (NFL margins cluster on 3/7) → a later refinement; P2
  uses a Normal baseline.
- Any change to the MLB pipeline. NFL is additive.

## Data (extends P1's snapshot)

Re-run `scripts/build_nfl_snapshots.py` with `normalize_schedule` keeping the market/result
columns nflverse already returns: add `result` (home−away final margin), `total` (final total
points), `spread_line`, `total_line`, `away_moneyline`, `home_moneyline` to the retained
schedule columns. (These were dropped in P1's minimal snapshot.)

**Spread-line convention — verify in Task 1:** nflverse `spread_line` is documented as *positive
when the home team is favored*. Confirm empirically that `spread_line` positively correlates with
`result` (home margin) on the committed data before using it as the market margin; record the
confirmed convention. The backtest's `market_margin = spread_line` (home perspective) and
`market_total = total_line`.

## Points / total model (`points.py`)

Opponent-adjusted points ratings, reusing P1's iterative solve pattern (`srs.py`).

- **Ratings:** each team gets `off` (points scored above league average, opponent-adjusted) and
  `def` (points allowed above league average, opponent-adjusted; a good defense has negative
  `def`). Expected points for team *i* vs team *j* = `LG_AVG + off_i + def_j`.
- **Solve (season-to-date):** iterate to a fixed point (Gauss-Seidel, like `srs.compute_srs`):
  `off_i = mean_over_games(points_scored_i − LG_AVG − def_opponent)`;
  `def_i = mean_over_games(points_allowed_i − LG_AVG − off_opponent)`. Pin each of `off`/`def`
  to zero-mean. `LG_AVG` = season-to-date league average points per team per game.
- **Early-season shrinkage (points-level "sane early"):** shrink each team's `off`/`def` toward
  0 by an empirical-Bayes factor `n_games / (n_games + K_points)` (`K_points` a small constant,
  tuned or fixed ~4), so a team with few games barely deviates from league average — before the
  market shrinkage even applies.
- **Outputs:** `expected_home_pts = LG_AVG + off_home + def_away`,
  `expected_away_pts = LG_AVG + off_away + def_home`;
  `expected_total_model = expected_home_pts + expected_away_pts`.
- **API:** `compute_points_ratings(games_so_far, k_points) -> {team: {off, def}}, lg_avg`;
  `expected_total(ratings, lg_avg, home, away) -> float`.

**Margin vs total split (settled):** the game line's **margin** comes from P1
(`ratings.expected_margin`, the Elo×SoS blend); the **total** comes from `points.py`. The
per-team predicted scores are then decomposed consistently:
`pred_home_score = (total + margin) / 2`, `pred_away_score = (total − margin) / 2`. (The points
model's own margin is not used for the game line — Elo/SoS is the margin authority, per the
parent spec.)

## Market shrinkage (`shrink.py`)

The "sane early" mechanism. Blend the model estimate toward the current market line:

- `shrunk_margin = (1 − w_m(week)) · model_margin + w_m(week) · market_margin`
- `shrunk_total  = (1 − w_t(week)) · model_total  + w_t(week) · market_total`

`market_margin = spread_line` (home perspective, convention confirmed in Task 1);
`market_total = total_line`. Live, the market comes from the latest captured NFL odds
(`odds_snapshot`); in the backtest, from nflverse `spread_line`/`total_line`.

- **Weight curve** (parametric, separate for margin and total):
  `w(week) = floor + (start − floor) · exp(−decay · (week − 1))`, clamped to `[floor, start]`;
  playoff weeks (week > 18) clamp to `floor`. Starting shape from the parent spec (~0.75 Wk1 →
  ~0.35 Wk8 → floor ~0.2); `start`, `floor`, `decay` fit **separately for margin and total** by
  the backtest. If the market has no line for a game (early/live gap), `w = 0` (model-only
  fallback) — mirrors P1's cold-start-to-safe-default discipline.
- **API:** `w_curve(week, params) -> float`; `shrink(model_value, market_value, week, params) ->
  float` (returns `model_value` when `market_value is None`).

## Game-line orchestrator (`gameline.py`)

Given a game, P1 ratings, points ratings, the market line, and the week, produce the serving row:

1. `model_margin = ratings.expected_margin(...)` (P1); `model_total = points.expected_total(...)`.
2. `margin = shrink(model_margin, market_margin, week, w_m_params)`;
   `total = shrink(model_total, market_total, week, w_t_params)`.
3. **Distributions (Normal baseline):**
   - `margin_dist` = discretize `Normal(margin, σ_margin)` over an integer margin grid into the
     serving format `{"kind":"margin","offset":OFF,"pmf":[...]}`. **`OFF` sized for NFL** (margins
     to ~±75, so `OFF ≈ 75` and the pmf spans −75..+75) — larger than MLB's 25; the board/grade
     read `offset` from the dist, so this is self-describing.
   - `total_dist` = discretize `Normal(total, σ_total)` over integer totals into the serving pmf
     format (`{"kind":"pmf", ...}` as MLB totals use).
   - `win_prob = P(margin > 0)` from `margin_dist` (strict `>`, a 0 margin is a tie/push).
4. **Scores:** `pred_home_score = (total + margin)/2`, `pred_away_score = (total − margin)/2`,
   `pred_total = total`, `pred_margin = margin`.
5. Reuse the existing `sportsmodel` distribution helpers (`prob_over_dist`, `prob_cover`,
   `apply_affine`) that the board/grade already use, plus a small `normal_to_pmf`/
   `normal_to_margin_pmf` builder (add to the shared distributions module or `gameline.py`).
- **API:** `build_gameline(game, ratings, points_ratings, lg_avg, market, week, cfg) -> dict`
  returning the `game_predictions`-shaped fields (`margin_dist`, `total_dist`, `home_win_prob`,
  `pred_home_score`, `pred_away_score`, `pred_total`, `pred_margin`).

## Backtest + calibration (`scripts/backtest_nfl_gameline.py`)

Walk-forward over the committed schedules (2002+), using nflverse `spread_line`/`total_line` as
the market and `result`/`total` as the outcomes.

- **Leak-free:** ratings (Elo/SoS from P1, points from `points.py`) for game *i* use only prior
  games; the market line is the game's own closing line (known pre-game — it is the shrink
  target, not the outcome). Reuse P1's walk-forward discipline; **add a leak-regression test**
  (the P1 review's carried-forward item).
- **Fit σ_margin, σ_total:** to the empirical spread of `(actual − shrunk_prediction)`
  (method-of-moments) and/or by minimizing win-prob Brier + total-over/under log-loss; validate
  out-of-sample (train 2002–2019, validate 2020–2025), as in P1.
- **Fit the `w(week)` curves:** grid/coordinate search over `(start, floor, decay)` for margin
  and for total, minimizing out-of-sample margin MAE / total MAE (and win-prob Brier), on the
  train span, validated OOS.
- **Baselines to beat / bound:** market-only (`w=1`) and model-only (`w=0`). The blend must not
  be worse than model-only and should approach market-only early; **the honest expectation is
  that it lands between them** — NFL game lines are sharp, so beating the closing line outright
  is not a P2 gate (CLV is the season-long judge). Report all three, and use the same
  statistical-honesty framing as P1 (state SE / whether differences are within noise).
- **Sane-early check:** report `|shrunk − market|` by week; Week-1 deviations must be small
  (the acceptance bar).
- **Output:** committed `assets/nfl/gameline.json` (σ_margin, σ_total, the two `w(week)` param
  triples, `LG_AVG` handling, `K_points`, `OFF`) + a findings report under
  `docs/superpowers/reports/`.

## Acceptance / done bar (P2)

- The extended snapshot loads with the market columns; the `spread_line` convention is confirmed.
- `points.py` reproduces a hand-computed opponent-adjusted case in tests; ratings zero-mean;
  early-season shrinkage pulls a low-games team toward league average.
- `shrink.py`: `w(week)` decays start→floor and clamps; `shrink` returns model-only when the
  market is missing; blend endpoints (`w=0`→model, `w=1`→market) exact.
- `gameline.build_gameline` emits a valid serving row: `margin_dist`/`total_dist` are proper
  pmfs (sum≈1, correct `offset`/`kind`), `win_prob = P(margin>0)`, scores reconstruct
  `total`/`margin`. Consumable by the existing `prob_over_dist`/`prob_cover` unchanged.
- Backtest: σ's + `w(week)` fit and validated OOS; blend ≥ model-only on margin/total error;
  sane-early holds; win prob calibrated. Findings reported with statistical-honesty framing.
- **No MLB regression** — full suite green; NFL additive.

## Testing approach (TDD)

- `points.py`: hand-built opponent-adjusted case (a team that scored a lot vs strong defenses
  rates above one that scored the same vs weak defenses); zero-mean off/def; convergence;
  early-season shrinkage factor.
- `shrink.py`: `w_curve` monotone decay + clamp + playoff clamp; `shrink` endpoints and
  missing-market fallback.
- `gameline.py`: `build_gameline` output is a valid serving row (pmf sums, offset, kind,
  win_prob from margin_dist, score reconstruction); `w=1` reproduces the market line's implied
  margin/total; `normal_to_pmf` integrates to ~1 and centers on the mean.
- Distributions: `win_prob = P(margin>0)` matches `prob_cover(margin_dist, 0)` sign convention;
  NFL `offset` round-trips.
- Backtest: runs on a small season subset → metrics dict (margin MAE, total MAE, Brier, cover%,
  ou%); tune returns σ's + `w` params; determinism; **a leak-regression test**.
- Full MLB suite stays green.

## Risks

- **Sharp NFL game market** → the blend is expected to land between model and market, not beat
  the close; +EV is thin on game lines (props are the edge, P3). Frame honestly (P1 lesson).
- **Spread-line sign convention** → verified in Task 1 against `result` before use.
- **Normal misses key numbers (3/7)** → baseline only; win-prob and half-point cover are fine;
  key-number distribution flagged as a later refinement.
- **Early-season points-model instability** (few games) → the `K_points` empirical-Bayes
  shrinkage plus the market shrinkage both pull toward sane defaults; the backtest validates.
- **Playoff weeks / bye-week gaps** → `w` clamps to floor past week 18; missing market → `w=0`.
