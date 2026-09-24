# NFL Props ML — learned sim inputs + per-market distribution models, blended and calibrated (design spec)

Date: 2026-09-24 · Status: approved design, pending spec review · Sport: NFL (CFB has no sim — out of scope)

## Goal

Make the NFL sim's player-prop distributions more accurate and more honestly
calibrated by bringing machine learning in where the evidence says it can help:
player/team **volume** (the dominant prop error source), **game context** (rest,
travel, weather), optionally **market-anchored game environment**, and a
**calibration** layer the sim does not have today. Then (Phase 2) add the same
context to the game win/margin/total predictions.

Every component ships only if it beats the current sim out-of-sample (the gate,
§4). A component that fails is skipped; if everything fails, the current sim stays.

## What already exists (this extends, it does not reinvent)

- Walk-forward validation everywhere (no random splits); the sim uses prior +
  current season with `SEASON_DECAY=0.4` (validated sweep, 2026-09-22).
- Opponent adjustment: opponent-adjusted EPA (`nfl/efficiency.py`), sim defense
  adjustment, Elo ratings tilt (`SIM_RATINGS_WEIGHT=0.5`).
- Gradient boosting for win/ATS/O-U: the cover/total stacked ensemble
  (2026-09-21, re-gated 2026-09-22) — **a trustworthy negative**: walk-forward
  2015–2025 Brier 0.2505 vs coin-flip 0.2500. Not re-litigated here.
- Platt calibration infra (`model/calibration.py`, `fit_calibration_live.py`).
- The drive-based Monte Carlo sim with `usage.active_usage` (active set,
  Out/Doubtful dropped, Questionable volume ×0.75), B.3 Poisson attempt volume,
  B.4 projected-usage evaluation lens.
- +EV prop board, prop line capture (`nfl_prop_lines`) and grading
  (`nfl_prop_grades` — 0 graded rows on 2026-09-24; live samples are future).

## Pushback recorded (from the idea list the user supplied)

- **X,Y tracking / StatsBomb / Opta xG:** soccer providers; raw NFL tracking is
  not publicly sold. Substitute: nflverse Next Gen Stats weekly aggregates + FTN
  charting (free).
- **Transformers / 1D CNNs:** ~285 games/season, no public tracking sequences —
  a deep net would fit noise. Gradient boosting is the ceiling for this data.
- **XGBoost/LightGBM for win/ATS/O-U:** already built (HistGradientBoosting ≈
  LightGBM); failed for feature reasons, not algorithm reasons.
- **Isotonic calibration:** used only where n is in the thousands (props);
  game-level probabilities stay Platt.
- **"Exactly 70 of 100":** judged with reliability curves + confidence bands,
  not point counts (n=100 carries ±9 pts of noise).

## Decisions locked (user, 2026-09-24)

1. Scope: **both, props first** — props (Props-1, Props-2), then game context (Phase 2).
2. Market spread/total as prop-sim input: **let the backtest decide** (an on/off
   rung of the ladder; historical closing lines exist in schedules back to 2002).
3. Approach: **C — blend** A (learned inputs → sim) with B (direct per-market
   distribution models); blend weights set walk-forward (B's weight may be 0).
4. Large ML files (feature table, model binaries) live in a **GitHub Release**,
   not git.

## 1. Architecture

```
nflverse (pbp, weekly, snaps, NGS, FTN) + schedules (rest, roof, weather, closing lines)
        │   + Open-Meteo forecast for upcoming games
        ▼
Player-week feature table (leakage-safe, built weekly)
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
 A. Learned sim inputs         B. Per-market distribution models
   → Monte Carlo sim             → full distribution per player-market
   ▼                             ▼
 A marginals ── blend (per-market w, walk-forward) ──┐
                                                     ▼
                         Calibration (per market, isotonic on PIT; TD Platt)
                                                     ▼
          Final marginal, quantile-mapped onto the sim's draws
          (single props use it; parlays keep the sim's correlations)
```

## 2. Player-week feature table

**Unit:** one row per player-game, QB/RB/WR/TE on the active depth chart,
2016–2026 (~6.5k rows/season, ~65k total).

**Targets.** Volume: targets, carries, pass attempts (QB), team pass/rush
attempts. Outcomes: rec_yds, rush_yds, receptions, pass_yds, rush_att,
pass_tds, anytime_td.

**Features** — rolling mean + EWMA over the last 3/5/10 games, shrunk toward the
player's prior season early in the year (shrink weight by games played):

| Group | Features |
|---|---|
| Usage | snap %, route participation (FTN, 2022+), target share, air-yards share, carry share, red-zone + goal-line share, depth-chart rank |
| Efficiency | yards/route run, yards/target, catch rate, yards/carry, YAC |
| NGS (2016+) | separation, cushion, intended air yards, xYAC over expected, RYOE/att; QB: time to throw, CPOE, aggressiveness |
| Team | plays/game, seconds/play, neutral-situation pass rate over expected, opponent-adjusted offensive EPA |
| Opponent | opponent-adjusted defensive EPA vs pass/run, yards allowed per target to the player's position (shrunk), pressure rate, pace allowed |
| Status | injury designation, QB change since last game, vacated share (targets/carries freed by teammates ruled Out), games since return, rookie flag |
| Context | rest-day diff, short week, off bye, travel distance, time zones crossed, West-coast team in a 1 PM ET game, roof, surface, temp, wind, precipitation, division game, home/away |
| Market (gated rung) | spread, game total, implied team total |

**Leakage rules (tested, §4):**
1. Rolling features for week *w* use only games strictly before *w*; nothing from
   the target game's box score.
2. Pre-game information for week *w* is allowed: that week's injury report,
   depth chart, and the pre-kickoff line.
3. A traded player keeps his own usage history; team/opponent features switch
   to the new team.
4. Features absent for an era (FTN pre-2022) stay NaN — HistGradientBoosting
   handles NaN natively; no imputation.

**Known train/serve differences (stated, not hidden):** weather is observed
historically but a forecast live; market lines are closing historically but
at-run-time live. Both small; the injury watch re-runs near kickoff.

**Build:** `scripts/build_player_features.py` → `player_week_features.parquet`
(Release asset). The upcoming week's rows come from the **same builder
functions** at sim time, so train and serve can't drift.

## 3. Models (scikit-learn HistGradientBoosting — LightGBM-family)

Hyperparameters (depth, learning rate, iterations) from a small inner
walk-forward with early stopping. Monotonic constraints where the direction is
certain (e.g. targets non-decreasing in target-share EWMA).

### A. Learned inputs for the sim (Monte Carlo engine unchanged)

| Current sim input | Replaced by |
|---|---|
| Team pass/rush attempts/game (season rates × game_env) | Team volume models (Poisson loss): pass att, rush att — team/opponent/context (+ market if earned) |
| Player shares (weighted last-5, renormalized over active set) | Player targets / carries models (Poisson) → converted to shares, renormalized over the active set. `active_usage` keeps its structure (Out/Doubtful drop, Questionable ×0.75); only the share source changes |
| Per-touch efficiency (shrunk averages) | Learned efficiency means (ypt, ypc, catch rate) — **its own gated rung** |
| game_env / score tilt | Unchanged unless the market rung passes, then implied team total drives team volume + tilt |

After A, re-tune `_USAGE_CONCENTRATION` once (as in B.2).

### B. Per-market distribution models (bypass the sim)

| Market type | Model |
|---|---|
| Yards (rec_yds, rush_yds, pass_yds) | Quantile HGB at 19 quantiles (0.05…0.95), rearranged monotone, interpolated to a CDF |
| Counts (receptions, rush_att, pass_tds) | Poisson HGB mean + negative-binomial dispersion fit per market × usage tier |
| anytime_td | HGB classifier (log-loss) |

### Blend

Per market: `F = w·F_A + (1−w)·F_B`, `w ∈ {0, 0.1, …, 1}`, chosen for test
season S using only seasons < S. anytime_td: weighted average on the logit scale.

### Calibration

Per market, an isotonic map on held-out PIT values (`u = F(actual)`),
`F' = G∘F`. anytime_td: Platt. Later, once live graded props reach ~500 per
market, add the MLB-style live Platt refit on P(over line), shrunk toward
identity (`s = n/(n+75)`) and slope-clamped, run by the training workflow.

### Quantile mapping onto the sim

Per player-market, rank the sim draws (random jitter within ties for counts) and
replace each draw with `F'^{-1}(rank/(n+1))`. Rank correlation across players and
teams is preserved, so parlays, team totals and cross-player correlation keep
working. **Accepted cost:** receivers' yards no longer sum exactly to the QB's
pass_yds after mapping (they remain rank-correlated).

### Output

Same `prop_predictions` dist format under `model_version = 'nfl-sim-ml-v1'`.
No schema or site change. Blend weights + calibration maps are versioned with the
artifacts for traceability.

## 4. Validation and ship gate

**Walk-forward.** Test seasons 2021–2025. For season S, train on 2016…S−1 plus
in-season weeks before the refit point; refit before weeks 1, 5, 9, 13, 17
(production refits weekly). One season is also run with weekly refits to confirm
the every-4-weeks shortcut doesn't bias results.

**Nested tuning.** Hyperparameters, blend `w`, season down-weighting
(`{1.0, 0.8, 0.6}` per season back), and market on/off are chosen using seasons
< S only. Calibration maps are fit on out-of-fold PIT from seasons < S.

**Population.** B.4 projected-usage lens (projected ≥3 targets, ≥5 carries, or
QB starter), **fixed by the current sim's projections** so no candidate can
improve its score by changing who is scored.

**Metrics per market:** CRPS (primary; RPS for counts); PIT coverage at
p10/p50/p90 + decile ECE; mean MAE + bias; Brier of P(over a proxy line = the
current sim's median rounded to .5) as the betting-relevant check. anytime_td:
log-loss, Brier, reliability curve.

**Significance.** Paired bootstrap over game-weeks (players in one game are
correlated); an improvement counts only if the 95% CI excludes zero.

**Ablation ladder** (a failed rung is skipped, the ladder continues):

```
current sim → +A volume → +A efficiency → +context → +market env → +B blend → +calibration
```

A rung passes iff: pooled CRPS improves (CI excludes 0) **and** no market's CRPS
worsens > 1% **and** PIT calibration does not worsen.

**Final gate:** the assembled pipeline beats the current sim on pooled
2021–2025 CRPS **and** on 2025 alone (era-drift guard).

**Leakage tests** (ported from `train_cover_ensemble`): features for week *w* are
identical with and without post-*w* data present; a planted leaky feature is
caught.

**Rollout.** (1) Shadow ≥1 week: v1 is written alongside the current sim; the
board keeps the current sim; compare against real lines in `nfl_prop_grades`
(a bug check — one week is not significance). (2) Switch the board to v1 if no
regression beyond noise. (3) Weekly retrain must pass the quick gate (§5) to
replace artifacts; on failure the last passing model stays and the run is
flagged red — never a silent fallback.

## 5. Automation and files

**Artifacts.** GitHub Release `props-ml-latest` (feature table, model binaries);
the previous one is kept as `props-ml-prev` for one-step rollback. Published with
the workflow's `GITHUB_TOKEN` — no new secret. Small JSON (config, blend
weights, calibration maps, gate reports) is committed to git.

**Workflows.**

| Workflow | When | What |
|---|---|---|
| `train-props-ml.yml` (new) | Tue 14:00 UTC weekly | Build features → retrain with tuned config → quick gate (last 2 weeks scored by a model trained without them vs the current sim; feature-coverage + prediction sanity checks) → publish Release only on pass |
| same, monthly (first Tuesday) | monthly | Full walk-forward ladder 2021–2025 → re-tune → report under `docs/superpowers/reports/` |
| `generate-sim-nfl`, `injury-watch`, `desk-auto` | unchanged | `generate_sim_nfl.py` downloads the Release, builds upcoming features (Open-Meteo forecast), runs per repo variable `SIM_ML_MODE = off \| shadow \| live` (starts `shadow`). Missing/unloadable artifacts → current sim, logged loudly |

**Code.**
- `nfl/nflverse.py` — add NGS + FTN datasets to `load_release`; validate
  expected columns and fail loudly (lesson of the depth-chart schema break).
- `nfl/context.py` — rest, travel/time zones, stadium coordinates + roof,
  weather (history + Open-Meteo forecast).
- `nfl/player_features.py` — pure feature builders (§2), shared by backtest and live.
- `sim/nfl/learned.py` — A's volume/efficiency models into `active_usage` and team rates.
- `model/props_ml/` — `dist_models.py` (B), `blend.py`, `pit_calibration.py`, `quantile_map.py`.
- `scripts/build_player_features.py`, `scripts/train_props_ml.py` (walk-forward, ladder, gate, publish).
- No new dependencies (scikit-learn, joblib present).

## 6. Phase 2 — game predictions

Context (rest, short week, travel/time zones, wind, temp, precipitation, roof) as
a **ridge-regression correction** to the current model's margin and total, fit
walk-forward on the model's own residuals, 2016–2025 (≈270 games/season — a
regularized linear model, not trees). Gate: margin MAE, total MAE, win Brier with
bootstrap CIs. Expectation: wind on totals is the real effect (1–3 pts in high
wind); rest/travel ~0.5 pt. Improves prediction accuracy; does not create an ATS
edge. The sim receives the same context through A.

## Build order (each plan ships on its own)

1. **Props-1** — feature table, context, leakage tests, walk-forward ladder
   harness, A volume + efficiency. Deliverable: first gate verdict for A.
2. **Props-2** — B models, blend, calibration, quantile mapping, shadow serving,
   workflows + Release artifacts. Deliverable: `nfl-sim-ml-v1` shadow → live.
3. **Game context** — Phase 2 ridge correction.

## Risks

- nflverse schema change or late release → loaders fail loudly; quick gate flags
  coverage drops.
- Gate returns a negative for some markets → those markets stay on the current
  sim, reported plainly (as with the cover ensemble).
- Train/serve mismatch on weather and lines (stated above).

## Non-goals

CFB; deep learning; paid tracking data; re-attempting the ATS/O-U edge with
team-quality features.
