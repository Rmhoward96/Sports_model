# NFL P1 — Data pipeline + margin-adjusted Elo + validation backtest

**Date:** 2026-08-24
**Status:** Approved design (pre-plan)
**Parent spec:** `docs/superpowers/specs/2026-08-23-nfl-model.md` (P1 phase)
**Depends on:** P0 (merged, commit c2abd2a) — `SportConfig` registry, sport-parameterized
producers, and the data-shape spike `docs/superpowers/reports/2026-08-23-nfl-data-spike.md`.

## Problem & goal

P0 built the multi-sport plumbing. P1 builds the **NFL data layer and the game-line rating
engine**: commit nflverse historical snapshots, implement a margin-adjusted Elo (with
last-season carryover) that produces team ratings and an expected margin per game, add the
ESPN live adapters (schedule / final scores / inactives) and the NFL `game_pk` matcher, and
**validate + tune the Elo** on a walk-forward backtest so its parameters are proven before P2
builds probability distributions on top of it.

The design philosophy (chosen earlier): predictions must be **sane early** (Week 1, no 2026
data) and grow independent as data accumulates. P1 delivers the rating engine and its tuned
parameters; P2 adds the market-shrinkage blend and the distributions that make it bettable.

## Scope / non-goals

**In (P1):**
- nflverse ingest wrappers + committed parquet snapshots (schedules 2002+; weekly / rosters /
  injuries 2015+).
- Team-abbreviation normalization (single source of truth).
- Margin-adjusted Elo engine: ratings history, final ratings, preseason carryover, expected
  margin. Tunable `K`, `HFA`, `carryover`.
- ESPN public-API adapters: weekly schedule (upcoming games + kickoff + event id), final
  scores, game-day inactives — parsing the exact fields the P0 spike verified.
- NFL `game_pk` matcher: `game_pk` = ESPN event id; reconcile nflverse (`schedules.espn`) and
  The-Odds-API events (team-name + kickoff date).
- Walk-forward Elo backtest + parameter tuning (train/validate split) + an acceptance bar.

**Out (later phases):**
- Win-probability / margin **distributions**, market-shrinkage weight `w(week)`, and σ tuning
  → **P2**.
- Prop volume/efficiency modeling (uses the weekly snapshots) → **P3**.
- The `generate_nfl` producer, NFL board/grade passes, front-end NFL logos, and wiring the
  three P0-parked items (`commence_shift_hours` threading, the results-provider
  `home_score/away_score` contract, the NULL-sport grade filter) → **P4**.
- Changing anything in the MLB pipeline. NFL is strictly additive.

## Package & assets

New package `src/sportsmodel/nfl/`:

- `data.py` — wrappers over `nfl_data_py`: `load_schedules(seasons)`, `load_weekly(seasons)`,
  `load_rosters(seasons)`, `load_injuries(seasons)`. Each returns a normalized DataFrame
  (team columns run through `normalize_team`). The nflverse team column is `recent_team`
  (weekly) / `team` (rosters) / `home_team`,`away_team` (schedules) — normalized to a single
  abbreviation vocabulary.
- `teams.py` — `normalize_team(abbr) -> str` and the canonical 32-team set. Known
  normalizations from the spike: `LAR→LA`, `WSH→WAS`. This is the single source of truth used
  by data ingest, the ESPN adapter, and the matcher.
- `elo.py` — the rating engine (see **Elo model**).
- `espn.py` — live adapters (see **ESPN adapters**).
- `matcher.py` — the `game_pk` matcher (see **Matcher**).

Committed snapshots under `assets/nfl/` (mirrors the MLB `assets/profiles` pattern, so CI runs
without live pulls):
- `schedules.parquet` — seasons **2002–current** (32-team era; stable Elo team set). Columns
  kept: `game_id`, `season`, `week`, `gameday`, `gametime`, `home_team`, `away_team`,
  `home_score`, `away_score`, `espn` (ESPN event id), plus `game_type` (REG/POST) so the
  backtest can scope to regular season if desired.
- `weekly.parquet`, `rosters.parquet`, `injuries.parquet` — seasons **2015–current** (for P3;
  ingested now so the data layer is built once).
- Built by `scripts/build_nfl_snapshots.py`; regenerated weekly in-season by a
  `refresh-nfl.yml` GitHub Action (analogous to `refresh-profiles.yml`). Snapshots are
  committed so the daily/live path and CI never call nflverse.

## Elo model (`elo.py`)

FiveThirtyEight-style margin-adjusted Elo. Starting parameter values below are the backtest's
initial point; the tuning step (see **Backtest**) sets the shipped values.

- **State:** a rating per team, base **1500**.
- **Expected home win prob:**
  `E_home = 1 / (1 + 10^(-(elo_home + HFA_elo - elo_away)/400))`.
- **Margin-of-victory multiplier** (dampens favorite blowouts, 538 autocorrelation form):
  `mov_mult = ln(|margin| + 1) * (2.2 / (0.001 * elo_diff_winner + 2.2))`,
  where `elo_diff_winner` = (winner pregame rating incl. HFA) − (loser pregame rating incl.
  HFA), from the **winner's** perspective.
- **Update** (zero-sum; home gains what away loses):
  `elo_home += K * mov_mult * (result_home − E_home)`, `result_home ∈ {1, 0.5, 0}`.
- **Preseason carryover:** at each new season start,
  `elo_start = 1500 + carryover * (elo_prev_final − 1500)` (regress toward the mean; `carryover
  ≈ 0.75` ⇒ 25% reversion). New/relocated franchises with no prior season start at 1500.
- **Expected margin (points), for P2/producer consumption:**
  `expected_margin = (elo_home + HFA_elo − elo_away) / 25` (25 Elo ≈ 1 point).
- **Starting params (tuned by the backtest):** `K ≈ 20`, `HFA_elo ≈ 65` (~+2.5 pts),
  `carryover ≈ 0.75`.
- **API:**
  - `EloConfig(k, hfa_elo, carryover, base=1500)`.
  - `run_elo(schedule_df, config) -> EloResult` where `EloResult` exposes the **pre-game**
    home/away ratings and `E_home` for every game (for backtesting) and the **final** ratings
    per team (for carryover / live prediction).
  - `expected_margin(elo_home, elo_away, config) -> float`.
  - Distributions and win-prob-at-a-line are **NOT** in P1 (P2 wraps `expected_margin` in a
    Normal and discretizes it).

## ESPN adapters (`espn.py`)

Public API `https://site.api.espn.com/apis/site/v2/sports/football/nfl/...`. Fields confirmed
by the P0 spike.

- `fetch_schedule(season, week, season_type=2) -> list[dict]`: uses the query pattern the spike
  found (`?dates=<season>&seasontype=<n>&week=<n>`), returning per game: `game_pk` (event
  `id`, int), `commence_time` (event `date`), normalized `home_team`/`away_team`, `status`.
- `fetch_final(event_id) -> dict | None`: `home_score`/`away_score` (ESPN returns them as
  strings — cast to int) and a `final` bool gated on `status.type.name == "STATUS_FINAL"`;
  returns `None` if not final.
- `fetch_inactives(event_id) -> list[str]`: game-day inactive player names (may be empty
  before game day — return `[]` gracefully; used by P3/P4 to suppress props).
- All adapters normalize team strings via `teams.normalize_team` so ESPN, nflverse, and
  Odds-API share one vocabulary.

## Matcher (`matcher.py`)

- `game_pk` = **ESPN event id** (integer), the canonical key across the pipeline.
- Historical: nflverse `schedules.espn` already equals the ESPN event id (spike-verified) → the
  snapshot carries `game_pk` directly.
- Live: `match_odds_event(odds_event, espn_games) -> game_pk | None` matches a The-Odds-API
  event to an ESPN game by **(normalized home, normalized away, kickoff date)**. The exact
  Odds-API NFL team-name strings are the one P0-deferred unknown — verified at the first CI
  odds capture; until then the matcher normalizes via `teams.normalize_team` and the design
  assumes full-name→abbr mapping, adjusted when the first live payload lands.

## Validation backtest (`scripts/backtest_nfl_elo.py`)

- **Walk-forward** over the committed schedules (2002+): season by season, week by week, predict
  each game from **pre-game** Elo (carryover applied at season boundaries), then update.
- **Metrics** (no historical NFL odds exist — this project is capture-forward — so CLV/ATS is
  not a P1 metric): straight-up **win accuracy**, **Brier score**, and **margin MAE/RMSE** vs.
  actual, plus a **calibration/reliability** check on `E_home`.
- **Baselines** to beat: home-team-always, and prior-season win% pick.
- **Tuning:** coordinate/grid search over `(K, HFA_elo, carryover)` on a **train span**
  (2002–2019), **validated out-of-sample** on a **held-out span** (2020–2025). The shipped
  params are the ones that win on the validation span (guarding against in-sample overfit — the
  same discipline documented for the MLB model).
- **Output:** the tuned `EloConfig` values recorded (committed as an `assets/nfl/elo.json` or
  as documented constants in `elo.py`) + a short findings report under
  `docs/superpowers/reports/`.

## Acceptance / done bar (P1)

- Committed nflverse snapshots exist and load with the expected columns; team names normalized.
- `run_elo` reproduces a hand-computed rating sequence in tests; carryover regresses toward
  1500; expected margin sign/scale is correct.
- ESPN adapters parse **recorded JSON fixtures** into the documented shapes (no live calls in
  the test suite).
- The matcher keys on ESPN event id and resolves an Odds-API event by normalized names + date,
  including a relocation case (`LAR/LA`).
- The Elo backtest **beats both naive baselines** on Brier + margin MAE and **generalizes
  out-of-sample** (validation span ≈ train span). Beating the market/CLV is a season-long,
  post-launch judge, not a P1 gate.
- **No regression to MLB** — the full existing suite stays green; NFL is additive.

## Testing approach (TDD)

- `elo.py`: deterministic unit tests — a two-game hand-computed sequence → known ratings;
  MOV-multiplier monotonicity in `|margin|`; carryover pulls a 1700 team toward 1500 by the
  configured fraction; `expected_margin` sign and ~1pt/25-Elo scale.
- `teams.py`: normalization map (`LAR→LA`, `WSH→WAS`), idempotence, the 32-team set.
- `matcher.py`: ESPN event id as `game_pk`; Odds-API event resolves by normalized names+date; a
  relocation/abbr case; a no-match returns `None`.
- `espn.py`: parse committed fixture JSON for schedule/final/inactives; `STATUS_FINAL` gating;
  string-score casting; empty-inactives.
- `data.py`: snapshot builder writes the expected columns; normalization applied; season
  windows respected.
- Backtest: runs on a small season subset producing a metrics dict; the tuning search returns a
  best `EloConfig`; determinism (fixed input → fixed metrics).

## Risks

- **No historical NFL odds** → P1 tunes raw Elo accuracy (wins/margins), not market edge; the
  market-shrinkage that makes Week-1 sane is P2. Acceptable — P1's job is a well-tuned rating,
  not a bettable line.
- **Odds-API NFL team-name strings unverified** (P0 deferral) → the matcher's live path is
  finalized when the first capture lands; historical matching (via `schedules.espn`) is
  unaffected.
- **nflverse team-abbreviation drift across eras** → centralized `normalize_team`; 2002+ is
  stable for the 32-team set.
- **ESPN endpoint shape changes** → adapters isolated in `espn.py` with fixture tests; a change
  is contained to one module.
