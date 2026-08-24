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
- **Explicit strength-of-schedule rating** (SRS/Massey retrodictive, opponent-adjusted) blended
  with Elo, so beating strong opponents is rewarded beyond Elo's per-game adjustment; blend
  weight + cold-start threshold tuned in the backtest.
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
- `elo.py` — the sequential Elo rating engine (see **Elo model**).
- `srs.py` — the retrodictive strength-of-schedule rating (see **Strength-of-schedule rating &
  blend**).
- `ratings.py` — the combiner: blends the Elo and SRS expected margins into the single
  `expected_margin` the backtest / P2 / producer consume.
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
- **Elo expected margin (points):**
  `elo_expected_margin = (elo_home + HFA_elo − elo_away) / 25` (25 Elo ≈ 1 point). This is
  Elo's contribution to the final blended margin (see the SoS section).
- **Starting params (tuned by the backtest):** `K ≈ 20`, `HFA_elo ≈ 65` (~+2.5 pts),
  `carryover ≈ 0.75`.
- **API:**
  - `EloConfig(k, hfa_elo, carryover, base=1500)`.
  - `run_elo(schedule_df, config) -> EloResult` where `EloResult` exposes the **pre-game**
    home/away ratings and `E_home` for every game (for backtesting) and the **final** ratings
    per team (for carryover / live prediction).
  - `elo_expected_margin(elo_home, elo_away, config) -> float`.
  - Distributions and win-prob-at-a-line are **NOT** in P1 (P2 wraps the blended
    `expected_margin` in a Normal and discretizes it).

## Strength-of-schedule rating & blend (`srs.py`, `ratings.py`)

Elo already rewards beating stronger opponents per game (the `result − E_home` term is large
when a low-`E` underdog wins, tiny when a favorite beats a weak team; the MOV multiplier's
autocorrelation term reinforces this). The explicit SoS rating adds a **retrodictive, whole-
schedule** view that Elo's sequential update reflects only slowly — so a team that beat a
murderers'-row schedule is rated above a same-record team that beat cupcakes.

- **SRS (Simple Rating System / Massey-style), `srs.py`:** over the games played so far in the
  season, solve `rating_i = avg_point_margin_i + avg_opponent_rating_i` to a fixed point
  (iterate to convergence; ratings are zero-mean, in **points**). A team's rating rises for
  beating high-rated opponents and barely moves for beating low-rated ones — SoS is baked into
  the solve. `compute_srs(games_so_far) -> dict[team, points]`.
  - `srs_expected_margin = srs_home − srs_away + HFA_points` (`HFA_points = HFA_elo / 25`).
- **Blend (`ratings.py`):** the single margin the rest of the pipeline consumes is
  `expected_margin = (1 − w) * elo_expected_margin + w * srs_expected_margin`.
  - **Cold-start fallback:** SRS needs a minimum sample. When either team has played fewer than
    `srs_min_games` games this season, `w = 0` (pure Elo, which carries prior-season signal via
    carryover). This keeps **Week 1 sane** — Elo's carryover drives it; SRS phases in only once
    there is enough current-season schedule to measure.
  - **Blend weight `w`:** tuned in the backtest (see below). Hypothesis to test: `w` is
    meaningful in the early-to-mid season where records diverge but Elo hasn't converged, and
    can decay later as Elo catches up; the backtest decides whether a flat `w` or a
    week-decaying `w(week)` scores better, and its magnitude. Bounded (e.g. `w ≤ ~0.5`) so Elo
    stays the backbone.
  - `expected_margin(elo_result, srs_ratings, home, away, week, config) -> float` is the P1
    engine's single output for a game (what P2 will wrap in a distribution).

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
- **Tuning:** coordinate/grid search over `(K, HFA_elo, carryover, w_sos, srs_min_games)` on a
  **train span** (2002–2019), **validated out-of-sample** on a **held-out span** (2020–2025).
  The shipped params are the ones that win on the validation span (guarding against in-sample
  overfit — the same discipline documented for the MLB model). The search **must compare the
  blend against pure Elo** (`w_sos = 0`) so the SoS component only ships if it demonstrably
  improves the validation metrics; if it doesn't, `w_sos` tunes to 0 and Elo stands alone.
- **Output:** the tuned rating parameters recorded (committed as `assets/nfl/rating.json`
  holding the `EloConfig` fields + the blend `w_sos`/`srs_min_games`, or as documented
  constants) + a short findings report under `docs/superpowers/reports/` stating whether the
  SoS blend beat pure Elo out-of-sample and by how much.

## Acceptance / done bar (P1)

- Committed nflverse snapshots exist and load with the expected columns; team names normalized.
- `run_elo` reproduces a hand-computed rating sequence in tests; carryover regresses toward
  1500; Elo expected margin sign/scale is correct.
- SRS ranks a team that beat strong opponents above a same-record team that beat weak ones; the
  blend falls back to pure Elo below `srs_min_games` (Week-1 sanity), and `ratings.expected_margin`
  returns Elo's margin when SRS is unavailable.
- ESPN adapters parse **recorded JSON fixtures** into the documented shapes (no live calls in
  the test suite).
- The matcher keys on ESPN event id and resolves an Odds-API event by normalized names + date,
  including a relocation case (`LAR/LA`).
- The rating backtest **beats both naive baselines** on Brier + margin MAE and **generalizes
  out-of-sample** (validation span ≈ train span). The SoS blend is shipped only if it beats
  pure Elo (`w_sos = 0`) out-of-sample; otherwise `w_sos` tunes to 0. Beating the market/CLV is
  a season-long, post-launch judge, not a P1 gate.
- **No regression to MLB** — the full existing suite stays green; NFL is additive.

## Testing approach (TDD)

- `elo.py`: deterministic unit tests — a two-game hand-computed sequence → known ratings;
  MOV-multiplier monotonicity in `|margin|`; carryover pulls a 1700 team toward 1500 by the
  configured fraction; `elo_expected_margin` sign and ~1pt/25-Elo scale.
- `srs.py`: a small hand-built round-robin where the retrodictive solve converges to known
  ratings; a team beating strong opponents outranks a same-record team beating weak ones;
  ratings are zero-mean.
- `ratings.py`: the blend equals `(1−w)·elo + w·srs` when SRS is available; falls back to pure
  Elo below `srs_min_games`; `w=0` reproduces Elo exactly.
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
- **SoS blend could overfit or add noise** → the backtest tunes it against pure Elo (`w_sos=0`)
  on a held-out span and ships it only if it beats Elo out-of-sample; the weight is bounded and
  falls back to pure Elo before `srs_min_games`, so a weak SoS signal degrades gracefully to the
  Elo baseline rather than hurting it.
