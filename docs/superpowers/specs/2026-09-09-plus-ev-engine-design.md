# +EV Engine (Pinnacle sharp vs. true probability) — system design

**Status:** in review (2026-09-09)

**Goal:** A new **+EV view**, toggled against the existing prediction pages for
NFL and CFB, that finds **+EV bets by the probability gap** — the difference
between an **estimated true probability** and the market's **implied
probability**. True probability comes from a feature-rich model (Elo, L10 win %,
rest days, offensive/defensive EPA/PPA, strength of schedule, strength of
victory, +more) **with the decision desk factored directly into the estimate**
(e.g. the model computes 47%, the desk surfaces news that moves it to 51% — 51%
is the true probability used). Implied probability comes from the **Pinnacle
line** — used as the cleanest *baseline* (sharp, low-vig), not an adversary to
beat. The view surfaces the **gap and EV**, with EV shown both at the Pinnacle
price and at the best soft book. Moneyline, spread, and total, both sports.
Disciplined and CLV-gated: calibrated true-prob, forward grading vs the Pinnacle
close, honest "pass / no edge" when the gap is small.

## Non-goals / constraints

- **The prediction pages and prediction model are unchanged.** The true-prob
  model is a *separate* model that powers only the +EV view. Predictions keep
  reading `predictions_current`; nothing in `generate_nfl.py` / `generate_cfb.py`
  or the prediction UI changes.
- **Gaps must be real, not artifacts.** The engine surfaces a bet when true
  probability and implied probability diverge enough to be +EV. The discipline
  (calibration so a stated 56% really hits ~56%; `EV_CEILING`; forward CLV vs the
  Pinnacle close) exists so those gaps reflect genuine information — the desk's
  news, a rest/EPA edge the price hasn't absorbed — rather than a mis-calibrated
  model or a stale line. It is not a promise of profit: some slates will show few
  or no qualifying gaps, and that "pass" is a valid, honest output, not a
  failure.
- Secrets (`ODDS_API_KEY`, `CFBD_API_KEY`, `DATABASE_URL`) are GitHub Actions
  secrets, read only in scripts' `main()`; never in the session. The user runs
  all Supabase SQL. `app.js` is external (`/Users/ryan/Desktop/CappingAlpha`),
  redeployed by the user.

## What already exists and is reused

- `src/sportsmodel/serving/board.py` — `implied_prob`, `novig`, `ev`,
  `best_price` (MAJOR_BOOKS already includes `pinnacle`), `EV_CEILING`, and the
  `moneyline_row` / `spread_row` / `total_row` builders (model_prob vs
  implied_prob vs EV). Built for MLB; reused here.
- `src/sportsmodel/model/distributions.py` — `normal_to_margin_pmf`,
  `normal_to_pmf`, `prob_cover`, `prob_over_dist`, `apply_affine`. The true-prob
  model wraps its margin/total point estimates + σ in these to get
  P(win)/P(cover)/P(over).
- `src/sportsmodel/model/odds.py` — `implied_prob`, `no_vig_two_way`, CLV.
- `src/sportsmodel/model/calibration.py` — `calibrate` (Platt/isotonic).
- `nfl/gameline.py` — reference pattern (margin_dist/total_dist/win_prob with
  σ_margin≈13.2, σ_total≈10). CFB has **no** gameline module; the new true-prob
  model supplies margin+σ / total+σ uniformly for both sports.
- CFBD adapter (`cfb/cfbd.py`) already hits **PPA** (`percentPPA`,
  `percentPassingPPA`) — CFB's EPA analog.
- Elo/SRS/points ratings (`nfl/elo.py`, `srs.py`, `points.py`, `ratings.py`),
  schedules parquet (dates → rest, scores → L10/SOV, + ratings → SOS).
- The decision desk (`desk_current`, `desk_picks`) — conviction-tiered ML/
  spread/total picks per game — is the desk-influence input.

## Architecture / data flow

```
The Odds API (regions=us,eu; Pinnacle in eu)  ─┐
                                               ├─> pinnacle_odds + book_odds
feature pipeline (Elo,L10,rest,EPA/PPA,SOS,SOV)┘         │
        │                                               │
        ▼                                               │
  true-prob model  ── margin+σ, total+σ ──> distributions ──> P(win/cover/over)
        │  (+ desk nudge, clamped)                       │
        ▼                                                ▼
   edge/EV engine (reuse board.py): per market/side compute
     pinnacle_novig, true_prob, edge = true−pinnacle_novig,
     ev_pinnacle (at Pinnacle price), ev_best (best soft book)
        │
        ▼
   ev_board  ──> ev_current view ──> +EV page (toggle vs Predictions)
        │
        ▼
   forward grading vs Pinnacle close ──> ev_results (CLV + calibration)
```

## Sub-project 1 — Odds ingestion (Pinnacle + soft books)

- Source: **The Odds API** v4, `sports=americanfootball_nfl` and
  `americanfootball_ncaaf`, `markets=h2h,spreads,totals`, `regions=us,eu`
  (Pinnacle is an **eu** bookmaker — `us` alone misses it), `oddsFormat=american`.
- Store, per (sport, game, market, side, line, book): american price +
  `captured_at`. Two consumers: **Pinnacle** rows (the sharp baseline) and the
  **all-book** rows (best-soft-book EV via `board.py::best_price`). A separate
  **closing snapshot** near kickoff feeds CLV grading (sub-project 6).
- Join key: The Odds API event → our `game_pk` (ESPN event id) via team-name +
  commence-time matching (mirror the desk's `_rekey_by_espn_name` approach).
- `ODDS_API_KEY` secret; pulled in a CI workflow (like `desk-inputs`). Non-fatal
  per game. Note API credit cost scales with sports×regions×markets — modest.

## Sub-project 2 — Feature pipeline (incl. EPA/PPA)

Per upcoming game (and historically, for fitting): home/away values + their
differentials for —
- **Elo** — pre-game rating (existing engine).
- **L10 win %** — last-10 from schedules.
- **Rest days** — days since each team's previous game (schedule dates); bye flag.
- **Off/Def EPA** — **NFL:** real EPA/play from **nflverse/nflfastR** play-by-play
  (new ingest: rolling season-to-date offensive & defensive EPA/play).
  **CFB:** **CFBD PPA** (`/ppa/teams`) offensive/defensive — already integrated.
- **Strength of schedule** — average opponent rating faced to date.
- **Strength of victory** — average rating of teams beaten to date.
- Home field; optionally neutral-site flag, pace.

Output: a per-game feature table (upcoming for scoring, historical for fitting).
Missing features degrade gracefully (imputed to league mean, flagged), never crash.

## Sub-project 3 — True-probability model (the hard core)

- **Two fitted models:** a **margin model** (home margin ~ features) with a
  calibrated residual **σ_margin**, and a **total model** (total ~ features,
  **σ_total**). Start simple and honest (regularized linear / GLM) before
  anything fancier — interpretability + not overfitting a thin sample matters
  more than squeezing R².
- **Probabilities** via the existing distributions: P(win)=`prob_cover(N(margin,σ),0)`,
  P(cover @ line), P(over) from the total distribution.
- **Calibration:** Platt/isotonic (`model/calibration.py`) so a stated 56% hits
  ~56%; validated on held-out/walk-forward data.
- **Desk is a factor in the true probability, not a post-hoc overlay.** When the
  desk has a pick on the game, its information enters the true-probability
  estimate here: the model's feature-based number is adjusted by the desk's
  signal (conviction tier + side agreement) and the result *is* the true
  probability the +EV engine uses. Example: features give 47%; the desk surfaces
  news worth +4pp → true probability 51%. Mechanically this is a **margin
  adjustment of up to ±3.5 pts** scaled by conviction/agreement, with the desk's
  net contribution to the final probability **hard-clamped to ≤ ~10 percentage
  points** so news moves the number meaningfully but never egregiously (a 30%
  can't be dragged near 60%). Caps are named constants; the desk's per-game
  contribution is recorded so the effect is auditable. Games with no desk pick
  use the un-adjusted feature probability.
- **Backtest:** walk-forward over historical seasons — margin/total MAE vs the
  closing line, win-prob calibration (reliability), and a **CLV proxy** vs the
  closing line. This is the go/no-go evidence for whether any edge exists.

## Sub-project 4 — Edge/EV engine + desk influence

- Reuse `board.py`. Per market/side produce: `pinnacle_novig` (no-vig implied
  from the two-way Pinnacle price), `true_prob` (from sub-project 3),
  **`edge = true_prob − pinnacle_novig`**, **`ev_pinnacle`** (`ev(true_prob,
  pinnacle_price)`), **`ev_best`** (`ev(true_prob, best_soft_book_price)`) + the
  book name, and the delta between them (the soft-book benefit).
- The `true_prob` consumed here **already includes the desk** (it is folded into
  the true-probability estimate in sub-project 3, clamped to ≤ ~10pp). This
  engine does not apply a second desk adjustment; it just reads the
  desk-inclusive true probability and the per-game desk contribution recorded
  alongside it (for display/auditing).
- **Discipline gates:** `EV_CEILING` (kill stale-line artifacts), a minimum
  edge threshold (below it → **pass / no edge**, the default), and only surface
  markets where both a Pinnacle and a soft-book two-way price exist.
- Write `ev_board` (one row per market/side considered, pick-flagged).

## Sub-project 5 — +EV page / toggle (both sports)

- **Toggle** on each NFL/CFB page: "Predictions" (unchanged, reads
  `predictions_current`) ↔ "**+EV**" (reads `ev_current`). Likely a segmented
  control persisted in `localStorage`.
- **+EV row** per pick: matchup, market/side, **Pinnacle implied %** (no-vig),
  **true %**, **edge (gap)**, **EV @ Pinnacle**, **EV @ best book (book name)**,
  and the soft-book delta. A "read the desk's reasoning" panel where the desk
  influenced the number (reuse the desk reasoning-panel component). Honest
  empty/pass state when nothing clears the threshold.
- `app.js` (external) edit; user redeploys. No prediction-page changes.

## Sub-project 6 — Forward grading / CLV + calibration tracking

- After each game: grade every surfaced +EV pick against the result **and** the
  **Pinnacle closing line** (CLV = did we beat the sharp close). Store to
  `ev_results`.
- Maintain a **reliability/calibration tracker** (predicted vs realized hit rate
  by probability bucket) and a rolling CLV record — the honest scoreboard that
  says whether the engine actually has edge. Feeds recalibration.

## Schema (new; the user runs the SQL)

- `pinnacle_odds` / `book_odds` — captured lines (open/current/close snapshots),
  keyed by (sport, game_pk, market, side, line, book, captured_at).
- feature store — per-game features (parquet asset and/or table).
- `ev_board` (+ `ev_current` view: upcoming, pick-flagged, one current row per
  market/side) — the board the page reads.
- `ev_results` — graded picks: win/cover/over, CLV vs Pinnacle close, calibration
  bucket. All `(sport, game_pk, …)`-keyed, public-read for the anon browser like
  the existing views.

## Global constraints (bind every sub-project)

- Prediction model/pages untouched; reuse `board.py`/`distributions.py`/
  `calibration.py`/`odds.py` rather than reimplementing implied-prob/EV/no-vig.
- **Pinnacle is region `eu`** in The Odds API; always request `regions=us,eu`.
- Edge is measured at the **Pinnacle no-vig** number; EV shown at both Pinnacle
  and best soft-book price.
- The desk is folded **into** the true probability (sub-project 3), not applied
  as a separate step: ≤ ±3.5 pts margin, hard-clamped to ≤ ~10pp probability
  contribution; every cap a named constant; effect recorded per game.
- Disciplined framing everywhere: calibrated, `EV_CEILING`, min-edge "pass"
  default, forward CLV-gated; the UI never implies bankable profit before the
  CLV/calibration record supports it.
- Secrets only in `main()`; user runs SQL; `app.js` external.

## Build order

1 (odds) + 2 (features) in parallel → 3 (true-prob model + backtest: the go/no-go)
→ 4 (edge/EV + desk) → 5 (page) with 6 (grading/CLV) alongside 4–5. Each
sub-project gets its own implementation plan and runs through subagent-driven
development.

## Open questions to settle during planning (not blockers)

- Exact feature list finalization after the backtest shows which features carry
  signal (drop dead weight rather than ship a bloated model).
- Whether CFB's weaker inputs justify a CFB-specific σ / calibration or a shared
  one.
- Historical depth for fitting (how many seasons of nflverse EPA / CFBD PPA).
