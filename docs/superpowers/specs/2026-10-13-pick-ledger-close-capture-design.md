# Pick ledger + pre-kick close capture (profit project, step 1) — design spec

Date: 2026-10-13 · Status: design approved in chat; awaiting user review of this file
Evidence: `docs/superpowers/reports/2026-10-13-edge-audit.md` (branch `docs/edge-audit`). Step 1 of the profit roadmap: make every pick measurable against a trustworthy close before any edge claim is tested (step 2 = shrink model probabilities toward the market; step 3 = early-week stale-number paper tickets).

## Goal

Record every pick the system surfaces exactly as it was at pick time (never overwritten), capture a true closing price shortly before every kickoff (game lines and the props we pick, US books + Pinnacle), and grade every pick against that close — so CLV and ROI are measured honestly from now on.

## User decisions (2026-10-13)

- Single user; no site relabels; no site changes in this step.
- Bets at several US books, early in the week, props included.
- Odds API: 30,000 credits/month. Option B chosen: game lines + props near kickoff, US books + Pinnacle (eu region). Current usage ≈ 2,300 credits/week (≈ 10,000/month); this step adds ≈ 360/week (≈ 1,550/month) during NFL + CFB + MLB postseason; a full MLB regular season would add ≈ 300–400/week more for MLB props.

## Audit facts this fixes

- `ev_picks` / `ev_prop_picks` overwrite price, probability and EV on every rebuild; `game_predictions` is overwritten (its `generated_at` is after kickoff for most graded games); `prediction_accuracy` has no `model_version`.
- "Closes" are stale: last Pinnacle capture > 12 h before kickoff for 53% of NFL and 24% of CFB games; only 44% of NFL / 71% of CFB games have a capture within 3 h of kickoff.
- Props: zero Pinnacle prop prices (prop pulls use `regions=us` only); only 39% of prop picks have a ≥ 2-book US-consensus close at their exact line; `ev_prop_results.clv` is NULL for all 542 graded props.

## 1. Pick ledger (append-only)

Table `pick_ledger` (new; user runs the migration). One row per *observation* of a pick:

- Identity: `source` (`model_game` | `ev_game` | `ev_prop` | `parlay_leg`), `sport`, `game_pk`, `market`, `side`, `line` (nullable), `player_id` / `player_name` (props), `parlay_id` (legs), `pick_key` (deterministic text key of the above, excluding price).
- Model at pick time: `model_version`, `model_prob` (the model's probability for this side at `line`; NULL when the source has none).
- Market at pick time: `best_price` (American), `best_book`, `ref_prob` (fair probability of this side at `line`), `ref_kind` (`pinnacle_novig` | `us_consensus_novig` | NULL), `ref_books` (count used), `ev_pct` (as the source computed it).
- Timing: `observed_at` (UTC), `commence_time`, `minutes_to_kick`.
- Provenance: `writer` (script name), `writer_run` (GitHub run id or local).

Write rule: the writers that already build these picks (generate/board/props/parlay scripts and `grade_predictions` for model game picks) append a ledger row when a `pick_key` is first seen, and again only when `best_price` changes, `line` changes, or `model_prob` / `ref_prob` moves by ≥ 1.0 percentage point since that key's last row. Rows are never updated or deleted: enforced by a DB trigger that raises on UPDATE/DELETE, plus `REVOKE UPDATE, DELETE` from every non-owner role. Ledger writes are best-effort: a ledger failure logs a warning and never fails the job that produced the pick. Existing tables and pages are unchanged.

Index: `(sport, game_pk)`, `(pick_key, observed_at)`.

## 2. Pre-kick close capture

New script `scripts/capture_close.py` + workflow `capture-close.yml`:

- Schedule: every 10 minutes during game windows — Thu/Fri/Sat/Sun/Mon for football, daily Mar–Nov for MLB — bounded so idle hours cost nothing (a run with no kickoff in range exits before any API call).
- Each run: load upcoming kickoffs (schedules already ingested for NFL/CFB/MLB); select events kicking off in 5–35 minutes that have no close capture yet; group by sport.
  - Game lines: one Odds API call per sport (all events; markets h2h/spreads/totals; regions us,eu) per run with targets.
  - Props: one per-event call (regions us,eu; the sport's live prop markets) for each target event that has ≥ 1 prop pick in the ledger.
- Store the captured prices in `odds_snapshot` (existing table) and record one row per event in new table `close_capture` (`sport`, `game_pk`, `event_id`, `kind` = lines|props, `captured_at`, `commence_time`, `minutes_to_kick`, `credits_used`, `status`). Unique on (`sport`, `game_pk`, `kind`) so an event is captured once; a failed attempt is retried by the next run while still in the window.
- Pinnacle props probe first (≈ 10 credits, one NFL event, regions=eu): record whether Pinnacle prop prices exist; if none, prop captures use regions=us only (saves credits) and the report states it.
- Budget guard: the script reads `x-requests-remaining`; below 3,000 remaining it skips prop captures and warns; below 1,000 it skips everything and warns. Credits per run are logged.

## 3. Grading against the close + weekly gate

New script `scripts/grade_ledger.py` (nightly workflow `grade-ledger.yml`) writes `pick_ledger_results` (one row per `pick_key`, graded from the key's FIRST ledger row = the earliest bettable observation, and also from its LAST pre-kick row):

- Close reference for the pick's side: Pinnacle no-vig from the `close_capture` snapshot; else ≥ 3-book US-consensus no-vig at the exact line; else NULL with `close_kind = 'missing'`.
- Metrics: `result` (W/L/P, from existing actuals / boxscores), `units_at_pick` (1 u at `best_price`), `clv_prob` (close fair prob − pick-time implied prob of `best_price`, same line only), `clv_pts` (for spreads/totals when the close line differs: line points gained, signed), `beat_close` (bool), `close_kind`, `close_minutes_to_kick`.
- Weekly gate report `scripts/report_ledger_gate.py` → markdown under `docs/superpowers/reports/` (committed by the workflow) checking:
  - ≥ 95% of graded games have a game-line close captured within 30 min of kickoff,
  - ≥ 90% of graded prop picks have a Pinnacle or ≥ 3-book close at their exact line,
  - 0 ledger rows updated or deleted (trigger log + row-count audit),
  - for 2 consecutive football weekends. Step 1 passes only when all hold; a miss blocks steps 2–3.

## Migration (user runs)

`db/migration_pick_ledger.sql`: tables `pick_ledger`, `close_capture`, `pick_ledger_results`; indexes; append-only trigger on `pick_ledger`; RLS enabled with anon SELECT only and writes revoked from anon/authenticated (same pattern as `db/migration_site_panels.sql`). Idempotent, additive only. Jobs warn and exit 0 until the tables exist (same missing-table guard as the site-panel jobs).

## Testing

Network- and DB-free unit tests: pick_key determinism; ledger append rule (first seen, price change, ≥ 1 pp move, no-op otherwise); ledger failure never fails the writer; close-capture window selection (5–35 min, already-captured skip, retry after failure, idle run makes no API call); budget guard thresholds; Pinnacle-probe handling; reference-price selection (Pinnacle → ≥ 3-book consensus at exact line → missing); CLV maths (same-line prob, line-moved points, push handling); gate report pass/fail logic; workflow schedule tests. Existing tests stay green.

## Out of scope

Any change to picks, models, probabilities, boards or the site; steps 2 and 3 of the roadmap; bet sizing.

## Risks

- GitHub cron can run late or skip; the 5–35 min window and 10-min cadence absorb typical delays, and the gate measures actual coverage.
- Pinnacle may not offer props through the Odds API; then props rely on US consensus (≥ 3 books at the exact line).
- Kickoff-time changes (flexed games, delays) — targets are recomputed every run from the latest schedule.
