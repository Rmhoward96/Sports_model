# Sub-project B.2 — NFL sim usage model for viable player props (design spec)

**Status:** draft for review · **Date:** 2026-09-18
**Program:** A (done) → B (built, NOT activated) → **B.2 (this)** → C (props productization, gated on B.2 passing).

## Why B.2 exists (the B.1 gate result)

B (drive-based sim + B.1 calibration) is built, tested, and isolated, but its
walk-forward gate on **propable players** (players who'd actually have a line)
failed decisively:

| Market | mean err | P50 cov (want .50) | P90 cov (want .90) |
|---|---|---|---|
| pass_yds | **178** | 0.22 | 0.30 |
| rush_yds | 34 | 0.11 | 0.46 |
| rec_yds | 29 | 0.13 | 0.55 |
| receptions | 2.5 | 0.15 | 0.55 |

Two structural causes (not parameter tuning):
1. **Usage dilution** — the sim spreads *season-average* target/carry shares
   across the whole roster (incl. players inactive that week), so the week's
   actual featured players are badly under-projected (P50 cov 0.11–0.15).
2. **QB attribution/join mismatch** — team pass_yds isn't landing on the actual
   starting QB (pass_yds MAE 178).

## Goal

Replace season-average usage with a **per-week active-roster usage model** so the
sim projects the right players' workloads for the specific game, and fix QB
attribution — then **re-gate** on propable players. B.2 succeeds only if the gate
passes; otherwise props are a documented NO-GO.

## Non-goals

- **Not** C: no prop-odds ingestion, prop +EV board, or prop grading yet — those
  come only if B.2 clears the gate.
- **Not** a claim of edge on mainstream markets (QB pass yds, star WR/RB yds are
  heavily shopped). B.2 is about *calibration first*; the realistic edge (for C)
  is softer/niche props + forward CLV, same discipline as the game +EV pilot.
- Within-game correlation stays secondary — get the marginals right first.
- Does not touch A/desk/+EV/CFB/MLB or the still-dormant B activation.

## Approach

### 1. Per-week active-roster usage model (the biggest fix)

New `src/sportsmodel/sim/nfl/usage.py` (pure over passed DataFrames, leakage-free):
- **Active set** from depth charts (`import_depth_charts`, per-week, keyed by
  `gsis_id`, `depth_team` 1=starter/2/3): take the expected-active offensive
  skill players (QB1, RB1–2, WR1–3, TE1–2, plus rotation) for each team for the
  target (season, week). Drop players marked OUT/Doubtful in the injury report.
- **Shares** from **recent** usage (last N games, recency-weighted; default N=5),
  computed from `weekly` targets/carries — but **renormalized over the active
  set only**, so usage concentrates on players who will actually play (this is
  what kills the dilution).
- **Efficiency** (ypr, ypc, catch_rate, td_share) from recent games (recency-
  weighted), per active player.
- **Snap counts** (`import_snap_counts`, `offense_pct`) optionally weight the
  rotation; note the join wrinkle: snaps use `pfr_player_id` while depth
  charts/weekly use `gsis_id` — bridge via `import_ids` (or name+team+week) and
  document the fallback when a player can't be matched.
- All strictly leakage-free: only data before (season, week).

`inputs.build_spec` consumes this active-roster usage instead of the season-share
`player_inputs_from_weekly` for NFL sim specs.

### 2. QB→starter attribution fix

- The projected **starting QB** = depth-chart QB1 (fallback: most recent starter
  by attempts). Attribute team `pass_yds` and pass attempts to that QB's
  `gsis_id`. Ensure `generate_sim_nfl` and the backtest evaluate `pass_yds`
  against **that** player's actual — pairing by the correct `gsis_id` so a MAE
  like 178 (wrong-player comparison) can't recur.

### 3. Per-market / per-position yardage calibration

- Tune the skewed-yardage shapes and usage concentration
  (`_REC_YDS_SHAPE`, `_RUSH_YDS_SHAPE`, `_USAGE_CONCENTRATION`, `_GAME_ENV_K`,
  and team play-volume) against the propable-player backtest, **per position
  where it matters** (RB rush variance ≠ WR rec variance). Objective: coverage
  P50≈.50 and P90≈.90 with unbiased means.

### 4. Honest backtest (fix the accepted-limitation confound)

- Backtest must apply the **same per-week active-roster/inactive filter** it
  will use live (no more `injuries={}` and no full-roster dilution), so the gate
  reflects production. Keep the propable-player metric (usage-gated) from B.1b.

## Ship gate (concrete pass/fail, walk-forward, leakage-free)

Propable players (pass: attempts≥10; rush: carries≥5; rec/receptions: targets≥3):

| Market | MAE target | P50 cov | P90 cov |
|---|---|---|---|
| pass_yds | ≤ 65 | .45–.55 | .85–.93 |
| rush_yds | ≤ 24 | .45–.55 | .85–.93 |
| rec_yds | ≤ 20 | .45–.55 | .85–.93 |
| receptions | ≤ 1.5 | .45–.55 | .85–.93 |

**Pass = all four markets within these bands.** Partial pass (some markets) →
those markets may proceed to C, others stay NO-GO. Clear fail → props shelved
with the honest write-up.

## Data sources (all nflverse, leakage-free)

`import_depth_charts` (per-week starters, gsis_id, depth_team), `import_weekly_data`
(recent usage/efficiency), `import_snap_counts` (rotation weight; pfr_player_id —
bridge via `import_ids`), `import_injuries` (per-week inactives for the backtest).

## Files (anticipated)

- New `src/sportsmodel/sim/nfl/usage.py` (+ tests).
- `inputs.py` (consume active-roster usage), `generate_sim_nfl.py` (build specs
  from it; QB attribution), `backtest_sim_nfl.py` (per-week active filter; keep
  propable metric).
- Kernel tuning constants only (no kernel structure change expected).

## Open questions

- **Recency window N** (default 5) and recency weighting — tune vs the gate.
- **Rookies / role changes / mid-season trades** — depth chart handles most;
  document the cold-start fallback (no recent usage → depth-position priors).
- **snap↔gsis id bridge** coverage — measure match rate; fall back to
  weekly-usage shares when a snap row can't be matched.
- **How hard to chase mainstream-market calibration** vs accept it's shopped and
  aim C at softer props — revisit after the first gate read.
