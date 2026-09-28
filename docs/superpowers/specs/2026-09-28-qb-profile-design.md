# QB career profiles, QB-change adjustment and defensive injuries (design spec)

Date: 2026-09-28 · Status: approved in chat, pending user review of this file

## Goal

Stop the NFL sim (ML `nfl-sim-ml-v1` and the current sim it's built on) from
mis-projecting a game when the starting QB is not the QB whose throws built the
team's recent passing numbers — a backup, a veteran returning after years as a
backup (Case Keenum, PHI @ CHI 2026-09-28: served 122.5 pass yds vs books
170.5–202.5), a rookie, or the regular starter coming back. Every QB gets a
career passing profile from his own history, adjusted for the defenses he faced
and weighted by recency and sample size; the game's passing is scaled by how
tonight's QB compares with the recent QBs. Separately, the opponent's
defensive injuries start counting.

## Decisions (user, 2026-09-28)

1. Every QB — any backup, any veteran — gets a profile built from his history
   (yards per attempt and the other rates below); no hand-picked list.
2. Recency- and sample-weighted: a long career (Keenum) is mostly his own
   record; a thin one leans toward replacement level.
3. Weighted against the defense: each past game is judged against that
   defense's pass-defense strength vs league average.
4. Injuries count: the opponent's Out/Doubtful defenders weaken its pass
   defense in proportion to the pass-defense snaps they normally play.
5. Nothing ships without a walk-forward validation; the user sees the numbers
   either way.

## Root cause (for the record)

- B (per-market model, 70 % of pass_yds) runs for QBs with
  `p_y_pass_att_ewm >= 10`. Keenum's EWM (15.7) is built only from 2021–23
  games (last pass: 2023 wk16) → B projected him from three-year-old tiny
  samples. Bagent (EWM 7.0) was out of role → A only (225).
- A (learned-inputs sim) sets team pass volume from team features and yards
  from receivers' efficiency features — it has no notion of who is throwing.
- `st_qb_changed` reads the nflverse depth chart, which still had the old QB1.
- History 2021–25, team pass yds vs its trailing 5-game mean (median):
  regular starter 0.995× (n=2528), experienced-but-stale 0.945× (n=59),
  low-volume backup 0.885× (n=73), no NFL history 0.746× (n=25).

## What already exists (not rebuilt)

- Tonight's opponent pass-defense strength already enters the learned models
  through `op_def_adj`/`op_def_prev` (EPA), `op_ypt_allowed_{wr,te,rb}_{r5,ewm}`
  and `op_press_rate_*`. The QB adjustment below is a RATIO of QB profiles, so
  it does not double-count the opponent.
- Offensive injuries: Out players dropped, Questionable scaled, `st_vacated_*`.
- Starting-QB check from the books (`sim/nfl/qb_market.py`).

## Design

### 1. QB passing profile (`src/sportsmodel/nfl/qb_profile.py`, PURE + a loader)

Source: nflverse weekly player stats, regular season, **1999 onward** (older
seasons only feed profiles; the model's other features stay 2016+).

Per QB-game with ≥ 1 pass attempt, opponent-adjusted rates:
- `ypa` = pass yds / attempts, `td_rate` = pass TDs / attempts,
  `sack_rate` = sacks / (attempts + sacks).
- Opponent adjustment: subtract the opponent's allowed rate minus the league
  rate, both from that season's games **strictly before** the game being
  adjusted (shrunk toward the league rate with the opponent's prior-season
  value when few games exist). Result: the rate "vs an average defense".

Profile for QB q as of (season S, week w) — only games strictly before (S, w):
- weight per game = attempts × 0.5^(age_in_seasons / H), age measured in
  seasons (fractional by week) between the game and (S, w).
- raw rate = Σ weight·rate / Σ weight; effective attempts n = Σ weight.
- shrunk rate = (n·raw + k·replacement) / (n + k).
- `replacement` = attempt-weighted league rate of non-regular QBs (QBs outside
  their team's season attempts leader) from seasons before S.
- A QB with no NFL passes gets `replacement` (a rookie-specific prior is out
  of scope).
- H (half-life, seasons) ∈ {1, 2, 3, 4} and k (prior strength, attempts)
  ∈ {100, 200, 400, 800} are tuned (§6).

### 2. QB-change adjustment (applied to the sim spec, current + ML)

For team T in game g (season S, week w):
- `tonight` = profile of the sim's QB1 (after the books check).
- `recent` = attempt-weighted mix of the profiles of the QBs who threw T's
  passes over the same recency window the receiver efficiency features use
  (EWM, same half-life as `player_features`), each profile taken as of (S, w).
- ratios: `r_ypa = tonight.ypa / recent.ypa`, `r_td`, `r_sack` likewise,
  each clamped to [0.6, 1.3].
- Applied to the spec: every receiver's `ypr` (and `ypt`) × `r_ypa`; team
  `pass_td_share`/receivers' `rec_td_share` × `r_td` (renormalized so team
  TD totals follow the drive model); team `sack_rate` × `r_sack`.
- Team scoring: the team's drive outcome TD probability is shifted by
  `e · (r_ypa − 1)` (mass moved to/from FG and punt proportionally), with the
  elasticity `e` tuned on history (§6). This keeps projected score, win prob
  and spread consistent with the QB's props.
- A team with its usual starter has ratios ≈ 1 → no change.
- Applied after `learned.apply_to_spec` in the ML path and after
  `build_spec_from_usage` in the current-sim path, so both serve the same
  adjustment.

### 3. Stale-QB guard for B

For `pass_yds` / `pass_tds`, a QB is out of B's role when
`p_y_pass_att_r10` is missing or 0 (no passes in his last 10 roster weeks);
B := A for him and his history enters through §1–2 instead. Serving-only (the
gated B models are not refit); covered by the validation.

### 4. Defensive injuries (all games)

- Pass-defense snap share per defender: from nflverse snap counts, each
  defender's share of his team's defensive snaps over the recent window (EWM),
  grouped into coverage (CB, S) and pass rush (DE, OLB, DT/DL, edge).
- `vacated_cov` / `vacated_rush` for the opponent in game g = summed recent
  snap shares of its Out/Doubtful defenders in that group (pre-game injury
  report — live: the shared `current_report`; history: nflverse injuries).
- Effect: offense `ypa` multiplier `1 + a·vacated_cov` and `sack_rate`
  multiplier `1 − b·vacated_rush`, coefficients a, b fitted walk-forward on
  history (§6), each multiplier clamped to [0.9, 1.15]. Applied in the same
  spec step as §2.
- Separately gated from §1–3 so one can ship without the other.

### 5. Consistency fixes

- `st_qb_changed` for a team whose QB1 the books check promoted is set from
  the promoted QB (live).
- The live run logs, per team, the QB1, both profiles and the applied ratios
  (`qb-adjust:` line) and the vacated shares (`def-inj:` line).
- Kill switch: repo variable `SIM_QB_ADJUST` (`off` default until the gate
  passes; `on` applies §2–3) and `SIM_DEF_INJ` (same for §4).

### 6. Validation (walk-forward, 2021–2025)

- Replay the production pipeline (current sim → ML A/B/blend) with and
  without each adjustment, same seeds, same games.
- Populations: **QB-change games** (QB1 ≠ previous game's pass-attempts
  leader, or `r_ypa` outside [0.95, 1.05]) and **all other games**.
- Metrics: QB pass_yds and pass_tds RPS, team receiving yards (sum of
  rec_yds) MAE, team points MAE, win-prob Brier.
- Tuning: H, k, e, a, b chosen on 2021–2023; verdict on 2024–2025 only.
- Ship rule (§1–3): QB-change games pass_yds RPS improves with a paired
  bootstrap CI (season-week clusters) excluding 0, and all other games are
  not worse than +0.5 % on any metric. §4: all-games pass_yds and rec_yds RPS
  improve (CI excludes 0), points/Brier not worse than +0.5 %.
- Report: `docs/superpowers/reports/2026-09-xx-qb-profile-gate.md` + json,
  including Keenum 2026-09-28 as a spot check once graded.

## Out of scope

- Rookie-specific priors (draft slot, college stats); age curves.
- Offensive-line injuries.
- Refitting the gated A/B models (the adjustment sits on their inputs/outputs;
  a refit would need the full Props-2 gate).

## Risks

- Small QB-change sample (a few hundred games) → wide CIs; tuning on
  2021–23 and verdict on 2024–25 halves it further. If the gate is
  inconclusive the switch stays off and the numbers go to the user.
- nflverse weekly stats before 2016 use older schemas — loader validates
  columns per season.
- Snap counts are PFR-keyed (`pfr_player_id`); defenders are bridged to gsis
  through the existing ids table for injury matching.
