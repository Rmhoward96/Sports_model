# Matchups, QB career profiles and defensive injuries in the props-ML model (design spec)

Date: 2026-09-28 · Status: approved in chat (Option 1), pending user review of this file

## Goal

Make every NFL projection — every position, starters and backups alike —
respond to the matchup and to who is actually playing:

- A defense's run and pass strength, relative to league average and adjusted
  for the offenses it faced, moves each player's projection. **Strong run D /
  weak pass D ⇒ RB rushing down, QB / WR / TE passing up; strong pass D / weak
  run D ⇒ the reverse.**
- Every QB — starter, backup, veteran returning after years on the bench,
  rookie — gets a career passing profile from his own history, adjusted for
  the defenses he faced and weighted by recency and sample size; the team's
  passing is projected with tonight's QB, not whoever threw last month.
- The opponent's Out/Doubtful defenders weaken the part of its defense they
  play (coverage, pass rush, run defense) in proportion to their usual snaps.

Trigger: PHI @ CHI 2026-09-28 served Case Keenum 122.5 pass yds vs books
170.5–202.5, and a strong-run/weak-pass opponent barely moves an RB today.

## Decisions (user, 2026-09-28)

1. Every QB gets a profile built from his history (yards per attempt and the
   rates below) — no hand-picked list; recency- and sample-weighted (a long
   career like Keenum's is mostly his own record, a thin one leans toward
   replacement level).
2. Everything is judged against the defense: past games are
   opponent-adjusted; tonight's opponent's run and pass defense vs league
   average moves every position, both directions.
3. Defensive injuries count, per defensive unit.
4. Approach: **Option 1** — these become model INPUTS and the ML models are
   refit and re-gated (not fixed multipliers layered on the current models,
   which would double-count the pass-defense features already inside them).
5. Nothing ships without the walk-forward gate; the user sees the numbers
   either way.

## Root cause and gaps (for the record)

- B (per-market model, 70 % of pass_yds) runs for QBs with
  `p_y_pass_att_ewm >= 10`; Keenum's 15.7 comes only from 2021–23 games (last
  pass 2023 wk16) → B used three-year-old tiny samples. Bagent (7.0) was out
  of role → A only (225).
- A (learned volume / shares / efficiency) has no notion of who is throwing.
- `st_qb_changed` reads the nflverse depth chart (still the old QB1).
- Opponent features today: `op_def_adj`/`op_def_prev` (overall EPA),
  `op_ypt_allowed_{wr,te,rb}_{r5,ewm}` (raw, not opponent-adjusted),
  `op_press_rate_*`, `op_pass_att_*`/`op_rush_att_*` (volume). **No run-defense
  efficiency, no pass/run split of team or opponent efficiency, nothing
  adjusted for schedule, no defensive injuries.**
- History 2021–25, team pass yds vs its trailing 5-game mean (median):
  regular starter 0.995× (n=2528), experienced-but-stale 0.945× (n=59),
  low-volume backup 0.885× (n=73), no NFL history 0.746× (n=25).

## Design

All new columns are built by `sportsmodel.nfl.player_features` (the single
builder used by training, backtest and live serving), strictly pre-game
(rows for (S, w) read only games before (S, w); the live injury report is
pre-game information, as today).

### 1. Opponent-adjusted unit efficiency (team table, `tm_`/`op_` prefixes)

Per team-game from nflverse play-by-play (regular season):
- pass unit: EPA/dropback, net yards per dropback, sack rate, success rate;
- run unit: EPA/rush, yards per carry, success rate, stuff rate (≤ 0 yds).

Opponent adjustment (iterative, per season, as-of): offense rate adjusted =
raw − (opponent's allowed − league); defense allowed adjusted = raw −
(opponent offense's − league); two passes of the alternating adjustment,
each using only games before the target week; early-season values shrunk
toward the prior season's (regressed to league) value.

Features, each as `_r5`, `_ewm`, `_prev` and **relative to league average**
(difference from the league mean of games before (S, w)):
- offense: `tm_pass_epa_adj`, `tm_pass_nypd_adj`, `tm_sack_rate_adj`,
  `tm_rush_epa_adj`, `tm_ypc_adj`, `tm_rush_success_adj`;
- opponent defense: `op_pass_epa_allowed_adj`, `op_nypd_allowed_adj`,
  `op_sack_rate_gen_adj`, `op_rush_epa_allowed_adj`, `op_ypc_allowed_adj`,
  `op_stuff_rate_adj`, plus opponent-adjusted `op_ypt_allowed_adj_{wr,te,rb}`;
- matchup edges: `mx_pass_edge = tm_pass_epa_adj − op_pass_epa_allowed_adj`
  (sign: + favors offense), `mx_rush_edge` likewise, and
  `mx_pass_minus_rush = mx_pass_edge − mx_rush_edge` (the "run D strong, pass
  D weak" axis).

These join to every player row (all positions), so a backup RB or WR gets the
same matchup context as the starter.

### 2. Defensive injuries (team table, `di_` prefix)

- Each defender's recent share of his team's defensive snaps (nflverse snap
  counts, EWM, PFR ids bridged to gsis through the existing ids table),
  grouped: coverage (CB, S, nickel/DB), pass rush (DE, OLB/EDGE), run
  defense (DT/NT/DL, ILB/MLB/LB). A player can count in one group only
  (primary position).
- For game g, the opponent's `di_vacated_cov`, `di_vacated_rush`,
  `di_vacated_run` = summed recent snap shares of its Out/Doubtful defenders
  in that group (history: nflverse injury reports; live: the shared
  `current_report`, same path as offense). Missing report ⇒ NaN (as the
  existing `st_` features).

### 3. QB career profile (team + player tables, `qb_` prefix)

New module `src/sportsmodel/nfl/qb_profile.py` (pure; loader separate).

Source: nflverse weekly player stats, regular season, **1999 onward** (older
seasons only feed profiles).

Per QB-game with ≥ 1 attempt: `ypa`, `td_rate` (TD/att), `int_rate`,
`sack_rate` (sacks / (att + sacks)), each **opponent-adjusted** (subtract the
opponent's allowed rate minus league, both from that season's games before
the game; prior-season shrink early in the season).

Profile of QB q as of (S, w), from games strictly before (S, w):
- weight = attempts × 0.5^(age_seasons / H), age fractional by week;
- raw = Σ w·rate / Σ w, effective attempts n = Σ w;
- shrunk = (n·raw + k·replacement) / (n + k), `replacement` = attempt-weighted
  league rate of non-regular QBs (not their team's season attempts leader) in
  seasons before S; no NFL passes ⇒ replacement;
- H ∈ {1, 2, 3, 4} seasons and k ∈ {100, 200, 400, 800} attempts tuned in the
  gate ladder (2021–23 only, see §6).

Features for team T in game g, where QB1 = the sim's starter (depth chart,
injury report, then the books check live; history: the game's actual
pass-attempts leader is NOT used — the as-of depth-chart QB1 is, so training
matches serving):
- `qb_ypa`, `qb_td_rate`, `qb_int_rate`, `qb_sack_rate`, `qb_n_eff` for QB1;
- `qb_ratio_ypa` / `_td` / `_sack` = QB1 profile ÷ the attempt-weighted
  profile mix of the QBs who threw T's passes over the receiver-feature EWM
  window (≈ 1 with the usual starter);
- `qb_changed` (replaces `st_qb_changed`'s source: QB1 as above vs the
  previous game's attempts leader).

Joined to every player row of the team, so receivers', RBs' (checkdowns) and
the QB's own models see who is throwing.

### 4. Models (refit, same machinery)

- A (`sim/nfl/learned.py`): team volume, shares and efficiency models
  (`ypr`, `catch_rate`, `ypc`) take the new columns via new rung prefixes:
  `mx_`/adjusted `tm_`/`op_` (rung **"matchup"**), `di_` (rung
  **"def_injuries"**), `qb_` (rung **"qb_profile"**). Each rung is kept only if
  it passes the ladder (Props-1 rule), so any one can drop out.
- B (`model/props_ml/dist_models.py`): refit on the kept columns.
  QB role for `pass_yds`/`pass_tds` additionally requires a pass in the last 10
  roster weeks (`p_y_pass_att_r10 > 0`) — trained and served the same way;
  out-of-role QBs get B := A (their history now arrives through `qb_`).
- Drive outcomes (kernel TD/FG/punt mix, not learned): the offense's TD
  probability is shifted by `e · (qb_ratio_ypa − 1)` (mass to/from FG and
  punt proportionally), `e` tuned on 2021–23; the current kernel's off/def
  averaging of drive rates already carries the matchup into scoring.
- Blend weights and calibration re-chosen by the existing ladder
  (`train_props_ml_b.py`), final fit by `fit_props_ml_final.py`.

### 5. Serving and rollout

- New model version **`nfl-sim-ml-v2`** (props + game lines), so the accuracy
  record separates the models. `ML_MODEL_VERSION` becomes a served-version
  lookup; v1 artifacts stay published (`props-ml-v1` release) for rollback.
- The site's "Model: ML v1" label reads the version (`ML v2`).
- Live logs per team: QB1, `qb_ypa`, ratios, `mx_*` edges, `di_*` vacated
  shares (`matchup:` line).
- Switch: user runs `UPDATE nfl_sim_serving SET model_version='nfl-sim-ml-v2'`
  after the gate passes; rollback = back to `nfl-sim-ml-v1`.

### 6. Gate (walk-forward 2021–2025, same games/seeds as Props-2)

- Ladder on top of the current served pipeline (v1): + matchup, +
  def_injuries, + qb_profile, then B refit, blend, calibration — each rung
  kept only if it beats the kept composite (existing `rung_decision` +
  ECE check). H, k, e tuned on 2021–23; **verdict on 2024–2025**.
- Ship rule (as Props-2): served composite beats v1 on pooled RPS AND 2025
  alone, ECE not worse than +0.005 per market; game-level (points MAE,
  win/cover/over Brier) not worse than +0.5 %.
- Required sub-checks (reported; the first two must pass):
  1. **QB-change games** (`qb_changed` = 1 or `qb_ratio_ypa` outside
     [0.95, 1.05]): pass_yds RPS improves, paired season-week bootstrap CI
     excluding 0.
  2. **Matchup response, both directions:** bucket player-games by opponent
     `mx_pass_minus_rush` quintile. In the top quintile (strong run D / weak
     pass D) the served v2 mean projection vs the player's own baseline must
     be **lower for RB rush_yds and higher for QB pass_yds, WR/TE rec_yds**;
     in the bottom quintile (strong pass D / weak run D) **the reverse**. The
     v2 deviations must have the same sign as the ACTUAL deviations in each
     bucket and correlate with them better than v1's.
  3. Spot checks: Keenum PHI @ CHI 2026-09-28 (and other 2026 backup starts).
- Report `docs/superpowers/reports/2026-09-xx-matchup-qb-gate.md` + json.

## Out of scope

- Rookie-specific priors (draft slot, college stats); age curves.
- Offensive-line injuries (OL snaps are in the same data — a later rung).
- Changing the +EV game-line base (still Pinnacle no-vig).

## Risks

- Compute: the ladder reruns the 1,359-game × 1,000-sim backtest per rung
  (hours per rung, as Props-1/2).
- Small QB-change sample → wide CIs; if sub-check 1 is inconclusive the
  qb_profile rung is reported but the rule still decides.
- Play-by-play and snap-count schemas differ by season; loaders validate
  per season. Pre-2016 weekly stats only feed QB profiles.
- A new model version touches site labels, views filtered by served version
  and the accuracy record; the rollout keeps v1 intact for rollback.
