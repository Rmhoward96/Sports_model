# NFL P4 go-live validation runbook

Date: 2026-08-26
Owner: run manually, in order, by the user (Supabase + GitHub Actions access required)
Scope: this is the RUNBOOK for the live end-to-end validation described in
`docs/superpowers/specs/2026-08-25-nfl-p4-golive.md` (§H) and
`docs/superpowers/plans/2026-08-25-nfl-p4-golive.md` (Task 9). The runtime steps
themselves (running the migration, dispatching workflows, inspecting Supabase) were
**not executed by this task** — no Supabase/Actions access from this session. This
document is what to run and what "pass" looks like, so the user (with the controller)
can execute it post-merge before flipping Week 1 on.

Do this on a **completed 2025 (or preseason) NFL week** first — never on Week 1 cold.

---

## Step 0 — migration (must run before any NFL producer/board activity)

1. Open Supabase → SQL Editor.
2. Run `db/migration_nfl_sport.sql` in full. It is idempotent (`ADD COLUMN IF NOT
   EXISTS`, `UPDATE ... WHERE sport IS NULL`) — safe to re-run if unsure whether it
   already ran.
3. Confirm: `game_predictions` and `prop_predictions` both have a `sport TEXT` column
   (default `'mlb'`), and existing rows show `sport = 'mlb'` (no NULLs). `board_picks`/
   `picks` already carry `sport` from the serving layer — no change needed there.

**Pass condition:** both `ALTER TABLE` statements succeed (or no-op on rerun), no rows
have `sport IS NULL` in either table.

---

## Step 1 — producer (`generate-nfl.yml`)

1. GitHub → Actions → `generate-nfl` → **Run workflow** (workflow_dispatch) on the
   branch this lands on, for the current/target NFL week.
2. In the run log, `scripts/generate_nfl.py`'s summary line reports `predicted N games,
   M prop rows` — record N and M.
3. In Supabase, query:
   ```sql
   select model_version, count(*) from game_predictions where sport = 'nfl' group by 1;
   select model_version, count(*) from prop_predictions where sport = 'nfl' group by 1;
   ```
4. Confirm `game_predictions.sport = 'nfl'` rows exist with `model_version =
   'nfl-elo-v1'`, count = N games for the target week; `prop_predictions.sport = 'nfl'`
   rows exist with `model_version = 'nfl-props-v1'`, count = M prop rows.

**Pass condition:** N game rows + M prop rows written, both tagged `sport='nfl'` with
the expected model_version strings. No exception in the run log.

**Known risk to watch:** none new here — P1–P3 assembly is unit-tested; this step is
mechanical wiring (schedule → ratings → gameline → props → write).

---

## Step 2 — odds + board (`capture-odds.yml`, NFL leg)

1. GitHub → Actions → `capture-odds` → **Run workflow**. The workflow's NFL leg runs
   `ingest_odds.py --sport nfl` then `generate_board.py --sport nfl`.
2. In the run log for the NFL leg, check the matcher summary line: `NFL events: X
   matched to game_pk, Y unmatched`. Any `no ESPN match for NFL event ...: <home> vs
   <away>` lines name the exact Odds-API team strings that failed to match.
3. **This is the live verification point for the Odds-API NFL team-name strings.** The
   matcher (`src/sportsmodel/nfl/matcher.py::match_odds_event`) does an exact
   case/whitespace-insensitive string match between the Odds-API event's
   `home_team`/`away_team` and ESPN's `displayName` (surfaced via
   `espn.parse_schedule`'s `home_name`/`away_name`), joined with the commence date. If
   `Y > 0`, open the printed unmatched name(s) and diff them against ESPN's
   `displayName` for that franchise. If they differ (e.g. an alternate naming), add a
   normalization step to `matcher._norm_name` (or a small alias table alongside it) —
   do not touch the exact-key algorithm otherwise, it is what MLB's game_pk join also
   assumes.
4. Confirm in Supabase:
   ```sql
   select market, count(*) from odds_snapshot where game_pk in
     (select game_pk from game_predictions where sport = 'nfl') group by 1;
   ```
   Expect rows for `moneyline`/`spread`/`total` (game lines) AND the 7 prop markets
   (`passing_yards`, `passing_tds`, `receiving_yards`, `receptions`, `rushing_yards`,
   the rush+rec combo, anytime-TD) — not just game lines.
5. Confirm `board_picks`/`picks`:
   ```sql
   select market, count(*) from board_picks where sport = 'nfl' group by 1;
   ```
   Expect PROP rows (not only moneyline/spread/total), each with a best-book price, an
   `ev` value, and only `is_pick = true` where EV clears the EV-or-pass gate (some rows
   with `is_pick = false` — priced but not qualifying — are expected and fine).

**Pass condition:** matcher match rate is 100% (or any misses are understood + fixed
before Step 3); `odds_snapshot` has both game-line and prop rows keyed by NFL `game_pk`;
`board_picks`/`picks` show `sport='nfl'` rows for game lines AND props.

**Known risk to watch (from spec §H/Risks):** "Odds-API NFL team names unverified until
live" — this step is that verification. Fix in `matcher.py`, not in the DB.

---

## Step 3 — grade (`grade-results.yml`, NFL leg, after games finish)

1. Wait until the target week's games are final (or pick an already-completed past
   week for this validation run).
2. GitHub → Actions → `grade-results` → **Run workflow**. The NFL leg runs
   `grade_results.py --sport nfl`.
3. **This is the live verification point for two things at once:**
   - **ESPN box-score shape.** `src/sportsmodel/ingest/nfl_results.py::fetch_results`
     parses a real ESPN box score for per-player prop actuals (passing/rushing/
     receiving yards, receptions, TDs, anytime-TD). The parser is fixture-tested
     against a committed sample box score (`tests/fixtures/nfl`), but ESPN's live
     summary payload can have per-game quirks the fixture didn't cover (missing
     category, different key casing, a DNP player omitted vs present). If prop grading
     comes back empty or wrong for a game, compare that game's live
     `site.api.espn.com/.../summary` payload against the fixture and adjust
     `_category_stats`/`parse_results` — the contract `_actual_for` expects
     (`home_score`/`away_score` + per-player-per-market actuals) should not change.
   - **ESPN-athlete-id → gsis crosswalk.** `nfl_results.fetch_results` re-keys ESPN's
     `players` dict (keyed by ESPN athlete id) to gsis `player_id` via
     `_espn_id_crosswalk()` (built from the committed `rosters.parquet` `espn_id`
     column). If this crosswalk misses a player (no `espn_id` row, or a stale/traded
     player), that player's props silently fail to resolve. **"0 props graded" for a
     game that had NFL props on the board is the signature of a crosswalk miss** —
     don't mistake it for "nothing to grade."
4. Confirm in Supabase:
   ```sql
   select result, count(*) from picks where sport = 'nfl' and status = 'graded' group by 1;
   select count(*) from picks where sport = 'nfl' and status = 'graded' and clv is not null;
   ```
   Cross-check the count of graded NFL prop picks against the number of NFL prop
   `board_picks` rows from Step 2 for the same week — they should be close (some props
   may still be pending if a game hasn't finished).

**Pass condition:** game-line picks grade to `win`/`loss`/`push` with `home_score`/
`away_score` populated; prop picks grade with a non-null `actual` and `result`; CLV is
populated on graded rows; prop-graded count is NOT "0 props graded" when props were on
the board pre-game.

**Known risk to watch (from spec §H/Risks):** "ESPN box-score shape for prop actuals ...
finalized against a real box score during validation" and the id-crosswalk join — both
covered above.

---

## Step 4 — front-end (NFL logos)

1. The external CappingAlpha site (`app.js`, `nfl.html`, deployed to Cloudflare Pages)
   was edited in this task to add an NFL team-logo path parallel to the existing MLB
   one — see "Front-end changes" below. This file is **not part of this repo** and is
   **not committed by this task**.
2. The user redeploys the CappingAlpha site to Cloudflare Pages with the updated
   `app.js`.
3. Open the deployed site's NFL page (`nfl.html`) once Step 2/3 have produced
   `board_picks`/`picks` rows with `sport = 'nfl'`. Confirm:
   - Game-line rows show both team logos next to the matchup, and the pick's team logo
     next to the pick label (spreads/moneylines; totals show no team logo, as with MLB).
   - Prop rows show the player's team logo next to the player name.
   - Track Record page, "All graded plays" table, filtered to NFL, also shows logos in
     the matchup/player and pick columns.
   - Broken-image icons (a `NFL_TEAM_ABBR`/`NFL_CODE_TO_ESPN` lookup miss) do not
     appear — the `onerror` handler hides a failed image, so a *missing* logo (blank
     space where one is expected) is the failure signature to look for, not a broken
     icon.

**Pass condition:** NFL rows on both `nfl.html` and the Track Record page render team
logos with no gaps for any of the 32 teams that appear in this week's data.

---

## Go-live checklist (before flipping Week 1 on)

- [ ] Step 0 migration run in Supabase (idempotent; confirm no `sport IS NULL` rows).
- [ ] Step 1: `generate-nfl.yml` run produces the expected game/prop counts, tagged
      `sport='nfl'`, `nfl-elo-v1`/`nfl-props-v1`.
- [ ] Step 2: `capture-odds.yml` NFL leg matches 100% of Odds-API events to `game_pk`
      (or any misses are fixed in `matcher.py` and re-verified); `board_picks` shows
      both game-line and prop rows.
- [ ] Step 3: `grade-results.yml` NFL leg grades finals + props with CLV populated; no
      "0 props graded" surprises; box-score parser and id-crosswalk hold up against a
      live game.
- [ ] Step 4: `app.js` redeployed; NFL logos render on `nfl.html` and Track Record.
- [ ] Sane-check pass (below) on at least one full validation week.
- [ ] `generate-nfl.yml`, and the NFL legs of `capture-odds.yml`/`grade-results.yml`,
      are enabled/unpaused for the live Week 1 schedule (~Sept 10, 2026 kickoff week).
- [ ] Full MLB suite still green / MLB board+track record unaffected (spot-check the
      site's MLB page and the `pytest` suite) — this work is additive only.

## Known live-verification items (first-run watch list)

These are the three seams the spec explicitly flags as "verified live, not by unit
test" — read their write-ups in Steps 2–3 above before assuming a failure is a bug
elsewhere:

1. **Odds-API NFL team-name strings** vs ESPN `displayName` (Step 2) — exact-string
   matcher; a mismatch shows as `no ESPN match for NFL event ...` in the
   `capture-odds` log, not a crash.
2. **ESPN box-score shape** for prop actuals (Step 3) — the parser is fixture-tested,
   not live-tested; a live shape difference shows as wrong/empty prop actuals for a
   specific game, not a crash.
3. **ESPN-athlete-id → gsis crosswalk** (Step 3) — a missing/stale roster row shows as
   "0 props graded" for a game that clearly had props on the board.

## Sane-check list (run against the validation week's output)

- **Game lines vs market:** the model's predicted spread/total should sit reasonably
  close to the captured market line for each game — large systematic deviations (e.g.
  every home team favored by more than the market by several points) suggest a sign
  or shrinkage-config bug, not a real edge. (Recall the P4 round-1 fix: market-line
  sign flip — re-verify this hasn't regressed.)
- **Player projections are plausible:** a WR1's projected receiving yards should be in
  a normal range for their usage (not e.g. 300+ yards or a starting RB projected for
  single-digit rushing yards) — spot-check 3-5 marquee players against public
  projections/context.
- **No props for inactive players:** cross-reference the week's ESPN inactives list
  against `prop_predictions`/`board_picks` for that game — no player on the inactive
  list should have a prop row. This was explicitly guarded in the producer
  (`universe.active_universe`) — confirm it held on a real inactives list.
- **Backup bump is real, not a placeholder:** when a starter is OUT, confirm the
  backup at that position/team shows an elevated projected share relative to a normal
  week (the proportional backup-bump fix from P4 round 1).
- **EV-or-pass gate is working:** `board_picks` should include priced-but-not-picked
  rows (`is_pick = false`) alongside picks — if literally everything on the board is
  `is_pick = true`, the gate likely isn't filtering.

---

## Front-end changes (this task)

File: `/Users/ryan/Desktop/CappingAlpha/app.js` (external site, NOT in this repo — the
user deploys it separately to Cloudflare Pages; it is not committed by this task).

Added, alongside the existing MLB `TEAM_ABBR`/`logoImg`/`logoPair`/`pickLogo` helpers
(unchanged):

- `NFL_TEAM_ABBR` — 32 NFL teams, full ESPN `displayName` → ESPN CDN slug (e.g.
  `"Kansas City Chiefs": "kc"`). Used the same way MLB's `TEAM_ABBR` is used for
  `matchup`/`pick_label` strings, which carry full team names for NFL too.
- `NFL_CODE_TO_ESPN` — the nflverse-style franchise code stored in NFL prop rows'
  `team` column (via `normalize_team()` in `src/sportsmodel/nfl/teams.py`, e.g. `KC`,
  `WAS`, `LA`) → ESPN CDN slug. This is a separate map from `NFL_TEAM_ABBR` because the
  `team` field on prop rows is a short code, not a full display name, and a few codes
  differ from their ESPN slug (`WAS`→`wsh`, `LA`→`lar`).
- `logoImg`, `logoPair`, `pickLogo` gained an optional `sport` parameter (default
  behavior unchanged when omitted, so no existing MLB call site changed behavior).
  When `sport === "nfl"`, `logoImg` falls back to `NFL_TEAM_ABBR` then
  `NFL_CODE_TO_ESPN` and points the CDN path at `teamlogos/nfl/500/` instead of
  `teamlogos/mlb/500/`.
- `gameRow`, `propRow` (used by `buildLeague`, i.e. `nfl.html`/`mlb.html`/`nba.html`)
  and `betCell`/`pickCell` (used by `buildTrack`, i.e. the Track Record page's
  all-leagues graded table) now pass the row's own `r.sport` through to these helpers,
  so NFL rows resolve NFL logos and MLB rows are unaffected (their `r.sport === 'mlb'`
  still resolves through the original `TEAM_ABBR` path).

**What the user should verify after redeploying:**
- Visually confirm all 32 NFL team logos load correctly on `nfl.html` once real board
  data is present (Step 4 above) — the CDN slugs were sourced from the standard ESPN
  team-logo naming and cross-checked against `teams.py`'s alias table, but this was not
  runtime-verified against the live CDN from this session.
- Confirm no MLB visual regression on `index.html`/`mlb.html`/`track-record.html`
  after the change (the diff is additive/parameterized, but a manual look-over is
  cheap insurance since this file has no test suite).
