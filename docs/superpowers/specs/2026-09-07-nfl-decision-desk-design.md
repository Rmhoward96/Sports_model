# NFL Decision Desk — design

**Status:** approved (2026-09-07)

**Goal:** Extend the existing CFB decision desk to NFL, so the same three-agent
(statistics / sports-analyst / news) on-demand desk produces tiered ML / Spread
/ Total picks for NFL games, graded forward against the ESPN closing line with
CLV, and surfaced on the NFL page of the site.

## Context — what already exists and is sport-generic

The CFB desk (built under `2026-09-07-cfb-decision-desk-design.md`) is already
partly sport-agnostic. Reused **unchanged** for NFL:

- **Schema** — `desk_picks`, `desk_pick_results`, and the `desk_current` /
  `desk_record` views are all keyed by `(sport, game_pk, …)`. NFL rows appear
  the moment they are written. **No migration is needed.**
- **Writer** — `scripts/write_desk_picks.py` (`validate_picks` / `writable_picks`)
  is sport-agnostic: it writes whatever `sport` each pick JSON entry carries.
- **Grader** — `scripts/grade_desk_picks.py` already loops
  `FINAL_PROVIDERS = {"nfl": nfl_espn, "cfb": cfb_espn}` and grades both sports.
  `nfl.espn.fetch_final` already returns `market_spread`/`market_total` from
  ESPN pickcenter in the same sportsbook convention CLV expects. **No change.**
- **Front-end component** — `deskCurrent(sport)`, `deskRecord(sport)`,
  `deskSection`, `deskCard`, the reasoning panel, and `logoImg(team, sport)` in
  `app.js` are already parameterized by sport.
- **NFL predictions** — `predictions_current` is already populated for
  `sport = 'nfl'` by `scripts/generate_nfl.py`, and stores `home_team_name` /
  `away_team_name` as the ESPN **displayName** (e.g. "Philadelphia Eagles") —
  the same identifier space CFB uses.

## What must be built

### 1. NFL SportsDataIO adapter (`src/sportsmodel/nfl/sportsdata.py`)

Mirror `cfb/sportsdata.py`, with the NFL base and endpoint paths from the NFL
OpenAPI swagger:

- `_BASE = "https://api.sportsdata.io/v3/nfl"`.
- `_get(path, api_key, params=None)` — identical retry + header auth
  (`Ocp-Apim-Subscription-Key`) as the CFB adapter.
- `parse_injuries(payload)` — identical logic to CFB: the NFL
  `InjuredPlayers` endpoint returns the **same `Player[]` shape**
  (`FirstName`, `LastName`, `Team` = abbreviation, `Position`, `InjuryStatus`,
  `InjuryBodyPart`, `InjuryNotes`). Returns `{abbrev -> [ {player, position,
  status, note} ]}`.
- `parse_teams(payload)` — the NFL `Teams` endpoint's `Team[]` has **no
  `School` field**; use `FullName` (e.g. "Philadelphia Eagles"). Returns
  `{Key -> FullName}`. `FullName` matches ESPN's NFL displayName, so the
  existing `_rekey_by_espn_name` prefix match in `desk_inputs.py` joins injuries
  onto games with no new logic.
- Module-level path constants so `desk_inputs.py` can stay adapter-generic:
  `INJURED_PLAYERS_PATH = "/projections/json/InjuredPlayers"` and
  `TEAMS_PATH = "/scores/json/Teams"`.
  **NFL injuries live under `/projections/json/`, not `/scores/json/` like CFB.**

The matching constants are also added to `cfb/sportsdata.py`
(`INJURED_PLAYERS_PATH = "/scores/json/InjuredPlayers"`,
`TEAMS_PATH = "/scores/json/Teams"`) so both adapters expose the same names.

`parse_injuries` is duplicated rather than shared: the repo's established
pattern is a full per-sport module (`cfb/espn.py` and `nfl/espn.py` already
duplicate substantially), and keeping the sports isolated is preferred over a
shared helper.

### 2. NFL team crosswalk asset (`assets/nfl/nfl_teams.json`)

CFB's recent-form join uses `assets/cfb/fbs_teams.json` to translate the
schedule's team **ids** to ESPN displayNames. NFL's `assets/nfl/schedules.parquet`
keys teams by **abbreviation** (`home_team` = "NYG", "SF", …), so the parallel
asset is `{abbreviation -> ESPN displayName}`.

A one-shot builder `scripts/build_nfl_teams.py` fetches ESPN's NFL teams list
and writes `{normalize_team(abbr) -> displayName}` for the 32 teams; the asset
is committed. This keeps NFL recent-form independent of the (non-fatal)
SportsDataIO fetch, exactly as CFB's form is independent of it.

### 3. Parameterize `scripts/desk_inputs.py` by `--sport {cfb,nfl}`

Default `cfb` (back-compat — the existing workflow/runbook keep working with no
args). Per-sport differences, all resolved from one small sport-config block:

- **adapter module** — `cfb.sportsdata` vs `nfl.sportsdata`; injuries/teams
  paths read from the adapter's `INJURED_PLAYERS_PATH` / `TEAMS_PATH` constants.
- **output path** — `config.DATA_DIR / sport / "desk_bundle.json"`.
- **predictions query** — `WHERE sport = %(sport)s`.
- **schedules** — `assets/{sport}/schedules.parquet`.
- **schedule-team → displayName map** — CFB: `fbs_teams.json`, keyed by id;
  NFL: `nfl_teams.json`, keyed by abbreviation. Both are the same *use* (map the
  schedule's team key to the ESPN displayName so `compute_recent_form` can join
  on displayName). The NFL schedule's `game_type` value for regular season is
  `"REG"` (same as CFB), so `compute_recent_form` needs no change.

`build_bundle`, `compute_recent_form`, `_rekey_by_espn_name`, and the injuries
rekey chain (abbrev → SDIO name → ESPN displayName) are all reused unchanged.

### 4. Workflow (`.github/workflows/desk-inputs.yml`)

Add a `sport` input (default `cfb`) and pass `--sport`. `write-desk-picks.yml`
and `grade-desk-picks.yml` are unchanged (writer and grader are already
sport-generic).

### 5. Runbook (`docs/decision-desk-runbook.md`)

Generalize to both sports: note the `--sport nfl` invocation, the NFL bundle
path, the NFL injuries endpoint path difference, and that NFL injuries are keyed
by abbreviation crosswalked via `FullName`. The picks-JSON contract is
unchanged (the writer is the authority).

### 6. Front-end (`app.js`, external — controller-applied)

`app.js` lives outside this repo (`/Users/ryan/Desktop/CappingAlpha/app.js`,
redeployed by the user), so it is **not** part of the worktree and is edited by
the controller directly, not a worktree subagent. The change is a one-line
condition in `buildLeague`: render the desk section for `sport === "cfb" ||
sport === "nfl"` (everything below it — `deskCurrent`, `deskRecord`,
`deskSection`, `logoImg`, `NFL_TEAM_ABBR`, the reasoning panel — is already
sport-generic). No new SQL for the user to run.

## Validation & framing

Unchanged from CFB: forward-only, CLV-gated, on-demand in-session,
every-game-tiered, **"assistive · not a proven edge"** until the forward CLV
record validates it. The news agent's injury signal is only as good as the
SportsDataIO tier (a free/trial key returns scrambled values); spot-check one
known injury per slate.

## Out of scope

- Any change to the model, ratings, or `generate_nfl.py`.
- Weather / news-headline endpoints (SportsDataIO NFL has a News endpoint, but
  the desk's news agent uses injuries + in-session web search, matching CFB;
  the News endpoint is not wired in).
- Automating the desk run (stays on-demand / in-session).
