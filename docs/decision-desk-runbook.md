# Decision-desk runbook (CFB + NFL)

A midweek, in-session runbook: a Claude Code controller (you) reads this doc
and drives the CFB or NFL decision desk end to end — bundle → three agents →
synthesis → write. Nothing here runs unattended; it's invoked on demand by a
person sitting at the controller.

This is **not** a spec for new code. Every script named below already
exists (Tasks 1–5). This doc only defines the *procedure* and the
**picks-JSON contract** the synthesis step must produce, which is enforced
verbatim by `scripts/write_desk_picks.py::validate_picks` — read that
function's docstring if this doc and the code ever disagree; the code wins.

## Prerequisites

- `SPORTSDATA_API_KEY` — SportsDataIO key (injuries via InjuredPlayers, for `desk_inputs.py`).
- `DATABASE_URL` — Supabase Postgres, for both `desk_inputs.py` (reads
  `predictions_current`) and `write_desk_picks.py` (writes `desk_picks`).
- `desk_inputs.py` takes `--sport {cfb,nfl}` (default `cfb`) — pick the sport
  before running Step 1.
- `assets/<sport>/schedules.parquet` and `assets/<sport>/fbs_teams.json`
  (CFB) or `assets/<sport>/nfl_teams.json` (NFL) present (recent-form
  computation reads these).
- `predictions_current` already populated for the chosen `sport` (i.e.
  `scripts/generate_cfb.py` or `scripts/generate_nfl.py` has run for the
  current week, matching the sport) — `desk_inputs.py` does not generate
  predictions, it only reads them.

## Step 1 — Prepare the bundle

```
SPORTSDATA_API_KEY=... DATABASE_URL=... \
  PYTHONPATH=src uv run python scripts/desk_inputs.py --sport cfb --out data/cfb/desk_bundle.json
```

```
SPORTSDATA_API_KEY=... DATABASE_URL=... \
  PYTHONPATH=src uv run python scripts/desk_inputs.py --sport nfl --out data/nfl/desk_bundle.json
```

Optional: `--days-ahead N` (default 7) narrows/widens the slate window.

Note on injuries: the NFL injuries endpoint is
`/projections/json/InjuredPlayers` (CFB uses `/scores/json/InjuredPlayers`).
For both sports, injuries come back keyed by team abbreviation and are
crosswalked to ESPN display names via SportsDataIO's `FullName`. Free/trial
SportsDataIO tiers return scrambled values for some fields — spot-check one
known injury per slate before trusting the feed, for either sport.

This produces one JSON object per **upcoming** game (FBS for CFB, all 32
franchises for NFL; already-started games are excluded). Each entry:

```jsonc
{
  "game_pk": 401628383,
  "matchup": "Georgia @ Alabama",
  "commence_time": "2026-09-12T23:30:00Z",
  "market_spread": -3.5,      // ESPN pickcenter, sportsbook convention:
  "market_total": 54.5,       // home favored -> negative. Same source/
                               // convention the closing-line grader uses.
  "model": {
    "margin": 5.1,             // pred_home_score - pred_away_score
    "total": 57.2,
    "win_prob": 0.71           // home win probability
  },
  "form": {
    "home": {"record": "3-0", "last_n": ["W","W","W"], "avg_margin": 14.3, "pace": 61.0},
    "away": {"record": "2-1", "last_n": ["L","W","W"], "avg_margin": 3.7,  "pace": 55.3}
  },
  "news": {
    "injuries": {"home": [ /* SportsDataIO injury dicts */ ], "away": [ /* ... */ ]},
    "weather": null            // always null: no usable weather endpoint is wired in (CFB or NFL)
  }
}
```

Notes for the controller:
- `model` fields are `null` when the model hasn't produced a prediction for
  that game yet — the game still appears in the bundle; treat it as
  "no model view" rather than skipping it.
- `form.home`/`form.away` is a **scoring/pace trend** (last-N W/L, avg
  margin, avg combined points), *not* a true ATS record — there's no
  historical-line join here. Don't describe it to the user as ATS.
- `news.weather` is always `null` — no usable weather endpoint is wired in
  for either sport; don't fabricate a value.
- The bundle carries **no headlines** — the desk does not pull a news feed
  (NFL has a News endpoint but it is intentionally not wired in; the news
  agent uses injuries + in-session web search, same as CFB).
  `news.injuries` (from the InjuredPlayers endpoint) is the only provider
  signal. The news agent supplements it with **in-session web search** for
  suspensions, weather, and situational angles (see Step 2, agent 3).

## Step 2 — Run the three agents (whole-slate, one pass each)

Load the **entire bundle** into context once and dispatch three passes —
**not one dispatch per game**. Each agent reads every game in the bundle and
returns a structured brief per game.

1. **statistics agent** — model vs. the line, and matchup context.
   For each game: compare `model.margin`/`model.total`/`model.win_prob`
   against `market_spread`/`market_total`; flag the size and direction of
   any edge; note anything about the matchup that bears on whether the
   model's number is trustworthy this week (e.g. thin sample, early season).

2. **sports-analyst agent** — recent form / momentum / scoring-pace trends.
   For each game: read `form.home`/`form.away`; call out momentum
   (win/loss streaks), scoring-pace mismatches, and any trend that cuts
   against the raw model number (e.g. a team on a 3-game skid the model
   still favors).

3. **news agent** — injuries / suspensions / weather / situational.
   For each game: read `news.injuries` (the SportsDataIO InjuredPlayers
   feed), and **use web search in-session** to fill the rest (suspensions,
   weather, rivalry/trap-game/short-week angles) — the bundle has no
   headlines or weather. Surface anything that could move the line or
   change who covers — a named starter out, a suspension, a short week, a
   rivalry/trap-game angle, a long road trip. If nothing material, say so rather
   than inventing a narrative.

Each agent's output is a **structured per-game brief** (a dict keyed by
`game_pk`, or an ordered list — controller's choice), not free-flowing prose
covering the whole slate at once. Keep the three briefs separate; do not let
one agent's pass overwrite or paraphrase another's — synthesis needs to be
able to attribute which agent said what (see `agent_notes` below).

## Step 3 — Synthesis (desk head)

Read all three briefs. Optionally do **one round** of targeted follow-up —
e.g. ask the news agent to confirm a rumored injury's status, or ask
statistics to re-check a specific matchup — before finalizing. Do not loop
indefinitely; one follow-up round, then commit to picks.

Per game, decide whether to make a pick on each of three markets — **ML**,
**Spread**, **Total** — independently. It is valid to pick a market and
decline the others (see the null-line rule in the contract below); it is
also valid to decline every market for a game the desk has no edge on (in
which case that game contributes no pick — don't force one).

For each pick made:
- **conviction_tier**: `high` / `medium` / `low`. **"Best bets" = the `high`
  tier** — that's the whole convention; there is no separate "best bets"
  flag or field.
- **confidence**: a number in `[0, 1]`, the desk's own calibrated
  probability/strength for that pick (distinct from `conviction_tier`,
  which is the coarse bucket used for the record-tracking views).

  Note on the record itself: `desk_record`'s `ats_pct`/`total_pct` grade
  the pick against the CLOSING line, not the line the desk actually bet —
  a harder bar than the ~52.4% -110 breakeven (which is defined against
  the line you bet, not the close). `mean_clv_spread`/`mean_clv_total`
  measure the separate question of whether the desk's pick-time number
  beat the close.
- **rationale**: prose explaining the pick. **Every pick must cite at least
  one concrete fact** surfaced by an agent — a named injury, a specific
  trend, a specific number (e.g. "model has Alabama -8.9 vs. a market
  -3.5, and Georgia is starting its backup QB per the news brief after
  [starter]'s ankle injury"). A rationale that is pure narrative with no
  named fact backing it (e.g. "Alabama looks sharper lately") is not
  acceptable — reject or rework it before writing.
- **agent_notes**: a JSON object capturing which agent(s) drove the call
  and their key point(s) — this is what lets someone later audit *why* the
  desk made a pick. A reasonable shape:
  ```json
  {
    "statistics": "model -8.9 vs market -3.5, model favors home by more than the line",
    "sports_analyst": "Alabama on a 3-game win streak, avg margin +14.3; Georgia 2-1 with a narrow escape last week",
    "news": "Georgia backup QB starting (starter out, ankle) per injury report"
  }
  ```
  Keys are free-form (this field has no enforced schema beyond "valid
  JSON" — it's written straight into a `JSONB` column) but should stay
  keyed by agent so the audit trail is legible.

## Step 4 — Write

Assemble the picks into a single JSON **list** of pick objects (see exact
schema below), save it to a file, then:

```
DATABASE_URL=... PYTHONPATH=src uv run python scripts/write_desk_picks.py --in path/to/picks.json
```

This is **fail-closed**: `write_desk_picks.py` validates the entire list
first; if *any* pick has *any* problem, it prints every problem (each
naming the offending `game_pk`/field) and writes nothing to the database.
Fix the flagged picks and re-run. Only after validation passes does it
filter to pre-kickoff picks (`commence_time` strictly after "now" — a pick
for a game that has already started is silently dropped, never written)
and upsert the rest into `desk_picks`.

Re-running this script for the same `(sport, game_pk, model_version)` is an
**overwrite**, not an insert — the desk can revise a pick any number of
times before kickoff. `created_at` reflects the *first* time that key was
written, not the most recent edit.

---

## The picks-JSON contract (authoritative: `write_desk_picks.validate_picks`)

Each pick is a flat JSON object. This contract is sport-agnostic: `sport` is
`"cfb"` or `"nfl"` depending on which desk produced the pick.
`scripts/write_desk_picks.py` remains the authority regardless of sport.

### Required fields (must be present **and non-null**)

`sport`, `game_pk`, `model_version`, `commence_time`, `matchup`, `ml_pick`,
`confidence`, `conviction_tier`, `rationale`

### Required *keys* that may be `null`

`spread_side`, `spread_line`, `total_side`, `total_line` — these keys must
still be present in every pick dict, but `null` is a legitimate value: it
means "the desk is not picking this market for this game." (See the
null-line rule below for what combinations are allowed.)

### `game_date` — required by this runbook, even though the writer does not currently enforce it

**Include `game_date` on every pick.** `desk_picks.game_date` is a real
column (`db/migration_decision_desk.sql`) and `scripts/grade_desk_picks.py`
windows its query on it (`WHERE dp.sport = %(sport)s AND dp.game_date >=
%(start)s`) — a pick written with `game_date = NULL` will **never match
that filter** (SQL `NULL >= anything` is not true) and so will **never be
graded**, even though it was written successfully. `validate_picks` does
**not** currently list `game_date` in its required-field set, so a pick
missing it will pass validation and be silently written ungraded — this is
a real gap between the contract and this operational requirement (see
"Cross-check / gap" below).

Derive it the same way `generate_cfb.py::_game_date_from_commence` does —
shift the UTC `commence_time` back 8 hours before taking the date, which
maps CFB's kickoff window (roughly 15:00 UTC through 04:00–07:00 UTC the
next day) onto the correct US game day without a timezone lookup:

```python
from datetime import datetime, timedelta

def game_date_from_commence(commence_iso: str) -> str:
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    return (dt - timedelta(hours=8)).date().isoformat()
```

### Enum checks (only applied when the value is non-null)

| field | allowed values |
|---|---|
| `conviction_tier` | `high`, `medium`, `low` |
| `ml_pick` | `home`, `away` |
| `spread_side` (if non-null) | `home`, `away` |
| `total_side` (if non-null) | `over`, `under` |

(A `null` value on any of these is caught by the required-field check
above where applicable, or is simply valid where nullable — it's never
double-flagged by the enum check.)

### Numeric checks

- `confidence` — a number (int/float, not `bool`) in `[0, 1]` inclusive.
- `spread_line`, `total_line` (if non-null) — numeric (int/float, not
  `bool`).

### Null-line rule

- `spread_side` set (non-null) but `spread_line` null → **invalid**
  ("spread_side is set but spread_line is null"). A side pick needs a line
  to grade against.
- The reverse is fine: `spread_line` present with `spread_side = null` is
  **valid** — the desk saw a line and chose not to pick that market.
- `spread_side = null, spread_line = null` together (no market picked at
  all) is **valid**.
- Same rule, symmetrically, for `total_side` / `total_line`.

### Everything else

- Unknown/extra keys in a pick dict are silently ignored by the writer —
  they don't fail validation, but they also don't get stored (only the
  columns in `db._DESK_PICKS_COLS` are written). Don't rely on extra keys
  surviving the round trip.
- Problems accumulate independently per pick and per field — one bad pick
  can produce several problem lines at once (e.g. a bad `conviction_tier`
  *and* a missing `rationale` on the same game both get reported).

## Worked example — two games that PASS validation

```json
[
  {
    "sport": "cfb",
    "game_pk": 401628383,
    "model_version": "cfb-v1",
    "game_date": "2026-09-12",
    "commence_time": "2026-09-12T23:30:00Z",
    "matchup": "Georgia @ Alabama",
    "ml_pick": "home",
    "spread_side": "home",
    "spread_line": -3.5,
    "total_side": "under",
    "total_line": 54.5,
    "confidence": 0.71,
    "conviction_tier": "high",
    "rationale": "Model has Alabama -8.9 vs. a market of -3.5 — a 5.4-point edge. Georgia's backup QB is starting after the starter's ankle injury (news brief), and Alabama is riding a 3-game win streak averaging +14.3 margin (analyst brief). Taking Alabama ML/-3.5 and the under given both defenses have trended toward lower combined scoring the last two weeks.",
    "agent_notes": {
      "statistics": "model margin +8.9 home vs market -3.5; win_prob 0.71",
      "sports_analyst": "Alabama 3-0 last_n [W,W,W] avg_margin +14.3; Georgia 2-1 avg_margin +3.7",
      "news": "Georgia backup QB starting; starter out with ankle injury"
    }
  },
  {
    "sport": "cfb",
    "game_pk": 401628412,
    "model_version": "cfb-v1",
    "game_date": "2026-09-13",
    "commence_time": "2026-09-13T17:00:00Z",
    "matchup": "Kansas @ Iowa State",
    "ml_pick": "away",
    "spread_side": null,
    "spread_line": -6.5,
    "total_side": null,
    "total_line": null,
    "confidence": 0.55,
    "conviction_tier": "low",
    "rationale": "Model has a slight lean to Kansas (win_prob 0.55) off a favorable pace matchup (analyst brief: Kansas averaging 34.2 ppg over its last 3), but the spread edge is inside noise (-6.5 line vs model margin -5.1) so the desk is passing on the spread and total, taking only the moneyline flier.",
    "agent_notes": {
      "statistics": "model margin -5.1 (away) vs market -6.5; win_prob 0.55 away",
      "sports_analyst": "Kansas averaging 34.2 ppg last 3 games",
      "news": "no material injury/weather/situational factors this week"
    }
  }
]
```

Notes on the example: game 2 shows the null-line rule in action — the
market has a spread (`spread_line = -6.5`) but the desk declined that
market (`spread_side = null`), which is valid; it did **not** set a side
with no line, which would be invalid. Both games include `game_date` per
the rule above even though the writer doesn't currently require it.

## Cross-check / gap (read before treating this doc as final)

This runbook's schema was cross-checked field-for-field against
`scripts/write_desk_picks.py::validate_picks` (Task 4) as of this writing
and **matches** it exactly: required fields, nullable fields, enum values,
numeric/range checks, and the null-line rule above are all copied directly
from that function's logic, not re-derived.

The one confirmed discrepancy: **`game_date` is a `desk_picks` column and
is read by `grade_desk_picks.py`'s grading window, but it is not in
`validate_picks.REQUIRED_FIELDS`.** A picks JSON that omits `game_date`
will currently pass `validate_picks` and be written with `game_date =
NULL`, and will then never be selected by the grader's `game_date >=
%(start)s` filter — i.e. it is written but silently never graded. This
runbook makes `game_date` mandatory as an **operational** rule for the
controller (see above), but the code does not yet enforce it. If this gap
is to be closed at the code level, the fix is a one-line addition to
`REQUIRED_FIELDS` in `scripts/write_desk_picks.py` plus updating this doc's
Step 4 note accordingly — out of scope for this doc-only task.
