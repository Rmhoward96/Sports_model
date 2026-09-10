# Automated decision-desk runbook (scheduled, unattended)

This is the runbook a **scheduled Claude Code session** follows to run the
CFB or NFL decision desk end to end, unattended, the day before a slate. It
is the automated counterpart to `docs/decision-desk-runbook.md` (the manual,
in-session version) and it **supersedes that doc's methodology** where they
differ — in particular, this flow runs through GitHub Actions (not local
scripts), declines totals, treats the model as supporting context only, and
does **not** do ad-hoc web search.

The scheduled session runs on the user's machine and does **not** hold the
Supabase / SportsDataIO secrets (those live only as GitHub Actions secrets).
So every step that touches the database or a paid API goes through
`gh workflow run`; the session itself only builds the bundle artifact,
synthesizes picks locally, and commits the picks JSON.

**Repo:** `/Users/ryan/Desktop/Sports Model` · **Branch:** `main` ·
Requires `gh` (authenticated) and `uv` on PATH.

---

## Pipeline (run for exactly one sport S ∈ {nfl, cfb} per invocation)

1. **Sync main.**
   ```bash
   cd "/Users/ryan/Desktop/Sports Model"
   git checkout main && git pull --ff-only origin main
   ```

2. **Build the bundle in Actions** (has the secrets), then wait for it.
   ```bash
   gh workflow run desk-inputs.yml --ref main -f sport=S -f days_ahead=7
   # find the run id it started, then:
   gh run watch <run_id> --exit-status
   ```

3. **Download the bundle artifact.**
   ```bash
   gh run download <run_id> -n desk-bundle -D <tmpdir>
   # -> <tmpdir>/desk_bundle.json : a JSON list, one object per upcoming game
   ```
   Each game: `game_pk`, `matchup` ("Away @ Home"), `commence_time`,
   `market_spread` (home-margin convention: home favored → negative),
   `market_total`, `model{margin,total,win_prob}` (home-referenced),
   `form{home,away}{record,last_n,avg_margin,pace}`,
   `news.injuries{home,away}` (list of `{player,position,status,note}`).

4. **Synthesize picks** per the methodology below → write a JSON **list** to
   `assets/desk/picks_<S>_current.json` (e.g. `picks_nfl_current.json`).

5. **Validate locally (fail-closed).** Must print `CLEAN`:
   ```bash
   PYTHONPATH=src uv run python -c "import json,sys; from scripts.write_desk_picks import validate_picks; p=json.load(open('assets/desk/picks_<S>_current.json')); pr=validate_picks(p); print('CLEAN' if not pr else pr); sys.exit(1 if pr else 0)"
   ```
   Fix any problems before continuing — never write an invalid list.

6. **Commit + push the picks JSON** to main (only that file).

7. **Write picks to the DB** (Actions holds `DATABASE_URL`), then wait:
   ```bash
   gh workflow run write-desk-picks.yml --ref main -f picks_path=assets/desk/picks_<S>_current.json
   gh run watch <run_id> --exit-status   # expect "Upserted N ... skipped M already-started"
   ```
   `write_desk_picks.py` is fail-closed and skips any game that has already
   kicked off (pre-kickoff immutability), so running the same sport on
   multiple days each week is safe — started games are never clobbered.

8. **Rebuild the +EV board** so the desk overlay takes effect, then wait:
   ```bash
   gh workflow run build-ev-board.yml --ref main
   gh run watch <run_id> --exit-status   # prints "sport=<S> games=.. picks=.."
   ```

9. **Report** a short summary: games in slate, spread leans by tier, picks
   upserted, board picks. Note that `app.js` must be redeployed by the user
   to surface changes on the site (the desk/board data is already live).

---

## Synthesis methodology (the disciplined desk)

The desk is a 3-agent synthesis — **statistics** (model vs line),
**analyst** (form/momentum), **news** (injuries) — resolved into tiered
picks. The discipline below is load-bearing; it comes from this project's
backtest findings.

- **The ratings model does NOT beat the closing market** (backtest NO-GO:
  model margin MAE 10.21 vs line 9.78). So a large model-vs-line gap is
  **mostly noise, not a signal.** Use `model.*` only as *supporting
  context* / a tiebreaker — never as the sole reason for a lean.

- **Spread leans are driven by injuries and form:**
  - *Injuries* — a key player OUT/DOUBTFUL at an impact position (QB, top
    WR/RB, multiple starters in one unit) that the number may not fully
    reflect.
  - *Form* — record + avg margin that the market underweights early in the
    season (e.g. an unbeaten team getting points, a winless team laying a
    big number).
  Lean only where injuries or form give a real edge **and** the model at
  least doesn't contradict it. Cite a concrete fact in every rationale
  (named injury / specific record+avg-margin / the model-vs-line numbers).

- **Decline totals across the board:** `total_side = null`, `total_line =
  null` on every pick. Model totals failed the backtest and are noisy; the
  +EV engine still prices totals off Pinnacle on its own.

- **`ml_pick` is required on every game** (contract): the more likely
  winner — `home` if `win_prob >= 0.5` else `away`. This is independent of
  the spread lean (you can like the favorite to win but the dog to cover).

- **`spread_side`** is the cover lean (`home`/`away`) only where an edge
  exists; otherwise `null` (decline the spread — most games). `spread_line`
  is the home-referenced number for the chosen side: `market_spread` if
  `spread_side=="home"`, else `-market_spread`.

- **Tier honestly.** `conviction_tier` ∈ {high, medium, low}, and it drives
  the +EV desk nudge (high=1.0, medium=0.6, low=0.3 × 3.5 pts, prob move
  clamped to ±0.10). Most leans are **low** or **medium**. Reserve **high**
  for a genuine multi-signal edge (injuries + form + model all aligned on a
  mispriced side) — it is normal to have **zero** high picks in a slate.
  `confidence` ∈ [0,1]: ~0.52–0.55 low, ~0.56–0.60 medium, ~0.62–0.68 high.

- **`agent_notes`** = `{"statistics": ..., "analyst": ..., "news": ...}`,
  each a short string citing that agent's key point for the game.

### Sport-specific notes

- **CFB** — injuries are real (SportsDataIO CFB feed). When fading an
  inflated number on a big favorite, do it only when the favorite's **own**
  form is shaky (e.g. a 2-3 team laying 30+) or the dog's form is strong —
  never fade a rolling, dominant favorite on model margin alone.

- **NFL** — injuries come from nflverse's official game-day report (real,
  not scrambled; see `sportsmodel.nfl.injuries_nflverse`). The NFL market is
  **sharp**, so expect **few, small** leans. The official report posts
  Wed–Fri: a Wednesday run has full data only for the Thursday game;
  Saturday/Sunday runs pick up the Sunday/Monday reports. Do not invent
  injury narratives for games whose report hasn't posted yet — lean on
  model+form there and say the report is pending.

### The picks-JSON contract (authoritative: `write_desk_picks.validate_picks`)

Flat objects in a list. Required + non-null: `sport`, `game_pk`,
`model_version` (use `"desk-<S>-v1"`), `commence_time`, `matchup`,
`ml_pick` (home/away), `confidence` (number in [0,1]), `conviction_tier`
(high/medium/low), `rationale`. Required **keys** but nullable:
`spread_side` (home/away), `spread_line` (number), `total_side`
(over/under), `total_line` (number) — if a side is set, its line must be
non-null. `agent_notes` is free-form JSON. If validation and this doc ever
disagree, the validator wins.
