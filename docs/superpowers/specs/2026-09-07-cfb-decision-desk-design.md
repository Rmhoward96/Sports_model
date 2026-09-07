# CFB Decision Desk — Design Spec

**Status:** Draft for review
**Date:** 2026-09-07
**Sport:** CFB first (method later extends to NFL)

## Why

The ratings model has **no edge** vs the closing line, and we now know exactly
why (measured over 6,805 games): the model's margin MAE is 13.36 vs the line's
12.37, it correlates 0.90 with the line, and its disagreements with the line
are right only ~48–50% — noise, not signal. The line already contains
everything the model uses (team strength) **plus** what it can't see:
injuries, suspensions, weather, matchup specifics, motivation, sharp money.

To beat the line you must add information the ratings model lacks. The decision
desk is three specialists that inject exactly that, then argue to a pick.

## What it is

A per-slate **CFB decision desk**: three LLM agents that each analyze a
different information stream, exchange takes, and produce **ML / Spread / Total
picks** per game — each with a side, the market number it's taken against, a
confidence, and a written rationale. Picks are logged **before kickoff** and
graded **forward** against both the result and the **closing line (CLV)**.

**It is assistive until proven.** A news-reading desk cannot be cleanly
backtested (historical news leaks outcomes), so its only honest validation is
forward paper-trading. Until it demonstrably beats the line over a real sample,
it is a decision aid with rationale — never sold as an edge.

### Locked decisions (from review)

- **Provider:** SportsDataIO (key `SPORTSDATA_API_KEY`, a GitHub secret).
- **Execution:** **on-demand / manual** — the desk runs inside a Claude Code
  session that the user triggers (I drive the three agents as subagents over a
  prepared input bundle and write the picks). There is **no scheduled
  Claude-API workflow and no `ANTHROPIC_API_KEY` in CI** — the LLM reasoning
  happens in-session. Only the *grading* is automated (it needs no LLM).
- **Cadence:** one pass per week, run **midweek**, when the user triggers it.
- **Coverage:** **every game, tiered** — a pick on every FBS game, each tagged
  with a conviction tier (high / medium / low), with a highlighted "best bets"
  (high-conviction) subset. Full graded sample for the CLV test *and* the
  strongest looks surfaced.
- **Scope:** CFB first.

## The three agents

1. **Statistics agent** — reads our own model output (`predictions_current`:
   model margin/total/win-prob, the forward-looking priors) and matchup
   context (head-to-head, ratings gaps). Frames where the model sits vs the
   market number. *Source: our existing DB/assets — no new dependency.*
2. **Sports analyst agent** — recent games, form, momentum, scoring/pace
   trends, ATS trends, blowout/close-game patterns. *Source:
   `schedules.parquet` + results — no new dependency.*
3. **News agent** — per-game injuries, suspensions, depth-chart changes,
   weather, and situational motivation (rivalry, letdown/lookahead, travel).
   *Source: a NEW paid provider (below).*

### How they "communicate" (bounded, not open-ended)

Each agent first produces a **structured brief** from its stream (parallel).
A **desk-head synthesis** step then reads all three briefs, may issue **one
round of targeted follow-up questions** to any agent, and emits the final
picks with a rationale that cites which agent drove each call and where they
disagreed. One follow-up round captures "they communicate" without unbounded,
expensive debate; more rounds are a later option if the forward test justifies
the cost.

## Data sources & new dependencies

- **News/injuries/weather:** a paid sports-data provider behind a pluggable
  adapter (mirroring our `cfbd`/`odds`/`espn` adapters). **Recommended:
  SportsDataIO** (CFB injuries, news, and weather in one API). The key is a
  GitHub Actions secret (`SPORTSDATA_API_KEY`), provisioned by the user, never
  seen by the assistant or committed — same handling as `CFBD_API_KEY`. The
  adapter isolates the provider so it can be swapped.
- **LLM:** the three agents + synthesis run as **subagents inside the
  on-demand Claude Code session** — no Anthropic API key or CI budget. The
  session's controller prepares each agent's input bundle (files), dispatches
  the agents, and writes the resulting picks. Grading is pure code (no LLM).
- **Closing line:** the CFBD lines we already ingest (`lines.parquet` /
  live ESPN pickcenter) — the benchmark the picks are graded against.

## Data flow

```
per upcoming CFB game:
  statistics_brief  <- predictions_current + matchup context      (our data)
  analyst_brief     <- recent form / momentum / ATS trends        (our data)
  news_brief        <- injuries / weather / situational           (paid API)
        |
  desk_head synthesis (reads 3 briefs, 1 follow-up round)
        |
  -> ML pick, Spread pick (+ market line), Total pick (+ line),
     confidence, rationale, per-agent notes
        |
  store in `desk_picks` (pre-kickoff, immutable once the game starts)
        |
  grade forward -> `desk_pick_results`: correct? covered the line? CLV?
```

## Storage & serving

- `desk_picks` (new table): game_pk, sport, game_date, matchup, ml_pick,
  spread_side + spread_line, total_side + total_line, confidence,
  conviction_tier (high/medium/low), rationale, agent_notes (json),
  model_version, created_at. Written pre-kickoff; idempotent per
  (game_pk, model_version). A pick on every FBS game; the tier drives the
  "best bets" highlighting on the site and lets the CLV test segment by
  conviction.
- `desk_pick_results` (new table/view): joins picks to finals + the closing
  line → ml_correct, spread_cover (vs line), total_result (vs line), CLV.
- Front-end: a **Decision Desk** section on the CFB page showing each game's
  picks + rationale, and a running record (accuracy, ATS-vs-line %, CLV).

## Validation gate (forward-only, CLV)

- Log every pick **before kickoff**; never regrade a pick after the game.
- Grade against the result AND the closing line. Track, per confidence tier:
  ML accuracy, ATS-vs-line %, total-vs-line %, and mean CLV.
- **The bar:** ATS/total win% > ~52.4% and/or positive CLV over a meaningful
  sample (target ≥ ~a half-season of graded picks) before the desk is framed
  as an edge. Below that, it stays labeled "assistive — not a proven edge,"
  reported straight either way.

## Cost

Because the desk runs **on-demand in a Claude Code session** (not a paid CI
loop), there is no standing API bill — the LLM cost is the session's own
subagent usage on the weeks the user runs it, once per week. The only external
cost is the SportsDataIO subscription (the user's key). Grading runs in CI but
is pure code (free). To keep a single run tractable over ~50 FBS games, the
briefs are batched (one agent pass over the whole slate's bundle, not one
dispatch per game) rather than fanned out per game.

## Non-goals

- **No backtesting the desk** — news leakage makes it invalid; validation is
  forward only.
- **NFL** — later, after CFB proves the method.
- **No auto-betting / no bet sizing** — the desk outputs opinions, never places
  or stakes wagers.
- **The desk does not change the ratings model** — it's an additive layer that
  consumes model output; the model keeps running as-is.

## Risks

- **LLM confabulation:** agents can produce confident, well-written, and
  non-predictive narratives. Mitigation: structured outputs, require each pick
  to cite concrete facts (a named injury, a specific trend), and the CLV gate
  as the ruthless arbiter — rhetoric doesn't move the record.
- **News timeliness/leakage-in-reverse:** picks must be logged before kickoff
  and immutable, or the record is worthless.
- **The bar is very high:** closing lines are efficient; most public news is
  already priced. Beating them is genuinely hard and may not happen — the
  forward test will tell us honestly.
- **Cost creep:** capped per run; revisited if the forward test shows no edge.

## Resolved (review complete)

- Provider → **SportsDataIO**.
- Cadence → **one midweek pass per week**, user-triggered.
- Coverage → **every game, tiered** (best-bets = high-conviction subset).
- Execution → **on-demand in-session** (no CI Claude-API loop).

Implementation plan follows in
`docs/superpowers/plans/2026-09-07-cfb-decision-desk.md`.
