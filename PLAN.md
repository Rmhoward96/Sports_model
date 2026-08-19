# Multi-Sport Prediction Engine — Research + Build Plan

**Goal:** Ingest historical + current/forecast data across NBA, NFL, MLB, NHL and produce, per upcoming game and *independent of sportsbook odds*: expected team scores, expected total, win probability, and player-prop projections (as full distributions). Output is for **manual lookup** — you compare the model's expectations against the books yourself to spot edges.

This document is the synthesis of four research tracks (NBA/NFL data, MLB/NHL data, modeling methodology, pipeline architecture) plus a concrete phased build plan.

---

## 0. The core insight that shapes everything

Every one of your four outputs reduces to the **same two-engine pipeline**, per sport:

1. **Ratings engine** — how good is each team? (Elo / power ratings) → gives expected margin + win probability.
2. **Scoring engine** — how many points/runs/goals? (opportunity × efficiency) → gives each team's expected score + the total.
3. **Wrap a distribution around every mean.** A mean alone cannot price an over/under. The whole game is: **`P(over) = 1 − CDF(line)`**.
4. **Player props = the team model, disaggregated** — project a player's rate, scale by projected opportunity (minutes/snaps/ice-time/plate-appearances) and pace, adjust for the matchup, then wrap a distribution around it.
5. **Judge the model by calibration + Closing Line Value (CLV), not win rate.**

Two decisions fall out of this immediately, and they're the two most important calls in the whole project:

- **You should capture the closing line even though you don't model against it.** Your daily workflow stays purely expectation-driven — but storing the closing line lets you backtest whether your model actually has edge. Beating the closing line (positive CLV) is the industry gold standard for "is this real or am I fooling myself." Without it you'll ride a lucky streak and never know your model is noise. This is cheap to add (The Odds API free tier, or ESPN) and pays for itself the first time it saves you from a bad model.
- **The variance matters as much as the mean.** A correct projected mean with the wrong spread mis-prices *every* over/under. Fitting distributions properly is not optional polish — it's the product.

---

## 1. Data sources — recommended stack per league

**Headline: this is almost entirely free.** The strong open-source sports-data ecosystem covers ~90% of what you need; you pay only for a keyed feed of last-minute injuries/lineups if/when props demand it.

### NFL — *best free ecosystem, start here*
- **`nflreadpy`** (the 2025 successor to the now-deprecated `nfl_data_py`; returns Polars): reads pre-built nflverse Parquet releases. Play-by-play back to **1999** with EPA/WPA/CPOE/success-rate **already computed**, plus weekly player stats, rosters, **depth charts**, **snap counts**, Next Gen Stats, and **schedule fields including stadium, roof, surface, and game-time temp/wind/humidity**. No scraping, no rate limits, nightly in-season updates.
- **ESPN hidden API** (free, no key) as a daily scores/injuries cross-check.
- **Forecast weather** for outdoor stadiums: join **Open-Meteo** (free) by stadium lat/long.

### MLB — *deepest free data of the four*
- **`pybaseball`** (returns pandas; pin the version, watch for breakage when Savant/FanGraphs change): pitch-level **Statcast** (exit velocity, launch angle, xwOBA), FanGraphs, Baseball-Reference, plus **Retrosheet** (event data 1897–2025) and **Lahman** (season data 1871–2025).
- **MLB StatsAPI** (`statsapi.mlb.com`, free, no key): live scores, schedules, **probable pitchers** (`hydrate=probablePitcher`), box scores.
- **Baseball Savant**: park factors (temperature/elevation/roof/wind adjusted).
- Player projections — don't build from scratch; reuse **Steamer / ZiPS / THE BAT X** (FanGraphs) as inputs.

### NHL — *free, with one soft spot*
- **`nhl-api-py`** wrapping `api-web.nhle.com/v1` — ⚠️ the old `statsapi.web.nhl.com` was **killed in 2023-24**, so ignore any tutorial referencing it. Schedules, boxscores, play-by-play, EDGE tracking. Verify endpoints against the community `Zmalski/NHL-API-Reference`.
- **MoneyPuck** (free CSV, 2007-08→present, nightly): shot-level **xG** — the single best free NHL advanced-stats source for totals and props.
- **Natural Stat Trick** for deeper Corsi/Fenwick/xG splits (scrape politely, no API).
- **Soft spot:** projected goalies/lineups have no clean free API (DailyFaceoff / RotoWire = scrape or manual). This is the one place a cheap paid feed earns its keep.

### NBA — *works, but the ops headache; do it last*
- **`nba_api`** wrapping `stats.nba.com`: everything (advanced box scores, play-by-play from ~1996, tracking from ~2013), **but** aggressively rate-limited (~1 req/0.6–1s, realistic headers) and **it blocks datacenter IPs** — so the NBA daily pull can't run from GitHub Actions cloud runners without mitigation (run it from your laptop or a cheap residential-IP box).
- **balldontlie** (free tier + $9.99–39.99/mo) as a clean modern alternative for lineups/props/PBP.
- **ESPN hidden API** for free daily scores/injuries.

### Optional paid layer (only if props need last-minute lineups)
- **API-Sports.io** (~$19–39/mo, covers all four leagues, normalized schema) or **balldontlie GOAT** ($39.99/mo, NBA lineups+props) or **SportsDataIO** (~$25–599/mo, best injuries/depth-charts/projected-starters).
- **Total to run a serious pipeline: $0 to ~$75/mo.** Sportradar ($10k+/mo) is enterprise overkill — ignore it.

### ⚠️ Do NOT scrape as a primary feed
Basketball/Pro-Football/Hockey-Reference (Sports Reference) actively block bots via Cloudflare (**≤20 req/min → 24-hour IP ban**) and their ToS prohibits automated bulk scraping. Everything they offer is available legally free via nflverse / nba_api / MoneyPuck / Retrosheet. Use them for manual validation only.

---

## 2. Architecture — the lean solo-builder stack

**Recommended stack, one line:**
> Python 3.11 + Polars · `nflreadpy` / `nhl-api-py` / `pybaseball` / `nba_api` · httpx + tenacity + Pydantic · **Parquet-on-disk** raw layer · **DuckDB** for backfill/crunch · **Supabase Postgres** for serving/daily state · **GitHub Actions cron** for daily ingest · **Streamlit** for lookup.

**Deliberately avoid:** Airflow, Spark, Kafka, BigQuery, a custom web front-end, and a universal cross-sport fact table. Every one is a time sink with no early payoff at your data volume (the entire multi-sport backfill is **tens of GB — it fits on your laptop**).

### Two-engine storage (not one database)
- **DuckDB + Parquet** = backfill and heavy analytical crunch, on your laptop, free. Parquet files partitioned by `sport/season` are your durable, re-runnable source of truth.
- **Supabase Postgres** = the always-on serving layer holding *current* data + model predictions that the Streamlit app (and you) query.

Flow: **raw API → Parquet (bronze) → DuckDB transforms (silver → gold) → push gold predictions to Supabase → Streamlit reads Supabase.**

### Medallion schema, shared dims + per-sport facts
Share the *dimensions* across sports (teams, players, games, seasons); specialize the *facts* (NBA play-by-play and MLB pitch data have genuinely different grain — don't force them together).

- **bronze/raw** — as-ingested, append-only, stamped with `source` + `ingested_at`. Never edit; it's your replay log.
- **silver/clean** — deduplicated, typed, conformed IDs, one row per real event. Upserts + ID-crosswalk resolve here.
- **gold/features** — model-ready rolling features + the `gold_*_predictions` tables the app reads.

Every dimension row stores its own canonical ID **plus** a `source_ids` JSONB map of every provider's ID for that entity — so the crosswalk lives *in* the dimension.

### The two hard parts (budget real time for these)
1. **ID crosswalk across providers** — `nba_api` numeric IDs vs BBRef slugs; MLBAM vs FanGraphs vs Retrosheet; NFL GSIS vs PFR vs ESPN. **Lean on existing crosswalks first**: nflverse `load_ff_playerids()`, baseball's Chadwick register (`pybaseball.playerid_lookup`). Then fuzzy-match the residual (name + birthdate + team) into a **manual-review override table** you persist so you never re-solve a match.
2. **Idempotency** — both backfill and daily jobs must be safely re-runnable to identical end state. Upsert on natural keys (`ON CONFLICT … DO UPDATE`); keep backfill **resumable** via an `ingest_log`; **re-pull the last 3–7 days of finals every run** because sports stats get restated (a corrected assist, a reversed hit/error days later). Stamp `model_version` on every prediction so you compare model generations without mutating history.

---

## 3. Modeling — method per sport × target

### Ratings engine (who's better)
- **Elo** is the right first engine for **all four sports** — needs only scores + dates. Use the FiveThirtyEight refinements: margin-of-victory multiplier, season-to-season mean reversion (NBA ~0.75 retention, NFL ~0.667), and sport-specific adjustments (**NFL: starting QB is huge**, plus travel/rest).
- Cross-check with **Massey** (least-squares on margins) and **Pythagorean expectation** (regression-to-mean prior; exponent ≈1.83 MLB, ≈13.91 NBA).
- **RAPM** (ridge-regularized adjusted plus-minus) for **NBA and NHL** to build bottom-up team strength from player parts and handle injuries/lineup swaps.

### Scoring engine (how many) + its distribution
| Sport | Score/total method | Distribution |
|---|---|---|
| **NHL** | shot volume × xG, goalie-adjusted | **(bivariate) Poisson / Dixon–Coles** — low-scoring, correlated |
| **MLB** | lineup vs. starter + bullpen, park-adjusted | **Poisson or Negative Binomial** (runs overdisperse) |
| **NBA** | **pace × efficiency** (poss × pts/100) | **Normal** on margin/total (SD ≈ 11–12) |
| **NFL** | drives × EPA/play | **Normal** on margin/total (SD ≈ 13–14) |

### Win / cover / total probability
Derive from the score model so everything is internally consistent: `P(home win) = Φ(μ/σ)` (Normal) or sum the Poisson score matrix; `P(cover s) = 1 − Φ((s−μ)/σ)`; `P(over T)` from the joint distribution. Optionally add a direct **XGBoost/LightGBM** classifier (what nflfastR uses) and reconcile. **Always calibrate** — a raw 70% must actually win ~70% (Platt/isotonic).

### Player props (the DFS pipeline)
`mean = baseline rate × projected opportunity × pace × matchup`, then wrap a distribution:
- **Small-integer counts** (rebounds, assists, made-3s, hits, SOG, receptions, TDs): **Poisson** or **Negative Binomial** (for the fatter real-world tail).
- **Continuous / large counts** (yards, NBA points): **Normal/Gamma with an empirically fit SD**, then `P(over) = 1 − Φ((line−μ)/σ)`.
- **Phase 2 upgrade: Monte-Carlo game simulation** — simulate minutes/pace/correlated usage together, tally each stat, read `P(over)` off the empirical distribution. Keeps props consistent with the team total and gets the tails right.
- **Projected opportunity (minutes/snaps/ice-time/PA) is the hardest and most important input** — a teammate's injury redistributes it (usage can't exceed 100%).

### Evaluation (where amateurs go wrong)
- Score with **log-loss / Brier + calibration plots**, not accuracy.
- **Breakeven at −110 is 52.4%** (`110/210`) — raw win rate near 50% is worthless without price edge.
- **Remove the vig** before comparing: no-vig prob = `imp_i / (imp_home + imp_away)`. Compare *your* prob to the *no-vig* market prob.
- **Track CLV** against closing lines across a large sample — that's the real validation signal. Stake with **fractional Kelly** once calibrated.
- **Pitfalls that sink models:** overfitting (regularize; time-based splits, never random shuffles), **data leakage** (freeze features at prediction time — no post-game minutes/confirmed-inactives), pricing props off a point estimate (always model variance), and stale inputs (a scratched **goalie**, late **QB** ruling, or changed **starter** dwarfs any model refinement).

---

## 3b. MLB player props — locked MVP set + matchup methodology

**Props covered:** Hitters — Hits, Total Bases, Home Run (Y/N), Hits+Runs+RBIs. Pitchers — Strikeouts, Outs Recorded, Hits Allowed.

### One engine, seven props
All seven derive from a single artifact: **`feat_matchup_pa_outcomes`** — the per-plate-appearance outcome vector `{out, BB, K, 1B, 2B, 3B, HR}` for each batter vs. the opposing starter, plus projected PA. Hitter props read one batter's row; pitcher props aggregate the starter's confrontations across the opposing lineup.

| Prop | Derivation | Distribution |
|---|---|---|
| Hits | per-PA `p_hit` summed over PA | Poisson-binomial (≈ Poisson) |
| Total Bases | per-PA base value `1·1B+2·2B+3·3B+4·HR`, summed | convolution / sim (`empirical`) |
| Home Run (Y/N) | `1 − (1−p_hr)^PA` | Bernoulli |
| Hits+Runs+RBIs | H direct; R/RBI need on-base + lineup context | marginal now → Monte-Carlo Phase 4 |
| Strikeouts | `Σ p_k·PA` over opposing lineup | Negative Binomial |
| Hits Allowed | `Σ p_hit·PA` over opposing lineup | Poisson / NB |
| Outs Recorded | projected BF × out rate, capped by workload/hook | Normal / NB |

### How the per-PA vector is built (talent × talent, NOT head-to-head history)
1. **Odds-ratio / log5 blend** — `Odds(matchup) = Odds(batter) × Odds(pitcher) / Odds(league)` per outcome, from `feat_batter_profile` × `feat_pitcher_profile` (both split by handedness).
2. **Platoon split** — batter hand vs. pitcher hand (`vs_hand` on the profile tables).
3. **Pitch-mix / batted-ball matchup** — from `fact_mlb_pitch` (Statcast), large-sample, the modern replacement for literal BvP.
4. **Talent prior** — Steamer/ZiPS/THE BAT X projections as stable inputs.
5. **Context** — park factor (`ref_mlb_park_factors`), weather (`ref_game_weather`).
6. **Opportunity** — projected PA from `feat_batting_order_pa` (lineup slot is the biggest counting-stat lever).

**Literal batter-vs-pitcher history (`feat_bvp_rollup`) is stored for display only, heavily regressed, near-zero model weight** — small-sample BvP is a documented false signal that fails out-of-sample. Two context-heavy props get honest treatment: **HRR** (R/RBI depend on baserunners/lineup) and **Outs Recorded** (depends on manager hook, not just matchup) — rough marginals in Phase 1, Monte-Carlo game sim in Phase 4.

---

## 4. Phased build roadmap

**Guiding principle: one sport, end-to-end, before any second sport.** A thin vertical slice (ingest → warehouse → one dumb model → lookup) teaches more than a wide ingest layer with no model on top.

**Start with NFL** — nflverse gives clean pre-built Parquet (no scraping, no rate limits), the best-documented modeling ecosystem, and the fewest games (trivial daily ops). It's the gentlest possible on-ramp.

### Phase 0 — Skeleton (½ day)
Repo + Python env (`uv`), confirm Supabase project, DuckDB installed, `.env` secrets, one GitHub Actions workflow running a hello-world that connects to Supabase.

### Phase 1 — NFL vertical slice = the MVP
1. Backfill NFL with `nflreadpy` → Parquet (schedules, pbp, player stats, rosters, player-ID map).
2. DuckDB builds bronze → silver → gold (plain SQL first; add dbt-core later if it earns its place). Add basic data-quality checks (freshness, uniqueness, not-null).
3. **Baseline model, deliberately simple** — Elo for winner + a pace/EPA-lite total, wrapped in a Normal distribution → emit `P(win)`, `P(cover)`, `P(over)`. The point is to exercise the gold→prediction path, not to be good yet.
4. Daily GitHub Actions job: pull upcoming schedule + injuries, generate predictions, write `gold_game_predictions` to Supabase.
5. Output to a **CSV / Google Sheet** first, then a minimal **Streamlit** lookup. **Ship it — this is a working product.**

### Phase 2 — Harden the ingest
Idempotent upserts everywhere, resumable backfill (`ingest_log`), restatement re-pull window, retries/backoff (`tenacity`), data-quality gates, watermarks. Make the pipeline boring and trustworthy before scaling breadth. Add the closing-line capture + a CLV tracking sheet here.

### Phase 3 — Add sports one at a time
**NHL** (`nhl-api-py` + MoneyPuck) → **MLB** (`pybaseball`, Statcast chunking, Chadwick crosswalk) → **NBA last** (fight the rate-limit/bot-protection headache only once the pipeline is mature). Each reuses the dimension + medallion pattern; only the fact tables and ingest adapter are new.

### Phase 4 — Modeling depth + player props
Layer XGBoost/LightGBM, EPA/xG/RAPM features, and **player props** (`gold_player_prop_predictions`) with Monte-Carlo simulation. Add `model_version` tracking + a backtest/accuracy log so you can tell if a new model is actually better.

### Phase 5 — Polish (only if warranted)
Nicer Streamlit or a real web app; migrate cron → Prefect only if you hit genuine multi-step-dependency pain. Don't do this preemptively.

---

## 5. Open decisions for you

1. **MVP sport** — plan assumes **NFL first** (easiest data + ops). If you'd rather prove the *distribution/props* machinery on a cleaner scoring model, **NHL** (Poisson + free MoneyPuck xG) is the alternative. Your call.
2. **Paid feed now or later?** Plan defers it — free sources cover Phases 1–3. Only NBA and last-minute props force the question. Recommend deferring until you hit the wall.
3. **Transforms: plain SQL vs dbt-core vs SQLMesh** — plan starts with plain SQL in DuckDB and adds dbt-core when the transform layer grows. SQLMesh is a defensible greenfield alternative if you want compile-time SQL validation.
4. **Odds capture** — strongly recommended to add in Phase 2 for CLV backtesting even though your daily workflow ignores odds.
