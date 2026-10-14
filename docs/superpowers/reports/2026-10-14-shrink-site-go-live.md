# Shrink game-level model numbers on the site: go-live checklist (2026-10-14)

Branch `feat/shrink-site`. Everything below is for you to run; nothing has been deployed, no workflow was changed and no code connected to the database. The fit used only committed files.

## What this changes, in plain words

Until now the sport pages and the game page showed the model's own fair moneyline, spread and total, and measured every edge against the market from that raw number. The edge audit showed that was too bold: against the closing line the CFB model adds almost nothing (coefficient +0.04, standard error 0.16) and the NFL model nothing at all. From the first load of the new pages, every game-level model number for an **upcoming** game is `market + k x (model - market)`, where k is how much of its disagreement with the close the model has earned in graded history. The same shrunk number drives the CappingAlpha columns, Edge, the moneyline Alpha Score, the Edge and Alpha sorts, the game-page Read, the Model Projection table, the Cover Probability and the dashboard's model-vs-market gaps, so they always agree. The raw projection stays on the Projected Score / Spread / Total and Win Probability cards and in a grey "Raw model projection" line under the Read.

## The k values committed with this change

Taken from `assets/game_shrink_k.json` (served model `cfb-ratings-v2`):

| Sport | Moneyline | Spread | Total |
|---|---|---|---|
| NFL | 0.000 | 0.000 | 0.000 |
| CFB | 0.033 | 0.000 | 0.043 |
| MLB | 0.000 | 0.000 | 0.000 |

cfb moneyline: n=3561, seasons [2021, 2025], unclamped slope 0.0326
cfb spread: n=7163, seasons [2016, 2025], unclamped slope -0.0158 (clamped to 0)
cfb total: n=7154, seasons [2016, 2025], unclamped slope 0.0429

NFL and MLB are **0** for all three markets: the current NFL game model (the ML sim, in place since 2026-09-28) has about six weeks of graded games and the MLB game model almost none, both under the 300-game minimum. k = 0 means the site shows the market, so NFL and MLB edges are 0 until a fit exists. `assets/game_shrink_k.json` also holds the per-season stability table (k and out-of-sample error for raw, shrunk and market for each held-out season 2018-2025).

## What you will notice

| Where | Before | After |
|---|---|---|
| Sport page, upcoming games: CappingAlpha ML / Spread / Total | the model's own numbers | the shrunk numbers (NFL and MLB: equal to the market) |
| Edge columns, Edge sort | model minus market, often several points | shrunk minus market: tiny for CFB, exactly 0.0 for NFL and MLB |
| Alpha Score (moneyline fallback), Alpha sort | from the raw win probability | from the shrunk probability; a game with no spread or only one moneyline price has none |
| Game page Read | the strongest raw divergence | usually "No model edge on this game" |
| Game page Model Projection table | raw model at the market line | shrunk line and shrunk points (see below) |
| Game page Cover Probability, spread and total | raw model at the market line | the shifted-distribution probabilities (see below) |
| A game with no market line | model numbers with an edge | the raw projection labelled "proj" / "(projection)", no edge |
| A game with a spread but no moneyline | a fair ML | "—" |
| Game page Alpha card | Pinnacle no-vig +EV numbers | unchanged on purpose: it shows Pinnacle no-vig +EV numbers, not model output |
| Past-week results, Track Record, Model vs Market and Model Projections bands, props board, +EV board, early tickets | as before | unchanged |

**Spread sign convention.** Every spread the site shows, whether market, shrunk or raw projection, uses the sportsbook convention: a minus spread is the favourite and a plus spread is the underdog. Team A -3.5 means Team A is favoured by 3.5. A shrunk favourite's line stays negative and sits between the market line and the model line; it only changes sign if the shrunk margin itself crosses zero.

### Game-page spread and total probabilities

The Cover Probability and the spread and total probabilities in the Model Projection table do not simply pull a probability toward 50%. Instead the model's margin and total distributions are **shifted so that their median lands on `market + k x (median - market)`**. The probability of each side is then read off the shifted distribution at the market line. The points shown in the table (the shrunk line, in points) use the point projection, with the same formula.

- At k = 0 the median sits on the market line, so neither side of the market line is meaningfully above 50%.
- Pushes are excluded. When a key-number lump (3, 7 and similar) sits exactly on the median, one side can read up to about 51.6%. That is still below the -110 implied 52.4%, so no phantom edge appears at standard prices.

## 1. Merge, then deploy the site yourself

1. Merge the PR into `main`.
2. From the repository root run `scripts/sync_site.sh` (it backs up your CappingAlpha folder, then copies `site/` there without the tests), then deploy that folder to Cloudflare Pages as usual.
3. Hard reload (Cmd-Shift-R). The scripts carry the cache key `20261014a`, so a normal reload also picks the new files up.

You deploy the site yourself. Nothing touches the database: there is no migration, no workflow and no secret involved.

## 2. Check it by eye (desktop first)

At 1280, 1440 and 1920 pixels wide, with a hard reload, open `nfl.html`, `cfb.html` and one game page linked from the CFB board. NFL Edge cells should read 0.0 and the Alpha Score column should be low or a dash. CFB edges should be small. The Read card should carry the grey "Raw model projection" line and say "No model edge" for NFL games. A past week's Final table should look exactly as it did. Nothing should say `NaN` or `undefined`.

## 3. Refitting k

The fit is manual: there is no scheduled job. Refit at the end of each season, or when the served game model changes.

```bash
uv run python scripts/fit_game_shrink.py --last-season 2026
```

It reads only committed files (about 15 seconds), prints the per-season stability table, and rewrites `assets/game_shrink_k.json` and `site/js/shrink-k.js` together; a test fails if the two ever disagree. Add `--dry-run` to print the tables without writing. Commit both, run `cd site && npm test`, `scripts/sync_site.sh`, and redeploy. Bump the cache key (the `?v=` on every script and on `theme.css` in the ten `site/*.html` pages, and the two constants in `site/tests/smoke.test.mjs`) so browsers load the new table.

The script fits CFB only. NFL and MLB stay explicit zeros until a follow-up adds their fit from graded game predictions, once the current NFL game model reaches 300 graded games.

## 4. If the served model changes

k is fitted for one model. `assets/game_shrink_k.json` records which (`model_version.cfb`, now `cfb-ratings-v2`). If the CFB model changes (v3, a new prior, a new `gameline.json`) the old k no longer describes it: refit before trusting the edges, or set the CFB k values to 0 in `assets/game_shrink_k.json` and rerun the script so the site shows the market until the new fit exists. The same holds for a new NFL game model.

## 5. Known limits

- **The fit is against the closing line; the site compares with the current line.** Conservative: the model gets no credit for beating stale numbers it has not been shown to beat.
- **The fitted model is the plain walk-forward, not exactly what is served.** Served CFB v2 also blends a preseason prior into each team's Elo; the walk-forward (`raw_model_predictions`) does not. The fit therefore sees a slightly weaker model in the early weeks of each season, which can only make k smaller.
- **Quarter-point and odd lines.** At k = 0 the page shows the market's own number, so the edge is exactly 0 even on a 51.75 total.
- **Edges come from the shrink, not from line shopping.** On the game page the shrunk win probability is compared with the best available price, so a moneyline can still show a small positive read when one book is out of line with the consensus. That is a price gap, not a model edge.
- **A sub-1pp phantom edge is possible at reduced-juice prices.** Because the game-page spread and total probabilities come from a shifted distribution with pushes excluded, a large key-number lump on the median can push one side to about 51.6%. At standard -110 (52.4% implied) that shows no edge, but at -105 or better a phantom edge of under one percentage point can appear.
- **Expect almost no model edges.** With k near 0 for CFB and 0 for NFL and MLB, the Read will almost always say "No model edge on this game". That is the intended result of the audit, not a fault.

## 6. If something looks wrong

Any page problem: revert the merge commit (`git revert -m 1 <merge sha>`), run `scripts/sync_site.sh` and redeploy; the data and the models were never touched. To show the raw model again without reverting, set every k to 1 in both `assets/game_shrink_k.json` and `site/js/shrink-k.js` (the agreement test needs both) and redeploy. This is not recommended: it removes the protection the audit asked for.
