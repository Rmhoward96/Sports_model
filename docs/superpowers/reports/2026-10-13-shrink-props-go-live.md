# Shrink prop probabilities: go-live checklist (2026-10-13)

Branch `feat/shrink-props`. Everything below is for you to run; nothing here has been executed. No migration was run, nothing was pushed, no workflow was dispatched, and no code ever connected to the database during the build.

## What this changes, in plain words

Until now the prop board trusted the model's probability. The audit showed that was too bold: the model's prop leans claimed 64.8% and hit 51.8%. From the first fit on, every prop pick is priced on a probability **pulled toward the market's price**: `market + k x (model - market)`, where k is how much the model has earned from its own graded history (the audit points to about 0.15). The same shrunk number drives the pick, the EV, the edge and Alpha Score on the site, the +EV parlays and the pick ledger. The sims and models are not touched, and the raw record on the NFL prop-lines pages stays as it was.

## What you will notice

- **Far fewer prop picks.** That is the intended result, not a fault.
- **MLB props are price-only** in this part (k = 0 for every MLB market), so there will be almost no MLB prop picks.
- **Best parlays are often empty or game-line only.** A prop leg needs a shrunk probability of at least 0.55 to qualify, and few will reach it.
- **Alpha Score and edge numbers on prop rows get much smaller**, because they are now computed from the shrunk edge.

| Where | Before | After |
|---|---|---|
| +EV props board (NFL, MLB) | the model's probability decides the pick | the shrunk probability decides: far fewer picks; MLB props are price-only, so almost none |
| Alpha Score / edge / confidence on prop rows | from the model's raw edge | from the shrunk edge (much smaller, honest) |
| +EV parlays | prop legs often qualify | prop legs rarely reach 0.55; parlays may be empty or game-line only |
| Pick ledger (`ev_prop`, `parlay_leg`) | model probability | the shrunk probability that made the pick |
| `ev_prop_picks` rows | `model_prob` | `model_prob` (shrunk), plus `raw_model_prob` and `shrink_k` for audit |
| NFL prop-lines / grades pages | raw model | unchanged (they are the fit's source) |

## 1. Run the migration

Paste `db/migration_shrink_props.sql` into the Supabase SQL editor and run it once (safe to re-run; it adds two empty columns to `ev_prop_picks`). **The board works before it is run:** until the columns exist it simply leaves them out and prints a `WARN` line. So this step is not urgent and can be done before or after the merge.

Check it worked (should return two rows):

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'ev_prop_picks' AND column_name IN ('raw_model_prob', 'shrink_k');
```

## 2. Merge, then dispatch the first fit at once

Merge the PR. There is no k file yet, so every prop board build that runs before the first fit is **price-only** (k = 0): the model is ignored and a prop is a pick only when two books' best prices disagree enough to be +EV on their own (rare). Right after the merge, Claude dispatches the fit workflow **once**:

`gh workflow run fit-shrink-k.yml --ref main` (or on GitHub: Actions, `fit-shrink-k`, Run workflow, branch `main`).

**Watch this first run.** It is the first time the fit's database query (`GRADES_SQL`) touches the real database; in the build it was only tested against fakes and against the migration files. It takes about a minute. When it is green:

- The log ends with a line like `[fit-shrink-k] wrote shrink_k.json: pooled k=0.15 on n=... (version nfl-sim-ml-v2); ...-shrink-gate.md: status=collecting scored=...`.
- A commit `chore: weekly shrink-k refit [skip ci]` appears on `main` with two files: `assets/shrink_k.json` and `docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md`.
- The next prop board build (or `gh workflow run build-ev-props.yml --ref main`) picks the file up.

**A red run is safe: it changes nothing.** The script only reads the database, writes the files only after everything succeeded, and commits nothing on failure. The board simply stays price-only until a fit succeeds. Read the log; the error line names the cause (database unreachable, a missing table or column, or a read that returned fewer graded legs than the committed file, see below). Tell Claude and it will fix the query.

How to read `assets/shrink_k.json`: `pooled.k` and `pooled.n` are the overall number and the legs behind it. If `pooled.n` is under 300 every k is 0 (not enough evidence yet), by design. `markets.nfl.<market>.k` is what each market uses; a market with `n` = 0 (for example `pass_tds`, which has no graded history) stays price-only. MLB is 0 everywhere in this part.

## 3. Every Tuesday, and the pass test

`fit-shrink-k` runs **Tuesdays at 15:30 UTC** (after Monday night is graded), refits k on all graded legs, and commits the file and a dated report in `docs/superpowers/reports/`. The report shows the current k per market with n, how many legs have been scored so far, the running accuracy gain over the market and the calibration slope, and the status.

The pass test is judged **once**, at the first weekly report with at least **2,000 scored legs** (each week is scored with a k fitted only on earlier weeks, so there is no peeking). Until then it says COLLECTING and nothing can pass or fail. **That count may not be reached this season:** only the three markets the board serves are scored, and the first 300 legs are used up before any week can be scored. If the season ends first, collection continues next season. After it is judged, the verdict never changes in later reports.

- **Pass** (the shrunk model beats the market price by at least 0.001 Brier with the lower confidence bound above 0, and its calibration slope is between 0.8 and 1.2): shrunk model prop picks stay. This does not prove profit; closing-line value in the pick ledger judges that.
- **Fail:** you and Claude decide together what happens to model prop picks. Nothing is removed automatically; picks keep using the shrunk probabilities until then.

## 4. If something looks wrong

- **Board has no prop picks at all:** expected right after the merge (price-only) and when k is small. Check `assets/shrink_k.json` exists on `main` and `pooled.n` is at least 300.
- **A `::warning::shrink: ... using k=0` line in a build log:** the k file is missing, unreadable, or lacks a market. The board is price-only for those markets and keeps running. It never falls back to the raw model.
- **A weekly run goes red with "fewer eligible legs than the committed file":** the number of graded legs dropped, for example after a data correction or a truncated read. The refit refuses to overwrite the file in that case, so the old k stays in use. This needs Claude's attention (it will check whether the drop is real before the file is allowed to move); do not just re-run it.

## 5. Rollback

- **Delete or replace `assets/shrink_k.json`.** Deleting it makes the board price-only (k = 0). It does **not** bring back the raw model: a missing file means price-only, never the old behaviour. To use a k you choose, replace the file with one of your own, for example:

  ```json
  {"markets": {"nfl": {"rush_yds": {"k": 0.1}, "reception_yds": {"k": 0.1}, "receptions": {"k": 0.1}, "pass_tds": {"k": 0}}}}
  ```

  k is a number from 0 (ignore the model) to 1 (trust the model fully). MLB stays 0 whatever the file says.
- **Disable the `fit-shrink-k` workflow** (Actions, `fit-shrink-k`, the three dots, Disable workflow) so the next Tuesday does not overwrite your hand-set file.
- The two new database columns can stay; they are harmless.
- Putting the board fully back to the old raw-model behaviour means reverting the merge on GitHub (Revert button on the PR); ask Claude first, since the raw model was shown to be overconfident.

## 6. What was not changed

The sims and props-ML models, `nfl_prop_lines` and `nfl_prop_grades`, the game-line boards and early-week tickets, all game-level numbers on the site (a separate Part B follows), the pick thresholds (a pick is still a positive EV up to 25%, with a tiny floor of 0.000000001 so a zero-value tie from rounding is never a pick), and all site code.

## 7. Known limits

- **Pinnacle:** the market probability used in the fit comes from the best prices across all books on the prop-lines table (Pinnacle included); the board's market reference uses the best non-Pinnacle prices. The formula is identical, so k is internally consistent, but the two are not the same number to the last decimal.
- Legs are matched to their model version by the sim row whose mean equals the line row's projection. Legs whose sim row no longer exists count only in the all-versions pool. Until 300 legs of the current model (props-ML v2) are graded, k is fitted on all versions pooled, which includes the older model's legs.
- The audit already saw the existing graded weeks, so the test is not fully blind on that data. The walk-forward scoring and fixed thresholds limit the harm.
