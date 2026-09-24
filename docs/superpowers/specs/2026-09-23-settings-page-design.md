# Settings page (books, unit, min EV, Kelly) — design

Date: 2026-09-23 · Status: approved in chat ("yes that looks right"; book logic =
re-price at my books; unit applies everywhere $ is shown; Kelly: bankroll $
setting, fraction selectable default Quarter).

## Settings (per browser, localStorage key `ca-settings`, no accounts)
| Setting | Control | Default | Valid |
|---|---|---|---|
| books | tile grid (logo + name), toggle; Select all / Clear | all 10 US books | non-empty subset of the list below |
| unit | $ input + presets $10/$25/$50/$100 | 10 | 0 < unit ≤ 100000 |
| bankroll | $ input | 1000 | 0 < bankroll ≤ 100000000 |
| kelly | Full / Half / Quarter / Eighth | 0.25 (Quarter) | {1, 0.5, 0.25, 0.125} |
| minEv | slider 0–20% step 0.5 | 0 | 0 ≤ minEv ≤ 20 |

US books (key → name): draftkings DraftKings, fanduel FanDuel, betmgm BetMGM,
williamhill_us Caesars, fanatics Fanatics, espnbet ESPN BET, hardrockbet Hard Rock
Bet, thescore theScore Bet, bet365 Bet365, ballybet Bally Bet. (`caesars` in data
is treated as Caesars = `williamhill_us`.) Changes save instantly; "Reset to
defaults" button. Invalid/missing stored values fall back per field.

New page `settings.html` (`data-page="settings"`), nav link "Settings" after Track Record.

## Effects
- **+EV page**: each game-line/prop pick is re-priced at the best price among
  the selected books (`ev_pick_prices_current.prices`; if a pick has no entry
  there, fall back to its own best_book/best_price), EV recomputed
  `prob × decimal − 1` (game lines: `true_prob`; props: `model_prob`). A pick is
  shown only if some selected book prices it and its EV ≥ minEv/100 (and > 0).
  Parlays shown only if the ticket's book is selected and ticket EV ≥ minEv/100.
  Header counts/top EV and the sort use the re-priced values. New column
  **KELLY** per pick (and on each parlay card): `f = EV / (decimal − 1)`,
  stake = `kelly × f × bankroll`, shown `$X (X.Xu)` where u = stake / unit
  (dollars rounded to $1 when ≥ $10, else cents; units one decimal); nothing
  shown when f ≤ 0.
- **Game tiles**: best moneyline among selected books (from
  `game_moneylines_current.home_prices/away_prices`), else model fair line.
- **Unit size**: every $ amount scales by `unit / 10` — Predictions and +EV
  profit trackers (cards, chart, total wagered), Props by game, graded
  parlays, parlay "to win".

## Data (one migration `db/migration_settings_prices.sql`)
- `ev_pick_prices_current`: one row per current +EV pick: `kind ('game'|'prop'),
  sport, game_pk, market, side, player_id, line, prices jsonb {book: american}` —
  each US book's latest pre-kickoff capture within 48h at the pick's line
  (game lines via `ev_best_lines`; props via the pick's line + normalized player
  name). Verified live: 99 rows, 78 ms.
- `game_moneylines_current` gains `home_prices`, `away_prices` jsonb.
