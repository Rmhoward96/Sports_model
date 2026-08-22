# Serving contract — Blue Edge front-end ↔ Supabase

The Lovable app reads four **public read-only** objects (anon key + Supabase REST).
The Python pipeline (GitHub Actions) is the only writer. MLB only for now
(`sport = 'mlb'`); NFL/NBA return no rows until those models exist.

Run `db/serving_bootstrap.sql` once in the Supabase SQL Editor to create everything.

## Tables & views

### `board_picks` — the live board
One row per game×market (game lines) and per player×market (props) for the current
slate, refreshed after each odds capture. Shows the model's chosen side at the **best
book**.

| column | meaning |
|---|---|
| `sport` | `mlb` |
| `game_pk`, `game_date`, `commence_time` | game id, date, first pitch (UTC) |
| `matchup` | `"Away @ Home"` |
| `market`, `market_label` | `moneyline`/`spread`/`total`/`hits`/… and its display label |
| `player_id`, `player_name`, `team` | `0`/null for game lines; set for props |
| `pick_label` | e.g. `"Orioles ML"`, `"Orioles +1.5"`, `"Under 9.5"`, `"Over 0.5"` |
| `side`, `line` | `home/away/over/under`; the number (null for ML) |
| `odds`, `book` | best-book American price + the sportsbook offering it |
| `model_prob`, `implied_prob`, `ev` | model %, no-vig market %, EV (fractions; ×100 for %) |
| `is_pick` | `true` when `ev > 0` |

### `picks` — the tracked bet log
Insert-once when a pick first turns +EV (bet price locked); graded after the game.

Board-relevant/extra columns beyond the above: `bet_odds`, `bet_book`, `novig_bet`,
`ev_bet`, `bet_at`, `status` (`pending`/`graded`), `actual`, `result`
(`win`/`loss`/`push`), `profit` (1u at bet price), `novig_close`, `clv`, `graded_at`.

### `track_record_segments` (view)
`sport, market, wins, losses, pushes, win_pct, units, roi, avg_ev, avg_clv` — the
"By league & market" table. `win_pct/roi/avg_ev/avg_clv` are already ×100 (percent).

### `cumulative_units_weekly` (view)
`week, units, cumulative_units` — the cumulative-units chart.

## Example queries (supabase-js)

```js
// Board (MLB), full slate for the table + filters
const { data } = await supabase.from('board_picks')
  .select('*').eq('sport','mlb').eq('game_date', today)
  .order('ev', { ascending: false });
// UI: show all moneyline rows; filter other markets to is_pick === true.

// Dashboard "top edges"
await supabase.from('board_picks').select('*')
  .eq('sport','mlb').order('ev', { ascending:false }).limit(9);

// Track record — by segment
await supabase.from('track_record_segments').select('*');

// Track record — cumulative units chart
await supabase.from('cumulative_units_weekly').select('*');

// Headline aggregate (sum across segments) — compute client-side from the segments,
// or read graded picks directly:
await supabase.from('picks').select('result,profit,ev_bet,clv').eq('status','graded');
```

## Notes

- **Odds appear near game time.** Props (and thus prop board rows / picks) only populate
  within the odds-capture window (~2.5h before first pitch); early in the day the board is
  mostly game lines.
- **Best book** = the sportsbook with the most favorable price for the pick; `implied_prob`
  and CLV use the market **consensus** (median), not the best book.
- **CLV** (`picks.clv`) = consensus no-vig close − no-vig at the locked bet price; it fills
  when the game is graded.
