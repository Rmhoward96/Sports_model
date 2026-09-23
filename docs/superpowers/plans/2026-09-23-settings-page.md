# Settings Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Settings page (books, unit size, bankroll, Kelly fraction, min EV) whose choices re-price the +EV page at the user's books, add Kelly stakes, filter by min EV, pick tile moneylines from the user's books, and scale every $ amount by the unit.

**Architecture:** One SQL migration exposes per-book prices for current +EV picks and per-book moneylines for tiles. Everything else is client-side in CappingAlpha `app.js`: a small settings store (localStorage), a new settings page, and pure helpers applied where picks/tiles/P&L render.

**Tech Stack:** Supabase Postgres views; vanilla JS (`/Users/ryan/Desktop/CappingAlpha/app.js`, not a git repo).

**Spec:** `docs/superpowers/specs/2026-09-23-settings-page-design.md`

## Global Constraints
- The user runs every Supabase migration; never apply DDL.
- Settings are per browser: localStorage key `ca-settings`; every read/write wrapped in try/catch; the site must work with storage unavailable (defaults).
- Defaults: all 10 US books, unit $10, bankroll $1000, Kelly 0.25, minEv 0 — with defaults the site must look/behave exactly as today (except the new Kelly column).
- Kelly: f = EV / (decimal − 1); stake = kelly × f × bankroll; display `$X (X.Xu)`; nothing when f ≤ 0.
- All new Supabase reads `.catch(() => [])`; missing views degrade to today's behavior.
- Cache version for this plan's final state: `app.js?v=20260923m` in all `*.html` (+ new settings.html).

---

### Task 1: Migration

**Files:** Create `db/migration_settings_prices.sql` (Sports Model repo). Commit on branch `feat/settings-page`.

- [ ] Write exactly:

```sql
-- =============================================================================
-- Settings page data: per-book prices for current +EV picks, and per-book
-- moneylines for the game tiles, so the site can re-price at the user's books.
-- =============================================================================
-- Idempotent -- safe to re-run. Run in the Supabase SQL Editor. Requires
-- ev_best_lines (migration_ev_best_lines.sql) and american_to_decimal
-- (migration_pnl_views.sql). US books = serving.board.MAJOR_BOOKS minus Pinnacle.

-- One row per current +EV pick: each US book's latest pre-kickoff capture
-- (within 48h) at the pick's own line, as {book: american_price}.
CREATE OR REPLACE VIEW ev_pick_prices_current AS
  WITH gpicks AS (
    SELECT e.sport, e.game_pk, e.market, e.side, b.line
    FROM ev_current e LEFT JOIN ev_best_lines b USING (sport, game_pk, market, side)
    WHERE e.is_pick
  ), glast AS (
    SELECT DISTINCT ON (o.game_pk, o.market, o.side, o.book) o.game_pk, o.market, o.side, o.book, o.line, o.price
    FROM odds_snapshot o
    WHERE o.game_pk IN (SELECT game_pk FROM gpicks) AND o.market IN ('moneyline', 'spread', 'total')
      AND COALESCE(o.player_name, '') = '' AND o.captured_at <= o.commence_time
      AND o.captured_at > now() - interval '48 hours' AND o.price IS NOT NULL AND o.price <> 0
      AND o.book = ANY (ARRAY['draftkings','fanduel','fanatics','hardrockbet','thescore','espnbet',
                              'williamhill_us','caesars','bet365','betmgm','ballybet'])
    ORDER BY o.game_pk, o.market, o.side, o.book, o.captured_at DESC
  ), g AS (
    SELECT 'game'::text AS kind, p.sport, p.game_pk, p.market, p.side, NULL::text AS player_id, p.line,
           jsonb_object_agg(l.book, l.price) AS prices
    FROM gpicks p JOIN glast l ON l.game_pk = p.game_pk AND l.market = p.market AND l.side = p.side
     AND (p.market = 'moneyline' OR l.line = p.line)
    GROUP BY 1, 2, 3, 4, 5, 6, 7
  ), ppicks AS (
    SELECT game_pk, player_id, market, side, line,
           CASE market WHEN 'rec_yds' THEN 'reception_yds' ELSE market END AS omarket,
           lower(regexp_replace(regexp_replace(trim(player_name), '[.''’]', '', 'g'),
                                '\s+(jr|sr|ii|iii|iv|v)$', '', 'i')) AS nkey
    FROM ev_prop_picks_current WHERE is_pick
  ), pcap AS (
    SELECT o.game_pk, o.market, o.player_name, o.book, max(o.captured_at) AS cap
    FROM odds_snapshot o
    WHERE o.game_pk IN (SELECT game_pk FROM ppicks) AND o.market IN (SELECT omarket FROM ppicks)
      AND o.captured_at <= o.commence_time AND o.captured_at > now() - interval '48 hours'
      AND o.book = ANY (ARRAY['draftkings','fanduel','fanatics','hardrockbet','thescore','espnbet',
                              'williamhill_us','caesars','bet365','betmgm','ballybet'])
    GROUP BY 1, 2, 3, 4
  ), prows AS (
    SELECT o.game_pk, o.market, o.side, o.book, o.line, o.price,
           lower(regexp_replace(regexp_replace(trim(o.player_name), '[.''’]', '', 'g'),
                                '\s+(jr|sr|ii|iii|iv|v)$', '', 'i')) AS nkey
    FROM odds_snapshot o
    JOIN pcap c ON c.game_pk = o.game_pk AND c.market = o.market AND c.player_name = o.player_name
               AND c.book = o.book AND c.cap = o.captured_at
    WHERE o.price IS NOT NULL AND o.price <> 0
  ), p AS (
    SELECT 'prop'::text AS kind, 'nfl'::text AS sport, pp.game_pk, pp.market, pp.side, pp.player_id, pp.line,
           jsonb_object_agg(r.book, r.price) AS prices
    FROM ppicks pp JOIN prows r ON r.game_pk = pp.game_pk AND r.market = pp.omarket AND r.side = pp.side
                               AND r.line = pp.line AND r.nkey = pp.nkey
    GROUP BY 1, 2, 3, 4, 5, 6, 7
  )
  SELECT * FROM g UNION ALL SELECT * FROM p;

-- game_moneylines_current (migration_game_moneylines.sql) + per-book maps.
CREATE OR REPLACE VIEW game_moneylines_current AS
  WITH last AS (
    SELECT DISTINCT ON (game_pk, side, book) game_pk, side, book, price
    FROM odds_snapshot
    WHERE market = 'moneyline' AND COALESCE(player_name, '') = ''
      AND commence_time > now() AND captured_at <= commence_time
      AND captured_at > now() - interval '48 hours'
      AND price IS NOT NULL AND price <> 0
      AND book IN ('draftkings', 'fanduel', 'fanatics', 'hardrockbet', 'thescore', 'espnbet',
                   'williamhill_us', 'caesars', 'bet365', 'betmgm', 'ballybet')
    ORDER BY game_pk, side, book, captured_at DESC
  ), best AS (
    SELECT DISTINCT ON (game_pk, side) game_pk, side, book, price
    FROM last
    ORDER BY game_pk, side, american_to_decimal(price) DESC, book
  ), maps AS (
    SELECT game_pk,
           jsonb_object_agg(book, price) FILTER (WHERE side = 'home') AS home_prices,
           jsonb_object_agg(book, price) FILTER (WHERE side = 'away') AS away_prices
    FROM last GROUP BY game_pk
  )
  SELECT b.game_pk,
         max(b.price) FILTER (WHERE b.side = 'home') AS home_price,
         max(b.book)  FILTER (WHERE b.side = 'home') AS home_book,
         max(b.price) FILTER (WHERE b.side = 'away') AS away_price,
         max(b.book)  FILTER (WHERE b.side = 'away') AS away_book,
         (array_agg(m.home_prices))[1] AS home_prices,
         (array_agg(m.away_prices))[1] AS away_prices
  FROM best b JOIN maps m USING (game_pk)
  GROUP BY b.game_pk;

GRANT SELECT ON ev_pick_prices_current, game_moneylines_current TO anon, authenticated;
```

- [ ] Commit `feat(settings): per-book price views for the settings page`. (Controller verifies the SQL live, read-only, before merge.)

### Task 2: Settings store + Settings page (CappingAlpha)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`; create `/Users/ryan/Desktop/CappingAlpha/settings.html` (copy `nfl.html`, `data-page="settings"`, title `CappingAlpha | Settings`, description "CappingAlpha settings — sportsbooks, unit size, bankroll, Kelly and minimum EV.").

**Produces (Task 3/4 rely on these names):**
- `US_BOOKS` = ordered `[["draftkings","DraftKings"],["fanduel","FanDuel"],["betmgm","BetMGM"],["williamhill_us","Caesars"],["fanatics","Fanatics"],["espnbet","ESPN BET"],["hardrockbet","Hard Rock Bet"],["thescore","theScore Bet"],["bet365","Bet365"],["ballybet","Bally Bet"]]`
- `SETTINGS_DEFAULTS = {books: <all 10 keys>, unit: 10, bankroll: 1000, kelly: 0.25, minEv: 0}`
- `getSettings()` → validated settings object (per-field fallback to defaults; books filtered to known keys; empty → all)
- `saveSettings(partial)` → merges, validates, writes localStorage `ca-settings` (try/catch)
- `bookSelected(book, s)` → boolean; treats `caesars` as `williamhill_us`
- `unitScale(s)` → `s.unit / 10`
- `kellyStake(prob, american, s)` → `{dollars, units}` or `null` when f ≤ 0 / bad input; `fmtKelly(k)` → `"$X (X.Xu)"` (dollars rounded to $1 when ≥ 10, else 2 decimals; units 1 decimal)

- [ ] Nav: add `<a class="${page === 'settings' ? 'active' : ''}" href="settings.html">Settings</a>` after Track Record. Render dispatch: `else if (page === "settings") body = buildSettings();` and wire `wireSettings()` after render.
- [ ] `buildSettings()` page: heading ("SETTINGS" eyebrow, "Your settings", sub "Saved in this browser — they apply across the site."); sections:
  1. **Sportsbooks** — tile grid of US_BOOKS (logo via existing `bookLogo`-style favicon img at 28px + name; selected = accent border + check); "Select all" / "Clear" buttons; if the user clears all, keep the last one selected and show an inline note "At least one sportsbook must stay selected."
  2. **Unit size** — `$` number input + preset chips $10/$25/$50/$100; note "Every $ amount on the site (profit trackers, parlay payouts, Kelly units) uses this."
  3. **Bankroll & Kelly** — `$` bankroll input; segmented Full/Half/Quarter/Eighth; note "Kelly suggests a stake from your bankroll and each bet's edge. Quarter Kelly is the common choice when edges are model estimates."
  4. **Minimum EV** — range 0–20 step 0.5 with live "X.X%" readout; note "The +EV page hides picks below this EV at your books."
  5. **Reset to defaults** button.
  Every change calls `saveSettings` immediately and shows a brief "Saved" indicator.
- [ ] CSS for the page (reuse site tokens/look: dark cards, `#3d82d0` accent; grid auto-fill min 150px; stacks on phones).
- [ ] Verify: `node --check`; Node vm harness (scratchpad) for getSettings/saveSettings/validation/defaults, bookSelected (caesars alias), kellyStake math (e.g. prob 0.55 at −110: dec 1.9091, EV 0.05, f 0.055, quarter × $1000 → $13.75 → "$14 (1.4u)" at unit $10), fmtKelly formatting, storage throwing → defaults; browser check of the page (toggle, persist on reload, reset).

### Task 3: +EV page — re-price at my books, min EV, Kelly

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`.

- [ ] In `buildEv`, also fetch `ev_pick_prices_current?select=*` (`.catch(() => [])`) → Map keyed `game|{game_pk}|{market}|{side}` and `prop|{game_pk}|{player_id}|{market}|{side}`.
- [ ] Pure `repricePick(pick, pricesMap, s, probKey)` → `{book, price, ev} | null`: candidates = prices entries for the pick (fallback: `{[pick.best_book]: pick.best_price}` when the map has none) filtered by `bookSelected`; best = highest decimal odds; ev = prob × dec − 1; null if no candidate or ev ≤ 0 or ev < s.minEv/100.
- [ ] Game lines and props: re-price each pick (game `true_prob`, prop `model_prob`); drop nulls; render the re-priced book/price/EV (BEST BOOK column shows the chosen book); initial order EV desc; `evSortAttrs` uses re-priced values.
- [ ] New **KELLY** column (last) in both tables: `fmtKelly(kellyStake(prob, price, s))` or "—".
- [ ] Parlays: show a ticket only if `bookSelected(p.book, s)` and `+p.ev >= s.minEv/100`; add a Kelly line on the card (`Kelly $X (X.Xu)` using `true_prob` and `parlay_price`); the "$10 to win $Y" becomes "$U to win $Y×U/10" with U = unit (format `$25 to win $123.45`).
- [ ] Header stat cards and TOP EV use the filtered/re-priced lists; add a small line under the page heading when settings differ from defaults: "Filtered by your settings: N books · min EV X% · <a href="settings.html">Change</a>".
- [ ] Verify with the Node harness (repricePick cases: best among selected, fallback, min EV cut, caesars alias, no selected book → null) and a browser check (turn off a book → its picks re-price or drop; min EV hides picks; Kelly shown).

### Task 4: Tiles from my books + unit scaling everywhere

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`; all `*.html` → `v=20260923m`.

- [ ] Tiles (`gameLinkRow`): when the moneylines row has `home_prices/away_prices`, use the best price among selected books for each side (book logo = that book); if none of the user's books priced a side, fall back to the model fair line (dimmed "model"); if the maps are absent (old view), keep today's `home_price/home_book` behavior.
- [ ] Unit scaling (`unitScale(getSettings())`): multiply every $ amount — `pnlBody` (cards, subtexts, chart series, total wagered), `propGamesSection` + `propGameDetail`, graded parlays list, parlay "to win". Labels saying "$10 per bet"/"$10/bet" become "$U per bet" with the user's unit. Percentages (ROI) are unchanged.
- [ ] Verify: harness/browser — unit $25 doubles-and-a-half every $ value and label; defaults show exactly today's numbers.
