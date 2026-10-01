# CappingAlpha site redesign (Dashboard, game page, +EV, Track Record) — design spec

Date: 2026-10-01 · Status: decisions made in chat; awaiting user review of this file

## Goal

Rebuild the CappingAlpha site (`/Users/ryan/Desktop/CappingAlpha`, static, Cloudflare
Pages, user redeploys) to match the four mockups the user supplied — Dashboard, game
page ("Matchup Story"), +EV, Track Record — as closely as possible, showing **real
data only** (the mockups' numbers are placeholders; nothing is invented), and add the
data the mockups need that we do not collect yet.

## User decisions (2026-10-01)

1. **Alpha Score** = edge × track record (formula §3); tiers HIGH ≥ 85, STRONG 75–84,
   MEDIUM 65–74; below 65 is not listed as an opportunity.
2. **NBA / MLB** stay in the nav exactly as in the mockups, with honest empty states
   (NBA: no model yet; MLB: model paused since 2026-08-31). Counts show 0.
3. **Missing data:** build what is free; hide (never fake) any panel whose data is
   still unavailable.
4. **Accounts:** none. Watchlist is saved per browser; the avatar opens a menu
   (Settings, Power Rankings). Portfolio = the model's own units, not the user's bets.

Defaults taken (override on review): Power Rankings becomes a tab on the NFL / CFB
pages and a link in the avatar menu; Settings moves into the avatar menu.

## 1. Design system (from the mockups)

- **Theme:** light. Page background warm off-white (#F7F5F0-ish), white cards with a
  1px hairline border and 10–12px radius, navy header bar (#0F2340-ish) with white
  nav text and an underline on the active item.
- **Type:** serif display face for page titles and section heads ("Today at a
  Glance", "Best Opportunities Right Now") — Google Fonts *Libre Caslon Text* (closest
  free match; swap if the user names the exact face); Inter for UI and numbers
  (tabular figures).
- **Color semantics:** positive green (#1F8A4C), negative red (#C8372D), confidence
  pills — HIGH green-tint, STRONG blue-tint, MEDIUM amber-tint; Alpha Score cell =
  amber-tint box, green-tint when ≥ 85.
- **Components:** stat card (label, big number, sub-line, optional sparkline / donut
  / mini bar chart), pill tab group, data table (logo + text cells, right-aligned
  numbers, row arrow button), filter bar (dropdowns + search), list panels.
- **Charts:** inline SVG (no chart library): sparkline, area line (cumulative units /
  net wins), donut (hit rate, exposure), grouped bars (by market, calibration),
  histogram (EV distribution), quarter line chart (game flow).
- Dark theme is retired (the mockups are light). Mobile: cards stack, tables scroll
  horizontally inside their card; 16px gutters.

## 2. Pages and data mapping

Legend: **have** = existing table/view; **calc** = new computation from existing
data; **new data** = §4 pipeline; **hidden** = not shown until data exists.

### Header (all pages)
Wordmark "Cappingαlpha" (serif), nav Dashboard · MLB · NFL · NBA · CFB · +EV · Track
Record, search (client-side over today's games, teams, players with props), avatar
menu (Settings, Power Rankings, Watchlist).

### Dashboard (`index.html`)
- Date control (prev / date / next) — every panel reads the selected date.
- Cards: Games Tracked by sport (have: `predictions_current`); Live +EV
  Opportunities + "vs yesterday" + 7-day mini bars (calc: daily counts from
  `ev_picks` / `ev_prop_picks` history); Best Current Edge (have); Model Hit Rate
  (30D) + "vs market" + donut (calc: hit rate − mean implied probability of the same
  graded picks); Units (30D) + ROI + sparkline (have: `prediction_pnl_daily`,
  `ev_pnl_daily`); Active Signals = counts by tier (calc, §3).
- Best Opportunities Right Now: sport / Game Lines / Player Props tabs; columns as
  mockup incl. best-odds book logo (new: logo assets), Alpha Score, Confidence.
- Today's Slate: per game its single highest-Alpha market (calc).
- Performance Snapshot 7D / 30D / Season: ROI, Units, Hit Rate, Avg Edge, cumulative
  units area chart (have + calc).
- Market Movers tabs: Line Moves (calc from `odds_snapshot` open→current and
  `open_pinnacle_price`), Model vs Market (have), New Signals (calc: picks first seen
  in the last 24h), Stale Lines (calc: book price lagging the consensus). Ticket /
  money % sub-lines from splits (new data, §4.4) — hidden per game when absent.
- Portfolio & Exposure: model units by sport / by market type, 30D (calc).
- Watchlist: games / teams / players starred by the user (localStorage).
- Recent Model Updates feed (new data, §4.1).

### Game page (`game.html`, NFL / CFB / MLB / NBA)
- Hero: logos, records with conference/division record (calc: CFB
  `conference_game`; NFL division map), kickoff, network, venue + city, weather
  (new data, §4.2; indoor → "Indoors").
- Odds boxes ML / spread / total (have: consensus / best).
- CappingAlpha Read: the market with the largest positive edge, market-implied vs
  model probability, edge box (calc).
- Tabs: Overview (Alpha Score, win prob, projected score / spread / total vs market,
  Model Projection table) · Matchup (existing matchup grades + power rank) · Market
  (line history chart from `odds_snapshot`, splits when present, best book per
  side) · Trends (existing history + trends) · Players (existing NFL props; empty
  state elsewhere).
- Key Insights: generated sentences from available stats — explosive-play rank,
  turnover rate, CFB havoc rate (new data §4.3), unit mismatches from matchup grades,
  weather; NFL pressure rate only if the source check in §4.3 passes, else omitted.
- Projected Game Flow by quarter (new data, §4.5).
- Cover Probability bar (model). The mockup's "% of Money" caption is shown only
  with splits data.

### +EV (`ev.html`)
- Cards: Total +EV Opportunities + vs yesterday, Average Edge, Highest Edge, Model Hit
  Rate (30D), ROI (30D).
- Filters: Sport, League, Market Type, Sportsbook, Minimum Edge, Confidence, Date,
  search; tabs All / Game Lines / Player Props; Sort By.
- Table: as mockup (Model Prob, Impl. Prob, Edge, EV, Alpha Score, Confidence, Game
  Time).
- Right rail: Top Alpha Opportunities (top 5), Market Pulse (biggest line move,
  highest-confidence edge, most mispriced total, average divergence vs closing lines
  = mean CLV of graded picks), EV Distribution histogram, Performance by Edge Bucket
  (hits, ROI, units per bucket from graded `ev_results` / prop results).
- Existing parlay and settings-driven stake features stay, restyled.

### Track Record (`track-record.html`)
- Range 7D / 30D / Season / All Time + date range.
- Cards: Overall record W-L-P + win %, Accuracy vs Closing Line (= share of picks
  that beat the close, plus mean CLV), Total Predictions, Current Streak (calc),
  Average Confidence (= mean model probability).
- Cumulative net wins by league; Performance by Market (ML / spread / total
  W-L-P bars); Performance by League; Confidence Calibration (predicted vs actual by
  band); Recent Form L10 / L25 / L50 / L100; filterable results table (league,
  market, confidence, result, sort, search); Best Performing Segments; Model
  Calibration (Brier score, predicted vs actual).
- Scope = the existing published record (`in_track_record`: NFL since the 2026-09-28
  restart, CFB / MLB per their starts). The archived pre-restart NFL record stays
  reachable through an "Archive" toggle. History will look short; that is the truth.

### NFL / CFB / MLB / NBA board pages
Restyled to the same system (not in the mockups): game cards → slate table like
Today's Slate with all markets; Power Rankings tab (NFL / CFB).

## 3. Alpha Score

Per pick (game line or prop) at its best available price:

    s_ev    = clip(EV% / 8, 0, 1)                 # EV at best price, % of stake
    s_edge  = clip(edge_pp / 8, 0, 1)             # model prob − implied prob, points
    roi_s   = roi_segment * n / (n + 100)         # graded ROI of sport×market, shrunk
    s_track = 0.5 + clip(roi_s / 20, -0.5, 0.5)
    alpha   = round(40 + 30 * s_ev + 20 * s_edge + 10 * s_track)   # 40..100

Tiers: HIGH ≥ 85, STRONG 75–84, MEDIUM 65–74, < 65 not an opportunity. Computed
in one SQL view (`alpha_scores_current`) so the dashboard, +EV and game pages agree.
Calibration is checked once 200+ graded picks exist (do higher scores win more?);
the weights are a starting point, not fitted.

## 4. New data (pipelines; user runs migrations)

1. **Projection history** — table `projection_history` (sport, game_pk, generated_at,
   model_version, home/away score, win prob, spread, total, best market + Alpha) —
   appended by every generate run; the Recent Model Updates feed lists material
   changes (|Δ win prob| ≥ 2 pts, |Δ total| ≥ 1, |Δ spread| ≥ 1, tier change, new
   week released).
2. **Venue + weather** — from ESPN scoreboard / summary (venue name, city, indoor
   flag, forecast temperature / conditions / wind when ESPN provides it); stored on a
   `game_info` table refreshed with the daily jobs. Missing → hidden.
3. **Insight stats** — CFB havoc rate and turnover rates from CFBD (season advanced
   stats); NFL turnover rate from nflverse pbp; NFL pressure rate only if a free
   nflverse source covers the current season (checked first; otherwise omitted).
4. **Betting splits** — fix CFB capture (0 rows ever); add an earlier daily capture
   (e.g. morning of game day) so dashboards have splits before the 3-hour closing
   window. Uses the existing provider; any added cost is flagged before enabling.
5. **Quarter game flow** — projected points by quarter = projected team score ×
   that team's scoring share by quarter (last 2 seasons, shrunk to the league
   average), labelled "projected pace". No sim change.
6. **Sportsbook logos** — small logo images for the books we price, in the site's
   `public/` folder.

## 5. Phasing

- **Phase A — site:** design system, the four mockup pages + restyled board pages,
  Alpha Score view, client-side search, watchlist, empty states. Uses existing data;
  panels needing §4 data stay hidden.
- **Phase B — data:** §4.1–4.6, each lighting up its hidden panel.
Each phase ships as its own PR(s) + site deploy.

## Out of scope

NBA model, MLB restart, user accounts / personal bet tracking, paid data feeds.

## Risks

- Short track record makes 30D / Season numbers noisy; labels show sample sizes.
- Alpha Score weights are judgment until enough picks are graded.
- ESPN weather availability varies (often only near kickoff); panels hide cleanly.
- The splits provider may not support CFB; then CFB splits stay hidden.
