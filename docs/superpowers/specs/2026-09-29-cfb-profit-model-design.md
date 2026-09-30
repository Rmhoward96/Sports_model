# CFB profit model: bets tuned for Kelly bankroll growth (design spec)

Date: 2026-09-29 · Status: approved in chat, pending user review of this file

## Goal

Replace "the CFB ratings model's side of the line" as the source of CFB bets
with a model trained to **beat the price being bet**, sized by fractional
Kelly, and judged on held-out **bankroll growth** — pure profitability, not
point-estimate accuracy. CFB game lines only (moneyline, spread, total); NFL
gets the same treatment in a later round.

## Why (current state)

- `cfb-ratings-v1` (Elo + SRS + opponent-adjusted points, shrunk toward the
  close) was tuned to minimize MAE, which pushed it to the maximal market
  weight. On 2023–24 held-out games its own side of the closing line hit
  **49.7 % ATS** (n=1410) and **51.6 % O/U** (n=1416) — below the 52.4 %
  breakeven (`docs/superpowers/reports/2026-09-01-cfb-gameline-backtest.md`).
- 2026 record: model picks ML +$110 (144), spread −$27.90 (209), total
  +$27.60 (209); the Pinnacle line-shopping board's CFB ML +$92.40 (63).
- Historical prices on file: only the median **closing** spread/total
  2015–2024 (`assets/cfb/lines.parquet`) — no openers, no moneylines.

## Decisions (user, 2026-09-29)

1. Scope: CFB game lines first (NFL later).
2. Bet timing: "whenever the board says" — re-price live against current
   prices; test historically at the opener and at the close.
3. Objective: risk-adjusted growth — fractional Kelly.
4. Approach A: a profit-trained model; the current ratings model's picks are
   the baseline in the gate; the Pinnacle line-shopping board stays as is.

## Design

### 1. Price history (data)

- Re-pull CFBD `/lines` for 2015–2025 into `assets/cfb/lines.parquet` (new
  columns, existing ones unchanged): per game the median across providers of
  `spread_open`, `spread_close` (= today's `market_spread`), `total_open`,
  `total_close` (= `market_total`), `ml_home`, `ml_away` (American; closing
  moneylines as CFBD reports them), plus `n_providers`. Home-margin
  convention as today.
- Coverage report per season and field. The **moneyline market is tested only
  on seasons where ≥ 70 % of priced games have both moneylines**; a spread
  or total season needs ≥ 70 % opener coverage to count at the opener.
- 2026 live prices come from `odds_snapshot` (open, current, best available
  across tracked books).

### 2. Model

Three probability models, one per market, each predicting the chance a bet at
a given line wins; the line is an input so one model prices the opener, the
close, or the live board:
- spread: P(home margin > −L | no push) for home line L;
- total: P(total > T | no push);
- moneyline: P(home wins).

Features (all pre-game; ratings walk-forward as in `backtest_cfb_gameline.py`):
- ratings: pre-game Elo diff, SRS diff, opponent-adjusted points (off/def),
  the ratings model's own margin and total;
- preseason priors (`assets/cfb/priors.parquet`, `build_cfb_priors.py`): SP+,
  returning production, recruiting, transfer portal, coaching change — each
  side and the difference;
- market: the line being priced, (model number − line), and opener → priced
  line movement (0 at the opener);
- context: neutral site, week, conference game, P4 / G5 / FCS matchup class,
  rest days;
- weather at kickoff where the venue is known (optional; NaN otherwise).

Learner: histogram gradient boosting classifiers (scikit-learn, as NFL v2).
Training rows: each historical game once at its opener and once at its close
(same outcome; pushes dropped), so the model learns price dependence.
Walk-forward by season: season S is predicted by models trained on seasons
< S only. Calibration: isotonic regression fitted on out-of-fold predictions
of seasons < S (per market), applied to season S.

### 3. Betting policy

Per game and market: price both sides at the bet price (tests: the historical
opener or close at −110 for spread/total, real moneylines; live: the best
available price and line across tracked books).
- edge = p × decimal_odds − 1; a side is eligible if edge ≥ `min_edge`;
  at most one side per market per game (the larger edge).
- stake fraction = `kelly_frac` × edge / (decimal_odds − 1), capped at
  **3 %** of bankroll per bet; a Saturday (slate-day) cap of **15 %** of
  bankroll scales that day's stakes down proportionally if exceeded.
- push → stake returned.
- `kelly_frac` ∈ {1/4, 1/3, 1/2} and `min_edge` ∈ {0, 1, 2, 3, 4, 5, 6} %
  are chosen per market on the tuning seasons by mean log bankroll growth,
  then frozen.

### 4. Gate (walk-forward)

- Tuning seasons: 2019–2022 (models trained on earlier seasons; `kelly_frac`
  and `min_edge` chosen here). Verdict seasons: **2023, 2024, 2025** — never
  used for any choice; simulated at the opener and at the close.
- Baselines: (1) the current ratings model's own cover/over/win
  probabilities (its margin/total distributions vs the line) through the same
  policy, tuned the same way; (2) flat $10 on every A bet above `min_edge`.
- Metrics per market and pooled: mean log growth per bet, ending bankroll
  from 100 units, ROI on money staked with a season-week cluster bootstrap
  95 % CI, max drawdown, calibration (reliability deciles, ECE), bets placed.
- **Ship rule, per market** (each market passes or fails on its own):
  1. Kelly log growth > 0 over 2023–25 combined AND in ≥ 2 of the 3 seasons,
     at the opener;
  2. ROI CI lower bound > 0 on the combined verdict seasons (at the opener);
  3. calibration ECE < 3 points;
  4. beats baseline (1) on combined log growth.
  At-the-close results are reported, not required. A failing market is not
  served (the line-shopping board still covers it). The user sees all numbers
  either way.
- Report `docs/superpowers/reports/2026-09-xx-cfb-profit-gate.md` + json.

### 5. Serving

- New step after each odds capture (same schedule as the +EV board): for
  every upcoming CFB game and each passing market, price both sides at the
  best available line/price, apply the policy, and write
  `cfb_model_bets` (sport, game_pk, market, side, line, price, book,
  model_prob, edge, stake_pct, kelly_frac, model_version, created_at).
  Current view `cfb_model_bets_current` = the latest flag per
  (game, market) before kickoff.
- Weekly retrain on all completed games + a quick holdout check on the most
  recent completed weeks before publishing artifacts (props-ML pattern:
  Release, manifest-verified download).
- Repo variable `CFB_PROFIT_MODEL` (default `off` → today's behavior).
- Unchanged: the ratings model still produces CFB projected scores / win
  probability for game pages (`cfb-ratings-v1`); the Pinnacle line-shopping
  +EV board keeps running for all markets.

### 6. Site (CappingAlpha, user redeploys)

- +EV page: CFB model bets tagged **Model**, with **stake: X % of bankroll
  (≈ $Y)**; line-shop picks show no stake.
- Settings: a **bankroll** field (default $1,000), stakes shown in dollars.
- CFB game page: the model's bet per flagged market with its stake.
- Track record: a **CFB model bets** section — each bet graded at the last
  price flagged before kickoff (the +EV record's rule): Kelly bankroll from
  100 units (chart + current), flat $10 P&L, ROI, win rate, CLV vs the
  Pinnacle close, by market. Respects `track_record_start` like the rest.
- Grading: `cfb_model_bet_results` filled by a new step in the existing
  grading run.

### 7. Rollback

`CFB_PROFIT_MODEL=off` stops new flags; the site hides CFB model bets; the
record stays. Nothing else changes.

## Out of scope

- NFL game lines and NFL props (a later round, same framework).
- Live in-game betting; player props for CFB.
- Changing the ratings model that drives CFB projections.

## Risks

- CFBD moneyline/opener coverage may be thin before ~2020 → the moneyline
  market may have too few verdict seasons; reported, and the ship rule then
  can't pass for it.
- Historical openers are a CFBD median, not a single bettable number at a
  single time; live prices are best-of-books. The test is therefore an
  approximation of live execution (optimistic vs a single book's opener,
  pessimistic vs best-of-books).
- Kelly is sensitive to miscalibration; the ECE gate and fractional Kelly +
  caps limit the damage.
- Three held-out seasons ≈ 2,100 priced games per market — enough for log
  growth, still noisy per season.
