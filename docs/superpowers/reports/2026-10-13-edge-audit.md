# Edge audit of all graded picks, 2026-10-13

Read-only spike. Question: where, if anywhere, does the data gathered so far show a statistically credible betting edge, and where does it not? Nothing in the product, workflows, site or database was changed. Every database connection was read-only (SELECT only). Analysis scripts are scratch files and are not committed.

Data window: odds captures start 2026-08-20. Football (NFL, CFB) has about 5 to 6 weeks of graded games (NFL from 2026-09-09, CFB from 2026-08-29, CFB weeks 2 to 7 carry nearly all of it). MLB has one week of pre-pause picks (2026-08-23 to 08-29, 1,972 graded) and about one week of postseason (10 graded games, 90 graded props). Every sample here is small by betting standards. Where a result is noise, it is labelled noise.

## 1. Executive summary

**Bottom line: no statistically credible, profitable edge exists in any product, sport or market on the data gathered so far.** Several diagnostics are credibly negative, and a few structural effects are real but too small to survive the vig.

| Rank | Finding | Evidence |
|---|---|---|
| Credible edge | None | -- |
| Credible structural effect, too small to profit | CFB totals drift down from opener to close, about -0.4 pt. In the 2021-25 history (3,701 games) the drift is -0.42 pt, SE 0.035, and negative in 4 of 5 seasons. Betting every CFB under at the opener hit 51.9% (ROI -1.0%, 95% CI [-4.1%, +2.0%] at -110). | History: `assets/cfb/lines.parquet`. Live 2026: -0.37 pt over 211 games (t = -3.8), but unders at the open hit 48.8% (n=207). |
| Promising but unproven | Early "line-adjusted" shopping on CFB totals: a US-book number or price better than Pinnacle's no-vig fair value at the first captured pull. EV measured against Pinnacle's valid close is +2.9% (95% CI [+1.0%, +4.9%]) over 57 games flagged at 1% or more. CFB spreads +1.3% [-1.1%, +3.9%]; NFL spreads about 0. | Rests on a normal-approximation line-to-probability conversion (key numbers undervalued). 62% of opportunities last one pull (about 2 h). Realized ROI of the same bets is -13.5% [-37%, +10%] (noise at n=57). Needs a prospective test (section 6). |
| No evidence | Model ATS and O/U picks (CFB 160-155-5 and 164-156; NFL 37-38 and 33-46). Opener-to-close line movement toward the model (slopes 0.02 to 0.05, all |t| < 2 except one multiple-comparison hit). NFL and MLB prop picks (452 and 90 bets; ROI -5.0% [-14.1%, +3.7%] and +1.5% [-16.6%, +20.1%]). Pre-pause MLB (1,972 picks, ROI -4.0% [-9.5%, +1.4%], CLV +0.06 pp). Best parlays (2-12). Same-line shopping on spreads and totals (+EV in under 1% of pulls). Prop shopping persistence (EV vanishes by the close). | Sections 2, 4, 5. |
| Credibly negative | (a) Model probabilities for cover, over and props are overconfident. NFL prop leans claim 64.8% and hit 51.8%; the recalibration slope is 0.15 (SE 0.07, n=1,587). CFB cover probabilities claim 61.2% against 50.8% actual. (b) CFB moneyline "value side" bets (model edge vs Pinnacle no-vig close) lose: -18.9% at any edge, -41.9% [-62.3%, -20.0%] at 8+ pp, because the model's win probabilities are compressed (slope 0.75 vs market) and the bets are mostly underdogs. (c) NFL total picks 33-46, ROI -20.3% [-42.0%, -0.9%]. (d) The +EV board's 165 graded game-line picks (all Decision Desk overlay; see below) have negative CLV: -0.53 pp [-0.85, -0.22]. | Sections 2 and 3. |

Three structural findings shape everything else.

1. **The +EV game-line board never produced a line-shopping pick that was graded.** All 165 graded `ev_picks` rows carry a non-zero `desk_delta`: they are Decision Desk overlay picks. The Desk was retired 2026-09-28. Since then the pure line-shopping board has produced 0 picks, because 15 of 1,286 desk-free rows were +EV at all and 2 above 1%. The board, its CLV and the site's `ev_pnl` are measuring the retired Desk.
2. **The "close" used for CLV is often stale.** The last Pinnacle capture before kickoff is more than 12 h early for 53% of NFL games and 24% of CFB games. Only 44% of NFL and 71% of CFB games have a Pinnacle capture within 3 h of kickoff. Props have no Pinnacle prices at all, so `ev_prop_results.clv` is NULL for all 542 graded props.
3. **Moneyline "line-shopping edges" are mostly an artifact.** 937 of 952 flagged CFB ML opportunities (+1% EV or more) are underdogs, with Pinnacle fair about 15%. The multiplicative no-vig method overstates longshot probabilities: CFB dogs won 22.6% against 27.8% implied (z = -1.7, n=195). Those "edges" persist to the close because they are structural, and they do not realize (CFB flagged-ML ROI -18.2% [-58%, +27%], n=60).

## 2. Per-product results

Conventions (details in the method appendix). ROI is per 1 unit staked. 95% CIs are bootstrap (4,000 resamples; clustered by game where picks share a game). "Valid close" means the last capture is 3 h or less before kickoff. CLV is measured against the no-vig closing price. "Fair-move CLV" is the change in Pinnacle (or US-consensus) no-vig probability from pick time to close, which removes the vig and the shopping effect. "CLV at bet price" is the no-vig close probability minus 1/decimal(bet price), so it includes the vig the bettor paid; a zero-skill bet scores about -2 to -4 pp on this measure, not 0.

### 2.1 Model game picks (`prediction_accuracy`, closing line = ESPN pickcenter)

Each game's model lean, bet at the closing line. Spread and total at -110 flat (breakeven 52.4%); the site's `prediction_pnl` uses the median closing price and differs by a few points of ROI. Moneyline = predicted winner at the best US closing price.

| Sport / market | N | W-L-P | Hit rate (95% CI) | ROI (95% CI) |
|---|---|---|---|---|
| CFB spread | 315 | 160-155-5 | 50.8% (45.3, 56.3) | -3.0% (-13.3, +7.9) |
| CFB total | 320 | 164-156 | 51.2% (45.8, 56.7) | -2.2% (-12.3, +8.0) |
| CFB moneyline (best US close) | 206 | 153-53 | 74.4% | +3.5% (-6.7, +13.4) |
| NFL spread | 75 | 37-38 | 49.3% (38.3, 60.4) | -5.8% (-28.7, +17.1) |
| NFL total | 79 | 33-46 | 41.8% (31.5, 52.8) | **-20.3% (-42.0, -0.9)** |
| NFL moneyline (best US close) | 76 | 48-28 | 63.2% | -0.8% (-19.0, +16.6) |
| MLB (all markets) | 10 games | ML 5-5, spread 6-4, total 3-6-1 | -- | far too small to read |

Notes.
- NFL totals are the only credibly negative row (upper CI bound below 0, one of 15 comparisons, so treat as suggestive). NFL Elo (47 games) and ML sim (32 games) are both below 50% on totals (42.6% and 40.6%).
- The favorite-vs-underdog split is noise: CFB underdog picks 105-101, favorite picks 55-54; NFL underdog picks 27-23 (ROI +3.1%, CI -23.6% to +29.8%), favorite picks 10-15.
- By model version: CFB v1 spread 106-99, v2 54-56; totals v1 111-98, v2 53-58. `prediction_accuracy` has no `model_version` column; versions were joined from `game_predictions`.
- Edge-size buckets (|model line - closing line|) show no monotonic relationship. CFB spread hit rate by bucket: 0-1 pt 68.6% (n=35), 1-2 pt 43.9%, 2-3 pt 47.4%, 3-5 pt 43.0%, 5-7 pt 47.9%, 7+ pt 58.1% (n=74, ROI +10.9%, an isolated point). At 5+ points: n=122, hit 54.1%, ROI +3.3% [-13.9%, +20.5%].
- CFB conference tier (Power 5 = ACC, Big 12, Big Ten, Pac-12, SEC conference ids; FBS vs FBS only, so no FCS games): P5 v P5 ATS 75-62 (ROI +4.5%), P5 v G5 41-40, G5 v G5 44-53 (ROI -13.4%). No tier differs from 50% beyond noise, and the model's information over the closing line is zero in every tier (coefficients -0.16, +0.32, -0.07, SE about 0.28).
- Does the model know anything the closing line does not? Regressing the actual margin on the closing margin plus (model margin minus closing margin): coefficient on the model-minus-market term is **+0.04 (SE 0.16) for CFB** and +0.00 (SE 0.42) for NFL; totals -0.02 (SE 0.16) and -0.12 (SE 0.56). Model MAE is worse than the closing line's (CFB margin 12.61 vs 11.42; NFL 9.88 vs 9.70; totals CFB 12.79 vs 12.06, NFL 11.59 vs 10.49).

**Blind betting of the model at the opener** (Pinnacle first capture, 24+ h before kickoff, at Pinnacle's opener price; edge = |model number - opener|):

| Sport / market | Edge cut | N | Hit | ROI (95% CI) | Fair-move CLV (pp) |
|---|---|---|---|---|---|
| CFB spread | all | 211 | 48.3% | -6.8% (-19.8, +6.2) | +0.47 (-0.14, +1.10) |
| CFB spread | 3+ pt | 136 | 50.4% | -3.1% (-18.9, +13.1) | +0.37 (-0.40, +1.14) |
| CFB spread | 5+ pt | 85 | 50.0% | -4.0% (-24.5, +16.7) | +0.58 (-0.35, +1.58) |
| CFB total | all | 211 | 52.7% | +0.7% (-12.2, +13.7) | +0.01 (-0.60, +0.62) |
| CFB total | 5+ pt | 74 | 54.8% | +5.0% (-16.1, +26.2) | -0.30 (-1.36, +0.72) |
| NFL spread | all | 76 | 53.4% | +2.2% (-20.8, +23.1) | +0.35 (-0.37, +1.09) |
| NFL spread | 3+ pt | 30 | 56.7% | +7.4% (-25.1, +39.6) | +0.88 (-0.29, +2.07) |
| NFL total | all | 76 | 44.7% | -14.3% (-34.6, +8.0) | **-0.77 (-1.38, -0.15)** |

No cut is distinguishable from zero except NFL totals, where the market moves against the model's number. Betting at the opener versus the close makes no reliable difference (CFB spread at the close: 111-98, ROI +1.4%).

**Moneyline, model win probability vs Pinnacle no-vig** (n=195 CFB games with Pinnacle ML, n=77 NFL): see section 3 (compression) for why this is where the CFB model loses money.

### 2.2 +EV board, game lines (`ev_picks`, `ev_results`): 165 graded picks, all Decision Desk overlay

| Sport / market | N | W-L-P | Hit | ROI at recorded best price (95% CI) | Grader CLV, Pinnacle price move (pp, 95% CI) |
|---|---|---|---|---|---|
| CFB moneyline | 63 | 40-23-0 | 63.5% | +14.7% (-9.9, +37.7) | -0.62 (-1.12, -0.12) |
| CFB spread | 44 | 23-21-0 | 52.3% | +0.7% (-26.2, +27.5) | -0.61 (-1.03, -0.21) |
| NFL moneyline | 32 | 17-15-0 | 53.1% | -5.5% (-37.3, +26.2) | -0.04 (-0.96, +0.91) |
| NFL spread | 26 | 10-14-2 | 41.7% | -18.5% (-52.1, +15.5) | -0.79 (-1.65, +0.04) |
| All | 165 | 90-73-2 | 55.2% | +1.8% (-12.9, +15.9) | **-0.53 (-0.85, -0.22)** |

The ROI uses the recorded `best_price`, which is the price at the last board rebuild before kickoff (rows are overwritten on rebuild; only `created_at` and `open_pinnacle_price` are frozen). I rebuilt the price at first posting (`created_at`) from `odds_snapshot` for the 149 picks that have a Pinnacle capture at that time:

- ROI at the first-posting best US price: -5.3% [-20.6%, +10.5%].
- CLV at bet price vs no-vig Pinnacle valid close: -1.85 pp [-2.37, -1.34] (n=89 with a valid close; 22% beat the close).
- Fair-move CLV: -0.44 pp [-0.89, -0.01] (n=98).

The zero-skill baseline on this scale is about -2.3% for moneylines and -3.9% for spreads (section 5), so the desk picks sit slightly above random bets, but still below zero, and the Pinnacle fair price on average moved slightly against the desk side.

Splits (n small throughout; these use the 149-pick reconstructed subset): favorites 58-33-1 (ROI +6.8%, CI -9.9% to +23.3%), underdogs 22-34-1 (-11.8%). By desk edge, 0-3 pp: ROI -8.9%; 3-5 pp: +1.2%; 5-8 pp: -5.6%; the 5 picks above 8 pp all won, but 6 "high conviction" picks is not a sample. By hours before kickoff at first posting: 96+ h (n=42) hit 68.3% and +17.7% ROI (CI -6.9% to +40.9%); 48-96 h (n=84) hit 48%, ROI about -15%. The grader's `clv` for spread and total is price-based only: it ignores any line movement, so it can misstate CLV when the number moved.

### 2.3 +EV board, game-line parlays

`ev_parlays` is empty. `ev_best_parlays` has 14 NFL three-leg prop tickets (2026-09-25 to 10-12): 2 wins, 12 losses, -$23.1 on $140 (-16.5%). Claimed true probability per ticket averaged 26.9% (claimed EV +81%) against 14.9% implied. Of 35 graded legs, 48.6% hit against a claimed 64.4%. Too small to estimate a ROI; fully consistent with the overconfidence in section 3.

### 2.4 Player props (`ev_prop_picks`, `ev_prop_results`)

These are model-driven picks: the simulation's P(over) at the book's main line against the best non-Pinnacle US price. **There are no Pinnacle prop prices in `odds_snapshot` (0 rows), so `pinnacle_price`, `open_pinnacle_price` and `ev_prop_results.clv` are NULL for every prop.** I built a consensus close instead: the median no-vig price across US books that still quote the pick's exact line within 3 h of kickoff (at least 2 books). Only 39% of picks have one (214 of 542). Profit is already in units.

| Sport / market | N | W-L | Hit | ROI (95% CI) | Fair-move CLV vs US consensus (pp) |
|---|---|---|---|---|---|
| NFL, all | 452 | 225-227 | 49.8% | -5.0% (-14.1, +3.7) | -0.25 (-0.56, +0.06), n=128 |
| NFL rec_yds | 167 | 85-82 | 50.9% | -3.8% (-18.4, +11.0) | -0.01 (-0.10, +0.07), n=39 |
| NFL receptions | 150 | 70-80 | 46.7% | -10.1% (-25.3, +6.0) | -0.50 (-1.18, +0.17) |
| NFL rush_yds | 84 | 40-44 | 47.6% | -9.8% (-30.1, +10.5) | +0.01 (-0.11, +0.17) |
| NFL pass_tds | 51 | 30-21 | 58.8% | +13.8% (-13.4, +41.4) | -0.28 (-1.16, +0.63) |
| MLB, all | 90 | 51-39 | 56.7% | +1.5% (-16.6, +20.1) | -0.08 (-0.19, 0.00), n=78 |
| MLB total_bases | 64 | 36-28 | 56.2% | -2.2% (-23.5, +19.0) | -0.02 (-0.08, +0.02) |
| MLB hits_allowed / pitcher_ks / outs | 8 / 9 / 9 | 6-2 / 6-3 / 3-6 | -- | not interpretable | -- |
| Both | 542 | 276-266 | 50.9% | -3.9% (-11.6, +4.0) | -0.18 (-0.38, +0.01), n=206 |

- Claimed model probability on the picked side is 58.0% against a no-vig market probability of 50.2% and an actual hit rate of 50.9%. Claimed EV 3% to 25%; realized ROI by claimed-EV bucket: 0-3%: -1.6%; 3-6%: +1.2%; 6-10%: -9.8%; 10-15%: -3.1%; 15-25%: -5.6%. Bigger claimed edges did not pay more.
- Timing: picks first posted 48-72 h before kickoff (n=226) show a fair-move CLV of -0.64 pp [-1.23, -0.07] (market moved against them); within 6 h before kickoff (n=122), -0.06 [-0.14, 0.00]. No early-betting advantage is visible for model props.
- Line movement at the close, for picks with a visible close: the line moved in our favor on 11% of picks and against on 6%.
- Pass-only comparison: `nfl_prop_grades` has every sim lean with a book line, not only board picks (weeks 3 to 5, n=1,587 after excluding pushes): hit 51.8% (95% CI [49.3%, 54.2%]), ROI at the leaned side's best price -2.5% [-7.3%, +2.2%], sim MAE 15.14 vs line MAE 14.41. Weekly drift: week 3 hit 48.4%, week 4 52.2%, week 5 55.0% (n=489), and the week-5 recalibration slope rises to 0.47; with v2 live from 09-29 it is too early to call an improvement.

### 2.5 Pre-pause MLB (`picks`, legacy)

1,972 graded picks over about 7 days (2026-08-23 to 08-29; bets mostly 0.7 to 2.3 h before first pitch; 32 NFL rows were never graded). Cluster-by-game bootstrap.

| Market | N | W-L-P | Hit | ROI (95% CI) | CLV, no-vig move (pp, 95% CI) |
|---|---|---|---|---|---|
| All | 1,972 | 1,002-966-4 | 50.9% | -4.0% (-9.5, +1.4) | +0.06 (+0.01, +0.11) |
| total_bases | 433 | 235-198 | 54.3% | -1.3% (-10.1, +6.9) | -0.08 (-0.18, +0.02) |
| hits | 422 | 191-231 | 45.3% | -7.6% (-17.0, +1.7) | +0.30 (+0.19, +0.42) |
| hrr | 751 | 393-358 | 52.3% | -4.7% (-12.6, +3.2) | +0.02 (-0.05, +0.09) |
| pitcher_ks / hits_allowed / outs | 68 / 65 / 61 | -- | -- | +/-1% to +4%, CIs of +/-25% | about 0 |
| moneyline / spread / total | 43 / 62 / 67 | -- | -- | +0.5% / -2.9% / -10.2%, CIs of +/-25% | +0.30 / +0.06 / +0.10 |

The record is -79 units. CLV is essentially zero overall. The one nominally positive CLV, `hits` overs (+0.30 pp, 62% beat the close), came with a -7.6% ROI; 0.3 pp is under 1% EV and was dropped from the product anyway. Claimed model probability 57.2% vs actual 50.9% (recalibration slope 0.82, SE 0.13); the model does not beat the market probability on Brier score (0.2489 vs 0.2440). MLB postseason (10 game picks, 90 props) is in the tables above and is not interpretable.

### 2.6 Site P&L views (reference)

`prediction_pnl` (the site's per-bet $10 view, median closing price, all graded games): CFB moneyline +$43.9 on 245 bets (+1.8%), CFB spread -$98.2 (-3.1%), CFB total -$70.2 (-2.2%), NFL moneyline -$20.0, NFL spread -$53.9, NFL total -$169.1 (-21.7%). `ev_pnl_daily` only shows rows after each sport's `track_record_start` (NFL restarted 2026-09-29), so it shows 288 NFL props rather than the 452 analyzed here.

## 3. Calibration findings

**CFB and NFL win probabilities are compressed toward 50% relative to the market.**

| | CFB (n=195 with Pinnacle ML) | NFL (n=77) |
|---|---|---|
| Model p regressed on market p (slope) | 0.75 (intercept 0.14) | 0.64 (0.20) |
| Logistic recalibration slope, model alone (full sample) | 1.26 (SE 0.16; n=320) | 1.21 (SE 0.46; n=78) |
| Outcome ~ logit(market) + (model - market), coefficient on the difference | -0.32 (SE 0.33) | +0.58 (SE 0.70) |
| Brier: model / Pinnacle no-vig close (paired) | 0.1894 / 0.1623 | 0.2255 / 0.2231 |

Read this carefully. Relative to realized outcomes, the model is mildly under-confident (slope above 1, though the CI includes 1). Relative to the market, it is far less extreme. The market wins the Brier comparison decisively in CFB. Stretching the model's probabilities would not create an edge: after calibration, the model-minus-market term carries no information (-0.32 and +0.58, both well inside 2 SE).

The practical consequence is in the bets. The "value side" is mostly an underdog, because the model's compression pushes it above the market for underdogs. Betting the model's ML value side at the Pinnacle close price, with vig:

| Sport | Edge cut | N | Hit | ROI (95% CI) |
|---|---|---|---|---|
| CFB | any | 195 | 36.4% | -18.9% (-37.8, +1.8) |
| CFB | 4+ pp | 136 | 30.9% | -28.9% (-49.0, -6.9) |
| CFB | 8+ pp | 99 | 26.3% | **-41.9% (-62.3, -20.0)** |
| CFB | opener price, any | 195 | 36.4% | -21.0% (-38.5, -0.5) |
| NFL | any | 76 | 43.4% | +3.7% (-25.0, +34.5) |
| NFL | 8+ pp | 32 | 46.9% | +23.8% (-26.1, +74.1) |
| NFL | opener price, 8+ pp | 30 | 53.3% | +33.0% (-15.7, +83.7) |

CFB is credibly negative. The NFL positive is a 30-game result with CIs spanning -16% to +84%. It is the thing to retest prospectively (section 6) and is not an edge yet. Always betting CFB underdogs at the Pinnacle close: -26.9% (95% CI [-47.5%, -4.1%]); always betting favorites: +4.1% [-4.7%, +12.5%]. The historical 2021-25 CFB data show how misleading that last number is: favorites lost 5.1% (95% CI [-7.2%, -3.0%]) over 3,561 games, and dogs lost 7.0%. NFL 2006-26 closing moneylines (5,115 games): favorites -3.3% [-5.3%, -1.4%], dogs -3.3% [-7.2%, +0.7%]; dogs priced below 20% win probability lose 12% to 57%.

**Cover, over and prop probabilities are strongly overconfident.**

| Quantity | N | Claimed P(win) | Actual | Recalibration slope (SE) |
|---|---|---|---|---|
| CFB spread pick prob (`spread_pick_prob`) | 315 | 61.2% | 50.8% | 0.14 (0.29) |
| CFB total pick prob | 320 | 61.0% | 51.2% | 0.26 (0.29) |
| NFL spread pick prob | 74 | 56.5% | 48.6% | 0.39 (0.94) |
| NFL total pick prob | 78 | 56.8% | 41.0% | -1.67 (0.92) |
| NFL prop leans, all (`nfl_prop_grades`) | 1,587 | 64.8% | 51.8% | **0.15 (0.07)** |
| Props board picks (both sports) | 542 | 58.0% | 50.9% | 0.76 (0.32) |
| NFL props board only | 452 | 58.1% | 49.8% | 0.44 (0.35) |
| Pre-pause MLB picks | 1,972 | 57.2% | 50.9% | 0.82 (0.13) |

On 87% of CFB spread and total picks the model's claimed probability is above the -110 breakeven; the mean claim is 61% and the delivered hit rate 50.8% to 51.2%. In the NFL prop case only about 15% of the model's deviation from 50% is real (slope 0.15, n=1,587, tight CI). This audit measures the symptom, not its cause; the cause was not diagnosed here.

## 4. Timing and opener findings

**Does the line move toward the model between the first capture and the close?** Pinnacle first capture is a median 139 h (CFB) and 141 h (NFL) before kickoff, so it is close to a true opener. Regress (close margin - open margin) on (model margin - open margin):

| Sport / model | Spread slope (SE) | N | Total slope (SE) | N |
|---|---|---|---|---|
| CFB, all | +0.015 (0.016) | 211 | +0.031 (0.018) | 211 |
| CFB v1 | +0.022 (0.016) | 123 | +0.004 (0.021) | 123 |
| CFB v2 | -0.004 (0.035) | 88 | +0.066 (0.031) | 88 |
| NFL, all | +0.046 (0.035) | 76 | -0.039 (0.032) | 76 |
| NFL Elo | +0.050 (0.043) | 45 | -0.053 (0.033) | 45 |
| NFL ML sim | +0.039 (0.060) | 31 | -0.079 (0.074) | 31 |

Interpretation: the market moves 2% to 5% of the model's disagreement toward the model on spreads, and not at all (or against) on NFL totals. A 10-point disagreement is worth about half a point of movement. Opener and close differ on only 44% of CFB and 58% of NFL games, and the standard deviation of the move is 1.1 to 1.5 pts. The single t > 2 (CFB v2 totals, 2.1) is one of 24 tests. The NFL ML sim contains market features, so its disagreement with the opener is partly the market. CFB and NFL Elo are model-only (no market input), so they are the clean tests.

**Verdict on a timing edge in the model: none supported.** Even if the close is unbeatable, a model that does not move lines gives early betting nothing to harvest.

**A structural timing effect that is real, historical and small: CFB totals fall from opener to close.**

| Season | Games | Mean (close - open total), pts | SE | Under hit at the open | Under hit at the close |
|---|---|---|---|---|---|
| 2021 | 730 | +0.18 | 0.08 | 53.8% | 53.4% |
| 2022 | 723 | -0.50 | 0.09 | 52.3% | 50.6% |
| 2023 | 741 | -0.78 | 0.07 | 49.9% | 48.9% |
| 2024 | 751 | -0.44 | 0.07 | 49.9% | 48.0% |
| 2025 | 756 | -0.54 | 0.08 | 53.3% | 51.3% |
| 2021-25 | 3,701 | **-0.42** | 0.035 | 51.9% | -- |

Betting every under at the opener returns -1.0% (95% CI [-4.1%, +2.0%]) at -110. The drift is worth about 1.4 pp of win probability against betting at the close (under hit 51.9% at the open vs about 50.4% at the close), which roughly offsets the vig; at -105 (a best-of-books price) breakeven is 51.2% and the history shows 51.9%, i.e. about +1% ROI, an effect that would take about 70,000 bets to confirm from outcomes. It must be judged by CLV. Live 2026: -0.37 pts (n=211, t = -3.8; weekly -0.01, -0.40, -0.47, -0.15, -0.83); opener-to-close under hit rate 48.8% (n=207, noise). In the live sample, the best-US-book first-pull under had a closing EV vs Pinnacle fair of +0.2% [-1.2%, +1.7%] against -2.4% at bet time, while CFB overs lost 5.5%; every other CFB and NFL spread and total side at the first pull lost 1.7% to 3.8% by the close. NFL shows no totals drift (-0.06 pt, n=77). CFB spreads show no stable favorite-direction drift (-0.09 pt favorite growth over 2021-25, SE 0.03, mixed by season).

**Timing of pick generation.** `game_predictions` is overwritten on each run and `generated_at` is after kickoff for most graded games (median 20 h after for CFB), so the stored model number cannot be audited for "pre-game only". CFB uses a week-entering ratings state (checked in `generate_cfb.py`), so regeneration after kickoff should not leak; I did not verify the same for NFL Elo or the NFL ML sim, and the ML sim includes market features.

## 5. Line-shopping findings

All pulls are one cross-section: every book's price at the same `captured_at`. Best US price = best of DraftKings, FanDuel, Caesars (williamhill_us), BetRivers, BetMGM, Fanatics (and others if present); offshore books excluded. Fair value = Pinnacle no-vig at the same pull. Same-line = identical number to Pinnacle's.

**How often is the best US price better than Pinnacle's fair price?** Per pull (game x side):

| Sport / market | Pulls | Mean EV vs Pinnacle fair | % of pulls EV > 0 | % of pulls EV > 1% | Max |
|---|---|---|---|---|---|
| CFB spread | 5,568 | -3.84% | 1.0% | 0.22% | +1.8% |
| CFB total | 4,630 | -4.01% | 0.7% | 0.19% | +1.9% |
| NFL spread | 1,900 | -3.58% | 4.3% | 0.42% | +4.3% |
| NFL total | 1,792 | -3.88% | 1.1% | 0.50% | +1.7% |
| CFB moneyline | 6,250 | -0.89% | 18.1% | 15.2% | +47.9% |
| NFL moneyline | 2,344 | -2.31% | 13.0% | 9.5% | +15.2% |

Best-of-six US books is still 2% to 4% worse than Pinnacle's no-vig price on average: shopping alone does not beat the vig. On spreads and totals at the same number, +EV is almost never available. Moneylines show large "edges" but 98% of the CFB ones are underdogs (see executive summary item 3): treat them as a de-vig artifact, not value.

**Do the same-line spread and total opportunities beat the close?** No. For flagged same-line spreads and totals (EV over 0 at pick time, fair 15% to 85%), EV at the Pinnacle valid close was -4.6% [-10.3%, +0.2%] (CFB spread, 15 games), -2.8% [-5.3%, +0.8%] (CFB total, 10), -0.8% [-2.2%, +0.4%] (NFL spread, 13) and -6.3% [-6.7%, -0.1%] (NFL total, 2). Those prices were stale quotes that the close erased.

**Line-adjusted shopping (a different number than Pinnacle's, value converted with a normal approximation at sigma = 13 pts).** First flagged pull per game and side, EV at pick time and EV against the Pinnacle valid close (95% CI, game-clustered):

| Sport / market | Cut | Bets | Median h before KO | EV at pick | EV at close (95% CI) | Realized ROI (95% CI) |
|---|---|---|---|---|---|---|
| CFB total | 1%+ | 57 | 139 | +2.4% | **+2.9% (+1.0, +4.9)** | -13.5% (-37.2, +9.8) |
| CFB total | 2%+ | 31 | 142 | +3.3% | **+4.6% (+1.8, +7.7)** | -15.1% (-47.0, +17.3) |
| CFB spread | 1%+ | 64 | 122 | +2.9% | +1.3% (-1.1, +3.9) | -7.0% (-26.3, +13.6) |
| CFB spread | 2%+ | 44 | 122 | +4.1% | +2.3% (-0.1, +4.8) | -8.3% (-34.6, +17.3) |
| NFL spread | 1%+ | 21 | 114 | +2.5% | 0.0% (-1.6, +1.9) | +9.0% (-26.8, +45.5) |

(Games with a Pinnacle valid close; first flagged pull at least 1 h before that close; Pinnacle fair between 15% and 85%. NFL totals produced only 2 flagged games and are omitted.) The CFB totals line holds up against the close at about +3% EV, which is the one place the data suggests shopping across numbers, early in the week, adds value. The caveats are decisive for calling it unproven: it uses a smooth normal conversion for half-points (understating the value of the key numbers 3 and 7); the realized ROI is -13% to -15% with CIs that swallow +10%; N is 57 games; and about 6 weeks of data produced it. Part of the CFB total result is likely the opener drift above (early soft-book totals tend to be the highest numbers).

**Lifespan.** Median pull interval is 1.9 h (CFB) and 2.1 h (NFL). For line-adjusted EV above 1%: 62% of CFB runs last a single pull, median run length of 1 pull; NFL spreads last a median of 2 pulls, 2.4 h, p90 51 h. The data cannot resolve less than about 2 h. If these require manual execution inside a 1 to 2 h window, many would vanish.

**Props shopping (US-consensus fair; Pinnacle has no prop prices).** At pulls with 4 or more US books quoting the same line (1,849 pulls, 3,698 side-rows; 1,398 side-rows have a later valid close at least 1 h away), the best price beats the leave-one-out consensus (EV > 0) in 8.3% of those side-rows and by 2% or more in 2.4%. Where EV was above 0 at pick time, it was +1.3% (NFL, 38 pulls) and +1.6% (MLB, 78); by the close it was +0.0% [-1.2%, +0.9%] and -0.6% [-1.1%, 0.0%]. Prop shopping opportunities exist, but the data shows they disappear by the close. Coverage is thin (21 NFL games with a valid close).

## 6. Sample-size realism

Required N is for a two-sided 95% test with 80% power, using the observed standard deviation: N = (2.8 x SD / effect)^2. "Weeks" uses the current pace.

| Candidate | Observed effect | Per-bet SD | N required | Pace | Weeks |
|---|---|---|---|---|---|
| ROI of any 2% edge (any product) | +2% | 1.0 | 19,600 bets | -- | not achievable; judge by CLV |
| NFL prop board ROI (if true value were +3%) | +3% | 0.955 | 7,950 | 133/wk | about 60 |
| NFL prop board, confirm a -4% loss | -4% | 0.955 | 4,475 | 133/wk | about 34 |
| Desk fair-move CLV (retired product) | -0.44 pp | 2.26 pp | 207 picks | n/a | n/a (currently 98) |
| NFL prop fair-move CLV | -0.18 pp | 1.44 pp | about 500 | -- | needs a valid close on 3x more props |
| CFB total line-adjusted shopping | +2.9% EV | 8.0% | about 62 bets (current n=57) | about 10/wk | 6 |
| CFB/NFL spread line-adjusted shopping | +1.0% EV | 9.7% | about 720 | about 15/wk | about 48 |
| CFB model information beyond close | coefficient +0.10 | SE 0.156 at n=320 | about 6,100 games | about 55/wk | about 110 (use history) |
| CFB opener-drift under bets ROI | +1% | 0.95 | about 70,000 | -- | not testable; CLV test needs about 200 games |

Take-away: ROI cannot be used to validate a 2% to 3% edge within a season. CLV and the fair-move statistic converge 10x to 100x faster. The CFB total shopping line is the only candidate that is both promising and testable within about 6 to 12 weeks, because it uses CLV on a high-frequency market.

## 7. Recommended roadmap

Priority order. Each item has a pre-registered pass/fail test, to be fixed before data is looked at.

1. **Make the measurement trustworthy (prerequisite, a few days).**
   Build: an append-only `pick_ledger` that records, at the moment a bet is surfaced, every US book's line and price, Pinnacle's no-vig fair and the US-consensus fair, the model probability, and a frozen timestamp; a pre-kick capture at T-15 min for every game and prop; and a CLV computed against a valid close with line-aware adjustment. Retire the price-only spread and total CLV.
   Pass: at least 95% of ledger rows have a valid close (capture within 30 min of kickoff) across 2 consecutive weekends; at least 90% of prop rows have a US-consensus close from at least 3 books at the pick's line; 0 rows overwritten. Fail: any week below those rates blocks every other item.

2. **Shrink every model probability to the market, then stop model-only bets that fail (about 1 week).**
   Build: replace the pick probability with market_novig + k x (model - market), fitting k walk-forward (current evidence implies k about 0.15 for props and about 0.1 to 0.3 for cover probabilities); gate props and cover/total picks on the shrunk edge only.
   Pass: on at least 2,000 graded prop legs (NFL leans table already has 1,587), the shrunk probability beats the market no-vig on Brier by at least 0.001 with a paired-bootstrap 95% CI excluding 0, and calibration slope within [0.8, 1.2]. Fail: model-derived props and cover/total picks are removed from the betting product and kept only as research output.

3. **Early-week stale-number monitor for CFB totals and spreads and NFL spreads, with paper tickets (about 1 to 2 weeks to build, 8 to 12 weeks to judge).**
   Build: at each pull in the first 24 h after lines post, flag a US-book number or price whose line-adjusted EV against Pinnacle fair is at least 2%; log a paper ticket with the exact bettable price; grade by CLV against the T-15 min Pinnacle close, with real stakes only after the test passes. Use a key-number-aware line value (empirical, from the 2021-25 CFB and nflverse histories) in place of the normal approximation.
   Pass (pre-registered): at least 150 pooled flagged tickets (CFB totals + CFB spreads + NFL spreads), including at least 60 CFB-totals tickets, each tested separately for: mean EV vs Pinnacle valid close of at least +1.5%, the game-clustered 95% lower bound above 0, and at least 55% beating the close. Fail: drop it. A positive CLV result still does not prove profit: the next gate is realized ROI over at least 300 tickets with the lower bound above -3%.

4. **Test the NFL ML-sim moneyline signal prospectively (low cost, low prior).**
   Pre-register: freeze the model, bet nothing; log edge versus Pinnacle opener. Pass: at least 150 NFL games with model-minus-market win-probability gap of 4 pp or more, paper ROI at the opener above 0 with the lower 95% bound above -10%, and the calibrated gap coefficient (outcome on logit(market) + gap) above 0 with p < 0.05. At the current pace of about 16 NFL games per week this takes more than a season, so also run the same test on the 2021-25 backtest (the props-ML infrastructure) before spending live weeks.

5. **Do not build further**: more model versions for CFB or NFL margin and total picks (v3 and v3.1 failed their gates; the closing line beats the model on MAE in every cut, and the model-minus-market coefficient is +0.04 +/- 0.16); more Desk-style overlay; prop shopping with US-only consensus (its edge disappears by the close); parlays of model-only legs.

## 8. Data gaps that block conclusions

1. **No sharp reference for props.** `odds_snapshot` has zero Pinnacle prop rows, so the board and grader CLV are NULL for all 542 graded props. Only 39% of prop picks have a 2-or-more-book US-consensus close at their exact line. A sharp prop source (or a deliberate consensus of 5 or more books captured at T-15 min) is required to measure prop CLV.
2. **Stale closes.** The last Pinnacle capture is more than 12 h before kickoff for 53% of NFL and 24% of CFB games; only 44% of NFL and 71% of CFB games have a capture within 3 h (the crons are Sun/Mon/Thu/Sat only and cluster at 16:00 to 23:00 UTC). All CLV in this report uses valid closes only, which cuts n by 30% to 60%.
3. **The +EV board overwrites itself.** `ev_picks` and `ev_prop_picks` keep the last-build `best_price`, `true_prob` and `ev_best`; the pick-time price survives only for `created_at` and the (mostly NULL) `open_pinnacle_price`. The ROI in `ev_pnl` is therefore at the last-build price, not the first-surfaced one.
4. **No append-only prediction history.** `game_predictions` is upserted and `generated_at` is after kickoff for most graded games; `prediction_accuracy` has no `model_version`. We cannot prove that the stored number was pre-game, or which version produced it (CFB v1 and v2, NFL Elo and ML sim are mixed).
5. **The graded +EV game-line sample is entirely retired Desk picks.** The pure line-shopping board has 0 graded picks, so line-shopping CLV rests on reconstructed snapshots, not on picks the product actually made.
6. **Short and uneven history.** 5 to 6 weeks of football; MLB has 1 week pre-pause and 1 week postseason; 11 MLB games with a Pinnacle quote; about 110 captured games (mostly non-modelled football and Aug MLB) have no sport label; 48 `game_pk`s carry conflicting kickoff times (Aug MLB) and were excluded; NFL weeks 1 to 2 have no prop lines.
7. **Closing lines come from different sources.** `prediction_accuracy.market_spread` and `market_total` are the ESPN pickcenter number (one book); Pinnacle's last capture differs by up to 5 pts when stale (mean difference NFL -0.18 pt, CFB +0.09 pt, SD 0.65 to 0.85).
8. **No open prices at US books for most picks.** Opening-line analysis uses Pinnacle's first capture. US books often post their own openers later and at different numbers, which the shopping analysis in section 5 captures only from the first pull on.
9. **Historical data that could add power but is not used by the live system.** `assets/cfb/lines.parquet` (CFBD consensus open and close and moneyline, 2021-25) and the nflverse schedules (closing spread, total, moneyline since 2002) give thousands of games to test structural effects (as done in sections 3 and 4); there are no historical props odds or Pinnacle prices.

## Method appendix

- **Sources.** `prediction_accuracy` (409 rows), `game_predictions`, `ev_picks` / `ev_results` (1,640 / 165), `ev_prop_picks` / `ev_prop_results` (1,170 / 542), `ev_best_parlays` (14), `picks` legacy (2,004), `nfl_prop_grades` (1,587), `odds_snapshot` (1.44M rows; 1.12M game-line rows and 321k prop rows extracted), `track_record_start`, `desk_pick_results` (retired; its price-based CLV values are not comparable and are not reported), and the local `assets/cfb/lines.parquet`, `assets/cfb/schedules.parquet`, `assets/nfl/schedules.parquet`.
- **Game-line snapshots.** Pairs of sides at the same `captured_at`; spread pairs require line(away) = -line(home). Captures after kickoff dropped. Games with more than one commence time dropped (48).
- **No-vig.** Multiplicative (pa/(pa+pb)). It overstates longshot probability (favorite-longshot bias); that is why high-"EV" underdog moneylines are discounted.
- **Line value.** For a bet at a different number than the close, the closing no-vig probability is shifted with Phi(Phi^-1(p) + advantage / sigma), sigma = 13 points, advantage in points of line in the bettor's favor. This ignores key numbers and understates the half-point at 3 and 7; all line-adjusted EV results carry that caveat.
- **Bootstrap.** 4,000 resamples (3,000 clustered by game_pk for multi-pick games, props, parlays and shopping rows); percentile 95% intervals. Wilson intervals for hit rates. Logistic fits by BFGS with Fisher-information SEs. OLS SEs are classical.
- **Valid close.** Last capture at or before kickoff and within 3 h of it. For props, at least 2 US books at the pick's exact line (3 for shopping tests); US books = DraftKings, FanDuel, Caesars, BetRivers, BetMGM, Fanatics, plus Hard Rock, ESPN BET, theScore, bet365, Bally if present.
- **Multiple comparisons.** Roughly 150 intervals appear in this report; at 95%, about 7 would exclude the truth by chance. Single hits (NFL totals, CFB v2 totals movement) are labelled accordingly.
- **Scripts.** Scratch analysis only, in the session scratchpad (`edge-audit/`), not committed.
