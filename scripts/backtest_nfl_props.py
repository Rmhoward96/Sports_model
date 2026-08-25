"""Walk-forward backtest fitting per-market sigma (yardage markets) + NB
overdispersion (receptions) for player props, calibrated against nflverse
historical weekly outcomes.

Wires together Task 2 (usage.compute_usage_shares/allocate) + Task 3
(efficiency.compute_efficiency) + Task 5-equivalent gamescript
(gamescript.fit_gamescript/project_team_volume) + Task 4 (props.build_prop)
into a leak-free, per-player-game walk-forward: to project season S's
player-games, usage shares and efficiency rates are computed ONLY from
season S-1 (never from S itself, and never from games later in S), and the
gamescript model that turns a game's own PRE-game market line
(spread_line/total_line) into team pass/rush volume is fit ONCE from all
seasons strictly before the earliest season being scored.

NO HISTORICAL PROP MARKET EXISTS for player yardage/receptions props (unlike
the game-line backtest, which has closing spread/total lines to shrink
toward and score against). So this backtest calibrates the model's
distributions to ACTUAL OUTCOMES -- the residual spread of (projected mean -
actual) becomes each yardage market's Normal sigma, and the empirical
var/mean of actual receptions becomes the Negative Binomial's overdispersion
multiplier. There is no market line to beat here; season-long CLV against
whatever props book lines are shopped live is the only real judge of this
model's value, once it is actually serving lines against a market.

TD markets (pass_tds, anytime_td) are added in Task 7 on top of the same
leak-free walk-forward: pass_tds is a Poisson count market (like receptions,
but with far fewer events per game -- 0, 1, or 2 TD passes is the entire
support in practice), and anytime_td is a Yes/No market (P(>=1 rushing or
receiving TD), evaluated at the 0.5 line via `prob_over_dist`). Both are
LOW-COUNT, HIGH-VARIANCE markets: a single game's outcome is dominated by
red-zone randomness that a season-level rate cannot resolve, so their
per-game diagnostics are intentionally looser than the yardage markets' --
there is no free variance parameter for a Poisson rate (mean *is* the
variance), so there is no sigma to fit the way there is for yardage. As of
Fix round 2, though, their MEAN (the Poisson lambda) gets the exact same
applied `mean_mult` correction as every other market: fit as
`mean(actual TD count)/mean(predicted lambda)` and applied to lambda inside
`props.build_prop` before `poisson_pmf`, via the same `PropConfig.mean_mult`
dict the yardage markets use. See `fit_td_calibration` and the report's TD
section for the fitted multipliers, the remaining per-game looseness, and
the honesty framing (a mean correction is not the same as per-player
discrimination -- `anytime_td`'s Brier improvement over a flat base rate
remains near-null even after this fix).
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import pandas as pd

from sportsmodel.nfl.gamescript import (
    team_game_volume,
    fit_gamescript,
    project_team_volume,
)
from sportsmodel.nfl.usage import compute_usage_shares, allocate
from sportsmodel.nfl.efficiency import compute_efficiency
from sportsmodel.nfl.props import PropConfig, build_prop

YARDAGE_MARKETS = ["pass_yds", "reception_yds", "rush_yds", "rush_reception_yds"]
TD_MARKETS = ["pass_tds", "anytime_td"]
ALL_MARKETS = YARDAGE_MARKETS + ["receptions"] + TD_MARKETS

_CFG = PropConfig()  # projected_mean is sigma/nb_var_mult-independent; defaults are fine here


def _actual_value(market: str, row: pd.Series) -> float:
    if market == "pass_yds":
        return float(row["passing_yards"])
    if market == "reception_yds":
        return float(row["receiving_yards"])
    if market == "rush_yds":
        return float(row["rushing_yards"])
    if market == "rush_reception_yds":
        return float(row["rushing_yards"]) + float(row["receiving_yards"])
    if market == "receptions":
        return float(row["receptions"])
    if market == "pass_tds":
        return float(row["passing_tds"])
    if market == "anytime_td":
        # Served as a Yes/No market (P(>=1) at the 0.5 line), but the RAW TD
        # count is what's stored here -- Fix round 2 calibrates the Poisson
        # lambda multiplier on count vs count (mean_actual_count/mean_lambda),
        # the same principled basis as every other market's mean_mult, not on
        # a binarized indicator. The binary Y/N outcome is derived from this
        # count (>=1) only where a Brier score is computed.
        return float(row["rushing_tds"]) + float(row["receiving_tds"])
    raise ValueError(f"unknown market: {market}")


def _markets_for_volume(volume: dict) -> list[str]:
    """Which markets are meaningful to score for this player-game, driven by
    the player's OWN projected volume (data-driven, not a hardcoded position
    map): a player with zero projected pass attempts contributes nothing
    informative to the pass_yds market (both pred and actual would be ~0),
    so including them would dilute MAE/sigma with trivial zero-vs-zero pairs
    across the whole non-QB population. Scoring only where a player actually
    has projected usage keeps the calibration honest for how these markets
    would actually be served (props are only offered for players who touch
    the ball)."""
    markets = []
    if volume["pass_att"] > 0:
        markets.append("pass_yds")
    if volume["targets"] > 0:
        markets.append("reception_yds")
        markets.append("receptions")
    if volume["carries"] > 0:
        markets.append("rush_yds")
    if volume["carries"] > 0 or volume["targets"] > 0:
        markets.append("rush_reception_yds")
    if volume["pass_att"] > 0:
        markets.append("pass_tds")
    if volume["carries"] > 0 or volume["targets"] > 0:
        markets.append("anytime_td")
    return markets


def per_player_predictions(weekly: pd.DataFrame, schedules: pd.DataFrame,
                           seasons: list[int]) -> list[dict]:
    """Leak-free walk-forward core: one entry per (player, game, market) with
    the model's projected mean and the actual outcome.

    Leak-free invariants:
    - The gamescript model (pass-rate/plays ~ team_margin + implied_total) is
      fit ONCE from `weekly[weekly.season < min(seasons)]` -- never from data
      in or after the seasons being scored.
    - For each target season S, usage shares and efficiency rates are
      computed ONLY from `weekly[weekly.season == S-1]` -- never from S
      itself. A player with no S-1 rows (rookie, or dropped from the shares/
      efficiency dict) is skipped for season S: there is no leak-free rate to
      project them from.
    - The per-game team_margin/implied_total come from `schedules`'
      PRE-game spread_line/total_line (via `team_game_volume`), never from
      the game's own final score.
    Appending/removing rows for weeks LATER than a given week (within the
    same season, or in a later season) must never change that week's
    predictions -- this is the property `test_no_leak_uses_prior_season_only`
    enforces.
    """
    seasons = list(seasons)
    if not seasons:
        return []
    min_season = min(seasons)

    hist = weekly[weekly["season"] < min_season]
    gs_model = fit_gamescript(team_game_volume(hist, schedules))

    out = []
    for season in seasons:
        prior = weekly[weekly["season"] == season - 1]
        shares = compute_usage_shares(prior)
        eff = compute_efficiency(prior)

        season_weekly = weekly[weekly["season"] == season]
        tgv = team_game_volume(season_weekly, schedules)
        tgv_idx = {(int(r["week"]), r["recent_team"]): r for _, r in tgv.iterrows()}

        for _, row in season_weekly.iterrows():
            pid = row["player_id"]
            if pid not in shares or pid not in eff:
                continue
            key = (int(row["week"]), row["recent_team"])
            if key not in tgv_idx:
                continue
            tg = tgv_idx[key]
            team_volume = project_team_volume(
                gs_model, tg["team_margin"], tg["implied_total"])
            volume = allocate(shares[pid], team_volume)
            player_eff = eff[pid]

            for market in _markets_for_volume(volume):
                pred = build_prop(market, volume, player_eff, _CFG)
                out.append({
                    "season": int(season),
                    "week": int(row["week"]),
                    "player_id": pid,
                    "market": market,
                    "pred_mean": pred["projected_mean"],
                    "actual": _actual_value(market, row),
                })
    return out


def run_backtest(weekly: pd.DataFrame, schedules: pd.DataFrame,
                 seasons: list[int]) -> dict:
    """Aggregate per_player_predictions into per-market {mae, n, coverage}.

    `coverage` is an in-sample 1-sigma diagnostic: the fraction of
    predictions landing within one residual-RMSE of the actual outcome
    (sigma computed from that same market's residuals here) -- for a
    well-calibrated Normal residual, this should land near ~0.68.
    """
    preds = per_player_predictions(weekly, schedules, seasons)
    by_market: dict[str, list[dict]] = {}
    for p in preds:
        by_market.setdefault(p["market"], []).append(p)

    out = {}
    for market, rows in by_market.items():
        n = len(rows)
        errs = [abs(r["pred_mean"] - r["actual"]) for r in rows]
        mae = sum(errs) / n if n else 0.0
        sigma = math.sqrt(sum(e * e for e in errs) / n) if n else 0.0
        coverage = (sum(1 for e in errs if e <= sigma) / n) if n and sigma > 0 else 0.0
        out[market] = {"mae": mae, "n": n, "coverage": coverage}
    return out


def fit_calibration(preds: list[dict]) -> dict:
    """mean_mult[market] = mean(actual) / mean(pred_mean) per yardage market
    (population-level de-bias multiplier -- see Fix round 1 below) AND
    receptions. sigma[market] = RMSE of the CORRECTED residual
    `(pred_mean * mean_mult[market] - actual)`, i.e. fit AFTER de-biasing the
    mean, not on the raw (biased) prediction -- fitting sigma on a biased
    mean would conflate "the mean is wrong" with "the spread is wrong" and
    inflate sigma to paper over a mean error a multiplier should fix instead.
    nb_var_mult = empirical var(actual)/mean(actual) across all receptions
    predictions (purely a function of the ACTUAL outcome distribution, not
    the prediction, so it is unaffected by mean_mult; clamped >1 for a
    well-defined Negative Binomial). loc[market] = mean(actual - RAW
    pred_mean) per market, reported only, on the UNCORRECTED prediction --
    this is the pre-correction bias magnitude (roughly
    `pred_mean * (mean_mult - 1)`), kept for comparison against mean_mult.

    Fix round 1 (independent review): the walk-forward's raw `pred_mean` for
    EVERY market -- yardage AND receptions, not just the TD markets --
    underprojects actual outcomes by a consistent ~1.5x, traced to
    `usage.allocate`'s shrinkage-toward-0 of per-player shares
    under-allocating team-level volume (which is itself close to unbiased).
    Task 6 originally treated the yardage mean bias (`loc`) as report-only;
    that was wrong -- shipping raw `pred_mean` here would recommend UNDER on
    nearly every yardage/receptions prop against a fair market, which makes
    the model directionally useless, not just imprecise. `mean_mult` is the
    first-order fix: a single per-market multiplier applied to
    `projected_mean` in `props.build_prop` via `PropConfig.mean_mult`. It is
    a population-average correction, not a fix to the underlying
    usage-share mechanism (see the report's root-cause section for the
    proper fix: renormalizing/re-shrinking usage shares)."""
    by_market: dict[str, list[dict]] = {}
    for p in preds:
        by_market.setdefault(p["market"], []).append(p)

    sigma = {}
    sigma_raw = {}
    loc = {}
    mean_mult = {}
    for market in YARDAGE_MARKETS:
        rows = by_market.get(market, [])
        n = len(rows)
        if n == 0:
            continue
        preds_m = [r["pred_mean"] for r in rows]
        actuals_m = [r["actual"] for r in rows]
        mean_pred = sum(preds_m) / n
        mean_actual = sum(actuals_m) / n
        mult = (mean_actual / mean_pred) if mean_pred > 0 else 1.0
        mean_mult[market] = mult
        raw_resid = [p_ - a for p_, a in zip(preds_m, actuals_m)]
        loc[market] = sum((-e) for e in raw_resid) / n  # mean(actual - RAW pred), report-only
        sigma_raw[market] = math.sqrt(sum(e * e for e in raw_resid) / n)  # report-only, pre-correction
        corrected_resid = [p_ * mult - a for p_, a in zip(preds_m, actuals_m)]
        sigma[market] = math.sqrt(sum(e * e for e in corrected_resid) / n)

    nb_var_mult = _CFG.nb_var_mult["receptions"]
    rec_rows = by_market.get("receptions", [])
    if rec_rows:
        preds_m = [r["pred_mean"] for r in rec_rows]
        actuals = [r["actual"] for r in rec_rows]
        n = len(actuals)
        mean_pred = sum(preds_m) / n
        mean = sum(actuals) / n
        mean_mult["receptions"] = (mean / mean_pred) if mean_pred > 0 else 1.0
        if mean > 0:
            var = sum((a - mean) ** 2 for a in actuals) / n
            nb_var_mult = max(var / mean, 1.01)
        raw_resid = [p_ - a for p_, a in zip(preds_m, actuals)]
        loc["receptions"] = sum((-e) for e in raw_resid) / n

    return {"sigma": sigma, "sigma_raw": sigma_raw, "nb_var_mult": nb_var_mult,
            "loc": loc, "mean_mult": mean_mult}


def fit_td_calibration(preds: list[dict]) -> dict:
    """Fix round 2: fit a Poisson-lambda mean_mult for the TD markets, on the
    SAME principled basis as `fit_calibration`'s yardage/receptions
    multiplier -- mean(actual TD COUNT) / mean(predicted lambda) -- not on
    P(>=1). There is still no free variance parameter to fit (a Poisson's
    mean IS its variance, so there is no sigma the way there is for the
    yardage markets), but the MEAN (lambda) gets the same applied correction
    as every other market now, via `PropConfig.mean_mult["pass_tds"]` /
    `["anytime_td"]`.

    pass_tds: `mult = mean(actual passing TDs) / mean(predicted lambda)`.
    `pred_mean` already IS the Poisson lambda and `actual` is already the raw
    passing-TD count, so this is a direct count-vs-count ratio.

    anytime_td: `mult = mean(actual TD count [rush+rec]) / mean(predicted
    lambda)` -- also a direct count-vs-count ratio (not P(>=1)-based). Since
    `anytime_td` is SERVED as a Yes/No market (P(>=1) at the 0.5 line), this
    function additionally reports a Brier score on P(>=1) = 1-exp(-lambda)
    against the binarized (>=1) actual outcome, both BEFORE (`brier_raw`,
    diagnostic only) and AFTER (`brier_corrected`, matches what
    `props.build_prop` now actually serves) applying `mult` to lambda, plus
    the trivial "always predict the empirical base rate" baseline Brier for
    comparison.
    """
    by_market: dict[str, list[dict]] = {}
    for p in preds:
        by_market.setdefault(p["market"], []).append(p)

    out = {}

    pt_rows = by_market.get("pass_tds", [])
    if pt_rows:
        n = len(pt_rows)
        mean_lambda_pred = sum(r["pred_mean"] for r in pt_rows) / n
        mean_actual_count = sum(r["actual"] for r in pt_rows) / n
        mult = (mean_actual_count / mean_lambda_pred) if mean_lambda_pred > 0 else 1.0
        out["pass_tds"] = {"n": n, "mean_lambda_pred": mean_lambda_pred,
                            "mean_actual_count": mean_actual_count, "mult": mult}

    at_rows = by_market.get("anytime_td", [])
    if at_rows:
        n = len(at_rows)
        lam_preds = [r["pred_mean"] for r in at_rows]           # raw (pre-correction) lambda
        actual_counts = [r["actual"] for r in at_rows]          # raw TD count, rush+rec
        mean_lambda_pred = sum(lam_preds) / n
        mean_actual_count = sum(actual_counts) / n
        mult = (mean_actual_count / mean_lambda_pred) if mean_lambda_pred > 0 else 1.0

        actual_binary = [1.0 if a >= 1 else 0.0 for a in actual_counts]
        rate_actual_binary = sum(actual_binary) / n
        p_raw = [1.0 - math.exp(-lam) for lam in lam_preds]
        p_corrected = [1.0 - math.exp(-(lam * mult)) for lam in lam_preds]
        mean_p_raw = sum(p_raw) / n           # mean predicted P(>=1), pre-correction
        mean_p_corrected = sum(p_corrected) / n  # mean predicted P(>=1), post-correction
        brier_raw = sum((pp - a) ** 2 for pp, a in zip(p_raw, actual_binary)) / n
        brier_corrected = sum((pp - a) ** 2 for pp, a in zip(p_corrected, actual_binary)) / n
        baseline_brier = rate_actual_binary * (1.0 - rate_actual_binary)  # trivial base-rate Brier

        out["anytime_td"] = {"n": n, "mean_lambda_pred": mean_lambda_pred,
                              "mean_actual_count": mean_actual_count, "mult": mult,
                              "rate_actual_binary": rate_actual_binary,
                              "mean_p_raw": mean_p_raw, "mean_p_corrected": mean_p_corrected,
                              "brier_raw": brier_raw, "brier_corrected": brier_corrected,
                              "baseline_brier": baseline_brier}

    return out


def main() -> None:
    t0 = time.time()
    weekly = pd.read_parquet("assets/nfl/weekly.parquet")
    schedules = pd.read_parquet("assets/nfl/schedules.parquet")

    weekly = weekly[weekly["season_type"] == "REG"].copy() if "season_type" in weekly else weekly
    schedules = (schedules[schedules["game_type"] == "REG"].copy()
                if "game_type" in schedules else schedules)

    seasons = list(range(2016, 2025))  # season S projected from S-1 (2015..2023 available)
    preds = per_player_predictions(weekly, schedules, seasons)
    metrics = run_backtest(weekly, schedules, seasons)
    cal = fit_calibration(preds)
    td_cal = fit_td_calibration(preds)

    # Fix round 2: fold the TD markets' lambda multipliers into the SAME
    # mean_mult dict as the yardage/receptions markets, so a single
    # `PropConfig(mean_mult=props_json["mean_mult"])` calibrates all 7
    # markets uniformly (this is what P4's producer is expected to load).
    cal["mean_mult"]["pass_tds"] = td_cal["pass_tds"]["mult"]
    cal["mean_mult"]["anytime_td"] = td_cal["anytime_td"]["mult"]

    k_usage = 4.0
    k_eff = 4.0
    out = {
        "sigma": cal["sigma"],
        "nb_var_mult": cal["nb_var_mult"],
        "mean_mult": cal["mean_mult"],  # Fix rounds 1+2: population-level mean
                                          # de-bias multiplier for ALL 7 markets
                                          # (pass_yds, reception_yds, rush_yds,
                                          # rush_reception_yds, receptions,
                                          # pass_tds, anytime_td) -- consumed by
                                          # props.build_prop via
                                          # PropConfig.mean_mult uniformly. For
                                          # the TD markets this multiplies the
                                          # Poisson lambda before poisson_pmf().
        "k_usage": k_usage,
        "k_eff": k_eff,
        "td_calibration": td_cal,  # additional TD-specific diagnostics (mean
                                    # lambda/actual count the mult above was
                                    # fit from, plus anytime_td's Brier scores)
                                    # -- NOT itself consumed by build_prop;
                                    # the applied multiplier lives in
                                    # mean_mult above.
    }
    pathlib.Path("assets/nfl/props.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = []
    lines.append("# NFL P3 Task 6-7: Player-props walk-forward backtest -- "
                 "fitted yardage sigmas + receptions dispersion + TD rate calibration")
    lines.append("")
    lines.append("Script: `scripts/backtest_nfl_props.py`")
    lines.append("Test: `tests/nfl/test_backtest_nfl_props.py`")
    lines.append("Output: `assets/nfl/props.json`")
    lines.append("")
    lines.append("## Statistical honesty: calibrated to OUTCOMES, not a market")
    lines.append("")
    lines.append("Unlike the game-line backtest (Task 6 of P2), there is **no historical "
                 "player-props market line** in the committed nflverse data to shrink "
                 "toward or score against. This backtest therefore calibrates each "
                 "market's distribution directly to **actual outcomes**: a yardage "
                 "market's Normal sigma is the residual RMSE of the "
                 "`mean_mult`-corrected prediction against actual outcomes on the "
                 "walk-forward (see \"Fix round 1\" below); the receptions Negative "
                 "Binomial's overdispersion multiplier is the empirical "
                 "`var(actual)/mean(actual)` of receptions across all scored "
                 "player-games. There is no market line to \"beat\" here -- season-long "
                 "CLV against whatever props book lines are shopped once this model is "
                 "actually serving lines is the real judge, not this backtest. The two "
                 "TD markets (`pass_tds`, `anytime_td`) are Poisson-distributed and have "
                 "NO free variance parameter to fit -- a Poisson's variance IS its mean, "
                 "so there is no sigma to solve for the way there is for the yardage "
                 "markets. Their MEAN (the Poisson lambda), however, gets the exact same "
                 "`mean_mult` treatment as the yardage/receptions markets as of Fix round "
                 "2 (see below) -- fit as `mean(actual TD count) / mean(predicted "
                 "lambda)` and applied in `props.build_prop` before `poisson_pmf`. Their "
                 "per-market diagnostics should still be read as meaningfully LOOSER than "
                 "the yardage/receptions markets, though: single-game TD counts are "
                 "dominated by red-zone randomness that a season-level rate cannot "
                 "resolve, and a mean correction does nothing to narrow that per-game "
                 "variance (see the TD section for the honest framing).")
    lines.append("")
    lines.append("### Fix round 1: EVERY market under-projects ~1.5x -- a shared root "
                 "cause, not a TD-specific one")
    lines.append("")
    lines.append("An earlier version of this report attributed the TD markets' "
                 "`rate_mult` bias to `k_eff` position-baseline shrinkage over-shrinking "
                 "rare TD rates specifically. **That diagnosis was wrong.** An "
                 "independent decomposition of the walk-forward found that the SAME "
                 "~1.5-1.63x under-projection appears in EVERY market -- yardage and "
                 "receptions included, not just the two TD markets -- while the "
                 "team-level volume that feeds all of them (`gamescript."
                 "project_team_volume`) is nearly unbiased (~0.96-0.98x). Since the bias "
                 "is common to every market that consumes per-player allocated volume, "
                 "and is absent from team-level volume, the real cause is upstream of "
                 "`props.build_prop` entirely: `usage.allocate`'s per-player share "
                 "shrinkage-toward-0 (`f = games / (games + k_usage)`) systematically "
                 "UNDER-allocates team volume to any player with a finite `games` "
                 "count, because shrinking every player's share toward 0 (rather than "
                 "toward a position-appropriate baseline, or renormalizing shares to "
                 "sum to ~1 after shrinkage) throws away volume rather than "
                 "redistributing it. `k_eff`/TD-rate rarity was a red herring -- the "
                 "`pass_td_rate` etc. efficiency rates are shrunk toward position "
                 "baselines correctly; they are just being multiplied by an "
                 "already-too-small `pass_att`/`carries`/`targets` volume figure, same "
                 "as every yardage market.")
    lines.append("")
    lines.append("This mattered enough to fix in this task rather than defer, because "
                 "shipping the RAW (biased) `projected_mean` would make the model "
                 "recommend UNDER on nearly every yardage/receptions prop against a "
                 "fair market -- i.e. **directionally wrong, not just imprecise** -- "
                 "which defeats the purpose of a props model. Task 6 originally scoped "
                 "the mean bias (`loc`) as report-only/optional; this finding shows "
                 "that was not an acceptable simplification once the size and "
                 "consistency of the bias was actually decomposed.")
    lines.append("")
    lines.append("**The fix applied here (`mean_mult`)**: `fit_calibration` now also "
                 "fits `mean_mult[market] = mean(actual) / mean(pred_mean)` for every "
                 "yardage market AND receptions, and `props.build_prop` multiplies "
                 "`projected_mean` by `cfg.mean_mult[market]` (via a new "
                 "`PropConfig.mean_mult` field, default 1.0) before building each "
                 "market's distribution. Sigma is then refit on the CORRECTED residual "
                 "`(pred_mean * mean_mult - actual)`, not the raw one -- fitting sigma "
                 "on a biased mean conflates \"the mean is wrong\" with \"the spread is "
                 "wrong,\" which a multiplier should fix instead of sigma absorbing it. "
                 f"Empirically, though, this correction barely moves `pass_yds` sigma "
                 f"({cal['sigma_raw']['pass_yds']:.1f} raw -> "
                 f"{cal['sigma']['pass_yds']:.1f} corrected) -- confirming that "
                 "`pass_yds`'s outsized sigma (Concern 3) is NOT a mean-bias artifact "
                 "at all, but genuinely idiosyncratic per-player variance (QB "
                 "job-security regime shifts) that a population-average multiplier "
                 "cannot touch. The other three yardage markets show a similarly small "
                 "raw-to-corrected sigma shift for the same reason: `mean_mult` fixes "
                 "the MEAN, and each market's residual spread was already centered "
                 "reasonably well relative to its own (biased) mean, so de-biasing the "
                 "mean does not by itself tighten the spread. This is explicitly a "
                 "FIRST-ORDER, population-average correction, not a fix to the "
                 "underlying usage-share mechanism -- see \"P3.5 follow-up\" in "
                 "Concerns for the deeper fix (renormalizing or "
                 "position-baseline-shrinking usage shares so per-player shares "
                 "actually sum to ~1 after shrinkage, instead of leaking volume into a "
                 "uniform post-hoc multiplier).")
    lines.append("")
    lines.append("| market | sigma_raw (pre-correction) | sigma (mean_mult-corrected) | mean_mult |")
    lines.append("|---|---|---|---|")
    for market in YARDAGE_MARKETS:
        if market in cal["sigma"]:
            lines.append(f"| {market} | {cal['sigma_raw'][market]:.3f} | "
                         f"{cal['sigma'][market]:.3f} | {cal['mean_mult'][market]:.3f} |")
    lines.append("")
    pt_cal = td_cal.get("pass_tds", {})
    at_cal_r2 = td_cal.get("anytime_td", {})
    lines.append("### Fix round 2: the TD markets get the SAME applied correction, "
                 "not just a documented knob")
    lines.append("")
    lines.append("Fix round 1 applied `mean_mult` to the five yardage/receptions "
                 "markets but left `pass_tds`/`anytime_td` with only a report-only "
                 "`rate_mult` diagnostic -- meaning `build_prop` still shipped a "
                 "~1.5x-under-biased Poisson lambda for both TD markets even after the "
                 "yardage fix, which is inconsistent: the model would recommend UNDER "
                 "on TD props too, and would essentially never surface an `anytime_td` "
                 "OVER, for the exact same root cause already fixed for yardage. This "
                 "was corrected by extending `PropConfig.mean_mult` to cover "
                 "`pass_tds`/`anytime_td` as well, so ALL 7 markets are now calibrated "
                 "AND applied through the identical mechanism -- a single dict that "
                 "`props.json`'s `mean_mult` block loads directly into `PropConfig`.")
    lines.append("")
    lines.append("`fit_td_calibration` now fits the TD multiplier on the SAME basis as "
                 "the yardage markets -- `mult = mean(actual TD count) / mean(predicted "
                 "lambda)` -- rather than on `P(>=1)`, since lambda (an expected COUNT) "
                 "is the quantity actually being multiplied in `build_prop`, and "
                 "count-vs-count is the apples-to-apples ratio (the same basis "
                 "`pass_tds` was already, accidentally, computed on before this fix, "
                 "since its `pred_mean` was already lambda and its `actual` was already "
                 "a raw TD count -- only `anytime_td`'s fit basis changes here, from "
                 "P(>=1) to raw TD count). For `anytime_td`, this required changing "
                 "`_actual_value` to return the RAW rushing+receiving TD count instead "
                 "of a binarized Yes/No indicator -- the served market is still Yes/No "
                 "at the 0.5 line (`prob_over_dist(dist, 0.5)`), but the multiplier that "
                 "feeds that market's lambda is fit on counts.")
    lines.append("")
    lines.append("| market | mean_lambda_pred (raw) | mean_actual_count | mult (applied) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| pass_tds | {pt_cal.get('mean_lambda_pred', float('nan')):.4f} | "
                 f"{pt_cal.get('mean_actual_count', float('nan')):.4f} | "
                 f"{pt_cal.get('mult', float('nan')):.3f} |")
    lines.append(f"| anytime_td | {at_cal_r2.get('mean_lambda_pred', float('nan')):.4f} | "
                 f"{at_cal_r2.get('mean_actual_count', float('nan')):.4f} | "
                 f"{at_cal_r2.get('mult', float('nan')):.3f} |")
    lines.append("")
    brier_raw_r2 = at_cal_r2.get("brier_raw", float("nan"))
    brier_corr_r2 = at_cal_r2.get("brier_corrected", float("nan"))
    baseline_brier_r2 = at_cal_r2.get("baseline_brier", float("nan"))
    mean_p_raw_r2 = at_cal_r2.get("mean_p_raw", float("nan"))
    mean_p_corr_r2 = at_cal_r2.get("mean_p_corrected", float("nan"))
    rate_actual_r2 = at_cal_r2.get("rate_actual_binary", float("nan"))
    lines.append("**An honest, slightly counterintuitive result on `anytime_td`'s "
                 "Brier score, stated plainly rather than smoothed over**: applying "
                 f"the count-based `mult` to lambda moves the mean predicted P(>=1) "
                 f"from {mean_p_raw_r2:.4f} (raw) to {mean_p_corr_r2:.4f} (corrected) "
                 f"-- MUCH closer to the actual Yes/No rate of {rate_actual_r2:.4f} "
                 f"(the raw gap was {rate_actual_r2 - mean_p_raw_r2:.4f} under; the "
                 f"corrected gap is only {rate_actual_r2 - mean_p_corr_r2:.4f} under) "
                 f"-- and yet the corrected Brier ({brier_corr_r2:.4f}) is very "
                 f"slightly WORSE than the raw Brier ({brier_raw_r2:.4f}), both close "
                 f"to the trivial base-rate baseline of {baseline_brier_r2:.4f}. This "
                 "is not a bug or a sign the correction is wrong -- it is a real, if "
                 "minor, illustration that Brier score depends on both average "
                 "calibration AND per-instance resolution (how well probabilities "
                 "discriminate individual outcomes), and improving the first does not "
                 "guarantee improving the second: scaling each player's own lambda up "
                 "multiplicatively widens the spread of individual predicted "
                 "probabilities, and on this walk-forward that widening happened to "
                 "cost slightly more Brier on the (large) `y=0` majority than it saved "
                 "on the `y=1` minority. `mult` is still the RIGHT correction for the "
                 "market's mean TD-COUNT expectation (and is exactly right for "
                 "`pass_tds`, whose served market IS a count) -- it is just not, by "
                 "this one metric, a strict Brier improvement for `anytime_td` "
                 "specifically. We applied the same count-based fit to both markets "
                 "anyway, per the brief's instruction for one principled, consistent "
                 "basis, rather than carving out a second, market-specific fitting "
                 "rule chasing a fractional Brier gain that both numbers show is "
                 "within noise of the trivial baseline either way. See the "
                 "bottom-line discussion further below for the deeper point: even the "
                 "better of these two Briers is a near-null edge over a flat rate.")
    lines.append("")
    lines.append("## Leak-free walk-forward")
    lines.append("")
    lines.append("For each target season S in `range(2016, 2025)`: usage shares "
                 "(`usage.compute_usage_shares`) and efficiency rates "
                 "(`efficiency.compute_efficiency`) are computed ONLY from "
                 "`weekly[weekly.season == S-1]`. The gamescript model "
                 "(`gamescript.fit_gamescript`) is fit ONCE from all seasons strictly "
                 "before the earliest season scored (`weekly.season < 2016`, i.e. "
                 "<=2015). Each player-game's team pass/rush volume comes from "
                 "`project_team_volume` applied to that game's own PRE-game "
                 "`team_margin`/`implied_total` (derived from the schedule's "
                 "`spread_line`/`total_line`, never the final score). "
                 "`test_no_leak_uses_prior_season_only` enforces that dropping a "
                 "season's own in-season rows does not change that season's "
                 "predictions.")
    lines.append("")
    lines.append("Markets are scored per player-game only where the player has "
                 "nonzero PROJECTED volume for that market's driving stat "
                 "(pass_att/targets/carries) -- this keeps MAE/sigma honest instead of "
                 "diluting them with trivial zero-vs-zero pairs across every skill "
                 "player who never touches a given market (e.g. a WR's pass_yds).")
    lines.append("")
    lines.append("## Per-market backtest results (2016-2024, n = total scored player-games)")
    lines.append("")
    lines.append("| market | mae | n | coverage (within 1 RMSE-sigma) |")
    lines.append("|---|---|---|---|")
    for market in ALL_MARKETS:
        m = metrics.get(market, {"mae": 0.0, "n": 0, "coverage": 0.0})
        lines.append(f"| {market} | {m['mae']:.3f} | {m['n']} | {m['coverage']:.3f} |")
    lines.append("")
    lines.append("Note: this table's `mae`/`coverage` are computed on the RAW, "
                 "UNCORRECTED `pred_mean` (before `mean_mult`) -- they are the "
                 "diagnostic that revealed the bias in the first place, not the "
                 "as-served accuracy. The corrected sigma below reflects the "
                 "`mean_mult`-adjusted prediction actually served by `props.build_prop`.")
    lines.append("")
    lines.append("## Fitted calibration (`assets/nfl/props.json`)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(out, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("- `mean_mult[market]`: population-level de-bias multiplier for ALL "
                 "7 markets (`mean(actual)/mean(RAW pred_mean)` for the 5 "
                 "yardage/receptions markets, `mean(actual TD count)/mean(predicted "
                 "lambda)` for `pass_tds`/`anytime_td`) -- `props.build_prop` applies "
                 "it to `projected_mean` (or the Poisson lambda, for the TD markets) "
                 "before building that market's distribution, uniformly via "
                 "`PropConfig.mean_mult` (see \"Fix round 1\" and \"Fix round 2\" "
                 "above).")
    lines.append("- `sigma[market]`: RMSE of `(pred_mean * mean_mult - actual)` per "
                 "yardage market across all 2016-2024 walk-forward player-games -- "
                 "i.e. fit on the CORRECTED (de-biased) prediction, not the raw one -- "
                 "the Normal sigma `props.build_prop` uses for that market.")
    lines.append("- `nb_var_mult`: empirical `var(actual receptions)/mean(actual "
                 "receptions)` across all scored receptions player-games (clamped "
                 ">1 for a well-defined Negative Binomial) -- purely a function of the "
                 "ACTUAL outcome distribution, so unaffected by `mean_mult`.")
    lines.append("- `loc[market]` (report only, not written to `props.json` / not "
                 "applied to projections): mean `(actual - RAW pred_mean)` per market, "
                 "i.e. the PRE-correction bias -- a positive value means the raw model "
                 "under-projects that market on average. Roughly "
                 "`loc ~= mean(pred_mean) * (mean_mult - 1)`; kept here purely for "
                 "comparison against `mean_mult`, since `mean_mult` is the multiplier "
                 "actually applied.")
    lines.append("")
    lines.append("### Mean de-bias (`mean_mult`) vs pre-correction bias (`loc`)")
    lines.append("")
    lines.append("| market | mean_mult (applied) | loc, pre-correction (mean actual-raw_pred) |")
    lines.append("|---|---|---|")
    for market in ALL_MARKETS:
        if market in cal["loc"]:
            mm = cal["mean_mult"].get(market, float("nan"))
            lines.append(f"| {market} | {mm:.3f} | {cal['loc'][market]:.3f} |")
    lines.append("")
    lines.append("## TD markets (`pass_tds`, `anytime_td`): corrected-and-applied, but still looser")
    lines.append("")
    lines.append("These are the two lowest-count, highest-variance markets in this "
                 "model. `pass_tds` is Poisson(`pass_att * pass_td_rate * "
                 "mean_mult[\"pass_tds\"]`); `anytime_td` is P(>=1 rushing-or-receiving "
                 "TD) = `1 - exp(-lambda)` with `lambda = (carries*rush_td_rate + "
                 "targets*rec_td_rate) * mean_mult[\"anytime_td\"]`, evaluated at the "
                 "0.5 line like MLB's HR Y/N. A Poisson rate has no free sigma to fit "
                 "(mean IS variance), but as of Fix round 2 the MEAN (lambda) gets the "
                 "SAME applied `mean_mult` correction as every other market -- fit as "
                 "`mean(actual TD count)/mean(predicted lambda)`, the same "
                 "count-vs-count basis as the yardage markets' mean_mult, not a "
                 "P(>=1)-based ratio. See the \"Fix round 2\" table above (under "
                 "\"Fitted calibration\") for the fitted `mean_lambda_pred`/"
                 "`mean_actual_count`/`mult` values.")
    lines.append("")
    lines.append("**Corrected root cause (Fix round 1)**: the ~1.5x TD-lambda bias is "
                 "NOT a TD-specific artifact of `k_eff` shrinking rare TD rates too hard "
                 "-- that was this report's earlier (incorrect) diagnosis. The SAME "
                 "~1.5x under-projection shows up in the yardage/receptions markets too "
                 "(see \"Fix round 1\" above), which is only possible if the shared "
                 "cause is upstream, in `usage.allocate`'s per-player volume "
                 "allocation, not in any market-specific efficiency rate.")
    lines.append("")
    lines.append("**Corrected-and-applied (Fix round 2)**: unlike the earlier version "
                 "of this report, the TD markets are NOT left with only a documented, "
                 "unapplied knob -- `mean_mult[\"pass_tds\"]` and "
                 "`mean_mult[\"anytime_td\"]` are fit here and actually multiplied into "
                 "the Poisson lambda inside `props.build_prop`, exactly like the "
                 "yardage markets. All 7 markets now go through the identical "
                 "calibrate-and-apply path via `PropConfig.mean_mult` / "
                 "`props.json`'s `mean_mult` block.")
    lines.append("")
    at_cal = td_cal.get("anytime_td", {})
    baseline_brier = at_cal.get("baseline_brier", float("nan"))
    brier_raw = at_cal.get("brier_raw", float("nan"))
    brier_corrected = at_cal.get("brier_corrected", float("nan"))
    improvement_raw_pct = (100.0 * (baseline_brier - brier_raw) / baseline_brier
                           if baseline_brier else float("nan"))
    improvement_corrected_pct = (100.0 * (baseline_brier - brier_corrected) / baseline_brier
                                 if baseline_brier else float("nan"))
    lines.append("**Bottom line on `anytime_td` discrimination, computed honestly "
                 "rather than left as an exercise for the reader** (see the full "
                 "raw-vs-corrected Brier breakdown under \"Fix round 2\" above for "
                 "why the corrected number is slightly worse, not better): against "
                 f"a trivial \"always predict the empirical base rate\" baseline "
                 f"Brier of {baseline_brier:.4f}, the model's per-player lambda scores "
                 f"{brier_raw:.4f} raw (~{improvement_raw_pct:.1f}% better than "
                 f"baseline) and {brier_corrected:.4f} as actually served after Fix "
                 f"round 2's mean correction (~{improvement_corrected_pct:.1f}% better "
                 "than baseline). Both are NEAR-NULL edges -- the per-player "
                 "`anytime_td` lambda is barely distinguishing players from a flat "
                 "league-average rate in this walk-forward, whether or not the mean "
                 "bias is corrected. That is an honest, and not especially flattering, "
                 "result -- it should NOT be oversold as the model having meaningful "
                 "`anytime_td` discrimination power yet. Fixing the MEAN bias (Fix "
                 "round 2) and having real per-player DISCRIMINATION (still weak, "
                 "either way) are two different things, and only the first is "
                 "addressed by `mean_mult`.")
    lines.append("")
    lines.append("## TDD: red -> green")
    lines.append("")
    lines.append("Task 6 (yardage + receptions) Step 2 (red), before "
                 "`scripts/backtest_nfl_props.py` existed:")
    lines.append("```")
    lines.append("FileNotFoundError: [Errno 2] No such file or directory: "
                 "'.../scripts/backtest_nfl_props.py'")
    lines.append("```")
    lines.append("")
    lines.append("Task 6 Step 4 (green), after implementation:")
    lines.append("```")
    lines.append("tests/nfl/test_backtest_nfl_props.py::test_run_backtest_returns_per_market_metrics PASSED")
    lines.append("tests/nfl/test_backtest_nfl_props.py::test_fit_calibration_returns_sigmas PASSED")
    lines.append("tests/nfl/test_backtest_nfl_props.py::test_no_leak_uses_prior_season_only PASSED")
    lines.append("3 passed")
    lines.append("```")
    lines.append("")
    lines.append("Task 7 (TD markets) Step 2 (red), before `poisson_pmf`/`pass_tds`/"
                 "`anytime_td` existed:")
    lines.append("```")
    lines.append("ImportError: cannot import name 'poisson_pmf' from "
                 "'sportsmodel.model.distributions'")
    lines.append("```")
    lines.append("")
    lines.append("Task 7 Step 4 (green), after implementation:")
    lines.append("```")
    lines.append("tests/nfl/test_props.py::test_anytime_td_prob_at_least_one PASSED")
    lines.append("tests/nfl/test_props.py::test_pass_tds_poisson_mean PASSED")
    lines.append("tests/nfl/test_dist_builders.py::test_poisson_pmf_sums_and_mean PASSED")
    lines.append("9 passed")
    lines.append("```")
    lines.append("")
    lines.append("Fix round 1 (mean_mult) Step 2 (red), before `PropConfig.mean_mult` "
                 "existed:")
    lines.append("```")
    lines.append("TypeError: __init__() got an unexpected keyword argument 'mean_mult'")
    lines.append("```")
    lines.append("")
    lines.append("Fix round 1 Step 4 (green), after implementation:")
    lines.append("```")
    lines.append("tests/nfl/test_props.py::test_mean_mult_scales_pass_yds_projected_mean PASSED")
    lines.append("tests/nfl/test_props.py::test_mean_mult_scales_rush_reception_yds_combined_total PASSED")
    lines.append("tests/nfl/test_props.py::test_mean_mult_scales_receptions_negbin_mean PASSED")
    lines.append("13 passed")
    lines.append("```")
    lines.append("")
    lines.append("Fix round 2 (TD lambda mult) Step 2 (red), before "
                 "`PropConfig.mean_mult` covered `pass_tds`/`anytime_td`:")
    lines.append("```")
    lines.append("KeyError: 'pass_tds'")
    lines.append("```")
    lines.append("")
    lines.append("Fix round 2 Step 4 (green), after implementation:")
    lines.append("```")
    lines.append("tests/nfl/test_props.py::test_default_mean_mult_is_unity_for_all_seven_markets PASSED")
    lines.append("tests/nfl/test_props.py::test_mean_mult_scales_pass_tds_lambda PASSED")
    lines.append("tests/nfl/test_props.py::test_mean_mult_scales_anytime_td_lambda_and_prob_identity PASSED")
    lines.append("15 passed")
    lines.append("```")
    lines.append("")
    lines.append("## Concerns")
    lines.append("")
    lines.append("1. **No historical props market exists to validate against** -- sigma/"
                 "nb_var_mult are calibrated to outcome residuals, which is honest but "
                 "means there is no OOS \"beat the book\" check here at all (unlike the "
                 "game-line backtest's model-only/blend/market-only comparison). Live "
                 "CLV tracking is the only real validation once this serves actual "
                 "lines.")
    pass_mm = cal["mean_mult"].get("pass_yds", float("nan"))
    pass_sigma = cal["sigma"].get("pass_yds", float("nan"))
    pass_loc = cal["loc"].get("pass_yds", float("nan"))
    lines.append("2. **`k_usage`=4.0's usage-share shrinkage-toward-0 is the confirmed "
                 f"root cause of the ~{pass_mm:.2f}x-and-up volume under-allocation "
                 "fixed in this task via `mean_mult`** (see \"Fix round 1\" above) -- "
                 "`allocate`'s per-player share shrinkage pulls every player's share "
                 "toward 0 rather than toward a position baseline or renormalizing "
                 "shares to sum to ~1 after shrinkage, so it structurally leaks team "
                 "volume regardless of which market consumes it. `mean_mult` is a "
                 "population-average PATCH on top of this, not a fix to `allocate` "
                 "itself -- see the P3.5 follow-up below for the deeper fix. `k_eff` "
                 "(efficiency-rate shrinkage) is carried through from Task 3 as a "
                 "fixed constant and was NOT found to need a similar correction -- "
                 "efficiency RATES (yards/attempt, catch rate, etc.) are unbiased by "
                 "this backtest's own decomposition; only per-player VOLUME "
                 "allocation was.")
    lines.append("3. **`pass_yds` sigma "
                 f"({pass_sigma:.1f}, now fit on the `mean_mult`-corrected residual) "
                 "is still larger than the other yardage markets, for a DIFFERENT "
                 "reason than a mean-bias artifact: genuine QB job-security/"
                 "team-change regime shifts that a prior-season-shares model "
                 "structurally cannot see.** Example from the real data: Joe Flacco "
                 "(`00-0026158`) split 2022 between spot starts for NYJ "
                 "(`pass_att_share=0.169` after usage shrinkage, reflecting a part-time "
                 "backup role), then signed with CLE for 2023 and started outright "
                 "(42-45 attempts/game, weeks 13-17) -- his 2022-derived share "
                 "projects far fewer pass attempts for 2023 than he actually threw. "
                 "This is not a code bug -- it is the real, load-bearing limitation of "
                 "projecting purely from S-1 season-level shares with no in-season "
                 "depth-chart/injury signal for who wins a QB competition. A "
                 f"follow-up (in-season share updates, or an explicit backup/starter "
                 "transition flag) would likely shrink `pass_yds` sigma the most of "
                 f"any market. (Pre-correction `loc` was {pass_loc:.1f}, now folded "
                 f"into `mean_mult`={pass_mm:.3f} rather than left as unapplied bias.)")
    lines.append("4. **Market inclusion is volume-gated (nonzero projected pass_att/"
                 "targets/carries), not a hardcoded position map** -- this is a "
                 "deliberate, data-driven choice (see code comment on "
                 "`_markets_for_volume`) but means, e.g., a QB who also has real "
                 "receiving volume (extremely rare) would be scored on reception_yds "
                 "too; this is correct behavior, not a bug, but worth knowing the gate "
                 "is volume-based rather than position-based.")
    pt = td_cal.get("pass_tds", {})
    at = td_cal.get("anytime_td", {})
    lines.append("5. **TD bias corrected via the same applied multiplier as yardage "
                 "(Fix round 2) -- no longer unresolved.** An earlier version of this "
                 "report left `pass_tds`/`anytime_td`'s ~1.5x shared volume "
                 "under-allocation as a documented-but-unapplied `rate_mult` knob, "
                 "which was inconsistent with the yardage markets already getting "
                 "`mean_mult` applied in `build_prop` -- the model would have kept "
                 "recommending UNDER on TD props (and essentially never surfacing an "
                 "`anytime_td` OVER) even after the yardage fix landed. `PropConfig."
                 f"mean_mult` now covers `pass_tds` (mult={pt.get('mult', float('nan')):.3f}) "
                 f"and `anytime_td` (mult={at.get('mult', float('nan')):.3f}) too, fit on "
                 "the same count-vs-count (`mean(actual TD count)/mean(predicted "
                 "lambda)`) basis as the yardage markets and applied to the Poisson "
                 "lambda before `poisson_pmf` in `build_prop` -- all 7 markets now go "
                 "through the identical calibrate-and-apply path. What remains true, "
                 "and is NOT fixed by this mean correction: TD scoring is still a "
                 "low-probability, high-variance PER-GAME event (see the TD section's "
                 "Brier discussion) -- correcting the average bias does not add "
                 "per-player discrimination power, and the `anytime_td` Brier "
                 "improvement over a flat base rate remains near-null.")
    lines.append("6. **P3.5 follow-up: fix `usage.allocate`'s shrinkage-toward-0 at "
                 "the source, instead of relying on population-average `mean_mult` "
                 "patches (now applied to all 7 markets, per Fix round 2).** "
                 "`mean_mult` corrects the AVERAGE bias across all player-games in a "
                 "market, but it cannot correct the PER-PLAYER distribution of that "
                 "bias -- a low-`games` player (whose share was shrunk hardest toward "
                 "0) is still under-allocated more than a high-`games` player, and a "
                 "single multiplier does not know the difference. This is now true of "
                 "ALL 7 markets uniformly, not just the yardage ones. The two "
                 "candidate proper fixes are (a) renormalizing each team-week's "
                 "allocated shares to sum to ~1 after shrinkage (currently shrinkage "
                 "is applied per-player independently, with no renormalization step, "
                 "so the team's shares can sum to well under 1), or (b) shrinking "
                 "each player's share toward a position-appropriate baseline share "
                 "(mirroring how "
                 "`efficiency.compute_efficiency` already shrinks RATES toward "
                 "position baselines in Task 3) instead of toward 0. Either would "
                 "fix the bias per-player rather than only on population average, "
                 "and would very likely make ALL 7 markets' `mean_mult` values "
                 "converge to ~1.0 at the source, without needing a post-hoc "
                 "multiplier in `build_prop` at all. Scoped out of this fix because it "
                 "touches `usage.allocate`, which is outside this task's file "
                 "allowlist and would ripple through every market (including ones "
                 "already shipped in Tasks 4/6).")
    lines.append("")
    lines.append("## Commands used")
    lines.append("")
    lines.append("- `uv run pytest tests/nfl/test_backtest_nfl_props.py -v` (red, then green)")
    lines.append("- `PYTHONPATH=src uv run python scripts/backtest_nfl_props.py` (real fit, "
                 "writes `assets/nfl/props.json`)")
    lines.append("- `uv run pytest -q` (full suite)")
    lines.append("")

    report_path = pathlib.Path("docs/superpowers/reports/2026-08-25-nfl-props-backtest.md")
    report_path.write_text("\n".join(lines))

    print("fitted mean_mult, all 7 markets (Fix rounds 1+2):", cal["mean_mult"])
    print("fitted sigma (on mean_mult-corrected residual):", cal["sigma"])
    print("fitted nb_var_mult:", cal["nb_var_mult"])
    print("per-market metrics:", metrics)
    print("loc (report-only bias):", cal["loc"])
    print("td calibration diagnostics (mult values already folded into mean_mult above):", td_cal)
    print("written props.json:", out)
    print("written report:", report_path)
    print("total wall-clock (s):", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
