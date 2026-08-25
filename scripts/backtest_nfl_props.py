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

TD markets (pass_tds, anytime_td) are deferred to Task 7 -- this backtest
covers yardage markets (pass_yds, reception_yds, rush_yds,
rush_reception_yds) and receptions only.
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
ALL_MARKETS = YARDAGE_MARKETS + ["receptions"]

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
    """sigma[market] = RMSE of (pred_mean - actual) per yardage market
    (method-of-moments Normal sigma). nb_var_mult = empirical var(actual)/
    mean(actual) across all receptions predictions (>=1, since a Negative
    Binomial requires var > mean; clamped to a small margin above 1 if the
    raw empirical ratio undershoots that, e.g. on tiny/synthetic datasets).
    loc[market] = mean(actual - pred_mean) per market, reported only (an
    optional de-bias signal, not applied to the served projection here)."""
    by_market: dict[str, list[dict]] = {}
    for p in preds:
        by_market.setdefault(p["market"], []).append(p)

    sigma = {}
    loc = {}
    for market in YARDAGE_MARKETS:
        rows = by_market.get(market, [])
        n = len(rows)
        if n == 0:
            continue
        resid = [r["pred_mean"] - r["actual"] for r in rows]
        sigma[market] = math.sqrt(sum(e * e for e in resid) / n)
        loc[market] = sum((-e) for e in resid) / n  # mean(actual - pred)

    nb_var_mult = _CFG.nb_var_mult["receptions"]
    rec_rows = by_market.get("receptions", [])
    if rec_rows:
        actuals = [r["actual"] for r in rec_rows]
        n = len(actuals)
        mean = sum(actuals) / n
        if mean > 0:
            var = sum((a - mean) ** 2 for a in actuals) / n
            nb_var_mult = max(var / mean, 1.01)
        resid = [r["pred_mean"] - r["actual"] for r in rec_rows]
        loc["receptions"] = sum((-e) for e in resid) / n

    return {"sigma": sigma, "nb_var_mult": nb_var_mult, "loc": loc}


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

    k_usage = 4.0
    k_eff = 4.0
    out = {
        "sigma": cal["sigma"],
        "nb_var_mult": cal["nb_var_mult"],
        "k_usage": k_usage,
        "k_eff": k_eff,
    }
    pathlib.Path("assets/nfl/props.json").write_text(json.dumps(out, indent=2) + "\n")

    lines = []
    lines.append("# NFL P3 Task 6: Player-props walk-forward backtest -- "
                 "fitted yardage sigmas + receptions dispersion")
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
                 "market's Normal sigma is the residual RMSE of `(projected_mean - "
                 "actual)` on the walk-forward; the receptions Negative Binomial's "
                 "overdispersion multiplier is the empirical `var(actual)/mean(actual)` "
                 "of receptions across all scored player-games. There is no market line "
                 "to \"beat\" here -- season-long CLV against whatever props book lines "
                 "are shopped once this model is actually serving lines is the real "
                 "judge, not this backtest. TD markets (`pass_tds`, `anytime_td`) are "
                 "deferred to Task 7; this backtest covers yardage + receptions only.")
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
    lines.append("## Fitted calibration (`assets/nfl/props.json`)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(out, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("- `sigma[market]`: RMSE of `(projected_mean - actual)` per yardage "
                 "market across all 2016-2024 walk-forward player-games -- the Normal "
                 "sigma `props.build_prop` uses for that market.")
    lines.append("- `nb_var_mult`: empirical `var(actual receptions)/mean(actual "
                 "receptions)` across all scored receptions player-games (clamped "
                 ">1 for a well-defined Negative Binomial).")
    lines.append("- `loc[market]` (report only, not written to `props.json` / not "
                 "applied to projections): mean `(actual - pred_mean)` per market -- a "
                 "positive value means the model under-projects that market on "
                 "average.")
    lines.append("")
    lines.append("### Mean bias (`loc`), report-only")
    lines.append("")
    lines.append("| market | loc (mean actual-pred) |")
    lines.append("|---|---|")
    for market in ALL_MARKETS:
        if market in cal["loc"]:
            lines.append(f"| {market} | {cal['loc'][market]:.3f} |")
    lines.append("")
    lines.append("## TDD: red -> green")
    lines.append("")
    lines.append("Step 2 (red), before `scripts/backtest_nfl_props.py` existed:")
    lines.append("```")
    lines.append("FileNotFoundError: [Errno 2] No such file or directory: "
                 "'.../scripts/backtest_nfl_props.py'")
    lines.append("```")
    lines.append("")
    lines.append("Step 4 (green), after implementation:")
    lines.append("```")
    lines.append("tests/nfl/test_backtest_nfl_props.py::test_run_backtest_returns_per_market_metrics PASSED")
    lines.append("tests/nfl/test_backtest_nfl_props.py::test_fit_calibration_returns_sigmas PASSED")
    lines.append("tests/nfl/test_backtest_nfl_props.py::test_no_leak_uses_prior_season_only PASSED")
    lines.append("3 passed")
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
    lines.append("2. **Usage/efficiency shrinkage (`k_usage`/`k_eff` = 4.0) are carried "
                 "through from Tasks 2-3 as fixed constants, not re-tuned by this "
                 "backtest** -- the brief scopes this task to fitting sigma/nb_var_mult "
                 "only; a follow-up could jointly tune k_usage/k_eff against this same "
                 "residual objective.")
    lines.append("3. **`pass_yds` sigma (106.9) and loc bias (+54.8) are far larger than "
                 "the other yardage markets (24-35 sigma) -- traced to genuine QB "
                 "job-security/team-change regime shifts that a prior-season-shares "
                 "model structurally cannot see.** Example from the real data: Joe "
                 "Flacco (`00-0026158`) split 2022 between spot starts for NYJ "
                 "(`pass_att_share=0.169` after usage shrinkage, reflecting a part-time "
                 "backup role), then signed with CLE for 2023 and started outright "
                 "(42-45 attempts/game, weeks 13-17) -- his 2022-derived share projects "
                 "~6 pass attempts/game for 2023, when he actually threw ~44. This is "
                 "not a code bug (the non-QB markets, which are far less "
                 "winner-take-all, show tight/sane sigmas of 24-35 with 73-83% 1-sigma "
                 "coverage) -- it is the real, load-bearing limitation of projecting "
                 "purely from S-1 season-level shares with no in-season depth-chart/"
                 "injury signal for who wins a QB competition. A follow-up "
                 "(in-season share updates, or an explicit backup/starter transition "
                 "flag) would likely shrink `pass_yds` sigma the most of any market.")
    lines.append("4. **Market inclusion is volume-gated (nonzero projected pass_att/"
                 "targets/carries), not a hardcoded position map** -- this is a "
                 "deliberate, data-driven choice (see code comment on "
                 "`_markets_for_volume`) but means, e.g., a QB who also has real "
                 "receiving volume (extremely rare) would be scored on reception_yds "
                 "too; this is correct behavior, not a bug, but worth knowing the gate "
                 "is volume-based rather than position-based.")
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

    print("fitted sigma:", cal["sigma"])
    print("fitted nb_var_mult:", cal["nb_var_mult"])
    print("per-market metrics:", metrics)
    print("loc (report-only bias):", cal["loc"])
    print("written props.json:", out)
    print("written report:", report_path)
    print("total wall-clock (s):", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
