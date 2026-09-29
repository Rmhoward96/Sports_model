"""CFB profit-model gate: tune on 2019-22, verdict on 2023-25.

Pipeline (spec docs/superpowers/specs/2026-09-29-cfb-profit-model-design.md
sections 3-4; plan Task 6; controller rulings D1 and M4):

1. Load schedules x lines (`backtest_cfb_gameline.load_merged_schedule`), run
   the shared ratings walk-forward (`walkforward.raw_model_predictions`) ONCE
   over 2015-2025, and build the (game, market, price_point) feature table
   (`profit_features.build_rows`).
2. Price-point coverage per season (spec section 1): among PRICED games (any
   closing/opening line or both moneylines), the share carrying that market's
   price at that price point (moneyline: BOTH moneylines). A season counts
   for a market at a price point only with coverage >= 70 %.
3. Per market (each market is gated on its own):
   a. Hyperparameters (M4): a 16-combo HGB grid, each combo scored by the
      walk-forward OOF log-loss of the calibrated p over the tuning seasons.
      The frame handed to the model is cut to seasons <= 2022 first, so no
      verdict row can reach it. Ties -> the more regularized combo.
   b. Full walk-forward OOF 2017-2025 with the chosen hyperparameters
      (training from 2015). Seasons with no earlier labelled row of the
      market are skipped (moneylines exist only from 2021).
   c. Kelly policy (spec section 3): kelly_frac x min_edge chosen by total log
      bankroll growth summed over the tuning price-point streams (D1: closes
      2019-22 + openers where they exist, 2021-22; moneyline closes 2021-22),
      each stream a continuous bankroll over its counted tuning seasons. Ties
      -> smaller kelly_frac, then larger min_edge. `tune_policy` refuses any
      verdict-season row. The ratings-model baseline is tuned the same way.
   d. Verdict 2023-25 at every price point: per season and combined (counted
      seasons only), Kelly bankroll (`kelly.simulate`), flat $10
      (`kelly.flat_pnl`), stake-unit ROI + season-week cluster bootstrap CI
      (`kelly.roi_ci`), ECE, bets placed; the same for the baseline.
   e. `ship_decision` on the primary price point (the opener for spread and
      total; the close for moneyline, the only historical moneyline price).
4. The CLI writes assets/cfb/profit_gate.json and
   docs/superpowers/reports/<date>-cfb-profit-gate.md; `run_gate` itself
   writes nothing.

Betting details: one side per (game, market, price_point) row; spread/total
priced at -110 both sides (plan ruling P5), moneyline at the row's
ml_home/ml_away. Slate day = the kickoff's US-Eastern calendar date. Pushes
(and tied moneylines) are not in the feature table (`build_rows` drops them),
so they are never bet: a push would return the stake, so the only effect is
that it does not take up room under the 15 % day cap.

ROI units (both reported, labelled):
- "Kelly ROI ($)": `simulate`'s profit / dollars staked with a compounding
  bankroll (later bets are bigger dollars when the bankroll has grown);
- "stake-unit ROI": sum(stake_frac x unit P&L) / sum(stake_frac), the unit
  `roi_ci` bootstraps; the ship rule's CI clause uses this one.

Baseline (1): the ratings model's own P(home covers) / P(over) / P(home wins)
from a Normal around its raw `model_margin` / `model_total` (no market shrink,
no bias correction) with gameline.json's sigma_margin / sigma_total, with the
integer outcome's push mass excluded (continuity correction), clipped to
[0.01, 0.99] like the model's probabilities.

Usage:
  .venv/bin/python scripts/gate_cfb_profit.py                  # the real gate
  .venv/bin/python scripts/gate_cfb_profit.py --tuning-only --markets spread
      # data cut to <= 2022 before anything runs; prints; writes nothing
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from sportsmodel.cfb import kelly  # noqa: E402
from sportsmodel.cfb.profit_features import build_rows  # noqa: E402
from sportsmodel.cfb.profit_model import P_MAX, P_MIN, ece, walk_forward_oof  # noqa: E402

# ------------------------------------------------------------------ constants
MIN_TRAIN_SEASON = 2015
DATA_SEASONS = (2015, 2025)                 # inclusive span of the walk-forward
OOF_SEASONS = tuple(range(2017, 2026))
TUNING_SEASONS = (2019, 2020, 2021, 2022)
VERDICT_SEASONS = (2023, 2024, 2025)

MARKETS = ("spread", "total", "moneyline")
PRICE_POINTS = {"spread": ("open", "close"), "total": ("open", "close"),
                "moneyline": ("close",)}
# ship rule "at the opener"; CFBD has no moneyline opener, so the moneyline is
# judged at its only historical price (the close) -- flagged in the report
PRIMARY_PP = {"spread": "open", "total": "open", "moneyline": "close"}
SIDES = {"spread": ("home", "away"), "total": ("over", "under"),
         "moneyline": ("home", "away")}

KELLY_FRACS = (1 / 4, 1 / 3, 1 / 2)
MIN_EDGES = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
HP_KEYS = ("learning_rate", "max_iter", "min_samples_leaf", "l2_regularization")
HP_GRID = [dict(zip(HP_KEYS, v))
           for v in itertools.product((0.03, 0.05), (100, 300), (100, 400), (1.0, 10.0))]

COVERAGE_MIN = 0.70
ECE_MAX = 0.03
SPREAD_TOTAL_AMERICAN = -110.0
DEC_SPREAD_TOTAL = kelly.american_to_decimal(SPREAD_TOTAL_AMERICAN)
START_BANKROLL = 100.0
_TIE = 12                                   # decimals for tie comparisons

_PRICE_FIELDS = {("spread", "open"): ["spread_open"],
                 ("spread", "close"): ["market_spread"],
                 ("total", "open"): ["total_open"],
                 ("total", "close"): ["market_total"],
                 ("moneyline", "close"): ["ml_home", "ml_away"]}

OofFn = Callable[..., pd.DataFrame]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------- coverage
def coverage_table(games: pd.DataFrame) -> dict:
    """{market: {price_point: {season: coverage}}} over PRICED games (any line
    or both moneylines). Moneyline coverage needs both moneylines."""
    all_cols = sorted({c for cols in _PRICE_FIELDS.values() for c in cols})
    g = games.reindex(columns=["season", *all_cols])
    g[all_cols] = g[all_cols].apply(pd.to_numeric, errors="coerce")
    has = {k: g[cols].notna().all(axis=1) for k, cols in _PRICE_FIELDS.items()}
    priced = np.logical_or.reduce([h.to_numpy() for h in has.values()])
    out = {m: {pp: {} for pp in PRICE_POINTS[m]} for m in MARKETS}
    for season in sorted(g.loc[priced, "season"].dropna().unique()):
        mask = priced & (g["season"] == season).to_numpy()
        n = int(mask.sum())
        for (m, pp), h in has.items():
            out[m][pp][int(season)] = float(h.to_numpy()[mask].sum() / n) if n else 0.0
    return out


def _counted(cov: dict, market: str, pp: str, season: int) -> bool:
    return cov.get(market, {}).get(pp, {}).get(int(season), 0.0) >= COVERAGE_MIN


# ------------------------------------------------------------------- baseline
def baseline_probs(rows: pd.DataFrame, sigmas: dict) -> np.ndarray:
    """Ratings-model P(first side wins | no push) per row: Normal around
    f_model_margin (spread, moneyline at line 0) or f_model_total (total),
    integer outcomes via continuity correction, push mass excluded."""
    market = rows["market"].to_numpy()
    is_total = market == "total"
    mean = np.where(is_total, rows["f_model_total"].to_numpy(dtype=float),
                    rows["f_model_margin"].to_numpy(dtype=float))
    sd = np.where(is_total, float(sigmas["sigma_total"]), float(sigmas["sigma_margin"]))
    line = np.where(market == "moneyline", 0.0, rows["line"].to_numpy(dtype=float))
    upper = np.floor(line) + 0.5            # outcome > line  <=> integer >= floor(line)+1
    lower = np.ceil(line) - 0.5             # outcome < line  <=> integer <= ceil(line)-1
    p_first = norm.sf(upper, mean, sd)
    p_second = norm.cdf(lower, mean, sd)
    denom = p_first + p_second
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(denom > 0, p_first / denom, (mean > line).astype(float))
    return np.clip(p, P_MIN, P_MAX)


# ----------------------------------------------------------------------- bets
_BET_COLS = ["season", "week", "game_pk", "day", "side", "p_side", "edge", "dec",
             "stake_frac", "result"]


def _chronological(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows with their ET slate `day`, sorted chronologically. A row without a
    kickoff takes the most common slate day of its (season, week) among the
    rows that have one (the week's Saturday, in practice); with none it is
    dropped. Returns (rows, n_dropped)."""
    kick = pd.to_datetime(rows["start_date"], utc=True, errors="coerce")
    day = kick.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    out = rows.assign(_kick=kick, day=day)
    missing = out["day"].isna()
    if missing.any():
        known = out[~missing]
        week_day = (known.groupby(["season", "week"])["day"]
                    .agg(lambda d: d.value_counts().sort_index().idxmax()).to_dict())
        fill = [week_day.get((s, w)) for s, w in zip(out.loc[missing, "season"],
                                                     out.loc[missing, "week"])]
        out.loc[missing, "day"] = fill
    n_before = len(out)
    out = out[out["day"].notna()]
    n_dropped = n_before - len(out)
    if n_dropped:
        _log(f"  warning: {n_dropped} row(s) with no kickoff and no dated game in their "
             "week dropped from betting")
    out = out.sort_values(["season", "day", "_kick", "game_pk"], na_position="last",
                          kind="mergesort")
    return out, n_dropped


def _first_side_result(rows: pd.DataFrame) -> pd.Series:
    """'win'/'loss'/'push'/'void' of the first side; from `result` when the
    table carries it (build_rows), else from y (NaN -> void)."""
    if "result" in rows.columns:
        return rows["result"].fillna("void")
    return rows["y"].map({1.0: "win", 0.0: "loss"}).fillna("void")


def make_bets(rows: pd.DataFrame, market: str, p_col: str, kelly_frac: float,
              min_edge: float) -> pd.DataFrame:
    """Kelly bets for one price-point stream, in chronological order.

    `rows[p_col]` is P(first side wins) (home covers / over / home wins).
    At most one side per row (= per game, market, price_point); stakes capped
    at 3 % per bet and 15 % per ET slate day (`kelly.apply_day_cap`). Pushed
    and void rows are bet like any other (the bet is placed before the result
    is known) and settle as a push: stake returned, but it is staked, counted
    and takes up day-cap room. `bets.attrs["n_dropped_no_day"]` counts rows
    left out for want of a slate day."""
    r = rows[rows[p_col].notna()].assign(_res=_first_side_result(rows))
    n_dropped = 0
    if not r.empty:
        r, n_dropped = _chronological(r)
    if r.empty:
        bets = pd.DataFrame(columns=_BET_COLS)
        bets.attrs["n_dropped_no_day"] = n_dropped
        return bets
    sides = SIDES[market]
    recs = []
    cols = ["season", "week", "game_pk", "day", p_col, "_res", "ml_home", "ml_away"]
    for season, week, pk, day, p, res, ml_h, ml_a in r[cols].itertuples(index=False):
        if market == "moneyline":
            try:
                d_first = kelly.american_to_decimal(ml_h)
                d_second = kelly.american_to_decimal(ml_a)
            except (ValueError, TypeError):
                continue
        else:
            d_first = d_second = DEC_SPREAD_TOTAL
        pick = kelly.choose_side(float(p), d_first, d_second, min_edge, sides=sides)
        if pick is None:
            continue
        side, edge = pick
        first = side == sides[0]
        p_side = float(p) if first else 1.0 - float(p)
        dec = d_first if first else d_second
        frac = kelly.stake_fraction(p_side, dec, kelly_frac)
        if not frac > 0:
            continue
        if res in ("push", "void"):
            result = "push"
        else:
            result = "win" if (res == "win") == first else "loss"
        recs.append({"season": int(season), "week": int(week), "game_pk": pk, "day": day,
                     "side": side, "p_side": p_side, "edge": edge, "dec": dec,
                     "stake_frac": frac, "result": result})
    bets = pd.DataFrame(recs, columns=_BET_COLS)
    bets.attrs["n_dropped_no_day"] = n_dropped
    if bets.empty:
        return bets
    capped = bets["stake_frac"].to_numpy(dtype=float).copy()
    for _, idx in bets.groupby("day", sort=False).indices.items():
        capped[idx] = kelly.apply_day_cap(list(capped[idx]))
    bets["stake_frac"] = capped
    out = bets.reset_index(drop=True)
    out.attrs["n_dropped_no_day"] = n_dropped
    return out


def _stake_unit_roi(bets: pd.DataFrame) -> float:
    if bets.empty:
        return float("nan")
    res = bets["result"].to_numpy()
    unit = np.where(res == "win", bets["dec"].astype(float) - 1.0,
                    np.where(res == "push", 0.0, -1.0))
    stake = bets["stake_frac"].to_numpy(dtype=float)
    return float((stake * unit).sum() / stake.sum()) if stake.sum() > 0 else float("nan")


def _log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        return float("nan")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(rows: pd.DataFrame, market: str, p_col: str, policy: dict) -> dict:
    """All verdict metrics for one stream of rows under a frozen policy. Bets
    include pushed/void rows; ECE, log-loss and n_rows use labelled rows only."""
    bets = make_bets(rows, market, p_col, policy["kelly_frac"], policy["min_edge"])
    sim = kelly.simulate(bets, START_BANKROLL)
    lo, hi = kelly.roi_ci(bets)
    lab = rows[rows["y"].notna()]
    return {
        "n_rows": int(len(lab)),
        "n_bets": int(len(bets)),
        "n_push": int((bets["result"] == "push").sum()),
        "n_dropped_no_day": int(bets.attrs.get("n_dropped_no_day", 0)),
        "win_rate": float((bets["result"] == "win").mean()) if len(bets) else float("nan"),
        "sim": sim,
        "flat": kelly.flat_pnl(bets),
        "roi_units": _stake_unit_roi(bets),
        "roi_units_ci": [lo, hi],
        "ece": ece(lab[p_col], lab["y"]) if len(lab) else float("nan"),
        "log_loss": _log_loss(lab[p_col], lab["y"]),
    }


def reliability(p, y, bins: int = 10) -> list[dict]:
    """Equal-width reliability bins (the same bins as profit_model.ece)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.clip(np.floor(p * bins).astype(int), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                    "mean_p": float(p[m].mean()) if m.any() else float("nan"),
                    "mean_y": float(y[m].mean()) if m.any() else float("nan")})
    return out


# --------------------------------------------------------------------- tuning
def _predictable_seasons(d: pd.DataFrame, candidates: Iterable[int]) -> list[int]:
    """Candidate seasons with rows AND at least one earlier labelled row
    (training from MIN_TRAIN_SEASON) -- what walk_forward_oof can predict."""
    lab = d[d["y"].notna()]
    out = []
    for s in sorted({int(c) for c in candidates}):
        has_rows = bool((d["season"] == s).any())
        has_train = bool(((lab["season"] >= MIN_TRAIN_SEASON) & (lab["season"] < s)).any())
        if has_rows and has_train:
            out.append(s)
    return out


def tune_hyperparameters(rows: pd.DataFrame, market: str, oof_fn: OofFn,
                         hp_grid: list[dict] = HP_GRID) -> tuple[dict, list[dict], list[int]]:
    """M4: pick the HGB hyperparameters by OOF log-loss of the calibrated p on
    the tuning seasons. Only rows of seasons <= the last tuning season are
    handed to the model. Ties -> larger l2, larger min_samples_leaf, fewer
    iterations, smaller learning rate (the more regularized)."""
    d = rows[(rows["market"] == market) & (rows["season"] <= max(TUNING_SEASONS))]
    d = d[d["y"].notna()]
    seasons = _predictable_seasons(d, TUNING_SEASONS)
    if not seasons:
        raise ValueError(f"{market}: no predictable tuning season")
    grid = []
    for i, hp in enumerate(hp_grid):
        t0 = time.time()
        oof = oof_fn(d, market, seasons, min_train_season=MIN_TRAIN_SEASON, **hp)
        if (oof["season"] > max(TUNING_SEASONS)).any():
            raise AssertionError("hyperparameter tuning produced a verdict-season row")
        lab = oof[oof["y"].notna() & oof["season"].isin(seasons)]
        grid.append({**hp, "log_loss": _log_loss(lab["p"], lab["y"]), "n": int(len(lab))})
        _log(f"  [{market}] hp {i + 1}/{len(hp_grid)} {hp} "
             f"log_loss={grid[-1]['log_loss']:.5f} ({time.time() - t0:.1f}s)")
    best = min(grid, key=lambda g: (round(g["log_loss"], _TIE), -g["l2_regularization"],
                                    -g["min_samples_leaf"], g["max_iter"],
                                    g["learning_rate"]))
    return {k: best[k] for k in HP_KEYS}, grid, seasons


def tune_policy(streams: list[pd.DataFrame], market: str, p_col: str,
                kelly_fracs=KELLY_FRACS, min_edges=MIN_EDGES) -> tuple[dict, list[dict]]:
    """Choose (kelly_frac, min_edge) by total log growth summed over the
    streams (each a separate continuous bankroll). Ties -> smaller kelly_frac,
    then larger min_edge. Verdict-season rows are refused."""
    for s in streams:
        if s["season"].isin(VERDICT_SEASONS).any():
            raise ValueError("policy tuning was handed a verdict-season row")
    grid = []
    for kf in kelly_fracs:
        for me in min_edges:
            lg, n = 0.0, 0
            for s in streams:
                sim = kelly.simulate(make_bets(s, market, p_col, kf, me), START_BANKROLL)
                lg += sim["log_growth_total"]
                n += sim["n_bets"]
            grid.append({"kelly_frac": kf, "min_edge": me, "log_growth_total": lg,
                         "n_bets": n, "log_growth_per_bet": lg / n if n else 0.0})
    best = max(grid, key=lambda g: (round(g["log_growth_total"], _TIE), -g["kelly_frac"],
                                    g["min_edge"]))
    return {"kelly_frac": best["kelly_frac"], "min_edge": best["min_edge"]}, grid


# ---------------------------------------------------------------------- gate
def ship_decision(inp: dict) -> dict:
    """Global-Constraints ship rule for ONE market (primary price point):
    1. log growth > 0 on the combined verdict seasons AND in >= 2 of the 3
       verdict seasons (an uncounted season -- None -- is not positive);
    2. stake-unit ROI 95 % CI lower bound > 0 (combined);
    3. ECE < 0.03 (combined);
    4. combined log growth > the ratings-model baseline's."""
    def ok(x) -> bool:
        return x is not None and not (isinstance(x, float) and math.isnan(x))

    def gt(x, t) -> bool:
        return ok(x) and float(x) > t

    lg = inp.get("log_growth_combined")
    by = inp.get("log_growth_by_season") or {}
    season_lg = {s: by.get(s, by.get(str(s))) for s in VERDICT_SEASONS}
    n_pos = sum(gt(v, 0.0) for v in season_lg.values())
    roi_lo = inp.get("roi_ci_lower")
    e = inp.get("ece")
    base = inp.get("baseline_log_growth_combined")
    clauses = {
        "log_growth_combined_positive": gt(lg, 0.0),
        "log_growth_2_of_3_seasons": n_pos >= 2,
        "roi_ci_lower_positive": gt(roi_lo, 0.0),
        "ece_below_0.03": ok(e) and float(e) < ECE_MAX,
        "beats_baseline": ok(lg) and ok(base) and float(lg) > float(base),
    }
    why = {
        "log_growth_combined_positive": f"combined log growth {lg!r} is not > 0",
        "log_growth_2_of_3_seasons": (f"log growth > 0 in only {n_pos} of 3 verdict seasons "
                                      f"({season_lg})"),
        "roi_ci_lower_positive": f"ROI 95% CI lower bound {roi_lo!r} is not > 0",
        "ece_below_0.03": f"ECE {e!r} is not < {ECE_MAX}",
        "beats_baseline": f"log growth {lg!r} does not beat the baseline's {base!r}",
    }
    reasons = [why[k] for k, v in clauses.items() if not v]
    return {"pass": all(clauses.values()), "clauses": clauses, "reasons": reasons}


def run_market(rows: pd.DataFrame, cov: dict, market: str, sigmas: dict, *,
               oof_fn: OofFn = walk_forward_oof, hp_grid: list[dict] = HP_GRID,
               tuning_only: bool = False) -> dict:
    t0 = time.time()
    # every row of the market: labelled rows train/calibrate/score; pushed and
    # void rows (y NaN) are only predicted and bet
    d = rows[rows["market"] == market].copy()
    d["p_base"] = baseline_probs(d, sigmas)

    _log(f"[{market}] hyperparameter grid ({len(hp_grid)} combos)")
    hp, hp_grid_res, hp_seasons = tune_hyperparameters(d, market, oof_fn, hp_grid)

    top = int(d["season"].max())
    oof_seasons = _predictable_seasons(d, [s for s in OOF_SEASONS if s <= top])
    _log(f"[{market}] full OOF {oof_seasons} with {hp}")
    oof = oof_fn(d, market, oof_seasons, min_train_season=MIN_TRAIN_SEASON, **hp)

    streams = {}
    for pp in PRICE_POINTS[market]:
        ss = [s for s in TUNING_SEASONS if _counted(cov, market, pp, s)
              and bool(((oof["season"] == s) & (oof["price_point"] == pp)).any())]
        if ss:
            streams[pp] = ss
    frames = [oof[(oof["price_point"] == pp) & oof["season"].isin(ss)]
              for pp, ss in streams.items()]
    _log(f"[{market}] policy grid on streams {streams}")
    policy, policy_grid = tune_policy(frames, market, "p")
    base_policy, base_grid = tune_policy(frames, market, "p_base")

    res = {"market": market, "primary_price_point": PRIMARY_PP[market],
           "hyperparameters": hp, "hp_grid": hp_grid_res, "hp_tuning_seasons": hp_seasons,
           "oof_seasons": oof_seasons, "tuning_streams": streams,
           "policy": policy, "policy_grid": policy_grid,
           "baseline_policy": base_policy, "baseline_policy_grid": base_grid,
           "coverage": cov.get(market, {})}
    if tuning_only:
        res["runtime_s"] = time.time() - t0
        return res

    verdict = {}
    for pp in PRICE_POINTS[market]:
        at_pp = oof[oof["price_point"] == pp]
        seasons_res, counted = {}, []
        for s in VERDICT_SEASONS:
            fr = at_pp[at_pp["season"] == s]
            c = _counted(cov, market, pp, s) and len(fr) > 0
            seasons_res[s] = {"coverage": cov.get(market, {}).get(pp, {}).get(s, 0.0),
                              "counted": c,
                              "model": evaluate(fr, market, "p", policy),
                              "baseline": evaluate(fr, market, "p_base", base_policy)}
            if c:
                counted.append(s)
        comb = at_pp[at_pp["season"].isin(counted)]
        verdict[pp] = {"seasons": seasons_res,
                       "combined": {"seasons": counted,
                                    "model": evaluate(comb, market, "p", policy),
                                    "baseline": evaluate(comb, market, "p_base", base_policy),
                                    "calibration": reliability(
                                        comb.loc[comb["y"].notna(), "p"],
                                        comb.loc[comb["y"].notna(), "y"])}}
    res["verdict"] = verdict

    pv = verdict[PRIMARY_PP[market]]
    no_seasons = not pv["combined"]["seasons"]
    nan = float("nan")
    res["ship_inputs"] = {
        "price_point": PRIMARY_PP[market],
        "log_growth_combined": nan if no_seasons else pv["combined"]["model"]["sim"]["log_growth_total"],
        "log_growth_by_season": {s: (pv["seasons"][s]["model"]["sim"]["log_growth_total"]
                                     if pv["seasons"][s]["counted"] else None)
                                 for s in VERDICT_SEASONS},
        "roi_ci_lower": pv["combined"]["model"]["roi_units_ci"][0],
        "ece": pv["combined"]["model"]["ece"],
        "baseline_log_growth_combined": (nan if no_seasons else
                                         pv["combined"]["baseline"]["sim"]["log_growth_total"]),
    }
    res["ship"] = ship_decision(res["ship_inputs"])
    res["runtime_s"] = time.time() - t0
    return res


def run_gate(rows: pd.DataFrame, games: pd.DataFrame, sigmas: dict, *,
             oof_fn: OofFn | None = None, markets: Iterable[str] = MARKETS,
             hp_grid: list[dict] = HP_GRID, tuning_only: bool = False) -> dict:
    """The gate over a prepared feature table (`build_rows` output) and the
    walk-forward game rows it came from (for coverage). Writes nothing.
    `oof_fn` defaults to profit_model.walk_forward_oof."""
    oof_fn = oof_fn or walk_forward_oof
    cov = coverage_table(games)
    out = {"generated": date.today().isoformat(),
           "tuning_seasons": list(TUNING_SEASONS), "verdict_seasons": list(VERDICT_SEASONS),
           "coverage_min": COVERAGE_MIN, "ece_max": ECE_MAX, "sigmas": dict(sigmas),
           "spread_total_price": SPREAD_TOTAL_AMERICAN, "tuning_only": tuning_only,
           "markets": {}}
    for m in markets:
        out["markets"][m] = run_market(rows, cov, m, sigmas, oof_fn=oof_fn,
                                       hp_grid=hp_grid, tuning_only=tuning_only)
    if not tuning_only:
        out["passing_markets"] = [m for m, r in out["markets"].items() if r["ship"]["pass"]]
    return out


# ------------------------------------------------------------------- output
def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isinf(x):
            return "-inf" if x < 0 else "inf"      # a wiped-out bankroll stays visible
        return None if math.isnan(x) else x
    return x


def to_json(results: dict) -> str:
    return json.dumps(_clean(results), indent=2, allow_nan=False)


def _f(x, nd: int = 3, pct: bool = False) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "—"
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{100 * v:.{max(nd - 2, 1)}f}%" if pct else f"{v:.{nd}f}"


def _policy_str(p: dict) -> str:
    return f"kelly_frac {p['kelly_frac']:.3f}, min_edge {p['min_edge']:.2f}"


def _verdict_table(v: dict) -> list[str]:
    lines = ["| season | coverage | counted | rows | bets | log growth (total) | "
             "log growth / bet | pushes | end bankroll | Kelly ROI ($) | "
             "stake-unit ROI [95% CI] | "
             "flat $10 P&L (ROI) | max DD | ECE | baseline bets | baseline log growth | "
             "baseline end bankroll |",
             "|" + "---|" * 17]
    entries = [(str(s), r) for s, r in v["seasons"].items()]
    entries.append(("combined", {**v["combined"], "coverage": None,
                                 "counted": ",".join(map(str, v["combined"]["seasons"])) or "none"}))
    for label, r in entries:
        m, b = r["model"], r["baseline"]
        counted = r["counted"] if isinstance(r["counted"], str) else ("yes" if r["counted"] else "NO")
        lo, hi = m["roi_units_ci"]
        lines.append(
            f"| {label} | {_f(r['coverage'], 3, pct=True)} | {counted} | {m['n_rows']} | "
            f"{m['n_bets']} | {_f(m['sim']['log_growth_total'], 4)} | "
            f"{_f(m['sim']['log_growth_per_bet'], 5)} | {m['n_push']} | "
            f"{_f(m['sim']['end_bankroll'], 2)} | "
            f"{_f(m['sim']['roi'], 3, pct=True)} | {_f(m['roi_units'], 3, pct=True)} "
            f"[{_f(lo, 3, pct=True)}, {_f(hi, 3, pct=True)}] | "
            f"{_f(m['flat']['profit'], 2)} ({_f(m['flat']['roi'], 3, pct=True)}) | "
            f"{_f(m['sim']['max_drawdown'], 3, pct=True)} | {_f(m['ece'], 4)} | "
            f"{b['n_bets']} | {_f(b['sim']['log_growth_total'], 4)} | "
            f"{_f(b['sim']['end_bankroll'], 2)} |")
    return lines


def render_report(results: dict) -> str:
    L = [f"# CFB profit-model gate ({results.get('generated', '')})", ""]
    L += ["Spec: `docs/superpowers/specs/2026-09-29-cfb-profit-model-design.md` §3–4. "
          f"Tuning seasons {results['tuning_seasons']} (hyperparameters, Kelly policy, "
          f"baseline policy); verdict seasons {results['verdict_seasons']} — never used "
          "for any choice.",
          "",
          f"- Prices: spread/total at {SPREAD_TOTAL_AMERICAN:+.0f} both sides (ruling P5); "
          "moneyline at the CFBD median closing moneylines. Openers and moneylines exist "
          "only from 2021 (ruling D1): tuning uses closes 2019–22 + openers 2021–22; "
          "moneyline tunes on the seasons it can be predicted in.",
          f"- A season counts for a market at a price point only with ≥ "
          f"{COVERAGE_MIN:.0%} coverage (moneyline: both moneylines).",
          "- Bankroll starts at 100 units; stakes = fractional Kelly, ≤ 3 % per bet, ≤ 15 % "
          "per ET slate day; one side per game, market and price point.",
          "- **Kelly ROI ($)** = `simulate` profit / dollars staked with the compounding "
          "bankroll. **Stake-unit ROI** = Σ stake_frac·P&L / Σ stake_frac (the unit the "
          "season-week cluster bootstrap CI uses; the ship rule's CI clause uses it). "
          "Flat $10 = the same bets at $10 each.",
          "- Pushes (and voids) stay in the bet set — a bet is placed before the result "
          "is known — and settle at 0 with the stake returned; they count as bets and "
          "as staked and take up day-cap room. ECE, log-loss, the calibration deciles and "
          "the table's `rows` use the labelled rows only (pushes excluded).",
          "- The ship rule's ECE is over ALL labelled combined-verdict rows at the primary "
          "price point, bet or not.",
          "- Baseline = the ratings model's own Normal probabilities (sigmas "
          f"{_f(results['sigmas'].get('sigma_margin'), 2)} / "
          f"{_f(results['sigmas'].get('sigma_total'), 2)}) through the same policy, tuned "
          "the same way.",
          "- The ship rule is judged at the **opener** for spread and total; CFBD has no "
          "moneyline opener, so the moneyline is judged at the **close**.",
          ""]
    if results.get("tuning_only"):
        L += ["**TUNING-ONLY RUN — no verdict.**", ""]

    L += ["## Summary", "",
          "| market | judged at | policy | log growth (combined) | baseline log growth | "
          "stake-unit ROI 95% CI | ECE | verdict |", "|---|---|---|---|---|---|---|---|"]
    for m, r in results["markets"].items():
        if "ship" not in r:
            L.append(f"| {m} | {r['primary_price_point']} | {_policy_str(r['policy'])} "
                     "| — | — | — | — | tuning only |")
            continue
        si = r["ship_inputs"]
        comb = r["verdict"][r["primary_price_point"]]["combined"]["model"]
        lo, hi = comb["roi_units_ci"]
        L.append(f"| {m} | {r['primary_price_point']} | {_policy_str(r['policy'])} | "
                 f"{_f(si['log_growth_combined'], 4)} | "
                 f"{_f(si['baseline_log_growth_combined'], 4)} | "
                 f"[{_f(lo, 3, pct=True)}, {_f(hi, 3, pct=True)}] | {_f(si['ece'], 4)} | "
                 f"**{'PASS' if r['ship']['pass'] else 'FAIL'}** |")
    L.append("")

    for m, r in results["markets"].items():
        L += [f"## {m}", ""]
        if "ship" in r:
            L += [f"**{'PASS' if r['ship']['pass'] else 'FAIL'}** "
                  f"(judged at the {r['primary_price_point']})", ""]
            for k, v in r["ship"]["clauses"].items():
                L.append(f"- {'✅' if v else '❌'} {k}")
            for why in r["ship"]["reasons"]:
                L.append(f"  - {why}")
            L.append("")

        L += ["### Coverage (share of priced games)", "",
              "| price point | " + " | ".join(str(s) for s in range(2015, 2026)) + " |",
              "|---|" + "---|" * 11]
        for pp, by in r["coverage"].items():
            L.append(f"| {pp} | " + " | ".join(_f(by.get(s), 3, pct=True)
                                              for s in range(2015, 2026)) + " |")
        L.append("")

        hp = r["hyperparameters"]
        L += ["### Hyperparameters (OOF log-loss, tuning seasons "
              f"{r['hp_tuning_seasons']})", "",
              "Chosen: " + ", ".join(f"{k}={hp[k]}" for k in HP_KEYS), "",
              "| learning_rate | max_iter | min_samples_leaf | l2_regularization | log-loss | rows |",
              "|---|---|---|---|---|---|"]
        for g in sorted(r["hp_grid"], key=lambda g: g["log_loss"]):
            mark = " **←**" if all(g[k] == hp[k] for k in HP_KEYS) else ""
            L.append(f"| {g['learning_rate']} | {g['max_iter']} | {g['min_samples_leaf']} | "
                     f"{g['l2_regularization']} | {_f(g['log_loss'], 5)}{mark} | {g['n']} |")
        L += ["", f"OOF seasons (full walk-forward): {r['oof_seasons']}", ""]

        L += ["### Kelly policy (tuning streams " + ", ".join(
            f"{pp} {ss}" for pp, ss in r["tuning_streams"].items()) + ")", "",
              f"Model: **{_policy_str(r['policy'])}** · baseline: "
              f"**{_policy_str(r['baseline_policy'])}**", "",
              "| kelly_frac | min_edge | model log growth | model bets | baseline log growth | "
              "baseline bets |", "|---|---|---|---|---|---|"]
        for g, b in zip(r["policy_grid"], r["baseline_policy_grid"]):
            L.append(f"| {g['kelly_frac']:.3f} | {g['min_edge']:.2f} | "
                     f"{_f(g['log_growth_total'], 4)} | {g['n_bets']} | "
                     f"{_f(b['log_growth_total'], 4)} | {b['n_bets']} |")
        L.append("")

        if "verdict" in r:
            for pp, v in r["verdict"].items():
                L += [f"### Verdict at the {pp}", ""] + _verdict_table(v) + [""]
            cal = r["verdict"][r["primary_price_point"]]["combined"]["calibration"]
            L += [f"### Calibration deciles (combined verdict, {r['primary_price_point']})", "",
                  "| bin | n | mean p | observed |", "|---|---|---|---|"]
            for c in cal:
                L.append(f"| {c['bin']} | {c['n']} | {_f(c['mean_p'], 3)} | "
                         f"{_f(c['mean_y'], 3)} |")
            L.append("")
    if "passing_markets" in results:
        L += ["## Decision", "",
              "Passing markets: " + (", ".join(results["passing_markets"]) or "none") + ".",
              ""]
    return "\n".join(L)


# ----------------------------------------------------------------------- CLI
def _load_backtest():
    path = ROOT / "scripts" / "backtest_cfb_gameline.py"
    spec = importlib.util.spec_from_file_location("backtest_cfb_gameline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_inputs(max_season: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(feature rows, walk-forward game rows, sigmas) for seasons
    DATA_SEASONS[0]..max_season, from the committed assets."""
    from sportsmodel.cfb.walkforward import raw_model_predictions
    from sportsmodel.nfl.elo import EloConfig
    from sportsmodel.nfl.ratings import BlendConfig

    a = ROOT / "assets" / "cfb"
    merged = _load_backtest().load_merged_schedule(str(a / "schedules.parquet"),
                                                   str(a / "lines.parquet"))
    span = merged[(merged["season"] >= DATA_SEASONS[0]) & (merged["season"] <= max_season)]
    j = json.loads((a / "rating.json").read_text())
    elo_cfg = EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"],
                        base=j.get("base", 1500.0))
    blend_cfg = BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"])
    t0 = time.time()
    raw = raw_model_predictions(span, elo_cfg, blend_cfg)
    _log(f"walk-forward {DATA_SEASONS[0]}-{max_season}: {len(raw)} games "
         f"({time.time() - t0:.1f}s)")
    priors = pd.read_parquet(a / "priors.parquet")
    fbs = set(json.loads((a / "fbs_teams.json").read_text()))
    rows = build_rows(raw, priors, fbs, keep_pushes=True)   # pushes are bet (G3)
    gl = json.loads((a / "gameline.json").read_text())
    sigmas = {"sigma_margin": gl["sigma_margin"], "sigma_total": gl["sigma_total"]}
    return rows, pd.DataFrame(raw), sigmas


def main(argv: list[str] | None = None, *, oof_fn: OofFn | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--markets", nargs="+", choices=MARKETS, default=list(MARKETS))
    ap.add_argument("--tuning-only", action="store_true",
                    help="cut all data to seasons <= 2022 first; tune only; print; write nothing")
    ap.add_argument("--json-out", default=str(ROOT / "assets" / "cfb" / "profit_gate.json"))
    ap.add_argument("--report-out", default=None,
                    help="default docs/superpowers/reports/<today>-cfb-profit-gate.md")
    args = ap.parse_args(argv)

    t0 = time.time()
    max_season = max(TUNING_SEASONS) if args.tuning_only else DATA_SEASONS[1]
    rows, games, sigmas = load_inputs(max_season)
    res = run_gate(rows, games, sigmas, markets=args.markets, tuning_only=args.tuning_only,
                   oof_fn=oof_fn)
    res["runtime_s"] = time.time() - t0
    if args.tuning_only:
        summary = {m: {k: r[k] for k in ("hyperparameters", "hp_tuning_seasons",
                                          "tuning_streams", "policy", "baseline_policy",
                                          "runtime_s")}
                   for m, r in res["markets"].items()}
        print(to_json({"tuning_only": True, "runtime_s": res["runtime_s"], "markets": summary}))
        return 0

    json_out = Path(args.json_out)
    report_out = Path(args.report_out) if args.report_out else (
        ROOT / "docs" / "superpowers" / "reports" / f"{res['generated']}-cfb-profit-gate.md")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(to_json(res) + "\n")
    report_out.write_text(render_report(res))
    _log(f"wrote {json_out} and {report_out} ({res['runtime_s']:.0f}s); "
         f"passing: {res['passing_markets'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
