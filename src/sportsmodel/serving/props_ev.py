"""Prop market/name reconciliation between the NFL sim and the Odds API (C v1).

Two pure, independently-testable pieces glue the sim's per-player prop
distributions (`sportsmodel.sim.nfl.aggregate.nfl_player_prop_dists`) to the
odds side (`SportConfig["nfl"].prop_market_map` in `sportsmodel.sports`, and
the Odds API's `player_name` field) so a board producer can join them:

  MARKET NAMES: the sim's aggregate markets are pass_yds/rush_yds/rec_yds/
  receptions. `SportConfig["nfl"].prop_market_map` (and the Odds API) key
  the same markets as rush_yds/reception_yds/receptions/pass_yds. The ONLY
  naming mismatch is rec_yds (sim) <-> reception_yds (odds); rush_yds and
  receptions already agree. `pass_yds` is intentionally EXCLUDED from C v1
  (see Task 2 brief / Ruling C1 lineage) -- `odds_market_for("pass_yds")`
  returns None so callers skip it rather than silently mis-mapping it.

  PLAYER NAMES: `nfl_player_sim` carries the player's display name (e.g. from
  nflverse), while the Odds API's `player_name` field is independently
  formatted. `normalize_player_name` produces a join key robust to punctuation
  and generational-suffix differences (e.g. "A.J. Brown", "AJ Brown", and
  "A.J. Brown Jr." all normalize to "aj brown"). This is deliberately
  stronger than `sportsmodel.nfl.matcher._norm_name` (strip+lower only),
  which is too weak for player-name joins specifically -- that function is
  left unchanged since it's tuned for its own (team/date) matching use.

This module also HOSTS the projected-usage gate (`PROJECTED_USAGE_GATE`,
`is_propable_projected`), relocated here (Ruling C1) from
`scripts/backtest_sim_nfl.py` so both the backtest ship-gate and the
upcoming prop board producer import the same deployment-population gate
from one place, instead of duplicating/drifting thresholds.
"""
from __future__ import annotations

import re

from ..model.distributions import prob_over_dist
from .board import best_price, ev, novig

# Sim aggregate market -> Odds API / SportConfig["nfl"].prop_market_map key.
# The sim's markets are pass_yds/rush_yds/rec_yds/receptions; the odds side
# uses rush_yds/reception_yds/receptions/pass_yds. rec_yds <-> reception_yds
# is the only name mismatch. pass_yds is intentionally EXCLUDED from C v1 (not
# a key in this dict) -- see module docstring.
SIM_TO_ODDS_MARKET: dict[str, str] = {
    "rush_yds": "rush_yds",
    "rec_yds": "reception_yds",
    "receptions": "receptions",
}


def odds_market_for(sim_market: str) -> str | None:
    """Odds-side market key for a sim aggregate market, or None if `sim_market`
    is unknown or intentionally excluded from C v1 (e.g. pass_yds)."""
    return SIM_TO_ODDS_MARKET.get(sim_market)


# Trailing generational-suffix tokens stripped from a normalized player name
# (matched as whole tokens post-normalization, case already lowered).
_SUFFIX_TOKENS = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})

# Characters dropped outright (not replaced with a space) when normalizing a
# player name -- periods and apostrophes, so "A.J." -> "aj" and "Ja'Marr" ->
# "jamarr" rather than leaving stray gaps.
_DROP_CHARS_RE = re.compile(r"[.’']")

# Any run of whitespace collapses to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_player_name(name: str) -> str:
    """Robust join key for a player's display name, for matching the sim's
    player name (from nfl_player_sim) to the Odds API `player_name` field.

    Lowercases, strips surrounding whitespace, removes periods and
    apostrophes, collapses internal whitespace to single spaces, and strips
    trailing generational-suffix tokens (jr/sr/ii/iii/iv/v). So "A.J. Brown",
    "AJ Brown", and "A.J. Brown Jr." all normalize to "aj brown".

    Deliberately stronger than `sportsmodel.nfl.matcher._norm_name` (which is
    only strip+lower) -- that helper is tuned for team/date event matching,
    not player-name joins, and is left unchanged.
    """
    s = (name or "").strip().lower()
    s = _DROP_CHARS_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    tokens = s.split(" ") if s else []
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens)


# =============================================================================
# Projected-usage gate (relocated from scripts/backtest_sim_nfl.py, Ruling C1)
# =============================================================================

# Per-market sim dist-mean thresholds approximating "the model projects this
# player featured enough to have a line" -- i.e. the PROJECTED (pre-game,
# no-selection-bias) counterpart to backtest_sim_nfl.py's ACTUAL-usage gate
# (`_MARKET_USAGE_GATE`/`is_propable`). This is the deployment population:
# it's what a shippability read should grade against, since books set lines
# (and props get offered) off projected usage, not off what a player ends up
# doing that week. Tunable.
PROJECTED_USAGE_GATE: dict[str, float] = {
    "pass_yds": 150.0,
    "rush_yds": 25.0,
    "rec_yds": 25.0,
    "receptions": 2.5,
}


def is_propable_projected(market: str, dist_mean: float) -> bool:
    """True if the sim's PROJECTED usage for a player that week -- i.e. the
    sim's own dist mean for `market`, computed pre-game from leakage-free
    rates/shares -- clears the threshold at which that market would
    realistically have had a prop line offered.

    This is the deployment-population counterpart to
    `scripts.backtest_sim_nfl.is_propable`: that function gates on ACTUAL
    (post-game, realized) usage, which is selection-biased -- it only ever
    grades weeks the model could not have foreseen, silently favoring weeks
    where a featured player happened to stay featured. Gating on the sim's
    own PROJECTED usage instead reproduces how books actually decide whether
    to post a line (pre-game, off expectation, not off what happens to
    occur), so it's the metric that should be read as the ship-gate's
    headline shippability number. Unknown markets are never propable.
    """
    threshold = PROJECTED_USAGE_GATE.get(market)
    if threshold is None:
        return False
    return float(dist_mean) >= threshold


# =============================================================================
# assemble_prop_rows -- pure join of nfl_player_sim_current to odds_snapshot
# prop rows, producing ev_prop_picks-shaped dicts (Task 4).
# =============================================================================

def assemble_prop_rows(sim_rows: list[dict], odds_rows: list[dict], model_version: str) -> list[dict]:
    """Join sim player-prop distributions to book prop odds and emit +EV pick
    rows shaped exactly like `sportsmodel.db._EV_PROP_PICKS_COLS`.

    `sim_rows`: dicts from `nfl_player_sim_current` (game_pk, player_id, name,
    market, mean, dist, commence_time, optional matchup). `odds_rows`: latest
    prop `odds_snapshot` rows (game_pk, market, side, player_name, book, line,
    price), where `market` is already the sim-side/prop_market_map CODE (e.g.
    "reception_yds").

    For each sim row: map its market to the odds-side market
    (`odds_market_for`), skipping markets excluded from C v1 (e.g. pass_yds);
    apply the projected-usage deployment gate (`is_propable_projected`);
    find odds rows for the same game + market + (name-normalized) player;
    pick the MAIN line (offered by the most distinct books across over+under,
    ties broken by the lowest line); no-vig the best over/under prices at
    that line against the sim's own P(over); and keep the higher-EV side.
    Rows failing any step (no odds match, missing one side, NaN prob, no
    MAJOR_BOOKS price) are simply excluded -- this function never raises on
    a single row's bad/missing data.
    """
    rows: list[dict] = []
    for sim_row in sim_rows:
        sim_market = sim_row.get("market")
        odds_market = odds_market_for(sim_market)
        if odds_market is None:
            continue
        mean = sim_row.get("mean")
        if mean is None:
            continue
        try:
            propable = is_propable_projected(sim_market, mean)
        except (TypeError, ValueError):
            continue
        if not propable:
            continue

        game_pk = sim_row.get("game_pk")
        name_key = normalize_player_name(sim_row.get("name"))
        candidates = [
            o for o in odds_rows
            if o.get("game_pk") == game_pk
            and o.get("market") == odds_market
            and normalize_player_name(o.get("player_name")) == name_key
        ]
        if not candidates:
            continue

        # Group by line, counting distinct books across both sides; the main
        # line is the one the most books post, ties broken by the lowest line.
        books_by_line: dict[float, set] = {}
        for o in candidates:
            line = o.get("line")
            if line is None:
                continue
            books_by_line.setdefault(line, set()).add(o.get("book"))
        if not books_by_line:
            continue
        main_line = min(books_by_line, key=lambda l: (-len(books_by_line[l]), l))

        over_entries = [
            (o.get("book"), o.get("price")) for o in candidates
            if o.get("line") == main_line and o.get("side") == "over"
        ]
        under_entries = [
            (o.get("book"), o.get("price")) for o in candidates
            if o.get("line") == main_line and o.get("side") == "under"
        ]
        if not over_entries or not under_entries:
            continue

        p_over = prob_over_dist(sim_row.get("dist"), main_line)
        if p_over != p_over:  # NaN
            continue

        ob = best_price(over_entries)
        ub = best_price(under_entries)
        if ob is None or ub is None:
            continue

        novig_over = novig(ob[1], ub[1])
        ev_over = ev(p_over, ob[1])
        ev_under = ev(1 - p_over, ub[1])

        if ev_over >= ev_under:
            side = "over"
            model_prob, market_prob, ev_best, picked, side_entries = (
                p_over, novig_over, ev_over, ob, over_entries,
            )
        else:
            side = "under"
            model_prob, market_prob, ev_best, picked, side_entries = (
                1 - p_over, 1 - novig_over, ev_under, ub, under_entries,
            )

        pinnacle_price = None
        for bk, price in side_entries:
            if "pinnacle" in (bk or "").lower():
                pinnacle_price = price
                break

        rows.append({
            "sport": "nfl",
            "game_pk": game_pk,
            "player_id": sim_row.get("player_id"),
            "player_name": sim_row.get("name"),
            "market": sim_market,
            "side": side,
            "line": main_line,
            "model_version": model_version,
            "matchup": sim_row.get("matchup"),
            "commence_time": sim_row.get("commence_time"),
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": model_prob - market_prob,
            "ev_best": ev_best,
            "best_book": picked[0],
            "best_price": picked[1],
            "pinnacle_price": pinnacle_price,
            "open_pinnacle_price": pinnacle_price,
            "is_pick": ev_best > 0,
        })
    return rows
