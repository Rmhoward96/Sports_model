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

import json
import re

from ..model.distributions import prob_over_dist
from .board import EV_CEILING, best_price, decimal_odds, ev, novig

# Sim aggregate market -> Odds API / SportConfig["nfl"].prop_market_map key.
# The sim's markets are pass_yds/rush_yds/rec_yds/receptions; the odds side
# uses rush_yds/reception_yds/receptions/pass_yds. rec_yds <-> reception_yds
# is the only name mismatch. pass_yds is intentionally EXCLUDED from C v1 (not
# a key in this dict) -- see module docstring.
SIM_TO_ODDS_MARKET: dict[str, str] = {
    "rush_yds": "rush_yds",
    "rec_yds": "reception_yds",
    "receptions": "receptions",
    "pass_tds": "pass_tds",  # QB passing TDs -- two-sided line (over/under X.5)
    # anytime_td is projection-only (single-sided market) -- deliberately NOT here.
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


def latest_capture_only(rows: list[dict]) -> list[dict]:
    """Keep only rows from each book's most recent pre-kickoff capture per
    (game_pk, market, player_name, book). odds_snapshot retains every capture,
    so without this a book's superseded line (moved since) would still count
    toward the main line and be shopped as a live price."""
    latest_at: dict[tuple, object] = {}
    for r in rows:
        key = (r["game_pk"], r["market"], r["player_name"], r["book"])
        ts = r["captured_at"]
        if key not in latest_at or ts > latest_at[key]:
            latest_at[key] = ts
    return [
        r for r in rows
        if r["captured_at"] == latest_at[(r["game_pk"], r["market"], r["player_name"], r["book"])]
    ]


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
    "pass_tds": 1.0,  # a starting QB projected for >=1 passing TD is "propable"
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

def _is_pinnacle(book: str | None) -> bool:
    """True if `book` is Pinnacle (case-insensitive substring match, same
    convention used elsewhere for this field)."""
    return "pinnacle" in (book or "").lower()


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
    ties broken by the lowest line); no-vig the best NON-Pinnacle ("soft")
    over/under prices at that line against the sim's own P(over); and keep
    the higher-EV side. Pinnacle is the sharp reference (kept only as
    `pinnacle_price`/`open_pinnacle_price` for CLV grading), never shopped as
    a soft book -- a side offered only by Pinnacle is excluded, same as a
    missing side. `is_pick` requires `0 < ev_best <= board.EV_CEILING`, same
    convention as the game pilot, since an EV above that ceiling is almost
    always a stale-line artifact rather than real value (props are more
    stale-line-prone than game markets, not less). Rows failing any step (no
    odds match, missing one side, NaN prob, no shoppable soft price) are
    simply excluded -- this function never raises on a single row's bad/
    missing data.
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

        # Pinnacle is the SHARP reference (kept for pinnacle_price/CLV below),
        # not a shoppable soft book -- best_book/best_price (and the EV/no-vig
        # math derived from them) are shopped only among non-Pinnacle books.
        # A side offered ONLY by Pinnacle has no soft price to shop, so it's
        # excluded via the ob/ub None check just like a missing side.
        soft_over_entries = [(bk, p) for bk, p in over_entries if not _is_pinnacle(bk)]
        soft_under_entries = [(bk, p) for bk, p in under_entries if not _is_pinnacle(bk)]
        ob = best_price(soft_over_entries)
        ub = best_price(soft_under_entries)
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
            if _is_pinnacle(bk):
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
            # Same EV-ceiling convention as the game pilot (board.EV_CEILING):
            # an EV above this is almost always a stale-line artifact, not
            # real value, so it's excluded from the graded record. Props are
            # MORE stale-line-prone than game markets, so this matters more
            # here, not less.
            "is_pick": 0 < ev_best <= EV_CEILING,
        })
    return rows


# =============================================================================
# assemble_prop_line_rows -- pure prop-LINE builder (Task 4): one row per sim
# (game, player, market) projection that has a book line, no usage gate, no
# EV filter. Feeds nfl_prop_lines, tracking every sim projection's accuracy
# against the market line rather than only the +EV board's picks.
# =============================================================================

# Sim market -> odds-side code for the prop-LINES table (every projection with a
# book line, tracked for accuracy). Separate from SIM_TO_ODDS_MARKET so the +EV
# board's market scope is untouched: this one includes pass_yds and rush_att.
LINE_MARKETS: dict[str, str] = {
    "pass_yds": "pass_yds",
    "rush_yds": "rush_yds",
    "rec_yds": "reception_yds",
    "receptions": "receptions",
    "rush_att": "rush_att",
}


def _best_any(entries: list[tuple]) -> tuple | None:
    """(book, price) at the highest decimal odds across ALL books (the line
    display shows the best available price, not only MAJOR_BOOKS)."""
    entries = [(bk, p) for bk, p in entries if p]
    return max(entries, key=lambda e: decimal_odds(e[1])) if entries else None


def assemble_prop_line_rows(sim_rows: list[dict], odds_rows: list[dict]) -> list[dict]:
    """PURE. One nfl_prop_lines row per sim (game, player, market) in
    LINE_MARKETS that has a book line: the main line (most books, ties ->
    lowest), best over/under price across books (either may be None), the
    sim mean as `projection`, P(over) from the sim distribution at the line,
    and the sim's `lean` ("over" iff p_over > 0.5). No usage gate, no EV
    filter. Rows with no odds match or a NaN P(over) are skipped.

    `dist` may arrive as a JSON string (as stored/retrieved from the DB) or
    already as a dict -- both are handled.
    """
    by_key: dict[tuple, list[dict]] = {}
    for o in odds_rows:
        by_key.setdefault(
            (o.get("game_pk"), o.get("market"), normalize_player_name(o.get("player_name"))), []
        ).append(o)
    rows: list[dict] = []
    for s in sim_rows:
        odds_market = LINE_MARKETS.get(s.get("market"))
        if odds_market is None or s.get("mean") is None:
            continue
        cands = [o for o in by_key.get((s.get("game_pk"), odds_market,
                                        normalize_player_name(s.get("name"))), [])
                 if o.get("line") is not None]
        if not cands:
            continue
        books_by_line: dict[float, set] = {}
        for o in cands:
            books_by_line.setdefault(o["line"], set()).add(o.get("book"))
        line = min(books_by_line, key=lambda l: (-len(books_by_line[l]), l))
        at = [o for o in cands if o["line"] == line]
        over = _best_any([(o.get("book"), o.get("price")) for o in at if o.get("side") == "over"])
        under = _best_any([(o.get("book"), o.get("price")) for o in at if o.get("side") == "under"])
        dist = s.get("dist")
        if isinstance(dist, str):
            dist = json.loads(dist)
        p_over = prob_over_dist(dist, line)
        if p_over != p_over:  # NaN
            continue
        rows.append({
            "game_pk": s.get("game_pk"), "player_id": s.get("player_id"),
            "player_name": s.get("name"), "team": s.get("team"), "market": s.get("market"),
            "line": line,
            "over_price": over[1] if over else None, "over_book": over[0] if over else None,
            "under_price": under[1] if under else None, "under_book": under[0] if under else None,
            "projection": float(s["mean"]), "p_over": float(p_over),
            "lean": "over" if p_over > 0.5 else "under",
            "n_books": len(books_by_line[line]), "commence_time": s.get("commence_time"),
        })
    return rows


# =============================================================================
# grade_prop_pick -- pure forward grading of an ev_prop_picks row against the
# player's actual weekly stat + the closing Pinnacle price (Task 5).
# =============================================================================

# Sim/pick market -> nflverse `import_weekly_data` actual-stat column. Keys
# are the SAME sim-side market codes `ev_prop_picks.market` stores (rush_yds/
# rec_yds/receptions -- see SIM_TO_ODDS_MARKET above), not the odds-side
# codes; values are the nflverse weekly column carrying the realized stat for
# that market, joined by gsis player_id. pass_yds is intentionally absent
# (excluded from C v1 -- see module docstring).
SIM_MARKET_TO_WEEKLY: dict[str, str] = {
    "rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "receptions": "receptions",
    "pass_tds": "passing_tds",
}


def _implied_prob(american_price: int) -> float:
    """Single-sided implied probability from an American price. Same formula
    as grade_ev._implied_prob -- kept local here so this module stays pure
    and import-free of scripts/grade_ev.py (which isn't a package module)."""
    if american_price > 0:
        return 100.0 / (american_price + 100.0)
    return -american_price / (-american_price + 100.0)


def grade_prop_pick(pick: dict, actual: float | None, closing_price: int | None) -> dict:
    """Pure grading: one `ev_prop_picks` row (as a dict) + the player's
    ACTUAL realized stat that week + the closing Pinnacle price for that same
    (game, player, market, side) -> an `ev_prop_results` row (dict, columns
    matching `db._EV_PROP_RESULTS_COLS`). No network, no DB.

    `pick` = {sport, game_pk, player_id, player_name, market, side
    ("over"/"under"), line, model_version, commence_time, model_prob,
    best_price (the pick-time best-book American price the bet was placed
    at), pinnacle_price (the pick-time Pinnacle American price for `side`)}.

    `actual`: the player's realized stat value for `market` that week (e.g.
    receiving yards), or None if it couldn't be resolved (caller skips
    grading in that case -- this function still degrades gracefully and
    returns result/profit=None rather than raising).

    `closing_price`: the CLOSING Pinnacle American price for (game_pk,
    market, side, player, line) -- the last odds_snapshot row at/before
    commence_time. None if no such snapshot exists.

    result: side=="over" -> "win" if actual > line, "loss" if actual < line,
    "push" if actual == line. side=="under" -> mirrored (win if actual <
    line). None if actual or line is missing, or side isn't "over"/"under".

    profit: at the PICK-TIME price the bet was actually placed at
    (`pick["best_price"]`) -- a placed bet's payout is fixed by the price
    taken, not by where the price later moved. win -> decimal_odds(best_price)
    - 1; loss -> -1.0; push -> 0.0. None if best_price is missing on a win
    (no price to grade profit against) or if `result` itself is None.

    novig_close: the single-sided implied probability of `closing_price`
    (None if `closing_price` is None). Not de-vigged (mirrors grade_ev's
    _implied_prob convention -- see its docstring) -- just the closing side's
    own implied probability, useful as a standalone "how short did the market
    close" reference alongside `clv`.

    clv: signed closing-line value from the side the pick took --
    `_implied_prob(closing_price) - _implied_prob(pick["pinnacle_price"])`,
    computed only when BOTH prices are available (None otherwise; no
    number-based fallback exists for props, unlike grade_ev_pick's
    spread/total fallback, since there's no closing-line-NUMBER source for
    props independent of odds_snapshot). Positive means the market moved
    toward this side after the pick was placed (the pick-time price was the
    cheaper, better one) -- same sign convention as grade_ev_pick's clv.
    Computed regardless of `result`, same as grade_ev_pick.
    """
    market = pick.get("market")
    side = pick.get("side")
    line = pick.get("line")

    # ---- result / profit ----
    result = None
    if actual is not None and line is not None and side in ("over", "under"):
        if side == "over":
            if actual > line:
                result = "win"
            elif actual < line:
                result = "loss"
            else:
                result = "push"
        else:  # under
            if actual < line:
                result = "win"
            elif actual > line:
                result = "loss"
            else:
                result = "push"

    profit = None
    if result == "push":
        profit = 0.0
    elif result == "win":
        best_price = pick.get("best_price")
        if best_price is not None:
            profit = decimal_odds(best_price) - 1
    elif result == "loss":
        profit = -1.0

    # ---- clv / novig_close ----
    close_implied = _implied_prob(closing_price) if closing_price is not None else None
    novig_close = close_implied

    clv = None
    pinnacle_price = pick.get("pinnacle_price")
    if close_implied is not None and pinnacle_price is not None:
        clv = close_implied - _implied_prob(pinnacle_price)

    return {
        "sport": pick.get("sport"),
        "game_pk": pick.get("game_pk"),
        "player_id": pick.get("player_id"),
        "player_name": pick.get("player_name"),
        "market": market,
        "side": side,
        "line": line,
        "model_version": pick.get("model_version"),
        "commence_time": pick.get("commence_time"),
        "model_prob": pick.get("model_prob"),
        "novig_close": novig_close,
        "actual": actual,
        "result": result,
        "clv": clv,
        "profit": profit,
    }
