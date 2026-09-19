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
