"""Drive-based Monte Carlo kernel for NFL simulation.

Provides pure kernel helpers for sampling drive outcomes and computing
game results.
"""
from __future__ import annotations

import numpy as np

from sportsmodel.sim.nfl.spec import NflGameSims, NflGameSpec, PlayerInput, TeamRates

# Per-play yardage on a single reception/carry is drawn from a right-skewed
# Gamma distribution (real boom/bust yardage is nothing like a thin Normal:
# most plays gain modestly, a few break big). The Gamma's scale is set each
# draw as mean/shape so the player's rate stat (ypr for receptions, ypc for
# carries) is preserved regardless of shape; shape alone controls variance/
# skew (lower shape => fatter right tail). These are tunable -- retune
# against the walk-forward backtest.
_REC_YDS_SHAPE = 1.0
_RUSH_YDS_SHAPE = 1.5

# Lower support bound for a single carry's yardage (rushing can lose yards,
# e.g. a tackle for loss); a reception's yardage floors at 0 instead (you
# can't lose yards on a completed catch). Implemented as a left-shift of the
# Gamma draw (see `_skewed_play_yards`) so the mean is still exactly the
# player's rate stat, with a defensive max() floor against any rounding.
_MIN_PLAY_YDS = -5.0

# Per-game usage dispersion: before the multinomial target/carry split, each
# player's share is multiplied by its own Gamma(mean=1) draw (one draw per
# player per team per sim) and the results renormalized. This makes a
# player's game-to-game target/carry volume itself boom/bust, on top of the
# multinomial sampling noise within a single game. Concentration k -- higher
# k means the multiplier hugs 1.0 more tightly (CV = 1/sqrt(k)); tunable.
_USAGE_DISPERSION_K = 4.0

# Shared game-environment multiplier: one Gamma(mean=1) draw per sim, applied
# to BOTH teams' expected drive count, so a "fast/high-scoring" or
# "slow/defensive" simulated game moves both teams' volume together --
# without this, home and away scoring are independent within a sim, which
# understates real single-game total variance and the total/margin
# dependence. Tunable.
_GAME_ENV_K = 8.0

# Drive-count and play-count assumptions for converting a team's simulated
# drives into an offensive play count to feed attribute_offense.
_MIN_DRIVES_PER_GAME = 6
_PLAYS_PER_DRIVE = 6.0

_PLAYER_STAT_NAMES = ("pass_yds", "rush_yds", "rec_yds", "receptions", "td")


def sample_drive(off: TeamRates, deff: TeamRates, rng) -> tuple[str, int]:
    """Sample a single drive outcome combining offense and defense rates.

    Combines the offense and defense drive-outcome probabilities by averaging
    them element-wise, then draws a single outcome from the resulting
    multinomial distribution.

    Args:
        off: Offensive team rates.
        deff: Defensive team rates.
        rng: numpy random Generator (e.g., np.random.default_rng()).

    Returns:
        Tuple of (outcome_key, points) where outcome_key is one of
        'td', 'fg', 'punt', 'turnover', 'downs', 'end' and points is
        7 for TD, 3 for FG, 0 otherwise.
    """
    # Average the drive outcome rates element-wise
    combined = {}
    for key in off.drive_outcomes:
        combined[key] = (off.drive_outcomes[key] + deff.drive_outcomes[key]) / 2.0

    # Renormalize to sum to 1
    total = sum(combined.values())
    combined = {k: v / total for k, v in combined.items()}

    # Prepare for multinomial draw
    outcome_keys = list(combined.keys())
    probs = np.array([combined[k] for k in outcome_keys])

    # Create cumulative probabilities
    cumsum = np.cumsum(probs)

    # Draw uniform random value and find corresponding outcome
    rand_val = rng.random()
    outcome_idx = np.searchsorted(cumsum, rand_val)

    # Cap to valid range (handles edge case where rand_val is very close to 1.0)
    outcome_idx = min(outcome_idx, len(outcome_keys) - 1)
    outcome_key = outcome_keys[outcome_idx]

    # Map outcome to points
    points_map = {"td": 7, "fg": 3}
    points = points_map.get(outcome_key, 0)

    return (outcome_key, points)


def _normalized_probs(shares: list[float]) -> np.ndarray:
    """Clip negative shares to 0 and normalize to sum to 1.

    Falls back to a uniform distribution when all shares are non-positive,
    so a multinomial draw never receives an all-zero pvals vector.
    """
    arr = np.clip(np.array(shares, dtype=float), 0.0, None)
    total = arr.sum()
    if total <= 0:
        return np.full(len(arr), 1.0 / len(arr)) if len(arr) else arr
    return arr / total


def _apply_usage_dispersion(probs: np.ndarray, rng) -> np.ndarray:
    """Multiply each player's share by its own Gamma(mean=1) draw and renormalize.

    Drawn once per call (i.e. once per team per sim by `attribute_offense`'s
    caller), this is what makes a player's overall game usage boom/bust from
    one simulated game to the next, on top of the within-game multinomial
    sampling noise. A no-op when `probs` is empty; falls back to the
    unmultiplied probs if every player happens to draw (or already has) a
    zero share, so `_normalized_probs`'s uniform-fallback guarantee is
    preserved.
    """
    if len(probs) == 0:
        return probs
    multipliers = rng.gamma(_USAGE_DISPERSION_K, 1.0 / _USAGE_DISPERSION_K, size=len(probs))
    dispersed = probs * multipliers
    total = dispersed.sum()
    if total <= 0:
        return probs
    return dispersed / total


def _skewed_play_yards(mean: float, shape: float, floor: float, rng) -> float:
    """Draw a single play's yardage from a right-skewed Gamma with the given mean.

    The Gamma is left-shifted by `floor` (its lower support bound) so the
    result can be negative (for carries, `floor=_MIN_PLAY_YDS`) or is
    guaranteed >= 0 (for receptions, `floor=0.0`), while still averaging out
    to exactly `mean`: scale = (mean - floor) / shape, draw = floor +
    Gamma(shape, scale). A final max() against `floor` is a defensive
    no-op (the shifted Gamma's support already starts at `floor`) that
    guards against a non-positive `mean - floor` (e.g. mean <= floor),
    where the Gamma scale would otherwise be degenerate.
    """
    spread = mean - floor
    if spread <= 0:
        return floor
    scale = spread / shape
    return max(floor + rng.gamma(shape, scale), floor)


def attribute_offense(
    players: list[PlayerInput],
    n_pass: int,
    n_rush: int,
    n_off_tds: int,
    rng,
) -> dict[str, dict[str, int]]:
    """Attribute a drive-based offensive box score to individual players.

    Allocates `n_pass` targets across `players` by `target_share` and
    `n_rush` carries by `carry_share` (both via multinomial draws, after
    dispersing each player's share by its own per-call Gamma(mean=1) draw --
    see `_apply_usage_dispersion` -- to add game-to-game usage boom/bust),
    then simulates each individual target/carry: a target becomes a
    reception w.p. `catch_rate`, and each reception's yards are drawn from a
    right-skewed Gamma centered on the player's `ypr` (yards per reception,
    NOT `ypt` -- using ypt here would understate a receiver's output by
    roughly a factor of catch_rate); yards per carry are drawn from a
    right-skewed Gamma centered on the player's `ypc`. Touchdowns
    (`n_off_tds`) are allocated across players by `td_share` via a separate
    multinomial draw (shares not dispersed here -- FIX 3 targets game usage
    volume, not TD credit). The team's passing yards are attributed to
    the QB (the player whose `pos == "QB"`; if zero or
    multiple players have that position, the player with the highest
    `target_share` is used instead) as the sum of the whole team's receiving
    yards.

    Args:
        players: Offensive players to attribute stats to.
        n_pass: Number of pass attempts (targets) to allocate this game.
        n_rush: Number of rush attempts (carries) to allocate this game.
        n_off_tds: Number of offensive touchdowns to allocate this game.
        rng: numpy random Generator (e.g., np.random.default_rng()).

    Returns:
        Dict mapping player_id -> {"pass_yds", "rush_yds", "rec_yds",
        "receptions", "td"}, all ints.
    """
    if not players:
        return {}

    stats: dict[str, dict[str, int]] = {
        p.player_id: {"pass_yds": 0, "rush_yds": 0, "rec_yds": 0, "receptions": 0, "td": 0}
        for p in players
    }

    target_probs = _apply_usage_dispersion(
        _normalized_probs([p.target_share for p in players]), rng
    )
    carry_probs = _apply_usage_dispersion(
        _normalized_probs([p.carry_share for p in players]), rng
    )
    td_probs = _normalized_probs([p.td_share for p in players])

    target_counts = rng.multinomial(n_pass, target_probs)
    carry_counts = rng.multinomial(n_rush, carry_probs)
    td_counts = rng.multinomial(n_off_tds, td_probs)

    for player, n_targets in zip(players, target_counts):
        pdata = stats[player.player_id]
        for _ in range(int(n_targets)):
            if rng.random() < player.catch_rate:
                yds = _skewed_play_yards(player.ypr, _REC_YDS_SHAPE, 0.0, rng)
                pdata["rec_yds"] += int(round(yds))
                pdata["receptions"] += 1

    for player, n_carries in zip(players, carry_counts):
        pdata = stats[player.player_id]
        for _ in range(int(n_carries)):
            yds = _skewed_play_yards(player.ypc, _RUSH_YDS_SHAPE, _MIN_PLAY_YDS, rng)
            pdata["rush_yds"] += int(round(yds))

    for player, n_td in zip(players, td_counts):
        stats[player.player_id]["td"] += int(n_td)

    qb_candidates = [p for p in players if p.pos == "QB"]
    if qb_candidates:
        # One or more players at QB: pick the one with the most passing
        # work among the QBs themselves (never fall through to the whole
        # roster, or a high-target WR/RB would get tagged as the passer).
        qb = max(qb_candidates, key=lambda p: p.target_share)
    else:
        # No listed QB: last resort, use the highest-target player on the
        # whole roster as a stand-in passer.
        qb = max(players, key=lambda p: p.target_share)
    team_rec_yds = sum(pdata["rec_yds"] for pdata in stats.values())
    stats[qb.player_id]["pass_yds"] = team_rec_yds

    return stats


def _simulate_team_drives(
    off: TeamRates,
    deff: TeamRates,
    players: list[PlayerInput],
    rng,
    game_env: float = 1.0,
) -> tuple[int, dict[str, dict[str, int]]]:
    """Simulate one team's possessions for a single game and attribute stats.

    Draws a Poisson drive count around `off.drives_per_game * game_env`
    (floored at `_MIN_DRIVES_PER_GAME`), samples each drive via
    `sample_drive` to get the team's points and TD count, converts the drive
    count into an offensive play count (`_PLAYS_PER_DRIVE` plays/drive, split
    pass/run by `off.pass_rate`), and attributes the game's plays and TDs to
    `players` via a single `attribute_offense` call.

    `game_env` is the shared per-sim game-environment multiplier (see
    `simulate_game`): passing the SAME value in for both teams in a sim is
    what correlates their scoring (a "shootout" or "defensive slog" sim
    lifts/depresses both teams' drive counts together), rather than treating
    each team's game as independent. Defaults to 1.0 (no scaling) so this
    function is usable standalone/in tests without opting into that.

    Returns:
        Tuple of (points, box) where box is attribute_offense's per-player
        stat dict for this team.
    """
    n_drives = max(_MIN_DRIVES_PER_GAME, int(rng.poisson(off.drives_per_game * game_env)))

    points = 0
    n_off_tds = 0
    for _ in range(n_drives):
        outcome, pts = sample_drive(off, deff, rng)
        points += pts
        if outcome == "td":
            n_off_tds += 1

    n_plays = round(n_drives * _PLAYS_PER_DRIVE)
    n_pass = round(n_plays * off.pass_rate)
    n_rush = n_plays - n_pass

    box = attribute_offense(players, n_pass, n_rush, n_off_tds, rng)
    return points, box


def simulate_game(spec: NflGameSpec, n_sims: int, rng) -> NflGameSims:
    """Simulate n_sims independent NFL games from a full game spec.

    Per simulation, a single shared `game_env` multiplier is drawn (Gamma,
    mean 1, concentration `_GAME_ENV_K`) and applied to BOTH teams' expected
    drive counts before `_simulate_team_drives` samples them -- this is what
    correlates home and away scoring within a sim (a "shootout" sim runs hot
    for both offenses; a "slog" sim runs cold for both), matching the real
    within-game total-variance/margin-total dependence that two fully
    independent team simulations would miss. Team strength differences (via
    each team's own drive_outcomes/pass_rate/etc.) still drive the margin.
    Home/away scores and per-player stat arrays are then accumulated across
    sims, with both teams' players stored in a single `player_stats` dict
    keyed by player_id.

    Args:
        spec: Full game specification (both teams' rates and rosters).
        n_sims: Number of independent games to simulate.
        rng: numpy random Generator (e.g., np.random.default_rng()).

    Returns:
        NflGameSims with home_score/away_score (length n_sims int arrays)
        and player_stats keyed by player_id.
    """
    home_score = np.zeros(n_sims, dtype=np.int64)
    away_score = np.zeros(n_sims, dtype=np.int64)

    all_players = [*spec.home_players, *spec.away_players]
    player_stats: dict[str, dict[str, np.ndarray]] = {
        p.player_id: {m: np.zeros(n_sims, dtype=np.int64) for m in _PLAYER_STAT_NAMES}
        for p in all_players
    }

    for i in range(n_sims):
        game_env = rng.gamma(_GAME_ENV_K, 1.0 / _GAME_ENV_K)
        h_points, h_box = _simulate_team_drives(
            spec.home, spec.away, spec.home_players, rng, game_env=game_env
        )
        a_points, a_box = _simulate_team_drives(
            spec.away, spec.home, spec.away_players, rng, game_env=game_env
        )
        home_score[i] = h_points
        away_score[i] = a_points

        for box in (h_box, a_box):
            for pid, box_stats in box.items():
                for market in _PLAYER_STAT_NAMES:
                    player_stats[pid][market][i] = box_stats[market]

    return NflGameSims(home_score, away_score, player_stats)
