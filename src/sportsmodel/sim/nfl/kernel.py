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

# Per-game usage dispersion: before the multinomial target/carry split, the
# whole share vector is redrawn from a Dirichlet(alpha_i = C * p_i) (one draw
# per team per sim) in place of the fixed shares. This makes a player's
# overall game usage boom/bust from one simulated game to the next, on top of
# the multinomial sampling noise within a single game -- and, being a true
# Dirichlet, it is exactly mean-preserving (E[dispersed_share_i] == p_i), so
# it widens variance without dragging shares toward equal. Concentration C --
# higher C means the draw hugs the true shares more tightly (a naive
# `p_i * Gamma(k, 1/k)` then renormalize is NOT mean-preserving and was
# rejected: it regresses shares toward equal, understating workhorse usage).
# Retuned in B.2 (2026-09-18): raised 20 -> 150 against the propable-player
# walk-forward. At 20 the per-game usage draw was over-dispersed (too
# boom/bust), dragging each player's simulated median below the true median
# (coverage_p50 ~.33-.41) with a fat upper tail (coverage_p90 high); 150 hugs
# the recent-usage shares more tightly, lifting the median toward the mean
# (receptions coverage_p50 ~.47-.48, in the .45-.55 band) and pulling p90 into
# ~.88-.90 -- all while leaving the mean (and thus MAE) unchanged. Isolated as
# the single effective lever: raising _GAME_ENV_K helped far less and risks the
# shared-env correlation feature; yard-shape changes only trade p50 for p90.
# Tunable -- retune against the walk-forward backtest.
_USAGE_CONCENTRATION = 150.0

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

# Clamp for the combined home-field + power-rating per-team scoring tilt, so even
# a lopsided matchup keeps a sane drive-outcome mix (no all-TD / all-punt games).
_MIN_TILT = 0.4
_MAX_TILT = 1.6

_PLAYER_STAT_NAMES = ("pass_yds", "rush_yds", "rec_yds", "receptions", "td", "pass_tds", "rush_att")

# TD allocation blends each player's recent-window TD share with their
# usage/opportunity share (targets*catch for receiving, carries for rushing).
# Recent per-player TD counts over a ~5-game window are extremely noisy -- a
# receiver can post real volume yet zero TDs by variance -- so a pure recent-TD
# share hard-zeros the anytime-TD probability of anyone who simply hasn't scored
# lately, which is wrong (a 5-catch receiver has a real TD chance). Shrinking
# toward opportunity floors every player with real volume at a nonzero TD chance
# while still crediting recent scorers. 0.0 = pure opportunity, 1.0 = pure
# recent-TD share. Tunable -- retune against the walk-forward backtest.
_TD_RECENCY_WEIGHT = 0.5


def sample_drive(off: TeamRates, deff: TeamRates, rng, score_tilt: float = 1.0) -> tuple[str, int]:
    """Sample a single drive outcome combining offense and defense rates.

    Combines the offense's drive-outcome probabilities with the DEFENSE's
    drives-allowed probabilities by averaging them element-wise, then draws a
    single outcome from the resulting multinomial distribution.

    Args:
        off: Offensive team rates.
        deff: Defensive (drives-allowed) team rates -- see
            `rates.team_defense_rates_from_pbp`. (Historically this was the
            opponent's OFFENSIVE rates; that mismodeled defense.)
        rng: numpy random Generator (e.g., np.random.default_rng()).
        score_tilt: multiplies the TD+FG mass before renormalizing (>1 => more
            scoring, <1 => less). 1.0 is a no-op; the home-field edge passes
            1+home_field for the home offense and 1-home_field for the away.

    Returns:
        Tuple of (outcome_key, points) where outcome_key is one of
        'td', 'fg', 'punt', 'turnover', 'downs', 'end' and points is
        7 for TD, 3 for FG, 0 otherwise.
    """
    # Average the offense's drive outcomes with the defense's drives-allowed.
    combined = {}
    for key in off.drive_outcomes:
        combined[key] = (off.drive_outcomes[key] + deff.drive_outcomes[key]) / 2.0

    # Home-field / environment tilt: scale scoring outcomes, clip to >=0.
    if score_tilt != 1.0:
        combined["td"] = max(0.0, combined.get("td", 0.0) * score_tilt)
        combined["fg"] = max(0.0, combined.get("fg", 0.0) * score_tilt)

    # Renormalize to sum to 1
    total = sum(combined.values())
    combined = {k: v / total for k, v in combined.items()} if total > 0 else combined

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
    """Redraw the share vector from a Dirichlet(alpha_i = C * p_i) and return it.

    Drawn once per call (i.e. once per team per sim by `attribute_offense`'s
    caller), this is what makes a player's overall game usage boom/bust from
    one simulated game to the next, on top of the within-game multinomial
    sampling noise. Implemented via Gamma draws rather than
    `rng.dirichlet` directly, since a player with p_i == 0 needs alpha_i == 0
    (a hard zero, not just a tiny Dirichlet weight) and `rng.dirichlet`
    rejects zero alphas: for each player with p_i > 0, raw_i ~ Gamma(C * p_i,
    1); players with p_i == 0 get raw_i = 0; dispersed_i = raw_i / sum(raw).
    This is exactly mean-preserving (E[dispersed_i] == p_i) -- unlike a naive
    `p_i * Gamma(k, 1/k)` then renormalize, which regresses shares toward
    equal and was rejected for that reason. A no-op when `probs` is empty;
    falls back to the undispersed probs if every player has a zero share
    (raw all-zero), so `_normalized_probs`'s uniform-fallback guarantee is
    preserved.
    """
    if len(probs) == 0:
        return probs
    raw = np.zeros(len(probs), dtype=float)
    positive = probs > 0
    if np.any(positive):
        raw[positive] = rng.gamma(_USAGE_CONCENTRATION * probs[positive], 1.0)
    total = raw.sum()
    if total <= 0:
        return probs
    return raw / total


def _blend_td_weights(td_shares: list[float], opp_weights: list[float]) -> np.ndarray:
    """Blend a recent-TD share vector with an opportunity-weight vector into a
    single normalized TD-allocation probability vector.

    Each input is first normalized to a probability vector, then combined as
    ``_TD_RECENCY_WEIGHT * td + (1 - _TD_RECENCY_WEIGHT) * opp`` and renormalized.
    When the team has NO recent TD signal at all (``sum(td_shares) <= 0``) the
    result is pure opportunity -- blending in a uniform-fallback TD vector there
    would wrongly hand TD credit to zero-opportunity players (e.g. a receiving
    TD to a QB). This floors any player with real opportunity at a nonzero TD
    probability while still crediting recent scorers more.
    """
    opp = _normalized_probs(opp_weights)
    if sum(s for s in td_shares if s > 0) <= 0:
        return opp
    td = _normalized_probs(td_shares)
    return _normalized_probs(_TD_RECENCY_WEIGHT * td + (1.0 - _TD_RECENCY_WEIGHT) * opp)


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
    pass_td_share: float = 0.58,
) -> dict[str, dict[str, int]]:
    """Attribute a drive-based offensive box score to individual players.

    Allocates `n_pass` targets across `players` by `target_share` and
    `n_rush` carries by `carry_share` (both via multinomial draws, after
    redrawing the share vector from a mean-preserving Dirichlet per call --
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
        "receptions", "td", "pass_tds", "rush_att"}, all ints.
    """
    if not players:
        return {}

    stats: dict[str, dict[str, int]] = {
        p.player_id: {"pass_yds": 0, "rush_yds": 0, "rec_yds": 0, "receptions": 0, "td": 0, "pass_tds": 0, "rush_att": 0}
        for p in players
    }

    target_probs = _apply_usage_dispersion(
        _normalized_probs([p.target_share for p in players]), rng
    )
    carry_probs = _apply_usage_dispersion(
        _normalized_probs([p.carry_share for p in players]), rng
    )
    # Split the game's offensive TDs into passing (receiving) vs rushing, then
    # allocate each pool separately: passing TDs by receiving-TD share (credits
    # the receiver AND the QB's pass_tds), rushing TDs by rushing-TD share. Each
    # pool's recent-TD share is BLENDED with opportunity (targets*catch for
    # receiving, carries for rushing) via `_blend_td_weights`, so a player with
    # real volume but no recent TDs still gets a nonzero TD chance instead of a
    # hard zero, while recent scorers keep the edge. Receiving TDs never go to a
    # QB (its receiving opportunity and share are both zeroed here).
    n_pass_td = int(rng.binomial(n_off_tds, pass_td_share)) if n_off_tds > 0 else 0
    n_rush_td = n_off_tds - n_pass_td
    rec_shares = [p.rec_td_share if p.pos != "QB" else 0.0 for p in players]
    rec_opp = [p.target_share * p.catch_rate if p.pos != "QB" else 0.0 for p in players]
    rush_shares = [p.rush_td_share for p in players]
    rush_opp = [p.carry_share for p in players]
    rec_td_probs = _blend_td_weights(rec_shares, rec_opp)
    rush_td_probs = _blend_td_weights(rush_shares, rush_opp)

    target_counts = rng.multinomial(n_pass, target_probs)
    carry_counts = rng.multinomial(n_rush, carry_probs)
    rec_td_counts = rng.multinomial(n_pass_td, rec_td_probs)
    rush_td_counts = rng.multinomial(n_rush_td, rush_td_probs)

    for player, n_targets in zip(players, target_counts):
        pdata = stats[player.player_id]
        for _ in range(int(n_targets)):
            if rng.random() < player.catch_rate:
                yds = _skewed_play_yards(player.ypr, _REC_YDS_SHAPE, 0.0, rng)
                pdata["rec_yds"] += int(round(yds))
                pdata["receptions"] += 1

    for player, n_carries in zip(players, carry_counts):
        pdata = stats[player.player_id]
        pdata["rush_att"] += int(n_carries)
        for _ in range(int(n_carries)):
            yds = _skewed_play_yards(player.ypc, _RUSH_YDS_SHAPE, _MIN_PLAY_YDS, rng)
            pdata["rush_yds"] += int(round(yds))

    for player, n_rt, n_ru in zip(players, rec_td_counts, rush_td_counts):
        stats[player.player_id]["td"] += int(n_rt) + int(n_ru)

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
    # Every passing (receiving) TD is a passing TD for the starting QB.
    stats[qb.player_id]["pass_tds"] = int(n_pass_td)

    return stats


def _simulate_team_drives(
    off: TeamRates,
    deff: TeamRates,
    players: list[PlayerInput],
    rng,
    game_env: float = 1.0,
    score_tilt: float = 1.0,
) -> tuple[int, dict[str, dict[str, int]]]:
    """Simulate one team's possessions for a single game and attribute stats.

    Draws a Poisson drive count around `off.drives_per_game * game_env`
    (floored at `_MIN_DRIVES_PER_GAME`), samples each drive via
    `sample_drive` to get the team's points and TD count, then determines an
    offensive play count to feed `attribute_offense`:

    - B.3 (primary path): when `off` carries real per-game volume
      (`pass_att_pg` and/or `rush_att_pg` > 0), those counts -- which already
      exclude sacks, so they're real attempted-pass targets, not
      pass-play-snaps -- are used directly as `n_pass`/`n_rush`, scaled by
      the same `game_env` that scales the drive count. This anchors
      box-score volume to the team's real per-game attempts instead of
      deriving it from the (coarser, and previously biased +63 pass_yds /
      -6 rush_yds) `n_drives * _PLAYS_PER_DRIVE * pass_rate` estimate.
    - Legacy fallback: when `off` has no volume fields (both 0.0, the
      `TeamRates` default), falls back to the pre-B.3 derivation --
      converts the drive count into a play count (`_PLAYS_PER_DRIVE`
      plays/drive) split pass/run by `off.pass_rate` -- so specs/tests built
      before B.3 keep working unchanged.

    `game_env` is the shared per-sim game-environment multiplier (see
    `simulate_game`): passing the SAME value in for both teams in a sim is
    what correlates their scoring AND volume (a "shootout" or "defensive
    slog" sim lifts/depresses both teams' drive counts and attempt counts
    together), rather than treating each team's game as independent.
    Defaults to 1.0 (no scaling) so this function is usable standalone/in
    tests without opting into that.

    Returns:
        Tuple of (points, box) where box is attribute_offense's per-player
        stat dict for this team (containing "pass_yds", "rush_yds", "rec_yds",
        "receptions", "td", "pass_tds", "rush_att").
    """
    n_drives = max(_MIN_DRIVES_PER_GAME, int(rng.poisson(off.drives_per_game * game_env)))

    points = 0
    n_off_tds = 0
    for _ in range(n_drives):
        outcome, pts = sample_drive(off, deff, rng, score_tilt=score_tilt)
        points += pts
        if outcome == "td":
            n_off_tds += 1

    if off.pass_att_pg > 0.0 or off.rush_att_pg > 0.0:
        # B.3: anchor box-score volume to the team's real per-game attempts
        # (pass_att_pg already EXCLUDES sacks), scaled by the shared game_env.
        # Draw the counts from a Poisson around that mean so per-game attempt
        # volume carries realistic game-to-game variance (game script, pace,
        # injuries). A deterministic round() left only the shared game_env as a
        # noise source, under-dispersing every player marginal (walk-forward
        # coverage_p90 fell to ~.78-.83); the Poisson restores that spread
        # without changing the mean.
        n_pass = int(rng.poisson(off.pass_att_pg * game_env))
        n_rush = int(rng.poisson(off.rush_att_pg * game_env))
    else:
        # legacy fallback for specs/tests without volume fields
        n_plays = round(n_drives * _PLAYS_PER_DRIVE)
        n_pass = round(n_plays * off.pass_rate)
        n_rush = n_plays - n_pass

    box = attribute_offense(players, n_pass, n_rush, n_off_tds, rng, pass_td_share=off.pass_td_share)
    return points, box


def simulate_game(spec: NflGameSpec, n_sims: int, rng, home_field: float = 0.0,
                  ratings_tilt: float = 0.0) -> NflGameSims:
    """Simulate n_sims independent NFL games from a full game spec.

    Each team's scoring is drawn against the OPPONENT'S DEFENSE (spec.away_def
    for the home offense, spec.home_def for the away offense); when a spec omits
    the defensive rates the engine falls back to the opponent's offensive rates
    (legacy behavior). `home_field` (>=0) tilts the home offense's scoring up by
    (1+home_field) and the away offense's down by (1-home_field) -- the sim's
    home-field edge, 0.0 = neutral. `ratings_tilt` (signed, positive = home
    stronger) is an additional power-rating scoring tilt for the matchup on top
    of home_field, letting a top team separate from a weak one more than the
    bottom-up drive rates alone do; the caller derives it from Elo/SRS. Both
    combine into per-team score tilts, clamped to [_MIN_TILT, _MAX_TILT] so a
    huge mismatch can't produce a degenerate all-scoring/all-punting drive mix.

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
        and player_stats keyed by player_id, with per-stat arrays for each of
        "pass_yds", "rush_yds", "rec_yds", "receptions", "td", "pass_tds", "rush_att".
    """
    home_score = np.zeros(n_sims, dtype=np.int64)
    away_score = np.zeros(n_sims, dtype=np.int64)

    all_players = [*spec.home_players, *spec.away_players]
    player_stats: dict[str, dict[str, np.ndarray]] = {
        p.player_id: {m: np.zeros(n_sims, dtype=np.int64) for m in _PLAYER_STAT_NAMES}
        for p in all_players
    }

    # Opponent defense for each offense; fall back to opponent offense (legacy).
    home_deff = spec.away_def if spec.away_def is not None else spec.away
    away_deff = spec.home_def if spec.home_def is not None else spec.home
    # Home-field + power-rating tilt (positive ratings_tilt favors home), clamped.
    home_tilt = min(_MAX_TILT, max(_MIN_TILT, 1.0 + home_field + ratings_tilt))
    away_tilt = min(_MAX_TILT, max(_MIN_TILT, 1.0 - home_field - ratings_tilt))

    for i in range(n_sims):
        game_env = rng.gamma(_GAME_ENV_K, 1.0 / _GAME_ENV_K)
        h_points, h_box = _simulate_team_drives(
            spec.home, home_deff, spec.home_players, rng, game_env=game_env, score_tilt=home_tilt
        )
        a_points, a_box = _simulate_team_drives(
            spec.away, away_deff, spec.away_players, rng, game_env=game_env, score_tilt=away_tilt
        )
        home_score[i] = h_points
        away_score[i] = a_points

        for box in (h_box, a_box):
            for pid, box_stats in box.items():
                for market in _PLAYER_STAT_NAMES:
                    player_stats[pid][market][i] = box_stats[market]

    return NflGameSims(home_score, away_score, player_stats)
