"""Game-level MLB model: per-PA vectors -> expected runs -> score dist -> win prob.

Implements docs/methodology.md Part B. All functions are pure and testable without
data. Constants are the tunable defaults from the spec's parameter table.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import lgamma, log

# wOBA linear weights (2023-era) and run-environment constants. [tunable, recalibrate]
WOBA_WEIGHTS = {"p_bb": 0.69, "p_1b": 0.89, "p_2b": 1.27, "p_3b": 1.62, "p_hr": 2.10}
LG_WOBA = 0.320
WOBA_SCALE = 1.24
LG_RUNS_PER_PA = 0.118
PA_PER_TEAM_GAME = 38.0
RUNS_VAR_MULT = 1.25          # team-runs overdispersion: Var = mult * mean
EXTRA_INNING_HOME_EDGE = 0.04
_MAX_RUNS = 30                 # truncation for the runs PMF


BETA_TEMP = 0.006          # HR multiplier per °F above 70 [tunable]
_HR_MULT_CLAMP = (0.80, 1.20)


def weather_hr_multiplier(temp_f: float) -> float:
    """Warm air carries the ball: HR multiplier from game-time temperature."""
    m = 1 + BETA_TEMP * (temp_f - 70)
    return min(max(m, _HR_MULT_CLAMP[0]), _HR_MULT_CLAMP[1])


def apply_hr_multiplier(vec: dict[str, float], mult: float) -> dict[str, float]:
    """Scale a vector's HR rate and renormalize to a valid per-PA distribution."""
    v = dict(vec)
    v["p_hr"] = v["p_hr"] * mult
    total = sum(v.values())
    return {k: val / total for k, val in v.items()}


def apply_park_to_vector(vec: dict[str, float], pf: float) -> dict[str, float]:
    """Fold a run park factor into a per-PA vector (scale all hit outcomes, renormalize).

    For props we need park at the outcome level (more hits/HR at Coors), not just as a
    team-run multiplier. Approximate: scale 1B/2B/3B/HR by the run factor.
    """
    v = dict(vec)
    for k in ("p_1b", "p_2b", "p_3b", "p_hr"):
        v[k] = v[k] * pf
    total = sum(v.values())
    return {k: val / total for k, val in v.items()}


def apply_bip_defense(vec: dict[str, float], factor: float) -> dict[str, float]:
    """Scale ball-in-play hits (1B/2B/3B) by the fielding team's defense factor.

    factor < 1 = better defense (fewer BIP hits). HR excluded (defense can't touch it).
    """
    v = dict(vec)
    for k in ("p_1b", "p_2b", "p_3b"):
        v[k] = v[k] * factor
    total = sum(v.values())
    return {k: val / total for k, val in v.items()}


def expected_woba(vec: dict[str, float]) -> float:
    """Team/'batter' wOBA from a per-PA outcome vector."""
    return sum(w * vec[o] for o, w in WOBA_WEIGHTS.items())


def expected_runs(vec: dict[str, float], *, park_factor: float = 1.0,
                  pa: float = PA_PER_TEAM_GAME) -> float:
    """Expected runs for a team from its per-PA vector (wRC conversion)."""
    woba = expected_woba(vec)
    runs_per_pa = (woba - LG_WOBA) / WOBA_SCALE + LG_RUNS_PER_PA
    return max(runs_per_pa, 0.0) * pa * park_factor


def _nb_pmf(mean: float, xmax: int = _MAX_RUNS) -> list[float]:
    """Negative-binomial PMF with Var = RUNS_VAR_MULT*mean (overdispersed)."""
    if mean <= 0:
        pmf = [0.0] * (xmax + 1)
        pmf[0] = 1.0
        return pmf
    var = RUNS_VAR_MULT * mean
    r = mean * mean / (var - mean)          # = mean/(mult-1)
    p = mean / var                          # = 1/mult
    # P(x) = gamma(x+r)/(gamma(r) x!) p^r (1-p)^x
    logp, log1mp = log(p), log(1 - p)
    out = []
    for x in range(xmax + 1):
        logpmf = (
            lgamma(x + r) - lgamma(r) - lgamma(x + 1)
            + r * logp + x * log1mp
        )
        out.append(pow(2.718281828459045, logpmf))
    s = sum(out)
    return [v / s for v in out]             # renormalize after truncation


def score_distribution(mean_runs: float) -> list[float]:
    """PMF over a team's runs (index = runs), given expected runs."""
    return _nb_pmf(mean_runs)


def win_total_probabilities(
    home_runs: float, away_runs: float
) -> dict[str, float]:
    """Home win prob and expected total from the two teams' expected runs.

    Assumes independence of the two teams' run counts (Phase-1 approximation).
    Ties resolved by an extra-innings home edge.
    """
    hp = score_distribution(home_runs)
    ap = score_distribution(away_runs)
    p_home, p_tie = 0.0, 0.0
    for h, ph in enumerate(hp):
        for a, pa in enumerate(ap):
            if h > a:
                p_home += ph * pa
            elif h == a:
                p_tie += ph * pa
    p_home += p_tie * (0.5 + EXTRA_INNING_HOME_EDGE)
    return {
        "home_win_prob": p_home,
        "pred_home_score": home_runs,
        "pred_away_score": away_runs,
        "pred_total": home_runs + away_runs,
        "pred_margin": home_runs - away_runs,
    }


def prob_total_over(home_runs: float, away_runs: float, line: float) -> float:
    """P(home_runs + away_runs > line) from the joint independent score dist."""
    hp = score_distribution(home_runs)
    ap = score_distribution(away_runs)
    p = 0.0
    for h, ph in enumerate(hp):
        for a, pa in enumerate(ap):
            if h + a > line:
                p += ph * pa
    return p
