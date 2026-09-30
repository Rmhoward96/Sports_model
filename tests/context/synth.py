"""Synthetic unit-games league for context.units / context.matchup tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.context.units import METRICS, UNIT_GAME_COLUMNS

TEAMS = list("ABCDEFGH")
BASE = {"pass_epa": 0.05, "pass_success": 0.45, "pass_explosive": 0.08,
        "run_epa": -0.05, "run_success": 0.40, "run_explosive": 0.10}
SD = {"epa": 0.15, "success": 0.05, "explosive": 0.03}


def _sd(metric: str) -> float:
    return SD[metric.split("_", 1)[1]]


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Circle method: len(teams)-1 rounds of (home, away) pairs."""
    t = list(teams)
    rounds = []
    for _ in range(len(t) - 1):
        rounds.append([(t[i], t[-1 - i]) for i in range(len(t) // 2)])
        t = [t[0], t[-1]] + t[1:-1]
    return rounds


def league(seasons, n_weeks: int = 14, profile: dict | None = None,
           noise: float = 0.5, seed: int = 7) -> pd.DataFrame:
    """One offense row per team-game (UNIT_GAME_COLUMNS).

    ``profile[team]`` = {"pass_o", "pass_d", "run_o", "run_d"} in SD units; ``_o`` is
    offensive strength (+ = better), ``_d`` is defensive weakness (+ = allows more).
    """
    profile = profile or {}
    rng = np.random.default_rng(seed)
    sched = round_robin(TEAMS)
    rows = []
    for s in seasons:
        for w in range(1, n_weeks + 1):
            for gi, (h, a) in enumerate(sched[(w - 1) % len(sched)]):
                gid = f"{s}_{w:02d}_{gi}"
                for team, opp in ((h, a), (a, h)):
                    r = {"season": s, "week": w, "game_id": gid, "team": team,
                         "opponent": opp, "pass_plays": 35.0, "run_plays": 25.0,
                         "pass_rate": 35 / 60}
                    for m in METRICS:
                        unit = m.split("_", 1)[0]
                        q = (profile.get(team, {}).get(f"{unit}_o", 0.0)
                             + profile.get(opp, {}).get(f"{unit}_d", 0.0))
                        r[m] = BASE[m] + _sd(m) * (0.5 * q + noise * rng.standard_normal())
                    rows.append(r)
    return pd.DataFrame(rows, columns=UNIT_GAME_COLUMNS)
