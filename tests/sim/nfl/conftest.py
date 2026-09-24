"""Shared fixtures for the NFL sim tests.

`tbl` is a tiny SYNTHETIC props-ML feature table (never the real parquet):
3 seasons x 17 weeks, 2 teams, 4 players each, with the real tables' column
conventions (keys, `y_*` labels, `p_`/`ngs_`/`tm_`/`op_`/`st_`/`cx_`/`mk_`
features, `p_pos` as category). Targets are Poisson with a rate that rises
with `p_target_share_ewm`. ~10% of rows are "stub" rows with NaN labels, and
`ngs_empty` is 100% NaN, mirroring real-data quirks.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sportsmodel.sim.nfl.spec import NflGameSpec, PlayerInput, TeamRates

SEASONS = (2022, 2023, 2024)
WEEKS = range(1, 18)
TEAMS = {"HOM": "AWY", "AWY": "HOM"}
PLAYERS = {
    "HOM": [("h1", "QB"), ("h2", "WR"), ("h3", "RB"), ("h4", "TE")],
    "AWY": [("a1", "QB"), ("a2", "WR"), ("a3", "RB"), ("a4", "TE")],
}
_BASE_TGT = {"QB": 0.02, "WR": 0.25, "RB": 0.12, "TE": 0.18}
_BASE_CAR = {"QB": 0.12, "WR": 0.02, "RB": 0.60, "TE": 0.01}


def _build_tables(seed: int = 7):
    rng = np.random.default_rng(seed)
    prow, trow = [], []
    for season in SEASONS:
        for week in WEEKS:
            for team, opp in TEAMS.items():
                pass_ewm = rng.uniform(28, 40)
                rush_ewm = rng.uniform(20, 30)
                cx_rest = float(rng.integers(6, 14))
                mk_total = rng.uniform(38, 52)
                trow.append({
                    "team": team, "season": season, "week": week, "opponent": opp,
                    "y_team_pass_att": float(rng.poisson(pass_ewm)),
                    "y_team_rush_att": float(rng.poisson(rush_ewm)),
                    "tm_pass_att_ewm": pass_ewm, "tm_rush_att_ewm": rush_ewm,
                    "op_pass_att_ewm": rng.uniform(28, 40),
                    "cx_rest": cx_rest, "mk_total": mk_total,
                })
                for pid, pos in PLAYERS[team]:
                    tshare = float(np.clip(_BASE_TGT[pos] + rng.normal(0, 0.06), 0.0, 0.5))
                    cshare = float(np.clip(_BASE_CAR[pos] + rng.normal(0, 0.06), 0.0, 0.9))
                    tgts = float(rng.poisson(1.0 + 40 * tshare))
                    cars = float(rng.poisson(0.5 + 25 * cshare))
                    rec = float(rng.binomial(int(tgts), 0.65))
                    row = {
                        "player_id": pid, "season": season, "week": week,
                        "team": team, "opponent": opp, "position": pos,
                        "y_targets": tgts, "y_carries": cars,
                        "y_receptions": rec,
                        "y_rec_yds": rec * 11.0 + rng.normal(0, 3) * (rec > 0),
                        "y_rush_yds": cars * 4.2 + rng.normal(0, 3) * (cars > 0),
                        "p_target_share_ewm": tshare,
                        "p_carry_share_ewm": cshare,
                        "p_pos": pos,
                        "ngs_avg_separation_ewm": rng.uniform(2, 4),
                        "ngs_empty": np.nan,
                        "tm_pass_att_ewm": pass_ewm,
                        "op_pass_att_ewm": rng.uniform(28, 40),
                        "st_questionable": float(rng.random() < 0.05),
                        "cx_rest": cx_rest,
                        "mk_total": mk_total,
                    }
                    if rng.random() < 0.10:  # stub row: no labels
                        for k in list(row):
                            if k.startswith("y_"):
                                row[k] = np.nan
                    prow.append(row)
    player = pd.DataFrame(prow)
    player["p_pos"] = player["p_pos"].astype(
        pd.CategoricalDtype(["QB", "RB", "WR", "TE"]))
    team = pd.DataFrame(trow)
    return player, team


@pytest.fixture(scope="session")
def tbl():
    player, team = _build_tables()

    def rows_for(season, week):
        m = (player["season"] == season) & (player["week"] == week)
        return player[m].reset_index(drop=True)

    def team_rows_for(season, week):
        m = (team["season"] == season) & (team["week"] == week)
        return team[m].reset_index(drop=True)

    return SimpleNamespace(player=player, team=team,
                           rows_for=rows_for, team_rows_for=team_rows_for)


def _rates(pass_att: float, rush_att: float) -> TeamRates:
    return TeamRates(
        drive_outcomes={"td": 0.2, "fg": 0.15, "punt": 0.4, "turnover": 0.12,
                        "downs": 0.05, "end": 0.08},
        pass_rate=0.58, drives_per_game=11.0, rz_td_rate=0.55,
        pass_att_pg=pass_att, rush_att_pg=rush_att,
        sack_rate=0.06, completion_pct=0.64, pass_td_share=0.6,
    )


def _players(team: str) -> list[PlayerInput]:
    return [
        PlayerInput(player_id=pid, name=pid.upper(), pos=pos,
                    target_share=0.25, carry_share=0.25,
                    ypt=7.0, ypc=4.0, ypr=10.5, catch_rate=0.66,
                    td_share=0.1, rec_td_share=0.2, rush_td_share=0.2)
        for pid, pos in PLAYERS[team]
    ]


@pytest.fixture
def spec() -> NflGameSpec:
    """Sentinel pass/rush rates (1.0) so a learned replacement is visible."""
    return NflGameSpec(
        home_team="HOM", away_team="AWY",
        home=_rates(1.0, 1.0), away=_rates(1.0, 1.0),
        home_players=_players("HOM"), away_players=_players("AWY"),
    )
