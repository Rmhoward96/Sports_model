"""Opponent-adjusted EPA efficiency features for the NFL cover/total ensemble.

Pure functions: pandas frames/dicts in, dicts out, no IO. `team_game_epa`
mirrors the conventions in `sportsmodel.nfl.epa` (offense = mean `epa`
where the team is `posteam`; defense = mean `epa` where the team is
`defteam`, so lower is a better defense) but at (season, week, team)
granularity so `adjusted_efficiency` can build leakage-free, as-of-week
opponent-adjusted ratings for the cover/total models.
"""

from __future__ import annotations

import pandas as pd


def team_game_epa(pbp: pd.DataFrame) -> dict[tuple[int, int, str], dict]:
    """Aggregate per-(season, week, team) offensive/defensive EPA.

    Pure: DataFrame in, dict out, no IO. Only rows with non-null
    `posteam`, `defteam`, and `epa` participate, so every retained row
    identifies both teams in that matchup.

    For a given `(season, week, team)` key:
      - `off`: mean `epa` over rows where `team` is `posteam` (the team's
        own offensive efficiency that game). `None` if `team` had no
        offensive rows that (season, week).
      - `def`: mean `epa` over rows where `team` is `defteam` -- i.e. the
        opponent's offensive epa against `team`, which is `team`'s
        defensive efficiency that game (lower is a better defense).
        `None` if `team` had no defensive rows that (season, week).
      - `n`: number of offensive plays backing `off` (0 if `off` is
        `None`).
      - `opp`: the opponent faced that (season, week), read directly off
        the paired `posteam`/`defteam` columns (a team plays exactly one
        opponent per week). Kept so `adjusted_efficiency` can do a real
        opponent adjustment without re-deriving matchups from raw pbp.

    Returns a dict over the union of (season, week, team) keys appearing
    on offense or defense.
    """
    valid = pbp[
        pd.notna(pbp["epa"]) & pd.notna(pbp["posteam"]) & pd.notna(pbp["defteam"])
    ]

    off_group = valid.groupby(["season", "week", "posteam"])
    off_mean = off_group["epa"].mean()
    off_count = off_group["epa"].size()
    off_opp = off_group["defteam"].first()

    def_group = valid.groupby(["season", "week", "defteam"])
    def_mean = def_group["epa"].mean()
    def_opp = def_group["posteam"].first()

    keys = set(off_mean.index) | set(def_mean.index)

    result: dict[tuple[int, int, str], dict] = {}
    for season, week, team in keys:
        idx = (season, week, team)
        has_off = idx in off_mean.index
        has_def = idx in def_mean.index

        opp = off_opp.loc[idx] if has_off else None
        if opp is None and has_def:
            opp = def_opp.loc[idx]

        result[(int(season), int(week), team)] = {
            # Rounded to avoid float-summation noise (e.g. mean(0.2, 0.4)
            # landing on 0.30000000000000004 instead of 0.3) at a
            # precision far finer than an epa-per-play value needs.
            "off": round(float(off_mean.loc[idx]), 9) if has_off else None,
            "def": round(float(def_mean.loc[idx]), 9) if has_def else None,
            "n": int(off_count.loc[idx]) if has_off else 0,
            "opp": opp,
        }
    return result


def adjusted_efficiency(
    game_epa: dict[tuple[int, int, str], dict], season: int, upto_week: int
) -> dict[str, dict]:
    """Opponent-adjusted off/def EPA ratings for `season`, as of `upto_week`.

    Leakage-free (Global Constraint): only entries with
    `key[0] == season and key[1] < upto_week` are used -- the target week
    (and any later week) never contributes, mirroring the per-season
    isolation in `epa.team_epa_by_season`.

    Method -- a single opponent-adjustment pass, kept deliberately simple
    (no iterative re-solving of ratings) so it stays stable on the small
    samples seen early in a season:
      1. For each team, take the plain (unweighted) mean of its `off` /
         `def` game values within the window -> `raw_off` / `raw_def`.
      2. `league_off` / `league_def` are the mean of those raw team
         values across the league (0.0 if the window is empty -- the
         guard against dividing by zero when there is no prior data).
      3. For each game a team played, adjust its raw `off` value by
         subtracting the opponent's defensive deviation from league
         average (`raw_def[opp] - league_def`): facing a
         tougher-than-average defense raises the adjusted value, facing a
         weaker one lowers it. `off_adj` is the mean of these per-game
         adjusted values over the team's games. `def_adj` is computed
         symmetrically, subtracting the opponent's offensive deviation
         from league average.
      4. A team with no offensive/defensive games in the window, or whose
         opponent in a given game is absent from the window, falls back
         to the league average for the missing piece (0.0 league-average
         when nothing at all is known -- the empty-input guard).
    """
    window = {
        key: value
        for key, value in game_epa.items()
        if key[0] == season and key[1] < upto_week
    }

    off_games: dict[str, list[tuple[float, str | None]]] = {}
    def_games: dict[str, list[tuple[float, str | None]]] = {}
    for (_, _, team), value in window.items():
        opp = value.get("opp")
        if value.get("off") is not None:
            off_games.setdefault(team, []).append((value["off"], opp))
        if value.get("def") is not None:
            def_games.setdefault(team, []).append((value["def"], opp))

    raw_off = {
        team: sum(v for v, _ in games) / len(games)
        for team, games in off_games.items()
    }
    raw_def = {
        team: sum(v for v, _ in games) / len(games)
        for team, games in def_games.items()
    }

    league_off = sum(raw_off.values()) / len(raw_off) if raw_off else 0.0
    league_def = sum(raw_def.values()) / len(raw_def) if raw_def else 0.0

    teams = set(off_games) | set(def_games)
    result: dict[str, dict] = {}
    for team in teams:
        o_games = off_games.get(team, [])
        if o_games:
            off_adj = sum(
                v - (raw_def.get(opp, league_def) - league_def) for v, opp in o_games
            ) / len(o_games)
        else:
            off_adj = league_off

        d_games = def_games.get(team, [])
        if d_games:
            def_adj = sum(
                v - (raw_off.get(opp, league_off) - league_off) for v, opp in d_games
            ) / len(d_games)
        else:
            def_adj = league_def

        result[team] = {"off_adj": off_adj, "def_adj": def_adj}

    return result


def efficiency_features(adj: dict[str, dict], home: str, away: str) -> dict[str, float]:
    """Build home/away efficiency features for the spread/total ensemble
    from opponent-adjusted ratings produced by `adjusted_efficiency`.

    `def_adj` is opponent-adjusted epa allowed, so lower is a better
    defense; a team's net rating is `off_adj - def_adj`. `eff_diff` is the
    home team's net rating minus the away team's -- positive values favor
    the home team. `total_off` sums both teams' offensive ratings and
    feeds the total-points side of the ensemble.
    """
    home_off_adj = adj[home]["off_adj"]
    home_def_adj = adj[home]["def_adj"]
    away_off_adj = adj[away]["off_adj"]
    away_def_adj = adj[away]["def_adj"]

    home_net = home_off_adj - home_def_adj
    away_net = away_off_adj - away_def_adj

    return {
        "eff_diff": home_net - away_net,
        "home_off_adj": home_off_adj,
        "home_def_adj": home_def_adj,
        "away_off_adj": away_off_adj,
        "away_def_adj": away_def_adj,
        "total_off": home_off_adj + away_off_adj,
    }
