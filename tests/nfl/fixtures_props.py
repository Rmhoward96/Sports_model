"""Synthetic 4-team x 2-season x 4-week league for the props-ML feature tests.

`feature_inputs(perturb_from=(S, w))` multiplies every box-score stat at or
after (S, w) by 10 -- in `pg`, `tg`, `rz`, `ngs` and `game_epa`. NGS week-0
rows are season aggregates (they contain every game of that season), so a
week-0 row of season >= S counts as "at or after". Pre-game inputs (`ctx`,
`injuries`, `depth`, `stubs`) are never touched.

Scripted events (2024 week 3 is the perturbation target):
- KC_WR and BAL_QB are Out in 2024 wk3 (no game row); BAL_QB2 starts in his
  place (depth QB1 that week only). MIA_TE is Questionable in 2024 wk3.
- BUF_WR is Doubtful in 2024 wk3 but plays (his own wk3 game must not feed
  his teammates' vacated share).
- KC_RB2 plays every 2023 game, then is an active-but-no-snap stub all 2024.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.efficiency import team_game_epa

TEAMS = ("KC", "BAL", "BUF", "MIA")
SEASONS = (2023, 2024)
WEEKS = (1, 2, 3, 4)
_PAIRS = {1: [("KC", "BAL"), ("BUF", "MIA")], 2: [("KC", "BUF"), ("BAL", "MIA")],
          3: [("KC", "MIA"), ("BAL", "BUF")], 4: [("BAL", "KC"), ("MIA", "BUF")]}
_STARTERS = ("QB", "RB", "WR", "TE")
_OUT = {(2024, 3): ["KC_WR", "BAL_QB"]}

PG_STATS = ["snap_pct", "y_targets", "y_carries", "y_pass_att", "y_receptions", "y_rec_yds", "y_rush_yds",
            "y_pass_yds", "y_pass_tds", "y_anytime_td", "rec_air_yards", "yac", "target_share", "air_yards_share"]
TG_STATS = ["pass_att", "rush_att", "plays", "dropbacks", "pressures_allowed", "neutral_pass_rate"]
RZ_STATS = ["rz_targets", "rz_carries", "gl_carries"]
NGS_STATS = {
    "rec": ["avg_separation", "avg_cushion", "avg_intended_air_yards", "avg_yac_above_expectation"],
    "rush": ["rush_yards_over_expected_per_att", "efficiency", "percent_attempts_gte_eight_defenders"],
    "pass": ["avg_time_to_throw", "completion_percentage_above_expectation", "aggressiveness"],
}
_NGS_POS = {"rec": ("WR", "TE"), "rush": ("RB",), "pass": ("QB",)}


def _games() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        for w in WEEKS:
            for home, away in _PAIRS[w]:
                rows += [(s, w, home, away, 1), (s, w, away, home, 0)]
    return pd.DataFrame(rows, columns=["season", "week", "team", "opponent", "is_home"])


def _players_in(team: str, season: int, week: int) -> list[tuple[str, str]]:
    ps = [(f"{team}_{p}", p) for p in _STARTERS if f"{team}_{p}" not in _OUT.get((season, week), [])]
    if team == "BAL" and (season, week) == (2024, 3):
        ps.append(("BAL_QB2", "QB"))
    if team == "KC" and season == 2023:
        ps.append(("KC_RB2", "RB"))
    return ps


def _player_games(rng, games) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        for pid, pos in _players_in(g.team, g.season, g.week):
            tg = int(rng.integers(3, 10)) if pos in ("WR", "TE") else int(rng.integers(0, 5)) if pos == "RB" else 0
            rec = int(rng.integers(1, tg + 1)) if tg else 0
            car = int(rng.integers(8, 20)) if pos == "RB" else int(rng.integers(1, 6)) if pos == "QB" else 0
            att = int(rng.integers(25, 40)) if pos == "QB" else 0
            rows.append({
                "player_id": pid, "season": g.season, "week": g.week, "team": g.team, "opponent": g.opponent,
                "position": pos, "snap_pct": float(rng.uniform(0.3, 1.0)),
                "y_targets": float(tg), "y_carries": float(car), "y_pass_att": float(att),
                "y_receptions": float(rec), "y_rec_yds": float(rec * rng.uniform(5, 15)),
                "y_rush_yds": float(car * rng.uniform(2, 6)), "y_pass_yds": float(att * rng.uniform(5, 9)),
                "y_pass_tds": float(rng.integers(0, 4)) if att else 0.0, "y_anytime_td": float(rng.integers(0, 2)),
                "rec_air_yards": float(tg * rng.uniform(3, 12)), "yac": float(rec * rng.uniform(1, 6)),
                "target_share": float(tg / 30), "air_yards_share": float(rng.uniform(0, 0.4)) if tg else 0.0,
            })
    return pd.DataFrame(rows)


def _team_games(rng, games) -> pd.DataFrame:
    t = games[["season", "week", "team", "opponent"]].copy()
    n = len(t)
    t["pass_att"] = rng.integers(25, 45, n)
    t["rush_att"] = rng.integers(18, 35, n)
    t["plays"] = t["pass_att"] + t["rush_att"] + rng.integers(0, 6, n)
    t["dropbacks"] = t["pass_att"] + rng.integers(1, 5, n)
    t["pressures_allowed"] = rng.integers(3, 14, n)
    t["neutral_pass_rate"] = rng.uniform(0.4, 0.7, n)
    return t


def _redzone(rng, pg) -> pd.DataFrame:
    r = pg[pg["position"].isin(["RB", "WR", "TE"])][["player_id", "season", "week"]].copy()
    n = len(r)
    r["rz_targets"], r["rz_carries"], r["gl_carries"] = (
        rng.integers(0, 3, n).astype(float), rng.integers(0, 4, n).astype(float), rng.integers(0, 2, n).astype(float))
    return r[r[RZ_STATS].sum(axis=1) > 0].reset_index(drop=True)   # like player_redzone: only rows with touches


def _ngs(rng, pg) -> dict[str, pd.DataFrame]:
    out = {}
    for kind, cols in NGS_STATS.items():
        p = pg[pg["position"].isin(_NGS_POS[kind])][["player_id", "season", "week"]]
        p = p.rename(columns={"player_id": "player_gsis_id"})
        agg = p.drop_duplicates(["player_gsis_id", "season"]).assign(week=0)   # week-0 season aggregates
        f = pd.concat([p, agg], ignore_index=True).assign(season_type="REG")
        for c in cols:
            f[c] = rng.uniform(1.0, 5.0, len(f))
        out[kind] = f
    return out


def _context(rng, games) -> pd.DataFrame:
    c = games.copy()
    n = len(c)
    c["stadium_id"] = "STAD" + c["team"]
    c["cx_rest"] = rng.choice([6.0, 7.0, 10.0], n)
    c["cx_indoor"] = rng.integers(0, 2, n).astype(float)
    c["mk_spread"] = rng.normal(0, 4, n).round(1)
    c["mk_total"] = rng.uniform(38, 52, n).round(1)
    c["mk_implied"] = (c["mk_total"] + c["mk_spread"]) / 2
    return c


def _injuries() -> pd.DataFrame:
    return pd.DataFrame({
        "season": [2024, 2024, 2024, 2024, 2023, 2024],
        "week": [3, 3, 3, 3, 2, 3],
        "team": ["KC", "BAL", "MIA", "BUF", "KC", "BUF"],
        "gsis_id": ["KC_WR", "BAL_QB", "MIA_TE", "BUF_RB", "KC_TE", "BUF_WR"],
        "full_name": ["a", "b", "c", "d", "e", "f"],
        "report_status": ["Out", "Out", "Questionable", np.nan, "Doubtful", "Doubtful"],
    })


def _depth(games) -> pd.DataFrame:
    rows = []
    for g in games.itertuples(index=False):
        extra = [("KC_RB2", "RB")] if g.team == "KC" else [("BAL_QB2", "QB")] if g.team == "BAL" else []
        for pid, pos in [(f"{g.team}_{p}", p) for p in _STARTERS] + extra:
            rank = 2 if pid in ("KC_RB2", "BAL_QB2") else 1
            if (g.season, g.week) == (2024, 3) and pid in ("BAL_QB", "BAL_QB2"):
                rank = 3 - rank                                   # backup starts wk3
            rows.append({"season": g.season, "week": g.week, "club_code": g.team, "depth_team": float(rank),
                         "position": pos, "gsis_id": pid, "full_name": pid, "football_name": pid})
    return pd.DataFrame(rows)


def _game_epa(rng, games) -> dict:
    rows = []
    for g in games.itertuples(index=False):
        for _ in range(3):
            rows.append({"season": g.season, "week": g.week, "posteam": g.team, "defteam": g.opponent,
                         "epa": float(rng.normal(0, 0.5))})
    return team_game_epa(pd.DataFrame(rows))


def _stubs(games) -> pd.DataFrame:
    kc = games[(games["team"] == "KC") & (games["season"] == 2024)]
    return kc[["season", "week", "team", "opponent"]].assign(player_id="KC_RB2", position="RB")


def _at_or_after(df: pd.DataFrame, s: int, w: int, *, week0_is_season: bool = False) -> pd.Series:
    later = (df["week"] >= w) | ((df["week"] == 0) if week0_is_season else False)
    return (df["season"] > s) | ((df["season"] == s) & later)


def feature_inputs(perturb_from: tuple[int, int] | None = None, offset: float = 0.0) -> dict:
    """`offset` (default 0) is added after the x10 so per-game ratios (yds/target,
    carries/team rushes, ...) change too -- a pure x10 leaves them invariant."""
    rng = np.random.default_rng(0)
    games = _games()
    pg = _player_games(rng, games)
    tg = _team_games(rng, games)
    rz = _redzone(rng, pg)
    ngs = _ngs(rng, pg)
    ctx = _context(rng, games)
    game_epa = _game_epa(rng, games)
    tg[TG_STATS] = tg[TG_STATS].astype(float)       # same dtypes with or without perturbation
    if perturb_from is not None:
        s, w = perturb_from
        for df, cols in ((pg, PG_STATS), (tg, TG_STATS), (rz, RZ_STATS)):
            m = _at_or_after(df, s, w)
            df.loc[m, cols] = df.loc[m, cols] * 10 + offset
        for kind, df in ngs.items():
            m = _at_or_after(df, s, w, week0_is_season=True)
            df.loc[m, NGS_STATS[kind]] = df.loc[m, NGS_STATS[kind]] * 10 + offset
        game_epa = {k: ({**v, "off": None if v["off"] is None else v["off"] * 10 + offset,
                         "def": None if v["def"] is None else v["def"] * 10 + offset} if k[:2] >= (s, w) else v)
                    for k, v in game_epa.items()}
    return {"pg": pg, "tg": tg, "rz": rz, "ctx": ctx, "ngs": ngs, "injuries": _injuries(),
            "depth": _depth(games), "game_epa": game_epa, "stubs": _stubs(games)}
