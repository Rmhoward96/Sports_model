"""Matchup grades: an offense unit vs the opponent's defense unit (A-F). PURE.

Descriptive only -- not a pick.

Components (per side, per game), from ``units.unit_ratings_asof`` rows:
``pass_epa = off_pass_epa[offense] + def_pass_epa[defense]`` (both league-relative, +
= good for the offense) and likewise for success and explosiveness, pass and run.

Scores: each component is divided by its scale -- the std of that component over the
grading history (all team-games of the previous three complete seasons), so EPA,
success rate and explosiveness are on one footing -- then combined 0.6 EPA / 0.25
success / 0.15 explosiveness into ``pass`` and ``run``; ``overall`` = pass_rate x pass +
(1 - pass_rate) x run with the offense's blended season pass rate.

Grades: percentiles of the history distribution of the same score (overall, pass and
run separately), with each history game scored from ratings as of that game
(walk-forward). Frozen per season: the table for season S uses seasons S-3..S-1 only.
A >= p90, B p70-90, C p30-70, D p10-30, F < p10. ``early`` when either team has played
< 3 games this season. A side whose offense or defense has no rating (CFB FCS
opponents: no advanced stats) or that is flagged non-FBS gets no scores and no grades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .units import unit_ratings_asof, window_ratings

WEIGHTS = {"epa": 0.6, "success": 0.25, "explosive": 0.15}
UNITS = ("pass", "run")
COMPONENTS: tuple[str, ...] = tuple(f"{u}_{m}" for u in UNITS for m in WEIGHTS)
KINDS = ("overall", "pass", "run")
BANDS = ((90, "A"), (70, "B"), (30, "C"), (10, "D"))
EARLY_GAMES = 3
HISTORY_SEASONS = 3
DEFAULT_PASS_RATE = 0.5


def _f(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(x) else x


def components(off_row, def_row) -> dict:
    """Raw (unscaled) matchup components + the offense's pass rate."""
    out = {c: float(off_row[f"off_{c}"]) + float(def_row[f"def_{c}"]) for c in COMPONENTS}
    pr = _f(off_row.get("pass_rate") if hasattr(off_row, "get") else off_row["pass_rate"])
    out["pass_rate"] = DEFAULT_PASS_RATE if pr is None else pr
    return out


def _combine(comp, scales: dict | None) -> dict:
    sc = scales or {}
    out = {}
    for u in UNITS:
        out[u] = sum(w * comp[f"{u}_{m}"] / sc.get(f"{u}_{m}", 1.0) for m, w in WEIGHTS.items())
    pr = comp["pass_rate"]
    out["overall"] = pr * out["pass"] + (1 - pr) * out["run"]
    return out


def side_scores(off_team_row, def_team_row, scales: dict | None = None) -> dict:
    """pass / run / overall scores for one side. ``scales`` (from ``grade_table``)
    standardizes each component; ``None`` = raw units (scale 1)."""
    comp = components(off_team_row, def_team_row)
    return {**_combine(comp, scales), "pass_rate": comp["pass_rate"]}


def score_frame(comp: pd.DataFrame, scales: dict | None) -> pd.DataFrame:
    """Vectorized ``_combine`` over a frame of components."""
    c = comp.copy()
    c["pass_rate"] = c["pass_rate"].fillna(DEFAULT_PASS_RATE) if "pass_rate" in c else DEFAULT_PASS_RATE
    return pd.DataFrame(_combine(c, scales), index=comp.index)[list(KINDS)]


def _component_frame(pairs: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """``pairs`` (team = offense, opponent = defense) -> components; NaN when unrated."""
    r = ratings.set_index("team")
    o = r.reindex(pairs["team"].to_numpy())
    d = r.reindex(pairs["opponent"].to_numpy())
    out = pd.DataFrame(index=pairs.index)
    for c in COMPONENTS:
        out[c] = o[f"off_{c}"].to_numpy() + d[f"def_{c}"].to_numpy()
    out["pass_rate"] = o["pass_rate"].to_numpy()
    out["games"] = o["games"].to_numpy()
    out["opp_games"] = d["games"].to_numpy()
    return out


def matchup_history(unit_games: pd.DataFrame, seasons) -> pd.DataFrame:
    """Components for every team-game of ``seasons``, each from ratings as of its week."""
    frames = []
    for s in sorted(int(x) for x in seasons):
        rows = unit_games[unit_games["season"] == s]
        if rows.empty:
            continue
        prev = window_ratings(unit_games, s - 1, 99)
        for wk in sorted(rows["week"].unique()):
            ratings = unit_ratings_asof(unit_games, s, int(wk), prev=prev)
            g = rows[rows["week"] == wk][["season", "week", "game_id", "team", "opponent"]]
            frames.append(pd.concat([g, _component_frame(g, ratings)], axis=1))
    if not frames:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "opponent",
                                     *COMPONENTS, "pass_rate", "games", "opp_games"])
    h = pd.concat(frames, ignore_index=True)
    return h.dropna(subset=list(COMPONENTS)).reset_index(drop=True)


def grade_table(history_scores: pd.DataFrame) -> dict | None:
    """Scales + percentile cutoffs from a ``matchup_history`` frame (None if empty)."""
    h = history_scores.dropna(subset=list(COMPONENTS))
    if h.empty:
        return None
    scales = {}
    for c in COMPONENTS:
        sd = float(h[c].std(ddof=0))
        scales[c] = sd if np.isfinite(sd) and sd > 0 else 1.0
    sc = score_frame(h, scales)
    grid = np.arange(101)
    cutoffs, quantiles = {}, {}
    for k in KINDS:
        q = np.percentile(sc[k].to_numpy(), grid)
        quantiles[k] = [float(v) for v in q]
        cutoffs[k] = {f"p{p}": float(q[p]) for p in (10, 30, 70, 90)}
    seasons = sorted(int(s) for s in h["season"].unique()) if "season" in h else []
    return {"seasons": seasons, "n": int(len(h)), "scales": scales,
            "cutoffs": cutoffs, "quantiles": quantiles}


def cutoffs_for_season(unit_games: pd.DataFrame, season: int,
                       n_seasons: int = HISTORY_SEASONS) -> dict | None:
    """The frozen grade table for ``season``: seasons season-n..season-1 only."""
    return grade_table(matchup_history(unit_games, range(season - n_seasons, season)))


def letter(score, cut: dict) -> str | None:
    s = _f(score)
    if s is None or not cut:
        return None
    for p, g in BANDS:
        if s >= cut[f"p{p}"]:
            return g
    return "F"


def _pct(score, q) -> float | None:
    s = _f(score)
    if s is None or not q:
        return None
    return float(np.interp(s, q, np.arange(101)))


def _flag_false(v) -> bool:
    """True only for an explicit False flag (python/numpy bool); None/NaN = unknown."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return isinstance(v, (bool, np.bool_)) and not bool(v)


def _r4(x):
    x = _f(x)
    return None if x is None else round(x, 4)


def _units_json(o, d) -> dict:
    return {
        "games": None if o is None else int(o["games"]),
        "opp_games": None if d is None else int(d["games"]),
        "pass_rate": None if o is None else _r4(o["pass_rate"]),
        "off": None if o is None else {c: _r4(o[f"off_{c}"]) for c in COMPONENTS},
        "def": None if d is None else {c: _r4(d[f"def_{c}"]) for c in COMPONENTS},
    }


def grades_for_games(games: pd.DataFrame, ratings: pd.DataFrame, cutoffs: dict | None) -> pd.DataFrame:
    """Per (game, side) scores + letters.

    ``games``: one row per game with ``game_pk``, ``home_team``, ``away_team`` (optional
    ``home_is_fbs`` / ``away_is_fbs``; other columns such as season/week pass through).
    ``ratings``: one ``unit_ratings_asof`` table. ``cutoffs``: ``cutoffs_for_season``
    (None -> units only, no scores/grades). ``side`` = the offense's side (home/away).
    """
    r = ratings.set_index("team")
    extra = [c for c in games.columns if c not in
             ("game_pk", "home_team", "away_team", "home_is_fbs", "away_is_fbs")]
    rows = []
    for g in games.to_dict("records"):
        for side, other in (("home", "away"), ("away", "home")):
            team, opp = g[f"{side}_team"], g[f"{other}_team"]
            fbs = not any(_flag_false(g.get(f"{x}_is_fbs")) for x in (side, other))
            o = r.loc[team] if team in r.index else None
            d = r.loc[opp] if opp in r.index else None
            row = {"game_pk": g["game_pk"], "side": side, "team": team, "opponent": opp,
                   **{c: g[c] for c in extra}}
            gm = [x["games"] for x in (o, d) if x is not None]
            row["early"] = bool(len(gm) < 2 or min(gm) < EARLY_GAMES)
            ok = fbs and o is not None and d is not None and cutoffs is not None
            sc = side_scores(o, d, cutoffs["scales"]) if ok else {}
            for k in KINDS:
                row[f"{k}_score"] = _r4(sc.get(k)) if ok else None
                row[k] = letter(sc.get(k), cutoffs["cutoffs"][k]) if ok else None
                row[f"{k}_pct"] = (round(_pct(sc.get(k), cutoffs["quantiles"][k]), 1)
                                   if ok else None)
            row["units"] = _units_json(o, d) if (fbs and (o is not None or d is not None)) else None
            rows.append(row)
    cols = ["game_pk", "side", "team", "opponent", *extra, *KINDS,
            *(f"{k}_score" for k in KINDS), *(f"{k}_pct" for k in KINDS), "early", "units"]
    return pd.DataFrame(rows, columns=cols)
