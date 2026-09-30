"""CFB profit-model feature table.

`build_rows(raw, priors, fbs)` turns the shared walk-forward rows
(`sportsmodel.cfb.walkforward.raw_model_predictions`) into one row per
(game, market, price_point) where that price exists:

- spread  x {open (spread_open), close (market_spread)}
- total   x {open (total_open),  close (market_total)}
- moneyline x close only (CFBD gives median closing moneylines; no opener)

Spread convention: `line` is the home-margin line L (home favored => positive).
A home spread bet wins iff actual_margin > L; pushes are dropped (kept, unlabelled,
with keep_pushes=True for betting simulations). Total label is
the over; moneyline label is the home win (ties dropped).

Leakage: every f_* value uses only information before the game -- the
walk-forward's pre-game ratings, the priced line itself (and its opener),
static schedule context, preseason priors, and rest days computed from the
team's EARLIER games this season.

Priors audit (scripts/build_cfb_priors.py; Ruling L1): CFBD `/ratings/sp?year=S`
is the END-OF-SEASON SP+ for S, so same-season `sp_rating` is never a feature --
`f_sp_prev_*` uses the team's PREVIOUS season's final SP+ ((S-1, team) row).
`coach_first_year` for S is built from CFBD coach tenures that include in-season
interim coaches (a mid-season firing flags the school True for S), so it too
is taken from the previous season (`f_coach_new_prev_*`, known before S).
returning_pct / qb_returning (`/player/returning`, last season's production
returning), recruiting_points (the class signed before S) and portal_net
(transfers for season S) are preseason and used same-season. prior_sos is
preseason; forward_sos_shift uses same-season (end-of-season) SP+ and is NOT a
feature.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sportsmodel.cfb.teams import FCS

# --- conference tiers (ESPN conference ids, as they appear in
# assets/cfb/schedules.parquet home_conf/away_conf; verified against team names)
P4_CONF_IDS = frozenset({
    "1",   # ACC
    "4",   # Big 12
    "5",   # Big Ten
    "8",   # SEC
})
PAC12_CONF_ID = "9"            # Pac-12: P4 through 2023; the 2024+ remnant /
PAC12_P4_LAST_SEASON = 2023    # rebuilt Pac-12 is G5
G5_CONF_IDS = frozenset({
    "12",   # Conference USA
    "15",   # MAC
    "17",   # Mountain West
    "37",   # Sun Belt
    "151",  # American (AAC)
    "18",   # FBS Independents (G5 tier except P4_INDEPENDENT_TEAMS) -- incl. BYU
            # 2015-2022 (G5 then; Big 12 = P4 from 2023), Army, UConn, UMass,
            # Liberty 2018-22, NMSU
})
# Notre Dame (ESPN id 87) is an independent (conference 18; ACC in 2020) but
# is priced and scheduled as a power program, so it counts as P4 every season.
P4_INDEPENDENT_TEAMS = frozenset({"87"})
FBS_CONF_IDS = P4_CONF_IDS | G5_CONF_IDS | {PAC12_CONF_ID}

MARKETS = ("spread", "total", "moneyline")

# (feature stem, priors column, season offset) -- each yields
# f_<stem>_home/_away/_diff from the priors row for (season + offset, team).
# offset -1 = the previous season's value (see the priors audit above).
PRIOR_FEATURES = (
    ("sp_prev", "sp_rating", -1),
    ("ret", "returning_pct", 0),
    ("rec", "recruiting_points", 0),
    ("portal", "portal_net", 0),
    ("coach_new_prev", "coach_first_year", -1),
    ("qb_ret", "qb_returning", 0),
)
PRIOR_COLS = [f"f_{stem}_{side}" for stem, _, _ in PRIOR_FEATURES
              for side in ("home", "away", "diff")]

_COMMON = ["f_model_margin", "f_model_total", "f_elo_diff", "f_srs_diff", "f_edge_pts",
           "f_neutral", "f_conf_game", "f_week", "f_class", "f_rest_home", "f_rest_away",
           *PRIOR_COLS]
FEATURE_COLS: dict[str, list[str]] = {
    "spread": ["line", "f_move", *_COMMON],
    "total": ["line", "f_move", *_COMMON],
    # f_edge_pts == f_model_margin for moneyline -> dropped as a duplicate
    "moneyline": [c for c in _COMMON if c != "f_edge_pts"],
}

KEY_COLS = ["season", "week", "game_pk", "home_team", "away_team", "market", "price_point"]
_EXTRA_COLS = ["start_date", "actual_margin", "actual_total", "result"]
# `result` of the FIRST side (home covers / over / home wins): "win" (y=1),
# "loss" (y=0), "push" (kept only with keep_pushes=True; y NaN) or "void"
# (no result yet / never played; y NaN)
_RESULT = {1.0: "win", 0.0: "loss"}
_ALL_FEATURES = ["line", "f_move", *_COMMON]
COLUMNS = [*KEY_COLS, "line", "ml_home", "ml_away", *_ALL_FEATURES[1:], "y", *_EXTRA_COLS]


def _num(value) -> float:
    """Anything -> float, NaN when missing (None / NaN / pd.NA / non-numeric)."""
    if value is None:
        return math.nan
    try:
        if pd.isna(value):
            return math.nan
    except (TypeError, ValueError):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _conf(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s or None


def _tier(season: int, team: str, conf: str | None, fbs: set[str]) -> str | None:
    """'P4' / 'G5' / 'FCS', or None when an FBS team's conference is unknown."""
    if team == FCS or team not in fbs:
        return "FCS"
    if conf is None:
        return None
    if conf not in FBS_CONF_IDS:          # e.g. a current-FBS program in its FCS era
        return "FCS"
    if team in P4_INDEPENDENT_TEAMS:
        return "P4"
    if conf in P4_CONF_IDS or (conf == PAC12_CONF_ID and season <= PAC12_P4_LAST_SEASON):
        return "P4"
    return "G5"


def game_class(season, home, away, home_conf, away_conf, fbs: set[str]) -> float:
    """0 = both P4, 1 = P4 vs G5, 2 = both G5, 3 = FBS vs FCS; NaN if unknown."""
    th = _tier(int(season), str(home), _conf(home_conf), fbs)
    ta = _tier(int(season), str(away), _conf(away_conf), fbs)
    if th == "FCS" or ta == "FCS":
        return 3.0
    if th is None or ta is None:
        return math.nan
    n_p4 = (th == "P4") + (ta == "P4")
    return {2: 0.0, 1: 1.0, 0: 2.0}[n_p4]


def _kickoff(start_date) -> datetime | None:
    """Parse an ESPN UTC `start_date` ('2022-09-03T19:00Z'); None if missing/bad."""
    if start_date is None or (isinstance(start_date, float) and math.isnan(start_date)):
        return None
    try:
        return datetime.fromisoformat(str(start_date).replace("Z", "+00:00"))
    except ValueError:
        return None


def _rest_days(raw: list[dict]) -> list[tuple[float, float]]:
    """(rest_home, rest_away) per raw row: whole days between the US game dates
    (UTC kickoff - 8h, as generate_cfb does) of this game and the team's
    previous game this season. NaN for its first game, for the FCS
    pseudo-team, or when a date is missing. Only earlier games are used."""
    kick = [_kickoff(r.get("start_date")) for r in raw]
    order = sorted((i for i in range(len(raw)) if kick[i] is not None),
                   key=lambda i: (int(raw[i]["season"]), kick[i]))
    last: dict[tuple[int, str], object] = {}
    rest = [(math.nan, math.nan)] * len(raw)
    for i in order:
        r = raw[i]
        d = (kick[i] - timedelta(hours=8)).date()
        teams = (str(r["home_team"]), str(r["away_team"]))
        pair = []
        for team in teams:
            prev = last.get((int(r["season"]), team))
            pair.append(float((d - prev).days) if (prev is not None and team != FCS)
                        else math.nan)
        rest[i] = (pair[0], pair[1])
        for team in teams:
            last[(int(r["season"]), team)] = d
    return rest


def _priors_lookup(priors: pd.DataFrame) -> dict[tuple[int, str], dict]:
    if priors is None or priors.empty:
        return {}
    p = priors.drop_duplicates(subset=["season", "team_espn_id"], keep="first")
    return {(int(r["season"]), str(r["team_espn_id"])): r
            for r in p.to_dict("records")}


def _prior_feats(lookup: dict, season: int, home: str, away: str) -> dict:
    out = {}
    for stem, col, offset in PRIOR_FEATURES:
        ph = lookup.get((season + offset, home), {})
        pa = lookup.get((season + offset, away), {})
        h, a = _num(ph.get(col)), _num(pa.get(col))
        out[f"f_{stem}_home"] = h
        out[f"f_{stem}_away"] = a
        out[f"f_{stem}_diff"] = h - a
    return out


def _label(market: str, r: dict, line: float) -> float | None:
    """1/0 label, NaN when the actual is unknown, None for a push/tie (drop)."""
    if market == "total":
        actual = _num(r.get("actual_total"))
    else:
        actual = _num(r.get("actual_margin"))
    if math.isnan(actual):
        return math.nan
    ref = 0.0 if market == "moneyline" else line
    if actual == ref:
        return None
    return 1.0 if actual > ref else 0.0


def _result(y: float | None) -> str:
    if y is None:
        return "push"
    return _RESULT.get(y, "void")


def build_rows(raw: list[dict], priors: pd.DataFrame, fbs: set[str], *,
               keep_pushes: bool = False) -> pd.DataFrame:
    """Walk-forward rows -> the (game, market, price_point) feature table.

    Rows whose result pushes the priced line (or a tied moneyline) are dropped,
    unless keep_pushes=True: then they are kept with y = NaN (never trained or
    calibrated on) and result "push", so a betting simulation can settle them
    (stake returned) instead of conditioning the bet set on the final score.
    Unscored games (raw from `raw_model_predictions(..., include_unscored=True)`,
    actual_* None) are KEPT with y = NaN -- no label, no push drop -- so the
    live step prices them with the same code. Training consumers must drop
    y-NaN rows. Rest days count every raw game with a start_date (played or
    not), so pass include_unscored=True to count unplayed scheduled games too.
    Extra non-feature columns: start_date (slate day), actual_margin/total,
    result ("win"/"loss" of the first side, "push", "void" = no result).
    """
    lookup = _priors_lookup(priors)
    rests = _rest_days(raw)
    out = []
    for r, (rest_h, rest_a) in zip(raw, rests):
        season, home, away = int(r["season"]), str(r["home_team"]), str(r["away_team"])
        mm, mt = _num(r.get("model_margin")), _num(r.get("model_total"))
        common = {
            "f_model_margin": mm, "f_model_total": mt,
            "f_elo_diff": _num(r.get("elo_home")) - _num(r.get("elo_away")),
            "f_srs_diff": _num(r.get("srs_home")) - _num(r.get("srs_away")),
            "f_neutral": _num(r.get("neutral_site")),
            "f_conf_game": _num(r.get("conference_game")),
            "f_week": float(r["week"]),
            "f_class": game_class(season, home, away, r.get("home_conf"),
                                  r.get("away_conf"), fbs),
            "f_rest_home": rest_h, "f_rest_away": rest_a,
            **_prior_feats(lookup, season, home, away),
        }
        base = {"season": season, "week": int(r["week"]), "game_pk": r.get("game_pk"),
                "home_team": home, "away_team": away,
                "start_date": r.get("start_date"),
                "actual_margin": _num(r.get("actual_margin")),
                "actual_total": _num(r.get("actual_total"))}

        priced = []   # (market, price_point, line, opener, model value)
        for market, open_key, close_key, model in (
                ("spread", "spread_open", "market_spread", mm),
                ("total", "total_open", "market_total", mt)):
            opener = _num(r.get(open_key))
            close = _num(r.get(close_key))
            if not math.isnan(opener):
                priced.append((market, "open", opener, opener, model))
            if not math.isnan(close):
                priced.append((market, "close", close, opener, model))
        for market, pp, line, opener, model in priced:
            y = _label(market, r, line)
            if y is None and not keep_pushes:
                continue
            out.append({**base, "market": market, "price_point": pp, "line": line,
                        "ml_home": math.nan, "ml_away": math.nan,
                        "f_move": line - opener, "f_edge_pts": model - line,
                        **common, "y": math.nan if y is None else y,
                        "result": _result(y)})

        ml_h, ml_a = _num(r.get("ml_home")), _num(r.get("ml_away"))
        if not (math.isnan(ml_h) or math.isnan(ml_a)):
            y = _label("moneyline", r, 0.0)
            if y is not None or keep_pushes:
                out.append({**base, "market": "moneyline", "price_point": "close",
                            "line": math.nan, "ml_home": ml_h, "ml_away": ml_a,
                            "f_move": math.nan, "f_edge_pts": mm, **common,
                            "y": math.nan if y is None else y, "result": _result(y)})

    df = pd.DataFrame(out, columns=COLUMNS)
    float_cols = [c for c in COLUMNS if c.startswith("f_")] + [
        "line", "ml_home", "ml_away", "y", "actual_margin", "actual_total"]
    df[float_cols] = df[float_cols].astype(np.float64)
    return df
