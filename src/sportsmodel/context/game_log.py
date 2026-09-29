"""Team game log: one row per team per game (NFL + CFB), played or upcoming.

Line convention (see global constraints): ``team_line`` is the team's own
closing spread, + = favored by that many points. NFL ``spread_line`` and CFB
``market_spread`` are both home-margin (home-favored-positive), so the home
team gets ``+line`` and the away team ``-line``. A team covers iff
``margin > team_line``; equal is a push. Over iff ``pf + pa > total_line``.

Unplayed games are kept with every outcome column None/NaN so upcoming
matchups can be looked up; lines are kept when known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

COLUMNS = ["sport", "season", "week", "game_key", "kickoff", "date_et", "team",
           "opponent", "venue", "pf", "pa", "margin", "team_line", "total_line",
           "role", "su", "ats", "ou", "game_pk"]

_ET = "America/New_York"


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _side_rows(g: dict) -> list[dict]:
    """Expand one home-perspective game dict into its two team rows."""
    hs, as_ = _num(g["home_score"]), _num(g["away_score"])
    played = not (np.isnan(hs) or np.isnan(as_))
    spread = _num(g.get("home_line"))
    total = _num(g.get("total_line"))
    out = []
    for is_home in (True, False):
        pf, pa = (hs, as_) if is_home else (as_, hs)
        team, opp = ((g["home_team"], g["away_team"]) if is_home
                     else (g["away_team"], g["home_team"]))
        if g["neutral"]:
            venue = "neutral"
        else:
            venue = "home" if is_home else "away"
        line = float("nan") if np.isnan(spread) else (spread if is_home else -spread)
        margin = pf - pa if played else float("nan")
        if np.isnan(line):
            role = None
        elif line > 0:
            role = "fav"
        elif line < 0:
            role = "dog"
        else:
            role = "pick"
        if not played:
            su = ats = ou = None
        else:
            su = "W" if margin > 0 else ("L" if margin < 0 else "T")
            if np.isnan(line):
                ats = None
            else:
                ats = ("W" if margin > line else ("L" if margin < line else "P"))
            if np.isnan(total):
                ou = None
            else:
                pts = pf + pa
                ou = "O" if pts > total else ("U" if pts < total else "P")
        out.append({
            "sport": g["sport"], "season": g["season"], "week": g["week"],
            "game_key": g["game_key"], "kickoff": g["kickoff"],
            "date_et": g["date_et"], "team": team, "opponent": opp,
            "venue": venue, "pf": pf if played else float("nan"),
            "pa": pa if played else float("nan"), "margin": margin,
            "team_line": line, "total_line": total, "role": role, "su": su,
            "ats": ats, "ou": ou, "game_pk": g.get("game_pk"),
        })
    return out


def _finish(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("role", "su", "ats", "ou"):
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    if not df.empty:
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
        df = df.sort_values(["kickoff", "game_key", "venue"],
                            kind="stable").reset_index(drop=True)
    return df


def nfl_game_log(schedules: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game from nflverse schedules (REG + postseason)."""
    rows: list[dict] = []
    for r in schedules.itertuples(index=False):
        gameday = pd.Timestamp(r.gameday)
        gametime = getattr(r, "gametime", None)
        hhmm = gametime if isinstance(gametime, str) and ":" in gametime else "13:00"
        local = pd.Timestamp(f"{gameday.date()} {hhmm}").tz_localize(
            _ET, ambiguous=True, nonexistent="shift_forward")
        rows += _side_rows({
            "sport": "nfl", "season": int(r.season), "week": int(r.week),
            "game_key": str(r.game_id), "kickoff": local.tz_convert("UTC"),
            "date_et": str(gameday.date()),
            "home_team": normalize_team(r.home_team),
            "away_team": normalize_team(r.away_team),
            "home_score": r.home_score, "away_score": r.away_score,
            "neutral": str(getattr(r, "location", "Home")).lower() == "neutral",
            "home_line": r.spread_line, "total_line": r.total_line,
            "game_pk": None,
        })
    return _finish(rows)


def _cfb_merged(schedules: pd.DataFrame, lines: pd.DataFrame,
                live_close: pd.DataFrame | None) -> pd.DataFrame:
    keys = ["season", "week", "home_team", "away_team"]
    sched = schedules[schedules["home_team"] != schedules["away_team"]].copy()
    ln = lines[lines["home_team"] != lines["away_team"]].drop_duplicates(
        subset=keys, keep="first")
    m = sched.merge(ln[keys + ["market_spread", "market_total"]], on=keys,
                    how="left")
    if live_close is not None and len(live_close):
        lc = live_close.copy()
        lc["_k"] = lc["game_pk"].astype(str)
        lc = lc.drop_duplicates("_k", keep="last").set_index("_k")
        k = m["game_pk"].astype(str)
        m["market_spread"] = m["market_spread"].fillna(
            k.map(lc["close_spread_home"]))
        m["market_total"] = m["market_total"].fillna(k.map(lc["close_total"]))
    return m


def cfb_game_log(schedules: pd.DataFrame, lines: pd.DataFrame,
                 live_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Two rows per CFB game. ``lines`` (CFBD closing consensus) supplies the
    market; ``live_close`` (from ``live_closing_consensus``) fills games the
    lines asset lacks. Non-FBS opponents appear as team ``"FCS"``."""
    m = _cfb_merged(schedules, lines, live_close)
    rows: list[dict] = []
    for r in m.itertuples(index=False):
        kick = pd.Timestamp(r.start_date)
        kick = kick.tz_localize("UTC") if kick.tzinfo is None else kick.tz_convert("UTC")
        rows += _side_rows({
            "sport": "cfb", "season": int(r.season), "week": int(r.week),
            "game_key": str(r.game_pk), "kickoff": kick,
            "date_et": str(kick.tz_convert(_ET).date()),
            "home_team": str(r.home_team), "away_team": str(r.away_team),
            "home_score": r.home_score, "away_score": r.away_score,
            "neutral": bool(r.neutral_site),
            "home_line": r.market_spread, "total_line": r.market_total,
            "game_pk": r.game_pk,
        })
    return _finish(rows)


def live_closing_consensus(odds_rows: pd.DataFrame) -> pd.DataFrame:
    """Per game_pk closing consensus from ``odds_snapshot`` rows.

    Each book's last capture at/before kickoff, converted to a home-margin
    spread (book convention: home -7 means home favored by 7, so
    ``close_spread_home = -home_line``, or ``+away_line`` when only the away
    side was captured), then the median across books. Totals likewise.
    Games with no pre-kickoff capture are omitted.
    """
    cols = ["game_pk", "close_spread_home", "close_total"]
    if odds_rows is None or odds_rows.empty:
        return pd.DataFrame(columns=cols)
    d = odds_rows[odds_rows["market"].isin(["spread", "total"])].copy()
    d["captured_at"] = pd.to_datetime(d["captured_at"], utc=True)
    d["commence_time"] = pd.to_datetime(d["commence_time"], utc=True)
    d = d[(d["captured_at"] <= d["commence_time"]) & d["line"].notna()]
    if d.empty:
        return pd.DataFrame(columns=cols)
    is_spread = d["market"] == "spread"
    line = d["line"].astype(float)
    d["val"] = np.where(is_spread & (d["side"] == "home"), -line, line)
    d = d[~(is_spread & ~d["side"].isin(["home", "away"]))]
    d["prio"] = d["side"].isin(["home", "over"]).astype(int)
    last = (d.sort_values(["captured_at", "prio"], kind="stable")
              .drop_duplicates(["game_pk", "book", "market"], keep="last"))
    med = last.groupby(["game_pk", "market"])["val"].median().unstack("market")
    out = pd.DataFrame({
        "game_pk": med.index,
        "close_spread_home": med["spread"].to_numpy() if "spread" in med else np.nan,
        "close_total": med["total"].to_numpy() if "total" in med else np.nan,
    }).reset_index(drop=True)
    return out
