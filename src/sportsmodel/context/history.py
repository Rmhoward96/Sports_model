"""Team history as of a game: rolling records, streaks, splits.

Built on the team game log (``context.game_log``). Everything is descriptive
and uses only the team's games that kicked off strictly before ``asof`` and
have a result (``su`` not None).

Definitions
-----------
* ``L5/L10/L20``: the team's last N games with a result, across seasons.
  Postseason games are included (controller ruling H1).
* ``season``: season-to-date. The season is the one passed via ``season=``;
  by default it is the season of the team's next game at/after ``asof`` (the
  upcoming game), falling back to the season of its latest prior game when
  nothing is scheduled at/after ``asof``.
* Records: SU ``"W-L"`` (``"-T"`` appended only when there are ties); ATS
  ``"W-L-P"`` and O/U ``"O-U-P"`` are counted only over games in the window
  that have that outcome, so ``n_ats``/``n_ou`` can be smaller than ``n``.
* ``avg_cover`` = mean(margin - team_line) over games with a team line;
  ``avg_total_vs_line`` = mean(pf + pa - total_line) over games with a total.
* Streaks: the current run of the same outcome, newest backwards. Ties (SU)
  and pushes (ATS, O/U) are outcomes that break a streak (a run of them is
  reported as ``T``/``P``). Games with no line have no ATS/O-U outcome and are
  skipped in those two streams (they neither extend nor break a streak).
  ``notes[k]`` = ``"U6 of 7"`` when at least 6 of the last 7 outcomes in that
  stream share one value.
* Splits (season): home (venue home), away (venue away; neutral in neither),
  fav, dog (role pick / no line in neither); each ``{su, ats, n}``.
* The pooled FCS pseudo-team (``team_is_fbs`` False) never gets a history:
  ``team_history_asof`` returns None and ``history_for_games`` skips it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = {"L5": 5, "L10": 10, "L20": 20}
_NOTE_LAST, _NOTE_MIN = 7, 6


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _f(x):
    """NaN/None -> None, else python float (keeps output JSON-able)."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else x


def _su(d: pd.DataFrame) -> str:
    w, l, t = (int((d["su"] == k).sum()) for k in ("W", "L", "T"))
    return f"{w}-{l}-{t}" if t else f"{w}-{l}"


def _ats(d: pd.DataFrame) -> str:
    return "-".join(str(int((d["ats"] == k).sum())) for k in ("W", "L", "P"))


def _ou(d: pd.DataFrame) -> str:
    return "-".join(str(int((d["ou"] == k).sum())) for k in ("O", "U", "P"))


def _window(d: pd.DataFrame) -> dict:
    lined = d["team_line"].notna()
    totl = d["total_line"].notna()
    cover = (d["margin"] - d["team_line"])[lined]
    tvl = (d["pf"] + d["pa"] - d["total_line"])[totl]
    return {
        "su": _su(d), "ats": _ats(d), "ou": _ou(d), "n": int(len(d)),
        "n_ats": int(d["ats"].notna().sum()), "n_ou": int(d["ou"].notna().sum()),
        "avg_margin": _f(d["margin"].mean()) if len(d) else None,
        "avg_cover": _f(cover.mean()) if len(cover) else None,
        "avg_total_vs_line": _f(tvl.mean()) if len(tvl) else None,
    }


def _split(d: pd.DataFrame) -> dict:
    return {"su": _su(d), "ats": _ats(d), "n": int(len(d))}


def _streak(outcomes: list[str]) -> tuple[str | None, str | None]:
    """(current run e.g. 'W3', 'U6 of 7' note) from oldest->newest outcomes."""
    if not outcomes:
        return None, None
    last = outcomes[-1]
    run = 0
    for o in reversed(outcomes):
        if o != last:
            break
        run += 1
    note = None
    if len(outcomes) >= _NOTE_LAST:
        tail = outcomes[-_NOTE_LAST:]
        for v in dict.fromkeys(tail):
            if tail.count(v) >= _NOTE_MIN:
                note = f"{v}{tail.count(v)} of {_NOTE_LAST}"
                break
    return f"{last}{run}", note


def _season_of(rows: pd.DataFrame, played: pd.DataFrame, asof: pd.Timestamp):
    upcoming = rows[rows["kickoff"] >= asof]
    if len(upcoming):
        return int(upcoming.sort_values("kickoff", kind="stable")["season"].iloc[0])
    if len(played):
        return int(played["season"].iloc[-1])
    return None


def team_history_asof(log: pd.DataFrame, team: str, asof,
                      season: int | None = None) -> dict | None:
    """History of ``team`` using only its games with a result and
    ``kickoff < asof``. Returns None for the pooled FCS pseudo-team."""
    rows = log[log["team"] == team]
    if len(rows) and not rows["team_is_fbs"].astype(bool).any():
        return None
    asof = _utc(asof)
    rows = rows.sort_values("kickoff", kind="stable")
    played = rows[(rows["kickoff"] < asof) & rows["su"].notna()]
    if season is None:
        season = _season_of(rows, played, asof)
    sea = played[played["season"] == season] if season is not None else played.iloc[0:0]

    windows = {name: _window(played.tail(n)) for name, n in WINDOWS.items()}
    windows["season"] = _window(sea)

    streaks: dict = {}
    notes: dict = {}
    for key in ("su", "ats", "ou"):
        seq = [o for o in played[key].tolist() if o is not None and o == o]
        streaks[key], note = _streak(seq)
        if note:
            notes[key] = note
    streaks["notes"] = notes

    splits = {
        "home": _split(sea[sea["venue"] == "home"]),
        "away": _split(sea[sea["venue"] == "away"]),
        "fav": _split(sea[sea["role"] == "fav"]),
        "dog": _split(sea[sea["role"] == "dog"]),
    }
    return {"team": team, "season": season, "windows": windows,
            "streaks": streaks, "splits": splits}


_ID_COLS = ["sport", "season", "week", "game_key", "kickoff", "team",
            "opponent", "venue"]


def history_for_games(log: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, side). ``games`` is in game-log format (e.g. the
    unplayed rows of the log): each row's history is taken as of its kickoff
    and its season. FCS pseudo-team rows are skipped."""
    cols = [c for c in _ID_COLS if c in games.columns] + ["windows", "streaks", "splits"]
    out = []
    for g in games.to_dict("records"):
        lg = log[log["sport"] == g["sport"]] if "sport" in g else log
        h = team_history_asof(lg, g["team"], g["kickoff"], season=g.get("season"))
        if h is None:
            continue
        out.append({**{c: g[c] for c in _ID_COLS if c in g},
                    "windows": h["windows"], "streaks": h["streaks"],
                    "splits": h["splits"]})
    return pd.DataFrame(out, columns=cols)
