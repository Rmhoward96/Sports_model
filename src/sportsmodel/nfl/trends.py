"""NFL situational betting trends, computed from nflverse schedules.

For each upcoming game and each team: which situations apply THIS week (home/
road, favorite/underdog, off a road/home game, off a win/loss, off a bye,
division game, primetime), and the team's ATS (W-L-P) and O/U (O-U-P) record in
past games that were in the same situation, over the current season + last 3.
Descriptive only -- the model does not use these. PURE (DataFrames in, dicts out).

nflverse conventions: spread_line > 0 means the HOME team was favored by that
many; result = home_score - away_score; gametime is Eastern "HH:MM".
"""
from __future__ import annotations

from datetime import date

import pandas as pd

NFL_DIVISIONS = {
    **dict.fromkeys(["BUF", "MIA", "NE", "NYJ"], "AFC East"),
    **dict.fromkeys(["BAL", "CIN", "CLE", "PIT"], "AFC North"),
    **dict.fromkeys(["HOU", "IND", "JAX", "TEN"], "AFC South"),
    **dict.fromkeys(["DEN", "KC", "LV", "LAC"], "AFC West"),
    **dict.fromkeys(["DAL", "NYG", "PHI", "WAS"], "NFC East"),
    **dict.fromkeys(["CHI", "DET", "GB", "MIN"], "NFC North"),
    **dict.fromkeys(["ATL", "CAR", "NO", "TB"], "NFC South"),
    **dict.fromkeys(["ARI", "LA", "SF", "SEA"], "NFC West"),
}
SITUATION_LABELS = {
    "home": "at home", "road": "on the road", "favorite": "as a favorite", "underdog": "as an underdog",
    "off_road": "off a road game", "off_home": "off a home game", "off_win": "off a win",
    "off_loss": "off a loss", "off_bye": "off a bye", "division": "in division games",
    "primetime": "in primetime",
}
BYE_DAYS = 13
PRIMETIME_ET = "19:00"


def _vs(x: float) -> str:
    return "W" if x > 0 else "L" if x < 0 else "P"


def team_game_log(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) for completed games: team, opp, season, gameday,
    gametime, is_home, team_line (the team's own spread: home -spread_line),
    su (W/L/P), ats (W/L/P or None without a line), ou (O/U/P or None), sorted
    by team then date."""
    done = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    rows = []
    for g in done.itertuples(index=False):
        margin_home = float(g.home_score) - float(g.away_score)
        has_line = pd.notna(g.spread_line)
        has_total = pd.notna(g.total_line)
        total = float(g.home_score) + float(g.away_score)
        ou = None if not has_total else ("O" if total > g.total_line else "U" if total < g.total_line else "P")
        for team, opp, is_home, margin in ((g.home_team, g.away_team, True, margin_home),
                                           (g.away_team, g.home_team, False, -margin_home)):
            line = None if not has_line else (-float(g.spread_line) if is_home else float(g.spread_line))
            rows.append({"team": team, "opp": opp, "season": int(g.season), "gameday": str(g.gameday),
                         "gametime": str(g.gametime or ""), "is_home": is_home, "team_line": line,
                         "su": _vs(margin), "ats": None if line is None else _vs(margin + line), "ou": ou})
    log = pd.DataFrame(rows)
    return log.sort_values(["team", "gameday"]).reset_index(drop=True) if len(log) else log


def situations_for(team, is_home, team_line, kickoff_et, log_before, *, opp, season, gameday) -> list[str]:
    """Situations that apply to `team` for one game. `log_before` = the team's
    completed games strictly before this one (team_game_log rows, sorted).
    team_line: the team's own spread for this game (negative = favored; None or
    0 -> neither favorite nor underdog). kickoff_et: "YYYY-MM-DD HH:MM" Eastern."""
    s = ["home" if is_home else "road"]
    if team_line is not None and team_line != 0:
        s.append("favorite" if team_line < 0 else "underdog")
    if len(log_before):
        prev = log_before.iloc[-1]
        s.append("off_home" if prev["is_home"] else "off_road")
        if prev["su"] in ("W", "L"):
            s.append("off_win" if prev["su"] == "W" else "off_loss")
        if int(prev["season"]) == int(season):
            gap = (date.fromisoformat(gameday) - date.fromisoformat(str(prev["gameday"])[:10])).days
            if gap >= BYE_DAYS:
                s.append("off_bye")
    if NFL_DIVISIONS.get(team) and NFL_DIVISIONS.get(team) == NFL_DIVISIONS.get(opp):
        s.append("division")
    if kickoff_et and kickoff_et[-5:] >= PRIMETIME_ET:
        s.append("primetime")
    return s


def compute_game_trends(sched: pd.DataFrame, upcoming: list[dict], current_season: int,
                        min_n: int = 5) -> list[dict]:
    """Rows {game_pk, team, situation, label, ats_w, ats_l, ats_p, ou_o, ou_u,
    ou_p, n, since_season} for each upcoming game x team x applicable situation
    with >= min_n past games in that situation (seasons current-3..current).
    `upcoming`: {game_pk, home_team, away_team (nflverse abbrs), home_line (home
    spread, negative = home favored; None allowed), gameday "YYYY-MM-DD",
    gametime "HH:MM" ET}."""
    since = current_season - 3
    log = team_game_log(sched[sched["season"] >= since - 1])   # one extra season so the first game has a "previous"
    out: list[dict] = []
    for g in upcoming:
        for team, opp, is_home in ((g["home_team"], g["away_team"], True), (g["away_team"], g["home_team"], False)):
            tlog = log[log["team"] == team] if len(log) else log
            before = tlog[tlog["gameday"] < g["gameday"]] if len(tlog) else tlog
            line = g.get("home_line")
            team_line = None if line is None else (line if is_home else -line)
            now_sits = situations_for(team, is_home, team_line, f"{g['gameday']} {g.get('gametime') or ''}".strip(),
                                      before, opp=opp, season=current_season, gameday=g["gameday"])
            # Situations of each past game (in-window), evaluated against ITS own previous game.
            hist = {k: [] for k in now_sits}
            games = before.reset_index(drop=True)
            for i in range(len(games)):
                row = games.iloc[i]
                if int(row["season"]) < since or row["ats"] is None:
                    continue
                past_sits = situations_for(team, bool(row["is_home"]), row["team_line"],
                                           f"{row['gameday']} {row['gametime']}", games.iloc[:i],
                                           opp=row["opp"], season=int(row["season"]), gameday=str(row["gameday"])[:10])
                for k in now_sits:
                    if k in past_sits:
                        hist[k].append(row)
            for k, rows in hist.items():
                if len(rows) < min_n:
                    continue
                ats = [r["ats"] for r in rows]
                ou = [r["ou"] for r in rows if r["ou"] is not None]
                out.append({"game_pk": g["game_pk"], "team": team, "situation": k,
                            "label": SITUATION_LABELS[k],
                            "ats_w": ats.count("W"), "ats_l": ats.count("L"), "ats_p": ats.count("P"),
                            "ou_o": ou.count("O"), "ou_u": ou.count("U"), "ou_p": ou.count("P"),
                            "n": len(rows), "since_season": since})
    return out
