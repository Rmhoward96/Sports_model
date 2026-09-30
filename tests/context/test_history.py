import json

import numpy as np
import pandas as pd
import pytest

from sportsmodel.context.game_log import cfb_game_log, nfl_game_log
from sportsmodel.context.history import history_for_games, team_history_asof

T = "KC"


def _sched(rows):
    """rows: (season, week, venue, pf, pa, team_line, total) for team KC.
    venue in home/away/neutral; pf None => unplayed. line/total may be None."""
    out = []
    for i, (season, week, venue, pf, pa, line, total) in enumerate(rows):
        home = venue != "away"
        hs, as_ = (pf, pa) if home else (pa, pf)
        if pf is None:
            hs = as_ = np.nan
        sl = np.nan if line is None else (line if home else -line)
        out.append(dict(
            game_id=f"{season}_{week:02d}_{i}", season=season, week=week,
            game_type="REG", gameday=str(pd.Timestamp("2023-09-10")
                                         + pd.Timedelta(days=7 * i)),
            gametime="13:00", home_team=T if home else "BAL",
            away_team="BAL" if home else T, home_score=hs, away_score=as_,
            location="Neutral" if venue == "neutral" else "Home",
            spread_line=sl, total_line=np.nan if total is None else total))
    return nfl_game_log(pd.DataFrame(out))


# 12 played games over two seasons + one upcoming (2024 wk 7).
# (season, week, venue, pf, pa, team_line, total)
FIX = [
    (2023, 1, "home", 24, 17, 3, 40),      # W  W  O   cover +4  tot +1
    (2023, 2, "away", 10, 20, -3, 45),     # L  L  U   cover -7  tot -15
    (2023, 3, "home", 20, 20, 0, 40),      # T  P  P   cover 0   tot 0   (pick)
    (2023, 4, "away", 27, 24, 3, 51),      # W  P  P   cover 0   tot 0   (fav)
    (2023, 5, "home", 14, 21, None, None), # L  -  -   no lines
    (2023, 6, "neutral", 31, 10, 6, 48),   # W  W  U   cover +15 tot -7
    (2024, 1, "home", 28, 14, 10, 44),     # W  W  U   cover +4  tot -2
    (2024, 2, "away", 17, 24, -2, 41),     # L  L  P   cover -5  tot 0
    (2024, 3, "home", 35, 20, 7, 50),      # W  W  O   cover +8  tot +5
    (2024, 4, "away", 21, 17, -3, 40),     # W  W  U   cover +7  tot -2
    (2024, 5, "home", 13, 20, 4, 30),      # L  L  O   cover -11 tot +3
    (2024, 6, "home", 24, 10, 3.5, 45),    # W  W  U   cover +10.5 tot -11
    (2024, 7, "home", None, None, 6, 44),  # upcoming
]


@pytest.fixture
def log():
    return _sched(FIX)


@pytest.fixture
def asof(log):
    kt = log[(log.team == T) & log.su.isna()]["kickoff"]
    return kt.iloc[0]


def test_windows_hand_computed(log, asof):
    h = team_history_asof(log, T, asof)
    w = h["windows"]
    l5 = w["L5"]
    assert l5["su"] == "3-2" and l5["ats"] == "3-2-0" and l5["ou"] == "2-2-1"
    assert l5["n"] == 5
    assert l5["avg_margin"] == pytest.approx(3.8)
    assert l5["avg_cover"] == pytest.approx(1.9)
    assert l5["avg_total_vs_line"] == pytest.approx(-1.0)

    l10 = w["L10"]  # games 3..12 (includes tie, pushes, an unlined game)
    assert l10["su"] == "6-3-1" and l10["n"] == 10
    assert l10["ats"] == "5-2-2" and l10["n_ats"] == 9
    assert l10["ou"] == "2-4-3" and l10["n_ou"] == 9
    assert l10["avg_margin"] == pytest.approx(5.0)
    assert l10["avg_cover"] == pytest.approx(28.5 / 9)
    assert l10["avg_total_vs_line"] == pytest.approx(-14 / 9)

    l20 = w["L20"]  # only 12 games exist, spans both seasons
    assert l20["su"] == "7-4-1" and l20["n"] == 12
    assert l20["ats"] == "6-3-2" and l20["n_ats"] == 11
    assert l20["ou"] == "3-5-3" and l20["n_ou"] == 11
    assert l20["avg_margin"] == pytest.approx(47 / 12)
    assert l20["avg_cover"] == pytest.approx(25.5 / 11)
    assert l20["avg_total_vs_line"] == pytest.approx(-28 / 11)

    s = w["season"]  # 2024 = season of the upcoming game
    assert s["su"] == "4-2" and s["ats"] == "4-2-0" and s["ou"] == "2-3-1"
    assert s["n"] == 6
    assert s["avg_margin"] == pytest.approx(5.5)
    assert s["avg_cover"] == pytest.approx(2.25)
    assert s["avg_total_vs_line"] == pytest.approx(-7 / 6)


def test_season_explicit_override(log, asof):
    h = team_history_asof(log, T, asof, season=2023)
    assert h["windows"]["season"]["su"] == "3-2-1"  # W L T W L W
    assert h["windows"]["season"]["n"] == 6
    assert h["windows"]["season"]["ats"] == "2-1-2"


def test_splits_hand_computed(log, asof):
    sp = team_history_asof(log, T, asof)["splits"]
    # 2024 season only: home = g7, g9, g11, g12 ; away = g8, g10
    assert sp["home"] == {"su": "3-1", "ats": "3-1-0", "n": 4}
    assert sp["away"] == {"su": "1-1", "ats": "1-1-0", "n": 2}
    # fav = g7, g9, g11, g12 ; dog = g8, g10
    assert sp["fav"] == {"su": "3-1", "ats": "3-1-0", "n": 4}
    assert sp["dog"] == {"su": "1-1", "ats": "1-1-0", "n": 2}
    sp23 = team_history_asof(log, T, asof, season=2023)["splits"]
    # 2023: home g1(W) g3(T) g5(L); away g2(L) g4(W); neutral g6 in neither
    assert sp23["home"] == {"su": "1-1-1", "ats": "1-0-1", "n": 3}
    assert sp23["away"] == {"su": "1-1", "ats": "0-1-1", "n": 2}
    # fav: g1 W/W, g4 W/P, g6 W/W ; dog: g2 L/L ; pick g3 and unlined g5 in neither
    assert sp23["fav"] == {"su": "3-0", "ats": "2-0-1", "n": 3}
    assert sp23["dog"] == {"su": "0-1", "ats": "0-1-0", "n": 1}


def test_streaks_main_fixture(log, asof):
    st = team_history_asof(log, T, asof)["streaks"]
    assert st["su"] == "W1" and st["ats"] == "W1" and st["ou"] == "U1"
    assert st["notes"] == {}


def _seq(rows):
    return _sched([(2024, i + 1, "home", pf, pa, line, tot)
                   for i, (pf, pa, line, tot) in enumerate(rows)]
                  + [(2024, len(rows) + 1, "home", None, None, 1, 40)])


def _hist(log):
    kt = log[(log.team == T) & log.su.isna()]["kickoff"].iloc[0]
    return team_history_asof(log, T, kt)


def test_streak_run_and_tie_breaks():
    # SU: L W W T W W -> tie breaks the streak, current = W2
    lg = _seq([(0, 7, 0, 40), (7, 0, 0, 40), (7, 0, 0, 40), (7, 7, 0, 40),
               (7, 0, 0, 40), (7, 0, 0, 40)])
    assert _hist(lg)["streaks"]["su"] == "W2"
    # ending on a tie -> T1
    lg = _seq([(7, 0, 0, 40), (7, 0, 0, 40), (7, 7, 0, 40)])
    assert _hist(lg)["streaks"]["su"] == "T1"


def test_ats_ou_pushes_break_and_unlined_skipped():
    # ats: W W P W (pushes break) -> W1 ; margin 7 vs line 3 = W, margin 3 vs 3 = P
    lg = _seq([(10, 3, 3, 40), (10, 3, 3, 40), (13, 10, 3, 40),
               (10, 3, 3, 40)])
    st = _hist(lg)["streaks"]
    assert st["ats"] == "W1"
    # unlined game in the middle does not break: W W (none) W -> W3
    lg = _seq([(10, 3, 3, 40), (10, 3, 3, 40), (10, 3, None, None),
               (10, 3, 3, 40)])
    st = _hist(lg)["streaks"]
    assert st["ats"] == "W3"
    # O/U: pts 17 vs 40 -> U ; pts 40 vs 40 -> P (breaks)
    lg = _seq([(10, 7, 3, 40), (10, 7, 3, 40), (20, 20, 3, 40),
               (10, 7, 3, 40), (10, 7, 3, 40)])
    assert _hist(lg)["streaks"]["ou"] == "U2"


def test_x_of_last_y_note():
    # OU: U U U O U U U -> streak U3, 6 of last 7 are U
    rows = [(10, 7, 3, 40)] * 3 + [(30, 20, 3, 40)] + [(10, 7, 3, 40)] * 3
    st = _hist(_seq(rows))["streaks"]
    assert st["ou"] == "U3" and st["notes"]["ou"] == "U6 of 7"
    # 5 of 7 -> no note
    rows = [(10, 7, 3, 40)] * 2 + [(30, 20, 3, 40)] * 2 + [(10, 7, 3, 40)] * 3
    st = _hist(_seq(rows))["streaks"]
    assert "ou" not in st["notes"]
    # only 6 games available -> no note (needs 7)
    st = _hist(_seq([(10, 7, 3, 40)] * 6))["streaks"]
    assert st["ou"] == "U6" and "ou" not in st["notes"]
    # 7 in a row
    st = _hist(_seq([(10, 7, 3, 40)] * 7))["streaks"]
    assert st["notes"]["ou"] == "U7 of 7"


def test_leak_test_at_or_after_asof_never_changes_result(log, asof):
    base = team_history_asof(log, T, asof)
    extra = _sched(FIX + [(2024, 8, "away", 3, 40, -9, 30),   # played, later
                          (2024, 9, "home", 50, 0, 3, 40)])
    # game exactly at asof (the upcoming game) now has a result too
    extra_at = extra.copy()
    m = (extra_at.team == T) & (extra_at.kickoff == asof)
    extra_at.loc[m, ["su", "ats", "ou"]] = ["W", "W", "O"]
    extra_at.loc[m, ["pf", "pa", "margin"]] = [99, 0, 99]
    assert team_history_asof(extra, T, asof) == base
    assert team_history_asof(extra_at, T, asof) == base


def test_early_and_empty_history():
    lg = _sched([(2024, 1, "home", None, None, 3, 40)])
    h = _hist(lg)
    assert h["windows"]["L5"]["n"] == 0 and h["windows"]["L5"]["su"] == "0-0"
    assert h["windows"]["L5"]["avg_margin"] is None
    assert h["streaks"]["su"] is None
    assert h["splits"]["home"]["n"] == 0


def test_fcs_pseudo_team_gets_no_history():
    sched = pd.DataFrame([dict(
        season=2024, week=1, home_team="Ohio State", away_team="Grambling",
        home_score=52, away_score=0, neutral_site=False,
        start_date="2024-09-01T16:00:00Z", game_pk=1, game_type="REG")])
    sched.loc[0, "away_team"] = "FCS"
    lines = pd.DataFrame(columns=["season", "week", "home_team", "away_team",
                                  "market_spread", "market_total"])
    lg = cfb_game_log(sched, lines)
    asof = pd.Timestamp("2025-01-01", tz="UTC")
    assert team_history_asof(lg, "FCS", asof) is None
    h = team_history_asof(lg, "Ohio State", asof)
    # FBS team's game vs FCS counts for SU, not ATS (no line)
    assert h["windows"]["L5"]["su"] == "1-0"
    assert h["windows"]["L5"]["ats"] == "0-0-0" and h["windows"]["L5"]["n_ats"] == 0


def test_naive_asof_treated_as_utc(log, asof):
    naive = asof.tz_convert("UTC").tz_localize(None)
    assert team_history_asof(log, T, naive) == team_history_asof(log, T, asof)


def test_history_for_games(log):
    upcoming = log[log.su.isna()]
    out = history_for_games(log, upcoming)
    assert len(out) == 2 and set(out.team) == {"KC", "BAL"}
    r = out[out.team == "KC"].iloc[0]
    assert r["windows"]["L5"]["su"] == "3-2"
    assert r["game_key"] == upcoming[upcoming.team == "KC"].iloc[0]["game_key"]
    # BAL was KC's opponent in every game: mirror-image record
    assert out[out.team == "BAL"].iloc[0]["windows"]["L5"]["su"] == "2-3"
    for col in ("windows", "streaks", "splits"):
        json.dumps(list(out[col]))  # JSON-able (no numpy scalars / NaN objects)


def test_history_for_games_skips_fcs():
    sched = pd.DataFrame([dict(
        season=2024, week=1, home_team="Ohio State", away_team="FCS",
        home_score=np.nan, away_score=np.nan, neutral_site=False,
        start_date="2024-09-01T16:00:00Z", game_pk=1, game_type="REG")])
    lines = pd.DataFrame(columns=["season", "week", "home_team", "away_team",
                                  "market_spread", "market_total"])
    lg = cfb_game_log(sched, lines)
    out = history_for_games(lg, lg)
    assert list(out.team) == ["Ohio State"]
