import numpy as np
import pandas as pd
import pytest

from sportsmodel.context.game_log import (
    cfb_game_log, live_closing_consensus, nfl_game_log)

COLS = ["sport", "season", "week", "game_key", "kickoff", "date_et", "team",
        "opponent", "venue", "pf", "pa", "margin", "team_line", "total_line",
        "role", "su", "ats", "ou", "game_pk", "game_type", "is_post",
        "team_is_fbs", "opp_is_fbs"]


def _nfl(**kw):
    row = dict(game_id="2024_01_AAA_BBB", season=2024, week=1, game_type="REG",
               gameday="2024-09-08", gametime="13:00", home_team="KC",
               away_team="BAL", home_score=27, away_score=20, location="Home",
               spread_line=7.0, total_line=44.5)
    row.update(kw)
    return pd.DataFrame([row])


def _pair(df):
    return df.set_index("team")


def test_nfl_two_rows_columns_and_convention():
    log = nfl_game_log(_nfl())
    assert list(log.columns) == COLS
    assert len(log) == 2
    r = _pair(log)
    assert r.loc["KC", "venue"] == "home" and r.loc["BAL", "venue"] == "away"
    assert r.loc["KC", "opponent"] == "BAL"
    assert r.loc["KC", "team_line"] == 7.0 and r.loc["BAL", "team_line"] == -7.0
    assert r.loc["KC", "role"] == "fav" and r.loc["BAL", "role"] == "dog"
    assert r.loc["KC", "margin"] == 7 and r.loc["BAL", "margin"] == -7
    assert r.loc["KC", "pf"] == 27 and r.loc["KC", "pa"] == 20
    assert r.loc["KC", "su"] == "W" and r.loc["BAL", "su"] == "L"
    # margin 7 == line 7 -> push for both
    assert r.loc["KC", "ats"] == "P" and r.loc["BAL", "ats"] == "P"
    # total 47 > 44.5
    assert r.loc["KC", "ou"] == "O" and r.loc["BAL", "ou"] == "O"
    # kickoff is 13:00 ET (EDT) -> 17:00 UTC
    assert log["kickoff"].iloc[0] == pd.Timestamp("2024-09-08 17:00", tz="UTC")
    assert log["date_et"].iloc[0] == "2024-09-08"
    assert (log["sport"] == "nfl").all()
    assert (log["game_type"] == "REG").all() and not log["is_post"].any()
    assert log["team_is_fbs"].all() and log["opp_is_fbs"].all()


def test_nfl_cover_and_under_and_dog_cover():
    log = nfl_game_log(_nfl(home_score=30, away_score=20, spread_line=3.0,
                            total_line=55.0))
    r = _pair(log)
    assert r.loc["KC", "ats"] == "W" and r.loc["BAL", "ats"] == "L"
    assert r.loc["KC", "ou"] == "U"
    log = nfl_game_log(_nfl(home_score=20, away_score=17, spread_line=7.0,
                            total_line=37.0))
    r = _pair(log)
    assert r.loc["KC", "ats"] == "L" and r.loc["BAL", "ats"] == "W"
    assert r.loc["KC", "su"] == "W" and r.loc["BAL", "su"] == "L"
    assert r.loc["KC", "ou"] == "P"


def test_nfl_tie_pick_and_neutral():
    log = nfl_game_log(_nfl(home_score=20, away_score=20, spread_line=0.0,
                            location="Neutral", game_type="SB"))
    r = _pair(log)
    assert set(log["venue"]) == {"neutral"}
    assert r.loc["KC", "su"] == "T" and r.loc["BAL", "su"] == "T"
    assert r.loc["KC", "role"] == "pick" and r.loc["BAL", "role"] == "pick"
    assert r.loc["KC", "ats"] == "P"
    assert (log["game_type"] == "SB").all() and log["is_post"].all()


def test_nfl_missing_line_and_unplayed():
    log = nfl_game_log(_nfl(spread_line=np.nan, total_line=np.nan))
    r = _pair(log)
    assert r.loc["KC", "ats"] is None and r.loc["KC", "ou"] is None
    assert r.loc["KC", "role"] is None
    assert r.loc["KC", "su"] == "W"
    fut = nfl_game_log(_nfl(home_score=np.nan, away_score=np.nan))
    assert len(fut) == 2
    r = _pair(fut)
    assert r.loc["KC", "su"] is None and r.loc["KC", "ats"] is None
    assert r.loc["KC", "ou"] is None
    assert np.isnan(r.loc["KC", "margin"]) and np.isnan(r.loc["KC", "pf"])
    assert r.loc["KC", "team_line"] == 7.0  # line still known pre-game


def test_nfl_normalizes_team_codes():
    log = nfl_game_log(_nfl(home_team="OAK", away_team="LAR"))
    assert set(log["team"]) == {"LV", "LA"}


def _cfb_sched(**kw):
    row = dict(season=2024, week=1, home_team="333", away_team="61",
               home_score=31, away_score=17, game_type="REG", game_pk=4001,
               start_date="2024-09-01T00:30Z", neutral_site=False,
               conference_game=True, home_conf="8", away_conf="8")
    row.update(kw)
    return pd.DataFrame([row])


def _cfb_lines(**kw):
    row = dict(season=2024, week=1, home_team="333", away_team="61",
               market_spread=10.5, market_total=48.0)
    row.update(kw)
    return pd.DataFrame([row])


def test_cfb_log_basic_and_line_convention():
    log = cfb_game_log(_cfb_sched(), _cfb_lines(), None)
    assert list(log.columns) == COLS
    r = _pair(log)
    assert r.loc["333", "team_line"] == 10.5 and r.loc["61", "team_line"] == -10.5
    assert r.loc["333", "margin"] == 14
    assert r.loc["333", "ats"] == "W" and r.loc["61", "ats"] == "L"
    assert r.loc["333", "ou"] == "P"  # 31 + 17 == 48.0 exactly
    assert (log["game_pk"] == 4001).all() and (log["game_key"] == "4001").all()
    assert (log["sport"] == "cfb").all()
    assert (log["game_type"] == "REG").all() and not log["is_post"].any()
    assert log["team_is_fbs"].all() and log["opp_is_fbs"].all()
    # 00:30Z on Sep 1 is Aug 31 evening ET
    assert log["date_et"].iloc[0] == "2024-08-31"
    assert log["kickoff"].iloc[0] == pd.Timestamp("2024-09-01 00:30", tz="UTC")


def test_cfb_neutral_site_and_missing_line_and_selfmatch_dropped():
    sched = pd.concat([_cfb_sched(neutral_site=True),
                       _cfb_sched(home_team="X", away_team="X", game_pk=4002)],
                      ignore_index=True)
    log = cfb_game_log(sched, _cfb_lines(market_spread=np.nan,
                                         market_total=np.nan), None)
    assert len(log) == 2 and set(log["venue"]) == {"neutral"}
    assert log["ats"].isna().all() and log["ou"].isna().all()
    assert log["role"].isna().all()


def test_cfb_live_close_fills_missing_only():
    sched = pd.concat([_cfb_sched(),
                       _cfb_sched(home_team="2", away_team="3", game_pk=4002,
                                  home_score=np.nan, away_score=np.nan)],
                      ignore_index=True)
    lines = _cfb_lines()  # only game 4001 has a line
    live = pd.DataFrame({"game_pk": [4001, 4002], "close_spread_home": [3.0, -6.5],
                         "close_total": [50.0, 61.5]})
    r = cfb_game_log(sched, lines, live)
    g1 = r[r["game_pk"] == 4001].set_index("team")
    assert g1.loc["333", "team_line"] == 10.5  # lines win over live
    g2 = r[r["game_pk"] == 4002].set_index("team")
    assert g2.loc["2", "team_line"] == -6.5 and g2.loc["3", "team_line"] == 6.5
    assert g2.loc["2", "total_line"] == 61.5
    assert g2.loc["2", "role"] == "dog"
    assert g2.loc["2", "su"] is None and g2.loc["2", "ats"] is None


def _odds(game_pk, book, market, side, line, captured, commence="2024-10-01T00:00:00Z"):
    return dict(game_pk=game_pk, book=book, market=market, side=side, line=line,
                price=-110, captured_at=captured, commence_time=commence)


def test_live_consensus_median_of_last_pre_kickoff_captures():
    rows = [
        # book A: spread moves -6 -> -7, then a post-kickoff live line -20 (ignored)
        _odds(1, "A", "spread", "home", -6.0, "2024-09-29T12:00:00Z"),
        _odds(1, "A", "spread", "away", 6.0, "2024-09-29T12:00:00Z"),
        _odds(1, "A", "spread", "home", -7.0, "2024-09-30T23:00:00Z"),
        _odds(1, "A", "spread", "away", 7.0, "2024-09-30T23:00:00Z"),
        _odds(1, "A", "spread", "home", -20.0, "2024-10-01T01:00:00Z"),
        _odds(1, "B", "spread", "home", -8.0, "2024-09-30T22:00:00Z"),
        _odds(1, "C", "spread", "away", 9.0, "2024-09-30T22:00:00Z"),  # away only
        _odds(1, "A", "total", "over", 51.0, "2024-09-30T23:00:00Z"),
        _odds(1, "A", "total", "under", 51.0, "2024-09-30T23:00:00Z"),
        _odds(1, "B", "total", "over", 52.0, "2024-09-30T22:00:00Z"),
        _odds(1, "A", "moneyline", "home", -300, "2024-09-30T22:00:00Z"),
        # game 2: only a post-kickoff capture -> nothing
        _odds(2, "A", "spread", "home", -3.0, "2024-10-01T02:00:00Z"),
    ]
    out = live_closing_consensus(pd.DataFrame(rows)).set_index("game_pk")
    assert list(out.columns) == ["close_spread_home", "close_total"]
    # home margins: A 7, B 8, C 9 -> median 8
    assert out.loc[1, "close_spread_home"] == 8.0
    assert out.loc[1, "close_total"] == 51.5
    assert 2 not in out.index


def test_live_consensus_empty():
    out = live_closing_consensus(pd.DataFrame(
        columns=["game_pk", "book", "market", "side", "line", "price",
                 "captured_at", "commence_time"]))
    assert list(out.columns) == ["game_pk", "close_spread_home", "close_total"]
    assert out.empty


def test_cfb_fbs_flags_for_pooled_fcs_pseudo_team():
    log = cfb_game_log(_cfb_sched(away_team="FCS"), _cfb_lines(away_team="FCS"),
                       None)
    r = _pair(log)
    assert r.loc["333", "team_is_fbs"] and not r.loc["333", "opp_is_fbs"]
    assert not r.loc["FCS", "team_is_fbs"] and r.loc["FCS", "opp_is_fbs"]


# Exact column set of the nflverse release `schedules` dataset.
RELEASE_COLS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "away_score", "home_team", "home_score", "location", "result",
    "total", "overtime", "old_game_id", "gsis", "nfl_detail_id", "pfr", "pff",
    "espn", "ftn", "away_rest", "home_rest", "away_moneyline", "home_moneyline",
    "spread_line", "away_spread_odds", "home_spread_odds", "total_line",
    "under_odds", "over_odds", "div_game", "roof", "surface", "temp", "wind",
    "away_qb_id", "home_qb_id", "away_qb_name", "home_qb_name", "away_coach",
    "home_coach", "referee", "stadium_id", "stadium"]


def test_nfl_release_schema_fixture_neutral_and_post():
    base = {c: None for c in RELEASE_COLS}
    rows = []
    for gid, loc, gt in [("2024_22_KC_PHI", "Neutral", "SB"),
                         ("2024_01_KC_BAL", "Home", "REG")]:
        rows.append({**base, "game_id": gid, "season": 2024, "week": 22,
                     "game_type": gt, "gameday": "2025-02-09",
                     "gametime": "18:30", "home_team": "KC", "away_team": "PHI",
                     "home_score": 22, "away_score": 40, "location": loc,
                     "spread_line": -1.5, "total_line": 49.5})
    log = nfl_game_log(pd.DataFrame(rows, columns=RELEASE_COLS))
    by = log[log["game_key"] == "2024_22_KC_PHI"]
    assert set(by["venue"]) == {"neutral"} and by["is_post"].all()
    assert set(log[log["game_key"] == "2024_01_KC_BAL"]["venue"]) == {"home", "away"}


def test_nfl_requires_location_column():
    df = _nfl().drop(columns=["location"])
    with pytest.raises(ValueError, match="location"):
        nfl_game_log(df)


def test_nfl_real_release_2024_neutral_games():
    from sportsmodel.nfl.nflverse import load_release
    try:
        sch = load_release("schedules", [2024])
    except Exception as exc:  # no network
        pytest.skip(f"nflverse release unavailable: {exc}")
    log = nfl_game_log(sch)
    assert len(log) == 2 * len(sch)
    neutral_games = log[log["venue"] == "neutral"]["game_key"].nunique()
    assert neutral_games == (sch["location"] == "Neutral").sum() >= 1
    assert log[log["game_type"] == "SB"]["venue"].eq("neutral").all()
    assert log["is_post"].sum() == 2 * (sch["game_type"] != "REG").sum()


def test_live_consensus_capture_at_kickoff_excluded():
    rows = [
        _odds(1, "A", "spread", "home", -3.0, "2024-09-30T23:00:00Z"),
        _odds(1, "A", "spread", "home", -9.0, "2024-10-01T00:00:00Z"),  # == kickoff
    ]
    out = live_closing_consensus(pd.DataFrame(rows)).set_index("game_pk")
    assert out.loc[1, "close_spread_home"] == 3.0
    only_boundary = live_closing_consensus(pd.DataFrame(rows[1:]))
    assert only_boundary.empty


def test_live_consensus_book_latest_capture_is_away_only():
    rows = [
        _odds(1, "A", "spread", "home", -3.0, "2024-09-30T20:00:00Z"),
        _odds(1, "A", "spread", "away", 3.0, "2024-09-30T20:00:00Z"),
        # later capture only has the away side, line moved
        _odds(1, "A", "spread", "away", 5.5, "2024-09-30T23:00:00Z"),
    ]
    out = live_closing_consensus(pd.DataFrame(rows)).set_index("game_pk")
    assert out.loc[1, "close_spread_home"] == 5.5
