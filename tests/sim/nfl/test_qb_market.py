from datetime import datetime, timedelta, timezone

import pandas as pd

from sportsmodel.sim.nfl.qb_market import books_qb_names, promote_books_qb
from sportsmodel.sim.nfl.usage import active_usage

T = datetime(2026, 9, 28, 20, 56, tzinfo=timezone.utc)


def _odds(name, at, game_pk=1):
    return {"game_pk": game_pk, "player_name": name, "captured_at": at}


def _chart(rows, week=4, team="CHI"):
    return pd.DataFrame([
        {"club_code": team, "season": 2026, "week": week, "position": pos, "gsis_id": gid,
         "full_name": name, "football_name": name.split()[0], "depth_team": dt}
        for gid, name, pos, dt in rows
    ])


BEARS = [("00-B", "Tyson Bagent", "QB", "1"), ("00-K", "Case Keenum", "QB", "2"),
         ("00-W", "DJ Moore", "WR", "1")]


def test_books_names_keep_only_the_latest_pull():
    rows = [_odds("Tyson Bagent", T - timedelta(days=2)),
            _odds("Case Keenum", T), _odds("Jalen Hurts", T),
            _odds("Case Keenum", T - timedelta(hours=5))]
    assert books_qb_names(rows) == {1: {"case keenum", "jalen hurts"}}


def test_books_names_tolerate_a_pull_spread_over_minutes():
    rows = [_odds("Case Keenum", T - timedelta(minutes=20)), _odds("Jalen Hurts", T)]
    assert books_qb_names(rows)[1] == {"case keenum", "jalen hurts"}


def test_promotes_the_books_qb_and_active_usage_starts_him():
    depth = _chart(BEARS)
    fixed, switch = promote_books_qb(depth, "CHI", 2026, 4, {"case keenum", "jalen hurts"})
    assert switch == ("Tyson Bagent", "Case Keenum")
    assert depth.loc[depth.gsis_id == "00-K", "depth_team"].item() == "2"   # input untouched
    empty = pd.DataFrame(columns=["season", "week", "player_id", "targets", "carries"])
    players, qb1 = active_usage("CHI", 2026, 4, fixed, empty, None, None, set())
    assert qb1 == "00-K"
    assert [p.name for p in players if p.pos == "QB"] == ["Case Keenum"]


def test_no_change_when_the_chart_already_agrees():
    fixed, switch = promote_books_qb(_chart(BEARS), "CHI", 2026, 4, {"tyson bagent"})
    assert switch is None and fixed["depth_team"].tolist() == ["1", "2", "1"]


def test_no_change_without_a_line_or_when_ambiguous():
    assert promote_books_qb(_chart(BEARS), "CHI", 2026, 4, set())[1] is None
    assert promote_books_qb(_chart(BEARS), "CHI", 2026, 4, {"jalen hurts"})[1] is None
    both = {"tyson bagent", "case keenum"}
    assert promote_books_qb(_chart(BEARS), "CHI", 2026, 4, both)[1] is None


def test_out_players_are_ignored_on_both_sides():
    # Bagent Out: Keenum is QB1 anyway -> nothing to promote.
    assert promote_books_qb(_chart(BEARS), "CHI", 2026, 4, {"case keenum"},
                            out_names={"Tyson Bagent"})[1] is None
    # The books' QB is Out: never promoted.
    assert promote_books_qb(_chart(BEARS), "CHI", 2026, 4, {"case keenum"},
                            out_names={"Case Keenum"})[1] is None


def test_uses_the_latest_chart_when_the_target_week_has_none():
    fixed, switch = promote_books_qb(_chart(BEARS, week=3), "CHI", 2026, 4, {"case keenum"})
    assert switch == ("Tyson Bagent", "Case Keenum")
    assert fixed.loc[fixed.gsis_id == "00-K", "depth_team"].item() == "0"


def test_other_teams_untouched():
    depth = pd.concat([_chart(BEARS), _chart([("00-H", "Jalen Hurts", "QB", "1")], team="PHI")])
    fixed, switch = promote_books_qb(depth, "PHI", 2026, 4, {"case keenum", "jalen hurts"})
    assert switch is None and fixed is depth


def test_duplicate_index_labels_only_touch_the_promoted_row():
    # concat keeps both frames' 0..n labels: PHI's Hurts shares label 1 with Keenum
    depth = pd.concat([_chart(BEARS), _chart([("00-X", "Backup Guy", "QB", "2"),
                                              ("00-H", "Jalen Hurts", "QB", "1")], team="PHI")])
    fixed, switch = promote_books_qb(depth, "CHI", 2026, 4, {"case keenum"})
    assert switch == ("Tyson Bagent", "Case Keenum")
    assert fixed["depth_team"].tolist() == ["1", "0", "1", "2", "1"]


def test_numeric_depth_slots_stay_numeric():
    # the live nflverse chart stores depth_team as int32 -- a "0" string raised
    depth = _chart(BEARS)
    depth["depth_team"] = depth["depth_team"].astype("int32")
    fixed, switch = promote_books_qb(depth, "CHI", 2026, 4, {"case keenum"})
    assert switch == ("Tyson Bagent", "Case Keenum")
    assert fixed["depth_team"].tolist() == [1, 0, 1] and fixed["depth_team"].dtype == "int32"
    empty = pd.DataFrame(columns=["season", "week", "player_id", "targets", "carries"])
    assert active_usage("CHI", 2026, 4, fixed, empty, None, None, set())[1] == "00-K"
