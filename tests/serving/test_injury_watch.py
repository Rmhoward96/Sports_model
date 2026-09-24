"""Pure injury-watch helpers: per-game Out/Doubtful statuses, fingerprints,
human diffs, and changed-game detection."""
from sportsmodel.serving.injury_watch import changed_games, diff_statuses, fingerprint, game_statuses

HOME, AWAY = "Atlanta Falcons", "Carolina Panthers"

def test_game_statuses_out_doubtful_only_sorted():
    inj = {
        HOME: [{"player": "Zed Z", "status": "Out"},
               {"player": "Amy A", "status": "Doubtful"},
               {"player": "Qu Q", "status": "Questionable"},
               {"player": "Ir I", "status": "Injured Reserve"}],
        AWAY: [{"player": "Bob B", "status": "Probable"}],
        "Other Team": [{"player": "X", "status": "Out"}],
    }
    got = game_statuses(HOME, AWAY, inj)
    assert got == [
        {"team": HOME, "player": "Amy A", "status": "Doubtful"},
        {"team": HOME, "player": "Ir I", "status": "Out"},
        {"team": HOME, "player": "Zed Z", "status": "Out"},
    ]


def test_game_statuses_missing_teams_empty():
    assert game_statuses(HOME, AWAY, {}) == []


def test_questionable_only_change_same_fingerprint():
    a = {HOME: [{"player": "P", "status": "Out"}]}
    b = {HOME: [{"player": "P", "status": "Out"}, {"player": "Q", "status": "Questionable"}]}
    assert fingerprint(game_statuses(HOME, AWAY, a)) == fingerprint(game_statuses(HOME, AWAY, b))


def test_out_to_active_changes_fingerprint():
    a = {HOME: [{"player": "Michael Penix Jr.", "status": "Out"}]}
    b = {HOME: [{"player": "Michael Penix Jr.", "status": "Active"}]}
    assert fingerprint(game_statuses(HOME, AWAY, a)) != fingerprint(game_statuses(HOME, AWAY, b))


def test_fingerprint_order_insensitive():
    s1 = [{"team": HOME, "player": "A", "status": "Out"}, {"team": AWAY, "player": "B", "status": "Doubtful"}]
    assert fingerprint(s1) == fingerprint(list(reversed(s1)))


def test_fingerprint_is_sha1_hex():
    fp = fingerprint([])
    assert len(fp) == 40 and all(c in "0123456789abcdef" for c in fp)


def test_diff_statuses_lines():
    old = [{"team": HOME, "player": "Michael Penix Jr.", "status": "Out"},
           {"team": AWAY, "player": "B", "status": "Doubtful"}]
    new = [{"team": AWAY, "player": "B", "status": "Out"},
           {"team": AWAY, "player": "C", "status": "Doubtful"}]
    assert diff_statuses(old, new) == [
        f"{HOME}: Michael Penix Jr. Out -> (none)",
        f"{AWAY}: B Doubtful -> Out",
        f"{AWAY}: C (none) -> Doubtful",
    ]


def test_diff_statuses_no_change_empty():
    s = [{"team": HOME, "player": "A", "status": "Out"}]
    assert diff_statuses(s, list(s)) == []


def test_changed_games_missing_stored_is_changed():
    assert changed_games({1: "a", 2: "b", 3: "c"}, {1: "a", 2: "x"}) == [2, 3]


def test_changed_games_none_changed():
    assert changed_games({1: "a"}, {1: "a", 9: "z"}) == []
