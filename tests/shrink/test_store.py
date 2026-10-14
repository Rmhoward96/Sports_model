"""assets/shrink_k.json loading and the k = 0 fail-safe. No network, no database."""
import json
import pytest

from sportsmodel import config
from sportsmodel.shrink import store
from sportsmodel.shrink.store import ShrinkTable, load_table


def write(tmp_path, doc, name="shrink_k.json"):
    path = tmp_path / name
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return path


GOOD = {"markets": {"nfl": {"rush_yds": {"k": 0.2, "n": 400, "k_raw": 0.25}, "reception_yds": {"k": 0.1, "n": 50, "k_raw": 0.0}},
                    "mlb": {"total_bases": {"k": 0.0, "n": 0, "k_raw": None}}}}


def test_the_default_path_is_assets_shrink_k_json_in_the_project():
    assert store.SHRINK_K_PATH == config.PROJECT_ROOT / "assets" / "shrink_k.json"


def test_a_good_file_is_read_per_sport_and_market(tmp_path):
    assert load_table(write(tmp_path, GOOD)).k("nfl", "rush_yds") == 0.2
    t = ShrinkTable.from_dict(GOOD)
    assert t.k("nfl", "rush_yds") == 0.2 and t.k("nfl", "reception_yds") == 0.1
    assert t.warning() is None and t.problem is None


def test_a_missing_file_gives_k_zero_everywhere_and_a_warning(tmp_path):
    t = load_table(tmp_path / "nope.json")
    assert t.k("nfl", "rush_yds") == 0.0 and t.k("nfl", "anything") == 0.0
    assert "does not exist yet" in t.warning() and "k=0" in t.warning()


@pytest.mark.parametrize("text", ["", "{not json", "[]", "null", "42", '"x"', '{"markets": 5}', '{"markets": []}', "{}"])
def test_an_unusable_file_gives_k_zero_and_never_raises(tmp_path, text):
    t = load_table(write(tmp_path, text))
    assert t.k("nfl", "rush_yds") == 0.0 and t.warning().startswith("::warning::shrink:")


def test_an_unreadable_path_gives_k_zero(tmp_path):
    t = load_table(tmp_path)                       # a directory, not a file
    assert t.k("nfl", "rush_yds") == 0.0 and "unreadable" in t.warning()


def test_a_non_utf8_file_gives_k_zero(tmp_path):
    path = tmp_path / "shrink_k.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    assert load_table(path).k("nfl", "rush_yds") == 0.0


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-0.1", "1.5", '"0.2"', "null", "true", "[0.2]", "{}"])
def test_an_entry_whose_k_is_not_a_number_in_zero_one_is_ignored_not_trusted(tmp_path, bad):
    doc = '{"markets": {"nfl": {"rush_yds": {"k": %s, "n": 5}, "receptions": {"k": 0.3}}}}' % bad
    t = load_table(write(tmp_path, doc))
    assert t.k("nfl", "rush_yds") == 0.0 and t.k("nfl", "receptions") == 0.3
    assert "nfl/rush_yds" in t.warning() and "malformed" in t.warning()


def test_k_of_exactly_zero_and_one_are_valid():
    t = ShrinkTable.from_dict({"markets": {"nfl": {"a": {"k": 0}, "b": {"k": 1}, "c": {"k": 1.0}}}})
    assert (t.k("nfl", "a"), t.k("nfl", "b"), t.k("nfl", "c")) == (0.0, 1.0, 1.0) and t.problem is None


def test_a_missing_market_entry_is_zero_and_warned_once_per_market():
    t = ShrinkTable.from_dict(GOOD)
    assert t.k("nfl", "pass_tds") == 0.0 and t.k("nfl", "pass_tds") == 0.0 and t.k("nfl", "receptions") == 0.0
    w = t.warning()
    assert w.count("nfl/pass_tds") == 1 and "nfl/receptions" in w and w.count("::warning::") == 1


def test_mlb_is_always_zero_even_if_the_file_says_otherwise():
    t = ShrinkTable.from_dict({"markets": {"mlb": {"total_bases": {"k": 0.9}}, "nfl": {}}})
    assert t.k("mlb", "total_bases") == 0.0 and t.k("mlb", "pitcher_ks") == 0.0
    assert t.warning() is None                          # an MLB lookup is never a "missing entry"


def test_the_committed_file_if_present_is_a_valid_table():
    path = store.SHRINK_K_PATH
    if path.exists():                                   # absent until the first fit_shrink_k run
        t = load_table(path)
        assert t.problem is None
