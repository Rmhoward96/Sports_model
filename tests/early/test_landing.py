"""landing.py: table building (windows, widening, thin tails), lookups, and the committed JSON assets. No network."""
import json

import pytest

from sportsmodel.early import landing
from sportsmodel.early.constants import LANDING_TABLES, MIN_GAMES


def pairs_at(line, results):
    return [(line, r) for r in results]


# 400 games closing at 3.0: 40 land on exactly 3 (the key number), the rest spread over -20..20 skipping 3
RESULTS = [3] * 40 + [r for r in range(-20, 21) if r != 3] * 9 + [0] * 0
BASE = pairs_at(3.0, RESULTS)


def table(pairs, market="spread"):
    return landing.build_table(pairs, sport="nfl", market=market, seasons=(2010, 2025), source="test")


def test_the_sample_is_big_enough_for_these_tests():
    assert len(BASE) >= MIN_GAMES


def test_pmf_sums_to_one_and_keeps_the_key_number_mass():
    t = landing.LandingTable(table(BASE))
    pmf = t.pmf(3.0)
    assert sum(pmf.probs.values()) == pytest.approx(1.0)
    assert pmf.mass(3) == pytest.approx(40 / len(BASE)) and pmf.mass(3) > pmf.mass(4) * 3
    assert pmf.n == len(BASE) and pmf.window == 1.0


def test_whole_number_lines_carry_push_mass_and_half_lines_are_on_the_grid_too():
    t = landing.LandingTable(table(BASE + pairs_at(4.0, RESULTS)))
    assert t.pmf(3.0).mass(3) == pytest.approx(40 / len(BASE))     # the push mass of a whole-number threshold
    assert t.pmf(3.5) is not None and t.pmf(3.5).n == 2 * len(BASE)


def test_window_is_one_point_unless_that_gives_fewer_than_300_games_then_two():
    narrow = BASE
    assert landing.LandingTable(table(narrow)).pmf(3.0).window == 1.0
    sparse = pairs_at(3.0, RESULTS[:100]) + pairs_at(4.5, RESULTS[:150]) + pairs_at(1.5, RESULTS[:150])
    t = landing.LandingTable(table(sparse))
    assert t.pmf(3.0).window == 2.0 and t.pmf(3.0).n == 400           # 100 within +/-1, 400 within +/-2
    assert t.pmf(3.0).mass(3) > 0


def test_grid_values_still_short_at_two_points_are_left_out():
    sparse = pairs_at(3.0, RESULTS[:100]) + pairs_at(20.0, RESULTS[:100])
    t = landing.LandingTable(table(sparse))
    assert t.pmf(3.0) is None and t.pmf(20.0) is None and t.pmf(10.0) is None


def test_the_window_edges_are_inclusive():
    edge = pairs_at(2.0, RESULTS[:150]) + pairs_at(4.0, RESULTS[:150])            # exactly 1.0 point either side of 3.0
    assert landing.LandingTable(table(edge)).pmf(3.0).n == 300


def test_off_grid_and_out_of_range_lines_answer_none():
    t = landing.LandingTable(table(BASE))
    assert t.pmf(3.25) is None and t.pmf(3.75) is None and t.pmf(99.5) is None and t.pmf(None) is None


def test_a_pick_em_never_looks_up_negative_zero():
    pairs = pairs_at(0.0, RESULTS)
    t = landing.LandingTable(table(pairs))
    assert landing.grid_key(-0.0) == "0.0" and t.pmf(-0.0) is not None and t.pmf(0.0) is not None


def test_counts_not_probabilities_are_stored_so_the_loader_normalises_exactly():
    raw = table(BASE)
    entry = raw["entries"]["3.0"]
    assert entry["lo"] == -20 and all(isinstance(c, int) for c in entry["counts"]) and sum(entry["counts"]) == entry["n"]
    assert raw["result"] == "home_margin" and table(BASE, "total")["result"] == "total_points"
    assert raw["games"] == len(BASE) and raw["seasons"] == [2010, 2025]


def test_building_is_deterministic_and_order_independent():
    assert json.dumps(table(BASE), sort_keys=True) == json.dumps(table(list(reversed(BASE))), sort_keys=True)


def test_build_rejects_empty_input_and_unknown_markets():
    with pytest.raises(ValueError):
        table([])
    with pytest.raises(ValueError):
        table(BASE, market="moneyline")


# ---- the committed assets ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def committed():
    return landing.load_tables()


def test_all_three_committed_tables_load(committed):
    assert set(committed) == set(LANDING_TABLES)


def test_committed_tables_normalise_and_respect_the_sample_floor(committed):
    for (sport, market), t in committed.items():
        assert t.meta["entries"], (sport, market)
        for key, entry in t.meta["entries"].items():
            pmf = t.pmf(float(key))
            assert sum(pmf.probs.values()) == pytest.approx(1.0)
            assert entry["n"] >= MIN_GAMES and entry["window"] in (1.0, 2.0)


def test_committed_nfl_spread_table_keeps_its_key_numbers(committed):
    pmf = committed[("nfl", "spread")].pmf(3.0)                       # closing line 3: home favoured by 3
    assert pmf.mass(3) > 0.08 and pmf.mass(7) > 0.04 and pmf.mass(3) > 2 * pmf.mass(4)
    assert committed[("nfl", "spread")].pmf(-3.0).mass(-3) > 0.07


def test_committed_cfb_tables_cover_the_lines_the_monitor_will_see(committed):
    spread, total = committed[("cfb", "spread")], committed[("cfb", "total")]
    for centre in (-7.0, -3.5, 3.0, 10.5, 21.0):
        assert spread.pmf(centre) is not None, centre
    for centre in (45.5, 52.5, 55.5, 60.0, 65.5):
        assert total.pmf(centre) is not None, centre
    assert abs(sum(k * v for k, v in total.pmf(55.5).probs.items()) - 55.5) < 3.0       # centred on its line


def test_a_missing_table_file_is_a_loud_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        landing.load_tables(tmp_path)
