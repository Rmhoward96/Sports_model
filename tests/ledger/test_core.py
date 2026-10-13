"""Pick key, observation record and the append rule. Pure -- no DB, no network."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel.ledger import core

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
KICK = NOW + timedelta(hours=26)


def _obs(**kw):
    base = dict(source="ev_prop", sport="nfl", game_pk=401, market="rec_yds", side="over", commence_time=KICK,
                line=64.5, player_id="00-0036", player_name="A. Brown", model_prob=0.58, best_price=-110,
                best_book="draftkings", ref_prob=0.52, ref_kind=None, ev_pct=6.0)
    return core.observation(**{**base, **kw})


def _last(**kw):
    base = dict(best_price=-110, line=64.5, model_prob=0.58, ref_prob=0.52)
    return {**base, **kw}


def plan(cands, last=None, now=NOW):
    return core.plan_appends(cands, last or {}, now=now, writer="w", writer_run="r1")


# ---- pick_key ------------------------------------------------------------------------------------

def test_pick_key_is_deterministic_and_ignores_price_and_line():
    a, b = _obs(), _obs(best_price=+120, line=66.5, model_prob=0.7, best_book="fanduel")
    assert a["pick_key"] == b["pick_key"] == "ev_prop|nfl|401|rec_yds|over|00-0036||"


def test_pick_key_separates_every_identity_component():
    base = _obs()["pick_key"]
    for change in (dict(side="under"), dict(market="receptions"), dict(game_pk=402), dict(sport="mlb"),
                   dict(player_id="00-0099"), dict(source="parlay_leg", parlay_id="dk|g:1")):
        assert _obs(**change)["pick_key"] != base, change


def test_pick_key_includes_model_version_only_for_model_game():
    mg = lambda v: core.observation(source="model_game", sport="nfl", game_pk=1, market="spread", side="home",
                                    commence_time=KICK, model_version=v)["pick_key"]
    assert mg("sim-nfl-ml-v2") != mg("elo-v1") and mg("elo-v1").endswith("|elo-v1|")
    assert _obs(model_version="props-sim-v1")["pick_key"] == _obs(model_version="props-sim-v2")["pick_key"]


def test_pick_key_normalises_integral_game_pks():
    assert _obs(game_pk=401.0)["pick_key"] == _obs(game_pk=401)["pick_key"] == _obs(game_pk="401")["pick_key"]


@pytest.mark.parametrize("change", [dict(market="a|b"), dict(side="over|x")])
def test_pick_key_rejects_a_pipe_in_market_or_side(change):
    with pytest.raises(ValueError):
        _obs(**change)


def test_pick_key_rejects_an_unknown_source_and_observation_an_unknown_ref_kind():
    with pytest.raises(ValueError):
        core.pick_key("desk", "nfl", 1, "spread", "home")
    with pytest.raises(ValueError):
        _obs(ref_kind="vibes")


def test_observation_coerces_numeric_types():
    o = _obs(game_pk="401", best_price="-115", line=64, model_prob=0)
    assert o["game_pk"] == 401 and o["best_price"] == -115 and o["line"] == 64.0 and o["model_prob"] == 0.0


# ---- append rule ---------------------------------------------------------------------------------

def test_first_sighting_is_appended_with_timing_and_provenance():
    rows, counts = plan([_obs()])
    (r,) = rows
    assert counts == {"first": 1}
    assert r["observed_at"] == NOW and r["commence_time"] == KICK and r["minutes_to_kick"] == 26 * 60
    assert r["writer"] == "w" and r["writer_run"] == "r1"


def test_unchanged_pick_is_not_appended():
    o = _obs()
    rows, counts = plan([o], {o["pick_key"]: _last()})
    assert rows == [] and counts == {"unchanged": 1}


@pytest.mark.parametrize("cand, why", [
    (dict(best_price=-105), "price"),
    (dict(best_price=None), "price"),
    (dict(line=65.5), "line"),
    (dict(line=None), "line"),
    (dict(model_prob=0.59), "prob"),          # exactly 1.0pp (float noise must not hide it)
    (dict(model_prob=0.57), "prob"),
    (dict(ref_prob=0.53), "prob"),
    (dict(ref_prob=None), "prob"),
])
def test_price_line_or_one_point_probability_move_appends(cand, why):
    o = _obs(**cand)
    rows, counts = plan([o], {o["pick_key"]: _last()})
    assert counts == {why: 1} and len(rows) == 1


@pytest.mark.parametrize("cand", [dict(model_prob=0.5899), dict(model_prob=0.5701), dict(ref_prob=0.5249),
                                  dict(line=64.5 + 1e-12)])
def test_sub_threshold_moves_are_ignored(cand):
    o = _obs(**cand)
    rows, counts = plan([o], {o["pick_key"]: _last()})
    assert rows == [] and counts == {"unchanged": 1}


def test_the_comparison_is_against_the_keys_last_row_not_the_first():
    o = _obs(model_prob=0.585)
    assert plan([o], {o["pick_key"]: _last(model_prob=0.58)})[0] == []                 # 0.5pp from last
    assert len(plan([o], {o["pick_key"]: _last(model_prob=0.572)})[0]) == 1           # 1.3pp from last


def test_picks_for_started_games_or_without_a_kickoff_are_never_recorded():
    started = _obs(commence_time=NOW - timedelta(minutes=1))
    at_kick = _obs(player_id="2", commence_time=NOW)
    nokick = _obs(player_id="3", commence_time=None)
    garbled = _obs(player_id="4", commence_time="soon")
    rows, counts = plan([started, at_kick, nokick, garbled])
    assert rows == [] and counts == {"started": 2, "invalid": 2}


def test_duplicate_keys_in_one_batch_keep_the_last_candidate():
    rows, _ = plan([_obs(best_price=-110), _obs(best_price=-120)])
    assert [r["best_price"] for r in rows] == [-120]


def test_string_kickoffs_and_naive_datetimes_are_read_as_utc():
    rows, _ = plan([_obs(commence_time="2026-10-15T17:00:00Z"),
                    _obs(player_id="9", commence_time=datetime(2026, 10, 15, 17))])
    assert {r["commence_time"] for r in rows} == {datetime(2026, 10, 15, 17, tzinfo=timezone.utc)}


def test_a_malformed_candidate_is_counted_as_skipped_and_the_rest_still_plan():
    rows, counts = plan([_obs(), {"no": "pick_key"}, None])
    assert len(rows) == 1 and counts == {"first": 1, "skipped": 2}


def test_pick_key_requires_sport_market_and_side():
    for kw in ({"sport": None}, {"market": None}, {"side": ""}):
        args = {"source": "ev_game", "sport": "nfl", "game_pk": 1, "market": "spread", "side": "home", **kw}
        with pytest.raises(ValueError):
            core.pick_key(**args)


@pytest.mark.parametrize("kw", [{"sport": "nf|l"}, {"player_id": "00|1"}, {"model_version": "v|2"}])
def test_pick_key_rejects_the_separator_in_every_free_text_part(kw):
    args = {"source": "model_game", "sport": "nfl", "game_pk": 1, "market": "spread", "side": "home", **kw}
    with pytest.raises(ValueError):
        core.pick_key(**args)


def test_pick_key_rejects_the_separator_in_model_version_for_every_source():
    with pytest.raises(ValueError):
        core.pick_key("ev_game", "nfl", 1, "spread", "home", model_version="a|b")


def test_pick_key_rejects_a_non_integral_game_pk_but_accepts_whole_numbers():
    for bad in (401.5, "401.5", float("nan"), float("inf"), None, True):
        with pytest.raises((ValueError, TypeError)):
            core.pick_key("ev_game", "nfl", bad, "spread", "home")
    assert core.pick_key("ev_game", "nfl", 401.0, "spread", "home") == core.pick_key("ev_game", "nfl", 401, "spread", "home")
    assert core.pick_key("ev_game", "nfl", "401", "spread", "home").split("|")[2] == "401"


def test_observation_with_a_fractional_game_pk_is_rejected_not_truncated():
    with pytest.raises(ValueError):
        core.observation(source="ev_game", sport="nfl", game_pk=401.5, market="spread", side="home",
                         commence_time=KICK)
