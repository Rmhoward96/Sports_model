"""Team-based desk synthesis: the pick side is resolved from a TEAM NAME against
the game's own matchup, so the desk can never invert home/away (the KC/Denver
bug). PURE parts only -- no API calls."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).parents[1] / "scripts" / "synthesize_desk_picks.py"
_spec = importlib.util.spec_from_file_location("synthesize_desk_picks", _p)
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)


def _game(gpk=1, matchup="Denver Broncos @ Kansas City Chiefs",
          win_prob=0.436, market_spread=-2.5):
    return {
        "sport": "nfl", "game_pk": gpk, "matchup": matchup,
        "commence_time": "2026-09-15T00:15:00Z",
        "market_spread": market_spread, "market_total": 43.5,
        "model": {"margin": -1.65, "total": 45.0, "win_prob": win_prob},
    }


def test_resolve_side_exact_and_substring():
    assert syn._resolve_side("Kansas City Chiefs", "Denver Broncos", "Kansas City Chiefs") == "home"
    assert syn._resolve_side("Denver Broncos", "Denver Broncos", "Kansas City Chiefs") == "away"
    assert syn._resolve_side("Chiefs", "Denver Broncos", "Kansas City Chiefs") == "home"
    assert syn._resolve_side("Broncos", "Denver Broncos", "Kansas City Chiefs") == "away"
    assert syn._resolve_side("Los Angeles Rams", "Denver Broncos", "Kansas City Chiefs") is None
    assert syn._resolve_side(None, "Denver Broncos", "Kansas City Chiefs") is None


def test_enrich_sets_model_pick_from_win_prob():
    # home win_prob 0.436 -> the model favors the AWAY team (Denver)
    e = syn._enrich(_game(win_prob=0.436))
    assert e["home_team"] == "Kansas City Chiefs"
    assert e["away_team"] == "Denver Broncos"
    assert e["model_pick_team"] == "Denver Broncos"
    e2 = syn._enrich(_game(win_prob=0.564))
    assert e2["model_pick_team"] == "Kansas City Chiefs"


def test_desk_can_override_model_to_home_team_without_inversion():
    # THE REGRESSION: model favors away (Denver, win_prob 0.436) but the desk
    # backs Kansas City. The pick must resolve to home=KC, not away.
    bundle = [_game(win_prob=0.436, market_spread=-2.5)]
    dec = [{"game_pk": 1, "ml_pick_team": "Kansas City Chiefs",
            "spread_pick_team": "Kansas City Chiefs", "conviction_tier": "medium",
            "confidence": 0.58, "rationale": "KC", "agent_notes": {}}]
    picks = syn.assemble_picks(bundle, dec, "nfl")
    assert len(picks) == 1
    p = picks[0]
    assert p["ml_pick"] == "home"          # KC, not the model's away pick
    assert p["spread_side"] == "home"
    # spread_line is ALWAYS the home-referenced number (= market_spread)
    assert p["spread_line"] == -2.5
    assert p["total_side"] is None and p["total_line"] is None


def test_away_team_pick_keeps_home_referenced_spread_line():
    bundle = [_game(win_prob=0.564, market_spread=-2.5)]  # model favors home KC
    dec = [{"game_pk": 1, "ml_pick_team": "Denver Broncos",
            "spread_pick_team": "Denver Broncos", "conviction_tier": "low",
            "confidence": 0.54, "rationale": "Denver dog", "agent_notes": {}}]
    p = syn.assemble_picks(bundle, dec, "nfl")[0]
    assert p["ml_pick"] == "away"
    assert p["spread_side"] == "away"
    assert p["spread_line"] == -2.5  # home line, NOT flipped to +2.5


def test_unresolvable_team_falls_back_to_model_side():
    bundle = [_game(win_prob=0.436)]  # model favors away
    dec = [{"game_pk": 1, "ml_pick_team": "Chicago Bears",  # not in this game
            "spread_pick_team": None, "conviction_tier": "low",
            "confidence": 0.52, "rationale": "x", "agent_notes": {}}]
    p = syn.assemble_picks(bundle, dec, "nfl")[0]
    assert p["ml_pick"] == "away"          # fallback = model's favored side
    assert p["spread_side"] is None        # declined


def test_declined_spread_when_no_team_given():
    bundle = [_game()]
    dec = [{"game_pk": 1, "ml_pick_team": "Kansas City Chiefs",
            "spread_pick_team": None, "conviction_tier": "low",
            "confidence": 0.52, "rationale": "x", "agent_notes": {}}]
    p = syn.assemble_picks(bundle, dec, "nfl")[0]
    assert p["spread_side"] is None and p["spread_line"] is None


# --- flag_pick_issues (rationale/pick consistency guard) --------------------

def _bundle_with_form():
    # Denver (away) 4-1, Kansas City (home) 0-5 -- the real KC/Denver game.
    return [{
        "sport": "nfl", "game_pk": 1,
        "matchup": "Denver Broncos @ Kansas City Chiefs",
        "commence_time": "2026-09-15T00:15:00Z", "market_spread": -2.5,
        "model": {"win_prob": 0.436},
        "form": {"home": {"record": "0-5"}, "away": {"record": "4-1"}},
    }]


def test_guard_flags_record_misattribution():
    # rationale claims "Chiefs are 4-1" but KC is 0-5 (4-1 is Denver's record)
    bundle = _bundle_with_form()
    picks = [{"game_pk": 1, "ml_pick": "away", "matchup": bundle[0]["matchup"],
              "rationale": "Chiefs are 4-1 and roll over winless Denver."}]
    flags = syn.flag_pick_issues(picks, bundle)
    assert any("Kansas City Chiefs" in f and "4-1" in f for f in flags)


def test_guard_clean_when_records_correct():
    bundle = _bundle_with_form()
    picks = [{"game_pk": 1, "ml_pick": "away", "matchup": bundle[0]["matchup"],
              "rationale": "Broncos are 4-1 (+4.8) vs a winless Chiefs (0-5); back Denver."}]
    assert syn.flag_pick_issues(picks, bundle) == []


def test_guard_flags_picked_team_unmentioned():
    bundle = _bundle_with_form()
    # pick resolves to Denver (away) but the rationale only talks about KC
    picks = [{"game_pk": 1, "ml_pick": "away", "matchup": bundle[0]["matchup"],
              "rationale": "The Chiefs defense looks strong here."}]
    flags = syn.flag_pick_issues(picks, bundle)
    assert any("Denver Broncos" in f and "never names" in f for f in flags)


# --- demote_flagged_picks (escalation) -------------------------------------

def test_demote_flagged_pick_drops_lean_caps_tier_marks_rationale():
    picks = [
        {"game_pk": 1, "ml_pick": "away", "spread_side": "away", "spread_line": -2.5,
         "conviction_tier": "high", "rationale": "Chiefs are 4-1..."},
        {"game_pk": 2, "ml_pick": "home", "spread_side": "home", "spread_line": -6.0,
         "conviction_tier": "medium", "rationale": "clean pick"},
    ]
    flags = ["game 1: rationale ties Kansas City Chiefs to record 4-1 but its record is 0-5"]
    n = syn.demote_flagged_picks(picks, flags)
    assert n == 1
    # game 1 demoted
    assert picks[0]["spread_side"] is None and picks[0]["spread_line"] is None
    assert picks[0]["conviction_tier"] == "low"
    assert picks[0]["rationale"].startswith(syn._DEMOTE_MARKER)
    assert picks[0]["ml_pick"] == "away"  # ml_pick kept (contract)
    # game 2 untouched
    assert picks[1]["spread_side"] == "home" and picks[1]["conviction_tier"] == "medium"
    assert not picks[1]["rationale"].startswith(syn._DEMOTE_MARKER)


def test_demote_is_idempotent_on_rationale_marker():
    picks = [{"game_pk": 1, "ml_pick": "home", "spread_side": None, "spread_line": None,
              "conviction_tier": "low", "rationale": syn._DEMOTE_MARKER + "already marked"}]
    syn.demote_flagged_picks(picks, ["game 1: pick is X but the rationale never names it"])
    assert picks[0]["rationale"].count(syn._DEMOTE_MARKER) == 1


# --- apply_disagreement_cap (sim vs. model disagreement) -------------------

def _pick(gpk=1, tier="high", rationale="strong edge"):
    return {"game_pk": gpk, "ml_pick": "home", "spread_side": "home",
            "spread_line": -3.0, "conviction_tier": tier, "rationale": rationale}


def _bundle_with_sim(gpk=1, disagreement=0.25):
    return [{"game_pk": gpk, "matchup": "Denver Broncos @ Kansas City Chiefs",
              "sim": {"home_win_prob": 0.6, "margin": -3.0, "total": 44.0,
                       "disagreement": disagreement}}]


def test_high_conviction_above_threshold_downgraded_and_noted():
    picks = [_pick(tier="high")]
    bundle = _bundle_with_sim(disagreement=0.25)
    n = syn.apply_disagreement_cap(picks, bundle)
    assert n == 1
    assert picks[0]["conviction_tier"] == "medium"
    assert picks[0]["rationale"].startswith("[sim disagreement 0.25")
    assert picks[0]["rationale"].endswith("strong edge")


def test_high_conviction_at_or_below_threshold_unchanged():
    picks = [_pick(tier="high")]
    bundle = _bundle_with_sim(disagreement=0.15)  # exactly at threshold -> not capped
    n = syn.apply_disagreement_cap(picks, bundle)
    assert n == 0
    assert picks[0]["conviction_tier"] == "high"
    assert picks[0]["rationale"] == "strong edge"

    picks2 = [_pick(tier="high")]
    bundle2 = _bundle_with_sim(disagreement=0.05)
    assert syn.apply_disagreement_cap(picks2, bundle2) == 0
    assert picks2[0]["conviction_tier"] == "high"


def test_medium_and_low_unchanged_regardless_of_disagreement():
    picks = [_pick(gpk=1, tier="medium"), _pick(gpk=2, tier="low")]
    bundle = [_bundle_with_sim(gpk=1, disagreement=0.9)[0],
              _bundle_with_sim(gpk=2, disagreement=0.9)[0]]
    n = syn.apply_disagreement_cap(picks, bundle)
    assert n == 0
    assert picks[0]["conviction_tier"] == "medium"
    assert picks[1]["conviction_tier"] == "low"


def test_sim_absent_or_none_is_a_noop():
    # sim key entirely absent (e.g. a CFB bundle)
    picks = [_pick(gpk=1, tier="high")]
    bundle = [{"game_pk": 1, "matchup": "Denver Broncos @ Kansas City Chiefs"}]
    assert syn.apply_disagreement_cap(picks, bundle) == 0
    assert picks[0]["conviction_tier"] == "high"

    # sim explicitly None (NFL game the sim hasn't covered yet)
    picks2 = [_pick(gpk=1, tier="high")]
    bundle2 = [{"game_pk": 1, "matchup": "Denver Broncos @ Kansas City Chiefs", "sim": None}]
    assert syn.apply_disagreement_cap(picks2, bundle2) == 0
    assert picks2[0]["conviction_tier"] == "high"


def test_disagreement_cap_returns_correct_count_across_multiple_picks():
    picks = [_pick(gpk=1, tier="high"), _pick(gpk=2, tier="high"), _pick(gpk=3, tier="medium")]
    bundle = [_bundle_with_sim(gpk=1, disagreement=0.3)[0],
              _bundle_with_sim(gpk=2, disagreement=0.05)[0],
              _bundle_with_sim(gpk=3, disagreement=0.9)[0]]
    n = syn.apply_disagreement_cap(picks, bundle)
    assert n == 1
    assert picks[0]["conviction_tier"] == "medium"
    assert picks[1]["conviction_tier"] == "high"
    assert picks[2]["conviction_tier"] == "medium"  # unrelated to cap; already medium


def test_disagreement_cap_uses_custom_threshold():
    picks = [_pick(tier="high")]
    bundle = _bundle_with_sim(disagreement=0.2)
    assert syn.apply_disagreement_cap(picks, bundle, threshold=0.25) == 0
    assert syn.apply_disagreement_cap(picks, bundle, threshold=0.1) == 1


# --- F1: output-token overflow ---------------------------------------------
import json  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason


def _install_fake_anthropic(monkeypatch, responses):
    """Fake `anthropic` module whose client returns `responses` in order. Only
    `messages.stream` exists: at max_tokens > ~21k the real SDK refuses a
    non-streaming `messages.create` client-side ("Streaming is required ..."),
    so the desk must stream."""
    calls = []

    class _Stream:
        def __init__(self, resp):
            self._resp = resp

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return self._resp

    class _Messages:
        def stream(self, **kw):
            calls.append(kw)
            return _Stream(responses[len(calls) - 1])

    class _Client:
        def __init__(self, *a, **kw):
            self.messages = _Messages()

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _Client
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return calls


def test_max_tokens_defaults_to_32000_and_env_overrides(monkeypatch):
    monkeypatch.delenv("DESK_SYNTH_MAX_TOKENS", raising=False)
    assert syn._max_tokens() == 32000
    monkeypatch.setenv("DESK_SYNTH_MAX_TOKENS", "")
    assert syn._max_tokens() == 32000
    monkeypatch.setenv("DESK_SYNTH_MAX_TOKENS", "5000")
    assert syn._max_tokens() == 5000


def test_model_decisions_raises_clear_error_when_truncated(monkeypatch):
    _install_fake_anthropic(monkeypatch, [_Resp('[{"game_pk": 1, "ra', stop_reason="max_tokens")])
    with pytest.raises(RuntimeError, match=r"desk response truncated at max_tokens=123; raise DESK_SYNTH_MAX_TOKENS"):
        syn._model_decisions([_game()], "fake-model", 123)


def test_model_decisions_parses_normal_response(monkeypatch):
    calls = _install_fake_anthropic(monkeypatch, [_Resp('[{"game_pk": 1}]')])
    assert syn._model_decisions([_game()], "fake-model", 123) == [{"game_pk": 1}]
    assert calls[0]["max_tokens"] == 123


def test_repair_call_raises_clear_error_when_truncated(monkeypatch, tmp_path):
    # first response: valid contract but the rationale never names the picked
    # team -> consistency flag -> repair round; the repair response is truncated.
    first = json.dumps([{
        "game_pk": 1, "ml_pick_team": "Kansas City Chiefs", "spread_pick_team": None,
        "conviction_tier": "low", "confidence": 0.53, "rationale": "no team named here",
        "agent_notes": {"statistics": "s", "analyst": "a", "news": "n"}}])
    calls = _install_fake_anthropic(monkeypatch, [_Resp(first), _Resp("[{", stop_reason="max_tokens")])
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps([_game()]))
    out = tmp_path / "picks.json"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("DESK_SYNTH_MAX_TOKENS", "777")
    monkeypatch.setattr(sys, "argv", ["synth", "--sport", "nfl", "--bundle", str(bundle), "--out", str(out)])
    with pytest.raises(RuntimeError, match=r"truncated at max_tokens=777"):
        syn.main()
    assert len(calls) == 2
    assert not out.exists()
