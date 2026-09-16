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
