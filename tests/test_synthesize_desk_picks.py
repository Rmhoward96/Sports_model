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
