"""Test the SYSTEM_PROMPT in synthesize_desk_picks for trends integration."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).parents[1] / "scripts" / "synthesize_desk_picks.py"
_spec = importlib.util.spec_from_file_location("synthesize_desk_picks", _p)
syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(syn)


def test_system_prompt_mentions_trends():
    """SYSTEM_PROMPT must mention 'trends'."""
    assert "trends" in syn.SYSTEM_PROMPT.lower()


def test_system_prompt_mentions_analyst_notes():
    """SYSTEM_PROMPT must mention analyst notes."""
    prompt_lower = syn.SYSTEM_PROMPT.lower()
    assert "agent_notes.analyst" in prompt_lower or "analyst note" in prompt_lower


def test_system_prompt_mentions_never_the_sole():
    """SYSTEM_PROMPT must contain the phrase 'never the sole'."""
    assert "never the sole" in syn.SYSTEM_PROMPT.lower()
