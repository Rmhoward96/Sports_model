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


def test_system_prompt_ou_trends_context_only_and_totals_still_declined():
    """F5(a): O/U trends are context only because totals are declined."""
    p = " ".join(syn.SYSTEM_PROMPT.split())
    assert "O/U trends are context only because totals are declined" in p
    assert "Decline totals entirely" in p


def test_system_prompt_concrete_fact_is_straight_up_record():
    """F5(b): the required concrete fact is a straight-up record, not an ATS trend."""
    p = " ".join(syn.SYSTEM_PROMPT.split())
    assert "a specific straight-up record" in p


def test_system_prompt_asks_for_compact_trend_citations():
    """F1: compact, one-line-per-team trend citation format."""
    p = " ".join(syn.SYSTEM_PROMPT.split())
    assert "COMPACT" in p
    assert "one short line per team" in p
    assert "BUF ATS 2-1" in p
