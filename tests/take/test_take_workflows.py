"""Structural checks of the two 'take these' workflows (plain text; no network)."""
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parents[2] / ".github" / "workflows"

CASES = [
    ("take-these-pull.yml", '"0 12-23,0-4 * * *"', "take-these-pull", "timeout-minutes: 8", "scripts/take_pull.py"),
    ("take-these-close.yml", '"3-59/10 * * * *"', "take-these-close", "timeout-minutes: 5", "scripts/take_close.py"),
]


@pytest.mark.parametrize("name,cron,group,timeout,script", CASES)
def test_workflow_shape(name, cron, group, timeout, script):
    t = (WF / name).read_text()
    assert f"- cron: {cron}" in t and "workflow_dispatch:" in t
    assert "if: vars.TAKE_THESE == 'on' || github.event_name == 'workflow_dispatch'" in t
    assert f"group: {group}\n  cancel-in-progress: false" in t and timeout in t
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in t and "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in t
    assert script in t and "dry_run:" in t


def test_pull_workflow_passes_the_cadence_variable_and_schedules_with_the_flag():
    t = (WF / "take-these-pull.yml").read_text()
    assert "TAKE_PULL_EVERY_HOURS: ${{ vars.TAKE_PULL_EVERY_HOURS }}" in t
    assert "scripts/take_pull.py --scheduled" in t and "scripts/take_pull.py --dry-run" in t
