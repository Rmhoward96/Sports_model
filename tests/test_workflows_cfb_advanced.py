"""Structural checks of .github/workflows/build-cfb-advanced.yml (final-review fix wave):
the seasons input description matches the behavior (blank = current season) and a
concurrency group serializes runs without cancelling an in-flight pull."""
from __future__ import annotations

from tests.test_workflows_props_ml import WF, load

NAME = "build-cfb-advanced.yml"


def test_seasons_description_matches_blank_behavior():
    wf = load(NAME)
    desc = wf["on"]["workflow_dispatch"]["inputs"]["seasons"]["description"]
    assert "current season" in desc and "2015 through" not in desc
    assert "$(date +%Y)" in (WF / NAME).read_text()   # blank -> current year


def test_concurrency_group_does_not_cancel():
    wf = load(NAME)
    assert wf["concurrency"]["group"] == "build-cfb-advanced"
    assert str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
