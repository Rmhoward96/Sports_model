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


def test_monday_job_refreshes_game_data_commits_four_parquets_then_rebuilds_the_panel_tables():
    from tests.test_workflows_props_ml import runs, step_index, steps_of
    (job,) = load(NAME)["jobs"].values()
    steps = steps_of(job)
    adv = step_index(steps, runs("scripts/build_cfb_advanced.py"))
    pull = step_index(steps, runs("scripts/build_cfb_game_data.py"))
    commit = step_index(steps, runs("git commit"))
    panels = step_index(steps, runs("scripts/build_cfb_panels.py"))
    assert adv < pull < commit < panels                       # the assets are committed before the DB rebuild
    assert steps[pull]["env"]["CFBD_API_KEY"] == "${{ secrets.CFBD_API_KEY }}"
    assert 'scripts/build_cfb_game_data.py --datasets games havoc team_stats --seasons "$(date +%Y)"' in steps[pull]["run"]
    for f in ("advanced_games", "cfbd_games", "havoc_games", "team_game_stats"):
        assert f"assets/cfb/{f}.parquet" in steps[commit]["run"]
    assert steps[panels]["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert "CFBD_API_KEY" not in steps[panels].get("env", {})


def test_commit_step_runs_even_if_the_game_data_pull_fails_and_the_gap_is_documented():
    from tests.test_workflows_props_ml import runs, step_index, steps_of
    (job,) = load(NAME)["jobs"].values()
    steps = steps_of(job)
    commit = steps[step_index(steps, runs("git commit"))]
    assert commit["if"] == "${{ !cancelled() }}"              # the advanced_games refresh still commits
    pull = steps[step_index(steps, runs("scripts/build_cfb_game_data.py"))]
    assert "January/February" in pull["run"] or "January/February" in (WF / NAME).read_text()
