"""Structural checks of the 'take these' workflows (plain text; no network)."""
from pathlib import Path

import pytest

WF = Path(__file__).resolve().parents[2] / ".github" / "workflows"

CASES = [
    ("take-these-pull.yml", None, "take-these-pull", "timeout-minutes: 8", "scripts/take_pull.py"),
    ("take-these-close.yml", None, "take-these-close", "timeout-minutes: 5", "scripts/take_close.py"),
    ("take-these-scores.yml", '"30 9 * * *"', "take-these-scores", "timeout-minutes: 5", "scripts/take_scores.py"),
    ("take-these-watch.yml", '"7 */2 * * *"', "take-these-watch", "timeout-minutes: 355", "scripts/take_watch.py"),
]
TAKE_WORKFLOWS = ["take-these-pull.yml", "take-these-close.yml", "take-these-scores.yml", "take-these-watch.yml",
                  "report-take-these.yml"]


@pytest.mark.parametrize("name,cron,group,timeout,script", CASES)
def test_workflow_shape(name, cron, group, timeout, script):
    t = (WF / name).read_text()
    assert "workflow_dispatch:" in t
    if cron is None:
        assert "schedule:" not in t and "cron:" not in t          # take-these-watch runs these two on schedule
    else:
        assert f"- cron: {cron}" in t
    assert "if: vars.TAKE_THESE == 'on' || github.event_name == 'workflow_dispatch'" in t
    assert f"group: {group}\n  cancel-in-progress: false" in t and timeout in t
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in t and "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in t
    assert script in t and "dry_run:" in t


def test_pull_and_close_workflows_are_manual_only_with_a_dry_run_flag():
    t = (WF / "take-these-pull.yml").read_text()
    assert "scripts/take_pull.py --dry-run" in t and "--scheduled" not in t and "EVENT_NAME" not in t
    t = (WF / "take-these-close.yml").read_text()
    assert "scripts/take_close.py --dry-run" in t


def test_watch_workflow_passes_the_cadence_and_dry_run_and_reads_contents_only():
    t = (WF / "take-these-watch.yml").read_text()
    assert "TAKE_PULL_EVERY_HOURS: ${{ vars.TAKE_PULL_EVERY_HOURS }}" in t
    assert "DRY_RUN: ${{ inputs.dry_run }}" in t and 'if [ "$DRY_RUN" = "true" ]; then' in t
    assert "uv run python scripts/take_watch.py --dry-run" in t and "uv run python scripts/take_watch.py\n" in t
    assert "permissions:\n  contents: read\n" in t
    assert t.count("permissions:") == 1 and "contents: write" not in t and "actions: write" not in t
    assert "timeout-minutes: 355" in t and "default: false" in t


def test_watch_deadline_fits_inside_the_job_timeout():
    from sportsmodel.take import constants
    assert constants.WATCH_MINUTES + 5 <= 355                  # the deadline plus the checkout / sync before it


@pytest.mark.parametrize("name", TAKE_WORKFLOWS)
def test_no_take_workflow_dispatches_or_triggers_another_workflow(name):
    t = (WF / name).read_text()
    for banned in ("gh workflow", "gh run", "gh api", "repository_dispatch", "createWorkflowDispatch", "workflow_run",
                   "/dispatches", "actions/workflows", "workflow_call", "uses: ./"):
        assert banned not in t, banned


def test_report_workflow_is_weekly_gated_and_commits_with_the_retry_loop():
    t = (WF / "report-take-these.yml").read_text()
    assert '- cron: "0 13 * * 1"' in t and "workflow_dispatch:" in t
    assert "if: vars.TAKE_THESE == 'on' || github.event_name == 'workflow_dispatch'" in t
    assert "ODDS_API_KEY" not in t                                            # reads the database only, never spends credits
    assert "group: report-take-these\n  cancel-in-progress: false" in t and "permissions:\n  contents: write" in t
    assert "DATABASE_URL: ${{ secrets.DATABASE_URL }}" in t and "uv run python scripts/report_take_these.py" in t
    assert 'git commit -m "docs(report): take these [skip ci]"' in t and "for attempt in 1 2 3; do" in t
    assert "git add docs/superpowers/reports/" in t
