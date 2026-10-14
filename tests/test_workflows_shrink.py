"""Structural checks of .github/workflows/fit-shrink-k.yml: Tuesday 15:30 UTC, env-passed secrets, timeout, serialised runs,
write permission, and the 3-attempt pull --rebase push retry. Reuses the block-YAML reader and the tiny cron evaluator."""
from datetime import timedelta

from tests.ledger.test_workflows_capture import cron_sets, fires, utc
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

FIT = "fit-shrink-k.yml"


def test_the_schedule_is_tuesdays_at_1530_utc_plus_manual_dispatch():
    wf = load(FIT)
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["30 15 * * 2"] and "workflow_dispatch" in wf["on"]
    t, week = utc(2026, 10, 12, 0, 0), []
    while t < utc(2026, 10, 19, 0, 0):
        if fires("30 15 * * 2", t):
            week.append(t)
        t += timedelta(minutes=1)
    assert week == [utc(2026, 10, 13, 15, 30)]


def test_it_runs_after_the_player_actuals_capture_and_before_the_early_monitor_report():
    m, h, mon, dow = cron_sets(load("capture-player-actuals.yml")["on"]["schedule"][0]["cron"])
    assert 2 in dow and 15 in h and m == {0}                                   # Tuesday 15:00 grades Monday night
    assert [c["cron"] for c in load("report-early-tickets.yml")["on"]["schedule"]] == ["52 15 * * 2"]


def test_the_job_has_a_timeout_write_permission_and_serialised_runs():
    wf = load(FIT)
    assert wf["permissions"] == {"contents": "write"}
    assert wf["concurrency"]["group"] == "fit-shrink-k" and str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
    (job,) = wf["jobs"].values()
    assert str(job["timeout-minutes"]) == "10"


def test_secrets_go_through_env_and_no_expression_is_interpolated_into_a_shell_script():
    (job,) = load(FIT)["jobs"].values()
    steps = steps_of(job)
    fit = steps[step_index(steps, runs("scripts/fit_shrink_k.py"))]
    assert fit["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}
    assert all("${{" not in st.get("run", "") for st in steps)
    text = (WF / FIT).read_text()
    assert "ODDS_API_KEY" not in text and "SM_LEDGER_OFF" not in text


def test_fit_runs_before_the_commit_and_the_commit_stages_only_the_two_outputs():
    (job,) = load(FIT)["jobs"].values()
    steps = steps_of(job)
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("fit_shrink_k.py")) < step_index(steps, runs("git commit"))
    commit = steps[step_index(steps, runs("git commit"))]["run"]
    assert "assets/shrink_k.json docs/superpowers/reports/*-shrink-gate.md" in commit and "[skip ci]" in commit
    assert "git add ." not in commit and "git add -A" not in commit


def test_the_push_retries_three_times_with_pull_rebase():
    (job,) = load(FIT)["jobs"].values()
    commit = steps_of(job)[step_index(steps_of(job), runs("git commit"))]["run"]
    assert "for attempt in 1 2 3" in commit and "git pull --rebase" in commit and "git push" in commit
    assert 'push failed after 3 attempts' in commit and commit.rstrip().endswith("exit 1")


def test_a_failed_fit_step_stops_the_job_before_any_commit():
    (job,) = load(FIT)["jobs"].values()
    steps = steps_of(job)
    for st in steps:
        assert "continue-on-error" not in st and "if" not in st, st               # default `success()`: no commit after a failed fit
