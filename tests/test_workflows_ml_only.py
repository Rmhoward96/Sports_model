"""Structural checks of the ML-only NFL workflow wiring (ML-only NFL Task 3).

The LLM Decision Desk is retired: injury-watch.yml re-runs model -> sim (NFL)
-> +EV board -> +EV parlays -> snapshot record, with no desk bundle /
synthesis / write steps in either job, and the snapshot is recorded from
injury_watch.py's own injury source (no `--bundle`). The desk workflow files
themselves are kept (disabled by the controller after merge, not deleted).

Reuses the tiny block-YAML reader from test_workflows_props_ml (pyyaml is not
a project dependency). Nothing here touches the network.
"""
from __future__ import annotations

import pytest

from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

DESK_SCRIPTS = ("desk_inputs.py", "synthesize_desk_picks.py", "write_desk_picks.py")
CHANGED = "steps.check.outputs.changed == 'true'"


@pytest.mark.parametrize("job", ["nfl", "cfb"])
def test_injury_watch_job_has_no_desk_steps(job):
    steps = steps_of(load("injury-watch.yml")["jobs"][job])
    for s in steps:
        run = s.get("run") or ""
        assert not any(script in run for script in DESK_SCRIPTS), s.get("name")
        assert "desk_bundle" not in run, s.get("name")
        assert "--bundle" not in run, s.get("name")
        assert "ANTHROPIC_API_KEY" not in (s.get("env") or {}), s.get("name")
        assert "DESK_SYNTH_MODEL" not in (s.get("env") or {}), s.get("name")


@pytest.mark.parametrize("job", ["nfl", "cfb"])
def test_injury_watch_record_step_uses_own_injury_source(job):
    steps = steps_of(load("injury-watch.yml")["jobs"][job])
    rec = steps[step_index(steps, runs("--record"))]
    assert rec["run"] == 'uv run python scripts/injury_watch.py --sport "$SPORT" --record'
    assert rec["if"] == CHANGED
    # --record re-fetches injuries exactly as --check does, so it needs the key
    assert rec["env"]["SPORTSDATA_API_KEY"] == "${{ secrets.SPORTSDATA_API_KEY }}"
    assert rec["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"


@pytest.mark.parametrize("job,chain", [
    ("nfl", ["--check", "generate_nfl.py", "generate_sim_nfl.py",
             "build_ev_board.py", "build_best_parlays.py", "--record"]),
    ("cfb", ["--check", "generate_cfb.py",
             "build_ev_board.py", "build_best_parlays.py", "--record"]),
])
def test_injury_watch_rerun_chain_order(job, chain):
    steps = steps_of(load("injury-watch.yml")["jobs"][job])
    idx = [step_index(steps, runs(script)) for script in chain]
    assert idx == sorted(idx)
    for i in idx[1:]:
        assert CHANGED in steps[i]["if"]


def test_injury_watch_text_has_no_desk_pipeline():
    text = (WF / "injury-watch.yml").read_text()
    for script in DESK_SCRIPTS:
        assert script not in text
    assert "desk_bundle.json" not in text
    assert "ANTHROPIC_API_KEY" not in text


def test_desk_workflow_files_are_kept():
    """Retired, not deleted: the controller disables these after merge."""
    for name in ("desk-auto-nfl.yml", "desk-auto-cfb.yml", "desk-inputs.yml",
                 "write-desk-picks.yml", "grade-desk-picks.yml"):
        assert (WF / name).exists(), name


# ---- ML_GAME_LINES switch (ML-only NFL Task 4) ------------------------------------------

ML_GAME_LINES_ENV = "${{ vars.ML_GAME_LINES || 'off' }}"


@pytest.mark.parametrize("wf,job", [("generate-sim-nfl.yml", "generate"), ("injury-watch.yml", "nfl")])
def test_ml_game_lines_passed_from_repo_variable_default_off(wf, job):
    assert load(wf)["jobs"][job]["env"]["ML_GAME_LINES"] == ML_GAME_LINES_ENV


def test_cfb_injury_watch_job_has_no_ml_game_lines():
    assert "ML_GAME_LINES" not in (load("injury-watch.yml")["jobs"]["cfb"].get("env") or {})


def test_injury_watch_generate_nfl_runs_only_when_ml_game_lines_not_on():
    steps = steps_of(load("injury-watch.yml")["jobs"]["nfl"])
    step = steps[step_index(steps, runs("generate_nfl.py"))]
    assert step["if"] == f"{CHANGED} && env.ML_GAME_LINES != 'on'"
    # the sim step (which writes the ML game lines when on) still runs on every change
    sim = steps[step_index(steps, runs("generate_sim_nfl.py"))]
    assert sim["if"] == CHANGED


# ---- final fix wave: daily cron, concurrency, case-insensitive conditions ----------------

DAILY_CRON = "30 13 * * *"


def test_generate_sim_nfl_keeps_its_triggers_and_adds_the_daily_cron():
    """Ruling (b): with generate-nfl disabled, a daily run advances the slate
    to the new week and refreshes kickoffs / lines (13:30 UTC, after Tuesday's
    12:00 refresh-nfl-snapshots commit)."""
    on = load("generate-sim-nfl.yml")["on"]
    assert on["schedule"] == [{"cron": "0 21 * * 0,3,6"}, {"cron": DAILY_CRON}]
    assert on["workflow_dispatch"] == {}


def test_generate_sim_nfl_daily_cron_runs_only_when_ml_game_lines_on():
    """ML_GAME_LINES off => unchanged: the daily cron skips the job; the
    Wed/Sat/Sun cron and dispatches (github.event.schedule unset) always run,
    with the same job env (repo-variable defaults) for every trigger."""
    job = load("generate-sim-nfl.yml")["jobs"]["generate"]
    assert job["if"] == f"github.event.schedule != '{DAILY_CRON}' || vars.ML_GAME_LINES == 'on'"
    assert job["env"] == {"SIM_ML_MODE": "${{ vars.SIM_ML_MODE || 'off' }}",
                          "ML_GAME_LINES": ML_GAME_LINES_ENV}


def test_generate_sim_nfl_shares_injury_watch_nfl_concurrency_group():
    """M2: both write the ML slate and game_predictions, so they serialize."""
    sim = load("generate-sim-nfl.yml")["jobs"]["generate"]["concurrency"]
    watch = load("injury-watch.yml")["jobs"]["nfl"]["concurrency"]
    assert sim == watch == {"group": "injury-pipeline-nfl", "cancel-in-progress": "false"}


def test_no_workflow_uses_to_lower():
    """Ruling (c): Actions `==`/`!=` string comparisons already ignore case,
    and the expression language has no toLower() -- a call to it fails the
    run, so none may appear."""
    for path in sorted(WF.glob("*.y*ml")):
        text = path.read_text()
        assert "tolower(" not in text.lower(), path.name
