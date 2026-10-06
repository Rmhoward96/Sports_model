"""Structural checks of the MLB (reactivated 2026-10-05) GitHub Actions wiring.

MLB-only crons exist for odds capture and the +EV game-line / prop boards (every day Mar-Nov), the
football crons keep pulling nfl,cfb only, and the three MLB-only workflows
(daily-ingest / refresh-props / refresh-profiles) are present for the controller to enable.
Plain text checks plus the tiny YAML reader from test_workflows_props_ml; no network.
"""
from __future__ import annotations

import re

import pytest

from tests.test_workflows_props_ml import WF, load


def text(name: str) -> str:
    return (WF / name).read_text()


def crons(name: str) -> list[str]:
    return re.findall(r'-\s*cron:\s*"([^"]+)"', text(name))


MLB_CAPTURE = ["30 16 * 3-11 *", "0 19 * 3-11 *", "30 21 * 3-11 *", "0 0 * 3-11 *", "30 2 * 3-11 *"]
MLB_BOARD = ["35 16 * 3-11 *", "5 19 * 3-11 *", "35 21 * 3-11 *", "5 0 * 3-11 *", "35 2 * 3-11 *"]
MLB_PROPS = ["37 16 * 3-11 *", "7 19 * 3-11 *", "37 21 * 3-11 *", "7 0 * 3-11 *", "37 2 * 3-11 *"]


def test_mlb_workflow_files_present():
    for name in ("daily-ingest.yml", "refresh-props.yml", "refresh-profiles.yml",
                 "capture-odds.yml", "build-ev-board.yml", "build-ev-props.yml",
                 "grade-predictions.yml", "grade-ev.yml", "grade-ev-props.yml"):
        assert (WF / name).exists(), name
        load(name)  # parses with the repo's block-YAML reader


def test_capture_odds_has_daily_mlb_only_crons_and_unchanged_football_crons():
    cs = crons("capture-odds.yml")
    for c in MLB_CAPTURE:
        assert c in cs
    for c in ("0 15 * * *", "0 16 * * 0,1,4,6", "0 18 * * 0,1,4,6", "0 20 * * 0,1,4,6", "0 23 * * 0,1,4,6"):
        assert c in cs                       # football/slate cadence untouched
    assert len(cs) == len(set(cs))           # no two firings share a cron string (the case below keys on it)


def test_capture_odds_routes_sports_by_schedule():
    t = text("capture-odds.yml")
    # the case statement keys on exactly the MLB-only crons, and everything else is football or all
    for c in MLB_CAPTURE:
        assert f'"{c}"' in t
    assert 'sports="mlb"' in t and 'sports="nfl,cfb,mlb"' in t and 'sports="nfl,cfb"' in t
    assert 'export ODDS_SPORTS="${SPORTS_INPUT:-$sports}"' in t
    assert "scripts/ingest_odds.py" in t
    # an MLB-only firing must not inherit a blank PROP_WINDOW_MIN (blank == props off)
    assert 'export PROP_WINDOW_MIN="${PROP_WINDOW_MIN:-150}"' in t
    assert "sports:" in t                    # manual dispatch input


def test_ev_board_builds_mlb_on_mlb_crons_only_and_football_on_football_crons():
    t = text("build-ev-board.yml")
    cs = crons("build-ev-board.yml")
    for c in MLB_BOARD:
        assert c in cs
    assert "build_ev_board.py --sport mlb" in t
    assert "build_ev_board.py --sport nfl" in t and "build_ev_board.py --sport cfb" in t
    # football steps are skipped on MLB firings, the MLB step on football firings
    nfl_step = t[t.index("Build +EV board (NFL)"):t.index("Build +EV board (CFB)")]
    mlb_step = t[t.index("Build +EV board (MLB)"):t.index("Build +EV parlays")]
    assert "env.MLB_FIRING != 'true'" in nfl_step
    assert "env.FOOTBALL_FIRING != 'true'" in mlb_step
    for c in MLB_BOARD:
        assert f"'{c}'" in t                 # MLB_FIRING lists every MLB cron
    for c in ("30 20 * * 0,1,4,6", "30 23 * * 0,1,4,6"):
        assert f"'{c}'" in t                 # FOOTBALL_FIRING lists the football crons


def test_ev_board_mlb_crons_trail_the_capture_crons_by_minutes_not_hours():
    # board minute > capture minute within the same hour -> the board reads the fresh pull
    for cap, board in zip(MLB_CAPTURE, MLB_BOARD):
        cm, ch = cap.split()[:2]
        bm, bh = board.split()[:2]
        assert ch == bh and int(bm) > int(cm), (cap, board)


def test_ev_props_builds_mlb_straight_props_only_and_not_parlays():
    t = text("build-ev-props.yml")
    for c in MLB_PROPS:
        assert c in crons("build-ev-props.yml")
    assert "build_ev_props_board.py --sport mlb" in t
    for c in MLB_PROPS:
        assert f"'{c}'" in t                 # MLB_FIRING lists every MLB prop cron (incl. the late 02:37 run)
    nfl_step = t[t.index("Build NFL prop +EV board"):t.index("Build MLB prop +EV board")]
    assert "env.MLB_FIRING != 'true'" in nfl_step
    parlay_step = t[t.index("Build +EV parlays"):]
    assert "env.MLB_FIRING != 'true'" in parlay_step   # MLB prop picks are never parlay legs


def test_grading_workflows_cover_mlb_through_their_scripts():
    # grade_predictions / grade_ev / grade_ev_props loop the sports themselves.
    for name, script in (("grade-predictions.yml", "grade_predictions.py"), ("grade-ev.yml", "grade_ev.py"),
                         ("grade-ev-props.yml", "grade_ev_props.py")):
        assert script in text(name)
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "scripts"
    for script in ("grade_predictions", "grade_ev"):
        spec = importlib.util.spec_from_file_location(f"{script}_wf", root / f"{script}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert "mlb" in mod.FINAL_PROVIDERS


def test_mlb_only_workflows_run_the_sim_and_profiles():
    assert "scripts/daily_ingest.py" in text("daily-ingest.yml") and "scripts/generate_sim.py" in text("daily-ingest.yml")
    assert "scripts/generate_sim.py" in text("refresh-props.yml")
    rp = text("refresh-profiles.yml")
    assert "scripts/build_profiles.py" in rp and "scripts/backfill_mlb.py" in rp
    assert "workflow_dispatch" in rp         # the first post-merge run is manual (profiles are weeks stale)
