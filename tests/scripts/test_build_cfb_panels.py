"""build_cfb_panels.py: assets in, two table row lists out (no network, no DB)."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

from tests.cfb.test_panels import _fixture, _game

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_panels.py"
_s = importlib.util.spec_from_file_location("build_cfb_panels", _p)
bcp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bcp)


def _write(tmp_path, team_stats=True):
    hv, ad, ts = _fixture()
    hv.to_parquet(tmp_path / "havoc_games.parquet")
    ad.to_parquet(tmp_path / "advanced_games.parquet")
    pd.DataFrame([_game(2026, "1", "2", [7, 7, 7, 7], [3, 3, 3, 3]), _game(2025, "3", "4", [0, 7, 0, 7], [7, 0, 7, 0])]).to_parquet(
        tmp_path / "cfbd_games.parquet")
    if team_stats:
        ts.to_parquet(tmp_path / "team_game_stats.parquet")


def test_build_defaults_to_the_latest_havoc_season_and_filters_to_fbs(tmp_path):
    _write(tmp_path)
    season, ins, shares = bcp.build(tmp_path)
    assert season == 2026 and set(ins["team"]) == {"1", "2", "3", "4", "5", "6"} and set(shares["team"]) == {"1", "2", "3", "4"}
    _, ins2, shares2 = bcp.build(tmp_path, fbs={"1", "2"})
    assert set(ins2["team"]) == {"1", "2"} and set(shares2["team"]) == {"1", "2"}


def test_build_without_team_stats_warns_and_nulls_turnovers(tmp_path, capsys):
    _write(tmp_path, team_stats=False)
    _, ins, _ = bcp.build(tmp_path)
    assert ins["turnover_margin"].isna().all() and "team_game_stats.parquet missing" in capsys.readouterr().out


def test_build_missing_required_asset_exits(tmp_path):
    with pytest.raises(SystemExit):
        bcp.build(tmp_path)


def test_main_dry_run_and_upsert(tmp_path, monkeypatch, capsys):
    _write(tmp_path)
    monkeypatch.setattr(bcp, "ASSETS", tmp_path)
    monkeypatch.setattr(bcp, "load_fbs_ids", lambda: {"1", "2", "3", "4"})
    sent = {}
    monkeypatch.setattr(bcp.db, "upsert_cfb_team_insights", lambda rows: sent.setdefault("ins", rows) and len(rows))
    monkeypatch.setattr(bcp.db, "upsert_cfb_quarter_shares", lambda rows: sent.setdefault("sh", rows) and len(rows))
    bcp.main(["--dry-run"])
    assert not sent and "team_insights=4" in capsys.readouterr().out
    monkeypatch.setattr(bcp.config, "DATABASE_URL", "postgres://x")
    bcp.main(["--season", "2026"])
    assert {r["team"] for r in sent["ins"]} == {"1", "2", "3", "4"} and sent["sh"][0]["games_used"] >= 1
    monkeypatch.setattr(bcp.config, "DATABASE_URL", None)
    with pytest.raises(SystemExit):
        bcp.main([])
