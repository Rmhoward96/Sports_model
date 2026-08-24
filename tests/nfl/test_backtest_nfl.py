import importlib.util
import pathlib

import pandas as pd

from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_nfl_elo.py"
_s = importlib.util.spec_from_file_location("backtest_nfl_elo", _p)
bt = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bt)


def _season(year, rows):
    return [{"season": year, "week": w + 1, "home_team": h, "away_team": a,
             "home_score": hs, "away_score": as_}
            for w, (h, a, hs, as_) in enumerate(rows)]


SCHED = pd.DataFrame(
    _season(2018, [("A", "B", 24, 20), ("C", "D", 30, 10), ("A", "C", 21, 17),
                   ("B", "D", 14, 13), ("A", "D", 28, 7), ("B", "C", 20, 24)])
    + _season(2019, [("B", "A", 17, 21), ("D", "C", 10, 20), ("C", "A", 13, 16)])
)


def test_run_backtest_returns_metrics():
    m = bt.run_backtest(SCHED, EloConfig(), BlendConfig())
    assert set(m) >= {"brier", "win_acc", "margin_mae", "margin_rmse", "n"}
    assert 0.0 <= m["brier"] <= 1.0
    assert m["n"] > 0


def test_run_backtest_deterministic():
    a = bt.run_backtest(SCHED, EloConfig(), BlendConfig())
    b = bt.run_backtest(SCHED, EloConfig(), BlendConfig())
    assert a == b


def test_tune_returns_config_from_grid():
    grid = {"k": [15, 20], "hfa_elo": [65], "carryover": [0.75],
            "w_sos": [0.0, 0.3], "srs_min_games": [4]}
    best, results = bt.tune(SCHED, SCHED, grid)
    assert isinstance(best[0], EloConfig) and isinstance(best[1], BlendConfig)
    assert len(results) == 4   # 2 k x 2 w_sos
