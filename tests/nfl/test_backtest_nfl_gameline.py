import importlib.util, pathlib
import pandas as pd
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig
from sportsmodel.nfl.gameline import GameLineConfig

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_nfl_gameline.py"
_s = importlib.util.spec_from_file_location("backtest_nfl_gameline", _p)
bt = importlib.util.module_from_spec(_s); _s.loader.exec_module(bt)

def _season(year, rows):
    out = []
    for w, (h, a, hs, as_, sl, tl) in enumerate(rows):
        out.append({"season": year, "week": w + 1, "home_team": h, "away_team": a,
                    "home_score": hs, "away_score": as_, "result": hs - as_,
                    "total": hs + as_, "spread_line": sl, "total_line": tl})
    return out

SCHED = pd.DataFrame(
    _season(2018, [("A","B",24,20,2.5,45.5), ("C","D",30,10,6.5,44.5),
                   ("A","C",21,17,1.5,42.5), ("B","D",14,13,-1.5,41.5),
                   ("A","D",28,7,7.5,46.5), ("B","C",20,24,-3.5,43.5)])
    + _season(2019, [("B","A",17,21,-2.5,44.5), ("D","C",10,20,-6.5,43.5),
                     ("C","A",13,16,-1.5,42.5)]))

def test_run_backtest_returns_metrics():
    m = bt.run_backtest(SCHED, EloConfig(), BlendConfig(), GameLineConfig())
    assert set(m) >= {"margin_mae", "total_mae", "brier", "cover_acc", "ou_acc", "n"}
    assert m["n"] > 0 and 0.0 <= m["brier"] <= 1.0

def test_run_backtest_deterministic():
    a = bt.run_backtest(SCHED, EloConfig(), BlendConfig(), GameLineConfig())
    b = bt.run_backtest(SCHED, EloConfig(), BlendConfig(), GameLineConfig())
    assert a == b

def test_no_leak_future_game_does_not_change_past_prediction():
    # Appending a LATER-week game must not change an earlier game's prediction error.
    base = bt.run_backtest(SCHED.iloc[:5].copy(), EloConfig(), BlendConfig(), GameLineConfig())
    extra = pd.concat([SCHED.iloc[:5], SCHED.iloc[[5]]], ignore_index=True)
    withfuture = bt.run_backtest(extra, EloConfig(), BlendConfig(), GameLineConfig())
    # the first 5 games' contribution is identical; only n and sums grow by the 6th game
    # (checked via per-game predictions the harness exposes)
    assert bt.per_game_predictions(SCHED.iloc[:5].copy(), EloConfig(), BlendConfig(), GameLineConfig()) \
        == bt.per_game_predictions(extra, EloConfig(), BlendConfig(), GameLineConfig())[:5]
