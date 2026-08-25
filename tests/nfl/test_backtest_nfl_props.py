import importlib.util, pathlib
import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_nfl_props.py"
_s = importlib.util.spec_from_file_location("backtest_nfl_props", _p)
bt = importlib.util.module_from_spec(_s); _s.loader.exec_module(bt)

def _wk(pid, team, pos, season, wk, **k):
    base = {"player_id":pid,"player_name":pid,"recent_team":team,"position":pos,
            "season":season,"week":wk,"attempts":0,"passing_yards":0,"passing_tds":0,
            "targets":0,"receptions":0,"receiving_yards":0,"receiving_tds":0,
            "carries":0,"rushing_yards":0,"rushing_tds":0}
    base.update(k); return base

def _dataset():
    rows=[]
    for season in (2018,2019):
        for wk in range(1,4):
            rows += [_wk("qb","KC","QB",season,wk,attempts=34,passing_yards=255,passing_tds=2),
                     _wk("wr","KC","WR",season,wk,targets=9,receptions=6,receiving_yards=80),
                     _wk("rb","KC","RB",season,wk,carries=16,rushing_yards=70),
                     _wk("qb2","LA","QB",season,wk,attempts=38,passing_yards=270),
                     _wk("wr2","LA","WR",season,wk,targets=8,receptions=5,receiving_yards=70)]
    weekly=pd.DataFrame(rows)
    sched=pd.DataFrame([{"season":s,"week":w,"home_team":"KC","away_team":"LA",
                         "spread_line":-2.0,"total_line":48.0} for s in (2018,2019) for w in range(1,4)])
    return weekly, sched

def test_run_backtest_returns_per_market_metrics():
    weekly, sched = _dataset()
    m = bt.run_backtest(weekly, sched, seasons=[2019])   # 2019 projected from 2018
    assert "pass_yds" in m and "mae" in m["pass_yds"] and m["pass_yds"]["n"] > 0

def test_fit_calibration_returns_sigmas():
    weekly, sched = _dataset()
    preds = bt.per_player_predictions(weekly, sched, seasons=[2019])
    cal = bt.fit_calibration(preds)
    assert "sigma" in cal and "pass_yds" in cal["sigma"] and cal["sigma"]["pass_yds"] > 0

def test_no_leak_uses_prior_season_only():
    weekly, sched = _dataset()
    # dropping 2019's own rows must NOT change 2019 predictions (they use 2018 rates)
    base = bt.per_player_predictions(weekly, sched, seasons=[2019])
    trimmed = weekly[(weekly["season"]==2018) | (weekly["week"]<=1)]
    also = bt.per_player_predictions(trimmed, sched, seasons=[2019])
    # week-1 2019 predictions identical (both computed from full 2018)
    b1 = [p for p in base if p["season"]==2019 and p["week"]==1]
    a1 = [p for p in also if p["season"]==2019 and p["week"]==1]
    assert b1 == a1
