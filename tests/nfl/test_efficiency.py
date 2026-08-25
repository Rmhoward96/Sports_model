import pandas as pd
from sportsmodel.nfl.efficiency import compute_efficiency

def _wk(pid, team, pos, wk, **k):
    base = {"player_id":pid,"recent_team":team,"position":pos,"week":wk,
            "attempts":0,"passing_yards":0,"passing_tds":0,"targets":0,"receptions":0,
            "receiving_yards":0,"receiving_tds":0,"carries":0,"rushing_yards":0,"rushing_tds":0}
    base.update(k); return base

def test_efficiency_rates_and_position_shrinkage():
    weekly = pd.DataFrame([
        # QB1 heavy sample ~8.0 ypa; QB2 tiny sample extreme 20 ypa -> pulled toward QB baseline
        _wk("qb1","KC","QB",1,attempts=40,passing_yards=320,passing_tds=3),
        _wk("qb1","KC","QB",2,attempts=40,passing_yards=320,passing_tds=1),
        _wk("qb2","LA","QB",1,attempts=2,passing_yards=40,passing_tds=0),  # 20 ypa, 2 att
    ])
    eff = compute_efficiency(weekly, k_eff=10.0)
    assert abs(eff["qb1"]["ypa"] - 8.0) < 0.5           # heavy sample ~ its own rate
    assert eff["qb2"]["ypa"] < 20.0 and eff["qb2"]["ypa"] > 8.0  # pulled toward baseline, not 0
