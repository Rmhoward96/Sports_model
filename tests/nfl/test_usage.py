import math
import pandas as pd
from sportsmodel.nfl.usage import compute_usage_shares, allocate

def _wk(pid, name, team, pos, wk, targets=0, carries=0, att=0):
    return {"player_id":pid,"player_name":name,"recent_team":team,"position":pos,
            "week":wk,"targets":targets,"carries":carries,"attempts":att}

def test_shares_and_allocation():
    weekly = pd.DataFrame([
        _wk("qb","QB1","KC","QB",1,att=30), _wk("qb","QB1","KC","QB",2,att=34),
        _wk("wr","WR1","KC","WR",1,targets=10), _wk("wr","WR1","KC","WR",2,targets=8),
        _wk("rb","RB1","KC","RB",1,carries=15), _wk("rb","RB1","KC","RB",2,carries=13),
    ])
    s = compute_usage_shares(weekly, k_usage=0.0)  # no shrink
    assert math.isclose(s["qb"]["pass_att_share"], 1.0)     # only QB throws
    assert math.isclose(s["wr"]["target_share"], 1.0)       # only WR targeted
    alloc = allocate(s["wr"], {"pass_att":40,"rush_att":24,"plays":64})
    assert math.isclose(alloc["targets"], 40*1.0)

def test_shrinkage_reduces_low_sample_share():
    weekly = pd.DataFrame([
        _wk("wr","WR1","KC","WR",1,targets=10),                 # 1 game
        _wk("wr2","WR2","KC","WR",1,targets=0), _wk("wr2","WR2","KC","WR",2,targets=0),
    ])
    s0 = compute_usage_shares(weekly, k_usage=0.0)
    s4 = compute_usage_shares(weekly, k_usage=4.0)
    assert s4["wr"]["target_share"] < s0["wr"]["target_share"]   # shrunk toward 0
    assert math.isclose(s4["wr"]["target_share"], s0["wr"]["target_share"] * (1/(1+4)))
