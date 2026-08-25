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

def test_traded_player_uses_primary_team_deterministically():
    # Player "x" plays 3 games for team A (heavy usage) then 1 game for team B
    # after a trade. Their primary team is A (most games), and the result
    # must be stable across repeated calls (no groupby-order dependence).
    weekly = pd.DataFrame([
        _wk("x","X","A","WR",1,targets=8),
        _wk("x","X","A","WR",2,targets=8),
        _wk("x","X","A","WR",3,targets=8),
        _wk("x","X","B","WR",4,targets=2),
    ])
    s = compute_usage_shares(weekly, k_usage=0.0)
    assert s["x"]["team"] == "A"
    assert math.isclose(s["x"]["target_share"], 1.0)  # sole target-getter on team A

    s_again = compute_usage_shares(weekly, k_usage=0.0)
    assert s_again["x"] == s["x"]

def test_multi_team_shares_isolated_by_team_denominator():
    # Team B has a much larger target total than team A; team A's shares must
    # be computed against team A's own totals only, not diluted by team B.
    weekly = pd.DataFrame([
        _wk("a1","A1","A","WR",1,targets=5), _wk("a1","A1","A","WR",2,targets=5),
        _wk("a2","A2","A","WR",1,targets=5), _wk("a2","A2","A","WR",2,targets=5),
        _wk("b1","B1","B","WR",1,targets=100), _wk("b1","B1","B","WR",2,targets=100),
    ])
    s = compute_usage_shares(weekly, k_usage=0.0)
    assert math.isclose(s["a1"]["target_share"], 0.5)  # 10 / (10+10) on team A
    assert math.isclose(s["a2"]["target_share"], 0.5)

def test_zero_team_targets_gives_zero_share_not_nan():
    # Team A has only rushing volume (0 targets, 0 pass attempts); target_share
    # and pass_att_share must be 0.0, not NaN or a crash from dividing by zero.
    weekly = pd.DataFrame([
        _wk("rb1","RB1","A","RB",1,carries=10),
        _wk("rb1","RB1","A","RB",2,carries=10),
    ])
    s = compute_usage_shares(weekly, k_usage=0.0)
    assert s["rb1"]["target_share"] == 0.0
    assert s["rb1"]["pass_att_share"] == 0.0
    assert not math.isnan(s["rb1"]["target_share"])
    assert math.isclose(s["rb1"]["carry_share"], 1.0)
