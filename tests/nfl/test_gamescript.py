import pandas as pd
from sportsmodel.nfl.gamescript import team_game_volume, fit_gamescript, project_team_volume

def test_team_game_volume_aggregates_and_joins():
    weekly = pd.DataFrame([
        {"season":2023,"week":1,"recent_team":"KC","attempts":30,"carries":25},
        {"season":2023,"week":1,"recent_team":"KC","attempts":5,"carries":0},   # 2nd QB
        {"season":2023,"week":1,"recent_team":"LA","attempts":40,"carries":18},
    ])
    sched = pd.DataFrame([{"season":2023,"week":1,"home_team":"KC","away_team":"LA",
                           "spread_line":3.0,"total_line":44.0}])
    tv = team_game_volume(weekly, sched)
    kc = tv[tv["recent_team"]=="KC"].iloc[0]
    assert kc["pass_att"]==35 and kc["rush_att"]==25 and kc["plays"]==60
    assert kc["team_margin"]==3.0
    assert kc["implied_total"]==(44.0+3.0)/2   # 23.5
    la = tv[tv["recent_team"]=="LA"].iloc[0]
    assert la["team_margin"]==-3.0 and la["implied_total"]==(44.0-3.0)/2  # 20.5

def test_underdog_projects_more_pass_attempts():
    # synthetic: pass_rate falls with team_margin (favorites run), plays ~ constant.
    # total_line is cycled through several values INDEPENDENTLY of team_margin so the
    # design matrix [1, team_margin, implied_total] is full-rank (rank 3), not a
    # collinear rank-2 system where lstsq would arbitrarily split the two coefficients.
    margins = list(range(-10, 11, 2))
    totals = [42.0, 45.0, 48.0, 51.0]
    rows=[]
    for i, m in enumerate(margins):
        pr = 0.60 - 0.01*m           # underdog (m<0) -> higher pass rate
        plays = 62
        rows.append({"season":2023,"week":1,"recent_team":f"T{m}","attempts":round(plays*pr),
                     "carries":round(plays*(1-pr))})
    weekly=pd.DataFrame(rows)
    sched=pd.DataFrame([{"season":2023,"week":1,"home_team":f"T{m}","away_team":"X",
                         "spread_line":float(m),
                         "total_line":totals[i % len(totals)]}
                        for i, m in enumerate(margins)])
    tv=team_game_volume(weekly,sched)
    model=fit_gamescript(tv[tv["recent_team"]!="X"])
    # design is full-rank now, so this actually checks the fit separates the two
    # regressors: pass_rate should decrease as team_margin increases (favorites run).
    assert model.pr_coef[1] < 0
    dog=project_team_volume(model, team_margin=-7.0, implied_total=22.0)
    fav=project_team_volume(model, team_margin=+7.0, implied_total=25.0)
    assert dog["pass_att"] > fav["pass_att"]
    assert 45 <= dog["plays"] <= 85 and 45 <= fav["plays"] <= 85
