from __future__ import annotations
import pandas as pd

def compute_usage_shares(weekly: pd.DataFrame, k_usage: float = 4.0) -> dict:
    team = (weekly.groupby("recent_team", as_index=False)
            .agg(t_targets=("targets", "sum"), t_carries=("carries", "sum"),
                 t_att=("attempts", "sum")))
    ply = (weekly.groupby(["player_id", "player_name", "recent_team", "position"], as_index=False)
           .agg(targets=("targets", "sum"), carries=("carries", "sum"),
                att=("attempts", "sum"), games=("week", "nunique")))
    m = ply.merge(team, on="recent_team")
    # A player may have rows for multiple teams within the window (mid-season
    # trade). Keep only their PRIMARY team: the one with the most games,
    # tie-broken deterministically by total volume, then by team name.
    m["_vol"] = m["targets"] + m["carries"] + m["att"]
    m = m.sort_values(
        by=["games", "_vol", "recent_team"],
        ascending=[False, False, True],
    )
    m = m.drop_duplicates(subset="player_id", keep="first")
    out = {}
    for _, r in m.iterrows():
        f = r["games"] / (r["games"] + k_usage) if (r["games"] + k_usage) > 0 else 0.0
        ts = (r["targets"] / r["t_targets"] if r["t_targets"] else 0.0) * f
        cs = (r["carries"] / r["t_carries"] if r["t_carries"] else 0.0) * f
        ps = (r["att"] / r["t_att"] if r["t_att"] else 0.0) * f
        out[r["player_id"]] = {"target_share": ts, "carry_share": cs, "pass_att_share": ps,
                               "position": r["position"], "team": r["recent_team"],
                               "player_name": r["player_name"]}
    return out

def allocate(shares: dict, team_volume: dict) -> dict:
    return {"targets": team_volume["pass_att"] * shares["target_share"],
            "carries": team_volume["rush_att"] * shares["carry_share"],
            "pass_att": team_volume["pass_att"] * shares["pass_att_share"]}
