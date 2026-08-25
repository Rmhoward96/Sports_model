from __future__ import annotations
import pandas as pd

def _safe(n, d):
    return (n / d) if d else 0.0

def compute_efficiency(weekly: pd.DataFrame, k_eff: float = 4.0) -> dict:
    p = (weekly.groupby(["player_id", "position"], as_index=False)
         .agg(att=("attempts", "sum"), pass_yds=("passing_yards", "sum"), pass_td=("passing_tds", "sum"),
              tgt=("targets", "sum"), rec=("receptions", "sum"), rec_yds=("receiving_yards", "sum"),
              rec_td=("receiving_tds", "sum"), car=("carries", "sum"), rush_yds=("rushing_yards", "sum"),
              rush_td=("rushing_tds", "sum")))
    # position baselines (volume-weighted league rates within a position)
    pos = (p.groupby("position", as_index=False)
           .agg(att=("att", "sum"), pass_yds=("pass_yds", "sum"), pass_td=("pass_td", "sum"),
                tgt=("tgt", "sum"), rec=("rec", "sum"), rec_yds=("rec_yds", "sum"),
                rec_td=("rec_td", "sum"), car=("car", "sum"), rush_yds=("rush_yds", "sum"),
                rush_td=("rush_td", "sum")))
    base = {r["position"]: r for _, r in pos.iterrows()}
    def shrink(vol, num, b_num, b_den):
        b = _safe(b_num, b_den)
        return (num + k_eff * b) / (vol + k_eff) if (vol + k_eff) > 0 else b
    out = {}
    for _, r in p.iterrows():
        b = base[r["position"]]
        out[r["player_id"]] = {
            "ypa": shrink(r["att"], r["pass_yds"], b["pass_yds"], b["att"]),
            "pass_td_rate": shrink(r["att"], r["pass_td"], b["pass_td"], b["att"]),
            "catch_rate": shrink(r["tgt"], r["rec"], b["rec"], b["tgt"]),
            "ypr": shrink(r["rec"], r["rec_yds"], b["rec_yds"], b["rec"]),
            "rec_td_rate": shrink(r["tgt"], r["rec_td"], b["rec_td"], b["tgt"]),
            "ypc": shrink(r["car"], r["rush_yds"], b["rush_yds"], b["car"]),
            "rush_td_rate": shrink(r["car"], r["rush_td"], b["rush_td"], b["car"]),
        }
    return out
