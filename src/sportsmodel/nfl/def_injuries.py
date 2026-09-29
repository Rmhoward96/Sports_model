"""Opponent defensive injuries by unit (the `di_` features). PURE.

A defender's weight is his recent share of his team's defensive snaps (EWM,
halflife 4 of his games strictly before the target week, for that team). A
defense's `di_vacated_<group>` at (S, w) sums the weights of its Out/Doubtful
defenders in that group on the (S, w) report. Groups (plan ruling R6):
coverage, pass rush, run defense; each defender counts in his primary
(most frequent) snap position's group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

DEF_GROUPS: dict[str, str] = {
    "CB": "cov", "S": "cov", "FS": "cov", "SS": "cov", "DB": "cov",
    "DE": "rush", "OLB": "rush", "EDGE": "rush",
    "DT": "run", "NT": "run", "DL": "run", "LB": "run", "ILB": "run", "MLB": "run",
}
GROUPS = ("cov", "rush", "run")
_HALFLIFE = 4.0
_KEYS = ["season", "week", "team"]


def _norm(code):
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def _ord(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype("int64") * 100 + df["week"].astype("int64")


def defender_snaps(snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> pd.DataFrame:
    s = snaps[(snaps["game_type"] == "REG") & (snaps["defense_snaps"] > 0)
              & snaps["position"].isin(list(DEF_GROUPS))].copy()
    s["player_id"] = s["pfr_player_id"].map(pfr2gsis)
    s["team"] = s["team"].map(_norm)
    s = s.dropna(subset=["player_id", "team"])
    share = s["defense_pct"].astype(float)
    s["share"] = np.where(share.max() > 1.0, share / 100.0, share)
    primary = s.groupby("player_id")["position"].agg(lambda x: x.value_counts().index[0])
    s["group"] = s["player_id"].map(primary).map(DEF_GROUPS)
    return s[["player_id", "season", "week", "team", "group", "share"]].astype(
        {"season": "int64", "week": "int64", "player_id": object})


def vacated_by_defense(dsnaps: pd.DataFrame, injuries: pd.DataFrame | None,
                       team_weeks: pd.DataFrame) -> pd.DataFrame:
    cols = [f"di_vacated_{g}" for g in GROUPS]
    out = team_weeks[_KEYS].copy()
    if injuries is None or not len(injuries):
        return out.assign(**{c: np.nan for c in cols})
    h = dsnaps.sort_values(["player_id", "season", "week"]).copy()
    h["_ewm"] = h.groupby("player_id")["share"].transform(
        lambda x: x.ewm(halflife=_HALFLIFE, ignore_na=True).mean())
    h = h.assign(_ord=_ord(h)).rename(columns={"team": "_hteam"})
    inj = injuries.dropna(subset=["gsis_id"]).assign(team=injuries["team"].map(_norm))
    o = inj[inj["report_status"].isin(["Out", "Doubtful"])].dropna(subset=["team"])
    o = o.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "week", "team"]]
    o = o.astype({"season": "int64", "week": "int64", "player_id": object}).drop_duplicates()
    o = o.assign(_ord=_ord(o)).sort_values("_ord")
    m = pd.merge_asof(o, h[["player_id", "_ord", "_hteam", "group", "_ewm"]].sort_values("_ord"),
                      on="_ord", by="player_id", allow_exact_matches=False)
    m = m[(m["_hteam"] == m["team"]) & m["group"].notna()]
    tot = m.pivot_table(index=_KEYS, columns="group", values="_ewm", aggfunc="sum")
    tot = tot.reindex(columns=list(GROUPS)).add_prefix("di_vacated_").reset_index()
    out = out.merge(tot, on=_KEYS, how="left")
    reported = pd.MultiIndex.from_frame(out[["season", "week"]].astype("int64")).isin(
        pd.MultiIndex.from_frame(injuries[["season", "week"]].dropna().astype("int64").drop_duplicates()))
    out[cols] = out[cols].fillna(0.0)
    out.loc[~reported, cols] = np.nan
    return out
