"""Player-week feature table for the props-ML models. PURE (DataFrames in,
DataFrames out). Leakage contract: every feature for (season S, week w) is
built from rows strictly before (S, w) -- enforced by shift(1) before any
rolling statistic and tested by perturbing week-w-and-later box scores.

Column prefixes select feature groups per ladder rung (sim.nfl.learned):
p_ usage/efficiency, ngs_ Next Gen Stats, tm_ own team, op_ opponent,
st_ status, cx_ context, mk_ market; y_ are labels (never features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.efficiency import adjusted_efficiency
from sportsmodel.nfl.teams import normalize_team

SKILL = ("QB", "RB", "WR", "TE")
KEYS = ["player_id", "season", "week"]
_STAT_COLS = {
    "targets": "y_targets", "carries": "y_carries", "attempts": "y_pass_att", "receptions": "y_receptions",
    "receiving_yards": "y_rec_yds", "rushing_yards": "y_rush_yds", "passing_yards": "y_pass_yds",
    "passing_tds": "y_pass_tds", "receiving_air_yards": "rec_air_yards", "receiving_yards_after_catch": "yac",
    "target_share": "target_share", "air_yards_share": "air_yards_share",
    "receiving_tds": "_rec_tds", "rushing_tds": "_rush_tds",
}


def _norm(code) -> str | None:
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def player_games(weekly: pd.DataFrame, snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> pd.DataFrame:
    """One row per REG-season player-game for QB/RB/WR/TE with offense_snaps > 0.
    Snap counts define who played (weekly omits players with no stats); weekly
    stats are left-joined and zero-filled (played + no stat = 0, not NaN)."""
    s = snaps[(snaps["game_type"] == "REG") & (snaps["offense_snaps"] > 0)
              & snaps["position"].isin(SKILL)].copy()
    s["player_id"] = s["pfr_player_id"].map(pfr2gsis)
    s["team"], s["opponent"] = s["team"].map(_norm), s["opponent"].map(_norm)
    s = s.dropna(subset=["player_id", "team", "opponent"])
    s = s.rename(columns={"offense_pct": "snap_pct"})[
        ["player_id", "season", "week", "team", "opponent", "position", "snap_pct"]]
    w = weekly[weekly["season_type"] == "REG"] if "season_type" in weekly.columns else weekly
    stats = w[KEYS + list(_STAT_COLS)].rename(columns=_STAT_COLS)
    out = s.merge(stats, on=KEYS, how="left").drop_duplicates(KEYS)
    stat_cols = list(_STAT_COLS.values())
    out[stat_cols] = out[stat_cols].fillna(0.0)
    out["y_anytime_td"] = ((out["_rec_tds"] + out["_rush_tds"]) > 0).astype(float)
    return out.drop(columns=["_rec_tds", "_rush_tds"]).reset_index(drop=True)


def team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, offense team) REG-season volume: pass attempts exclude
    sacks (matches sim rates B.3), neutral pass rate over wp 0.2-0.8, downs 1-2,
    quarters 1-3."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()].copy()
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    sack = p["sack"].fillna(0) == 1
    p["_pass"] = (p["play_type"] == "pass") & ~sack
    p["_rush"] = p["play_type"] == "run"
    p["_db"] = p["play_type"] == "pass"
    p["_press"] = p["_db"] & (sack | (p["qb_hit"].fillna(0) == 1))
    neutral = p["wp"].between(0.2, 0.8) & p["down"].isin([1, 2]) & (p["qtr"] <= 3)
    p["_n"], p["_npass"] = neutral, neutral & p["_db"]
    g = p.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        pass_att=("_pass", "sum"), rush_att=("_rush", "sum"), plays=("play_type", "size"),
        dropbacks=("_db", "sum"), pressures_allowed=("_press", "sum"), _n=("_n", "sum"), _npass=("_npass", "sum"))
    g["neutral_pass_rate"] = np.where(g["_n"] > 0, g["_npass"] / g["_n"].where(g["_n"] > 0, 1), np.nan)
    g["team"], g["opponent"] = g["posteam"].map(_norm), g["defteam"].map(_norm)
    return g.drop(columns=["posteam", "defteam", "_n", "_npass"]).dropna(subset=["team", "opponent"])


def player_redzone(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per player-game red-zone targets/carries (yardline_100 <= 20) and
    goal-line carries (<= 5). Only pass/run plays count (QB kneels and
    non-plays would otherwise inflate rz/gl carries)."""
    p = pbp[pbp["yardline_100"].notna() & pbp["play_type"].isin(["pass", "run"])]
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    rz = p[p["yardline_100"] <= 20]
    t = rz.dropna(subset=["receiver_player_id"]).groupby(["receiver_player_id", "season", "week"]).size().rename("rz_targets")
    c = rz.dropna(subset=["rusher_player_id"]).groupby(["rusher_player_id", "season", "week"]).size().rename("rz_carries")
    gl = p[p["yardline_100"] <= 5].dropna(subset=["rusher_player_id"]).groupby(
        ["rusher_player_id", "season", "week"]).size().rename("gl_carries")
    for s in (t, c, gl):
        s.index = s.index.set_names(KEYS)
    return pd.concat([t, c, gl], axis=1).fillna(0.0).reset_index()


# --------------------------------------------------------------------------
# Leakage-safe rolling feature tables
# --------------------------------------------------------------------------
_ORDER = ["season", "week"]
_TEAM_KEYS = ["season", "week", "team"]
_OPP_KEYS = ["season", "week", "opponent"]
_RZ_COLS = ["rz_targets", "rz_carries", "gl_carries"]
_TM_COLS = ["pass_att", "rush_att", "plays", "neutral_pass_rate"]
_OP_COLS = ["pass_att", "rush_att", "press_rate"]
_REC_POS = ("RB", "WR", "TE")
_HALFLIFE = 4.0
_FEATURE_PREFIXES = ("p_", "ngs_", "tm_", "op_", "st_", "cx_", "mk_")


def roll_features(df: pd.DataFrame, group: str, cols: list[str], prefix: str,
                  *, windows=(3, 5, 10), halflife: float = 4.0, sparse: bool = False) -> pd.DataFrame:
    """Add strictly-prior rolling features for `cols` within `group`.
    Every statistic is computed on the group's values shifted by one game, so
    a row never sees its own game or any later one."""
    out = df.sort_values([group, *_ORDER]).reset_index(drop=True)
    g = out.groupby(group, sort=False)
    new = {f"{prefix}n_season": out.groupby([group, "season"]).cumcount()}
    for c in cols:
        prev = g[c].shift(1)
        pg = prev.groupby(out[group])
        if not sparse:
            for n in windows:
                new[f"{prefix}{c}_r{n}"] = pg.transform(lambda x, n=n: x.rolling(n, min_periods=1).mean())
            std = out[c].groupby([out[group], out["season"]]).transform(lambda x: x.shift(1).expanding().mean())
            new[f"{prefix}{c}_std"] = std
        new[f"{prefix}{c}_ewm"] = pg.transform(lambda x: x.ewm(halflife=halflife, ignore_na=True).mean())
    # New columns added in one concat (column-by-column inserts fragment the
    # frame -> PerformanceWarning on wide inputs).
    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    # Previous season's full mean -- merged once AFTER the loop so the groupby
    # objects above stay aligned with `out`'s index.
    prev_season = out.groupby([group, "season"])[cols].mean().add_prefix(prefix).add_suffix("_prev").reset_index()
    prev_season["season"] += 1
    return out.merge(prev_season, on=[group, "season"], how="left")


def _safe_div(num: pd.Series, den: pd.Series) -> np.ndarray:
    """num / den, NaN where den is 0 or missing (so rolling means skip it)."""
    return np.where(den > 0, num / den.where(den > 0, 1), np.nan)


def _ord(df: pd.DataFrame) -> pd.Series:
    """Chronological ordinal season*100 + week (for as-of joins)."""
    return (df["season"].astype("int64") * 100 + df["week"].astype("int64")).rename("_ord")


def _ratios(pg: pd.DataFrame) -> pd.DataFrame:
    """Per-game efficiency ratios; NaN when the denominator is 0 so rolling
    means ignore games with no opportunity."""
    d = pg.copy()
    d["ypt"], d["catch_rate"] = _safe_div(d["y_rec_yds"], d["y_targets"]), _safe_div(d["y_receptions"], d["y_targets"])
    d["ypr"], d["ypc"] = _safe_div(d["y_rec_yds"], d["y_receptions"]), _safe_div(d["y_rush_yds"], d["y_carries"])
    d["yac_pr"] = _safe_div(d["yac"], d["y_receptions"])
    return d


_P_COLS = ["snap_pct", "target_share", "air_yards_share", "y_targets", "y_carries", "y_pass_att",
           "rz_targets", "rz_carries", "gl_carries", "carry_share",
           "ypt", "catch_rate", "ypr", "ypc", "yac_pr"]
_NGS_COLS = {
    "rec": ["avg_separation", "avg_cushion", "avg_intended_air_yards", "avg_yac_above_expectation"],
    "rush": ["rush_yards_over_expected_per_att", "efficiency", "percent_attempts_gte_eight_defenders"],
    "pass": ["avg_time_to_throw", "completion_percentage_above_expectation", "aggressiveness"],
}


def _player_base(pg: pd.DataFrame, tg: pd.DataFrame, rz: pd.DataFrame, stubs: pd.DataFrame | None) -> pd.DataFrame:
    """Played rows with per-game ratios, red-zone counts (no rz row = 0) and
    `carry_share` = carries / team rush_att, plus `stubs` rows (labels NaN)
    that are not already played rows. `_played` flags real games."""
    base = _ratios(pg).merge(rz[KEYS + _RZ_COLS], on=KEYS, how="left")
    base[_RZ_COLS] = base[_RZ_COLS].fillna(0.0)
    ra = tg[_TEAM_KEYS + ["rush_att"]].drop_duplicates(_TEAM_KEYS)
    base = base.merge(ra, on=_TEAM_KEYS, how="left")
    base["carry_share"] = _safe_div(base["y_carries"], base["rush_att"])
    base = base.drop(columns="rush_att").assign(_played=1.0)
    if stubs is not None and len(stubs):
        s = stubs[KEYS + ["team", "opponent", "position"]].merge(base[KEYS], on=KEYS, how="left", indicator=True)
        s = s[s["_merge"] == "left_only"].drop(columns="_merge").assign(_played=0.0)
        if len(s):
            base = pd.concat([base, s], ignore_index=True)
    return base


def _player_rolls(base: pd.DataFrame) -> pd.DataFrame:
    """p_ rolled usage/efficiency, p_career_games (prior played games,
    capped 100) and p_pos (position category)."""
    out = roll_features(base, "player_id", _P_COLS, "p_")
    played = out.groupby("player_id")["_played"].cumsum() - out["_played"]
    out["p_career_games"] = played.clip(upper=100)
    out["p_pos"] = pd.Categorical(out["position"], categories=list(SKILL))
    return out


def _ngs_rolls(base: pd.DataFrame, ngs: dict | None) -> pd.DataFrame:
    """ngs_ features: EWM and previous-season mean of weekly REG Next Gen
    Stats (week 0 rows are season aggregates -- excluded). Missing frames
    leave the columns NaN."""
    cols: list[str] = []
    for kind, kcols in _NGS_COLS.items():
        f = (ngs or {}).get(kind)
        if f is None or not len(f):
            base = base.assign(**{c: np.nan for c in kcols})
        else:
            f = f[f["week"] > 0]
            if "season_type" in f.columns:
                f = f[f["season_type"] == "REG"]
            f = f.rename(columns={"player_gsis_id": "player_id"})[KEYS + kcols].drop_duplicates(KEYS)
            base = base.merge(f, on=KEYS, how="left")
        cols += kcols
    return roll_features(base, "player_id", cols, "ngs_", sparse=True).drop(columns="ngs_n_season")


def _team_base(tg: pd.DataFrame, ctx: pd.DataFrame | None) -> pd.DataFrame:
    """Played team-games (`tg`) plus scheduled-but-unplayed team-weeks from
    `ctx` (stats NaN) so an upcoming week gets history-only features."""
    t = tg.drop_duplicates(_TEAM_KEYS)
    if ctx is not None and len(ctx):
        sched = ctx[_TEAM_KEYS + ["opponent"]].merge(t[_TEAM_KEYS], on=_TEAM_KEYS, how="left", indicator=True)
        sched = sched[sched["_merge"] == "left_only"].drop(columns="_merge")
        if len(sched):
            t = pd.concat([t, sched], ignore_index=True)
    return t


def _opp_view(base: pd.DataFrame) -> pd.DataFrame:
    """Each team-game re-keyed to the defense (team := opponent): offensive
    volume it faced and the pressure rate it generated, rolled per defense.
    Returned keyed by `opponent` for merging onto the offense's row."""
    d = base[["season", "week", "opponent", "pass_att", "rush_att"]].rename(columns={"opponent": "team"})
    d["press_rate"] = _safe_div(base["pressures_allowed"], base["dropbacks"])
    r = roll_features(d, "team", _OP_COLS, "op_")
    return r[_TEAM_KEYS + [c for c in r.columns if c.startswith("op_")]].rename(columns={"team": "opponent"})


def _epa_features(t: pd.DataFrame, game_epa: dict) -> pd.DataFrame:
    """tm_off_adj/tm_def_adj from adjusted_efficiency(S, w) (same-season
    weeks < w; NaN in week 1), tm_*_prev from season S-1, and the opponent's
    op_def_adj/op_def_prev. Cached per (season, upto_week)."""
    cache: dict[tuple[int, int], dict] = {}

    def get(s: int, w: int, team, key: str) -> float:
        if (s, w) not in cache:
            cache[(s, w)] = adjusted_efficiency(game_epa, s, w)
        v = cache[(s, w)].get(team)
        return float(v[key]) if v is not None else np.nan

    rows = []
    for s, w, tm, op in t[["season", "week", "team", "opponent"]].itertuples(index=False):
        s, w = int(s), int(w)
        rows.append((get(s, w, tm, "off_adj"), get(s, w, tm, "def_adj"), get(s - 1, 99, tm, "off_adj"),
                     get(s - 1, 99, tm, "def_adj"), get(s, w, op, "def_adj"), get(s - 1, 99, op, "def_adj")))
    cols = ["tm_off_adj", "tm_def_adj", "tm_off_prev", "tm_def_prev", "op_def_adj", "op_def_prev"]
    return t.join(pd.DataFrame(rows, columns=cols, index=t.index))


def _ctx_features(ctx: pd.DataFrame | None) -> pd.DataFrame:
    """cx_*/mk_* per (season, week, team), plus cx_home from is_home."""
    if ctx is None or not len(ctx):
        return pd.DataFrame(columns=_TEAM_KEYS)
    c = ctx.drop_duplicates(_TEAM_KEYS)
    out = c[_TEAM_KEYS + [x for x in c.columns if x.startswith(("cx_", "mk_"))]].copy()
    if "is_home" in c.columns:
        out["cx_home"] = c["is_home"].astype(float)
    return out


def build_team_table(tg: pd.DataFrame, ctx: pd.DataFrame | None, game_epa: dict) -> pd.DataFrame:
    """One row per (team, season, week): labels y_team_pass_att /
    y_team_rush_att and strictly-prior tm_ (own offense), op_ (opponent's
    defense), plus pre-game cx_/mk_ context."""
    base = _team_base(tg, ctx)
    t = roll_features(base, "team", _TM_COLS, "tm_")
    t = t.merge(_opp_view(base), on=_OPP_KEYS, how="left")
    t = _epa_features(t, game_epa)
    t["y_team_pass_att"], t["y_team_rush_att"] = t["pass_att"], t["rush_att"]
    t = t.merge(_ctx_features(ctx), on=_TEAM_KEYS, how="left")
    feats = [c for c in t.columns if c.startswith(("tm_", "op_", "cx_", "mk_"))]
    t = t[["team", "season", "week", "opponent", "y_team_pass_att", "y_team_rush_att"] + feats]
    return t.sort_values(["team", *_ORDER]).reset_index(drop=True)


def _op_ypt_allowed(pg: pd.DataFrame, team_tbl: pd.DataFrame) -> pd.DataFrame:
    """op_ypt_allowed_{pos}_{r5,ewm}: per defense-game shrunk yards/target
    allowed to each receiving position, (yds + 7.5*20) / (tgts + 20), rolled
    per (defense, position). Keyed by `opponent` (the defense)."""
    a = pg[pg["position"].isin(_REC_POS)].groupby(_OPP_KEYS + ["position"], as_index=False)[
        ["y_rec_yds", "y_targets"]].sum()
    a["ypt_allowed"] = (a["y_rec_yds"] + 7.5 * 20) / (a["y_targets"] + 20)
    # every defense-game x position, so weeks without that position still roll
    sk = team_tbl[_TEAM_KEYS].rename(columns={"team": "opponent"}).merge(
        pd.DataFrame({"position": list(_REC_POS)}), how="cross")
    a = sk.merge(a[_OPP_KEYS + ["position", "ypt_allowed"]], on=_OPP_KEYS + ["position"], how="left")
    a["_grp"] = a["opponent"].astype(str) + "|" + a["position"].astype(str)
    r = roll_features(a, "_grp", ["ypt_allowed"], "op_", windows=(5,))
    wide = r.pivot(index=_OPP_KEYS, columns="position", values=["op_ypt_allowed_r5", "op_ypt_allowed_ewm"])
    wide.columns = [f"op_ypt_allowed_{pos.lower()}_{stat.rsplit('_', 1)[1]}" for stat, pos in wide.columns]
    return wide.reset_index()


def _injuries(injuries: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalized (season, week, team, player_id, report_status), one row per player-week."""
    if injuries is None or not len(injuries):
        return None
    i = injuries[["season", "week", "team", "gsis_id", "report_status"]].dropna(subset=["gsis_id"])
    i = i.rename(columns={"gsis_id": "player_id"}).assign(team=i["team"].map(_norm))
    return i.drop_duplicates(KEYS)


def _vacated(out: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    """st_vacated_tgt / st_vacated_car: sum over teammates Out/Doubtful for
    (S, w) of their target_share / carry_share EWM as of their latest game
    strictly before (S, w) for this team (the player's own share excluded)."""
    played = out[out["_played"] == 1].sort_values(KEYS)
    h = played[KEYS + ["team"]].rename(columns={"team": "_hteam"})
    for src, dst in (("target_share", "_tgt"), ("carry_share", "_car")):
        h[dst] = played.groupby("player_id")[src].transform(
            lambda x: x.ewm(halflife=_HALFLIFE, ignore_na=True).mean())
    h["_ord"] = _ord(h)
    o = inj[inj["report_status"].isin(["Out", "Doubtful"])].dropna(subset=["team"])
    o = o[KEYS + ["team"]].assign(_ord=_ord(o)).sort_values("_ord")
    # merge_asof requires identical `by` dtypes: real injury ids arrive as
    # StringDtype, pg's (pfr->gsis .map) as object.
    o["player_id"], h["player_id"] = o["player_id"].astype(object), h["player_id"].astype(object)
    m = pd.merge_asof(o, h[["player_id", "_ord", "_hteam", "_tgt", "_car"]].sort_values("_ord"),
                      on="_ord", by="player_id", allow_exact_matches=False)
    m = m[m["_hteam"] == m["team"]]
    tot = m.groupby(_TEAM_KEYS, as_index=False)[["_tgt", "_car"]].sum()
    res = out.merge(tot, on=_TEAM_KEYS, how="left").merge(
        m[KEYS + ["_tgt", "_car"]].rename(columns={"_tgt": "_own_tgt", "_car": "_own_car"}), on=KEYS, how="left")
    vac = pd.DataFrame({
        "st_vacated_tgt": (res["_tgt"].fillna(0.0) - res["_own_tgt"].fillna(0.0)).to_numpy(),
        "st_vacated_car": (res["_car"].fillna(0.0) - res["_own_car"].fillna(0.0)).to_numpy(),
    }, index=out.index)
    return pd.concat([out, vac], axis=1)


def _qb_changed(out: pd.DataFrame, pg: pd.DataFrame, depth: pd.DataFrame | None) -> pd.DataFrame:
    """st_qb_changed: the as-of depth-chart QB1 for (S, w) differs from the
    pass-attempts leader of the team's previous game (NaN if either unknown)."""
    tw = out[_TEAM_KEYS].drop_duplicates()
    tw = tw.assign(_ord=_ord(tw)).sort_values("_ord")
    lead = pg[pg["y_pass_att"] > 0].sort_values("y_pass_att", ascending=False, kind="stable")
    lead = lead.drop_duplicates(_TEAM_KEYS)[_TEAM_KEYS + ["player_id"]].rename(columns={"player_id": "_lead"})
    lead = lead.assign(_ord=_ord(lead)).sort_values("_ord").drop(columns=["season", "week"])
    tw["team"], lead["team"] = tw["team"].astype(object), lead["team"].astype(object)   # merge_asof: same `by` dtype
    tw = pd.merge_asof(tw, lead, on="_ord", by="team", allow_exact_matches=False)
    if depth is not None and len(depth):
        d = depth[depth["position"] == "QB"].assign(team=depth["club_code"].map(_norm),
                                                     _rank=pd.to_numeric(depth["depth_team"], errors="coerce"))
        qb1 = d.dropna(subset=["team", "_rank", "gsis_id"]).sort_values("_rank", kind="stable")
        qb1 = qb1.drop_duplicates(_TEAM_KEYS)[_TEAM_KEYS + ["gsis_id"]].rename(columns={"gsis_id": "_qb1"})
        tw = tw.merge(qb1, on=_TEAM_KEYS, how="left")
    else:
        tw["_qb1"] = np.nan
    known = tw["_lead"].notna() & tw["_qb1"].notna()
    tw["st_qb_changed"] = np.where(known, (tw["_qb1"].astype(str) != tw["_lead"].astype(str)).astype(float), np.nan)
    return out.merge(tw[_TEAM_KEYS + ["st_qb_changed"]], on=_TEAM_KEYS, how="left")


def _status(out: pd.DataFrame, pg: pd.DataFrame, injuries: pd.DataFrame | None,
            depth: pd.DataFrame | None) -> pd.DataFrame:
    """st_ features from the week-w injury report and as-of depth chart
    (pre-game) plus strictly-prior usage. No injury data -> NaN."""
    inj = _injuries(injuries)
    if inj is None:
        out = out.assign(st_questionable=np.nan, st_vacated_tgt=np.nan, st_vacated_car=np.nan)
    else:
        q = inj.loc[inj["report_status"] == "Questionable", KEYS].assign(st_questionable=1.0)
        out = out.merge(q, on=KEYS, how="left")
        out["st_questionable"] = out["st_questionable"].fillna(0.0)
        out = _vacated(out, inj)
    return _qb_changed(out, pg, depth)


def _prior_snap_ewm(pg: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """For each (player_id, season, week) row of `rows`: the player's snap_pct
    EWM over played games strictly before (season, week) -- the same value as
    the table's p_snap_pct_ewm (stub weeks carry NaN snaps, which the
    ignore_na EWM skips). NaN with no prior game. Aligned to `rows.index`."""
    played = pg[KEYS + ["snap_pct"]].dropna(subset=["player_id"]).sort_values(KEYS)
    hist = played[["player_id"]].assign(
        _ord=_ord(played),
        _snap_ewm=played.groupby("player_id")["snap_pct"].transform(
            lambda x: x.ewm(halflife=_HALFLIFE, ignore_na=True).mean()))
    q = rows[["player_id"]].assign(_ord=_ord(rows), _i=np.arange(len(rows)))
    # merge_asof requires identical `by` dtypes (StringDtype depth ids vs object pg ids)
    q["player_id"], hist["player_id"] = q["player_id"].astype(object), hist["player_id"].astype(object)
    m = pd.merge_asof(q.sort_values("_ord"), hist.sort_values("_ord"), on="_ord", by="player_id",
                      allow_exact_matches=False)
    return pd.Series(m.sort_values("_i")["_snap_ewm"].to_numpy(), index=rows.index)


def _depth_rank(depth: pd.DataFrame | None, pg: pd.DataFrame) -> pd.DataFrame:
    """p_depth_rank: the player's rank 1..n within his as-of (pre-game) depth
    chart's (club_code, season, week, position) group -- the same meaning in
    both nflverse eras. Old weekly schema (<= 2024) lists slot depth
    (WR1/WR2/WR3 are each depth_team 1 at LWR/RWR/SWR); the 2025+ snapshot
    lists pos_rank (already a within-position order). Order: depth_team
    ascending, ties by the strictly-prior snap-share EWM descending (NaN
    last), then gsis_id. A player listed more than once in a group counts once
    (best slot); listed in several groups -> his best rank. Old-schema
    Non-offense listings are ignored in both eras (`formation` Special Teams /
    Defense: old-schema KR/PR slots are filed under the player's own
    position; snapshot KR/PR are their own pos_abb, tagged from pos_grp by
    depth_charts_asof -- otherwise a KR1 WR would rank 1); a missing
    formation is kept."""
    if depth is None or not len(depth):
        return pd.DataFrame(columns=KEYS + ["p_depth_rank"])
    if "formation" in depth.columns:
        depth = depth[depth["formation"].isna() | (depth["formation"] == "Offense")]
    grp = ["club_code", "season", "week", "position"]
    d = depth[grp + ["gsis_id"]].assign(_dt=pd.to_numeric(depth["depth_team"], errors="coerce"))
    d = d.dropna(subset=["gsis_id", "club_code", "position"]).rename(columns={"gsis_id": "player_id"})
    d = d.astype({"season": "int64", "week": "int64", "player_id": object, "club_code": object, "position": object})
    d = d.sort_values("_dt", kind="stable", na_position="last").drop_duplicates(grp + ["player_id"])
    d = d.assign(_snap=_prior_snap_ewm(pg, d))
    d = d.sort_values(grp + ["_dt", "_snap", "player_id"], ascending=[True] * 5 + [False, True],
                      na_position="last", kind="stable")
    d["p_depth_rank"] = (d.groupby(grp).cumcount() + 1).astype(float)
    return d.groupby(KEYS, as_index=False)["p_depth_rank"].min()


def build_feature_table(pg, tg, rz, ctx, ngs, injuries, depth, game_epa, stubs=None) -> pd.DataFrame:
    """One row per (player_id, season, week): every played row plus `stubs`
    (active-but-no-snap players the sim may still need; labels NaN). `ngs` is
    {"rec": df, "rush": df, "pass": df}. Returns KEYS, team, opponent,
    position, `is_stub` (bool: a stub row, never a feature), y_* labels and
    p_/ngs_/tm_/op_/st_/cx_/mk_ features. Same-game
    box-score columns (snap_pct, target_share, ...) are never returned."""
    out = _ngs_rolls(_player_rolls(_player_base(pg, tg, rz, stubs)), ngs)
    team_tbl = build_team_table(tg, ctx, game_epa)
    team_feats = [c for c in team_tbl.columns if c.startswith(("tm_", "op_", "cx_", "mk_"))]
    out = out.merge(team_tbl[_TEAM_KEYS + team_feats], on=_TEAM_KEYS, how="left")
    out = out.merge(_op_ypt_allowed(pg, team_tbl), on=_OPP_KEYS, how="left")
    out = _status(out, pg, injuries, depth)
    out = out.merge(_depth_rank(depth, pg), on=KEYS, how="left")
    labels = [c for c in pg.columns if c.startswith("y_")]
    feats = [c for c in out.columns if c.startswith(_FEATURE_PREFIXES)]
    out["is_stub"] = out["_played"] == 0
    out = out[KEYS + ["team", "opponent", "position", "is_stub"] + labels + feats]
    return out.sort_values(KEYS).reset_index(drop=True)
