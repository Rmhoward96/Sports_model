"""QB career profiles (the `qb_` features). PURE (loaders live in the scripts).

Every QB (starter, backup, veteran back from years on the bench, rookie) gets a
profile from his own regular-season passing history (nflverse weekly stats,
1999 onward):

1. `qb_games` -- one row per QB-game with a dropback: attempts, yards, TDs,
   INTs, sacks and the raw rates `ypa`, `td_rate`, `int_rate`, `sack_rate`.
2. `opponent_adjust` -- `<rate>_adj` = rate - (opponent's allowed rate -
   league rate), both attempt-weighted over that season's games strictly
   before the game's week; an opponent with < 100 attempts faced in that
   window uses its previous season's full allowed rate against the previous
   season's league rate; nothing known -> no adjustment.
3. `profile_asof` -- as of (S, w), from games strictly before (S, w):
   weight = att x 0.5^(age / H) (age in seasons, fractional by week),
   n = sum of weights, shrunk = (n * raw + k * replacement) / (n + k).
   `replacement` = attempt-weighted `<rate>_adj` of non-regular QBs (not their
   team's season attempts leader) in seasons before S.
4. `qb1_by_team_week` / `team_qb_features` -- the as-of depth-chart QB1's
   profile per team-week, his ratio to the recent passer mix, and whether he
   differs from the previous game's attempts leader.
5. `tune` -- choose (H, k) by attempt-weighted MSE of next-game `ypa_adj`.

Leakage: every value for (S, w) uses only games with (season, week) < (S, w)
(the as-of depth chart and that week's injury report are pre-game inputs).
Team codes are normalized (`normalize_team`); `team_weeks` passed in must use
normalized codes too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

RATES = ("ypa", "td_rate", "int_rate", "sack_rate")
ADJ = tuple(f"{r}_adj" for r in RATES)
PROFILE_COLS = ("qb_ypa", "qb_td_rate", "qb_int_rate", "qb_sack_rate", "qb_n_eff")
RATIO_COLS = {"qb_ratio_ypa": "qb_ypa", "qb_ratio_td": "qb_td_rate", "qb_ratio_sack": "qb_sack_rate"}
TEAM_COLS = ("season", "week", "team", *PROFILE_COLS, *RATIO_COLS, "qb_changed")
MIN_OPP_ATT = 100          # attempts faced before the current-season window is trusted
WEEKS_PER_SEASON = 18      # fractional age: (week - 1) / 18
MIX_GAMES = 10             # passer-mix window (team games)
MIX_HALFLIFE = 4.0         # passer-mix weight 0.5^(games_ago / 4)
RATIO_CLAMP = (0.6, 1.3)
_KEYS = ["season", "week", "team"]


def _norm(code):
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def _ord(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype("int64") * 100 + df["week"].astype("int64")


def _t(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype(float) + (df["week"].astype(float) - 1.0) / WEEKS_PER_SEASON


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)


def qb_games(weekly: pd.DataFrame) -> pd.DataFrame:
    """REG QB rows with attempts + sacks > 0, normalized teams, raw rates."""
    w = weekly[(weekly["season_type"] == "REG") & (weekly["position"] == "QB") & weekly["player_id"].notna()]
    att, sacks = _num(w["attempts"]), _num(w["sacks_suffered"])
    w, att, sacks = w[(att + sacks) > 0], att[(att + sacks) > 0], sacks[(att + sacks) > 0]
    codes = pd.unique(pd.concat([w["recent_team"], w["opponent_team"]]).dropna().astype(str))
    cmap = {c: _norm(c) for c in codes}
    g = pd.DataFrame({
        "player_id": w["player_id"].astype(str).to_numpy(),
        "season": w["season"].astype(int).to_numpy(),
        "week": w["week"].astype(int).to_numpy(),
        "team": w["recent_team"].astype(str).map(cmap).to_numpy(),
        "opponent": w["opponent_team"].astype(str).map(cmap).to_numpy(),
        "att": att.to_numpy(), "yds": _num(w["passing_yards"]).to_numpy(),
        "tds": _num(w["passing_tds"]).to_numpy(), "ints": _num(w["passing_interceptions"]).to_numpy(),
        "sacks": sacks.to_numpy(),
    })
    g = g[g["team"].notna() & g["opponent"].notna()]
    a = g["att"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = g.assign(ypa=np.where(a > 0, g["yds"] / a, np.nan),
                     td_rate=np.where(a > 0, g["tds"] / a, np.nan),
                     int_rate=np.where(a > 0, g["ints"] / a, np.nan),
                     sack_rate=np.where(a + g["sacks"] > 0, g["sacks"] / (a + g["sacks"]), np.nan))
    return g.sort_values(["season", "week", "team", "player_id"], kind="stable").reset_index(drop=True)


def _parts(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Per-row attempt-weighted numerator/denominator of each rate (NaN rate -> 0/0)."""
    cols = {"att": g["att"].to_numpy(dtype=float)}
    for r in RATES:
        ok = g[r].notna().to_numpy()
        n = np.where(ok, g["att"].to_numpy(dtype=float), 0.0)
        cols[f"_n_{r}"] = n
        cols[f"_s_{r}"] = n * np.where(ok, g[r].to_numpy(dtype=float), 0.0)
    return pd.DataFrame(cols, index=g.index), list(cols)


def opponent_adjust(qg: pd.DataFrame) -> pd.DataFrame:
    """Add `<rate>_adj` (see module doc). Row order and columns of `qg` kept."""
    g = qg.reset_index(drop=True)
    parts, cols = _parts(g)
    p = pd.concat([g[["season", "week", "opponent"]], parts], axis=1)

    # defense windows: sums over that season's weeks strictly before the game's week
    dw = p.groupby(["season", "opponent", "week"], sort=True)[cols].sum()
    dbefore = dw.groupby(level=["season", "opponent"]).cumsum().groupby(
        level=["season", "opponent"]).shift(1).fillna(0.0)
    lw = p.groupby(["season", "week"], sort=True)[cols].sum()
    lbefore = lw.groupby(level="season").cumsum().groupby(level="season").shift(1).fillna(0.0)
    # previous season's full allowed (defense) and league totals, keyed to the next season
    dprev = p.groupby(["season", "opponent"])[cols].sum().reset_index()
    dprev["season"] += 1
    lprev = p.groupby("season")[cols].sum().reset_index()
    lprev["season"] += 1

    k = g[["season", "week", "opponent"]]
    d = k.merge(dbefore.reset_index(), on=["season", "opponent", "week"], how="left")
    lg = k.merge(lbefore.reset_index(), on=["season", "week"], how="left")
    dp = k.merge(dprev, on=["season", "opponent"], how="left")
    lp = k.merge(lprev, on="season", how="left")

    use_cur = (d["att"] >= MIN_OPP_ATT).to_numpy()
    out = g.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for r in RATES:
            n, s = f"_n_{r}", f"_s_{r}"
            cur = (d[s] / d[n] - lg[s] / lg[n]).to_numpy(dtype=float)
            prev = (dp[s] / dp[n] - lp[s] / lp[n]).to_numpy(dtype=float)
            shift = np.where(use_cur & np.isfinite(cur), cur, np.where(np.isfinite(prev), prev, 0.0))
            out[f"{r}_adj"] = g[r].to_numpy(dtype=float) - shift
    return out


def replacement(qga: pd.DataFrame, before_season: int) -> dict[str, float]:
    """Attempt-weighted `<rate>_adj` of non-regular QBs in seasons < before_season.

    Regular = the QB with the most attempts for (season, team). If no
    non-regular attempts exist (tiny fixtures), all QBs are used; nothing at
    all -> NaN."""
    h = qga[qga["season"] < before_season]
    if not len(h):
        return {a: float("nan") for a in ADJ}
    tot = h.groupby(["season", "team", "player_id"], as_index=False)["att"].sum()
    lead = tot.sort_values(["season", "team", "att", "player_id"], ascending=[True, True, False, True],
                           kind="stable").drop_duplicates(["season", "team"])
    m = h.merge(lead[["season", "team", "player_id"]].assign(_reg=1),
                on=["season", "team", "player_id"], how="left")
    src = m[m["_reg"].isna()]
    if src["att"].sum() <= 0:
        src = m
    out = {}
    for a in ADJ:
        ok = src[a].notna() & (src["att"] > 0)
        den = src.loc[ok, "att"].sum()
        out[a] = float((src.loc[ok, "att"] * src.loc[ok, a]).sum() / den) if den > 0 else float("nan")
    return out


def _weighted_sums(qga: pd.DataFrame, keys: pd.DataFrame, H: float) -> pd.DataFrame:
    """For each key row (player_id, season, week): n_eff = sum w and, per rate,
    `_n_<r>` = sum w over games with a non-NaN `<r>_adj` and `_s_<r>` = sum
    w * `<r>_adj`, over the player's games strictly before (season, week);
    w = att * 0.5^(age / H). Row-aligned with `keys` (0 when no games).

    Vectorized via 0.5^((t_key - t_g)/H) = 0.5^((t_key - t0)/H) * 2^((t_g - t0)/H)
    with t0 the player's first game: per-player cumulative sums + merge_asof."""
    kk = pd.DataFrame({"player_id": keys["player_id"].astype(str).to_numpy(),
                       "_ord": _ord(keys).to_numpy(), "_tk": _t(keys).to_numpy(),
                       "_i": np.arange(len(keys))})
    sums = ["_a"] + [f"_n_{r}" for r in RATES] + [f"_s_{r}" for r in RATES]
    res = pd.DataFrame(0.0, index=np.arange(len(keys)), columns=sums)
    if not len(qga) or not len(keys):
        return res
    g = pd.DataFrame({"player_id": qga["player_id"].astype(str).to_numpy(),
                      "_ord": _ord(qga).to_numpy(), "_tg": _t(qga).to_numpy(),
                      "att": qga["att"].to_numpy(dtype=float)})
    t0 = g.groupby("player_id")["_tg"].min()
    e = np.power(2.0, (g["_tg"] - g["player_id"].map(t0)).to_numpy() / H)
    att = g["att"].to_numpy()
    g["_a"] = att * e
    for r in RATES:
        v = qga[f"{r}_adj"].to_numpy(dtype=float)
        ok = np.isfinite(v)
        g[f"_n_{r}"] = np.where(ok, att, 0.0) * e
        g[f"_s_{r}"] = np.where(ok, att * np.where(ok, v, 0.0), 0.0) * e
    agg = g.groupby(["player_id", "_ord"], sort=True)[sums].sum()
    cum = agg.groupby(level="player_id").cumsum().reset_index()
    m = pd.merge_asof(kk.sort_values("_ord", kind="stable"), cum.sort_values("_ord", kind="stable"),
                      on="_ord", by="player_id", direction="backward", allow_exact_matches=False)
    m = m.sort_values("_i")
    f = np.power(0.5, (m["_tk"] - m["player_id"].map(t0)).to_numpy(dtype=float) / H)
    for c in sums:
        res[c] = np.nan_to_num(m[c].to_numpy(dtype=float) * f, nan=0.0)
    return res


def _shrink(ws: pd.DataFrame, k: float, repl: dict) -> pd.DataFrame:
    out = {}
    for r in RATES:
        rv = repl[f"{r}_adj"]
        rv = rv.to_numpy(dtype=float) if isinstance(rv, pd.Series) else rv
        out[f"qb_{r}"] = (ws[f"_s_{r}"].to_numpy() + k * rv) / (ws[f"_n_{r}"].to_numpy() + k)
    out["qb_n_eff"] = ws["_a"].to_numpy()
    return pd.DataFrame(out, index=ws.index)


def profile_asof(qga: pd.DataFrame, keys: pd.DataFrame, H: float, k: float, repl: dict) -> pd.DataFrame:
    """`keys` (player_id, season, week) + the shrunk profile from games strictly before."""
    base = keys[["player_id", "season", "week"]].reset_index(drop=True)
    prof = _shrink(_weighted_sums(qga, base, float(H)), float(k), repl)
    return pd.concat([base, prof.reset_index(drop=True)], axis=1)


def qb1_by_team_week(depth: pd.DataFrame, injuries: pd.DataFrame | None, team_weeks: pd.DataFrame,
                     override: dict | None = None) -> pd.DataFrame:
    """QB1 per team-week: the lowest `depth_team` QB on the as-of chart (exact
    week else the team's latest earlier chart, `usage.chart_weeks_asof`) who is
    not Out/Doubtful on that week's report; `override[(season, week, team)]`
    wins. No chart -> NaN. Row-aligned with `team_weeks`."""
    from sportsmodel.sim.nfl.usage import chart_weeks_asof

    tw = team_weeks[_KEYS].reset_index(drop=True).astype({"season": int, "week": int})
    qb1 = pd.Series(np.nan, index=tw.index, dtype=object)
    if depth is not None and len(depth) and len(tw):
        dn = depth.assign(club_code=depth["club_code"].map(_norm)).dropna(subset=["club_code", "season", "week"])
        dn = dn.astype({"season": int, "week": int})
        cw = chart_weeks_asof(dn, tw).assign(_i=np.arange(len(tw))).dropna(subset=["chart_season"])
        cw = cw.astype({"chart_season": int, "chart_week": int})
        d = dn[(dn["position"] == "QB") & dn["gsis_id"].notna()]
        d = pd.DataFrame({"team": d["club_code"], "chart_season": d["season"], "chart_week": d["week"],
                          "gsis_id": d["gsis_id"].astype(str),
                          "_dt": pd.to_numeric(d["depth_team"], errors="coerce").fillna(99.0)})
        c = cw.merge(d, on=["team", "chart_season", "chart_week"], how="inner")
        if injuries is not None and len(injuries):
            out = injuries[injuries["report_status"].isin(["Out", "Doubtful"])]
            out = out.dropna(subset=["gsis_id", "season", "week"])
            out = pd.DataFrame({"gsis_id": out["gsis_id"].astype(str), "season": out["season"].astype(int),
                                "week": out["week"].astype(int)}).drop_duplicates()
            c = c.merge(out.assign(_out=1), on=["gsis_id", "season", "week"], how="left")
            c = c[c["_out"].isna()]
        c = c.sort_values(["_i", "_dt"], kind="stable").drop_duplicates("_i")
        qb1.iloc[c["_i"].to_numpy()] = c["gsis_id"].to_numpy()
    if override:
        keys = zip(tw["season"], tw["week"], tw["team"])
        qb1 = pd.Series([override.get((int(s), int(w), t), q) for (s, w, t), q in zip(keys, qb1)],
                        index=tw.index, dtype=object)
    return tw.assign(qb1_id=qb1)


def team_qb_features(team_weeks: pd.DataFrame, qb1: pd.DataFrame, qga: pd.DataFrame, H: float, k: float,
                     repl: dict) -> pd.DataFrame:
    """Per team-week: QB1's profile (`qb_*`), `qb_ratio_*` (QB1 / the passer mix
    of the team's previous 10 games, weights att x 0.5^(games_ago / 4), mix
    profiles as of (S, w); clamped; NaN with no prior games) and `qb_changed`
    (QB1 vs the previous game's attempts leader). Row-aligned with `team_weeks`."""
    tw = team_weeks[_KEYS].reset_index(drop=True).astype({"season": int, "week": int})
    q1 = qb1[_KEYS + ["qb1_id"]].astype({"season": int, "week": int}).drop_duplicates(_KEYS)
    tw = tw.merge(q1, on=_KEYS, how="left")
    tw["_i"] = np.arange(len(tw))
    tw["_ord"] = _ord(tw)
    out = tw[_KEYS].copy()
    for c in (*PROFILE_COLS, *RATIO_COLS, "qb_changed"):
        out[c] = np.nan
    has = tw["qb1_id"].notna().to_numpy()
    if has.any():
        p1 = profile_asof(qga, tw.loc[has, ["qb1_id", "season", "week"]].rename(columns={"qb1_id": "player_id"}),
                          H, k, repl)
        for c in PROFILE_COLS:
            out.loc[has, c] = p1[c].to_numpy()
    if not len(qga) or not len(tw):
        return out

    # team game index (all seasons), latest prior game per key
    tg = qga[["team", "season", "week"]].drop_duplicates().assign(_ord=lambda x: _ord(x))
    tg = tg.sort_values(["team", "_ord"], kind="stable")
    tg["_gi"] = tg.groupby("team").cumcount()
    m = pd.merge_asof(tw[["team", "_ord", "_i"]].sort_values("_ord", kind="stable"),
                      tg[["team", "_ord", "_gi"]].sort_values("_ord", kind="stable"),
                      on="_ord", by="team", direction="backward", allow_exact_matches=False)
    last = m.sort_values("_i")["_gi"].to_numpy(dtype=float)

    # passers per team game; the attempts leader of each game
    pas = qga[["team", "season", "week", "player_id", "att"]].assign(_ord=lambda x: _ord(x))
    pas = pas.merge(tg[["team", "_ord", "_gi"]], on=["team", "_ord"], how="left")
    pas = pas.groupby(["team", "_gi", "player_id"], as_index=False)["att"].sum()
    lead = pas.sort_values(["team", "_gi", "att", "player_id"], ascending=[True, True, False, True],
                           kind="stable").drop_duplicates(["team", "_gi"])

    ok = np.isfinite(last)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    # qb_changed: QB1 vs the previous game's attempts leader
    prev = pd.DataFrame({"_i": idx, "team": tw["team"].to_numpy()[idx], "_gi": last[idx].astype(int)})
    prev = prev.merge(lead[["team", "_gi", "player_id"]], on=["team", "_gi"], how="left")
    q = tw["qb1_id"].to_numpy(dtype=object)[prev["_i"].to_numpy()]
    known = pd.notna(q) & prev["player_id"].notna().to_numpy()
    out.loc[prev["_i"].to_numpy()[known], "qb_changed"] = (
        q[known] != prev["player_id"].to_numpy(dtype=object)[known]).astype(float)

    # passer mix over the previous MIX_GAMES team games
    ga = np.tile(np.arange(1, MIX_GAMES + 1), len(idx))
    ex = pd.DataFrame({"_i": np.repeat(idx, MIX_GAMES), "games_ago": ga})
    ex["team"] = tw["team"].to_numpy()[ex["_i"]]
    ex["_gi"] = np.repeat(last[idx].astype(int), MIX_GAMES) - ga + 1
    ex = ex[ex["_gi"] >= 0].merge(pas, on=["team", "_gi"], how="inner")
    ex["_w"] = ex["att"] * np.power(0.5, ex["games_ago"] / MIX_HALFLIFE)
    mw = ex.groupby(["_i", "player_id"], as_index=False)["_w"].sum()
    mw["season"] = tw["season"].to_numpy()[mw["_i"]]
    mw["week"] = tw["week"].to_numpy()[mw["_i"]]
    pk = mw[["player_id", "season", "week"]].drop_duplicates().reset_index(drop=True)
    prof = profile_asof(qga, pk, H, k, repl)
    mw = mw.merge(prof, on=["player_id", "season", "week"], how="left")
    cols = list(RATIO_COLS.values())
    num = mw[cols].mul(mw["_w"], axis=0).groupby(mw["_i"]).sum()
    den = mw.groupby("_i")["_w"].sum()
    mix = num.div(den.where(den > 0), axis=0)
    ii = mix.index.to_numpy()
    for rc, pc in RATIO_COLS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = out[pc].to_numpy(dtype=float)[ii] / mix[pc].to_numpy(dtype=float)
        out.loc[ii, rc] = np.clip(ratio, *RATIO_CLAMP)
    return out


GRID_H = (0.5, 1, 1.5, 2, 3, 4)
GRID_K = (25, 50, 100, 200, 400, 800)


def weighted_se(sq_err: np.ndarray, w: np.ndarray) -> float:
    """Standard error of the w-weighted mean of `sq_err`:
    sqrt(sum w_i^2 (e_i - mean)^2) / sum w_i (non-finite errors dropped)."""
    e, w = np.asarray(sq_err, dtype=float), np.asarray(w, dtype=float)
    ok = np.isfinite(e) & np.isfinite(w)
    e, w = e[ok], w[ok]
    if not len(e) or w.sum() <= 0:
        return float("nan")
    m = np.sum(w * e) / np.sum(w)
    return float(np.sqrt(np.sum(w ** 2 * (e - m) ** 2)) / np.sum(w))


def select_one_se(grid: list[dict], se: float) -> tuple[dict, dict]:
    """(best, chosen): best = the min-mse grid point (ties -> first); chosen =
    among points with mse <= best.mse + se, the largest H, then the largest k
    (ties -> first). A non-finite `se` counts as 0."""
    pts = [g for g in grid if np.isfinite(g["mse"])]
    if not pts:
        return grid[0], grid[0]
    best = pts[0]
    for g in pts[1:]:
        if g["mse"] < best["mse"]:
            best = g
    tol = best["mse"] + (se if np.isfinite(se) else 0.0)
    chosen = best
    for g in pts:
        if g["mse"] <= tol and (g["H"], g["k"]) > (chosen["H"], chosen["k"]):
            chosen = g
    return best, chosen


def tune(qga: pd.DataFrame, seasons: list[int], grid_h=GRID_H, grid_k=GRID_K) -> dict:
    """Choose (H, k) by attempt-weighted MSE of each QB-game's `ypa_adj`
    (att >= 10, season in `seasons`) predicted by `profile_asof` at that game
    (replacement from seasons before the game's season), by the one-standard-
    error rule: SE = weighted SE of the best point's per-game squared errors;
    among points with mse <= best + SE, the largest H, then the largest k."""
    sc = qga[qga["season"].isin(list(seasons)) & (qga["att"] >= 10) & qga["ypa_adj"].notna()]
    keys = sc[["player_id", "season", "week"]].reset_index(drop=True)
    y = sc["ypa_adj"].to_numpy(dtype=float)
    wt = sc["att"].to_numpy(dtype=float)
    rmap = {int(s): replacement(qga, int(s)) for s in pd.unique(keys["season"])}
    repl = {a: keys["season"].map(lambda s, a=a: rmap[int(s)][a]).astype(float) for a in ADJ}
    grid: list[dict] = []
    errs: list[np.ndarray] = []
    for H in grid_h:
        ws = _weighted_sums(qga, keys, float(H))
        for k in grid_k:
            pred = _shrink(ws, float(k), repl)["qb_ypa"].to_numpy()
            e = (pred - y) ** 2
            ok = np.isfinite(e)
            mse = float(np.sum(wt[ok] * e[ok]) / np.sum(wt[ok])) if ok.any() else float("nan")
            grid.append({"H": float(H), "k": float(k), "mse": mse})
            errs.append(e)
    best, _ = select_one_se(grid, 0.0)
    se = weighted_se(errs[next(i for i, g in enumerate(grid) if g is best)], wt)
    best, chosen = select_one_se(grid, se)
    return {"H": chosen["H"], "k": chosen["k"], "mse": chosen["mse"],
            "best": {"H": best["H"], "k": best["k"], "mse": best["mse"]},
            "se": se, "rule": "1se", "n_games": int(len(keys)), "grid": grid}
