"""v2-vs-v1 props-ML gate checks (matchup / QB profile spec §6). PURE, no IO.

Inputs are the two model versions' SERVED composites (``served__<tag>{name}.parquet``
from ``scripts/train_props_ml_b.py``: season, week, home, player_id, market,
mean, rps, pit, actual; one row per population key) and the player feature
table (``player_week_features.parquet``: player_id, season, week, team,
opponent, position, is_stub, labels ``y_*``, ``qb_changed``,
``qb_ratio_ypa``, ``mx_pass_minus_rush``).

- ``paired_served``: v1 (``_b``) vs v2 (``_c``) per key, the frame the RPS
  rule and the QB-change check score.
- ``qb_change_mask`` / ``qb_change_check`` (sub-check 1): pass_yds records of
  QB-change team-weeks; v2 must improve RPS with a season-week cluster
  bootstrap CI excluding 0.
- ``player_baseline`` / ``matchup_response`` (sub-check 2): per opponent
  ``mx_pass_minus_rush`` quintile (+ = pass D weak relative to run D; ruling
  R1), the served mean's deviation from the player's own baseline must move
  in the expected direction in BOTH tails, match the actual deviation's sign,
  and track the actual deviations better than v1.
- ``verdict``: the spec §6 ship rule.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .. import game_gate, props_eval
from .dist_models import LABELS

KEY = ["season", "week", "player_id", "market"]
QB_RATIO_BAND = (0.95, 1.05)
QB_CHANGE_MARKET = "pass_yds"
BASELINE_GAMES = 8
BASELINE_MIN_GAMES = 3
# (position, market) groups of the matchup check; the sign each must show in
# the TOP quintile of mx_pass_minus_rush (strong run D / weak pass D). The
# bottom quintile needs the reverse.
MATCHUP_GROUPS: tuple[tuple[str, str], ...] = (("RB", "rush_yds"), ("QB", "pass_yds"),
                                               ("WR", "rec_yds"), ("TE", "rec_yds"))
TOP_SIGN = {"RB": -1, "QB": 1, "WR": 1, "TE": 1}
N_QUINTILES = 5
FINAL_SEASON = 2025
ECE_TOL = 0.005
GAME_TOL = 1.005   # game-level metrics: v2 <= v1 x (1 + 0.5 %)
STUB_COL = "is_stub"


def _key_frame(records: pd.DataFrame) -> pd.DataFrame:
    return records[["season", "week", "player_id"]].astype(
        {"season": int, "week": int, "player_id": str})


def _feat_keys(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    f = feats[["season", "week", "player_id"] + cols].copy()
    f = f.astype({"season": int, "week": int, "player_id": str})
    return f.drop_duplicates(["season", "week", "player_id"], keep="last")


# ---- pairing ---------------------------------------------------------------------------------

def paired_served(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, player_id, market): season, week, home,
    player_id, market, rps_b / pit_b / mean_b (v1), rps_c / pit_c / mean_c
    (v2), actual and ``cluster`` = ``f"{season}-{week}-{home}"`` (as
    ``props_eval.paired_frame``). ValueError when the two key sets differ or a
    key repeats (both versions serve the SAME baseline population)."""
    k1, k2 = [set(map(tuple, d[KEY].itertuples(index=False))) for d in (v1, v2)]
    if len(k1) != len(v1) or len(k2) != len(v2):
        raise ValueError("served records repeat a (season, week, player_id, market) key")
    if k1 != k2:
        raise ValueError(f"v1 and v2 served key sets differ: {len(k1 - k2)} only in v1, "
                         f"{len(k2 - k1)} only in v2")
    a = v1[KEY + ["home", "rps", "pit", "mean"]].rename(
        columns={"rps": "rps_b", "pit": "pit_b", "mean": "mean_b"})
    b = v2[KEY + ["rps", "pit", "mean", "actual"]].rename(
        columns={"rps": "rps_c", "pit": "pit_c", "mean": "mean_c"})
    out = a.merge(b, on=KEY, how="inner").sort_values(KEY, kind="stable").reset_index(drop=True)
    out["cluster"] = (out["season"].astype(int).astype(str) + "-"
                      + out["week"].astype(int).astype(str) + "-" + out["home"].astype(str))
    return out


# ---- sub-check 1: QB-change games ------------------------------------------------------------

def qb_change_mask(records: pd.DataFrame, feats: pd.DataFrame) -> pd.Series:
    """True where the record's team-week is a QB-change week: any player row
    of that (season, week, team) has ``qb_changed == 1`` or ``qb_ratio_ypa``
    outside ``QB_RATIO_BAND`` (NaN is not outside). The team comes from the
    record's own (season, week, player_id) feature row; no row -> False.
    Indexed like ``records``."""
    f = feats[["season", "week", "team", "qb_changed", "qb_ratio_ypa"]].copy()
    lo, hi = QB_RATIO_BAND
    ratio = pd.to_numeric(f["qb_ratio_ypa"], errors="coerce")
    f["flag"] = ((pd.to_numeric(f["qb_changed"], errors="coerce") == 1)
                 | (ratio < lo) | (ratio > hi))
    f = f.astype({"season": int, "week": int})
    tw = f.groupby(["season", "week", "team"])["flag"].any().rename("qb_change").reset_index()
    team = _feat_keys(feats, ["team"])
    k = _key_frame(records).reset_index(drop=True)
    j = k.merge(team, on=["season", "week", "player_id"], how="left").merge(
        tw, on=["season", "week", "team"], how="left")
    return pd.Series(j["qb_change"].fillna(False).astype(bool).to_numpy(), index=records.index,
                     name="qb_change")


def qb_change_check(paired: pd.DataFrame, mask: pd.Series, *, n_boot: int = 1000,
                    seed: int = 0) -> dict:
    """pass_yds records of QB-change games (``mask``): mean RPS v2 - v1 with a
    season-week cluster bootstrap 95% CI; pass iff the CI's upper bound < 0."""
    m = mask.reindex(paired.index).fillna(False).astype(bool)
    sub = paired[m & (paired["market"] == QB_CHANGE_MARKET)].copy()
    if sub.empty:
        return {"n": 0, "n_clusters": 0, "rps_v1": float("nan"), "rps_v2": float("nan"),
                "diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "pass": False,
                "reason": "no QB-change pass_yds records"}
    sub["cluster"] = sub["season"].astype(int).astype(str) + "-" + sub["week"].astype(int).astype(str)
    stat = lambda d: float((d["rps_c"] - d["rps_b"]).mean())  # noqa: E731
    diff, lo, hi = props_eval.cluster_bootstrap(sub, stat, n_boot=n_boot, seed=seed)
    return {"n": int(len(sub)), "n_clusters": int(sub["cluster"].nunique()),
            "rps_v1": float(sub["rps_b"].mean()), "rps_v2": float(sub["rps_c"].mean()),
            "diff": float(diff), "lo": float(lo), "hi": float(hi), "pass": bool(hi < 0)}


# ---- sub-check 2: two-way matchup response ---------------------------------------------------

def player_baseline(records: pd.DataFrame, feats: pd.DataFrame,
                    n: int = BASELINE_GAMES) -> pd.Series:
    """The player's mean actual of the record's market (``LABELS``) over his
    previous ``n`` PLAYED games strictly before the record's (season, week):
    feature rows with a non-null label that are not stubs. NaN with fewer than
    ``BASELINE_MIN_GAMES`` such games (or an unknown market). Indexed like
    ``records``."""
    out = pd.Series(np.nan, index=records.index, dtype=float, name="baseline")
    stub = (feats[STUB_COL].fillna(False).astype(bool) if STUB_COL in feats.columns
            else pd.Series(False, index=feats.index))
    for market, recs in records.groupby("market"):
        label = LABELS.get(str(market))
        if label is None or label not in feats.columns:
            continue
        f = feats.loc[~stub & feats[label].notna(), ["player_id", "season", "week", label]]
        f = f.astype({"season": int, "week": int, "player_id": str})
        f = f.drop_duplicates(["player_id", "season", "week"], keep="last")
        f["t"] = f["season"] * 100 + f["week"]
        f = f.sort_values(["player_id", "t"])
        f["base"] = (f.groupby("player_id")[label]
                     .transform(lambda x: x.rolling(n, min_periods=BASELINE_MIN_GAMES).mean()))
        r = _key_frame(recs)
        r["t"] = r["season"] * 100 + r["week"]
        r["_idx"] = recs.index
        r = r.sort_values("t")
        j = pd.merge_asof(r, f[["player_id", "t", "base"]].sort_values("t"), on="t",
                          by="player_id", allow_exact_matches=False, direction="backward")
        out.loc[j["_idx"].to_numpy()] = j["base"].to_numpy(dtype=float)
    return out


def _pearson(x, y) -> float:
    """Pearson r; 0.0 when either side is constant (no linear association)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _sign(x: float) -> int:
    return 0 if not np.isfinite(x) or x == 0 else (1 if x > 0 else -1)


def matchup_response(v1: pd.DataFrame, v2: pd.DataFrame, feats: pd.DataFrame) -> dict:
    """Sub-check 2 (both directions). Records of the ``MATCHUP_GROUPS``
    (position from the record's feature row) are bucketed by the OPPONENT
    defense's as-of ``mx_pass_minus_rush`` (the record's own feature row: the
    builder puts the opponent's value, from games strictly before the week, on
    it) into quintiles 1..5 whose edges are the 20/40/60/80 % quantiles of the
    distinct (season, week, opponent) values. Per group x quintile:
    ``pred_dev = Σmean / Σbaseline - 1`` (v1 and v2) and ``act_dev =
    Σactual / Σbaseline - 1`` over records present in both versions with a
    baseline (``player_baseline``).

    ``pass_top``: in quintile 5 (strong run D / weak pass D) v2's RB dev < 0
    and QB / WR / TE dev > 0, each with the sign of act_dev; ``pass_bottom``:
    the reverse in quintile 1; ``corr_v1`` / ``corr_v2``: Pearson of pred_dev
    vs act_dev over the 20 cells (0.0 for a constant side); ``pass`` =
    pass_top and pass_bottom and corr_v2 > corr_v1."""
    p = paired_served(v1, v2)
    groups = {m for _, m in MATCHUP_GROUPS}
    p = p[p["market"].isin(groups)].reset_index(drop=True)
    fk = _feat_keys(feats, ["position", "opponent", "mx_pass_minus_rush"])
    p = p.merge(fk, on=["season", "week", "player_id"], how="left")
    want = set(MATCHUP_GROUPS)
    p = p[[(pos, m) in want for pos, m in zip(p["position"], p["market"])]].reset_index(drop=True)
    p["baseline"] = player_baseline(p, feats).to_numpy()
    p = p[p["baseline"].notna() & (p["baseline"] > 0) & p["mx_pass_minus_rush"].notna()]
    defs = p.drop_duplicates(["season", "week", "opponent"])["mx_pass_minus_rush"].to_numpy(float)
    edges = (np.quantile(defs, [0.2, 0.4, 0.6, 0.8]).tolist() if len(defs)
             else [float("nan")] * 4)
    p = p.assign(quintile=1 + (p["mx_pass_minus_rush"].to_numpy(float)[:, None]
                               > np.asarray(edges)[None, :]).sum(axis=1))
    table = []
    for pos, market in MATCHUP_GROUPS:
        for q in range(1, N_QUINTILES + 1):
            c = p[(p["position"] == pos) & (p["market"] == market) & (p["quintile"] == q)]
            sb = float(c["baseline"].sum())

            def dev(col: str) -> float:
                return float(c[col].sum()) / sb - 1.0 if sb > 0 else float("nan")

            table.append({"group": pos, "market": market, "quintile": q, "n": int(len(c)),
                          "pred_dev_v1": dev("mean_b"), "pred_dev_v2": dev("mean_c"),
                          "act_dev": dev("actual")})
    cell = {(t["group"], t["quintile"]): t for t in table}

    def tail_ok(q: int, direction: int) -> bool:
        for pos, _ in MATCHUP_GROUPS:
            t = cell[(pos, q)]
            want_sign = direction * TOP_SIGN[pos]
            if not (_sign(t["pred_dev_v2"]) == want_sign == _sign(t["act_dev"])):
                return False
        return True

    pass_top, pass_bottom = tail_ok(N_QUINTILES, 1), tail_ok(1, -1)
    act = [t["act_dev"] for t in table]
    corr_v1 = _pearson([t["pred_dev_v1"] for t in table], act)
    corr_v2 = _pearson([t["pred_dev_v2"] for t in table], act)
    return {"table": table, "edges": [float(e) for e in edges], "n_records": int(len(p)),
            "pass_top": bool(pass_top), "pass_bottom": bool(pass_bottom),
            "corr_v1": corr_v1, "corr_v2": corr_v2,
            "pass": bool(pass_top and pass_bottom and corr_v2 > corr_v1)}


# ---- ship rule -------------------------------------------------------------------------------

def ece_by_market(served: pd.DataFrame) -> dict[str, float]:
    """Decile PIT ECE per market of one version's served records."""
    return {str(m): float(props_eval.decile_ece(g["pit"].to_numpy(dtype=float)))
            for m, g in served.groupby("market")}


def game_level(game_diffs: pd.DataFrame, tol: float = GAME_TOL) -> dict:
    """Per ``game_gate.METRICS`` metric over games where both versions are
    scorable (``game_gate.paired_diffs(v2_records, v1_records)``: ``_ml`` =
    v2, ``_elo`` = v1): v1 / v2 means and pass iff v2 <= ``tol`` x v1."""
    metrics, reasons = {}, []
    for metric in game_gate.METRICS:
        col = game_gate._COLUMN[metric]
        sub = game_diffs.dropna(subset=[f"{col}_ml", f"{col}_elo"])
        if len(sub) == 0:
            metrics[metric] = {"n": 0, "v1": float("nan"), "v2": float("nan"), "pass": False}
            reasons.append(f"game {metric}: no paired scorable games")
            continue
        v1, v2 = float(sub[f"{col}_elo"].mean()), float(sub[f"{col}_ml"].mean())
        ok = v2 <= tol * v1
        metrics[metric] = {"n": int(len(sub)), "v1": v1, "v2": v2, "pass": bool(ok)}
        if not ok:
            reasons.append(f"game {metric}: v2 {v2:.5f} > {tol} x v1 {v1:.5f}")
    return {"metrics": metrics, "pass": not reasons, "reasons": reasons}


def verdict(paired_rps_df: pd.DataFrame, ece_v1: Mapping[str, float],
            ece_v2: Mapping[str, float], game_diffs: pd.DataFrame, qb_change_ci: Mapping,
            matchup: Mapping, *, final_season: int = FINAL_SEASON) -> dict:
    """Spec §6 ship rule, v2 (candidate ``_c``) vs v1 (``_b``) served composites:

    - RPS (as Props-2): ``props_eval.rung_decision`` passes on the verdict
      seasons pooled AND on ``final_season`` alone (``paired_rps_df`` needs
      ``season``; no final-season rows -> fail);
    - ECE: every market's v2 ECE <= v1's + ``ECE_TOL`` (``ece_v1`` / ``ece_v2``
      on the pooled verdict seasons; a market missing on either side fails);
    - game level: every ``game_gate.METRICS`` metric v2 <= ``GAME_TOL`` x v1
      (``game_level(game_diffs)``);
    - required sub-checks: QB-change (``qb_change_ci["pass"]``) and matchup
      response (``matchup["pass"]``).
    """
    reasons: list[str] = []
    pooled = props_eval.rung_decision(paired_rps_df)
    if not pooled["pass"]:
        reasons.append("RPS pooled: " + "; ".join(pooled["reasons"]))
    fin = paired_rps_df[paired_rps_df["season"] == final_season]
    s_final = props_eval.rung_decision(fin) if len(fin) else None
    if s_final is None:
        reasons.append(f"RPS {final_season} alone: no {final_season} records")
    elif not s_final["pass"]:
        reasons.append(f"RPS {final_season} alone: " + "; ".join(s_final["reasons"]))
    ece = {}
    for m in sorted(set(ece_v1) | set(ece_v2)):
        a, b = ece_v1.get(m), ece_v2.get(m)
        ok = a is not None and b is not None and b <= a + ECE_TOL
        ece[m] = {"v1": a, "v2": b, "pass": bool(ok)}
        if not ok:
            reasons.append(f"ECE {m}: v2 {b} > v1 {a} + {ECE_TOL}")
    game = game_level(game_diffs)
    reasons += game["reasons"]
    if not qb_change_ci.get("pass"):
        reasons.append("QB-change sub-check: pass_yds RPS CI does not exclude 0 on the better side")
    if not matchup.get("pass"):
        reasons.append(f"matchup sub-check: top {matchup.get('pass_top')}, bottom "
                       f"{matchup.get('pass_bottom')}, corr v2 {matchup.get('corr_v2')} vs v1 "
                       f"{matchup.get('corr_v1')}")
    return {"pooled": pooled, "season_2025": s_final, "ece": ece, "game": game,
            "qb_change": bool(qb_change_ci.get("pass")), "matchup": bool(matchup.get("pass")),
            "pass": not reasons, "reasons": reasons}

