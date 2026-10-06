"""Residual check (REPORT ONLY -- nothing here ships). Regularised models on v3's own training
residuals, using features v3 does not use, scored out of sample on the held-out seasons.

Features (all leak-free; each is home minus away):
  cfbd_elo_diff          CFBD's own PRE-game Elo (games.homePregameElo / awayPregameElo)
  fpi_diff_prev, srs_diff_prev       CFBD FPI / SRS of the PREVIOUS season (end-of-season ratings are
                         only ever joined to the next season)
  returning_usage_diff   priors.parquet returning_starters (CFBD returning-usage share)
  std_down_*, pass_down_*, line_yards, stuff_rate, power_success   season-to-date raw means of
                         earlier games (down/distance splits and run-game shape)
  unused_<ctx>, unused_pt_<f>   v3 context terms / efficiency points-map features fitted to weight 0
No market-derived field (CFBD pregame win probability, spreads, lines) is ever used.

A feature is a v4 candidate when, fitted alone on the training residuals, it cuts held-out MAE by
at least MIN_GAIN points in EVERY held-out season (the spec's "all three seasons" rule, with a
materiality floor so a 0.0001 coincidence does not count).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV

from .efficiency import POINT_FEATURES

DOWN_COLS = {"std_down_success": "off_std_down_success", "pass_down_success": "off_pass_down_success",
             "std_down_ppa": "off_std_down_ppa", "pass_down_ppa": "off_pass_down_ppa",
             "line_yards": "off_line_yards", "stuff_rate": "off_stuff_rate",
             "power_success": "off_power_success"}
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MIN_GAIN = 0.005      # points of held-out MAE a feature must save in EVERY held-out season to be a v4 candidate


def prior_game_means(adv: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """For every regular-season team-game: the team's mean of each DOWN_COLS metric over its EARLIER
    games of the season (NaN for the opener). Rows: game_id, team, <name>."""
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    a = adv[adv["season_type"].fillna("regular") == "regular"] if "season_type" in adv else adv
    a = a.drop(columns=["week"]).merge(reg[["game_pk", "week"]].rename(columns={"game_pk": "game_id"}),
                                       on="game_id", how="inner")
    a = a.sort_values(["season", "team", "week"]).reset_index(drop=True)
    out = a[["game_id", "team"]].copy()
    grp = a.groupby(["season", "team"])
    for name, col in DOWN_COLS.items():
        out[name] = grp[col].transform(lambda s: s.expanding().mean().shift(1)) if col in a else np.nan
    return out


def residual_features(table: pd.DataFrame, means: pd.DataFrame, meta: pd.DataFrame | None,
                      priors_rows: dict, prior_ratings: pd.DataFrame | None,
                      margin_coefs: dict, total_coefs: dict, points_coefs: dict) -> pd.DataFrame:
    """Residual-check feature frame aligned to `table` rows (home - away; unknown -> 0.0)."""
    out = pd.DataFrame(index=table.index)
    pk, home, away, season = table["game_pk"], table["home_team"], table["away_team"], table["season"]
    if meta is not None:
        m = meta.drop_duplicates("game_id").set_index("game_id")
        out["cfbd_elo_diff"] = pk.map(m["home_pregame_elo"]) - pk.map(m["away_pregame_elo"])
    if prior_ratings is not None:
        pr = prior_ratings.set_index(["season", "team"])
        for col, name in (("fpi", "fpi_diff_prev"), ("srs", "srs_diff_prev")):
            lut = pr[col].to_dict()
            out[name] = [lut.get((s - 1, h), np.nan) - lut.get((s - 1, a), np.nan)
                         for s, h, a in zip(season, home, away)]
    usage = {(s, r["team_espn_id"]): r.get("returning_starters")
             for s, rows in priors_rows.items() for r in rows}
    out["returning_usage_diff"] = [_f(usage.get((s, h))) - _f(usage.get((s, a)))
                                   for s, h, a in zip(season, home, away)]
    idx = means.set_index(["game_id", "team"])
    for name in DOWN_COLS:
        lut = idx[name].to_dict()
        out[f"{name}_diff"] = [lut.get((g, h), np.nan) - lut.get((g, a), np.nan)
                               for g, h, a in zip(pk, home, away)]
    for c, v in {**margin_coefs, **total_coefs}.items():
        if v == 0.0 and c in table:
            out[f"unused_{c}"] = table[c]
    for f in POINT_FEATURES:
        if points_coefs.get(f, 0.0) == 0.0:
            out[f"unused_pt_{f}"] = table[f"h_{f}"] - table[f"a_{f}"]
    return out.astype(float).fillna(0.0)


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def _mae_change(resid: np.ndarray, pred: np.ndarray, base: float) -> float:
    """MAE(resid - pred) minus MAE of the constant train-mean prediction (negative = the features
    help). The constant baseline keeps a pure intercept shift from counting as a feature win."""
    return float(np.mean(np.abs(resid - pred)) - np.mean(np.abs(resid - base)))


def check_target(feat: pd.DataFrame, resid: np.ndarray, seasons: np.ndarray,
                 train_seasons, test_seasons) -> dict:
    """Ridge + shallow gradient boosting fit on the train residuals; held-out R^2, MAE change
    vs the constant train-mean residual (negative = better) combined and per season, top features and v4 candidates."""
    tr, te = np.isin(seasons, list(train_seasons)), np.isin(seasons, list(test_seasons))
    feat = feat.loc[:, feat[tr].std(ddof=0) > 0]            # constant (all-neutral) columns carry nothing
    cols = list(feat.columns)
    mu, sd = feat[tr].mean(), feat[tr].std(ddof=0).replace(0, 1.0)
    z = ((feat - mu) / sd).to_numpy(float)
    base = float(resid[tr].mean())

    def summarize(pred_te: np.ndarray) -> dict:
        r = resid[te]
        sst = float(np.sum((r - base) ** 2))
        out = {"r2_oos": float(1 - np.sum((r - pred_te) ** 2) / sst) if sst > 0 else float("nan"),
               "mae_change": _mae_change(r, pred_te, base),
               "mae_change_by_season": {int(s): _mae_change(r[seasons[te] == s], pred_te[seasons[te] == s], base)
                                        for s in sorted(set(seasons[te]))}}
        return out

    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(z[tr], resid[tr])
    res = {"ridge": {**summarize(ridge.predict(z[te])), "alpha": float(ridge.alpha_),
                     "top_features": [{"feature": cols[i], "std_coef": float(ridge.coef_[i])}
                                      for i in np.argsort(-np.abs(ridge.coef_))[:5]]}}
    hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                        l2_regularization=1.0, random_state=0).fit(z[tr], resid[tr])
    imp = permutation_importance(hgb, z[te], resid[te], n_repeats=3, random_state=0,
                                 scoring="neg_mean_absolute_error").importances_mean
    res["hgb"] = {**summarize(hgb.predict(z[te])),
                  "top_features": [{"feature": cols[i], "perm_importance": float(imp[i])}
                                   for i in np.argsort(-imp)[:5]]}
    cands = []
    for i, c in enumerate(cols):
        m = Ridge(alpha=float(ridge.alpha_)).fit(z[tr][:, [i]], resid[tr])
        per = summarize(m.predict(z[te][:, [i]]))["mae_change_by_season"]
        if per and all(v < -MIN_GAIN for v in per.values()):
            cands.append({"feature": c, "mae_change_by_season": per})
    res["v4_candidates"] = cands
    return res


def run_residual_check(feat: pd.DataFrame, resid_margin, resid_total, seasons,
                       train_seasons, test_seasons) -> dict:
    seasons = np.asarray(seasons)
    return {"features": list(feat.columns),
            "margin": check_target(feat, np.asarray(resid_margin, float), seasons, train_seasons, test_seasons),
            "total": check_target(feat, np.asarray(resid_total, float), seasons, train_seasons, test_seasons),
            "note": "report only: nothing from the residual models ships in cfb-ratings-v3"}
