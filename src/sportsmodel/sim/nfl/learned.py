"""Learned volume / efficiency inputs for the NFL sim (props ML, rung A).

The Monte Carlo engine is unchanged; these models only replace some of its
INPUTS:

* team volume -- ``team_pass`` / ``team_rush`` (Poisson HGB on
  ``y_team_pass_att`` / ``y_team_rush_att``, team-table features) replace
  ``TeamRates.pass_att_pg`` / ``rush_att_pg``;
* player shares -- ``targets`` / ``carries`` (Poisson HGB on ``y_targets`` /
  ``y_carries``) are converted to shares renormalized over the active set;
* efficiency (only with the ``"efficiency"`` toggle) -- ``ypr``,
  ``catch_rate``, ``ypc`` (squared-error HGB, touch-weighted) replace the
  player's per-touch means; ``ypt = catch_rate * ypr``. TD shares untouched.

Leakage contract
----------------
``fit_models`` filters ``(season, week) < upto`` ITSELF and requires the
label to be present (stub rows with NaN labels never train), so callers
cannot leak by passing the full feature table. Training is walk-forward only
(no random splits, no early stopping with a random validation fraction).
NaN features stay NaN (HistGradientBoosting routes missing values natively);
a column that is 100% NaN in a model's training slice is dropped for that fit
and the used columns are recorded on the fitted model, so prediction always
uses exactly the fitted columns.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance

from sportsmodel.sim.nfl.spec import NflGameSpec, PlayerInput, TeamRates

RUNG_PREFIXES: dict[str, tuple[str, ...]] = {
    "volume": ("p_", "ngs_", "tm_", "op_", "st_"),
    "context": ("cx_",),
    "market": ("mk_",),
}

# Monotonic constraints where the direction is certain (applied only when the
# column is among the model's fitted features).
_TARGETS_MONO = {"p_target_share_ewm": 1}
_CARRIES_MONO = {"p_carry_share_ewm": 1}

TUNE_DECAYS: tuple[float, ...] = (1.0, 0.8, 0.6)
TUNE_MAX_ITERS: tuple[int, ...] = (150, 300)

CATCH_RATE_BOUNDS = (0.05, 1.0)
# Learned yards-per-touch floor: a <= 0 ypr/ypc prediction would give ypt <= 0
# and can make the kernel fail (a silently skipped game biases the gate).
YARDS_PER_TOUCH_FLOOR = 0.5


def feature_columns(df: pd.DataFrame, toggles: frozenset[str]) -> list[str]:
    """Model feature columns of ``df`` for the given rung toggles, in df order.

    ``volume`` prefixes are always included (volume is on whenever any
    learned model is used); ``context`` adds ``cx_``, ``market`` adds ``mk_``.
    ``efficiency`` toggles the efficiency MODELS, not columns. Keys, ``y_*``
    labels and any non-prefixed column are excluded by construction.
    """
    prefixes = list(RUNG_PREFIXES["volume"])
    for rung in ("context", "market"):
        if rung in toggles:
            prefixes.extend(RUNG_PREFIXES[rung])
    pref = tuple(prefixes)
    return [c for c in df.columns if c.startswith(pref) and not c.startswith("y_")]


@dataclass
class FittedModel:
    """An HGB estimator plus the exact columns it was fitted on."""

    estimator: HistGradientBoostingRegressor
    cols: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X[self.cols])


@dataclass
class LearnedModels:
    """Fitted rung-A models. A ``None`` model had no training rows.

    ``player_cols`` / ``team_cols`` are the CANDIDATE feature columns; each
    ``FittedModel.cols`` is the subset actually used by that fit.
    ``share_fallbacks`` counts sides whose shares ``apply_to_spec`` left
    unchanged because some active player had no feature row.
    """

    team_pass: FittedModel | None
    team_rush: FittedModel | None
    targets: FittedModel | None
    carries: FittedModel | None
    eff: dict[str, FittedModel | None] | None
    player_cols: list[str]
    team_cols: list[str]
    share_fallbacks: int = field(default=0)


def _hgb(loss: str, max_iter: int, monotonic_cst: dict[str, int] | None
         ) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, categorical_features="from_dtype",
        random_state=0, max_iter=max_iter, early_stopping=False,
        monotonic_cst=monotonic_cst,
    )


def _before(df: pd.DataFrame, upto: tuple[int, int]) -> pd.Series:
    s, w = upto
    return (df["season"] < s) | ((df["season"] == s) & (df["week"] < w))


def _fit_one(X: pd.DataFrame, y: pd.Series, weight: np.ndarray, *, loss: str,
             max_iter: int, mono: dict[str, int] | None) -> FittedModel | None:
    """Fit one HGB on a prepared slice; drop 100%-NaN columns; None if empty."""
    if len(X) == 0:
        return None
    cols = [c for c in X.columns if X[c].notna().any()]
    if not cols:
        return None
    cst = {k: v for k, v in (mono or {}).items() if k in cols} or None
    est = _hgb(loss, max_iter, cst)
    est.fit(X[cols], y.to_numpy(dtype=float), sample_weight=weight)
    return FittedModel(estimator=est, cols=cols)


def _fit_count(df: pd.DataFrame, label: str, cols: list[str], upto, test_season,
               decay, max_iter, mono) -> FittedModel | None:
    tr = df[_before(df, upto) & df[label].notna()]
    w = decay ** (test_season - tr["season"].to_numpy(dtype=float))
    return _fit_one(tr[cols], tr[label], w, loss="poisson", max_iter=max_iter, mono=mono)


# efficiency model -> (numerator label, denominator/weight label)
_EFF_SPECS: dict[str, tuple[str, str]] = {
    "ypr": ("y_rec_yds", "y_receptions"),
    "catch_rate": ("y_receptions", "y_targets"),
    "ypc": ("y_rush_yds", "y_carries"),
}


def _fit_eff(df: pd.DataFrame, num: str, den: str, cols: list[str], upto,
             test_season, decay, max_iter) -> FittedModel | None:
    m = _before(df, upto) & df[num].notna() & (df[den] > 0)
    tr = df[m]
    y = tr[num] / tr[den]
    w = tr[den].to_numpy(dtype=float) * decay ** (
        test_season - tr["season"].to_numpy(dtype=float))
    return _fit_one(tr[cols], y, w, loss="squared_error", max_iter=max_iter, mono=None)


def fit_models(player_df: pd.DataFrame, team_df: pd.DataFrame,
               toggles: frozenset[str], *, upto: tuple[int, int],
               test_season: int, decay: float, max_iter: int) -> LearnedModels:
    """Fit rung-A models on rows with ``(season, week) < upto`` and labels present.

    Sample weight is ``decay ** (test_season - season)`` (times the touch
    count for efficiency models). Efficiency models are fitted only when
    ``"efficiency" in toggles``.
    """
    pcols = feature_columns(player_df, toggles)
    tcols = feature_columns(team_df, toggles)
    common = dict(upto=upto, test_season=test_season, decay=decay, max_iter=max_iter)
    targets = _fit_count(player_df, "y_targets", pcols, mono=_TARGETS_MONO, **common)
    carries = _fit_count(player_df, "y_carries", pcols, mono=_CARRIES_MONO, **common)
    team_pass = _fit_count(team_df, "y_team_pass_att", tcols, mono=None, **common)
    team_rush = _fit_count(team_df, "y_team_rush_att", tcols, mono=None, **common)
    eff = None
    if "efficiency" in toggles:
        eff = {name: _fit_eff(player_df, num, den, pcols, **common)
               for name, (num, den) in _EFF_SPECS.items()}
    return LearnedModels(team_pass=team_pass, team_rush=team_rush,
                         targets=targets, carries=carries, eff=eff,
                         player_cols=pcols, team_cols=tcols)


def tune(player_df: pd.DataFrame, test_season: int, toggles: frozenset[str]
         ) -> tuple[float, int]:
    """Inner walk-forward search for ``(decay, max_iter)``.

    Trains the targets model on seasons < ``test_season - 1`` (weights
    relative to the validation season) and scores mean Poisson deviance on
    labelled rows of season ``test_season - 1``. Ties keep the first grid
    point. Raises ``ValueError`` if either slice is empty.
    """
    val_season = test_season - 1
    cols = feature_columns(player_df, toggles)
    val = player_df[(player_df["season"] == val_season) & player_df["y_targets"].notna()]
    n_train = int(((player_df["season"] < val_season) & player_df["y_targets"].notna()).sum())
    if val.empty or n_train == 0:
        raise ValueError(
            f"tune({test_season}): need labelled rows before {val_season} "
            f"(have {n_train}) and in {val_season} (have {len(val)})")
    best: tuple[float, float, int] | None = None
    for decay in TUNE_DECAYS:
        for max_iter in TUNE_MAX_ITERS:
            m = _fit_count(player_df, "y_targets", cols, upto=(val_season, 0),
                           test_season=val_season, decay=decay,
                           max_iter=max_iter, mono=_TARGETS_MONO)
            if m is None:
                continue
            dev = mean_poisson_deviance(val["y_targets"].to_numpy(dtype=float),
                                        m.predict(val))
            if best is None or dev < best[0]:
                best = (dev, decay, max_iter)
    if best is None:
        raise ValueError(f"tune({test_season}): no model could be fitted")
    return best[1], best[2]


def _team_rates(rates: TeamRates, team: str, models: LearnedModels,
                team_rows: pd.DataFrame) -> TeamRates:
    row = team_rows[team_rows["team"] == team].head(1)
    if row.empty:
        return rates
    changes = {}
    if models.team_pass is not None:
        changes["pass_att_pg"] = float(models.team_pass.predict(row)[0])
    if models.team_rush is not None:
        changes["rush_att_pg"] = float(models.team_rush.predict(row)[0])
    return dataclasses.replace(rates, **changes) if changes else rates


def _side_players(players: list[PlayerInput], models: LearnedModels,
                  player_rows: pd.DataFrame, questionable: set[str],
                  q_weight: float) -> list[PlayerInput]:
    ids = [p.player_id for p in players]
    rows = (player_rows[player_rows["player_id"].isin(ids)]
            .drop_duplicates("player_id").set_index("player_id"))
    have = [pid for pid in ids if pid in rows.index]
    X = rows.loc[have]
    changes: dict[str, dict[str, float]] = {pid: {} for pid in ids}

    # --- shares: learned counts -> shares over the WHOLE active set --------
    all_rows = len(have) == len(ids) and len(ids) > 0
    if not all_rows:
        models.share_fallbacks += 1
    for model, attr in ((models.targets, "target_share"), (models.carries, "carry_share")):
        if model is None or not all_rows:
            continue
        pred = np.clip(model.predict(X), 0.0, None)
        pred = pred * np.array([q_weight if pid in questionable else 1.0 for pid in have])
        total = float(pred.sum())
        if total > 0:
            for pid, v in zip(have, pred / total):
                changes[pid][attr] = float(v)

    # --- efficiency: per player with a row ---------------------------------
    if models.eff and have:
        for name, model in models.eff.items():
            if model is None:
                continue
            vals = model.predict(X)
            if name == "catch_rate":
                vals = np.clip(vals, *CATCH_RATE_BOUNDS)
            else:  # ypr / ypc
                vals = np.maximum(vals, YARDS_PER_TOUCH_FLOOR)
            for pid, v in zip(have, vals):
                changes[pid][name] = float(v)

    out = []
    for p in players:
        ch = changes[p.player_id]
        if "ypr" in ch or "catch_rate" in ch:
            ch["ypt"] = ch.get("catch_rate", p.catch_rate) * ch.get("ypr", p.ypr)
        out.append(dataclasses.replace(p, **ch) if ch else p)
    return out


def apply_to_spec(spec: NflGameSpec, models: LearnedModels,
                  player_rows: pd.DataFrame, team_rows: pd.DataFrame,
                  questionable: set[str], q_weight: float) -> NflGameSpec:
    """Return a copy of ``spec`` with learned team volume, shares, efficiency.

    ``player_rows`` / ``team_rows`` are the (season, week) feature rows of
    both teams. Team rates change only when that team's row exists. A side's
    shares are replaced only if EVERY active player has a feature row
    (otherwise left unchanged and counted in ``models.share_fallbacks``);
    questionable players' predicted counts are multiplied by ``q_weight``
    before renormalizing. Efficiency is replaced per player with a row;
    learned ``ypr`` / ``ypc`` are floored at ``YARDS_PER_TOUCH_FLOOR``.
    """
    return dataclasses.replace(
        spec,
        home=_team_rates(spec.home, spec.home_team, models, team_rows),
        away=_team_rates(spec.away, spec.away_team, models, team_rows),
        home_players=_side_players(spec.home_players, models, player_rows,
                                   questionable, q_weight),
        away_players=_side_players(spec.away_players, models, player_rows,
                                   questionable, q_weight),
    )
