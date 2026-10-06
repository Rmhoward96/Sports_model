"""Opponent-adjusted per-play efficiency ratings for CFB (cfb-ratings-v3). PURE.

One weighted ridge regression per metric per week over the season's completed
team-games:

    y(team vs opp) = league_mean + off[team] - def[opp] + hfa * home

(`home` is +1 for the home team, -1 for the away team, 0 at a neutral site; `off` > 0 is a
better offense, `def` > 0 a better defense.) Observations are weighted by plays (scoring
opportunities for `ppo`), normalised to mean 1, and ridge-shrunk toward the league mean with
`ridge` pseudo-games of strength.

Early season: the season-to-date ratings are blended with a prior = the team's previous-
season FINAL ratings (whole season, bowls included) shrunk by a per-team retention
k = clip(k0 + k_ret*z(returning production) + k_tal*z(talent), 0, 1). The blend weight on the
prior is `priors.prior_weight(games, decay)` (exponential half-life in games + floor, the
same family as priors_decay.json).

LEAK RULE: the state entering week W of season S is built from regular-season team-games
with schedule week < W of season S plus the previous season's final ratings; nothing else.
Appending a later game never changes an earlier state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportsmodel.cfb.priors import DecayConfig, prior_weight, season_features_z, zscore

METRICS = ("ppa", "pass_ppa", "rush_ppa", "success", "explosiveness", "havoc", "ppo")
LEAGUE_PLAYS_SEED = 70.0
LEAGUE_RUSH_SHARE_SEED = 0.5
RETENTION_BOUNDS = (0.0, 1.0)


@dataclass(frozen=True)
class EffConfig:
    ridge: float = 4.0            # pseudo-games of shrinkage toward the league mean
    half_life_games: float = 3.0  # prior weight halves after this many games
    prior_floor: float = 0.0      # prior weight never drops below this
    k0: float = 0.6               # baseline retention of last season's rating
    k_ret: float = 0.0            # + per z of returning production
    k_tal: float = 0.0            # + per z of talent composite

    @property
    def decay(self) -> DecayConfig:
        return DecayConfig(half_life_games=self.half_life_games, prior_floor=self.prior_floor)


@dataclass(frozen=True)
class MetricRatings:
    off: dict
    deff: dict
    mu: float
    hfa: float


EMPTY = MetricRatings({}, {}, float("nan"), 0.0)


@dataclass(frozen=True)
class EffState:
    ratings: dict        # metric -> MetricRatings
    games: dict          # team -> team-games used this season
    pace: dict           # team -> offensive plays per game
    rush_share: dict     # team -> rushes / (rushes + passes)
    lg_plays: float
    lg_rush_share: float


# ------------------------------------------------------------------ the ridge --

def ridge_adjust(obs: pd.DataFrame, ridge: float) -> MetricRatings:
    """obs columns: team, opponent, home, y, w. Rows with a NaN y or w (or w <= 0) are dropped,
    never treated as 0. No usable rows -> EMPTY."""
    obs = obs.dropna(subset=["y", "w"])
    obs = obs[obs["w"] > 0]
    if obs.empty:
        return EMPTY
    teams = sorted(set(obs["team"]) | set(obs["opponent"]))
    ix = {t: i for i, t in enumerate(teams)}
    t_n, n = len(teams), len(obs)
    w = obs["w"].to_numpy(float)
    w = w / w.mean()
    y = obs["y"].to_numpy(float)
    mu = float((w * y).sum() / w.sum())
    x = np.zeros((n, 2 * t_n + 1))
    r = np.arange(n)
    x[r, obs["team"].map(ix).to_numpy()] = 1.0
    x[r, t_n + obs["opponent"].map(ix).to_numpy()] = -1.0
    x[:, 2 * t_n] = obs["home"].to_numpy(float)
    xw = x * w[:, None]
    a = x.T @ xw
    diag = np.arange(2 * t_n)
    a[diag, diag] += ridge
    a[2 * t_n, 2 * t_n] += 1e-6            # hfa: unpenalised but kept invertible
    theta = np.linalg.solve(a, xw.T @ (y - mu))
    return MetricRatings({t: float(theta[ix[t]]) for t in teams},
                         {t: float(theta[t_n + ix[t]]) for t in teams}, mu, float(theta[2 * t_n]))


def metric_obs(games: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The (team, opponent, home, y, w) observations of one metric from an eff_games frame."""
    nan = pd.Series(np.nan, index=games.index)
    return pd.DataFrame({"team": games["team"], "opponent": games["opponent"], "home": games["home"],
                         "y": games.get(f"y_{metric}", nan), "w": games.get(f"w_{metric}", nan)})


def raw_state(games: pd.DataFrame, ridge: float) -> EffState:
    """Season-only (no prior) state from team-games."""
    if games.empty:
        return EffState({m: EMPTY for m in METRICS}, {}, {}, {}, LEAGUE_PLAYS_SEED, LEAGUE_RUSH_SHARE_SEED)
    ratings = {m: ridge_adjust(metric_obs(games, m), ridge) for m in METRICS}
    rush = games.groupby("team")["rush_plays"].sum()
    pas = games.groupby("team")["pass_plays"].sum()
    share = (rush / (rush + pas)).dropna()
    tot = float(games["rush_plays"].sum() + games["pass_plays"].sum())
    lg_rs = float(games["rush_plays"].sum() / tot) if tot > 0 else LEAGUE_RUSH_SHARE_SEED
    plays = games["plays"].dropna()
    return EffState(ratings, games.groupby("team").size().to_dict(),
                    games.groupby("team")["plays"].mean().dropna().to_dict(), share.to_dict(),
                    float(plays.mean()) if len(plays) else LEAGUE_PLAYS_SEED, lg_rs)


# ------------------------------------------------------------- prior + blend --

def retention_for(teams, z_ret: dict, z_tal: dict, cfg: EffConfig) -> dict:
    lo, hi = RETENTION_BOUNDS
    return {t: float(np.clip(cfg.k0 + cfg.k_ret * z_ret.get(t, 0.0) + cfg.k_tal * z_tal.get(t, 0.0), lo, hi))
            for t in teams}


def build_prior(final: EffState, retention: dict) -> EffState:
    """Previous-season final state with every team's ratings shrunk by its retention."""
    ratings = {}
    for m, r in final.ratings.items():
        ratings[m] = MetricRatings({t: retention.get(t, 0.0) * v for t, v in r.off.items()},
                                   {t: retention.get(t, 0.0) * v for t, v in r.deff.items()},
                                   r.mu, r.hfa)
    return EffState(ratings, {}, dict(final.pace), dict(final.rush_share), final.lg_plays, final.lg_rush_share)


def season_prior(eff_games: pd.DataFrame, season: int, priors_rows: list[dict],
                 talent: pd.DataFrame | None, cfg: EffConfig,
                 _cache: dict | None = None) -> EffState | None:
    """The prior entering `season`: last season's final ratings (all its games, bowls included)
    shrunk by returning production (priors.parquet `returning_pct`, season-S preseason value) and
    the talent composite. None when there is no previous-season data (no prior). `_cache`
    (optional) memoises the previous season's ridge by (ridge, season - 1)."""
    prev = eff_games[eff_games["season"] == season - 1]
    if prev.empty:
        return None
    fkey = ("final", cfg.ridge, season - 1)
    if _cache is not None and fkey in _cache:
        final = _cache[fkey]
    else:
        final = raw_state(prev, cfg.ridge)
        if _cache is not None:
            _cache[fkey] = final
    # only the target season's preseason rows: multi-season rows must not overwrite by team id
    rows = [r for r in (priors_rows or []) if r.get("season", season) == season]
    z_ret = {t: z["returning_pct"] for t, z in season_features_z(rows).items()} if rows else {}
    tal = {}
    if talent is not None and len(talent):
        cur = talent[talent["season"] == season].dropna(subset=["talent"])
        tal = zscore({str(t): float(v) for t, v in zip(cur["team"], cur["talent"])}) if len(cur) else {}
    return build_prior(final, retention_for(final.games.keys(), z_ret, tal, cfg))


def blend_state(cur: EffState, prior: EffState | None, cfg: EffConfig) -> EffState:
    """Per-team decay blend: w * prior + (1 - w) * season-to-date, w = prior_weight(games)."""
    if prior is None:
        return cur
    ratings = {}
    # home-field edge: blended with the same decay curve as the ratings, at the mean prior weight
    # of the teams that have played (the ridge leaves hfa unpenalised, so a thin early-season fit
    # is noisy); no games yet -> the prior's hfa
    w_bar = float(np.mean([prior_weight(n, cfg.decay) for n in cur.games.values()])) if cur.games else 1.0
    for m in METRICS:
        c, p = cur.ratings.get(m, EMPTY), prior.ratings.get(m, EMPTY)
        off, deff = {}, {}
        for t in set(c.off) | set(p.off):
            w = prior_weight(cur.games.get(t, 0), cfg.decay)
            off[t] = w * p.off.get(t, 0.0) + (1 - w) * c.off.get(t, 0.0)
            deff[t] = w * p.deff.get(t, 0.0) + (1 - w) * c.deff.get(t, 0.0)
        ratings[m] = MetricRatings(off, deff, c.mu if c.mu == c.mu else p.mu,
                                   w_bar * p.hfa + (1 - w_bar) * c.hfa if c.off else p.hfa)
    lg_plays = cur.lg_plays if cur.games else prior.lg_plays
    lg_rs = cur.lg_rush_share if cur.games else prior.lg_rush_share

    def mix(c: dict, p: dict, lg: float) -> dict:
        return {t: prior_weight(cur.games.get(t, 0), cfg.decay) * p.get(t, lg)
                + (1 - prior_weight(cur.games.get(t, 0), cfg.decay)) * c.get(t, lg)
                for t in set(c) | set(p)}

    return EffState(ratings, cur.games, mix(cur.pace, prior.pace, lg_plays),
                    mix(cur.rush_share, prior.rush_share, lg_rs), lg_plays, lg_rs)


def state_before(eff_games: pd.DataFrame, season: int, week: int,
                 prior: EffState | None, cfg: EffConfig, _cache: dict | None = None) -> EffState:
    """The efficiency state ENTERING schedule week `week` of `season`: regular-season team-games
    with week < `week` (NaN-week postseason rows never qualify) blended with `prior`.
    `_cache` (optional) memoises the season-only ridge by (ridge, season, week)."""
    key = (cfg.ridge, season, week)
    if _cache is not None and key in _cache:
        cur = _cache[key]
    else:
        rows = eff_games[(eff_games["season"] == season) & (eff_games["week"] < week)]
        cur = raw_state(rows, cfg.ridge)
        if _cache is not None:
            _cache[key] = cur
    return blend_state(cur, prior, cfg)


# --------------------------------------------------------- matchup features --

def _dev(state: EffState, metric: str, x: str, y: str, hs: float) -> float:
    r = state.ratings.get(metric, EMPTY)
    return r.off.get(x, 0.0) - r.deff.get(y, 0.0) + r.hfa * hs


def side_features(state: EffState, x: str, y: str, hs: float) -> dict:
    """Deviation-from-league features of offense `x` against defense `y` (hs: +1 home offense,
    -1 away offense, 0 neutral). Missing metrics / unrated teams contribute a 0.0 deviation, so a
    metric absent for a whole season is neutral rather than NaN. PPA per play mixes the rush and
    pass ratings by x's season-to-date rush share; plays is the pace both teams imply."""
    rs = state.rush_share.get(x, state.lg_rush_share)
    ppa_mix = rs * _dev(state, "rush_ppa", x, y, hs) + (1 - rs) * _dev(state, "pass_ppa", x, y, hs)
    plays = state.lg_plays + 0.5 * ((state.pace.get(x, state.lg_plays) - state.lg_plays)
                                    + (state.pace.get(y, state.lg_plays) - state.lg_plays))
    return {"ppa_plays": ppa_mix * plays, "success": _dev(state, "success", x, y, hs),
            "explosiveness": _dev(state, "explosiveness", x, y, hs), "ppo": _dev(state, "ppo", x, y, hs),
            "havoc": _dev(state, "havoc", x, y, hs), "plays_dev": plays - state.lg_plays}


POINT_FEATURES = ("ppa_plays", "success", "explosiveness", "ppo", "havoc", "plays_dev")


# ------------------------------------------------------------ the eff frame --

def _relative_havoc(a: pd.DataFrame) -> pd.Series:
    """Ruling H1: havoc rate / that (season, season_type, week)'s play-weighted league mean havoc rate
    (over the team-games that have havoc data). A feed-level level shift in raw havoc (the real data
    runs ~0.23 in 2019 weeks 3-9 vs ~0.165 everywhere else) cancels in the ratio. NaN stays NaN."""
    plays = a["havoc_plays"].where(a["havoc_plays"] > 0)
    ok = a["off_havoc"].notna() & plays.notna()
    keys = [a["season"], a["season_type"], a["week"]]
    num = (a["off_havoc"] * plays).where(ok).groupby(keys).transform("sum")
    den = plays.where(ok).groupby(keys).transform("sum")
    mean = num / den
    return (a["off_havoc"] / mean).where(ok & (mean > 0))


def build_eff_games(adv: pd.DataFrame, havoc: pd.DataFrame | None, drives: pd.DataFrame | None,
                    meta: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per FBS team-game (regular season AND postseason) with the observations the ridge
    needs: y_<metric> / w_<metric> (y_havoc is relative to the week's league mean, ruling H1),
    plays, rush_plays, pass_plays, `home` (+1/-1/0) and `week`
    (the ESPN schedule week, NaN for postseason or games absent from the REG schedule -- those rows
    feed only the previous-season final ratings, never an in-season state)."""
    key = ["season", "game_id", "team"]
    a = adv.copy()
    a["season_type"] = a["season_type"].fillna("regular") if "season_type" in a else "regular"
    if havoc is not None and len(havoc):
        a = a.merge(havoc[key + ["off_havoc", "off_plays"]].rename(
            columns={"off_plays": "havoc_plays"}), on=key, how="left")
    else:
        a["off_havoc"], a["havoc_plays"] = np.nan, np.nan
    a["off_havoc"] = _relative_havoc(a)
    if drives is not None and len(drives):
        a = a.merge(drives[key + ["off_points_per_opp", "off_opps"]], on=key, how="left")
    else:
        a["off_points_per_opp"], a["off_opps"] = np.nan, np.nan
    a = a.merge(meta[["game_id", "home_team", "neutral_site"]], on="game_id", how="inner")
    a["home"] = np.where(a["neutral_site"], 0.0, np.where(a["team"] == a["home_team"], 1.0, -1.0))
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    wk = reg[["game_pk", "week"]].rename(columns={"game_pk": "game_id", "week": "sched_week"})
    a = a.drop(columns=["week"]).merge(wk, on="game_id", how="left")
    a["week"] = a["sched_week"].where(a["season_type"] == "regular")
    split_w = lambda col: a[col].fillna(0.5 * a["off_plays"])      # noqa: E731
    out = pd.DataFrame({
        "season": a["season"].astype("int64"), "game_id": a["game_id"].astype("int64"),
        "team": a["team"].astype(str), "opponent": a["opponent"].astype(str),
        "season_type": a["season_type"], "week": a["week"].astype("float64"), "home": a["home"],
        "plays": a["off_plays"], "rush_plays": a["off_rush_plays"], "pass_plays": a["off_pass_plays"],
        "y_ppa": a["off_ppa"], "w_ppa": a["off_plays"],
        "y_pass_ppa": a["off_pass_ppa"], "w_pass_ppa": split_w("off_pass_plays"),
        "y_rush_ppa": a["off_rush_ppa"], "w_rush_ppa": split_w("off_rush_plays"),
        "y_success": a["off_success"], "w_success": a["off_plays"],
        "y_explosiveness": a["off_explosiveness"], "w_explosiveness": a["off_plays"],
        "y_havoc": a["off_havoc"], "w_havoc": a["havoc_plays"].fillna(a["off_plays"]),
        "y_ppo": a["off_points_per_opp"], "w_ppo": a["off_opps"],
    })
    return out.reset_index(drop=True)


def load_eff_config(path) -> EffConfig:
    """EffConfig from assets/cfb/eff_config.json; a missing file -> the defaults."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return EffConfig()
    return EffConfig(**{k: float(v) for k, v in json.loads(p.read_text()).items()})
