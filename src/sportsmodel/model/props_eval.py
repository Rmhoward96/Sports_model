"""Scoring rules for the props-ML ship gate (pure, no IO)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from ..serving.props_ev import is_propable_projected


def rps_pmf(pmf, actual: float) -> float:
    """Ranked probability score (discrete CRPS) of a pmf over 0..K for an
    integer-valued actual; the actual is clipped into [0, K]. Lower is better."""
    p = np.asarray(pmf, dtype=float)
    k = len(p) - 1
    a = int(min(max(round(actual), 0), k))
    cdf = np.cumsum(p)
    step = (np.arange(k + 1) >= a).astype(float)
    return float(np.sum((cdf - step) ** 2))


def pit_uniform(season: int, week: int, player_id: str, market: str) -> float:
    """Deterministic U(0,1) per record (crc32 of the key), identical across
    candidates so randomized PITs are paired."""
    h = zlib.crc32(f"{season}|{week}|{player_id}|{market}".encode())
    return (h % 1_000_003) / 1_000_003


def pit_pmf(pmf, actual: float, u: float) -> float:
    """Randomized PIT for a discrete forecast: F(a-1) + u * P(a)."""
    p = np.asarray(pmf, dtype=float)
    k = len(p) - 1
    a = int(min(max(round(actual), 0), k))
    below = float(p[:a].sum())
    return below + u * float(p[a])


def decile_ece(pits: np.ndarray | list) -> float:
    """Expected Calibration Error for PIT values using decile binning.

    Splits pits into 10 equal-probability bins and computes the mean absolute
    deviation of empirical from expected frequencies. For perfect calibration,
    each decile should contain ~10% of observations.

    NaN values are dropped; pits are clipped to [0, 1].

    Returns: mean of |empirical_share - 0.1| across the 10 bins.
    """
    pits = np.asarray(pits, dtype=float).flatten()
    # Drop NaN values
    pits = pits[np.isfinite(pits)]
    if len(pits) == 0:
        return 0.0

    # Clip to [0, 1]
    pits = np.clip(pits, 0, 1)

    n = len(pits)
    errors = []

    for i in range(10):
        lower = i / 10.0
        upper = (i + 1) / 10.0
        # Count pits in [lower, upper); last bin includes upper
        if i < 9:
            count = np.sum((pits >= lower) & (pits < upper))
        else:
            count = np.sum((pits >= lower) & (pits <= upper))
        share = count / n
        errors.append(abs(share - 0.1))

    return float(np.mean(errors))


def paired_frame(base: list[dict], cand: list[dict], population: set[tuple]) -> pd.DataFrame:
    """Inner-join base and candidate records on (season, week, player_id, market).

    Filters to only rows in the population set of (season, week, player_id, market)
    tuples. Adds a 'cluster' column as f"{season}-{week}-{home}".

    Returns DataFrame with columns: rps_b, rps_c, pit_b, pit_c, cluster.

    Row order is deterministic (sorted by key) regardless of input order.
    """
    # Create lookup dicts keyed by (season, week, player_id, market)
    base_lookup = {
        (r["season"], r["week"], r["player_id"], r["market"]): r for r in base
    }
    cand_lookup = {
        (r["season"], r["week"], r["player_id"], r["market"]): r for r in cand
    }

    # Inner join: only keys present in both base and cand
    common_keys = set(base_lookup.keys()) & set(cand_lookup.keys())

    # Filter to population
    common_keys = common_keys & population

    # Sort for deterministic row order across processes
    common_keys = sorted(common_keys)

    rows = []
    for key in common_keys:
        b = base_lookup[key]
        c = cand_lookup[key]
        season, week, player_id, market = key
        cluster = f"{season}-{week}-{b['home']}"
        rows.append({
            "rps_b": b["rps"],
            "rps_c": c["rps"],
            "pit_b": b["pit"],
            "pit_c": c["pit"],
            "cluster": cluster,
            "market": market,
        })

    return pd.DataFrame(rows)


def relative_skill(df: pd.DataFrame) -> float:
    """Compute pooled relative skill score.

    For each market, computes 1 - mean(rps_c) / mean(rps_b). Returns the mean
    across all markets. Scale-free metric: positive means candidate is better.
    """
    if df.empty:
        return 0.0

    market_scores = []
    for market, group in df.groupby("market"):
        mean_rps_b = group["rps_b"].mean()
        mean_rps_c = group["rps_c"].mean()

        if mean_rps_b > 0:
            skill = 1.0 - mean_rps_c / mean_rps_b
            market_scores.append(skill)

    if not market_scores:
        return 0.0

    return float(np.mean(market_scores))


def cluster_bootstrap(
    df: pd.DataFrame,
    stat,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap clusters with replacement and compute percentiles of a statistic.

    Resamples unique clusters (with replacement) using numpy default_rng,
    then recomputes the statistic (callable) on each bootstrap sample.

    Clusters drawn multiple times contribute multiple copies of their rows to
    each bootstrap replicate (proper resampling with replacement).

    Returns: (point_estimate, percentile_2.5, percentile_97.5)
    """
    point = stat(df)

    if df.empty or "cluster" not in df.columns:
        return (point, point, point)

    # Build deterministic sorted array of clusters and per-cluster row groups
    clusters = sorted(df["cluster"].unique())
    groups = {cluster: df[df["cluster"] == cluster] for cluster in clusters}

    rng = np.random.default_rng(seed)

    boot_stats = []
    for _ in range(n_boot):
        # Sample cluster IDs with replacement
        boot_cluster_ids = rng.choice(clusters, size=len(clusters), replace=True)
        # Concatenate row groups for sampled clusters (with repetition)
        boot_dfs = [groups[c] for c in boot_cluster_ids]
        boot_df = pd.concat(boot_dfs, ignore_index=True)
        if not boot_df.empty:
            boot_stats.append(stat(boot_df))

    if not boot_stats:
        return (point, point, point)

    boot_stats = np.array(boot_stats)
    lo = float(np.percentile(boot_stats, 2.5))
    hi = float(np.percentile(boot_stats, 97.5))

    return (point, lo, hi)


def rung_decision(df: pd.DataFrame) -> dict:
    """Gate decision: does candidate pass all constraints?

    Pass iff:
    - Bootstrap 2.5% bound > 0
    - No market with rps_c > 1.01 * rps_b
    - No market with ece_c > ece_b + 0.005

    Returns dict with keys:
    - skill: pooled relative skill score
    - lo, hi: bootstrap bounds
    - per_market: {market: {n, rps_b, rps_c, ece_b, ece_c}}
    - pass: bool
    - reasons: list of failure reasons
    """
    skill, lo, hi = cluster_bootstrap(df, relative_skill)

    per_market = {}
    reasons = []

    if df.empty:
        return {
            "skill": skill,
            "lo": lo,
            "hi": hi,
            "per_market": per_market,
            "pass": False,
            "reasons": ["Empty DataFrame"],
        }

    # Check per-market constraints
    for market, group in df.groupby("market"):
        rps_b = group["rps_b"].values
        rps_c = group["rps_c"].values
        pit_b = group["pit_b"].values
        pit_c = group["pit_c"].values

        mean_rps_b = float(rps_b.mean())
        mean_rps_c = float(rps_c.mean())
        ece_b = decile_ece(pit_b)
        ece_c = decile_ece(pit_c)

        per_market[market] = {
            "n": len(group),
            "rps_b": mean_rps_b,
            "rps_c": mean_rps_c,
            "ece_b": ece_b,
            "ece_c": ece_c,
        }

        # Check rps_c > 1.01 * rps_b (worse is failure)
        if mean_rps_b > 0 and mean_rps_c > 1.01 * mean_rps_b:
            reasons.append(f"market {market}: rps_c ({mean_rps_c:.4f}) > 1.01 * rps_b ({1.01 * mean_rps_b:.4f})")

        # Check ece_c > ece_b + 0.005
        if ece_c > ece_b + 0.005:
            reasons.append(f"market {market}: ece_c ({ece_c:.4f}) > ece_b ({ece_b:.4f}) + 0.005")

    # Check bootstrap bound
    if lo <= 0:
        reasons.append(f"bootstrap 2.5% bound ({lo:.4f}) <= 0")

    passes = lo > 0 and len(reasons) == 0

    return {
        "skill": skill,
        "lo": lo,
        "hi": hi,
        "per_market": per_market,
        "pass": passes,
        "reasons": reasons,
    }


def population_from_baseline(base_records: list[dict]) -> set[tuple]:
    """Filter baseline records by projected-usage gate.

    Returns set of (season, week, player_id, market) tuples for records whose
    baseline mean passes is_propable_projected(market, mean).
    """
    population = set()
    for r in base_records:
        market = r.get("market")
        mean = r.get("mean")
        if market is not None and mean is not None:
            if is_propable_projected(market, mean):
                key = (r["season"], r["week"], r["player_id"], market)
                population.add(key)
    return population
