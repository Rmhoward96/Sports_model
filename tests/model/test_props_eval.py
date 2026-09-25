import numpy as np
import pandas as pd
from sportsmodel.model.props_eval import (
    decile_ece,
    paired_frame,
    relative_skill,
    cluster_bootstrap,
    rung_decision,
    population_from_baseline,
    pit_pmf,
    pit_uniform,
    rps_pmf,
)


def test_rps_zero_for_point_mass_on_actual():
    assert rps_pmf([0, 0, 1, 0], 2) == 0.0


def test_rps_penalizes_distance():
    near, far = [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]
    assert rps_pmf(near, 2) < rps_pmf(far, 2)


def test_pit_bounds_and_determinism():
    u = pit_uniform(2024, 3, "p", "rec_yds")
    assert u == pit_uniform(2024, 3, "p", "rec_yds") and 0 <= u < 1
    assert pit_pmf([0.25, 0.25, 0.5], 1, 0.0) == 0.25
    assert pit_pmf([0.25, 0.25, 0.5], 1, 1.0) == 0.5


def test_decile_ece_uniform_is_small():
    rng = np.random.default_rng(0)
    assert decile_ece(rng.uniform(size=20000)) < 0.01
    assert decile_ece(np.full(1000, 0.05)) > 0.15


def _recs(rps, market="rec_yds", n=200, shift=0.0):
    return [
        {
            "season": 2024,
            "week": 1 + i % 17,
            "home": f"T{i % 8}",
            "player_id": f"p{i}",
            "market": market,
            "rps": rps + shift * (i % 3),
            "pit": (i % 10) / 10 + 0.05,
            "mean": 50.0,
        }
        for i in range(n)
    ]


def test_rung_passes_on_clear_improvement():
    base, cand = _recs(10.0), _recs(9.0)
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    d = rung_decision(paired_frame(base, cand, pop))
    assert d["pass"] and d["lo"] > 0


def test_rung_fails_when_one_market_worsens():
    base = _recs(10.0) + _recs(2.0, market="receptions")
    cand = _recs(8.0) + _recs(2.1, market="receptions")  # receptions 5% worse
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    d = rung_decision(paired_frame(base, cand, pop))
    assert not d["pass"] and any("receptions" in r for r in d["reasons"])


def test_population_uses_projected_gate():
    recs = [
        {"season": 2024, "week": 1, "player_id": "a", "market": "rec_yds", "mean": 60.0},
        {"season": 2024, "week": 1, "player_id": "b", "market": "rec_yds", "mean": 1.0},
    ]
    assert population_from_baseline(recs) == {(2024, 1, "a", "rec_yds")}


def test_cluster_bootstrap_resamples_with_replacement():
    """Verify clusters drawn multiple times contribute multiple row copies."""
    # Create data with 3 clusters of very different sizes
    # Cluster A: 50 rows, Cluster B: 1 row, Cluster C: 1 row
    rows = [
        {"season": 2024, "week": 1, "home": "T1", "player_id": f"p{i}", "market": "rec_yds",
         "rps_b": 5.0, "rps_c": 5.0, "pit_b": 0.5, "pit_c": 0.5, "cluster": "2024-1-TA"}
        for i in range(50)
    ]
    rows.append({"season": 2024, "week": 1, "home": "T2", "player_id": "pB", "market": "rec_yds",
                 "rps_b": 5.0, "rps_c": 5.0, "pit_b": 0.5, "pit_c": 0.5, "cluster": "2024-1-TB"})
    rows.append({"season": 2024, "week": 1, "home": "T3", "player_id": "pC", "market": "rec_yds",
                 "rps_b": 5.0, "rps_c": 5.0, "pit_b": 0.5, "pit_c": 0.5, "cluster": "2024-1-TC"})

    df = pd.DataFrame(rows)

    # Stat that counts rows
    def row_count(d):
        return len(d)

    point, lo, hi = cluster_bootstrap(df, row_count, n_boot=1000, seed=123)

    # Point should be 52 (all rows)
    assert point == 52, f"Point should be 52, got {point}"
    # With 3 clusters and replacement, some replicates will have >52 rows
    # (when cluster A is drawn twice, we get 52 + 50 = 102 rows)
    assert hi > 52, f"Expected hi > 52 with replacement, got hi={hi}"


def test_cluster_bootstrap_deterministic_and_order_invariant():
    """Same seed gives identical results; row order of input doesn't matter."""
    # Create base data with per-record noise in RPS (not constant)
    rng = np.random.default_rng(7)
    base_rows = [
        {
            "season": 2024,
            "week": 1 + i % 17,
            "home": f"T{i % 8}",
            "player_id": f"p{i}",
            "market": "rec_yds",
            "rps": 10.0 + rng.normal(0, 0.5),  # Noisy RPS per record
            "pit": (i % 10) / 10 + 0.05,
            "mean": 50.0,
        }
        for i in range(60)
    ]

    # Candidate records with different RPS (~9.5 with noise)
    cand_rows = [
        {**b, "rps": 9.5 + rng.normal(0, 0.5)}
        for b in base_rows
    ]

    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base_rows}

    # Build paired frame in original order
    df1 = paired_frame(base_rows, cand_rows, pop)

    # Shuffle input and rebuild (same records, different order)
    import random
    base_shuffled = base_rows.copy()
    cand_shuffled = cand_rows.copy()
    random.Random(99).shuffle(base_shuffled)
    random.Random(99).shuffle(cand_shuffled)

    df2 = paired_frame(base_shuffled, cand_shuffled, pop)

    # Paired frames should be identical (same rows, same order) despite shuffled input
    assert df1.equals(df2), "paired_frame should be deterministic regardless of input order"

    # Bootstrap both with same seed should give identical results
    point1, lo1, hi1 = cluster_bootstrap(df1, relative_skill, n_boot=50, seed=0)
    point2, lo2, hi2 = cluster_bootstrap(df2, relative_skill, n_boot=50, seed=0)

    assert point1 == point2, f"Points differ: {point1} vs {point2}"
    assert lo1 == lo2, f"LO bounds differ: {lo1} vs {lo2}"
    assert hi1 == hi2, f"HI bounds differ: {hi1} vs {hi2}"

    # Prove the data is noisy enough to create variance in bootstrap
    assert lo1 < hi1, f"Bootstrap bounds should differ (data is noisy), got lo={lo1}, hi={hi1}"


def test_rung_fails_on_zero_skill():
    """Candidate with same RPS + noise should fail lo > 0 constraint."""
    rng = np.random.default_rng(123)
    rows = [
        {"season": 2024, "week": 1 + i % 10, "home": f"T{i % 3}", "player_id": f"p{i}", "market": "rec_yds",
         "rps": 10.0, "pit": 0.5, "mean": 50.0}
        for i in range(150)
    ]

    base = rows
    # Candidate has same base RPS + noise, so skill ≈ 0 and bootstrap lo should be <= 0
    cand = [
        {**r, "rps": r["rps"] + rng.normal(0, 0.3), "pit": np.clip(r["pit"] + rng.normal(0, 0.05), 0, 1)}
        for r in rows
    ]

    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    df = paired_frame(base, cand, pop)
    d = rung_decision(df)

    # Should fail on lo <= 0
    assert not d["pass"], f"Expected fail on lo <= 0, got pass={d['pass']}"
    assert any("2.5%" in r or "bootstrap" in r for r in d["reasons"]), \
        f"Expected bootstrap bound reason in {d['reasons']}"


def test_rung_fails_on_ece_regression():
    """Candidate with concentrated PITs (poor calibration) should fail ECE constraint."""
    rows = [
        {"season": 2024, "week": 1 + i % 8, "home": f"T{i % 4}", "player_id": f"p{i}", "market": "rec_yds",
         "rps": 10.0, "pit": 0.3 + 0.4 * (i % 10) / 10, "mean": 50.0}
        for i in range(200)
    ]

    base = rows
    # Candidate has concentrated PITs (all ~0.05) - poor calibration
    cand = [
        {**r, "rps": r["rps"] * 0.99, "pit": 0.05}
        for r in rows
    ]

    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    df = paired_frame(base, cand, pop)
    d = rung_decision(df)

    # Should fail on ECE constraint
    assert not d["pass"], f"Expected fail on ECE, got pass={d['pass']}"
    ece_c = d["per_market"]["rec_yds"]["ece_c"]
    ece_b = d["per_market"]["rec_yds"]["ece_b"]
    assert ece_c > ece_b + 0.005, f"Expected ECE regression: {ece_c} > {ece_b + 0.005}"
    assert any("ece_c" in r for r in d["reasons"]), f"Expected ECE reason in {d['reasons']}"
    assert any("rec_yds" in r for r in d["reasons"]), f"Expected market name in reasons: {d['reasons']}"
