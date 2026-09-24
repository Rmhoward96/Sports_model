"""Scoring rules for the props-ML ship gate (pure, no IO)."""
from __future__ import annotations

import zlib

import numpy as np


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
