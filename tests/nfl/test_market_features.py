import math

from sportsmodel.nfl import market_features as mf


def _snap(game_pk, market, side, book, line, price, captured_at):
    return {"game_pk": game_pk, "market": market, "side": side, "book": book,
            "line": line, "price": price, "captured_at": captured_at}


def test_line_movement_open_close_and_leakage():
    snaps = [
        _snap(1, "spread", "home", "dk", -3.0, -110, "2026-09-20T10:00:00Z"),
        _snap(1, "spread", "home", "fd", -3.5, -110, "2026-09-20T14:00:00Z"),  # close (<= decision)
        _snap(1, "spread", "home", "dk", -6.0, -110, "2026-09-20T18:00:00Z"),  # AFTER decision -> excluded
    ]
    lm = mf.line_movement(snaps, 1, "spread", decision_ts="2026-09-20T15:00:00Z")
    assert lm["open_line"] == -3.0
    assert lm["close_line"] == -3.5          # the 18:00 snapshot must not leak in
    assert lm["dline"] == -0.5
    assert lm["abs_dline"] == 0.5


def test_line_movement_insufficient_is_nan():
    snaps = [_snap(1, "spread", "home", "dk", -3.0, -110, "2026-09-20T10:00:00Z")]
    lm = mf.line_movement(snaps, 1, "spread", decision_ts="2026-09-20T15:00:00Z")
    assert all(math.isnan(v) for v in lm.values())


def test_sharp_vs_soft_pinnacle_minus_consensus():
    # Pinnacle rates home higher than the soft books -> positive divergence.
    snaps = [
        _snap(1, "spread", "home", "pinnacle", -3.0, -140, "2026-09-20T14:00:00Z"),
        _snap(1, "spread", "away", "pinnacle", 3.0, +120, "2026-09-20T14:00:00Z"),
        _snap(1, "spread", "home", "dk", -3.0, -110, "2026-09-20T14:00:00Z"),
        _snap(1, "spread", "away", "dk", 3.0, -110, "2026-09-20T14:00:00Z"),
    ]
    d = mf.sharp_vs_soft(snaps, 1, "spread", "home", decision_ts="2026-09-20T15:00:00Z")
    assert d is not None and d > 0        # pinnacle home novig > soft home novig
    assert mf.sharp_vs_soft(snaps, 1, "spread", "home", "2026-09-20T13:00:00Z") is None  # nothing before


def test_sharp_vs_soft_none_when_no_pinnacle():
    snaps = [
        _snap(1, "spread", "home", "dk", -3.0, -110, "2026-09-20T14:00:00Z"),
        _snap(1, "spread", "away", "dk", 3.0, -110, "2026-09-20T14:00:00Z"),
    ]
    assert mf.sharp_vs_soft(snaps, 1, "spread", "home", "2026-09-20T15:00:00Z") is None


def test_split_features_divergence_and_nan_safety():
    f = mf.split_features({"cash_pct": 62.0, "ticket_pct": 48.0})
    assert f["cash_minus_ticket"] == 14.0
    n = mf.split_features(None)
    assert all(math.isnan(v) for v in n.values())
    partial = mf.split_features({"cash_pct": 60.0, "ticket_pct": None})
    assert partial["cash_pct"] == 60.0 and math.isnan(partial["cash_minus_ticket"])


def test_reverse_line_movement_fade_and_follow_and_none():
    # public heavy on home (70% tickets) but line moved OFF home (dline +1) -> RLM fade
    assert mf.reverse_line_movement(1.0, 70.0) == 1.0
    # public on home and line moved TOWARD home (dline -1) -> follows public
    assert mf.reverse_line_movement(-1.0, 70.0) == -1.0
    # public on away (30%) and line moved toward home -> fades the away public
    assert mf.reverse_line_movement(-2.0, 30.0) == 2.0
    assert mf.reverse_line_movement(0.0, 70.0) == 0.0     # flat
    assert mf.reverse_line_movement(1.0, None) is None    # no splits
    assert mf.reverse_line_movement(float("nan"), 70.0) is None
