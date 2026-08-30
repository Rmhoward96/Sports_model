"""Recalibrate prop / win-prob Platt maps on LIVE graded picks.

The 2024-backtest-fit calibration (assets/calibration.json) under-corrects live: over
~1,970 graded picks the model was 5-8pts overconfident (model_prob 60-70% won ~56%).
This refits each kept market's Platt layer from the real graded outcomes.

Method, per target:
  * Reconstruct the CALIBRATED prob_over per pick from its bet side
    (over -> model_prob, under -> 1 - model_prob) and the over-outcome (actual > line;
    ties dropped). win_prob uses P(home) vs home_score > away_score.
  * Fit a fresh Platt (a', b') on (prob_over_calibrated, y_over) -- this is a live
    CORRECTION applied on top of the already-calibrated prob.
  * Shrink the correction toward identity by s = n/(n+K) so thin/selected samples
    can't overcorrect (the picks are a +EV-selected sample -- mildly biased).
  * COMPOSE with the existing stored params (a, b) into a single [a, b] so the
    apply-code stays unchanged: sigmoid(a'*(a*logit(raw)+b)+b') = sigmoid(a'a*logit + a'b+b').

Prints Brier + 5-bin ECE before/after; only writes a market whose after-error improves
and that clears MIN_N. Run with --write to persist; default is dry-run.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sportsmodel.model import calibration  # noqa: E402
from sportsmodel import config  # noqa: E402

SUPA = "https://uydbzhzsscmrdwawnzlx.supabase.co"
ANON = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV5ZGJ6aHp"
        "zc2NtcmR3YXduemx4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODcxNjE4MDYsImV4cCI6MjEwMjczNzgwNn0."
        "lmQtx6Hz6oTtJOd3dmpxGTBKn5L8VPCA4jLPgF5E-CE")

PROP_TARGETS = ["total_bases", "pitcher_ks", "hits_allowed", "outs_recorded"]
MIN_N = 60
SHRINK_K = 75.0
A_MIN, A_MAX = 0.10, 1.50  # never let the composed slope invert (a<0) or over-steepen


def fetch_picks():
    cols = "sport,market,side,line,model_prob,actual,result,home_score,away_score"
    rows, off = [], 0
    while True:
        url = (f"{SUPA}/rest/v1/picks?status=eq.graded&select={cols}"
               f"&order=game_date.asc&limit=1000&offset={off}")
        req = urllib.request.Request(url, headers={"apikey": ANON, "Authorization": f"Bearer {ANON}"})
        page = json.load(urllib.request.urlopen(req))
        rows += page
        if len(page) < 1000:
            break
        off += 1000
    return rows


def samples_for(target, picks):
    """(prob_over_calibrated, y_over) pairs for a target."""
    xs, ys = [], []
    if target == "win_prob":
        for r in picks:
            if r["market"] != "moneyline" or r["model_prob"] is None:
                continue
            if r["home_score"] is None or r["away_score"] is None:
                continue
            p_home = r["model_prob"] if r["side"] == "home" else 1 - r["model_prob"]
            xs.append(p_home)
            ys.append(1.0 if r["home_score"] > r["away_score"] else 0.0)
        return xs, ys
    for r in picks:
        if r["market"] != target or r["model_prob"] is None or r["line"] is None or r["actual"] is None:
            continue
        if r["actual"] == r["line"]:
            continue  # push
        p_over = r["model_prob"] if r["side"] == "over" else 1 - r["model_prob"]
        xs.append(p_over)
        ys.append(1.0 if r["actual"] > r["line"] else 0.0)
    return xs, ys


def brier(probs, ys):
    return sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(ys)


def ece(probs, ys, bins=5):
    tot = len(ys); e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi or (b == bins - 1 and p == hi))]
        if not idx:
            continue
        mp = sum(probs[i] for i in idx) / len(idx)
        ma = sum(ys[i] for i in idx) / len(idx)
        e += len(idx) / tot * abs(mp - ma)
    return e


def main():
    write = "--write" in sys.argv
    picks = fetch_picks()
    existing = json.loads((config.PROJECT_ROOT / "assets" / "calibration.json").read_text())
    merged = dict(existing)

    print(f"Live graded picks: {len(picks)}   (MIN_N={MIN_N}, SHRINK_K={SHRINK_K})\n")
    print(f"{'target':<15}{'n':>5}{'shrink':>8}   {'Brier b->a':>16}   {'ECE b->a':>16}   {'params (a,b) new':>22}  status")
    for target in PROP_TARGETS + ["win_prob"]:
        xs, ys = samples_for(target, picks)
        n = len(ys)
        if n < MIN_N:
            print(f"{target:<15}{n:>5}{'-':>8}   {'-':>16}   {'-':>16}   {'(kept existing)':>22}  SKIP n<{MIN_N}")
            continue
        a_prime, b_prime = calibration.fit(xs, ys)
        s = n / (n + SHRINK_K)
        a_s, b_s = 1 + s * (a_prime - 1), s * b_prime          # shrink correction toward identity
        a0, b0 = existing.get(target, [1.0, 0.0])
        new_a = min(max(a_s * a0, A_MIN), A_MAX)               # compose, then clamp slope
        new = [new_a, a_s * b0 + b_s]
        # recover raw model prob (pre-existing-calibration) so we can score the FINAL
        # clamped composed params, not just the unclamped correction:
        raw = [calibration.apply(p, (1.0 / a0, -b0 / a0)) for p in xs]  # inverse of (a0,b0)
        after = [calibration.apply(r, new) for r in raw]
        br_b, br_a = brier(xs, ys), brier(after, ys)
        ec_b, ec_a = ece(xs, ys), ece(after, ys)
        improved = br_a <= br_b + 1e-9
        status = "WRITE" if improved else "skip(worse)"
        if improved:
            merged[target] = new
        print(f"{target:<15}{n:>5}{s:>8.2f}   {br_b:>7.4f}->{br_a:<7.4f}   {ec_b:>7.4f}->{ec_a:<7.4f}   "
              f"[{new[0]:+.3f}, {new[1]:+.3f}]  {status}")

    if write:
        (config.PROJECT_ROOT / "assets" / "calibration.json").write_text(json.dumps(merged, indent=2) + "\n")
        print("\nWROTE assets/calibration.json")
    else:
        print("\n(dry-run) re-run with --write to persist")


if __name__ == "__main__":
    main()
