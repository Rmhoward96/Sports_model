"""Historical lean backtest: apply the site's shrink + lean rule (shrinkLean, site/js/shrink.js) to past games.

lean rule: shrunk = close + k (model - close); quoted = half point (r05) unless equal to close; lean when |quoted - close| >= 0.25.
All numbers are in expected-HOME-MARGIN space (positive = home favoured) for spreads, so 'home' lean = home covers.
CFB / NFL graded at -110 vs the closing number. MLB graded at the consensus closing price of the chosen side.
k: walk-forward (fit on prior seasons only, game_fit.fit_k: 300-game minimum, else 0) and a fixed-k sweep.
"""
from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
S = ROOT / "data" / "backtest_leans"   # caches (git-ignored)
S.mkdir(parents=True, exist_ok=True)

from sportsmodel.shrink import game_fit  # noqa: E402

SWEEP = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
WIN_110 = 100 / 110


def r05(x):  # JS Math.round(x*2)/2
    return math.floor(x * 2 + 0.5) / 2


def lean(model, close, k):
    """(side, edge) or None; side 'hi' = home cover / over."""
    sh = close + k * (model - close)
    q = close if abs(sh - close) < 1e-9 else r05(sh)
    e = q - close
    if abs(e) < 0.25:
        return None
    return ("hi" if e > 0 else "lo"), e


def grade(side, actual, close):
    d = actual - close
    if d == 0:
        return 0
    return 1 if (d > 0) == (side == "hi") else -1


def payout(res, price=None):
    if res == 0:
        return 0.0
    if res < 0:
        return -1.0
    if price is None:
        return WIN_110
    return price / 100 if price > 0 else 100 / -price


# ---------------------------------------------------------------- data
def cfb_legs():
    cache = S / "bt_cfb_rows.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import fit_game_shrink as f
    raw = f.load_raw(2025)
    gl = game_fit.gameline_config(json.loads((ROOT / "assets/cfb/gameline.json").read_text()))
    rows = game_fit.served_rows(raw, gl)
    out = {m: [l for l in game_fit.point_legs(rows, m) if l["season"] >= 2016] for m in ("spread", "total")}
    cache.write_text(json.dumps(out))
    return out


def nfl_legs():
    import pandas as pd
    d = pd.read_parquet(ROOT / "data/props_ml/game_records__s2021-2025__every4.parquet")
    out = {"spread": [], "total": []}
    for r in d.itertuples():
        if all(map(math.isfinite, (r.pred_margin, r.spread_line, r.actual_margin))):
            out["spread"].append({"season": int(r.season), "model": r.pred_margin, "close": r.spread_line,
                                  "actual": r.actual_margin, "x": r.pred_margin - r.spread_line, "y": r.actual_margin - r.spread_line})
        if all(map(math.isfinite, (r.pred_total, r.total_line, r.actual_total))):
            out["total"].append({"season": int(r.season), "model": r.pred_total, "close": r.total_line,
                                 "actual": r.actual_total, "x": r.pred_total - r.total_line, "y": r.actual_total - r.total_line})
    return out


def mlb_scores(pks):
    cache = S / "bt_mlb_scores.json"
    have = json.loads(cache.read_text()) if cache.exists() else {}
    need = [p for p in pks if str(p) not in have]
    for i in range(0, len(need), 50):
        chunk = ",".join(str(p) for p in need[i:i + 50])
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePks={chunk}"
        data = json.load(urllib.request.urlopen(url, timeout=30))
        for dt in data.get("dates", []):
            for g in dt["games"]:
                if g["status"]["abstractGameState"] != "Final":
                    continue
                h, a = g["teams"]["home"].get("score"), g["teams"]["away"].get("score")
                if h is not None and a is not None:
                    have[str(g["gamePk"])] = [h, a]
    cache.write_text(json.dumps(have))
    return have


MLB_SQL = """
with pre as (
  select o.game_pk, o.market, o.side, o.book, o.line, o.price, o.captured_at,
         max(o.captured_at) over (partition by o.game_pk, o.market) last_cap
  from odds_snapshot o
  where o.market in ('spread','total') and o.captured_at < o.commence_time
    and o.game_pk in (select game_pk from game_predictions where sport='mlb')
)
select game_pk, market, side, book, line::float, price::float from pre where captured_at = last_cap
"""


def mlb_legs():
    import psycopg
    from dotenv import dotenv_values
    url = dotenv_values(ROOT / ".env")["DATABASE_URL"]
    with psycopg.connect(url) as c:
        preds = c.execute("""select distinct on (game_pk) game_pk, pred_margin::float, pred_total::float, model_version
                             from game_predictions where sport='mlb' order by game_pk, generated_at desc""").fetchall()
        odds = c.execute(MLB_SQL).fetchall()
    by = defaultdict(list)
    for pk, mk, side, book, line, price in odds:
        by[(pk, mk)].append((side, book, line, price))
    scores = mlb_scores([p[0] for p in preds])
    out = {"spread": [], "total": []}
    for pk, pm, pt, mv in preds:
        sc = scores.get(str(pk))
        if sc is None:
            continue
        am, at = sc[0] - sc[1], sc[0] + sc[1]
        # total: consensus (modal) line; prices of over/under at that line
        tq = by.get((pk, "total"))
        if tq and pt is not None:
            lines = [l for s, b, l, p in tq if s == "over"]
            if lines:
                line = max(set(lines), key=lines.count)
                med = lambda s: sorted(p for ss, b, l, p in tq if ss == s and l == line)  # noqa: E731
                ov, un = med("over"), med("under")
                if ov and un:
                    out["total"].append({"season": 2026, "model": pt, "close": line, "actual": at, "mv": mv,
                                         "p_hi": ov[len(ov) // 2], "p_lo": un[len(un) // 2]})
        # run line: home book line h (e.g. -1.5) -> close in home-margin space = -h
        sq = by.get((pk, "spread"))
        if sq and pm is not None:
            hl = [l for s, b, l, p in sq if s == "home"]
            if hl:
                h = max(set(hl), key=hl.count)
                hp = sorted(p for s, b, l, p in sq if s == "home" and l == h)
                ap = sorted(p for s, b, l, p in sq if s == "away" and l == -h)
                if hp and ap:
                    out["spread"].append({"season": 2026, "model": pm, "close": -h, "actual": am, "mv": mv,
                                          "p_hi": hp[len(hp) // 2], "p_lo": ap[len(ap) // 2]})
    return out


# ---------------------------------------------------------------- evaluation
def bets(legs, kfn, priced=False):
    out = []
    for l in legs:
        k = kfn(l)
        if k is None:
            continue
        ln = lean(l["model"], l["close"], k)
        if ln is None:
            continue
        side, e = ln
        res = grade(side, l["actual"], l["close"])
        price = (l["p_hi"] if side == "hi" else l["p_lo"]) if priced else None
        out.append({"season": l["season"], "res": res, "u": payout(res, price), "edge": abs(e), "side": side})
    return out


def summary(bs, seed=7):
    n = len(bs)
    if n == 0:
        return "    0 bets"
    w = sum(b["res"] == 1 for b in bs)
    lo = sum(b["res"] == -1 for b in bs)
    p = n - w - lo
    u = sum(b["u"] for b in bs)
    rnd = random.Random(seed)
    boots = sorted(sum(rnd.choice(bs)["u"] for _ in range(n)) / n for _ in range(2000))
    ci = (boots[50] * 100, boots[1949] * 100)
    wr = w / (w + lo) * 100 if w + lo else 0
    return f"{n:5d} bets  {w}-{lo}-{p}  win {wr:5.1f}%  {u:+7.1f}u  ROI {u / n * 100:+6.1f}%  95% CI [{ci[0]:+.1f}%, {ci[1]:+.1f}%]"


def wf_k(legs, market):
    ks = {}
    for s in sorted({l["season"] for l in legs}):
        ks[s] = game_fit.fit_k([l for l in legs if l["season"] < s], market)
    return ks


def report(name, legs_by_mkt, priced=False, walk_forward=True, first_season=None):
    print(f"\n==================== {name} ====================")
    for mkt, legs in legs_by_mkt.items():
        if first_season:
            hist = legs
            legs = [l for l in legs if l["season"] >= first_season]
        else:
            hist = legs
        print(f"\n--- {name} {mkt}: {len(legs)} graded games ---")
        if walk_forward:
            ks = wf_k(hist, mkt)
            bs = bets(legs, lambda l: ks[l["season"]], priced)
            print("walk-forward k (fit on prior seasons only; site's 300-game minimum):")
            print("  ALL  " + summary(bs))
            for s in sorted({l["season"] for l in legs}):
                sb = [b for b in bs if b["season"] == s]
                print(f"  {s} k={ks[s]:.3f} " + summary(sb))
            full_k = game_fit.fit_k(hist, mkt)
            print(f"  (in-sample k on all seasons = {full_k:.3f}, raw slope = {game_fit._slope(hist, mkt)})")
        print("fixed-k sweep (same k every season -- hindsight, not a live result):")
        for k in SWEEP:
            print(f"  k={k:<4} " + summary(bets(legs, lambda l, k=k: k, priced)))
        # bigger edges only, at k=1 (raw model) and k=0.3
        for k in (0.3, 1.0):
            bs = bets(legs, lambda l, k=k: k, priced)
            for thr in (1, 2, 3):
                print(f"  k={k} edge>={thr}: " + summary([b for b in bs if b["edge"] >= thr]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["cfb", "nfl", "mlb"]
    if "cfb" in which:
        report("CFB (cfb-ratings walk-forward, served numbers)", cfb_legs(), first_season=2018)
    if "nfl" in which:
        report("NFL (ML sim walk-forward records)", nfl_legs())
    if "mlb" in which:
        m = mlb_legs()
        report("MLB 2026 (mlb-hybrid-v1 / v2-defense, consensus close, actual prices)", m, priced=True, walk_forward=False)
