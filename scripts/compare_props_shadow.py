"""Props-ML shadow comparison report: the current sim (`sim-nfl-v1`) vs the
ML version (`nfl-sim-ml-v1`) on finished NFL games, from the rows shadow mode
wrote into `nfl_player_sim`.

After a week or more of SIM_ML_MODE=shadow, this report decides whether to
switch the site to the ML version (see db/migration_nfl_sim_serving.sql).

Population: (game_pk, player_id, market) keys present in BOTH versions with a
realized stat in `nfl_player_actuals` (paired). Per market and version:
  - n       paired keys
  - RPS     ranked probability score of the pmf vs the actual (lower is better)
  - ECE     decile ECE of the randomized PIT (keyed U(0,1), identical across
            versions so the PITs are paired)
  - Brier   of P(over line) vs the line outcome, on the paired keys that have a
            graded book line in `nfl_prop_grades`; pushes are excluded.
            P(over) = 1 - F(floor(line)) from the version's pmf.

`nfl_prop_grades` carries pass_yds / rush_yds / rec_yds / receptions /
rush_att (serving.props_ev.LINE_MARKETS; its `market` column already uses the
sim's names, rec_yds included). pass_tds and anytime_td have no graded lines,
so they are compared on RPS/ECE only.

Verdict per market: "ML better" if ML RPS < v1 RPS and ML Brier <= v1 Brier
(when lines exist); "ML worse" if ML RPS > v1 RPS and ML Brier > v1 Brier (when
lines exist); "no difference" if RPS (and Brier) are identical (a market the ML
version serves from the current sim); else "mixed". Overall: "inconclusive
(n < 300 per market)" when any compared market has n < 300; otherwise
"ML better" / "ML worse" when every differing market agrees, else "mixed".

Read-only. `--since` defaults to the first nfl-sim-ml-v1 row's created_at;
with no ML rows it prints a message and exits 0 without writing a report.

Usage:
    DATABASE_URL=... PYTHONPATH=src uv run python scripts/compare_props_shadow.py [--since 2026-09-20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.model.props_eval import decile_ece, pit_pmf, pit_uniform, rps_pmf

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "superpowers" / "reports"

V1_VERSION = "sim-nfl-v1"
ML_VERSION = "nfl-sim-ml-v1"
MIN_N = 300
MARKET_ORDER = ("pass_yds", "rush_yds", "rec_yds", "receptions", "rush_att",
                "pass_tds", "anytime_td")
_EPS = 1e-12


# ---------------------------------------------------------------- pure scoring

def p_over_pmf(pmf, line: float) -> float:
    """P(X > line) = 1 - F(floor(line)) for a pmf over 0..K. For an integer
    line this is the strict P(X > line) (the push mass is excluded)."""
    p = np.asarray(pmf, dtype=float)
    cut = math.floor(float(line))
    if cut < 0:
        return float(p.sum())
    return float(p[cut + 1:].sum())


def _pmf(dist) -> list | None:
    if isinstance(dist, str):
        dist = json.loads(dist)
    if not isinstance(dist, dict):
        return None
    pmf = dist.get("pmf")
    return list(pmf) if pmf else None


def _key(r: dict) -> tuple:
    return (int(r["game_pk"]), str(r["player_id"]), str(r["market"]))


def _cmp(a: float, b: float) -> int:
    """-1 if a < b, 1 if a > b, 0 if equal within _EPS."""
    if abs(a - b) <= _EPS:
        return 0
    return -1 if a < b else 1


def market_verdict(rps_v1: float, rps_ml: float,
                   brier_v1: float | None, brier_ml: float | None) -> str:
    """Per-market verdict (ML relative to v1); Brier is ignored when None."""
    r = _cmp(rps_ml, rps_v1)
    b = None if brier_v1 is None or brier_ml is None else _cmp(brier_ml, brier_v1)
    if r == 0 and b in (None, 0):
        return "no difference"
    if r < 0 and b in (None, -1, 0):
        return "ML better"
    if r > 0 and b in (None, 1):
        return "ML worse"
    return "mixed"


def overall_verdict(markets: dict) -> str:
    """Overall verdict from per-market {n, verdict}."""
    if not markets:
        return "inconclusive (no paired rows)"
    if any(m["n"] < MIN_N for m in markets.values()):
        return f"inconclusive (n < {MIN_N} per market)"
    differing = {m["verdict"] for m in markets.values()} - {"no difference"}
    if not differing:
        return "no difference"
    if differing == {"ML better"}:
        return "ML better"
    if differing == {"ML worse"}:
        return "ML worse"
    return "mixed"


def score_shadow(v1_rows: list[dict], ml_rows: list[dict],
                 actuals: list[dict], lines: list[dict]) -> dict:
    """PURE. Score both versions on the paired (game_pk, player_id, market)
    keys that have an actual. Rows: sims {game_pk, player_id, market, dist
    (dict or JSON str)}; actuals {game_pk, player_id, market, actual, season,
    week}; lines {game_pk, player_id, market, line}.

    Returns {"markets": {market: {n, rps_v1, rps_ml, ece_v1, ece_ml, n_line,
    brier_v1, brier_ml, verdict}}, "skipped": int, "verdict": str}; `skipped`
    counts paired keys with an actual dropped for a missing/empty pmf."""
    v1 = {_key(r): r for r in v1_rows}
    ml = {_key(r): r for r in ml_rows}
    act = {_key(r): r for r in actuals if r.get("actual") is not None}
    line_by = {_key(r): float(r["line"]) for r in lines if r.get("line") is not None}

    per: dict[str, dict[str, list]] = {}
    skipped = 0
    for key in sorted(set(v1) & set(ml) & set(act)):
        pmf_v1, pmf_ml = _pmf(v1[key].get("dist")), _pmf(ml[key].get("dist"))
        if pmf_v1 is None or pmf_ml is None:
            skipped += 1
            continue
        _, player_id, market = key
        a = act[key]
        y = float(a["actual"])
        u = pit_uniform(a.get("season"), a.get("week"), player_id, market)
        acc = per.setdefault(market, {k: [] for k in
                                      ("rps_v1", "rps_ml", "pit_v1", "pit_ml", "br_v1", "br_ml")})
        acc["rps_v1"].append(rps_pmf(pmf_v1, y))
        acc["rps_ml"].append(rps_pmf(pmf_ml, y))
        acc["pit_v1"].append(pit_pmf(pmf_v1, y, u))
        acc["pit_ml"].append(pit_pmf(pmf_ml, y, u))
        line = line_by.get(key)
        if line is not None and y != line:  # a push is excluded from Brier
            hit = 1.0 if y > line else 0.0
            acc["br_v1"].append((p_over_pmf(pmf_v1, line) - hit) ** 2)
            acc["br_ml"].append((p_over_pmf(pmf_ml, line) - hit) ** 2)

    markets: dict[str, dict] = {}
    for market, acc in per.items():
        rps_v1, rps_ml = float(np.mean(acc["rps_v1"])), float(np.mean(acc["rps_ml"]))
        n_line = len(acc["br_v1"])
        brier_v1 = float(np.mean(acc["br_v1"])) if n_line else None
        brier_ml = float(np.mean(acc["br_ml"])) if n_line else None
        markets[market] = {
            "n": len(acc["rps_v1"]),
            "rps_v1": rps_v1, "rps_ml": rps_ml,
            "ece_v1": decile_ece(acc["pit_v1"]), "ece_ml": decile_ece(acc["pit_ml"]),
            "n_line": n_line, "brier_v1": brier_v1, "brier_ml": brier_ml,
            "verdict": market_verdict(rps_v1, rps_ml, brier_v1, brier_ml),
        }
    return {"markets": markets, "skipped": skipped, "verdict": overall_verdict(markets)}


# ---------------------------------------------------------------- rendering

def _ordered(markets: dict) -> list[str]:
    known = [m for m in MARKET_ORDER if m in markets]
    return known + sorted(set(markets) - set(MARKET_ORDER))


def _f(x: float | None) -> str:
    return "–" if x is None else f"{x:.4f}"


def format_table(result: dict) -> str:
    rows = ["| market | n | RPS v1 | RPS ML | ECE v1 | ECE ML | n line | Brier v1 | Brier ML | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for market in _ordered(result["markets"]):
        m = result["markets"][market]
        rows.append(
            f"| {market} | {m['n']} | {_f(m['rps_v1'])} | {_f(m['rps_ml'])} | "
            f"{_f(m['ece_v1'])} | {_f(m['ece_ml'])} | {m['n_line']} | "
            f"{_f(m['brier_v1'])} | {_f(m['brier_ml'])} | {m['verdict']} |")
    return "\n".join(rows)


def render_report(result: dict, since: str, run_date: str) -> str:
    n_by = ", ".join(f"{m} {result['markets'][m]['n']}" for m in _ordered(result["markets"]))
    lines = [
        f"# Props-ML shadow comparison — {run_date}",
        "",
        f"**Verdict: {result['verdict']}.**",
        "",
        f"Per-market n: {n_by or 'none'}.",
        "",
        "## Run",
        "",
        f"- Versions: {V1_VERSION} (current sim) vs {ML_VERSION} (ML), from nfl_player_sim.",
        f"- Since: {since} (games with commence_time on or after this, already kicked off).",
        "- Population: (game_pk, player_id, market) keys present in both versions with a "
        "realized stat in nfl_player_actuals (paired).",
        f"- Paired keys skipped for a missing/empty pmf: {result['skipped']}.",
        "- RPS: ranked probability score vs the actual (lower is better). ECE: decile ECE of "
        "the randomized PIT (paired uniforms).",
        "- Brier: P(over line) = 1 - F(floor(line)) vs the graded line outcome from "
        "nfl_prop_grades; pushes excluded. pass_tds and anytime_td have no graded lines.",
        f"- Market verdict: ML better = lower RPS and Brier no worse; ML worse = higher RPS and "
        f"higher Brier; otherwise mixed. Overall is inconclusive while any market has n < {MIN_N}.",
        "",
        "## Per market",
        "",
        format_table(result),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- DB (read-only)

def load_shadow_data(since: str | None) -> dict | None:
    """The only DB access. Returns None when no nfl-sim-ml-v1 rows exist, else
    {"since", "v1", "ml", "actuals", "lines"} for games with commence_time in
    [since, now()]. `since` defaults to the first ML row's created_at."""
    from sportsmodel.db import get_postgres

    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute("SELECT min(created_at) FROM nfl_player_sim WHERE model_version = %s",
                    (ML_VERSION,))
        first = cur.fetchone()[0]
        if first is None:
            return None
        since_val = since if since is not None else first
        cur.execute(
            """
            SELECT game_pk, player_id, model_version, market, dist
            FROM nfl_player_sim
            WHERE model_version IN (%s, %s)
              AND commence_time >= %s::timestamptz AND commence_time <= now()
            """,
            (V1_VERSION, ML_VERSION, since_val),
        )
        sims = cur.fetchall()
        cur.execute(
            """
            SELECT a.game_pk, a.player_id, a.market, a.actual, a.season, a.week
            FROM nfl_player_actuals a
            WHERE a.game_pk IN (
                SELECT DISTINCT game_pk FROM nfl_player_sim
                WHERE model_version = %s
                  AND commence_time >= %s::timestamptz AND commence_time <= now())
            """,
            (ML_VERSION, since_val),
        )
        actuals = cur.fetchall()
        cur.execute(
            """
            SELECT game_pk, player_id, market, line
            FROM nfl_prop_grades
            WHERE commence_time >= %s::timestamptz
            """,
            (since_val,),
        )
        lines = cur.fetchall()

    sim_dicts = [dict(zip(("game_pk", "player_id", "model_version", "market", "dist"), r))
                 for r in sims]
    return {
        "since": since_val.isoformat() if hasattr(since_val, "isoformat") else str(since_val),
        "v1": [r for r in sim_dicts if r["model_version"] == V1_VERSION],
        "ml": [r for r in sim_dicts if r["model_version"] == ML_VERSION],
        "actuals": [dict(zip(("game_pk", "player_id", "market", "actual", "season", "week"), r))
                    for r in actuals],
        "lines": [dict(zip(("game_pk", "player_id", "market", "line"), r)) for r in lines],
    }


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=None,
                    help="start date/timestamp (default: first nfl-sim-ml-v1 row's created_at)")
    ap.add_argument("--out-dir", default=str(REPORT_DIR))
    ap.add_argument("--run-date", default=None, help="report date (default: today)")
    args = ap.parse_args(argv)

    data = load_shadow_data(args.since)
    if data is None:
        print(f"no {ML_VERSION} rows in nfl_player_sim yet -- nothing to compare "
              "(run generate_sim_nfl.py with SIM_ML_MODE=shadow first); no report written.")
        return 0

    result = score_shadow(data["v1"], data["ml"], data["actuals"], data["lines"])
    run_date = args.run_date or date.today().isoformat()
    print(f"Props-ML shadow comparison since {data['since']}: {result['verdict']}")
    print(format_table(result))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_date}-props-ml-shadow.md"
    out.write_text(render_report(result, since=data["since"], run_date=run_date))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
