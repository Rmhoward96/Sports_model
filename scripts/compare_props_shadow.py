"""Props-ML shadow comparison report: the current sim (`sim-nfl-v1`) vs an
ML version (`--ml-version` / env PROPS_SHADOW_ML_VERSION; default the SERVED
ML version from nfl_sim_serving, else `nfl-sim-ml-v1`) on finished NFL games,
from the rows shadow / live mode wrote into `nfl_player_sim`.

After a week or more of SIM_ML_MODE=shadow, this report decides whether to
switch the site to the ML version (see db/migration_nfl_sim_serving.sql).

Population (verdict): the GATED population -- (game_pk, player_id, market)
keys inside `props_eval.gate_population` of the v1 (current sim) dist means,
the same projected-usage gate the ship gate scored and serving applies (only
gated player-markets are ever served from ML) -- present in BOTH versions
with a realized stat in `nfl_player_actuals` (paired). The same scoring over
ALL paired keys (ungated) is reported in a separate table and never feeds
the verdict. Pairs whose ML and v1 pmfs are
identical (same length and np.array_equal -- per-game ML fallbacks and markets
the ML pipeline sources from the current sim) are EXCLUDED from n and every
metric and reported separately as n_identical (per market and total); a market
with only identical pairs shows "all identical (n_identical=K)". Per market and
version, over the remaining (differing) pairs:
  - n       paired keys with differing pmfs
  - RPS     ranked probability score of the pmf vs the actual (lower is better)
  - ECE     decile ECE of the randomized PIT (keyed U(0,1), identical across
            versions so the PITs are paired)
  - Brier   of P(over line) vs the line outcome, over the `n line` keys: the
            n keys that have a graded book line in `nfl_prop_grades`, pushes
            excluded. P(over) = 1 - F(floor(line)) from the version's pmf.

`nfl_prop_grades` carries pass_yds / rush_yds / rec_yds / receptions /
rush_att (serving.props_ev.LINE_MARKETS; its `market` column already uses the
sim's names, rec_yds included). pass_tds and anytime_td have no graded lines,
so they are compared on RPS/ECE only.

Verdict per market (symmetric dominance; Brier only when n line > 0, ties
within 1e-12): "ML better" if neither RPS nor Brier is worse and at least one
is strictly better (RPS lower with Brier <= , or RPS tied with Brier strictly
lower); "ML worse" is the mirror image; "no difference" if every metric ties;
else "mixed". Overall: markets with n = 0 (all identical) are ignored;
"inconclusive (n < 300 per market)" when any remaining market has n < 300;
otherwise "ML better" / "ML worse" when every differing market agrees, else
"mixed"; "no difference (all paired pmfs identical)" when nothing differs.

Read-only (the DB session is set read_only). `--since` defaults to the first
row's created_at of the compared ML version; with no rows of it it prints a
message and exits 0 without writing a report. The report is named
`<date>-props-ml-shadow.md` for nfl-sim-ml-v1 and
`<date>-props-ml-shadow-<version>.md` for another version.

Usage:
    DATABASE_URL=... PYTHONPATH=src uv run python scripts/compare_props_shadow.py \
        [--since 2026-09-20] [--ml-version nfl-sim-ml-v2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.model.props_eval import decile_ece, gate_population, pit_pmf, pit_uniform, rps_pmf

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "superpowers" / "reports"

V1_VERSION = "sim-nfl-v1"
ML_VERSIONS = ("nfl-sim-ml-v1", "nfl-sim-ml-v2")
ML_VERSION = ML_VERSIONS[0]   # the default when no ML version is served / readable
ML_VERSION_ENV = "PROPS_SHADOW_ML_VERSION"
MIN_N = 300
MARKET_ORDER = ("pass_yds", "rush_yds", "rec_yds", "receptions", "rush_att",
                "pass_tds", "anytime_td")
_EPS = 1e-12


# ---------------------------------------------------------------- ML version

def resolve_ml_version(cli: str | None, env: Mapping[str, str],
                       served: Callable[[], str | None]) -> tuple[str, str]:
    """(the ML version to compare, where it came from): ``--ml-version``, else
    env ``PROPS_SHADOW_ML_VERSION``, else the served version (``served()``,
    i.e. nfl_sim_serving) when it is an ML version, else ``ML_VERSION``
    (nfl-sim-ml-v1; also when the served version cannot be read). ValueError
    for an explicit version that is not an ML version."""
    for value, source in ((cli, "--ml-version"), (env.get(ML_VERSION_ENV), ML_VERSION_ENV)):
        if value and value.strip():
            v = value.strip()
            if v not in ML_VERSIONS:
                raise ValueError(f"{source} {v!r}: expected one of {', '.join(ML_VERSIONS)}")
            return v, source
    try:
        s = served()
    except Exception as exc:  # noqa: BLE001 -- the default must not need the serving table
        print(f"could not read nfl_sim_serving ({type(exc).__name__}: {exc}); comparing {ML_VERSION}")
        s = None
    if s in ML_VERSIONS:
        return str(s), "served (nfl_sim_serving)"
    return ML_VERSION, "default (no ML version served)"


def report_name(run_date: str, ml_version: str = ML_VERSION) -> str:
    """The report file name; v1 keeps the historical name."""
    sfx = "" if ml_version == ML_VERSION else f"-{ml_version}"
    return f"{run_date}-props-ml-shadow{sfx}.md"


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
    """Per-market verdict (ML relative to v1), symmetric dominance: "ML better"
    iff no metric is worse and at least one is strictly better, "ML worse" the
    mirror, "no difference" if all tie, else "mixed". Brier is ignored when
    None (no graded lines)."""
    signs = [_cmp(rps_ml, rps_v1)]
    if brier_v1 is not None and brier_ml is not None:
        signs.append(_cmp(brier_ml, brier_v1))
    if all(x == 0 for x in signs):
        return "no difference"
    if all(x <= 0 for x in signs):
        return "ML better"
    if all(x >= 0 for x in signs):
        return "ML worse"
    return "mixed"


def overall_verdict(markets: dict) -> str:
    """Overall verdict from per-market {n, verdict}. Markets with n = 0 (only
    identical pmf pairs) are ignored, including by the n < MIN_N gate."""
    if not markets:
        return "inconclusive (no paired rows)"
    scored = {k: m for k, m in markets.items() if m["n"] > 0}
    if not scored:
        return "no difference (all paired pmfs identical)"
    if any(m["n"] < MIN_N for m in scored.values()):
        return f"inconclusive (n < {MIN_N} per market)"
    differing = {m["verdict"] for m in scored.values()} - {"no difference"}
    if not differing:
        return "no difference"
    if differing == {"ML better"}:
        return "ML better"
    if differing == {"ML worse"}:
        return "ML worse"
    return "mixed"


def _mean(r: dict) -> float | None:
    """A sim row's projected mean: the `mean` column when present (DB rows),
    else the dist's "mean"."""
    if r.get("mean") is not None:
        return float(r["mean"])
    dist = r.get("dist")
    if isinstance(dist, str):
        dist = json.loads(dist)
    if isinstance(dist, dict) and dist.get("mean") is not None:
        return float(dist["mean"])
    return None


def shadow_population(v1_rows: list[dict]) -> set[tuple]:
    """PURE. The gated (game_pk, player_id, market) keys:
    `props_eval.gate_population` of the v1 (current sim) means."""
    return gate_population((_key(r), _mean(r)) for r in v1_rows)


def compare_shadow(v1_rows: list[dict], ml_rows: list[dict],
                   actuals: list[dict], lines: list[dict]) -> dict:
    """PURE. `score_shadow` over the GATED population (its markets / verdict
    are THE result) plus "ungated": `score_shadow` over every paired key,
    and "n_gated_keys"."""
    pop = shadow_population(v1_rows)
    gated = score_shadow(v1_rows, ml_rows, actuals, lines, population=pop)
    return {**gated, "n_gated_keys": len(pop),
            "ungated": score_shadow(v1_rows, ml_rows, actuals, lines)}


def score_shadow(v1_rows: list[dict], ml_rows: list[dict],
                 actuals: list[dict], lines: list[dict],
                 population: set[tuple] | None = None) -> dict:
    """PURE. Score both versions on the paired (game_pk, player_id, market)
    keys that have an actual -- only those in `population` when given.
    Rows: sims {game_pk, player_id, market, dist (dict or JSON str)};
    actuals {game_pk, player_id, market, actual, season, week}; lines
    {game_pk, player_id, market, line}.

    Pairs whose pmfs are identical (equal length and np.array_equal) are
    excluded from n and every metric and counted in n_identical.

    Returns {"markets": {market: {n, n_identical, rps_v1, rps_ml, ece_v1,
    ece_ml, n_line, brier_v1, brier_ml, verdict}}, "n_identical": int,
    "skipped": int, "verdict": str}; metrics are None for a market with n = 0.
    `skipped` counts paired keys with an actual dropped for a missing/empty pmf."""
    v1 = {_key(r): r for r in v1_rows}
    ml = {_key(r): r for r in ml_rows}
    act = {_key(r): r for r in actuals if r.get("actual") is not None}
    line_by = {_key(r): float(r["line"]) for r in lines if r.get("line") is not None}

    per: dict[str, dict[str, list]] = {}
    identical: dict[str, int] = {}
    skipped = 0
    keys = set(v1) & set(ml) & set(act)
    if population is not None:
        keys &= population
    for key in sorted(keys):
        pmf_v1, pmf_ml = _pmf(v1[key].get("dist")), _pmf(ml[key].get("dist"))
        if pmf_v1 is None or pmf_ml is None:
            skipped += 1
            continue
        _, player_id, market = key
        arr_v1, arr_ml = np.asarray(pmf_v1, dtype=float), np.asarray(pmf_ml, dtype=float)
        if arr_v1.shape == arr_ml.shape and np.array_equal(arr_v1, arr_ml):
            identical[market] = identical.get(market, 0) + 1
            continue
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
    for market in set(per) | set(identical):
        n_ident = identical.get(market, 0)
        acc = per.get(market)
        if acc is None:  # only identical pairs: nothing to score
            markets[market] = {
                "n": 0, "n_identical": n_ident,
                "rps_v1": None, "rps_ml": None, "ece_v1": None, "ece_ml": None,
                "n_line": 0, "brier_v1": None, "brier_ml": None,
                "verdict": f"all identical (n_identical={n_ident})",
            }
            continue
        rps_v1, rps_ml = float(np.mean(acc["rps_v1"])), float(np.mean(acc["rps_ml"]))
        n_line = len(acc["br_v1"])
        brier_v1 = float(np.mean(acc["br_v1"])) if n_line else None
        brier_ml = float(np.mean(acc["br_ml"])) if n_line else None
        markets[market] = {
            "n": len(acc["rps_v1"]), "n_identical": n_ident,
            "rps_v1": rps_v1, "rps_ml": rps_ml,
            "ece_v1": decile_ece(acc["pit_v1"]), "ece_ml": decile_ece(acc["pit_ml"]),
            "n_line": n_line, "brier_v1": brier_v1, "brier_ml": brier_ml,
            "verdict": market_verdict(rps_v1, rps_ml, brier_v1, brier_ml),
        }
    return {"markets": markets, "n_identical": sum(identical.values()),
            "skipped": skipped, "verdict": overall_verdict(markets)}


# ---------------------------------------------------------------- rendering

def _ordered(markets: dict) -> list[str]:
    known = [m for m in MARKET_ORDER if m in markets]
    return known + sorted(set(markets) - set(MARKET_ORDER))


def _f(x: float | None) -> str:
    return "–" if x is None else f"{x:.4f}"


def format_table(result: dict) -> str:
    rows = ["| market | n | n identical | RPS v1 | RPS ML | ECE v1 | ECE ML | n line "
            "| Brier v1 | Brier ML | verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for market in _ordered(result["markets"]):
        m = result["markets"][market]
        rows.append(
            f"| {market} | {m['n']} | {m['n_identical']} | {_f(m['rps_v1'])} | {_f(m['rps_ml'])} | "
            f"{_f(m['ece_v1'])} | {_f(m['ece_ml'])} | {m['n_line']} | "
            f"{_f(m['brier_v1'])} | {_f(m['brier_ml'])} | {m['verdict']} |")
    return "\n".join(rows)


def render_report(result: dict, since: str, run_date: str, null_commence: int = 0,
                  ml_version: str = ML_VERSION) -> str:
    order = _ordered(result["markets"])
    n_by = ", ".join(f"{m} {result['markets'][m]['n']}" for m in order)
    ident_by = ", ".join(f"{m} {result['markets'][m]['n_identical']}" for m in order
                         if result["markets"][m]["n_identical"])
    ident_line = f"- Identical-pmf pairs excluded: {result['n_identical']}"
    ident_line += f" ({ident_by})." if ident_by else "."
    lines = [
        f"# Props-ML shadow comparison — {run_date}",
        "",
        f"**Verdict: {result['verdict']}.**",
        "",
        f"Per-market n: {n_by or 'none'}.",
        "",
        "## Run",
        "",
        f"- Versions: {V1_VERSION} (current sim) vs {ml_version} (ML), from nfl_player_sim.",
        f"- Since: {since} (games with commence_time on or after this, already kicked off).",
        "- Population (verdict): the GATED population -- keys inside the shared props-ML "
        "projected-usage gate (props_eval.gate_population) on the v1 means, the population "
        "the ship gate scored and serving applies -- present in both versions with a "
        "realized stat in nfl_player_actuals (paired)"
        + (f"; {result['n_gated_keys']} gated v1 keys" if "n_gated_keys" in result else "")
        + ". Pairs whose ML and v1 pmfs are "
        "identical (per-game ML fallbacks, baseline-sourced markets) are excluded from n "
        "and every metric and counted as n identical. The ungated table (all paired keys) "
        "follows the gated one and is not used for the verdict.",
        ident_line,
        f"- Paired keys skipped for a missing/empty pmf: {result['skipped']}.",
        f"- nfl_player_sim rows dropped for NULL commence_time: {null_commence}.",
        "- RPS: ranked probability score vs the actual (lower is better). ECE: decile ECE of "
        "the randomized PIT (paired uniforms).",
        "- Brier: P(over line) = 1 - F(floor(line)) vs the graded line outcome from "
        "nfl_prop_grades. Brier is over the n line keys (the n keys with a graded line, "
        "pushes excluded). pass_tds and anytime_td have no graded lines.",
        "- Market verdict (symmetric; Brier only when n line > 0): ML better = no metric "
        "worse and at least one strictly better; ML worse = the mirror; no difference = all "
        "tie; otherwise mixed. A market with only identical pairs shows all identical.",
        f"- Overall: inconclusive while any market with n > 0 has n < {MIN_N}; all-identical "
        "markets are ignored.",
        "",
        "## Per market",
        "",
        format_table(result),
        "",
    ]
    if "ungated" in result:
        un = result["ungated"]
        lines += [
            "## Ungated (all paired keys — not used for the verdict)",
            "",
            f"Ungated verdict: {un['verdict']}; identical-pmf pairs excluded: "
            f"{un['n_identical']}.",
            "",
            format_table(un),
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------- DB (read-only)

def load_shadow_data(since: str | None, ml_version: str = ML_VERSION) -> dict | None:
    """The only DB access (read-only session). Returns None when no
    `ml_version` rows exist, else {"since", "v1", "ml", "actuals", "lines",
    "null_commence"} for games with commence_time in [since, now()]. `since`
    defaults to the first ML row's created_at. `null_commence` counts sim rows
    (either version, created since `since`) the window drops for a NULL
    commence_time."""
    from sportsmodel.db import get_postgres

    with get_postgres() as pg:
        pg.read_only = True  # before any statement: the session cannot write
        cur = pg.cursor()
        cur.execute("SELECT min(created_at) FROM nfl_player_sim WHERE model_version = %s",
                    (ml_version,))
        first = cur.fetchone()[0]
        if first is None:
            return None
        since_val = since if since is not None else first
        cur.execute(
            """
            SELECT count(*) FROM nfl_player_sim
            WHERE model_version IN (%s, %s) AND commence_time IS NULL
              AND created_at >= %s::timestamptz
            """,
            (V1_VERSION, ml_version, since_val),
        )
        null_commence = int(cur.fetchone()[0])
        cur.execute(
            """
            SELECT game_pk, player_id, model_version, market, mean, dist
            FROM nfl_player_sim
            WHERE model_version IN (%s, %s)
              AND commence_time >= %s::timestamptz AND commence_time <= now()
            """,
            (V1_VERSION, ml_version, since_val),
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
            (ml_version, since_val),
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

    sim_dicts = [dict(zip(("game_pk", "player_id", "model_version", "market", "mean", "dist"), r))
                 for r in sims]
    return {
        "since": since_val.isoformat() if hasattr(since_val, "isoformat") else str(since_val),
        "v1": [r for r in sim_dicts if r["model_version"] == V1_VERSION],
        "ml": [r for r in sim_dicts if r["model_version"] == ml_version],
        "actuals": [dict(zip(("game_pk", "player_id", "market", "actual", "season", "week"), r))
                    for r in actuals],
        "lines": [dict(zip(("game_pk", "player_id", "market", "line"), r)) for r in lines],
        "null_commence": null_commence,
    }


# ---------------------------------------------------------------- CLI

def _served_version() -> str | None:
    """nfl_sim_serving's model_version (IO; `db.served_nfl_sim_version`)."""
    from sportsmodel.db import served_nfl_sim_version

    return served_nfl_sim_version()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default=None,
                    help="start date/timestamp (default: the compared ML version's first row's "
                         "created_at)")
    ap.add_argument("--ml-version", default=None, choices=ML_VERSIONS,
                    help=f"ML version to compare (default: env {ML_VERSION_ENV}, else the served "
                         f"ML version, else {ML_VERSION})")
    ap.add_argument("--out-dir", default=str(REPORT_DIR))
    ap.add_argument("--run-date", default=None, help="report date (default: today)")
    args = ap.parse_args(argv)

    ml_version, source = resolve_ml_version(args.ml_version, os.environ, _served_version)
    print(f"comparing {V1_VERSION} vs {ml_version} ({source})")
    data = load_shadow_data(args.since, ml_version)
    if data is None:
        print(f"no {ml_version} rows in nfl_player_sim yet -- nothing to compare "
              "(run generate_sim_nfl.py with SIM_ML_MODE=shadow first); no report written.")
        return 0

    result = compare_shadow(data["v1"], data["ml"], data["actuals"], data["lines"])
    run_date = args.run_date or date.today().isoformat()
    print(f"Props-ML shadow comparison since {data['since']} (gated population): {result['verdict']} "
          f"(identical-pmf pairs excluded: {result['n_identical']}; "
          f"NULL commence_time rows dropped: {data.get('null_commence', 0)})")
    print(format_table(result))
    print(f"Ungated (all paired keys; not used for the verdict): {result['ungated']['verdict']}")
    print(format_table(result["ungated"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / report_name(run_date, ml_version)
    out.write_text(render_report(result, since=data["since"], run_date=run_date,
                                 null_commence=data.get("null_commence", 0),
                                 ml_version=ml_version))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
