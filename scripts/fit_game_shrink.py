"""Fit the game-level shrink factors from the committed CFB history (profit step 2 Part B) and write assets/game_shrink_k.json and the
generated site table site/js/shrink-k.js.

Runs the leak-free CFB walk-forward (cfb.walkforward.raw_model_predictions, about 10 seconds) once over the committed schedules and
closing lines, converts every game to the numbers the site would have shown (build_gameline with an empty market, as scripts/generate_cfb.py
serves them), fits k for the moneyline, spread and total (300-game minimum, per-season walk-forward stability report), prints the stability
table, and writes the two files. NFL and MLB are written as explicit zeros. It reads only committed assets: nothing else is touched.

    uv run python scripts/fit_game_shrink.py [--last-season 2025] [--out assets/game_shrink_k.json]
                                             [--site-out site/js/shrink-k.js] [--as-of ISO8601] [--dry-run]

Run by hand at season's end (pass --last-season <year>) or when the served game model changes; there is no scheduled job.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.shrink import game_fit  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
OUT_JSON = ROOT / "assets" / "game_shrink_k.json"
OUT_JS = ROOT / "site" / "js" / "shrink-k.js"


def _load_backtest():
    """scripts/backtest_cfb_gameline.py, for its load_merged_schedule (the same merge the gates and the live fix use)."""
    spec = importlib.util.spec_from_file_location("backtest_cfb_gameline", ROOT / "scripts" / "backtest_cfb_gameline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_raw(last_season: int) -> list[dict]:
    """The walk-forward rows for every merged game up to and including `last_season` (Elo is continuous across seasons)."""
    from sportsmodel.cfb.walkforward import raw_model_predictions
    from sportsmodel.nfl.elo import EloConfig
    from sportsmodel.nfl.ratings import BlendConfig

    merged = _load_backtest().load_merged_schedule(str(ASSETS / "schedules.parquet"), str(ASSETS / "lines.parquet"))
    span = merged[merged["season"] <= last_season]
    j = json.loads((ASSETS / "rating.json").read_text())
    elo_cfg = EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"], base=j.get("base", 1500.0))
    blend_cfg = BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"])
    return raw_model_predictions(span, elo_cfg, blend_cfg)


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to a temp file next to `path`, then replace: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main(argv: list[str] | None = None, *, raw_fn=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--last-season", type=int, default=game_fit.LAST_SEASON)
    ap.add_argument("--out", default=str(OUT_JSON))
    ap.add_argument("--site-out", default=str(OUT_JS))
    ap.add_argument("--as-of", default=None, help="ISO 8601 timestamp recorded as fitted_at (default: now, UTC)")
    ap.add_argument("--dry-run", action="store_true", help="print the tables, write nothing")
    args = ap.parse_args(argv)

    fitted_at = args.as_of or datetime.now(timezone.utc).isoformat()
    raw = (raw_fn or load_raw)(args.last_season)
    gl_cfg = game_fit.gameline_config(json.loads((ASSETS / "gameline.json").read_text()))
    rows = game_fit.served_rows(raw, gl_cfg)
    if not rows:
        print("fit_game_shrink: the walk-forward produced no rows; nothing written", file=sys.stderr)
        return 1
    doc = game_fit.build_document(rows, fitted_at, last_season=args.last_season)
    print(game_fit.format_tables(doc))
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    write_atomic(Path(args.out), json.dumps(doc, indent=2, sort_keys=True) + "\n")
    write_atomic(Path(args.site_out), game_fit.render_js(doc))
    print(f"wrote {args.out} and {args.site_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
