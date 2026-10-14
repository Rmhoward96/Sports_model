"""scripts/fit_game_shrink.py: injected walk-forward rows in, two generated files out, nothing else. No database, no network."""
import importlib.util
import json
import random
import sys
from pathlib import Path

from sportsmodel.shrink import game_fit

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fit_game_shrink", ROOT / "scripts" / "fit_game_shrink.py")
fgs = importlib.util.module_from_spec(_spec)
sys.modules["fit_game_shrink"] = fgs
_spec.loader.exec_module(fgs)

AS_OF = "2026-10-14T12:00:00+00:00"


def raw_rows(k_true=0.4, seasons=(2016, 2017, 2018, 2019), n=400):
    rng = random.Random(5)
    out = []
    for s in seasons:
        for i in range(n):
            close = rng.uniform(-20, 20)
            model = close + rng.uniform(-8, 8)
            out.append({"season": s, "week": 1 + i % 14, "home_team": "H", "away_team": "A", "model_margin": model,
                        "model_total": 52.0, "market_spread": close, "market_total": 50.0,
                        "actual_margin": close + k_true * (model - close), "actual_total": 51.0, "ml_home": None, "ml_away": None})
    return out


def run(tmp_path, rows, *extra, seen=None):
    out, js = tmp_path / "game_shrink_k.json", tmp_path / "shrink-k.js"

    def raw_fn(last_season):
        if seen is not None:
            seen.append(last_season)
        return rows

    code = fgs.main(["--as-of", AS_OF, "--out", str(out), "--site-out", str(js), *extra], raw_fn=raw_fn)
    return code, out, js


def test_it_writes_the_document_and_the_site_table_and_they_agree(tmp_path, capsys):
    code, out, js = run(tmp_path, raw_rows())
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["fitted_at"] == AS_OF and doc["model_version"] == {"cfb": "cfb-ratings-v2"}
    assert 0.3 < doc["markets"]["cfb"]["spread"]["k"] < 0.5         # the served bias (-0.58) nudges x slightly; k_true is 0.4
    assert all(e["k"] == 0.0 and e["n"] == 0 for s in ("nfl", "mlb") for e in doc["markets"][s].values())
    assert js.read_text() == game_fit.render_js(doc)
    shown = capsys.readouterr().out
    assert "cfb spread" in shown and "2018" in shown and "nfl moneyline" in shown     # the stability table is printed


def test_two_runs_with_the_same_inputs_write_identical_files(tmp_path):
    _, out, js = run(tmp_path, raw_rows())
    first = (out.read_text(), js.read_text())
    run(tmp_path, raw_rows())
    assert (out.read_text(), js.read_text()) == first
    assert sorted(p.name for p in tmp_path.iterdir()) == ["game_shrink_k.json", "shrink-k.js"]     # no temp file left behind


def test_dry_run_prints_but_writes_nothing(tmp_path, capsys):
    code, out, js = run(tmp_path, raw_rows(), "--dry-run")
    assert code == 0 and not out.exists() and not js.exists()
    assert "cfb spread" in capsys.readouterr().out


def test_no_walk_forward_rows_exits_non_zero_and_leaves_existing_files_alone(tmp_path):
    out, js = tmp_path / "game_shrink_k.json", tmp_path / "shrink-k.js"
    out.write_text("keep")
    js.write_text("keep")
    code, _, _ = run(tmp_path, [])
    assert code == 1 and out.read_text() == "keep" and js.read_text() == "keep"


def test_the_last_season_is_passed_to_the_walk_forward(tmp_path):
    seen = []
    run(tmp_path, raw_rows(), seen=seen)
    run(tmp_path, raw_rows(), "--last-season", "2026", seen=seen)
    assert seen == [2025, 2026]


def test_the_script_touches_no_database_network_or_env_file():
    text = (ROOT / "scripts" / "fit_game_shrink.py").read_text()
    for banned in ("DATABASE_URL", "get_postgres", "sportsmodel.db", "import db", "httpx", "requests", "urllib", "dotenv",
                   ".env", "psycopg"):
        assert banned not in text, banned
