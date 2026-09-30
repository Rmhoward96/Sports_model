"""build_team_context.py: the daily team-context job (game log, history, matchup
grades, power rankings) on synthetic NFL / CFB fixtures, plus the db upserts.

No network, no DB: the loaders are monkeypatched with fixtures and
``db.get_postgres`` is replaced by a fake connection (or made to raise).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel import db
from sportsmodel.cfb.teams import load_fbs_ids
from tests.context import synth

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_team_context.py"
_spec = importlib.util.spec_from_file_location("build_team_context", _p)
btc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(btc)

NFL_TEAMS = ["KC", "BUF", "BAL", "CIN", "DAL", "PHI", "SF", "DET"]
CUR = 2026
N_WEEKS, PLAYED = 6, 3
NOW = pd.Timestamp("2026-09-26T12:00:00Z")   # 2 days before week 4 (week 5 is > 8 days out)
TABLE_KEYS = {"team_game_log": ("sport", "game_key", "team"),
              "team_history": ("sport", "game_pk", "side"),
              "matchup_grades": ("sport", "game_pk", "side"),
              "power_rankings": ("sport", "season", "week", "team")}


_STRENGTH = {"A": 2.0, "H": -2.0}


def _day(season: int, week: int) -> pd.Timestamp:
    return pd.Timestamp(f"{season}-09-07") + pd.Timedelta(days=7 * (week - 1))


def _schedule(teams: list[str], seasons, seed: int = 3):
    """(season, week, gi, home, away, played, hs, as, spread) for a round robin."""
    rng = np.random.default_rng(seed)
    rr = synth.round_robin(synth.TEAMS)
    m = dict(zip(synth.TEAMS, teams))
    out = []
    for s in seasons:
        for w in range(1, N_WEEKS + 1):
            for gi, (h, a) in enumerate(rr[(w - 1) % len(rr)]):
                played = not (s == CUR and w > PLAYED)
                hs, as_ = ((float(rng.integers(10, 38)), float(rng.integers(10, 38)))
                           if played else (np.nan, np.nan))
                # closing spread tracks the synthetic strengths (A strong, H weak) so the
                # NFL market-scale fit has a positive slope
                sp = 1.5 + 4.0 * (_STRENGTH.get(h, 0.0) - _STRENGTH.get(a, 0.0))
                out.append((s, w, gi, m[h], m[a], hs, as_,
                            float(sp + rng.choice([-1.5, -0.5, 0.5, 1.5]))))
    return out


def _unit_games(teams: list[str], seasons) -> pd.DataFrame:
    ug = synth.league(list(seasons), n_weeks=N_WEEKS,
                      profile={"A": {"pass_o": 2.0, "run_o": 1.0}, "H": {"pass_d": 2.0}})
    m = dict(zip(synth.TEAMS, teams))
    ug["team"], ug["opponent"] = ug["team"].map(m), ug["opponent"].map(m)
    return ug[~((ug["season"] == CUR) & (ug["week"] > PLAYED))].reset_index(drop=True)


def nfl_sources() -> dict:
    seasons = range(CUR - 4, CUR + 1)
    rows = []
    for s, w, gi, h, a, hs, as_, sp in _schedule(NFL_TEAMS, seasons):
        rows.append({"game_id": f"{s}_{w:02d}_{a}_{h}", "season": s, "game_type": "REG",
                     "week": w, "gameday": str(_day(s, w).date()), "gametime": "13:00",
                     "away_team": a, "away_score": as_, "home_team": h, "home_score": hs,
                     "location": "Home", "espn": str(400_000_000 + s * 1000 + w * 10 + gi),
                     "spread_line": sp, "total_line": 44.5})
    return {"schedules": pd.DataFrame(rows), "unit_games": _unit_games(NFL_TEAMS, seasons)}


def _cfb_teams() -> list[str]:
    return sorted(load_fbs_ids(), key=int)[:8]


def _advanced(ug: pd.DataFrame) -> pd.DataFrame:
    """Task 1 advanced schema from offense unit games (def_* = opponent's offense)."""
    cmap = {"pass_epa": "pass_ppa", "pass_success": "pass_success",
            "pass_explosive": "pass_explosiveness", "run_epa": "rush_ppa",
            "run_success": "rush_success", "run_explosive": "rush_explosiveness"}
    opp = ug.rename(columns={"team": "opponent", "opponent": "team"})
    m = ug.merge(opp[["season", "game_id", "team", *cmap]], on=["season", "game_id", "team"],
                 suffixes=("", "_def"))
    gid = {g: i + 1 for i, g in enumerate(sorted(m["game_id"].unique()))}
    out = pd.DataFrame({"season": m["season"], "week": m["week"],
                        "game_id": m["game_id"].map(gid).astype("int64"),
                        "team": m["team"], "opponent": m["opponent"]})
    for u, c in cmap.items():
        out[f"off_{c}"] = m[u]
        out[f"def_{c}"] = m[f"{u}_def"]
    for c in ("plays", "ppa", "success", "explosiveness"):
        out[f"off_{c}"] = np.nan
        out[f"def_{c}"] = np.nan
    out["off_pass_plays"], out["off_rush_plays"] = m["pass_plays"], m["run_plays"]
    return out


def cfb_sources(advanced: bool = True) -> dict:
    teams = _cfb_teams()
    seasons = range(CUR - 3, CUR + 1)
    rows, lines, close = [], [], []
    for s, w, gi, h, a, hs, as_, sp in _schedule(teams, seasons, seed=5):
        pk = 500_000_000 + s * 1000 + w * 10 + gi
        kick = _day(s, w) + pd.Timedelta(hours=19)
        rows.append({"season": s, "week": w, "home_team": h, "away_team": a,
                     "home_score": hs, "away_score": as_, "game_type": "REG", "game_pk": pk,
                     "start_date": kick.strftime("%Y-%m-%dT%H:%MZ"), "neutral_site": False,
                     "conference_game": True, "home_conf": "8", "away_conf": "8"})
        if s < CUR:
            lines.append({"season": s, "week": w, "home_team": h, "away_team": a,
                          "market_spread": sp, "market_total": 52.5})
        elif w <= PLAYED:
            close.append({"game_pk": pk, "close_spread_home": sp, "close_total": 55.0})
    # an FBS team vs the pooled FCS pseudo-team (week 1 of the current season)
    rows.append({"season": CUR, "week": 1, "home_team": teams[0], "away_team": "FCS",
                 "home_score": 49.0, "away_score": 7.0, "game_type": "REG", "game_pk": 599_999_999,
                 "start_date": (_day(CUR, 1) + pd.Timedelta(hours=16)).strftime("%Y-%m-%dT%H:%MZ"),
                 "neutral_site": False, "conference_game": False, "home_conf": "8",
                 "away_conf": None})
    ug = _unit_games(teams, seasons)
    return {"schedules": pd.DataFrame(rows), "lines": pd.DataFrame(lines),
            "live_close": pd.DataFrame(close),
            "advanced": _advanced(ug) if advanced else None}


def _check_keys(table: str, df: pd.DataFrame):
    keys = list(TABLE_KEYS[table])
    assert not df.empty, table
    assert df[keys].notna().all().all(), table
    assert not df.duplicated(keys).any(), table


# ------------------------------------------------------------------ frames
def test_nfl_build_produces_all_four_frames():
    out = btc.build_nfl(**nfl_sources(), now=NOW)
    assert set(out) == set(btc.TABLES)
    for t, df in out.items():
        _check_keys(t, df)
        assert (df["sport"] == "nfl").all()
    log = out["team_game_log"]
    assert sorted(log["season"].unique()) == [CUR - 2, CUR - 1, CUR]   # current + 2 prior
    assert len(log) == 3 * N_WEEKS * 4 * 2
    assert log["game_pk"].notna().all()                                # NFL game_pk = ESPN id
    # upcoming = week 4 only (week 5 is > 8 days out): 4 games x 2 sides
    hist, gr = out["team_history"], out["matchup_grades"]
    assert len(hist) == 8 and len(gr) == 8
    assert set(hist["week"]) == {4} and set(gr["week"]) == {4}
    assert set(hist["side"]) == {"home", "away"}
    wk4 = log[(log["season"] == CUR) & (log["week"] == 4)]
    assert set(hist["game_pk"]) == set(wk4["game_pk"])
    # history is JSON-able and uses only prior games
    h0 = hist.iloc[0]
    json.dumps({"w": h0["windows"], "s": h0["streaks"], "p": h0["splits"]})
    assert h0["windows"]["season"]["n"] == PLAYED
    assert h0["windows"]["L20"]["n"] == 2 * N_WEEKS + PLAYED   # log = current + 2 prior
    # grades: letters from the frozen cutoffs, not early (3 games played)
    assert gr["overall"].isin(list("ABCDF")).all()
    assert not gr["early"].any()
    assert gr["units"].map(lambda u: isinstance(u, dict)).all()
    # rankings: upcoming week, all 8 teams, previous week present
    rk = out["power_rankings"]
    assert set(rk["week"]) == {4} and set(rk["season"]) == {CUR}
    assert sorted(rk["rank"]) == list(range(1, 9))
    assert rk["prev_rank"].notna().all()
    assert rk.iloc[0]["rating"] == rk["rating"].max()
    assert rk["units"].map(lambda u: isinstance(u, dict)).all()


def test_cfb_build_produces_all_four_frames():
    out = btc.build_cfb(**cfb_sources(), now=NOW)
    assert set(out) == set(btc.TABLES)
    for t, df in out.items():
        _check_keys(t, df)
        assert (df["sport"] == "cfb").all()
    teams = _cfb_teams()
    hist, gr, rk = out["team_history"], out["matchup_grades"], out["power_rankings"]
    assert len(hist) == 8 and len(gr) == 8
    assert "FCS" not in set(hist["team"]) and "FCS" not in set(rk["team"])
    assert set(rk["team"]) == set(teams)
    assert gr["overall"].notna().all()
    # current-season lines come from live_close: played 2026 games have ATS
    log = out["team_game_log"]
    cur = log[(log["season"] == CUR) & log["su"].notna() & (log["team"] != "FCS")
              & (log["opponent"] != "FCS")]
    assert cur["ats"].notna().all()
    # FCS game: team row kept in the log, never ranked / given a history
    assert ((log["team"] == "FCS") & (log["season"] == CUR)).sum() == 1
    assert rk["conf"].eq("8").all()


def test_cfb_missing_advanced_skips_grades_with_warning(capsys):
    out = btc.build_cfb(**cfb_sources(advanced=False), now=NOW)
    msg = capsys.readouterr().out
    assert "::warning::" in msg and "advanced_games.parquet" in msg and "skipped" in msg
    assert out["matchup_grades"].empty
    assert list(out["matchup_grades"].columns) == btc.TABLE_COLUMNS["matchup_grades"]
    for t in ("team_game_log", "team_history", "power_rankings"):
        _check_keys(t, out[t])
    assert out["power_rankings"]["units"].isna().all()


def test_no_upcoming_games_gives_empty_history_and_rankings():
    later = pd.Timestamp("2027-06-01T00:00:00Z")
    out = btc.build_nfl(**nfl_sources(), now=later)
    assert out["team_history"].empty and out["matchup_grades"].empty
    assert out["power_rankings"].empty
    assert not out["team_game_log"].empty


def test_grades_with_no_gradeable_games_returns_empty_with_warning(capsys):
    """Every upcoming game lacking an ESPN id leaves `games` without matches: the
    concat over zero frames used to raise ValueError and fail the whole sport run."""
    up = pd.DataFrame({"game_key": ["nope"]})
    games = pd.DataFrame({"game_key": ["other"], "game_pk": [1], "home_team": ["KC"],
                          "away_team": ["BUF"], "home_is_fbs": [True], "away_is_fbs": [True],
                          "season": [CUR], "week": [4],
                          "kickoff": [pd.Timestamp("2026-09-28T17:00:00Z")]})
    out = btc._grades(up, games, btc._Ratings(None), "nfl")
    assert out.empty and list(out.columns) == btc.TABLE_COLUMNS["matchup_grades"]
    assert "::warning::" in capsys.readouterr().out


# ------------------------------------------------------------------ records
def test_records_are_db_ready():
    out = btc.build_nfl(**nfl_sources(), now=NOW)
    recs = btc.table_records(out)
    assert set(recs) == set(btc.TABLES)
    for t, rows in recs.items():
        assert list(rows[0]) == btc.TABLE_COLUMNS[t]
        for r in rows:
            for k, v in r.items():
                assert not (isinstance(v, float) and np.isnan(v)), (t, k)
                assert v is not pd.NA, (t, k)
                assert not isinstance(v, (np.integer, np.floating, np.bool_)), (t, k)
    rk = recs["power_rankings"]
    assert all(isinstance(r["rank"], int) for r in rk)
    # prev_rank / move NA -> None
    out2 = btc.build_nfl(**nfl_sources(), now=NOW)
    out2["power_rankings"].loc[0, "prev_rank"] = pd.NA
    out2["power_rankings"].loc[0, "move"] = pd.NA
    r0 = btc.table_records(out2)["power_rankings"][0]
    assert r0["prev_rank"] is None and r0["move"] is None


# ------------------------------------------------------------------ ESPN merge
def test_merge_espn_schedule_appends_new_games_only():
    asset = cfb_sources()["schedules"]
    asset = asset[asset["home_score"].notna()]
    existing = int(asset["game_pk"].iloc[-1])
    teams = _cfb_teams()
    espn_games = [
        {"game_pk": existing, "home_team": teams[0], "away_team": teams[1], "home_score": 1,
         "away_score": 2, "commence_time": "2026-09-26T19:00Z", "start_date": "2026-09-26T19:00Z",
         "neutral_site": False, "conference_game": True, "home_conf": "8", "away_conf": "8",
         "status": "STATUS_FINAL", "week": 3, "season": CUR},
        {"game_pk": 1, "home_team": teams[2], "away_team": teams[3], "home_score": 21,
         "away_score": 14, "commence_time": "2026-09-27T19:00Z", "start_date": "2026-09-27T19:00Z",
         "neutral_site": True, "conference_game": False, "home_conf": "8", "away_conf": "8",
         "status": "STATUS_FINAL", "week": 4, "season": CUR},
        {"game_pk": 2, "home_team": teams[4], "away_team": teams[5], "home_score": 7,
         "away_score": 0, "commence_time": "2026-10-03T19:00Z", "start_date": "2026-10-03T19:00Z",
         "neutral_site": False, "conference_game": True, "home_conf": "8", "away_conf": "8",
         "status": "STATUS_IN_PROGRESS", "week": 5, "season": CUR},
    ]
    m = btc.merge_espn_schedule(asset, espn_games, season_type=2)
    assert len(m) == len(asset) + 2
    assert (m["game_pk"] == existing).sum() == 1                       # asset wins
    g1 = m[m["game_pk"] == 1].iloc[0]
    assert g1["home_score"] == 21 and bool(g1["neutral_site"]) and g1["game_type"] == "REG"
    g2 = m[m["game_pk"] == 2].iloc[0]
    assert np.isnan(g2["home_score"]) and np.isnan(g2["away_score"])  # not final -> unplayed
    post = btc.merge_espn_schedule(asset, [{**espn_games[1], "week": 1}], season_type=3)
    p = post[post["game_pk"] == 1].iloc[0]
    assert p["game_type"] == "POST"
    assert p["week"] > asset[asset["season"] == CUR]["week"].max()   # after the REG weeks


# ------------------------------------------------------------------ CLI
def _patch_loaders(monkeypatch, advanced=True):
    monkeypatch.setattr(btc, "load_nfl_sources", lambda now: nfl_sources())
    monkeypatch.setattr(btc, "load_cfb_sources", lambda now: cfb_sources(advanced))


def test_dry_run_prints_counts_and_never_touches_db(monkeypatch, capsys):
    _patch_loaders(monkeypatch)

    def boom():
        raise AssertionError("dry run must not open a DB connection")
    monkeypatch.setattr(db, "get_postgres", boom)
    btc.main(["--sport", "all", "--dry-run", "--now", NOW.isoformat()])
    out = capsys.readouterr().out
    for sport in ("nfl", "cfb"):
        assert f"[{sport}]" in out
    for t in btc.TABLES:
        assert t in out
    assert "runtime" in out and "top 5" in out.lower()


class _Cur:
    def __init__(self, sink):
        self.sink = sink

    def executemany(self, sql, rows):
        self.sink.append((sql, list(rows)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, sink, state):
        self.sink, self.state = sink, state

    def cursor(self):
        return _Cur(self.sink)

    def commit(self):
        self.state["commits"] += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_write_upserts_every_table_with_its_key(monkeypatch):
    _patch_loaders(monkeypatch)
    sink, state = [], {"commits": 0}
    monkeypatch.setattr(db, "get_postgres", lambda: _Conn(sink, state))
    monkeypatch.setattr(db.config, "DATABASE_URL", "postgres://fake")
    btc.main(["--sport", "nfl", "--now", NOW.isoformat()])
    assert state["commits"] == 1                       # one transaction per sport
    by_table = {}
    for sql, rows in sink:
        table = sql.split("INSERT INTO ", 1)[1].split(" ", 1)[0]
        by_table[table] = (sql, rows)
    assert set(by_table) == set(btc.TABLES)
    for t, (sql, rows) in by_table.items():
        assert f"ON CONFLICT ({', '.join(TABLE_KEYS[t])}) DO UPDATE SET" in sql
        assert "computed_at = now()" in sql
        cols = sql.split("(", 1)[1].split(")", 1)[0].split(", ")
        assert cols == db.TEAM_CONTEXT_COLUMNS[t]
        keys = [tuple(r[cols.index(k)] for k in TABLE_KEYS[t]) for r in rows]
        assert len(keys) == len(set(keys)) and all(None not in k for k in keys)
    # jsonb columns are JSON text; SQL NULL stays None
    sql, rows = by_table["team_history"]
    cols = db.TEAM_CONTEXT_COLUMNS["team_history"]
    assert isinstance(json.loads(rows[0][cols.index("windows")]), dict)
    sql, rows = by_table["power_rankings"]
    cols = db.TEAM_CONTEXT_COLUMNS["power_rankings"]
    assert isinstance(json.loads(rows[0][cols.index("units")]), dict)


def test_upsert_team_context_null_json_and_empty(monkeypatch):
    sink, state = [], {"commits": 0}
    monkeypatch.setattr(db, "get_postgres", lambda: _Conn(sink, state))
    row = {c: None for c in db.TEAM_CONTEXT_COLUMNS["power_rankings"]}
    row.update({"sport": "cfb", "season": 2026, "week": 5, "team": "333", "rank": 1,
                "rating": 20.5, "units": None})
    n = db.upsert_team_context({"power_rankings": [row], "matchup_grades": []})
    assert n == {"power_rankings": 1, "matchup_grades": 0}
    (sql, rows), = sink
    cols = db.TEAM_CONTEXT_COLUMNS["power_rankings"]
    assert rows[0][cols.index("units")] is None        # SQL NULL, not the string 'null'
    assert state["commits"] == 1
    assert db.upsert_team_context({"power_rankings": []}) == {"power_rankings": 0}


def test_main_without_database_url_refuses_to_write(monkeypatch):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(db.config, "DATABASE_URL", None)
    with pytest.raises(SystemExit):
        btc.main(["--sport", "nfl", "--now", NOW.isoformat()])


# ------------------------------------------------------------------ migration
def test_migration_matches_db_columns_and_keys():
    import re
    dbdir = pathlib.Path(__file__).resolve().parents[2] / "db"
    sql = (dbdir / "migration_team_context.sql").read_text()
    later = (dbdir / "migration_power_results.sql").read_text()   # ALTER ... ADD COLUMN
    for t, cols in db.TEAM_CONTEXT_COLUMNS.items():
        body = re.search(rf"CREATE TABLE IF NOT EXISTS {t} \((.*?)\n\);", sql, re.S).group(1)
        names = [ln.split()[0] for ln in body.splitlines()
                 if ln.strip() and not ln.strip().startswith(("PRIMARY", "--"))]
        added = re.findall(rf"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS (\w+)", later)
        assert names[-1] == "computed_at", t
        assert names[:-1] + added == cols, t
        assert f"PRIMARY KEY ({', '.join(TABLE_KEYS[t])})" in body, t
        assert db.TEAM_CONTEXT_KEYS[t] == TABLE_KEYS[t]
        assert f'CREATE POLICY "public read {t}" ON {t} FOR SELECT USING (true)' in sql
    assert "CREATE OR REPLACE VIEW power_rankings_current" in sql
    assert "power_rankings_current TO anon, authenticated" in sql


def test_one_sport_failing_does_not_block_the_other(monkeypatch, capsys):
    def bad(now):
        raise RuntimeError("nflverse down")
    monkeypatch.setattr(btc, "load_nfl_sources", bad)
    monkeypatch.setattr(btc, "load_cfb_sources", lambda now: cfb_sources())
    with pytest.raises(SystemExit):
        btc.main(["--sport", "all", "--dry-run", "--now", NOW.isoformat()])
    out = capsys.readouterr().out
    assert "::error::team-context: nfl failed" in out
    assert "[cfb] runtime" in out


class _OddsCur:
    def __init__(self, rows, seen):
        self.rows, self.seen = rows, seen

    def execute(self, sql, params):
        self.seen.append((sql, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _OddsConn:
    def __init__(self, rows, seen):
        self.rows, self.seen = rows, seen

    def cursor(self):
        return _OddsCur(self.rows, self.seen)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_load_cfb_odds_closing_consensus(monkeypatch, capsys):
    monkeypatch.setattr(db.config, "DATABASE_URL", None)
    assert btc.load_cfb_odds([1]) is None
    assert "DATABASE_URL unset" in capsys.readouterr().out
    kick = pd.Timestamp("2026-09-26T19:00Z")
    rows = [(1, "spread", "home", "bk1", -7.0, kick, kick - pd.Timedelta(hours=1)),
            (1, "spread", "home", "bk2", -6.0, kick, kick - pd.Timedelta(hours=2)),
            (1, "total", "over", "bk1", 51.5, kick, kick - pd.Timedelta(hours=1))]
    seen = []
    monkeypatch.setattr(db.config, "DATABASE_URL", "postgres://fake")
    monkeypatch.setattr(db, "get_postgres", lambda: _OddsConn(rows, seen))
    lc = btc.load_cfb_odds([1, 2])
    (sql, params), = seen
    assert "captured_at < commence_time" in sql and "player_name = ''" in sql
    assert params == ([1, 2],)
    r = lc.set_index("game_pk").loc[1]
    assert r["close_spread_home"] == 6.5 and r["close_total"] == 51.5




def _rp():
    from sportsmodel.context.results_power import ResultsParams
    return ResultsParams(cap=28.0, hfa=2.5, beta=3.0, sigma=15.0, rho=0.6)


def test_cfb_rankings_use_the_results_rating_with_sov_and_venue_records():
    from sportsmodel.context.results_power import power_asof
    src = cfb_sources()
    teams = _cfb_teams()
    priors = {t: 1500.0 + 40.0 * i for i, t in enumerate(teams)}
    out = btc.build_cfb(**src, now=NOW, priors=priors, rp=_rp(), fbs=set(teams))
    rk = out["power_rankings"].set_index("team")
    s, w = int(rk["season"].iloc[0]), int(rk["week"].iloc[0])
    members = set(teams)
    pre = {y: btc.cfb_prior_points(y, members) for y in range(s - 3, s)}
    pre[s] = btc._points_from_elo_scale(priors, members)
    exp, _ = power_asof(btc.cfb_results_games(src["schedules"]), _rp(), s, w,
                        preseason=pre, members=members)
    exp = exp.set_index("team")
    for t in teams:
        assert rk.loc[t, "rating"] == pytest.approx(exp.loc[t, "rating"])
    assert rk["home_record"].notna().all() and rk["road_record"].notna().all()
    assert rk["sov"].notna().any()
    for c in ("sov", "home_record", "road_record"):
        assert c in btc.TABLE_COLUMNS["power_rankings"]


def test_nfl_rankings_use_the_results_rating():
    from sportsmodel.context.results_power import power_asof
    src = nfl_sources()
    out = btc.build_nfl(**src, now=NOW, rp=_rp())
    rk = out["power_rankings"].set_index("team")
    s, w = int(rk["season"].iloc[0]), int(rk["week"].iloc[0])
    exp, prev = power_asof(btc.nfl_results_games(src["schedules"]), _rp(), s, w)
    exp = exp.set_index("team")
    for t in rk.index:
        assert rk.loc[t, "rating"] == pytest.approx(exp.loc[t, "rating"])
    assert rk["prev_rank"].notna().all()
    wins = rk["su"].str.split("-").str[0].astype(int)
    home_w = rk["home_record"].str.split("-").str[0].astype(int)
    road_w = rk["road_record"].str.split("-").str[0].astype(int)
    assert (home_w + road_w == wins).all()      # fixtures have no neutral games
