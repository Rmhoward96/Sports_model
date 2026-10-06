"""Tests for the pure seams in build_cfb_advanced.py (parse_advanced, merge_frames, coverage).

Payload shape follows the documented CFBD /stats/game/advanced schema
(offense/defense objects with passingPlays / rushingPlays sub-objects).
"""
import importlib.util
import math
import pathlib

import pandas as pd

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_advanced.py"
_spec = importlib.util.spec_from_file_location("build_cfb_advanced", _p)
bca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bca)


def _unit(ppa, sr, ex, pass_=None, rush=None, plays=60):
    u = {"plays": plays, "drives": 10, "ppa": ppa, "totalPPA": ppa * plays,
         "successRate": sr, "explosiveness": ex, "powerSuccess": 0.6, "stuffRate": 0.2}
    if pass_ is not None:
        u["passingPlays"] = pass_
    if rush is not None:
        u["rushingPlays"] = rush
    return u


def _split(ppa, sr, ex, **extra):
    return {"ppa": ppa, "totalPPA": ppa * 20, "successRate": sr, "explosiveness": ex, **extra}


PAYLOAD = [
    {"gameId": 401, "season": 2023, "week": 3, "team": "Alabama", "opponent": "Georgia",
     "offense": _unit(0.30, 0.48, 1.30, pass_=_split(0.40, 0.50, 1.5, plays=25),
                      rush=_split(0.15, 0.45, 1.1, plays=35)),
     "defense": _unit(0.05, 0.40, 1.10, pass_=_split(0.10, 0.42, 1.2),
                      rush=_split(-0.02, 0.38, 1.0))},
    # unmapped team (FCS school) -> dropped and counted
    {"gameId": 402, "season": 2023, "week": 3, "team": "Alabama", "opponent": "Nowhere State Fighting Pickles",
     "offense": _unit(0.5, 0.5, 1.2), "defense": _unit(0.0, 0.3, 1.0)},
    # missing sub-objects -> NaN; no play counts on splits -> pass/rush plays NaN
    {"gameId": 403, "season": 2023, "week": 4, "team": "Georgia", "opponent": "Alabama",
     "offense": {"plays": 70, "ppa": 0.2, "successRate": 0.44, "explosiveness": 1.2,
                 "passingPlays": {"ppa": 0.3, "successRate": 0.46, "explosiveness": 1.4},
                 "rushingPlays": {"ppa": 0.1, "successRate": 0.42, "explosiveness": 1.0}}},
]


def test_parse_maps_fields_and_teams():
    df = bca.parse_advanced(PAYLOAD)
    assert len(df) == 2
    assert df.attrs["dropped"] == 1
    r = df.iloc[0]
    from sportsmodel.cfb.teams import cfbd_to_espn
    assert r["team"] == cfbd_to_espn("Alabama") and r["opponent"] == cfbd_to_espn("Georgia")
    assert (r["season"], r["week"], r["game_id"]) == (2023, 3, 401)
    assert r["off_plays"] == 60 and r["off_ppa"] == 0.30
    assert r["off_success"] == 0.48 and r["off_explosiveness"] == 1.30
    assert r["off_pass_ppa"] == 0.40 and r["off_pass_success"] == 0.50
    assert r["off_pass_explosiveness"] == 1.5
    assert r["off_rush_ppa"] == 0.15 and r["off_rush_success"] == 0.45
    assert r["off_rush_explosiveness"] == 1.1
    assert r["off_pass_plays"] == 25 and r["off_rush_plays"] == 35
    assert r["def_ppa"] == 0.05 and r["def_success"] == 0.40 and r["def_plays"] == 60
    assert r["def_rush_ppa"] == -0.02 and r["def_pass_explosiveness"] == 1.2


def test_missing_subobjects_are_nan_not_zero():
    r = bca.parse_advanced(PAYLOAD).iloc[1]
    assert r["off_ppa"] == 0.2 and r["off_pass_ppa"] == 0.3
    assert math.isnan(r["off_pass_plays"]) and math.isnan(r["off_rush_plays"])
    for c in ("def_plays", "def_ppa", "def_success", "def_explosiveness", "def_pass_ppa",
              "def_rush_success", "def_rush_explosiveness"):
        assert math.isnan(r[c]), c


def test_empty_payload_keeps_schema():
    df = bca.parse_advanced([])
    assert len(df) == 0 and set(bca.COLUMNS) == set(df.columns)


def test_merge_replaces_only_given_seasons():
    old = bca.parse_advanced(PAYLOAD)
    old2 = old.copy()
    old2["season"] = 2022
    old2["game_id"] = old2["game_id"] + 1000
    existing = pd.concat([old, old2], ignore_index=True)
    new = old.copy()
    new["off_ppa"] = 9.9
    out = bca.merge_frames(existing, new, [2023])
    assert set(out["season"]) == {2022, 2023}
    assert (out[out.season == 2023]["off_ppa"] == 9.9).all()
    assert (out[out.season == 2022]["off_ppa"] != 9.9).all()


def test_coverage_share_of_fbs_games_with_both_sides():
    sched = pd.DataFrame({
        "season": [2023, 2023, 2023, 2023],
        "home_team": ["333", "61", "FCS", "2"], "away_team": ["61", "333", "2", "5"],
        "game_pk": [1, 2, 3, 4],
    })
    adv = pd.DataFrame({
        "season": [2023, 2023, 2023], "game_id": [1, 1, 2],
        "team": ["333", "61", "61"], "opponent": ["61", "333", "333"],
    })
    cov = bca.coverage(adv, sched, fbs={"333", "61", "2", "5"})
    # FBS-vs-FBS games: 1, 2, 4 (3 has FCS). Game 1 has both sides; 2 only one; 4 none.
    assert math.isclose(cov.loc[2023, "both_sides"], 1 / 3)
    assert cov.loc[2023, "fbs_games"] == 3


def _run_main(monkeypatch, tmp_path, payloads, existing, argv):
    """Run bca.main() with fetch_year stubbed and _OUT redirected to tmp_path."""
    out = tmp_path / "advanced_games.parquet"
    if existing is not None:
        existing.to_parquet(out)
    monkeypatch.setattr(bca, "_OUT", out)
    monkeypatch.setattr(bca, "_SCHEDULES", tmp_path / "missing.parquet")
    monkeypatch.setattr(bca, "ROOT", tmp_path)
    monkeypatch.setattr(bca, "fetch_year", lambda y, key, st="regular":
                        payloads[y] if st == "regular" else payloads.get(("post", y), []))
    monkeypatch.setenv("CFBD_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["build_cfb_advanced.py", *argv])
    bca.main()
    return pd.read_parquet(out)


def test_merge_empty_payload_keeps_existing_season(monkeypatch, tmp_path, capsys):
    old = bca.parse_advanced(PAYLOAD)                 # season 2023
    old22 = old.assign(season=2022, game_id=old["game_id"] + 1000)
    existing = pd.concat([old, old22], ignore_index=True)
    got = _run_main(monkeypatch, tmp_path, {2023: [], 2022: [dict(PAYLOAD[0], season=2022, gameId=1401)]}, existing,
                    ["--merge", "--seasons", "2023", "2022"])
    msg = capsys.readouterr().out
    assert "::warning::" in msg and "KEEPING" in msg and "2023" in msg
    assert set(got["season"]) == {2022, 2023}
    assert len(got[got.season == 2023]) == len(old)   # not wiped
    assert (got[got.season == 2023]["game_id"].isin(old["game_id"])).all()


def test_merge_all_empty_keeps_file_and_nonempty_still_replaces(monkeypatch, tmp_path):
    old = bca.parse_advanced(PAYLOAD)
    got = _run_main(monkeypatch, tmp_path, {2023: []}, old, ["--merge", "--seasons", "2023"])
    assert len(got) == len(old)
    fresh = [dict(PAYLOAD[0], gameId=999)]
    got = _run_main(monkeypatch, tmp_path, {2023: fresh}, old, ["--merge", "--seasons", "2023"])
    assert set(got["game_id"]) == {999}               # a real payload still replaces the season


def test_empty_payload_without_existing_rows_is_not_an_error(monkeypatch, tmp_path):
    got = _run_main(monkeypatch, tmp_path, {2024: []}, None, ["--merge", "--seasons", "2024"])
    assert len(got) == 0


def _real_split(ppa, n, sr=0.45, ex=1.2):
    # CFBD's real split shape: no `plays`, count only implied by totalPPA / ppa
    return {"ppa": ppa, "totalPPA": ppa * n, "successRate": sr, "explosiveness": ex}


def test_split_plays_derived_from_total_ppa_when_count_missing():
    g = {"gameId": 404, "season": 2023, "week": 5, "team": "Alabama", "opponent": "Georgia",
         "offense": _unit(0.2, 0.45, 1.2, pass_=_real_split(0.417316, 31),
                          rush=_real_split(-0.354743, 38))}
    r = bca.parse_advanced([g]).iloc[0]
    assert r["off_pass_plays"] == 31 and r["off_rush_plays"] == 38


def test_split_plays_nan_when_ratio_not_a_whole_count():
    for ppa, total in ((0.0, 0.0), (0.3, 7.45), (0.3, -3.0)):
        g = {"gameId": 405, "season": 2023, "week": 5, "team": "Alabama", "opponent": "Georgia",
             "offense": _unit(0.2, 0.45, 1.2, pass_={"ppa": ppa, "totalPPA": total},
                              rush=_real_split(0.1, 30))}
        r = bca.parse_advanced([g]).iloc[0]
        assert math.isnan(r["off_pass_plays"]), (ppa, total)
        assert r["off_rush_plays"] == 30


def _post(payload_row, game_id):
    return dict(payload_row, gameId=game_id, seasonType="postseason", week=1)


def test_parse_stamps_season_type_and_reads_down_distance_splits():
    g = dict(PAYLOAD[0], seasonType="postseason")
    g["offense"] = dict(g["offense"], standardDowns={"ppa": 0.2, "successRate": 0.55},
                        passingDowns={"ppa": -0.1, "successRate": 0.31},
                        lineYards=3.1, stuffRate=0.17, powerSuccess=0.7)
    df = bca.parse_advanced([g, PAYLOAD[0]])
    assert list(df["season_type"]) == ["postseason", "regular"]      # missing seasonType = regular
    r = df.iloc[0]
    assert r["off_std_down_success"] == 0.55 and r["off_pass_down_ppa"] == -0.1
    assert r["off_line_yards"] == 3.1 and r["off_stuff_rate"] == 0.17 and r["off_power_success"] == 0.7
    assert math.isnan(df.iloc[1]["off_std_down_success"])            # absent split -> NaN, not 0


def test_keep_postseason_preserves_committed_bowls_on_empty_post_pull():
    reg = bca.parse_advanced([PAYLOAD[0]])
    old_post = bca.parse_advanced([_post(PAYLOAD[0], 777)])
    existing = pd.concat([reg, old_post], ignore_index=True)
    kept = bca.keep_postseason(existing, reg, 2023)
    assert set(kept["game_id"]) == {401, 777}
    fresh_post = bca.parse_advanced([_post(PAYLOAD[0], 888)])
    new = pd.concat([reg, fresh_post], ignore_index=True)
    assert set(bca.keep_postseason(existing, new, 2023)["game_id"]) == {401, 888}   # real pull wins
    assert bca.keep_postseason(None, reg, 2023) is reg


def test_main_pulls_both_season_types_by_default(monkeypatch, tmp_path):
    payloads = {2023: [PAYLOAD[0]], ("post", 2023): [_post(PAYLOAD[0], 555)]}
    got = _run_main(monkeypatch, tmp_path, payloads, None, ["--seasons", "2023"])
    assert dict(zip(got["game_id"], got["season_type"])) == {401: "regular", 555: "postseason"}


def test_main_empty_postseason_keeps_existing_bowls(monkeypatch, tmp_path):
    old = pd.concat([bca.parse_advanced([PAYLOAD[0]]),
                     bca.parse_advanced([_post(PAYLOAD[0], 777)])], ignore_index=True)
    got = _run_main(monkeypatch, tmp_path, {2023: [dict(PAYLOAD[0], gameId=402)]}, old,
                    ["--merge", "--seasons", "2023"])
    assert set(got["game_id"]) == {402, 777}


def test_main_empty_regular_pull_keeps_existing_season_even_with_postseason_rows(monkeypatch, tmp_path, capsys):
    old = bca.parse_advanced([PAYLOAD[0]])
    payloads = {2023: [], ("post", 2023): [_post(PAYLOAD[0], 555)]}      # postseason-only pull
    got = _run_main(monkeypatch, tmp_path, payloads, old, ["--merge", "--seasons", "2023"])
    assert set(got["game_id"]) == {401}                                  # regular rows not replaced
    assert "KEEPING" in capsys.readouterr().out


def test_main_regular_not_requested_still_replaces(monkeypatch, tmp_path):
    old = bca.parse_advanced([PAYLOAD[0]])
    payloads = {2023: [], ("post", 2023): [_post(PAYLOAD[0], 555)]}
    got = _run_main(monkeypatch, tmp_path, payloads, old,
                    ["--merge", "--seasons", "2023", "--season-types", "postseason"])
    assert 555 in set(got["game_id"])


def test_merge_frames_column_order_is_stable():
    new = bca.parse_advanced([PAYLOAD[0]])
    shuffled = new[list(reversed(new.columns))]
    assert list(bca.merge_frames(shuffled, new, [2024]).columns) == bca.COLUMNS
    assert list(bca.merge_frames(None, shuffled, [2023]).columns) == bca.COLUMNS
