"""Starting-QB check against the books (live runs only). PURE — no IO.

The sim's QB1 is the depth chart's top active QB (usage.active_usage). A
benching or a late switch doesn't reach the nflverse chart or the injury
report, but it does reach the books: they post a passing-yards prop only for
the QB they expect to start, and pull the benched QB's. So when exactly one of
a team's charted QBs has a pass_yds line in the game's latest odds pull and he
isn't the chart's QB1, he's promoted to QB1 (depth_team "0") before usage is
built. Anything ambiguous (no line, or two of the team's QBs with lines)
leaves the chart alone. Backtests have no odds history, so this never runs
there.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from sportsmodel.nfl.injury_report import name_key
from sportsmodel.sim.nfl.usage import _depth_team_int, _latest_depth_week

# A player counts as "posted" when his latest pass_yds capture is within this
# long of the game's latest pass_yds capture (one pull stamps all its rows at
# the same moment; the slack covers a pull that straddles a few minutes).
LATEST_PULL_SLACK = timedelta(minutes=60)


def books_qb_names(odds_rows: list[dict]) -> dict[int, set[str]]:
    """{game_pk -> name_keys of players with a pass_yds line in that game's
    latest pull}. `odds_rows`: dicts with game_pk, player_name, captured_at
    (pass_yds rows only). A player whose line was pulled drops out because his
    last capture is older than the game's latest pull."""
    latest_by_player: dict[tuple, object] = {}
    latest_by_game: dict[int, object] = {}
    for r in odds_rows:
        g, ts = r["game_pk"], r["captured_at"]
        key = (g, name_key(r["player_name"]))
        if key not in latest_by_player or ts > latest_by_player[key]:
            latest_by_player[key] = ts
        if g not in latest_by_game or ts > latest_by_game[g]:
            latest_by_game[g] = ts
    out: dict[int, set[str]] = {}
    for (g, nk), ts in latest_by_player.items():
        if nk and ts >= latest_by_game[g] - LATEST_PULL_SLACK:
            out.setdefault(g, set()).add(nk)
    return out


def promote_books_qb(
    depth_df: pd.DataFrame,
    team: str,
    upto_season: int,
    upto_week: int,
    posted: set[str],
    out_names: set[str] | None = None,
) -> tuple[pd.DataFrame, tuple[str, str] | None]:
    """Return (depth_df, (old QB1, new QB1)) with the books' QB promoted to
    depth_team "0" on the chart `active_usage` will read for `team` (the
    exact week, else its latest-chart fallback), or (depth_df unchanged, None)
    when there's nothing to do. Names in `out_names` (the injury report's Out
    list) never count as the chart's QB1 or as the books' pick."""
    if not posted:
        return depth_df, None   # no current lines for this game: the chart decides
    week = (upto_season, upto_week)
    exact = ((depth_df["club_code"] == team) & (depth_df["season"] == upto_season)
             & (depth_df["week"] == upto_week))
    if not exact.any():
        week = _latest_depth_week(depth_df, team, upto_season, upto_week)
        if week is None:
            return depth_df, None
    mask = ((depth_df["club_code"] == team) & (depth_df["season"] == week[0])
            & (depth_df["week"] == week[1])
            & (depth_df["position"].astype(str).str.strip().str.upper() == "QB"))
    qbs = depth_df[mask]
    out = {name_key(n) for n in (out_names or set())}

    def keys(row) -> set[str]:
        return {name_key(n) for n in (row.get("full_name"), row.get("football_name"))
                if isinstance(n, str) and n.strip()} - {""}

    healthy = [(idx, row) for idx, row in qbs.iterrows() if not keys(row) & out]
    if not healthy:
        return depth_df, None
    matched = [(idx, row) for idx, row in healthy if keys(row) & posted]
    if len({str(row.get("gsis_id")) for _, row in matched}) != 1:
        return depth_df, None   # no line for this team's QBs, or ambiguous
    pick = matched[0][1]
    pick_id = str(pick.get("gsis_id"))
    dt = lambda row: _depth_team_int(row.get("depth_team"))
    others = [row for _, row in healthy if str(row.get("gsis_id")) != pick_id]
    pick_dt = min(dt(row) for _, row in matched)
    if not others or all(pick_dt < dt(row) for row in others):
        return depth_df, None   # the chart already has him as the sole QB1
    top = min(others, key=dt)
    fixed = depth_df.copy()
    # boolean mask, not index labels: depth_df's index needn't be unique
    # the slot keeps the column's own type (the live chart's is int32; older
    # schemas carry strings)
    slot = 0 if pd.api.types.is_numeric_dtype(fixed["depth_team"]) else "0"
    fixed.loc[(mask & (depth_df["gsis_id"].astype(str) == pick_id)).to_numpy(), "depth_team"] = slot
    name = lambda row: str(row.get("full_name") or row.get("football_name") or row.get("gsis_id"))
    return fixed, (name(top), name(pick))
