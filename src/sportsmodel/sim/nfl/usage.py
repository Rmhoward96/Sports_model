"""Per-week active-roster usage model for the NFL sim (NFL-only).

`active_usage` is the module's main export: given a team's target-week depth
chart, leakage-free weekly history, and a resolved injury list, it produces the
active players' `PlayerInput` shares (renormalized over the ACTIVE set only, the
dilution fix over season-average usage) plus the starting QB's gsis_id. It is
pure — DataFrames in, values out, no IO.

Supporting this, the module bridges nflverse's two player-id namespaces (Pro
Football Reference `pfr_id` and nflverse's own `gsis_id`) via
`build_pfr_to_gsis` (pure: DataFrame in, dict out) and fetches the raw
depth-chart / snap-count / id-crosswalk frames the usage model consumes.

`fetch_usage_sources` is the only IO in this module and is not unit-tested.

`abbrev_alignment` is a small pure diagnostic shared by both callers
(`scripts/backtest_sim_nfl.py` and `scripts/generate_sim_nfl.py`): it flags
team-abbreviation codes in the depth-chart/injuries sources that fall outside
the canonical `sportsmodel.nfl.teams.TEAMS` set, which is the silent failure
mode behind `active_usage` handing back an empty roster (see its docstring).
"""
from __future__ import annotations

import pandas as pd

from sportsmodel.sim.nfl.spec import PlayerInput

# Offensive skill positions that participate in the target/carry usage model.
_SKILL_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})

# --- Cold-start priors --------------------------------------------------------
# A cold-start player is one on the target-week depth chart with NO weekly usage
# rows before the cutoff (rookie / returnee / just-signed). Instead of a
# zero-everything row (which would give them a 0 share and could leave the
# active-set renorm denominator empty), they get a small per-game pseudo-usage
# by (position, is_starter) where is_starter == (depth_team == 1). These values
# are intentionally well below a real starter's usage: they seed a plausible
# nonzero-but-small share without dominating players who actually have tape.
_COLD_TARGETS: dict[tuple[str, bool], float] = {
    ("WR", True): 5.0, ("WR", False): 1.5,
    ("TE", True): 3.5, ("TE", False): 1.0,
    ("RB", True): 2.0, ("RB", False): 0.8,
    ("QB", True): 0.0, ("QB", False): 0.0,
}
_COLD_CARRIES: dict[tuple[str, bool], float] = {
    ("RB", True): 10.0, ("RB", False): 3.0,
    ("WR", True): 0.3, ("WR", False): 0.1,
    ("TE", True): 0.0, ("TE", False): 0.0,
    ("QB", True): 1.5, ("QB", False): 0.5,
}
_COLD_TDS: dict[tuple[str, bool], float] = {
    ("WR", True): 0.35, ("WR", False): 0.10,
    ("TE", True): 0.25, ("TE", False): 0.07,
    ("RB", True): 0.40, ("RB", False): 0.12,
    ("QB", True): 0.05, ("QB", False): 0.02,
}
# Cold-start efficiency defaults (league-ish per-position constants), as
# (ypt, ypc, ypr, catch_rate). Guarded ratios can't be derived without tape,
# so these documented constants stand in.
_COLD_EFF: dict[str, tuple[float, float, float, float]] = {
    "WR": (8.1, 5.0, 13.0, 0.62),
    "TE": (7.1, 0.0, 10.5, 0.68),
    "RB": (5.6, 4.3, 7.5, 0.75),
    "QB": (0.0, 4.0, 0.0, 0.0),
}


def _is_missing_id(value: object) -> bool:
    """True if `value` is null or a blank/whitespace-only string."""
    if pd.isna(value):
        return True
    return str(value).strip() == ""


def build_pfr_to_gsis(ids_df: pd.DataFrame) -> dict[str, str]:
    """Map Pro Football Reference `pfr_id` -> nflverse `gsis_id`.

    Expects `ids_df` (nflverse `import_ids()`) with `pfr_id` and `gsis_id`
    columns. Rows with a null/blank id on either side are skipped. If the
    same `pfr_id` appears more than once, the last row wins (matches
    `import_ids()`'s de-facto one-row-per-player shape; duplicates are not
    expected in practice, but last-wins keeps this deterministic).
    """
    mapping: dict[str, str] = {}
    for row in ids_df.itertuples(index=False):
        pfr_id = getattr(row, "pfr_id", None)
        gsis_id = getattr(row, "gsis_id", None)
        if _is_missing_id(pfr_id) or _is_missing_id(gsis_id):
            continue
        mapping[str(pfr_id).strip()] = str(gsis_id).strip()
    return mapping


def _latest_depth_week(
    depth_df: pd.DataFrame, team: str, upto_season: int, upto_week: int
) -> tuple[int, int] | None:
    """Latest (season, week) <= (upto_season, upto_week) with `club_code ==
    team` rows in `depth_df`. PURE. Returns None if the team has no rows at
    or before the target at all.

    The compare is the same compound (season, week) ordering used elsewhere
    in this module: `season < upto_season OR (season == upto_season AND week
    <= upto_week)`.
    """
    mask = (depth_df["club_code"] == team) & (
        (depth_df["season"] < upto_season)
        | ((depth_df["season"] == upto_season) & (depth_df["week"] <= upto_week))
    )
    rows = depth_df[mask]
    if len(rows) == 0:
        return None
    best_season = int(rows["season"].max())
    best_week = int(rows.loc[rows["season"] == best_season, "week"].max())
    return (best_season, best_week)


def _depth_team_int(value: object) -> int:
    """Coerce a depth-chart slot ("1"/"2"/2/…) to int; unknown -> 99 (deep backup)."""
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return 99


def active_usage(
    team: str,
    upto_season: int,
    upto_week: int,
    depth_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    snaps_df: pd.DataFrame,
    pfr2gsis: dict[str, str],
    injuries_out_names: set[str],
    n_recent: int = 5,
) -> tuple[list[PlayerInput], str | None]:
    """Per-week active-roster usage: the dilution fix over season averages.

    Returns ``(active PlayerInputs, starting_qb_gsis_id)``. PURE — no IO.

    What this does differently from ``rates.player_inputs_from_weekly`` (which
    spreads shares over everyone who ever played, diluting active players by
    departed/benched ones):

    1. **Active set** is defined by the *target-week* depth chart
       (``club_code==team, season==upto_season, week==upto_week``), restricted
       to offensive skill positions (QB/RB/WR/TE), keyed by ``gsis_id`` and
       de-duplicated to each player's most prominent (lowest ``depth_team``)
       row. Players whose ``full_name`` or ``football_name`` (case-insensitive)
       is in ``injuries_out_names`` are dropped. The target-week depth chart and
       the injury list are pre-game info and ARE allowed to define who's active;
       everything else is strictly leakage-free.

       **Fallback (depth-chart availability lag):** if ``depth_df`` has NO rows
       for the team at the exact ``(upto_season, upto_week)`` (the current
       week's chart hasn't been published yet -- common early in a week or
       season), the active set is instead built from the team's most recent
       AVAILABLE chart at or before the target, via
       ``_latest_depth_week(depth_df, team, upto_season, upto_week)``. A prior
       PUBLISHED depth chart is still pre-game info known before the target
       week -- using it is not leakage, it's just a staler (but still
       backward-looking) source for the same "who's active" question. If the
       exact target week HAS rows, the fallback is never consulted. If the
       team has no chart rows at or before the target at all, the active set
       stays empty (graceful: returns ``([], None)``, same as before). Shares
       and efficiency (item 2 below) are untouched by this fallback -- they
       always come from the leakage-free recent weekly window.

    2. **Shares/efficiency** come from ``weekly_df`` rows STRICTLY before
       ``(upto_season, upto_week)`` — ``season < upto_season OR (season ==
       upto_season AND week < upto_week)`` — taking each active player's last
       ``n_recent`` games (most recent by ``(season, week)``).

       *Recency weighting scheme:* the ``k = min(n_recent, games_available)``
       selected games, ordered oldest -> newest, receive linearly increasing
       integer weights ``1, 2, …, k`` (most recent highest). Per-game usage
       (targets/carries/tds) is the weighted mean ``sum(w_i * x_i) /
       sum(w_i)``; efficiency ratios are weighted-sum-over-weighted-sum, e.g.
       ``ypr = sum(w*rec_yds) / sum(w*receptions)`` (all /0 guarded to 0.0).

       ``target_share`` / ``carry_share`` / ``td_share`` are renormalized
       **over the active set only** (denominator = sum of the active players'
       weighted per-game usage), NOT the full roster. This is the dilution fix.

    3. **Cold start:** an active player with no recent rows gets a small
       ``(position, depth_team)`` pseudo-usage prior (``_COLD_*`` tables) and a
       positional efficiency default (``_COLD_EFF``), so they carry a nonzero
       but small share and never break the renorm.

    4. **Starting QB (Ruling C1):** QB1 = the active QB with the minimal
       ``depth_team``; ties are broken by pass ``attempts`` (an unweighted sum
       over the recent window, when that column is present), then by most recent
       game, then by ``gsis_id`` for determinism. Its ``gsis_id`` is returned as the second
       element. The returned list contains EXACTLY ONE ``pos=="QB"`` — QB1, with
       ``player_id == its gsis_id`` — so the kernel can attribute team pass_yds
       to it deterministically; all other QBs are dropped. If no active QB
       exists, returns ``(players, None)`` with no QB PlayerInput.

    5. **Snaps:** ``snaps_df`` (``pfr_player_id`` + ``offense_pct``, bridged to
       gsis via ``pfr2gsis``) is accepted but NOT consumed in v1. Shares come
       from weekly usage alone. TODO: an optional modest snap-share weighting of
       rotation players is a future refinement; integrating it here added real
       complexity for little v1 benefit, so correctness was preferred. Both
       ``snaps_df`` and ``pfr2gsis`` are reserved for that.
    """
    del snaps_df, pfr2gsis  # reserved for future snap weighting (see docstring)

    injuries = {
        str(n).strip().lower()
        for n in (injuries_out_names or set())
    }

    # --- 1. Active set from the target-week depth chart ---
    dmask = (
        (depth_df["club_code"] == team)
        & (depth_df["season"] == upto_season)
        & (depth_df["week"] == upto_week)
    )
    drows = depth_df[dmask]

    if len(drows) == 0:
        fallback = _latest_depth_week(depth_df, team, upto_season, upto_week)
        if fallback is not None:
            fb_season, fb_week = fallback
            fmask = (
                (depth_df["club_code"] == team)
                & (depth_df["season"] == fb_season)
                & (depth_df["week"] == fb_week)
            )
            drows = depth_df[fmask]

    # gsis_id -> {"pos", "name", "depth_team"}
    active: dict[str, dict] = {}
    for row in drows.itertuples(index=False):
        pos = str(getattr(row, "position", "")).strip().upper()
        if pos not in _SKILL_POSITIONS:
            continue
        gsis = getattr(row, "gsis_id", None)
        if _is_missing_id(gsis):
            continue
        gsis = str(gsis).strip()

        full_name = getattr(row, "full_name", None)
        football_name = getattr(row, "football_name", None)
        names_lower = {
            str(n).strip().lower()
            for n in (full_name, football_name)
            if not _is_missing_id(n)
        }
        if names_lower & injuries:
            continue

        dt = _depth_team_int(getattr(row, "depth_team", None))
        if not _is_missing_id(full_name):
            display_name = str(full_name).strip()
        elif not _is_missing_id(football_name):
            display_name = str(football_name).strip()
        else:
            display_name = gsis

        prior = active.get(gsis)
        if prior is None or dt < prior["depth_team"]:
            active[gsis] = {"pos": pos, "name": display_name, "depth_team": dt}

    # --- 2. Recent (leakage-free) weekly rows, grouped by player_id (== gsis) ---
    if len(weekly_df) > 0:
        wmask = (weekly_df["season"] < upto_season) | (
            (weekly_df["season"] == upto_season) & (weekly_df["week"] < upto_week)
        )
        wf = weekly_df[wmask]
    else:
        wf = weekly_df

    recent_by_pid: dict[str, pd.DataFrame] = {}
    if len(wf) > 0:
        for pid, pdf in wf.groupby("player_id"):
            recent_by_pid[str(pid)] = pdf

    def _weighted(pdf: pd.DataFrame) -> dict:
        """Recency-weighted per-game usage and efficiency for one player."""
        g = pdf.sort_values(["season", "week"]).tail(n_recent)
        weights = list(range(1, len(g) + 1))  # oldest -> newest: 1..k
        sw = float(sum(weights))

        def wsum(col: str) -> float:
            # Guard NaN/missing values (nflverse weekly stat columns can carry
            # NaN for a given row) so a single bad cell can't propagate to a
            # NaN share -> NaN pval -> rng.multinomial raising downstream.
            return float(
                sum(
                    w * (v if pd.notna(v) else 0.0)
                    for w, v in zip(weights, g[col].tolist())
                )
            )

        w_targets = wsum("targets")
        w_carries = wsum("carries")
        w_rec = wsum("receptions")
        w_rec_yds = wsum("receiving_yards")
        w_rush_yds = wsum("rushing_yards")
        w_rec_tds = wsum("receiving_tds")
        w_rush_tds = wsum("rushing_tds")

        return {
            "avg_targets": w_targets / sw if sw > 0 else 0.0,
            "avg_carries": w_carries / sw if sw > 0 else 0.0,
            "avg_tds": (w_rec_tds + w_rush_tds) / sw if sw > 0 else 0.0,
            "avg_rec_tds": w_rec_tds / sw if sw > 0 else 0.0,
            "avg_rush_tds": w_rush_tds / sw if sw > 0 else 0.0,
            "ypt": w_rec_yds / w_targets if w_targets > 0 else 0.0,
            "ypc": w_rush_yds / w_carries if w_carries > 0 else 0.0,
            "ypr": w_rec_yds / w_rec if w_rec > 0 else 0.0,
            "catch_rate": w_rec / w_targets if w_targets > 0 else 0.0,
        }

    # --- 3. Starting QB (Ruling C1): pick QB1, drop other QBs ---
    qb_ids = [g for g, info in active.items() if info["pos"] == "QB"]
    qb1: str | None = None
    if qb_ids:
        def _qb_key(g: str) -> tuple:
            dt = active[g]["depth_team"]
            att = 0.0
            recent = (-1, -1)
            pdf = recent_by_pid.get(g)
            if pdf is not None and len(pdf) > 0:
                gg = pdf.sort_values(["season", "week"]).tail(n_recent)
                if "attempts" in gg.columns:
                    att = float(gg["attempts"].sum())
                last = gg.iloc[-1]
                recent = (int(last["season"]), int(last["week"]))
            # min depth_team, then most attempts, then most recent, then gsis.
            return (dt, -att, -recent[0], -recent[1], g)

        qb1 = min(qb_ids, key=_qb_key)
        for g in qb_ids:
            if g != qb1:
                del active[g]

    def _cold_prior(pos: str, is_starter: bool) -> dict:
        """The small positional pseudo-usage prior for a player with no usable
        recent usage (see `_COLD_*`). Used both for players with no tape at all
        and for rostered players whose recent window shows zero targets AND zero
        carries -- a depth/special-teams role still merits a nonzero floor
        rather than a hard zero ('no stats yet' != 'won't produce')."""
        ypt, ypc, ypr, catch_rate = _COLD_EFF.get(pos, (0.0, 0.0, 0.0, 0.0))
        cold_td = _COLD_TDS.get((pos, is_starter), 0.0)
        # Split the cold-start TD prior into receiving vs rushing by position:
        # WR/TE score through the air, RBs mostly on the ground, QBs rushing.
        rec_frac = {"WR": 1.0, "TE": 1.0, "RB": 0.25, "QB": 0.0}.get(pos, 0.0)
        return {
            "avg_targets": _COLD_TARGETS.get((pos, is_starter), 0.0),
            "avg_carries": _COLD_CARRIES.get((pos, is_starter), 0.0),
            "avg_tds": cold_td,
            "avg_rec_tds": cold_td * rec_frac,
            "avg_rush_tds": cold_td * (1.0 - rec_frac),
            "ypt": ypt, "ypc": ypc, "ypr": ypr, "catch_rate": catch_rate,
        }

    # --- 4. Per-player metrics (recent tape or cold-start prior) ---
    metrics: dict[str, dict] = {}
    for gsis, info in active.items():
        pos = info["pos"]
        is_starter = info["depth_team"] == 1
        pdf = recent_by_pid.get(gsis)
        w = _weighted(pdf) if (pdf is not None and len(pdf) > 0) else None
        # Fall back to the cold-start floor when there's no tape at all OR the
        # recent window carried zero targets and zero carries (a rostered player
        # who simply didn't touch the ball recently still gets a small floor,
        # not a hard zero that renders 0-yards-across-the-board).
        if w is not None and (w["avg_targets"] > 0.0 or w["avg_carries"] > 0.0):
            metrics[gsis] = w
        else:
            metrics[gsis] = _cold_prior(pos, is_starter)

    # --- 5. Renormalize shares OVER THE ACTIVE SET ONLY ---
    tot_targets = sum(m["avg_targets"] for m in metrics.values())
    tot_carries = sum(m["avg_carries"] for m in metrics.values())
    tot_tds = sum(m["avg_tds"] for m in metrics.values())
    tot_rec_tds = sum(m.get("avg_rec_tds", 0.0) for m in metrics.values())
    tot_rush_tds = sum(m.get("avg_rush_tds", 0.0) for m in metrics.values())

    players: list[PlayerInput] = []
    for gsis, info in active.items():
        m = metrics[gsis]
        players.append(
            PlayerInput(
                player_id=gsis,
                name=info["name"],
                pos=info["pos"],
                target_share=m["avg_targets"] / tot_targets if tot_targets > 0 else 0.0,
                carry_share=m["avg_carries"] / tot_carries if tot_carries > 0 else 0.0,
                ypt=m["ypt"],
                ypc=m["ypc"],
                ypr=m["ypr"],
                catch_rate=m["catch_rate"],
                td_share=m["avg_tds"] / tot_tds if tot_tds > 0 else 0.0,
                rec_td_share=m.get("avg_rec_tds", 0.0) / tot_rec_tds if tot_rec_tds > 0 else 0.0,
                rush_td_share=m.get("avg_rush_tds", 0.0) / tot_rush_tds if tot_rush_tds > 0 else 0.0,
            )
        )

    return players, qb1


def _clean_codes(series) -> set[str]:
    """A pandas Series -> {stripped str}, dropping NaN/blank values."""
    out: set[str] = set()
    for v in series:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            out.add(s)
    return out


def abbrev_alignment(
    depth_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
    game_teams: set[str],
    known_teams: set[str],
) -> dict[str, list[str]]:
    """Sanity check: do the team-abbreviation conventions used by the
    depth-chart, injuries, and schedule/slate sources all fall inside
    `known_teams` (the canonical set `normalize_team` maps onto)? PURE.

    A `depth_df["club_code"]` or `injuries_df["team"]` value NOT in
    `known_teams` means `active_usage`'s depth-chart lookup (or the caller's
    per-team out-names dict) can never match a game's normalized team code,
    in which case `active_usage` silently returns `([], None)` for that team
    -- no exception, no warning -- rather than the mismatch being loud. Both
    `scripts/backtest_sim_nfl.py`'s `n_empty_active` counter and
    `scripts/generate_sim_nfl.py`'s empty-roster count are the runtime
    symptom of exactly this; this helper is the "why" diagnostic, run once up
    front against the whole fetched span/slate rather than discovered
    game-by-game.

    Returns `{"depth_unknown": [...], "injuries_unknown": [...],
    "games_unknown": [...]}` (each sorted), where "games_unknown" are
    `game_teams` entries (normalized/crosswalked team codes) NOT in
    `known_teams` -- normally empty, since `known_teams` should already be
    `normalize_team`'s own codomain, but included for parity/defensiveness.
    NaN/blank codes in either DataFrame column are dropped, not flagged
    (they're absent data, not a naming mismatch).
    """
    depth_codes = _clean_codes(depth_df["club_code"]) if "club_code" in depth_df.columns else set()
    injuries_codes = _clean_codes(injuries_df["team"]) if "team" in injuries_df.columns else set()

    return {
        "depth_unknown": sorted(depth_codes - known_teams),
        "injuries_unknown": sorted(injuries_codes - known_teams),
        "games_unknown": sorted(set(game_teams) - known_teams),
    }


def fetch_usage_sources(seasons: list[int]) -> dict:
    """Thin IO wrapper around nfl_data_py imports. Not unit-tested.

    Returns {"depth": DataFrame, "snaps": DataFrame, "ids": DataFrame}.
    """
    import nfl_data_py as nfl

    from sportsmodel.nfl.nflverse import import_by_season

    return {
        "depth": import_by_season(nfl.import_depth_charts, seasons, "depth"),
        "snaps": import_by_season(nfl.import_snap_counts, seasons, "snaps"),
        "ids": nfl.import_ids(),
    }
