"""NFL injuries from nflverse's official game-day injury report.

SportsDataIO's free-tier NFL `InjuredPlayers` feed returns SCRAMBLED injury
status, so the decision desk pulls NFL injuries from nflverse instead
(`nfl_data_py.import_injuries`), which mirrors the official NFL injury report:
`report_status` is Out/Doubtful/Questionable and `report_primary_injury` is the
body part. This module parses that into the same `{player, position, status,
note}` shape the CFB SportsDataIO adapter produces (see
`sportsmodel.cfb.sportsdata.parse_injuries`), keyed by nflverse team
ABBREVIATION (e.g. "ARI", "PHI", "LA") -- the same abbreviations used as keys in
`assets/nfl/nfl_teams.json`, so `scripts/desk_inputs.py` can rekey it onto ESPN
display names through the crosswalk it already loads for form.

The parse is PURE (`parse_injuries`); `current_injuries` is the thin IO wrapper
that fetches the season's report. nflverse needs no API key.
"""
from __future__ import annotations

from datetime import datetime

# nflverse `report_status` values that represent a real injury-report
# designation. nflverse also carries blank/NaN-status rows (roster context,
# inactives lists) that are NOT part of the injury report -- those are dropped.
_REPORT_STATUSES = {"Out", "Doubtful", "Questionable"}


def nfl_season(now: datetime) -> int:
    """NFL season year for `now`.

    The season spans Sept through Feb, so January/February belong to the prior
    calendar year's season; March onward is the new season (injury reporting
    begins with summer training camp / preseason)."""
    return now.year if now.month >= 3 else now.year - 1


def _clean(x) -> str | None:
    """NaN / None / blank -> None; otherwise the stripped string."""
    import pandas as pd

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    return s or None


def parse_injuries(df, *, week: int | None = None) -> dict[str, list[dict]]:
    """nflverse injuries DataFrame -> `{team_abbrev -> [injury dict]}`. PURE.

    Keeps only rows whose `report_status` is a real designation
    (Out/Doubtful/Questionable). When `week` is None, only the latest week that
    has any such reported row is used -- i.e. the current injury report; pass an
    explicit `week` to pin a specific week.

    Each kept row becomes `{player, position, status, note}`, matching the CFB
    adapter's shape:
      - "player": `full_name`, stripped.
      - "position": `position`.
      - "status": `report_status` (Out/Doubtful/Questionable).
      - "note": `report_primary_injury`, `report_secondary_injury` and
        `practice_status` joined with " - ", skipping blanks; None when all are
        blank.

    An empty or None DataFrame (or one with no reported rows) yields `{}`.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return {}
    reported = df[df["report_status"].isin(_REPORT_STATUSES)]
    if len(reported) == 0:
        return {}
    if week is None:
        week = int(reported["week"].max())
    reported = reported[reported["week"] == week]

    out: dict[str, list[dict]] = {}
    for row in reported.itertuples(index=False):
        team = _clean(getattr(row, "team", None))
        if team is None:
            continue
        player = str(getattr(row, "full_name", "") or "").strip()
        parts = [
            _clean(getattr(row, "report_primary_injury", None)),
            _clean(getattr(row, "report_secondary_injury", None)),
            _clean(getattr(row, "practice_status", None)),
        ]
        parts = [p for p in parts if p]
        note = " - ".join(parts) if parts else None
        out.setdefault(team, []).append({
            "player": player,
            "position": _clean(getattr(row, "position", None)),
            "status": _clean(getattr(row, "report_status", None)),
            "note": note,
        })
    return out


def current_injuries(now: datetime) -> dict[str, list[dict]]:
    """Fetch + parse the current NFL season's latest injury report. IO.

    Returns `{team_abbrev -> [injury dict]}` for the most recent reported week.
    """
    import nfl_data_py as nfl

    df = nfl.import_injuries([nfl_season(now)])
    return parse_injuries(df)
