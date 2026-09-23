"""Team-name normalization shared by the trend captures (Action Network names ->
our ESPN display names). Same normalization as scripts/desk_inputs._norm_name."""
from __future__ import annotations

import unicodedata


def norm_team(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join("".join(c for c in s if c.isalnum() or c.isspace()).split())


def match_team_names(an_names: list[str], ours: list[str]) -> dict[str, str]:
    """{AN name -> our name} for exact normalized-name matches only."""
    by_norm = {norm_team(n): n for n in ours}
    return {a: by_norm[norm_team(a)] for a in an_names if norm_team(a) in by_norm}
