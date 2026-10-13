"""Pick-ledger core: the pick key, the observation record and the append rule. PURE -- no IO.

A ledger row is one OBSERVATION of a pick. The key names the bet slot (which game, market, side, player,
parlay); it deliberately excludes the price AND the line, so a price or number moving between rebuilds is one
pick observed twice, not two picks. `plan_appends` decides which candidate observations become new rows:
a key's first sighting, then again only when the best price changes, the line changes, or the model / reference
probability moves by 1.0 percentage point or more since that key's last row.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

SOURCES = ("model_game", "ev_game", "ev_prop", "parlay_leg")
REF_KINDS = ("pinnacle_novig", "us_consensus_novig")
PROB_MOVE = 0.01          # 1.0 percentage point
LINE_EPS = 1e-9


def as_utc(value) -> datetime | None:
    """A tz-aware UTC datetime from a datetime or ISO string (naive = UTC); None when missing / unparseable."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def pick_key(source: str, sport: str, game_pk, market: str, side: str, *, player_id=None,
             parlay_id: str | None = None, model_version: str | None = None) -> str:
    """Deterministic key of a bet slot. Fixed positions, never the price or the line.

    `model_version` is part of the key ONLY for model_game picks (two models can lean the same side of the same
    game; each is its own track record). parlay_id comes last because it may itself contain '|'."""
    if source not in SOURCES:
        raise ValueError(f"unknown ledger source: {source!r}")
    mv = (model_version or "") if source == "model_game" else ""
    return "|".join([source, str(sport), str(game_pk), str(market), str(side),
                     "" if player_id is None else str(player_id), mv, parlay_id or ""])


def _f(x):
    return None if x is None else float(x)


def observation(*, source: str, sport: str, game_pk, market: str, side: str, commence_time, line=None,
                player_id=None, player_name=None, parlay_id=None, model_version=None, model_prob=None,
                best_price=None, best_book=None, ref_prob=None, ref_kind=None, ref_books=None,
                ev_pct=None) -> dict:
    """A candidate observation (everything a writer knows). `observed_at`, `minutes_to_kick`, `writer` and
    `writer_run` are stamped by `plan_appends`."""
    if ref_kind is not None and ref_kind not in REF_KINDS:
        raise ValueError(f"unknown ref_kind: {ref_kind!r}")
    return {
        "pick_key": pick_key(source, sport, game_pk, market, side, player_id=player_id,
                             parlay_id=parlay_id, model_version=model_version),
        "source": source, "sport": sport, "game_pk": int(game_pk), "market": market, "side": side,
        "line": _f(line), "player_id": None if player_id is None else str(player_id),
        "player_name": player_name, "parlay_id": parlay_id, "model_version": model_version,
        "model_prob": _f(model_prob), "best_price": None if best_price is None else int(best_price),
        "best_book": best_book, "ref_prob": _f(ref_prob), "ref_kind": ref_kind,
        "ref_books": None if ref_books is None else int(ref_books), "ev_pct": _f(ev_pct),
        "commence_time": commence_time,
    }


def _line_changed(a, b) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(float(a) - float(b)) > LINE_EPS


def _prob_moved(a, b) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return round(abs(float(a) - float(b)), 9) >= PROB_MOVE      # round: 0.60 - 0.59 is 1.0pp, not 0.9999999pp


def change_reason(cand: dict, last: dict | None) -> str | None:
    """Why `cand` should be appended after `last` (the key's most recent row), or None to skip it."""
    if last is None:
        return "first"
    if cand.get("best_price") != last.get("best_price"):
        return "price"
    if _line_changed(cand.get("line"), last.get("line")):
        return "line"
    if _prob_moved(cand.get("model_prob"), last.get("model_prob")) or \
            _prob_moved(cand.get("ref_prob"), last.get("ref_prob")):
        return "prob"
    return None


def plan_appends(candidates: list[dict], last_by_key: dict[str, dict], *, now, writer: str,
                 writer_run: str) -> tuple[list[dict], dict]:
    """(rows to insert, counts). Duplicate keys within a batch keep the LAST candidate. A pick whose game has
    already started (commence_time <= now) is never recorded -- a post-kickoff observation is not bettable --
    and neither is one with no usable commence_time."""
    now = as_utc(now)
    counts: Counter = Counter()
    by_key = {c["pick_key"]: c for c in candidates}
    rows: list[dict] = []
    for key, cand in by_key.items():
        kick = as_utc(cand.get("commence_time"))
        if kick is None:
            counts["invalid"] += 1
            continue
        if kick <= now:
            counts["started"] += 1
            continue
        reason = change_reason(cand, last_by_key.get(key))
        if reason is None:
            counts["unchanged"] += 1
            continue
        counts[reason] += 1
        rows.append({**cand, "commence_time": kick, "observed_at": now,
                     "minutes_to_kick": (kick - now).total_seconds() / 60.0,
                     "writer": writer, "writer_run": writer_run})
    return rows, dict(counts)
