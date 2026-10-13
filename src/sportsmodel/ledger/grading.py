"""Grade ledger picks against results and the captured close. PURE -- no IO.

Every pick_key is graded twice into pick_ledger_results: basis 'first' (its FIRST ledger row = the earliest
bettable observation) and basis 'last' (its LAST row before kickoff). A pick is graded only once its result is
known; the close comes from `reference.close_reference` over the close snapshot of its game.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from . import reference as ref
from .core import as_utc


def basis_rows(rows: list[dict]) -> dict[str, dict]:
    """{'first': earliest, 'last': latest} among a key's rows observed BEFORE kickoff (a row at or after kickoff
    was never bettable). Empty when no row qualifies."""
    ok = []
    for r in rows:
        obs, kick = as_utc(r.get("observed_at")), as_utc(r.get("commence_time"))
        if obs is not None and kick is not None and obs < kick:
            ok.append((obs, r.get("id") or 0, r))
    if not ok:
        return {}
    ok.sort(key=lambda t: (t[0], t[1]))
    return {"first": ok[0][2], "last": ok[-1][2]}


def merge_close(primary: list[dict], fallback: list[dict]) -> dict[tuple, dict]:
    """{(game_pk, is_prop): {'rows': [...], 'captured_at': datetime}}. `primary` = snapshot rows stored by the
    close capture; `fallback` = the latest pre-kick snapshot within the stale cutoff, used only for a
    (game, kind) with no capture. Rows carry game_pk, player_name and captured_at."""
    out: dict[tuple, dict] = {}
    for rows in (primary, fallback):
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[(r["game_pk"], bool(r.get("player_name")))].append(r)
        for key, rs in grouped.items():
            if key not in out:
                out[key] = {"rows": rs, "captured_at": max(as_utc(r["captured_at"]) for r in rs)}
    return out


def grade_pick_row(row: dict, basis: str, finals: dict, actuals: dict, close: dict) -> dict | None:
    """One pick_ledger_results row for ledger `row`, or None while its result is unknown.

    finals: {(sport, game_pk): {'actual_margin', 'actual_total'}};
    actuals: {(sport, game_pk, player_id, market): float};
    close: merge_close output."""
    is_prop = row.get("player_id") is not None
    if is_prop:
        result = ref.settle_prop(row["side"], row.get("line"),
                                 actuals.get((row["sport"], row["game_pk"], str(row["player_id"]), row["market"])))
    else:
        f = finals.get((row["sport"], row["game_pk"])) or {}
        result = ref.settle_game(row["market"], row["side"], row.get("line"), f.get("actual_margin"),
                                 f.get("actual_total"))
    if result is None:
        return None

    is_model = row.get("source") == "model_game"
    # Model picks carry no market price: spread / total are graded at flat -110 (as the audit did); a model
    # moneyline has no price at all (units NULL). Their CLV never uses a price either.
    if is_model:
        price = ref.STANDARD_PRICE if row["market"] in ("spread", "total") else None
        clv_price = None
    else:
        price = clv_price = row.get("best_price")
    snap = close.get((row["game_pk"], is_prop))
    reference = ref.close_reference(row, snap["rows"]) if snap else \
        {"close_kind": "missing", "fair_prob": None, "close_line": None, "same_line": None, "books": 0}
    # `fair_prob` is, by contract, the no-vig probability AT THE PICK'S OWN LINE (None when no such price exists),
    # so it is comparable with the price taken whether or not the close line moved. `same_line` only says whether
    # close_line == the pick's line (a consensus close can report Pinnacle's moved line); it does not gate this.
    prob_clv = ref.clv_prob(reference["fair_prob"], clv_price)
    pts_clv = ref.clv_pts(row["market"], row["side"], row.get("line"), reference["close_line"])
    kick = as_utc(row["commence_time"])
    close_minutes = None
    if snap and reference["close_kind"] != "missing":
        close_minutes = (kick - snap["captured_at"]).total_seconds() / 60.0
    return {
        "pick_key": row["pick_key"], "basis": basis, "ledger_id": row["id"], "source": row["source"],
        "sport": row["sport"], "game_pk": row["game_pk"], "market": row["market"], "side": row["side"],
        "line": row.get("line"), "player_id": row.get("player_id"), "player_name": row.get("player_name"),
        "best_price": row.get("best_price"), "observed_at": row["observed_at"], "commence_time": row["commence_time"],
        "minutes_to_kick": row["minutes_to_kick"], "result": result,
        "units_at_pick": ref.units_at_pick(result, price),
        "close_kind": reference["close_kind"], "close_fair_prob": reference["fair_prob"],
        "close_line": reference["close_line"], "close_same_line": reference["same_line"],
        "close_books": reference["books"] or None, "close_minutes_to_kick": close_minutes,
        "clv_prob": prob_clv, "clv_pts": pts_clv, "beat_close": ref.beat_close(prob_clv, pts_clv),
    }


def grade_rows(ledger_rows: list[dict], finals: dict, actuals: dict, close: dict) -> list[dict]:
    """Grade every pick_key found in `ledger_rows` (all rows of those keys) on both bases."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in ledger_rows:
        by_key[r["pick_key"]].append(r)
    out = []
    for rows in by_key.values():
        for basis, r in basis_rows(rows).items():
            g = grade_pick_row(r, basis, finals, actuals, close)
            if g is not None:
                out.append(g)
    return out
