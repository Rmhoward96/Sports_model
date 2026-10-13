"""Adapters: what each pick writer already holds -> candidate ledger observations. PURE -- no IO.

Every adapter exists in two forms:

  `<name>_candidates(...)`    a list of ZERO-ARGUMENT CALLABLES, each building one `core.observation` dict. The pick
                              writers hand these to `record.safe_record`, which calls them one by one: a candidate
                              that raises is skipped (and counted) instead of dropping the whole batch.
  `<name>_observations(...)`  the same candidates, already built (bad ones skipped) -- plain dicts, for tests and
                              callers that want the data.

Nothing here changes a pick, a probability or a price: it only copies what the writer computed.
"""
from __future__ import annotations

from ..model.distributions import pick_dist, spread_pick_prob, total_pick_prob
from ..serving.board import decimal_odds
from . import core


def _pct(x):
    return None if x is None else float(x) * 100.0


def build_all(candidates: list) -> list[dict]:
    """Call every candidate callable; one that raises is skipped (same rule as `record.safe_record`)."""
    out = []
    for c in candidates:
        try:
            out.append(c() if callable(c) else c)
        except Exception:  # noqa: BLE001 -- a bad candidate is skipped, never fatal
            continue
    return out


# ---- +EV game board -------------------------------------------------------------------------------------

def ev_game_line(game: dict, market: str, side: str) -> float | None:
    """The SIDE's own line on the +EV game board: home spread (negated for away), the total, None for moneyline.
    Read from the Pinnacle main line the board prices against (`assemble_games` output)."""
    pin = (game or {}).get("pinnacle") or {}
    if market == "spread":
        hl = (pin.get("spread") or {}).get("home_line")
        return None if hl is None else (hl if side == "home" else -hl)
    if market == "total":
        return (pin.get("total") or {}).get("line")
    return None


def ev_game_candidates(picks: list[dict], games: list[dict]) -> list:
    """+EV game-line picks (`ev_rows_for_game` rows with is_pick) -> ev_game candidates. The reference is the
    board's own Pinnacle no-vig `base_prob`; the board carries no model probability. A spread / total pick whose
    line cannot be read is skipped (it could not be graded)."""
    by_pk = {g.get("game_pk"): g for g in games}
    out = []
    for r in picks:
        try:
            line = ev_game_line(by_pk.get(r.get("game_pk")), r.get("market"), r.get("side"))
        except (TypeError, AttributeError):          # a malformed pinnacle line: skip this pick only
            continue
        if r.get("market") in ("spread", "total") and line is None:
            continue
        base = r.get("base_prob")
        out.append(lambda r=r, line=line, base=base: core.observation(
            source="ev_game", sport=r.get("sport"), game_pk=r.get("game_pk"), market=r.get("market"),
            side=r.get("side"), commence_time=r.get("commence_time"), line=line,
            model_version=r.get("model_version"), best_price=r.get("best_price"), best_book=r.get("best_book"),
            ref_prob=base, ref_kind="pinnacle_novig" if base is not None else None,
            ref_books=1 if base is not None else None, ev_pct=_pct(r.get("ev_best"))))
    return out


def ev_game_observations(picks: list[dict], games: list[dict]) -> list[dict]:
    return build_all(ev_game_candidates(picks, games))


# ---- +EV player-prop board ------------------------------------------------------------------------------

def _prop_slot(r: dict) -> tuple:
    """The pick-key slot of a prop row: the line is deliberately NOT part of it."""
    pid = r.get("player_id")
    return (r.get("sport"), r.get("game_pk"), r.get("market"), r.get("side"), None if pid is None else str(pid))


def ev_prop_candidates(rows: list[dict]) -> list:
    """+EV prop picks (`assemble_prop_rows` rows with is_pick) -> ev_prop candidates.

    ONE candidate per pick key: the line the board actually shows. `line` is not part of the key, so alternate
    lines of the same (game, player, market, side) would otherwise be several observations of one pick, each
    flipping the append rule's 'line changed' trigger. `assemble_prop_rows` already emits only the main line;
    if the rows ever hold two lines for one slot the LAST one wins (the order the board upserts them).

    `ref_prob` is the board's own `market_prob` (no-vig of the best over / best under prices); it is not a graded
    reference, so ref_kind stays NULL. Graded closes come from the close capture, never from this field."""
    chosen: dict[tuple, dict] = {}
    for r in rows:
        if r.get("line") is None or r.get("player_id") is None:
            continue
        chosen[_prop_slot(r)] = r
    return [lambda r=r: core.observation(
        source="ev_prop", sport=r.get("sport"), game_pk=r.get("game_pk"), market=r.get("market"),
        side=r.get("side"), commence_time=r.get("commence_time"), line=r.get("line"),
        player_id=r.get("player_id"), player_name=r.get("player_name"), model_version=r.get("model_version"),
        model_prob=r.get("model_prob"), best_price=r.get("best_price"), best_book=r.get("best_book"),
        ref_prob=r.get("market_prob"), ev_pct=_pct(r.get("ev_best"))) for r in chosen.values()]


def ev_prop_observations(rows: list[dict]) -> list[dict]:
    return build_all(ev_prop_candidates(rows))


# ---- parlay legs ----------------------------------------------------------------------------------------

def _leg_ev_pct(price, prob):
    return None if price is None or prob is None else (float(prob) * decimal_odds(price) - 1) * 100.0


def parlay_leg_candidates(tickets: list[dict]) -> list:
    """Each leg of each +EV parlay ticket -> a parlay_leg candidate at the ticket's book and the leg's price."""
    out = []
    for t in tickets:
        for leg in t.get("legs") or []:
            out.append(lambda t=t, leg=leg: core.observation(
                source="parlay_leg", sport=leg.get("sport"), game_pk=leg.get("game_pk"),
                market=leg.get("market"), side=leg.get("side"), commence_time=leg.get("commence_time"),
                line=leg.get("line"), player_id=leg.get("player_id"), player_name=leg.get("player_name"),
                parlay_id=t.get("parlay_id"), model_version=t.get("model_version"), model_prob=leg.get("prob"),
                best_price=leg.get("price"), best_book=t.get("book"),
                ev_pct=_leg_ev_pct(leg.get("price"), leg.get("prob"))))
    return out


def parlay_leg_observations(tickets: list[dict]) -> list[dict]:
    return build_all(parlay_leg_candidates(tickets))


# ---- the model's own game picks ---------------------------------------------------------------------------

def _margin_total(row: dict) -> tuple[float | None, float | None]:
    """(pred_margin, pred_total) the way scripts/grade_predictions.py `_accuracy_row` derives them."""
    ph, pa = row.get("pred_home_score"), row.get("pred_away_score")
    if ph is not None and pa is not None:
        return ph - pa, ph + pa
    return row.get("pred_margin"), row.get("pred_total")


def model_game_candidates(rows: list[dict]) -> list:
    """`game_predictions` rows as the generators write them -> the model's game picks, using the SAME side logic
    as `prediction_accuracy` (scripts/grade_predictions.py):

      moneyline  home when home_win_prob >= 0.5, else away;
      spread     edge = pred_margin + market_spread: > 0 home (line = market_spread), < 0 away
                 (line = -market_spread), == 0 no pick;
      total      Over when pred_total > market_total, Under when below, equal = no pick.

    model_prob is the model's probability for that side at that line (spread / total from the stored
    distribution, pushes excluded; NULL without one). No market price is attached: model picks are graded at
    flat -110 on spreads / totals and have no price on moneylines."""
    out = []
    for r in rows:
        try:
            out.extend(_model_game_row(r))
        except (TypeError, AttributeError):          # non-numeric prediction fields: skip this game only
            continue
    return out


def _model_game_row(r: dict) -> list:
    base = dict(source="model_game", sport=r.get("sport"), game_pk=r.get("game_pk"),
                commence_time=r.get("commence_time"), model_version=r.get("model_version"))
    out = []
    hwp = r.get("home_win_prob")
    if hwp is not None:
        ml_home = hwp >= 0.5
        out.append(lambda: core.observation(
            **base, market="moneyline", side="home" if ml_home else "away",
            model_prob=hwp if ml_home else 1.0 - hwp))
    margin, total = _margin_total(r)
    ms, mt = r.get("market_spread"), r.get("market_total")
    if margin is not None and ms is not None and margin + ms != 0:
        sp_home = margin + ms > 0
        out.append(lambda: core.observation(
            **base, market="spread", side="home" if sp_home else "away", line=ms if sp_home else -ms,
            model_prob=spread_pick_prob(pick_dist("margin_dist", r), margin, ms)))
    if total is not None and mt is not None and total != mt:
        out.append(lambda: core.observation(
            **base, market="total", side="over" if total > mt else "under", line=mt,
            model_prob=total_pick_prob(pick_dist("total_dist", r), total, mt)))
    return out


def model_game_observations(rows: list[dict]) -> list[dict]:
    return build_all(model_game_candidates(rows))
