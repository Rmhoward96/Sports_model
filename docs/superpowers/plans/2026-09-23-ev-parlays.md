# +EV Parlays Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Up to 3 three-leg parlays per build from the ≥55%-probability, highest-EV current +EV picks (game lines + NFL props), each priced at the best single book, shown on the +EV page ("Parlays") and tracked at $10/ticket.

**Architecture:** A pure module `sportsmodel/serving/best_parlays.py` (leg assembly, ticket building, leg/ticket settlement) is driven by two thin scripts: `build_best_parlays.py` (after the full prop-board runs) and `grade_best_parlays.py` (in the grading jobs). Tickets live in `ev_best_parlays`; graded tickets in `ev_best_parlay_results`; `ev_pnl` gains parlay rows. CappingAlpha reads both.

**Tech Stack:** Python 3 / uv / pytest, Supabase Postgres (psycopg), GitHub Actions, vanilla JS (`/Users/ryan/Desktop/CappingAlpha/app.js`, not a git repo).

**Spec:** `docs/superpowers/specs/2026-09-23-ev-parlays-design.md`

## Global Constraints

- The user runs every Supabase migration; never apply DDL.
- Secrets live in GitHub Actions; never print them.
- Branch `feat/ev-parlays`; merge/push only when the user asks.
- Leg win probability floor 0.55; exactly 3 legs; up to 3 tickets; one leg per game; no shared legs across tickets; ticket EV must satisfy `0.01 < ev <= 1.0`.
- Books: `serving.board.MAJOR_BOOKS` keys except `pinnacle`; a book's price counts only at the pick's line (moneyline: no line) and only if the leg is +EV at that price.
- Stake $10. Loss −10; push legs drop out; win pays `10 × (Π decimal of winning legs − 1)`; all-push = 0.
- +EV page section title: exactly `Parlays`.
- Run tests with `uv run pytest`.

---

### Task 1: Pure leg assembly + ticket builder

**Files:**
- Create: `src/sportsmodel/serving/best_parlays.py`
- Modify: `src/sportsmodel/serving/props_ev.py` (move `latest_capture_only` here), `scripts/build_ev_props_board.py` (import it from props_ev; remove local copy)
- Test: `tests/serving/test_best_parlays.py`

**Produces (later tasks rely on these exact names):**
- `MIN_LEG_PROB = 0.55`, `N_LEGS = 3`, `MAX_TICKETS = 3`, `MIN_EV = 0.01`, `EV_CEILING = 1.0`, `PARLAY_BOOKS` (frozenset of MAJOR_BOOKS keys minus "pinnacle")
- `assemble_game_legs(picks: list[dict], odds_rows: list[dict]) -> list[dict]`
- `assemble_prop_legs(picks: list[dict], odds_rows: list[dict]) -> list[dict]`
- `build_best_parlays(legs: list[dict], excluded_keys: frozenset = frozenset()) -> list[dict]`
- Leg dict keys: `key, kind ("game"|"prop"), sport, game_pk, market, side, line, prob, label, matchup, commence_time, player_id, player_name, book_prices ({book: american})`
- Ticket dict keys: `parlay_id, book, legs (leg dicts WITHOUT book_prices, WITH price), n_legs, parlay_dec, parlay_price, true_prob, ev, sport, first_commence, last_commence`

- [ ] **Step 1: Move `latest_capture_only`** verbatim from `scripts/build_ev_props_board.py` into `props_ev.py` (below `normalize_player_name`), and in the script replace the definition with `from sportsmodel.serving.props_ev import latest_capture_only` (keep the name exported from the script module too, so `tests/test_build_ev_props_board.py` that calls `build_ev_props_board.latest_capture_only` still passes). Run `uv run pytest tests/test_build_ev_props_board.py -q` → PASS.

- [ ] **Step 2: Failing tests** — `tests/serving/test_best_parlays.py`:

```python
from sportsmodel.serving.best_parlays import (
    PARLAY_BOOKS, assemble_game_legs, assemble_prop_legs, build_best_parlays)


def _leg(key, game_pk, prob, prices, kind="game"):
    return {"key": key, "kind": kind, "sport": "nfl", "game_pk": game_pk, "market": "moneyline",
            "side": "home", "line": None, "prob": prob, "label": key, "matchup": "A @ B",
            "commence_time": f"2026-09-27T1{game_pk % 10}:00:00Z", "player_id": None,
            "player_name": None, "book_prices": prices}


def test_pinnacle_is_not_a_parlay_book():
    assert "pinnacle" not in PARLAY_BOOKS and "draftkings" in PARLAY_BOOKS


def test_builds_best_single_book_ticket_one_leg_per_game():
    legs = [_leg("a", 1, 0.60, {"draftkings": -120, "fanduel": -110}),
            _leg("a2", 1, 0.58, {"draftkings": -105}),            # same game as a
            _leg("b", 2, 0.62, {"draftkings": -130, "fanduel": -125}),
            _leg("c", 3, 0.57, {"draftkings": +100, "fanduel": -105}),
            _leg("d", 4, 0.56, {"fanduel": +105})]
    [t, *_] = build_best_parlays(legs)
    assert t["n_legs"] == 3 and len({l["game_pk"] for l in t["legs"]}) == 3
    assert t["book"] in ("draftkings", "fanduel")
    prices = {l["key"]: l["price"] for l in t["legs"]}
    assert all(prices[k] == next(x for x in legs if x["key"] == k)["book_prices"][t["book"]] for k in prices)
    dec = 1.0
    prob = 1.0
    for l in t["legs"]:
        dec *= (1 + (l["price"] / 100 if l["price"] > 0 else 100 / -l["price"]))
        prob *= l["prob"]
    assert abs(t["parlay_dec"] - dec) < 1e-9 and abs(t["true_prob"] - prob) < 1e-9
    assert abs(t["ev"] - (prob * dec - 1)) < 1e-9
    assert t["parlay_id"].startswith(t["book"] + "|")


def test_prob_floor_excluded_keys_and_no_shared_legs():
    legs = [_leg(k, g, 0.60, {"draftkings": +100}) for k, g in
            [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6), ("g", 7)]]
    legs.append(_leg("low", 8, 0.54, {"draftkings": +300}))       # under 55% floor
    tickets = build_best_parlays(legs, excluded_keys=frozenset({"g"}))
    used = [l["key"] for t in tickets for l in t["legs"]]
    assert len(tickets) == 2                                       # 6 usable legs -> 2 tickets
    assert len(used) == len(set(used)) and "low" not in used and "g" not in used


def test_leg_must_be_plus_ev_at_the_book_and_ticket_needs_three_legs():
    legs = [_leg("a", 1, 0.56, {"draftkings": -150}),              # 0.56*1.667-1 < 0
            _leg("b", 2, 0.60, {"draftkings": +100}),
            _leg("c", 3, 0.60, {"draftkings": +100})]
    assert build_best_parlays(legs) == []


def test_mixed_sport_ticket_labelled_mixed():
    legs = [_leg("a", 1, 0.6, {"draftkings": +100}), _leg("b", 2, 0.6, {"draftkings": +100}),
            {**_leg("c", 3, 0.6, {"draftkings": +100}), "sport": "cfb"}]
    [t] = build_best_parlays(legs)
    assert t["sport"] == "mixed"


def test_assemble_game_legs_matches_line_and_skips_pinnacle():
    picks = [{"sport": "nfl", "game_pk": 1, "market": "spread", "side": "home",
              "matchup": "Falcons @ Packers", "commence_time": "t", "true_prob": 0.57, "line": -3.5}]
    odds = [{"game_pk": 1, "market": "spread", "side": "home", "book": "draftkings", "line": -3.5, "price": -105},
            {"game_pk": 1, "market": "spread", "side": "home", "book": "fanduel", "line": -3.0, "price": -120},
            {"game_pk": 1, "market": "spread", "side": "home", "book": "pinnacle", "line": -3.5, "price": +100}]
    [leg] = assemble_game_legs(picks, odds)
    assert leg["book_prices"] == {"draftkings": -105}
    assert leg["label"] == "Packers -3.5" and leg["key"] == "g:1:spread:home" and leg["prob"] == 0.57


def test_assemble_game_legs_moneyline_and_total_labels():
    picks = [{"sport": "cfb", "game_pk": 2, "market": "moneyline", "side": "away", "matchup": "Oregon Ducks @ USC Trojans",
              "commence_time": "t", "true_prob": 0.6, "line": None},
             {"sport": "cfb", "game_pk": 3, "market": "total", "side": "under", "matchup": "X @ Y",
              "commence_time": "t", "true_prob": 0.55, "line": 49.5}]
    odds = [{"game_pk": 2, "market": "moneyline", "side": "away", "book": "fanduel", "line": None, "price": -139},
            {"game_pk": 3, "market": "total", "side": "under", "book": "betmgm", "line": 49.5, "price": -104}]
    legs = assemble_game_legs(picks, odds)
    assert [l["label"] for l in legs] == ["Oregon Ducks ML", "Under 49.5"]


def test_assemble_prop_legs_matches_player_market_side_line():
    picks = [{"game_pk": 9, "player_id": "00-1", "player_name": "A.J. Brown", "market": "rec_yds",
              "side": "under", "line": 66.5, "model_prob": 0.61, "matchup": "M", "commence_time": "t"}]
    odds = [{"game_pk": 9, "market": "reception_yds", "side": "under", "player_name": "AJ Brown", "book": "fanduel", "line": 66.5, "price": -112},
            {"game_pk": 9, "market": "reception_yds", "side": "under", "player_name": "AJ Brown", "book": "draftkings", "line": 64.5, "price": -110},
            {"game_pk": 9, "market": "reception_yds", "side": "over", "player_name": "AJ Brown", "book": "fanduel", "line": 66.5, "price": -108}]
    [leg] = assemble_prop_legs(picks, odds)
    assert leg["book_prices"] == {"fanduel": -112}
    assert leg["key"] == "p:9:00-1:rec_yds:under" and leg["kind"] == "prop" and leg["sport"] == "nfl"
    assert leg["label"] == "A.J. Brown Rec Yds Under 66.5"
```

- [ ] **Step 3:** `uv run pytest tests/serving/test_best_parlays.py -q` → FAIL (ImportError).

- [ ] **Step 4: Implement** `src/sportsmodel/serving/best_parlays.py`:

```python
"""+EV Parlays: up to 3 three-leg tickets built from the current +EV picks,
each priced at ONE book (a real parlay is one ticket at one book).

Legs: every current +EV pick -- NFL/CFB game lines and NFL player props -- with
win probability >= MIN_LEG_PROB, one per game within a ticket (independent legs,
so the ticket's true probability is the product). For each US book, take the
legs it prices at the pick's own line that are still +EV at that price, keep
the top N_LEGS by EV-at-book, and price the ticket at that book; the best-EV
book wins. Remove those legs and repeat, up to MAX_TICKETS tickets.

PURE -- no IO. scripts/build_best_parlays.py loads the inputs and persists.
"""
from __future__ import annotations

import math

from .board import MAJOR_BOOKS, decimal_odds
from .parlay import american_from_decimal
from .props_ev import SIM_TO_ODDS_MARKET, normalize_player_name

MIN_LEG_PROB = 0.55
N_LEGS = 3
MAX_TICKETS = 3
MIN_EV = 0.01
EV_CEILING = 1.0
PARLAY_BOOKS = frozenset(b for b in MAJOR_BOOKS if b != "pinnacle")

_PROP_LABEL = {"rush_yds": "Rush Yds", "rec_yds": "Rec Yds", "receptions": "Receptions",
               "pass_yds": "Pass Yds", "pass_tds": "Pass TDs", "rush_att": "Rush Att"}


def _fmt_line(x: float) -> str:
    return f"{x:g}"


def _game_label(market: str, side: str, matchup: str, line) -> str:
    away, _, home = (matchup or " @ ").partition(" @ ")
    if market == "total":
        return f"{'Over' if side == 'over' else 'Under'} {_fmt_line(line)}"
    team = home if side == "home" else away
    if market == "moneyline":
        return f"{team} ML"
    return f"{team} {'+' if line > 0 else ''}{_fmt_line(line)}" if line else f"{team} PK"


def assemble_game_legs(picks: list[dict], odds_rows: list[dict]) -> list[dict]:
    """Game-line +EV picks (ev_current rows with `line` from ev_best_lines) ->
    legs. A book's price counts only at the pick's line (moneyline: any)."""
    legs = []
    for p in picks:
        line = p.get("line")
        if p["market"] != "moneyline" and line is None:
            continue
        prices = {}
        for o in odds_rows:
            if (o["game_pk"] == p["game_pk"] and o["market"] == p["market"] and o["side"] == p["side"]
                    and o["book"] in PARLAY_BOOKS and o.get("price")
                    and (p["market"] == "moneyline" or o.get("line") == line)):
                prices[o["book"]] = o["price"]
        legs.append({
            "key": f"g:{p['game_pk']}:{p['market']}:{p['side']}", "kind": "game",
            "sport": p["sport"], "game_pk": p["game_pk"], "market": p["market"], "side": p["side"],
            "line": line, "prob": float(p["true_prob"]),
            "label": _game_label(p["market"], p["side"], p.get("matchup"), line),
            "matchup": p.get("matchup"), "commence_time": p.get("commence_time"),
            "player_id": None, "player_name": None, "book_prices": prices,
        })
    return legs


def assemble_prop_legs(picks: list[dict], odds_rows: list[dict]) -> list[dict]:
    """NFL prop +EV picks (ev_prop_picks_current) -> legs. A book's price counts
    only for the same player (normalized name), odds market, side and line."""
    legs = []
    for p in picks:
        odds_market = SIM_TO_ODDS_MARKET.get(p["market"])
        if odds_market is None:
            continue
        name = normalize_player_name(p.get("player_name"))
        prices = {}
        for o in odds_rows:
            if (o["game_pk"] == p["game_pk"] and o["market"] == odds_market and o["side"] == p["side"]
                    and o.get("line") == p["line"] and o["book"] in PARLAY_BOOKS and o.get("price")
                    and normalize_player_name(o.get("player_name")) == name):
                prices[o["book"]] = o["price"]
        legs.append({
            "key": f"p:{p['game_pk']}:{p['player_id']}:{p['market']}:{p['side']}", "kind": "prop",
            "sport": "nfl", "game_pk": p["game_pk"], "market": p["market"], "side": p["side"],
            "line": p["line"], "prob": float(p["model_prob"]),
            "label": f"{p.get('player_name')} {_PROP_LABEL.get(p['market'], p['market'])} "
                     f"{'Over' if p['side'] == 'over' else 'Under'} {_fmt_line(p['line'])}",
            "matchup": p.get("matchup"), "commence_time": p.get("commence_time"),
            "player_id": p.get("player_id"), "player_name": p.get("player_name"), "book_prices": prices,
        })
    return legs


def _ticket_at_book(book: str, legs: list[dict]) -> dict | None:
    best_by_game: dict = {}
    for leg in legs:
        price = leg["book_prices"].get(book)
        if price is None:
            continue
        ev = leg["prob"] * decimal_odds(price) - 1
        if ev <= 0:
            continue
        cur = best_by_game.get(leg["game_pk"])
        if cur is None or ev > cur[1]:
            best_by_game[leg["game_pk"]] = (leg, ev, price)
    chosen = sorted(best_by_game.values(), key=lambda t: (-t[1], t[0]["key"]))[:N_LEGS]
    if len(chosen) < N_LEGS:
        return None
    dec = math.prod(decimal_odds(price) for _, _, price in chosen)
    prob = math.prod(leg["prob"] for leg, _, _ in chosen)
    out_legs = [{**{k: v for k, v in leg.items() if k != "book_prices"}, "price": price}
                for leg, _, price in chosen]
    sports = {l["sport"] for l in out_legs}
    times = [l["commence_time"] for l in out_legs if l.get("commence_time") is not None]
    return {
        "parlay_id": book + "|" + "|".join(sorted(l["key"] for l in out_legs)),
        "book": book, "legs": out_legs, "n_legs": len(out_legs),
        "parlay_dec": dec, "parlay_price": american_from_decimal(dec),
        "true_prob": prob, "ev": prob * dec - 1,
        "sport": sports.pop() if len(sports) == 1 else "mixed",
        "first_commence": min(times) if times else None,
        "last_commence": max(times) if times else None,
    }


def build_best_parlays(legs: list[dict], excluded_keys: frozenset = frozenset()) -> list[dict]:
    """Up to MAX_TICKETS three-leg tickets, best-EV first, no shared legs.
    `excluded_keys`: leg keys already riding on a locked (started, ungraded)
    ticket -- never reused."""
    pool = [l for l in legs if l["prob"] >= MIN_LEG_PROB and l["key"] not in excluded_keys]
    tickets: list[dict] = []
    for _ in range(MAX_TICKETS):
        books = sorted({b for l in pool for b in l["book_prices"]})
        best = None
        for book in books:
            t = _ticket_at_book(book, pool)
            if t is not None and MIN_EV < t["ev"] <= EV_CEILING and (best is None or t["ev"] > best["ev"]):
                best = t
        if best is None:
            break
        tickets.append(best)
        used = {l["key"] for l in best["legs"]}
        pool = [l for l in pool if l["key"] not in used]
    return tickets
```

(If `test_builds_best_single_book_ticket_one_leg_per_game` shows the ticket's EV fails the `MIN_EV` floor with these fixture prices, adjust only fixture PRICES upward so the expected ticket is clearly +EV — never the thresholds.)

- [ ] **Step 5:** `uv run pytest tests/serving -q` then full suite → PASS.
- [ ] **Step 6: Commit** `feat(parlays): pure +EV parlay builder (one book, 3 legs, up to 3 tickets)`.

### Task 2: Pure settlement

**Files:** Modify `src/sportsmodel/serving/best_parlays.py`; Test `tests/serving/test_best_parlays.py` (append).

**Produces:** `game_leg_result(leg: dict, final: dict | None) -> str | None`, `prop_leg_result(leg: dict, actual: float | None, game_captured: bool) -> str | None`, `settle_parlay(legs: list[dict], results: list[str | None], stake: float = 10.0) -> dict | None` returning `{"result": "win"|"loss"|"push", "pnl": float, "payout_dec": float | None}`. Leg results are `"win"|"loss"|"push"` or `None` (pending); prop void → `"push"`.

- [ ] **Step 1: Failing tests** (append):

```python
from sportsmodel.serving.best_parlays import game_leg_result, prop_leg_result, settle_parlay


def _g(market, side, line=None):
    return {"kind": "game", "market": market, "side": side, "line": line}


def test_game_leg_results():
    final = {"actual_margin": 7, "actual_total": 44}           # home won by 7, 44 total
    assert game_leg_result(_g("moneyline", "home"), final) == "win"
    assert game_leg_result(_g("moneyline", "away"), final) == "loss"
    assert game_leg_result(_g("moneyline", "home"), {"actual_margin": 0, "actual_total": 40}) == "push"
    assert game_leg_result(_g("spread", "home", -7.0), final) == "push"
    assert game_leg_result(_g("spread", "home", -6.5), final) == "win"
    assert game_leg_result(_g("spread", "away", 6.5), final) == "loss"
    assert game_leg_result(_g("spread", "away", 7.5), final) == "win"
    assert game_leg_result(_g("total", "over", 43.5), final) == "win"
    assert game_leg_result(_g("total", "under", 44.0), final) == "push"
    assert game_leg_result(_g("total", "under", 43.5), None) is None   # not final yet


def test_prop_leg_results_and_void():
    leg = {"kind": "prop", "side": "under", "line": 66.5}
    assert prop_leg_result(leg, 50.0, True) == "win"
    assert prop_leg_result(leg, 70.0, True) == "loss"
    assert prop_leg_result({**leg, "side": "over", "line": 5.0}, 5.0, True) == "push"
    assert prop_leg_result(leg, None, True) == "push"        # didn't play -> void
    assert prop_leg_result(leg, None, False) is None         # game not captured yet


def test_settle_parlay():
    legs = [{"price": -110}, {"price": +100}, {"price": -120}]
    assert settle_parlay(legs, ["win", "loss", None]) == {"result": "loss", "pnl": -10.0, "payout_dec": None}
    assert settle_parlay(legs, ["win", "win", None]) is None
    win = settle_parlay(legs, ["win", "win", "win"])
    dec = (1 + 100 / 110) * 2.0 * (1 + 100 / 120)
    assert win["result"] == "win" and abs(win["pnl"] - 10 * (dec - 1)) < 1e-9
    one_push = settle_parlay(legs, ["win", "push", "win"])
    assert abs(one_push["payout_dec"] - (1 + 100 / 110) * (1 + 100 / 120)) < 1e-9
    assert settle_parlay(legs, ["push", "push", "push"]) == {"result": "push", "pnl": 0.0, "payout_dec": 1.0}
```

- [ ] **Step 2:** run → FAIL. **Step 3: Implement** (append to module):

```python
def _vs(x: float) -> str:
    return "win" if x > 0 else "loss" if x < 0 else "push"


def game_leg_result(leg: dict, final: dict | None) -> str | None:
    """Settle a game-line leg from a prediction_accuracy row (actual_margin =
    home - away, actual_total). None while the game isn't graded."""
    if not final or final.get("actual_margin") is None:
        return None
    margin = float(final["actual_margin"])
    side_margin = margin if leg["side"] == "home" else -margin
    if leg["market"] == "moneyline":
        return _vs(side_margin)
    if leg["market"] == "spread":
        return _vs(side_margin + float(leg["line"]))
    total = float(final["actual_total"])
    return _vs(total - float(leg["line"]) if leg["side"] == "over" else float(leg["line"]) - total)


def prop_leg_result(leg: dict, actual: float | None, game_captured: bool) -> str | None:
    """Settle a prop leg vs the player's actual. A player with no actual once the
    game's actuals are captured didn't record the stat line -> void (push)."""
    if actual is None:
        return "push" if game_captured else None
    diff = float(actual) - float(leg["line"])
    return _vs(diff if leg["side"] == "over" else -diff)


def settle_parlay(legs: list[dict], results: list[str | None], stake: float = 10.0) -> dict | None:
    """Ticket result from per-leg results: any loss -> loss now; any pending ->
    None; else pushes drop out and the win pays stake x (product of winning legs'
    decimals - 1); all pushes -> push."""
    if "loss" in results:
        return {"result": "loss", "pnl": -stake, "payout_dec": None}
    if any(r is None for r in results):
        return None
    dec = math.prod(decimal_odds(l["price"]) for l, r in zip(legs, results) if r == "win")
    if not any(r == "win" for r in results):
        return {"result": "push", "pnl": 0.0, "payout_dec": 1.0}
    return {"result": "win", "pnl": stake * (dec - 1), "payout_dec": dec}
```

- [ ] **Step 4:** tests → PASS; full suite. **Step 5: Commit** `feat(parlays): leg + ticket settlement`.

### Task 3: Migration + DB helpers

**Files:** Create `db/migration_ev_best_parlays.sql`; Modify `src/sportsmodel/db.py`; Test `tests/test_db_best_parlays.py` (FakeConn pattern from `tests/test_db_nfl_player_actuals.py`, extended with `execute`/`fetchall`).

**Produces:** `_BEST_PARLAY_COLS`, `replace_unlocked_best_parlays(tickets: list[dict]) -> int` (one transaction: `DELETE FROM ev_best_parlays WHERE first_commence > now()` then insert each ticket, `legs` as `json.dumps`), `locked_leg_keys() -> frozenset[str]`, `ungraded_locked_parlays() -> list[dict]`, `upsert_best_parlay_results(records: list[dict]) -> int`.

- [ ] **Step 1: Migration** `db/migration_ev_best_parlays.sql`:

```sql
-- +EV Parlays: up to 3 three-leg tickets per build, each at one book, and their
-- graded results. Idempotent. Run in the Supabase SQL Editor.
CREATE TABLE IF NOT EXISTS ev_best_parlays (
    parlay_id       TEXT PRIMARY KEY,         -- book|sorted leg keys
    sport           TEXT,                     -- nfl | cfb | mixed
    book            TEXT NOT NULL,
    legs            JSONB NOT NULL,           -- [{key,kind,sport,game_pk,market,side,line,prob,price,label,matchup,commence_time,player_id,player_name}]
    n_legs          INTEGER,
    parlay_dec      DOUBLE PRECISION,
    parlay_price    INTEGER,
    true_prob       DOUBLE PRECISION,
    ev              DOUBLE PRECISION,
    first_commence  TIMESTAMPTZ,              -- locks at the first leg's kickoff
    last_commence   TIMESTAMPTZ,
    model_version   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ev_best_parlays_first ON ev_best_parlays (first_commence);

CREATE TABLE IF NOT EXISTS ev_best_parlay_results (
    parlay_id       TEXT PRIMARY KEY,
    sport           TEXT,
    book            TEXT,
    parlay_price    INTEGER,
    first_commence  TIMESTAMPTZ,
    result          TEXT,                     -- win | loss | push
    pnl             DOUBLE PRECISION,         -- $10 stake
    payout_dec      DOUBLE PRECISION,
    legs            JSONB,                    -- ticket legs, each with "result"
    graded_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE OR REPLACE VIEW ev_best_parlays_current AS
  SELECT parlay_id, sport, book, legs, n_legs, parlay_dec, parlay_price, true_prob, ev,
         first_commence, last_commence, created_at
  FROM ev_best_parlays
  WHERE first_commence > now();

-- ev_pnl (migration_pnl_views.sql) + graded parlays, market 'parlay'.
CREATE OR REPLACE VIEW ev_pnl AS
  SELECT r.sport, r.game_pk, r.market, r.side, p.commence_time,
         CASE WHEN r.won IS NULL THEN 'push' WHEN r.won THEN 'win' ELSE 'loss' END AS result,
         CASE WHEN r.won IS NULL THEN 0.0
              WHEN r.won THEN 10 * (american_to_decimal(p.best_price) - 1)
              ELSE -10.0 END AS pnl
  FROM ev_results r
  JOIN LATERAL (
    SELECT commence_time, best_price FROM ev_picks e
    WHERE e.game_pk = r.game_pk AND e.market = r.market AND e.side = r.side AND e.is_pick
    ORDER BY e.created_at DESC LIMIT 1
  ) p ON true
  WHERE r.won IS NOT NULL OR p.best_price IS NOT NULL
  UNION ALL
  SELECT sport, game_pk, 'prop', side, commence_time,
         CASE result WHEN 'win' THEN 'win' WHEN 'loss' THEN 'loss' ELSE 'push' END,
         10 * profit
  FROM ev_prop_results
  WHERE result IS NOT NULL AND profit IS NOT NULL
  UNION ALL
  SELECT sport, NULL::bigint, 'parlay', NULL::text, first_commence, result, pnl
  FROM ev_best_parlay_results;

ALTER TABLE ev_best_parlays ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_best_parlays" ON ev_best_parlays;
CREATE POLICY "public read ev_best_parlays" ON ev_best_parlays FOR SELECT USING (true);
ALTER TABLE ev_best_parlay_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read ev_best_parlay_results" ON ev_best_parlay_results;
CREATE POLICY "public read ev_best_parlay_results" ON ev_best_parlay_results FOR SELECT USING (true);
GRANT SELECT ON ev_best_parlays, ev_best_parlays_current, ev_best_parlay_results, ev_pnl, ev_pnl_daily
  TO anon, authenticated;
```

Also update the `ev_pnl` definition in `db/migration_pnl_views.sql` to the same three-branch view but WITHOUT the parlay branch guarded — instead add a comment there: `-- ev_pnl is redefined with a parlay branch in migration_ev_best_parlays.sql; run that after this file.` (Do not add the parlay branch to migration_pnl_views.sql, since its tables may not exist yet.)

- [ ] **Step 2: Failing tests** for the four helpers with a FakeConn that records `execute(sql, params)` calls and returns canned `fetchall()` rows: `replace_unlocked_best_parlays([t])` issues the DELETE (containing `first_commence > now()`) before an INSERT into `ev_best_parlays` and commits, with `legs` JSON-encoded; `locked_leg_keys()` flattens `legs[*].key` from rows returned for a query containing `first_commence <= now()` and `NOT EXISTS`; `ungraded_locked_parlays()` returns dicts; `upsert_best_parlay_results` uses `ON CONFLICT (parlay_id) DO UPDATE` with `legs` JSON-encoded. Empty inputs return 0 without touching the DB where applicable.

- [ ] **Step 3: Implement** in `db.py` next to `upsert_ev_parlays`:

```python
_BEST_PARLAY_COLS = ["parlay_id", "sport", "book", "legs", "n_legs", "parlay_dec", "parlay_price",
                     "true_prob", "ev", "first_commence", "last_commence", "model_version"]


def replace_unlocked_best_parlays(tickets: list[dict]) -> int:
    """Swap in the current +EV parlay set: delete tickets whose first leg hasn't
    started (unlocked), insert `tickets`. Locked tickets (first leg started) are
    never touched -- they're what gets graded. One transaction."""
    import json
    placeholders = ", ".join(["%s"] * len(_BEST_PARLAY_COLS))
    sql = (f"INSERT INTO ev_best_parlays ({', '.join(_BEST_PARLAY_COLS)}) VALUES ({placeholders}) "
           f"ON CONFLICT (parlay_id) DO NOTHING")
    rows = [tuple(json.dumps(t[c], default=str) if c == "legs" else t.get(c) for c in _BEST_PARLAY_COLS)
            for t in tickets]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM ev_best_parlays WHERE first_commence > now()")
        if rows:
            cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def locked_leg_keys() -> frozenset:
    """Leg keys riding on locked (started) tickets that aren't graded yet."""
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT p.legs FROM ev_best_parlays p
            WHERE p.first_commence <= now()
              AND NOT EXISTS (SELECT 1 FROM ev_best_parlay_results r WHERE r.parlay_id = p.parlay_id)
        """)
        rows = cur.fetchall()
    keys = set()
    for (legs,) in rows:
        for leg in (legs if isinstance(legs, list) else __import__("json").loads(legs)):
            keys.add(leg["key"])
    return frozenset(keys)


def ungraded_locked_parlays() -> list[dict]:
    cols = ["parlay_id", "sport", "book", "legs", "parlay_price", "first_commence"]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT {', '.join('p.' + c for c in cols)} FROM ev_best_parlays p
            WHERE p.first_commence <= now()
              AND NOT EXISTS (SELECT 1 FROM ev_best_parlay_results r WHERE r.parlay_id = p.parlay_id)
        """)
        return [dict(zip(cols, r)) for r in cur.fetchall()]


_BEST_PARLAY_RESULT_COLS = ["parlay_id", "sport", "book", "parlay_price", "first_commence",
                            "result", "pnl", "payout_dec", "legs"]


def upsert_best_parlay_results(records: list[dict]) -> int:
    import json
    if not records:
        return 0
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _BEST_PARLAY_RESULT_COLS if c != "parlay_id")
    placeholders = ", ".join(["%s"] * len(_BEST_PARLAY_RESULT_COLS))
    sql = (f"INSERT INTO ev_best_parlay_results ({', '.join(_BEST_PARLAY_RESULT_COLS)}) "
           f"VALUES ({placeholders}) ON CONFLICT (parlay_id) DO UPDATE SET {updates}, graded_at = now()")
    rows = [tuple(json.dumps(r[c], default=str) if c == "legs" else r.get(c) for c in _BEST_PARLAY_RESULT_COLS)
            for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
```

(Use a module-level `import json` if `db.py` already imports it; don't use `__import__` — shown only to keep the snippet self-contained.)

- [ ] **Step 4:** tests → PASS; full suite. **Step 5: Commit** `feat(parlays): ev_best_parlays tables, views, ev_pnl parlay rows + db helpers`.

### Task 4: Builder script + workflow

**Files:** Create `scripts/build_best_parlays.py`; Modify `.github/workflows/build-ev-props.yml`; Test `tests/test_build_best_parlays.py` (importlib-load the script; test only the pure `ticket_summary` helper and that `main` short-circuits on no legs with loaders monkeypatched).

- [ ] **Step 1:** Script:

```python
"""Build the +EV Parlays: up to 3 three-leg tickets at one book each, from the
current +EV game-line picks (ev_current) and NFL prop picks
(ev_prop_picks_current). Replaces the unlocked tickets; locked tickets (first
leg started) are left for grading. See serving/best_parlays.py.

Usage: DATABASE_URL=... uv run python scripts/build_best_parlays.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import get_postgres, locked_leg_keys, replace_unlocked_best_parlays
from sportsmodel.serving.best_parlays import (
    assemble_game_legs, assemble_prop_legs, build_best_parlays)
from sportsmodel.serving.props_ev import SIM_TO_ODDS_MARKET, latest_capture_only

MODEL_VERSION = "ev-parlays-v1"


def _q(sql: str, params: list | None = None) -> list[dict]:
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def load_game_picks() -> list[dict]:
    return _q("""
        SELECT e.sport, e.game_pk, e.market, e.side, e.matchup, e.commence_time, e.true_prob, b.line
        FROM ev_current e
        LEFT JOIN ev_best_lines b USING (sport, game_pk, market, side)
        WHERE e.is_pick
    """)


def load_game_odds(game_pks: list[int]) -> list[dict]:
    if not game_pks:
        return []
    return _q("""
        SELECT DISTINCT ON (game_pk, market, side, book) game_pk, market, side, book, line, price
        FROM odds_snapshot
        WHERE game_pk = ANY(%s) AND market IN ('moneyline','spread','total')
          AND COALESCE(player_name, '') = '' AND captured_at <= commence_time
        ORDER BY game_pk, market, side, book, captured_at DESC
    """, [game_pks])


def load_prop_picks() -> list[dict]:
    return _q("""
        SELECT game_pk, player_id, player_name, market, side, line, model_prob, matchup, commence_time
        FROM ev_prop_picks_current WHERE is_pick
    """)


def load_prop_odds(game_pks: list[int]) -> list[dict]:
    if not game_pks:
        return []
    rows = _q("""
        SELECT game_pk, market, side, player_name, book, line, price, captured_at
        FROM odds_snapshot
        WHERE game_pk = ANY(%s) AND market = ANY(%s) AND captured_at <= commence_time
    """, [game_pks, sorted(set(SIM_TO_ODDS_MARKET.values()))])
    return latest_capture_only(rows)


def ticket_summary(t: dict) -> str:
    return f"{t['book']} {t['parlay_price']:+d} ev={t['ev']:.3f} :: " + " / ".join(l["label"] for l in t["legs"])


def main() -> None:
    game_picks, prop_picks = load_game_picks(), load_prop_picks()
    legs = (assemble_game_legs(game_picks, load_game_odds(sorted({p["game_pk"] for p in game_picks})))
            + assemble_prop_legs(prop_picks, load_prop_odds(sorted({p["game_pk"] for p in prop_picks}))))
    tickets = build_best_parlays(legs, excluded_keys=locked_leg_keys())
    for t in tickets:
        t["model_version"] = MODEL_VERSION
    n = replace_unlocked_best_parlays(tickets)
    priced = sum(1 for l in legs if l["book_prices"])
    print(f"[build_best_parlays] legs={len(legs)} priced={priced} tickets={n}")
    for t in tickets:
        print("  " + ticket_summary(t))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** Workflow — append to `build-ev-props.yml` a step that runs only on full (non-lines-only) runs:

```yaml
      - name: Build +EV parlays
        if: ${{ !(github.event.schedule == '35 15 * * *' || github.event.schedule == '35 16,18 * * 0,1,4,6' || github.event.inputs.lines_only == 'true') }}
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/build_best_parlays.py
```

- [ ] **Step 3:** tests (pure `ticket_summary`; `main` with all loaders + `locked_leg_keys` + `replace_unlocked_best_parlays` monkeypatched, asserting replace is called with `[]` when there are no legs) → PASS; YAML parses (`uv run --with pyyaml python -c ...`); full suite. **Step 4: Commit** `feat(parlays): build_best_parlays script + workflow step`.

### Task 5: Grader script + workflows

**Files:** Create `scripts/grade_best_parlays.py`; Modify `.github/workflows/grade-ev-props.yml` and `.github/workflows/grade-predictions.yml` (append a step each); Test `tests/test_grade_best_parlays.py`.

- [ ] **Step 1:** Script with a PURE `grade_ticket(ticket, finals: dict[int, dict], actuals: dict[tuple, float], captured_games: set[int]) -> dict | None` (per leg: `game_leg_result(leg, finals.get(leg["game_pk"]))` or `prop_leg_result(leg, actuals.get((leg["game_pk"], leg["player_id"], leg["market"])), leg["game_pk"] in captured_games)`; then `settle_parlay`; returns a results row `{parlay_id, sport, book, parlay_price, first_commence, result, pnl, payout_dec, legs: [leg + {"result": r}]}` or None if pending), and an IO `main()` that loads `ungraded_locked_parlays()`, finals from `prediction_accuracy` (`SELECT game_pk, actual_margin, actual_total FROM prediction_accuracy WHERE game_pk = ANY(%s)`), actuals from `nfl_player_actuals` for the prop legs' game_pks, and upserts results. `legs` from the DB may be a JSON string — decode.
- [ ] **Step 2:** Tests for `grade_ticket`: pending → None; one leg lost while others pending → loss row; all win → win row with pnl; prop void (captured game, no actual) → push leg dropped from the payout.
- [ ] **Step 3:** Workflow steps (after the existing grading step in each file):

```yaml
      - name: Grade +EV parlays
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/grade_best_parlays.py
```

- [ ] **Step 4:** tests → PASS; YAML parses; full suite. **Step 5: Commit** `feat(parlays): grade_best_parlays + grading workflow steps`.

### Task 6: Front-end (CappingAlpha)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`, all `*.html` → `app.js?v=20260923f`.

- [ ] **+EV page** (`buildEv`): fetch `ev_best_parlays_current?select=*&order=ev.desc` (`.catch(() => [])`); new section titled exactly `Parlays` placed after the header cards/chips and before "Game lines", with `data-evsec="parlays"` (visible for All; hidden for NFL/CFB/Player props filters — update `wireEvPage`). One card per ticket: header `{parlay_price as American} @ {book name}` + `EV {evSigned(ev)} · win {evPctVal(true_prob)}` + `$10 pays ${(10*(parlay_dec-1)).toFixed(2)}`; then each leg: label (game legs get `logoImg` of the team when a team side), price at that book (`evPrice`), prob, matchup + kickoff (`timeET`). Empty state: `No +EV parlays right now — needs 3 legs (≥55% win probability, one per game) that are all +EV at the same book.` Add `+EV PARLAYS` count to the header stat cards (replace nothing; add a 5th card only if the compact-stats grid supports it, else fold into the TOP EV card subtext).
- [ ] **Track Record** +EV profit tracker: add `["parlay", "+EV PARLAYS"]` to its markets list; add a "Graded parlays" list in the +EV tab after the profit tracker reading `ev_best_parlay_results?order=first_commence.desc&limit=50` — each: book, price, result badge, $ result, legs with ✓/✗/P per leg. Empty state when none.
- [ ] CSS for `.parlay-card` (reuse `.pg-game` look), `node --check`, bump cache version, and a browser check with stubbed `sb` for the two new endpoints (real data won't exist until the migration + first build).
