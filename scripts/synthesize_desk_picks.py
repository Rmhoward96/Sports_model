"""Synthesize decision-desk picks from a bundle via the Anthropic API.

Server-side counterpart to the in-session desk. Given a desk bundle (produced
by `desk_inputs.py`), this calls Claude with the disciplined desk methodology
and emits a validated picks JSON list ready for `write_desk_picks.py`. This is
what lets the desk run **unattended in GitHub Actions** -- no live Claude app
required (see docs/desk-auto-runbook.md).

Design that keeps the model honest:
  - The model returns only judgment fields per game -- ml_pick, spread_side,
    total_side, conviction_tier, confidence, rationale, agent_notes -- keyed by
    game_pk. It never supplies numbers we can derive ourselves.
  - THIS SCRIPT fills every deterministic field from the bundle: the game
    identity (sport, game_pk, commence_time, matchup, model_version) and the
    spread/total LINES (from market_spread/market_total, with the correct sign
    per side). That eliminates any chance of a game_pk mismatch or a
    spread-line sign error coming from the model.
  - Totals are declined across the board (current desk policy): total_side and
    total_line are forced to null regardless of what the model returns.

Fail-closed: if the model output can't be parsed, or fails `validate_picks`
even after one repair round, the script writes nothing and exits non-zero.

Usage:
    ANTHROPIC_API_KEY=... uv run python scripts/synthesize_desk_picks.py \\
        --sport nfl --bundle desk_bundle.json --out picks.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # so `scripts.*` imports work when run as a file

from scripts.write_desk_picks import validate_picks  # noqa: E402  (contract)

DEFAULT_MODEL = "claude-opus-4-8"
_TIERS = {"high", "medium", "low"}
_ML_SIDES = {"home", "away"}
_SPREAD_SIDES = {"home", "away"}

SYSTEM_PROMPT = """\
You are the decision desk for a sports-betting model: three analysts
(statistics, sports-analyst, news) plus the desk head, synthesized into one
disciplined set of picks. You will receive a JSON "bundle" of upcoming games
and must return picks as a JSON array.

METHODOLOGY (load-bearing -- follow exactly):
- The ratings model does NOT beat the closing market (a backtest proved this).
  So `model.margin`/`model.total` are SUPPORTING CONTEXT ONLY -- never the sole
  reason for a lean. A large model-vs-line gap is mostly noise.
- Spread leans come from REAL edges: injuries (a key player Out/Doubtful at an
  impact position -- QB, top WR/RB, multiple starters in one unit) and
  early-season form (record + avg margin the market may underweight). Lean only
  where injuries or form give a genuine edge AND the model does not contradict
  it. Every spread lean's rationale MUST cite a concrete fact (a named injury, a
  specific record/avg-margin, or the model-vs-line numbers).
- Decline totals entirely: set "total_side" to null on every game.
- `ml_pick` is REQUIRED on every game: the more likely winner -- "home" if
  model.win_prob >= 0.5 else "away". It is independent of the spread lean (you
  can like the favorite to win but the dog to cover).
- `spread_side` is the cover lean ("home"/"away") ONLY where a real edge exists;
  otherwise null (decline -- most games decline).
- Tier honestly with "conviction_tier": mostly "low"/"medium". Reserve "high"
  for a strong multi-signal edge (injuries + form + model all aligned on a
  mispriced side). Having ZERO high picks in a slate is normal and correct.
  "confidence" is a number in [0,1]: ~0.52-0.55 low, ~0.56-0.60 medium,
  ~0.62-0.68 high.
- Sport notes: NFL injuries are the official report (real) but the market is
  sharp -- expect FEW, small leans; a report that hasn't posted yet means lean
  on model+form and say so, never invent injuries. CFB injuries are real; when
  fading an inflated number on a big favorite, do it only when the favorite's
  OWN form is shaky or the dog's form is strong -- never fade a rolling
  dominant favorite on model margin alone.

OUTPUT: return ONLY a JSON array (no prose, no markdown fences). One object per
game in the bundle, each EXACTLY:
{
  "game_pk": <int, copied from the bundle game>,
  "ml_pick": "home" | "away",
  "spread_side": "home" | "away" | null,
  "conviction_tier": "high" | "medium" | "low",
  "confidence": <number in [0,1]>,
  "rationale": "<prose citing at least one concrete fact>",
  "agent_notes": {"statistics": "...", "analyst": "...", "news": "..."}
}
Do NOT include spread_line, total_side, total_line, matchup, sport, or
commence_time -- those are filled in downstream. Return every game_pk from the
bundle exactly once.
"""


def _model_decisions(bundle: list[dict], model: str, max_tokens: int) -> list[dict]:
    """Call the Anthropic API and return the parsed decision array. Raises on
    an unparseable response."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
    user = (
        "Here is the games bundle. Synthesize the desk picks per the "
        "methodology and return only the JSON array.\n\n"
        + json.dumps(bundle, default=str)
    )
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return _parse_array(text)


def _parse_array(text: str) -> list[dict]:
    """Extract a JSON array from a model response, tolerating stray prose or
    ```json fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model response")
    return json.loads(text[start : end + 1])


def assemble_picks(bundle: list[dict], decisions: list[dict], sport: str) -> list[dict]:
    """Merge the model's per-game judgment with the bundle's authoritative
    identity + deterministic lines. PURE.

    Only games the model returned a decision for become picks. Deterministic
    fields (game_pk, commence_time, matchup, spread_line, total_side/line) come
    from the bundle/policy, never the model -- so a bad game_pk or a wrong
    spread-line sign is impossible.
    """
    by_pk = {g["game_pk"]: g for g in bundle}
    model_version = f"desk-{sport}-v1"
    picks: list[dict] = []
    for dec in decisions:
        g = by_pk.get(dec.get("game_pk"))
        if g is None:
            continue  # model hallucinated a game_pk -> drop it

        ml_pick = dec.get("ml_pick")
        if ml_pick not in _ML_SIDES:
            # fall back to the methodology's deterministic rule
            wp = (g.get("model") or {}).get("win_prob")
            ml_pick = "home" if (wp is not None and wp >= 0.5) else "away"

        spread_side = dec.get("spread_side")
        if spread_side not in _SPREAD_SIDES:
            spread_side = None
        mkt_s = g.get("market_spread")
        if spread_side is not None and mkt_s is not None:
            # spread_line is the HOME line; flip sign for the away side.
            spread_line = mkt_s if spread_side == "home" else -mkt_s
        else:
            spread_side, spread_line = None, None  # no line -> can't grade -> decline

        tier = dec.get("conviction_tier")
        if tier not in _TIERS:
            tier = "low"
        conf = dec.get("confidence")
        try:
            conf = min(max(float(conf), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.5

        picks.append({
            "sport": sport,
            "game_pk": g["game_pk"],
            "model_version": model_version,
            "commence_time": g["commence_time"],
            "matchup": g["matchup"],
            "ml_pick": ml_pick,
            "spread_side": spread_side,
            "spread_line": spread_line,
            "total_side": None,   # desk policy: decline totals across the board
            "total_line": None,
            "confidence": conf,
            "conviction_tier": tier,
            "rationale": dec.get("rationale") or "",
            "agent_notes": dec.get("agent_notes"),
        })
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize desk picks via the Anthropic API.")
    ap.add_argument("--sport", choices=["cfb", "nfl"], required=True)
    ap.add_argument("--bundle", type=Path, required=True, help="desk_inputs.py bundle JSON.")
    ap.add_argument("--out", type=Path, required=True, help="Where to write the picks JSON list.")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (add it as a GitHub Actions secret).")

    # `or DEFAULT_MODEL` (not .get's default) so an empty env var -- e.g. an
    # unset `${{ vars.DESK_SYNTH_MODEL }}` in CI, which expands to "" -- still
    # falls back rather than sending an empty model id.
    model = os.environ.get("DESK_SYNTH_MODEL") or DEFAULT_MODEL
    max_tokens = int(os.environ.get("DESK_SYNTH_MAX_TOKENS") or "16000")

    bundle = json.loads(args.bundle.read_text())
    if not bundle:
        print("Empty bundle -- writing empty picks list.")
        args.out.write_text("[]")
        return

    decisions = _model_decisions(bundle, model, max_tokens)
    picks = assemble_picks(bundle, decisions, args.sport)

    problems = validate_picks(picks)
    if problems:
        # One repair round: hand the model its own output + the exact problems.
        print(f"{len(problems)} validation problem(s); attempting one repair round.")
        import anthropic

        client = anthropic.Anthropic()
        repair = (
            "Your previous picks failed validation. Here are the problems:\n"
            + "\n".join(f"- {p}" for p in problems)
            + "\n\nHere was your output:\n"
            + json.dumps(decisions, default=str)
            + "\n\nReturn a corrected JSON array (same schema, only the "
            "judgment fields). Fix every problem."
        )
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": repair}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        decisions = _parse_array(text)
        picks = assemble_picks(bundle, decisions, args.sport)
        problems = validate_picks(picks)

    if problems:
        print(f"Still {len(problems)} problem(s) after repair; writing NOTHING:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    args.out.write_text(json.dumps(picks, indent=2))
    leans = sum(1 for p in picks if p["spread_side"])
    from collections import Counter
    tiers = dict(Counter(p["conviction_tier"] for p in picks if p["spread_side"]))
    print(f"Synthesized {len(picks)} picks ({leans} spread leans, tiers {tiers}) via {model} -> {args.out}")


if __name__ == "__main__":
    main()
