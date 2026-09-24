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
import re
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

HOME / AWAY -- READ CAREFULLY. Each game gives `home_team`, `away_team`, and
`model_pick_team` (the team the MODEL projects to win, derived from its
win_prob/margin). ALWAYS refer to teams by NAME. Do NOT reason in "home"/"away"
labels and do NOT try to re-derive which team is home -- it is given. `model`
values are home-referenced: `model.win_prob` is the HOME team's win prob and a
NEGATIVE `model.margin`/`market_spread` means the HOME team is favored.

YOUR JOB: the model has already made a pick (`model_pick_team`). For each game
you either CONFIRM that pick or OVERRIDE it to the other team, based on form +
injuries. It is common and correct to OVERRIDE when the model looks wrong (e.g.
the model favors a winless team over a strong one) -- when you do, your pick is
simply the other team. Say plainly in the rationale whether you confirm or
override, and which TEAM you are backing.

METHODOLOGY (load-bearing -- follow exactly):
- The ratings model does NOT beat the closing market (a backtest proved this).
  So `model.*` is SUPPORTING CONTEXT ONLY -- never the sole reason for a lean,
  and the model is sometimes simply wrong (override it then).
- Spread leans come from REAL edges: injuries (a key player Out/Doubtful at an
  impact position -- QB, top WR/RB, multiple starters in one unit) and
  early-season form (record + avg margin the market may underweight). Every
  spread lean's rationale MUST cite a concrete fact (a named injury, a specific
  straight-up record/avg-margin, or the model-vs-line numbers).
- Decline totals entirely: set "total_side" to null on every game.
- `ml_pick_team` is REQUIRED on every game: the exact team name (must equal this
  game's `home_team` or `away_team`) you think wins. Independent of the spread
  lean.
- `spread_pick_team` is the team you back against the spread ONLY where a real
  edge exists; otherwise null (decline -- most games decline). Must be an exact
  team name.
- Tier honestly with "conviction_tier": mostly "low"/"medium". Reserve "high"
  for a strong multi-signal edge (injuries + form + model all aligned on a
  mispriced side). Having ZERO high picks in a slate is normal and correct.
  "confidence" is a number in [0,1]: ~0.52-0.55 low, ~0.56-0.60 medium,
  ~0.62-0.68 high.
- Sport notes: NFL injuries are the official report (real) but the market is
  sharp -- expect FEW, small leans; a report that hasn't posted yet means lean
  on model+form and say so, never invent injuries (when `stale`, ESPN statuses
  replace the unposted report). CFB injuries are real; when
  fading an inflated number on a big favorite, do it only when the favorite's
  OWN form is shaky or the dog's form is strong -- never fade a rolling
  dominant favorite on model margin alone.
- TRENDS (`trends.home` / `trends.away`): season betting records (ATS, O/U,
  units -- overall, by venue, by favorite/underdog role, last 5) and, for NFL,
  situational trends ("7-2-0 ATS off a road game since 2023"). The SPORTS-ANALYST
  note (`agent_notes.analyst`) MUST cite every trend given for BOTH teams and say
  which way each points. Use a COMPACT citation format -- one short line per
  team, no prose restating each trend, e.g. "Trends: BUF ATS 2-1, ATS road 1-0,
  O/U 2-1; MIA ATS 0-3 as dog, 7-2 ATS off a road game since 2023 -- favors
  BUF". O/U trends are context only because totals are declined. Trends are
  SUPPORTING EVIDENCE: they may raise or lower
  conviction or tip a close call, but they are never the sole basis for a spread
  lean -- a lean still needs a concrete edge (injury, form, model-vs-line). Most
  ATS trends are small-sample noise; weigh lopsided, larger-sample ones more. If
  a trend influenced the call, say so in the rationale. If `trends` is null for a
  team, say trends were unavailable.
- INJURY FRESHNESS (`news.injury_report`): injury statuses are the CURRENT
  report -- (NFL) verified against ESPN's live list. Never state a player is out,
  doubtful or questionable unless he appears in `news.injuries` with that
  status. If `stale` is true, the official weekly report for this week hasn't
  posted yet and statuses come from ESPN only -- say so if an injury drives
  your call. If `conflicts` lists a player, the sources disagreed; the
  bundle's status is the resolved one -- name the conflict if you rely on it.
  If `espn_available` is false, statuses are the official weekly report only
  and may be last week's -- say so before relying on one.

OUTPUT: return ONLY a JSON array (no prose, no markdown fences). One object per
game in the bundle, each EXACTLY:
{
  "game_pk": <int, copied from the bundle game>,
  "ml_pick_team": "<exact team name -- this game's home_team or away_team>",
  "spread_pick_team": "<exact team name>" | null,
  "conviction_tier": "high" | "medium" | "low",
  "confidence": <number in [0,1]>,
  "rationale": "<prose citing at least one concrete fact; name the team>",
  "agent_notes": {"statistics": "...", "analyst": "...", "news": "..."}
}
Do NOT emit "home"/"away", spread_line, total fields, matchup, sport, or
commence_time -- teams resolve to sides downstream. Return every game_pk once.
"""


def _split_matchup(matchup: str) -> tuple[str, str]:
    """('away_team', 'home_team') from an 'Away @ Home' matchup string."""
    parts = [p.strip() for p in str(matchup or "").split(" @ ")]
    return (parts[0], parts[1]) if len(parts) == 2 else ("", "")


def _resolve_side(team: str | None, away_team: str, home_team: str) -> str | None:
    """Map a team NAME to 'home'/'away' for its game; None if it matches
    neither. Tolerant of minor vari/substring differences so the desk can't
    invert home/away."""
    if not team:
        return None
    t = team.strip().lower()
    h, a = home_team.lower(), away_team.lower()
    if t == h:
        return "home"
    if t == a:
        return "away"
    # substring fallback (e.g. "Chiefs" vs "Kansas City Chiefs")
    h_hit = t in h or h in t
    a_hit = t in a or a in t
    if h_hit and not a_hit:
        return "home"
    if a_hit and not h_hit:
        return "away"
    return None


def _enrich(game: dict) -> dict:
    """Add explicit home_team/away_team and the model's own pick (by team name)
    so the LLM never has to infer home/away."""
    away, home = _split_matchup(game.get("matchup", ""))
    wp = (game.get("model") or {}).get("win_prob")
    model_pick_team = None
    if wp is not None:
        model_pick_team = home if wp >= 0.5 else away
    return {**game, "home_team": home, "away_team": away,
            "model_pick_team": model_pick_team}


DEFAULT_MAX_TOKENS = 32000  # a CFB bundle (~50-65 games) citing every trend needs headroom


def _max_tokens() -> int:
    """DESK_SYNTH_MAX_TOKENS, else DEFAULT_MAX_TOKENS (an empty env var -- an
    unset `${{ vars.* }}` in CI -- also falls back)."""
    return int(os.environ.get("DESK_SYNTH_MAX_TOKENS") or DEFAULT_MAX_TOKENS)


def _response_text(resp, max_tokens: int) -> str:
    """The text of a Messages API response. Raises a clear RuntimeError when the
    response was cut off at max_tokens (a truncated array would otherwise
    surface as a confusing JSON parse error)."""
    if getattr(resp, "stop_reason", None) == "max_tokens":
        raise RuntimeError(
            f"desk response truncated at max_tokens={max_tokens}; raise DESK_SYNTH_MAX_TOKENS")
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


def _create_message(client, **kw):
    """messages.create, but STREAMED: at max_tokens above ~21k the SDK refuses a
    non-streaming request client-side ("Streaming is required for operations
    that may take longer than 10 minutes"). Returns the final Message."""
    with client.messages.stream(**kw) as stream:
        return stream.get_final_message()


def _model_decisions(bundle: list[dict], model: str, max_tokens: int) -> list[dict]:
    """Call the Anthropic API and return the parsed decision array. Raises on
    an unparseable response."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the env
    user = (
        "Here is the games bundle. Synthesize the desk picks per the "
        "methodology and return only the JSON array.\n\n"
        + json.dumps([_enrich(g) for g in bundle], default=str)
    )
    resp = _create_message(
        client,
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return _parse_array(_response_text(resp, max_tokens))


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

    Only games the model returned a decision for become picks. The desk picks a
    TEAM (ml_pick_team / spread_pick_team); this resolves the team to home/away
    against the game's own matchup, so the desk can never invert the side.
    Deterministic fields (game_pk, commence_time, matchup, spread_line,
    total_side/line) come from the bundle/policy, never the model.
    """
    by_pk = {g["game_pk"]: g for g in bundle}
    model_version = f"desk-{sport}-v1"
    picks: list[dict] = []
    for dec in decisions:
        g = by_pk.get(dec.get("game_pk"))
        if g is None:
            continue  # model hallucinated a game_pk -> drop it
        away_team, home_team = _split_matchup(g.get("matchup", ""))

        ml_pick = _resolve_side(dec.get("ml_pick_team"), away_team, home_team)
        if ml_pick not in _ML_SIDES:
            # unresolvable team name -> fall back to the model's favored side
            wp = (g.get("model") or {}).get("win_prob")
            ml_pick = "home" if (wp is not None and wp >= 0.5) else "away"

        spread_side = _resolve_side(dec.get("spread_pick_team"), away_team, home_team)
        mkt_s = g.get("market_spread")
        if spread_side is not None and mkt_s is not None:
            # spread_line is ALWAYS the HOME-team line (grade_desk_picks +
            # the front-end both read it home-referenced); the side, not the
            # sign, records which team the desk backed.
            spread_line = mkt_s
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


_RECORD_RE = re.compile(r"\b\d{1,2}-\d{1,2}\b")


def _nickname(team: str) -> str:
    """The distinctive last word of a team name ("Kansas City Chiefs" ->
    "chiefs"), used to find the team in free-text rationale."""
    return team.split()[-1].lower() if team else ""


def flag_pick_issues(picks: list[dict], bundle: list[dict]) -> list[str]:
    """Consistency guard between a pick's rationale prose and the ground truth.
    PURE. Returns human-readable flags (empty = clean). Catches the failure
    mode where the desk's rationale contradicts its own pick:

      1. Record misattribution -- the rationale ties a team's nickname to the
         OTHER team's record (e.g. "Chiefs are 4-1" when KC is 0-5 and 4-1 is
         Denver's). This is how a hallucinated rationale flips which team looks
         strong. Only checked when the two records differ.
      2. Picked team unmentioned -- the rationale never names the team the pick
         actually resolves to, so the prose is about a different team.
    """
    by_pk = {g["game_pk"]: g for g in bundle}
    flags: list[str] = []
    for p in picks:
        g = by_pk.get(p["game_pk"])
        if g is None:
            continue
        away_team, home_team = _split_matchup(g.get("matchup", ""))
        form = g.get("form") or {}
        rec_home = (form.get("home") or {}).get("record")
        rec_away = (form.get("away") or {}).get("record")
        text = (p.get("rationale") or "").lower()

        # (1) record misattribution
        if rec_home and rec_away and rec_home != rec_away:
            for team, own, other in ((home_team, rec_home, rec_away),
                                     (away_team, rec_away, rec_home)):
                nick = _nickname(team)
                for m in re.finditer(re.escape(nick), text):
                    seg = text[m.start(): m.end() + 45]
                    if other in seg and own not in seg:
                        flags.append(
                            f"game {p['game_pk']}: rationale ties {team} to record "
                            f"{other} but its record is {own}")
                        break

        # (2) picked team must be named in the rationale
        picked_team = home_team if p.get("ml_pick") == "home" else away_team
        if text and _nickname(picked_team) and _nickname(picked_team) not in text:
            flags.append(
                f"game {p['game_pk']}: pick is {picked_team} but the rationale "
                f"never names it")
    return flags


_DEMOTE_MARKER = "[flagged: rationale consistency check — treat with caution] "


def _flagged_game_pks(flags: list[str]) -> set[int]:
    """Game ids referenced by flag strings ('game <pk>: ...')."""
    pks: set[int] = set()
    for f in flags:
        m = re.match(r"game (\d+):", f)
        if m:
            pks.add(int(m.group(1)))
    return pks


def demote_flagged_picks(picks: list[dict], flags: list[str]) -> int:
    """Neutralize picks whose rationale still failed the consistency guard after
    the repair round: drop the spread lean and cap conviction to 'low' (so the
    desk barely nudges the board), and prefix a caution marker to the rationale
    so a shaky pick never surfaces as a confident one. ml_pick is kept (contract
    requires it) but at minimum conviction. Returns how many were demoted."""
    pks = _flagged_game_pks(flags)
    n = 0
    for p in picks:
        if p.get("game_pk") in pks:
            p["spread_side"] = None
            p["spread_line"] = None
            p["conviction_tier"] = "low"
            rat = p.get("rationale") or ""
            if not rat.startswith(_DEMOTE_MARKER):
                p["rationale"] = _DEMOTE_MARKER + rat
            n += 1
    return n


_DISAGREEMENT_THRESHOLD = 0.15


def apply_disagreement_cap(
    picks: list[dict], bundle: list[dict], threshold: float = _DISAGREEMENT_THRESHOLD,
) -> int:
    """Cap conviction when the sim engine and the analytic model sharply
    disagree. PURE. For each pick whose bundle game carries a `sim` block
    with `disagreement` > `threshold` AND whose `conviction_tier == "high"`,
    downgrade the tier to "medium" and prepend a short note to the
    rationale. Picks that are already medium/low, or whose game has no sim
    block (absent key or explicit None -- e.g. a CFB bundle, or an NFL game
    the sim hasn't covered yet), are left untouched. Returns how many picks
    were capped."""
    by_pk = {g["game_pk"]: g for g in bundle}
    n = 0
    for p in picks:
        if p.get("conviction_tier") != "high":
            continue
        g = by_pk.get(p.get("game_pk"))
        sim = (g or {}).get("sim")
        if not sim:
            continue
        disagreement = sim.get("disagreement")
        if disagreement is None or disagreement <= threshold:
            continue
        p["conviction_tier"] = "medium"
        note = f"[sim disagreement {disagreement:.2f} — conviction capped] "
        p["rationale"] = note + (p.get("rationale") or "")
        n += 1
    return n


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
    max_tokens = _max_tokens()

    bundle = json.loads(args.bundle.read_text())
    if not bundle:
        print("Empty bundle -- writing empty picks list.")
        args.out.write_text("[]")
        return

    decisions = _model_decisions(bundle, model, max_tokens)
    picks = assemble_picks(bundle, decisions, args.sport)

    problems = validate_picks(picks)
    flags = flag_pick_issues(picks, bundle)
    if problems or flags:
        # One repair round: hand the model its own output + every issue. Contract
        # violations (problems) and rationale/pick consistency flags are both
        # things the model must fix.
        print(f"{len(problems)} validation problem(s), {len(flags)} consistency "
              f"flag(s); attempting one repair round.")
        import anthropic

        client = anthropic.Anthropic()
        repair = (
            "Your previous picks had issues. Fix every one.\n"
            + "Contract problems:\n" + "\n".join(f"- {p}" for p in problems)
            + "\nRationale/pick consistency flags (the rationale must match the "
              "team you actually pick and cite the correct records):\n"
            + "\n".join(f"- {f}" for f in flags)
            + "\n\nHere was your output:\n"
            + json.dumps(decisions, default=str)
            + "\n\nReturn a corrected JSON array (same schema, only the "
            "judgment fields)."
        )
        resp = _create_message(
            client,
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": repair}],
        )
        decisions = _parse_array(_response_text(resp, max_tokens))
        picks = assemble_picks(bundle, decisions, args.sport)
        problems = validate_picks(picks)
        flags = flag_pick_issues(picks, bundle)

    if problems:
        print(f"Still {len(problems)} problem(s) after repair; writing NOTHING:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    # Residual consistency flags don't block the write (the pick SIDE is correct
    # via team resolution), but a shaky rationale shouldn't surface as a
    # confident pick: drop its spread lean, cap it to low conviction, and mark
    # the rationale. Surfaced loudly so it's never silent.
    if flags:
        print(f"WARNING: {len(flags)} rationale/pick consistency flag(s) remain "
              f"after repair:")
        for f in flags:
            print(f"  - {f}")
        demoted = demote_flagged_picks(picks, flags)
        print(f"auto-demoted {demoted} flagged pick(s): spread lean dropped, "
              f"conviction -> low, rationale marked.")

    capped = apply_disagreement_cap(picks, bundle)
    if capped:
        print(f"capped {capped} high-conviction pick(s): sim/model disagreement "
              f"> {_DISAGREEMENT_THRESHOLD} -> conviction -> medium.")

    args.out.write_text(json.dumps(picks, indent=2))
    leans = sum(1 for p in picks if p["spread_side"])
    from collections import Counter
    tiers = dict(Counter(p["conviction_tier"] for p in picks if p["spread_side"]))
    print(f"Synthesized {len(picks)} picks ({leans} spread leans, tiers {tiers}) via {model} -> {args.out}")


if __name__ == "__main__":
    main()
