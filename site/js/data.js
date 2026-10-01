/* Data layer for the redesign: one Opportunity model (game lines + props) scored with
   the Alpha Score, plus line moves, splits and +EV history. Fetchers never throw:
   a missing view/table yields empty data and the panel hides. */
const SPORTS = ["nfl", "cfb", "mlb", "nba"];
const LIVE_SPORTS = ["nfl", "cfb"];
const SPORT_STATUS = { mlb: "MLB model paused (last projections Aug 31, 2026)", nba: "NBA model not live yet" };
const PROP_LABEL = { pass_yds: "Pass Yds", pass_tds: "Pass TDs", rush_yds: "Rush Yds", rec_yds: "Rec Yds", receptions: "Receptions", rush_att: "Rush Att", completions: "Completions", pass_att: "Pass Att", interceptions: "INTs" };
const segmentKey = (sport, kind, market) => `${sport}|${kind === "prop" ? "prop" : market}`;
const finite = (x) => x != null && x !== "" && Number.isFinite(+x);

function segmentRecords(evPnlRows) {
  const by = new Map();
  (evPnlRows || []).forEach((r) => { const k = `${r.sport}|${r.market}`; (by.get(k) || by.set(k, []).get(k)).push(r); });
  const out = new Map();
  by.forEach((rows, k) => { const a = aggPnl(rows); out.set(k, { roiPct: a.roiPct, n: a.n }); });
  return out;
}

function toOpportunity(r, kind, seg) {
  const isProp = kind === "prop";
  const modelProb = isProp ? r.model_prob : r.true_prob;
  if (!finite(modelProb) || !finite(r.best_price) || !finite(r.ev_best)) return null;
  const implied = !isProp && finite(r.best_line_implied) ? +r.best_line_implied : americanToProb(+r.best_price);
  const evPct = +r.ev_best * 100, edgePp = (+modelProb - implied) * 100;
  const alpha = alphaScore({ evPct, edgePp, segRoiPct: seg?.roiPct || 0, segN: seg?.n || 0 });
  return {
    kind, sport: r.sport, game_pk: r.game_pk, matchup: r.matchup || "", market: r.market,
    marketLabel: isProp ? (PROP_LABEL[r.market] || r.market) : ({ moneyline: "ML", spread: "Spread", total: "Total" }[r.market] || r.market),
    side: r.side, playerName: isProp ? r.player_name : null, line: finite(r.line) ? +r.line : null,
    commence: r.commence_time, book: r.best_book, odds: +r.best_price,
    modelProb: +modelProb, impliedProb: implied, edgePp, evPct, alpha, tier: alphaTier(alpha),
  };
}

async function loadOpportunities() {
  const [evPnl, ...per] = await Promise.all([
    sb("ev_pnl_daily?select=*").catch(() => []),
    ...LIVE_SPORTS.flatMap((s) => [evCurrent(s).catch(() => []), evPropsCurrent(s).catch(() => [])]),
  ]);
  const segs = segmentRecords(evPnl);
  const out = [];
  LIVE_SPORTS.forEach((s, i) => {
    for (const r of per[2 * i] || []) { if (r.is_pick !== true) continue; const o = toOpportunity(r, "line", segs.get(segmentKey(s, "line", r.market))); if (o) out.push(o); }
    for (const r of per[2 * i + 1] || []) { const o = toOpportunity(r, "prop", segs.get(segmentKey(s, "prop", r.market))); if (o) out.push(o); }
  });
  return out.sort((a, b) => b.alpha - a.alpha || b.evPct - a.evPct);
}

async function loadLineMoves() {
  const rows = await sb("line_moves_current?select=*").catch(() => []);
  return (rows || []).map((r) => ({ ...r, score: lineMoveScore(r) })).filter((r) => r.score > 0).sort((a, b) => b.score - a.score);
}

async function loadSplits() {
  const [nfl, cfb] = await Promise.all([
    sb("nfl_betting_splits_current?select=*").catch(() => []),
    sb("cfb_betting_splits_current?select=*").catch(() => []),
  ]);
  const m = new Map();
  [...(nfl || []), ...(cfb || [])].forEach((r) => {
    const k = `${r.game_pk}|${r.market}|${r.side}`, prev = m.get(k);
    if (!prev || r.captured_at > prev.captured_at) m.set(k, r);
  });
  return m;
}

async function loadEvHistory(days) {
  // Look back `days + 14` so a pick first flagged before the window but rebuilt inside it is not counted as new.
  const since = new Date(Date.now() - (days + 14) * 864e5).toISOString();
  const q = "&order=created_at.asc&limit=10000";
  const [lines, props] = await Promise.all([
    sb(`ev_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,market,side,created_at${q}`).catch(() => []),
    sb(`ev_prop_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,player_id,market,side,created_at${q}`).catch(() => []),
  ]);
  const first = new Map();
  const add = (r, kind) => {
    const k = `${kind}|${r.sport}|${r.game_pk}|${r.player_id || ""}|${r.market}|${r.side}`;
    if (!first.has(k) || r.created_at < first.get(k).created_at) first.set(k, { ...r, kind });
  };
  (lines || []).forEach((r) => add(r, "line")); (props || []).forEach((r) => add(r, "prop"));
  const oldest = etDateStr(new Date(Date.now() - (days - 1) * 864e5).toISOString());  // today inclusive
  return [...first.values()].map((r) => ({ date: etDateStr(r.created_at), kind: r.kind, sport: r.sport }))
    .filter((r) => r.date >= oldest);
}
