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
  return latestSplitMap([...(nfl || []), ...(cfb || [])]);
}
// Newest capture of each game|market|side from betting-splits rows.
function latestSplitMap(rows) {
  const m = new Map();
  (rows || []).forEach((r) => {
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
    sb(`ev_prop_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,player_id,player_name,market,side,created_at${q}`).catch(() => []),
  ]);
  const first = new Map();
  const add = (r, kind) => {
    const k = `${kind}|${r.sport}|${r.game_pk}|${r.player_id || ""}|${r.market}|${r.side}`;
    if (!first.has(k) || r.created_at < first.get(k).created_at) first.set(k, { ...r, kind });
  };
  (lines || []).forEach((r) => add(r, "line")); (props || []).forEach((r) => add(r, "prop"));
  const oldest = etDateStr(new Date(Date.now() - (days - 1) * 864e5).toISOString());  // today inclusive
  // date = ET day first flagged; the pick's identity + first-flag time ride along (dashboard New Signals).
  return [...first.values()].map((r) => ({ date: etDateStr(r.created_at), kind: r.kind, sport: r.sport, game_pk: r.game_pk,
    market: r.market, side: r.side, player_id: r.player_id ?? null, player_name: r.player_name ?? null, at: r.created_at }))
    .filter((r) => r.date >= oldest);
}

/* ── Shared page helpers (used by more than one page) ─────────────────────
   Depend on app.js (etDateStr, inTrackRecord, bpKick, logoImg, CFB_2W_MASCOTS, ctxEsc), metrics.js and ui.js. */
const SPORT_NAME = { nfl: "NFL", cfb: "CFB", mlb: "MLB", nba: "NBA" };
const LEAGUE_LOGO = { nfl: "teamlogos/leagues/500/nfl", cfb: "espn/misc_logos/500/ncaa_football", mlb: "teamlogos/leagues/500/mlb", nba: "teamlogos/leagues/500/nba" };
const addDays = (d, n) => new Date(Date.parse(`${d}T12:00:00Z`) + n * 864e5).toISOString().slice(0, 10);
const timeMs = (iso) => (iso ? Date.parse(iso) : NaN);
const matchupSides = (matchup) => { const [a = "", h = ""] = String(matchup || "").split(" @ "); return [a, h]; };
const lineStr = (x) => (x === 0 ? "PK" : x > 0 ? `+${x}` : `${x}`);
const signedStr = (x, d = 1, unit = "") => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}${unit}`;
const signCls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");
const emptyMsg = (msg) => `<p class="ca-empty">${ctxEsc(msg)}</p>`;
const leagueLogo = (s) => (LEAGUE_LOGO[s] ? `<img class="ca-league" src="https://a.espncdn.com/i/${LEAGUE_LOGO[s]}.png" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : "");
const gameHref = (sport, gamePk) => `game.html?sport=${encodeURIComponent(sport)}&game=${encodeURIComponent(gamePk)}`;
const goLink = (sport, gamePk) => `<a class="ca-go" href="${gameHref(sport, gamePk)}" aria-label="Open game">→</a>`;
// "Oct 13" / "Oct 13, 2026" for a YYYY-MM-DD ET game date (UTC-pinned so the calendar day never shifts).
const shortDate = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
const fullDate = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric", year: "numeric" });
// Area-chart points for a dated series [{date, ...}]: x labels only on a sparse cadence (every 2nd slot of areaChart's own ceil(n/7)
// step beyond 14 points) plus the last point, skipping a slot that would collide with it.
function dateAxisPoints(items, valueOf) {
  const rs = items || [], n = rs.length, every = Math.max(1, Math.ceil(n / 7)), step = n > 14 ? 2 * every : every;
  return rs.map((q, i) => ({ x: i === n - 1 || (i % step === 0 && n - 1 - i >= Math.max(2, step * 0.5)) ? shortDate(q.date) : null, y: valueOf(q) }));
}
const kickLabel = (iso) => (iso ? bpKick(iso).replace(/ ET$/, "") : "");

// Display name as the mockups show it: NFL nickname ("Lions"), CFB school ("Ohio State"), else as-is.
function shortTeam(name, sport) {
  const n = String(name || "").trim();
  if (!n) return "";
  const w = n.split(/\s+/);
  if (sport === "nfl") return w[w.length - 1];
  if (sport === "cfb" && w.length > 1) {
    const last2 = w.slice(-2).join(" ").toLowerCase().replace(/[^a-z ]/g, "");
    return w.slice(0, Math.max(1, w.length - (CFB_2W_MASCOTS.has(last2) ? 2 : 1))).join(" ");
  }
  return n;
}
const shortMatchup = (away, home, sport) => `${shortTeam(away, sport)} @ ${shortTeam(home, sport)}`;

// The bet as placed: "Lions ML", "Chiefs -6.5", "Over 56.5", "Under 64.5 Rec Yds".
// Spread/total numbers come from the best book's line (ev_best_lines); none known -> no number.
function pickLabel(o, lineBy) {
  const ou = o.side === "under" ? "Under" : "Over";
  if (o.kind === "prop") return `${ou}${o.line != null ? ` ${o.line}` : ""} ${o.marketLabel || o.market}`;
  const l = lineBy && lineBy.get(`${o.game_pk}|${o.market}|${o.side}`);
  if (o.market === "total") return `${ou} ${l != null ? l : "total"}`;
  const [away, home] = matchupSides(o.matchup);
  const team = shortTeam(o.side === "home" ? home : away, o.sport);
  if (o.market === "moneyline") return `${team} ML`;
  return `${team} ${l != null ? lineStr(l) : "spread"}`;
}
// pickLabel with the player's name in front for props ("Josh Allen Over 64.5 Rec Yds").
const oppLabel = (o, lineBy) => `${o.kind === "prop" ? `${o.playerName || ""} ` : ""}${pickLabel(o, lineBy)}`.trim();
// ET calendar day of a predictions / results row ("" when it has neither a kickoff nor a game_date).
const gameDate = (r) => (r && r.commence_time ? etDateStr(r.commence_time) : (r && r.game_date) || "");

// One row per game (predictions), carrying its highest-alpha opportunity (ties -> higher EV) or null.
function slateRows(predRows, opps) {
  const best = new Map();
  for (const o of opps || []) {
    if (!o) continue;
    const k = `${o.sport}|${o.game_pk}`, b = best.get(k);
    if (!b || o.alpha > b.alpha || (o.alpha === b.alpha && (o.evPct ?? -Infinity) > (b.evPct ?? -Infinity))) best.set(k, o);
  }
  const seen = new Set(), out = [];
  for (const r of predRows || []) {
    if (!r) continue;
    const k = `${r.sport}|${r.game_pk}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ sport: r.sport, game_pk: r.game_pk, away: r.away_team_name, home: r.home_team_name, commence: r.commence_time || null, pred: r, opp: best.get(k) || null });
  }
  const t = (r) => { const v = timeMs(r.commence); return Number.isFinite(v) ? v : Infinity; };
  return out.sort((a, b) => t(a) - t(b) || String(a.game_pk).localeCompare(String(b.game_pk)));
}

// Compact "best bet" text for a slate row: props lead with the player's last name ("Allen Over 64.5 Rec Yds"), lines read as pickLabel.
const oppSlateLabel = (o, lineBy) => `${o.kind === "prop" ? `${String(o.playerName || "").split(/\s+/).slice(-1)[0]} ` : ""}${pickLabel(o, lineBy)}`.trim();
// Logo of the team a game-line opportunity is on ("" for props and totals).
const oppLogo = (o) => {
  if (o.kind === "prop" || o.market === "total") return "";
  const [away, home] = matchupSides(o.matchup);
  return logoImg(o.side === "home" ? home : away, o.sport);
};

// One throwing card must not blank the page: it renders a small placeholder and logs the error.
function safeCard(name, fn, D, cls = "ca-card", id = "") {
  try { return fn(D); }
  catch (e) {
    console.error(`card "${name}" failed to render`, e);
    return `<section class="${cls}"${id ? ` id="${id}"` : ""}><p class="ca-empty">This panel couldn't load.</p></section>`;
  }
}

// Graded game-line +EV picks inside the track record: won, flagged price, implied prob, edge (pp),
// profit in units at the flagged price (win: decimal - 1, loss: -1) and CLV.
function gradedLinePicks(results, picks, starts) {
  const key = (r) => `${r.sport}|${r.game_pk}|${r.market}|${r.side}`;
  const by = new Map();   // latest rebuild of each pick (max created_at) sets the price
  for (const p of picks || []) { const prev = by.get(key(p)); if (!prev || String(p.created_at || "") >= String(prev.created_at || "")) by.set(key(p), p); }
  const out = [];
  for (const r of results || []) {
    if (!r || (r.won !== true && r.won !== false)) continue;
    const p = by.get(key(r));
    if (!p || !p.commence_time || !inTrackRecord(starts, r.sport, p.commence_time)) continue;
    const imp = americanToProb(p.best_price), implied = Number.isFinite(imp) ? imp : null;
    out.push({ sport: r.sport, game_pk: r.game_pk, market: r.market, won: r.won, implied,
      edgePp: implied != null && finite(p.true_prob) ? (+p.true_prob - implied) * 100 : null, date: etDateStr(p.commence_time),
      profitUnits: r.won ? (implied != null ? 1 / implied - 1 : null) : -1, clv: finite(r.clv) ? +r.clv : null });
  }
  return out;
}

// One population for every performance number: graded +EV picks (ev_pnl_daily, all markets) in [from, to].
// Hit rate = wins / (wins + losses). vs-market and avg edge use the graded game-line picks with a known flagged price.
function perfWindow(evRows, graded, from, to) {
  const rows = (evRows || []).filter((r) => r && r.game_date >= from && r.game_date <= to);
  const a = aggPnl(rows), decidedN = a.w + a.l;
  const g = (graded || []).filter((r) => r.date >= from && r.date <= to);
  const edges = g.map((r) => r.edgePp).filter((x) => x != null);
  const priced = g.filter((r) => r.implied != null);
  return { rows, n: a.n, units: a.units, roiPct: a.roiPct, wins: a.w, losses: a.l,
    hitRate: decidedN ? a.w / decidedN : null,
    vsMarket: hitRateVsMarket(priced), pricedN: priced.length,
    avgEdge: edges.length ? edges.reduce((s, x) => s + x, 0) / edges.length : null, edgeN: edges.length };
}

// One captured line move (line_moves_current row `m`) for its game `p` (a predictions row), as display text:
// title ("Ravens -2.5 → -4.0"), matchup, change (moneyline: implied-prob %, else points), signed delta, ticket split.
function lineMoveInfo(m, p, splits) {
  const team = shortTeam(m.side === "home" ? p.home_team_name : p.away_team_name, p.sport), mu = shortMatchup(p.away_team_name, p.home_team_name, p.sport);
  let title, chg, d;
  if (m.market === "moneyline") {
    d = (americanToProb(m.cur_price) - americanToProb(m.open_price)) * 100;
    title = `${team} ML ${oddsStr(m.open_price)} → ${oddsStr(m.cur_price)}`; chg = signedStr(d, 1, "%");
  } else {
    d = +m.cur_line - +m.open_line;
    title = m.market === "total" ? `${teamShort(p.away_team_name, p.sport)} @ ${teamShort(p.home_team_name, p.sport)} O/U ${+m.open_line} → ${+m.cur_line}` : `${team} ${lineStr(+m.open_line)} → ${lineStr(+m.cur_line)}`;
    chg = `${signedStr(d, 1)} pts`;
  }
  const sp = splits && splits.get(`${m.game_pk}|${m.market}|${m.side}`);
  const tickets = sp && finite(sp.ticket_pct) ? `${Math.round(+sp.ticket_pct)}% of tickets on ${m.market === "total" ? (m.side === "under" ? "Under" : "Over") : team}` : "";
  return { team, mu, title, chg, d, tickets };
}
