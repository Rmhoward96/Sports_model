/* Data layer for the redesign: one Opportunity model (game lines + props) scored with
   the Alpha Score, plus line moves, splits and +EV history. Fetchers never throw:
   a missing view/table yields empty data and the panel hides. Reads go through app.js sb / sbAll (paged). */
const SPORTS = ["nfl", "cfb", "mlb", "nba"];
const LIVE_SPORTS = ["nfl", "cfb", "mlb"];     // sports with a running model: their pages, picks and record are live (a sport's own empty states say why a day has no games)
const SPORT_STATUS = { nba: "NBA model not live yet" };     // sports with NO model; a live sport never carries a status (no hard-coded "paused" text)
const PROP_LABEL = { pass_yds: "Pass Yds", pass_tds: "Pass TDs", rush_yds: "Rush Yds", rec_yds: "Rec Yds", receptions: "Receptions", rush_att: "Rush Att", completions: "Completions", pass_att: "Pass Att", interceptions: "INTs",
  total_bases: "Total Bases", pitcher_ks: "Strikeouts", hits_allowed: "Hits Allowed", outs_recorded: "Outs Recorded" };
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
  // The date filter bounds the read; sbAll pages through it (PostgREST returns at most 1,000 rows per request).
  const since = new Date(Date.now() - (days + 14) * 864e5).toISOString();
  const [lines, props] = await Promise.all([
    sbAll(`ev_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,market,side,created_at&order=created_at.asc,sport.asc,game_pk.asc,market.asc,side.asc,model_version.asc`).catch(() => []),
    sbAll(`ev_prop_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,player_id,player_name,market,side,created_at&order=created_at.asc,game_pk.asc,player_id.asc,market.asc,line.asc,model_version.asc`).catch(() => []),
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
   Depend on app.js (pct1, etDateStr, inTrackRecord, bpKick, logoImg, CFB_2W_MASCOTS, ctxEsc), metrics.js and ui.js. */
const SPORT_NAME = { nfl: "NFL", cfb: "CFB", mlb: "MLB", nba: "NBA" };
const LEAGUE_LOGO = { nfl: "teamlogos/leagues/500/nfl", cfb: "espn/misc_logos/500/ncaa_football", mlb: "teamlogos/leagues/500/mlb", nba: "teamlogos/leagues/500/nba" };
const addDays = (d, n) => new Date(Date.parse(`${d}T12:00:00Z`) + n * 864e5).toISOString().slice(0, 10);
const timeMs = (iso) => (iso ? Date.parse(iso) : NaN);
const matchupSides = (matchup) => { const [a = "", h = ""] = String(matchup || "").split(" @ "); return [a, h]; };
const lineStr = (x) => (x === 0 ? "PK" : x > 0 ? `+${x}` : `${x}`);
// The MODEL's spread as every page shows it (board, game page): rounded to the half point, "PK" at 0 ("ATL -0.2" -> "PK").
const modelLineStr = (x) => lineStr(r05(+x) || 0);
const signedStr = (x, d = 1, unit = "") => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}${unit}`;
const signCls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");
const emptyMsg = (msg) => `<p class="ca-empty">${ctxEsc(msg)}</p>`;
// ESPN's ncaa_football logo renders as a black person silhouette, so CFB gets an inline brown football with white laces instead (sized by .ca-league like the other league logos).
const CFB_BALL = `<svg class="ca-league" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><g transform="rotate(-45 12 12)"><ellipse cx="12" cy="12" rx="11.5" ry="7" fill="#8B4A22" stroke="#5A2E12" stroke-width="1"/><path d="M5.4 7.4v9.2M18.6 7.4v9.2" stroke="#fff" stroke-width="1.1" fill="none" stroke-linecap="round"/><path d="M8.2 12h7.6M10 10.1v3.8M12 9.9v4.2M14 10.1v3.8" stroke="#fff" stroke-width="1.15" stroke-linecap="round" fill="none"/></g></svg>`;
const leagueLogo = (s) => (s === "cfb" ? CFB_BALL : LEAGUE_LOGO[s] ? `<img class="ca-league" src="https://a.espncdn.com/i/${LEAGUE_LOGO[s]}.png" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : "");
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
// R19: an opportunity's probability is the MODEL's only for props; a game line's is Pinnacle's no-vig fair price
// (ev_current.true_prob). The column header says so and every game-line value carries a small "fair" marker.
const OPP_PROB_TIP = "Model for player props; sharp fair price (Pinnacle no-vig) for game lines";
const oppProbTh = (label = "Model Prob.") => `<th title="${OPP_PROB_TIP}">${label}</th>`;
const oppProb = (o) => `${pct1(o.modelProb)}${o.kind === "prop" ? "" : `<span class="ca-fair" title="Sharp fair price (Pinnacle no-vig), not the model">fair</span>`}`;
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

/* ── Graded model picks (shared by Track Record and the sport pages) ─────────────────────────────────────────────────
   Moved from track.js: trackRows turns prediction_accuracy rows into one row per graded market; trackClosing fetches the
   moneyline closing prices of those games. Depend on app.js (sb), metrics.js (numOrNull, unitsFromPnl) and this file. */
const TRACK_MARKETS = ["moneyline", "spread", "total"];
const TRACK_CLOSING_CHUNK = 60;
const TRACK_ACC_SELECT = "sport,game_pk,game_date,home_team_name,away_team_name,win_prob,predicted_winner,actual_winner,winner_correct,pred_margin,actual_margin,pred_total,actual_total,market_spread,market_total,spread_pick_correct,total_pick_correct";

// Decimal odds -> American price (null when unusable or not above 1.00).
function decToAmerican(dec) {
  const d = numOrNull(dec);
  if (d == null || d <= 1) return null;
  return d >= 2 ? Math.round((d - 1) * 100) : -Math.round(100 / (d - 1));
}
// game_closing_prices rows ({game_pk, side, close_dec}) -> Map "game_pk|side" -> American price.
function trackClosingMap(rows) {
  const m = new Map();
  for (const r of rows || []) {
    const a = r ? decToAmerican(r.close_dec) : null;
    if (a != null) m.set(`${r.game_pk}|${r.side}`, a);
  }
  return m;
}
const trackNum0 = (x) => (x === 0 ? 0 : x);   // never -0
const trackGradeOf = (b) => (b === true ? "W" : b === false ? "L" : "P");

// One row per graded pick of a graded game: moneyline (always), spread (when a market line is stored AND the model has a side), total
// (likewise). A spread / total with no model pick (model exactly on the line, or no model numbers) is not a bet and gets no row,
// like prediction_pnl's no-bet exclusion (R23). Confidence is the model's moneyline win probability, so only moneyline rows carry it. A game without actual_winner is ungraded and gives no rows (the same population as prediction_pnl).
// prob = the model's favorite-side win probability (moneyline only; spread / total rows carry none). units = the matching
// prediction_pnl row's pnl / 10, else null. `closing`: Map "game_pk|side" -> moneyline closing price (from trackClosingMap).
function trackRows(accuracyRows, pnlRows, closing) {
  const units = new Map();
  for (const p of pnlRows || []) if (p && finite(p.pnl)) units.set(`${p.game_pk}|${p.market}`, unitsFromPnl(p.pnl));
  const out = [];
  for (const r of accuracyRows || []) {
    if (!r || r.actual_winner == null) continue;
    const wp = numOrNull(r.win_prob), conf = wp == null ? null : Math.max(wp, 1 - wp);
    const am = numOrNull(r.actual_margin), at = numOrNull(r.actual_total);
    const finalScore = `${am === 0 ? "Tie" : `${r.actual_winner}${am != null ? ` by ${Math.abs(am)}` : ""}`}${at != null ? ` · ${at} total` : ""}`;
    const base = { date: r.game_date, sport: r.sport, game_pk: r.game_pk, home: r.home_team_name, away: r.away_team_name,
      matchup: `${r.away_team_name} @ ${r.home_team_name}`, winner: r.actual_winner, finalScore };
    const push = (market, side, pick, closingLine, modelLine, prob, grade) => out.push({ ...base, market, side, pick, closing: closingLine, modelLine, prob, conf: market === "moneyline" ? conf : null,
      result: trackGradeOf(grade), won: grade === true ? true : grade === false ? false : null, units: units.get(`${r.game_pk}|${market}`) ?? null });
    const mlSide = r.predicted_winner === r.home_team_name ? "home" : r.predicted_winner === r.away_team_name ? "away" : null;
    push("moneyline", mlSide, r.predicted_winner ?? null, mlSide && closing && closing.has(`${r.game_pk}|${mlSide}`) ? closing.get(`${r.game_pk}|${mlSide}`) : null, null, conf, r.winner_correct);
    const ms = numOrNull(r.market_spread), pm = numOrNull(r.pred_margin);
    if (ms != null) {
      const edge = pm == null ? null : pm + ms, side = edge == null || edge === 0 ? null : edge > 0 ? "home" : "away";
      if (side) push("spread", side, side === "home" ? r.home_team_name : r.away_team_name,
        trackNum0(side === "home" ? ms : -ms), trackNum0(side === "home" ? r05(-pm) : r05(pm)), null, r.spread_pick_correct);
    }
    const mt = numOrNull(r.market_total), pt = numOrNull(r.pred_total);
    if (mt != null) {
      const side = pt == null || pt === mt ? null : pt > mt ? "over" : "under";
      if (side) push("total", side, side === "over" ? "Over" : "Under", mt, r05(pt), null, r.total_pick_correct);
    }
  }
  const mkt = (r) => TRACK_MARKETS.indexOf(r.market);
  return out.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : (+b.game_pk - +a.game_pk) || mkt(a) - mkt(b)));
}
// Win % = W / (W + L): a push is not decided.
function trackTally(rows) {
  const rs = rows || [], w = rs.filter((r) => r.result === "W").length, l = rs.filter((r) => r.result === "L").length, p = rs.filter((r) => r.result === "P").length;
  return { w, l, p, n: rs.length, pct: w + l ? w / (w + l) : null };
}
// Moneyline closing prices of the graded games, in parallel chunks of 60 game ids: the whole view scan takes 1.3-2.7s against the
// 3s anon statement limit (a timeout would drop every price), a chunk answers in ~0.5s. A failed chunk leaves its games unpriced.
async function trackClosing(gamePks) {
  const ids = [...new Set((gamePks || []).filter((x) => x != null))], chunks = [];
  for (let i = 0; i < ids.length; i += TRACK_CLOSING_CHUNK) chunks.push(ids.slice(i, i + TRACK_CLOSING_CHUNK));
  const parts = await Promise.all(chunks.map((c) => sb(`game_closing_prices?market=eq.moneyline&game_pk=in.(${c.join(",")})&select=game_pk,side,close_dec`).catch(() => [])));
  return parts.flat();
}

// The record scope (track_record_start). trackRecordStarts() swallows errors into "no restarts", which would pull the archived
// pre-restart rows into the published record, so the record pages load it themselves and fail closed.
async function trackStarts() {
  try {
    const rows = await sb("track_record_start?select=sport,starts_at,model_version");
    return { starts: new Map((rows || []).map((r) => [r.sport, r])), failed: false };
  } catch (e) { console.error("track record: track_record_start failed to load", e); return { starts: new Map(), failed: true }; }
}

/* ── Seasons and weeks (NFL / CFB) ────────────────────────────────────────────────────────────────────────────────────
   The tables carry no week column (power_rankings' week counts played weeks, not a game's week), so a game's week comes from
   its ET date. Both leagues use Tuesday-to-Monday weeks (the new week starts the day after Monday night), which keeps a
   Wednesday opener, Thursday / Friday games, the weekend and Monday night together.
     NFL week 1 starts on the Tuesday after Labor Day (the Tuesday before the first Thursday game), weeks 1..23 (23 = Super Bowl, which lands two Tuesday-Monday weeks after the conference title games).
     CFB week 1 starts on the Tuesday before the Saturday of Labor Day weekend, so "Week 0" is the previous Tuesday-to-Monday
       (the Aug opener Saturday), weeks 0..21 (bowls and the title game run into January).
   The season is the calendar year of its opening; January / February games belong to the season that started the year before. */
const WEEK_BOUNDS = { nfl: [1, 23], cfb: [0, 21] };
const dayDiff = (a, b) => Math.round((Date.parse(`${a}T12:00:00Z`) - Date.parse(`${b}T12:00:00Z`)) / 864e5);
// First Monday of September (YYYY-MM-DD).
function laborDay(year) {
  const dow = new Date(`${year}-09-01T12:00:00Z`).getUTCDay();
  return addDays(`${year}-09-01`, (1 - dow + 7) % 7);
}
const seasonOf = (date) => (+String(date).slice(5, 7) >= 3 ? +String(date).slice(0, 4) : +String(date).slice(0, 4) - 1);
// The Tuesday on which `week` of `season` starts.
function weekStart(sport, season, week) {
  const ld = laborDay(season), w1 = sport === "cfb" ? addDays(ld, -6) : addDays(ld, 1);
  return addDays(w1, 7 * (week - 1));
}
// The week (a number, unclamped) a YYYY-MM-DD date falls in.
function weekOf(sport, date) {
  const season = seasonOf(date);
  return Math.floor(dayDiff(date, weekStart(sport, season, 1)) / 7) + 1;
}
const clampWeek = (sport, w) => { const [lo, hi] = WEEK_BOUNDS[sport] || [1, 1]; return Math.min(hi, Math.max(lo, w)); };
// "Oct 9 – Oct 13, 2025" (a single day: "Oct 13, 2025"; across a year: both years).
function rangeLabel(from, to) {
  if (!from) return "";
  if (!to || to === from) return fullDate(from);
  return from.slice(0, 4) === to.slice(0, 4) ? `${shortDate(from)} – ${fullDate(to)}` : `${fullDate(from)} – ${fullDate(to)}`;
}

/* ── Market Pulse rows (the +EV rail and the sport pages' Market Intelligence card) ─────────────────────────────────────
   Moved from ev.js: the four rows (Biggest Line Move, Highest Confidence Edge, Most Mispriced Total, Average Market Divergence).
   D = { preds (current predictions rows), moves (loadLineMoves), splits, opps (tiered opportunities), lineBy, nowMs, graded
   (graded +EV picks, gradedLinePicks), today }; scope a sport by passing only that sport's rows. Depends on ui.js icons. */
// Mean closing-line value (in %) of graded picks whose game day is in [from, to].
function marketMeanClv(rows, from, to) {
  const v = (rows || []).filter((r) => r && r.date >= from && r.date <= to && finite(r.clv)).map((r) => +r.clv * 100);
  return v.length ? { pct: v.reduce((s, x) => s + x, 0) / v.length, n: v.length } : null;
}
// The UPCOMING game (kickoff after nowMs; unknown kickoff = unverifiable, skipped) whose model total is furthest from
// the market total (unrounded model total).
function marketMispricedTotal(preds, nowMs = Date.now()) {
  let best = null;
  for (const r of preds || []) {
    if (!r || !(timeMs(r.commence_time) > nowMs) || !finite(r.pred_home_score) || !finite(r.pred_away_score) || !finite(r.market_total)) continue;
    const model = +r.pred_home_score + +r.pred_away_score, diff = model - +r.market_total;
    if (!best || Math.abs(diff) > Math.abs(best.diff)) best = { pred: r, market: +r.market_total, model, diff };
  }
  return best;
}

function marketPulseRow(icon, title, sub, val, valCls, valSub, href) {
  return `<a class="ca-ev-pr" href="${href}"><span class="ca-ev-pr-ic">${icon}</span><span class="ca-ev-pr-main"><b>${title}</b><small class="ca-ell" title="${uiTitleText(sub)}">${sub}</small></span><span class="ca-ev-pr-r"><b class="${valCls}">${val}</b><small>${valSub}</small></span></a>`;
}
function marketPulseRows(D) {
  const rows = [];
  const byPk = new Map(D.preds.map((r) => [String(r.game_pk), r]));
  const mv = D.moves.map((m) => [m, byPk.get(String(m.game_pk))]).find(([, p]) => p);
  if (mv) {
    const info = lineMoveInfo(mv[0], mv[1], D.splits);
    rows.push(marketPulseRow(ICON_TREND, "Biggest Line Move", ctxEsc(info.title), info.chg, signCls(info.d), ctxEsc(info.tickets), gameHref(mv[1].sport, mv[1].game_pk)));
  } else rows.push(`<div class="ca-ev-pr ca-ev-pr-empty"><span class="ca-ev-pr-ic">${ICON_TREND}</span><span class="ca-ev-pr-main"><b>Biggest Line Move</b><small>No line moves captured yet.</small></span></div>`);
  const top = [...D.opps].sort((a, b) => b.alpha - a.alpha || b.evPct - a.evPct)[0];
  rows.push(top ? marketPulseRow(ICON_CLOCK, "Highest Confidence Edge", ctxEsc(`${oppLabel(top, D.lineBy)} (${oddsStr(top.odds)})`), pStr(top.edgePp), "pos", `Alpha Score: ${top.alpha}`, gameHref(top.sport, top.game_pk))
    : `<div class="ca-ev-pr ca-ev-pr-empty"><span class="ca-ev-pr-ic">${ICON_CLOCK}</span><span class="ca-ev-pr-main"><b>Highest Confidence Edge</b><small>No +EV opportunities on the board.</small></span></div>`);
  const tot = marketMispricedTotal(D.preds, D.nowMs);
  if (tot) {
    const p = tot.pred;
    rows.push(marketPulseRow(ICON_CLOCK, "Most Mispriced Total", ctxEsc(`${shortMatchup(p.away_team_name, p.home_team_name, p.sport)} O/U ${tot.market}`), `${signedStr(tot.diff, 1)} pts`, signCls(tot.diff), `Model: ${tot.model.toFixed(1)}`, gameHref(p.sport, p.game_pk)));
  } else rows.push(`<div class="ca-ev-pr ca-ev-pr-empty"><span class="ca-ev-pr-ic">${ICON_CLOCK}</span><span class="ca-ev-pr-main"><b>Most Mispriced Total</b><small>No model totals vs. market totals yet.</small></span></div>`);
  const clv = marketMeanClv(D.graded, addDays(D.today, -6), D.today);
  rows.push(clv ? marketPulseRow(ICON_WAVE, "Average Market Divergence", `Graded +EV picks this week (${clv.n})`, pStr(clv.pct), signCls(clv.pct), "vs. closing lines", "track-record.html")
    : `<div class="ca-ev-pr ca-ev-pr-empty"><span class="ca-ev-pr-ic">${ICON_WAVE}</span><span class="ca-ev-pr-main"><b>Average Market Divergence</b><small>No graded +EV picks this week yet.</small></span></div>`);
  return rows.join("");
}

/* ── Top Alpha rows (the +EV rail and the sport pages' Top Alpha Edges card) ───────────────────────────────────────────
   Moved from ev.js: one row = rank box, team logos, the bet, "model/fair prob vs implied prob" and the green edge box. The 4th cell
   defaults to the best price (+EV); `aux` replaces it (the sport page's W / L / P result chip, "" = no 4th cell) and `at` puts an
   "@" between the logos. `o` = an Opportunity (toOpportunity), `lineBy` = evBestLines. */
function topAlphaRow(o, i, lineBy, { aux, at = false } = {}) {
  const cell = aux === undefined ? `<span class="ca-ev-ta-odds">${oddsStr(o.odds)}</span>` : aux ? `<span class="ca-ev-ta-odds">${aux}</span>` : "";
  const [away, home] = matchupSides(o.matchup), logos = at ? `${logoImg(away, o.sport)}<i class="ca-ev-at">@</i>${logoImg(home, o.sport)}` : logoPair(o.matchup, o.sport);
  return `<a class="ca-ev-ta${cell ? "" : " ca-ev-ta-noaux"}" href="${gameHref(o.sport, o.game_pk)}"><span class="ca-ev-rank">#${i + 1}</span><span class="ca-ev-logos">${logos}</span>
    <span class="ca-ev-ta-main"><b class="ca-ell" title="${ctxEsc(oppLabel(o, lineBy))}">${ctxEsc(oppLabel(o, lineBy))}</b><small>${oppProb(o)} vs ${pct1(o.impliedProb)}</small></span>
    ${cell}<span class="ca-ev-edge"><b>${pStr(o.edgePp)}</b>Edge</span></a>`;
}

/* ── NFL divisions (same map as src/sportsmodel/nfl/trends.py, keyed by the site's team code: teamShort(name, "nfl")) ─── */
const NFL_DIVISION = {
  ...Object.fromEntries(["BUF", "MIA", "NE", "NYJ"].map((t) => [t, "AFC East"])), ...Object.fromEntries(["BAL", "CIN", "CLE", "PIT"].map((t) => [t, "AFC North"])),
  ...Object.fromEntries(["HOU", "IND", "JAX", "TEN"].map((t) => [t, "AFC South"])), ...Object.fromEntries(["DEN", "KC", "LV", "LAC"].map((t) => [t, "AFC West"])),
  ...Object.fromEntries(["DAL", "NYG", "PHI", "WSH"].map((t) => [t, "NFC East"])), ...Object.fromEntries(["CHI", "DET", "GB", "MIN"].map((t) => [t, "NFC North"])),
  ...Object.fromEntries(["ATL", "CAR", "NO", "TB"].map((t) => [t, "NFC South"])), ...Object.fromEntries(["ARI", "LAR", "SF", "SEA"].map((t) => [t, "NFC West"])),
};
// The division of an NFL team name ("Buffalo Bills" -> "AFC East"), "" when unknown.
const nflDivision = (name) => NFL_DIVISION[teamShort(name, "nfl")] || "";
