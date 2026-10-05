/* Sport pages, left column (nfl.html / cfb.html / mlb.html / nba.html): Season Performance, Model vs. Market, Market Intelligence and
   Key Insights. Real data only: every number is computed from graded rows, every card has an honest empty state, nothing is invented.
   Mounts into board.js: BOARD_CARDS[id] = (D) => html for the four BOARD_SLOTS.left ids, one BOARD_LOADERS entry that puts everything
   this column needs into D.left (memoised for a minute per sport: a period change re-renders the page but the season's record does not change).
   Season Performance: the model's graded picks inside the published record (track_record_start), tallied by the same trackRows /
   trackTally the Track Record page uses (so W-L and win % equal Track Record's "Performance by League" row for the league), units from
   prediction_pnl_daily ($10 a bet, / 10 = units: the Track Record profit tracker's source, ROI = pnl / (decided bets x $10)). It shows the
   SELECTED PERIOD: a past week / day opens on that week's (day's) numbers, the current period opens on the season; a Week / Season toggle
   under the title switches, and the title and caption always name the period the numbers cover.
   Model vs. Market: graded games binned by how far the model's number was from the market's (x), against how often the home / over side
   happened (y), see blMvm. Market Intelligence: the +EV Market Pulse rows (data.js marketPulseRows) for this sport only. Key Insights:
   up to four rows computed from the sport's graded record, each shown only with a sample of at least 8 games.
   Depends on app.js (sb, sbAll, predictions, evBestLines, evResultsRows, evGradedPicks, trackRecordStarts, inTrackRecord, pnlAgg, uStr, pStr, pct1,
   numOrNull, CFB_TEAM_ID, rkConfId), metrics.js, ui.js and data.js, and board.js (BOARD_*, boardStatusText, boardName). */
const BL_MIN_MVM = 10;      // fewer graded points than this -> the Model vs. Market empty state
const BL_MIN_INSIGHT = 8;   // a Key Insights row needs at least this many DECIDED picks (wins + losses, pushes do not count)
const BL_TTL = 60000;
const BL_SHORT_REST = 5;    // NFL: a team with this many days (or fewer) since its last game is on a short week
const BL_MM = [["spread", "Spread", "Spread"], ["moneyline", "Moneyline", "ML"], ["total", "Total", "Total"], ["props", "Player Props", "Props"]];
const BL_CACHE = {};

/* ── pure helpers ─────────────────────────────────────────────────────── */
// Start of the season the scope runs over: Aug 1 of the season's year for the live leagues (Track Record's "Season" range); a sport
// with no live model has no season window (its whole record, if any).
const blSeasonStart = (sport, today) => (LIVE_SPORTS.includes(sport) ? `${+String(today).slice(5, 7) >= 8 ? String(today).slice(0, 4) : +String(today).slice(0, 4) - 1}-08-01` : "0000-00-00");
// One row per graded pick inside the published record (Track Record's scope: inRecord, not the archive).
const blRecordRows = (acc, closingMap, starts) => trackRows(acc, [], closingMap).filter((r) => inTrackRecord(starts, r.sport, r.date));
const blAccInRecord = (acc, starts) => (acc || []).filter((r) => r && r.actual_winner != null && inTrackRecord(starts, r.sport, r.game_date));

// Mean gap (pp) between the model's win probability and the market's no-vig closing probability, over the moneyline picks with both closing prices.
function blEdge(rows, closing) {
  const v = [];
  for (const r of rows || []) {
    if (r.market !== "moneyline" || r.conf == null || !r.side || !closing) continue;
    const a = americanToProb(closing.get(`${r.game_pk}|${r.side}`)), b = americanToProb(closing.get(`${r.game_pk}|${r.side === "home" ? "away" : "home"}`));
    if (Number.isFinite(a) && Number.isFinite(b) && a + b > 0) v.push((r.conf - a / (a + b)) * 100);
  }
  return v.length ? { mean: v.reduce((s, x) => s + x, 0) / v.length, n: v.length } : null;
}
// Units per game week of the scope (oldest first, weeks with no bets between the first and last are 0), from prediction_pnl_daily rows.
function blWeekUnits(sport, pnl) {
  const by = new Map(), key = (d) => (BOARD_RANKED.includes(sport) ? weekOf(sport, d) : Math.floor(dayDiff(d, pnl[0].game_date) / 7));
  for (const r of pnl || []) if (r && finite(r.pnl)) by.set(key(r.game_date), (by.get(key(r.game_date)) || 0) + +r.pnl);
  const ks = [...by.keys()].sort((a, b) => a - b);
  return ks.length ? Array.from({ length: ks[ks.length - 1] - ks[0] + 1 }, (_, i) => unitsFromPnl(by.get(ks[0] + i) || 0)).slice(-10) : [];
}
// The numbers of one scope [from, to] (YYYY-MM-DD, inclusive): W-L-P + win % (trackTally of the record rows), units / ROI (prediction_pnl_daily:
// pnl / 10, ROI = pnl over the decided bets x $10), the mean ML edge, and the units per week for the bars.
function blPerf(sport, rows, pnl, closing, from, to) {
  const rs = (rows || []).filter((r) => r.date >= from && r.date <= to), t = trackTally(rs);
  const pr = (pnl || []).filter((r) => r && r.game_date >= from && r.game_date <= to), a = pnlAgg(pr), decided = a.w + a.l;
  return { ...t, graded: new Set(rs.map((r) => String(r.game_pk))).size, bets: a.n, units: pr.length ? unitsFromPnl(a.pnl) : null,
    roi: decided ? a.pnl / (decided * 10) * 100 : null, edge: blEdge(rs, closing), weeks: pr.length ? blWeekUnits(sport, pr) : [] };
}

// ── Season Performance's period ──
const blKey = (per) => `${per.kind}:${per.kind === "week" ? per.week : per.date}`;
const blPeriodWord = (per) => (per.kind === "week" ? `Week ${per.week}` : shortDate(per.date));
// Which scope a card shows: `pick` (the user's choice for THIS period, "period" | "season" | null), else the period itself once it is over and the
// season otherwise. Season Performance and the right column's Model Projections each keep their own pick.
function blScopeOf(D, pick) {
  const per = D.period, over = per.to < D.today, key = pick || (over ? "period" : "season");
  const season = blSeasonStart(D.sport, D.today);
  if (key === "period") return { key, from: per.from, to: per.to, word: blPeriodWord(per), caption: rangeLabel(per.from, per.kind === "week" ? per.to : per.from) };
  const ds = ((D.left && D.left.rec) || []).map((r) => r.date).filter((d) => d >= season && d <= D.today).sort(), name = LIVE_SPORTS.includes(D.sport) ? `Season ${seasonOf(D.today)}` : "Published record";
  return { key, from: season, to: D.today, word: "Season", caption: ds.length ? `${name} · ${rangeLabel(ds[0], ds[ds.length - 1])}` : name };
}
function blScope(D) { const st = blState(); return blScopeOf(D, st.perfFor === blKey(D.period) ? st.perf : null); }
// The sentence for a scope with no graded picks: a week before the record restart (archived), a period with none yet, or an empty record.
function blNoPicksMsg(D, L, sc) {
  const name = boardName(D.sport), rs = L.starts && L.starts.get ? L.starts.get(D.sport) : null, since = rs && rs.starts_at ? etDateStr(rs.starts_at) : "";
  if (sc.key === "period" && since && sc.to < since) return `${sc.word} is before the published ${name} record, which restarted ${fullDate(since)}${rs.model_version === "nfl-sim-ml-v2" ? " with the ML v2 model" : ""}. Earlier results are archived.`;
  return sc.key === "period" ? `No graded ${name} picks in ${sc.word} yet.` : `No graded ${name} picks in the published record yet.`;
}

// ── Model vs. Market ──
// The graded points of a market: [{ x (model minus market), y (1 when the home side / the over happened, else 0), m (the market's own home-win probability, moneyline) }].
//   spread: x = model home margin + market spread (pts; + = the model likes the home side more), y = home covered (pushes dropped)
//   total: x = model total - market total (pts), y = went over (pushes dropped)
//   moneyline: x = model home win prob - the market's no-vig closing prob (pp), y = home won (needs both closing prices)
//   props (NFL): x = (sim projection - line) / line (%), y = went over (a lean over that hit, or a lean under that missed)
function blMvmPoints(market, L) {
  const out = [];
  if (market === "props") {
    for (const r of L.props || []) {
      const pj = numOrNull(r.projection), ln = numOrNull(r.line);
      if (pj == null || ln == null || ln === 0 || (r.result !== "hit" && r.result !== "miss") || (r.lean !== "over" && r.lean !== "under")) continue;
      out.push({ x: (pj - ln) / ln * 100, y: (r.lean === "over") === (r.result === "hit") ? 1 : 0 });
    }
    return out;
  }
  for (const r of L.accRec || []) {
    const am = numOrNull(r.actual_margin), at = numOrNull(r.actual_total);
    if (market === "spread") {
      const ms = numOrNull(r.market_spread), pm = numOrNull(r.pred_margin);
      if (ms == null || pm == null || am == null || am + ms === 0) continue;
      out.push({ x: pm + ms, y: am + ms > 0 ? 1 : 0 });
    } else if (market === "total") {
      const mt = numOrNull(r.market_total), pt = numOrNull(r.pred_total);
      if (mt == null || pt == null || at == null || at === mt) continue;
      out.push({ x: pt - mt, y: at > mt ? 1 : 0 });
    } else if (market === "moneyline") {
      const wp = numOrNull(r.win_prob), conf = wp == null ? null : Math.max(wp, 1 - wp);
      const pHome = conf == null ? null : r.predicted_winner === r.home_team_name ? conf : r.predicted_winner === r.away_team_name ? 1 - conf : wp;
      const h = americanToProb(L.closing && L.closing.get(`${r.game_pk}|home`)), a = americanToProb(L.closing && L.closing.get(`${r.game_pk}|away`));
      if (pHome == null || !Number.isFinite(h) || !Number.isFinite(a) || h + a <= 0 || (r.actual_winner !== r.home_team_name && r.actual_winner !== r.away_team_name)) continue;
      const mkt = h / (h + a);
      out.push({ x: (pHome - mkt) * 100, y: r.actual_winner === r.home_team_name ? 1 : 0, m: mkt * 100 });
    }
  }
  return out;
}
// Equal-count bins (3..8 of them, about 8+ games each): one dot per bin at (mean x, outcome rate %). `mean` = mean |x| over all points
// (the callout's Model Edge). null below BL_MIN_MVM points.
function blMvmBins(pts) {
  const n = pts.length;
  if (n < BL_MIN_MVM) return null;
  const s = pts.slice().sort((a, b) => a.x - b.x), k = clip(Math.floor(n / 8), 3, 8), mean = (v) => v.reduce((t, x) => t + x, 0) / v.length;
  const bins = Array.from({ length: k }, (_, i) => s.slice(Math.round(i * n / k), Math.round((i + 1) * n / k))).filter((b) => b.length)
    .map((b) => ({ n: b.length, x: mean(b.map((p) => p.x)), y: mean(b.map((p) => p.y)) * 100, m: b.every((p) => p.m != null) ? mean(b.map((p) => p.m)) : null }));
  return { n, bins, mean: mean(pts.map((p) => Math.abs(p.x))) };
}
const BL_MVM = {
  spread: { unit: " pts", edgeWord: "points", hit: "the home team covered", tip: "x = the model's home margin plus the market spread, in points: positive means the model likes the home team more than the market does. y = how often the home team actually covered, in games grouped by that gap. The dashed line is the 50% a fair market line implies. Model Edge = the average size of that gap over every graded game this season." },
  moneyline: { unit: " pp", edgeWord: "prob. points", hit: "the home team won", tip: "x = the model's home win probability minus the market's no-vig closing probability, in percentage points. y = how often the home team actually won, in games grouped by that gap. The dashed line is the market's own average win probability for the same games. Model Edge = the average size of that gap over every graded game this season." },
  total: { unit: " pts", edgeWord: "points", hit: "the game went over", tip: "x = the model's total minus the market total, in points: positive means the model expects more scoring. y = how often the game went over, in games grouped by that gap. The dashed line is the 50% a fair market total implies. Model Edge = the average size of that gap over every graded game this season." },
  props: { unit: "%", edgeWord: "%", hit: "the prop went over", tip: "x = the sim's projection minus the prop line, as a percentage of the line (NFL player props). y = how often the prop actually went over, in props grouped by that gap. The dashed line is the 50% a fair line implies. Model Edge = the average size of that gap over every graded prop this season." },
};
// Per market: the graded points and their bins (null bins = too few for a chart).
function blMvmData(L) {
  const out = {};
  for (const [k] of BL_MM) { const pts = blMvmPoints(k, L); out[k] = { n: pts.length, ...(blMvmBins(pts) || { bins: null, mean: null }) }; }
  return out;
}

// "this season", or "in the published record" when the sport's record starts after the season does (NFL: restarted Sep 29).
function blWhen(L, sport, today) {
  const st = L && L.starts && L.starts.get ? L.starts.get(sport) : null, since = st && st.starts_at ? etDateStr(st.starts_at) : "";
  return since && since > blSeasonStart(sport, today) ? "in the published record" : "this season";
}
// ── Key Insights ──
const blIcon = (paths) => `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;
const BL_ICONS = { home: blIcon(`<path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>`), under: blIcon(`<path d="M3 7l6 6 4-4 8 8"/><path d="M15 17h6v-6"/>`),
  over: blIcon(`<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>`), group: blIcon(`<circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.4"/><path d="M3 20c0-3.4 2.7-6 6-6s6 2.6 6 6"/><path d="M15.5 14.2c3 0 5.5 2 5.5 5"/>`),
  short: blIcon(`<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>`) };
// Conference of a CFB team name from power_rankings_current (team = ESPN id, conf = conference id), "" when unknown.
const blConf = (L, name) => { const id = typeof CFB_TEAM_ID === "object" ? CFB_TEAM_ID[name] : null, c = id == null || !L.conf ? null : L.conf.get(String(id)); return c == null ? "" : c; };
// Per game: the attributes the insights group by. group: NFL division game / CFB conference game (null = unknown); short: either NFL team has <= BL_SHORT_REST days of rest (null = unknown).
function blGameInfo(L, r, sport) {
  const ms = numOrNull(r.market_spread);
  let group = null, short = null;
  if (sport === "nfl") { const a = nflDivision(r.home_team_name), b = nflDivision(r.away_team_name); group = a && b ? a === b : null; }
  else if (sport === "cfb") { const a = blConf(L, r.home_team_name), b = blConf(L, r.away_team_name); group = a && b ? a === b : null; }
  const rest = L.rest && L.rest.get(String(r.game_pk));
  if (rest) { const v = [rest.home, rest.away].filter((x) => x != null); short = v.length ? v.some((x) => x <= BL_SHORT_REST) : null; }
  return { homeFav: ms == null || ms === 0 ? null : ms < 0, group, short };
}
// Up to four rows [{ key, icon, title, stat, n, pct }] from the sport's graded record; a row needs a sample of >= BL_MIN_INSIGHT decided picks.
// "Home favorites" = the model's spread picks in games the home team was favoured; Unders / Overs = the model's total picks on that side;
// Divisional / Conference = the model's moneyline accuracy in games between division / conference rivals; Short-week unders (NFL) = under picks
// when a team had <= 5 days of rest (needs team_game_log). Rows are ordered by how far the hit rate is from 50%.
function blInsights(L, sport, when = "this season") {
  const info = new Map((L.accRec || []).map((r) => [String(r.game_pk), blGameInfo(L, r, sport)]));
  const picks = (L.rec || []).filter((r) => info.has(String(r.game_pk)));
  const rec = (t) => `${t.w}-${t.l}${t.p ? `-${t.p}` : ""}`, pc = (t) => `${Math.round(t.pct * 100)}%`;
  const word = (t, up, down, flat) => (t.pct >= 0.55 ? up : t.pct <= 0.45 ? down : flat);
  const defs = [
    ["homefav", "home", (r) => r.market === "spread" && info.get(String(r.game_pk)).homeFav === true, (t) => ["Home favorites", `Model spread picks ${rec(t)} (${pc(t)}) ${when}`]],
    ["unders", "under", (r) => r.market === "total" && r.side === "under", (t) => [word(t, "Unders performing", "Unders struggling", "Unders"), `${pc(t)} hit rate (${rec(t)}) ${when}`]],
    ["overs", "over", (r) => r.market === "total" && r.side === "over", (t) => [word(t, "Overs performing", "Overs struggling", "Overs"), `${pc(t)} hit rate (${rec(t)}) ${when}`]],
  ];
  if (sport === "nfl" || sport === "cfb") defs.push(["group", "group", (r) => r.market === "moneyline" && info.get(String(r.game_pk)).group === true, (t) => [sport === "nfl" ? "Divisional matchups" : "Conference matchups", `${pc(t)} model accuracy (${rec(t)})`]]);
  if (sport === "nfl") defs.push(["short", "short", (r) => r.market === "total" && r.side === "under" && info.get(String(r.game_pk)).short === true, (t) => ["Short-week unders", `${pc(t)} hit rate (${rec(t)}) · a team on ${BL_SHORT_REST} or fewer days' rest`]]);
  const rows = [];
  for (const [key, icon, test, text] of defs) {
    const t = trackTally(picks.filter(test));
    if (t.w + t.l < BL_MIN_INSIGHT) continue;      // decided picks only: a push is neither a win nor a loss
    const [title, stat] = text(t);
    rows.push({ key, icon, title, stat, n: t.w + t.l, pct: t.pct });
  }
  return rows.sort((a, b) => Math.abs(b.pct - 0.5) - Math.abs(a.pct - 0.5) || b.n - a.n).slice(0, 4);
}

/* ── data ─────────────────────────────────────────────────────────────── */
function blState() {
  if (!window.__caBL) window.__caBL = { mm: "spread", perf: null, perfFor: "" };
  return window.__caBL;
}
// The two card controls: the Model vs. Market pill (a market key) and the Season Performance Week / Season toggle (remembered for THIS period only).
const blSetMarket = (key) => { blState().mm = key; };
const blSetScope = (key, periodKey) => { const st = blState(); st.perf = key; st.perfFor = periodKey; };
async function blPulseLoad(sport, nowMs, today, startsP) {
  const [preds, moves, splits, oppsAll, lineBy, results, picks, scope] = await Promise.all([
    predictions(sport).catch(() => []), loadLineMoves().catch(() => []), loadSplits().catch(() => new Map()), loadOpportunities().catch(() => []), evBestLines().catch(() => new Map()),
    evResultsRows().catch(() => []), evGradedPicks().catch(() => []), startsP,
  ]);
  return { preds: preds || [], moves: moves || [], splits, opps: (oppsAll || []).filter((o) => o.tier && o.sport === sport), lineBy, nowMs, today,
    graded: scope.failed ? [] : gradedLinePicks(results, picks, scope.starts).filter((r) => r.sport === sport) };      // no record scope: no graded picks (never the archived ones)
}
// NFL rest days: for each game, the days since each team's previous game of the season (team_game_log), null for a team's first game.
async function blRestLoad(season) {
  const rows = await sbAll(`team_game_log?sport=eq.nfl&season=eq.${season}&select=game_pk,team,venue,date_et&order=team.asc,date_et.asc,game_pk.asc`);
  const by = new Map(), out = new Map();
  for (const r of rows || []) if (r && r.team && r.date_et) (by.get(r.team) || by.set(r.team, []).get(r.team)).push(r);
  by.forEach((gs) => gs.forEach((r, i) => {
    const e = out.get(String(r.game_pk)) || out.set(String(r.game_pk), { home: null, away: null }).get(String(r.game_pk));
    if (r.venue === "home" || r.venue === "away") e[r.venue] = i ? dayDiff(r.date_et, gs[i - 1].date_et) : null;
  }));
  return out;
}
async function blCfbConf() {
  const rows = await sb("power_rankings_current?sport=eq.cfb&select=team,conf");
  return new Map((rows || []).filter((r) => r && r.team != null).map((r) => [String(r.team), rkConfId(r.conf)]).filter(([, c]) => c != null));
}
// Everything the column needs for a sport; each source fails on its own (a failed one leaves its field out and its card says so).
async function blLoadAll(D) {
  const sport = D.sport, L = { sport, starts: null, failed: {}, rec: [], accRec: [], closing: new Map(), pnl: [], props: [], pulse: null, rest: null, conf: null };
  const startsP = trackStarts();      // the record scope, loaded once for the record and the pulse (fails closed)
  const part = (k, p) => p.then((v) => { L[k] = v; }).catch((e) => { L.failed[k] = true; console.error(`board left: ${k} failed to load`, e); });
  const recordP = (async () => {
    const [acc, st] = await Promise.all([sbAll(`prediction_accuracy?sport=eq.${sport}&actual_winner=not.is.null&select=${TRACK_ACC_SELECT}&order=game_date.asc,game_pk.asc`), startsP]);
    if (st.failed) throw new Error("track_record_start");      // fail closed: without the record scope no record number is shown
    L.starts = st.starts;
    const rows = blAccInRecord(acc, st.starts), closing = trackClosingMap(await trackClosing(rows.map((r) => r.game_pk)).catch(() => []));
    L.accRec = rows; L.closing = closing; L.rec = blRecordRows(rows, closing, st.starts);
  })();
  const jobs = [part("record", recordP), part("pnl", sbAll(`prediction_pnl_daily?sport=eq.${sport}&select=*&order=game_date.asc,market.asc`))];
  if (LIVE_SPORTS.includes(sport)) {
    jobs.push(part("pulse", blPulseLoad(sport, D.nowMs, D.today, startsP)));
    if (sport === "nfl") {
      jobs.push(part("props", sbAll(`nfl_prop_pnl?season=eq.${seasonOf(D.today)}&select=game_pk,player_id,market,line,lean,projection,result&order=game_pk.asc,player_id.asc,market.asc,line.asc`)));
      jobs.push(part("rest", blRestLoad(seasonOf(D.today))));
    } else jobs.push(part("conf", blCfbConf()));
  }
  await Promise.all(jobs);
  return L;
}
BOARD_LOADERS.push(async (D) => {
  if (D.view !== "games") return;
  const hit = BL_CACHE[D.sport];
  if (hit && Date.now() - hit.at < BL_TTL) { D.left = await hit.p; return; }
  const p = blLoadAll(D);
  BL_CACHE[D.sport] = { at: Date.now(), p };
  D.left = await p;
});

/* ── cards ────────────────────────────────────────────────────────────── */
const blLive = (D) => LIVE_SPORTS.includes(D.sport);
const blCard = (id, body, cls = "") => `<section class="ca-card ca-board-card ca-bl-card${cls ? ` ${cls}` : ""}" id="${id}">${body}</section>`;
const blHead = (title, right = "") => `<div class="ca-card-head ca-bl-head"><h2>${ctxEsc(title)}</h2>${right}</div>`;
// A card with no data: its title plus one honest sentence (a non-live sport says why).
const blEmpty = (id, title, msg) => blCard(id, `${blHead(title)}${emptyMsg(msg)}`);
// Why a sport has no model, in one short sentence (the paused date is the newest stored projection, D.lastProj).
const blNoModel = (D) => (D.lastProj ? `${boardName(D.sport)} model paused since ${shortDate(D.lastProj)}.` : D.sport === "mlb" ? "MLB model paused." : `No ${boardName(D.sport)} model yet.`);
// A muted one-line caption under a card's title.
const blCap = (text) => `<p class="ca-bl-cap">${ctxEsc(text)}</p>`;
// Model vs. Market and Key Insights always cover the published record / season, whatever period is browsed: off the current period they say so.
const blRecordNote = (D, L) => (D.period.isCurrent ? "" : blCap(`${blWhen(L, D.sport, D.today) === "in the published record" ? "Published record" : "Season record"}, not ${blPeriodWord(D.period)}.`));
const blFailed = (id, title) => blEmpty(id, title, "This panel couldn't load.");

function blSeasonPerf(D) {
  const L = D.left, id = "board-season-perf", name = boardName(D.sport);
  if (!L || L.failed.record || L.failed.pnl) return blFailed(id, "Season Performance");
  if (!L.rec.length && !blLive(D)) return blEmpty(id, "Season Performance", `${blNoModel(D)} No graded ${name} picks in the published record.`);
  const per = D.period, sc = blScope(D), p = blPerf(D.sport, L.rec, L.pnl, L.closing, sc.from, sc.to);
  const title = sc.key === "period" ? `${sc.word} Performance` : "Season Performance";
  const toggle = `<div class="ca-bl-scope">${pills("bl-scope", [["period", ctxEsc(blPeriodWord(per))], ["season", "Season"]], sc.key)}</div>`;
  const head = `${blHead(title, `<a class="ca-link" href="track-record.html"><span class="ca-bl-v">View </span>Track Record →</a>`)}${toggle}<p class="ca-bl-cap">${ctxEsc(sc.caption)}</p>`;
  if (!p.n) return blCard(id, `${head}${emptyMsg(blNoPicksMsg(D, L, sc))}`);
  const pct = p.pct == null ? "—" : `${(p.pct * 100).toFixed(1)}%`;
  const bars = p.weeks.length > 1 ? miniBars(p.weeks, { w: 46, h: 30 }) : "";
  const boxes = [
    statCard({ label: "W - L", value: `<span class="ca-bl-wl">${p.w} - ${p.l}</span><small class="${p.pct != null && p.pct >= 0.5 ? "pos" : ""}">${pct}</small>`, sub: p.p ? `${p.p} push${p.p > 1 ? "es" : ""}` : "no pushes",
      tip: "Moneyline, spread and total picks graded W or L (pushes do not count in the win %): the same record as Track Record's Performance by League." }),
    statCard({ label: "Units", value: p.units == null ? "—" : uStr(p.units), valueClass: p.units == null ? "" : signCls(p.units), sub: p.units == null ? "No bets priced" : `${p.bets} bets`,
      tip: "Profit at the price captured for each pick, 1 unit = $10 a bet (the Track Record profit tracker's source, prediction_pnl_daily)." }),
    statCard({ label: "ROI", value: p.roi == null ? "—" : pStr(p.roi), valueClass: p.roi == null ? "" : signCls(p.roi), visual: bars ? `<span title="Units by week">${bars}</span>` : "",
      tip: "Profit divided by the stake on the decided bets (pushes are refunded)." }),
    statCard({ label: "Avg. Edge", value: p.edge ? pStr(p.edge.mean) : "—", valueClass: p.edge ? signCls(p.edge.mean) : "", sub: p.edge ? `${p.edge.n} ML picks` : "No priced ML picks",
      tip: "Average gap between the model's win probability and the market's no-vig closing probability on its moneyline picks." }),
  ].join("");
  return blCard(id, `${head}<div class="ca-bl-grid">${boxes}</div>`, "ca-bl-perf");
}

const blMmPills = (D, data, active) => {
  const items = BL_MM.filter(([k]) => data[k].n > 0).map(([k, l, s]) => [k, `<span class="ca-bl-l">${l}</span><span class="ca-bl-s">${s}</span>`]);
  return items.length ? `<div class="ca-bl-pills">${pills("bl-mm", items, active)}</div>` : "";
};
function blModelMarket(D) {
  const L = D.left, id = "board-model-market", title = "Model vs. Market";
  if (!L || L.failed.record) return blFailed(id, title);
  const data = blMvmData(L), avail = BL_MM.map(([k]) => k).filter((k) => data[k].n > 0);
  if (!avail.length) return blEmpty(id, title, blLive(D) ? "No graded games with a market line yet." : `${blNoModel(D)} No graded ${boardName(D.sport)} games to compare with the market.`);
  const st = blState(), sel = avail.includes(st.mm) ? st.mm : avail[0], d = data[sel], m = BL_MVM[sel];
  const head = blHead(title, infoTip(sel === "props" ? m.tip : m.tip.replace("this season", blWhen(L, D.sport, D.today))));
  const rn = blRecordNote(D, L);
  if (!d.bins) return blCard(id, `${head}${rn}${blMmPills(D, data, sel)}${emptyMsg(`Not enough graded games yet: ${d.n} of ${BL_MIN_MVM} needed.`)}`);
  const dots = d.bins.map((b) => ({ x: b.x, y: b.y, n: b.n, title: `${b.n} ${sel === "props" ? "props" : "games"} · average gap ${signedStr(b.x, 1, m.unit)} · ${m.hit} ${b.y.toFixed(0)}% of the time` }));
  const isMl = sel === "moneyline", chart = scatterChart({ dots, line: d.bins.map((b) => ({ x: b.x, y: b.y })), base: isMl ? d.bins.map((b) => ({ x: b.x, y: b.m })) : null, flat: isMl ? null : 50 }, { h: 150, xUnit: sel === "props" ? "%" : "" });
  const legend = `<div class="ca-bl-legend"><span><i class="ca-bl-lg-ca"></i>CappingAlpha</span><span><i class="ca-bl-lg-mk"></i>Market</span></div>`;
  const note = `<p class="ca-bl-cap">${ctxEsc(`Share of ${sel === "props" ? "props" : "games"} where ${m.hit}, by the gap between the model and the market`)}</p>`;
  const edge = `<div class="ca-bl-edge"><span>Model Edge</span><b>${(+d.mean).toFixed(1)} ${ctxEsc(m.edgeWord)}</b></div>`;
  return blCard(id, `${head}${rn}${blMmPills(D, data, sel)}${legend}${chart}${note}<div class="ca-bl-edgerow"><span class="muted">${d.n} graded ${sel === "props" ? "props" : "games"}</span>${edge}</div>`);
}

function blMarketIntel(D) {
  const L = D.left, id = "board-market-intel", title = "Market Intelligence";
  if (!blLive(D)) return blEmpty(id, title, `${blNoModel(D)} No live ${boardName(D.sport)} market to read.`);
  if (!L || !L.pulse) return blFailed(id, title);
  const per = D.period, note = per.isCurrent ? "" : blCap(`The live market, not ${blPeriodWord(per)}.`);
  return blCard(id, `${blHead(title)}${note}<div class="ca-bl-pulse">${marketPulseRows(L.pulse)}</div>`, "ca-ev-rail-card");
}

function blKeyInsights(D) {
  const L = D.left, id = "board-key-insights", title = "Key Insights";
  if (!L || L.failed.record) return blFailed(id, title);
  if (!blLive(D) && !L.accRec.length) return blEmpty(id, title, `${blNoModel(D)} No graded games to read insights from.`);
  const rows = blInsights(L, D.sport, blWhen(L, D.sport, D.today));
  if (!rows.length) return blLive(D) ? "" : blEmpty(id, title, `Not enough graded games yet (${BL_MIN_INSIGHT} needed).`);     // a live sport's card is hidden while no row has a sample of 8 games
  return blCard(id, `${blHead(title)}${blRecordNote(D, L)}${rows.map((r) => `<div class="ca-bl-in" data-insight="${ctxEsc(r.key)}"><span class="ca-bl-in-ic ca-bl-in-${ctxEsc(r.key)}">${BL_ICONS[r.icon]}</span><span class="ca-bl-in-t"><b>${ctxEsc(r.title)}</b><small title="${ctxEsc(`${r.stat} · ${r.n} decided picks`)}">${ctxEsc(r.stat)}</small></span></div>`).join("")}`);
}

BOARD_CARDS["board-season-perf"] = blSeasonPerf;
BOARD_CARDS["board-model-market"] = blModelMarket;
BOARD_CARDS["board-market-intel"] = blMarketIntel;
BOARD_CARDS["board-key-insights"] = blKeyInsights;

/* ── interaction ──────────────────────────────────────────────────────── */
// Redraw one card from the cached load (a pill click needs no refetch).
function blRedraw(id, fn) {
  const D = window.__caBoardData, el = typeof document !== "undefined" ? document.getElementById(id) : null;
  if (D && D.view === "games" && el) el.outerHTML = safeCard(id, fn, D, "ca-card ca-board-card", id);
}
function blClick(e) {
  const pill = e.target && e.target.closest ? e.target.closest("[data-pill]") : null, D = window.__caBoardData;
  if (!pill || !D || D.view !== "games") return;
  if (pill.dataset.pill === "bl-mm") { blSetMarket(pill.dataset.key); blRedraw("board-model-market", blModelMarket); }
  else if (pill.dataset.pill === "bl-scope") { blSetScope(pill.dataset.key, blKey(D.period)); blRedraw("board-season-perf", blSeasonPerf); }
}
if (typeof document !== "undefined" && document.addEventListener && !window.__caBLWired) { window.__caBLWired = true; document.addEventListener("click", blClick); }
