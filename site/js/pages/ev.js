/* +EV page (ev.html) — mockup #3. Real data only: every card has an empty state and the mockup's numbers are
   never rendered. Opportunities = current is_pick rows (game lines + props) with a non-null Alpha tier.
   Filters / tab / sort live in window.__caEv AND in the URL query (so a reload or share keeps them) and survive the
   5-minute re-render. The stat cards and the right rail describe the whole board; the filters narrow the table and
   the tab counts. Performance numbers (Model Hit Rate 30D, ROI 30D, Performance by Edge Bucket, Average Market
   Divergence) all come from ONE population: graded +EV picks (ev_pnl_daily / ev_results + ev_prop_results).
   Depends on app.js (sb, sbAll, predictions, evBestLines, evResultsRows, evGradedPicks, propResultsRows, evBestParlays,
   evParlaysCurrent, bestParlaysSection, parlaySection, bookSelected, getSettings, trackRecordStarts,
   inTrackRecord, etDateStr, timeET, logoImg, logoPair, evBookName, ctxEsc, pStr, uStr, pct1, render),
   metrics.js, ui.js, shell.js and data.js. */
const EV_TIERS = ["HIGH", "STRONG", "MEDIUM"];
const EV_SORTS = [["edge", "Highest Edge"], ["ev", "Highest EV"], ["alpha", "Alpha Score"], ["time", "Game Time"]];
const EV_DATES = [["", "All Dates"], ["today", "Today"], ["tomorrow", "Tomorrow"], ["week", "This Week"]];
const EV_MIN_EDGES = [0, 2, 5, 10];
const EV_DEFAULT = { sport: "", market: "", book: "", minEdge: 0, tier: "", date: "", q: "", kind: "all", sort: "edge" };
const EV_PARAMS = [["sport", "sport"], ["market", "market"], ["book", "book"], ["minEdge", "min"], ["tier", "conf"], ["date", "date"], ["q", "q"], ["kind", "tab"], ["sort", "sort"]];

/* ── pure helpers ─────────────────────────────────────────────────────── */
// Narrow opportunities by f = {sport, kind, market, book, minEdge, tier, date, q, sort}; omitted / empty / "all" keys do not
// filter. `date`: "today" | "tomorrow" | "week" (today .. today + 6, ET; `f.today` overrides "today's" date for tests).
// sort: edge | ev | alpha (descending) or time (ascending); none keeps the input order. Never mutates the input.
function evFilter(opps, f = {}) {
  const today = f.today || etDateStr(new Date().toISOString()), q = String(f.q || "").trim().toLowerCase();
  const dateOk = (o) => {
    if (!f.date) return true;
    if (!o.commence) return false;
    const d = etDateStr(o.commence);
    return f.date === "today" ? d === today : f.date === "tomorrow" ? d === addDays(today, 1) : f.date === "week" ? d >= today && d <= addDays(today, 6) : true;
  };
  const out = (opps || []).filter((o) => o
    && (!f.sport || o.sport === f.sport)
    && (!f.kind || f.kind === "all" || o.kind === f.kind)
    && (!f.market || o.market === f.market)
    && (!f.book || String(o.book || "").toLowerCase() === String(f.book).toLowerCase())
    && (f.minEdge == null || f.minEdge === "" || (finite(o.edgePp) && +o.edgePp >= +f.minEdge))
    && (!f.tier || o.tier === f.tier)
    && dateOk(o)
    && (!q || `${o.matchup || ""} ${o.playerName || ""}`.toLowerCase().includes(q)));
  const by = { edge: (a, b) => b.edgePp - a.edgePp, ev: (a, b) => b.evPct - a.evPct, alpha: (a, b) => b.alpha - a.alpha || b.evPct - a.evPct,
    time: (a, b) => (timeMs(a.commence) || Infinity) - (timeMs(b.commence) || Infinity) }[f.sort];
  return by ? out.sort((a, b) => by(a, b) || 0) : out;
}

// edgeBuckets input from graded game lines + graded props: only rows with a known edge AND profit.
function evBucketRows(gradedLines, gradedProps) {
  return [...(gradedLines || []), ...(gradedProps || [])].filter((r) => r && finite(r.edgePp) && finite(r.profitUnits))
    .map((r) => ({ edgePct: +r.edgePp, profitUnits: +r.profitUnits, won: r.won }));
}

// Graded prop +EV picks inside the track record (ev_prop_results, wins and losses only). The flagged price is the stored
// best_price of the matching ev_prop_picks row (same game / player / market / line / model version); with no stored price
// the row has no implied prob or edge (it is then left out of the edge buckets). `profit` is already in units.
function gradedPropPicks(results, picks, starts) {
  const key = (r) => `${r.game_pk}|${r.player_id}|${r.market}|${r.line}|${r.model_version}`;
  const price = new Map((picks || []).map((p) => [key(p), p.best_price]));
  const out = [];
  for (const r of results || []) {
    if (!r || (r.result !== "win" && r.result !== "loss") || !r.commence_time || !inTrackRecord(starts, r.sport, r.commence_time)) continue;
    const imp = americanToProb(price.get(key(r))), implied = Number.isFinite(imp) ? imp : null;
    out.push({ sport: r.sport, game_pk: r.game_pk, market: "prop", won: r.result === "win", implied,
      edgePp: implied != null && finite(r.model_prob) ? (+r.model_prob - implied) * 100 : null, date: etDateStr(r.commence_time),
      profitUnits: finite(r.profit) ? +r.profit : null, clv: finite(r.clv) ? +r.clv : null });
  }
  return out;
}

const evxDays = (days, today) => Array.from({ length: days }, (_, i) => addDays(today, i - (days - 1)));
// P&L units per ET day for the `days` days ending on `today` (ev_pnl_daily rows).
function evxDailyUnits(rows, days, today) {
  return evxDays(days, today).map((date) => ({ date, units: aggPnl((rows || []).filter((r) => r && r.game_date === date)).units }));
}
// Mean flagged edge (pp) of graded picks per game day, null where nothing graded.
function evxEdgeByDay(graded, days, today) {
  return evxDays(days, today).map((date) => {
    const v = (graded || []).filter((r) => r && r.date === date && finite(r.edgePp)).map((r) => +r.edgePp);
    return { date, v: v.length ? v.reduce((s, x) => s + x, 0) / v.length : null };
  });
}
// EV% distribution of the listed opportunities: red below 0, green above.
function evxBins(opps) {
  const edges = [-Infinity, -5, 0, 5, 10, 15, Infinity], labels = ["<-5", "-5–0", "0–5", "5–10", "10–15", ">15"];
  return histogram((opps || []).map((o) => o.evPct), edges).map((b, i) => ({ label: labels[i], n: b.n, color: b.hi <= 0 ? "var(--red)" : "var(--green)" }));
}
/* ── filter state <-> URL ─────────────────────────────────────────────── */
function evxParse(search) {
  let p;
  try { p = new URLSearchParams(search || ""); } catch { p = new URLSearchParams(""); }
  const get = (k) => p.get(k) || "", min = +get("min"), tab = get("tab"), sort = get("sort");
  return { sport: get("sport"), market: get("market"), book: get("book"),
    minEdge: EV_MIN_EDGES.includes(min) ? min : 0,
    tier: EV_TIERS.includes(get("conf")) ? get("conf") : "",
    date: EV_DATES.some(([k]) => k && k === get("date")) ? get("date") : "",
    q: get("q").slice(0, 80), kind: tab === "line" || tab === "prop" ? tab : "all",
    sort: EV_SORTS.some(([k]) => k === sort) ? sort : "edge" };
}
// Only non-default values: an untouched page has a clean URL.
function evxQuery(s) {
  const p = new URLSearchParams();
  for (const [key, name] of EV_PARAMS) if (s[key] !== EV_DEFAULT[key]) p.set(name, String(s[key]));
  return p.toString();
}
function evxState() {
  if (!window.__caEv) { let s = ""; try { s = location.search; } catch { s = ""; } window.__caEv = evxParse(s); }
  return window.__caEv;
}
function evxSync(s) {
  try {
    const u = new URL(location.href), q = evxQuery(s);
    u.search = q;
    history.replaceState(null, "", u.toString());
  } catch { /* keep the current URL */ }
}

/* ── small view helpers ───────────────────────────────────────────────── */
const ICON_REFRESH = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-2.6-6.4"/><path d="M21 4v5h-5"/></svg>`;
const ICON_STAR_BIG = `<svg class="ca-ev-star" width="38" height="38" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.5l2.9 6.2 6.8.8-5 4.7 1.3 6.7L12 17.5l-6 3.4 1.3-6.7-5-4.7 6.8-.8z" fill="#F2A71B"/></svg>`;
const evxStamp = (iso) => `${new Date(iso).toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" })} ${timeET(iso)}`;
const evxBy = (arr, key) => [...new Set((arr || []).map((x) => x[key]).filter((v) => v != null && v !== ""))];

/* ── data ─────────────────────────────────────────────────────────────── */
async function evxLoad() {
  const nowMs = Date.now(), today = etDateStr(new Date(nowMs).toISOString());
  const newest = (path) => sb(`${path}?is_pick=eq.true&select=created_at&order=created_at.desc&limit=1`).catch(() => []);
  const [predsBy, oppsAll, lineBy, hist, evPnl, results, picks, propRes, propPicks, starts, moves, splits, lastLine, lastProp, parlaysAll, ...parlaysBy] = await Promise.all([
    Promise.all(SPORTS.map((s) => predictions(s).catch(() => []))),
    loadOpportunities().catch(() => []),
    evBestLines().catch(() => new Map()),
    loadEvHistory(8).catch(() => []),
    sb("ev_pnl_daily?select=*").catch(() => []),
    evResultsRows().catch(() => []),
    evGradedPicks().catch(() => []),
    propResultsRows().catch(() => []),
    // every prop pick's stored price (graded props join on it): paged past PostgREST's 1,000-row clamp, order ends on the primary key
    sbAll("ev_prop_picks?is_pick=eq.true&select=game_pk,player_id,market,line,model_version,best_price,created_at&order=created_at.asc,game_pk.asc,player_id.asc,market.asc,line.asc,model_version.asc").catch(() => []),
    trackRecordStarts().catch(() => new Map()),
    loadLineMoves().catch(() => []),
    loadSplits().catch(() => new Map()),
    newest("ev_picks"), newest("ev_prop_picks"),
    evBestParlays().catch(() => []),
    ...LIVE_SPORTS.map((s) => evParlaysCurrent(s).catch(() => [])),
  ]);
  const opps = (oppsAll || []).filter((o) => o.tier);
  const gradedLines = gradedLinePicks(results, picks, starts), gradedProps = gradedPropPicks(propRes, propPicks, starts);
  const stamps = [...(lastLine || []), ...(lastProp || [])].map((r) => r && r.created_at).filter(Boolean).sort();
  return { nowMs, today, preds: (predsBy || []).flat(), opps, lineBy, hist: hist || [], pnl: evPnl || [],
    gradedLines, gradedProps, graded: [...gradedLines, ...gradedProps], moves: moves || [], splits, updated: stamps.length ? stamps[stamps.length - 1] : null,
    parlays: parlaysAll || [], sportParlays: LIVE_SPORTS.map((s, i) => [s, parlaysBy[i] || []]) };
}

/* ── stat cards ───────────────────────────────────────────────────────── */
function evxStatTotal(D) {
  // R14: "N new today" = picks first flagged today (ET); the bars are new picks per day, last 7 days.
  const c = dailyCounts(D.hist, (r) => r.date, 7, D.today), n = c[c.length - 1].n;
  return statCard({ label: "Total +EV Opportunities", value: String(D.opps.length), valueClass: "",
    sub: `<span title="Picks first flagged today (ET), all sports, game lines and props"><span class="${n ? "pos" : ""}">${n ? "▲ " : ""}${n} new today</span></span>`,
    visual: c.some((x) => x.n > 0) ? `<span title="New picks per day, last 7 days">${miniBars(c.map((x) => x.n), { w: 56, h: 46 })}</span>` : "" });
}
function evxStatAvg(D) {
  const edges = D.opps.map((o) => o.edgePp).filter(finite), mean = edges.length ? edges.reduce((s, x) => s + +x, 0) / edges.length : null;
  if (mean == null) return statCard({ label: "Average Edge", value: "—", sub: "No +EV opportunities on the board." });
  const daily = evxEdgeByDay(D.graded, 7, D.today).map((x) => x.v), spark = sparkline(daily, { w: 72, h: 40 });
  return statCard({ label: "Average Edge", value: pStr(mean), valueClass: signCls(mean), sub: "Across all opportunities",
    visual: spark ? `<span title="Average flagged edge of graded +EV picks per game day, last 7 days">${spark}</span>` : "" });
}
function evxStatHighest(D) {
  const best = [...D.opps].sort((a, b) => b.edgePp - a.edgePp)[0];
  if (!best) return statCard({ label: "Highest Edge", value: "—", sub: "No +EV opportunities on the board." });
  const [away, home] = matchupSides(best.matchup);
  const mu = shortMatchup(away, home, best.sport), pick = pickLabel(best, D.lineBy);
  // ONE line (ellipsised, full text in the title): "Lions ML vs. Chiefs", "Chris Olave Under 85.5 Rec Yds", "Over 47.5 · Falcons @ Saints"
  const line = best.kind === "prop" ? `${best.playerName || ""} ${pick}`.trim()
    : best.market === "total" ? `${pick} · ${mu}` : `${pick} vs. ${shortTeam(best.side === "home" ? away : home, best.sport)}`;
  return statCard({ label: "Highest Edge", value: pStr(best.edgePp), valueClass: "pos",
    sub: `<a class="ca-ev-hi" href="${gameHref(best.sport, best.game_pk)}" title="${ctxEsc(`${line} · ${mu}`)}"><span class="ca-ell">${ctxEsc(line)}</span></a>`, visual: ICON_STAR_BIG });
}
function evxStatHit(D) {
  const p = perfWindow(D.pnl, D.graded, addDays(D.today, -29), D.today);
  return statCard({ label: "Model Hit Rate", labelNote: "(Last 30 Days)", labelNoteShort: "(30D)", value: p.hitRate == null ? "—" : `${(p.hitRate * 100).toFixed(1)}%`,
    sub: p.hitRate == null ? "No graded +EV picks in 30 days." : `<span class="muted">${p.wins}-${p.losses} · ${p.n} graded</span>`,
    tip: p.hitRate == null ? "" : `${p.wins}-${p.losses} · ${p.n} graded +EV picks in the last 30 days (all markets): wins / (wins + losses).`,
    visual: p.hitRate == null ? "" : donut(p.hitRate, { size: 66, stroke: 10 }) });
}
function evxStatRoi(D) {
  const p = perfWindow(D.pnl, D.graded, addDays(D.today, -29), D.today), bars = evxDailyUnits(D.pnl, 7, D.today);
  if (!p.n) return statCard({ label: "ROI", labelNote: "(Last 30 Days)", labelNoteShort: "(30D)", value: "—", sub: "No graded +EV picks in 30 days." });
  return statCard({ label: "ROI", labelNote: "(Last 30 Days)", labelNoteShort: "(30D)", value: `${pStr(p.roiPct)}<small class="${signCls(p.units)}">${uStr(p.units)}</small>`, valueClass: signCls(p.roiPct),
    tip: `${p.n} graded +EV picks in the last 30 days (all markets); units staked at the flagged price.`,
    visual: bars.some((x) => x.units) ? `<span title="Daily units, last 7 days">${miniBars(bars.map((x) => x.units), { w: 56, h: 46 })}</span>` : "" });
}
const EV_STATS = [["Total +EV Opportunities", evxStatTotal], ["Average Edge", evxStatAvg], ["Highest Edge", evxStatHighest], ["Model Hit Rate", evxStatHit], ["ROI", evxStatRoi]];
const evxStatCards = (D) => `<div class="ca-stats ca-ev-stats">${EV_STATS.map(([name, fn]) => safeCard(name, fn, D, "ca-card ca-stat")).join("")}</div>`;

/* ── filter card ──────────────────────────────────────────────────────── */
const evxSelect = (key, label, options, value, extraCls = "") => selectField("data-ev", key, label, options, value, extraCls);
function evxFilterCard(D) {
  const s = evxState();
  const sports = [...new Set([...LIVE_SPORTS, ...evxBy(D.opps, "sport")])].map((x) => [x, SPORT_NAME[x] || String(x).toUpperCase()]);
  const markets = [["moneyline", "Moneyline"], ["spread", "Spread"], ["total", "Total"], ...evxBy(D.opps.filter((o) => o.kind === "prop"), "market").sort().map((m) => [m, PROP_LABEL[m] || m])];
  const books = evxBy(D.opps, "book").sort().map((b) => [b, evBookName(b)]);
  return `<section class="ca-card ca-ev-filters" id="ev-filters"><div class="ca-ev-fgrid">
    ${evxSelect("sport", "Sport", [["", "All Sports"], ...sports], s.sport)}${evxSelect("league", "League", [["", "All Leagues"], ...sports], s.sport)}
    ${evxSelect("market", "Market Type", [["", "All Markets"], ...markets], s.market)}${evxSelect("book", "Sportsbook", [["", "All Books"], ...books], s.book)}
    ${evxSelect("minEdge", "Minimum Edge", EV_MIN_EDGES.map((m) => [m, `≥ ${m}%`]), s.minEdge)}${evxSelect("tier", "Confidence", [["", "All Confidence"], ...EV_TIERS.map((t) => [t, t])], s.tier)}
    ${evxSelect("date", "Date", EV_DATES, s.date, "ca-ev-date")}</div></section>`;
}
// E4: the search box is its own card to the right of the filter card (same row, equal height).
const evxSearchCard = () => searchCard("ev-search", "data-ev-q", evxState().q);

/* ── tabs + table ─────────────────────────────────────────────────────── */
// R19 detail behind the subtitle's (i): which probability the Model Prob. column shows.
const EV_SUB_TIP = "Edges across all sports: model probability for props, sharp fair price (Pinnacle no-vig) for game lines, with current odds and expected value.";
const evxFilters = (s, extra = {}) => ({ ...s, ...extra });
function evxTabs(D) {
  const s = evxState(), base = evFilter(D.opps, evxFilters(s, { kind: "all", sort: "" }));
  const n = (k) => base.filter((o) => k === "all" || o.kind === k).length;
  return `<div id="ev-tabs">${pills("ev-tab", [["all", `All Opportunities (${n("all")})`], ["line", `Game Lines (${n("line")})`], ["prop", `Player Props (${n("prop")})`]], s.kind)}</div>`;
}
function evxRow(o, i, D) {
  const [away, home] = matchupSides(o.matchup), pick = pickLabel(o, D.lineBy);
  const who = o.kind === "prop"
    ? `<span class="ca-ev-player">${o.playerName ? starButton("players", o.playerName, o.playerName) : ""}<span class="ca-ev-who" title="${ctxEsc(`${o.playerName || ""} · ${shortMatchup(away, home, o.sport)}`)}"><b class="ca-ell">${ctxEsc(o.playerName || "")}</b><small class="ca-ell">${ctxEsc(shortMatchup(away, home, o.sport))}</small></span></span>`
    : `<span class="ca-team" title="${ctxEsc(shortMatchup(away, home, o.sport))}">${oppLogo(o) || logoImg(away, o.sport)}<span class="ca-ell">${ctxEsc(shortMatchup(away, home, o.sport))}</span></span>`;
  return `<tr data-href="${gameHref(o.sport, o.game_pk)}"><td class="muted">${i + 1}</td><td><span class="ca-team" title="${ctxEsc(SPORT_NAME[o.sport] || "")}">${leagueLogo(o.sport)}<span class="ca-ev-sport-t">${SPORT_NAME[o.sport] || ""}</span></span></td><td>${who}</td>
    <td><span class="ca-ell ca-ev-mkt" title="${ctxEsc(pick)}">${ctxEsc(pick)}</span></td><td><span class="ca-dash-odds">${bookBadge(o.book)}${oddsStr(o.odds)}</span></td>
    <td class="ca-ev-prob">${oppProb(o)}</td><td>${pct1(o.impliedProb)}</td><td class="${signCls(o.edgePp)} ca-b">${pStr(o.edgePp)}</td><td class="${signCls(o.evPct)} ca-b">${pStr(o.evPct)}</td>
    <td>${alphaCell(o.alpha)}</td><td>${confPill(o.tier)}</td><td class="ca-dash-gtime">${kickLabel(o.commence)}</td><td>${goLink(o.sport, o.game_pk)}</td></tr>`;
}
function evxTableCard(D) {
  const s = evxState(), list = evFilter(D.opps, evxFilters(s));
  let body;
  if (!list.length) {
    const filtered = ["sport", "market", "book", "minEdge", "tier", "date", "q", "kind"].some((k) => s[k] !== EV_DEFAULT[k]);
    body = !D.opps.length ? emptyMsg(SPORT_STATUS[s.sport] || "No +EV opportunities on the board right now.")
      : `${emptyMsg(SPORT_STATUS[s.sport] || "No opportunities match these filters.")}${filtered ? `<p class="ca-ev-reset"><button class="ca-btn" data-ev-reset>Reset filters</button></p>` : ""}`;
  } else {
    body = `<div class="ca-table-wrap"><table class="ca-table ca-dash-table ca-ev-table"><thead><tr><th>#</th><th>Sport</th><th>Matchup / Player</th><th>Market</th><th>Best Odds</th>${oppProbTh()}<th>Impl. Prob.</th><th>Edge</th><th>EV</th><th>Alpha Score</th><th>Confidence</th><th>Game Time</th><th>View</th></tr></thead><tbody>${list.map((o, i) => evxRow(o, i, D)).join("")}</tbody></table></div>`;
  }
  return `<section class="ca-card ca-ev-tablecard" id="ev-table"><div class="ca-card-head"><h2>+EV Opportunities</h2><p title="${ctxEsc(EV_SUB_TIP)}">Model projections, current odds, and expected value across all sports.</p>${infoTip(EV_SUB_TIP)}</div>${body}</section>`;
}

/* ── right rail ───────────────────────────────────────────────────────── */
function evxTopAlpha(D) {
  const top = [...D.opps].sort((a, b) => b.alpha - a.alpha || b.evPct - a.evPct).slice(0, 5);
  const rows = top.map((o, i) => `<a class="ca-ev-ta" href="${gameHref(o.sport, o.game_pk)}"><span class="ca-ev-rank">#${i + 1}</span><span class="ca-ev-logos">${logoPair(o.matchup, o.sport)}</span>
    <span class="ca-ev-ta-main"><b class="ca-ell">${ctxEsc(oppLabel(o, D.lineBy))}</b><small>${oppProb(o)} vs ${pct1(o.impliedProb)}</small></span>
    <span class="ca-ev-ta-odds">${oddsStr(o.odds)}</span><span class="ca-ev-edge"><b>${pStr(o.edgePp)}</b>Edge</span></a>`).join("");
  return `<section class="ca-card ca-ev-rail-card" id="ev-top"><div class="ca-card-head"><h2>Top Alpha Opportunities</h2><a class="ca-link" href="ev.html?sort=alpha">View All →</a></div>${rows || emptyMsg("No +EV opportunities on the board.")}</section>`;
}
function evxPulse(D) {
  return `<section class="ca-card ca-ev-rail-card" id="ev-pulse"><div class="ca-card-head"><h2>Market Pulse</h2><a class="ca-link" href="index.html">View All →</a></div>${marketPulseRows(D)}</section>`;
}
function evxDist(D) {
  const chart = histogramChart(evxBins(D.opps), { w: 170, h: 150 });
  return `<section class="ca-card ca-ev-rail-card" id="ev-dist"><div class="ca-card-head"><h2>EV Distribution</h2></div>${chart ? `${chart}<p class="ca-ev-axis">Expected Value (%)</p>` : emptyMsg("No +EV opportunities on the board.")}</section>`;
}
function evxBuckets(D) {
  const rows = evBucketRows(D.gradedLines, D.gradedProps), b = edgeBuckets(rows, [10, 5, 2, 0]);
  const body = rows.length
    ? `<table class="ca-table ca-ev-bt"><thead><tr><th>Edge Range</th><th>n</th><th>Hits</th><th>ROI</th><th>Units</th></tr></thead><tbody>${b.map((x) => x.n
      ? `<tr title="${x.hits} of ${x.n} graded picks"><td>${ctxEsc(x.label)}</td><td class="muted">${x.n}</td><td>${Math.round(x.hits / x.n * 100)}%</td><td class="${signCls(x.roiPct)} ca-b">${pStr(x.roiPct)}</td><td class="${signCls(x.units)} ca-b">${uStr(x.units)}</td></tr>`
      : `<tr class="muted"><td>${ctxEsc(x.label)}</td><td>0</td><td>—</td><td>—</td><td>—</td></tr>`).join("")}</tbody></table>
      <p class="ca-cap">${rows.length} graded +EV picks (game lines + props) with a stored flagged price and edge.</p>`
    : emptyMsg("No graded +EV picks with a flagged price yet.");
  return `<section class="ca-card ca-ev-rail-card" id="ev-buckets"><div class="ca-card-head"><h2>Performance by Edge Bucket</h2></div>${body}</section>`;
}
const EV_RAIL = [["Top Alpha Opportunities", evxTopAlpha, "ev-top"], ["Market Pulse", evxPulse, "ev-pulse"], ["EV Distribution", evxDist, "ev-dist"], ["Performance by Edge Bucket", evxBuckets, "ev-buckets"]];
const evxRailCard = (D, i) => safeCard(EV_RAIL[i][0], EV_RAIL[i][1], D, "ca-card ca-ev-rail-card", EV_RAIL[i][2]);

/* ── legacy sections kept below the table ─────────────────────────────── */
function evxParlays(D) {
  const s = getSettings(), mine = (D.parlays || []).filter((p) => bookSelected(p.book, s) && +p.ev >= s.minEv / 100);
  return `${bestParlaysSection(mine, s)}${D.sportParlays.map(([, rows]) => parlaySection(rows)).join("")}`;
}

/* ── page ─────────────────────────────────────────────────────────────── */
async function buildEvPage() {
  evxState();
  const D = await evxLoad();
  window.__caEvData = D;
  // E5: Last Updated = newest created_at among the current is_pick rows of ev_picks / ev_prop_picks (the last model build); hidden when there is none.
  const right = `${D.updated ? `<div class="ca-ev-updated"><span>Last Updated</span><b>${ctxEsc(evxStamp(D.updated))}</b></div>` : ""}<button class="ca-dn-btn ca-ev-refresh" data-ev-refresh aria-label="Refresh" title="Reload the page data">${ICON_REFRESH}</button>`;
  const sortSel = `<label class="ca-ev-sort"><span>Sort By</span><select class="ca-select" data-ev="sort">${EV_SORTS.map(([k, l]) => `<option value="${k}"${k === evxState().sort ? " selected" : ""}>${l}</option>`).join("")}</select></label>`;
  return `<div class="ca-ev">${pageTitle("+EV", "Find the best expected value opportunities across all sports, powered by the CappingAlpha model.", right)}
    ${evxStatCards(D)}<div class="ca-ev-filterrow">${safeCard("Filters", evxFilterCard, D, "ca-card ca-ev-filters", "ev-filters")}${safeCard("Search", evxSearchCard, D, "ca-card ca-sfind", "ev-search")}</div>
    <div class="ca-ev-main"><div class="ca-ev-left"><div class="ca-ev-tabsrow">${safeCard("Tabs", evxTabs, D, "ca-card", "ev-tabs")}${sortSel}</div>${safeCard("+EV Opportunities", evxTableCard, D, "ca-card ca-ev-tablecard", "ev-table")}</div>
      <aside class="ca-ev-rail" id="ev-rail">${evxRailCard(D, 0)}${evxRailCard(D, 1)}<div class="ca-ev-pair">${evxRailCard(D, 2)}${evxRailCard(D, 3)}</div></aside></div>
    <div class="ca-ev-legacy" id="ev-legacy">${safeCard("Parlays", evxParlays, D, "ca-card")}</div></div>`;
}

// Redraw the tabs + table from cached data (filter / tab / sort changes need no refetch).
function evxRedraw() {
  const D = window.__caEvData;
  if (!D) return;
  const tabs = document.getElementById("ev-tabs"), table = document.getElementById("ev-table");
  if (tabs) tabs.outerHTML = safeCard("Tabs", evxTabs, D, "ca-card", "ev-tabs");
  if (table) table.outerHTML = safeCard("+EV Opportunities", evxTableCard, D, "ca-card ca-ev-tablecard", "ev-table");
}
function evxSet(key, value) {
  const s = evxState();
  if (key === "league") key = "sport";
  s[key] = key === "minEdge" ? (EV_MIN_EDGES.includes(+value) ? +value : 0) : value;
  evxSync(s);
}
function wireEvPage2() {
  const root = document.querySelector(".ca-ev");
  if (!root) return;
  root.addEventListener("change", (e) => {
    const sel = e.target.closest("select[data-ev]");
    if (!sel) return;
    evxSet(sel.dataset.ev, sel.value);
    if (sel.dataset.ev === "sport" || sel.dataset.ev === "league") root.querySelectorAll('select[data-ev="sport"],select[data-ev="league"]').forEach((x) => { x.value = sel.value; });
    evxRedraw();
  });
  const q = root.querySelector("[data-ev-q]");
  if (q) q.addEventListener("input", () => { evxSet("q", q.value); evxRedraw(); });
  root.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-pill]");
    if (pill && pill.dataset.pill === "ev-tab") { evxSet("kind", pill.dataset.key); evxRedraw(); return; }
    if (e.target.closest("[data-ev-refresh]")) { render(); return; }
    if (e.target.closest("[data-ev-reset]")) {
      Object.assign(evxState(), EV_DEFAULT); evxSync(evxState());
      render();
      return;
    }
  });
}
