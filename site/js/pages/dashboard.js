/* Dashboard (index.html) — "Today at a Glance", mockup #1.
   Real data only: every card has an empty state and the mockup's numbers are never rendered.
   Every "today" panel reads the selected ET date (?date=YYYY-MM-DD, default today); the 30D /
   7D / Season windows end on that date. Phase B panels (Recent Model Updates) are omitted.
   Performance numbers (Units, ROI, Hit Rate, Snapshot, Exposure) all come from ONE population: graded +EV
   picks in ev_pnl_daily (every market). "Best Opportunities Right Now" is the one undated panel: on today's
   date it lists the whole upcoming 7-day board.
   Depends on app.js (sb, predictions, gameMoneylines, evBestLines, evResultsRows, evGradedPicks,
   trackRecordStarts, etDateStr, timeET, logoImg, teamShort, ctxEsc, getSettings, tileBest, r05,
   uStr, pStr, pct1, render), metrics.js, ui.js, shell.js and data.js (shared helpers: pickLabel,
   shortTeam, gradedLinePicks, perfWindow, safeCard, ...). */
const DASH_COLOR = { nfl: "var(--navy)", cfb: "var(--red)", mlb: "var(--blue)", nba: "var(--amber)" };
const DASH_MKT_COLOR = ["var(--navy)", "var(--blue)", "var(--amber)", "var(--muted)"];
// Explicit market classes: anything not listed here is ignored (and warned about), never defaulted into Player Props.
const DASH_MKT_TYPES = [["Game Lines", (m) => m === "moneyline" || m === "spread"],
  ["Player Props", (m) => m === "prop" || Object.prototype.hasOwnProperty.call(PROP_LABEL, m)],
  ["Totals", (m) => m === "total"],
  ["Parlays", (m) => m === "parlay"]];
const dashMktType = (m) => { const t = DASH_MKT_TYPES.find(([, f]) => f(m)); return t ? t[0] : null; };
const dashWarned = new Set();
// UI state that survives the 5-minute re-render (the date lives in the URL).
const dashState = window.__caDash || (window.__caDash = { op: "all", kind: "all", slate: "all", perf: "30d", movers: "moves", expo: "sport", watch: "games" });

/* ── pure helpers ─────────────────────────────────────────────────────── */

// Exposure over graded bets since `sinceDate`: shares = bets staked (graded count n), units = P&L.
// Rows whose market is not explicitly classified are left out (console.warn once per market).
// Game Lines / Player Props / Totals always list; Parlays only when present.
function dashExposure(evPnlRows, predPnlRows, sinceDate) {
  const all = [...(evPnlRows || []), ...(predPnlRows || [])].filter((r) => r && r.game_date >= sinceDate && finite(r.n) && +r.n > 0);
  const rows = all.filter((r) => {
    if (dashMktType(r.market)) return true;
    if (!dashWarned.has(r.market)) { dashWarned.add(r.market); console.warn(`dashboard: ignoring unclassified market "${r.market}" in exposure`); }
    return false;
  });
  const staked = rows.reduce((s, r) => s + +r.n, 0);
  const slice = (rs) => { const a = aggPnl(rs); return { staked: a.n, units: a.units, pct: staked ? a.n * 100 / staked : 0 }; };
  const bySport = [...new Set(rows.map((r) => r.sport))].map((sport) => ({ sport, ...slice(rows.filter((r) => r.sport === sport)) }))
    .sort((a, b) => b.staked - a.staked || String(a.sport).localeCompare(String(b.sport)));
  const byMarket = DASH_MKT_TYPES.map(([label]) => ({ label, ...slice(rows.filter((r) => dashMktType(r.market) === label)) }))
    .filter((x) => x.label !== "Parlays" || x.staked > 0);
  return { staked, totalUnits: aggPnl(rows).units, bySport, byMarket };
}

// Cumulative P&L units per calendar day, from the first day with data in [from, to] through `to`.
function dashCumUnits(rows, from, to) {
  const by = new Map();
  for (const r of rows || []) {
    if (r && r.game_date >= from && r.game_date <= to) by.set(r.game_date, (by.get(r.game_date) || 0) + (finite(r.pnl) ? unitsFromPnl(r.pnl) : 0));
  }
  if (!by.size) return [];
  const out = [];
  let cum = 0;
  for (let d = [...by.keys()].sort()[0]; d <= to; d = addDays(d, 1)) { cum += by.get(d) || 0; out.push({ date: d, units: cum }); }
  return out;
}

// Opportunities for "Best Opportunities Right Now": on today's date the whole upcoming board (kickoff in
// [now, now + 7d]); on any other date just that date's games. Sorted by EV desc.
function dashBoardOpps(tiered, date, nowMs, isToday) {
  const rows = isToday
    ? (tiered || []).filter((o) => { const t = timeMs(o.commence); return Number.isFinite(t) && t >= nowMs && t <= nowMs + 7 * 864e5; })
    : (tiered || []).filter((o) => o.commence && etDateStr(o.commence) === date);
  return rows.sort((a, b) => b.evPct - a.evPct);
}

// Model line vs market line per game (spread on the home side, total), biggest gap first.
function dashModelVsMarket(preds) {
  const out = [];
  for (const r of preds || []) {
    if (!r || !finite(r.pred_home_score) || !finite(r.pred_away_score)) continue;
    const margin = +r.pred_home_score - +r.pred_away_score, total = +r.pred_home_score + +r.pred_away_score;
    if (finite(r.market_spread)) { const to = r05(-margin) || 0; out.push({ pred: r, market: "spread", from: +r.market_spread, to, diff: to - +r.market_spread }); }
    if (finite(r.market_total)) { const to = r05(total); out.push({ pred: r, market: "total", from: +r.market_total, to, diff: to - +r.market_total }); }
  }
  return out.filter((x) => x.diff !== 0).sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
}

// Opportunities whose pick was first flagged in the 24h before nowMs (history from loadEvHistory), newest first.
const dashSigKey = (kind, sport, gamePk, market, side, who) => `${kind}|${sport}|${gamePk}|${market}|${side}|${kind === "prop" ? who || "" : ""}`;
function dashNewSignals(opps, hist, nowMs) {
  const first = new Map();
  for (const h of hist || []) if (h && h.at) first.set(dashSigKey(h.kind, h.sport, h.game_pk, h.market, h.side, h.player_name), h.at);
  return (opps || []).map((o) => ({ opp: o, at: first.get(dashSigKey(o.kind, o.sport, o.game_pk, o.market, o.side, o.playerName)) }))
    .filter((x) => x.at && timeMs(x.at) >= nowMs - 864e5).sort((a, b) => timeMs(b.at) - timeMs(a.at));
}

// Multi-segment donut with a centred value + caption (ui.js `donut` draws a single fraction).
function dashDonut(segs, { size = 136, stroke = 18, center = "", caption = "" } = {}) {
  const ss = (segs || []).filter((s) => s && finite(s.pct) && +s.pct > 0);
  if (!ss.length) return "";
  const r = (size - stroke) / 2, C = 2 * Math.PI * r, c = size / 2, tot = ss.reduce((t, s) => t + +s.pct, 0);
  const gap = ss.length > 1 ? 2 : 0;
  let off = 0;
  const arcs = ss.map((s) => {
    const len = C * +s.pct / tot, draw = Math.max(0.5, len - gap);
    const a = `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${ctxEsc(s.color || "var(--navy)")}" stroke-width="${stroke}" stroke-dasharray="${draw.toFixed(2)} ${(C - draw).toFixed(2)}" stroke-dashoffset="${(-off).toFixed(2)}" transform="rotate(-90 ${c} ${c})"/>`;
    off += len;
    return a;
  }).join("");
  return `<div class="ca-dash-donut" style="width:${size}px;height:${size}px"><svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}"><circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="#E9E5DB" stroke-width="${stroke}"/>${arcs}</svg><div class="ca-dash-donut-c"><b>${ctxEsc(center)}</b><span>${ctxEsc(caption)}</span></div></div>`;
}

/* ── small view helpers ───────────────────────────────────────────────── */
const dashClock = (iso) => (iso ? timeET(iso).replace(/ ET$/, "") : "");
const dashDateLabel = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric", year: "numeric" });
function dashAgo(iso, nowMs) {
  const m = Math.max(0, Math.round((nowMs - timeMs(iso)) / 6e4));
  if (!Number.isFinite(m)) return "";
  return m < 60 ? `${m}m ago` : m < 48 * 60 ? `${Math.round(m / 60)}h ago` : `${Math.round(m / 1440)}d ago`;
}

/* ── data ─────────────────────────────────────────────────────────────── */
function dashDate() {
  let q = null;
  try { q = new URLSearchParams(location.search).get("date"); } catch { q = null; }
  return q && /^\d{4}-\d{2}-\d{2}$/.test(q) && Number.isFinite(Date.parse(`${q}T12:00:00Z`)) ? q : etDateStr(new Date().toISOString());
}

async function dashLoad(date) {
  const nowMs = Date.now(), isToday = date === etDateStr(new Date(nowMs).toISOString());
  // History reaches back far enough that the 7 days ending on `date` are covered (>= 8 days).
  const histDays = Math.max(8, Math.ceil((nowMs - Date.parse(`${date}T12:00:00Z`)) / 864e5) + 8);
  const [predsBy, oppsAll, lineBy, mls, hist, evPnl, results, picks, starts, moves, splits, pickRows] = await Promise.all([
    Promise.all(SPORTS.map((s) => predictions(s).catch(() => []))),
    loadOpportunities().catch(() => []),
    evBestLines().catch(() => new Map()),
    gameMoneylines().catch(() => new Map()),
    loadEvHistory(histDays).catch(() => []),
    sb("ev_pnl_daily?select=*").catch(() => []),
    evResultsRows().catch(() => []),
    evGradedPicks().catch(() => []),
    trackRecordStarts().catch(() => new Map()),
    loadLineMoves().catch(() => []),
    loadSplits().catch(() => new Map()),
    sb("ev_current?is_pick=eq.true&select=sport,game_pk,market,side,matchup,commence_time,soft_vs_sharp_gap,pinnacle_price,best_book,best_price").catch(() => []),
  ]);
  const preds = predsBy.flat();
  const onDate = (iso) => !!iso && etDateStr(iso) === date;
  const tiered = (oppsAll || []).filter((o) => o.tier);
  const opps = tiered.filter((o) => onDate(o.commence));     // date-scoped: Live +EV count, Best Edge, Signals, Slate
  const upToDate = (r) => r && r.game_date <= date;
  const pnl = (evPnl || []).filter(upToDate);                // the one performance population: graded +EV picks
  return {
    date, nowMs, isToday, preds, lineBy, mls, splits, hist: hist || [],
    datePreds: preds.filter((r) => gameDate(r) === date),
    opps, tiered, allOpps: oppsAll || [],
    boardOpps: dashBoardOpps(tiered, date, nowMs, isToday),  // separate list for the Best Opportunities card
    counts7: dailyCounts(hist || [], (r) => r.date, 7, date),
    pnl, expo: dashExposure(pnl, [], addDays(date, -29)),
    graded: gradedLinePicks(results, picks, starts).filter((r) => r.date <= date),
    moves: (moves || []).filter((m) => onDate(m.commence_time)),
    gaps: (pickRows || []).filter((r) => onDate(r.commence_time) && finite(r.soft_vs_sharp_gap) && +r.soft_vs_sharp_gap > 0)
      .sort((a, b) => +b.soft_vs_sharp_gap - +a.soft_vs_sharp_gap),
  };
}

/* ── stat cards ───────────────────────────────────────────────────────── */
function dashStatGames(D) {
  const per = SPORTS.map((s) => [s, D.datePreds.filter((r) => r.sport === s).length]);
  return `<div class="ca-card ca-stat ca-dash-stat"><div class="ca-stat-label">Games Tracked</div><div class="ca-stat-value">${per.reduce((t, [, n]) => t + n, 0)}</div>
    <div class="ca-dash-gt">${per.map(([s, n]) => `<div class="ca-dash-gt-item"${!n && SPORT_STATUS[s] ? ` title="${ctxEsc(SPORT_STATUS[s])}"` : ""}>${leagueLogo(s)}<div><span>${SPORT_NAME[s]}</span><b>${n}</b></div></div>`).join("")}</div></div>`;
}
function dashStatLive(D) {
  // R14: "N new today" = picks first flagged on this ET date; the bars are new picks per day, last 7 days.
  const c = D.counts7, newToday = c.length ? c[c.length - 1].n : 0;
  const sub = `<span title="Picks first flagged on this date (all sports, game lines and props)"><span class="${newToday ? "pos" : ""}">${newToday} new ${D.isToday ? "today" : "that day"}</span></span>`;
  return statCard({ label: "Live +EV Opportunities", value: String(D.opps.length), valueClass: D.opps.length ? "pos" : "", sub,
    visual: c.some((x) => x.n > 0) ? `<span title="New picks per day, last 7 days">${miniBars(c.map((x) => x.n), { w: 56, h: 46 })}</span>` : "" });
}
function dashStatBest(D) {
  const best = [...D.opps].sort((a, b) => b.evPct - a.evPct)[0];
  if (!best) return statCard({ label: "Best Current Edge", value: "—", sub: "No +EV opportunities on this date." });
  const [away, home] = matchupSides(best.matchup);
  const txt = best.kind === "prop" ? `${ctxEsc(best.playerName || "")}<br>${ctxEsc(pickLabel(best, D.lineBy))}` : `${ctxEsc(pickLabel(best, D.lineBy))}<br>${ctxEsc(shortMatchup(away, home, best.sport))}`;
  return statCard({ label: "Best Current Edge", value: pStr(best.evPct), valueClass: "pos",
    sub: `<a class="ca-dash-be" href="${gameHref(best.sport, best.game_pk)}">${logoImg(away, best.sport)}<span>${txt}</span><i>at</i>${logoImg(home, best.sport)}</a>` });
}
const dashRecord = (p) => `${p.wins}-${p.losses}`;
function dashStatHit(D) {
  const p = perfWindow(D.pnl, D.graded, addDays(D.date, -29), D.date), hr = p.hitRate;
  const vs = p.vsMarket == null ? "" : `<span class="${signCls(p.vsMarket)}">${pStr(p.vsMarket)}</span> vs. market <span class="muted">(game lines)</span><br>`;
  return statCard({ label: "Model Hit Rate", labelNote: "(30D)", value: hr == null ? "—" : `${(hr * 100).toFixed(1)}%`,
    sub: hr == null ? "No graded +EV picks in 30 days." : `${vs}<span class="muted">${dashRecord(p)} · ${p.n} graded +EV picks</span>`,
    visual: hr == null ? "" : donut(hr, { size: 66, stroke: 10 }) });
}
function dashStatUnits(D) {
  const from = addDays(D.date, -29), p = perfWindow(D.pnl, D.graded, from, D.date);
  return statCard({ label: "Units", labelNote: "(30D)", value: p.n ? uStr(p.units) : "—", valueClass: signCls(p.units),
    sub: p.n ? `<span class="${signCls(p.roiPct)}">${pStr(p.roiPct)} ROI</span><br><span class="muted">${p.n} graded +EV picks</span>` : "No graded +EV picks in 30 days.",
    visual: sparkline(dashCumUnits(p.rows, from, D.date).map((x) => x.units), { w: 72, h: 44 }) });
}
function dashStatSignals(D) {
  const tiers = [["HIGH", "High Conviction", "var(--green)"], ["STRONG", "Strong Value", "var(--blue)"], ["MEDIUM", "Medium Value", "var(--amber)"]];
  return `<div class="ca-card ca-stat ca-dash-stat"><div class="ca-stat-label">Active Signals</div><div class="ca-dash-sig">${tiers.map(([t, l, col]) =>
    `<div class="ca-dash-sig-row"><i style="border-color:${col}"></i><b>${D.opps.filter((o) => o.tier === t).length}</b><span>${l}</span></div>`).join("")}</div></div>`;
}
const DASH_STATS = [["Games Tracked", dashStatGames], ["Live +EV Opportunities", dashStatLive], ["Best Current Edge", dashStatBest],
  ["Model Hit Rate", dashStatHit], ["Units", dashStatUnits], ["Active Signals", dashStatSignals]];
function dashStatCards(D) {
  return `<div class="ca-stats ca-dash-stats">${DASH_STATS.map(([name, fn]) => safeCard(name, fn, D, "ca-card ca-stat ca-dash-stat")).join("")}</div>`;
}

/* ── Best Opportunities Right Now ─────────────────────────────────────── */
function dashOppsCard(D) {
  const st = dashState;
  const list = D.boardOpps.filter((o) => (st.op === "all" || o.sport === st.op) && (st.kind === "all" || o.kind === st.kind)).slice(0, 10);
  const sportPills = pills("dash-op", [["all", "All Sports"], ["nfl", "NFL"], ["cfb", "CFB"], ["mlb", "MLB"], ["nba", "NBA"]], st.op);
  const kindPills = pills("dash-kind", [["line", "Game Lines"], ["prop", "Player Props"]], st.kind);
  let body;
  if (!list.length) {
    body = emptyMsg(SPORT_STATUS[st.op] || (st.kind === "line" ? "No +EV game lines on the board right now."
      : st.kind === "prop" ? "No +EV player props on the board right now." : "No +EV opportunities on the board right now."));
  } else {
    const rows = list.map((o, i) => {
      const [away, home] = matchupSides(o.matchup);
      const who = o.kind === "prop" ? `<span class="ca-team" title="${ctxEsc(shortMatchup(away, home, o.sport))}">${ctxEsc(o.playerName || "")}</span>`
        : `<span class="ca-team">${oppLogo(o) || logoImg(away, o.sport)}${ctxEsc(shortMatchup(away, home, o.sport))}</span>`;
      return `<tr data-href="${gameHref(o.sport, o.game_pk)}"><td class="muted">${i + 1}</td><td><span class="ca-team">${leagueLogo(o.sport)}${SPORT_NAME[o.sport] || ""}</span></td>
        <td>${who}</td><td class="ca-dash-gtime">${kickLabel(o.commence)}</td><td><span class="ca-ell ca-dash-mkt" title="${ctxEsc(pickLabel(o, D.lineBy))}">${ctxEsc(pickLabel(o, D.lineBy))}</span></td><td><span class="ca-dash-odds">${bookBadge(o.book)}${oddsStr(o.odds)}</span></td>
        <td>${pct1(o.modelProb)}</td><td>${pct1(o.impliedProb)}</td><td class="${signCls(o.edgePp)} ca-b">${pStr(o.edgePp)}</td>
        <td>${alphaCell(o.alpha)}</td><td>${confPill(o.tier)}</td><td>${goLink(o.sport, o.game_pk)}</td></tr>`;
    }).join("");
    body = `<div class="ca-table-wrap"><table class="ca-table ca-dash-table ca-dash-opps"><thead><tr><th>#</th><th>Sport</th><th>Matchup / Player</th><th>Game Time</th><th>Market</th><th>Best Odds</th><th>Model Prob.</th><th>Market Prob.</th><th>Edge</th><th>Alpha Score</th><th>Confidence</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  return `<section class="ca-card ca-dash-card" id="dash-opps"><div class="ca-card-head"><h2>Best Opportunities Right Now</h2><p>${D.isToday ? "Top model edges for the next 7 days" : "Top model edges for this date"}, sorted by expected value.</p><a class="ca-link" href="ev.html">View All →</a></div>
    <div class="ca-dash-pillrow">${sportPills}<span class="ca-dash-pillsep"></span>${kindPills}</div>${body}</section>`;
}

/* ── Today's Slate ────────────────────────────────────────────────────── */
function dashSlateCard(D) {
  const s = dashState.slate;
  const rows = slateRows(D.datePreds.filter((r) => s === "all" || r.sport === s), D.opps);
  const counts = SPORTS.map((x) => [x, D.datePreds.filter((r) => r.sport === x).length]).sort((a, b) => b[1] - a[1]);
  const allHref = `${s === "all" ? counts[0][0] : s}.html`;
  const settings = getSettings();
  let body;
  if (!rows.length) body = emptyMsg(SPORT_STATUS[s] || "No games on the slate for this date.");
  else {
    body = `<div class="ca-table-wrap ca-dash-slate-wrap"><table class="ca-table ca-dash-table ca-dash-slate"><thead><tr><th>Time (ET)</th><th>Matchup / Player</th><th>Market</th><th>Line</th><th>Alpha Score</th><th></th></tr></thead><tbody>${rows.map((g) => {
      const start = timeMs(g.commence), mins = (start - D.nowMs) / 6e4;
      const past = D.date < etDateStr(new Date(D.nowMs).toISOString());
      const [dot, dotTitle] = !Number.isFinite(start) ? ["later", "Kickoff time unknown"]
        : !D.isToday ? (past ? ["final", "Final"] : ["later", "Scheduled"])
        : mins <= 0 ? ["live", "Live / started"] : mins <= 180 ? ["soon", "Starts within 3 hours"] : ["later", "Later today"];
      let market, line, tip = "";
      if (g.opp) {
        tip = oppSlateLabel(g.opp, D.lineBy); market = ctxEsc(tip); line = oddsStr(g.opp.odds);
      }
      else {
        // No +EV edge: the model's lean (predicted winner) and that side's best moneyline at the user's books.
        const p = g.pred, homeWins = finite(p.home_win_prob) ? +p.home_win_prob >= 0.5 : null;
        if (homeWins == null) { market = "<span class='muted'>—</span>"; line = ""; }
        else {
          const side = homeWins ? "home" : "away", m = D.mls && D.mls.get(String(g.game_pk));
          const bestMl = m ? tileBest(m[`${side}_prices`], m[`${side}_price`], m[`${side}_book`], settings) : null;
          tip = `${shortTeam(homeWins ? g.home : g.away, g.sport)} ML — model lean, not a +EV pick`;
          market = `<span class="muted">${ctxEsc(shortTeam(homeWins ? g.home : g.away, g.sport))} ML</span>`;
          line = bestMl ? oddsStr(bestMl.am) : "";
        }
      }
      const label = shortMatchup(g.away, g.home, g.sport);
      return `<tr data-href="${gameHref(g.sport, g.game_pk)}"><td><span class="ca-dash-dot ${dot}" title="${dotTitle}"></span>${dashClock(g.commence)}</td>
        <td><span class="ca-team ca-dash-pair" title="${ctxEsc(label)}">${logoImg(g.away, g.sport)}${logoImg(g.home, g.sport)}<span class="ca-ell">${ctxEsc(label)}</span></span></td><td><span class="ca-ell ca-dash-mkt" title="${ctxEsc(tip)}">${market}</span></td><td>${line}</td>
        <td>${g.opp ? alphaCell(g.opp.alpha) : "<span class='muted'>—</span>"}</td><td class="ca-dash-actions">${starButton("games", g.game_pk, label)}${goLink(g.sport, g.game_pk)}</td></tr>`;
    }).join("")}</tbody></table></div>`;
  }
  return `<section class="ca-card ca-dash-card" id="dash-slate"><div class="ca-card-head"><h2>Today’s Slate</h2>${pills("dash-slate", [["all", "All"], ["nfl", "NFL"], ["cfb", "CFB"], ["mlb", "MLB"], ["nba", "NBA"]], s)}<a class="ca-link" href="${allHref}">View All →</a></div>${body}</section>`;
}

/* ── Performance Snapshot ─────────────────────────────────────────────── */
function dashPerfCard(D) {
  const w = dashState.perf, from = w === "7d" ? addDays(D.date, -6) : w === "30d" ? addDays(D.date, -29) : "0000-00-00";
  const p = perfWindow(D.pnl, D.graded, from, D.date);
  const mini = (label, value, cls = "") => `<div class="ca-dash-mini"><span>${label}</span><b class="${cls}">${value}</b></div>`;
  const pts = dateAxisPoints(dashCumUnits(p.rows, from, D.date), (q) => q.units);
  const chart = areaChart(pts, { h: 190, yTicks: 5 });
  const cap = p.n ? `<p class="ca-dash-cap">${p.n} graded +EV picks (${dashRecord(p)}) · ROI, Units and Hit Rate cover the same bets${p.edgeN ? ` · Avg. Edge: ${p.edgeN} game-line picks with a flagged price` : ""}</p>` : "";
  return `<section class="ca-card ca-dash-card" id="dash-perf"><div class="ca-card-head"><h2>Performance Snapshot</h2>${pills("dash-perf", [["7d", "7D"], ["30d", "30D"], ["season", "Season"]], w)}<a class="ca-link" href="track-record.html">View Track Record →</a></div>
    <div class="ca-dash-minis">${mini("ROI", p.n ? pStr(p.roiPct) : "—", signCls(p.roiPct))}${mini("Units", p.n ? uStr(p.units) : "—", signCls(p.units))}${mini("Hit Rate", p.hitRate == null ? "—" : `${(p.hitRate * 100).toFixed(1)}%`)}${mini("Avg. Edge", p.avgEdge == null ? "—" : pStr(p.avgEdge), signCls(p.avgEdge))}</div>
    ${cap}${chart || emptyMsg("No graded +EV picks in this window yet.")}</section>`;
}

/* ── Market Movers ────────────────────────────────────────────────────── */
function dashMoverRow({ href, icon, title, sub, chg, chgCls = "", when }) {
  return `<a class="ca-dash-mv" href="${href}"><span class="ca-dash-mv-ic">${icon}</span><span class="ca-dash-mv-main"><b>${title}</b><small>${sub}</small></span><span class="ca-dash-mv-r"><b class="${chgCls}">${chg}</b><small>${when}</small></span></a>`;
}
function dashMoversCard(D) {
  const tab = dashState.movers, byPk = new Map(D.preds.map((r) => [String(r.game_pk), r]));
  let rows = [], empty = "";
  if (tab === "moves") {
    const seen = new Set();
    for (const m of D.moves) {
      const p = byPk.get(String(m.game_pk)), k = `${m.game_pk}|${m.market}`;
      if (!p || seen.has(k)) continue;
      seen.add(k);
      const info = lineMoveInfo(m, p, D.splits);
      rows.push(dashMoverRow({ href: gameHref(p.sport, p.game_pk), icon: m.market === "total" ? ICON_TREND : logoImg(m.side === "home" ? p.home_team_name : p.away_team_name, p.sport),
        title: ctxEsc(info.title), sub: ctxEsc(`${info.mu}${info.tickets ? ` • ${info.tickets}` : ""}`), chg: info.chg, chgCls: signCls(info.d), when: m.cur_at ? dashAgo(m.cur_at, D.nowMs) : "" }));
      if (rows.length >= 5) break;
    }
    empty = "No line moves captured for upcoming games yet.";
  } else if (tab === "model") {
    rows = dashModelVsMarket(D.datePreds).slice(0, 5).map((x) => {
      const p = x.pred, mu = shortMatchup(p.away_team_name, p.home_team_name, p.sport);
      const title = x.market === "spread" ? `${shortTeam(p.home_team_name, p.sport)} ${lineStr(x.from)} → ${lineStr(x.to)}`
        : `${teamShort(p.away_team_name, p.sport)} @ ${teamShort(p.home_team_name, p.sport)} O/U ${x.from} → ${x.to}`;
      return dashMoverRow({ href: gameHref(p.sport, p.game_pk), icon: x.market === "total" ? ICON_TREND : logoImg(p.home_team_name, p.sport), title: ctxEsc(title),
        sub: ctxEsc(`Market → model ${x.market} · ${mu}`), chg: `${signedStr(x.diff, 1)} pts`, chgCls: signCls(x.diff), when: dashClock(p.commence_time) });
    });
    empty = "No model vs. market gaps for this date.";
  } else if (tab === "new") {
    rows = dashNewSignals(D.opps, D.hist, D.nowMs).slice(0, 5).map(({ opp: o, at }) => {
      const [away, home] = matchupSides(o.matchup);
      return dashMoverRow({ href: gameHref(o.sport, o.game_pk), icon: oppLogo(o) || leagueLogo(o.sport),
        title: ctxEsc(`${o.kind === "prop" ? `${o.playerName} ` : ""}${pickLabel(o, D.lineBy)}`), sub: ctxEsc(`${shortMatchup(away, home, o.sport)} • Alpha ${o.alpha} • ${o.tier}`),
        chg: `${pStr(o.evPct)} EV`, chgCls: "pos", when: dashAgo(at, D.nowMs) });
    });
    empty = "No new signals in the last 24 hours.";
  } else {
    rows = D.gaps.slice(0, 5).map((r) => {
      const [away, home] = matchupSides(r.matchup), o = { kind: "line", ...r };
      return dashMoverRow({ href: gameHref(r.sport, r.game_pk), icon: oppLogo(o) || ICON_TREND,
        title: ctxEsc(`${pickLabel(o, D.lineBy)} ${oddsStr(r.best_price)}`), sub: ctxEsc(`${(BOOK_STYLE[String(r.best_book || "").toLowerCase()] || [r.best_book])[0]} vs. Pinnacle ${oddsStr(r.pinnacle_price)} • ${shortMatchup(away, home, r.sport)}`),
        chg: `${signedStr(+r.soft_vs_sharp_gap * 100, 1, "%")}`, chgCls: "pos", when: dashClock(r.commence_time) });
    });
    empty = "No stale lines for this date.";
  }
  const tabs = pills("dash-movers", [["moves", "Line Moves"], ["model", "Model vs. Market"], ["new", "New Signals"], ["stale", "Stale Lines"]], tab);
  return `<section class="ca-card ca-dash-card" id="dash-movers"><div class="ca-card-head"><h2>Market Movers</h2></div><div class="ca-dash-tabs">${tabs}</div>
    <div class="ca-dash-mvlist">${rows.length ? rows.join("") : emptyMsg(empty)}</div></section>`;
}

/* ── Portfolio & Exposure ─────────────────────────────────────────────── */
function dashExpoCard(D) {
  const e = D.expo, mode = dashState.expo;
  const segs = mode === "sport"
    ? e.bySport.map((x) => ({ label: SPORT_NAME[x.sport] || String(x.sport).toUpperCase(), pct: x.pct, units: x.units, color: DASH_COLOR[x.sport] || "var(--muted)" }))
    : e.byMarket.filter((x) => x.staked > 0).map((x) => ({ label: x.label, pct: x.pct, units: x.units, color: DASH_MKT_COLOR[DASH_MKT_TYPES.findIndex(([l]) => l === x.label)] || "var(--muted)" }));
  const tabs = pills("dash-expo", [["sport", "By Sport"], ["market", "By Market Type"]], mode);
  const body = e.staked
    ? `<div class="ca-dash-expo">${dashDonut(segs, { size: 96, stroke: 14, center: uStr(e.totalUnits), caption: "Total Units" })}${donutLegend(segs.map((s) => ({ label: s.label, pct: s.pct, value: uStr(s.units), color: s.color })))}</div>
      <p class="ca-dash-cap">Graded +EV picks, last 30 days · ${e.staked} bets staked</p>
      <div class="ca-dash-mtx"><h3>Market Type Exposure</h3>${e.byMarket.map((x) => `<div class="ca-dash-mtx-row"><span>${x.label}</span><i><b style="width:${x.pct.toFixed(1)}%"></b></i><em>${Math.round(x.pct)}%</em></div>`).join("")}</div>`
    : emptyMsg("No graded +EV picks in the last 30 days.");
  return `<section class="ca-card ca-dash-card" id="dash-expo"><div class="ca-card-head"><h2>Portfolio &amp; Exposure</h2></div><div class="ca-dash-tabs ca-dash-tabs-2">${tabs}</div>${body}</section>`;
}

/* ── Watchlist ────────────────────────────────────────────────────────── */
function dashWatchBody(D) {
  const w = watchlistGet(), tab = dashState.watch, byPk = new Map(D.preds.map((r) => [String(r.game_pk), r]));
  const bestFor = (sport, pk) => D.tiered.filter((o) => o.sport === sport && String(o.game_pk) === String(pk)).sort((a, b) => b.alpha - a.alpha)[0];
  const row = (star, main, pick, when, href) => `<div class="ca-dash-wl-row">${star}<a href="${href}"><b>${main}</b>${pick ? `<small>${pick}</small>` : ""}</a><span class="muted">${when}</span></div>`;
  const oppTxt = (o) => `${o.kind === "prop" ? `${String(o.playerName || "").split(/\s+/).slice(-1)[0]} ` : ""}${pickLabel(o, D.lineBy)} ${oddsStr(o.odds)}`;
  let rows = [];
  if (tab === "games") {
    rows = w.games.map((id) => byPk.get(id)).filter(Boolean).sort((a, b) => timeMs(a.commence_time) - timeMs(b.commence_time)).map((p) => {
      const mu = shortMatchup(p.away_team_name, p.home_team_name, p.sport), o = bestFor(p.sport, p.game_pk);
      return row(starButton("games", p.game_pk, mu), ctxEsc(mu), o ? ctxEsc(oppTxt(o)) : "", kickLabel(p.commence_time), gameHref(p.sport, p.game_pk));
    });
  } else if (tab === "teams") {
    rows = w.teams.map((id) => {
      const up = String(id).toUpperCase();
      const hit = (n, s) => n && (n === id || shortTeam(n, s) === id || String(teamShort(n, s)).toUpperCase() === up);
      const p = D.preds.filter((r) => hit(r.home_team_name, r.sport) || hit(r.away_team_name, r.sport)).sort((a, b) => timeMs(a.commence_time) - timeMs(b.commence_time))[0];
      if (!p) return row(starButton("teams", id, id), ctxEsc(id), "No upcoming game", "", "#watchlist");
      return row(starButton("teams", id, id), ctxEsc(id), ctxEsc(shortMatchup(p.away_team_name, p.home_team_name, p.sport)), kickLabel(p.commence_time), gameHref(p.sport, p.game_pk));
    });
  } else {
    rows = w.players.map((id) => {
      const o = D.allOpps.filter((x) => x.kind === "prop" && x.playerName === id).sort((a, b) => b.alpha - a.alpha)[0];
      if (!o) return row(starButton("players", id, id), ctxEsc(id), "No +EV prop on the board", "", "#watchlist");
      return row(starButton("players", id, id), ctxEsc(id), ctxEsc(`${pickLabel(o, D.lineBy)} ${oddsStr(o.odds)}`), kickLabel(o.commence), gameHref(o.sport, o.game_pk));
    });
  }
  return rows.length ? rows.join("") : emptyMsg("Star games, teams or players to follow them here.");
}
function dashWatchCard(D) {
  return `<section class="ca-card ca-dash-card" id="watchlist"><div class="ca-card-head"><h2>Watchlist</h2></div><div class="ca-dash-tabs">${pills("dash-watch", [["games", "Games"], ["teams", "Teams"], ["players", "Players"]], dashState.watch)}</div>
    <div class="ca-dash-wl">${dashWatchBody(D)}</div></section>`;
}

/* ── page ─────────────────────────────────────────────────────────────── */
const DASH_CARDS = { "dash-opps": ["Best Opportunities", dashOppsCard], "dash-slate": ["Today’s Slate", dashSlateCard], "dash-perf": ["Performance Snapshot", dashPerfCard],
  "dash-movers": ["Market Movers", dashMoversCard], "dash-expo": ["Portfolio & Exposure", dashExpoCard], watchlist: ["Watchlist", dashWatchCard] };
const dashCard = (id, D) => safeCard(DASH_CARDS[id][0], DASH_CARDS[id][1], D, "ca-card ca-dash-card", id);
const DASH_PILL = { "dash-op": ["op", "dash-opps"], "dash-kind": ["kind", "dash-opps"], "dash-slate": ["slate", "dash-slate"], "dash-perf": ["perf", "dash-perf"],
  "dash-movers": ["movers", "dash-movers"], "dash-expo": ["expo", "dash-expo"], "dash-watch": ["watch", "watchlist"] };

async function buildDashboard() {
  const date = dashDate(), D = await dashLoad(date);
  window.__caDashData = D;
  const nav = `<div class="ca-datenav"><button class="ca-dn-btn" data-dash-date="${addDays(date, -1)}" aria-label="Previous day">‹</button>
    <label class="ca-dn-date">${ICON_CAL}<span>${dashDateLabel(date)}</span><input type="date" value="${date}" data-dash-datepick aria-label="Pick a date"></label>
    <button class="ca-dn-btn" data-dash-date="${addDays(date, 1)}" aria-label="Next day">›</button></div>`;
  return `<div class="ca-dash">${pageTitle("Today at a Glance", "Key opportunities, performance, and model insights across all sports.", nav)}
    ${dashStatCards(D)}
    <div class="ca-dash-main">
      <div class="ca-dash-col">${dashCard("dash-opps", D)}<div class="ca-dash-sub ca-dash-sub-l">${dashCard("dash-perf", D)}${dashCard("dash-movers", D)}</div></div>
      <div class="ca-dash-col">${dashCard("dash-slate", D)}<div class="ca-dash-sub ca-dash-sub-r">${dashCard("dash-expo", D)}${dashCard("watchlist", D)}</div></div>
    </div></div>`;
}

function dashGoDate(d) {
  try {
    const u = new URL(location.href);
    if (d === etDateStr(new Date().toISOString())) u.searchParams.delete("date"); else u.searchParams.set("date", d);
    history.replaceState(null, "", u.toString());
  } catch { /* keep the current URL */ }
  window.scrollTo(0, 0);
  render();
}
// Re-draw one card from the cached data (pill changes and watchlist changes need no refetch).
function dashRedraw(id) {
  const el = document.getElementById(id), D = window.__caDashData;
  if (!el || !D || !DASH_CARDS[id]) return;
  el.outerHTML = dashCard(id, D);
}
let dashWatchBound = false;
function wireDashboard() {
  const root = document.querySelector(".ca-dash");
  if (!root) return;
  root.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-pill]");
    if (pill && DASH_PILL[pill.dataset.pill]) {
      const [key, card] = DASH_PILL[pill.dataset.pill], k = pill.dataset.key;
      dashState[key] = key === "kind" && dashState.kind === k ? "all" : k;   // Game Lines / Player Props toggle off
      dashRedraw(card);
      return;
    }
    const nav = e.target.closest("[data-dash-date]");
    if (nav) { dashGoDate(nav.dataset.dashDate); return; }
    const pick = e.target.closest(".ca-dn-date");
    if (pick && !e.target.matches("input")) { const inp = pick.querySelector("input"); try { inp.showPicker(); } catch { inp.focus(); } return; }
  });
  const inp = root.querySelector("[data-dash-datepick]");
  if (inp) inp.addEventListener("change", () => { if (/^\d{4}-\d{2}-\d{2}$/.test(inp.value)) dashGoDate(inp.value); });
  // Stars are handled by shell.js's delegated listener; the Watchlist card just re-reads the list when it changes.
  if (!dashWatchBound) {
    dashWatchBound = true;
    document.addEventListener("ca-watchlist-change", () => dashRedraw("watchlist"));
  }
}
