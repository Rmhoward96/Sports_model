/* Dashboard (index.html) — "Today at a Glance", mockup #1.
   Real data only: every card has an empty state and the mockup's numbers are never rendered.
   Every "today" panel reads the selected ET date (?date=YYYY-MM-DD, default today); the 30D /
   7D / Season windows end on that date. Phase B panels (Recent Model Updates) are omitted.
   Depends on app.js (sb, predictions, gameMoneylines, evBestLines, evResultsRows, evGradedPicks,
   trackRecordStarts, inTrackRecord, etDateStr, timeET, bpKick, logoImg, teamShort, ctxEsc,
   getSettings, tileBest, r05, CFB_2W_MASCOTS, GAME_MARKETS, uStr, pStr, pct1, render),
   metrics.js, ui.js, shell.js and data.js. */
const DASH_SPORT = { nfl: "NFL", cfb: "CFB", mlb: "MLB", nba: "NBA" };
const DASH_COLOR = { nfl: "var(--navy)", cfb: "var(--red)", mlb: "var(--blue)", nba: "var(--amber)" };
const DASH_MKT_COLOR = ["var(--navy)", "var(--blue)", "var(--amber)"];
const DASH_LEAGUE = { nfl: "teamlogos/leagues/500/nfl", cfb: "espn/misc_logos/500/ncaa_football", mlb: "teamlogos/leagues/500/mlb", nba: "teamlogos/leagues/500/nba" };
const DASH_MKT_TYPES = [["Game Lines", (m) => m === "moneyline" || m === "spread"], ["Player Props", (m) => !GAME_MARKETS.has(m)], ["Totals", (m) => m === "total"]];
// UI state that survives the 5-minute re-render (the date lives in the URL).
const dashState = window.__caDash || (window.__caDash = { op: "all", kind: "all", slate: "all", perf: "30d", movers: "moves", expo: "sport", watch: "games" });

/* ── pure helpers ─────────────────────────────────────────────────────── */
const dashFin = (x) => x != null && x !== "" && Number.isFinite(+x);
const dashAddDays = (d, n) => new Date(Date.parse(`${d}T12:00:00Z`) + n * 864e5).toISOString().slice(0, 10);
const dashGameDate = (r) => (r && r.commence_time ? etDateStr(r.commence_time) : (r && r.game_date) || "");
const dashTime = (iso) => (iso ? Date.parse(iso) : NaN);
const dashSides = (matchup) => { const [a = "", h = ""] = String(matchup || "").split(" @ "); return [a, h]; };
const dashLine = (x) => (x === 0 ? "PK" : x > 0 ? `+${x}` : `${x}`);

// Display name as the mockup shows it: NFL nickname ("Lions"), CFB school ("Ohio State"), else as-is.
function dashTeam(name, sport) {
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
const dashMatchup = (away, home, sport) => `${dashTeam(away, sport)} @ ${dashTeam(home, sport)}`;

// The bet as placed: "Lions ML", "Chiefs -6.5", "Over 56.5", "Under 64.5 Rec Yds".
// Spread/total numbers come from the best book's line (ev_best_lines); none known -> no number.
function dashPick(o, lineBy) {
  const ou = o.side === "under" ? "Under" : "Over";
  if (o.kind === "prop") return `${ou}${o.line != null ? ` ${o.line}` : ""} ${o.marketLabel || o.market}`;
  const l = lineBy && lineBy.get(`${o.game_pk}|${o.market}|${o.side}`);
  if (o.market === "total") return `${ou} ${l != null ? l : "total"}`;
  const [away, home] = dashSides(o.matchup);
  const team = dashTeam(o.side === "home" ? home : away, o.sport);
  if (o.market === "moneyline") return `${team} ML`;
  return `${team} ${l != null ? dashLine(l) : "spread"}`;
}

// One row per game (predictions), carrying its highest-alpha opportunity (ties -> higher EV) or null.
function dashSlate(predRows, opps) {
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
  const t = (r) => { const v = dashTime(r.commence); return Number.isFinite(v) ? v : Infinity; };
  return out.sort((a, b) => t(a) - t(b) || String(a.game_pk).localeCompare(String(b.game_pk)));
}

// Exposure over graded bets since `sinceDate`: shares = bets staked (graded count n), units = P&L.
function dashExposure(evPnlRows, predPnlRows, sinceDate) {
  const rows = [...(evPnlRows || []), ...(predPnlRows || [])].filter((r) => r && r.game_date >= sinceDate && dashFin(r.n) && +r.n > 0);
  const staked = rows.reduce((s, r) => s + +r.n, 0);
  const slice = (rs) => { const a = aggPnl(rs); return { staked: a.n, units: a.units, pct: staked ? a.n * 100 / staked : 0 }; };
  const bySport = [...new Set(rows.map((r) => r.sport))].map((sport) => ({ sport, ...slice(rows.filter((r) => r.sport === sport)) }))
    .sort((a, b) => b.staked - a.staked || String(a.sport).localeCompare(String(b.sport)));
  const byMarket = DASH_MKT_TYPES.map(([label, f]) => ({ label, ...slice(rows.filter((r) => f(r.market))) }));
  return { staked, totalUnits: aggPnl(rows).units, bySport, byMarket };
}

// Cumulative P&L units per calendar day, from the first day with data in [from, to] through `to`.
function dashCumUnits(rows, from, to) {
  const by = new Map();
  for (const r of rows || []) {
    if (r && r.game_date >= from && r.game_date <= to) by.set(r.game_date, (by.get(r.game_date) || 0) + (dashFin(r.pnl) ? unitsFromPnl(r.pnl) : 0));
  }
  if (!by.size) return [];
  const out = [];
  let cum = 0;
  for (let d = [...by.keys()].sort()[0]; d <= to; d = dashAddDays(d, 1)) { cum += by.get(d) || 0; out.push({ date: d, units: cum }); }
  return out;
}

// Graded game-line +EV picks inside the track record: won, implied prob of the best price, edge (pp).
function dashGraded(results, picks, starts) {
  const key = (r) => `${r.sport}|${r.game_pk}|${r.market}|${r.side}`;
  const by = new Map((picks || []).map((p) => [key(p), p]));
  const out = [];
  for (const r of results || []) {
    if (!r || (r.won !== true && r.won !== false)) continue;
    const p = by.get(key(r));
    if (!p || !p.commence_time || !inTrackRecord(starts, r.sport, p.commence_time)) continue;
    const imp = americanToProb(p.best_price), implied = Number.isFinite(imp) ? imp : null;
    out.push({ sport: r.sport, game_pk: r.game_pk, market: r.market, won: r.won, implied,
      edgePp: implied != null && dashFin(p.true_prob) ? (+p.true_prob - implied) * 100 : null, date: etDateStr(p.commence_time) });
  }
  return out;
}

// Model line vs market line per game (spread on the home side, total), biggest gap first.
function dashModelVsMarket(preds) {
  const out = [];
  for (const r of preds || []) {
    if (!r || !dashFin(r.pred_home_score) || !dashFin(r.pred_away_score)) continue;
    const margin = +r.pred_home_score - +r.pred_away_score, total = +r.pred_home_score + +r.pred_away_score;
    if (dashFin(r.market_spread)) { const to = r05(-margin) || 0; out.push({ pred: r, market: "spread", from: +r.market_spread, to, diff: to - +r.market_spread }); }
    if (dashFin(r.market_total)) { const to = r05(total); out.push({ pred: r, market: "total", from: +r.market_total, to, diff: to - +r.market_total }); }
  }
  return out.filter((x) => x.diff !== 0).sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));
}

// Opportunities whose pick was first flagged in the 24h before nowMs (history from loadEvHistory), newest first.
const dashSigKey = (kind, sport, gamePk, market, side, who) => `${kind}|${sport}|${gamePk}|${market}|${side}|${kind === "prop" ? who || "" : ""}`;
function dashNewSignals(opps, hist, nowMs) {
  const first = new Map();
  for (const h of hist || []) if (h && h.at) first.set(dashSigKey(h.kind, h.sport, h.game_pk, h.market, h.side, h.player_name), h.at);
  return (opps || []).map((o) => ({ opp: o, at: first.get(dashSigKey(o.kind, o.sport, o.game_pk, o.market, o.side, o.playerName)) }))
    .filter((x) => x.at && dashTime(x.at) >= nowMs - 864e5).sort((a, b) => dashTime(b.at) - dashTime(a.at));
}

// Multi-segment donut with a centred value + caption (ui.js `donut` draws a single fraction).
function dashDonut(segs, { size = 136, stroke = 18, center = "", caption = "" } = {}) {
  const ss = (segs || []).filter((s) => s && dashFin(s.pct) && +s.pct > 0);
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
const dashLeagueLogo = (s) => (DASH_LEAGUE[s] ? `<img class="ca-league" src="https://a.espncdn.com/i/${DASH_LEAGUE[s]}.png" alt="" loading="lazy" onerror="this.style.visibility='hidden'">` : "");
const dashHref = (sport, gamePk) => `game.html?sport=${encodeURIComponent(sport)}&game=${encodeURIComponent(gamePk)}`;
const dashGo = (sport, gamePk) => `<a class="ca-go" href="${dashHref(sport, gamePk)}" aria-label="Open game">→</a>`;
const dashClock = (iso) => (iso ? timeET(iso).replace(/ ET$/, "") : "");
const dashKick = (iso) => (iso ? bpKick(iso).replace(/ ET$/, "") : "");
const dashShortDate = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
const dashDateLabel = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric", year: "numeric" });
const dashSigned = (x, d = 1, unit = "") => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}${unit}`;
const dashCls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");
function dashAgo(iso, nowMs) {
  const m = Math.max(0, Math.round((nowMs - dashTime(iso)) / 6e4));
  if (!Number.isFinite(m)) return "";
  return m < 60 ? `${m}m ago` : m < 48 * 60 ? `${Math.round(m / 60)}h ago` : `${Math.round(m / 1440)}d ago`;
}
const dashEmpty = (msg) => `<p class="ca-empty">${ctxEsc(msg)}</p>`;
const dashPickLogo = (o) => {
  if (o.kind === "prop" || o.market === "total") return "";
  const [away, home] = dashSides(o.matchup);
  return logoImg(o.side === "home" ? home : away, o.sport);
};
const ICON_CAL = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>`;
const ICON_TREND = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>`;

/* ── data ─────────────────────────────────────────────────────────────── */
function dashDate() {
  let q = null;
  try { q = new URLSearchParams(location.search).get("date"); } catch { q = null; }
  return q && /^\d{4}-\d{2}-\d{2}$/.test(q) && Number.isFinite(Date.parse(`${q}T12:00:00Z`)) ? q : etDateStr(new Date().toISOString());
}

async function dashLoad(date) {
  const nowMs = Date.now();
  // History reaches back far enough that the 7 days ending on `date` are covered (>= 8 days).
  const histDays = Math.max(8, Math.ceil((nowMs - Date.parse(`${date}T12:00:00Z`)) / 864e5) + 8);
  const [predsBy, oppsAll, lineBy, mls, hist, evPnl, predPnl, results, picks, starts, moves, splits, pickRows] = await Promise.all([
    Promise.all(SPORTS.map((s) => predictions(s).catch(() => []))),
    loadOpportunities().catch(() => []),
    evBestLines().catch(() => new Map()),
    gameMoneylines().catch(() => new Map()),
    loadEvHistory(histDays).catch(() => []),
    sb("ev_pnl_daily?select=*").catch(() => []),
    sb("prediction_pnl_daily?select=*").catch(() => []),
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
  const opps = tiered.filter((o) => onDate(o.commence));
  const upToDate = (r) => r && r.game_date <= date;
  const pnl = [...(evPnl || []), ...(predPnl || [])].filter(upToDate);
  return {
    date, nowMs, preds, lineBy, mls, splits, hist: hist || [],
    datePreds: preds.filter((r) => dashGameDate(r) === date),
    opps, tiered, allOpps: oppsAll || [],
    counts7: dailyCounts(hist || [], (r) => r.date, 7, date),
    pnl, expo: dashExposure((evPnl || []).filter(upToDate), (predPnl || []).filter(upToDate), dashAddDays(date, -29)),
    graded: dashGraded(results, picks, starts).filter((r) => r.date <= date),
    moves: (moves || []).filter((m) => onDate(m.commence_time)),
    gaps: (pickRows || []).filter((r) => onDate(r.commence_time) && dashFin(r.soft_vs_sharp_gap) && +r.soft_vs_sharp_gap > 0)
      .sort((a, b) => +b.soft_vs_sharp_gap - +a.soft_vs_sharp_gap),
  };
}

/* ── stat cards ───────────────────────────────────────────────────────── */
function dashStatCards(D) {
  const per = SPORTS.map((s) => [s, D.datePreds.filter((r) => r.sport === s).length]);
  const games = `<div class="ca-card ca-stat ca-dash-stat"><div class="ca-stat-label">Games Tracked</div><div class="ca-stat-value">${per.reduce((t, [, n]) => t + n, 0)}</div>
    <div class="ca-dash-gt">${per.map(([s, n]) => `<div class="ca-dash-gt-item"${!n && SPORT_STATUS[s] ? ` title="${ctxEsc(SPORT_STATUS[s])}"` : ""}>${dashLeagueLogo(s)}<div><span>${DASH_SPORT[s]}</span><b>${n}</b></div></div>`).join("")}</div></div>`;

  const c = D.counts7, diff = c.length >= 2 ? c[c.length - 1].n - c[c.length - 2].n : 0;
  const vsY = `<span title="New +EV picks first flagged on this date vs. the day before">${diff > 0 ? `<span class="pos">↑ ${diff}</span>` : diff < 0 ? `<span class="neg">↓ ${-diff}</span>` : "<span>0</span>"} vs. yesterday</span>`;
  const live = statCard({ label: "Live +EV Opportunities", value: String(D.opps.length), valueClass: D.opps.length ? "pos" : "", sub: vsY,
    visual: c.some((x) => x.n > 0) ? miniBars(c.map((x) => x.n), { w: 56, h: 46 }) : "" });

  const best = [...D.opps].sort((a, b) => b.evPct - a.evPct)[0];
  let bestCard;
  if (best) {
    const [away, home] = dashSides(best.matchup);
    const txt = best.kind === "prop" ? `${ctxEsc(best.playerName || "")}<br>${ctxEsc(dashPick(best, D.lineBy))}` : `${ctxEsc(dashPick(best, D.lineBy))}<br>${ctxEsc(dashMatchup(away, home, best.sport))}`;
    bestCard = statCard({ label: "Best Current Edge", value: pStr(best.evPct), valueClass: "pos",
      sub: `<a class="ca-dash-be" href="${dashHref(best.sport, best.game_pk)}">${logoImg(away, best.sport)}<span>${txt}</span><i>at</i>${logoImg(home, best.sport)}</a>` });
  } else bestCard = statCard({ label: "Best Current Edge", value: "—", sub: "No +EV opportunities on this date." });

  const g30 = D.graded.filter((r) => r.date >= dashAddDays(D.date, -29));
  const hr = hitRate(g30), vm = hitRateVsMarket(g30.filter((r) => r.implied != null));
  const hit = statCard({ label: "Model Hit Rate", labelNote: "(30D)", value: hr == null ? "—" : `${(hr * 100).toFixed(1)}%`,
    sub: hr == null ? "No graded +EV picks in 30 days." : vm == null ? "" : `<span class="${dashCls(vm)}">${pStr(vm)}</span> vs. market`,
    visual: hr == null ? "" : donut(hr, { size: 66, stroke: 10 }) });

  const rows30 = D.pnl.filter((r) => r.game_date >= dashAddDays(D.date, -29)), a = aggPnl(rows30);
  const units = statCard({ label: "Units", labelNote: "(30D)", value: a.n ? uStr(a.units) : "—", valueClass: dashCls(a.units),
    sub: a.n ? `<span class="${dashCls(a.roiPct)}">${pStr(a.roiPct)} ROI</span>` : "No graded bets in 30 days.",
    visual: sparkline(dashCumUnits(rows30, dashAddDays(D.date, -29), D.date).map((p) => p.units), { w: 72, h: 44 }) });

  const tiers = [["HIGH", "High Conviction", "var(--green)"], ["STRONG", "Strong Value", "var(--blue)"], ["MEDIUM", "Medium Value", "var(--amber)"]];
  const signals = `<div class="ca-card ca-stat ca-dash-stat"><div class="ca-stat-label">Active Signals</div><div class="ca-dash-sig">${tiers.map(([t, l, col]) =>
    `<div class="ca-dash-sig-row"><i style="border-color:${col}"></i><b>${D.opps.filter((o) => o.tier === t).length}</b><span>${l}</span></div>`).join("")}</div></div>`;
  return `<div class="ca-stats ca-dash-stats">${games}${live}${bestCard}${hit}${units}${signals}</div>`;
}

/* ── Best Opportunities Right Now ─────────────────────────────────────── */
function dashOppsCard(D) {
  const st = dashState;
  const list = D.opps.filter((o) => (st.op === "all" || o.sport === st.op) && (st.kind === "all" || o.kind === st.kind))
    .sort((a, b) => b.evPct - a.evPct).slice(0, 10);
  const sportPills = pills("dash-op", [["all", "All Sports"], ["nfl", "NFL"], ["cfb", "CFB"], ["mlb", "MLB"], ["nba", "NBA"]], st.op);
  const kindPills = pills("dash-kind", [["line", "Game Lines"], ["prop", "Player Props"]], st.kind);
  let body;
  if (!list.length) {
    body = dashEmpty(SPORT_STATUS[st.op] || (st.kind === "line" ? "No +EV game lines on the board right now."
      : st.kind === "prop" ? "No +EV player props on the board right now." : "No +EV opportunities on the board right now."));
  } else {
    const rows = list.map((o, i) => {
      const [away, home] = dashSides(o.matchup);
      const who = o.kind === "prop" ? `<span class="ca-team" title="${ctxEsc(dashMatchup(away, home, o.sport))}">${ctxEsc(o.playerName || "")}</span>`
        : `<span class="ca-team">${dashPickLogo(o) || logoImg(away, o.sport)}${ctxEsc(dashMatchup(away, home, o.sport))}</span>`;
      return `<tr data-href="${dashHref(o.sport, o.game_pk)}"><td class="muted">${i + 1}</td><td><span class="ca-team">${dashLeagueLogo(o.sport)}${DASH_SPORT[o.sport] || ""}</span></td>
        <td>${who}</td><td><span class="ca-ell ca-dash-mkt" title="${ctxEsc(dashPick(o, D.lineBy))}">${ctxEsc(dashPick(o, D.lineBy))}</span></td><td><span class="ca-dash-odds">${bookBadge(o.book)}${oddsStr(o.odds)}</span></td>
        <td>${pct1(o.modelProb)}</td><td>${pct1(o.impliedProb)}</td><td class="${dashCls(o.edgePp)} ca-b">${pStr(o.edgePp)}</td>
        <td>${alphaCell(o.alpha)}</td><td>${confPill(o.tier)}</td><td>${dashGo(o.sport, o.game_pk)}</td></tr>`;
    }).join("");
    body = `<div class="ca-table-wrap"><table class="ca-table ca-dash-table"><thead><tr><th>#</th><th>Sport</th><th>Matchup / Player</th><th>Market</th><th>Best Odds</th><th>Model Prob.</th><th>Market Prob.</th><th>Edge</th><th>Alpha Score</th><th>Confidence</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>`;
  }
  return `<section class="ca-card ca-dash-card" id="dash-opps"><div class="ca-card-head"><h2>Best Opportunities Right Now</h2><p>Top model edges across all sports, sorted by expected value.</p><a class="ca-link" href="ev.html">View All →</a></div>
    <div class="ca-dash-pillrow">${sportPills}<span class="ca-dash-pillsep"></span>${kindPills}</div>${body}</section>`;
}

/* ── Today's Slate ────────────────────────────────────────────────────── */
function dashSlateCard(D) {
  const s = dashState.slate;
  const rows = dashSlate(D.datePreds.filter((r) => s === "all" || r.sport === s), D.opps);
  const counts = SPORTS.map((x) => [x, D.datePreds.filter((r) => r.sport === x).length]).sort((a, b) => b[1] - a[1]);
  const allHref = `${s === "all" ? counts[0][0] : s}.html`;
  const settings = getSettings();
  let body;
  if (!rows.length) body = dashEmpty(SPORT_STATUS[s] || "No games on the slate for this date.");
  else {
    body = `<div class="ca-table-wrap ca-dash-slate-wrap"><table class="ca-table ca-dash-table ca-dash-slate"><thead><tr><th>Time (ET)</th><th>Matchup / Player</th><th>Market</th><th>Line</th><th>Alpha Score</th><th></th></tr></thead><tbody>${rows.map((g) => {
      const start = dashTime(g.commence), mins = (start - D.nowMs) / 6e4;
      const [dot, dotTitle] = !Number.isFinite(start) ? ["later", "Kickoff time unknown"] : mins <= 0 ? ["live", "Live / started"] : mins <= 180 ? ["soon", "Starts within 3 hours"] : ["later", "Later today"];
      let market, line, tip = "";
      if (g.opp) {
        const who = g.opp.kind === "prop" ? `${String(g.opp.playerName || "").split(/\s+/).slice(-1)[0]} ` : "";   // props: player's last name first
        tip = who + dashPick(g.opp, D.lineBy); market = ctxEsc(tip); line = oddsStr(g.opp.odds);
      }
      else {
        // No +EV edge: the model's lean (predicted winner) and that side's best moneyline at the user's books.
        const p = g.pred, homeWins = dashFin(p.home_win_prob) ? +p.home_win_prob >= 0.5 : null;
        if (homeWins == null) { market = "<span class='muted'>—</span>"; line = ""; }
        else {
          const side = homeWins ? "home" : "away", m = D.mls && D.mls.get(String(g.game_pk));
          const bestMl = m ? tileBest(m[`${side}_prices`], m[`${side}_price`], m[`${side}_book`], settings) : null;
          tip = `${dashTeam(homeWins ? g.home : g.away, g.sport)} ML — model lean, not a +EV pick`;
          market = `<span class="muted">${ctxEsc(dashTeam(homeWins ? g.home : g.away, g.sport))} ML</span>`;
          line = bestMl ? oddsStr(bestMl.am) : "";
        }
      }
      const label = dashMatchup(g.away, g.home, g.sport);
      return `<tr data-href="${dashHref(g.sport, g.game_pk)}"><td><span class="ca-dash-dot ${dot}" title="${dotTitle}"></span>${dashClock(g.commence)}</td>
        <td><span class="ca-team ca-dash-pair" title="${ctxEsc(label)}">${logoImg(g.away, g.sport)}${logoImg(g.home, g.sport)}<span class="ca-ell">${ctxEsc(label)}</span></span></td><td><span class="ca-ell ca-dash-mkt" title="${ctxEsc(tip)}">${market}</span></td><td>${line}</td>
        <td>${g.opp ? alphaCell(g.opp.alpha) : "<span class='muted'>—</span>"}</td><td class="ca-dash-actions">${starButton("games", g.game_pk, label)}${dashGo(g.sport, g.game_pk)}</td></tr>`;
    }).join("")}</tbody></table></div>`;
  }
  return `<section class="ca-card ca-dash-card" id="dash-slate"><div class="ca-card-head"><h2>Today’s Slate</h2>${pills("dash-slate", [["all", "All"], ["nfl", "NFL"], ["cfb", "CFB"], ["mlb", "MLB"], ["nba", "NBA"]], s)}<a class="ca-link" href="${allHref}">View All →</a></div>${body}</section>`;
}

/* ── Performance Snapshot ─────────────────────────────────────────────── */
function dashPerfCard(D) {
  const w = dashState.perf, from = w === "7d" ? dashAddDays(D.date, -6) : w === "30d" ? dashAddDays(D.date, -29) : "0000-00-00";
  const rows = D.pnl.filter((r) => r.game_date >= from), a = aggPnl(rows);
  const g = D.graded.filter((r) => r.date >= from), hr = hitRate(g);
  const edges = g.map((r) => r.edgePp).filter((x) => x != null), avgEdge = edges.length ? edges.reduce((s, x) => s + x, 0) / edges.length : null;
  const mini = (label, value, cls = "") => `<div class="ca-dash-mini"><span>${label}</span><b class="${cls}">${value}</b></div>`;
  // x labels on areaChart's own cadence (every ceil(n/7)th point + the last), minus the one that would collide with the last.
  // The card is narrow: label every 2nd slot of areaChart's own cadence (ceil(n/7)) plus the last point.
  const cum = dashCumUnits(rows, from, D.date), every = Math.max(1, Math.ceil(cum.length / 7)), step = cum.length > 14 ? 2 * every : every;
  const pts = cum.map((p, i) => ({ x: i === cum.length - 1 || (i % step === 0 && cum.length - 1 - i >= Math.max(2, step * 0.5)) ? dashShortDate(p.date) : null, y: p.units }));
  const chart = areaChart(pts, { h: 190, yTicks: 5 });
  return `<section class="ca-card ca-dash-card" id="dash-perf"><div class="ca-card-head"><h2>Performance Snapshot</h2>${pills("dash-perf", [["7d", "7D"], ["30d", "30D"], ["season", "Season"]], w)}<a class="ca-link" href="track-record.html">View Track Record →</a></div>
    <div class="ca-dash-minis">${mini("ROI", a.n ? pStr(a.roiPct) : "—", dashCls(a.roiPct))}${mini("Units", a.n ? uStr(a.units) : "—", dashCls(a.units))}${mini("Hit Rate", hr == null ? "—" : `${(hr * 100).toFixed(1)}%`)}${mini("Avg. Edge", avgEdge == null ? "—" : pStr(avgEdge), dashCls(avgEdge))}</div>
    ${chart || dashEmpty("No graded bets in this window yet.")}</section>`;
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
      const team = dashTeam(m.side === "home" ? p.home_team_name : p.away_team_name, p.sport), mu = dashMatchup(p.away_team_name, p.home_team_name, p.sport);
      let title, chg, d;
      if (m.market === "moneyline") {
        d = (americanToProb(m.cur_price) - americanToProb(m.open_price)) * 100;
        title = `${team} ML ${oddsStr(m.open_price)} → ${oddsStr(m.cur_price)}`; chg = dashSigned(d, 1, "%");
      } else {
        d = +m.cur_line - +m.open_line;
        title = m.market === "total" ? `${teamShort(p.away_team_name, p.sport)} @ ${teamShort(p.home_team_name, p.sport)} O/U ${+m.open_line} → ${+m.cur_line}` : `${team} ${dashLine(+m.open_line)} → ${dashLine(+m.cur_line)}`;
        chg = `${dashSigned(d, 1)} pts`;
      }
      const sp = D.splits && D.splits.get(`${m.game_pk}|${m.market}|${m.side}`);
      const spTxt = sp && dashFin(sp.ticket_pct) ? ` • ${Math.round(+sp.ticket_pct)}% of tickets on ${m.market === "total" ? (m.side === "under" ? "Under" : "Over") : team}` : "";
      rows.push(dashMoverRow({ href: dashHref(p.sport, p.game_pk), icon: m.market === "total" ? ICON_TREND : logoImg(m.side === "home" ? p.home_team_name : p.away_team_name, p.sport),
        title: ctxEsc(title), sub: ctxEsc(`${mu}${spTxt}`), chg, chgCls: dashCls(d), when: m.cur_at ? dashAgo(m.cur_at, D.nowMs) : "" }));
      if (rows.length >= 5) break;
    }
    empty = "No line-move history yet — Pinnacle open → current moves appear once the line-moves view is live.";
  } else if (tab === "model") {
    rows = dashModelVsMarket(D.datePreds).slice(0, 5).map((x) => {
      const p = x.pred, mu = dashMatchup(p.away_team_name, p.home_team_name, p.sport);
      const title = x.market === "spread" ? `${dashTeam(p.home_team_name, p.sport)} ${dashLine(x.from)} → ${dashLine(x.to)}`
        : `${teamShort(p.away_team_name, p.sport)} @ ${teamShort(p.home_team_name, p.sport)} O/U ${x.from} → ${x.to}`;
      return dashMoverRow({ href: dashHref(p.sport, p.game_pk), icon: x.market === "total" ? ICON_TREND : logoImg(p.home_team_name, p.sport), title: ctxEsc(title),
        sub: ctxEsc(`Market → model ${x.market} · ${mu}`), chg: `${dashSigned(x.diff, 1)} pts`, chgCls: dashCls(x.diff), when: dashClock(p.commence_time) });
    });
    empty = "No model vs. market gaps for this date.";
  } else if (tab === "new") {
    rows = dashNewSignals(D.opps, D.hist, D.nowMs).slice(0, 5).map(({ opp: o, at }) => {
      const [away, home] = dashSides(o.matchup);
      return dashMoverRow({ href: dashHref(o.sport, o.game_pk), icon: dashPickLogo(o) || dashLeagueLogo(o.sport),
        title: ctxEsc(`${o.kind === "prop" ? `${o.playerName} ` : ""}${dashPick(o, D.lineBy)}`), sub: ctxEsc(`${dashMatchup(away, home, o.sport)} • Alpha ${o.alpha} • ${o.tier}`),
        chg: `${pStr(o.evPct)} EV`, chgCls: "pos", when: dashAgo(at, D.nowMs) });
    });
    empty = "No new signals in the last 24 hours.";
  } else {
    rows = D.gaps.slice(0, 5).map((r) => {
      const [away, home] = dashSides(r.matchup), o = { kind: "line", ...r };
      return dashMoverRow({ href: dashHref(r.sport, r.game_pk), icon: dashPickLogo(o) || ICON_TREND,
        title: ctxEsc(`${dashPick(o, D.lineBy)} ${oddsStr(r.best_price)}`), sub: ctxEsc(`${(BOOK_STYLE[String(r.best_book || "").toLowerCase()] || [r.best_book])[0]} vs. Pinnacle ${oddsStr(r.pinnacle_price)} • ${dashMatchup(away, home, r.sport)}`),
        chg: `${dashSigned(+r.soft_vs_sharp_gap * 100, 1, "%")}`, chgCls: "pos", when: dashClock(r.commence_time) });
    });
    empty = "No stale lines for this date.";
  }
  const tabs = pills("dash-movers", [["moves", "Line Moves"], ["model", "Model vs. Market"], ["new", "New Signals"], ["stale", "Stale Lines"]], tab);
  return `<section class="ca-card ca-dash-card" id="dash-movers"><div class="ca-card-head"><h2>Market Movers</h2></div><div class="ca-dash-tabs">${tabs}</div>
    <div class="ca-dash-mvlist">${rows.length ? rows.join("") : dashEmpty(empty)}</div></section>`;
}

/* ── Portfolio & Exposure ─────────────────────────────────────────────── */
function dashExpoCard(D) {
  const e = D.expo, mode = dashState.expo;
  const segs = mode === "sport"
    ? e.bySport.map((x) => ({ label: DASH_SPORT[x.sport] || String(x.sport).toUpperCase(), pct: x.pct, units: x.units, color: DASH_COLOR[x.sport] || "var(--muted)" }))
    : e.byMarket.filter((x) => x.staked > 0).map((x) => ({ label: x.label, pct: x.pct, units: x.units, color: DASH_MKT_COLOR[DASH_MKT_TYPES.findIndex(([l]) => l === x.label)] }));
  const tabs = pills("dash-expo", [["sport", "By Sport"], ["market", "By Market Type"]], mode);
  const body = e.staked
    ? `<div class="ca-dash-expo">${dashDonut(segs, { size: 96, stroke: 14, center: uStr(e.totalUnits), caption: "Total Units" })}${donutLegend(segs.map((s) => ({ label: s.label, pct: s.pct, value: uStr(s.units), color: s.color })))}</div>
      <div class="ca-dash-mtx"><h3>Market Type Exposure</h3>${e.byMarket.map((x) => `<div class="ca-dash-mtx-row"><span>${x.label}</span><i><b style="width:${x.pct.toFixed(1)}%"></b></i><em>${Math.round(x.pct)}%</em></div>`).join("")}</div>`
    : dashEmpty("No graded bets in the last 30 days.");
  return `<section class="ca-card ca-dash-card" id="dash-expo"><div class="ca-card-head"><h2>Portfolio &amp; Exposure</h2></div><div class="ca-dash-tabs ca-dash-tabs-2">${tabs}</div>${body}</section>`;
}

/* ── Watchlist ────────────────────────────────────────────────────────── */
function dashWatchBody(D) {
  const w = watchlistGet(), tab = dashState.watch, byPk = new Map(D.preds.map((r) => [String(r.game_pk), r]));
  const bestFor = (sport, pk) => D.tiered.filter((o) => o.sport === sport && String(o.game_pk) === String(pk)).sort((a, b) => b.alpha - a.alpha)[0];
  const row = (star, main, pick, when, href) => `<div class="ca-dash-wl-row">${star}<a href="${href}"><b>${main}</b>${pick ? `<small>${pick}</small>` : ""}</a><span class="muted">${when}</span></div>`;
  const oppTxt = (o) => `${o.kind === "prop" ? `${String(o.playerName || "").split(/\s+/).slice(-1)[0]} ` : ""}${dashPick(o, D.lineBy)} ${oddsStr(o.odds)}`;
  let rows = [];
  if (tab === "games") {
    rows = w.games.map((id) => byPk.get(id)).filter(Boolean).sort((a, b) => dashTime(a.commence_time) - dashTime(b.commence_time)).map((p) => {
      const mu = dashMatchup(p.away_team_name, p.home_team_name, p.sport), o = bestFor(p.sport, p.game_pk);
      return row(starButton("games", p.game_pk, mu), ctxEsc(mu), o ? ctxEsc(oppTxt(o)) : "", dashKick(p.commence_time), dashHref(p.sport, p.game_pk));
    });
  } else if (tab === "teams") {
    rows = w.teams.map((id) => {
      const up = String(id).toUpperCase();
      const hit = (n, s) => n && (n === id || dashTeam(n, s) === id || String(teamShort(n, s)).toUpperCase() === up);
      const p = D.preds.filter((r) => hit(r.home_team_name, r.sport) || hit(r.away_team_name, r.sport)).sort((a, b) => dashTime(a.commence_time) - dashTime(b.commence_time))[0];
      if (!p) return row(starButton("teams", id, id), ctxEsc(id), "No upcoming game", "", "#watchlist");
      return row(starButton("teams", id, id), ctxEsc(id), ctxEsc(dashMatchup(p.away_team_name, p.home_team_name, p.sport)), dashKick(p.commence_time), dashHref(p.sport, p.game_pk));
    });
  } else {
    rows = w.players.map((id) => {
      const o = D.allOpps.filter((x) => x.kind === "prop" && x.playerName === id).sort((a, b) => b.alpha - a.alpha)[0];
      if (!o) return row(starButton("players", id, id), ctxEsc(id), "No +EV prop on the board", "", "#watchlist");
      return row(starButton("players", id, id), ctxEsc(id), ctxEsc(`${dashPick(o, D.lineBy)} ${oddsStr(o.odds)}`), dashKick(o.commence), dashHref(o.sport, o.game_pk));
    });
  }
  return rows.length ? rows.join("") : dashEmpty("Star games, teams or players to follow them here.");
}
function dashWatchCard(D) {
  return `<section class="ca-card ca-dash-card" id="watchlist"><div class="ca-card-head"><h2>Watchlist</h2></div><div class="ca-dash-tabs">${pills("dash-watch", [["games", "Games"], ["teams", "Teams"], ["players", "Players"]], dashState.watch)}</div>
    <div class="ca-dash-wl">${dashWatchBody(D)}</div></section>`;
}

/* ── page ─────────────────────────────────────────────────────────────── */
const DASH_CARDS = { "dash-opps": dashOppsCard, "dash-slate": dashSlateCard, "dash-perf": dashPerfCard, "dash-movers": dashMoversCard, "dash-expo": dashExpoCard, watchlist: dashWatchCard };
const DASH_PILL = { "dash-op": ["op", "dash-opps"], "dash-kind": ["kind", "dash-opps"], "dash-slate": ["slate", "dash-slate"], "dash-perf": ["perf", "dash-perf"],
  "dash-movers": ["movers", "dash-movers"], "dash-expo": ["expo", "dash-expo"], "dash-watch": ["watch", "watchlist"] };

async function buildDashboard() {
  const date = dashDate(), D = await dashLoad(date);
  window.__caDashData = D;
  const nav = `<div class="ca-datenav"><button class="ca-dn-btn" data-dash-date="${dashAddDays(date, -1)}" aria-label="Previous day">‹</button>
    <label class="ca-dn-date">${ICON_CAL}<span>${dashDateLabel(date)}</span><input type="date" value="${date}" data-dash-datepick aria-label="Pick a date"></label>
    <button class="ca-dn-btn" data-dash-date="${dashAddDays(date, 1)}" aria-label="Next day">›</button></div>`;
  return `<div class="ca-dash">${pageTitle("Today at a Glance", "Key opportunities, performance, and model insights across all sports.", nav)}
    ${dashStatCards(D)}
    <div class="ca-dash-main">
      <div class="ca-dash-col">${dashOppsCard(D)}<div class="ca-dash-sub ca-dash-sub-l">${dashPerfCard(D)}${dashMoversCard(D)}</div></div>
      <div class="ca-dash-col">${dashSlateCard(D)}<div class="ca-dash-sub ca-dash-sub-r">${dashExpoCard(D)}${dashWatchCard(D)}</div></div>
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
// Re-draw one card from the cached data (pill / star changes need no refetch). Stars inside the
// new markup are wired here (R5: wireShell only wires what exists at render time).
function dashRedraw(id) {
  const el = document.getElementById(id), D = window.__caDashData;
  if (!el || !D || !DASH_CARDS[id]) return;
  el.outerHTML = DASH_CARDS[id](D);
  const fresh = document.getElementById(id);
  if (fresh) wireStars(fresh);
}
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
    const tr = e.target.closest("tr[data-href]");
    if (tr && !e.target.closest("a,button")) location.href = tr.dataset.href;
  });
  const inp = root.querySelector("[data-dash-datepick]");
  if (inp) inp.addEventListener("change", () => { if (/^\d{4}-\d{2}-\d{2}$/.test(inp.value)) dashGoDate(inp.value); });
  // Star toggles (wired by wireShell, which stops propagation) -> refresh the Watchlist after the toggle lands.
  // The same game can have a star in the Slate and in the Watchlist: re-sync every copy, then redraw the Watchlist.
  root.addEventListener("click", (e) => {
    const b = e.target.closest("[data-star-kind]");
    if (!b) return;
    setTimeout(() => {
      const on = watchlistHas(b.dataset.starKind, b.dataset.starId);
      root.querySelectorAll("[data-star-kind]").forEach((x) => {
        if (x.dataset.starKind !== b.dataset.starKind || x.dataset.starId !== b.dataset.starId) return;
        x.classList.toggle("on", on); x.textContent = on ? "★" : "☆"; x.setAttribute("aria-pressed", String(on));
      });
      dashRedraw("watchlist");
    }, 0);
  }, true);
}
