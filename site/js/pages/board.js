/* Sport pages (nfl.html / cfb.html / mlb.html / nba.html). Real data only: every card has an empty state and no number
   is ever invented (the mockup's numbers are placeholders).
   Layout: a title row (sport, subtitle, Power Rankings link on NFL / CFB, the period navigator and the period's date range),
   then three columns (left ~23%, centre ~50%, right ~27%). The centre column holds the "This Week's Games" card; the
   left / right columns are mount points (BOARD_SLOTS) other files fill by registering a renderer in BOARD_CARDS[slotId] =
   (D) => html, and by pushing an async (D) => void into BOARD_LOADERS to load extra data into D. An unfilled slot is an empty,
   hidden div, so nothing fake is ever drawn.
   Period: NFL / CFB are browsed by season week (?week=N, Tuesday-to-Monday weeks, see data.js), MLB / NBA by day (?date=YYYY-MM-DD).
   The default (no param) is the current week / today. Period + view live in window.__caBoard AND in the URL (non-default
   values only); the market pill, team / time filters and sort live in window.__caBoard only. A period change refetches
   (render()), a filter change redraws the games card from the cached load (window.__caBoardData).
   Games card, two forms. A game with a graded final (prediction_accuracy) is "final": its row shows the final score, the
   closing lines (game_closing_prices for the moneyline, the stored market_spread / market_total), the model's lines and the
   W / L / P chip per market (trackRows, data.js). Every other game is "upcoming": market lines (game_moneylines_current for the
   moneyline, predictions market_spread / market_total), the model's lines and the edge, plus the game's best opportunity as
   the Alpha Score. A week holding both shows the finished games first (their own table), then the rest.
   All lines are shown from the market favourite's side (the favourite by the spread, else the model's pick, else home): market
   ML / spread / total, the model's fair ML (from home_win_prob) / spread (pred_margin, else the projected score) / total, and the
   Edge = model minus market: ML in probability points vs the best price's implied probability, spread in points on the
   favourite (model margin - market margin), total in points (model - market; + leans Over). The model's spread and total are
   rounded to the half point (modelLineStr / r05, the same as the game page), so the edge is exactly what the two columns show.
   NFL / CFB: "Power Rankings →" opens ?view=rankings, which renders the legacy sortable rankingsTable (app.js: rankingsCard,
   wireRankings) under the same title row.
   Depends on app.js (predictions, rankingsLoad, rankingsCard, wireRankings, etDateStr, ctxEsc, logoImg, teamShort, sb, sbAll,
   gameMoneylines, render), metrics.js, ui.js, shell.js (pageTitle, starButton) and data.js (slateRows, gameDate, week helpers,
   trackRows, trackClosing, safeCard, ...). */
const BOARD_RANKED = ["nfl", "cfb"];            // the sports that publish power rankings and are browsed by week
const BOARD_SUB = "Model projections, market edges, and performance tracking.";
const BOARD_MARKETS = [["all", "All Games"], ["moneyline", "Moneyline"], ["spread", "Spread"], ["total", "Total"], ["props", "Player Props"]];
const BOARD_SORTS = [["time", "Sort by: Start Time"], ["alpha", "Sort by: Alpha Score"], ["edge", "Sort by: Edge"]];
const BOARD_PRED_SELECT = "sport,game_pk,game_date,home_team_name,away_team_name,home_win_prob,pred_home_score,pred_away_score,commence_time,market_spread,market_total,model_version";
const BOARD_EMPTY = { mlb: "MLB model paused since Aug 31 (last projections Aug 31, 2026).", nba: "No NBA model yet. NBA projections are not live." };
const BOARD_DEFAULT = { week: null, date: null, view: "games" };
const BOARD_FILTER_DEFAULT = { market: "all", team: "", time: "", sort: "time" };
// Mount points for the cards of the left and right columns, top to bottom. A card registers BOARD_CARDS[id] = (D) => html.
const BOARD_SLOTS = {
  left: ["board-season-perf", "board-model-market", "board-market-intel", "board-key-insights"],
  right: ["board-top-alpha", "board-betting-splits", "board-model-projections"],
};
const BOARD_CARDS = {};      // slot id -> (D) => html (registered by the files that own the cards)
const BOARD_LOADERS = [];    // async (D) => void: load extra data into D (each failure is isolated)

/* ── pure helpers ─────────────────────────────────────────────────────── */
const boardIsDate = (s) => /^\d{4}-\d{2}-\d{2}$/.test(s || "") && !Number.isNaN(Date.parse(`${s}T12:00:00Z`)) && addDays(s, 0) === s;
// ?week=N (NFL / CFB, inside the league's week bounds), ?date=YYYY-MM-DD (MLB / NBA), ?view=rankings (NFL / CFB). null = the default period.
function boardParse(search, sport) {
  let p;
  try { p = new URLSearchParams(search || ""); } catch { p = new URLSearchParams(""); }
  const ranked = BOARD_RANKED.includes(sport), w = p.get("week"), d = p.get("date");
  const [lo, hi] = WEEK_BOUNDS[sport] || [1, 1];
  const week = ranked && /^\d{1,2}$/.test(w || "") && +w >= lo && +w <= hi ? +w : null;
  return { week, date: !ranked && boardIsDate(d) ? d : null, view: ranked && p.get("view") === "rankings" ? "rankings" : BOARD_DEFAULT.view };
}
// Only non-default values: an untouched page has a clean URL.
function boardQuery(s) {
  const p = new URLSearchParams();
  if (s.week != null) p.set("week", String(s.week));
  if (s.date != null) p.set("date", s.date);
  if (s.view !== BOARD_DEFAULT.view) p.set("view", s.view);
  return p.toString();
}
// The period being browsed: a season week (NFL / CFB) or a day (MLB / NBA), with its date window [from, to], whether it is the
// default (current) one and the previous / next values (null at the ends of the season).
function boardPeriod(sport, s, today) {
  if (BOARD_RANKED.includes(sport)) {
    const [lo, hi] = WEEK_BOUNDS[sport], season = seasonOf(today), def = clampWeek(sport, weekOf(sport, today));
    const week = s && s.week != null ? clampWeek(sport, s.week) : def, from = weekStart(sport, season, week);
    return { kind: "week", sport, season, week, from, to: addDays(from, 6), def, isCurrent: week === def, lo, hi, prev: week > lo ? week - 1 : null, next: week < hi ? week + 1 : null };
  }
  const date = s && s.date ? s.date : today;
  return { kind: "day", sport, date, from: date, to: date, def: today, isCurrent: date === today, prev: addDays(date, -1), next: addDays(date, 1) };
}
// The span shown in the date-range box: the dates the loaded games actually fall on (a Wednesday opener, a Friday CFB game),
// else the Thursday-to-Monday NFL window / the whole CFB window, else the day.
function boardSpan(period, games) {
  const ds = (games || []).map((g) => g.date).filter(Boolean).sort();
  if (ds.length) return rangeLabel(ds[0], ds[ds.length - 1]);
  if (period.kind === "day") return rangeLabel(period.date);
  return rangeLabel(period.sport === "nfl" ? addDays(period.from, 2) : period.from, period.to);
}
const boardNum = (x) => (x == null ? null : numOrNull(x));
const boardSortKey = (g) => { const v = timeMs(g.commence); return Number.isFinite(v) ? v : Date.parse(`${g.date || "9999-12-31"}T12:00:00Z`); };

// One entry per game of the period (slateRows: one per game, by kickoff, with its best opportunity), each with its graded row
// (`acc`) when it is final. A graded game with no projection row still appears (from prediction_accuracy).
function boardGames(D) {
  const accBy = new Map(), have = new Set();
  for (const a of (D.acc || [])) if (a && a.actual_winner != null) accBy.set(String(a.game_pk), a);
  const out = slateRows(D.preds || [], D.opps || []).map((g) => { have.add(String(g.game_pk)); const acc = accBy.get(String(g.game_pk)) || null; return { ...g, acc, final: !!acc, date: gameDate(g.pred) || (acc && acc.game_date) || "" }; });
  accBy.forEach((a, pk) => { if (!have.has(pk)) out.push({ sport: a.sport, game_pk: a.game_pk, away: a.away_team_name, home: a.home_team_name, commence: null, pred: null, opp: null, acc: a, final: true, date: a.game_date || "" }); });
  return out.sort((a, b) => boardSortKey(a) - boardSortKey(b) || String(a.game_pk).localeCompare(String(b.game_pk)));
}

// The market, the model and the edge of one game, all from the favourite's side. Missing numbers are null (shown as a dash).
//   mkt = { ml (American price), spread (the favourite's line), total }, model = { ml (fair American), spread, total }, edge = { ml (prob pts), spread (pts), total (pts) }.
// A final game uses the graded row's model numbers and the closing prices (D.closing: "game_pk|side" -> American); an upcoming game
// uses the projection and the best current prices (D.mlBy: game_pk -> game_moneylines_current row).
function boardLines(g, D) {
  const a = g.acc, p = g.pred || {}, pk = String(g.game_pk);
  let pHome, margin, total;
  if (a) {
    const wp = boardNum(a.win_prob), conf = wp == null ? null : Math.max(wp, 1 - wp);
    pHome = conf == null ? boardNum(p.home_win_prob) : a.predicted_winner === a.home_team_name ? conf : a.predicted_winner === a.away_team_name ? 1 - conf : wp;
    margin = boardNum(a.pred_margin); total = boardNum(a.pred_total);
  } else { pHome = boardNum(p.home_win_prob); margin = null; total = boardNum(p.pred_total); }
  const h = boardNum(p.pred_home_score), aw = boardNum(p.pred_away_score);
  if (margin == null && h != null && aw != null) margin = h - aw;
  if (total == null && h != null && aw != null) total = h + aw;
  const mktSpread = boardNum(a && a.market_spread != null ? a.market_spread : p.market_spread), mktTotal = boardNum(a && a.market_total != null ? a.market_total : p.market_total);
  const prices = a ? { home: D.closing && D.closing.get(`${pk}|home`), away: D.closing && D.closing.get(`${pk}|away`) }
    : (() => { const m = D.mlBy && D.mlBy.get(pk); return m ? { home: boardNum(m.home_price), away: boardNum(m.away_price) } : {}; })();
  const hasAny = pHome != null || margin != null || mktSpread != null || prices.home != null || prices.away != null;
  const fav = !hasAny ? null : mktSpread != null && mktSpread !== 0 ? (mktSpread < 0 ? "home" : "away")
    : pHome != null && pHome !== 0.5 ? (pHome > 0.5 ? "home" : "away") : margin != null && margin !== 0 ? (margin > 0 ? "home" : "away") : "home";
  const sign = fav === "away" ? -1 : 1;
  const favName = fav === "away" ? g.away : g.home;
  const pFav = pHome == null ? null : fav === "away" ? 1 - pHome : pHome;
  const mktMl = fav ? (fav === "home" ? prices.home : prices.away) ?? null : null;
  const mktLine = fav && mktSpread != null ? sign * mktSpread : null;          // the favourite's line (negative when it is the favourite)
  const modLine = fav && margin != null ? r05(-sign * margin) || 0 : null;     // the model's numbers are quoted to the half point everywhere (modelLineStr, game page), so the edge is exactly model - market as shown
  if (total != null) total = r05(total);
  const modMl = pFav != null && pFav > 0 && pFav < 1 ? probToAmerican(pFav) : null;
  const edge = { ml: pFav != null && mktMl != null && Number.isFinite(americanToProb(mktMl)) ? (pFav - americanToProb(mktMl)) * 100 : null,
    spread: mktLine != null && modLine != null ? mktLine - modLine : null, total: mktTotal != null && total != null ? total - mktTotal : null };
  return { fav, favName, favAbbr: fav ? teamShort(favName, g.sport) : "", mkt: { ml: mktMl, spread: mktLine, total: mktTotal }, model: { ml: modMl, spread: modLine, total }, edge };
}

// Filters (team / kickoff label) then the sort. Edge sort = the largest |edge| of the selected market pill (all markets: the largest of the three).
function boardView(games, D, s) {
  let rows = games.filter((g) => (!s.team || g.away === s.team || g.home === s.team) && (!s.time || kickLabel(g.commence) === s.time));
  const edgeOf = (g) => { const e = boardLines(g, D).edge, v = (s.market === "moneyline" ? [e.ml] : s.market === "spread" ? [e.spread] : s.market === "total" ? [e.total] : [e.ml, e.spread, e.total]).filter((x) => x != null);
    return v.length ? Math.max(...v.map(Math.abs)) : -1; };
  if (s.sort === "alpha") rows = rows.slice().sort((a, b) => (b.opp ? b.opp.alpha : -1) - (a.opp ? a.opp.alpha : -1) || boardSortKey(a) - boardSortKey(b));
  else if (s.sort === "edge") rows = rows.slice().sort((a, b) => edgeOf(b) - edgeOf(a) || boardSortKey(a) - boardSortKey(b));
  return rows;
}
// The final score "BUF 27 – ATL 20" parts from the graded margin (home - away) and total, null when they do not give whole scores.
function boardScore(a) {
  const m = boardNum(a && a.actual_margin), t = boardNum(a && a.actual_total);
  if (m == null || t == null) return null;
  const home = (t + m) / 2, away = (t - m) / 2;
  return Number.isInteger(home) && Number.isInteger(away) && home >= 0 && away >= 0 ? { home, away } : null;
}

/* ── state <-> URL ────────────────────────────────────────────────────── */
function boardState(sport) {
  if (!window.__caBoard || window.__caBoard.sport !== sport) {
    let q = "";
    try { q = location.search; } catch { q = ""; }
    window.__caBoard = { sport, ...boardParse(q, sport), ...BOARD_FILTER_DEFAULT };
  }
  return window.__caBoard;
}
function boardSync(s) {
  try {
    const u = new URL(location.href);
    u.search = boardQuery(s);
    history.replaceState(null, "", u.toString());
  } catch { /* keep the current URL */ }
}
// Select a period (a week number / a YYYY-MM-DD date). The default period is stored as null so its URL stays clean.
function boardGoPeriod(sport, value, today) {
  const s = boardState(sport), per = boardPeriod(sport, { week: null, date: null }, today);
  if (per.kind === "week" ? !Number.isFinite(+value) : !boardIsDate(value)) return false;   // a value of the wrong kind never reaches the URL
  const v = per.kind === "week" ? clampWeek(sport, +value) : value;
  if (per.kind === "week") s.week = v === per.def ? null : v; else s.date = v === per.def ? null : v;
  boardSync(s);
  return true;
}
function boardSetFilter(sport, key, value) { boardState(sport)[key] = value; }
function boardSetView(sport, view) { const s = boardState(sport); s.view = view; boardSync(s); }

/* ── data ─────────────────────────────────────────────────────────────── */
async function boardLoad(sport, view) {
  if (view === "rankings") return { sport, view, today: etDateStr(new Date().toISOString()), R: await rankingsLoad(sport) };
  const s = boardState(sport), nowMs = Date.now(), today = etDateStr(new Date(nowMs).toISOString()), period = boardPeriod(sport, s, today);
  const live = LIVE_SPORTS.includes(sport), widen = (n) => addDays(period.from, n), upTo = addDays(period.to, 1);
  const inWin = (r) => { const d = gameDate(r); return !!d && d >= period.from && d <= period.to; };
  const [anyRows, curRows, accRows] = await Promise.all([
    sbAll(`predictions_any?sport=eq.${sport}&game_date=gte.${widen(-1)}&game_date=lte.${upTo}&select=${BOARD_PRED_SELECT}&order=commence_time.asc,game_pk.asc`).catch(() => []),
    period.to >= today ? predictions(sport).catch(() => []) : [],
    sbAll(`prediction_accuracy?sport=eq.${sport}&game_date=gte.${period.from}&game_date=lte.${period.to}&select=${TRACK_ACC_SELECT}&order=game_date.asc,game_pk.asc`).catch(() => []),
  ]);
  const by = new Map();
  for (const r of [...(anyRows || []), ...(curRows || [])]) if (r) by.set(String(r.game_pk), { ...by.get(String(r.game_pk)), ...r, sport });   // current rows win
  const preds = [...by.values()].filter(inWin);
  const acc = (accRows || []).filter((r) => r && r.actual_winner != null);
  const accSet = new Set(acc.map((r) => String(r.game_pk)));
  const pending = preds.some((r) => !accSet.has(String(r.game_pk)));
  const upcoming = live && period.to >= today;
  const [oppsAll, lineBy, mlMap, closingRows] = await Promise.all([
    upcoming ? loadOpportunities().catch(() => []) : [],
    upcoming ? evBestLines().catch(() => new Map()) : new Map(),
    pending ? gameMoneylines().catch(() => new Map()) : new Map(),
    acc.length ? trackClosing(acc.map((r) => r.game_pk)).catch(() => []) : [],
  ]);
  const pks = new Set(preds.map((r) => String(r.game_pk)));
  const D = { sport, view, nowMs, today, period, preds, acc, mlBy: mlMap, lineBy, closing: trackClosingMap(closingRows),
    opps: (oppsAll || []).filter((o) => o.tier && o.sport === sport && pks.has(String(o.game_pk))) };   // R9: an opportunity is a pick with a tier
  await Promise.all(BOARD_LOADERS.map((f) => Promise.resolve().then(() => f(D)).catch((e) => console.error("board loader failed", e))));
  return D;
}

/* ── title row ────────────────────────────────────────────────────────── */
const boardName = (sport) => SPORT_NAME[sport] || String(sport).toUpperCase();
const boardDash = `<span class="muted">—</span>`;
const boardDayLabel = (d) => new Date(`${d}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" });
const boardPageHref = (sport, view) => `${sport}.html${view === "rankings" ? "?view=rankings" : ""}`;

function boardNav(D, games) {
  const per = D.period, week = per.kind === "week";
  const step = (dir, label, glyph, cls, to) => `<button class="${cls}" data-board-step="${dir}" aria-label="${ctxEsc(label)}"${to == null ? " disabled" : ""}>${glyph}</button>`;
  let box;
  if (week) {
    const opts = Array.from({ length: per.hi - per.lo + 1 }, (_, i) => per.lo + i).map((w) => `<option value="${w}"${w === per.week ? " selected" : ""}>Week ${w}</option>`).join("");
    box = `<div class="ca-bd-period"><select class="ca-bd-weeksel" data-board-week aria-label="Week">${opts}</select>${step(1, "Next week", "›", "ca-bd-next", per.next)}</div>`;
  } else {
    box = `<div class="ca-bd-period"><label class="ca-bd-daypick"><span>${ctxEsc(boardDayLabel(per.date))}</span><input type="date" value="${ctxEsc(per.date)}" data-board-date aria-label="Pick a date"></label>${step(1, "Next day", "›", "ca-bd-next", per.next)}</div>`;
  }
  return `<div class="ca-bd-nav">${step(-1, week ? "Previous week" : "Previous day", "‹", "ca-dn-btn", per.prev)}${box}<div class="ca-daterange" id="board-daterange">${ICON_CAL}<span>${ctxEsc(boardSpan(per, games))}</span></div></div>`;
}
function boardTitleRight(D, games) {
  if (D.view === "rankings") return `<a class="ca-bd-rk" href="${boardPageHref(D.sport, "games")}">← Games</a>`;
  const rk = BOARD_RANKED.includes(D.sport) ? `<a class="ca-bd-rk" href="${boardPageHref(D.sport, "rankings")}">Power Rankings →</a>` : "";
  return `${rk}${boardNav(D, games)}`;
}

/* ── games card ───────────────────────────────────────────────────────── */
// "Sun 1:00 PM" as two short lines (day, time) so the narrow Kickoff column never wraps mid-time.
const boardKick = (iso) => { const k = kickLabel(iso), i = k.indexOf(" "); return !k ? boardDash : i < 0 ? ctxEsc(k) : `<span>${ctxEsc(k.slice(0, i))}</span><span>${ctxEsc(k.slice(i + 1))}</span>`; };
// American price; a blowout's -100000 reads "-100k" so it never overflows its narrow column (the exact price is in the hover title).
const boardOdds = (x) => (x == null ? boardDash : Math.abs(x) >= 10000 ? `<span title="${ctxEsc(oddsStr(x))}">${x < 0 ? "-" : "+"}${Math.round(Math.abs(x) / 1000)}k</span>` : oddsStr(x));
const boardAbbr = (name, sport) => ctxEsc(teamShort(name, sport));
const boardEdge = (x, unit = "") => (x == null ? boardDash : `<span class="${signCls(x)}">${signedStr(x, 1, unit)}</span>`);
const boardMktSpread = (L) => (L.mkt.spread == null ? boardDash : `<span class="ca-bd-sp"><i>${ctxEsc(L.favAbbr)}</i> ${lineStr(L.mkt.spread)}</span>`);
const boardModSpread = (L) => (L.model.spread == null ? boardDash : `<span class="ca-bd-sp"><i>${ctxEsc(L.favAbbr)}</i> ${modelLineStr(L.model.spread)}</span>`);
const boardTotal = (x) => (x == null ? boardDash : (+x).toFixed(1));

// One <col> per column: their widths are percentages in theme.css (per table form and per card width), so a narrow centre column never needs a horizontal scroll.
const boardCols = (n) => `<colgroup>${"<col>".repeat(n)}</colgroup>`;
function boardMatchup(g) {
  const label = `${g.away} @ ${g.home}`;
  return `<td class="ca-bd-match"><span class="ca-team ca-bd-pair" title="${ctxEsc(label)}">${starButton("games", g.game_pk, shortMatchup(g.away, g.home, g.sport))}<span class="ca-bd-side">${logoImg(g.away, g.sport)}<b>${boardAbbr(g.away, g.sport)}</b></span><i>@</i><span class="ca-bd-side">${logoImg(g.home, g.sport)}<b>${boardAbbr(g.home, g.sport)}</b></span></span></td>`;
}
const boardGoCell = (g) => `<td class="ca-bd-go">${goLink(g.sport, g.game_pk)}</td>`;
const boardModelCells = (L) => `<td data-m="moneyline" class="ca-bd-gs">${boardOdds(L.model.ml)}</td><td data-m="spread">${boardModSpread(L)}</td><td data-m="total">${boardTotal(L.model.total)}</td>`;
// Sub-header cells (ML · Spread · Total) for the three market groups.
const boardSubHead = () => [0, 1, 2].map(() => [["moneyline", "ML"], ["spread", "Spread"], ["total", "Total"]]
  .map(([m, l], i) => `<th data-m="${m}" class="ca-bd-sub${i === 0 ? " ca-bd-gs" : ""}" title="${m === "moneyline" ? "Moneyline" : l} of the market favourite">${l}</th>`).join("")).join("");

function boardUpcomingTable(rows, D, s) {
  const head = `${boardCols(13)}<thead><tr><th rowspan="2" class="ca-bd-th-match">Matchup</th><th rowspan="2" class="ca-bd-th-time">Kickoff (ET)</th><th colspan="3" class="ca-bd-grp">Market</th><th colspan="3" class="ca-bd-grp">CappingAlpha</th><th colspan="3" class="ca-bd-grp">Edge</th><th rowspan="2">Alpha Score</th><th rowspan="2">View</th></tr><tr>${boardSubHead()}</tr></thead>`;
  const body = rows.map((g) => {
    const L = boardLines(g, D), tip = g.opp ? oppSlateLabel(g.opp, D.lineBy) : "";
    return `<tr data-href="${gameHref(g.sport, g.game_pk)}">${boardMatchup(g)}<td class="ca-bd-time">${boardKick(g.commence)}</td>
      <td data-m="moneyline" class="ca-bd-gs">${boardOdds(L.mkt.ml)}</td><td data-m="spread">${boardMktSpread(L)}</td><td data-m="total">${boardTotal(L.mkt.total)}</td>
      ${boardModelCells(L)}
      <td data-m="moneyline" class="ca-bd-gs">${boardEdge(L.edge.ml, "%")}</td><td data-m="spread">${boardEdge(L.edge.spread)}</td><td data-m="total">${boardEdge(L.edge.total)}</td>
      <td class="ca-bd-alpha"${tip ? ` title="${ctxEsc(tip)}"` : ""}>${g.opp ? alphaCell(g.opp.alpha) : boardDash}</td>${boardGoCell(g)}</tr>`;
  }).join("");
  return `<table class="ca-table ca-board-table ca-bd-up${s.market === "all" ? "" : ` ca-bd-em-${ctxEsc(s.market)}`}">${head}<tbody>${body}</tbody></table>`;
}

function boardFinalCell(g) {
  const a = g.acc, sc = boardScore(a), label = `${g.away} @ ${g.home}`;
  if (!sc) return `<td class="ca-bd-final" title="${ctxEsc(label)}"><b>${boardAbbr(a.actual_winner, g.sport)}</b> W</td>`;
  const t = (name, n, win) => `<span${win ? ` class="ca-bd-win"` : ""}>${boardAbbr(name, g.sport)} ${n}</span>`;
  return `<td class="ca-bd-final" title="${ctxEsc(`${g.away} ${sc.away}, ${g.home} ${sc.home}`)}">${t(g.away, sc.away, sc.away > sc.home)}${t(g.home, sc.home, sc.home > sc.away)}</td>`;
}
// One graded market's pick + W / L / P chip ("—" when the model made no pick, e.g. no market line).
function boardResultCell(m, tr, g, cls = "") {
  const r = tr && tr[m];
  if (!r) return `<td data-m="${m}" class="ca-bd-res${cls}">${boardDash}</td>`;
  const pick = m === "total" ? (r.pick === "Over" ? "O" : "U") : ctxEsc(teamShort(r.pick, g.sport));
  const word = { W: "won", L: "lost", P: "pushed" }[r.result];
  const what = m === "total" ? `${r.pick} ${r.closing != null ? (+r.closing).toFixed(1) : ""}`.trim() : m === "spread" ? `${teamShort(r.pick, g.sport)} ${r.closing != null ? lineStr(r.closing) : ""}`.trim() : `${teamShort(r.pick, g.sport)} ML`;
  return `<td data-m="${m}" class="ca-bd-res${cls}"><span class="ca-bd-pk">${pick}</span>${resultChip(r.result, `Model pick ${what} ${word}`)}</td>`;
}
function boardFinalTable(rows, D, s) {
  const trs = new Map();
  for (const r of trackRows(rows.map((g) => g.acc), [], D.closing)) { const m = trs.get(String(r.game_pk)) || {}; m[r.market] = r; trs.set(String(r.game_pk), m); }
  const head = `${boardCols(13)}<thead><tr><th rowspan="2" class="ca-bd-th-match">Matchup</th><th rowspan="2" class="ca-bd-th-time">Date</th><th rowspan="2">Final</th><th colspan="3" class="ca-bd-grp">Closing</th><th colspan="3" class="ca-bd-grp">CappingAlpha</th><th colspan="3" class="ca-bd-grp">Result</th><th rowspan="2">View</th></tr><tr>${boardSubHead()}</tr></thead>`;
  const body = rows.map((g) => {
    const L = boardLines(g, D), tr = trs.get(String(g.game_pk));
    return `<tr data-href="${gameHref(g.sport, g.game_pk)}">${boardMatchup(g)}<td class="ca-bd-time">${g.date ? ctxEsc(shortDate(g.date)) : boardDash}</td>${boardFinalCell(g)}
      <td data-m="moneyline" class="ca-bd-gs">${boardOdds(L.mkt.ml)}</td><td data-m="spread">${boardMktSpread(L)}</td><td data-m="total">${boardTotal(L.mkt.total)}</td>
      ${boardModelCells(L)}
      ${boardResultCell("moneyline", tr, g, " ca-bd-gs")}${boardResultCell("spread", tr, g)}${boardResultCell("total", tr, g)}${boardGoCell(g)}</tr>`;
  }).join("");
  return `<table class="ca-table ca-board-table ca-bd-fin${s.market === "all" ? "" : ` ca-bd-em-${ctxEsc(s.market)}`}">${head}<tbody>${body}</tbody></table>`;
}

// The Player Props pill: the prop opportunities (a tiered pick, R9) of the shown games.
function boardPropsTable(rows, D) {
  const pks = new Set(rows.map((g) => String(g.game_pk))), by = new Map(rows.map((g) => [String(g.game_pk), g]));
  const props = (D.opps || []).filter((o) => o.kind === "prop" && pks.has(String(o.game_pk)));
  if (!props.length) return emptyMsg("No player-prop opportunities for these games.");
  const body = props.map((o) => { const g = by.get(String(o.game_pk)), t = oppLabel(o, D.lineBy);
    return `<tr data-href="${gameHref(o.sport, o.game_pk)}">${boardMatchup(g)}<td class="ca-bd-time">${ctxEsc(kickLabel(o.commence))}</td><td><span class="ca-ell ca-bd-mkt" title="${ctxEsc(t)}">${ctxEsc(t)}</span></td>
      <td><span class="ca-bd-odds">${bookBadge(o.book)}${oddsStr(o.odds)}</span></td><td>${pct1(o.modelProb)}</td><td>${alphaCell(o.alpha)}${confPill(o.tier)}</td>${boardGoCell(g)}</tr>`; }).join("");
  return `<table class="ca-table ca-board-table ca-bd-props"><thead><tr><th>Matchup</th><th>Kickoff (ET)</th><th>Player Prop</th><th>Best Price</th><th>Model Prob.</th><th>Alpha</th><th>View</th></tr></thead><tbody>${body}</tbody></table>`;
}

function boardHeading(D, games) {
  const per = D.period, name = boardName(D.sport), n = games.length, nFinal = games.filter((g) => g.final).length, allFinal = n > 0 && nFinal === n;
  if (per.kind === "week") {
    const wk = `Week ${per.week}`;
    if (allFinal) return { title: `${wk} Results`, sub: `Final scores, model picks and results for ${wk}.` };
    const title = per.isCurrent ? "This Week's Games" : `${wk} Games`;
    return { title, sub: nFinal ? `Final scores and model results so far, then projections and lines for the games still to play in ${wk}.` : `Model projections, current lines, and edges for all ${wk} matchups.` };
  }
  const d = shortDate(per.date);
  if (allFinal) return { title: `Results · ${d}`, sub: `Final scores, model picks and results for ${d}.` };
  return { title: per.isCurrent ? "Today's Games" : `Games · ${d}`, sub: `Model projections, current lines, and edges for all ${name} games ${per.isCurrent ? "today" : `on ${d}`}.` };
}
const boardEmptyText = (D) => (D.period.kind === "week" ? `No ${boardName(D.sport)} games in Week ${D.period.week}.` : `No ${boardName(D.sport)} games on ${shortDate(D.period.date)}.`);

function boardTools(D, games) {
  const s = boardState(D.sport), propsOk = (D.opps || []).some((o) => o.kind === "prop");
  const items = BOARD_MARKETS.filter(([k]) => k !== "props" || propsOk), active = items.some(([k]) => k === s.market) ? s.market : "all";
  const sel = (key, value, opts, cls = "") => `<select class="ca-select ca-bd-sel ${cls}" data-board="${key}" aria-label="${ctxEsc(opts[0][1])}">${opts.map(([k, l]) => `<option value="${ctxEsc(k)}"${k === value ? " selected" : ""}>${ctxEsc(l)}</option>`).join("")}</select>`;
  const teams = [...new Set(games.flatMap((g) => [g.away, g.home]).filter(Boolean))].sort((a, b) => shortTeam(a, D.sport).localeCompare(shortTeam(b, D.sport)));
  const times = [...new Set(games.map((g) => kickLabel(g.commence)).filter(Boolean))];
  return `<div class="ca-bd-tools">${pills("board-market", items, active)}<div class="ca-bd-filters">${sel("team", s.team, [["", "All Teams"], ...teams.map((t) => [t, shortTeam(t, D.sport)])])}${sel("time", s.time, [["", "All Times"], ...times.map((t) => [t, t])])}${sel("sort", s.sort, BOARD_SORTS, "ca-bd-sortsel")}</div></div>`;
}

function boardGamesCard(D) {
  const s = boardState(D.sport), games = boardGames(D), h = boardHeading(D, games);
  if (!games.length && !LIVE_SPORTS.includes(D.sport)) h.sub = "";     // a paused / not-live sport: the empty state says why, the usual subtitle would contradict it
  let body;
  if (!games.length) body = emptyMsg(!LIVE_SPORTS.includes(D.sport) && BOARD_EMPTY[D.sport] ? BOARD_EMPTY[D.sport] : boardEmptyText(D));
  else {
    const rows = boardView(games, D, s), fin = rows.filter((g) => g.final), up = rows.filter((g) => !g.final);
    if (s.market === "props" && (D.opps || []).some((o) => o.kind === "prop")) body = boardPropsTable(rows, D);
    else if (!rows.length) body = emptyMsg("No games match these filters.");
    else body = `${fin.length ? `${up.length ? `<p class="ca-bd-sec">Final · ${fin.length}</p>` : ""}<div class="ca-table-wrap">${boardFinalTable(fin, D, s)}</div>` : ""}${up.length ? `${fin.length ? `<p class="ca-bd-sec">Upcoming · ${up.length}</p>` : ""}<div class="ca-table-wrap">${boardUpcomingTable(up, D, s)}</div>` : ""}`;
  }
  return `<section class="ca-card ca-board-card ca-bd-games" id="board-games"><div class="ca-bd-head"><h2>${ctxEsc(h.title)}</h2>${h.sub ? `<p>${ctxEsc(h.sub)}</p>` : ""}</div>${games.length ? boardTools(D, games) : ""}${body}</section>`;
}

function boardRankingsCard(D) { return rankingsCard(D.sport, D.R.rows, D.R.st); }
const boardSlot = (id, D) => (BOARD_CARDS[id] ? safeCard(id, BOARD_CARDS[id], D, "ca-card ca-board-card", id) : `<div class="ca-board-slot" id="${id}" data-board-slot="${id}"></div>`);

/* ── page ─────────────────────────────────────────────────────────────── */
async function buildBoardPage(sport) {
  const s = boardState(sport), D = await boardLoad(sport, s.view);
  window.__caBoardData = D;
  const title = pageTitle(boardName(sport), BOARD_SUB, D.view === "rankings" ? boardTitleRight(D, []) : boardTitleRight(D, boardGames(D)));
  if (D.view === "rankings") return `<div class="ca-board" data-sport="${ctxEsc(sport)}">${title}${safeCard("Power Rankings", boardRankingsCard, D, "ca-card ca-rk-card", "rk-card")}</div>`;
  const col = (side) => `<div class="ca-board-col" id="board-${side}">${BOARD_SLOTS[side].map((id) => boardSlot(id, D)).join("")}</div>`;
  return `<div class="ca-board" data-sport="${ctxEsc(sport)}">${title}<div class="ca-board-main">${col("left")}<div class="ca-board-col ca-board-center" id="board-center">${safeCard("Games", boardGamesCard, D, "ca-card ca-board-card", "board-games")}</div>${col("right")}</div></div>`;
}

// Re-draw the games card from the cached load (a filter change needs no refetch).
function boardRedraw() {
  const D = window.__caBoardData;
  if (!D || D.view !== "games") return;
  const games = document.getElementById("board-games");
  if (games) games.outerHTML = safeCard("Games", boardGamesCard, D, "ca-card ca-board-card", "board-games");
}
// Move to another period: URL + a full re-render (it needs different data).
function boardGo(value) {
  const D = window.__caBoardData;
  if (!D) return;
  if (!boardGoPeriod(D.sport, value, D.today)) return;
  try { window.scrollTo(0, 0); } catch { /* no window */ }
  render();
}

function wireBoardPage() {
  const root = document.querySelector(".ca-board");
  if (!root) return;
  root.addEventListener("click", (e) => {
    const D = window.__caBoardData;
    if (!D) return;
    const pill = e.target.closest("[data-pill]"), step = e.target.closest("[data-board-step]"), pick = e.target.closest(".ca-bd-daypick");
    if (pill && pill.dataset.pill === "board-market") { boardSetFilter(D.sport, "market", pill.dataset.key); boardRedraw(); return; }
    if (step && D.period && !step.disabled) { const to = +step.dataset.boardStep < 0 ? D.period.prev : D.period.next; if (to != null) boardGo(to); return; }
    if (pick && !e.target.matches("input")) { const inp = pick.querySelector("input"); try { inp.showPicker(); } catch { inp.focus(); } }
  });
  root.addEventListener("change", (e) => {
    const D = window.__caBoardData, t = e.target;
    if (!D || !t || !t.matches) return;
    if (t.matches("[data-board]")) { boardSetFilter(D.sport, t.dataset.board, t.value); boardRedraw(); }
    else if (t.matches("[data-board-week]")) boardGo(+t.value);
    else if (t.matches("[data-board-date]") && boardIsDate(t.value)) boardGo(t.value);
  });
  wireRankings();   // the Power Rankings view's sort headers + conference select (no-op in the Games view)
  // Stars and row clicks are handled by shell.js's delegated listeners.
}
