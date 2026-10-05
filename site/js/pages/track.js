/* Track Record (track-record.html) — mockup #4. Real data only: every card has an empty state and the mockup's numbers
   are never rendered. The main view ("Model picks") is the model's graded predictions: one row per graded market
   (moneyline / spread / total) from prediction_accuracy, scoped to the published record (track_record_start) and to
   the range pills; the "Record" select (filter row) switches the source: "Model picks" (default), "+EV picks" (the legacy profit
   tracker and +EV / prop sections, same data as before) and the pre-restart NFL archive.
   Range, archive, view, league tab, segment tab, filters, sort, search and the page size live in window.__caTrk AND in
   the URL (so a reload or share keeps them) and survive the 5-minute re-render; changing them redraws from the cached
   load (window.__caTrkData) without a refetch (only the +EV view refetches, it needs different data).
   Probabilities are the MODEL's (win_prob); the prediction carries no spread / total probability, so those rows show
   "—". prediction_pnl (per-pick P&L) times out under the anon 3s statement limit and is never requested, so `units`
   stays null in the page (trackRows still joins it when given); the moneyline closing price comes from
   game_closing_prices (market = moneyline, fetched per chunk of game ids: the unfiltered view scan is near the limit), spread / total closing
   lines are the stored market_spread / market_total.
   Depends on app.js (sb, trackRecordStarts, inTrackRecord, etDateStr, ctxEsc, logoImg, getSettings, unitLabel, pnlSection,
   wirePnl, wirePropGames, evResultsRows, evGradedPicks, propResultsRows, propLineGradesRows, parlayResultsRows,
   evTrackSection, propAccuracySection, propGamesSection, propTrackSection, gradedParlaysSection, r05, pct1, pStr,
   render), metrics.js, ui.js, shell.js and data.js. */
const TRACK_BREAKEVEN = 0.524;                       // -110 breakeven
const TRACK_PAGE = 25, TRACK_SHOW_MAX = 1000, TRACK_ROWS_CAP = 1000, TRACK_CLOSING_CHUNK = 60;
const TRACK_RANGES = [["7d", "7D"], ["30d", "30D"], ["season", "Season"], ["all", "All Time"]];
const TRACK_CHARTS = [["overall", "Overall"], ["nfl", "NFL"], ["cfb", "CFB"], ["mlb", "MLB"], ["nba", "NBA"]];
const TRACK_MARKET_PILLS = [["all", "All Predictions"], ["moneyline", "Moneyline"], ["spread", "Spread"], ["total", "Totals"]];
const TRACK_MARKET_OPTS = [["all", "All Markets"], ["moneyline", "Moneyline"], ["spread", "Spread"], ["total", "Total"]];
const TRACK_MARKETS = ["moneyline", "spread", "total"];
const TRACK_MKT_LABEL = { moneyline: "Moneyline", spread: "Spread", total: "Total" };
const TRACK_MKT_PLURAL = { moneyline: "Moneyline", spread: "Spreads", total: "Totals" };
const TRACK_BANDS = [[0.5, 0.55], [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 1.0001]];
const TRACK_BAND_KEYS = ["50-55", "55-60", "60-65", "65-70", "70+"];
const TRACK_BAND_LABELS = ["50–55%", "55–60%", "60–65%", "65–70%", "70%+"];
const TRACK_CONF_OPTS = [["", "All Confidence"], ...TRACK_BAND_KEYS.map((k, i) => [k, TRACK_BAND_LABELS[i]])];
const TRACK_RESULT_OPTS = [["", "All Results"], ["W", "Win"], ["L", "Loss"], ["P", "Push"]];
const TRACK_SORTS = [["recent", "Most Recent"], ["conf", "Highest Confidence"]];
const TRACK_SEGS = [["market", "Market"], ["league", "League"], ["confidence", "Confidence"]];
// The Record select (F2): one control for the three sources. "archive" = the model view scoped to the pre-restart NFL rows (state.archive).
const trackRecordKey = (s) => (s.view === "ev" ? "ev" : s.archive ? "archive" : "model");
function trackRecordOpts(starts) {
  const st = starts && starts.get ? starts.get("nfl") : null;
  return [["model", "Model picks"], ["ev", "+EV picks"], ["archive", st ? `NFL archive (pre-${trackStartDay(st.starts_at)})` : "NFL archive"]];
}
const trackStartDay = (iso) => new Date(iso).toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric" });
const TRACK_DEFAULT = { range: "all", archive: false, view: "model", chart: "overall", market: "all", league: "", conf: "", result: "", sort: "recent", q: "", seg: "market", n: TRACK_PAGE };
const TRACK_PARAMS = ["range", "archive", "view", "chart", "market", "league", "conf", "result", "sort", "q", "seg", "n"];
const TRACK_ACC_SELECT = "sport,game_pk,game_date,home_team_name,away_team_name,win_prob,predicted_winner,actual_winner,winner_correct,pred_margin,actual_margin,pred_total,actual_total,market_spread,market_total,spread_pick_correct,total_pick_correct";
const TRACK_WIN = "#1E8E4E", TRACK_LOSS = "#A5BFEA", TRACK_PUSH = "#B9B3A5";

/* ── pure helpers ─────────────────────────────────────────────────────── */
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

// Date window [from, to] ending today (ET). Season = since Aug 1 of the season year; unknown range = All Time.
function trackRange(range, today) {
  if (range === "7d") return { from: addDays(today, -6), to: today };
  if (range === "30d") return { from: addDays(today, -29), to: today };
  if (range === "season") { const y = +today.slice(0, 4), m = +today.slice(5, 7); return { from: `${m >= 8 ? y : y - 1}-08-01`, to: today }; }
  return { from: "0000-00-00", to: today };
}
// Rows (carrying inRecord) inside the published record, or only the pre-restart archive with the toggle, then inside the range.
function trackScope(rows, { range = "all", archive = false } = {}, today) {
  const { from, to } = trackRange(range, today);
  return (rows || []).filter((r) => r && (r.inRecord !== false) !== !!archive && r.date >= from && r.date <= to);
}
// Win % = W / (W + L): a push is not decided.
function trackTally(rows) {
  const rs = rows || [], w = rs.filter((r) => r.result === "W").length, l = rs.filter((r) => r.result === "L").length, p = rs.filter((r) => r.result === "P").length;
  return { w, l, p, n: rs.length, pct: w + l ? w / (w + l) : null };
}
// Share of spread / total picks on the right side of the closing line, and how far that is above the 52.4% breakeven (pp).
function trackAccuracyVsLine(rows) {
  const t = trackTally((rows || []).filter((r) => r && (r.market === "spread" || r.market === "total")));
  const n = t.w + t.l;
  return n ? { share: t.w / n, n, vsMarket: (t.w / n - TRACK_BREAKEVEN) * 100 } : null;
}
// Running accuracy-vs-line share after each game day with a decided spread / total pick (sparkline input).
function trackShareSeries(rows) {
  const rs = (rows || []).filter((r) => r && (r.market === "spread" || r.market === "total") && r.won != null).sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  const out = [];
  let w = 0, n = 0;
  rs.forEach((r, i) => {
    n++; if (r.won) w++;
    if (i === rs.length - 1 || rs[i + 1].date !== r.date) out.push(w / n);
  });
  return out;
}
const trackBandIdx = (conf) => (finite(conf) ? TRACK_BANDS.findIndex(([lo, hi]) => +conf >= lo && +conf < hi) : -1);

// Table rows: league / market / confidence band / result / search narrow; sort recent (date desc) or confidence (desc, unknown last).
function trackFilter(rows, f = {}) {
  const q = String(f.q || "").trim().toLowerCase(), band = f.conf ? TRACK_BAND_KEYS.indexOf(f.conf) : -1;
  const out = (rows || []).filter((r) => r
    && (!f.market || f.market === "all" || r.market === f.market)
    && (!f.league || r.sport === f.league)
    && (!f.conf || (band >= 0 && r.market === "moneyline" && trackBandIdx(r.conf) === band))
    && (!f.result || r.result === f.result)
    && (!q || `${r.matchup || ""} ${r.pick || ""} ${r.sport || ""}`.toLowerCase().includes(q)));
  const byDate = (a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0);
  return out.sort(f.sort === "conf" ? (a, b) => (finite(b.conf) ? +b.conf : -1) - (finite(a.conf) ? +a.conf : -1) || byDate(a, b) : byDate);
}

// Win % per segment ("market" | "league" | "confidence"), decided picks only, best first (ties: more decided picks).
function trackSegments(rows, kind) {
  const defs = kind === "league" ? SPORTS.map((s) => [s, SPORT_NAME[s], (r) => r.sport === s])
    : kind === "confidence" ? TRACK_BANDS.map((b, i) => [TRACK_BAND_KEYS[i], TRACK_BAND_LABELS[i], (r) => r.market === "moneyline" && trackBandIdx(r.conf) === i])
    : TRACK_MARKETS.map((m) => [m, TRACK_MKT_PLURAL[m], (r) => r.market === m]);
  return defs.map(([key, label, test]) => ({ key, label, ...trackTally((rows || []).filter((r) => r && test(r))) }))
    .filter((s) => s.w + s.l > 0).sort((a, b) => b.pct - a.pct || b.w + b.l - (a.w + a.l));
}

// Confidence calibration (moneyline picks with a model probability): the metrics `calibration` bands as paired bars.
function trackCalibration(rows) {
  const ml = (rows || []).filter((r) => r && finite(r.prob) && (r.won === true || r.won === false));
  const bands = calibration(ml, TRACK_BANDS);
  const groups = bands.map((b, i) => ({ label: TRACK_BAND_LABELS[i], sub: `(${b.n})`, bars: b.n ? [
    { value: b.predicted * 100, color: "var(--navy)", label: `${Math.round(b.predicted * 100)}%` },
    { value: b.actual * 100, color: "var(--green)", label: `${Math.round(b.actual * 100)}%` }] : [] })).filter((g) => g.bars.length);
  return { bands, groups };
}

// Current streak (pushes neither extend nor break it), the last five results and their W-L-P. `rows` newest first.
function trackStreak(rows) {
  const rs = rows || [], dots = rs.slice(0, 5).map((r) => r.result);
  return { streak: currentStreak(rs), dots, w: dots.filter((d) => d === "W").length, l: dots.filter((d) => d === "L").length, p: dots.filter((d) => d === "P").length };
}
const trackDots = (dots) => `<span class="ca-trk-dots">${(dots || []).map((d) => `<i class="ca-trk-dot ${ctxEsc(d)}">${ctxEsc(d)}</i>`).join("")}</span>`;
// Recent form over the newest 10 / 25 / 50 / 100 picks (`rows` newest first).
const trackForms = (rows) => [10, 25, 50, 100].map((n) => ({ label: `Last ${n}`, n: Math.min(n, (rows || []).length), ...recentForm(rows, n) }));

// Model Calibration card (moneyline picks): Brier score, predicted vs actual, difference in points and the verdict chip.
function trackModelCalibration(rows) {
  const ml = (rows || []).filter((r) => r && finite(r.prob) && (r.won === true || r.won === false));
  if (!ml.length) return null;
  const predicted = ml.reduce((s, r) => s + +r.prob, 0) / ml.length, actual = ml.filter((r) => r.won).length / ml.length, diff = (actual - predicted) * 100;
  return { n: ml.length, brier: brier(ml), predicted, actual, diff, chip: Math.abs(diff) <= 3 + 1e-9 ? "Well Calibrated" : diff < 0 ? "Overconfident" : "Underconfident" };
}

// Running net wins (W - L) per calendar day, from a zero start the day before the first decided pick, through the last one.
function trackCumulative(rows, sport) {
  const rs = (rows || []).filter((r) => r && (r.won === true || r.won === false) && (!sport || sport === "overall" || r.sport === sport));
  if (!rs.length) return [];
  const by = new Map();
  for (const r of rs) by.set(r.date, (by.get(r.date) || 0) + (r.won ? 1 : -1));
  const dates = [...by.keys()].sort(), last = dates[dates.length - 1], out = [{ date: addDays(dates[0], -1), net: 0 }];
  let net = 0;
  for (let d = dates[0]; d <= last; d = addDays(d, 1)) { net += by.get(d) || 0; out.push({ date: d, net }); }
  return out;
}
const trackWeekStart = (d) => addDays(d, -((new Date(`${d}T12:00:00Z`).getUTCDay() + 6) % 7));   // Monday
// Record per Monday-Sunday week, oldest first.
function trackWeekly(rows) {
  const by = new Map();
  for (const r of rows || []) if (r && r.date) { const k = trackWeekStart(r.date); (by.get(k) || by.set(k, []).get(k)).push(r); }
  const keys = [...by.keys()].sort(), out = [];
  for (let w = keys[0]; keys.length && w <= keys[keys.length - 1]; w = addDays(w, 7)) out.push({ week: w, ...trackTally(by.get(w)) });   // empty weeks stay (n = 0) so bars never shift
  return out;
}

/* ── filter state <-> URL ─────────────────────────────────────────────── */
function trackParse(search) {
  let p;
  try { p = new URLSearchParams(search || ""); } catch { p = new URLSearchParams(""); }
  const get = (k) => p.get(k) || "", pick = (v, list, d) => (list.some(([k]) => k === v) ? v : d), n = parseInt(get("n"), 10);
  return { range: pick(get("range"), TRACK_RANGES, "all"), archive: get("archive") === "1", view: get("view") === "ev" ? "ev" : "model",
    chart: pick(get("chart"), TRACK_CHARTS, "overall"), market: pick(get("market"), TRACK_MARKET_OPTS, "all"),
    league: SPORTS.includes(get("league")) ? get("league") : "", conf: TRACK_BAND_KEYS.includes(get("conf")) ? get("conf") : "",
    result: ["W", "L", "P"].includes(get("result")) ? get("result") : "", sort: pick(get("sort"), TRACK_SORTS, "recent"), q: get("q").slice(0, 80),
    seg: pick(get("seg"), TRACK_SEGS, "market"), n: Number.isFinite(n) ? clip(n, TRACK_PAGE, TRACK_SHOW_MAX) : TRACK_PAGE };
}
// Only non-default values: an untouched page has a clean URL.
function trackQuery(s) {
  const p = new URLSearchParams();
  for (const key of TRACK_PARAMS) if (s[key] !== TRACK_DEFAULT[key]) p.set(key, key === "archive" ? "1" : String(s[key]));
  return p.toString();
}
function trackState() {
  if (!window.__caTrk) { let s = ""; try { s = location.search; } catch { s = ""; } window.__caTrk = trackParse(s); }
  return window.__caTrk;
}
function trackSync(s) {
  try {
    const u = new URL(location.href);
    u.search = trackQuery(s);
    history.replaceState(null, "", u.toString());
  } catch { /* keep the current URL */ }
}

/* ── data ─────────────────────────────────────────────────────────────── */
// Moneyline closing prices of the graded games, in parallel chunks of 60 game ids: the whole view scan takes 1.3-2.7s against the
// 3s anon statement limit (a timeout would drop every price), a chunk answers in ~0.5s. A failed chunk leaves its games unpriced.
async function trackClosing(gamePks) {
  const ids = [...new Set((gamePks || []).filter((x) => x != null))], chunks = [];
  for (let i = 0; i < ids.length; i += TRACK_CLOSING_CHUNK) chunks.push(ids.slice(i, i + TRACK_CLOSING_CHUNK));
  const parts = await Promise.all(chunks.map((c) => sb(`game_closing_prices?market=eq.moneyline&game_pk=in.(${c.join(",")})&select=game_pk,side,close_dec`).catch(() => [])));
  return parts.flat();
}
// The record scope (track_record_start). trackRecordStarts() swallows errors into "no restarts", which would pull the archived
// pre-restart rows into the published record, so this page loads it itself and fails closed.
async function trackStarts() {
  try {
    const rows = await sb("track_record_start?select=sport,starts_at,model_version");
    return { starts: new Map((rows || []).map((r) => [r.sport, r])), failed: false };
  } catch (e) { console.error("track record: track_record_start failed to load", e); return { starts: new Map(), failed: true }; }
}
const TRACK_ERRORS = { scope: "Couldn't load the record scope — try again.", picks: "Couldn't load graded picks." };
async function trackLoad() {
  const today = etDateStr(new Date().toISOString());
  let picksFailed = false;
  const [acc, scope, predPnl] = await Promise.all([
    sb(`prediction_accuracy?actual_winner=not.is.null&order=game_date.desc&limit=${TRACK_ROWS_CAP}&select=${TRACK_ACC_SELECT}`).catch((e) => { picksFailed = true; console.error("track record: prediction_accuracy failed to load", e); return []; }),
    trackStarts(),
    sb("prediction_pnl_daily?select=*").catch(() => []),
  ]);
  const error = scope.failed ? "scope" : picksFailed ? "picks" : null, starts = scope.starts;
  if (error) return { today, rows: [], starts, predPnl: [], error, capped: false };
  const capped = (acc || []).length >= TRACK_ROWS_CAP;
  if (capped) console.warn(`track record: hit the ${TRACK_ROWS_CAP}-row cap on prediction_accuracy; older graded games may be missing`);
  const closingRows = await trackClosing((acc || []).map((r) => r.game_pk));
  const rows = trackRows(acc, [], trackClosingMap(closingRows)).map((r) => ({ ...r, inRecord: inTrackRecord(starts, r.sport, r.date) }));
  return { today, rows, starts, predPnl: predPnl || [], error: null, capped };
}
// One render context: the cached load plus the URL state and the rows inside range / record.
const trackCtx = (D) => { const s = trackState(); return { ...D, s, scope: trackScope(D.rows, s, D.today) }; };

/* ── small view helpers ───────────────────────────────────────────────── */
const trackPct = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
const trackRec = (t) => `${t.w} - ${t.l} - ${t.p}`;
const trackLine = (x) => (x == null ? "—" : Math.abs(x) < 0.05 ? "PK" : x > 0 ? `+${x.toFixed(1)}` : x.toFixed(1));
const trackSportName = (s) => SPORT_NAME[s] || String(s || "").toUpperCase();
const trackLegend = (items) => `<div class="ca-trk-legend">${items.map(([label, color]) => `<span><i style="background:${ctxEsc(color)}"></i>${label}</span>`).join("")}</div>`;
function trackRangeLabel(rows) {
  const ds = (rows || []).map((r) => r.date).filter(Boolean).sort();
  return ds.length ? `${fullDate(ds[0])} – ${fullDate(ds[ds.length - 1])}` : "No graded picks";
}
// What the model picked, at the model's own number: "Chiefs ML", "Texas -6.0", "Over 55.5"; "—" when it had no side.
function trackPickText(r) {
  if (!r.pick) return "—";
  if (r.market === "moneyline") return `${shortTeam(r.pick, r.sport)} ML`;
  if (r.market === "total") return `${r.pick} ${(r.modelLine ?? r.closing) != null ? (+(r.modelLine ?? r.closing)).toFixed(1) : ""}`.trim();
  return `${shortTeam(r.pick, r.sport)} ${trackLine(r.modelLine ?? r.closing)}`;
}
// Final score as the table shows it: the winner's short name ("Ravens by 6 · 42 total"); the full text is the cell's tooltip.
const trackFinalText = (r) => (r.winner && r.finalScore.startsWith(r.winner) ? `${shortTeam(r.winner, r.sport)}${r.finalScore.slice(r.winner.length)}` : r.finalScore);
const trackClosingText = (r) => (r.closing == null ? "—" : r.market === "moneyline" ? oddsStr(r.closing) : r.market === "total" ? (+r.closing).toFixed(1) : trackLine(r.closing));

/* ── stat cards ───────────────────────────────────────────────────────── */
// The record mixes two kinds of graded pick, so the card says so (tooltip on the label + a caption).
const TRACK_RECORD_TIP = "Combines moneyline winner picks with spread and total picks graded against the line";
// The restart note (formerly a header line) and what the record combines: the (i) tooltip on OVERALL RECORD.
function trackRecordTip(C) {
  const st = C.starts && C.starts.get ? C.starts.get("nfl") : null;
  const when = st ? new Date(st.starts_at).toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" }) : "";
  const note = st ? `NFL record restarted with ${st.model_version === "nfl-sim-ml-v2" ? "the ML v2 model" : st.model_version} on ${when}; earlier NFL results are archived for comparison.`
    : "NFL predictions switched to the ML model on Sep 28, 2026; earlier games were graded against the previous model.";
  return `${TRACK_RECORD_TIP}. ${note}`;
}
function trackStatRecord(C) {
  const t = trackTally(C.scope), tip = trackRecordTip(C);
  if (!t.n) return statCard({ label: "OVERALL RECORD", tip, value: "—", sub: "No graded picks in this range." });
  const weeks = trackWeekly(C.scope).slice(-10).map((w) => (w.pct == null ? 0 : w.pct * 100)), bars = miniBars(weeks, { w: 70, h: 44 });
  const rec = `<span class="ca-trk-rec">${[["W", t.w], ["L", t.l], ["P", t.p]].map(([k, v]) => `<span><b>${v}</b><em>${k}</em></span>`).join("<i>-</i>")}</span>`;
  return statCard({ label: "OVERALL RECORD", tip, value: `${rec}<small class="${t.pct != null && t.pct >= 0.5 ? "pos" : ""}">${trackPct(t.pct)}</small>`,
    visual: bars ? `<span title="Win % by week">${bars}</span>` : "" });
}
function trackStatLine(C) {
  const v = trackAccuracyVsLine(C.scope);
  if (!v) return statCard({ label: "ACCURACY VS CLOSING LINE", value: "—", sub: "No graded spread or total picks in this range." });
  const spark = sparkline(trackShareSeries(C.scope), { w: 84, h: 40 });
  return statCard({ label: "ACCURACY VS CLOSING LINE", value: trackPct(v.share), valueClass: signCls(v.vsMarket),
    sub: `<span class="${signCls(v.vsMarket)}">${pStr(v.vsMarket)}</span> vs market`,
    tip: `${v.n} spread and total picks · share on the right side of the closing line, vs the 52.4% breakeven`,
    visual: spark ? `<span title="Running share of spread and total picks on the right side of the close">${spark}</span>` : "" });
}
function trackStatTotal(C) {
  const weeks = trackWeekly(C.scope).slice(-10).map((w) => w.n), bars = miniBars(weeks, { w: 70, h: 44, color: "var(--muted)" });
  return statCard({ label: "TOTAL PREDICTIONS", value: String(C.scope.length), sub: "Across all leagues and markets", visual: bars ? `<span title="Predictions per week">${bars}</span>` : "" });
}
function trackStatStreak(C) {
  const s = trackStreak(C.scope);
  if (!s.streak) return statCard({ label: "CURRENT STREAK", value: "—", sub: C.scope.length ? "No decided picks yet." : "No graded picks in this range." });
  return statCard({ label: "CURRENT STREAK", value: `${s.streak.kind} ${s.streak.n}${trackDots([...s.dots].reverse())}`, valueClass: s.streak.kind === "W" ? "pos" : "neg",
    sub: `Last 5: ${s.w} - ${s.l} - ${s.p}` });
}
function trackStatAvg(C) {
  const ps = C.scope.map((r) => r.prob).filter(finite).map(Number), mean = ps.length ? ps.reduce((a, b) => a + b, 0) / ps.length : null;
  if (mean == null) return statCard({ label: "AVERAGE CONFIDENCE", value: "—", sub: "No graded moneyline picks in this range." });
  return statCard({ label: "AVERAGE CONFIDENCE", value: trackPct(mean), sub: "On moneyline picks", visual: donut(mean, { size: 72, stroke: 11 }) });
}

/* ── charts ───────────────────────────────────────────────────────────── */
function trackCumCard(C) {
  const pts = dateAxisPoints(trackCumulative(C.scope, C.s.chart), (p) => p.net), chart = areaChart(pts, { h: 168, yTicks: 5, unit: "", nice: true });
  return `<section class="ca-card ca-trk-card" id="trk-cum"><div class="ca-card-head"><div><h2>Cumulative Prediction Performance</h2><p>Running record (net wins) over time.</p></div>${pills("trk-chart", TRACK_CHARTS, C.s.chart)}</div>
    ${chart || emptyMsg(C.s.chart === "overall" ? "Not enough graded results in this range yet." : `Not enough graded ${trackSportName(C.s.chart)} results in this range yet.`)}</section>`;
}
function trackMarketCard(C) {
  const groups = TRACK_MARKETS.map((m) => ({ m, t: trackTally(C.scope.filter((r) => r.market === m)) })).filter((x) => x.t.n).map(({ m, t }) => ({ label: TRACK_MKT_LABEL[m], sub: `${trackRec(t)} (${t.n})`,
    bars: [[t.w, TRACK_WIN], [t.l, TRACK_LOSS], [t.p, TRACK_PUSH]].map(([k, color]) => ({ value: k / t.n * 100, color, label: `${(k / t.n * 100).toFixed(1)}%` })) }));
  const chart = groupedBars(groups, { h: 176, bw: 36, bgap: 8, ggap: 44 });
  return `<section class="ca-card ca-trk-card" id="trk-market"><div class="ca-card-head"><div><h2>Performance by Market</h2><p>Win rate by market type.</p></div>${trackLegend([["Win %", TRACK_WIN], ["Loss %", TRACK_LOSS], ["Push %", TRACK_PUSH]])}</div>
    ${chart ? `<div class="ca-trk-bars">${chart}</div>` : emptyMsg("No graded picks in this range yet.")}</section>`;
}
function trackLeagueCard(C) {
  const rows = SPORTS.map((s) => ({ s, t: trackTally(C.scope.filter((r) => r.sport === s)) })).filter((x) => x.t.n).map(({ s, t }) =>
    `<div class="ca-trk-lg"><span class="ca-trk-lg-n">${leagueLogo(s)}<b>${trackSportName(s)}</b></span><i class="ca-trk-bar"><b style="width:${t.pct == null ? 0 : (t.pct * 100).toFixed(1)}%"></b></i><b class="ca-trk-lg-p">${trackPct(t.pct)}</b><span class="ca-trk-lg-r">${trackRec(t)}</span><em>(${t.n})</em></div>`).join("");
  return `<section class="ca-card ca-trk-card" id="trk-league"><div class="ca-card-head"><div><h2>Performance by League</h2><p>Win rate and record by league.</p></div></div>${rows || emptyMsg("No graded picks in this range yet.")}</section>`;
}
function trackCalibCard(C) {
  const c = trackCalibration(C.scope), chart = groupedBars(c.groups, { h: 176, bw: 36, bgap: 8, ggap: 40 });
  return `<section class="ca-card ca-trk-card" id="trk-calib"><div class="ca-card-head"><div><h2>Confidence Calibration</h2><p title="Moneyline picks: the model&#39;s win probability against how often it won">Predicted confidence vs. actual hit rate.</p></div>${trackLegend([["Predicted Confidence", "var(--navy)"], ["Actual Hit Rate", "var(--green)"]])}</div>
    ${chart ? `<div class="ca-trk-bars">${chart}</div>` : emptyMsg("No graded moneyline picks in this range yet.")}</section>`;
}
function trackFormCard(C) {
  const rows = trackForms(C.scope).map((f) => `<div class="ca-trk-fm"><span>${f.label}</span><b>${f.n ? `${f.w} - ${f.l} - ${f.p}` : "—"}</b><strong class="${f.pct == null ? "" : f.pct >= 0.5 ? "pos" : "neg"}">${trackPct(f.pct)}</strong></div>`).join("");
  return `<section class="ca-card ca-trk-card" id="trk-form"><div class="ca-card-head"><div><h2>Recent Form</h2><p>Performance over different time periods.</p></div></div>${rows}</section>`;
}

/* ── filters + table ──────────────────────────────────────────────────── */
// Row 4 controls (T7): market pills + the League / Market Type / Confidence / Result / Sort By / Record selects (labels above) in one
// card; the search box is its own card beside it (id trk-search, not redrawn while typing).
const trackRecordSelect = (C) => selectField("data-trk-record", "record", "Record", trackRecordOpts(C.starts), trackRecordKey(C.s));
function trackFilterCard(C) {
  const s = C.s, leagues = [["", "All Leagues"], ...SPORTS.map((x) => [x, trackSportName(x)])];
  return `<section class="ca-card ca-trk-filters" id="trk-filters">${pills("trk-mkt", TRACK_MARKET_PILLS, s.market)}
    <div class="ca-trk-fgrid">${selectField("data-trk", "league", "League", leagues, s.league)}${selectField("data-trk", "market", "Market Type", TRACK_MARKET_OPTS, s.market)}
    ${selectField("data-trk", "conf", `<span title="The model's win probability: applies to moneyline picks only">Confidence</span>`, TRACK_CONF_OPTS, s.conf)}${selectField("data-trk", "result", "Result", TRACK_RESULT_OPTS, s.result)}
    ${selectField("data-trk", "sort", "Sort By", TRACK_SORTS, s.sort)}${trackRecordSelect(C)}</div></section>`;
}
const trackSearchCard = (C) => searchCard("trk-search", "data-trk-q", C.s.q);
function trackTr(r) {
  const team = r.side === "home" ? r.home : r.away, chipTitle = r.units != null ? ` title="${ctxEsc(signedStr(r.units, 2, "u"))}"` : "";
  const text = trackPickText(r);
  return `<tr data-href="${gameHref(r.sport, r.game_pk)}"><td class="ca-trk-date">${fullDate(r.date)}</td><td><span class="ca-team">${leagueLogo(r.sport)}<span class="ca-trk-lgt">${ctxEsc(trackSportName(r.sport))}</span></span></td>
    <td><span class="ca-team" title="${ctxEsc(r.matchup)}">${logoImg(team || r.away, r.sport)}<span class="ca-ell">${ctxEsc(shortMatchup(r.away, r.home, r.sport))}</span></span></td>
    <td class="muted">${TRACK_MKT_LABEL[r.market]}</td><td><span class="ca-ell ca-trk-pick" title="${ctxEsc(text)}">${ctxEsc(text)}</span></td><td>${ctxEsc(trackClosingText(r))}</td>
    <td>${pct1(r.prob)}</td><td><span class="ca-trk-chip ${r.result}"${chipTitle}>${r.result === "P" ? "Push" : r.result}</span></td>
    <td>${r.conf == null ? "—" : `${Math.round(r.conf * 100)}%`}</td><td><span class="ca-ell ca-trk-fs" title="${ctxEsc(r.finalScore)}">${ctxEsc(trackFinalText(r))}</span></td><td>${goLink(r.sport, r.game_pk)}</td></tr>`;
}
function trackTableCard(C) {
  const s = C.s, list = trackFilter(C.scope, s), shown = list.slice(0, s.n);
  let body;
  if (!C.scope.length) body = emptyMsg(s.archive ? "No archived pre-restart picks in this range." : "No graded picks in this range yet.");
  else if (!list.length) body = `${emptyMsg("No predictions match these filters.")}<p class="ca-trk-reset"><button class="ca-btn" data-trk-reset>Reset filters</button></p>`;
  else {
    body = `<div class="ca-table-wrap"><table class="ca-table ca-dash-table ca-trk-table"><thead><tr><th>Date</th><th>League</th><th>Matchup / Player</th><th>Market</th><th>Model Prediction</th><th>Closing Line</th><th>Model Prob.</th><th>Result</th><th>Confidence</th><th>Final Score / Outcome</th><th>View</th></tr></thead><tbody>${shown.map(trackTr).join("")}</tbody></table></div>
      <div class="ca-trk-more"><span class="muted">Showing ${shown.length} of ${list.length}${C.capped ? ` <span class="ca-trk-capnote">· Showing the most recent ${TRACK_ROWS_CAP.toLocaleString("en-US")} graded games.</span>` : ""}</span>${shown.length < list.length ? `<button class="ca-btn" data-trk-more>Show more</button>` : ""}</div>`;
  }
  return `<section class="ca-card ca-trk-tablecard" id="trk-table">${body}</section>`;
}

/* ── right column ─────────────────────────────────────────────────────── */
function trackSegmentsCard(C) {
  const segs = trackSegments(C.scope, C.s.seg).slice(0, 6);
  const rows = segs.map((x) => `<div class="ca-trk-sg"><span>${ctxEsc(x.label)}</span><b class="${x.pct >= 0.5 ? "pos" : "neg"}">${trackPct(x.pct)}</b><span class="muted">${trackRec(x)}</span></div>`).join("");
  return `<section class="ca-card ca-trk-rail-card" id="trk-segments"><div class="ca-card-head"><h2>Best Performing Segments</h2></div><div class="ca-dash-tabs">${pills("trk-seg", TRACK_SEGS, C.s.seg)}</div>${rows || emptyMsg("No decided picks in this range yet.")}${C.s.seg === "confidence" && rows ? `<p class="ca-cap">Moneyline picks only: confidence is the model's win probability.</p>` : ""}</section>`;
}
function trackModelCard(C) {
  const m = trackModelCalibration(C.scope);
  const row = (label, value, extra = "") => `<div class="ca-trk-mc${extra ? "" : " solo"}"><span>${label}</span><b>${value}</b>${extra}</div>`;
  const body = m ? `${row("Brier Score", m.brier.toFixed(3), `<em class="ca-trk-verdict ${m.chip === "Well Calibrated" ? "ok" : "off"}">${m.chip}</em>`)}${row("Predicted Avg. Confidence", trackPct(m.predicted))}${row("Actual Hit Rate", trackPct(m.actual))}${row("Calibration Difference", signedStr(m.diff, 1, "%"))}
    <p class="ca-cap">${m.n} graded moneyline picks · lower Brier is better · well calibrated = within 3 points.</p>` : emptyMsg("No graded moneyline picks in this range yet.");
  return `<section class="ca-card ca-trk-rail-card" id="trk-model"><div class="ca-card-head"><h2>Model Calibration</h2></div>${body}</section>`;
}
// The model picks' profit tracker (prediction_pnl_daily, already cut at the record start); not offered for the archive.
function trackProfitCard(C) {
  if (C.s.archive) return "";
  const s = getSettings(), U = unitLabel(s);
  return `<div class="ca-trk-legacy" id="trk-profit">${pnlSection(`Profit tracker <span style="font-size:.6em;opacity:.6">${U} per bet</span>`, `What ${U} on every model pick would have made — moneyline (predicted winner), spread &amp; total picks — at the closing price (median across books). Spread/total with no captured price assume −110.`,
    C.predPnl, [["moneyline", "MONEYLINE"], ["spread", "SPREAD"], ["total", "TOTAL"]], s)}</div>`;
}

/* ── page ─────────────────────────────────────────────────────────────── */
const TRACK_CARDS = {
  "trk-record": ["Overall Record", trackStatRecord, "ca-card ca-stat"], "trk-line": ["Accuracy vs Closing Line", trackStatLine, "ca-card ca-stat"],
  "trk-total": ["Total Predictions", trackStatTotal, "ca-card ca-stat"], "trk-streak": ["Current Streak", trackStatStreak, "ca-card ca-stat"], "trk-avg": ["Average Confidence", trackStatAvg, "ca-card ca-stat"],
  "trk-cum": ["Cumulative Prediction Performance", trackCumCard, "ca-card ca-trk-card"], "trk-market": ["Performance by Market", trackMarketCard, "ca-card ca-trk-card"],
  "trk-league": ["Performance by League", trackLeagueCard, "ca-card ca-trk-card"], "trk-calib": ["Confidence Calibration", trackCalibCard, "ca-card ca-trk-card"], "trk-form": ["Recent Form", trackFormCard, "ca-card ca-trk-card"],
  "trk-filters": ["Filters", trackFilterCard, "ca-card ca-trk-filters"], "trk-search": ["Search", trackSearchCard, "ca-card ca-sfind"], "trk-table": ["Predictions", trackTableCard, "ca-card ca-trk-tablecard"],
  "trk-segments": ["Best Performing Segments", trackSegmentsCard, "ca-card ca-trk-rail-card"], "trk-model": ["Model Calibration", trackModelCard, "ca-card ca-trk-rail-card"],
  "trk-profit": ["Profit tracker", trackProfitCard, "ca-card"],
};
const trackCard = (id, C) => safeCard(TRACK_CARDS[id][0], TRACK_CARDS[id][1], C, TRACK_CARDS[id][2], id);

// Title row: "Track Record" + subtitle inline; range pills + the range's date span on the right (the +EV view has no range).
function trackTitle(C) {
  const s = C.s, right = s.view === "ev" ? "" : `${pills("trk-range", TRACK_RANGES, s.range)}<div class="ca-trk-daterange">${ICON_CAL}<span>${ctxEsc(trackRangeLabel(C.scope))}</span></div>`;
  return pageTitle("Track Record", "Prediction performance across leagues and markets.", right);
}
// The +EV view and a failed load have no filter row, so the Record select sits alone under the title.
const trackRecordBar = (C) => `<section class="ca-card ca-trk-recordbar" id="trk-recordbar">${trackRecordSelect(C)}</section>`;
// A failed load renders no record numbers at all: one notice card (with a retry) under the title.
function trackErrorHtml(C) {
  return `${trackTitle(C)}${trackRecordBar(C)}
    <section class="ca-card ca-trk-error" id="trk-error" role="alert"><p class="ca-empty">${ctxEsc(TRACK_ERRORS[C.error])}</p><p class="ca-trk-reset"><button class="ca-btn" data-trk-retry>Try again</button></p></section>`;
}
function trackModelHtml(C) {
  if (C.error) return trackErrorHtml(C);
  const banner = C.s.archive ? `<p class="ca-trk-banner">Showing archived pre-restart NFL results (the earlier model). They are kept for comparison and are not part of the published record. <button class="ca-link" data-trk-record-back>Back to the record</button></p>` : "";
  return `${trackTitle(C)}${banner}
    <div class="ca-stats ca-trk-stats">${["trk-record", "trk-line", "trk-total", "trk-streak", "trk-avg"].map((id) => trackCard(id, C)).join("")}</div>
    <div class="ca-trk-r1">${trackCard("trk-cum", C)}${trackCard("trk-market", C)}</div>
    <div class="ca-trk-r2">${trackCard("trk-league", C)}${trackCard("trk-calib", C)}${trackCard("trk-form", C)}</div>
    <div class="ca-trk-filterrow">${trackCard("trk-filters", C)}${trackCard("trk-search", C)}</div>
    <div class="ca-trk-main"><div class="ca-trk-left">${trackCard("trk-table", C)}</div><aside class="ca-trk-rail">${trackCard("trk-segments", C)}${trackCard("trk-model", C)}</aside></div>
    ${trackCard("trk-profit", C)}`;
}

// +EV view data: the legacy track sources, cut to the published record exactly as the legacy page did.
async function trackEvLoad() {
  const [evRes, evPk, propRes, propGradeRows, evPnl, propGames, parlayRes, scope] = await Promise.all([
    evResultsRows().catch(() => []), evGradedPicks().catch(() => []), propResultsRows(), propLineGradesRows(),
    sb("ev_pnl_daily?select=*").catch(() => []), sb("nfl_prop_pnl_by_game?select=*&order=commence_time.asc").catch(() => []),
    parlayResultsRows(), trackStarts(),
  ]);
  const starts = scope.starts;
  if (scope.failed) return { today: etDateStr(new Date().toISOString()), starts, scope: [], error: "scope" };
  const pickKick = new Map((evPk || []).map((p) => [`${p.sport}|${p.game_pk}|${p.market}|${p.side}`, p.commence_time]));
  return { today: etDateStr(new Date().toISOString()), starts, evPk: evPk || [], evPnl: evPnl || [], propGradeRows: propGradeRows || [], propGames: propGames || [], scope: [], error: null,
    evRes: (evRes || []).filter((r) => inTrackRecord(starts, r.sport, pickKick.get(`${r.sport}|${r.game_pk}|${r.market}|${r.side}`) || r.graded_at)),
    propRes: (propRes || []).filter((r) => inTrackRecord(starts, r.sport || "nfl", r.commence_time)),
    parlayRes: (parlayRes || []).filter((r) => inTrackRecord(starts, r.sport || "nfl", r.first_commence)) };
}
function trackEvHtml(C) {
  if (C.error) return trackErrorHtml(C);
  const s = getSettings(), U = unitLabel(s), sec = (name, fn) => safeCard(name, fn, C, "ca-card");
  return `${trackTitle(C)}${trackRecordBar(C)}<div class="ca-trk-legacy">
    ${sec("+EV profit tracker", () => pnlSection(`+EV profit tracker <span style="font-size:.6em;opacity:.6">${U} per bet</span>`, `What ${U} on every graded +EV pick would have made, at the price it was flagged at · a parlay is one ${U} ticket.`, C.evPnl,
      [["moneyline", "+EV MONEYLINE"], ["spread", "+EV SPREAD"], ["prop", "+EV PROPS"], ["parlay", "+EV PARLAYS"]], s))}
    ${sec("Graded parlays", () => gradedParlaysSection(C.parlayRes, s))}${sec("+EV pick performance", () => evTrackSection(C.evRes, C.evPk))}
    ${sec("Sim prop accuracy", () => propAccuracySection(C.propGradeRows))}${sec("Props by game", () => propGamesSection(C.propGames, s))}${sec("+EV player-prop performance", () => propTrackSection(C.propRes))}</div>`;
}

async function buildTrackPage() {
  const s = trackState();
  window.__caPnl = [];                     // pnlSection registers its rows here (legacy wirePnl reads them)
  const D = s.view === "ev" ? await trackEvLoad() : await trackLoad();
  window.__caTrkData = D;
  const C = trackCtx(D);
  return `<div class="ca-trk" id="trk-root">${s.view === "ev" ? trackEvHtml(C) : trackModelHtml(C)}</div>`;
}

/* ── interaction ──────────────────────────────────────────────────────── */
function trackRerender() {   // everything from the cached load (range / archive changes need no refetch)
  const root = document.getElementById("trk-root"), D = window.__caTrkData;
  if (!root || !D) return;
  window.__caPnl = [];
  root.innerHTML = trackModelHtml(trackCtx(D));
  wirePnl();
}
function trackRedraw(...ids) {
  const D = window.__caTrkData;
  if (!D) return;
  const C = trackCtx(D);
  for (const id of ids) { const el = document.getElementById(id); if (el) el.outerHTML = trackCard(id, C); }
}
function trackSet(key, value) {
  const s = trackState();
  s[key] = value;
  if (["market", "league", "conf", "result", "sort", "q"].includes(key)) s.n = TRACK_PAGE;
  trackSync(s);
}
// The Record select: model picks / +EV picks / the pre-restart NFL archive. Only the +EV view needs different data (refetch); the
// archive is the model view's rows scoped to before the restart (redraw from the cached load).
function trackSetRecord(key) {
  const s = trackState(), was = s.view;
  s.view = key === "ev" ? "ev" : "model";
  s.archive = key === "archive";
  trackSync(s);
  if (s.view !== was) render(); else trackRerender();
}
function wireTrackPage() {
  const root = document.getElementById("trk-root");
  if (!root) return;
  wirePnl();
  if (trackState().view === "ev") wirePropGames();
  root.addEventListener("click", (e) => {
    const t = e.target, pill = t.closest("[data-pill]");
    if (pill) {
      const k = pill.dataset.key;
      switch (pill.dataset.pill) {
        case "trk-range": trackSet("range", k); trackRerender(); return;
        case "trk-chart": trackSet("chart", k); trackRedraw("trk-cum"); return;
        case "trk-seg": trackSet("seg", k); trackRedraw("trk-segments"); return;
        case "trk-mkt": trackSet("market", k); trackRedraw("trk-filters", "trk-table"); return;
      }
    }
    if (t.closest("[data-trk-retry]")) { render(); return; }
    if (t.closest("[data-trk-record-back]")) { trackSetRecord("model"); return; }
    if (t.closest("[data-trk-more]")) { trackSet("n", Math.min(TRACK_SHOW_MAX, trackState().n + TRACK_PAGE)); trackRedraw("trk-table"); return; }
    if (t.closest("[data-trk-reset]")) {
      const s = trackState();
      Object.assign(s, { market: "all", league: "", conf: "", result: "", sort: "recent", q: "", n: TRACK_PAGE });
      trackSync(s); trackRedraw("trk-filters", "trk-search", "trk-table");
    }
  });
  root.addEventListener("change", (e) => {
    const rec = e.target.closest("select[data-trk-record]");
    if (rec) { trackSetRecord(rec.value); return; }
    const sel = e.target.closest("select[data-trk]");
    if (!sel) return;
    trackSet(sel.dataset.trk, sel.value);
    trackRedraw("trk-filters", "trk-table");
  });
  const q = root.querySelector("[data-trk-q]");
  if (q) root.addEventListener("input", (e) => { if (e.target.matches("[data-trk-q]")) { trackSet("q", e.target.value); trackRedraw("trk-table"); } });
}
