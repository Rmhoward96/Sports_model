/* Take These page (take.html): sportsbook prices that beat Pinnacle's no-vig fair price, from take_list_v (db/migration_take_these.sql),
   plus the progress of the one-time CLV test (take_test_v, take_verdict). Real data only: every section has an empty state.
   The live table is every active pick whose game has not started (one not seen for TAKE_STALE_MIN minutes sorts last, badged stale); picks that dropped (price moved) or started in the last 24 hours are
   listed greyed beneath it. The Model column compares the pick with the shrunk model lean (shrinkLean, js/shrink.js) for the model
   leagues' spreads and totals; it is blank everywhere else.
   Depends on app.js (sb, sbAll, ctxEsc, pct1, pStr, evBookName), js/shrink.js, js/ui.js (bookBadge), js/shell.js (pageTitle) and
   js/data.js (shortTeam, emptyMsg, safeCard). The helpers are take-prefixed because the scripts share one global scope. */
const TAKE_TEST_START = "2026-11-01T00:00:00Z";     // = TEST_START in src/sportsmodel/take/constants.py
const TAKE_NEEDED = 300;                           // = N_REQUIRED
const TAKE_BEAT_MIN = 0.55;                        // = BEAT_MIN
const TAKE_EV_MIN = 0.015;                         // = EV_MIN
const TAKE_STALE_MIN = 150;                        // a live pick last verified longer ago than this (about two pull periods) is stale
const TAKE_MODEL_LEAGUES = ["nfl", "cfb", "mlb"];  // = MODEL_LEAGUES

/* ── pure helpers ─────────────────────────────────────────────────────── */
// "+105" / "−110" (a real minus sign); "—" for a missing or non-numeric price.
function takeAmerican(n) {
  if (n == null || n === "" || !Number.isFinite(+n)) return "—";
  const v = Math.round(+n);
  return v > 0 ? `+${v}` : v < 0 ? `−${Math.abs(v)}` : "0";
}
// A spread line as the side's own number: "−3.5", "+3.5", "PK".
const takeSpreadStr = (x) => (!finite(x) ? "" : +x === 0 ? "PK" : +x > 0 ? `+${+x}` : `−${Math.abs(+x)}`);
// What to bet: spread "Chiefs −3.5" (a + line is the underdog, a − line the favourite), total "Over 8.5", ml "Arsenal ML", soccer draw "Draw".
function takeBetLabel(p) {
  if (p.side === "draw") return "Draw";
  if (p.market === "total") return `${p.side === "over" ? "Over" : "Under"} ${finite(p.line) ? +p.line : ""}`.trim();
  const team = shortTeam(p.side === "home" ? p.home_team : p.away_team, p.league);
  return p.market === "spread" ? `${team} ${takeSpreadStr(p.line)}`.trim() : `${team} ML`;
}
// "≥ 2.1%" when a book's line is better than Pinnacle's (the edge is a floor: the fair price at that line is worse than the shown one), else "2.1%".
const takeEdge = (p) => `${p.better_line ? "≥ " : ""}${(+p.ev * 100).toFixed(1)}%`;
// Time until a start: "2h 15m", "45m", or "1d 3h" past 24 hours; "Started" once it has begun.
function takeCountdown(commenceIso, now) {
  const ms = Date.parse(commenceIso) - now.getTime();
  if (!Number.isFinite(ms)) return "";
  if (ms <= 0) return "Started";
  const mins = Math.floor(ms / 60000), d = Math.floor(mins / 1440), h = Math.floor((mins % 1440) / 60), m = mins % 60;
  return d >= 1 ? `${d}d ${h}h` : h >= 1 ? `${h}h ${m}m` : `${m}m`;
}
// "3m ago" / "2h ago" / "1d ago" for the Found / Checked column.
function takeAgo(iso, now) {
  const ms = now.getTime() - Date.parse(iso);
  if (!Number.isFinite(ms)) return "—";
  const mins = Math.floor(Math.max(0, ms) / 60000);
  return mins < 1 ? "just now" : mins < 60 ? `${mins}m ago` : mins < 1440 ? `${Math.floor(mins / 60)}h ago` : `${Math.floor(mins / 1440)}d ago`;
}
// A pick last seen more than TAKE_STALE_MIN minutes before now, or with no parsable last_seen, can no longer be trusted at its price.
const takeIsStale = (r, now) => !((now.getTime() - Date.parse(r.last_seen)) <= TAKE_STALE_MIN * 60000);
// live = active and not started: fresh picks first, then stale ones (takeIsStale), each best edge first; dropped = everything else
// (inactive, or started), most recent first. Never mutates the input.
function takeSplit(rows, now) {
  const t = now.getTime(), start = (r) => Date.parse(r.commence_time);
  const fresh = [], stale = [], dropped = [];
  for (const r of rows || []) (r.active && start(r) > t ? (takeIsStale(r, now) ? stale : fresh) : dropped).push(r);
  const when = (r) => Date.parse(r.dropped_at || r.commence_time) || 0;
  fresh.sort((a, b) => b.ev - a.ev);
  stale.sort((a, b) => b.ev - a.ev);
  dropped.sort((a, b) => when(b) - when(a));
  return { live: [...fresh, ...stale], dropped };
}
// Does the shrunk model lean the same way as the pick? Only for a model league's spread or total with a prediction, and only when the
// shrunk model leans at all (shrinkLean null -> blank). Spread: shrinkLean wants the HOME book line; ref_line is Pinnacle's line for the
// pick's own side, so the home line is ref_line for a home pick and -ref_line for an away pick.
// predictions_any is one row per sport + game_pk with the model's expected scores (no pred_margin / pred_total columns): margin = home - away
// (the site's pred_margin convention), total = home + away. A missing score -> null (no model note).
function takePredFromRow(r) {
  if (!r || !finite(r.pred_home_score) || !finite(r.pred_away_score)) return null;
  return { pred_margin: +r.pred_home_score - +r.pred_away_score, pred_total: +r.pred_home_score + +r.pred_away_score };
}
function takeModelNote(p, pred) {
  if (!pred || !TAKE_MODEL_LEAGUES.includes(p.league) || typeof shrinkLean !== "function") return "";
  let lean = null;
  if (p.market === "spread") {
    if (!finite(p.ref_line) || !finite(pred.pred_margin)) return "";
    lean = shrinkLean(p.league, "spread", +pred.pred_margin, p.side === "home" ? +p.ref_line : -p.ref_line);
  } else if (p.market === "total") {
    if (!finite(p.ref_line) || !finite(pred.pred_total)) return "";
    lean = shrinkLean(p.league, "total", +pred.pred_total, +p.ref_line);
  } else return "";
  return lean ? (lean.side === p.side ? "model agrees" : "model disagrees") : "";
}
// The test's progress. Counts only picks first seen on or after Nov 1 that were graded by closing line (clv) or by direction, the same rows
// the verdict judges (src/sportsmodel/take/verdict.py). beatRate = share with beat === true; meanClv over the rows that have a CLV.
function takeProgress(testRows, verdict) {
  const rows = (testRows || []).filter((r) => r && Date.parse(r.first_seen) >= Date.parse(TAKE_TEST_START) && (r.graded_by === "clv" || r.graded_by === "direction"));
  const clvs = rows.filter((r) => finite(r.clv)).map((r) => +r.clv);
  return { graded: rows.length, shown: Math.min(rows.length, TAKE_NEEDED), needed: TAKE_NEEDED,   // shown: the verdict judges the first 300 only
    beatRate: rows.length ? rows.filter((r) => r.beat === true).length / rows.length : null,
    meanClv: clvs.length ? clvs.reduce((a, b) => a + b, 0) / clvs.length : null,
    judged: !!verdict, passed: verdict ? !!verdict.passed : null };
}

/* ── rendering ────────────────────────────────────────────────────────── */
const TAKE_LEAGUE = { nfl: "NFL", cfb: "CFB", mlb: "MLB", nhl: "NHL", nba: "NBA", wnba: "WNBA", epl: "EPL", mls: "MLS" };
const takeClv = (x) => (finite(x) ? `${+x >= 0 ? "+" : "−"}${(Math.abs(+x) * 100).toFixed(2)}%` : "—");
const takePct = (x) => (finite(x) ? `${(+x * 100).toFixed(1)}%` : "—");

function takeProgressCard(testRows, verdict, now) {
  const pr = takeProgress(testRows, verdict), before = now.getTime() < Date.parse(TAKE_TEST_START);
  let body;
  if (verdict) {
    body = `<p class="ca-take-verdict ${verdict.passed ? "pos" : "neg"}"><b>${verdict.passed ? "Passed" : "Did not pass"}</b> · judged on ${ctxEsc(String(verdict.n))} picks: beat the close ${takePct(verdict.beat_rate)}, mean CLV ${takeClv(verdict.mean_clv)}${finite(verdict.lower) ? `, 95% lower bound ${takeClv(verdict.lower)}` : ""}.</p>`;
  } else if (before) {
    body = `<p>Test starts Nov 1; October picks are a shakedown and are not judged</p>`;
  } else {
    const w = (pr.shown / pr.needed) * 100;
    body = `<div class="ca-take-bar" role="progressbar" aria-valuemin="0" aria-valuemax="${pr.needed}" aria-valuenow="${pr.shown}"><i style="width:${w.toFixed(1)}%"></i></div>
      <p><b>${pr.shown}</b> of ${pr.needed} picks graded · beat the close ${takePct(pr.beatRate)} (needs ${Math.round(TAKE_BEAT_MIN * 100)}%) · mean CLV ${takeClv(pr.meanClv)}</p>`;
  }
  return `<section class="ca-card ca-take-progress" id="take-progress"><div class="ca-card-head"><h2>CLV test</h2></div>${body}</section>`;
}

// One qualifying book beneath the best one: "FD −108" (with the line for spreads / totals).
function takeOtherStr(p, o) {
  const line = p.market === "spread" ? takeSpreadStr(o.line) : p.market === "total" && finite(o.line) ? `${+o.line}` : "";
  return `${ctxEsc(evBookName(o.book))}${line ? ` ${line}` : ""} ${takeAmerican(o.price)}`;
}

function takeRow(p, pred, now, dropped) {
  const away = shortTeam(p.away_team, p.league), home = shortTeam(p.home_team, p.league);
  const others = (Array.isArray(p.others) ? p.others : []).map((o) => `<small class="ca-take-other">${takeOtherStr(p, o)}</small>`).join("");
  const started = Date.parse(p.commence_time) <= now.getTime();
  const when = new Date(p.commence_time).toLocaleString("en-US", { weekday: "short", hour: "numeric", minute: "2-digit" });
  const status = dropped ? (started ? "Started" : "Dropped") : takeCountdown(p.commence_time, now);
  const fairAm = typeof fairAmerican === "function" ? fairAmerican(+p.fair_prob) : null;
  const stale = !dropped && takeIsStale(p, now);
  const note = takeModelNote(p, pred), noteCls = note === "model agrees" ? "pos" : note === "model disagrees" ? "neg" : "";
  return `<tr class="ca-take-row"><td data-label="League">${ctxEsc(TAKE_LEAGUE[p.league] || String(p.league).toUpperCase())}</td>
    <td data-label="Matchup" class="ca-take-match">${ctxEsc(`${away} @ ${home}`)}</td>
    <td data-label="Start"><b>${ctxEsc(status)}</b><small>${ctxEsc(when)}</small>${stale ? `<small class="ca-take-stale">${ctxEsc("stale — recheck price")}</small>` : ""}</td>
    <td data-label="Bet" class="ca-b">${ctxEsc(takeBetLabel(p))}</td>
    <td data-label="Book &amp; price"><span class="ca-dash-odds">${bookBadge(p.book)}<b>${takeAmerican(p.price)}</b></span>${others}</td>
    <td data-label="Fair">${takePct(p.fair_prob)}${fairAm != null ? `<small>${takeAmerican(fairAm)}</small>` : ""}</td>
    <td data-label="Edge" class="pos ca-b">${ctxEsc(takeEdge(p))}</td>
    <td data-label="Good until" title="Worst price at this line that still clears the ${(TAKE_EV_MIN * 100).toFixed(1)}% edge">${takeAmerican(p.good_until)}</td>
    <td data-label="Found / Checked">${ctxEsc(takeAgo(p.first_seen, now))}<small>${ctxEsc(takeAgo(p.last_seen, now))}</small></td>
    <td data-label="Model" class="${noteCls}">${ctxEsc(note)}</td></tr>`;
}
const TAKE_HEAD = `<thead><tr><th>League</th><th>Matchup</th><th>Start</th><th>Bet</th><th>Book &amp; price</th><th>Fair</th><th>Edge</th><th>Good until</th><th>Found / Checked</th><th>Model</th></tr></thead>`;
const takeTable = (rows, preds, now, dropped) => `<div class="ca-table-wrap"><table class="ca-table ca-take-table">${TAKE_HEAD}<tbody>${
  rows.map((p) => takeRow(p, preds.get(`${p.league}|${p.game_pk}`), now, dropped)).join("")}</tbody></table></div>`;

async function takeLoad() {
  const list = await sbAll("take_list_v?select=*&order=commence_time.asc,event_id.asc,market.asc,side.asc");
  const [test, verdictRows] = await Promise.all([
    sbAll("take_test_v?select=id,event_id,league,market,side,better_line,first_seen,graded_by,clv,beat,graded_at,outcome,units&order=id.asc").catch(() => []),
    sb("take_verdict?select=*").catch(() => []),
  ]);
  // The model's numbers for the listed games from predictions_any (already latest-only, one row per sport + game_pk; game_predictions is not
  // readable by the anon key); a failed read just blanks the Model column.
  const preds = new Map();
  const pks = [...new Set(list.filter((p) => TAKE_MODEL_LEAGUES.includes(p.league) && p.game_pk != null).map((p) => p.game_pk))];
  if (pks.length) {
    const rows = await sbAll(`predictions_any?game_pk=in.(${pks.join(",")})&sport=in.(${TAKE_MODEL_LEAGUES.join(",")})&select=sport,game_pk,pred_home_score,pred_away_score&order=sport.asc,game_pk.asc`).catch(() => []);
    for (const r of rows) { const pred = takePredFromRow(r); if (pred) preds.set(`${r.sport}|${r.game_pk}`, pred); }
  }
  return { list, test, verdict: (verdictRows || [])[0] || null, preds };
}

async function buildTakePage() {
  const D = await takeLoad(), now = new Date(), { live, dropped } = takeSplit(D.list, now);
  const liveBody = live.length ? takeTable(live, D.preds, now, false) : emptyMsg("No bets beat Pinnacle's fair price right now.");
  const droppedCard = dropped.length
    ? `<section class="ca-card ca-take-dropped" id="take-dropped" style="opacity:.5"><div class="ca-card-head"><h2>Dropped in the last 24 hours</h2></div>${takeTable(dropped, D.preds, now, true)}</section>` : "";
  return `<div class="ca-take">${pageTitle("Take These", "Sportsbook prices that beat Pinnacle's no-vig fair price.")}
    ${safeCard("CLV test", () => takeProgressCard(D.test, D.verdict, now), D, "ca-card ca-take-progress", "take-progress")}
    <section class="ca-card ca-take-live" id="take-live"><div class="ca-card-head"><h2>Take these</h2><p>${live.length} ${live.length === 1 ? "bet" : "bets"} right now</p></div>${liveBody}</section>
    ${droppedCard}</div>`;
}
function wireTakePage() { /* static page: the 5-minute auto-render refreshes it */ }
