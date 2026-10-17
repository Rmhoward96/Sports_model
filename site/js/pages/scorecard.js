/* Pikkit Scorecard (scorecard.html): the user drops their Pikkit CSV export and sees whether their prices beat the closing line and where they
   slip. The file is read in the browser (FileReader) and is never sent anywhere; the text is kept in localStorage (scorecard.csv) so a reload
   keeps the view, and the unit size under scorecard.unit. The only network read is the public take_test_v view, for the side-by-side.
   All the maths is in js/pikkit.js (pk-prefixed). Depends on app.js (sbAll, ctxEsc), js/ui.js (statCard, lineChart), js/shell.js (pageTitle),
   js/data.js (emptyMsg, safeCard). The helpers are sc-prefixed because the scripts share one global scope. */
const SC_CSV_KEY = "scorecard.csv", SC_UNIT_KEY = "scorecard.unit";
const SC_DIM_LABEL = { league: "League", book: "Book", market: "Market", type: "Bet type", legs: "Parlay legs", stake: "Stake size", tag: "Pikkit tag" };
const SC_MARKET_LABEL = { moneyline: "Moneyline", spread: "Spread", total: "Total", prop: "Prop", other: "Other" };
let scTestRows = [];     // take_test_v rows, loaded once per page build

/* ── storage (every access wrapped: it can throw or be empty in private windows) ───────────────────── */
function scGet(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
function scSet(key, val) { try { localStorage.setItem(key, val); return true; } catch (e) { return false; } }
function scDel(key) { try { localStorage.removeItem(key); } catch (e) { /* nothing to remove */ } }
function scUnit() { const u = parseFloat(scGet(SC_UNIT_KEY)); return Number.isFinite(u) && u > 0 ? u : 1; }

/* ── pure helpers ─────────────────────────────────────────────────────── */
const scFin = (x) => x != null && x !== "" && Number.isFinite(+x);
const scPct = (x) => (scFin(x) ? `${(+x * 100).toFixed(1)}%` : "—");
const scSignedPct = (x) => (scFin(x) ? `${+x >= 0 ? "+" : "−"}${(Math.abs(+x) * 100).toFixed(2)}%` : "—");
const scUnits = (x) => (scFin(x) ? `${+x >= 0 ? "+" : "−"}${Math.abs(+x).toFixed(1)}u` : "—");
const scDay = (d) => d.toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric" });
const scDayYear = (d) => d.toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" });
const scWeekLabel = (key) => new Date(`${key}T12:00:00Z`).toLocaleDateString("en-US", { timeZone: "UTC", month: "short", day: "numeric" });
const scRateCls = (r) => (!scFin(r) ? "" : +r < 0.4 ? "neg" : +r >= 0.5 ? "pos" : "");

/* The list's beat rate (graded clv / direction picks in take_test_v first seen between the user's first and last placed bets) next to the
   user's own beat rate over the same span. Each side is {n, rate}: n = bets (picks) counted, rate null when there are none. */
function scSideBySide(bets, testRows) {
  const list = bets || [];
  if (!list.length) return { mine: { n: 0, rate: null }, list: { n: 0, rate: null }, from: null, to: null };
  const times = list.map((b) => b.placed.getTime());
  const from = new Date(Math.min(...times)), to = new Date(Math.max(...times));
  const mine = pkSummary(list);
  const picks = (testRows || []).filter((r) => {
    if (!r || (r.graded_by !== "clv" && r.graded_by !== "direction")) return false;
    const t = Date.parse(r.first_seen);
    return Number.isFinite(t) && t >= from.getTime() && t <= to.getTime();
  });
  return { mine: { n: mine.withClose, rate: mine.beatRate }, from, to,
    list: { n: picks.length, rate: picks.length ? picks.filter((r) => r.beat === true).length / picks.length : null } };
}

/* ── rendering ────────────────────────────────────────────────────────── */
function scAlertCard(alert) {
  if (!alert) return "";
  const rows = alert.slices.map((s) => `<tr><td data-label="Slice">${ctxEsc(SC_DIM_LABEL[s.dim] || s.dim)}: <b>${ctxEsc(s.dim === "market" ? SC_MARKET_LABEL[s.value] || s.value : s.value)}</b></td>
    <td class="neg ca-b">${scPct(s.rate)}</td><td>${s.n}</td><td>${s.priorRate == null ? "no bets" : scPct(s.priorRate)}</td></tr>`).join("");
  const table = rows ? `<div class="ca-table-wrap"><table class="ca-table ca-sc-table"><thead><tr><th>Slice</th><th>Beat close</th><th>Bets</th><th>Prior 100</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : `<p class="muted">No single slice has 15 or more bets under 40% yet.</p>`;
  return `<section class="ca-card ca-sc-alert" id="sc-alert" role="alert"><div class="ca-card-head"><h2>Slipping</h2></div>
    <p>Your beat-close rate over the last 100 bets with a close is <b class="neg">${scPct(alert.rate)}</b>, below 40%. Slices with at least 15 bets in that window and a rate under 40%, worst first:</p>${table}</section>`;
}

function scStatsHtml(sum, rolling) {
  const last = rolling.length ? rolling[rolling.length - 1].rate : null;
  const tiles = [
    statCard({ label: "Bets", value: String(sum.bets), sub: `${sum.withClose} with a close` }),
    statCard({ label: "Beat the close", value: scPct(sum.beatRate), valueClass: scRateCls(sum.beatRate), sub: "odds above closing line" }),
    statCard({ label: "Rolling 100", value: scPct(last), valueClass: scRateCls(last), sub: "latest bets with a close" }),
    statCard({ label: "Avg price vs close", value: scSignedPct(sum.avgVsClose), valueClass: scFin(sum.avgVsClose) ? (sum.avgVsClose >= 0 ? "pos" : "neg") : "", sub: "odds / closing line − 1" }),
    statCard({ label: "Staked", value: `${sum.units.toFixed(1)}u` }),
    statCard({ label: "Profit", value: scUnits(sum.profitUnits), valueClass: sum.profitUnits >= 0 ? "pos" : "neg" }),
  ];
  return `<div class="ca-stats ca-sc-stats">${tiles.join("")}</div>`;
}

const SC_HEAD = (first) => `<thead><tr><th>${ctxEsc(first)}</th><th>Bets</th><th>With close</th><th>Beat close</th><th>Avg vs close</th><th>Staked</th><th>Profit</th></tr></thead>`;
function scRow(label, s) {
  return `<tr><td data-label="" class="ca-b">${label}</td><td data-label="Bets">${s.bets}</td><td data-label="With close">${s.withClose}</td>
    <td data-label="Beat close" class="${scRateCls(s.beatRate)} ca-b">${scPct(s.beatRate)}</td><td data-label="Avg vs close">${scSignedPct(s.avgVsClose)}</td>
    <td data-label="Staked">${s.units.toFixed(1)}u</td><td data-label="Profit" class="${s.profitUnits >= 0 ? "pos" : "neg"}">${scUnits(s.profitUnits)}</td></tr>`;
}
const scTable = (first, rows) => `<div class="ca-table-wrap"><table class="ca-table ca-sc-table">${SC_HEAD(first)}<tbody>${rows.join("")}</tbody></table></div>`;

function scWeeklyCard(bets) {
  const rows = pkWeekly(bets).map((w) => scRow(ctxEsc(`Week of ${scWeekLabel(w.week)}`), w));
  return `<section class="ca-card" id="sc-weekly"><div class="ca-card-head"><h2>Weekly</h2><p>Monday to Sunday, US Eastern</p></div>${scTable("Week", rows)}</section>`;
}

function scRollingCard(bets) {
  const pts = pkRolling(bets);
  const chart = pts.length >= 2 ? lineChart([
    { label: "Beat-close rate, rolling 100 bets", values: pts.map((p) => p.rate * 100), color: "var(--navy)" },
    { label: "40% alert line", values: pts.map(() => 40), color: "var(--red)", dash: "6 5" }],
  pts.map((p) => scDay(p.placed)), { h: 240, yTicks: 5, unit: "%", nice: true }) : "";
  return `<section class="ca-card" id="sc-rolling"><div class="ca-card-head"><h2>Rolling 100</h2><p>Beat-close rate over your last 100 bets with a close, bet by bet</p></div>${
    chart || emptyMsg("The line appears once you have at least two bets with a closing line.")}</section>`;
}

function scBreakdownCards(bets) {
  const bd = pkBreakdowns(bets);
  return Object.keys(SC_DIM_LABEL).map((dim) => {
    const rows = (bd[dim] || []).map((r) => scRow(ctxEsc(dim === "market" ? SC_MARKET_LABEL[r.value] || r.value : r.value), r));
    return `<section class="ca-card" id="sc-by-${dim}"><div class="ca-card-head"><h2>${ctxEsc(SC_DIM_LABEL[dim])}</h2></div>${scTable(SC_DIM_LABEL[dim], rows)}</section>`;
  }).join("");
}

function scSideCard(bets) {
  const sb2 = scSideBySide(bets, scTestRows);
  let body;
  if (!sb2.from) body = emptyMsg("No bets to compare yet.");
  else if (!sb2.list.n) body = emptyMsg(`No graded Take These picks between ${scDayYear(sb2.from)} and ${scDayYear(sb2.to)} yet.`);
  else body = `<div class="ca-sc-side"><div><span class="muted">You</span><b class="${scRateCls(sb2.mine.rate)}">${scPct(sb2.mine.rate)}</b><small>${sb2.mine.n} bets with a close</small></div>
    <div><span class="muted">Take These list</span><b class="${scRateCls(sb2.list.rate)}">${scPct(sb2.list.rate)}</b><small>${sb2.list.n} graded picks</small></div></div>
    <p class="muted">Beat-close rate, ${ctxEsc(scDayYear(sb2.from))} to ${ctxEsc(scDayYear(sb2.to))} (your first and last bet).</p>`;
  return `<section class="ca-card" id="sc-side"><div class="ca-card-head"><h2>You vs the list</h2></div>${body}
    <p class="muted"><small>Pikkit's close is not necessarily Pinnacle's.</small></p></section>`;
}

/* Everything under the drop zone for one CSV text. Never throws: a bad file becomes a message. */
function scResultsHtml(csvText, unit) {
  if (!csvText) return emptyMsg("Drop your Pikkit export above to see your scorecard.");
  let bets;
  try { bets = pkToBets(pkParseCsv(csvText), unit); }
  catch (e) { console.error("scorecard: could not read the file", e); return emptyMsg("That file couldn't be read as a Pikkit export."); }
  if (!bets.length) return emptyMsg("No usable bets found in that file. Export your transactions from Pikkit as CSV.");
  const sum = pkSummary(bets), rolling = pkRolling(bets);
  return [
    safeCard("alert", () => scAlertCard(pkAlert(bets)), bets, "ca-card ca-sc-alert", "sc-alert"),
    safeCard("headline", () => scStatsHtml(sum, rolling), bets, "ca-card"),
    safeCard("weekly", () => scWeeklyCard(bets), bets, "ca-card", "sc-weekly"),
    safeCard("rolling", () => scRollingCard(bets), bets, "ca-card", "sc-rolling"),
    safeCard("breakdowns", () => scBreakdownCards(bets), bets, "ca-card"),
    safeCard("side-by-side", () => scSideCard(bets), bets, "ca-card", "sc-side"),
  ].join("");
}

async function scLoadTest() {
  return sbAll("take_test_v?select=id,event_id,league,market,side,better_line,first_seen,graded_by,clv,beat,graded_at,outcome,units&order=id.asc").catch(() => []);
}

async function buildScorecardPage() {
  scTestRows = await scLoadTest();
  const csv = scGet(SC_CSV_KEY), unit = scUnit();
  return `<div class="ca-sc">${pageTitle("Scorecard", "Do your prices beat the close? Drop your Pikkit CSV export; it is read in this browser and never uploaded.")}
    <section class="ca-card ca-sc-input" id="sc-input">
      <label class="ca-sc-drop" id="sc-drop" for="sc-file"><span><b>Drop your Pikkit CSV here</b>, or click to choose a file</span>
        <input type="file" id="sc-file" class="ca-sc-file" accept=".csv,text/csv"></label>
      <div class="ca-sc-controls"><label>Unit size ($) <input type="number" id="sc-unit" class="ca-input" min="0" step="any" inputmode="decimal" value="${ctxEsc(unit)}"></label>
        <button type="button" class="ca-btn" id="sc-clear"${csv ? "" : " hidden"}>Clear saved file</button></div>
    </section>
    <div id="sc-results">${scResultsHtml(csv, unit)}</div></div>`;
}

function scRefresh() {
  const out = document.getElementById("sc-results");
  if (out) out.innerHTML = scResultsHtml(scGet(SC_CSV_KEY), scUnit());
  const clear = document.getElementById("sc-clear");
  if (clear) clear.hidden = !scGet(SC_CSV_KEY);
}
function scReadFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => { scSet(SC_CSV_KEY, String(reader.result || "")); scRefresh(); };
  reader.onerror = () => { const out = document.getElementById("sc-results"); if (out) out.innerHTML = emptyMsg("That file couldn't be read."); };
  reader.readAsText(file);
}

function wireScorecardPage() {
  const drop = document.getElementById("sc-drop"), file = document.getElementById("sc-file");
  if (file) file.addEventListener("change", () => { scReadFile(file.files && file.files[0]); file.value = ""; });
  if (drop) {
    const over = (on) => (e) => { e.preventDefault(); drop.classList.toggle("over", on); };
    drop.addEventListener("dragover", over(true));
    drop.addEventListener("dragleave", over(false));
    drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); scReadFile(e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]); });
  }
  const unit = document.getElementById("sc-unit");
  if (unit) unit.addEventListener("change", () => {
    const u = parseFloat(unit.value);
    scSet(SC_UNIT_KEY, Number.isFinite(u) && u > 0 ? String(u) : "1");
    unit.value = String(scUnit());
    scRefresh();
  });
  const clear = document.getElementById("sc-clear");
  if (clear) clear.addEventListener("click", () => { scDel(SC_CSV_KEY); scRefresh(); });
}
