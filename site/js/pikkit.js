/* Pikkit scorecard maths (scorecard.html): pure functions over the user's Pikkit CSV export, no DOM and no network. The CSV never leaves the
   browser. Every top-level name carries a pk prefix because the site scripts share one global scope (a bare `alert` would shadow window.alert).
   odds and closing_line are DECIMAL. A bet "beats the close" when odds > closing_line; price vs close is odds / closing_line - 1.
   Bets without a close are left out of those two numbers but still count toward volume and profit; VOID bets are dropped entirely. */
const PK_ALERT_RATE = 0.40, PK_ALERT_WINDOW = 100, PK_ALERT_MIN_SLICE = 15;

/* RFC 4180 with a header row: strips a leading BOM; quoted fields may hold commas, "" escapes and newlines. Blank lines are skipped. */
function pkParseCsv(text) {
  const s = String(text ?? "").replace(/^﻿/, "");
  const table = [];
  let row = [], field = "", inQ = false, i = 0;
  const endField = () => { row.push(field); field = ""; };
  const endRow = () => { endField(); if (row.length > 1 || row[0] !== "") table.push(row); row = []; };
  while (i < s.length) {
    const c = s[i];
    if (inQ) {
      if (c === '"') { if (s[i + 1] === '"') { field += '"'; i += 2; continue; } inQ = false; i++; continue; }
      field += c; i++; continue;
    }
    if (c === '"') { inQ = true; i++; }
    else if (c === ",") { endField(); i++; }
    else if (c === "\r") { endRow(); i += s[i + 1] === "\n" ? 2 : 1; }
    else if (c === "\n") { endRow(); i++; }
    else { field += c; i++; }
  }
  if (field !== "" || row.length) endRow();
  if (!table.length) return [];
  const head = table[0].map((h) => h.trim());
  return table.slice(1).map((r) => Object.fromEntries(head.map((h, k) => [h, r[k] ?? ""])));
}

/* Book names as Pikkit prints them -> the names the rest of the site uses; anything else just loses a trailing " Sportsbook". */
const PK_BOOKS = { "draftkings sportsbook": "DraftKings", "fanduel sportsbook": "FanDuel", "caesars sportsbook": "Caesars", "hard rock sportsbook": "Hard Rock",
  fanatics: "Fanatics", betmgm: "BetMGM", bet365: "Bet365", novig: "Novig" };
function pkNormBook(name) {
  const n = String(name ?? "").trim();
  return PK_BOOKS[n.toLowerCase()] || n.replace(/\s+Sportsbook$/i, "");
}

/* One leg of bet_info -> moneyline | spread | total | prop | other. Period markers (1st half, innings, quarters) win, so a first-half line is "other". */
function pkLegMarket(leg) {
  const s = String(leg ?? "");
  if (/\b(1st|2nd|3rd|\d+th)\b|Inning|Half|Quarter|Period/i.test(s)) return "other";
  if (/Moneyline/i.test(s)) return "moneyline";
  if (/\bSpread\b|Run Line|Puck Line/i.test(s)) return "spread";
  if (/^(Over|Under) [\d.]+ Total\b/i.test(s)) return "total";
  if (/^(Over|Under) [\d.]+ |\d+\+|Yds|Yards|Touchdown|Scorer|Hits|Strikeouts|Doubles|Rebounds|Assists|Points|Shots|Saves/i.test(s)) return "prop";
  return "other";
}

const pkNum = (v) => { const t = String(v ?? "").replace(/[$,\s]/g, ""); return t !== "" && Number.isFinite(+t) ? +t : null; };

/* CSV rows -> bets. units / profitUnits are dollars divided by unitSize (1 when the size is missing or not positive). Rows with VOID in the
   status, no usable odds, or no placement time are dropped. close is null when blank; placed falls back to time_placed. */
function pkToBets(rows, unitSize) {
  const size = Number.isFinite(+unitSize) && +unitSize > 0 ? +unitSize : 1;
  const out = [];
  for (const r of rows || []) {
    const status = String(r.status ?? "").trim();
    if (/VOID/i.test(status)) continue;
    const odds = pkNum(r.odds);
    if (odds == null) continue;
    let placed = new Date(r.time_placed_iso);
    if (!Number.isFinite(placed.getTime())) placed = new Date(r.time_placed);
    if (!Number.isFinite(placed.getTime())) continue;
    const closeRaw = pkNum(r.closing_line), parts = String(r.bet_info ?? "").split(" | ");
    const markets = parts.map(pkLegMarket), leagues = String(r.leagues ?? "").trim();
    out.push({
      id: String(r.bet_id ?? ""), book: pkNormBook(r.sportsbook), type: String(r.type ?? "").trim().toLowerCase(), status, odds,
      close: closeRaw != null && closeRaw > 0 ? closeRaw : null, placed, legs: parts.length,
      market: markets.every((m) => m === markets[0]) ? markets[0] : "other",
      league: leagues.includes(" | ") ? "Multiple" : leagues || "Unknown",
      tag: String(r.tags ?? "").trim() || "untagged",
      units: (pkNum(r.amount) ?? 0) / size, profitUnits: (pkNum(r.profit) ?? 0) / size,
    });
  }
  return out;
}

const pkBeat = (b) => (b && b.close != null ? b.odds > b.close : null);
const pkPriceVsClose = (b) => (b && b.close != null ? b.odds / b.close - 1 : null);

function pkSummary(bets) {
  const list = bets || [], withClose = list.filter((b) => b.close != null);
  const sum = (arr, f) => arr.reduce((a, b) => a + f(b), 0);
  return { bets: list.length, withClose: withClose.length,
    beatRate: withClose.length ? withClose.filter((b) => pkBeat(b)).length / withClose.length : null,
    avgVsClose: withClose.length ? sum(withClose, pkPriceVsClose) / withClose.length : null,
    units: sum(list, (b) => b.units), profitUnits: sum(list, (b) => b.profitUnits) };
}

/* The Monday (YYYY-MM-DD) of the Monday-to-Sunday week a moment falls in, by the US Eastern calendar day. */
const PK_ET = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" });
function pkWeekKey(date) {
  const p = Object.fromEntries(PK_ET.formatToParts(date).map((x) => [x.type, x.value]));
  const day = new Date(Date.UTC(+p.year, +p.month - 1, +p.day));
  day.setUTCDate(day.getUTCDate() - ((day.getUTCDay() + 6) % 7));
  return day.toISOString().slice(0, 10);
}
function pkWeekly(bets) {
  const by = new Map();
  for (const b of bets || []) { const k = pkWeekKey(b.placed); if (!by.has(k)) by.set(k, []); by.get(k).push(b); }
  return [...by.keys()].sort().reverse().map((week) => ({ week, ...pkSummary(by.get(week)) }));
}

const pkByPlaced = (bets) => (bets || []).filter((b) => b.close != null).sort((a, b) => a.placed - b.placed);
/* One point per bet that has a close, in placed order: the beat-close rate over that bet and the n-1 before it (fewer at the start). */
function pkRolling(bets, n = PK_ALERT_WINDOW) {
  const list = pkByPlaced(bets);
  let beats = 0;
  return list.map((b, i) => {
    if (pkBeat(b)) beats++;
    if (i >= n && pkBeat(list[i - n])) beats--;
    return { placed: b.placed, rate: beats / Math.min(i + 1, n) };
  });
}

const pkLegBucket = (n) => (n >= 5 ? "5+" : String(Math.max(1, Math.floor(n))));
const pkUnitBucket = (u) => (u <= 0.5 + 1e-9 ? "≤0.5u" : u <= 1 + 1e-9 ? "≤1u" : u <= 2 + 1e-9 ? "≤2u" : ">2u");

const PK_DIMENSIONS = { league: (b) => b.league, book: (b) => b.book, market: (b) => b.market, type: (b) => b.type,
  legs: (b) => pkLegBucket(b.legs), stake: (b) => pkUnitBucket(b.units), tag: (b) => b.tag };
const PK_ORDER = { legs: ["1", "2", "3", "4", "5+"], stake: ["≤0.5u", "≤1u", "≤2u", ">2u"] };

/* {dimension: [{value, ...summary}]}: most bets first, or the natural order for legs and stake. */
function pkBreakdowns(bets) {
  const out = {};
  for (const [dim, f] of Object.entries(PK_DIMENSIONS)) {
    const by = new Map();
    for (const b of bets || []) { const v = f(b); if (!by.has(v)) by.set(v, []); by.get(v).push(b); }
    const rows = [...by].map(([value, arr]) => ({ value, ...pkSummary(arr) }));
    const order = PK_ORDER[dim];
    rows.sort(order ? (a, b) => order.indexOf(a.value) - order.indexOf(b.value) : (a, b) => b.bets - a.bets || String(a.value).localeCompare(String(b.value)));
    out[dim] = rows;
  }
  return out;
}

/* Fires when the beat-close rate over the last 100 bets with a close is under 40%. Slices (dimension values with >= 15 of those bets and their
   own rate under 40%) come worst first, each with its rate over the 100 bets before (null when it had none). */
function pkAlert(bets) {
  const list = pkByPlaced(bets), current = list.slice(-PK_ALERT_WINDOW), prior = list.slice(-2 * PK_ALERT_WINDOW, -PK_ALERT_WINDOW);
  if (!current.length) return null;
  const rate = (arr) => (arr.length ? arr.filter(pkBeat).length / arr.length : null), overall = rate(current);
  if (!(overall < PK_ALERT_RATE)) return null;
  const slices = [];
  for (const [dim, f] of Object.entries(PK_DIMENSIONS)) {
    const cur = new Map(), pri = new Map();
    for (const b of current) { const v = f(b); (cur.get(v) || cur.set(v, []).get(v)).push(b); }
    for (const b of prior) { const v = f(b); (pri.get(v) || pri.set(v, []).get(v)).push(b); }
    for (const [value, arr] of cur) {
      const r = rate(arr);
      if (arr.length >= PK_ALERT_MIN_SLICE && r < PK_ALERT_RATE) slices.push({ dim, value, n: arr.length, rate: r, priorRate: rate(pri.get(value) || []) });
    }
  }
  slices.sort((a, b) => a.rate - b.rate || b.n - a.n);
  return { rate: overall, slices };
}
