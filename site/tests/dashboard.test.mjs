import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/boot.js"]);

test("slate keeps one row per game: the highest-alpha opportunity", () => {
  const preds = [{ sport: "nfl", game_pk: 1, home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions", commence_time: "2026-10-05T20:25:00Z" },
                 { sport: "cfb", game_pk: 2, home_team_name: "Oregon", away_team_name: "Ohio State", commence_time: "2026-10-04T23:30:00Z" }];
  const opps = [{ game_pk: 1, sport: "nfl", alpha: 70, marketLabel: "Spread", tier: "MEDIUM" },
                { game_pk: 1, sport: "nfl", alpha: 89, marketLabel: "ML", tier: "HIGH" }];
  const s = g.slateRows(preds, opps);
  assert.equal(s.length, 2);
  assert.equal(s.find((r) => r.game_pk === 1).opp.alpha, 89);
  assert.equal(s.find((r) => r.game_pk === 2).opp, null);
  assert.deepEqual(s.map((r) => r.game_pk), [2, 1]);   // by kickoff
});

test("exposure shares sum to 100 and use graded bet counts", () => {
  const ev = [{ game_date: "2026-09-20", sport: "cfb", market: "moneyline", n: 6, wins: 3, losses: 3, pushes: 0, pnl: 10 },
              { game_date: "2026-09-21", sport: "nfl", market: "prop", n: 4, wins: 2, losses: 2, pushes: 0, pnl: -5 }];
  const e = g.dashExposure(ev, "2026-09-01");
  assert.deepEqual(e.bySport.map((x) => [x.sport, x.pct]), [["cfb", 60], ["nfl", 40]]);
  assert.equal(Math.round(e.bySport.reduce((s, x) => s + x.pct, 0)), 100);
});

// ---- extra coverage for the other pure helpers ----------------------------------------------

test("slate: an opportunity only attaches to the game of its own sport; same-game duplicates collapse", () => {
  const preds = [{ sport: "nfl", game_pk: 7, home_team_name: "A", away_team_name: "B", commence_time: "2026-10-05T17:00:00Z" },
                 { sport: "nfl", game_pk: 7, home_team_name: "A", away_team_name: "B", commence_time: "2026-10-05T17:00:00Z" }];
  const opps = [{ game_pk: 7, sport: "cfb", alpha: 99 }, { game_pk: 7, sport: "nfl", alpha: 66, evPct: 2 }, { game_pk: 7, sport: "nfl", alpha: 66, evPct: 5 }];
  const s = g.slateRows(preds, opps);
  assert.equal(s.length, 1);
  assert.equal(s[0].opp.evPct, 5, "ties on alpha keep the higher EV");
  assert.equal(s[0].away, "B"); assert.equal(s[0].home, "A");
  assert.deepEqual(g.slateRows(null, null), []);
});

test("exposure: window cut, every graded row counts, market types, P&L units beside each slice", () => {
  const ev = [{ game_date: "2026-08-01", sport: "nfl", market: "spread", n: 50, wins: 25, losses: 25, pushes: 0, pnl: 0 },   // before the window
              { game_date: "2026-09-20", sport: "cfb", market: "spread", n: 2, wins: 2, losses: 0, pushes: 0, pnl: 20 }];
  const pred = [{ game_date: "2026-09-20", sport: "cfb", market: "total", n: 3, wins: 1, losses: 2, pushes: 0, pnl: -10 },
                { game_date: "2026-09-21", sport: "nfl", market: "moneyline", n: 5, wins: 5, losses: 0, pushes: 0, pnl: 30 }];
  const e = g.dashExposure([...ev, ...pred], "2026-09-01");
  assert.deepEqual(e.bySport.map((x) => [x.sport, x.staked, x.pct]), [["cfb", 5, 50], ["nfl", 5, 50]]);
  assert.equal(e.bySport.find((x) => x.sport === "cfb").units, 1);    // (20 - 10) / 10
  assert.equal(e.totalUnits, 4);
  assert.equal(e.staked, 10);
  assert.deepEqual(e.byMarket.map((x) => [x.label, x.pct]), [["Game Lines", 70], ["Player Props", 0], ["Totals", 30]]);
  const none = g.dashExposure([], "2026-09-01");
  assert.match(fs.readFileSync(new URL("../js/pages/dashboard.js", import.meta.url), "utf8"), /function dashExposure\(evPnlRows, sinceDate\)/, "no dead predPnlRows parameter");
  assert.deepEqual(none.bySport, []); assert.equal(none.staked, 0);
  assert.ok(none.byMarket.every((x) => x.pct === 0));
});

test("short team names: NFL nickname, CFB school, others unchanged", () => {
  assert.equal(g.shortTeam("Detroit Lions", "nfl"), "Lions");
  assert.equal(g.shortTeam("San Francisco 49ers", "nfl"), "49ers");
  assert.equal(g.shortTeam("Alabama Crimson Tide", "cfb"), "Alabama");
  assert.equal(g.shortTeam("Oregon", "cfb"), "Oregon");
  assert.equal(g.shortTeam("Boston Red Sox", "mlb"), "Boston Red Sox");
  assert.equal(g.shortTeam("", "nfl"), "");
});

test("pick labels: ML, spread with the best book's line, total, prop", () => {
  const lineBy = new Map([["1|spread|home", -6.5], ["1|total|over", 56.5]]);
  const base = { kind: "line", sport: "nfl", game_pk: 1, matchup: "Detroit Lions @ Kansas City Chiefs" };
  assert.equal(g.pickLabel({ ...base, market: "moneyline", side: "away" }, lineBy), "Lions ML");
  assert.equal(g.pickLabel({ ...base, market: "spread", side: "home" }, lineBy), "Chiefs -6.5");
  assert.equal(g.pickLabel({ ...base, market: "spread", side: "away" }, lineBy), "Lions spread");   // no line known -> no number invented
  assert.equal(g.pickLabel({ ...base, market: "total", side: "over" }, lineBy), "Over 56.5");
  assert.equal(g.pickLabel({ ...base, kind: "prop", market: "rec_yds", marketLabel: "Rec Yds", side: "under", line: 64.5, playerName: "X" }, lineBy), "Under 64.5 Rec Yds");
  assert.equal(g.pickLabel({ ...base, market: "spread", side: "home" }, new Map([["1|spread|home", 3]])), "Chiefs +3");
});

test("date helpers: add days across month ends; cumulative daily units fill gaps", () => {
  assert.equal(g.addDays("2026-10-01", -1), "2026-09-30");
  assert.equal(g.addDays("2026-12-31", 1), "2027-01-01");
  const rows = [{ game_date: "2026-09-20", n: 1, wins: 1, losses: 0, pushes: 0, pnl: 10 },
                { game_date: "2026-09-22", n: 2, wins: 0, losses: 2, pushes: 0, pnl: -20 },
                { game_date: "2026-09-25", n: 1, wins: 1, losses: 0, pushes: 0, pnl: 5 }];   // after `to`
  const c = g.dashCumUnits(rows, "2026-09-19", "2026-09-23");
  assert.deepEqual(c.map((p) => [p.date, p.units]), [["2026-09-20", 1], ["2026-09-21", 1], ["2026-09-22", -1], ["2026-09-23", -1]]);
  assert.deepEqual(g.dashCumUnits([], "2026-09-01", "2026-09-30"), []);
});

test("graded +EV picks: record cut, implied from the best price, edge in points", () => {
  const starts = new Map([["nfl", { starts_at: "2026-09-29T17:00:00Z" }]]);
  const results = [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", won: true },
                   { sport: "nfl", game_pk: 2, market: "moneyline", side: "home", won: false },     // before the NFL restart
                   { sport: "cfb", game_pk: 3, market: "spread", side: "away", won: null },        // ungraded
                   { sport: "cfb", game_pk: 4, market: "spread", side: "away", won: false }];
  const picks = [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", true_prob: 0.6, best_price: 100, commence_time: "2026-10-04T17:00:00Z" },
                 { sport: "nfl", game_pk: 2, market: "moneyline", side: "home", true_prob: 0.6, best_price: 100, commence_time: "2026-09-20T17:00:00Z" },
                 { sport: "cfb", game_pk: 3, market: "spread", side: "away", true_prob: 0.6, best_price: 100, commence_time: "2026-09-20T17:00:00Z" },
                 { sport: "cfb", game_pk: 4, market: "spread", side: "away", true_prob: 0.55, best_price: null, commence_time: "2026-09-20T17:00:00Z" }];
  const rows = g.gradedLinePicks(results, picks, starts);
  assert.deepEqual(rows.map((r) => r.game_pk), [1, 4]);
  assert.equal(rows[0].implied, 0.5); assert.ok(Math.abs(rows[0].edgePp - 10) < 1e-9); assert.equal(rows[0].date, "2026-10-04");
  assert.equal(rows[1].implied, null); assert.equal(rows[1].edgePp, null);
});

test("model vs market: largest gaps first, spreads from the home side, missing lines skipped", () => {
  const preds = [{ sport: "nfl", game_pk: 1, home_team_name: "H", away_team_name: "A", pred_home_score: 24, pred_away_score: 20, market_spread: -1.5, market_total: 40 },
                 { sport: "nfl", game_pk: 2, home_team_name: "H2", away_team_name: "A2", pred_home_score: 20, pred_away_score: 20, market_spread: null, market_total: null }];
  const m = g.dashModelVsMarket(preds);
  assert.deepEqual(m.map((x) => [x.market, x.from, x.to, x.diff]), [["total", 40, 44, 4], ["spread", -1.5, -4, -2.5]]);
});

test("new signals: first flagged inside the last 24h, matched to opportunities on the board", () => {
  const NOW = Date.parse("2026-10-02T15:00:00Z");
  const hist = [{ kind: "line", sport: "nfl", game_pk: 1, market: "moneyline", side: "home", at: "2026-10-02T01:00:00Z" },
                { kind: "line", sport: "nfl", game_pk: 2, market: "spread", side: "away", at: "2026-09-30T01:00:00Z" },
                { kind: "prop", sport: "nfl", game_pk: 3, market: "rec_yds", side: "over", player_name: "P", at: "2026-10-02T10:00:00Z" }];
  const opps = [{ kind: "line", sport: "nfl", game_pk: 1, market: "moneyline", side: "home", alpha: 70 },
                { kind: "line", sport: "nfl", game_pk: 2, market: "spread", side: "away", alpha: 90 },
                { kind: "prop", sport: "nfl", game_pk: 3, market: "rec_yds", side: "over", playerName: "P", alpha: 66 }];
  const s = g.dashNewSignals(opps, hist, NOW);
  assert.deepEqual(s.map((x) => [x.opp.game_pk, x.at]), [[3, "2026-10-02T10:00:00Z"], [1, "2026-10-02T01:00:00Z"]]);   // newest first
});

test("multi-segment donut: arcs only for positive shares, empty input -> empty string", () => {
  const svg = g.dashDonut([{ pct: 60, color: "red" }, { pct: 0, color: "blue" }, { pct: 40, color: "green" }], { center: "4.0u", caption: "Total Units" });
  assert.equal((svg.match(/stroke-dasharray/g) || []).length, 2);
  assert.ok(svg.includes("4.0u") && svg.includes("Total Units"));
  assert.equal(g.dashDonut([], {}), "");
  assert.equal(g.dashDonut([{ pct: 0 }], {}), "");
});

test("buildDashboard renders every card from stubbed data and never throws on empty sources", async () => {
  const D = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/boot.js"],
    { globals: { fetch: async () => ({ ok: true, json: async () => [] }) } });
  const html = await D.buildDashboard();
  for (const t of ["Today at a Glance", "Games Tracked", "Live +EV Opportunities", "Best Current Edge", "Model Hit Rate", "Units",
    "Active Signals", "Best Opportunities Right Now", "Today’s Slate", "Performance Snapshot", "Market Movers", "Portfolio &amp; Exposure", "Watchlist"]) {
    assert.ok(html.includes(t), `missing ${t}`);
  }
  assert.ok(html.includes("No +EV opportunities on the board right now."));
  assert.ok(!html.includes("Recent Model Updates"), "Phase B card omitted");
  assert.ok(!/NaN|undefined/.test(html), "no NaN / undefined leaks");
});

// ---- fix round 1 ----------------------------------------------------------------------------
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/boot.js"];
const quiet = () => { const logs = { warn: [], error: [] }; return { logs, console: { ...console, warn: (...a) => logs.warn.push(a.join(" ")), error: (...a) => logs.error.push(a.join(" ")) } }; };

test("board opps: today = every tiered pick kicking off in [now, now+7d] by EV desc; other dates stay date-scoped", () => {
  const NOW = Date.parse("2026-10-01T15:00:00Z");
  const at = (h) => new Date(NOW + h * 36e5).toISOString();
  const o = (id, h, ev) => ({ game_pk: id, commence: at(h), evPct: ev });
  const tiered = [o(1, -2, 50), o(2, 1, 5), o(3, 24 * 3, 20), o(4, 24 * 7, 9), o(5, 24 * 7 + 1, 99), o(6, 30, 12)];
  const today = g.dashBoardOpps(tiered, "2026-10-01", NOW, true);
  assert.deepEqual(today.map((x) => x.game_pk), [3, 6, 4, 2]);   // past (1) and beyond 7d (5) are out
  const day = g.dashBoardOpps(tiered, "2026-10-02", NOW, false);   // ET date of +30h is 10-02
  assert.deepEqual(day.map((x) => x.game_pk), [6]);
});

test("perf: one population (ev_pnl_daily), hit rate = wins/(wins+losses), vs-market and edge from priced game lines only", () => {
  const ev = [{ game_date: "2026-09-20", sport: "nfl", market: "moneyline", n: 10, wins: 6, losses: 3, pushes: 1, pnl: 50 },
              { game_date: "2026-09-21", sport: "nfl", market: "prop", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 0 },
              { game_date: "2026-08-01", sport: "cfb", market: "spread", n: 9, wins: 9, losses: 0, pushes: 0, pnl: 90 }];   // outside range
  const graded = [{ date: "2026-09-20", won: true, implied: 0.5, edgePp: 10 }, { date: "2026-09-20", won: false, implied: 0.5, edgePp: 6 },
                  { date: "2026-09-20", won: true, implied: null, edgePp: null }];
  const p = g.perfWindow(ev, graded, "2026-09-01", "2026-09-30");
  assert.equal(p.n, 12); assert.equal(p.wins, 7); assert.equal(p.losses, 4);
  assert.ok(Math.abs(p.hitRate - 7 / 11) < 1e-12);
  assert.equal(p.units, 5); assert.ok(Math.abs(p.roiPct - 5 / 12 * 100) < 1e-9);
  assert.equal(p.pricedN, 2); assert.equal(p.vsMarket, 0);   // 1 of 2 won vs 50% implied
  assert.equal(p.avgEdge, 8); assert.equal(p.edgeN, 2);
  const none = g.perfWindow([], [], "2026-09-01", "2026-09-30");
  assert.equal(none.hitRate, null); assert.equal(none.avgEdge, null); assert.equal(none.n, 0);
});

test("exposure classes are explicit: parlays are their own row (only when present); unknown markets are ignored with a warning", () => {
  const q = quiet();
  const D = loadScripts(FILES, { globals: { console: q.console } });
  const ev = [{ game_date: "2026-09-20", sport: "nfl", market: "moneyline", n: 4, wins: 2, losses: 2, pushes: 0, pnl: 0 },
              { game_date: "2026-09-20", sport: "nfl", market: "prop", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 0 },
              { game_date: "2026-09-20", sport: "nfl", market: "rec_yds", n: 1, wins: 1, losses: 0, pushes: 0, pnl: 10 },
              { game_date: "2026-09-20", sport: "nfl", market: "parlay", n: 1, wins: 0, losses: 1, pushes: 0, pnl: -10 },
              { game_date: "2026-09-20", sport: "nfl", market: "mystery", n: 50, wins: 50, losses: 0, pushes: 0, pnl: 500 }];
  const e = D.dashExposure(ev, "2026-09-01");
  assert.deepEqual(e.byMarket.map((x) => [x.label, x.staked]), [["Game Lines", 4], ["Player Props", 3], ["Totals", 0], ["Parlays", 1]]);
  assert.equal(e.staked, 8, "the unknown market is not counted anywhere");
  assert.equal(q.logs.warn.length, 1); assert.match(q.logs.warn[0], /mystery/);
  const noParlay = D.dashExposure(ev.slice(0, 2), "2026-09-01");
  assert.deepEqual(noParlay.byMarket.map((x) => x.label), ["Game Lines", "Player Props", "Totals"]);
});

// ---- deterministic pick price ----------------------------------------------------------------
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };
test("evGradedPicks keeps the LATEST rebuild per pick, reads every page in a total created_at order, no cap warning", async () => {
  const q = quiet(); const urls = [];
  const row = (k, price, at) => ({ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", best_price: price, created_at: at, ...k });
  const rows = [row({}, 120, "2026-09-30T12:00:00Z"), row({}, 100, "2026-09-30T08:00:00Z"), row({ side: "away" }, -110, "2026-09-30T09:00:00Z")];   // unordered on purpose
  const D = loadScripts(FILES, { globals: { console: q.console, fetch: async (u) => { urls.push(u); return { ok: true, json: async () => rows }; } } });
  const picks = await D.evGradedPicks();
  assert.equal(picks.length, 2);
  assert.equal(picks.find((p) => p.side === "home").best_price, 120);
  assert.match(urls[0], /order=created_at\.asc,sport\.asc,game_pk\.asc,market\.asc,side\.asc,model_version\.asc/, "ties broken by the primary key");
  assert.match(urls[0], /&limit=1000&offset=0$/); assert.equal(urls.length, 1, "a short first page ends the read");
  assert.doesNotMatch(urls[0], /limit=4000/);
  assert.equal(q.logs.warn.length, 0);
  // 1,500 pick rows: two pages, the newest rows (page 2) are kept
  const all = Array.from({ length: 1500 }, (_, i) => row({ game_pk: i }, 100 + i, "2026-09-30T08:00:00Z"));
  const D2 = loadScripts(FILES, { globals: { console: q.console, fetch: async (u) => { const off = +u.match(/offset=(\d+)/)[1]; return { ok: true, json: async () => all.slice(off, off + 1000) }; } } });
  const p2 = await D2.evGradedPicks();
  assert.equal(p2.length, 1500); assert.ok(p2.some((p) => p.game_pk === 1499));
  assert.equal(q.logs.warn.length, 0);
});

test("gradedLinePicks: the latest-created pick row sets the price whatever the input order", () => {
  const results = [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", won: true }];
  const mk = (price, at) => ({ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", true_prob: 0.6, best_price: price, created_at: at, commence_time: "2026-10-04T17:00:00Z" });
  for (const picks of [[mk(100, "2026-09-30T08:00:00Z"), mk(200, "2026-09-30T12:00:00Z")], [mk(200, "2026-09-30T12:00:00Z"), mk(100, "2026-09-30T08:00:00Z")]]) {
    assert.ok(Math.abs(g.gradedLinePicks(results, picks, new Map())[0].implied - 1 / 3) < 1e-9);
  }
});

// ---- card isolation --------------------------------------------------------------------------
test("safeCard: a throwing card renders a placeholder and logs, it does not throw", () => {
  const q = quiet();
  const D = loadScripts(FILES, { globals: { console: q.console } });
  const html = D.safeCard("Boom", () => { throw new Error("kaput"); }, {}, "ca-card ca-dash-card", "dash-x");
  assert.match(html, /id="dash-x"/); assert.ok(html.includes("This panel couldn't load.")); assert.match(html, /ca-empty/);
  assert.equal(q.logs.error.length, 1); assert.match(q.logs.error[0], /Boom/);
  assert.equal(D.safeCard("Fine", () => "<p>ok</p>", {}), "<p>ok</p>");
});

// ---- populated render ------------------------------------------------------------------------
const EVIL_AWAY = 'Evil "Q" <b>Crew</b>', HOME = 'Kansas City "Chiefs"', PLAYER = 'O"Brien <i>Zed</i>';
function populated({ search = "", watch = null, predH = 0 } = {}) {
  const NOW = Date.now(), iso = (h) => new Date(NOW + h * 36e5).toISOString(), dayAgo = (d) => new Date(NOW - d * 864e5).toISOString().slice(0, 10);
  const line = (o) => ({ sport: "nfl", game_pk: 1, matchup: `${EVIL_AWAY} @ ${HOME}`, market: "moneyline", side: "home", true_prob: 0.7, best_line_implied: 0.5,
    best_price: 100, best_book: "draftkings", ev_best: 0.3, is_pick: true, commence_time: iso(48), ...o });
  const evCurrentNfl = [line({}),                                                     // +2d  EV 30
    line({ game_pk: 2, commence_time: iso(24 * 10), ev_best: 0.4 }),                  // +10d: outside the 7-day board
    line({ game_pk: 3, commence_time: iso(-24), ev_best: 0.5 }),                      // yesterday: not upcoming
    line({ game_pk: 4, commence_time: iso(96), ev_best: 0.12, true_prob: 0.6, side: "away" })];   // +4d  EV 12
  const props = [{ sport: "nfl", game_pk: 5, matchup: `${EVIL_AWAY} @ ${HOME}`, market: "rec_yds", side: "over", line: 64.5, player_name: PLAYER, model_prob: 0.62,
    best_price: -110, best_book: "fanduel", ev_best: 0.15, is_pick: true, commence_time: iso(72) }];
  const pred = { game_pk: 1, home_team_name: HOME, away_team_name: EVIL_AWAY, commence_time: iso(predH), home_win_prob: 0.6,
    pred_home_score: 27, pred_away_score: 20, market_spread: -3, market_total: 41.5 };
  const evPnl = [{ game_date: dayAgo(5), sport: "nfl", market: "moneyline", n: 10, wins: 6, losses: 3, pushes: 1, pnl: 50 },
    { game_date: dayAgo(3), sport: "cfb", market: "spread", n: 4, wins: 1, losses: 3, pushes: 0, pnl: -20 },
    { game_date: dayAgo(2), sport: "nfl", market: "prop", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 0 }];
  const results = [{ sport: "nfl", game_pk: 11, market: "moneyline", side: "home", won: true }, { sport: "nfl", game_pk: 12, market: "moneyline", side: "home", won: false }];
  const gradedPick = (pk) => ({ sport: "nfl", game_pk: pk, market: "moneyline", side: "home", true_prob: 0.6, best_price: 100, created_at: iso(-100), commence_time: iso(-72) });
  const requested = [];
  const fetch = async (url) => {
    const { path, q } = urlParts(url); requested.push(path);
    let rows = [];
    if (path === "predictions_current") rows = q.includes("sport=eq.nfl") ? [pred] : [];
    else if (path === "ev_current") rows = q.includes("sport=eq.nfl") ? evCurrentNfl : [];
    else if (path === "ev_prop_picks_current") rows = q.includes("sport=eq.nfl") ? props : [];
    else if (path === "ev_pnl_daily") rows = evPnl;
    else if (path === "ev_results") rows = results;
    // iso(-0.01): flagged 36 seconds ago, so the pick stays inside today's ET day even when the suite runs just after midnight
    else if (path === "ev_picks") rows = q.includes("created_at=gte") ? [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", created_at: iso(-0.01) }] : [gradedPick(11), gradedPick(12)];
    return { ok: true, json: async () => rows };
  };
  const storage = new Map(); if (watch) storage.set("ca-watchlist", JSON.stringify(watch));
  const D = loadScripts(FILES, { storage, globals: { fetch, location: { search, href: `http://localhost/${search}` } } });
  return { D, requested, NOW };
}
const slice = (html, from, to) => html.slice(html.indexOf(`id="${from}"`), html.indexOf(`id="${to}"`));

test("populated render (today): board lists the next 7 days by EV (kickoff in the matchup tooltip); stat cards use one performance population", async () => {
  const { D, requested } = populated();
  const html = await D.buildDashboard();
  const opps = slice(html, "dash-opps", "dash-slate");
  const rows = opps.match(/<tr data-href/g) || [];
  assert.equal(rows.length, 3, "A (+2d), prop (+3d), D (+4d); not the past game or the +10d game");
  assert.ok(!opps.includes("Game Time") && !opps.includes("ca-dash-gtime"), "F3: no Game Time column on the dashboard");
  assert.match(opps, /class="ca-team ca-dash-who" title="[^"]+ · [A-Z][a-z]{2} \d{1,2}:\d{2} [AP]M"/, "kickoff time rides in the matchup cell's tooltip");
  assert.deepEqual([...opps.matchAll(/<tr data-href="game\.html\?sport=nfl&amp;game=(\d+)"|<tr data-href="game\.html\?sport=nfl&game=(\d+)"/g)].map((m) => m[1] || m[2]), ["1", "5", "4"], "sorted by EV desc (30, 15, 12)");
  // one population: ev_pnl_daily only
  assert.ok(!requested.includes("prediction_pnl_daily"), "prediction_pnl_daily is not fetched");
  assert.ok(html.includes("+3.0u"), "units (30D)");
  assert.ok(html.includes("16 graded +EV picks"), "caption names the population");
  assert.ok(html.includes("53.3%"), "hit rate = 8 / (8 + 7)");
  assert.ok(html.includes("8-7"));
  assert.match(html, /1 new today/);
  assert.ok(html.includes("(game lines)"), "vs-market is labelled as game lines");
  assert.ok(html.includes("Graded +EV picks, last 30 days"));
  assert.ok(!/NaN|undefined/.test(html), "no NaN / undefined leaks");
});

test("populated render (a date): the board is date-scoped to that date", async () => {
  const { NOW } = populated();
  const d = new Date(NOW + 48 * 36e5).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const P = populated({ search: `?date=${d}` });
  const html = await P.D.buildDashboard();
  const opps = slice(html, "dash-opps", "dash-slate");
  assert.equal((opps.match(/<tr data-href/g) || []).length, 1);
  assert.ok(opps.includes("game=1") && !opps.includes("game=5") && !opps.includes("game=4"));
  assert.ok(html.includes("Top edges for this date — model for props, sharp fair price for game lines"));
});

test("populated render: team and player names are escaped everywhere, including a Movers tab", async () => {
  const { D } = populated();
  D.dashState.movers = "model";
  const html = await D.buildDashboard();
  assert.ok(html.includes("&lt;b&gt;Crew&lt;/b&gt;"), "team name is escaped");
  assert.ok(html.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt;"), "player name is escaped");
  assert.ok(!html.includes("<b>Crew</b>") && !html.includes("<i>Zed</i>"), "no raw markup from data");
  const mv = slice(html, "dash-movers", "dash-expo").slice(0, 4000);
  assert.ok(mv.includes("Market → model") && mv.includes("pts"), "Model vs. Market tab has rows");
  assert.ok(!/NaN|undefined/.test(html));
  D.dashState.movers = "moves";
});

test("populated render: Watchlist shows a starred game (on star) and a starred player; slate star reflects it", async () => {
  const { D } = populated({ watch: { games: ["1"], teams: [], players: [PLAYER] } });
  let html = await D.buildDashboard();
  const wl = html.slice(html.indexOf('id="watchlist"'));
  assert.match(wl, /class="ca-star on"[^>]*data-star-kind="games" data-star-id="1"/);
  assert.ok(wl.includes("&lt;b&gt;Crew&lt;/b&gt;"));
  const slate = slice(html, "dash-slate", "dash-perf");
  assert.match(slate, /class="ca-star on"[^>]*data-star-kind="games" data-star-id="1"/);
  D.dashState.watch = "players";
  html = await D.buildDashboard();
  const wlp = html.slice(html.indexOf('id="watchlist"'));
  assert.ok(wlp.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt;") && wlp.includes("Over 64.5 Rec Yds"));
  assert.ok(!/NaN|undefined/.test(html));
  D.dashState.watch = "games";
});

test("pill state in window.__caDash survives a second buildDashboard()", async () => {
  const { D } = populated();
  D.dashState.kind = "prop"; D.dashState.slate = "nfl";
  const a = await D.buildDashboard(), b = await D.buildDashboard();
  for (const html of [a, b]) {
    assert.match(html, /class="ca-pill on" data-pill="dash-kind" data-key="prop"/);
    assert.match(html, /class="ca-pill on" data-pill="dash-slate" data-key="nfl"/);
    assert.equal((slice(html, "dash-opps", "dash-slate").match(/<tr data-href/g) || []).length, 1, "prop filter applies");
  }
  assert.equal(D.dashState.kind, "prop");
});

test("slate status dot: past dates are neutral Final, future dates say Scheduled (never Later today / Live)", async () => {
  const P = populated();
  const day = (h) => new Date(P.NOW + h * 36e5).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
  const past = await populated({ search: `?date=${day(-48)}`, predH: -48 }).D.buildDashboard();
  assert.match(past, /ca-dash-dot final" title="Final"/);
  assert.ok(!/ca-dash-dot live|Later today|Live \/ started/.test(past));
  const future = await populated({ search: `?date=${day(72)}`, predH: 72 }).D.buildDashboard();
  assert.match(future, /ca-dash-dot later" title="Scheduled"/);
  assert.ok(!future.includes("Later today"));
  assert.match(await P.D.buildDashboard(), /ca-dash-dot (live|soon|later)" title="(Live \/ started|Starts within 3 hours|Later today)"/);   // today keeps its states
});

test("Line Moves empty state is user-facing copy", async () => {
  const { D } = populated();
  D.dashState.movers = "moves";
  const html = await D.buildDashboard();
  assert.ok(html.includes("No line moves captured for upcoming games yet."));
  assert.ok(!/migration|view/i.test(slice(html, "dash-movers", "dash-expo").replace(/viewBox/g, "")));
});

test("R19: Best Opportunities marks game-line probabilities as the sharp fair price; prop rows carry no marker", async () => {
  const { D } = populated();
  const html = await D.buildDashboard();
  const opps = slice(html, "dash-opps", "dash-slate");
  assert.match(opps, /<th title="Model for player props; sharp fair price \(Pinnacle no-vig\) for game lines">Model Prob\.<\/th>/);
  const rows = opps.split("<tr data-href").slice(1);
  const lineRow = rows.find((r) => r.includes("game=1")), propRow = rows.find((r) => r.includes("game=5"));
  assert.match(lineRow, /70\.0%<span class="ca-fair"[^>]*>fair<\/span>/, "game line: fair marker beside the %");
  assert.ok(!propRow.includes("ca-fair"), "prop: the model's own probability, no marker");
  // D2: the visible subtitle is the short mockup line; the R19 wording lives in its tooltip
  assert.ok(opps.includes("<p title=\"Top edges for the next 7 days — model for props, sharp fair price for game lines, sorted by expected value.\">Top model edges across all sports, sorted by expected value.</p>"));
});

test("Best Current Edge picks and shows the largest edge in probability points (not EV%), with the matchup line", () => {
  const base = { kind: "line", sport: "nfl", market: "moneyline", side: "home", matchup: "Detroit Lions @ Kansas City Chiefs", tier: "HIGH" };
  const D = { lineBy: new Map(), opps: [{ ...base, game_pk: 1, evPct: 30, edgePp: 5 }, { ...base, game_pk: 2, side: "away", evPct: 8, edgePp: 12.4 }] };
  const html = g.dashStatBest(D);
  assert.match(html, /class="ca-stat-value pos">\+12\.4%</, "edge in points, from the higher-edge opportunity");
  assert.ok(html.includes("game=2") && html.includes("Lions ML") && html.includes("Lions @ Chiefs"));
  assert.ok(!html.includes("+30.0%"));
  assert.match(g.dashStatBest({ lineBy: new Map(), opps: [] }), /No \+EV opportunities on this date\./);
});

test("R24: Watchlist Teams tab lists starred teams with their next game on the board, else just the name", async () => {
  const { D } = populated({ watch: { games: [], teams: [HOME, "Nowhere FC"], players: [] } });
  D.dashState.watch = "teams";
  const html = await D.buildDashboard();
  const wl = html.slice(html.indexOf('id="watchlist"'));
  const rows = wl.split('class="ca-dash-wl-row"').slice(1);
  assert.equal(rows.length, 2);
  assert.match(rows[0], /class="ca-star on" data-star-kind="teams" data-star-id="Kansas City &quot;Chiefs&quot;"/);
  assert.ok(rows[0].includes("game.html?sport=nfl&game=1") && rows[0].includes("Kansas City &quot;Chiefs&quot;") && rows[0].includes("&lt;b&gt;Crew&lt;/b&gt;"), "links the team's next game, escaped");
  assert.ok(rows[1].includes("Nowhere FC") && rows[1].includes("No upcoming game"));
  assert.ok(!/NaN|undefined/.test(html));
  D.dashState.watch = "games";
});

test("R24: Watchlist Teams tab skips a team's finished game; Players tab shows a starred player without a prop by name only", async () => {
  const { D } = populated({ watch: { games: [], teams: [HOME], players: ["Nobody Special"] }, predH: -30 });
  D.dashState.watch = "teams";
  let html = await D.buildDashboard();
  assert.ok(html.slice(html.indexOf('id="watchlist"')).includes("No upcoming game"), "a game that kicked off 30h ago is not 'next'");
  D.dashState.watch = "players";
  html = await D.buildDashboard();
  const wl = html.slice(html.indexOf('id="watchlist"'));
  assert.match(wl, /class="ca-star on" data-star-kind="players" data-star-id="Nobody Special"/);
  assert.ok(wl.includes("No +EV prop on the board"));
  D.dashState.watch = "games";
});

// ---- fidelity pass: D1-D8 ------------------------------------------------------------------------
const ths = (html) => [...html.matchAll(/<th[^>]*>([^<]*)<\/th>/g)].map((m) => m[1]);
const statBlock = (html, label) => { const i = html.indexOf(`<span class="ca-stat-lt">${label}`); return html.slice(html.lastIndexOf('<div class="ca-card ca-stat', i), html.indexOf("</div></div></div>", i) + 18); };

test("D3: Opportunities table columns are exactly # · Sport · Matchup/Player · Market · Best Odds · Model Prob. · Market Prob. · Edge · Alpha Score · Confidence · arrow", async () => {
  const { D } = populated();
  const html = await D.buildDashboard();
  const opps = slice(html, "dash-opps", "dash-slate");
  assert.deepEqual(ths(opps), ["#", "Sport", "Matchup / Player", "Market", "Best Odds", "Model Prob.", "Market Prob.", "Edge", "Alpha Score", "Confidence", ""]);
  const row = opps.split("<tr data-href")[1];
  assert.equal((row.match(/<td/g) || []).length, 11, "one cell per header");
  assert.ok(row.includes("ca-dash-sport-t"), "sport label is collapsible on narrow cards");
});

test("D2: Opportunities head is title + one-line subtitle + View All; pills are sports, a gap, then Game Lines / Player Props", async () => {
  const { D } = populated();
  const opps = slice(await D.buildDashboard(), "dash-opps", "dash-slate");
  assert.match(opps, /<div class="ca-card-head"><h2>Best Opportunities Right Now<\/h2><p title="[^"]+">Top model edges across all sports, sorted by expected value\.<\/p><span class="ca-info"[^>]*>i<\/span><a class="ca-link" href="ev\.html">View All →<\/a><\/div>/);
  const row = opps.slice(opps.indexOf('class="ca-dash-pillrow"'), opps.indexOf('<div class="ca-table-wrap">'));
  assert.deepEqual([...row.matchAll(/data-pill="(dash-op|dash-kind)" data-key="(\w+)"/g)].map((m) => m[2]), ["all", "nfl", "cfb", "mlb", "nba", "line", "prop"]);
  assert.ok(row.indexOf('class="ca-dash-pillsep"') > row.indexOf('data-key="nba"') && row.indexOf('class="ca-dash-pillsep"') < row.indexOf('data-key="line"'), "gap between the two pill groups");
});

test("D4: Today's Slate has its pills on the title row (before View All) and the columns Time · Matchup/Player · Market · Line · Alpha Score · arrow", async () => {
  const { D } = populated();
  const slate = slice(await D.buildDashboard(), "dash-slate", "dash-perf");
  const head = slate.slice(0, slate.indexOf("</a></div>") + 10);
  assert.match(head, /^id="dash-slate"><div class="ca-card-head"><h2>Today’s Slate<\/h2><div class="ca-pills"[\s\S]*?<\/div><a class="ca-link" href="[a-z]+\.html">View All →<\/a><\/div>$/);
  assert.deepEqual([...head.matchAll(/data-pill="dash-slate" data-key="(\w+)"/g)].map((m) => m[1]), ["all", "nfl", "cfb", "mlb", "nba"]);
  assert.deepEqual(ths(slate), ["Time (ET)", "Matchup / Player", "Market", "Line", "Alpha Score", ""]);
});

test("D5: top row = Opportunities + Slate; bottom row = Performance, Movers, Portfolio, Watchlist in ONE four-card row", async () => {
  const { D } = populated();
  const html = await D.buildDashboard();
  const top = html.slice(html.indexOf('class="ca-dash-top"'), html.indexOf('class="ca-dash-bottom"'));
  const bottom = html.slice(html.indexOf('class="ca-dash-bottom"'));
  const ids = (h) => [...h.matchAll(/<section class="ca-card ca-dash-card" id="([\w-]+)"/g)].map((m) => m[1]);
  assert.deepEqual(ids(top), ["dash-opps", "dash-slate"]);
  assert.deepEqual(ids(bottom), ["dash-perf", "dash-movers", "dash-expo", "watchlist"]);
  assert.ok(!html.includes("ca-dash-main") && !html.includes("ca-dash-sub"), "the two-column layout is gone");
  const theme = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  assert.match(theme, /\.ca-dash-bottom\{display:grid;grid-template-columns:minmax\(0,1\.48fr\) minmax\(0,1\.29fr\) minmax\(0,1fr\) minmax\(0,1fr\);[^}]*align-items:stretch/, "31 / 27 / 21 / 21 %, equal heights");
});

test("D6: Performance Snapshot: pills + View Track Record on the title row, four mini boxes, population caption in a tooltip", async () => {
  const { D } = populated();
  const perf = slice(await D.buildDashboard(), "dash-perf", "dash-movers");
  assert.match(perf, /<div class="ca-card-head"><h2>Performance Snapshot<\/h2><span class="ca-info" title="16 graded \+EV picks[^"]*"[\s\S]*?<div class="ca-pills"[\s\S]*?<\/div><a class="ca-link" href="track-record\.html">View Track Record →<\/a><\/div>/);
  assert.deepEqual([...perf.matchAll(/<div class="ca-dash-mini"><span>([^<]+)<\/span>/g)].map((m) => m[1]), ["ROI", "Units", "Hit Rate", "Avg. Edge"]);
  assert.ok(!perf.includes("ca-dash-cap"), "no caption paragraph under the boxes");
  assert.match(perf, /height:140px/, "short cumulative-units chart");
});

test("D7: Portfolio & Exposure caption is a tooltip beside the title, not a paragraph", async () => {
  const { D } = populated();
  const expo = slice(await D.buildDashboard(), "dash-expo", "watchlist");
  assert.match(expo, /<h2>Portfolio &amp; Exposure<\/h2><span class="ca-info" title="Graded \+EV picks, last 30 days · \d+ bets staked"/);
  assert.ok(!expo.includes("ca-dash-cap"));
  assert.ok(expo.includes("Total Units") && expo.includes("Market Type Exposure"));
});

test("D1: stat cards keep to one short sub-line; the detail moves into the (i) tooltip", async () => {
  const { D } = populated();
  const html = await D.buildDashboard();
  const hit = statBlock(html, "Model Hit Rate"), units = statBlock(html, "Units");
  assert.match(hit, /class="ca-stat-sub " title="[+-]?[\d.]+% vs\. market">/, "hit rate sub = '+X% vs. market' only");
  assert.ok(!hit.slice(hit.indexOf("ca-stat-sub")).includes("8-7"), "record is not in the sub-line");
  assert.match(hit, /class="ca-info" title="8-7 · 16 graded \+EV picks[^"]*\(game lines\)/, "record + population in the tooltip");
  assert.match(units, /class="ca-stat-sub " title="[+-]?[\d.]+% ROI">/);
  assert.match(units, /class="ca-info" title="16 graded \+EV picks/);
  assert.ok(!units.slice(units.indexOf("ca-stat-sub")).includes("graded +EV picks"));
  assert.match(statBlock(html, "Live +EV Opportunities"), /ca-stat-label" title="Live \+EV Opportunities"/);
});

test("D1: Best Current Edge is ONE line: logo, 'Lions ML' (or 'Player Market'), at, logo; matchup in the hover title", () => {
  const base = { sport: "nfl", market: "moneyline", side: "home", matchup: "Detroit Lions @ Kansas City Chiefs", tier: "HIGH", game_pk: 7, edgePp: 8 };
  const line = g.dashStatBest({ lineBy: new Map(), opps: [{ ...base, kind: "line" }] });
  assert.match(line, /<a class="ca-dash-be"[^>]* title="Chiefs ML · Lions @ Chiefs">[\s\S]*?<span>Chiefs ML<\/span><i>at<\/i>/);
  assert.ok(!line.includes("<br>"), "no forced line break");
  const prop = g.dashStatBest({ lineBy: new Map(), opps: [{ ...base, kind: "prop", market: "prop", side: "under", line: 85.5, marketLabel: "Rec Yds", playerName: "Chris <Olave>" }] });
  assert.ok(prop.includes("<span>Chris &lt;Olave&gt; Under 85.5 Rec Yds</span>") && !prop.includes("<br>"));
});

test("D8: Watchlist keeps Games / Teams / Players pills and Recent Model Updates stays hidden", async () => {
  const { D } = populated();
  const html = await D.buildDashboard();
  const wl = html.slice(html.indexOf('id="watchlist"'));
  assert.deepEqual([...wl.matchAll(/data-pill="dash-watch" data-key="(\w+)"/g)].map((m) => m[1]), ["games", "teams", "players"]);
  assert.ok(!html.includes("Recent Model Updates"));
});
