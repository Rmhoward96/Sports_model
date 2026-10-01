import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/boot.js"]);

test("slate keeps one row per game: the highest-alpha opportunity", () => {
  const preds = [{ sport: "nfl", game_pk: 1, home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions", commence_time: "2026-10-05T20:25:00Z" },
                 { sport: "cfb", game_pk: 2, home_team_name: "Oregon", away_team_name: "Ohio State", commence_time: "2026-10-04T23:30:00Z" }];
  const opps = [{ game_pk: 1, sport: "nfl", alpha: 70, marketLabel: "Spread", tier: "MEDIUM" },
                { game_pk: 1, sport: "nfl", alpha: 89, marketLabel: "ML", tier: "HIGH" }];
  const s = g.dashSlate(preds, opps);
  assert.equal(s.length, 2);
  assert.equal(s.find((r) => r.game_pk === 1).opp.alpha, 89);
  assert.equal(s.find((r) => r.game_pk === 2).opp, null);
  assert.deepEqual(s.map((r) => r.game_pk), [2, 1]);   // by kickoff
});

test("exposure shares sum to 100 and use graded bet counts", () => {
  const ev = [{ game_date: "2026-09-20", sport: "cfb", market: "moneyline", n: 6, wins: 3, losses: 3, pushes: 0, pnl: 10 },
              { game_date: "2026-09-21", sport: "nfl", market: "prop", n: 4, wins: 2, losses: 2, pushes: 0, pnl: -5 }];
  const e = g.dashExposure(ev, [], "2026-09-01");
  assert.deepEqual(e.bySport.map((x) => [x.sport, x.pct]), [["cfb", 60], ["nfl", 40]]);
  assert.equal(Math.round(e.bySport.reduce((s, x) => s + x.pct, 0)), 100);
});

// ---- extra coverage for the other pure helpers ----------------------------------------------

test("slate: an opportunity only attaches to the game of its own sport; same-game duplicates collapse", () => {
  const preds = [{ sport: "nfl", game_pk: 7, home_team_name: "A", away_team_name: "B", commence_time: "2026-10-05T17:00:00Z" },
                 { sport: "nfl", game_pk: 7, home_team_name: "A", away_team_name: "B", commence_time: "2026-10-05T17:00:00Z" }];
  const opps = [{ game_pk: 7, sport: "cfb", alpha: 99 }, { game_pk: 7, sport: "nfl", alpha: 66, evPct: 2 }, { game_pk: 7, sport: "nfl", alpha: 66, evPct: 5 }];
  const s = g.dashSlate(preds, opps);
  assert.equal(s.length, 1);
  assert.equal(s[0].opp.evPct, 5, "ties on alpha keep the higher EV");
  assert.equal(s[0].away, "B"); assert.equal(s[0].home, "A");
  assert.deepEqual(g.dashSlate(null, null), []);
});

test("exposure: window cut, prediction rows count, market types, P&L units beside each slice", () => {
  const ev = [{ game_date: "2026-08-01", sport: "nfl", market: "spread", n: 50, wins: 25, losses: 25, pushes: 0, pnl: 0 },   // before the window
              { game_date: "2026-09-20", sport: "cfb", market: "spread", n: 2, wins: 2, losses: 0, pushes: 0, pnl: 20 }];
  const pred = [{ game_date: "2026-09-20", sport: "cfb", market: "total", n: 3, wins: 1, losses: 2, pushes: 0, pnl: -10 },
                { game_date: "2026-09-21", sport: "nfl", market: "moneyline", n: 5, wins: 5, losses: 0, pushes: 0, pnl: 30 }];
  const e = g.dashExposure(ev, pred, "2026-09-01");
  assert.deepEqual(e.bySport.map((x) => [x.sport, x.staked, x.pct]), [["cfb", 5, 50], ["nfl", 5, 50]]);
  assert.equal(e.bySport.find((x) => x.sport === "cfb").units, 1);    // (20 - 10) / 10
  assert.equal(e.totalUnits, 4);
  assert.equal(e.staked, 10);
  assert.deepEqual(e.byMarket.map((x) => [x.label, x.pct]), [["Game Lines", 70], ["Player Props", 0], ["Totals", 30]]);
  const none = g.dashExposure([], [], "2026-09-01");
  assert.deepEqual(none.bySport, []); assert.equal(none.staked, 0);
  assert.ok(none.byMarket.every((x) => x.pct === 0));
});

test("short team names: NFL nickname, CFB school, others unchanged", () => {
  assert.equal(g.dashTeam("Detroit Lions", "nfl"), "Lions");
  assert.equal(g.dashTeam("San Francisco 49ers", "nfl"), "49ers");
  assert.equal(g.dashTeam("Alabama Crimson Tide", "cfb"), "Alabama");
  assert.equal(g.dashTeam("Oregon", "cfb"), "Oregon");
  assert.equal(g.dashTeam("Boston Red Sox", "mlb"), "Boston Red Sox");
  assert.equal(g.dashTeam("", "nfl"), "");
});

test("pick labels: ML, spread with the best book's line, total, prop", () => {
  const lineBy = new Map([["1|spread|home", -6.5], ["1|total|over", 56.5]]);
  const base = { kind: "line", sport: "nfl", game_pk: 1, matchup: "Detroit Lions @ Kansas City Chiefs" };
  assert.equal(g.dashPick({ ...base, market: "moneyline", side: "away" }, lineBy), "Lions ML");
  assert.equal(g.dashPick({ ...base, market: "spread", side: "home" }, lineBy), "Chiefs -6.5");
  assert.equal(g.dashPick({ ...base, market: "spread", side: "away" }, lineBy), "Lions spread");   // no line known -> no number invented
  assert.equal(g.dashPick({ ...base, market: "total", side: "over" }, lineBy), "Over 56.5");
  assert.equal(g.dashPick({ ...base, kind: "prop", market: "rec_yds", marketLabel: "Rec Yds", side: "under", line: 64.5, playerName: "X" }, lineBy), "Under 64.5 Rec Yds");
  assert.equal(g.dashPick({ ...base, market: "spread", side: "home" }, new Map([["1|spread|home", 3]])), "Chiefs +3");
});

test("date helpers: add days across month ends; cumulative daily units fill gaps", () => {
  assert.equal(g.dashAddDays("2026-10-01", -1), "2026-09-30");
  assert.equal(g.dashAddDays("2026-12-31", 1), "2027-01-01");
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
  const rows = g.dashGraded(results, picks, starts);
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
