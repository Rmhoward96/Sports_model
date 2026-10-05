import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
import fs from "node:fs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"]);
test("track rows: one per graded market, favorite-side probability, push handling", () => {
  const acc = [{ sport: "cfb", game_pk: 9, game_date: "2026-09-27", home_team_name: "Texas", away_team_name: "Oklahoma",
    win_prob: 0.581, predicted_winner: "Texas", actual_winner: "Texas", winner_correct: true, pred_margin: 6.2, actual_margin: 10,
    market_spread: -3.5, spread_pick_correct: true, pred_total: 55, actual_total: 54.5, market_total: 54.5, total_pick_correct: null }];
  const rows = g.trackRows(acc, [{ game_pk: 9, market: "spread", pnl: 9.09 }]);
  assert.deepEqual(rows.map((r) => [r.market, r.result]), [["moneyline", "W"], ["spread", "W"], ["total", "P"]]);
  assert.ok(Math.abs(rows[0].prob - 0.581) < 1e-12);
  assert.equal(rows[0].finalScore, "Texas by 10 · 54.5 total");
  assert.ok(Math.abs(rows[1].units - 0.909) < 1e-12);
  assert.equal(rows[0].units, null);
  const noLines = g.trackRows([{ ...acc[0], market_spread: null, market_total: null }], []);
  assert.deepEqual(noLines.map((r) => r.market), ["moneyline"]);
});

/* ── fixtures ─────────────────────────────────────────────────────────── */
let PK = 100;
const A = (o = {}) => ({ sport: "nfl", game_pk: ++PK, game_date: "2026-10-04", home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions",
  win_prob: 0.4, predicted_winner: "Detroit Lions", actual_winner: "Detroit Lions", winner_correct: true, pred_margin: -2.2, actual_margin: -7,
  market_spread: 1.5, spread_pick_correct: false, pred_total: 52.4, actual_total: 41, market_total: 46.5, total_pick_correct: true, ...o });
// A synthetic result list: `n` rows with a given result + date + prob, markets pinned to moneyline.
const R = (result, date = "2026-10-04", o = {}) => ({ date, sport: "nfl", game_pk: ++PK, matchup: "A @ B", market: "moneyline", pick: "B", closing: null, prob: 0.6, conf: 0.6,
  result, won: result === "W" ? true : result === "L" ? false : null, finalScore: "", units: null, ...o });

test("trackRows: away favorite probability, spread/total pick sides, model numbers, closing line, conf, sort order", () => {
  const away = g.trackRows([A()], []);   // home win 0.4 -> the model favors the away Lions at 0.6
  assert.deepEqual(away.map((r) => r.market), ["moneyline", "spread", "total"]);
  const [ml, sp, tot] = away;
  assert.ok(Math.abs(ml.prob - 0.6) < 1e-12 && Math.abs(ml.conf - 0.6) < 1e-12);
  assert.equal(ml.pick, "Detroit Lions"); assert.equal(ml.side, "away");
  assert.equal(ml.matchup, "Detroit Lions @ Kansas City Chiefs");
  assert.equal(ml.date, "2026-10-04"); assert.equal(ml.sport, "nfl");
  // pred_margin -2.2 + market_spread 1.5 < 0 -> away; away closing line = -(1.5); model number on the away side = r05(pred_margin) = -2
  assert.equal(sp.pick, "Detroit Lions"); assert.equal(sp.side, "away"); assert.equal(sp.closing, -1.5); assert.equal(sp.modelLine, -2);
  assert.equal(sp.result, "L");
  // pred_total 52.4 > 46.5 -> over; closing = the market total; model number = r05(52.4) = 52.5
  assert.equal(tot.pick, "Over"); assert.equal(tot.side, "over"); assert.equal(tot.closing, 46.5); assert.equal(tot.modelLine, 52.5);
  assert.equal(tot.result, "W");
  assert.equal(sp.prob, null, "the prediction carries no cover probability: never invented"); assert.equal(tot.prob, null);
  assert.equal(sp.conf, null, "confidence is the moneyline win probability: spread / total rows carry none"); assert.equal(tot.conf, null);
  assert.equal(ml.won, true); assert.equal(sp.won, false);
  const home = g.trackRows([A({ win_prob: 0.7, predicted_winner: "Kansas City Chiefs", pred_margin: 8, market_spread: -3.5, pred_total: 40, winner_correct: false, actual_winner: "Detroit Lions" })], []);
  assert.equal(home[0].result, "L"); assert.ok(Math.abs(home[0].prob - 0.7) < 1e-12); assert.equal(home[0].side, "home");
  assert.equal(home[1].side, "home"); assert.equal(home[1].closing, -3.5); assert.equal(home[1].modelLine, -8);
  assert.equal(home[2].side, "under");
  const rows = g.trackRows([A({ game_pk: 1, game_date: "2026-10-03" }), A({ game_pk: 2, game_date: "2026-10-04" }), A({ game_pk: 3, game_date: "2026-10-04" })], []);
  assert.deepEqual(rows.map((r) => [r.date, r.game_pk, r.market]).slice(0, 4), [["2026-10-04", 3, "moneyline"], ["2026-10-04", 3, "spread"], ["2026-10-04", 3, "total"], ["2026-10-04", 2, "moneyline"]]);
  assert.equal(rows[rows.length - 1].date, "2026-10-03", "newest first");
});

test("trackRows: ungraded games are skipped; a spread / total with no model pick is not a bet (R23) and gets no row; no NaN / undefined", () => {
  assert.deepEqual(g.trackRows([A({ actual_winner: null, winner_correct: null })], []), []);
  assert.deepEqual(g.trackRows(null, null), []);
  const onLine = g.trackRows([A({ pred_margin: -1.5, market_spread: 1.5, pred_total: 46.5, market_total: 46.5 })], []);   // model exactly on both lines
  assert.deepEqual(onLine.map((r) => r.market), ["moneyline"], "no pick side: no spread / total row");
  const oneSided = g.trackRows([A({ pred_margin: -1.5, market_spread: 1.5, pred_total: 50, market_total: 46.5 })], []);
  assert.deepEqual(oneSided.map((r) => r.market), ["moneyline", "total"], "each market stands on its own");
  const sparse = g.trackRows([A({ pred_margin: null, pred_total: null, actual_margin: null, actual_total: null, win_prob: null, winner_correct: null })], []);
  assert.deepEqual(sparse.map((r) => r.market), ["moneyline"], "missing model numbers: no spread / total picks");
  assert.equal(sparse[0].prob, null); assert.equal(sparse[0].conf, null); assert.equal(sparse[0].result, "P", "an unknowable moneyline grade is a push, never a win");
  assert.ok(!/NaN|undefined|null/.test(sparse[0].finalScore)); assert.equal(sparse[0].finalScore, "Detroit Lions");
  const noLines = g.trackRows([A({ market_spread: null, market_total: null })], []);
  assert.deepEqual(noLines.map((r) => r.market), ["moneyline"]);
  assert.ok(onLine.every((r) => r.pick != null), "every spread / total row has a pick");
});

test("finalScore text: winner by margin, total, ties", () => {
  const one = (o) => g.trackRows([A(o)], [])[0].finalScore;
  assert.equal(one({ actual_margin: -7, actual_total: 41 }), "Detroit Lions by 7 · 41 total");
  assert.equal(one({ actual_margin: 0, actual_total: 40, actual_winner: "Detroit Lions" }), "Tie · 40 total");
  assert.equal(one({ actual_margin: null, actual_total: 44.5 }), "Detroit Lions · 44.5 total");
  assert.equal(one({ actual_margin: 3, actual_total: null }), "Detroit Lions by 3");
});

test("units join on game_pk|market (pnl / 10); a spread bet's pnl never lands on the moneyline row", () => {
  const rows = g.trackRows([A({ game_pk: 7 })], [{ game_pk: 7, market: "total", pnl: -10 }, { game_pk: 7, market: "moneyline", pnl: 4.7 }, { game_pk: 8, market: "spread", pnl: 99 }]);
  assert.deepEqual(rows.map((r) => r.units == null ? null : Math.round(r.units * 100) / 100), [0.47, null, -1]);
});

test("moneyline closing price: game_closing_prices decimal -> American, on the picked side", () => {
  assert.equal(g.decToAmerican(2.28), 128); assert.equal(g.decToAmerican(1.4), -250); assert.equal(g.decToAmerican(2), 100);
  assert.equal(g.decToAmerican(1), null); assert.equal(g.decToAmerican(null), null); assert.equal(g.decToAmerican("x"), null);
  const closing = g.trackClosingMap([{ game_pk: 9, side: "home", close_dec: 1.4 }, { game_pk: 9, side: "away", close_dec: 3.1 }, { game_pk: 10, side: "home", close_dec: null }]);
  assert.equal(closing.get("9|home"), -250); assert.equal(closing.get("9|away"), 210); assert.ok(!closing.has("10|home"));
  const rows = g.trackRows([A({ game_pk: 9, predicted_winner: "Kansas City Chiefs", win_prob: 0.7 })], [], closing);
  assert.equal(rows[0].closing, -250, "the moneyline row's closing line is the picked side's price");
  assert.equal(g.trackRows([A({ game_pk: 11 })], [], closing)[0].closing, null, "no captured price: null, never a guess");
});

test("range windows end today; Season starts Aug 1 of the season year; All Time is unbounded", () => {
  assert.deepEqual(g.trackRange("7d", "2026-10-04"), { from: "2026-09-28", to: "2026-10-04" });
  assert.deepEqual(g.trackRange("30d", "2026-10-04"), { from: "2026-09-05", to: "2026-10-04" });
  assert.deepEqual(g.trackRange("season", "2026-10-04"), { from: "2026-08-01", to: "2026-10-04" });
  assert.deepEqual(g.trackRange("season", "2027-02-10"), { from: "2026-08-01", to: "2027-02-10" });
  assert.deepEqual(g.trackRange("all", "2026-10-04"), { from: "0000-00-00", to: "2026-10-04" });
  assert.deepEqual(g.trackRange("bogus", "2026-10-04"), { from: "0000-00-00", to: "2026-10-04" });
});

test("scope: in-record rows by default, the pre-restart archive only with the toggle, then the range", () => {
  const rows = [R("W", "2026-10-04", { inRecord: true }), R("L", "2026-09-10", { inRecord: true }), R("W", "2026-09-20", { inRecord: false }), R("W", "2026-08-01", { inRecord: true })];
  assert.deepEqual(g.trackScope(rows, { range: "all", archive: false }, "2026-10-04").map((r) => r.date), ["2026-10-04", "2026-09-10", "2026-08-01"]);
  assert.deepEqual(g.trackScope(rows, { range: "all", archive: true }, "2026-10-04").map((r) => r.date), ["2026-09-20"]);
  assert.deepEqual(g.trackScope(rows, { range: "7d", archive: false }, "2026-10-04").map((r) => r.date), ["2026-10-04"]);
  assert.deepEqual(g.trackScope(rows, { range: "30d", archive: false }, "2026-10-04").map((r) => r.date), ["2026-10-04", "2026-09-10"]);
  assert.deepEqual(g.trackScope(null, { range: "all" }, "2026-10-04"), []);
});

test("tally + accuracy vs the closing line (spread / total only) and +X vs the 52.4% breakeven", () => {
  const rows = [R("W"), R("W"), R("L"), R("P"), R("W", "2026-10-04", { market: "spread" }), R("L", "2026-10-04", { market: "spread" }), R("W", "2026-10-04", { market: "total" }), R("P", "2026-10-04", { market: "total" })];
  const t = g.trackTally(rows);
  assert.deepEqual([t.w, t.l, t.p, t.n], [4, 2, 2, 8]);
  assert.ok(Math.abs(t.pct - 4 / 6) < 1e-12, "win % = W / (W + L); pushes are not decided");
  assert.equal(g.trackTally([]).pct, null); assert.equal(g.trackTally(null).n, 0);
  const v = g.trackAccuracyVsLine(rows);
  assert.equal(v.n, 3); assert.ok(Math.abs(v.share - 2 / 3) < 1e-12); assert.ok(Math.abs(v.vsMarket - (2 / 3 - 0.524) * 100) < 1e-9);
  assert.equal(g.trackAccuracyVsLine([R("W")]), null, "moneyline picks have no closing-line grade");
  assert.equal(g.trackAccuracyVsLine([]), null);
});

test("table filters: league, market, confidence band, result, search; sort recent / confidence", () => {
  const rows = [R("W", "2026-10-04", { sport: "nfl", market: "moneyline", conf: 0.72, matchup: "Detroit Lions @ Kansas City Chiefs", pick: "Detroit Lions" }),
    R("L", "2026-10-03", { sport: "cfb", market: "moneyline", conf: 0.52, matchup: "Oklahoma @ Texas", pick: "Texas" }),
    R("P", "2026-10-02", { sport: "cfb", market: "total", conf: null, matchup: "Ohio State @ Oregon", pick: "Over" }),
    R("W", "2026-10-01", { sport: "cfb", market: "spread", conf: null, matchup: "Boise @ Fresno", pick: "Boise" }),
    R("W", "2026-10-05", { sport: "mlb", market: "moneyline", conf: null, matchup: "Yankees @ Blue Jays", pick: "Yankees" })];
  const f = (o) => g.trackFilter(rows, { market: "all", league: "", conf: "", result: "", q: "", sort: "recent", ...o }).map((r) => r.pick);
  assert.deepEqual(f({}), ["Yankees", "Detroit Lions", "Texas", "Over", "Boise"]);
  assert.deepEqual(f({ league: "cfb" }), ["Texas", "Over", "Boise"]);
  assert.deepEqual(f({ market: "moneyline" }), ["Yankees", "Detroit Lions", "Texas"]);
  assert.deepEqual(f({ market: "total" }), ["Over"]);
  assert.deepEqual(f({ conf: "70+" }), ["Detroit Lions"]);
  assert.deepEqual(f({ conf: "50-55" }), ["Texas"]);
  assert.deepEqual(f({ conf: "60-65" }), [], "the Confidence filter is moneyline-only: spread / total rows have no confidence");
  assert.deepEqual(f({ result: "L" }), ["Texas"]);
  assert.deepEqual(f({ result: "P" }), ["Over"]);
  assert.deepEqual(f({ q: "  OHIO " }), ["Over"]);
  assert.deepEqual(f({ q: "lions" }), ["Detroit Lions"]);
  assert.deepEqual(f({ sort: "conf" }), ["Detroit Lions", "Texas", "Yankees", "Over", "Boise"], "highest confidence first; unknown last (then most recent first)");
  assert.equal(g.trackFilter(null, {}).length, 0);
});

test("segments: win % over decided picks by market / league / confidence, best first", () => {
  const rows = [R("W", "2026-10-04", { market: "moneyline", sport: "nfl", conf: 0.72 }), R("W", "2026-10-04", { market: "moneyline", sport: "nfl", conf: 0.72 }), R("L", "2026-10-04", { market: "moneyline", sport: "nfl", conf: 0.72 }),
    R("W", "2026-10-04", { market: "spread", sport: "cfb", conf: null }), R("P", "2026-10-04", { market: "spread", sport: "cfb", conf: null }),
    R("L", "2026-10-04", { market: "total", sport: "cfb", conf: null }), R("L", "2026-10-04", { market: "total", sport: "cfb", conf: null }),
    R("W", "2026-10-04", { market: "moneyline", sport: "cfb", conf: 0.52 }), R("L", "2026-10-04", { market: "moneyline", sport: "cfb", conf: 0.56 })];
  const m = g.trackSegments(rows, "market");
  assert.deepEqual(m.map((s) => [s.label, s.w, s.l, s.p]), [["Spreads", 1, 0, 1], ["Moneyline", 3, 2, 0], ["Totals", 0, 2, 0]]);
  assert.ok(Math.abs(m[1].pct - 3 / 5) < 1e-12); assert.equal(m[0].n, 2);
  const lg = g.trackSegments(rows, "league");
  assert.deepEqual(lg.map((s) => [s.label, s.w, s.l, s.p]), [["NFL", 2, 1, 0], ["CFB", 2, 3, 1]]);
  const cf = g.trackSegments(rows, "confidence");
  assert.deepEqual(cf.map((s) => [s.label, s.w, s.l, s.p]), [["50–55%", 1, 0, 0], ["70%+", 2, 1, 0], ["55–60%", 0, 1, 0]], "moneyline picks only (spread / total rows have no confidence); best win % first");
  assert.deepEqual(g.trackSegments([], "market"), []);
  assert.deepEqual(g.trackSegments([R("P")], "market"), [], "a segment with no decided pick has no win %");
});

test("calibration display: bands 50-55 ... 70+, predicted vs actual bars, n under each, empty bands dropped", () => {
  const ml = (p, won) => R(won ? "W" : "L", "2026-10-04", { prob: p, conf: p });
  const rows = [ml(0.52, true), ml(0.54, false), ml(0.56, true), ml(0.58, true), ml(0.72, true), ml(0.9, false), R("W", "2026-10-04", { market: "spread", prob: null })];
  const c = g.trackCalibration(rows);
  assert.deepEqual(c.bands.map((b) => b.n), [2, 2, 0, 0, 2]);
  assert.deepEqual(c.groups.map((x) => [x.label, x.sub]), [["50–55%", "(2)"], ["55–60%", "(2)"], ["70%+", "(2)"]]);
  const b0 = c.groups[0].bars;
  assert.ok(Math.abs(b0[0].value - 53) < 1e-9 && b0[0].label === "53%", "predicted mean 53%");
  assert.ok(Math.abs(b0[1].value - 50) < 1e-9 && b0[1].label === "50%", "actual hit rate 1 of 2");
  assert.equal(c.groups[1].bars[1].value, 100); assert.equal(c.groups[1].bars[1].label, "100%");
  assert.ok(Math.abs(c.groups[2].bars[0].value - 81) < 1e-9);
  assert.deepEqual(g.trackCalibration([]).groups, []);
  const html = g.groupedBars(c.groups, { h: 190 });
  assert.match(html, /\(2\)/); assert.match(html, /70%\+/);
});

test("streak card: current streak, last-5 dots (pushes grey), W-L-P counts", () => {
  const rows = [R("W"), R("W"), R("P"), R("L"), R("W"), R("W")];
  const s = g.trackStreak(rows);
  assert.deepEqual(s.streak, { kind: "W", n: 2 });   // the push neither extends nor breaks the run
  assert.deepEqual(s.dots, ["W", "W", "P", "L", "W"]);
  assert.deepEqual([s.w, s.l, s.p], [3, 1, 1]);
  assert.equal(g.trackStreak([R("P")]).streak, null);
  assert.deepEqual(g.trackStreak([]).dots, []);
  assert.equal(g.trackStreak(null).streak, null);
  const dots = g.trackDots(["W", "L", "P"]);
  assert.equal((dots.match(/class="ca-trk-dot /g) || []).length, 3); assert.match(dots, /ca-trk-dot W/); assert.match(dots, /ca-trk-dot L/); assert.match(dots, /ca-trk-dot P/);
});

test("recent form: Last 10 / 25 / 50 / 100 over the newest picks", () => {
  const rows = [...Array(8).fill(0).map(() => R("W")), R("L"), R("L"), ...Array(20).fill(0).map(() => R("L"))];
  const f = g.trackForms(rows);
  assert.deepEqual(f.map((x) => x.label), ["Last 10", "Last 25", "Last 50", "Last 100"]);
  assert.deepEqual([f[0].w, f[0].l, f[0].p, f[0].pct], [8, 2, 0, 0.8]);
  assert.deepEqual([f[1].w, f[1].l, f[1].n], [8, 17, 25]);
  assert.deepEqual([f[3].w, f[3].l, f[3].n], [8, 22, 30], "fewer picks than the window: the whole list");
  assert.equal(g.trackForms([]).every((x) => x.pct === null && x.n === 0), true);
});

test("model calibration card: Brier, predicted avg, actual hit rate, difference, Well Calibrated within 3 points", () => {
  const ml = (p, won) => R(won ? "W" : "L", "2026-10-04", { prob: p, conf: p });
  const rows = [ml(0.6, true), ml(0.6, true), ml(0.6, false), ml(0.6, true), ml(0.6, false), R("W", "2026-10-04", { market: "spread", prob: null })];
  const m = g.trackModelCalibration(rows);
  assert.equal(m.n, 5);
  assert.ok(Math.abs(m.predicted - 0.6) < 1e-12); assert.ok(Math.abs(m.actual - 0.6) < 1e-12); assert.ok(Math.abs(m.diff) < 1e-9);
  assert.ok(Math.abs(m.brier - (3 * 0.16 + 2 * 0.36) / 5) < 1e-12);
  assert.equal(m.chip, "Well Calibrated");
  const over = g.trackModelCalibration([ml(0.8, true), ml(0.8, false), ml(0.8, false), ml(0.8, false)]);   // predicted 80, actual 25
  assert.ok(Math.abs(over.diff - -55) < 1e-9); assert.equal(over.chip, "Overconfident");
  const under = g.trackModelCalibration([ml(0.55, true), ml(0.55, true), ml(0.55, true), ml(0.55, true)]);
  assert.equal(under.chip, "Underconfident");
  const edge = g.trackModelCalibration([ml(0.6, true), ml(0.6, false)]);   // actual 50 vs predicted 60: -10
  assert.equal(edge.chip, "Overconfident");
  assert.equal(g.trackModelCalibration([R("W", "2026-10-04", { market: "total", prob: null })]), null);
  assert.equal(g.trackModelCalibration([]), null);
  const ok3 = g.trackModelCalibration([...Array(53).fill(0).map(() => ml(0.5, true)), ...Array(47).fill(0).map(() => ml(0.5, false))]);   // +3.0 pts exactly
  assert.equal(ok3.chip, "Well Calibrated", "|diff| <= 3 inclusive");
});

test("cumulative performance: running net wins per calendar day from a zero start, per league", () => {
  const rows = [R("W", "2026-10-02"), R("L", "2026-10-02"), R("W", "2026-10-02"), R("L", "2026-10-04", { sport: "cfb" }), R("P", "2026-10-04"), R("W", "2026-10-04")];
  const all = g.trackCumulative(rows, "overall");
  assert.deepEqual(all.map((p) => [p.date, p.net]), [["2026-10-01", 0], ["2026-10-02", 1], ["2026-10-03", 1], ["2026-10-04", 1]]);
  assert.deepEqual(g.trackCumulative(rows, "cfb").map((p) => [p.date, p.net]), [["2026-10-03", 0], ["2026-10-04", -1]]);
  assert.deepEqual(g.trackCumulative(rows, "mlb"), []);
  assert.deepEqual(g.trackCumulative([], "overall"), []);
});

test("weekly buckets (Mon-Sun): record and win % per week, oldest first", () => {
  const rows = [R("W", "2026-10-04"), R("L", "2026-10-05"), R("W", "2026-10-05"), R("W", "2026-09-28"), R("L", "2026-09-27")];   // Oct 4 = Sun, Oct 5 = Mon
  const w = g.trackWeekly(rows);
  assert.deepEqual(w.map((x) => [x.week, x.w, x.l, x.n]), [["2026-09-21", 0, 1, 1], ["2026-09-28", 2, 0, 2], ["2026-10-05", 1, 1, 2]]);
  assert.equal(w[0].pct, 0); assert.equal(w[1].pct, 1);
  assert.deepEqual(g.trackWeekly([]), []);
  const gap = g.trackWeekly([R("W", "2026-09-14"), R("L", "2026-09-30")]);
  assert.deepEqual(gap.map((x) => [x.week, x.n]), [["2026-09-14", 1], ["2026-09-21", 0], ["2026-09-28", 1]], "empty weeks stay in the series so mini bars never shift");
  assert.equal(gap[1].pct, null);
});

test("URL state: parse, defaults, only non-defaults serialize, junk falls back", () => {
  const d = g.trackParse("");
  assert.deepEqual({ ...d }, { range: "all", archive: false, view: "model", chart: "overall", market: "all", league: "", conf: "", result: "", sort: "recent", q: "", seg: "market", n: 25 });
  assert.equal(g.trackQuery(d), "");
  const s = g.trackParse("?range=7d&archive=1&view=ev&chart=cfb&market=spread&league=nfl&conf=60-65&result=W&sort=conf&q=lions&seg=league&n=75");
  assert.deepEqual({ ...s }, { range: "7d", archive: true, view: "ev", chart: "cfb", market: "spread", league: "nfl", conf: "60-65", result: "W", sort: "conf", q: "lions", seg: "league", n: 75 });
  assert.equal(new URLSearchParams(g.trackQuery(s)).get("range"), "7d");
  assert.equal(g.trackParse(g.trackQuery(s)).q, "lions", "round-trips");
  const junk = g.trackParse("?range=zzz&view=x&chart=zz&market=zz&league=zz&conf=9&result=Q&sort=x&seg=x&n=-5&archive=0");
  assert.deepEqual({ ...junk }, { ...d });
  assert.equal(g.trackParse("?n=99999").n, 1000, "n is capped"); assert.equal(g.trackParse("?n=12").n, 25, "n is at least one page");
});

/* ── render ───────────────────────────────────────────────────────────── */
const EVIL_HOME = `Texas <b>Crew</b>`, EVIL_AWAY = `Oklahoma "Sooners" <i>Zed</i>`;
const dayOff = (n) => new Date(Date.now() + n * 864e5).toISOString().slice(0, 10);
const urlParts = (u) => { const [p, q = ""] = String(u).replace(/^.*\/rest\/v1\//, "").split("?"); return { path: p, q }; };
function populated({ search = "", acc, closing, starts, fail = [], quiet = null, document } = {}) {
  const accRows = acc || [
    A({ sport: "cfb", game_pk: 1, game_date: dayOff(-1), home_team_name: EVIL_HOME, away_team_name: EVIL_AWAY, predicted_winner: EVIL_HOME, actual_winner: EVIL_HOME, win_prob: 0.62, winner_correct: true, pred_margin: 6, actual_margin: 10, market_spread: -3.5, spread_pick_correct: true, pred_total: 55, market_total: 54.5, actual_total: 60, total_pick_correct: true }),
    A({ sport: "cfb", game_pk: 2, game_date: dayOff(-2), home_team_name: "Ohio State", away_team_name: "Oregon", predicted_winner: "Oregon", actual_winner: "Ohio State", win_prob: 0.45, winner_correct: false, pred_margin: -2, actual_margin: 3, market_spread: -1, spread_pick_correct: false, pred_total: 50, market_total: 50.5, actual_total: 45, total_pick_correct: true }),
    A({ sport: "nfl", game_pk: 3, game_date: dayOff(-3), home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions", predicted_winner: "Detroit Lions", actual_winner: "Detroit Lions", win_prob: 0.4, winner_correct: true, pred_margin: -2, actual_margin: -7, market_spread: 1.5, spread_pick_correct: null, pred_total: 50, market_total: 46.5, actual_total: 41, total_pick_correct: false }),
    A({ sport: "nfl", game_pk: 4, game_date: dayOff(-9), home_team_name: "Buffalo Bills", away_team_name: "Miami Dolphins", predicted_winner: "Buffalo Bills", actual_winner: "Buffalo Bills", win_prob: 0.75, winner_correct: true, pred_margin: 7, actual_margin: 14, market_spread: -6.5, spread_pick_correct: true, pred_total: 48, market_total: 47.5, actual_total: 50, total_pick_correct: false }),
  ];
  const requested = [];
  const fetch = async (url) => {
    const { path, q } = urlParts(url); requested.push(path + (q ? `?${q}` : ""));
    if (fail.includes(path)) return { ok: false, status: 500, text: async () => "boom", json: async () => ({}) };
    let rows = [];
    if (path === "prediction_accuracy") rows = accRows;
    else if (path === "game_closing_prices") rows = closing || [{ game_pk: 1, side: "home", close_dec: 1.6 }];
    else if (path === "track_record_start") rows = starts || [{ sport: "nfl", starts_at: dayOff(-5) + "T12:00:00+00:00", model_version: "nfl-sim-ml-v2" }];
    else if (path === "prediction_pnl_daily") rows = [{ game_date: dayOff(-1), sport: "cfb", market: "spread", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 1.5 }];
    return { ok: true, json: async () => rows };
  };
  const T = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"],
    { page: "track", globals: { fetch, ...(quiet ? { console: quiet } : {}), ...(document ? { document } : {}), location: { search, href: `http://localhost/track-record.html${search}` } } });
  return { T, requested };
}
const clean = (html) => html.replace(/\bnull\b(?=[^<]*>)/g, "");

test("buildTrackPage: title, range pills, five stat cards, every chart card, filters, table and right column; names escaped; no NaN / undefined", async () => {
  const { T, requested } = populated();
  const html = await T.buildTrackPage();
  for (const t of ["Track Record", "Prediction performance across leagues and markets.", "7D", "30D", "Season", "All Time", "Model picks", "+EV picks", "NFL archive (pre-",
    "OVERALL RECORD", "ACCURACY VS CLOSING LINE", "TOTAL PREDICTIONS", "CURRENT STREAK", "AVERAGE CONFIDENCE", "vs market", "Across all leagues and markets",
    "Cumulative Prediction Performance", "Overall", "Performance by Market", "Win %", "Loss %", "Push %", "Performance by League", "Confidence Calibration", "Predicted Confidence", "Actual Hit Rate",
    "Recent Form", "Last 10", "Last 25", "Last 50", "Last 100", "All Predictions", "Moneyline", "Spread", "Totals", "All Leagues", "All Markets", "All Confidence", "All Results", "Most Recent", "Highest Confidence",
    "Search teams, players, or games", "Matchup / Player", "Model Prediction", "Closing Line", "Model Prob.", "Result", "Confidence", "Final Score / Outcome", "View",
    "Best Performing Segments", "Model Calibration", "Brier Score", "Predicted Avg. Confidence", "Calibration Difference"]) assert.ok(html.includes(t), `missing ${t}`);
  assert.ok(html.includes("Texas &lt;b&gt;Crew&lt;/b&gt;") && html.includes("Oklahoma &quot;Sooners&quot; &lt;i&gt;Zed&lt;/i&gt;"), "hostile names are escaped");
  assert.ok(!html.includes("<b>Crew</b>") && !html.includes("<i>Zed</i>"));
  assert.ok(!/NaN|undefined/.test(clean(html)), "no NaN / undefined leaks");
  assert.ok(!/Aug 1, 2023|Oct 13, 2025|\b184\b|\b327\b/.test(html.replace(/<[^>]*>/g, " ")), "the mockup's placeholder numbers are never rendered");
  assert.ok(!requested.some((r) => r.startsWith("prediction_pnl?") || r.startsWith("prediction_pnl&")), "prediction_pnl times out under the anon 3s limit: never requested");
  assert.ok(requested.some((r) => r.startsWith("game_closing_prices?market=eq.moneyline&game_pk=in.(1,2,3,4)")), "closing prices are fetched per chunk of the graded game ids");
  assert.ok(!/class="chart"|chart-tooltip/.test(html), "no legacy dark-tooltip chart");
});

test("buildTrackPage: only in-record rows count; the pre-restart NFL game (-9d) is excluded, shown only with ?archive=1", async () => {
  const main = await populated().T.buildTrackPage();
  assert.ok(!main.includes("Buffalo Bills") && !main.includes("Dolphins"), "archived NFL game is not in the published record");
  assert.ok(main.includes("Lions @ Chiefs"));
  assert.ok(!main.includes("archived pre-restart"), "no archive banner in the record view");
  const arc = await populated({ search: "?archive=1" }).T.buildTrackPage();
  assert.ok(arc.includes("Dolphins @ Bills"), "archive mode lists the archived game");
  assert.ok(!arc.includes("Sooners") && !arc.includes("Lions @ Chiefs"), "archive mode shows only archived rows");
  assert.ok(arc.includes("archived pre-restart"), "archive banner");
});

test("buildTrackPage: URL range / filters narrow the table; cards follow the range; unknown filter values fall back", async () => {
  const win = await populated({ search: "?range=7d" }).T.buildTrackPage();
  assert.ok(win.includes("Oklahoma") && win.includes("Ohio State"));
  const one = await populated({ search: "?league=cfb&market=spread&result=W" }).T.buildTrackPage();
  const rowsHtml = one.slice(one.indexOf("<tbody>"), one.indexOf("</tbody>"));
  assert.equal((rowsHtml.match(/<tr /g) || []).length, 1, "one cfb spread win");
  assert.ok(rowsHtml.includes("Texas &lt;b&gt;Crew"));
  const none = await populated({ search: "?q=zzzzzz" }).T.buildTrackPage();
  assert.match(none, /No predictions match these filters/);
  assert.ok(!/NaN|undefined/.test(clean(none)));
});

test("table rows: shell-owned row click (data-href to the game page), no star, W / L / Push chips, em-dash for unknown probability", async () => {
  const html = await populated().T.buildTrackPage();
  const tbody = html.slice(html.indexOf("<tbody>"), html.indexOf("</tbody>"));
  assert.match(tbody, /<tr data-href="game\.html\?sport=cfb&game=1"/);
  assert.ok(!/data-star|ca-star/.test(tbody), "pages own no star logic");
  assert.match(tbody, /ca-trk-chip W[^>]*>W</); assert.match(tbody, /ca-trk-chip L[^>]*>L</); assert.match(tbody, /ca-trk-chip P[^>]*>Push</);
  assert.ok(tbody.includes("Texas &lt;b&gt;Crew&lt;/b&gt; by 10 · 60 total"), "final score text from the brief");
  assert.ok(tbody.includes("-167"), "moneyline closing price from game_closing_prices (1.6 decimal = -167)");
  assert.match(tbody, /class="ca-go"/);
});

test("closing prices are fetched in parallel chunks of 60 game ids; a failed chunk only leaves its games unpriced", async () => {
  const many = Array.from({ length: 130 }, (_, i) => A({ sport: "cfb", game_pk: 1000 + i, game_date: dayOff(-1) }));
  const reqs = [];
  const T = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"], { page: "track", globals: {
    console: { ...console, error() {}, warn() {} }, location: { search: "?n=1000", href: "http://localhost/track-record.html?n=1000" },
    fetch: async (url) => {
      const { path, q } = urlParts(url); reqs.push(path + "?" + q);
      if (path === "prediction_accuracy") return { ok: true, json: async () => many };
      if (path === "game_closing_prices") {
        if (q.includes("game_pk=in.(1060,")) return { ok: false, status: 500, text: async () => "timeout", json: async () => ({}) };
        return { ok: true, json: async () => [{ game_pk: 1000, side: "away", close_dec: 2 }, { game_pk: 1120, side: "away", close_dec: 3 }] };
      }
      return { ok: true, json: async () => [] };
    } } });
  const rows = await T.trackClosing(many.map((r) => r.game_pk));
  assert.equal(reqs.filter((r) => r.startsWith("game_closing_prices")).length, 3, "130 games -> 60 + 60 + 10");
  assert.equal(rows.length, 4, "the failed chunk contributes nothing, the others still do");
  const html = await T.buildTrackPage();
  assert.ok(html.includes("+100") && html.includes("+200"), "priced games show their closing price");
  assert.ok(!/NaN|undefined/.test(clean(html)));
});

test("pagination: 25 rows at a time with Show more; the shown count lives in the URL state", async () => {
  const many = Array.from({ length: 20 }, (_, i) => A({ sport: "cfb", game_pk: 500 + i, game_date: dayOff(-1 - (i % 5)) }));   // 60 rows
  const html = await populated({ acc: many }).T.buildTrackPage();
  const tbody = html.slice(html.indexOf("<tbody>"), html.indexOf("</tbody>"));
  assert.equal((tbody.match(/<tr /g) || []).length, 25);
  assert.match(html, /data-trk-more[^>]*>Show more/);
  assert.match(html, /Showing 25 of 60/);
  const more = await populated({ acc: many, search: "?n=50" }).T.buildTrackPage();
  assert.equal((more.slice(more.indexOf("<tbody>"), more.indexOf("</tbody>")).match(/<tr /g) || []).length, 50);
  const all = await populated({ acc: many, search: "?n=75" }).T.buildTrackPage();
  assert.ok(!/data-trk-more/.test(all), "everything shown: no button");
});

test("empty and failing sources: every card still renders, page never throws, no NaN / undefined", async () => {
  const { T } = populated({ acc: [] });
  const html = await T.buildTrackPage();
  assert.ok(html.includes("Track Record") && html.includes("Cumulative Prediction Performance") && html.includes("Best Performing Segments"));
  assert.ok(!/NaN|undefined/.test(clean(html)));
  const quiet = { ...console, error() {}, warn() {} };
  const E = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"],
    { page: "track", globals: { console: quiet, fetch: async () => { throw new Error("down"); } } });
  const down = await E.buildTrackPage();
  assert.ok(down.includes("Track Record") && !/NaN|undefined/.test(clean(down)));
});

test("every card is isolated: a card handed unusable data renders the placeholder and logs, never throws", () => {
  const errors = [];
  const E = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"],
    { page: "track", globals: { console: { ...console, error: (...a) => errors.push(a.join(" ")) } } });
  const ids = Object.keys(E.TRACK_CARDS);
  assert.ok(ids.length >= 12, "stat cards, charts, filters, table and right column are all registered cards");
  for (const id of ids) {
    const html = E.trackCard(id, null);
    assert.ok(html.includes("This panel couldn't load.") && html.includes(`id="${id}"`), `${id} isolated`);
  }
  assert.equal(errors.length, ids.length);
});

test("+EV view: URL view=ev renders the legacy profit tracker and +EV sections instead of the model cards, without throwing on empty sources", async () => {
  const { T, requested } = populated({ search: "?view=ev" });
  const html = await T.buildTrackPage();
  assert.ok(html.includes("Model picks") && html.includes("+EV picks"));
  assert.ok(html.includes("+EV profit tracker") && html.includes("Graded parlays") && html.includes("+EV pick performance"));
  assert.ok(!html.includes("OVERALL RECORD") && !html.includes("Cumulative Prediction Performance"));
  assert.ok(requested.some((r) => r.startsWith("ev_pnl_daily")) && requested.some((r) => r.startsWith("ev_results")));
  assert.ok(!requested.some((r) => r.startsWith("prediction_accuracy")), "the model view's data is not fetched in the +EV view");
  assert.ok(!/NaN|undefined/.test(clean(html)));
});

test("model view keeps the model-picks profit tracker (prediction_pnl_daily) below the table", async () => {
  const html = await populated().T.buildTrackPage();
  assert.ok(html.includes("Profit tracker") && html.includes("ca-trk-legacy"));
});

test("state survives the 5-minute re-render: a second buildTrackPage keeps the URL / in-memory state", async () => {
  const { T } = populated({ search: "?range=30d&league=cfb&seg=league&chart=cfb" });
  const a = await T.buildTrackPage(), b = await T.buildTrackPage();
  assert.equal(a, b);
  assert.match(a, /class="ca-pill on" data-pill="trk-range" data-key="30d"/);
  assert.match(a, /class="ca-pill on" data-pill="trk-seg" data-key="league"/);
  assert.match(a, /class="ca-pill on" data-pill="trk-chart" data-key="cfb"/);
  assert.match(a, /<option value="cfb" selected>/);
});

/* ── fix round 1 ──────────────────────────────────────────────────────── */
const loud = () => { const errors = []; return { errors, console: { ...console, error: (...a) => errors.push(a.map(String).join(" ")), warn() {} } }; };

test("fail closed: a failed track_record_start renders a notice and NO record numbers (archived rows never enter the record)", async () => {
  const q = loud();
  const { T } = populated({ fail: ["track_record_start"], quiet: q.console });
  const html = await T.buildTrackPage();
  assert.ok(html.includes("Couldn&#39;t load the record scope") || html.includes("Couldn't load the record scope — try again."), "scope notice");
  assert.match(html, /id="trk-error"/); assert.match(html, /data-trk-retry/);
  assert.ok(!html.includes("OVERALL RECORD") && !html.includes("Lions @ Chiefs") && !html.includes("Dolphins @ Bills") && !html.includes("<tbody>"), "no record numbers, no rows (archived NFL rows would be among them)");
  assert.ok(q.errors.some((e) => e.includes("track_record_start")), "console.error");
  assert.ok(!/NaN|undefined/.test(clean(html)));
  const arc = await populated({ fail: ["track_record_start"], quiet: q.console, search: "?archive=1" }).T.buildTrackPage();
  assert.ok(arc.includes("Couldn't load the record scope") && !arc.includes("<tbody>"), "the archive view fails closed too");
});

test("a failed prediction_accuracy shows 'Couldn't load graded picks.' (not the empty-record message) and logs", async () => {
  const q = loud();
  const html = await populated({ fail: ["prediction_accuracy"], quiet: q.console }).T.buildTrackPage();
  assert.ok(html.includes("Couldn't load graded picks."));
  assert.ok(!html.includes("No graded picks in this range yet") && !html.includes("OVERALL RECORD"));
  assert.ok(q.errors.some((e) => e.includes("prediction_accuracy")));
  const both = await populated({ fail: ["prediction_accuracy", "track_record_start"], quiet: q.console }).T.buildTrackPage();
  assert.ok(both.includes("Couldn't load the record scope"), "scope failure wins");
  const ok = await populated({ quiet: q.console }).T.buildTrackPage();
  assert.ok(!ok.includes("Couldn't load"), "healthy page has no notice");
});

test("+EV view fails closed on a failed track_record_start too", async () => {
  const q = loud();
  const html = await populated({ search: "?view=ev", fail: ["track_record_start"], quiet: q.console }).T.buildTrackPage();
  assert.ok(html.includes("Couldn't load the record scope") && !html.includes("+EV profit tracker") && !html.includes("+EV pick performance"));
});

test("the 1,000-row cap on prediction_accuracy is announced on the page", async () => {
  const many = Array.from({ length: 1000 }, (_, i) => A({ sport: "cfb", game_pk: 5000 + i, game_date: dayOff(-1) }));
  const q = loud();
  const html = await populated({ acc: many, quiet: q.console }).T.buildTrackPage();
  assert.ok(html.includes("Showing the most recent 1,000 graded games."));
  assert.ok(!(await populated().T.buildTrackPage()).includes("most recent 1,000"));
});

test("mini bars keep empty weeks as zero-height slots (the bars do not shift)", async () => {
  const acc = [A({ sport: "cfb", game_pk: 1, game_date: dayOff(-1) }), A({ sport: "cfb", game_pk: 2, game_date: dayOff(-22) })];
  const html = await populated({ acc, starts: [] }).T.buildTrackPage();
  const total = html.slice(html.indexOf("TOTAL PREDICTIONS"), html.indexOf("CURRENT STREAK"));
  assert.ok((total.match(/<rect /g) || []).length >= 4, "three-plus weeks apart: the gap weeks are bars too");
});

// A tiny document: just enough for wireTrackPage's delegated handlers, with every redraw recorded.
function fakeTrackDoc() {
  const handlers = {}, root = { innerHTML: "", addEventListener: (t, f) => { handlers[t] = f; }, querySelector: () => ({}) }, els = {};
  const doc = { body: { dataset: { page: "track" }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {}, createElement: () => ({}),
    getElementById: (id) => (id === "trk-root" ? root : (els[id] = els[id] || { id, outerHTML: "" })), querySelector: () => null, querySelectorAll: () => [], addEventListener() {} };
  return { doc, handlers, root, els };
}
const hit = (map) => ({ target: { closest: (sel) => map[sel] || null, matches: (sel) => !!map[`matches:${sel}`], value: map.value } });

test("interaction: range, archive, filters, Show more and the URL (non-default values only)", async () => {
  const D = fakeTrackDoc(), urls = [];
  const many = Array.from({ length: 20 }, (_, i) => A({ sport: "cfb", game_pk: 700 + i, game_date: dayOff(-1 - (i % 5)) }));
  const T2 = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"], { page: "track", globals: {
    document: D.doc, location: { search: "", href: "http://localhost/track-record.html" }, history: { replaceState: (a, b, u) => urls.push(u) },
    fetch: async (url) => { const { path } = urlParts(url); return { ok: true, json: async () => (path === "prediction_accuracy" ? many : path === "track_record_start" ? [] : []) }; } } });
  const html = await T2.buildTrackPage();
  assert.ok(html.includes('id="trk-root"'));
  T2.wireTrackPage();
  const last = () => new URL(urls[urls.length - 1]);
  // range pill: state + URL + full redraw from the cached load
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "trk-range", key: "7d" } } }));
  assert.equal(last().searchParams.get("range"), "7d"); assert.equal([...last().searchParams.keys()].join(), "range", "non-default values only");
  assert.match(D.root.innerHTML, /class="ca-pill on" data-pill="trk-range" data-key="7d"/);
  // the Record select: archive (redraw from the cached load), back to model picks, +EV picks (refetch = render())
  const record = (value) => D.handlers.change({ target: { closest: (sel) => (sel === "select[data-trk-record]" ? { value } : null) } });
  record("archive");
  assert.equal(last().searchParams.get("archive"), "1"); assert.deepEqual([...last().searchParams.keys()].sort(), ["archive", "range"]);
  assert.match(D.root.innerHTML, /archived pre-restart/);
  D.handlers.click(hit({ "[data-trk-record-back]": {} }));
  assert.equal(last().searchParams.has("archive"), false, "back to the record removes the param");
  assert.doesNotMatch(D.root.innerHTML, /archived pre-restart/);
  record("archive"); record("model");
  assert.equal(last().searchParams.has("archive"), false);
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "trk-range", key: "all" } } }));
  assert.equal(last().search, "", "everything back to default: a clean URL");
  // a select change redraws the filters + table cards only and resets the page size
  D.handlers.change({ target: { closest: (sel) => (sel === "select[data-trk]" ? { dataset: { trk: "league" }, value: "cfb" } : null) } });
  assert.equal(last().searchParams.get("league"), "cfb");
  assert.ok(D.els["trk-filters"].outerHTML.includes('<option value="cfb" selected>') && D.els["trk-table"].outerHTML.includes("<tbody>"));
  // Show more
  D.handlers.click(hit({ "[data-trk-more]": {} }));
  assert.equal(last().searchParams.get("n"), "50"); assert.equal((D.els["trk-table"].outerHTML.match(/<tr data-href/g) || []).length, 50);
  D.handlers.click(hit({ "[data-trk-more]": {} }));
  assert.equal(last().searchParams.get("n"), "75");
  assert.ok(!D.els["trk-table"].outerHTML.includes("data-trk-more"), "all 60 shown");
  // a filter change goes back to the first page
  D.handlers.change({ target: { closest: (sel) => (sel === "select[data-trk]" ? { dataset: { trk: "result" }, value: "W" } : null) } });
  assert.equal(last().searchParams.has("n"), false); assert.equal(last().searchParams.get("result"), "W");
  // search typing redraws the table only
  const before = D.els["trk-filters"].outerHTML;
  D.handlers.input({ target: { matches: (sel) => sel === "[data-trk-q]", value: "detroit" } });
  assert.equal(last().searchParams.get("q"), "detroit"); assert.equal(D.els["trk-filters"].outerHTML, before, "the search box is not re-rendered while typing");
  // pills + reset
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "trk-mkt", key: "spread" } } }));
  assert.equal(last().searchParams.get("market"), "spread");
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "trk-chart", key: "cfb" } } }));
  assert.equal(last().searchParams.get("chart"), "cfb"); assert.ok(D.els["trk-cum"].outerHTML.includes("Cumulative"));
  D.handlers.click(hit({ "[data-trk-reset]": {} }));
  assert.deepEqual([...last().searchParams.keys()], ["chart"], "reset clears the filters only");
});

test("CSS contract: every ca-* class the +EV and Track Record filter bars emit exists in theme.css (no dangling renames)", async () => {
  const fs = await import("node:fs");
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  const classesIn = (html) => [...new Set([...html.matchAll(/class="([^"]*)"/g)].flatMap((m) => m[1].split(/\s+/)).filter((c) => /^ca-/.test(c)))];
  const hooks = new Set(["ca-minibars", "ca-spark", "ca-donut", "ca-bar"]);   // SVG hook classes from ui.js, deliberately unstyled
  const defined = (c, strict) => new RegExp(`${strict ? "(^|\\n|,)" : ""}\\.${c}(?![\\w-])`).test(css);
  // strict: a top-level rule (a class that is only in a media-query override is a dangling rename); loose: anywhere in the CSS
  const missing = (html, strict = true) => classesIn(html).filter((c) => !hooks.has(c) && !defined(c, strict));
  const trk = (await populated().T.buildTrackPage());
  const bar = trk.slice(trk.indexOf('class="ca-card ca-trk-filters"'), trk.indexOf('id="trk-table"'));
  assert.ok(classesIn(bar).length >= 6, "found the filter bar classes");
  assert.deepEqual(missing(bar), [], "track filter bar classes are all styled");
  const E = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/ev.js", "js/boot.js"], { page: "ev", globals: { fetch: async () => ({ ok: true, json: async () => [] }) } });
  const ev = await E.buildEvPage();
  const evBar = ev.slice(ev.indexOf('class="ca-card ca-ev-filters"'), ev.indexOf('class="ca-ev-main"'));
  assert.ok(classesIn(evBar).includes("ca-ev-filters") && classesIn(evBar).includes("ca-ev-fgrid"), "the +EV bar emits its own grid classes");
  assert.deepEqual(missing(evBar), [], "+EV filter bar classes are all styled");
  assert.deepEqual(missing(trk, false), [], "every ca-* class on the Track Record page appears in the CSS");
  assert.deepEqual(missing(ev, false), [], "every ca-* class on the +EV page appears in the CSS");
  assert.ok(!/\.ca-ev-f[ {.]|\.ca-ev-search|\.ca-ev-cap/.test(css), "no dead CSS for the renamed shared classes");
});

test("Overall Record: the (i) tooltip says what the record combines AND carries the NFL restart note; no caption line, no header note", async () => {
  const { T } = populated();
  const html = await T.buildTrackPage();
  const card = html.slice(html.indexOf("OVERALL RECORD") - 120, html.indexOf("ACCURACY VS CLOSING LINE"));
  assert.match(card, /<span class="ca-info" title="Combines moneyline winner picks with spread and total picks graded against the line\. NFL record restarted with the ML v2 model on [A-Z][a-z]{2} \d{1,2}, \d{4}; earlier NFL results are archived for comparison\."/);
  assert.match(card, /class="ca-stat-label"[^>]*><span class="ca-stat-lt">OVERALL RECORD<\/span><span class="ca-info"/, "the (i) sits right next to the label");
  assert.ok(!card.includes("Moneyline winners + spread / total picks vs the line"), "the caption line moved into the tooltip");
  assert.ok(!html.includes("ca-trk-note") && !/ca-trk-sub/.test(html), "the restart-note line and the toggle row are gone from the page");
  assert.ok(!/ca-trk-arch|data-trk-archive|data-pill="trk-view"/.test(html), "no Archive pill and no Model / +EV toggle");
  const none = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"], { page: "track", globals: { fetch: async (u) => ({ ok: true, json: async () => (String(u).includes("prediction_accuracy") ? [A({ sport: "cfb", game_pk: 1, game_date: dayOff(-1) })] : []) } ) } });
  assert.match(await none.buildTrackPage(), /title="Combines moneyline winner picks[^"]*NFL predictions switched to the ML model on Sep 28, 2026/, "no restart row: the fallback wording");
  assert.ok(!fs.readFileSync(new URL("../js/pages/track.js", import.meta.url), "utf8").includes("__cappingAlphaChartSeries"), "dead legacy chart global removed");
});

/* ── fidelity task 4 (T1-T7) ──────────────────────────────────────────── */
const optionsOf = (html, attr) => { const m = html.match(new RegExp(`<select class="ca-select" ${attr}="record">([\\s\\S]*?)</select>`)); return m ? [...m[1].matchAll(/<option value="([^"]*)"( selected)?>([^<]*)</g)].map((x) => [x[1], x[3], !!x[2]]) : null; };

test("T2 Record select: Model picks (default) / +EV picks / NFL archive (pre-<restart day>), labelled, in the filter row, replacing the Archive pill and the toggle", async () => {
  const html = await populated().T.buildTrackPage();
  const opts = optionsOf(html, "data-trk-record");
  assert.ok(opts, "a Record select exists");
  assert.deepEqual(opts.map((o) => o.slice(0, 2)), [["model", "Model picks"], ["ev", "+EV picks"], ["archive", `NFL archive (pre-${new Date(Date.now() - 5 * 864e5).toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric" })})`]]);
  assert.deepEqual(opts.filter((o) => o[2]).map((o) => o[0]), ["model"], "default = Model picks");
  const bar = html.slice(html.indexOf('id="trk-filters"'), html.indexOf('id="trk-search"'));
  assert.match(bar, /<label class="ca-f "><span>Record<\/span><select class="ca-select" data-trk-record="record">/, "labelled Record, inside the filter card");
  const labels = [...bar.matchAll(/<label class="ca-f[^"]*"><span>(.*?)<\/span>/g)].map((m) => m[1].replace(/<[^>]*>/g, ""));
  assert.deepEqual(labels, ["League", "Market Type", "Confidence", "Result", "Sort By", "Record"], "selects in the mockup order, labels above");
  assert.equal((html.match(/data-trk-record="record"/g) || []).length, 1);
  const arc = optionsOf(await populated({ search: "?archive=1" }).T.buildTrackPage(), "data-trk-record"), ev = optionsOf(await populated({ search: "?view=ev" }).T.buildTrackPage(), "data-trk-record");
  assert.deepEqual(arc.filter((o) => o[2]).map((o) => o[0]), ["archive"], "?archive=1 selects the archive");
  assert.deepEqual(ev.filter((o) => o[2]).map((o) => o[0]), ["ev"], "?view=ev selects +EV picks (the +EV view has no filter row: the Record select stands alone under the title)");
  assert.ok((await populated({ search: "?view=ev" }).T.buildTrackPage()).includes('id="trk-recordbar"'));
  const noStart = optionsOf(await populated({ starts: [] }).T.buildTrackPage(), "data-trk-record");
  assert.equal(noStart[2][1], "NFL archive", "no restart row: no date to quote");
  const failed = loud(), err = await populated({ fail: ["track_record_start"], quiet: failed.console }).T.buildTrackPage();
  assert.ok(optionsOf(err, "data-trk-record") && err.includes('id="trk-error"'), "a failed load still offers the Record select");
});

test("T2 Record select: switching source keeps all three data paths (model rows, the pre-restart archive, +EV sections)", async () => {
  const model = await populated().T.buildTrackPage(), arc = await populated({ search: "?archive=1" }).T.buildTrackPage(), ev = await populated({ search: "?view=ev" }).T.buildTrackPage();
  assert.ok(model.includes("Lions @ Chiefs") && !model.includes("Dolphins @ Bills"));
  assert.ok(arc.includes("Dolphins @ Bills") && !arc.includes("Lions @ Chiefs") && arc.includes("archived pre-restart"), "archive = the pre-restart NFL rows only");
  assert.ok(ev.includes("+EV profit tracker") && !ev.includes("Cumulative Prediction Performance"));
});

test("T1 header: title and subtitle inline, range pills + date range on the right, nothing else", async () => {
  const html = await populated().T.buildTrackPage();
  const row = html.slice(html.indexOf('class="ca-title-row"'), html.indexOf('class="ca-stats'));
  assert.match(row, /<div class="ca-title"><h1>Track Record<\/h1><p>Prediction performance across leagues and markets\.<\/p><\/div>/);
  const right = row.slice(row.indexOf('class="ca-title-right"'));
  assert.deepEqual([...right.matchAll(/data-key="([^"]+)"/g)].map((m) => m[1]), ["7d", "30d", "season", "all"]);
  assert.ok(right.includes("ca-trk-daterange"));
  assert.ok(!/Archive|Model picks|\+EV picks|restarted/.test(row), "no Archive pill, no toggle row, no restart note in the header area");
});

test("T3 stat cards: five, short sub-lines; the long explanations moved into (i) tooltips", async () => {
  const html = await populated().T.buildTrackPage();
  const cards = html.slice(html.indexOf('class="ca-stats ca-trk-stats"'), html.indexOf('class="ca-trk-r1"')).split('<div class="ca-card ca-stat">').slice(1);
  assert.equal(cards.length, 5);
  const sub = (c) => (c.match(/class="ca-stat-sub[^"]*"[^>]*>([\s\S]*?)<\/div>/) || [])[1];
  assert.equal(sub(cards[0]), undefined, "Overall Record: no sub-line (W / L / P sit under the numbers)");
  assert.ok(/<em>W<\/em>/.test(cards[0]) && /<em>L<\/em>/.test(cards[0]) && /<em>P<\/em>/.test(cards[0]));
  assert.match(sub(cards[1]).replace(/<[^>]*>/g, ""), /^[+-]?\d+(\.\d+)?% vs market$/, "Accuracy: just the vs-market line");
  assert.ok(/class="ca-info" title="\d+ spread and total picks/.test(cards[1]) && !/breakeven<\/span>/.test(sub(cards[1])), "pick count + breakeven moved into the tooltip");
  assert.equal(sub(cards[2]), "Across all leagues and markets");
  assert.match(sub(cards[3]), /^Last 5: \d - \d - \d$/);
  assert.match(sub(cards[4]), /^On moneyline picks$/);
});

test("T7 structure: row 2 pair, row 3 trio, row 4 = full-width filter row (filters + search) then table + rail", async () => {
  const html = await populated().T.buildTrackPage();
  const at = (s) => html.indexOf(s);
  const r1 = html.slice(at('class="ca-trk-r1"'), at('class="ca-trk-r2"')), r2 = html.slice(at('class="ca-trk-r2"'), at('class="ca-trk-filterrow"'));
  assert.deepEqual([...r1.matchAll(/<section class="ca-card ca-trk-card" id="([^"]+)"/g)].map((m) => m[1]), ["trk-cum", "trk-market"], "row 2: cumulative + by-market");
  assert.deepEqual([...r2.matchAll(/<section class="ca-card ca-trk-card" id="([^"]+)"/g)].map((m) => m[1]), ["trk-league", "trk-calib", "trk-form"], "row 3: league + calibration + form");
  const fr = html.slice(at('class="ca-trk-filterrow"'), at('class="ca-trk-main"'));
  assert.deepEqual([...fr.matchAll(/<section class="[^"]*" id="([^"]+)"/g)].map((m) => m[1]), ["trk-filters", "trk-search"], "row 4a: filter card, then the search card beside it");
  assert.ok(/class="ca-card ca-sfind" id="trk-search"/.test(fr) && fr.includes("data-trk-q"), "the shared search card");
  const main = html.slice(at('class="ca-trk-main"'), at('id="trk-profit"') > 0 ? at('id="trk-profit"') : undefined);
  assert.ok(at('id="trk-table"') > at('class="ca-trk-main"') && at('class="ca-trk-left"') < at('id="trk-table"') && at('id="trk-table"') < at('<aside class="ca-trk-rail"'));
  assert.deepEqual([...main.slice(main.indexOf("<aside")).matchAll(/<section class="ca-card ca-trk-rail-card" id="([^"]+)"/g)].map((m) => m[1]), ["trk-segments", "trk-model"], "rail: segments then calibration");
  assert.ok(at('class="ca-trk-r1"') > at('class="ca-stats ca-trk-stats"') && at('class="ca-trk-filterrow"') > at('class="ca-trk-r2"'));
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  assert.match(css, /\.ca-trk-r1\{[^}]*grid-template-columns:minmax\(0,63fr\) minmax\(0,37fr\)/, "row 2 = 63 / 37");
  assert.match(css, /\.ca-trk-r2\{[^}]*grid-template-columns:minmax\(0,32fr\) minmax\(0,45fr\) minmax\(0,23fr\)/, "row 3 = 32 / 45 / 23");
  assert.match(css, /\.ca-trk-filterrow\{[^}]*grid-template-columns:minmax\(0,77fr\) minmax\(0,23fr\)/, "filter row = 77 / 23 (search over the rail)");
  assert.match(css, /\.ca-trk-main\{[^}]*grid-template-columns:minmax\(0,77fr\) minmax\(0,23fr\)/, "table + rail = 77 / 23");
  assert.match(css, /\.ca-trk-stats \.ca-stat-label\{[^}]*text-transform:uppercase/, "T7: UPPERCASE stat labels, scoped to this page");
  assert.match(css, /\.ca-trk \.ca-card h2\{font-size:var\(--fs-card\)/, "shared fluid type tokens, not the old fixed 24px");
  assert.ok(!/\.ca-trk \.ca-card h2\{font-size:24px\}/.test(css));
  assert.match(css, /\.ca-trk-tablecard\{[^}]*container-type:inline-size/, "the table card is a size container (no inner scroll at 1280-1920)");
});

test("T7 results table: exact column list, Result chip, row height content; one search card, no dead filters", async () => {
  const html = await populated().T.buildTrackPage();
  const heads = [...html.slice(html.indexOf("<thead>"), html.indexOf("</thead>")).matchAll(/<th>([^<]*)<\/th>/g)].map((m) => m[1]);
  assert.deepEqual(heads, ["Date", "League", "Matchup / Player", "Market", "Model Prediction", "Closing Line", "Model Prob.", "Result", "Confidence", "Final Score / Outcome", "View"]);
  const tbody = html.slice(html.indexOf("<tbody>"), html.indexOf("</tbody>"));
  for (const tr of tbody.split("<tr ").slice(1)) assert.equal((tr.match(/<td/g) || []).length, heads.length, "every row has one cell per column");
});

test("T7 filter row: the search card is its own card, redrawn on reset (clears the box); the Record select does not use the League/Market data attribute", async () => {
  const D = fakeTrackDoc(), urls = [];
  const T2 = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"], { page: "track", globals: {
    document: D.doc, location: { search: "?q=lions", href: "http://localhost/track-record.html?q=lions" }, history: { replaceState: (a, b, u) => urls.push(u) },
    fetch: async (url) => ({ ok: true, json: async () => (urlParts(url).path === "prediction_accuracy" ? [A({ sport: "cfb", game_pk: 1, game_date: dayOff(-1) })] : []) }) } });
  const html = await T2.buildTrackPage();
  assert.ok(html.includes('value="lions"'), "the URL's search text fills the search card");
  T2.wireTrackPage();
  D.handlers.click(hit({ "[data-trk-reset]": {} }));
  assert.ok(D.els["trk-search"].outerHTML.includes('id="trk-search"') && !D.els["trk-search"].outerHTML.includes('value="lions"'), "reset redraws the search card with the cleared text");
  assert.ok(!html.includes('data-trk="record"'));
  const sel = (key, value) => D.handlers.change({ target: { closest: (s) => (s === "select[data-trk]" ? { dataset: { trk: key }, value } : null) } });
  assert.doesNotThrow(() => sel("league", "cfb"));
});


// A permissive document: enough for app.js render() (shell + wireShell + the page wiring) to run end to end.
function renderDoc() {
  const dummy = new Proxy(function () {}, { get: (t, k) => (k === Symbol.toPrimitive ? () => "" : k === "length" ? 0 : dummy), apply: () => dummy, set: () => true });
  const handlers = {}, root = { addEventListener: (t, f) => { handlers[t] = f; }, querySelector: () => null }, shell = { html: "", get innerHTML() { return this.html; }, set innerHTML(v) { this.html = v; }, querySelector: () => dummy };
  const doc = { body: new Proxy({ dataset: { page: "track" } }, { get: (t, k) => (k in t ? t[k] : dummy) }), head: dummy, documentElement: dummy, createElement: () => dummy, addEventListener() {}, querySelectorAll: () => [],
    getElementById: (id) => (id === "trk-root" ? root : dummy), querySelector: (sel) => (sel === ".page-shell" ? shell : dummy) };
  return { doc, shell, handlers };
}
test("T2 Record select: +EV picks re-renders the page with the +EV sections (and back); archive redraws without a refetch", async () => {
  const R = renderDoc(), urls = [], reqs = [];
  const T2 = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"], { page: "track", globals: {
    document: R.doc, location: { search: "", href: "http://localhost/track-record.html" }, history: { replaceState: (a, b, u) => urls.push(u) }, scrollTo() {},
    fetch: async (url) => { const { path } = urlParts(url); reqs.push(path); return { ok: true, json: async () => (path === "prediction_accuracy" ? [A({ sport: "cfb", game_pk: 1, game_date: dayOff(-1) })] : path === "track_record_start" ? [{ sport: "nfl", starts_at: dayOff(-5) + "T12:00:00+00:00", model_version: "nfl-sim-ml-v2" }] : []) }; } } });
  await T2.render();
  assert.ok(R.shell.html.includes("Cumulative Prediction Performance") && !R.shell.html.includes("+EV profit tracker"));
  const change = (value) => R.handlers.change({ target: { closest: (sel) => (sel === "select[data-trk-record]" ? { value } : null) } });
  const last = () => new URL(urls[urls.length - 1]);
  const before = reqs.length;
  change("archive");
  assert.equal(last().searchParams.get("archive"), "1"); assert.equal(reqs.length, before, "the archive redraws from the cached load: no fetch");
  change("ev");
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(last().searchParams.get("view"), "ev"); assert.equal(last().searchParams.has("archive"), false, "+EV picks leaves the archive");
  assert.ok(R.shell.html.includes("+EV profit tracker") && !R.shell.html.includes("Cumulative Prediction Performance"), "the +EV sections replace the model cards");
  assert.ok(reqs.includes("ev_pnl_daily"), "the +EV data is fetched");
  change("model");
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(last().searchParams.has("view"), false, "model picks is the default: a clean URL");
  assert.ok(R.shell.html.includes("Cumulative Prediction Performance") && !R.shell.html.includes("+EV profit tracker"));
});
