import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["js/pikkit.js"]);

// Synthetic bet: a win-the-close bet unless beat === false; placed i minutes after a fixed start.
const T0 = Date.parse("2026-10-01T12:00:00Z");
const B = (i, beat, o = {}) => ({ id: String(i), book: "FanDuel", type: "straight", status: "SETTLED_WIN", odds: beat ? 2.0 : 1.8, close: beat === null ? null : 1.9,
  placed: new Date(T0 + i * 60000), legs: 1, market: "spread", league: "NFL", tag: "own", units: 1, profitUnits: 0.5, ...o });

test("parseCsv handles BOM, quotes, blank close", () => {
  const rows = g.pkParseCsv('﻿bet_id,odds,closing_line,bet_info\n1,1.91,,"Over 8.5 Total Runs, live | Yankees Moneyline"\n');
  assert.equal(rows[0].bet_id, "1"); assert.equal(rows[0].closing_line, ""); assert.match(rows[0].bet_info, /, live \| /);
  assert.equal(rows.length, 1);
});
test("parseCsv: escaped quotes, embedded newlines, CRLF, short rows", () => {
  const rows = g.pkParseCsv('a,b,c\r\n"say ""hi""","line1\nline2",x\r\n2,only\r\n');
  assert.equal(rows[0].a, 'say "hi"'); assert.equal(rows[0].b, "line1\nline2"); assert.equal(rows[0].c, "x");
  assert.equal(rows[1].a, "2"); assert.equal(rows[1].c, "");
  assert.deepEqual(g.pkParseCsv(""), []);
});
test("normBook", () => {
  assert.equal(g.pkNormBook("Draftkings Sportsbook"), "DraftKings"); assert.equal(g.pkNormBook("Novig"), "Novig");
  assert.equal(g.pkNormBook("Fanduel Sportsbook"), "FanDuel"); assert.equal(g.pkNormBook("Hard Rock Sportsbook"), "Hard Rock");
  assert.equal(g.pkNormBook("Caesars Sportsbook"), "Caesars"); assert.equal(g.pkNormBook("BetMGM"), "BetMGM");
  assert.equal(g.pkNormBook("Fanatics Sportsbook"), "Fanatics"); assert.equal(g.pkNormBook("BetRivers Sportsbook"), "BetRivers");
});
test("legMarket", () => {
  assert.equal(g.pkLegMarket("Duke Blue Devils -6.0 Spread Duke Blue Devils at Georgia Tech Yellow Jackets"), "spread");
  assert.equal(g.pkLegMarket("New York Yankees -1.5 Run Line Baltimore Orioles @ New York Yankees"), "spread");
  assert.equal(g.pkLegMarket("PHI Flyers Moneyline PHI Flyers @ DET Red Wings"), "moneyline");
  assert.equal(g.pkLegMarket("Under 4.5 Total Runs - 1st 3 Innings ARI Diamondbacks @ COL Rockies"), "other");
  assert.equal(g.pkLegMarket("Under 0.5 Nolan Arenado - Hits Miami Marlins at Arizona Diamondbacks"), "prop");
  assert.equal(g.pkLegMarket("Over 47.5 Total Points BUF Bills @ KC Chiefs"), "total");
  assert.equal(g.pkLegMarket("Something unrecognised"), "other");
});

const ROW = (o) => ({ bet_id: "1", sportsbook: "Draftkings Sportsbook", type: "straight", status: "SETTLED_WIN", odds: "2.0", closing_line: "1.9", amount: "50", profit: "50",
  time_placed_iso: "2026-10-02T15:00:00Z", bet_info: "Over 47.5 Total Points BUF Bills @ KC Chiefs", tags: "", leagues: "NFL", ...o });
test("toBets drops VOID, counts legs, tags untagged", () => {
  const rows = [ROW({ bet_id: "a" }), ROW({ bet_id: "v", status: "SETTLED_VOID" }),
    ROW({ bet_id: "p", type: "parlay", bet_info: "Over 47.5 Total Points BUF Bills @ KC Chiefs | PHI Flyers Moneyline PHI Flyers @ DET Red Wings | Duke -6.0 Spread Duke at GT", leagues: "NFL | NHL | CFB", tags: "tail-Pro", closing_line: "" })];
  const bets = g.pkToBets(rows, 25);
  assert.deepEqual(bets.map((b) => b.id), ["a", "p"]);
  const [a, p] = bets;
  assert.equal(a.tag, "untagged"); assert.equal(a.legs, 1); assert.equal(a.market, "total"); assert.equal(a.book, "DraftKings");
  assert.equal(a.units, 2); assert.equal(a.profitUnits, 2); assert.equal(a.close, 1.9); assert.equal(a.odds, 2);
  assert.ok(a.placed instanceof Date); assert.equal(a.placed.toISOString(), "2026-10-02T15:00:00.000Z");
  assert.equal(p.legs, 3); assert.equal(p.market, "other"); assert.equal(p.league, "Multiple"); assert.equal(p.tag, "tail-Pro"); assert.equal(p.close, null);
  assert.equal(p.type, "parlay");
});
test("toBets: shared leg market, blank close, bad unit size, PLACED kept", () => {
  const same = g.pkToBets([ROW({ type: "parlay", bet_info: "A Moneyline x | B Moneyline y", profit: "" , status: "PLACED"})], 0)[0];
  assert.equal(same.market, "moneyline"); assert.equal(same.units, 50, "unit size falls back to 1"); assert.equal(same.profitUnits, 0); assert.equal(same.status, "PLACED");
});
test("beat and priceVsClose", () => {
  const b = { odds: 2.0, close: 1.9 };
  assert.equal(g.pkBeat(b), true); assert.ok(Math.abs(g.pkPriceVsClose(b) - 0.0526316) < 1e-6);
  assert.equal(g.pkBeat({ odds: 1.9, close: 1.9 }), false, "equal is not a beat");
  assert.equal(g.pkBeat({ odds: 2, close: null }), null); assert.equal(g.pkPriceVsClose({ odds: 2, close: null }), null);
});
test("summary leaves no-close bets out of beat and vs-close but keeps them in volume and profit", () => {
  const s = g.pkSummary([B(1, true), B(2, false), B(3, null)]);
  assert.equal(s.bets, 3); assert.equal(s.withClose, 2); assert.equal(s.beatRate, 0.5); assert.equal(s.units, 3); assert.equal(s.profitUnits, 1.5);
  assert.ok(Math.abs(s.avgVsClose - ((2 / 1.9 - 1) + (1.8 / 1.9 - 1)) / 2) < 1e-12);
  const e = g.pkSummary([]);
  assert.equal(e.beatRate, null); assert.equal(e.avgVsClose, null); assert.equal(e.bets, 0);
});
test("weekKey is Monday in US Eastern", () => {
  assert.equal(g.pkWeekKey(new Date("2026-10-19T03:00:00Z")), "2026-10-12");   // Sun 23:00 ET belongs to the prior week
  assert.equal(g.pkWeekKey(new Date("2026-10-19T05:00:00Z")), "2026-10-19");
  assert.equal(g.pkWeekKey(new Date("2026-11-02T04:59:00Z")), "2026-10-26", "after the Nov 1 clock change (EST): Sun 23:59 ET");
  assert.equal(g.pkWeekKey(new Date("2026-11-02T05:00:00Z")), "2026-11-02");
});
test("weekly groups by week, newest first", () => {
  const w = g.pkWeekly([B(1, true, { placed: new Date("2026-10-13T16:00:00Z") }), B(2, false, { placed: new Date("2026-10-20T16:00:00Z") }), B(3, true, { placed: new Date("2026-10-14T16:00:00Z") })]);
  assert.deepEqual(w.map((x) => x.week), ["2026-10-19", "2026-10-12"]);
  assert.equal(w[1].bets, 2); assert.equal(w[1].beatRate, 1); assert.equal(w[0].beatRate, 0);
});
test("rolling: one point per bet with a close, in placed order, window of n", () => {
  const bets = [B(3, false), B(1, true), B(2, true), B(4, null), B(5, true)];   // out of order, one without a close
  const r = g.pkRolling(bets, 3);
  assert.deepEqual(r.map((p) => p.rate), [1, 1, 2 / 3, 2 / 3], "bets 1,2,3,5: window drops bet 1 at the fourth point");
  assert.ok(r[0].placed instanceof Date); assert.deepEqual(g.pkRolling([], 3), []);
  assert.equal(g.pkRolling(Array.from({ length: 150 }, (_, i) => B(i, i >= 50))).at(-1).rate, 1, "default window is 100");
});
test("unitBucket and legBucket edges", () => {
  assert.equal(g.pkUnitBucket(0.5), "≤0.5u"); assert.equal(g.pkUnitBucket(0.51), "≤1u"); assert.equal(g.pkUnitBucket(1), "≤1u");
  assert.equal(g.pkUnitBucket(2), "≤2u"); assert.equal(g.pkUnitBucket(2.01), ">2u");
  assert.equal(g.pkLegBucket(7), "5+"); assert.equal(g.pkLegBucket(1), "1"); assert.equal(g.pkLegBucket(4), "4"); assert.equal(g.pkLegBucket(5), "5+");
});
test("breakdowns cover every dimension, legs and stake in natural order", () => {
  const bd = g.pkBreakdowns([B(1, true, { book: "Novig", legs: 3, units: 3 }), B(2, false), B(3, true)]);
  assert.deepEqual(Object.keys(bd), ["league", "book", "market", "type", "legs", "stake", "tag"]);
  assert.deepEqual(bd.book.map((r) => [r.value, r.bets]), [["FanDuel", 2], ["Novig", 1]]);
  assert.deepEqual(bd.legs.map((r) => r.value), ["1", "3"]); assert.deepEqual(bd.stake.map((r) => r.value), ["≤1u", ">2u"]);
  assert.equal(bd.book[0].beatRate, 0.5);
});

// 200 bets with a close: the prior 100 are FanDuel at 50%; the current 100 (30 beats) are DraftKings 20 @ 20%, FanDuel 16 @ 37.5%,
// Hard Rock 10 @ 0% (too few for a slice) and Novig 54 @ 37%.
function alertSet(currentBeats = [4, 6, 0, 20]) {
  const out = []; let i = 0;
  for (let k = 0; k < 100; k++) out.push(B(i++, k % 2 === 0));
  const groups = [["DraftKings", 20], ["FanDuel", 16], ["Hard Rock", 10], ["Novig", 54]];
  groups.forEach(([book, n], gi) => { for (let k = 0; k < n; k++) out.push(B(i++, k < currentBeats[gi], { book })); });
  return out;
}
test("alert fires below 40% and lists slices worst first with prior rate", () => {
  const a = g.pkAlert(alertSet());
  assert.ok(a); assert.equal(a.rate, 0.3);
  const books = a.slices.filter((s) => s.dim === "book");
  assert.deepEqual(books.map((s) => s.value), ["DraftKings", "Novig", "FanDuel"], "worst first; Hard Rock (10 bets) omitted");
  assert.deepEqual(books[0], { dim: "book", value: "DraftKings", n: 20, rate: 0.2, priorRate: null });
  assert.equal(books[2].n, 16); assert.equal(books[2].rate, 0.375); assert.equal(books[2].priorRate, 0.5);
  for (let k = 1; k < a.slices.length; k++) assert.ok(a.slices[k - 1].rate <= a.slices[k].rate, "ascending by rate");
  const league = a.slices.find((s) => s.dim === "league");
  assert.equal(league.value, "NFL"); assert.equal(league.n, 100); assert.equal(league.priorRate, 0.5);
});
test("alert null at or above 40%", () => {
  assert.equal(g.pkAlert(alertSet([4, 6, 0, 30])), null, "exactly 40 of 100");
  assert.equal(g.pkAlert(alertSet([20, 16, 10, 54])), null);
  assert.equal(g.pkAlert([]), null);
  assert.equal(g.pkAlert([B(1, null)]), null, "no bet with a close");
});
test("alert window is the latest 100 bets with a close, by placed time", () => {
  const set = alertSet().reverse();   // input order must not matter
  assert.equal(g.pkAlert(set).rate, 0.3);
});
test("no top-level name can shadow a window global", () => {
  for (const n of ["alert", "beat", "weekly", "rolling", "summary", "normBook"]) assert.equal(g[n], undefined, n);
});
