import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/game.js", "js/boot.js"];
const g = loadScripts(FILES);
const src = (f) => fs.readFileSync(new URL("../" + f, import.meta.url), "utf8");

/* ── pure helpers ───────────────────────────────────────────────────────── */
test("read picks the largest positive edge; none -> null", () => {
  const r = g.gameRead([{ market: "spread", label: "Alabama -10.5", modelProb: 0.618, impliedProb: 0.554 },
                        { market: "total", label: "Over 52.5", modelProb: 0.573, impliedProb: 0.501 },
                        { market: "moneyline", label: "Alabama ML", modelProb: 0.821, impliedProb: 0.809 }]);
  assert.equal(r.market, "total"); assert.ok(Math.abs(r.edgePp - 7.2) < 1e-9);
  assert.equal(g.gameRead([{ market: "spread", label: "x", modelProb: 0.4, impliedProb: 0.5 }]), null);
});

test("gameRead: empty / null / non-finite inputs and zero edge give null; result carries the market's fields", () => {
  assert.equal(g.gameRead([]), null);
  assert.equal(g.gameRead(null), null);
  assert.equal(g.gameRead([{ market: "total", label: "Over 1", modelProb: 0.5, impliedProb: 0.5 }]), null, "zero edge is not an edge");
  assert.equal(g.gameRead([{ market: "total", label: "Over 1", modelProb: NaN, impliedProb: 0.5 }, { market: "spread", label: "a", modelProb: null, impliedProb: 0.5 }]), null);
  const r = g.gameRead([{ market: "spread", side: "home", label: "KC -3", modelProb: 0.6, impliedProb: 0.5 }]);
  assert.deepEqual([r.market, r.label, r.modelProb, r.impliedProb, r.side], ["spread", "KC -3", 0.6, 0.5, "home"]);
});

test("gmLean: moneyline favorite, spread / total lean at half-point rounding, none when on the line", () => {
  const p = { home_win_prob: 0.7, pred_home_score: 30, pred_away_score: 20, market_spread: -3, market_total: 45.5 };
  assert.deepEqual({ ...g.gmLean(p) }, { ml: "home", spread: "home", total: "over" });
  assert.equal(g.gmLean({ ...p, home_win_prob: 0.4 }).ml, "away");
  assert.equal(g.gmLean({ ...p, market_spread: -13 }).spread, "away", "model margin 10 vs line -13: away covers");
  assert.equal(g.gmLean({ ...p, market_spread: -10 }).spread, null, "model sits on the line");
  assert.equal(g.gmLean({ ...p, market_total: 60 }).total, "under");
  const none = g.gmLean({ home_win_prob: null, pred_home_score: null, pred_away_score: null, market_spread: null, market_total: null });
  assert.deepEqual({ ...none }, { ml: null, spread: null, total: null });
});

test("gmWhen: 'Mon, Oct 12 · 7:00 PM ET' in Eastern time; blank for none", () => {
  assert.equal(g.gmWhen("2026-10-12T23:00:00Z"), "Mon, Oct 12 · 7:00 PM ET");
  assert.equal(g.gmWhen("2026-10-06T00:15:00Z"), "Mon, Oct 5 · 8:15 PM ET", "00:15Z is still the 5th in ET");
  assert.equal(g.gmWhen(null), "");
  assert.equal(g.gmWhen("garbage"), "");
});

test("consensusAmerican: median implied probability across books (object or JSON string); null when nothing usable", () => {
  assert.equal(g.consensusAmerican({ a: -110, b: -110, c: -110 }), -110);
  assert.equal(g.consensusAmerican('{"a":-120,"b":-110,"c":-130}'), -120);
  assert.equal(g.consensusAmerican({ a: 200, b: 220, c: 240 }), 220);
  assert.equal(g.consensusAmerican({ a: null, b: "x", c: 50, d: 0 }), null, "junk and |price| < 100 are skipped");
  assert.equal(g.consensusAmerican(null), null);
  assert.equal(g.consensusAmerican("not json"), null);
  const mixed = g.consensusAmerican({ a: -105, b: 105 });   // dog/fav straddling even money still returns a price
  assert.ok(Number.isFinite(mixed) && Math.abs(mixed) >= 100);
});

/* shared fixtures */
const AWAY_C = "South Carolina Gamecocks", HOME_C = "Alabama Crimson Tide";
const cfbPred = { sport: "cfb", game_pk: 7, away_team_name: AWAY_C, home_team_name: HOME_C, home_win_prob: 0.821, pred_home_score: 34, pred_away_score: 20.8,
  market_spread: -10.5, market_total: 52.5, commence_time: "2026-10-12T23:00:00Z" };
// ev_picks rows: true_prob is Pinnacle's no-vig line, NOT the model. Poison it (0.99) so any leak into a model number shows.
const evRow = (o) => ({ sport: "cfb", game_pk: 7, market: "spread", side: "home", true_prob: 0.99, best_line_implied: 0.5, best_price: -110, best_book: "fanduel", is_pick: false, created_at: "2026-10-10T12:00:00Z", ...o });
const uniform = (lo, hi, len) => Array.from({ length: len }, (_, i) => (i >= lo && i <= hi ? 1 / (hi - lo + 1) : 0));
const MARGIN = { kind: "margin", offset: 10, pmf: uniform(0, 20, 21) };    // home margin -10..10, uniform
const TOTAL = { kind: "pmf", pmf: uniform(40, 59, 101) };                  // total 40..59, uniform
// model: home 75%, margin / total distributions; market: spread -3, total 50
const P = { ...cfbPred, home_win_prob: 0.75, pred_home_score: 30, pred_away_score: 20, market_spread: -3, market_total: 50, margin_dist: JSON.stringify(MARGIN), total_dist: TOTAL };
const EV = [evRow({ market: "moneyline", side: "home", best_line_implied: 0.7, best_price: -233 }), evRow({ market: "moneyline", side: "away", best_line_implied: 0.31, best_price: 225 }),
  evRow({ market: "spread", side: "home", best_line_implied: 0.5 }), evRow({ market: "spread", side: "away", best_line_implied: 0.5 }),
  evRow({ market: "total", side: "over", best_line_implied: 0.45, best_price: 120 })];

test("gmOdds: the model's probability comes from the distributions at the market line (never true_prob); pushes excluded", () => {
  const o = g.gmOdds(P, EV, null, null, null);
  // home covers at -3 when margin > 3: 4..10 = 7 of 20 (3 is a push); away at +3 when margin < 3: -10..2 = 13 of 20
  assert.ok(Math.abs(o.spread.home.modelProb - 7 / 20) < 1e-9); assert.ok(Math.abs(o.spread.away.modelProb - 13 / 20) < 1e-9);
  assert.equal(o.spread.home.line, -3); assert.equal(o.spread.away.line, 3);
  // over 50: 51..59 = 9 of 19; under: 40..49 = 10 of 19
  assert.ok(Math.abs(o.total.over.modelProb - 9 / 19) < 1e-9); assert.ok(Math.abs(o.total.under.modelProb - 10 / 19) < 1e-9);
  assert.equal(o.moneyline.home.modelProb, 0.75); assert.ok(Math.abs(o.moneyline.away.modelProb - 0.25) < 1e-9);
  assert.equal(o.spread.home.implied, 0.5); assert.equal(o.total.over.implied, 0.45); assert.equal(o.total.under.implied, null, "no price known for that side");
  const json = JSON.stringify(o);
  assert.ok(!json.includes("0.99"), "true_prob (Pinnacle) appears in no model number");
});

test("gmOdds: evaluated at the line the market row is priced at (ev_best_lines), and the label shows that line", () => {
  const lineBy = new Map([["7|spread|home", -4], ["7|spread|away", 4]]);
  const o = g.gmOdds(P, EV, null, null, lineBy);
  assert.equal(o.spread.home.line, -4);
  assert.ok(Math.abs(o.spread.home.modelProb - 6 / 20) < 1e-9, "home covers -4 when margin > 4: 5..10");
  assert.ok(Math.abs(o.spread.away.modelProb - 14 / 20) < 1e-9, "away covers +4 when margin < 4: -10..3");
  const c = g.gmCandidates(P, o).find((x) => x.market === "spread" && x.side === "home");
  assert.equal(c.label, "Alabama -4");
});

test("gmOdds: the prediction row's distributions win over the sim's; the sim is the fallback; no distribution -> no spread / total probability (moneyline still has one)", () => {
  const simMargin = { kind: "margin", offset: 10, pmf: uniform(20, 20, 21) };   // all mass at +10
  const withPred = g.gmOdds(P, EV, null, { margin_dist: simMargin, total_dist: TOTAL }, null);
  assert.ok(Math.abs(withPred.spread.home.modelProb - 7 / 20) < 1e-9, "prediction row's margin_dist is used");
  const bare = { ...P, margin_dist: null, total_dist: undefined };
  const viaSim = g.gmOdds(bare, EV, null, { margin_dist: JSON.stringify(MARGIN), total_dist: JSON.stringify(TOTAL) }, null);
  assert.ok(Math.abs(viaSim.spread.home.modelProb - 7 / 20) < 1e-9 && Math.abs(viaSim.total.over.modelProb - 9 / 19) < 1e-9);
  const none = g.gmOdds(bare, EV, null, null, null);
  assert.equal(none.spread.home.modelProb, null); assert.equal(none.total.over.modelProb, null);
  assert.equal(none.moneyline.home.modelProb, 0.75);
  assert.equal(g.gmCover(none), null);
  assert.ok(!JSON.stringify(none).includes("0.99"));
});

test("gmOdds: consensus moneyline price, best price kept for the read; moneyline falls back to the moneylines view's best price", () => {
  const ml = { home_price: -400, home_book: "fd", away_price: 330, away_book: "mgm", home_prices: { dk: -425, fd: -425, mgm: -450 }, away_prices: { dk: 320, fd: 320, mgm: 300 } };
  const o = g.gmOdds(P, [], ml, null, null);
  assert.equal(o.moneyline.home.price, -425); assert.equal(o.moneyline.away.price, 320);
  assert.ok(Math.abs(o.moneyline.home.implied - 425 / 525) < 1e-9);
  assert.equal(o.moneyline.home.bestPrice, -400); assert.ok(Math.abs(o.moneyline.home.bestImplied - 400 / 500) < 1e-9, "the read compares against the price you could take");
  const e = g.gmOdds({ sport: "cfb" }, [], null, null, null);
  assert.equal(e.moneyline.home.price, null); assert.equal(e.spread.home.modelProb, null); assert.equal(e.spread.line, null);
});

test("gameCandidates / read: the model disagrees with the market and NO row is a pick -> the Read still shows the model's edge", () => {
  const cands = g.gmCandidates(P, g.gmOdds(P, EV, null, null, null));
  assert.deepEqual(cands.map((c) => `${c.market}|${c.side}`), ["moneyline|away", "moneyline|home", "spread|away", "spread|home", "total|over"]);
  const r = g.gameRead(cands);
  assert.equal(r.label, "South Carolina +3");                      // away +3 covers 65% vs 50% implied: +15.0
  assert.equal(r.market, "spread"); assert.ok(Math.abs(r.edgePp - 15) < 1e-9);
  assert.ok(Math.abs(r.modelProb - 0.65) < 1e-9); assert.equal(r.impliedProb, 0.5);
  const rowsWithOnlyPoison = g.gmCandidates(P, g.gmOdds({ ...P, margin_dist: null, total_dist: null, home_win_prob: null }, EV, null, null, null));
  assert.deepEqual(rowsWithOnlyPoison, [], "without a model number nothing is compared, true_prob is never used");
});

test("modelProjectionRows: Market / CappingAlpha / Difference at the same line; the CappingAlpha cell is the model, never true_prob", () => {
  const rows = g.modelProjectionRows(P, g.gmOdds(P, EV, null, null, null));
  assert.deepEqual(rows.map((r) => r.label), ["Moneyline", "Spread", "Total"]);
  const [ml, sp, tot] = rows;
  assert.equal(ml.market, "ALA -233 (70.0%)"); assert.equal(ml.model, "75.0%"); assert.ok(Math.abs(ml.diff - (0.75 - 233 / 333) * 100) < 1e-9);
  assert.equal(sp.market, "ALA -3 (50.0%)"); assert.equal(sp.model, "ALA -10 (35.0%)");
  assert.ok(Math.abs(sp.diff + 15) < 1e-9, "35% cover vs 50% implied"); assert.ok(Math.abs(sp.pts - 7) < 1e-9);
  assert.equal(tot.market, "O 50 (45.0%)"); assert.equal(tot.model, "50.0 (47.4%)"); assert.ok(Math.abs(tot.diff - (9 / 19 - 0.45) * 100) < 1e-9);
  assert.ok(!JSON.stringify(rows).includes("99.0%"), "Pinnacle's true_prob never reaches the table");
});

test("modelProjectionRows: away favorite, under lean, missing odds / probabilities show dashes (never NaN) and fall back to points", () => {
  const p = { ...cfbPred, home_win_prob: 0.3, pred_home_score: 17, pred_away_score: 30, market_spread: 2.5, market_total: 52.5 };
  const [ml, sp, tot] = g.modelProjectionRows(p, g.gmOdds(p, [], null, null, null));
  assert.equal(ml.market, "—"); assert.equal(ml.model, "70.0%"); assert.equal(ml.diff, null);
  assert.equal(sp.market, "SC -2.5", "away line = -home line; no price known so no implied %");
  assert.equal(sp.model, "SC -13"); assert.equal(sp.diff, null); assert.ok(Math.abs(sp.pts - 10.5) < 1e-9, "model likes the away side by 10.5 more points than the market");
  assert.equal(tot.market, "U 52.5"); assert.equal(tot.model, "47.0"); assert.ok(Math.abs(tot.pts - 5.5) < 1e-9);
  const blank = g.modelProjectionRows({ sport: "cfb", home_team_name: "A", away_team_name: "B" }, g.gmOdds({ sport: "cfb" }, [], null, null, null));
  assert.equal(blank.length, 3);
  assert.ok(blank.every((r) => r.market === "—" && r.model === "—" && r.diff === null && r.pts === null));
  assert.ok(!JSON.stringify(blank).includes("NaN"));
});

test("gmCover: model cover probability per side at the market lines; null when the model has no distribution", () => {
  const c = g.gmCover(g.gmOdds(P, EV, null, null, null));
  assert.ok(Math.abs(c.home - 0.35) < 1e-9 && Math.abs(c.away - 0.65) < 1e-9);
  assert.equal(g.gmCover(g.gmOdds({ ...P, margin_dist: null }, EV, null, null, null)), null);
});

/* team context fixtures */
const grade = (side, team, o) => ({ side, team, overall: "B", pass: "A", run: "C", overall_pct: 78, early: false,
  units: { pass_rate: 0.5, off: { pass_explosive: 0.02, run_explosive: 0.01 }, def: { pass_explosive: 0.01, run_explosive: 0 } }, ...o });
const histRow = (side, team, o) => ({ side, team, windows: { season: { n: 6, su: "4-2", ats: "3-3-0", ou: "3-3-0", n_ats: 6, n_ou: 6 } }, streaks: { su: "W1", ats: "W1", ou: "O1", notes: {} }, ...o });
const pw = (team, rank, rating, su) => ({ team, rank, rating, su, units: { pass_off: { rank: 5 }, run_off: { rank: 9 }, pass_def: { rank: 3 }, run_def: { rank: 12 } } });

test("gmRecord: season straight-up record from team_history, power_rankings as backup, none -> empty (no conference record exists)", () => {
  const hist = [histRow("home", "333"), histRow("away", "87", { windows: { season: { n: 0, su: "0-0" } } })];
  const power = [pw("333", 2, 18.4, "5-1"), pw("87", 31, -2.1, "2-3")];
  assert.equal(g.gmRecord(hist, power, "home"), "4-2");
  assert.equal(g.gmRecord(hist, power, "away"), "2-3", "no games in the season window -> power_rankings su");
  assert.equal(g.gmRecord([], [], "home"), "");
  assert.equal(g.gmRecord(null, null, "away"), "");
});

test("gmKeyInsights: up to four sentences from unit grades, explosive plays, power ranks and streaks; empty without data", () => {
  const D = { sport: "cfb", r: cfbPred,
    ctxGrades: [grade("home", "333", { overall: "A", overall_pct: 92, units: { pass_rate: 0.5, off: { pass_explosive: 0.3, run_explosive: 0.2 }, def: { pass_explosive: 0.1, run_explosive: 0.1 } } }),
                grade("away", "87", { units: { pass_rate: 0.5, off: { pass_explosive: 0, run_explosive: 0 }, def: { pass_explosive: 0, run_explosive: 0 } } })],
    ctxHist: [histRow("home", "333", { streaks: { su: "W4", ats: "W1", ou: "O1" } }), histRow("away", "87")],
    ctxPower: [pw("333", 2, 18.4, "5-1"), pw("87", 31, -2.1, "2-3")] };
  const ins = g.gmKeyInsights(D);
  assert.equal(ins.length, 4);
  assert.deepEqual(ins.map((i) => i.kind), ["grade", "explosive", "power", "streak"]);
  assert.match(ins[0].title, /Alabama offense grades A vs South Carolina/); assert.match(ins[0].body, /Pass A, run C/);
  assert.match(ins[1].title, /Big-play grades favor Alabama/); assert.ok(!/pass and run/.test(ins[1].body), "only what is computed");
  assert.match(ins[2].title, /Alabama ranks #2/); assert.match(ins[2].body, /South Carolina ranks #31/); assert.match(ins[2].body, /20\.5/);
  assert.match(ins[3].title, /Alabama won 4 straight/);
  assert.ok(ins.every((i) => !/NaN|undefined/.test(i.title + i.body)));
  assert.deepEqual(g.gmKeyInsights({ sport: "cfb", r: cfbPred, ctxGrades: [], ctxHist: [], ctxPower: [] }), []);
  assert.deepEqual(g.gmKeyInsights({ sport: "mlb", r: cfbPred }), [], "missing arrays never throw");
  const early = g.gmKeyInsights({ sport: "cfb", r: cfbPred, ctxGrades: [], ctxHist: [histRow("home", "333", { streaks: { su: "W2", ats: "L3", ou: "U1" } }), histRow("away", "87")], ctxPower: [] });
  assert.equal(early.length, 1); assert.match(early[0].title, /Alabama failed to cover 3 straight/, "streaks shorter than 3 are skipped; the longest wins");
});

test("gmBestBooks: moneyline best price per side from the moneylines view; spread / total from the latest pick builds", () => {
  const ml = { home_price: -400, home_book: "fanduel", away_price: 330, away_book: "betmgm" };
  const rows = g.gmBestBooks(cfbPred, ml, [evRow({ best_price: -124, best_book: "fanduel" }), evRow({ market: "total", side: "under", best_price: -105, best_book: "draftkings" })]);
  assert.deepEqual(rows.map((r) => [r.label, r.price, r.book]), [["South Carolina ML", 330, "betmgm"], ["Alabama ML", -400, "fanduel"], ["Alabama -10.5", -124, "fanduel"], ["Under 52.5", -105, "draftkings"]]);
  assert.deepEqual(g.gmBestBooks(cfbPred, null, []), []);
});

/* ── populated render ───────────────────────────────────────────────────── */
const EVIL_AWAY = 'Evil "Q" <b>Crew</b>', HOME = 'Kansas City "Chiefs"', PLAYER = 'O"Brien <i>Zed</i>';
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };
const gauss = (n, mu, sd) => { const v = Array.from({ length: n }, (_, i) => Math.exp(-0.5 * ((i - mu) / sd) ** 2)), s = v.reduce((a, b) => a + b, 0); return v.map((x) => x / s); };

function populated({ search = "?sport=nfl&game=5", moves = "rows", noEdge = false, sport = "nfl", pred: predOver = {}, ctx = true, noPred = false, propOpp = false } = {}) {
  const NOW = Date.now(), iso = (h) => new Date(NOW + h * 36e5).toISOString();
  const pred = { game_pk: 5, sport, home_team_name: HOME, away_team_name: EVIL_AWAY, commence_time: iso(30), home_win_prob: 0.62, pred_home_score: 27.4, pred_away_score: 20.3,
    market_spread: -3.5, market_total: 44.5, model_version: "nfl-sim-ml-v2", ...predOver };
  // ev_picks rows are NOT picks and carry a poisoned true_prob (Pinnacle's line, 0.99): the page must read the model, never this.
  const ev = (o) => ({ sport, game_pk: 5, matchup: `${EVIL_AWAY} @ ${HOME}`, market: "spread", side: "home", true_prob: 0.99, best_line_implied: noEdge ? 0.95 : 0.55, best_price: -122, best_book: "fanduel",
    ev_best: 0.3, is_pick: false, model_version: "ev-pilot-v1", created_at: iso(-3), commence_time: iso(30), ...o });
  const evRows = [ev({}), ev({ market: "total", side: "over", best_line_implied: noEdge ? 0.95 : 0.45, best_price: 120, best_book: "draftkings" }),
    ev({ market: "moneyline", side: "home", best_line_implied: noEdge ? 0.95 : 0.78, best_price: -355 })];
  const sim = { game_pk: 5, model_version: "nfl-sim-ml-v2", created_at: iso(-2), margin_dist: JSON.stringify({ kind: "margin", offset: 40, pmf: gauss(81, 47, 12) }),
    total_dist: JSON.stringify({ kind: "pmf", pmf: gauss(101, 47, 9) }), away_score_dist: JSON.stringify({ kind: "pmf", pmf: gauss(80, 20, 7) }), home_score_dist: JSON.stringify({ kind: "pmf", pmf: gauss(80, 27, 7) }) };
  const pdist = JSON.stringify({ kind: "pmf", pmf: gauss(150, 70, 25) });
  const playerSims = [{ player_id: "p1", name: PLAYER, pos: "WR", team: HOME, market: "rec_yds", mean: 71, dist: pdist, created_at: iso(-2), model_version: "nfl-sim-ml-v2" }];
  const hist = [{ side: "away", team: "AAA", windows: { season: { n: 4, su: "3-1", ats: "2-2-0", ou: "2-2-0", n_ats: 4, n_ou: 4, avg_margin: 3 }, L5: { n: 4, su: "3-1", ats: "2-2-0", ou: "2-2-0", n_ats: 4, n_ou: 4, avg_margin: 3 } }, streaks: { su: "W3", ats: "W1", ou: "O1", notes: {} }, splits: { home: { n: 2, su: "1-1", ats: "1-1-0" } } },
    { side: "home", team: "KC", windows: { season: { n: 4, su: "2-2", ats: "2-2-0", ou: "1-3-0", n_ats: 4, n_ou: 4, avg_margin: -1 } }, streaks: { su: "L1", ats: "L1", ou: "U2", notes: {} }, splits: {} }];
  const grades = [{ side: "away", team: "AAA", overall: "B", pass: "A", run: "C", overall_pct: 78, early: false, units: { pass_rate: 0.55, off: { pass_explosive: 0.03, run_explosive: 0.01 }, def: { pass_explosive: 0.02, run_explosive: 0 } } },
    { side: "home", team: "KC", overall: "D", pass: "D", run: "F", overall_pct: 21, early: false, units: { pass_rate: 0.6, off: { pass_explosive: -0.01, run_explosive: -0.02 }, def: { pass_explosive: 0, run_explosive: 0.01 } } }];
  const power = [pw("AAA", 4, 6.2, "3-1"), pw("KC", 19, -1.4, "2-2")];
  const splits = [{ game_pk: 5, market: "spread", side: "away", cash_pct: 61, ticket_pct: 40, captured_at: iso(-1) }, { game_pk: 5, market: "spread", side: "home", cash_pct: 39, ticket_pct: 60, captured_at: iso(-1) }];
  const mlRow = noEdge ? { game_pk: 5, home_price: -2000, home_book: "fanduel", away_price: -2000, away_book: "betmgm", home_prices: { dk: -2000 }, away_prices: { dk: -2000 } }
    : { game_pk: 5, home_price: -330, home_book: "fanduel", away_price: 290, away_book: "betmgm", home_prices: { dk: -340, fd: -330, mgm: -350 }, away_prices: { dk: 280, fd: 290, mgm: 270 } };
  const oppEv = [ev({ true_prob: 0.7, best_line_implied: 0.5, best_price: 100, ev_best: 0.3, is_pick: true })];
  const propRows = propOpp ? [{ sport, game_pk: 5, matchup: `${EVIL_AWAY} @ ${HOME}`, market: "rec_yds", side: "over", line: 64.5, player_name: PLAYER, model_prob: 0.7, best_price: -110, best_book: "fanduel", ev_best: 0.5, is_pick: true, commence_time: iso(30) }] : [];
  const requested = [];
  const fetch = async (url) => {
    const { path, q } = urlParts(url); requested.push(path + q);
    let rows = [];
    if (path === "predictions_any" || path === "predictions_current") rows = noPred ? [] : [pred];
    else if (path === "ev_picks") rows = evRows;
    else if (path === "nfl_sim_serving") rows = [{ model_version: "nfl-sim-ml-v2" }];
    else if (path === "nfl_sim") rows = [sim];
    else if (path === "nfl_player_sim") rows = playerSims;
    else if (path === "game_moneylines_current") rows = [mlRow];
    else if (path === "ev_current") rows = !noEdge && q.includes(`sport=eq.${sport}`) ? oppEv : [];
    else if (path === "ev_prop_picks_current") rows = q.includes(`sport=eq.${sport}`) ? propRows : [];
    else if (path === "nfl_betting_splits_current" || path === "cfb_betting_splits_current") rows = splits;
    else if (path === "team_history") rows = ctx ? hist : [];
    else if (path === "matchup_grades") rows = ctx ? grades : [];
    else if (path === "power_rankings_current") rows = ctx ? power : [];
    else if (path === "line_moves_current") {
      if (moves === 404) return { ok: false, status: 404, text: async () => "not found", json: async () => ({}) };
      rows = moves === "rows" ? [{ game_pk: 5, market: "spread", side: "home", open_line: -2.5, open_price: -110, cur_line: -3.5, cur_price: -110, open_at: iso(-30), cur_at: iso(-1), commence_time: iso(30) },
        { game_pk: 99, market: "total", side: "over", open_line: 40, open_price: -110, cur_line: 45, cur_price: -110, open_at: iso(-30), cur_at: iso(-1), commence_time: iso(30) }] : [];
    }
    return { ok: true, json: async () => rows };
  };
  const D = loadScripts(FILES, { page: "game", globals: { fetch, location: { search, href: `http://localhost/game.html${search}` } } });
  return { D, requested };
}
const tabHtml = async (tab, o = {}) => { const { D } = populated({ search: `?sport=nfl&game=5&tab=${tab}`, ...o }); return D.buildGamePage(); };
const clean = (html) => html.replace(/\bnull\b(?=[^<]*>)/g, "");
// the populated game's model, computed independently of the page: sim margin = i - 40 (mean +7), total = i (mean 47)
const tail = (arr, from) => arr.slice(from).reduce((a, b) => a + b, 0);
const HOME_COVER = tail(gauss(81, 47, 12), 44), TOTAL_OVER = tail(gauss(101, 47, 9), 45);   // margin > 3.5 ; total > 44.5 (half lines: no pushes)
const p1 = (x) => `${(x * 100).toFixed(1)}%`, sp1 = (pp) => `${pp >= 0 ? "+" : ""}${pp.toFixed(1)}%`;

test("buildGamePage: hero, odds boxes, CappingAlpha Read, tabs and all five Overview cards from stubbed data; names escaped; no NaN / undefined", async () => {
  const { D, requested } = populated();
  const html = await D.buildGamePage();
  for (const t of ["‹ NFL Board", " · ", " ET", "MONEYLINE", "SPREAD", "TOTAL", "CAPPINGALPHA READ", "shows the strongest model divergence", "MODEL EDGE",
    "Overview", "Matchup", "Market", "Trends", "Players", "Alpha Score", "Win Probability", "Projected Score", "Projected Spread", "Projected Total",
    "Model Projection", "Moneyline", "CappingAlpha", "Difference", "Key Insights", "Cover Probability"]) assert.ok(html.includes(t), `missing ${t}`);
  assert.ok(html.includes("&lt;b&gt;Crew&lt;/b&gt;") && html.includes("Kansas City &quot;Chiefs&quot;"), "hostile names are escaped");
  assert.ok(!html.includes("<b>Crew</b>") && !html.includes("<i>Zed</i>"));
  assert.ok(!/NaN|undefined/.test(clean(html)), "no NaN / undefined leaks");
  assert.ok(!html.includes("Projected Game Flow") && !/Tuscaloosa|Bryant-Denny/.test(html), "Phase B items and mockup placeholders are never rendered");
  assert.ok(requested.some((r) => r.startsWith("line_moves_current")));
});

test("hero odds boxes: consensus moneyline, lines, total; the model's side is marked; records come from team_history", async () => {
  const html = await populated().D.buildGamePage();
  const hero = html.slice(html.indexOf("ca-gm-hero"), html.indexOf("ca-gm-read"));
  assert.ok(hero.includes("-340") && !hero.includes("-330"), "consensus (median) home price shown, not the best price");
  assert.ok(hero.includes("+280"), "median of +280 / +290 / +270 away");
  assert.ok(hero.includes("+3.5") && hero.includes("-3.5"), "spread: away +3.5 / home -3.5");
  assert.ok(hero.includes("O 44.5") && hero.includes("U 44.5"));
  assert.match(html, /class="ca-gm-hero has-odds"/, "hero with odds boxes lets them straddle its edge");
  assert.match(hero, /class="ca-gm-v lean">-340</, "home is the model's moneyline side");
  assert.ok(hero.includes("3-1") && hero.includes("2-2"), "season records");
  assert.ok(!/\(\d+-\d+ [A-Z]+\)/.test(hero), "no conference record is ever invented");
});

test("Read card: the largest model-minus-market edge over every side (no pick gate), at the line used, with the edge box", async () => {
  const html = await populated().D.buildGamePage();
  const read = html.slice(html.indexOf("ca-gm-read"), html.indexOf("ca-gm-tabs"));
  assert.ok(read.includes("<h2>Over 44.5 shows the strongest model divergence</h2>"), "total over 44.5: model beats the 45.0% implied by the most");
  assert.ok(read.includes(`The market implies a 45.0% hit probability. CappingAlpha estimates <b>${p1(TOTAL_OVER)}</b>.`));
  assert.ok(read.includes(`<b>${sp1((TOTAL_OVER - 0.45) * 100)}</b><span>MODEL EDGE</span>`));
  assert.ok(!read.includes("99.0%"), "Pinnacle's true_prob (poisoned to 0.99 here) is never shown as CappingAlpha");
});

test("no side beats its price: 'No model edge on this game', no edge box, page still renders", async () => {
  const html = await populated({ noEdge: true }).D.buildGamePage();
  const read = html.slice(html.indexOf("ca-gm-read"), html.indexOf("ca-gm-tabs"));
  assert.ok(read.includes("No model edge on this game"));
  assert.ok(!read.includes("ca-gm-edge") && !read.includes("MODEL EDGE"));
  assert.ok(html.includes("Model Projection"));
  assert.ok(!/NaN|undefined/.test(clean(html)));
});

test("a started game shows no live-looking edge box: 'The model read is shown before kickoff.'", async () => {
  const started = new Date(Date.now() - 5 * 36e5).toISOString();
  const html = await populated({ pred: { commence_time: started } }).D.buildGamePage();
  const read = html.slice(html.indexOf("ca-gm-read"), html.indexOf("ca-gm-tabs"));
  assert.ok(read.includes("Pre-game read — this game has started") && read.includes("The model read is shown before kickoff."));
  assert.ok(!read.includes("ca-gm-edge") && !read.includes("MODEL EDGE"));
  const upcoming = await populated().D.buildGamePage();
  assert.ok(upcoming.includes("MODEL EDGE"), "same data, upcoming: the edge is shown");
});

test("Overview cards: Alpha Score from this game's best opportunity, win probability, projected score / spread / total with the market line", async () => {
  const html = await populated().D.buildGamePage();
  const cards = html.slice(html.indexOf('id="gm-cards"'), html.indexOf('id="gm-projection"'));
  assert.match(cards, /Alpha Score[\s\S]*?95/, "alpha score of the best opportunity for this game");
  assert.ok(cards.includes("62.0%") && cards.includes("38.0%"), "win probability, both sides");
  assert.ok(cards.includes("<b>20</b><span>-</span><b>27</b>"), "projected score, away - home");
  assert.ok(cards.includes("Market: -3.5") && cards.includes("Market: 44.5"));
  assert.ok(cards.includes('ca-gm-mid">47.7<'), "projected total = 27.4 + 20.3");
  assert.ok(cards.includes(' -7</div><div class="ca-gm-sub">Market: -3.5'), "projected spread = home by 7.1, at the half point like the board");
  const none = await populated({ noEdge: true }).D.buildGamePage();
  assert.match(none.slice(none.indexOf('id="gm-cards"'), none.indexOf('id="gm-projection"')), /Alpha Score[\s\S]*—/);
});

test("Model Projection, Key Insights and Cover Probability: the model at the market line, with the splits caption", async () => {
  const html = await populated().D.buildGamePage();
  const proj = html.slice(html.indexOf('id="gm-projection"'), html.indexOf('id="gm-insights"'));
  assert.ok(/Moneyline/.test(proj) && /Spread/.test(proj) && /Total/.test(proj));
  assert.ok(proj.includes("-3.5 (55.0%)"), "market: line and the implied probability of its price");
  assert.ok(proj.includes("-7 (" + p1(HOME_COVER) + ")"), "CappingAlpha: model margin (half point) and the model's cover probability at -3.5");
  assert.ok(proj.includes(sp1((HOME_COVER - 0.55) * 100)), "difference = model cover minus implied");
  assert.ok(proj.includes("62.0%") && !proj.includes("99.0%"));
  const ins = html.slice(html.indexOf('id="gm-insights"'), html.indexOf('id="gm-cover"'));
  assert.ok(ins.includes("offense grades B vs") && /rank #4 in the power rankings/.test(ins), "NFL nicknames are plural");
  const cover = html.slice(html.indexOf('id="gm-cover"'));
  assert.ok(cover.includes(p1(HOME_COVER)) && cover.includes(p1(1 - HOME_COVER)));
  assert.ok(/% of Money/.test(cover) && cover.includes("61%") && cover.includes("39%"), "caption only when splits exist");
});

test("without splits the Cover Probability card has no '% of Money' caption; without a model distribution it says so", async () => {
  const mk = (extra) => loadScripts(FILES, { page: "game", globals: { location: { search: "?sport=nfl&game=5", href: "http://localhost/game.html?sport=nfl&game=5" },
    fetch: async (url) => {
      const { path } = urlParts(url);
      const pred = { game_pk: 5, sport: "nfl", home_team_name: HOME, away_team_name: EVIL_AWAY, commence_time: "2026-10-12T17:00:00Z", home_win_prob: 0.6, pred_home_score: 27, pred_away_score: 20, market_spread: -3.5, market_total: 44.5, ...extra };
      return { ok: true, json: async () => (path === "predictions_any" ? [pred] : []) };
    } } });
  const html = await mk({ margin_dist: JSON.stringify(MARGIN) }).buildGamePage();     // uniform margin -10..10, home covers -3.5 when margin >= 4: 7 of 21
  const cover = html.slice(html.indexOf('id="gm-cover"'));
  assert.ok(cover.includes("33.3%") && cover.includes("66.7%"));
  assert.ok(!cover.includes("% of Money"));
  const none = await mk({}).buildGamePage();
  assert.ok(none.slice(none.indexOf('id="gm-cover"')).includes("Model cover probability isn't available for this game yet."));
});

test("Alpha Score card names the player for a prop opportunity (shared oppLabel)", async () => {
  const html = await populated({ propOpp: true }).D.buildGamePage();
  const cards = html.slice(html.indexOf('id="gm-cards"'), html.indexOf('id="gm-projection"'));
  assert.ok(cards.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt; Over 64.5 Rec Yds"));
  assert.ok(!cards.includes("<i>Zed</i>"));
});

test("Matchup tab: NFL prediction block, unit grades, power rankings, sim widget", async () => {
  const html = await tabHtml("matchup");
  assert.match(html, /class="ca-gm-tab on" data-gm-tab="matchup"/);
  for (const t of ["Prediction", "PROJECTED WINNER", "Matchup", "offense", "Power Rankings", "#4", "#19", "Simulation spread"]) assert.ok(html.includes(t), `missing ${t}`);
  assert.ok(html.includes("&lt;b&gt;Crew&lt;/b&gt;") && !html.includes("<b>Crew</b>"));
  assert.ok(!html.includes('id="gm-cards"'), "only the active tab renders");
  assert.ok(!/NaN|undefined/.test(clean(html)));
});

test("Market tab: this game's line moves, best price per side with the book, public betting splits", async () => {
  const html = await tabHtml("market");
  assert.match(html, /class="ca-gm-tab on" data-gm-tab="market"/);
  const m = html.slice(html.indexOf('id="gm-moves"'));
  assert.ok(/-2\.5 → -3\.5/.test(m), "open -> current for this game");
  assert.ok(!m.includes("40 → 45"), "other games' moves are not shown");
  for (const t of ["Best Price by Side", "BetMGM", "FanDuel", "Public Betting"]) assert.ok(html.includes(t), `missing ${t}`);
  assert.ok(!/NaN|undefined/.test(clean(html)));
  const none = await tabHtml("market", { moves: "empty" });
  assert.ok(none.includes("No line moves captured for this game yet."));
  const e404 = await tabHtml("market", { moves: 404 });
  assert.ok(e404.includes("No line moves captured for this game yet.") && e404.includes("Best Price by Side"), "a 404 on line_moves_current never breaks the tab");
});

test("Trends tab: history and betting trends; Players tab: props + boxscore for NFL", async () => {
  const t = await tabHtml("trends");
  assert.ok(t.includes("History") && t.includes("Trends"));
  assert.ok(t.includes("&lt;b&gt;Crew&lt;/b&gt;") && !t.includes("<b>Crew</b>"));
  const p = await tabHtml("players");
  assert.ok(p.includes("Boxscore") && p.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt;") && !p.includes("<i>Zed</i>"));
  assert.ok(!/NaN|undefined/.test(clean(t)) && !/NaN|undefined/.test(clean(p)));
});

test("Players tab says 'NFL-only for now' for a CFB game", async () => {
  const { D } = populated({ search: "?sport=cfb&game=5&tab=players", sport: "cfb" });
  const html = await D.buildGamePage();
  assert.ok(html.includes("Player projections are NFL-only for now."));
});

test("tab state: ?tab= selects the panel, invalid tab falls back to overview, the choice survives a re-render, setting a tab writes the URL", async () => {
  const calls = [];
  const { D } = populated({ search: "?sport=nfl&game=5&tab=market" });
  let html = await D.buildGamePage();
  assert.match(html, /class="ca-gm-tab on" data-gm-tab="market"/);
  html = await D.buildGamePage();                                    // the 5-minute re-render
  assert.match(html, /class="ca-gm-tab on" data-gm-tab="market"/);
  assert.equal(D.gmState().tab, "market");
  const bad = await tabHtml("nonsense");
  assert.match(bad, /class="ca-gm-tab on" data-gm-tab="overview"/);
  const E = loadScripts(FILES, { page: "game", globals: { location: { search: "?sport=nfl&game=5", href: "http://localhost/game.html?sport=nfl&game=5" }, history: { replaceState: (a, b, u) => calls.push(u) } } });
  E.gmSetTab("trends"); E.gmSetTab("overview");
  assert.equal(new URL(calls[0]).search, "?sport=nfl&game=5&tab=trends");
  assert.equal(new URL(calls[1]).search, "?sport=nfl&game=5", "overview is the default: clean URL");
  assert.equal(E.gmState().tab, "overview");
});

test("MLB / NBA game page: hero when a prediction exists plus the sport status line; no tabs", async () => {
  const { D } = populated({ search: "?sport=mlb&game=5", sport: "mlb", pred: { home_team_name: "Boston Red Sox", away_team_name: "New York Yankees" } });
  const html = await D.buildGamePage();
  assert.ok(html.includes("Boston Red Sox") && html.includes("ca-gm-hero"));
  assert.ok(html.includes("MLB model paused (last projections Aug 31, 2026)"));
  assert.ok(!html.includes("ca-gm-tabs") && !html.includes("CAPPINGALPHA READ"));
  const nba = populated({ search: "?sport=nba&game=5", sport: "nba", noPred: true });
  const h2 = await nba.D.buildGamePage();
  assert.ok(h2.includes("NBA model not live yet"));
  assert.ok(!/NaN|undefined/.test(clean(h2)));
});

test("no game, no prediction: friendly states", async () => {
  const none = await populated({ search: "?sport=nfl" }).D.buildGamePage();
  assert.ok(none.includes("No game selected"));
  const missing = await populated({ noPred: true }).D.buildGamePage();
  assert.ok(missing.includes("Game not found"));
});

test("empty sources: every card renders its empty state, no throw, no NaN / undefined", async () => {
  const E = loadScripts(FILES, { page: "game", globals: { location: { search: "?sport=cfb&game=7", href: "http://localhost/game.html?sport=cfb&game=7" },
    fetch: async (url) => ({ ok: true, json: async () => (urlParts(url).path === "predictions_any" ? [{ ...cfbPred, market_spread: null, market_total: null, pred_home_score: null, pred_away_score: null, home_win_prob: null }] : []) }) } });
  const html = await E.buildGamePage();
  for (const t of ["Alpha Score", "Win Probability", "Projected Score", "Model Projection", "No model edge on this game"]) assert.ok(html.includes(t), t);
  assert.ok(!/NaN|undefined/.test(clean(html)));
});

test("a throwing card is isolated: placeholder in its own slot, the rest of the page renders", async () => {
  const logs = []; const quiet = { ...console, error: (...a) => logs.push(a.join(" ")), warn() {} };
  const E = loadScripts(FILES, { page: "game", globals: { console: quiet, location: { search: "?sport=nfl&game=5", href: "http://localhost/game.html?sport=nfl&game=5" },
    fetch: async (url) => { const p = urlParts(url).path; return { ok: true, json: async () => (p === "predictions_any" ? [{ ...cfbPred, sport: "nfl" }] : []) }; } } });
  E.GM_OVERVIEW[1][1] = () => { throw new Error("win prob broke"); };
  const html = await E.buildGamePage();
  assert.ok(html.includes("This panel couldn't load."));
  assert.ok(html.includes("Alpha Score") && html.includes("Projected Score") && html.includes("Model Projection"));
  assert.ok(logs.some((l) => /Win Probability/.test(l)));
});

test("the game route uses the new page; no dark hero / chart tooltip styling comes from the game page", () => {
  const app = src("app.js"), game = src("js/pages/game.js"), css = src("css/theme.css");
  assert.match(app, /page === "game"\) body = await buildGamePage\(\)/);
  assert.match(app, /page === "game"\) wireGamePage\(\)/);
  for (const dark of ["#0a0f16", "#080d13", "#101b28", "chart-tooltip"]) assert.ok(!game.includes(dark) && !css.includes(dark), `${dark} in the game page / theme CSS`);
  assert.ok(!/tr\[data-href\]/.test(game), "no page-level row handlers");
  assert.ok(!/watchlistToggle|starToggle/.test(game), "pages own no star logic");
});

test("possessives for plural team names; legacy sections reused by the page escape every team / player name", () => {
  assert.equal(g.gmPoss("Falcons"), "Falcons'"); assert.equal(g.gmPoss("Alabama"), "Alabama's");
  const r = { sport: "nfl", home_team_name: HOME, away_team_name: EVIL_AWAY, home_win_prob: 0.6, pred_home_score: 24, pred_away_score: 20, market_spread: -3, market_total: 44 };
  const pred = g.nflPredictionSection(r, { actual_total: 44, actual_margin: 4, actual_winner: EVIL_AWAY, winner_correct: true }, "");
  assert.ok(!pred.includes("<b>Crew</b>") && pred.includes("&lt;b&gt;Crew&lt;/b&gt;"));
  const splits = g.splitsSection([{ market: "spread", side: "away", ticket_pct: 40 }, { market: "spread", side: "home", ticket_pct: 60 }], r, "#111", "#222");
  assert.ok(!splits.includes("<b>Crew</b>") && splits.includes("&lt;b&gt;Crew&lt;/b&gt;"));
  const ev = g.evSection([{ sport: "nfl", matchup: `${EVIL_AWAY} @ ${HOME}`, market: "moneyline", side: "away", is_pick: true, ev_best: 0.1 }]);
  assert.ok(!ev.includes("<b>Crew</b>") && ev.includes("&lt;b&gt;Crew&lt;/b&gt;"));
});

test("wireGamePage / gmRedraw are safe without a game view (other pages, early renders)", () => {
  g.wireGamePage();
  g.gmRedraw();
});

test("R24: the hero carries a star per team (id = full team name) and the Players tab a star per player; stars stay shell-owned", async () => {
  const html = await populated().D.buildGamePage();
  const hero = html.slice(html.indexOf("ca-gm-hero"), html.indexOf('id="gm-read"'));
  assert.match(hero, /class="ca-star" data-star-kind="teams" data-star-id="Evil &quot;Q&quot; &lt;b&gt;Crew&lt;\/b&gt;"/, "away team star, escaped");
  assert.match(hero, /class="ca-star" data-star-kind="teams" data-star-id="Kansas City &quot;Chiefs&quot;"/, "home team star");
  const p = await tabHtml("players");
  const props = p.slice(p.indexOf("Projected player props"), p.indexOf("Boxscore"));
  assert.match(props, /data-star-kind="players" data-star-id="O&quot;Brien &lt;i&gt;Zed&lt;\/i&gt;"[^>]*>☆<\/button><b>O&quot;Brien/, "player star before the name");
  assert.ok(!/addEventListener\("click"[^)]*star/i.test(src("js/pages/game.js")), "no page-level star handler (R16)");
});

test("R24: a starred team / player renders filled (watchlist read by the shared starButton); no star option -> no star", async () => {
  const storage = new Map([["ca-watchlist", JSON.stringify({ games: [], teams: [HOME], players: [PLAYER] })]]);
  const L = loadScripts(FILES, { page: "game", storage });
  assert.match(L.starButton("teams", HOME, HOME), /class="ca-star on"/);
  assert.match(L.propsProjectionSection([{ player_id: "p1", name: PLAYER, pos: "WR", team: HOME, market: "rec_yds", mean: 71 }], [], [], [], "", { star: (n) => L.starButton("players", n, n) }), /class="ca-star on" data-star-kind="players"/);
  assert.ok(!L.propsProjectionSection([{ player_id: "p1", name: PLAYER, pos: "WR", team: HOME, market: "rec_yds", mean: 71 }], [], [], []).includes("ca-star"), "no star option -> no star (legacy callers unchanged)");
});

test("one model-spread formatter on the board and the game page: half-point rounding, PK at 0 (no 'ATL -0.2')", () => {
  assert.equal(g.modelLineStr(-0.2), "PK"); assert.equal(g.modelLineStr(0.2), "PK"); assert.equal(g.modelLineStr(-0), "PK");
  assert.equal(g.modelLineStr(-7.1), "-7"); assert.equal(g.modelLineStr(-7.3), "-7.5"); assert.equal(g.modelLineStr(3.26), "+3.5");
  const pred = { sport: "nfl", home_team_name: "Atlanta Falcons", away_team_name: "Tampa Bay Buccaneers", pred_home_score: 24.1, pred_away_score: 23.9, market_spread: -1.5 };
  assert.match(g.gmSpreadCard({ sport: "nfl", r: pred }), /ca-gm-mid">ATL PK</);
  assert.match(g.modelProjectionRows(pred, g.gmOdds(pred, [], null, null, null))[1].model, /^ATL PK$/);
  assert.ok(!src("js/pages/game.js").includes("gmLine1") && src("js/pages/board.js").includes("modelLineStr("), "both pages use the shared formatter");
});

test("Alpha Score card without an opportunity says 'No +EV opportunity on this game.'", async () => {
  const none = await populated({ noEdge: true }).D.buildGamePage();
  assert.ok(none.includes("No +EV opportunity on this game.") && !none.includes("No graded opportunity"));
});
