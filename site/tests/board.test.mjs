import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/board.js", "js/boot.js"];
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/board.js", "js/boot.js"]);

test("search finds teams, matchups and prop players", () => {
  const idx = g.searchIndex([{ sport: "nfl", game_pk: 1, home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions" }],
                            [{ kind: "prop", sport: "nfl", game_pk: 1, playerName: "Amon-Ra St. Brown", marketLabel: "Rec Yds" }]);
  assert.ok(g.searchQuery(idx, "lions").some((r) => r.href === "game.html?sport=nfl&game=1"));
  assert.ok(g.searchQuery(idx, "st. brown").some((r) => /Rec Yds/.test(r.sub)));
  assert.equal(g.searchQuery(idx, "zzz").length, 0);
});

// ---- search: pure helpers -------------------------------------------------------------------------
const PRED = (pk, away, home, extra = {}) => ({ sport: "nfl", game_pk: pk, home_team_name: home, away_team_name: away, commence_time: "2026-10-05T17:00:00Z", ...extra });

test("searchIndex: matchup + both teams per game (one entry each, duplicates collapse), props by player, game-line opps add nothing", () => {
  const idx = g.searchIndex([PRED(1, "Detroit Lions", "Kansas City Chiefs"), PRED(1, "Detroit Lions", "Kansas City Chiefs"), PRED(2, "Ohio State Buckeyes", "Oregon Ducks", { sport: "cfb" })],
    [{ kind: "prop", sport: "nfl", game_pk: 1, playerName: "Josh Allen", marketLabel: "Pass Yds" }, { kind: "prop", sport: "nfl", game_pk: 1, playerName: "Josh Allen", marketLabel: "Pass Yds" },
     { kind: "line", sport: "nfl", game_pk: 1, market: "moneyline" }, null]);
  assert.equal(idx.filter((r) => r.label === "Lions @ Chiefs").length, 1);
  assert.equal(idx.filter((r) => r.label === "Detroit Lions").length, 1);
  assert.equal(idx.filter((r) => r.label === "Kansas City Chiefs").length, 1);
  assert.equal(idx.filter((r) => r.label === "Josh Allen").length, 1);
  assert.equal(idx.length, 4 + 3, "2 games x (matchup + 2 teams) + 1 prop player");
  assert.ok(idx.find((r) => r.label === "Ohio State @ Oregon").href === "game.html?sport=cfb&game=2");
  assert.deepEqual(g.searchIndex(null, null), []);
  assert.deepEqual(g.searchIndex([null, {}], [undefined]), [], "rows without a game or player are skipped");
});

test("searchQuery: case-insensitive on label and sub, trims, first 8 only, empty query gives nothing", () => {
  const preds = Array.from({ length: 12 }, (_, i) => PRED(i + 1, `Away${i}`, `Home${i}`));
  const idx = g.searchIndex(preds, []);
  assert.deepEqual(g.searchQuery(idx, "  AWAY10  ").map((r) => r.label), ["Away10 @ Home10", "Away10", "Home10"], "matchup, away team, and the home team via the shared sub");
  assert.equal(g.searchQuery(idx, "home").length, 8, "capped at 8");
  assert.deepEqual(g.searchQuery(idx, ""), []);
  assert.deepEqual(g.searchQuery(idx, "   "), []);
  assert.deepEqual(g.searchQuery(null, "x"), []);
  assert.ok(g.searchQuery(idx, "nfl").length === 8, "the sub (league) is searchable");
});

test("searchResultsHtml: escapes every data string, marks the active row, empty states", () => {
  const html = g.searchResultsHtml([{ label: 'Evil <b>"Q"</b>', sub: "NFL · <i>x</i>", href: 'game.html?sport=nfl&game="1"' }, { label: "B", sub: "s", href: "h" }], 1);
  assert.ok(html.includes("Evil &lt;b&gt;&quot;Q&quot;&lt;/b&gt;") && html.includes("NFL · &lt;i&gt;x&lt;/i&gt;"));
  assert.ok(!html.includes('"Q"') && !html.includes("<i>x") && !html.includes("<b>\""), "no raw markup or quotes from data");
  assert.ok(html.includes('href="game.html?sport=nfl&amp;game=&quot;1&quot;"'));
  assert.match(html, /class="ca-find-item"[^>]*>(?:(?!ca-find-item).)*Evil/s);
  assert.match(html, /class="ca-find-item on"[^>]*>(?:(?!ca-find-item).)*>B</s, "active index 1 is highlighted");
  assert.ok(g.searchResultsHtml([], 0, "lions").includes("No matches for “lions”"));
  assert.ok(g.searchResultsHtml([], 0, "<x>").includes("&lt;x&gt;"));
  assert.ok(g.searchResultsHtml([], 0, "").includes("Search teams, games and players"));
});

test("searchKeyAction: arrows wrap, Enter opens the active result, Esc closes, other keys do nothing", () => {
  const r = [{ href: "a" }, { href: "b" }, { href: "c" }];
  assert.deepEqual(g.searchKeyAction("ArrowDown", r, 0), { active: 1 });
  assert.deepEqual(g.searchKeyAction("ArrowDown", r, 2), { active: 0 });
  assert.deepEqual(g.searchKeyAction("ArrowUp", r, 0), { active: 2 });
  assert.deepEqual(g.searchKeyAction("Enter", r, 1), { active: 1, href: "b" });
  assert.deepEqual(g.searchKeyAction("Enter", [], 0), { active: 0 });
  assert.deepEqual(g.searchKeyAction("Escape", r, 1), { active: 1, close: true });
  assert.deepEqual(g.searchKeyAction("a", r, 1), { active: 1 });
  assert.deepEqual(g.searchKeyAction("ArrowDown", [], 0), { active: 0 });
});

// ---- board: period + URL state -------------------------------------------------------------------
test("week math: NFL weeks run Tuesday-Monday from the Tuesday after Labor Day; CFB week 1 starts the Tuesday before Labor Day weekend (week 0 = the opener week)", () => {
  assert.equal(g.laborDay(2026), "2026-09-07"); assert.equal(g.laborDay(2025), "2025-09-01");
  assert.equal(g.weekStart("nfl", 2025, 6), "2025-10-07", "the mockup's Week 6 (Oct 9 - Oct 13, 2025) sits inside Oct 7 - Oct 13");
  assert.equal(g.weekOf("nfl", "2025-10-09"), 6); assert.equal(g.weekOf("nfl", "2025-10-13"), 6); assert.equal(g.weekOf("nfl", "2025-10-14"), 7);
  assert.equal(g.weekOf("nfl", "2026-09-09"), 1, "the Wednesday opener is week 1"); assert.equal(g.weekOf("nfl", "2026-09-14"), 1); assert.equal(g.weekOf("nfl", "2026-09-15"), 2);
  assert.equal(g.weekOf("nfl", "2026-10-05"), 4, "Monday night belongs to the week it closes");
  assert.equal(g.weekOf("cfb", "2026-08-29"), 0); assert.equal(g.weekOf("cfb", "2026-09-05"), 1); assert.equal(g.weekOf("cfb", "2026-10-03"), 5);
  assert.equal(g.seasonOf("2027-01-10"), 2026, "January playoff games belong to the season that started the year before"); assert.equal(g.seasonOf("2026-09-01"), 2026);
  assert.equal(g.clampWeek("nfl", 0), 1); assert.equal(g.clampWeek("nfl", 99), 23, "the Super Bowl (week 23) is reachable"); assert.equal(g.clampWeek("cfb", -3), 0);
  assert.equal(g.rangeLabel("2025-10-09", "2025-10-13"), "Oct 9 – Oct 13, 2025"); assert.equal(g.rangeLabel("2026-10-13", "2026-10-13"), "Oct 13, 2026");
  assert.equal(g.rangeLabel("2026-12-30", "2027-01-04"), "Dec 30, 2026 – Jan 4, 2027");
});

test("boardParse / boardQuery: the period (?week= / ?date=) and the view live in the URL (non-default only); rankings only for NFL / CFB", () => {
  assert.deepEqual(g.boardParse("", "nfl"), { week: null, date: null, view: "games" });
  assert.deepEqual(g.boardParse("?week=6", "nfl"), { week: 6, date: null, view: "games" });
  assert.deepEqual(g.boardParse("?week=0&view=rankings", "cfb"), { week: 0, date: null, view: "rankings" });
  assert.deepEqual(g.boardParse("?week=0", "nfl"), { week: null, date: null, view: "games" }, "NFL has no week 0");
  assert.deepEqual(g.boardParse("?week=23", "nfl"), { week: 23, date: null, view: "games" }, "the Super Bowl week");
  assert.deepEqual(g.boardParse("?week=24", "nfl"), { week: null, date: null, view: "games" }, "out of the season");
  assert.deepEqual(g.boardParse("?week=abc&range=week", "nfl"), { week: null, date: null, view: "games" }, "junk and the retired ?range= are ignored");
  assert.deepEqual(g.boardParse("?date=2026-10-13", "mlb"), { week: null, date: "2026-10-13", view: "games" });
  assert.deepEqual(g.boardParse("?date=2026-02-31", "nba"), { week: null, date: null, view: "games" }, "an impossible date");
  assert.deepEqual(g.boardParse("?date=2026-10-13&week=4", "nfl"), { week: 4, date: null, view: "games" }, "NFL reads week, never date");
  assert.deepEqual(g.boardParse("?view=rankings", "nba"), { week: null, date: null, view: "games" }, "no rankings for NBA");
  assert.equal(g.boardQuery({ week: null, date: null, view: "games" }), "");
  assert.equal(g.boardQuery({ week: 6, date: null, view: "games" }), "week=6");
  assert.equal(g.boardQuery({ week: 3, date: null, view: "rankings" }), "week=3&view=rankings");
  assert.equal(g.boardQuery({ week: null, date: "2026-10-13", view: "games" }), "date=2026-10-13");
});

test("boardPeriod: the default is the current week / today; prev / next step by one and stop at the ends of the season", () => {
  const nfl = g.boardPeriod("nfl", { week: null }, "2026-10-05");
  assert.deepEqual([nfl.kind, nfl.week, nfl.from, nfl.to, nfl.isCurrent, nfl.prev, nfl.next], ["week", 4, "2026-09-29", "2026-10-05", true, 3, 5]);
  const past = g.boardPeriod("nfl", { week: 2 }, "2026-10-05");
  assert.deepEqual([past.week, past.from, past.to, past.isCurrent, past.def], [2, "2026-09-15", "2026-09-21", false, 4]);
  assert.equal(g.boardPeriod("nfl", { week: 1 }, "2026-10-05").prev, null); assert.equal(g.boardPeriod("nfl", { week: 23 }, "2026-10-05").next, null); assert.equal(g.boardPeriod("nfl", { week: 22 }, "2026-10-05").next, 23);
  assert.equal(g.weekOf("nfl", "2027-02-14"), 23, "Super Bowl LXI (Feb 14, 2027) falls in week 23"); assert.equal(g.weekStart("nfl", 2026, 23), "2027-02-09");
  assert.equal(g.boardPeriod("nfl", { week: null }, "2026-07-15").week, 1, "before the season: week 1");
  const cfb = g.boardPeriod("cfb", { week: null }, "2026-10-05");
  assert.deepEqual([cfb.week, cfb.from, cfb.to], [5, "2026-09-29", "2026-10-05"]);
  assert.equal(g.boardPeriod("cfb", { week: 0 }, "2026-10-05").prev, null);
  const day = g.boardPeriod("mlb", { date: null }, "2026-10-05"), d2 = g.boardPeriod("nba", { date: "2026-10-13" }, "2026-10-05");
  assert.deepEqual([day.kind, day.date, day.from, day.to, day.isCurrent, day.prev, day.next], ["day", "2026-10-05", "2026-10-05", "2026-10-05", true, "2026-10-04", "2026-10-06"]);
  assert.deepEqual([d2.date, d2.isCurrent, d2.prev, d2.next], ["2026-10-13", false, "2026-10-12", "2026-10-14"]);
});

test("boardSpan: the dates the games fall on, else Thursday-Monday (NFL) / the whole week (CFB) / the day", () => {
  const per = g.boardPeriod("nfl", { week: 4 }, "2026-10-05");
  assert.equal(g.boardSpan(per, [{ date: "2026-10-05" }, { date: "2026-10-01" }, { date: "" }]), "Oct 1 – Oct 5, 2026");
  assert.equal(g.boardSpan(per, []), "Oct 1 – Oct 5, 2026", "Thursday to Monday");
  assert.equal(g.boardSpan(g.boardPeriod("cfb", { week: 4 }, "2026-10-05"), []), "Sep 22 – Sep 28, 2026");
  assert.equal(g.boardSpan(g.boardPeriod("mlb", { date: "2026-10-13" }, "2026-10-05"), []), "Oct 13, 2026");
});

// ---- board: model vs market lines -----------------------------------------------------------------
test("boardLines: every line from the market favourite's side; edge = model minus market (ML in prob points, spread / total in points)", () => {
  const G = (o) => ({ sport: "nfl", game_pk: 7, away: "Buffalo Bills", home: "Atlanta Falcons", ...o });
  // home favoured by the market (-4.5), the model likes the home side more: -6.8, 70%, total 49 vs 47.5
  const L = g.boardLines(G({ pred: { home_win_prob: 0.7, pred_home_score: 28.9, pred_away_score: 20.1, market_spread: -4.5, market_total: 47.5 } }), { mlBy: new Map([["7", { home_price: -210, away_price: 180 }]]) });
  assert.equal(L.fav, "home"); assert.equal(L.favAbbr, "ATL");
  assert.equal(L.mkt.spread, -4.5); assert.equal(L.mkt.ml, -210); assert.equal(L.mkt.total, 47.5);
  assert.ok(L.model.spread === -9 && L.model.ml === -233 && L.model.total === 49, "the model's spread is rounded to the half point (-8.8 -> -9)");
  assert.ok(Math.abs(L.edge.spread - 4.5) < 1e-9, "model margin 9 vs market 4.5: +4.5 points on the favourite");
  const pH = 210 / 310, pA = 100 / 280;
  assert.ok(Math.abs(L.edge.ml - (0.7 - pH / (pH + pA)) * 100) < 1e-9, "ML edge vs the NO-VIG probability (-210 / +180 normalised: 61.9%, not the vigged 67.7%)"); assert.equal(L.mlVig, false);
  assert.ok(Math.abs(L.edge.total - 1.5) < 1e-9);
  // away favoured by the market (+2 home line) while the model prefers the home side: negative edge on the favourite
  const A = g.boardLines(G({ pred: { home_win_prob: 0.55, pred_home_score: 24, pred_away_score: 22, market_spread: 2, market_total: 50 } }), { mlBy: new Map([["7", { home_price: 120, away_price: -140 }]]) });
  assert.equal(A.fav, "away"); assert.equal(A.favAbbr, "BUF");
  assert.equal(A.mkt.spread, -2, "the away favourite's line"); assert.equal(A.mkt.ml, -140);
  assert.ok(Math.abs(A.model.spread - 2) < 1e-9, "the model has the away team a 2-point underdog"); assert.equal(A.model.ml, 122, "fair price for 45% = +122");
  assert.ok(Math.abs(A.edge.spread - -4) < 1e-9 && A.edge.ml < 0 && Math.abs(A.edge.total - -4) < 1e-9);
  // only one side priced: the vigged implied probability (flagged), never a made-up opposite price
  const V = g.boardLines(G({ pred: { home_win_prob: 0.7, market_spread: -4.5 } }), { mlBy: new Map([["7", { home_price: -210, away_price: null }]]) });
  assert.ok(Math.abs(V.edge.ml - (0.7 - 210 / 310) * 100) < 1e-9 && V.mlVig === true, "no price for the other side: vigged fallback, flagged");
  assert.equal(g.boardLines(G({ pred: { home_win_prob: 0.7 } }), {}).mlVig, false, "no price at all: no edge, nothing to flag");
  // no market at all: the model's favourite, dashes for the market and the edge
  const N = g.boardLines(G({ pred: { home_win_prob: 0.4, pred_home_score: 20, pred_away_score: 24 } }), {});
  assert.equal(N.fav, "away"); assert.equal(N.mkt.spread, null); assert.equal(N.mkt.ml, null); assert.equal(N.edge.spread, null); assert.equal(N.edge.ml, null); assert.equal(N.edge.total, null);
  assert.equal(N.model.spread, -4); assert.equal(g.boardLines(G({ pred: {} }), {}).fav, null, "nothing known: no side, no numbers");
  // a final game: the graded row's model numbers + closing prices
  const F = g.boardLines(G({ pred: null, acc: { win_prob: 0.64, predicted_winner: "Atlanta Falcons", home_team_name: "Atlanta Falcons", away_team_name: "Buffalo Bills", pred_margin: 5.4, pred_total: 43.4, market_spread: -6.5, market_total: 40.5 } }),
    { closing: new Map([["7|home", -270], ["7|away", 230]]) });
  assert.deepEqual([F.fav, F.mkt.spread, F.mkt.ml, F.mkt.total], ["home", -6.5, -270, 40.5]);
  assert.ok(F.model.spread === -5.5 && F.model.total === 43.5 && F.model.ml === -178 && Math.abs(F.edge.spread - -1) < 1e-9);
  assert.equal(g.boardLines(G({ pred: null, acc: { win_prob: 0.64, predicted_winner: "Buffalo Bills", home_team_name: "Atlanta Falcons", away_team_name: "Buffalo Bills" } }), {}).model.ml, -178, "predicted winner = the away team: its win prob is the stored one");
});

test("boardScore / boardGames / boardView: scores from margin + total; graded games without a projection still appear; filters and sorts", () => {
  assert.deepEqual(g.boardScore({ actual_margin: 7, actual_total: 33 }), { home: 20, away: 13 });
  assert.deepEqual(g.boardScore({ actual_margin: -5, actual_total: 25 }), { home: 10, away: 15 });
  assert.equal(g.boardScore({ actual_margin: 7, actual_total: 34 }), null, "odd sum: no whole scores"); assert.equal(g.boardScore({}), null); assert.equal(g.boardScore(null), null);
  const P = (pk, h, extra = {}) => ({ sport: "nfl", game_pk: pk, home_team_name: `H${pk}`, away_team_name: `A${pk}`, commence_time: `2026-10-0${h}T17:00:00Z`, game_date: `2026-10-0${h}`, ...extra });
  const D = { preds: [P(2, 4), P(1, 1), P(3, 5, { market_spread: -3, pred_home_score: 30, pred_away_score: 20 })], acc: [{ sport: "nfl", game_pk: 1, game_date: "2026-10-01", actual_winner: "H1", home_team_name: "H1", away_team_name: "A1" },
    { sport: "nfl", game_pk: 9, game_date: "2026-10-02", actual_winner: "H9", home_team_name: "H9", away_team_name: "A9" }, { sport: "nfl", game_pk: 2, actual_winner: null }], opps: [] };
  const games = g.boardGames(D);
  assert.deepEqual(games.map((x) => [x.game_pk, x.final]), [[1, true], [9, true], [2, false], [3, false]], "by date; an ungraded acc row is not final; pk 9 comes from prediction_accuracy alone");
  const s = (o) => ({ market: "all", team: "", time: "", sort: "time", ...o });
  assert.deepEqual(g.boardView(games, D, s({ team: "H2" })).map((x) => x.game_pk), [2]);
  assert.deepEqual(g.boardView(games, D, s({ time: "Sun 1:00 PM" })).map((x) => x.game_pk), [2], "kickoff label filter (ET); a game with no kickoff time never matches");
  assert.deepEqual(g.boardView(games, D, s({ sort: "edge", market: "spread" })).map((x) => x.game_pk), [1, 9, 3, 2], "finished games stay in kickoff order (they show no edge); the upcoming ones sort by the largest edge (only pk 3 has a spread edge)");
  const withAlpha = { ...D, opps: [] }; games[3].opp = { alpha: 80 }; games[2].opp = { alpha: 90 };
  assert.deepEqual(g.boardView(games, withAlpha, s({ sort: "alpha" })).map((x) => x.game_pk), [1, 9, 2, 3], "finished first by time, then the upcoming by alpha");
  assert.deepEqual(g.boardView(games.filter((x) => x.final), withAlpha, s({ sort: "alpha" })).map((x) => x.game_pk), [1, 9], "a period with only finished games: Alpha Score / Edge sorts are a no-op, kickoff order");
});

// ---- populated render ----------------------------------------------------------------------------
const AWAY = 'Evil "Q" <b>Crew</b>', HOME = 'Kansas City "Chiefs"', PLAYER = 'O"Brien <i>Zed</i>';
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };
const quiet = () => { const logs = { warn: [], error: [] }; return { logs, console: { ...console, warn: (...a) => logs.warn.push(a.join(" ")), error: (...a) => logs.error.push(a.join(" ")) } }; };
const NOW = Date.parse("2026-10-05T16:00:00Z");     // Monday 12:00 ET: NFL week 4 (Sep 29 - Oct 5), CFB week 5
class FakeDate extends Date { constructor(...a) { if (a.length) super(...a); else super(NOW); } static now() { return NOW; } }
function populated({ search = "", sport = "nfl", any = null, cur = null, acc = null, ranks = null, fail = [], q = null, extra = {} } = {}) {
  const row = (pk, iso, date, extra = {}) => ({ sport, game_pk: pk, home_team_name: HOME, away_team_name: AWAY, commence_time: iso, game_date: date, home_win_prob: 0.6,
    pred_home_score: 27.2, pred_away_score: 20.4, market_spread: -3, market_total: 41.5, model_version: "nfl-sim-ml-v2", ...extra });
  const g3 = row(3, "2026-10-06T00:15:00Z", "2026-10-05", { market_spread: -1.5, market_total: 46.5, home_win_prob: 0.55 });
  const g20 = row(20, "2026-10-11T17:00:00Z", "2026-10-11", { market_spread: null, market_total: null });
  const anyRows = any || [row(1, "2026-10-02T00:15:00Z", "2026-10-01"), row(2, "2026-10-04T17:00:00Z", "2026-10-04", { market_spread: 2.5, market_total: 44.5, home_win_prob: 0.45, pred_home_score: 21.4, pred_away_score: 22.6 }),
    g3, row(10, "2026-09-27T17:00:00Z", "2026-09-27"), g20];
  const curRows = cur || [g3, g20];
  const accRows = acc || [
    { sport, game_pk: 1, game_date: "2026-10-01", home_team_name: HOME, away_team_name: AWAY, win_prob: 0.6, predicted_winner: HOME, actual_winner: HOME, winner_correct: true, pred_margin: 6.8, actual_margin: 7,
      pred_total: 47.6, actual_total: 41, market_spread: -3, market_total: 41.5, spread_pick_correct: true, total_pick_correct: false },
    { sport, game_pk: 2, game_date: "2026-10-04", home_team_name: HOME, away_team_name: AWAY, win_prob: 0.45, predicted_winner: AWAY, actual_winner: AWAY, winner_correct: true, pred_margin: -1.2, actual_margin: -3,
      pred_total: 44, actual_total: 45, market_spread: 2.5, market_total: 44.5, spread_pick_correct: false, total_pick_correct: null },
    { sport, game_pk: 10, game_date: "2026-09-27", home_team_name: HOME, away_team_name: AWAY, win_prob: 0.6, predicted_winner: HOME, actual_winner: AWAY, winner_correct: false, pred_margin: 6, actual_margin: -4,
      pred_total: 47, actual_total: 50, market_spread: -3, market_total: 41.5, spread_pick_correct: false, total_pick_correct: true }];
  const line = (o) => ({ sport: "nfl", game_pk: 3, matchup: `${AWAY} @ ${HOME}`, market: "moneyline", side: "home", true_prob: 0.7, best_line_implied: 0.5,
    best_price: 100, best_book: "draftkings", ev_best: 0.3, is_pick: true, commence_time: "2026-10-06T00:15:00Z", ...o });
  const evCurrentNfl = [line({}), line({ market: "total", side: "over", ev_best: 0.1, true_prob: 0.58, best_price: -110, best_line_implied: 0.524 }), line({ game_pk: 20, ev_best: 0.02, true_prob: 0.53, best_line_implied: 0.5, commence_time: "2026-10-11T17:00:00Z" })];
  const props = [{ sport: "nfl", game_pk: 3, matchup: `${AWAY} @ ${HOME}`, market: "rec_yds", side: "over", line: 64.5, player_name: PLAYER, model_prob: 0.62,
    best_price: -110, best_book: "fanduel", ev_best: 0.15, is_pick: true, commence_time: "2026-10-06T00:15:00Z" }];
  const rk = ranks || [{ sport, rank: 1, team: "Detroit Lions", rating: 6.2, prev_rank: 2, move: 1, season: 2026, week: 5, su: "4-0", ats: "3-1", units: {} },
    { sport, rank: 2, team: 'Evil "Q" <b>Crew</b>', rating: 4.1, prev_rank: 1, move: -1, season: 2026, week: 5, su: "3-1", ats: "1-3", units: {} }];
  const requested = [];
  const inRange = (qs, key, d) => new URLSearchParams(qs).getAll(key).every((c) => (c.startsWith("gte.") ? d >= c.slice(4) : c.startsWith("lte.") ? d <= c.slice(4) : true));
  const fetch = async (url) => {
    const { path, q: qs } = urlParts(url); requested.push(path + qs);
    if (fail.includes(path)) return { ok: false, status: 500, text: async () => "boom", json: async () => [] };
    let r = [];
    if (path === "predictions_any") r = qs.includes(`sport=eq.${sport}`) ? anyRows.filter((x) => inRange(qs, "game_date", x.game_date)) : [];
    else if (path === "predictions_current") r = qs.includes(`sport=eq.${sport}`) ? curRows : [];
    else if (path === "prediction_accuracy") r = qs.includes(`sport=eq.${sport}`) ? accRows.filter((x) => inRange(qs, "game_date", x.game_date)) : [];
    else if (path === "game_closing_prices") r = [{ game_pk: 1, market: "moneyline", side: "home", close_dec: 1.5 }, { game_pk: 1, market: "moneyline", side: "away", close_dec: 2.7 },
      { game_pk: 2, market: "moneyline", side: "home", close_dec: 2.1 }, { game_pk: 2, market: "moneyline", side: "away", close_dec: 1.8 }];
    else if (path === "game_moneylines_current") r = [{ game_pk: 3, home_price: -125, away_price: 105 }, { game_pk: 20, home_price: -150, away_price: 130 }];
    else if (path === "ev_current") r = qs.includes("sport=eq.nfl") ? evCurrentNfl : [];
    else if (path === "ev_prop_picks_current") r = qs.includes("sport=eq.nfl") ? props : [];
    else if (path === "ev_best_lines") r = [{ game_pk: 3, market: "total", side: "over", line: 47.5 }];
    else if (path === "power_rankings_current") r = rk;
    return { ok: true, json: async () => r };
  };
  const G = loadScripts(FILES, { page: sport, globals: { Date: FakeDate, fetch, location: { search, href: `http://localhost/${sport}.html${search}` }, ...(q ? { console: q.console } : {}), ...extra } });
  return { G, requested };
}
// The th texts of a table's header rows, in order.
const headRows = (html) => [...html.matchAll(/<tr>((?:<th[^>]*>.*?<\/th>)+)<\/tr>/g)].map((m) => [...m[1].matchAll(/<th[^>]*>(.*?)<\/th>/g)].map((x) => x[1].replace(/<[^>]*>/g, "")));
const tableOf = (html, cls) => { const i = html.indexOf(`class="ca-table ca-board-table ${cls}`); return i < 0 ? "" : html.slice(i, html.indexOf("</table>", i)); };
const pks = (html) => [...html.matchAll(/<tr data-href="game\.html\?sport=\w+&(?:amp;)?game=(\d+)"/g)].map((m) => m[1]);

test("title row: the sport name, the subtitle, the period navigator and the date range; the old Games / Power Rankings and All / Today / This Week pills are gone", async () => {
  const html = await populated().G.buildBoardPage("nfl");
  assert.ok(html.includes("<h1>NFL</h1>") && html.includes("Model projections, market edges, and performance tracking."));
  assert.ok(!html.includes("NFL Board") && !html.includes("board-range") && !html.includes("board-view") && !/This Week<\/button>/.test(html), "old pill pairs removed");
  assert.match(html, /data-board-step="-1"[^>]*aria-label="Previous week"/); assert.match(html, /data-board-step="1"[^>]*aria-label="Next week"/);
  assert.ok(html.includes('<option value="4" selected>Week 4</option>') && html.includes('<option value="1">Week 1</option>') && html.includes('<option value="23">Week 23</option>'));
  assert.match(html, /id="board-daterange"[^>]*>.*?<span>Oct 1 – Oct 5, 2026<\/span>/s, "the span of the week's games");
  assert.ok(!/NaN|undefined|Infinity/.test(html));
  const w1 = await populated({ search: "?week=1" }).G.buildBoardPage("nfl");
  assert.match(w1, /data-board-step="-1"[^>]*disabled/, "no previous week before week 1");
  assert.ok(w1.includes("Sep 9 – Sep 14, 2026") || w1.includes("Sep 10 – Sep 14, 2026"), "week 1's span (Thursday-Monday when there are no games)");
});

test("Power Rankings link: in the title row on NFL and CFB only, to ?view=rankings", async () => {
  const nfl = await populated().G.buildBoardPage("nfl"), cfb = await populated({ sport: "cfb" }).G.buildBoardPage("cfb");
  assert.ok(nfl.includes('<a class="ca-bd-rk" href="nfl.html?view=rankings">Power Rankings →</a>'));
  assert.ok(cfb.includes('<a class="ca-bd-rk" href="cfb.html?view=rankings">Power Rankings →</a>'));
  for (const sp of ["mlb", "nba"]) { const h = await populated({ sport: sp, any: [], cur: [], acc: [] }).G.buildBoardPage(sp); assert.ok(!h.includes("Power Rankings") && !h.includes("view=rankings"), `${sp}: no rankings link`); }
  const rk = await populated({ search: "?view=rankings" }).G.buildBoardPage("nfl");
  assert.ok(rk.includes('href="nfl.html"') && rk.includes("← Games") && !rk.includes("data-board-step"), "the rankings view links back and has no period navigator");
});

test("upcoming form: grouped headers Matchup | Kickoff | Market | CappingAlpha | Edge | Alpha Score | View, ML / Spread / Total under each group", async () => {
  const html = await populated({ search: "?week=5" }).G.buildBoardPage("nfl");
  assert.ok(html.includes("<h2>Week 5 Games</h2>") && html.includes("Model projections, current lines, and edges for all Week 5 matchups."));
  const t = tableOf(html, "ca-bd-up");
  assert.deepEqual(headRows(t), [["Matchup", "Kickoff (ET)", "Market", "CappingAlpha", "Edge", "Alpha Score", "View"], ["ML", "Spread", "ML", "Spread", "Total", "ML", "Spread", "Total", "ML", "Spread", "Total"].slice(0, 0).concat(["ML", "Spread", "Total", "ML", "Spread", "Total", "ML", "Spread", "Total"])]);
  assert.deepEqual(pks(html), ["20"]);
  assert.ok(!html.includes("ca-bd-fin"), "no results table for a week with no final");
  assert.ok(html.includes("-150") && !html.includes("Final ·"));
  const cur = await populated().G.buildBoardPage("nfl");
  assert.ok(cur.includes("<h2>This Week's Games</h2>"), "the current week keeps its name");
});

test("past form: Matchup | Date | Final | Closing | CappingAlpha | Result | View with the score, the closing line and a W / L / P chip per market", async () => {
  const html = await populated({ search: "?week=3" }).G.buildBoardPage("nfl");
  assert.ok(html.includes("<h2>Week 3 Results</h2>") && html.includes("Final scores, model picks and results for Week 3."));
  const t = tableOf(html, "ca-bd-fin");
  assert.deepEqual(headRows(t), [["Matchup", "Date", "Final", "Closing", "CappingAlpha", "Result", "View"], ["ML", "Spread", "Total", "ML", "Spread", "Total", "ML", "Spread", "Total"]]);
  assert.deepEqual(pks(html), ["10"]);
  assert.ok(!html.includes("ca-bd-up"), "no upcoming table when every game is final");
  assert.ok(t.includes("Sep 27") && /ca-res L/.test(t), "date and a loss chip (the home pick lost)");
  assert.ok(!/NaN|undefined|Infinity/.test(html));
});

test("a week with finished and upcoming games: finished games first in their own table (Final), then the rest (Upcoming); each form's numbers", async () => {
  const html = await populated().G.buildBoardPage("nfl");
  assert.ok(html.includes("<h2>This Week's Games</h2>") && html.includes("Final · 2") && html.includes("Upcoming · 1"));
  assert.ok(html.indexOf("ca-bd-fin") < html.indexOf("ca-bd-up"));
  assert.deepEqual(pks(tableOf(html, "ca-bd-fin")), ["1", "2"]); assert.deepEqual(pks(tableOf(html, "ca-bd-up")), ["3"]);
});

const rowOf = (table, pk) => { const i = table.indexOf(`game=${pk}"`); return table.slice(i, table.indexOf("</tr>", i)); };
test("final rows: the score (winner bold), the closing prices, the model's lines and a chip per market with the pick", async () => {
  const html = await populated().G.buildBoardPage("nfl"), fin = tableOf(html, "ca-bd-fin");
  const r1 = rowOf(fin, 1);
  assert.match(r1, /<span class="ca-bd-win">Kansas City &quot;Chiefs&quot; 24<\/span>/, "margin 7, total 41 -> home 24, away 17; the home winner is bold");
  assert.ok(r1.includes("17"), "the away score"); assert.ok(r1.includes("-200"), "closing moneyline of the favourite (1.50 decimal)");
  assert.ok(r1.includes("-3</span>") && r1.includes("41.5"), "closing spread and total");
  assert.ok(r1.includes("-7</span>") && r1.includes("47.5"), "the model's spread (pred_margin 6.8 -> -7) and total (47.6 -> 47.5), half-point rounded");
  assert.equal((r1.match(/ca-res W/g) || []).length, 2, "moneyline and spread picks won"); assert.equal((r1.match(/ca-res L/g) || []).length, 1, "the over lost");
  assert.ok(r1.includes('title="Model pick Over 41.5 lost"'));
  const r2 = rowOf(fin, 2);
  assert.ok(/ca-res P/.test(r2) && /ca-res L/.test(r2) && /ca-res W/.test(r2), "a W, an L and a push (null grade) chip");
  assert.ok(r2.includes("-125"), "closing moneyline of the favourite (the away side, 1.80 decimal)");
  assert.ok(!/NaN|undefined|Infinity/.test(fin));
});

test("upcoming rows: market ML / spread / total, the model's, the edge in green / red, the Alpha Score, a link to the game", async () => {
  const html = await populated().G.buildBoardPage("nfl"), up = tableOf(html, "ca-bd-up"), r = rowOf(up, 3);
  assert.ok(r.includes("-125") && r.includes("-1.5") && r.includes("46.5"), "market: best moneyline, spread on the favourite, total");
  assert.ok(r.includes("-122") && r.includes("-7</span>") && r.includes("47.5"), "model: fair ML for 55%, spread from the projected score (half point), total");
  assert.ok(r.includes('<span class="pos">+5.5</span>') && r.includes('<span class="pos">+1.0</span>') && r.includes('<span class="pos">+1.8%</span>'), "edge = model minus market (ML vs the no-vig price: -125 / +105 -> 53.2%), coloured by sign"); 
  assert.match(r, /class="ca-bd-alpha"[^>]*><span class="ca-alpha-cell/, "the game's best opportunity as the Alpha Score");
  assert.ok(r.includes('<a class="ca-go" href="game.html?sport=nfl&game=3"'), "View links to the game page");
  assert.match(r, /data-star-kind="games" data-star-id="3"/);
  assert.ok(r.includes("Mon") && r.includes("8:15 PM"), "kickoff in ET as day + time");
});

test("populated board: names are escaped everywhere (matchup, prop player, rankings team)", async () => {
  const html = await populated().G.buildBoardPage("nfl");
  assert.ok(html.includes("&lt;b&gt;Crew&lt;/b&gt;"), "team name is escaped");
  assert.ok(!html.includes("<b>Crew</b>") && !html.includes("<i>Zed</i>"), "no raw markup from data");
  const P = populated(); P.G.boardSetFilter("nfl", "market", "props");
  const props = await P.G.buildBoardPage("nfl");
  assert.ok(props.includes("ca-bd-props") && props.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt; Over 64.5 Rec Yds"), "the Player Props pill lists the prop opportunities");
  assert.ok(!props.includes("<i>Zed</i>"), "prop player escaped");
  const rk = await populated({ search: "?view=rankings" }).G.buildBoardPage("nfl");
  assert.ok(rk.includes("&lt;b&gt;Crew&lt;/b&gt;") && !rk.includes("<b>Crew</b>"));
});

test("market pills: All Games · Moneyline · Spread · Total, plus Player Props only when the games have prop opportunities", async () => {
  const cur = await populated().G.buildBoardPage("nfl");
  for (const k of ["all", "moneyline", "spread", "total", "props"]) assert.match(cur, new RegExp(`data-pill="board-market" data-key="${k}"`));
  assert.match(cur, /class="ca-pill on" data-pill="board-market" data-key="all"/);
  const past = await populated({ search: "?week=3" }).G.buildBoardPage("nfl");
  assert.ok(!past.includes('data-key="props"') && past.includes('data-key="spread"'), "no props for a past week");
});

test("filters + sort selects: All Teams, All Times, Sort by Start Time (Alpha Score, Edge)", async () => {
  const html = await populated().G.buildBoardPage("nfl");
  assert.ok(html.includes('<option value="" selected>All Teams</option>') && html.includes('<option value="" selected>All Times</option>'));
  assert.ok(html.includes('data-board="sort"') && html.includes('<option value="time" selected>Sort by: Start Time</option>') && html.includes("Sort by: Alpha Score") && html.includes("Sort by: Edge"));
});

test("sort select: Alpha Score / Edge are disabled (and the select reads Start Time) when every game of the period is final; a mixed week keeps them", async () => {
  const mixed = await populated().G.buildBoardPage("nfl");
  assert.ok(!/<option value="(alpha|edge)"[^>]* disabled/.test(mixed), "upcoming games to sort: nothing disabled");
  const P = populated({ search: "?week=3" }), past = await P.G.buildBoardPage("nfl");
  assert.ok(/<td class="ca-bd-final"/.test(past) && !/ca-bd-up/.test(past), "fixture: week 3 is all final");
  assert.ok(/<option value="alpha" disabled[^>]*>Sort by: Alpha Score/.test(past) && /<option value="edge" disabled[^>]*>Sort by: Edge/.test(past) && !/<option value="time"[^>]* disabled/.test(past) && past.includes('<option value="time" selected>'));
});

test("kickoff cell keeps day and time as two spans with a space between (one line when the card is wide, two when narrow)", async () => {
  const html = await populated().G.buildBoardPage("nfl"), r = rowOf(tableOf(html, "ca-bd-up"), 3);
  assert.ok(/<span>Mon<\/span> <span>8:15 PM<\/span>/.test(r), r.slice(0, 600));
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  assert.ok(/@container \(min-width:665px\)\{[^@]*\.ca-bd-up \.ca-bd-time span\{display:inline\}/.test(css), "wide card: Kickoff on one line");
});

test("empty states: a week with no games, MLB paused, NBA without a model", async () => {
  const w = await populated({ search: "?week=19" }).G.buildBoardPage("nfl");
  assert.ok(w.includes("No projections yet for Week 19.") && !w.includes("<tbody>"), "a week still to come: no projections yet");
  const pastEmpty = await populated({ search: "?week=1", any: [], cur: [], acc: [] }).G.buildBoardPage("nfl");
  assert.ok(pastEmpty.includes("No NFL games in Week 1.") && !pastEmpty.includes("No projections yet"), "a past week with nothing: no games");
  const c = await populated({ sport: "cfb", search: "?week=14", any: [], cur: [], acc: [] }).G.buildBoardPage("cfb");
  assert.ok(c.includes("No projections yet for Week 14."));
  const mlbRow = { sport: "mlb", game_pk: 5, home_team_name: "Boston Red Sox", away_team_name: "New York Yankees", commence_time: "2026-08-31T23:00:00Z", game_date: "2026-08-31", pred_home_score: 5, pred_away_score: 4 };
  const mlb = await populated({ sport: "mlb", any: [mlbRow], cur: [], acc: [] }).G.buildBoardPage("mlb");
  assert.ok(mlb.includes("<h1>MLB</h1>") && mlb.includes("MLB model paused since Aug 31 (last projections Aug 31, 2026).") && !mlb.includes("<tbody>"), "the paused date comes from the newest stored projection");
  const mlbOct = await populated({ sport: "mlb", any: [{ ...mlbRow, game_date: "2026-09-12", commence_time: "2026-09-12T23:00:00Z" }], cur: [], acc: [] }).G.buildBoardPage("mlb");
  assert.ok(mlbOct.includes("paused since Sep 12 (last projections Sep 12, 2026)") && !mlbOct.includes("Aug 31"), "no hard-coded date");
  const mlbNone = await populated({ sport: "mlb", any: [], cur: [], acc: [] }).G.buildBoardPage("mlb");
  assert.ok(mlbNone.includes("MLB model paused.") && !/last projections/.test(mlbNone), "no projections at all: no date");
  assert.ok(mlb.includes("<h2>Today's Games</h2>") && mlb.includes("data-board-step") && mlb.includes('type="date"') && mlb.includes("Oct 5, 2026"), "the day navigator and the date");
  const nba = await populated({ sport: "nba", any: [], cur: [], acc: [] }).G.buildBoardPage("nba");
  assert.ok(nba.includes("<h1>NBA</h1>") && nba.includes("No NBA model yet") && !nba.includes("<tbody>"));
  const past = await populated({ sport: "nba", search: "?date=2026-10-13", any: [], cur: [], acc: [] }).G.buildBoardPage("nba");
  assert.ok(past.includes("<h2>Games · Oct 13</h2>") && past.includes("Oct 13, 2026") && past.includes("No NBA model yet"));
  const pastMlb = await populated({ sport: "mlb", search: "?date=2026-08-31", any: [mlbRow], cur: [], acc: [] }).G.buildBoardPage("mlb");
  assert.ok(pastMlb.includes("Model projections and lines for all MLB games on Aug 31.") && !pastMlb.includes("current lines, and edges"), "a past day without graded rows does not promise current lines or edges");
  const todayNba = await populated({ sport: "nba", any: [], cur: [], acc: [] }).G.buildBoardPage("nba");
  assert.ok(todayNba.includes("<h2>Today's Games</h2></div>"), "an empty paused / not-live sport shows no subtitle");
  const rows = [{ sport: "mlb", game_pk: 9, home_team_name: "Boston Red Sox", away_team_name: "New York Yankees", commence_time: "2026-10-05T23:00:00Z", game_date: "2026-10-05", pred_home_score: 5.1, pred_away_score: 3.9, market_spread: -1.5, market_total: 8.5 }];
  const withRows = await populated({ sport: "mlb", any: rows, cur: [], acc: [] }).G.buildBoardPage("mlb");
  assert.ok(withRows.includes("<tbody>") && withRows.includes("game.html?sport=mlb") && !withRows.includes("MLB model paused"), "an MLB day that has projections shows them");
  assert.ok(!/NaN|undefined/.test(withRows));
});

test("three-column scaffold with named, empty mount points for the left and right cards (nothing fake)", async () => {
  const html = await populated().G.buildBoardPage("nfl");
  const order = ["board-left", "board-center", "board-right"].map((id) => html.indexOf(`id="${id}"`));
  assert.ok(order.every((i) => i > 0) && order[0] < order[1] && order[1] < order[2], "left, centre, right");
  assert.deepEqual(JSON.parse(JSON.stringify(populated().G.BOARD_SLOTS)), { left: ["board-season-perf", "board-model-market", "board-market-intel", "board-key-insights"], right: ["board-top-alpha", "board-betting-splits", "board-model-projections"] });
  for (const id of ["board-season-perf", "board-model-market", "board-market-intel", "board-key-insights", "board-top-alpha", "board-betting-splits", "board-model-projections"])
    assert.match(html, new RegExp(`<div class="ca-board-slot" id="${id}" data-board-slot="${id}"></div>`), `${id} is an empty mount point`);
  assert.ok(html.indexOf('id="board-season-perf"') < html.indexOf('id="board-center"') && html.indexOf('id="board-top-alpha"') > html.indexOf('id="board-center"'));
  assert.ok(!html.includes("ca-board-card\" id=\"board-season-perf"), "no placeholder card");
});

test("mount points: a registered renderer fills its slot (and receives D); a throwing one degrades to a placeholder; BOARD_LOADERS add data to D", async () => {
  const q = quiet(), P = populated({ q });
  P.G.BOARD_CARDS["board-top-alpha"] = (D) => `<section class="ca-card" id="board-top-alpha">top ${D.period.week} ${D.preds.length}</section>`;
  P.G.BOARD_CARDS["board-key-insights"] = () => { throw new Error("boom"); };
  P.G.BOARD_LOADERS.push(async (D) => { D.extra = 42; });
  P.G.BOARD_LOADERS.push(async () => { throw new Error("loader boom"); });
  P.G.BOARD_CARDS["board-season-perf"] = (D) => `<section class="ca-card" id="board-season-perf">extra ${D.extra}</section>`;
  const html = await P.G.buildBoardPage("nfl");
  assert.ok(html.includes('<section class="ca-card" id="board-top-alpha">top 4 3</section>') && html.includes("extra 42"));
  assert.ok(html.includes("This panel couldn't load.") && html.includes("ca-board-main"), "one failing card does not blank the page");
  assert.ok(q.logs.error.some((e) => /boom/.test(e)));
});

test("fetch failures degrade to empty states; a failing opportunity feed only drops the Alpha Score", async () => {
  const q = quiet();
  const html = await populated({ fail: ["ev_current", "ev_prop_picks_current", "ev_pnl_daily", "ev_best_lines", "game_moneylines_current", "game_closing_prices"], q }).G.buildBoardPage("nfl");
  assert.ok(html.includes("<tbody>") && html.includes("ca-board") && !/ca-alpha-cell/.test(html), "no opportunities and no prices: the Alpha Score cannot be computed (dash)");
  assert.ok(!/NaN|undefined/.test(html));
  const p = await populated({ fail: ["predictions_any", "predictions_current", "prediction_accuracy"], q }).G.buildBoardPage("nfl");
  assert.ok(p.includes("No NFL games in Week 4."));
});

test("data: the week's rows are fetched by date window (never margin_dist), accuracy by the week, closing prices only for graded games", async () => {
  const P = populated({ search: "?week=3" });
  await P.G.buildBoardPage("nfl");
  const any = P.requested.filter((u) => u.startsWith("predictions_any"));
  assert.equal(any.length, 1);
  assert.ok(any[0].includes("game_date=gte.2026-09-21") && any[0].includes("game_date=lte.2026-09-29"), "week 3 (Sep 22 - Sep 28) widened by a day each side");
  assert.ok(!any[0].includes("margin_dist") && any[0].includes("select=sport,game_pk"));
  assert.ok(P.requested.some((u) => u.startsWith("prediction_accuracy") && u.includes("game_date=gte.2026-09-22") && u.includes("game_date=lte.2026-09-28")));
  assert.ok(P.requested.some((u) => u.startsWith("game_closing_prices")), "closing prices of the graded game");
  assert.ok(!P.requested.some((u) => u.startsWith("game_moneylines_current") || u.startsWith("ev_current") || u.startsWith("predictions_current")), "a past week needs no live feeds");
  const cur = populated();
  await cur.G.buildBoardPage("nfl");
  assert.ok(cur.requested.some((u) => u.startsWith("game_moneylines_current")) && cur.requested.some((u) => u.startsWith("predictions_current")), "the current week reads the live feeds");
});

test("Power Rankings view: the legacy sortable rankingsTable renders in place for the page's sport under the same title row", async () => {
  const P = populated({ sport: "cfb", search: "?view=rankings" });
  const html = await P.G.buildBoardPage("cfb");
  assert.ok(html.includes("<h1>CFB</h1>") && html.includes('class="rk-body"') && html.includes('data-sort="rank"') && html.includes('data-sort="rating"'));
  assert.ok(html.includes('<th data-sort="conf"') && html.includes('id="rk-conf"'), "CFB: the conference column + select");
  assert.ok(P.requested.some((u) => u.startsWith("power_rankings_current") && u.includes("sport=eq.cfb")), "rankings of THIS page's sport");
  assert.ok(!P.requested.some((u) => u.startsWith("predictions_any") || u.startsWith("prediction_accuracy")), "the games data is not fetched in the rankings view");
  assert.ok(html.includes("Season 2026 · Week 5") && !/NaN|undefined/.test(html));
  assert.equal(P.G.rkSport(), "cfb", "the sort wiring resolves the sport from the page, not ?sport=");
  const empty = await populated({ sport: "nfl", search: "?view=rankings", ranks: [] }).G.buildBoardPage("nfl");
  assert.ok(empty.includes("No NFL power rankings yet."));
});

// ---- interaction ----------------------------------------------------------------------------------
function fakeBoardDoc(page = "nfl") {
  const handlers = {}, root = { addEventListener: (t, f) => { handlers[t] = f; } }, els = {}, shell = { innerHTML: "", querySelector: () => null };
  const doc = { body: { dataset: { page }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {}, createElement: () => ({}),
    querySelector: (sel) => (sel === ".ca-board" ? root : sel === ".page-shell" ? shell : null),
    getElementById: (id) => (els[id] = els[id] || { id, outerHTML: "" }), querySelectorAll: () => [], addEventListener() {} };
  return { doc, handlers, els, shell };
}
const hit = (map) => ({ target: { closest: (sel) => map[sel] || null, matches: () => false } });
const step = (dir, disabled = false) => hit({ "[data-board-step]": { dataset: { boardStep: String(dir) }, disabled } });
const settle = () => new Promise((r) => setTimeout(r, 15));
async function wired(sport, opts = {}) {
  const D = fakeBoardDoc(sport), urls = [];
  const P = populated({ sport, ...opts, extra: { document: D.doc, scrollTo() {}, history: { replaceState: (a, b, u) => urls.push(u) } } });
  await P.G.buildBoardPage(sport); P.G.wireBoardPage();
  return { D, urls, P, last: () => new URL(urls[urls.length - 1]) };
}
test("interaction: prev / next / the week select move the period (URL + a re-render); the current week keeps a clean URL; a disabled step does nothing", async () => {
  const { D, urls, last } = await wired("nfl");
  D.handlers.click(step(-1)); await settle();
  assert.equal(last().searchParams.get("week"), "3");
  assert.ok(D.shell.innerHTML.includes('<option value="3" selected>Week 3</option>') && D.shell.innerHTML.includes("Week 3 Results"), "render() redrew the page on week 3, showing its results");
  D.handlers.click(step(-1)); await settle();
  assert.equal(last().searchParams.get("week"), "2");
  D.handlers.change({ target: { value: "4", matches: (sel) => sel === "[data-board-week]", dataset: {} } }); await settle();
  assert.equal(last().search, "", "back to the current week: a clean URL");
  assert.ok(D.shell.innerHTML.includes("This Week's Games"));
  D.handlers.click(step(1)); await settle();
  assert.equal(last().searchParams.get("week"), "5"); assert.ok(D.shell.innerHTML.includes("<h2>Week 5 Games</h2>"));
  const n = urls.length;
  D.handlers.click(step(1, true)); await settle();
  assert.equal(urls.length, n, "a disabled step does nothing");
  D.handlers.change({ target: { value: "2026-10-13", matches: (sel) => sel === "[data-board-date]", dataset: {} } }); await settle();
  assert.equal(urls.length, n, "NFL has no date input: nothing happens");
});

test("interaction (MLB / NBA): the day navigator steps by one day; today keeps a clean URL; the date picker jumps to a day", async () => {
  const { D, urls, last } = await wired("nba", { any: [], cur: [], acc: [] });
  D.handlers.click(step(1)); await settle();
  assert.equal(last().searchParams.get("date"), "2026-10-06"); assert.ok(D.shell.innerHTML.includes("Games · Oct 6") && D.shell.innerHTML.includes("Tue, Oct 6"));
  D.handlers.click(step(-1)); await settle();
  assert.equal(last().search, "", "back to today: a clean URL");
  D.handlers.change({ target: { value: "2026-10-13", matches: (sel) => sel === "[data-board-date]", dataset: {} } }); await settle();
  assert.equal(last().searchParams.get("date"), "2026-10-13"); assert.ok(D.shell.innerHTML.includes("Oct 13, 2026"));
  D.handlers.change({ target: { value: "garbage", matches: (sel) => sel === "[data-board-date]", dataset: {} } }); await settle();
  assert.equal(last().searchParams.get("date"), "2026-10-13", "an invalid date is ignored");
  void urls;
});

test("interaction: a market pill, the team / time selects and the sort redraw the games card from the cache (no refetch)", async () => {
  const { D, P } = await wired("nfl");
  const fetched = P.requested.length, games = () => D.els["board-games"].outerHTML;
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-market", key: "spread" } } }));
  assert.ok(games().includes("ca-bd-em-spread") && /class="ca-pill on" data-pill="board-market" data-key="spread"/.test(games()), "Spread emphasised, the pill is on");
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-market", key: "all" } } }));
  assert.ok(!games().includes("ca-bd-em-"), "All Games: nothing dimmed");
  D.handlers.change({ target: { value: "Thu 8:15 PM", matches: (sel) => sel === "[data-board]", dataset: { board: "time" } } });
  assert.deepEqual(pks(games()), ["1"], "only the Thursday game");
  assert.ok(games().includes('<option value="Thu 8:15 PM" selected>'));
  D.handlers.change({ target: { value: "", matches: (sel) => sel === "[data-board]", dataset: { board: "time" } } });
  D.handlers.change({ target: { value: "Nobody", matches: (sel) => sel === "[data-board]", dataset: { board: "team" } } });
  assert.ok(games().includes("No games match these filters.") && !games().includes("<tbody>"));
  D.handlers.change({ target: { value: "", matches: (sel) => sel === "[data-board]", dataset: { board: "team" } } });
  D.handlers.change({ target: { value: "edge", matches: (sel) => sel === "[data-board]", dataset: { board: "sort" } } });
  assert.deepEqual(pks(games()), ["1", "2", "3"].filter((x) => pks(games()).includes(x)), "sorted within each table");
  assert.equal(P.requested.length, fetched, "no refetch for a filter change");
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-market", key: "props" } } }));
  assert.ok(games().includes("ca-bd-props"), "the Player Props pill swaps the table");
});

test("interaction: the Power Rankings link is a plain link to ?view=rankings; the rankings view renders from the URL", async () => {
  const html = await populated({ sport: "cfb" }).G.buildBoardPage("cfb");
  assert.ok(html.includes('href="cfb.html?view=rankings"'));
  const rk = await populated({ sport: "cfb", search: "?view=rankings&week=3" }).G.buildBoardPage("cfb");
  assert.ok(rk.includes('class="rk-body"') && rk.includes("← Games"));
});

// ---- search overlay wiring ------------------------------------------------------------------------
function fakeSearchDoc() {
  const on = { doc: {}, input: {} };
  const input = { value: "", focus() { this.focused = true; }, addEventListener: (t, f) => { on.input[t] = f; } };
  const items = [];
  const list = { innerHTML: "", querySelectorAll: () => items, querySelector: () => null };
  const overlay = { hidden: true, querySelector: (sel) => (sel === ".ca-find-input" ? input : list) };
  let mounted = false;
  const doc = { body: { dataset: { page: "nfl" }, appendChild() { mounted = true; }, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {},
    createElement: () => ({ set innerHTML(v) { this.html = v; }, firstElementChild: overlay }),
    getElementById: (id) => (id === "ca-find" && mounted ? overlay : null), querySelector: () => null, querySelectorAll: () => [],
    addEventListener: (t, f) => { (on.doc[t] = on.doc[t] || []).push(f); } };
  const fire = (type, e) => (on.doc[type] || []).forEach((f) => f(e));
  return { doc, on, input, list, overlay, fire, isMounted: () => mounted };
}
test("search overlay: the header button opens it, typing lists escaped matches, arrows + Enter navigate, Esc and outside clicks close; data loads lazily", async () => {
  const S = fakeSearchDoc(), loc = { search: "", href: "http://localhost/index.html" }, requested = [];
  const NOW = Date.now(), iso = (h) => new Date(NOW + h * 36e5).toISOString();
  const G = loadScripts(FILES, { page: "nfl", globals: { document: S.doc, location: loc, fetch: async (url) => {
    const { path, q } = urlParts(url); requested.push(path);
    let rows = [];
    if (path === "predictions_current") rows = q.includes("sport=eq.nfl") ? [{ game_pk: 1, home_team_name: HOME, away_team_name: "Detroit Lions", commence_time: iso(30) }] : [];
    else if (path === "ev_prop_picks_current") rows = q.includes("sport=eq.nfl") ? [{ sport: "nfl", game_pk: 1, matchup: `Detroit Lions @ ${HOME}`, market: "rec_yds", side: "over", line: 64.5, player_name: PLAYER, model_prob: 0.62, best_price: -110, best_book: "fanduel", ev_best: 0.15, is_pick: true, commence_time: iso(30) }] : [];
    return { ok: true, json: async () => rows };
  } } });
  G.wireShell();
  assert.equal(requested.length, 0, "wiring the shell fetches nothing");
  assert.equal(S.isMounted(), false);
  S.fire("click", { target: { closest: (sel) => (sel === "[data-search-open]" ? {} : null) } });
  assert.ok(S.isMounted() && S.overlay.hidden === false && S.input.focused, "opened and focused");
  assert.ok(requested.includes("predictions_current"), "the index loads on first open");
  await new Promise((r) => setTimeout(r, 20));
  S.input.value = "lions"; S.on.input.input();
  assert.ok(S.list.innerHTML.includes("Detroit Lions") && S.list.innerHTML.includes("game.html?sport=nfl&amp;game=1"));
  S.input.value = "zed"; S.on.input.input();
  assert.ok(S.list.innerHTML.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt;") && !S.list.innerHTML.includes("<i>Zed"), "prop player, escaped");
  S.input.value = "qqq"; S.on.input.input();
  assert.ok(S.list.innerHTML.includes("No matches for “qqq”"));
  // Enter on the first match navigates to its game
  S.input.value = "lions"; S.on.input.input();
  let prevented = 0;
  S.on.input.keydown({ key: "ArrowDown", preventDefault: () => prevented++ });
  S.on.input.keydown({ key: "ArrowUp", preventDefault: () => prevented++ });
  S.on.input.keydown({ key: "Enter", preventDefault: () => prevented++ });
  assert.equal(loc.href, "game.html?sport=nfl&game=1");
  assert.equal(S.overlay.hidden, true, "closed on navigation");
  // Esc on the input and on the document, and a click outside the box
  S.fire("click", { target: { closest: (sel) => (sel === "[data-search-open]" ? {} : null) } });
  assert.equal(S.overlay.hidden, false);
  S.on.input.keydown({ key: "Escape", preventDefault() {} });
  assert.equal(S.overlay.hidden, true);
  S.fire("click", { target: { closest: (sel) => (sel === "[data-search-open]" ? {} : null) } });
  S.fire("keydown", { key: "Escape" });
  assert.equal(S.overlay.hidden, true);
  S.fire("click", { target: { closest: (sel) => (sel === "[data-search-open]" ? {} : null) } });
  S.fire("click", { target: { closest: () => null } });
  assert.equal(S.overlay.hidden, true, "a click outside the search box closes it");
  assert.equal(S.input.value, "", "closing clears the box");
  const n = requested.filter((p) => p === "predictions_current").length;
  S.fire("click", { target: { closest: (sel) => (sel === "[data-search-open]" ? {} : null) } });
  assert.equal(requested.filter((p) => p === "predictions_current").length, n, "the loaded index is reused within the refresh window");
});

// ---- the row cap ("Show all N games") -----------------------------------------------------------------------
// 25 upcoming games (current week, Oct 1 - Oct 5), one graded game, kickoffs 10 minutes apart from Oct 5 06:00 ET (all inside the current week); the game number is in the home team name.
const manyGames = (n, graded = 0) => Array.from({ length: n }, (_, i) => ({ sport: "nfl", game_pk: 100 + i, home_team_name: `Home ${i}`, away_team_name: `Away ${i}`, commence_time: new Date(Date.parse("2026-10-05T10:00:00Z") + i * 6e5).toISOString(),
  game_date: "2026-10-05", home_win_prob: 0.5 + (i % 7) / 50, pred_home_score: 24, pred_away_score: 20 + (i % 3), market_spread: -3, market_total: 44, model_version: "nfl-sim-ml-v2" }));
const gradedRow = (g) => ({ sport: "nfl", game_pk: g.game_pk, game_date: "2026-10-04", home_team_name: g.home_team_name, away_team_name: g.away_team_name, win_prob: 0.6, predicted_winner: g.home_team_name, actual_winner: g.home_team_name, winner_correct: true,
  pred_margin: 5, actual_margin: 7, pred_total: 47, actual_total: 51, market_spread: -3, market_total: 44, spread_pick_correct: true, total_pick_correct: true });
const rowsIn = (html) => (html.match(/<tr data-href=/g) || []).length;

test("row cap: a table shows 20 rows in the current order and a full-width 'Show all N games' button; <= 20 rows show everything and no button", async () => {
  const gs = manyGames(25), P = populated({ any: gs, cur: gs, acc: [] }), html = await P.G.buildBoardPage("nfl");
  assert.equal(rowsIn(html), 20, "20 of 25");
  assert.deepEqual(pks(html), gs.slice(0, 20).map((g) => String(g.game_pk)), "the first 20 by kickoff, order untouched");
  assert.match(html, /<button type="button" class="ca-bd-more" data-board-more="up" aria-expanded="false">Show all 25 games<\/button>/);
  assert.ok(html.indexOf("</table>") < html.indexOf("ca-bd-more"), "under the table");
  const fit = await populated({ any: manyGames(20), cur: manyGames(20), acc: [] }).G.buildBoardPage("nfl");
  assert.equal(rowsIn(fit), 20); assert.ok(!fit.includes("ca-bd-more"), "exactly 20: no button");
  assert.ok(!(await populated().G.buildBoardPage("nfl")).includes("ca-bd-more"), "the usual week fits");
});

test("row cap: the toggle expands in place (label 'Show fewer'), collapses again, is not in the URL, and a new period starts collapsed", async () => {
  const gs = manyGames(25), els = {}, loc = { search: "", href: "http://localhost/nfl.html" }, handlers = {};
  const doc = { body: { dataset: { page: "nfl" }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {}, createElement: () => ({}),
    querySelector: (sel) => (sel === ".ca-board" ? { addEventListener: (t, f) => { handlers[t] = f; } } : null), getElementById: (id) => (els[id] = els[id] || { id, outerHTML: "" }), querySelectorAll: () => [], addEventListener() {} };
  const P = populated({ any: gs, cur: gs, acc: [], extra: { document: doc } }), G = P.G;
  await G.buildBoardPage("nfl"); G.wireBoardPage();
  const click = () => handlers.click({ target: { closest: (sel) => (sel === "[data-board-more]" ? { dataset: { boardMore: "up" } } : null) } });
  click();
  assert.equal(rowsIn(els["board-games"].outerHTML), 25, "all 25 after the click");
  assert.ok(els["board-games"].outerHTML.includes('aria-expanded="true">Show fewer</button>') && els["board-games"].outerHTML.includes('id="board-games"'));
  assert.equal(loc.search, "", "no URL state");
  click();
  assert.equal(rowsIn(els["board-games"].outerHTML), 20); assert.ok(els["board-games"].outerHTML.includes("Show all 25 games"));
  click(); G.boardGoPeriod("nfl", 3, "2026-10-05");
  assert.deepEqual({ ...G.boardState("nfl").more }, {}, "another week starts collapsed");
});

test("row cap: filters and sort run on ALL rows first, the cap trims afterwards; a mixed week caps the Final and the Upcoming table separately", async () => {
  const gs = manyGames(25);
  // team filter: game 23 is past the first 20 by kickoff, but the filter runs on all 25 so it is the one row and no button appears
  const P = populated({ any: gs, cur: gs, acc: [], extra: {} }), D = await P.G.boardLoad("nfl", "games");
  const set = (k, v) => P.G.boardSetFilter("nfl", k, v); set("team", "Home 23");
  const one = P.G.boardGamesCard(D); assert.equal(rowsIn(one), 1); assert.ok(one.includes("game=123") && !one.includes("ca-bd-more"));
  set("team", ""); set("sort", "alpha"); set("more", {});
  const sorted = P.G.boardGamesCard(D), view = P.G.boardView(P.G.boardGames(D), D, { market: "all", team: "", time: "", sort: "alpha" }).map((g) => String(g.game_pk));
  assert.deepEqual(pks(sorted), view.slice(0, 20), "the cap takes the top 20 of the SORTED list");
  set("sort", "time");
  // mixed: 22 graded + 24 upcoming
  const fin = manyGames(46).slice(0, 22).map((g) => ({ ...g, game_date: "2026-10-04", commence_time: new Date(Date.parse("2026-10-04T17:00:00Z") + g.game_pk * 6e4).toISOString() })), up = manyGames(46).slice(22).map((g) => ({ ...g, game_pk: g.game_pk + 500 }));
  const M = populated({ any: [...fin, ...up], cur: up, acc: fin.map(gradedRow) }), MD = await M.G.boardLoad("nfl", "games"), card = M.G.boardGamesCard(MD);
  assert.equal(rowsIn(card), 40, "20 + 20");
  assert.ok(card.includes("Final · 22") && card.includes("Upcoming · 24"), "section counts are the full counts");
  assert.ok(card.includes('data-board-more="fin" aria-expanded="false">Show all 22 games') && card.includes('data-board-more="up" aria-expanded="false">Show all 24 games'), "one button per table");
  M.G.boardSetFilter("nfl", "more", { fin: true });
  const half = M.G.boardGamesCard(MD);
  assert.equal(rowsIn(half), 42, "Final expanded (22) + Upcoming still 20");
  assert.ok(half.includes("Show fewer") && half.includes("Show all 24 games"));
});

// ---- CSS contract --------------------------------------------------------------------------------
test("CSS contract: every ca-* class the board, rankings and settings pages emit exists in theme.css", async () => {
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  const classesIn = (html) => [...new Set([...html.matchAll(/class="([^"]*)"/g)].flatMap((m) => m[1].split(/\s+/)).filter((c) => /^ca-/.test(c)))];
  const hooks = new Set(["ca-minibars", "ca-spark", "ca-donut", "ca-bar"]);
  const defined = (c) => new RegExp(`\\.${c}(?![\\w-])`).test(css);
  const missing = (html) => classesIn(html).filter((c) => !hooks.has(c) && !defined(c));
  const pages = [await populated().G.buildBoardPage("nfl"), await populated({ sport: "cfb", search: "?view=rankings" }).G.buildBoardPage("cfb"),
    await populated({ sport: "mlb", any: [], cur: [], acc: [] }).G.buildBoardPage("mlb")];
  const A = loadScripts(FILES, { page: "rankings", globals: { location: { search: "?sport=cfb", href: "http://localhost/rankings.html?sport=cfb" },
    fetch: async () => ({ ok: true, json: async () => [{ sport: "cfb", rank: 1, team: "Oregon", rating: 5, conf: 5, season: 2026, week: 5, units: {} }] }) } });
  pages.push(await A.buildRankings(), A.buildSettings(), g.searchOverlayHtml());
  for (const html of pages) assert.deepEqual(missing(html), [], "all ca-* classes are styled");
  for (const c of classesIn(pages[5])) assert.equal(css.split(new RegExp(`(?:^|\\n)\\.${c}\\{`)).length - 1, 1, `.${c} is defined once (the overlay classes must not collide with the shared +EV / Track search field)`);
  for (const extra of [await populated({ search: "?week=3" }).G.buildBoardPage("nfl"), await populated({ search: "?week=5" }).G.buildBoardPage("nfl")]) assert.deepEqual(missing(extra), [], "past and upcoming forms are styled");
  assert.ok(pages[0].includes("ca-board-table") && classesIn(pages[4]).includes("ca-set"), "found the new classes");
});

test("rankings.html and settings.html keep their own pages inside the new shell (title via pageTitle, content in a .ca-card, wiring hooks intact)", async () => {
  const A = loadScripts(FILES, { page: "rankings", globals: { location: { search: "?sport=nfl", href: "http://localhost/rankings.html?sport=nfl" },
    fetch: async () => ({ ok: true, json: async () => [{ sport: "nfl", rank: 1, team: "Detroit Lions", rating: 5, season: 2026, week: 5, units: {} }] }) } });
  const rk = await A.buildRankings();
  assert.ok(rk.includes('class="ca-title-row"') && rk.includes("<h1>Power Rankings</h1>") && /class="ca-card[^"]*"/.test(rk));
  assert.ok(rk.includes('class="rk-body"') && rk.includes('data-sort="team"') && rk.includes("rankings.html?sport=cfb"));
  assert.ok(!rk.includes("page-heading") && !/(?<![\w-])table-wrap/.test(rk), "legacy heading and table-wrap are gone");
  const st = A.buildSettings();
  assert.ok(st.includes('class="ca-title-row"') && st.includes("<h1>Settings</h1>") && st.includes("Sportsbooks, unit size, bankroll, Kelly and minimum EV."));
  assert.match(st, /<main class="settings"/);
  for (const hook of ['id="set-unit"', 'id="set-bankroll"', 'id="set-minev"', 'id="set-minev-out"', "set-books-min", "set-saved", "set-reset", 'data-books="all"', 'data-books="none"', "data-unit=", "data-kelly=", "data-book="])
    assert.ok(st.includes(hook), `settings keeps ${hook}`);
  assert.ok(/<section class="ca-card ca-set"/.test(st) && !st.includes("set-card") && !st.includes("page-heading"));
});

test("a blowout's extreme moneyline reads compactly (-100k) with the exact price on hover; ordinary prices are untouched", () => {
  assert.equal(g.boardOdds(-210), "-210"); assert.equal(g.boardOdds(105), "+105"); assert.ok(g.boardOdds(null).includes("—"));
  assert.equal(g.boardOdds(-100000), '<span title="-100000">-100k</span>'); assert.equal(g.boardOdds(25000), '<span title="+25000">+25k</span>');
});

// ---- Alpha Score on every upcoming row, ML odds, matchup, URL hygiene ----------------------------
test("boardAlpha: the game's opportunity score, else the better moneyline side (model prob vs best price) through alphaScore; null when it cannot be computed", () => {
  const G = (o) => ({ sport: "nfl", game_pk: 7, away: "Buffalo Bills", home: "Atlanta Falcons", ...o });
  const D = (m) => ({ mlBy: new Map([["7", m]]) });
  const hi = g.boardAlpha(G({ pred: { home_win_prob: 0.7 } }), D({ home_price: 100, away_price: -120 }));
  assert.equal(hi.score, g.alphaScore({ evPct: 40, edgePp: 20 })); assert.equal(hi.score, 95); assert.equal(hi.tier, "HIGH");
  assert.ok(/Falcons ML \+100|ATL ML \+100/.test(hi.basis));
  const lo = g.boardAlpha(G({ pred: { home_win_prob: 0.6 } }), D({ home_price: -150, away_price: 130 }));
  assert.equal(lo.score, 45); assert.equal(lo.tier, null, "below 65: no tier");
  const away = g.boardAlpha(G({ pred: { home_win_prob: 0.3 } }), D({ home_price: -250, away_price: 120 }));
  assert.ok(away.score > g.boardAlpha(G({ pred: { home_win_prob: 0.6 } }), D({ home_price: -250, away_price: 120 })).score - 100 && /BUF|Bills/.test(away.basis), "the away side can be the better one");
  assert.equal(g.boardAlpha(G({ opp: { alpha: 81, tier: "STRONG", kind: "line", market: "total", side: "over", sport: "nfl" } }), {}).score, 81, "a tiered opportunity wins");
  assert.equal(g.boardAlpha(G({ pred: { home_win_prob: 0.7 } }), {}), null, "no price");
  assert.equal(g.boardAlpha(G({ pred: {} }), D({ home_price: 100, away_price: -120 })), null, "no model probability");
  assert.equal(g.boardAlpha(G({ pred: { home_win_prob: 0.7 }, final: true }), D({ home_price: 100, away_price: -120 })), null, "a final game has no Alpha Score");
  assert.equal(g.boardAlpha(G({ pred: { home_win_prob: 0.7 } }), D({ home_price: null, away_price: "" })), null);
  assert.ok(g.boardAlphaCell({ score: 45 }).includes("ca-alpha-cell lo") && !g.boardAlphaCell({ score: 45 }).includes("hi"), "neutral grey below 65");
  assert.ok(g.boardAlphaCell({ score: 70 }).includes("ca-alpha-cell") && !g.boardAlphaCell({ score: 70 }).includes(" lo"), "tier colours from 65");
  assert.ok(g.boardAlphaCell({ score: 90 }).includes("ca-alpha-cell hi"));
});

test("Alpha Score column: every upcoming row with a price and a model probability shows one (grey below 65); a dash only when it cannot be computed", async () => {
  const html = await populated({ search: "?week=5" }).G.buildBoardPage("nfl"), r = rowOf(tableOf(html, "ca-bd-up"), 20);
  assert.match(r, /class="ca-bd-alpha" title="[^"]*ML[^"]*"><span class="ca-alpha-cell lo">45<\/span>/, "game 20 has no tiered opportunity: the model's moneyline score, neutral");
  const cur = await populated().G.buildBoardPage("nfl");
  assert.match(rowOf(tableOf(cur, "ca-bd-up"), 3), /class="ca-alpha-cell/, "game 3 keeps its opportunity score");
  const noPrice = await populated({ search: "?week=5", fail: ["game_moneylines_current"] }).G.buildBoardPage("nfl");
  assert.ok(rowOf(tableOf(noPrice, "ca-bd-up"), 20).includes('class="ca-bd-alpha"><span class="muted">—</span>'), "no price: a dash");
  const P = populated({ search: "?week=4" }); P.G.boardSetFilter("nfl", "sort", "alpha");
  assert.ok((await P.G.buildBoardPage("nfl")).includes('value="alpha" selected'), "Alpha sort uses the computed score");
});

test("4-digit moneylines carry the exact price on hover; the narrow-card ML columns are wide enough for -1439", () => {
  assert.equal(g.boardOdds(-1439), '<span title="-1439">-1439</span>'); assert.equal(g.boardOdds(1250), '<span title="+1250">+1250</span>');
  assert.equal(g.boardOdds(-999), "-999");
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  assert.match(css, /@container \(max-width:780px\)\{\n\.ca-bd-up col:nth-child\(1\)\{width:21%\}\.ca-bd-up col:nth-child\(2\)\{width:7\.5%\}\.ca-bd-up col:nth-child\(3\)\{width:6\.5%\}/, "mid-width ML column widened");
});

test("the matchup keeps its '@' when it stacks (home line reads '@ HOME')", async () => {
  const html = await populated().G.buildBoardPage("nfl");
  assert.ok(/<span class="ca-bd-side"><i>@<\/i><img|<span class="ca-bd-side"><i>@<\/i>[^<]*<b>/.test(html), "the @ sits inside the home side, so the stacked layout shows it");
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  assert.ok(!/\.ca-bd-pair i\{display:none\}/.test(css), "the @ is never hidden");
});

test("the URL: unknown params survive a period change; an invalid ?week= is normalised away on load", async () => {
  const urls = [];
  const P = populated({ search: "?week=99&utm=x&view=bogus", extra: { history: { replaceState: (a, b, u) => urls.push(u) }, location: { search: "?week=99&utm=x&view=bogus", href: "http://localhost/nfl.html?week=99&utm=x&view=bogus" } } });
  await P.G.buildBoardPage("nfl");
  assert.equal(urls.length, 1, "normalised once on load");
  const u = new URL(urls[0]); assert.equal(u.searchParams.get("utm"), "x"); assert.equal(u.searchParams.get("week"), null); assert.equal(u.searchParams.get("view"), null);
  const ok = []; const Q = populated({ search: "?week=2&utm=x", extra: { history: { replaceState: (a, b, u2) => ok.push(u2) }, location: { search: "?week=2&utm=x", href: "http://localhost/nfl.html?week=2&utm=x" } } });
  await Q.G.buildBoardPage("nfl");
  assert.equal(ok.length, 0, "a valid URL is left alone");
  Q.G.boardGoPeriod("nfl", 3, "2026-10-05");
  const v = new URL(ok[0]); assert.equal(v.searchParams.get("week"), "3"); assert.equal(v.searchParams.get("utm"), "x", "foreign params preserved");
  Q.G.boardGoPeriod("nfl", 4, "2026-10-05");
  assert.equal(new URL(ok[1]).search, "?utm=x", "the default week removes ?week= only");
});
