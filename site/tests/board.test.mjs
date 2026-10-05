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

// ---- board: URL state ----------------------------------------------------------------------------
test("boardParse / boardQuery: range + view live in the URL (non-default only); rankings only for NFL / CFB", () => {
  assert.deepEqual(g.boardParse("", "nfl"), { range: "all", view: "games" });
  assert.deepEqual(g.boardParse("?range=week&view=rankings", "cfb"), { range: "week", view: "rankings" });
  assert.deepEqual(g.boardParse("?range=today", "mlb"), { range: "today", view: "games" });
  assert.deepEqual(g.boardParse("?view=rankings", "nba"), { range: "all", view: "games" }, "no rankings for NBA");
  assert.deepEqual(g.boardParse("?range=bogus&view=bogus", "nfl"), { range: "all", view: "games" });
  assert.equal(g.boardQuery({ range: "all", view: "games" }), "");
  assert.equal(g.boardQuery({ range: "week", view: "games" }), "range=week");
  assert.equal(g.boardQuery({ range: "today", view: "rankings" }), "range=today&view=rankings");
});

// ---- board: range filter -------------------------------------------------------------------------
test("boardFilter: All keeps everything; Today = the ET date; This Week = today through +6 days (ET), never the past", () => {
  const at = (iso) => ({ commence_time: iso });
  const rows = [at("2026-10-03T23:30:00Z"), at("2026-10-04T13:30:00Z"), at("2026-10-04T23:30:00Z"), at("2026-10-05T01:00:00Z"), at("2026-10-10T20:00:00Z"), at("2026-10-11T20:00:00Z"), { game_date: "2026-10-04" }, {}];
  // 2026-10-05T01:00Z is 9:00 PM ET on Oct 4
  assert.equal(g.boardFilter(rows, "all", "2026-10-04").length, 8);
  assert.deepEqual(g.boardFilter(rows, "today", "2026-10-04").map((r) => r.commence_time || r.game_date), ["2026-10-04T13:30:00Z", "2026-10-04T23:30:00Z", "2026-10-05T01:00:00Z", "2026-10-04"]);
  assert.deepEqual(g.boardFilter(rows, "week", "2026-10-04").map((r) => r.commence_time || r.game_date),
    ["2026-10-04T13:30:00Z", "2026-10-04T23:30:00Z", "2026-10-05T01:00:00Z", "2026-10-10T20:00:00Z", "2026-10-04"], "Oct 4..Oct 10 inclusive; Oct 3 (past) and Oct 11 (day 7) are out; undated rows never match a range");
  assert.deepEqual(g.boardFilter(null, "all", "2026-10-04"), []);
});

// ---- board: model vs market ----------------------------------------------------------------------
test("boardModel: projected score, model spread (home line) and total from the projection; pred_margin / pred_total win when present", () => {
  const m = g.boardModel({ pred_home_score: 25.35, pred_away_score: 27.12, market_spread: 4.5, market_total: 46.5 });
  assert.deepEqual([m.away, m.home], [27, 25]);
  assert.equal(m.spread, 2, "home projected to lose by 1.77 -> home +2 (nearest half point)");
  assert.equal(m.total, 52.5);
  assert.equal(m.mktSpread, 4.5); assert.equal(m.mktTotal, 46.5);
  const p = g.boardModel({ pred_home_score: 20, pred_away_score: 10, pred_margin: 6.2, pred_total: 40.1, market_spread: -3, market_total: 41.5 });
  assert.equal(p.spread, -6, "pred_margin 6.2 -> home -6.0 (nearest half point), not the score gap");
  assert.equal(p.total, 40);
  const n = g.boardModel({ home_win_prob: 0.5 });
  assert.deepEqual([n.away, n.home, n.spread, n.total, n.mktSpread, n.mktTotal], [null, null, null, null, null, null]);
  assert.equal(g.boardModel({ pred_home_score: 24, pred_away_score: 24, market_spread: "", market_total: null }).spread, 0, "an even projection is a pick'em (never -0)");
  assert.equal(Object.is(g.boardModel({ pred_home_score: 24, pred_away_score: 24 }).spread, -0), false);
  assert.equal(g.boardModel(null).spread, null);
});

// ---- populated render ----------------------------------------------------------------------------
const AWAY = 'Evil "Q" <b>Crew</b>', HOME = 'Kansas City "Chiefs"', PLAYER = 'O"Brien <i>Zed</i>';
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };
const quiet = () => { const logs = { warn: [], error: [] }; return { logs, console: { ...console, warn: (...a) => logs.warn.push(a.join(" ")), error: (...a) => logs.error.push(a.join(" ")) } }; };
function populated({ search = "", sport = "nfl", rows = null, ranks = null, fail = [], q = null } = {}) {
  const NOW = Date.now(), iso = (h) => new Date(NOW + h * 36e5).toISOString();
  const pred = (pk, h, extra = {}) => ({ game_pk: pk, home_team_name: HOME, away_team_name: AWAY, commence_time: iso(h), home_win_prob: 0.6,
    pred_home_score: 27.2, pred_away_score: 20.4, market_spread: -3, market_total: 41.5, ...extra });
  const preds = rows || [pred(1, 30), pred(2, 50, { market_spread: null, market_total: null }), pred(3, 24 * 12), pred(4, 0, { pred_home_score: null, pred_away_score: null })];
  const line = (o) => ({ sport: "nfl", game_pk: 1, matchup: `${AWAY} @ ${HOME}`, market: "moneyline", side: "home", true_prob: 0.7, best_line_implied: 0.5,
    best_price: 100, best_book: "draftkings", ev_best: 0.3, is_pick: true, commence_time: iso(30), ...o });
  const evCurrentNfl = [line({}), line({ market: "total", side: "over", ev_best: 0.1, true_prob: 0.58, best_price: -110, best_line_implied: 0.524 }), line({ game_pk: 3, ev_best: 0.02, true_prob: 0.53, best_line_implied: 0.5 })];
  const props = [{ sport: "nfl", game_pk: 2, matchup: `${AWAY} @ ${HOME}`, market: "rec_yds", side: "over", line: 64.5, player_name: PLAYER, model_prob: 0.62,
    best_price: -110, best_book: "fanduel", ev_best: 0.15, is_pick: true, commence_time: iso(50) }];
  const rk = ranks || [{ sport, rank: 1, team: "Detroit Lions", rating: 6.2, prev_rank: 2, move: 1, season: 2026, week: 5, su: "4-0", ats: "3-1", units: {} },
    { sport, rank: 2, team: 'Evil "Q" <b>Crew</b>', rating: 4.1, prev_rank: 1, move: -1, season: 2026, week: 5, su: "3-1", ats: "1-3", units: {} }];
  const requested = [];
  const fetch = async (url) => {
    const { path, q: qs } = urlParts(url); requested.push(path + qs);
    if (fail.includes(path)) return { ok: false, status: 500, text: async () => "boom", json: async () => [] };
    let r = [];
    if (path === "predictions_current") r = qs.includes(`sport=eq.${sport}`) ? preds : [];
    else if (path === "ev_current") r = qs.includes("sport=eq.nfl") ? evCurrentNfl : [];
    else if (path === "ev_prop_picks_current") r = qs.includes("sport=eq.nfl") ? props : [];
    else if (path === "ev_best_lines") r = [{ game_pk: 1, market: "total", side: "over", line: 41.5 }];
    else if (path === "power_rankings_current") r = rk;
    return { ok: true, json: async () => r };
  };
  const G = loadScripts(FILES, { page: sport, globals: { fetch, location: { search, href: `http://localhost/${sport}.html${search}` }, ...(q ? { console: q.console } : {}) } });
  return { G, requested, NOW };
}

test("populated NFL board: a row per game with the model projection beside the market; best opportunity is the game's top alpha, else a dash", async () => {
  const { G } = populated();
  const html = await G.buildBoardPage("nfl");
  assert.ok(html.includes("<h1>NFL Board</h1>") && html.includes("Every game with the model's projection and best market."));
  const rows = html.match(/<tr data-href="game\.html\?sport=nfl&(?:amp;)?game=\d+"/g) || [];
  assert.equal(rows.length, 4, "all four games (All range)");
  const row = (pk) => html.slice(html.indexOf(`game=${pk}"`, html.indexOf("<tbody>")), html.indexOf("</tr>", html.indexOf(`game=${pk}"`, html.indexOf("<tbody>"))));
  const r1 = row(1);
  assert.ok(r1.includes("20–27"), "projected score away–home, rounded");
  assert.ok(r1.includes("-7</b>"), "model spread: home projected to win by 6.8 -> home -7 (nearest half-point)");
  assert.ok(r1.includes("<small>Mkt -3</small>"), "the market spread beside it");
  assert.ok(r1.includes("<b>47.5</b><small>Mkt 41.5</small>"), "model total (47.6 -> 47.5) vs market total");
  assert.ok(r1.includes("ca-alpha-cell") && /ca-conf (HIGH|STRONG|MEDIUM)/.test(r1), "best opportunity carries Alpha and Confidence");
  assert.ok(r1.includes("ML"), "the higher-alpha opportunity (the moneyline, not the total)");
  assert.match(r1, /data-star-kind="games" data-star-id="1"/);
  const r2 = row(2);
  assert.ok(r2.includes("Over 64.5 Rec Yds"), "a prop can be a game's best opportunity");
  assert.equal((r2.match(/<small>Mkt —<\/small>/g) || []).length, 2, "no market lines -> dashes, never numbers");
  const r3 = row(3);
  assert.ok(r3.includes("—"), "an untiered game shows an em dash for Best Opportunity");
  assert.ok(!/ca-alpha-cell/.test(r3), "alpha 53 is not an opportunity (no tier)");
  const r4 = row(4);
  assert.ok(!/NaN|undefined/.test(r4) && r4.includes("—"), "a game with no projection renders dashes");
  assert.ok(!/NaN|undefined|Infinity/.test(html), "no NaN / undefined leaks");
});

test("populated board: names are escaped everywhere (matchup, prop player, rankings team)", async () => {
  const { G } = populated();
  const html = await G.buildBoardPage("nfl");
  assert.ok(html.includes("&lt;b&gt;Crew&lt;/b&gt;"), "team name is escaped");
  assert.ok(!html.includes("<b>Crew</b>") && !html.includes("<i>Zed</i>"), "no raw markup from data");
  assert.ok(html.includes("&quot;"), "quotes are escaped");
  const rk = await populated({ search: "?view=rankings" }).G.buildBoardPage("nfl");
  assert.ok(rk.includes("&lt;b&gt;Crew&lt;/b&gt;") && !rk.includes("<b>Crew</b>"));
});

test("populated board: range pills filter from the URL and survive a second build; the table is sorted by kickoff", async () => {
  const { G } = populated({ search: "?range=today" });
  const html = await G.buildBoardPage("nfl");
  assert.match(html, /class="ca-pill on" data-pill="board-range" data-key="today"/);
  const pks = [...html.matchAll(/<tr data-href="game\.html\?sport=nfl&(?:amp;)?game=(\d+)"/g)].map((m) => m[1]);
  assert.deepEqual(pks, ["4"], "only the game kicking off now; the others are +30h or later");
  const wk = populated({ search: "?range=week" });
  const w = await wk.G.buildBoardPage("nfl"), w2 = await wk.G.buildBoardPage("nfl");
  for (const h of [w, w2]) {
    const ids = [...h.matchAll(/<tr data-href="game\.html\?sport=nfl&(?:amp;)?game=(\d+)"/g)].map((m) => m[1]);
    assert.ok(ids.includes("1") && ids.includes("2") && !ids.includes("3"), "week = next 7 days: +30h and +50h in, +12d out");
    assert.match(h, /class="ca-pill on" data-pill="board-range" data-key="week"/);
  }
  const all = await populated().G.buildBoardPage("nfl");
  assert.deepEqual([...all.matchAll(/<tr data-href="game\.html\?sport=nfl&(?:amp;)?game=(\d+)"/g)].map((m) => m[1]), ["4", "1", "2", "3"], "by kickoff");
  assert.match(all, /class="ca-pill on" data-pill="board-range" data-key="all"/);
});

test("populated board: a range with no games gives an empty state, not an empty table", async () => {
  const html = await populated({ rows: [] }).G.buildBoardPage("nfl");
  assert.ok(html.includes("No NFL games on the board right now."));
  assert.ok(!html.includes("<tbody>"));
});

test("NFL / CFB boards: the Games · Power Rankings pill group; MLB / NBA have none", async () => {
  const nfl = await populated().G.buildBoardPage("nfl");
  assert.match(nfl, /data-pill="board-view" data-key="games"/); assert.match(nfl, /data-pill="board-view" data-key="rankings"/);
  assert.ok(nfl.includes("Power Rankings"));
  const mlb = await populated({ sport: "mlb", rows: [] }).G.buildBoardPage("mlb");
  assert.ok(!/board-view/.test(mlb));
  const nba = await populated({ sport: "nba", rows: [] }).G.buildBoardPage("nba");
  assert.ok(!/board-view/.test(nba));
});

test("Power Rankings pill: the legacy sortable rankingsTable renders in place for the page's sport", async () => {
  const P = populated({ sport: "cfb", search: "?view=rankings" });
  const html = await P.G.buildBoardPage("cfb");
  assert.match(html, /class="ca-pill on" data-pill="board-view" data-key="rankings"/);
  assert.ok(html.includes('class="rk-body"') && html.includes('data-sort="rank"') && html.includes('data-sort="rating"'));
  assert.ok(html.includes('<th data-sort="conf"'), "CFB: the conference column");
  assert.ok(html.includes('id="rk-conf"'), "CFB: the conference select");
  assert.ok(P.requested.some((u) => u.startsWith("power_rankings_current") && u.includes("sport=eq.cfb")), "rankings of THIS page's sport");
  assert.ok(!P.requested.some((u) => u.startsWith("predictions_current")), "the Games data is not fetched in the rankings view");
  assert.ok(html.includes("Season 2026 · Week 5"));
  assert.ok(!/NaN|undefined/.test(html));
  assert.equal(P.G.rkSport(), "cfb", "the sort wiring resolves the sport from the page, not ?sport=");
  const empty = await populated({ sport: "nfl", search: "?view=rankings", ranks: [] }).G.buildBoardPage("nfl");
  assert.ok(empty.includes("No NFL power rankings yet."));
});

test("MLB / NBA boards: the status card; MLB rows still in the table appear below it", async () => {
  const mlb = await populated({ sport: "mlb", rows: [] }).G.buildBoardPage("mlb");
  assert.match(mlb, /class="ca-card[^"]*"[^>]*>(?:(?!<\/section>).)*MLB model paused \(last projections Aug 31, 2026\)/s);
  assert.ok(mlb.includes("MLB Board") && !mlb.includes("<tbody>"));
  assert.ok(!mlb.includes("board-range"), "no range pills when there is nothing to filter");
  const nba = await populated({ sport: "nba", rows: [] }).G.buildBoardPage("nba");
  assert.ok(nba.includes("NBA model not live yet") && nba.includes("NBA Board"));
  const withRows = await populated({ sport: "mlb", rows: [{ game_pk: 9, home_team_name: "Boston Red Sox", away_team_name: "New York Yankees", commence_time: new Date().toISOString(), pred_home_score: 5.1, pred_away_score: 3.9, market_spread: -1.5, market_total: 8.5 }] }).G.buildBoardPage("mlb");
  assert.ok(withRows.includes("MLB model paused") && withRows.includes("<tbody>") && withRows.includes("game.html?sport=mlb"));
  assert.ok(withRows.includes('data-pill="board-range"'), "range pills return once rows exist");
  assert.ok(!/NaN|undefined/.test(withRows));
});

test("cards are isolated: a board failure does not blank the page, and fetch failures degrade to empty states", async () => {
  const q = quiet();
  const html = await populated({ fail: ["ev_current", "ev_prop_picks_current", "ev_pnl_daily", "ev_best_lines"], q }).G.buildBoardPage("nfl");
  assert.ok(html.includes("<tbody>") && html.includes("ca-board"), "games still render without opportunities");
  assert.ok(!/ca-alpha-cell/.test(html));
  const p = await populated({ fail: ["predictions_current"], q }).G.buildBoardPage("nfl");
  assert.ok(p.includes("No NFL games on the board right now."));
});

// ---- interaction ----------------------------------------------------------------------------------
function fakeBoardDoc() {
  const handlers = {}, root = { addEventListener: (t, f) => { handlers[t] = f; } }, els = {}, shell = { innerHTML: "" };
  const doc = { body: { dataset: { page: "nfl" }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {}, createElement: () => ({}),
    querySelector: (sel) => (sel === ".ca-board" ? root : sel === ".page-shell" ? shell : null),
    getElementById: (id) => (els[id] = els[id] || { id, outerHTML: "" }), querySelectorAll: () => [], addEventListener() {} };
  return { doc, handlers, els, shell };
}
const hit = (map) => ({ target: { closest: (sel) => map[sel] || null } });
test("interaction: a range pill updates state + URL and redraws the table from the cache; the view pill writes the URL", async () => {
  const D = fakeBoardDoc(), urls = [];
  const NOW = Date.now();
  const rows = [{ game_pk: 1, home_team_name: "A", away_team_name: "B", commence_time: new Date(NOW + 30 * 36e5).toISOString(), pred_home_score: 20, pred_away_score: 10 },
    { game_pk: 2, home_team_name: "C", away_team_name: "D", commence_time: new Date(NOW + 24 * 12 * 36e5).toISOString(), pred_home_score: 20, pred_away_score: 10 }];
  const G = loadScripts(FILES, { page: "nfl", globals: { document: D.doc, scrollTo() {}, location: { search: "", href: "http://localhost/nfl.html" }, history: { replaceState: (a, b, u) => urls.push(u) },
    fetch: async (url) => { const path = urlParts(url).path; return { ok: true, json: async () => (path === "predictions_current" ? rows
      : path === "power_rankings_current" ? [{ sport: "nfl", rank: 1, team: "Detroit Lions", rating: 3, season: 2026, week: 5, units: {} }] : []) }; } } });
  await G.buildBoardPage("nfl");
  G.wireBoardPage();
  const last = () => new URL(urls[urls.length - 1]);
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-range", key: "week" } } }));
  assert.equal(last().searchParams.get("range"), "week");
  assert.ok(D.els["board-games"].outerHTML.includes("<tbody>") && (D.els["board-games"].outerHTML.match(/<tr data-href/g) || []).length === 1, "this week: one game");
  assert.match(D.els["board-bar"].outerHTML, /class="ca-pill on" data-pill="board-range" data-key="week"/);
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-range", key: "all" } } }));
  assert.equal(last().search, "", "default range: a clean URL");
  assert.equal((D.els["board-games"].outerHTML.match(/<tr data-href/g) || []).length, 2);
  // the view pill: URL + a full re-render (it needs different data) that lands on the rankings
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-view", key: "rankings" } } }));
  assert.equal(last().searchParams.get("view"), "rankings");
  await new Promise((r) => setTimeout(r, 10));
  assert.ok(D.shell.innerHTML.includes('class="rk-body"') && D.shell.innerHTML.includes("data-pill=\"board-view\""), "render() redrew the page in the Power Rankings view");
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-view", key: "rankings" } } }));
  assert.equal(urls.length, 3, "the already-selected view is a no-op");
  D.handlers.click(hit({ "[data-pill]": { dataset: { pill: "board-view", key: "games" } } }));
  assert.equal(last().search, "", "back to the default view: a clean URL");
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

// ---- CSS contract --------------------------------------------------------------------------------
test("CSS contract: every ca-* class the board, rankings and settings pages emit exists in theme.css", async () => {
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  const classesIn = (html) => [...new Set([...html.matchAll(/class="([^"]*)"/g)].flatMap((m) => m[1].split(/\s+/)).filter((c) => /^ca-/.test(c)))];
  const hooks = new Set(["ca-minibars", "ca-spark", "ca-donut", "ca-bar"]);
  const defined = (c) => new RegExp(`\\.${c}(?![\\w-])`).test(css);
  const missing = (html) => classesIn(html).filter((c) => !hooks.has(c) && !defined(c));
  const pages = [await populated().G.buildBoardPage("nfl"), await populated({ sport: "cfb", search: "?view=rankings" }).G.buildBoardPage("cfb"),
    await populated({ sport: "mlb", rows: [] }).G.buildBoardPage("mlb")];
  const A = loadScripts(FILES, { page: "rankings", globals: { location: { search: "?sport=cfb", href: "http://localhost/rankings.html?sport=cfb" },
    fetch: async () => ({ ok: true, json: async () => [{ sport: "cfb", rank: 1, team: "Oregon", rating: 5, conf: 5, season: 2026, week: 5, units: {} }] }) } });
  pages.push(await A.buildRankings(), A.buildSettings(), g.searchOverlayHtml());
  for (const html of pages) assert.deepEqual(missing(html), [], "all ca-* classes are styled");
  for (const c of classesIn(pages[5])) assert.equal(css.split(new RegExp(`(?:^|\\n)\\.${c}\\{`)).length - 1, 1, `.${c} is defined once (the overlay classes must not collide with the shared +EV / Track search field)`);
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
