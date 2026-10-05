import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";
// The sport pages' right column (js/pages/board-right.js): Top Alpha Edges, Betting Splits, Model Projections.
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/ev.js", "js/pages/board.js", "js/pages/board-left.js", "js/pages/board-right.js", "js/boot.js"];
const NOW = Date.parse("2026-10-05T16:00:00Z");     // Monday 12:00 ET: NFL week 4 (Sep 29 - Oct 5)
class FakeDate extends Date { constructor(...a) { if (a.length) super(...a); else super(NOW); } static now() { return NOW; } }
const quiet = () => { const logs = { warn: [], error: [] }; return { logs, console: { ...console, warn: (...a) => logs.warn.push(a.join(" ")), error: (...a) => logs.error.push(a.join(" ")) } }; };
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };

// Week 3 (Sep 22 - 28): games 1, 2.  Week 4 (Sep 29 - Oct 5): games 3 (graded, Oct 1), 4 (upcoming, Oct 5 ET).
const A = "Buffalo Bills", H = "Atlanta Falcons", A2 = "Detroit Lions", H2 = "Chicago Bears";
const pred = (pk, date, iso, away, home, extra = {}) => ({ sport: "nfl", game_pk: pk, game_date: date, home_team_name: home, away_team_name: away, commence_time: iso, home_win_prob: 0.55, pred_home_score: 24, pred_away_score: 21, market_spread: -3, market_total: 44, model_version: "nfl-sim-ml-v2", ...extra });
const ANY = [pred(1, "2026-09-27", "2026-09-27T17:00:00Z", A, H), pred(2, "2026-09-28", "2026-09-28T17:00:00Z", A2, H2), pred(3, "2026-10-01", "2026-10-02T00:15:00Z", A, H2), pred(4, "2026-10-05", "2026-10-06T00:15:00Z", A2, H)];
// graded: edge (model line - market line from the favourite's side) / pick results chosen to land in different bands
const acc = (pk, date, away, home, o) => ({ sport: "nfl", game_pk: pk, game_date: date, home_team_name: home, away_team_name: away, win_prob: 0.6, predicted_winner: home, actual_winner: home, winner_correct: true, pred_margin: 7, actual_margin: 7, pred_total: 50, actual_total: 51,
  market_spread: -3, market_total: 44, spread_pick_correct: true, total_pick_correct: true, ...o });
const ACC = [acc(1, "2026-09-27", A, H, {}), acc(2, "2026-09-28", A2, H2, { pred_margin: 1, spread_pick_correct: false, total_pick_correct: null, pred_total: 44.5 }), acc(3, "2026-10-01", A, H2, { pred_margin: 12, spread_pick_correct: true, actual_winner: A, winner_correct: false, total_pick_correct: false, pred_total: 40 })];
// the stored +EV picks: lines by (game, market, side) with a rebuild row for pick 1 spread (the later one is the one kept)
const evp = (pk, market, side, o = {}) => ({ sport: "nfl", game_pk: pk, market, side, matchup: pk === 1 ? `${A} @ ${H}` : pk === 2 ? `${A2} @ ${H2}` : `${A} @ ${H2}`, commence_time: pk === 1 ? "2026-09-27T17:00:00Z" : pk === 2 ? "2026-09-28T17:00:00Z" : "2026-10-02T00:15:00Z",
  true_prob: 0.7, best_line_implied: 0.5, best_price: 100, best_book: "draftkings", ev_best: 0.3, is_pick: true, created_at: "2026-09-26T10:00:00Z", model_version: "ev-pilot-v1", ...o });
const PICKS = [evp(1, "moneyline", "home", { ev_best: 0.4, true_prob: 0.75 }), evp(1, "spread", "home", { ev_best: 0.1, true_prob: 0.55, created_at: "2026-09-25T10:00:00Z" }), evp(1, "spread", "home", { ev_best: 0.35, true_prob: 0.72 }),
  evp(2, "total", "over", { ev_best: 0.3, true_prob: 0.68 }), evp(2, "moneyline", "away", { ev_best: 0.002, true_prob: 0.52, best_line_implied: 0.5 }), evp(3, "moneyline", "away", { ev_best: 0.28, true_prob: 0.66 })];
const RES = [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", won: true }, { sport: "nfl", game_pk: 1, market: "spread", side: "home", won: false }, { sport: "nfl", game_pk: 2, market: "total", side: "over", won: null },
  { sport: "nfl", game_pk: 3, market: "moneyline", side: "away", won: true }];
const PROPPICKS = [{ sport: "nfl", game_pk: 1, player_id: "p1", player_name: 'Zed <b>"Q"</b>', market: "rec_yds", side: "under", line: 41.5, model_version: "props-sim-v1", matchup: `${A} @ ${H}`, commence_time: "2026-09-27T17:00:00Z", model_prob: 0.8, best_price: -110, best_book: "betmgm", ev_best: 0.5, created_at: "2026-09-26T10:00:00Z" }];
const PROPRES = [{ sport: "nfl", game_pk: 1, player_id: "p1", market: "rec_yds", line: 41.5, model_version: "props-sim-v1", result: "loss" }];
const SPLITS = (pk, commence) => [["spread", "away", 40, 55], ["spread", "home", 60, 45], ["moneyline", "away", 30, 25], ["moneyline", "home", 70, 75], ["total", "over", 66, 70], ["total", "under", 34, 30]]
  .map(([market, side, t, c]) => ({ game_pk: pk, market, side, ticket_pct: t, cash_pct: c, commence_time: commence, captured_at: "2026-09-27T15:02:00Z" }));
const LIVE_OPP = { sport: "nfl", game_pk: 4, matchup: `${A2} @ ${H}`, market: "moneyline", side: "away", true_prob: 0.7, best_line_implied: 0.5, best_price: 100, best_book: "draftkings", ev_best: 0.3, is_pick: true, commence_time: "2026-10-06T00:15:00Z" };

function populated({ search = "", sport = "nfl", acc: accRows = ACC, picks = PICKS, results = RES, props = PROPPICKS, propRes = PROPRES, splits = [...SPLITS(1, "2026-09-27T17:00:00Z"), ...SPLITS(3, "2026-10-02T00:15:00Z")], cfbSplits = [], any = ANY, cur = [ANY[3]], live = [LIVE_OPP], starts = [], fail = [], q = null, doc = null } = {}) {
  const requested = [];
  const fetch = async (url) => {
    const { path, q: qs } = urlParts(url); requested.push(path + qs);
    if (fail.includes(path)) return { ok: false, status: 500, text: async () => "boom", json: async () => [] };
    const own = (rows) => (qs.includes(`sport=eq.${sport}`) ? rows : []);
    const inWin = (rows, key) => { const p = new URLSearchParams(qs), g = p.getAll(key); return rows.filter((x) => g.every((c) => (c.startsWith("gte.") ? String(x[key]) >= c.slice(4) : c.startsWith("lte.") ? String(x[key]) <= c.slice(4) : c.startsWith("lt.") ? String(x[key]) < c.slice(3) : true))); };
    let r = [];
    if (path === "prediction_accuracy") r = inWin(own(accRows), "game_date");
    else if (path === "predictions_any") r = own(any); else if (path === "predictions_current") r = own(cur);
    else if (path === "track_record_start") r = starts;
    else if (path === "game_closing_prices") r = accRows.flatMap((a) => [{ game_pk: a.game_pk, market: "moneyline", side: "home", close_dec: 1.7 }, { game_pk: a.game_pk, market: "moneyline", side: "away", close_dec: 2.3 }]);
    else if (path === "prediction_pnl_daily") r = [];
    else if (path === "ev_picks") r = inWin(own(picks), "commence_time");
    else if (path === "ev_prop_picks") r = inWin(own(props), "commence_time");
    else if (path === "ev_results") r = own(results); else if (path === "ev_prop_results") r = own(propRes);
    else if (path === "ev_current") r = own(live);
    else if (path === "nfl_betting_splits_current") r = splits; else if (path === "cfb_betting_splits_current") r = cfbSplits;
    return { ok: true, json: async () => r };
  };
  const G = loadScripts(FILES, { page: sport, globals: { ...(doc ? { document: doc } : {}), Date: FakeDate, fetch, location: { search, href: `http://localhost/${sport}.html${search}` }, ...(q ? { console: q.console } : {}) } });
  return { G, requested };
}
const rightOf = (html) => html.slice(html.indexOf('id="board-right"'));
const IDS = ["board-top-alpha", "board-betting-splits", "board-model-projections"];
const card = (html, id) => { const r = rightOf(html), i = r.indexOf(`id="${id}"`); if (i < 0) return ""; const nxt = IDS.map((x) => r.indexOf(`id="${x}"`, i + 5)).filter((j) => j > 0); return r.slice(i, nxt.length ? Math.min(...nxt) : r.length); };
const text = (h) => h.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
const rowsOf = (h) => h.split('<a class="ca-ev-ta').slice(1);

test("the three cards mount in the right column in order, each with its title; no empty slot is left", async () => {
  const html = await populated().G.buildBoardPage("nfl"), r = rightOf(html), at = IDS.map((id) => r.indexOf(`id="${id}"`));
  assert.ok(at.every((i) => i > 0) && at.every((i, k) => !k || at[k - 1] < i), "top to bottom");
  assert.ok(!r.includes("ca-board-slot"));
  assert.ok(card(html, IDS[0]).includes("<h2>Top Alpha Edges</h2>") && card(html, IDS[1]).includes("<h2>Betting Splits</h2>") && card(html, IDS[2]).includes("<h2>Model Projections</h2>"));
  assert.ok(!/NaN|undefined|Infinity/.test(r));
});

test("Top Alpha Edges, current week: the live board through the shared row (rank, logos with @, line, 'vs', green edge box), View All -> +EV for this sport", async () => {
  const html = await populated().G.buildBoardPage("nfl"), ta = card(html, IDS[0]);
  assert.ok(ta.includes('href="ev.html?sport=nfl&amp;sort=alpha"') && ta.includes("All →"), "View All");
  assert.deepEqual([...ta.matchAll(/data-pill="br-ta" data-key="(\w+)"/g)].map((m) => m[1]), ["all", "moneyline"], "pills for the markets present (only a moneyline opportunity is live)");
  const rows = rowsOf(ta);
  assert.ok(rows.length >= 1 && rows.length <= 5);
  assert.ok(rows[0].includes('class="ca-ev-rank">#1<') && rows[0].includes('class="ca-ev-logos"') && rows[0].includes('class="ca-ev-at">@<') && rows[0].includes('class="ca-ev-ta-main"') && rows[0].includes('class="ca-ev-edge"') && rows[0].includes(">Edge<"));
  assert.ok(/vs [\d.]+%/.test(text(rows[0])) && rows[0].includes('class="ca-fair"'), "'model/fair vs implied', the R19 fair marker on a game line");
  assert.ok(rows[0].includes('href="game.html?sport=nfl&game=4"'), "the live opportunity of game 4 (the current week)");
  const live = rows.find((r) => r.includes("game=4"));
  assert.ok(live && !live.includes("ca-res ") && live.includes("Not graded yet"), "the live game has no result yet (a dash; game 3's stored pick of the same week is already graded and carries its chip)");
});

// The board's D as the right cards receive it.
async function withD(P, sport = "nfl") {
  let D = null; const orig = P.G.BOARD_CARDS["board-top-alpha"];
  P.G.BOARD_CARDS["board-top-alpha"] = (d) => { D = d; return orig(d); };
  const html = await P.G.buildBoardPage(sport);
  return { html, D };
}

test("Top Alpha Edges, a past week: that week's stored picks (latest rebuild), ranked by alpha, each with a W / L / P chip from the graded results", async () => {
  const P = populated({ search: "?week=3" }), html = await P.G.buildBoardPage("nfl"), ta = card(html, IDS[0]), rows = rowsOf(ta);
  assert.ok(text(ta).includes("Week 3's top edges with their graded results"), text(ta));
  assert.ok(!ta.includes("game=3") && !ta.includes("game=4"), "only week 3's games (1, 2)");
  assert.equal(rows.length, 4, "prop, moneyline, spread, total");
  // alpha order: the prop (0.5 EV, 0.8 prob), the moneyline (0.4 EV), the spread rebuild with 0.35 EV (the later created_at, NOT the 0.1 one), the total
  assert.ok(rows[0].includes("Rec Yds") && rows[0].includes('class="ca-res L"') && rows[0].includes("Zed &lt;b&gt;&quot;Q&quot;&lt;/b&gt;"), "the escaped prop pick, lost");
  assert.ok(rows[1].includes("Falcons ML") && rows[1].includes('class="ca-res W"'));
  assert.ok(rows[2].includes("Falcons spread") && rows[2].includes('class="ca-res L"') && rows[2].includes(">+22.0%<"), "the latest rebuild of the spread pick: edge (0.72 - 0.5) = +22.0%");
  assert.ok(rows[3].includes("Over total") && rows[3].includes('class="ca-res P"') && rows[3].includes("Push"), "won = null is a push");
  assert.ok(rows.every((r) => r.includes("ca-res")) && !ta.includes("ca-ev-ta-noaux"), "the result cell is on every row");
  assert.ok(!ta.includes("Lions ML") && !ta.includes("Bears ML"), "R9: the 0.2% EV pick of game 2 has no tier and is not an opportunity");
  assert.ok(P.requested.some((u) => u.startsWith("ev_picks") && u.includes("commence_time=gte.2026-09-21T") && u.includes("sport=eq.nfl")), "windowed by the period");
  assert.ok(P.requested.some((u) => u.startsWith("ev_results") && u.includes("game_pk=in.(1,2)")) && P.requested.some((u) => u.startsWith("ev_prop_results")), "results of those games");
  assert.ok(text(ta).includes("Week 3"), "the period is named");
});

test("Top Alpha Edges: a stored pick of a finished game lends its result in the current week; empty and failed states are honest", async () => {
  const live = [{ ...LIVE_OPP, game_pk: 3, matchup: `${A} @ ${H2}`, side: "away", commence_time: "2026-10-02T00:15:00Z" }, LIVE_OPP];
  const html = await populated({ live, any: ANY, cur: [ANY[2], ANY[3]] }).G.buildBoardPage("nfl"), rows = rowsOf(card(html, IDS[0]));
  assert.equal(rows.length, 2);
  const done = rows.find((r) => r.includes("game=3")), todo = rows.find((r) => r.includes("game=4"));
  assert.ok(done.includes('class="ca-res W"') && todo.includes("Not graded yet") && !todo.includes("ca-res "), "game 3 is graded (Bills ML won), game 4 is not: a dash keeps the columns aligned");
  const none = await populated({ search: "?week=2", picks: [], props: [] }).G.buildBoardPage("nfl");
  assert.ok(card(none, IDS[0]).includes("No +EV picks were stored for Week 2.") && !card(none, IDS[0]).includes("ca-ev-ta"), text(card(none, IDS[0])));
  const idle = await populated({ live: [], picks: [], props: [] }).G.buildBoardPage("nfl");
  assert.ok(card(idle, IDS[0]).includes("No +EV opportunities on the board right now."));
  const bad = quiet(), failed = await populated({ fail: ["ev_picks"], q: bad }).G.buildBoardPage("nfl");
  assert.ok(card(failed, IDS[0]).includes("This panel couldn't load."), "a failed read is not an empty week");
  assert.ok(bad.logs.error.some((e) => /board right: stored failed/.test(e)));
});

test("Top Alpha pills: only the markets present, a pill keeps its market, a market with no rows in this period falls back to All", async () => {
  const P = populated({ search: "?week=3" }), { D } = await withD(P), G = P.G;
  assert.deepEqual([...G.BOARD_CARDS["board-top-alpha"](D).matchAll(/data-key="(\w+)"/g)].map((m) => m[1]), ["all", "spread", "moneyline", "total", "props"]);
  G.brSetTop("props");
  const only = rowsOf(G.BOARD_CARDS["board-top-alpha"](D));
  assert.equal(only.length, 1); assert.ok(only[0].includes("Rec Yds"));
  G.brSetTop("moneyline");
  assert.equal(rowsOf(G.BOARD_CARDS["board-top-alpha"](D)).length, 1, "game 2's moneyline pick has no tier");
  const P4 = populated(), d4 = await withD(P4); P4.G.brSetTop("props");
  assert.ok(rowsOf(P4.G.BOARD_CARDS["board-top-alpha"](d4.D)).length >= 1 && !/data-key="props"/.test(P4.G.BOARD_CARDS["board-top-alpha"](d4.D)), "week 4 has no prop: the stale 'props' choice falls back to All");
});

test("Betting Splits: tabs, a game selector (default = the top-alpha game with splits), the donut + legend, % of Tickets and % of Money bars", async () => {
  const html = await populated({ search: "?week=3" }).G.buildBoardPage("nfl"), sp = card(html, IDS[1]);
  assert.ok(sp.includes("<h2>Betting Splits</h2>"));
  assert.deepEqual([...sp.matchAll(/data-pill="br-sp" data-key="(\w+)"/g)].map((m) => m[1]), ["spread", "moneyline", "total"]);
  assert.deepEqual([...sp.matchAll(/<option value="(\d+)"[^>]*>([^<]*)</g)].map((m) => [m[1], m[2]]), [["1", "BUF @ ATL"]], "only the games of week 3 that have splits (game 3 is in week 4)");
  assert.ok(sp.includes("ca-donut") && sp.includes('class="ca-br-dc"') && sp.includes("<b>60%</b>") && sp.includes("<span>ATL</span>"), "the spread: 60% of tickets on ATL (home)");
  assert.deepEqual([...sp.matchAll(/class="ca-br-lg"><i[^>]*><\/i><span[^>]*>([^<]*)<\/span><b>(\d+%)<\/b>/g)].map((m) => [m[1], m[2]]), [["Bills", "40%"], ["Falcons", "60%"]]);
  assert.ok(sp.includes("% of Tickets") && sp.includes("% of Money") && sp.includes("<span>40% BUF</span><span>60% ATL</span>") && sp.includes("<span>55% BUF</span><span>45% ATL</span>"), "tickets and money");
  assert.ok(sp.includes("Captured Sep 27, 11:02 AM ET"));
  const w4 = await populated().G.buildBoardPage("nfl");
  assert.ok(card(w4, IDS[1]).includes('<option value="3" selected>BUF @ CHI</option>'), "week 4: game 3 has splits");
});

test("Betting Splits: the default game is the one of the highest-alpha edge that has splits; a total reads Over / Under; a tab or game pick sticks for this period only", async () => {
  const two = [...SPLITS(1, "2026-09-27T17:00:00Z"), ...SPLITS(2, "2026-09-28T17:00:00Z")];
  const P = populated({ search: "?week=3", splits: two }), { D, html } = await withD(P), G = P.G;
  assert.ok(card(html, IDS[1]).includes('<option value="1" selected>'), "the top edge of week 3 is game 1 (the prop, then the moneyline)");
  const lowFirst = populated({ search: "?week=3", splits: two, picks: PICKS.filter((p) => p.game_pk === 2 || p.market === "total"), props: [] });
  assert.ok(card(await lowFirst.G.buildBoardPage("nfl"), IDS[1]).includes('<option value="2" selected>'), "no pick for game 1 -> game 2's");
  G.brSetSplit({ tab: "total", game: "" }, G.blKey(D.period));
  const t = G.BOARD_CARDS["board-betting-splits"](D);
  assert.ok(t.includes("<span>Over</span>") && t.includes("<b>66%</b>") && t.includes("Under") && t.includes("<span>66% Over</span><span>34% Under</span>") && t.includes("<span>70% Over</span><span>30% Under</span>"), text(t));
  G.brSetSplit({ game: "2" }, G.blKey(D.period));
  assert.ok(G.BOARD_CARDS["board-betting-splits"](D).includes('<option value="2" selected>'));
  G.brSetSplit({ tab: "moneyline" }, "week:9");
  assert.ok(!G.BOARD_CARDS["board-betting-splits"](D).includes('data-key="moneyline" role="tab" aria-selected="true"'), "a choice made for another period is ignored");
});

test("Betting Splits are hidden when the period has none; a game with only one side's share is not a split; MLB / NBA say there are none", async () => {
  assert.ok(!rightOf(await populated({ search: "?week=2", splits: [] }).G.buildBoardPage("nfl")).includes('id="board-betting-splits"'), "no splits at all");
  assert.ok(!rightOf(await populated({ search: "?week=2" }).G.buildBoardPage("nfl")).includes('id="board-betting-splits"'), "splits exist but none for a game of week 2");
  const cfb = rightOf(await populated({ sport: "cfb", picks: [], props: [], any: [], cur: [], live: [], acc: [] }).G.buildBoardPage("cfb"));
  assert.ok(!cfb.includes('id="board-betting-splits"') && cfb.includes('id="board-top-alpha"') && cfb.includes('id="board-model-projections"'), "CFB: no splits, no card");
  const half = SPLITS(1, "2026-09-27T17:00:00Z").filter((x) => !(x.market === "spread" && x.side === "home"));
  assert.ok(!card(await populated({ search: "?week=3", splits: half }).G.buildBoardPage("nfl"), IDS[1]).includes('data-key="spread"'), "the spread tab needs both sides");
});

// ---- Model Projections ---------------------------------------------------------------------------------------------------
const L_OF = (rows, market = "spread") => ({ rec: rows.map((r) => ({ market, result: r.res, game_pk: r.pk, date: r.date || "2026-10-01" })),
  accRec: rows.map((r) => ({ sport: "nfl", game_pk: r.pk, game_date: r.date || "2026-10-01", home_team_name: "H", away_team_name: "A", pred_margin: r.pm, market_spread: -3, market_total: 44, pred_total: r.pt ?? 44, win_prob: r.wp ?? 0.6, predicted_winner: "H", actual_winner: "H" })), closing: r0close(rows) });
const r0close = (rows) => new Map(rows.flatMap((r) => [[`${r.pk}|home`, r.home ?? -150], [`${r.pk}|away`, r.away ?? 130]]));

test("Model Projections bands (spread): edge = favourite's market line minus the model's, boundaries fall in the documented band, hit rate = W / (W + L), pushes counted in games only", async () => {
  const G = populated().G;
  // market home -3: model line = -pm, edge = pm - 3 -> 11, 10, 5.5 | 5, 3, 2.5 | 2, 0, -2 | -2.5, -4, -5 | -5.5, -6, -7
  const rows = [[14, "W"], [13, "L"], [8.5, "W"], [8, "W"], [6, "L"], [5.5, "W"], [5, "L"], [3, "W"], [1, "L"], [0.5, "L"], [-1, "W"], [-2, "L"], [-2.5, "W"], [-3, "L"], [-4, "L"]].map(([pm, res], i) => ({ pk: i + 1, pm, res }));
  const d = G.brProjData(L_OF(rows), "spread", "2026-10-01", "2026-10-01"), by = d.bands.map((b) => [b.label, b.n, b.w, b.l]);
  assert.deepEqual(by, [["> +10", 1, 1, 0], ["+5 to +10", 2, 1, 1], ["+2 to +5", 3, 2, 1], ["-2 to +2", 3, 1, 2], ["-5 to -2", 3, 1, 2], ["< -5", 3, 1, 2]]);
  assert.deepEqual([d.total.n, d.total.w, d.total.l, d.skipped], [15, 7, 8, 0]);
  assert.equal(d.bands[0].pct, 1); assert.ok(Math.abs(d.bands[1].pct - 0.5) < 1e-9);
  const withPush = L_OF([{ pk: 1, pm: 14, res: "P" }, { pk: 2, pm: 14, res: "W" }]);
  const p = G.brProjData(withPush, "spread", "2026-10-01", "2026-10-01");
  assert.deepEqual([p.bands[0].n, p.bands[0].w, p.bands[0].l, p.bands[0].p, p.bands[0].pct], [2, 1, 0, 1, 1], "a push is a game but not decided");
  // only picks in [from, to]; a pick with no market line is counted as skipped, not bucketed
  const mixed = L_OF([{ pk: 1, pm: 14, res: "W", date: "2026-10-01" }, { pk: 2, pm: 14, res: "L", date: "2026-09-20" }]);
  assert.equal(G.brProjData(mixed, "spread", "2026-10-01", "2026-10-05").total.n, 1);
  mixed.accRec[0].market_spread = null;
  const sk = G.brProjData(mixed, "spread", "2026-09-01", "2026-10-05");
  assert.deepEqual([sk.total.n, sk.skipped, sk.picks], [1, 1, 2]);
});

test("Model Projections bands (moneyline in probability points, total in points)", async () => {
  const G = populated().G;
  // moneyline: the market favourite is the home side (spread -3); the model's home win prob wp vs the closing price's implied prob (-150 -> 60%)
  const ml = L_OF([{ pk: 1, wp: 0.75, res: "W" }, { pk: 2, wp: 0.68, res: "L" }, { pk: 3, wp: 0.6, res: "W" }, { pk: 4, wp: 0.5, res: "L" }], "moneyline");
  const dm = G.brProjData(ml, "moneyline", "2026-10-01", "2026-10-01");
  assert.equal(dm.unit, "pp");
  assert.deepEqual(dm.bands.map((b) => b.n), [1, 1, 0, 1, 0, 1], "+15, +8, 0 and -10 pp against the 60% implied by the -150 close");
  // total: model total pt vs the market 44 (an edge of + means Over)
  const tot = L_OF([{ pk: 1, pt: 56, res: "W" }, { pk: 2, pt: 38, res: "L" }, { pk: 3, pt: 44.5, res: "W" }], "total");
  const dt = G.brProjData(tot, "total", "2026-10-01", "2026-10-01");
  assert.deepEqual(dt.bands.map((b) => b.n), [1, 0, 0, 1, 0, 1]); assert.equal(dt.unit, "pts");
});

test("Model Projections card: select, Week / Season toggle, Range / # of Games / Hit Rate (no Units), band labels with units, total row", async () => {
  const html = await populated().G.buildBoardPage("nfl"), mp = card(html, IDS[2]);
  assert.ok(mp.includes("<h2>Model Projections</h2>") && mp.includes('data-br="mp-market"'));
  assert.deepEqual([...mp.matchAll(/<option value="(\w+)"/g)].map((m) => m[1]), ["spread", "moneyline", "total"]);
  assert.deepEqual([...mp.matchAll(/<th>([^<]*)<\/th>/g)].map((m) => m[1]), ["Range", "# of Games", "Hit Rate"], "no Units column: there is no per-pick profit");
  assert.ok(!/Units/.test(text(mp).replace(/Units are not shown[^.]*\./, "")), "no Units column or cell");
  assert.deepEqual([...mp.matchAll(/<tr[^>]*><td>([^<]*)<\/td>/g)].map((m) => m[1]), ["&gt; +10 pts", "+5 to +10", "+2 to +5", "-2 to +2", "-5 to -2", "&lt; -5 pts", "All graded"]);
  assert.deepEqual([...mp.matchAll(/data-pill="br-mp" data-key="(\w+)" role="tab" aria-selected="(\w+)"/g)].map((m) => [m[1], m[2]]), [["period", "false"], ["season", "true"]], "the current week opens on the season");
  assert.ok(mp.includes("Season 2026 ·"), "the caption names what the numbers cover");
  const total = mp.match(/class="ca-br-tot"><td>All graded<\/td><td>(\d+)<\/td>/);
  assert.equal(+total[1], 3, "3 graded spread picks in the season (games 1, 2, 3)");
});

test("Model Projections: the Week / Season toggle and the select change the numbers; a week before the record says so; no market line", async () => {
  const P = populated({ search: "?week=3" }), { D } = await withD(P), G = P.G;
  const wk = G.BOARD_CARDS["board-model-projections"](D);
  assert.deepEqual([...wk.matchAll(/data-pill="br-mp" data-key="(\w+)" role="tab" aria-selected="(\w+)"/g)].map((m) => [m[1], m[2]]), [["period", "true"], ["season", "false"]], "a past week opens on the week");
  assert.ok(wk.includes("Sep 22 – Sep 28, 2026") && wk.includes(">Week 3</button>") && /class="ca-br-tot"><td>All graded<\/td><td>2<\/td>/.test(wk), "week 3 holds games 1 and 2");
  G.brSetProj({ scope: "season" }, G.blKey(D.period));
  assert.ok(/class="ca-br-tot"><td>All graded<\/td><td>3<\/td>/.test(G.BOARD_CARDS["board-model-projections"](D)), "the season holds all three");
  G.brSetProj({ market: "total" }, G.blKey(D.period));
  const tot = G.BOARD_CARDS["board-model-projections"](D);
  assert.ok(tot.includes('<option value="total" selected>') && /class="ca-br-tot"><td>All graded<\/td><td>3<\/td>/.test(tot) && /<td>1<\/td><td>/.test(tot), "3 total picks in the season (game 2's is a push)");
  // the record restart: week 3 is archived
  const starts = [{ sport: "nfl", starts_at: "2026-09-29T17:00:00Z", model_version: "nfl-sim-ml-v2" }];
  const arch = card(await populated({ search: "?week=3", starts }).G.buildBoardPage("nfl"), IDS[2]);
  assert.ok(text(arch).includes("Week 3 is before the published NFL record, which restarted Sep 29, 2026 with the ML v2 model."), text(arch));
  const none = card(await populated({ acc: [] }).G.buildBoardPage("nfl"), IDS[2]);
  assert.ok(none.includes("No graded NFL picks in the published record yet.") && !none.includes("<table"));
  const failed = quiet(), bad = card(await populated({ fail: ["track_record_start"], q: failed }).G.buildBoardPage("nfl"), IDS[2]);
  assert.ok(bad.includes("This panel couldn't load."), "no record scope: no numbers (fails closed)");
});

test("MLB and NBA: every right-column card is an honest empty state, nothing is requested for the live feeds", async () => {
  for (const [sport, why] of [["mlb", "MLB model paused"], ["nba", "No NBA model yet."]]) {
    const P = populated({ sport, any: [], cur: [], acc: [], picks: [], props: [], live: [] }), html = await P.G.buildBoardPage(sport), r = rightOf(html);
    for (const id of IDS) assert.ok(card(html, id).includes(why) && card(html, id).includes('class="ca-empty"'), `${sport} ${id}: ${text(card(html, id))}`);
    assert.ok(card(html, IDS[0]).includes("No +EV edges to rank.") && card(html, IDS[1]).includes("No betting splits are captured") && card(html, IDS[2]).includes("No graded"));
    assert.ok(!r.includes("ca-ev-ta") && !r.includes("<table") && !r.includes("ca-donut") && !/NaN|undefined/.test(r), "no fake rows");
    assert.ok(!P.requested.some((u) => /ev_picks|ev_prop_picks|ev_results|betting_splits/.test(u)), "a sport with no model reads none of the +EV or splits tables");
  }
});

test("Rankings view has no right column and the card loader does nothing there", async () => {
  const P = populated({ search: "?view=rankings" }), html = await P.G.buildBoardPage("nfl");
  assert.ok(!html.includes('id="board-right"'));
  assert.ok(!P.requested.some((u) => /ev_picks|betting_splits/.test(u)));
});

test("interaction: the pills and selects redraw their own card from the loaded data (no refetch)", async () => {
  const els = {}, doc = { body: { dataset: { page: "nfl" }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {}, createElement: () => ({}),
    querySelector: () => null, getElementById: (id) => (els[id] = els[id] || { id, outerHTML: "" }), querySelectorAll: () => [], addEventListener() {} };
  const P = populated({ search: "?week=3", doc, splits: [...SPLITS(1, "2026-09-27T17:00:00Z"), ...SPLITS(2, "2026-09-28T17:00:00Z")] }), G = P.G;
  await G.buildBoardPage("nfl");
  const fetched = P.requested.length, pill = (name, key) => ({ target: { closest: (sel) => (sel === "[data-pill]" ? { dataset: { pill: name, key } } : null) } });
  G.brClick(pill("br-ta", "props"));
  assert.equal(rowsOf(els["board-top-alpha"].outerHTML).length, 1); assert.ok(els["board-top-alpha"].outerHTML.includes('id="board-top-alpha"'));
  G.brClick(pill("br-sp", "total"));
  assert.ok(els["board-betting-splits"].outerHTML.includes("<span>66% Over</span>") && els["board-betting-splits"].outerHTML.includes('id="board-betting-splits"'));
  G.brChange({ target: { matches: (s) => s === "[data-br]", dataset: { br: "sp-game" }, value: "2" } });
  assert.ok(els["board-betting-splits"].outerHTML.includes('<option value="2" selected>'));
  G.brClick(pill("br-mp", "season")); assert.ok(els["board-model-projections"].outerHTML.includes('data-key="season" role="tab" aria-selected="true"'));
  G.brChange({ target: { matches: (s) => s === "[data-br]", dataset: { br: "mp-market" }, value: "total" } });
  assert.ok(els["board-model-projections"].outerHTML.includes('<option value="total" selected>'));
  assert.equal(els["board-season-perf"], undefined, "the left column was not touched");
  G.brClick({ target: { closest: () => null } }); G.brClick({ target: {} }); G.brChange({ target: {} });
  assert.equal(P.requested.length, fetched, "no refetch");
});

test("the shared Top Alpha row (data.js) serves both +EV and the sport page: odds cell by default, aux replaces it, '' removes it", () => {
  const G = populated().G, o = { kind: "line", sport: "nfl", game_pk: 7, matchup: `${A} @ ${H}`, market: "moneyline", side: "home", odds: 120, modelProb: 0.6, impliedProb: 0.5, edgePp: 10 };
  const std = G.topAlphaRow(o, 0, new Map());
  assert.ok(std.includes("+120") && std.includes("ca-ev-ta-odds") && !std.includes("ca-ev-at") && !std.includes("noaux"));
  const aux = G.topAlphaRow(o, 1, new Map(), { aux: "<i>W</i>", at: true });
  assert.ok(aux.includes("<i>W</i>") && aux.includes("ca-ev-at") && !aux.includes("+120") && aux.includes("#2"));
  assert.ok(G.topAlphaRow(o, 0, new Map(), { aux: "" }).includes("ca-ev-ta-noaux"));
  assert.ok(fs.readFileSync(new URL("../js/pages/ev.js", import.meta.url), "utf8").includes("topAlphaRow(o, i, D.lineBy)"), "ev.js uses it");
  assert.ok(!fs.readFileSync(new URL("../js/pages/board-right.js", import.meta.url), "utf8").includes('class="ca-ev-ta"'), "no copied row markup");
});

test("donut: a split ring takes a track colour and a butt cap (default unchanged)", () => {
  const G = populated().G;
  assert.ok(G.donut(0.5).includes('stroke="#E9E5DB"') && G.donut(0.5).includes('stroke-linecap="round"'));
  const d = G.donut(0.6, { color: "#112233", track: "#445566", cap: "butt" });
  assert.ok(d.includes('stroke="#445566"') && d.includes('stroke="#112233"') && d.includes('stroke-linecap="butt"'));
});

test("CSS contract: the right-column rules exist (result chip cell, no-aux row, donut, tabs, bands table) and use the fluid type tokens, no fixed table font", () => {
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  for (const sel of [".ca-ev-ta-noaux{", ".ca-br-ta .ca-ev-ta{", ".ca-br-donut{", ".ca-br-tabs .ca-pills{", ".ca-br-table{", ".ca-br-bar{"]) assert.ok(css.includes(sel), sel);
  assert.match(css, /@container \(max-width:330px\)\{\.ca-br-pills \.ca-bl-l\{display:none\}/, "short pill labels in a narrow rail");
  assert.match(css, /\.ca-br-table\{font-size:clamp\(/);
});

test("Top Alpha Edges, current week: a finished game with no result yet is graded-pending (a dash), not live; edges still to play rank first", async () => {
  // game 3's pick (alpha higher than the live one) started on Oct 1-2 and has no stored result; game 4 kicks off after NOW
  const live = [{ ...LIVE_OPP, game_pk: 3, matchup: `${A} @ ${H2}`, side: "away", commence_time: "2026-10-02T00:15:00Z", ev_best: 0.9, true_prob: 0.9 }, LIVE_OPP];
  const html = await populated({ live, any: ANY, cur: [ANY[2], ANY[3]], results: [] }).G.buildBoardPage("nfl"), rows = rowsOf(card(html, IDS[0]));
  assert.equal(rows.length, 2);
  assert.ok(rows[0].includes("game=4") && rows[1].includes("game=3"), "the live edge first even though the finished one has the higher alpha");
  assert.ok(rows[1].includes("Not graded yet") && !rows[1].includes("ca-res "), "finished, not graded: a dash");
  // only finished, ungraded picks in the column: the result column still renders (a dash), the rows are not shown as live
  const onlyDone = await populated({ live: [live[0]], any: ANY, cur: [ANY[2]], results: [] }).G.buildBoardPage("nfl"), od = rowsOf(card(onlyDone, IDS[0]));
  assert.ok(od.length === 1 && od[0].includes("Not graded yet") && !od[0].includes("ca-ev-ta-noaux"), "no chip column missing");
  // a past week keeps ranking by alpha only
  const past = await populated({ search: "?week=3" }).G.buildBoardPage("nfl");
  assert.ok(rowsOf(card(past, IDS[0]))[0].includes("Rec Yds"));
});

test("the stored +EV prop picks are read in a total order (side is part of the sort) so paged reads cannot skip or repeat rows", async () => {
  const P = populated({ search: "?week=3" }); await P.G.buildBoardPage("nfl");
  assert.ok(P.requested.some((u) => u.startsWith("ev_prop_picks") && u.includes("line.asc,side.asc,model_version.asc")), P.requested.filter((u) => u.startsWith("ev_prop_picks")).join("\n"));
});
