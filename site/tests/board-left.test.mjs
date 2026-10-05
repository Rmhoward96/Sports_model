import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
// The sport pages' left column (js/pages/board-left.js): Season Performance, Model vs. Market, Market Intelligence, Key Insights.
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/ev.js", "js/pages/board.js", "js/pages/board-left.js", "js/boot.js"];
const NOW = Date.parse("2026-10-05T16:00:00Z");     // Monday 12:00 ET: NFL week 4 (Sep 29 - Oct 5), CFB week 5
class FakeDate extends Date { constructor(...a) { if (a.length) super(...a); else super(NOW); } static now() { return NOW; } }
const quiet = () => { const logs = { warn: [], error: [] }; return { logs, console: { ...console, warn: (...a) => logs.warn.push(a.join(" ")), error: (...a) => logs.error.push(a.join(" ")) } }; };
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };

// 12 NFL games: 6 in week 3 (Sep 24 - 28) and 6 in week 4 (Sep 30 - Oct 4). Every home team is the favourite (-3), the model picks the home side
// (pred margin 5), over 47 vs 44. Outcomes by game index: home margin -4 (i % 3 == 0) else +7; total 40 (i % 4 == 0) else 50.
const PAIRS = [["Buffalo Bills", "Miami Dolphins"], ["Detroit Lions", "Chicago Bears"], ["Kansas City Chiefs", "Denver Broncos"], ["Dallas Cowboys", "Philadelphia Eagles"],
  ["Green Bay Packers", "Minnesota Vikings"], ["Buffalo Bills", "Detroit Lions"], ["Dallas Cowboys", "Houston Texans"], ["Tampa Bay Buccaneers", "Seattle Seahawks"],
  ["Baltimore Ravens", "Arizona Cardinals"], ["Las Vegas Raiders", "New England Patriots"], ["Miami Dolphins", "Kansas City Chiefs"], ["Chicago Bears", "Atlanta Falcons"]];
const DATES = ["2026-09-24", "2026-09-27", "2026-09-27", "2026-09-28", "2026-09-28", "2026-09-28", "2026-09-30", "2026-10-02", "2026-10-02", "2026-10-04", "2026-10-04", "2026-10-04"];
const accRow = (i, sport = "nfl") => {
  const [away, home] = PAIRS[i], am = i % 3 === 0 ? -4 : 7, at = i % 4 === 0 ? 40 : 50;
  return { sport, game_pk: 100 + i, game_date: DATES[i], home_team_name: home, away_team_name: away, win_prob: 0.6, predicted_winner: home, actual_winner: am > 0 ? home : away, winner_correct: am > 0,
    pred_margin: 5, actual_margin: am, pred_total: 47, actual_total: at, market_spread: -3, market_total: 44, spread_pick_correct: am - 3 > 0, total_pick_correct: at > 44 };
};
const ACC = PAIRS.map((_, i) => accRow(i));
const PNL = [{ game_date: "2026-09-27", sport: "nfl", market: "moneyline", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 15 }, { game_date: "2026-09-28", sport: "nfl", market: "spread", n: 3, wins: 2, losses: 1, pushes: 0, pnl: 25 },
  { game_date: "2026-10-02", sport: "nfl", market: "total", n: 2, wins: 0, losses: 2, pushes: 0, pnl: -22 }, { game_date: "2026-10-04", sport: "nfl", market: "moneyline", n: 4, wins: 3, losses: 1, pushes: 0, pnl: 40 }];
// props: x = (projection - line) / line, y = went over
const PROPS = Array.from({ length: 12 }, (_, i) => ({ game_pk: 100 + i, player_id: `p${i}`, market: "rec_yds", line: 50, lean: i % 2 ? "under" : "over", projection: 50 + (i - 6) * 3, result: i % 3 ? "hit" : "miss" }));

function populated({ search = "", sport = "nfl", acc = ACC, pnl = PNL, props = PROPS, starts = [], fail = [], rest = null, ranks = null, any = [], cur = [], q = null } = {}) {
  const requested = [];
  const sp = sport;
  const fetch = async (url) => {
    const { path, q: qs } = urlParts(url); requested.push(path + qs);
    if (fail.includes(path)) return { ok: false, status: 500, text: async () => "boom", json: async () => [] };
    let r = [];
    const own = (rows) => (qs.includes(`sport=eq.${sp}`) ? rows : []);
    if (path === "prediction_accuracy") r = own(acc).filter((x) => { const m = qs.match(/game_date=gte\.([\d-]+)/), n = qs.match(/game_date=lte\.([\d-]+)/); return (!m || x.game_date >= m[1]) && (!n || x.game_date <= n[1]); });
    else if (path === "predictions_any") r = own(any); else if (path === "predictions_current") r = own(cur);
    else if (path === "track_record_start") r = starts;
    else if (path === "game_closing_prices") r = ACC.flatMap((a) => [{ game_pk: a.game_pk, market: "moneyline", side: "home", close_dec: 1.7 }, { game_pk: a.game_pk, market: "moneyline", side: "away", close_dec: 2.3 }]);
    else if (path === "prediction_pnl_daily") r = own(pnl);
    else if (path === "nfl_prop_pnl") r = props;
    else if (path === "team_game_log") r = rest || [];
    else if (path === "power_rankings_current") r = ranks || [];
    return { ok: true, json: async () => r };
  };
  const G = loadScripts(FILES, { page: sport, globals: { Date: FakeDate, fetch, location: { search, href: `http://localhost/${sport}.html${search}` }, ...(q ? { console: q.console } : {}) } });
  return { G, requested };
}
// The board's D as the left cards receive it: wrap one card renderer and keep what it was called with.
async function withD(P, sport = "nfl") {
  let D = null; const orig = P.G.BOARD_CARDS["board-season-perf"];
  P.G.BOARD_CARDS["board-season-perf"] = (d) => { D = d; return orig(d); };
  const html = await P.G.buildBoardPage(sport);
  return { html, D };
}
// The left column's HTML only (between the left column's id and the centre column's).
const leftOf = (html) => html.slice(html.indexOf('id="board-left"'), html.indexOf('id="board-center"'));
const card = (html, id) => { const l = leftOf(html), i = l.indexOf(`id="${id}"`); if (i < 0) return ""; const nxt = ["board-season-perf", "board-model-market", "board-market-intel", "board-key-insights"].map((x) => l.indexOf(`id="${x}"`, i + 5)).filter((j) => j > 0); return l.slice(i, nxt.length ? Math.min(...nxt) : l.length); };
const text = (h) => h.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
const wl = (html) => { const m = card(html, "board-season-perf").match(/<span class="ca-bl-wl">(\d+) - (\d+)<\/span><small class="[^"]*">([\d.]+)%<\/small>/); return m ? { w: +m[1], l: +m[2], pct: +m[3] } : null; };
// Independent count of the graded picks (moneyline always, spread / total when the model has a side) of the games the filter keeps.
const expectTally = (keep) => { let w = 0, l = 0; for (const [i, a] of ACC.entries()) if (keep(a, i)) for (const g of [a.winner_correct, a.spread_pick_correct, a.total_pick_correct]) g ? w++ : l++; return { w, l }; };

test("the four cards mount in the left column in order, each with its title (and Track Record link, pills, rows)", async () => {
  const html = await populated().G.buildBoardPage("nfl"), l = leftOf(html);
  const ids = ["board-season-perf", "board-model-market", "board-market-intel", "board-key-insights"], at = ids.map((id) => l.indexOf(`id="${id}"`));
  assert.ok(at.every((i) => i > 0) && at.every((i, k) => !k || at[k - 1] < i), "all four, top to bottom");
  assert.ok(!l.includes("ca-board-slot"), "no empty mount point is left in the left column");
  const sp = card(html, "board-season-perf"), mm = card(html, "board-model-market"), mi = card(html, "board-market-intel"), ki = card(html, "board-key-insights");
  assert.ok(sp.includes("<h2>Season Performance</h2>") && sp.includes('href="track-record.html"') && sp.includes("Track Record →"));
  assert.deepEqual([...sp.matchAll(/class="ca-stat-lt">([^<]*)</g)].map((m) => m[1]), ["W - L", "Units", "ROI", "Avg. Edge"], "2x2 stat boxes");
  assert.ok(mm.includes("<h2>Model vs. Market</h2>") && mm.includes("CappingAlpha") && mm.includes("Market</span>") && mm.includes("ca-scatter") && mm.includes("Model Edge"));
  assert.deepEqual([...mm.matchAll(/data-pill="bl-mm" data-key="(\w+)"/g)].map((m) => m[1]), ["spread", "moneyline", "total", "props"]);
  assert.ok(mi.includes("<h2>Market Intelligence</h2>"));
  for (const t of ["Biggest Line Move", "Highest Confidence Edge", "Most Mispriced Total", "Average Market Divergence"]) assert.ok(mi.includes(t), t);
  assert.ok(ki.includes("<h2>Key Insights</h2>"));
  assert.deepEqual([...ki.matchAll(/data-insight="(\w+)"/g)].map((m) => m[1]).sort(), ["homefav", "overs"], "12 home favourites and 12 over picks qualify; 5 division games and no rest data do not");
  assert.ok(!/NaN|undefined|Infinity/.test(l), "no NaN / undefined in the column");
});

test("Season Performance = Track Record's numbers: W-L-P from trackTally(trackRows), units / ROI from prediction_pnl_daily ($10 a bet), edge vs the no-vig close", async () => {
  const P = populated(), html = await P.G.buildBoardPage("nfl"), sp = card(html, "board-season-perf");
  const season = expectTally(() => true);
  assert.deepEqual(wl(html), { w: season.w, l: season.l, pct: +(season.w / (season.w + season.l) * 100).toFixed(1) }, "the season record = every graded pick");
  const tr = P.G.trackTally(P.G.trackRows(ACC, [], new Map()));
  assert.equal(tr.w, season.w); assert.equal(tr.l, season.l);
  assert.ok(sp.includes("+5.8u") && sp.includes("11 bets"), "units = sum of pnl / 10 (15 + 25 - 22 + 40 = 58 -> +5.8u); bets = sum of n");
  assert.ok(sp.includes("+52.7%"), "ROI = pnl / (decided bets x $10) = 58 / (6 wins + 5 losses = 11 x 10) = 52.7%");
  // the edge: the model's 60% on the home side vs the close 1.70 / 2.30 (no-vig home = (1/1.7) / (1/1.7 + 1/2.3) = 57.5%) -> +2.5 on the 12 moneyline picks
  assert.ok(sp.includes("+2.5%") && sp.includes("12 ML picks"), text(sp));
  assert.ok(sp.includes("Season 2026 · Sep 24 – Oct 4, 2026"), "the caption names the period the numbers cover");
});

test("the record scope: a restart date (track_record_start) cuts the rows before it, and a failed scope load shows no record at all (fails closed)", async () => {
  const starts = [{ sport: "nfl", starts_at: "2026-09-29T17:00:00Z", model_version: "nfl-sim-ml-v2" }];
  const html = await populated({ starts }).G.buildBoardPage("nfl"), after = expectTally((a) => a.game_date >= "2026-09-29");
  assert.deepEqual({ w: wl(html).w, l: wl(html).l }, after, "only the games from the restart on");
  assert.ok(card(html, "board-season-perf").includes("Season 2026 · Sep 30 – Oct 4, 2026"));
  const q = quiet(), bad = await populated({ fail: ["track_record_start"], q }).G.buildBoardPage("nfl");
  assert.ok(card(bad, "board-season-perf").includes("This panel couldn't load.") && !wl(bad), "no number without the record scope");
  assert.ok(card(bad, "board-key-insights").includes("couldn't load") && card(bad, "board-model-market").includes("couldn't load"));
  assert.ok(q.logs.error.some((e) => /board left: record failed/.test(e)));
});

test("the selected period: a past week shows THAT week's numbers (title, caption, W-L, units), the current week opens on the season, the toggle switches", async () => {
  const past = await populated({ search: "?week=3" }).G.buildBoardPage("nfl"), sp = card(past, "board-season-perf");
  const wk3 = expectTally((a) => a.game_date >= "2026-09-22" && a.game_date <= "2026-09-28");
  assert.ok(sp.includes("<h2>Week 3 Performance</h2>") && sp.includes("Sep 22 – Sep 28, 2026"), "titled and dated by the week");
  assert.deepEqual({ w: wl(past).w, l: wl(past).l }, wk3);
  assert.ok(sp.includes("+4.0u") && sp.includes("5 bets"), "the week's units: (15 + 25) / 10 over 2 + 3 bets");
  assert.deepEqual([...sp.matchAll(/data-key="(\w+)" role="tab" aria-selected="(\w+)"/g)].map((m) => [m[1], m[2]]), [["period", "true"], ["season", "false"]], "Week 3 | Season, the week selected");
  assert.ok(sp.includes(">Week 3</button>") && sp.includes(">Season</button>"));
  const wk4 = await populated({ search: "?week=4" }).G.buildBoardPage("nfl"), c = await populated().G.buildBoardPage("nfl");
  assert.ok(card(c, "board-season-perf").includes("<h2>Season Performance</h2>"), "the current week opens on the season");
  assert.ok(card(wk4, "board-season-perf").includes("<h2>Season Performance</h2>"), "?week=4 is today's week (the default)");
  assert.ok(wl(past).w + wl(past).l < wl(c).w + wl(c).l, "a week is a part of the season");
  // the toggle: an explicit pick for THIS period wins; another period starts from its own default again
  const P = populated({ search: "?week=3" }), { D } = await withD(P);
  assert.equal(P.G.blScope(D).key, "period");
  P.G.blSetScope("season", P.G.blKey(D.period));
  assert.equal(P.G.blScope(D).key, "season"); assert.ok(P.G.blSeasonPerf(D).includes("<h2>Season Performance</h2>") && P.G.blSeasonPerf(D).includes("Season 2026"));
  const other = { ...D, period: { ...D.period, week: 2, from: "2026-09-15", to: "2026-09-21" } };
  assert.equal(P.G.blScope(other).key, "period", "the pick belongs to Week 3 only");
  const future = { ...D, period: { ...D.period, week: 6, from: "2026-10-13", to: "2026-10-19" } };
  assert.equal(P.G.blScope(future).key, "season", "a week still to come has nothing graded: it opens on the season");
});

test("a week before the published record (archived) says so instead of showing zeros; a week with no graded picks yet and an empty record say what is missing", async () => {
  const starts = [{ sport: "nfl", starts_at: "2026-09-29T17:00:00Z", model_version: "nfl-sim-ml-v2" }];
  const html = card(await populated({ search: "?week=3", starts }).G.buildBoardPage("nfl"), "board-season-perf");
  assert.ok(html.includes("Week 3 is before the published NFL record, which restarted Sep 29, 2026 with the ML v2 model. Earlier results are archived."), text(html));
  const none = card(await populated({ search: "?week=3", acc: [] }).G.buildBoardPage("nfl"), "board-season-perf");
  assert.ok(none.includes("No graded NFL picks in Week 3 yet.") && !none.includes("ca-bl-grid"));
  const seasonNone = card(await populated({ acc: [] }).G.buildBoardPage("nfl"), "board-season-perf");
  assert.ok(seasonNone.includes("No graded NFL picks in the published record yet."));
});

test("blPerf: a week's units come from the pnl rows of its dates only; no pnl rows -> units / ROI are dashes, W-L still shown", () => {
  const G = populated().G, rows = G.trackRows(ACC, [], new Map());
  const wk = G.blPerf("nfl", rows, PNL, new Map(), "2026-09-29", "2026-10-05");
  assert.equal(wk.units, 1.8); assert.equal(wk.bets, 6); assert.equal(wk.roi, 18 / (6 * 10) * 100, "decided bets: 3 wins + 3 losses");
  const none = G.blPerf("nfl", rows, [], new Map(), "2026-09-22", "2026-09-28");
  assert.equal(none.units, null); assert.equal(none.roi, null); assert.ok(none.w + none.l > 0);
  assert.equal(none.edge, null, "no closing prices: no edge");
  assert.deepEqual([...G.blWeekUnits("nfl", PNL)], [4, 1.8], "units per NFL week, oldest first (week 3: 15 + 25, week 4: -22 + 40)");
  assert.deepEqual([...G.blWeekUnits("nfl", [PNL[0], { ...PNL[3], game_date: "2026-10-11" }])], [1.5, 0, 4], "a week with no bets between the first and last is a zero bar");
});

test("Model vs. Market: under 10 graded points is an empty state; with data it binns, has the legend, the Market baseline and a real mean for the callout", async () => {
  const few = card(await populated({ acc: ACC.slice(0, 8), props: [] }).G.buildBoardPage("nfl"), "board-model-market");
  assert.ok(few.includes("Not enough graded games yet: 8 of 10 needed.") && !few.includes("ca-scatter") && !few.includes('class="ca-bl-edge"'), text(few));
  const none = card(await populated({ acc: [], props: [] }).G.buildBoardPage("nfl"), "board-model-market");
  assert.ok(none.includes("No graded games with a market line yet.") && !none.includes("data-pill"), "no data at all: no pills, just the message");
  const html = await populated().G.buildBoardPage("nfl"), mm = card(html, "board-model-market");
  assert.ok(mm.includes("ca-scatter") && mm.includes('stroke-dasharray="5 4"') && mm.includes("Share of games where the home team covered") && mm.includes('class="ca-bl-edge"'));
  assert.equal((mm.match(/class="ca-dot ca-sc-dot"/g) || []).length, 3, "12 games -> 3 equal-count bins");
  // every game has gap = 5 - 3 = +2 pts, so the mean |gap| is exactly 2.0
  assert.ok(mm.includes("<b>2.0 points</b>") && mm.includes("12 graded games"), text(mm));
  assert.ok(mm.includes('title="'), "a dot carries its sample and rate as a hover title");
  // the (i) states the reading
  assert.ok(mm.includes('class="ca-info" title="x = the model\'s home margin plus the market spread, in points'), "the reading is stated in the (i)");
});

test("blMvmPoints / blMvmBins: spread, total, moneyline (no-vig close) and props; pushes dropped; bins are equal-count and the mean is |gap|", () => {
  const G = populated().G, L = { accRec: ACC, closing: new Map(ACC.flatMap((a) => [[`${a.game_pk}|home`, -143], [`${a.game_pk}|away`, 130]])), props: PROPS };
  const sp = G.blMvmPoints("spread", L), tt = G.blMvmPoints("total", L), ml = G.blMvmPoints("moneyline", L), pr = G.blMvmPoints("props", L);
  assert.equal(sp.length, 12); assert.ok(sp.every((p) => p.x === 2)); assert.equal(sp.filter((p) => p.y === 1).length, 8, "home covers when the home margin is +7 (8 of 12)");
  assert.ok(tt.every((p) => p.x === 3)); assert.equal(tt.filter((p) => p.y === 1).length, 9, "over when the total is 50");
  const mkt = (143 / 243) / (143 / 243 + 100 / 230) * 100;      // the no-vig home probability of -143 / +130
  assert.equal(ml.length, 12); assert.ok(Math.abs(ml[0].m - mkt) < 1e-9 && Math.abs(ml[0].x - (60 - mkt)) < 1e-9, "x = the model's 60% minus the no-vig close");
  assert.equal(pr.length, 12); assert.ok(pr[0].x < 0 && pr[11].x > 0, "projection below / above the line");
  const push = G.blMvmPoints("spread", { accRec: [{ ...ACC[0], actual_margin: 3 }, ACC[1]] });
  assert.equal(push.length, 1, "a margin exactly on the spread is a push and is dropped");
  assert.equal(G.blMvmPoints("moneyline", { accRec: ACC, closing: new Map() }).length, 0, "no closing price -> no moneyline point");
  const b = G.blMvmBins(Array.from({ length: 40 }, (_, i) => ({ x: i - 20, y: i % 2 })));
  assert.equal(b.bins.length, 5); assert.ok(b.bins.every((x) => x.n === 8)); assert.equal(b.n, 40);
  assert.equal(G.blMvmBins(Array.from({ length: 9 }, () => ({ x: 1, y: 1 }))), null, "9 points: below the minimum");
  assert.equal(G.blMvmBins(Array.from({ length: 10 }, () => ({ x: -2, y: 1 }))).mean, 2, "the mean is of |gap|");
});

test("Model vs. Market pills: only the markets with data; Player Props is hidden without prop data and for CFB; the market pill redraws the card", async () => {
  const noProps = card(await populated({ props: [] }).G.buildBoardPage("nfl"), "board-model-market");
  assert.deepEqual([...noProps.matchAll(/data-pill="bl-mm" data-key="(\w+)"/g)].map((m) => m[1]), ["spread", "moneyline", "total"], "no prop data: no Player Props pill");
  const cfb = card(await populated({ sport: "cfb", acc: ACC.map((a) => ({ ...a, sport: "cfb" })), pnl: [] }).G.buildBoardPage("cfb"), "board-model-market");
  assert.ok(!cfb.includes('data-key="props"') && cfb.includes('data-key="moneyline"'), "props are NFL only");
  const P = populated(), { D } = await withD(P);
  P.G.blSetMarket("moneyline");
  const ml = P.G.blModelMarket(D);
  assert.ok(ml.includes("Share of games where the home team won") && ml.includes("prob. points") && ml.includes('aria-selected="true"'), "the moneyline reading, in probability points");
  P.G.blSetMarket("props");
  assert.ok(P.G.blModelMarket(D).includes("Share of props where the prop went over"));
  P.G.blSetMarket("total"); assert.ok(P.G.blModelMarket(D).includes("Share of games where the game went over"));
});

test("Key Insights threshold: a row needs 8 decided picks (pushes do not count); the card is hidden while no row qualifies; rows name their sample", async () => {
  const G = populated().G, mk = (n) => ({ accRec: ACC.slice(0, n), rec: G.trackRows(ACC.slice(0, n), [], new Map()) });
  assert.deepEqual(G.blInsights(mk(7), "nfl"), [], "7 games: no group reaches 8");
  const by = Object.fromEntries(G.blInsights(mk(8), "nfl").map((r) => [r.key, r]));
  assert.ok(by.homefav && by.homefav.n === 8 && by.homefav.title === "Home favorites", "8 home-favourite spread picks qualify exactly at the threshold");
  assert.ok(by.overs && by.overs.n === 8 && !by.unders && !by.group && !by.short, "over picks qualify too; nothing else has 8 games");
  assert.equal(by.homefav.pct, 5 / 8, "the home side covered (home margin +7) in games 1, 2, 4, 5, 7 of the first 8");
  assert.equal(by.homefav.stat, "Model spread picks 5-3 (63%) this season");
  // the sample counts decided picks: a push is neither a win nor a loss
  const push = (n) => { const a = ACC.slice(0, n).map((x, i) => (i === 0 ? { ...x, actual_margin: 3, spread_pick_correct: null } : x)); return { accRec: a, rec: G.trackRows(a, [], new Map()) }; };
  assert.equal(G.blInsights(push(8), "nfl").find((r) => r.key === "homefav"), undefined, "8 spread picks with a push: only 7 decided, no row");
  const p9 = G.blInsights(push(9), "nfl").find((r) => r.key === "homefav");
  assert.ok(p9 && p9.n === 8 && p9.stat === "Model spread picks 6-2-1 (75%) this season", JSON.stringify(p9));
  const l = G.blInsights(mk(12), "nfl");
  assert.ok(l.length <= 4 && l.every((r) => r.n >= 8));
  assert.ok(l.every((r, i) => !i || Math.abs(l[i - 1].pct - 0.5) >= Math.abs(r.pct - 0.5) - 1e-9), "ordered by distance from 50%");
  assert.equal(card(await populated({ acc: ACC.slice(0, 7), props: [] }).G.buildBoardPage("nfl"), "board-key-insights"), "", "no row qualifies: the card is hidden");
  assert.ok(card(await populated({ acc: ACC.slice(0, 8), props: [] }).G.buildBoardPage("nfl"), "board-key-insights").includes("Home favorites"));
});

test("Key Insights rows: Divisional matchups (NFL division map), Short-week unders (rest days from team_game_log), Conference matchups (CFB), wording follows the rate", () => {
  const G = populated().G, recOf = (rows) => G.trackRows(rows, [], new Map()), find = (L, sport, key) => G.blInsights(L, sport).find((r) => r.key === key);
  assert.equal(G.nflDivision("Buffalo Bills"), "AFC East"); assert.equal(G.nflDivision("Los Angeles Rams"), "NFC West"); assert.equal(G.nflDivision("Washington Commanders"), "NFC East"); assert.equal(G.nflDivision("Nobody"), "");
  assert.equal(find({ accRec: ACC, rec: recOf(ACC) }, "nfl", "group"), undefined, "5 division games: below the threshold");
  const dv = Array.from({ length: 9 }, (_, i) => ({ ...accRow(i < 5 ? 1 : 0), game_pk: 500 + i })), d9 = find({ accRec: dv, rec: recOf(dv) }, "nfl", "group");
  assert.ok(d9 && d9.title === "Divisional matchups" && d9.n === 9 && d9.stat === "56% model accuracy (5-4)", JSON.stringify(d9));
  assert.equal(find({ accRec: dv, rec: recOf(dv) }, "mlb", "group"), undefined, "no division / conference row outside NFL and CFB");
  // short week: a team with <= 5 days' rest; under picks need a model total below the market total
  const gm = Array.from({ length: 10 }, (_, i) => ({ ...accRow(5), game_pk: 600 + i, pred_total: 40, actual_total: i < 7 ? 38 : 50, total_pick_correct: i < 7 })), L = { accRec: gm, rec: recOf(gm) };
  const rest = (home, away) => new Map(gm.map((a) => [String(a.game_pk), { home, away }]));
  const s = find({ ...L, rest: rest(4, 7) }, "nfl", "short");
  assert.ok(s && s.title === "Short-week unders" && s.n === 10 && s.stat === "70% hit rate (7-3) · a team on 5 or fewer days' rest", JSON.stringify(s));
  assert.ok(find({ ...L, rest: rest(7, 5) }, "nfl", "short"), "either team counts (5 days is short)");
  assert.equal(find({ ...L, rest: rest(7, 6) }, "nfl", "short"), undefined, "6 / 7 days' rest is not a short week");
  assert.equal(find({ ...L, rest: rest(null, null) }, "nfl", "short"), undefined, "rest not derivable: no short-week row");
  assert.equal(find(L, "nfl", "short"), undefined, "no rest data loaded: no short-week row");
  // CFB: same-conference games from power_rankings (team id -> conference id), names -> ids via CFB_TEAM_ID
  const ids = G.CFB_TEAM_ID, names = Object.keys(ids).slice(0, 3), conf = new Map(names.map((n) => [String(ids[n]), "8"]));
  const cg = Array.from({ length: 9 }, (_, i) => ({ ...accRow(0, "cfb"), game_pk: 700 + i, home_team_name: names[0], away_team_name: names[1] }));
  const cc = find({ accRec: cg, rec: recOf(cg), conf }, "cfb", "group");
  assert.ok(cc && cc.title === "Conference matchups" && cc.n === 9);
  assert.equal(find({ accRec: cg, rec: recOf(cg), conf: new Map() }, "cfb", "group"), undefined, "unknown conferences: no row");
  assert.equal(find({ accRec: cg, rec: recOf(cg), conf: new Map([[String(ids[names[0]]), "8"], [String(ids[names[1]]), "5"]]) }, "cfb", "group"), undefined, "different conferences: not a conference game");
  // wording follows the rate: 6 of 8 under picks hit performs, 2 of 8 struggles, 4 of 8 is neutral
  const unders = (wins) => { const a = Array.from({ length: 8 }, (_, i) => ({ ...accRow(5), game_pk: 900 + i, pred_total: 40, actual_total: i < wins ? 38 : 50, total_pick_correct: i < wins })); return find({ accRec: a, rec: recOf(a) }, "nfl", "unders"); };
  assert.equal(unders(6).title, "Unders performing"); assert.equal(unders(2).title, "Unders struggling"); assert.equal(unders(4).title, "Unders"); assert.equal(unders(6).stat, "75% hit rate (6-2) this season");
});

test("Market Intelligence: the +EV Market Pulse rows of this sport only, escaped; a different period says it is the live market", async () => {
  const evil = 'Evil "Q" <b>Crew</b>';
  const html = await populated({ search: "?week=3" }).G.buildBoardPage("nfl"), mi = card(html, "board-market-intel");
  assert.ok(mi.includes("The live market, not Week 3.") && mi.includes("ca-ev-pr"));
  assert.ok(card(await populated().G.buildBoardPage("nfl"), "board-market-intel").indexOf("The live market") < 0, "the current week: no note");
  const G = populated().G, rows = G.marketPulseRows({ preds: [{ sport: "nfl", game_pk: 1, home_team_name: evil, away_team_name: "<i>x</i>", commence_time: "2026-10-06T00:00:00Z", pred_home_score: 30, pred_away_score: 24, market_total: 41.5 }],
    moves: [], splits: new Map(), opps: [], lineBy: new Map(), nowMs: NOW, today: "2026-10-05", graded: [] });
  assert.ok(rows.includes("Most Mispriced Total") && !/<b>Crew|<i>x/.test(rows) && rows.includes("&lt;"), "escaped");
  // ev.js renders the same rows
  assert.ok(G.evxPulse({ preds: [], moves: [], splits: new Map(), opps: [], lineBy: new Map(), nowMs: NOW, today: "2026-10-05", graded: [] }).includes("Average Market Divergence"));
});

test("MLB / NBA: honest empty states in every card; MLB names when it paused; nothing is invented", async () => {
  const mlb = await populated({ sport: "mlb", acc: [], pnl: [], any: [{ sport: "mlb", game_pk: 1, game_date: "2026-08-31", home_team_name: "Boston Red Sox", away_team_name: "New York Yankees", commence_time: "2026-08-31T23:00:00Z" }] }).G.buildBoardPage("mlb");
  const l = leftOf(mlb);
  assert.ok(!l.includes("ca-board-slot") && !l.includes("ca-bl-grid") && !l.includes("ca-scatter") && !l.includes("ca-bl-in\"") && !l.includes("ca-ev-pr"), "no numbers, chart, insight rows or pulse rows");
  assert.ok(card(mlb, "board-season-perf").includes("MLB model paused since Aug 31. No graded MLB picks in the published record."), text(card(mlb, "board-season-perf")));
  assert.ok(card(mlb, "board-model-market").includes("No graded MLB games to compare with the market."));
  assert.ok(card(mlb, "board-market-intel").includes("No live MLB market to read."));
  assert.ok(card(mlb, "board-key-insights").includes("No graded games to read insights from."));
  const nba = await populated({ sport: "nba", acc: [], pnl: [] }).G.buildBoardPage("nba");
  for (const id of ["board-season-perf", "board-model-market", "board-market-intel", "board-key-insights"]) assert.ok(card(nba, id).includes("No NBA model yet."), id);
  assert.ok(!/NaN|undefined/.test(leftOf(nba)));
  // an MLB record, if one exists, is shown as its historical record
  const hist = ACC.slice(0, 6).map((a) => ({ ...a, sport: "mlb", game_date: "2026-08-20" })), rec = await populated({ sport: "mlb", acc: hist, pnl: [] }).G.buildBoardPage("mlb");
  assert.ok(card(rec, "board-season-perf").includes("ca-bl-grid") && card(rec, "board-season-perf").includes("Published record"), "a historical MLB record is shown");
});

test("data: the left column's reads (record per sport, pnl per sport, props / rest days for NFL only), fail-closed scope, no margin_dist, memoised per sport", async () => {
  const P = populated(), D = await P.G.buildBoardPage("nfl");
  void D;
  const has = (p) => P.requested.filter((u) => u.startsWith(p));
  assert.equal(has("prediction_accuracy?sport=eq.nfl&actual_winner=not.is.null").length, 1, "the record's graded games (one page)");
  assert.equal(has("prediction_pnl_daily?sport=eq.nfl").length, 1);
  assert.equal(has("track_record_start").length, 1); assert.ok(has("nfl_prop_pnl?season=eq.2026").length === 1 && has("team_game_log?sport=eq.nfl&season=eq.2026").length === 1);
  assert.ok(!P.requested.some((u) => /margin_dist|total_dist/.test(u)));
  assert.ok(has("game_closing_prices?market=eq.moneyline").length >= 1, "closing prices of the record's games");
  await P.G.buildBoardPage("nfl");
  assert.equal(has("prediction_pnl_daily").length, 1, "a second render within a minute reuses the loaded record");
  const C = populated({ sport: "cfb", acc: ACC.map((a) => ({ ...a, sport: "cfb" })) }); await C.G.buildBoardPage("cfb");
  assert.ok(C.requested.some((u) => u.startsWith("power_rankings_current?sport=eq.cfb")) && !C.requested.some((u) => u.startsWith("nfl_prop_pnl") || u.startsWith("team_game_log")), "CFB reads conferences, not props / rest days");
  const M = populated({ sport: "mlb", acc: [], pnl: [] }); await M.G.buildBoardPage("mlb");
  assert.ok(!M.requested.some((u) => u.startsWith("ev_current") || u.startsWith("nfl_prop_pnl") || u.startsWith("power_rankings_current")), "a sport with no model reads only its (empty) record");
  const q = quiet(), F = await populated({ fail: ["prediction_pnl_daily"], q }).G.buildBoardPage("nfl");
  assert.ok(card(F, "board-season-perf").includes("couldn't load") && card(F, "board-key-insights").includes("Home favorites"), "one failed source only affects the cards that need it");
});

test("scatterChart: nothing for no finite dot; dots, dashed baseline and solid trend positioned by percentage; symmetric x domain", () => {
  const G = populated().G;
  assert.equal(G.scatterChart({ dots: [] }), ""); assert.equal(G.scatterChart({ dots: [{ x: NaN, y: 5 }] }), ""); assert.equal(G.scatterChart(), "");
  const h = G.scatterChart({ dots: [{ x: -4, y: 40, n: 5, title: 'a "b"' }, { x: 4, y: 60, n: 10 }], line: [{ x: -4, y: 40 }, { x: 4, y: 60 }], flat: 50 });
  assert.equal((h.match(/ca-sc-dot/g) || []).length, 2); assert.ok(h.includes('title="a &quot;b&quot;"'), "dot titles escaped");
  assert.ok(h.includes("stroke-dasharray") && h.includes('stroke="var(--navy)"'));
  assert.match(h, /left:\d+\.\d+%;top:60\.00%/, "y = 60 sits at 40% from the top of the 0..100 axis");
  const left = (re) => +h.match(re)[1];
  assert.ok(Math.abs(left(/style="left:(\d+\.\d+)%;top:40\.00%/) + left(/style="left:(\d+\.\d+)%;top:60\.00%/) - 100) < 1e-6, "x = -4 and +4 are mirrored around the centre (symmetric domain)");
  assert.ok(left(/style="left:(\d+\.\d+)%;top:60\.00%/) < 100, "headroom: the outermost dot is not on the edge");
});

// ---- interaction ----------------------------------------------------------------------------------
test("interaction: the Model vs. Market pill and the Week / Season toggle redraw their own card from the cached load (no refetch); the toggle choice is per period", async () => {
  const els = {}, doc = { body: { dataset: { page: "nfl" }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: {}, createElement: () => ({}),
    querySelector: () => null, getElementById: (id) => (els[id] = els[id] || { id, outerHTML: "" }), querySelectorAll: () => [], addEventListener() {} };
  const requested = []; const G = loadScripts(FILES, { page: "nfl", globals: { document: doc, Date: FakeDate, location: { search: "?week=3", href: "http://localhost/nfl.html?week=3" },
    fetch: async (url) => { const { path } = urlParts(url); requested.push(path); return { ok: true, json: async () => (path === "prediction_accuracy" ? ACC : path === "prediction_pnl_daily" ? PNL : path === "nfl_prop_pnl" ? PROPS : path === "game_closing_prices" ? [] : []) }; } } });
  await G.buildBoardPage("nfl");
  const fetched = requested.length, pill = (name, key) => ({ target: { closest: (sel) => (sel === "[data-pill]" ? { dataset: { pill: name, key } } : null) } });
  G.blClick(pill("bl-mm", "total"));
  assert.ok(els["board-model-market"].outerHTML.includes("Share of games where the game went over") && els["board-model-market"].outerHTML.includes('id="board-model-market"'), "the card was redrawn in place with the total reading");
  assert.equal(els["board-season-perf"], undefined, "the other card was not touched");
  G.blClick(pill("bl-scope", "season"));
  assert.ok(els["board-season-perf"].outerHTML.includes("<h2>Season Performance</h2>") && els["board-season-perf"].outerHTML.includes('data-key="season" role="tab" aria-selected="true"'));
  G.blClick(pill("bl-scope", "period"));
  assert.ok(els["board-season-perf"].outerHTML.includes("<h2>Week 3 Performance</h2>"));
  G.blClick({ target: { closest: () => null } }); G.blClick({ target: {} });
  assert.equal(requested.length, fetched, "no refetch");
});

test("Key Insights / Model vs. Market wording: \"this season\" only when the record starts with the season; a later record start (NFL restart) says \"in the published record\"", async () => {
  const G = populated().G, starts = (iso) => new Map([["nfl", { starts_at: iso }]]);
  assert.equal(G.blWhen({ starts: new Map() }, "nfl", "2026-10-05"), "this season");
  assert.equal(G.blWhen({ starts: starts("2026-09-29T17:00:00Z") }, "nfl", "2026-10-05"), "in the published record");
  assert.equal(G.blWhen({ starts: starts("2026-07-01T17:00:00Z") }, "nfl", "2026-10-05"), "this season", "a record that starts before the season does not shorten it");
  assert.match(G.blInsights({ accRec: ACC, rec: G.trackRows(ACC, [], new Map()) }, "nfl", "in the published record")[0].stat, /in the published record$/);
  const st = [{ sport: "nfl", starts_at: "2026-09-25T17:00:00Z", model_version: "nfl-sim-ml-v2" }], html = await populated({ starts: st }).G.buildBoardPage("nfl");
  assert.ok(card(html, "board-key-insights").includes("in the published record") && !card(html, "board-key-insights").includes("this season"));
  assert.ok(card(html, "board-model-market").includes("over every graded game in the published record"), "the (i) says it too");
  const plain = await populated().G.buildBoardPage("nfl");
  assert.ok(card(plain, "board-key-insights").includes("this season") && !card(plain, "board-key-insights").includes("published record"));
});
