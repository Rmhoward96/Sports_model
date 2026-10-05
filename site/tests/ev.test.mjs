import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/ev.js", "js/boot.js"];
const g = loadScripts(FILES);
const O = (o) => ({ kind: "line", sport: "nfl", market: "moneyline", book: "draftkings", edgePp: 5, evPct: 5, alpha: 75, tier: "STRONG", commence: "2026-10-05T17:00:00Z", matchup: "BUF @ ATL", playerName: null, ...o });
test("filters by sport, kind, min edge, tier, book and search; sorts", () => {
  const opps = [O({ game_pk: 1 }), O({ game_pk: 2, kind: "prop", playerName: "Josh Allen", edgePp: 12, evPct: 9, alpha: 90, tier: "HIGH" }),
                O({ game_pk: 3, sport: "cfb", edgePp: 1, evPct: 2, alpha: 66, tier: "MEDIUM", book: "fanduel" })];
  assert.deepEqual(g.evFilter(opps, { sport: "nfl" }).map((o) => o.game_pk), [1, 2]);
  assert.deepEqual(g.evFilter(opps, { kind: "prop" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { minEdge: 2 }).map((o) => o.game_pk).sort(), [1, 2]);
  assert.deepEqual(g.evFilter(opps, { tier: "HIGH" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { book: "fanduel" }).map((o) => o.game_pk), [3]);
  assert.deepEqual(g.evFilter(opps, { q: "allen" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { sort: "edge" }).map((o) => o.game_pk), [2, 1, 3]);
  assert.deepEqual(g.evFilter(opps, { sort: "alpha" }).map((o) => o.game_pk), [2, 1, 3]);
});

test("evFilter: market, date window (ET), search over matchup, time sort ascending, no sort keeps input order, input untouched", () => {
  const opps = [O({ game_pk: 1, commence: "2026-10-07T17:00:00Z", evPct: 3 }), O({ game_pk: 2, market: "spread", commence: "2026-10-05T17:00:00Z", evPct: 9, matchup: "Detroit Lions @ Kansas City Chiefs" }),
                O({ game_pk: 3, commence: "2026-10-06T00:30:00Z", evPct: 6 }), O({ game_pk: 4, commence: "2026-10-20T17:00:00Z", edgePp: null })];
  const today = "2026-10-05";
  assert.deepEqual(g.evFilter(opps, {}).map((o) => o.game_pk), [1, 2, 3, 4], "no filters, no sort: input order");
  assert.deepEqual(g.evFilter(opps, { market: "spread" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { date: "today", today }).map((o) => o.game_pk), [2, 3], "00:30Z on the 6th is still the 5th in ET");
  assert.deepEqual(g.evFilter(opps, { date: "tomorrow", today }).map((o) => o.game_pk), []);
  assert.deepEqual(g.evFilter(opps, { date: "week", today }).map((o) => o.game_pk), [1, 2, 3]);
  assert.deepEqual(g.evFilter(opps, { q: "  KANSAS " }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { sort: "time" }).map((o) => o.game_pk), [2, 3, 1, 4]);
  assert.deepEqual(g.evFilter(opps, { sort: "ev" }).map((o) => o.game_pk), [2, 3, 4, 1]);   // EV 9, 6, 5, 3
  assert.deepEqual(g.evFilter(opps, { minEdge: 0 }).map((o) => o.game_pk), [1, 2, 3], "a missing edge never passes a minimum");
  assert.deepEqual(g.evFilter(opps, { kind: "all", sport: "", tier: "", book: "", market: "" }).length, 4, "empty / all values do not filter");
  assert.deepEqual(opps.map((o) => o.game_pk), [1, 2, 3, 4]);
});

test("evBucketRows: game lines and props become edgeBuckets input; rows without edge or profit are skipped", () => {
  const lines = [{ edgePp: 11, profitUnits: 0.9, won: true }, { edgePp: 3, profitUnits: -1, won: false }, { edgePp: null, profitUnits: 1, won: true }, { edgePp: 6, profitUnits: null, won: true }];
  const props = [{ edgePp: 6, profitUnits: 0.91, won: true }, { edgePp: -1, profitUnits: -1, won: false }];
  const rows = g.evBucketRows(lines, props);
  assert.equal(rows.length, 4);
  const b = g.edgeBuckets(rows, [10, 5, 2, 0]);
  assert.deepEqual(b.map((x) => [x.label, x.n, x.hits]), [["> 10%", 1, 1], ["5% to 10%", 1, 1], ["2% to 5%", 1, 0], ["0% to 2%", 0, 0], ["< 0%", 1, 0]]);
  assert.ok(Math.abs(b[1].units - 0.91) < 1e-9);
  assert.deepEqual(g.evBucketRows(null, undefined), []);
});

test("gradedPropPicks: win/loss only, price joined from the stored pick, profit is already in units", () => {
  const starts = new Map([["nfl", { starts_at: "2026-09-29T17:00:00Z" }]]);
  const res = (o) => ({ sport: "nfl", game_pk: 1, player_id: "p1", market: "rec_yds", line: 64.5, model_version: "v1", side: "over", model_prob: 0.6, result: "win", profit: 0.9091, clv: 0.02, commence_time: "2026-10-04T17:00:00Z", ...o });
  const picks = [{ game_pk: 1, player_id: "p1", market: "rec_yds", line: 64.5, model_version: "v1", best_price: -110 }];
  const rows = g.gradedPropPicks([res({}), res({ result: "push", profit: 0 }), res({ player_id: "p9" }), res({ game_pk: 2, result: "loss", profit: -1, commence_time: "2026-09-20T17:00:00Z" }),
    res({ game_pk: 1, player_id: "p1", result: null })], picks, starts);
  assert.equal(rows.length, 2, "push, ungraded and out-of-record rows are dropped");
  const r = rows[0];
  assert.equal(r.won, true); assert.equal(r.profitUnits, 0.9091); assert.equal(r.clv, 0.02); assert.equal(r.date, "2026-10-04");
  assert.ok(Math.abs(r.implied - 110 / 210) < 1e-9); assert.ok(Math.abs(r.edgePp - (0.6 - 110 / 210) * 100) < 1e-9);
  assert.equal(rows[1].implied, null, "no stored price: no implied prob, no edge");
  assert.equal(rows[1].edgePp, null);
});

test("daily units, daily mean edge and mean CLV over a date window", () => {
  const pnl = [{ game_date: "2026-10-04", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 5 }, { game_date: "2026-10-04", n: 1, wins: 1, losses: 0, pushes: 0, pnl: 10 },
               { game_date: "2026-10-01", n: 1, wins: 0, losses: 1, pushes: 0, pnl: -10 }, { game_date: "2026-09-01", n: 1, wins: 1, losses: 0, pushes: 0, pnl: 99 }];
  const u = g.evxDailyUnits(pnl, 3, "2026-10-05");
  assert.deepEqual(u.map((x) => [x.date, x.units]), [["2026-10-03", 0], ["2026-10-04", 1.5], ["2026-10-05", 0]]);
  const graded = [{ date: "2026-10-04", edgePp: 4 }, { date: "2026-10-04", edgePp: 8 }, { date: "2026-10-05", edgePp: null }, { date: "2026-10-03", edgePp: 2 }];
  const e = g.evxEdgeByDay(graded, 3, "2026-10-05");
  assert.deepEqual(e.map((x) => [x.date, x.v]), [["2026-10-03", 2], ["2026-10-04", 6], ["2026-10-05", null]]);
  const clv = g.evxMeanClv([{ date: "2026-10-04", clv: 0.04 }, { date: "2026-10-05", clv: -0.01 }, { date: "2026-10-05", clv: null }, { date: "2026-09-01", clv: 0.5 }], "2026-09-29", "2026-10-05");
  assert.equal(clv.n, 2); assert.ok(Math.abs(clv.pct - 1.5) < 1e-9);
  assert.equal(g.evxMeanClv([], "2026-09-29", "2026-10-05"), null);
});

test("EV distribution bins and the most-mispriced total", () => {
  const bins = g.evxBins([{ evPct: -7 }, { evPct: 2 }, { evPct: 3 }, { evPct: 12 }, { evPct: 40 }, { evPct: 15 }]);
  assert.deepEqual(bins.map((b) => [b.label, b.n]), [["<-5", 1], ["-5–0", 0], ["0–5", 2], ["5–10", 0], ["10–15", 1], [">15", 2]]);
  assert.ok(bins[0].color.includes("red") && bins[2].color.includes("green"));
  const NOW = Date.parse("2026-10-05T12:00:00Z"), future = "2026-10-05T17:00:00Z", past = "2026-10-05T08:00:00Z";
  const preds = [{ sport: "nfl", game_pk: 1, home_team_name: "A", away_team_name: "B", pred_home_score: 24, pred_away_score: 20.3, market_total: 40.5, commence_time: future },
                 { sport: "nfl", game_pk: 2, home_team_name: "C", away_team_name: "D", pred_home_score: 20, pred_away_score: 20, market_total: 47, commence_time: future },
                 { sport: "nfl", game_pk: 3, home_team_name: "E", away_team_name: "F", pred_home_score: 20, pred_away_score: 20, market_total: null, commence_time: future },
                 { sport: "nfl", game_pk: 4, home_team_name: "G", away_team_name: "H", pred_home_score: 10, pred_away_score: 10, market_total: 60, commence_time: past },   // already started: the biggest gap, must be excluded
                 { sport: "nfl", game_pk: 5, home_team_name: "I", away_team_name: "J", pred_home_score: 10, pred_away_score: 10, market_total: 60 }];   // unknown kickoff: unverifiable, excluded
  const m = g.evxMispricedTotal(preds, NOW);
  assert.equal(m.pred.game_pk, 2); assert.equal(m.market, 47); assert.equal(m.model, 40); assert.equal(m.diff, -7);
  assert.equal(g.evxMispricedTotal(preds.slice(3), NOW), null, "only started / undated games -> nothing to show");
  assert.equal(g.evxMispricedTotal([]), null);
});

test("lineMoveInfo: spread, total, moneyline text, ticket split", () => {
  const p = { sport: "nfl", game_pk: 1, home_team_name: "Atlanta Falcons", away_team_name: "Buffalo Bills" };
  const splits = new Map([["1|spread|home", { ticket_pct: 62.4 }]]);
  const sp = g.lineMoveInfo({ game_pk: 1, market: "spread", side: "home", open_line: -2.5, cur_line: -4, open_price: -110, cur_price: -110 }, p, splits);
  assert.equal(sp.title, "Falcons -2.5 → -4"); assert.equal(sp.chg, "−1.5 pts"); assert.equal(sp.d, -1.5); assert.equal(sp.tickets, "62% of tickets on Falcons");
  const tot = g.lineMoveInfo({ game_pk: 1, market: "total", side: "over", open_line: 41, cur_line: 42.5 }, p, null);
  assert.equal(tot.chg, "+1.5 pts"); assert.equal(tot.tickets, ""); assert.match(tot.title, /O\/U 41 → 42.5/);
  const ml = g.lineMoveInfo({ game_pk: 1, market: "moneyline", side: "away", open_price: 120, cur_price: 100 }, p, null);
  assert.equal(ml.title, "Bills ML +120 → +100"); assert.equal(ml.chg, "+4.5%"); assert.equal(ml.d > 0, true);   // the price shortened: implied prob up
});

test("URL params <-> filter state: valid values only, defaults stay out of the URL", () => {
  const s = g.evxParse("?sport=cfb&market=spread&book=fanduel&min=5&conf=HIGH&date=week&q=Allen&tab=prop&sort=time");
  assert.deepEqual(JSON.parse(JSON.stringify(s)), { sport: "cfb", market: "spread", book: "fanduel", minEdge: 5, tier: "HIGH", date: "week", q: "Allen", kind: "prop", sort: "time" });
  const bad = g.evxParse("?min=abc&conf=NOPE&date=never&tab=zzz&sort=zzz&sport=");
  assert.deepEqual(JSON.parse(JSON.stringify(bad)), { sport: "", market: "", book: "", minEdge: 0, tier: "", date: "", q: "", kind: "all", sort: "edge" });
  assert.equal(g.evxQuery(bad), "");
  assert.equal(g.evxQuery(s), "sport=cfb&market=spread&book=fanduel&min=5&conf=HIGH&date=week&q=Allen&tab=prop&sort=time");
  assert.deepEqual(JSON.parse(JSON.stringify(g.evxParse("?" + g.evxQuery(s)))), JSON.parse(JSON.stringify(s)));
});

// ---- populated render --------------------------------------------------------------------------
const EVIL_AWAY = 'Evil "Q" <b>Crew</b>', HOME = 'Kansas City "Chiefs"', PLAYER = 'O"Brien <i>Zed</i>';
const urlParts = (url) => { const u = new URL(url); return { path: u.pathname.split("/").pop(), q: u.search }; };
function populated({ search = "", moves = [], lastBuild = true } = {}) {
  const NOW = Date.now(), iso = (h) => new Date(NOW + h * 36e5).toISOString(), day = (d) => new Date(NOW - d * 864e5).toISOString().slice(0, 10);
  const line = (o) => ({ sport: "nfl", game_pk: 1, matchup: `${EVIL_AWAY} @ ${HOME}`, market: "moneyline", side: "home", true_prob: 0.7, best_line_implied: 0.5,
    best_price: 100, best_book: "draftkings", ev_best: 0.3, is_pick: true, commence_time: iso(48), ...o });
  const evNfl = [line({}), line({ game_pk: 2, commence_time: iso(96), ev_best: 0.12, true_prob: 0.6, side: "away", best_book: "fanduel" }), line({ game_pk: 3, ev_best: 0.1, true_prob: 0.58, is_pick: false })];
  const props = [{ sport: "nfl", game_pk: 5, matchup: `${EVIL_AWAY} @ ${HOME}`, market: "rec_yds", side: "over", line: 64.5, player_name: PLAYER, model_prob: 0.62, best_price: -110, best_book: "fanduel", ev_best: 0.15, is_pick: true, commence_time: iso(72) }];
  const pred = { game_pk: 1, sport: "nfl", home_team_name: HOME, away_team_name: EVIL_AWAY, commence_time: iso(48), home_win_prob: 0.6, pred_home_score: 30, pred_away_score: 20, market_spread: -3, market_total: 41.5 };
  const evPnl = [{ game_date: day(5), sport: "nfl", market: "moneyline", n: 10, wins: 6, losses: 3, pushes: 1, pnl: 50 }, { game_date: day(2), sport: "nfl", market: "prop", n: 2, wins: 1, losses: 1, pushes: 0, pnl: 0 }];
  const results = [{ sport: "nfl", game_pk: 11, market: "moneyline", side: "home", won: true, clv: 0.03 }, { sport: "nfl", game_pk: 12, market: "moneyline", side: "home", won: false, clv: -0.01 }];
  const gradedPick = (pk) => ({ sport: "nfl", game_pk: pk, market: "moneyline", side: "home", true_prob: 0.6, best_price: 100, created_at: iso(-100), commence_time: iso(-24) });
  const propRes = [{ sport: "nfl", game_pk: 21, player_id: "p1", market: "rec_yds", line: 50.5, model_version: "v1", side: "over", model_prob: 0.6, result: "win", profit: 0.9091, clv: 0.02, commence_time: iso(-24) }];
  const propPk = [{ game_pk: 21, player_id: "p1", market: "rec_yds", line: 50.5, model_version: "v1", best_price: -110 }];
  const requested = [];
  const fetch = async (url) => {
    const { path, q } = urlParts(url); requested.push(path + q);
    let rows = [];
    if (path === "predictions_current") rows = q.includes("sport=eq.nfl") ? [pred] : [];
    else if (path === "ev_current") rows = q.includes("sport=eq.nfl") ? evNfl : [];
    else if (path === "ev_prop_picks_current") rows = q.includes("sport=eq.nfl") ? props : [];
    else if (path === "ev_pnl_daily") rows = evPnl;
    else if (path === "ev_results") rows = results;
    else if (path === "ev_prop_results") rows = propRes;
    else if (path === "ev_prop_picks") rows = q.includes("select=created_at") ? (lastBuild ? [{ created_at: iso(-1) }] : []) : q.includes("created_at=gte") ? [] : propPk;
    else if (path === "ev_picks") rows = q.includes("select=created_at") ? (lastBuild ? [{ created_at: iso(-1) }] : []) : q.includes("created_at=gte") ? [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", created_at: iso(-1) }] : [gradedPick(11), gradedPick(12)];
    else if (path === "line_moves_current") { if (moves === 404) return { ok: false, status: 404, text: async () => "not found", json: async () => ({}) }; rows = moves.map((m) => m(iso)); }
    return { ok: true, json: async () => rows };
  };
  const D = loadScripts(FILES, { page: "ev", globals: { fetch, location: { search, href: `http://localhost/ev.html${search}` } } });
  return { D, requested, NOW };
}
const sect = (html, from, to) => html.slice(html.indexOf(`id="${from}"`), to ? html.indexOf(`id="${to}"`) : undefined);

test("buildEvPage renders every mockup section from stubbed data; names escaped; no NaN / undefined", async () => {
  const { D, requested } = populated();
  const html = await D.buildEvPage();
  for (const t of [">+EV<", "Find the best expected value opportunities", "Last Updated", "Total +EV Opportunities", "Average Edge", "Highest Edge", "Model Hit Rate", "ROI",
    "Sport", "League", "Market Type", "Sportsbook", "Minimum Edge", "Confidence", "Date", "Search teams, players, or games",
    "All Opportunities (3)", "Game Lines (2)", "Player Props (1)", "Sort By", "+EV Opportunities", "Top Alpha Opportunities", "Market Pulse", "EV Distribution", "Performance by Edge Bucket",
    "Biggest Line Move", "Highest Confidence Edge", "Most Mispriced Total", "Average Market Divergence"]) assert.ok(html.includes(t), `missing ${t}`);
  const table = sect(html, "ev-table", "ev-rail");
  for (const h of ["#", "Sport", "Matchup / Player", "Market", "Best Odds", "Model Prob.", "Impl. Prob.", "Edge", "EV", "Alpha Score", "Confidence", "Game Time", "View"]) assert.ok(table.includes(`<th>${h}</th>`) || table.includes(`<th>${h}`) || table.includes(`">${h}</th>`), `column ${h}`);
  assert.equal((table.match(/<tr data-href/g) || []).length, 3, "all listed opportunities; the non-pick row is not listed");
  assert.ok(html.includes("&lt;b&gt;Crew&lt;/b&gt;") && html.includes("O&quot;Brien &lt;i&gt;Zed&lt;/i&gt;"));
  assert.ok(!html.includes("<b>Crew</b>") && !html.includes("<i>Zed</i>"));
  assert.ok(html.includes("No line moves captured yet."), "line_moves_current empty -> user-facing line");
  assert.ok(!/migration/i.test(html));
  assert.ok(!/NaN|undefined|null/.test(html.replace(/\bnull\b(?=[^<]*>)/g, "")), "no NaN / undefined leaks");
  assert.ok(requested.every((r) => !r.startsWith("prediction_pnl_daily")), "performance never reads prediction_pnl");
});

test("buildEvPage: one performance population for hit rate, ROI and the edge-bucket table", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const stats = html.slice(html.indexOf("ca-ev-stats"), html.indexOf('id="ev-filters"'));
  assert.ok(stats.includes("63.6%"), "hit rate = 7 wins / (7 + 4 losses) from ev_pnl_daily");
  assert.ok(stats.includes("+41.7%") && stats.includes("+5.0u"), "ROI 5.0u / 12 graded picks, units beside it");
  assert.ok(stats.includes("12 graded +EV picks"));
  assert.match(stats, /Total \+EV Opportunities[\s\S]*>3</, "3 listed opportunities");
  const b = sect(html, "ev-buckets", "ev-pulse") || html.slice(html.indexOf('id="ev-buckets"'));
  assert.ok(b.includes("3 graded +EV picks"), "2 game lines (edge 10pp, units 1 / -1) + 1 prop");
  assert.ok(b.includes("&gt; 10%") && b.includes("&lt; 0%"), "labels are escaped");
});

test("buildEvPage: filters from the URL narrow the table and tab counts; state survives a second build", async () => {
  const { D } = populated({ search: "?tab=prop&sort=time&q=zed" });
  let html = await D.buildEvPage();
  let table = html.slice(html.indexOf('id="ev-table"'), html.indexOf('id="ev-rail"'));
  assert.equal((table.match(/<tr data-href/g) || []).length, 1);
  assert.ok(table.includes("O&quot;Brien"));
  assert.match(html, /class="ca-pill on" data-pill="ev-tab" data-key="prop"/);
  assert.match(html, /<option value="time" selected>/);
  assert.match(html, /data-ev-q[^>]*value="zed"/);
  assert.ok(html.includes("All Opportunities (1)") && html.includes("Game Lines (0)") && html.includes("Player Props (1)"), "tab counts follow the filters except the tab itself");
  html = await D.buildEvPage();                               // the 5-minute re-render
  assert.match(html, /class="ca-pill on" data-pill="ev-tab" data-key="prop"/);
  assert.equal(D.evxState().kind, "prop");
  D.evxSet("q", "nobody");
  html = await D.buildEvPage();
  table = html.slice(html.indexOf('id="ev-table"'), html.indexOf('id="ev-rail"'));
  assert.ok(table.includes("No opportunities match these filters.") && table.includes("data-ev-reset"));
  assert.ok(html.includes("All Opportunities (0)"));
});

test("buildEvPage: line moves fill Biggest Line Move (with ticket split); a 404 on line_moves_current does not break the page", async () => {
  const mv = [(iso) => ({ game_pk: 1, market: "spread", side: "home", open_line: -2.5, open_price: -110, cur_line: -4, cur_price: -110, open_at: iso(-30), cur_at: iso(-1), commence_time: iso(48) })];
  const { D } = populated({ moves: mv });
  const html = await D.buildEvPage();
  const pulse = html.slice(html.indexOf('id="ev-pulse"'), html.indexOf('id="ev-dist"'));
  assert.ok(pulse.includes("−1.5 pts") && pulse.includes("-2.5 → -4"));
  assert.ok(!pulse.includes("No line moves captured yet."));
  const e = await populated({ moves: 404 }).D.buildEvPage();
  assert.ok(e.includes("No line moves captured yet.") && e.includes("Market Pulse"));
});

test("buildEvPage: empty sources render every empty state, no throw, no NaN / undefined", async () => {
  const E = loadScripts(FILES, { page: "ev", globals: { fetch: async () => ({ ok: true, json: async () => [] }) } });
  const html = await E.buildEvPage();
  for (const t of ["+EV Opportunities", "Top Alpha Opportunities", "Market Pulse", "EV Distribution", "Performance by Edge Bucket"]) assert.ok(html.includes(t), t);
  assert.ok(html.includes("No +EV opportunities on the board right now."));
  assert.ok(html.includes("No graded +EV picks with a flagged price yet."));
  assert.ok(!/NaN|undefined/.test(html));
});

test("a throwing card is isolated: placeholder in its own slot, the rest of the page renders", async () => {
  const logs = []; const quiet = { ...console, error: (...a) => logs.push(a.join(" ")), warn() {} };
  const E = loadScripts(FILES, { page: "ev", globals: { console: quiet, fetch: async () => ({ ok: true, json: async () => [] }) } });
  E.EV_RAIL[1][1] = () => { throw new Error("pulse broke"); };
  const html = await E.buildEvPage();
  assert.ok(html.includes("This panel couldn't load."));
  assert.ok(html.includes('id="ev-pulse"') && html.includes("Top Alpha Opportunities") && html.includes("Performance by Edge Bucket"));
  assert.ok(logs.some((l) => /Market Pulse/.test(l)));
});

test("top alpha opportunities: five at most, by alpha", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const top = html.slice(html.indexOf('id="ev-top"'), html.indexOf('id="ev-pulse"'));
  assert.equal((top.match(/class="ca-ev-ta"/g) || []).length, 3);
  assert.ok(top.indexOf("#1") < top.indexOf("#2"));
});

test("evxSet writes the changed filter into the URL (history.replaceState) and maps League onto Sport", () => {
  const calls = [];
  const E = loadScripts(FILES, { page: "ev", globals: { location: { search: "", href: "http://localhost/ev.html" }, history: { replaceState: (a, b, u) => calls.push(u) } } });
  E.evxSet("minEdge", "5"); E.evxSet("league", "cfb");
  assert.equal(calls.length, 2);
  assert.equal(new URL(calls[1]).search, "?sport=cfb&min=5");
  E.evxSet("minEdge", "7");   // not an offered value -> default
  assert.equal(new URL(calls[2]).search, "?sport=cfb");
});

test("ev_prop_picks prices are read page by page past the 1,000-row clamp (no 4000 cap, no warning)", async () => {
  const warns = [], offsets = [];
  const all = Array.from({ length: 2350 }, (_, i) => ({ game_pk: i, player_id: "p", market: "rec_yds", line: 1, model_version: "v", best_price: -110, created_at: "2026-09-30T00:00:00Z" }));
  const E = loadScripts(FILES, { page: "ev", globals: { console: { ...console, warn: (...a) => warns.push(a.join(" ")) },
    fetch: async (u) => {
      const { path, q } = urlParts(u);
      if (path !== "ev_prop_picks" || q.includes("select=created_at") || q.includes("created_at=gte")) return { ok: true, json: async () => [] };
      const off = +q.match(/offset=(\d+)/)[1]; offsets.push(off);
      assert.match(q, /order=created_at\.asc,game_pk\.asc,player_id\.asc,market\.asc,line\.asc,model_version\.asc&limit=1000&offset=/);
      return { ok: true, json: async () => all.slice(off, off + 1000) };
    } } });
  await E.buildEvPage();
  assert.deepEqual(offsets, [0, 1000, 2000]);
  assert.equal(warns.length, 0);
  assert.ok(!require_src("js/pages/ev.js").includes("4000"), "the old cap constant is gone");
});

test("Edge Bucket rows show the sample size n; clicking rows is handled once by the shell", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const b = html.slice(html.indexOf('id="ev-buckets"'));
  assert.ok(/<th>n<\/th>/.test(b));
  assert.match(b, /<tr title="2 of 3 graded picks"><td>5% to 10%<\/td><td class="muted">3<\/td>/, "n is a visible cell, not just the tooltip");
  assert.ok(!/tr\[data-href\]/.test(require_src("js/pages/ev.js")) && !/tr\[data-href\]/.test(require_src("js/pages/dashboard.js")), "no page-level row handlers");
});
import fs from "node:fs";
const require_src = (f) => fs.readFileSync(new URL("../" + f, import.meta.url), "utf8");

test("R19: +EV table and Top Alpha rail mark game-line probabilities as the sharp fair price; props carry none", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const table = sect(html, "ev-table", "ev-rail");
  assert.match(table, /<th title="Model for player props; sharp fair price \(Pinnacle no-vig\) for game lines">Model Prob\.<\/th>/);
  const rows = table.split("<tr data-href").slice(1);
  assert.match(rows.find((r) => r.includes("game=1")), /70\.0%<span class="ca-fair"[^>]*>fair<\/span>/);
  assert.ok(!rows.find((r) => r.includes("game=5")).includes("ca-fair"));
  // E3: the subtitle is the short mockup line; the R19 wording lives in its (i) tooltip
  assert.ok(table.includes("Model projections, current odds, and expected value across all sports."));
  assert.match(table, /<span class="ca-info" title="[^"]*model probability for props, sharp fair price \(Pinnacle no-vig\) for game lines[^"]*"/);
  assert.ok(!/<p[^>]*>[^<]*sharp fair price/.test(table), "the long R19 sentence is not visible subtitle text");
  const top = sect(html, "ev-top", "ev-pulse"), items = top.split('class="ca-ev-ta"').slice(1);
  assert.match(items.find((x) => x.includes("game=1") || x.includes("Chiefs")) || "", /fair<\/span> vs 50\.0%/);
  assert.ok(items.some((x) => x.includes("O&quot;Brien") && !x.includes("ca-fair")), "prop rail row: no marker");
});

test("R24: prop rows on the +EV table carry the player's star (shell-owned); game-line rows carry none", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const rows = sect(html, "ev-table", "ev-rail").split("<tr data-href").slice(1);
  assert.match(rows.find((r) => r.includes("game=5")), /<button class="ca-star" data-star-kind="players" data-star-id="O&quot;Brien &lt;i&gt;Zed&lt;\/i&gt;"/);
  assert.ok(rows.filter((r) => !r.includes("game=5")).every((r) => !r.includes("data-star-kind")), "line rows: no star");
  assert.ok(!/starToggle\(|starDelegate|data-star-kind/.test(require_src("js/pages/ev.js")), "no page-level star logic (R16)");
});

// ---- fidelity pass (E1-E6) -----------------------------------------------------------------------
const idsIn = (html, ids) => ids.map((id) => html.indexOf(`id="${id}"`));
const ascending = (a) => a.every((v, i) => v >= 0 && (i === 0 || v > a[i - 1]));

test("E3: the +EV table has exactly the mockup's 13 columns, in order", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const head = html.slice(html.indexOf('<table class="ca-table ca-dash-table ca-ev-table"'), html.indexOf("</thead>"));
  const cols = [...head.matchAll(/<th[^>]*>([^<]*)<\/th>/g)].map((m) => m[1]);
  assert.deepEqual(cols, ["#", "Sport", "Matchup / Player", "Market", "Best Odds", "Model Prob.", "Impl. Prob.", "Edge", "EV", "Alpha Score", "Confidence", "Game Time", "View"]);
  const row = html.slice(html.indexOf("<tr data-href")).split("</tr>")[0];
  assert.equal((row.match(/<td/g) || []).length, 13, "one cell per column");
});

test("E2: rail order is Top Alpha, Market Pulse, then EV Distribution + Performance by Edge Bucket as a side-by-side pair", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  assert.ok(ascending(idsIn(html, ["ev-rail", "ev-top", "ev-pulse", "ev-dist", "ev-buckets"])));
  const rail = html.slice(html.indexOf('id="ev-rail"'));
  const pair = rail.slice(rail.indexOf('<div class="ca-ev-pair">'));
  assert.ok(pair.indexOf('id="ev-dist"') > 0 && pair.indexOf('id="ev-dist"') < pair.indexOf('id="ev-buckets"'), "the pair wrapper holds Distribution then Buckets");
  assert.ok(rail.indexOf('<div class="ca-ev-pair">') > rail.indexOf('id="ev-pulse"'), "the pair sits below Market Pulse");
  assert.ok(!rail.slice(0, rail.indexOf('<div class="ca-ev-pair">')).includes('id="ev-dist"'), "neither pair card is a full-width rail row");
  const css = require_src("css/theme.css");
  assert.match(css, /\.ca-ev-pair\{display:grid;grid-template-columns:minmax\(0,[.\d]+fr\) minmax\(0,[.\d]+fr\)/, "two columns");
  assert.match(css, /\.ca-ev-main\{display:grid;grid-template-columns:minmax\(0,7fr\) minmax\(0,3fr\)/, "left ~70% / rail ~30%");
});

test("E4: filter card and a separate search card share one row; tabs + Sort By sit above the table inside the left column", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const row = html.slice(html.indexOf('class="ca-ev-filterrow"'), html.indexOf('class="ca-ev-main"'));
  assert.ok(ascending(idsIn(row, ["ev-filters", "ev-search"])), "filters then search, same row wrapper");
  const filters = row.slice(row.indexOf('id="ev-filters"'), row.indexOf('id="ev-search"')), search = row.slice(row.indexOf('id="ev-search"'));
  for (const l of ["Sport", "League", "Market Type", "Sportsbook", "Minimum Edge", "Confidence", "Date"]) assert.ok(filters.includes(`<span>${l}</span>`), l);
  assert.ok(!filters.includes("data-ev-q"), "no search box inside the filter card");
  assert.ok(search.includes("data-ev-q") && search.includes("Search teams, players, or games..."), "the search box is its own card");
  assert.match(require_src("css/theme.css"), /\.ca-ev-filterrow\{display:grid;grid-template-columns:minmax\(0,74fr\) minmax\(0,26fr\)/);
  const main = html.slice(html.indexOf('class="ca-ev-main"'));
  const left = main.slice(0, main.indexOf('id="ev-rail"'));
  assert.ok(ascending([left.indexOf('class="ca-ev-tabsrow"'), left.indexOf('id="ev-tabs"'), left.indexOf("Sort By"), left.indexOf('id="ev-table"')]), "tabs row (pills + Sort By) precedes the table card");
  assert.ok(left.slice(left.indexOf('class="ca-ev-tabsrow"'), left.indexOf('id="ev-table"')).includes('data-ev="sort"'));
  assert.ok(main.indexOf('id="ev-rail"') > left.indexOf('id="ev-table"'), "the rail is the right-hand sibling");
});

test("E5: Last Updated shows the newest pick build time (ET) when known, is hidden (refresh stays) when not", async () => {
  const shown = await populated().D.buildEvPage();
  assert.match(shown, /<div class="ca-ev-updated"><span>Last Updated<\/span><b>[A-Z][a-z]{2} \d{1,2}, \d{4} \d{1,2}:\d{2} [AP]M[^<]*<\/b><\/div>/);
  assert.ok(shown.includes("data-ev-refresh"));
  const hidden = await populated({ lastBuild: false }).D.buildEvPage();
  assert.ok(!hidden.includes("Last Updated") && !hidden.includes("ca-ev-updated"), "no timestamp -> no block, no dash placeholder");
  assert.ok(hidden.includes("data-ev-refresh"), "the refresh button stays");
});

test("E1/E6: stat sub-lines are one short line (long text in tooltips); Highest Edge is a single ellipsised line; rail rows keep rank / logos / edge box", async () => {
  const { D } = populated();
  const html = await D.buildEvPage();
  const stats = html.slice(html.indexOf("ca-ev-stats"), html.indexOf('class="ca-ev-filterrow"'));
  assert.equal((stats.match(/class="ca-card ca-stat"/g) || []).length, 5, "five stat cards");
  assert.match(stats, /<span class="muted">7-4 · 12 graded<\/span>/, "Model Hit Rate sub = record + short count");
  assert.match(stats, /<span class="ca-info" title="7-4 · 12 graded \+EV picks in the last 30 days/, "population in the (i)");
  assert.match(stats, /\+41\.7%<small class="pos">\+5\.0u<\/small>/, "ROI with the units inline");
  assert.ok(!stats.includes("graded +EV picks</span>"), "no long population sub-line");
  const hi = stats.slice(stats.indexOf("Highest Edge"), stats.indexOf("Model Hit Rate"));
  assert.equal((hi.match(/class="ca-ell"/g) || []).length, 1, "one ellipsised line");
  assert.match(hi, /<span class="ca-ell">&quot;Chiefs&quot; ML vs\. &lt;b&gt;Crew&lt;\/b&gt;<\/span>/, "pick vs. opponent, escaped, one line");
  const top = sect(html, "ev-top", "ev-pulse");
  assert.ok(ascending([top.indexOf('class="ca-ev-rank"'), top.indexOf('class="ca-ev-logos"'), top.indexOf('class="ca-ev-ta-main"'), top.indexOf('class="ca-ev-ta-odds"'), top.indexOf('class="ca-ev-edge"')]));
  assert.ok(top.includes("View All →") && sect(html, "ev-pulse", "ev-dist").includes("View All →"), "View All on both rail title rows");
});

test("E3 type scale: the +EV page uses the shared fluid tokens, not the old 24px / 12px overrides", () => {
  const css = require_src("css/theme.css"), ev = css.slice(css.indexOf("/* +EV page"), css.indexOf("/* Game page"));
  assert.ok(!/\.ca-ev \.ca-card h2\{font-size:24px\}/.test(css) && !/\.ca-ev-table\{font-size:12(\.5)?px\}/.test(ev));
  assert.match(css, /\.ca-dash \.ca-card h2,\.ca-ev \.ca-card h2\{font-size:var\(--fs-card\)/);
  assert.match(css, /\.ca-ev \.ca-dash-table(,[^{]*)?\{font-size:var\(--fs-table\)/);
  assert.match(css, /\.ca-dash-stats \.ca-stat-value,\.ca-ev-stats \.ca-stat-value\{font-size:var\(--fs-stat\)/);
  assert.match(css, /#ev-table[^{]*\{[^}]*container-type|\.ca-ev-tablecard\{[^}]*container-type:inline-size/, "table card is a size container for the column-fitting queries");
});
