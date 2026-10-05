import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/shell.js", "js/data.js", "js/boot.js"]);

test("line opportunity: edge vs best-price implied, EV %, alpha + tier", () => {
  const o = g.toOpportunity({ sport: "nfl", game_pk: 1, matchup: "DET @ KC", market: "moneyline", side: "away",
    commence_time: "2026-10-05T20:25:00Z", true_prob: 0.604, best_line_implied: 0.439, best_price: 128,
    best_book: "draftkings", ev_best: 0.165 }, "line", { roiPct: 0, n: 0 });
  assert.equal(o.kind, "line"); assert.equal(o.odds, 128); assert.equal(o.book, "draftkings");
  assert.ok(Math.abs(o.edgePp - 16.5) < 1e-9); assert.ok(Math.abs(o.evPct - 16.5) < 1e-9);
  assert.equal(o.alpha, 95); assert.equal(o.tier, "HIGH");
});

test("prop opportunity uses model_prob and the best price", () => {
  const o = g.toOpportunity({ sport: "nfl", game_pk: 2, matchup: "BUF @ ATL", market: "rec_yds", side: "over", line: 64.5,
    player_name: "A. Receiver", commence_time: "2026-10-05T17:00:00Z", model_prob: 0.58, best_price: -110,
    best_book: "fanduel", ev_best: 0.107 }, "prop", { roiPct: 10, n: 100 });
  assert.equal(o.playerName, "A. Receiver"); assert.equal(o.line, 64.5);
  assert.ok(Math.abs(o.edgePp - (0.58 - 110 / 210) * 100) < 1e-9);
  assert.equal(o.alpha, g.alphaScore({ evPct: 10.7, edgePp: o.edgePp, segRoiPct: 10, segN: 100 }));
});

test("missing inputs -> null; segment records from ev_pnl_daily", () => {
  assert.equal(g.toOpportunity({ sport: "cfb", true_prob: null, best_price: 100, ev_best: 0.1 }, "line", { roiPct: 0, n: 0 }), null);
  const m = g.segmentRecords([{ sport: "cfb", market: "moneyline", n: 10, wins: 6, losses: 4, pushes: 0, pnl: 20 },
                              { sport: "nfl", market: "prop", n: 5, wins: 3, losses: 2, pushes: 0, pnl: -5 }]);
  assert.deepEqual(m.get("cfb|moneyline"), { roiPct: 20, n: 10 });
  assert.deepEqual(m.get("nfl|prop"), { roiPct: -10, n: 5 });
  assert.equal(g.segmentKey("nfl", "prop", "rec_yds"), "nfl|prop");
});

test("toOpportunity accepts string numerics; missing best_price or ev_best -> null", () => {
  const o = g.toOpportunity({ sport: "nfl", game_pk: 1, market: "moneyline", side: "away", true_prob: "0.604",
    best_line_implied: "0.439", best_price: "128", ev_best: "0.165" }, "line", { roiPct: 0, n: 0 });
  assert.equal(o.odds, 128); assert.ok(Math.abs(o.edgePp - 16.5) < 1e-9); assert.ok(Math.abs(o.evPct - 16.5) < 1e-9);
  assert.equal(g.toOpportunity({ sport: "nfl", true_prob: 0.6, best_price: null, ev_best: 0.1 }, "line", null), null);
  assert.equal(g.toOpportunity({ sport: "nfl", true_prob: 0.6, best_price: 100, ev_best: null }, "line", null), null);
});

// ---- loader tests: stub fetch by URL path -------------------------------------------------
const loaderWith = (rowsFor, { failPaths = [], throwAll = false } = {}) => loadScripts(
  ["app.js", "js/metrics.js", "js/shell.js", "js/data.js", "js/boot.js"],
  { globals: { fetch: async (url) => {
    if (throwAll) throw new Error("network down");
    const path = new URL(url).pathname.split("/").pop();
    if (failPaths.includes(path)) return { ok: false, status: 500, text: async () => "boom", json: async () => ({}) };
    return { ok: true, json: async () => rowsFor(path, url) };
  } } });

const lineRow = (o) => ({ sport: "nfl", game_pk: 1, matchup: "A @ B", market: "moneyline", side: "home", true_prob: 0.6,
  best_price: 100, best_book: "dk", ev_best: 0.1, is_pick: true, commence_time: "2026-10-05T17:00:00Z", ...o });

test("loadOpportunities: picks only, sorted by alpha then EV, tolerates a failing source", () => {
  const rows = (path) => path === "ev_current" ? [
    lineRow({ game_pk: 1, ev_best: 0.05, true_prob: 0.55 }),
    lineRow({ game_pk: 2, ev_best: 0.30, true_prob: 0.70, is_pick: false }),
    lineRow({ game_pk: 3, ev_best: 0.12, true_prob: 0.60 }),
    lineRow({ game_pk: 4, ev_best: 0.12, true_prob: 0.60, side: "away" }),
    lineRow({ game_pk: 5, ev_best: 0.12, true_prob: 0.60, is_pick: undefined }),
  ] : path === "ev_prop_picks_current" ? [{ sport: "nfl", game_pk: 9, market: "rec_yds", side: "over", line: 50.5,
    player_name: "P", model_prob: 0.7, best_price: -110, ev_best: 0.3, is_pick: true }] : [];
  return Promise.all([
    loaderWith(rows).loadOpportunities(),
    loaderWith(rows, { failPaths: ["ev_prop_picks_current"] }).loadOpportunities(),
  ]).then(([all, noProps]) => {
    assert.ok(!all.some((o) => o.game_pk === 2 || o.game_pk === 5), "non-pick line rows dropped");
    assert.ok(all.some((o) => o.kind === "prop"));
    for (let i = 1; i < all.length; i++) {
      assert.ok(all[i - 1].alpha > all[i].alpha || (all[i - 1].alpha === all[i].alpha && all[i - 1].evPct >= all[i].evPct), "alpha desc, then evPct desc");
    }
    assert.ok(noProps.length > 0 && noProps.every((o) => o.kind === "line"), "line opps survive a failing props source");
  });
});

test("loadLineMoves: drops null-score and zero-move rows, sorts by score desc", async () => {
  const L = loaderWith(() => [
    { game_pk: 1, market: "spread", side: "home", open_line: -3, cur_line: -4.5 },
    { game_pk: 2, market: "spread", side: "home", open_line: -3, cur_line: -3 },
    { game_pk: 3, market: "total", side: "over", open_line: null, cur_line: 45 },
    { game_pk: 4, market: "total", side: "over", open_line: 44, cur_line: 45 },
    { game_pk: 5, market: "moneyline", side: "home", open_price: "", cur_price: -120 },
  ]);
  const out = await L.loadLineMoves();
  assert.deepEqual(out.map((r) => r.game_pk), [1, 4]);
  assert.ok(out[0].score > out[1].score);
});

test("loadEvHistory: window-edge rebuilds not counted, first date kept, ET dates, wide capped query", async () => {
  const NOW = Date.parse("2026-10-02T15:00:00Z");   // 11:00 ET on 2026-10-02
  class FakeDate extends Date {
    constructor(...a) { if (a.length) super(...a); else super(NOW); }
    static now() { return NOW; }
  }
  const urls = [];
  const mk = (game_pk, created_at) => ({ sport: "nfl", game_pk, market: "moneyline", side: "home", created_at });
  const L = loadScripts(["app.js", "js/metrics.js", "js/shell.js", "js/data.js", "js/boot.js"], { globals: { Date: FakeDate,
    fetch: async (url) => { urls.push(url); const path = new URL(url).pathname.split("/").pop();
      return { ok: true, json: async () => path === "ev_picks" ? [
        mk(1, "2026-09-20T16:00:00Z"), mk(1, "2026-09-28T16:00:00Z"),   // first flagged before the 8-day window -> excluded
        mk(2, "2026-09-26T16:00:00Z"), mk(2, "2026-09-30T16:00:00Z"),   // dedupe keeps 09-26
        mk(3, "2026-10-02T02:00:00Z"),                                  // 22:00 ET on 10-01
      ] : [] }; } } });
  const out = await L.loadEvHistory(8);
  assert.deepEqual(out.map((r) => r.date).sort(), ["2026-09-26", "2026-10-01"]);
  assert.ok(out.every((r) => r.kind === "line" && r.sport === "nfl"));
  assert.ok(urls.every((u) => u.includes("order=created_at.asc") && u.includes("limit=10000")));
  const since = decodeURIComponent(urls[0].match(/created_at=gte\.([^&]+)/)[1]);
  assert.equal(since, new Date(NOW - 22 * 864e5).toISOString(), "lookback is days + 14");
});

test("loaders return empty data when fetch throws", async () => {
  const L = loaderWith(() => [], { throwAll: true });
  assert.deepEqual(await L.loadOpportunities(), []);
  assert.deepEqual(await L.loadLineMoves(), []);
  assert.equal((await L.loadSplits()).size, 0);
  assert.deepEqual(await L.loadEvHistory(8), []);
});

test("latestSplitMap keeps the newest capture per game|market|side; oppLabel puts the player's name before a prop pick", () => {
  const m = g.latestSplitMap([{ game_pk: 1, market: "spread", side: "home", captured_at: "2026-10-01T00:00:00Z", ticket_pct: 40 },
    { game_pk: 1, market: "spread", side: "home", captured_at: "2026-10-02T00:00:00Z", ticket_pct: 55 }, { game_pk: 1, market: "total", side: "over", captured_at: "2026-10-01T00:00:00Z", ticket_pct: 60 }]);
  assert.equal(m.size, 2); assert.equal(m.get("1|spread|home").ticket_pct, 55);
  assert.equal(g.latestSplitMap(null).size, 0);
  assert.equal(g.oppLabel({ kind: "prop", market: "rec_yds", marketLabel: "Rec Yds", side: "under", line: 64.5, playerName: "Josh Allen", sport: "nfl" }), "Josh Allen Under 64.5 Rec Yds");
  assert.equal(g.oppLabel({ kind: "line", sport: "nfl", game_pk: 1, matchup: "Detroit Lions @ Kansas City Chiefs", market: "moneyline", side: "away" }, new Map()), "Lions ML");
});
