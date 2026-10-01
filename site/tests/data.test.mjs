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
