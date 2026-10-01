import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/boot.js"]);

test("alpha score formula and tiers", () => {
  assert.equal(g.alphaScore({ evPct: 8, edgePp: 8, segRoiPct: 0, segN: 0 }), 95);
  assert.equal(g.alphaScore({ evPct: 4, edgePp: 4, segRoiPct: 0, segN: 0 }), 70);
  assert.equal(g.alphaScore({ evPct: 0, edgePp: 0, segRoiPct: -50, segN: 1e9 }), 40);
  assert.equal(g.alphaScore({ evPct: 20, edgePp: 20, segRoiPct: 40, segN: 1e9 }), 100);
  // shrinkage: 10% ROI on 100 graded -> roi_s 5 -> s_track .75 -> +2.5 over neutral
  assert.equal(g.alphaScore({ evPct: 4, edgePp: 4, segRoiPct: 10, segN: 100 }), 73);
  assert.equal(g.alphaTier(85), "HIGH"); assert.equal(g.alphaTier(84), "STRONG");
  assert.equal(g.alphaTier(75), "STRONG"); assert.equal(g.alphaTier(74), "MEDIUM");
  assert.equal(g.alphaTier(65), "MEDIUM"); assert.equal(g.alphaTier(64), null);
});

test("pnl aggregation in units", () => {
  const a = g.aggPnl([{ n: "3", wins: 2, losses: 1, pushes: 0, pnl: "8.18" }, { n: 2, wins: 0, losses: 2, pushes: 0, pnl: -20 }]);
  assert.deepEqual({ n: a.n, w: a.w, l: a.l, p: a.p }, { n: 5, w: 2, l: 3, p: 0 });
  assert.ok(Math.abs(a.units - (-1.182)) < 1e-9);
  assert.ok(Math.abs(a.roiPct - (-23.64)) < 1e-9);
  assert.equal(g.aggPnl([]).roiPct, 0);
});

test("hit rate, vs market, streak, recent form", () => {
  const rs = [{ won: true, implied: 0.5 }, { won: false, implied: 0.6 }, { won: null, implied: 0.5 }, { won: true, implied: 0.4 }];
  assert.ok(Math.abs(g.hitRate(rs) - 2 / 3) < 1e-12);
  assert.ok(Math.abs(g.hitRateVsMarket(rs) - (2 / 3 - 0.5) * 100) < 1e-9);
  assert.deepEqual(g.currentStreak([{ won: true }, { won: null }, { won: true }, { won: false }]), { kind: "W", n: 2 });
  assert.equal(g.currentStreak([]), null);
  assert.deepEqual(g.recentForm([{ won: true }, { won: false }, { won: null }, { won: true }], 3), { w: 1, l: 1, p: 1, pct: 0.5 });
});

test("calibration flips underdog probabilities; brier", () => {
  const rows = [{ prob: 0.62, won: true }, { prob: 0.38, won: true }, { prob: 0.72, won: false }];
  const c = g.calibration(rows, [[0.5, 0.55], [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 1.01]]);
  const b60 = c.find((x) => x.band[0] === 0.6);
  assert.equal(b60.n, 2);                       // .62 won, .38->.62 lost
  assert.ok(Math.abs(b60.predicted - 0.62) < 1e-12);
  assert.equal(b60.actual, 0.5);
  assert.ok(Math.abs(g.brier(rows) - ((0.38 ** 2 + 0.62 ** 2 + 0.72 ** 2) / 3)) < 1e-12);
});

test("edge buckets, histogram, daily counts, odds", () => {
  const eb = g.edgeBuckets([{ edgePct: 12, won: true, profitUnits: 0.9 }, { edgePct: 3, won: false, profitUnits: -1 }, { edgePct: -1, won: true, profitUnits: 1 }], [10, 5, 2, 0]);
  assert.deepEqual(eb.map((b) => [b.label, b.n]), [["> 10%", 1], ["5% to 10%", 0], ["2% to 5%", 1], ["0% to 2%", 0], ["< 0%", 1]]);
  assert.equal(eb[0].roiPct, 90);
  assert.deepEqual(g.histogram([-3, 1, 2, 7], [-5, 0, 5, 10]).map((b) => b.n), [1, 2, 1]);
  const dc = g.dailyCounts([{ d: "2026-09-30" }, { d: "2026-10-01" }, { d: "2026-10-01" }], (r) => r.d, 3, "2026-10-01");
  assert.deepEqual(dc, [{ date: "2026-09-29", n: 0 }, { date: "2026-09-30", n: 1 }, { date: "2026-10-01", n: 2 }]);
  assert.ok(Math.abs(g.americanToProb(-110) - 110 / 210) < 1e-12);
  assert.equal(g.probToAmerican(0.6), -150);
  assert.equal(g.probToAmerican(0.4), 150);
  // Null-safety: null/undefined inputs must not fabricate numbers
  assert.ok(Number.isNaN(g.americanToProb(null)));
  assert.ok(Number.isNaN(g.americanToProb(undefined)));
  assert.equal(g.lineMoveScore({market:"moneyline", open_price:-110, cur_price:null}), null);
  assert.equal(g.lineMoveScore({market:"spread", open_line:-2.5, cur_line:-4}), 1.5);
  assert.deepEqual(g.histogram([null, 1, undefined, NaN], [0, 5]).map((b) => b.n), [1]);
});
