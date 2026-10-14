import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";

const BASE = ["app.js", "js/metrics.js"];
const live = loadScripts([...BASE, "js/shrink-k.js", "js/shrink.js", "js/boot.js"]);                              // the committed table
const mixed = loadScripts([...BASE, "tests/fixtures/shrink-k-mixed.js", "js/shrink.js", "js/boot.js"]);          // NFL 0, CFB 0.5
const bad = loadScripts([...BASE, "tests/fixtures/shrink-k-bad.js", "js/shrink.js", "js/boot.js"]);
const none = loadScripts([...BASE, "js/shrink.js", "js/boot.js"]);                                                // no table loaded at all
const near = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);

test("shrinkProb: market + k (model - market), clipped to [0.01, 0.99]", () => {
  near(mixed.shrinkProb(0.70, 0.50, 0), 0.50); near(mixed.shrinkProb(0.70, 0.50, 1), 0.70);
  near(mixed.shrinkProb(0.70, 0.50, 0.25), 0.55); near(mixed.shrinkProb(0.30, 0.50, 0.25), 0.45);
  assert.equal(mixed.shrinkProb(1, 0.5, 1), 0.99); assert.equal(mixed.shrinkProb(0, 0.5, 1), 0.01);
  assert.equal(mixed.shrinkProb(0.5, 0.995, 0), 0.99, "a market at 99.5% is clipped too"); assert.equal(mixed.shrinkProb(0.5, 0.004, 0), 0.01);
});

test("shrinkProb / shrinkLine: anything that is not a finite number gives null, never NaN", () => {
  for (const bad of [NaN, null, undefined, "0.5", Infinity]) {
    assert.equal(mixed.shrinkProb(bad, 0.5, 0.5), null); assert.equal(mixed.shrinkProb(0.5, bad, 0.5), null); assert.equal(mixed.shrinkProb(0.5, 0.5, bad), null);
    assert.equal(mixed.shrinkLine(bad, 7, 0.5), null); assert.equal(mixed.shrinkLine(7, bad, 0.5), null); assert.equal(mixed.shrinkLine(7, 7, bad), null);
  }
});

test("shrinkLine: market + k (model - market), in either sign convention", () => {
  near(mixed.shrinkLine(11, 7, 0.5), 9); near(mixed.shrinkLine(11, 7, 0), 7); near(mixed.shrinkLine(11, 7, 1), 11);
  near(mixed.shrinkLine(-11, -7, 0.5), -9, 1e-9);                    // the same gap in home-line space
  near(mixed.shrinkLine(55, 50, 0.25), 51.25);
});

test("fairAmerican: the American price of a probability; null outside (0, 1)", () => {
  assert.equal(mixed.fairAmerican(0.75), -300); assert.equal(mixed.fairAmerican(0.4), 150); assert.equal(mixed.fairAmerican(0.5), -100);
  for (const bad of [0, 1, -0.2, 1.2, NaN, null, undefined]) assert.equal(mixed.fairAmerican(bad), null, String(bad));
});

test("shrinkHalf: the half point, except that a model equal to the market IS the market (no rounding artefact)", () => {
  assert.equal(mixed.shrinkHalf(53.2, 50), 53); assert.equal(mixed.shrinkHalf(-8.8, -4.5), -9); assert.equal(mixed.shrinkHalf(52.9, 52.6), 53);
  assert.equal(mixed.shrinkHalf(51.75, 51.75), 51.75, "a quarter-point market at k = 0"); assert.equal(mixed.shrinkHalf(51.75 + 1e-12, 51.75), 51.75);
  assert.equal(mixed.shrinkHalf(NaN, 50), null); assert.equal(mixed.shrinkHalf(7, NaN), 7, "no market to equal: just the half point");
});

test("shrinkK: the committed table; an unknown sport or market is k = 0, never the raw model", () => {
  for (const sport of ["nfl", "mlb"]) for (const m of ["moneyline", "spread", "total"]) assert.equal(live.shrinkK(sport, m), 0, `${sport} ${m} is explicitly 0`);
  for (const [s, m] of [["nba", "spread"], ["cfb", "puckline"], ["", ""], [undefined, undefined], ["constructor", "spread"], ["__proto__", "total"]]) {
    assert.equal(live.shrinkK(s, m), 0, `${s}/${m}`);
  }
  assert.equal(mixed.shrinkK("cfb", "spread"), 0.5); assert.equal(mixed.shrinkK("nfl", "moneyline"), 0); assert.equal(mixed.shrinkK("mlb", "total"), 0, "no entry");
});

test("shrinkK: a malformed or missing table is k = 0", () => {
  for (const [s, m] of [["cfb", "moneyline"], ["cfb", "spread"], ["cfb", "total"], ["mlb", "moneyline"], ["mlb", "spread"], ["mlb", "total"], ["nfl", "spread"]]) {
    assert.equal(bad.shrinkK(s, m), 0, `${s}/${m}`);
  }
  assert.equal(none.shrinkK("cfb", "spread"), 0, "js/shrink-k.js not loaded: the lookup still answers 0");
});

test("js/shrink-k.js agrees with assets/game_shrink_k.json for every sport and market; NFL and MLB are explicit zeros", () => {
  const doc = JSON.parse(fs.readFileSync(new URL("../../assets/game_shrink_k.json", import.meta.url), "utf8"));
  for (const sport of ["nfl", "cfb", "mlb"]) for (const m of ["moneyline", "spread", "total"]) {
    const k = doc.markets[sport][m].k;
    assert.ok(k >= 0 && k <= 1, `${sport} ${m}`);
    assert.equal(live.shrinkK(sport, m), k, `${sport} ${m}`);
  }
  for (const sport of ["nfl", "mlb"]) for (const m of ["moneyline", "spread", "total"]) assert.equal(doc.markets[sport][m].k, 0);
  assert.ok(Object.isFrozen(live.SHRINK_K), "the table is frozen");
});
