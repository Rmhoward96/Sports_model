import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";

test("app.js loads without running the page (boot moved to js/boot.js)", () => {
  const g = loadScripts(["app.js", "js/boot.js"], { page: "dashboard" });
  assert.equal(typeof g.render, "function");
  assert.equal(typeof g.boot, "function");
  assert.equal(g.fmtOdds(120), "+120");
});
