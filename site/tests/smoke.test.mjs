import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";

test("app.js loads without running the page (boot moved to js/boot.js)", () => {
  const g = loadScripts(["app.js", "js/boot.js"], { page: "dashboard" });
  assert.equal(typeof g.render, "function");
  assert.equal(typeof g.boot, "function");
  assert.equal(g.pStr(1.25), "+1.3%");
  assert.equal(g.fmtOdds, undefined, "unused formatter removed");
  assert.equal(typeof g.sbAll, "function");
});

import fs from "node:fs";
import { fileURLToPath } from "node:url";
test("every page loads the scripts in order with the current cache key", () => {
  const order = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/ev.js", "js/pages/game.js", "js/pages/track.js", "js/pages/board.js", "js/boot.js"];
  for (const f of fs.readdirSync(new URL("..", import.meta.url)).filter((f) => f.endsWith(".html"))) {
    const html = fs.readFileSync(new URL(`../${f}`, import.meta.url), "utf8");
    const srcs = [...html.matchAll(/<script src="([^"?]+)\?v=([^"]+)"/g)];
    const sport = ["nfl.html", "cfb.html", "mlb.html", "nba.html"].includes(f);   // the sport pages also load the left and right columns (js/pages/board-left.js, board-right.js) after board.js
    assert.deepEqual(srcs.map((m) => m[1]), sport ? order.flatMap((s) => (s === "js/pages/board.js" ? [s, "js/pages/board-left.js", "js/pages/board-right.js"] : [s])) : order, f);
    assert.ok(srcs.every((m) => m[2] === srcs[0][2]), `${f}: one cache key`);
    assert.equal(srcs[0][2], "20261007a", `${f}: the current cache key`);
    assert.match(html, /css\/theme\.css\?v=20261007a"/, `${f}: theme.css carries the same key`);
    assert.match(html, /css\/theme\.css/); assert.doesNotMatch(html, /styles\.css/);
  }
});

test("the cache key was bumped to 20261007a everywhere: no file in site/ still carries the previous key", () => {
  const old = ["2026", "1006", "a"].join(""), seen = [];
  const walk = (dir) => { for (const e of fs.readdirSync(dir, { withFileTypes: true })) { if (e.name === "node_modules") continue; const p = `${dir}/${e.name}`; if (e.isDirectory()) walk(p); else if (/\.(html|js|mjs|css|json)$/.test(e.name) && fs.readFileSync(p, "utf8").includes(old)) seen.push(p); } };
  walk(fileURLToPath(new URL("..", import.meta.url)).replace(/\/$/, ""));
  assert.deepEqual(seen, [], "stale cache key");
  const html = fs.readFileSync(new URL("../nfl.html", import.meta.url), "utf8");
  assert.equal([...html.matchAll(/\?v=(\w+)/g)].filter((m) => m[1] !== "20261007a").length, 0);
});
