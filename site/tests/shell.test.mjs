import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const load = (storage) => loadScripts(["app.js", "js/shell.js", "js/boot.js"], { page: "ev", storage });

test("header has the mockup nav in order with the active item marked", () => {
  const g = load();
  const html = g.siteHeader("ev");
  const labels = [...html.matchAll(/data-nav="([a-z]+)"/g)].map((m) => m[1]);
  assert.deepEqual(labels, ["dashboard", "mlb", "nfl", "nba", "cfb", "ev", "track"]);
  assert.match(html, /data-nav="ev"[^>]*aria-current="page"/);
  assert.match(html, /href="settings.html"/);
  assert.match(html, /href="rankings.html"/);
});

test("watchlist toggles and persists", () => {
  const storage = new Map();
  const g = load(storage);
  assert.equal(g.watchlistToggle("games", "401"), true);
  assert.equal(g.watchlistHas("games", "401"), true);
  assert.deepEqual(JSON.parse(storage.get("ca-watchlist")).games, ["401"]);
  assert.equal(g.watchlistToggle("games", "401"), false);
  assert.equal(g.watchlistHas("games", "401"), false);
});

test("watchlist survives a throwing storage", () => {
  const g = load();
  g.localStorage.setItem = () => { throw new Error("blocked"); };
  assert.equal(g.watchlistToggle("teams", "KC"), true);
  assert.equal(g.watchlistHas("teams", "KC"), true);
});
