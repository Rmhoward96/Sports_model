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

// ---- stars: one delegated handler ----------------------------------------------------------
const fakeBtn = (kind, id, label = "X") => {
  const cls = new Set(); const attrs = {};
  return { dataset: { starKind: kind, starId: id, starLabel: label }, textContent: "☆", cls, attrs,
    classList: { toggle: (c, on) => (on ? cls.add(c) : cls.delete(c)) }, setAttribute: (k, v) => { attrs[k] = v; } };
};

test("starButton markup + aria for off and on", () => {
  const g = load();
  const off = g.starButton("games", "7", 'A "B" <C>');
  assert.match(off, /class="ca-star"/);
  assert.match(off, /aria-pressed="false"/);
  assert.match(off, /aria-label="Add A &quot;B&quot; &lt;C&gt; to watchlist"/);
  assert.ok(off.includes(">☆</button>"));
  assert.ok(!off.includes("<C>"), "label is escaped");
  g.watchlistToggle("games", "7");
  const on = g.starButton("games", "7", "Lions @ Chiefs");
  assert.match(on, /class="ca-star on"/);
  assert.match(on, /aria-pressed="true"/);
  assert.match(on, /aria-label="Remove Lions @ Chiefs from watchlist"/);
  assert.ok(on.includes(">★</button>"));
});

test("starToggle syncs EVERY button with the same kind+id (class, glyph, aria) and no others", () => {
  const g = load();
  const a = fakeBtn("games", "7", "Lions @ Chiefs"), b = fakeBtn("games", "7", "Lions @ Chiefs"), other = fakeBtn("games", "8"), team = fakeBtn("teams", "7");
  const events = [];
  const doc = { querySelectorAll: () => [a, b, other, team], dispatchEvent: (e) => events.push(e) };
  assert.equal(g.starToggle("games", "7", doc), true);
  for (const x of [a, b]) {
    assert.ok(x.cls.has("on")); assert.equal(x.textContent, "★");
    assert.equal(x.attrs["aria-pressed"], "true"); assert.equal(x.attrs["aria-label"], "Remove Lions @ Chiefs from watchlist");
  }
  assert.equal(other.textContent, "☆"); assert.equal(team.textContent, "☆");
  assert.equal(g.watchlistHas("games", "7"), true);
  assert.equal(g.starToggle("games", "7", doc), false);
  assert.equal(a.textContent, "☆"); assert.equal(b.attrs["aria-pressed"], "false"); assert.equal(a.attrs["aria-label"], "Add Lions @ Chiefs to watchlist");
});

test("starDelegate toggles for a star target, ignores other clicks", () => {
  const g = load();
  const btn = fakeBtn("players", "P1", "P1");
  const doc = { querySelectorAll: () => [btn], dispatchEvent() {} };
  let prevented = 0, stopped = 0;
  const ev = (t) => ({ target: t, preventDefault: () => prevented++, stopPropagation: () => stopped++ });
  assert.equal(g.starDelegate(ev({ closest: () => btn }), doc), true);
  assert.equal(prevented, 1); assert.equal(stopped, 1);
  assert.equal(g.watchlistHas("players", "P1"), true); assert.equal(btn.textContent, "★");
  assert.equal(g.starDelegate(ev({ closest: () => null }), doc), false);
  assert.equal(g.starDelegate(ev(null), doc), false);
  assert.equal(prevented, 1);
  assert.equal(typeof g.wireStars, "function"); g.wireStars(); // compatibility no-op
});

test("rowDelegate: a row click navigates; clicks on links, buttons and stars (inside the row) do not", () => {
  const g = load();
  const row = { dataset: { href: "game.html?sport=nfl&game=1" } };
  const click = (inner) => ({ target: { closest: (sel) => (sel === "tr[data-href]" ? row : sel === "a,button" ? inner : null) } });
  const seen = [];
  assert.equal(g.rowDelegate(click(null), (u) => seen.push(u)), true);
  assert.deepEqual(seen, ["game.html?sport=nfl&game=1"]);
  assert.equal(g.rowDelegate(click({}), (u) => seen.push(u)), false);
  assert.equal(seen.length, 1);
  assert.equal(g.rowDelegate({ target: { closest: () => null } }, (u) => seen.push(u)), false);
});
