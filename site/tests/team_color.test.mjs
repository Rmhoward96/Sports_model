import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";

const g = loadScripts(["app.js"]);
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const lin = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
const wcagLum = (hex) => { const [r, g, b] = rgb(hex).map(lin); return 0.2126 * r + 0.7152 * g + 0.0722 * b; };
const contrastOnWhite = (hex) => 1.05 / (wcagLum(hex) + 0.05);

test("readableTeamColor keeps light grays gray (no invented hue)", () => {
  const out = g.readableTeamColor("#A5ACAF");   // Raiders alternate (silver)
  if (out === null) return;
  const [r, gg, b] = rgb(out);
  assert.ok(Math.abs(r - gg) <= 24 && Math.abs(gg - b) <= 24, `${out} is no longer gray`);
  assert.ok(contrastOnWhite(out) >= 2.5, `${out} too light on white`);
});

test("readableTeamColor darkens a saturated light color, preserving hue", () => {
  const out = g.readableTeamColor("#FFB612");   // Steelers gold
  assert.ok(out && out !== "#FFB612");
  assert.ok(contrastOnWhite(out) >= 2.5, `${out} too light on white`);
  const [h0] = g.hexToHsl("#FFB612"), [h1] = g.hexToHsl(out);
  assert.ok(Math.abs(h0 - h1) < 15, `hue drifted ${h0} -> ${h1}`);
});

test("readableTeamColor leaves readable dark colors and rejects near-white", () => {
  assert.equal(g.readableTeamColor("#002A5C"), "#002A5C");
  assert.equal(g.readableTeamColor("#FFFFFF"), null);
});

const dist = (a, b) => Math.hypot(...rgb(a).map((v, i) => v - rgb(b)[i]));
test("gameTeamColors: Georgia vs Alabama (both crimson, no usable alternates) are guarded: home becomes the site navy", () => {
  assert.ok(dist("#BA0C2F", "#9E1B32") < 100, "the raw primaries are too close");
  const c = g.gameTeamColors("Georgia Bulldogs", "Alabama Crimson Tide", "cfb");
  assert.equal(c.away, "#BA0C2F");
  assert.equal(c.home, "#0E2238");
  assert.ok(dist(c.away, c.home) >= 100);
});
test("gameTeamColors: a usable alternate is preferred to the navy fallback", () => {
  // Bills #00338D vs Giants #003C7F are two close blues; the Bills' alternate is a red
  const c = g.gameTeamColors("Buffalo Bills", "New York Giants", "nfl");
  assert.ok(dist(c.away, c.home) >= 100, JSON.stringify(c));
  assert.notEqual(c.home, "#0E2238");
});
test("gameTeamColors: two distinct colors are unchanged", () => {
  assert.deepEqual({ ...g.gameTeamColors("Kansas City Chiefs", "Buffalo Bills", "nfl") }, { away: g.teamAccent("Kansas City Chiefs", "nfl"), home: g.teamAccent("Buffalo Bills", "nfl") });
  assert.deepEqual({ ...g.gameTeamColors("Georgia Bulldogs", "Michigan Wolverines", "cfb") }, { away: g.teamAccent("Georgia Bulldogs", "cfb"), home: g.teamAccent("Michigan Wolverines", "cfb") });
});
test("gameTeamColors: an unmapped team (or a sport with no colors) still falls back to the default blue exactly as before", () => {
  const blue = g.teamAccent("Nobody FC", "cfb");
  assert.deepEqual({ ...g.gameTeamColors("Nobody FC", "Nobody United", "cfb") }, { away: blue, home: blue });
  assert.deepEqual({ ...g.gameTeamColors("Nobody FC", "Alabama Crimson Tide", "cfb") }, { away: blue, home: "#9E1B32" });
  assert.deepEqual({ ...g.gameTeamColors("Boston Red Sox", "New York Yankees", "mlb") }, { away: blue, home: blue });
});
