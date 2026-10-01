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
