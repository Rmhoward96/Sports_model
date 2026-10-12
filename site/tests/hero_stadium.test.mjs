import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
const rule = (sel) => { const m = css.match(new RegExp(`(?:^|\\n|\\})${sel.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\{([^}]*)\\}`)); return m ? m[1] : ""; };
const PHOTO = { cfb: "football", nfl: "football", mlb: "baseball", nba: "basketball" };

test("stadium photos exist in site/public/stadiums, are JPEGs and stay light enough not to inline (< 500 KB)", () => {
  for (const f of ["football", "baseball", "basketball"]) {
    const u = new URL(`../public/stadiums/${f}.jpg`, import.meta.url), b = fs.readFileSync(u);
    assert.deepEqual([...b.subarray(0, 3)], [0xff, 0xd8, 0xff], `${f}.jpg is a JPEG`);
    assert.ok(b.length > 10_000 && b.length < 500_000, `${f}.jpg size ${b.length}`);
  }
});

test("each sport's hero rule sets its own stadium photo under a dark overlay (path is relative to css/, so ../public/...)", () => {
  for (const [sport, file] of Object.entries(PHOTO)) {
    const r = rule(`.ca-gm-hero.sport-${sport}`);
    assert.ok(r, `.ca-gm-hero.sport-${sport} rule exists`);
    assert.match(r, new RegExp(`background-image:linear-gradient\\([^;]*\\),url\\(\\.\\./public/stadiums/${file}\\.jpg\\),linear-gradient\\(180deg,#16314F`), `${sport} -> ${file}.jpg beneath an overlay, over the gradient fallback`);
    assert.ok(fs.existsSync(new URL(`../public/stadiums/${file}.jpg`, import.meta.url)), `${file}.jpg is where the rule points`);
  }
  assert.equal(new Set(["cfb", "nfl"].map((s) => rule(`.ca-gm-hero.sport-${s}`))).size, 1, "NFL and CFB share the football photo");
  assert.ok(/football\.jpg/.test(rule(".ca-gm-hero.sport-cfb")) && !/baseball|basketball/.test(rule(".ca-gm-hero.sport-cfb")));
});

test("photo sizing: cover, centred on the field / lights band, no tiling; rounded corners clip the photo", () => {
  const base = rule(".ca-gm-hero");
  assert.match(base, /border-radius:14px/, "hero keeps its 14px radius");
  const shared = css.match(/\.ca-gm-hero\.sport-cfb,[^{]*\{([^}]*)\}/);
  assert.ok(shared, "one shared rule for the sport photo sizing");
  assert.match(shared[1], /background-size:cover/);
  assert.match(shared[1], /background-position:center 40%/);
  assert.match(shared[1], /background-repeat:no-repeat/);
  for (const s of ["nfl", "cfb", "mlb", "nba"]) assert.ok(new RegExp(`\\.ca-gm-hero\\.sport-${s}[,{]`).test(css), `${s} listed`);
});

test("the dark overlay keeps white text readable: every overlay stop is >= .45 opaque and the base colour matches the dark gradient (no white flash while the photo loads)", () => {
  for (const sport of Object.keys(PHOTO)) {
    const r = rule(`.ca-gm-hero.sport-${sport}`), grad = r.match(/linear-gradient\(([^;]*?)\),url/)[1];
    const alphas = [...grad.matchAll(/rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*(\.?\d*\.?\d+)\s*\)/g)].map((m) => Number(m[1]));
    assert.ok(alphas.length >= 3, `${sport}: overlay has top / middle / bottom stops`);
    assert.ok(alphas.every((a) => a >= 0.45 && a <= 0.9), `${sport}: overlay alphas ${alphas}`);
  }
  assert.match(rule(".ca-gm-hero"), /background-color:#0C1C30/i, "dark base colour on the hero itself");
});

test("fallback: the original dark gradient stays on .ca-gm-hero, so a missing photo or an unknown sport still renders a dark hero", () => {
  const base = rule(".ca-gm-hero");
  assert.match(base, /background:[^;]*linear-gradient\(180deg,#16314F/, "gradient stays on the base rule");
  assert.doesNotMatch(base, /stadiums/, "the base rule names no photo");
  // the sport rules use background-image only (never the `background` shorthand), so they cannot erase the colour
  for (const s of Object.keys(PHOTO)) assert.doesNotMatch(rule(`.ca-gm-hero.sport-${s}`), /(^|;)\s*background:/, `${s} rule does not use the shorthand`);
  assert.deepEqual([...new Set([...css.matchAll(/stadiums\/(\w+)\.jpg/g)].map((m) => m[1]))].sort(), ["baseball", "basketball", "football"], "only the three supplied photos are referenced");
});

test("hero secondary text stays bright enough over the lightened overlay (measured >= 4.5:1 per text box on all three photos at 1440)", () => {
  assert.match(rule(".ca-gm-venue"), /color:#F2F6FB/i, "venue line");
  assert.match(rule(".ca-gm-rec"), /color:#EEF3F9/i, "team record");
  assert.match(rule(".ca-gm-when"), /color:#DCE6F2/i, "kickoff line");
});
