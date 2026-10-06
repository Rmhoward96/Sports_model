// MLB reactivation (2026-10-05): MLB is a live sport and no hard-coded "paused" text may remain in the site source.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";

const read = (p) => fs.readFileSync(new URL(`../${p}`, import.meta.url), "utf8");
const SRC = ["app.js", "js/data.js", "js/shell.js", "js/ui.js", "js/metrics.js", "js/boot.js", ...fs.readdirSync(new URL("../js/pages", import.meta.url)).map((f) => `js/pages/${f}`)];

test("no MLB-paused flag or date is hard-coded anywhere in the site source", () => {
  for (const f of SRC) {
    const s = read(f);
    assert.ok(!/MLB model paused \(/.test(s), `${f}: hard-coded MLB pause status`);
    assert.ok(!/Aug 31, 2026/.test(s), `${f}: hard-coded pause date`);
    assert.ok(!/"MLB model paused\."/.test(s), `${f}: hard-coded MLB paused sentence`);
  }
});

test("data.js: MLB is live with no status; only NBA carries a no-model status; MLB props are labelled", () => {
  const s = read("js/data.js");
  assert.match(s, /const LIVE_SPORTS = \["nfl", "cfb", "mlb"\]/);
  const status = s.match(/const SPORT_STATUS = (\{[^}]*\})/)[1];
  assert.ok(!/mlb/.test(status) && /nba/.test(status));
  for (const m of ["total_bases", "pitcher_ks", "hits_allowed", "outs_recorded"]) assert.ok(s.includes(`${m}:`), m);
  for (const dropped of ["hits:", "hrr:", "home_run:"]) assert.ok(!new RegExp(`PROP_LABEL[\\s\\S]{0,600}${dropped}`).test(s.split("const segmentKey")[0]), dropped);
});
