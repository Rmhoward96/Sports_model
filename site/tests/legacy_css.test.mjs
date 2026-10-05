import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
test("legacy injected CSS uses theme tokens, not dark-theme colors", () => {
  const src = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const css = src.slice(src.indexOf("function injectStylesOnce"), src.indexOf("async function render"));
  for (const dark of ["#080d13", "#101720", "#0b1118", "#111923", "#0f1620", "#141d27", "#131a22", "#18202a", "#1a2430", "#202a35", "#263341"]) {
    assert.ok(!css.toLowerCase().includes(dark), `dark color ${dark} still in legacy CSS`);
  }
  assert.ok(!fs.existsSync(new URL("../styles.css", import.meta.url)), "styles.css should be deleted");
});

test("theme.css carries no rules for classes nothing renders (.ca-ev-pill, .ca-grid, .ca-section-title, .ca-trk-note)", () => {
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  for (const dead of [".ca-ev-pill", ".ca-grid{", ".ca-section-title", ".ca-trk-note"]) assert.ok(!css.includes(dead), `${dead} still styled`);
  assert.ok(css.includes(".ca-grid-line"), "the chart grid-line rule stays");
});

test("final-review CSS contracts: Movers tabs stay inside the card, controls inherit Inter, dashboard headers follow the fluid table token", () => {
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  // 1: equal columns may shrink (minmax(0,1fr)) with an ellipsis guard, and the compact container size wraps the labels
  assert.match(css, /\.ca-dash-tabs \.ca-pills\{[^}]*grid-auto-columns:minmax\(0,1fr\)/, "tab pills use minmax(0,1fr) columns");
  assert.doesNotMatch(css, /grid-auto-columns:1fr/, "no bare 1fr columns (they grow past the card)");
  assert.match(css, /\.ca-dash-tabs \.ca-pill\{[^}]*min-width:0[^}]*text-overflow:ellipsis/, "tab pills clip with an ellipsis instead of overflowing");
  assert.match(css, /#dash-movers\{container-type:inline-size\}/, "Market Movers is a size container");
  assert.match(css, /@container \(max-width:470px\)\{\.ca-dash-card \.ca-dash-tabs \.ca-pills\{[^}]*\}\.ca-dash-card \.ca-dash-tabs \.ca-pill\{[^}]*white-space:normal/, "compact container query lets tab labels wrap");
  // 2: buttons / selects / inputs inherit the page font (pills rendered in Arial)
  assert.match(css, /(^|\n)button,select,input,textarea\{font-family:inherit\}/, "global form-control font inheritance");
  // 4: no fixed 11.5px dashboard header; th follows --fs-table
  assert.doesNotMatch(css, /\.ca-dash-table th\{[^}]*font-size:11\.5px/, "no fixed 11.5px dashboard header size");
  assert.match(css, /\.ca-dash-table th\{[^}]*font-size:var\(--fs-table\)/, "base dashboard th follows --fs-table");
  assert.match(css, /\.ca-dash \.ca-dash-table th,[^{]*\{[^}]*font-size:var\(--fs-table\)/, "scoped th follows --fs-table");
});

test("the CFB league logo is an inline football (ESPN's ncaa_football image is a person silhouette); other leagues keep their images", async () => {
  const { loadScripts } = await import("./load.mjs");
  const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js"]);
  const cfb = g.leagueLogo("cfb");
  assert.match(cfb, /^<svg class="ca-league"/);
  assert.doesNotMatch(cfb, /ncaa_football|<img/);
  for (const s of ["nfl", "mlb", "nba"]) assert.match(g.leagueLogo(s), /^<img class="ca-league" src="https:\/\/a\.espncdn\.com\/i\/teamlogos\/leagues\/500\//);
  assert.equal(g.leagueLogo("zzz"), "");
});
