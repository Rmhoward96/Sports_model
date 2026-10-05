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

test("theme.css carries no rules for classes nothing renders (.ca-ev-pill, .ca-grid, .ca-section-title)", () => {
  const css = fs.readFileSync(new URL("../css/theme.css", import.meta.url), "utf8");
  for (const dead of [".ca-ev-pill", ".ca-grid{", ".ca-section-title"]) assert.ok(!css.includes(dead), `${dead} still styled`);
  assert.ok(css.includes(".ca-grid-line"), "the chart grid-line rule stays");
});
