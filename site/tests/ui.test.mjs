import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/boot.js"]);
const count = (s, re) => (s.match(re) || []).length;

test("conf pill + alpha cell", () => {
  assert.equal(g.confPill(null), "");
  assert.match(g.confPill("STRONG"), /class="ca-conf STRONG">STRONG</);
  assert.match(g.alphaCell(89), /ca-alpha-cell hi/);
  assert.doesNotMatch(g.alphaCell(70), / hi/);
  assert.equal(g.alphaCell(null), "");
  assert.equal(g.oddsStr(128), "+128"); assert.equal(g.oddsStr(-110), "-110");
  assert.equal(g.oddsStr(null), "");
});
test("sparkline points and empty input", () => {
  assert.equal(g.sparkline([1]), "");
  assert.equal(g.sparkline(null), "");
  const svg = g.sparkline([0, 5, 10], { w: 100, h: 20 });
  assert.match(svg, /points="0\.0,20\.0 50\.0,10\.0 100\.0,0\.0"/);
  assert.match(svg, /stroke="var\(--green\)"/);
});
test("sparkline ignores null and non-finite values instead of drawing zeros", () => {
  const svg = g.sparkline([0, null, NaN, 10], { w: 100, h: 20 });
  assert.match(svg, /points="0\.0,20\.0 100\.0,0\.0"/);
  assert.equal(g.sparkline([5, null, undefined]), "");
});
test("donut arc length matches the fraction", () => {
  const svg = g.donut(0.25, { size: 100, stroke: 10 });
  const C = 2 * Math.PI * 45;
  assert.match(svg, new RegExp(`stroke-dasharray="${(C * 0.25).toFixed(2)} ${(C * 0.75).toFixed(2)}"`));
  assert.match(svg, /stroke="#E9E5DB"/);
  assert.equal(g.donut(null), "");
  assert.equal(g.donut(NaN), "");
});
test("area chart renders one path per series and y ticks", () => {
  const svg = g.areaChart([{ x: "a", y: -10 }, { x: "b", y: 0 }, { x: "c", y: 30 }], { yTicks: 5 });
  assert.equal(count(svg, /<path /g), 2);   // fill + line
  assert.match(svg, />30u?</);
  assert.match(svg, /<circle /);            // end dot
  assert.match(svg, />c</);                 // x label
});
test("area chart: too-short or null data renders nothing; nulls are skipped", () => {
  assert.equal(g.areaChart([]), "");
  assert.equal(g.areaChart(null), "");
  assert.equal(g.areaChart([{ x: "a", y: 1 }, { x: "b", y: null }]), "");
  const svg = g.areaChart([{ x: "a", y: 0 }, { x: "b", y: null }, { x: "c", y: 10 }]);
  assert.equal(count(svg, /<path /g), 2);
  assert.doesNotMatch(svg, />b</);
});
test("miniBars: one rect per finite value, empty input renders nothing", () => {
  const svg = g.miniBars([1, 2, 3, null, 4], { w: 70, h: 40 });
  assert.equal(count(svg, /<rect /g), 4);
  assert.equal(g.miniBars([]), "");
  assert.equal(g.miniBars(null), "");
  assert.equal(g.miniBars([null, NaN]), "");
});
test("groupedBars: one bar + value label per bar, group labels, empty renders nothing", () => {
  const groups = [
    { label: "NFL", bars: [{ value: 55, color: "#1E8E4E", label: "55%" }, { value: 60, color: "#0E2238" }] },
    { label: "CFB", bars: [{ value: 52, color: "#1E8E4E" }, { value: null, color: "#0E2238" }] },
  ];
  const svg = g.groupedBars(groups, { h: 180 });
  assert.equal(count(svg, /class="ca-bar"/g), 3);        // null bar skipped
  assert.equal(count(svg, /class="ca-bar-val"/g), 3);
  assert.match(svg, />55%</);                            // explicit label wins
  assert.match(svg, />NFL</); assert.match(svg, />CFB</);
  assert.equal(g.groupedBars([]), "");
  assert.equal(g.groupedBars(null), "");
  assert.equal(g.groupedBars([{ label: "x", bars: [{ value: null }] }]), "");
});
test("histogramChart: one bar per bin with its color, empty or all-zero renders nothing", () => {
  const bins = [{ label: "-2", n: 3, color: "#C8372D" }, { label: "0", n: 7, color: "#1E8E4E" }, { label: "2", n: null, color: "#1E8E4E" }];
  const svg = g.histogramChart(bins);
  assert.equal(count(svg, /class="ca-bar"/g), 2);
  assert.match(svg, /fill="#C8372D"/); assert.match(svg, /fill="#1E8E4E"/);
  assert.match(svg, />7</);
  assert.equal(g.histogramChart([]), "");
  assert.equal(g.histogramChart(null), "");
  assert.equal(g.histogramChart([{ label: "a", n: 0 }, { label: "b", n: 0 }]), "");
});
test("lineChart: one path per drawable series, gaps for nulls, empty renders nothing", () => {
  const series = [
    { label: "Model", color: "#1E8E4E", values: [0, 5, 10] },
    { label: "Market", color: "#0E2238", values: [0, null, 4] },
    { label: "Short", color: "#C8372D", values: [3] },
  ];
  const svg = g.lineChart(series, ["Jan", "Feb", "Mar"]);
  assert.equal(count(svg, /<path /g), 2);            // "Short" has < 2 finite points
  assert.match(svg, /d="M[^"]* M[^"]*"/);              // the null in "Market" splits its path into two segments
  assert.match(svg, />Jan</); assert.match(svg, />Mar</);
  assert.match(svg, />Model</);                      // legend
  assert.equal(g.lineChart([], []), "");
  assert.equal(g.lineChart(null, null), "");
  assert.equal(g.lineChart([{ label: "x", values: [1] }], ["a"]), "");
});
test("donutLegend: one row per labelled row, escapes text, empty renders nothing", () => {
  const html = g.donutLegend([
    { label: "NFL <ML>", pct: 62.4, value: "$1,240", color: "#1E8E4E" },
    { label: "CFB", pct: 37.6, value: "$748", color: "#0E2238" },
    { pct: 1, value: "x" },
  ]);
  assert.equal(count(html, /class="ca-dl-row"/g), 2);
  assert.match(html, /NFL &lt;ML&gt;/);
  assert.match(html, />62%</);
  assert.match(html, /background:#1E8E4E/);
  assert.equal(g.donutLegend([]), "");
  assert.equal(g.donutLegend(null), "");
});
test("statCard, pills, bookBadge, teamCell", () => {
  const card = g.statCard({ label: "ROI", value: "+4.2%", valueClass: "pos", sub: "30d", visual: "<i></i>" });
  assert.match(card, /ca-stat-value pos">\+4\.2%</); assert.match(card, /ca-stat-visual/);
  assert.doesNotMatch(g.statCard({ label: "x", value: "1" }), /ca-stat-visual|ca-stat-sub/);
  const p = g.pills("sport", [["nfl", "NFL"], ["cfb", "CFB"]], "cfb");
  assert.match(p, /data-pill="sport" data-key="nfl"[^>]*aria-selected="false"/);
  assert.match(p, /ca-pill on" data-pill="sport" data-key="cfb"[^>]*aria-selected="true"/);
  assert.match(g.bookBadge("draftkings"), />DK</);
  assert.match(g.bookBadge("weird<book>"), /title="weird&lt;book&gt;"/);
  assert.match(g.teamCell("Kansas City Chiefs", "nfl"), /class="ca-team">.*Kansas City Chiefs</);
  assert.match(g.teamCell("A<B", "nfl"), /A&lt;B/);
});
