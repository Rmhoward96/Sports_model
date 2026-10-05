import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/boot.js"]);
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
  assert.match(svg, /class="ca-dot"/);      // end dot (HTML, crisp at any width)
  assert.match(svg, /<span class="ca-xl[^"]*" style="left:100\.00%">c</);   // x label
});
test("area chart: too-short data renders nothing; nulls break the line (no interpolation, no zeros)", () => {
  assert.equal(g.areaChart([]), "");
  assert.equal(g.areaChart(null), "");
  assert.equal(g.areaChart([{ x: "a", y: 1 }, { x: "b", y: null }]), "");
  const svg = g.areaChart([{ x: "a", y: 0 }, { x: "b", y: 5 }, { x: "c", y: null }, { x: "d", y: 10 }, { x: "e", y: 12 }]);
  assert.equal(count(svg, /<path /g), 2);
  const line = svg.match(/<path d="([^"]*)" fill="none"/)[1];
  assert.equal(count(line, /M/g), 2);       // two segments: a-b and d-e
  assert.equal(count(svg, /class="ca-dot"/g), 1);
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
test("groupedBars: an optional per-group `sub` caption renders a second line under the group label, escaped", () => {
  const svg = g.groupedBars([{ label: "Moneyline", sub: "109-76-9 <(194)>", bars: [{ value: 59, label: "59%" }] }, { label: "Spread", bars: [{ value: 40 }] }], { h: 190 });
  assert.equal(count(svg, /class="ca-axis-sub"/g), 1);
  assert.match(svg, />109-76-9 &lt;\(194\)&gt;</);
  assert.doesNotMatch(svg, /<\(194\)>/);
  const plain = g.groupedBars([{ label: "NFL", bars: [{ value: 5 }] }], { h: 180 });
  assert.doesNotMatch(plain, /ca-axis-sub/);
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
test("statCard, pills, bookBadge", () => {
  const card = g.statCard({ label: "ROI", value: "+4.2%", valueClass: "pos", sub: "30d", visual: "<i></i>" });
  assert.match(card, /ca-stat-value pos">\+4\.2%</); assert.match(card, /ca-stat-visual/);
  assert.doesNotMatch(g.statCard({ label: "x", value: "1" }), /ca-stat-visual|ca-stat-sub/);
  const tip = g.statCard({ label: "Model Hit Rate", value: "56%", sub: "<b>+4%</b> vs. \"market\"", tip: "68-52 · 120 graded <picks>" });
  assert.match(tip, /<span class="ca-info" title="68-52 · 120 graded &lt;picks&gt;" tabindex="0"/, "long text moves into an escaped (i) tooltip next to the label");
  assert.match(tip, /class="ca-stat-sub " title="\+4% vs\. &quot;market&quot;"/, "the sub-line carries its full text as a hover title");
  assert.doesNotMatch(g.statCard({ label: "x", value: "1" }), /ca-info/);
  // the label carries its own tag-stripped, escaped text as a title (readable when ellipsised); the (i) sits outside the ellipsised span
  const lab = g.statCard({ label: "Live +EV <b>\"Opps\"</b>", labelNote: "(30D)", value: "1", tip: "t" });
  assert.match(lab, /<div class="ca-stat-label" title="Live \+EV &quot;Opps&quot; \(30D\)"><span class="ca-stat-lt">Live \+EV <b>"Opps"<\/b> <span class="muted">\(30D\)<\/span><\/span><span class="ca-info"/);
  assert.equal(g.infoTip(""), "");
  const p = g.pills("sport", [["nfl", "NFL"], ["cfb", "CFB"]], "cfb");
  assert.match(p, /data-pill="sport" data-key="nfl"[^>]*aria-selected="false"/);
  assert.match(p, /ca-pill on" data-pill="sport" data-key="cfb"[^>]*aria-selected="true"/);
  assert.match(g.bookBadge("draftkings"), />DK</);
  assert.match(g.bookBadge("weird<book>"), /title="weird&lt;book&gt;"/);
});

test("uiTitleText: tags become spaces (no fused words), whitespace collapsed, quotes escaped", () => {
  assert.equal(g.uiTitleText('Rec Yds</span><i>at</i>'), "Rec Yds at");
  assert.equal(g.uiTitleText('<b>Rec</b><b>Yds</b>'), "Rec Yds");
  assert.equal(g.uiTitleText('  Under   85.5 \n <span class="x">Rec Yds</span> '), "Under 85.5 Rec Yds");
  assert.equal(g.uiTitleText('say "hi" <i>now</i>'), "say &quot;hi&quot; now");
  const card = g.statCard({ label: "x", value: "1", sub: "Falco<small>ns</small> <i>at</i> Saints" });
  assert.match(card, /class="ca-stat-sub " title="Falco ns at Saints"/);
});

const pct = (svg, cls) => [...svg.matchAll(new RegExp(`class="${cls}[^"]*" style="[^"]*?(?:top|left):(-?[\\d.]+)%`, "g"))].map((m) => +m[1]);
const lineYs = (svg) => [...svg.matchAll(/[ML][\d.]+ ([\d.]+)/g)].map((m) => +m[1]);

test("area/line charts: HTML text overlay, SVG only draws strokes/fill, no <text>", () => {
  const a = g.areaChart([{ x: "a", y: 1 }, { x: "b", y: 3 }]);
  assert.match(a, /^<div class="ca-chart" style="position:relative;height:260px">/);
  assert.match(a, /<svg viewBox="0 0 1000 100" preserveAspectRatio="none">/);
  assert.doesNotMatch(a, /<text/);
  assert.match(g.lineChart([{ label: "A", values: [1, 2] }], ["x", "y"]), /class="ca-chart"/);
});
test("flat data: ticks stay inside the chart and the line sits mid-height", () => {
  const a = g.areaChart([{ x: "a", y: 0 }, { x: "b", y: 0 }]);
  const ay = pct(a, "ca-yl");
  assert.equal(ay.length, 5);
  assert.ok(ay.every((p) => p >= 0 && p <= 100), `ticks ${ay}`);
  assert.deepEqual([...new Set(lineYs(a.match(/<path d="([^"]*)" fill="none"/)[1]))], [50]);
  assert.equal(+a.match(/class="ca-dot" style="left:[\d.]+%;top:([\d.]+)%/)[1], 50);
  const l = g.lineChart([{ label: "S", values: [5, 5, 5] }], ["a", "b", "c"]);
  const ly = pct(l, "ca-yl");
  assert.ok(ly.every((p) => p >= 0 && p <= 100), `ticks ${ly}`);
  assert.deepEqual([...new Set(lineYs(l.match(/<path d="([^"]*)" fill="none"/)[1]))], [50]);
  assert.match(l, />4\.75</); assert.match(l, />5\.00</);   // exact tick labels, flat value is a tick
  // non-flat data: first/last ticks are exactly the data range
  const m = pct(g.lineChart([{ values: [1, 3] }], ["a", "b"]), "ca-yl");
  assert.equal(Math.min(...m), 0); assert.equal(Math.max(...m), 100);
});
test("x labels thin out on busy axes and escape markup", () => {
  const pts = Array.from({ length: 30 }, (_, i) => ({ x: i === 3 ? "<b>" : `d${i}`, y: i }));
  const svg = g.areaChart(pts);
  assert.ok(count(svg, /class="ca-xl/g) <= 8);
  assert.ok(count(svg, /ca-xl-opt/g) >= count(svg, /class="ca-xl/g) - 4);   // at most 4 survive on narrow screens
  const svg2 = g.areaChart([{ x: "<img src=x>", y: 1 }, { x: "b", y: 2 }]);
  assert.match(svg2, /&lt;img src=x&gt;/); assert.doesNotMatch(svg2, /<img/);
});
test("series colors and labels are escaped in line charts", () => {
  const bad = 'red;" onclick="alert(1)';
  const svg = g.lineChart([{ label: "<i>x</i>", color: bad, values: [1, 2, 3] }], ["a", "b", "c"]);
  assert.doesNotMatch(svg, /" onclick="/);
  assert.match(svg, /&quot; onclick=&quot;/);
  assert.match(svg, /&lt;i&gt;x&lt;\/i&gt;/);
  const area = g.areaChart([{ x: "a", y: 1 }, { x: "b", y: 2 }], { color: bad });
  assert.doesNotMatch(area, /" onclick="/);
});
test("miniBars: n bars always fit inside w", () => {
  for (const n of [1, 5, 30, 90]) {
    const svg = g.miniBars(Array.from({ length: n }, (_, i) => i + 1), { w: 70 });
    const rects = [...svg.matchAll(/<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)"/g)].map((m) => [+m[1], +m[2]]);
    assert.equal(rects.length, n);
    const [x, w] = rects.at(-1);
    assert.ok(x + w <= 70.05, `n=${n}: last bar ends at ${x + w}`);
    assert.ok(rects.every(([, bw]) => bw > 0));
  }
});
test("groupedBars: negative bars extend below the baseline and labels do not collide", () => {
  const svg = g.groupedBars([{ label: "Group", bars: [{ value: 10, color: "#1E8E4E" }, { value: -8, color: "#C8372D" }] }], { h: 180 });
  const base = +svg.match(/<line [^>]*y1="([\d.]+)"/)[1];
  const [up, down] = [...svg.matchAll(/<path class="ca-bar" d="([^"]*)"/g)].map((m) => m[1].replace(/[A-Z]/g, " ").trim().split(/\s+/).map(Number).filter((_, i) => i % 2));   // y coords only
  assert.ok(Math.min(...up) < base - 1 && Math.max(...up) <= base + 0.05);       // positive: above
  assert.ok(Math.max(...down) > base + 1 && Math.min(...down) >= base - 0.05);   // negative: below
  const vals = [...svg.matchAll(/class="ca-bar-val" x="[\d.]+" y="([\d.]+)"/g)].map((m) => +m[1]);
  const groupY = +svg.match(/class="ca-axis" x="[\d.]+" y="([\d.]+)"/)[1];
  assert.ok(vals[1] > base);                                  // negative label sits below its bar start
  assert.ok(vals[1] <= groupY - 12, `neg label ${vals[1]} vs group label ${groupY}`);
  assert.ok(Math.max(...down) < 180 && vals[0] > 0);          // everything stays inside the viewBox
});
test("selectField / searchField: labelled select keeps an unknown current value, escapes everything; search box carries its attribute", () => {
  const html = g.selectField("data-x", "league", "League <b>", [["", "All"], ["nfl", "NFL"]], "cfb<", "wide");
  assert.match(html, /<label class="ca-f wide"><span>League <b><\/span>/);   // the label is trusted markup from the page
  assert.match(html, /<select class="ca-select" data-x="league">/);
  assert.match(html, /<option value="cfb&lt;" selected>cfb&lt;<\/option>/);
  assert.equal(count(html, /<option /g), 3);
  assert.match(g.selectField("data-x", "k", "L", [["a", "A"]], "a"), /<option value="a" selected>A<\/option>/);
  const s = g.searchField("data-q", `x"y`, "Search...", "Search it");
  assert.match(s, /data-q placeholder="Search\.\.\."/); assert.match(s, /value="x&quot;y"/); assert.match(s, /aria-label="Search it"/);
});
test("areaChart nice ticks: counts get round steps and the line stays inside the widened range; default ticks unchanged", () => {
  const pts = [0, 40, 171].map((y, i) => ({ x: ["a", "b", "c"][i], y }));
  const nice = g.areaChart(pts, { yTicks: 5, unit: "", nice: true });
  for (const t of ["0", "50", "100", "150", "200"]) assert.match(nice, new RegExp(`>${t}<`), `tick ${t}`);
  assert.doesNotMatch(nice, />42\.8</);
  const plain = g.areaChart(pts, { yTicks: 5, unit: "" });
  assert.match(plain, />42\.8</); assert.match(plain, />171\.0</);
  const neg = g.areaChart([{ x: "a", y: -18 }, { x: "b", y: 7 }, { x: "c", y: 33 }], { yTicks: 5, unit: "", nice: true });
  assert.match(neg, />-20</); assert.match(neg, />40</); assert.match(neg, />0</);
  const flat = g.areaChart([{ x: "a", y: 3 }, { x: "b", y: 3 }], { nice: true });
  assert.ok(flat.includes("ca-chart"), "flat data still draws");
});
test("groupedBars: bar width / gap options widen the bars (defaults unchanged)", () => {
  const groups = [{ label: "A", bars: [{ value: 5 }, { value: 7 }] }];
  const w = (o) => +g.groupedBars(groups, o).match(/viewBox="0 0 (\d+) /)[1];
  assert.equal(w({}), 2 * 26 + 6 + 30, "default: two 26px bars, 6px gap, 30px group gap");
  assert.equal(w({ bw: 40, bgap: 10, ggap: 50 }), 2 * 40 + 10 + 50);
});

test("resultChip: W / L / P chips with an escaped hover title, nothing for an ungraded pick", () => {
  assert.equal(g.resultChip("W"), '<span class="ca-res W">W</span>');
  assert.equal(g.resultChip("L", 'Lions "ML" <lost>'), '<span class="ca-res L" title="Lions &quot;ML&quot; &lt;lost&gt;">L</span>');
  assert.equal(g.resultChip("P"), '<span class="ca-res P">P</span>');
  assert.equal(g.resultChip(null), ""); assert.equal(g.resultChip("X"), "");
});
