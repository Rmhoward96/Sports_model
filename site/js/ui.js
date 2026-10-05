/* UI components + inline SVG charts. Pure: data in, HTML string out.
   Real data only: every chart returns "" for missing / too-short data and skips
   null / non-finite values instead of drawing them as zero.
   Depends on app.js globals: ctxEsc, logoImg. */
const uiFin = (v) => v != null && v !== "" && Number.isFinite(+v);
const uiNums = (arr) => (Array.isArray(arr) ? arr.filter(uiFin).map(Number) : []);
const uiClean = (t) => (/^-0(\.0+)?$/.test(t) ? t.slice(1) : t);
// compact number for chart labels: 12 / 12.5 / 0.25
const uiFmt = (v) => uiClean(String(+(+v).toFixed(Math.abs(v) >= 100 ? 0 : 1)));
const uiTicks = (lo, hi, n) => {
  const count = Math.max(2, n | 0), step = (hi - lo) / (count - 1) || 1;
  let dec = 0;
  while (dec < (step >= 1 ? 1 : 2) && Math.abs(step * 10 ** dec - Math.round(step * 10 ** dec)) > 1e-6) dec++;   // enough decimals that every tick label is exact
  return Array.from({ length: count }, (_, i) => { const v = lo + step * i; return { v, text: uiClean(v.toFixed(dec)) }; });
};
// vertical bar from baseline `base`, value height `hgt` (signed: +up / -down), rounded on the free end
function uiBarPath(x, base, hgt, w) {
  const h = Math.abs(hgt), r = Math.min(4, w / 2, h);
  const f = (n) => n.toFixed(1);
  if (hgt >= 0) {
    const t = base - h;
    return `M${f(x)} ${f(base)}L${f(x)} ${f(t + r)}Q${f(x)} ${f(t)} ${f(x + r)} ${f(t)}L${f(x + w - r)} ${f(t)}Q${f(x + w)} ${f(t)} ${f(x + w)} ${f(t + r)}L${f(x + w)} ${f(base)}Z`;
  }
  const b = base + h;
  return `M${f(x)} ${f(base)}L${f(x)} ${f(b - r)}Q${f(x)} ${f(b)} ${f(x + r)} ${f(b)}L${f(x + w - r)} ${f(b)}Q${f(x + w)} ${f(b)} ${f(x + w)} ${f(b - r)}L${f(x + w)} ${f(base)}Z`;
}

const oddsStr = (o) => (!uiFin(o) ? "" : +o > 0 ? `+${+o}` : `${+o}`);
const confPill = (tier) => (tier ? `<span class="ca-conf ${ctxEsc(tier)}">${ctxEsc(tier)}</span>` : "");
const alphaCell = (s) => (uiFin(s) ? `<span class="ca-alpha-cell${+s >= 85 ? " hi" : ""}">${s}</span>` : "");
const BOOK_STYLE = { draftkings: ["DK", "#0B3D2E"], fanduel: ["FD", "#1493FF"], betmgm: ["MGM", "#B59A5B"], williamhill_us: ["CZR", "#173F35"],
  fanatics: ["FAN", "#D21F3C"], espnbet: ["ESPN", "#D00"], hardrockbet: ["HR", "#5A2D82"], thescore: ["SCR", "#1E5BFF"],
  bet365: ["365", "#027B5B"], ballybet: ["BAL", "#C8102E"], pinnacle: ["PIN", "#0E2238"], betrivers: ["BR", "#1B3B6F"] };
function bookBadge(key) {
  const [abbr, color] = BOOK_STYLE[String(key || "").toLowerCase()] || [String(key || "?").slice(0, 3).toUpperCase(), "#5B6675"];
  return `<span class="ca-book" title="${ctxEsc(key)}" style="background:${color}">${ctxEsc(abbr)}</span>`;
}
// G2: text that does not fit a compact card moves into a native tooltip behind a small (i). `text` is plain text (escaped here).
const infoTip = (text) => (text ? `<span class="ca-info" title="${ctxEsc(text)}" tabindex="0" role="img" aria-label="${ctxEsc(text)}">i</span>` : "");
// Plain text of an HTML snippet, safe inside a double-quoted title="" attribute (tags stripped, quotes escaped).
const uiTitleText = (html) => String(html).replace(/<[^>]*>/g, "").replace(/"/g, "&quot;").trim();
// Stat-card label line: the (ellipsising) label text, then the optional (i) tooltip. The whole line carries its full text as a hover
// title so a label cut short by a narrow card stays readable. `note` = muted suffix such as "(30D)".
const statLabel = (label, note = "", tip = "") =>
  `<div class="ca-stat-label" title="${uiTitleText(`${label}${note ? ` ${note}` : ""}`)}"><span class="ca-stat-lt">${label}${note ? ` <span class="muted">${note}</span>` : ""}</span>${infoTip(tip)}</div>`;
// Compact stat card: label, big value and ONE short sub-line (ellipsised when long; its full text is the hover title). `tip` = optional
// longer explanation, shown as the (i) tooltip next to the label.
function statCard({ label, labelNote = "", value, valueClass = "", sub = "", subClass = "", visual = "", tip = "" }) {
  return `<div class="ca-card ca-stat">${statLabel(label, labelNote, tip)}
    <div class="ca-stat-body"><div><div class="ca-stat-value ${valueClass}">${value}</div>${sub ? `<div class="ca-stat-sub ${subClass}" title="${uiTitleText(sub)}">${sub}</div>` : ""}</div>${visual ? `<div class="ca-stat-visual">${visual}</div>` : ""}</div></div>`;
}
function pills(name, items, active) {
  return `<div class="ca-pills" role="tablist">${items.map(([k, l]) => `<button class="ca-pill${k === active ? " on" : ""}" data-pill="${ctxEsc(name)}" data-key="${ctxEsc(k)}" role="tab" aria-selected="${k === active}">${l}</button>`).join("")}</div>`;
}

function sparkline(values, { w = 120, h = 36, color = "var(--green)" } = {}) {
  const v = uiNums(values);
  if (v.length < 2) return "";
  const min = Math.min(...v), max = Math.max(...v), r = max - min || 1;
  const pts = v.map((x, i) => `${(i / (v.length - 1) * w).toFixed(1)},${(h - (x - min) / r * h).toFixed(1)}`).join(" ");
  return `<svg class="ca-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><polyline points="${pts}" fill="none" stroke="${ctxEsc(color)}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

function miniBars(values, { w = 70, h = 40, color = "var(--green)" } = {}) {
  const v = uiNums(values);
  if (!v.length) return "";
  const lo = Math.min(0, ...v), hi = Math.max(0, ...v), r = hi - lo || 1;
  const slot = w / v.length, gap = Math.min(3, slot * 0.3), bw = slot - gap, base = hi / r * h;   // n bars always fit in w
  const rects = v.map((x, i) => {
    const bh = Math.max(1, Math.abs(x) / r * h), y = x >= 0 ? base - bh : base;
    return `<rect x="${(i * slot).toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="${Math.min(2, bw / 2).toFixed(1)}" fill="${ctxEsc(color)}"/>`;
  }).join("");
  return `<svg class="ca-minibars" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${rects}</svg>`;
}

function donut(f, { size = 78, stroke = 10, color = "var(--navy)" } = {}) {
  if (!uiFin(f)) return "";
  const r = (size - stroke) / 2, C = 2 * Math.PI * r, x = Math.max(0, Math.min(1, +f)), c = size / 2;
  const arc = x > 0 ? `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${ctxEsc(color)}" stroke-width="${stroke}" stroke-dasharray="${(C * x).toFixed(2)} ${(C * (1 - x)).toFixed(2)}" transform="rotate(-90 ${c} ${c})" stroke-linecap="round"/>` : "";
  return `<svg class="ca-donut" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}"><circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="#E9E5DB" stroke-width="${stroke}"/>${arc}</svg>`;
}

// Shared axis scaffold for areaChart / lineChart. The SVG (grid, fill, lines) is stretched with
// preserveAspectRatio="none"; every piece of text and every dot is an HTML element positioned by
// percentage inside .ca-plot, so it renders at its CSS size at any container width.
// Flat data (all values equal, incl. zero) gets a symmetric pad so the line sits mid-chart, and ticks
// are generated from the same lo/hi the Y scale uses (they never leave [lo, hi]).
// `nice` widens [lo, hi] to multiples of a 1 / 2 / 5 x 10^k step (for counts: ticks 0, 50, 100 instead of 0, 42.75, 85.5).
function uiNiceTicks(lo, hi, n) {
  const raw = (hi - lo) / (Math.max(2, n | 0) - 1), pow = 10 ** Math.floor(Math.log10(raw)), m = raw / pow;
  const step = (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * pow;
  const a = Math.floor(lo / step + 1e-9) * step, b = Math.ceil(hi / step - 1e-9) * step, dec = step >= 1 ? 0 : Math.ceil(-Math.log10(step) - 1e-9);
  const ticks = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push({ v, text: uiClean(v.toFixed(dec)) });
  return { lo: a, hi: b, ticks };
}
function uiChartFrame(vals, { h, yTicks, unit, includeZero, xLabels, n, nice }) {
  let lo = Math.min(...vals), hi = Math.max(...vals), ticks = null;
  if (hi === lo) { const pad = lo ? Math.abs(lo) * 0.1 : 1; lo -= pad; hi += pad; }
  else {
    if (includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (nice) { const t = uiNiceTicks(lo, hi, yTicks); lo = t.lo; hi = t.hi; ticks = t.ticks; }
  }
  const r = hi - lo;
  const Y = (v) => (hi - v) / r * 100, X = (i) => (n > 1 ? i / (n - 1) * 100 : 0);   // percentages of the plot box
  ticks = ticks || uiTicks(lo, hi, yTicks);
  const grid = ticks.map((t) => `<line x1="0" x2="1000" y1="${Y(t.v).toFixed(2)}" y2="${Y(t.v).toFixed(2)}" class="ca-grid-line"/>`).join("");
  const yl = ticks.map((t) => `<span class="ca-yl" style="top:${Y(t.v).toFixed(2)}%">${t.text}${ctxEsc(unit)}</span>`).join("");
  const labels = Array.isArray(xLabels) ? xLabels : [], every = Math.max(1, Math.ceil(n / 7));
  const shown = labels.map((l, i) => i).filter((i) => labels[i] != null && (i % every === 0 || i === n - 1));
  const step = Math.max(1, Math.ceil(shown.length / 4)), last = shown.length - 1;   // <= 4 labels on narrow screens
  const xl = shown.map((i, k) => {
    const cls = `ca-xl${i === shown[0] && i === 0 ? " ca-xl-start" : ""}${i === n - 1 ? " ca-xl-end" : ""}${k === last || (k % step === 0 && last - k >= step) ? "" : " ca-xl-opt"}`;
    return `<span class="${cls}" style="left:${X(i).toFixed(2)}%">${ctxEsc(labels[i])}</span>`;
  }).join("");
  const dot = (i, v, color, small) => `<span class="ca-dot${small ? " ca-dot-sm" : ""}" style="left:${X(i).toFixed(2)}%;top:${Y(v).toFixed(2)}%;background:${ctxEsc(color)}"></span>`;
  const render = (svgBody, overlay = "") => `<div class="ca-chart" style="position:relative;height:${h}px"><div class="ca-plot"><svg viewBox="0 0 1000 100" preserveAspectRatio="none">${grid}${svgBody}</svg>${yl}${xl}${overlay}</div></div>`;
  return { X, Y, dot, render };
}
// Contiguous runs of finite values: nulls break a line instead of being interpolated or drawn as 0.
function uiSegments(values) {
  const segs = [];
  let cur = null;
  values.forEach((v, i) => { if (!uiFin(v)) { cur = null; return; } if (!cur) segs.push(cur = []); cur.push({ i, v: +v }); });
  return segs;
}
const uiSegPath = (segs, F) => segs.map((sg) => sg.map((p, k) => `${k ? "L" : "M"}${(F.X(p.i) * 10).toFixed(1)} ${F.Y(p.v).toFixed(2)}`).join(" ")).join(" ");

function areaChart(points, { w, h = 260, yTicks = 5, color = "var(--green)", unit = "u", nice = false } = {}) {
  const pts = Array.isArray(points) ? points : [];
  const ys = pts.map((p) => (p && uiFin(p.y) ? +p.y : null));
  const finite = ys.filter((v) => v != null);
  if (finite.length < 2) return "";
  const n = pts.length, col = ctxEsc(color), segs = uiSegments(ys);
  const F = uiChartFrame(finite, { h, yTicks, unit, includeZero: true, xLabels: pts.map((p) => p && p.x), n, nice });
  const runs = segs.filter((sg) => sg.length > 1);
  const fill = runs.map((sg) => `${uiSegPath([sg], F)} L${(F.X(sg[sg.length - 1].i) * 10).toFixed(1)} 100 L${(F.X(sg[0].i) * 10).toFixed(1)} 100Z`).join(" ");
  const endI = ys.length - 1 - [...ys].reverse().findIndex((v) => v != null);
  const isolated = segs.filter((sg) => sg.length === 1 && sg[0].i !== endI).map((sg) => F.dot(sg[0].i, sg[0].v, color, true)).join("");
  const body = (fill ? `<path d="${fill}" fill="${col}" opacity=".12"/>` : "") + `<path d="${uiSegPath(segs, F)}" fill="none" stroke="${col}" stroke-width="2.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>`;
  return F.render(body, isolated + F.dot(endI, ys[endI], color, false));
}

// Bar + value label, shared by groupedBars and histogramChart. hgt is signed (+up / -down from base).
function uiBar(x, base, hgt, bw, color, text) {
  const ly = hgt >= 0 ? base - hgt - 5 : base - hgt + 13;
  return `<path class="ca-bar" d="${uiBarPath(x, base, hgt, bw)}" fill="${ctxEsc(color || "var(--green)")}"/><text class="ca-bar-val" x="${(x + bw / 2).toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle">${ctxEsc(text)}</text>`;
}

function groupedBars(groups, { h = 180, bw = 26, bgap = 6, ggap = 30 } = {}) {
  const gs = (Array.isArray(groups) ? groups : []).map((g) => ({ label: g && g.label, sub: g && g.sub, bars: ((g && g.bars) || []).filter((b) => b && uiFin(b.value)) })).filter((g) => g.bars.length);
  if (!gs.length) return "";
  const all = gs.flatMap((g) => g.bars.map((b) => +b.value));
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all), r = hi - lo || 1;
  const sub = gs.some((g) => g.sub != null && g.sub !== "");   // a second caption line (e.g. "W-L-P (n)") under the group label
  const top = 20, bottom = (lo < 0 ? 42 : 26) + (sub ? 16 : 0), plot = h - top - bottom, base = top + hi / r * plot;
  let x = ggap / 2, out = "";
  for (const g of gs) {
    const gw = g.bars.length * bw + (g.bars.length - 1) * bgap;
    g.bars.forEach((b, i) => { out += uiBar(x + i * (bw + bgap), base, +b.value / r * plot, bw, b.color, b.label != null && b.label !== "" ? b.label : uiFmt(b.value)); });
    const cx = (x + gw / 2).toFixed(1), hasSub = g.sub != null && g.sub !== "";
    if (g.label != null) out += `<text class="ca-axis${hasSub ? " ca-axis-main" : ""}" x="${cx}" y="${h - (hasSub ? 24 : 8)}" text-anchor="middle">${ctxEsc(g.label)}</text>`;
    if (hasSub) out += `<text class="ca-axis-sub" x="${cx}" y="${h - 7}" text-anchor="middle">${ctxEsc(g.sub)}</text>`;
    x += gw + ggap;
  }
  const W = Math.round(x - ggap / 2);
  return `<svg class="ca-bars" viewBox="0 0 ${W} ${h}" width="${W}" height="${h}"><line x1="0" x2="${W}" y1="${base.toFixed(1)}" y2="${base.toFixed(1)}" class="ca-grid-line"/>${out}</svg>`;
}

function histogramChart(bins, { w = 520, h = 180 } = {}) {
  const bs = Array.isArray(bins) ? bins.filter((b) => b && uiFin(b.n) && +b.n >= 0) : [];
  const max = Math.max(0, ...bs.map((b) => +b.n));
  if (!bs.length || max <= 0) return "";
  const top = 20, bottom = 24, plot = h - top - bottom, base = top + plot, slot = w / bs.length, bw = Math.min(44, slot * 0.72);
  const every = Math.max(1, Math.ceil(bs.length / 12));
  const out = bs.map((b, i) => {
    const x = i * slot + (slot - bw) / 2;
    return uiBar(x, base, +b.n / max * plot, bw, b.color, uiFmt(b.n))
      + (i % every === 0 ? `<text class="ca-axis" x="${(x + bw / 2).toFixed(1)}" y="${h - 6}" text-anchor="middle">${ctxEsc(b.label)}</text>` : "");
  }).join("");
  return `<svg class="ca-hist" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><line x1="0" x2="${w}" y1="${base}" y2="${base}" class="ca-grid-line"/>${out}</svg>`;
}

function lineChart(series, xLabels, { w, h = 260, yTicks = 5, unit = "" } = {}) {
  const PALETTE = ["var(--green)", "var(--navy)", "var(--blue)", "var(--red)"];
  const ss = (Array.isArray(series) ? series : []).map((s, si) => ({ label: s && s.label, color: (s && s.color) || PALETTE[si % PALETTE.length], values: (s && Array.isArray(s.values)) ? s.values : [] }))
    .filter((s) => s.values.filter(uiFin).length >= 2);
  if (!ss.length) return "";
  const n = Math.max(Array.isArray(xLabels) ? xLabels.length : 0, ...ss.map((s) => s.values.length));
  const F = uiChartFrame(ss.flatMap((s) => uiNums(s.values)), { h, yTicks, unit, includeZero: false, xLabels, n });
  const lines = ss.map((s) => `<path d="${uiSegPath(uiSegments(s.values), F)}" fill="none" stroke="${ctxEsc(s.color)}" stroke-width="2.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>`).join("");
  const isolated = ss.map((s) => uiSegments(s.values).filter((sg) => sg.length === 1).map((sg) => F.dot(sg[0].i, sg[0].v, s.color, true)).join("")).join("");
  const legend = ss.filter((s) => s.label != null).map((s) => `<span class="ca-legend-item"><i style="background:${ctxEsc(s.color)}"></i>${ctxEsc(s.label)}</span>`).join("");
  return `<div class="ca-linechart">${F.render(lines, isolated)}${legend ? `<div class="ca-legend">${legend}</div>` : ""}</div>`;
}

function donutLegend(rows) {
  const rs = (Array.isArray(rows) ? rows : []).filter((r) => r && r.label != null && r.label !== "");
  if (!rs.length) return "";
  return `<div class="ca-donut-legend">${rs.map((r) => `<div class="ca-dl-row"><i class="ca-dl-dot" style="background:${ctxEsc(r.color || "var(--navy)")}"></i><span class="ca-dl-label">${ctxEsc(r.label)}</span>${uiFin(r.pct) ? `<span class="ca-dl-pct">${Math.round(+r.pct)}%</span>` : ""}${r.value != null && r.value !== "" ? `<span class="ca-dl-val">${ctxEsc(r.value)}</span>` : ""}</div>`).join("")}</div>`;
}

// Shared 22px line icons (stroke follows currentColor).
const ICON_TREND = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/></svg>`;
const ICON_CLOCK = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>`;
const ICON_WAVE = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20c4 0 4-16 8-16s4 16 8 16"/></svg>`;
const ICON_CAL = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>`;

// Labelled <select> for the filter rows (+EV, Track Record). options: [[value, label]]; a current value that is not in the list is
// appended so the control never shows something other than the state. `attr` is the data attribute the page's handler listens for.
function selectField(attr, key, label, options, value, extraCls = "") {
  const opts = options.some(([k]) => String(k) === String(value)) ? options : [...options, [value, String(value)]];
  return `<label class="ca-f ${extraCls}"><span>${label}</span><select class="ca-select" ${attr}="${ctxEsc(key)}">${opts.map(([k, l]) => `<option value="${ctxEsc(k)}"${String(k) === String(value) ? " selected" : ""}>${ctxEsc(l)}</option>`).join("")}</select></label>`;
}
// Search box with the magnifier (ICON_SEARCH, shell.js).
const searchField = (attr, value, placeholder, aria) => `<label class="ca-search">${ICON_SEARCH}<input class="ca-input" type="search" ${attr} placeholder="${ctxEsc(placeholder)}" value="${ctxEsc(value)}" aria-label="${ctxEsc(aria || placeholder)}" autocomplete="off"></label>`;
