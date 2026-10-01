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
  const dec = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
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
const teamCell = (name, sport) => `<span class="ca-team">${logoImg(name, sport)}${ctxEsc(name)}</span>`;
function statCard({ label, labelNote = "", value, valueClass = "", sub = "", subClass = "", visual = "" }) {
  return `<div class="ca-card ca-stat"><div class="ca-stat-label">${label}${labelNote ? ` <span class="muted">${labelNote}</span>` : ""}</div>
    <div class="ca-stat-body"><div><div class="ca-stat-value ${valueClass}">${value}</div>${sub ? `<div class="ca-stat-sub ${subClass}">${sub}</div>` : ""}</div>${visual ? `<div class="ca-stat-visual">${visual}</div>` : ""}</div></div>`;
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
  const lo = Math.min(0, ...v), hi = Math.max(0, ...v), r = hi - lo || 1, gap = 3;
  const bw = Math.max(1, (w - gap * (v.length - 1)) / v.length), base = hi / r * h;
  const rects = v.map((x, i) => {
    const bh = Math.max(1, Math.abs(x) / r * h), y = x >= 0 ? base - bh : base;
    return `<rect x="${(i * (bw + gap)).toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="2" fill="${ctxEsc(color)}"/>`;
  }).join("");
  return `<svg class="ca-minibars" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}">${rects}</svg>`;
}

function donut(f, { size = 78, stroke = 10, color = "var(--navy)" } = {}) {
  if (!uiFin(f)) return "";
  const r = (size - stroke) / 2, C = 2 * Math.PI * r, x = Math.max(0, Math.min(1, +f)), c = size / 2;
  const arc = x > 0 ? `<circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="${ctxEsc(color)}" stroke-width="${stroke}" stroke-dasharray="${(C * x).toFixed(2)} ${(C * (1 - x)).toFixed(2)}" transform="rotate(-90 ${c} ${c})" stroke-linecap="round"/>` : "";
  return `<svg class="ca-donut" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}"><circle cx="${c}" cy="${c}" r="${r}" fill="none" stroke="#E9E5DB" stroke-width="${stroke}"/>${arc}</svg>`;
}

function areaChart(points, { w = 1000, h = 260, yTicks = 5, color = "var(--green)", unit = "u" } = {}) {
  const pts = Array.isArray(points) ? points.filter((p) => p && uiFin(p.y)) : [];
  if (pts.length < 2) return "";
  const ys = pts.map((p) => +p.y), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), r = hi - lo || 1, n = pts.length;
  const pad = 36, right = 8, col = ctxEsc(color);
  const X = (i) => pad + i / (n - 1) * (w - pad - right), Y = (v) => 8 + (hi - v) / r * (h - 32);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(+p.y).toFixed(1)}`).join(" ");
  const tickSvg = uiTicks(lo, hi, yTicks).map((t) => `<line x1="${pad}" x2="${w}" y1="${Y(t.v).toFixed(1)}" y2="${Y(t.v).toFixed(1)}" class="ca-grid-line"/><text x="0" y="${(Y(t.v) + 4).toFixed(1)}" class="ca-axis">${t.text}${ctxEsc(unit)}</text>`).join("");
  const xEvery = Math.max(1, Math.ceil(n / 7));
  const xs = pts.map((p, i) => (i % xEvery === 0 || i === n - 1 ? `<text x="${X(i).toFixed(1)}" y="${h - 4}" text-anchor="${i === n - 1 ? "end" : "middle"}" class="ca-axis">${ctxEsc(p.x)}</text>` : "")).join("");
  return `<svg class="ca-area" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${tickSvg}<path d="${line} L${X(n - 1).toFixed(1)} ${Y(lo).toFixed(1)} L${X(0).toFixed(1)} ${Y(lo).toFixed(1)}Z" fill="${col}" opacity=".12"/><path d="${line}" fill="none" stroke="${col}" stroke-width="2.5" vector-effect="non-scaling-stroke"/><circle cx="${X(n - 1).toFixed(1)}" cy="${Y(ys[n - 1]).toFixed(1)}" r="5" fill="${col}"/>${xs}</svg>`;
}

function groupedBars(groups, { h = 180 } = {}) {
  const gs = (Array.isArray(groups) ? groups : []).map((g) => ({ label: g && g.label, bars: ((g && g.bars) || []).filter((b) => b && uiFin(b.value)) })).filter((g) => g.bars.length);
  if (!gs.length) return "";
  const all = gs.flatMap((g) => g.bars.map((b) => +b.value));
  const lo = Math.min(0, ...all), hi = Math.max(0, ...all), r = hi - lo || 1;
  const top = 20, bottom = lo < 0 ? 42 : 26, plot = h - top - bottom, base = top + hi / r * plot;
  const bw = 26, bgap = 6, ggap = 30;
  let x = ggap / 2, out = "";
  const widths = gs.map((g) => g.bars.length * bw + (g.bars.length - 1) * bgap);
  for (const [gi, g] of gs.entries()) {
    g.bars.forEach((b, i) => {
      const bx = x + i * (bw + bgap), hgt = +b.value / r * plot, text = b.label != null && b.label !== "" ? b.label : uiFmt(b.value);
      const ly = hgt >= 0 ? base - hgt - 5 : base - hgt + 13;
      out += `<path class="ca-bar" d="${uiBarPath(bx, base, hgt, bw)}" fill="${ctxEsc(b.color || "var(--green)")}"/><text class="ca-bar-val" x="${(bx + bw / 2).toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="middle">${ctxEsc(text)}</text>`;
    });
    if (g.label != null) out += `<text class="ca-axis" x="${(x + widths[gi] / 2).toFixed(1)}" y="${h - 8}" text-anchor="middle">${ctxEsc(g.label)}</text>`;
    x += widths[gi] + ggap;
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
    const x = i * slot + (slot - bw) / 2, hgt = +b.n / max * plot;
    return `<path class="ca-bar" d="${uiBarPath(x, base, hgt, bw)}" fill="${ctxEsc(b.color || "var(--green)")}"/><text class="ca-bar-val" x="${(x + bw / 2).toFixed(1)}" y="${(base - hgt - 5).toFixed(1)}" text-anchor="middle">${uiFmt(b.n)}</text>`
      + (i % every === 0 ? `<text class="ca-axis" x="${(x + bw / 2).toFixed(1)}" y="${h - 6}" text-anchor="middle">${ctxEsc(b.label)}</text>` : "");
  }).join("");
  return `<svg class="ca-hist" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><line x1="0" x2="${w}" y1="${base}" y2="${base}" class="ca-grid-line"/>${out}</svg>`;
}

function lineChart(series, xLabels, { w = 1000, h = 260, yTicks = 5, unit = "" } = {}) {
  const PALETTE = ["var(--green)", "var(--navy)", "var(--blue)", "var(--red)"];
  const ss = (Array.isArray(series) ? series : []).map((s, si) => ({ label: s && s.label, color: (s && s.color) || PALETTE[si % PALETTE.length], values: (s && Array.isArray(s.values)) ? s.values : [] }))
    .filter((s) => s.values.filter(uiFin).length >= 2);
  if (!ss.length) return "";
  const all = ss.flatMap((s) => uiNums(s.values)), lo = Math.min(...all), hi = Math.max(...all), r = hi - lo || 1;
  const n = Math.max(Array.isArray(xLabels) ? xLabels.length : 0, ...ss.map((s) => s.values.length));
  const pad = 36, right = 8;
  const X = (i) => pad + (n > 1 ? i / (n - 1) : 0) * (w - pad - right), Y = (v) => 8 + (hi - v) / r * (h - 32);
  const grid = uiTicks(lo, hi, yTicks).map((t) => `<line x1="${pad}" x2="${w}" y1="${Y(t.v).toFixed(1)}" y2="${Y(t.v).toFixed(1)}" class="ca-grid-line"/><text x="0" y="${(Y(t.v) + 4).toFixed(1)}" class="ca-axis">${t.text}${ctxEsc(unit)}</text>`).join("");
  const lines = ss.map((s) => {
    let d = "", pen = false;
    s.values.forEach((v, i) => { if (!uiFin(v)) { pen = false; return; } d += `${pen ? "L" : "M"}${X(i).toFixed(1)} ${Y(+v).toFixed(1)} `; pen = true; });
    return `<path d="${d.trim()}" fill="none" stroke="${ctxEsc(s.color)}" stroke-width="2.5" vector-effect="non-scaling-stroke" stroke-linecap="round" stroke-linejoin="round"/>`;
  }).join("");
  const xEvery = Math.max(1, Math.ceil(n / 7));
  const xs = (Array.isArray(xLabels) ? xLabels : []).map((l, i) => (l != null && (i % xEvery === 0 || i === n - 1) ? `<text x="${X(i).toFixed(1)}" y="${h - 4}" text-anchor="${i === n - 1 ? "end" : i === 0 ? "start" : "middle"}" class="ca-axis">${ctxEsc(l)}</text>` : "")).join("");
  const legend = ss.filter((s) => s.label != null).map((s) => `<span class="ca-legend-item"><i style="background:${ctxEsc(s.color)}"></i>${ctxEsc(s.label)}</span>`).join("");
  return `<div class="ca-linechart"><svg class="ca-area" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${grid}${lines}${xs}</svg>${legend ? `<div class="ca-legend">${legend}</div>` : ""}</div>`;
}

function donutLegend(rows) {
  const rs = (Array.isArray(rows) ? rows : []).filter((r) => r && r.label != null && r.label !== "");
  if (!rs.length) return "";
  return `<div class="ca-donut-legend">${rs.map((r) => `<div class="ca-dl-row"><i class="ca-dl-dot" style="background:${ctxEsc(r.color || "var(--navy)")}"></i><span class="ca-dl-label">${ctxEsc(r.label)}</span>${uiFin(r.pct) ? `<span class="ca-dl-pct">${Math.round(+r.pct)}%</span>` : ""}${r.value != null && r.value !== "" ? `<span class="ca-dl-val">${ctxEsc(r.value)}</span>` : ""}</div>`).join("")}</div>`;
}
