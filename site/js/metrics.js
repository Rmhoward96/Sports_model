/* Pure calculations shared by every page. No DOM, no fetch. */
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
function alphaScore({ evPct = 0, edgePp = 0, segRoiPct = 0, segN = 0 }) {
  const sEv = clip(evPct / 8, 0, 1), sEdge = clip(edgePp / 8, 0, 1);
  const roiS = segN > 0 ? segRoiPct * segN / (segN + 100) : 0;
  const sTrack = 0.5 + clip(roiS / 20, -0.5, 0.5);
  return Math.round(40 + 30 * sEv + 20 * sEdge + 10 * sTrack);
}
const alphaTier = (s) => (s >= 85 ? "HIGH" : s >= 75 ? "STRONG" : s >= 65 ? "MEDIUM" : null);
const unitsFromPnl = (pnl) => +pnl / 10;
function aggPnl(rows) {
  const a = (rows || []).reduce((t, r) => ({ n: t.n + +r.n, w: t.w + +r.wins, l: t.l + +r.losses, p: t.p + +r.pushes, pnl: t.pnl + +r.pnl }),
    { n: 0, w: 0, l: 0, p: 0, pnl: 0 });
  const units = unitsFromPnl(a.pnl);
  return { n: a.n, w: a.w, l: a.l, p: a.p, units, roiPct: a.n ? units / a.n * 100 : 0 };
}
const decided = (rs) => (rs || []).filter((r) => r.won === true || r.won === false);
function hitRate(rs) { const d = decided(rs); return d.length ? d.filter((r) => r.won).length / d.length : null; }
function hitRateVsMarket(rs) {
  const d = decided(rs).filter((r) => Number.isFinite(+r.implied));
  if (!d.length) return null;
  const hr = d.filter((r) => r.won).length / d.length, imp = d.reduce((s, r) => s + +r.implied, 0) / d.length;
  return (hr - imp) * 100;
}
function currentStreak(rs) {
  const d = decided(rs);
  if (!d.length) return null;
  const kind = d[0].won ? "W" : "L";
  let n = 0;
  for (const r of d) { if ((r.won ? "W" : "L") !== kind) break; n++; }
  return { kind, n };
}
function recentForm(rs, n) {
  const s = (rs || []).slice(0, n);
  const w = s.filter((r) => r.won === true).length, l = s.filter((r) => r.won === false).length;
  return { w, l, p: s.length - w - l, pct: w + l ? w / (w + l) : null };
}
function calibration(rows, bands) {
  const fav = (rows || []).filter((r) => Number.isFinite(+r.prob) && (r.won === true || r.won === false))
    .map((r) => (+r.prob >= 0.5 ? { p: +r.prob, won: r.won } : { p: 1 - +r.prob, won: !r.won }));
  return bands.map(([lo, hi]) => {
    const b = fav.filter((r) => r.p >= lo && r.p < hi);
    return { band: [lo, hi], n: b.length, predicted: b.length ? b.reduce((s, r) => s + r.p, 0) / b.length : null,
      actual: b.length ? b.filter((r) => r.won).length / b.length : null };
  });
}
function brier(rows) {
  const d = (rows || []).filter((r) => Number.isFinite(+r.prob) && (r.won === true || r.won === false));
  return d.length ? d.reduce((s, r) => s + (+r.prob - (r.won ? 1 : 0)) ** 2, 0) / d.length : null;
}
function edgeBuckets(rows, edges) {
  const [e0, ...rest] = edges;               // e.g. [10, 5, 2, 0]
  const specs = [{ label: `> ${e0}%`, test: (x) => x > e0 }];
  [e0, ...rest].forEach((hi, i, a) => { if (i < a.length - 1) { const lo = a[i + 1]; specs.push({ label: `${lo}% to ${hi}%`, test: (x) => x > lo && x <= hi }); } });
  specs.push({ label: `< ${edges.at(-1)}%`, test: (x) => x <= edges.at(-1) });
  return specs.map(({ label, test }) => {
    const b = (rows || []).filter((r) => Number.isFinite(+r.edgePct) && test(+r.edgePct));
    const units = b.reduce((s, r) => s + (+r.profitUnits || 0), 0);
    return { label, n: b.length, hits: b.filter((r) => r.won === true).length, units, roiPct: b.length ? units / b.length * 100 : null };
  });
}
function histogram(values, edges) {
  return edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1],
    n: (values || []).filter((v) => Number.isFinite(v) && (v >= lo && (i === edges.length - 2 ? v <= edges[i + 1] : v < edges[i + 1]))).length }));
}
function dailyCounts(rows, dateOf, days, today) {
  const counts = new Map();
  (rows || []).forEach((r) => { const d = dateOf(r); counts.set(d, (counts.get(d) || 0) + 1); });
  const out = [], t = new Date(`${today}T12:00:00Z`);
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(t.getTime() - i * 864e5).toISOString().slice(0, 10);
    out.push({ date: d, n: counts.get(d) || 0 });
  }
  return out;
}
const americanToProb = (o) => {
  if (o == null || o === "") return NaN;
  const n = +o;
  if (!Number.isFinite(n)) return NaN;
  return n > 0 ? 100 / (n + 100) : -n / (-n + 100);
};
const probToAmerican = (p) => (p >= 0.5 ? -Math.round(p / (1 - p) * 100) : Math.round((1 - p) / p * 100));
function lineMoveScore(m) {
  if (m.market === "moneyline") {
    if (m.open_price == null || m.cur_price == null) return null;
    const open = +m.open_price, cur = +m.cur_price;
    if (!Number.isFinite(open) || !Number.isFinite(cur)) return null;
    return Math.abs(americanToProb(cur) - americanToProb(open)) * 100 / 2.5;
  }
  if (m.open_line == null || m.cur_line == null) return null;
  const open = +m.open_line, cur = +m.cur_line;
  if (!Number.isFinite(open) || !Number.isFinite(cur)) return null;
  return Math.abs(cur - open);
}
