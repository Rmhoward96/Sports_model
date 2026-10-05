/* Game page (game.html?sport=<s>&game=<game_pk>) — mockup #2 "Matchup Story". Real data only: every card has an empty
   state and the mockup's numbers are never rendered. The tab (Overview / Matchup / Market / Trends / Players) lives in
   window.__caGame AND in the URL (?tab=) and survives the 5-minute re-render; switching tabs redraws from the cached load
   (window.__caGameData) without a refetch. Phase B items (venue / weather line, Projected Game Flow, TV network) are omitted.
   "Opportunities" are is_pick rows with a non-null Alpha tier (loadOpportunities); the Read card reads only the game's pick
   rows, because a non-pick row's true_prob is the sharp line, not the model.
   Depends on app.js (sb, gameParams, predictions-row helpers, the legacy sections matchupSection / historySection /
   trendsSection / splitsSection / propsProjectionSection / gameSimVisual / nflPredictionSection / boxscoreSection / evSection,
   wireGameSim, ctxEsc, ctxOrd, logoImg, teamShort, timeET, evBookName, gameMoneylines, evBestLines, nflServedSimVersion,
   simVersionFilter, dedupLatest, distParse, distProbs, mlModelTag, gameTeamColors, pct1, pStr, r05), metrics.js, ui.js,
   shell.js and data.js. */
const GM_TABS = [["overview", "Overview"], ["matchup", "Matchup"], ["market", "Market"], ["trends", "Trends"], ["players", "Players"]];
const ICON_GRADE = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>`;
const ICON_RANK = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 20V11M12 20V4M19 20v-7"/></svg>`;

/* ── pure helpers ─────────────────────────────────────────────────────── */
// The strongest model divergence among candidate sides [{market, label, modelProb, impliedProb, ...}]: the largest
// positive (modelProb - impliedProb), in percentage points. null when there is none. The candidate's fields ride along.
function gameRead(markets) {
  let best = null;
  for (const m of markets || []) {
    if (!m || !finite(m.modelProb) || !finite(m.impliedProb)) continue;
    const edgePp = (+m.modelProb - +m.impliedProb) * 100;
    if (edgePp > 0 && (!best || edgePp > best.edgePp)) best = { ...m, edgePp };
  }
  return best;
}

// The side the model favors in each market: moneyline favorite; spread / total only when the model's half-point
// line is off the market line (same rule as the legacy lean); null when unknowable or on the line.
function gmLean(pred) {
  const p = pred || {}, wp = numOrNull(p.home_win_prob), h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score);
  const ls = numOrNull(p.market_spread), lt = numOrNull(p.market_total);
  const out = { ml: wp == null ? null : wp >= 0.5 ? "home" : "away", spread: null, total: null };
  if (ls != null && h != null && a != null) {
    const edge = r05(h - a) + ls;
    out.spread = Math.abs(edge) < 0.25 ? null : edge > 0 ? "home" : "away";
  }
  if (lt != null && h != null && a != null) out.total = h + a > lt ? "over" : h + a < lt ? "under" : null;
  return out;
}

// "Mon, Oct 12 · 7:00 PM ET" (Eastern); "" for a missing / invalid time.
function gmWhen(iso) {
  if (!iso || !Number.isFinite(Date.parse(iso))) return "";
  return `${new Date(iso).toLocaleDateString("en-US", { timeZone: "America/New_York", weekday: "short", month: "short", day: "numeric" })} · ${timeET(iso)}`;
}

// Market consensus: the median implied probability across books in a {book: american} map (object or JSON string),
// back as an American price. Junk and |price| < 100 are skipped; null when no book is usable.
function consensusAmerican(prices) {
  let map = prices;
  if (typeof map === "string") { try { map = JSON.parse(map); } catch { map = null; } }
  if (!map || typeof map !== "object") return null;
  const ps = Object.values(map).map(numOrNull).filter((a) => a != null && Math.abs(a) >= 100).map(americanToProb).sort((x, y) => x - y);
  if (!ps.length) return null;
  const mid = ps.length >> 1, p = ps.length % 2 ? ps[mid] : (ps[mid - 1] + ps[mid]) / 2;
  return probToAmerican(p);
}

// Latest build of each market/side from ev_picks rows (newest created_at wins; ties keep the first).
function gmLatest(evRows) {
  const by = new Map();
  for (const r of evRows || []) {
    if (!r) continue;
    const k = `${r.market}|${r.side}`, prev = by.get(k);
    if (!prev || String(r.created_at || "") > String(prev.created_at || "")) by.set(k, r);
  }
  return [...by.values()];
}
// The market line a side is bet at: the best book's line (ev_best_lines) else the posted market line. Spread lines
// are from the side's own view (away = -home). null when unknown.
function gmSideLine(pred, row, lineBy) {
  const l = lineBy && lineBy.get(`${pred.game_pk}|${row.market}|${row.side}`);
  if (finite(l)) return +l;
  if (row.market === "total") return numOrNull(pred.market_total);
  const ls = numOrNull(pred.market_spread);
  return ls == null ? null : row.side === "home" ? ls : -ls;
}
const gmTeam = (pred, side) => shortTeam(side === "home" ? pred.home_team_name : pred.away_team_name, pred.sport);
function gmSideLabel(pred, row, line) {
  if (row.market === "total") return `${row.side === "under" ? "Under" : "Over"} ${line != null ? line : "total"}`;
  const t = gmTeam(pred, row.side);
  if (row.market === "moneyline") return `${t} ML`;
  return `${t} ${line != null ? lineStr(line) : "spread"}`;
}
const gmImplied = (r) => (finite(r.best_line_implied) ? +r.best_line_implied : (Number.isFinite(americanToProb(r.best_price)) ? americanToProb(r.best_price) : null));

// Candidate reads: this game's current PICK rows (is_pick, finite model + market probabilities).
function gmMarkets(pred, evRows, lineBy) {
  return gmLatest(evRows).filter((r) => r.is_pick === true && finite(r.true_prob) && gmImplied(r) != null).map((r) => {
    const line = r.market === "moneyline" ? null : gmSideLine(pred, r, lineBy);
    return { market: r.market, side: r.side, label: gmSideLabel(pred, r, line), line, price: numOrNull(r.best_price), book: r.best_book || null,
      modelProb: +r.true_prob, impliedProb: gmImplied(r) };
  });
}

// P(home covers) / P(away covers) at the market spread and P(over) / P(under) at the market total from the NFL sim's
// distributions, pushes excluded. null parts when there is no distribution or no line.
function gmSimProbs(pred, sim) {
  const out = { home: null, away: null, over: null, under: null };
  if (!sim) return out;
  const split = (dist, line) => {
    const d = distParse(dist);
    if (!d || line == null) return null;
    const { under, over } = distProbs(d, line), s = under + over;
    return s > 0 ? { over: over / s, under: under / s } : null;
  };
  const ls = numOrNull(pred.market_spread), lt = numOrNull(pred.market_total);
  const sp = split(sim.margin_dist, ls == null ? null : -ls), tt = split(sim.total_dist, lt);
  if (sp) { out.home = sp.over; out.away = sp.under; }
  if (tt) { out.over = tt.over; out.under = tt.under; }
  return out;
}

// Per-market view of the game's odds for the Overview: consensus moneyline prices, the market spread / total line and,
// per side, the implied probability (best price) and the model probability (a pick row's true_prob, else the NFL sim).
function gmOdds(pred, evRows, mlRow, sim) {
  const p = pred || {}, rows = gmLatest(evRows), by = new Map(rows.map((r) => [`${r.market}|${r.side}`, r])), sp = gmSimProbs(p, sim);
  const implied = (r) => (r ? gmImplied(r) : null);
  const pickP = (r) => (r && r.is_pick === true && finite(r.true_prob) ? +r.true_prob : null);
  const mlSide = (side) => {
    const r = by.get(`moneyline|${side}`), cons = mlRow ? consensusAmerican(mlRow[`${side}_prices`]) : null;
    const price = cons != null ? cons : r ? numOrNull(r.best_price) : null;
    const imp = price != null ? americanToProb(price) : null;
    return { price, implied: imp != null && Number.isFinite(imp) ? imp : null };
  };
  // A two-way market: a pick row's probability is the model's; the other side is its complement unless that side is a pick
  // too. With no pick row the NFL sim supplies both sides. Otherwise the model has no number (a non-pick true_prob is the sharp line).
  const twoWay = (market, a, b, simA, simB) => {
    const ra = by.get(`${market}|${a}`), rb = by.get(`${market}|${b}`), pa = pickP(ra), pb = pickP(rb);
    const [ma, mb] = pa != null || pb != null ? [pa != null ? pa : 1 - pb, pb != null ? pb : 1 - pa] : [simA, simB];
    const one = (r, m) => ({ implied: implied(r), modelProb: m, price: r ? numOrNull(r.best_price) : null });
    return { [a]: one(ra, ma), [b]: one(rb, mb) };
  };
  return {
    moneyline: { away: mlSide("away"), home: mlSide("home") },
    spread: { line: numOrNull(p.market_spread), ...twoWay("spread", "away", "home", sp.away, sp.home) },
    total: { line: numOrNull(p.market_total), ...twoWay("total", "over", "under", sp.over, sp.under) },
  };
}

// Model cover probability for each side of the spread; one side known -> the other is its complement.
function gmCover(odds) {
  const s = odds && odds.spread;
  if (!s) return null;
  let home = s.home.modelProb, away = s.away.modelProb;
  if (home == null && away == null) return null;
  if (home == null) home = 1 - away;
  if (away == null) away = 1 - home;
  return { home, away };
}

const gmPct = (x) => `${(x * 100).toFixed(1)}%`;
const gmLine1 = (x) => (Math.abs(x) < 0.05 ? "PK" : x > 0 ? `+${x.toFixed(1)}` : x.toFixed(1));
const gmPctP = (p) => (p != null ? ` (${gmPct(p)})` : "");

// Model Projection table: Market / CappingAlpha / Difference for moneyline (the model's favorite), spread (the model's
// side) and total (the model's lean). diff = model minus market implied probability, in points (null when either is
// unknown); pts = how many more points the model asks for than the market, from the same side (null when unknown).
function modelProjectionRows(pred, odds) {
  const p = pred || {}, o = odds || {}, sport = p.sport;
  const wp = numOrNull(p.home_win_prob), h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score);
  const blank = (label) => ({ label, market: "—", model: "—", diff: null, pts: null });
  const ab = (side) => teamShort(side === "home" ? p.home_team_name : p.away_team_name, sport);
  const rows = [];
  // moneyline
  const ml = blank("Moneyline");
  if (wp != null) {
    const side = wp >= 0.5 ? "home" : "away", prob = Math.max(wp, 1 - wp), e = o.moneyline && o.moneyline[side];
    ml.model = gmPct(prob);
    if (e && e.price != null) { ml.market = `${ab(side)} ${oddsStr(e.price)}${gmPctP(e.implied)}`; if (e.implied != null) ml.diff = (prob - e.implied) * 100; }
  }
  rows.push(ml);
  // spread
  const sp = blank("Spread");
  if (h != null && a != null) {
    const margin = h - a, side = margin >= 0 ? "home" : "away", modelLine = side === "home" ? -margin : margin;
    const e = o.spread && o.spread[side], L = o.spread ? o.spread.line : null;
    sp.model = `${ab(side)} ${gmLine1(modelLine)}${gmPctP(e ? e.modelProb : null)}`;
    if (L != null) {
      const ml1 = side === "home" ? L : -L;
      sp.market = `${ab(side)} ${lineStr(ml1)}${gmPctP(e ? e.implied : null)}`;
      sp.pts = ml1 - modelLine;
    }
    if (e && e.modelProb != null && e.implied != null) sp.diff = (e.modelProb - e.implied) * 100;
  }
  rows.push(sp);
  // total
  const tt = blank("Total");
  if (h != null && a != null) {
    const model = h + a, T = o.total ? o.total.line : null, lean = T == null ? null : model >= T ? "over" : "under";
    const e = lean && o.total[lean];
    tt.model = `${model.toFixed(1)}${gmPctP(e ? e.modelProb : null)}`;
    if (T != null) {
      tt.market = `${lean === "under" ? "U" : "O"} ${T}${gmPctP(e ? e.implied : null)}`;
      tt.pts = lean === "over" ? model - T : T - model;
    }
    if (e && e.modelProb != null && e.implied != null) tt.diff = (e.modelProb - e.implied) * 100;
  }
  rows.push(tt);
  return rows;
}

// Season straight-up record "W-L" (team_history season window, else power_rankings). No conference record exists in the
// data, so none is ever shown.
function gmRecord(hist, power, side) {
  const h = (hist || []).find((x) => x.side === side);
  const w = h && ctxJson(h.windows), s = w && w.season;
  if (s && +s.n > 0 && s.su) return String(s.su);
  const pr = h && (power || []).find((x) => String(x.team) === String(h.team));
  return pr && pr.su ? String(pr.su) : "";
}

// Up to four generated sentences from data that exists: offense-vs-defense unit grade, explosive-play edge, power-rank
// gap, current streak. Each is {kind, title, body} (plain text; escaped when rendered).
const gmPoss = (n) => (/s$/i.test(n) ? `${n}'` : `${n}'s`);
const GM_STREAK = { su: { W: "won", L: "lost" }, ats: { W: "covered", L: "failed to cover" }, ou: { O: "has gone over in", U: "has gone under in" } };
function gmKeyInsights(D) {
  const r = (D && D.r) || {}, grades = (D && D.ctxGrades) || [], hist = (D && D.ctxHist) || [], power = (D && D.ctxPower) || [];
  const name = (side) => shortTeam(side === "home" ? r.home_team_name : r.away_team_name, D && D.sport);
  const other = (s) => (s === "home" ? "away" : "home");
  const out = [];
  const graded = grades.filter((g) => g && g.overall && finite(g.overall_pct)).sort((x, y) => y.overall_pct - x.overall_pct)[0];
  if (graded) out.push({ kind: "grade", title: `${name(graded.side)} offense grades ${graded.overall} vs ${name(other(graded.side))}`,
    body: `Pass ${graded.pass || "–"}, run ${graded.run || "–"} against ${gmPoss(name(other(graded.side)))} defense (${ctxOrd(graded.overall_pct)} percentile overall).` });
  const xpl = (g) => {
    const u = g && ctxJson(g.units), o = u && u.off, d = u && u.def;
    if (!o || !d) return null;
    const v = [o.pass_explosive, d.pass_explosive, o.run_explosive, d.run_explosive].map(numOrNull);
    if (v.some((x) => x == null)) return null;
    const pr = numOrNull(u.pass_rate) ?? 0.5;
    return pr * (v[0] + v[1]) + (1 - pr) * (v[2] + v[3]);
  };
  const xh = xpl(grades.find((g) => g.side === "home")), xa = xpl(grades.find((g) => g.side === "away"));
  if (xh != null && xa != null && Math.abs(xh - xa) > 1e-9) {
    const w = xh > xa ? "home" : "away";
    out.push({ kind: "explosive", title: `Explosive plays favor ${name(w)}`,
      body: `${gmPoss(name(w))} offense vs ${gmPoss(name(other(w)))} defense grades out ahead on big plays, pass and run, compared with the reverse matchup.` });
  }
  const code = (side) => { const x = hist.find((y) => y.side === side) || grades.find((y) => y.side === side); return x ? String(x.team) : null; };
  const pr = (side) => power.find((x) => String(x.team) === code(side));
  const ph = pr("home"), pa = pr("away");
  if (ph && pa && finite(ph.rank) && finite(pa.rank) && ph.rank !== pa.rank) {
    const [bs, b, w] = ph.rank < pa.rank ? ["home", ph, pa] : ["away", pa, ph];
    const gap = finite(b.rating) && finite(w.rating) ? ` Rating gap: ${(b.rating - w.rating).toFixed(1)} pts.` : "";
    out.push({ kind: "power", title: `${name(bs)} ranks #${b.rank} in the power rankings`, body: `${name(other(bs))} ranks #${w.rank}.${gap}` });
  }
  let st = null;
  for (const h of hist) {
    const S = ctxJson(h.streaks) || {};
    for (const k of ["su", "ats", "ou"]) {
      const m = /^([A-Z])(\d+)$/.exec(String(S[k] || ""));
      if (m && GM_STREAK[k][m[1]] && +m[2] >= 3 && (!st || +m[2] > st.n)) st = { side: h.side, k, kind: m[1], n: +m[2] };
    }
  }
  if (st) out.push({ kind: "streak", title: `${name(st.side)} ${GM_STREAK[st.k][st.kind]} ${st.n} straight`,
    body: `Current ${{ su: "straight-up", ats: "against-the-spread", ou: "over/under" }[st.k]} streak entering this game.` });
  return out.slice(0, 4);
}

// Best price per side: moneyline from the moneylines view (else the pick builds), spread / total from the latest builds.
function gmBestBooks(pred, mlRow, evRows) {
  const rows = gmLatest(evRows), out = [];
  for (const side of ["away", "home"]) {
    const ev = rows.find((r) => r.market === "moneyline" && r.side === side);
    const price = mlRow && finite(mlRow[`${side}_price`]) ? +mlRow[`${side}_price`] : ev ? numOrNull(ev.best_price) : null;
    const book = mlRow && finite(mlRow[`${side}_price`]) ? mlRow[`${side}_book`] : ev ? ev.best_book : null;
    if (price != null) out.push({ label: `${gmTeam(pred, side)} ML`, price, book });
  }
  for (const r of rows) {
    if (r.market === "moneyline" || numOrNull(r.best_price) == null) continue;
    out.push({ label: gmSideLabel(pred, r, gmSideLine(pred, r, null)), price: +r.best_price, book: r.best_book || null });
  }
  return out;
}

const gmSplitMap = (splits) => {
  const m = new Map();
  (splits || []).forEach((r) => { const k = `${r.game_pk}|${r.market}|${r.side}`, prev = m.get(k); if (!prev || String(r.captured_at) > String(prev.captured_at)) m.set(k, r); });
  return m;
};

/* ── state <-> URL ────────────────────────────────────────────────────── */
function gmParseTab(search) {
  let t = "";
  try { t = new URLSearchParams(search || "").get("tab") || ""; } catch { t = ""; }
  return GM_TABS.some(([k]) => k === t) ? t : "overview";
}
function gmState() {
  if (!window.__caGame) { let s = ""; try { s = location.search; } catch { s = ""; } window.__caGame = { tab: gmParseTab(s) }; }
  return window.__caGame;
}
function gmSetTab(tab) {
  const s = gmState();
  s.tab = GM_TABS.some(([k]) => k === tab) ? tab : "overview";
  try {
    const u = new URL(location.href);
    if (s.tab === "overview") u.searchParams.delete("tab"); else u.searchParams.set("tab", s.tab);
    history.replaceState(null, "", u.toString());
  } catch { /* keep the current URL */ }
}

/* ── data ─────────────────────────────────────────────────────────────── */
// Reads by game_pk from the BASE views (predictions_any has no date floor), so a detail page keeps its projections after
// kickoff; prediction_accuracy carries the graded final. NFL adds the served sim version's rows; NFL + CFB add splits,
// betting trends and team context. Every read is caught -> [] and its card hides or shows its empty state.
async function gmLoad(sport, rawGame) {
  const game = encodeURIComponent(rawGame);
  const isNfl = sport === "nfl", live = isNfl || sport === "cfb", none = (v) => Promise.resolve(v);
  const [predsAny, predsCur, evRows, accRows, servedVersion, mls, opps, moves, lineBy] = await Promise.all([
    sb(`predictions_any?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    sb(`predictions_current?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    live ? sb(`ev_picks?sport=eq.${sport}&game_pk=eq.${game}&order=created_at.desc`).catch(() => []) : none([]),
    sb(`prediction_accuracy?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    isNfl ? nflServedSimVersion() : none(null),
    live ? gameMoneylines().catch(() => new Map()) : none(new Map()),
    live ? loadOpportunities().catch(() => []) : none([]),
    live ? loadLineMoves().catch(() => []) : none([]),
    live ? evBestLines().catch(() => new Map()) : none(new Map()),
  ]);
  const r = predsAny[0] || predsCur[0];
  if (!r) return null;
  r.sport = sport;
  let sims = [], props = [], simRows = [], playerActuals = [], propLines = [], splits = [], trendRecs = [], trendSits = [], ctxHist = [], ctxGrades = [], ctxPower = [];
  if (isNfl) {
    // Served-version filter; a failed served read falls back to the newest rows of any version, and so does an empty strict read
    // (a pre-switch game whose rows exist only under an earlier version).
    const mv = simVersionFilter(servedVersion);
    const simRead = async (table) => {
      const rows = await sb(`${table}?game_pk=eq.${game}${mv}&order=created_at.desc`).catch(() => []);
      if (rows.length || !mv) return rows;
      return sb(`${table}?game_pk=eq.${game}&order=created_at.desc`).catch(() => []);
    };
    [sims, props, simRows, playerActuals, propLines] = await Promise.all([
      simRead("nfl_player_sim").then(dedupLatest),
      sb(`ev_prop_picks?sport=eq.nfl&game_pk=eq.${game}`).catch(() => []),
      simRead("nfl_sim"),
      sb(`nfl_player_actuals?game_pk=eq.${game}`).catch(() => []),
      sb(`nfl_prop_lines?game_pk=eq.${game}`).catch(() => []),
    ]);
  }
  if (live) {
    const q = (n) => `"${encodeURIComponent(n)}"`;
    [splits, trendRecs, trendSits, ctxHist, ctxGrades] = await Promise.all([
      sb(`${sport}_betting_splits_current?game_pk=eq.${game}`).catch(() => []),
      sb(`team_betting_records?sport=eq.${sport}&team_name=in.(${q(r.away_team_name)},${q(r.home_team_name)})&order=season.desc`).catch(() => []),
      isNfl ? sb(`nfl_game_trends?game_pk=eq.${game}`).catch(() => []) : none([]),
      sb(`team_history?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
      sb(`matchup_grades?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    ]);
    // current power rank of both teams; the pooled FCS team is never ranked
    const codes = [...new Set([...ctxHist, ...ctxGrades].map((x) => x.team).filter((t) => t && t !== "FCS"))];
    if (ctxHist.length && codes.length)
      ctxPower = await sb(`power_rankings_current?sport=eq.${sport}&team=in.(${codes.map((c) => q(String(c))).join(",")})`).catch(() => []);
  }
  // Player pmf map for the interactive per-player distribution panel, plus the team colors the sim charts share.
  const players = {};
  (sims || []).forEach((s) => {
    if (!players[s.player_id]) players[s.player_id] = { name: s.name, pos: s.pos, team: s.team, markets: {} };
    const d = distParse(s.dist);
    if (d) { d.__mean = s.mean; players[s.player_id].markets[s.market] = d; }
  });
  const { away: awayCol, home: homeCol } = gameTeamColors(r.away_team_name, r.home_team_name, sport);
  window.__caGameSim = { players, awayCol, homeCol, awayTeam: r.away_team_name, homeTeam: r.home_team_name };
  const myOpps = (opps || []).filter((o) => String(o.game_pk) === String(game) && o.tier);
  const myMoves = (moves || []).filter((m) => String(m.game_pk) === String(game));
  const shownSimVersion = (sims[0] && sims[0].model_version) || null;
  return { sport, game: String(rawGame), r, actual: accRows[0] || null, evRows: evRows || [], mlRow: mls.get(String(game)) || null, opps: myOpps, moves: myMoves, lineBy: lineBy || new Map(),
    splits: splits || [], sims, props, simRows, playerActuals, propLines, trendRecs, trendSits, ctxHist, ctxGrades, ctxPower, awayCol, homeCol,
    isNfl, simTag: isNfl ? mlModelTag(shownSimVersion) : "" };
}

/* ── hero + read ──────────────────────────────────────────────────────── */
const gmEsc = ctxEsc;
const gmName = (D, side) => { const n = side === "home" ? D.r.home_team_name : D.r.away_team_name; return D.sport === "cfb" ? shortTeam(n, "cfb") : n; };

function gmFinalLine(D) {
  const a = D.actual, r = D.r;
  if (!a || a.actual_total == null || a.actual_margin == null) return "";
  const ah = Math.round((+a.actual_total + +a.actual_margin) / 2), aa = Math.round((+a.actual_total - +a.actual_margin) / 2);
  if (![ah, aa].every(Number.isFinite)) return "";
  const call = a.winner_correct == null ? "" : ` · <span class="${a.winner_correct ? "ca-gm-ok" : "ca-gm-miss"}">model ${a.winner_correct ? "called it ✓" : "missed ✗"}</span>`;
  return `<p class="ca-gm-final">Final · ${gmEsc(r.away_team_name)} ${aa} – ${ah} ${gmEsc(r.home_team_name)}${call}</p>`;
}
function gmOddsBox(title, left, right, leanSide) {
  const cell = (v, side) => `<span class="ca-gm-v${leanSide === side ? " lean" : ""}">${gmEsc(v)}</span>`;
  return `<div class="ca-gm-ob"><div class="ca-gm-ob-vals">${cell(left, "l")}${cell(right, "r")}</div><div class="ca-gm-ob-t">${title}</div></div>`;
}
function gmHero(D) {
  const r = D.r, sport = D.sport, o = gmOdds(r, D.evRows, D.mlRow, null), lean = gmLean(r), live = sport === "nfl" || sport === "cfb";
  const when = gmWhen(r.commence_time);
  const team = (side) => {
    const rec = gmRecord(D.ctxHist, D.ctxPower, side), nm = side === "home" ? r.home_team_name : r.away_team_name;
    return `<div class="ca-gm-team">${logoImg(nm, sport)}<h1>${gmEsc(gmName(D, side))}</h1>${rec ? `<p class="ca-gm-rec">${gmEsc(rec)}</p>` : ""}</div>`;
  };
  const price = (x) => (x != null ? oddsStr(x) : "—");
  const L = o.spread.line, T = o.total.line;
  const boxes = live ? `<div class="ca-gm-odds">
    ${gmOddsBox("MONEYLINE", price(o.moneyline.away.price), price(o.moneyline.home.price), lean.ml === "away" ? "l" : lean.ml === "home" ? "r" : "")}
    ${gmOddsBox("SPREAD", L != null ? lineStr(-L) : "—", L != null ? lineStr(L) : "—", lean.spread === "away" ? "l" : lean.spread === "home" ? "r" : "")}
    ${gmOddsBox("TOTAL", T != null ? `O ${T}` : "—", T != null ? `U ${T}` : "—", lean.total === "over" ? "l" : lean.total === "under" ? "r" : "")}</div>` : "";
  return `<section class="ca-gm-hero"><div class="ca-gm-top"><a class="ca-gm-back" href="${sport}.html">‹ ${SPORT_NAME[sport] || sport.toUpperCase()} Board</a><span class="ca-gm-when">${gmEsc(when)}</span><span></span></div>
    <div class="ca-gm-teams">${team("away")}<span class="ca-gm-at">@</span>${team("home")}</div>${gmFinalLine(D)}${boxes}</section>`;
}

function gmReadCard(D) {
  const m = gameRead(gmMarkets(D.r, D.evRows, D.lineBy));
  const noun = m && { moneyline: "win", spread: "cover", total: "hit" }[m.market];
  const body = m
    ? `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>${gmEsc(m.label)} shows the strongest model divergence</h2>
        <p>The market implies a ${gmPct(m.impliedProb)} ${noun} probability. CappingAlpha estimates <b>${gmPct(m.modelProb)}</b>.</p></div>
       <div class="ca-gm-edge"><b>${pStr(m.edgePp)}</b><span>MODEL EDGE</span></div>`
    : `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>No model edge on this game</h2>
        <p>No game-line market prices above the model's estimate right now.</p></div>`;
  return `<section class="ca-card ca-gm-read" id="gm-read"><div class="ca-gm-mark" aria-hidden="true">C<i>α</i></div>${body}</section>`;
}

/* ── Overview ─────────────────────────────────────────────────────────── */
function gmBestOpp(D) { return (D.opps || [])[0] || null; }
function gmAlphaCard(D) {
  const o = gmBestOpp(D);
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Alpha Score</div>${o
    ? `<div class="ca-gm-big">${gmEsc(o.alpha)}<small> / 100</small></div>${confPill(o.tier)}<div class="ca-gm-sub ca-ell" title="${gmEsc(pickLabel(o, D.lineBy))}">${gmEsc(pickLabel(o, D.lineBy))}</div>`
    : `<div class="ca-gm-big">—</div><div class="ca-gm-sub">No graded opportunity on this game.</div>`}</div>`;
}
function gmWinCard(D) {
  const wp = numOrNull(D.r.home_win_prob), row = (name, p) => `<div class="ca-gm-wp">${logoImg(name, D.sport)}<b>${p != null ? gmPct(p) : "—"}</b></div>`;
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Win Probability</div>${row(D.r.home_team_name, wp)}${row(D.r.away_team_name, wp != null ? 1 - wp : null)}</div>`;
}
function gmScoreCard(D) {
  const h = numOrNull(D.r.pred_home_score), a = numOrNull(D.r.pred_away_score);
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Score</div>${h != null && a != null
    ? `<div class="ca-gm-score">${logoImg(D.r.away_team_name, D.sport)}<b>${Math.round(a)}</b><span>-</span><b>${Math.round(h)}</b>${logoImg(D.r.home_team_name, D.sport)}</div>`
    : `<div class="ca-gm-big">—</div>`}</div>`;
}
function gmSpreadCard(D) {
  const r = D.r, h = numOrNull(r.pred_home_score), a = numOrNull(r.pred_away_score), L = numOrNull(r.market_spread);
  if (h == null || a == null) return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Spread</div><div class="ca-gm-big">—</div></div>`;
  const margin = h - a, side = margin >= 0 ? "home" : "away", nm = teamShort(side === "home" ? r.home_team_name : r.away_team_name, D.sport);
  const mk = L == null ? "" : `<div class="ca-gm-sub">Market: ${lineStr(side === "home" ? L : -L)}</div>`;
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Spread</div><div class="ca-gm-mid">${gmEsc(nm)} ${gmEsc(gmLine1(side === "home" ? -margin : margin))}</div>${mk}</div>`;
}
function gmTotalCard(D) {
  const r = D.r, h = numOrNull(r.pred_home_score), a = numOrNull(r.pred_away_score), T = numOrNull(r.market_total);
  if (h == null || a == null) return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Total</div><div class="ca-gm-big">—</div></div>`;
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Total</div><div class="ca-gm-mid">${(h + a).toFixed(1)}</div>${T != null ? `<div class="ca-gm-sub">Market: ${T}</div>` : ""}</div>`;
}
const GM_OVERVIEW = [["Alpha Score", gmAlphaCard], ["Win Probability", gmWinCard], ["Projected Score", gmScoreCard], ["Projected Spread", gmSpreadCard], ["Projected Total", gmTotalCard]];

function gmProjectionCard(D) {
  const rows = modelProjectionRows(D.r, gmOdds(D.r, D.evRows, D.mlRow, D.simRows[0]));
  const body = rows.map((x) => {
    const d = x.diff != null ? `<td class="${signCls(x.diff)} ca-b">${pStr(x.diff)}</td>` : x.pts != null ? `<td class="${signCls(x.pts)} ca-b">${signedStr(x.pts, 1, " pts")}</td>` : `<td class="muted">—</td>`;
    return `<tr><td>${gmEsc(x.label)}</td><td>${gmEsc(x.market)}</td><td class="ca-b">${gmEsc(x.model)}</td>${d}</tr>`;
  }).join("");
  return `<section class="ca-card ca-gm-proj" id="gm-projection"><div class="ca-card-head"><h2>Model Projection</h2></div><table class="ca-table ca-gm-table"><thead><tr><th></th><th>Market</th><th>CappingAlpha</th><th>Difference</th></tr></thead><tbody>${body}</tbody></table></section>`;
}
const GM_INSIGHT_ICON = { grade: ["ca-gm-ic-red", ICON_GRADE], explosive: ["ca-gm-ic-amber", ICON_TREND], power: ["ca-gm-ic-blue", ICON_RANK], streak: ["ca-gm-ic-green", ICON_WAVE] };
function gmInsightsCard(D) {
  const ins = gmKeyInsights(D);
  if (!ins.length) return "";
  const rows = ins.map((i) => `<div class="ca-gm-ins"><span class="ca-gm-ic ${GM_INSIGHT_ICON[i.kind][0]}">${GM_INSIGHT_ICON[i.kind][1]}</span><div><b>${gmEsc(i.title)}</b><p>${gmEsc(i.body)}</p></div></div>`).join("");
  return `<section class="ca-card ca-gm-ins-card" id="gm-insights"><div class="ca-card-head"><h2>Key Insights</h2></div>${rows}</section>`;
}
function gmCoverCard(D) {
  const c = gmCover(gmOdds(D.r, D.evRows, D.mlRow, D.simRows[0]));
  let body;
  if (!c) body = emptyMsg("Model cover probability isn't available for this game yet.");
  else {
    const away = Math.max(0, Math.min(100, c.away / (c.away + c.home) * 100));
    const money = gmSplitMap(D.splits), ma = money.get(`${D.game}|spread|away`), mh = money.get(`${D.game}|spread|home`);
    const cap = ma && mh && finite(ma.cash_pct) && finite(mh.cash_pct)
      ? `<p class="ca-gm-cap">% of Money · ${gmEsc(teamShort(D.r.away_team_name, D.sport))} ${Math.round(ma.cash_pct)}% · ${gmEsc(teamShort(D.r.home_team_name, D.sport))} ${Math.round(mh.cash_pct)}%</p>` : "";
    body = `<div class="ca-gm-cov"><span class="ca-gm-cov-s">${logoImg(D.r.away_team_name, D.sport)}<b>${gmPct(c.away)}</b></span>
      <span class="ca-gm-bar"><i style="width:${away.toFixed(1)}%;background:${gmEsc(D.awayCol)}"></i><i style="width:${(100 - away).toFixed(1)}%;background:${gmEsc(D.homeCol)}"></i></span>
      <span class="ca-gm-cov-s"><b>${gmPct(c.home)}</b>${logoImg(D.r.home_team_name, D.sport)}</span></div>${cap}`;
  }
  return `<section class="ca-card ca-gm-cover" id="gm-cover"><div class="ca-card-head"><h2>Cover Probability</h2></div>${body}</section>`;
}
function gmOverview(D) {
  const cards = GM_OVERVIEW.map(([n, fn]) => safeCard(n, fn, D, "ca-card ca-gm-card")).join("");
  const ins = safeCard("Key Insights", gmInsightsCard, D, "ca-card ca-gm-ins-card", "gm-insights");
  return `<div class="ca-gm-cards" id="gm-cards">${cards}</div>${safeCard("Model Projection", gmProjectionCard, D, "ca-card ca-gm-proj", "gm-projection")}
    <div class="ca-gm-lower${ins ? "" : " solo"}">${ins}${safeCard("Cover Probability", gmCoverCard, D, "ca-card ca-gm-cover", "gm-cover")}</div>`;
}

/* ── other tabs ───────────────────────────────────────────────────────── */
const gmLegacy = (html, empty) => (html ? `<div class="ca-gm-legacy">${html}</div>` : empty ? emptyMsg(empty) : "");
function gmPowerCard(D) {
  const code = (side) => { const x = D.ctxHist.find((y) => y.side === side) || D.ctxGrades.find((y) => y.side === side); return x ? String(x.team) : null; };
  const rows = ["away", "home"].map((side) => {
    const p = D.ctxPower.find((x) => String(x.team) === code(side));
    if (!p) return "";
    const u = ctxJson(p.units) || {}, rk = (k) => (u[k] && finite(u[k].rank) ? `#${u[k].rank}` : "—");
    const nm = side === "home" ? D.r.home_team_name : D.r.away_team_name;
    return `<tr><td><span class="ca-team">${logoImg(nm, D.sport)}${gmEsc(nm)}</span></td><td class="ca-b">#${gmEsc(p.rank)}</td><td>${finite(p.rating) ? signedStr(+p.rating, 1) : "—"}</td><td>${rk("pass_off")}</td><td>${rk("run_off")}</td><td>${rk("pass_def")}</td><td>${rk("run_def")}</td></tr>`;
  }).join("");
  if (!rows) return "";
  return `<section class="ca-card ca-gm-power" id="gm-power"><div class="ca-card-head"><h2>Power Rankings</h2><p>Rating = points vs an average team on a neutral field.</p></div>
    <div class="ca-table-wrap"><table class="ca-table"><thead><tr><th>Team</th><th>Rank</th><th>Rating</th><th>Pass Off</th><th>Run Off</th><th>Pass Def</th><th>Run Def</th></tr></thead><tbody>${rows}</tbody></table></div></section>`;
}
function gmMatchupTab(D) {
  const r = D.r;
  const pred = D.isNfl ? nflPredictionSection(r, D.actual, D.simTag) : "";
  const sim = D.isNfl ? gameSimVisual(D.simRows[0], r.away_team_name, r.home_team_name, D.awayCol, D.homeCol) : "";
  const mu = matchupSection(D.ctxGrades, r, D.awayCol, D.homeCol, D.sport);
  return `${safeCard("Power Rankings", gmPowerCard, D, "ca-card", "gm-power")}${gmLegacy(pred + mu + sim, "No matchup data for this game yet.")}`;
}
function gmMovesCard(D) {
  const sm = gmSplitMap(D.splits);
  const rows = D.moves.map((m) => lineMoveInfo(m, D.r, sm)).map((i) =>
    `<div class="ca-gm-mv"><div><b>${gmEsc(i.title)}</b>${i.tickets ? `<small>${gmEsc(i.tickets)}</small>` : ""}</div><span class="${signCls(i.d)} ca-b">${gmEsc(i.chg)}</span></div>`).join("");
  return `<section class="ca-card ca-gm-moves" id="gm-moves"><div class="ca-card-head"><h2>Line Moves</h2><p>Pinnacle, opening to current.</p></div>${rows || emptyMsg("No line moves captured for this game yet.")}</section>`;
}
function gmBooksCard(D) {
  const rows = gmBestBooks(D.r, D.mlRow, D.evRows).map((x) =>
    `<tr><td>${gmEsc(x.label)}</td><td class="ca-b">${oddsStr(x.price)}</td><td><span class="ca-dash-odds">${x.book ? bookBadge(x.book) : ""}${gmEsc(evBookName(x.book))}</span></td></tr>`).join("");
  return `<section class="ca-card ca-gm-books" id="gm-books"><div class="ca-card-head"><h2>Best Price by Side</h2></div>${rows
    ? `<table class="ca-table"><thead><tr><th>Side</th><th>Price</th><th>Book</th></tr></thead><tbody>${rows}</tbody></table>` : emptyMsg("No current prices for this game.")}</section>`;
}
function gmMarketTab(D) {
  const latest = gmLatest(D.evRows), picks = latest.some((x) => x.is_pick) ? evSection(latest.map((x) => ({ ...x, sport: D.sport }))) : "";
  return `<div class="ca-gm-two">${safeCard("Line Moves", gmMovesCard, D, "ca-card", "gm-moves")}${safeCard("Best Price by Side", gmBooksCard, D, "ca-card", "gm-books")}</div>
    ${gmLegacy(splitsSection(D.splits, D.r, D.awayCol, D.homeCol) + picks, "")}`;
}
function gmTrendsTab(D) {
  const r = D.r;
  return gmLegacy(historySection(D.ctxHist, D.ctxPower, r, D.awayCol, D.homeCol, D.sport)
    + trendsSection(r.away_team_name, r.home_team_name, D.trendRecs, D.trendSits, D.awayCol, D.homeCol, r.market_spread, D.sport), "No trends for this game yet.");
}
function gmPlayersTab(D) {
  if (!D.isNfl) return emptyMsg("Player projections are NFL-only for now.");
  return gmLegacy(propsProjectionSection(D.sims, D.props, D.playerActuals, D.propLines, D.simTag) + boxscoreSection(D.sims, D.r), "No player projections for this game yet.");
}
const GM_PANELS = { overview: gmOverview, matchup: gmMatchupTab, market: gmMarketTab, trends: gmTrendsTab, players: gmPlayersTab };

/* ── page ─────────────────────────────────────────────────────────────── */
function gmTabsBar(tab) {
  return `<nav class="ca-gm-tabs" role="tablist" aria-label="Game sections">${GM_TABS.map(([k, l]) =>
    `<button class="ca-gm-tab${k === tab ? " on" : ""}" data-gm-tab="${k}" role="tab" aria-selected="${k === tab}">${l}</button>`).join("")}</nav>`;
}
function gmView(D) {
  const tab = gmState().tab, live = D.sport === "nfl" || D.sport === "cfb";
  const hero = safeCard("Hero", gmHero, D, "ca-gm-hero");
  if (!live) return `<main class="game-view ca-gm">${hero}${emptyMsg(SPORT_STATUS[D.sport] || "No game detail for this sport yet.")}</main>`;
  const panel = safeCard(`${tab} tab`, (x) => GM_PANELS[tab](x), D, "ca-card");
  return `<main class="game-view ca-gm">${hero}${safeCard("CappingAlpha Read", gmReadCard, D, "ca-card ca-gm-read", "gm-read")}${gmTabsBar(tab)}<div class="ca-gm-panel" id="gm-panel">${panel}</div></main>`;
}
async function buildGamePage() {
  const { sport, game } = gameParams();
  const back = `<a class="back-link" href="${encodeURIComponent(sport)}.html">← All ${gmEsc(sport.toUpperCase())} games</a>`;
  if (!game) return `<main class="game-view"><section class="section">${back}<h2>No game selected</h2></section></main>`;
  gmState();
  const D = SPORTS.includes(sport) ? await gmLoad(sport, game) : null;
  if (!D) return `<main class="game-view"><section class="section">${back}<h2>Game not found</h2><p class="muted">No prediction is on record for this game.</p>${SPORT_STATUS[sport] ? emptyMsg(SPORT_STATUS[sport]) : ""}</section></main>`;
  window.__caGameData = D;
  return gmView(D);
}

// Tab clicks redraw the whole view from the cached load (no refetch) and re-bind.
function gmRedraw() {
  const D = window.__caGameData, view = document.querySelector(".game-view");
  if (!D || !view) return;
  view.outerHTML = gmView(D);
  wireGamePage();
}
function wireGamePage() {
  const view = document.querySelector(".game-view");
  if (!view) return;
  view.querySelectorAll("[data-gm-tab]").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.gmTab === gmState().tab) return;
    gmSetTab(b.dataset.gmTab);
    gmRedraw();
  }));
  wireGameSim();
}
