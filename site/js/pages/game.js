/* Game page (game.html?sport=<s>&game=<game_pk>) — mockup #2 "Matchup Story". Real data only: every card has an empty
   state and the mockup's numbers are never rendered. The tab (Overview / Matchup / Market / Trends / Players) lives in
   window.__caGame AND in the URL (?tab=) and survives the 5-minute re-render; switching tabs redraws from the cached load
   (window.__caGameData) without a refetch. The venue / weather hero line (game_info, NFL + CFB), the havoc / turnover / weather Key Insights
   (cfb_team_insights, game_info) and the CFB Projected Game Flow (cfb_quarter_shares, game_info.line_score) render only from rows that
   exist: a missing table or row hides its panel. The TV network is still omitted.
   "Opportunities" are is_pick rows with a non-null Alpha tier (loadOpportunities). Every "CappingAlpha" number is the MODEL's
   (home_win_prob; margin / total distributions at the market line), never ev_picks.true_prob, which is Pinnacle's no-vig line.
   The Read card compares the model with the price available on every side (no pick gate) and is not shown for started games.
   Depends on app.js (sb, gameParams, predictions-row helpers, the legacy sections matchupSection / historySection /
   trendsSection / splitsSection / propsProjectionSection / gameSimVisual / nflPredictionSection / boxscoreSection / evSection,
   wireGameSim, ctxEsc, ctxOrd, logoImg, teamShort, timeET, evBookName, gameMoneylines, evBestLines, nflServedSimVersion,
   simVersionFilter, dedupLatest, distParse, distProbs, mlModelTag, gameTeamColors, pct1, pStr, r05), metrics.js, ui.js,
   shell.js and data.js. */
const GM_TABS = [["overview", "Overview"], ["matchup", "Matchup"], ["market", "Market"], ["trends", "Trends"], ["players", "Players"]];
const ICON_GRADE = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6z"/><path d="M9 12l2 2 4-4"/></svg>`;
const ICON_HAVOC = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L4 14h7l-1 8 9-12h-7z"/></svg>`;
const ICON_SWAP = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 7h11l-3-3M17 17H6l3 3"/></svg>`;
const ICON_CLOUD = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 18a4 4 0 0 1-.5-7.97A5.5 5.5 0 0 1 17 8.5 4.5 4.5 0 0 1 17 18z"/></svg>`;
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

// Rain below this many inches is not "measurable" (CFBD precipitation is inches).
const GM_MEASURABLE_IN = 0.01;
// Weather older than this after kickoff is no longer shown (the venue stays).
const GM_WEATHER_STALE_MS = 3 * 864e5;
// Ms since kickoff (negative before it), or null for an unknown kickoff.
const gmSinceKick = (kickoffIso, nowMs) => { const k = timeMs(kickoffIso); return Number.isFinite(k) ? (nowMs == null ? Date.now() : nowMs) - k : null; };
// Is the weather still shown? Unknown kickoff: yes; otherwise only until 3 days after kickoff. Shared by the hero line and the weather insight.
const gmWeatherFresh = (kickoffIso, nowMs) => { const s = gmSinceKick(kickoffIso, nowMs); return s == null || s <= GM_WEATHER_STALE_MS; };
// Weather pieces in reading order: conditions, "41°F", "wind 14 mph", "20% rain" (or "0.04 in rain"). A missing reading is omitted.
function gmWeatherPieces(i) {
  const t = numOrNull(i.temp_f), w = numOrNull(i.wind_mph), pc = numOrNull(i.precip_chance), pi = numOrNull(i.precip_in);
  const out = [];
  if (i.conditions && String(i.conditions).trim()) out.push(String(i.conditions).trim());
  if (t != null) out.push(`${Math.round(t)}°F`);
  if (w != null) out.push(`wind ${Math.round(w)} mph`);
  if (pc != null && pc > 0) out.push(`${Math.round(pc)}% rain`);
  else if (pi != null && pi >= GM_MEASURABLE_IN) out.push(`${pi.toFixed(2)} in rain`);
  return out;
}
// The hero's venue line from a game_info row: "Venue · City, ST · 41°F, wind 14 mph, 20% rain". Any missing piece is omitted; an indoor
// venue says "Indoors" in place of the weather. The weather label follows kickoff (kickoffIso, nowMs): before kickoff the forecast is shown
// as is; after kickoff a reading is labelled by weather_kind ("Observed" = measured, "Forecast" = still the pre-game forecast, as NFL
// rows always are); more than 3 days after kickoff the weather is hidden. Unknown kickoff: no time-based rule. Plain text ("" = hide).
function gmVenueLine(info, kickoffIso, nowMs) {
  if (!info || typeof info !== "object") return "";
  const place = [info.city, info.state].filter((x) => x && String(x).trim()).join(", ");
  const since = gmSinceKick(kickoffIso, nowMs);    // ms after kickoff (negative before it); null when unknown
  let wx = "";
  if (info.indoor === true) wx = "Indoors";
  else if (gmWeatherFresh(kickoffIso, nowMs)) {
    const p = gmWeatherPieces(info);
    if (p.length) {
      const label = info.weather_kind === "observed" ? "Observed " : info.weather_kind === "forecast" && since != null && since > 0 ? "Forecast " : "";
      wx = label + p.join(", ");
    }
  }
  return [info.venue_name, place, wx].filter((x) => x && String(x).trim()).join(" · ");
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
// The line a side is priced at: the best book's line (ev_best_lines) when known, else the posted market line, from the side's
// own view (away spread = -home). null when unknown. The model is evaluated at this same line, and the label shows it.
function gmSideLine(pred, market, side, lineBy) {
  const l = lineBy && lineBy.get(`${pred.game_pk}|${market}|${side}`);
  if (finite(l)) return +l;
  if (market === "total") return numOrNull(pred.market_total);
  const ls = numOrNull(pred.market_spread);
  return ls == null ? null : side === "home" ? ls : -ls;
}
// The bet as placed ("Lions ML", "Chiefs -6.5", "Over 56.5"), through the shared pickLabel, at an explicit line.
const gmLabel = (pred, market, side, line) => pickLabel({ kind: "line", sport: pred.sport, game_pk: pred.game_pk, market, side,
  matchup: `${pred.away_team_name} @ ${pred.home_team_name}` }, new Map([[`${pred.game_pk}|${market}|${side}`, line]]));
const gmImplied = (r) => { const p = finite(r.best_line_implied) ? +r.best_line_implied : americanToProb(r.best_price); return Number.isFinite(p) ? p : null; };

// The MODEL's probabilities (never ev_picks.true_prob, which is Pinnacle's no-vig line): the margin / total distributions
// of the prediction row when it carries them (predictions_any.margin_dist / total_dist, db/migration_site_redesign_a.sql),
// else of the NFL sim row; null parts when neither has a distribution. Sign convention: margin = HOME minus AWAY points
// ({kind:"margin", offset, pmf}, pmf[i] = P(margin = i - offset)), as the CFB / NFL writers emit it
// (sportsmodel.nfl.gameline.build_gameline: home_win_prob = P(margin > 0)); gmOdds' cover math assumes exactly that.
function gmDists(pred, sim) {
  const get = (k) => { for (const src of [pred, sim]) { const d = src && distParse(src[k]); if (d && Array.isArray(d.pmf)) return d; } return null; };
  return { margin: get("margin_dist"), total: get("total_dist") };
}
// P(value > threshold) / P(value < threshold) of a distribution, pushes excluded; null without a distribution or threshold.
function gmSplit(dist, threshold) {
  if (!dist || threshold == null) return null;
  const { under, over } = distProbs(dist, threshold), s = under + over;
  return s > 0 ? { over: over / s, under: under / s } : null;
}

// Per-market view of the game's odds: consensus moneyline prices, and per side the line it is priced at, the implied
// probability of the price you would take (latest ev_picks row; moneyline falls back to the moneylines view's best price)
// and the model probability at that same line. modelProb is null whenever the model has no number for the market.
function gmOdds(pred, evRows, mlRow, sim, lineBy) {
  const p = pred || {}, by = new Map(gmLatest(evRows).map((r) => [`${r.market}|${r.side}`, r])), d = gmDists(p, sim);
  const wp = numOrNull(p.home_win_prob);
  const mlSide = (side) => {
    const r = by.get(`moneyline|${side}`), cons = mlRow ? consensusAmerican(mlRow[`${side}_prices`]) : null;
    const bestPrice = r && numOrNull(r.best_price) != null ? +r.best_price : mlRow ? numOrNull(mlRow[`${side}_price`]) : null;
    const price = cons != null ? cons : bestPrice;
    const prob = (x) => (x != null && Number.isFinite(americanToProb(x)) ? americanToProb(x) : null);
    return { price, implied: prob(price), bestPrice, bestImplied: r && gmImplied(r) != null ? gmImplied(r) : prob(bestPrice),
      modelProb: wp == null ? null : side === "home" ? wp : 1 - wp };
  };
  const one = (market, side, modelAt) => {
    const r = by.get(`${market}|${side}`), line = gmSideLine(p, market, side, lineBy), imp = r ? gmImplied(r) : null, price = r ? numOrNull(r.best_price) : null;
    return { line, price, implied: imp, bestPrice: price, bestImplied: imp, modelProb: line == null ? null : modelAt(line) };
  };
  const m = d.margin, t = d.total;
  return {
    moneyline: { away: mlSide("away"), home: mlSide("home") },
    spread: { line: numOrNull(p.market_spread),
      away: one("spread", "away", (L) => { const s = gmSplit(m, L); return s && s.under; }),     // away covers when the home margin < its line
      home: one("spread", "home", (L) => { const s = gmSplit(m, -L); return s && s.over; }) },   // home covers when the home margin > -its line
    total: { line: numOrNull(p.market_total),
      over: one("total", "over", (L) => { const s = gmSplit(t, L); return s && s.over; }),
      under: one("total", "under", (L) => { const s = gmSplit(t, L); return s && s.under; }) },
  };
}

// Every side of the game with both a model probability and a market probability, as read candidates
// [{market, side, label, line, price, modelProb, impliedProb}] (no pick gate: the model's view against the price available).
function gmCandidates(pred, odds) {
  const out = [];
  for (const [market, side] of [["moneyline", "away"], ["moneyline", "home"], ["spread", "away"], ["spread", "home"], ["total", "over"], ["total", "under"]]) {
    const s = odds && odds[market] && odds[market][side];
    if (!s || s.modelProb == null || s.bestImplied == null) continue;
    const line = market === "moneyline" ? null : s.line;
    out.push({ market, side, label: gmLabel(pred, market, side, line), line, price: s.bestPrice, modelProb: s.modelProb, impliedProb: s.bestImplied });
  }
  return out;
}

// Model cover probability for each side of the spread, at the lines the market prices; null when the model has no distribution.
function gmCover(odds) {
  const s = odds && odds.spread;
  return s && s.home.modelProb != null && s.away.modelProb != null ? { home: s.home.modelProb, away: s.away.modelProb } : null;
}

const gmPctP = (p) => (p != null ? ` (${pct1(p)})` : "");

// Model Projection table: Market / CappingAlpha / Difference for moneyline (the model's favorite), spread (the model's
// side) and total (the model's lean). The CappingAlpha cell is the model, at the same line the Market cell shows. diff =
// model minus market implied probability, in points (null when either is unknown); pts = how many more points the model
// asks for than the market, from the same side (null when unknown).
function modelProjectionRows(pred, odds) {
  const p = pred || {}, o = odds || {}, sport = p.sport;
  const wp = numOrNull(p.home_win_prob), h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score);
  const blank = (label) => ({ label, market: "—", model: "—", diff: null, pts: null });
  const ab = (side) => teamShort(side === "home" ? p.home_team_name : p.away_team_name, sport);
  const rows = [];
  const ml = blank("Moneyline");
  if (wp != null) {
    const side = wp >= 0.5 ? "home" : "away", e = o.moneyline && o.moneyline[side];
    ml.model = pct1(Math.max(wp, 1 - wp));
    if (e && e.price != null) { ml.market = `${ab(side)} ${oddsStr(e.price)}${gmPctP(e.implied)}`; if (e.implied != null) ml.diff = (Math.max(wp, 1 - wp) - e.implied) * 100; }
  }
  rows.push(ml);
  const sp = blank("Spread");
  if (h != null && a != null) {
    const margin = h - a, side = margin >= 0 ? "home" : "away", modelLine = r05(side === "home" ? -margin : margin) || 0;   // the half-point line the cell shows
    const e = o.spread && o.spread[side];
    sp.model = `${ab(side)} ${modelLineStr(modelLine)}${gmPctP(e ? e.modelProb : null)}`;
    if (e && e.line != null) {
      sp.market = `${ab(side)} ${lineStr(e.line)}${gmPctP(e.implied)}`;
      sp.pts = e.line - modelLine;
      if (e.modelProb != null && e.implied != null) sp.diff = (e.modelProb - e.implied) * 100;
    }
  }
  rows.push(sp);
  const tt = blank("Total");
  if (h != null && a != null) {
    const model = h + a, T0 = o.total ? o.total.line : null, lean = T0 == null ? null : model >= T0 ? "over" : "under";
    const e = lean && o.total[lean], T = e && e.line != null ? e.line : T0;
    tt.model = `${model.toFixed(1)}${gmPctP(e ? e.modelProb : null)}`;
    if (T != null) {
      tt.market = `${lean === "under" ? "U" : "O"} ${T}${gmPctP(e ? e.implied : null)}`;
      tt.pts = lean === "over" ? model - T : T - model;
      if (e && e.modelProb != null && e.implied != null) tt.diff = (e.modelProb - e.implied) * 100;
    }
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

// Up to four generated sentences from data that exists, strongest first. Candidates: offense-vs-defense unit grade, explosive-play
// edge, power-rank gap, current streak (all sports) and, from game_info / cfb_team_insights, a CFB havoc mismatch, a CFB turnover edge
// and a weather row (CFB + NFL). Each is {kind, strength, title, body} (plain text; escaped when rendered); strength is 0-100 and
// derived below, ties keep the order the candidates are generated in, so the pick of four and its order are deterministic.
const gmPoss = (n) => (/s$/i.test(n) ? `${n}'` : `${n}'s`);
// team_history / matchup_grades / power_rankings identify a side by team code.
const gmTeamCode = (hist, grades, side) => { const x = (hist || []).find((y) => y.side === side) || (grades || []).find((y) => y.side === side); return x ? String(x.team) : null; };
const GM_STREAK = { su: { W: ["won", "won"], L: ["lost", "lost"] }, ats: { W: ["covered", "covered"], L: ["failed to cover", "failed to cover"] },
  ou: { O: ["has gone over in", "have gone over in"], U: ["has gone under in", "have gone under in"] } };
// Strengths. Existing rows sit in 30-70 from their own thresholds (grade: the offense's percentile; explosive: shown whenever both
// sides are computable; power: the rank gap; streak: its length, 3 or more); the new rows start at 50 and are only generated past
// their notable thresholds, so a real mismatch / weather outranks a routine row but a routine row still fills an empty slot.
const GM_STRENGTH = { grade: (pct) => Math.round(30 + 0.4 * pct), explosive: () => 45, power: (gap) => 30 + Math.min(40, gap), streak: (n) => Math.min(70, 30 + 8 * (n - 2)) };
const GM_NOTABLE_GAP = 30, GM_NOTABLE_PCT = 0.15;
const gmCut = (n) => Math.ceil(GM_NOTABLE_PCT * n);   // how many teams are the top (or bottom) 15% of an n-team field
// The newest-season cfb_team_insights row of a side's team, or null (rows arrive newest season first).
const gmInsRow = (D, side) => { const c = gmTeamCode(D.ctxHist, D.ctxGrades, side); return c == null ? null : (D.cfbIns || []).find((x) => String(x.team) === c) || null; };
const gmPct1 = (x) => (finite(x) ? `${(+x * 100).toFixed(1)}%` : "");
// CFB: one defense's havoc-created rank against the other offense's havoc-allowed rank (1 = best on both). Notable when the gap
// (offense rank minus defense rank, positive = defense edge) is 30+, or the defense has the edge and is top 15% / the offense bottom 15%.
function gmHavocInsight(D, name) {
  let best = null;
  for (const dSide of ["home", "away"]) {
    const d = gmInsRow(D, dSide), o = gmInsRow(D, dSide === "home" ? "away" : "home");
    if (!d || !o || !finite(d.def_havoc_rank) || !finite(o.off_havoc_allowed_rank) || !finite(d.n_ranked ?? o.n_ranked)) continue;
    const n = +(d.n_ranked ?? o.n_ranked), gap = o.off_havoc_allowed_rank - d.def_havoc_rank, cut = gmCut(n);
    const extreme = gap > 0 && (d.def_havoc_rank <= cut || o.off_havoc_allowed_rank > n - cut);
    if (!(gap >= GM_NOTABLE_GAP || extreme)) continue;
    const strength = Math.round(50 + Math.min(45, gap / 3));
    if (best && best.strength >= strength) continue;
    const dn = name(dSide), on = name(dSide === "home" ? "away" : "home"), dr = gmPct1(d.def_havoc_rate), or = gmPct1(o.off_havoc_allowed_rate);
    best = { kind: "havoc", strength, title: `${dn} defense creates havoc against ${on}`,
      body: `${dn} ranks #${d.def_havoc_rank} in havoc created${dr ? ` (${dr} of plays)` : ""}; ${gmPoss(on)} offense ranks #${o.off_havoc_allowed_rank} in havoc allowed${or ? ` (${or})` : ""}, where #1 allows the least.` };
  }
  return best;
}
// CFB: turnover-margin rank gap of 30+, or either team in the top / bottom 15% (equal ranks: no edge).
function gmTurnoverInsight(D, name) {
  const h = gmInsRow(D, "home"), a = gmInsRow(D, "away");
  if (!h || !a || !finite(h.turnover_margin_rank) || !finite(a.turnover_margin_rank) || !finite(h.n_ranked ?? a.n_ranked)) return null;
  const n = +(h.n_ranked ?? a.n_ranked), hr = +h.turnover_margin_rank, ar = +a.turnover_margin_rank, gap = Math.abs(hr - ar), cut = gmCut(n);
  const ext = (r) => r <= cut || r > n - cut;
  if (hr === ar || !(gap >= GM_NOTABLE_GAP || ext(hr) || ext(ar))) return null;
  const [bs, b, w] = hr < ar ? ["home", h, a] : ["away", a, h], per = (x) => (finite(x.turnover_margin_per_game) ? ` is ${signedStr(+x.turnover_margin_per_game, 1)} per game` : "");
  return { kind: "turnover", strength: Math.round(50 + Math.min(45, gap / 3)), title: `${name(bs)} owns the turnover edge`,
    body: `${name(bs)}${per(b) || " ranks"} (#${b.turnover_margin_rank} nationally); ${name(bs === "home" ? "away" : "home")}${per(w) || " ranks"} (#${w.turnover_margin_rank}).` };
}
// CFB + NFL: outdoor games only. Wind 15+ mph, under 40°F, 50%+ chance of rain or measurable rain; strength from the worst factor.
// Hidden once the weather is stale (3 days after kickoff), by the same rule as the hero's venue line.
function gmWeatherInsight(D) {
  const i = D && D.gameInfo;
  if (!i || i.indoor === true || !gmWeatherFresh(D.r && D.r.commence_time, D.nowMs)) return null;    // D.nowMs is a test seam; production leaves it unset and Date.now() is used
  const t = numOrNull(i.temp_f), w = numOrNull(i.wind_mph), pc = numOrNull(i.precip_chance), pi = numOrNull(i.precip_in), f = [], sc = [];
  if (w != null && w >= 15) { f.push(`wind ${Math.round(w)} mph`); sc.push(55 + Math.min(35, (w - 15) * 3)); }
  if (t != null && t < 40) { f.push(`${Math.round(t)}°F`); sc.push(55 + Math.min(30, (40 - t) * 2)); }
  if (pc != null && pc >= 50) { f.push(`${Math.round(pc)}% chance of rain`); sc.push(55 + Math.min(30, (pc - 50) * 0.6)); }
  else if (pi != null && pi >= GM_MEASURABLE_IN) { f.push(`${pi.toFixed(2)} in of rain`); sc.push(60); }
  if (!f.length) return null;
  return { kind: "weather", strength: Math.round(Math.max(...sc)), title: `Weather: ${f.join(", ")}`,
    body: `${i.weather_kind === "observed" ? "Observed" : "Forecast"} conditions${i.venue_name ? ` at ${i.venue_name}` : ""}. Descriptive only, not a pick.` };
}
// CFB Projected Game Flow. Each team's projected points by quarter = its projected score x the average of its own scored shares and its
// opponent's allowed shares (cfb_quarter_shares: Q1-Q4 shares of points, shrunk to the league, each vector sums to 1), renormalised to
// 1. Returns {home: [q1..q4], away: [q1..q4], actual: {home, away} | null, games: {home, away}} or null when it is not CFB, has no
// projected score, or either team lacks a share row (the card hides). `actual` = game_info.line_score once the game has a graded result or has started.
const gmShareRow = (D, side) => { const c = gmTeamCode(D.ctxHist, D.ctxGrades, side); return c == null ? null : (D.cfbShares || []).find((x) => String(x.team) === c) || null; };
function gmLineScore(info) {
  const ls = info && ctxJson(info.line_score);
  const ok = (a) => Array.isArray(a) && a.length === 4 && a.every((v) => finite(v) && +v >= 0);
  return ls && ok(ls.home) && ok(ls.away) ? { home: ls.home.map(Number), away: ls.away.map(Number) } : null;
}
function gmGameFlow(D) {
  if (!D || D.sport !== "cfb") return null;
  const r = D.r || {}, h = numOrNull(r.pred_home_score), a = numOrNull(r.pred_away_score);
  if (h == null || a == null || h < 0 || a < 0 || h + a <= 0) return null;
  const sh = gmShareRow(D, "home"), sa = gmShareRow(D, "away");
  if (!sh || !sa) return null;
  const vec = (row, kind) => [1, 2, 3, 4].map((q) => numOrNull(row[`${kind}_q${q}`]));
  const blend = (own, opp) => {
    const sc = vec(own, "scored"), al = vec(opp, "allowed");
    if ([...sc, ...al].some((x) => x == null || x < 0)) return null;
    const m = sc.map((x, i) => (x + al[i]) / 2), tot = m.reduce((x, y) => x + y, 0);
    return tot > 0 ? m.map((x) => x / tot) : null;
  };
  const bh = blend(sh, sa), ba = blend(sa, sh);
  if (!bh || !ba) return null;
  return { home: bh.map((x) => x * h), away: ba.map((x) => x * a), actual: (D.actual || gmStarted(r)) ? gmLineScore(D.gameInfo) : null, games: { home: numOrNull(sh.games_used), away: numOrNull(sa.games_used) } };
}
function gmKeyInsights(D) {
  const r = (D && D.r) || {}, grades = (D && D.ctxGrades) || [], hist = (D && D.ctxHist) || [], power = (D && D.ctxPower) || [];
  const name = (side) => shortTeam(side === "home" ? r.home_team_name : r.away_team_name, D && D.sport);
  const other = (s) => (s === "home" ? "away" : "home");
  const pl = (D && D.sport) === "nfl" ? 1 : 0;   // NFL short names are plural nicknames ("Saints rank"), CFB schools singular ("Alabama ranks")
  const out = [];
  const graded = grades.filter((g) => g && g.overall && finite(g.overall_pct)).sort((x, y) => y.overall_pct - x.overall_pct)[0];
  if (graded) out.push({ kind: "grade", strength: GM_STRENGTH.grade(+graded.overall_pct), title: `${name(graded.side)} offense grades ${graded.overall} vs ${name(other(graded.side))}`,
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
    out.push({ kind: "explosive", strength: GM_STRENGTH.explosive(), title: `Big-play grades favor ${name(w)}`,
      body: `${gmPoss(name(w))} offense vs ${gmPoss(name(other(w)))} defense is ahead of the reverse matchup on pass-rate-weighted explosive-play ratings.` });
  }
  const pr = (side) => power.find((x) => String(x.team) === gmTeamCode(hist, grades, side));
  const ph = pr("home"), pa = pr("away");
  if (ph && pa && finite(ph.rank) && finite(pa.rank) && ph.rank !== pa.rank) {
    const [bs, b, w] = ph.rank < pa.rank ? ["home", ph, pa] : ["away", pa, ph];
    const gap = finite(b.rating) && finite(w.rating) ? ` Rating gap: ${(b.rating - w.rating).toFixed(1)} pts.` : "";
    out.push({ kind: "power", strength: GM_STRENGTH.power(Math.abs(ph.rank - pa.rank)), title: `${name(bs)} ${pl ? "rank" : "ranks"} #${b.rank} in the power rankings`, body: `${name(other(bs))} ${pl ? "rank" : "ranks"} #${w.rank}.${gap}` });
  }
  let st = null;
  for (const h of hist) {
    const S = ctxJson(h.streaks) || {};
    for (const k of ["su", "ats", "ou"]) {
      const m = /^([A-Z])(\d+)$/.exec(String(S[k] || ""));
      if (m && GM_STREAK[k][m[1]] && +m[2] >= 3 && (!st || +m[2] > st.n)) st = { side: h.side, k, kind: m[1], n: +m[2] };
    }
  }
  if (st) out.push({ kind: "streak", strength: GM_STRENGTH.streak(st.n), title: `${name(st.side)} ${GM_STREAK[st.k][st.kind][pl]} ${st.n} straight`,
    body: `Current ${{ su: "straight-up", ats: "against-the-spread", ou: "over/under" }[st.k]} streak entering this game.` });
  if (D && D.sport === "cfb") for (const x of [gmHavocInsight(D, name), gmTurnoverInsight(D, name)]) if (x) out.push(x);
  const wx = gmWeatherInsight(D);
  if (wx) out.push(wx);
  return out.map((x, i) => ({ x, i })).sort((p, q) => q.x.strength - p.x.strength || p.i - q.i).slice(0, 4).map((p) => p.x);
}

// Best price per side: moneyline from the moneylines view (else the pick builds), spread / total from the latest builds.
function gmBestBooks(pred, mlRow, evRows) {
  const rows = gmLatest(evRows), out = [];
  for (const side of ["away", "home"]) {
    const ev = rows.find((r) => r.market === "moneyline" && r.side === side);
    const price = mlRow && finite(mlRow[`${side}_price`]) ? +mlRow[`${side}_price`] : ev ? numOrNull(ev.best_price) : null;
    const book = mlRow && finite(mlRow[`${side}_price`]) ? mlRow[`${side}_book`] : ev ? ev.best_book : null;
    if (price != null) out.push({ label: gmLabel(pred, "moneyline", side, null), price, book });
  }
  for (const r of rows) {
    if (r.market === "moneyline" || numOrNull(r.best_price) == null) continue;
    out.push({ label: gmLabel(pred, r.market, r.side, gmSideLine(pred, r.market, r.side, null)), price: +r.best_price, book: r.best_book || null });
  }
  return out;
}

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
  const isNfl = sport === "nfl", live = LIVE_SPORTS.includes(sport), none = (v) => Promise.resolve(v);
  const [predsAny, predsCur, evRows, accRows, servedVersion, mls, opps, moves, lineBy, gameInfoRows] = await Promise.all([
    sb(`predictions_any?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    sb(`predictions_current?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    live ? sb(`ev_picks?sport=eq.${sport}&game_pk=eq.${game}&order=created_at.desc`).catch(() => []) : none([]),
    sb(`prediction_accuracy?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    isNfl ? nflServedSimVersion() : none(null),
    live ? gameMoneylines().catch(() => new Map()) : none(new Map()),
    live ? loadOpportunities().catch(() => []) : none([]),
    live ? loadLineMoves().catch(() => []) : none([]),
    live ? evBestLines().catch(() => new Map()) : none(new Map()),
    // venue / weather (+ CFB line score); the table may not exist yet -> [] and the hero line / weather row / actual quarters hide
    isNfl || sport === "cfb" ? sb(`game_info?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []) : none([]),
  ]);
  const r = predsAny[0] || predsCur[0];
  if (!r) return null;
  r.sport = sport;
  let sims = [], props = [], simRows = [], playerActuals = [], propLines = [], splits = [], trendRecs = [], trendSits = [], ctxHist = [], ctxGrades = [], ctxPower = [], cfbIns = [], cfbShares = [];
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
    // CFB only: both teams' havoc / turnover ranks (every stored season, newest first) and quarter-scoring shares; a missing table -> []
    if (sport === "cfb" && codes.length) {
      const list = codes.map((c) => q(String(c))).join(",");
      [cfbIns, cfbShares] = await Promise.all([
        sb(`cfb_team_insights?team=in.(${list})&order=season.desc`).catch(() => []),
        sb(`cfb_quarter_shares?team=in.(${list})`).catch(() => []),
      ]);
    }
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
    gameInfo: (gameInfoRows || [])[0] || null, cfbIns: cfbIns || [], cfbShares: cfbShares || [],
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
  const r = D.r, sport = D.sport, o = gmOdds(r, D.evRows, D.mlRow, null, D.lineBy), lean = gmLean(r), live = LIVE_SPORTS.includes(sport);
  const when = gmWhen(r.commence_time);
  const team = (side) => {
    const rec = gmRecord(D.ctxHist, D.ctxPower, side), nm = side === "home" ? r.home_team_name : r.away_team_name;
    // R24: the team's star (shell-owned toggle) fills the dashboard Watchlist's Teams tab; id = the full team name.
    return `<div class="ca-gm-team">${logoImg(nm, sport)}<div class="ca-gm-name"><h1>${gmEsc(gmName(D, side))}</h1>${starButton("teams", nm, nm)}</div>${rec ? `<p class="ca-gm-rec">${gmEsc(rec)}</p>` : ""}</div>`;
  };
  const price = (x) => (x != null ? oddsStr(x) : "—");
  const L = o.spread.line, T = o.total.line;
  const boxes = live ? `<div class="ca-gm-odds">
    ${gmOddsBox("MONEYLINE", price(o.moneyline.away.price), price(o.moneyline.home.price), lean.ml === "away" ? "l" : lean.ml === "home" ? "r" : "")}
    ${gmOddsBox("SPREAD", L != null ? lineStr(-L) : "—", L != null ? lineStr(L) : "—", lean.spread === "away" ? "l" : lean.spread === "home" ? "r" : "")}
    ${gmOddsBox("TOTAL", T != null ? `O ${T}` : "—", T != null ? `U ${T}` : "—", lean.total === "over" ? "l" : lean.total === "under" ? "r" : "")}</div>` : "";
  const venue = gmVenueLine(D.gameInfo, r.commence_time);
  return `<section class="ca-gm-hero${boxes ? " has-odds" : ""}"><div class="ca-gm-top"><a class="ca-gm-back" href="${sport}.html">‹ ${SPORT_NAME[sport] || sport.toUpperCase()} Board</a><span class="ca-gm-when">${gmEsc(when)}</span><span></span></div>
    ${venue ? `<p class="ca-gm-venue">${gmEsc(venue)}</p>` : ""}<div class="ca-gm-teams">${team("away")}<span class="ca-gm-at">@</span>${team("home")}</div>${gmFinalLine(D)}${boxes}</section>`;
}

const gmOddsOf = (D) => gmOdds(D.r, D.evRows, D.mlRow, D.simRows[0], D.lineBy);
const gmStarted = (r) => timeMs(r.commence_time) <= Date.now();
function gmReadCard(D) {
  const started = gmStarted(D.r), m = started ? null : gameRead(gmCandidates(D.r, gmOddsOf(D)));
  const noun = m && { moneyline: "win", spread: "cover", total: "hit" }[m.market];
  const body = started
    ? `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>Pre-game read — this game has started</h2>
        <p>The model read is shown before kickoff.</p></div>`
    : m
    ? `<div class="ca-gm-read-text"><p class="ca-gm-kicker">CAPPINGALPHA READ</p><h2>${gmEsc(m.label)} shows the strongest model divergence</h2>
        <p>The market implies a ${pct1(m.impliedProb)} ${noun} probability. CappingAlpha estimates <b>${pct1(m.modelProb)}</b>.</p></div>
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
    ? `<div class="ca-gm-big">${gmEsc(o.alpha)}<small> / 100</small></div>${confPill(o.tier)}<div class="ca-gm-sub ca-ell" title="${gmEsc(oppLabel(o, D.lineBy))}">${gmEsc(oppLabel(o, D.lineBy))}</div>`
    : `<div class="ca-gm-big">—</div><div class="ca-gm-sub">No +EV opportunity on this game.</div>`}</div>`;
}
function gmWinCard(D) {
  const wp = numOrNull(D.r.home_win_prob), row = (name, p) => `<div class="ca-gm-wp">${logoImg(name, D.sport)}<b>${p != null ? pct1(p) : "—"}</b></div>`;
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Win Probability</div>${row(D.r.home_team_name, wp)}${row(D.r.away_team_name, wp != null ? 1 - wp : null)}</div>`;
}
// A projected score: whole points for football, one decimal for MLB (a 4.6 - 3.9 game is not "5 - 4").
const gmScoreNum = (D, x) => (D.sport === "mlb" ? (+x).toFixed(1) : Math.round(x));
function gmScoreCard(D) {
  const h = numOrNull(D.r.pred_home_score), a = numOrNull(D.r.pred_away_score);
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Score</div>${h != null && a != null
    ? `<div class="ca-gm-score">${logoImg(D.r.away_team_name, D.sport)}<b>${gmScoreNum(D, a)}</b><span>-</span><b>${gmScoreNum(D, h)}</b>${logoImg(D.r.home_team_name, D.sport)}</div>`
    : `<div class="ca-gm-big">—</div>`}</div>`;
}
function gmSpreadCard(D) {
  const r = D.r, h = numOrNull(r.pred_home_score), a = numOrNull(r.pred_away_score), L = numOrNull(r.market_spread);
  if (h == null || a == null) return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Spread</div><div class="ca-gm-big">—</div></div>`;
  const margin = h - a, side = margin >= 0 ? "home" : "away", nm = teamShort(side === "home" ? r.home_team_name : r.away_team_name, D.sport);
  const mk = L == null ? "" : `<div class="ca-gm-sub">Market: ${lineStr(side === "home" ? L : -L)}</div>`;
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Spread</div><div class="ca-gm-mid">${gmEsc(nm)} ${gmEsc(modelLineStr(side === "home" ? -margin : margin))}</div>${mk}</div>`;
}
function gmTotalCard(D) {
  const r = D.r, h = numOrNull(r.pred_home_score), a = numOrNull(r.pred_away_score), T = numOrNull(r.market_total);
  if (h == null || a == null) return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Total</div><div class="ca-gm-big">—</div></div>`;
  return `<div class="ca-card ca-gm-card"><div class="ca-gm-cl">Projected Total</div><div class="ca-gm-mid">${(h + a).toFixed(1)}</div>${T != null ? `<div class="ca-gm-sub">Market: ${T}</div>` : ""}</div>`;
}
const GM_OVERVIEW = [["Alpha Score", gmAlphaCard], ["Win Probability", gmWinCard], ["Projected Score", gmScoreCard], ["Projected Spread", gmSpreadCard], ["Projected Total", gmTotalCard]];

function gmProjectionCard(D) {
  const rows = modelProjectionRows(D.r, gmOddsOf(D));
  const body = rows.map((x) => {
    const d = x.diff != null ? `<td class="${signCls(x.diff)} ca-b">${pStr(x.diff)}</td>` : x.pts != null ? `<td class="${signCls(x.pts)} ca-b">${signedStr(x.pts, 1, " pts")}</td>` : `<td class="muted">—</td>`;
    return `<tr><td>${gmEsc(x.label)}</td><td>${gmEsc(x.market)}</td><td class="ca-b">${gmEsc(x.model)}</td>${d}</tr>`;
  }).join("");
  return `<section class="ca-card ca-gm-proj" id="gm-projection"><div class="ca-card-head"><h2>Model Projection</h2></div><table class="ca-table ca-gm-table"><thead><tr><th></th><th>Market</th><th>CappingAlpha</th><th>Difference</th></tr></thead><tbody>${body}</tbody></table></section>`;
}
const GM_INSIGHT_ICON = { grade: ["ca-gm-ic-red", ICON_GRADE], explosive: ["ca-gm-ic-amber", ICON_TREND], power: ["ca-gm-ic-blue", ICON_RANK], streak: ["ca-gm-ic-green", ICON_WAVE],
  havoc: ["ca-gm-ic-red", ICON_HAVOC], turnover: ["ca-gm-ic-amber", ICON_SWAP], weather: ["ca-gm-ic-blue", ICON_CLOUD] };
function gmInsightsCard(D) {
  const ins = gmKeyInsights(D);
  if (!ins.length) return "";
  const rows = ins.map((i) => `<div class="ca-gm-ins"><span class="ca-gm-ic ${GM_INSIGHT_ICON[i.kind][0]}">${GM_INSIGHT_ICON[i.kind][1]}</span><div><b>${gmEsc(i.title)}</b><p>${gmEsc(i.body)}</p></div></div>`).join("");
  return `<section class="ca-card ca-gm-ins-card" id="gm-insights"><div class="ca-card-head"><h2>Key Insights</h2></div>${rows}</section>`;
}
const GM_FLOW_TIP = "Each team's own quarter-by-quarter scoring pattern over the last two seasons and this one (shrunk toward the league average), averaged with what its opponent allows by quarter, applied to the projected score. A pace estimate, not a prediction of the score by quarter. Overtime is excluded.";
// Cumulative points after each quarter, starting from 0: [0, q1, q1+q2, q1+q2+q3, total].
const gmFlowCumulative = (qs) => qs.reduce((acc, q) => (acc.push(acc[acc.length - 1] + q), acc), [0]);
// The Cover Probability block (away % | two-colour bar | home %, plus the "% of Money" caption when splits exist). Shared by the
// standalone card and the Projected Game Flow card.
function gmCoverBody(D) {
  const c = gmCover(gmOddsOf(D));
  if (!c) return emptyMsg("Model cover probability isn't available for this game yet.");
  const away = Math.max(0, Math.min(100, c.away / (c.away + c.home) * 100));
  const money = latestSplitMap(D.splits), ma = money.get(`${D.game}|spread|away`), mh = money.get(`${D.game}|spread|home`);
  const cap = ma && mh && finite(ma.cash_pct) && finite(mh.cash_pct)
    ? `<p class="ca-gm-cap">% of Money · ${gmEsc(teamShort(D.r.away_team_name, D.sport))} ${Math.round(ma.cash_pct)}% · ${gmEsc(teamShort(D.r.home_team_name, D.sport))} ${Math.round(mh.cash_pct)}%</p>` : "";
  return `<div class="ca-gm-cov"><span class="ca-gm-cov-s">${logoImg(D.r.away_team_name, D.sport)}<b>${pct1(c.away)}</b></span>
      <span class="ca-gm-bar"><i style="width:${away.toFixed(1)}%;background:${gmEsc(D.awayCol)}"></i><i style="width:${(100 - away).toFixed(1)}%;background:${gmEsc(D.homeCol)}"></i></span>
      <span class="ca-gm-cov-s"><b>${pct1(c.home)}</b>${logoImg(D.r.home_team_name, D.sport)}</span></div>${cap}`;
}
// One narrow card: cumulative projected points by quarter (solid team-coloured lines with dots; on a started / final game the actual
// cumulative line is overlaid dashed), then the Cover Probability block underneath. "" when there is no flow (gmGameFlow null).
function gmFlowCard(D) {
  const f = gmGameFlow(D);
  if (!f) return "";
  const away = shortTeam(D.r.away_team_name, "cfb"), home = shortTeam(D.r.home_team_name, "cfb");
  const series = [{ color: D.awayCol, values: gmFlowCumulative(f.away) }, { color: D.homeCol, values: gmFlowCumulative(f.home) }];
  if (f.actual) series.push({ color: D.awayCol, values: gmFlowCumulative(f.actual.away), dash: "6 4" }, { color: D.homeCol, values: gmFlowCumulative(f.actual.home), dash: "6 4" });
  const chart = lineChart(series, [null, "Q1", "Q2", "Q3", "Q4"], { h: 230, yTicks: 5, nice: true, dots: true });
  const item = (col, name) => `<span class="ca-legend-item"><i style="background:${gmEsc(col)}"></i>${gmEsc(name)}</span>`;
  const legend = `<div class="ca-legend">${item(D.awayCol, away)}${item(D.homeCol, home)}${f.actual ? `<span class="ca-legend-note">dashed = actual</span>` : ""}</div>`;
  const n = f.games.away != null && f.games.home != null ? `Shares come from ${gmEsc(away)}'s ${f.games.away} and ${gmEsc(home)}'s ${f.games.home} games over the last two seasons and this one.` : "";
  const notes = [f.actual ? "Actual (dashed) excludes overtime." : "", n].filter(Boolean).map((t) => `<p class="ca-gm-cap ca-gm-flow-note">${t}</p>`).join("");
  return `<section class="ca-card ca-gm-flow" id="gm-flow"><div class="ca-card-head"><h2>Projected Game Flow</h2>${infoTip(GM_FLOW_TIP)}</div>${legend}${chart}${notes}
    <div class="ca-gm-flow-cover"><h3>Cover Probability</h3>${gmCoverBody(D)}</div></section>`;
}
function gmCoverCard(D) {
  return `<section class="ca-card ca-gm-cover" id="gm-cover"><div class="ca-card-head"><h2>Cover Probability</h2></div>${gmCoverBody(D)}</section>`;
}
function gmOverview(D) {
  const cards = GM_OVERVIEW.map(([n, fn]) => safeCard(n, fn, D, "ca-card ca-gm-card")).join("");
  const ins = safeCard("Key Insights", gmInsightsCard, D, "ca-card ca-gm-ins-card", "gm-insights");
  const flow = safeCard("Projected Game Flow", gmFlowCard, D, "ca-card ca-gm-flow", "gm-flow");
  // CFB with a flow: Key Insights on the left, the combined Flow + Cover Probability card on the right. Otherwise the standalone Cover card.
  const right = flow || safeCard("Cover Probability", gmCoverCard, D, "ca-card ca-gm-cover", "gm-cover");
  return `<div class="ca-gm-cards" id="gm-cards">${cards}</div>${safeCard("Model Projection", gmProjectionCard, D, "ca-card ca-gm-proj", "gm-projection")}
    <div class="ca-gm-lower${ins ? "" : " solo"}">${ins}${right}</div>`;
}

/* ── other tabs ───────────────────────────────────────────────────────── */
const gmLegacy = (html, empty) => (html ? `<div class="ca-gm-legacy">${html}</div>` : empty ? emptyMsg(empty) : "");
function gmPowerCard(D) {
  const rows = ["away", "home"].map((side) => {
    const p = D.ctxPower.find((x) => String(x.team) === gmTeamCode(D.ctxHist, D.ctxGrades, side));
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
  const sm = latestSplitMap(D.splits);
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
  if (!D.isNfl) return emptyMsg(D.sport === "mlb" ? "MLB player projections are not on the game page yet; this game's +EV player props are on the Best Opportunities board." : "Player projections are NFL-only for now.");
  // R24: a star per player (shell-owned) fills the Watchlist's Players tab; id = the player's name as ev_prop_picks carries it.
  const star = (name) => starButton("players", name, name);
  return gmLegacy(propsProjectionSection(D.sims, D.props, D.playerActuals, D.propLines, D.simTag, { star }) + boxscoreSection(D.sims, D.r), "No player projections for this game yet.");
}
const GM_PANELS = { overview: gmOverview, matchup: gmMatchupTab, market: gmMarketTab, trends: gmTrendsTab, players: gmPlayersTab };

/* ── page ─────────────────────────────────────────────────────────────── */
function gmTabsBar(tab) {
  return `<nav class="ca-gm-tabs" role="tablist" aria-label="Game sections">${GM_TABS.map(([k, l]) =>
    `<button class="ca-gm-tab${k === tab ? " on" : ""}" data-gm-tab="${k}" role="tab" aria-selected="${k === tab}">${l}</button>`).join("")}</nav>`;
}
function gmView(D) {
  const tab = gmState().tab, live = LIVE_SPORTS.includes(D.sport);
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
