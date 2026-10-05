/* Sport pages, right column (nfl.html / cfb.html / mlb.html / nba.html): Top Alpha Edges, Betting Splits and Model Projections.
   Real data only: every card has an honest empty state and nothing is invented. Mounts into board.js: BOARD_CARDS[id] = (D) => html for the
   three BOARD_SLOTS.right ids and one BOARD_LOADERS entry that puts the period's stored +EV picks and the betting splits into D.right.
   Top Alpha Edges: the +EV opportunities of the selected period, five rows through the shared topAlphaRow (data.js). The current period
   shows the live board; a period that is over shows that period's stored +EV picks (ev_picks / ev_prop_picks, the latest rebuild of each) with
   a W / L / P chip from the graded results (ev_results / ev_prop_results). "model prob" of a game line is Pinnacle's no-vig price (R19).
   Betting Splits: nfl_/cfb_betting_splits_current for the games of the period (tickets and money); the card is hidden when none exist.
   Model Projections: the sport's graded picks of the published record (D.left.rec / accRec, Task B's memoised load) bucketed by the
   model-minus-market edge (the centre table's Edge: ML in probability points vs the no-vig price, spread / total in points), hit rate = picks won / (won + lost);
   a Week / Season toggle picks the period. No per-pick units exist (prediction_pnl is $10 a bet and only daily aggregates are readable), so
   there is no Units column.
   Depends on app.js, metrics.js, ui.js, data.js, board.js (BOARD_*, boardGames, boardLines, boardName) and board-left.js (bl* card helpers). */
const BR_TOP_N = 5;
const BR_TA_PILLS = [["all", "All", "All"], ["spread", "Spread", "Spread"], ["moneyline", "Moneyline", "ML"], ["total", "Total", "Total"], ["props", "Player Props", "Props"]];
const BR_SPLIT_TABS = [["spread", "Spread"], ["moneyline", "Moneyline"], ["total", "Total"]];
const BR_PROJ_MARKETS = [["spread", "Spread", "spread", "pts"], ["moneyline", "Moneyline", "ml", "pp"], ["total", "Total", "total", "pts"]];
// The edge bands, top to bottom (an edge of exactly 10 is in "+5 to +10", of exactly 2 in "-2 to +2").
const BR_BANDS = [["> +10", (e) => e > 10], ["+5 to +10", (e) => e > 5 && e <= 10], ["+2 to +5", (e) => e > 2 && e <= 5], ["-2 to +2", (e) => e >= -2 && e <= 2], ["-5 to -2", (e) => e >= -5 && e < -2], ["< -5", (e) => e < -5]];
const BR_CHUNK = 60;

/* ── pure helpers ─────────────────────────────────────────────────────── */
// Which of the rows' markets a Top Alpha pill keeps ("props" = every player prop).
const brKeeps = (o, key) => key === "all" || (key === "props" ? o.kind === "prop" : o.kind === "line" && o.market === key);
// One opportunity's identity: a game line is (game, market, side), a prop adds the player and the line.
const brKey = (o) => (o.kind === "prop" ? `${o.game_pk}|${o.playerName}|${o.market}|${o.line}|${o.side}` : `${o.sport}|${o.game_pk}|${o.market}|${o.side}`);
// A pick whose game has started (commence in the past): it is no longer live; until its result lands it is "graded-pending".
const brStarted = (o, now) => { const t = Date.parse(o.commence); return Number.isFinite(t) && t < now; };
const brGrade = (won) => (won === true ? "W" : won === false ? "L" : "P");     // a graded row with no winner is a push (spread / total refunded)

// The stored +EV picks of a period as opportunities with a tier (R9), each carrying `result` ("W" | "L" | "P" | null when not graded).
// The latest rebuild of a pick (max created_at) sets its price; a result joins on the pick's key.
function brStoredOpps({ lines, lineRes, props, propRes }) {
  const latest = (rows, key) => { const m = new Map(); for (const r of rows || []) { if (!r) continue; const k = key(r), prev = m.get(k); if (!prev || String(r.created_at || "") >= String(prev.created_at || "")) m.set(k, r); } return [...m.values()]; };
  const lineKey = (r) => `${r.sport}|${r.game_pk}|${r.market}|${r.side}`, propKey = (r) => `${r.game_pk}|${r.player_id}|${r.market}|${r.line}|${r.model_version}`;
  const lr = new Map((lineRes || []).filter(Boolean).map((r) => [lineKey(r), r])), pr = new Map((propRes || []).filter(Boolean).map((r) => [propKey(r), r]));
  const out = [];
  for (const r of latest(lines, lineKey)) { const o = toOpportunity(r, "line", null), g = lr.get(lineKey(r)); if (o && o.tier) out.push({ ...o, result: g ? brGrade(g.won) : null }); }
  for (const r of latest(props, propKey)) {
    const o = toOpportunity(r, "prop", null), g = pr.get(propKey(r));
    if (o && o.tier) out.push({ ...o, result: g && g.result === "win" ? "W" : g && g.result === "loss" ? "L" : g && g.result === "push" ? "P" : null });
  }
  return out;
}
// The opportunities of the period for Top Alpha: the live board (D.opps, no result yet) plus the stored picks of the period not already on it
// (a stored pick that is also live lends its result), by alpha then EV. Every row carries `result`.
function brTopOpps(D) {
  const per = D.period, stored = ((D.right && D.right.stored) || []).filter((o) => { const d = o.commence ? etDateStr(o.commence) : ""; return d >= per.from && d <= per.to; });
  const byKey = new Map(stored.map((o) => [brKey(o), o])), seen = new Set();
  const live = (D.opps || []).map((o) => { seen.add(brKey(o)); const s = byKey.get(brKey(o)); return { ...o, result: s ? s.result : null }; });
  const now = D.nowMs != null ? D.nowMs : Date.now(), done = (o) => brStarted(o, now);
  // The current period ranks the edges still to be played first (a live edge before a finished one), then by alpha and EV.
  return [...live, ...stored.filter((o) => !seen.has(brKey(o)))].sort((a, b) => (per.isCurrent ? done(a) - done(b) : 0) || b.alpha - a.alpha || b.evPct - a.evPct || String(a.game_pk).localeCompare(String(b.game_pk)));
}
// Which edge band an edge falls in (index into BR_BANDS; edges are rounded to a tenth so 5.0000001 is 5).
const brBandOf = (e) => { const x = Math.round(e * 10) / 10; return BR_BANDS.findIndex(([, f]) => f(x)); };
// Graded picks of one market in [from, to] bucketed by their edge: bands [{ label, n, w, l, p, pct }] (n counts pushes, pct = w / (w + l) or null)
// plus a total row and how many picks had no edge (no market line / no closing price) and so are not bucketed.
function brProjData(L, market, from, to) {
  const def = BR_PROJ_MARKETS.find(([k]) => k === market) || BR_PROJ_MARKETS[0], accBy = new Map(((L && L.accRec) || []).map((r) => [String(r.game_pk), r]));
  const bands = BR_BANDS.map(([label]) => ({ label, n: 0, w: 0, l: 0, p: 0, pct: null })), tot = { n: 0, w: 0, l: 0, p: 0, pct: null };
  let picks = 0;
  for (const r of (L && L.rec) || []) {
    if (r.market !== market || r.date < from || r.date > to) continue;
    picks++;
    const a = accBy.get(String(r.game_pk));
    const e = a ? boardLines({ sport: a.sport, game_pk: a.game_pk, away: a.away_team_name, home: a.home_team_name, acc: a, pred: null }, { closing: L.closing }).edge[def[2]] : null;
    if (e == null || !Number.isFinite(e)) continue;
    for (const b of [bands[brBandOf(e)], tot]) { b.n++; if (r.result === "W") b.w++; else if (r.result === "L") b.l++; else b.p++; }
  }
  for (const b of [...bands, tot]) b.pct = b.w + b.l ? b.w / (b.w + b.l) : null;
  return { market: def[0], unit: def[3], bands, total: tot, picks, skipped: picks - tot.n };
}
// The two sides of a split market in display order [{ key, pct, cash }] (null when either ticket share is missing); spread / moneyline: away, home; total: over, under.
function brSplitSides(splits, pk, market) {
  const keys = market === "total" ? ["over", "under"] : ["away", "home"], rows = keys.map((s) => splits && splits.get(`${pk}|${market}|${s}`));
  if (rows.some((r) => !r || !finite(r.ticket_pct))) return null;
  return keys.map((key, i) => ({ key, pct: clip(+rows[i].ticket_pct, 0, 100), cash: finite(rows[i].cash_pct) ? clip(+rows[i].cash_pct, 0, 100) : null, at: rows[i].captured_at || null }));
}

/* ── state ────────────────────────────────────────────────────────────── */
function brState() {
  if (!window.__caBR) window.__caBR = { ta: "all", sp: { tab: "spread", game: "", for: "" }, mp: { market: "spread", scope: null, for: "" } };
  return window.__caBR;
}
const brSetTop = (key) => { brState().ta = key; };
const brSetSplit = (patch, periodKey) => { const st = brState(); st.sp = { ...st.sp, ...patch, for: periodKey }; };
const brSetProj = (patch, periodKey) => { const st = brState(); st.mp = { ...st.mp, ...patch, for: periodKey }; };

/* ── data ─────────────────────────────────────────────────────────────── */
// The stored picks (lines + NFL props) of the period and their results, as brStoredOpps takes them. Each source fails on its own.
async function brPicksLoad(D) {
  const sport = D.sport, per = D.period, win = `commence_time=gte.${addDays(per.from, -1)}T00:00:00Z&commence_time=lt.${addDays(per.to, 2)}T00:00:00Z`;
  const [lines, props] = await Promise.all([
    sbAll(`ev_picks?is_pick=eq.true&sport=eq.${sport}&${win}&select=sport,game_pk,market,side,matchup,commence_time,true_prob,ev_best,best_book,best_price,best_line_implied,created_at&order=created_at.asc,game_pk.asc,market.asc,side.asc,model_version.asc`),
    sport === "nfl" ? sbAll(`ev_prop_picks?is_pick=eq.true&sport=eq.${sport}&${win}&select=sport,game_pk,player_id,player_name,market,side,line,model_version,matchup,commence_time,model_prob,ev_best,best_book,best_price,created_at&order=created_at.asc,game_pk.asc,player_id.asc,market.asc,line.asc,side.asc,model_version.asc`) : [],
  ]);
  const chunks = (rows) => { const ids = [...new Set((rows || []).map((r) => r.game_pk).filter((x) => x != null))], out = []; for (let i = 0; i < ids.length; i += BR_CHUNK) out.push(ids.slice(i, i + BR_CHUNK)); return out; };
  const lineRes = (await Promise.all(chunks(lines).map((c) => sb(`ev_results?sport=eq.${sport}&game_pk=in.(${c.join(",")})&select=sport,game_pk,market,side,won`).catch(() => [])))).flat();
  const propRes = (await Promise.all(chunks(props).map((c) => sb(`ev_prop_results?sport=eq.${sport}&game_pk=in.(${c.join(",")})&select=sport,game_pk,player_id,market,line,model_version,result`).catch(() => [])))).flat();
  return brStoredOpps({ lines, lineRes, props, propRes });
}
async function brLoadAll(D) {
  const R = { stored: [], splits: new Map(), failed: {} }, part = (k, p) => p.then((v) => { R[k] = v; }).catch((e) => { R.failed[k] = true; console.error(`board right: ${k} failed to load`, e); });
  const jobs = [part("splits", loadSplits())];
  if (D.period.from <= D.today) jobs.push(part("stored", brPicksLoad(D)));     // a period that has not started has no stored picks
  await Promise.all(jobs);
  return R;
}
BOARD_LOADERS.push(async (D) => {
  if (D.view !== "games" || !LIVE_SPORTS.includes(D.sport)) return;
  D.right = await brLoadAll(D);
});

/* ── cards ────────────────────────────────────────────────────────────── */
const brPeriodWord = (D) => blPeriodWord(D.period);
const brMsel = (key, items, value) => `<select class="ca-select ca-br-sel" data-br="${ctxEsc(key)}" aria-label="${ctxEsc(key === "mp-market" ? "Market" : "Game")}">${items.map(([k, l]) => `<option value="${ctxEsc(k)}"${k === value ? " selected" : ""}>${ctxEsc(l)}</option>`).join("")}</select>`;

function brTopAlpha(D) {
  const id = "board-top-alpha", title = "Top Alpha Edges";
  if (!blLive(D)) return blEmpty(id, title, `${blNoModel(D)} No +EV edges to rank.`);
  if (!D.right || D.right.failed.stored) return blFailed(id, title);
  const per = D.period, all = brTopOpps(D);
  const present = BR_TA_PILLS.filter(([k]) => k === "all" || all.some((o) => brKeeps(o, k)));
  const st = brState(), key = present.some(([k]) => k === st.ta) ? st.ta : "all", rows = all.filter((o) => brKeeps(o, key)).slice(0, BR_TOP_N);
  const link = `<a class="ca-link" href="ev.html?sport=${encodeURIComponent(D.sport)}&amp;sort=alpha"${per.isCurrent ? "" : ` title="The live +EV board, not ${ctxEsc(brPeriodWord(D))}"`}><span class="ca-bl-v">View </span>All →</a>`;
  const pl = present.length > 1 ? `<div class="ca-bl-pills ca-br-pills">${pills("br-ta", present.map(([k, l, s]) => [k, `<span class="ca-bl-l">${l}</span><span class="ca-bl-s">${s}</span>`]), key)}</div>` : "";
  const over = per.to < D.today, cap = over ? `<p class="ca-bl-cap">${ctxEsc(`${brPeriodWord(D)}'s top edges with their graded results`)}</p>` : "";
  if (!rows.length) {
    const none = all.length ? `No ${(BR_TA_PILLS.find(([k]) => k === key) || [])[1] || ""} edges in ${brPeriodWord(D)}.` : over ? `No +EV picks were stored for ${brPeriodWord(D)}.` : per.from > D.today ? `No +EV opportunities for ${brPeriodWord(D)} yet.` : "No +EV opportunities on the board right now.";
    return blCard(id, `${blHead(title, link)}${pl}${cap}${emptyMsg(none)}`, "ca-br-ta");
  }
  // A row with a result, or whose game already started (finished but not graded yet), gets the result column; the ungraded ones show a dash.
  const now = D.nowMs != null ? D.nowMs : Date.now(), graded = rows.some((o) => o.result || brStarted(o, now)), word = { W: "Won", L: "Lost", P: "Push (stake refunded)" };
  const html = rows.map((o, i) => topAlphaRow(o, i, D.lineBy, { at: true, aux: graded ? (resultChip(o.result, `${word[o.result]} at the flagged price`) || `<span class="muted" title="Not graded yet">–</span>`) : "" })).join("");
  return blCard(id, `${blHead(title, link)}${pl}${cap}<div class="ca-br-rows">${html}</div>`, "ca-br-ta");
}

// Betting Splits: the games of the period with a captured split for a market, the selected game's tickets and money. Hidden (empty string) when
// the period has no splits at all; MLB / NBA show an honest empty state.
function brSplits(D) {
  const id = "board-betting-splits", title = "Betting Splits";
  if (!blLive(D)) return blEmpty(id, title, `${blNoModel(D)} No betting splits are captured for ${boardName(D.sport)}.`);
  if (!D.right) return "";
  const per = D.period, games = boardGames(D), splits = D.right.splits;
  const by = Object.fromEntries(BR_SPLIT_TABS.map(([k]) => [k, games.filter((g) => brSplitSides(splits, g.game_pk, k))]));
  const tabs = BR_SPLIT_TABS.filter(([k]) => by[k].length);
  if (!tabs.length) return "";
  const st = brState().sp, mine = st.for === blKey(per), tab = tabs.some(([k]) => k === (mine ? st.tab : "")) ? st.tab : tabs[0][0], list = by[tab];
  // the game: the user's pick, else the game of the highest-alpha edge that has splits, else the first game of the period
  const topPk = brTopOpps(D).map((o) => String(o.game_pk)).find((pk) => list.some((g) => String(g.game_pk) === pk));
  const g = list.find((x) => mine && String(x.game_pk) === String(st.game)) || list.find((x) => String(x.game_pk) === topPk) || list[0];
  const sides = brSplitSides(splits, g.game_pk, tab), isTotal = tab === "total", col = gameTeamColors(g.away, g.home, D.sport);
  const names = isTotal ? ["Over", "Under"] : [shortTeam(g.away, D.sport), shortTeam(g.home, D.sport)], abbr = isTotal ? ["Over", "Under"] : [teamShort(g.away, D.sport), teamShort(g.home, D.sport)];
  const colors = isTotal ? ["#2563EB", "#F59E0B"] : [col.away, col.home], sum = sides[0].pct + sides[1].pct || 1, lead = sides[0].pct >= sides[1].pct ? 0 : 1;
  const pc = (x) => `${Math.round(x)}%`;
  const bar = (label, a, b) => `<div class="ca-br-bar"><span class="ca-br-bl">${label}</span><div class="ca-br-b" role="img" aria-label="${ctxEsc(`${label}: ${abbr[0]} ${pc(a)}, ${abbr[1]} ${pc(b)}`)}"><i style="width:${(a / ((a + b) || 1) * 100).toFixed(1)}%;background:${ctxEsc(colors[0])}"></i><i style="flex:1;background:${ctxEsc(colors[1])}"></i></div>
    <div class="ca-br-bv"><span>${pc(a)} ${ctxEsc(abbr[0])}</span><span>${pc(b)} ${ctxEsc(abbr[1])}</span></div></div>`;
  const money = sides[0].cash != null && sides[1].cash != null ? bar("% of Money", sides[0].cash, sides[1].cash) : "";
  const cap = sides[0].at ? `${shortDate(etDateStr(sides[0].at))}, ${timeET(sides[0].at)}` : "";
  const opts = list.map((x) => [String(x.game_pk), `${teamShort(x.away, D.sport)} @ ${teamShort(x.home, D.sport)}`]);
  const tip = "Public betting splits from the latest capture: the share of tickets (bets) and of money (handle) on each side. Splits are captured near kickoff, so a game can have none.";
  const legend = [0, 1].map((i) => `<div class="ca-br-lg"><i style="background:${ctxEsc(colors[i])}"></i><span class="ca-br-lgn" title="${ctxEsc(names[i])}">${ctxEsc(names[i])}</span><b>${pc(sides[i].pct)}</b></div>`).join("");
  const body = `${blHead(title, infoTip(tip))}<div class="ca-br-tabs">${pills("br-sp", tabs.map(([k, l]) => [k, l]), tab)}</div>
    <div class="ca-br-game">${brMsel("sp-game", opts, String(g.game_pk))}</div>
    <div class="ca-br-split"><div class="ca-br-donut">${donut(sides[0].pct / sum, { size: 120, stroke: 26, color: colors[0], track: colors[1], cap: "butt" })}<div class="ca-br-dc"><b>${pc(sides[lead].pct)}</b><span>${ctxEsc(abbr[lead])}</span></div></div><div class="ca-br-leg">${legend}</div></div>
    ${bar("% of Tickets", sides[0].pct, sides[1].pct)}${money}${cap ? `<p class="ca-bl-cap ca-br-asof">Captured ${ctxEsc(cap)}</p>` : ""}`;
  return blCard(id, body, "ca-br-sp");
}

function brProjections(D) {
  const id = "board-model-projections", title = "Model Projections", L = D.left;
  if (!blLive(D)) return blEmpty(id, title, `${blNoModel(D)} No graded ${boardName(D.sport)} picks to bucket.`);
  if (!L || L.failed.record) return blFailed(id, title);
  const per = D.period, st = brState().mp, mine = st.for === blKey(per), sc = blScopeOf(D, mine ? st.scope : null), market = BR_PROJ_MARKETS.some(([k]) => k === st.market) ? st.market : "spread";
  const d = brProjData(L, market, sc.from, sc.to), def = BR_PROJ_MARKETS.find(([k]) => k === market);
  const tip = `Graded ${boardName(D.sport)} picks of the published record, grouped by how far the model's number was from the market's (${def[3] === "pp" ? "the model's win probability minus the market's no-vig closing probability, in percentage points (the closing price with the bookmaker's margin removed using the other side's price; a game with only one side's closing price uses the vigged price, which makes the edge look smaller)" : "the model's line minus the market's line, in points"}; positive = the model likes the market favourite${market === "total" ? " / the Over" : ""} more than the market does). Hit rate = picks won / (won + lost); a push is not decided. Units are not shown: the stored P&L is $10 a bet in daily totals, so there is no per-pick profit.`;
  const head = `${blHead(title, `${brMsel("mp-market", BR_PROJ_MARKETS.map(([k, l]) => [k, l]), market)}${infoTip(tip)}`)}<div class="ca-bl-scope">${pills("br-mp", [["period", ctxEsc(blPeriodWord(per))], ["season", "Season"]], sc.key)}</div><p class="ca-bl-cap">${ctxEsc(sc.caption)}</p>`;
  if (!d.total.n) return blCard(id, `${head}${emptyMsg(d.picks ? `${d.picks} graded ${def[1]} pick${d.picks > 1 ? "s" : ""} in ${sc.word} have no market ${market === "moneyline" ? "closing price" : "line"} to measure an edge against.` : blNoPicksMsg(D, L, sc))}`, "ca-br-mp");
  const rec = (b) => `${b.w}-${b.l}${b.p ? `-${b.p}` : ""}`, hit = (b) => (b.pct == null ? "—" : `<span class="${b.pct > 0.5 ? "pos" : b.pct < 0.5 ? "neg" : ""}" title="${ctxEsc(`${rec(b)} (${b.n} games)`)}">${(b.pct * 100).toFixed(1)}%</span>`);
  const lab = (b, i) => `${b.label}${i === 0 || i === BR_BANDS.length - 1 ? ` ${def[3]}` : ""}`;
  const rows = d.bands.map((b, i) => `<tr${b.n ? "" : ' class="muted"'}><td>${ctxEsc(lab(b, i))}</td><td>${b.n}</td><td>${hit(b)}</td></tr>`).join("");
  const foot = `<tr class="ca-br-tot"><td>All graded</td><td>${d.total.n}</td><td>${hit(d.total)}</td></tr>`;
  const note = d.skipped ? `<p class="ca-bl-cap">${d.skipped} more pick${d.skipped > 1 ? "s have" : " has"} no market ${market === "moneyline" ? "closing price" : "line"} and ${d.skipped > 1 ? "are" : "is"} not bucketed.</p>` : "";
  return blCard(id, `${head}<table class="ca-table ca-br-table"><thead><tr><th>Range</th><th># of Games</th><th>Hit Rate</th></tr></thead><tbody>${rows}${foot}</tbody></table>${note}`, "ca-br-mp");
}

BOARD_CARDS["board-top-alpha"] = brTopAlpha;
BOARD_CARDS["board-betting-splits"] = brSplits;
BOARD_CARDS["board-model-projections"] = brProjections;

/* ── interaction ──────────────────────────────────────────────────────── */
// Redraw one card from the loaded data (a pill / select change needs no refetch). A card that hides itself (no splits) has no element to redraw.
function brClick(e) {
  const pill = e.target && e.target.closest ? e.target.closest("[data-pill]") : null, D = window.__caBoardData;
  if (!pill || !D || D.view !== "games") return;
  const pk = blKey(D.period);
  if (pill.dataset.pill === "br-ta") { brSetTop(pill.dataset.key); blRedraw("board-top-alpha", brTopAlpha); }
  else if (pill.dataset.pill === "br-sp") { brSetSplit({ tab: pill.dataset.key, game: "" }, pk); blRedraw("board-betting-splits", brSplits); }
  else if (pill.dataset.pill === "br-mp") { brSetProj({ scope: pill.dataset.key }, pk); blRedraw("board-model-projections", brProjections); }
}
function brChange(e) {
  const t = e.target, D = window.__caBoardData;
  if (!t || !t.matches || !D || D.view !== "games" || !t.matches("[data-br]")) return;
  const pk = blKey(D.period);
  if (t.dataset.br === "sp-game") { brSetSplit({ game: t.value }, pk); blRedraw("board-betting-splits", brSplits); }
  else if (t.dataset.br === "mp-market") { brSetProj({ market: t.value }, pk); blRedraw("board-model-projections", brProjections); }
}
if (typeof document !== "undefined" && document.addEventListener && !window.__caBRWired) { window.__caBRWired = true; document.addEventListener("click", brClick); document.addEventListener("change", brChange); }
