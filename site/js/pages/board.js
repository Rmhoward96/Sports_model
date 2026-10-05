/* Board pages (nfl.html / cfb.html / mlb.html / nba.html): every game of the sport with the model's projection beside
   the market. Real data only: every card has an empty state and no number is ever invented.
   One row per game (predictions_current): Time, Matchup (logos + the shell's star), Projected Score, Model Spread vs
   Market (the HOME line), Model Total vs Market, Best Opportunity (the game's highest-alpha opportunity: a +EV pick with a
   confidence tier, R9) and a link to the game page. The "Model" numbers are the model's own projection (R18, never the
   Pinnacle true_prob): the projected score, the spread from pred_margin and the total from pred_total when the row has
   them, else from the projected score (predictions_current only carries the scores), against market_spread / market_total.
   NFL / CFB add a "Games · Power Rankings" pill group: Power Rankings renders the legacy sortable rankingsTable in place
   (app.js: rankingsCard, wireRankings). MLB / NBA show SPORT_STATUS in a card (plus any rows still in the table).
   The range pill (All / Today / This Week) and the view pill live in window.__caBoard AND in the URL (?range=&view=,
   non-default values only) and survive the 5-minute re-render; a range change redraws from the cached load
   (window.__caBoardData), a view change refetches (it needs different data).
   Depends on app.js (predictions, rankingsLoad, rankingsCard, wireRankings, etDateStr, ctxEsc, logoImg, r05, render),
   metrics.js (numOrNull), ui.js, shell.js (pageTitle, starButton) and data.js (slateRows, gameDate, safeCard, ...). */
const BOARD_RANGES = [["all", "All"], ["today", "Today"], ["week", "This Week"]];
const BOARD_VIEWS = [["games", "Games"], ["rankings", "Power Rankings"]];
const BOARD_RANKED = ["nfl", "cfb"];            // the sports that publish power rankings
const BOARD_DEFAULT = { range: "all", view: "games" };

/* ── pure helpers ─────────────────────────────────────────────────────── */
function boardParse(search, sport) {
  let p;
  try { p = new URLSearchParams(search || ""); } catch { p = new URLSearchParams(""); }
  const range = p.get("range"), view = p.get("view");
  return { range: BOARD_RANGES.some(([k]) => k === range) ? range : BOARD_DEFAULT.range,
    view: view === "rankings" && BOARD_RANKED.includes(sport) ? "rankings" : BOARD_DEFAULT.view };
}
// Only non-default values: an untouched page has a clean URL.
function boardQuery(s) {
  const p = new URLSearchParams();
  for (const k of ["range", "view"]) if (s[k] !== BOARD_DEFAULT[k]) p.set(k, s[k]);
  return p.toString();
}
// today / this week (today through +6 days) by ET calendar day; a row with no date matches no range but All.
function boardFilter(rows, range, today) {
  const list = (rows || []).filter(Boolean);
  if (range !== "today" && range !== "week") return list;
  const end = addDays(today, range === "today" ? 0 : 6);
  return list.filter((r) => { const d = gameDate(r); return !!d && d >= today && d <= end; });
}
// The model's projection of one game: score (away, home), spread as the HOME line (negative = home favoured) and total,
// each null when the row has no usable number, next to the market's. pred_margin / pred_total win over the projected score.
function boardModel(r) {
  const p = r || {}, h = numOrNull(p.pred_home_score), a = numOrNull(p.pred_away_score);
  const margin = numOrNull(p.pred_margin) ?? (h != null && a != null ? h - a : null);
  const total = numOrNull(p.pred_total) ?? (h != null && a != null ? h + a : null);
  return { away: a == null ? null : Math.round(a), home: h == null ? null : Math.round(h),
    spread: margin == null ? null : r05(-margin) || 0, total: total == null ? null : r05(total),
    mktSpread: numOrNull(p.market_spread), mktTotal: numOrNull(p.market_total) };
}

/* ── state <-> URL ────────────────────────────────────────────────────── */
function boardState(sport) {
  if (!window.__caBoard || window.__caBoard.sport !== sport) {
    let q = "";
    try { q = location.search; } catch { q = ""; }
    window.__caBoard = { sport, ...boardParse(q, sport) };
  }
  return window.__caBoard;
}
function boardSet(sport, key, value) {
  const s = boardState(sport);
  s[key] = value;
  try {
    const u = new URL(location.href);
    u.search = boardQuery(s);
    history.replaceState(null, "", u.toString());
  } catch { /* keep the current URL */ }
}

/* ── data ─────────────────────────────────────────────────────────────── */
async function boardLoad(sport, view) {
  if (view === "rankings") return { sport, view, R: await rankingsLoad(sport) };
  const live = LIVE_SPORTS.includes(sport);
  const [preds, oppsAll, lineBy] = await Promise.all([
    predictions(sport).catch(() => []),
    live ? loadOpportunities().catch(() => []) : [],
    live ? evBestLines().catch(() => new Map()) : new Map(),
  ]);
  const nowMs = Date.now();
  return { sport, view, nowMs, today: etDateStr(new Date(nowMs).toISOString()), preds: preds || [], lineBy,
    opps: (oppsAll || []).filter((o) => o.tier && o.sport === sport) };       // R9: an opportunity is a pick with a tier
}

/* ── cards ────────────────────────────────────────────────────────────── */
const boardName = (sport) => SPORT_NAME[sport] || String(sport).toUpperCase();
const boardRangeEmpty = (range, name) => (range === "today" ? `No ${name} games today.` : range === "week" ? `No ${name} games in the next 7 days.` : `No ${name} games on the board right now.`);
const boardDash = `<span class="muted">—</span>`;

function boardBar(D) {
  const s = boardState(D.sport);
  const view = BOARD_RANKED.includes(D.sport) ? pills("board-view", BOARD_VIEWS, s.view) : "";
  const range = s.view === "games" && (LIVE_SPORTS.includes(D.sport) || D.preds.length) ? pills("board-range", BOARD_RANGES, s.range) : "";   // nothing to filter on a paused sport
  return `<div class="ca-board-bar" id="board-bar">${view}${view && range ? `<span class="ca-board-sep"></span>` : ""}${range}</div>`;
}

function boardRow(g, D) {
  const m = boardModel(g.pred), label = shortMatchup(g.away, g.home, g.sport), homeName = shortTeam(g.home, g.sport);
  const mkt = (txt) => `<small>Mkt ${txt}</small>`;
  const score = m.away != null && m.home != null
    ? `<b title="${ctxEsc(`${g.away} ${m.away}, ${g.home} ${m.home}`)}">${m.away}–${m.home}</b>` : boardDash;
  const spread = `${m.spread != null ? `<b>${ctxEsc(homeName)} ${modelLineStr(m.spread)}</b>` : boardDash}${mkt(m.mktSpread != null ? lineStr(m.mktSpread) : "—")}`;
  const total = `${m.total != null ? `<b>${m.total.toFixed(1)}</b>` : boardDash}${mkt(m.mktTotal != null ? m.mktTotal.toFixed(1) : "—")}`;
  let best = boardDash;
  if (g.opp) {
    const t = oppSlateLabel(g.opp, D.lineBy);
    best = `<span class="ca-bd-opp"><span class="ca-ell ca-bd-mkt" title="${ctxEsc(t)}">${ctxEsc(t)}</span><span class="ca-bd-odds">${bookBadge(g.opp.book)}${oddsStr(g.opp.odds)}</span>${alphaCell(g.opp.alpha)}${confPill(g.opp.tier)}</span>`;
  }
  return `<tr data-href="${gameHref(g.sport, g.game_pk)}"><td class="ca-bd-time">${kickLabel(g.commence)}</td>
    <td><span class="ca-team ca-bd-pair" title="${ctxEsc(label)}">${starButton("games", g.game_pk, label)}${logoImg(g.away, g.sport)}${logoImg(g.home, g.sport)}<span class="ca-ell">${ctxEsc(label)}</span></span></td>
    <td>${score}</td><td class="ca-bd-model">${spread}</td><td class="ca-bd-model">${total}</td><td>${best}</td><td class="ca-bd-go">${goLink(g.sport, g.game_pk)}</td></tr>`;
}

function boardGamesCard(D) {
  const s = boardState(D.sport), name = boardName(D.sport);
  const rows = slateRows(boardFilter(D.preds, s.range, D.today), D.opps);
  const body = !rows.length ? emptyMsg(boardRangeEmpty(s.range, name))
    : `<div class="ca-table-wrap"><table class="ca-table ca-board-table"><thead><tr><th>Time (ET)</th><th>Matchup</th><th>Projected Score</th><th>Model Spread vs Market</th><th>Model Total vs Market</th><th>Best Opportunity</th><th></th></tr></thead><tbody>${rows.map((g) => boardRow(g, D)).join("")}</tbody></table></div>`;
  return `<section class="ca-card ca-board-card" id="board-games"><div class="ca-card-head"><h2>${name} Games</h2><p>${rows.length} ${rows.length === 1 ? "game" : "games"} · the model's projection vs the market, spreads from the home side.</p></div>${body}</section>`;
}

// MLB / NBA: why there is nothing to show (SPORT_STATUS).
function boardStatusCard(D) {
  return `<section class="ca-card ca-board-card" id="board-status">${emptyMsg(SPORT_STATUS[D.sport] || `No ${boardName(D.sport)} games on the board right now.`)}</section>`;
}

function boardRankingsCard(D) { return rankingsCard(D.sport, D.R.rows, D.R.st); }

// The page body under the bar: the cards of the selected view, each isolated.
function boardBody(D) {
  if (D.view === "rankings") return safeCard("Power Rankings", boardRankingsCard, D, "ca-card ca-rk-card", "rk-card");
  const live = LIVE_SPORTS.includes(D.sport);
  return `${live ? "" : safeCard("Status", boardStatusCard, D, "ca-card ca-board-card", "board-status")}${live || D.preds.length ? safeCard("Games", boardGamesCard, D, "ca-card ca-board-card", "board-games") : ""}`;
}

/* ── page ─────────────────────────────────────────────────────────────── */
async function buildBoardPage(sport) {
  const s = boardState(sport), D = await boardLoad(sport, s.view);
  window.__caBoardData = D;
  return `<div class="ca-board" data-sport="${ctxEsc(sport)}">${pageTitle(`${boardName(sport)} Board`, "Every game with the model's projection and best market.")}${boardBar(D)}${boardBody(D)}</div>`;
}

// Re-draw the pill bar and the games card from the cached load (a range change needs no refetch).
function boardRedraw() {
  const D = window.__caBoardData;
  if (!D || D.view !== "games") return;
  const bar = document.getElementById("board-bar"), games = document.getElementById("board-games");
  if (bar) bar.outerHTML = boardBar(D);
  if (games) games.outerHTML = safeCard("Games", boardGamesCard, D, "ca-card ca-board-card", "board-games");
}

function wireBoardPage() {
  const root = document.querySelector(".ca-board");
  if (!root) return;
  root.addEventListener("click", (e) => {
    const pill = e.target.closest("[data-pill]"), D = window.__caBoardData;
    if (!pill || !D) return;
    const k = pill.dataset.key;
    if (pill.dataset.pill === "board-range") { boardSet(D.sport, "range", k); boardRedraw(); }
    else if (pill.dataset.pill === "board-view" && boardState(D.sport).view !== k) { boardSet(D.sport, "view", k); render(); }
  });
  wireRankings();   // the Power Rankings view's sort headers + conference select (no-op in the Games view)
  // Stars and row clicks are handled by shell.js's delegated listeners.
}
