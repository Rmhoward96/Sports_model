/* Site shell: header + nav, avatar menu, watchlist (per browser), page titles. */
const NAV_ITEMS = [["dashboard", "Dashboard", "index.html"], ["mlb", "MLB", "mlb.html"], ["nfl", "NFL", "nfl.html"],
  ["nba", "NBA", "nba.html"], ["cfb", "CFB", "cfb.html"], ["ev", "+EV", "ev.html"], ["track", "Track Record", "track-record.html"]];
const ICON_SEARCH = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`;

function siteHeader(activePage) {
  const items = NAV_ITEMS.map(([key, label, href]) =>
    `<a data-nav="${key}" href="${href}"${key === activePage ? ' aria-current="page" class="active"' : ""}>${label}</a>`).join("");
  return `<header class="ca-header"><div class="ca-header-inner">
    <a class="ca-wordmark" href="index.html" aria-label="CappingAlpha home">Capping<span class="ca-alpha">α</span>lpha</a>
    <nav class="ca-nav" aria-label="Primary">${items}</nav>
    <div class="ca-header-tools">
      <button class="ca-icon-btn" data-search-open aria-label="Search">${ICON_SEARCH}</button>
      <div class="ca-avatar-wrap"><button class="ca-avatar" data-avatar aria-haspopup="menu" aria-expanded="false">R</button>
        <div class="ca-menu" role="menu" hidden>
          <a role="menuitem" href="rankings.html">Power Rankings</a>
          <a role="menuitem" href="index.html#watchlist">Watchlist</a>
          <a role="menuitem" href="settings.html">Settings</a>
        </div></div>
    </div></div></header>`;
}

function pageTitle(title, subtitle, rightHtml = "") {
  return `<div class="ca-title-row"><div class="ca-title"><h1>${title}</h1><p>${subtitle}</p></div><div class="ca-title-right">${rightHtml}</div></div>`;
}

/* Watchlist: {games:[], teams:[], players:[]} under ca-watchlist; in-memory fallback. */
const WATCH_KEY = "ca-watchlist";
let watchMem = null;
function watchlistGet() {
  if (watchMem) return watchMem;
  try {
    const raw = JSON.parse(localStorage.getItem(WATCH_KEY) || "null");
    const ok = raw && typeof raw === "object" ? raw : {};
    return { games: [...(ok.games || [])], teams: [...(ok.teams || [])], players: [...(ok.players || [])] };
  } catch { return { games: [], teams: [], players: [] }; }
}
function watchlistSave(w) {
  watchMem = w;
  try { localStorage.setItem(WATCH_KEY, JSON.stringify(w)); watchMem = null; } catch { /* keep in memory */ }
}
function watchlistHas(kind, id) { return watchlistGet()[kind].includes(String(id)); }
function watchlistToggle(kind, id) {
  const w = watchlistGet(), list = w[kind], k = String(id), i = list.indexOf(k);
  if (i >= 0) list.splice(i, 1); else list.push(k);
  watchlistSave(w);
  return i < 0;
}
function starAttrs(on, label) {
  return { cls: `ca-star${on ? " on" : ""}`, text: on ? "★" : "☆", label: `${on ? "Remove" : "Add"} ${label} ${on ? "from" : "to"} watchlist` };
}
function starButton(kind, id, label) {
  const on = watchlistHas(kind, id), a = starAttrs(on, ctxEsc(label));
  return `<button class="${a.cls}" data-star-kind="${kind}" data-star-id="${ctxEsc(id)}" data-star-label="${ctxEsc(label)}" aria-pressed="${on}" aria-label="${a.label}">${a.text}</button>`;
}
// Toggle one star in the watchlist, then sync EVERY button for that kind+id in the document
// (a game can be starred in the Slate and the Watchlist at once). Returns the new state.
function starToggle(kind, id, doc = document) {
  const on = watchlistToggle(kind, id);
  doc.querySelectorAll("[data-star-kind][data-star-id]").forEach((b) => {
    if (b.dataset.starKind !== kind || b.dataset.starId !== String(id)) return;
    b.classList.toggle("on", on); b.textContent = on ? "★" : "☆"; b.setAttribute("aria-pressed", String(on));
    b.setAttribute("aria-label", starAttrs(on, b.dataset.starLabel || String(id)).label);
  });
  if (typeof CustomEvent === "function") doc.dispatchEvent(new CustomEvent("ca-watchlist-change", { detail: { kind, id: String(id), on } }));
  return on;
}
// The one delegated click handler (capture phase on the document): a star never triggers a row / link handler.
function starDelegate(e, doc = document) {
  const b = e.target && e.target.closest ? e.target.closest("[data-star-kind][data-star-id]") : null;
  if (!b) return false;
  e.preventDefault(); e.stopPropagation();
  starToggle(b.dataset.starKind, b.dataset.starId, doc);
  return true;
}
// Kept for compatibility: stars are handled by the one delegated listener bound in wireShell.
function wireStars() {}
// Rows with data-href are clickable (any page); clicks on links, buttons and stars keep their own behaviour.
function rowDelegate(e, go = (url) => { location.href = url; }) {
  const t = e.target, tr = t && t.closest ? t.closest("tr[data-href]") : null;
  if (!tr || t.closest("a,button")) return false;
  go(tr.dataset.href);
  return true;
}
let shellDocBound = false, shellStarsBound = false, shellRowsBound = false;
function wireShell() {
  const av = document.querySelector("[data-avatar]"), menu = document.querySelector(".ca-menu");
  if (av && menu) {
    av.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; av.setAttribute("aria-expanded", String(!menu.hidden)); });
  }
  if (!shellDocBound) {  // render() re-runs every refresh: bind the outside-click closer once, against the live menu
    shellDocBound = true;
    document.addEventListener("click", () => {
      const m = document.querySelector(".ca-menu"), a = document.querySelector("[data-avatar]");
      if (m) m.hidden = true;
      if (a) a.setAttribute("aria-expanded", "false");
    });
  }
  if (!shellRowsBound) {  // one delegated row-click handler for every page
    shellRowsBound = true;
    document.addEventListener("click", (e) => rowDelegate(e));
  }
  if (!shellStarsBound) {  // one delegated listener for every star on every page, present or future markup
    shellStarsBound = true;
    document.addEventListener("click", starDelegate, true);
  }
  wireSearch();   // the header's search button (bound once, against the live button)
}

/* ── Search: the header's magnifier opens an overlay over teams, matchups and prop players ─────────────
   searchIndex / searchQuery / searchResultsHtml / searchKeyAction are pure. The data behind the index loads
   lazily on the first open (predictions of the live sports + loadOpportunities, from data.js / app.js) and never
   blocks a page render. Depends at call time on data.js (gameHref, shortMatchup, kickLabel, SPORT_NAME, SPORTS,
   LIVE_SPORTS, loadOpportunities, matchupSides) and app.js (predictions, ctxEsc, REFRESH_MS). */
const SEARCH_MAX = 8;
// [{label, sub, href}]: per game its matchup and both teams (-> the game page), per prop opportunity its player.
function searchIndex(preds, opps) {
  const out = [], seen = new Set();
  const add = (label, sub, href) => {
    if (!label || !href) return;
    const k = `${label}\u0001${sub}\u0001${href}`;
    if (seen.has(k)) return;
    seen.add(k); out.push({ label, sub, href });
  };
  for (const r of preds || []) {
    if (!r || r.game_pk == null || !r.sport) continue;
    const away = String(r.away_team_name || ""), home = String(r.home_team_name || "");
    if (!away && !home) continue;
    const lg = SPORT_NAME[r.sport] || String(r.sport).toUpperCase(), href = gameHref(r.sport, r.game_pk), short = shortMatchup(away, home, r.sport);
    add(short, [lg, `${away} @ ${home}`, kickLabel(r.commence_time)].filter(Boolean).join(" · "), href);
    add(away, `${lg} · ${short}`, href); add(home, `${lg} · ${short}`, href);
  }
  for (const o of opps || []) {
    if (!o || o.kind !== "prop" || !o.playerName || o.game_pk == null || !o.sport) continue;
    const [a, h] = matchupSides(o.matchup);
    add(o.playerName, [SPORT_NAME[o.sport] || String(o.sport).toUpperCase(), o.marketLabel || o.market, a || h ? shortMatchup(a, h, o.sport) : ""].filter(Boolean).join(" · "), gameHref(o.sport, o.game_pk));
  }
  return out;
}
// Case-insensitive substring over label + sub; the first SEARCH_MAX matches in index order.
function searchQuery(index, q) {
  const n = String(q || "").trim().toLowerCase();
  if (!n || !Array.isArray(index)) return [];
  return index.filter((r) => `${r.label} ${r.sub}`.toLowerCase().includes(n)).slice(0, SEARCH_MAX);
}
function searchResultsHtml(results, active, q = "") {
  if (!results || !results.length) {
    const t = String(q || "").trim();
    return `<p class="ca-find-empty">${t ? `No matches for “${ctxEsc(t)}”` : "Search teams, games and players"}</p>`;
  }
  return results.map((r, i) => `<a class="ca-find-item${i === active ? " on" : ""}" href="${ctxEsc(r.href)}" data-search-i="${i}" role="option"><b>${ctxEsc(r.label)}</b><small>${ctxEsc(r.sub)}</small></a>`).join("");
}
// Keyboard model: arrows move (wrapping), Enter opens the active result, Escape closes.
function searchKeyAction(key, results, active) {
  const n = (results || []).length;
  if (key === "Escape") return { active, close: true };
  if ((key === "ArrowDown" || key === "ArrowUp") && n) return { active: (active + (key === "ArrowDown" ? 1 : n - 1)) % n };
  if (key === "Enter" && results && results[active]) return { active, href: results[active].href };
  return { active };
}
const searchOverlayHtml = () => `<div class="ca-find" id="ca-find" hidden><div class="ca-find-box" role="dialog" aria-label="Search">
  <div class="ca-find-field">${ICON_SEARCH}<input class="ca-find-input" type="search" autocomplete="off" spellcheck="false" placeholder="Search teams, games and players" aria-label="Search teams, games and players"><span class="ca-find-esc">Esc</span></div>
  <div class="ca-find-results" role="listbox">${searchResultsHtml([], 0, "")}</div></div></div>`;

const searchState = { index: null, at: 0, loading: null, active: 0, results: [] };
function searchLoad() {
  if (searchState.index && Date.now() - searchState.at < REFRESH_MS) return Promise.resolve(searchState.index);
  if (!searchState.loading) {
    searchState.loading = (async () => {
      try {
        const [predsBy, opps] = await Promise.all([Promise.all(LIVE_SPORTS.map((s) => predictions(s).catch(() => []))), loadOpportunities().catch(() => [])]);
        searchState.index = searchIndex(predsBy.flat(), opps); searchState.at = Date.now();
        return searchState.index;
      } catch { return searchState.index || []; }
      finally { searchState.loading = null; }
    })();
  }
  return searchState.loading;
}
function searchOverlay(doc = document) {
  let el = doc.getElementById("ca-find");
  if (el) return el;
  const wrap = doc.createElement("div");
  wrap.innerHTML = searchOverlayHtml();
  el = wrap.firstElementChild;
  doc.body.appendChild(el);
  const input = el.querySelector(".ca-find-input"), list = el.querySelector(".ca-find-results");
  const show = () => {
    if (!searchState.index) { list.innerHTML = `<p class="ca-find-empty">${input.value.trim() ? "Loading…" : "Search teams, games and players"}</p>`; return; }
    searchState.results = searchQuery(searchState.index, input.value);
    searchState.active = 0;
    list.innerHTML = searchResultsHtml(searchState.results, 0, input.value);
  };
  el._refresh = show;
  input.addEventListener("input", show);
  input.addEventListener("keydown", (e) => {
    const a = searchKeyAction(e.key, searchState.results, searchState.active);
    if (a.close) { e.preventDefault(); searchClose(doc); return; }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") e.preventDefault();
    if (a.active !== searchState.active) {
      searchState.active = a.active;
      list.querySelectorAll(".ca-find-item").forEach((n, i) => n.classList.toggle("on", i === a.active));
      const on = list.querySelector(".ca-find-item.on");
      if (on && on.scrollIntoView) on.scrollIntoView({ block: "nearest" });
    }
    if (a.href) { e.preventDefault(); searchClose(doc); location.href = a.href; }
  });
  return el;
}
function searchOpen(doc = document) {
  const el = searchOverlay(doc), input = el.querySelector(".ca-find-input");
  el.hidden = false; input.focus();
  searchLoad().then(() => el._refresh && el._refresh());
}
function searchClose(doc = document) {
  const el = doc.getElementById("ca-find");
  if (el) { el.hidden = true; const i = el.querySelector(".ca-find-input"); if (i) i.value = ""; if (el._refresh) el._refresh(); }
}
let searchBound = false;
function wireSearch(doc = document) {
  if (searchBound) return;
  searchBound = true;
  doc.addEventListener("click", (e) => {
    const t = e.target;
    if (!t || !t.closest) return;
    if (t.closest("[data-search-open]")) { searchOpen(doc); return; }
    const el = doc.getElementById("ca-find");
    if (el && !el.hidden && (t.closest(".ca-find-item") || !t.closest(".ca-find-box"))) searchClose(doc);   // a result link navigates natively
  });
  doc.addEventListener("keydown", (e) => {
    const el = doc.getElementById("ca-find");
    if (e.key === "Escape" && el && !el.hidden) searchClose(doc);
  });
}
