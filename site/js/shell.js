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
}
