# CappingAlpha Site Redesign — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the CappingAlpha site to match the user's four mockups (Dashboard, game page, +EV, Track Record) with a light design system, using only data that exists today; panels needing Phase B data stay hidden.

**Architecture:** The static site (vanilla JS, no build) moves into this repo under `site/` so it is versioned and testable; `scripts/sync_site.sh` copies it to `~/Desktop/CappingAlpha` for the user's usual Cloudflare Pages redeploy. The 2,838-line `site/app.js` stays as the legacy data/section layer; new code lives in focused classic scripts (`site/js/*.js`) that share globals and load in order before `site/js/boot.js`. Pure logic (metrics, Alpha Score, chart/markup builders) is unit-tested with `node --test` by loading the scripts into a `vm` context.

**Tech Stack:** HTML + vanilla JS (classic scripts, ES2022), CSS custom properties, inline SVG charts, Supabase PostgREST (anon read), Node 24 `node:test` for tests, Postgres view (one migration, run by the user).

**Spec:** `docs/superpowers/specs/2026-10-01-site-redesign-design.md`

## Global Constraints

- **Real data only.** Never render a number the database does not support. A panel whose data is missing or empty renders an empty state or is omitted — never placeholder values from the mockups.
- **Mockup fidelity:** layout, order, labels, and visual style of the four mockups (the user's images, summarized per task) — light theme, navy header, serif display headings, white cards, pill tabs, green/blue/amber confidence pills.
- **Design tokens (exact):** `--bg #F6F4EE`, `--card #FFFFFF`, `--line #E6E1D6`, `--ink #14202E`, `--muted #6B7280`, `--navy #0E2238`, `--green #1E8E4E`, `--green-tint #E3F4E8`, `--red #C8372D`, `--red-tint #FBE5E3`, `--blue #2563EB`, `--blue-tint #E3ECFD`, `--amber #B7791F`, `--amber-tint #FCF0D9`, `--radius 12px`.
- **Fonts:** Google Fonts `Libre Caslon Text` (700) for page titles and section heads; `Inter` (400/500/600/700) for everything else, `font-variant-numeric: tabular-nums` on numbers.
- **Alpha Score (exact):** `s_ev = clip(EV%/8,0,1)`, `s_edge = clip(edge_pp/8,0,1)`, `roi_s = roi_pct * n/(n+100)`, `s_track = 0.5 + clip(roi_s/20,-0.5,0.5)`, `alpha = round(40 + 30*s_ev + 20*s_edge + 10*s_track)`. Tiers: HIGH ≥ 85, STRONG 75–84, MEDIUM 65–74; < 65 is not an opportunity.
- **Units:** P&L rows (`prediction_pnl_daily`, `ev_pnl_daily`, `prediction_pnl`) are graded at $10 per bet → units = pnl / 10.
- **Track record scope:** only rows inside the published record (`inTrackRecord` / `track_record_start`); the pre-restart NFL archive is reachable only via an explicit "Archive" toggle.
- **NBA / MLB:** nav items present; NBA page/empty states say "NBA model not live yet"; MLB says "MLB model paused (last projections Aug 31, 2026)". Their counts are 0.
- **Accounts:** none. Watchlist in `localStorage` key `ca-watchlist`; settings stay in `ca-settings`. Every storage access is wrapped in try/catch and the page works without storage.
- **Keep existing features reachable:** parlays, props board, NFL sim widget, splits, matchup / history / trends sections, P&L trackers, prop accuracy tables, settings (books / unit / bankroll / Kelly / min EV), power rankings.
- **No new runtime dependencies or CDNs** beyond Google Fonts and the ESPN logo CDN already used.
- **Times** shown in ET; data auto-refreshes every 5 minutes without losing the user's tab / filter / sort state.
- **Mobile:** ≤ 620px — cards stack one per row, 16px side gutters, tables scroll horizontally inside their card, no horizontal page scroll.
- **Cache key:** every HTML page loads its scripts with `?v=20261002a` (bump the suffix letter on later deploys).

---

## File Structure

```
site/                                  # copy of ~/Desktop/CappingAlpha (Task 1)
  index.html nfl.html cfb.html mlb.html nba.html game.html ev.html
  track-record.html rankings.html settings.html     # script tags updated
  app.js              # legacy: data fetchers + legacy sections (boot lines removed)
  css/theme.css       # design system (Task 2)
  js/shell.js         # header, nav, avatar menu, watchlist store, search box (Tasks 2, 10)
  js/ui.js            # components + SVG charts, pure string builders (Task 3)
  js/metrics.js       # pure calculations incl. Alpha Score (Task 4)
  js/data.js          # new fetchers + the shared "opportunities" model (Task 5)
  js/pages/dashboard.js  ev.js  game.js  track.js  board.js   (Tasks 6–10)
  js/boot.js          # render dispatch + refresh (Task 1, extended per page)
  tests/load.mjs      # vm loader for classic scripts
  tests/*.test.mjs
  package.json        # "test": "node --test tests/"
scripts/sync_site.sh                   # site/ -> ~/Desktop/CappingAlpha (Task 1)
db/migration_site_redesign_a.sql       # line_moves_current view + index (Task 5)
```

Script load order on every page: `app.js`, `js/metrics.js`, `js/ui.js`, `js/shell.js`, `js/data.js`, `js/pages/*.js` (all five), `js/boot.js`.

---

### Task 1: Bring the site into the repo; boot split; test harness; sync script

**Files:**
- Create: `site/` (copy of `~/Desktop/CappingAlpha`), `site/js/boot.js`, `site/tests/load.mjs`, `site/tests/smoke.test.mjs`, `scripts/sync_site.sh`
- Modify: `site/app.js` (remove the 4 boot lines at the end), every `site/*.html` (script tags), `site/package.json`

**Interfaces:**
- Produces: `loadScripts(files, {page, storage}) -> context` in `site/tests/load.mjs` (a `vm` context whose globals are the scripts' top-level declarations; `context.document.body.dataset.page === page`). `site/js/boot.js` defines `async function boot()` that runs `injectStylesOnce(); mountGradientBackground(); render(); setInterval(...)` exactly as the removed lines did.

- [ ] **Step 1: Copy the site**

```bash
rsync -a --exclude .DS_Store ~/Desktop/CappingAlpha/ site/
ls site   # app.js cfb.html ev.html game.html index.html mlb.html nba.html nfl.html package.json public rankings.html settings.html styles.css track-record.html
```

- [ ] **Step 2: Write the vm loader and a failing smoke test**

`site/tests/load.mjs`:
```js
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const SITE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// Minimal browser stand-ins: enough for top-level script evaluation, not rendering.
function fakeDom(page) {
  const el = () => ({ innerHTML: "", style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} },
    appendChild() {}, setAttribute() {}, addEventListener() {}, querySelector: () => null, querySelectorAll: () => [] });
  return {
    body: { dataset: { page }, appendChild() {}, classList: { add() {}, remove() {} } },
    head: { appendChild() {} }, documentElement: el(),
    createElement: el, createElementNS: el, getElementById: () => null,
    querySelector: () => null, querySelectorAll: () => [], addEventListener() {},
  };
}

export function loadScripts(files, { page = "dashboard", storage = new Map() } = {}) {
  const localStorage = {
    getItem: (k) => (storage.has(k) ? storage.get(k) : null),
    setItem: (k, v) => storage.set(k, String(v)), removeItem: (k) => storage.delete(k),
  };
  const ctx = { console, URL, URLSearchParams, Intl, Date, Math, JSON, setTimeout, clearTimeout,
    setInterval: () => 0, fetch: async () => { throw new Error("no network in tests"); },
    localStorage, location: { search: "", href: "http://localhost/" }, history: { replaceState() {} } };
  ctx.window = ctx;
  ctx.__CA_TEST__ = true;            // js/boot.js skips boot() in tests
  ctx.document = fakeDom(page);
  vm.createContext(ctx);
  // Classic scripts share one global scope: concatenate so top-level const/let are visible to later files.
  const src = files.map((f) => fs.readFileSync(path.join(SITE, f), "utf8")).join("\n;\n");
  vm.runInContext(src + "\n;globalThis.__exports = { " + exportNames(src) + " };", ctx, { filename: files.join("+") });
  return Object.assign(ctx.__exports, { localStorage: ctx.localStorage });
}

// Every top-level function / const / let / class name, so tests can reach them.
function exportNames(src) {
  const names = new Set();
  for (const m of src.matchAll(/^(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|^(?:const|let|class)\s+([A-Za-z_$][\w$]*)/gm)) {
    names.add(m[1] || m[2]);
  }
  return [...names].join(", ");
}
```

`site/tests/smoke.test.mjs`:
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";

test("app.js loads without running the page (boot moved to js/boot.js)", () => {
  const g = loadScripts(["app.js", "js/boot.js"], { page: "dashboard" });
  assert.equal(typeof g.render, "function");
  assert.equal(typeof g.boot, "function");
  assert.equal(g.fmtOdds(120), "+120");
});
```

- [ ] **Step 3: Run it — expect FAIL** (`js/boot.js` missing)

Run: `cd site && node --test tests/`
Expected: FAIL — `ENOENT ... js/boot.js`

- [ ] **Step 4: Move the boot lines**

Delete the last 4 lines of `site/app.js` (`injectStylesOnce();`, `mountGradientBackground();`, `render();`, `if (page !== "settings") setInterval(render, REFRESH_MS); ...`). Create `site/js/boot.js`:
```js
/* Boot: runs last on every page, after app.js and the redesign scripts. */
async function boot() {
  injectStylesOnce();
  mountGradientBackground();
  render();
  if (page !== "settings") setInterval(render, REFRESH_MS);  // settings has no live data; a re-render would blur inputs mid-edit
}
if (typeof window !== "undefined" && window.document && !window.__CA_TEST__) boot();
```
(`loadScripts` sets `__CA_TEST__` on the context, which is also `window`, so `boot()` is skipped in tests.)

In every `site/*.html`, replace `<script src="app.js?v=20260930d"></script>` with:
```html
<script src="app.js?v=20261002a"></script><script src="js/boot.js?v=20261002a"></script>
```
`site/package.json`: add `"test": "node --test tests/"` to `scripts`.

- [ ] **Step 5: Run the test — expect PASS**

Run: `cd site && npm test`
Expected: 1 passing.

- [ ] **Step 6: Sync script**

`scripts/sync_site.sh`:
```bash
#!/usr/bin/env bash
# Copy site/ to the folder the user deploys to Cloudflare Pages from.
# Backs the target up first; tests and dev files are not deployed.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/site/"
DEST="${1:-$HOME/Desktop/CappingAlpha}"
[ -d "$DEST" ] && cp -R "$DEST" "${DEST}-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"
rsync -a --delete --exclude tests/ --exclude .DS_Store "$SRC" "$DEST/"
echo "synced site/ -> $DEST (backup kept next to it)"
```
`chmod +x scripts/sync_site.sh`. Verify with a throwaway target: `scripts/sync_site.sh /tmp/ca-sync-test && ls /tmp/ca-sync-test && test ! -d /tmp/ca-sync-test/tests`.

- [ ] **Step 7: Manual check** — serve `site/` (`cd site && python3 -m http.server 8765`) and confirm `index.html` renders exactly as the live site (dark theme, same content).

- [ ] **Step 8: Commit**
```bash
git add site scripts/sync_site.sh
git commit -m "chore(site): bring CappingAlpha into the repo; boot split; node test harness; sync script"
```

---

### Task 2: Design system + new header (light theme)

**Files:**
- Create: `site/css/theme.css`, `site/js/shell.js`, `site/tests/shell.test.mjs`
- Modify: every `site/*.html` (head: fonts + `css/theme.css`, drop `styles.css`; scripts: add `js/shell.js` before `js/boot.js`), `site/app.js` (`render()` uses `siteHeader()` instead of `nav()`; `mountGradientBackground` becomes a no-op), `site/js/boot.js`

**Interfaces:**
- Consumes: `page` (global from app.js), `getSettings()`.
- Produces (globals in `js/shell.js`):
  - `NAV_ITEMS` — `[["dashboard","Dashboard","index.html"],["mlb","MLB","mlb.html"],["nfl","NFL","nfl.html"],["nba","NBA","nba.html"],["cfb","CFB","cfb.html"],["ev","+EV","ev.html"],["track","Track Record","track-record.html"]]`
  - `siteHeader(activePage: string) -> string` (header bar: wordmark, nav, search button, avatar button + menu with Settings, Power Rankings, Watchlist)
  - `watchlist` store: `watchlistGet() -> {games:string[], teams:string[], players:string[]}`, `watchlistToggle(kind: "games"|"teams"|"players", id: string) -> boolean` (true = now starred), `watchlistHas(kind, id) -> boolean`; key `ca-watchlist`; storage failures fall back to an in-memory copy.
  - `starButton(kind, id, label) -> string` (☆/★ toggle; `data-star-kind`, `data-star-id`) and `wireStars(root)`.
  - `pageTitle(title: string, subtitle: string, rightHtml = "") -> string` (serif H1 + muted subtitle + right slot).

- [ ] **Step 1: Failing tests** — `site/tests/shell.test.mjs`:
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const load = (storage) => loadScripts(["app.js", "js/shell.js", "js/boot.js"], { page: "ev", storage });

test("header has the mockup nav in order with the active item marked", () => {
  const g = load();
  const html = g.siteHeader("ev");
  const labels = [...html.matchAll(/data-nav="([a-z]+)"/g)].map((m) => m[1]);
  assert.deepEqual(labels, ["dashboard", "mlb", "nfl", "nba", "cfb", "ev", "track"]);
  assert.match(html, /data-nav="ev"[^>]*aria-current="page"/);
  assert.match(html, /href="settings.html"/);
  assert.match(html, /href="rankings.html"/);
});

test("watchlist toggles and persists", () => {
  const storage = new Map();
  const g = load(storage);
  assert.equal(g.watchlistToggle("games", "401"), true);
  assert.equal(g.watchlistHas("games", "401"), true);
  assert.deepEqual(JSON.parse(storage.get("ca-watchlist")).games, ["401"]);
  assert.equal(g.watchlistToggle("games", "401"), false);
  assert.equal(g.watchlistHas("games", "401"), false);
});

test("watchlist survives a throwing storage", () => {
  const g = load();
  g.localStorage.setItem = () => { throw new Error("blocked"); };
  assert.equal(g.watchlistToggle("teams", "KC"), true);
  assert.equal(g.watchlistHas("teams", "KC"), true);
});
```

- [ ] **Step 2: Run — expect FAIL** (`siteHeader is not defined`)

- [ ] **Step 3: Implement `site/js/shell.js`**
```js
/* Site shell: header + nav, avatar menu, watchlist (per browser), page titles. */
const NAV_ITEMS = [["dashboard", "Dashboard", "index.html"], ["mlb", "MLB", "mlb.html"], ["nfl", "NFL", "nfl.html"],
  ["nba", "NBA", "nba.html"], ["cfb", "CFB", "cfb.html"], ["ev", "+EV", "ev.html"], ["track", "Track Record", "track-record.html"]];
const shellEsc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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
const ICON_SEARCH = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`;

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
function starButton(kind, id, label) {
  const on = watchlistHas(kind, id);
  return `<button class="ca-star${on ? " on" : ""}" data-star-kind="${kind}" data-star-id="${shellEsc(id)}" aria-pressed="${on}" aria-label="${on ? "Remove" : "Add"} ${shellEsc(label)} ${on ? "from" : "to"} watchlist">${on ? "★" : "☆"}</button>`;
}
function wireStars(root = document) {
  root.querySelectorAll("[data-star-kind]").forEach((b) => b.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    const on = watchlistToggle(b.dataset.starKind, b.dataset.starId);
    b.classList.toggle("on", on); b.textContent = on ? "★" : "☆"; b.setAttribute("aria-pressed", String(on));
  }));
}
function wireShell() {
  const av = document.querySelector("[data-avatar]"), menu = document.querySelector(".ca-menu");
  if (av && menu) {
    av.addEventListener("click", (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; av.setAttribute("aria-expanded", String(!menu.hidden)); });
    document.addEventListener("click", () => { menu.hidden = true; av.setAttribute("aria-expanded", "false"); });
  }
  wireStars(document);
}
```
(Search box behavior is Task 10; the button exists now and does nothing until then.)

- [ ] **Step 4: `site/css/theme.css`** — tokens and base components. Required selectors (each styled per the mockups):
```css
:root{--bg:#F6F4EE;--card:#FFFFFF;--line:#E6E1D6;--ink:#14202E;--muted:#6B7280;--navy:#0E2238;
  --green:#1E8E4E;--green-tint:#E3F4E8;--red:#C8372D;--red-tint:#FBE5E3;--blue:#2563EB;--blue-tint:#E3ECFD;
  --amber:#B7791F;--amber-tint:#FCF0D9;--radius:12px;--serif:"Libre Caslon Text",Georgia,serif;
  --sans:"Inter",system-ui,sans-serif;--shadow:0 1px 2px rgba(16,24,40,.04)}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 var(--sans)}
.num,td,.ca-stat-value{font-variant-numeric:tabular-nums}
a{color:inherit}
.ca-header{background:var(--navy);color:#fff}
.ca-header-inner{max-width:1440px;margin:0 auto;height:64px;display:flex;align-items:center;gap:40px;padding:0 32px}
.ca-wordmark{font:700 28px var(--serif);color:#fff;text-decoration:none;letter-spacing:-.5px}
.ca-alpha{font-style:italic}
.ca-nav{display:flex;gap:28px}
.ca-nav a{color:#fff;text-decoration:none;font-weight:600;font-size:15px;padding:20px 0;border-bottom:2px solid transparent}
.ca-nav a.active{border-bottom-color:#fff}
.ca-header-tools{margin-left:auto;display:flex;align-items:center;gap:18px}
.ca-icon-btn{background:none;border:0;color:#fff;cursor:pointer;padding:6px}
.ca-avatar{width:36px;height:36px;border-radius:50%;border:0;background:#5B7FA6;color:#fff;font-weight:700;cursor:pointer}
.ca-avatar-wrap{position:relative}
.ca-menu{position:absolute;right:0;top:44px;background:#fff;border:1px solid var(--line);border-radius:10px;box-shadow:0 8px 24px rgba(16,24,40,.12);min-width:180px;z-index:20;padding:6px}
.ca-menu a{display:block;padding:9px 12px;border-radius:6px;color:var(--ink);text-decoration:none;font-weight:500}
.ca-menu a:hover{background:var(--bg)}
.ca-page{max-width:1440px;margin:0 auto;padding:24px 32px 48px}
.ca-title-row{display:flex;align-items:center;gap:16px;margin:8px 0 20px}
.ca-title{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}
.ca-title h1{font:700 44px/1.1 var(--serif);margin:0;letter-spacing:-.5px}
.ca-title p{margin:0;color:var(--muted);font-size:16px}
.ca-title-right{margin-left:auto;display:flex;gap:10px;align-items:center}
.ca-card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:18px 20px}
.ca-card h2,.ca-section-title{font:700 26px/1.2 var(--serif);margin:0}
.ca-card-head{display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap}
.ca-card-head p{margin:0;color:var(--muted);font-size:14px}
.ca-card-head .ca-link{margin-left:auto;color:var(--blue);font-weight:600;text-decoration:none;font-size:14px}
.ca-grid{display:grid;gap:16px}
.ca-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:16px}
.ca-stat-label{font-weight:600;font-size:15px}
.ca-stat-value{font:700 34px/1.15 var(--serif);margin-top:6px}
.ca-stat-sub{color:var(--muted);font-size:13px;margin-top:4px}
.pos{color:var(--green)}.neg{color:var(--red)}.muted{color:var(--muted)}
.ca-pills{display:flex;gap:8px;flex-wrap:wrap}
.ca-pill{border:1px solid var(--line);background:#F1EEE6;color:var(--ink);border-radius:8px;padding:8px 16px;font-weight:600;font-size:14px;cursor:pointer}
.ca-pill.on{background:var(--navy);border-color:var(--navy);color:#fff}
.ca-table-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
.ca-table{width:100%;border-collapse:collapse;font-size:14px}
.ca-table th{text-align:left;font-weight:600;color:var(--muted);padding:10px 12px;border-bottom:1px solid var(--line);background:#FBFAF6;white-space:nowrap}
.ca-table td{padding:11px 12px;border-bottom:1px solid #F0ECE3;white-space:nowrap}
.ca-table tr:last-child td{border-bottom:0}
.ca-team{display:inline-flex;align-items:center;gap:8px}
.ca-team img{width:22px;height:22px;object-fit:contain}
.ca-conf{display:inline-block;min-width:78px;text-align:center;border-radius:6px;padding:4px 10px;font-size:12px;font-weight:700;letter-spacing:.04em}
.ca-conf.HIGH{background:var(--green-tint);color:var(--green)}
.ca-conf.STRONG{background:var(--blue-tint);color:var(--blue)}
.ca-conf.MEDIUM{background:var(--amber-tint);color:var(--amber)}
.ca-alpha-cell{display:inline-block;min-width:56px;text-align:center;border-radius:6px;padding:5px 8px;background:var(--amber-tint);font-weight:600}
.ca-alpha-cell.hi{background:var(--green-tint)}
.ca-go{display:inline-grid;place-items:center;width:32px;height:28px;border-radius:7px;background:var(--blue-tint);color:var(--blue);text-decoration:none;font-weight:700}
.ca-empty{color:var(--muted);padding:18px 4px}
.ca-star{background:none;border:0;cursor:pointer;color:var(--amber);font-size:17px}
.ca-select,.ca-input{border:1px solid var(--line);background:#fff;border-radius:8px;padding:9px 12px;font:500 14px var(--sans);color:var(--ink)}
.ca-btn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;font-weight:600;cursor:pointer}
@media (max-width:620px){
  .ca-header-inner{padding:0 16px;gap:14px;height:auto;flex-wrap:wrap}
  .ca-nav{order:3;width:100%;overflow-x:auto;gap:18px}
  .ca-page{padding:16px}
  .ca-title h1{font-size:32px}
  .ca-stats{grid-template-columns:1fr}
}
```
Fonts in every page `<head>` (replace the old font link and `styles.css`):
```html
<link href="https://fonts.googleapis.com/css2?family=Libre+Caslon+Text:wght@700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><link rel="stylesheet" href="css/theme.css?v=20261002a">
```
Add `<script src="js/shell.js?v=20261002a"></script>` before `js/boot.js` on every page.

- [ ] **Step 5: Wire the header** — in `site/app.js` `render()`: replace `shell.innerHTML = nav() + body + footer();` with `shell.innerHTML = siteHeader(page === "rankings" || page === "settings" ? "" : page) + \`<div class="ca-page">${body}</div>\` + footer(); wireShell();`. Make `mountGradientBackground()` return immediately (dark gradient retired). Leave `nav()` defined (unused) for now.

- [ ] **Step 6: Run tests — expect PASS**; serve `site/` and confirm every page shows the navy header with Dashboard · MLB · NFL · NBA · CFB · +EV · Track Record, the search icon, the "R" avatar opening a menu with Power Rankings / Watchlist / Settings. (Page bodies still use legacy dark styles — fixed in Task 3.)

- [ ] **Step 7: Commit** `git commit -m "feat(site): light design system, new header, avatar menu, watchlist store"`

---

### Task 3: Light-theme the legacy CSS (sections reused by the new pages)

**Files:**
- Modify: `site/app.js` (`injectStylesOnce` CSS block), remove `site/styles.css` usage (already dropped from heads in Task 2; delete the file)
- Test: `site/tests/legacy_css.test.mjs`

**Interfaces:**
- Consumes: tokens from `css/theme.css`.
- Produces: legacy sections (`matchupSection`, `historySection`, `trendsSection`, `splitsSection`, `propsProjectionSection`, `parlaySection`, `bestParlaysSection`, `propsSection`, `evSection`, `pnlSection`, `gameSimVisual`, `rankingsTable`, settings form) render legibly on the light background.

- [ ] **Step 1: Failing test** — no dark-theme colors remain in the injected CSS:
```js
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
test("legacy injected CSS uses theme tokens, not dark-theme colors", () => {
  const src = fs.readFileSync(new URL("../app.js", import.meta.url), "utf8");
  const css = src.slice(src.indexOf("function injectStylesOnce"), src.indexOf("async function render"));
  for (const dark of ["#080d13", "#101720", "#0b1118", "#111923", "#0f1620", "#141d27", "#131a22", "#18202a", "#1a2430", "#202a35", "#263341"]) {
    assert.ok(!css.toLowerCase().includes(dark), `dark color ${dark} still in legacy CSS`);
  }
  assert.ok(!fs.existsSync(new URL("../styles.css", import.meta.url)), "styles.css should be deleted");
});
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Convert.** In the `injectStylesOnce` CSS string, map colors: dark backgrounds (`#080d13 #0b1118 #101720 #111923 #0f1620 #131a22 #141d27 #18202a`) → `var(--card)` for panels / `var(--bg)` for page-level; borders (`#1a2430 #202a35 #263341 #26303a #283542 #34404d`) → `var(--line)`; light text (`#e9eef3 #eef3f8 #edf2f6 #dce5ec #c2ccd6 #bcc5cd`) → `var(--ink)`; muted text (`#7f8c9a #8290a0 #96a1ae #929eaa #8a97a6`) → `var(--muted)`; accent blues (`#4d99ff #59a2ff #69a8ff #6eaaff`) → `var(--blue)`; greens (`#3ddc84 #48d69a #58d2aa`) → `var(--green)`; reds (`#ff6b6b #ee8490`) → `var(--red)`; golds (`#f5b942 #f0b354`) → `var(--amber)`; tinted chip backgrounds (`#2a2410 #153153 #123e34 #48242a`) → `var(--amber-tint) var(--blue-tint) var(--green-tint) var(--red-tint)` respectively. Keep the A–F grade-chip colors (`#22a45d #e8c22e #ec8a2c #d9423f`) as they are (they read on light). Delete `site/styles.css`.
- [ ] **Step 4: Run test — expect PASS.** Serve and open `game.html?sport=cfb&game=<any current CFB game_pk>`, `track-record.html`, `ev.html`, `settings.html`, `rankings.html?sport=nfl`: every section is readable (dark text on light), no dark panels remain.
- [ ] **Step 5: Commit** `git commit -m "style(site): legacy sections on the light theme"`

---

### Task 4: Pure metrics + Alpha Score

**Files:**
- Create: `site/js/metrics.js`, `site/tests/metrics.test.mjs`
- Modify: every `site/*.html` (add `js/metrics.js` after `app.js`)

**Interfaces (globals, all pure):**
- `alphaScore({evPct, edgePp, segRoiPct, segN}) -> number` (40..100, Global Constraints formula); `alphaTier(score) -> "HIGH"|"STRONG"|"MEDIUM"|null`.
- `unitsFromPnl(pnlDollars) -> number` (= pnl / 10).
- `aggPnl(rows) -> {n, w, l, p, units, roiPct}` over P&L rows `{n, wins, losses, pushes, pnl}` (numeric strings allowed); `roiPct = units / n * 100` (0 when n = 0).
- `hitRate(results) -> number|null` — results `{won: true|false|null}`; pushes (`null`) excluded.
- `hitRateVsMarket(picks) -> number|null` — picks `{won, implied}`; hit rate minus mean implied prob over decided picks, in percentage points.
- `currentStreak(results) -> {kind:"W"|"L", n}|null` — results sorted newest first, `{won}`; pushes skipped.
- `recentForm(results, n) -> {w, l, p, pct}` — newest-first, first `n` rows.
- `calibration(rows, bands) -> [{band, n, predicted, actual}]` — rows `{prob, won}`; `bands` = `[[0.5,0.55],[0.55,0.6],[0.6,0.65],[0.65,0.7],[0.7,1.01]]`; prob < 0.5 uses `1 - prob` and flips `won` (favorite view).
- `brier(rows) -> number|null` — mean of `(prob - won)^2`.
- `edgeBuckets(rows, edges) -> [{label, n, hits, roiPct, units}]` — rows `{edgePct, won, profitUnits}`; `edges` = `[10, 5, 2, 0]` → labels `"> 10%" "5% to 10%" "2% to 5%" "0% to 2%" "< 0%"`.
- `histogram(values, edges) -> [{lo, hi, n}]`.
- `dailyCounts(rows, dateOf, days, today) -> [{date, n}]` — `days` consecutive ET dates ending at `today` (YYYY-MM-DD), zero-filled.
- `americanToProb(odds) -> number`, `probToAmerican(p) -> number`.
- `lineMoveScore(move) -> number` — ranks moves: spread/total `|cur_line - open_line|`; moneyline `|implied(cur) - implied(open)| * 100 / 2.5` (≈ points-equivalent).

- [ ] **Step 1: Write the failing tests** (`site/tests/metrics.test.mjs`):
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/boot.js"]);

test("alpha score formula and tiers", () => {
  assert.equal(g.alphaScore({ evPct: 8, edgePp: 8, segRoiPct: 0, segN: 0 }), 95);
  assert.equal(g.alphaScore({ evPct: 4, edgePp: 4, segRoiPct: 0, segN: 0 }), 70);
  assert.equal(g.alphaScore({ evPct: 0, edgePp: 0, segRoiPct: -50, segN: 1e9 }), 40);
  assert.equal(g.alphaScore({ evPct: 20, edgePp: 20, segRoiPct: 40, segN: 1e9 }), 100);
  // shrinkage: 10% ROI on 100 graded -> roi_s 5 -> s_track .75 -> +2.5 over neutral
  assert.equal(g.alphaScore({ evPct: 4, edgePp: 4, segRoiPct: 10, segN: 100 }), 73);
  assert.equal(g.alphaTier(85), "HIGH"); assert.equal(g.alphaTier(84), "STRONG");
  assert.equal(g.alphaTier(75), "STRONG"); assert.equal(g.alphaTier(74), "MEDIUM");
  assert.equal(g.alphaTier(65), "MEDIUM"); assert.equal(g.alphaTier(64), null);
});

test("pnl aggregation in units", () => {
  const a = g.aggPnl([{ n: "3", wins: 2, losses: 1, pushes: 0, pnl: "8.18" }, { n: 2, wins: 0, losses: 2, pushes: 0, pnl: -20 }]);
  assert.deepEqual({ n: a.n, w: a.w, l: a.l, p: a.p }, { n: 5, w: 2, l: 3, p: 0 });
  assert.ok(Math.abs(a.units - (-1.182)) < 1e-9);
  assert.ok(Math.abs(a.roiPct - (-23.64)) < 1e-9);
  assert.equal(g.aggPnl([]).roiPct, 0);
});

test("hit rate, vs market, streak, recent form", () => {
  const rs = [{ won: true, implied: 0.5 }, { won: false, implied: 0.6 }, { won: null, implied: 0.5 }, { won: true, implied: 0.4 }];
  assert.ok(Math.abs(g.hitRate(rs) - 2 / 3) < 1e-12);
  assert.ok(Math.abs(g.hitRateVsMarket(rs) - (2 / 3 - 0.5) * 100) < 1e-9);
  assert.deepEqual(g.currentStreak([{ won: true }, { won: null }, { won: true }, { won: false }]), { kind: "W", n: 2 });
  assert.equal(g.currentStreak([]), null);
  assert.deepEqual(g.recentForm([{ won: true }, { won: false }, { won: null }, { won: true }], 3), { w: 1, l: 1, p: 1, pct: 0.5 });
});

test("calibration flips underdog probabilities; brier", () => {
  const rows = [{ prob: 0.62, won: true }, { prob: 0.38, won: true }, { prob: 0.72, won: false }];
  const c = g.calibration(rows, [[0.5, 0.55], [0.55, 0.6], [0.6, 0.65], [0.65, 0.7], [0.7, 1.01]]);
  const b60 = c.find((x) => x.band[0] === 0.6);
  assert.equal(b60.n, 2);                       // .62 won, .38->.62 lost
  assert.ok(Math.abs(b60.predicted - 0.62) < 1e-12);
  assert.equal(b60.actual, 0.5);
  assert.ok(Math.abs(g.brier(rows) - ((0.38 ** 2 + 0.62 ** 2 + 0.72 ** 2) / 3)) < 1e-12);
});

test("edge buckets, histogram, daily counts, odds", () => {
  const eb = g.edgeBuckets([{ edgePct: 12, won: true, profitUnits: 0.9 }, { edgePct: 3, won: false, profitUnits: -1 }, { edgePct: -1, won: true, profitUnits: 1 }], [10, 5, 2, 0]);
  assert.deepEqual(eb.map((b) => [b.label, b.n]), [["> 10%", 1], ["5% to 10%", 0], ["2% to 5%", 1], ["0% to 2%", 0], ["< 0%", 1]]);
  assert.equal(eb[0].roiPct, 90);
  assert.deepEqual(g.histogram([-3, 1, 2, 7], [-5, 0, 5, 10]).map((b) => b.n), [1, 2, 1]);
  const dc = g.dailyCounts([{ d: "2026-09-30" }, { d: "2026-10-01" }, { d: "2026-10-01" }], (r) => r.d, 3, "2026-10-01");
  assert.deepEqual(dc, [{ date: "2026-09-29", n: 0 }, { date: "2026-09-30", n: 1 }, { date: "2026-10-01", n: 2 }]);
  assert.ok(Math.abs(g.americanToProb(-110) - 110 / 210) < 1e-12);
  assert.equal(g.probToAmerican(0.6), -150);
  assert.equal(g.probToAmerican(0.4), 150);
});
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `site/js/metrics.js`:**
```js
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
    n: (values || []).filter((v) => v >= lo && (i === edges.length - 2 ? v <= edges[i + 1] : v < edges[i + 1])).length }));
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
const americanToProb = (o) => (+o > 0 ? 100 / (+o + 100) : -o / (-o + 100));
const probToAmerican = (p) => (p >= 0.5 ? -Math.round(p / (1 - p) * 100) : Math.round((1 - p) / p * 100));
function lineMoveScore(m) {
  if (m.market === "moneyline") return Math.abs(americanToProb(m.cur_price) - americanToProb(m.open_price)) * 100 / 2.5;
  return Math.abs(+m.cur_line - +m.open_line);
}
```
- [ ] **Step 4: Run — expect PASS** (`cd site && npm test`).
- [ ] **Step 5: Commit** `git commit -m "feat(site): pure metrics + Alpha Score"`

---

### Task 5: Data layer — opportunities model, line-moves view

**Files:**
- Create: `site/js/data.js`, `site/tests/data.test.mjs`, `db/migration_site_redesign_a.sql`
- Modify: every `site/*.html` (add `js/data.js` after `js/shell.js`)

**Interfaces:**
- Consumes: `sb()`, `evCurrent(sport)`, `evPropsCurrent(sport)`, `trackRecordStarts()`, `inTrackRecord()` (app.js); `alphaScore`, `alphaTier`, `americanToProb` (metrics.js).
- Produces (globals):
  - `SPORTS = ["nfl", "cfb", "mlb", "nba"]`, `LIVE_SPORTS = ["nfl", "cfb"]`, `SPORT_STATUS = {mlb: "MLB model paused (last projections Aug 31, 2026)", nba: "NBA model not live yet"}`.
  - `toOpportunity(row, kind, seg) -> Opportunity|null` (pure) where `kind` is `"line"` (row from `ev_current`) or `"prop"` (row from `ev_prop_picks_current`), `seg` = `{roiPct, n}` for that sport×segment. `Opportunity` = `{kind, sport, game_pk, matchup, market, marketLabel, side, playerName, line, commence, book, odds, modelProb, impliedProb, edgePp, evPct, alpha, tier}`.
    - line: `modelProb = true_prob`, `impliedProb = best_line_implied ?? americanToProb(best_price)`, `odds = best_price`, `book = best_book`, `evPct = ev_best*100`, `edgePp = (modelProb - impliedProb)*100`.
    - prop: `modelProb = model_prob`, `impliedProb = americanToProb(best_price)`, `evPct = ev_best*100`, `edgePp = (model_prob - impliedProb)*100`, `line = line`, `playerName = player_name`.
    - returns `null` when `modelProb`, `odds` or `ev_best` is missing; `tier` may be null (callers filter `tier != null` for "opportunities").
  - `segmentKey(sport, kind, market) -> string` = `${sport}|${kind === "prop" ? "prop" : market}`.
  - `segmentRecords(evPnlRows) -> Map<segmentKey, {roiPct, n}>` from `ev_pnl_daily` rows (`aggPnl` per sport×market; market `"prop"` is the props segment).
  - `async loadOpportunities() -> Opportunity[]` — all live sports, lines + props, scored, sorted by `alpha` desc then `evPct` desc; failures of either source → that part empty.
  - `async loadLineMoves() -> LineMove[]` — rows of `line_moves_current` (`[]` if the view does not exist yet), each with `score = lineMoveScore(row)`, sorted by score desc.
  - `async loadSplits() -> Map<"game_pk|market|side", {cash_pct, ticket_pct}>` from `nfl_betting_splits_current` (+ `cfb_betting_splits_current`), latest per key; `[]`/empty on error.
  - `async loadEvHistory(days) -> {date, kind, sport}[]` — first-flagged ET date of each `is_pick` row from `ev_picks` (+ `ev_prop_picks`), last `days` days.

- [ ] **Step 1: Migration** `db/migration_site_redesign_a.sql`:
```sql
-- Site redesign Phase A: opening -> current Pinnacle line per upcoming game market/side,
-- for the dashboard's Market Movers and the +EV Market Pulse. Run before deploying the site.
CREATE INDEX IF NOT EXISTS idx_odds_snapshot_book_commence ON odds_snapshot (book, commence_time);

CREATE OR REPLACE VIEW line_moves_current WITH (security_invoker = true) AS
WITH s AS (
  SELECT game_pk, market, side, line, price, captured_at, commence_time
  FROM odds_snapshot
  WHERE book = 'pinnacle' AND coalesce(player_name, '') = ''
    AND market IN ('moneyline', 'spread', 'total') AND commence_time > now()
), o AS (
  SELECT DISTINCT ON (game_pk, market, side) game_pk, market, side, line AS open_line,
         price AS open_price, captured_at AS open_at
  FROM s ORDER BY game_pk, market, side, captured_at ASC
), c AS (
  SELECT DISTINCT ON (game_pk, market, side) game_pk, market, side, line AS cur_line,
         price AS cur_price, captured_at AS cur_at, commence_time
  FROM s ORDER BY game_pk, market, side, captured_at DESC
)
SELECT c.game_pk, c.market, c.side, o.open_line, o.open_price, o.open_at,
       c.cur_line, c.cur_price, c.cur_at, c.commence_time
FROM c JOIN o USING (game_pk, market, side)
WHERE o.open_at < c.cur_at;

GRANT SELECT ON line_moves_current TO anon, authenticated;
```
Controller validates the view body before the user runs it: execute the `WITH ... SELECT` (without `CREATE VIEW`) read-only and confirm it returns rows for upcoming games in < 2 s. (If `player_name` for game lines is not `''`/NULL, adjust the filter.)

- [ ] **Step 2: Failing tests** (`site/tests/data.test.mjs`):
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/shell.js", "js/data.js", "js/boot.js"]);

test("line opportunity: edge vs best-price implied, EV %, alpha + tier", () => {
  const o = g.toOpportunity({ sport: "nfl", game_pk: 1, matchup: "DET @ KC", market: "moneyline", side: "away",
    commence_time: "2026-10-05T20:25:00Z", true_prob: 0.604, best_line_implied: 0.439, best_price: 128,
    best_book: "draftkings", ev_best: 0.165 }, "line", { roiPct: 0, n: 0 });
  assert.equal(o.kind, "line"); assert.equal(o.odds, 128); assert.equal(o.book, "draftkings");
  assert.ok(Math.abs(o.edgePp - 16.5) < 1e-9); assert.ok(Math.abs(o.evPct - 16.5) < 1e-9);
  assert.equal(o.alpha, 95); assert.equal(o.tier, "HIGH");
});

test("prop opportunity uses model_prob and the best price", () => {
  const o = g.toOpportunity({ sport: "nfl", game_pk: 2, matchup: "BUF @ ATL", market: "rec_yds", side: "over", line: 64.5,
    player_name: "A. Receiver", commence_time: "2026-10-05T17:00:00Z", model_prob: 0.58, best_price: -110,
    best_book: "fanduel", ev_best: 0.107 }, "prop", { roiPct: 10, n: 100 });
  assert.equal(o.playerName, "A. Receiver"); assert.equal(o.line, 64.5);
  assert.ok(Math.abs(o.edgePp - (0.58 - 110 / 210) * 100) < 1e-9);
  assert.equal(o.alpha, g.alphaScore({ evPct: 10.7, edgePp: o.edgePp, segRoiPct: 10, segN: 100 }));
});

test("missing inputs -> null; segment records from ev_pnl_daily", () => {
  assert.equal(g.toOpportunity({ sport: "cfb", true_prob: null, best_price: 100, ev_best: 0.1 }, "line", { roiPct: 0, n: 0 }), null);
  const m = g.segmentRecords([{ sport: "cfb", market: "moneyline", n: 10, wins: 6, losses: 4, pushes: 0, pnl: 20 },
                              { sport: "nfl", market: "prop", n: 5, wins: 3, losses: 2, pushes: 0, pnl: -5 }]);
  assert.deepEqual(m.get("cfb|moneyline"), { roiPct: 20, n: 10 });
  assert.deepEqual(m.get("nfl|prop"), { roiPct: -10, n: 5 });
  assert.equal(g.segmentKey("nfl", "prop", "rec_yds"), "nfl|prop");
});
```
- [ ] **Step 3: Run — expect FAIL.**
- [ ] **Step 4: Implement `site/js/data.js`:**
```js
/* Data layer for the redesign: one Opportunity model (game lines + props) scored with
   the Alpha Score, plus line moves, splits and +EV history. Fetchers never throw:
   a missing view/table yields empty data and the panel hides. */
const SPORTS = ["nfl", "cfb", "mlb", "nba"];
const LIVE_SPORTS = ["nfl", "cfb"];
const SPORT_STATUS = { mlb: "MLB model paused (last projections Aug 31, 2026)", nba: "NBA model not live yet" };
const PROP_LABEL = { pass_yds: "Pass Yds", pass_tds: "Pass TDs", rush_yds: "Rush Yds", rec_yds: "Rec Yds", receptions: "Receptions", rush_att: "Rush Att", completions: "Completions", pass_att: "Pass Att", interceptions: "INTs" };
const segmentKey = (sport, kind, market) => `${sport}|${kind === "prop" ? "prop" : market}`;
const finite = (x) => x != null && x !== "" && Number.isFinite(+x);

function segmentRecords(evPnlRows) {
  const by = new Map();
  (evPnlRows || []).forEach((r) => { const k = `${r.sport}|${r.market}`; (by.get(k) || by.set(k, []).get(k)).push(r); });
  const out = new Map();
  by.forEach((rows, k) => { const a = aggPnl(rows); out.set(k, { roiPct: a.roiPct, n: a.n }); });
  return out;
}

function toOpportunity(r, kind, seg) {
  const isProp = kind === "prop";
  const modelProb = isProp ? r.model_prob : r.true_prob;
  if (!finite(modelProb) || !finite(r.best_price) || !finite(r.ev_best)) return null;
  const implied = !isProp && finite(r.best_line_implied) ? +r.best_line_implied : americanToProb(+r.best_price);
  const evPct = +r.ev_best * 100, edgePp = (+modelProb - implied) * 100;
  const alpha = alphaScore({ evPct, edgePp, segRoiPct: seg?.roiPct || 0, segN: seg?.n || 0 });
  return {
    kind, sport: r.sport, game_pk: r.game_pk, matchup: r.matchup || "", market: r.market,
    marketLabel: isProp ? (PROP_LABEL[r.market] || r.market) : ({ moneyline: "ML", spread: "Spread", total: "Total" }[r.market] || r.market),
    side: r.side, playerName: isProp ? r.player_name : null, line: finite(r.line) ? +r.line : null,
    commence: r.commence_time, book: r.best_book, odds: +r.best_price,
    modelProb: +modelProb, impliedProb: implied, edgePp, evPct, alpha, tier: alphaTier(alpha),
  };
}

async function loadOpportunities() {
  const [evPnl, ...per] = await Promise.all([
    sb("ev_pnl_daily?select=*").catch(() => []),
    ...LIVE_SPORTS.flatMap((s) => [evCurrent(s).catch(() => []), evPropsCurrent(s).catch(() => [])]),
  ]);
  const segs = segmentRecords(evPnl);
  const out = [];
  LIVE_SPORTS.forEach((s, i) => {
    for (const r of per[2 * i] || []) { const o = toOpportunity(r, "line", segs.get(segmentKey(s, "line", r.market))); if (o) out.push(o); }
    for (const r of per[2 * i + 1] || []) { const o = toOpportunity(r, "prop", segs.get(segmentKey(s, "prop", r.market))); if (o) out.push(o); }
  });
  return out.sort((a, b) => b.alpha - a.alpha || b.evPct - a.evPct);
}

async function loadLineMoves() {
  const rows = await sb("line_moves_current?select=*").catch(() => []);
  return (rows || []).map((r) => ({ ...r, score: lineMoveScore(r) })).filter((r) => r.score > 0).sort((a, b) => b.score - a.score);
}

async function loadSplits() {
  const [nfl, cfb] = await Promise.all([
    sb("nfl_betting_splits_current?select=*").catch(() => []),
    sb("cfb_betting_splits_current?select=*").catch(() => []),
  ]);
  const m = new Map();
  [...(nfl || []), ...(cfb || [])].forEach((r) => {
    const k = `${r.game_pk}|${r.market}|${r.side}`, prev = m.get(k);
    if (!prev || r.captured_at > prev.captured_at) m.set(k, r);
  });
  return m;
}

async function loadEvHistory(days) {
  const since = new Date(Date.now() - days * 864e5).toISOString();
  const [lines, props] = await Promise.all([
    sb(`ev_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,market,side,created_at`).catch(() => []),
    sb(`ev_prop_picks?is_pick=eq.true&created_at=gte.${since}&select=sport,game_pk,player_id,market,side,created_at`).catch(() => []),
  ]);
  const first = new Map();
  const add = (r, kind) => {
    const k = `${kind}|${r.sport}|${r.game_pk}|${r.player_id || ""}|${r.market}|${r.side}`;
    if (!first.has(k) || r.created_at < first.get(k).created_at) first.set(k, { ...r, kind });
  };
  (lines || []).forEach((r) => add(r, "line")); (props || []).forEach((r) => add(r, "prop"));
  return [...first.values()].map((r) => ({ date: etDateStr(r.created_at), kind: r.kind, sport: r.sport }));
}
```
(`ev_prop_picks` has a `created_at` column? Verify with `select column_name from information_schema.columns where table_name='ev_prop_picks'`; if it is named differently, use that column and keep the same output shape.)
- [ ] **Step 5: Run — expect PASS.**
- [ ] **Step 6: Commit** `git commit -m "feat(site): opportunities model (Alpha Score), line moves view, splits + +EV history fetchers"`

---

### Task 6: Charts + components (`js/ui.js`)

**Files:**
- Create: `site/js/ui.js`, `site/tests/ui.test.mjs`
- Modify: every `site/*.html` (add `js/ui.js` after `js/metrics.js`); `site/css/theme.css` (chart classes)

**Interfaces (pure string builders):**
- `statCard({label, labelNote?, value, valueClass?, sub?, subClass?, visual?}) -> string`
- `pills(name, items:[[key,label]], activeKey) -> string` (buttons `data-pill="${name}" data-key=...`)
- `confPill(tier) -> string` (`<span class="ca-conf HIGH">HIGH</span>`; `""` when null)
- `alphaCell(score) -> string` (`ca-alpha-cell`, `hi` when ≥ 85)
- `oddsStr(odds) -> string` (`+128` / `-110`)
- `bookBadge(bookKey) -> string` (rounded square with the book's initials and brand color; real logos come in Phase B)
- `teamCell(name, sport) -> string` (`logoImg` + name)
- `sparkline(values, {w=120,h=36,color}) -> string` (SVG polyline; `""` when < 2 values)
- `miniBars(values, {w=70,h=40,color}) -> string`
- `donut(fraction, {size=78, stroke=10, color}) -> string` (`stroke-dasharray` = `C*f C*(1-f)`)
- `areaChart(points:[{x:label,y:number}], {w=1000,h=260, yTicks=5, color}) -> string` (axis ticks + filled area + line)
- `groupedBars(groups:[{label, bars:[{value, color, label}]}], {h=180}) -> string`
- `histogramChart(bins:[{label, n, color}]) -> string`
- `lineChart(series:[{label, color, values}], xLabels) -> string`
- `donutLegend(rows:[{label, pct, value, color}]) -> string`

- [ ] **Step 1: Failing tests:**
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/boot.js"]);

test("conf pill + alpha cell", () => {
  assert.equal(g.confPill(null), "");
  assert.match(g.confPill("STRONG"), /class="ca-conf STRONG">STRONG</);
  assert.match(g.alphaCell(89), /ca-alpha-cell hi/);
  assert.doesNotMatch(g.alphaCell(70), / hi/);
  assert.equal(g.oddsStr(128), "+128"); assert.equal(g.oddsStr(-110), "-110");
});
test("sparkline points and empty input", () => {
  assert.equal(g.sparkline([1]), "");
  const svg = g.sparkline([0, 5, 10], { w: 100, h: 20 });
  assert.match(svg, /points="0\.0,20\.0 50\.0,10\.0 100\.0,0\.0"/);
});
test("donut arc length matches the fraction", () => {
  const svg = g.donut(0.25, { size: 100, stroke: 10 });
  const C = 2 * Math.PI * 45;
  assert.match(svg, new RegExp(`stroke-dasharray="${(C * 0.25).toFixed(2)} ${(C * 0.75).toFixed(2)}"`));
});
test("area chart renders one path per series and y ticks", () => {
  const svg = g.areaChart([{ x: "a", y: -10 }, { x: "b", y: 0 }, { x: "c", y: 30 }], { yTicks: 5 });
  assert.equal((svg.match(/<path /g) || []).length, 2);   // fill + line
  assert.match(svg, />30u?</);
});
```
- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Implement `site/js/ui.js`.** Every builder returns a string; SVGs use `viewBox` + `preserveAspectRatio="none"` (area/line charts) so they scale with the card; colors default to `var(--green)`. Core implementations (the rest follow the same pattern — scale values into the viewBox, emit `<rect>` / `<polyline>` / `<path>`):
```js
/* UI components + inline SVG charts. Pure: data in, HTML string out. */
const uiEsc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const oddsStr = (o) => (+o > 0 ? `+${+o}` : `${+o}`);
const confPill = (tier) => (tier ? `<span class="ca-conf ${tier}">${tier}</span>` : "");
const alphaCell = (s) => `<span class="ca-alpha-cell${s >= 85 ? " hi" : ""}">${s}</span>`;
const BOOK_STYLE = { draftkings: ["DK", "#0B3D2E"], fanduel: ["FD", "#1493FF"], betmgm: ["MGM", "#B59A5B"], williamhill_us: ["CZR", "#173F35"],
  fanatics: ["FAN", "#D21F3C"], espnbet: ["ESPN", "#D00"], hardrockbet: ["HR", "#5A2D82"], thescore: ["SCR", "#1E5BFF"],
  bet365: ["365", "#027B5B"], ballybet: ["BAL", "#C8102E"], pinnacle: ["PIN", "#0E2238"], betrivers: ["BR", "#1B3B6F"] };
function bookBadge(key) {
  const [abbr, color] = BOOK_STYLE[String(key || "").toLowerCase()] || [String(key || "?").slice(0, 3).toUpperCase(), "#5B6675"];
  return `<span class="ca-book" title="${uiEsc(key)}" style="background:${color}">${abbr}</span>`;
}
const teamCell = (name, sport) => `<span class="ca-team">${logoImg(name, sport)}${uiEsc(name)}</span>`;
function statCard({ label, labelNote = "", value, valueClass = "", sub = "", subClass = "", visual = "" }) {
  return `<div class="ca-card ca-stat"><div class="ca-stat-label">${label}${labelNote ? ` <span class="muted">${labelNote}</span>` : ""}</div>
    <div class="ca-stat-body"><div><div class="ca-stat-value ${valueClass}">${value}</div>${sub ? `<div class="ca-stat-sub ${subClass}">${sub}</div>` : ""}</div>${visual ? `<div class="ca-stat-visual">${visual}</div>` : ""}</div></div>`;
}
function pills(name, items, active) {
  return `<div class="ca-pills" role="tablist">${items.map(([k, l]) => `<button class="ca-pill${k === active ? " on" : ""}" data-pill="${name}" data-key="${k}" role="tab" aria-selected="${k === active}">${l}</button>`).join("")}</div>`;
}
function sparkline(values, { w = 120, h = 36, color = "var(--green)" } = {}) {
  if (!values || values.length < 2) return "";
  const min = Math.min(...values), max = Math.max(...values), r = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1) * w).toFixed(1)},${(h - (v - min) / r * h).toFixed(1)}`).join(" ");
  return `<svg class="ca-spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}"><polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}
function donut(f, { size = 78, stroke = 10, color = "var(--navy)" } = {}) {
  const r = (size - stroke) / 2, C = 2 * Math.PI * r, x = Math.max(0, Math.min(1, +f || 0));
  return `<svg class="ca-donut" viewBox="0 0 ${size} ${size}" width="${size}" height="${size}"><circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="#E9E5DB" stroke-width="${stroke}"/><circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="${stroke}" stroke-dasharray="${(C * x).toFixed(2)} ${(C * (1 - x)).toFixed(2)}" transform="rotate(-90 ${size / 2} ${size / 2})" stroke-linecap="round"/></svg>`;
}
function areaChart(points, { w = 1000, h = 260, yTicks = 5, color = "var(--green)", unit = "u" } = {}) {
  if (!points || points.length < 2) return "";
  const ys = points.map((p) => +p.y), lo = Math.min(0, ...ys), hi = Math.max(0, ...ys), r = hi - lo || 1;
  const pad = 36, X = (i) => pad + i / (points.length - 1) * (w - pad), Y = (v) => 8 + (hi - v) / r * (h - 32);
  const line = points.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(+p.y).toFixed(1)}`).join(" ");
  const ticks = Array.from({ length: yTicks }, (_, i) => lo + r * i / (yTicks - 1));
  const tickSvg = ticks.map((t) => `<line x1="${pad}" x2="${w}" y1="${Y(t).toFixed(1)}" y2="${Y(t).toFixed(1)}" class="ca-grid-line"/><text x="0" y="${(Y(t) + 4).toFixed(1)}" class="ca-axis">${Math.round(t)}${unit}</text>`).join("");
  const xEvery = Math.max(1, Math.ceil(points.length / 7));
  const xs = points.map((p, i) => (i % xEvery === 0 || i === points.length - 1 ? `<text x="${X(i).toFixed(1)}" y="${h - 4}" text-anchor="middle" class="ca-axis">${uiEsc(p.x)}</text>` : "")).join("");
  return `<svg class="ca-area" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${tickSvg}<path d="${line} L${X(points.length - 1).toFixed(1)} ${Y(lo).toFixed(1)} L${X(0).toFixed(1)} ${Y(lo).toFixed(1)}Z" fill="${color}" opacity=".12"/><path d="${line}" fill="none" stroke="${color}" stroke-width="2.5" vector-effect="non-scaling-stroke"/><circle cx="${X(points.length - 1).toFixed(1)}" cy="${Y(ys.at(-1)).toFixed(1)}" r="5" fill="${color}"/>${xs}</svg>`;
}
```
Implement `miniBars`, `groupedBars`, `histogramChart`, `lineChart`, `donutLegend` in the same style (bars as `<rect>` scaled to the max value, value labels above bars for `groupedBars`, legend rows as `<div>`s). Add CSS: `.ca-book{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:6px;color:#fff;font:700 9px var(--sans)}`, `.ca-stat-body{display:flex;align-items:flex-end;justify-content:space-between;gap:10px}`, `.ca-axis{font:12px var(--sans);fill:var(--muted)}`, `.ca-grid-line{stroke:#ECE8DF;stroke-width:1;vector-effect:non-scaling-stroke}`, `.ca-area{width:100%;height:260px}`.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(site): UI components + SVG charts"`

---

### Task 7: Dashboard (mockup #1)

**Files:**
- Create: `site/js/pages/dashboard.js`, `site/tests/dashboard.test.mjs`
- Modify: `site/app.js` (`render()`: `page === "dashboard"` → `await buildDashboard()`, then `wireDashboard()`), every `site/*.html` (add `js/pages/dashboard.js`)

**Interfaces:**
- Consumes: Tasks 2–6 globals; `predictions(sport)`, `trackRecordStarts`, `inTrackRecord`, `etDateStr`, `sb` (app.js).
- Produces: `async buildDashboard() -> string`, `wireDashboard()`, and pure helpers `dashSlate(predRows, opps) -> SlateRow[]` (one row per game: its highest-alpha opportunity; games without one show their model pick with no Alpha) and `dashExposure(evPnlRows, predPnlRows, sinceDate) -> {bySport:[{sport, units, pct}], byMarket:[{label, pct}]}` (units *staked* = count of graded bets; P&L units beside each sport).

**Layout (top → bottom, as the mockup):**
1. `pageTitle("Today at a Glance", "Key opportunities, performance, and model insights across all sports.", dateNav)` — `dateNav` = ‹ [📅 Tue, Oct 13, 2025-style ET date] › ; changes `?date=YYYY-MM-DD` and re-renders. All "today" panels use that date (ET).
2. Six stat cards (`.ca-stats` 6 columns ≥ 1280px): **Games Tracked** (total + per-sport logo/label/count row for NFL, CFB, MLB, NBA — from `predictions(s)` filtered to the date); **Live +EV Opportunities** (count of opportunities with a tier for games on the date; "↑ N vs. yesterday" from `loadEvHistory(8)`; `miniBars` of the last 7 days); **Best Current Edge** (max `evPct` opportunity: value + "matchup / market" + both team logos); **Model Hit Rate (30D)** (game-line +EV picks graded in the last 30 days in the record: `hitRate`; sub "+X% vs. market" = `hitRateVsMarket` with `implied` from the pick's best price; `donut`); **Units (30D)** (`aggPnl` of `ev_pnl_daily` + `prediction_pnl_daily` rows in the last 30 days: units + "+ROI% ROI", `sparkline` of cumulative daily units); **Active Signals** (counts by tier: dots green/blue/amber, labels "High Conviction", "Strong Value", "Medium Value").
3. Two-column row (2fr / 1fr): **Best Opportunities Right Now** card — subtitle "Top model edges across all sports, sorted by expected value."; "View All →" to `ev.html`; pills All Sports / NFL / CFB / MLB / NBA + separate pill group Game Lines / Player Props; table columns `# · Sport · Matchup / Player · Market · Best Odds (bookBadge + odds) · Model Prob. · Market Prob. · Edge · Alpha Score · Confidence · →` (top 10, sorted by `evPct` desc per the mockup subtitle). **Today's Slate** card — pills All / NFL / CFB / MLB / NBA; columns `Time (ET) · Matchup / Player (status dot + both logos) · Market · Line · Alpha Score · →`; status dot red = live/started, navy = upcoming within 3h, amber = later; rows from `dashSlate`.
4. Three-column row: **Performance Snapshot** (pills 7D / 30D / Season; four mini stats ROI, Units, Hit Rate, Avg Edge; `areaChart` of cumulative units over the window); **Market Movers** (pills Line Moves / Model vs Market / New Signals / Stale Lines; rows icon + "Team −2.5 → −4.0" + sub-line + right-aligned change and "Nh ago"; Line Moves from `loadLineMoves()` with sub-line "N% of tickets on X" only when splits exist; Model vs Market = largest |model − market| spread/total from `predictions`; New Signals = opportunities first flagged in the last 24h; Stale Lines = picks whose `soft_vs_sharp_gap` is largest); **Portfolio & Exposure** (pills By Sport / By Market Type; `donut` segments + `donutLegend` with units; "Market Type Exposure" bars Game Lines / Player Props / Totals) and **Watchlist** (pills Games / Teams / Players; starred items with matchup, pick, time; empty state "Star games, teams or players to follow them here.") stacked beside **Recent Model Updates** — Phase B data: omit the card entirely in Phase A.
5. Every table row and slate row links to `game.html?sport=<s>&game=<game_pk>`; star buttons on slate rows feed the Watchlist.

- [ ] **Step 1: Failing tests** for the pure helpers:
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/boot.js"]);

test("slate keeps one row per game: the highest-alpha opportunity", () => {
  const preds = [{ sport: "nfl", game_pk: 1, home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions", commence_time: "2026-10-05T20:25:00Z" },
                 { sport: "cfb", game_pk: 2, home_team_name: "Oregon", away_team_name: "Ohio State", commence_time: "2026-10-04T23:30:00Z" }];
  const opps = [{ game_pk: 1, sport: "nfl", alpha: 70, marketLabel: "Spread", tier: "MEDIUM" },
                { game_pk: 1, sport: "nfl", alpha: 89, marketLabel: "ML", tier: "HIGH" }];
  const s = g.dashSlate(preds, opps);
  assert.equal(s.length, 2);
  assert.equal(s.find((r) => r.game_pk === 1).opp.alpha, 89);
  assert.equal(s.find((r) => r.game_pk === 2).opp, null);
  assert.deepEqual(s.map((r) => r.game_pk), [2, 1]);   // by kickoff
});

test("exposure shares sum to 100 and use graded bet counts", () => {
  const ev = [{ game_date: "2026-09-20", sport: "cfb", market: "moneyline", n: 6, wins: 3, losses: 3, pushes: 0, pnl: 10 },
              { game_date: "2026-09-21", sport: "nfl", market: "prop", n: 4, wins: 2, losses: 2, pushes: 0, pnl: -5 }];
  const e = g.dashExposure(ev, [], "2026-09-01");
  assert.deepEqual(e.bySport.map((x) => [x.sport, x.pct]), [["cfb", 60], ["nfl", 40]]);
  assert.equal(Math.round(e.bySport.reduce((s, x) => s + x.pct, 0)), 100);
});
```
- [ ] **Step 2: Run — expect FAIL.** **Step 3: Implement** `dashboard.js` per the layout, using only the Interfaces above; every card checks for empty data and renders `<p class="ca-empty">…</p>` (Best Opportunities: "No +EV opportunities on the board right now."). **Step 4: Run tests — PASS.** **Step 5: Visual check** — serve `site/`, open `index.html` at 1440px and 390px, compare with mockup #1 section by section (layout, order, labels, pill styles, table columns); fix differences. **Step 6: Commit** `git commit -m "feat(site): dashboard (Today at a Glance)"`.

---

### Task 8: +EV page (mockup #3)

**Files:**
- Create: `site/js/pages/ev.js`, `site/tests/ev.test.mjs`
- Modify: `site/app.js` (`render()`: `page === "ev"` → `await buildEvPage()` then `wireEvPage2()`; the legacy `buildEv`/`wireEvPage` stay for their parlay/props sections), every HTML (add the script)

**Interfaces:**
- Produces: `async buildEvPage() -> string`, `wireEvPage2()`, pure `evFilter(opps, f) -> Opportunity[]` with `f = {sport, kind:"all"|"line"|"prop", market, book, minEdge, tier, date, q, sort}` (`sort`: `"edge"|"ev"|"alpha"|"time"`, all descending except `"time"` ascending; no `sort` keeps the input order; omitted filter keys do not filter), `evBucketRows(gradedLines, gradedProps) -> edgeBuckets(...)` input rows.

**Layout (mockup #3):**
1. `pageTitle("+EV", "Find the best expected value opportunities across all sports, powered by the CappingAlpha model.", lastUpdated + refresh button)` — "Last Updated" = newest `created_at` of the current picks, ET.
2. Five stat cards: Total +EV Opportunities (+ "▲ N vs. yesterday", mini bars); Average Edge (mean `edgePp` of listed opps, sparkline of the 7-day daily mean); Highest Edge (value + "matchup / market", star icon); Model Hit Rate (Last 30 Days) + donut; ROI (Last 30 Days) + "+Nu" + mini bars.
3. Filter card: Sport, League (NFL/CFB/… — same as sport until other leagues exist), Market Type (All / Moneyline / Spread / Total / each prop market present), Sportsbook (books present in the data), Minimum Edge (≥ 0 / 2 / 5 / 10%), Confidence (All / HIGH / STRONG / MEDIUM), Date (Today / Tomorrow / This week), and the search box (team, player, or game). Filters persist in `?` query params and survive the 5-minute refresh.
4. Pill tabs "All Opportunities (N)", "Game Lines (N)", "Player Props (N)" + "Sort By" select (Highest Edge / Highest EV / Alpha Score / Game Time).
5. **+EV Opportunities** table: `# · Sport · Matchup / Player · Market · Best Odds · Model Prob. · Impl. Prob. · Edge · EV · Alpha Score · Confidence · Game Time · View →`.
6. Right rail: **Top Alpha Opportunities** (top 5 by alpha: rank badge, both logos, market + odds, "model% vs implied%", edge box); **Market Pulse** (Biggest Line Move from `loadLineMoves()[0]` + ticket % if split exists; Highest Confidence Edge; Most Mispriced Total = max |model total − market total| from `predictions`; Average Market Divergence = mean CLV of graded +EV picks this week "vs. closing lines"); **EV Distribution** (`histogramChart` of listed opps' `evPct`, bins `<-10, -5–0, 0–5, 5–10, 10–15, >15`, red below 0 / green above); **Performance by Edge Bucket** (table Edge Range / Hits / ROI / Units from `edgeBuckets` over graded +EV picks with their flagged edge: game lines join `ev_results` ↔ `ev_picks` (edge = (true_prob − implied(best_price))·100, profit = won ? decimal−1 : −1), props from `ev_prop_results` (`profit` / 10 units, edge = (model_prob − implied)·100 at the flagged price — if the price is not stored, use `novig_close` only for CLV and skip the row from buckets)).
7. Below the main table, the legacy sections stay (restyled by Task 3): `bestParlaysSection`, `parlaySection` per sport.

- [ ] **Step 1: Failing tests** for `evFilter`:
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/ev.js", "js/boot.js"]);
const O = (o) => ({ kind: "line", sport: "nfl", market: "moneyline", book: "draftkings", edgePp: 5, evPct: 5, alpha: 75, tier: "STRONG", commence: "2026-10-05T17:00:00Z", matchup: "BUF @ ATL", playerName: null, ...o });
test("filters by sport, kind, min edge, tier, book and search; sorts", () => {
  const opps = [O({ game_pk: 1 }), O({ game_pk: 2, kind: "prop", playerName: "Josh Allen", edgePp: 12, evPct: 9, alpha: 90, tier: "HIGH" }),
                O({ game_pk: 3, sport: "cfb", edgePp: 1, evPct: 2, alpha: 66, tier: "MEDIUM", book: "fanduel" })];
  assert.deepEqual(g.evFilter(opps, { sport: "nfl" }).map((o) => o.game_pk), [1, 2]);
  assert.deepEqual(g.evFilter(opps, { kind: "prop" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { minEdge: 2 }).map((o) => o.game_pk).sort(), [1, 2]);
  assert.deepEqual(g.evFilter(opps, { tier: "HIGH" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { book: "fanduel" }).map((o) => o.game_pk), [3]);
  assert.deepEqual(g.evFilter(opps, { q: "allen" }).map((o) => o.game_pk), [2]);
  assert.deepEqual(g.evFilter(opps, { sort: "edge" }).map((o) => o.game_pk), [2, 1, 3]);
  assert.deepEqual(g.evFilter(opps, { sort: "alpha" }).map((o) => o.game_pk), [2, 1, 3]);
});
```
- [ ] **Steps 2–6:** run (FAIL) → implement → run (PASS) → visual check vs mockup #3 at 1440px and 390px → commit `feat(site): +EV page redesign`.

---

### Task 9: Game page — "Matchup Story" (mockup #2)

**Files:**
- Create: `site/js/pages/game.js`, `site/tests/game.test.mjs`
- Modify: `site/app.js` (`render()`: `page === "game"` → `await buildGamePage()` then `wireGamePage()`; keep `wireGameSim()` for the sim widget), every HTML

**Interfaces:**
- Consumes: legacy `buildGame()` data calls (read it first: it fetches the prediction row, odds/moneylines, sim, props, splits, trends, team context) and sections `matchupSection`, `historySection`, `trendsSection`, `splitsSection`, `propsProjectionSection`, `gameSimVisual`, `nflPredictionSection`, `boxscoreSection`.
- Produces: `async buildGamePage() -> string`, `wireGamePage()`, pure `gameRead(markets) -> {market, label, modelProb, impliedProb, edgePp}|null` (largest positive edge among ML / spread / total for this game; null when none positive) and `modelProjectionRows(pred, odds) -> [{label, market, model, diff}]`.

**Layout (mockup #2):**
1. Hero card (navy, stadium-dark overlay gradient): "‹ CFB Board" back link, top line "<NETWORK if known> · <Day, Mon D · h:mm PM ET>", both logos large, team names, records "W-L (conf W-L CONF)" from `team_history` / `power_rankings` (`su`; conference record only when available, else omit the parentheses), "@" between; venue / weather line is Phase B → omitted. Three odds boxes: MONEYLINE (away / home), SPREAD (away / home), TOTAL (O / U) — consensus current prices (existing odds fetch); the side the model favors in red-bold like the mockup.
2. **CappingAlpha Read** card: "Cα" mark, "CAPPINGALPHA READ", headline "<Team> <line> shows the strongest model divergence", "The market implies a X% cover probability. CappingAlpha estimates Y%." and the green edge box "+Z% MODEL EDGE" — from `gameRead`; if null: headline "No model edge on this game" and no edge box.
3. Tabs: Overview · Matchup · Market · Trends · Players (state in `?tab=`).
   - **Overview:** five cards — Alpha Score (best opportunity of this game, or "—"), Win Probability (both logos + %), Projected Score, Projected Spread ("ALA −13.2", "Market: −10.5"), Projected Total ("56.8", "Market: 52.5"); **Model Projection** table (Market / CappingAlpha / Difference for Moneyline, Spread, Total with implied % in parentheses); **Key Insights** (Phase A: up to four generated sentences from data present — explosive-play rank and unit grades from `matchup_grades.units`, power ranks, current streaks; skip the block if none) beside **Projected Game Flow** (Phase B → omitted) and **Cover Probability** bar (model cover % per side; no "% of Money" caption unless splits exist).
   - **Matchup:** `matchupSection` + power ranks; NFL: `nflPredictionSection` and the sim widget (`gameSimVisual`).
   - **Market:** line history (Phase B data not needed: use `loadLineMoves()` open→current for this game), best book per side, `splitsSection` when splits exist.
   - **Trends:** `historySection` + `trendsSection`.
   - **Players:** `propsProjectionSection` (NFL); other sports: "Player projections are NFL-only for now."
4. Works for NFL and CFB; MLB/NBA game pages show the hero (when a prediction exists) and the sport status message.

- [ ] **Step 1: Failing tests:**
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/game.js", "js/boot.js"]);
test("read picks the largest positive edge; none -> null", () => {
  const r = g.gameRead([{ market: "spread", label: "Alabama -10.5", modelProb: 0.618, impliedProb: 0.554 },
                        { market: "total", label: "Over 52.5", modelProb: 0.573, impliedProb: 0.501 },
                        { market: "moneyline", label: "Alabama ML", modelProb: 0.821, impliedProb: 0.809 }]);
  assert.equal(r.market, "total"); assert.ok(Math.abs(r.edgePp - 7.2) < 1e-9);
  assert.equal(g.gameRead([{ market: "spread", label: "x", modelProb: 0.4, impliedProb: 0.5 }]), null);
});
```
- [ ] **Steps 2–6:** run (FAIL) → implement → run (PASS) → visual check vs mockup #2 on a live CFB and a live NFL game at 1440px and 390px, plus a game with no edge → commit `feat(site): game page (Matchup Story)`.

---

### Task 10: Track Record (mockup #4)

**Files:**
- Create: `site/js/pages/track.js`, `site/tests/track.test.mjs`
- Modify: `site/app.js` (`render()`: `page === "track"` → `await buildTrackPage()` then `wireTrackPage()`), every HTML

**Interfaces:**
- Consumes: the same sources as legacy `buildTrack()` (read it): `prediction_accuracy` (graded game picks: winner / spread / total), `prediction_pnl` (per-pick result + pnl), `prediction_pnl_daily`, `accuracy_by_confidence`, `game_closing_prices`, `trackRecordStarts` + `inTrackRecord`; `ev_results`/`ev_picks` for the +EV view.
- Produces: `async buildTrackPage() -> string`, `wireTrackPage()`, pure `trackRows(accuracyRows, pnlRows) -> TrackRow[]` — one row per graded pick `{date, sport, game_pk, matchup, market:"moneyline"|"spread"|"total", pick, closing, prob, result:"W"|"L"|"P", finalScore, units}`:
  - moneyline row when `actual_winner` is set: `result = winner_correct ? "W" : "L"`, `prob = max(win_prob, 1 − win_prob)`, `pick = predicted_winner`;
  - spread row only when `market_spread` is set: `spread_pick_correct` true → W, false → L, null → P;
  - total row only when `market_total` is set: `total_pick_correct` true → W, false → L, null → P;
  - `prob` for spread / total = the model's cover / over probability when the prediction carries it, else `null` (the Model Prob. cell shows "—");
  - `finalScore` = `"<actual_winner> by <|actual_margin|> · <actual_total> total"` (`prediction_accuracy` stores margins and totals, not team scores);
  - `units` = the matching `prediction_pnl` row's `pnl / 10` (keyed `game_pk|market`), else `null`;
  - rows outside the published record (`inTrackRecord`) are dropped by the caller, not here.

**Layout (mockup #4):**
1. `pageTitle("Track Record", "Prediction performance across leagues and markets.", rangePills + dateRange)` — pills 7D / 30D / Season / All Time; date range text "<first> – <last>" of rows in range; an "Archive" toggle (pre-restart NFL) at the end of the pill group.
2. Five stat cards: OVERALL RECORD (W-L-P + win % + mini bars by week); ACCURACY VS CLOSING LINE (share of spread/total picks on the right side of the close — the existing `spread_pick_correct`/`total_pick_correct` vs closing line — plus "+X% vs market" = that share − 52.4% breakeven, sparkline); TOTAL PREDICTIONS ("Across all leagues and markets", mini bars); CURRENT STREAK (`currentStreak`, "Last 5: W-L-P", five W/L dots); AVERAGE CONFIDENCE (mean `prob`, donut).
3. **Cumulative Prediction Performance** (pills Overall / NFL / CFB / MLB / NBA) — `areaChart` of running net wins (W − L) by date.
4. **Performance by Market** — `groupedBars` Win % / Loss % / Push % for Moneyline, Spread, Total with "W-L-P (n)" under each.
5. **Performance by League** (rows: logo, bar, win %, W-L-P, (n)); **Confidence Calibration** (`calibration` bands 50–55 … 70%+ → paired bars predicted vs actual, n under each); **Recent Form** (L10 / L25 / L50 / L100 via `recentForm`).
6. Filter row: pills All Predictions / Moneyline / Spread / Totals; selects League, Market Type, Confidence (50–55% … 70%+), Result (All / W / L / P), Sort By (Most Recent / Highest Confidence); search box. Results table: `Date · League · Matchup / Player · Market · Model Prediction · Closing Line · Model Prob. · Result (W/L/Push chip) · Confidence · Final Score / Outcome · View →` (paginate 25 at a time with "Show more").
7. Right column: **Best Performing Segments** (pills Market / League / Confidence; rows label, win %, W-L-P); **Model Calibration** (Brier score + "Well Calibrated" chip when |predicted − actual| ≤ 3 pts, Predicted Avg. Confidence, Actual Hit Rate, Calibration Difference).
8. A secondary toggle "Model picks · +EV picks": the +EV view reuses the legacy P&L tracker and `evTrackSection` / prop sections (restyled), so nothing currently on the page is lost.

- [ ] **Step 1: Failing tests:**
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/track.js", "js/boot.js"]);
test("track rows: one per graded market, favorite-side probability, push handling", () => {
  const acc = [{ sport: "cfb", game_pk: 9, game_date: "2026-09-27", home_team_name: "Texas", away_team_name: "Oklahoma",
    win_prob: 0.581, predicted_winner: "Texas", actual_winner: "Texas", winner_correct: true, pred_margin: 6.2, actual_margin: 10,
    market_spread: -3.5, spread_pick_correct: true, pred_total: 55, actual_total: 54.5, market_total: 54.5, total_pick_correct: null }];
  const rows = g.trackRows(acc, [{ game_pk: 9, market: "spread", pnl: 9.09 }]);
  assert.deepEqual(rows.map((r) => [r.market, r.result]), [["moneyline", "W"], ["spread", "W"], ["total", "P"]]);
  assert.ok(Math.abs(rows[0].prob - 0.581) < 1e-12);
  assert.equal(rows[0].finalScore, "Texas by 10 · 54.5 total");
  assert.ok(Math.abs(rows[1].units - 0.909) < 1e-12);
  assert.equal(rows[0].units, null);
  const noLines = g.trackRows([{ ...acc[0], market_spread: null, market_total: null }], []);
  assert.deepEqual(noLines.map((r) => r.market), ["moneyline"]);
});
```
- [ ] **Steps 2–6:** run (FAIL) → implement → run (PASS) → visual check vs mockup #4 at 1440px and 390px → commit `feat(site): track record redesign`.

---

### Task 11: Board pages (NFL / CFB / MLB / NBA), Rankings tab, Settings, search

**Files:**
- Create: `site/js/pages/board.js`, `site/tests/board.test.mjs`
- Modify: `site/app.js` (`render()`: `nfl`/`cfb`/`mlb`/`nba` → `await buildBoardPage(page)` then `wireBoardPage()`; `rankings` and `settings` keep their builders inside the new shell), `site/js/shell.js` (search), every HTML

**Interfaces:**
- Produces: `async buildBoardPage(sport) -> string`, `wireBoardPage()`; `searchIndex(preds, opps) -> [{label, sub, href}]` (pure; teams, matchups, prop players) and `searchQuery(index, q) -> first 8 matches` (case-insensitive substring on label + sub); `wireSearch()` in shell (search button opens an input overlay; Enter / click navigates).

**Layout:** `pageTitle("<SPORT> Board", "Every game with the model's projection and best market.")`, pills All / Today / This Week; a slate table like Today's Slate with all markets per game (Time, Matchup with logos + star, Projected Score, Model Spread vs Market, Model Total vs Market, Best Opportunity (market + Alpha + Confidence), →). NFL/CFB: a second pill group "Games · Power Rankings" — Power Rankings renders `rankingsTable` (legacy, restyled) in place. MLB/NBA: `SPORT_STATUS[sport]` in a `.ca-card` empty state (and any MLB predictions still in the table for the selected date if present). Settings page: existing form inside a `.ca-card`, title via `pageTitle("Settings", "Sportsbooks, unit size, bankroll, Kelly and minimum EV.")`.

- [ ] **Step 1: Failing tests:**
```js
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const g = loadScripts(["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/board.js", "js/boot.js"]);
test("search finds teams, matchups and prop players", () => {
  const idx = g.searchIndex([{ sport: "nfl", game_pk: 1, home_team_name: "Kansas City Chiefs", away_team_name: "Detroit Lions" }],
                            [{ kind: "prop", sport: "nfl", game_pk: 1, playerName: "Amon-Ra St. Brown", marketLabel: "Rec Yds" }]);
  assert.ok(g.searchQuery(idx, "lions").some((r) => r.href === "game.html?sport=nfl&game=1"));
  assert.ok(g.searchQuery(idx, "st. brown").some((r) => /Rec Yds/.test(r.sub)));
  assert.equal(g.searchQuery(idx, "zzz").length, 0);
});
```
- [ ] **Steps 2–6:** run (FAIL) → implement → run (PASS) → visual check of nfl.html, cfb.html, mlb.html, nba.html, rankings.html, settings.html at 1440px / 390px → commit `feat(site): board pages, rankings tab, settings, search`.

---

### Task 12: Remove dead legacy UI, final QA, deploy prep

**Files:**
- Modify: `site/app.js` (delete `nav`, `buildDash`, the legacy `render()` branches now unused, `mountGradientBackground` and its call, any CSS rules for removed markup), `site/js/boot.js`
- Test: `site/tests/smoke.test.mjs` (extend)

- [ ] **Step 1: Extend the smoke test** — every page's builder exists and every HTML file loads the scripts in the documented order:
```js
import fs from "node:fs";
test("every page loads the scripts in order with the current cache key", () => {
  const order = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/ev.js", "js/pages/game.js", "js/pages/track.js", "js/pages/board.js", "js/boot.js"];
  for (const f of fs.readdirSync(new URL("..", import.meta.url)).filter((f) => f.endsWith(".html"))) {
    const html = fs.readFileSync(new URL(`../${f}`, import.meta.url), "utf8");
    const srcs = [...html.matchAll(/<script src="([^"?]+)\?v=([^"]+)"/g)];
    assert.deepEqual(srcs.map((m) => m[1]), order, f);
    assert.ok(srcs.every((m) => m[2] === srcs[0][2]), `${f}: one cache key`);
    assert.match(html, /css\/theme\.css/); assert.doesNotMatch(html, /styles\.css/);
  }
});
```
- [ ] **Step 2: Run — fix until PASS.** **Step 3:** delete the dead code; `npm test` green. **Step 4: QA pass** — all pages at 1440px and 390px; no console errors (`read_console_messages`); every number traceable to data (spot-check five against SQL); empty states for MLB/NBA; star/watchlist persists across reloads; 5-minute refresh keeps tab/filter/sort. **Step 5: Commit** `git commit -m "chore(site): remove dead legacy UI; final QA"`.
- [ ] **Step 6: Hand-off** (controller, not a subagent): user runs `db/migration_site_redesign_a.sql`; merge the PR; run `scripts/sync_site.sh`; user redeploys CappingAlpha.

---

## Self-review notes

- **Spec coverage:** §1 design system → Tasks 2–3, 6; §2 Dashboard → 7, +EV → 8, game page → 9, Track Record → 10, boards → 11; header / search / avatar / watchlist → 2, 11; §3 Alpha → 4–5; NBA/MLB empty states → 5, 11; §4 (Phase B) panels omitted: Recent Model Updates, venue/weather, quarter game flow, havoc/pressure insights, sportsbook logos (text badges until then) — each named where it would render.
- **Deviation from the spec (controller ruling):** spec §3 puts the Alpha Score in a SQL view; this plan computes it in one JS function (`alphaScore`, Task 4) used by every page — same single source of truth, unit-tested, no migration. Phase B ports it to Python when `projection_history` needs to store it.
- **Phase A migration:** only `line_moves_current` (+ index). If the user has not run it, Market Movers / Market Pulse line moves are simply empty.
