import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { loadScripts } from "./load.mjs";
const require_src = (f) => fs.readFileSync(new URL("../" + f, import.meta.url), "utf8");
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/ev.js", "js/pages/game.js", "js/pages/track.js", "js/pages/board.js", "js/pages/board-left.js", "js/boot.js"];
const EVIL = '<img src=x onerror="alert(1)">';
const RAW = /<img src=x|<script>|<b>bad<\/b>/;

// A test document: the page shell records what render() paints; everything else is inert.
function testDoc(pageName) {
  const shell = { innerHTML: "", querySelector: (sel) => (sel === ".ca-header" && shell.innerHTML.includes('class="ca-header"') ? {} : null) };
  return { shell, doc: {
    body: { dataset: { page: pageName }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {}, insertAdjacentHTML() {} },
    documentElement: {}, activeElement: null, createElement: () => ({ style: {}, dataset: {} }), getElementById: () => null,
    querySelector: (sel) => (sel === ".page-shell" ? shell : null), querySelectorAll: () => [], addEventListener() {}, dispatchEvent() {},
  } };
}

test("legacy sections escape every database string: parlays, graded parlays, +EV / prop track tables, props by game", () => {
  const g = loadScripts(FILES);
  const leg = { matchup: `${EVIL} @ <b>bad</b>`, side: "home", market: "moneyline", price: -300, label: EVIL, kind: "game", sport: "nfl", result: '"><script>', commence_time: "2026-10-05T17:00:00Z", prob: 0.7 };
  const html = [
    g.parlaySection([{ legs: [leg, { ...leg, side: "away" }], n_legs: EVIL, parlay_price: 150, book: "<script>x</script>", ev: 0.1, true_prob: 0.4, sport: "nfl" }]),
    g.bestParlaysSection([{ legs: JSON.stringify([leg]), parlay_price: 200, parlay_dec: 3, book: EVIL, ev: 0.05, true_prob: 0.35 }], g.getSettings()),
    g.gradedParlaysSection([{ legs: [leg], parlay_price: 300, book: "<script>x</script>", result: '"><script>', pnl: 10, first_commence: "2026-10-05T17:00:00Z" }]),
    g.evTrackSection([{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", won: true, clv: 0.01 }, { sport: "nfl", game_pk: 2, market: EVIL, side: "<b>bad</b>", won: false }],
      [{ sport: "nfl", game_pk: 1, market: "moneyline", side: "home", matchup: `${EVIL} @ <b>bad</b>`, best_price: 110, best_book: "<script>x</script>", ev_best: 0.02 }]),
    g.propTrackSection([{ player_name: EVIL, market: "<b>bad</b>", side: "over", line: 50.5, actual: 60, result: "win", clv: 0.01 }]),
    g.propGameDetail([{ player_name: EVIL, team: "<b>bad</b>", market: "rec_yds", line: 50.5, projection: 55, lean: "over", bet_price: -110, actual: 60, result: "hit", pnl: 9.1 }]),
  ].join("\n");
  assert.doesNotMatch(html, RAW, "no raw markup from the database");
  assert.ok(html.includes("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;") && html.includes("&lt;script&gt;x&lt;/script&gt;") && html.includes("&lt;b&gt;bad&lt;/b&gt;"));
  assert.match(g.evSortAttrs(0.1, 150, '"><b>bad</b>', 0.5), /data-sbook="&quot;&gt;&lt;b&gt;bad&lt;\/b&gt;"/, "attribute value escaped");
  // props by game: the week list escapes the matchup and the game id attribute
  const P = loadScripts(FILES);
  P.propGamesSection([{ season: 2026, week: 5, game_pk: '1" onclick="x', matchup: EVIL, commence_time: "2026-10-05T17:00:00Z", n: 3, hits: 2, misses: 1, pushes: 0, pnl: 5, ev_n: 0 }]);
  const wk = P.propGamesWeek("2026-5");
  assert.doesNotMatch(wk, RAW); assert.ok(wk.includes('data-game="1&quot; onclick=&quot;x"'));
});

test("the error page escapes the error message", async () => {
  const { shell, doc } = testDoc(EVIL);   // an unknown page name ends up in the thrown message
  const errors = [];
  const g = loadScripts(FILES, { globals: { document: doc, console: { ...console, error: (...a) => errors.push(a) }, scrollY: 0, scrollTo() {} } });
  await g.render();
  assert.ok(shell.innerHTML.includes("Couldn’t load data"));
  assert.ok(shell.innerHTML.includes("Unknown page: &lt;img src=x onerror=&quot;alert(1)&quot;&gt;"));
  assert.doesNotMatch(shell.innerHTML, /<img src=x/);
  assert.equal(errors.length, 1);
});

// Rankings page with a fetch whose responses the test releases one by one (call order = render order).
function deferredRankings() {
  const { shell, doc } = testDoc("rankings");
  const pending = [];
  const fetch = (url) => new Promise((resolve) => pending.push({ url, resolve }));
  const g = loadScripts(FILES, { page: "rankings", globals: { document: doc, fetch, scrollY: 0, scrollTo() {}, location: { search: "?sport=nfl", href: "http://localhost/rankings.html?sport=nfl" } } });
  const reply = (i, rows) => pending[i].resolve({ ok: true, json: async () => rows });
  return { g, shell, doc, pending, reply };
}
const tick = () => new Promise((r) => setTimeout(r, 0));
const RK_ROW = { sport: "nfl", team: "KC", rank: 1, season: 2026, week: 5, rating: 6.1, su: "4-0", units: {} };

test("render(): the header and a small Loading… paint at once on first load; a re-render does not flash Loading…", async () => {
  const R = deferredRankings();
  const done = R.g.render();
  await tick();
  assert.ok(R.shell.innerHTML.includes('class="ca-header"'), "header painted before the data arrives");
  assert.ok(R.shell.innerHTML.includes('<p class="ca-loading">Loading…</p>'));
  R.reply(0, [RK_ROW]); await done;
  assert.ok(R.shell.innerHTML.includes("Week 5") && !R.shell.innerHTML.includes("Loading…"));
  const again = R.g.render();
  await tick();
  assert.ok(!R.shell.innerHTML.includes("Loading…"), "the 5-minute refresh keeps the current page up while it loads");
  R.reply(1, [RK_ROW]); await again;
});

test("render(): an older in-flight render never paints over a newer one (render token)", async () => {
  const R = deferredRankings();
  const first = R.g.render(), second = R.g.render();
  await tick();
  assert.equal(R.pending.length, 2);
  R.reply(1, [RK_ROW]); await second;          // the newer render lands first
  assert.ok(R.shell.innerHTML.includes("Week 5"));
  R.reply(0, []); await first;                 // the stale one resolves later with other data
  assert.ok(R.shell.innerHTML.includes("Week 5") && !R.shell.innerHTML.includes("No NFL power rankings yet"), "stale render dropped");
});

test("render({auto:true}) skips the refresh tick while focus is in a field on the page; manual renders still run", async () => {
  const R = deferredRankings();
  R.doc.activeElement = { closest: (sel) => (sel === ".ca-page" ? {} : null), matches: (sel) => /input/.test(sel) };
  await R.g.render({ auto: true });
  assert.equal(R.pending.length, 0, "no fetch, no repaint while the user types in a page field");
  assert.equal(R.shell.innerHTML, "");
  const manual = R.g.render(); await tick();
  assert.equal(R.pending.length, 1, "a manual render (refresh button, reset) still runs");
  R.reply(0, [RK_ROW]); await manual;
  R.doc.activeElement = { closest: () => null, matches: () => true };   // focus in the header search, not the page
  const auto = R.g.render({ auto: true }); await tick();
  assert.equal(R.pending.length, 2, "focus outside .ca-page does not block the refresh");
  R.reply(1, [RK_ROW]); await auto;
  assert.ok(require_src("js/boot.js").includes("render({ auto: true })"), "the 5-minute interval passes auto");
});

test("the footer is styled in theme.css (muted 13px, aligned to the .ca-page content box, top border)", () => {
  const css = require_src("css/theme.css");
  const rule = css.match(/\nfooter\{([^}]*)\}/);
  assert.ok(rule, "a footer rule exists");
  for (const d of ["width:calc(100% - 2 * var(--gutter))", "margin:0 auto", "border-top:1px solid var(--line)", "color:var(--muted)", "font-size:13px"]) assert.ok(rule[1].includes(d), d);
  assert.match(css, /@media \(max-width:620px\)\{[^@]*:root\{--gutter:16px\}/, "phone gutter is shared by header, page and footer");
});

test("fidelity G1/G3: full-width page and header on the shared --gutter, stars hidden until hover / focus / starred", () => {
  const css = require_src("css/theme.css");
  assert.match(css, /--gutter:clamp\(20px,3vw,64px\)/);
  const page = css.match(/\n\.ca-page\{([^}]*)\}/)[1], head = css.match(/\n\.ca-header-inner\{([^}]*)\}/)[1];
  assert.ok(!/max-width/.test(page) && !/max-width/.test(head), "no 1440px cap on the page or the header");
  assert.ok(page.includes("var(--gutter)") && head.includes("var(--gutter)"), "header contents align with the page gutters");
  assert.match(css, /\n\.ca-star\{[^}]*opacity:0/, "stars are hidden by default");
  const show = css.match(/\n\.ca-star\.on,([^{]*)\{opacity:1\}/);
  assert.ok(show, "a rule shows them again");
  for (const sel of [".ca-star:focus-visible", ":hover>.ca-star", "tr:hover .ca-star", ".ca-gm-team:hover .ca-star"]) assert.ok(show[1].includes(sel), sel);
  assert.match(css, /@media \(hover:none\)\{\.ca-star\{opacity:1\}\}/, "touch screens keep visible stars");
});

test("fidelity M1: the game hero does not clip, and the odds boxes straddle its bottom edge", () => {
  const css = require_src("css/theme.css");
  const hero = css.match(/\n\.ca-gm-hero\{([^}]*)\}/)[1];
  assert.ok(!/overflow:hidden/.test(hero) && hero.includes("display:flow-root"));
  assert.match(css, /\.ca-gm-hero\.has-odds \.ca-gm-odds\{margin-bottom:calc\(var\(--ob-h\) \/ -2\)\}/);
  assert.match(css, /\.ca-gm-hero\.has-odds\{padding-bottom:0;/);
});
