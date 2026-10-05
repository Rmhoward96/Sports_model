import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/dashboard.js", "js/pages/ev.js", "js/pages/game.js", "js/pages/track.js", "js/pages/board.js", "js/boot.js"];
const EVIL = '<img src=x onerror="alert(1)">';
const RAW = /<img src=x|<script>|<b>bad<\/b>/;

// A test document: the page shell records what render() paints; everything else is inert.
function testDoc(pageName) {
  const shell = { innerHTML: "", querySelector: () => null };
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
