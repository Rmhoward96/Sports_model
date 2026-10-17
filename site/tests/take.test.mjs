import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "tests/fixtures/shrink-k-identity.js", "js/shrink.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/take.js", "js/boot.js"];
const g = loadScripts(FILES);
const NOW = new Date("2026-11-02T15:45:00Z");
const P = (o) => ({ league: "nfl", market: "spread", side: "home", home_team: "Kansas City Chiefs", away_team: "Detroit Lions", line: -3.5, ref_line: -3.5,
  price: -105, ev: 0.021, better_line: false, active: true, commence_time: "2026-11-02T18:00:00Z", game_pk: 1, ...o });

test("takeAmerican uses a real minus sign", () => {
  assert.equal(g.takeAmerican(105), "+105");
  assert.equal(g.takeAmerican(-110), "−110");
  assert.equal(g.takeAmerican(null), "—");
});
test("takeBetLabel", () => {
  assert.equal(g.takeBetLabel({ league: "epl", market: "ml", side: "draw" }), "Draw");
  assert.equal(g.takeBetLabel({ league: "mlb", market: "total", side: "over", line: 8.5 }), "Over 8.5");
  assert.equal(g.takeBetLabel({ league: "mlb", market: "total", side: "under", line: 8.5 }), "Under 8.5");
  assert.equal(g.takeBetLabel(P({})), "Chiefs −3.5", "favourite: negative line");
  assert.equal(g.takeBetLabel(P({ side: "away", line: 3.5 })), "Lions +3.5", "underdog: positive line");
  assert.equal(g.takeBetLabel({ league: "epl", market: "ml", side: "home", home_team: "Arsenal", away_team: "Chelsea" }), "Arsenal ML");
});
test("takeEdge marks better line", () => {
  assert.equal(g.takeEdge({ ev: 0.021, better_line: true }), "≥ 2.1%");
  assert.equal(g.takeEdge({ ev: 0.021, better_line: false }), "2.1%");
});
test("takeCountdown", () => {
  assert.equal(g.takeCountdown("2026-11-02T18:00:00Z", NOW), "2h 15m");
  assert.equal(g.takeCountdown("2026-11-03T19:00:00Z", NOW), "1d 3h");
  assert.equal(g.takeCountdown("2026-11-02T16:30:00Z", NOW), "45m");
  assert.equal(g.takeCountdown("2026-11-02T15:00:00Z", NOW), "Started");
});
test("takeSplit drops started and inactive picks", () => {
  const rows = [P({ event_id: "a", ev: 0.02 }), P({ event_id: "b", ev: 0.05 }), P({ event_id: "c", commence_time: "2026-11-02T14:00:00Z" }),
    P({ event_id: "d", active: false, dropped_at: "2026-11-02T15:00:00Z" }), P({ event_id: "e", active: false, dropped_at: "2026-11-02T15:30:00Z" })];
  const { live, dropped } = g.takeSplit(rows, NOW);
  assert.deepEqual(live.map((r) => r.event_id), ["b", "a"], "best edge first");
  assert.deepEqual(dropped.map((r) => r.event_id), ["e", "d", "c"].sort((x, y) => ({ e: 0, d: 1, c: 2 }[x] - { e: 0, d: 1, c: 2 }[y])), "most recent drop first; a started pick sorts by its start");
  assert.equal(rows.length, 5, "input untouched");
});
test("takeModelNote agrees/disagrees/blank", () => {
  const pred = { pred_margin: 7, pred_total: 50 };            // identity shrink: model leans Chiefs by 7 against a -3.5 line
  assert.equal(g.takeModelNote(P({ league: "nhl" }), pred), "", "not a model league");
  assert.equal(g.takeModelNote(P({ market: "ml" }), pred), "", "moneyline");
  assert.equal(g.takeModelNote(P({}), null), "", "no prediction");
  assert.equal(g.takeModelNote(P({}), pred), "model agrees", "home pick, model leans home");
  assert.equal(g.takeModelNote(P({ side: "away", line: 3.5, ref_line: 3.5 }), pred), "model disagrees", "away pick, model leans home");
  assert.equal(g.takeModelNote(P({}), { pred_margin: 3.5 }), "", "model equals the line: no lean");
  assert.equal(g.takeModelNote(P({ market: "total", side: "over", ref_line: 44.5 }), pred), "model agrees");
  assert.equal(g.takeModelNote(P({ market: "total", side: "under", ref_line: 44.5 }), pred), "model disagrees");
});
test("takeProgress ignores October and no_close", () => {
  const rows = [
    { first_seen: "2026-11-02T12:00:00Z", graded_by: "clv", clv: 0.02, beat: true },
    { first_seen: "2026-11-03T12:00:00Z", graded_by: "direction", clv: null, beat: false },
    { first_seen: "2026-10-30T12:00:00Z", graded_by: "clv", clv: 0.5, beat: true },
    { first_seen: "2026-11-04T12:00:00Z", graded_by: "no_close", clv: null, beat: null },
    { first_seen: "2026-11-05T12:00:00Z", graded_by: null, clv: null, beat: null }];
  const p = g.takeProgress(rows, null);
  assert.equal(p.graded, 2); assert.equal(p.needed, 300); assert.equal(p.beatRate, 0.5); assert.equal(p.meanClv, 0.02);
  assert.equal(p.judged, false); assert.equal(p.passed, null);
  const v = g.takeProgress(rows, { passed: true });
  assert.equal(v.judged, true); assert.equal(v.passed, true);
  assert.equal(g.takeProgress([], null).beatRate, null);
});
test("takeRow renders the pick, escapes names and shows the other books", () => {
  const html = g.takeRow(P({ home_team: "A<b>", others: [{ book: "fanduel", line: -3.5, price: -108, ev: 0.018 }], fair_prob: 0.52, good_until: -112,
    first_seen: "2026-11-02T15:30:00Z", last_seen: "2026-11-02T15:44:00Z", book: "draftkings" }), { pred_margin: 7 }, NOW, false);
  assert.match(html, /2h 15m/); assert.match(html, /FanDuel −3\.5 −108/); assert.match(html, /model agrees/); assert.match(html, /−112/);
  assert.ok(!html.includes("A<b>"));
});
