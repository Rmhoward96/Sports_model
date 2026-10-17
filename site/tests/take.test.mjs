import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "tests/fixtures/shrink-k-identity.js", "js/shrink.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/take.js", "js/boot.js"];
const g = loadScripts(FILES);
const NOW = new Date("2026-11-02T15:45:00Z");
const P = (o) => ({ league: "nfl", market: "spread", side: "home", home_team: "Kansas City Chiefs", away_team: "Detroit Lions", line: -3.5, ref_line: -3.5,
  price: -105, ev: 0.021, better_line: false, active: true, commence_time: "2026-11-02T18:00:00Z", game_pk: 1, last_seen: "2026-11-02T15:40:00Z", ...o });

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
  assert.deepEqual(dropped.map((r) => r.event_id), ["e", "d", "c"], "most recent drop first; a started pick sorts by its start");
  assert.equal(rows.length, 5, "input untouched");
});
test("takeSplit puts stale picks (last_seen older than 150 minutes, or unparsable) after fresh ones, each group by edge", () => {
  const rows = [P({ event_id: "old-big", ev: 0.09, last_seen: "2026-11-02T13:00:00Z" }), P({ event_id: "fresh-small", ev: 0.02 }),
    P({ event_id: "bad", ev: 0.07, last_seen: "nope" }), P({ event_id: "none", ev: 0.06, last_seen: null }),
    P({ event_id: "edge", ev: 0.03, last_seen: "2026-11-02T13:15:00Z" }), P({ event_id: "fresh-big", ev: 0.04 })];
  const snapshot = JSON.stringify(rows);
  const { live } = g.takeSplit(rows, NOW);
  assert.deepEqual(live.map((r) => r.event_id), ["fresh-big", "edge", "fresh-small", "old-big", "bad", "none"], "13:15 is exactly 150 min old: still fresh");
  assert.equal(JSON.stringify(rows), snapshot, "input rows untouched");
});
test("takeRow shows the stale badge only on a stale live pick", () => {
  const base = { book: "draftkings", fair_prob: 0.52, good_until: -112, first_seen: "2026-11-02T10:00:00Z" };
  const stale = g.takeRow(P({ ...base, last_seen: "2026-11-02T12:00:00Z" }), null, NOW, false);
  assert.match(stale, /class="ca-take-stale">stale — recheck price</);
  assert.doesNotMatch(g.takeRow(P({ ...base, last_seen: "2026-11-02T15:40:00Z" }), null, NOW, false), /ca-take-stale/);
  assert.doesNotMatch(g.takeRow(P({ ...base, last_seen: "2026-11-02T12:00:00Z" }), null, NOW, true), /ca-take-stale/, "dropped rows are not badged");
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

test("takeProgressCard clamps the shown count to 300 while takeProgress keeps the raw one", () => {
  const rows = Array.from({ length: 340 }, () => ({ first_seen: "2026-11-02T12:00:00Z", graded_by: "clv", clv: 0.01, beat: true }));
  assert.equal(g.takeProgress(rows, null).graded, 340);
  assert.equal(g.takeProgress(rows, null).shown, 300);
  const card = g.takeProgressCard(rows, null, NOW);
  assert.match(card, /<b>300<\/b> of 300 picks graded/); assert.match(card, /aria-valuenow="300"/); assert.match(card, /width:100\.0%/);
  assert.doesNotMatch(card, /340/);
});
test("takeProgress compares instants, not strings: +00:00 at exactly Nov 1 counts, one second before does not", () => {
  const rows = [{ first_seen: "2026-11-01T00:00:00+00:00", graded_by: "clv", clv: 0.01, beat: true },
    { first_seen: "2026-10-31T23:59:59+00:00", graded_by: "clv", clv: 0.01, beat: true },
    { first_seen: "2026-10-31T20:00:00-04:00", graded_by: "clv", clv: 0.01, beat: true }];   // = Nov 1 00:00 UTC
  assert.equal(g.takeProgress(rows, null).graded, 2);
});
test("takePredFromRow derives margin and total from the expected scores", () => {
  assert.deepEqual(g.takePredFromRow({ pred_home_score: 27.5, pred_away_score: 20 }), { pred_margin: 7.5, pred_total: 47.5 });
  assert.equal(g.takePredFromRow({ pred_home_score: 27.5, pred_away_score: null }), null);
  assert.equal(g.takePredFromRow({ pred_home_score: null, pred_away_score: 20 }), null);
  assert.equal(g.takePredFromRow(null), null);
});
test("takeProgressCard: pre-Nov-1 sentence, progress bar after, verdict panel", () => {
  const rows = [{ first_seen: "2026-11-02T12:00:00Z", graded_by: "clv", clv: 0.02, beat: true }];
  const before = g.takeProgressCard(rows, null, new Date("2026-10-17T12:00:00Z"));
  assert.match(before, /Test starts Nov 1; October picks are a shakedown and are not judged/);
  assert.doesNotMatch(before, /ca-take-bar/);
  const after = g.takeProgressCard(rows, null, NOW);
  assert.match(after, /ca-take-bar/); assert.match(after, /<b>1<\/b> of 300 picks graded/); assert.match(after, /\+2\.00%/);
  const passed = g.takeProgressCard(rows, { passed: true, n: 300, beat_rate: 0.56, mean_clv: 0.012, lower: 0.004 }, NOW);
  assert.match(passed, /Passed/); assert.match(passed, /judged on 300 picks/); assert.match(passed, /95% lower bound \+0\.40%/);
  const failed = g.takeProgressCard(rows, { passed: false, n: 300, beat_rate: 0.5, mean_clv: null, lower: null }, NOW);
  assert.match(failed, /Did not pass/); assert.match(failed, /mean CLV —/); assert.doesNotMatch(failed, /lower bound/);
});
test("buildTakePage: empty state, and a listed pick gets its model note from predictions_any", async () => {
  const mk = (data) => loadScripts(FILES, { page: "take", globals: { fetch: async (u) => {
    const t = String(u).split("/rest/v1/")[1].split("?")[0]; assert.notEqual(t, "game_predictions");
    return { ok: true, json: async () => data[t] || [] }; } } });
  const empty = await mk({}).buildTakePage();
  assert.match(empty, /No bets beat Pinnacle&#39;s fair price right now\.|No bets beat Pinnacle's fair price right now\./);
  const soon = new Date(Date.now() + 3 * 3600e3).toISOString();
  const pick = P({ event_id: "e", book: "fanduel", fair_prob: 0.5, good_until: -110, first_seen: soon, last_seen: soon, commence_time: soon, others: [] });
  const html = await mk({ take_list_v: [pick], predictions_any: [{ sport: "nfl", game_pk: 1, pred_home_score: 28, pred_away_score: 21 }] }).buildTakePage();
  assert.match(html, /model agrees/);
});
