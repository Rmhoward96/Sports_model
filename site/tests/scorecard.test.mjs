import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "tests/fixtures/shrink-k-identity.js", "js/shrink.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pikkit.js", "js/pages/scorecard.js", "js/boot.js"];
const g = loadScripts(FILES);

const bet = (placedIso, beat, o = {}) => ({ id: "x", book: "FanDuel", type: "straight", status: "SETTLED_WIN", odds: beat ? 2 : 1.8, close: beat === null ? null : 1.9,
  placed: new Date(placedIso), legs: 1, market: "spread", league: "NFL", tag: "own", units: 1, profitUnits: 0.5, ...o });
const pick = (firstSeen, beat, graded_by = "clv") => ({ first_seen: firstSeen, graded_by, beat });

test("scSideBySide uses the user's date range", () => {
  const bets = [bet("2026-10-05T12:00:00Z", true), bet("2026-10-08T12:00:00Z", false), bet("2026-10-10T12:00:00Z", true), bet("2026-10-09T12:00:00Z", null)];
  const rows = [pick("2026-10-04T23:59:59Z", true), pick("2026-10-05T12:00:00Z", true), pick("2026-10-07T00:00:00Z", false), pick("2026-10-09T00:00:00Z", true, "direction"),
    pick("2026-10-10T12:00:00Z", false), pick("2026-10-10T12:00:01Z", true), pick("2026-10-06T00:00:00Z", null, "no_close"), pick("2026-10-06T00:00:00Z", true, null)];
  const s = g.scSideBySide(bets, rows);
  assert.equal(new Date(s.from).toISOString(), "2026-10-05T12:00:00.000Z"); assert.equal(new Date(s.to).toISOString(), "2026-10-10T12:00:00.000Z");
  assert.equal(s.mine.n, 3); assert.equal(s.mine.rate, 2 / 3);
  assert.equal(s.list.n, 4, "in range and graded by clv or direction: the two before/after the range, no_close and ungraded are out");
  assert.equal(s.list.rate, 0.5);
});
test("scSideBySide with no bets or no picks", () => {
  const none = g.scSideBySide([], [pick("2026-10-05T12:00:00Z", true)]);
  assert.equal(none.from, null); assert.equal(none.list.rate, null); assert.equal(none.mine.rate, null);
  const s = g.scSideBySide([bet("2026-10-05T12:00:00Z", true)], null);
  assert.equal(s.list.n, 0); assert.equal(s.list.rate, null); assert.equal(s.mine.rate, 1);
});

const HEAD = "bet_id,sportsbook,type,status,odds,closing_line,ev,amount,profit,time_placed,time_settled,time_placed_iso,time_settled_iso,bet_info,tags,sports,leagues";
const csvRow = (id, o = {}) => {
  const r = { bet_id: id, sportsbook: "Draftkings Sportsbook", type: "straight", status: "SETTLED_WIN", odds: "2.0", closing_line: "1.9", ev: "", amount: "25", profit: "25", time_placed: "",
    time_settled: "", time_placed_iso: "2026-10-02T15:00:00Z", time_settled_iso: "", bet_info: "Over 47.5 Total Points BUF Bills @ KC Chiefs", tags: "", sports: "Football", leagues: "NFL", ...o };
  return HEAD.split(",").map((h) => `"${String(r[h]).replace(/"/g, '""')}"`).join(",");
};
const CSV = [HEAD, csvRow("1"), csvRow("2", { odds: "1.8", status: "SETTLED_LOSS", profit: "-25" }), csvRow("3", { status: "SETTLED_VOID" }), csvRow("4", { closing_line: "" })].join("\n");

test("scResultsHtml renders every section from a synthetic export", () => {
  const html = g.scResultsHtml(CSV, 25);
  for (const id of ["sc-weekly", "sc-rolling", "sc-side", "sc-by-league", "sc-by-book", "sc-by-market", "sc-by-type", "sc-by-legs", "sc-by-stake", "sc-by-tag"]) assert.match(html, new RegExp(`id="${id}"`), id);
  assert.match(html, /Week of Sep 28/); assert.match(html, /DraftKings/); assert.match(html, /untagged/); assert.match(html, /Pikkit's close is not necessarily Pinnacle's\./);
  assert.match(html, /<div class="ca-stat-value ">3<\/div>/, "3 bets after the VOID is dropped");
});
test("scResultsHtml escapes the user's strings", () => {
  const html = g.scResultsHtml([HEAD, csvRow("1", { tags: "<img src=x onerror=alert(1)>", sportsbook: "Evil<b>Book", leagues: "<script>NFL" })].join("\n"), 1);
  assert.ok(!html.includes("<img src=x"), "tag escaped"); assert.ok(!html.includes("Evil<b>"), "book escaped"); assert.ok(!html.includes("<script>NFL"), "league escaped");
  assert.match(html, /&lt;img src=x/);
});
test("scResultsHtml: empty, junk and no-usable-bets inputs give messages, not exceptions", () => {
  assert.match(g.scResultsHtml(null, 1), /Drop your Pikkit export/);
  assert.match(g.scResultsHtml("not,a,pikkit\nfile,here,ok", 1), /No usable bets/);
  assert.match(g.scResultsHtml(HEAD, 1), /No usable bets/);
});
test("the alert banner shows only when the rolling rate is under 40%", () => {
  const rows = (beat) => [HEAD, ...Array.from({ length: 30 }, (_, i) => csvRow(String(i), { odds: i < beat ? "2.0" : "1.8", time_placed_iso: new Date(Date.parse("2026-10-02T15:00:00Z") + i * 60000).toISOString() }))].join("\n");
  assert.match(g.scResultsHtml(rows(6), 1), /id="sc-alert"[^>]*role="alert"|role="alert"[^>]*id="sc-alert"/);
  assert.match(g.scResultsHtml(rows(6), 1), /Slipping/);
  assert.ok(!g.scResultsHtml(rows(20), 1).includes("Slipping"));
});

test("buildScorecardPage reads the saved file and unit from localStorage and survives a failed take_test_v read", async () => {
  const storage = new Map([["scorecard.csv", CSV], ["scorecard.unit", "25"]]);
  const h = loadScripts(FILES, { storage });
  const html = await h.buildScorecardPage();
  assert.match(html, /id="sc-drop"/); assert.match(html, /value="25"/); assert.match(html, /id="sc-weekly"/);
  assert.ok(!/id="sc-clear"[^>]*hidden/.test(html), "clear button shown when a file is saved");
  const empty = await g.buildScorecardPage();
  assert.match(empty, /Drop your Pikkit export above/); assert.match(empty, /id="sc-clear" hidden|id="sc-clear"[^>]*hidden/); assert.match(empty, /value="1"/);
});
test("storage helpers swallow a throwing localStorage", () => {
  const h = loadScripts(FILES, { globals: { localStorage: { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, removeItem() { throw new Error("blocked"); } } } });
  assert.equal(h.scGet("scorecard.csv"), null); assert.equal(h.scSet("k", "v"), false); assert.equal(h.scUnit(), 1); h.scDel("k");
});

test("the list's rate ignores graded picks whose beat is null", () => {
  const bets = [bet("2026-10-05T12:00:00Z", true), bet("2026-10-10T12:00:00Z", true)];
  const s = g.scSideBySide(bets, [pick("2026-10-06T00:00:00Z", true), pick("2026-10-07T00:00:00Z", null), pick("2026-10-08T00:00:00Z", null, "direction"), pick("2026-10-09T00:00:00Z", false)]);
  assert.equal(s.list.n, 2); assert.equal(s.list.rate, 0.5);
  assert.equal(g.scSideBySide(bets, [pick("2026-10-06T00:00:00Z", null)]).list.rate, null);
});
test("the alert banner states the real window size", () => {
  const rows = [HEAD, ...Array.from({ length: 12 }, (_, i) => csvRow(String(i), { odds: i < 2 ? "2.0" : "1.8", time_placed_iso: new Date(Date.parse("2026-10-02T15:00:00Z") + i * 60000).toISOString() }))].join("\n");
  const html = g.scResultsHtml(rows, 1);
  const banner = html.slice(html.indexOf('id="sc-alert"'), html.indexOf("</section>", html.indexOf('id="sc-alert"')));
  assert.match(banner, /last 12 bets with a close/); assert.ok(!/last 100 bets/.test(banner));
});

// A tiny fake page: elements keep their handlers so a test can fire change events; FileReader reads the stub file's text synchronously.
function fakePage(extra = {}) {
  const els = {};
  for (const id of ["sc-drop", "sc-file", "sc-unit", "sc-clear", "sc-results", "sc-side"]) {
    els[id] = { id, innerHTML: "", outerHTML: "", hidden: false, value: "", files: null, handlers: {}, classList: { add() {}, remove() {}, toggle() {} }, addEventListener(t, f) { this.handlers[t] = f; } };
  }
  const el = () => ({ innerHTML: "", style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} }, appendChild() {}, setAttribute() {}, addEventListener() {}, querySelector: () => null, querySelectorAll: () => [] });
  const document = { body: { dataset: { page: "scorecard" }, appendChild() {}, classList: { add() {}, remove() {} } }, head: { appendChild() {} }, documentElement: el(),
    createElement: el, createElementNS: el, getElementById: (id) => els[id] || null, querySelector: () => null, querySelectorAll: () => [], addEventListener() {} };
  class FileReader { readAsText(f) { this.result = f.text; this.onload(); } }
  return { els, globals: { document, FileReader, ...extra } };
}
const BLOCKED = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("full"); }, removeItem() { throw new Error("blocked"); } };

test("a dropped file renders and the unit change sticks when localStorage throws on every call", async () => {
  const { els, globals } = fakePage({ localStorage: BLOCKED });
  const h = loadScripts(FILES, { globals });
  const html = await h.buildScorecardPage();
  assert.match(html, /Drop your Pikkit export above/);
  h.wireScorecardPage();
  els["sc-file"].files = [{ text: CSV }];
  els["sc-file"].handlers.change();
  assert.match(els["sc-results"].innerHTML, /id="sc-weekly"/, "the view renders although nothing could be saved");
  assert.match(els["sc-results"].innerHTML, /<div class="ca-stat-value ">75\.0u<\/div>/, "unit size 1: 75 staked");
  assert.equal(els["sc-clear"].hidden, false);
  els["sc-unit"].value = "25"; els["sc-unit"].handlers.change();
  assert.equal(els["sc-unit"].value, "25", "the input does not snap back");
  assert.match(els["sc-results"].innerHTML, /<div class="ca-stat-value ">3\.0u<\/div>/, "unit size 25: 3 staked");
  els["sc-unit"].value = "-4"; els["sc-unit"].handlers.change();
  assert.equal(els["sc-unit"].value, "1", "an invalid size falls back to 1");
  els["sc-clear"].handlers.click();
  assert.match(els["sc-results"].innerHTML, /Drop your Pikkit export above/); assert.equal(els["sc-clear"].hidden, true);
});
test("a dropped file is mirrored to localStorage and read back on the next load", async () => {
  const storage = new Map();
  const a = fakePage(), h = loadScripts(FILES, { storage, globals: a.globals });
  await h.buildScorecardPage(); h.wireScorecardPage();
  a.els["sc-file"].files = [{ text: CSV }]; a.els["sc-file"].handlers.change();
  a.els["sc-unit"].value = "25"; a.els["sc-unit"].handlers.change();
  assert.equal(storage.get("scorecard.csv"), CSV); assert.equal(storage.get("scorecard.unit"), "25");
  const h2 = loadScripts(FILES, { storage });
  const html = await h2.buildScorecardPage();
  assert.match(html, /id="sc-weekly"/); assert.match(html, /value="25"/);
});

test("the saved view paints before take_test_v answers, then the side-by-side fills in", async () => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const fetchStub = async () => { await gate; return { ok: true, json: async () => [{ first_seen: "2026-10-02T15:00:00Z", graded_by: "clv", beat: true }] }; };
  const { els, globals } = fakePage({ fetch: fetchStub });
  const h = loadScripts(FILES, { storage: new Map([["scorecard.csv", CSV]]), globals });
  const html = await h.buildScorecardPage();            // resolves while the read is still pending
  assert.match(html, /id="sc-weekly"/); assert.match(html, /Loading the list's results/);
  h.wireScorecardPage();
  release(); await new Promise((r) => setTimeout(r, 10));
  assert.match(els["sc-side"].outerHTML, /Take These list/); assert.match(els["sc-side"].outerHTML, /1 graded picks/); assert.match(els["sc-side"].outerHTML, /Pikkit's close is not necessarily/);
});
test("a failed take_test_v read leaves a short unavailable note", async () => {
  const { els, globals } = fakePage();   // the default test fetch rejects
  const h = loadScripts(FILES, { storage: new Map([["scorecard.csv", CSV]]), globals });
  await h.buildScorecardPage(); h.wireScorecardPage();
  await new Promise((r) => setTimeout(r, 10));
  assert.match(els["sc-side"].outerHTML, /list data unavailable/); assert.match(els["sc-side"].outerHTML, /id="sc-side"/);
});
