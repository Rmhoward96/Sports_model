import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/game.js", "js/boot.js"];
const g = loadScripts(FILES);

/* ── hero venue / weather line ──────────────────────────────────────────── */
const INFO = { venue_name: "Bryant-Denny Stadium", city: "Tuscaloosa", state: "AL", indoor: false, temp_f: 41.2, wind_mph: 14.4, precip_chance: 20, precip_in: null, conditions: null, weather_kind: "forecast" };

test("gmVenueLine: every piece, in order, joined with ' · '", () => {
  assert.equal(g.gmVenueLine(INFO), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph, 20% rain");
});

test("gmVenueLine: a missing piece is omitted, never guessed", () => {
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: null }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph");
  assert.equal(g.gmVenueLine({ ...INFO, state: null }), "Bryant-Denny Stadium · Tuscaloosa · 41°F, wind 14 mph, 20% rain");
  assert.equal(g.gmVenueLine({ ...INFO, city: "", state: "" }), "Bryant-Denny Stadium · 41°F, wind 14 mph, 20% rain");
  assert.equal(g.gmVenueLine({ ...INFO, venue_name: null, temp_f: null, wind_mph: null, precip_chance: null }), "Tuscaloosa, AL");
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: 0 }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph", "0% is not shown");
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: null, precip_in: 0.04 }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph, 0.04 in rain");
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: null, precip_in: 0.004 }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph", "below 0.01 in is not measurable");
  assert.equal(g.gmVenueLine({ ...INFO, conditions: "Cloudy", temp_f: "38" }), "Bryant-Denny Stadium · Tuscaloosa, AL · Cloudy, 38°F, wind 14 mph, 20% rain");
  assert.equal(g.gmVenueLine({ venue_name: "V", temp_f: null, wind_mph: "n/a" }), "V");
});

test("gmVenueLine: indoors replaces the weather; observed readings are labelled; nothing to show -> empty", () => {
  assert.equal(g.gmVenueLine({ ...INFO, indoor: true }), "Bryant-Denny Stadium · Tuscaloosa, AL · Indoors");
  assert.equal(g.gmVenueLine({ ...INFO, weather_kind: "observed", precip_chance: null }), "Bryant-Denny Stadium · Tuscaloosa, AL · Observed 41°F, wind 14 mph");
  for (const none of [null, undefined, {}, "x", { venue_name: "", city: " ", state: null, indoor: null }]) assert.equal(g.gmVenueLine(none), "");
});

/* ── page fixtures ──────────────────────────────────────────────────────── */
const urlPath = (url) => new URL(url).pathname.split("/").pop();
const PRED = (sport, over = {}) => ({ game_pk: 9, sport, home_team_name: sport === "cfb" ? "Alabama Crimson Tide" : "Kansas City Chiefs", away_team_name: sport === "cfb" ? "Georgia Bulldogs" : "Buffalo Bills",
  commence_time: new Date(Date.now() + 2 * 864e5).toISOString(), home_win_prob: 0.7, pred_home_score: 30, pred_away_score: 20, market_spread: -7.5, market_total: 50.5, ...over });
const HIST = [{ side: "away", team: "61", windows: { season: { n: 5, su: "4-1" } }, streaks: {} }, { side: "home", team: "333", windows: { season: { n: 5, su: "5-0" } }, streaks: {} }];
function page({ sport = "cfb", rows = {}, missing = [], pred = {} } = {}) {
  const search = `?sport=${sport}&game=9`;
  const requested = [];
  const fetch = async (url) => {
    const path = urlPath(url); requested.push(decodeURIComponent(path + new URL(url).search));
    if (missing.includes(path)) return { ok: false, status: 404, text: async () => "relation does not exist", json: async () => ({}) };
    const base = { predictions_any: [PRED(sport, pred)], team_history: sport === "cfb" ? HIST : [] };
    return { ok: true, json: async () => rows[path] ?? base[path] ?? [] };
  };
  const D = loadScripts(FILES, { page: "game", globals: { fetch, location: { search, href: `http://localhost/game.html${search}` } } });
  return { D, requested };
}
const heroOf = (html) => html.slice(html.indexOf("ca-gm-hero"), html.indexOf("ca-gm-read"));

test("hero: the venue line sits under the kickoff line, escaped, for CFB and NFL; absent without a game_info row or table", async () => {
  const evil = { ...INFO, venue_name: 'Bryant <b>"Denny"</b>' };
  for (const sport of ["cfb", "nfl"]) {
    const html = await page({ sport, rows: { game_info: [evil] } }).D.buildGamePage();
    const hero = heroOf(html);
    assert.ok(hero.includes('<p class="ca-gm-venue">Bryant &lt;b&gt;&quot;Denny&quot;&lt;/b&gt; · Tuscaloosa, AL · 41°F, wind 14 mph, 20% rain</p>'), sport);
    assert.ok(hero.indexOf("ca-gm-when") < hero.indexOf("ca-gm-venue") && hero.indexOf("ca-gm-venue") < hero.indexOf("ca-gm-teams"));
    assert.ok(!html.includes("<b>\"Denny\"</b>"));
  }
  const none = await page({ rows: { game_info: [] } }).D.buildGamePage();
  assert.ok(!none.includes("ca-gm-venue"), "no row: no line");
  const noTable = await page({ missing: ["game_info"] }).D.buildGamePage();
  assert.ok(!noTable.includes("ca-gm-venue") && noTable.includes("ca-gm-hero"), "missing table: the page still renders");
  const empty = await page({ rows: { game_info: [{ sport: "cfb", game_pk: 9 }] } }).D.buildGamePage();
  assert.ok(!empty.includes("ca-gm-venue"), "a row with nothing in it: no empty line");
});

test("game_info is read by sport and game_pk for NFL and CFB only", async () => {
  const { D, requested } = page();
  await D.buildGamePage();
  assert.ok(requested.includes("game_info?sport=eq.cfb&game_pk=eq.9"), requested.join("\n"));
  const mlb = page({ sport: "mlb" });
  await mlb.D.buildGamePage();
  assert.ok(!mlb.requested.some((r) => r.startsWith("game_info")), "MLB has no game_info rows");
});

/* ── R1: the weather label follows kickoff and weather_kind ──────────────── */
const NOW = Date.parse("2026-10-10T18:00:00Z");
const kick = (hoursFromNow) => new Date(NOW + hoursFromNow * 36e5).toISOString();
const WX = "41°F, wind 14 mph, 20% rain", BASE = "Bryant-Denny Stadium · Tuscaloosa, AL";

test("gmVenueLine: before kickoff the forecast is shown unlabelled", () => {
  assert.equal(g.gmVenueLine(INFO, kick(5), NOW), `${BASE} · ${WX}`);
  assert.equal(g.gmVenueLine(INFO, kick(24 * 6), NOW), `${BASE} · ${WX}`);
});

test("gmVenueLine: a CFB observed reading after kickoff is labelled Observed", () => {
  const obs = { ...INFO, weather_kind: "observed", precip_chance: null };
  assert.equal(g.gmVenueLine(obs, kick(-3), NOW), `${BASE} · Observed 41°F, wind 14 mph`);
});

test("gmVenueLine: a forecast after kickoff (NFL) is labelled Forecast", () => {
  const nfl = { ...INFO, wind_mph: null, conditions: null };
  assert.equal(g.gmVenueLine(nfl, kick(-3), NOW), `${BASE} · Forecast 41°F, 20% rain`);
  assert.equal(g.gmVenueLine(INFO, kick(-1), NOW), `${BASE} · Forecast ${WX}`);
});

test("gmVenueLine: weather is hidden once more than 3 days after kickoff; the venue stays", () => {
  assert.equal(g.gmVenueLine(INFO, kick(-24 * 3 - 1), NOW), BASE);
  assert.equal(g.gmVenueLine({ ...INFO, weather_kind: "observed" }, kick(-24 * 4), NOW), BASE);
  assert.equal(g.gmVenueLine(INFO, kick(-24 * 3 + 1), NOW), `${BASE} · Forecast ${WX}`, "just inside 3 days still shows");
  assert.equal(g.gmVenueLine({ ...INFO, indoor: true }, kick(-24 * 9), NOW), `${BASE} · Indoors`, "indoors is a fact about the venue, not the weather");
  assert.equal(g.gmVenueLine({ venue_name: null, city: null, state: null, temp_f: 41, weather_kind: "forecast" }, kick(-24 * 9), NOW), "", "nothing left -> hidden");
});

test("gmVenueLine: unknown kickoff falls back to the plain behaviour; unlabelled kind stays unlabelled", () => {
  assert.equal(g.gmVenueLine(INFO, null, NOW), `${BASE} · ${WX}`);
  assert.equal(g.gmVenueLine(INFO, "garbage", NOW), `${BASE} · ${WX}`);
  assert.equal(g.gmVenueLine({ ...INFO, weather_kind: null }, kick(-3), NOW), `${BASE} · ${WX}`);
});

test("hero: kickoff-aware label reaches the page (observed, kicked-off forecast, stale hidden)", async () => {
  const hours = (h) => new Date(Date.now() + h * 36e5).toISOString();
  const cases = [
    ["cfb", { ...INFO, weather_kind: "observed", precip_chance: null }, hours(-3), "· Observed 41°F, wind 14 mph</p>"],
    ["nfl", { ...INFO, wind_mph: null, weather_kind: "forecast" }, hours(-3), "· Forecast 41°F, 20% rain</p>"],
    ["cfb", INFO, hours(-24 * 5), "Tuscaloosa, AL</p>"],
  ];
  for (const [sport, info, ct, tail] of cases) {
    const html = await page({ sport, rows: { game_info: [info] }, pred: { commence_time: ct } }).D.buildGamePage();
    assert.ok(heroOf(html).includes(tail), `${sport} ${ct}: ${heroOf(html).match(/<p class="ca-gm-venue">.*?<\/p>/)}`);
  }
});
