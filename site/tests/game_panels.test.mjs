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

/* ── Key Insights: havoc, turnover, weather candidates ──────────────────── */
const R = (sport = "cfb") => PRED(sport);
const insRow = (team, o = {}) => ({ season: 2026, team, games: 5, n_ranked: 134, def_havoc_rate: 0.2132, def_havoc_rank: 60, off_havoc_allowed_rate: 0.1812, off_havoc_allowed_rank: 60,
  turnover_margin: 0, turnover_margin_per_game: 0, turnover_margin_rank: 67, ...o });
const base = (o = {}) => ({ sport: "cfb", r: R(), ctxGrades: [], ctxHist: HIST, ctxPower: [], cfbIns: [], gameInfo: null, ...o });
const kinds = (D) => g.gmKeyInsights(D).map((i) => i.kind);

test("havoc mismatch: a 30+ rank gap (defense edge), or an edge involving a top / bottom 15% unit; strength 50 + gap / 3 (max 95)", () => {
  const D = base({ cfbIns: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118, off_havoc_allowed_rate: 0.2491 })] });
  const [h] = g.gmKeyInsights(D);
  assert.equal(h.kind, "havoc"); assert.equal(h.strength, 87);      // 50 + min(45, 110 / 3)
  assert.equal(h.title, "Alabama defense creates havoc against Georgia");
  assert.equal(h.body, "Alabama ranks #8 in havoc created (21.3% of plays); Georgia's offense ranks #118 in havoc allowed (24.9%), where #1 allows the least.");
  // gap under 30 but the defense is top 15% (<= ceil(.15 * 134) = 21) and has the edge
  const ext = g.gmKeyInsights(base({ cfbIns: [insRow("333", { def_havoc_rank: 5 }), insRow("61", { off_havoc_allowed_rank: 15 })] }));
  assert.deepEqual([ext[0].kind, ext[0].strength], ["havoc", 53]);
  // the offense is bottom 15% (> 134 - 21 = 113) with a small edge
  assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: 110 }), insRow("61", { off_havoc_allowed_rank: 114 })] })).includes("havoc"), true);
  // not notable: gap 29 in the middle of the pack; a negative gap (the offense protects the ball better); extremes without the edge
  for (const [d, o] of [[40, 69], [90, 30], [10, 5], [60, 60]])
    assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: d }), insRow("61", { off_havoc_allowed_rank: o })] })).includes("havoc"), false, `${d}/${o}`);
});

test("havoc mismatch: the stronger direction wins; unranked / missing rows and non-CFB games show nothing", () => {
  const D = base({ cfbIns: [insRow("333", { def_havoc_rank: 8, off_havoc_allowed_rank: 125 }), insRow("61", { def_havoc_rank: 40, off_havoc_allowed_rank: 100 })] });
  const [h] = g.gmKeyInsights(D);                                    // home D 8 vs away O 100 = 92; away D 40 vs home O 125 = 85
  assert.match(h.title, /^Alabama defense creates havoc against Georgia$/); assert.equal(h.strength, 81);
  const unranked = base({ cfbIns: [insRow("333", { def_havoc_rank: null }), insRow("61", { off_havoc_allowed_rank: 118 })] });
  assert.equal(kinds(unranked).includes("havoc"), false);
  assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: 8, n_ranked: null }), insRow("61", { off_havoc_allowed_rank: 118, n_ranked: undefined })] })).includes("havoc"), false, "no field size, no 15% rule");
  assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: 8 })] })).includes("havoc"), false, "one team only");
  assert.equal(kinds(base({ sport: "nfl", cfbIns: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })] })).includes("havoc"), false, "CFB only");
  const old = [insRow("333", { season: 2025, def_havoc_rank: 90 }), insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })];   // newest season first in practice
  assert.equal(g.gmKeyInsights(base({ cfbIns: [old[1], old[0], old[2]] }))[0].kind, "havoc", "the first (newest) row per team is used");
});

test("turnover edge: rank gap 30+ or a top / bottom 15% team; names the better team; equal ranks show nothing", () => {
  const D = base({ cfbIns: [insRow("333", { turnover_margin_rank: 6, turnover_margin_per_game: 1.8 }), insRow("61", { turnover_margin_rank: 112, turnover_margin_per_game: -0.9 })] });
  const [t] = g.gmKeyInsights(D);
  assert.equal(t.kind, "turnover"); assert.equal(t.strength, 85);   // 50 + min(45, 106 / 3)
  assert.equal(t.title, "Alabama owns the turnover edge");
  assert.equal(t.body, "Alabama is +1.8 per game (#6 nationally); Georgia is −0.9 per game (#112).");
  const away = g.gmKeyInsights(base({ cfbIns: [insRow("333", { turnover_margin_rank: 90 }), insRow("61", { turnover_margin_rank: 20, turnover_margin_per_game: 1.1 })] }));
  assert.equal(away[0].title, "Georgia owns the turnover edge", "gap 70, rank 20 is inside the top 21");
  assert.equal(kinds(base({ cfbIns: [insRow("333", { turnover_margin_rank: 10 }), insRow("61", { turnover_margin_rank: 25 })] })).includes("turnover"), true, "gap 15 but a top-15% team");
  for (const [a, b] of [[50, 60], [67, 67], [30, 59]]) assert.equal(kinds(base({ cfbIns: [insRow("333", { turnover_margin_rank: a }), insRow("61", { turnover_margin_rank: b })] })).includes("turnover"), false, `${a}/${b}`);
  assert.equal(kinds(base({ cfbIns: [insRow("333", { turnover_margin_rank: null }), insRow("61", { turnover_margin_rank: 112 })] })).includes("turnover"), false);
  const bare = g.gmKeyInsights(base({ cfbIns: [insRow("333", { turnover_margin_rank: 6, turnover_margin_per_game: null }), insRow("61", { turnover_margin_rank: 112, turnover_margin_per_game: null })] }));
  assert.equal(bare[0].body, "Alabama ranks (#6 nationally); Georgia ranks (#112).", "no per-game number, no invented one");
});

test("weather insight: wind 15+, under 40°F, 50%+ chance or measurable rain; outdoor only; CFB and NFL; strength from the worst factor", () => {
  const wx = (o) => g.gmKeyInsights(base({ gameInfo: { ...INFO, wind_mph: 5, temp_f: 60, precip_chance: null, precip_in: null, ...o } })).find((i) => i.kind === "weather");
  const a = wx({ wind_mph: 18 });
  assert.deepEqual([a.strength, a.title], [64, "Weather: wind 18 mph"]);        // 55 + (18 - 15) * 3
  assert.equal(a.body, "Forecast conditions at Bryant-Denny Stadium. Descriptive only, not a pick.");
  const b = wx({ wind_mph: 20, temp_f: 35, precip_chance: 60 });
  assert.deepEqual([b.strength, b.title], [70, "Weather: wind 20 mph, 35°F, 60% chance of rain"]);   // max(70, 65, 61)
  assert.equal(wx({ temp_f: 20 }).strength, 85);                                 // 55 + min(30, (40 - 20) * 2)
  assert.equal(wx({ temp_f: 39.4 }).title, "Weather: 39°F");
  assert.equal(wx({ precip_in: 0.05 }).title, "Weather: 0.05 in of rain");
  assert.equal(wx({ precip_in: 0.05 }).strength, 60);
  assert.match(wx({ wind_mph: 18, weather_kind: "observed" }).body, /^Observed conditions at/);
  for (const calm of [{}, { wind_mph: 14.9 }, { temp_f: 40 }, { precip_chance: 49 }, { precip_in: 0.004 }, { indoor: true, wind_mph: 30 }, { wind_mph: null, temp_f: null }]) assert.equal(wx(calm), undefined, JSON.stringify(calm));
  assert.equal(g.gmKeyInsights(base({ gameInfo: null })).find((i) => i.kind === "weather"), undefined);
  const nfl = g.gmKeyInsights({ sport: "nfl", r: R("nfl"), ctxGrades: [], ctxHist: [], ctxPower: [], gameInfo: { ...INFO, wind_mph: 22, venue_name: null } });
  assert.equal(nfl[0].kind, "weather"); assert.equal(nfl[0].body, "Forecast conditions. Descriptive only, not a pick.");
});

test("weather insight: hidden more than 3 days after kickoff, same rule as the hero line (R1)", () => {
  const hours = (h) => new Date(Date.now() + h * 36e5).toISOString();
  const wx = (ct) => g.gmKeyInsights(base({ r: PRED("cfb", { commence_time: ct }), gameInfo: { ...INFO, wind_mph: 22 } })).find((i) => i.kind === "weather");
  assert.ok(wx(hours(-3)), "a game that just kicked off still shows");
  assert.ok(wx(hours(-24 * 3 + 1)), "just inside 3 days");
  assert.equal(wx(hours(-24 * 3 - 1)), undefined, "just past 3 days");
  assert.equal(wx(hours(-24 * 6)), undefined);
  assert.ok(wx(null), "unknown kickoff: no time-based rule");
  assert.ok(g.gmKeyInsights({ sport: "nfl", r: PRED("nfl", { commence_time: hours(-2) }), ctxGrades: [], ctxHist: [], ctxPower: [], gameInfo: { ...INFO, wind_mph: 22 } }).some((i) => i.kind === "weather"));
});

test("Key Insights keeps the strongest four, strongest first, ties in generation order", () => {
  const grades = [{ side: "home", team: "333", overall: "A", pass: "A", run: "B", overall_pct: 90, units: null }];
  const power = [{ team: "333", rank: 4, rating: 20 }, { team: "61", rank: 40, rating: 2 }];
  const D = base({ ctxGrades: grades, ctxPower: power, ctxHist: [{ side: "home", team: "333", streaks: { su: "W5" } }, { side: "away", team: "61", streaks: {} }],
    cfbIns: [insRow("333", { def_havoc_rank: 8, turnover_margin_rank: 6, turnover_margin_per_game: 1.8 }), insRow("61", { off_havoc_allowed_rank: 118, turnover_margin_rank: 112, turnover_margin_per_game: -0.9 })],
    gameInfo: { ...INFO, wind_mph: 18 } });
  const out = g.gmKeyInsights(D);
  // grade 30 + 0.4 * 90 = 66 and power 30 + min(40, 36) = 66 tie: generation order (grade first). weather 64 and streak W5 54 are cut.
  assert.deepEqual(out.map((i) => [i.kind, i.strength]), [["havoc", 87], ["turnover", 85], ["grade", 66], ["power", 66]]);
  const again = g.gmKeyInsights(D);
  assert.deepEqual(again, out, "deterministic");
});

test("Key Insights card: the new rows render with their icons and escaped text; missing tables leave the old rows alone", async () => {
  const rows = { cfb_team_insights: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })],
    game_info: [{ ...INFO, venue_name: "<i>Field</i>", wind_mph: 18 }], team_history: HIST };
  const html = await page({ rows }).D.buildGamePage();
  const ins = html.slice(html.indexOf('id="gm-insights"'), html.indexOf('id="gm-cover"'));
  assert.ok(ins.includes("Alabama defense creates havoc against Georgia") && ins.includes("Weather: wind 18 mph"));
  assert.ok(ins.includes("&lt;i&gt;Field&lt;/i&gt;") && !ins.includes("<i>Field</i>"));
  assert.equal((ins.match(/class="ca-gm-ins"/g) || []).length, 2);
  assert.ok(ins.includes("ca-gm-ic-red") && ins.includes("ca-gm-ic-blue"));
  const none = await page({ missing: ["cfb_team_insights", "cfb_quarter_shares", "game_info"], rows: { team_history: HIST } }).D.buildGamePage();
  assert.ok(none.includes("ca-gm-hero") && !none.includes("creates havoc") && !none.includes("Weather:"), "tables missing: the page renders without the new rows");
});

test("cfb_team_insights / cfb_quarter_shares are read for both team codes (CFB only)", async () => {
  const cfb = page();
  await cfb.D.buildGamePage();
  assert.ok(cfb.requested.includes('cfb_team_insights?team=in.("61","333")&order=season.desc'), cfb.requested.join("\n"));
  assert.ok(cfb.requested.includes('cfb_quarter_shares?team=in.("61","333")'));
  const nfl = page({ sport: "nfl" });
  await nfl.D.buildGamePage();
  assert.ok(!nfl.requested.some((r) => r.startsWith("cfb_")), "NFL never reads the CFB tables");
});
