/* ─────────────────────────────────────────────────────────────────────────
   CONFIG — paste your Supabase project URL + anon (publishable) key here.
   Find them in Supabase → Project Settings → API. The anon key is safe to
   ship in a public site; the tables are read-only via RLS.
   ───────────────────────────────────────────────────────────────────────── */
const CONFIG = {
  SUPABASE_URL: "https://uydbzhzsscmrdwawnzlx.supabase.co",
  SUPABASE_ANON_KEY: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InV5ZGJ6aHpzc2NtcmR3YXduemx4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODcxNjE4MDYsImV4cCI6MjEwMjczNzgwNn0.lmQtx6Hz6oTtJOd3dmpxGTBKn5L8VPCA4jLPgF5E-CE",
};

const page = document.body.dataset.page;

/* ── Supabase REST helper ─────────────────────────────────────────────── */
async function sb(pathAndQuery) {
  const res = await fetch(`${CONFIG.SUPABASE_URL}/rest/v1/${pathAndQuery}`, {
    headers: {
      apikey: CONFIG.SUPABASE_ANON_KEY,
      Authorization: `Bearer ${CONFIG.SUPABASE_ANON_KEY}`,
    },
  });
  if (!res.ok) throw new Error(`Supabase ${res.status}: ${await res.text()}`);
  return res.json();
}

/* ── Formatters + mappers ─────────────────────────────────────────────── */
const GAME_MARKETS = new Set(["moneyline", "spread", "total"]);
const fmtOdds = (o) => (o > 0 ? `+${o}` : `${o}`);
const pct1 = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
const evp = (x) => (x == null ? "—" : `${x * 100 >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`);
const timeET = (iso) =>
  iso ? new Date(iso).toLocaleTimeString("en-US",
    { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) + " ET" : "";
const uStr = (u) => `${u >= 0 ? "+" : ""}${(+u).toFixed(1)}u`;
const pStr = (p) => `${+p >= 0 ? "+" : ""}${(+p).toFixed(1)}%`;

/* ── Settings store (localStorage "ca-settings"; see settings.html) ──────── */
// US books the site prices at, in display order. Caesars' odds feed key is
// williamhill_us; "caesars" is accepted as an alias everywhere.
const US_BOOKS = [["draftkings", "DraftKings"], ["fanduel", "FanDuel"], ["betmgm", "BetMGM"], ["williamhill_us", "Caesars"],
  ["fanatics", "Fanatics"], ["espnbet", "ESPN BET"], ["hardrockbet", "Hard Rock Bet"], ["thescore", "theScore Bet"],
  ["bet365", "Bet365"], ["ballybet", "Bally Bet"]];
const KELLY_FRACTIONS = [[1, "Full"], [0.5, "Half"], [0.25, "Quarter"], [0.125, "Eighth"]];
const SETTINGS_DEFAULTS = Object.freeze({ books: Object.freeze(US_BOOKS.map((b) => b[0])), unit: 10, bankroll: 1000, kelly: 0.25, minEv: 0 });
const SETTINGS_MAX = Object.freeze({ unit: 100000, bankroll: 100000000 });  // spec upper bounds
const SETTINGS_KEY = "ca-settings";
let settingsMem = null;  // in-memory copy when storage writes fail, so the page still works this visit
const bookKey = (b) => { const k = String(b || "").toLowerCase(); return k === "caesars" ? "williamhill_us" : k; };
const setNum = (v) => (typeof v === "number" || (typeof v === "string" && v.trim() !== "") ? Number(v) : NaN);
// Per-field validation: a bad field falls back to `base`; books keep only known
// keys (canonical order), and an empty selection means all books.
function normSettings(raw, base = SETTINGS_DEFAULTS) {
  const r = raw && typeof raw === "object" ? raw : {};
  const pos = (v, d, max) => { const n = setNum(v); return Number.isFinite(n) && n > 0 && n <= max ? n : d; };
  let books = Array.isArray(r.books) ? r.books.map(bookKey) : null;
  books = books ? US_BOOKS.map((b) => b[0]).filter((k) => books.includes(k)) : [...base.books];
  if (!books.length) books = [...SETTINGS_DEFAULTS.books];
  const kelly = setNum(r.kelly), minEv = setNum(r.minEv);
  return {
    books,
    unit: pos(r.unit, base.unit, SETTINGS_MAX.unit),
    bankroll: pos(r.bankroll, base.bankroll, SETTINGS_MAX.bankroll),
    kelly: KELLY_FRACTIONS.some((f) => f[0] === kelly) ? kelly : base.kelly,
    minEv: Number.isFinite(minEv) && minEv >= 0 && minEv <= 20 ? minEv : base.minEv,
  };
}
function getSettings() {
  if (settingsMem) return normSettings(settingsMem);
  let raw = null;
  try { raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "null"); } catch {}
  return normSettings(raw);
}
function saveSettings(partial) {
  const cur = getSettings();
  const s = normSettings({ ...cur, ...(partial || {}) }, cur);
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); settingsMem = null; } catch { settingsMem = s; }
  return s;
}
const settingsStorageOk = () => {
  try { localStorage.setItem("ca-settings-probe", "1"); localStorage.removeItem("ca-settings-probe"); return true; } catch { return false; }
};
const bookSelected = (book, s = getSettings()) => s.books.includes(bookKey(book));
const unitScale = (s = getSettings()) => s.unit / 10;
// Kelly stake for a bet with win probability `prob` at American odds:
// f* = (p·dec − 1) / (dec − 1), scaled by the chosen fraction and bankroll.
// null when there's no edge (f ≤ 0) or the inputs are unusable.
function kellyStake(prob, american, s = getSettings()) {
  const p = +prob, am = +american;
  if (prob == null || american == null || !(p > 0 && p < 1) || !Number.isFinite(am) || (am > -100 && am < 100)) return null;
  const dec = am > 0 ? 1 + am / 100 : 1 + 100 / -am;
  const f = (p * dec - 1) / (dec - 1);
  if (!(f > 0)) return null;
  const dollars = f * s.kelly * s.bankroll;
  return { dollars, units: dollars / s.unit };
}
// Round first, then format: $10+ as whole dollars with separators ($12,500),
// under $10 to the cent; units to one decimal with separators (1,250.0u).
const fmtKelly = (k) => {
  if (!k) return "—";
  const cents = Math.round(k.dollars * 100) / 100;
  const d = cents >= 10 ? Math.round(cents).toLocaleString("en-US") : cents.toFixed(2);
  const u = (Math.round(k.units * 10) / 10).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  return `$${d} (${u}u)`;
};

/* ── Team logos (ESPN CDN, by abbreviation) ───────────────────────────── */
const TEAM_ABBR = {
  "Arizona Diamondbacks": "ari", "Atlanta Braves": "atl", "Baltimore Orioles": "bal",
  "Boston Red Sox": "bos", "Chicago Cubs": "chc", "Chicago White Sox": "chw",
  "Cincinnati Reds": "cin", "Cleveland Guardians": "cle", "Colorado Rockies": "col",
  "Detroit Tigers": "det", "Houston Astros": "hou", "Kansas City Royals": "kc",
  "Los Angeles Angels": "laa", "Los Angeles Dodgers": "lad", "Miami Marlins": "mia",
  "Milwaukee Brewers": "mil", "Minnesota Twins": "min", "New York Mets": "nym",
  "New York Yankees": "nyy", "Oakland Athletics": "oak", "Athletics": "oak",
  "Philadelphia Phillies": "phi", "Pittsburgh Pirates": "pit", "San Diego Padres": "sd",
  "San Francisco Giants": "sf", "Seattle Mariners": "sea", "St. Louis Cardinals": "stl",
  "Tampa Bay Rays": "tb", "Texas Rangers": "tex", "Toronto Blue Jays": "tor",
  "Washington Nationals": "wsh",
};

// NFL: matchup/pick_label carry full ESPN displayNames (e.g. "Kansas City
// Chiefs"), same shape as MLB's TEAM_ABBR keys — but the raw `team` field on
// NFL prop rows carries the nflverse-style franchise code (KC, SF, WAS, LA,
// ...) instead of a full name, so it needs its own code -> ESPN-slug map.
// Added alongside MLB, not touching it — dispatch is by the row's `sport`.
const NFL_TEAM_ABBR = {
  "Arizona Cardinals": "ari", "Atlanta Falcons": "atl", "Baltimore Ravens": "bal",
  "Buffalo Bills": "buf", "Carolina Panthers": "car", "Chicago Bears": "chi",
  "Cincinnati Bengals": "cin", "Cleveland Browns": "cle", "Dallas Cowboys": "dal",
  "Denver Broncos": "den", "Detroit Lions": "det", "Green Bay Packers": "gb",
  "Houston Texans": "hou", "Indianapolis Colts": "ind", "Jacksonville Jaguars": "jax",
  "Kansas City Chiefs": "kc", "Las Vegas Raiders": "lv", "Los Angeles Chargers": "lac",
  "Los Angeles Rams": "lar", "Miami Dolphins": "mia", "Minnesota Vikings": "min",
  "New England Patriots": "ne", "New Orleans Saints": "no", "New York Giants": "nyg",
  "New York Jets": "nyj", "Philadelphia Eagles": "phi", "Pittsburgh Steelers": "pit",
  "San Francisco 49ers": "sf", "Seattle Seahawks": "sea", "Tampa Bay Buccaneers": "tb",
  "Tennessee Titans": "ten", "Washington Commanders": "wsh",
};
// nflverse/normalize_team() franchise code (as stored on prop rows' `team`
// column) -> ESPN CDN slug; differs from the code for a handful of teams.
const NFL_CODE_TO_ESPN = {
  ARI: "ari", ATL: "atl", BAL: "bal", BUF: "buf", CAR: "car", CHI: "chi",
  CIN: "cin", CLE: "cle", DAL: "dal", DEN: "den", DET: "det", GB: "gb",
  HOU: "hou", IND: "ind", JAX: "jax", KC: "kc", LA: "lar", LAC: "lac",
  LV: "lv", MIA: "mia", MIN: "min", NE: "ne", NO: "no", NYG: "nyg",
  NYJ: "nyj", PHI: "phi", PIT: "pit", SEA: "sea", SF: "sf", TB: "tb",
  TEN: "ten", WAS: "wsh",
};

// CFB: ESPN displayName -> ESPN team id; logos at ncaa/500/<id>.png (from assets/cfb/fbs_teams.json)
const CFB_TEAM_ID = {
  "Air Force Falcons": "2005", "Akron Zips": "2006", "Alabama Crimson Tide": "333",
  "App State Mountaineers": "2026", "Arizona State Sun Devils": "9", "Arizona Wildcats": "12",
  "Arkansas Razorbacks": "8", "Arkansas State Red Wolves": "2032", "Army Black Knights": "349",
  "Auburn Tigers": "2", "BYU Cougars": "252", "Ball State Cardinals": "2050",
  "Baylor Bears": "239", "Boise State Broncos": "68", "Boston College Eagles": "103",
  "Bowling Green Falcons": "189", "Buffalo Bulls": "2084", "California Golden Bears": "25",
  "Central Michigan Chippewas": "2117", "Charlotte 49ers": "2429",
  "Cincinnati Bearcats": "2132", "Clemson Tigers": "228",
  "Coastal Carolina Chanticleers": "324", "Colorado Buffaloes": "38",
  "Colorado State Rams": "36", "Delaware Blue Hens": "48", "Duke Blue Devils": "150",
  "East Carolina Pirates": "151", "Eastern Michigan Eagles": "2199",
  "Florida Atlantic Owls": "2226", "Florida Gators": "57",
  "Florida International Panthers": "2229", "Florida State Seminoles": "52",
  "Fresno State Bulldogs": "278", "Georgia Bulldogs": "61", "Georgia Southern Eagles": "290",
  "Georgia State Panthers": "2247", "Georgia Tech Yellow Jackets": "59",
  "Hawai'i Rainbow Warriors": "62", "Houston Cougars": "248",
  "Illinois Fighting Illini": "356", "Indiana Hoosiers": "84", "Iowa Hawkeyes": "2294",
  "Iowa State Cyclones": "66", "Jacksonville State Gamecocks": "55",
  "James Madison Dukes": "256", "Kansas Jayhawks": "2305", "Kansas State Wildcats": "2306",
  "Kennesaw State Owls": "338", "Kent State Golden Flashes": "2309", "Kentucky Wildcats": "96",
  "LSU Tigers": "99", "Liberty Flames": "2335", "Louisiana Ragin' Cajuns": "309",
  "Louisiana Tech Bulldogs": "2348", "Louisville Cardinals": "97",
  "Marshall Thundering Herd": "276", "Maryland Terrapins": "120",
  "Massachusetts Minutemen": "113", "Memphis Tigers": "235", "Miami (OH) RedHawks": "193",
  "Miami Hurricanes": "2390", "Michigan State Spartans": "127", "Michigan Wolverines": "130",
  "Middle Tennessee Blue Raiders": "2393", "Minnesota Golden Gophers": "135",
  "Mississippi State Bulldogs": "344", "Missouri State Bears": "2623",
  "Missouri Tigers": "142", "NC State Wolfpack": "152", "Navy Midshipmen": "2426",
  "Nebraska Cornhuskers": "158", "Nevada Wolf Pack": "2440", "New Mexico Lobos": "167",
  "New Mexico State Aggies": "166", "North Carolina Tar Heels": "153",
  "North Texas Mean Green": "249", "Northern Illinois Huskies": "2459",
  "Northwestern Wildcats": "77", "Notre Dame Fighting Irish": "87", "Ohio Bobcats": "195",
  "Ohio State Buckeyes": "194", "Oklahoma Sooners": "201", "Oklahoma State Cowboys": "197",
  "Old Dominion Monarchs": "295", "Ole Miss Rebels": "145", "Oregon Ducks": "2483",
  "Oregon State Beavers": "204", "Penn State Nittany Lions": "213",
  "Pittsburgh Panthers": "221", "Purdue Boilermakers": "2509", "Rice Owls": "242",
  "Rutgers Scarlet Knights": "164", "SMU Mustangs": "2567", "Sam Houston Bearkats": "2534",
  "San Diego State Aztecs": "21", "San José State Spartans": "23",
  "South Alabama Jaguars": "6", "South Carolina Gamecocks": "2579",
  "South Florida Bulls": "58", "Southern Miss Golden Eagles": "2572",
  "Stanford Cardinal": "24", "Syracuse Orange": "183", "TCU Horned Frogs": "2628",
  "Temple Owls": "218", "Tennessee Volunteers": "2633", "Texas A&M Aggies": "245",
  "Texas Longhorns": "251", "Texas State Bobcats": "326", "Texas Tech Red Raiders": "2641",
  "Toledo Rockets": "2649", "Troy Trojans": "2653", "Tulane Green Wave": "2655",
  "Tulsa Golden Hurricane": "202", "UAB Blazers": "5", "UCF Knights": "2116",
  "UCLA Bruins": "26", "UConn Huskies": "41", "UL Monroe Warhawks": "2433",
  "UNLV Rebels": "2439", "USC Trojans": "30", "UTEP Miners": "2638",
  "UTSA Roadrunners": "2636", "Utah State Aggies": "328", "Utah Utes": "254",
  "Vanderbilt Commodores": "238", "Virginia Cavaliers": "258", "Virginia Tech Hokies": "259",
  "Wake Forest Demon Deacons": "154", "Washington Huskies": "264",
  "Washington State Cougars": "265", "West Virginia Mountaineers": "277",
  "Western Kentucky Hilltoppers": "98", "Western Michigan Broncos": "2711",
  "Wisconsin Badgers": "275", "Wyoming Cowboys": "2751",
};

// CFB: ESPN displayName -> ESPN team abbreviation (LIB, OSU, ...), from the
// college-football teams API. Keyed the same as CFB_TEAM_ID.
const CFB_TEAM_ABBR = {
  "Air Force Falcons": "AF", "Akron Zips": "AKR", "Alabama Crimson Tide": "ALA", "App State Mountaineers": "APP",
  "Arizona State Sun Devils": "ASU", "Arizona Wildcats": "ARIZ", "Arkansas Razorbacks": "ARK", "Arkansas State Red Wolves": "ARST",
  "Army Black Knights": "ARMY", "Auburn Tigers": "AUB", "BYU Cougars": "BYU", "Ball State Cardinals": "BALL",
  "Baylor Bears": "BAY", "Boise State Broncos": "BOIS", "Boston College Eagles": "BC", "Bowling Green Falcons": "BGSU",
  "Buffalo Bulls": "BUF", "California Golden Bears": "CAL", "Central Michigan Chippewas": "CMU", "Charlotte 49ers": "CLT",
  "Cincinnati Bearcats": "CIN", "Clemson Tigers": "CLEM", "Coastal Carolina Chanticleers": "CCU", "Colorado Buffaloes": "COLO",
  "Colorado State Rams": "CSU", "Delaware Blue Hens": "DEL", "Duke Blue Devils": "DUKE", "East Carolina Pirates": "ECU",
  "Eastern Michigan Eagles": "EMU", "Florida Atlantic Owls": "FAU", "Florida Gators": "FLA", "Florida International Panthers": "FIU",
  "Florida State Seminoles": "FSU", "Fresno State Bulldogs": "FRES", "Georgia Bulldogs": "UGA", "Georgia Southern Eagles": "GASO",
  "Georgia State Panthers": "GAST", "Georgia Tech Yellow Jackets": "GT", "Hawai'i Rainbow Warriors": "HAW", "Houston Cougars": "HOU",
  "Illinois Fighting Illini": "ILL", "Indiana Hoosiers": "IU", "Iowa Hawkeyes": "IOWA", "Iowa State Cyclones": "ISU",
  "Jacksonville State Gamecocks": "JXST", "James Madison Dukes": "JMU", "Kansas Jayhawks": "KU", "Kansas State Wildcats": "KSU",
  "Kennesaw State Owls": "KENN", "Kent State Golden Flashes": "KENT", "Kentucky Wildcats": "UK", "LSU Tigers": "LSU",
  "Liberty Flames": "LIB", "Louisiana Ragin' Cajuns": "UL", "Louisiana Tech Bulldogs": "LT", "Louisville Cardinals": "LOU",
  "Marshall Thundering Herd": "MRSH", "Maryland Terrapins": "MD", "Massachusetts Minutemen": "MASS", "Memphis Tigers": "MEM",
  "Miami (OH) RedHawks": "M-OH", "Miami Hurricanes": "MIA", "Michigan State Spartans": "MSU", "Michigan Wolverines": "MICH",
  "Middle Tennessee Blue Raiders": "MTSU", "Minnesota Golden Gophers": "MINN", "Mississippi State Bulldogs": "MSST", "Missouri State Bears": "MOST",
  "Missouri Tigers": "MIZ", "NC State Wolfpack": "NCSU", "Navy Midshipmen": "NAVY", "Nebraska Cornhuskers": "NEB",
  "Nevada Wolf Pack": "NEV", "New Mexico Lobos": "UNM", "New Mexico State Aggies": "NMSU", "North Carolina Tar Heels": "UNC",
  "North Texas Mean Green": "UNT", "Northern Illinois Huskies": "NIU", "Northwestern Wildcats": "NU", "Notre Dame Fighting Irish": "ND",
  "Ohio Bobcats": "OHIO", "Ohio State Buckeyes": "OSU", "Oklahoma Sooners": "OU", "Oklahoma State Cowboys": "OKST",
  "Old Dominion Monarchs": "ODU", "Ole Miss Rebels": "MISS", "Oregon Ducks": "ORE", "Oregon State Beavers": "ORST",
  "Penn State Nittany Lions": "PSU", "Pittsburgh Panthers": "PITT", "Purdue Boilermakers": "PUR", "Rice Owls": "RICE",
  "Rutgers Scarlet Knights": "RUTG", "SMU Mustangs": "SMU", "Sam Houston Bearkats": "SHSU", "San Diego State Aztecs": "SDSU",
  "San José State Spartans": "SJSU", "South Alabama Jaguars": "USA", "South Carolina Gamecocks": "SC", "South Florida Bulls": "USF",
  "Southern Miss Golden Eagles": "USM", "Stanford Cardinal": "STAN", "Syracuse Orange": "SYR", "TCU Horned Frogs": "TCU",
  "Temple Owls": "TEM", "Tennessee Volunteers": "TENN", "Texas A&M Aggies": "TA&M", "Texas Longhorns": "TEX",
  "Texas State Bobcats": "TXST", "Texas Tech Red Raiders": "TTU", "Toledo Rockets": "TOL", "Troy Trojans": "TROY",
  "Tulane Green Wave": "TULN", "Tulsa Golden Hurricane": "TLSA", "UAB Blazers": "UAB", "UCF Knights": "UCF",
  "UCLA Bruins": "UCLA", "UConn Huskies": "CONN", "UL Monroe Warhawks": "ULM", "UNLV Rebels": "UNLV",
  "USC Trojans": "USC", "UTEP Miners": "UTEP", "UTSA Roadrunners": "UTSA", "Utah State Aggies": "USU",
  "Utah Utes": "UTAH", "Vanderbilt Commodores": "VAN", "Virginia Cavaliers": "UVA", "Virginia Tech Hokies": "VT",
  "Wake Forest Demon Deacons": "WAKE", "Washington Huskies": "WASH", "Washington State Cougars": "WSU", "West Virginia Mountaineers": "WVU",
  "Western Kentucky Hilltoppers": "WKU", "Western Michigan Broncos": "WMU", "Wisconsin Badgers": "WIS", "Wyoming Cowboys": "WYO",
};

// Team colors {p: primary, a: alternate} from ESPN's team API. The game detail page
// colors each team's side with its primary; gameTeamColors() makes it readable
// on the dark background. Unmapped teams fall back to the default blue.
const NFL_TEAM_COLORS = {
  "Arizona Cardinals": { p: "#A40227", a: "#FFFFFF" }, "Atlanta Falcons": { p: "#A71930", a: "#000000" },
  "Baltimore Ravens": { p: "#29126F", a: "#000000" }, "Buffalo Bills": { p: "#00338D", a: "#D50A0A" },
  "Carolina Panthers": { p: "#0085CA", a: "#000000" }, "Chicago Bears": { p: "#0B1C3A", a: "#E64100" },
  "Cincinnati Bengals": { p: "#FB4F14", a: "#000000" }, "Cleveland Browns": { p: "#472A08", a: "#FF3C00" },
  "Dallas Cowboys": { p: "#002A5C", a: "#B0B7BC" }, "Denver Broncos": { p: "#0A2343", a: "#FC4C02" },
  "Detroit Lions": { p: "#0076B6", a: "#BBBBBB" }, "Green Bay Packers": { p: "#204E32", a: "#FFB612" },
  "Houston Texans": { p: "#021018", a: "#EB0028" }, "Indianapolis Colts": { p: "#003B75", a: "#FFFFFF" },
  "Jacksonville Jaguars": { p: "#007487", a: "#D7A22A" }, "Kansas City Chiefs": { p: "#E31837", a: "#FFB612" },
  "Las Vegas Raiders": { p: "#000000", a: "#A5ACAF" }, "Los Angeles Chargers": { p: "#0080C6", a: "#FFC20E" },
  "Los Angeles Rams": { p: "#003594", a: "#FFD100" }, "Miami Dolphins": { p: "#008E97", a: "#FC4C02" },
  "Minnesota Vikings": { p: "#4F2683", a: "#FFC62F" }, "New England Patriots": { p: "#002A5C", a: "#C60C30" },
  "New Orleans Saints": { p: "#D3BC8D", a: "#000000" }, "New York Giants": { p: "#003C7F", a: "#C9243F" },
  "New York Jets": { p: "#115740", a: "#FFFFFF" }, "Philadelphia Eagles": { p: "#06424D", a: "#000000" },
  "Pittsburgh Steelers": { p: "#000000", a: "#FFB612" }, "San Francisco 49ers": { p: "#AA0000", a: "#B3995D" },
  "Seattle Seahawks": { p: "#002A5C", a: "#69BE28" }, "Tampa Bay Buccaneers": { p: "#BD1C36", a: "#3E3A35" },
  "Tennessee Titans": { p: "#4495D2", a: "#001532" }, "Washington Commanders": { p: "#5A1414", a: "#FFB612" },
};
const CFB_TEAM_COLORS = {
  "Air Force Falcons": { p: "#003594", a: "#FFFFFF" }, "Akron Zips": { p: "#041E42", a: "#C5B783" },
  "Alabama Crimson Tide": { p: "#9E1B32", a: "#FFFFFF" }, "App State Mountaineers": { p: "#000000", a: "#FFCD00" },
  "Arizona State Sun Devils": { p: "#FFC627", a: "#8C1D40" }, "Arizona Wildcats": { p: "#CC0033", a: "#003366" },
  "Arkansas Razorbacks": { p: "#A32136", a: "#FFFFFF" }, "Arkansas State Red Wolves": { p: "#CC092F", a: "#000000" },
  "Army Black Knights": { p: "#000000", a: "#D3BC8D" }, "Auburn Tigers": { p: "#002B5C", a: "#F26522" },
  "BYU Cougars": { p: "#0047BA", a: "#002E5D" }, "Ball State Cardinals": { p: "#BA0C2F", a: "#FFFFFF" },
  "Baylor Bears": { p: "#154734", a: "#FFB81C" }, "Boise State Broncos": { p: "#0033A0", a: "#D64309" },
  "Boston College Eagles": { p: "#8C2232", a: "#DBCCA6" }, "Bowling Green Falcons": { p: "#FD5000", a: "#4F2C1D" },
  "Buffalo Bulls": { p: "#005BBB", a: "#FFFFFF" }, "California Golden Bears": { p: "#041E42", a: "#FFC72C" },
  "Central Michigan Chippewas": { p: "#4C0027", a: "#FBAB18" }, "Charlotte 49ers": { p: "#005035", a: "#A49665" },
  "Cincinnati Bearcats": { p: "#000000", a: "#E00122" }, "Clemson Tigers": { p: "#F56600", a: "#FFFFFF" },
  "Coastal Carolina Chanticleers": { p: "#006F71", a: "#A27752" }, "Colorado Buffaloes": { p: "#CFB87C", a: "#000000" },
  "Colorado State Rams": { p: "#004C23", a: "#C8C372" }, "Delaware Blue Hens": { p: "#00539F", a: "#FFD200" },
  "Duke Blue Devils": { p: "#00539B", a: "#FFFFFF" }, "East Carolina Pirates": { p: "#582C83", a: "#FFC72C" },
  "Eastern Michigan Eagles": { p: "#006938", a: "#FFFFFF" }, "Florida Atlantic Owls": { p: "#003366", a: "#CC0000" },
  "Florida Gators": { p: "#0021A5", a: "#FA4616" }, "Florida International Panthers": { p: "#091F3F", a: "#C3993F" },
  "Florida State Seminoles": { p: "#782F40", a: "#CEB888" }, "Fresno State Bulldogs": { p: "#B1102B", a: "#13284C" },
  "Georgia Bulldogs": { p: "#BA0C2F", a: "#2C2A29" }, "Georgia Southern Eagles": { p: "#041E42", a: "#A3AAAE" },
  "Georgia State Panthers": { p: "#0039A6", a: "#FFFFFF" }, "Georgia Tech Yellow Jackets": { p: "#B3A369", a: "#FFFFFF" },
  "Hawai'i Rainbow Warriors": { p: "#005737", a: "#000000" }, "Houston Cougars": { p: "#C8102E", a: "#FFFFFF" },
  "Illinois Fighting Illini": { p: "#FF5F05", a: "#13294B" }, "Indiana Hoosiers": { p: "#970310", a: "#FFFFFF" },
  "Iowa Hawkeyes": { p: "#231F20", a: "#FCD116" }, "Iowa State Cyclones": { p: "#AE192D", a: "#FFC72A" },
  "Jacksonville State Gamecocks": { p: "#CC0000", a: "#000000" }, "James Madison Dukes": { p: "#450084", a: "#CBB677" },
  "Kansas Jayhawks": { p: "#0051BA", a: "#E8000D" }, "Kansas State Wildcats": { p: "#330A57", a: "#E2E3E4" },
  "Kennesaw State Owls": { p: "#FDBB30", a: "#0B1315" }, "Kent State Golden Flashes": { p: "#002664", a: "#EAAB00" },
  "Kentucky Wildcats": { p: "#0033A0", a: "#FFFFFF" }, "LSU Tigers": { p: "#461D76", a: "#FDD023" },
  "Liberty Flames": { p: "#0A254E", a: "#B72025" }, "Louisiana Ragin' Cajuns": { p: "#CE181E", a: "#000000" },
  "Louisiana Tech Bulldogs": { p: "#003087", a: "#CB333B" }, "Louisville Cardinals": { p: "#C9001F", a: "#FFFFFF" },
  "Marshall Thundering Herd": { p: "#00B140", a: "#000000" }, "Maryland Terrapins": { p: "#CE1126", a: "#FFFFFF" },
  "Massachusetts Minutemen": { p: "#881C1C", a: "#FFFFFF" }, "Memphis Tigers": { p: "#004991", a: "#8E908F" },
  "Miami Hurricanes": { p: "#F47423", a: "#035131" }, "Michigan State Spartans": { p: "#173F35", a: "#FFFFFF" },
  "Michigan Wolverines": { p: "#00274C", a: "#FFCB05" }, "Middle Tennessee Blue Raiders": { p: "#036EB7", a: "#FFFFFF" },
  "Minnesota Golden Gophers": { p: "#5E0A2F", a: "#FAB41C" }, "Mississippi State Bulldogs": { p: "#5D1725", a: "#C1C6C8" },
  "Missouri State Bears": { p: "#5E0009", a: "#FFFFFF" }, "Missouri Tigers": { p: "#F1B82D", a: "#000000" },
  "NC State Wolfpack": { p: "#CC0000", a: "#FFFFFF" }, "Navy Midshipmen": { p: "#00225B", a: "#B5A67C" },
  "Nebraska Cornhuskers": { p: "#E31937", a: "#FFFFFF" }, "Nevada Wolf Pack": { p: "#041E42", a: "#8A8D8F" },
  "New Mexico Lobos": { p: "#BA0C2F", a: "#A7A8AA" }, "New Mexico State Aggies": { p: "#7E141B", a: "#231F20" },
  "North Carolina Tar Heels": { p: "#7BAFD4", a: "#13294B" }, "North Texas Mean Green": { p: "#068F33", a: "#FFFFFF" },
  "Northern Illinois Huskies": { p: "#C8102E", a: "#000000" }, "Northwestern Wildcats": { p: "#492F92", a: "#FFFFFF" },
  "Notre Dame Fighting Irish": { p: "#062340", a: "#C99700" }, "Ohio Bobcats": { p: "#154734", a: "#FFFFFF" },
  "Ohio State Buckeyes": { p: "#BA0C2F", a: "#A8ADB4" }, "Oklahoma Sooners": { p: "#990000", a: "#FFFFFF" },
  "Oklahoma State Cowboys": { p: "#FE5C00", a: "#000000" }, "Old Dominion Monarchs": { p: "#003768", a: "#A1D2F1" },
  "Ole Miss Rebels": { p: "#13294B", a: "#CF142B" }, "Oregon Ducks": { p: "#00934B", a: "#FFF41B" },
  "Oregon State Beavers": { p: "#DC4405", a: "#000000" }, "Penn State Nittany Lions": { p: "#061440", a: "#FFFFFF" },
  "Pittsburgh Panthers": { p: "#003594", a: "#FFB81C" }, "Purdue Boilermakers": { p: "#CEB888", a: "#000000" },
  "Rice Owls": { p: "#00205B", a: "#C1C6C8" }, "Rutgers Scarlet Knights": { p: "#CE0E2D", a: "#FFFFFF" },
  "SMU Mustangs": { p: "#A80000", a: "#0033A1" }, "Sam Houston Bearkats": { p: "#F56423", a: "#FFFFFF" },
  "San Diego State Aztecs": { p: "#A6192E", a: "#000000" }, "San José State Spartans": { p: "#0038A8", a: "#FFB81A" },
  "South Alabama Jaguars": { p: "#00205B", a: "#BF0D3E" }, "South Carolina Gamecocks": { p: "#73000A", a: "#000000" },
  "South Florida Bulls": { p: "#006747", a: "#CFC493" }, "Southern Miss Golden Eagles": { p: "#FFC72C", a: "#231F20" },
  "Stanford Cardinal": { p: "#8C1515", a: "#FFFFFF" }, "Syracuse Orange": { p: "#000E54", a: "#FF431B" },
  "TCU Horned Frogs": { p: "#4D1979", a: "#FFFFFF" }, "Temple Owls": { p: "#A41E35", a: "#FFFFFF" },
  "Tennessee Volunteers": { p: "#FF8200", a: "#FFFFFF" }, "Texas A&M Aggies": { p: "#500000", a: "#FFFFFF" },
  "Texas Longhorns": { p: "#AF5C37", a: "#FFFFFF" }, "Texas State Bobcats": { p: "#501214", a: "#6A5638" },
  "Texas Tech Red Raiders": { p: "#DA291C", a: "#000000" }, "Toledo Rockets": { p: "#0B2240", a: "#FFCD00" },
  "Troy Trojans": { p: "#862633", a: "#B1B1B1" }, "Tulane Green Wave": { p: "#006747", a: "#418FDE" },
  "Tulsa Golden Hurricane": { p: "#003595", a: "#D0B787" }, "UAB Blazers": { p: "#1A5632", a: "#FDB913" },
  "UCF Knights": { p: "#000000", a: "#B4A169" }, "UCLA Bruins": { p: "#2774AE", a: "#F2A900" },
  "UConn Huskies": { p: "#0C2340", a: "#A2AAAD" }, "UL Monroe Warhawks": { p: "#840029", a: "#FDB913" },
  "UNLV Rebels": { p: "#CF0A2C", a: "#CAC8C8" }, "USC Trojans": { p: "#9D2235", a: "#FFC72C" },
  "UTEP Miners": { p: "#FF8200", a: "#041E42" }, "UTSA Roadrunners": { p: "#0C2340", a: "#F15A22" },
  "Utah State Aggies": { p: "#0F2439", a: "#FFFFFF" }, "Utah Utes": { p: "#BE0000", a: "#FFFFFF" },
  "Vanderbilt Commodores": { p: "#000000", a: "#CFAE70" }, "Virginia Cavaliers": { p: "#232D4B", a: "#F84C1E" },
  "Virginia Tech Hokies": { p: "#6A2C3E", a: "#CF4520" }, "Wake Forest Demon Deacons": { p: "#CEB888", a: "#2C2A29" },
  "Washington Huskies": { p: "#33006F", a: "#E8D3A2" }, "Washington State Cougars": { p: "#A60F2D", a: "#4D4D4D" },
  "West Virginia Mountaineers": { p: "#EAAA00", a: "#002855" }, "Western Kentucky Hilltoppers": { p: "#E13A3E", a: "#FFFFFF" },
  "Western Michigan Broncos": { p: "#532E1F", a: "#F1C500" }, "Wisconsin Badgers": { p: "#A00000", a: "#FFFFFF" },
  "Wyoming Cowboys": { p: "#492F24", a: "#FFC425" },
};
const DEFAULT_ACCENT = "#2563EB";   // = --blue (hex: team-color math needs it)
function relLum(hex) {
  const n = String(hex || "").replace("#", "");
  if (n.length < 6) return 0.5;
  const r = parseInt(n.slice(0, 2), 16), g = parseInt(n.slice(2, 4), 16), b = parseInt(n.slice(4, 6), 16);
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}
function hexToHsl(hex) {
  const n = String(hex || "").replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(n.slice(i, i + 2), 16) / 255);
  const max = Math.max(r, g, b), min = Math.min(r, g, b), l = (max + min) / 2, d = max - min;
  if (!d) return [0, 0, l];
  const s = d / (1 - Math.abs(2 * l - 1));
  const h = max === r ? ((g - b) / d) % 6 : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [(h * 60 + 360) % 360, s, l];
}
function hslToHex(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = l - c / 2;
  const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
  return "#" + [r, g, b].map((v) => Math.round((v + m) * 255).toString(16).padStart(2, "0")).join("");
}
// One team color made readable on the light page: a too-light but COLORED hex
// (Steelers gold, Cardinals yellow) is darkened in its own hue until it holds
// contrast on white; dark colored hexes (navy, maroon) read fine as-is. A
// colorless one (black/gray, e.g. Raiders) or near-white returns null so the
// caller falls back to the team's alternate.
function readableTeamColor(hex) {
  if (!hex || String(hex).replace("#", "").length < 6) return null;
  const lum = relLum(hex);
  if (lum > 0.97) return null;
  const [h, s, l] = hexToHsl(hex);
  const colorless = s < 0.2;
  if (colorless && (lum < 0.25 || lum > 0.8)) return null;
  if (lum <= 0.5) return hex;
  // Keep a gray gray: only colored hexes get a saturation floor while darkening.
  const sat = colorless ? s : Math.max(s, 0.55);
  let out = hex;
  for (let L = l; L > 0.25 && relLum(out) > 0.5; L -= 0.03) out = hslToHex(h, sat, L);
  return out;
}
function teamAccent(name, sport, useAlt = false) {
  const m = sport === "nfl" ? NFL_TEAM_COLORS[name] : sport === "cfb" ? CFB_TEAM_COLORS[name] : null;
  if (!m) return DEFAULT_ACCENT;
  const [first, second] = useAlt ? [m.a, m.p] : [m.p, m.a];
  return readableTeamColor(first) || readableTeamColor(second) || DEFAULT_ACCENT;
}
// Both teams' game-page colors: each team's PRIMARY (darkened if too light to
// read on the light bg; the alternate only when the primary is black/gray/white).
// When the two land too close to tell apart (e.g. two royal blues), the AWAY
// team switches to its alternate.
function gameTeamColors(awayName, homeName, sport) {
  const home = teamAccent(homeName, sport);
  let away = teamAccent(awayName, sport);
  const rgb = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  const dist = (a, b) => { const [x, y] = [rgb(a), rgb(b)]; return Math.hypot(x[0] - y[0], x[1] - y[1], x[2] - y[2]); };
  if (dist(away, home) < 90) {
    const alt = teamAccent(awayName, sport, true);
    if (dist(alt, home) > dist(away, home)) away = alt;
  }
  return { away, home };
}

function logoImg(name, sport) {
  let a = TEAM_ABBR[name];
  let league = "mlb";
  if (!a && sport === "nfl") {
    a = NFL_TEAM_ABBR[name] || NFL_CODE_TO_ESPN[String(name || "").toUpperCase()];
    league = "nfl";
  } else if (!a && sport === "cfb") {
    a = CFB_TEAM_ID[name];   // ESPN team id; CFB logos live under the ncaa/ path
    league = "ncaa";
  }
  return a ? `<img class="tlogo" src="https://a.espncdn.com/i/teamlogos/${league}/500/${a}.png" alt="" loading="lazy" onerror="this.style.display='none'">` : "";
}
function logoPair(matchup, sport) {
  const [away, home] = String(matchup).split(" @ ");
  return logoImg(away, sport) + logoImg(home, sport);
}
function pickLogo(label, sport) {  // the team the pick is on (game lines); "" for totals
  const names = sport === "nfl" ? NFL_TEAM_ABBR : sport === "cfb" ? CFB_TEAM_ID : TEAM_ABBR;
  for (const nm in names) if (String(label).startsWith(nm)) return logoImg(nm, sport);
  return "";
}

// CFB mascots that are TWO words, so stripping them leaves the school name
// ("Alabama Crimson Tide" -> "Alabama"); everything else strips one word.
const CFB_2W_MASCOTS = new Set([
  "crimson tide", "golden eagles", "fighting irish", "green wave", "red wolves",
  "blue devils", "tar heels", "demon deacons", "mean green", "scarlet knights",
  "nittany lions", "fighting illini", "golden gophers", "golden bears", "sun devils",
  "red raiders", "yellow jackets", "horned frogs", "black knights", "mountain hawks",
  "ragin cajuns", "golden flashes", "golden hurricane", "wolf pack", "red hawks",
  "blue raiders", "fighting hawks", "golden panthers", "rainbow warriors",
]);
// Short team label: NFL/CFB/MLB -> the ESPN letter code (ATL, OSU, LIB, NYY).
// CFB falls back to the school name (mascot stripped) for any team with no ESPN
// abbrev; all sports fall back to the full name when no mapping exists.
function teamShort(name, sport) {
  if (!name) return "";
  if (sport === "nfl") { const s = NFL_TEAM_ABBR[name]; return s ? s.toUpperCase() : String(name); }
  if (sport === "cfb") {
    if (CFB_TEAM_ABBR[name]) return CFB_TEAM_ABBR[name];   // ESPN abbrev (LIB, OSU, ...)
    const w = String(name).trim().split(/\s+/);
    if (w.length <= 1) return name;
    const last2 = w.slice(-2).join(" ").toLowerCase().replace(/[^a-z ]/g, "");
    const drop = CFB_2W_MASCOTS.has(last2) ? 2 : 1;
    const school = w.slice(0, Math.max(1, w.length - drop)).join(" ") || name;
    return school.replace(/\bState\b/g, "St.");
  }
  const a = TEAM_ABBR[name]; return a ? a.toUpperCase() : String(name);  // mlb
}

const gameRow = (r) => [logoPair(r.matchup, r.sport) + r.matchup, timeET(r.commence_time), r.market_label,
  pickLogo(r.pick_label, r.sport) + r.pick_label, fmtOdds(r.odds), pct1(r.model_prob), evp(r.ev), r.book];
const propRow = (r) => [logoImg(r.team, r.sport) + r.player_name, `${r.team || ""} · ${r.matchup}`, r.market_label,
  r.pick_label, fmtOdds(r.odds), pct1(r.model_prob), evp(r.ev), r.book];

/* ── Presentational templates (unchanged visuals) ─────────────────────── */
const nav = () => `<header class="site-header"><a class="brand" href="index.html" aria-label="CappingAlpha home"><span class="brand-wordmark">Capping<span class="brand-alpha"><img src="public/capping-alpha-mark-white.png" alt="alpha"></span>lpha</span></a><nav aria-label="Primary navigation"><a class="${page === 'dashboard' ? 'active' : ''}" href="index.html">Dashboard</a><a class="${page === 'cfb' ? 'active' : ''}" href="cfb.html">CFB</a><a class="${page === 'nfl' ? 'active' : ''}" href="nfl.html">NFL</a><a class="${page === 'rankings' ? 'active' : ''}" href="rankings.html">Rankings</a><a class="${page === 'ev' ? 'active' : ''}" href="ev.html">+EV</a><a class="${page === 'track' ? 'active' : ''}" href="track-record.html">Track Record</a><a class="${page === 'settings' ? 'active' : ''}" href="settings.html">Settings</a></nav><div class="live-status"><span></span>PREDICTIONS · LIVE</div></header>`;
const footer = () => `<footer>CappingAlpha model outputs are for informational purposes. Bet responsibly · 21+</footer>`;
const stat = (a, b, c, d = "") => `<article><span>${a}</span><strong class="${d}">${b}</strong><small>${c}</small></article>`;
const table = (rows, game = false) => rows.length
  ? `<div class="table-wrap"><table><thead><tr><th>${game ? "MATCHUP" : "PLAYER"}</th><th>MARKET</th><th>PICK</th><th>ODDS</th><th>MODEL</th><th>EV</th><th>BOOK</th></tr></thead><tbody>${rows.map(r => `<tr><td><b>${r[0]}</b><small>${r[1]}</small></td><td>${r[2]}</td><td class="pick">${r[3]}</td><td>${r[4]}</td><td><b>${r[5]}</b></td><td><span class="ev">${r[6]} EV</span></td><td>${r[7]}</td></tr>`).join("")}</tbody></table></div>`
  : `<div class="table-wrap"><p style="padding:24px;opacity:.6">No qualifying plays on the board yet — they post as odds are captured near game time.</p></div>`;
const edge = (name, meta, ev, items) => `<article class="edge-card"><div class="matchup"><div><h3>${name}</h3><p>${meta}</p></div><span>${ev} EV</span></div>${items.map(x => `<div class="market"><small>${x[0]}</small><b>${x[1]}</b><strong>${x[2]}</strong><i style="width:${x[3]}%"></i><p>${x[4]} <span>vs ${x[5]}</span></p><em>${x[6]}<br><u>${x[7]}</u></em></div>`).join("")}</article>`;

/* ── Aggregation from track_record_segments ───────────────────────────── */
function agg(segs) {
  const W = segs.reduce((s, r) => s + (+r.wins || 0), 0);
  const L = segs.reduce((s, r) => s + (+r.losses || 0), 0);
  const P = segs.reduce((s, r) => s + (+r.pushes || 0), 0);
  const U = segs.reduce((s, r) => s + (+r.units || 0), 0);
  const n = W + L + P;
  const clvNum = segs.reduce((s, r) => s + (+r.avg_clv || 0) * ((+r.wins || 0) + (+r.losses || 0) + (+r.pushes || 0)), 0);
  return {
    record: `${W}-${L}-${P}`,
    units: uStr(U),
    roi: pStr(n ? (U / n) * 100 : 0),
    winrate: pStr(W + L ? (W / (W + L)) * 100 : 0),
    clv: n ? pStr(clvNum / n) : "—",
    n,
  };
}

/* ── Page builders ────────────────────────────────────────────────────── */
// Prediction tool: confidence = how far the win prob is from a coin flip.
const conf = (wp) => Math.max(wp, 1 - wp);
const confPct = (wp) => `${(conf(wp) * 100).toFixed(0)}%`;
const predWinner = (r) => (r.home_win_prob >= 0.5 ? r.home_team_name : r.away_team_name);
const projScore = (r) => (r.pred_away_score == null || r.pred_home_score == null) ? "—" : `${Math.round(r.pred_away_score)}–${Math.round(r.pred_home_score)}`;
// Score flanked by each team's logo on its own side (away logo · away–home · home logo),
// for the predictions table's PROJECTED column. Kept separate from projScore so the
// confidence-card caption stays logo-free (it already shows logos in its title).
const projScoreLogos = (r) =>
  `${logoImg(r.away_team_name, r.sport)}${Math.round(r.pred_away_score)}–${Math.round(r.pred_home_score)}${logoImg(r.home_team_name, r.sport)}`;
const matchupOf = (r) => `${r.away_team_name} @ ${r.home_team_name}`;

// The model's line for each market (home-team perspective; the matchup shows away @ home).
const r05 = (x) => Math.round(x * 2) / 2;                        // nearest half-point
const homeSpread = (r) => {                                       // model spread on the home team
  const m = r05(r.pred_home_score - r.pred_away_score);
  return m > 0 ? `-${m}` : m < 0 ? `+${-m}` : "PK";
};
const modelTotal = (r) => r05(r.pred_home_score + r.pred_away_score).toFixed(1);
const mlFromProb = (p) => {                                       // model fair moneyline from a win prob
  p = Math.min(Math.max(p, 0.001), 0.999);
  const o = Math.min(Math.round(p >= 0.5 ? 100 * p / (1 - p) : 100 * (1 - p) / p), 9999);
  return p >= 0.5 ? `-${o}` : `+${o}`;                            // cap at ±9999 (books don't post beyond)
};
const homeML = (r) => mlFromProb(r.home_win_prob);               // model fair ML, home team
const awayML = (r) => mlFromProb(1 - r.home_win_prob);           // model fair ML, away team
// The model's LEAN vs the Vegas line (ESPN pre-game). market_spread is the HOME
// line (home favored -> negative). The model leans whichever side its projected
// margin beats: pred_margin + market_spread > 0 -> take home, else the away dog.
// Shown from the picked side (team logo + that side's line). Falls back to the
// model's own line when no market line is posted for the game.
const spreadLean = (r) => {
  if (r.market_spread == null || r.pred_home_score == null) return homeSpread(r);
  // Compare the model's spread ROUNDED to the half-point it's displayed at
  // (see homeSpread) against the line, so the shown lean can never contradict
  // the "model N" number beneath it. When they match, the model is sitting on
  // the line -> pick'em, not a confident dog pick on a fractional-point edge.
  const edge = r05(r.pred_home_score - r.pred_away_score) + r.market_spread;
  if (Math.abs(edge) < 0.25) return "PK";
  const likesHome = edge > 0;
  const team = likesHome ? r.home_team_name : r.away_team_name;
  const line = likesHome ? r.market_spread : -r.market_spread;
  return `${logoImg(team, r.sport)}${fmtLine(line)}`;
};
// Over if the model's projected total exceeds the Vegas total, else Under.
const totalLean = (r) => {
  if (r.market_total == null || r.pred_home_score == null) return modelTotal(r);
  const mt = r.pred_home_score + r.pred_away_score;
  return `${mt > r.market_total ? "Over" : "Under"} ${(+r.market_total).toFixed(1)}`;
};

async function predictions(sport) {
  const rows = await sb(`predictions_current?sport=eq.${sport}&order=commence_time.asc`);
  // Chronological: date + kickoff time. commence_time is an ISO string so a
  // string compare is chronological; fall back to game_date for any older row
  // written before commence_time was stored.
  const key = (r) => r.commence_time || r.game_date || "";
  return rows.map((r) => ({ ...r, c: conf(r.home_win_prob), sport }))
             .sort((a, b) => key(a).localeCompare(key(b)));
}
async function gradedPreds(q) { return sb(`prediction_accuracy?${q}`); }

/* ── +EV board (sharp-vs-best-line) ────────────────────────────────────── */
// Base true probability = the Pinnacle no-vig line (empirically the sharpest
// estimate). A row is a "pick" when the best available price is +EV vs that
// true prob -- i.e. line-shopping (the best soft price beats the sharp fair).
// Assistive · not a proven edge until the forward CLV record supports it.
async function evCurrent(sport) { return sb(`ev_current?sport=eq.${sport}&order=commence_time.asc`); }

const evPctVal = (p) => (p == null ? "—" : `${(p * 100).toFixed(1)}%`);
const evSigned = (x) => (x == null ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}%`);
// Market/side label from the matchup ("Away @ Home") + side.
const evSideLabel = (r) => {
  const [away, home] = String(r.matchup || " @ ").split(" @ ");
  if (r.market === "total") return r.side === "over" ? "Over" : "Under";
  const team = r.side === "home" ? home : away;
  return `${logoImg(team, r.sport)}${ctxEsc(team)} ${r.market === "moneyline" ? "ML" : "spread"}`;
};
// green when positive (edge/value), red when negative (worse than sharp), else neutral.
const evCls = (x) => (x == null ? "" : x > 1e-9 ? "ev-good" : x < -1e-9 ? "ev-bad" : "");
const EV_BOOKS = { draftkings: "DraftKings", fanduel: "FanDuel", fanatics: "Fanatics", "hard rock bet": "Hard Rock Bet", hardrockbet: "Hard Rock Bet", thescore: "theScore Bet", espnbet: "ESPN BET", "espn bet": "ESPN BET", williamhill_us: "Caesars", caesars: "Caesars", bet365: "Bet365", betmgm: "BetMGM", ballybet: "Bally Bet", "bally bet": "Bally Bet" };
const evBookName = (b) => (b ? (EV_BOOKS[String(b).toLowerCase()] || b) : "—");

function evRow(r) {
  const price = (am) => (am == null ? "—" : am > 0 ? `+${am}` : `${am}`);
  return `<tr class="ev-pick-row">
    <td><b>${logoPair(r.matchup, r.sport)}${ctxEsc(r.matchup)}</b><small>${timeET(r.commence_time)}</small></td>
    <td class="pick">${evSideLabel(r)}</td>
    <td>${evPctVal(r.base_prob)}<small>sharp / true</small></td>
    <td><b>${evPctVal(r.true_prob)}</b></td>
    <td>${evPctVal(r.best_line_implied)}<small>${evBookName(r.best_book)}</small></td>
    <td class="${evCls(r.soft_vs_sharp_gap)}">${evSigned(r.soft_vs_sharp_gap)}<small>vs sharp price</small></td>
    <td>${price(r.pinnacle_price)}<br><span class="${evCls(r.ev_pinnacle)}">${evSigned(r.ev_pinnacle)}</span><small>Pinnacle</small></td>
    <td><b>${price(r.best_price)}</b> <b class="ev-book">${evBookName(r.best_book)}</b><br><span class="${evCls(r.ev_best)}">${evSigned(r.ev_best)}</span></td>
  </tr>`;
}

function evSection(rows) {
  const picks = (rows || []).filter((r) => r.is_pick).sort((a, b) => (b.ev_best ?? -1) - (a.ev_best ?? -1));
  const body = picks.length
    ? `<div class="table-wrap"><table class="ev-table"><thead><tr>
        <th>MATCHUP</th><th>PICK</th><th>PINNACLE %</th><th>TRUE %</th><th>BEST %</th><th>GAP</th>
        <th>PINN PRICE · EV</th><th>BEST PRICE · EV</th></tr></thead>
        <tbody>${picks.map(evRow).join("")}</tbody></table></div>`
    : `<div class="ev-empty"><b>No +EV bets on the board right now.</b><p>Every game's best available price is within the vig of the sharp Pinnacle line — that's a valid pass, not a miss. Rows appear when a soft book prices better than the sharp fair.</p></div>`;
  return `<section class="section"><div class="section-title"><div><h2>+EV board</h2><p>Best available price vs the sharp Pinnacle line · <em>assistive · not a proven edge</em></p></div><div class="page-head-stat"><span>+EV PICKS</span><strong>${picks.length}</strong><small>on the board</small></div></div>${body}</section>`;
}

// NFL player-prop +EV board: the sim's per-player prop distributions vs the
// book prop line, surfaced only for players the sim projects as featured
// (projected-usage gate). Read from ev_prop_picks_current. Calibration was
// shown walk-forward; edge is a forward-CLV question (see track record).
async function evPropsCurrent(sport) {
  return sb(`ev_prop_picks_current?sport=eq.${sport}&is_pick=eq.true&order=ev_best.desc`).catch(() => []);
}
const PROP_MKT = { rush_yds: "Rush Yds", rec_yds: "Rec Yds", receptions: "Receptions", pass_yds: "Pass Yds", rush_att: "Rush Att" };
const propMktLabel = (m) => PROP_MKT[m] || m;
const propPickLabel = (r) => `${r.side === "over" ? "Over" : "Under"} ${r.line != null ? (+r.line).toFixed(1) : "—"}`;

function propsSection(rows) {
  const picks = (rows || []).filter((r) => r.is_pick).sort((a, b) => (b.ev_best ?? -1) - (a.ev_best ?? -1));
  const body = picks.length
    ? `<div class="table-wrap"><table class="ev-table"><thead><tr>
        <th>PLAYER</th><th>PROP</th><th>PICK</th><th>MODEL %</th><th>MARKET %</th><th>BEST PRICE · EV</th></tr></thead>
        <tbody>${picks.map((r) => `<tr class="ev-pick-row">
          <td><b>${r.player_name || "—"}</b><small>${r.matchup || ""} · ${timeET(r.commence_time)}</small></td>
          <td>${propMktLabel(r.market)}</td>
          <td class="pick">${propPickLabel(r)}</td>
          <td><b>${evPctVal(r.model_prob)}</b></td>
          <td>${evPctVal(r.market_prob)}<small>no-vig</small></td>
          <td><b>${evPrice(r.best_price)}</b> <b class="ev-book">${evBookName(r.best_book)}</b><br><span class="${evCls(r.ev_best)}">${evSigned(r.ev_best)}</span></td>
        </tr>`).join("")}</tbody></table></div>`
    : `<div class="ev-empty"><b>No +EV player props on the board right now.</b><p>Props appear for players the sim projects as featured (projected-usage gate) whose distribution beats a book's posted line. Rush &amp; receiving yards and receptions only; calibrated walk-forward, edge judged by forward CLV.</p></div>`;
  return `<section class="section"><div class="section-title"><div><h2>+EV player props <span style="font-size:.6em;opacity:.6">NFL · sim</span></h2><p>Sim projection vs the book prop line · <em>projected-featured only · assistive · not a proven edge</em></p></div><div class="page-head-stat"><span>+EV PROPS</span><strong>${picks.length}</strong><small>on the board</small></div></div>${body}</section>`;
}

// Auto-parlays: juiced-favorite legs (straight price <= -250) recombined into a
// single +EV plus-money ticket. legs is a JSONB array on ev_parlays_current.
async function evParlaysCurrent(sport) { return sb(`ev_parlays_current?sport=eq.${sport}&order=commence_time.asc`); }

const evPrice = (am) => (am == null ? "—" : am > 0 ? `+${am}` : `${am}`);
const parlayLegs = (legs) => {
  if (!legs) return [];
  if (typeof legs === "string") { try { return JSON.parse(legs); } catch { return []; } }
  return legs;
};
const parlayLegLabel = (leg, sport) => {
  const [away, home] = String(leg.matchup || " @ ").split(" @ ");
  const team = leg.market === "total" ? (leg.side === "over" ? "Over" : "Under")
    : `${logoImg(leg.side === "home" ? home : away, sport)}${leg.side === "home" ? home : away}`;
  const mkt = leg.market === "moneyline" ? "ML" : leg.market === "spread" ? "spread" : "";
  return `<li>${team} ${mkt} <span class="ev-leg-price">${evPrice(leg.price)}</span></li>`;
};
const parlayCard = (p) => {
  const legs = parlayLegs(p.legs);
  return `<article class="parlay-card">
    <div class="parlay-head"><b>${p.n_legs}-leg parlay</b><span class="parlay-price">${evPrice(p.parlay_price)} <small>${evBookName(p.book)}</small></span></div>
    <ol class="parlay-legs">${legs.map((l) => parlayLegLabel(l, p.sport)).join("")}</ol>
    <div class="parlay-foot"><span class="ev-good">${evSigned(p.ev)} EV</span><small>hit prob ${evPctVal(p.true_prob)} · legs must all win</small></div>
  </article>`;
};
function parlaySection(parlays) {
  const live = (parlays || []).filter((p) => p.is_pick !== false);
  if (!live.length) return "";
  return `<section class="section"><div class="section-title"><div><h2>+EV parlays</h2><p>Juiced favorites (worse than -250) recombined into a plus-money +EV ticket · <em>all legs must win</em></p></div></div><div class="parlay-grid">${live.map(parlayCard).join("")}</div></section>`;
}

// +EV pick PERFORMANCE (track record): graded picks from ev_results, joined to
// the ev_picks that produced them for matchup/price context. clv = closing-line
// value vs the Pinnacle close (the honest scoreboard).
async function evResultsRows() { return sb("ev_results?order=graded_at.desc&limit=1000"); }
// A pick can have several rebuild rows: keep the LATEST (max created_at) per (sport, game_pk, market, side)
// so the price shown is deterministic. If the 4000-row cap is hit, older picks may be missing -> warn.
const EV_PICKS_CAP = 4000;
async function evGradedPicks() {
  const rows = await sb(`ev_picks?is_pick=eq.true&select=sport,game_pk,market,side,matchup,commence_time,true_prob,base_prob,ev_best,best_book,best_price,pinnacle_price,created_at&order=created_at.asc&limit=${EV_PICKS_CAP}`);
  if ((rows || []).length >= EV_PICKS_CAP) console.warn(`evGradedPicks: hit the ${EV_PICKS_CAP}-row cap; older graded picks may be missing prices`);
  const latest = new Map();
  for (const r of rows || []) {
    const k = `${r.sport}|${r.game_pk}|${r.market}|${r.side}`, prev = latest.get(k);
    if (!prev || String(r.created_at || "") >= String(prev.created_at || "")) latest.set(k, r);
  }
  return [...latest.values()];
}

function evTrackSection(results, picks) {
  const key = (r) => `${r.sport}|${r.game_pk}|${r.market}|${r.side}`;
  const byKey = new Map((picks || []).map((p) => [key(p), p]));
  const graded = (results || []).filter((r) => r.won != null || r.clv != null);
  const decided = graded.filter((r) => r.won != null);
  const wins = decided.filter((r) => r.won).length;
  const winPct = decided.length ? `${(wins / decided.length * 100).toFixed(1)}%` : "—";
  const clvs = graded.filter((r) => r.clv != null).map((r) => +r.clv);
  const meanClv = clvs.length ? `${(clvs.reduce((s, x) => s + x, 0) / clvs.length * 100).toFixed(2)}%` : "—";
  const clvPos = clvs.length ? `${(clvs.filter((x) => x > 0).length / clvs.length * 100).toFixed(0)}%` : "—";
  const row = (r) => {
    const p = byKey.get(key(r)) || {};
    const price = (am) => (am == null ? "—" : am > 0 ? `+${am}` : `${am}`);
    const badge = r.won == null ? "—" : `<span class="${r.won ? "win" : "loss"}">${r.won ? "✓" : "✗"}</span>`;
    return `<tr data-league="${r.sport}"><td><b>${logoPair(p.matchup || "", r.sport)}${p.matchup || `${r.sport.toUpperCase()} ${r.game_pk}`}</b><small>${String(r.sport).toUpperCase()}</small></td>
      <td class="pick">${p.matchup ? evSideLabel({ ...p }) : `${r.market} · ${r.side}`}</td>
      <td><b>${price(p.best_price)}</b> <b class="ev-book">${evBookName(p.best_book)}</b></td>
      <td class="${evCls(p.ev_best)}">${evSigned(p.ev_best)}</td>
      <td>${badge}</td>
      <td class="${evCls(r.clv)}">${evSigned(r.clv)}</td></tr>`;
  };
  const table = graded.length
    ? `<div class="table-wrap"><table class="ev-table"><thead><tr><th>MATCHUP</th><th>PICK</th><th>BEST PRICE</th><th>EV (at pick)</th><th>RESULT</th><th>CLV</th></tr></thead><tbody>${graded.map(row).join("")}</tbody></table></div>`
    : `<div class="ev-empty"><b>No graded +EV picks yet.</b><p>Picks are graded after games settle (the grade-ev job runs Tuesdays). This is the honest scoreboard — win rate and closing-line value vs the sharp Pinnacle close — and the +EV board only earns trust once this record holds up.</p></div>`;
  return `<section class="compact-stats">${stat('+EV RECORD', `${wins}-${decided.length - wins}`, `${decided.length} graded`, 'blue')}${stat('WIN RATE', winPct, 'vs ~52.4% breakeven')}${stat('MEAN CLV', meanClv, 'vs Pinnacle close', 'cyan')}${stat('CLV POSITIVE', clvPos, 'share of picks')}</section>
    <section class="section"><div class="section-title"><div><h2>+EV pick performance</h2><p>Graded vs the result + the Pinnacle closing line · <em>assistive · not a proven edge</em></p></div></div>${table}</section>`;
}

// Graded best +EV parlays (ev_best_parlay_results), newest first: one $10 ticket
// each (shown at the user's unit); a pushed leg drops out of the payout, so a ✓✓P ticket pays the 2-leg price.
async function parlayResultsRows() { return sb("ev_best_parlay_results?order=first_commence.desc&limit=50").catch(() => []); }
function gradedParlaysSection(rows, s = getSettings()) {
  rows = rows || [];
  const k = unitScale(s);
  const head = `<div class="section-title"><div><h2>Graded parlays <span style="font-size:.6em;opacity:.6">${unitLabel(s)} per ticket</span></h2><p>Every best +EV parlay once it's decided (any losing leg, or every leg settled) · ✓ won · ✗ lost · P push (the leg drops out and the ticket pays on the rest).</p></div></div>`;
  if (!rows.length)
    return `<section class="section">${head}<div class="ev-empty"><b>No graded parlays yet — tickets grade once it's decided (any losing leg, or every leg settled).</b></div></section>`;
  const mark = (res) => `<span class="pc-mark ${res || ""}">${res === "win" ? "✓" : res === "loss" ? "✗" : res === "push" ? "P" : "—"}</span>`;
  const day = (iso) => iso ? new Date(iso).toLocaleDateString("en-US", { timeZone: "America/New_York", weekday: "short", month: "short", day: "numeric" }) : "";
  const card = (r) => {
    const legs = parlayLegs(r.legs);
    const leg = (l) => `<li class="pc-leg"><span><b>${mark(l.result)}${bpLegLogo(l)}${l.label || "—"}</b><small>${l.matchup || ""}</small></span><span class="pc-odds"><b>${evPrice(l.price)}</b></span></li>`;
    return `<article class="parlay-card"><div class="pc-head"><span><b>${evPrice(r.parlay_price)} @ ${evBookName(r.book)} <span class="pc-res ${r.result || ""}">${String(r.result || "—").toUpperCase()}</span></b><small>${day(r.first_commence)} · ${legs.length} legs</small></span><strong class="${pnlCls(+r.pnl)}">${money((+r.pnl || 0) * k)}</strong></div><ol class="pc-legs">${legs.map(leg).join("")}</ol></article>`;
  };
  return `<section class="section">${head}<div class="parlay-grid">${rows.map(card).join("")}</div></section>`;
}

// NFL prop +EV PERFORMANCE (track record): graded prop picks from
// ev_prop_results. result is win/loss/push; clv = closing-line value vs the
// prop closing price. The honest scoreboard for the prop board.
async function propResultsRows() { return sb("ev_prop_results?order=created_at.desc&limit=1000").catch(() => []); }
function propTrackSection(results) {
  const graded = (results || []).filter((r) => r.result != null || r.clv != null);
  const decided = graded.filter((r) => r.result === "win" || r.result === "loss");
  const wins = decided.filter((r) => r.result === "win").length;
  const winPct = decided.length ? `${(wins / decided.length * 100).toFixed(1)}%` : "—";
  const clvs = graded.filter((r) => r.clv != null).map((r) => +r.clv);
  const meanClv = clvs.length ? `${(clvs.reduce((s, x) => s + x, 0) / clvs.length * 100).toFixed(2)}%` : "—";
  const clvPos = clvs.length ? `${(clvs.filter((x) => x > 0).length / clvs.length * 100).toFixed(0)}%` : "—";
  const profits = decided.filter((r) => r.profit != null).map((r) => +r.profit);
  const roi = profits.length ? `${(profits.reduce((s, x) => s + x, 0) / profits.length * 100).toFixed(1)}%` : "—";
  const badge = (res) => res == null ? "—" : `<span class="${res === "win" ? "win" : res === "loss" ? "loss" : ""}">${res === "win" ? "✓" : res === "loss" ? "✗" : "P"}</span>`;
  const rowsHtml = graded.map((r) => `<tr>
      <td><b>${r.player_name || "—"}</b><small>${propMktLabel(r.market)}</small></td>
      <td class="pick">${propPickLabel(r)}</td>
      <td>${r.actual != null ? (+r.actual).toFixed(1) : "—"}</td>
      <td>${badge(r.result)}</td>
      <td class="${evCls(r.clv)}">${evSigned(r.clv)}</td></tr>`).join("");
  const table = graded.length
    ? `<div class="table-wrap"><table class="ev-table"><thead><tr><th>PLAYER</th><th>PICK</th><th>ACTUAL</th><th>RESULT</th><th>CLV</th></tr></thead><tbody>${rowsHtml}</tbody></table></div>`
    : `<div class="ev-empty"><b>No graded prop picks yet.</b><p>Prop picks are graded after games settle (grade-ev-props). This is the honest scoreboard — win rate and closing-line value on the sim's prop picks — and the prop board earns trust only once this record holds up.</p></div>`;
  return `<section class="compact-stats">${stat('PROP RECORD', `${wins}-${decided.length - wins}`, `${decided.length} graded`, 'blue')}${stat('WIN RATE', winPct, 'vs ~52.4% breakeven')}${stat('ROI', roi, 'per 1u at pick price', 'cyan')}${stat('MEAN CLV', meanClv, `${clvPos} positive`)}</section>
    <section class="section"><div class="section-title"><div><h2>+EV player-prop performance <span style="font-size:.6em;opacity:.6">NFL · sim</span></h2><p>Graded vs the actual + the prop closing line · <em>assistive · not a proven edge</em></p></div></div>${table}</section>`;
}

// Sim accuracy vs the captured pre-kickoff prop line, per market -- read from
// the pre-aggregated nfl_prop_accuracy view (GROUP BY ROLLUP(market) over
// nfl_prop_grades) rather than paging raw grade rows: PostgREST caps
// unordered/limited reads at 1000 rows, which silently truncated the old
// per-row aggregation to an arbitrary subset once grades passed ~2 weeks.
// "Lean" is the side of the book line the sim's distribution favored; a
// lower MAE than the line means the sim's number landed closer to the
// actual than the book's.
async function propLineGradesRows() { return sb("nfl_prop_accuracy?select=*").catch(() => []); }
function propAccuracySection(rows) {
  const ORDER = ["pass_yds", "rush_yds", "rec_yds", "receptions", "rush_att"];
  // Numeric columns arrive as strings from PostgREST -- coerce with `+`.
  const fmtMae = (x) => (x == null ? "—" : (+x).toFixed(1));
  const fmtBias = (x) => (x == null ? "—" : `${+x >= 0 ? "+" : ""}${(+x).toFixed(1)}`);
  const row = (label, r) => {
    const n = +r.n, hits = +r.hits, decided = +r.decided;
    const leanPct = decided ? `${(hits / decided * 100).toFixed(1)}%` : "—";
    return `<tr><td><b>${label}</b></td><td>${n}</td><td>${leanPct}</td><td>${fmtMae(r.sim_mae)}</td><td>${fmtMae(r.line_mae)}</td><td>${fmtBias(r.bias)}</td></tr>`;
  };
  const head = `<div class="section-title"><div><h2>Sim prop accuracy <span style="font-size:.6em;opacity:.6">NFL · sim</span></h2><p>Graded vs the pre-kickoff book line · lean = the side of the line the sim's distribution favored · a lower MAE than the line means the sim landed closer to the actual than the book.</p></div></div>`;
  // ROLLUP over zero graded rows still yields one {market:'all', n:0} row.
  if (!rows || !rows.some((r) => +r.n > 0))
    return `<section class="section">${head}<div class="ev-empty"><b>No graded prop lines yet — fills in after games with captured lines settle.</b></div></section>`;
  const byMkt = new Map(rows.map((r) => [r.market, r]));
  const marketRows = ORDER
    .filter((mk) => byMkt.has(mk) && +byMkt.get(mk).n > 0)
    .map((mk) => row(PROP_MKT[mk] || mk, byMkt.get(mk)));
  const all = byMkt.get("all");
  const rowsHtml = marketRows.join("") + (all ? row("All", all) : "");
  const table = `<div class="table-wrap"><table class="ev-table"><thead><tr><th>MARKET</th><th>N</th><th>LEAN HIT %</th><th>SIM MAE</th><th>LINE MAE</th><th>BIAS</th></tr></thead><tbody>${rowsHtml}</tbody></table></div>`;
  return `<section class="section">${head}${table}</section>`;
}

/* ── +EV page: every +EV pick in one place ─────────────────────────────── */
// NFL + CFB game lines (ev_current) and NFL player props (ev_prop_picks_current),
// upcoming games only. ev_current carries the best book + price but not the
// spread/total number; ev_best_lines (db/migration_ev_best_lines.sql) supplies
// the best book's line per pick, since odds_snapshot isn't browser-readable.
async function evBestLines() {
  const rows = await sb("ev_best_lines?select=game_pk,market,side,line").catch(() => []);
  return new Map(rows.map((r) => [`${r.game_pk}|${r.market}|${r.side}`, r.line == null ? null : +r.line]));
}
// The bet as you'd place it: "Packers -3.5", "Over 43.5", "Falcons ML".
function evGamePick(r, line) {
  const [away, home] = String(r.matchup || " @ ").split(" @ ");
  if (r.market === "total") return `${r.side === "over" ? "Over" : "Under"}${line == null ? "" : ` ${line}`}`;
  const team = r.side === "home" ? home : away;
  const tail = r.market === "moneyline" ? " ML" : line == null ? " spread" : line === 0 ? " PK" : ` ${line > 0 ? "+" : ""}${line}`;
  return `${logoImg(team, r.sport)}${team}${tail}`;
}
// Best +EV parlays (ev_best_parlays_current): 3 legs, one per game, each ≥55%
// to win and +EV at the SAME book, so the ticket can actually be placed there.
// Legs carry their own label/price/prob; legs may arrive as a JSON string.
async function evBestParlays() { return sb("ev_best_parlays_current?select=*&order=ev.desc").catch(() => []); }
// Team logo for a moneyline/spread game leg (side -> team from "Away @ Home"); totals + props get none.
const bpLegLogo = (l) => {
  if (l.kind !== "game" || (l.market !== "moneyline" && l.market !== "spread")) return "";
  const [away, home] = String(l.matchup || " @ ").split(" @ ");
  return logoImg(l.side === "home" ? home : away, l.sport);
};
// Legs span days (Thu–Mon), so kickoff carries the weekday: "Sun 1:00 PM ET".
const bpKick = (iso) => iso ? `${new Date(iso).toLocaleDateString("en-US", { timeZone: "America/New_York", weekday: "short" })} ${timeET(iso)}` : "";
// Sort keys for the +EV page's "Sort by" (EV, payout via decimal odds, book
// name, confidence = the model's / sim's win probability for the bet).
const evSortAttrs = (ev, american, book, prob) => {
  const dec = american == null || +american === 0 ? 0 : (+american > 0 ? 1 + american / 100 : 1 + 100 / -american);
  const conf = prob == null || !Number.isFinite(+prob) ? -1 : +prob;
  return `data-sev="${ev == null ? -1 : +ev}" data-sodds="${dec}" data-sbook="${evBookName(book)}" data-sconf="${conf}"`;
};
// Unit stake label: "$10", "$12.50".
const unitLabel = (s) => `$${Number.isInteger(s.unit) ? s.unit : s.unit.toFixed(2)}`;
function bestParlayCard(p, s) {
  const legs = parlayLegs(p.legs), dec = p.parlay_dec == null ? NaN : +p.parlay_dec;   // +null would be 0
  const leg = (l) => `<li class="pc-leg"><span><b>${bpLegLogo(l)}${l.label || "—"}</b><small>${l.matchup || ""}${l.commence_time ? ` · ${bpKick(l.commence_time)}` : ""}</small></span><span class="pc-odds"><b>${evPrice(l.price)}</b><small>${evPctVal(l.prob)} win</small></span></li>`;
  const k = kellyStake(p.true_prob, p.parlay_price, s);
  return `<article class="parlay-card" ${evSortAttrs(p.ev, p.parlay_price, p.book, p.true_prob)}><div class="pc-head"><span><b>${evPrice(p.parlay_price)} @ ${evBookName(p.book)}</b><small><span class="${evCls(+p.ev)}">EV ${evSigned(+p.ev)}</span> · win ${evPctVal(p.true_prob)}</small></span>${Number.isFinite(dec) && dec > 1 ? `<strong>${unitLabel(s)} to win $${(10 * (dec - 1) * unitScale(s)).toFixed(2)}</strong>` : ""}</div>${k ? `<div class="pc-kelly">Kelly ${fmtKelly(k)}</div>` : ""}<ol class="pc-legs">${legs.map(leg).join("")}</ol></article>`;
}
function bestParlaysSection(parlays, s) {
  const body = parlays.length
    ? `<div class="parlay-grid">${parlays.map((p) => bestParlayCard(p, s)).join("")}</div>`
    : `<div class="ev-empty"><b>No +EV parlays right now — needs 3 legs (≥55% win probability, one per game) that are all +EV at the same book.</b></div>`;
  return `<section class="section" data-evsec="parlays"><div class="section-title"><div><h2>Parlays</h2><p>Three +EV legs, one per game, each ≥55% to win, all priced at one book · win % = every leg hitting · <em>all legs must win</em>.</p></div></div>${body}</section>`;
}
// Per-book prices for each current +EV pick (ev_pick_prices_current,
// db/migration_settings_prices.sql) so the page can re-price at the user's
// books. Missing view → [] → every pick keeps its own best book.
async function evPickPrices() { return sb("ev_pick_prices_current?select=*").catch(() => []); }
const evPriceKey = (kind, r) => (kind === "prop"
  ? `prop|${r.game_pk}|${r.player_id}|${r.market}|${r.side}` : `game|${r.game_pk}|${r.market}|${r.side}`);
function evPricesMap(rows) {
  const m = new Map();
  for (const r of rows || []) {
    let p = r.prices;
    if (typeof p === "string") { try { p = JSON.parse(p); } catch { p = null; } }
    if (p && typeof p === "object" && Object.keys(p).length) m.set(evPriceKey(r.kind, r), p);
  }
  return m;
}
// Re-price one +EV pick at the user's books: the best (highest decimal) price
// among the selected books in its per-book map, or the pick's own best book
// when the map has no entry for it. EV = prob × dec − 1 at that price. The
// board's gates apply per book: a price whose EV is above EV_CEILING is a
// stale-line artifact and a game straight at MAX_STRAIGHT_JUICE or worse is
// never listed -- such a book is skipped and the next-best selected book used.
// The chosen price must then clear the board's floor (game EV > 1%, prop EV >
// 0) and the user's min EV; otherwise null. Pure: settings + prices come in as
// arguments. probKey: true_prob (game) / model_prob (prop).
const EV_CEILING = 0.25, EV_GAME_MIN = 0.01, MAX_STRAIGHT_JUICE = -250;   // serving/board.py, ev_pilot.py
function repricePick(pick, pricesMap, s, probKey) {
  const prob = pick ? +pick[probKey] : NaN;
  if (pick?.[probKey] == null || !(prob > 0 && prob < 1)) return null;
  const kind = pick.kind || (pick.player_id != null ? "prop" : "game");
  const prices = (pricesMap && pricesMap.get(evPriceKey(kind, pick)))
    || (pick.best_book != null && pick.best_price != null ? { [pick.best_book]: pick.best_price } : {});
  let best = null;
  for (const [book, am] of Object.entries(prices)) {
    const a = +am;
    if (am == null || !Number.isFinite(a) || (a > -100 && a < 100) || !bookSelected(book, s)) continue;
    if (kind === "game" && a <= MAX_STRAIGHT_JUICE) continue;
    const dec = a > 0 ? 1 + a / 100 : 1 + 100 / -a, ev = prob * dec - 1;
    if (ev > EV_CEILING) continue;
    // Ties keep the board's own best book.
    if (!best || dec > best.dec || (dec === best.dec && book === pick.best_book)) best = { book, price: a, dec, ev };
  }
  if (!best) return null;
  const ev = best.ev;
  if (!(ev > (kind === "game" ? EV_GAME_MIN : 0)) || ev < s.minEv / 100) return null;
  return { book: best.book, price: best.price, ev };
}
// True when the user's books / min EV narrow the board (unit, bankroll and
// Kelly only change stakes, not which picks show).
const evFiltered = (s) => s.books.length !== US_BOOKS.length || s.minEv > 0;
async function buildEv() {
  const s = getSettings();
  const [nfl, cfb, props, parlaysAll, priceRows] = await Promise.all([
    evCurrent("nfl").catch(() => []), evCurrent("cfb").catch(() => []), evPropsCurrent("nfl"), evBestParlays(), evPickPrices(),
  ]);
  const priceMap = evPricesMap(priceRows);
  // Each shown pick carries rp = its re-priced {book, price, ev} and prob.
  const reprice = (rows, probKey) => rows.filter((r) => r.is_pick)
    .map((r) => { const rp = repricePick(r, priceMap, s, probKey); return rp && { ...r, rp, prob: +r[probKey] }; })
    .filter(Boolean).sort((a, b) => b.rp.ev - a.rp.ev);
  const games = reprice([...nfl.map((r) => ({ ...r, sport: "nfl" })), ...cfb.map((r) => ({ ...r, sport: "cfb" }))], "true_prob");
  const lineBy = await evBestLines();
  const propPicks = reprice(props || [], "model_prob");
  const parlays = (parlaysAll || []).filter((p) => bookSelected(p.book, s) && +p.ev >= s.minEv / 100);
  const nN = games.filter((r) => r.sport === "nfl").length, nC = games.length - nN;
  const gameLink = (r) => `game.html?sport=${r.sport}&game=${r.game_pk}`;
  const kellyTd = (r) => `<td><b>${fmtKelly(kellyStake(r.prob, r.rp.price, s))}</b></td>`;
  const gameRows = games.map((r) => `<tr class="ev-pick-row" data-evsport="${r.sport}" data-evmkt="${r.market}" ${evSortAttrs(r.rp.ev, r.rp.price, r.rp.book, r.prob)}>
      <td><a href="${gameLink(r)}"><b>${logoPair(r.matchup, r.sport)}${r.matchup}</b></a><small>${r.sport.toUpperCase()} · ${timeET(r.commence_time)}</small></td>
      <td class="pick">${evGamePick(r, lineBy.get(`${r.game_pk}|${r.market}|${r.side}`) ?? null)}</td>
      <td><b>${evPrice(r.rp.price)}</b> <b class="ev-book">${evBookName(r.rp.book)}</b></td>
      <td class="${evCls(r.rp.ev)}"><b>${evSigned(r.rp.ev)}</b></td>
      <td><b>${evPctVal(r.true_prob)}</b><small>vs ${evPctVal(r.best_line_implied)} market</small></td>${kellyTd(r)}</tr>`).join("");
  const emptyGames = `<div class="ev-empty"><b>No +EV game lines right now.</b><p>Every game's best available price is within the vig of the sharp line — that's a pass, not a miss.</p></div>`;
  const gameTable = games.length
    ? `<div class="table-wrap"><table class="ev-table"><thead><tr><th>MATCHUP</th><th>PICK</th><th>BEST BOOK</th><th>EV</th><th>MODEL %</th><th>KELLY</th></tr></thead><tbody>${gameRows}</tbody></table></div><div class="ev-empty ev-filter-empty" hidden><b>No +EV game lines match this filter right now.</b></div>`
    : emptyGames;
  const propRows = propPicks.map((r) => `<tr class="ev-pick-row" data-evsport="nfl" data-evmkt="prop" ${evSortAttrs(r.rp.ev, r.rp.price, r.rp.book, r.prob)}>
      <td><a href="game.html?sport=nfl&game=${r.game_pk}"><b>${r.player_name || "—"}</b></a><small>${r.matchup || ""} · ${timeET(r.commence_time)}</small></td>
      <td class="pick">${propMktLabel(r.market)} · ${propPickLabel(r)}</td>
      <td><b>${evPrice(r.rp.price)}</b> <b class="ev-book">${evBookName(r.rp.book)}</b></td>
      <td class="${evCls(r.rp.ev)}"><b>${evSigned(r.rp.ev)}</b></td>
      <td><b>${evPctVal(r.model_prob)}</b><small>vs ${evPctVal(r.market_prob)} market</small></td>${kellyTd(r)}</tr>`).join("");
  const propTable = propPicks.length
    ? `<div class="table-wrap"><table class="ev-table"><thead><tr><th>PLAYER</th><th>PICK</th><th>BEST BOOK</th><th>EV</th><th>SIM %</th><th>KELLY</th></tr></thead><tbody>${propRows}</tbody></table></div>`
    : `<div class="ev-empty"><b>No +EV player props right now.</b><p>Props appear when the sim's distribution beats a book's posted line for a player it projects as featured.</p></div>`;
  const total = games.length + propPicks.length;
  const evs = [...games.map((r) => r.rp.ev), ...propPicks.map((r) => r.rp.ev)];
  const nb = s.books.length;
  const note = evFiltered(s)
    ? `<div class="ev-settings-note">Filtered by your settings: ${nb} book${nb === 1 ? "" : "s"}${s.minEv > 0 ? ` · min EV ${s.minEv.toFixed(1)}%` : ""} · <a href="settings.html">Change</a></div>` : "";
  const chipRow = (key, label, opts) => `<div class="track-league-filter ev-filter" data-evgroup="${key}" role="group" aria-label="${label}">${opts
    .map(([k, l]) => `<button data-evf="${k}" class="${k === evFilterState[key] ? "selected" : ""}">${l}</button>`).join("")}</div>`;
  const chips = chipRow("league", "League", [["all", "All"], ["nfl", "NFL"], ["cfb", "CFB"]])
    + chipRow("market", "Bet type", [["all", "All bets"], ["moneyline", "ML"], ["spread", "Spread"], ["total", "Total"], ["prop", "Props"]]);
  return `<main><section class="page-heading"><div><p class="eyebrow">+EV BOARD</p><h1>+EV picks</h1><p>Every +EV bet on the board — NFL &amp; CFB game lines and NFL player props — with the best book, its price and line. Filter by league and bet type; sort by EV, odds, sportsbook or confidence. <em>Assistive · not a proven edge.</em></p>${note}</div><div class="page-head-stat"><span>+EV PICKS</span><strong>${total}</strong><small>on the board</small></div></section>
    <section class="compact-stats">${stat("NFL GAME LINES", nN, "+EV picks")}${stat("CFB GAME LINES", nC, "+EV picks", "blue")}${stat("PLAYER PROPS", propPicks.length, "NFL · sim", "cyan")}${stat("TOP EV", evs.length ? evSigned(Math.max(...evs)) : "—", `best on the board · ${parlays.length} parlay${parlays.length === 1 ? "" : "s"}`)}</section>
    <div class="ev-toolbar"><div class="ev-filters">${chips}</div>
      <label class="ev-sort">Sort by <select id="ev-sort"><option value="ev">EV: High to Low</option><option value="odds">Odds: High to Low</option><option value="book">Sportsbook: A to Z</option><option value="conf">Confidence: High to Low</option></select></label></div>
    ${bestParlaysSection(parlays, s)}
    <section class="section" data-evsec="games"><div class="section-title"><div><h2>Game lines <span style="font-size:.6em;opacity:.6">NFL · CFB</span></h2><p>Best available price vs the sharp line · model % = true probability vs the best price's implied %.</p></div></div>${gameTable}</section>
    <section class="section" data-evsec="props"><div class="section-title"><div><h2>Player props <span style="font-size:.6em;opacity:.6">NFL · sim</span></h2><p>Sim distribution vs the book line · sim % vs the no-vig market %.</p></div></div>${propTable}</section>
    <div class="ev-empty ev-none" hidden><b>No +EV picks match this filter.</b><p>Player props are NFL only.</p></div></main>`;
}
// Reorder every +EV list (parlay cards, game-line rows, prop rows) in place.
// Ties fall back to EV, high to low. The choice is remembered per browser.
function sortEvPage(mode) {
  const num = (el, k) => +el.dataset[k] || 0;
  const cmp = {
    ev: (a, b) => num(b, "sev") - num(a, "sev"),
    odds: (a, b) => num(b, "sodds") - num(a, "sodds") || num(b, "sev") - num(a, "sev"),
    book: (a, b) => (a.dataset.sbook || "").localeCompare(b.dataset.sbook || "") || num(b, "sev") - num(a, "sev"),
    conf: (a, b) => num(b, "sconf") - num(a, "sconf") || num(b, "sev") - num(a, "sev"),
  }[mode] || ((a, b) => num(b, "sev") - num(a, "sev"));
  document.querySelectorAll('[data-evsec] tbody, [data-evsec] .parlay-grid').forEach((box) => {
    [...box.children].filter((el) => el.dataset.sev != null).sort(cmp).forEach((el) => box.appendChild(el));
  });
}
function wireEvPage() {
  const sel = document.getElementById("ev-sort");
  if (sel) {
    let saved = null;
    try { saved = localStorage.getItem("ca-ev-sort"); } catch {}
    if (saved && [...sel.options].some((o) => o.value === saved)) { sel.value = saved; sortEvPage(saved); }
    sel.addEventListener("change", () => {
      sortEvPage(sel.value);
      try { localStorage.setItem("ca-ev-sort", sel.value); } catch {}
    });
  }
  const bar = document.querySelector(".ev-filters"); if (!bar) return;
  bar.addEventListener("click", (e) => {
    const b = e.target.closest("[data-evgroup] button[data-evf]"); if (!b) return;
    const group = b.closest("[data-evgroup]");
    group.querySelectorAll("button").forEach((x) => x.classList.toggle("selected", x === b));
    evFilterState[group.dataset.evgroup] = b.dataset.evf;
    applyEvFilter();
  });
  applyEvFilter();   // re-apply the kept choice after the periodic re-render
}
// League (All/NFL/CFB) × bet type (All/ML/Spread/Total/Props). Kept in memory
// so the 5-minute re-render doesn't reset it; a page load starts at All.
const evFilterState = { league: "all", market: "all" };
function applyEvFilter() {
  const { league, market } = evFilterState;
  const gs = document.querySelector('[data-evsec="games"]'), ps = document.querySelector('[data-evsec="props"]');
  if (!gs || !ps) return;
  const rowOk = (tr) => (league === "all" || tr.dataset.evsport === league) && (market === "all" || tr.dataset.evmkt === market);
  gs.hidden = market === "prop";
  ps.hidden = league === "cfb" || (market !== "all" && market !== "prop");   // props are NFL only
  const pp = document.querySelector('[data-evsec="parlays"]');
  if (pp) pp.hidden = league !== "all" || market !== "all";   // tickets mix leagues + bet types, so they only fit All
  const rows = [...gs.querySelectorAll("tr[data-evmkt]")];
  rows.forEach((tr) => { tr.hidden = !rowOk(tr); });
  const empty = gs.querySelector(".ev-filter-empty");
  if (empty) empty.hidden = !rows.length || rows.some((tr) => !tr.hidden);
  const none = document.querySelector(".ev-none");
  if (none) none.hidden = !(gs.hidden && ps.hidden);
}

/* ── Profit trackers (a unit per bet) ──────────────────────────────────── */
// Daily P&L rows from prediction_pnl_daily / ev_pnl_daily:
// {game_date, sport, market, n, wins, losses, pushes, pnl}. Numerics may be strings.
const money = (x) => `${x < 0 ? "−" : "+"}$${Math.abs(x).toFixed(2)}`;
const pnlCls = (x) => (x > 0 ? "pnl-pos" : x < 0 ? "pnl-neg" : "");
function pnlAgg(rs) {
  return rs.reduce((a, r) => ({ n: a.n + +r.n, w: a.w + +r.wins, l: a.l + +r.losses, p: a.p + +r.pushes, pnl: a.pnl + +r.pnl }),
    { n: 0, w: 0, l: 0, p: 0, pnl: 0 });
}
// Total wagered: whole dollars stay whole ($1,200); a fractional total (from a
// fractional unit) shows exactly two decimals ($1,137.50).
const fmtWagered = (x) => {
  const c = Math.round(x * 100) / 100;
  return c.toLocaleString("en-US", Number.isInteger(c) ? { maximumFractionDigits: 0 } : { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};
// Cumulative P&L chart: a blue total-wagered line ($10 x every bet) and the
// profit line, green above $0 / red below (a hard-stop gradient at the $0 y, so
// a line crossing zero changes color exactly there), on ONE shared $ scale.
let pnlChartSeq = 0;
function pnlChart(profit, wagered) {
  if (profit.length < 2) return "";
  const all = [...profit, ...wagered, 0];
  const max = Math.max(...all), min = Math.min(...all), range = max - min || 1;
  const y = (v) => 250 - ((v - min) / range) * 240;
  const x = (i) => i / (profit.length - 1) * 1000;
  const path = (vals) => "M" + vals.map((v, i) => `${x(i).toFixed(0)} ${y(v).toFixed(1)}`).join(" L");
  const y0 = y(0), id = `pnlg${++pnlChartSeq}`, off = (y0 / 260 * 100).toFixed(2);
  const last = profit.at(-1);
  return `<defs><linearGradient id="${id}" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="260"><stop offset="${off}%" stop-color="var(--green)"/><stop offset="${off}%" stop-color="var(--red)"/></linearGradient></defs>`
    + `<line x1="0" x2="1000" y1="${y0.toFixed(1)}" y2="${y0.toFixed(1)}" stroke="var(--muted)" stroke-width="1" stroke-dasharray="6 6" vector-effect="non-scaling-stroke"/>`
    + `<path d="${path(wagered)}" fill="none" stroke="var(--blue)" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`
    + `<path d="${path(profit)}" fill="none" stroke="url(#${id})" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`
    + `<circle cx="1000" cy="${y(last).toFixed(1)}" r="6" fill="${last >= 0 ? "var(--green)" : "var(--red)"}"/>`;
}
// markets: [[marketKey, label], ...]; an "All" card is appended.
// Cards + chart for one P&L section, filtered to `sport` ("all" | "nfl" | "cfb").
// Rows are graded at $10 a bet; every $ amount is scaled to the user's unit
// (ROI and W-L are scale-free).
function pnlBody(allRows, markets, sport, s = getSettings()) {
  const k = unitScale(s);
  const rows = sport === "all" ? allRows : allRows.filter((r) => r.sport === sport);
  const bySport = (rs) => sport !== "all" ? "" : ["nfl", "cfb", "mixed"].map((sp) => { const a = pnlAgg(rs.filter((r) => r.sport === sp)); return a.n ? `${sp === "mixed" ? "MIXED" : sp.toUpperCase()} ${money(a.pnl * k)}` : null; }).filter(Boolean).join(" · ");
  const card = (label, rs) => {
    const a = pnlAgg(rs);
    if (!a.n) return stat(label, "—", "no graded bets yet");
    const risked = (a.w + a.l) * 10;
    const roi = risked ? `${(a.pnl / risked * 100).toFixed(1)}% ROI` : "";
    const split = bySport(rs);
    return stat(label, money(a.pnl * k), `${a.w}-${a.l}${a.p ? `-${a.p}` : ""} · ${roi}${split ? ` · ${split}` : ""}`, pnlCls(a.pnl));
  };
  const cards = markets.map(([mk, label]) => card(label, rows.filter((r) => r.market === mk))).join("") + card("ALL", rows);
  // Cumulative across every market, by game date (oldest -> newest): profit and
  // total wagered (a unit per bet, pushes included -- the stake was still risked).
  const byDate = new Map();
  rows.forEach((r) => {
    const d = byDate.get(r.game_date) || { pnl: 0, n: 0 };
    byDate.set(r.game_date, { pnl: d.pnl + +r.pnl, n: d.n + +r.n });
  });
  let cum = 0, staked = 0;
  const dates = [...byDate.keys()].sort();
  const series = dates.map((d) => (cum += byDate.get(d).pnl * k));
  const wagered = dates.map((d) => (staked += byDate.get(d).n * 10 * k));
  const chart = pnlChart(series, wagered);
  const scope = sport === "all" ? "all markets" : `${sport.toUpperCase()} · all markets`;
  return `<div class="compact-stats pnl-stats">${cards}</div>
    <div class="pnl-chart">${chart ? `<svg viewBox="0 0 1000 260" preserveAspectRatio="none">${chart}</svg><small><span class="pnl-key wag"></span>Total wagered $${fmtWagered(staked)} · <span class="pnl-key ${cum >= 0 ? "up" : "down"}"></span>Profit ${money(cum)} · cumulative, ${scope}</small>` : '<p style="opacity:.6;padding:20px">Chart fills in once graded bets accumulate.</p>'}</div>`;
}
// A P&L section with All · NFL · CFB chips; rows are stashed per section so a
// chip click redraws just that section's cards + chart (wirePnl).
function pnlSection(title, sub, rows, markets, s = getSettings()) {
  rows = rows || [];
  window.__caPnl = window.__caPnl || [];
  const id = window.__caPnl.push({ rows, markets, s }) - 1;
  const chips = [["all", "All"], ["nfl", "NFL"], ["cfb", "CFB"]]
    .map(([k, l]) => `<button data-pnlsport="${k}" class="${k === "all" ? "selected" : ""}">${l}</button>`).join("");
  return `<section class="section" data-pnl="${id}"><div class="section-title"><div><h2>${title}</h2><p>${sub}</p></div></div>
    <div class="pg-weeks pnl-filter">${chips}</div><div class="pnl-body">${pnlBody(rows, markets, "all", s)}</div></section>`;
}
function wirePnl() {
  document.querySelectorAll("[data-pnl]").forEach((sec) => {
    const cfg = (window.__caPnl || [])[+sec.dataset.pnl]; if (!cfg) return;
    sec.querySelector(".pnl-filter").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-pnlsport]"); if (!b) return;
      sec.querySelectorAll(".pnl-filter button").forEach((x) => x.classList.toggle("selected", x === b));
      sec.querySelector(".pnl-body").innerHTML = pnlBody(cfg.rows, cfg.markets, b.dataset.pnlsport, cfg.s);
    });
  });
}

// NFL props by game: rollup rows from nfl_prop_pnl_by_game, grouped by week;
// a game's props load on demand (nfl_prop_pnl?game_pk=eq.X) when expanded.
function propGamesSection(games, s = getSettings()) {
  games = games || [];
  const k = unitScale(s);
  const head = `<div class="section-title"><div><h2>Props by game <span style="font-size:.6em;opacity:.6">NFL · sim · ${unitLabel(s)}/bet</span></h2><p>Every graded prop line per game — the sim's lean bet at the leaned side's best price, plus any +EV prop picks. Tap a game for its props.</p></div></div>`;
  if (!games.length)
    return `<section class="section">${head}<div class="ev-empty"><b>No graded prop games yet — fills in after games with captured lines settle.</b></div></section>`;
  const tot = games.reduce((a, g) => ({ w: a.w + +g.hits, l: a.l + +g.misses, pnl: a.pnl + (+g.pnl || 0), ew: a.ew + +g.ev_wins, el: a.el + +g.ev_losses, epnl: a.epnl + (+g.ev_pnl || 0) }), { w: 0, l: 0, pnl: 0, ew: 0, el: 0, epnl: 0 });
  const cards = stat("SIM LEANS P&L", money(tot.pnl * k), `${tot.w}-${tot.l} · every graded prop line`, pnlCls(tot.pnl))
    + stat("+EV PROPS P&L", tot.ew + tot.el ? money(tot.epnl * k) : "—", tot.ew + tot.el ? `${tot.ew}-${tot.el} · +EV prop picks` : "no graded +EV props yet", pnlCls(tot.epnl));
  const weeks = [...new Set(games.map((g) => `${g.season}-${g.week}`))]
    .sort((a, b) => { const [sa, wa] = a.split("-").map(Number), [sb, wb] = b.split("-").map(Number); return sa - sb || wa - wb; });
  window.__caPropGames = games;
  window.__caPropS = s;   // the week chips + game rows redraw at this page build's unit
  const latest = weeks.at(-1);
  const btns = weeks.map((k) => `<button class="${k === latest ? "selected" : ""}" data-pgweek="${k}">Week ${k.split("-")[1]}</button>`).join("");
  return `<section class="section">${head}<div class="compact-stats pnl-stats">${cards}</div><div class="pg-weeks">${btns}</div><div class="pg-body">${propGamesWeek(latest, s)}</div></section>`;
}
function propGamesWeek(key, s = getSettings()) {
  const k = unitScale(s);
  const games = (window.__caPropGames || []).filter((g) => `${g.season}-${g.week}` === key)
    .sort((a, b) => (a.commence_time > b.commence_time ? 1 : -1));
  if (!games.length) return `<p style="opacity:.6;padding:16px">No graded prop games this week.</p>`;
  return games.map((g) => {
    const ev = +g.ev_n ? ` · +EV ${g.ev_wins}-${g.ev_losses} <b class="${pnlCls(+g.ev_pnl)}">${money((+g.ev_pnl || 0) * k)}</b>` : "";
    return `<div class="pg-game" data-game="${g.game_pk}"><button class="pg-head"><span><b>${g.matchup || `Game ${g.game_pk}`}</b><small>${g.n} props · leans ${g.hits}-${g.misses}${+g.pushes ? `-${g.pushes}` : ""}${ev}</small></span><strong class="${pnlCls(+g.pnl)}">${money((+g.pnl || 0) * k)}</strong></button><div class="pg-detail" hidden></div></div>`;
  }).join("");
}
function propGameDetail(rows, s = getSettings()) {
  const k = unitScale(s);
  if (!rows.length) return `<p style="opacity:.6;padding:12px">No graded props for this game.</p>`;
  const fmtAm = (p) => (p == null ? "—" : p > 0 ? `+${p}` : `${p}`);
  const num = (x) => (x == null ? "—" : Math.round(x));   // yards & counts are whole numbers
  const badge = (res) => res === "hit" ? `<span class="win">✓</span>` : res === "miss" ? `<span class="loss">✗</span>` : `<span>P</span>`;
  const body = rows.map((r) => `<tr>
      <td><b>${r.player_name || "—"}</b><small>${r.team || ""}</small></td>
      <td>${propMktLabel(r.market)}${r.ev_side ? ` <span class="prop-ev">+EV ${r.ev_side === "over" ? "o" : "u"}</span>` : ""}</td>
      <td>${r.line != null ? (+r.line).toFixed(1) : "—"}</td>
      <td>${r.projection != null ? (+r.projection).toFixed(1) : "—"}</td>
      <td class="pick">${r.lean === "over" ? "Over" : "Under"} ${fmtAm(r.bet_price)}</td>
      <td>${num(r.actual)}</td>
      <td>${badge(r.result)}</td>
      <td class="${pnlCls(+r.pnl)}">${r.pnl == null ? "—" : money(+r.pnl * k)}</td></tr>`).join("");
  return `<div class="table-wrap"><table class="ev-table"><thead><tr><th>PLAYER</th><th>PROP</th><th>LINE</th><th>SIM</th><th>LEAN @ PRICE</th><th>ACTUAL</th><th>RESULT</th><th>P&amp;L</th></tr></thead><tbody>${body}</tbody></table></div>`;
}
function wirePropGames() {
  // The props week row lives in the same section as .pg-body; the P&L trackers'
  // sport chips also carry .pg-weeks (styling), so don't match document-wide.
  const body = document.querySelector(".pg-body");
  const wk = body && body.closest(".section")?.querySelector(".pg-weeks:not(.pnl-filter)");
  if (!wk || !body) return;
  const s = window.__caPropS || getSettings();
  wk.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-pgweek]"); if (!b) return;
    wk.querySelectorAll("button").forEach((x) => x.classList.toggle("selected", x === b));
    body.innerHTML = propGamesWeek(b.dataset.pgweek, s);
  });
  body.addEventListener("click", async (e) => {
    const h = e.target.closest(".pg-head"); if (!h) return;
    const box = h.parentElement, detail = box.querySelector(".pg-detail");
    detail.hidden = !detail.hidden;
    box.classList.toggle("open", !detail.hidden);
    if (!detail.hidden && !detail.dataset.loaded) {
      detail.innerHTML = `<p style="opacity:.6;padding:12px">Loading…</p>`;
      const rows = await sb(`nfl_prop_pnl?game_pk=eq.${box.dataset.game}&order=market.asc,player_name.asc`).catch(() => []);
      detail.innerHTML = propGameDetail(rows, s);
      detail.dataset.loaded = "1";
    }
  });
}

// accuracy summary from graded prediction rows
function accSummary(graded) {
  const dec = graded.filter((g) => g.actual_winner != null);
  const correct = dec.filter((g) => g.winner_correct).length;
  return {
    n: dec.length,
    acc: dec.length ? `${(correct / dec.length * 100).toFixed(1)}%` : "—",
    record: `${correct}-${dec.length - correct}`,
    mae: dec.length ? (dec.reduce((s, g) => s + (+g.margin_error || 0), 0) / dec.length).toFixed(1) : "—",
  };
}

const predTable = (rows) => rows.length
  ? `<div class="table-wrap"><table><thead><tr><th>MATCHUP</th><th>DATE</th><th>SPREAD</th><th>TOTAL</th><th>MONEYLINE</th><th>PROJECTED</th><th>CONF</th></tr></thead><tbody>${rows.map((r) => `<tr><td><b>${logoPair(matchupOf(r), r.sport)}${matchupOf(r)}</b><small>lean vs the line · model number below</small></td><td>${(r.game_date || "").slice(5)}${r.commence_time ? `<small>${timeET(r.commence_time)}</small>` : ""}</td><td class="pick">${spreadLean(r)}<small>model ${homeSpread(r)}</small></td><td class="pick">${totalLean(r)}<small>model ${modelTotal(r)}</small></td><td>${homeML(r)}</td><td class="proj">${projScoreLogos(r)}</td><td><b>${confPct(r.home_win_prob)}</b></td></tr>`).join("")}</tbody></table></div>`
  : `<div class="table-wrap"><p style="padding:24px;opacity:.6">No predictions posted yet — they publish ahead of each week's games.</p></div>`;

const confCard = (r) => `<article class="edge-card"><div class="matchup"><div><h3>${logoPair(matchupOf(r), r.sport)}${matchupOf(r)}</h3><p>${r.sport.toUpperCase()} · ${(r.game_date || "").slice(5)}</p></div><span>${confPct(r.home_win_prob)}</span></div><div class="market"><small>PREDICTED WINNER · ${projScore(r)}</small><b>${predWinner(r)}</b><strong>${homeSpread(r)} · O/U ${modelTotal(r)}</strong><i style="width:${(conf(r.home_win_prob) * 100).toFixed(0)}%"></i><p>model spread & total · ML ${homeML(r)} (home)</p></div></article>`;

async function buildDash() {
  // Simplified landing: hero + clickable game lists per league. No predictions
  // or edges here -- those live on each game's detail page (game.html).
  const [nflP, cfbP, mls] = await Promise.all([predictions("nfl"), predictions("cfb"), gameMoneylines()]);
  const s = getSettings();
  return `<main><section class="hero"><div class="grid-surface"></div><div class="hero-content"><p class="eyebrow">LIVE PREDICTIONS</p><h1>Predict the game, <em>not the market.</em></h1><p class="hero-copy">Tap any game for its predicted score, predictions, +EV picks and projected player props.</p><div class="actions"><a class="button primary" href="nfl.html">NFL games</a><a class="button secondary" href="cfb.html">CFB games</a></div></div></section>
    <section class="section"><div class="section-title"><div><h2>NFL games</h2></div><a href="nfl.html">All NFL →</a></div>${gameList(nflP, "nfl", "No upcoming NFL games right now.", mls, s)}</section>
    <section class="section"><div class="section-title"><div><h2>CFB games</h2></div><a href="cfb.html">All CFB →</a></div>${gameList(cfbP, "cfb", "No upcoming CFB games right now.", mls, s)}</section></main>`;
}

/* ── Game detail page (game.html?sport=<s>&game=<game_pk>) ─────────────── */
function gameParams() {
  const p = new URLSearchParams(location.search);
  return { sport: (p.get("sport") || "nfl").toLowerCase(), game: p.get("game") };
}

// ==== Simulation distribution charts (Apple/MVPEAV-style) ====================
// A stored pmf is {kind:"pmf",pmf:[...]} (value at index i is i) or
// {kind:"margin",offset,pmf} (value is i-offset). These helpers read/plot
// either shape and compute Under/At/Over at any line.
function distParse(d) {
  if (d == null) return null;
  if (typeof d !== "string") return d;
  try { return JSON.parse(d); } catch { return null; }
}
function distValueAt(dist, i) { return dist && dist.kind === "margin" ? i - dist.offset : i; }
function distPairs(dist) {
  const pmf = (dist && dist.pmf) || [];
  return pmf.map((p, i) => [distValueAt(dist, i), p]);
}
function distMedian(dist) {
  const pairs = distPairs(dist);
  let c = 0;
  for (const [v, p] of pairs) { c += p; if (c >= 0.5) return v; }
  return pairs.length ? pairs[pairs.length - 1][0] : 0;
}
function distProbs(dist, line) {
  let under = 0, at = 0, over = 0;
  for (const [v, p] of distPairs(dist)) {
    if (v < line) under += p; else if (v > line) over += p; else at += p;
  }
  const s = under + at + over || 1;
  return { under: under / s, at: at / s, over: over / s };
}
// Group the fine per-point pmf into ~targetBars display buckets.
function distBins(dist, targetBars = 26) {
  const pairs = distPairs(dist).filter(([, p]) => p > 0);
  if (!pairs.length) return [];
  const lo = pairs[0][0], hi = pairs[pairs.length - 1][0];
  const span = hi - lo + 1;
  const step = Math.max(1, Math.ceil(span / targetBars));
  const bins = [];
  for (let start = lo; start <= hi; start += step) {
    const end = start + step - 1;
    let f = 0;
    for (const [v, p] of distPairs(dist)) if (v >= start && v <= end) f += p;
    bins.push({ lo: start, hi: end, mid: (start + end) / 2, label: step === 1 ? `${start}` : `${start}–${end}`, freq: f });
  }
  return bins;
}
function histogramSVG(dist, opts = {}) {
  const { line = null, height = 150 } = opts;
  const accent = opts.accent || DEFAULT_ACCENT;
  const colorFn = opts.colorFn || (() => accent);
  const bins = distBins(dist);
  if (!bins.length) return `<div class="ev-empty" style="padding:16px">No distribution for this selection.</div>`;
  // Render at the container's real pixel width so the 1:1 viewBox mapping keeps
  // text crisp and in-font — preserveAspectRatio="none" against a fixed 700-wide
  // box stretched the labels out of the site's Inter face.
  const W = Math.max(220, Math.round(opts.width || 700)), H = height, padB = 24, padL = 4, padR = 4;
  const maxF = Math.max(...bins.map((b) => b.freq)) || 1;
  const bw = (W - padL - padR) / bins.length;
  const bars = bins.map((b, i) => {
    const h = (b.freq / maxF) * (H - padB - 8);
    const x = padL + i * bw, y = H - padB - h;
    return `<rect x="${(x + bw * 0.1).toFixed(1)}" y="${y.toFixed(1)}" width="${(bw * 0.8).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" rx="1.5" fill="${colorFn(b.mid)}" opacity="0.92"></rect>`;
  }).join("");
  const nLab = Math.min(7, bins.length);
  const labs = [];
  for (let k = 0; k < nLab; k++) {
    const i = Math.round(k * (bins.length - 1) / (nLab - 1 || 1));
    const x = padL + i * bw + bw / 2;
    labs.push(`<text x="${x.toFixed(1)}" y="${H - 7}" text-anchor="middle" font-size="10" fill="var(--muted)">${bins[i].label}</text>`);
  }
  let marker = "";
  if (line != null) {
    const lo = bins[0].lo, hi = bins[bins.length - 1].hi + 1;
    const frac = Math.max(0, Math.min(1, (line - lo) / (hi - lo || 1)));
    const x = padL + frac * (W - padL - padR);
    marker = `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="4" y2="${H - padB}" stroke="var(--amber)" stroke-width="2" stroke-dasharray="4 3"></line><text x="${x.toFixed(1)}" y="14" text-anchor="middle" font-size="10" fill="var(--amber)">Line ${+line}</text>`;
  }
  return `<svg class="hist" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%;height:${H}px">${bars}${marker}${labs.join("")}</svg>`;
}
// Bar coloring for a game-sim tab: Spread splits at pick'em (away color for the
// away-ahead/left side, home color for home-ahead/right — matching the hero
// board's away-left / home-right); each team-total tab is that team's color;
// Total (team-agnostic) keeps the under/over split at the line.
function gameChartColorFn(tab, line, awayCol, homeCol) {
  if (tab === "away") return () => awayCol;
  if (tab === "home") return () => homeCol;
  if (tab === "margin") return (mid) => (mid < 0 ? awayCol : homeCol);
  return (mid) => (line != null && mid >= line ? "#e4564e" : "#5b8fce"); // total
}
function probVsLineRow(dist, line) {
  if (line == null || line === "") return `<p class="prob-line"><span style="opacity:.6">Set a line to see Under / At / Over.</span></p>`;
  const { under, at, over } = distProbs(dist, +line);
  const pct = (x) => `${(x * 100).toFixed(1)}%`;
  return `<p class="prob-line"><b>Probability vs Line</b> · <span class="under">Under ${pct(under)}</span> · <span class="at">At ${pct(at)}</span> · <span class="over">Over ${pct(over)}</span></p>`;
}

// The game-level simulation panel: median team totals + a tabbed, interactive
// Spread / Total / team-total histogram. Returns "" (graceful) when the sim row
// predates the distribution migration.
function gameSimVisual(sim, away, home, awayCol, homeCol) {
  if (!sim) return "";
  const dists = {
    margin: distParse(sim.margin_dist), total: distParse(sim.total_dist),
    away: distParse(sim.away_score_dist), home: distParse(sim.home_score_dist),
  };
  if (!dists.margin || !dists.total) return "";
  const awayMed = distMedian(dists.away), homeMed = distMedian(dists.home);
  window.__caGameSim = Object.assign(window.__caGameSim || {}, { dists, awayCol, homeCol });
  const tabs = [["margin", "Spread"], ["total", "Total"], ["away", `${ctxEsc(away)} total`], ["home", `${ctxEsc(home)} total`]];
  const defLine = distMedian(dists.margin);
  const teamTotals = `<div class="team-totals"><div class="tt"><span class="tt-name">${ctxEsc(away)}</span><span class="tt-score" style="color:${awayCol}">${awayMed}</span></div><span class="tt-vs">sim median</span><div class="tt"><span class="tt-score" style="color:${homeCol}">${homeMed}</span><span class="tt-name">${ctxEsc(home)}</span></div></div>`;
  const tabBar = `<div class="sim-tabs">${tabs.map(([k, l], i) => `<button class="sim-tab${i === 0 ? " active" : ""}" data-dist="${k}">${l}</button>`).join("")}</div>`;
  const controls = `<div class="sim-controls">Line <input id="sim-line" type="number" step="0.5" value="${defLine}" inputmode="decimal"></div>`;
  const chart = `<div id="sim-chart">${histogramSVG(dists.margin, { line: defLine, colorFn: gameChartColorFn("margin", defLine, awayCol, homeCol) })}</div>`;
  const prob = `<div id="sim-prob">${probVsLineRow(dists.margin, defLine)}</div>`;
  return `<section class="section sim-visual"><div class="section-title"><div><h2>Simulation spread &amp; totals <span class="sim-tag">NFL · sim</span></h2><p>Distribution of the simulated games — pick a tab, set a line for Under/At/Over.</p></div></div>${teamTotals}${tabBar}${controls}${chart}${prob}</section>`;
}

// Median boxscore by player (from the sim means), grouped per team into QBs /
// Rushers / Receivers. Rows are clickable -> per-player distribution.
function boxscoreSection(sims, r) {
  const players = window.__caGameSim && window.__caGameSim.players;
  if (!players || !Object.keys(players).length) return "";
  const mean = (id, mk) => { const d = players[id] && players[id].markets[mk]; return d == null ? null : (d.__mean != null ? d.__mean : distMean(d)); };
  const num = (x, dec = 0) => x == null ? "—" : (dec ? x.toFixed(dec) : Math.round(x));
  const atd = (id) => { const d = players[id] && players[id].markets.anytime_td; if (!d || !d.pmf || d.pmf.length < 2) return "—"; return `${(d.pmf[1] * 100).toFixed(0)}%`; };
  const clk = (id, mk) => `data-player-id="${id}" data-market="${mk}"`;
  const teamPlayers = (teamName) => Object.entries(players).filter(([, p]) => p.team === teamName).map(([id, p]) => ({ id, ...p }));
  const qbTable = (list) => {
    const qbs = list.filter((p) => p.pos === "QB" && mean(p.id, "pass_yds")).sort((a, b) => mean(b.id, "pass_yds") - mean(a.id, "pass_yds"));
    if (!qbs.length) return "";
    return `<table class="ev-table box"><thead><tr><th>QB</th><th>Pass Yds</th><th>Pass TD</th><th>Rush Yds</th></tr></thead><tbody>${qbs.map((p) => `<tr class="box-row" ${clk(p.id, "pass_yds")}><td>${ctxEsc(p.name)}</td><td>${num(mean(p.id, "pass_yds"))}</td><td>${num(mean(p.id, "pass_tds"), 1)}</td><td>${num(mean(p.id, "rush_yds"))}</td></tr>`).join("")}</tbody></table>`;
  };
  const rushTable = (list) => {
    const rb = list.filter((p) => (mean(p.id, "rush_yds") || 0) >= 5 && p.pos !== "QB").sort((a, b) => mean(b.id, "rush_yds") - mean(a.id, "rush_yds")).slice(0, 6);
    if (!rb.length) return "";
    return `<table class="ev-table box"><thead><tr><th>Rusher</th><th>Rush Yds</th><th>Any TD</th></tr></thead><tbody>${rb.map((p) => `<tr class="box-row" ${clk(p.id, "rush_yds")}><td>${ctxEsc(p.name)}</td><td>${num(mean(p.id, "rush_yds"))}</td><td>${atd(p.id)}</td></tr>`).join("")}</tbody></table>`;
  };
  const recTable = (list) => {
    const wr = list.filter((p) => (mean(p.id, "rec_yds") || 0) >= 5 && p.pos !== "QB").sort((a, b) => mean(b.id, "rec_yds") - mean(a.id, "rec_yds")).slice(0, 7);
    if (!wr.length) return "";
    return `<table class="ev-table box"><thead><tr><th>Receiver</th><th>Rec Yds</th><th>Rec</th><th>Any TD</th></tr></thead><tbody>${wr.map((p) => `<tr class="box-row" ${clk(p.id, "rec_yds")}><td>${ctxEsc(p.name)}</td><td>${num(mean(p.id, "rec_yds"))}</td><td>${num(mean(p.id, "receptions"), 1)}</td><td>${atd(p.id)}</td></tr>`).join("")}</tbody></table>`;
  };
  const col = (teamName) => { const list = teamPlayers(teamName); return `<div class="box-col"><h3>${ctxEsc(teamName)}</h3>${qbTable(list)}${rushTable(list)}${recTable(list)}</div>`; };
  return `<section class="section boxscore"><div class="section-title"><div><h2>Boxscore <span class="sim-tag">median by player</span></h2><p>Sim median stat line · tap a player for their full distribution.</p></div></div><div class="box-grid">${col(r.away_team_name)}${col(r.home_team_name)}</div><div id="player-dist"></div></section>`;
}
function distMean(dist) {
  let m = 0;
  for (const [v, p] of distPairs(dist)) m += v * p;
  return m;
}

// Wire the interactive game-sim charts after render() sets innerHTML: tab
// switching + line inputs for the game chart, and click-to-open per-player
// distribution. All state lives on window.__caGameSim (set by buildGame).
function wireGameSim() {
  const view = document.querySelector(".game-view");
  if (!view) return;
  const g = window.__caGameSim || {};
  const PD_ACCENT = "#8b5cf6";
  const MK = { pass_yds: "Pass Yds", pass_tds: "Pass TD", rush_yds: "Rush Yds", rec_yds: "Rec Yds", receptions: "Receptions", anytime_td: "Any TD" };
  const widthOf = (el) => (el && el.clientWidth) ? el.clientWidth : 700;

  const redrawGame = () => {
    if (!g.dists) return;
    const active = view.querySelector(".sim-tab.active")?.dataset.dist || "margin";
    const dist = g.dists[active];
    const lineEl = view.querySelector("#sim-line");
    const line = lineEl && lineEl.value !== "" ? +lineEl.value : null;
    const chart = view.querySelector("#sim-chart"), prob = view.querySelector("#sim-prob");
    if (chart) chart.innerHTML = histogramSVG(dist, { line, width: widthOf(chart), colorFn: gameChartColorFn(active, line, g.awayCol, g.homeCol) });
    if (prob) prob.innerHTML = probVsLineRow(dist, line);
  };

  const renderPlayer = (id, mk) => {
    const pl = g.players && g.players[id]; if (!pl) return;
    const markets = Object.keys(pl.markets);
    if (!markets.length) return;
    mk = mk && pl.markets[mk] ? mk : markets[0];
    g.activePlayer = id;
    const dist = pl.markets[mk];
    const line = distMedian(dist);
    const opts = markets.map((k) => `<option value="${k}"${k === mk ? " selected" : ""}>${MK[k] || k}</option>`).join("");
    const panel = view.querySelector("#player-dist");
    if (!panel) return;
    panel.innerHTML = `<div class="player-dist-inner"><div class="section-title"><div><h2>${ctxEsc(pl.name)} — ${ctxEsc(pl.team)} · ${ctxEsc(pl.pos)}</h2></div><button class="pd-close" data-close-pd>Close</button></div><div class="sim-controls">Stat <select id="pd-stat" data-cur="${mk}">${opts}</select> Line <input id="pd-line" type="number" step="0.5" value="${line}" inputmode="decimal"></div><div id="pd-chart">${histogramSVG(dist, { line, accent: playerColor(pl), width: widthOf(panel) })}</div><div id="pd-prob">${probVsLineRow(dist, line)}</div></div>`;
  };
  const redrawPlayer = () => {
    const pl = g.players && g.players[g.activePlayer]; if (!pl) return;
    const statEl = view.querySelector("#pd-stat"), lineEl = view.querySelector("#pd-line");
    if (!statEl || !lineEl) return;
    const mk = statEl.value, dist = pl.markets[mk];
    if (statEl.dataset.cur !== mk) { lineEl.value = distMedian(dist); statEl.dataset.cur = mk; }
    const line = lineEl.value !== "" ? +lineEl.value : null;
    const chartEl = view.querySelector("#pd-chart");
    chartEl.innerHTML = histogramSVG(dist, { line, accent: playerColor(pl), width: widthOf(chartEl) });
    view.querySelector("#pd-prob").innerHTML = probVsLineRow(dist, line);
  };
  // A player's bars take their own team's hero color (away or home).
  const playerColor = (pl) => pl && pl.team === g.homeTeam ? g.homeCol : (pl && pl.team === g.awayTeam ? g.awayCol : PD_ACCENT);

  view.addEventListener("click", (e) => {
    const tab = e.target.closest(".sim-tab");
    if (tab && view.contains(tab)) {
      view.querySelectorAll(".sim-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const dist = g.dists && g.dists[tab.dataset.dist];
      const lineEl = view.querySelector("#sim-line");
      if (lineEl && dist) lineEl.value = distMedian(dist);
      redrawGame();
      return;
    }
    if (e.target.closest("[data-close-pd]")) { const p = view.querySelector("#player-dist"); if (p) p.innerHTML = ""; g.activePlayer = null; return; }
    const row = e.target.closest("[data-player-id]");
    if (row && view.contains(row)) {
      renderPlayer(row.dataset.playerId, row.dataset.market);
      view.querySelector("#player-dist")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
  view.addEventListener("input", (e) => {
    if (e.target.id === "sim-line") redrawGame();
    else if (e.target.id === "pd-stat" || e.target.id === "pd-line") redrawPlayer();
  });

  redrawGame(); // re-render at the container's real width so text stays crisp/in-font
  // Re-render on resize so the 1:1 width mapping (and thus the fonts) stay right.
  window.__caRedrawSim = () => { redrawGame(); if (g.activePlayer) redrawPlayer(); };
  if (!window.__caResizeHooked) {
    window.__caResizeHooked = true;
    let rt;
    window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => window.__caRedrawSim && window.__caRedrawSim(), 150); });
  }
}

// The sim's projected stat line per featured player for one game, +EV flagged.
// Once the game is captured (nfl_player_actuals), each cell also shows the
// player's ACTUAL stat beneath the projection with a ✓/✗ (met/beat our number).
function propsProjectionSection(sims, props, actuals, lines, modelTag = "") {
  const norm = (s) => String(s || "").toLowerCase().replace(/[.’']/g, "").replace(/\s+(jr|sr|ii|iii|iv|v)$/, "").replace(/\s+/g, " ").trim();
  const pickBy = new Map();
  (props || []).filter((p) => p.is_pick).forEach((p) => pickBy.set(`${norm(p.player_name)}|${p.market}`, p));
  // Actuals keyed by player_id|market (same gsis_id namespace as the sim rows).
  const actualBy = new Map();
  (actuals || []).forEach((a) => actualBy.set(`${a.player_id}|${a.market}`, a.actual));
  const hasActuals = actualBy.size > 0;
  // Captured pre-kickoff prop lines (nfl_prop_lines), keyed the same way, so a
  // cell can show the book's number + best price alongside the sim's own.
  const lineBy = new Map();
  (lines || []).forEach((l) => lineBy.set(`${l.player_id}|${l.market}`, l));
  const hasLines = lineBy.size > 0;
  const fmtAm = (p) => (p == null ? "—" : p > 0 ? `+${p}` : `${p}`);
  const byPlayer = new Map();
  (sims || []).forEach((s) => {
    if (!byPlayer.has(s.player_id)) byPlayer.set(s.player_id, { id: s.player_id, name: s.name, pos: s.pos, team: s.team, m: {}, atd: null });
    const p = byPlayer.get(s.player_id);
    p.m[s.market] = s.mean;
    if (s.market === "anytime_td") {          // P(scores >= 1) lives in the pmf, not the mean
      const d = typeof s.dist === "string" ? (() => { try { return JSON.parse(s.dist); } catch { return null; } })() : s.dist;
      p.atd = d && d.pmf && d.pmf.length > 1 ? d.pmf[1] : (s.mean != null ? Math.min(s.mean, 1) : null);
    }
  });
  const rel = (p) => Math.max(p.m.pass_yds || 0, (p.m.rush_yds || 0) * 3, (p.m.rec_yds || 0) * 3, (p.m.receptions || 0) * 10, (p.m.rush_att || 0) * 5, (p.atd || 0) * 80);
  const players = [...byPlayer.values()].filter((p) => rel(p) >= 30).sort((a, b) => rel(b) - rel(a));
  const head = `<div class="section-title"><div><h2>Projected player props <span class="sim-tag">NFL · sim</span> ${modelTag}</h2><p>Model's projected stat line per player · +EV vs a book line flagged</p></div></div>`;
  if (!players.length)
    return `<section class="section">${head}<div class="ev-empty"><b>No player projections for this game yet.</b><p>The sim writes player distributions on game days once current-season data is in.</p></div></section>`;
  // Actual stat beneath a projection, ✓/✗ against the captured book line when one
  // exists (hit = actual landed on the sim's leaned side; a push shows "P"),
  // else the old met/beat-the-projection fallback.
  const actualLine = (p, mk, proj, dec = 0, ln = null) => {
    if (!hasActuals) return "";
    const act = actualBy.get(`${p.id}|${mk}`);
    if (act == null) return "";
    // Count markets (receptions, rush attempts) are always whole numbers --
    // show the ACTUAL as an integer even when the projection prints with a
    // decimal (e.g. "17" not "17.0"). The ✓/✗ comparison logic below is
    // unaffected; only the displayed string changes.
    const isCount = mk === "receptions" || mk === "rush_att";
    const shown = isCount ? Math.round(act) : (dec ? (+act).toFixed(dec) : Math.round(act));
    if (ln && ln.line != null) {
      const a = +act, l = +ln.line;
      if (a === l) return `<span class="prop-actual">→ ${shown} P</span>`;
      const hit = ln.lean === "over" ? a > l : a < l;
      return `<span class="prop-actual ${hit ? "hit" : "miss"}">→ ${shown} ${hit ? "✓" : "✗"}</span>`;
    }
    // No projection (e.g. a market added after this game was simmed): show the
    // actual with no ✓/✗ -- there's nothing to have met.
    if (proj == null) return `<span class="prop-actual">→ ${shown}</span>`;
    // Compare at the SAME precision shown, so the ✓/✗ matches the printed
    // numbers (e.g. actual 20 vs a projection that displays as 20 reads as met).
    const round = (x) => dec ? +(+x).toFixed(dec) : Math.round(x);
    const hit = round(act) >= round(proj);
    return `<span class="prop-actual ${hit ? "hit" : "miss"}">→ ${shown} ${hit ? "✓" : "✗"}</span>`;
  };
  const cell = (p, mk, dec = 0) => {
    const proj = p.m[mk];
    const v = proj == null ? "—" : (dec ? proj.toFixed(dec) : Math.round(proj));
    const q = pickBy.get(`${norm(p.name)}|${mk}`);
    const flag = q ? ` <span class="prop-ev">${q.side === "over" ? "o" : "u"}${(+q.line).toFixed(1)} ${evSigned(q.ev_best)}</span>` : "";
    const ln = lineBy.get(`${p.id}|${mk}`);
    const lineHtml = ln ? (() => {
      const over = fmtAm(ln.over_price), under = fmtAm(ln.under_price);
      const overShown = ln.lean === "over" ? `<b>${over}</b>` : over;
      const underShown = ln.lean === "under" ? `<b>${under}</b>` : under;
      return `<span class="prop-line">o/u ${ln.line} · ${overShown}/${underShown}</span>`;
    })() : "";
    return `${v}${flag}${lineHtml}${actualLine(p, mk, proj, dec, ln)}`;
  };
  const anyTd = (p) => {
    const proj = p.atd == null ? "—" : `${(p.atd * 100).toFixed(0)}%`;
    if (!hasActuals) return proj;
    const act = actualBy.get(`${p.id}|anytime_td`);
    if (act == null) return proj;
    const scored = act >= 1;
    return `${proj}<span class="prop-actual ${scored ? "hit" : "miss"}">${scored ? "TD ✓" : "— ✗"}</span>`;
  };
  // Split by position. QB = each team's starter only (most projected passing
  // yards; backups dropped). FB/HB fold into RB. Other positions aren't shown.
  const posOf = (p) => { const x = String(p.pos || "").toUpperCase(); return x === "FB" || x === "HB" ? "RB" : x; };
  const starters = new Map();
  [...byPlayer.values()].filter((p) => posOf(p) === "QB").forEach((p) => {
    const cur = starters.get(p.team);
    if (!cur || (p.m.pass_yds || 0) > (cur.m.pass_yds || 0)) starters.set(p.team, p);
  });
  const qbs = [...starters.values()].sort((a, b) => (b.m.pass_yds || 0) - (a.m.pass_yds || 0));
  const atPos = (pos) => players.filter((p) => posOf(p) === pos);
  const QB_COLS = [["PASS YDS", (p) => cell(p, "pass_yds")], ["RUSH YDS", (p) => cell(p, "rush_yds")],
    ["PASS TD", (p) => cell(p, "pass_tds", 1)], ["RUSH TD", anyTd]];
  const SKILL_COLS = [["REC YDS", (p) => cell(p, "rec_yds")], ["REC'NS", (p) => cell(p, "receptions", 1)],
    ["RUSH YDS", (p) => cell(p, "rush_yds")], ["RUSH ATT", (p) => cell(p, "rush_att", 1)], ["TD", anyTd]];
  const group = (title, list, cols) => !list.length ? "" : `<div class="prop-group"><h3>${title}<small>${list.length}</small></h3><div class="table-wrap"><table class="ev-table prop-proj"><thead><tr><th>PLAYER</th>${cols.map(([h]) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${list.map((p) => `<tr><td><b>${ctxEsc(p.name)}</b><small>${ctxEsc(p.team || "")}</small></td>${cols.map(([, f]) => `<td>${f(p)}</td>`).join("")}</tr>`).join("")}</tbody></table></div></div>`;
  const groups = group("Quarterbacks", qbs, QB_COLS) + group("Running Backs", atPos("RB"), SKILL_COLS)
    + group("Wide Receivers", atPos("WR"), SKILL_COLS) + group("Tight Ends", atPos("TE"), SKILL_COLS);
  const note = (hasActuals
    ? `TD / RUSH TD = projected probability the player scores at least once (a QB's is his rushing TD) · <b>→ actual</b> shown beneath each projection.`
    : `TD / RUSH TD = projected probability the player scores at least once (a QB's is his rushing TD) · projection only.`)
    + (hasLines ? ` o/u = main book line · best over/under price, the sim's side in bold; ✓ = actual landed on the sim's side of the line (✓ = met the projection where no line was posted).` : "")
    + ` Green bubble = +EV vs the best book price.`;
  return `<section class="section">${head}${groups}<p class="sim-note">${note}</p></section>`;
}

// nfl_player_sim can hold rows from more than one sim run per game. Rows come
// back newest-first (order=created_at.desc), so the first row seen for each
// player+market is the latest projection; keep that one and drop older dupes.
function dedupLatest(rows) {
  const seen = new Set(), out = [];
  for (const r of (rows || [])) {
    const k = `${r.player_id}|${r.market}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(r);
  }
  return out;
}

// Public betting splits (share of TICKETS) for Moneyline / Spread / Total, from
// the latest capture (nfl_betting_splits_current). Each market renders a split
// bar tinted with the two teams' colors (Over/Under use neutral accents) plus
// the ticket% on each side. Returns "" when no splits are captured for the game
// yet (splits are captured near kickoff, so past/early games may have none).
function splitsSection(splits, r, awayCol, homeCol) {
  if (!splits || !splits.length) return "";
  const pick = (market, side) => {
    const row = splits.find((s) => s.market === market && s.side === side);
    return row && row.ticket_pct != null ? Math.round(row.ticket_pct) : null;
  };
  const bar = (label, lLbl, lPct, lCol, rLbl, rPct, rCol) => {
    if (lPct == null && rPct == null) return "";
    const lp = lPct == null ? (rPct == null ? 50 : 100 - rPct) : lPct;
    const rp = rPct == null ? 100 - lp : rPct;
    const t = (p) => (p == null ? "–" : p + "%");
    return `<div class="split-row">
      <div class="split-mkt">${label}</div>
      <div class="split-bar"><span style="width:${lp}%;background:${lCol}"></span><span style="width:${rp}%;background:${rCol}"></span></div>
      <div class="split-legend"><span><b style="color:${lCol}">${t(lPct)}</b> ${ctxEsc(lLbl)}</span><span>${ctxEsc(rLbl)} <b style="color:${rCol}">${t(rPct)}</b></span></div>
    </div>`;
  };
  const away = r.away_team_name, home = r.home_team_name;
  const OVER = "#38bdf8", UNDER = "#f59e0b";
  const ml = bar("Moneyline", away, pick("moneyline", "away"), awayCol, home, pick("moneyline", "home"), homeCol);
  const sp = bar("Spread", away, pick("spread", "away"), awayCol, home, pick("spread", "home"), homeCol);
  const tot = bar("Total", "Over", pick("total", "over"), OVER, "Under", pick("total", "under"), UNDER);
  if (!ml && !sp && !tot) return "";
  const cap = splits[0] && splits[0].captured_at ? ` · as of ${timeET(splits[0].captured_at)}` : "";
  return `<section class="section"><div class="section-title"><div><h2>Public Betting</h2><p>Share of tickets (bets) on each side${cap}</p></div></div>
    <style>
      .splits-wrap .split-row{margin:14px 0}
      .splits-wrap .split-mkt{font-weight:600;font-size:.82rem;letter-spacing:.02em;opacity:.85;margin-bottom:5px}
      .splits-wrap .split-bar{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--line)}
      .splits-wrap .split-bar span{display:block;height:100%}
      .splits-wrap .split-legend{display:flex;justify-content:space-between;font-size:.82rem;opacity:.9;margin-top:5px}
    </style>
    <div class="splits-wrap">${ml}${sp}${tot}</div></section>`;
}

// Betting trends: each team's newest-season Action Network records (ATS, ATS at
// this venue / in this role, last 5, O/U, units) + NFL situational trends. Role
// comes from the market HOME line (negative = home favored; pick'em/none -> no
// role row). `trend-hot` marks any ATS/O/U record where one side has >= 70% of
// >= 8 decided games. Descriptive only -- the model never reads these.
const trendHot = (a, b) => { const d = (+a || 0) + (+b || 0); return d >= 8 && Math.max(+a || 0, +b || 0) / d >= 0.7; };
function trendsSection(awayName, homeName, recs, sits, awayCol, homeCol, homeLine, sport) {
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const latest = {};  // rows arrive season.desc -> first row per team is the newest season
  (recs || []).forEach((x) => { if (!latest[x.team_name]) latest[x.team_name] = x; });
  const recsOf = (name) => {
    let v = latest[name] && latest[name].records;
    if (typeof v === "string") { try { v = JSON.parse(v); } catch (e) { v = null; } }
    return v && typeof v === "object" ? v : {};
  };
  const n0 = (x) => +x || 0;
  const rec3 = (a, b, p) => `${n0(a)}-${n0(b)}-${n0(p)}`;
  const share = (a, b) => (n0(a) + n0(b) ? Math.round((100 * n0(a)) / (n0(a) + n0(b))) + "%" : "—");
  const row = (label, val, hot) => `<li class="trend-row${hot ? " trend-hot" : ""}"><span>${label}</span><b>${val}</b></li>`;
  const seen = (a, b, p) => n0(a) + n0(b) + n0(p) > 0;  // drop all-null / 0-0-0 records
  const ats = (label, c) => c && seen(c.w, c.l, c.p) && row(label, `${rec3(c.w, c.l, c.p)} · ${share(c.w, c.l)}`, trendHot(c.w, c.l));
  const ou = (label, c) => c && seen(c.o, c.u, c.p) && row(label, `<span title="overs-unders-pushes">${rec3(c.o, c.u, c.p)}</span> · ${n0(c.o) >= n0(c.u) ? share(c.o, c.u) + " O" : share(c.u, c.o) + " U"}`, trendHot(c.o, c.u));
  const line = homeLine == null ? 0 : +homeLine || 0;
  const col = (name, color, isHome) => {
    const R = recsOf(name);
    const venue = isHome ? "home" : "road";
    const fav = line === 0 ? null : (line < 0) === isHome;  // home line < 0 -> home favored
    // Action Network stores units lost as a negative number (0-2 SU -> l = -2): net = w − |l|.
    const u = R.units && R.units.w != null ? Math.round((n0(R.units.w) - Math.abs(n0(R.units.l))) * 10) / 10 : null;
    const rows = [
      ats("ATS", R.ats),
      ats(isHome ? "ATS at home" : "ATS on road", R[`ats_${venue}`]),
      fav == null ? "" : ats(fav ? "ATS as favorite" : "ATS as underdog", R[fav ? "ats_fav" : "ats_dog"]),
      ats("ATS last 5", R.ats_last_5),
      ou("O/U", R.over_under),
      ou(isHome ? "O/U at home" : "O/U on road", R[`over_under_${venue}`]),
      u == null ? "" : row("Units", `<span class="${u >= 0 ? "trend-pos" : "trend-neg"}">${uStr(u)}</span>`),
    ].filter(Boolean).join("");
    const mine = (sits || []).filter((s) => s.team_name === name).sort((a, b) => n0(b.n) - n0(a.n));
    const sitHtml = mine.map((s) =>
      `<li class="trend-sit${trendHot(s.ats_w, s.ats_l) || trendHot(s.ou_o, s.ou_u) ? " trend-hot" : ""}">${rec3(s.ats_w, s.ats_l, s.ats_p)} ATS · O/U ${rec3(s.ou_o, s.ou_u, s.ou_p)} ${esc(s.label)} since ${esc(s.since_season)}</li>`).join("");
    const body = rows || sitHtml
      ? `${rows ? `<ul class="trend-list">${rows}</ul>` : ""}${sitHtml ? `<ul class="trend-sits">${sitHtml}</ul>` : ""}`
      : `<p class="trend-none">No betting trends yet.</p>`;
    return { has: !!(rows || sitHtml), html: `<div class="trend-col" style="--tc:${color}"><h3>${logoImg(name, sport)}${esc(name)}</h3>${body}</div>` };
  };
  const a = col(awayName, awayCol, false), h = col(homeName, homeCol, true);
  const head = `<div class="section-title"><div><h2>Trends</h2><p>Latest-season ATS, over/under and units records</p></div></div>`;
  if (!a.has && !h.has) return `<section class="section trends-sec">${head}<p class="trend-empty">No betting trends for this matchup yet.</p></section>`;
  return `<section class="section trends-sec">${head}<div class="trends-grid">${a.html}${h.html}</div>
    <p class="trend-foot">Trends are descriptive, not predictive — the model doesn't use them. Records via Action Network; situational trends computed from nflverse (current season + last 3, ≥5 games).</p></section>`;
}

/* ── Team context: matchup grades, team history, power rankings ─────────────
   Written daily by the model repo's team-context job (tables team_history,
   matchup_grades, power_rankings / power_rankings_current). Everything here is
   DESCRIPTIVE — not a pick. Team ids: NFL = nflverse franchise code (KC, LA,
   WAS, ...); CFB = ESPN team id as a string ("333"); non-FBS CFB opponents are
   the pooled "FCS" team (no history, no grades, never ranked). Every read is
   caught -> [] so a missing table/view just hides the section. */
const ctxEsc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const ctxJson = (v) => {
  if (typeof v === "string") { try { v = JSON.parse(v); } catch (e) { v = null; } }
  return v && typeof v === "object" ? v : null;
};
const ctxNum = (x) => (x == null || x === "" || !Number.isFinite(+x) ? null : +x);
const ctxSigned = (x, d = 1) => { const n = ctxNum(x); return n == null ? "—" : `${n > 0 ? "+" : n < 0 ? "−" : ""}${Math.abs(n).toFixed(d)}`; };
const ctxOrd = (n) => { const v = Math.round(n), t = v % 100; return `${v}${t >= 11 && t <= 13 ? "th" : ["th", "st", "nd", "rd"][v % 10] || "th"}`; };
const CTX_NOT_A_PICK = `<em class="ctx-np">descriptive — not a pick</em>`;
const CTX_EARLY = `<span class="ctx-early" title="Either team has played fewer than 3 games this season — the ratings still lean on last season">early</span>`;

// NFL franchise code -> ESPN displayName (via the ESPN slug maps above); CFB ESPN
// id -> displayName (CFB_TEAM_ID inverted). Unknown ids fall back to the id.
const NFL_CODE_TO_NAME = (() => {
  const bySlug = {};
  for (const nm in NFL_TEAM_ABBR) bySlug[NFL_TEAM_ABBR[nm]] = nm;
  const out = {};
  for (const code in NFL_CODE_TO_ESPN) if (bySlug[NFL_CODE_TO_ESPN[code]]) out[code] = bySlug[NFL_CODE_TO_ESPN[code]];
  return out;
})();
const CFB_ID_TO_NAME = (() => { const out = {}; for (const nm in CFB_TEAM_ID) out[CFB_TEAM_ID[nm]] = nm; return out; })();
function ctxTeamName(team, sport) {
  const t = String(team ?? "");
  if (sport === "nfl") return NFL_CODE_TO_NAME[t.toUpperCase()] || t;
  if (sport === "cfb") return CFB_ID_TO_NAME[t] || (t === "FCS" ? "FCS" : `Team ${t}`);
  return t;
}
function ctxLogo(team, sport) {   // by id (CFB ids missing from CFB_TEAM_ID still get a logo)
  if (sport === "cfb" && /^\d+$/.test(String(team ?? "")))
    return `<img class="tlogo" src="https://a.espncdn.com/i/teamlogos/ncaa/500/${team}.png" alt="" loading="lazy" onerror="this.style.display='none'">`;
  return logoImg(team, sport);
}

// Power rating: points vs an average team on a neutral field; weekly move.
const rkRating = (x) => (ctxNum(x) == null ? "—" : `${ctxSigned(x)} vs avg`);
// ▲n moved up / ▼n down / – unchanged; NEW when prev_rank is null (only when the
// table has a previous week at all -- the first ranked week shows –, not NEW for all).
function rkMove(row, hasPrev = true) {
  if (row.prev_rank == null) return hasPrev ? `<span class="rk-new">NEW</span>` : `<span class="rk-flat">–</span>`;
  const m = ctxNum(row.move) ?? (ctxNum(row.prev_rank) - ctxNum(row.rank));
  if (!m) return `<span class="rk-flat">–</span>`;
  return m > 0 ? `<span class="rk-up">▲${m}</span>` : `<span class="rk-down">▼${-m}</span>`;
}
const rkPowerLine = (p, hasPrev = true) => p
  ? `<span class="ctx-power" title="Power rating weighted to this season">#${ctxEsc(p.rank)} power · ${rkRating(p.rating)} ${rkMove(p, hasPrev)}</span>` : "";

// A–F chip (null / anything else -> an ungraded dash).
function gradeChip(letter, big = false) {
  const L = typeof letter === "string" && /^[A-F]$/.test(letter) && letter !== "E" ? letter : null;
  return `<b class="grade-chip${big ? " big" : ""} g-${L || "na"}"${L ? "" : ' title="not graded"'}>${L || "–"}</b>`;
}

// Matchup: each OFFENSE vs the other side's defense (away card first, like the hero).
// Hidden when no side carries a letter (no rows, or both ungraded).
function matchupSection(grades, r, awayCol, homeCol, sport) {
  const rows = grades || [];
  const graded = (g) => !!(g && (g.overall || g.pass || g.run));
  if (!rows.some(graded)) return "";
  const card = (side) => {
    const g = rows.find((x) => x.side === side);
    const off = side === "home" ? r.home_team_name : r.away_team_name;
    const def = side === "home" ? r.away_team_name : r.home_team_name;
    const col = side === "home" ? homeCol : awayCol;
    const head = `<h3>${logoImg(off, sport)}${ctxEsc(off)} offense${g && g.early ? CTX_EARLY : ""}</h3><p class="mu-vs">vs ${ctxEsc(def)} defense</p>`;
    if (!graded(g)) {
      const fcs = g && (g.team === "FCS" || g.opponent === "FCS");
      return `<div class="trend-col mu-card" style="--tc:${col}">${head}<p class="trend-none">Not graded — ${fcs ? "no advanced stats for the FCS side" : "no unit ratings yet"}.</p></div>`;
    }
    const pct = (p) => { const n = ctxNum(p); return n == null ? null : Math.max(0, Math.min(100, n)); };
    const unit = (label, letter, p) => {
      const v = pct(p);
      return `<div class="mu-unit"><span>${label}</span>${gradeChip(letter)}<div class="mu-bar"><i class="g-${letter && /^[A-F]$/.test(letter) ? letter : "na"}" style="width:${v == null ? 0 : v}%"></i></div><small>${v == null ? "—" : `${ctxOrd(v)} pct`}</small></div>`;
    };
    const u = ctxJson(g.units);
    const pr = u && ctxNum(u.pass_rate) != null ? ` · pass rate ${Math.round(u.pass_rate * 100)}%` : "";
    const op = pct(g.overall_pct);
    return `<div class="trend-col mu-card" style="--tc:${col}">${head}
      <div class="mu-overall">${gradeChip(g.overall, true)}<div><b>Overall</b><small>${op == null ? "—" : `${ctxOrd(op)} percentile`}${pr}</small></div></div>
      ${unit("Pass", g.pass, g.pass_pct)}${unit("Run", g.run, g.run_pct)}</div>`;
  };
  return `<section class="section ctx-sec mu-sec"><div class="section-title"><div><h2>Matchup</h2><p>Each offense vs the opponent's defense — EPA, success rate &amp; explosiveness, graded A–F vs the last 3 seasons · ${CTX_NOT_A_PICK}</p></div></div>
    <div class="trends-grid">${card("away")}${card("home")}</div></section>`;
}

// History: L5/L10/L20/season records, streaks, home/away + fav/dog splits and the
// team's current power rank. Hidden when neither side has a history row.
function historySection(hist, power, r, awayCol, homeCol, sport) {
  const rows = hist || [];
  if (!rows.length) return "";
  const pw = power || [];
  const hasPrev = pw.some((p) => p.prev_rank != null);
  const col = (side) => {
    const h = rows.find((x) => x.side === side);
    const name = side === "home" ? r.home_team_name : r.away_team_name;
    const color = side === "home" ? homeCol : awayCol;
    const p = h ? pw.find((x) => String(x.team) === String(h.team)) : null;
    const head = `<h3>${logoImg(name, sport)}${ctxEsc(name)}</h3>${rkPowerLine(p, hasPrev)}`;
    if (!h) return `<div class="trend-col hist-col" style="--tc:${color}">${head}<p class="trend-none">No history on record${sport === "cfb" ? " (FCS teams aren't tracked)" : ""}.</p></div>`;
    const W = ctxJson(h.windows) || {};
    const wrow = (label, w) => {
      if (!w || !(+w.n > 0)) return `<tr><td>${label}</td><td colspan="4" class="hist-na">no games yet</td></tr>`;
      return `<tr><td>${label}<small>${+w.n} g</small></td><td><b>${ctxEsc(w.su)}</b></td><td>${+w.n_ats > 0 ? ctxEsc(w.ats) : "—"}</td><td>${+w.n_ou > 0 ? ctxEsc(w.ou) : "—"}</td><td>${ctxSigned(w.avg_margin)}</td></tr>`;
    };
    const table = `<div class="table-wrap hist-wrap"><table class="hist-tbl"><thead><tr><th></th><th>SU</th><th>ATS</th><th>O/U</th><th>AVG MARGIN</th></tr></thead><tbody>
      ${wrow("Last 5", W.L5)}${wrow("Last 10", W.L10)}${wrow("Last 20", W.L20)}${wrow("Season", W.season)}</tbody></table></div>`;
    const S = ctxJson(h.streaks) || {};
    const notes = ctxJson(S.notes) || {};
    const LBL = { su: "SU", ats: "ATS", ou: "O/U" };
    const chips = ["su", "ats", "ou"].map((k) => [
      S[k] ? `<span class="ctx-chip">${LBL[k]} ${ctxEsc(S[k])}</span>` : "",
      notes[k] ? `<span class="ctx-chip hot" title="last 7 ${LBL[k]} outcomes">${LBL[k]} ${ctxEsc(notes[k])}</span>` : ""].join("")).join("");
    const SP = ctxJson(h.splits) || {};
    const split = (label, s) => (s && +s.n > 0 ? `<li class="trend-row"><span>${label}</span><b>${ctxEsc(s.su)} SU · ${ctxEsc(s.ats)} ATS</b></li>` : "");
    const splits = [split("Home", SP.home), split("Away", SP.away), split("As favorite", SP.fav), split("As underdog", SP.dog)].join("");
    return `<div class="trend-col hist-col" style="--tc:${color}">${head}${table}
      ${chips ? `<div class="ctx-chips"><small>STREAKS</small>${chips}</div>` : ""}
      ${splits ? `<small class="hist-lbl">THIS SEASON</small><ul class="trend-list">${splits}</ul>` : ""}</div>`;
  };
  return `<section class="section ctx-sec hist-sec"><div class="section-title"><div><h2>History</h2><p>Straight-up, against the spread and over/under vs the closing line · ${CTX_NOT_A_PICK}</p></div></div>
    <div class="trends-grid">${col("away")}${col("home")}</div></section>`;
}

async function buildGame() {
  const { sport, game } = gameParams();
  const backList = `<a class="back-link" href="${sport}.html">← All ${sport.toUpperCase()} games</a>`;
  if (!game) return `<main><section class="section">${backList}<h2>No game selected</h2></section></main>`;
  // Read by game_pk from the BASE tables, not the upcoming-only `_current`
  // views: once a game kicks off it drops out of those views, but a detail
  // page should keep showing its projections forever. `predictions_any` is the
  // latest prediction per game with NO date floor (falls back to
  // predictions_current if the view isn't deployed yet); `prediction_accuracy`
  // carries the graded final result once the game is over.
  // NFL also reads the served sim version (nfl_sim_serving, one row) so the raw
  // nfl_player_sim / nfl_sim reads below show that version's rows. The
  // "Model: ML vN" label on the prediction block comes from `r.model_version`
  // (predictions_any/predictions_current expose it as their last column) --
  // that's the version of the row actually displayed and graded, not a
  // separate lookup. Undefined until the model_version migration runs, which
  // just means no label.
  const isNfl = sport === "nfl";
  const [predsAny, predsCur, evRows, accRows, servedVersion] = await Promise.all([
    sb(`predictions_any?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    sb(`predictions_current?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    sb(`ev_picks?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    sb(`prediction_accuracy?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    isNfl ? nflServedSimVersion() : Promise.resolve(null),
  ]);
  const r = predsAny[0] || predsCur[0];
  const actual = accRows[0] || null;
  if (!r) return `<main><section class="section">${backList}<h2>Game not found</h2><p style="opacity:.6">No prediction is on record for this game.</p></section></main>`;
  r.sport = sport;
  let sims = [], props = [], simRows = [], playerActuals = [], splits = [], lines = [];
  if (isNfl) {
    // Served-version filter; when the served read failed (null) fall back to
    // the newest rows of any version (the pre-serving-table behavior). Also,
    // when the STRICT filter comes back empty -- a pre-switch game whose
    // nfl_sim / nfl_player_sim rows only exist under an earlier version --
    // retry without the filter so those games keep showing what they have,
    // deduped the same way.
    const mv = simVersionFilter(servedVersion);
    const simRead = async (table) => {
      const rows = await sb(`${table}?game_pk=eq.${game}${mv}&order=created_at.desc`).catch(() => []);
      if (rows.length || !mv) return rows;
      return sb(`${table}?game_pk=eq.${game}&order=created_at.desc`).catch(() => []);
    };
    [sims, props, simRows, playerActuals, lines] = await Promise.all([
      simRead("nfl_player_sim").then(dedupLatest),
      sb(`ev_prop_picks?sport=eq.nfl&game_pk=eq.${game}`).catch(() => []),
      simRead("nfl_sim"),
      sb(`nfl_player_actuals?game_pk=eq.${game}`).catch(() => []),
      sb(`nfl_prop_lines?game_pk=eq.${game}`).catch(() => []),
    ]);
  }
  // Public betting splits: per-sport table (nfl_betting_splits_current /
  // cfb_betting_splits_current). Missing table/view -> caught -> [] -> hidden.
  // Betting trends: newest-season Action Network records per team
  // (team_betting_records, names quoted inside in.()) + NFL situational trends
  // (nfl_game_trends). Missing tables -> caught -> [] -> Trends empty state.
  // Team context (descriptive): history + matchup grades for this game, then the
  // current power rank of both teams (ids from those rows; the pooled FCS team is
  // never ranked). Missing tables -> caught -> [] -> sections hidden.
  let trendRecs = [], trendSits = [], ctxHist = [], ctxGrades = [], ctxPower = [];
  if (sport === "nfl" || sport === "cfb") {
    const q = (n) => `"${encodeURIComponent(n)}"`;
    [splits, trendRecs, trendSits, ctxHist, ctxGrades] = await Promise.all([
      sb(`${sport}_betting_splits_current?game_pk=eq.${game}`).catch(() => []),
      sb(`team_betting_records?sport=eq.${sport}&team_name=in.(${q(r.away_team_name)},${q(r.home_team_name)})&order=season.desc`).catch(() => []),
      sport === "nfl" ? sb(`nfl_game_trends?game_pk=eq.${game}`).catch(() => []) : Promise.resolve([]),
      sb(`team_history?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
      sb(`matchup_grades?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []),
    ]);
    const codes = [...new Set([...ctxHist, ...ctxGrades].map((x) => x.team).filter((t) => t && t !== "FCS"))];
    if (ctxHist.length && codes.length)
      ctxPower = await sb(`power_rankings_current?sport=eq.${sport}&team=in.(${codes.map((c) => q(String(c))).join(",")})`).catch(() => []);
  }
  // Player pmf map for the interactive per-player distribution panel.
  const playerMap = {};
  (sims || []).forEach((s) => {
    if (!playerMap[s.player_id]) playerMap[s.player_id] = { name: s.name, pos: s.pos, team: s.team, markets: {} };
    const d = distParse(s.dist);
    if (d) { d.__mean = s.mean; playerMap[s.player_id].markets[s.market] = d; }
  });
  window.__caGameSim = { players: playerMap };
  // Apple Sports-style hero: the AWAY team's color lives on the LEFT, the HOME
  // team's on the RIGHT, as two soft radial blobs that slowly drift/scale so the
  // gradient is alive, not static. They overlap in the middle for the mesh look.
  // Section accents below stay tinted with the home color on the dark page bg.
  const { away: awayCol, home: homeCol } = gameTeamColors(r.away_team_name, r.home_team_name, sport);
  const accent = homeCol;
  // Share the hero's team colors with the sim charts so bars match each team's
  // score-side color in the hero board (away = left, home = right).
  window.__caGameSim = Object.assign(window.__caGameSim || {}, {
    awayCol, homeCol, awayTeam: r.away_team_name, homeTeam: r.home_team_name,
  });
  const theme = `<style>
    .game-view .game-hero{position:relative;overflow:hidden;border:0;border-radius:0 0 20px 20px;padding:28px 16px 22px;min-height:200px;background:#0a0f16;box-shadow:inset 0 -52px 60px -30px #080d13}
    .game-view .game-hero::before,.game-view .game-hero::after{content:"";position:absolute;top:50%;width:78%;height:210%;border-radius:50%;filter:blur(64px);opacity:.6;z-index:0;pointer-events:none}
    .game-view .game-hero::before{left:-20%;background:radial-gradient(circle at center, ${awayCol} 0%, transparent 68%);animation:caBlobAway 16s ease-in-out infinite alternate}
    .game-view .game-hero::after{right:-20%;background:radial-gradient(circle at center, ${homeCol} 0%, transparent 68%);animation:caBlobHome 19s ease-in-out infinite alternate}
    .game-view .game-hero > *{position:relative;z-index:1}
    .game-view .game-hero .eyebrow,.game-view .game-hero .back-link{color:#e6edf4}
    .game-view .game-score,.game-view .game-hero h1{color:#fff;text-shadow:0 2px 16px rgba(0,0,0,.6)}
    .game-view .game-hero .game-winner,.game-view .game-hero .final-line{color:#eef3f9;text-shadow:0 1px 12px rgba(0,0,0,.55)}
    @keyframes caBlobAway{0%{transform:translate(-4%,-50%) scale(1)}50%{transform:translate(9%,-58%) scale(1.18)}100%{transform:translate(2%,-44%) scale(1.06)}}
    @keyframes caBlobHome{0%{transform:translate(4%,-50%) scale(1.08)}50%{transform:translate(-9%,-42%) scale(1.22)}100%{transform:translate(-2%,-58%) scale(1)}}
    @media (prefers-reduced-motion: reduce){.game-view .game-hero::before,.game-view .game-hero::after{animation:none}}
    .game-view .section-title h2{border-left:3px solid ${accent};padding-left:11px}
    .game-view .compact-stats article{border-top:2px solid ${accent}66}
  </style>`;
  // Final result banner (present once graded): both scores are recovered from
  // the stored home margin + total — home=(total+margin)/2, away=(total-margin)/2.
  let finalLine = "";
  if (actual && actual.actual_total != null && actual.actual_margin != null) {
    const ah = Math.round((actual.actual_total + actual.actual_margin) / 2);
    const aa = Math.round((actual.actual_total - actual.actual_margin) / 2);
    const w = actual.actual_winner || (actual.actual_margin >= 0 ? r.home_team_name : r.away_team_name);
    const call = actual.winner_correct == null ? ""
      : ` · <span class="${actual.winner_correct ? "call-ok" : "call-miss"}">model ${actual.winner_correct ? "called it ✓" : "missed ✗"}</span>`;
    finalLine = `<p class="final-line">Final · ${r.away_team_name} ${aa} – ${ah} ${r.home_team_name} · winner <b>${w}</b>${call}</p>`;
  }
  const hero = `<section class="game-hero">${backList}<p class="eyebrow">${sport.toUpperCase()} · ${timeET(r.commence_time)}</p><div class="game-score">${projScoreLogos(r)}</div><h1>${matchupOf(r)}</h1><p class="game-winner">Projected winner: <b>${predWinner(r)}</b> · ${confPct(r.home_win_prob)} confidence</p>${finalLine}</section>`;
  // NFL: one prediction block (winner, win %, projected score, spread/total
  // leans; the actual result beside each once graded). CFB keeps its cards.
  const predictionsSec = isNfl
    ? nflPredictionSection(r, actual, mlModelTag(r.model_version))
    : `<section class="section"><div class="section-title"><h2>Predictions</h2></div><section class="compact-stats">${stat('PROJECTED WINNER', predWinner(r), `${confPct(r.home_win_prob)} confidence`)}${stat('MODEL SPREAD', spreadLean(r), `model ${homeSpread(r)}`, 'blue')}${stat('MODEL TOTAL', totalLean(r), `model ${modelTotal(r)}`, 'cyan')}${stat('PROJECTED SCORE', projScore(r), matchupOf(r))}</section></section>`;
  const simVisualSec = sport === "nfl" ? gameSimVisual(simRows[0], r.away_team_name, r.home_team_name, awayCol, homeCol) : "";
  const boxSec = sport === "nfl" ? boxscoreSection(sims, r) : "";
  const splitsSec = (sport === "nfl" || sport === "cfb") ? splitsSection(splits, r, awayCol, homeCol) : "";
  const trendsSec = (sport === "nfl" || sport === "cfb")
    ? trendsSection(r.away_team_name, r.home_team_name, trendRecs, trendSits, awayCol, homeCol, r.market_spread, sport) : "";
  const matchupSec = (sport === "nfl" || sport === "cfb") ? matchupSection(ctxGrades, r, awayCol, homeCol, sport) : "";
  const historySec = (sport === "nfl" || sport === "cfb") ? historySection(ctxHist, ctxPower, r, awayCol, homeCol, sport) : "";
  const evSec = evSection(evRows.map((x) => ({ ...x, sport })));
  // Label the props section from the rows actually shown, not the served
  // version: once the R2 fallback below serves rows of any version, the served
  // version and the shown rows' version can differ, and a false "ML v1" label
  // must not appear on non-ML rows.
  const shownSimVersion = (sims[0] && sims[0].model_version) || null;
  const propsSec = isNfl ? propsProjectionSection(sims, props, playerActuals, lines, mlModelTag(shownSimVersion)) : "";
  return `<main class="game-view">${theme}${hero}${predictionsSec}${splitsSec}${matchupSec}${historySec}${trendsSec}${simVisualSec}${boxSec}${evSec}${propsSec}</main>`;
}

// The sim version the site serves: the single row of nfl_sim_serving (anon
// readable). null when the read fails or the table is empty -> callers fall
// back to the newest rows of any version.
async function nflServedSimVersion() {
  try {
    const rows = await sb("nfl_sim_serving?select=model_version&limit=1");
    return (rows && rows[0] && rows[0].model_version) || null;
  } catch (e) { return null; }
}
// PostgREST filter fragment for the served sim version ("" = no filter).
const simVersionFilter = (v) => (v ? `&model_version=eq.${encodeURIComponent(v)}` : "");
// Small "Model: ML vN" tag, only for an ML sim version string
// ("nfl-sim-ml-v1" -> "ML v1", "nfl-sim-ml-v2" -> "ML v2", ...).
const ML_SIM_PREFIX = "nfl-sim-ml-";
const mlModelTag = (v) => {
  if (typeof v !== "string" || !v.startsWith(ML_SIM_PREFIX)) return "";
  const m = /^v(\d+)$/.exec(v.slice(ML_SIM_PREFIX.length));
  return m ? `<span class="sim-tag model-tag">Model: ML v${m[1]}</span>` : "";
};

// NFL prediction block: the served pre-game prediction (winner + win %,
// projected score, spread and total leans vs the market line), with the actual
// result beside each once the game is graded (prediction_accuracy). Works for
// any prediction row, so finished games from before the ML switch still show
// whatever prediction was on record.
function nflPredictionSection(r, actual, modelTag = "") {
  const graded = !!(actual && actual.actual_total != null && actual.actual_margin != null);
  const ok = (v) => (v == null ? "" : ` <span class="${v ? "call-ok" : "call-miss"}">${v ? "✓" : "✗"}</span>`);
  const card = (label, value, sub, act, cls = "") =>
    `<article><span>${label}</span><strong class="${cls}">${value}</strong><small>${sub}</small>${act ? `<small class="pred-actual">${act}</small>` : ""}</article>`;
  let aWin = "", aScore = "", aMargin = "", aTotal = "";
  if (graded) {
    const ah = Math.round((actual.actual_total + actual.actual_margin) / 2);
    const aa = Math.round((actual.actual_total - actual.actual_margin) / 2);
    const m = +actual.actual_margin;
    const w = actual.actual_winner || (m >= 0 ? r.home_team_name : r.away_team_name);
    aWin = `Actual: <b>${ctxEsc(w)}</b>${ok(actual.winner_correct)}`;
    aScore = `Actual: <b>${aa}–${ah}</b>`;
    aMargin = `Actual: <b>${m === 0 ? "tie" : `${ctxEsc(m > 0 ? r.home_team_name : r.away_team_name)} by ${Math.abs(m)}`}</b>${ok(actual.spread_pick_correct)}`;
    aTotal = `Actual: <b>${(+actual.actual_total).toFixed(0)}</b>${ok(actual.total_pick_correct)}`;
  }
  const head = `<div class="section-title"><div><h2>Prediction ${modelTag}</h2><p>Pre-game projection${graded ? " · actual result beside each once graded" : ""}</p></div></div>`;
  return `<section class="section pred-block">${head}<section class="compact-stats">
    ${card("PROJECTED WINNER", ctxEsc(predWinner(r)), `${confPct(r.home_win_prob)} win probability`, aWin)}
    ${card("PROJECTED SCORE", projScore(r), ctxEsc(matchupOf(r)), aScore)}
    ${card("SPREAD LEAN", spreadLean(r), `model ${homeSpread(r)}`, aMargin, "blue")}
    ${card("TOTAL LEAN", totalLean(r), `model ${modelTotal(r)}`, aTotal, "cyan")}
  </section></section>`;
}

// Best available moneyline per side (game_moneylines_current: each US book's
// latest price, best per side -- the two sides can be different books).
// One retry: a cold-cache first query can hit the anon statement timeout, and
// the retry lands on a warm cache -- without it every tile shows the model line.
async function gameMoneylines() {
  const q = "game_moneylines_current?select=*";
  const rows = await sb(q).catch(() => sb(q)).catch(() => []);
  return new Map(rows.map((m) => [String(m.game_pk), m]));
}
// Sportsbook logo = the book's own site icon (Google favicon service); if it
// fails to load, a short text badge takes its place.
const BOOK_SITE = { draftkings: ["draftkings.com", "DK"], fanduel: ["fanduel.com", "FD"], betmgm: ["betmgm.com", "MGM"],
  williamhill_us: ["caesars.com", "CZR"], caesars: ["caesars.com", "CZR"], fanatics: ["fanatics.com", "FAN"],
  espnbet: ["espnbet.com", "ESPN"], hardrockbet: ["hardrock.bet", "HR"], thescore: ["thescore.bet", "SCR"],
  bet365: ["bet365.com", "365"], ballybet: ["ballybet.com", "BAL"] };
const bookLogo = (book) => {
  const [site, short] = BOOK_SITE[book] || [null, String(book || "").slice(0, 3).toUpperCase()];
  const name = evBookName(book);
  return site
    ? `<img class="bk-logo" src="https://www.google.com/s2/favicons?domain=${site}&sz=64" alt="${name}" title="${name}" loading="lazy" onerror="this.outerHTML='<span class=&quot;bk-txt&quot; title=&quot;${name}&quot;>${short}</span>'">`
    : `<span class="bk-txt" title="${name}">${short}</span>`;
};

// One side's price at the user's books: the highest decimal odds among the
// selected books in the side's {book: american} map (object or JSON string);
// a tie keeps the view's own best book when it's selected, else the
// alphabetically first book. No map (older view) → the view's own best price,
// only if that book is selected. null = none of the user's books priced it.
function tileBest(prices, viewPrice, viewBook, s) {
  let map = prices;
  if (typeof map === "string") { try { map = JSON.parse(map); } catch { map = null; } }
  if (!map || typeof map !== "object")
    return viewPrice != null && bookSelected(viewBook, s) ? { am: viewPrice, book: viewBook } : null;
  let best = null;
  for (const book of Object.keys(map).sort()) {
    const am = map[book], a = +am;
    if (am == null || !Number.isFinite(a) || (a > -100 && a < 100) || !bookSelected(book, s)) continue;
    const dec = a > 0 ? 1 + a / 100 : 1 + 100 / -a;
    if (!best || dec > best.dec || (dec === best.dec && bookKey(book) === bookKey(viewBook))) best = { am: a, book, dec };
  }
  return best;
}

// A game card: away team on top, home below, each with its logo + short name and
// the best moneyline at the user's books + that book's logo; kickoff time along
// the bottom. With no price at any of the user's books, the model's fair line
// shows instead, dimmed and labelled. Cards flow into a responsive grid (gameList).
const gameLinkRow = (r, sport, mls, s = getSettings()) => {
  const m = mls && mls.get(String(r.game_pk));
  const price = (best, fair) => best
    ? `<span class="gc-ml">${evPrice(best.am)}${bookLogo(best.book)}</span>`
    : `<span class="gc-ml gc-model" title="Model fair line — none of your selected books have priced this yet">${fair}<small>model</small></span>`;
  const side = (k) => (m ? tileBest(m[`${k}_prices`], m[`${k}_price`], m[`${k}_book`], s) : null);
  const teamRow = (name, ml) =>
    `<div class="gc-row"><span class="gc-team">${logoImg(name, sport)}<b>${teamShort(name, sport)}</b></span>${ml}</div>`;
  return `<a class="game-card" href="game.html?sport=${sport}&game=${r.game_pk}">
    ${teamRow(r.away_team_name, price(side("away"), awayML(r)))}
    ${teamRow(r.home_team_name, price(side("home"), homeML(r)))}
    <div class="gc-time">${timeET(r.commence_time)}</div></a>`;
};

const gameList = (preds, sport, empty, mls, s = getSettings()) =>
  preds.length ? `<div class="game-grid">${preds.map((r) => gameLinkRow(r, sport, mls, s)).join("")}</div>`
               : `<p style="opacity:.6">${empty}</p>`;

async function buildLeague(sport) {
  const [preds, mls] = await Promise.all([predictions(sport), gameMoneylines()]);
  const name = sport.toUpperCase(), s = getSettings();
  return `<main><section class="page-heading"><div><p class="eyebrow">${name}</p><h1>${name} games</h1><p>Tap a game for the predicted score, predictions, +EV picks and projected player props.</p></div><div class="page-head-stat"><span>GAMES</span><strong>${preds.length}</strong><small>this week</small></div></section>${gameList(preds, sport, `No upcoming ${name} games right now.`, mls, s)}</main>`;
}

/* ── Power rankings page (rankings.html?sport=nfl|cfb) ────────────────────
   power_rankings_current = the latest (season, week) per sport. Sortable
   columns + (CFB) a conference filter; the view state lives on
   window.__caRank so the 5-minute re-render keeps the user's sort/filter. */
const CFB_CONF = { 1: "ACC", 4: "Big 12", 5: "Big Ten", 8: "SEC", 9: "Pac-12", 12: "C-USA", 15: "MAC",
  17: "Mountain West", 18: "FBS Independents", 37: "Sun Belt", 151: "American" };
const rkConfId = (c) => (c == null || c === "" ? null : String(c).replace(/\.0+$/, ""));
const rkConfName = (c) => { const id = rkConfId(c); return id == null ? "—" : (CFB_CONF[id] || `Conf ${id}`); };
const rkUnitRank = (row, k) => { const u = ctxJson(row.units); return u && u[k] ? ctxNum(u[k].rank) : null; };
const rkRec = (s) => (typeof s === "string" && /^\d+(-\d+)+$/.test(s) ? s.split("-").map(Number) : null);
const rkSuPct = (s) => { const r = rkRec(s); if (!r) return null; const [w, l, t = 0] = r; return w + l + t ? (w + t / 2) / (w + l + t) : null; };
const rkAtsPct = (s) => { const r = rkRec(s); if (!r) return null; const [w, l] = r; return w + l ? w / (w + l) : null; };
// [key, header, default direction, sort value, cell html]; the rank tie-break is always ascending.
const RK_UNITS = [["pass_off", "PASS OFF"], ["run_off", "RUN OFF"], ["pass_def", "PASS DEF"], ["run_def", "RUN DEF"]];
function rkColumns(sport, hasPrev) {
  const cols = [
    ["rank", "RANK", "asc", (x) => ctxNum(x.rank), (x) => `<b>${ctxEsc(x.rank)}</b>`],
    ["team", "TEAM", "asc", (x) => ctxTeamName(x.team, sport).toLowerCase(),
      (x) => `<b class="rk-team">${ctxLogo(x.team, sport)}${ctxEsc(ctxTeamName(x.team, sport))}</b>`],
    ["rating", "RATING", "desc", (x) => ctxNum(x.rating), (x) => `<b class="rk-rating">${rkRating(x.rating)}</b>`],
    ["move", "MOVE", "desc", (x) => (x.prev_rank == null ? null : ctxNum(x.move)), (x) => rkMove(x, hasPrev)],
    ...RK_UNITS.map(([k, h]) => [k, h, "asc", (x) => rkUnitRank(x, k), (x) => { const v = rkUnitRank(x, k); return v == null ? "—" : `#${v}`; }]),
    ["sos", "SOS", "desc", (x) => ctxNum(x.sos), (x) => ctxSigned(x.sos)],
    ["sov", "SOV", "desc", (x) => ctxNum(x.sov), (x) => (ctxNum(x.sov) == null ? "—" : ctxSigned(x.sov))],
    ["su", "RECORD", "desc", (x) => rkSuPct(x.su), (x) => ctxEsc(x.su || "—")],
    ["home_record", "HOME", "desc", (x) => rkSuPct(x.home_record), (x) => ctxEsc(x.home_record || "—")],
    ["road_record", "ROAD", "desc", (x) => rkSuPct(x.road_record), (x) => ctxEsc(x.road_record || "—")],
    ["ats", "ATS", "desc", (x) => rkAtsPct(x.ats), (x) => ctxEsc(x.ats || "—")],
  ];
  if (sport === "cfb") cols.splice(2, 0, ["conf", "CONF", "asc", (x) => rkConfName(x.conf).toLowerCase(), (x) => ctxEsc(rkConfName(x.conf))]);
  return cols;
}
// Filter (CFB conference id or "all") then sort; nulls always sink to the bottom.
function rankingsRows(rows, sport, state) {
  const st = state || {};
  const col = rkColumns(sport, true).find((c) => c[0] === st.key) || rkColumns(sport, true)[0];
  const dir = st.dir === "desc" ? -1 : 1, val = col[3];
  const conf = st.conf && st.conf !== "all" ? String(st.conf) : null;
  return (rows || [])
    .filter((x) => !conf || rkConfId(x.conf) === conf)
    .map((x) => ({ x, v: val(x) }))
    .sort((a, b) => {
      if (a.v == null || b.v == null) { if (a.v == null && b.v == null) return (ctxNum(a.x.rank) ?? 1e9) - (ctxNum(b.x.rank) ?? 1e9); return a.v == null ? 1 : -1; }
      const c = typeof a.v === "string" ? a.v.localeCompare(b.v) : a.v - b.v;
      return c ? c * dir : (ctxNum(a.x.rank) ?? 1e9) - (ctxNum(b.x.rank) ?? 1e9);
    })
    .map((o) => o.x);
}
function rankingsTable(rows, sport, state) {
  const st = state || {};
  const hasPrev = (rows || []).some((x) => x.prev_rank != null);
  const cols = rkColumns(sport, hasPrev);
  const key = cols.some((c) => c[0] === st.key) ? st.key : "rank";
  const shown = rankingsRows(rows, sport, { ...st, key });
  const head = cols.map(([k, h]) => {
    const on = k === key, arrow = on ? (st.dir === "desc" ? " ▾" : " ▴") : "";
    return `<th data-sort="${k}" class="rk-th${on ? " sorted" : ""}" aria-sort="${on ? (st.dir === "desc" ? "descending" : "ascending") : "none"}" tabindex="0">${h}${arrow}</th>`;
  }).join("");
  const body = shown.length
    ? shown.map((x) => `<tr>${cols.map((c) => `<td>${c[4](x)}</td>`).join("")}</tr>`).join("")
    : `<tr><td colspan="${cols.length}" class="rk-empty">No ranked teams in this conference.</td></tr>`;
  return `<div class="table-wrap"><table class="rk-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
function rkSport() {
  const s = (new URLSearchParams(location.search).get("sport") || "nfl").toLowerCase();
  return s === "cfb" ? "cfb" : "nfl";
}
async function buildRankings() {
  const sport = rkSport(), name = sport.toUpperCase();
  const rows = await sb(`power_rankings_current?sport=eq.${sport}&order=rank.asc`).catch(() => []);
  const prev = window.__caRank;
  const st = window.__caRank = prev && prev.sport === sport ? prev : { sport, key: "rank", dir: "asc", conf: "all" };
  window.__caRankRows = rows;
  const tabs = ["nfl", "cfb"].map((s) => `<a class="${s === sport ? "selected" : ""}" href="rankings.html?sport=${s}">${s.toUpperCase()}</a>`).join("");
  const wk = rows[0] ? `Season ${ctxEsc(rows[0].season)} · Week ${ctxEsc(rows[0].week)}` : "";
  const intro = `Rating = points better than an average team on a neutral field, earned from this season's results: point margin (blowouts capped, adjusted for home vs road), wins weighed against how likely they were (road upsets earn more, bad home losses cost more) and strength of schedule. This season counts games ÷ (games + 1) — 75% after 3 games — the rest is the preseason rating. SOS = average rating of opponents played; SOV = average rating of teams beaten; unit ranks by opponent-adjusted EPA/play (shown, not rated); records are regular season · ${CTX_NOT_A_PICK}`;
  const heading = `<section class="page-heading"><div><p class="eyebrow">${name} · POWER RANKINGS</p><h1>${name} power rankings</h1><p>${intro}</p></div><div class="page-head-stat"><span>TEAMS RANKED</span><strong>${rows.length}</strong><small>${wk || "not published yet"}</small></div></section>`;
  if (!rows.length) return `<main class="rk-page">${heading}<div class="rk-bar"><div class="rk-sport">${tabs}</div></div><div class="ev-empty"><b>No ${name} power rankings yet.</b><p>They publish with the daily team-context job ahead of each week's games.</p></div></main>`;
  let confSel = "";
  if (sport === "cfb") {
    const ids = [...new Set(rows.map((x) => rkConfId(x.conf)).filter((c) => c != null))]
      .sort((a, b) => rkConfName(a).localeCompare(rkConfName(b)));
    if (st.conf !== "all" && !ids.includes(String(st.conf))) st.conf = "all";
    confSel = `<label class="rk-conf">Conference <select id="rk-conf"><option value="all">All conferences</option>${ids.map((c) => `<option value="${ctxEsc(c)}"${String(st.conf) === c ? " selected" : ""}>${ctxEsc(rkConfName(c))}</option>`).join("")}</select></label>`;
  }
  return `<main class="rk-page">${heading}<div class="rk-bar"><div class="rk-sport">${tabs}</div>${confSel}</div><div class="rk-body">${rankingsTable(rows, sport, st)}</div></main>`;
}
function wireRankings() {
  const body = document.querySelector(".rk-body"); if (!body) return;
  const redraw = () => { body.innerHTML = rankingsTable(window.__caRankRows || [], rkSport(), window.__caRank); };
  const sortBy = (th) => {
    const st = window.__caRank, k = th.dataset.sort;
    const col = rkColumns(rkSport(), true).find((c) => c[0] === k); if (!col) return;
    if (st.key === k) st.dir = st.dir === "desc" ? "asc" : "desc";
    else { st.key = k; st.dir = col[2]; }
    redraw();
  };
  body.addEventListener("click", (e) => { const th = e.target.closest("th[data-sort]"); if (th) sortBy(th); });
  body.addEventListener("keydown", (e) => {
    const th = e.target.closest("th[data-sort]");
    if (th && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); sortBy(th); }
  });
  const sel = document.getElementById("rk-conf");
  if (sel) sel.addEventListener("change", () => { window.__caRank.conf = sel.value; redraw(); });
}

/* ── Settings page (values filled + kept in sync by wireSettings) ──────── */
function buildSettings() {
  const card = (title, sub, inner, extra = "") => `<section class="section set-card"><div class="section-title"><div><h2>${title}</h2>${sub ? `<p>${sub}</p>` : ""}</div>${extra}</div>${inner}</section>`;
  const note = (t) => `<p class="set-note">${t}</p>`;
  const money = (id, label, max) => `<span class="set-money"><span>$</span><input type="number" id="${id}" min="0.01" max="${max}" step="any" inputmode="decimal" aria-label="${label}"></span>`;
  const books = US_BOOKS.map(([k, name]) => `<button type="button" class="set-book" data-book="${k}" aria-pressed="false">${bookLogo(k)}<span>${name}</span><i aria-hidden="true">✓</i></button>`).join("");
  const units = [10, 25, 50, 100].map((u) => `<button type="button" data-unit="${u}">$${u}</button>`).join("");
  const kelly = KELLY_FRACTIONS.map(([f, l]) => `<button type="button" data-kelly="${f}" aria-pressed="false">${l}</button>`).join("");
  return `<main class="settings"><section class="page-heading"><div><p class="eyebrow">SETTINGS</p><h1>Your settings</h1><p>Saved in this browser — they apply across the site.</p></div></section>${settingsStorageOk() ? "" : `<p class="set-warn">Settings can't be saved in this browser (storage blocked)</p>`}
    ${card("Sportsbooks", "The books you bet at.", `<div class="set-books">${books}</div><p class="set-note set-books-min" hidden>At least one sportsbook must stay selected.</p>`,
      `<div class="set-actions"><button type="button" data-books="all">Select all</button><button type="button" data-books="none">Clear</button></div>`)}
    ${card("Unit size", "", `<div class="set-row">${money("set-unit", "Unit size in dollars", SETTINGS_MAX.unit)}<div class="set-chips">${units}</div></div>${note("Every $ amount on the site (profit trackers, parlay payouts, Kelly units) uses this.")}`)}
    ${card("Bankroll &amp; Kelly", "", `<div class="set-row"><label class="set-field"><small>BANKROLL</small>${money("set-bankroll", "Bankroll in dollars", SETTINGS_MAX.bankroll)}</label><div class="set-field"><small>KELLY FRACTION</small><div class="set-seg" role="group" aria-label="Kelly fraction">${kelly}</div></div></div>${note("Kelly suggests a stake from your bankroll and each bet's edge. Quarter Kelly is the common choice when edges are model estimates.")}${note("Kelly sizes each bet on its own — bets on the same game (e.g. moneyline and spread) are correlated, so don't stack full stakes on both.")}`)}
    ${card("Minimum EV", "", `<div class="set-range"><input type="range" id="set-minev" min="0" max="20" step="0.5" aria-label="Minimum EV percent"><output id="set-minev-out" for="set-minev"></output></div>${note("The +EV page hides picks below this EV at your books.")}`)}
    <section class="section"><button type="button" class="set-reset">Reset to defaults</button></section>
    <div class="set-saved" role="status" aria-live="polite"></div></main>`;
}

function wireSettings() {
  const root = document.querySelector("main.settings"); if (!root) return;
  const q = (sel) => root.querySelector(sel);
  const unitIn = q("#set-unit"), bankIn = q("#set-bankroll"), evIn = q("#set-minev"), evOut = q("#set-minev-out");
  const minNote = q(".set-books-min"), savedTag = q(".set-saved"), canSave = settingsStorageOk();
  let savedTimer;
  const sync = (s) => {
    root.querySelectorAll(".set-book").forEach((b) => { const on = s.books.includes(b.dataset.book); b.classList.toggle("selected", on); b.setAttribute("aria-pressed", on); });
    root.querySelectorAll("[data-unit]").forEach((b) => b.classList.toggle("selected", +b.dataset.unit === s.unit));
    root.querySelectorAll("[data-kelly]").forEach((b) => { const on = +b.dataset.kelly === s.kelly; b.classList.toggle("selected", on); b.setAttribute("aria-pressed", on); });
    if (document.activeElement !== unitIn) unitIn.value = s.unit;   // don't fight the user mid-typing
    if (document.activeElement !== bankIn) bankIn.value = s.bankroll;
    evIn.value = s.minEv; evOut.textContent = `${s.minEv.toFixed(1)}%`;
  };
  const save = (partial) => {
    sync(saveSettings(partial));
    if (!canSave) return;  // the storage-blocked note is already showing
    savedTag.textContent = "Saved ✓"; savedTag.classList.add("show");
    clearTimeout(savedTimer); savedTimer = setTimeout(() => savedTag.classList.remove("show"), 1400);
  };
  q(".set-books").addEventListener("click", (e) => {
    const b = e.target.closest(".set-book"); if (!b) return;
    const cur = getSettings().books, k = b.dataset.book, on = cur.includes(k);
    minNote.hidden = !(on && cur.length === 1);
    if (!minNote.hidden) return;  // the last selected book stays selected
    save({ books: on ? cur.filter((x) => x !== k) : [...cur, k] });
  });
  q('[data-books="all"]').addEventListener("click", () => { minNote.hidden = true; save({ books: US_BOOKS.map((b) => b[0]) }); });
  q('[data-books="none"]').addEventListener("click", () => { minNote.hidden = false; save({ books: getSettings().books.slice(0, 1) }); });
  // Number inputs save each valid keystroke; an invalid entry is ignored and
  // snaps back to the last valid value when the field is left.
  [[unitIn, "unit"], [bankIn, "bankroll"]].forEach(([inp, key]) => {
    inp.addEventListener("input", () => { const v = setNum(inp.value); if (Number.isFinite(v) && v > 0 && v <= SETTINGS_MAX[key]) save({ [key]: v }); });
    inp.addEventListener("change", () => { inp.value = getSettings()[key]; });
  });
  root.querySelectorAll("[data-unit]").forEach((b) => b.addEventListener("click", () => save({ unit: +b.dataset.unit })));
  root.querySelectorAll("[data-kelly]").forEach((b) => b.addEventListener("click", () => save({ kelly: +b.dataset.kelly })));
  evIn.addEventListener("input", () => save({ minEv: +evIn.value }));
  q(".set-reset").addEventListener("click", () => { minNote.hidden = true; save({ ...SETTINGS_DEFAULTS, books: [...SETTINGS_DEFAULTS.books] }); });
  sync(getSettings());
}

function chartPath(vals) {
  if (vals.length < 2) return "";
  const max = Math.max(...vals, 0), min = Math.min(...vals, 0), range = max - min || 1;
  const pts = vals.map((v, i) => [i / (vals.length - 1) * 1000, 250 - ((v - min) / range) * 240]);
  const line = "M" + pts.map((p) => `${p[0].toFixed(0)} ${p[1].toFixed(0)}`).join(" L");
  const [endX, endY] = pts.at(-1);
  return `<path d="${line}" fill="none" stroke="var(--blue)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/><circle cx="${endX.toFixed(0)}" cy="${endY.toFixed(0)}" r="6" fill="var(--blue)"/>`;
}

const spreadFromMargin = (m) => { if (m == null) return "—"; const v = r05(m); return v > 0 ? `-${v}` : v < 0 ? `+${-v}` : "PK"; };
const fmtLine = (x) => x > 0 ? `+${x}` : x < 0 ? `${x}` : "PK";
// The side the model took against the closing market spread, shown from that
// side's perspective (team logo + its line). market_spread is the HOME line;
// pred_margin + market_spread > 0 means the model's projection covers the home
// side. Falls back to the model's own line when no market line was captured.
const spreadPick = (r) => {
  if (r.market_spread == null || r.pred_margin == null) return spreadFromMargin(r.pred_margin);
  // Rounded-margin comparison (matches the displayed model number); a model that
  // lands on the line reads PK rather than flipping to the dog. See spreadLean.
  const edge = r05(r.pred_margin) + r.market_spread;
  if (Math.abs(edge) < 0.25) return "PK";
  const likesHome = edge > 0;
  const team = likesHome ? r.home_team_name : r.away_team_name;
  const line = likesHome ? r.market_spread : -r.market_spread;
  return `${logoImg(team, r.sport)}${fmtLine(line)}`;
};
// The model's Over/Under lean against the closing market total (not its own number).
const totalPick = (r) => {
  if (r.market_total == null || r.pred_total == null)
    return r.pred_total != null ? (+r.pred_total).toFixed(1) : "—";
  return `${r.pred_total > r.market_total ? "Over" : "Under"} ${(+r.market_total).toFixed(1)}`;
};

// ==== Track record: league -> week -> games ================================
// A "football week" buckets Thu–Mon games (incl. Monday Night) together: the
// Tuesday on/before the game date. Returns YYYY-MM-DD of that Tuesday.
function footballWeekStart(dateStr) {
  const d = new Date(`${dateStr}T12:00:00Z`);
  const delta = (d.getUTCDay() - 2 + 7) % 7; // 2 = Tuesday
  d.setUTCDate(d.getUTCDate() - delta);
  return d.toISOString().slice(0, 10);
}
// Attach a per-sport ordinal week number (_week) to each graded row, numbering
// each sport's distinct football-weeks 1..N in date order. Returns
// {sport: [weekStart, ...]} so callers know how many weeks each sport has.
function attachTrackWeeks(rows) {
  const startsBySport = {};
  rows.forEach((r) => {
    const s = footballWeekStart(r.game_date);
    r._wkStart = s;
    (startsBySport[r.sport] = startsBySport[r.sport] || new Set()).add(s);
  });
  const orderBySport = {};
  Object.entries(startsBySport).forEach(([sp, set]) => { orderBySport[sp] = [...set].sort(); });
  rows.forEach((r) => { r._week = orderBySport[r.sport].indexOf(r._wkStart) + 1; });
  return orderBySport;
}
function trackRowHtml(r) {
  const badge = (ok) => ok == null ? "—" : `<span class="${ok ? "win" : "loss"}">${ok ? "✓" : "✗"}</span>`;
  const fin = (g) => { const h = (g.actual_total + g.actual_margin) / 2, a = (g.actual_total - g.actual_margin) / 2; return `${Math.round(a)}–${Math.round(h)}`; };
  return `<tr><td>${(r.game_date || "").slice(5)}</td><td><b>${logoPair(`${r.away_team_name} @ ${r.home_team_name}`, r.sport)}${r.away_team_name} @ ${r.home_team_name}</b></td><td>${badge(r.winner_correct)} ${logoImg(r.predicted_winner, r.sport)}${r.predicted_winner}</td><td><b>${r.win_prob != null ? confPct(r.win_prob) : "—"}</b></td><td>${badge(r.spread_pick_correct)} ${spreadPick(r)}</td><td>${badge(r.total_pick_correct)} ${totalPick(r)}</td><td>${fin(r)}</td></tr>`;
}
function trackWeekButtons(sport, activeWeek) {
  const n = ((window.__caTrack || {}).weeksBySport?.[sport] || []).length;
  if (!n) return `<span style="opacity:.6;font-size:13px">No graded weeks yet</span>`;
  const weeks = []; for (let w = 1; w <= n; w++) weeks.push(w); // oldest first (Week 1 leftmost)
  return weeks.map((w) => `<button class="${w === activeWeek ? "selected" : ""}" data-week="${w}">Week ${w}</button>`).join("");
}
function trackWeekTable(sport, week) {
  const rows = ((window.__caTrack || {}).rows || []).filter((r) => r.sport === sport && r._week === week);
  if (!rows.length) return `<p style="opacity:.6;padding:16px">No graded games for this week yet.</p>`;
  const wl = (key) => { const d = rows.filter((r) => r[key] != null); const w = d.filter((r) => r[key]).length; return d.length ? `${w}-${d.length - w}` : "—"; };
  const rec = `<div class="week-record"><b>Week ${week}</b> · ${rows.length} games · ${wl("winner_correct")} ML · ${wl("spread_pick_correct")} ATS · ${wl("total_pick_correct")} O/U</div>`;
  const body = [...rows].sort((a, b) => (a.game_date < b.game_date ? 1 : -1)).map(trackRowHtml).join("");
  return `${rec}<div class="table-wrap"><table><thead><tr><th>DATE</th><th>MATCHUP</th><th>MONEYLINE</th><th>CONF</th><th>SPREAD</th><th>TOTAL</th><th>FINAL</th></tr></thead><tbody>${body}</tbody></table></div>`;
}
// Wire the track league/week drill-down after render() sets innerHTML.
function wireTrack(restoreLeague, restoreWeek) {
  const t = window.__caTrack; if (!t) return;
  const lf = document.querySelector(".track-league-filter"); if (!lf) return;
  const wf = document.querySelector(".track-week-filter");
  const body = document.querySelector(".track-week-body");
  const pick = (sport, week) => {
    if (!sport || !(t.weeksBySport[sport] || []).length) sport = t.defaultSport;
    const n = (t.weeksBySport[sport] || []).length;
    const wk = week && week <= n ? week : n; // default = latest week
    lf.querySelectorAll("button").forEach((b) => b.classList.toggle("selected", b.dataset.league === sport));
    wf.innerHTML = trackWeekButtons(sport, wk);
    body.innerHTML = trackWeekTable(sport, wk);
  };
  lf.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => pick(b.dataset.league)));
  wf.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-week]"); if (!b) return;
    const sport = lf.querySelector("button.selected")?.dataset.league || t.defaultSport;
    wf.querySelectorAll("button").forEach((x) => x.classList.remove("selected"));
    b.classList.add("selected");
    body.innerHTML = trackWeekTable(sport, +b.dataset.week);
  });
  pick(restoreLeague, restoreWeek);
}

// Per-sport track-record restart (table track_record_start): a sport's rows
// before its starts_at are archived -- kept in the database and its *_all views
// for comparison, but left out of the record shown here. The server-side
// record views apply the same cut; these helpers cut the raw tables read below.
async function trackRecordStarts() {
  const rows = await sb("track_record_start?select=sport,starts_at,model_version").catch(() => []);
  return new Map((rows || []).map((r) => [r.sport, r]));
}
const etDateStr = (iso) => new Date(iso).toLocaleDateString("en-CA", { timeZone: "America/New_York" });
// `when`: an ISO timestamp (kickoff) or a YYYY-MM-DD game date; unknown -> kept.
function inTrackRecord(starts, sport, when) {
  const s = starts.get(sport);
  if (!s || !when) return true;
  return /^\d{4}-\d{2}-\d{2}$/.test(when) ? when >= etDateStr(s.starts_at) : new Date(when) >= new Date(s.starts_at);
}

async function buildTrack() {
  window.__caPnl = [];
  let [tiers, graded, evRes, evPk, propRes, propGradeRows, predPnl, evPnl, propGames, parlayRes, starts] = await Promise.all([
    sb("accuracy_by_confidence?select=*"),
    sb("prediction_accuracy?order=game_date.desc&limit=300&select=sport,game_date,home_team_name,away_team_name,win_prob,predicted_winner,actual_winner,winner_correct,pred_margin,actual_margin,margin_error,pred_total,actual_total,total_error,market_spread,market_total,spread_pick_correct,total_pick_correct"),
    evResultsRows().catch(() => []),
    evGradedPicks().catch(() => []),
    propResultsRows(),
    propLineGradesRows(),
    sb("prediction_pnl_daily?select=*").catch(() => []),
    sb("ev_pnl_daily?select=*").catch(() => []),
    sb("nfl_prop_pnl_by_game?select=*&order=commence_time.asc").catch(() => []),
    parlayResultsRows(),
    trackRecordStarts(),
  ]);
  // Archived (pre-restart) rows are cut from the raw tables; the record views are cut server-side.
  const pickKick = new Map((evPk || []).map((p) => [`${p.sport}|${p.game_pk}|${p.market}|${p.side}`, p.commence_time]));
  graded = (graded || []).filter((g) => inTrackRecord(starts, g.sport, g.game_date));
  evRes = (evRes || []).filter((r) => inTrackRecord(starts, r.sport, pickKick.get(`${r.sport}|${r.game_pk}|${r.market}|${r.side}`) || r.graded_at));
  propRes = (propRes || []).filter((r) => inTrackRecord(starts, r.sport || "nfl", r.commence_time));
  parlayRes = (parlayRes || []).filter((r) => inTrackRecord(starts, r.sport || "nfl", r.first_commence));
  const nflStart = starts.get("nfl");
  const restartNote = nflStart
    ? `NFL record restarted with ${nflStart.model_version === "nfl-sim-ml-v2" ? "the ML v2 model" : nflStart.model_version} on ${new Date(nflStart.starts_at).toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" })}; earlier NFL results are archived for comparison.`
    : "NFL predictions switched to the ML model on Sep 28, 2026; earlier games were graded against the previous model.";
  const dec = graded.filter((g) => g.actual_winner != null);
  const s = getSettings(), U = unitLabel(s);
  // Group the graded games by football week, per sport, for the drill-down.
  const weeksBySport = attachTrackWeeks(dec);
  const latest = [...dec].sort((a, b) => (a.game_date < b.game_date ? 1 : -1))[0];
  const defaultSport = (latest && weeksBySport[latest.sport]) ? latest.sport
    : (weeksBySport.nfl ? "nfl" : Object.keys(weeksBySport)[0]);
  window.__caTrack = { rows: dec, weeksBySport, defaultSport };
  const all = accSummary(graded);
  const hitPct = (key) => { const d = dec.filter((g) => g[key] != null); return d.length ? `${(d.filter((g) => g[key]).length / d.length * 100).toFixed(1)}%` : "—"; };
  const spreadPct = hitPct("spread_pick_correct"), totalPct = hitPct("total_pick_correct");
  const totMae = dec.filter((g) => g.total_error != null);
  const totMaeStr = totMae.length ? (totMae.reduce((s, g) => s + +g.total_error, 0) / totMae.length).toFixed(1) : "—";
  // cumulative winner-accuracy % over time (oldest -> newest)
  let n = 0, c = 0;
  const series = [...dec].reverse().map((g) => { n++; if (g.winner_correct) c++; return { value: c / n * 100, label: g.game_date }; });
  const chart = chartPath(series.map((r) => r.value));
  window.__cappingAlphaChartSeries = series;
  const tierRows = [...tiers]
    .sort((x, y) => (x.sport > y.sport ? 1 : x.sport < y.sport ? -1 : (y.conf_tier > x.conf_tier ? 1 : -1)))
    .map((t) => `<div><b>${String(t.sport).toUpperCase()} · ${t.conf_tier}%</b><span>${t.games} games</span><strong>${t.winner_pct}% ML</strong><em>${t.spread_ats_pct}% ATS · ${t.total_pick_pct}% O/U</em></div>`).join("")
    || `<div><b>—</b><span>no graded games yet</span></div>`;
  // League -> week drill-down (fills .track-week-filter / .track-week-body via
  // wireTrack after render) so the page shows one league + one week at a time.
  const LG = { cfb: "CFB", nfl: "NFL" };
  const leagues = ["cfb", "nfl"].filter((s) => (weeksBySport[s] || []).length);
  const leagueBtns = leagues.map((s) => `<button data-league="${s}">${LG[s]}</button>`).join("");
  const gradedTable = dec.length
    ? `<div class="track-league-filter">${leagueBtns}</div><div class="track-week-filter"></div><div class="track-week-body"></div>`
    : `<p style="opacity:.6">No graded predictions yet — accuracy posts after games settle.</p>`;
  return `<main><section class="page-heading"><div><p class="eyebrow">PREDICTION ACCURACY</p><h1>Accuracy record</h1><p>Every model prediction is graded against the final — moneyline (winner), spread (did the model's pick cover the closing line), and total (did the model's over/under lean beat the closing line).</p><div class="ev-settings-note">${restartNote}</div></div><div class="page-head-stat"><span>MONEYLINE ACCURACY</span><strong class="blue">${all.acc}</strong><small>${all.n} graded games</small></div></section><div class="ev-toggle"><button data-view="predictions" class="selected">Predictions accuracy</button><button data-view="ev">+EV picks</button></div><div data-evview="predictions"><section class="compact-stats">${stat('MONEYLINE (WINNER)', all.acc, `${all.record} · ${all.n} games`)}${stat('SPREAD (ATS)', spreadPct, 'model pick vs the line', 'blue')}${stat('TOTAL (O/U)', totalPct, `model lean vs the line · ±${totMaeStr} pts`, 'cyan')}${stat('AVG MARGIN ERROR', all.mae, 'points off the result')}</section>${pnlSection(`Profit tracker <span style="font-size:.6em;opacity:.6">${U} per bet</span>`, `What ${U} on every model pick would have made — moneyline (predicted winner), spread &amp; total picks — at the closing price (median across books). Spread/total with no captured price assume −110.`, predPnl, [["moneyline", "MONEYLINE"], ["spread", "SPREAD"], ["total", "TOTAL"]], s)}<section class="section chart-card"><div class="section-title"><div><h2>Moneyline accuracy</h2><p>Cumulative · all games</p></div><div class="chart-legend"><span></span>Accuracy %</div></div><div class="chart"><svg viewBox="0 0 1000 260" preserveAspectRatio="none"><defs><linearGradient id="fill" x1="0" x2="0" y1="0" y2="1"><stop stop-color="var(--blue)" stop-opacity=".22"/><stop offset="1" stop-color="var(--blue)" stop-opacity="0"/></linearGradient></defs>${chart}</svg>${chart ? "" : '<p style="opacity:.6;padding:20px">Chart fills in once graded results accumulate.</p>'}</div></section><section class="record-grid section"><article><h2>Accuracy by confidence</h2><div class="league-performance">${tierRows}</div></article></section><section class="section"><div class="section-title"><div><h2>Graded games by week</h2><p>Pick a league, then a week — moneyline vs the final, spread & total vs the closing line.</p></div></div>${gradedTable}</section></div><div data-evview="ev" hidden>${pnlSection(`+EV profit tracker <span style="font-size:.6em;opacity:.6">${U} per bet</span>`, `What ${U} on every graded +EV pick would have made, at the price it was flagged at · a parlay is one ${U} ticket.`, evPnl, [["moneyline", "+EV MONEYLINE"], ["spread", "+EV SPREAD"], ["prop", "+EV PROPS"], ["parlay", "+EV PARLAYS"]], s)}${gradedParlaysSection(parlayRes, s)}${evTrackSection(evRes, evPk)}${propAccuracySection(propGradeRows)}${propGamesSection(propGames, s)}${propTrackSection(propRes)}</div></main>`;
}

/* ── Render ───────────────────────────────────────────────────────────── */
const REFRESH_MS = 5 * 60 * 1000;  // auto-pull fresh Supabase data every 5 minutes

function injectStylesOnce() {  // one-time; re-renders must not keep appending <style> blocks
  if (document.getElementById("ca-styles")) return;
  document.head.insertAdjacentHTML("beforeend", `<style id="ca-styles">
    html,body{background:var(--bg)}
    .page-shell{position:relative;z-index:1}
    /* Legacy section base, ported from the retired styles.css onto the light tokens. */
    .blue{color:var(--blue)}.cyan{color:var(--blue)}
    .eyebrow{font-size:11px;letter-spacing:.14em;font-weight:700;color:var(--blue);margin:0 0 10px}
    .section{margin-top:28px}
    .section h2{font:700 24px/1.2 var(--serif);margin:0}
    .section-title{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:14px}
    .section-title>a{font-size:12px;font-weight:600;color:var(--blue);text-decoration:none}
    .section-title p{font-size:13px;color:var(--muted);margin:5px 0 0}
    .stat-grid,.compact-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px}
    .stat-grid article,.compact-stats article,.edge-card,.chart-card,.record-grid article{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:18px 20px;color:var(--ink)}
    .stat-grid span,.compact-stats span,.page-head-stat span{display:block;color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.04em}
    .stat-grid strong,.compact-stats strong{display:block;font:700 24px var(--sans);margin-top:8px;color:var(--ink)}
    .stat-grid small,.compact-stats small,.page-head-stat small{display:block;font-size:11px;color:var(--muted);margin-top:5px}
    .edge-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
    .matchup{display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid var(--line);padding-bottom:13px}
    .matchup h3{font-size:15px;margin:0 0 3px}
    .matchup p{font-size:11px;color:var(--muted);margin:0}
    .matchup>span,.ev{align-self:start;background:var(--blue-tint);color:var(--blue);border-radius:4px;padding:4px 7px;font-size:11px;font-weight:600;white-space:nowrap}
    .market{position:relative;padding:14px 64px 3px 0;display:grid;grid-template-columns:74px 1fr;grid-template-rows:16px 18px 12px}
    .market small{font-size:10px;color:var(--muted)}
    .market b{font-size:12px}
    .market strong{position:absolute;right:0;top:13px;font-size:12px;font-weight:700}
    .market i{display:block;grid-column:1 / span 2;height:4px;background:var(--blue);border-radius:2px;align-self:end}
    .market p{position:absolute;right:64px;bottom:-1px;margin:0;color:var(--ink);font-size:10px}
    .market p span{color:var(--muted)}
    .market em{position:absolute;right:0;bottom:-2px;font-size:10px;color:var(--muted);text-align:right;font-style:normal}
    .market u{color:var(--blue);text-decoration:none}
    .table-wrap{border:1px solid var(--line);border-radius:var(--radius);overflow:auto;background:var(--card)}
    .table-wrap table{border-collapse:collapse;width:100%;min-width:800px}
    .table-wrap th{text-align:left;font-size:11px;font-weight:600;color:var(--muted);padding:12px 16px;border-bottom:1px solid var(--line);background:#FBFAF6}
    .table-wrap td{padding:11px 16px;font-size:12px;color:var(--ink);border-bottom:1px solid #F0ECE3}
    /* boxscore tables are bare <table class="ev-table box"> in .box-col (no .table-wrap) */
    .ev-table.box{border-collapse:collapse;width:100%}
    .ev-table.box th{text-align:left;font-weight:600;color:var(--muted);border-bottom:1px solid var(--line);background:#FBFAF6}
    .ev-table.box td{color:var(--ink);border-bottom:1px solid #F0ECE3}
    .ev-table.box tbody tr:last-child td{border:0}
    .ev-table.box td b{color:var(--ink)}
    .table-wrap tbody tr:last-child td{border:0}
    .table-wrap tbody tr:hover{background:var(--bg)}
    .table-wrap td b{font-size:12px;color:var(--ink)}
    .table-wrap td small{display:block;margin-top:4px;font-size:10px;color:var(--muted)}
    .pick{color:var(--blue);font-weight:600}
    .win{background:var(--green-tint);color:var(--green)}
    .loss{background:var(--red-tint);color:var(--red)}
    .page-heading{display:flex;align-items:center;justify-content:space-between;gap:24px;border-bottom:1px solid var(--line);padding:20px 0 24px}
    .page-heading h1{font:700 40px/1.1 var(--serif);margin:0;letter-spacing:-.5px}
    .page-heading p:not(.eyebrow){color:var(--muted);line-height:1.6;font-size:15px;max-width:610px;margin:13px 0 0}
    .page-head-stat{text-align:right;padding:12px 0 12px 28px;border-left:1px solid var(--line);min-width:190px}
    .page-head-stat strong{display:block;font:700 30px var(--sans);margin-top:8px}
    .filter-bar{display:flex;gap:6px;flex-wrap:wrap}
    .filter-bar button{border:1px solid var(--line);background:var(--card);color:var(--muted);border-radius:6px;padding:7px 12px;font-size:12px;font-weight:600;cursor:pointer}
    .filter-bar button:hover,.filter-bar button.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--blue)}
    .chart-legend{font-size:11px;color:var(--muted)}
    .chart-legend span{display:inline-block;width:20px;height:3px;background:var(--blue);vertical-align:middle;margin-right:6px}
    .chart{height:300px;position:relative;padding:15px 0 28px 50px;background:repeating-linear-gradient(to bottom,transparent 0,transparent 61px,var(--line) 62px)}
    .chart svg{height:100%;width:100%}
    .axis{position:absolute;left:0;top:10px;bottom:31px;display:flex;flex-direction:column;justify-content:space-between}
    .axis i,.months{font-size:10px;color:var(--muted);font-style:normal}
    .months{display:flex;justify-content:space-between;position:absolute;left:50px;right:0;bottom:3px}
    .record-grid{display:grid;grid-template-columns:1.2fr .8fr;gap:14px}
    .league-performance{margin-top:14px}
    .league-performance>div{display:grid;grid-template-columns:70px 1fr 100px 90px;padding:14px 0;border-bottom:1px solid var(--line);font-size:12px;align-items:center}
    .league-performance>div:last-child{border:0}
    .league-performance span{color:var(--muted)}
    .league-performance strong{color:var(--ink)}
    .league-performance em{color:var(--blue);font-style:normal;text-align:right}
    .recent-grades{margin-top:10px}
    .recent-grades p{margin:0;padding:12px 0;border-bottom:1px solid var(--line);font-size:12px}
    .recent-grades p:last-child{border:0}
    .recent-grades span{display:inline-block;border-radius:4px;padding:3px 6px;margin-right:8px;font-size:10px;font-weight:700}
    .recent-grades b{float:right;color:var(--blue)}
    @media(max-width:900px){.edge-grid,.record-grid{grid-template-columns:1fr}.stat-grid,.compact-stats{grid-template-columns:repeat(2,1fr)}.page-heading{align-items:flex-start;flex-direction:column}.page-head-stat{border-left:0;border-top:1px solid var(--line);padding:16px 0 0;text-align:left;width:100%}}
    @media(max-width:620px){.stat-grid,.compact-stats{grid-template-columns:1fr}.section-title{align-items:flex-start;flex-direction:column}.filter-bar{width:100%}.page-heading h1{font-size:32px}.chart{height:230px;padding-left:38px}.months{left:38px}.league-performance>div{grid-template-columns:50px 1fr;gap:7px}.league-performance strong,.league-performance em{text-align:left}}
    .tlogo{height:18px!important;width:18px!important;max-width:18px;max-height:18px;vertical-align:middle;margin-right:5px;object-fit:contain;display:inline-block;flex:none}
    td.pick .tlogo,.pick .tlogo{height:16px!important;width:16px!important;margin-right:4px}
    .edge-card h3 .tlogo{height:20px!important;width:20px!important}
    td.proj{white-space:nowrap}
    td.proj .tlogo{height:20px!important;width:20px!important;max-width:20px;max-height:20px;margin:0 6px}
    .market-filters.filter-bar{display:flex;gap:14px;align-items:center;overflow-x:auto;padding:26px 4px 8px;scrollbar-width:thin}
    .market-filters.filter-bar button{flex:0 0 auto;border:2px solid var(--line);background:transparent;color:var(--muted);border-radius:999px;padding:12px 24px;font-size:16px;font-weight:600;line-height:1.75;cursor:pointer;transition:.15s}
    .market-filters.filter-bar button:hover,.market-filters.filter-bar button.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--blue)}
    .market-filters.filter-bar button.selected{box-shadow:inset 0 0 0 1px rgba(37,99,235,.18)}
    @media(max-width:620px){.market-filters.filter-bar{gap:9px;padding-top:20px}.market-filters.filter-bar button{padding:8px 17px;font-size:14px}}
    .league-filter{display:flex;gap:10px;padding:8px 4px 18px;flex-wrap:wrap}
    .league-filter button{border:2px solid var(--line);background:transparent;color:var(--muted);border-radius:999px;padding:8px 18px;font-weight:600;cursor:pointer;transition:.15s}
    .league-filter button:hover,.league-filter button.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--blue)}
    .track-league-filter{display:flex;gap:10px;padding:4px 0 12px;flex-wrap:wrap}
    .track-league-filter button{border:2px solid var(--line);background:transparent;color:var(--muted);border-radius:999px;padding:8px 22px;font-weight:700;cursor:pointer;transition:.15s}
    .track-league-filter button:hover,.track-league-filter button.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--blue)}
    .track-week-filter,.pg-weeks{display:flex;gap:7px;padding:2px 0 14px;flex-wrap:wrap}
    .track-week-filter button,.pg-weeks button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:6px 13px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
    .track-week-filter button:hover,.pg-weeks button:hover{border-color:var(--blue)}
    .track-week-filter button.selected,.pg-weeks button.selected{background:var(--blue-tint);border-color:var(--blue);color:var(--ink)}
    .week-record{margin:2px 0 12px;color:var(--ink);font-size:13px}.week-record b{color:var(--ink)}
    .ev-toggle{display:flex;gap:8px;padding:6px 4px 4px}
    .ev-toggle button{border:2px solid var(--line);background:transparent;color:var(--muted);border-radius:999px;padding:8px 22px;font-weight:700;cursor:pointer;transition:.15s}
    .ev-toggle button:hover,.ev-toggle button.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--blue)}
    .ev-table td small{display:block;opacity:.55;font-size:11px}
    .ev-good{color:var(--green);font-weight:700}
    .ev-bad{color:var(--red);font-weight:700}
    .ev-book{color:var(--blue);font-size:12px}
    .sim-note{margin-top:10px;color:var(--muted);font-size:12px}
    .model-tag{margin-left:6px;padding:2px 7px;border:1px solid var(--line);border-radius:999px;color:var(--blue);opacity:1;white-space:nowrap}
    .pred-block .compact-stats .pred-actual{color:var(--ink)}
    .pred-block .pred-actual b{color:var(--ink)}
    .pred-block .pred-actual .call-ok{color:var(--green);font-weight:700}
    .pred-block .pred-actual .call-miss{color:var(--red);font-weight:700}
    .final-line{margin-top:8px;font-size:14px;color:var(--ink);font-weight:600}
    .final-line b{color:var(--ink)}
    .final-line .call-ok{color:var(--green);font-weight:700}
    .final-line .call-miss{color:var(--red);font-weight:700}
    .pnl-pos{color:var(--green)!important}.pnl-neg{color:var(--red)!important}
    .pnl-chart{position:relative;height:180px;margin-top:14px}
    .pnl-chart svg{width:100%;height:150px;display:block}
    .pnl-chart small{display:block;opacity:.75;font-size:12px;margin-top:6px}
    .pnl-key{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
    .pnl-key.wag{background:var(--blue)}.pnl-key.up{background:var(--green)}.pnl-key.down{background:var(--red)}
    .pg-game{border:1px solid var(--line);border-radius:12px;margin-bottom:8px;background:var(--card)}
    .pg-head{all:unset;box-sizing:border-box;display:flex;width:100%;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;cursor:pointer}
    .pg-head span{display:flex;flex-direction:column;gap:3px}
    .pg-head small{opacity:.65;font-size:12px}
    .pg-head strong{font-size:16px;white-space:nowrap}
    .pg-game.open .pg-head{border-bottom:1px solid var(--line)}
    .pg-detail{padding:4px 8px 10px}
    .prop-group{margin-top:16px}.prop-group:first-of-type{margin-top:4px}
    .prop-group h3{font-size:15px;margin:0 0 8px;color:var(--ink);letter-spacing:.2px}
    .prop-group h3 small{opacity:.5;font-weight:600;font-size:12px;margin-left:8px}
    .ev-settings-note{margin-top:12px;font-size:12px;color:var(--muted)}
    .ev-settings-note a{color:var(--blue);text-decoration:none}
    .ev-settings-note a:hover{text-decoration:underline}
    .ev-toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
    .ev-filters{display:flex;flex-direction:column}
    .ev-filters .ev-filter{padding:2px 0 10px}
    .ev-filters [data-evgroup="market"] button{padding:6px 16px;font-size:13px}
    .ev-sort{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;font-weight:600;padding:4px 0 12px}
    .ev-sort select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px;font:inherit;font-weight:600;cursor:pointer}
    .ev-sort select:hover,.ev-sort select:focus{border-color:var(--blue);outline:none}
    .prop-proj .prop-actual{display:block;font-size:11px;font-weight:700;margin-top:2px}
    .prop-proj .prop-actual.hit{color:var(--green)}.prop-proj .prop-actual.miss{color:var(--red)}
    .prop-proj .prop-line{display:block;font-size:10.5px;opacity:.7;margin-top:2px}
    .prop-proj .prop-line b{opacity:1;color:var(--blue)}
    .team-totals{display:flex;align-items:center;justify-content:center;gap:22px;margin:6px 0 16px}
    .team-totals .tt{display:flex;align-items:center;gap:12px}
    .team-totals .tt-name{color:var(--ink);font-weight:600;font-size:14px;max-width:130px}
    .team-totals .tt-score{font-size:40px;font-weight:800;line-height:1}
    .team-totals .tt-vs{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}
    .sim-tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
    .sim-tab{padding:7px 13px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);font-size:12px;font-weight:600;cursor:pointer}
    .sim-tab.active{background:var(--blue-tint);border-color:var(--blue);color:var(--ink)}
    .sim-controls{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;margin-bottom:10px}
    .sim-controls input,.sim-controls select{background:var(--card);border:1px solid var(--line);border-radius:6px;color:var(--ink);padding:6px 8px;font-size:13px;min-width:74px}
    .hist{display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px}
    .prob-line{margin-top:10px;color:var(--ink);font-size:13px}
    .prob-line .under{color:var(--blue)}.prob-line .at{color:var(--amber)}.prob-line .over{color:var(--red)}
    .box-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .box-col h3{margin:0 0 8px;font-size:14px;color:var(--ink)}
    .ev-table.box{margin-bottom:12px}
    .ev-table.box th,.ev-table.box td{padding:6px 8px;font-size:12px}
    .box-row{cursor:pointer}.box-row:hover{background:var(--bg)}
    #player-dist:empty{display:none}
    .player-dist-inner{margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
    .pd-close{margin-left:auto;background:var(--card);border:1px solid var(--line);border-radius:6px;color:var(--ink);padding:5px 12px;font-size:12px;cursor:pointer}
    @media (max-width:640px){.box-grid{grid-template-columns:1fr}.team-totals .tt-score{font-size:32px}.team-totals{gap:12px}}
    .ev-empty{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px 22px;color:var(--muted)}
    .ev-empty b{color:var(--ink);font-size:16px;display:block;margin-bottom:6px}
    table td .win{color:var(--green);font-weight:700}table td .loss{color:var(--red);font-weight:700}
    .edge-grid.edge-carousel{display:flex;grid-template-columns:none;gap:14px;overflow-x:auto;overflow-y:hidden;padding:2px 1px 14px;scroll-snap-type:x mandatory;scrollbar-color:var(--line) transparent}
    .edge-grid.edge-carousel .edge-card{flex:0 0 calc((100% - 28px) / 3);min-width:0;scroll-snap-align:start}
    @media(max-width:900px){.edge-grid.edge-carousel .edge-card{flex-basis:calc((100% - 14px) / 2)}}
    @media(max-width:620px){.edge-grid.edge-carousel .edge-card{flex-basis:86%}}
    .parlay-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}
    .parlay-card{border:1px solid var(--line);border-radius:12px;background:var(--card);overflow:hidden}
    .pc-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:12px 16px;border-bottom:1px solid var(--line)}
    .pc-head>span,.pc-leg>span{display:flex;flex-direction:column;gap:3px;min-width:0}
    .pc-head b{color:var(--ink);font-size:15px}
    .pc-head small,.pc-leg small{opacity:.65;font-size:12px}
    .pc-head strong{font-size:16px;white-space:nowrap}
    .pc-kelly{padding:8px 16px;border-bottom:1px solid var(--line);font-size:12px;font-weight:600;color:var(--muted)}
    .pc-legs{list-style:none;margin:0;padding:2px 16px 8px}
    .pc-leg{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid var(--line)}
    .pc-leg:last-child{border-bottom:0}
    .pc-leg b{color:var(--ink);font-size:13px}
    .pc-leg .pc-odds{align-items:flex-end;text-align:right;flex:none}
    .pc-mark{display:inline-block;width:14px;margin-right:6px;font-weight:700;color:var(--muted)}
    .pc-mark.win{color:var(--green);background:none}.pc-mark.loss{color:var(--red);background:none}
    .pc-res{display:inline-block;margin-left:6px;font-size:10px;font-weight:700;letter-spacing:.04em;padding:2px 7px;border-radius:999px;vertical-align:middle;background:var(--line);color:var(--muted)}
    .pc-res.win{background:var(--green-tint);color:var(--green)}.pc-res.loss{background:var(--red-tint);color:var(--red)}
    @media(min-width:901px){.pnl-body>.compact-stats.pnl-stats{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}}  /* 5 cards with +EV PARLAYS */
    .parlay-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
    .parlay-head b{color:var(--ink);font-size:14px}
    .parlay-price{color:var(--green);font-weight:700;font-size:16px}
    .parlay-price small{color:var(--blue);font-weight:700;font-size:11px;margin-left:4px}
    .parlay-legs{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:7px}
    .parlay-legs li{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:var(--ink);border-bottom:1px solid var(--line);padding-bottom:6px}
    .parlay-legs li:last-child{border-bottom:0;padding-bottom:0}
    .ev-leg-price{color:var(--muted);font-weight:600}
    .parlay-foot{display:flex;justify-content:space-between;align-items:baseline;margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}
    .parlay-foot small{color:var(--muted);font-size:11px}
    .game-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(184px,1fr));gap:10px;margin-top:6px}
    .game-card{display:flex;flex-direction:column;gap:9px;padding:14px 15px;border:1px solid var(--line);border-radius:14px;background:var(--card);color:var(--ink);text-decoration:none;transition:.15s}
    .game-card:hover{border-color:var(--blue);background:var(--blue-tint)}
    .game-card .gc-row{display:flex;align-items:center;justify-content:space-between;gap:8px}
    .game-card .gc-team{display:flex;align-items:center;gap:7px;min-width:0}
    .game-card .gc-team b{font-weight:700;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .game-card .gc-team .tlogo{width:21px;height:21px;flex:none;object-fit:contain}
    .game-card .gc-ml{display:flex;align-items:center;gap:6px;color:var(--ink);font-weight:600;font-size:13px;white-space:nowrap;font-variant-numeric:tabular-nums;flex:none}
    .game-card .bk-logo{width:16px;height:16px;border-radius:4px;flex:none;background:#fff}
    .game-card .bk-txt{font-size:9px;font-weight:800;letter-spacing:.02em;padding:2px 4px;border-radius:4px;background:var(--line);color:var(--muted);flex:none}
    .game-card .gc-model{color:var(--muted)}
    .game-card .gc-model small{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;opacity:.8}
    .game-card .gc-time{color:var(--muted);font-size:12px;margin-top:1px;padding-top:8px;border-top:1px solid var(--line)}
    @media(max-width:620px){.game-grid{grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:8px}.game-card{padding:12px 13px}.game-card .gc-team b{font-size:13px}}
    .back-link{display:inline-block;margin-bottom:14px;color:var(--blue);text-decoration:none;font-size:13px;font-weight:600}
    .back-link:hover{text-decoration:underline}
    .game-hero{padding:24px 4px 8px}
    .game-hero h1{margin:6px 0 4px;font-size:26px}
    .game-hero .game-score{font-size:34px;font-weight:800;color:var(--ink);display:flex;align-items:center;gap:6px}
    .game-hero .game-score .tlogo{height:30px!important;width:30px!important;max-width:30px;max-height:30px}
    .game-hero .game-winner{color:var(--muted);font-size:14px}
    .prop-proj td small{display:block;opacity:.55;font-size:11px}
    .prop-ev{display:inline-block;margin-left:4px;font-size:10px;font-weight:700;padding:1px 6px;border-radius:999px;background:var(--green-tint);color:var(--green);vertical-align:middle}
    .sim-tag{font-size:.6em;opacity:.6;font-weight:600}
    .sim-tag.model-tag{opacity:1}
    .trends-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
    .trend-col{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--tc,var(--blue));border-radius:8px;padding:14px 16px}
    .trend-col h3{display:flex;align-items:center;margin:0 0 10px;font-size:15px;color:var(--tc,var(--ink))}
    .trend-list,.trend-sits{list-style:none;margin:0;padding:0}
    .trend-row{display:flex;justify-content:space-between;gap:10px;padding:7px 0;border-bottom:1px solid var(--line);font-size:13px}
    .trend-row:last-child{border-bottom:0}
    .trend-row>span{color:var(--muted)}
    .trend-row b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums;white-space:nowrap}
    .trend-sits{display:flex;flex-direction:column;gap:6px}
    .trend-list+.trend-sits{margin-top:10px;padding-top:8px;border-top:1px solid var(--line)}
    .trend-sit{font-size:12px;color:var(--ink);line-height:1.45}
    .trend-hot{background:var(--amber-tint);box-shadow:inset 3px 0 0 var(--amber);padding-left:8px;border-radius:4px}
    .trend-hot b,.trend-sit.trend-hot{color:var(--amber)}
    .trend-pos{color:var(--green)}.trend-neg{color:var(--red)}
    .trend-none,.trend-empty{opacity:.6;margin:0}
    .trend-foot{margin:12px 0 0;font-size:11px;color:var(--muted)}
    @media(max-width:620px){.trends-grid{grid-template-columns:1fr}}
    /* Team context: matchup grades, history, power rankings (descriptive — not a pick) */
    .ctx-np{font-style:normal;color:var(--amber);white-space:nowrap}
    .ctx-early{display:inline-block;margin-left:8px;padding:2px 7px;border-radius:999px;background:var(--amber-tint);border:1px solid var(--amber);color:var(--amber);font-size:10px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;vertical-align:middle}
    .grade-chip{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:6px;font-size:13px;font-weight:800;line-height:1;text-align:center;flex:none}
    .grade-chip.big{width:44px;height:44px;border-radius:10px;font-size:24px}
    /* A green · B yellow · C orange · D/F red */
    .grade-chip.g-A,.mu-bar i.g-A{background:#22a45d;color:#04160c}
    .grade-chip.g-B,.mu-bar i.g-B{background:#e8c22e;color:#1c1602}
    .grade-chip.g-C,.mu-bar i.g-C{background:#ec8a2c;color:#1f0e01}
    .grade-chip.g-D,.mu-bar i.g-D,.grade-chip.g-F,.mu-bar i.g-F{background:#d9423f;color:#fff}
    .grade-chip.g-na,.mu-bar i.g-na{background:var(--line);color:var(--muted)}
    .mu-card h3{flex-wrap:wrap}
    .mu-vs{margin:-6px 0 12px;font-size:12px;color:var(--muted)}
    .mu-overall{display:flex;align-items:center;gap:12px;padding-bottom:10px;margin-bottom:6px;border-bottom:1px solid var(--line)}
    .mu-overall div b{display:block;color:var(--ink);font-size:14px}.mu-overall small{display:block;color:var(--muted);font-size:12px;margin-top:2px}
    .mu-unit{display:grid;grid-template-columns:44px 24px 1fr 76px;align-items:center;gap:10px;padding:7px 0;font-size:13px}
    .mu-unit>span{color:var(--muted)}.mu-unit small{color:var(--muted);font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
    .mu-bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden}.mu-bar i{display:block;height:100%;border-radius:4px}
    .ctx-power{display:block;margin:-4px 0 10px;font-size:12px;color:var(--ink)}
    .hist-wrap{border:0;border-radius:0;background:none;margin:0 0 10px;-webkit-overflow-scrolling:touch}
    .hist-tbl{min-width:0!important;width:100%;margin:0}
    .hist-tbl th{padding:6px 8px;font-size:10px}.hist-tbl th:first-child,.hist-tbl td:first-child{padding-left:0}
    .hist-tbl td{padding:7px 8px;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
    .hist-tbl td small{display:inline;margin:0 0 0 6px;color:var(--muted)}
    .hist-tbl td.hist-na{color:var(--muted)}
    .ctx-chips{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:4px 0 10px}
    .ctx-chips>small,.hist-lbl{display:block;color:var(--muted);font-size:10px;font-weight:600;letter-spacing:.08em;margin-right:4px}
    .hist-lbl{margin:8px 0 2px}
    .ctx-chip{padding:3px 9px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--ink);font-size:12px;font-weight:600;font-variant-numeric:tabular-nums}
    .ctx-chip.hot{border-color:var(--amber);background:var(--amber-tint);color:var(--amber)}
    .rk-up{color:var(--green);font-weight:700}.rk-down{color:var(--red);font-weight:700}.rk-flat{color:var(--muted)}
    .rk-new{display:inline-block;padding:1px 6px;border-radius:4px;background:var(--blue-tint);color:var(--blue);font-size:10px;font-weight:800;letter-spacing:.04em}
    .rk-bar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px;padding:22px 0 14px}
    .rk-sport{display:flex;gap:10px}
    .rk-sport a{border:2px solid var(--line);color:var(--muted);border-radius:999px;padding:8px 22px;font-weight:700;text-decoration:none;transition:.15s}
    .rk-sport a:hover,.rk-sport a.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--blue)}
    .rk-conf{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;font-weight:600}
    .rk-conf select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:13px}
    .rk-table th.rk-th{cursor:pointer;user-select:none;white-space:nowrap}
    .rk-table th.rk-th:hover,.rk-table th.sorted{color:var(--blue)}
    .rk-table td{white-space:nowrap;font-variant-numeric:tabular-nums}
    .rk-team{display:inline-flex;align-items:center}
    .rk-empty{text-align:center;color:var(--muted);padding:24px}
    @media(max-width:620px){.mu-unit{grid-template-columns:40px 24px 1fr 64px;gap:8px}.rk-bar{flex-direction:column;align-items:stretch}.rk-conf select{flex:1}}
    .set-warn{margin:22px 0 0;padding:11px 15px;border:1px solid var(--line);border-left:3px solid var(--amber);border-radius:8px;background:var(--amber-tint);color:var(--ink);font-size:13px}
    .set-card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:20px 22px}
    .set-card .section-title{margin-bottom:16px}
    .set-note{margin:14px 0 0;color:var(--muted);font-size:12px;line-height:1.55;max-width:640px}
    .set-note.set-books-min{color:var(--amber);font-weight:600}
    .set-actions,.set-chips{display:flex;gap:8px;flex-wrap:wrap}
    .set-actions button,.set-chips button,.set-seg button,.set-reset{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
    .set-actions button:hover,.set-chips button:hover,.set-seg button:hover{border-color:var(--blue);color:var(--ink)}
    .set-chips button.selected,.set-seg button.selected{background:var(--blue-tint);border-color:var(--blue);color:var(--ink)}
    .set-books{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
    .set-book{position:relative;display:flex;align-items:center;gap:10px;min-width:0;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--muted);font-size:14px;font-weight:600;text-align:left;cursor:pointer;transition:.15s}
    .set-book:hover{border-color:var(--blue)}
    .set-book.selected{border-color:var(--blue);background:var(--blue-tint);color:var(--ink);box-shadow:inset 0 0 0 1px var(--blue)}
    .set-book span{min-width:0;line-height:1.25;padding-right:16px}
    .set-book .bk-logo{width:28px;height:28px;border-radius:6px;background:#fff;flex:none;opacity:.55;transition:.15s}
    .set-book .bk-txt{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:6px;background:var(--line);color:var(--muted);font-size:9px;font-weight:800;flex:none}
    .set-book.selected .bk-logo{opacity:1}
    .set-book i{position:absolute;top:7px;right:8px;display:flex;align-items:center;justify-content:center;width:17px;height:17px;border-radius:50%;background:var(--blue);color:#fff;font-size:10px;font-style:normal;font-weight:800;opacity:0;transition:.15s}
    .set-book.selected i{opacity:1}
    .set-row{display:flex;gap:16px 28px;align-items:flex-end;flex-wrap:wrap}
    .set-field{display:flex;flex-direction:column;gap:7px}
    .set-field>small{color:var(--muted);font-size:10px;font-weight:600;letter-spacing:.08em}
    .set-money{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:8px;background:var(--card);padding:0 12px;transition:.15s}
    .set-money:focus-within{border-color:var(--blue)}
    .set-money>span{color:var(--muted);font-weight:700;margin-right:4px}
    .set-money input{width:120px;background:transparent;border:0;outline:none;color:var(--ink);font-size:16px;font-weight:700;padding:10px 0;font-variant-numeric:tabular-nums}
    .set-seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
    .set-seg button{border:0;border-radius:0;border-right:1px solid var(--line);background:var(--card)}
    .set-seg button:last-child{border-right:0}
    .set-range{display:flex;align-items:center;gap:18px;max-width:560px}
    .set-range input{flex:1;min-width:0;accent-color:var(--blue);cursor:pointer}
    .set-range output{min-width:64px;text-align:right;font-size:22px;font-weight:800;color:var(--blue);font-variant-numeric:tabular-nums}
    .set-reset:hover{border-color:var(--red);color:var(--red)}
    .set-saved{position:fixed;right:20px;bottom:20px;z-index:5;padding:9px 15px;border:1px solid var(--green);border-radius:8px;background:var(--green-tint);color:var(--green);font-size:13px;font-weight:700;opacity:0;transform:translateY(6px);transition:.2s;pointer-events:none}
    .set-saved.show{opacity:1;transform:none}
    @media(max-width:620px){.set-card{padding:16px}.set-books{gap:8px}.set-book{gap:8px;padding:11px 10px;font-size:13px}.set-row{flex-direction:column;align-items:stretch}.set-money input{width:100%}.set-seg{display:flex}.set-seg button{flex:1;padding:8px 6px}.set-range{gap:12px}}
  </style>`);
}

async function render() {
  const shell = document.querySelector(".page-shell");
  // preserve scroll + active filter so the 5-min refresh isn't disruptive
  const scrollY = window.scrollY;
  const selMarket = document.querySelector(".filter-bar button.selected")?.dataset.filter;
  const selLeague = document.querySelector(".league-filter button.selected")?.dataset.league;
  // Preserve the track drill-down (league + week) across the periodic re-render.
  const selTrackLeague = document.querySelector(".track-league-filter button.selected")?.dataset.league;
  const selTrackWeekRaw = document.querySelector(".track-week-filter button.selected")?.dataset.week;
  const selTrackWeek = selTrackWeekRaw ? +selTrackWeekRaw : null;
  try {
    if (CONFIG.SUPABASE_URL.includes("YOUR-PROJECT")) {
      throw new Error("Set CONFIG.SUPABASE_URL and CONFIG.SUPABASE_ANON_KEY at the top of app.js.");
    }
    let body;
    if (page === "dashboard") body = await buildDashboard();
    else if (page === "track") body = await buildTrack();
    else if (page === "game") body = await buildGamePage();
    else if (page === "ev") body = await buildEvPage();
    else if (page === "settings") body = buildSettings();
    else if (page === "rankings") body = await buildRankings();
    else body = await buildLeague(page); // cfb / nfl
    shell.innerHTML = siteHeader(page === "rankings" || page === "settings" ? "" : page) + `<div class="ca-page">${body}</div>` + footer(); wireShell();
    if (page === "dashboard") wireDashboard();
    if (page === "game") wireGamePage();
    if (page === "track") { wireTrack(selTrackLeague, selTrackWeek); wirePropGames(); wirePnl(); }
    if (page === "ev") wireEvPage2();
    if (page === "settings") wireSettings();
    if (page === "rankings") wireRankings();
    const chartSvg = document.querySelector(".chart svg");
    const chartSeries = window.__cappingAlphaChartSeries || [];
    if (chartSvg && chartSeries.length >= 2) {
      const chartWrap = chartSvg.closest(".chart");
      const values = chartSeries.map((point) => point.value);
      const max = Math.max(...values, 0), min = Math.min(...values, 0), range = max - min || 1;
      const svgNode = (tag, attrs) => {
        const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
      };
      const guide = svgNode("line", { x1: 0, x2: 0, y1: 0, y2: 260, stroke: "#2563EB", "stroke-width": 1, "stroke-dasharray": "4 4", visibility: "hidden" });
      const marker = svgNode("circle", { cx: 0, cy: 0, r: 6, fill: "#2563EB", stroke: "#fff", "stroke-width": 2, visibility: "hidden" });
      chartSvg.append(guide, marker);
      const tooltip = document.createElement("div");
      tooltip.className = "chart-tooltip";
      tooltip.hidden = true;
      chartWrap.appendChild(tooltip);
      const updateChartHover = (event) => {
        const rect = chartSvg.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
        const index = Math.round(ratio * (chartSeries.length - 1));
        const point = chartSeries[index];
        const x = index / (chartSeries.length - 1) * 1000;
        const y = 250 - ((point.value - min) / range) * 240;
        guide.setAttribute("x1", x); guide.setAttribute("x2", x); guide.setAttribute("visibility", "visible");
        marker.setAttribute("cx", x); marker.setAttribute("cy", y); marker.setAttribute("visibility", "visible");
        tooltip.innerHTML = `<strong>${point.value.toFixed(1)}%</strong><span>${point.label || ""}</span>`;
        tooltip.style.left = `${index / (chartSeries.length - 1) * 100}%`;
        tooltip.style.top = `${y / 260 * 100}%`;
        tooltip.hidden = false;
      };
      const clearChartHover = () => { guide.setAttribute("visibility", "hidden"); marker.setAttribute("visibility", "hidden"); tooltip.hidden = true; };
      chartSvg.addEventListener("pointermove", updateChartHover);
      chartSvg.addEventListener("pointerleave", clearChartHover);
      chartSvg.addEventListener("pointerdown", updateChartHover);
      document.head.insertAdjacentHTML("beforeend", `<style>
        .chart svg{cursor:crosshair}.chart-tooltip{position:absolute;z-index:2;transform:translate(-50%,-112%);min-width:112px;padding:8px 10px;border:1px solid #3d82d0;border-radius:5px;background:#101b28;color:#e9f3ff;box-shadow:0 8px 24px rgba(0,0,0,.3);pointer-events:none;text-align:center;font-size:11px}.chart-tooltip strong{display:block;color:#69a8ff;font-size:13px}.chart-tooltip span{display:block;color:#98a8b8;margin-top:3px;font-size:10px}
      </style>`);
    }

    const filterBar = document.querySelector(".filter-bar");
    if (filterBar) {
      const filtersBySport = {
        mlb: ["All", "Moneyline", "Spread", "Total", "Hits", "Total Bases", "Hits + Runs + RBIs", "Strikeouts", "Hits Allowed", "Outs Recorded"],
        nfl: ["All", "Moneyline", "Spread", "Total", "Pass Yards", "Pass TDs", "Rush Yards", "Receiving Yards", "Receptions"],
        nba: ["All", "Moneyline", "Spread", "Total", "Points", "Rebounds", "Assists", "3PT Made", "Steals + Blocks"],
      };
      filterBar.classList.add("market-filters");
      filterBar.innerHTML = (filtersBySport[page] || ["All"]).map((label, index) =>
        `<button class="${index === 0 ? "selected" : ""}" data-filter="${label.toLowerCase()}">${label}</button>`
      ).join("");
      filterBar.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => {
        filterBar.querySelectorAll("button").forEach((item) => item.classList.remove("selected"));
        button.classList.add("selected");
        const filter = button.dataset.filter;
        document.querySelectorAll("table tbody tr").forEach((row) => {
          const market = row.cells[1]?.textContent.trim().toLowerCase();
          row.hidden = filter !== "all" && market !== filter;
        });
      }));
    }

    const leagueBar = document.querySelector(".league-filter");
    if (leagueBar) {
      leagueBar.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        leagueBar.querySelectorAll("button").forEach((x) => x.classList.remove("selected"));
        b.classList.add("selected");
        const lg = b.dataset.league;
        document.querySelectorAll("table tbody tr[data-league]").forEach((row) => {
          row.hidden = lg !== "all" && row.dataset.league !== lg;
        });
      }));
    }

    const edgeCarousel = document.querySelector(".edge-grid");
    if (edgeCarousel) edgeCarousel.classList.add("edge-carousel");

    // Predictions ↔ +EV toggle (used on league pages AND the track page):
    // show/hide containers marked [data-evview]; persist the choice.
    const evToggleBar = document.querySelector(".ev-toggle");
    if (evToggleBar) {
      const views = document.querySelectorAll("[data-evview]");
      const showView = (view) => {
        if (!views.length) return;
        views.forEach((el) => { el.hidden = el.dataset.evview !== view; });
        evToggleBar.querySelectorAll("button").forEach((x) =>
          x.classList.toggle("selected", x.dataset.view === view));
        try { localStorage.setItem("cappingAlphaLeagueView", view); } catch (_) {}
      };
      evToggleBar.querySelectorAll("button").forEach((b) =>
        b.addEventListener("click", () => showView(b.dataset.view)));
      let saved = "predictions";
      try { saved = localStorage.getItem("cappingAlphaLeagueView") || "predictions"; } catch (_) {}
      showView(saved);
    }

    // re-apply the filter the user had selected before the refresh
    if (selMarket && filterBar) {
      const b = [...filterBar.querySelectorAll("button")].find((x) => x.dataset.filter === selMarket);
      if (b) b.click();
    }
    if (selLeague && leagueBar) {
      const b = [...leagueBar.querySelectorAll("button")].find((x) => x.dataset.league === selLeague);
      if (b) b.click();
    }
    window.scrollTo(0, scrollY);
  } catch (e) {
    shell.innerHTML = siteHeader("") + `<div class="ca-page"><main><section class="section"><div class="section-title"><h2>Couldn’t load data</h2></div><p style="opacity:.7">${e.message}</p></section></main></div>` + footer(); wireShell();
    console.error(e);
  }
}

/* ── Animated blue/black gradient background (WebGL, all pages) ─────────────
   Vanilla port of the Velaris simplex-noise shader: a fixed full-viewport
   canvas mounted behind all content. Degrades silently to the CSS fallback
   background (see injectStylesOnce) when WebGL is unavailable. */
const BG_VERT = `attribute vec2 position;varying vec2 vUv;void main(){vUv=position*0.5+0.5;gl_Position=vec4(position,0.0,1.0);}`;
const BG_FRAG = `precision highp float;varying vec2 vUv;
uniform vec2 u_resolution;uniform float u_time;uniform float u_grain;uniform vec3 u_colors[4];uniform vec3 u_bg;
vec3 permute(vec3 x){return mod(((x*34.0)+1.0)*x,289.0);}
float snoise(vec2 v){const vec4 C=vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
vec2 i=floor(v+dot(v,C.yy));vec2 x0=v-i+dot(i,C.xx);vec2 i1=(x0.x>x0.y)?vec2(1.0,0.0):vec2(0.0,1.0);
vec4 x12=x0.xyxy+C.xxzz;x12.xy-=i1;i=mod(i,289.0);
vec3 p=permute(permute(i.y+vec3(0.0,i1.y,1.0))+i.x+vec3(0.0,i1.x,1.0));
vec3 m=max(0.5-vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.0);m=m*m;m=m*m;
vec3 x=2.0*fract(p*C.www)-1.0;vec3 h=abs(x)-0.5;vec3 ox=floor(x+0.5);vec3 a0=x-ox;
m*=1.79284291400159-0.85373472095314*(a0*a0+h*h);vec3 g;g.x=a0.x*x0.x+h.x*x0.y;g.yz=a0.yz*x12.xz+h.yz*x12.yw;
return 130.0*dot(m,g);}
void main(){vec2 uv=vUv;float ratio=u_resolution.x/u_resolution.y;vec2 p=uv-0.5;p.x*=ratio;
float t=u_time*0.1;
float n1=snoise(p*0.4+vec2(t*0.2,-t*0.3));
float n2=snoise(p*0.55+vec2(-t*0.15,t*0.25)+n1*0.25);
float n3=snoise(p*0.75+vec2(t*0.1,-t*0.2)+n2*0.2);
vec3 col=u_bg;float dist=length(p)*1.5;float vignette=1.0-smoothstep(0.3,1.2,dist);
col=mix(col,u_colors[0],smoothstep(-0.2,0.5,n1)*0.85);
col=mix(col,u_colors[1],smoothstep(-0.1,0.6,n2)*0.7);
col=mix(col,u_colors[2],smoothstep(-0.3,0.4,n3)*0.6);
col=mix(col,u_colors[3],smoothstep(0.0,0.7,n1*n2)*0.5);
float glow=smoothstep(0.8,0.0,dist)*0.3;col+=u_colors[1]*glow;
col=mix(col*0.2,col,vignette);
float grain=fract(sin(dot(uv,vec2(12.9898,78.233)))*43758.5453+u_time);
col+=(grain-0.5)*u_grain*0.1;gl_FragColor=vec4(col,1.0);}`;

// Blue -> deep-navy -> black, matching the site's blue accents on a dark base.
const BG_COLORS = ["#1e63d6", "#2f7bf0", "#0a1b3f", "#000000"];
const BG_BASE = "#03060d";

function mountGradientBackground({ bg = BG_BASE, colors = BG_COLORS, speed = 1.4, grain = 0.25 } = {}) {
  return; // dark gradient retired by the light redesign (code below kept until Task 3 cleanup)
  if (document.getElementById("ca-bg")) return;
  const hexToRgb = (hex) => {
    const h = hex.replace("#", "");
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
  };
  const canvas = document.createElement("canvas");
  canvas.id = "ca-bg";
  document.body.insertBefore(canvas, document.body.firstChild);
  const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
  if (!gl) { canvas.remove(); return; }   // CSS fallback bg takes over

  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  const program = gl.createProgram();
  gl.attachShader(program, sh(gl.VERTEX_SHADER, BG_VERT));
  gl.attachShader(program, sh(gl.FRAGMENT_SHADER, BG_FRAG));
  gl.linkProgram(program); gl.useProgram(program);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  const L = {
    res: gl.getUniformLocation(program, "u_resolution"),
    time: gl.getUniformLocation(program, "u_time"),
    grain: gl.getUniformLocation(program, "u_grain"),
    colors: gl.getUniformLocation(program, "u_colors"),
    bg: gl.getUniformLocation(program, "u_bg"),
  };
  const colorFlat = new Float32Array(colors.slice(0, 4).flatMap(hexToRgb));
  const bgRgb = hexToRgb(bg);

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(window.innerWidth * dpr);
    canvas.height = Math.floor(window.innerHeight * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  };
  resize();
  window.addEventListener("resize", resize);

  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const draw = (ms) => {
    gl.uniform2f(L.res, canvas.width, canvas.height);
    gl.uniform1f(L.time, ms * 0.001 * speed);
    gl.uniform1f(L.grain, grain);
    gl.uniform3fv(L.colors, colorFlat);
    gl.uniform3f(L.bg, bgRgb[0], bgRgb[1], bgRgb[2]);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (!reduce) requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);   // reduced-motion: renders one static frame
}

