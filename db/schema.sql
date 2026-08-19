-- =============================================================================
-- Multi-Sport Prediction Engine — Database Schema (v0.1, MLB-first)
-- =============================================================================
-- Design goals:
--   * Shared conformed DIMENSIONS across all four sports; specialized FACTS per sport.
--   * Medallion layering: bronze (raw) -> silver (clean/conformed) -> gold (features + predictions).
--   * Every entity carries its own canonical id PLUS a source_ids map, so the
--     cross-provider ID crosswalk lives inside the dimension.
--   * Idempotent by natural key (upsert on conflict); backfill resumable via ingest_log.
--   * Model outputs stamped with model_version so history is never mutated.
--
-- Portability: written in Postgres (Supabase) dialect. The same DDL runs in DuckDB
-- with two swaps: JSONB -> JSON, and SERIAL/identity -> INTEGER (or use BIGINT keys
-- you assign yourself). The bronze/silver tables typically live in DuckDB+Parquet on
-- your laptop; the gold_* and stg_* serving tables are what you push to Supabase.
--
-- Layer tag on each table:  [BRONZE] [SILVER] [GOLD] [STAGING] [REF] [INFRA]
-- =============================================================================


-- =============================================================================
-- CONFORMED DIMENSIONS  (shared across NBA / NFL / MLB / NHL)   [SILVER]
-- =============================================================================

CREATE TABLE dim_sport (
    sport_id    SMALLINT PRIMARY KEY,        -- 1=MLB, 2=NHL, 3=NFL, 4=NBA
    code        TEXT NOT NULL UNIQUE,         -- 'MLB','NHL','NFL','NBA'
    name        TEXT NOT NULL
);

CREATE TABLE dim_season (
    season_id     BIGINT PRIMARY KEY,          -- surrogate; e.g. sport_id*10000 + year
    sport_id      SMALLINT NOT NULL REFERENCES dim_sport(sport_id),
    season_year   SMALLINT NOT NULL,           -- 2024 = the 2024 MLB season
    season_type   TEXT NOT NULL,               -- 'regular' | 'postseason' | 'preseason' | 'spring'
    start_date    DATE,
    end_date      DATE,
    UNIQUE (sport_id, season_year, season_type)
);

CREATE TABLE dim_team (
    team_id        BIGINT PRIMARY KEY,          -- our canonical id
    sport_id       SMALLINT NOT NULL REFERENCES dim_sport(sport_id),
    canonical_name TEXT NOT NULL,               -- 'Los Angeles Dodgers'
    abbrev         TEXT NOT NULL,               -- 'LAD'
    location       TEXT,                        -- 'Los Angeles'
    nickname       TEXT,                        -- 'Dodgers'
    active         BOOLEAN NOT NULL DEFAULT TRUE,
    -- crosswalk: {"mlbam":119,"fangraphs":22,"bref":"LAD","espn":19,"retrosheet":"LAN"}
    source_ids     JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (sport_id, abbrev)
);

CREATE TABLE dim_player (
    player_id        BIGINT PRIMARY KEY,        -- our canonical id
    sport_id         SMALLINT NOT NULL REFERENCES dim_sport(sport_id),
    full_name        TEXT NOT NULL,
    first_name       TEXT,
    last_name        TEXT,
    birthdate        DATE,                      -- key fuzzy-match field for crosswalk
    bats             TEXT,                      -- 'L' | 'R' | 'S'   (MLB)
    throws           TEXT,                      -- 'L' | 'R'         (MLB)
    primary_position TEXT,                      -- 'SP','RP','C','1B',... (sport-specific)
    debut_date       DATE,
    active           BOOLEAN NOT NULL DEFAULT TRUE,
    -- crosswalk: {"mlbam":660271,"fangraphs":"sa3009901","bref":"ohtansh01","retrosheet":"ohtas001"}
    source_ids       JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_dim_player_name ON dim_player (sport_id, last_name, first_name);
CREATE INDEX idx_dim_player_srcids ON dim_player USING GIN (source_ids);

CREATE TABLE dim_venue (
    venue_id     BIGINT PRIMARY KEY,
    sport_id     SMALLINT NOT NULL REFERENCES dim_sport(sport_id),
    name         TEXT NOT NULL,                 -- 'Dodger Stadium'
    city         TEXT,
    state        TEXT,
    latitude     DOUBLE PRECISION,              -- for Open-Meteo weather joins (MLB/NFL)
    longitude    DOUBLE PRECISION,
    elevation_ft INTEGER,                       -- matters for MLB run environment (e.g. Coors)
    roof_type    TEXT,                          -- 'open' | 'retractable' | 'dome'
    surface      TEXT,
    source_ids   JSONB NOT NULL DEFAULT '{}'::jsonb
);


-- =============================================================================
-- GAME SPINE  (one row per game, all sports)   [SILVER]
-- =============================================================================

CREATE TABLE dim_game (
    game_id            BIGINT PRIMARY KEY,       -- our canonical id
    sport_id           SMALLINT NOT NULL REFERENCES dim_sport(sport_id),
    season_id          BIGINT NOT NULL REFERENCES dim_season(season_id),
    game_date          DATE NOT NULL,            -- local game date
    scheduled_start_utc TIMESTAMPTZ,
    home_team_id       BIGINT NOT NULL REFERENCES dim_team(team_id),
    away_team_id       BIGINT NOT NULL REFERENCES dim_team(team_id),
    venue_id           BIGINT REFERENCES dim_venue(venue_id),
    status             TEXT NOT NULL,            -- 'scheduled' | 'in_progress' | 'final' | 'postponed'
    home_score         INTEGER,                  -- NULL until final (runs for MLB)
    away_score         INTEGER,
    -- derived context features useful for every sport:
    home_rest_days     SMALLINT,                 -- computed from schedule
    away_rest_days     SMALLINT,
    -- crosswalk: {"mlbam":716352,"espn":"401472...","retrosheet":"LAN202404010"}
    source_ids         JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (sport_id, game_date, home_team_id, away_team_id)
);
CREATE INDEX idx_dim_game_date ON dim_game (sport_id, game_date);
CREATE INDEX idx_dim_game_season ON dim_game (season_id);


-- =============================================================================
-- MLB FACT TABLES   (specialized grain — this is the MVP sport)
-- =============================================================================

-- Pitch-level Statcast grain. The big table (tens of GB at full history);
-- lives in DuckDB/Parquet, partitioned by season. Only summaries go to Supabase.  [BRONZE->SILVER]
CREATE TABLE fact_mlb_pitch (
    game_id          BIGINT NOT NULL REFERENCES dim_game(game_id),
    at_bat_number    SMALLINT NOT NULL,
    pitch_number     SMALLINT NOT NULL,
    inning           SMALLINT,
    inning_half      TEXT,                       -- 'top' | 'bottom'
    pitcher_id       BIGINT REFERENCES dim_player(player_id),
    batter_id        BIGINT REFERENCES dim_player(player_id),
    pitch_type       TEXT,                       -- 'FF','SL','CH',...
    release_speed    REAL,                       -- mph
    release_spin_rate INTEGER,
    plate_x          REAL,
    plate_z          REAL,
    balls            SMALLINT,
    strikes          SMALLINT,
    description      TEXT,                        -- 'called_strike','ball','hit_into_play',...
    -- batted-ball / outcome fields (present only on contact):
    launch_speed     REAL,                        -- exit velocity, mph
    launch_angle     REAL,
    estimated_woba   REAL,                        -- xwOBA on contact
    events           TEXT,                        -- terminal PA event: 'single','strikeout','home_run',...
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    source           TEXT NOT NULL DEFAULT 'statcast',
    PRIMARY KEY (game_id, at_bat_number, pitch_number)
);
CREATE INDEX idx_mlb_pitch_pitcher ON fact_mlb_pitch (pitcher_id);
CREATE INDEX idx_mlb_pitch_batter  ON fact_mlb_pitch (batter_id);

-- Per-game per-batter box line. Feeds most hitter props.   [SILVER]
CREATE TABLE fact_mlb_batting_box (
    game_id       BIGINT NOT NULL REFERENCES dim_game(game_id),
    player_id     BIGINT NOT NULL REFERENCES dim_player(player_id),
    team_id       BIGINT NOT NULL REFERENCES dim_team(team_id),
    batting_order SMALLINT,                       -- 1-9, NULL if sub
    pa   SMALLINT, ab SMALLINT,
    h    SMALLINT, doubles SMALLINT, triples SMALLINT, hr SMALLINT,
    rbi  SMALLINT, r SMALLINT, bb SMALLINT, so SMALLINT,
    sb   SMALLINT, hbp SMALLINT, tb SMALLINT,     -- total bases = props staple
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT NOT NULL DEFAULT 'mlb_statsapi',
    PRIMARY KEY (game_id, player_id)
);

-- Per-game per-pitcher box line. Feeds pitcher props (Ks, outs, ER).   [SILVER]
CREATE TABLE fact_mlb_pitching_box (
    game_id        BIGINT NOT NULL REFERENCES dim_game(game_id),
    player_id      BIGINT NOT NULL REFERENCES dim_player(player_id),
    team_id        BIGINT NOT NULL REFERENCES dim_team(team_id),
    is_starter     BOOLEAN,
    outs_recorded  SMALLINT,                      -- store outs, not "IP" — avoids the .1/.2 mess
    bf   SMALLINT,                                -- batters faced
    h    SMALLINT, r SMALLINT, er SMALLINT,
    bb   SMALLINT, so SMALLINT, hr SMALLINT,
    pitches SMALLINT, strikes SMALLINT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT NOT NULL DEFAULT 'mlb_statsapi',
    PRIMARY KEY (game_id, player_id)
);

-- >>> When you add NHL/NFL/NBA, new fact tables slot in here at their own grain:
--     fact_nhl_pbp, fact_nhl_skater_box, fact_nhl_goalie_box
--     fact_nfl_pbp, fact_nfl_player_box
--     fact_nba_pbp, fact_nba_player_box
--     ...all referencing the SAME dim_game / dim_player / dim_team. Nothing above changes.


-- =============================================================================
-- REFERENCE / CONTEXT   [REF]
-- =============================================================================

-- Park factors (run/HR environment), from Baseball Savant. Multiplicative vs league avg.
CREATE TABLE ref_mlb_park_factors (
    venue_id     BIGINT NOT NULL REFERENCES dim_venue(venue_id),
    season_year  SMALLINT NOT NULL,
    factor_type  TEXT NOT NULL,                   -- 'runs','hr','hits','doubles',...
    factor_value REAL NOT NULL,                   -- 1.00 = neutral
    PRIMARY KEY (venue_id, season_year, factor_type)
);

-- Pre-game weather forecast for outdoor venues (Open-Meteo), keyed to a game.
CREATE TABLE ref_game_weather (
    game_id       BIGINT PRIMARY KEY REFERENCES dim_game(game_id),
    temp_f        REAL,
    wind_mph      REAL,
    wind_dir_deg  SMALLINT,                        -- vs. park orientation -> out/in to CF
    humidity_pct  REAL,
    precip_prob   REAL,
    forecast_at   TIMESTAMPTZ,                     -- when the forecast was pulled (leakage guard)
    source        TEXT NOT NULL DEFAULT 'open-meteo'
);


-- =============================================================================
-- DAILY OPERATIONAL / STAGING  (current + forecast inputs, refreshed daily)   [STAGING]
-- =============================================================================

CREATE TABLE stg_schedule (
    game_id       BIGINT PRIMARY KEY REFERENCES dim_game(game_id),
    sport_id      SMALLINT NOT NULL,
    game_date     DATE NOT NULL,
    home_team_id  BIGINT NOT NULL,
    away_team_id  BIGINT NOT NULL,
    scheduled_start_utc TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- MLB probable/confirmed starting pitchers — the single largest daily variable.
CREATE TABLE stg_mlb_probable_pitchers (
    game_id    BIGINT NOT NULL REFERENCES dim_game(game_id),
    team_id    BIGINT NOT NULL REFERENCES dim_team(team_id),
    player_id  BIGINT REFERENCES dim_player(player_id),
    confirmed  BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, team_id)
);

-- Projected/confirmed batting lineups (opportunity input for hitter props).
CREATE TABLE stg_lineups (
    game_id       BIGINT NOT NULL REFERENCES dim_game(game_id),
    team_id       BIGINT NOT NULL REFERENCES dim_team(team_id),
    player_id     BIGINT NOT NULL REFERENCES dim_player(player_id),
    batting_order SMALLINT,                        -- 1-9 (MLB); role slot for other sports
    position      TEXT,
    confirmed     BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_id, player_id)
);

CREATE TABLE stg_injuries (
    player_id   BIGINT NOT NULL REFERENCES dim_player(player_id),
    sport_id    SMALLINT NOT NULL,
    status      TEXT,                              -- 'out','day-to-day','IL-10',...
    description TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, updated_at)
);


-- =============================================================================
-- ODDS CAPTURE  (NOT modeled against — stored purely for CLV backtesting)   [REF]
-- =============================================================================
-- Your daily workflow ignores odds. This table exists so that AFTER the fact you
-- can measure whether your projections beat the closing line (closing line value),
-- the real signal that your model has edge. Capture opening + a close snapshot.
CREATE TABLE odds_line_snapshot (
    game_id     BIGINT NOT NULL REFERENCES dim_game(game_id),
    market      TEXT NOT NULL,                     -- 'moneyline','spread','total','player_prop'
    side        TEXT NOT NULL,                     -- 'home','away','over','under', or player+market
    player_id   BIGINT REFERENCES dim_player(player_id),  -- for player props
    book        TEXT NOT NULL,
    line        REAL,                              -- spread/total/prop line (NULL for moneyline)
    price       INTEGER NOT NULL,                  -- American odds, e.g. -110
    captured_at TIMESTAMPTZ NOT NULL,
    is_closing  BOOLEAN NOT NULL DEFAULT FALSE,
    source      TEXT NOT NULL DEFAULT 'the-odds-api',
    PRIMARY KEY (game_id, market, side, book, captured_at)
);


-- =============================================================================
-- GOLD — MODEL OUTPUTS  (what the Streamlit app reads)   [GOLD]
-- =============================================================================

-- One row per game per model version. Internally consistent: win prob, spread,
-- and total all derive from the same predicted score distribution.
CREATE TABLE gold_game_predictions (
    prediction_id      BIGSERIAL PRIMARY KEY,
    game_id            BIGINT NOT NULL REFERENCES dim_game(game_id),
    model_version      TEXT NOT NULL,              -- 'mlb-elo-poisson-0.1'
    generated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    pred_home_score    REAL NOT NULL,              -- expected runs
    pred_away_score    REAL NOT NULL,
    pred_total         REAL NOT NULL,              -- = home + away
    pred_margin        REAL NOT NULL,              -- home - away
    margin_sd          REAL,                       -- spread of the margin distribution
    total_sd           REAL,
    home_win_prob      REAL NOT NULL,              -- 0..1, calibrated
    dist_type          TEXT NOT NULL DEFAULT 'poisson',  -- how score dist was modeled
    features_snapshot  JSONB,                      -- inputs frozen at prediction time (leakage audit)
    UNIQUE (game_id, model_version, generated_at)
);
CREATE INDEX idx_gold_gp_game ON gold_game_predictions (game_id);

-- One row per player per market per model version. Stores the DISTRIBUTION, not just
-- the mean, so P(over line) is well-defined. If a book line is known you can pre-store
-- prob_over; otherwise the app computes it live from (dist_type, mean, dispersion) for
-- whatever line you type in.
CREATE TABLE gold_player_prop_predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    game_id         BIGINT NOT NULL REFERENCES dim_game(game_id),
    player_id       BIGINT NOT NULL REFERENCES dim_player(player_id),
    market          TEXT NOT NULL,                 -- 'strikeouts','hits','total_bases','home_runs',...
    model_version   TEXT NOT NULL,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    projected_mean  REAL NOT NULL,                 -- the expected value
    dist_type       TEXT NOT NULL,                 -- 'poisson' | 'negative_binomial' | 'normal'
    dispersion      REAL,                          -- NB size / Normal SD; NULL for pure Poisson
    -- optional convenience if a book line was supplied at generation time:
    ref_line        REAL,
    prob_over       REAL,                          -- P(stat > ref_line)
    prob_under      REAL,
    features_snapshot JSONB,
    UNIQUE (game_id, player_id, market, model_version, generated_at)
);
CREATE INDEX idx_gold_prop_game   ON gold_player_prop_predictions (game_id);
CREATE INDEX idx_gold_prop_player ON gold_player_prop_predictions (player_id);


-- =============================================================================
-- INFRA — pipeline control + crosswalk resolution   [INFRA]
-- =============================================================================

-- Makes backfill resumable and daily jobs idempotent. Job checks this before/after work.
CREATE TABLE ingest_log (
    run_id       BIGSERIAL PRIMARY KEY,
    sport_id     SMALLINT,
    entity       TEXT NOT NULL,                    -- 'mlb_pitch','schedule','probable_pitchers',...
    partition_key TEXT,                            -- e.g. 'season=2019' or 'date=2024-04-01'
    status       TEXT NOT NULL,                    -- 'started' | 'success' | 'failed'
    rows_written INTEGER,
    watermark    TEXT,                             -- max game_date / last id processed
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    error        TEXT
);
CREATE INDEX idx_ingest_log_entity ON ingest_log (entity, partition_key);

-- Manual/fuzzy-resolved ID matches that public crosswalks (Chadwick, nflverse) miss.
-- Persisted so a match is never re-solved. source_ids JSONB on the dims is the fast path;
-- this table is the audit + override trail for the residual you resolve by hand.
CREATE TABLE id_crosswalk_override (
    sport_id     SMALLINT NOT NULL,
    entity_type  TEXT NOT NULL,                    -- 'player' | 'team' | 'venue'
    canonical_id BIGINT NOT NULL,
    source       TEXT NOT NULL,                    -- 'fangraphs','bref','espn',...
    source_id    TEXT NOT NULL,
    confidence   REAL,                             -- 1.0 = exact, <1 = fuzzy
    resolved_by  TEXT,                             -- 'chadwick','fuzzy','manual'
    resolved_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_type, source, source_id)
);


-- =============================================================================
-- PLAYER-PROP MODELING SUPPORT   [REF] + [GOLD features]
-- =============================================================================
-- Design: every supported prop is an aggregation of ONE artifact —
-- feat_matchup_pa_outcomes (the per-plate-appearance outcome vector for each
-- batter vs. the opposing starter). Hitter props read one batter's row; pitcher
-- props aggregate the starter's confrontations across the opposing lineup.

-- Catalog of the markets we actually forecast (drives the app + validation).
CREATE TABLE ref_prop_market (
    market_code  TEXT PRIMARY KEY,           -- stable code used in gold_player_prop_predictions.market
    sport_id     SMALLINT NOT NULL REFERENCES dim_sport(sport_id),
    role         TEXT NOT NULL,              -- 'hitter' | 'pitcher'
    display_name TEXT NOT NULL,
    default_dist TEXT NOT NULL,              -- 'poisson'|'negative_binomial'|'normal'|'bernoulli'|'empirical'
    notes        TEXT
);

-- External projection-system inputs (Steamer/ZiPS/THE BAT X) used as the talent
-- PRIOR that per-PA rates regress toward (better than regressing to league avg).
CREATE TABLE ref_player_projection (
    player_id   BIGINT NOT NULL REFERENCES dim_player(player_id),
    season_year SMALLINT NOT NULL,
    source      TEXT NOT NULL,               -- 'steamer' | 'zips' | 'thebatx' | 'blend'
    as_of_date  DATE NOT NULL,               -- projections update; keep the vintage
    role        TEXT NOT NULL,               -- 'hitter' | 'pitcher'
    -- projected per-PA (hitter) or per-BF (pitcher) true-talent rates:
    proj_pa   INTEGER,
    p_bb REAL, p_k REAL, p_1b REAL, p_2b REAL, p_3b REAL, p_hr REAL, p_out REAL,
    proj_woba REAL,
    PRIMARY KEY (player_id, season_year, source, as_of_date)
);

-- Rolling/season batter rates, split by opposing pitcher handedness.
-- Per-PA terminal-event probabilities (mutually exclusive; ~sum to 1).
CREATE TABLE feat_batter_profile (
    player_id  BIGINT NOT NULL REFERENCES dim_player(player_id),
    as_of_date DATE NOT NULL,                -- leakage guard: rates known as of this date
    vs_hand    TEXT NOT NULL,                -- 'L' | 'R' | 'ALL' (opposing pitcher hand)
    window     TEXT NOT NULL,                -- 'season' | '30d' | 'career'
    pa         INTEGER,
    p_out REAL, p_bb REAL, p_k REAL,
    p_1b REAL, p_2b REAL, p_3b REAL, p_hr REAL,
    p_hit REAL, xwoba REAL, iso REAL,        -- convenience aggregates
    PRIMARY KEY (player_id, as_of_date, vs_hand, window)
);

-- Rolling/season pitcher rates ALLOWED, split by opposing batter handedness,
-- plus workload fields that drive the Outs-Recorded prop.
CREATE TABLE feat_pitcher_profile (
    player_id  BIGINT NOT NULL REFERENCES dim_player(player_id),
    as_of_date DATE NOT NULL,
    vs_hand    TEXT NOT NULL,                -- 'L' | 'R' | 'ALL' (opposing batter hand)
    window     TEXT NOT NULL,
    bf         INTEGER,                      -- batters faced (sample size)
    p_out REAL, p_bb REAL, p_k REAL,
    p_1b REAL, p_2b REAL, p_3b REAL, p_hr REAL,
    p_hit_allowed REAL, xwoba_allowed REAL,
    -- workload / hook model inputs (for Outs Recorded):
    avg_outs_per_start   REAL,
    avg_pitches_per_start REAL,
    pitches_per_pa       REAL,
    PRIMARY KEY (player_id, as_of_date, vs_hand, window)
);

-- League expected plate appearances by batting-order slot (the biggest lever for
-- counting-stat props). Simple lookup; refine with team-specific pace later.
CREATE TABLE feat_batting_order_pa (
    season_year   SMALLINT NOT NULL,
    batting_order SMALLINT NOT NULL,         -- 1-9
    expected_pa   REAL NOT NULL,
    PRIMARY KEY (season_year, batting_order)
);

-- THE SHARED ENGINE OUTPUT. One row per (game, batter, starter) after the
-- odds-ratio blend (batter_profile x pitcher_profile / league) + park/weather.
-- Hitter props read a batter's row; pitcher props aggregate all rows for a starter.
CREATE TABLE feat_matchup_pa_outcomes (
    game_id       BIGINT NOT NULL REFERENCES dim_game(game_id),
    batter_id     BIGINT NOT NULL REFERENCES dim_player(player_id),
    pitcher_id    BIGINT NOT NULL REFERENCES dim_player(player_id),  -- opposing starter
    model_version TEXT NOT NULL,
    generated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    projected_pa  REAL NOT NULL,             -- from lineup slot x expected game length
    -- per-PA outcome vector used to derive every prop:
    p_out REAL, p_bb REAL, p_k REAL,
    p_1b REAL, p_2b REAL, p_3b REAL, p_hr REAL,
    PRIMARY KEY (game_id, batter_id, model_version, generated_at)
);
CREATE INDEX idx_matchup_pitcher ON feat_matchup_pa_outcomes (game_id, pitcher_id);

-- Literal batter-vs-pitcher history. STORED FOR REFERENCE/DISPLAY ONLY — heavily
-- regressed toward the odds-ratio estimate; near-zero weight in the model. Small
-- samples here are noise, not signal (documented BvP trap).
CREATE TABLE feat_bvp_rollup (
    batter_id  BIGINT NOT NULL REFERENCES dim_player(player_id),
    pitcher_id BIGINT NOT NULL REFERENCES dim_player(player_id),
    pa INTEGER, h INTEGER, tb INTEGER, hr INTEGER, bb INTEGER, k INTEGER,
    PRIMARY KEY (batter_id, pitcher_id)
);


-- =============================================================================
-- SEED: the four sports
-- =============================================================================
INSERT INTO dim_sport (sport_id, code, name) VALUES
    (1,'MLB','Major League Baseball'),
    (2,'NHL','National Hockey League'),
    (3,'NFL','National Football League'),
    (4,'NBA','National Basketball Association')
ON CONFLICT (sport_id) DO NOTHING;

-- MLB prop markets covered by the model (MVP set).
INSERT INTO ref_prop_market (market_code, sport_id, role, display_name, default_dist, notes) VALUES
    ('hits',          1,'hitter', 'Hits',              'poisson',           'Poisson-binomial over projected PA'),
    ('total_bases',   1,'hitter', 'Total Bases',       'empirical',         'Sum of per-PA base values; distribution via convolution/sim'),
    ('home_run',      1,'hitter', 'Home Run (Y/N)',    'bernoulli',         'P(>=1 HR) = 1-(1-p_hr)^PA'),
    ('hrr',           1,'hitter', 'Hits+Runs+RBIs',    'empirical',         'Combo; R/RBI need lineup context, Monte-Carlo in Phase 4'),
    ('pitcher_ks',    1,'pitcher','Strikeouts',        'negative_binomial', 'Sum p_k*PA across opposing lineup'),
    ('outs_recorded', 1,'pitcher','Outs Recorded',     'normal',            'Projected BF x out rate, capped by workload/hook model'),
    ('hits_allowed',  1,'pitcher','Hits Allowed',      'poisson',           'Sum p_hit*PA across opposing lineup')
ON CONFLICT (market_code) DO NOTHING;
