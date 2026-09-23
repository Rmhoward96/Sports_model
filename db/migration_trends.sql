-- Betting trends: Action Network season betting records (NFL + CFB) and our
-- computed NFL situational trends. Idempotent. Run in the Supabase SQL Editor.
CREATE TABLE IF NOT EXISTS team_betting_records (
    sport          TEXT NOT NULL,          -- nfl | cfb
    season         INTEGER NOT NULL,
    team_name      TEXT NOT NULL,          -- our display name (AN name if unmatched)
    an_team_name   TEXT,
    abbr           TEXT,
    records        JSONB NOT NULL,         -- {category: {w,l,p,o,u}}
    captured_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, season, team_name)
);

CREATE TABLE IF NOT EXISTS nfl_game_trends (
    game_pk        BIGINT NOT NULL,
    team_name      TEXT NOT NULL,
    situation      TEXT NOT NULL,          -- home|road|favorite|underdog|off_road|off_home|off_win|off_loss|off_bye|division|primetime
    label          TEXT,                   -- "off a road game"
    ats_w INTEGER, ats_l INTEGER, ats_p INTEGER,
    ou_o  INTEGER, ou_u  INTEGER, ou_p  INTEGER,
    n              INTEGER,
    since_season   INTEGER,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, team_name, situation)
);

ALTER TABLE team_betting_records ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read team_betting_records" ON team_betting_records;
CREATE POLICY "public read team_betting_records" ON team_betting_records FOR SELECT USING (true);
ALTER TABLE nfl_game_trends ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_game_trends" ON nfl_game_trends;
CREATE POLICY "public read nfl_game_trends" ON nfl_game_trends FOR SELECT USING (true);
GRANT SELECT ON team_betting_records, nfl_game_trends TO anon, authenticated;
