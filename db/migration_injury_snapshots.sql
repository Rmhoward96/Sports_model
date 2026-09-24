-- Injury snapshots for the day-before injury watch. Internal (no anon access).
CREATE TABLE IF NOT EXISTS injury_snapshots (
    sport        TEXT NOT NULL,
    game_pk      BIGINT NOT NULL,
    fingerprint  TEXT NOT NULL,
    statuses     JSONB NOT NULL,       -- [{team, player, status}] Out/Doubtful only
    captured_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk)
);
ALTER TABLE injury_snapshots ENABLE ROW LEVEL SECURITY;
