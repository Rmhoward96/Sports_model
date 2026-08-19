# Hosting — run it always, from a browser, no terminal

Goal: the daily ingest runs itself in the cloud on a schedule, results land in Supabase,
and you browse them in Supabase's web UI. After the one-time setup below you never touch
a terminal.

## Architecture

```
GitHub Actions (cron, free)  ──daily──►  scripts/daily_ingest.py  ──►  Supabase Postgres
                                                                         (browse in Table Editor)
```

- **GitHub Actions** = the always-on scheduler. It already exists: `.github/workflows/daily-ingest.yml` runs `daily_ingest.py` every day and can be triggered manually with a button.
- **Supabase** = the durable store + your UI. The daily job writes to it automatically whenever `DATABASE_URL` is set (which it is, in the cloud).
- Raw Statcast (tens of GB) never goes in Supabase — only the small tables (schedule, later predictions) do. Free tier (500 MB) is plenty for those.

## One-time setup (all in the browser)

### 1. Create the Supabase project
1. Go to supabase.com → **New project** (free tier). Pick a strong DB password.
2. Open the **SQL Editor** → paste the contents of [`db/serving_bootstrap.sql`](../db/serving_bootstrap.sql) → **Run**. (Later, also run `db/schema.sql` when the modeling tables are needed.)
3. **Settings → Database → Connection string → URI**, choose the **Session pooler** string (works from GitHub's network). Copy it — this is your `DATABASE_URL`. It looks like `postgresql://postgres.xxxx:[PASSWORD]@aws-0-...pooler.supabase.com:5432/postgres`.

### 2. Put the code on GitHub (no terminal)
1. Install **GitHub Desktop** (desktop.github.com) — a GUI, no command line.
2. In GitHub Desktop: **Add → Add Existing Repository** → select this folder → **Publish repository** (private is fine).

### 3. Give the cloud job your database URL
1. On github.com, open your new repo → **Settings → Secrets and variables → Actions → New repository secret**.
2. Name: `DATABASE_URL`. Value: the Session-pooler string from step 1.3. Save.

### 4. Turn it on
1. Repo → **Actions** tab → enable workflows if prompted.
2. Open **daily-ingest** → **Run workflow** (manual trigger) to test it immediately.
3. It now also runs automatically every day at 13:00 UTC (edit the cron in the workflow file to change).

### 5. Look at your data
Supabase → **Table Editor → daily_schedule**. You'll see tonight's + tomorrow's slate with
probable pitchers, refreshed on every run. As the model comes online, `gold_game_predictions`
and `gold_player_prop_predictions` appear here too — that's where you'll scan for edges.

## What still needs a real machine (occasionally)

The one-time **Statcast backfill** (tens of GB) is too heavy for a serverless run. Options,
in order of least effort:
- Run it once from GitHub Actions via a manual "Run workflow" button (a `backfill.yml` we add
  when the transform layer exists — it computes the small rate profiles and writes only those
  to Supabase, discarding raw pitches).
- Or run it once on any computer with `uv run python scripts/backfill_mlb.py`.

After that first backfill, everything is hands-off and browser-only.
