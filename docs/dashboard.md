# Dashboard — deploy the online UI (browser only)

`streamlit_app.py` is a Streamlit dashboard that reads predictions + odds from Supabase.
Host it free on Streamlit Community Cloud — it deploys straight from your GitHub repo,
no terminal.

## What it shows

- **Board**: filter by league (MLB for now) and bet type (game lines: moneyline / total /
  run line; or the 7 player-prop markets). Each row shows the **model number next to the
  market line and the implied edge**.
- **vs Closing Line**: the track record of predictions against the closing line. It fills
  in as odds + predictions accumulate (closing line = last snapshot before first pitch),
  so it starts nearly empty and grows over the coming weeks.

## Deploy (one time, ~3 minutes)

1. Go to **share.streamlit.io** and sign in with your GitHub account.
2. **Create app** → **Deploy a public app from GitHub**:
   - Repository: `Rmhoward96/Sports_model`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
3. **Advanced settings → Secrets**, paste (TOML format, keep the quotes):
   ```toml
   DATABASE_URL = "postgresql://postgres.xxxx:[PASSWORD]@aws-0-...pooler.supabase.com:5432/postgres"
   ```
   Use the **Session pooler** connection string (same one as the GitHub secret).
4. **Deploy.** You'll get a public URL like `https://sports-model.streamlit.app`.

It auto-redeploys whenever you push to `main`. Data refreshes on a 5-minute cache.

## Notes

- Only **MLB** exists today; the league filter is built for NBA/NFL/NHL to slot in later.
- Props join to book lines by player name; a rare name-format mismatch may leave a book
  line blank — that's cosmetic, not a data error.
- The `requirements.txt` at the repo root is **only** for this dashboard. The rest of the
  project uses `pyproject.toml` / uv, and the GitHub Actions workflows ignore it.
