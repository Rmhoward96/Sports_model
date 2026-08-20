"""Sports Model — online dashboard (Streamlit Community Cloud).

Reads predictions + odds from Supabase. Filter by league and bet type; see today's
board with the model number next to the market line and the implied edge; and track
how predictions have compared to closing lines as that history accumulates.

Deploy: share.streamlit.io -> connect this repo -> main file streamlit_app.py ->
add secret DATABASE_URL (Supabase session-pooler string). See docs/dashboard.md.
"""
from __future__ import annotations

import os

import pandas as pd
import psycopg
import streamlit as st

st.set_page_config(page_title="Sports Model", page_icon="⚾", layout="wide")

MARKET_LABELS = {
    "moneyline": "Moneyline", "total": "Total (O/U)", "spread": "Run line",
    "hits": "Hits", "total_bases": "Total Bases", "home_run": "Home Run",
    "hrr": "Hits+Runs+RBIs", "pitcher_ks": "Strikeouts",
    "hits_allowed": "Hits Allowed", "outs_recorded": "Outs Recorded",
}
GAME_MARKETS = ["moneyline", "total", "spread"]
PROP_MARKETS = ["hits", "total_bases", "home_run", "hrr",
                "pitcher_ks", "hits_allowed", "outs_recorded"]


def _dburl() -> str:
    return st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL", ""))


@st.cache_resource
def _conn():
    url = _dburl()
    if not url:
        st.error("DATABASE_URL is not set. Add it in the app's Secrets.")
        st.stop()
    return psycopg.connect(url, autocommit=True)


@st.cache_data(ttl=300)
def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _conn().cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def american_to_prob(odds) -> float | None:
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


# ---- latest market line per game/market/side (consensus across books) ----
_LATEST = """
WITH latest AS (
    SELECT DISTINCT ON (game_pk, market, side, player_name, book)
           game_pk, market, side, player_name, book, line, price
    FROM odds_snapshot
    ORDER BY game_pk, market, side, player_name, book, captured_at DESC
)
SELECT game_pk, market, side, player_name,
       avg(line) AS line, avg(price) AS price, count(*) AS books
FROM latest GROUP BY game_pk, market, side, player_name
"""


@st.cache_data(ttl=300)
def latest_odds() -> pd.DataFrame:
    return q(_LATEST)


def game_board(model_version: str, game_market: str) -> pd.DataFrame:
    preds = q("""
        SELECT game_pk, game_date, away_team_name, home_team_name,
               pred_away_score, pred_home_score, pred_total, pred_margin, home_win_prob
        FROM game_predictions WHERE model_version = %s
    """, (model_version,))
    if preds.empty:
        return preds
    odds = latest_odds()
    rows = []
    for _, g in preds.iterrows():
        o = odds[odds.game_pk == g.game_pk]
        row = {"Game": f"{g.away_team_name} @ {g.home_team_name}",
               "Proj score": f"{g.pred_away_score:.1f}-{g.pred_home_score:.1f}"}
        if game_market == "total":
            mkt = o[o.market == "total"]["line"].mean()
            row["Model total"] = round(g.pred_total, 1)
            row["Market total"] = round(mkt, 1) if pd.notna(mkt) else None
            row["Edge"] = round(g.pred_total - mkt, 1) if pd.notna(mkt) else None
            row["Lean"] = ("OVER" if g.pred_total > mkt else "UNDER") if pd.notna(mkt) else "—"
        elif game_market == "moneyline":
            hp = o[(o.market == "moneyline") & (o.side == "home")]["price"].mean()
            ap = o[(o.market == "moneyline") & (o.side == "away")]["price"].mean()
            mh, ma = american_to_prob(hp), american_to_prob(ap)
            novig = mh / (mh + ma) if mh and ma else None
            row["Model home win%"] = f"{g.home_win_prob*100:.0f}%"
            row["Market home win%"] = f"{novig*100:.0f}%" if novig else "—"
            row["Edge"] = round((g.home_win_prob - novig) * 100, 1) if novig else None
        else:  # spread / run line
            row["Model margin"] = round(g.pred_margin, 1)
            mkt = o[(o.market == "spread") & (o.side == "home")]["line"].mean()
            row["Market run line"] = round(mkt, 1) if pd.notna(mkt) else None
        rows.append(row)
    return pd.DataFrame(rows)


def props_board(model_version: str, market: str) -> pd.DataFrame:
    preds = q("""
        SELECT p.game_pk, p.player_name, p.team_name, p.projected_mean, p.line,
               p.prob_over, p.lineup_source
        FROM prop_predictions p WHERE p.model_version = %s AND p.market = %s
    """, (model_version, market))
    if preds.empty:
        return preds
    odds = latest_odds()
    om = odds[odds.market == market].copy()
    om["key"] = om.player_name.str.lower().str.strip()
    over = om[om.side == "over"]
    rows = []
    for _, p in preds.iterrows():
        k = str(p.player_name).lower().strip()
        line = over[over.key == k]["line"].mean()
        price = over[over.key == k]["price"].mean()
        rows.append({
            "Player": p.player_name, "Team": p.team_name,
            "Model proj": round(p.projected_mean, 2),
            "Book line": round(line, 1) if pd.notna(line) else None,
            "Edge": round(p.projected_mean - line, 2) if pd.notna(line) else None,
            "Model P(over)": f"{p.prob_over*100:.0f}%",
            "Over price": int(price) if pd.notna(price) else None,
            "Lineup": p.lineup_source,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Edge", ascending=False, na_position="last") if "Edge" in df else df


def model_versions(table: str) -> list[str]:
    df = q(f"SELECT DISTINCT model_version FROM {table} ORDER BY model_version DESC")
    return df.model_version.tolist() if not df.empty else []


# ============================ UI ============================
st.title("⚾ Sports Model")

with st.sidebar:
    st.header("Filters")
    league = st.selectbox("League", ["MLB"], help="NBA/NFL/NHL coming later")
    section = st.radio("View", ["📋 Board", "📈 vs Closing Line"])
    bet_group = st.radio("Bet type", ["Game lines", "Player props"])
    if bet_group == "Game lines":
        market = st.selectbox("Market", GAME_MARKETS, format_func=MARKET_LABELS.get)
    else:
        market = st.selectbox("Market", PROP_MARKETS, format_func=MARKET_LABELS.get)

if section == "📋 Board":
    st.subheader(f"{league} — {MARKET_LABELS[market]}")
    if bet_group == "Game lines":
        mvs = model_versions("game_predictions")
        if not mvs:
            st.info("No game predictions yet. Run the daily-ingest workflow.")
        else:
            df = game_board(mvs[0], market)
            st.caption(f"Model: `{mvs[0]}` · Edge = model − market (positive favors the model's side)")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        mvs = model_versions("prop_predictions")
        if not mvs:
            st.info("No prop predictions yet. Run the daily-ingest / refresh-props workflow.")
        else:
            df = props_board(mvs[0], market)
            st.caption(f"Model: `{mvs[0]}` · sorted by edge (model projection − book line)")
            st.dataframe(df, use_container_width=True, hide_index=True)

else:  # vs Closing Line
    st.subheader(f"{league} — {MARKET_LABELS[market]}: predictions vs closing line")
    st.info(
        "This fills in as odds + predictions accumulate over the coming weeks. "
        "Closing line = the last odds snapshot before first pitch."
    )
    hist = q("""
        WITH closing AS (
            SELECT DISTINCT ON (game_pk, market, side, player_name)
                   game_pk, market, side, player_name, line, price, captured_at
            FROM odds_snapshot
            WHERE market = %s AND captured_at <= commence_time
            ORDER BY game_pk, market, side, player_name, captured_at DESC
        )
        SELECT count(*) AS closing_lines,
               count(DISTINCT game_pk) AS games
        FROM closing
    """, (market,))
    n = int(hist.closing_lines.iloc[0]) if not hist.empty else 0
    games = int(hist.games.iloc[0]) if not hist.empty else 0
    c1, c2 = st.columns(2)
    c1.metric("Closing lines captured", f"{n:,}")
    c2.metric("Games covered", f"{games:,}")
    if n == 0:
        st.caption("No closing lines captured for this market yet — check back after a few game days.")
    else:
        st.caption("Detailed CLV (model number vs closing, edge distribution, and W/L "
                   "once results are graded) is the next build on top of this data.")
