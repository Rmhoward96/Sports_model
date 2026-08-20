"""Sports Model — online dashboard (Streamlit Community Cloud).

Reads predictions + odds from Supabase. Filter by league, date, game, and bet type;
see the model number next to the market line and the implied edge; and track how
predictions have compared to closing lines as that history accumulates.

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


@st.cache_data(ttl=300)
def dates_available() -> list:
    df = q("SELECT DISTINCT game_date FROM daily_schedule ORDER BY game_date DESC")
    return df.game_date.tolist() if not df.empty else []


@st.cache_data(ttl=300)
def games_on(d) -> pd.DataFrame:
    return q("""SELECT game_pk, away_team_name || ' @ ' || home_team_name AS matchup
                FROM daily_schedule WHERE game_date = %s ORDER BY matchup""", (d,))


@st.cache_data(ttl=300)
def latest_odds(pks: tuple) -> pd.DataFrame:
    """Consensus latest line per (game, market, side, player) across books."""
    if not pks:
        return pd.DataFrame()
    return q("""
        WITH latest AS (
            SELECT DISTINCT ON (game_pk, market, side, player_name, book)
                   game_pk, market, side, player_name, book, line, price
            FROM odds_snapshot WHERE game_pk = ANY(%s)
            ORDER BY game_pk, market, side, player_name, book, captured_at DESC
        )
        SELECT game_pk, market, side, player_name,
               avg(line) AS line, avg(price) AS price, count(*) AS books
        FROM latest GROUP BY game_pk, market, side, player_name
    """, (list(pks),))


@st.cache_data(ttl=300)
def odds_status() -> tuple:
    df = q("SELECT max(captured_at) AS mx, count(DISTINCT game_pk) AS g FROM odds_snapshot")
    if df.empty or df.mx.iloc[0] is None:
        return None, 0
    return df.mx.iloc[0], int(df.g.iloc[0])


def game_board(mv, market, gdate, pks) -> tuple[pd.DataFrame, int, int]:
    preds = q("""
        SELECT game_pk, away_team_name, home_team_name, pred_away_score, pred_home_score,
               pred_total, pred_margin, home_win_prob
        FROM game_predictions
        WHERE model_version = %s AND game_date = %s AND game_pk = ANY(%s)
    """, (mv, gdate, list(pks)))
    if preds.empty:
        return preds, 0, 0
    odds = latest_odds(tuple(preds.game_pk.tolist()))
    rows, matched = [], 0
    for _, g in preds.iterrows():
        o = odds[odds.game_pk == g.game_pk] if not odds.empty else pd.DataFrame()
        has = False
        row = {"Game": f"{g.away_team_name} @ {g.home_team_name}",
               "Proj score": f"{g.pred_away_score:.1f}-{g.pred_home_score:.1f}"}
        if market == "total":
            mkt = o[o.market == "total"]["line"].mean() if not o.empty else float("nan")
            has = pd.notna(mkt)
            row["Model total"] = round(g.pred_total, 1)
            row["Market total"] = round(mkt, 1) if has else "—"
            row["Edge"] = round(g.pred_total - mkt, 1) if has else "—"
            row["Lean"] = ("OVER" if g.pred_total > mkt else "UNDER") if has else "—"
        elif market == "moneyline":
            hp = o[(o.market == "moneyline") & (o.side == "home")]["price"].mean() if not o.empty else float("nan")
            ap = o[(o.market == "moneyline") & (o.side == "away")]["price"].mean() if not o.empty else float("nan")
            mh, ma = american_to_prob(hp), american_to_prob(ap)
            novig = mh / (mh + ma) if mh and ma else None
            has = novig is not None
            row["Model home win%"] = f"{g.home_win_prob*100:.0f}%"
            row["Market home win%"] = f"{novig*100:.0f}%" if has else "—"
            row["Edge"] = f"{(g.home_win_prob - novig)*100:+.0f} pts" if has else "—"
        else:  # spread / run line
            mkt = o[(o.market == "spread") & (o.side == "home")]["line"].mean() if not o.empty else float("nan")
            has = pd.notna(mkt)
            row["Model margin (home)"] = round(g.pred_margin, 1)
            row["Market run line"] = round(mkt, 1) if has else "—"
        matched += int(has)
        rows.append(row)
    return pd.DataFrame(rows), matched, len(preds)


def props_board(mv, market, gdate, pks) -> tuple[pd.DataFrame, int, int]:
    preds = q("""
        SELECT game_pk, player_name, team_name, projected_mean, line, prob_over, lineup_source
        FROM prop_predictions
        WHERE model_version = %s AND market = %s AND game_date = %s AND game_pk = ANY(%s)
    """, (mv, market, gdate, list(pks)))
    if preds.empty:
        return preds, 0, 0
    odds = latest_odds(tuple(preds.game_pk.unique().tolist()))
    over = odds[(odds.market == market) & (odds.side == "over")].copy() if not odds.empty else pd.DataFrame()
    if not over.empty:
        over["key"] = over.player_name.str.lower().str.strip()
    rows, matched = [], 0
    for _, p in preds.iterrows():
        k = str(p.player_name).lower().strip()
        line = over[over.key == k]["line"].mean() if not over.empty else float("nan")
        price = over[over.key == k]["price"].mean() if not over.empty else float("nan")
        has = pd.notna(line)
        matched += int(has)
        rows.append({
            "Player": p.player_name, "Team": p.team_name,
            "Model proj": round(p.projected_mean, 2),
            "Book line": round(line, 1) if has else "—",
            "Edge": round(p.projected_mean - line, 2) if has else "—",
            "Model P(over)": f"{p.prob_over*100:.0f}%",
            "Over price": int(price) if pd.notna(price) else "—",
            "Lineup": p.lineup_source,
        })
    df = pd.DataFrame(rows)
    numeric_edge = pd.to_numeric(df["Edge"], errors="coerce")
    return df.assign(_e=numeric_edge).sort_values("_e", ascending=False, na_position="last").drop(columns="_e"), matched, len(preds)


def model_versions(table) -> list:
    df = q(f"SELECT DISTINCT model_version FROM {table} ORDER BY model_version DESC")
    return df.model_version.tolist() if not df.empty else []


# ============================ UI ============================
st.title("⚾ Sports Model")

mx, ng = odds_status()
if mx is not None:
    st.caption(f"Latest odds capture: **{mx:%Y-%m-%d %H:%M UTC}** · {ng} games with odds in the database")

dates = dates_available()
if not dates:
    st.warning("No games in the database yet. Run the daily-ingest workflow.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    league = st.selectbox("League", ["MLB"], help="NBA/NFL/NHL coming later")
    gdate = st.selectbox("Date", dates, format_func=lambda d: d.strftime("%a %b %d, %Y"))
    gdf = games_on(gdate)
    opts = dict(zip(gdf.matchup, gdf.game_pk)) if not gdf.empty else {}
    picked = st.multiselect("Games", list(opts), default=list(opts),
                            help="Filter to specific games")
    pks = tuple(opts[m] for m in picked)
    section = st.radio("View", ["📋 Board", "📈 vs Closing Line"])
    bet_group = st.radio("Bet type", ["Game lines", "Player props"])
    markets = GAME_MARKETS if bet_group == "Game lines" else PROP_MARKETS
    market = st.selectbox("Market", markets, format_func=MARKET_LABELS.get)

if not pks:
    st.info("Select at least one game.")
    st.stop()

if section == "📋 Board":
    st.subheader(f"{league} — {MARKET_LABELS[market]} — {gdate:%b %d}")
    if bet_group == "Game lines":
        mvs = model_versions("game_predictions")
        if not mvs:
            st.info("No game predictions yet.")
        else:
            df, matched, total = game_board(mvs[0], market, gdate, pks)
            st.caption(f"Model `{mvs[0]}` · **market data for {matched}/{total} games** · "
                       f"Edge = model − market")
            if matched == 0:
                st.warning("No market lines matched these games — the captured odds are "
                           "likely from a different date than these predictions. Odds match "
                           "predictions on the same game day.")
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        mvs = model_versions("prop_predictions")
        if not mvs:
            st.info("No prop predictions yet.")
        else:
            df, matched, total = props_board(mvs[0], market, gdate, pks)
            st.caption(f"Model `{mvs[0]}` · **book lines for {matched}/{total} players** · "
                       f"sorted by edge (model − book line)")
            st.dataframe(df, use_container_width=True, hide_index=True)

else:  # vs Closing Line — the graded track record
    st.subheader(f"{league} — {MARKET_LABELS[market]}: track record vs closing line")
    st.caption("Each graded bet = the side the model favored vs the closing line, staked "
               "1u at the closing price. Grows daily as games finalize.")
    df = q("""
        SELECT game_date, player_name, lean, model_number, closing_line, closing_price,
               actual, result, profit, edge
        FROM prediction_results
        WHERE market = %s AND result IS NOT NULL
        ORDER BY game_date DESC, edge DESC
    """, (market,))
    if df.empty:
        st.info("No graded results for this market yet — they appear after games finish "
                "and the grade-results job runs. Give it a few game days.")
    else:
        wins = int((df.result == "win").sum())
        losses = int((df.result == "loss").sum())
        pushes = int((df.result == "push").sum())
        decided = wins + losses
        winpct = wins / decided if decided else 0.0
        roi = df.profit.sum() / len(df) if len(df) else 0.0
        avg_edge = df.edge.mean()
        c1, c2, c3, c4 = st.columns(4)
        rec = f"{wins}-{losses}" + (f"-{pushes}" if pushes else "")
        c1.metric("Record", rec)
        c2.metric("Win %", f"{winpct*100:.1f}%")
        c3.metric("ROI at close", f"{roi*100:+.1f}%")
        c4.metric("Avg edge", f"{avg_edge:+.2f}")
        if decided < 30:
            st.warning(f"Only {decided} decided bets so far — far too small to mean anything. "
                       "Sample needs hundreds before ROI is signal, not noise.")
        st.dataframe(df, use_container_width=True, hide_index=True)
