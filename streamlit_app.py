"""Sports Model — online dashboard (Streamlit Community Cloud).

Reads predictions + odds from Supabase. Filter by league, date, game, and bet type;
see the model number next to the market line and the implied edge; and track how
predictions have compared to closing lines as that history accumulates.

Deploy: share.streamlit.io -> connect this repo -> main file streamlit_app.py ->
add secret DATABASE_URL (Supabase session-pooler string). See docs/dashboard.md.
"""
from __future__ import annotations

import json
import os
from math import erf, exp, log, sqrt
from pathlib import Path

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


def decimal_odds(price) -> float:
    if price is None or pd.isna(price) or float(price) == 0:
        return float("nan")  # 0 / NaN is not a valid American price
    price = float(price)
    return 1 + (price / 100 if price > 0 else 100 / -price)


@st.cache_resource
def _calib() -> dict:
    p = Path(__file__).resolve().parent / "assets" / "calibration.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def calibrate(target: str, raw: float) -> float:
    """Platt calibration, mirrored from sportsmodel.model.calibration (pure-python)."""
    params = _calib().get(target)
    if not params or raw != raw:
        return raw
    a, b = params
    r = min(max(raw, 1e-6), 1 - 1e-6)
    return 1.0 / (1.0 + exp(-(a * log(r / (1 - r)) + b)))


def prob_over_dist(dist, line: float) -> float:
    """Model P(X > line) from a stored distribution (pmf | normal)."""
    if not dist:
        return float("nan")
    if isinstance(dist, str):
        dist = json.loads(dist)
    if dist.get("kind") == "normal":
        sd = dist["sd"]
        if sd <= 0:
            return 1.0 if dist["mean"] > line else 0.0
        return 0.5 * (1 - erf((line - dist["mean"]) / (sd * sqrt(2))))
    return sum(p for k, p in enumerate(dist.get("pmf") or []) if k > line)


def prob_cover(margin_dist, home_line: float) -> float:
    """Model P(home covers `home_line`) from a stored margin dist. Home covers iff
    margin > -home_line (a margin equal to -home_line is a push, so strict)."""
    if isinstance(margin_dist, str):
        margin_dist = json.loads(margin_dist)
    if not margin_dist or margin_dist.get("kind") != "margin":
        return float("nan")
    offset = margin_dist["offset"]
    pmf = margin_dist.get("pmf") or []
    return sum(p for i, p in enumerate(pmf) if (i - offset) > -home_line)


def apply_affine(dist, loc: float, scale: float):
    """Location+scale remap of a stored pmf/margin dist, re-binned to integer support.
    Mirrors sportsmodel.model.distributions.apply_affine (pure-python for the app)."""
    if isinstance(dist, str):
        dist = json.loads(dist)
    if not dist:
        return dist
    pmf = dist.get("pmf") or []
    n = len(pmf)
    if n == 0:
        return dist
    offset = dist.get("offset", 0)
    mean = sum((i - offset) * p for i, p in enumerate(pmf))
    out = [0.0] * n
    for i, p in enumerate(pmf):
        x = scale * ((i - offset) - mean) + mean + loc
        j = min(n - 1, max(0, round(x + offset)))
        out[j] += p
    s = sum(out)
    if s > 0:
        out = [v / s for v in out]
    res = {"kind": dist.get("kind"), "pmf": out}
    if dist.get("kind") == "margin":
        res["offset"] = offset
    return res


def _dist_cal(key: str):
    c = _calib().get(key, {})
    return c.get("loc", 0.0), c.get("scale", 1.0)


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
    """Consensus per (game, market, side, player, LINE) across books.

    One row per distinct line — never average across different lines. Books post
    alternate lines (over 0.5 / 1.5 / 2.5…); collapsing them and averaging American
    prices produces meaningless numbers. `price` is the median posted price at that
    line; `books` counts how many books offer it (used to pick the main line).
    """
    if not pks:
        return pd.DataFrame()
    return q("""
        WITH latest AS (
            SELECT DISTINCT ON (game_pk, market, side, player_name, line, book)
                   game_pk, market, side, player_name, line, book, price
            FROM odds_snapshot WHERE game_pk = ANY(%s)
            ORDER BY game_pk, market, side, player_name, line, book, captured_at DESC
        )
        SELECT game_pk, market, side, player_name, line,
               count(*) AS books,
               percentile_disc(0.5) WITHIN GROUP (ORDER BY price) AS price
        FROM latest GROUP BY game_pk, market, side, player_name, line
    """, (list(pks),))


@st.cache_data(ttl=300)
def odds_status() -> tuple:
    df = q("SELECT max(captured_at) AS mx, count(DISTINCT game_pk) AS g FROM odds_snapshot")
    if df.empty or df.mx.iloc[0] is None:
        return None, 0
    return df.mx.iloc[0], int(df.g.iloc[0])


def _main_line(df) -> float:
    """Most-booked line in a market/side-filtered odds frame (tie -> lowest)."""
    if df is None or df.empty:
        return float("nan")
    g = df.groupby("line", as_index=False)["books"].sum()
    return float(g.sort_values(["books", "line"], ascending=[False, True]).iloc[0]["line"])


def _main_price(df) -> float:
    """Median price of the most-booked line (for moneyline, the single NULL line)."""
    if df is None or df.empty:
        return float("nan")
    d = df.dropna(subset=["price"])
    d = d[d.price != 0]
    if d.empty:
        return float("nan")
    d = d.sort_values("books", ascending=False)
    return float(d.iloc[0]["price"])


def _spread_price(o, side: str, ln: float) -> float:
    """Consensus price for a spread side at a specific line (most-booked)."""
    d = o[(o.market == "spread") & (o.side == side) & (o.line == ln)] if not o.empty else o
    d = d.dropna(subset=["price"]) if not d.empty else d
    d = d[d.price != 0] if not d.empty else d
    return float(d.sort_values("books", ascending=False).iloc[0]["price"]) if not d.empty else float("nan")


def game_board(mv, market, gdate, pks) -> tuple[pd.DataFrame, int, int]:
    preds = q("""
        SELECT game_pk, away_team_name, home_team_name, pred_away_score, pred_home_score,
               pred_total, pred_margin, home_win_prob, margin_dist
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
        pick = "—"
        row = {"Game": f"{g.away_team_name} @ {g.home_team_name}", "Pick": pick,
               "Proj score": f"{g.pred_away_score:.1f}-{g.pred_home_score:.1f}"}
        if market == "total":
            mkt = _main_line(o[o.market == "total"]) if not o.empty else float("nan")
            has = pd.notna(mkt)
            # location-calibrated total mean (the sim runs ~1 run light; the calibration
            # loc re-centers it, which is what stops the board leaning UNDER on everything)
            cal_total = g.pred_total + _dist_cal("total_dist")[0]
            row["Model total"] = round(cal_total, 1)
            row["Market total"] = round(mkt, 1) if has else "—"
            row["Edge"] = round(cal_total - mkt, 1) if has else "—"
            row["Lean"] = ("OVER" if cal_total > mkt else "UNDER") if has else "—"
            if has:
                pick = f"Over {mkt:g}" if cal_total > mkt else f"Under {mkt:g}"
        elif market == "moneyline":
            hp = _main_price(o[(o.market == "moneyline") & (o.side == "home")]) if not o.empty else float("nan")
            ap = _main_price(o[(o.market == "moneyline") & (o.side == "away")]) if not o.empty else float("nan")
            mh, ma = american_to_prob(hp), american_to_prob(ap)
            novig = mh / (mh + ma) if mh and ma else None
            has = novig is not None
            row["Model home win%"] = f"{g.home_win_prob*100:.0f}%"
            row["Market home win%"] = f"{novig*100:.0f}%" if has else "—"
            row["Edge"] = f"{(g.home_win_prob - novig)*100:+.0f} pts" if has else "—"
            if has:
                pick = g.home_team_name if g.home_win_prob > novig else g.away_team_name
        else:  # spread / run line
            mkt = _main_line(o[(o.market == "spread") & (o.side == "home")]) if not o.empty else float("nan")
            has = pd.notna(mkt)
            row["Model margin (home)"] = round(g.pred_margin, 1)
            row["Market run line"] = round(mkt, 1) if has else "—"
            row["Model cover%"] = "—"
            row["EV"] = "—"
            # Pick the run line by EV, not mean margin vs the number: a game's expected
            # margin sits inside 1.5 runs almost always, so a mean rule takes the juiced
            # +1.5 every time. Compare P(cover) x price on each side; +EV side wins, else pass.
            p_home = prob_cover(apply_affine(g.margin_dist, *_dist_cal("margin_dist")),
                                float(mkt)) if has else float("nan")
            hp = _spread_price(o, "home", mkt) if has else float("nan")
            ap = _spread_price(o, "away", -mkt) if has else float("nan")
            if has and p_home == p_home and pd.notna(hp) and pd.notna(ap):
                ev_h = p_home * decimal_odds(hp) - 1
                ev_a = (1 - p_home) * decimal_odds(ap) - 1
                if ev_h >= ev_a:
                    cover_p, ev, side_txt = p_home, ev_h, f"{g.home_team_name} {mkt:+g}"
                else:
                    cover_p, ev, side_txt = 1 - p_home, ev_a, f"{g.away_team_name} {-mkt:+g}"
                row["Model cover%"] = f"{cover_p*100:.0f}%"
                row["EV"] = f"{ev*100:+.1f}%"
                pick = side_txt if ev > 0 else "pass"
            elif has:  # no stored margin dist / missing a price -> legacy mean rule
                pick = (f"{g.home_team_name} {mkt:+g}" if g.pred_margin + mkt > 0
                        else f"{g.away_team_name} {-mkt:+g}")
        row["Pick"] = pick
        matched += int(has)
        rows.append(row)
    return pd.DataFrame(rows), matched, len(preds)


def props_board(mv, market, gdate, pks) -> tuple[pd.DataFrame, int, int]:
    preds = q("""
        SELECT game_pk, player_name, team_name, projected_mean, line, prob_over, dist, lineup_source
        FROM prop_predictions
        WHERE model_version = %s AND market = %s AND game_date = %s AND game_pk = ANY(%s)
    """, (mv, market, gdate, list(pks)))
    if preds.empty:
        return preds, 0, 0
    odds = latest_odds(tuple(preds.game_pk.unique().tolist()))
    # Drop invalid prices (0 / NULL — e.g. a book that posted only one side) so they
    # never contaminate the consensus price or cause a divide-by-zero in EV.
    valid = odds[odds.price.notna() & (odds.price != 0)] if not odds.empty else odds
    over = valid[(valid.market == market) & (valid.side == "over")].copy() if not valid.empty else pd.DataFrame()
    under = valid[(valid.market == market) & (valid.side == "under")].copy() if not valid.empty else pd.DataFrame()
    for d in (over, under):
        if not d.empty:
            d["key"] = d.player_name.str.lower().str.strip()
    rows, matched = [], 0
    for _, p in preds.iterrows():
        k = str(p.player_name).lower().strip()
        o = over[over.key == k] if not over.empty else pd.DataFrame()
        u = under[under.key == k] if not under.empty else pd.DataFrame()
        # Pick the player's MAIN line (most books; tie -> lowest line), then read the
        # over/under price AT that line. Never blend different lines together.
        if not o.empty:
            main = o.sort_values(["books", "line"], ascending=[False, True]).iloc[0]
            line, po = float(main["line"]), float(main["price"])
            um = u[u.line == line] if not u.empty else pd.DataFrame()
            pu = float(um["price"].iloc[0]) if not um.empty else float("nan")
        else:
            line = po = pu = float("nan")
        has = pd.notna(line) and pd.notna(po) and po != 0
        row = {"Player": p.player_name, "Team": p.team_name,
               "Model proj": round(p.projected_mean, 2),
               "Book line": round(line, 1) if has else "—"}
        ev_sort = float("-inf")
        if has:
            matched += 1
            p_over = calibrate(market, prob_over_dist(p.dist, float(line)))
            io = american_to_prob(po)
            iu = american_to_prob(pu) if pd.notna(pu) else (1 - io if io is not None else None)
            novig_over = io / (io + iu) if io and iu else None
            ev_over = p_over * decimal_odds(po) - 1 if p_over == p_over else float("nan")
            ev_under = (1 - p_over) * decimal_odds(pu) - 1 if (p_over == p_over and pd.notna(pu)) else float("-inf")
            if p_over != p_over:  # legacy row, no stored distribution
                row.update({"Side": "—", "Model P": "—", "Market P": "—",
                            "Edge": "—", "EV": "—", "Price": int(po) if pd.notna(po) else "—"})
            else:
                if ev_over >= ev_under:
                    side, price, mp, mkp, ev = "OVER", po, p_over, novig_over, ev_over
                else:
                    side, price, mp, mkp, ev = "UNDER", pu, 1 - p_over, \
                        (1 - novig_over if novig_over is not None else None), ev_under
                # Home runs are over-only longshots (books rarely post the "under"), so
                # the pick above always lands on OVER. Only call it a PICK when the over
                # is genuinely +EV; otherwise show "pass". The row's Model P / Market P /
                # EV stay visible so the math is still there to collect and analyze.
                display_side = side
                if market == "home_run" and not (ev > 0):
                    display_side = "pass"
                row.update({
                    "Side": display_side,
                    "Model P": f"{mp*100:.0f}%",
                    "Market P": f"{mkp*100:.0f}%" if mkp is not None else "—",
                    "Edge": f"{(mp - mkp)*100:+.0f} pts" if mkp is not None else "—",
                    "EV": f"{ev*100:+.1f}%",
                    "Price": int(price) if pd.notna(price) else "—",
                })
                ev_sort = ev
        else:
            row.update({"Side": "—", "Model P": "—", "Market P": "—",
                        "Edge": "—", "EV": "—", "Price": "—"})
        row["Lineup"] = p.lineup_source
        row["_ev"] = ev_sort
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("_ev", ascending=False).drop(columns="_ev")
    return df, matched, len(preds)


def model_versions(table) -> list:
    df = q(f"SELECT model_version FROM {table} GROUP BY model_version ORDER BY max(generated_at) DESC")
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
                       f"sorted by **EV**. Side = the model's +EV side at the book's line. "
                       f"EV = model P(win) × decimal odds − 1 (per 1u); "
                       f"Edge = model P − no-vig market P. Bet only clearly +EV.")
            st.dataframe(df, use_container_width=True, hide_index=True)

else:  # vs Closing Line — the graded track record
    st.subheader(f"{league} — {MARKET_LABELS[market]}: track record vs closing line")
    st.caption("Each graded bet = the side the model favored vs the closing line, staked "
               "1u at the closing price. Grows daily as games finalize.")
    df = q("""
        SELECT game_date, player_name, lean, model_number, closing_line, closing_price,
               model_prob, market_prob, ev, actual, result, profit, edge
        FROM prediction_results
        WHERE market = %s AND result IS NOT NULL
        ORDER BY game_date DESC, ev DESC NULLS LAST
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
        units = df.profit.sum()
        roi = units / len(df) if len(df) else 0.0
        avg_ev = pd.to_numeric(df.ev, errors="coerce").mean()
        c1, c2, c3, c4, c5 = st.columns(5)
        rec = f"{wins}-{losses}" + (f"-{pushes}" if pushes else "")
        c1.metric("Record", rec)
        c2.metric("Win %", f"{winpct*100:.1f}%")
        c3.metric("Net units", f"{units:+.2f}u")
        c4.metric("ROI at close", f"{roi*100:+.1f}%")
        c5.metric("Avg EV at close", f"{avg_ev*100:+.1f}%" if pd.notna(avg_ev) else "—")
        if decided < 30:
            st.warning(f"Only {decided} decided bets so far — far too small to mean anything. "
                       "Sample needs hundreds before ROI is signal, not noise.")
        # `player_name` holds the pick label the grader built (team for moneyline,
        # "Over/Under X" for totals, covering team + line for the run line). Surface it
        # as an explicit Pick column so a game-line report never reads as a bare win/loss.
        show = df.rename(columns={
            "game_date": "Date", "player_name": "Pick", "lean": "Side",
            "model_number": "Model #", "closing_line": "Close", "closing_price": "Price",
            "model_prob": "Model P", "market_prob": "Mkt P", "ev": "EV",
            "actual": "Actual", "result": "Result", "profit": "Units", "edge": "Edge",
        })
        show["Pick"] = show["Pick"].fillna("—").replace("", "—")
        for c in ("Model P", "Mkt P", "EV"):
            show[c] = pd.to_numeric(show[c], errors="coerce").map(
                lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "—")
        show["Units"] = pd.to_numeric(show["Units"], errors="coerce").map(
            lambda v: f"{v:+.2f}u" if pd.notna(v) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)
