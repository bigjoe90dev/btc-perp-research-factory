"""
ict-bot Command Center
Algorithmic Forex Trading Bot Dashboard

Run: python -m dashboard.app
"""

import json
import os
from datetime import datetime, timezone, timedelta

import dash
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output

from shared.db import connect


# ====================================================================
# Symbol Config
# ====================================================================
SYMBOL_NAME = (os.getenv("CTRADER_SYMBOL_NAME", "EURUSD") or "EURUSD").upper()
SYMBOL_ID = int(os.getenv("CTRADER_SYMBOL_ID", "0") or 0)
SYMBOL_ENV = (os.getenv("CTRADER_ENV", "demo") or "demo").strip().lower()
SYMBOL_ACCOUNT_ID = (os.getenv("CTRADER_ACCOUNT_ID", "") or "").strip()

SYMBOL_DIGITS = int(os.getenv("CTRADER_DIGITS", "5") or 5)
PRICE_DIVISOR_RAW = os.getenv("CTRADER_PRICE_DIVISOR", "").strip()
PRICE_DIVISOR = float(PRICE_DIVISOR_RAW) if PRICE_DIVISOR_RAW else 0.0
PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "10000") or 10000)


def _format_symbol(sym: str) -> str:
    if "/" in sym:
        return sym
    if len(sym) == 6:
        return f"{sym[:3]}/{sym[3:]}"
    if sym.endswith("USD") and len(sym) > 3:
        return f"{sym[:-3]}/USD"
    return sym


SYMBOL_LABEL = _format_symbol(SYMBOL_NAME)
SYMBOL_DESC = {
    "XAUUSD": "Gold / US Dollar",
    "EURUSD": "Euro / US Dollar",
    "GBPUSD": "British Pound / US Dollar",
}.get(SYMBOL_NAME, "FX / CFD")

TRADINGVIEW_SYMBOL = (os.getenv("TRADINGVIEW_SYMBOL", "") or "").strip()
if not TRADINGVIEW_SYMBOL:
    if SYMBOL_NAME == "XAUUSD":
        TRADINGVIEW_SYMBOL = "OANDA:XAUUSD"
    elif SYMBOL_NAME.endswith("USD") and len(SYMBOL_NAME) == 6:
        TRADINGVIEW_SYMBOL = f"OANDA:{SYMBOL_NAME}"
    else:
        TRADINGVIEW_SYMBOL = SYMBOL_NAME


def _normalize_price(v: float) -> float:
    if v is None:
        return v
    if PRICE_DIVISOR:
        # Only scale if values still look like raw integers
        return v / PRICE_DIVISOR if abs(v) > 10000 else v
    if abs(v) > 1000:
        scaled = v / (10 ** SYMBOL_DIGITS)
        if SYMBOL_NAME in ("BTCUSD", "ETHUSD") and scaled > 1_000_000:
            scaled = scaled / 1000
        return scaled
    return v


def _price_fmt(v: float) -> str:
    return f"{v:,.{SYMBOL_DIGITS}f}"


# ====================================================================
# Colour Palette
# ====================================================================
C = {
    "bg":           "#0a0e17",
    "surface":      "#111827",
    "card":         "#1a2332",
    "card_border":  "#1e3a5f",
    "text":         "#e2e8f0",
    "text_dim":     "#94a3b8",
    "text_muted":   "#64748b",
    "accent":       "#00d4ff",
    "accent_dim":   "#0891b2",
    "green":        "#00c853",
    "green_dim":    "#065f46",
    "red":          "#ff3366",
    "red_dim":      "#7f1d1d",
    "orange":       "#ff9f43",
    "purple":       "#a78bfa",
    "yellow":       "#fbbf24",
    "grid":         "#1e293b",
    "header_bg":    "#0d1421",
    "chart_green":  "#00c853",
    "chart_fill":   "rgba(0,200,83,0.12)",
    "pill_active":  "#1d4ed8",
}

# ====================================================================
# Data Fetchers
# ====================================================================

def fetch_latest_heartbeat():
    with connect() as conn:
        return conn.execute(
            "SELECT ts_utc, status, note FROM heartbeat ORDER BY id DESC LIMIT 1"
        ).fetchone()

def is_stale(ts_utc, max_age=15):
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts_utc)).total_seconds() > max_age
    except Exception:
        return True

def fetch_recent_signals(limit=50):
    with connect() as conn:
        return pd.read_sql(
            f"SELECT ts_utc, signal, fast_ma, slow_ma, mid, note FROM signals ORDER BY id DESC LIMIT {limit}", conn)

def fetch_paper_trades(limit=100):
    where = []
    params = []
    if SYMBOL_ENV:
        where.append("env=?")
        params.append(SYMBOL_ENV)
    if SYMBOL_ACCOUNT_ID:
        where.append("account_id=?")
        params.append(SYMBOL_ACCOUNT_ID)
    if SYMBOL_ID:
        where.append("symbol_id=?")
        params.append(SYMBOL_ID)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with connect() as conn:
        df = pd.read_sql(
            f"SELECT ts_utc, action, side, size_units, mid, note FROM paper_trades {where_sql} "
            f"ORDER BY id DESC LIMIT {limit}",
            conn, params=params)

    if df.empty:
        return df

    def _parse_pnl(note: str) -> float:
        if not note:
            return 0.0
        if "pnl=$" not in note:
            return 0.0
        try:
            val = note.split("pnl=$", 1)[1].split()[0]
            return float(val.replace(",", ""))
        except Exception:
            return 0.0

    df["pnl"] = df["note"].apply(_parse_pnl)
    return df

def _normalize_quote_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    scale_mask = (df["bid"].abs() > 1000) | (df["ask"].abs() > 1000)
    if scale_mask.any():
        df.loc[scale_mask, ["bid", "ask"]] = df.loc[scale_mask, ["bid", "ask"]] / 100000
    return df


def _clean_quotes_for_chart(df: pd.DataFrame, resample=None) -> pd.DataFrame:
    if df.empty:
        return df
    df = _normalize_quote_prices(df)
    # Quotes can contain mixed ISO8601 formats (with/without fractional seconds/timezone).
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    df.index.name = "ts"
    mid = (df["bid"] + df["ask"]) / 2
    median = float(mid.median()) if len(mid) else None
    if median and np.isfinite(median):
        mid = mid[(mid > median / 10) & (mid < median * 10)]

    if resample:
        ohlc = mid.resample(resample).ohlc()
    else:
        ohlc = pd.DataFrame({"open": mid, "high": mid, "low": mid, "close": mid})
    return ohlc.dropna().reset_index()


def fetch_quotes_for_chart(timeframe="1m"):
    """Fetch quotes with smart downsampling per timeframe."""
    tf_map = {
        "1m":  (1,    "1min"),   # last 24h, 1m candles
        "5m":  (1,    "5min"),   # last 24h, 5m candles
        "15m": (3,    "15min"),  # last 3 days, 15m candles
        "1d":  (1,    "1min"),
        "5d":  (5,    "5min"),
        "1mo": (30,   "15min"),
        "6m":  (180,  "30min"),
        "ytd": (None, "1h"),      # special: since Jan 1
        "1y":  (365,  "1h"),
        "5y":  (1825, "1D"),
        "all": (9999, "4h"),
    }
    days_back, resample = tf_map.get(timeframe, (30, "5min"))

    with connect() as conn:
        if timeframe == "ytd":
            cutoff = datetime.now(timezone.utc).replace(month=1, day=1, hour=0, minute=0, second=0).isoformat()
            df = pd.read_sql(
                "SELECT ts_utc, bid, ask FROM quotes WHERE ts_utc >= ? ORDER BY ts_utc",
                conn, params=(cutoff,))
        elif days_back is not None and days_back >= 9999:
            df = pd.read_sql(
                "SELECT ts_utc, bid, ask FROM quotes ORDER BY ts_utc", conn)
        else:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back or 30)).isoformat()
            df = pd.read_sql(
                "SELECT ts_utc, bid, ask FROM quotes WHERE ts_utc >= ? ORDER BY ts_utc",
                conn, params=(cutoff,))

    if df.empty:
        return df

    return _clean_quotes_for_chart(df, resample)

def fetch_price_stats():
    """Get latest price + period open price for change calculation."""
    where = []
    params = []
    if SYMBOL_ENV:
        where.append("env=?")
        params.append(SYMBOL_ENV)
    if SYMBOL_ACCOUNT_ID:
        where.append("account_id=?")
        params.append(SYMBOL_ACCOUNT_ID)
    if SYMBOL_ID:
        where.append("symbol_id=?")
        params.append(SYMBOL_ID)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    with connect() as conn:
        latest = conn.execute(
            f"SELECT bid, ask, ts_utc FROM quotes {where_sql} ORDER BY ts_utc DESC LIMIT 1",
            params,
        ).fetchone()

        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

        if where_sql:
            day_where = f"{where_sql} AND ts_utc >= ?"
        else:
            day_where = "WHERE ts_utc >= ?"

        first = conn.execute(
            f"SELECT bid, ask, ts_utc FROM quotes {day_where} ORDER BY ts_utc ASC LIMIT 1",
            (*params, day_start),
        ).fetchone()

        if first is None:
            first = conn.execute(
                f"SELECT bid, ask, ts_utc FROM quotes {where_sql} ORDER BY ts_utc ASC LIMIT 1",
                params,
            ).fetchone()
    return latest, first


def fetch_paper_pnl():
    """Sum realized P&L from paper_trades notes (CLOSE rows)."""
    where = []
    params = []
    if SYMBOL_ENV:
        where.append("env=?")
        params.append(SYMBOL_ENV)
    if SYMBOL_ACCOUNT_ID:
        where.append("account_id=?")
        params.append(SYMBOL_ACCOUNT_ID)
    if SYMBOL_ID:
        where.append("symbol_id=?")
        params.append(SYMBOL_ID)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    total = 0.0
    with connect() as conn:
        rows = conn.execute(
            f"SELECT note FROM paper_trades {where_sql} AND action='CLOSE' ORDER BY id ASC",
            params,
        ).fetchall()
    for r in rows:
        note = r[0] or ""
        if "pnl=$" in note:
            try:
                val = note.split("pnl=$", 1)[1].split()[0]
                total += float(val.replace(",", ""))
            except Exception:
                pass
    return total

def fetch_backtest_runs():
    with connect() as conn:
        return pd.read_sql(
            "SELECT id, ts_utc, strategy_name, symbol, date_from, date_to, note FROM backtest_runs ORDER BY id DESC LIMIT 20", conn)

def fetch_all_news(limit=20):
    with connect() as conn:
        return pd.read_sql(
            f"SELECT event_name, currency, impact, datetime_utc, actual, forecast, previous FROM news_events ORDER BY datetime_utc DESC LIMIT {limit}", conn)

def fetch_account_stats():
    with connect() as conn:
        closed = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE action='CLOSE'").fetchone()[0] or 0
        open_p = conn.execute("SELECT COUNT(*) FROM paper_positions WHERE status='OPEN'").fetchone()[0] or 0
        total_quotes = conn.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] or 0
        total_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0] or 0
        total_events = conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0] or 0
        total_bt = conn.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0] or 0
    return {
        "closed_trades": closed, "open_positions": open_p,
        "total_quotes": total_quotes, "total_signals": total_signals,
        "total_events": total_events, "total_backtests": total_bt,
    }


# ====================================================================
# Chart Theme
# ====================================================================
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=C["surface"],
    font=dict(color=C["text_dim"], size=11, family="JetBrains Mono, Fira Code, monospace"),
    margin=dict(l=50, r=20, t=10, b=40),
    xaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"], showgrid=True, gridwidth=1),
    yaxis=dict(gridcolor=C["grid"], zerolinecolor=C["grid"], showgrid=True, gridwidth=1,
               side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
)

TABLE_STYLE_HEADER = {
    "backgroundColor": C["header_bg"],
    "color": C["accent"],
    "fontWeight": "600",
    "border": f"1px solid {C['card_border']}",
    "fontSize": "11px",
    "fontFamily": "JetBrains Mono, Fira Code, monospace",
    "letterSpacing": "0.3px",
    "padding": "8px 12px",
}

TABLE_STYLE_DATA = {
    "backgroundColor": C["card"],
    "color": C["text"],
    "border": f"1px solid {C['card_border']}",
    "fontSize": "12px",
    "fontFamily": "JetBrains Mono, Fira Code, monospace",
    "padding": "6px 12px",
}

TABLE_STYLE_DATA_COND = [
    {"if": {"row_index": "odd"}, "backgroundColor": C["surface"]},
    {"if": {"state": "active"}, "backgroundColor": "#1e3a5f", "border": f"1px solid {C['accent']}"},
]

TABLE_STYLE_CELL = {
    "fontSize": "12px",
    "padding": "6px 12px",
    "fontFamily": "JetBrains Mono, Fira Code, monospace",
    "textAlign": "left",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "maxWidth": "200px",
}


# ====================================================================
# Reusable Components
# ====================================================================
def card(children, **extra):
    style = {
        "background": f"linear-gradient(135deg, {C['card']} 0%, {C['surface']} 100%)",
        "border": f"1px solid {C['card_border']}",
        "borderRadius": "12px",
        "padding": "20px",
        "marginBottom": "16px",
        "boxShadow": "0 4px 24px rgba(0, 0, 0, 0.3)",
        "position": "relative",
        "overflow": "hidden",
        **extra,
    }
    return html.Div(style=style, children=children)


def stat_pill(label, value, color=C["accent"]):
    return html.Div(
        style={
            "display": "inline-flex", "alignItems": "center", "gap": "8px",
            "background": f"{color}15", "border": f"1px solid {color}40",
            "borderRadius": "20px", "padding": "6px 14px", "marginRight": "8px",
            "marginBottom": "8px",
        },
        children=[
            html.Span(label, style={
                "color": C["text_muted"], "fontSize": "11px",
                "letterSpacing": "0.3px", "fontWeight": "500",
            }),
            html.Span(str(value), style={
                "color": color, "fontSize": "14px", "fontWeight": "700",
                "fontFamily": "JetBrains Mono, monospace",
            }),
        ],
    )


def section_header(title, subtitle=""):
    return html.Div(
        style={"display": "flex", "alignItems": "baseline", "gap": "10px", "marginBottom": "16px"},
        children=[
            html.Span(title, style={
                "color": C["text"], "fontSize": "14px", "fontWeight": "600",
                "letterSpacing": "0.3px",
            }),
            html.Span(subtitle, style={
                "color": C["text_muted"], "fontSize": "11px",
            }) if subtitle else None,
        ],
    )


def make_data_table(df, page_size=8):
    if df.empty:
        return html.Div("Awaiting data...", style={
            "color": C["text_muted"], "fontStyle": "italic", "padding": "20px",
            "textAlign": "center", "fontSize": "13px",
        })
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_header=TABLE_STYLE_HEADER,
        style_data=TABLE_STYLE_DATA,
        style_data_conditional=TABLE_STYLE_DATA_COND,
        style_cell=TABLE_STYLE_CELL,
        page_size=page_size,
        style_as_list_view=True,
    )


# ====================================================================
# App Layout
# ====================================================================
app = dash.Dash(
    __name__,
    title="ict-bot",
    update_title=None,
)
server = app.server

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
  html, body { margin: 0; padding: 0; background: """ + C["bg"] + """; overflow-x: hidden; }
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: """ + C["surface"] + """; }
  ::-webkit-scrollbar-thumb { background: """ + C["card_border"] + """; border-radius: 3px; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
  .tf-btn { transition: all 0.15s ease; }
  .tf-btn:hover { opacity: 0.85; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

TRADINGVIEW_WIDGET_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body { height: 100%; margin: 0; background: #111827; }
    #tv_chart { height: 100%; width: 100%; }
  </style>
</head>
<body>
  <div id="tv_chart"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({
      "container_id": "tv_chart",
      "symbol": "__TV_SYMBOL__",
      "interval": "1",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#111827",
      "enable_publishing": false,
      "allow_symbol_change": false,
      "hide_legend": false,
      "hide_top_toolbar": false,
      "withdateranges": true,
      "studies": [],
      "width": "100%",
      "height": "100%"
    });
  </script>
</body>
</html>"""

TRADINGVIEW_WIDGET_HTML = TRADINGVIEW_WIDGET_HTML.replace("__TV_SYMBOL__", TRADINGVIEW_SYMBOL)

app.layout = html.Div(
    style={
        "fontFamily": "Inter, -apple-system, system-ui, sans-serif",
        "backgroundColor": C["bg"],
        "minHeight": "100vh",
        "color": C["text"],
        "padding": "0",
        "margin": "0",
    },
    children=[
        # -- Header --
        html.Div(
            style={
                "background": f"linear-gradient(90deg, {C['bg']} 0%, {C['header_bg']} 50%, {C['bg']} 100%)",
                "borderBottom": f"1px solid {C['card_border']}",
                "padding": "14px 32px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                    html.Div(
                        style={
                            "width": "40px", "height": "40px", "borderRadius": "10px",
                            "background": f"linear-gradient(135deg, {C['accent']} 0%, {C['purple']} 100%)",
                            "display": "flex", "alignItems": "center", "justifyContent": "center",
                            "fontSize": "13px", "fontWeight": "800", "color": C["bg"],
                            "fontFamily": "JetBrains Mono, monospace",
                            "boxShadow": f"0 0 16px {C['accent']}30",
                        },
                        children="fx",
                    ),
                    html.Div([
                        html.Div("fx-bot", style={
                            "fontSize": "20px", "fontWeight": "700", "letterSpacing": "1px",
                            "background": f"linear-gradient(90deg, {C['accent']}, {C['chart_green']})",
                            "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                        }),
                        html.Div("Algorithmic Trading System", style={
                            "fontSize": "11px", "letterSpacing": "0.5px", "color": C["text_muted"],
                            "marginTop": "-2px",
                        }),
                    ]),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
                    html.Div(id="engine_dot"),
                    html.Div(id="header_mode", style={
                        "padding": "4px 12px", "borderRadius": "6px", "fontSize": "11px",
                        "fontWeight": "600", "letterSpacing": "0.5px",
                    }),
                    html.Div(id="header_clock", style={
                        "color": C["text_muted"], "fontSize": "12px",
                        "fontFamily": "JetBrains Mono, Fira Code, monospace",
                    }),
                ]),
            ],
        ),

        # -- Main Content --
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto", "padding": "24px 32px"},
            children=[
                dcc.Interval(id="tick", interval=3000, n_intervals=0),

                # Row 1: Stat pills
                html.Div(
                    id="system_bar",
                    style={
                        "display": "flex", "flexWrap": "wrap", "gap": "8px",
                        "marginBottom": "24px", "padding": "14px 16px",
                        "background": f"{C['surface']}80",
                        "borderRadius": "12px", "border": f"1px solid {C['card_border']}",
                    },
                ),

                # Row 1.5: Dashboard guide
                card([
                    section_header("Quick Guide", "What each metric means"),
                    html.Ul([
                        html.Li("Engine dot: green = heartbeat fresh (<15s); red = stale/offline."),
                        html.Li("Price: latest mid (bid/ask average). Change = move since 00:00 UTC."),
                        html.Li("Paper P&L: realized P&L from closed paper trades only."),
                        html.Li("Quotes/Signals/Trades: raw counts in DB (tables filter by symbol)."),
                        html.Li("Account panel: balance = start balance + realized P&L."),
                    ], style={
                        "margin": "0", "paddingLeft": "18px",
                        "color": C["text_muted"], "fontSize": "12px",
                        "lineHeight": "1.6",
                    }),
                ], padding="16px 20px"),


                # Row 2: TradingView live chart (full width)
                card([
                    # Price header
                    html.Div(id="price_header", style={"marginBottom": "8px"}),
                    html.Div(
                        style={"height": "540px", "marginTop": "12px"},
                        children=[
                            html.Iframe(
                                srcDoc=TRADINGVIEW_WIDGET_HTML,
                                style={
                                    "width": "100%",
                                    "height": "100%",
                                    "border": "none",
                                    "borderRadius": "8px",
                                },
                            ),
                        ],
                    ),
                ], padding="20px 20px 8px 20px"),

                # Row 3: Account + Data overview (compact)
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        card([
                            section_header("Account", "Paper trading"),
                            html.Div(id="account_stats"),
                        ]),
                        card([
                            section_header("Trade Log", "Paper execution history"),
                            html.Div(id="trades_table"),
                        ]),
                    ],
                ),

                # Row 4: Signals + News
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                    children=[
                        card([
                            section_header("Signal Feed", "Strategy signal output"),
                            html.Div(id="signals_table"),
                        ]),
                        card([
                            section_header("News Radar", "Economic calendar events"),
                            html.Div(id="news_table"),
                        ]),
                    ],
                ),

                # Row 5: Backtests (full width)
                card([
                    section_header("Backtest Vault", "Historical strategy analysis"),
                    html.Div(id="backtest_table"),
                ]),

                # Footer
                html.Div(
                    style={
                        "textAlign": "center", "padding": "32px 0 16px",
                        "color": C["text_muted"], "fontSize": "11px",
                        "letterSpacing": "0.3px", "borderTop": f"1px solid {C['card_border']}",
                        "marginTop": "32px",
                    },
                    children=[
                        html.Span("Built with "),
                        html.Span("Codex", style={"color": C["accent"], "fontWeight": "600"}),
                        html.Span(" + "),
                        html.Span("Python", style={"color": C["chart_green"], "fontWeight": "600"}),
                        html.Span("  /  cTrader Open API  /  EURUSD Strategies  /  "),
                        html.Span("Paper Mode", style={
                            "color": C["orange"], "fontWeight": "600",
                            "padding": "2px 8px", "borderRadius": "4px",
                            "border": f"1px solid {C['orange']}40",
                        }),
                    ],
                ),
            ],
        ),
    ],
)


# ====================================================================
# Callbacks
# ====================================================================
@app.callback(
    Output("engine_dot", "children"),
    Output("header_mode", "children"),
    Output("header_mode", "style"),
    Output("header_clock", "children"),
    Output("system_bar", "children"),
    Output("price_header", "children"),
    Output("account_stats", "children"),
    Output("signals_table", "children"),
    Output("trades_table", "children"),
    Output("news_table", "children"),
    Output("backtest_table", "children"),
    Input("tick", "n_intervals"),
)
def update_panels(_):
    now = datetime.now(timezone.utc)

    # Engine dot (compact heartbeat indicator in header)
    row = fetch_latest_heartbeat()
    if row:
        stale = is_stale(row[0])
        dot_color = C["red"] if stale else C["chart_green"]
        dot_title = "Offline" if stale else "Online"
    else:
        dot_color = C["orange"]
        dot_title = "No daemon"
    engine_dot = html.Span(
        title=dot_title,
        style={
            "width": "8px", "height": "8px", "borderRadius": "50%",
            "backgroundColor": dot_color, "display": "inline-block",
            "boxShadow": f"0 0 6px {dot_color}80",
            "animation": "pulse 2s infinite" if dot_color == C["chart_green"] else "none",
        },
    )

    mode_text = "Paper Mode"
    mode_style = {
        "padding": "4px 12px", "borderRadius": "6px", "fontSize": "11px",
        "fontWeight": "600", "letterSpacing": "0.5px",
        "background": f"{C['orange']}20", "color": C["orange"],
        "border": f"1px solid {C['orange']}50",
    }
    clock = now.strftime("%Y-%m-%d  %H:%M:%S UTC")

    # Stat pills
    stats = fetch_account_stats()
    pills = [
        stat_pill("Quotes", f"{stats['total_quotes']:,}", C["accent"]),
        stat_pill("Signals", str(stats["total_signals"]), C["purple"]),
        stat_pill("Trades", str(stats["closed_trades"]), C["chart_green"]),
        stat_pill("Open", str(stats["open_positions"]),
                  C["chart_green"] if stats["open_positions"] > 0 else C["text_muted"]),
        stat_pill("News", str(stats["total_events"]), C["yellow"]),
        stat_pill("Backtests", str(stats["total_backtests"]), C["accent_dim"]),
    ]

    # Price header (Symbol + P&L)
    total_pnl = fetch_paper_pnl()
    pnl_positive = total_pnl >= 0
    pnl_color = C["chart_green"] if pnl_positive else C["red"]
    pnl_sign = "+" if pnl_positive else ""

    latest, first = fetch_price_stats()
    if latest and first:
        bid, ask = _normalize_price(float(latest[0])), _normalize_price(float(latest[1]))
        open_bid, open_ask = _normalize_price(float(first[0])), _normalize_price(float(first[1]))
        mid = (bid + ask) / 2
        open_mid = (open_bid + open_ask) / 2
        change = mid - open_mid
        change_pct = (change / open_mid * 100) if open_mid else 0
        is_positive = change >= 0
        change_color = C["chart_green"] if is_positive else C["red"]
        sign = "+" if is_positive else ""

        price_header = html.Div([
            html.Div(style={
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "flex-end", "gap": "24px",
            }, children=[
                html.Div([
                    html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "12px"}, children=[
                        html.Span(SYMBOL_LABEL, style={
                            "fontSize": "18px", "fontWeight": "700", "color": C["text"],
                        }),
                        html.Span(SYMBOL_DESC, style={
                            "fontSize": "12px", "color": C["text_muted"],
                        }),
                    ]),
                    html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "12px",
                                     "marginTop": "4px"}, children=[
                        html.Span(_price_fmt(mid), style={
                            "fontSize": "32px", "fontWeight": "700", "color": C["text"],
                            "fontFamily": "JetBrains Mono, monospace",
                        }),
                        html.Span(f"{sign}{_price_fmt(change)} ({sign}{change_pct:.2f}%)", style={
                            "fontSize": "14px", "fontWeight": "600", "color": change_color,
                            "fontFamily": "JetBrains Mono, monospace",
                        }),

                        html.Span("since 00:00 UTC", style={
                            "fontSize": "10px", "color": C["text_muted"],
                            "marginLeft": "8px", "fontFamily": "JetBrains Mono, monospace",
                        }),
                    ]),
                ]),
                html.Div(style={"textAlign": "right"}, children=[
                    html.Div("Paper P&L", style={
                        "fontSize": "11px", "color": C["text_muted"], "letterSpacing": "0.6px",
                        "textTransform": "uppercase",
                    }),
                    html.Div(f"{pnl_sign}${total_pnl:,.2f}", style={
                        "fontSize": "40px", "fontWeight": "800",
                        "color": pnl_color, "fontFamily": "JetBrains Mono, monospace",
                    }),
                ]),
            ]),
        ])
    else:
        price_header = html.Div([
            html.Div(style={
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "flex-end", "gap": "24px",
            }, children=[
                html.Div([
                    html.Span(SYMBOL_LABEL, style={"fontSize": "18px", "fontWeight": "700", "color": C["text"]}),
                    html.Div("Awaiting data...", style={"color": C["text_muted"], "fontSize": "13px", "marginTop": "4px"}),
                ]),
                html.Div(style={"textAlign": "right"}, children=[
                    html.Div("Paper P&L", style={
                        "fontSize": "11px", "color": C["text_muted"], "letterSpacing": "0.6px",
                        "textTransform": "uppercase",
                    }),
                    html.Div(f"{pnl_sign}${total_pnl:,.2f}", style={
                        "fontSize": "40px", "fontWeight": "800",
                        "color": pnl_color, "fontFamily": "JetBrains Mono, monospace",
                    }),
                ]),
            ]),
        ])

    # Account (compact, no trailing zeros)
    balance_value = PAPER_START_BALANCE + total_pnl
    balance = f"${balance_value:,.2f}"
    account_div = html.Div(
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
        children=[
            _metric_box("Balance", balance, C["accent"]),
            _metric_box("Trades", str(stats["closed_trades"]), C["chart_green"]),
            _metric_box("Open", str(stats["open_positions"]),
                        C["chart_green"] if stats["open_positions"] > 0 else C["text_dim"]),
            _metric_box("Mode", "Paper", C["orange"]),
        ],
    )

    signals_div = make_data_table(fetch_recent_signals(20), page_size=6)
    trades_div = make_data_table(fetch_paper_trades(20), page_size=6)
    news_div = make_data_table(fetch_all_news(15), page_size=6)
    bt_div = make_data_table(fetch_backtest_runs(), page_size=5)

    return (engine_dot, mode_text, mode_style, clock, pills,
            price_header, account_div,
            signals_div, trades_div, news_div, bt_div)


def _metric_box(label, value, color):
    return html.Div(
        style={
            "flex": "1", "minWidth": "100px",
            "background": C["surface"], "borderRadius": "8px",
            "padding": "12px 14px", "border": f"1px solid {C['card_border']}",
            "textAlign": "center", "overflow": "hidden",
        },
        children=[
            html.Div(label, style={
                "fontSize": "10px", "color": C["text_muted"],
                "letterSpacing": "0.3px", "marginBottom": "4px",
                "fontWeight": "500",
            }),
            html.Div(value, style={
                "fontSize": "18px", "fontWeight": "700", "color": color,
                "fontFamily": "JetBrains Mono, monospace",
                "whiteSpace": "nowrap",
            }),
        ],
    )


# ====================================================================
# Main
# ====================================================================
if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8050, debug=True)
