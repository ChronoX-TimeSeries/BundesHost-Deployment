"""
BundesHost Streamlit App.

Tourism forecasting and hosting capacity analysis for German states.
Frontend-only: this app talks to the BundesHost API via HTTP and contains
no modeling or data-access logic of its own.
"""

import os
import warnings

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------------------------------
# General settings

warnings.filterwarnings("ignore")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# --------------------------------------------------
# Colours

BG = "#F7F8FA"
WHITE = "#FFFFFF"
SURFACE_2 = "#F0F2F5"
BORDER = "#E4E8EE"
BORDER_DARK = "#C8D0DC"
TEXT = "#0F1923"
TEXT_MID = "#4A5568"
TEXT_DIM = "#8A97A8"
ACCENT = "#1A6B5A"
ACCENT_LIGHT = "#EAF4F1"
ACCENT_MID = "#2A9D8F"
NAVY = "#1B2A3B"

GREEN_OK = "#16A34A"
GREEN_BG = "#F0FDF4"
GREEN_BORDER = "#86EFAC"

AMBER = "#D97706"
PURPLE = "#7C3AED"  # backfill_validated: model prediction vs actual
AMBER_BG = "#FFFBEB"
AMBER_BORDER = "#FCD34D"

RED = "#DC2626"
RED_BG = "#FEF2F2"
RED_BORDER = "#FCA5A5"


# --------------------------------------------------
# Page config

st.set_page_config(
    page_title="BundesHost",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── CSS ─────────────────────────────────────────────
st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
    background: {BG}; color: {TEXT};
    font-family: 'DM Sans', 'Segoe UI', sans-serif;
  }}
  [data-testid="stAppViewContainer"] > .main {{ background: {BG}; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding-top: 0; padding-bottom: 3rem; max-width: 1200px; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-thumb {{ background: {BORDER_DARK}; border-radius: 3px; }}

  /* ── Nav ── */
  .bh-nav {{
    display: flex; align-items: center;
    padding: 22px 0 18px; background: {BG};
  }}
  .bh-logo {{
    display: flex; align-items: baseline;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 36px; font-weight: 800;
    letter-spacing: -1px; line-height: 1;
  }}
  .bh-logo-bundes {{ color: {NAVY}; }}
  .bh-logo-host {{ color: {ACCENT_MID}; position: relative; }}
  .bh-logo-host::after {{
    content: ''; position: absolute; left: 0; bottom: -3px;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, {ACCENT_MID}, {ACCENT_LIGHT});
    border-radius: 2px;
  }}
  .bh-logo-dot {{
    width: 8px; height: 8px; background: {ACCENT_MID};
    border-radius: 50%; margin-left: 3px; margin-bottom: 6px; flex-shrink: 0;
  }}

  /* ── Hero ── */
  .bh-hero {{
    position: relative; border-radius: 20px; overflow: hidden;
    margin: 24px 0; min-height: 340px;
    background: linear-gradient(135deg, {NAVY} 0%, #2A4A6B 50%, {ACCENT} 100%);
    display: flex; align-items: center; padding: 48px 52px;
  }}
  .bh-hero-overlay {{
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
  }}
  .bh-hero-content {{ position: relative; z-index: 1; max-width: 560px; }}
  .bh-hero-eyebrow {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.85);
    letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 20px;
  }}
  .bh-hero h1 {{
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.8rem; font-weight: 800; color: #FFFFFF;
    line-height: 1.1; letter-spacing: -1px; margin-bottom: 14px;
  }}
  .bh-hero p {{ font-size: 15px; color: rgba(255,255,255,0.72); line-height: 1.65; max-width: 440px; }}
  .bh-hero-badge {{
    position: absolute; right: 48px; top: 48px;
    background: rgba(255,255,255,0.1); backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px; padding: 16px 20px; text-align: center;
  }}
  .bh-hero-badge-num {{
    font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500;
    color: #FFFFFF; line-height: 1;
  }}
  .bh-hero-badge-label {{ font-size: 11px; color: rgba(255,255,255,0.6); margin-top: 4px; }}

  /* ── Panel ── */
  .bh-panel {{ background: transparent; border: none; padding: 0 0 24px 0; margin-bottom: 8px; }}
  .bh-panel-label {{
    font-size: 15px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: {TEXT_DIM}; margin-bottom: 18px;
  }}

  /* ── Widget labels ── */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stNumberInput"] label,
  div[data-testid="stSlider"] label {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
    color: {TEXT_DIM} !important;
  }}

  /* ── Selectbox ── */
  [data-baseweb="select"] > div {{
    background: {WHITE} !important; border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important; color: {TEXT} !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
  }}
  [data-baseweb="select"] > div:focus-within {{
    border-color: {ACCENT_MID} !important;
    box-shadow: 0 0 0 3px {ACCENT_LIGHT} !important;
  }}
  [data-baseweb="select"] svg {{
    fill: {ACCENT_MID} !important; color: {ACCENT_MID} !important;
    width: 18px !important; height: 18px !important;
  }}

  /* ── Number input ── */
  [data-testid="stNumberInput"] [data-baseweb="input"] > div {{
    background: {WHITE} !important; border: 1.5px solid {BORDER} !important;
    border-radius: 10px !important; min-height: 42px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
  }}
  [data-testid="stNumberInput"] input {{
    border: none !important; outline: none !important; box-shadow: none !important;
    padding-left: 12px !important; color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
  }}
  [data-testid="stNumberInput"] [data-baseweb="input"] > div:focus-within {{
    border-color: {ACCENT_MID} !important;
    box-shadow: 0 0 0 3px {ACCENT_LIGHT} !important;
  }}
  [data-testid="stNumberInput"] button {{ display: none !important; }}

  /* ── Analyze button ── */
  div.stButton > button {{
    background: {NAVY}; color: {WHITE}; border: none; border-radius: 10px;
    font-family: 'DM Sans', sans-serif; font-size: 14px; font-weight: 700;
    padding: 0.65rem 1.5rem; width: 100%; letter-spacing: 0.01em;
    transition: background 0.18s; box-shadow: 0 2px 8px rgba(27,42,59,0.25);
  }}
  div.stButton > button:hover {{ background: #259284; }}
  div.stButton > button:focus {{
    outline: none !important; box-shadow: 0 0 0 3px {ACCENT_LIGHT} !important;
  }}

  /* ── Metric cards ── */
  .bh-cards {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 24px; }}
  .bh-card {{
    background: {WHITE}; border: 1.5px solid {BORDER}; border-radius: 14px;
    padding: 24px 26px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); transition: box-shadow 0.2s, transform 0.2s;
    min-height: 130px;
  }}
  .bh-card:hover {{ box-shadow: 0 8px 28px rgba(42,157,143,0.15); transform: translateY(-2px); }}
  .bh-card-icon {{ font-size: 22px; margin-bottom: 12px; }}
  .bh-card-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: {TEXT_DIM}; margin-bottom: 8px;
  }}
  .bh-card-value {{ font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 600; color: {NAVY}; line-height: 1.1; }}
  .bh-card-sub {{ font-size: 12px; color: {TEXT_DIM}; margin-top: 6px; }}

  /* ── Legend ── */
  .bh-legend {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 16px;
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
  }}
  .bh-legend-item {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: {TEXT_MID};
    font-weight: 500;
  }}
  .bh-legend-line {{
    width: 32px;
    height: 3px;
    border-radius: 2px;
  }}
  .bh-legend-line-solid {{
    background: {ACCENT_MID};
  }}
  .bh-legend-line-dashed-amber {{
    background: linear-gradient(90deg, {AMBER} 50%, transparent 50%);
    background-size: 8px 3px;
  }}
  .bh-legend-line-dashed-navy {{
    background: linear-gradient(90deg, {NAVY} 50%, transparent 50%);
    background-size: 8px 3px;
  }}
  .bh-legend-line-dashed-purple {{
    background: linear-gradient(90deg, {PURPLE} 50%, transparent 50%);
    background-size: 8px 3px;
  }}

  /* ── Section label ── */
  .bh-section {{
    font-size: 15px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
    color: {TEXT_DIM}; padding-bottom: 10px; border-bottom: 1.5px solid {BORDER};
    margin: 28px 0 16px;
  }}

  /* ── Expanders ── */
  .stExpander {{
    background: {WHITE} !important; border: 1.5px solid {BORDER} !important;
    border-radius: 12px !important; box-shadow: 0 1px 6px rgba(0,0,0,0.03) !important;
  }}

  /* ── Scope note ── */
  .bh-scope {{
    background: {ACCENT_LIGHT}; border-left: 3px solid {ACCENT_MID};
    border-radius: 0 10px 10px 0; padding: 12px 16px; font-size: 12px;
    color: {ACCENT}; margin: 24px 0; line-height: 1.65;
  }}

  /* ── Alt cards v2 ── */
  .bh-alt-card-v2 {{
    background: {WHITE};
    border: 1.5px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
    display: flex;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    transition: box-shadow 0.2s, transform 0.2s;
    min-height: 100px;
  }}
  .bh-alt-card-v2:hover {{
    box-shadow: 0 8px 28px rgba(42,157,143,0.15);
    transform: translateY(-2px);
  }}
  .bh-alt-card-v2-left {{
    width: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }}
  .bh-alt-card-v2-rank {{
    font-family: 'DM Sans', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: rgba(255,255,255,0.95);
    line-height: 1;
  }}
  .bh-alt-card-v2-right {{
    flex: 1;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}
  .bh-alt-card-v2-name {{
    font-size: 18px;
    font-weight: 700;
    color: {NAVY};
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .bh-alt-card-v2-sub {{
    font-size: 13px;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
  }}

  /* ── Footer ── */
  .bh-footer {{
    text-align: center; padding: 32px 0 8px;
    border-top: 1.5px solid {BORDER}; margin-top: 40px;
    font-size: 12px; color: {TEXT_DIM}; line-height: 2;
  }}
  .bh-footer strong {{ font-family: 'Playfair Display', serif; color: {NAVY}; font-size: 14px; }}

  hr {{ border-color: {BORDER} !important; }}
  [data-testid="stDataFrame"] {{ border: 1.5px solid {BORDER}; border-radius: 10px; }}
</style>
""",
    unsafe_allow_html=True,
)


# ── API client helpers ─────────────────────────────────


def _api_get(path: str, params: dict | None = None) -> dict:
    """Single point of contact with the API. Raises on non-2xx."""
    response = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=30.0)
    response.raise_for_status()
    return response.json()


@st.cache_data(show_spinner=False)
def fetch_states() -> list[str]:
    return _api_get("/states")


@st.cache_data(show_spinner=False)
def fetch_summary() -> dict:
    return _api_get("/summary")


@st.cache_data(show_spinner=False)
def fetch_history(state: str, last_n: int = 36) -> pd.DataFrame:
    data = _api_get(f"/history/{state}", params={"last_n": last_n})
    df = pd.DataFrame(data["history"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_forecast(state: str, horizon: int) -> pd.DataFrame:
    """Not cached — forecast depends on 'today', which changes over time."""
    data = _api_get(f"/forecast/{state}", params={"horizon": horizon})
    df = pd.DataFrame(data["forecast"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Hosting Capacity Score ─────────────────────────────


def compute_hcs(event_size: int, peak: float) -> float:
    if event_size <= 0 or peak <= 0:
        return 50.0
    return round(min(100.0, max(0.0, (peak / event_size) * 50)), 1)


def hcs_meta(score: float) -> tuple:
    if score >= 70:
        return "✅", "Likely Feasible", "bh-banner-go"
    elif score >= 40:
        return "⚠️", "Proceed with Caution", "bh-banner-caution"
    return "🚫", "Capacity Risk", "bh-banner-no"


# ── Formatting helper ─────────────────────────────────


def fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:.0f}"


# ── Charts ───────────────────────────────────────────


def make_forecast_chart(history_df, fcast_df, state, horizon):
    """
    Plot history (green) + backfill (amber dashed) + future (navy dashed).
    Connector lines between segments are drawn separately so the colors stay clean.
    The current month (today) is marked with a large hollow circle.
    """

    fig = go.Figure()

    # --------------------------------------------------
    # Historical (real data) — solid green

    hdates = history_df["date"]
    hvalues = history_df["arrivals"]

    fig.add_trace(
        go.Scatter(
            x=hdates,
            y=hvalues,
            mode="lines+markers",
            name="Historical",
            line=dict(color=ACCENT_MID, width=2.5),
            marker=dict(size=5, color=ACCENT_MID),
        )
    )

    # --------------------------------------------------
    # Split forecast

    validated_df = fcast_df[fcast_df["type"] == "backfill_validated"]
    backfill_df = fcast_df[fcast_df["type"] == "backfill"]
    future_df = fcast_df[fcast_df["type"] == "future"]

    last_hist_date = hdates.iloc[-1]
    last_hist_value = float(hvalues.iloc[-1])

    # --------------------------------------------------
    # Connector 1: last historical -> first forecast point (validated if it
    # exists, otherwise the gap backfill). Colored to match the segment it
    # leads into.

    # Only connect into the gap backfill (amber). The validated segment
    # (purple) is left detached on purpose: it is a backtest of the model
    # against real data, not a continuation of the historical line.
    if validated_df.empty and not backfill_df.empty:
        first_seg_df, first_color = backfill_df, AMBER
    else:
        first_seg_df, first_color = None, None

    if first_seg_df is not None:
        fig.add_trace(
            go.Scatter(
                x=[last_hist_date, first_seg_df["date"].iloc[0]],
                y=[last_hist_value, float(first_seg_df["forecast"].iloc[0])],
                mode="lines",
                line=dict(color=first_color, width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------
    # Backfill (validated) segment: model predicted these months and we now
    # have real data for them — a live backtest. Purple dashed.

    connector_last_date = last_hist_date
    connector_last_value = last_hist_value

    if not validated_df.empty:
        fig.add_trace(
            go.Scatter(
                x=validated_df["date"],
                y=validated_df["forecast"],
                mode="lines+markers",
                name="Model prediction (vs actual)",
                line=dict(color=PURPLE, width=2.5, dash="dash"),
                marker=dict(size=5, color=PURPLE),
            )
        )
        connector_last_date = validated_df["date"].iloc[-1]
        connector_last_value = float(validated_df["forecast"].iloc[-1])

    # --------------------------------------------------
    # Connector: last validated -> first gap backfill (amber line, no marker)

    if not validated_df.empty and not backfill_df.empty:
        fig.add_trace(
            go.Scatter(
                x=[connector_last_date, backfill_df["date"].iloc[0]],
                y=[connector_last_value, float(backfill_df["forecast"].iloc[0])],
                mode="lines",
                line=dict(color=AMBER, width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------
    # Backfill (gap) segment (amber dashed)

    if not backfill_df.empty:
        fig.add_trace(
            go.Scatter(
                x=backfill_df["date"],
                y=backfill_df["forecast"],
                mode="lines+markers",
                name="Backfill (gap)",
                line=dict(color=AMBER, width=2.5, dash="dash"),
                marker=dict(size=5, color=AMBER),
            )
        )
        connector_last_date = backfill_df["date"].iloc[-1]
        connector_last_value = float(backfill_df["forecast"].iloc[-1])

    # --------------------------------------------------
    # Connector 2: last backfill -> first future (navy line, no marker)

    if not future_df.empty:
        fig.add_trace(
            go.Scatter(
                x=[connector_last_date, future_df["date"].iloc[0]],
                y=[connector_last_value, float(future_df["forecast"].iloc[0])],
                mode="lines",
                line=dict(color=AMBER, width=2.5, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------
    # Future segment (navy dashed)

    if not future_df.empty:
        fig.add_trace(
            go.Scatter(
                x=future_df["date"],
                y=future_df["forecast"],
                mode="lines+markers",
                name="Future",
                line=dict(color=NAVY, width=2.5, dash="dash"),
                marker=dict(size=5, color=NAVY),
            )
        )

    # --------------------------------------------------
    # Current-month marker (large hollow circle on the first future point)

    if not future_df.empty:
        today_row = future_df.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[today_row["date"]],
                y=[float(today_row["forecast"])],
                mode="markers",
                name="Current month",
                marker=dict(
                    size=14,
                    color="rgba(0,0,0,0)",
                    line=dict(color=NAVY, width=2.5),
                ),
                hovertemplate="Current month: %{x|%b %Y}<extra></extra>",
                showlegend=False,
            )
        )

    # --------------------------------------------------
    # Layout

    fdates = fcast_df["date"]
    x_min = hdates.min()
    x_max = fdates.max() if not fcast_df.empty else hdates.max()

    fig.update_layout(
        title=dict(
            text=f"{state} — {horizon}-Month Tourist Arrivals Forecast",
            font=dict(color=TEXT_MID, size=13, family="DM Sans"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE_2,
        font=dict(color=TEXT_MID, size=11, family="DM Mono"),
        xaxis=dict(
            gridcolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title=None,
            linecolor=BORDER,
            range=[x_min, x_max],
        ),
        yaxis=dict(
            gridcolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title="Monthly Arrivals",
            tickformat=",",
        ),
        legend=dict(font=dict(color=TEXT_MID, size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=50, b=0),
        height=360,
        hovermode="x unified",
        showlegend=False,
    )

    return fig


def make_allstates_chart(summary_items):
    """Bar chart of latest arrivals across all states."""

    adf = pd.DataFrame(summary_items).rename(
        columns={"state": "State", "latest_arrivals": "Arrivals"}
    )
    adf = adf.sort_values("Arrivals", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=adf["Arrivals"],
            y=adf["State"],
            orientation="h",
            marker=dict(
                color=adf["Arrivals"],
                colorscale=[[0, ACCENT_LIGHT], [1, ACCENT]],
                line=dict(color="rgba(0,0,0,0)"),
            ),
            text=adf["Arrivals"].apply(fmt),
            textfont=dict(color=TEXT, size=10),
            textposition="outside",
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE_2,
        font=dict(color=TEXT_MID, family="DM Mono"),
        xaxis=dict(
            gridcolor=BORDER, tickformat=",", tickfont=dict(color=TEXT_DIM, size=9), title=None
        ),
        yaxis=dict(tickfont=dict(color=TEXT, size=9)),
        margin=dict(l=0, r=60, t=10, b=0),
        height=500,
        showlegend=False,
    )

    return fig


# ── Nav ──────────────────────────────────────────────
st.markdown(
    """
    <div class="bh-nav">
      <div class="bh-logo">
        <span class="bh-logo-bundes">Bundes</span>
        <span class="bh-logo-host">Host</span>
        <div class="bh-logo-dot"></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Hero ─────────────────────────────────────────────
st.markdown(
    """
<div class="bh-hero">
  <div class="bh-hero-overlay"></div>
  <div class="bh-hero-content">
    <div class="bh-hero-eyebrow">🏔️ &nbsp;Germany · 16 Federal States</div>
    <h1>Can your state handle the crowd?</h1>
    <p>Data-driven hosting capacity forecasts powered by SARIMA
    time-series models trained on 30 years of Eurostat tourism data.</p>
  </div>
  <div class="bh-hero-badge">
    <div class="bh-hero-badge-num">16</div>
    <div class="bh-hero-badge-label">States covered</div>
    <div style="margin-top:14px;">
      <div class="bh-hero-badge-num">30+</div>
      <div class="bh-hero-badge-label">Years of data</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ── Bootstrapping (fetch states once) ─────────────────

try:
    STATES = fetch_states()
except httpx.HTTPError as e:
    st.error(f"Cannot reach the BundesHost API at {API_BASE_URL}. ({e})")
    st.stop()


# ── Query panel ──────────────────────────────────────

st.markdown('<div class="bh-panel">', unsafe_allow_html=True)
st.markdown('<div class="bh-panel-label">Configure your analysis</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 3, 1])

with c1:
    selected_state = st.selectbox(
        "Federal State", STATES, index=STATES.index("Berlin") if "Berlin" in STATES else 0
    )

with c2:
    horizon_choice = st.selectbox(
        "Forecast Horizon (months)",
        ["1 month", "3 months", "5 months", "7 months", "9 months", "12 months", "Custom"],
        index=5,
    )

with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze →", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Defaults (hidden from user)

event_size = 50_000


# --------------------------------------------------
# Resolve horizon value

horizon_map = {
    "1 month": 1,
    "3 months": 3,
    "5 months": 5,
    "7 months": 7,
    "9 months": 9,
    "12 months": 12,
}

if horizon_choice == "Custom":
    horizon = st.number_input(
        "Enter custom horizon (months)",
        min_value=1,
        max_value=24,
        value=12,
        step=1,
        label_visibility="visible",
    )
else:
    horizon = horizon_map[horizon_choice]


# ── Results ──────────────────────────────────────────
if analyze:

    # --------------------------------------------------
    # Fetch history + forecast + summary

    try:
        with st.spinner("Loading history…"):
            history_df = fetch_history(selected_state, last_n=36)

        with st.spinner("Running forecast…"):
            fcast_df = fetch_forecast(selected_state, horizon)

        with st.spinner("Loading summary…"):
            summary = fetch_summary()

        api_ok = True
    except httpx.HTTPError as e:
        st.error(f"API error: {e}")
        api_ok = False

    if api_ok:

        latest = float(history_df["arrivals"].iloc[-1]) if len(history_df) else 0.0

        # Peak from the FUTURE part of the forecast (what users care about)
        future_df = fcast_df[fcast_df["type"] == "future"]
        if not future_df.empty:
            peak_row = future_df.loc[future_df["forecast"].idxmax()]
        else:
            peak_row = fcast_df.loc[fcast_df["forecast"].idxmax()]
        peak_value = float(peak_row["forecast"])
        peak_month = peak_row["date"].strftime("%B %Y")

        # Score (unused visually but kept for parity)
        score = compute_hcs(event_size, peak_value)
        emoji, label, css = hcs_meta(score)

        # --------------------------------------------------
        # Metric cards

        st.markdown(
            f"""
<div class="bh-cards">
  <div class="bh-card">
    <div class="bh-card-label">Latest Arrivals (monthly)</div>
    <div class="bh-card-value">{fmt(latest)}</div>
    <div class="bh-card-sub">{selected_state} · most recent</div>
  </div>
  <div class="bh-card">
    <div class="bh-card-label">Forecasted Peak</div>
    <div class="bh-card-value">{fmt(peak_value)}</div>
    <div class="bh-card-sub">Peak at {peak_month}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # --------------------------------------------------
        # Forecast chart

        st.markdown('<div class="bh-section">Forecast</div>', unsafe_allow_html=True)

        st.plotly_chart(
            make_forecast_chart(history_df, fcast_df, selected_state, horizon),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Legend (3 entries now)
        st.markdown(
            """
            <div class="bh-legend">
                <div class="bh-legend-item">
                    <div class="bh-legend-line bh-legend-line-solid"></div>
                    <span>Historical (real data)</span>
                </div>
                <div class="bh-legend-item">
                    <div class="bh-legend-line bh-legend-line-dashed-purple"></div>
                    <span>Model prediction (months we now have data for)</span>
                </div>
                <div class="bh-legend-item">
                    <div class="bh-legend-line bh-legend-line-dashed-amber"></div>
                    <span>Backfill (gap since training)</span>
                </div>
                <div class="bh-legend-item">
                    <div class="bh-legend-line bh-legend-line-dashed-navy"></div>
                    <span>Future forecast</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------
        # Forecast table

        if not fcast_df.empty:
            st.markdown('<div class="bh-section">Forecast Table</div>', unsafe_allow_html=True)

            disp = fcast_df.copy()
            disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%b %Y")
            disp = disp[["date", "type", "forecast", "lower_ci", "upper_ci"]]
            disp.columns = ["Month", "Type", "Forecast", "Lower CI (80%)", "Upper CI (80%)"]

            for col in ["Forecast", "Lower CI (80%)", "Upper CI (80%)"]:
                disp[col] = disp[col].apply(lambda x: f"{x:,.0f}")

            st.dataframe(disp, use_container_width=True, hide_index=True)

        # --------------------------------------------------
        # Scope note

        st.markdown(
            '<div class="bh-scope"><strong>Scope note:</strong> The Hosting Capacity Score '
            "is derived from time-series forecasts of tourist arrivals only. Accommodation "
            "stock, transport links, and venue capacity are planned stretch goals. "
            "Treat results as indicative, not operational.</div>",
            unsafe_allow_html=True,
        )

        # --------------------------------------------------
        # Alternative states (top 3, excluding selected)

        st.markdown(
            '<div class="bh-section">Alternative States — Top 3 by Latest Arrivals</div>',
            unsafe_allow_html=True,
        )

        alt = sorted(
            [s for s in summary["states"] if s["state"] != selected_state],
            key=lambda x: x["latest_arrivals"],
            reverse=True,
        )[:3]

        rank_colors = ["#1A6B5A", "#528B7F", "#679B8F"]

        acols = st.columns(3)
        for i, (col, row) in enumerate(zip(acols, alt)):
            with col:
                bg_color = rank_colors[i]
                st.markdown(
                    f'<div class="bh-alt-card-v2">'
                    f'<div class="bh-alt-card-v2-left" style="background: {bg_color};">'
                    f'<div class="bh-alt-card-v2-rank">#{i+1}</div>'
                    f"</div>"
                    f'<div class="bh-alt-card-v2-right">'
                    f'<div class="bh-alt-card-v2-name">{row["state"]}</div>'
                    f'<div class="bh-alt-card-v2-sub">Latest: {fmt(row["latest_arrivals"])}/mo</div>'
                    f"</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # --------------------------------------------------
        # All states chart

        st.markdown(
            '<div class="bh-section">Compare All States — Latest Arrivals</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            make_allstates_chart(summary["states"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ── Footer ────────────────────────────────────────────
st.markdown(
    """
<div class="bh-footer">
  <strong>BundesHost</strong> · ChronoX-TimeSeries · SPICED Academy Capstone 2026<br>
  <span style="font-size:11px;">Python · Streamlit · Statsmodels · Plotly</span>
</div>
""",
    unsafe_allow_html=True,
)
