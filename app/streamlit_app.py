"""
BundesHost — Streamlit Dashboard
ChronoX-TimeSeries · SPICED Academy Capstone 2026
"""

import warnings
import sys
from pathlib import Path

# --------------------------------------------------
# Add project root to path (for modeling imports)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# --------------------------------------------------

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modeling.data_pipeline import get_tourism_data
from modeling.feature_engineering import build_state_series
from modeling.predict import forecast_state
from modeling.config import MODEL_DIR
# --------------------------------------------------
# General settings

warnings.filterwarnings("ignore")


# --------------------------------------------------
# Colours

BG           = "#F7F8FA"
WHITE        = "#FFFFFF"
SURFACE_2    = "#F0F2F5"
BORDER       = "#E4E8EE"
BORDER_DARK  = "#C8D0DC"
TEXT         = "#0F1923"
TEXT_MID     = "#4A5568"
TEXT_DIM     = "#8A97A8"
ACCENT       = "#1A6B5A"
ACCENT_LIGHT = "#EAF4F1"
ACCENT_MID   = "#2A9D8F"
NAVY         = "#1B2A3B"

GREEN_OK     = "#16A34A"
GREEN_BG     = "#F0FDF4"
GREEN_BORDER = "#86EFAC"

AMBER        = "#D97706"
AMBER_BG     = "#FFFBEB"
AMBER_BORDER = "#FCD34D"

RED          = "#DC2626"
RED_BG       = "#FEF2F2"
RED_BORDER   = "#FCA5A5"


# --------------------------------------------------
# Model metadata (UI only)

from modeling.config import MODEL_ORDERS

# --------------------------------------------------
# Static UI lists

STATES = sorted(MODEL_ORDERS.keys())

MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]


# --------------------------------------------------
# Data paths

DATA_PATH  = BASE_DIR / "data" / "processed" / "tourism_long.csv"
from modeling.config import MODEL_DIR


# --------------------------------------------------
# Page config

st.set_page_config(
    page_title="BundesHost",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────
st.markdown(f"""
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
  .bh-legend-line-dashed {{
    background: linear-gradient(90deg, {NAVY} 50%, transparent 50%);
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

  /* ── Alt cards v2 (new design) ── */
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
""", unsafe_allow_html=True)



# ── Helpers ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


# --------------------------------------------------
# Import forecasting logic (from predict.py)

from modeling.predict import forecast_state
from modeling.feature_engineering import build_state_series

# --------------------------------------------------
# Build time series (ONLY for plotting history)


# --------------------------------------------------
# Hosting Capacity Score

def compute_hcs(event_size: int, peak: float) -> float:
    if event_size <= 0 or peak <= 0:
        return 50.0

    return round(
        min(100.0, max(0.0, (peak / event_size) * 50)),
        1
    )


def hcs_meta(score: float) -> tuple:
    if score >= 70:
        return "✅", "Likely Feasible", "bh-banner-go"
    elif score >= 40:
        return "⚠️", "Proceed with Caution", "bh-banner-caution"

    return "🚫", "Capacity Risk", "bh-banner-no"


# --------------------------------------------------
# Formatting helper (UI only)

def fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"

    return f"{v:.0f}"

# ── Charts ───────────────────────────────────────────

def make_forecast_chart(series, fcast, state, horizon):

    # fix order
    series = series.sort_index()

    # --------------------------------------------------
    # Last 36 months of history

    tail = series.tail(36)

    # Convert PeriodIndex → Timestamp (safe)
    hdates = pd.to_datetime(tail.index)


    # --------------------------------------------------
    # Create figure

    fig = go.Figure()

    # --------------------------------------------------
    # Historical line

    fig.add_trace(go.Scatter(
        x=hdates,
        y=tail.values,
        mode="lines",
        name="Historical",
        line=dict(color=ACCENT_MID, width=2.5),
    ))

    # --------------------------------------------------
    # Forecast (smooth connection)

    if not fcast.empty:

        fdates = pd.to_datetime(fcast["date"])

        # last historical point
        last_hist_date = hdates[-1]
        last_hist_value = tail.values[-1]

        # --------------------------------------------------
        # Connection line (Dec → Jan)

        last_hist_date = hdates[-1]
        last_hist_value = tail.values[-1]

        first_forecast_date = fdates[0]
        first_forecast_value = fcast["forecast"].iloc[0]

        fig.add_trace(go.Scatter(
          x=[last_hist_date, first_forecast_date],
          y=[last_hist_value, first_forecast_value],
          mode="lines",
          line=dict(color=ACCENT_MID, width=2.5),
          showlegend=False,
      ))
        # --------------------------------------------------
        # Forecast line

        fig.add_trace(go.Scatter(
          x=fdates,
          y=fcast["forecast"],
          mode="lines+markers",
          name="Forecast",
          line=dict(color=NAVY, width=2.5, dash="dash"),
          marker=dict(size=5, color=NAVY),
      ))

      # ── Split line (optional) ────────
        split_date = fdates.min()

        fig.add_shape(
           type="line",
           x0=split_date,
           x1=split_date,
           y0=0,
           y1=1,
           xref="x",
           yref="paper",
           line=dict(color="gray", dash="dot", width=1),
           opacity=0.4
        )

        fig.add_annotation(
           x=split_date,
           y=1,
           yref="paper",
           text="Forecast starts",
           showarrow=False,
           yshift=10,
           font=dict(size=10, color="gray")
       )
    # --------------------------------------------------
    # Layout

    fig.update_layout(
        title=dict(
            text=f"{state} — {horizon}-Month Tourist Arrivals Forecast",
            font=dict(color=TEXT_MID, size=13, family="DM Sans")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE_2,
        font=dict(color=TEXT_MID, size=11, family="DM Mono"),

        xaxis=dict(
            gridcolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title=None,
            linecolor=BORDER,
            range=[
                hdates.min(),
                fdates.max() if not fcast.empty else hdates.max()
            ]
        ),

        yaxis=dict(
            gridcolor=BORDER,
            tickfont=dict(color=TEXT_DIM),
            title="Monthly Arrivals",
            tickformat=","
        ),

        legend=dict(
            font=dict(color=TEXT_MID, size=10),
            bgcolor="rgba(0,0,0,0)"
        ),

        margin=dict(l=0, r=0, t=50, b=0),
        height=360,
        hovermode="x unified",
        showlegend=False,  # Hide plotly legend, we'll use custom legend below
    )

    return fig


def make_allstates_chart(df):

    # --------------------------------------------------
    # Extract latest arrivals per state

    rows = []

    for state in STATES:

        state_data = (
            df[df["state"] == state]
            .sort_values("date")["arrivals"]
        )

        if len(state_data) > 0:
            rows.append({
                "State": state,
                "Arrivals": float(state_data.iloc[-1])
            })

    adf = (
        pd.DataFrame(rows)
        .sort_values("Arrivals", ascending=True)
    )


    # --------------------------------------------------
    # Create bar chart

    fig = go.Figure(go.Bar(
        x=adf["Arrivals"],
        y=adf["State"],
        orientation="h",

        marker=dict(
            color=adf["Arrivals"],
            colorscale=[[0, ACCENT_LIGHT], [1, ACCENT]],
            line=dict(color="rgba(0,0,0,0)")
        ),

        text=adf["Arrivals"].apply(fmt),
        textfont=dict(color=TEXT, size=10),
        textposition="outside",
    ))


    # --------------------------------------------------
    # Layout

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE_2,

        font=dict(color=TEXT_MID, family="DM Mono"),

        xaxis=dict(
            gridcolor=BORDER,
            tickformat=",",
            tickfont=dict(color=TEXT_DIM, size=9),
            title=None
        ),

        yaxis=dict(
            tickfont=dict(color=TEXT, size=9)
        ),

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
st.markdown(f"""
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
""", unsafe_allow_html=True)

## ── Query panel ──────────────────────────────────────

st.markdown('<div class="bh-panel">', unsafe_allow_html=True)
st.markdown('<div class="bh-panel-label">Configure your analysis</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 3, 1])

with c1:
    selected_state = st.selectbox(
        "Federal State",
        STATES,
        index=STATES.index("Berlin") if "Berlin" in STATES else 0
    )

with c2:
    horizon_choice = st.selectbox(
        "Forecast Horizon (months)",
        ["1 month", "3 months", "5 months", "7 months",
         "9 months", "12 months", "Custom"],
        index=5,
    )

with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Analyze →", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# Set default values (hidden from user)

event_size = 50_000  # Default event size
target_month = "July"  # Default target month


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
    # Load data

    with st.spinner("Loading data…"):
        try:
            df = load_data()
            data_ok = True
        except FileNotFoundError:
            st.error(f"Data not found at `{DATA_PATH}`.")
            data_ok = False


    # --------------------------------------------------
    # Forecast pipeline

    if data_ok:

        # History (for chart + latest value)
        series = build_state_series(df, selected_state)
        latest = float(series.iloc[-1]) if len(series) else 0.0

        # Forecast (from predict.py → already DataFrame!)
        with st.spinner("Running forecast…"):
            fcast = forecast_state(selected_state, horizon)

    else:
        fcast = pd.DataFrame()
        latest = 0.0


    # --------------------------------------------------
    # Peak calculation

    if not fcast.empty:
        peak_row = fcast.loc[fcast["forecast"].idxmax()]
        peak_month = peak_row["date"].strftime("%B %Y")
        peak_value = peak_row["forecast"]
    else:
        peak = latest
        peak_month = "Latest"


    # --------------------------------------------------
    # Score

    score = compute_hcs(event_size, peak_value)
    emoji, label, css = hcs_meta(score)


    # --------------------------------------------------
    # Metric cards (only 2 cards, no banner)

    st.markdown(f"""
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
""", unsafe_allow_html=True)


    # --------------------------------------------------
    # Forecast Chart (full width, no gauge)

    st.markdown('<div class="bh-section">Forecast</div>', unsafe_allow_html=True)

    st.plotly_chart(
        make_forecast_chart(series, fcast, selected_state, horizon),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Legend card (below chart)
    st.markdown(
        f"""
        <div class="bh-legend">
            <div class="bh-legend-item">
                <div class="bh-legend-line bh-legend-line-solid"></div>
                <span>Historical</span>
            </div>
            <div class="bh-legend-item">
                <div class="bh-legend-line bh-legend-line-dashed"></div>
                <span>Forecast</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------
    # Forecast table

    if not fcast.empty:

        st.markdown('<div class="bh-section">Forecast Table</div>', unsafe_allow_html=True)

        disp = fcast.copy()

        # format date
        disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%b %Y")

        disp.columns = [
           "Month",
           "Forecast",
           "Lower CI (80%)",
           "Upper CI (80%)"
        ]

        for col in disp.columns[1:]:
           disp[col] = disp[col].apply(lambda x: f"{x:,.0f}")

        st.dataframe(disp, use_container_width=True, hide_index=True)

    # --------------------------------------------------
    # Scope note

    st.markdown(
        '<div class="bh-scope"><strong>Scope note:</strong> The Hosting Capacity Score '
        'is derived from time-series forecasts of tourist arrivals only. Accommodation '
        'stock, transport links, and venue capacity are planned stretch goals. '
        'Treat results as indicative, not operational.</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------
    # Alternative states (new design with gradient colors)

    st.markdown(
        '<div class="bh-section">Alternative States — Top 3 by Latest Arrivals</div>',
        unsafe_allow_html=True,
    )

    alt = sorted(
        [
            {
                "State": s,
                "Latest": float(build_state_series(df, s).iloc[-1])
            }
            for s in STATES
            if s != selected_state and len(build_state_series(df, s))
        ],
        key=lambda x: x["Latest"],
        reverse=True,
    )[:3]

    # Color gradient (darkest to lightest)
    rank_colors = [
        "#1A6B5A",  # Rank 1 - darkest green (ACCENT)
        "#528B7F",  # Rank 2 - medium green (removed the 'r' typo)
        "#679B8F",  # Rank 3 - lighter green (added missing #)
    ]

    acols = st.columns(3)

    for i, (col, row) in enumerate(zip(acols, alt)):
        with col:
            bg_color = rank_colors[i]
            st.markdown(
                f'<div class="bh-alt-card-v2">'
                f'<div class="bh-alt-card-v2-left" style="background: {bg_color};">'
                f'<div class="bh-alt-card-v2-rank">#{i+1}</div>'
                f'</div>'
                f'<div class="bh-alt-card-v2-right">'
                f'<div class="bh-alt-card-v2-name">{row["State"]}</div>'
                f'<div class="bh-alt-card-v2-sub">Latest: {fmt(row["Latest"])}/mo</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


    # --------------------------------------------------
    # All states chart

    st.markdown(
        '<div class="bh-section">Compare All States — Latest Arrivals</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        make_allstates_chart(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )

# ── Footer ────────────────────────────────────────────
st.markdown("""
<div class="bh-footer">
  <strong>BundesHost</strong> · ChronoX-TimeSeries · SPICED Academy Capstone 2026<br>
  <span style="font-size:11px;">Python · Streamlit · Statsmodels · Plotly</span>
</div>
""", unsafe_allow_html=True)