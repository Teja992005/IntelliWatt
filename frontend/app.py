import matplotlib.pyplot as plt
import requests
import streamlit as st

st.set_page_config(
    page_title="IntelliWatt – Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://127.0.0.1:8000"
WINDOW_SIZE = 599
SHORT_WINDOW = 60

APPLIANCE_OPTIONS = ["Fridge", "Kettle", "Washing Machine", "Microwave"]
EXPERIMENT_OPTIONS = ["6sec_cnn", "1min_cnn", "1min_bigru"]

# ── Color Palette ──────────────────────────────────────────────────────────────
# #EBF4F6 — light teal-white (page background, card surfaces)
# #088395 — ocean teal       (primary accent, buttons, badges, highlights)
# #065f6e — deep teal        (hero, sidebar, section headers, deep backgrounds)
# All custom HTML uses SOLID colors — never CSS variables — so the UI looks
# identical whether the user picks Streamlit's light OR dark theme.

st.markdown(
    """
    <style>
    /* ── Josefin Sans — loaded from local assets ───────
       Streamlit serves frontend/static/ at /app/static/
    ─────────────────────────────────────────────────── */
    @font-face {
        font-family: 'Josefin Sans';
        src: url('/app/static/fonts/JosefinSans-VariableFont_wght.ttf') format('truetype');
        font-weight: 100 700;
        font-style: normal;
        font-display: swap;
    }
    @font-face {
        font-family: 'Josefin Sans';
        src: url('/app/static/fonts/JosefinSans-Italic-VariableFont_wght.ttf') format('truetype');
        font-weight: 100 700;
        font-style: italic;
        font-display: swap;
    }

    /* ── Global typography ───────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Josefin Sans', sans-serif !important;
    }

    /* ── Main container ─────────────────────────────── */
    .main .block-container {
        max-width: 1300px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        background-color: #EBF4F6 !important;
    }

    /* Force main app background */
    .stApp {
        background-color: #EBF4F6 !important;
    }

    /* ── Sidebar ────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #065f6e 0%, #044855 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #d9f0f4 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12) !important;
    }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        border-radius: 12px !important;
        border: none !important;
        background: linear-gradient(135deg, #088395, #065f6e) !important;
        color: #ffffff !important;
        font-family: 'Josefin Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.72rem 1.4rem !important;
        box-shadow: 0 4px 16px rgba(8, 131, 149, 0.38) !important;
        transition: all 0.22s ease !important;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(8, 131, 149, 0.50) !important;
        background: linear-gradient(135deg, #0a9aad, #077080) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── Tabs ────────────────────────────────────────── */
    div[data-testid="stTabs"] button {
        font-family: 'Josefin Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.05em;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #088395 !important;
        border-bottom-color: #088395 !important;
    }

    /* ── Progress bar ────────────────────────────────── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #088395, #065f6e) !important;
        border-radius: 999px !important;
    }

    /* ── Spinner ─────────────────────────────────────── */
    .stSpinner > div {
        border-top-color: #088395 !important;
    }

    /* ═══════════════════════════════════════════════════
       Custom HTML Components
       All colors are SOLID — theme-toggle safe.
    ═══════════════════════════════════════════════════ */

    /* ── Hero ────────────────────────────────────────── */
    .iw-hero {
        background: linear-gradient(135deg, #065f6e 0%, #088395 60%, #065f6e 100%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 2.2rem 2.4rem 2rem 2.4rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .iw-hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(235,244,246,0.14) 0%, transparent 70%);
        pointer-events: none;
    }
    .iw-hero::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(8,131,149,0.30) 0%, transparent 70%);
        pointer-events: none;
    }
    .iw-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(235,244,246,0.16);
        color: #d9f0f4;
        border: 1px solid rgba(235,244,246,0.25);
        border-radius: 999px;
        padding: 4px 14px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .iw-hero-title {
        font-family: 'Josefin Sans', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 0 0 0.6rem 0;
    }
    .iw-hero-title span {
        color: #a8dde5;
    }
    .iw-hero-sub {
        color: rgba(217,240,244,0.85);
        font-size: 1rem;
        max-width: 680px;
        line-height: 1.65;
    }

    /* ── Stat strip ──────────────────────────────────── */
    .iw-stat-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .iw-stat {
        background: #ffffff;
        border: 1px solid rgba(8,131,149,0.14);
        border-radius: 16px;
        padding: 1.1rem 1.2rem;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        box-shadow: 0 2px 12px rgba(6,95,110,0.07);
    }
    .iw-stat-icon {
        font-size: 1.4rem;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .iw-stat-label {
        color: #4a9aaa;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .iw-stat-value {
        color: #065f6e;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1;
    }
    .iw-stat-desc {
        color: #5aa8b8;
        font-size: 0.76rem;
        font-weight: 500;
    }

    /* ── Section header ──────────────────────────────── */
    .iw-section-header {
        background: #ffffff;
        border: 1px solid rgba(8,131,149,0.14);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border-left: 4px solid #088395;
        box-shadow: 0 2px 12px rgba(6,95,110,0.07);
    }
    .iw-section-badge {
        display: inline-block;
        background: rgba(8,131,149,0.10);
        color: #088395;
        border: 1px solid rgba(8,131,149,0.20);
        border-radius: 999px;
        padding: 2px 11px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .iw-section-title {
        color: #033e4a;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
    }
    .iw-section-copy {
        color: #2a7585;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Metric card ─────────────────────────────────── */
    .iw-metric {
        background: #ffffff;
        border: 1px solid rgba(8,131,149,0.14);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.65rem;
        box-shadow: 0 2px 10px rgba(6,95,110,0.06);
    }
    .iw-metric-label {
        color: #4a9aaa;
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .iw-metric-value {
        color: #065f6e;
        font-size: 1.52rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        font-family: 'Josefin Sans', sans-serif;
    }

    /* ── Status pills ────────────────────────────────── */
    .iw-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .iw-pill-on     { background: rgba(16,185,129,0.12); color: #0d7a55; border: 1px solid rgba(16,185,129,0.28); }
    .iw-pill-off    { background: rgba(8,131,149,0.10);  color: #088395; border: 1px solid rgba(8,131,149,0.25); }
    .iw-pill-normal { background: rgba(16,185,129,0.12); color: #0d7a55; border: 1px solid rgba(16,185,129,0.28); }
    .iw-pill-mild   { background: rgba(217,119,6,0.10);  color: #92520a; border: 1px solid rgba(217,119,6,0.25); }
    .iw-pill-severe { background: rgba(8,131,149,0.12);  color: #065f6e; border: 1px solid rgba(8,131,149,0.28); }

    /* ── Input hint ──────────────────────────────────── */
    .iw-hint {
        background: rgba(8,131,149,0.06);
        border: 1px solid rgba(8,131,149,0.16);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        color: #2a7585;
        font-size: 0.82rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .iw-hint strong { color: #088395; }

    /* ── Result placeholder panel ────────────────────── */
    .iw-result-placeholder {
        background: #ffffff;
        border: 1px solid rgba(8,131,149,0.12);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        min-height: 200px;
        box-shadow: 0 2px 10px rgba(6,95,110,0.05);
    }
    .iw-result-placeholder-label {
        color: #4a9aaa;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .iw-result-placeholder-text {
        color: #7ac4d0;
        font-size: 0.9rem;
    }

    /* ── Overview module cards ───────────────────────── */
    .iw-module-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
    .iw-module-card {
        background: #ffffff;
        border: 1px solid rgba(8,131,149,0.13);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        box-shadow: 0 2px 10px rgba(6,95,110,0.06);
        transition: box-shadow 0.2s ease;
    }
    .iw-module-icon {
        font-size: 1.6rem;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .iw-module-name {
        color: #033e4a;
        font-size: 0.97rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .iw-module-desc {
        color: #2a7585;
        font-size: 0.82rem;
        line-height: 1.55;
    }

    /* ── Sidebar inline code ────────────────────────── */
    .iw-sb-code {
        background: rgba(255,255,255,0.15);
        color: #a8dde5;
        border-radius: 6px;
        padding: 1px 8px;
        font-size: 0.78rem;
        font-family: 'Inter', monospace;
        font-weight: 500;
    }

    /* ── Divider ─────────────────────────────────────── */
    .iw-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.10);
        margin: 1.2rem 0;
    }

    /* ── Responsiveness ──────────────────────────────── */
    @media (max-width: 900px) {
        .iw-hero-title { font-size: 1.9rem; }
        .iw-stat-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .iw-module-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_series(raw_text):
    cleaned = raw_text.replace("\n", ",").replace("\t", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    return [float(part) for part in parts]


def post_json(endpoint, payload):
    return requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=60)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="iw-metric">
            <div class="iw-metric-label">{label}</div>
            <div class="iw-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_pill(text, variant):
    icons = {
        "on":     "🟢",
        "off":    "🔴",
        "normal": "✅",
        "mild":   "⚠️",
        "severe": "🚨",
    }
    icon = icons.get(variant, "")
    st.markdown(
        f'<div class="iw-pill iw-pill-{variant}">{icon} {text}</div>',
        unsafe_allow_html=True,
    )


def section_header(badge, title, copy):
    st.markdown(
        f"""
        <div class="iw-section-header">
            <div class="iw-section-badge">{badge}</div>
            <div class="iw-section-title">{title}</div>
            <div class="iw-section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_placeholder(message="Submit the form on the left to see results here."):
    st.markdown(
        f"""
        <div class="iw-result-placeholder">
            <div class="iw-result-placeholder-label">Results</div>
            <div class="iw-result-placeholder-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_line_chart(series, title, ylabel="Power (W)", highlight_index=None, color="#088395"):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f2fafb")

    ax.plot(series, color=color, linewidth=2, alpha=0.9)
    if highlight_index is not None:
        ax.axvline(x=highlight_index, color="#065f6e", linestyle="--", linewidth=1.4, alpha=0.65)

    ax.set_title(title, fontsize=12, fontweight="bold", color="#033e4a", pad=10)
    ax.set_xlabel("Time Index", color="#4a9aaa", fontsize=10)
    ax.set_ylabel(ylabel, color="#4a9aaa", fontsize=10)
    ax.tick_params(colors="#5aa8b8", labelsize=9)
    ax.grid(alpha=0.12, color="#c8e8ee")
    for spine in ax.spines.values():
        spine.set_color("#d0edf2")
    fig.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size:1.4rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">
                ⚡ IntelliWatt
            </div>
            <div style="color:rgba(217,240,244,0.55); font-size:0.82rem; margin-top:4px; font-weight:500;">
                Energy Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="iw-divider">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="color:rgba(217,240,244,0.75); font-size:0.83rem; line-height:1.65;">
            AI-powered energy disaggregation, forecasting, anomaly detection,
            and research experiments — all in one professional cockpit.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="iw-divider">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="color:rgba(217,240,244,0.50); font-size:0.70rem; font-weight:800;
                    text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.7rem;">
            System Config
        </div>
        """,
        unsafe_allow_html=True,
    )

    config_items = [
        ("🌐", "Backend", "127.0.0.1:8000"),
        ("🪟", "NILM Window", "599 values"),
        ("📈", "Forecast Window", "60 values"),
        ("🔍", "Anomaly Window", "60 values"),
    ]
    for icon, label, val in config_items:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:7px 0; border-bottom:1px solid rgba(255,255,255,0.06);">
                <span style="color:rgba(217,240,244,0.75); font-size:0.82rem;">{icon} {label}</span>
                <span class="iw-sb-code">{val}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="iw-divider">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="color:rgba(217,240,244,0.50); font-size:0.70rem; font-weight:800;
                    text-transform:uppercase; letter-spacing:0.10em; margin-bottom:0.7rem;">
            Input Format Tips
        </div>
        <div style="color:rgba(217,240,244,0.70); font-size:0.82rem; line-height:1.85;">
            • Comma-separated values<br>
            • New lines treated as commas<br>
            • Match the exact window size<br>
            • Tab-separated values supported
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-top:1.8rem; padding:10px 12px;
                    background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14);
                    border-radius:12px;">
            <div style="color:#a8dde5; font-size:0.78rem; font-weight:700; margin-bottom:5px;">
                🔬 Research Models
            </div>
            <div style="color:rgba(217,240,244,0.60); font-size:0.78rem; line-height:1.7;">
                6-sec CNN &nbsp;•&nbsp; 1-min CNN &nbsp;•&nbsp; 1-min BiGRU
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="iw-hero">
        <div class="iw-hero-badge">⚡ Smart Energy Analytics</div>
        <div class="iw-hero-title">IntelliWatt <span>Dashboard</span></div>
        <div class="iw-hero-sub">
            A professional energy intelligence cockpit for appliance disaggregation,
            load forecasting, anomaly detection, and NILM research — powered by your
            existing AI pipeline with zero backend changes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Stat strip ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="iw-stat-strip">
        <div class="iw-stat">
            <div class="iw-stat-icon">🔌</div>
            <div class="iw-stat-label">Modules</div>
            <div class="iw-stat-value">4</div>
            <div class="iw-stat-desc">NILM · Forecast · Anomaly · Research</div>
        </div>
        <div class="iw-stat">
            <div class="iw-stat-icon">🪟</div>
            <div class="iw-stat-label">NILM Window</div>
            <div class="iw-stat-value">599</div>
            <div class="iw-stat-desc">6-second sliding window</div>
        </div>
        <div class="iw-stat">
            <div class="iw-stat-icon">📈</div>
            <div class="iw-stat-label">Forecast Horizon</div>
            <div class="iw-stat-value">1 Step</div>
            <div class="iw-stat-desc">Next-point prediction</div>
        </div>
        <div class="iw-stat">
            <div class="iw-stat-icon">🤖</div>
            <div class="iw-stat-label">Research Models</div>
            <div class="iw-stat-value">3</div>
            <div class="iw-stat-desc">CNN · CNN · BiGRU</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────

overview, nilm_tab, forecast_tab, anomaly_tab, research_tab = st.tabs(
    ["🏠 Overview", "🔌 NILM", "📈 Forecasting", "🔍 Anomaly", "🔬 Research"]
)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with overview:
    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown(
            """
            <div class="iw-section-header">
                <div class="iw-section-badge">Platform</div>
                <div class="iw-section-title">What is IntelliWatt?</div>
                <div class="iw-section-copy">
                    IntelliWatt is a professional energy intelligence cockpit that wraps
                    your FastAPI backend with a clean, structured interface. API routes,
                    payloads, model logic and validations stay completely untouched —
                    only the presentation layer has been upgraded.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div class="iw-section-header">
                <div class="iw-section-badge">Architecture</div>
                <div class="iw-section-title">How it works</div>
                <div class="iw-section-copy">
                    Paste your mains power readings into any module, hit Submit, and
                    IntelliWatt calls the backend API, parses the JSON response, and
                    renders results with clear metrics, status indicators, and charts.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="color:#065f6e; font-size:1.05rem; font-weight:700;
                    margin: 1rem 0 0.7rem 0; letter-spacing:-0.01em;">
            🧩 Available Modules
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="iw-module-grid">
            <div class="iw-module-card">
                <div class="iw-module-icon">🔌</div>
                <div>
                    <div class="iw-module-name">NILM — Appliance Disaggregation</div>
                    <div class="iw-module-desc">
                        Predicts per-appliance power from 599 aggregate mains readings.
                        Supports Fridge, Kettle, Washing Machine, and Microwave.
                    </div>
                </div>
            </div>
            <div class="iw-module-card">
                <div class="iw-module-icon">📈</div>
                <div>
                    <div class="iw-module-name">Forecasting — Load & Bill Estimation</div>
                    <div class="iw-module-desc">
                        Uses 60 recent readings to predict the next power point, estimate
                        daily energy (kWh), and project the monthly electricity bill (₹).
                    </div>
                </div>
            </div>
            <div class="iw-module-card">
                <div class="iw-module-icon">🔍</div>
                <div>
                    <div class="iw-module-name">Anomaly Detection</div>
                    <div class="iw-module-desc">
                        Scores reconstruction error on 60 readings and classifies the
                        pattern as Normal, Mild anomaly, or Severe anomaly.
                    </div>
                </div>
            </div>
            <div class="iw-module-card">
                <div class="iw-module-icon">🔬</div>
                <div>
                    <div class="iw-module-name">Research — NILM Experiments</div>
                    <div class="iw-module-desc">
                        Compare three fridge-focused research models: 6-sec CNN,
                        1-min CNN, and 1-min BiGRU — wired directly to the backend.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#065f6e; font-size:1.05rem; font-weight:700;
                    margin-bottom:0.7rem; letter-spacing:-0.01em;">
            📊 Quick Stats
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Appliances", "4")
    with c2:
        metric_card("NILM Rate", "6 sec")
    with c3:
        metric_card("Forecasting", "1 step ahead")
    with c4:
        metric_card("Anomaly Rule", "3 000 W limit")


# ══════════════════════════════════════════════════════════════════════════════
# NILM
# ══════════════════════════════════════════════════════════════════════════════

with nilm_tab:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Disaggregation",
        "Appliance Energy Disaggregation",
        f"Paste exactly {WINDOW_SIZE} aggregated mains readings. "
        "The model predicts appliance-level power at the center of the window.",
    )

    form_col, result_col = st.columns([1, 1.1], gap="large")

    with form_col:
        appliance = st.selectbox(
            "Select Appliance",
            APPLIANCE_OPTIONS,
            key="nilm_appliance",
            help="Choose the appliance you want to disaggregate.",
        )
        power_input = st.text_area(
            f"Mains Power Values  ({WINDOW_SIZE} values required)",
            height=240,
            placeholder="Paste comma-separated or newline-separated aggregate mains values here...",
            key="nilm_input",
        )
        st.markdown(
            f'<div class="iw-hint"><strong>Tip:</strong> Enter exactly <strong>{WINDOW_SIZE}</strong> '
            "numeric values separated by commas or new lines.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        nilm_submit = st.button("⚡ Predict Appliance Power", key="nilm_submit")

    with result_col:
        if not nilm_submit:
            result_placeholder("Submit the form on the left to see NILM prediction results here.")
        else:
            if power_input.strip() == "":
                st.error("⚠️ Please enter power values before submitting.")
            else:
                try:
                    data = parse_series(power_input)
                    if len(data) != WINDOW_SIZE:
                        st.error(
                            f"❌ Expected **{WINDOW_SIZE}** values but got **{len(data)}**. "
                            "Please check your input."
                        )
                    else:
                        payload = {
                            "appliance": appliance.lower().replace(" ", "_"),
                            "data": data,
                        }
                        with st.spinner("Predicting appliance power..."):
                            response = post_json("/nilm/predict", payload)
                        if response.status_code == 200:
                            result = response.json()

                            r1, r2, r3 = st.columns(3, gap="small")
                            with r1:
                                metric_card("Predicted Power", f"{result['predicted_power']:.2f} W")
                            with r2:
                                metric_card(
                                    "Appliance",
                                    result["appliance"].replace("_", " ").title(),
                                )
                            with r3:
                                metric_card("Confidence", f"{result['confidence'] * 100:.1f}%")

                            if result["state"] == "ON":
                                status_pill("Appliance is ON", "on")
                            else:
                                status_pill("Appliance is OFF", "off")

                            st.progress(int(result["confidence"] * 100))
                            st.pyplot(
                                build_line_chart(
                                    data,
                                    "Input Power Window",
                                    highlight_index=WINDOW_SIZE // 2,
                                    color="#088395",
                                )
                            )
                        else:
                            st.error(f"Backend error: {response.text}")
                except ValueError:
                    st.error("❌ Invalid input — make sure all values are numbers.")


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════════════════════

with forecast_tab:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Forecasting",
        "Load Forecast & Bill Estimation",
        "Provide 60 recent mains readings to predict the next power point, "
        "estimate daily energy use, and project the monthly electricity bill.",
    )

    form_col, result_col = st.columns([1, 1.1], gap="large")

    with form_col:
        forecast_input = st.text_area(
            f"Mains Power Values  ({SHORT_WINDOW} values required)",
            height=240,
            placeholder="Paste 60 recent power readings (comma or newline separated)...",
            key="forecast_input",
        )
        st.markdown(
            f'<div class="iw-hint"><strong>Tip:</strong> Enter exactly <strong>{SHORT_WINDOW}</strong> '
            "recent power readings to generate a forecast.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        forecast_submit = st.button("📈 Predict & Estimate Bill", key="forecast_submit")

    with result_col:
        if not forecast_submit:
            result_placeholder("Submit the form on the left to see forecast and bill estimation results here.")
        else:
            if forecast_input.strip() == "":
                st.error("⚠️ Please enter power values before submitting.")
            else:
                try:
                    data = parse_series(forecast_input)
                    if len(data) != SHORT_WINDOW:
                        st.error(
                            f"❌ Expected **{SHORT_WINDOW}** values but got **{len(data)}**. "
                            "Please check your input."
                        )
                    else:
                        payload = {"data": data}
                        with st.spinner("Generating forecast..."):
                            response = post_json("/forecast/predict", payload)
                        if response.status_code == 200:
                            result = response.json()

                            r1, r2, r3 = st.columns(3, gap="small")
                            with r1:
                                metric_card(
                                    "Predicted Next Power",
                                    f"{result['predicted_next_power_watts']:.2f} W",
                                )
                            with r2:
                                metric_card(
                                    "Daily Energy",
                                    f"{result['estimated_daily_energy_kwh']:.2f} kWh",
                                )
                            with r3:
                                metric_card(
                                    "Monthly Bill",
                                    f"₹ {result['estimated_monthly_bill_rupees']:.2f}",
                                )

                            st.pyplot(
                                build_line_chart(
                                    data,
                                    "Recent Power Window (60 readings)",
                                    color="#065f6e",
                                )
                            )
                        else:
                            st.error(f"Backend error: {response.text}")
                except ValueError:
                    st.error("❌ Invalid input — make sure all values are numbers.")


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY
# ══════════════════════════════════════════════════════════════════════════════

with anomaly_tab:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Detection",
        "Anomaly Detection",
        "Submit 60 recent mains readings to score reconstruction error and "
        "classify the pattern as Normal, Mild anomaly, or Severe anomaly.",
    )

    form_col, result_col = st.columns([1, 1.1], gap="large")

    with form_col:
        anomaly_input = st.text_area(
            f"Mains Power Values  ({SHORT_WINDOW} values required)",
            height=240,
            placeholder="Paste 60 power readings for anomaly analysis...",
            key="anomaly_input",
        )
        st.markdown(
            f'<div class="iw-hint"><strong>Tip:</strong> Values above <strong>3 000 W</strong> '
            "may trigger the severe anomaly threshold.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        anomaly_submit = st.button("🔍 Detect Anomaly", key="anomaly_submit")

    with result_col:
        if not anomaly_submit:
            result_placeholder("Submit the form on the left to see anomaly detection results here.")
        else:
            if anomaly_input.strip() == "":
                st.error("⚠️ Please enter power values before submitting.")
            else:
                try:
                    data = parse_series(anomaly_input)
                    if len(data) != SHORT_WINDOW:
                        st.error(
                            f"❌ Expected **{SHORT_WINDOW}** values but got **{len(data)}**. "
                            "Please check your input."
                        )
                    else:
                        payload = {"data": data}
                        with st.spinner("Analyzing for anomalies..."):
                            response = post_json("/anomaly/detect", payload)
                        if response.status_code == 200:
                            result = response.json()
                            severity = result["severity"]

                            if severity == "normal":
                                status_pill("Normal usage detected — no anomaly found", "normal")
                            elif severity == "mild":
                                status_pill("Mild anomaly detected — review recommended", "mild")
                            else:
                                status_pill("Severe anomaly detected — action required!", "severe")

                            r1, r2, r3 = st.columns(3, gap="small")
                            with r1:
                                metric_card(
                                    "Reconstruction Error",
                                    f"{result['reconstruction_error']:.4f}",
                                )
                            with r2:
                                metric_card(
                                    "Max Power Observed",
                                    f"{result['max_power_observed']:.2f} W",
                                )
                            with r3:
                                metric_card("Safe Limit", f"{result['safe_limit']:.0f} W")

                            st.pyplot(
                                build_line_chart(
                                    data,
                                    "Power Pattern Analysis",
                                    color="#088395",
                                )
                            )
                        else:
                            st.error(f"Backend error: {response.text}")
                except ValueError:
                    st.error("❌ Invalid input — make sure all values are numbers.")


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH
# ══════════════════════════════════════════════════════════════════════════════

with research_tab:
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Research",
        "NILM Paper Experiments",
        "Compare the three fridge research models already wired to the backend: "
        "6-second CNN, 1-minute CNN, and 1-minute BiGRU.",
    )

    form_col, result_col = st.columns([1, 1.1], gap="large")

    with form_col:
        experiment_model = st.selectbox(
            "Select Experiment Model",
            EXPERIMENT_OPTIONS,
            key="experiment_model",
            help="6sec_cnn requires 599 values; 1min_cnn and 1min_bigru require 510 values.",
        )
        exp_window = 599 if experiment_model == "6sec_cnn" else 510

        st.markdown(
            f"""
            <div style="background:rgba(8,131,149,0.07); border:1px solid rgba(8,131,149,0.18);
                        border-radius:10px; padding:0.6rem 0.9rem; margin-bottom:0.8rem;">
                <span style="color:#088395; font-size:0.82rem; font-weight:600;">
                    📐 Required input: <strong>{exp_window} values</strong>
                    for <strong>{experiment_model}</strong>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        experiment_input = st.text_area(
            f"Aggregate Power Values  ({exp_window} values required)",
            height=220,
            placeholder=f"Paste {exp_window} aggregate power values for {experiment_model}...",
            key="experiment_input",
        )
        st.markdown(
            '<div class="iw-hint"><strong>Note:</strong> Window size updates automatically '
            "when you switch models above.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        research_submit = st.button("🔬 Run NILM Experiment", key="research_submit")

    with result_col:
        if not research_submit:
            result_placeholder("Submit the form on the left to see NILM experiment results here.")
        else:
            if experiment_input.strip() == "":
                st.error("⚠️ Please enter power values before submitting.")
            else:
                try:
                    data = parse_series(experiment_input)
                    if len(data) != exp_window:
                        st.error(
                            f"❌ Model **{experiment_model}** requires **{exp_window}** values "
                            f"but got **{len(data)}**."
                        )
                    else:
                        payload = {"data": data}
                        with st.spinner(f"Running {experiment_model} experiment..."):
                            response = post_json(
                                f"/nilm/experiments/{experiment_model}",
                                payload,
                            )
                        if response.status_code == 200:
                            result = response.json()
                            prediction = result["prediction"]

                            r1, r2 = st.columns(2, gap="small")
                            with r1:
                                metric_card("Model Type", result["model_type"])
                            with r2:
                                metric_card("Prediction Shape", str(result["prediction_shape"]))

                            st.pyplot(
                                build_line_chart(
                                    data,
                                    "Input — Aggregate Power",
                                    color="#088395",
                                )
                            )
                            st.pyplot(
                                build_line_chart(
                                    prediction,
                                    "Output — Predicted Appliance Power",
                                    color="#065f6e",
                                )
                            )
                        else:
                            st.error(f"Backend error: {response.text}")
                except ValueError:
                    st.error("❌ Invalid input — make sure all values are numbers.")
