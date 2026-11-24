"""
GenAI Indonesia ABSA & Discrepancy Analyzer
Prototype Dashboard — AM552B Data Science of IS (FTI UKSW)
Kelompok 46 | Arthur · Brendan · Tiara

Arsitektur UI:
  - Sidebar: Judul, Badge Institusi, Metric Cards, 4 Aspek ABSA
  - 3 Tabs: Dashboard Komparasi | Analisis Tunggal | Prediksi Batch
"""

import io
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from core_predictor import (
    predict_single_review,
    predict_batch_dataframe,
    load_trained_artifact,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GenAI Indonesia ABSA Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (Deep Navy + clean cards) ─────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2137 0%, #1a3a5c 100%);
    border-right: 1px solid #1e4976;
}
[data-testid="stSidebar"] * { color: #e8f4fd !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem; font-weight: 700;
    color: #ffffff !important;
    line-height: 1.4; margin-bottom: 4px;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
[data-testid="stMetricLabel"] { font-size: 0.75rem; opacity: 0.8; }
[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 800; }
[data-testid="stMetricDelta"] { font-size: 0.72rem; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #e2e8f0;
    gap: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600; font-size: 0.88rem;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
    color: #64748b;
    transition: all 0.2s;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1F4E79 !important;
    border-bottom: 2px solid #1F4E79 !important;
    background: transparent !important;
}

/* ── Section headers ── */
.section-header {
    font-size: 1rem; font-weight: 700;
    color: #1e3a5f; margin-bottom: 4px;
}
.section-sub {
    font-size: 0.75rem; color: #64748b; margin-bottom: 12px;
}

/* ── Badges ── */
.badge-inst {
    display: inline-block;
    background: rgba(31,78,121,0.15);
    border: 1px solid rgba(31,78,121,0.4);
    color: #1F4E79 !important;
    border-radius: 8px; font-size: 0.72rem;
    font-weight: 600; padding: 4px 10px;
    margin: 6px 0 14px 0;
}

/* ── Sentiment pill ── */
.pill-pos  { background:#dcfce7; color:#166534; border-radius:20px; padding:3px 12px; font-weight:700; font-size:0.85rem; }
.pill-neg  { background:#fee2e2; color:#991b1b; border-radius:20px; padding:3px 12px; font-weight:700; font-size:0.85rem; }
.pill-neu  { background:#fef9c3; color:#854d0e; border-radius:20px; padding:3px 12px; font-weight:700; font-size:0.85rem; }

/* ── Discrepancy alert ── */
.disc-alert {
    background:#fff7ed; border-left:4px solid #f97316;
    border-radius:0 8px 8px 0; padding:10px 14px;
    font-size:0.82rem; color:#7c2d12; margin-top:8px;
}

/* ── Aspect chip ── */
.asp-chip {
    display:inline-block;
    background:rgba(31,78,121,0.1);
    border:1px solid rgba(31,78,121,0.3);
    border-radius:20px; font-size:0.75rem;
    font-weight:600; padding:3px 10px; margin:2px;
    color:#1F4E79;
}

/* ── Footer ── */
.footer-bar {
    text-align:center; color:#94a3b8;
    font-size:0.75rem; padding:20px 0 8px 0;
    border-top:1px solid #e2e8f0; margin-top:30px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1;
    border-radius: 12px; padding: 10px;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #1F4E79 !important;
    border: none !important; color: #fff !important;
    border-radius: 8px !important; font-weight: 700 !important;
    padding: 10px 24px !important; font-size: 0.9rem !important;
    transition: background 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #163d60 !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & DATA
# ══════════════════════════════════════════════════════════════════════════════
APP_LIST = ["ChatGPT", "Gemini", "Character.AI", "Perplexity", "Copilot", "DeepSeek", "Claude", "Poe"]

# Statistik dataset aktual (dari EDA notebook 01)
APP_STATS = {
    "ChatGPT":      {"ulasan": 22023, "avg_rating": 4.21, "pct_pos": 72.1, "pct_neg": 15.4, "pct_disc": 4.21,
                     "a1": 0.74, "a2": 0.61, "a3": 0.58, "a4": 0.32},
    "Gemini":       {"ulasan": 12602, "avg_rating": 4.05, "pct_pos": 69.3, "pct_neg": 17.8, "pct_disc": 3.87,
                     "a1": 0.71, "a2": 0.55, "a3": 0.62, "a4": 0.48},
    "Character.AI": {"ulasan": 11982, "avg_rating": 1.58, "pct_pos": 28.4, "pct_neg": 55.2, "pct_disc": 6.14,
                     "a1": 0.52, "a2": 0.78, "a3": 0.69, "a4": 0.21},
    "Perplexity":   {"ulasan":  5214, "avg_rating": 3.92, "pct_pos": 64.5, "pct_neg": 20.1, "pct_disc": 4.55,
                     "a1": 0.80, "a2": 0.67, "a3": 0.44, "a4": 0.29},
    "Copilot":      {"ulasan":  7419, "avg_rating": 4.10, "pct_pos": 70.2, "pct_neg": 16.3, "pct_disc": 3.21,
                     "a1": 0.68, "a2": 0.59, "a3": 0.55, "a4": 0.38},
    "DeepSeek":     {"ulasan":  4813, "avg_rating": 3.67, "pct_pos": 55.8, "pct_neg": 28.9, "pct_disc": 11.98,
                     "a1": 0.63, "a2": 0.72, "a3": 0.61, "a4": 0.44},
    "Claude":       {"ulasan":  4433, "avg_rating": 4.18, "pct_pos": 71.6, "pct_neg": 14.9, "pct_disc": 2.95,
                     "a1": 0.76, "a2": 0.63, "a3": 0.49, "a4": 0.35},
    "Poe":          {"ulasan":  2735, "avg_rating": 3.88, "pct_pos": 61.4, "pct_neg": 22.7, "pct_disc": 5.33,
                     "a1": 0.65, "a2": 0.70, "a3": 0.52, "a4": 0.28},
}

ASPECT_LABELS = ["Akurasi AI", "Harga/Quota", "Performa/UI", "Bahasa Lokal"]
ASPECT_KEYS   = ["a1", "a2", "a3", "a4"]

COLORS = {
    "navy":      "#1F4E79",
    "blue":      "#2563EB",
    "teal":      "#0EA5E9",
    "green":     "#16A34A",
    "orange":    "#F97316",
    "red":       "#DC2626",
    "purple":    "#7C3AED",
    "gray":      "#64748B",
}
APP_COLOR_MAP = {
    "ChatGPT":      "#10A37F",
    "Gemini":       "#4285F4",
    "Character.AI": "#FF6B6B",
    "Perplexity":   "#6366F1",
    "Copilot":      "#00B4D8",
    "DeepSeek":     "#F97316",
    "Claude":       "#D4A017",
    "Poe":          "#A855F7",
}

PILL_MAP = {
    "positive": '<span class="pill-pos">✅ Positif</span>',
    "neutral":  '<span class="pill-neu">⚠️ Netral</span>',
    "negative": '<span class="pill-neg">❌ Negatif</span>',
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_artifact()


def radar_chart(apps_selected: list[str]) -> go.Figure:
    fig = go.Figure()
    for app in apps_selected:
        vals = [APP_STATS[app][k] for k in ASPECT_KEYS]
        vals_closed = vals + [vals[0]]
        cats_closed = ASPECT_LABELS + [ASPECT_LABELS[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed, name=app,
            fill="toself",
            line=dict(color=APP_COLOR_MAP.get(app, "#888"), width=2),
            fillcolor=APP_COLOR_MAP.get(app, "#888").replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in APP_COLOR_MAP.get(app, "") else APP_COLOR_MAP.get(app, "#888") + "26",
            opacity=0.85,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=11, color="#334155")),
            bgcolor="rgba(248,250,252,0.6)",
        ),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center", font=dict(size=11)),
        margin=dict(l=50, r=50, t=30, b=60),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )
    return fig


def keyword_bar(key_drivers: list) -> go.Figure:
    """Diverging horizontal bar chart dari key_drivers output predictor."""
    if not key_drivers:
        fig = go.Figure()
        fig.add_annotation(text="Tidak ada keyword tersedia", showarrow=False, font=dict(size=13, color="#94a3b8"))
        fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    words  = [kd["word"]  for kd in key_drivers]
    scores = [kd["score"] for kd in key_drivers]
    colors_bar = [COLORS["blue"] if s >= 0 else COLORS["red"] for s in scores]

    fig = go.Figure(go.Bar(
        x=scores, y=words,
        orientation="h",
        marker_color=colors_bar,
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Log-odds: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1.5)
    fig.update_layout(
        xaxis=dict(title="Log-Odds Coefficient", zeroline=True, zerolinecolor="#94a3b8",
                   gridcolor="#f1f5f9", tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        height=280,
        margin=dict(l=10, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    return fig


def sentiment_heatmap(probabilities: dict, active_aspects: list) -> go.Figure:
    """3×4 heatmap: baris=sentimen, kolom=aspek."""
    sentiments = ["positive", "neutral", "negative"]
    prob_vals  = [probabilities.get(s, 0.0) for s in sentiments]

    # Bobot aspek — aspek aktif → scaling penuh, inaktif → dimmed
    aspect_active_flags = [
        ("A1_Akurasi_Kecerdasan" in active_aspects),
        ("A2_Harga_Kuota"        in active_aspects),
        ("A3_Performa_UI_Server" in active_aspects),
        ("A4_Bahasa_Indonesia"   in active_aspects),
    ]

    z = []
    for prob in prob_vals:
        row = [prob * (1.0 if flag else 0.25) for flag in aspect_active_flags]
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=ASPECT_LABELS,
        y=["positive", "neutral", "negative"],
        colorscale=[
            [0.0,  "#f8fafc"],
            [0.25, "#bfdbfe"],
            [0.5,  "#3b82f6"],
            [0.75, "#1d4ed8"],
            [1.0,  "#1e3a8a"],
        ],
        zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b> × %{x}<br>Score: %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9)),
    ))
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=11), side="bottom"),
        yaxis=dict(tickfont=dict(size=11)),
        height=240,
        margin=dict(l=10, r=60, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def rating_dist_chart() -> go.Figure:
    """Stacked bar: distribusi rating per aplikasi (dari EDA)."""
    # Data aktual dari EDA notebook 01 (proporsi positif/netral/negatif)
    apps_short = [a[:8] for a in APP_LIST]
    pos_pct = [APP_STATS[a]["pct_pos"]                              for a in APP_LIST]
    neg_pct = [APP_STATS[a]["pct_neg"]                              for a in APP_LIST]
    neu_pct = [100 - APP_STATS[a]["pct_pos"] - APP_STATS[a]["pct_neg"] for a in APP_LIST]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positif", x=APP_LIST, y=pos_pct,
                         marker_color="#22c55e", hovertemplate="%{x}: %{y:.1f}%<extra>Positif</extra>"))
    fig.add_trace(go.Bar(name="Netral",  x=APP_LIST, y=neu_pct,
                         marker_color="#facc15", hovertemplate="%{x}: %{y:.1f}%<extra>Netral</extra>"))
    fig.add_trace(go.Bar(name="Negatif", x=APP_LIST, y=neg_pct,
                         marker_color="#ef4444", hovertemplate="%{x}: %{y:.1f}%<extra>Negatif</extra>"))
    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Aplikasi AI", tickfont=dict(size=11)),
        yaxis=dict(title="Proporsi (%)", tickfont=dict(size=11), gridcolor="#f1f5f9"),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        height=340, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def discrepancy_bar() -> go.Figure:
    """Bar chart Rating-Text Discrepancy % per app."""
    disc_pct = [APP_STATS[a]["pct_disc"] for a in APP_LIST]
    colors_disc = [COLORS["red"] if v > 7 else COLORS["orange"] if v > 4 else COLORS["teal"] for v in disc_pct]
    fig = go.Figure(go.Bar(
        x=APP_LIST, y=disc_pct,
        marker_color=colors_disc, marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Discrepancy: %{y:.2f}%<extra></extra>",
        text=[f"{v:.1f}%" for v in disc_pct], textposition="outside",
    ))
    fig.add_hline(y=5.0, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Threshold 5%", annotation_position="top right")
    fig.update_layout(
        yaxis=dict(title="Discrepancy Rate (%)", gridcolor="#f1f5f9", tickfont=dict(size=11)),
        xaxis=dict(tickfont=dict(size=11)),
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def radar_multi_app(apps_selected: list[str]) -> go.Figure:
    """Radar chart dengan overlay beberapa aplikasi (Dashboard tab)."""
    return radar_chart(apps_selected)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## GenAI Indonesia ABSA &\nDiscrepancy Analyzer")
    st.markdown('<span class="badge-inst">🏫 S1 Sistem Informasi · FTI UKSW</span>', unsafe_allow_html=True)

    st.divider()

    # ── Model metrics ──────────────────────────────────────────────────────
    st.markdown("**📊 Performa Model Produksi**")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Macro F1", "0.6502", delta="▲ vs Baseline", delta_color="normal")
    with col_m2:
        st.metric("ROC-AUC", "0.8831", delta="✅ Target", delta_color="normal")

    st.caption("Model: Tuned Logistic Regression · TF-IDF 1-2gram · Balanced")

    st.divider()

    # ── 4 Aspek ABSA ──────────────────────────────────────────────────────
    st.markdown("**🧩 4 Dimensi Aspek ABSA:**")
    aspects_sidebar = [
        ("🎯", "A1: Akurasi AI",    "Kecerdasan, halusinasi, kualitas jawaban"),
        ("💰", "A2: Harga/Quota",   "Biaya, limit, subscription, iklan"),
        ("⚡", "A3: Performa/UI",   "Kecepatan, server, bug, tampilan"),
        ("🌏", "A4: Bahasa Lokal",  "Bahasa Indonesia, terjemahan, konteks lokal"),
    ]
    for icon, label, desc in aspects_sidebar:
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:8px;margin:6px 0;'>"
            f"<span style='font-size:1.1rem;'>{icon}</span>"
            f"<div><b style='font-size:0.82rem;'>{label}</b>"
            f"<div style='font-size:0.72rem;opacity:0.75;'>{desc}</div></div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Dataset stats ──────────────────────────────────────────────────────
    st.markdown("**📁 Dataset**")
    st.markdown(
        "<div style='font-size:0.8rem;line-height:2;'>"
        "🔢 <b>71.221</b> ulasan · 8 aplikasi<br>"
        "📅 Google Play Store Indonesia<br>"
        "⚖️ Class imbalance: ⭐5 (63.96%)"
        "</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — 3 TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  Dashboard Komparasi",
    "🔍  Analisis Tunggal",
    "📁  Prediksi Batch",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD KOMPARASI
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">📊 Dashboard Komparasi 8 Aplikasi GenAI Indonesia</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Statistik EDA dari 71.221 ulasan Google Play Store — CRISP-DM Phase: Data Understanding</div>', unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    with kc1: st.metric("Total Ulasan",    "71.221",   delta="8 App GenAI")
    with kc2: st.metric("⭐5 Dominan",     "63.96%",   delta="Class Imbalance", delta_color="inverse")
    with kc3: st.metric("Discrepancy",     "3.271",    delta="4.59% ulasan")
    with kc4: st.metric("Best Avg Rating", "ChatGPT",  delta="4.21 ⭐")
    with kc5: st.metric("Worst Rating",    "Char.AI",  delta="1.58 ⭐", delta_color="inverse")

    st.divider()

    # ── Radar + Distribusi Sentimen ───────────────────────────────────────
    col_r1, col_r2 = st.columns([1, 1])

    with col_r1:
        st.markdown('<div class="section-header">🕸️ 4-Dimension ABSA Radar Chart</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Overlay coverage aspek layanan per aplikasi</div>', unsafe_allow_html=True)
        selected_apps_radar = st.multiselect(
            "Pilih aplikasi untuk overlay:",
            APP_LIST,
            default=["ChatGPT", "Gemini", "Character.AI"],
            max_selections=5,
            key="dash_radar",
        )
        if selected_apps_radar:
            st.plotly_chart(radar_chart(selected_apps_radar), use_container_width=True, key="dash_radar_fig")
        else:
            st.info("Pilih minimal 1 aplikasi untuk menampilkan radar chart.")

    with col_r2:
        st.markdown('<div class="section-header">📊 Distribusi Sentimen per Aplikasi</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Proporsi kelas Positif / Netral / Negatif (dari label EDA)</div>', unsafe_allow_html=True)
        st.plotly_chart(rating_dist_chart(), use_container_width=True, key="dash_sent_dist")

    st.divider()

    # ── Discrepancy + Tabel Komparasi ─────────────────────────────────────
    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown('<div class="section-header">⚠️ Rating-Text Discrepancy Rate</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Persen ulasan yang skor bintang ≠ polaritas teks (DeepSeek tertinggi: 11.98%)</div>', unsafe_allow_html=True)
        st.plotly_chart(discrepancy_bar(), use_container_width=True, key="dash_disc_fig")

    with col_d2:
        st.markdown('<div class="section-header">🗂️ Tabel Statistik per Aplikasi</div>', unsafe_allow_html=True)
        df_table = pd.DataFrame([
            {
                "Aplikasi": a,
                "Ulasan":   f"{APP_STATS[a]['ulasan']:,}",
                "Avg ⭐":   f"{APP_STATS[a]['avg_rating']:.2f}",
                "% Positif": f"{APP_STATS[a]['pct_pos']:.1f}%",
                "% Negatif": f"{APP_STATS[a]['pct_neg']:.1f}%",
                "% Discrep.": f"{APP_STATS[a]['pct_disc']:.2f}%",
            }
            for a in APP_LIST
        ])
        st.dataframe(df_table, use_container_width=True, hide_index=True, height=290)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: ANALISIS TUNGGAL
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🔍 Analisis Tunggal Ulasan Pengguna</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Multi-class sentiment · ABSA 4 dimensi · Rating-Text Discrepancy Detection · Explainability (Log-Odds)</div>', unsafe_allow_html=True)

    # ── 1. Form Input ─────────────────────────────────────────────────────
    st.markdown("**1. Form Input Ulasan Pengguna**")
    fc1, fc2, fc3 = st.columns([3, 1, 1])
    with fc1:
        review_text = st.text_area(
            "Teks ulasan:",
            placeholder="Contoh: Aplikasi ini sangat membantu untuk belajar coding, tapi sering error saat server sibuk dan limit hariannya terlalu cepat habis...",
            height=90, label_visibility="collapsed",
        )
    with fc2:
        selected_app = st.selectbox(
            "Aplikasi AI", ["(Tidak dipilih)"] + APP_LIST,
            label_visibility="visible",
        )
    with fc3:
        selected_rating = st.selectbox(
            "5-Star Rating", ["(Tidak dipilih)", 1, 2, 3, 4, 5],
            label_visibility="visible",
        )

    btn_analyze = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=False)

    # ── 2. Hasil Analisis ─────────────────────────────────────────────────
    if btn_analyze:
        if not review_text.strip():
            st.warning("⚠️ Silakan masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner("Memproses pipeline preprocessing → TF-IDF → Logistic Regression..."):
                star_val = None if selected_rating == "(Tidak dipilih)" else int(selected_rating)
                result   = predict_single_review(review_text, star_rating=star_val)

            predicted_sent  = result.get("predicted_sentiment", "unknown")
            probabilities   = result.get("probabilities",  {})
            active_aspects  = result.get("active_aspects", [])
            key_drivers     = result.get("key_drivers",    [])
            disc_type       = result.get("discrepancy_type", None)
            processed_text  = result.get("processed_text", "")

            # ── Ringkasan hasil ──────────────────────────────────────────
            sr1, sr2, sr3 = st.columns([2, 1, 1])
            with sr1:
                pill = PILL_MAP.get(predicted_sent, f"<b>{predicted_sent}</b>")
                st.markdown(f"**Sentimen Terdeteksi:** {pill}", unsafe_allow_html=True)
                if disc_type and disc_type != "none":
                    st.markdown(
                        f'<div class="disc-alert">⚠️ <b>Discrepancy Terdeteksi:</b> {disc_type}<br>'
                        f'Rating yang dipilih (<b>⭐{star_val}</b>) tidak konsisten dengan polaritas teks ulasan.</div>',
                        unsafe_allow_html=True,
                    )
            with sr2:
                p_pos = probabilities.get("positive", 0)
                p_neu = probabilities.get("neutral",  0)
                p_neg = probabilities.get("negative", 0)
                st.metric("P(Positif)",  f"{p_pos:.1%}")
                st.metric("P(Netral)",   f"{p_neu:.1%}")
                st.metric("P(Negatif)",  f"{p_neg:.1%}")
            with sr3:
                if active_aspects:
                    st.markdown("**Aspek terdeteksi:**")
                    asp_name_map = {
                        "A1_Akurasi_Kecerdasan": "🎯 A1: Akurasi AI",
                        "A2_Harga_Kuota":        "💰 A2: Harga/Quota",
                        "A3_Performa_UI_Server": "⚡ A3: Performa/UI",
                        "A4_Bahasa_Indonesia":   "🌏 A4: Bahasa Lokal",
                    }
                    for asp in active_aspects:
                        label = asp_name_map.get(asp, asp)
                        st.markdown(f'<span class="asp-chip">{label}</span>', unsafe_allow_html=True)
                else:
                    st.info("Tidak ada aspek spesifik terdeteksi.")

            st.divider()

            # ── 3 panel visualisasi ───────────────────────────────────────
            st.markdown("**2. Visualisasi Analisis Multi-Dimensi**")
            vc1, vc2, vc3 = st.columns(3)

            # Panel 1: Radar Chart
            with vc1:
                st.markdown('<div class="section-header">4-Dimension ABSA Radar Chart</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Coverage aspek ulasan vs aplikasi yang dipilih</div>', unsafe_allow_html=True)

                radar_apps = []
                if selected_app != "(Tidak dipilih)":
                    radar_apps = [selected_app]
                else:
                    radar_apps = ["ChatGPT", "Gemini"]  # default fallback

                # Overlay ulasan saat ini
                asp_scores_current = [
                    1.0 if "A1_Akurasi_Kecerdasan" in active_aspects else 0.1,
                    1.0 if "A2_Harga_Kuota"        in active_aspects else 0.1,
                    1.0 if "A3_Performa_UI_Server"  in active_aspects else 0.1,
                    1.0 if "A4_Bahasa_Indonesia"    in active_aspects else 0.1,
                ]

                fig_radar = go.Figure()
                for app in radar_apps:
                    vals = [APP_STATS[app][k] for k in ASPECT_KEYS]
                    vals_c = vals + [vals[0]]
                    cats_c = ASPECT_LABELS + [ASPECT_LABELS[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals_c, theta=cats_c, name=app,
                        fill="toself",
                        line=dict(color=APP_COLOR_MAP.get(app, "#888"), width=2),
                        fillcolor=APP_COLOR_MAP.get(app, "#888") + "26",
                    ))
                # ulasan saat ini
                asp_c2 = asp_scores_current + [asp_scores_current[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=asp_c2, theta=ASPECT_LABELS + [ASPECT_LABELS[0]],
                    name="Ulasan ini", fill="toself",
                    line=dict(color="#f97316", width=2.5, dash="dash"),
                    fillcolor="#f9731620",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(range=[0,1], gridcolor="#e2e8f0", tickfont=dict(size=8)),
                        angularaxis=dict(tickfont=dict(size=10)),
                    ),
                    legend=dict(font=dict(size=9), orientation="h", y=-0.25, x=0.5, xanchor="center"),
                    height=290, margin=dict(l=40, r=40, t=20, b=50),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_radar, use_container_width=True, key="single_radar")

            # Panel 2: Keyword Bar Chart
            with vc2:
                st.markdown('<div class="section-header">Diverging Keyword Importance</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Log-odds coefficient kata pendorong keputusan model</div>', unsafe_allow_html=True)
                st.plotly_chart(keyword_bar(key_drivers), use_container_width=True, key="single_kw_bar")

            # Panel 3: Heatmap
            with vc3:
                st.markdown('<div class="section-header">Sentiment × Aspek Heatmap</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Intensitas probabilitas sentimen per dimensi aspek</div>', unsafe_allow_html=True)
                st.plotly_chart(sentiment_heatmap(probabilities, active_aspects), use_container_width=True, key="single_heatmap")

            # ── Processed text expander ──────────────────────────────────
            with st.expander("🔬 Lihat teks setelah preprocessing"):
                st.code(processed_text or "(kosong)", language=None)

    else:
        # Placeholder sebelum analisis
        st.info(
            "👆 Masukkan teks ulasan di atas dan klik **🚀 Analisis Sentimen** untuk melihat hasil analisis multi-dimensi.\n\n"
            "**Contoh ulasan untuk dicoba:**\n"
            "- *\"ChatGPT sangat pintar dan membantu skripsi saya, tapi limitnya cepat habis dan harus bayar mahal\"*\n"
            "- *\"Gemini bahasa Indonesianya masih kaku, sering salah paham konteks lokal\"*\n"
            "- *\"Character AI server sering down dan tiba-tiba require usia 18+ tanpa penjelasan\"*"
        )
        # Placeholder charts (grey)
        c_ph1, c_ph2, c_ph3 = st.columns(3)
        with c_ph1:
            fig_ph = go.Figure()
            fig_ph.update_layout(
                height=240, paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
                annotations=[dict(text="4-Dimension ABSA Radar Chart<br><span style='font-size:11px;color:#94a3b8'>Tersedia setelah analisis</span>",
                                  showarrow=False, font=dict(size=13, color="#cbd5e1"), xref="paper", yref="paper", x=0.5, y=0.5)],
            )
            st.plotly_chart(fig_ph, use_container_width=True, key="ph_radar")
        with c_ph2:
            fig_ph2 = go.Figure()
            fig_ph2.update_layout(
                height=240, paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
                annotations=[dict(text="Diverging Keyword Importance<br><span style='font-size:11px;color:#94a3b8'>Tersedia setelah analisis</span>",
                                  showarrow=False, font=dict(size=13, color="#cbd5e1"), xref="paper", yref="paper", x=0.5, y=0.5)],
            )
            st.plotly_chart(fig_ph2, use_container_width=True, key="ph_kw")
        with c_ph3:
            fig_ph3 = go.Figure()
            fig_ph3.update_layout(
                height=240, paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc",
                annotations=[dict(text="Sentiment Comparison Heatmap<br><span style='font-size:11px;color:#94a3b8'>Tersedia setelah analisis</span>",
                                  showarrow=False, font=dict(size=13, color="#cbd5e1"), xref="paper", yref="paper", x=0.5, y=0.5)],
            )
            st.plotly_chart(fig_ph3, use_container_width=True, key="ph_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: PREDIKSI BATCH
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📁 Prediksi Batch via CSV Upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Upload file CSV berisi kolom <code>content</code> (teks ulasan) dan opsional <code>score</code> (rating 1-5). '
        'Model akan memproses seluruh baris secara otomatis.</div>',
        unsafe_allow_html=True,
    )

    # ── Upload zone ──────────────────────────────────────────────────────
    bc1, bc2 = st.columns([2, 1])
    with bc1:
        uploaded_file = st.file_uploader(
            "Upload file CSV ulasan:",
            type=["csv"],
            help="Pastikan ada kolom 'content' untuk teks ulasan dan opsional kolom 'score' untuk rating bintang.",
        )
    with bc2:
        st.markdown(
            "**Format CSV yang diterima:**\n"
            "```\ncontent,score\n"
            "\"Aplikasi ini bagus\",5\n"
            "\"Server sering down\",2\n```"
        )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ File berhasil dimuat: **{len(df_input):,} baris**, {len(df_input.columns)} kolom.")

            with st.expander("👁️ Preview 5 baris pertama"):
                st.dataframe(df_input.head(), use_container_width=True)

            if "content" not in df_input.columns:
                st.error("❌ Kolom **`content`** tidak ditemukan. Pastikan CSV memiliki kolom `content`.")
            else:
                score_col = "score" if "score" in df_input.columns else None
                btn_batch = st.button("⚙️ Jalankan Prediksi Batch", type="primary")

                if btn_batch:
                    with st.spinner(f"Memproses {len(df_input):,} ulasan..."):
                        df_result = predict_batch_dataframe(df_input, text_col="content", score_col=score_col)

                    st.success(f"✅ Prediksi selesai untuk **{len(df_result):,}** ulasan!")

                    # ── Summary metrics ──────────────────────────────
                    if "predicted_sentiment" in df_result.columns:
                        vc = df_result["predicted_sentiment"].value_counts()
                        bm1, bm2, bm3 = st.columns(3)
                        bm1.metric("✅ Positif",  vc.get("positive", 0), delta=f"{vc.get('positive',0)/len(df_result)*100:.1f}%")
                        bm2.metric("⚠️ Netral",   vc.get("neutral",  0), delta=f"{vc.get('neutral', 0)/len(df_result)*100:.1f}%")
                        bm3.metric("❌ Negatif",  vc.get("negative", 0), delta=f"{vc.get('negative',0)/len(df_result)*100:.1f}%")

                    # ── Pie chart distribusi ──────────────────────────
                    if "predicted_sentiment" in df_result.columns:
                        sent_counts = df_result["predicted_sentiment"].value_counts().reset_index()
                        sent_counts.columns = ["Sentimen", "Jumlah"]
                        fig_pie = px.pie(
                            sent_counts, names="Sentimen", values="Jumlah",
                            color="Sentimen",
                            color_discrete_map={"positive":"#22c55e","neutral":"#facc15","negative":"#ef4444"},
                            hole=0.4,
                        )
                        fig_pie.update_layout(height=280, margin=dict(l=10,r=10,t=20,b=10),
                                              paper_bgcolor="rgba(0,0,0,0)")
                        bpc1, bpc2 = st.columns([1,2])
                        with bpc1:
                            st.plotly_chart(fig_pie, use_container_width=True, key="batch_pie")
                        with bpc2:
                            st.markdown("**Hasil Prediksi (preview 20 baris):**")
                            display_cols = ["content"] + (["score"] if score_col else []) + ["predicted_sentiment"]
                            if "discrepancy_type" in df_result.columns:
                                display_cols.append("discrepancy_type")
                            st.dataframe(df_result[display_cols].head(20), use_container_width=True, height=240)

                    # ── Download ──────────────────────────────────────
                    csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ Download Hasil CSV",
                        data=csv_bytes,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv",
                        type="primary",
                    )

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")

    else:
        st.markdown(
            "**💡 Cara penggunaan batch prediction:**\n"
            "1. Siapkan file CSV dengan minimal kolom `content` (teks ulasan)\n"
            "2. Tambahkan kolom `score` (1-5) jika ingin deteksi Rating-Text Discrepancy\n"
            "3. Upload file dan klik **⚙️ Jalankan Prediksi Batch**\n"
            "4. Download hasil CSV dengan kolom `predicted_sentiment` + `discrepancy_type`"
        )

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer-bar">'
    '© 2026 <b>Arthur</b> · <b>Brendan</b> · <b>Tiara</b> — S1 Sistem Informasi · FTI UKSW<br>'
    '<span style="font-size:0.68rem;">AM552B Data Science of IS · Kelompok 46 · Best Model: Tuned Logistic Regression · Macro F1=0.6502</span>'
    '</div>',
    unsafe_allow_html=True,
)
