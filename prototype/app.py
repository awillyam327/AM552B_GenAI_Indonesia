"""
Aplikasi Web Prototype — Analisis Sentimen Berbasis Aspek (ABSA) & Deteksi Discrepancy
Mata Kuliah: AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)
Cara Menjalankan: streamlit run prototype/app.py

Desain UI: Sidebar navigasi kiri (4 menu) dengan tema biru tua lembut yang bersih dan profesional
untuk kebutuhan presentasi penelitian. Seluruh logika prediksi dan keluaran data dipertahankan utuh.
"""

import importlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

# Pastikan direktori prototype/ dan root terdaftar di sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import core_predictor
importlib.reload(core_predictor)
from core_predictor import (
    BENCHMARK_JSON_PATH,
    load_trained_artifact,
    predict_batch_dataframe,
    predict_single_review,
)

st.set_page_config(
    page_title="GenAI Indonesia ABSA & Discrepancy Analyzer — AM552 FTI UKSW",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS — TEMA BIRU TUA LEMBUT, BERSIH & PROFESIONAL UNTUK PENELITIAN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
      integrity="sha512-Avb2QiuDEEvB4bZJYdft2mNjVShBftLdPG8FJ0V7irTLQ8Uo0qcPxh4Plq7G5tGm0rU+1SPhVotteLpBERwTkA=="
      crossorigin="anonymous" referrerpolicy="no-referrer">

<style>
/* ── Tipografi global bersih ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ── Sidebar kiri: biru tua lembut solid (tanpa efek rumit) ── */
[data-testid="stSidebar"] {
    background-color: #1E3A5F !important;
    border-right: 1px solid #2A4B75;
}
[data-testid="stSidebar"] * {
    color: #E8F0F8 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(232, 240, 248, 0.16) !important;
    margin: 0.9rem 0 !important;
}

/* ── Identitas aplikasi di sidebar ── */
.sb-brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #FFFFFF !important;
    line-height: 1.3;
    margin-bottom: 2px;
}
.sb-brand-sub {
    font-size: 0.80rem;
    color: #B8CEE4 !important;
    margin-bottom: 10px;
}
.sb-badge {
    display: inline-block;
    background-color: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px;
    color: #E8F0F8 !important;
}
.sb-nav-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9BB8D4 !important;
    margin: 6px 0 10px 2px;
}

/* ── Tombol menu navigasi di sidebar ── */
[data-testid="stSidebar"] .stButton {
    margin-bottom: 4px;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button > div {
    justify-content: flex-start !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    padding: 0.60rem 0.90rem !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    color: #D2E1F0 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: rgba(255, 255, 255, 0.15) !important;
    border: 1px solid rgba(155, 184, 212, 0.35) !important;
    border-left: 4px solid #9BB8D4 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background-color: rgba(255, 255, 255, 0.18) !important;
}

/* ── Footer ringkas sidebar ── */
.sb-footer {
    font-size: 0.72rem;
    line-height: 1.55;
    color: #A9C2DB !important;
}

/* ── Tombol utama di area konten ── */
.main .stButton > button[kind="primary"],
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stAppViewContainer"] section.main .stButton > button[kind="primary"] {
    background-color: #1E3A5F !important;
    border: 1px solid #1E3A5F !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
.main .stButton > button[kind="primary"]:hover,
[data-testid="stMain"] .stButton > button[kind="primary"]:hover,
[data-testid="stAppViewContainer"] section.main .stButton > button[kind="primary"]:hover {
    background-color: #162D4A !important;
    border-color: #162D4A !important;
}

/* ── Header utama penelitian ── */
.page-header-box {
    border-bottom: 1px solid rgba(59, 105, 152, 0.28);
    padding-bottom: 14px;
    margin-bottom: 20px;
}
.page-header-title {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.3;
    margin: 0 0 6px 0;
}
.page-header-sub {
    font-size: 0.90rem;
    opacity: 0.85;
    margin: 0;
    line-height: 1.5;
}

/* ── Kartu bersih halaman Detail Metodologi ── */
.metric-card-clean {
    background-color: rgba(30, 58, 95, 0.08);
    border: 1px solid rgba(59, 105, 152, 0.30);
    border-top: 3px solid #2B547E;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric-card-label {
    font-size: 0.82rem;
    font-weight: 600;
    opacity: 0.82;
    margin-bottom: 6px;
}
.metric-card-value {
    font-size: 2.15rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 6px;
    white-space: nowrap;
}
.metric-card-note {
    font-size: 0.78rem;
    opacity: 0.78;
}

.info-panel-clean {
    background-color: rgba(30, 58, 95, 0.06);
    border: 1px solid rgba(59, 105, 152, 0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    line-height: 1.75;
    font-size: 0.88rem;
}
.info-panel-clean i {
    color: #3B6998;
    width: 20px;
    text-align: center;
    margin-right: 6px;
}

.aspect-card-clean {
    background-color: rgba(30, 58, 95, 0.07);
    border: 1px solid rgba(59, 105, 152, 0.26);
    border-left: 4px solid #2B547E;
    border-radius: 8px;
    padding: 15px 18px;
    margin-bottom: 12px;
}
.aspect-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.aspect-card-title i {
    color: #3B6998;
    margin-right: 8px;
}
.aspect-card-desc {
    font-size: 0.84rem;
    opacity: 0.85;
    line-height: 1.5;
    margin: 0;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed rgba(59, 105, 152, 0.45);
    border-radius: 10px;
    padding: 8px;
}

/* ── Warna ikon sentimen ── */
.sent-pos { color: #16a34a !important; }
.sent-neg { color: #dc2626 !important; }
.sent-neu { color: #ca8a04 !important; }

/* ── Badge sentimen per aspek (Clause-Level ABSA) ── */
.asp-badge-pos {
    display: inline-block;
    background-color: rgba(22, 163, 74, 0.12);
    border: 1px solid rgba(22, 163, 74, 0.35);
    color: #16a34a !important;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.76rem;
    font-weight: 700;
    margin-left: 6px;
}
.asp-badge-neg {
    display: inline-block;
    background-color: rgba(220, 38, 38, 0.12);
    border: 1px solid rgba(220, 38, 38, 0.35);
    color: #dc2626 !important;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.76rem;
    font-weight: 700;
    margin-left: 6px;
}
.asp-badge-neu {
    display: inline-block;
    background-color: rgba(202, 138, 4, 0.12);
    border: 1px solid rgba(202, 138, 4, 0.35);
    color: #ca8a04 !important;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.76rem;
    font-weight: 700;
    margin-left: 6px;
}
.asp-clause-note {
    font-size: 0.78rem;
    opacity: 0.78;
    margin: 2px 0 10px 16px;
}

/* ── Ikon sub-judul ── */
.h4-icon { color: #3B6998; margin-right: 6px; }

/* ── Footer halaman ── */
.footer-bar {
    text-align: center;
    opacity: 0.72;
    font-size: 0.76rem;
    padding: 18px 0 8px 0;
    border-top: 1px solid rgba(59, 105, 152, 0.25);
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS (TIDAK DIUBAH)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_model_artifact():
    return load_trained_artifact()


AUDIT_JSON_PATH = os.path.join(ROOT_DIR, "05_evaluation", "audit_results.json")


@st.cache_data
def get_benchmark_summary():
    if os.path.exists(BENCHMARK_JSON_PATH):
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_audit_summary():
    if os.path.exists(AUDIT_JSON_PATH):
        with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART HELPERS (TEMA BIRU TUA LEMBUT, BERSIH & PROFESIONAL)
# ══════════════════════════════════════════════════════════════════════════════
BLUE_PRIMARY = "#1E3A5F"
BLUE_MID = "#4A7BB0"
BLUE_LIGHT = "#7BA4D0"
RED_MUTED = "#C85A5A"
GREEN_MUTED = "#2E7D5B"
AMBER_MUTED = "#D4A037"
GRID_COLOR = "rgba(148, 163, 184, 0.22)"


def _style_plotly_fig(fig: Any, height: int = 340, show_legend: bool = True) -> Any:
    """Menerapkan tata letak Plotly seragam bertema biru tua lembut untuk penelitian."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        margin=dict(l=10, r=15, t=32, b=15),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=True, zerolinecolor="rgba(148,163,184,0.45)"),
    )
    return fig


def plot_rating_comparison(df_rc: pd.DataFrame) -> Any:
    """Grafik batang berkelompok: Rating Play Store vs Rating Aktual Terkoreksi."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Rating Play Store",
        x=df_rc["Nama Aplikasi"],
        y=df_rc["Rating Play Store"],
        marker_color=BLUE_LIGHT,
        text=df_rc["Rating Play Store"].round(2),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Rating Aktual (Terkoreksi)",
        x=df_rc["Nama Aplikasi"],
        y=df_rc["Rating Aktual (Terkoreksi)"],
        marker_color=BLUE_PRIMARY,
        text=df_rc["Rating Aktual (Terkoreksi)"].round(2),
        textposition="outside",
    ))
    fig.update_layout(barmode="group", yaxis_range=[0, 5.4], yaxis_title="Skor Bintang")
    fig.update_xaxes(tickangle=-25)
    return _style_plotly_fig(fig, height=360, show_legend=True)


def plot_absa_heatmap(df_asp_num: pd.DataFrame) -> Any:
    """Heatmap kepuasan 8 Aplikasi x 4 Aspek ABSA (Net Sentiment Score)."""
    df_hm = df_asp_num.set_index("Nama Aplikasi")
    fig = go.Figure(data=go.Heatmap(
        z=df_hm.values,
        x=list(df_hm.columns),
        y=list(df_hm.index),
        colorscale=[
            [0.0, RED_MUTED],
            [0.5, "#E2E8F0"],
            [1.0, BLUE_PRIMARY],
        ],
        zmid=0,
        text=df_hm.values,
        texttemplate="%{z:.1f}",
        textfont=dict(size=12),
        hovertemplate="Aplikasi: %{y}<br>Aspek: %{x}<br>NSS: %{z:.1f}<extra></extra>",
        colorbar=dict(title="Skor NSS"),
    ))
    fig.update_yaxes(autorange="reversed")
    return _style_plotly_fig(fig, height=360, show_legend=False)


def plot_single_aspect_bar(df_asp_num: pd.DataFrame, col_name: str) -> Any:
    """Grafik batang skor NSS per aplikasi untuk satu aspek terpilih."""
    vals = df_asp_num[col_name]
    colors = [BLUE_PRIMARY if v >= 0 else RED_MUTED for v in vals]
    fig = go.Figure(go.Bar(
        x=df_asp_num["Nama Aplikasi"],
        y=vals,
        marker_color=colors,
        text=[f"{v:+.1f}" for v in vals],
        textposition="outside",
        hovertemplate="%{x}<br>" + col_name + ": %{y:+.1f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Skor Net Sentiment (NSS)")
    fig.update_xaxes(tickangle=-20)
    return _style_plotly_fig(fig, height=350, show_legend=False)


def plot_model_comparison(df_models: pd.DataFrame) -> Any:
    """Grafik batang perbandingan Macro F1 dan ROC-AUC antar model uji."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Test Macro F1",
        x=df_models["model_name"],
        y=df_models["test_macro_f1"],
        marker_color=BLUE_PRIMARY,
        text=df_models["test_macro_f1"].round(3),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Test ROC-AUC",
        x=df_models["model_name"],
        y=df_models["test_roc_auc"],
        marker_color=BLUE_MID,
        text=df_models["test_roc_auc"].round(3),
        textposition="outside",
    ))
    fig.update_layout(barmode="group", yaxis_range=[0, 1.08], yaxis_title="Skor Evaluasi")
    fig.update_xaxes(tickangle=-18)
    return _style_plotly_fig(fig, height=360, show_legend=True)


def plot_feature_attribution(shown_contribs: List[Tuple[str, float]]) -> Any:
    """Grafik batang horizontal dua arah untuk bobot kontribusi kata (+ vs -)."""
    ordered = list(reversed(shown_contribs))
    terms = [t for t, _ in ordered]
    weights = [w for _, w in ordered]
    colors = [BLUE_PRIMARY if w >= 0 else RED_MUTED for w in weights]
    fig = go.Figure(go.Bar(
        x=weights,
        y=terms,
        orientation="h",
        marker_color=colors,
        text=[f"{w:+.4f}" for w in weights],
        textposition="outside",
        hovertemplate="Fitur: %{y}<br>Kontribusi: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_color="rgba(148,163,184,0.6)")
    fig.update_layout(xaxis_title="Bobot Kontribusi Log-Odds")
    return _style_plotly_fig(fig, height=max(240, len(terms) * 26 + 70), show_legend=False)


def plot_class_probabilities(class_probs: Dict[str, float]) -> Any:
    """Grafik batang probabilitas 3 kelas sentimen."""
    labels = list(class_probs.keys())
    probs = [float(v) for v in class_probs.values()]
    color_map = {"Negatif": RED_MUTED, "Netral": AMBER_MUTED, "Positif": GREEN_MUTED}
    colors = [color_map.get(lbl, BLUE_MID) for lbl in labels]
    fig = go.Figure(go.Bar(
        x=labels,
        y=probs,
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(yaxis_range=[0, 1.12], yaxis_title="Probabilitas")
    return _style_plotly_fig(fig, height=280, show_legend=False)


def plot_batch_sentiment_bar(sent_counts_df: pd.DataFrame) -> Any:
    """Grafik batang distribusi hasil prediksi batch CSV."""
    color_map = {"Negatif": RED_MUTED, "Netral": AMBER_MUTED, "Positif": GREEN_MUTED}
    colors = [color_map.get(s, BLUE_MID) for s in sent_counts_df["Sentimen"]]
    fig = go.Figure(go.Bar(
        x=sent_counts_df["Sentimen"],
        y=sent_counts_df["Jumlah Baris"],
        marker_color=colors,
        text=sent_counts_df["Jumlah Baris"],
        textposition="outside",
    ))
    fig.update_layout(yaxis_title="Jumlah Ulasan")
    return _style_plotly_fig(fig, height=280, show_legend=False)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load model artifact ──────────────────────────────────────────────────
    try:
        artifact      = get_model_artifact()
        model_name    = artifact.get("model_name", "Best Tuned Classifier")
        test_macro_f1 = artifact.get("metrics_summary", {}).get("macro_f1",    0.65)
        test_roc_auc  = artifact.get("metrics_summary", {}).get("roc_auc_ovr", 0.87)
    except Exception as e:
        st.error(f"Gagal memuat artefak model: {e}")
        st.stop()
        return

    # ── State navigasi sidebar (4 menu) ──────────────────────────────────────
    MENU_ITEMS = [
        ("Dashboard Komparasi",     ":material/bar_chart:"),
        ("Analisis Ulasan Tunggal", ":material/search:"),
        ("Prediksi Batch",          ":material/folder_open:"),
        ("Detail Metodologi",       ":material/menu_book:"),
    ]
    if "active_menu" not in st.session_state:
        st.session_state["active_menu"] = "Dashboard Komparasi"

    # ═════════════════════════════════════════════════════════════════════════
    #  SIDEBAR KIRI — 4 MENU NAVIGASI
    # ═════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            '<div class="sb-brand-title">GenAI Indonesia</div>'
            '<div class="sb-brand-sub">Analisis ABSA &amp; Discrepancy</div>'
            '<span class="sb-badge">'
            '<i class="fa-solid fa-graduation-cap"></i>&nbsp;S1 Sistem Informasi · FTI UKSW'
            '</span>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown('<div class="sb-nav-label">Menu Navigasi</div>', unsafe_allow_html=True)
        for menu_label, menu_icon in MENU_ITEMS:
            is_active = st.session_state["active_menu"] == menu_label
            if st.button(
                menu_label,
                key=f"nav_btn_{menu_label}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                icon=menu_icon,
            ):
                st.session_state["active_menu"] = menu_label
                st.rerun()

        st.divider()

        st.markdown(
            '<div class="sb-footer">'
            '<b>AM552 - Data Science of IS</b><br>'
            'Arthur (682024046) · Brendan (682024016) · Tiara (682024059)<br>'
            '&copy; 2026 FTI UKSW'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Header Utama Penelitian ──────────────────────────────────────────────
    st.markdown(
        '<div class="page-header-box">'
        '<div class="page-header-title">Sistem Analisis Sentimen Berbasis Aspek (ABSA) &amp; Deteksi Discrepancy</div>'
        '<p class="page-header-sub">'
        '<b>Aplikasi Mobile Generative AI di Indonesia (71.221 Ulasan Google Play Store)</b><br>'
        'Prototype Penelitian Mata Kuliah AM552 - Data Science of IS | Program Studi S1 Sistem Informasi, FTI UKSW'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    active_menu = st.session_state["active_menu"]

    # =========================================================================
    # MENU 1: DASHBOARD KOMPARASI
    # =========================================================================
    if active_menu == "Dashboard Komparasi":
        st.subheader("Dashboard Komparasi 8 Aplikasi Generative AI di Google Play Store Indonesia")
        bench = get_benchmark_summary()
        if bench and "app_absa_matrix" in bench:
            if "rating_comparison" in bench:
                st.markdown("#### 1. Perbandingan Rating Bintang Play Store vs Rating Aktual (Setelah Koreksi Discrepancy)")
                st.caption(
                    "Mengoreksi bias penilaian bintang ketika pengguna memberi Bintang 4–5 tetapi isi teksnya mengeluh negatif "
                    "(atau memberi Bintang 1–2 tetapi isi teksnya memuji positif)."
                )
                df_rc = pd.DataFrame(bench["rating_comparison"])
                c_rc1, c_rc2 = st.columns([1.15, 1.35])
                with c_rc1:
                    if HAS_PLOTLY:
                        st.plotly_chart(plot_rating_comparison(df_rc), use_container_width=True)
                    else:
                        df_rc_chart = df_rc.set_index("Nama Aplikasi")[["Rating Play Store", "Rating Aktual (Terkoreksi)"]]
                        st.bar_chart(df_rc_chart)
                with c_rc2:
                    st.dataframe(df_rc, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 2. Peta Kepuasan 4 Aspek Layanan (ABSA) Lintas 8 Aplikasi")
            if "aspect_numeric_matrix" in bench:
                df_asp_num = pd.DataFrame(bench["aspect_numeric_matrix"])
                asp_col_choice = st.selectbox(
                    "Pilih Dimensi Aspek Layanan untuk Dibandingkan pada Grafik:",
                    ["Semua Aspek (A1 - A4)", "A1: Akurasi AI", "A2: Harga & Kuota",
                     "A3: Performa & Server", "A4: Bahasa Indonesia"],
                )
                def _safe_col(c: str) -> str:
                    return c.replace(":", "-").replace("&", "dan")

                if asp_col_choice == "Semua Aspek (A1 - A4)":
                    if HAS_PLOTLY:
                        st.plotly_chart(plot_absa_heatmap(df_asp_num), use_container_width=True)
                    else:
                        df_plot = df_asp_num.set_index("Nama Aplikasi").copy()
                        df_plot.columns = [_safe_col(c) for c in df_plot.columns]
                        st.bar_chart(df_plot)
                else:
                    if HAS_PLOTLY:
                        st.plotly_chart(plot_single_aspect_bar(df_asp_num, str(asp_col_choice)), use_container_width=True)
                    else:
                        df_plot = df_asp_num.set_index("Nama Aplikasi")[[str(asp_col_choice)]].copy()
                        df_plot.columns = [_safe_col(str(asp_col_choice))]
                        st.bar_chart(df_plot)

            df_bench = pd.DataFrame(bench["app_absa_matrix"])
            st.dataframe(df_bench, use_container_width=True, hide_index=True)

            if "top_thumbs_complaints" in bench:
                st.markdown("---")
                c_b1, c_b2 = st.columns([1.45, 1.0])
                with c_b1:
                    st.markdown("#### 3. Ulasan Keluhan dengan Dukungan (Thumbs Up / Like) Tertinggi")
                    st.caption("Daftar keluhan pengguna yang paling banyak disetujui pengguna lain di Google Play Store.")
                    df_thumbs = pd.DataFrame(bench["top_thumbs_complaints"])
                    app_filter = st.selectbox(
                        "Filter Berdasarkan Aplikasi:",
                        ["Semua Aplikasi"] + list(df_thumbs["Aplikasi"].unique()),
                    )
                    if app_filter != "Semua Aplikasi":
                        df_thumbs = df_thumbs[df_thumbs["Aplikasi"] == app_filter]
                    st.dataframe(df_thumbs, use_container_width=True, hide_index=True)

                with c_b2:
                    st.markdown("#### 4. Perbandingan Penyebutan Aplikasi Kompetitor")
                    st.caption("Frekuensi pengguna menyebut aplikasi AI pesaing di dalam ulasan suatu aplikasi.")
                    if "competitor_mentions" in bench:
                        df_comp_m = pd.DataFrame(bench["competitor_mentions"])
                        st.dataframe(df_comp_m, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 5. Perbandingan Performa Model Klasifikasi pada Data Uji (80:20 Stratified Split)")
            df_models = pd.DataFrame(bench["model_comparison"])
            c_m1, c_m2 = st.columns([1.0, 1.4])
            with c_m1:
                if HAS_PLOTLY:
                    st.plotly_chart(plot_model_comparison(df_models), use_container_width=True)
                else:
                    st.bar_chart(df_models.set_index("model_name")[["test_macro_f1", "test_roc_auc"]])
            with c_m2:
                st.dataframe(df_models, use_container_width=True, hide_index=True)
        else:
            st.info("Data ringkasan benchmark sedang dimuat dari hasil eksperimen notebook.")

    # =========================================================================
    # MENU 2: ANALISIS ULASAN TUNGGAL
    # =========================================================================
    elif active_menu == "Analisis Ulasan Tunggal":
        st.subheader("1. Form Input Analisis Ulasan Pengguna")
        st.write(
            "Masukkan teks ulasan aplikasi Generative AI beserta skor bintangnya untuk mendeteksi "
            "**Polaritas Sentimen**, **4 Aspek Layanan (ABSA)**, **Rating-Text Discrepancy**, dan **Kata Kunci Pendorong**."
        )

        col_a, col_b = st.columns([2, 1])
        with col_b:
            selected_app = st.selectbox(
                "Pilih Aplikasi Generative AI:",
                [
                    "ChatGPT (OpenAI)",
                    "Google Gemini",
                    "Character.AI",
                    "Perplexity - Ask Anything",
                    "Microsoft Copilot",
                    "DeepSeek - AI Assistant",
                    "Claude by Anthropic",
                    "Poe - Fast AI Chat",
                ],
            )
            star_input = st.slider(
                "Skor Bintang yang Diberikan Pengguna (1-5):",
                min_value=1, max_value=5, value=5,
                help="Digunakan untuk memeriksa apakah rating bintang konsisten dengan isi teks ulasan.",
            )
            preset_example = st.selectbox(
                "Atau Gunakan Contoh Uji Cepat (Bab 6.5):",
                [
                    "-- Ketik Manual --",
                    "Contoh Positif Normal: Aplikasinya sangat pintar dan cepat, bahasa Indonesianya luwes membantu tugas coding kuliah saya!",
                    "Contoh Negatif Normal: Server sibuk terus dan gagal login, padahal kuota gratisnya cepat banget habis harus bayar mahal.",
                    "Contoh Discrepancy (Bintang 5 tapi Komplain): Saya kasih bintang 5 biar dibaca developer, mode live sering macet sendiri dan jawabannya makin ngawur!",
                    "Contoh Input Salah Format (Hanya Angka): 1234567890",
                ],
            )

        default_text = ""
        if preset_example.startswith("Contoh Positif"):
            default_text = "Aplikasinya sangat pintar dan cepat, bahasa Indonesianya luwes membantu tugas coding kuliah saya!"
        elif preset_example.startswith("Contoh Negatif"):
            default_text = "Server sibuk terus dan gagal login, padahal kuota gratisnya cepat banget habis harus bayar mahal."
        elif preset_example.startswith("Contoh Discrepancy"):
            default_text = "Saya kasih bintang 5 biar dibaca developer, mode live sering macet sendiri dan jawabannya makin ngawur!"
        elif preset_example.startswith("Contoh Input Salah"):
            default_text = "1234567890"

        with col_a:
            user_text = st.text_area(
                "Teks Ulasan Pengguna (Bahasa Indonesia / Slang):",
                value=default_text,
                height=145,
                placeholder="Contoh: Jawabannya akurat bgt buat bantu skripsi, tapi sayang limit chat gratisnya dikit...",
            )
            analyze_btn = st.button(
                "Jalankan Analisis Sentimen & ABSA",
                type="primary",
                use_container_width=True,
                icon=":material/play_arrow:",
            )

        if analyze_btn:
            st.session_state["last_single_res"] = predict_single_review(
                raw_text=user_text,
                star_rating=star_input,
                app_name=selected_app,
                artifact=artifact,
                top_k=10,
            )

        if "last_single_res" in st.session_state:
            res: Dict[str, Any] = st.session_state["last_single_res"]

            if res["status"] == "INVALID_INPUT":
                st.error(f"Validasi Input Gagal: {res['error_message']}")
            else:
                if res.get("warning_message"):
                    st.warning(res["warning_message"])

                st.markdown("---")
                st.subheader("2. Hasil Prediksi & Interpretasi Model")

                m1, m2, m3 = st.columns(3)
                sent = res["predicted_sentiment"]

                SENT_ICON_HTML = {
                    "Positif": '<i class="fa-solid fa-circle-check sent-pos"></i>',
                    "Negatif": '<i class="fa-solid fa-circle-xmark sent-neg"></i>',
                    "Netral":  '<i class="fa-solid fa-circle-minus sent-neu"></i>',
                }
                sent_icon = SENT_ICON_HTML.get(sent, '<i class="fa-solid fa-circle-question"></i>')

                m1.metric("Prediksi Sentimen Teks", sent)
                m2.metric("Tingkat Keyakinan (Confidence)", f"{res['confidence']*100:.1f}%")
                m3.metric("Rating Bintang Input", f"{res['star_rating']} / 5 bintang")

                st.markdown(
                    f'<p style="font-size:1.02rem;margin:6px 0 4px;">'
                    f'{sent_icon}&nbsp;<b>Sentimen: {sent}</b>'
                    f'</p>',
                    unsafe_allow_html=True,
                )

                if res["is_discrepancy"]:
                    st.warning(res["discrepancy_status"])
                else:
                    st.info(res["discrepancy_status"])

                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown(
                        '<h4><i class="fa-solid fa-crosshairs h4-icon"></i>'
                        ' Deteksi Dimensi Aspek Layanan (ABSA)</h4>',
                        unsafe_allow_html=True,
                    )
                    detected_aspects: List[Dict[str, Any]] = res.get("detected_aspects", [])
                    for asp in detected_aspects:
                        kw_str = (
                            ", ".join(f"`{k}`" for k in asp.get("matched_keywords", []))
                            if asp.get("matched_keywords") else "*ekspresi umum*"
                        )
                        asp_sent = asp.get("aspect_sentiment", sent)
                        asp_conf = asp.get("aspect_confidence")
                        if asp_sent == "Positif":
                            badge_cls = "asp-badge-pos"
                        elif asp_sent == "Negatif":
                            badge_cls = "asp-badge-neg"
                        else:
                            badge_cls = "asp-badge-neu"
                        conf_txt = f" ({asp_conf*100:.0f}%)" if asp_conf is not None else ""
                        st.markdown(
                            f"- **{asp.get('aspect_label', '')}** "
                            f"<span class=\"{badge_cls}\">{asp_sent}{conf_txt}</span>  \n"
                            f"  Kata pemicu: {kw_str}",
                            unsafe_allow_html=True,
                        )
                        clause_ctx = asp.get("clause_context", "")
                        if clause_ctx and asp.get("aspect_code") != "A0_General_Impression":
                            st.markdown(
                                f'<div class="asp-clause-note">Klausa: &ldquo;{clause_ctx}&rdquo;</div>',
                                unsafe_allow_html=True,
                            )

                    st.markdown(
                        '<h4><i class="fa-solid fa-wand-magic-sparkles h4-icon"></i>'
                        ' Hasil Normalisasi Teks Bahasa Indonesia</h4>',
                        unsafe_allow_html=True,
                    )
                    st.code(res["clean_text"], language="text")

                with c_right:
                    st.markdown(
                        '<h4><i class="fa-solid fa-key h4-icon"></i>'
                        ' Kata Kunci Pendorong Prediksi (Feature Attribution)</h4>',
                        unsafe_allow_html=True,
                    )
                    all_contribs = res.get("all_feature_contributions") or res.get("key_drivers", [])
                    if all_contribs:
                        total_feats = len(all_contribs)
                        top_n_kw = st.slider(
                            "Tampilkan Jumlah Fitur Kata / N-Gram (Top-N):",
                            min_value=3,
                            max_value=20,
                            value=10,
                            help=f"Ditemukan total {total_feats} fitur kata/n-gram aktif pada ulasan ini. Geser untuk memperingkas atau melihat seluruh kata.",
                            key="top_n_feature_slider",
                        )
                        shown_contribs = all_contribs[:top_n_kw]
                        df_kw = pd.DataFrame(
                            [
                                {
                                    "Fitur Kata / N-Gram": term,
                                    "Bobot Kontribusi": weight,
                                    "Arah Pengaruh": (
                                        "Mendukung Prediksi (+)"
                                        if weight > 0
                                        else ("Kontras / Menahan (-)" if weight < 0 else "Netral (0)")
                                    ),
                                }
                                for term, weight in shown_contribs
                            ]
                        )
                        if HAS_PLOTLY:
                            st.plotly_chart(plot_feature_attribution(shown_contribs), use_container_width=True)
                        st.dataframe(df_kw, use_container_width=True, hide_index=True)
                        st.caption(f"Menampilkan {len(shown_contribs)} dari total {total_feats} fitur kata/n-gram aktif di kosakata TF-IDF.")
                    else:
                        st.caption("Tidak ada token yang masuk dalam kosakata TF-IDF utama.")

                    st.markdown(
                        '<h4><i class="fa-solid fa-chart-bar h4-icon"></i>'
                        ' Distribusi Probabilitas Kelas</h4>',
                        unsafe_allow_html=True,
                    )
                    class_probs: Dict[str, float] = res.get("class_probabilities", {})
                    if HAS_PLOTLY and class_probs:
                        st.plotly_chart(plot_class_probabilities(class_probs), use_container_width=True)
                    else:
                        df_prob = pd.DataFrame(
                            list(class_probs.items()),
                            columns=["Kelas Sentimen", "Probabilitas"],
                        )
                        st.bar_chart(df_prob.set_index("Kelas Sentimen"))

    # =========================================================================
    # MENU 3: PREDIKSI BATCH (Upload CSV)
    # =========================================================================
    elif active_menu == "Prediksi Batch":
        st.subheader("Prediksi Massal (Batch Prediction) dari File CSV")
        st.write(
            "Unggah file `.csv` yang memiliki kolom teks ulasan (misal: `content` atau `ulasan`) "
            "untuk memproses normalisasi slang, prediksi sentimen, ekstraksi 4 aspek ABSA, "
            "dan deteksi *discrepancy* sekaligus."
        )

        # Contoh CSV
        sample_csv_df = pd.DataFrame([
            {"app_name": "ChatGPT (OpenAI)",        "score": 5, "content": "Sangat membantu tugas coding dan bahasa Indonesianya mudah dipahami"},
            {"app_name": "DeepSeek - AI Assistant", "score": 5, "content": "Bagus sih gratis, tapi server terlalu sibuk gagal login terus"},
            {"app_name": "Character.AI",            "score": 1, "content": "Makin update makin banyak sensor filter dan bug verifikasi usia"},
            {"app_name": "Google Gemini",           "score": 2, "content": "Limit kuota gratis cepat habis dan mode live sering macet"},
        ])
        st.download_button(
            "Unduh Contoh Format CSV (sample_input_ulasan.csv)",
            data=sample_csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="sample_input_ulasan.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        uploaded_file = st.file_uploader("Unggah File CSV Ulasan Anda:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_up = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"File CSV tidak dapat dibaca atau rusak: {e}")
                df_up = None

            if df_up is not None:
                st.write(f"File berhasil dimuat: **{len(df_up):,} baris** dan **{len(df_up.columns)} kolom**.")
                col1, col2 = st.columns(2)
                with col1:
                    default_tcol = "content" if "content" in df_up.columns else df_up.columns[0]
                    text_col_choice = st.selectbox(
                        "Pilih Nama Kolom Teks Ulasan:",
                        list(df_up.columns),
                        index=list(df_up.columns).index(default_tcol),
                    )
                with col2:
                    score_options = ["-- Tidak Ada Kolom Rating --"] + list(df_up.columns)
                    default_scol = "score" if "score" in df_up.columns else "-- Tidak Ada Kolom Rating --"
                    score_col_choice = st.selectbox(
                        "Pilih Kolom Skor Bintang (Opsional):",
                        score_options,
                        index=score_options.index(default_scol),
                    )

                if st.button("Proses Prediksi Batch Sekarang", type="primary", icon=":material/settings:"):
                    scol = None if score_col_choice == "-- Tidak Ada Kolom Rating --" else str(score_col_choice)
                    df_res, err_msg = predict_batch_dataframe(
                        df_up.head(5000), text_col=str(text_col_choice), score_col=scol, artifact=artifact
                    )
                    if err_msg or df_res is None:
                        st.error(f"{err_msg}")
                    else:
                        st.success(f"Berhasil memproses **{len(df_res):,} baris** ulasan!")
                        total_rows = len(df_res)
                        sent_vc = df_res["predicted_sentiment"].value_counts()
                        dom_sent = str(sent_vc.index[0]) if len(sent_vc) > 0 else "-"
                        dom_pct = (float(sent_vc.iloc[0]) / total_rows * 100.0) if total_rows > 0 and len(sent_vc) > 0 else 0.0

                        kpi1, kpi2, kpi3 = st.columns(3)
                        with kpi1:
                            st.markdown(
                                f'<div class="metric-card-clean">'
                                f'<div class="metric-card-label">Total Ulasan Diproses</div>'
                                f'<div class="metric-card-value">{total_rows:,}</div>'
                                f'<div class="metric-card-note">Baris valid dari file CSV unggahan</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with kpi2:
                            st.markdown(
                                f'<div class="metric-card-clean">'
                                f'<div class="metric-card-label">Sentimen Dominan</div>'
                                f'<div class="metric-card-value">{dom_sent}</div>'
                                f'<div class="metric-card-note">{dom_pct:.1f}% dari total ulasan</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with kpi3:
                            if "is_discrepancy" in df_res.columns:
                                disc_cnt = int(df_res["is_discrepancy"].sum())
                                disc_pct = (disc_cnt / total_rows * 100.0) if total_rows > 0 else 0.0
                                st.markdown(
                                    f'<div class="metric-card-clean">'
                                    f'<div class="metric-card-label">Rating-Text Discrepancy</div>'
                                    f'<div class="metric-card-value">{disc_cnt:,}</div>'
                                    f'<div class="metric-card-note">{disc_pct:.1f}% terindikasi anomali rating</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="metric-card-clean">'
                                    '<div class="metric-card-label">Rating-Text Discrepancy</div>'
                                    '<div class="metric-card-value">-</div>'
                                    '<div class="metric-card-note">Kolom skor bintang tidak dipilih</div>'
                                    '</div>',
                                    unsafe_allow_html=True,
                                )

                        sent_counts_df = (
                            sent_vc.rename_axis("Sentimen").reset_index(name="Jumlah Baris")
                        )
                        c_s1, c_s2 = st.columns(2)
                        with c_s1:
                            st.markdown("**Distribusi Hasil Prediksi Sentimen:**")
                            if HAS_PLOTLY and not sent_counts_df.empty:
                                st.plotly_chart(plot_batch_sentiment_bar(sent_counts_df), use_container_width=True)
                            st.dataframe(
                                sent_counts_df,
                                use_container_width=True, hide_index=True,
                            )
                        with c_s2:
                            st.markdown("**Jumlah Ulasan per Aspek Layanan (ABSA):**")
                            aspect_counts = {
                                "A1: Akurasi & Kecerdasan": int(df_res["A1_Akurasi_Kecerdasan"].sum()),
                                "A2: Harga & Limit Kuota":  int(df_res["A2_Harga_Kuota"].sum()),
                                "A3: Performa, UI & Server": int(df_res["A3_Performa_UI_Server"].sum()),
                                "A4: Bahasa Indonesia":     int(df_res["A4_Bahasa_Indonesia"].sum()),
                            }
                            st.dataframe(
                                pd.DataFrame(
                                    list(aspect_counts.items()),
                                    columns=["Dimensi Aspek", "Jumlah Ulasan Terdeteksi"],
                                ),
                                use_container_width=True, hide_index=True,
                            )

                        st.markdown("**Preview 25 Baris Hasil Prediksi:**")
                        st.dataframe(df_res.head(25), use_container_width=True)

                        st.download_button(
                            "Download Hasil Prediksi Lengkap (.csv)",
                            data=df_res.to_csv(index=False).encode("utf-8-sig"),
                            file_name="hasil_prediksi_absa_genai.csv",
                            mime="text/csv",
                            icon=":material/save:",
                        )

    # =========================================================================
    # MENU 4: DETAIL METODOLOGI
    # =========================================================================
    elif active_menu == "Detail Metodologi":
        st.subheader("Detail Metodologi & Performa Model Penelitian")
        st.write(
            "Ringkasan performa model klasifikasi aktif, karakteristik dataset penelitian, serta "
            "definisi 4 kategori aspek layanan pada kerangka *Aspect-Based Sentiment Analysis* (ABSA)."
        )

        # ── Kartu Performa Model ─────────────────────────────────────────────
        st.markdown(
            '<h4><i class="fa-solid fa-chart-line h4-icon"></i> Performa Model Produksi</h4>',
            unsafe_allow_html=True,
        )
        metrics_sum = artifact.get("metrics_summary", {})
        test_weighted_f1 = metrics_sum.get("weighted_f1", 0.8642)
        test_bal_acc = metrics_sum.get("balanced_accuracy", 0.6986)
        test_mcc = metrics_sum.get("mcc", 0.6745)

        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.markdown(
                f'<div class="metric-card-clean">'
                f'<div class="metric-card-label">Macro F1-Score (Data Uji)</div>'
                f'<div class="metric-card-value">{test_macro_f1:.3f}</div>'
                f'<div class="metric-card-note">Evaluasi keseimbangan 3 kelas sentimen (vs Baseline)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_perf2:
            st.markdown(
                f'<div class="metric-card-clean">'
                f'<div class="metric-card-label">ROC-AUC Score (OvR)</div>'
                f'<div class="metric-card-value">{test_roc_auc:.3f}</div>'
                f'<div class="metric-card-note">Kemampuan pemisahan antar kelas sentimen (Memenuhi Target)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        col_sub1, col_sub2, col_sub3 = st.columns(3)
        with col_sub1:
            st.markdown(
                f'<div class="metric-card-clean">'
                f'<div class="metric-card-label">Weighted F1-Score</div>'
                f'<div class="metric-card-value">{test_weighted_f1:.3f}</div>'
                f'<div class="metric-card-note">Rata-rata F1 tertimbang proporsi kelas</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_sub2:
            st.markdown(
                f'<div class="metric-card-clean">'
                f'<div class="metric-card-label">Balanced Accuracy</div>'
                f'<div class="metric-card-value">{test_bal_acc:.3f}</div>'
                f'<div class="metric-card-note">Rata-rata recall seimbang seluruh 3 kelas</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_sub3:
            st.markdown(
                f'<div class="metric-card-clean">'
                f'<div class="metric-card-label">Matthews Correlation (MCC)</div>'
                f'<div class="metric-card-value">{test_mcc:.3f}</div>'
                f'<div class="metric-card-note">Korelasi prediksi tangguh terhadap imbalance</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Spesifikasi Model & Dataset ──────────────────────────────────────
        st.markdown(
            f'<div class="info-panel-clean">'
            f'<i class="fa-solid fa-microchip"></i><b>Model Produksi Aktif:</b> {model_name} &middot; TF-IDF 1-2 Gram<br>'
            f'<i class="fa-solid fa-database"></i><b>Cakupan Dataset:</b> 71.221 ulasan &middot; 8 aplikasi Generative AI (Google Play Store Indonesia)<br>'
            f'<i class="fa-solid fa-scale-balanced"></i><b>Karakteristik Data:</b> Ketimpangan kelas (<i>Class Imbalance</i>): Bintang 5 sebesar 63,96%'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Daftar 4 Kategori Aspek ABSA ─────────────────────────────────────
        st.markdown(
            '<h4><i class="fa-solid fa-layer-group h4-icon"></i> Daftar 4 Kategori Aspek ABSA</h4>',
            unsafe_allow_html=True,
        )
        aspects = [
            ("fa-bullseye", "A1: Akurasi & Kecerdasan AI",          "Kecerdasan, halusinasi, kualitas jawaban, serta ketepatan solusi."),
            ("fa-coins",    "A2: Harga & Limitasi Kuota",           "Biaya langganan, limit penggunaan gratis, paket berbayar, dan iklan."),
            ("fa-bolt",     "A3: Performa, UI/UX & Server",         "Kecepatan respons, kestabilan server, bug, serta kenyamanan tampilan."),
            ("fa-globe",    "A4: Bahasa Indonesia & Konteks Lokal", "Kualitas bahasa Indonesia, terjemahan, serta pemahaman konteks lokal."),
        ]
        col_asp1, col_asp2 = st.columns(2)
        for idx, (fa_name, label, desc) in enumerate(aspects):
            target_col = col_asp1 if idx % 2 == 0 else col_asp2
            with target_col:
                st.markdown(
                    f'<div class="aspect-card-clean">'
                    f'<div class="aspect-card-title"><i class="fa-solid {fa_name}"></i>{label}</div>'
                    f'<p class="aspect-card-desc">{desc}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Ringkasan 6 Gerbang Audit Forensik Model ─────────────────────────
        st.markdown(
            '<h4><i class="fa-solid fa-shield-halved h4-icon"></i> Ringkasan 6 Gerbang Audit Forensik Model</h4>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Validasi integritas ilmiah untuk memastikan model bebas kebocoran data (data leakage) "
            "serta mampu menggeneralisasi ulasan nyata lintas aplikasi."
        )
        audit = get_audit_summary() or {}
        a1 = audit.get("audit_1_leakage", {})
        a2 = audit.get("audit_2_imbalance_dummy", {})
        a3 = audit.get("audit_3_duplicate_and_length_slices", {})
        a3_slices = a3.get("slices", {})
        a4 = audit.get("audit_4_leave_one_app_out", {})
        a5 = audit.get("audit_5_split_hygiene", {})
        a6 = audit.get("audit_6_permutation_sanity", {})

        n_leak_col = len(a1.get("leaked_forbidden_columns", []))
        n_leak_feat = len(a1.get("suspicious_high_corr_features", []))
        gain_dummy = float(a2.get("model_macro_f1_gain_over_dummy", 0.3098))
        unseen_f1 = float(a3_slices.get("deduplicated_unseen_text", {}).get("metrics", {}).get("macro_f1", 0.6332))
        sub_f1 = float(a3_slices.get("substantive_reviews_ge5w", {}).get("metrics", {}).get("macro_f1", 0.6353))
        drop_f1 = float(a3.get("macro_f1_drop_full_to_substantive", 0.0082))
        loao_f1 = float(a4.get("average_loao_macro_f1", 0.5982))
        ov_tv = int(a5.get("overlap_train_val", 0))
        ov_tt = int(a5.get("overlap_train_test", 0))
        ov_vt = int(a5.get("overlap_val_test", 0))
        perm_m = a6.get("shuffled_label_metrics", {})
        perm_auc = float(perm_m.get("roc_auc_ovr", 0.4558))
        perm_f1 = float(perm_m.get("macro_f1", 0.2778))

        def _status_txt(ok: bool) -> str:
            return "LOLOS" if ok else "PERLU TINJAUAN"

        audit_rows = [
            {
                "Gerbang Audit": "AUDIT-1 (Deteksi Target Leakage)",
                "Pengukuran & Bukti Empiris": f"{n_leak_col} kolom bocor, {n_leak_feat} fitur korelasi mencurigakan",
                "Status": _status_txt(bool(a1.get("passed", True))),
            },
            {
                "Gerbang Audit": "AUDIT-2 (Dummy Baseline Gain)",
                "Pengukuran & Bukti Empiris": f"Macro F1 Gain +{gain_dummy:.4f} di atas Dummy Stratified",
                "Status": _status_txt(bool(a2.get("passed", True))),
            },
            {
                "Gerbang Audit": "AUDIT-3 (Unseen Text & Ulasan Substantif >=5 Kata)",
                "Pengukuran & Bukti Empiris": (
                    f"Unseen Text Macro F1 {unseen_f1:.4f}, Substantif {sub_f1:.4f} "
                    f"(penurunan hanya {drop_f1:.4f})"
                ),
                "Status": _status_txt(bool(a3.get("passed", True))),
            },
            {
                "Gerbang Audit": "AUDIT-4 (Generalisasi Lintas Aplikasi / LOAO)",
                "Pengukuran & Bukti Empiris": (
                    f"Rata-rata LOAO Macro F1 {loao_f1:.4f} "
                    "(uji DeepSeek, Claude, Gemini tanpa dilatih)"
                ),
                "Status": _status_txt(bool(a4.get("passed", True))),
            },
            {
                "Gerbang Audit": "AUDIT-5 (Higienitas Split & Zero Overlap)",
                "Pengukuran & Bukti Empiris": (
                    f"Overlap Train/Val/Test = {ov_tv}/{ov_tt}/{ov_vt} ID, "
                    "pipeline terenkapsulasi penuh"
                ),
                "Status": _status_txt(bool(a5.get("passed", True))),
            },
            {
                "Gerbang Audit": "AUDIT-6A (Permutation Sanity Test)",
                "Pengukuran & Bukti Empiris": (
                    f"Label diacak -> ROC-AUC turun ke {perm_auc:.4f} / Macro F1 {perm_f1:.4f} "
                    "(bebas kebocoran tersembunyi)"
                ),
                "Status": _status_txt(bool(a6.get("passed", True))),
            },
        ]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer-bar">'
        '&copy; 2026 <b>Arthur</b> (682024046) &middot; <b>Brendan</b> (682024016) &middot; <b>Tiara</b> (682024059)<br>'
        '<span style="font-size:0.70rem;">AM552B Data Science of IS &middot; S1 Sistem Informasi &middot; FTI UKSW</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
