"""
Aplikasi Web Prototype — Analisis Sentimen Berbasis Aspek (ABSA) & Deteksi Discrepancy
Mata Kuliah: AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)
Cara Menjalankan: streamlit run prototype/app.py

Desain UI: Sidebar navigasi kiri (4 menu) dengan tema biru tua lembut yang bersih dan profesional
untuk kebutuhan presentasi penelitian. Seluruh logika prediksi dan keluaran data dipertahankan utuh.
"""

from datetime import date, timedelta
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
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
DAILY_METRICS_PATH = os.path.join(ROOT_DIR, "models", "dashboard_daily_metrics.csv")
DATED_COMPLAINTS_PATH = os.path.join(ROOT_DIR, "models", "dashboard_dated_complaints.csv")
COPILOT_ANALYSIS_PATH = os.path.join(ROOT_DIR, "models", "copilot_excellence_analysis.json")
TOPIC_MODEL_JSON_PATH = os.path.join(ROOT_DIR, "04_modeling", "topic_modeling_results.json")


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


@st.cache_data
def get_dashboard_daily_metrics() -> Optional[pd.DataFrame]:
    if os.path.exists(DAILY_METRICS_PATH):
        df_d = pd.read_csv(DAILY_METRICS_PATH)
        df_d["date_dt"] = pd.to_datetime(df_d["date"], errors="coerce").dt.date
        return df_d
    return None


@st.cache_data
def get_dated_complaints() -> Optional[pd.DataFrame]:
    if os.path.exists(DATED_COMPLAINTS_PATH):
        df_c = pd.read_csv(DATED_COMPLAINTS_PATH)
        return df_c
    return None


@st.cache_data
def get_copilot_analysis() -> Optional[Dict[str, Any]]:
    if os.path.exists(COPILOT_ANALYSIS_PATH):
        with open(COPILOT_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def get_topic_modeling_results() -> Optional[Dict[str, Any]]:
    if os.path.exists(TOPIC_MODEL_JSON_PATH):
        with open(TOPIC_MODEL_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


ASPECT_META_DASHBOARD = [
    ("a1", "A1: Akurasi AI", "A1: Akurasi AI (Mention / NSS)", "A1: Akurasi & Kecerdasan AI"),
    ("a2", "A2: Harga & Kuota", "A2: Harga/Kuota (Mention / NSS)", "A2: Harga & Limitasi Kuota"),
    ("a3", "A3: Performa & Server", "A3: Performa/UI (Mention / NSS)", "A3: Performa, UI/UX & Server"),
    ("a4", "A4: Bahasa Indonesia", "A4: Bhs Indonesia (Mention / NSS)", "A4: Bahasa Indonesia & Konteks Lokal"),
]


def render_date_range_selector(
    min_d: date,
    max_d: date,
    key_prefix: str,
    title: str = "Filter Periode Rentang Tanggal:",
) -> Tuple[date, date, bool]:
    """Menampilkan kontrol preset & pemilih rentang tanggal bersih."""
    c_p1, c_p2, c_p3 = st.columns([1.2, 1.0, 1.0])
    preset_opts = [
        "Semua Periode",
        "30 Hari Terakhir",
        "90 Hari Terakhir",
        "6 Bulan Terakhir",
        "1 Tahun Terakhir",
        "Rentang Manual",
    ]
    with c_p1:
        preset = st.selectbox(title, preset_opts, key=f"{key_prefix}_preset")

    if preset == "30 Hari Terakhir":
        p_start, p_end = max(min_d, max_d - timedelta(days=30)), max_d
    elif preset == "90 Hari Terakhir":
        p_start, p_end = max(min_d, max_d - timedelta(days=90)), max_d
    elif preset == "6 Bulan Terakhir":
        p_start, p_end = max(min_d, max_d - timedelta(days=180)), max_d
    elif preset == "1 Tahun Terakhir":
        p_start, p_end = max(min_d, max_d - timedelta(days=365)), max_d
    else:
        p_start, p_end = min_d, max_d

    is_manual = preset == "Rentang Manual"
    with c_p2:
        start_d = st.date_input(
            "Dari Tanggal:",
            value=p_start,
            min_value=min_d,
            max_value=max_d,
            disabled=not is_manual,
            key=f"{key_prefix}_start",
        )
    with c_p3:
        end_d = st.date_input(
            "Sampai Tanggal:",
            value=p_end,
            min_value=min_d,
            max_value=max_d,
            disabled=not is_manual,
            key=f"{key_prefix}_end",
        )

    final_start = p_start if not is_manual else start_d
    final_end = p_end if not is_manual else end_d
    if final_start > final_end:
        final_start, final_end = final_end, final_start

    is_subperiod = (final_start > min_d) or (final_end < max_d)
    return final_start, final_end, is_subperiod


def _priority_status(nss: float) -> str:
    if nss < 0:
        return "Prioritas Tinggi (Perlu Perbaikan)"
    if nss <= 30.0:
        return "Perlu Dipantau"
    return "Baik"


def compute_dashboard_period_tables(
    df_daily_sub: pd.DataFrame,
    priority_app: str = "Semua Aplikasi",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mengagregasi metrik rating, ABSA, dan prioritas perbaikan dari slice harian."""
    agg_cols = [
        "n_reviews", "sum_score", "sum_adj_score", "disc_high_neg", "disc_low_pos",
        "a1_n", "a1_pos", "a1_neg",
        "a2_n", "a2_pos", "a2_neg",
        "a3_n", "a3_pos", "a3_neg",
        "a4_n", "a4_pos", "a4_neg",
    ]
    grouped = (
        df_daily_sub.groupby("app_name", as_index=False)[agg_cols]
        .sum()
        .sort_values("n_reviews", ascending=False)
    )

    rc_rows = []
    asp_num_rows = []
    bench_rows = []

    for _, row in grouped.iterrows():
        n_rev = int(row["n_reviews"])
        if n_rev == 0:
            continue
        app = str(row["app_name"])
        avg_star = round(float(row["sum_score"]) / n_rev, 2)
        adj_star = round(float(row["sum_adj_score"]) / n_rev, 2)
        hi_neg = int(row["disc_high_neg"])
        lo_pos = int(row["disc_low_pos"])
        disc_pct = round(((hi_neg + lo_pos) / n_rev) * 100.0, 2)

        rc_rows.append({
            "Nama Aplikasi": app,
            "Jumlah Ulasan": n_rev,
            "Rating Play Store": avg_star,
            "Rating Aktual (Terkoreksi)": adj_star,
            "Selisih Rating": round(adj_star - avg_star, 2),
            "Total Discrepancy (%)": disc_pct,
            "Bintang 4-5 Teks Negatif": hi_neg,
            "Bintang 1-2 Teks Positif": lo_pos,
        })

        num_r: Dict[str, Any] = {"Nama Aplikasi": app}
        b_r: Dict[str, Any] = {
            "Nama Aplikasi": app,
            "Jumlah Ulasan": n_rev,
            "Rata-rata Bintang": avg_star,
        }
        for pref, num_col, str_col, _ in ASPECT_META_DASHBOARD:
            cnt = int(row[f"{pref}_n"])
            pos_c = int(row[f"{pref}_pos"])
            neg_c = int(row[f"{pref}_neg"])
            nss = round(((pos_c - neg_c) / cnt) * 100.0, 1) if cnt > 0 else 0.0
            num_r[num_col] = nss
            b_r[str_col] = f"{cnt:,} ({nss:+.1f}%)" if cnt > 0 else "0 (0.0%)"

        asp_num_rows.append(num_r)
        bench_rows.append(b_r)

    # Tabel Prioritas Perbaikan ABSA untuk developer (diurutkan dari NSS terendah)
    df_prio_src = (
        df_daily_sub[df_daily_sub["app_name"] == priority_app]
        if priority_app != "Semua Aplikasi"
        else df_daily_sub
    )
    prio_sums = df_prio_src[agg_cols].sum() if len(df_prio_src) > 0 else pd.Series(0, index=agg_cols)
    prio_rows = []
    for pref, _, _, full_label in ASPECT_META_DASHBOARD:
        cnt = int(prio_sums[f"{pref}_n"])
        pos_c = int(prio_sums[f"{pref}_pos"])
        neg_c = int(prio_sums[f"{pref}_neg"])
        nss = round(((pos_c - neg_c) / cnt) * 100.0, 1) if cnt > 0 else 0.0
        prio_rows.append({
            "Kategori Aspek ABSA": full_label,
            "Total Mention": cnt,
            "Positif": pos_c,
            "Negatif": neg_c,
            "Skor NSS (%)": nss,
            "Prioritas Perbaikan": _priority_status(nss) if cnt > 0 else "Belum Ada Data",
        })
    df_priority = pd.DataFrame(prio_rows).sort_values("Skor NSS (%)", ascending=True)

    return (
        pd.DataFrame(rc_rows),
        pd.DataFrame(asp_num_rows),
        pd.DataFrame(bench_rows),
        df_priority,
    )


def compute_batch_absa_priority(df_res: pd.DataFrame) -> pd.DataFrame:
    """Menghitung skor NSS dan prioritas perbaikan per aspek ABSA dari hasil prediksi batch CSV."""
    aspect_cols = [
        ("A1_Akurasi_Kecerdasan", "A1: Akurasi & Kecerdasan AI"),
        ("A2_Harga_Kuota", "A2: Harga & Limitasi Kuota"),
        ("A3_Performa_UI_Server", "A3: Performa, UI/UX & Server"),
        ("A4_Bahasa_Indonesia", "A4: Bahasa Indonesia & Konteks Lokal"),
    ]
    rows = []
    for col, label in aspect_cols:
        sub = df_res[df_res[col] == 1]
        cnt = len(sub)
        pos_c = int((sub["predicted_sentiment"] == "Positif").sum()) if cnt > 0 else 0
        neg_c = int((sub["predicted_sentiment"] == "Negatif").sum()) if cnt > 0 else 0
        nss = round(((pos_c - neg_c) / cnt) * 100.0, 1) if cnt > 0 else 0.0
        rows.append({
            "Kategori Aspek ABSA": label,
            "Total Mention": cnt,
            "Positif": pos_c,
            "Negatif": neg_c,
            "Skor NSS (%)": nss,
            "Prioritas Perbaikan": _priority_status(nss) if cnt > 0 else "Tidak Terdeteksi",
        })
    return pd.DataFrame(rows).sort_values("Skor NSS (%)", ascending=True)


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


def plot_copilot_composite_index(df_comp_rank: pd.DataFrame) -> Any:
    """Grafik batang horizontal Skor Indeks Komposit 8 Aplikasi (menyorot Microsoft Copilot)."""
    df_s = df_comp_rank.sort_values("Skor Indeks Komposit", ascending=True)
    colors = [BLUE_PRIMARY if a == "Microsoft Copilot" else BLUE_LIGHT for a in df_s["Nama Aplikasi"]]
    fig = go.Figure(go.Bar(
        x=df_s["Skor Indeks Komposit"],
        y=df_s["Nama Aplikasi"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in df_s["Skor Indeks Komposit"]],
        textposition="outside",
        hovertemplate="%{y}<br>Skor Indeks Komposit: %{x:.2f}/100<extra></extra>",
    ))
    fig.update_layout(xaxis_range=[0, 98], xaxis_title="Skor Indeks Komposit (0–100)")
    return _style_plotly_fig(fig, height=340, show_legend=False)


def plot_copilot_vs_peers_absa(df_comp_rank: pd.DataFrame) -> Any:
    """Grafik perbandingan NSS 4 aspek ABSA: Microsoft Copilot vs 4 kompetitor teratas."""
    peers = [
        "Microsoft Copilot",
        "Perplexity - Ask Anything",
        "Poe - Fast AI Chat",
        "ChatGPT (OpenAI)",
        "Google Gemini",
    ]
    asp_cols = ["A1: Akurasi AI", "A2: Harga & Kuota", "A3: Performa & Server", "A4: Bahasa Indonesia"]
    palette = [BLUE_PRIMARY, BLUE_MID, BLUE_LIGHT, "#94A3B8", "#CBD5E1"]
    sub = df_comp_rank.set_index("Nama Aplikasi")
    fig = go.Figure()
    for idx, app_n in enumerate(peers):
        if app_n not in sub.index:
            continue
        vals = [float(sub.loc[app_n, c]) for c in asp_cols]
        fig.add_trace(go.Bar(
            name=app_n,
            x=["A1: Akurasi", "A2: Harga/Kuota", "A3: Performa/UI", "A4: Bhs Lokal"],
            y=vals,
            marker_color=palette[idx],
        ))
    fig.update_layout(barmode="group", yaxis_title="Net Sentiment Score / NSS (%)")
    return _style_plotly_fig(fig, height=340, show_legend=True)


def plot_lda_k_coherence(k_neg: List[Dict[str, Any]], k_pos: List[Dict[str, Any]]) -> Any:
    """Grafik kurva evaluasi koherensi topik C_UMass lintas kandidat K."""
    fig = go.Figure()
    if k_neg:
        fig.add_trace(go.Scatter(
            name="Korpus Keluhan (Bintang 1–2)",
            x=[f"K={r['k_topics']}" for r in k_neg],
            y=[float(r["umass_coherence"]) for r in k_neg],
            mode="lines+markers",
            line=dict(color=RED_MUTED, width=2.5),
            marker=dict(size=8),
        ))
    if k_pos:
        fig.add_trace(go.Scatter(
            name="Korpus Pujian (Bintang 4–5)",
            x=[f"K={r['k_topics']}" for r in k_pos],
            y=[float(r["umass_coherence"]) for r in k_pos],
            mode="lines+markers",
            line=dict(color=BLUE_PRIMARY, width=2.5),
            marker=dict(size=8),
        ))
    fig.update_layout(yaxis_title="Skor Koherensi Topik (C_UMass)", xaxis_title="Jumlah Klaster Topik (K)")
    return _style_plotly_fig(fig, height=300, show_legend=True)


def plot_topic_prevalence(topics: List[Dict[str, Any]], is_complaint: bool = True) -> Any:
    """Grafik batang horizontal proporsi dokumen (%) per klaster topik LDA."""
    sorted_t = sorted(topics, key=lambda x: float(x.get("doc_share_pct", 0.0)), reverse=False)
    labels = [f"T{t['topic_id']}: {t.get('business_label', '')[:34]}" for t in sorted_t]
    shares = [float(t.get("doc_share_pct", 0.0)) for t in sorted_t]
    bar_col = RED_MUTED if is_complaint else BLUE_PRIMARY
    fig = go.Figure(go.Bar(
        x=shares,
        y=labels,
        orientation="h",
        marker_color=bar_col,
        text=[f"{s:.1f}%" for s in shares],
        textposition="outside",
        hovertemplate="%{y}<br>Proporsi Dokumen: %{x:.2f}%<extra></extra>",
    ))
    max_s = max(shares) if shares else 50.0
    fig.update_layout(xaxis_range=[0, max_s * 1.22], xaxis_title="Proporsi Ulasan (%)")
    return _style_plotly_fig(fig, height=300, show_legend=False)


def plot_topic_keywords(topic: Dict[str, Any], is_complaint: bool = True) -> Any:
    """Grafik batang horizontal Top 10 kata kunci dominan beserta bobot LDA."""
    words = list(reversed(topic.get("top_words", [])[:10]))
    weights = list(reversed([float(w) for w in topic.get("weights", [])[:10]]))
    bar_col = RED_MUTED if is_complaint else BLUE_MID
    fig = go.Figure(go.Bar(
        x=weights,
        y=words,
        orientation="h",
        marker_color=bar_col,
        text=[f"{w:.1f}" for w in weights],
        textposition="outside",
    ))
    max_w = max(weights) if weights else 100.0
    fig.update_layout(xaxis_range=[0, max_w * 1.22], xaxis_title="Bobot Kata Kunci pada Topik (LDA)")
    return _style_plotly_fig(fig, height=320, show_legend=False)


def plot_app_topic_heatmap(df_mat: pd.DataFrame, is_complaint: bool = True) -> Any:
    """Heatmap distribusi proporsi 5 topik LDA (%) lintas 8 aplikasi."""
    topic_cols = [c for c in df_mat.columns if c not in ("Nama Aplikasi", "Jumlah Ulasan")]
    df_hm = df_mat.set_index("Nama Aplikasi")[topic_cols]
    cscale = (
        [[0.0, "#F8FAFC"], [0.5, "#E29595"], [1.0, "#991B1B"]]
        if is_complaint
        else [[0.0, "#F8FAFC"], [0.5, BLUE_LIGHT], [1.0, BLUE_PRIMARY]]
    )
    fig = go.Figure(data=go.Heatmap(
        z=df_hm.values,
        x=topic_cols,
        y=list(df_hm.index),
        colorscale=cscale,
        text=df_hm.values,
        texttemplate="%{z:.1f}%",
        textfont=dict(size=11),
        hovertemplate="Aplikasi: %{y}<br>%{x}<br>Proporsi: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="% Ulasan"),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-15)
    return _style_plotly_fig(fig, height=360, show_legend=False)


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

    # ── State navigasi sidebar (5 menu) ──────────────────────────────────────
    MENU_ITEMS = [
        ("Dashboard Komparasi",     ":material/bar_chart:"),
        ("Analisis Ulasan Tunggal", ":material/search:"),
        ("Prediksi Batch",          ":material/folder_open:"),
        ("Pemodelan Topik (LDA)",   ":material/hub:"),
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
        df_daily = get_dashboard_daily_metrics()
        df_dated_comp = get_dated_complaints()

        if bench and "app_absa_matrix" in bench:
            is_subperiod = False
            start_d: Optional[date] = None
            end_d: Optional[date] = None
            df_daily_sub: Optional[pd.DataFrame] = None

            if df_daily is not None and not df_daily.empty:
                min_d = df_daily["date_dt"].min()
                max_d = df_daily["date_dt"].max()
                start_d, end_d, is_subperiod = render_date_range_selector(
                    min_d,
                    max_d,
                    key_prefix="dash",
                    title="Filter Periode Rentang Tanggal Ulasan:",
                )
                df_daily_sub = df_daily[
                    (df_daily["date_dt"] >= start_d) & (df_daily["date_dt"] <= end_d)
                ]
                period_total = int(df_daily_sub["n_reviews"].sum()) if not df_daily_sub.empty else 0
                st.caption(
                    f"Periode aktif: **{start_d.strftime('%d %b %Y')}** s.d. "
                    f"**{end_d.strftime('%d %b %Y')}** · Total ulasan pada periode ini: **{period_total:,}**"
                )
                if period_total == 0:
                    st.warning("Tidak ada ulasan pada rentang tanggal yang dipilih. Silakan perluas periode tanggal.")

            if is_subperiod and df_daily_sub is not None and not df_daily_sub.empty:
                df_rc, df_asp_num, df_bench, _ = compute_dashboard_period_tables(df_daily_sub)
            else:
                df_rc = pd.DataFrame(bench["rating_comparison"]) if "rating_comparison" in bench else pd.DataFrame()
                df_asp_num = pd.DataFrame(bench["aspect_numeric_matrix"]) if "aspect_numeric_matrix" in bench else pd.DataFrame()
                df_bench = pd.DataFrame(bench["app_absa_matrix"])

            if not df_rc.empty:
                st.markdown("#### 1. Perbandingan Rating Bintang Play Store vs Rating Aktual (Setelah Koreksi Discrepancy)")
                st.caption(
                    "Mengoreksi bias penilaian bintang ketika pengguna memberi Bintang 4–5 tetapi isi teksnya mengeluh negatif "
                    "(atau memberi Bintang 1–2 tetapi isi teksnya memuji positif)."
                )
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
            if not df_asp_num.empty:
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

            if not df_bench.empty:
                st.dataframe(df_bench, use_container_width=True, hide_index=True)

            # ── Prioritas Perbaikan ABSA untuk Developer pada Periode Terpilih ─
            if df_daily_sub is not None and not df_daily_sub.empty:
                st.markdown("#### Prioritas Perbaikan Aspek ABSA bagi Pengembang (Periode Terpilih)")
                st.caption(
                    "Diurutkan dari skor Net Sentiment Score (NSS) terendah untuk membantu pengembang "
                    "menentukan aspek layanan yang paling mendesak ditingkatkan."
                )
                app_opts_prio = ["Semua Aplikasi"] + sorted(list(df_daily_sub["app_name"].unique()))
                prio_app_choice = st.selectbox(
                    "Fokus Analisis Prioritas Aplikasi:",
                    app_opts_prio,
                    key="dash_prio_app_filter",
                )
                _, _, _, df_priority = compute_dashboard_period_tables(
                    df_daily_sub, priority_app=str(prio_app_choice)
                )
                st.dataframe(df_priority, use_container_width=True, hide_index=True)

            if "top_thumbs_complaints" in bench or (df_dated_comp is not None and not df_dated_comp.empty):
                # ── Tabel 3 (Full Width — Atas) ───────────────────────────────
                st.markdown("---")
                st.markdown("#### 3. Ulasan Keluhan dengan Dukungan (Thumbs Up / Like) Tertinggi")
                st.caption("Daftar keluhan pengguna yang paling banyak disetujui pengguna lain di Google Play Store.")
                if df_dated_comp is not None and not df_dated_comp.empty and start_d is not None and end_d is not None:
                    df_thumbs = df_dated_comp.copy()
                    t_dates = pd.to_datetime(df_thumbs["Tanggal"], errors="coerce").dt.date
                    df_thumbs = df_thumbs[(t_dates >= start_d) & (t_dates <= end_d)]
                else:
                    df_thumbs = pd.DataFrame(bench.get("top_thumbs_complaints", []))

                if not df_thumbs.empty:
                    c_tf1, _ = st.columns([1.2, 2.0])
                    with c_tf1:
                        app_filter = st.selectbox(
                            "Filter Berdasarkan Aplikasi:",
                            ["Semua Aplikasi"] + list(df_thumbs["Aplikasi"].unique()),
                            key="dash_thumbs_app_filter",
                        )
                    if app_filter != "Semua Aplikasi":
                        df_thumbs = df_thumbs[df_thumbs["Aplikasi"] == app_filter]
                    df_thumbs_view = df_thumbs.head(20)
                    thumbs_col_cfg: Dict[str, Any] = {
                        "Aplikasi": st.column_config.TextColumn("Aplikasi", width="medium"),
                        "Bintang": st.column_config.NumberColumn("Bintang", format="%d", width="small"),
                        "Jumlah Like (Thumbs Up)": st.column_config.NumberColumn(
                            "Jumlah Like (Thumbs Up)", format="%d", width="small"
                        ),
                        "Aspek Terkait": st.column_config.TextColumn("Aspek Terkait", width="medium"),
                        "Cuplikan Keluhan Pengguna": st.column_config.TextColumn(
                            "Cuplikan Keluhan Pengguna", width="large"
                        ),
                    }
                    if "Tanggal" in df_thumbs_view.columns:
                        thumbs_col_cfg["Tanggal"] = st.column_config.TextColumn("Tanggal", width="small")
                    st.dataframe(
                        df_thumbs_view,
                        use_container_width=True,
                        hide_index=True,
                        height=min(480, (len(df_thumbs_view) + 1) * 38 + 12),
                        column_config=thumbs_col_cfg,
                    )
                else:
                    st.info("Tidak ada ulasan keluhan teratas pada rentang tanggal ini.")

                # ── Tabel 4 (Full Width — Bawah, Berurutan) ──────────────────
                if "competitor_mentions" in bench:
                    st.markdown("---")
                    st.markdown("#### 4. Perbandingan Penyebutan Aplikasi Kompetitor")
                    st.caption("Frekuensi pengguna menyebut aplikasi AI pesaing di dalam ulasan suatu aplikasi.")
                    df_comp_m = pd.DataFrame(bench["competitor_mentions"])
                    st.dataframe(
                        df_comp_m,
                        use_container_width=True,
                        hide_index=True,
                        height=min(480, (len(df_comp_m) + 1) * 36 + 12),
                        column_config={
                            "Ulasan pada Aplikasi": st.column_config.TextColumn("Ulasan pada Aplikasi", width="medium"),
                            "Menyebut Kompetitor": st.column_config.TextColumn("Menyebut Kompetitor", width="medium"),
                            "Jumlah Ulasan": st.column_config.NumberColumn("Jumlah Ulasan", format="%d", width="small"),
                        },
                    )

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

            # ── 6. Analisis Bukti Empiris Keunggulan Microsoft Copilot ───────
            cop_data = get_copilot_analysis()
            if cop_data and "composite_ranking_table" in cop_data:
                st.markdown("---")
                st.markdown("#### 6. Analisis Bukti Empiris: Mengapa Microsoft Copilot Paling Unggul Secara Menyeluruh?")
                st.caption(
                    "Evaluasi multidimensi berbasis 71.221 ulasan Google Play Store Indonesia yang menggabungkan "
                    "Rating Aktual Terkoreksi, keseimbangan 4 aspek ABSA, batas bawah kepercayaan Wilson 95%, serta tingkat keluhan."
                )
                kh = cop_data.get("key_highlights", {})
                k_c1, k_c2, k_c3, k_c4 = st.columns(4)
                with k_c1:
                    st.markdown(
                        f'<div class="metric-card-clean">'
                        f'<div class="metric-card-label">Peringkat Indeks Komposit</div>'
                        f'<div class="metric-card-value">#{kh.get("rank_composite", 1)} ({kh.get("composite_score", 85.53):.2f})</div>'
                        f'<div class="metric-card-note">Tertinggi dari seluruh 8 aplikasi AI</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with k_c2:
                    st.markdown(
                        f'<div class="metric-card-clean">'
                        f'<div class="metric-card-label">Rating Aktual Terkoreksi</div>'
                        f'<div class="metric-card-value">{kh.get("adjusted_rating", 4.62):.2f} / 5.00</div>'
                        f'<div class="metric-card-note">#1 setelah koreksi discrepancy ({kh.get("discrepancy_pct", 2.95):.2f}%)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with k_c3:
                    st.markdown(
                        f'<div class="metric-card-clean">'
                        f'<div class="metric-card-label">Rata-rata NSS 4 Aspek ABSA</div>'
                        f'<div class="metric-card-value">{kh.get("mean_absa_nss", 46.58):+.2f}%</div>'
                        f'<div class="metric-card-note">Satu-satunya aplikasi {kh.get("positive_aspects_count", "4/4")} aspek positif</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with k_c4:
                    st.markdown(
                        f'<div class="metric-card-clean">'
                        f'<div class="metric-card-label">Konsistensi Kepuasan (N=5.459)</div>'
                        f'<div class="metric-card-value">{kh.get("positive_share_pct", 91.9):.1f}%</div>'
                        f'<div class="metric-card-note">Wilson 95% LB {kh.get("wilson_95_lb_pct", 91.19):.1f}% · Bintang 1 hanya {kh.get("star1_complaint_pct", 4.51):.1f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                df_comp_rank = pd.DataFrame(cop_data["composite_ranking_table"])
                if HAS_PLOTLY and not df_comp_rank.empty:
                    ch_cop1, ch_cop2 = st.columns(2)
                    with ch_cop1:
                        st.markdown("**Peringkat Skor Indeks Komposit (0–100):**")
                        st.plotly_chart(plot_copilot_composite_index(df_comp_rank), use_container_width=True)
                    with ch_cop2:
                        st.markdown("**Profil NSS 4 Aspek ABSA (Copilot vs Kompetitor Utama):**")
                        st.plotly_chart(plot_copilot_vs_peers_absa(df_comp_rank), use_container_width=True)

                st.markdown("**Tabel Peringkat Keunggulan Multidimensi Lintas 8 Aplikasi:**")
                st.dataframe(df_comp_rank, use_container_width=True, hide_index=True)

                c_ev1, c_ev2 = st.columns([1.1, 1.2])
                with c_ev1:
                    st.markdown("**Temuan Head-to-Head terhadap Kompetitor Terdekat:**")
                    for h2h in cop_data.get("head_to_head_summary", []):
                        st.markdown(f"- **{h2h['Pembanding']}:** {h2h['Bukti Statistik Kredibel']}")
                    if cop_data.get("yearly_stability"):
                        st.markdown("**Konsistensi Rating Tahunan Microsoft Copilot (2023–2026):**")
                        st.dataframe(pd.DataFrame(cop_data["yearly_stability"]), use_container_width=True, hide_index=True)

                with c_ev2:
                    if cop_data.get("satisfaction_drivers"):
                        st.markdown("**Faktor Pendorong Kepuasan pada Ulasan Positif Copilot:**")
                        st.dataframe(pd.DataFrame(cop_data["satisfaction_drivers"]), use_container_width=True, hide_index=True)
                    if cop_data.get("verified_quotes"):
                        st.markdown("**Sampel Bukti Ulasan Nyata Pengguna Indonesia:**")
                        st.dataframe(pd.DataFrame(cop_data["verified_quotes"]).head(5), use_container_width=True, hide_index=True)
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
                    typo_corr = res.get("typo_corrections", [])
                    if typo_corr:
                        corr_str = ", ".join(f"`{c['asal']}` -> `{c['koreksi']}`" for c in typo_corr)
                        st.caption(f"Koreksi typo otomatis: {corr_str}")

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

        # Contoh CSV (dilengkapi kolom tanggal `at` untuk simulasi filter rentang tanggal)
        sample_csv_df = pd.DataFrame([
            {"app_name": "ChatGPT (OpenAI)",        "score": 5, "at": "2026-09-18", "content": "Sangat membantu tugas coding dan bahasa Indonesianya mudah dipahami"},
            {"app_name": "DeepSeek - AI Assistant", "score": 5, "at": "2026-10-06", "content": "Bagus sih gratis, tapi server terlalu sibuk gagal login terus"},
            {"app_name": "Character.AI",            "score": 1, "at": "2026-10-24", "content": "Makin update makin banyak sensor filter dan bug verifikasi usia"},
            {"app_name": "Google Gemini",           "score": 2, "at": "2026-11-12", "content": "Limit kuota gratis cepat habis dan mode live sering macet"},
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
                col1, col2, col3 = st.columns(3)
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
                with col3:
                    date_options = ["-- Tidak Ada Kolom Tanggal --"] + list(df_up.columns)
                    default_dcol = next(
                        (c for c in ("at", "date", "tanggal", "created_at") if c in df_up.columns),
                        "-- Tidak Ada Kolom Tanggal --",
                    )
                    date_col_choice = st.selectbox(
                        "Pilih Kolom Tanggal (Opsional):",
                        date_options,
                        index=date_options.index(default_dcol),
                    )

                df_filtered_batch = df_up
                batch_period_note = "Seluruh baris file CSV"
                if date_col_choice != "-- Tidak Ada Kolom Tanggal --":
                    parsed_dates = pd.to_datetime(df_up[str(date_col_choice)], errors="coerce").dt.date
                    valid_dates = parsed_dates.dropna()
                    if not valid_dates.empty:
                        b_min_d = valid_dates.min()
                        b_max_d = valid_dates.max()
                        b_start, b_end, _ = render_date_range_selector(
                            b_min_d,
                            b_max_d,
                            key_prefix="batch",
                            title="Filter Periode Rentang Tanggal Batch:",
                        )
                        date_mask = parsed_dates.notna() & (parsed_dates >= b_start) & (parsed_dates <= b_end)
                        df_filtered_batch = df_up[date_mask]
                        batch_period_note = f"{b_start.strftime('%d %b %Y')} s.d. {b_end.strftime('%d %b %Y')}"
                        st.caption(
                            f"Baris cocok pada periode **{batch_period_note}**: "
                            f"**{len(df_filtered_batch):,}** dari **{len(df_up):,} baris**."
                        )
                    else:
                        st.warning(f"Kolom '{date_col_choice}' tidak berisi format tanggal yang dapat dibaca. Semua baris akan diproses.")

                if st.button("Proses Prediksi Batch Sekarang", type="primary", icon=":material/settings:"):
                    if len(df_filtered_batch) == 0:
                        st.warning("Tidak ada baris ulasan pada rentang tanggal yang dipilih.")
                    else:
                        scol = None if score_col_choice == "-- Tidak Ada Kolom Rating --" else str(score_col_choice)
                        df_res, err_msg = predict_batch_dataframe(
                            df_filtered_batch.head(5000), text_col=str(text_col_choice), score_col=scol, artifact=artifact
                        )
                        if err_msg or df_res is None:
                            st.error(f"{err_msg}")
                        else:
                            st.success(f"Berhasil memproses **{len(df_res):,} baris** ulasan ({batch_period_note})!")
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
                                    f'<div class="metric-card-note">Periode: {batch_period_note}</div>'
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

                            st.markdown("**Prioritas Perbaikan ABSA (Berdasarkan Periode Terpilih):**")
                            st.caption("Membantu pengembang melihat kategori aspek dengan skor NSS terendah dari ulasan pada periode ini.")
                            st.dataframe(compute_batch_absa_priority(df_res), use_container_width=True, hide_index=True)

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
    # MENU 4: PEMODELAN TOPIK (LDA)
    # =========================================================================
    elif active_menu == "Pemodelan Topik (LDA)":
        st.subheader("Pemodelan Topik Pengalaman Pengguna (Unsupervised Latent Dirichlet Allocation)")
        st.write(
            "Ekstraksi otomatis tema-tema laten keluhan dan pujian pengguna lintas 8 aplikasi Generative AI "
            "menggunakan *Latent Dirichlet Allocation* (LDA) serta evaluasi koherensi topik $C_{\\text{UMass}}$."
        )

        lda_data = get_topic_modeling_results()
        if not lda_data:
            st.info("Artefak hasil pemodelan topik sedang dimuat.")
        else:
            n_tot_lda = int(lda_data.get("dataset_size", 71221))
            n_neg_lda = int(lda_data.get("negative_reviews_count", 0))
            n_pos_lda = int(lda_data.get("positive_reviews_count", 0))
            opt_k = int(lda_data.get("optimal_k", 5))
            slang_cnt = int(lda_data.get("slang_dictionary_size", 8618))

            m_l1, m_l2, m_l3, m_l4 = st.columns(4)
            with m_l1:
                st.markdown(
                    f'<div class="metric-card-clean">'
                    f'<div class="metric-card-label">Jumlah Topik Optimal</div>'
                    f'<div class="metric-card-value">K = {opt_k}</div>'
                    f'<div class="metric-card-note">Dipilih melalui Perplexity &amp; C_UMass</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m_l2:
                st.markdown(
                    f'<div class="metric-card-clean">'
                    f'<div class="metric-card-label">Korpus Keluhan Substantif</div>'
                    f'<div class="metric-card-value">{n_neg_lda:,}</div>'
                    f'<div class="metric-card-note">Ulasan Bintang 1–2 bermuatan topik</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m_l3:
                st.markdown(
                    f'<div class="metric-card-clean">'
                    f'<div class="metric-card-label">Korpus Pujian Substantif</div>'
                    f'<div class="metric-card-value">{n_pos_lda:,}</div>'
                    f'<div class="metric-card-note">Ulasan Bintang 4–5 bermuatan topik</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m_l4:
                st.markdown(
                    f'<div class="metric-card-clean">'
                    f'<div class="metric-card-label">Kamus Normalisasi Slang</div>'
                    f'<div class="metric-card-value">{slang_cnt:,}</div>'
                    f'<div class="metric-card-note">Entri baku dari {n_tot_lda:,} ulasan</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 1. Evaluasi Pemilihan Jumlah Topik Optimal K
            k_neg = lda_data.get("k_selection_complaint", [])
            k_pos = lda_data.get("k_selection_praise", [])
            if k_neg and k_pos:
                st.markdown("#### 1. Evaluasi Statistik Pemilihan Jumlah Topik Optimal ($K \\in \\{3, 4, 5, 6, 8\\}$)")
                st.caption(
                    "Jumlah topik $K=5$ memberikan titik keseimbangan terbaik antara nilai koherensi semantik "
                    "$C_{\\text{UMass}}$ dan keterpisahan interpretasi bisnis tanpa tumpang tindih klaster."
                )
                ck1, ck2 = st.columns([1.15, 1.25])
                with ck1:
                    if HAS_PLOTLY:
                        st.plotly_chart(plot_lda_k_coherence(k_neg, k_pos), use_container_width=True)
                with ck2:
                    k_rows = []
                    for rn, rp in zip(k_neg, k_pos):
                        k_rows.append({
                            "Kandidat K": f"K = {rn['k_topics']}",
                            "Perplexity (Keluhan)": rn["perplexity"],
                            "C_UMass (Keluhan)": rn["umass_coherence"],
                            "Perplexity (Pujian)": rp["perplexity"],
                            "C_UMass (Pujian)": rp["umass_coherence"],
                        })
                    st.dataframe(pd.DataFrame(k_rows), use_container_width=True, hide_index=True)

            st.markdown("---")

            # 2. Eksplorasi 5 Klaster Topik Global
            st.markdown("#### 2. Eksplorasi 5 Klaster Topik Global (Keluhan vs Pujian)")
            polarity_mode = st.radio(
                "Pilih Kategori Polaritas Korpus LDA:",
                ["Topik Keluhan Utama (Ulasan Negatif Bintang 1–2)", "Topik Pujian Utama (Ulasan Positif Bintang 4–5)"],
                horizontal=True,
                key="lda_polarity_toggle",
            )
            is_comp_mode = polarity_mode.startswith("Topik Keluhan")
            active_topics = (
                lda_data.get("global_complaint_topics", [])
                if is_comp_mode
                else lda_data.get("global_praise_topics", [])
            )

            if active_topics:
                ct_left, ct_right = st.columns([1.05, 1.25])
                with ct_left:
                    st.markdown("**Proporsi Dokumen per Klaster Topik (%):**")
                    if HAS_PLOTLY:
                        st.plotly_chart(plot_topic_prevalence(active_topics, is_complaint=is_comp_mode), use_container_width=True)

                    topic_opts = [
                        f"Topik {t['topic_id']}: {t.get('business_label', '')} ({t.get('doc_share_pct', 0.0):.1f}%)"
                        for t in active_topics
                    ]
                    sel_topic_lbl = st.selectbox("Pilih Klaster Topik untuk Bedah Kata Kunci:", topic_opts, key="lda_topic_sel")
                    sel_idx = topic_opts.index(sel_topic_lbl)
                    sel_t = active_topics[sel_idx]

                with ct_right:
                    st.markdown(f"**Top 10 Kata Kunci Dominan — Topik {sel_t['topic_id']}:**")
                    if HAS_PLOTLY:
                        st.plotly_chart(plot_topic_keywords(sel_t, is_complaint=is_comp_mode), use_container_width=True)

                st.markdown(
                    f'<div class="info-panel-clean">'
                    f'<i class="fa-solid fa-tag"></i><b>Label Bisnis Topik {sel_t["topic_id"]}:</b> {sel_t.get("business_label", "-")}<br>'
                    f'<i class="fa-solid fa-layer-group"></i><b>Keterkaitan Dimensi ABSA:</b> {sel_t.get("mapped_absa", "-")} '
                    f'&middot; <b>Cakupan:</b> {sel_t.get("doc_count", 0):,} ulasan ({sel_t.get("doc_share_pct", 0.0):.1f}%)<br>'
                    f'<i class="fa-solid fa-circle-info"></i><b>Interpretasi Temuan:</b> {sel_t.get("interpretation", "-")}<br>'
                    f'<i class="fa-solid fa-quote-left"></i><b>Contoh Ulasan Representatif ({sel_t.get("sample_app", "-")}):</b> '
                    f'<i>&ldquo;{sel_t.get("sample_quote", "-")}&rdquo;</i>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                summary_t_rows = []
                for t in active_topics:
                    summary_t_rows.append({
                        "ID Topik": f"Topik {t['topic_id']}",
                        "Label Bisnis Topik": t.get("business_label", "-"),
                        "Pemetaan ABSA": t.get("mapped_absa", "-"),
                        "Jumlah Ulasan": t.get("doc_count", 0),
                        "Proporsi (%)": t.get("doc_share_pct", 0.0),
                        "Kata Kunci Utama (LDA)": ", ".join(t.get("top_words", [])[:8]),
                    })
                st.dataframe(pd.DataFrame(summary_t_rows), use_container_width=True, hide_index=True)

            st.markdown("---")

            # 3. Distribusi Topik Lintas 8 Aplikasi & Topic Discovery Spesifik Aplikasi
            st.markdown("#### 3. Peta Distribusi Topik Lintas 8 Aplikasi & Bedah Spesifik Aplikasi")
            mat_key = "app_complaint_topic_matrix" if is_comp_mode else "app_praise_topic_matrix"
            mat_rows = lda_data.get(mat_key, [])
            if mat_rows:
                df_mat = pd.DataFrame(mat_rows)
                if HAS_PLOTLY:
                    st.plotly_chart(plot_app_topic_heatmap(df_mat, is_complaint=is_comp_mode), use_container_width=True)
                st.dataframe(df_mat, use_container_width=True, hide_index=True)

            app_disc = lda_data.get("app_specific_discovery", {})
            if app_disc:
                st.markdown("#### 4. Penemuan Topik Spesifik per Aplikasi (App-Specific Topic Discovery)")
                sel_app_lda = st.selectbox(
                    "Pilih Aplikasi Generative AI untuk Melihat Klaster Topik Spesifiknya:",
                    list(app_disc.keys()),
                    key="lda_app_specific_sel",
                )
                app_info = app_disc[sel_app_lda]
                st.caption(
                    f"Rata-rata Bintang: **{app_info.get('rating_avg', 0):.2f}** · "
                    f"Ulasan Pujian Substantif: **{app_info.get('n_praise_docs', 0):,}** · "
                    f"Ulasan Keluhan Substantif: **{app_info.get('n_complaint_docs', 0):,}**"
                )
                ca_pos, ca_neg = st.columns(2)
                with ca_pos:
                    st.markdown(f"**Klaster Topik Pujian — {sel_app_lda}:**")
                    p_list = app_info.get("praise_topics", [])
                    if p_list:
                        df_p_app = pd.DataFrame([
                            {
                                "Klaster": f"P{pt['topic_id']}",
                                "Tema Pujian": pt.get("business_label", "-"),
                                "Proporsi (%)": pt.get("doc_share_pct", "-"),
                                "Kata Kunci": ", ".join(pt.get("top_words", [])[:6]),
                            }
                            for pt in p_list
                        ])
                        st.dataframe(df_p_app, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Data pujian substantif tidak mencukupi untuk pemodelan klaster terpisah.")

                with ca_neg:
                    st.markdown(f"**Klaster Topik Keluhan — {sel_app_lda}:**")
                    c_list = app_info.get("complaint_topics", [])
                    if c_list:
                        df_c_app = pd.DataFrame([
                            {
                                "Klaster": f"K{ct['topic_id']}",
                                "Tema Keluhan": ct.get("business_label", "-"),
                                "Proporsi (%)": ct.get("doc_share_pct", "-"),
                                "Kata Kunci": ", ".join(ct.get("top_words", [])[:6]),
                            }
                            for ct in c_list
                        ])
                        st.dataframe(df_c_app, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Data keluhan substantif sangat minim pada aplikasi ini.")

    # =========================================================================
    # MENU 5: DETAIL METODOLOGI
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
