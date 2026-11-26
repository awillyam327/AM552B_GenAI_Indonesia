"""
Aplikasi Web Prototype — Analisis Sentimen Berbasis Aspek (ABSA) & Deteksi Discrepancy
Mata Kuliah: AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)
Cara Menjalankan: streamlit run prototype/app.py

UI: Seluruh emoji diganti dengan Font Awesome 6 (HTML) dan Material Icons (Streamlit native).
Seluruh logika prediksi, Tab 1-3, dan fungsi Arthur dipertahankan utuh.
"""

import importlib
import json
import os
import sys
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

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
    page_icon=":material/robot:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS + FONT AWESOME 6 FREE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
      integrity="sha512-Avb2QiuDEEvB4bZJYdft2mNjVShBftLdPG8FJ0V7irTLQ8Uo0qcPxh4Plq7G5tGm0rU+1SPhVotteLpBERwTkA=="
      crossorigin="anonymous" referrerpolicy="no-referrer">

<style>
/* ── Global font ── */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

/* ── Sidebar: dark navy gradient ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2137 0%, #1a3a5c 100%) !important;
    border-right: 1px solid #1e4976;
}
[data-testid="stSidebar"] * { color: #e8f4fd !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem; font-weight: 700; line-height: 1.4; margin-bottom: 2px;
}

/* ── Metric cards (sidebar) ── */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px; padding: 12px 16px; margin-bottom: 8px;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: 0.75rem; opacity: 0.85; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 800; }
[data-testid="stSidebar"] [data-testid="stMetricDelta"] { font-size: 0.72rem; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 2px solid #e2e8f0; }
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600; font-size: 0.88rem; padding: 10px 20px; color: #64748b;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1F4E79 !important;
    border-bottom: 2px solid #1F4E79 !important;
    background: transparent !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #1F4E79 !important; border: none !important;
    color: #fff !important; border-radius: 8px !important; font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover { background: #163d60 !important; }

/* ── Badge institusi ── */
.badge-inst {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 8px; font-size: 0.72rem; font-weight: 600;
    padding: 4px 10px; margin: 4px 0 12px 0; color: #e8f4fd !important;
}

/* ── Aspek item di sidebar ── */
.asp-item {
    display: flex; align-items: flex-start; gap: 8px; margin: 6px 0;
    padding: 6px 8px; background: rgba(255,255,255,0.06); border-radius: 8px;
}
.asp-icon { font-size: 0.95rem; min-width: 20px; color: #90c8e8 !important; padding-top: 2px; }
.asp-label { font-size: 0.8rem; font-weight: 700; line-height: 1.3; }
.asp-desc { font-size: 0.69rem; opacity: 0.72; }

/* ── Sidebar section header ── */
.sidebar-section-hdr {
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.03em;
    color: #90c8e8 !important; margin: 2px 0 8px 0;
}
.sidebar-section-hdr i { margin-right: 5px; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { border: 2px dashed #cbd5e1; border-radius: 12px; padding: 8px; }

/* ── Sentiment icon colors ── */
.sent-pos { color: #16a34a !important; }
.sent-neg { color: #dc2626 !important; }
.sent-neu { color: #ca8a04 !important; }

/* ── Section h4 with icon ── */
.h4-icon { color: #1F4E79; margin-right: 6px; }

/* ── Footer ── */
.footer-bar {
    text-align: center; color: #94a3b8; font-size: 0.75rem;
    padding: 20px 0 8px 0; border-top: 1px solid #e2e8f0; margin-top: 32px;
}

/* ── Dataset stats list ── */
.ds-list { font-size: 0.8rem; line-height: 2.2; }
.ds-list i { margin-right: 5px; color: #90c8e8; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS  (tidak diubah dari versi Arthur)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_model_artifact():
    return load_trained_artifact()


@st.cache_data
def get_benchmark_summary():
    if os.path.exists(BENCHMARK_JSON_PATH):
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.title("🤖 Sistem Analisis Sentimen Berbasis Aspek (ABSA) & Deteksi Discrepancy")
    st.markdown(
        "**Aplikasi Mobile Generative AI di Indonesia (71.221 Ulasan Google Play Store)**  \n"
        "*Prototype Tugas Akhir Mata Kuliah AM552 - Data Science of IS | Program Studi S1 Sistem Informasi, FTI UKSW*"
    )

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

    # ═════════════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ═════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## GenAI Indonesia\nABSA & Discrepancy Analyzer")
        st.markdown(
            '<span class="badge-inst">'
            '<i class="fa-solid fa-school"></i>&nbsp;S1 Sistem Informasi · FTI UKSW'
            '</span>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Metric cards — nilai real dari artifact
        st.markdown(
            '<div class="sidebar-section-hdr">'
            '<i class="fa-solid fa-chart-line"></i> Performa Model Produksi'
            '</div>',
            unsafe_allow_html=True,
        )
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Macro F1", f"{test_macro_f1:.4f}", delta="vs Baseline")
        with col_m2:
            st.metric("ROC-AUC", f"{test_roc_auc:.4f}", delta="Target")
        st.caption(f"Model: **{model_name}** · TF-IDF 1-2gram")

        st.divider()

        # 4 Dimensi Aspek ABSA
        st.markdown(
            '<div class="sidebar-section-hdr">'
            '<i class="fa-solid fa-layer-group"></i> 4 Dimensi Aspek ABSA'
            '</div>',
            unsafe_allow_html=True,
        )
        aspects = [
            ("fa-bullseye",   "A1: Akurasi AI",   "Kecerdasan, halusinasi, kualitas jawaban"),
            ("fa-coins",      "A2: Harga/Quota",  "Biaya, limit, subscription, iklan"),
            ("fa-bolt",       "A3: Performa/UI",  "Kecepatan, server, bug, tampilan"),
            ("fa-globe",      "A4: Bahasa Lokal", "Bahasa Indonesia, terjemahan, konteks lokal"),
        ]
        for fa_name, label, desc in aspects:
            st.markdown(
                f'<div class="asp-item">'
                f'<span class="asp-icon"><i class="fa-solid {fa_name}"></i></span>'
                f'<div><div class="asp-label">{label}</div>'
                f'<div class="asp-desc">{desc}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Dataset stats
        st.markdown(
            '<div class="sidebar-section-hdr">'
            '<i class="fa-solid fa-database"></i> Dataset'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ds-list">'
            '<i class="fa-solid fa-hashtag"></i><b>71.221</b> ulasan · 8 aplikasi<br>'
            '<i class="fa-solid fa-calendar-days"></i>Google Play Store Indonesia<br>'
            '<i class="fa-solid fa-scale-balanced"></i>Class imbalance: Bintang 5 (63.96%)'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(
            "© 2026 Arthur (682024046) · Brendan (682024016) · Tiara (682024059)\n"
            "S1 Sistem Informasi · FTI UKSW"
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  TABS — Material Icons
    # ═════════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs([
        ":material/bar_chart:  Dashboard Komparasi",
        ":material/search:  Analisis Ulasan Tunggal",
        ":material/folder_open:  Prediksi Batch (Upload CSV)",
    ])

    # =========================================================================
    # TAB 1: DASHBOARD KOMPARASI
    # =========================================================================
    with tab1:
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
                # Altair (backend st.bar_chart) tidak mendukung ':' dan '&' dalam nama kolom.
                # Rename kolom sementara untuk plotting agar tidak memicu ValueError.
                def _safe_col(c: str) -> str:
                    return c.replace(":", "-").replace("&", "dan")

                if asp_col_choice == "Semua Aspek (A1 - A4)":
                    df_plot = df_asp_num.set_index("Nama Aplikasi").copy()
                    df_plot.columns = [_safe_col(c) for c in df_plot.columns]
                    st.bar_chart(df_plot)
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
                st.bar_chart(df_models.set_index("model_name")[["test_macro_f1", "test_roc_auc"]])
            with c_m2:
                st.dataframe(df_models, use_container_width=True, hide_index=True)
        else:
            st.info("Data ringkasan benchmark sedang dimuat dari hasil eksperimen notebook.")

    # =========================================================================
    # TAB 2: ANALISIS ULASAN TUNGGAL
    # =========================================================================
    with tab2:
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
                icon=":material/rocket_launch:",
            )

        if analyze_btn:
            res: Dict[str, Any] = predict_single_review(
                raw_text=user_text,
                star_rating=star_input,
                app_name=selected_app,
                artifact=artifact,
            )

            if res["status"] == "INVALID_INPUT":
                st.error(f"Validasi Input Gagal: {res['error_message']}")
            else:
                if res.get("warning_message"):
                    st.warning(res["warning_message"])

                st.markdown("---")
                st.subheader("2. Hasil Prediksi & Interpretasi Model")

                m1, m2, m3 = st.columns(3)
                sent = res["predicted_sentiment"]

                # FA icon per sentimen (HTML)
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
                    f'<p style="font-size:1.05rem;margin:6px 0 2px;">'
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
                        st.markdown(f"- **{asp.get('aspect_label', '')}** — Kata pemicu: {kw_str}")

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
                    if res["key_drivers"]:
                        df_kw = pd.DataFrame(res["key_drivers"], columns=["Fitur Kata / N-Gram", "Bobot Kontribusi"])
                        st.dataframe(df_kw, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Tidak ada token yang masuk dalam kosakata TF-IDF utama.")

                    st.markdown(
                        '<h4><i class="fa-solid fa-chart-bar h4-icon"></i>'
                        ' Distribusi Probabilitas Kelas</h4>',
                        unsafe_allow_html=True,
                    )
                    class_probs: Dict[str, float] = res.get("class_probabilities", {})
                    df_prob = pd.DataFrame(
                        list(class_probs.items()),
                        columns=["Kelas Sentimen", "Probabilitas"],
                    )
                    st.bar_chart(df_prob.set_index("Kelas Sentimen"))

    # =========================================================================
    # TAB 3: PREDIKSI BATCH (Upload CSV)
    # =========================================================================
    with tab3:
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
                        c_s1, c_s2 = st.columns(2)
                        with c_s1:
                            st.markdown("**Distribusi Hasil Prediksi Sentimen:**")
                            st.dataframe(
                                df_res["predicted_sentiment"]
                                    .value_counts()
                                    .rename_axis("Sentimen")
                                    .reset_index(name="Jumlah Baris"),
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

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="footer-bar">'
        '&copy; 2026 <b>Arthur</b> (682024046) &middot; <b>Brendan</b> (682024016) &middot; <b>Tiara</b> (682024059)<br>'
        '<span style="font-size:0.68rem;">AM552B Data Science of IS &middot; S1 Sistem Informasi &middot; FTI UKSW</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
