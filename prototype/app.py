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
