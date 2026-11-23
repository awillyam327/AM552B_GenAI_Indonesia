"""
Aplikasi Web Prototype — Analisis Sentimen Berbasis Aspek (ABSA) & Deteksi Discrepancy
Mata Kuliah: AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)
Cara Menjalankan: streamlit run prototype/app.py
"""

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

from core_predictor import (
    BENCHMARK_JSON_PATH,
    load_trained_artifact,
    predict_batch_dataframe,
    predict_single_review,
)

st.set_page_config(
    page_title="GenAI Indonesia ABSA & Discrepancy Analyzer — AM552 FTI UKSW",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def get_model_artifact():
    return load_trained_artifact()


@st.cache_data
def get_benchmark_summary():
    if os.path.exists(BENCHMARK_JSON_PATH):
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    st.title("🤖 Sistem Analisis Sentimen Berbasis Aspek (ABSA) & Deteksi Discrepancy")
    st.markdown(
        "**Ekosistem Aplikasi Mobile Generative AI di Indonesia (71.221 Ulasan Google Play Store)**  \n"
        "*Prototype Tugas Akhir Mata Kuliah AM552 - Data Science of IS | Program Studi S1 Sistem Informasi, FTI UKSW*"
    )

    try:
        artifact = get_model_artifact()
        model_name = artifact.get("model_name", "Best Tuned Classifier")
        test_macro_f1 = artifact.get("metrics_summary", {}).get("macro_f1", 0.65)
        test_roc_auc = artifact.get("metrics_summary", {}).get("roc_auc_ovr", 0.87)
    except Exception as e:
        st.error(f"Gagal memuat artefak model: {e}")
        st.stop()
        return

    with st.sidebar:
        st.header("📌 Informasi Model & Proyek")
        st.success(f"**Model Aktif**: {model_name}")
        st.metric("Test Macro F1-Score", f"{test_macro_f1:.4f}")
        st.metric("Test ROC-AUC (OvR)", f"{test_roc_auc:.4f}")
        st.markdown("---")
        st.markdown(
            "**4 Dimensi Aspek ABSA:**\n"
            "1. **A1**: Akurasi & Kecerdasan AI\n"
            "2. **A2**: Harga & Limitasi Kuota\n"
            "3. **A3**: Performa, UI/UX & Server\n"
            "4. **A4**: Bahasa Indonesia & Lokal\n"
        )
        st.markdown("---")
        st.caption("© 2026 Arthur (682024046) • Brendan (682024016) • Tiara (682024059) — S1 Sistem Informasi FTI UKSW")

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Tab 1: Analisis Ulasan Tunggal",
            "📂 Tab 2: Prediksi Batch (Upload CSV)",
            "📊 Tab 3: Dashboard Komparasi 8 Aplikasi AI (N=71.221)",
        ]
    )

    # =========================================================================
    # TAB 1: ANALISIS ULASAN TUNGGAL
    # =========================================================================
    with tab1:
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
                "Skor Bintang yang Diberikan Pengguna (1–5):",
                min_value=1,
                max_value=5,
                value=5,
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
            analyze_btn = st.button("🚀 Jalankan Analisis Sentimen & ABSA", type="primary", use_container_width=True)

        if analyze_btn:
            res: Dict[str, Any] = predict_single_review(
                raw_text=user_text,
                star_rating=star_input,
                app_name=selected_app,
                artifact=artifact,
            )

            if res["status"] == "INVALID_INPUT":
                st.error(f"❌ **Validasi Input Gagal**: {res['error_message']}")
            else:
                if res.get("warning_message"):
                    st.warning(res["warning_message"])

                st.markdown("---")
                st.subheader("2. Hasil Prediksi & Interpretasi Model")

                m1, m2, m3 = st.columns(3)
                sent = res["predicted_sentiment"]
                badge_color = "🟢" if sent == "Positif" else ("🔴" if sent == "Negatif" else "🟡")
                m1.metric("Prediksi Sentimen Teks", f"{badge_color} {sent}")
                m2.metric("Tingkat Keyakinan (Confidence)", f"{res['confidence']*100:.1f}%")
                m3.metric("Rating Bintang Input", f"⭐ {res['star_rating']} / 5")

                if res["is_discrepancy"]:
                    st.warning(res["discrepancy_status"])
                else:
                    st.info(res["discrepancy_status"])

                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown("#### 🎯 Deteksi Dimensi Aspek Layanan (ABSA)")
                    detected_aspects: List[Dict[str, Any]] = res.get("detected_aspects", [])
                    for asp in detected_aspects:
                        kw_str = ", ".join(f"`{k}`" for k in asp.get("matched_keywords", [])) if asp.get("matched_keywords") else "*ekspresi umum*"
                        st.markdown(f"- **{asp.get('aspect_label', '')}** — Kata pemicu: {kw_str}")

                    st.markdown("#### 🧹 Hasil Normalisasi Teks Bahasa Indonesia")
                    st.code(res["clean_text"], language="text")

                with c_right:
                    st.markdown("#### 🔑 Kata Kunci Pendorong Prediksi (Feature Attribution)")
                    if res["key_drivers"]:
                        df_kw = pd.DataFrame(res["key_drivers"], columns=["Fitur Kata / N-Gram", "Bobot Kontribusi"])
                        st.dataframe(df_kw, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Tidak ada token yang masuk dalam kosakata TF-IDF utama.")

                    st.markdown("#### 📈 Distribusi Probabilitas Kelas")
                    class_probs: Dict[str, float] = res.get("class_probabilities", {})
                    df_prob = pd.DataFrame(
                        list(class_probs.items()),
                        columns=["Kelas Sentimen", "Probabilitas"],
                    )
                    st.bar_chart(df_prob.set_index("Kelas Sentimen"))

    # =========================================================================
    # TAB 2: PREDIKSI BATCH (UPLOAD CSV)
    # =========================================================================
    with tab2:
        st.subheader("Prediksi Massal (Batch Prediction) dari File CSV")
        st.write(
            "Unggah file `.csv` yang memiliki kolom teks ulasan (misal: `content` atau `ulasan`) "
            "untuk memproses normalisasi slang, prediksi sentimen, ekstraksi 4 aspek ABSA, dan deteksi *discrepancy* sekaligus."
        )

        # Sediakan tombol download contoh CSV
        sample_csv_df = pd.DataFrame(
            [
                {"app_name": "ChatGPT (OpenAI)", "score": 5, "content": "Sangat membantu tugas coding dan bahasa Indonesianya mudah dipahami"},
                {"app_name": "DeepSeek - AI Assistant", "score": 5, "content": "Bagus sih gratis, tapi server terlalu sibuk gagal login terus"},
                {"app_name": "Character.AI", "score": 1, "content": "Makin update makin banyak sensor filter dan bug verifikasi usia"},
                {"app_name": "Google Gemini", "score": 2, "content": "Limit kuota gratis cepat habis dan mode live sering macet"},
            ]
        )
        st.download_button(
            "📥 Unduh Contoh Format CSV (sample_input_ulasan.csv)",
            data=sample_csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="sample_input_ulasan.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("Unggah File CSV Ulasan Anda:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_up = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"❌ File CSV tidak dapat dibaca atau rusak: {e}")
                df_up = None

            if df_up is not None:
                st.write(f"File berhasil dimuat: **{len(df_up):,} baris** dan **{len(df_up.columns)} kolom**.")
                col1, col2 = st.columns(2)
                with col1:
                    default_tcol = "content" if "content" in df_up.columns else df_up.columns[0]
                    text_col_choice = st.selectbox("Pilih Nama Kolom Teks Ulasan:", list(df_up.columns), index=list(df_up.columns).index(default_tcol))
                with col2:
                    score_options = ["-- Tidak Ada Kolom Rating --"] + list(df_up.columns)
                    default_scol = "score" if "score" in df_up.columns else "-- Tidak Ada Kolom Rating --"
                    score_col_choice = st.selectbox("Pilih Kolom Skor Bintang (Opsional):", score_options, index=score_options.index(default_scol))

                if st.button("⚡ Proses Prediksi Batch Sekarang", type="primary"):
                    scol = None if score_col_choice == "-- Tidak Ada Kolom Rating --" else str(score_col_choice)
                    df_res, err_msg = predict_batch_dataframe(
                        df_up.head(5000), text_col=str(text_col_choice), score_col=scol, artifact=artifact
                    )
                    if err_msg or df_res is None:
                        st.error(f"❌ {err_msg}")
                    else:
                        st.success(f"✅ Berhasil memproses **{len(df_res):,} baris** ulasan!")
                        c_s1, c_s2 = st.columns(2)
                        with c_s1:
                            st.markdown("**Distribusi Hasil Prediksi Sentimen:**")
                            st.dataframe(
                                df_res["predicted_sentiment"].value_counts().rename_axis("Sentimen").reset_index(name="Jumlah Baris"),
                                use_container_width=True,
                                hide_index=True,
                            )
                        with c_s2:
                            st.markdown("**Jumlah Ulasan per Aspek Layanan (ABSA):**")
                            aspect_counts = {
                                "A1: Akurasi & Kecerdasan": int(df_res["A1_Akurasi_Kecerdasan"].sum()),
                                "A2: Harga & Limit Kuota": int(df_res["A2_Harga_Kuota"].sum()),
                                "A3: Performa, UI & Server": int(df_res["A3_Performa_UI_Server"].sum()),
                                "A4: Bahasa Indonesia": int(df_res["A4_Bahasa_Indonesia"].sum()),
                            }
                            st.dataframe(
                                pd.DataFrame(list(aspect_counts.items()), columns=["Dimensi Aspek", "Jumlah Ulasan Terdeteksi"]),
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.markdown("**Preview 25 Baris Hasil Prediksi:**")
                        st.dataframe(df_res.head(25), use_container_width=True)

                        st.download_button(
                            "💾 Download Hasil Prediksi Lengkap (.csv)",
                            data=df_res.to_csv(index=False).encode("utf-8-sig"),
                            file_name="hasil_prediksi_absa_genai.csv",
                            mime="text/csv",
                        )

    # =========================================================================
    # TAB 3: DASHBOARD KOMPARASI EKOSISTEM AI INDONESIA (N = 71.221)
    # =========================================================================
    with tab3:
        st.subheader("Dashboard Komparasi 8 Aplikasi Generative AI di Google Play Store Indonesia")
        bench = get_benchmark_summary()
        if bench and "app_absa_matrix" in bench:
            # 1. Perbandingan Rating Play Store vs Rating Aktual (Koreksi Discrepancy)
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
            # 2. Visualisasi Skor Kepuasan Bersih (Net Sentiment Score) 4 Aspek ABSA
            st.markdown("#### 2. Peta Kepuasan 4 Aspek Layanan (ABSA) Lintas 8 Aplikasi")
            if "aspect_numeric_matrix" in bench:
                df_asp_num = pd.DataFrame(bench["aspect_numeric_matrix"])
                asp_col_choice = st.selectbox(
                    "Pilih Dimensi Aspek Layanan untuk Dibandingkan pada Grafik:",
                    ["Semua Aspek (A1 - A4)", "A1: Akurasi AI", "A2: Harga & Kuota", "A3: Performa & Server", "A4: Bahasa Indonesia"],
                )
                if asp_col_choice == "Semua Aspek (A1 - A4)":
                    st.bar_chart(df_asp_num.set_index("Nama Aplikasi"))
                else:
                    st.bar_chart(df_asp_num.set_index("Nama Aplikasi")[[str(asp_col_choice)]])

            df_bench = pd.DataFrame(bench["app_absa_matrix"])
            st.dataframe(df_bench, use_container_width=True, hide_index=True)

            # 3. Keluhan Paling Banyak Disetujui Pengguna (Thumbs Up Tertinggi) & Penyebutan Kompetitor
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


if __name__ == "__main__":
    main()
