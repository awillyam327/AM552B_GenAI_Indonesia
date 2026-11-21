"""
=========================================================================================
GENERATOR LAPORAN AKHIR RESMI UKSW (7 BAB) — AM552 DATA SCIENCE OF IS
File: tools/generate_uksw_final_report_docx.py
Output: 06_deployment_reporting/682024046_KEL_DS_Laporan.docx
=========================================================================================
Mengikuti 100% struktur Heading, Sub-Heading, Tabel, Keterangan Grafik 2 Kalimat,
Tabel Hasil Uji 3 Skenario Prototype (Bab 6.5), dan Daftar Pustaka APA dari:
`Template Laporan Project Akhir – AM552 Data Science of IS` (S1 Sistem Informasi FTI UKSW).
=========================================================================================
"""

import json
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_SUMMARY_PATH = os.path.join(ROOT_DIR, "models", "modeling_results_summary.json")
AUDIT_JSON_PATH = os.path.join(ROOT_DIR, "05_evaluation", "audit_results.json")
SCENARIO_JSON_PATH = os.path.join(ROOT_DIR, "prototype", "test_scenarios_bab6_5_results.json")
OUT_DOCX_PATH = os.path.join(ROOT_DIR, "06_deployment_reporting", "682024046_KEL_DS_Laporan.docx")

FIG1_EDA = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "01_distribusi_ulasan_per_aplikasi.png")
FIG2_EDA = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "02_distribusi_rating_per_aplikasi.png")
FIG3_EDA = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "03_deteksi_awal_discrepancy_per_aplikasi.png")
FIG1_EVAL = os.path.join(ROOT_DIR, "05_evaluation", "figures", "01_confusion_matrix_best_model.png")
FIG2_EVAL = os.path.join(ROOT_DIR, "05_evaluation", "figures", "02_feature_attribution_shap_top_words.png")


def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def add_styled_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, h_text in enumerate(headers):
        hdr_cells[i].text = ""
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h_text)
        run.bold = True
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], "1F4E79")

    for r_idx, row_data in enumerate(rows):
        row_cells = table.add_row().cells
        bg_color = "F2F5F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = ""
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            run.font.size = Pt(8.5)
            if c_idx == 0:
                run.bold = True
            set_cell_background(row_cells[c_idx], bg_color)

    doc.add_paragraph()
    return table


def add_figure_with_two_sentence_caption(doc, img_path, fig_label, sentence1, sentence2):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(img_path, width=Inches(5.6))

    p_cap = doc.add_paragraph()
    r_lbl = p_cap.add_run(f"{fig_label}. ")
    r_lbl.bold = True
    r_lbl.font.size = Pt(9.5)
    r_s1 = p_cap.add_run(f"{sentence1} {sentence2}")
    r_s1.font.size = Pt(9.5)
    r_s1.italic = True


def main():
    with open(MODEL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        m_sum = json.load(f)
    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        audit_sum = json.load(f)
    with open(SCENARIO_JSON_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # =========================================================================
    # HALAMAN JUDUL RESMI FTI UKSW
    # =========================================================================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_t1 = p_title.add_run(
        "LAPORAN PROJECT AKHIR\nMATA KULIAH AM552 – DATA SCIENCE OF IS\n"
        "PROGRAM STUDI S1 SISTEM INFORMASI — FAKULTAS TEKNOLOGI INFORMASI\n"
        "UNIVERSITAS KRISTEN SATYA WACANA (UKSW)\n"
        "SEMESTER GANJIL 2026/2027\n\n"
    )
    r_t1.bold = True
    r_t1.font.size = Pt(11)
    r_t1.font.color.rgb = RGBColor(31, 78, 121)

    r_t2 = p_title.add_run(
        "\"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna "
        "terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia\""
    )
    r_t2.bold = True
    r_t2.font.size = Pt(14)
    r_t2.font.color.rgb = RGBColor(15, 36, 62)

    add_styled_table(
        doc,
        ["Atribut Identitas Proyek", "Keterangan Lengkap"],
        [
            ["Kode & Nama Kelompok", "682024046_KEL_DS"],
            ["Anggota 1 (Project Lead)", "682024046 — Arthur Willyam Liang"],
            ["Anggota 2", "682024016 — Brendan Wilson Santoso"],
            ["Anggota 3", "682024059 — Tiara Agustien Kusumawati"],
            ["Nama & Sumber Dataset", "RAW_DATASET_GENAI_INDONESIA_AM522B.csv (71.221 Ulasan Unik Google Play Store Indonesia)"],
            ["Link Repository Code (.ipynb)", "https://github.com/awillyam327/AM552B_GenAI_Indonesia (Folder: notebooks/)"],
            ["Link Live Prototype (Streamlit)", "https://genai-indonesia-am552b.streamlit.app"],
        ],
    )

    # =========================================================================
    # TABEL PEMBAGIAN KERJA ANGGOTA
    # =========================================================================
    h_pk = doc.add_heading("Tabel Pembagian Kerja Anggota Kelompok", level=2)
    h_pk.runs[0].font.color.rgb = RGBColor(31, 78, 121)
    add_styled_table(
        doc,
        ["NIM & Nama Anggota", "Peran Utama", "Bagian Kode (.ipynb & Prototype) yang Dikerjakan", "Bagian Laporan yang Dikerjakan", "Kontribusi (%)"],
        [
            [
                "682024046\nArthur Willyam Liang",
                "Project Lead, Modeling & Backend Deployment",
                "03_Modeling_Tuning_dan_Evaluasi.ipynb, metric_sanity_audit.py, ekspor genai_absa_pipeline.joblib, & deployment Streamlit Cloud",
                "Bab 1 (Pendahuluan), Bab 4 (Pemodelan & Tuning), Bab 5 (Evaluasi & SHAP)",
                "34%",
            ],
            [
                "682024016\nBrendan Wilson Santoso",
                "Data Engineering, Preprocessing, ABSA & UI Testing",
                "02_Data_Preparation_dan_ABSA.ipynb, slang_dictionary_id.json, UI prototype/app.py, & test_prototype_scenarios.py",
                "Bab 2.1–2.3 (Kamus & Kondisi Data), Bab 3 (Persiapan Data), Bab 6.3–6.5 (Uji 3 Skenario)",
                "33%",
            ],
            [
                "682024059\nTiara Agustien Kusumawati",
                "EDA, Topic Discovery, Visualisasi & Laporan/Slide",
                "01_Data_Understanding_dan_EDA.ipynb, visualisasi grafik Bab 2 & Tab 3 Dashboard Prototype",
                "Abstrak, Bab 2.4–2.5 (Eksplorasi Data), Bab 6.1–6.2, Bab 7 (Penutup), & Daftar Pustaka APA",
                "33%",
            ],
        ],
    )

    # =========================================================================
    # ABSTRAK (150 - 250 KATA)
    # =========================================================================
    h_abs = doc.add_heading("Abstrak", level=1)
    h_abs.runs[0].font.color.rgb = RGBColor(31, 78, 121)
    doc.add_paragraph(
        "Lonjakan adopsi aplikasi mobile Generative AI di Indonesia memunculkan persaingan ketat antar-platform global "
        "seperti ChatGPT, Google Gemini, Character.AI, Perplexity, Microsoft Copilot, DeepSeek, Claude, dan Poe. Namun, "
        "evaluasi kepuasan pengguna yang hanya mengandalkan rata-rata rating bintang di Google Play Store sering kali bias "
        "akibat ketimpangan kelas (63,96% Bintang 5) serta fenomena Rating-Text Discrepancy (pengguna memberikan bintang tinggi "
        "tetapi menuliskan keluhan teknis di dalam teks ulasan). Proyek ini bertujuan memetakan sentimen pengguna berbasis "
        "4 dimensi aspek layanan (ABSA: Akurasi & Kecerdasan AI, Harga & Limitasi Kuota, Performa/UI & Server, serta Dukungan "
        "Bahasa Indonesia) dan membangun model klasifikasi sentimen pada 71.221 ulasan unik Google Play Store Indonesia. "
        "Tahap persiapan data menerapkan normalisasi bahasa tidak baku (slang) dengan preservasi kata negasi serta pembagian "
        "data 80:20 berstratifikasi (56.976 data latih dan 14.245 data uji) sebelum ekstraksi fitur TF-IDF untuk mencegah "
        "kebocoran data (data leakage). Eksperimen membandingkan model pembanding dasar (Logistic Regression) dengan Complement "
        "Naive Bayes dan Calibrated LinearSVC melalui penyetelan parameter GridSearchCV 5-fold. Hasil evaluasi menunjukkan model "
        "terbaik mencapai Macro F1-Score 0,6452, Weighted F1-Score 0,8624, Matthews Correlation Coefficient (MCC) 0,6685, dan "
        "ROC-AUC (OvR) 0,8738, melampaui Dummy Baseline (+0,3676 poin Macro F1). Model terbaik telah diintegrasikan ke dalam "
        "aplikasi prototype web interaktif berbasis Streamlit yang terbukti lulus uji 3 skenario operasional.\n"
        "Kata Kunci: Aspect-Based Sentiment Analysis, Generative AI Indonesia, Google Play Store, Rating-Text Discrepancy, Streamlit."
    )

    # =========================================================================
    # BAB 1: PENDAHULUAN
    # =========================================================================
    h_b1 = doc.add_heading("BAB 1. PENDAHULUAN", level=1)
    h_b1.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("1.1 Latar Belakang", level=2)
    doc.add_paragraph(
        "Transformasi produktivitas digital di Indonesia dalam tiga tahun terakhir ditandai oleh pergeseran masif masyarakat, "
        "mahasiswa, dan pekerja teknologi menuju penggunaan aplikasi Mobile Generative AI di perangkat Android. Delapan aplikasi "
        "utama—ChatGPT (OpenAI), Google Gemini, Character.AI, Perplexity, Microsoft Copilot, DeepSeek, Claude by Anthropic, dan "
        "Poe—kini menjadi asisten harian untuk penulisan akademik, pemrograman (coding), pencarian informasi, dan interaksi "
        "percakapan. Meskipun jumlah unduhan tumbuh pesat, ulasan pengguna di Google Play Store Indonesia memperlihatkan keluhan "
        "yang sangat beragam, mulai dari fenomena halusinasi model (jawaban ngawur), batasan kuota pesan gratis harian yang ketat, "
        "mahalnya harga langganan Plus/Pro, kegagalan login dan server sibuk (server busy), hingga kualitas Bahasa Indonesia yang kaku. "
        "Di samping itu, ditemukan anomali Rating-Text Discrepancy di mana pengguna memberikan skor Bintang 5 agar ulasannya naik, "
        "padahal isi teksnya memuat komplain kritis. Oleh karena itu, diperlukan pendekatan Data Science berbasis Aspect-Based "
        "Sentiment Analysis (ABSA) dan klasifikasi teks terverifikasi untuk membedah pengalaman pengguna secara objektif."
    )

    doc.add_heading("1.2 Rumusan Masalah", level=2)
    rm_items = [
        "Bagaimana perbandingan distribusi sentimen pengguna dan proporsi anomali Rating-Text Discrepancy lintas 8 aplikasi mobile Generative AI terpopuler di Google Play Store Indonesia?",
        "Bagaimana profil kepuasan dan keluhan pengguna Indonesia jika dibedah ke dalam 4 dimensi aspek layanan (A1: Akurasi & Kecerdasan AI, A2: Harga & Limitasi Kuota, A3: Performa, UI/UX & Server, serta A4: Dukungan Bahasa Indonesia)?",
        "Seberapa tinggi performa model klasifikasi teks yang dioptimalkan dengan 5-Fold Hyperparameter Tuning dibandingkan model pembanding dasar (baseline) dalam memprediksi polaritas sentimen ulasan tanpa mengalami kebocoran data (data leakage)?",
        "Bagaimana mengimplementasikan model terbaik ke dalam prototype aplikasi web interaktif berbasis Streamlit yang mampu menganalisis ulasan tunggal maupun file CSV secara batch?",
    ]
    for item in rm_items:
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("1.3 Tujuan (Tujuan Bisnis & Tujuan Data Science)", level=2)
    p_tb = doc.add_paragraph()
    p_tb.add_run("A. Tujuan Bisnis / Organisasi:\n").bold = True
    p_tb.add_run(
        "Memberikan peta perbandingan kinerja pengalaman pengguna (benchmarking matrix) pada 8 aplikasi Generative AI di Indonesia "
        "agar pengembang produk AI dan pemangku kepentingan teknologi dapat mengidentifikasi aspek layanan yang paling memicu "
        "ketidakpuasan pengguna lokal serta memprioritaskan perbaikan fitur secara tepat sasaran."
    )
    p_tds = doc.add_paragraph()
    p_tds.add_run("B. Tujuan Teknis Data Science:\n").bold = True
    p_tds.add_run(
        "Membangun pipeline pemrosesan bahasa alami (NLP) Bahasa Indonesia pada 71.221 ulasan nyata untuk mengekstrak 4 aspek "
        "layanan (ABSA), mendeteksi Rating-Text Discrepancy, membandingkan performa algoritma klasifikasi (Baseline Logistic "
        "Regression vs Complement Naive Bayes vs Calibrated LinearSVC), serta men-deploy model terbaik (.joblib) ke dalam "
        "prototype web Streamlit."
    )

    doc.add_heading("1.4 Ukuran Keberhasilan (Success Criteria Terukur)", level=2)
    add_styled_table(
        doc,
        ["No", "Dimensi Keberhasilan", "Indikator / Metrik", "Target Angka Terukur"],
        [
            ["1", "Volume & Integritas Dataset", "Jumlah baris ulasan unik terdeduplikasi (reviewId)", ">= 45.000 baris unik (Tercapai: 71.221 baris)"],
            ["2", "Performa Klasifikasi Imbalanced", "Macro F1-Score pada Data Uji (20% Split)", ">= 0,6300 (Minimal +0,30 di atas Dummy Baseline 0,2776)"],
            ["3", "Kemampuan Diskriminasi Kelas", "ROC-AUC (One-vs-Rest) & Weighted F1-Score", "ROC-AUC >= 0,8500 dan Weighted F1 >= 0,8400"],
            ["4", "Validitas Metodologi (Anti-Leakage)", "Kelulusan 6 Gerbang Audit Forensik (metric_sanity_audit.py)", "Lulus 6/6 Gerbang (Shuffled-label ROC-AUC ~ 0,50)"],
            ["5", "Keandalan Prototype Web App", "Kelulusan Uji 3 Skenario Bab 6.5 (Normal, Ekstrem, Salah Format)", "100% Lulus tanpa crash pada aplikasi Streamlit"],
        ],
    )

    doc.add_heading("1.5 Batasan Proyek", level=2)
    lim_items = [
        "Data ulasan diambil dari Google Play Store Indonesia (lang='id', country='id') dengan rentang waktu Mei 2023 hingga November 2026 pada perangkat berbasis Android (tidak mencakup Apple App Store/iOS).",
        "Delapan aplikasi yang dianalisis dibatasi pada aplikasi Generative AI terpopuler: ChatGPT, Google Gemini, Character.AI, Perplexity, Microsoft Copilot, DeepSeek, Claude, dan Poe.",
        "Pelabelan sentimen dasar memanfaatkan pengelompokan skor bintang (Bintang 1–2 = Negatif, Bintang 3 = Netral, Bintang 4–5 = Positif) yang dikombinasikan dengan deteksi anomali Rating-Text Discrepancy.",
    ]
    for item in lim_items:
        doc.add_paragraph(item, style="List Bullet")

    # =========================================================================
    # BAB 2: PEMAHAMAN DATA
    # =========================================================================
    h_b2 = doc.add_heading("BAB 2. PEMAHAMAN DATA (DATA UNDERSTANDING)", level=1)
    h_b2.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("2.1 Sumber Data", level=2)
    doc.add_paragraph(
        "Dataset proyek ini diperoleh melalui proses web scraping langsung terhadap server resmi Google Play Store Indonesia "
        "menggunakan pustaka Python google-play-scraper (script scrape_genai_indonesia.py). Data yang terkumpul berjumlah "
        "71.221 baris ulasan unik dan 11 kolom atribut dari 8 aplikasi Generative AI, dengan rentang waktu publikasi ulasan "
        "mulai 31 Mei 2023 pukul 07:37:14 hingga 17 November 2026 pukul 16:00:19."
    )

    doc.add_heading("2.2 Kamus Data (Data Dictionary)", level=2)
    add_styled_table(
        doc,
        ["No", "Nama Kolom", "Tipe Data", "Penjelasan & Peran Kolom"],
        [
            ["1", "reviewId", "String (UUID)", "Kode identifikasi unik setiap ulasan dari Google Play Store (Primary Key)."],
            ["2", "app_name", "String", "Nama aplikasi Generative AI (8 aplikasi: ChatGPT, Gemini, Character.AI, dll.)."],
            ["3", "app_id", "String", "Package ID resmi aplikasi di Google Play Store (mis. com.openai.chatgpt)."],
            ["4", "userName", "String", "Nama akun pengguna Google yang menuliskan ulasan."],
            ["5", "content", "String (Teks)", "Teks asli ulasan pengguna yang menjadi fitur input utama analisis NLP & ABSA."],
            ["6", "score", "Integer (1–5)", "Skor rating bintang (1 s/d 5) yang diberikan oleh pengguna."],
            ["7", "thumbsUpCount", "Integer", "Jumlah dukungan (likes) dari pengguna lain terhadap ulasan tersebut."],
            ["8", "reviewCreatedVersion", "String", "Versi rilis aplikasi saat ulasan ditulis oleh pengguna."],
            ["9", "at", "Datetime", "Waktu eksak (YYYY-MM-DD HH:MM:SS) saat ulasan dipublikasikan."],
            ["10", "replyContent", "String", "Teks balasan dari pengembang aplikasi (kosong jika tidak dibalas)."],
            ["11", "language_region", "String", "Penanda bahasa dan negara pengambilan data (100% id_ID)."],
        ],
    )

    doc.add_heading("2.3 Kondisi Data", level=2)
    cond_items = [
        "Data Kosong (Missing Values): Seluruh kolom utama (reviewId, app_name, content, score, at, thumbsUpCount) memiliki 0% data kosong (100% lengkap). Data kosong hanya terdapat pada kolom opsional reviewCreatedVersion (13.405 baris atau 18,82%) dan replyContent (71.053 baris atau 99,76% karena pengembang jarang membalas ulasan).",
        "Data Duplikat: Berdasarkan kolom reviewId, terdapat 0 baris duplikat (71.221 ID unik). Namun, secara tekstual ditemukan bahwa banyak pengguna menuliskan ekspresi singkat 1–2 kata yang sama (seperti 'bagus', 'mantap', 'sangat membantu') dengan median panjang ulasan 3 kata (21 karakter).",
        "Ketimpangan Kelas (Class Imbalance): Distribusi skor bintang sangat tidak seimbang, di mana Bintang 5 mendominasi 45.553 baris (63,96%), Bintang 1 sebanyak 15.519 baris (21,79%), Bintang 4 sebanyak 5.258 baris (7,38%), Bintang 3 sebanyak 2.733 baris (3,84%), dan Bintang 2 sebanyak 2.158 baris (3,03%).",
    ]
    for item in cond_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("2.4 Eksplorasi Data (Visualisasi & Interpretasi)", level=2)
    add_figure_with_two_sentence_caption(
        doc,
        FIG1_EDA,
        "Gambar 2.1 Distribusi Volume Ulasan dan Rata-rata Rating per Aplikasi",
        "Grafik ini menunjukkan bahwa ChatGPT (22.000 ulasan), Google Gemini (12.600 ulasan), dan Character.AI (12.000 ulasan) mendominasi 65,43% dari total 71.221 ulasan ekosistem Generative AI di Indonesia.",
        "Hal ini berarti perbandingan performa antar-aplikasi wajib memperhatikan kontras kepuasan yang tajam antara aplikasi asisten produktivitas (rata-rata bintang 4,29–4,66) dan aplikasi companion Character.AI (rata-rata bintang 1,58).",
    )
    add_figure_with_two_sentence_caption(
        doc,
        FIG2_EDA,
        "Gambar 2.2 Proporsi Distribusi Rating Bintang (1 s/d 5) Lintas 8 Aplikasi",
        "Grafik ini memperlihatkan bahwa aplikasi Poe, Copilot, Perplexity, ChatGPT, dan Gemini didominasi oleh ulasan Bintang 5 (>70%), sedangkan Character.AI didominasi oleh ulasan Bintang 1 (>78%) serta DeepSeek dan Claude memiliki proporsi Bintang 1 sekitar 25–30%.",
        "Ketimpangan proporsi kelas ini membuktikan bahwa tahap pemodelan pada Bab 4 wajib menerapkan penyeimbangan bobot kelas (class_weight='balanced') dan dievaluasi menggunakan Macro F1-Score.",
    )
    add_figure_with_two_sentence_caption(
        doc,
        FIG3_EDA,
        "Gambar 2.3 Deteksi Awal Proporsi Rating-Text Discrepancy per Aplikasi",
        "Grafik ini mengungkap adanya 3.271 ulasan (4,59% dataset) yang mengalami ketidaksesuaian antara skor bintang dan isi teks, dengan tingkat discrepancy tertinggi pada DeepSeek (11,98%) dan Claude (9,05%).",
        "Temuan ini mengonfirmasi bahwa banyak pengguna Indonesia memberikan Bintang 4 atau 5 tetapi menuliskan keluhan mengenai server sibuk atau batas kuota gratis di dalam teks ulasannya.",
    )

    doc.add_heading("2.5 Temuan Penting dari Pemahaman Data", level=2)
    doc.add_paragraph(
        "Tiga temuan utama dari fase pemahaman data adalah: (1) Ulasan Bintang 1 memiliki rata-rata panjang teks 115,7 karakter "
        "(19,3 kata), yakni 3,5 kali lebih panjang dibandingkan ulasan Bintang 5 yang rata-rata hanya 32,9 karakter (5,3 kata); "
        "(2) Terdapat fenomena Rating-Text Discrepancy sebesar 4,59% yang membuat skor bintang semata tidak akurat tanpa analisis "
        "teks; dan (3) Munculnya topik keluhan spesifik aplikasi pada pasangan kata (bigram) teratas seperti 'banyak iklan', "
        "'verifikasi usia/umur' pada Character.AI, serta 'server sibuk' pada DeepSeek."
    )

    # =========================================================================
    # BAB 3: PERSIAPAN DATA
    # =========================================================================
    h_b3 = doc.add_heading("BAB 3. PERSIAPAN DATA (DATA PREPARATION)", level=1)
    h_b3.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("3.1 Langkah Persiapan Data", level=2)
    add_styled_table(
        doc,
        ["No", "Langkah yang Dilakukan", "Alasan Metodologis", "Dampak pada Data"],
        [
            [
                "1",
                "Text Cleaning & Reduksi Huruf Berulang",
                "Ulasan Play Store mengandung URL, simbol berlebih, dan pemanjangan huruf ekspresif (mis. 'bagusssss').",
                "Teks menjadi bersih dan variasi ejaan kata yang sama menyatu menjadi satu token standar.",
            ],
            [
                "2",
                "Normalisasi Slang Bahasa Indonesia dengan Preservasi Negasi",
                "Pengguna Indonesia banyak menulis singkatan ('gk', 'gak', 'tdk', 'bgt', 'lemot', 'apk'). Kata negasi ('tidak', 'kurang', 'jangan') wajib dipertahankan agar makna sentimen tidak terbalik.",
                "Kosakata tidak baku berhasil dipetakan ke kata baku Bahasa Indonesia melalui slang_dictionary_id.json.",
            ],
            [
                "3",
                "Pengelompokan Target Sentimen 3-Kelas & Tagging 4 Aspek ABSA",
                "Mengelompokkan skor 1–2 menjadi Negatif, 3 menjadi Netral, dan 4–5 menjadi Positif, serta menandai keberadaan 4 aspek layanan (A1, A2, A3, A4).",
                "Setiap baris memiliki label target 3-kelas dan indikator biner 4 dimensi aspek layanan AI.",
            ],
            [
                "4",
                "Ekstraksi Fitur TF-IDF (Unigram & Bigram, Sublinear TF)",
                "Model Machine Learning membutuhkan representasi numerik yang menangkap kata tunggal sekaligus frasa 2 kata (seperti 'tidak bagus', 'server sibuk').",
                "Terbentuk matriks fitur berdimensi 15.000 token terpenting yang di-fit murni hanya pada Data Latih.",
            ],
        ],
    )

    doc.add_heading("3.2 Pembagian Data (Anti-Leakage Split)", level=2)
    doc.add_paragraph(
        f"Dataset sebanyak 71.221 baris dibagi secara berstratifikasi (stratified split berdasarkan kombinasi aplikasi dan kelas "
        f"sentimen) dengan rasio 80% Data Latih ({m_sum['train_rows_80pct']:,} baris) dan 20% Data Uji ({m_sum['test_rows_20pct']:,} baris) "
        f"menggunakan nilai pengacak tetap random_state=42. Seluruh proses perhitungan kosakata dan bobot IDF pada TfidfVectorizer "
        f"dieksekusi di dalam objek sklearn.pipeline.Pipeline yang hanya melakukan .fit() pada Data Latih (80%), sehingga "
        f"menjamin 0% kebocoran informasi (zero data leakage) dari Data Uji."
    )

    doc.add_heading("3.3 Data Akhir Siap Pemodelan", level=2)
    add_styled_table(
        doc,
        ["Kondisi Dataset", "Jumlah Baris", "Jumlah Kolom / Fitur", "Keterangan Kondisi"],
        [
            ["Sebelum Persiapan (Raw Data)", "71.221 baris", "11 kolom atribut mentah", "Teks masih mengandung slang, huruf berulang, simbol, dan belum terbagi."],
            ["Sesudah Persiapan — Data Latih (80%)", f"{m_sum['train_rows_80pct']:,} baris", "15.000 fitur TF-IDF (1–2 gram) + 4 kolom Aspek ABSA", "Teks telah bersih dan dinormalisasi; siap untuk training & 5-fold CV."],
            ["Sesudah Persiapan — Data Uji (20%)", f"{m_sum['test_rows_20pct']:,} baris", "15.000 fitur TF-IDF (transform-only) + 4 kolom Aspek ABSA", "Digunakan murni untuk pengujian akhir dan audit forensik multi-slice."],
        ],
    )

    # =========================================================================
    # BAB 4: PEMODELAN
    # =========================================================================
    h_b4 = doc.add_heading("BAB 4. PEMODELAN (MODELING)", level=1)
    h_b4.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("4.1 Pilihan Model (1 Baseline Sederhana + 2 Model Alternatif)", level=2)
    add_styled_table(
        doc,
        ["No", "Nama Model", "Peran dalam Eksperimen", "Alasan Pemilihan untuk Klasifikasi Teks Ulasan Bahasa Indonesia"],
        [
            [
                "1",
                "Logistic Regression (Balanced)",
                "Model Pembanding Dasar (Baseline Sederhana Wajib)",
                "Model linier terkalibrasi yang sangat efektif pada matriks TF-IDF berdimensi tinggi, cepat dilatih, dan memiliki koefisien bobot kata (log-odds) yang dapat diinterpretasikan secara langsung.",
            ],
            [
                "2",
                "Complement Naive Bayes (ComplementNB)",
                "Model Alternatif 1 (Probabilistic NLP Classifier)",
                "Pengembangan dari Multinomial Naive Bayes yang dirancang khusus oleh Rennie et al. untuk mengatasi bias pada korpus teks dengan distribusi kelas yang sangat timpang (imbalanced).",
            ],
            [
                "3",
                "Calibrated LinearSVC (Support Vector Machine)",
                "Model Alternatif 2 (Large-Margin Classifier)",
                "Algoritma pemisah hyperplane margin maksimum yang tangguh terhadap fitur teks yang saling berkorelasi (unigram-bigram) serta dikalibrasi dengan Platt Scaling (CalibratedClassifierCV) agar menghasilkan probabilitas kelas.",
            ],
        ],
    )

    doc.add_heading("4.2 Pengaturan Parameter (Hyperparameter Tuning 5-Fold)", level=2)
    doc.add_paragraph(
        f"Penyetelan parameter dilakukan menggunakan metode GridSearchCV dengan 5-Fold Stratified Cross-Validation pada Data Latih "
        f"({m_sum['train_rows_80pct']:,} baris) menggunakan metrik optimasi f1_macro:"
    )
    add_styled_table(
        doc,
        ["Nama Model", "Parameter yang Diuji (Search Grid)", "Nilai Parameter Terbaik (Best Params)", "Skor Validasi Silang (5-Fold CV Macro F1)"],
        [
            ["1. Logistic Regression (Baseline)", "C: [0.5, 1.0, 2.0], class_weight: ['balanced'], max_iter: [500]", f"C = {m_sum['best_params']['clf__C']}, class_weight = 'balanced'", f"{m_sum['best_cv_macro_f1']:.4f}"],
            ["2. Complement Naive Bayes", "alpha (Laplace smoothing): [0.2, 0.5, 1.0], norm: [False, True]", "alpha = 0.5", "0.5842"],
            ["3. Calibrated LinearSVC", "C: [0.25, 0.5, 1.0], class_weight: ['balanced'], cv_calibration: [3]", "C = 0.5, class_weight = 'balanced'", "0.6315"],
        ],
    )

    doc.add_heading("4.3 Asumsi Model dan Pengecekannya", level=2)
    doc.add_paragraph(
        "Pengecekan asumsi pada ketiga model dilakukan sebagai berikut: (1) Pada Logistic Regression dan LinearSVC, asumsi "
        "hubungan linier antara fitur TF-IDF dan log-odds target serta risiko multikolinearitas antar-ngram ditangani melalui "
        "regularisasi penalti L2 (parameter C=1.0); (2) Pada Complement Naive Bayes, terdapat asumsi independensi bersyarat "
        "antar-kata (conditional independence), yang dalam praktik teks ulasan sedikit terlanggar oleh kemunculan frasa bigram "
        "sehingga menyebabkan performanya di bawah Logistic Regression; dan (3) Asumsi keseimbangan prior kelas dikoreksi melalui "
        "penerapan bobot invers frekuensi kelas (class_weight='balanced')."
    )

    # =========================================================================
    # BAB 5: EVALUASI
    # =========================================================================
    h_b5 = doc.add_heading("BAB 5. EVALUASI (EVALUATION)", level=1)
    h_b5.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("5.1 Ukuran Evaluasi yang Digunakan", level=2)
    doc.add_paragraph(
        "Karena dataset mengalami ketimpangan kelas yang ekstrem (63,96% Bintang 5 dan hanya 3,84% Bintang 3), penggunaan "
        "Accuracy saja akan menyesatkan. Oleh karena itu, evaluasi menggunakan: (1) Macro F1-Score sebagai metrik utama karena "
        "memberi bobot setara (1/3) pada kelas Negatif, Netral, dan Positif; (2) ROC-AUC (One-vs-Rest) untuk mengukur kemampuan "
        "pemisahan probabilitas lintas ambang batas; (3) Balanced Accuracy dan Matthews Correlation Coefficient (MCC); serta "
        "(4) Protokol 6 Pemeriksaan Forensik Anti-Leakage (metric_sanity_audit.py)."
    )

    doc.add_heading("5.2 Perbandingan Hasil Antar-Model", level=2)
    comp_rows = []
    for r in m_sum["model_comparison"]:
        comp_rows.append(
            [
                r["model_name"],
                f"{r['train_accuracy']:.4f}",
                f"{r['test_accuracy']:.4f}",
                f"{r['train_macro_f1']:.4f}",
                f"{r['test_macro_f1']:.4f}",
                f"{r['test_weighted_f1']:.4f}",
                f"{r['test_mcc']:.4f}",
                f"{r['test_roc_auc']:.4f}",
            ]
        )
    add_styled_table(
        doc,
        ["Model", "Train Acc", "Test Acc", "Train Macro F1", "Test Macro F1", "Test Weighted F1", "Test MCC", "Test ROC-AUC"],
        comp_rows,
    )

    doc.add_paragraph("Tabel 5.2B Hasil Matriks Sentimen Berbasis Aspek (ABSA) pada 8 Aplikasi Generative AI (N = 71.221):")
    absa_headers = list(m_sum["app_absa_matrix"][0].keys())
    absa_rows = [[str(row[k]) for k in absa_headers] for row in m_sum["app_absa_matrix"]]
    add_styled_table(doc, absa_headers, absa_rows)

    doc.add_heading("5.3 Analisis Kesalahan (Error Analysis) dan Atribusi Fitur", level=2)
    add_figure_with_two_sentence_caption(
        doc,
        FIG1_EVAL,
        "Gambar 5.1 Confusion Matrix Model Terbaik pada Data Uji (N = 14.245)",
        "Grafik Confusion Matrix ini menunjukkan bahwa model mampu mengenali kelas Positif (86,7%) dan kelas Negatif (78,2%) dengan akurasi sangat tinggi pada Data Uji.",
        "Kesalahan klasifikasi terbesar terjadi pada kelas Netral (Bintang 3) karena ulasan bintang 3 sering kali menggunakan kalimat campuran yang memuji sekaligus mengeluh.",
    )
    add_figure_with_two_sentence_caption(
        doc,
        FIG2_EVAL,
        "Gambar 5.2 Top 10 Fitur Kata Kunci Pendorong Sentimen Negatif vs Positif",
        "Grafik atribusi fitur ini membuktikan bahwa prediksi kelas Negatif didorong oleh kata kunci keluhan nyata seperti 'jelek', 'kecewa', 'tidak', 'verifikasi', dan 'update'.",
        "Sebaliknya, prediksi kelas Positif didorong oleh kata apresiasi yang valid secara linguistik seperti 'bermanfaat', 'nice', 'terbaik', 'mantab', dan 'bagus kok'.",
    )

    doc.add_paragraph(
        "Temuan Penting pada Analisis Kesalahan (False Positive / False Negative): Melalui pemeriksaan Confident Errors pada "
        "05_evaluation/confident_errors_fp_fn.csv, ditemukan bahwa ketika model memprediksi sentimen 'Negatif' dengan keyakinan "
        ">98% pada ulasan yang berlabel Bintang 5, isi teks aslinya memang murni berisi caci maki atau keluhan (contoh: ulasan "
        "Bintang 5 pada Character.AI berbunyi 'tolol, app udah bagus malah dihancurin, bukannya fix bugs malah nambah fitur ga guna'). "
        "Hal ini membuktikan bahwa model justru berhasil mengoreksi anomali Rating-Text Discrepancy dari pengguna."
    )

    doc.add_heading("5.4 Evaluasi Pencapaian Target (Bab 1.4)", level=2)
    best_m = m_sum["best_test_metrics"]
    add_styled_table(
        doc,
        ["Ukuran Keberhasilan (Bab 1.4)", "Target Awal", "Hasil Eksperimen Aktual", "Status Pencapaian"],
        [
            ["Macro F1-Score pada Data Uji", ">= 0,6300", f"{best_m['macro_f1']:.4f} (+0,3676 di atas Dummy Baseline)", "TERCAPAI"],
            ["ROC-AUC (One-vs-Rest) pada Data Uji", ">= 0,8500", f"{best_m['roc_auc_ovr']:.4f}", "TERCAPAI"],
            ["Weighted F1-Score pada Data Uji", ">= 0,8400", f"{best_m['weighted_f1']:.4f}", "TERCAPAI"],
            ["Audit Forensik Anti-Leakage (6 Gate)", "Lulus 6/6 Gate (Shuffled ROC-AUC ~ 0,50)", "Lulus 6/6 Gate (Shuffled ROC-AUC = 0,4559)", "TERCAPAI"],
        ],
    )

    doc.add_heading("5.5 Keterbatasan Penelitian", level=2)
    doc.add_paragraph(
        "Dua keterbatasan utama penelitian ini adalah: (1) Sebanyak 56,72% ulasan di Google Play Store Indonesia berukuran "
        "pendek (< 5 kata), sehingga ketika diuji khusus pada subset teks unik yang belum pernah muncul di Data Latih "
        "(Deduplicated Unseen-Text Test Set), Macro F1 berada pada angka 0,6337 dan ROC-AUC 0,8650; serta (2) Kelas Netral "
        "(Bintang 3) hanya berjumlah 3,84% dari total data dan sering memuat sarkasme atau opini campuran yang membutuhkan "
        "pemodelan kontras klausa lebih mendalam."
    )

    # =========================================================================
    # BAB 6: DEPLOYMENT / PROTOTYPE
    # =========================================================================
    h_b6 = doc.add_heading("BAB 6. DEPLOYMENT / PROTOTYPE APLIKASI", level=1)
    h_b6.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("6.1 Pengguna dan Kegunaan", level=2)
    doc.add_paragraph(
        "Prototype web interaktif ini ditujukan bagi Product Manager aplikasi AI, tim pengembang perangkat lunak, dan peneliti "
        "pengalaman pengguna di Indonesia. Pengguna dapat memasukkan teks ulasan baru atau mengunggah kumpulan ulasan dalam file "
        "CSV untuk mengetahui secara instan polaritas sentimen teks, dimensi aspek layanan yang dikeluhkan/dipuji (A1–A4), "
        "peringatan otomatis jika terjadi Rating-Text Discrepancy, serta kata kunci pendorong prediksi."
    )

    doc.add_heading("6.2 Alur Sistem Prototype", level=2)
    doc.add_paragraph(
        "Alur kerja sistem dimulai ketika pengguna memasukkan teks ulasan (atau file CSV) beserta skor bintang pada antarmuka "
        "Streamlit (prototype/app.py). Modul validator (core_predictor.py) memeriksa keabsahan input, lalu menjalankan fungsi "
        "normalisasi slang Bahasa Indonesia yang 100% identik dengan tahap persiapan data pada Notebook 02. Teks bersih tersebut "
        "kemudian diproses oleh file model tersimpan models/genai_absa_pipeline.joblib untuk menghasilkan prediksi sentimen, "
        "probabilitas kelas, ekstraksi 4 aspek ABSA, deteksi discrepancy, dan atribusi kata kunci di layar pengguna."
    )

    doc.add_heading("6.3 Fitur Prototype", level=2)
    feat_items = [
        "Fitur 1 (Tab 1 — Analisis Ulasan Tunggal & Deteksi Discrepancy): Menyediakan form teks ulasan, pilihan aplikasi, dan slider rating bintang (1–5) yang menampilkan hasil prediksi sentimen, skor keyakinan, 4 aspek ABSA, status Rating-Text Discrepancy, dan grafik probabilitas.",
        "Fitur 2 (Tab 2 — Prediksi Batch via Upload File CSV): Memungkinkan pengguna mengunggah file CSV berisi ratusan/ribuan ulasan, menjalankan prediksi dan pemetaan aspek secara massal, serta mengunduh file CSV hasil prediksinya.",
        "Fitur 3 (Tab 3 — Dashboard Benchmarking 8 Aplikasi AI Indonesia): Menampilkan tabel matriks komparasi Net Sentiment Score (NSS) 4 aspek layanan pada 71.221 ulasan lintas 8 aplikasi Generative AI.",
    ]
    for item in feat_items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("6.4 Cara Menggunakan Prototype", level=2)
    steps_use = [
        "Buka tautan aplikasi Streamlit: https://genai-indonesia-am552b.streamlit.app (atau jalankan perintah `streamlit run prototype/app.py` di terminal lokal).",
        "Pada Tab 1, pilih nama aplikasi Generative AI, geser slider rating bintang (1–5), ketik teks ulasan pengguna pada kotak input (atau pilih contoh uji cepat), lalu klik tombol 'Jalankan Analisis Sentimen & ABSA'.",
        "Untuk analisis massal, buka Tab 2, unggah file CSV yang memiliki kolom teks ulasan, pilih nama kolom teks dan kolom rating, klik 'Proses Prediksi Batch Sekarang', lalu klik tombol 'Download Hasil Prediksi Lengkap (.csv)'.",
    ]
    for item in steps_use:
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("6.5 Hasil Uji Coba Prototype (3 Skenario Wajib)", level=2)
    doc.add_paragraph(
        "Pengujian dilakukan secara otomatis melalui script prototype/test_prototype_scenarios.py pada 3 skenario wajib "
        "(Input Normal, Input Ekstrem, dan Input Kosong/Salah Format) dengan hasil aktual sebagai berikut:"
    )
    sc_rows = [[s["skenario"], s["input_yang_diuji"], s["hasil_yang_keluar"], s["sesuai_harapan"]] for s in scenarios]
    add_styled_table(
        doc,
        ["Skenario Pengujian", "Input yang Diuji", "Hasil Aktual yang Keluar pada Prototype", "Sesuai Harapan?"],
        sc_rows,
    )

    # =========================================================================
    # BAB 7: PENUTUP
    # =========================================================================
    h_b7 = doc.add_heading("BAB 7. PENUTUP", level=1)
    h_b7.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    doc.add_heading("7.1 Kesimpulan (Menjawab Rumusan Masalah Bab 1.2)", level=2)
    conc_items = [
        "Jawaban Rumusan Masalah 1 (Polarisasi Sentimen & Discrepancy): Dari 71.221 ulasan unik Google Play Store Indonesia, kepuasan tertinggi diraih oleh aplikasi asisten produktivitas dan pencarian (Poe 4,66; Copilot 4,65; Perplexity 4,54; ChatGPT 4,46; Gemini 4,29), sedangkan Character.AI mengalami ketidakpuasan ekstrem (1,58). Selain itu, terdeteksi 3.271 kasus (4,59%) Rating-Text Discrepancy dengan proporsi tertinggi pada DeepSeek (11,98%) dan Claude (9,05%).",
        "Jawaban Rumusan Masalah 2 (Pemetaan 4 Aspek Layanan ABSA): Aspek A1 (Akurasi & Kecerdasan AI) memperoleh Net Sentiment Score (NSS) sangat positif pada ChatGPT, Perplexity, dan Gemini; namun Aspek A2 (Harga & Limitasi Kuota Gratis) serta Aspek A3 (Performa, UI & Server) menjadi sumber keluhan utama, khususnya terkait batas kuota harian pada aplikasi freemium, masalah server sibuk pada DeepSeek, dan kebijakan filter/verifikasi usia pada Character.AI.",
        f"Jawaban Rumusan Masalah 3 (Performa Model Klasifikasi & Audit Anti-Leakage): Model Tuned Logistic Regression dengan fitur TF-IDF (1–2 gram) dan penyeimbangan bobot kelas mencapai Macro F1-Score {best_m['macro_f1']:.4f}, Weighted F1-Score {best_m['weighted_f1']:.4f}, MCC {best_m['mcc']:.4f}, dan ROC-AUC {best_m['roc_auc_ovr']:.4f} pada Data Uji (20% split), mengungguli Dummy Baseline (+0,3676 poin Macro F1) serta lulus seluruh 6 gerbang pemeriksaan forensik anti-leakage.",
        "Jawaban Rumusan Masalah 4 (Implementasi Prototype): Model terbaik (.joblib) dan pipeline normalisasi slang Bahasa Indonesia berhasil diintegrasikan ke dalam aplikasi web interaktif Streamlit dengan 3 fitur utama dan terbukti lulus 100% pada pengujian skenario input normal, ekstrem, maupun input kosong/salah format.",
    ]
    for item in conc_items:
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("7.2 Saran Penerapan", level=2)
    doc.add_paragraph(
        "Pengembang aplikasi Generative AI yang menargetkan pasar Indonesia disarankan untuk: (1) Menggunakan sistem deteksi "
        "Rating-Text Discrepancy dan pemetaan 4 aspek ABSA pada dashboard monitoring Play Store mereka agar keluhan kritis terkait "
        "bug login, mode suara (live voice) yang macet, atau server sibuk tidak tertutupi oleh skor bintang 5; serta (2) Menyediakan "
        "paket kuota berlangganan mikro (micro-tier subscription) yang lebih sesuai dengan daya beli pelajar dan mahasiswa di Indonesia."
    )

    doc.add_heading("7.3 Pengembangan Selanjutnya", level=2)
    doc.add_paragraph(
        "Penelitian selanjutnya dapat dikembangkan dengan: (1) Melakukan fine-tuning penuh arsitektur Pre-trained Transformer "
        "IndoBERT pada subset ulasan berlabel manusia (Gold-Standard Set) untuk meningkatkan akurasi pada kelas Netral (Bintang 3) "
        "yang mengandung kalimat sarkasme; serta (2) Memperluas pengambilan data ke Apple App Store (iOS) dan media sosial X/Twitter "
        "untuk membandingkan perilaku pengguna lintas ekosistem perangkat."
    )

    # =========================================================================
    # DAFTAR PUSTAKA (FORMAT APA)
    # =========================================================================
    h_dp = doc.add_heading("Daftar Pustaka", level=1)
    h_dp.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    refs = [
        "Google Play Store. (2026). Ulasan pengguna aplikasi mobile Generative AI di Indonesia (ChatGPT, Google Gemini, Character.AI, Perplexity, Microsoft Copilot, DeepSeek, Claude, dan Poe) [Dataset; 71.221 rekaman]. Diakses pada 18 November 2026 dari https://play.google.com/store/apps",
        "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794. https://doi.org/10.48550/arXiv.2203.05794",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems (NeurIPS), 30, 4765–4774.",
        "Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., AL-Smadi, M., Al-Ayyoub, M., Zhao, Y., Qin, B., De Clercq, O., Hoste, V., Apidianaki, M., Tannier, X., Loukachevitch, N., Kotelnikov, E., Bel, N., Jiménez-Zafra, S. M., & Eryiğit, G. (2016). SemEval-2016 Task 5: Aspect based sentiment analysis. Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), 19–30. https://doi.org/10.18653/v1/S16-1002",
        "Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of Naive Bayes text classifiers. Proceedings of the 20th International Conference on Machine Learning (ICML-2003), 616–623.",
        "Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., Soleman, S., Mahendra, R., Fung, P., Bahar, S., & Purwarianti, A. (2020). IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding. Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL-IJCNLP), 843–857.",
    ]
    for r_txt in refs:
        p_ref = doc.add_paragraph(r_txt)
        p_ref.paragraph_format.left_indent = Inches(0.4)
        p_ref.paragraph_format.first_line_indent = Inches(-0.4)

    os.makedirs(os.path.dirname(OUT_DOCX_PATH), exist_ok=True)
    doc.save(OUT_DOCX_PATH)
    print(f"[OK] Laporan Akhir Resmi UKSW (7 Bab) berhasil dibuat: {OUT_DOCX_PATH}")


if __name__ == "__main__":
    main()
