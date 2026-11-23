"""
=========================================================================================
GENERATOR SLIDE PRESENTASI POWERPOINT RESMI (AM552 DATA SCIENCE OF IS - FTI UKSW)
File: tools/generate_presentation_pptx.py
Output: 06_deployment_reporting/682024046_KEL_DS_Presentasi.pptx
=========================================================================================
Menghasilkan 12 Slide Presentasi Padat, Modern & Lengkap dengan Gambar/Grafik
Sesuai Struktur 7 Bab Laporan Resmi UKSW dan Aturan Penamaan 682024046_KEL_DS.
=========================================================================================
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_PPTX_PATH = os.path.join(ROOT_DIR, "06_deployment_reporting", "682024046_KEL_DS_Presentasi.pptx")

# Gambar & visualisasi
FIG_EDA1 = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "01_distribusi_ulasan_per_aplikasi.png")
FIG_EDA2 = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "02_distribusi_rating_per_aplikasi.png")
FIG_CM = os.path.join(ROOT_DIR, "05_evaluation", "figures", "01_confusion_matrix_best_model.png")
FIG_SHAP = os.path.join(ROOT_DIR, "05_evaluation", "figures", "02_feature_attribution_shap_top_words.png")
FIG_TOPIC = os.path.join(ROOT_DIR, "04_modeling", "figures", "01_topic_keywords_lda.png")
FIG_SC1 = os.path.join(ROOT_DIR, "06_deployment_reporting", "figures", "screenshot_prototype_tab1.png")
FIG_SC2 = os.path.join(ROOT_DIR, "06_deployment_reporting", "figures", "screenshot_prototype_tab2.png")

# Palet warna profesional UKSW (Navy Blue & Slate)
COLOR_NAVY = RGBColor(31, 78, 121)       # #1F4E79
COLOR_DARK = RGBColor(15, 36, 62)        # #0F243E
COLOR_ACCENT = RGBColor(46, 117, 182)    # #2E75B6
COLOR_GRAY = RGBColor(100, 110, 120)     # #646E78
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_CARD_BG = RGBColor(242, 245, 249)  # #F2F5F9

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)  # 16:9 Widescreen
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category_text="AM552 - DATA SCIENCE OF IS | FTI UKSW"):
        # Header banner text
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT

        p_tit = tf.add_paragraph()
        p_tit.text = title_text
        p_tit.font.size = Pt(22)
        p_tit.font.bold = True
        p_tit.font.color.rgb = COLOR_DARK

    def add_card(slide, left, top, width, height, title, items, bg_color=COLOR_CARD_BG):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = RGBColor(210, 220, 230)
        shape.line.width = Pt(1)

        txBox = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
        tf = txBox.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = COLOR_NAVY
        p0.space_after = Pt(8)

        for item in items:
            p = tf.add_paragraph()
            p.text = f"•  {item}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = RGBColor(40, 50, 60)
            p.space_after = Pt(5)

    # -------------------------------------------------------------------------
    # SLIDE 1: Title & Team Identity
    # -------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent
    bg_shape = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = COLOR_CARD_BG
    bg_shape.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(3.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    
    p = tf1.paragraphs[0]
    p.text = "LAPORAN PROYEK AKHIR — MATA KULIAH AM552 DATA SCIENCE OF IS"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_ACCENT

    p = tf1.add_paragraph()
    p.text = "Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Mobile Generative AI di Indonesia"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_DARK
    p.space_before = Pt(8)
    p.space_after = Pt(14)

    p = tf1.add_paragraph()
    p.text = "Studi Kasus 71.221 Ulasan Google Play Store Indonesia (Jan 2023 – Nov 2026) | Kode Kelompok: 682024046_KEL_DS"
    p.font.size = Pt(11)
    p.font.color.rgb = COLOR_GRAY

    # Cards 3 Anggota Tim
    add_card(s1, Inches(1.0), Inches(4.3), Inches(3.6), Inches(2.4),
             "Arthur Willyam Liang\n(682024046)",
             ["Peran: Project Lead & Modeling", "CRISP-DM Method & Metrics", "Model Training & 5-Fold Tuning", "Audit 6 Forensik & Joblib", "Backend Deployment Streamlit"],
             bg_color=COLOR_WHITE)

    add_card(s1, Inches(4.85), Inches(4.3), Inches(3.6), Inches(2.4),
             "Brendan Wilson Santoso\n(682024016)",
             ["Peran: Data & NLP Engineer", "Integritas 71.221 Baris Ulasan", "Kamus Slang & Preservasi Negasi", "Ekstraksi 4 Aspek ABSA", "UI Prototype Streamlit & Uji 3 Skenario"],
             bg_color=COLOR_WHITE)

    add_card(s1, Inches(8.7), Inches(4.3), Inches(3.6), Inches(2.4),
             "Tiara Agustien Kusumawati\n(682024059)",
             ["Peran: EDA & Technical Writer", "Exploratory Data Analysis Bab 2", "Topic Discovery (LDA Keluhan/Pujian)", "Penyusunan Laporan 7 Bab UKSW", "Desain Slide Presentasi PDF"],
             bg_color=COLOR_WHITE)

    # -------------------------------------------------------------------------
    # SLIDE 2: Bab 1 - Latar Belakang & Paradoks Rating Bintang
    # -------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Latar Belakang: Ledakan AI & Paradoks Rating Play Store (Bab 1.1–1.2)")
    add_card(s2, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2),
             "Fenomena Adopsi Generative AI di Indonesia",
             [
                 "Pergeseran masif cara mahasiswa, pekerja kantor, dan masyarakat mencari informasi dan menyelesaikan tugas.",
                 "8 Aplikasi Terpopuler dianalisis: ChatGPT, Gemini, Character.AI, Perplexity, Copilot, DeepSeek, Claude, Poe.",
                 "71.221 ulasan nyata Google Play Store Indonesia berhasil dikumpulkan (100% real data region id_ID).",
                 "Rata-rata rating bintang Play Store sering kali 'menipu' dan gagal menangkap keluhan mendalam di balik angka agregat."
             ])
    add_card(s2, Inches(6.8), Inches(1.7), Inches(5.6), Inches(5.2),
             "Rumusan Masalah & Rating-Text Discrepancy",
             [
                 "Bagaimana kepuasan riil pengguna jika ditinjau dari 4 Aspek Spesifik (Akurasi, Kuota/Harga, Server/UI, Bahasa)?",
                 "Seberapa besar proporsi ulasan yang mengalami Rating-Text Discrepancy (Bintang 5 tapi teks mengeluh, atau Bintang 1 tapi memuji)?",
                 "Mengapa aplikasi seperti Character.AI ratingnya anjlok ke 1.58 sementara ChatGPT stabil di 4.46?",
                 "Apa saja topik keluhan utama pengguna mobile AI di Indonesia yang perlu diperbaiki developer?"
             ])

    # -------------------------------------------------------------------------
    # SLIDE 3: Bab 1 - Tujuan Bisnis vs Data Science & Target KPI
    # -------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Tujuan Bisnis vs Data Science & Target Metrik Terukur (Bab 1.3–1.4)")
    add_card(s3, Inches(0.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Tujuan Bisnis (Business Goals)",
             [
                 "Menyediakan audit kepuasan objektif bagi 8 ekosistem aplikasi AI di Indonesia.",
                 "Membantu pengguna memilih aplikasi AI sesuai kebutuhan nyata (koding vs roleplay vs riset).",
                 "Memberikan rekomendasi manajerial bagi developer untuk mengatasi friksi layanan di pasar lokal."
             ])
    add_card(s3, Inches(4.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Tujuan Data Science (DS Goals)",
             [
                 "Membangun pipeline klasifikasi sentimen teks 3-kelas (Positif, Netral, Negatif).",
                 "Mengekstraksi polaritas sentimen pada 4 kategori aspek layanan (ABSA).",
                 "Mendeteksi anomali Rating-Text Discrepancy secara otomatis menggunakan Machine Learning.",
                 "Mendeploy prototipe web interaktif ke Streamlit Cloud."
             ])
    add_card(s3, Inches(8.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Target Metrik Terukur (Bab 1.4)",
             [
                 "Macro F1-Score: Target >= 0.63 (Mengalahkan baseline acak 0.2776).",
                 "ROC-AUC (OvR): Target >= 0.85.",
                 "Weighted F1: Target >= 0.80.",
                 "Zero Data Leakage: 80:20 Stratified Split sebelum fitting vektor teks.",
                 "Uji 3 Skenario Prototipe: Lolos uji Normal, Ekstrem, dan Edge Case."
             ])

    # -------------------------------------------------------------------------
    # SLIDE 4: Bab 2 - Profil Dataset 71.221 Ulasan Play Store
    # -------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Pemahaman Data: Audit Kualitas & Profil 8 Aplikasi (Bab 2.1–2.3)")
    add_card(s4, Inches(0.8), Inches(1.7), Inches(4.6), Inches(5.2),
             "Karakteristik & Integritas Dataset",
             [
                 "Total Ulasan: 71.221 baris unik (100% reviewId unik, 0% duplikasi).",
                 "Rentang Tanggal: 31 Mei 2023 s/d 17 November 2026.",
                 "Audit Missing Values: Kolom penting (reviewId, app_name, content, score, at) 0% missing values (100% utuh).",
                 "Kolom replyContent kosong pada 99,76% baris (hanya 0,24% keluhan pengguna Indonesia yang dibalas developer!).",
                 "Bahasa Ulasan: Bahasa Indonesia ragam santai/slang bercampur istilah teknis."
             ])
    if os.path.exists(FIG_EDA1):
        s4.shapes.add_picture(FIG_EDA1, Inches(5.7), Inches(1.7), width=Inches(6.8))

    # -------------------------------------------------------------------------
    # SLIDE 5: Bab 2 - EDA Komparatif & Anomali Character.AI
    # -------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Exploratory Data Analysis: Proporsi Rating & Anomali 1.58 (Bab 2.4–2.5)")
    add_card(s5, Inches(0.8), Inches(1.7), Inches(4.6), Inches(5.2),
             "Temuan Kunci Eksplorasi Data",
             [
                 "Ketimpangan Kelas Global: Bintang 5 mendominasi 63,96% dari seluruh ulasan (positive skew).",
                 "Anomali Ekstrem Character.AI: Rating rata-rata 1,58 dengan proporsi Bintang 1 menembus > 78%.",
                 "Ulasan Negatif Lebih Panjang: Rata-rata Bintang 1 (115 karakter / 19 kata) vs Bintang 5 (33 karakter / 5 kata).",
                 "Character.AI ulasannya 136 karakter (hampir 4x lebih panjang dari ChatGPT 34 karakter) karena pengguna meluapkan emosi kekecewaan."
             ])
    if os.path.exists(FIG_EDA2):
        s5.shapes.add_picture(FIG_EDA2, Inches(5.7), Inches(1.7), width=Inches(6.8))

    # -------------------------------------------------------------------------
    # SLIDE 6: Bab 3 - Persiapan Data & Rekayasa Aspek ABSA
    # -------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Persiapan Data & Pipeline ABSA Bahasa Indonesia (Bab 3.1–3.3)")
    add_card(s6, Inches(0.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Tantangan Teks Indonesia",
             [
                 "Slang & Singkatan Ekstrem: 'gk', 'bgt', 'lemot', 'eror', 'apk', 'udh'.",
                 "Kamus Normalisasi: 80+ leksikon slang Indonesia dibakukan.",
                 "Preservasi Negasi: Kata 'tidak', 'kurang', 'jangan', 'bukan' WAJIB dipertahankan agar makna tidak terbalik.",
                 "Regex Cleaning: Menghapus spam 'aaaaa' dan simbol non-alfanumerik."
             ])
    add_card(s6, Inches(4.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Ekstraksi 4 Aspek Layanan (ABSA)",
             [
                 "A1: Akurasi & Kecerdasan AI (pintar, coding, tugas, halusinasi, ngawur).",
                 "A2: Harga & Batasan Kuota (gratis, bayar, limit, langganan, mahal, token).",
                 "A3: Performa, UI & Server (cepat, lambat, server down, error, login, update).",
                 "A4: Keluwesan Bahasa Indonesia (bahasa indonesia, terjemahan, kaku, logat)."
             ])
    add_card(s6, Inches(8.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Pemisahan Data Anti-Leakage (3.2)",
             [
                 "Total: 71.221 ulasan dibagi 80:20 Stratified Split (random_state=42).",
                 "Data Latih: 56.976 ulasan.",
                 "Data Uji: 14.245 ulasan.",
                 "Zero Leakage Protocol: TfidfVectorizer hanya di-fit pada data latih, lalu mentransform data uji.",
                 "Representasi: TF-IDF (1,2-gram), 10.000 fitur sublinear TF."
             ])

    # -------------------------------------------------------------------------
    # SLIDE 7: Bab 4 - Pemodelan Machine Learning & Tuning
    # -------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Pemodelan Machine Learning & Hyperparameter Tuning (Bab 4.1–4.3)")
    add_card(s7, Inches(0.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Model 1: Baseline Sederhana",
             [
                 "Logistic Regression dengan pembobotan kelas seimbang (class_weight='balanced').",
                 "Tujuan: Menjadi standar performa minimal yang transparan dan cepat.",
                 "Parameter Terbaik: C=1.0, solver='liblinear'.",
                 "Keunggulan: Menghasilkan probabilitas terkalibrasi dan inferensi instan (< 5 ms)."
             ])
    add_card(s7, Inches(4.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Model 2: Alternatif Probabilistik",
             [
                 "Complement Naive Bayes (ComplementNB).",
                 "Dirancang khusus untuk korpus teks dengan distribusi kelas tidak seimbang.",
                 "Mengestimasi parameter dari komplemen tiap kelas untuk mengoreksi bias kelas mayoritas.",
                 "Parameter Terbaik: alpha=0.5, norm=True."
             ])
    add_card(s7, Inches(8.8), Inches(1.7), Inches(3.6), Inches(5.2),
             "Model 3: Alternatif Margin/Gradient",
             [
                 "Calibrated Linear Support Vector Machine (LinearSVC) & LightGBM.",
                 "Pemisahan hyperplane dengan margin maksimal berdimensi tinggi.",
                 "Kalibrasi probabilitas via CalibratedClassifierCV (Isotonic/Sigmoid).",
                 "Benchmark komparatif terhadap model pra-latih transformer IndoBERT."
             ])

    # -------------------------------------------------------------------------
    # SLIDE 8: Bab 5 - Evaluasi Komparatif & Pencapaian Target
    # -------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Hasil Evaluasi & Pembuktian Target Metrik (Bab 5.1–5.2 & 5.4)")
    add_card(s8, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2),
             "Tabel Komparasi Model (Data Uji 14.245 Ulasan)",
             [
                 "Dummy Baseline: Macro F1 = 0.2776 | ROC-AUC = 0.5000",
                 "ComplementNB: Macro F1 = 0.6124 | ROC-AUC = 0.8142",
                 "Tuned Logistic Regression: Macro F1 = 0.6382 | ROC-AUC = 0.8656 (Model Produksi)",
                 "Calibrated LinearSVC: Macro F1 = 0.6450 | ROC-AUC = 0.8680",
                 "Evaluasi Target Bab 1.4: Macro F1 >= 0.63 (TERCAPAI) & ROC-AUC >= 0.85 (TERCAPAI).",
                 "Peningkatan performa mencapai > +36 poin persentase di atas baseline tebakan acak."
             ])
    if os.path.exists(FIG_CM):
        s8.shapes.add_picture(FIG_CM, Inches(6.8), Inches(1.7), width=Inches(5.7))

    # -------------------------------------------------------------------------
    # SLIDE 9: Bab 5 - Explainable AI (SHAP) & Analisis Discrepancy
    # -------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Explainable AI (XAI) & Forensik Rating Discrepancy (Bab 5.3)")
    add_card(s9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2),
             "Atribusi Fitur & Forensik Kesalahan",
             [
                 "Kata Pendorong Positif Terkuat: 'membantu', 'keren', 'bagus', 'terbaik', 'pintar', 'cepat'.",
                 "Kata Pendorong Negatif Terkuat: 'error', 'limit', 'bayar', 'lambat', 'kecewa', 'login', 'rusak'.",
                 "Mengapa False Positive / False Negative Terjadi? Karena fenomena Rating-Text Discrepancy (4,59% data).",
                 "Contoh Nyata: Pengguna memberi Bintang 5 tapi menulis 'Tolong perbaiki server error terus'. Model memprediksi Negatif secara tepat berdasarkan isi kalimat!"
             ])
    if os.path.exists(FIG_SHAP):
        s9.shapes.add_picture(FIG_SHAP, Inches(6.8), Inches(1.7), width=Inches(5.7))

    # -------------------------------------------------------------------------
    # SLIDE 10: Topic Discovery (Unsupervised LDA - Member 3)
    # -------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Topic Discovery: Klaster Keluhan & Pujian Utama (Fase 4 & 5)")
    add_card(s10, Inches(0.8), Inches(1.7), Inches(4.6), Inches(5.2),
             "Klaster Topik LDA (Unsupervised Modeling)",
             [
                 "Topik 1 (Server & Jaringan): Masalah server sibuk/down saat jam puncak (DeepSeek).",
                 "Topik 2 (Filter & Sensor): Kebijakan sensor berlebihan & bot lupa ingatan (Character.AI).",
                 "Topik 3 (Limit Kuota & Biaya): Pembatasan kuota pesan harian & mahalnya paket Plus (ChatGPT/Claude).",
                 "Topik 4 (Bug Antarmuka): Layar hitam, crash, dan tombol macet pasca-update.",
                 "Topik 5 (Akurasi & Bahasa): Jawaban halusinasi dan terjemahan bahasa kaku."
             ])
    if os.path.exists(FIG_TOPIC):
        s10.shapes.add_picture(FIG_TOPIC, Inches(5.7), Inches(1.7), width=Inches(6.8))

    # -------------------------------------------------------------------------
    # SLIDE 11: Bab 6 - Prototipe Web Live & Uji 3 Skenario
    # -------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Deployment: Prototipe Streamlit & Uji 3 Skenario (Bab 6.1–6.5)")
    add_card(s11, Inches(0.8), Inches(1.7), Inches(4.6), Inches(5.2),
             "Arsitektur & Hasil Pengujian 3 Skenario",
             [
                 "URL Live: https://genai-indonesia-am552b.streamlit.app",
                 "Tab 1: Input ulasan tunggal instan + prediksi 4 aspek ABSA.",
                 "Tab 2: Batch processing file CSV (1.000 ulasan sekaligus).",
                 "Tab 3: Dashboard komparasi sentimen 8 aplikasi AI.",
                 "Skenario 1 (Normal): Sukses klasifikasi 100% akurat.",
                 "Skenario 2 (Ekstrem/Slang/Discrepancy): Berhasil menangkap anomali leksikal.",
                 "Skenario 3 (Salah Format/Kosong): Validasi input mencegah crash (100% andal)."
             ])
    if os.path.exists(FIG_SC1):
        s11.shapes.add_picture(FIG_SC1, Inches(5.7), Inches(1.7), width=Inches(6.8))

    # -------------------------------------------------------------------------
    # SLIDE 12: Bab 7 - Kesimpulan, Rekomendasi Manajerial & Q&A
    # -------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Kesimpulan, Rekomendasi Bisnis & Penutup (Bab 7.1–7.3)")
    add_card(s12, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.2),
             "Kesimpulan Menjawab Rumusan Masalah",
             [
                 "Rating agregat terbukti bias akibat positive skew (63,96%) dan discrepancy (4,59%).",
                 "Peta Sentimen Bersih (NSS): ChatGPT & Claude juara di Akurasi; DeepSeek juara di Kuota Gratis tapi anjlok di Server.",
                 "Character.AI rating 1.58 dipicu kekecewaan emosional filter kata dan amnesia bot pasca-update.",
                 "Model Machine Learning terbaik (Balanced LogReg) melampaui seluruh target KPI dosen (Macro F1 0.6382 & ROC-AUC 0.8656)."
             ])
    add_card(s12, Inches(6.8), Inches(1.7), Inches(5.6), Inches(5.2),
             "Rekomendasi Manajerial bagi Industri AI",
             [
                 "Pengembang AI Companion: Libatkan komunitas sebelum mengubah filter persona untuk mencegah review bombing.",
                 "Model Reasoning Gratis (DeepSeek): Prioritaskan server lokal Asia Tenggara untuk mengatasi server overload.",
                 "Transparansi Kuota: Berikan penghitung kuota jelas pada ChatGPT/Claude untuk meredam kekecewaan shock pricing.",
                 "Sesi Tanya Jawab (Q&A): Terima kasih! Repositori GitHub & Streamlit siap diuji."
             ])

    os.makedirs(os.path.dirname(OUT_PPTX_PATH), exist_ok=True)
    prs.save(OUT_PPTX_PATH)
    print(f"[SUCCESS] File Presentasi Resmi Tersimpan: {OUT_PPTX_PATH}")

if __name__ == "__main__":
    create_presentation()
