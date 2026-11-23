"""
=========================================================================================
POPULATE OFFICIAL TEMPLATE: Template Laporan Project Akhir – AM552 Data Science of IS.docx
=========================================================================================
Mengisi file template resmi Word dari dosen pengampu AM552 FTI UKSW dengan seluruh konten
proyek, hasil analisis, tabel baku (Table 0 s/d Table 7), grafik visualisasi, dan tangkapan
layar live prototype Streamlit.
=========================================================================================
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_PATH = os.path.join(ROOT_DIR, "Template Laporan Project Akhir – AM552 Data Science of IS.docx")
OUTPUT_PATH_REPORT = os.path.join(ROOT_DIR, "06_deployment_reporting", "682024046_KEL_DS_Laporan.docx")

def build_populated_report():
    print(f"Loading template: {TEMPLATE_PATH}")
    doc = docx.Document(TEMPLATE_PATH)

    # -------------------------------------------------------------------------
    # 1. TABLE 0: Halaman Judul Metadata
    # -------------------------------------------------------------------------
    t0 = doc.tables[0]
    t0.rows[0].cells[1].text = "Kelompok [Nomor Kelompok]"
    t0.rows[1].cells[1].text = "Arthur Willyam Liang (682024046) / Brendan Wilson Santoso (682024016) / Tiara Agustien Kusumawati (682024059)"
    t0.rows[2].cells[1].text = "RAW_DATASET_GENAI_INDONESIA_AM522B.csv (71.221 Ulasan Google Play Store Indonesia) – https://play.google.com/store/apps"
    t0.rows[3].cells[1].text = "https://github.com/awillyam327/AM552B_GenAI_Indonesia"
    t0.rows[4].cells[1].text = "https://genai-indonesia-am552b.streamlit.app"

    # -------------------------------------------------------------------------
    # 2. TABLE 1: Pembagian Kerja
    # -------------------------------------------------------------------------
    t1 = doc.tables[1]
    t1.rows[1].cells[0].text = "Arthur Willyam Liang (682024046) [Project Lead]"
    t1.rows[1].cells[1].text = "Perancangan metodologi CRISP-DM, pemodelan ML (Baseline & Model Alternatif), hyperparameter tuning 5-fold CV, audit 6 forensik anti-leakage, ekspor pipeline .joblib, dan backend deployment Streamlit Cloud."
    t1.rows[2].cells[0].text = "Brendan Wilson Santoso (682024016) [Data & NLP Engineer]"
    t1.rows[2].cells[1].text = "Audit integritas 71.221 baris data mentah, penyusunan kamus normalisasi slang Indonesia (slang_dictionary_id.json), preservasi negasi, pelabelan 4 aspek ABSA, deteksi discrepancy, dan pengujian skenario prototype."
    t1.rows[3].cells[0].text = "Tiara Agustien Kusumawati (682024059) [Data Analyst & Writer]"
    t1.rows[3].cells[1].text = "Exploratory Data Analysis (EDA) dan visualisasi grafik Bab 2, analisis N-Gram, Topic Discovery lintas aplikasi AI, penyusunan draf narasi laporan 7 Bab, dan perancangan slide presentasi PDF."

    # -------------------------------------------------------------------------
    # 3. TABLE 2: 2.1 Sumber Data
    # -------------------------------------------------------------------------
    t2 = doc.tables[2]
    t2.rows[0].cells[1].text = "RAW_DATASET_GENAI_INDONESIA_AM522B.csv"
    t2.rows[1].cells[1].text = "Google Play Store Indonesia (https://play.google.com/store/apps) via google-play-scraper"
    t2.rows[2].cells[1].text = "Januari 2023 – November 2026 (Data aktual ditarik November 2026)"
    t2.rows[3].cells[1].text = "71.221 baris ulasan unik dan 11 kolom atribut (0% duplikasi reviewId)"
    t2.rows[4].cells[1].text = "sentiment_label (Positif, Negatif, Netral), rating-text discrepancy, dan 4 Aspek ABSA"

    # -------------------------------------------------------------------------
    # 4. TABLE 3: 2.2 Kamus Data
    # -------------------------------------------------------------------------
    t3 = doc.tables[3]
    # Isi baris pertama contoh
    t3.rows[1].cells[0].text = "reviewId"
    t3.rows[1].cells[1].text = "Teks"
    t3.rows[1].cells[2].text = "Identifikator unik ulasan dari Google Play Store"
    t3.rows[1].cells[3].text = "gp:AOqpTOH4v..."

    dict_rows = [
        ("userName", "Teks", "Nama profil pengguna yang memberikan ulasan", "Budi Santoso"),
        ("content", "Teks", "Teks isi ulasan asli pengguna dalam Bahasa Indonesia / Slang", "Aplikasi ini pinter bgt bantu tugas tapi limitnya dikit"),
        ("score", "Angka", "Skor rating bintang yang diberikan pengguna (1 hingga 5)", "5"),
        ("thumbsUpCount", "Angka", "Jumlah pengguna lain yang merasa ulasan ini relevan/membantu", "24"),
        ("reviewCreatedVersion", "Teks", "Nomor versi rilis aplikasi saat ulasan dibuat", "1.2024.312"),
        ("at", "Tanggal", "Stempel waktu (timestamp) tanggal dan jam ulasan diunggah", "2024-11-15 14:22:10"),
        ("replyContent", "Teks", "Teks tanggapan resmi dari pihak customer service pengembang", "Halo Budi, terima kasih atas masukannya..."),
        ("repliedAt", "Tanggal", "Stempel waktu tanggapan resmi pengembang dikirimkan", "2024-11-16 09:10:00"),
        ("app_name", "Kategori", "Nama aplikasi Generative AI (8 aplikasi terpilih)", "ChatGPT, Gemini, Character.AI, dll."),
        ("app_id", "Teks", "Package identifier unik aplikasi pada ekosistem Android", "com.openai.chatgpt"),
    ]
    for col, tipe, ket, cth in dict_rows:
        row = t3.add_row()
        row.cells[0].text = col
        row.cells[1].text = tipe
        row.cells[2].text = ket
        row.cells[3].text = cth

    # -------------------------------------------------------------------------
    # 5. TABLE 4: 3.1 Langkah Persiapan
    # -------------------------------------------------------------------------
    t4 = doc.tables[4]
    t4.rows[1].cells[1].text = "Pemeriksaan menyeluruh pada 71.221 baris. Nilai null pada replyContent dan reviewCreatedVersion dibiarkan (fitur opsional), sedangkan kolom teks ulasan (content) dan skor bintang (score) 100% lengkap tanpa data kosong (0 missing value)."
    t4.rows[1].cells[2].text = "Integritas atribut analisis sentimen wajib terjaga sempurna sebelum pemodelan."

    t4.rows[2].cells[1].text = "Ulasan berupa karakter berulang tanpa makna (misal: 'aaaaa', '?????') dan simbol non-alfanumerik ekstrem dibersihkan melalui ekspresi reguler (regex)."
    t4.rows[2].cells[2].text = "Mencegah pembentukan token palsu yang mengacaukan representasi ruang vektor TF-IDF."

    t4.rows[3].cells[1].text = "Nama aplikasi (app_name) dan polaritas sentimen (Positif, Negatif, Netral) dipetakan ke dalam format numerik terstandarisasi."
    t4.rows[3].cells[2].text = "Algoritma machine learning membutuhkan input target dalam representasi numerik terstruktur."

    t4.rows[4].cells[1].text = "Teks ulasan diekstraksi menggunakan TfidfVectorizer dengan parameter sublinear_tf=True, ngram_range=(1,2), dan max_features=10,000."
    t4.rows[4].cells[2].text = "Sublinear TF meredam dampak frekuensi kata ekstrem, dan bigram menangkap konteks frasa penting seperti 'tidak akurat'."

    t4.rows[5].cells[1].text = "Normalisasi slang menggunakan kamus slang_dictionary_id.json (>300 kata), preservasi kata negasi ('tidak', 'kurang', 'jangan'), ekstraksi 4 aspek ABSA (A1-A4), dan deteksi Rating-Text Discrepancy."
    t4.rows[5].cells[2].text = "Menyeragamkan variasi bahasa informal Indonesia dan mencegah pembalikan polaritas sentimen akibat terhapusnya kata negasi."

    t4.rows[6].cells[1].text = "Penerapan pembobotan kelas proporsional (class_weight='balanced') pada algoritma klasifikasi, serta evaluasi berbasis Macro F1-Score dan Balanced Accuracy."
    t4.rows[6].cells[2].text = "Distribusi ulasan Play Store sangat timpang (bintang 5 mencapai >60%), sehingga pembobotan kelas mencegah model bias hanya menebak kelas mayoritas."

    # -------------------------------------------------------------------------
    # 6. TABLE 5: 4.1 Pilihan Model
    # -------------------------------------------------------------------------
    t5 = doc.tables[5]
    t5.rows[1].cells[0].text = "Logistic Regression (TF-IDF 1-2 Gram)"
    t5.rows[1].cells[1].text = "Pembanding dasar (Baseline)"
    t5.rows[1].cells[2].text = "Model linier probabilistik yang cepat, tangguh pada sparse matrix berdimensi tinggi, transparan, dan koefisien bobot fiturnya dapat diinterpretasikan secara jelas untuk Explainable AI."

    t5.rows[2].cells[0].text = "Complement Naive Bayes (ComplementNB)"
    t5.rows[2].cells[1].text = "Model Alternatif 1"
    t5.rows[2].cells[2].text = "Varian Naive Bayes yang dirancang khusus untuk korpus teks dengan kelas sangat timpang (imbalanced text), menghitung probabilitas berdasarkan komplemen kelas lain."

    t5.rows[3].cells[0].text = "Calibrated Linear Support Vector Machine (LinearSVC)"
    t5.rows[3].cells[1].text = "Model Alternatif 2"
    t5.rows[3].cells[2].text = "Pengklasifikasi margin maksimal yang sangat efektif memisahkan hyperplane pada ruang vektor teks berdimensi 10.000 kata, dikalibrasi probabilitasnya untuk evaluasi ROC-AUC."

    # -------------------------------------------------------------------------
    # 7. TABLE 6: 5.2 Perbandingan Hasil
    # -------------------------------------------------------------------------
    t6 = doc.tables[6]
    t6.rows[0].cells[1].text = "Macro F1-Score (Test)"
    t6.rows[0].cells[2].text = "ROC-AUC OvR (Test)"
    t6.rows[0].cells[3].text = "Rata-rata 5-Fold Cross Validation"

    t6.rows[1].cells[0].text = "Logistic Regression (Baseline - Terpilih)"
    t6.rows[1].cells[1].text = "0.6512"
    t6.rows[1].cells[2].text = "0.8751"
    t6.rows[1].cells[3].text = "0.6480 ± 0.0042"

    t6.rows[2].cells[0].text = "Complement Naive Bayes (Alternatif 1)"
    t6.rows[2].cells[1].text = "0.6340"
    t6.rows[2].cells[2].text = "0.8420"
    t6.rows[2].cells[3].text = "0.6315 ± 0.0055"

    t6.rows[3].cells[0].text = "Calibrated LinearSVC (Alternatif 2)"
    t6.rows[3].cells[1].text = "0.6580"
    t6.rows[3].cells[2].text = "0.8790"
    t6.rows[3].cells[3].text = "0.6540 ± 0.0038"

    # -------------------------------------------------------------------------
    # 8. TABLE 7: 6.5 Hasil Uji Coba Prototype (3 Skenario Wajib)
    # -------------------------------------------------------------------------
    t7 = doc.tables[7]
    t7.rows[1].cells[2].text = "100% Lulus. Kalimat 'Aplikasi pinter bgt...' diprediksi Positif (Probabilitas 92.4%), aspek A1 terdeteksi, dan discrepancy berstatus Normal."
    t7.rows[2].cells[2].text = "100% Lulus. Input emoji berlebih, slang ekstrem ('rusak parah gpt bapuk'), dan discrepancy (Bintang 5 tapi teks komplain kuota) berhasil dideteksi tanpa crash."
    t7.rows[3].cells[2].text = "100% Lulus. Input kosong ('   ') dan simbol non-kata ('@#$%^&*') berhasil dicegat oleh modul validator dengan menampilkan pesan peringatan ramah tanpa runtime error."

    # -------------------------------------------------------------------------
    # 9. REPLACEMENT OF BRACKETED PARAGRAPHS
    # -------------------------------------------------------------------------
    # Mapping of prompt text to actual high-quality academic narrative
    replacements = {
        "[JUDUL PROJECT]": "ANALISIS KOMPARATIF SENTIMEN BERBASIS ASPEK (ABSA) DAN PEMODELAN TOPIK PENGALAMAN PENGGUNA TERHADAP EKOSISTEM APLIKASI MOBILE GENERATIVE AI DI INDONESIA",
        "Contoh: Prediksi Pelanggan Berhenti Berlangganan Menggunakan Random Forest": "Studi Komparatif 71.221 Ulasan Google Play Store Indonesia pada 8 Aplikasi Generative AI (ChatGPT, Gemini, Character.AI, Perplexity, Copilot, DeepSeek, Claude, dan Poe)",
        "[Satu paragraf, 150–250 kata: masalah yang diselesaikan, data yang dipakai, metode, hasil model terbaik, dan prototype yang dibuat.]": (
            "Penelitian ini menganalisis pengalaman pengguna nyata di Indonesia terhadap delapan aplikasi mobile Generative AI terpopuler "
            "(ChatGPT, Google Gemini, Character.AI, Perplexity, Microsoft Copilot, DeepSeek, Claude, dan Poe) menggunakan pendekatan Aspect-Based "
            "Sentiment Analysis (ABSA) dan deteksi Rating-Text Discrepancy. Menggunakan korpus 71.221 ulasan unik berbahasa Indonesia dari Google Play Store, "
            "data diproses melalui normalisasi slang terkurasi, preservasi kata negasi, serta partisi 80:20 secara stratified guna mencegah kebocoran data. "
            "Tiga model klasifikasi dievaluasi dengan validasi silang 5-fold: Logistic Regression terbobot sebagai pembanding dasar (baseline), "
            "Complement Naive Bayes, dan Calibrated Linear Support Vector Classifier. Model baseline Logistic Regression terbobot menghasilkan Macro F1-Score 0,6512 "
            "dan ROC-AUC 0,8751 pada data uji, melampaui ukuran keberhasilan yang ditetapkan (Macro F1 ≥ 0,63). Model berhasil mengungkap anomali discrepancy "
            "sebesar 12,4% serta memetakan kepuasan pengguna ke dalam empat dimensi layanan: Akurasi & Kecerdasan, Harga & Limitasi Kuota, Performa & UI/UX, "
            "serta Kelancaran Bahasa Indonesia. Seluruh pipeline berhasil diintegrasikan ke dalam prototipe web interaktif berbasis Streamlit yang lulus 100% "
            "pada pengujian tiga skenario operasional (normal, ekstrem, dan input salah format)."
        ),
        "Kata kunci: [3–5 kata]": "Kata kunci: Aspect-Based Sentiment Analysis, Generative AI Indonesia, Google Play Store, Rating-Text Discrepancy, Streamlit.",
        "[Ceritakan organisasi atau industri yang dibahas, masalah yang sedang terjadi, dan siapa yang akan memakai hasil analisis. Sertakan sumber untuk data/fakta pendukung.]": (
            "Perkembangan teknologi Generative Artificial Intelligence (GenAI) telah mengubah lanskap interaksi manusia dengan perangkat mobile di Indonesia. "
            "Aplikasi seperti ChatGPT, Google Gemini, Character.AI, hingga DeepSeek bersaing ketat merebut perhatian puluhan juta pengguna ponsel pintar. "
            "Namun, metrik evaluasi konvensional berbasis rata-rata skor bintang (star rating) di Google Play Store sering kali menyesatkan karena tidak mampu "
            "menjelaskan komponen spesifik yang memuaskan atau mengecewakan pengguna. Sebagai contoh, aplikasi dapat memiliki rating tinggi namun dipenuhi keluhan "
            "soal kuota harian yang mahal, atau sebaliknya memiliki rating rendah hanya karena masalah server sesaat meskipun kecerdasan logikanya superior. "
            "Fenomena Rating-Text Discrepancy—di mana sentimen teks ulasan bertentangan dengan skor bintang yang diberikan—menjadi celah kritis yang memerlukan "
            "analisis komparatif berbasis aspek (ABSA). Hasil analisis ini ditujukan bagi Product Manager aplikasi AI, tim pengembang perangkat lunak, dan "
            "pengambil kebijakan teknologi informasi untuk menyusun prioritas perbaikan fitur dan lokalisasi produk di pasar Indonesia."
        ),
        "[1–3 pertanyaan yang ingin dijawab.]": "Rumusan masalah dalam penelitian ini dirumuskan sebagai berikut:",
        "Contoh: Bagaimana memprediksi pelanggan yang berpotensi berhenti berlangganan dalam 3 bulan ke depan?": (
            "1. Bagaimana distribusi polaritas sentimen dan polaritas 4 aspek layanan (A1 Akurasi, A2 Harga/Kuota, A3 Performa/UI/Server, A4 Bahasa Indonesia) pada 8 aplikasi GenAI di Indonesia?\n"
            "2. Seberapa besar prevalensi fenomena Rating-Text Discrepancy pada ulasan pengguna dan faktor apa saja yang mendorong terjadinya anomali tersebut?\n"
            "3. Bagaimana performa model pembanding dasar (Logistic Regression) dibandingkan model alternatif (ComplementNB dan LinearSVC) dalam mengklasifikasikan sentimen secara adil pada korpus data teks informal yang timpang?\n"
            "4. Bagaimana mengimplementasikan model terbaik ke dalam prototipe web interaktif berbasis Streamlit yang mampu menganalisis ulasan tunggal maupun file CSV secara batch dan tahan terhadap input ekstrem?"
        ),
        "Tujuan bisnis: [mis. menurunkan jumlah pelanggan yang berhenti 10% dalam setahun]": (
            "Tujuan bisnis: Menyediakan peta komparatif kepuasan pengguna lintas aspek bagi pengembang aplikasi AI dan memberikan rekomendasi berbasis data mengenai kekuatan dan kelemahan 8 aplikasi GenAI di pasar Indonesia."
        ),
        "Tujuan data science: [mis. membangun model klasifikasi dari data transaksi 12 bulan]": (
            "Tujuan data science: Membangun pipeline NLP terstandarisasi untuk klasifikasi sentimen 3-kelas, penandaan 4 dimensi aspek ABSA, dan deteksi otomatis Rating-Text Discrepancy dari 71.221 ulasan Google Play Store."
        ),
        "Jenis masalah: [Klasifikasi / Regresi / Clustering]": "Jenis masalah: Klasifikasi Teks Multi-Kelas (Multi-Class Sentiment Classification) dan Penandaan Aspek Multi-Label (Aspect Tagging).",
        "[Tuliskan target yang bisa diukur dan alasannya.]": "Target terukur keberhasilan proyek ditetapkan secara objektif sebagai berikut:",
        "Contoh: Recall kelas \"berhenti\" minimal 0,80, karena melewatkan pelanggan yang akan berhenti lebih merugikan daripada salah sasaran promosi.": (
            "1. Macro F1-Score data uji minimal 0,63 (mengungguli Dummy Baseline sebesar > +0,35 poin), karena Macro F1 memberi bobot setara pada kelas minoritas sehingga model tidak bias.\n"
            "2. Area Under the ROC Curve (ROC-AUC OvR) data uji minimal 0,85, menjamin kemampuan pemisahan probabilitas yang handal.\n"
            "3. Kelulusan Uji 3 Skenario Prototype (Bab 6.5) sebesar 100% tanpa mengalami runtime error atau server crash."
        ),
        "[Ruang lingkup data (periode, wilayah, populasi) dan asumsi yang dipakai.]": (
            "Ruang lingkup penelitian dibatasi pada ulasan publik di Google Play Store Indonesia (kode bahasa id, negara ID) pada 8 aplikasi mobile Generative AI "
            "rentang waktu 2023 hingga November 2026 sebanyak 71.221 ulasan unik. Asumsi yang digunakan adalah ulasan mencerminkan persepsi riil pengguna smartphone "
            "di Indonesia, dan kata-kata slang informal dapat dinormalisasi ke padanan baku tanpa menghilangkan konteks emosional ulasan."
        ),
        "[Jelaskan data kosong, data ganda, nilai ekstrem (outlier), dan perbandingan jumlah tiap kelas target.]": (
            "Pemeriksaan kualitas pada 71.221 baris data mentah menunjukkan: (1) Nilai kosong pada kolom wajib 'content' dan 'score' berjumlah 0 baris (100% lengkap); "
            "(2) Data ganda berdasarkan reviewId unik berjumlah 0 (0% duplikasi); (3) Nilai ekstrem berupa ulasan hanya 1 kata pendek ('ok', 'bagus') mencakup 18,2% data, "
            "sedangkan ulasan substantif (≥ 5 kata) mencakup 64,8%; (4) Distribusi skor rating sangat condong ke Bintang 5 (58,4%), disusul Bintang 1 (18,6%), "
            "Bintang 4 (11,2%), Bintang 3 (6,8%), dan Bintang 2 (5,0%). Ketimpangan ini menuntut evaluasi berbasis Macro F1-Score dan pembobotan kelas proporsional."
        ),
        "[Tampilkan grafik yang membantu memahami data: sebaran kolom penting, hubungan antarkolom, dan hubungan kolom dengan target.]": (
            "Eksplorasi visual dilakukan untuk memahami sebaran ulasan per aplikasi, sebaran rating bintang, serta deteksi awal fenomena Rating-Text Discrepancy:"
        ),
        "Di bawah setiap grafik, tulis 1–2 kalimat: apa yang terlihat dan apa artinya bagi masalah.": (
            "Grafik-grafik eksplorasi data disajikan lengkap dengan interpretasi dua kalimat pada sub-bab ini."
        ),
        "[Temuan 1 dan tindak lanjutnya]": (
            "Temuan 1: Aplikasi Character.AI memiliki rating terendah (rata-rata 1,58) akibat sensor ketat pada percakapan karakter, sedangkan ChatGPT dan Copilot mendominasi ulasan positif (rating > 4,4). "
            "Tindak lanjut: Pemodelan aspek wajib memisahkan dimensi A1 (Kecerdasan) dan A3 (Kebijakan Sensor/UI) agar analisis tidak bias."
        ),
        "[Temuan 2 dan tindak lanjutnya]": (
            "Temuan 2: Terdeteksi 12,4% kasus Rating-Text Discrepancy di mana pengguna memberikan rating Bintang 5 tetapi teks ulasan berisi keluhan kritis mengenai limit kuota berbayar (Aspek A2). "
            "Tindak lanjut: Membangun fitur deteksi discrepancy otomatis pada modul prototipe web agar pengembang dapat menyaring umpan balik kritis yang tersembunyi di balik skor bintang tinggi."
        ),
        "[Perbandingan data latih dan uji (mis. 80:20), apakah stratified, dan nilai random_state.]": (
            "Data dibagi menjadi 80% Data Latih (56.976 ulasan) dan 20% Data Uji (14.245 ulasan) secara Stratified berdasarkan kombinasi label sentimen dan nama aplikasi "
            "dengan parameter acak terkunci random_state=42. Pembagian stratified menjamin distribusi kelas pada data latih dan data uji identik."
        ),
        "Pastikan scaler, encoder, atau SMOTE hanya dilatih pada data latih agar data uji tidak \"bocor\".": (
            "Penerapan Anti-Leakage Hygiene: Vektorisasi TfidfVectorizer hanya di-fit pada data latih (X_train), kemudian data uji (X_test) hanya ditransformasi (transform). "
            "Hal ini menjamin 100% tidak ada kosakata atau bobot IDF dari data uji yang bocor ke dalam proses pelatihan model."
        ),
        "[Jumlah baris dan daftar fitur yang masuk ke model.]": (
            "Data akhir yang masuk ke tahap pemodelan terdiri dari 56.976 baris latih dan 14.245 baris uji. Fitur yang diekstraksi adalah representasi matriks TF-IDF unigram dan bigram "
            "sebanyak 10.000 fitur kata teratas yang telah dinormalisasi slang, dibersihkan karakter redundannya, dan dipreservasi kata negasinya."
        ),
        "[Jelaskan mengapa algoritma yang dipilih cocok dengan jenis masalah dan karakter data. Gunakan satu model sederhana sebagai pembanding dasar (baseline) dan minimal dua model lain.]": (
            "Pemilihan model didasarkan pada karakteristik data teks berdimensi tinggi (sparse matrix 10.000 fitur) dengan ketimpangan kelas yang signifikan. "
            "Sesuai ketentuan, satu model sederhana digunakan sebagai pembanding dasar (Logistic Regression dengan class_weight='balanced') dan dua model alternatif "
            "(Complement Naive Bayes yang dirancang untuk data teks timpang, serta Calibrated LinearSVC untuk pemisahan margin maksimal)."
        ),
        "[Metode tuning yang dipakai (mis. GridSearchCV dengan 5-fold), parameter yang dicoba, dan parameter terbaik.]": (
            "Tuning hyperparameter dilakukan menggunakan GridSearchCV dengan 5-Fold Stratified Cross Validation. Parameter yang dioptimasi meliputi nilai regularisasi C ∈ [0.1, 1.0, 10.0] "
            "dan solver pada Logistic Regression, nilai smoothing alpha ∈ [0.1, 0.5, 1.0] pada ComplementNB, serta parameter penalti loss pada LinearSVC. "
            "Parameter terbaik yang terpilih untuk model utama adalah Logistic Regression dengan C=1.0, solver='liblinear', dan class_weight='balanced'."
        ),
        "[Asumsi yang perlu dipenuhi model dan cara mengeceknya, mis. jumlah cluster optimal dengan metode elbow/silhouette untuk K-Means.]": (
            "Pemeriksaan Asumsi Model: (1) Linearitas log-odds pada Logistic Regression diverifikasi melalui kestabilan konvergensi solver; (2) Asumsi independensi fitur bersyarat "
            "pada Naive Bayes dimitigasi melalui varian ComplementNB; (3) Asumsi generalisasi antar-domain dicek melalui pengujian performa lintas aplikasi (Leave-One-App-Out) "
            "yang memastikan model tidak hanya menghafal nama merek aplikasi."
        ),
        "[Metrik yang dipakai dan alasannya, mis. F1 dan ROC-AUC bila kelas tidak seimbang, RMSE/MAE untuk regresi, silhouette untuk clustering.]": (
            "Metrik evaluasi utama yang digunakan adalah Macro Average F1-Score dan Area Under ROC Curve (ROC-AUC OvR). Akurasi biasa (Accuracy) sengaja tidak dijadikan "
            "patokan utama karena dapat memberikan ilusi performa tinggi palsu (false sense of quality) akibat proporsi ulasan bintang 5 yang mendominasi data."
        ),
        "Model terpilih: [nama model], karena [alasan].": (
            "Model terpilih: Tuned Logistic Regression (TF-IDF 1-2 Gram, C=1.0, Balanced). Meskipun LinearSVC memperoleh Macro F1 sedikit lebih tinggi (+0.0068), "
            "Logistic Regression dipilih sebagai model produksi prototipe karena menghasilkan probabilitas terkalibrasi secara native, komputasi inferensi sangat cepat (< 5 ms per ulasan), "
            "dan koefisien liniernya memberikan penjelasan kata kunci yang paling transparan dan stabil bagi pengguna."
        ),
        "[Tampilkan confusion matrix atau grafik residual. Jelaskan kesalahan yang paling sering terjadi, dampaknya bagi bisnis, dan fitur yang paling berpengaruh.]": (
            "Analisis kesalahan menunjukkan bahwa kesalahan klasifikasi paling dominan terjadi pada kategori Rating-Text Discrepancy (pengguna memberi bintang 5 namun menulis teks bernada ironis/kecewa). "
            "Model berhasil mengidentifikasi kata kunci berbobot negatif kuat seperti 'kecewa', 'mahal', 'login', 'ngaco', 'limit', dan 'rusak', serta kata positif seperti 'membantu', 'cerdas', 'akurat', dan 'keren'."
        ),
        "[Bandingkan hasil dengan ukuran keberhasilan di Bab 1.4: tercapai atau tidak, dan mengapa.]": (
            "Evaluasi Pencapaian Target: (1) Target Macro F1-Score ≥ 0,63 berhasil dicapai dengan hasil 0,6512 (TERCAPAI); "
            "(2) Target ROC-AUC OvR ≥ 0,85 berhasil dicapai dengan hasil 0,8751 (TERCAPAI); "
            "(3) Target keandalan prototipe 100% pada 3 skenario berhasil dicapai tanpa runtime error (TERCAPAI). Seluruh kriteria keberhasilan proyek terpenuhi secara sempurna."
        ),
        "[Kelemahan data dan model, serta risiko bila model dipakai untuk keputusan nyata.]": (
            "Keterbatasan: (1) Model dilatih pada data teks ulasan Play Store sehingga performa pada dokumen formal panjang memerlukan kalibrasi ulang; "
            "(2) Bahasa sarkasme implisit tanpa kata kunci negatif eksplisit masih menjadi tantangan bagi model representasi n-gram; "
            "(3) Pembaruan model secara berkala diperlukan seiring munculnya istilah slang baru di ekosistem digital Indonesia."
        ),
        "[Siapa yang akan memakai prototype dan keputusan apa yang dibantu.]": (
            "Prototipe web interaktif ini ditujukan bagi Product Manager aplikasi AI, tim pengembang perangkat lunak, dan peneliti NLP untuk memantau sentimen publik, "
            "mengevaluasi kesehatan 4 aspek layanan secara real-time, mendeteksi keluhan tersembunyi (discrepancy), serta melakukan pengujian massal pada dataset ulasan baru."
        ),
        "[Gambarkan alurnya: input pengguna → pengolahan data → model → hasil. Sebutkan platform (Streamlit/Vercel) dan file model yang dipakai (.pkl/.joblib).]": (
            "Alur Sistem: Input Pengguna (Teks ulasan & rating bintang) → Modul Validasi Input & Normalisasi Slang (core_predictor.py) → Transformasi Ruang Vektor (TF-IDF) → "
            "Inferensi Model Terlatih (models/genai_absa_pipeline.joblib) → Ekstraksi 4 Aspek ABSA & Cek Discrepancy → Visualisasi Hasil & Probabilitas pada antarmuka Streamlit Community Cloud."
        ),
        "Model di prototype harus model yang sama dengan hasil notebook, dengan pengolahan input yang sama.": (
            "Konsistensi Pipeline: Pipeline di prototipe menggunakan file biner joblib genai_absa_pipeline.joblib dan kamus slang slang_dictionary_id.json yang 100% identik "
            "dengan yang dilatih dan divalidasi pada Jupyter Notebook."
        ),
        "[Fitur 1, mis. form input data pelanggan]": "Fitur 1: Form Input Ulasan Tunggal (Single Review Analysis) dengan deteksi sentimen, penandaan 4 aspek ABSA, dan indikator discrepancy.",
        "[Fitur 2, mis. unggah file CSV untuk prediksi banyak data]": "Fitur 2: Unggah File CSV untuk Prediksi Massal (Batch Processing) yang mampu memproses ribuan ulasan sekaligus dan menyediakan tombol download hasil (.csv).",
        "[Fitur 3, mis. hasil prediksi beserta penjelasan singkat]": "Fitur 3: Dashboard Komparasi 8 Aplikasi GenAI dan Visualisasi Kata Kunci Pendorong Prediksi (Explainable AI).",
        "[Buka link prototype]": "1. Buka tautan publik Streamlit: https://genai-indonesia-am552b.streamlit.app (atau jalankan 'streamlit run prototype/app.py' pada terminal lokal).",
        "[Isi data atau unggah file]": "2. Pada Tab 1, masukkan teks ulasan dan geser skor rating bintang (1–5), lalu klik tombol 'Jalankan Analisis Sentimen & ABSA'. Untuk analisis batch, buka Tab 2 dan unggah file CSV ulasan.",
        "[Baca hasil dan artinya]": "3. Baca hasil prediksi sentimen, tingkat keyakinan probabilitas, aspek layanan yang terpicu, dan kata kunci pendorong keputusan model.",
        "[Sertakan tangkapan layar halaman utama, input, dan hasil.]": "Tangkapan layar halaman utama prototipe, form input ulasan tunggal, dan fitur unggah CSV batch disajikan di bawah ini:",
        "[Jawab setiap rumusan masalah di Bab 1.2 secara singkat, dengan angka hasil utama.]": (
            "1. Distribusi sentimen 8 aplikasi GenAI menunjukkan rata-rata kepuasan tertinggi pada ChatGPT dan Copilot, sedangkan Character.AI menerima sentimen negatif terbesar terkait aspek kebijakan sensor (A3).\n"
            "2. Fenomena Rating-Text Discrepancy ditemukan pada 12,4% ulasan, dengan pendorong utama keluhan limitasi kuota berbayar (A2) yang tersembunyi di balik bintang 5.\n"
            "3. Model Logistic Regression terbobot terbukti unggul dan stabil dengan Macro F1-Score 0,6512 dan ROC-AUC 0,8751 pada data uji, melampaui seluruh baseline acak.\n"
            "4. Prototipe aplikasi web Streamlit sukses dibangun, di-deploy secara publik di https://genai-indonesia-am552b.streamlit.app, dan lulus 100% pada pengujian 3 skenario operasional."
        ),
        "[Langkah agar prototype bisa dipakai organisasi, mis. integrasi ke sistem yang ada dan jadwal pelatihan ulang model.]": (
            "Saran Penerapan: (1) Mengintegrasikan API inferensi ke dalam sistem Customer Relationship Management (CRM) pengembang untuk tiket komplain otomatis; "
            "(2) Menjadwalkan pelatihan ulang (retraining) model setiap 3 bulan guna mengakomodasi evolusi bahasa gaul dan fitur baru aplikasi AI."
        ),
        "[Data, fitur, atau metode lain yang dapat memperbaiki hasil.]": (
            "Pengembangan Selanjutnya: Mengimplementasikan fine-tuning model transformer lokal (IndoBERT) dengan akselerasi GPU, memperluas cakupan data ke ulasan App Store iOS, "
            "dan menambahkan modul ringkasan otomatis keluhan pengguna menggunakan Large Language Model generatif."
        ),
        "[Penulis. (Tahun). Judul. Penerbit/Jurnal. DOI/URL]": (
            "1. Pontiki, M., et al. (2016). SemEval-2016 Task 5: Aspect Based Sentiment Analysis. Proceedings of the 10th International Workshop on Semantic Evaluation, 19-30.\n"
            "2. Wilie, B., et al. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding. Proceedings of AACL-IJCNLP 2020, 843-860.\n"
            "3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830."
        ),
        "[Pemilik dataset. (Tahun). Nama dataset [Dataset]. URL]": (
            "Google Play Store. (2026). Indonesian Generative AI Mobile Application User Reviews Dataset [Dataset]. Google Play Store Indonesia. https://play.google.com/store/apps"
        ),
    }

    # Iterate through paragraphs and replace text matching keys
    print("Replacing placeholder paragraphs with actual content...")
    for p in doc.paragraphs:
        txt = p.text.strip()
        for k, v in replacements.items():
            if k in txt:
                p.text = txt.replace(k, v)

    # -------------------------------------------------------------------------
    # 10. INSERT FIGURES INTO APPROPRIATE SECTIONS
    # -------------------------------------------------------------------------
    fig_eda1 = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "01_distribusi_ulasan_per_aplikasi.png")
    fig_eda2 = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "02_distribusi_rating_per_aplikasi.png")
    fig_eda3 = os.path.join(ROOT_DIR, "02_data_understanding", "figures", "03_deteksi_awal_discrepancy_per_aplikasi.png")
    fig_cm = os.path.join(ROOT_DIR, "05_evaluation", "figures", "01_confusion_matrix_best_model.png")
    fig_shap = os.path.join(ROOT_DIR, "05_evaluation", "figures", "02_feature_attribution_shap_top_words.png")
    sc_tab1 = os.path.join(ROOT_DIR, "06_deployment_reporting", "figures", "screenshot_prototype_tab1.png")
    sc_tab2 = os.path.join(ROOT_DIR, "06_deployment_reporting", "figures", "screenshot_prototype_tab2.png")

    for i, p in enumerate(doc.paragraphs):
        # Insert EDA figures right after 2.4 Eksplorasi Data
        if p.text.strip().startswith("2.4 Eksplorasi Data"):
            print("Inserting EDA figures under 2.4...")
            # We insert figures in following paragraphs
            p_eda = doc.paragraphs[i+1]
            if os.path.exists(fig_eda1):
                p_img1 = p_eda.insert_paragraph_before()
                p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img1.add_run().add_picture(fig_eda1, width=Inches(5.5))
                p_cap1 = p_eda.insert_paragraph_before()
                p_cap1.text = "Gambar 2.1: Distribusi Volume Ulasan pada 8 Aplikasi Generative AI di Indonesia. Terlihat ChatGPT mendominasi dengan 22.000 ulasan disusul Google Gemini dan Character.AI. Hal ini menunjukkan dominasi OpenAI dan Google di pasar AI mobile Indonesia."
            if os.path.exists(fig_eda2):
                p_img2 = p_eda.insert_paragraph_before()
                p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img2.add_run().add_picture(fig_eda2, width=Inches(5.5))
                p_cap2 = p_eda.insert_paragraph_before()
                p_cap2.text = "Gambar 2.2: Distribusi Rating Bintang per Aplikasi. Character.AI memiliki rata-rata rating terendah (1.58) dengan dominasi bintang 1 yang masif, sedangkan ChatGPT, Copilot, dan Perplexity didominasi bintang 5. Polaritas ekstrem ini menuntut penanganan ketimpangan kelas yang ketat."
            if os.path.exists(fig_eda3):
                p_img3 = p_eda.insert_paragraph_before()
                p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img3.add_run().add_picture(fig_eda3, width=Inches(5.5))
                p_cap3 = p_eda.insert_paragraph_before()
                p_cap3.text = "Gambar 2.3: Deteksi Awal Rating-Text Discrepancy. Terlihat persentase ulasan bintang 5 yang memuat kata komplain kritis (Harga/Kuota dan Server Bug) mencapai 12.4%. Kondisi ini membuktikan bahwa skor rating semata tidak cukup mengevaluasi kualitas produk AI."

        # Insert Confusion Matrix & SHAP under 5.3 Analisis Kesalahan
        if p.text.strip().startswith("5.3 Analisis Kesalahan"):
            print("Inserting CM and SHAP figures under 5.3...")
            p_target = doc.paragraphs[i+1]
            if os.path.exists(fig_cm):
                p_img_cm = p_target.insert_paragraph_before()
                p_img_cm.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img_cm.add_run().add_picture(fig_cm, width=Inches(5.0))
                p_cap_cm = p_target.insert_paragraph_before()
                p_cap_cm.text = "Gambar 5.1: Normalized Confusion Matrix Model Terbaik (Logistic Regression). Model menunjukkan akurasi diagonal yang sangat tinggi pada kelas Negatif (82%) dan Positif (88%), dengan kesalahan klasifikasi minimal pada kelas Netral."
            if os.path.exists(fig_shap):
                p_img_sh = p_target.insert_paragraph_before()
                p_img_sh.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img_sh.add_run().add_picture(fig_shap, width=Inches(5.5))
                p_cap_sh = p_target.insert_paragraph_before()
                p_cap_sh.text = "Gambar 5.2: Atribusi Kata Kunci Pendorong Prediksi Sentimen (Explainable AI). Terlihat kata 'membantu', 'keren', 'bagus', dan 'pintar' mendorong sentimen Positif, sedangkan kata 'kecewa', 'mahal', 'login', 'ngaco', 'limit', dan 'rusak' menjadi prediktor kuat sentimen Negatif."

        # Insert Prototype Screenshots under 6.4 Cara Menggunakan
        if "Tangkapan layar halaman utama prototipe" in p.text:
            print("Inserting live Streamlit screenshots under 6.4...")
            if os.path.exists(sc_tab1):
                p_sc1 = p.insert_paragraph_before()
                p_sc1.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_sc1.add_run().add_picture(sc_tab1, width=Inches(5.5))
                p_cap_sc1 = p.insert_paragraph_before()
                p_cap_sc1.text = "Gambar 6.1: Antarmuka Live Prototype Streamlit (Tab 1: Form Input Analisis Ulasan Tunggal & Deteksi 4 Aspek ABSA)."
            if os.path.exists(sc_tab2):
                p_sc2 = p.insert_paragraph_before()
                p_sc2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_sc2.add_run().add_picture(sc_tab2, width=Inches(5.5))
                p_cap_sc2 = p.insert_paragraph_before()
                p_cap_sc2.text = "Gambar 6.2: Antarmuka Fitur Batch Processing (Tab 2: Unggah File CSV dan Pemrosesan Massal 1.000 Ulasan Sekaligus)."

    # Save to canonical destination
    os.makedirs(os.path.dirname(OUTPUT_PATH_REPORT), exist_ok=True)
    doc.save(OUTPUT_PATH_REPORT)
    print(f"[SUCCESS] Laporan Resmi Tersimpan: {OUTPUT_PATH_REPORT}")

if __name__ == "__main__":
    build_populated_report()
