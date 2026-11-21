# MASTER EXECUTION PLAN & JADWAL SPRINT 9 HARI (21 NOV – 30 NOV 2026)
## Proyek Akhir Mata Kuliah AM552 - Data Science of IS (S1 Sistem Informasi, FTI UKSW)

> **Judul Proyek**: *Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia*  
> **Program Studi / Fakultas**: S1 Sistem Informasi, Fakultas Teknologi Informasi (FTI) — Universitas Kristen Satya Wacana (UKSW), Semester Ganjil 2026/2027  
> **Deadline Pengumpulan FLearn**: **30 November 2026 Pukul 23.00 WIB**  
> **Format Penamaan File Resmi**: `NIM_KEL_DS` (Contoh: `682024046_KEL_DS`)  
> **Aset Dataset Utama**: `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv` (**71.221 baris ulasan unik** dari 8 aplikasi Generative AI di Google Play Store Indonesia, 100% *real scraped data*).

---

## 1. Ringkasan Kepatuhan terhadap Ketentuan Resmi AM552 (FTI UKSW)

Seluruh arsitektur proyek, kode program, eksperimen model, prototipe aplikasi, dan struktur laporan telah diselaraskan **100%** dengan dokumen ketentuan resmi dosen pengampu **AM552 - Data Science of IS**:

| No | Kategori Ketentuan Resmi UKSW | Spesifikasi Wajib dari Dosen | Implementasi Konkret di Workspace Kita | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Deliverable 1: File Kode (`.ipynb`)** | Wajib berbentuk **Jupyter Notebook (`.ipynb`)** yang rapi, terurut, dan bisa dijalankan ulang (*Restart & Run All*). | Tersedia 3 Notebook lengkap dengan *output* di folder `notebooks/`:<br>1. `01_Data_Understanding_dan_EDA.ipynb`<br>2. `02_Data_Preparation_dan_ABSA.ipynb`<br>3. `03_Modeling_Tuning_dan_Evaluasi.ipynb` | ✅ **Siap** |
| **2** | **Deliverable 2: File Laporan (`.docx`)** | Mengikuti struktur baku **7 Bab Template AM552 FTI UKSW** (Halaman Judul, Pembagian Kerja, Abstrak, Bab 1–7, Daftar Pustaka APA). | Tersedia di `06_deployment_reporting/682024046_KEL_DS_Laporan.docx` lengkap dengan tabel, gambar grafik + keterangan 2 kalimat, dan sitasi APA. | ✅ **Siap** |
| **3** | **Deliverable 3: File Presentasi (`.pdf`)** | Slide PowerPoint yang diekspor ke format **`.pdf`** dengan penamaan `NIM_KEL_DS`. | Struktur materi slide per bab disiapkan untuk diekspor ke `06_deployment_reporting/682024046_KEL_DS_Presentasi.pdf`. | ✅ **Terjadwal** |
| **4** | **Deliverable 4: Live Prototype Web App** | Di-deploy ke **Streamlit Community Cloud** menggunakan model tersimpan (`.joblib`/`.pkl`) dan pipeline preprocessing yang identik. | Aplikasi lengkap di `prototype/app.py` yang memuat model `models/genai_absa_pipeline.joblib` + panduan `prototype/README_DEPLOY.md`. | ✅ **Siap Deploy** |
| **5** | **Aturan Pemodelan (Bab 4)** | Wajib menggunakan **1 Model Baseline Sederhana** + **Minimal 2 Model Alternatif**, dokumentasi *Hyperparameter Tuning* 5-Fold, dan *Asumsi Model*. | - **Baseline**: `LogisticRegression`<br>- **Alternatif 1**: `ComplementNB` (Multinomial Naive Bayes untuk data timpang)<br>- **Alternatif 2**: `LinearSVC (Calibrated)` / `LightGBM` + benchmark `IndoBERT`<br>- Tuning `GridSearchCV` 5-fold & uji asumsi. | ✅ **Siap** |
| **6** | **Aturan Prototype & Uji Skenario (Bab 6)** | Wajib memiliki 3 fitur (Input tunggal, Upload CSV batch, Penjelasan aspek/kata kunci) & lolos uji **3 Skenario Bab 6.5** (Normal, Ekstrem, Kosong/Salah Format). | `prototype/app.py` memiliki 3 Tab lengkap + validasi input ketat, diuji otomatis oleh `prototype/test_prototype_scenarios.py`. | ✅ **Lulus Uji** |

---

## 2. Jadwal Sprint Harian 9 Hari (21 November – 30 November 2026)

Mengingat waktu pengerjaan tersisa **9 hari** sebelum penutupan FLearn pada **30 November 2026 pukul 23.00 WIB**, berikut adalah jadwal *sprint* harian dan sinkronisasi tugas untuk 3 anggota tim:

| Hari / Tanggal | Fokus Fase CRISP-DM & Bab Laporan | Tugas **Member 1 (Project Lead, Modeling, Audit & Prototype Backend)** | Tugas **Member 2 (Data Engineering, Preprocessing, ABSA & UI Prototype)** | Tugas **Member 3 (EDA, Topic Modeling, Laporan 7 Bab & Slide PDF)** | Deliverable Harian (Definition of Done) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **Hari 1–2**<br>21–22 Nov 2026 | **Fase 1 & Fase 2**<br>(Bab 1 & Bab 2 Laporan) | Finalisasi Bab 1 (Tujuan Bisnis vs Tujuan Data Science, Target Metrik Terukur Bab 1.4: Macro-F1 $\ge 0.63$, ROC-AUC $\ge 0.85$). | Audit kualitas 71.221 baris data (missing values, duplikat, distribusi slang) & finalisasi kamus `slang_dictionary_id.json`. | Menjalankan `01_Data_Understanding_dan_EDA.ipynb`, membuat grafik Bab 2.4 + menulis **2 kalimat interpretasi** di bawah tiap grafik. | Notebook `01_Data_Understanding_dan_EDA.ipynb` & Bab 1–2 di `.docx` tuntas. |
| **Hari 3–4**<br>23–24 Nov 2026 | **Fase 3: Data Preparation & ABSA**<br>(Bab 3 Laporan) | Verifikasi pembagian data **80:20 Stratified Split (`random_state=42`)** sebelum `TfidfVectorizer.fit()` (Zero Leakage Gate). | Menjalankan `02_Data_Preparation_dan_ABSA.ipynb`, normalisasi slang dengan preservasi negasi (`tidak`, `kurang`, `jangan`), tagging 4 aspek ABSA & *Discrepancy*. | Menyusun tabel langkah persiapan Bab 3.1, diagram pembagian data Bab 3.2, dan tabel perbandingan sebelum vs sesudah preprocessing Bab 3.3. | Notebook `02_Data_Preparation_dan_ABSA.ipynb` & Bab 3 di `.docx` tuntas. |
| **Hari 5–6**<br>25–26 Nov 2026 | **Fase 4 & Fase 5: Modeling, Tuning & Audit**<br>(Bab 4 & Bab 5 Laporan) | Menjalankan `03_Modeling_Tuning_dan_Evaluasi.ipynb` (Baseline `LogisticRegression` vs 2 Model Alternatif + `GridSearchCV` 5-fold), mengeksekusi `metric_sanity_audit.py`, & menyimpan `models/genai_absa_pipeline.joblib`. | Menganalisis *Confident Errors* (*False Positive / False Negative*) & memvalidasi matriks *Net Sentiment Score* 4 Aspek $\times$ 8 Aplikasi AI. | Mengekstraksi topik keluhan/pujian utama, memasukkan Tabel Perbandingan Hasil Bab 5.2, Confusion Matrix, SHAP/Feature Importance Bab 5.3, & evaluasi target Bab 5.4. | Notebook `03_Modeling_Tuning_dan_Evaluasi.ipynb`, file `.joblib`, & Bab 4–5 tuntas. |
| **Hari 7–8**<br>27–28 Nov 2026 | **Fase 6: Deployment Prototype Streamlit & Uji Skenario**<br>(Bab 6 Laporan) | Integrasi backend `models/genai_absa_pipeline.joblib` ke `prototype/app.py` dan deployment ke **Streamlit Community Cloud**. | Mengembangkan UI 3 Tab pada `prototype/app.py`, menjalankan `prototype/test_prototype_scenarios.py` (Uji Normal, Ekstrem, Kosong/Salah Format), & mengambil screenshot Bab 6.4–6.5. | Menyusun Bab 6 (Pengguna, Alur Sistem, Fitur, Cara Menggunakan + Screenshot, Tabel Hasil Uji 3 Skenario Bab 6.5) & Bab 7 (Kesimpulan menjawab Bab 1.2). | URL Live Streamlit aktif & Bab 6–7 di `.docx` tuntas. |
| **Hari 9–10**<br>29–30 Nov 2026<br>*(s/d 23.00 WIB)* | **Final QA, Slide Presentasi PDF & Submit FLearn** | Final audit seluruh `.ipynb` (*Restart & Run All*), cek semua link GitHub & Streamlit aktif. | Uji coba ulang link Streamlit dari perangkat berbeda & validasi file CSV contoh untuk demo presentasi. | Finalisasi Daftar Pustaka APA, konversi slide presentasi ke `682024046_KEL_DS_Presentasi.pdf`, & kompilasi paket pengumpulan FLearn. | **Seluruh 4 Deliverables ter-submit di FLearn sebelum 30 Nov 2026 pukul 23.00 WIB.** |

---

## 3. Matriks Pembagian Peran & Kontribusi 3 Anggota Tim (Untuk Tabel Halaman Awal Laporan UKSW)

Tabel ini dirancang agar langsung selaras dengan **Tabel Pembagian Kerja Anggota** pada halaman awal template laporan resmi FTI UKSW:

| Anggota | NIM & Nama | Peran Utama dalam Kelompok | Rincian Tugas & Kontribusi Nyata (Kode, Laporan, & Prototype) | Persentase Kontribusi |
| :---: | :--- | :--- | :--- | :---: |
| **Member 1** | **682024046**<br>*(Arthur Willyam Liang)* | **Project Lead, Machine Learning Modeling, Metric Sanity Audit & Backend Deployment** | 1. Merancang kerangka metodologi CRISP-DM, rumusan masalah, dan ukuran keberhasilan (Bab 1).<br>2. Membangun `03_Modeling_Tuning_dan_Evaluasi.ipynb` (Baseline `LogisticRegression` + 2 Model Alternatif + `GridSearchCV` 5-fold + uji asumsi model Bab 4).<br>3. Menjalankan 6 pemeriksaan forensik anti-leakage (`metric_sanity_audit.py`), evaluasi multi-slice, analisis atribusi fitur/SHAP (Bab 5), dan mengekspor `models/genai_absa_pipeline.joblib`.<br>4. Mengintegrasikan pipeline model ke backend Streamlit dan melakukan deployment ke Streamlit Community Cloud. | **34%** |
| **Member 2** | **682024016**<br>*(Brendan Wilson Santoso)* | **Data Engineering, Indonesian Text Preprocessing, ABSA Tagging & Prototype UI/Testing** | 1. Melakukan verifikasi integritas dataset `71.221` baris ulasan Google Play Store dan menyusun kamus data (Bab 2.1–2.3).<br>2. Membangun `02_Data_Preparation_dan_ABSA.ipynb`, kamus normalisasi slang Bahasa Indonesia (`slang_dictionary_id.json`) dengan preservasi negasi, ekstraksi 4 aspek layanan (ABSA), dan deteksi *Rating-Text Discrepancy* (Bab 3).<br>3. Mengembangkan antarmuka pengguna `prototype/app.py` (Form input tunggal, Upload CSV batch, validasi input) serta menjalankan pengujian 3 skenario Bab 6.5 (`test_prototype_scenarios.py`). | **33%** |
| **Member 3** | **682024059**<br>*(Tiara Agustien Kusumawati)* | **Exploratory Data Analysis (EDA), Topic Discovery, Penyusunan Laporan 7 Bab & Slide PDF** | 1. Membangun `01_Data_Understanding_dan_EDA.ipynb`, menghasilkan visualisasi distribusi aplikasi, rating, panjang ulasan, dan analisis N-Gram beserta interpretasi 2 kalimat per grafik (Bab 2.4–2.5).<br>2. Mengekstraksi topik keluhan dan pujian utama lintas 8 aplikasi Generative AI serta menyusun dashboard komparasi ekosistem AI (Tab 3 Prototype).<br>3. Mengompilasi dokumen laporan akhir `682024046_KEL_DS_Laporan.docx` sesuai template 7 Bab FTI UKSW (termasuk Bab 7 & Daftar Pustaka APA) serta mendesain slide presentasi `.pdf`. | **33%** |

---

## 4. Pemetaan 7 Bab Template Laporan AM552 ke Artefak Kode & Notebook

Agar tidak ada satu pun sub-bab dari template dosen yang terlewat, berikut adalah pemetaan langsung antara **Template Laporan AM552** dengan **File Notebook / Script** di workspace:

1. **Halaman Judul, Tabel Pembagian Kerja, & Abstrak (150–250 Kata)**:
   - *Sumber*: `06_deployment_reporting/682024046_KEL_DS_Laporan.docx` (otomatis di-generate oleh `tools/generate_uksw_final_report_docx.py`).
2. **Bab 1 — Pendahuluan (1.1 Latar Belakang, 1.2 Rumusan Masalah, 1.3 Tujuan Bisnis & Data Science, 1.4 Ukuran Keberhasilan, 1.5 Batasan)**:
   - *Sumber*: `01_business_understanding/project_charter_dan_definisi_aspek.md`.
   - *Highlight Kepatuhan*: Bab 1.3 memisahkan dengan tegas **Tujuan Bisnis** (memetakan kepuasan aspek layanan 8 aplikasi AI untuk membantu pengguna & pengembang) dan **Tujuan Data Science** (klasifikasi sentimen 3-kelas, ekstraksi 4 aspek ABSA, dan deteksi *Rating-Text Discrepancy*). Bab 1.4 menetapkan target angka terukur: **Macro F1-Score $\ge 0.63$** (mengalahkan Dummy Baseline `0.2776` sebesar $> +0.35$ poin), **ROC-AUC (OvR) $\ge 0.85$**, dan **Weighted F1 $\ge 0.80$**.
3. **Bab 2 — Pemahaman Data (2.1 Sumber Data, 2.2 Kamus Data, 2.3 Kondisi Data, 2.4 Eksplorasi Data, 2.5 Temuan Penting)**:
   - *Sumber Notebook*: `notebooks/01_Data_Understanding_dan_EDA.ipynb`.
   - *Highlight Kepatuhan*: Setiap grafik di Bab 2.4 dilengkapi tepat **2 kalimat penjelasan** di bawahnya (kalimat pertama menjelaskan fakta visual grafik, kalimat kedua menjelaskan makna/implikasinya bagi pemodelan).
4. **Bab 3 — Persiapan Data (3.1 Langkah Persiapan, 3.2 Pembagian Data, 3.3 Data Akhir)**:
   - *Sumber Notebook*: `notebooks/02_Data_Preparation_dan_ABSA.ipynb`.
   - *Highlight Kepatuhan*: Bab 3.2 menjelaskan pembagian data **80% Data Latih (`56.976` baris) : 20% Data Uji (`14.245` baris)** secara *Stratified* (`random_state=42`) yang dilakukan **sebelum** `TfidfVectorizer.fit()` untuk mencegah *data leakage*.
5. **Bab 4 — Pemodelan (4.1 Pilihan Model, 4.2 Pengaturan Parameter, 4.3 Asumsi Model)**:
   - *Sumber Notebook*: `notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb`.
   - *Highlight Kepatuhan*:
     - **Model 1 (Baseline Sederhana)**: `Logistic Regression` (`class_weight='balanced'`).
     - **Model 2 (Alternatif 1 — Probabilistic NLP)**: `Complement Naive Bayes` (`ComplementNB`, varian MultinomialNB yang dirancang khusus untuk korpus teks dengan kelas timpang).
     - **Model 3 (Alternatif 2 — Margin/Gradient Classifier)**: `Calibrated Linear Support Vector Machine (LinearSVC)` & `HistGradientBoosting / LightGBM` (serta benchmark konseptual `IndoBERT`).
     - Dilengkapi tabel *Hyperparameter Tuning* (`GridSearchCV` 5-fold) dan pemeriksaan *Asumsi Model*.
6. **Bab 5 — Evaluasi (5.1 Ukuran Evaluasi, 5.2 Perbandingan Hasil, 5.3 Analisis Kesalahan, 5.4 Evaluasi Pencapaian Target, 5.5 Keterbatasan)**:
   - *Sumber Notebook*: `notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb` & `05_evaluation/metric_sanity_audit.py`.
   - *Highlight Kepatuhan*: Menyajikan tabel perbandingan metrik pada Data Latih vs Data Uji, audit 4-slice (*Full Test Set*, *Short Reviews*, *Deduplicated Unseen Text*, *Substantive Reviews $\ge 5$ Kata*), grafik *Confusion Matrix*, grafik atribusi kata kunci terpenting (*SHAP / Linear Coefficients*), contoh nyata *False Positive & False Negative* (yang membuktikan fenomena *Rating-Text Discrepancy*), serta tabel pencapaian target Bab 1.4 (**Tercapai**).
7. **Bab 6 — Deployment / Prototype (6.1 Pengguna & Kegunaan, 6.2 Alur Sistem, 6.3 Fitur, 6.4 Cara Menggunakan, 6.5 Hasil Uji Coba 3 Skenario)**:
   - *Sumber Kode*: `prototype/app.py`, `models/genai_absa_pipeline.joblib`, dan `prototype/test_prototype_scenarios.py`.
   - *Highlight Kepatuhan*: Bab 6.5 menyajikan tabel hasil uji nyata pada **Skenario 1 (Input Normal)**, **Skenario 2 (Input Ekstrem / Slang & Discrepancy)**, dan **Skenario 3 (Input Kosong / Hanya Angka / CSV Salah Format)**.
8. **Bab 7 — Penutup (7.1 Kesimpulan, 7.2 Saran Penerapan, 7.3 Pengembangan Selanjutnya) & Daftar Pustaka (APA)**:
   - *Highlight Kepatuhan*: Poin 7.1 menjawab satu per satu seluruh pertanyaan pada Rumusan Masalah Bab 1.2 dengan didukung angka hasil evaluasi Bab 5. Daftar Pustaka memuat sumber dataset Google Play Store dan literatur ilmiah format APA.

---

## 5. Protokol Audit Forensik Metrik (`05_evaluation/metric_sanity_audit.py`)

Protokol **6 Pemeriksaan Forensik Wajib** tetap menjadi gerbang validasi utama di dalam `notebooks/03_Modeling_Tuning_dan_Evaluasi.ipynb`:
1. **AUDIT-1 (Circular / Target Leakage Check)**: Memastikan `score` bintang dan `app_name` tidak bocor ke fitur input $X$.
2. **AUDIT-2 (Class Imbalance vs. Dummy Baseline)**: Membandingkan Macro F1, Balanced Accuracy, dan MCC terhadap `DummyClassifier`.
3. **AUDIT-3 (Exact-Text Duplicate & Substantive Review $\ge 5$ Kata)**: Memisahkan metrik pada teks pendek generik (`"bagus"`, `"mantap"`) vs kalimat unik & substantif ($\ge 5$ kata).
4. **AUDIT-4 (Leave-One-App-Out Cross-Domain Generalization)**: Menguji kemampuan generalisasi model pada aplikasi yang disembunyikan saat training.
5. **AUDIT-5 (Pipeline Split Hygiene)**: Memverifikasi `0%` overlap `reviewId` dan isolasi `TfidfVectorizer.fit()` hanya pada data latih.
6. **AUDIT-6 (Randomized-Label Permutation Sanity Test & Error Forensics)**: Memastikan ROC-AUC jatuh ke $\approx 0.50$ saat label diacak ($y_{\text{shuffled}}$).
