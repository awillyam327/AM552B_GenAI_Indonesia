# PANDUAN ONBOARDING TIM & KURIKULUM BELAJAR KILAT (PROYEK AKHIR AM522B)

> **Judul Proyek**: *Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia*  
> **Mata Kuliah**: Data Science (AM522B)  
> **Komposisi Tim**: 3 Anggota (1 Project Lead & Modeling + 1 Data Engineering & ABSA + 1 EDA, Topic Modeling & Visualisasi)  
> **Aset Dataset Utama**: `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv` (**71.221 baris ulasan asli** dari Google Play Store Indonesia, 100% *real scraped data*, 0% duplikat)

---

## 1. Penjelasan Topik Tanpa Jargon Rumit (*The Big Picture*)

### A. Masalah Apa yang Sedang Kita Pecahkan?
Dalam 2–3 tahun terakhir, cara mahasiswa, programmer, pekerja kantor, dan masyarakat Indonesia mencari informasi atau menyelesaikan tugas berubah drastis. Kalau dulu orang hanya mengandalkan Google Search, sekarang jutaan pengguna di Indonesia mengunduh aplikasi **Generative AI** di smartphone mereka:
1. **ChatGPT (OpenAI)** (`22.000` ulasan di dataset kita)
2. **Google Gemini** (`12.600` ulasan)
3. **Character.AI** (`12.000` ulasan)
4. **Perplexity - Ask Anything** (`8.043` ulasan)
5. **Microsoft Copilot** (`5.459` ulasan)
6. **DeepSeek - AI Assistant** (`5.374` ulasan)
7. **Claude by Anthropic** (`4.009` ulasan)
8. **Poe - Fast AI Chat** (`1.736` ulasan)

**Pertanyaan besarnya:** *Di mata pengguna Indonesia, aplikasi AI mana yang sebenarnya paling memuaskan, mana yang paling banyak dikeluhkan, dan **apa alasan spesifik** di balik pujian atau keluhan tersebut?*

Jika kita hanya melihat **rata-rata bintang** di Google Play Store, kita akan tertipu. Misalnya, aplikasi memiliki rata-rata bintang `4.4`, tetapi di balik angka itu ribuan pengguna mengeluhkan *"kuota gratis terlalu sedikit"*, *"sering error saat login"*, atau *"jawaban Bahasa Indonesianya kaku dan ngawur (halusinasi)"*. Di sisi lain, ada aplikasi seperti **Character.AI** yang rating rata-ratanya anjlok ke **`1.58`**, atau **DeepSeek** (`3.77`) yang viral karena gratis dan pintar namun sering dikeluhkan *"server busy"*.

Proyek kita membedah **71.221 ulasan nyata** tersebut menggunakan teknik *Natural Language Processing (NLP)* dan *Machine Learning* agar menghasilkan peta kekuatan dan kelemahan masing-masing aplikasi AI di Indonesia secara objektif.

---

### B. Intuisi 4 Konsep Inti Proyek Ini (Bahasa Sederhana)

#### 1. *Sentiment Analysis* Biasa vs. *Aspect-Based Sentiment Analysis (ABSA)*
* **Sentiment Analysis Biasa**: Hanya menebak apakah satu ulasan bernada **Positif**, **Netral**, atau **Negatif**.  
  *Kelemahan*: Bagaimana jika pengguna menulis: *"Jawaban AI-nya pintar banget dan Bahasa Indonesianya luwes (Positif), TAPI kuota gratisnya pelit dan aplikasinya sering macet (Negatif)"*? Analisis sentimen biasa akan bingung!
* **Aspect-Based Sentiment Analysis (ABSA)**: Kita memotong dan memetakan ulasan ke dalam **4 Aspek Spesifik Layanan**:
  1. **Aspek Akurasi & Kecerdasan AI** (Pintar, membantu tugas/coding vs. Halusinasi, jawaban ngawur/tidak nyambung).
  2. **Aspek Harga & Limitasi Kuota** (Gratis, hemat vs. Limit pesan cepat habis, langganan Plus/Pro kemahalan).
  3. **Aspek Performa, UI/UX & Bug Aplikasi** (Cepat, tampilan rapi vs. Server sibuk/down, gagal login, live voice macet, berat).
  4. **Aspek Kemampuan Bahasa Indonesia** (Bahasa natural/mudah dipahami vs. Terjemahan kaku, kurang paham konteks lokal).
  *Hasilnya*: Kita bisa membuat rapor per aplikasi—misalnya ChatGPT unggul di aspek *Akurasi*, tetapi banyak sentimen negatif di aspek *Harga/Kuota*; sedangkan DeepSeek unggul di aspek *Harga/Kuota Gratis*, tetapi banyak sentimen negatif di aspek *Performa Server*.

#### 2. *Rating-Text Discrepancy* (Paradoks Bintang vs. Isi Teks)
Di Google Play Store Indonesia, banyak pengguna yang:
* Memberi **Bintang 5 (⭐⭐⭐⭐⭐)**, padahal isi teksnya mengeluh: *"Saya kasih bintang 5 biar dibaca developer, tolong dong mode live-nya sering macet dan error terus!"*
* Memberi **Bintang 1 (⭐)**, padahal isi teksnya memuji: *"Aplikasinya bagus banget membantu tugas kuliah saya."* (salah pencet bintang).

Fenomena ketidaksesuaian antara angka bintang dan isi kalimat ini disebut **Rating-Text Discrepancy**. Dalam proyek ini, kita mendeteksi dan mengukur seberapa banyak ulasan yang mengalami *discrepancy*, lalu melatih model Machine Learning untuk memprediksi sentimen sejati dari teksnya.

#### 3. *Topic Modeling* (`LDA` & `BERTopic`)
Tidak mungkin manusia membaca 71.221 ulasan satu per satu untuk mencari tahu topik apa saja yang sedang ramai dibicarakan.
* **Topic Modeling** adalah algoritma *unsupervised learning* yang secara otomatis mengelompokkan ribuan ulasan yang mirip ke dalam "klaster topik" (misalnya: Topik #1 = *Keluhan Sensor/Filter Character.AI*, Topik #2 = *Masalah Login Akun Google*, Topik #3 = *Bantuan Coding & Skripsi*).
* Kita membandingkan algoritma klasik berbasis statistik kata (**LDA / Latent Dirichlet Allocation**) dengan algoritma modern berbasis *deep learning embeddings* (**BERTopic**).

#### 4. *Explainable AI* dengan `SHAP` (*SHapley Additive exPlanations*)
Saat model Machine Learning (seperti **CatBoost/XGBoost/LightGBM** atau **IndoBERT**) memprediksi sebuah ulasan sebagai "Negatif / Tidak Puas", dosen atau penguji pasti bertanya: *"Dari mana model tahu ulasan itu negatif? Kata atau fitur apa yang paling memengaruhi keputusan model?"*
* **SHAP** adalah teknik untuk "membuka kotak hitam" (*black-box*) model AI kita dan menampilkan grafik kata kunci serta fitur apa saja (misalnya kata `"limit"`, `"error"`, `"bayar"`, `"lambat"`, atau panjang ulasan) yang paling kuat mendorong ketidakpuasan pengguna.

---

## 2. Kurikulum Belajar Terarah (*Learning Checklist per Peran*)

Agar waktu belajar efisien, kalian **tidak perlu mempelajari semua cabang ilmu Data Science**. Cukup pelajari daftar materi terfokus di bawah ini (estimasi waktu belajar: 2–3 hari).

### A. Materi Wajib Bersama (Untuk Ketiga Anggota Tim — Estimasi 3 Jam)
- [ ] **1. Eksplorasi Dataset Proyek Kita**: Buka file `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv` dan pahami fungsi 11 kolomnya:
  - `reviewId` (ID unik ulasan), `app_name` (Nama aplikasi AI), `app_id` (ID paket Play Store), `userName` (Nama akun), `content` (Teks ulasan asli), `score` (Rating bintang 1–5), `thumbsUpCount` (Jumlah jempol/likes), `reviewCreatedVersion` (Versi aplikasi), `at` (Tanggal & jam ulasan), `replyContent` (Balasan developer), `language_region` (`id_ID`).
- [ ] **2. Dasar Manipulasi Data dengan `pandas`**:
  - Membaca CSV (`pd.read_csv`), filter baris (`df[df['score'] == 1]`), agregasi (`groupby`, `value_counts`), dan menangani nilai kosong (`isna`, `fillna`).
- [ ] **3. Konsep Dasar Pemrosesan Teks (NLP Bahasa Indonesia)**:
  - Memahami apa itu *Case Folding* (huruf kecil semua), *Tokenization* (memecah kalimat jadi kata), *Stopwords* (kata umum seperti *"yang"*, *"di"*, *"dan"*), *Slang Normalization* (mengubah *"gk"*, *"bgt"*, *"lemot"* menjadi kata baku), serta *TF-IDF* (mengubah teks menjadi angka bobot kata).

---

### B. Kurikulum Spesifik Berdasarkan Peran Anggota

#### 👨‍💻 Member 1 — Arthur Willyam Liang (682024046) [Project Lead, Predictive Modeling & Explainability]
*Fokus*: Melatih model klasifikasi sentimen & *discrepancy*, membandingkan algoritma Ensemble vs. Transformer, serta menjelaskan model dengan SHAP.
- [ ] **Konsep**: *Supervised Text Classification*, *Class Imbalance* (mengapa F1-Macro lebih adil daripada Accuracy saat Bintang 5 mendominasi 63,96% data), *Stratified K-Fold Cross-Validation*.
- [ ] **Algoritma**:
  - Gradient Boosting: `CatBoostClassifier`, `XGBClassifier`, `LGBMClassifier`.
  - Pre-trained Language Model Bahasa Indonesia: `IndoBERT` (`indobenchmark/indobert-base-p1`).
  - Explainable AI: `shap.TreeExplainer` untuk menampilkan *summary plot* kata/fitur terpenting.
- [ ] **Library Python & Kata Kunci Referensi**:
  - `scikit-learn` (`TfidfVectorizer`, `classification_report`, `f1_score`, `roc_auc_score`, `confusion_matrix`)
  - `catboost`, `xgboost`, `lightgbm`
  - `transformers`, `torch` (HuggingFace fine-tuning / feature extraction)
  - `shap`

#### 🛠️ Member 2 — Brendan Wilson Santoso (682024016) [Data Engineering, Indonesian Text Preprocessing & ABSA]
*Fokus*: Membersihkan 71.221 teks ulasan Bahasa Indonesia yang penuh singkatan/slang, mengekstrak label 4 aspek layanan (ABSA), dan membuat fitur teks.
- [ ] **Konsep**:
  - Tantangan teks media sosial/ulasan Indonesia: *typo*, singkatan (*"yg"*, *"gak"*, *"bgt"*, *"udh"*, *"apk"*), reduplikasi (*"tiba2"*), dan emoji.
  - Pentingnya **menyimpan kata negasi** (*"tidak"*, *"kurang"*, *"jangan"*, *"bukan"*, *"belum"*) saat menghapus *stopwords* agar kalimat *"tidak bagus"* tidak berubah menjadi *"bagus"*.
  - Pelabelan aspek (*Rule-based / Seed-keyword Aspect Tagging* & *Lexicon Sentiment Scoring*) serta aturan deteksi *Rating-Text Discrepancy*.
- [ ] **Library Python & Kata Kunci Referensi**:
  - `re` (Regular Expressions untuk hapus URL, tanda baca berlebih, karakter berulang seperti *"bagusssss"* -> *"bagus"*)
  - `PySastrawi` / `Sastrawi` (Stemmer & Stopword Remover Bahasa Indonesia)
  - Kamus Slang Bahasa Indonesia (*Colloquial Indonesian Lexicon* / `slang_dictionary_id.json` di folder `03_data_preparation/`)
  - `scikit-learn` (`TfidfVectorizer`, *N-gram feature engineering*)

#### 📊 Member 3 — Tiara Agustien Kusumawati (682024059) [Exploratory Data Analysis (EDA), Topic Modeling & Visualisasi/Laporan]
*Fokus*: Memvisualisasikan pola data lintas 8 aplikasi AI, menjalankan *Topic Modeling* (BERTopic & LDA), serta menyusun grafik publikasi dan narasi laporan akhir.
- [ ] **Konsep**:
  - *Time-Series Trend Analysis*: Melihat lonjakan ulasan (*review surge*) per bulan/minggu (misal saat DeepSeek rilis Januari 2025 atau update besar ChatGPT & Gemini).
  - *N-Gram Analysis*: Mencari pasangan 2 kata (*Bigram*, misal `"server sibuk"`, `"limit habis"`) dan 3 kata (*Trigram*) yang paling sering muncul di ulasan bintang 1 vs bintang 5.
  - *Topic Modeling*: Cara kerja `LDA` (distribusi kata probabilistik) vs. `BERTopic` (mengelompokkan kalimat menggunakan vektor makna/semantic embeddings).
- [ ] **Library Python & Kata Kunci Referensi**:
  - `matplotlib`, `seaborn`, `wordcloud` (visualisasi distribusi, heatmap aspek per aplikasi, tren waktu)
  - `scikit-learn` (`CountVectorizer`, `LatentDirichletAllocation`)
  - `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`

---

## 3. Matriks Pembagian Peran & Tanggung Jawab (3 Anggota Tim)

Pembagian kerja dirancang mengikuti alur **CRISP-DM** agar ketiga anggota dapat bekerja secara paralel dengan beban kerja (*workload*) yang adil dan saling terhubung:

| Dimensi Tugas | **Member 1: Arthur Willyam Liang (682024046)** | **Member 2: Brendan Wilson Santoso (682024016)** | **Member 3: Tiara Agustien Kusumawati (682024059)** |
| :--- | :--- | :--- | :--- |
| **Fase CRISP-DM Utama** | Fase 1 (*Business Understanding*), Fase 4 (*Supervised Modeling*), Fase 5 (*Evaluation & SHAP*) | Fase 2 (*Data Audit*), Fase 3 (*Data Preparation, Normalization & ABSA Tagging*) | Fase 2 (*Exploratory Data Analysis*), Fase 4 (*Topic Modeling*), Fase 6 (*Reporting & Deck*) |
| **Folder Kerja Utama** | `01_business_understanding/`<br>`04_modeling/`<br>`05_evaluation/` | `03_data_preparation/`<br>`04_modeling/` (fitur ABSA) | `02_data_understanding/`<br>`04_modeling/` (Topic Modeling)<br>`06_deployment_reporting/` |
| **Tugas Teknis Spesifik** | 1. Menyusun arsitektur eksperimen & metrik evaluasi.<br>2. Melatih model klasifikasi sentimen & *discrepancy* (`CatBoost`, `XGBoost`, `LightGBM`).<br>3. Menjalankan benchmark `IndoBERT`.<br>4. Menghasilkan grafik interpretasi fitur `SHAP` & *Confusion Matrix*. | 1. Membersihkan teks 71.221 ulasan (*cleaning, case folding, emoji/noise handling*).<br>2. Membangun & memperkaya kamus normalisasi slang Bahasa Indonesia.<br>3. Melabeli 4 kategori aspek (ABSA) & flag *Rating-Text Discrepancy*.<br>4. Mengekspor dataset bersih (`CLEAN_DATASET_ABSA.csv`) untuk Member 1 & 3. | 1. Menjalankan EDA komparatif 8 aplikasi (distribusi rating, panjang teks, *time-series*, N-Gram).<br>2. Menjalankan eksperimen *Topic Modeling* (`LDA` vs. `BERTopic`).<br>3. Membuat visualisasi matriks komparasi aspek antar-aplikasi.<br>4. Mengompilasi laporan akhir & slide presentasi. |
| **Deliverables (Output)** | - Script & model klasifikasi terlatih<br>- Tabel komparasi F1-Score & ROC-AUC<br>- Visualisasi & analisis SHAP | - Pipeline `preprocessing_pipeline.py`<br>- Kamus slang & kamus aspek ABSA<br>- File dataset bersih siap modeling | - Laporan & grafik EDA resolusi tinggi (`.png`)<br>- Visualisasi klaster topik per aplikasi<br>- Draft laporan akhir & slide presentasi |

---

## 4. Alur Kolaborasi Tim (Workflow Step-by-Step)

```
[RAW DATASET: 71.221 Baris]
         │
         ├──► (Langkah 1 - Paralel)
         │     ├─ Member 1: Finalisasi Project Charter & Setup Baseline Evaluasi (01_business_understanding)
         │     ├─ Member 3: Jalankan Initial EDA & Grafik Profil Dataset (02_data_understanding)
         │     └─ Member 2: Bangun Kamus Slang, Kamus Aspek ABSA & Pipeline Cleaning (03_data_preparation)
         │
         ▼
[CLEAN DATASET + LABEL ASPEK ABSA + LABEL DISCREPANCY] (Output Member 2)
         │
         ├──► (Langkah 2 - Paralel Modeling & Discovery)
         │     ├─ Member 1: Eksperimen 1 — Klasifikasi Sentimen & Discrepancy (Ensemble vs IndoBERT) (04_modeling)
         │     ├─ Member 2: Analisis Polaritas Sentimen per Aspek (ABSA Score Matrix across 8 Apps) (04_modeling)
         │     └─ Member 3: Eksperimen 2 — Topic Modeling Keluhan & Pujian (LDA vs BERTopic) (04_modeling)
         │
         ▼
[EVALUASI, SHAP EXPLAINABILITY & LAPORAN AKHIR]
         │
         └──► (Langkah 3 - Integrasi Akhir)
               ├─ Member 1: Evaluasi Metrik (Macro-F1, ROC-AUC, Confusion Matrix) & SHAP Analysis (05_evaluation)
               └─ Member 1, 2, 3: Kompilasi Dashboard Grafik, Laporan Akhir & Slide Presentasi (06_deployment_reporting)
```

---

## 5. Panduan Menjalankan Script di Workspace Ini

1. **Verifikasi Integritas Raw Dataset**:
   ```bash
   python verify_dataset.py
   ```
2. **Menjalankan Exploratory Data Analysis (EDA) Awal & Membuat Grafik**:
   ```bash
   python 02_data_understanding/run_initial_eda.py
   ```
3. **Menjalankan Template Preprocessing & Pelabelan Awal ABSA**:
   ```bash
   python 03_data_preparation/preprocessing_pipeline_template.py
   ```
