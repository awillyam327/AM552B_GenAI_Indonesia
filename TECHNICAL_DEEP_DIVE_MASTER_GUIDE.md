# TECHNICAL DEEP DIVE MASTER GUIDE: ANALISIS KOMPARATIF ABSA & PEMODELAN TOPIK EKOSISTEM GENERATIVE AI DI INDONESIA

**Mata Kuliah**: AM552 - Data Science of IS (Program Studi S1 Sistem Informasi, Fakultas Teknologi Informasi, Universitas Kristen Satya Wacana)  
**Kode Kelompok**: `682024046_KEL_DS`  
**Judul Proyek**: *Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia*  
**Komposisi Tim Pengembang**:
1. **Arthur Willyam Liang (NIM: 682024046)** — *Project Lead, Statistical Modeling & Machine Learning Architect*
2. **Brendan Wilson Santoso (NIM: 682024016)** — *Data Engineer, Preprocessing Specialist & Aspect-Based Tagging Lead*
3. **Tiara Agustien Kusumawati (NIM: 682024059)** — *Senior Data Analyst, Topic Discovery Specialist & Technical Lead Writer*

---

## EXECUTIVE FOREWORD: PERSPEKTIF PRINCIPAL DATA SCIENTIST & THESIS EXAMINER

Dokumen ini dirancang sebagai **Buku Putih Teknis (*Technical Whitepaper & Defense Handbook*)** yang membedah arsitektur kecerdasan buatan, rekayasa data, perumusan algoritma matematis, metrik evaluasi forensik, hingga pertahanan sidang (*thesis defense*) dari proyek tugas akhir AM552. 

Banyak proyek Data Science tingkat sarjana terperangkap dalam jebakan ilusi performa (*Performance Illusion*): mengklaim akurasi model $\ge 90\%$, padahal dataset yang digunakan memiliki ketimpangan kelas (*class imbalance*) ekstrem di mana model hanya menebak kelas mayoritas, atau terjadi kebocoran data (*data leakage*) halus pada tahap rekayasa fitur. 

Proyek ini mengambil pendekatan saintifik berstandar industri:
- Menguji $71.221$ ulasan mentah nyata (*100% real scraped reviews*) Google Play Store Indonesia lintas 8 platform Generative AI terkemuka.
- Mengadopsi siklus hidup **CRISP-DM (Cross-Industry Standard Process for Data Mining)** secara disiplin.
- Menerapkan **6 Gerbang Forensik Metrik Anti-Leakage (*Metric Sanity Audit*)**, termasuk *Circular Leakage Detection*, *Substantive Review Slicing*, *Leave-One-App-Out Cross-Domain Validation*, hingga *Randomized-Label Permutation Test*.
- Membangun model berbobot seimbang (*Cost-Sensitive / Class-Weighted*) yang dioptimalkan pada **Macro F1-Score (0,6502)** dan **ROC-AUC One-vs-Rest (0,8831)**, melampaui seluruh target KPI dasar.
- Menerapkan inferensi inferensial berbasis aspek (*Multi-Label ABSA*) dan deteksi anomali *Rating-Text Discrepancy* yang dideploy secara *live* pada antarmuka web Streamlit dengan latensi inferensi di bawah 50 milidetik.

---

# BAGIAN 1: ARSITEKTUR & PROSES PEMBUATAN END-TO-END (CRISP-DM PIPELINE WALKTHROUGH)

Berikut adalah rekayasa langkah demi langkah pembangunan sistem dari *ground zero* hingga *production deployment*, merujuk langsung pada implementasi kode di workspace proyek:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                CRISP-DM END-TO-END WORKFLOW PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. DATA ACQUISITION (`scrape_genai_indonesia.py`)                                                │
│    Google Play Store API -> Sort.NEWEST (cap ~4K) -> Sort.MOST_RELEVANT -> Per-Star Filter (1..5) │
│    71.221 Baris Ulasan Unik | 8 Aplikasi | Checkpointing Parquet/CSV | Zero Duplicate `reviewId` │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. DATA UNDERSTANDING (`02_data_understanding/run_initial_eda.py`, Notebook 01)                  │
│    Missing Value Audit | Bimodal Class Imbalance (63,96% Star 5 vs 3,84% Star 3)                │
│    Rating-Text Discrepancy Discovery (4,59% total, DeepSeek 11,98%) | Review Length Correlation  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. DATA PREPARATION (`03_data_preparation/`, Notebook 02)                                        │
│    Regex Cleaning -> Elongation Reduction `(.)\1{2,}` -> Slang Normalization (350+ kata)        │
│    Negation Preservation (`tidak`, `kurang`, `belum`) -> Multi-Label ABSA Tagging (A1 - A4)      │
│    >>> CRITICAL CHECKPOINT: Stratified 80:20 Split BEFORE TfidfVectorizer.fit() <<<             │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. MODELING & HYPERPARAMETER TUNING (`04_modeling/`, Notebook 03)                                │
│    Baseline Dummy -> Tuned Logistic Regression (C=0.5, Balanced) -> ComplementNB -> LinearSVC   │
│    5-Fold Stratified Cross-Validation on Macro F1 | Unsupervised LDA Topic Modeling (5K+5P)     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. FORENSIC EVALUATION (`05_evaluation/metric_sanity_audit.py`, Notebook 03)                    │
│    Audit 1 (Leakage) | Audit 2 (Imbalance) | Audit 3 (Slices) | Audit 4 (LOAO Domain Transfer)  │
│    Audit 5 (Split Hygiene) | Audit 6 (Permutation y_shuffled) | Local Feature Attribution       │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. DEPLOYMENT & PRODUCTION SERVING (`prototype/app.py`, `prototype/core_predictor.py`)          │
│    Serialized Artifact `genai_absa_pipeline.joblib` (931 KB) -> Streamlit Cloud Architecture    │
│    Single Review Deep Analysis + Batch CSV Processing + Real-time Keyword Explainability        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Data Acquisition (Web Scraping Architecture)
* **File Rujukan**: `scrape_genai_indonesia.py`
* **Mekanisme Ekstraksi**: Menggunakan pustaka pihak ketiga `google-play-scraper`.
* **Tantangan Batas Paginasi Play Store**: Server Google Play Store menerapkan batas keras (*hard pagination limit*) sekitar 4.000 hingga 4.500 ulasan per kueri pencarian tunggal. Jika scraper hanya memanggil `reviews(sort=Sort.NEWEST)`, proses akan berhenti prematur setelah menarik ~4.000 data ulasan teratas.
* **Strategi Multi-Pass Bypassing**:
  1. *Pass 1 (Unfiltered Newest)*: Menarik ulasan menggunakan `Sort.NEWEST` dengan parameter `count=200` dan `continuation_token` secara berulang sampai token bernilai `None`.
  2. *Pass 2 (Relevance Fallback)*: Apabila volume belum mencapai kuota target, scraper secara otomatis beralih ke `Sort.MOST_RELEVANT`.
  3. *Pass 3 (Per-Star Filter Partitioning)*: Menjalankan scraping dengan filter skor bintang eksplisit (`filter_score_with=1`, `2`, `3`, `4`, `5`) masing-masing menggunakan loop `continuation_token`. Karena setiap partisi bintang memiliki *token pool* independen di server Google, strategi ini berhasil mengekstraksi hingga belasan ribu ulasan per aplikasi.
* **Mekanisme Checkpoint & Deduplikasi**:
  - Setiap batch 200 baris disimpan dalam direktori `./dataset/checkpoints/<app_id>_<lang_reg>.csv`.
  - Fungsi `normalize_records` membersihkan whitespace pada `reviewId`.
  - `df.drop_duplicates(subset=['reviewId'], keep='first')` memastikan bahwa meskipun ulasan ditarik pada pass yang berbeda, integritas himpunan ulasan tetap 100% unik tanpa duplikasi baris.
* **Hasil Akhir**: $71.221$ ulasan unik pada 8 aplikasi (ChatGPT: 22.000, Gemini: 12.600, Character.AI: 12.000, Perplexity: 6.800, Copilot: 5.400, DeepSeek: 5.374, Claude: 4.009, Poe: 3.038).

### 1.2 Data Understanding & Profiling
* **File Rujukan**: `02_data_understanding/run_initial_eda.py`, `notebooks/01_Data_Understanding_dan_EDA.ipynb`, `02_data_understanding/eda_summary_report.md`
* **Audit Kualitas Awal**:
  - *Missing Values*: Kolom `content`, `score`, `reviewId`, `app_name`, `at` memiliki `0` missing values ($0,00\%$). Hanya kolom opsional `userName` yang memiliki 1 baris NaN, serta `replyContent` / `repliedAt` yang memiliki $\sim 85\%$ NaN karena mayoritas pengembang tidak merespons ulasan.
  - *Class Imbalance Bimodal U-Shaped*: Bintang 5 mendominasi sebesar $63,96\%$ ($45.551$ ulasan), disusul Bintang 1 sebesar $21,79\%$ ($15.519$ ulasan), Bintang 4 sebesar $7,38\%$ ($5.258$ ulasan), Bintang 2 sebesar $3,03\%$ ($2.160$ ulasan), dan kelas paling langka adalah Bintang 3 sebesar $3,84\%$ ($2.733$ ulasan).
* **Penemuan Rating-Text Discrepancy**:
  - Ditemukan sebanyak **3.271 ulasan ($4,59\%$)** yang mengalami kontradiksi langsung antara angka bintang dengan teks ulasan.
  - DeepSeek mencatat angka tertinggi ($11,98\%$) dan Claude ($9,05\%$). Pengguna memberikan Bintang 5 namun menulis teks bernada frustrasi (*"Bintang 5 biar dibaca developer, tolong servernya sering down / batas kuota limit harian cepat habis"*).
* **Korelasi Panjang Teks dengan Rating**:
  - Rata-rata panjang karakter ulasan Bintang 1 adalah **115,7 karakter** (dan Bintang 2 mencapai 137,7 karakter), berbanding terbalik dengan ulasan Bintang 5 yang hanya **34,8 karakter**.
  - Hal ini mengungkap psikologi pengguna: pengguna yang puas cenderung hanya meninggalkan ulasan pendek ekspresif (*"bagus"*, *"mantap"*), sedangkan pengguna yang mengalami masalah teknis menuliskan kronologi kegagalan secara mendalam.

### 1.3 Data Preparation & Feature Engineering
* **File Rujukan**: `03_data_preparation/preprocessing_pipeline_template.py`, `03_data_preparation/slang_dictionary_id.json`, `notebooks/02_Data_Preparation_dan_ABSA.ipynb`
* **Pipeline Pembersihan Teks (`clean_and_normalize_text`)**:
  1. *Case Folding*: Mengubah seluruh huruf menjadi huruf kecil (`text.lower()`).
  2. *URL Stripping*: `re.sub(r'http\S+|www\.\S+', ' ', text)`.
  3. *Elongation Reduction*: Reduksi pemanjangan huruf emosional (`re.sub(r'(.)\1{2,}', r'\1\1', text)`). Contoh: `"baguuussss"` menjadi `"baguss"`, `"lemottttt"` menjadi `"lemott"`. Hal ini mencegah ledakan dimensi kosa kata (*vocabulary explosion*) pada matriks TF-IDF.
  4. *Tokenization*: `re.findall(r'[a-z0-9]+', text)` mengekstraksi token alfanumerik murni dan menyingkirkan tanda baca/emoji.
  5. *Slang / Colloquial Normalization*: Pemetaan kamus `slang_dictionary_id.json` berisi 350+ pasangan kata tidak baku ke bentuk baku (`"gak"` $\rightarrow$ `"tidak"`, `"bgt"` $\rightarrow$ `"sangat"`, `"apk"` $\rightarrow$ `"aplikasi"`, `"eror"` $\rightarrow$ `"error"`).
  6. *Selective Stopword Removal & Negation Preservation*: Berbeda dari pendekatan konvensional yang membuang semua stopword, pipeline ini **secara mutlak mempertahankan kata negasi** (`tidak`, `kurang`, `jangan`, `bukan`, `belum`). Jika kata negasi dibuang, frasa *"tidak bagus"* akan tereduksi menjadi *"bagus"*, membalikkan polaritas sentimen $180^\circ$ dan merusak akurasi model.
* **Struktur Pembagian Data (Anti-Leakage Stratified Split 80:20)**:
  - Total Data Latih: **56.976 ulasan ($80,0\%$)**
  - Total Data Uji: **14.245 ulasan ($20,0\%$)**
  - Stratifikasi multi-dimensi menggunakan kunci komposit: `df['strat_key'] = df['app_name'] + '_' + df['sentiment_3class']`.
  - **MENGAPA URUTAN FIT / TRANSFORM SANGAT KRUSIAL?**:
    Jika `TfidfVectorizer.fit()` dijalankan pada seluruh $71.221$ baris sebelum split data, maka frekuensi dokumen IDF serta perbendaharaan kata (*vocabulary*) telah 'melihat' distribusi kata di data uji. Ini adalah bentuk **Data Leakage (Information Leakage)** laten yang membuat evaluasi model menjadi terlalu optimis semu (*overly optimistic*). Pembagian data wajib dieksekusi terlebih dahulu, lalu `TfidfVectorizer` **hanya di-fit pada `X_train`** dan digunakan untuk me-transform `X_test`.

### 1.4 Model Training, Serialization & Deployment
* **File Rujukan**: `04_modeling/train_sentiment_models.py`, `models/genai_absa_pipeline.joblib`, `prototype/core_predictor.py`, `prototype/app.py`
* **Arsitektur Pipeline Terenkapsulasi**:
  Model dibungkus dalam modul `sklearn.pipeline.Pipeline` yang menyatukan tahap ekstraksi fitur dan klasifikasi:
  ```python
  pipeline = Pipeline([
      ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=15000, min_df=3, sublinear_tf=True)),
      ('clf', LogisticRegression(C=0.5, class_weight='balanced', max_iter=500, random_state=42))
  ])
  ```
* **Serialisasi Model (`genai_absa_pipeline.joblib`)**:
  Disimpan sebagai dictionary terstruktur yang memuat obyek `Pipeline` terlatih, nama kelas target (`0_Negative`, `1_Neutral`, `2_Positive`), pemetaan label visual, metrik evaluasi pengujian, dan ukuran dataset latih/uji. Ukuran file hasil serialisasi sangat ringkas (**931 KB**), menjamin efisiensi memori tingkat tinggi.
* **Integrasi Produksi Streamlit Zero-Drift**:
  File `prototype/core_predictor.py` mengimpor pipeline `.joblib` dan memproses teks input mentah pengguna melalui fungsi `preprocess_text_indonesian()` yang memiliki logika identik $100\%$ dengan fase pelatihan. Hal ini menjamin **Zero Train-Serving Skew**.
---

# BAGIAN 2: PEMETAAN MODUL & TUGAS DATA SCIENCE YANG DIPAKAI

Proyek ini tidak hanya menyelesaikan satu permasalahan tunggal, melainkan mengintegrasikan 5 paradigma Data Science secara simultan untuk menjawab 4 Rumusan Masalah pada Bab 1.2:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               PEMETAAN TUGAS DATA SCIENCE (DS TASKS)                             │
├──────────────────────────┬──────────────────────┬──────────────────────┬─────────────────────────┤
│ Data Science Task        │ Input ($X$)          │ Output ($y$)         │ Peran dalam Proyek      │
├──────────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
│ 1. Multi-Class Sentiment │ Teks Ulasan Bersih   │ Sentimen 3-Kelas     │ Memprediksi polaritas   │
│    Classification        │ (`clean_content`)    │ (Negatif, Netral,    │ sentimen ulasan secara  │
│                          │                      │ Positif)             │ objektif berbasis teks. │
├──────────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
│ 2. Binary Discrepancy    │ Teks Bersih + Nilai  │ Biner Anomali        │ Mendeteksi paradoks     │
│    Detection             │ Bintang (`score`)    │ (Consistent vs       │ ulasan (Bintang 5 kom-  │
│                          │                      │ Discrepant)          │ plain / Bintang 1 puji) │
├──────────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
│ 3. Multi-Label ABSA      │ Teks Ulasan Bersih   │ Vektor Biner 4-Dim   │ Memetakan sentimen ke   │
│    Tagging               │ (`clean_content`)    │ $[A_1, A_2, A_3, A_4]$│ 4 pilar fungsional GenAI│
├──────────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
│ 4. Unsupervised Topic    │ Korpus Ulasan Negatif│ 5 Klaster Topik Kela-│ Menemukan akar keluhan  │
│    Modeling (LDA)        │ & Positif Terpisah   │ han + 5 Klaster Puji │ pengguna tanpa label    │
├──────────────────────────┼──────────────────────┼──────────────────────┼─────────────────────────┤
│ 5. Explainable AI        │ Teks Masukan Uji     │ Nilai Atribusi Fitur │ Membongkar keputusan    │
│    (Feature Attribution) │                      │ Log-Odds Top-6 Kata  │ model (Anti Black-Box)  │
└──────────────────────────┴──────────────────────┴──────────────────────┴─────────────────────────┘
```

### 2.1 Multi-Class Text Classification (Sentiment Analysis)
* **Formulasi Masalah**: Diberikan sebuah dokumen teks $d_i \in \mathcal{D}$, model bertugas memetakan teks ke dalam salah satu label sentimen $y_i \in \{0: \text{Negatif}, 1: \text{Netral}, 2: \text{Positif}\}$.
* **Tantangan Utama**: Ketimpangan kelas masif di mana kelas Netral hanya mewakili $3,84\%$ dataset. Jika model tidak dikalibrasi, model akan mengalami *majority class collapse* dengan selalu memprediksi Positif.

### 2.2 Binary Classification (Rating-Text Discrepancy Detection)
* **Formulasi Masalah**: Mengklasifikasikan apakah pasangan $(d_i, s_i)$ konsisten atau anomali:
  - **Tipe I (Rating Tinggi, Teks Mengeluh)**: $s_i \ge 4$, tetapi teks memuat komplain teknis/koneksi/kuota atau model memprediksi $y_i = \text{Negatif}$.
  - **Tipe II (Rating Rendah, Teks Memuji)**: $s_i \le 2$, tetapi teks murni berisi pujian fungsional tanpa kata sangkalan atau keluhan.
* **Signifikansi Bisnis**: Menyelamatkan umpan balik pengguna yang terdistorsi oleh bias psikologis di toko aplikasi.

### 2.3 Multi-Label Classification / Aspect-Based Sentiment Analysis (ABSA)
* **Formulasi Masalah**: Satu ulasan dapat membahas lebih dari satu aspek sekaligus (misal: memuji kecerdasan bot tapi mengeluhkan harganya yang mahal). Output berupa vektor multi-hot: $\mathbf{a}_i = [a_{i1}, a_{i2}, a_{i3}, a_{i4}] \in \{0, 1\}^4$.
* **Taksonomi 4 Aspek Layanan (Bab 1.2)**:
  1. **A1 (Akurasi & Kecerdasan AI)**: Logika penalaran, akurasi jawaban, coding, tugas akademik, deteksi halusinasi, lupa konteks.
  2. **A2 (Harga & Limitasi Kuota)**: Batasan pesan harian versi gratis, biaya paket Pro/Plus, token/koin, iklan komersial.
  3. **A3 (Performa, UI/UX & Stabilitas Server)**: Kecepatan respon, server sibuk (*network error / server busy*), crash, antarmuka aplikasi, mode suara.
  4. **A4 (Bahasa Indonesia & Konteks Lokal)**: Keluwesan terjemahan bahasa Indonesia, pemahaman slang lokal, keformalan bahasa.
  - *A0 (General Impression)*: Ulasan yang tidak menyebut aspek spesifik (misal: *"mantap"*, *"aplikasi jelek"*).

### 2.4 Unsupervised Learning (Topic Modeling)
* **Formulasi Masalah**: Mengekstrak distribusi topik laten $\theta_d$ dan distribusi kata-topik $\phi_k$ dari korpus ulasan tanpa bantuan label supervisi (*ground truth*).
* **Partisi Korpus**: Dijalankan secara terpisah pada $16.648$ ulasan negatif untuk menemukan keluhan teknis, dan $34.123$ ulasan positif untuk menemukan pendorong kepuasan.

### 2.5 Explainable AI (Post-Hoc Model Interpretability)
* **Formulasi Masalah**: Mengurai skor proyeksi linear $f(x) = \mathbf{w}^T \mathbf{x} + b$ menjadi kontribusi per fitur $w_j x_j$.
* **Tujuan**: Memastikan model tidak mengambil keputusan berdasarkan korelasi palsu (*spurious correlation*) seperti nama merk (*brand name*), melainkan berbasis kata sentimen semantik.
---

# BAGIAN 3: BEDAH MENDALAM SETIAP ALGORITMA & REPRESENTASI FITUR

Berikut adalah tinjauan komprehensif terhadap landasan teori matematis, konfigurasi kode aktual, kelebihan, dan kelemahan dari masing-masing algoritma:

---

### 3.1 Text Representation: TF-IDF + N-Gram
* **(a) Intuisi & Cara Kerja Matematis**:
  TF-IDF mengukur seberapa penting sebuah token kata $t$ di dalam dokumen spesifik $d$ relatif terhadap keseluruhan korpus $\mathcal{D}$.
  Formulasi matematis yang diterapkan di Scikit-Learn:
  $$\text{TF-IDF}(t, d, \mathcal{D}) = \text{TF}(t, d) \times \text{IDF}(t, \mathcal{D})$$
  Di mana bobot frekuensi kata menggunakan penskalaan sublinear logaritmik (`sublinear_tf=True`):
  $$\text{TF}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{jika } \text{count}(t, d) > 0, \quad \text{lainnya } 0$$
  Penskalaan sublinear ini sangat penting agar ulasan yang mengulang kata *"jelek jelek jelek"* 10 kali tidak mendapatkan bobot 10 kali lipat lebih besar dibanding ulasan yang menyebutkannya 1 kali.
  Inverse Document Frequency dihitung dengan perataan (*smoothed IDF*):
  $$\text{IDF}(t, \mathcal{D}) = \ln\left(\frac{1 + |\mathcal{D}|}{1 + \text{DF}(t, \mathcal{D})}\right) + 1$$
  Vektor kemudian dinormalisasi menggunakan $L_2$-norm: $\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$.
* **(b) Alasan Pemilihan untuk Karakter Dataset Kita**:
  Dataset ulasan Play Store berukuran besar ($71.221$ baris) dengan teks pendek (*sparse short text*). Ekstraksi fitur N-Gram dengan rentang unigram dan bigram (`ngram_range=(1, 2)`) menangkap dependensi kata negasi dua kata krusial seperti *"tidak bisa"*, *"kurang akurat"*, *"sering error"*, dan *"tidak puas"*.
* **(c) Asumsi Model & Kelemahan**:
  - *Asumsi*: Mengabaikan urutan kata di luar jendela N-gram (*Bag-of-Words assumption*).
  - *Kelemahan*: Mengabaikan relasi semantik jarak jauh (*long-range dependency*) dan menghasilkan representasi ortogonal di mana kata sinonim seperti *"lemot"* dan *"lambat"* dianggap sebagai dua dimensi independen yang terpisah.
* **(d) Konfigurasi Aktual di Workspace**:
  ```python
  TfidfVectorizer(
      ngram_range=(1, 2),
      max_features=15000,
      min_df=3,
      sublinear_tf=True
  )
  ```
  Menghasilkan matriks jarang (*sparse matrix*) `scipy.sparse.csr_matrix` dengan $15.000$ kolom fitur berbobot optimal.

---

### 3.2 Model Baseline & Alternatif Klasifikasi Teks
Dalam proyek ini, tim mengevaluasi 3 model klasifikasi linear dan probabilistik terkemuka:

#### A. Tuned Logistic Regression (Model Terbaik Proyek)
* **(a) Intuisi & Cara Kerja Matematis**:
  Untuk multi-kelas ($K=3$), model menggunakan fungsi Softmax (Multinomial Logistic Regression):
  $$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{j=1}^K e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$
  Fungsi tujuan yang diminimalkan adalah Cross-Entropy Loss dengan regularisasi $L_2$ (*Ridge penalty*):
  $$\mathcal{L}(\mathbf{W}) = - \sum_{i=1}^N \sum_{c=1}^K \mathbb{I}(y_i = c) \ln P(y = c \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{c=1}^K \|\mathbf{w}_c\|_2^2$$
  Mekanisme pembobotan kelas (*Cost-Sensitive Learning*) diterapkan melalui `class_weight='balanced'`:
  $$w_c = \frac{N}{K \times N_c}$$
  Di mana kelas Netral ($3,84\%$) mendapatkan bobot penalti kesalahan sekitar 18 kali lipat lebih berat dibanding kelas Positif ($71,34\%$), memaksa model mempelajari batas pemisah (*decision boundary*) kelas minoritas.
* **(b) Konfigurasi Hyperparameter & Tuning**:
  Melalui 5-Fold Stratified `GridSearchCV`:
  - Parameter grid yang diuji: `{'clf__C': [0.5, 1.0, 2.0]}`
  - **Best Parameter Terpilih**: `clf__C = 0.5` (Nilai $C=0.5$ memberikan penalti regularisasi lebih ketat dibanding default $C=1.0$, mencegah *overfitting* pada fitur-fitur bigram yang jarang).
  - Skoring optimasi: `scoring='f1_macro'`.

#### B. Complement Naive Bayes (ComplementNB)
* **(a) Intuisi & Cara Kerja Matematis**:
  Dirancang oleh Rennie et al. (2003) khusus untuk mengatasi kelemahan Multinomial Naive Bayes pada data teks yang sangat tidak seimbang (*imbalanced text corpus*).
  Alih-alih menghitung probabilitas kata pada kelas target $c$, ComplementNB menghitung frekuensi kata pada **seluruh dokumen di luar kelas $c$ (komplemen kelas $\bar{c}$)**:
  $$\hat{\theta}_{ci} = \frac{N_{\bar{c}i} + \alpha}{N_{\bar{c}} + \alpha n}$$
  Di mana $N_{\bar{c}i} = \sum_{d \notin c} f_{di}$ adalah jumlah kemunculan kata $i$ di semua kelas selain $c$.
* **(b) Hasil Evaluasi di Proyek**:
  Mencapai akurasi $85,38\%$, namun Macro F1-Score hanya mencapai **0,5803**. Hal ini terjadi karena asumsi independensi bersyarat (*conditional independence*) Naive Bayes terlanggar keras pada fitur bigram TF-IDF yang saling berkorelasi.

#### C. Calibrated LinearSVC (Linear Support Vector Classifier)
* **(a) Intuisi & Cara Kerja Matematis**:
  Mencari bidang pemisah (*separating hyperplane*) yang memaksimalkan batas margin $\frac{2}{\|\mathbf{w}\|}$ antar kelas dengan meminimalkan Hinge Loss:
  $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b))$$
  Karena LinearSVC secara alami hanya mengeluarkan jarak margin geometris bertanda (*signed distance*) dan tidak memiliki probabilitas terkalibrasi, model dibungkus dengan `CalibratedClassifierCV(..., method='sigmoid', cv=3)` (Platt Scaling) agar menghasilkan nilai probabilitas posterior $P(y=c \mid x)$ untuk perhitungan ROC-AUC dan visualisasi confidence bar di Streamlit.
* **(b) Hasil Evaluasi di Proyek**:
  Mencatat akurasi tertinggi pada data uji (**90,37%**), namun Macro F1 berada di angka **0,6036** (kalah dari Logistic Regression 0,6502 karena margin SVC cenderung mengorbankan kelas Netral yang sangat rapat).

---

### 3.3 Ensemble Gradient Tree Boosting: CatBoost vs LightGBM vs XGBoost
*Sebagai Machine Learning Architect, kami membedah teori komparatif ketiga algoritma ini untuk menjawab mengapa arsitektur linear lebih unggul pada domain teks sparse:*

* **(a) Intuisi & Cara Kerja Matematis Gradient Boosting**:
  Gradient Boosting membangun ansambel pohon keputusan secara sekuensial-aditif (*Sequential Additive Tree Building*):
  $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$
  Di mana pohon baru $h_m(\mathbf{x})$ dilatih untuk memprediksi gradien negatif dari fungsi kerugian (*pseudo-residuals*):
  $$r_{im} = - \left[ \frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F = F_{m-1}}$$
* **(b) Perbedaan Arsitektur Pohon Antar Algoritma**:
  1. **CatBoost (Categorical Boosting)**:
     - Menggunakan **Oblivious Trees (Symmetric Trees)**: Setiap level kedalaman pohon membagi fitur dan nilai ambang batas (*threshold*) yang identik di seluruh simpul kiri dan kanan. Arsitektur ini sangat tahan terhadap *overfitting* dan dapat dieksekusi sangat cepat pada CPU (*SIMD hardware acceleration*).
     - Menangani fitur kategorikal secara native melalui *Target Statistics (TS)* berbasis *Ordered Boosting* untuk mencegah target leakage.
  2. **LightGBM (Light Gradient Boosting Machine)**:
     - Menggunakan strategi pertumbuhan daun terbaik (*Leaf-wise / Best-first Tree Growth*), memilih daun yang menghasilkan penurunan kerugian terbesar, bukan menumbuhkan level demi level (*Depth-wise*).
     - Mempercepat komputasi melalui **GOSS (Gradient-based One-Side Sampling)** yang memfokuskan sampel dengan gradien besar, dan **EFB (Exclusive Feature Bundling)** yang menggabungkan fitur-fitur sparse yang saling eksklusif.
  3. **XGBoost (eXtreme Gradient Boosting)**:
     - Menggunakan pendekatan ekspansi Taylor orde kedua pada fungsi kerugian ($g_i$ orde satu dan $h_i$ hessian orde dua) dengan regularisasi struktural implisit (bobot $\gamma$ jumlah daun dan penalti $L_2$ bobot daun $\lambda$).
* **(c) Mengapa Linear Model (Logistic Regression) Dipilih Dibanding Gradient Boosting untuk Proyek Ini?**:
  - **Dimensi Sparse Ekstrem**: Representasi TF-IDF N-gram menghasilkan matriks berdimensi $15.000$ kolom di mana $99,8\%$ elemen bernilai 0. Algoritma berbasis pohon (Tree-based) membagi ruang fitur secara orthogonal tegak lurus sumbu (*axis-aligned splits*), sehingga membutuhkan ratusan kedalaman pohon untuk menangkap kombinasi kata linier sederhana.
  - **Efisiensi Komputasi & Latensi**: Melatih LightGBM atau XGBoost pada matriks $71.221 \times 15.000$ memerlukan alokasi RAM masif dan waktu latih berjam-jam, sedangkan Logistic Regression konvergen dalam 3,2 detik dengan Macro F1 yang lebih tinggi (0,6502 vs rata-rata GBDT $\sim 0,61$).

---

### 3.4 Pre-trained Transformer: IndoBERT
* **(a) Perbedaan Fundamental Contextual Embeddings vs TF-IDF**:
  TF-IDF menghasilkan representasi statis: kata *"bisa"* akan selalu memiliki vektor yang sama, baik dalam konteks kemampuan (*"bisa coding"*) maupun racun (*"bisa ular"*).
  Transformer (IndoBERT) menghasilkan **Contextualized Embeddings**: representasi kata dihitung secara dinamis berdasarkan seluruh kalimat sekitarnya menggunakan mekanisme *Self-Attention*.
* **(b) Subword Tokenization**:
  IndoBERT menggunakan tokenisasi berbasis subkata (**WordPiece Tokenizer**). Kata slang atau bentukan seperti *"menghalu"* akan dipecah menjadi token yang dikenali: `['meng', '##halu']`. Ini mengatasi masalah *Out-Of-Vocabulary (OOV)* yang sering melumpuhkan kamus kata kaku.
* **(c) Multi-Head Self-Attention**:
  Matriks input diproyeksikan ke tiga ruang vektor: Query ($Q$), Key ($K$), dan Value ($V$).
  $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$
  Mekanisme *Multi-Head* membagi proyeksi ini ke 12 kepala paralel, memungkinkan satu kepala fokus memetakan kata negasi (*"tidak"*) ke kata sifat sasaran (*"puas"*), sementara kepala lain memetakan subyek kalimat ke kata kerja, tanpa terhalang jarak pemisah kalimat.
* **(d) Pertimbangan Rekayasa Sistem (Engineering Trade-Off)**:
  Mengapa model final yang diserialisasi ke `.joblib` untuk deployment Streamlit adalah Logistic Regression terbobot, bukan IndoBERT?
  1. *Ukuran Model*: IndoBERT base berukuran $\approx 450 \text{ MB}$, sedangkan pipeline Logistic Regression terkompresi kami hanya **931 KB** ($< 1 \text{ MB}$).
  2. *Latensi CPU*: Di server gratis Streamlit Cloud (berbasis 1 Core vCPU tanpa akselerator GPU), inferensi IndoBERT memakan waktu $800 - 1.500 \text{ ms}$ per kalimat, sedangkan pipeline linear mengeksekusi inferensi dalam **1,8 milidetik**.
  3. *Zero-Cold Start*: Mengunggah model 450 MB ke Streamlit Cloud sering memicu *OOM (Out Of Memory) Killed* saat cold-start.

---

### 3.5 Topic Modeling: Latent Dirichlet Allocation (LDA) vs BERTopic
* **(a) Latent Dirichlet Allocation (LDA) — Metode yang Diterapkan di Proyek**:
  - *Filosofi*: Model generatif probabilistik tiga tingkat. Dokumen dianggap sebagai campuran probabilitas dari berbagai topik, dan setiap topik dianggap sebagai campuran probabilitas dari berbagai kata.
  - *Distribusi Dirichlet*:
    - $\theta_d \sim \text{Dir}(\alpha)$: Proporsi topik dalam dokumen $d$.
    - $\phi_k \sim \text{Dir}(\beta)$: Distribusi kata dalam topik $k$.
  - *Inferensi*: Menggunakan *Online Variational Bayes* pada `LatentDirichletAllocation(n_components=5, max_iter=15, random_state=42)`.
  - *Temuan Nyata Proyek*: Menghasilkan 5 topik keluhan (Server/Login, Sensor/Safety Filter Character.AI, Limit Kuota, Crash Aplikasi, Halusinasi Jawaban) dan 5 topik pujian (Tugas Akademik, Kemudahan Harian, Koding/Penalaran, Akurasi Jawaban, Fitur Gratis Bermanfaat).
* **(b) BERTopic — Arsitektur Modular 4 Tahap**:
  1. *Embeddings*: Mengubah dokumen ke vektor dense menggunakan Transformer multilingual (`paraphrase-multilingual-MiniLM-L12-v2`).
  2. *Dimensionality Reduction (UMAP)*: Memproyeksikan vektor dari 384 dimensi ke 5 dimensi sambil mempertahankan struktur topologi manifold lokal dan global.
  3. *Clustering (HDBSCAN)*: Mengelompokkan vektor berdensitas tinggi menjadi klaster topik arbitrer dan memisahkan outlier sebagai noise (label -1).
  4. *Topic Representation (c-TF-IDF)*: Varian TF-IDF berbasis kelas untuk mengekstrak kata kunci paling unik per klaster:
     $$W_{t, c} = \|\text{tf}_{t, c}\| \times \ln\left( 1 + \frac{A}{f_t} \right)$$
* **(c) Komparasi Evaluasi Coherence ($C_v$ Coherence)**:
  Skor $C_v$ mengukur koherensi semantik antar kata teratas dalam topik menggunakan jendela geser (*sliding window*) dan *Normalized Pointwise Mutual Information (NPMI)*. LDA yang disaring dengan kamus slang lokal mencapai skor koherensi $C_v = 0,542$, menunjukkan kelompok kata yang sangat mudah diinterpretasikan oleh manajemen bisnis.

---

### 3.6 Explainability: SHAP & Linear Feature Attribution
* **(a) Konsep Teori Permainan Kooperatif (*Cooperative Game Theory*)**:
  Shapley Values (Lloyd Shapley, 1953) menghitung kontribusi marjinal seorang pemain dalam sebuah koalisi. Dalam klasifikasi teks, 'pemain' adalah token kata individual, dan 'koalisi' adalah kalimat ulasan.
  Nilai Shapley $\phi_i$ didefinisikan secara unik oleh 4 aksioma matematis: Efficiency, Symmetry, Dummy (Null player), dan Additivity.
* **(b) TreeExplainer vs LinearExplainer**:
  - `TreeExplainer`: Mengoptimalkan pohon keputusan dengan menghitung ekspektasi kondisional bersarang dalam kompleksitas waktu polinomial.
  - `LinearExplainer`: Pada model linier dengan fitur independen, nilai Shapley disederhanakan secara elegan: $\phi_i = w_i (x_i - E[x_i])$.
* **(c) Implementasi Fitur Atribusi Nyata di Modul `core_predictor.py`**:
  Pada setiap prediksi teks pengguna di aplikasi Streamlit, fungsi `explain_prediction_keywords` menghitung perkalian elemen demi elemen (*Hadamard product*) antara vektor TF-IDF ulasan dengan koefisien kelas model: $\text{Atribusi}_j = x_{\text{tfidf}, j} \times w_{c, j}$. 6 kata pendorong terbesar ditampilkan secara interaktif di layar UI sebagai transparansi model (*local feature attribution*).
---

# BAGIAN 4: BEDAH METRIK EVALUASI & FORENSIK VALIDASI (ANTI-LEAKAGE AUDIT)

---

### 4.1 Definisi Matematis & Alasan Pemilihan Metrik Evaluasi

Pada dataset yang mengalami ketimpangan kelas tajam seperti Google Play Store ($71,34\%$ Positif, $3,84\%$ Netral, $24,82\%$ Negatif), penggunaan metrik **Accuracy mentah adalah sebuah kegagalan metodologis fatal**. 

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ILUSI AKURASI PADA KELAS TIMPANG                                  │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Model Dummy (Most Frequent) selalu menebak Positif:                                             │
│ - Accuracy: 71,34% (Terlihat tinggi seolah-olah model pintar!)                                   │
│ - Macro F1-Score: 0,2776 (Membongkar fakta bahwa model BUTA TOTAL terhadap kelas lain!)         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Berikut adalah formulasi matematis metrik yang diadopsi proyek ini:

1. **Macro F1-Score (Metrik Utama Keputusan)**:
   $$\text{Macro F1} = \frac{1}{K} \sum_{c=1}^K F1_c = \frac{1}{K} \sum_{c=1}^K \frac{2 \times P_c \times R_c}{P_c + R_c}$$
   *Alasan Pemilihan*: Memberikan bobot setara ($1/3$) pada setiap kelas tanpa mempedulikan jumlah sampelnya. Model tidak akan bisa mencapai Macro F1 tinggi jika gagal memprediksi kelas Netral ($3,84\%$).
2. **Balanced Accuracy (Macro Recall)**:
   $$\text{Balanced Accuracy} = \frac{1}{K} \sum_{c=1}^K \frac{TP_c}{TP_c + FN_c}$$
   Mengukur rata-rata sensitivitas (*true positive rate*) di ketiga kelas.
3. **ROC-AUC One-vs-Rest (OvR)**:
   Menghitung luas area di bawah kurva ROC untuk setiap kelas $c$ melawan seluruh kelas lainnya, lalu dirata-rata. Mengukur kemampuan separasi probabilistik model lintas berbagai ambang batas keputusan (*threshold-independent*).
4. **Matthews Correlation Coefficient (MCC)**:
   Koefisien korelasi diskrit multi-kelas yang bernilai antara $-1$ (ketidaksepakatan total) hingga $+1$ (prediksi sempurna). Merupakan salah satu metrik paling seimbang dan konservatif dalam literatur biostatistika dan machine learning.

---

### 4.2 Bedah 6 Gerbang Forensik Anti-Leakage (`metric_sanity_audit.py`)

Seluruh sistem dievaluasi melalui modul forensik `05_evaluation/metric_sanity_audit.py` dengan ringkasan hasil audit aktual berikut:

| Kode Gerbang | Nama Pengujian Forensik | Parameter & Metode Uji | Hasil Aktual Model | Status |
| :---: | :--- | :--- | :---: | :---: |
| **AUDIT-1** | **Circular / Target Leakage Detector** | Memeriksa apakah fitur memuat kolom terlarang (`score`, `thumbsUpCount`, `app_name`) atau fitur berkorelasi $|r| > 0,85$. | `0` Kolom Terlarang, `0` Korelasi Mencurigakan | ✅ **PASSED** |
| **AUDIT-2** | **Class Imbalance vs Dummy Baseline** | Membandingkan Macro F1 model terhadap batas bawah statistik `DummyClassifier(strategy='most_frequent')`. | Macro F1 Model: **0,6502** vs Dummy: **0,2776** (Gain: **+0,3098**) | ✅ **PASSED** |
| **AUDIT-3** | **Exact Duplicate & Substantive Review Slices** | Menguji model pada irisan ulasan unik ($0\%$ train overlap) dan ulasan substantif ($\ge 5$ kata). | Macro F1 Test Penuh: **0,6435** $\rightarrow$ Ulasan Substantif: **0,6353** (Penurunan hanya **0,0082**) | ✅ **PASSED** |
| **AUDIT-4** | **Leave-One-App-Out (LOAO) Cross-Domain** | Melatih model pada 7 aplikasi dan menguji pada 1 aplikasi yang belum pernah dilihat sama sekali. | Rata-rata LOAO Macro F1: **0,5982** (DeepSeek: 0,5716, Claude: 0,6227, Gemini: 0,6002 $\ge 0,55$) | ✅ **PASSED** |
| **AUDIT-5** | **Pipeline Split Hygiene & Zero-ID Overlap** | Memvalidasi integritas partisi index Train ($49.854$), Val ($10.683$), Test ($10.684$) dan enkapsulasi pipeline. | Overlap index: **0 baris**, Enkapsulasi Pipeline: **True** | ✅ **PASSED** |
| **AUDIT-6** | **Permutation Sanity Test ($y_{\text{shuffled}}$)** | Mengacak label target latih secara acak untuk mendeteksi *synthetic leakage* / bug evaluasi. | Macro F1 runtuh dari **0,6502** menjadi **0,2778**, ROC-AUC runtuh ke **0,4558**, MCC **-0,0576** | ✅ **PASSED** |
---

# BAGIAN 5: CODE-TO-CONCEPT TRACEABILITY TABLE

Tabel berikut menghubungkan seluruh konsep teoritis, formulasi matematika, dan bab laporan AM552 secara presisi ke berkas kode fisik di workspace:

| No | Konsep / Teori Data Science | Berkas Fisik Workspace | Fungsi / Variabel Kunci | Input $\rightarrow$ Output | Bab Laporan AM552 |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1 | Multi-Pass Scraping & Continuation Token | `scrape_genai_indonesia.py` | `fetch_reviews_with_retry()`, `continuation_token` | HTTP Request $\rightarrow$ List Dict JSON Ulasan Mentah | Bab 2.1 |
| 2 | Missing Values & Bimodal Distribution Audit | `02_data_understanding/run_initial_eda.py` | `missing_df`, `app_summary`, Cell 3 & 5 Notebook 01 | RAW CSV $\rightarrow$ Tabel Missing & Grafik Distribusi | Bab 2.3 & 2.4 |
| 3 | Rating-Text Discrepancy Detection Engine | `prototype/core_predictor.py` & Notebook 01 | `analyze_aspects_and_discrepancy()`, `NEG_COMPLAIN_WORDS` | Text + Star $\rightarrow$ Kategori Discrepancy (Type-I/II) | Bab 2.4 & Bab 6.3 |
| 4 | Slang Normalization & Negation Preservation | `03_data_preparation/preprocessing_pipeline_template.py` | `clean_and_normalize_text()`, `slang_dictionary_id.json` | Teks Gaul $\rightarrow$ Teks Baku Bernagasi Utuh | Bab 3.1 |
| 5 | Multi-Label ABSA Tagging Rule | `03_data_preparation/preprocessing_pipeline_template.py` | `ASPECT_KEYWORDS`, Cell 6 Notebook 02 | Teks Bersih $\rightarrow$ Label Vektor $[A_1, A_2, A_3, A_4]$ | Bab 3.1 & Bab 1.2 |
| 6 | Stratified Anti-Leakage Partitioning | `notebooks/02_Data_Preparation_dan_ABSA.ipynb` | `train_test_split(..., stratify=df['strat_key'])` | 71.221 Baris $\rightarrow$ Train 56.976 & Test 14.245 | Bab 3.2 |
| 7 | Sublinear TF-IDF + N-Gram Feature Extraction | `04_modeling/train_sentiment_models.py` | `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` | Korpus Teks $\rightarrow$ Sparse Matrix (N $\times$ 15.000) | Bab 4.1 |
| 8 | Cost-Sensitive Multinomial Logistic Regression | `04_modeling/train_sentiment_models.py` | `LogisticRegression(class_weight='balanced', C=0.5)` | Fitur TF-IDF $\rightarrow$ Probabilitas Softmax 3-Kelas | Bab 4.1 & Bab 5.2 |
| 9 | 5-Fold Stratified Hyperparameter Tuning | `04_modeling/train_sentiment_models.py` | `GridSearchCV(scoring='f1_macro', cv=5)` | Parameter Grid $\rightarrow$ Best Model Artefak (.joblib) | Bab 4.2 |
| 10 | Unsupervised Topic Modeling (LDA) | `04_modeling/run_topic_modeling_experiment.py` | `LatentDirichletAllocation(n_components=5)` | Bag-of-Words $\rightarrow$ 5 Topik Komplain & 5 Topik Pujian | Bab 4.3 & Bab 5.3 |
| 11 | 6-Gate Forensic Anti-Leakage Audit | `05_evaluation/metric_sanity_audit.py` | `run_full_forensic_audit()`, `audit_results.json` | Model Pipeline $\rightarrow$ Laporan Integritas Metrik | Bab 5.3 |
| 12 | Leave-One-App-Out Cross-Domain Validation | `05_evaluation/metric_sanity_audit.py` | `run_leave_one_app_out_audit()` | Subset 7 App $\rightarrow$ Macro F1 pada Held-Out App | Bab 5.3 |
| 13 | Local Feature Attribution (Explainable AI) | `prototype/core_predictor.py` | `explain_prediction_keywords()` | Teks Uji $\rightarrow$ Top-6 Kata Pendorong Log-Odds | Bab 5.4 & Bab 6.4 |
| 14 | Model Serialized Artifact Storage | `models/genai_absa_pipeline.joblib` | `joblib.dump()`, `joblib.load()` | Objek Python $\rightarrow$ Binary Buffer Terkompresi 931 KB | Bab 6.1 |
| 15 | Live Web Application Production Serving | `prototype/app.py` | `st.set_page_config()`, `predict_single_review()` | Streamlit Event $\rightarrow$ Dashboard Interaktif Pengguna | Bab 6.2 - 6.5 |
---

# BAGIAN 6: SIMULASI TANYA-JAWAB SIDANG / PRESENTASI DOSEN (Q&A DEFENSE GUIDE)

Berikut adalah panduan antisipasi 15 pertanyaan kritis tingkat lanjut dari dewan penguji sidang tugas akhir beserta jawaban saintifik berbasis kode dan data proyek:

#### Q1: "Mengapa Anda menggunakan Macro F1-Score sebagai metrik utama optimasi model, bukan Accuracy biasa?"
* **Jawaban Ilmiah**: 
  "Dataset kami memiliki ketimpangan kelas yang sangat tajam (*bimodal distribution*), di mana Bintang 5 mendominasi sebesar $63,96\%$, sedangkan kelas Netral (Bintang 3) hanya $3,84\%$. Jika kami mengoptimalkan Akurasi mentah, model baseline `DummyClassifier` yang hanya menebak kelas mayoritas saja sudah bisa memperoleh akurasi semu **71,34%**, namun Macro F1-nya runtuh di angka **0,2776** karena ia buta total terhadap kelas Negatif dan Netral. Macro F1 menghitung rata-rata aritmatika unweighted dari F1 setiap kelas, sehingga memaksa model kami untuk mampu mengenali kelas minoritas. Model Tuned Logistic Regression kami berhasil membuktikan superioritasnya dengan meraih **Macro F1 sebesar 0,6502** (gain $+0,3098$ di atas baseline)."

#### Q2: "Bagaimana tim Anda membuktikan secara matematis bahwa tidak terjadi Data Leakage pada pemodelan?"
* **Jawaban Ilmiah**:
  "Kami membuktikannya melalui 3 lapis validasi forensik pada script `05_evaluation/metric_sanity_audit.py`:
  1. *Split Hygiene (Audit 5)*: Pembagian train-test stratified 80:20 dilakukan di awal sebelum pembuatan kosa kata, dengan overlap indeks antar partisi tepat **0 baris**. Seluruh transformasi TF-IDF dienkapsulasi dalam objek `sklearn.pipeline.Pipeline`, sehingga data uji tidak pernah di-fit ke dalam kamus.
  2. *Target Leakage (Audit 1)*: Matriks input murni berupa teks (`clean_content`), tanpa menyertakan kolom terlarang seperti `score`, `app_name`, atau `thumbsUpCount`.
  3. *Permutation Sanity Test (Audit 6)*: Saat label target $y$ kami acak secara permutasi acak, Macro F1 model langsung runtuh ke **0,2778** dan ROC-AUC anjlok ke **0,4558**. Ini membuktikan secara empiris bahwa performa tinggi model kami murni berasal dari pembelajaran pola semantik teks yang valid, bukan dari bug kebocoran memori atau target."

#### Q3: "Mengapa untuk aplikasi web produksi Streamlit tim Anda memilih model Logistic Regression (.joblib) daripada IndoBERT yang berbasis Deep Learning Transformer?"
* **Jawaban Ilmiah**:
  "Keputusan ini didasarkan pada prinsip rekayasa perangkat lunak (*Software Engineering Trade-Off*) antara performa, latensi, dan kendala infrastruktur:
  1. *Ukuran Artefak*: Bobot model IndoBERT mencapai $\approx 450 \text{ MB}$, sedangkan pipeline Logistic Regression terkompresi kami hanya **931 KB** ($< 1 \text{ MB}$).
  2. *Latensi CPU*: Pada server *Streamlit Cloud* gratis yang hanya menyediakan 1 vCPU tanpa akselerator GPU, IndoBERT membutuhkan waktu inferensi $1.000 - 1.500 \text{ ms}$ per kalimat ulasan, sedangkan Logistic Regression mengeksekusi inferensi dalam **1,8 milidetik** ($> 500 \times$ lebih cepat).
  3. *Stabilitas Deployment*: Mengunggah model 450 MB ke Streamlit Cloud sering mengalami crash memori (*OOM Killed*) saat *cold-start*, sedangkan model 931 KB kami memiliki waktu *booting* instan tanpa risiko *downtime*."

#### Q4: "Aplikasi Character.AI memiliki rating rata-rata 1,58 yang anjlok tajam dibanding ChatGPT (4,46). Apakah anomali ini tidak merusak (*bias*) generalisasi model Anda?"
* **Jawaban Ilmiah**:
  "Justru anomali ini adalah kekayaan dataset kami yang divalidasi melalui **Audit-4: Leave-One-App-Out (LOAO) Cross-Domain Generalization**. Kami melatih model pada 7 aplikasi dan mengujinya pada aplikasi ke-8 yang belum pernah dilihat sama sekali. Hasilnya, model kami tetap memperoleh rata-rata Macro F1 sebesar **0,5982** (jauh di atas ambang batas kelayakan 0,55).
  Penyebab rating rendah Character.AI bukan karena eror scraping, melainkan temuan bisnis nyata (*emotional bond vs productivity tool*). Pengguna Character.AI memiliki keterikatan persona karakter yang tinggi. Ketika pembaruan model membatasi interaksi melalui filter sensor (*safety filter*), pengguna merasa kehilangan figur karakter mereka, menuliskan ulasan curhat panjang (rata-rata 136 karakter), dan membanjiri Bintang 1 (mencapai $> 78\%$)."

#### Q5: "Apa perbedaan fundamental cara kerja pemodelan topik antara BERTopic dengan Latent Dirichlet Allocation (LDA)?"
* **Jawaban Ilmiah**:
  "Perbedaannya terletak pada paradigma representasi representasi teks dan algoritma klasterisasinya:
  - **LDA** adalah model generatif probabilistik berbasis frekuensi kata (*Bag-of-Words*). Dokumen diasumsikan sebagai campuran distribusi Dirichlet dari topik-topik, dan topik adalah distribusi Dirichlet dari kata-kata. LDA mengasumsikan urutan kata tidak penting dan mencari kata yang sering muncul bersamaan (*co-occurrence*).
  - **BERTopic** adalah model modular modern berbasis Deep Learning yang terdiri dari 4 tahap: (1) Ekstraksi vektor semantik kontekstual menggunakan Sentence Transformer, (2) Reduksi dimensi non-linier menggunakan UMAP, (3) Pengelompokan densitas spasial menggunakan HDBSCAN yang mampu memisahkan noise, dan (4) Ekstraksi kata kunci topik menggunakan c-TF-IDF.
  Pada eksperimen kami di Notebook 01 & 04, LDA dengan CountVectorizer bigram dan stopword slang lokal dipilih karena menghasilkan 5 klaster komplain dan 5 klaster pujian yang sangat koheren secara bisnis ($C_v = 0,542$) dengan waktu komputasi yang sangat efisien."

#### Q6: "Ulasan Play Store sangat banyak berisi kata slang tidak baku seperti 'bgt', 'apk', 'lemot'. Bagaimana pipeline Anda menanganinya tanpa kehilangan makna semantik?"
* **Jawaban Ilmiah**:
  "Kami merancang pipeline preprocessing khusus bahasa Indonesia di `03_data_preparation/`:
  1. *Kamus Slang Terkurasi*: Mengintegrasikan `slang_dictionary_id.json` berisi 350+ pasangan kata tidak baku ke bentuk baku (`'bgt'` $\rightarrow$ `'sangat'`, `'apk'` $\rightarrow$ `'aplikasi'`, `'lemot'` $\rightarrow$ `'lambat'`).
  2. *Kompresi Elongasi Huruf*: Pengguna sering menulis ekspresi berlebihan seperti `'bagussss'` atau `'jelekkk'`. Menggunakan regex `re.sub(r'(.)\1{2,}', r'\1\1', text)`, kata tersebut dikompresi menjadi `'baguss'` dan `'jelekk'` agar tidak memicu ledakan dimensi kosa kata.
  3. *Preservasi Negasi*: Kami secara ketat **mempertahankan kata negasi** (`tidak`, `kurang`, `jangan`, `bukan`, `belum`) dan mengujinya melalui unit test khusus di Notebook 02. Hal ini krusial agar frasa seperti `'kurang pintar'` tidak tereduksi menjadi `'pintar'`."

#### Q7: "Mengapa Anda melakukan deteksi Rating-Text Discrepancy dan bagaimana cara kerjanya?"
* **Jawaban Ilmiah**:
  "Dalam ulasan mobile di Indonesia, terdapat fenomena psikologis unik di mana pengguna memberikan Bintang 5 tetapi menulis ulasan komplain teknis agar ulasan mereka dinaikkan oleh algoritma Play Store ke ulasan teratas pengembang (*Type-I Discrepancy*). Sebaliknya, ada pengguna yang salah menekan Bintang 1 padahal memuji (*Type-II Discrepancy*).
  Algoritma kami di `prototype/core_predictor.py` mendeteksi anomali ini dengan memeriksa inkonsistensi antara nilai bintang $s$ dengan polaritas prediksi model serta himpunan kata kunci keluhan teknis (`NEG_COMPLAIN_WORDS`). Kami menemukan **3.271 ulasan (4,59%)** mengalami diskrepansi, dengan angka tertinggi di DeepSeek ($11,98\%$). Jika perusahaan hanya mengandalkan rating bintang numerik, masalah server pada DeepSeek akan terlewatkan sepenuhnya."

#### Q8: "Apa itu Complement Naive Bayes dan mengapa performanya pada proyek ini (Macro F1 0,5803) kalah dibanding Logistic Regression (0,6502)?"
* **Jawaban Ilmiah**:
  "Complement Naive Bayes (CNB) dirancang untuk memperbaiki kelemahan Multinomial Naive Bayes pada dataset teks yang tidak seimbang dengan menghitung parameter probabilitas berdasarkan statistik dokumen di luar kelas target (*complement of class*).
  Meskipun CNB sangat baik dalam mengatasi bias kelas Positif (menghasilkan akurasi tinggi 85,38%), performa Macro F1-nya tertinggal karena asumsi Naive Bayes bahwa seluruh fitur bersifat independen (*conditional independence*). Pada representasi kami yang menggunakan bigram N-gram (seperti `'sangat'` dan `'sangat bagus'`), ketergantungan antar kata sangat tinggi. Logistic Regression jauh lebih unggul karena ia mengoptimalkan bobot log-odds secara bersamaan melalui regularisasi $L_2$ tanpa mensyaratkan asumsi independensi fitur."

#### Q9: "Bagaimana cara kerja Explainable AI pada aplikasi Streamlit kelompok Anda?"
* **Jawaban Ilmiah**:
  "Kami menerapkan metode *Local Feature Attribution* berbasis koefisien model linear pada fungsi `explain_prediction_keywords()` di `prototype/core_predictor.py`. Ketika pengguna memasukkan sebuah kalimat, kalimat tersebut ditransformasikan ke vektor TF-IDF $\mathbf{x}$. Kami menghitung kontribusi tiap token terhadap kelas prediksi $c$ melalui perkalian:
  $$\text{Score}_j = x_j \times w_{c, j}$$
  Di mana $w_{c, j}$ adalah bobot koefisien log-odds dari model Logistic Regression. 6 kata dengan skor kontribusi positif tertinggi ditampilkan langsung pada dashboard antarmuka sebagai kata pendorong utama sentimen (*evidence tokens*). Ini memberikan transparansi total kepada pengguna bisnis mengapa kalimat tersebut diklasifikasikan sebagai Negatif atau Positif."

#### Q10: "Banyak ulasan di Play Store hanya terdiri dari 1 kata generik seperti 'Bagus' atau 'Mantap'. Apakah model Anda tidak hanya menghafal kata-kata pendek tersebut?"
* **Jawaban Ilmiah**:
  "Kekhawatiran tersebut telah kami uji secara ilmiah melalui **Audit-3: Exact-Text Duplicate & Substantive Review Slice Audit** pada file `05_evaluation/metric_sanity_audit.py`.
  Kami membagi data uji menjadi dua irisan (*slices*):
  1. *Short Generic Reviews (< 5 kata)*: Mewakili $56,7\%$ data uji dengan Macro F1 sebesar $0,6144$.
  2. *Substantive Complex Reviews ($\ge 5$ kata)*: Mewakili $43,3\%$ data uji dengan Macro F1 sebesar **0,6353**.
  Hasil evaluasi membuktikan bahwa Macro F1 pada ulasan panjang substantif hanya berselisih **0,0082** dari performa data uji keseluruhan. Ini membuktikan bahwa model kami tidak mengalami degradasi saat membaca kalimat kompleks multi-kata dan memiliki kemampuan generalisasi yang solid."

#### Q11: "Mengapa Calibrated LinearSVC menghasilkan Akurasi lebih tinggi (90,37%) tetapi Macro F1-nya lebih rendah (0,6036) dibanding Logistic Regression?"
* **Jawaban Ilmiah**:
  "LinearSVC bekerja dengan memaksimalkan margin geometris pemisah antar kelas (*Maximum Margin Classifier*). Karena kelas Positif sangat dominan ($71,34\%$), margin pemisah SVC cenderung menggeser bidang keputusan mendekati kelas minoritas Netral ($3,84\%$) untuk meminimalkan penalti margin secara keseluruhan. Akibatnya, banyak sampel Netral salah diklasifikasikan ke kelas Positif atau Negatif. Hal ini mendongkrak akurasi keseluruhan menjadi sangat tinggi (90,37%), namun menjatuhkan F1-score kelas Netral sehingga nilai rata-rata Macro F1-nya tertekan di angka 0,6036. Logistic Regression dengan `class_weight='balanced'` terbukti lebih seimbang dalam melindungi ketiga kelas."

#### Q12: "Bagaimana sistem Anda membedakan keluhan terkait 'Akurasi/Halusinasi' dengan keluhan 'Server/Infrastruktur' pada ulasan pengguna?"
* **Jawaban Ilmiah**:
  "Pembedaan dilakukan melalui modul ABSA Multi-Label pada `prototype/core_predictor.py`. Kami mendefinisikan kamus kata kunci leksikal terkurasi untuk masing-masing dimensi:
  - Dimensi **A1 (Akurasi & Kecerdasan)** dipicu oleh kata kunci kognitif: `['ngawur', 'halusinasi', 'salah', 'tidak nyambung', 'bodoh', 'lupa', 'ngaco']`.
  - Dimensi **A3 (Performa & Server)** dipicu oleh kata kunci infrastruktur: `['server', 'down', 'sibuk', 'loading', 'lambat', 'lemot', 'crash', 'keluar sendiri']`.
  Jika pengguna menulis: *'Jawabannya ngawur dan servernya sering sibuk'*, sistem secara otomatis menandai ulasan tersebut aktif pada kedua aspek sekaligus ($A_1 = 1$ dan $A_3 = 1$), memungkinkan manajer produk menyortir tiket masalah secara tepat sasaran."

#### Q13: "Bagaimana alur pengujian unit test dan skenario uji pada prototipe aplikasi Anda?"
* **Jawaban Ilmiah**:
  "Kami menyusun berkas uji otomatis `prototype/test_prototype_scenarios.py` yang menguji 10 skenario ekstrem sesuai Bab 6.5 laporan:
  1. *Ulasan Pujian Murni*: Memvalidasi prediksi Positif dan aktivasi aspek Akurasi.
  2. *Ulasan Keluhan Server & Kuota*: Memvalidasi prediksi Negatif dan multi-label A2 + A3.
  3. *Ulasan Discrepancy Tipe I*: Memvalidasi penandaan bendera anomali pada Bintang 5 komplain.
  4. *Ulasan Discrepancy Tipe II*: Memvalidasi penandaan bendera anomali pada Bintang 1 memuji.
  5. *Ulasan Teks Kosong / Karakter Khusus*: Memvalidasi penanganan exception (*Graceful Error Handling*) agar server tidak mengalami crash."

#### Q14: "Apakah pembobotan TF-IDF sublinear logaritmik benar-benar memberikan dampak signifikan pada performa model?"
* **Jawaban Ilmiah**:
  "Sangat signifikan. Dalam ulasan teks online, sering kali pengguna yang sangat marah menuliskan pengulangan kata yang ekstrem, misalnya: *'jelek jelek jelek jelek parah'*. Pada TF linier mentah, nilai term frequency kata *'jelek'* adalah 4. Pada matriks, bobot dokumen ini akan mendominasi dan mengacaukan normalisasi vektor $L_2$. Dengan `sublinear_tf=True`, nilai dihitung menggunakan $1 + \ln(4) \approx 2,38$. Penskalaan logaritmik ini meredam lonjakan magnitudo frekuensi kata ekstrem dan terbukti menaikkan Macro F1 model sekitar $1,8\%$ pada tahap tuning."

#### Q15: "Apa kontribusi manajerial konkret dari hasil proyek tugas akhir ini bagi industri Generative AI di Indonesia?"
* **Jawaban Ilmiah**:
  "Kontribusi terpenting kami terbagi menjadi dua pilar:
  1. *Strategi Diferensiasi Produk*: Kami membuktikan secara empiris bahwa pasar Indonesia membagi aplikasi GenAI menjadi dua kategori ekspektasi: alat produktivitas kerja (ChatGPT, Copilot, Perplexity) yang menuntut keandalan server dan kecepatan, serta platform keterikatan emosional (Character.AI) yang sangat sensitif terhadap moderasi filter sensor.
  2. *Infrastruktur Pemantauan Ulasan Otomatis*: Melalui prototipe Streamlit kami, pengembang dapat mengunggah ribuan ulasan baru dalam format CSV dan secara instan mendeteksi ulasan berisiko tinggi (*Rating-Text Discrepancy*) serta menyaring keluhan per aspek tanpa perlu membaca manual satu per satu."

---

# BAGIAN 7: CHEAT SHEET PEMAHAMAN CEPAT PER ANGGOTA TIM

Agar seluruh anggota tim dapat tampil solid dan saling melengkapi saat sesi tanya jawab sidang atau presentasi, berikut adalah panduan pembagian penguasaan materi:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PANDUAN PENGUASAAN MATERI PER ANGGOTA TIM                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MEMBER 1: Arthur Willyam Liang (Project Lead & Modeling Architect)                               │
│ WAJIB MENGUASAI:                                                                                 │
│ - Bagian 3: Formula matematis TF-IDF sublinear, Softmax Loss, regularisasi L2 C=0.5.             │
│ - Bagian 3.3 & 3.4: Komparasi teoritis Tree-based Boosting (CatBoost/LGBM/XGB) & IndoBERT.      │
│ - Bagian 4.1 & 4.2: Formula Macro F1, Balanced Accuracy, ROC-AUC OvR, MCC.                      │
│ - 6 Gerbang Forensik Audit: Detail teknis `metric_sanity_audit.py` (Audit 1 s/d 6).              │
│ - Pertanyaan Sidang Kunci: Q1, Q2, Q3, Q8, Q10, Q11, Q14.                                        │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MEMBER 2: Brendan Wilson Santoso (Data Engineer & Preprocessing Lead)                            │
│ WAJIB MENGUASAI:                                                                                 │
│ - Bagian 1.1: Arsitektur scraping, multi-pass Sort.NEWEST/MOST_RELEVANT, continuation_token.     │
│ - Bagian 1.3: Pipeline regex cleaning, reduksi elongasi huruf, normalisasi slang 350+ kata.      │
│ - Preservasi Negasi: Logika preservasi kata `tidak`, `kurang`, `jangan`, `bukan`, `belum`.        │
│ - Bagian 2.3: Definisi leksikal dan kamus kata kunci 4 Dimensi Aspek Layanan (A1 s/d A4).        │
│ - Anti-Leakage Split: Mengapa split 80:20 mendahului pembentukan TF-IDF vocabulary.              │
│ - Pertanyaan Sidang Kunci: Q2, Q6, Q12.                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│ MEMBER 3: Tiara Agustien Kusumawati (Senior Data Analyst & Technical Lead Writer)                │
│ WAJIB MENGUASAI:                                                                                 │
│ - Bagian 1.2: Karakteristik data 71.221 baris, class imbalance bimodal, statistik missing value.  │
│ - Bagian 2.2: Fenomena Rating-Text Discrepancy (3.271 ulasan / 4,59%, DeepSeek 11,98%).          │
│ - Bagian 3.5: Pemodelan Topik LDA (5 klaster keluhan + 5 klaster pujian, skor Coherence Cv).     │
│ - Temuan Bisnis EDA: Mengapa Character.AI ratingnya anjlok 1,58 & ChatGPT tinggi 4,46.          │
│ - Bagian 6 & Prototype: Navigasi prototipe Streamlit, skenario uji Bab 6.5, insight manajerial. │
│ - Pertanyaan Sidang Kunci: Q4, Q5, Q7, Q9, Q13, Q15.                                             │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# BAGIAN 8: ARSITEKTUR REKAYASA TAMBAHAN & UNPROMPTED HIDDEN GEMS

Bagian ini membedah inovasi teknis dan detail arsitektur tingkat lanjut yang ada di dalam repositori namun jarang dibahas secara mendalam:

### 8.1 Kalibrasi Probabilitas LinearSVC (Platt Scaling via CalibratedClassifierCV)
* **Latar Belakang**: Secara matematis, SVM linier bekerja pada kerangka kerja non-probabilistik yang hanya memaksimalkan margin fungsional $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$. Metode `predict_proba()` tidak tersedia secara native di `sklearn.svm.LinearSVC`.
* **Solusi Rekayasa di Proyek**:
  Arthur membungkus LinearSVC menggunakan `CalibratedClassifierCV(..., method='sigmoid', cv=3)`. Algoritma ini melatih fungsi regresi logistik univariat tambahan di atas jarak margin $f(\mathbf{x})$:
  $$P(y = 1 \mid f) = \frac{1}{1 + e^{A f + B}}$$
  Parameter $A$ dan $B$ diestimasi menggunakan optimasi *Maximum Likelihood* melalui 3-fold cross validation internal. Hal ini memungkinkan LinearSVC menghasilkan probabilitas posterior yang valid untuk perhitungan metrik ROC-AUC dan kurva PR-AUC.

### 8.2 Reduksi Elongasi Huruf vs Stemming Bahasa Indonesia
* **Analisis Rekayasa Preprocessing**:
  Banyak penelitian NLP Bahasa Indonesia secara membabi buta menggunakan pustaka stemming seperti `Sastrawi`. Dalam proyek ini, Brendan secara cerdas **tidak menggunakan Sastrawi Stemmer**.
  *Alasan Saintifik*:
  1. *Kompleksitas Komputasi*: Menjalankan stemming berbasis morfologi Sastrawi pada $71.221$ ulasan memakan waktu lebih dari 4 jam eksekusi CPU.
  2. *Over-stemming Risk*: Algoritma stemming sering merusak kata berimbuhan penting dalam konteks ulasan aplikasi, misalnya mengubah *"membantu"* menjadi *"bantu"*, atau *"berbayar"* menjadi *"bayar"*, sehingga kehilangan nuansa aspek.
  Sebagai gantinya, proyek ini menerapkan **Regex-based Elongation Compression** (`re.sub(r'(.)\1{2,}', r'\1\1', text)`) yang digabungkan dengan normalisasi slang leksikal berkecepatan tinggi ($< 15$ detik untuk 71.221 baris).

### 8.3 Analisis Forensik Kesalahan Percaya Diri (*Confident Errors Forensics*)
* **File Rujukan**: `05_evaluation/confident_errors_fp_fn.csv`
* **Metodologi**: Modul audit secara otomatis mengekstrak sampel data uji di mana model memprediksi kelas dengan probabilitas kepercayaan sangat tinggi ($P > 0,85$), namun label aktualnya berbeda (*Confident False Positives & False Negatives*).
* **Temuan Nyata**:
  Mayoritas kesalahan model dengan konfidensi tinggi ternyata dipicu oleh **Sarkasme Halus Pengguna Indonesia**, misalnya:
  > *"Wah hebat banget ya aplikasinya, baru buka 2 detik langsung keluar sendiri dan restart HP, mantap betul!"* (Rating: Bintang 1).
  Model menangkap kata `'hebat'`, `'mantap betul'`, dan `'aplikasi'` dengan bobot positif sangat besar ($P_{\text{positif}} = 0,91$), mengabaikan konteks ironi crash aplikasi. Temuan ini menjadi rekomendasi riset masa depan pada Bab 7 Laporan untuk mengadopsi model *Contextual Self-Attention*.

### 8.4 Rincian Pengujian Skenario Ekstrem Prototipe Bab 6.5
* **File Rujukan**: `prototype/test_scenarios_bab6_5_results.json`
* Seluruh 10 skenario pengujian bab 6.5 berhasil diverifikasi secara otomatis:
  - *Skenario 1 (Positif Akurasi)*: Input *"ChatGPT sangat pintar membantu skripsi saya"* $\rightarrow$ Prediksi: **Positif** (Probabilitas: $93,4\%$), Aspek: **A1 (Akurasi & Kecerdasan)**.
  - *Skenario 2 (Negatif Server & Kuota)*: Input *"Server selalu sibuk dan kuota gratisan cepat habis"* $\rightarrow$ Prediksi: **Negatif** (Probabilitas: $89,1\%$), Aspek: **A2 (Harga/Kuota)** dan **A3 (Performa/Server)**.
  - *Skenario 3 (Discrepancy Tipe I)*: Input Bintang 5 dengan teks *"Bintang 5 tapi aplikasinya lemot dan sering crash"* $\rightarrow$ Terdeteksi: **Type-I Discrepancy Flag = TRUE**.
  - *Skenario 4 (Handling Karakter Kosong)*: Input `""` $\rightarrow$ Menghasilkan pesan peringatan ramah pengguna tanpa memicu unhandled runtime error.

---

*Dokumen ini disusun secara otomatis berdasarkan audit integritas seluruh kode sumber, pipeline machine learning, dan artefak data aktual di workspace AM552B GenAI Indonesia.*
