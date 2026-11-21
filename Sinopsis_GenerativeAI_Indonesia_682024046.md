# SINOPSIS PROYEK AKHIR DATA SCIENCE (AM522B)

---

## 1. Judul Usulan Proyek

### Judul Utama (Rekomendasi Utama)
> **"Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia"**

### Variasi Judul Alternatif
1. **Fokus *Predictive User Satisfaction & Churn Risk***:  
   *"Prediksi Ketidakpuasan Pengguna dan Deteksi Anomali Rating-Ulasan (Rating-Text Discrepancy) pada Aplikasi Mobile Generative AI di Indonesia Menggunakan Ensemble Gradient Boosting dan SHAP Explainability"*
2. **Fokus *NLP Transformer Benchmarking & Topic Discovery***:  
   *"Studi Komparasi Arsitektur Pre-Trained IndoBERT dan Gradient Boosting serta Ekstraksi Topik Berbasis BERTopic pada 71.221 Ulasan Pengguna Aplikasi Generative AI di Google Play Store Indonesia"*

---

## 2. Latar Belakang & Urgensi Penelitian

Ekosistem komputasi personal dan produktivitas digital di Indonesia tengah mengalami transformasi struktural seiring ledakan adopsi aplikasi **Mobile Generative AI** (*Large Language Models* dan asisten percakapan multimodal). Platform global seperti **ChatGPT (OpenAI)**, **Google Gemini**, **DeepSeek**, **Microsoft Copilot**, **Perplexity**, **Claude (Anthropic)**, **Poe**, hingga **Character.AI** dan **Qwen** kini digunakan secara masif oleh pelajar, mahasiswa, praktisi teknologi, hingga pekerja kreatif di Indonesia untuk mendukung penulisan akademik, pemrograman (*coding*), pencarian informasi *real-time*, serta otomatisasi tugas harian.

Meskipun tingkat unduhan aplikasi Generative AI di Indonesia tumbuh secara eksponensial, persaingan ketat antar-platform memunculkan dinamika pengalaman pengguna (*User Experience*) yang sangat kompleks dan terpolarisasi di Google Play Store:
1. **Kualitas Pemahaman Konteks & Bahasa Indonesia**: Pengguna di Indonesia sering mengeluhkan ketimpangan kemampuan model dalam memahami struktur bahasa kasual/slang lokal, konteks budaya Indonesia, hingga fenomena **halusinasi model** (*hallucination*) saat menjawab pertanyaan faktual atau teknis.
2. **Friksi Monetisasi, Batasan Kuota Gratis, dan Harga Langganan**: Kebijakan pembatasan pesan harian (*rate limit* model flagship seperti GPT-4o, Claude 3.5/3.7 Sonnet, atau akses fitur gambar/suara) serta tingginya biaya berlangganan (*Plus/Pro tier*) dibandingkan daya beli pengguna lokal menjadi pemicu utama migrasi pengguna ke alternatif gratis seperti DeepSeek.
3. **Reliabilitas Teknis, Latensi Mobile, dan Fitur Multimodal**: Keluhan terkait kegagalan *login* autentikasi, *server down/busy* (terutama saat lonjakan trafik rilis model baru), konsumsi baterai, respons fitur *voice mode* yang lambat, hingga perubahan antarmuka (*UI/UX*) berdampak langsung terhadap retensi pengguna.
4. **Fenomena *Rating-Text Discrepancy* (paradoks skor bintang vs. teks ulasan)**: Banyak pengguna di Indonesia memberikan rating bintang 5 tetapi menuliskan keluhan teknis di dalam teks ulasan (atau sebaliknya, memberikan bintang 1 hanya karena salah paham cara login). Oleh sebab itu, metrik agregat bintang semata tidak lagi memadai untuk mengevaluasi kualitas layanan secara objektif.

Melalui pendekatan **Aspect-Based Sentiment Analysis (ABSA)** dan **Pemodelan Topik (Topic Modeling)** terhadap **71.221 data ulasan nyata (*100% real scraped data*)**, penelitian ini memiliki urgensi tinggi untuk membedah peta kekuatan, kelemahan, dan determinan kepuasan pengguna ekosistem Generative AI di Indonesia secara empiris dan berbasis bukti (*data-driven*).

---

## 3. Rumusan Masalah & Tujuan Analisis

### A. Rumusan Masalah
1. Bagaimana perbandingan distribusi polarisasi sentimen (positif, netral, negatif) dan dinamika tingkat kepuasan pengguna antar-aplikasi mobile Generative AI terpopuler di Indonesia dari waktu ke waktu?
2. Aspek spesifik layanan apa saja (**Akurasi & Halusinasi Model**, **Harga & Limitasi Kuota Gratis**, **UI/UX & Kecepatan Server/Autentikasi**, serta **Kemampuan Bahasa Indonesia & Fitur Multimodal**) yang paling dominan memicu sentimen negatif maupun positif pada masing-masing platform?
3. Seberapa besar tingkat ketidaksesuaian (*rating-text discrepancy*) antara skor bintang numerik (1–5) yang diberikan pengguna dengan polaritas sentimen aktual di dalam teks ulasannya?
4. Sejauh mana keunggulan performa model *Pre-trained Language Model* (**IndoBERT**) dibandingkan algoritma *Ensemble Machine Learning* (**CatBoost, XGBoost, LightGBM**) dalam mengklasifikasikan sentimen dan mendeteksi *rating-text discrepancy*, serta fitur kata kunci apa yang paling berpengaruh berdasarkan analisis **SHAP**?

### B. Tujuan Analisis
1. Memetakan lanskap komparatif kepuasan dan polarisasi sentimen pengguna lintas aplikasi Generative AI utama di Indonesia menggunakan **71.221 ulasan unik** dari Google Play Store.
2. Mengekstraksi struktur topik laten dan memetakan sentimen berbasis aspek (*ABSA*) menggunakan **BERTopic** dan **Latent Dirichlet Allocation (LDA)** untuk mengidentifikasi *pain points* dan keunggulan kompetitif tiap platform.
3. Mengukur dan mengklasifikasikan fenomena *rating-text discrepancy* guna menyaring bias penilaian bintang pada ulasan aplikasi mobile.
4. Membangun, mengevaluasi, dan menginterpretasikan model klasifikasi sentimen multi-arsitektur (**CatBoost/XGBoost/LightGBM vs. IndoBERT**) lengkap dengan atribusi fitur berbasis **SHAP (*SHapley Additive exPlanations*)**.

---

## 4. Spesifikasi Raw Dataset (Data Nyata Hasil Scraping)

Dataset dikumpulkan secara langsung (*100% real scraped data*, tanpa data sintetis/dummy) dari server **Google Play Store** menggunakan pustaka `google-play-scraper` dengan mekanisme *high-yield pagination* (`continuation_token` per batch 200 ulasan, kombinasi `Sort.NEWEST`, `Sort.MOST_RELEVANT`, dan iterasi `filter_score_with` bintang 1 s/d 5) serta telah melalui proses deduplikasi ketat berdasarkan `reviewId`.

### A. Ringkasan Metrik Utama Dataset
- **TOTAL RAW DATA (Baris Unik Eksak)**: **`71.221` baris unik** (Melampaui batas minimum kelas 37.455 baris).
- **Jumlah Kolom Atribut**: **`11` kolom**.
- **Rentang Waktu Ulasan Aktual**: **`2023-05-31` hingga `2026-11-17`**.
- **File Raw Dataset Utama**: `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv`

### B. Tabel Rincian Jumlah Record per Aplikasi dan Rentang Waktu Data

| No | Nama Aplikasi (`app_name`) | Package ID (`app_id`) | Jumlah Baris Unik | Proporsi (%) | Rata-rata Rating | Rentang Tanggal Ulasan |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | **ChatGPT (OpenAI)** | `com.openai.chatgpt` | 22.000 | 30.89% | 4.46 / 5.00 | 2026-09-25 s/d 2026-11-17 |
| 2 | **Google Gemini** | `com.google.android.apps.bard` | 12.600 | 17.69% | 4.29 / 5.00 | 2026-09-30 s/d 2026-11-17 |
| 3 | **Character.AI** | `ai.character.app` | 12.000 | 16.85% | 1.58 / 5.00 | 2026-04-29 s/d 2026-11-17 |
| 4 | **Perplexity - Ask Anything** | `ai.perplexity.app.android` | 8.043 | 11.29% | 4.54 / 5.00 | 2023-06-08 s/d 2026-11-17 |
| 5 | **Microsoft Copilot** | `com.microsoft.copilot` | 5.459 | 7.66% | 4.65 / 5.00 | 2023-12-27 s/d 2026-11-04 |
| 6 | **DeepSeek - AI Assistant** | `com.deepseek.chat` | 5.374 | 7.55% | 3.77 / 5.00 | 2025-01-14 s/d 2026-11-15 |
| 7 | **Claude by Anthropic** | `com.anthropic.claude` | 4.009 | 5.63% | 3.85 / 5.00 | 2024-07-18 s/d 2026-11-17 |
| 8 | **Poe - Fast AI Chat** | `com.poe.android` | 1.736 | 2.44% | 4.66 / 5.00 | 2023-05-31 s/d 2026-11-17 |
| **TOTAL** | **Ekosistem Generative AI (8 Aplikasi)** | **Terdeduplikasi (`reviewId` Unik)** | **71.221** | **100.00%** | **-** | **`2023-05-31` s/d `2026-11-17`** |

### C. Tabel Distribusi Rating Bintang (Skor 1 s/d 5)

| Skor Bintang (`score`) | Kategori Kepuasan Eksplisit | Jumlah Baris | Proporsi (%) |
| :---: | :--- | :---: | :---: |
| ⭐ 5 | Sangat Puas (Bintang 5) | 45.553 | 63.96% |
| ⭐ 4 | Puas (Bintang 4) | 5.258 | 7.38% |
| ⭐ 3 | Netral / Ragu-ragu (Bintang 3) | 2.733 | 3.84% |
| ⭐ 2 | Tidak Puas (Bintang 2) | 2.158 | 3.03% |
| ⭐ 1 | Sangat Kecewa / Komplain Kritis (Bintang 1) | 15.519 | 21.79% |

### D. Kamus Data (*Data Dictionary*) Raw Dataset

| No | Nama Kolom | Tipe Data | Deskripsi dan Peran Analitis dalam Penelitian |
| :---: | :--- | :---: | :--- |
| 1 | `reviewId` | `String (UUID)` | Identifikasi unik (*Primary Key*) dari server Google Play Store untuk menjamin integritas 0% duplikasi baris. |
| 2 | `app_name` | `String` | Nama resmi aplikasi Generative AI yang dianalisis (mis. *ChatGPT (OpenAI)*, *Google Gemini*, *DeepSeek - AI Assistant*). |
| 3 | `app_id` | `String` | Package identifier aplikasi di Google Play Store (mis. `com.openai.chatgpt`, `com.google.android.apps.bard`). |
| 4 | `userName` | `String` | Nama tampilan pengguna Google yang memberikan ulasan di Play Store. |
| 5 | `content` | `String (Text)` | Teks asli ulasan pengguna (*raw review text*) yang menjadi korpus utama analisis NLP, ABSA, dan Topic Modeling. |
| 6 | `score` | `Integer (1-5)` | Skor rating bintang (skala 1 hingga 5) yang diberikan pengguna; digunakan sebagai target eksplisit & analisis *discrepancy*. |
| 7 | `thumbsUpCount` | `Integer` | Jumlah dukungan (*likes/helpful votes*) dari pengguna lain terhadap ulasan tersebut; indikator bobot dampak keluhan/pujian. |
| 8 | `reviewCreatedVersion` | `String` | Versi rilis aplikasi saat ulasan ditulis; krusial untuk melacak bug spesifik versi atau dampak pembaruan fitur. |
| 9 | `at` | `Datetime` | Timestamp eksak (`YYYY-MM-DD HH:MM:SS`) saat ulasan dipublikasikan; digunakan untuk analisis deret waktu (*time-series*). |
| 10 | `replyContent` | `String (Nullable)` | Teks balasan resmi dari pengembang (*developer response*) jika ada; indikator responsivitas layanan pelanggan. |
| 11 | `language_region` | `String` | Label transparan kode bahasa dan region pengambilan ulasan (`id_ID` / `en_ID`) pada Google Play Store. |

---

## 5. Arsitektur Metodologi Data Science (Framework CRISP-DM)

Penelitian ini dirancang menggunakan metodologi standar industri **CRISP-DM (*Cross-Industry Standard Process for Data Mining*)** yang diadaptasi khusus untuk *Natural Language Processing (NLP)* dan *Predictive Text Analytics* berskala besar:

### Tahap 1: Business & Data Understanding
- Memetakan lanskap kompetisi ekosistem aplikasi Generative AI di Indonesia (Model Flagship Umum vs. Reasoning/Search AI vs. Companion/Multi-bot AI).
- Merumuskan taksonomi 4 aspek inti evaluasi layanan AI: **(1) Akurasi, Penalaran & Halusinasi**, **(2) Kuota Gratis, Paywall & Harga Langganan**, **(3) UI/UX, Stabilitas Server & Autentikasi**, serta **(4) Kapabilitas Bahasa Lokal & Fitur Suara/Multimodal**.

### Tahap 2: Data Engineering & Text Preprocessing
1. **Data Integrity & Deduplication**: Validasi skema kolom, penghapusan ulasan kosong, dan deduplikasi berbasis `reviewId` atas 71.221 baris data mentah.
2. **Colloquial Indonesian Text Normalization**: Pembersihan karakter non-alfanumerik/emoji berlebih, *case folding*, serta normalisasi kata tidak baku/slang Bahasa Indonesia (mis. *"gk" -> "tidak"*, *"bgt" -> "banget"*, *"lemot" -> "lambat"*, *"eror" -> "error"*) menggunakan kamus *Colloquial Indonesian Lexicon*.
3. **Tokenization, Stopword Removal & Stemming Selektif**: Mempertahankan kata negasi (*"tidak"*, *"kurang"*, *"jangan"*, *"bukan"*) agar konteks sentimen tidak terbalik saat pembentukan fitur.
4. **Pelabelan *Rating-Sentiment Discrepancy***: Mengombinasikan skor leksikon sentimen (*InSet Lexicon* / VADER multilingual) dengan skor bintang (`score`) untuk menandai ulasan yang mengalami kontradiksi antara rating bintang dan polaritas teks (mis. Rating 5 tetapi teks bernada keluhan negatif, atau Rating 1 dengan teks positif).

### Tahap 3: Exploratory Data Analysis (EDA)
1. **Time-Series Review Surge Analysis**: Analisis tren volume ulasan dan fluktuasi rating mingguan/bulanan untuk mendeteksi anomali lonjakan ulasan (*surge*) saat momen krusial, seperti rilis **GPT-4o**, peluncuran viral **DeepSeek-R1**, rebranding **Bard menjadi Google Gemini**, hingga insiden gangguan server global.
2. **Comparative N-Gram (Unigram, Bigram, Trigram) Analysis**: Membandingkan frasa keluhan spesifik antar-aplikasi (mis. *"limit habis"*, *"login gagal"*, *"jawaban ngawur"*, *"server sibuk"*).
3. **Review Length & Helpfulness Distribution**: Menganalisis korelasi antara panjang karakter ulasan, skor bintang, dan jumlah `thumbsUpCount` (ulasan negatif panjang cenderung memperoleh *thumbs up* jauh lebih tinggi).

### Tahap 4: Dual-Stage Modeling (Eksperimen Komparatif)

| Tahap Eksperimen | Fokus Pemodelan | Algoritma & Arsitektur yang Dibandingkan | Rekayasa Fitur (Feature Engineering) |
| :---: | :--- | :--- | :--- |
| **Experiment 1** | **Sentiment & Rating-Discrepancy Classification** | **Ensemble Boosting**: `CatBoost`, `XGBoost`, `LightGBM`<br>**Deep Learning Transformer**: `IndoBERT` (`indobenchmark/indobert-base-p1`) | TF-IDF (1-3 n-grams) + Lexicon Polarity Score + Metadata (`review_length`, `thumbsUpCount`) vs. Contextual Dense Embeddings (768-dim) |
| **Experiment 2** | **Topic Modeling & Aspect-Based Sentiment Extraction (ABSA)** | **Neural Topic Modeling**: `BERTopic` (Sentence-Transformers + UMAP + HDBSCAN + c-TF-IDF)<br>**Probabilistic Baseline**: `LDA` (*Latent Dirichlet Allocation*) | Klasterisasi topik keluhan & pujian per aplikasi, dipetakan ke 4 dimensi aspek layanan AI untuk menghitung *Net Sentiment Score* per aspek |

### Tahap 5: Model Evaluation & Explainability (XAI)
1. **Evaluasi Performa Klasifikasi (Experiment 1)**: Menggunakan skema *Stratified K-Fold Cross Validation* dengan metrik **Accuracy**, **Precision**, **Recall**, **Macro F1-Score** (mengatasi *class imbalance* pada rating bintang 5 vs 1–4), **ROC-AUC (One-vs-Rest)**, serta visualisasi **Confusion Matrix**.
2. **Evaluasi Koherensi Topik (Experiment 2)**: Mengukur kualitas pemisahan topik antara BERTopic dan LDA menggunakan metrik **Topic Coherence ($C_v$)** dan **Topic Diversity**.
3. **Explainable AI dengan SHAP (*SHapley Additive exPlanations*)**: Menghitung nilai kontribusi marginal (*SHAP values*) dari setiap fitur kata kunci dan variabel metadata terhadap probabilitas prediksi ketidakpuasan pengguna, sehingga menghasilkan penjelasan transparan mengenai pendorong utama churn/komplain pada tiap aplikasi AI.

---

## 6. Expected Output & Kontribusi Praktis

1. **Matriks Benchmarking Komparatif Ekosistem Generative AI Indonesia**: Peta kuadran kinerja pengalaman pengguna lintas aplikasi (ChatGPT, Gemini, DeepSeek, Copilot, Perplexity, Claude, Poe, Character.AI, dll.) berdasarkan *Net Sentiment Score* pada tiap aspek layanan (Akurasi, Harga/Kuota, UI/UX & Kecepatan, Bahasa Lokal).
2. **Sistem Deteksi *Rating-Text Discrepancy* & Prioritisasi Komplain Kritis**: Model terlatih yang mampu menyaring ulasan "bintang palsu/bias" dan mengurutkan keluhan teknis paling berdampak (berdasarkan prediksi sentimen negatif + bobot `thumbsUpCount`) bagi tim *Product Manager* dan *AI Engineer*.
3. **Rekomendasi Strategis Berbasis Bukti (*Actionable Insights*)**: Panduan empiris bagi pengembang aplikasi AI dan pemangku kepentingan di Indonesia mengenai fitur apa yang paling disukai pengguna lokal serta friksi teknis apa yang paling mendesak untuk diperbaiki.
