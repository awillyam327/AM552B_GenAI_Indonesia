# LAPORAN EXPLORATORY DATA ANALYSIS (EDA) AWAL — CRISP-DM FASE 2: DATA UNDERSTANDING

> **Sumber Dataset**: `./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv`  
> **Total Baris Terverifikasi**: **`71.221` baris unik** (`100%` `reviewId` unik, `100%` region `id_ID`)  
> **Rentang Waktu Ulasan**: `2023-05-31 07:37:14` s/d `2026-11-17 16:00:19`  
> **Script Generator**: `02_data_understanding/run_initial_eda.py`

---

## 1. Profil Distribusi Baris, Rating, dan Panjang Teks per Aplikasi

| No | Nama Aplikasi (`app_name`) | Package ID (`app_id`) | Jumlah Baris | Proporsi | Rata-rata Rating | Rata-rata Panjang Ulasan | Rentang Waktu Data |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **ChatGPT (OpenAI)** | `com.openai.chatgpt` | 22.000 | 30.89% | 4.46 | 34.8 kar (5.6 kata) | 2026-09-25 s/d 2026-11-17 |
| 2 | **Google Gemini** | `com.google.android.apps.bard` | 12.600 | 17.69% | 4.29 | 32.0 kar (5.2 kata) | 2026-09-30 s/d 2026-11-17 |
| 3 | **Character.AI** | `ai.character.app` | 12.000 | 16.85% | 1.58 | 136.6 kar (23.0 kata) | 2026-04-29 s/d 2026-11-17 |
| 4 | **Perplexity - Ask Anything** | `ai.perplexity.app.android` | 8.043 | 11.29% | 4.54 | 32.4 kar (5.1 kata) | 2023-06-08 s/d 2026-11-17 |
| 5 | **Microsoft Copilot** | `com.microsoft.copilot` | 5.459 | 7.66% | 4.65 | 39.4 kar (6.2 kata) | 2023-12-27 s/d 2026-11-04 |
| 6 | **DeepSeek - AI Assistant** | `com.deepseek.chat` | 5.374 | 7.55% | 3.77 | 89.4 kar (14.1 kata) | 2025-01-14 s/d 2026-11-15 |
| 7 | **Claude by Anthropic** | `com.anthropic.claude` | 4.009 | 5.63% | 3.85 | 80.4 kar (13.0 kata) | 2024-07-18 s/d 2026-11-17 |
| 8 | **Poe - Fast AI Chat** | `com.poe.android` | 1.736 | 2.44% | 4.66 | 38.0 kar (6.1 kata) | 2023-05-31 s/d 2026-11-17 |
| **TOTAL** | **8 Aplikasi Generative AI** | **Terdeduplikasi Penuh** | **71.221** | **100.00%** | **3.89** | **58.3 kar (9.5 kata)** | **`2023-05-31` s/d `2026-11-17`** |

### Temuan Kunci Distribusi Aplikasi:
1. **Dominasi 3 Platform Utama**: **ChatGPT** (`30,89%`), **Google Gemini** (`17,69%`), dan **Character.AI** (`16,85%`) menyumbang **65,43%** dari total volume ulasan ekosistem Generative AI di Indonesia.
2. **Kontras Kepuasan Ekstrem**:
   - Kelompok **Asisten Produktivitas & Pencarian** memperoleh rata-rata rating sangat tinggi: **Poe** (`4.66`), **Microsoft Copilot** (`4.65`), **Perplexity** (`4.54`), **ChatGPT** (`4.46`), dan **Google Gemini** (`4.29`).
   - Kelompok **Challenger / Reasoning AI**: **Claude by Anthropic** (`3.85`) dan **DeepSeek** (`3.77`) memiliki proporsi bintang 1 yang cukup signifikan akibat friksi *login/paywall* (Claude) dan *server busy* saat lonjakan trafik (DeepSeek).
   - Kelompok **AI Companion**: **Character.AI** mengalami polarisasi ekstrem dengan rata-rata rating **`1.58`** (didominasi Bintang 1 akibat keluhan filter karakter, sensor, dan perubahan antarmuka).

---

## 2. Audit Kualitas Data & Persentase Missing Values

| No | Nama Kolom | Tipe Data | Jumlah Missing (NaN) | Persentase Missing (%) | Status & Interpretasi Data Engineering |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `reviewId` | `str` | 0 | 0.00% | ✅ 100% Lengkap |
| 2 | `app_name` | `str` | 0 | 0.00% | ✅ 100% Lengkap |
| 3 | `app_id` | `str` | 0 | 0.00% | ✅ 100% Lengkap |
| 4 | `userName` | `str` | 1 | 0.00% | ℹ️ Opsional (Mayoritas ulasan tidak dibalas developer) |
| 5 | `content` | `str` | 0 | 0.00% | ✅ 100% Lengkap |
| 6 | `score` | `int64` | 0 | 0.00% | ✅ 100% Lengkap |
| 7 | `thumbsUpCount` | `int64` | 0 | 0.00% | ✅ 100% Lengkap |
| 8 | `reviewCreatedVersion` | `str` | 13.405 | 18.82% | ℹ️ Opsional (Versi tidak tercatat di Play Store) |
| 9 | `at` | `str` | 0 | 0.00% | ✅ 100% Lengkap |
| 10 | `replyContent` | `str` | 71.053 | 99.76% | ℹ️ Opsional (Mayoritas ulasan tidak dibalas developer) |
| 11 | `language_region` | `str` | 0 | 0.00% | ✅ 100% Lengkap |

- **Kolom Kritis (`reviewId`, `app_name`, `app_id`, `content`, `score`, `at`, `thumbsUpCount`) memiliki `0%` missing values (100% utuh).**
- Kolom `reviewCreatedVersion` memiliki missing value sebesar `18.82%` (wajar pada Google Play Store ketika pengguna memberi ulasan via browser web atau perangkat yang tidak mengirimkan metadata versi build).Pada tahap preprocessing, nilai kosong ini akan diisi dengan kategori `"Unknown_Version"`.
- Kolom `replyContent` kosong pada `99.76%` baris, yang menunjukkan hanya **`0.24%`** ulasan pengguna Indonesia yang mendapat balasan resmi dari tim developer aplikasi AI.

---

## 3. Statistik Distribusi Panjang Karakter & Jumlah Kata Ulasan

| Metrik Statistik | Panjang Karakter (`char_length`) | Jumlah Kata (`word_count`) | Interpretasi Analitis |
| :--- | :---: | :---: | :--- |
| **Mean (Rata-rata)** | `58.30` karakter | `9.54` kata | Rata-rata ulasan terdiri dari 1–2 kalimat. |
| **Std Dev (Simpangan Baku)** | `91.44` karakter | `14.89` kata | Variasi tinggi antara ulasan singkat (1 kata) vs komplain panjang. |
| **Minimum** | `1` karakter | `1` kata | Ulasan berupa satu emoji/simbol (misal `"👍"`). |
| **Kuartil 1 (Q1 / 25%)** | `8` karakter | `1` kata | 25% ulasan berukuran sangat singkat ($\le 1$ kata, mis. *"bagus"*, *"mantap"*). |
| **Median (Q2 / 50%)** | `21` karakter | `3` kata | Nilai tengah panjang ulasan pengguna Indonesia. |
| **Kuartil 3 (Q3 / 75%)** | `62` karakter | `10` kata | 25% ulasan teratas memiliki pembahasan fitur yang cukup detail. |
| **Persentil 95% (P95)** | `260` karakter | `42` kata | Ulasan naratif panjang (umumnya berisi keluhan teknis mendalam). |
| **Maksimum** | `500` karakter | `104` kata | Batas maksimum panjang ulasan Google Play Store (~500 karakter). |

> **Perbandingan Panjang Ulasan Berdasarkan Skor Bintang**:
> - Rata-rata panjang ulasan **Bintang 1 (⭐)**: **`115.7` karakter** (`19.3` kata).
> - Rata-rata panjang ulasan **Bintang 5 (⭐⭐⭐⭐⭐)**: **`32.9` karakter** (`5.3` kata).
> - **Insight Penting**: Ulasan negatif (Bintang 1) rata-rata **lebih panjang hampir 2,5x lipat** dibandingkan ulasan Bintang 5 karena pengguna yang kecewa menuliskan rincian bug, fitur yang rusak, atau keluhan kuota secara spesifik.

---

## 4. Deteksi Awal *Rating-Text Discrepancy* (Paradoks Bintang vs. Isi Teks)

Menggunakan aturan deteksi leksikal awal pada `71.221` ulasan:
- **Total Kasus *Discrepancy* Terdeteksi Awal**: **`3.271` ulasan (`4.59%` dari keseluruhan dataset)**
  - **Type-I (*High Rating $\ge 4$, Negative Complaint Text*)**: **`2.438` ulasan (`3.42%`)** — Pengguna memberi Bintang 4 atau 5 namun menuliskan keluhan (`error`, `limit`, `lemot`, `macet`, `bayar`, `gagal`, dll.).
  - **Type-II (*Low Rating $\le 2$, Pure Positive Praise*)**: **`833` ulasan (`1.17%`)** — Pengguna memberi Bintang 1 atau 2 namun isi teksnya murni memuji tanpa kata keluhan/negasi.

### A. Rincian *Rating-Text Discrepancy* per Aplikasi

| No | Nama Aplikasi | Type-I: Rating $\ge 4$ tapi Isi Keluhan | Type-II: Rating $\le 2$ tapi Isi Pujian | Total Discrepancy per Aplikasi |
| :---: | :--- | :---: | :---: | :---: |
| 1 | **ChatGPT (OpenAI)** | 522 (2.37%) | 149 (0.68%) | **671 (3.05%)** |
| 2 | **Google Gemini** | 228 (1.81%) | 113 (0.90%) | **341 (2.71%)** |
| 3 | **Character.AI** | 412 (3.43%) | 357 (2.97%) | **769 (6.41%)** |
| 4 | **Perplexity - Ask Anything** | 172 (2.14%) | 44 (0.55%) | **216 (2.69%)** |
| 5 | **Microsoft Copilot** | 171 (3.13%) | 30 (0.55%) | **201 (3.68%)** |
| 6 | **DeepSeek - AI Assistant** | 570 (10.61%) | 74 (1.38%) | **644 (11.98%)** |
| 7 | **Claude by Anthropic** | 307 (7.66%) | 56 (1.40%) | **363 (9.05%)** |
| 8 | **Poe - Fast AI Chat** | 56 (3.23%) | 10 (0.58%) | **66 (3.80%)** |

### B. Sampel Bukti Nyata *Rating-Text Discrepancy* di Dataset Kita
**Contoh Nyata Type-I (Rating Bintang 4–5, tetapi Teks Mengeluh):**
- **[Google Gemini | ⭐ 5]** `El No`: *"saya kasih bintang walaupun bayar"*
- **[Google Gemini | ⭐ 4]** `Aman Transhp`: *"bersihkan sampah Di ponsel"*
- **[Google Gemini | ⭐ 4]** `Dewa Made`: *"tidak bisa belah layar"*
- **[Google Gemini | ⭐ 5]** `Imam Hanafi`: *"sangat rekomendasi kalau buat iklan"*
- **[Google Gemini | ⭐ 5]** `Fira Suhendri`: *"jelek"*

**Contoh Nyata Type-II (Rating Bintang 1–2, tetapi Teks Memuji):**
- **[Google Gemini | ⭐ 1]** `Bilqis azra kamilla Kamilla`: *"ikon bagus kualitas 0"*
- **[Google Gemini | ⭐ 1]** `Vhino Rompas`: *"bagus"*
- **[Google Gemini | ⭐ 1]** `Jenemia Lette`: *"terbaik"*
- **[Google Gemini | ⭐ 1]** `Matwani Wani`: *"bagus sekali"*
- **[Google Gemini | ⭐ 1]** `Muhamad Rizki`: *"kenapa tiba2 yang mode flash yang versi pro nya ngga ada fiturnya dan malah di ganti ke hydrogen????, padahal lebih bagus fitur yang flash p"*

---

## 5. Top 20 Kata (*Unigram*) dan Pasangan Kata (*Bigram*) Paling Sering Muncul

Setelah dilakukan pembersihan *stopwords* dasar dan normalisasi kata tidak baku (*slang normalization* seperti `"gk" -> "tidak"`, `"bgt" -> "banget"`, `"apk" -> "aplikasi"`):

| Peringkat | Top 20 Unigram (Kata Tunggal) | Frekuensi Unigram | Top 20 Bigram (Pasangan 2 Kata) | Frekuensi Bigram |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `bagus` | 16.721 | `sangat membantu` | 5.144 |
| 2 | `sangat` | 13.332 | `sangat bagus` | 2.673 |
| 3 | `tidak` | 12.509 | `bagus banget` | 1.718 |
| 4 | `ai` | 9.969 | `aplikasi sangat` | 877 |
| 5 | `aplikasi` | 9.213 | `bagus tapi` | 851 |
| 6 | `banget` | 7.865 | `bagus sekali` | 741 |
| 7 | `membantu` | 7.293 | `aplikasi bagus` | 681 |
| 8 | `tapi` | 5.116 | `luar biasa` | 660 |
| 9 | `mantap` | 3.796 | `tiba tiba` | 627 |
| 10 | `update` | 3.719 | `chat gpt` | 615 |
| 11 | `chat` | 3.653 | `sangat baik` | 613 |
| 12 | `banyak` | 3.510 | `banyak iklan` | 549 |
| 13 | `makin` | 3.436 | `tidak jelas` | 528 |
| 14 | `sekarang` | 3.263 | `membantu sekali` | 516 |
| 15 | `dulu` | 3.101 | `ai dulu` | 502 |
| 16 | `good` | 2.973 | `verifikasi usia` | 460 |
| 17 | `keren` | 2.881 | `terima kasih` | 435 |
| 18 | `suka` | 2.777 | `tolong perbaiki` | 430 |
| 19 | `baik` | 2.757 | `verifikasi umur` | 408 |
| 20 | `tolong` | 2.651 | `bagus sangat` | 392 |

---

## 6. Daftar Grafik Visualisasi yang Dihasilkan (`02_data_understanding/figures/`)

1. **`01_distribusi_ulasan_per_aplikasi.png`**: Visualisasi jumlah ulasan unik dan skor rating rata-rata lintas 8 aplikasi Generative AI.
2. **`02_distribusi_rating_per_aplikasi.png`**: Grafik *100% Stacked Horizontal Bar* proporsi bintang 1 s/d 5 pada masing-masing aplikasi.
3. **`03_deteksi_awal_discrepancy_per_aplikasi.png`**: Komparasi persentase *Rating-Text Discrepancy* (Type-I dan Type-II) per aplikasi.
