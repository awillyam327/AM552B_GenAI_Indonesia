# PROJECT CHARTER & DEFINISI ASPEK ABSA (CRISP-DM FASE 1: BUSINESS UNDERSTANDING)

> **Judul Proyek**: *Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia*  
> **Kode Mata Kuliah**: AM522B — Data Science  
> **Baseline Dataset**: `71.221` Ulasan Unik Google Play Store Indonesia (`lang='id'`, `country='id'`) pada 8 Aplikasi Utama Generative AI (`2023-05-31` s/d `2026-11-17`).

---

## 1. Tujuan Bisnis & Pertanyaan Penelitian Utama (*Research Questions*)

Pertumbuhan eksponensial pengguna aplikasi *Mobile Generative AI* di Indonesia memunculkan persaingan ketat antara 8 platform utama: **ChatGPT (OpenAI)**, **Google Gemini**, **Character.AI**, **Perplexity**, **Microsoft Copilot**, **DeepSeek**, **Claude (Anthropic)**, dan **Poe**. Penelitian ini bertujuan menjawab 4 pertanyaan penelitian utama:

1. **RQ-1 (Polarisasi Sentimen & Kepuasan Lintas Platform)**:  
   Bagaimana perbandingan distribusi sentimen aktual pengguna Indonesia antar-8 aplikasi Generative AI, dan mengapa terjadi kesenjangan kepuasan yang ekstrem antar-platform (misalnya antara aplikasi asisten produktivitas seperti ChatGPT/Copilot/Perplexity dengan rata-rata rating `> 4.45` dibanding Character.AI `1.58` atau DeepSeek `3.77`)?
2. **RQ-2 (Pemetaan Sentimen Berbasis Aspek / ABSA)**:  
   Aspek layanan manakah dari 4 dimensi utama — **(A1) Akurasi & Kecerdasan AI**, **(A2) Harga & Limitasi Kuota Gratis**, **(A3) Performa, Stabilitas & Bug/UI Aplikasi**, serta **(A4) Dukungan & Keluwesan Bahasa Indonesia** — yang menjadi pendorong utama pujian maupun keluhan kritis pada masing-masing aplikasi?
3. **RQ-3 (Anomali *Rating-Text Discrepancy*)**:  
   Seberapa besar proporsi ulasan di Google Play Store Indonesia yang mengalami ketidaksesuaian antara skor bintang numerik (`score` 1–5) dan polaritas teks ulasan (`content`), serta aplikasi mana yang paling terdampak oleh bias penilaian bintang tersebut?
4. **RQ-4 (Pemodelan Prediktif, Topik Laten & Explainability)**:  
   Sejauh mana arsitektur *Pre-trained Transformer* (**IndoBERT**) mengungguli *Gradient Boosting Ensemble* (**CatBoost, XGBoost, LightGBM**) dalam klasifikasi sentimen/discrepancy, topik laten apa yang terungkap melalui **BERTopic vs. LDA**, dan kata kunci apa yang memiliki nilai atribusi **SHAP** tertinggi terhadap ketidakpuasan pengguna?

---

## 2. Hipotesis Awal Penelitian (*Initial Hypotheses*)

* **H1 (*Asymmetric Aspect Pain-Points*)**: Aplikasi AI model *flagship* berbayar (**ChatGPT**, **Claude**, **Gemini**) memiliki sentimen positif tinggi pada aspek *Akurasi & Kecerdasan AI*, tetapi didominasi sentimen negatif pada aspek *Harga & Limitasi Kuota Gratis*; sebaliknya aplikasi gratis seperti **DeepSeek** memiliki sentimen positif pada aspek *Harga/Gratis*, namun didominasi keluhan pada aspek *Performa & Stabilitas Server* (*server busy/login gagal*).
* **H2 (*Companion AI Moderation Friction*)**: Rendahnya skor rating **Character.AI** (`1.58`) di Indonesia tidak disebabkan oleh ketidakmampuan berbahasa, melainkan terkonsentrasi pada aspek *Kebijakan Filter/Sensor, Penghapusan Fitur Lama, dan Perubahan UI/Bug*.
* **H3 (*Significant Rating-Text Discrepancy*)**: Terdapat minimal **5% – 12%** ulasan pengguna Indonesia yang mengalami *Rating-Text Discrepancy* (khususnya ulasan Bintang 4–5 yang memuat kata kunci keluhan teknis agar dibaca oleh pengembang).
* **H4 (*Model Performance & Contextual Advantage*)**: **IndoBERT** akan menghasilkan *Macro F1-Score* lebih tinggi pada ulasan yang mengandung sarkasme atau negasi ganda, sementara **CatBoost/LightGBM** dengan fitur *TF-IDF + Lexicon + Metadata* mampu mencapai >90% performa IndoBERT dengan waktu komputasi jauh lebih cepat dan interpretabilitas **SHAP** langsung pada level token kata.

---

## 3. Taksonomi & Kamus Definisi Aspek ABSA (*Aspect Definition Matrix*)

Setiap ulasan pada dataset (`content`) akan dipetakan ke satu atau lebih dari **4 Kategori Aspek Utama** (*Multi-Label Aspect Extraction*) dengan definisi operasional dan kamus kata kunci pemicu (*seed keywords*) sebagai berikut:

| Kode | Nama Kategori Aspek | Definisi Operasional | Kata Kunci Pemicu (*Seed Keywords* — Setelah Normalisasi Slang) | Contoh Konteks Positif vs. Negatif |
| :---: | :--- | :--- | :--- | :--- |
| **A1** | **Akurasi & Kecerdasan AI** (*Intelligence, Reasoning & Hallucination*) | Kemampuan model memahami instruksi, menjawab soal/tugas/coding dengan tepat, penalaran logis, atau sebaliknya mengalami halusinasi/jawaban ngawur. | `pintar`, `cerdas`, `akurat`, `tepat`, `paham`, `mengerti`, `jawaban`, `solusi`, `bantu`, `tugas`, `skripsi`, `coding`, `matematika`, `halusinasi`, `ngawur`, `salah`, `bodoh`, `tidak nyambung`, `ngaco`, `bertele`, `lupa`, `sensor`, `filter` | **(+)**: *"Jawabannya sangat akurat dan membantu tugas coding saya."*<br>**(-)**: *"Makin kesini jawabannya sering ngawur dan tidak nyambung."* |
| **A2** | **Harga & Limitasi Kuota Gratis** (*Pricing, Quota Limit & Paywall*) | Kebijakan pembatasan pesan/gambar harian pada versi gratis (*rate limit*), kemunculan iklan/paywall, serta keterjangkauan harga langganan *Plus/Pro*. | `gratis`, `free`, `bayar`, `berbayar`, `mahal`, `murah`, `harga`, `langganan`, `subscription`, `plus`, `pro`, `kuota`, `limit`, `batas`, `dibatasi`, `habis`, `koin`, `token`, `iklan`, `premium` | **(+)**: *"Sangat bagus karena gratis tanpa batas kuota harian."*<br>**(-)**: *"Limit foto dan chat cepat banget habis, harus bayar mahal."* |
| **A3** | **Performa, Stabilitas & Bug Aplikasi** (*App Performance, UI/UX, Server & Bugs*) | Kecepatan respons aplikasi, stabilitas server (*down/busy*), proses autentikasi *login*, fitur suara (*live voice*), ukuran aplikasi, dan antarmuka (*UI/UX*). | `cepat`, `lambat`, `lemot`, `lelet`, `loading`, `server`, `sibuk`, `down`, `error`, `bug`, `macet`, `crash`, `keluar sendiri`, `force close`, `login`, `masuk`, `akun`, `update`, `versi`, `tampilan`, `ui`, `suara`, `voice`, `live`, `jaringan`, `koneksi`, `berat`, `ringan` | **(+)**: *"Aplikasinya ringan, respon cepat, tampilan UI bersih."*<br>**(-)**: *"Mode live sering macet sendiri, server sibuk terus gagal login."* |
| **A4** | **Dukungan Bahasa Indonesia** (*Indonesian Language & Local Context*) | Kemampuan aplikasi berkomunikasi dalam Bahasa Indonesia yang alami, memahami bahasa gaul/daerah, terjemahan, serta aksen suara Bahasa Indonesia. | `bahasa`, `indonesia`, `indo`, `terjemahan`, `translate`, `logat`, `aksen`, `baku`, `gaul`, `lokal`, `kaku`, `inggris`, `english`, `pelafalan`, `bicara` | **(+)**: *"Bahasa Indonesianya luwes banget kayak ngobrol sama teman."*<br>**(-)**: *"Kadang tiba-tiba balas pakai bahasa Inggris dan bahasanya kaku."* |

> **Catatan Metodologis**: Ulasan singkat yang hanya berisi ekspresi kepuasan/ketidakpuasan umum tanpa menyebut aspek spesifik (contoh: *"Mantap"*, *"Bagus banget"*, *"Jelek"*) dikategorikan ke dalam **`A0: General Overall Impression`**.

---

## 4. Aturan Operasional Deteksi *Rating-Text Discrepancy*

Sebuah baris ulasan ditandai mengalami **`is_discrepancy = 1`** apabila memenuhi salah satu dari dua kondisi kontradiksi berikut:

1. **Positive Rating with Negative Complaint (`Discrepancy Type-I: High Star, Negative Text`)**:
   - Skor bintang tinggi (`score >= 4`, yaitu Bintang 4 atau 5), **TETAPI** teks ulasan mengandung kata kunci keluhan kritis/negatif (`error`, `macet`, `lemot`, `lambat`, `limit`, `mahal`, `gagal`, `kecewa`, `jelek`, `rusak`, `parah`, `ngawur`, `tidak bisa`, `jangan`, `tolong perbaiki`, `bug`, `gangguan`, `down`) dengan skor leksikon total bernilai negatif.
2. **Negative Rating with Pure Praise (`Discrepancy Type-II: Low Star, Positive Text`)**:
   - Skor bintang rendah (`score <= 2`, yaitu Bintang 1 atau 2), **TETAPI** teks ulasan murni berisi kata pujian (`bagus`, `mantap`, `keren`, `membantu`, `terbaik`, `puas`, `suka`, `pintar`, `good`, `best`) tanpa adanya kata negasi (`tidak`, `kurang`, `gak`, `bukan`, `jangan`) maupun kata keluhan.

---

## 5. Kriteria Sukses Teknis & KPI Proyek (*Technical Success Criteria*)

| Komponen Evaluasi | Metrik Utama | Target Minimum (Baseline) | Target Unggulan (Stretch Goal) |
| :--- | :--- | :---: | :---: |
| **Data Integrity & Volume** | Jumlah Baris Unik Terverifikasi (`reviewId`) | $\ge 45.000$ baris | **$71.221$ baris unik (Tercapai 100%)** |
| **Experiment 1: Klasifikasi Sentimen (Ensemble Boosting)** | *Macro F1-Score* & *ROC-AUC (OvR)* pada 5-Fold CV | Macro F1 $\ge 0.78$<br>ROC-AUC $\ge 0.88$ | Macro F1 $\ge 0.84$<br>ROC-AUC $\ge 0.92$ |
| **Experiment 1: Klasifikasi Sentimen (IndoBERT)** | *Macro F1-Score* pada Test Set | Macro F1 $\ge 0.82$ | Macro F1 $\ge 0.87$ |
| **Experiment 2: Topic Modeling (BERTopic vs. LDA)** | *Topic Coherence ($C_v$)* & Interpretabilitas Klaster | $C_v \ge 0.50$ dengan $\ge 8$ topik bermakna | $C_v \ge 0.60$ + Pemetaan topik per aplikasi |
| **Model Explainability (XAI)** | Visualisasi Global & Local **SHAP** | Top-20 Fitur Kata Kunci Pendorong Sentimen | Analisis SHAP per Aplikasi & per Aspek ABSA |
