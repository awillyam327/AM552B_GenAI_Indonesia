# Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genai-indonesia-am552b.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/awillyam327/AM552B_GenAI_Indonesia)

> **Live Prototype**: [https://genai-indonesia-am552b.streamlit.app](https://genai-indonesia-am552b.streamlit.app)  
> **Mata Kuliah**: Data Science of IS (AM552) — Proyek Akhir Kelompok  
> **Komposisi Tim**: 3 Anggota (Member 1: Project Lead & Modeling/SHAP, Member 2: Data Engineering/Preprocessing/ABSA, Member 3: EDA/Topic Modeling/Reporting)  
> **Aset Raw Data**: `71.221` Ulasan Unik Google Play Store Indonesia (`id_ID`) pada 8 Aplikasi Generative AI (`100% Real Scraped Data`, `0% Duplikasi reviewId`)

---

## 📌 1. Gambaran Umum Proyek (Project Overview)

Proyek ini mengevaluasi pengalaman pengguna (*User Experience*) nyata masyarakat Indonesia terhadap 8 aplikasi mobile Generative AI terpopuler di Google Play Store:
1. **ChatGPT (OpenAI)** (`22.000` ulasan, rating rata-rata `4.46`)
2. **Google Gemini** (`12.600` ulasan, rating rata-rata `4.29`)
3. **Character.AI** (`12.000` ulasan, rating rata-rata `1.58`)
4. **Perplexity - Ask Anything** (`8.043` ulasan, rating rata-rata `4.54`)
5. **Microsoft Copilot** (`5.459` ulasan, rating rata-rata `4.65`)
6. **DeepSeek - AI Assistant** (`5.374` ulasan, rating rata-rata `3.77`)
7. **Claude by Anthropic** (`4.009` ulasan, rating rata-rata `3.85`)
8. **Poe - Fast AI Chat** (`1.736` ulasan, rating rata-rata `4.66`)

Penelitian ini membedah ulasan ke dalam **4 Dimensi Aspek Layanan (ABSA)**:
- **A1 — Akurasi & Kecerdasan AI** (*Reasoning, Task Assistance vs. Hallucination/Ngawur*)
- **A2 — Harga & Limitasi Kuota Gratis** (*Rate Limit, Subscription Plus/Pro, Paywall, Ads*)
- **A3 — Performa, UI/UX & Kecepatan Server** (*Server Busy/Down, Force Close, Login, Live Voice*)
- **A4 — Dukungan Bahasa Indonesia** (*Indonesian Fluency, Local Slang vs. Kaku/Inggris*)

Serta meneliti anomali **Rating-Text Discrepancy** (pengguna memberi Bintang 5 tetapi isi teks mengeluh kritis, atau sebaliknya).

---

## 📂 2. Struktur Direktori Berbasis Standar CRISP-DM

```text
Project final/
├── README.md                                   # Dokumentasi utama proyek & panduan navigasi
├── requirements.txt                            # Daftar dependensi Python terverifikasi
├── .gitignore                                  # Aturan pengabaian file sementara & lock files
├── MASTER_EXECUTION_PLAN.md                    # Master Plan 7 Fase & Protokol Anti-Leakage
├── MASTER_EXECUTION_PLAN.docx                  # Master Plan versi Word siap cetak/sidang
├── TEAM_ONBOARDING_GUIDE.md                    # Panduan onboarding & kurikulum kilat 3 anggota
├── TEAM_ONBOARDING_GUIDE.docx                  # Panduan onboarding versi Word
├── Sinopsis_GenerativeAI_Indonesia_682024046.docx # Dokumen sinopsis pengajuan proyek (NIM)
├── Sinopsis_GenerativeAI_Indonesia_682024046.md   # Dokumen sinopsis pengajuan proyek (Markdown)
│
├── scrape_genai_indonesia.py                   # Script penarikan data Google Play Store (71.221 baris)
├── verify_dataset.py                           # Script audit integritas dimensi, rating, & baris CSV
│
├── dataset/
│   ├── RAW_DATASET_GENAI_INDONESIA_AM522B.csv  # Dataset mentah utama (71.221 baris unik, 11 kolom)
│   ├── audit_summary.json                      # Ringkasan audit dimensi & distribusi dataset
│   └── checkpoints/                            # Checkpoint CSV per aplikasi hasil scraping
│
├── 01_business_understanding/
│   └── project_charter_dan_definisi_aspek.md   # Rumusan masalah (RQ), hipotesis, & kamus 4 aspek ABSA
│
├── 02_data_understanding/
│   ├── run_initial_eda.py                      # Script otomatisasi visualisasi EDA & audit missing
│   ├── eda_summary_report.md                   # Laporan statistik panjang teks, rating, & unigram/bigram
│   └── figures/                                # Grafik resolusi tinggi (distribusi, rating, discrepancy)
│
├── 03_data_preparation/
│   ├── slang_dictionary_id.json                # Kamus pemetaan kata gaul/slang ke Bahasa Indonesia baku
│   ├── preprocessing_pipeline_template.py      # Template pipeline normalisasi teks & tagging ABSA
│   └── sample_preprocessed_absa_1000.csv       # Sampel 1.000 baris hasil uji pipeline
│
├── 04_modeling/
│   └── README.md                               # Panduan Track A (Supervised) & Track B (Topic Modeling)
│
├── 05_evaluation/
│   ├── metric_sanity_audit.py                  # Modul 6 pemeriksaan forensik metrik & anti-leakage
│   ├── metric_sanity_audit_report.md           # Laporan audit komparasi Dummy vs Multi-Slice Test
│   ├── audit_results.json                      # Metrik eksak komprehensif (Macro F1, MCC, PR-AUC)
│   ├── confident_errors_fp_fn.csv              # Top 20 bukti nyata Rating-Text Discrepancy
│   └── README.md                               # Panduan evaluasi & interpretasi SHAP
│
├── 06_deployment_reporting/
│   └── README.md                               # Matriks komparasi akhir 8 aplikasi & materi laporan
│
└── tools/
    ├── generate_master_execution_plan_docx.py  # Generator konversi Master Execution Plan ke Word
    ├── generate_team_onboarding.py             # Generator konversi Team Onboarding Guide ke Word
    └── generate_synopsis_docs.py               # Generator konversi Dokumen Sinopsis ke Word
```

---

## 👥 3. Pembagian Tugas Tim (3 Anggota)

| Anggota Tim | Peran Utama | Fokus Fase CRISP-DM | Modul / Script Utama |
| :--- | :--- | :--- | :--- |
| **Member 1 (Project Lead)** | Predictive Modeling & SHAP Explainability | Fase 0, 1, 4 (Track A), 5 | `05_evaluation/metric_sanity_audit.py`<br>`04_modeling/` (CatBoost, XGBoost, IndoBERT) |
| **Member 2** | Data Engineering, Preprocessing & ABSA | Fase 0, 2, 3, 4 (ABSA Matrix) | `03_data_preparation/preprocessing_pipeline_template.py`<br>`03_data_preparation/slang_dictionary_id.json` |
| **Member 3** | EDA, Topic Modeling & Reporting | Fase 0, 2, 4 (Track B), 6 | `02_data_understanding/run_initial_eda.py`<br>`04_modeling/` (BERTopic & LDA) |

---

## 🚀 4. Panduan Eksekusi Cepat (Quickstart)

### A. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### B. Verifikasi Dataset Utama (71.221 Baris)
```bash
python verify_dataset.py
```

### C. Menjalankan Exploratory Data Analysis (EDA) & Membuat Grafik
```bash
python 02_data_understanding/run_initial_eda.py
```
*Hasil grafik disimpan di folder `02_data_understanding/figures/`.*

### D. Menguji Pipeline Preprocessing & Multi-Label ABSA Tagging
```bash
python 03_data_preparation/preprocessing_pipeline_template.py
```

### E. Menjalankan 6 Pemeriksaan Forensik Audit Metrik & Anti-Leakage
```bash
python 05_evaluation/metric_sanity_audit.py
```
*Hasil evaluasi dan laporan forensik disimpan di `05_evaluation/metric_sanity_audit_report.md`.*

---

## 📄 5. Dokumen Indeks Utama

1. **[MASTER_EXECUTION_PLAN.md](MASTER_EXECUTION_PLAN.md)** / **[MASTER_EXECUTION_PLAN.docx](MASTER_EXECUTION_PLAN.docx)**: Rencana induk 7 fase pengerjaan, matriks dependensi tugas, dan protokol audit "angka terlalu tinggi".
2. **[TEAM_ONBOARDING_GUIDE.md](TEAM_ONBOARDING_GUIDE.md)** / **[TEAM_ONBOARDING_GUIDE.docx](TEAM_ONBOARDING_GUIDE.docx)**: Panduan konsep intuitif tanpa jargon rumit dan kurikulum belajar kilat per peran.
3. **[Sinopsis_GenerativeAI_Indonesia_682024046.docx](Sinopsis_GenerativeAI_Indonesia_682024046.docx)**: Proposal formal pengajuan proyek tugas akhir AM522B.
