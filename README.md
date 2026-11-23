# Analisis Komparatif Sentimen Berbasis Aspek (ABSA) dan Pemodelan Topik Pengalaman Pengguna terhadap Ekosistem Aplikasi Mobile Generative AI di Indonesia

Repositori ini berisi kode program, dataset, eksperimen pemodelan, dan prototipe aplikasi web untuk Tugas Akhir mata kuliah **AM552 - Data Science of IS** (Program Studi S1 Sistem Informasi, Fakultas Teknologi Informasi, Universitas Kristen Satya Wacana).

- **Aplikasi Web (Streamlit)**: [https://genai-indonesia-am552b.streamlit.app](https://genai-indonesia-am552b.streamlit.app)
- **Repositori GitHub**: [https://github.com/awillyam327/AM552B_GenAI_Indonesia](https://github.com/awillyam327/AM552B_GenAI_Indonesia)

---

## Anggota Kelompok

| Nama Mahasiswa | NIM | Pembagian Tugas |
| :--- | :---: | :--- |
| **Arthur Willyam Liang** | 682024046 | Pemodelan klasifikasi sentimen, *hyperparameter tuning* 5-fold CV, evaluasi model, dan deployment Streamlit |
| **Brendan Wilson Santoso** | 682024016 | *Data preprocessing*, kamus normalisasi slang bahasa Indonesia, pelabelan aspek ABSA, dan pengujian aplikasi |
| **Tiara Agustien Kusumawati** | 682024059 | *Exploratory Data Analysis* (EDA), pemodelan topik (*Latent Dirichlet Allocation*), penyusunan laporan dan slide presentasi |

---

## Ringkasan Penelitian

Penelitian ini menganalisis **71.221 ulasan pengguna** berbahasa Indonesia di Google Play Store pada 8 aplikasi mobile Generative AI:
1. **ChatGPT (OpenAI)** — 22.000 ulasan (rata-rata rating 4,46)
2. **Google Gemini** — 12.600 ulasan (rata-rata rating 4,29)
3. **Character.AI** — 12.000 ulasan (rata-rata rating 1,58)
4. **Perplexity - Ask Anything** — 8.043 ulasan (rata-rata rating 4,54)
5. **Microsoft Copilot** — 5.459 ulasan (rata-rata rating 4,65)
6. **DeepSeek - AI Assistant** — 5.374 ulasan (rata-rata rating 3,77)
7. **Claude by Anthropic** — 4.009 ulasan (rata-rata rating 3,85)
8. **Poe - Fast AI Chat** — 1.736 ulasan (rata-rata rating 4,66)

Ulasan diklasifikasikan ke dalam 3 kelas sentimen (**Positif**, **Netral**, **Negatif**) dan dipetakan ke dalam 4 dimensi aspek layanan (**ABSA**):
- **A1 (Akurasi & Kecerdasan AI)**: kualitas jawaban, penalaran, bantuan tugas/koding, atau masalah halusinasi.
- **A2 (Harga & Limitasi Kuota)**: batas kuota harian versi gratis dan harga langganan.
- **A3 (Performa, UI/UX & Server)**: kecepatan aplikasi, kendala *server sibuk*, gagal *login*, atau bug antarmuka.
- **A4 (Bahasa Indonesia)**: keluwesan bahasa Indonesia dan kualitas terjemahan.

Selain itu, proyek ini mengidentifikasi fenomena **Rating-Text Discrepancy**, yaitu kondisi ketika pengguna memberikan rating bintang tinggi (Bintang 4–5) namun isi teks ulasannya berupa keluhan teknis.

---

## Struktur Folder

```text
├── README.md
├── requirements.txt
├── scrape_genai_indonesia.py                   # Script pengambilan data ulasan Google Play Store
├── verify_dataset.py                           # Script pengecekan jumlah baris dan kolom dataset
│
├── dataset/
│   ├── RAW_DATASET_GENAI_INDONESIA_AM522B.csv  # Dataset utama (71.221 baris, 11 kolom)
│   └── audit_summary.json                      # Ringkasan statistik dataset
│
├── notebooks/
│   ├── 01_Data_Understanding_dan_EDA.ipynb     # Notebook Bab 2: Pemahaman Data & EDA
│   ├── 02_Data_Preparation_dan_ABSA.ipynb      # Notebook Bab 3: Preprocessing Slang & Pelabelan ABSA
│   └── 03_Modeling_Tuning_dan_Evaluasi.ipynb   # Notebook Bab 4 & 5: Pemodelan, Tuning 5-Fold & Evaluasi
│
├── 01_business_understanding/
│   └── project_charter_dan_definisi_aspek.md   # Rumusan masalah dan kamus kata kunci 4 aspek ABSA
│
├── 02_data_understanding/
│   ├── run_initial_eda.py                      # Script pembuatan grafik distribusi data
│   ├── eda_summary_report.md                   # Ringkasan hasil eksplorasi data
│   └── figures/                                # Grafik distribusi aplikasi, rating, dan discrepancy
│
├── 03_data_preparation/
│   ├── slang_dictionary_id.json                # Kamus normalisasi kata tidak baku/slang (145 entri)
│   ├── preprocessing_pipeline_template.py      # Modul pembersihan teks dan ekstraksi aspek
│   └── sample_preprocessed_absa_1000.csv       # Sampel 1.000 baris hasil preprocessing
│
├── 04_modeling/
│   ├── train_sentiment_models.py               # Script pelatihan model klasifikasi & ekspor .joblib
│   ├── run_topic_modeling_experiment.py        # Script pemodelan topik LDA (5 klaster topik)
│   ├── topic_modeling_report.md                # Ringkasan hasil klaster topik LDA
│   └── figures/                                # Visualisasi kata kunci per topik LDA
│
├── 05_evaluation/
│   ├── metric_sanity_audit.py                  # Script evaluasi model dan pengecekan data leakage
│   ├── metric_sanity_audit_report.md           # Ringkasan hasil evaluasi metrik pada data uji
│   ├── audit_results.json                      # Hasil metrik lengkap dalam format JSON
│   ├── confident_errors_fp_fn.csv              # Contoh kasus kesalahan prediksi & discrepancy
│   └── figures/                                # Plot Confusion Matrix dan Feature Attribution
│
├── 06_deployment_reporting/
│   ├── 682024046_KEL_DS_Laporan.docx           # Laporan Akhir Proyek (Format 7 Bab FTI UKSW)
│   ├── 682024046_KEL_DS_Presentasi.pdf         # Slide Presentasi Kelompok (PDF)
│   ├── 682024046_KEL_DS_Presentasi.pptx        # Slide Presentasi Kelompok (PowerPoint)
│   └── figures/                                # Tangkapan layar aplikasi prototype Streamlit
│
├── models/
│   ├── genai_absa_pipeline.joblib              # File model terbaik tersimpan (Logistic Regression + TF-IDF)
│   └── modeling_results_summary.json           # Ringkasan perbandingan hasil model
│
└── prototype/
    ├── app.py                                  # Aplikasi web interaktif Streamlit (3 Tab)
    ├── core_predictor.py                       # Modul inferensi model, validasi input, dan ABSA
    ├── test_prototype_scenarios.py             # Pengujian 3 skenario input (Normal, Ekstrem, Salah Format)
    └── requirements.txt                        # Dependensi deployment Streamlit Cloud
```

---

## Cara Menjalankan Proyek secara Lokal

1. **Instalasi pustaka Python**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Menjalankan Jupyter Notebook**:
   Buka folder `notebooks/` menggunakan VS Code atau Jupyter Lab, lalu jalankan berurutan dari `01_Data_Understanding_dan_EDA.ipynb`, `02_Data_Preparation_dan_ABSA.ipynb`, hingga `03_Modeling_Tuning_dan_Evaluasi.ipynb`.

3. **Menjalankan Aplikasi Prototype Streamlit**:
   ```bash
   streamlit run prototype/app.py
   ```
   Aplikasi akan berjalan di `http://localhost:8501`.
