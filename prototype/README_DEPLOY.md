# Panduan Deploy Cepat Prototype ke Streamlit Community Cloud (5 Menit)

Prototype web aplikasi **GenAI Indonesia ABSA & Discrepancy Analyzer** ini dirancang agar siap di-deploy gratis ke **Streamlit Community Cloud** sesuai ketentuan **Bab 6 Template Laporan AM552 - Data Science of IS (FTI UKSW)**.

---

## 1. Cara Menjalankan secara Lokal di Komputer Anda

Buka terminal pada root folder proyek, lalu jalankan:

```bash
pip install streamlit joblib scikit-learn pandas numpy
streamlit run prototype/app.py
```

Aplikasi akan langsung terbuka di browser pada alamat `http://localhost:8501`.

---

## 2. Langkah Deploy ke Streamlit Community Cloud (Gratis)

1. **Push Repository ke GitHub**:
   - Pastikan file berikut ikut ter-push ke repository GitHub kelompok Anda:
     * `prototype/app.py`
     * `prototype/core_predictor.py`
     * `prototype/requirements.txt`
     * `models/genai_absa_pipeline.joblib`
     * `models/modeling_results_summary.json`
     * `03_data_preparation/slang_dictionary_id.json`
2. **Buka Streamlit Community Cloud**:
   - Kunjungi [https://share.streamlit.io](https://share.streamlit.io) dan login menggunakan akun GitHub.
3. **Klik "Create app" / "New app"**:
   - **Repository**: Pilih repository GitHub proyek Anda.
   - **Branch**: `main`
   - **Main file path**: `prototype/app.py`
4. **Klik "Deploy!"**:
   - Dalam 1–2 menit, aplikasi Anda akan live dengan URL publik (contoh: `https://682024046-kel-ds-genai-absa.streamlit.app`).
   - Salin URL tersebut ke **Halaman Judul** dan **Bab 6** pada dokumen `682024046_KEL_DS_Laporan.docx`.
