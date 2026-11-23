# Panduan Menjalankan Prototype Aplikasi Web (Streamlit)

Aplikasi web ini memuat model klasifikasi sentimen `models/genai_absa_pipeline.joblib` dan kamus normalisasi slang `03_data_preparation/slang_dictionary_id.json` yang digunakan pada eksperimen notebook.

## 1. Menjalankan secara Lokal

Dari direktori utama proyek, jalankan perintah berikut:

```bash
pip install -r requirements.txt
streamlit run prototype/app.py
```

Aplikasi dapat diakses melalui browser di `http://localhost:8501`.

## 2. Deployment di Streamlit Community Cloud

- **URL Publik**: [https://genai-indonesia-am552b.streamlit.app](https://genai-indonesia-am552b.streamlit.app)
- **Main file path**: `prototype/app.py`
- **Dependensi**: `prototype/requirements.txt`
