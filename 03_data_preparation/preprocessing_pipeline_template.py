"""
CRISP-DM FASE 3: DATA PREPARATION & ABSA TAGGING PIPELINE TEMPLATE
Penanggung Jawab Utama: Member 2 (Data Engineering, Preprocessing & ABSA)
"""
import json
import os
import re
import pandas as pd

RAW_CSV_PATH = "./dataset/RAW_DATASET_GENAI_INDONESIA_AM522B.csv"
SLANG_JSON_PATH = "./03_data_preparation/slang_dictionary_id.json"
OUTPUT_SAMPLE_PATH = "./03_data_preparation/sample_preprocessed_absa_1000.csv"

# Kamus Kata Kunci Aspek (ABSA) sesuai Project Charter
ASPECT_KEYWORDS = {
    "aspect_accuracy_intelligence": {
        "pintar", "cerdas", "akurat", "tepat", "paham", "mengerti", "jawaban", "solusi",
        "membantu", "bantu", "tugas", "skripsi", "coding", "matematika", "halusinasi",
        "ngawur", "acau", "salah", "bodoh", "tersambung", "lupa", "sensor", "filter",
    },
    "aspect_price_quota": {
        "gratis", "free", "bayar", "berbayar", "mahal", "murah", "harga", "langganan",
        "subscription", "plus", "pro", "kuota", "limit", "batas", "dibatasi", "habis",
        "koin", "token", "iklan", "premium",
    },
    "aspect_performance_bugs": {
        "cepat", "lambat", "loading", "server", "sibuk", "down", "error", "bug", "macet",
        "crash", "keluar", "login", "masuk", "akun", "update", "versi", "tampilan",
        "ui", "suara", "voice", "live", "jaringan", "koneksi", "berat", "ringan",
    },
    "aspect_indonesian_lang": {
        "bahasa", "indonesia", "terjemahan", "translate", "logat", "aksen", "baku",
        "gaul", "lokal", "kaku", "english", "pelafalan", "bicara",
    },
}


def load_slang_dict():
    if os.path.exists(SLANG_JSON_PATH):
        with open(SLANG_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def clean_and_normalize_text(text, slang_dict):
    """Membersihkan teks ulasan & menormalisasi kata slang Bahasa Indonesia."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Hapus URL dan mention
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Kurangi karakter berulang > 2 kali (contoh: 'bagusssss' -> 'baguss')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Ambil token alfabetik
    tokens = re.findall(r"[a-z]+", text)
    norm_tokens = [slang_dict.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def extract_aspects(clean_text):
    """Mendeteksi keberadaan 4 aspek utama dalam teks ulasan yang telah dinormalisasi."""
    tokens = set(clean_text.split())
    result = {}
    for aspect_col, kw_set in ASPECT_KEYWORDS.items():
        result[aspect_col] = 1 if len(tokens.intersection(kw_set)) > 0 else 0
    return result


if __name__ == "__main__":
    print("Memuat kamus slang dan sampel 1.000 baris untuk uji pipeline Fase 3...")
    slang_dict = load_slang_dict()
    df_sample = pd.read_csv(RAW_CSV_PATH, nrows=1000)
    df_sample["clean_content"] = df_sample["content"].apply(lambda x: clean_and_normalize_text(x, slang_dict))

    aspect_records = df_sample["clean_content"].apply(extract_aspects).apply(pd.Series)
    df_out = pd.concat([df_sample, aspect_records], axis=1)
    df_out.to_csv(OUTPUT_SAMPLE_PATH, index=False, encoding="utf-8-sig")
    print(f"[OK] Sampel hasil preprocessing & tagging ABSA disimpan di: {OUTPUT_SAMPLE_PATH}")
