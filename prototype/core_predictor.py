"""
Modul Preprocessing Teks, Inferensi Model, Ekstraksi 4 Aspek ABSA & Validasi Input
Digunakan oleh prototype/app.py dan prototype/test_prototype_scenarios.py
"""

import json
import os
import re
import joblib
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple, Union

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "genai_absa_pipeline.joblib")
SLANG_PATH = os.path.join(ROOT_DIR, "03_data_preparation", "slang_dictionary_id.json")
BENCHMARK_JSON_PATH = os.path.join(ROOT_DIR, "models", "modeling_results_summary.json")

# Kamus 4 Aspek Layanan Generative AI (Sesuai Bab 1 & Bab 3)
ASPECT_DEFINITIONS = {
    "A1_Akurasi_Kecerdasan": {
        "label": "A1: Akurasi & Kecerdasan AI",
        "keywords": {
            "pintar", "cerdas", "akurat", "tepat", "paham", "mengerti", "jawaban", "solusi",
            "membantu", "bantu", "tugas", "skripsi", "coding", "matematika", "halusinasi",
            "ngawur", "acau", "salah", "bodoh", "tersambung", "nyambung", "lupa", "sensor",
            "filter", "logika", "penjelasan", "informasi", "fakta", "ilmu", "belajar",
        },
    },
    "A2_Harga_Kuota": {
        "label": "A2: Harga & Limitasi Kuota",
        "keywords": {
            "gratis", "free", "bayar", "berbayar", "mahal", "murah", "harga", "langganan",
            "subscription", "plus", "pro", "kuota", "limit", "batas", "dibatasi", "habis",
            "koin", "token", "iklan", "premium", "biaya", "akses", "berlangganan", "hemat",
        },
    },
    "A3_Performa_UI_Server": {
        "label": "A3: Performa, UI/UX & Server",
        "keywords": {
            "cepat", "lambat", "lemot", "lelet", "loading", "server", "sibuk", "down",
            "error", "eror", "bug", "macet", "crash", "keluar", "login", "masuk", "akun",
            "update", "versi", "tampilan", "ui", "suara", "voice", "live", "jaringan",
            "koneksi", "berat", "ringan", "stabil", "verifikasi", "usia", "umur", "layar",
        },
    },
    "A4_Bahasa_Indonesia": {
        "label": "A4: Bahasa Indonesia & Konteks Lokal",
        "keywords": {
            "bahasa", "indonesia", "indo", "terjemahan", "translate", "logat", "aksen",
            "baku", "gaul", "lokal", "kaku", "english", "inggris", "pelafalan", "bicara",
            "kalimat", "kata", "percakapan", "ngobrol",
        },
    },
}

NEG_COMPLAIN_WORDS = {
    "error", "eror", "macet", "lemot", "lambat", "lelet", "limit", "batas", "dibatasi",
    "mahal", "bayar", "berbayar", "gagal", "kecewa", "jelek", "buruk", "rusak", "parah",
    "ngawur", "acau", "halusinasi", "bodoh", "sampah", "bug", "down", "sibuk", "keluar",
    "crash", "iklan", "pelit", "susah", "sulit", "ribet", "kaku", "ngaco", "hilang",
    "hapus", "filter", "sensor", "gangguan", "masalah", "kendala", "tolol", "mengecewakan",
}

POS_PRAISE_WORDS = {
    "bagus", "mantap", "keren", "membantu", "bantu", "terbaik", "puas", "suka",
    "pintar", "cerdas", "cepat", "mudah", "akurat", "hebat", "recommended",
    "rekomendasi", "berguna", "bermanfaat", "canggih", "good", "best", "nice",
    "love", "awesome", "great", "helpful", "jos", "joss", "mantab", "top",
}

NEGATION_WORDS = {"tidak", "tak", "bukan", "jangan", "kurang", "belum", "gak", "ga", "gk", "nggak", "tdk"}

MAX_CHAR_LIMIT = 2000


def load_slang_dictionary():
    if os.path.exists(SLANG_PATH):
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


SLANG_DICT = load_slang_dictionary()


def load_trained_artifact() -> Dict[str, Any]:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifact belum ditemukan di {MODEL_PATH}. Jalankan tools/build_models_and_notebooks.py terlebih dahulu."
        )
    res = joblib.load(MODEL_PATH)
    assert isinstance(res, dict)
    return res


def validate_review_input(raw_text):
    """
    Validasi input ketat sesuai syarat Bab 6.5 Template Laporan AM552 FTI UKSW:
    Menangani:
      - Input kosong / None / hanya spasi
      - Input hanya angka atau tanda baca tanpa huruf alfabet bermakna
      - Input ekstrem (> 2.000 karakter) -> dipotong otomatis dengan peringatan ramah pengguna
    """
    if raw_text is None:
        return False, "Input teks ulasan tidak boleh kosong (None). Silakan masukkan kalimat ulasan aplikasi AI.", ""

    text_str = str(raw_text).strip()
    if len(text_str) == 0:
        return False, "Teks ulasan masih kosong atau hanya berisi spasi. Silakan ketik ulasan Anda terlebih dahulu.", ""

    alpha_chars = re.findall(r"[a-zA-Z]", text_str)
    if len(alpha_chars) < 2:
        return (
            False,
            "Input tidak valid: Teks hanya berisi angka, simbol, atau 1 huruf tanpa kata bermakna. Masukkan minimal 1 kata ulasan dalam Bahasa Indonesia/Inggris.",
            "",
        )

    warning_msg = None
    if len(text_str) > MAX_CHAR_LIMIT:
        text_str = text_str[:MAX_CHAR_LIMIT]
        warning_msg = f"Peringatan: Teks ulasan terlalu panjang (>{MAX_CHAR_LIMIT} karakter). Sistem otomatis memproses {MAX_CHAR_LIMIT} karakter pertama."

    return True, warning_msg, text_str


def preprocess_text_indonesian(text, slang_dict=None):
    """Pipeline preprocessing yang 100% identik dengan Notebook 02 & 03."""
    if slang_dict is None:
        slang_dict = SLANG_DICT
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    text_lower = re.sub(r"http\S+|www\.\S+", " ", text_lower)
    text_lower = re.sub(r"(.)\1{2,}", r"\1\1", text_lower)
    tokens = re.findall(r"[a-z0-9]+", text_lower)
    norm_tokens = [slang_dict.get(t, t) for t in tokens]
    return " ".join(norm_tokens)


def analyze_aspects_and_discrepancy(raw_text, clean_text, star_rating=None, predicted_sentiment=None):
    """
    Mengekstrak 4 Aspek ABSA (A1..A4) beserta indikasi sentimen aspek dan
    mendeteksi fenomena Rating-Text Discrepancy.
    """
    tokens = clean_text.split()
    token_set = set(tokens)

    detected_aspects = []
    for code, info in ASPECT_DEFINITIONS.items():
        matched_kw = sorted(list(token_set.intersection(info["keywords"])))
        if matched_kw:
            detected_aspects.append(
                {
                    "aspect_code": code,
                    "aspect_label": info["label"],
                    "matched_keywords": matched_kw,
                }
            )

    if not detected_aspects:
        detected_aspects.append(
            {
                "aspect_code": "A0_General_Impression",
                "aspect_label": "A0: Kesan Umum Aplikasi (General Impression)",
                "matched_keywords": tokens[:3] if tokens else [],
            }
        )

    # Deteksi Rating-Text Discrepancy
    has_complaint = (
        len(token_set.intersection(NEG_COMPLAIN_WORDS)) > 0
        or ("tidak bisa" in clean_text)
        or ("tidak tersambung" in clean_text)
    )
    has_praise = len(token_set.intersection(POS_PRAISE_WORDS)) > 0
    has_negation = len(token_set.intersection(NEGATION_WORDS)) > 0

    discrepancy_status = "Tidak Berlaku (Rating bintang tidak diisi)"
    is_discrepancy = False

    if star_rating is not None:
        star_int = int(star_rating)
        if star_int >= 4 and (has_complaint or predicted_sentiment == "Negatif"):
            is_discrepancy = True
            discrepancy_status = (
                f"⚠️ DISCREPANCY TYPE-I TERDETEKSI: Rating Tinggi (⭐ {star_int}), "
                f"tetapi isi teks mengandung keluhan/sentimen negatif."
            )
        elif star_int <= 2 and (
            (has_praise and not has_complaint and not has_negation) or predicted_sentiment == "Positif"
        ):
            is_discrepancy = True
            discrepancy_status = (
                f"⚠️ DISCREPANCY TYPE-II TERDETEKSI: Rating Rendah (⭐ {star_int}), "
                f"tetapi isi teks bermuatan pujian positif."
            )
        else:
            discrepancy_status = f"✅ KONSISTEN: Rating bintang (⭐ {star_int}) selaras dengan isi ulasan teks."

    return detected_aspects, is_discrepancy, discrepancy_status


def explain_prediction_keywords(pipeline, clean_text, predicted_class_idx, top_k=6):
    """
    Mengekstrak kata kunci di dalam ulasan pengguna yang memberikan bobot kontribusi
    terbesar terhadap keputusan model (Local Feature Attribution).
    """
    try:
        vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        x_vec = vec.transform([clean_text])
        nonzero_indices = x_vec.nonzero()[1]
        if len(nonzero_indices) == 0:
            return []

        feature_names = vec.get_feature_names_out()
        coefs = None
        if hasattr(clf, "coef_"):
            coefs = clf.coef_
        elif hasattr(clf, "calibrated_classifiers_"):
            base_est = clf.calibrated_classifiers_[0].estimator
            if hasattr(base_est, "coef_"):
                coefs = base_est.coef_

        contributions = []
        for idx in nonzero_indices:
            term = feature_names[idx]
            tfidf_val = float(x_vec[0, idx])
            if coefs is not None:
                c_idx = min(predicted_class_idx, coefs.shape[0] - 1)
                weight = float(coefs[c_idx, idx]) * tfidf_val
            else:
                weight = tfidf_val
            contributions.append((term, round(weight, 4)))

        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions[:top_k]
    except Exception:
        return []


def predict_single_review(
    raw_text: str,
    star_rating: Optional[int] = None,
    app_name: str = "ChatGPT (OpenAI)",
    artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fungsi end-to-end untuk memprediksi satu ulasan pengguna."""
    is_valid, msg, validated_text = validate_review_input(raw_text)
    if not is_valid:
        return {
            "status": "INVALID_INPUT",
            "error_message": msg,
            "raw_text": raw_text,
        }

    if artifact is None:
        artifact = load_trained_artifact()
    assert artifact is not None

    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    label_display_map = artifact.get(
        "label_display_map",
        {
            "0_Negative": "Negatif",
            "1_Neutral": "Netral",
            "2_Positive": "Positif",
        },
    )

    clean_text = preprocess_text_indonesian(validated_text)
    if not clean_text.strip():
        return {
            "status": "INVALID_INPUT",
            "error_message": "Setelah pembersihan karakter simbol/noise, tidak ditemukan kata bahasa Indonesia/Inggris yang dapat dianalisis.",
            "raw_text": raw_text,
        }

    pred_raw = pipeline.predict([clean_text])[0]
    probas = pipeline.predict_proba([clean_text])[0]
    pred_idx = list(classes).index(pred_raw)
    confidence = float(probas[pred_idx])
    pred_label = label_display_map.get(pred_raw, str(pred_raw))

    detected_aspects, is_discrepancy, discrepancy_status = analyze_aspects_and_discrepancy(
        validated_text, clean_text, star_rating=star_rating, predicted_sentiment=pred_label
    )
    key_drivers = explain_prediction_keywords(pipeline, clean_text, pred_idx)

    return {
        "status": "SUCCESS",
        "warning_message": msg,
        "app_name": app_name,
        "raw_text": validated_text,
        "clean_text": clean_text,
        "star_rating": star_rating,
        "predicted_class_raw": pred_raw,
        "predicted_sentiment": pred_label,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            label_display_map.get(c, c): round(float(p), 4) for c, p in zip(classes, probas)
        },
        "detected_aspects": detected_aspects,
        "is_discrepancy": is_discrepancy,
        "discrepancy_status": discrepancy_status,
        "key_drivers": key_drivers,
    }


def predict_batch_dataframe(df_input, text_col="content", score_col=None, artifact=None):
    """Fungsi prediksi batch untuk fitur Upload CSV pada Tab 2 Streamlit."""
    if df_input is None or len(df_input) == 0:
        return None, "File CSV kosong (0 baris). Silakan unggah file CSV yang memiliki data ulasan."

    if text_col not in df_input.columns:
        return (
            None,
            f"Kolom teks '{text_col}' tidak ditemukan di CSV. Kolom yang tersedia: {', '.join(df_input.columns)}",
        )

    if artifact is None:
        artifact = load_trained_artifact()
    assert artifact is not None

    pipeline = artifact["pipeline"]
    classes = artifact["classes"]
    label_display_map = artifact.get(
        "label_display_map",
        {
            "0_Negative": "Negatif",
            "1_Neutral": "Netral",
            "2_Positive": "Positif",
        },
    )

    df_out = df_input.copy()
    df_out["clean_content"] = df_out[text_col].fillna("").astype(str).apply(preprocess_text_indonesian)
    valid_mask = df_out["clean_content"].str.len() >= 2

    if valid_mask.sum() == 0:
        return None, "Tidak ada baris pada kolom tersebut yang berisi teks ulasan valid."

    preds_raw = np.full(len(df_out), "INVALID", dtype=object)
    confs = np.zeros(len(df_out), dtype=float)

    valid_texts = df_out.loc[valid_mask, "clean_content"].values
    valid_preds = pipeline.predict(valid_texts)
    valid_probas = pipeline.predict_proba(valid_texts)

    preds_raw[valid_mask] = [label_display_map.get(p, str(p)) for p in valid_preds]
    confs[valid_mask] = np.round(np.max(valid_probas, axis=1), 4)

    df_out["predicted_sentiment"] = preds_raw
    df_out["confidence"] = confs

    # Tagging 4 aspek secara cepat
    for code, info in ASPECT_DEFINITIONS.items():
        kw_set = info["keywords"]
        df_out[code] = df_out["clean_content"].apply(
            lambda s: 1 if len(set(s.split()).intersection(kw_set)) > 0 else 0
        )

    if score_col and score_col in df_out.columns:
        disc_flags = []
        for s_val, c_txt, p_sent in zip(
            df_out[score_col], df_out["clean_content"], df_out["predicted_sentiment"]
        ):
            try:
                _, is_disc, _ = analyze_aspects_and_discrepancy(
                    c_txt, c_txt, star_rating=int(float(s_val)), predicted_sentiment=p_sent
                )
            except Exception:
                is_disc = False
            disc_flags.append(is_disc)
        df_out["is_discrepancy"] = disc_flags

    return df_out, None
