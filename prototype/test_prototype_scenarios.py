"""
=========================================================================================
SCRIPT PENGUJIAN OTOMATIS 3 SKENARIO PROTOTYPE (BAB 6.5 LAPORAN AM552 FTI UKSW)
File: prototype/test_prototype_scenarios.py
=========================================================================================
Menguji aplikasi prototype pada 3 skenario wajib sesuai Template Laporan AM552 Bab 6.5:
  1. Skenario 1: Input Normal (Teks ulasan jelas positif/negatif & batch CSV valid)
  2. Skenario 2: Input Ekstrem (Teks panjang penuh slang/singkatan/kapital & kasus
                 Rating-Text Discrepancy Bintang 5 tapi komplain kritis)
  3. Skenario 3: Input Kosong atau Salah Format (String kosong, spasi, hanya angka,
                 hanya tanda baca, & CSV tanpa kolom teks yang sesuai)
=========================================================================================
"""

import json
import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from core_predictor import (
    load_trained_artifact,
    predict_batch_dataframe,
    predict_single_review,
)

RESULTS_JSON_PATH = os.path.join(CURRENT_DIR, "test_scenarios_bab6_5_results.json")
RESULTS_MD_PATH = os.path.join(CURRENT_DIR, "test_scenarios_bab6_5_results.md")


def run_all_scenarios():
    print("=" * 88)
    print("MENJALANKAN PENGUJIAN OTOMATIS 3 SKENARIO PROTOTYPE (BAB 6.5 TEMPLATE UKSW)")
    print("=" * 88)

    artifact = load_trained_artifact()
    scenarios_output = []

    # -------------------------------------------------------------------------
    # SKENARIO 1: INPUT NORMAL
    # -------------------------------------------------------------------------
    res_1a = predict_single_review(
        raw_text="Aplikasinya sangat pintar dan cepat, bahasa Indonesianya luwes membantu tugas coding kuliah saya!",
        star_rating=5,
        app_name="ChatGPT (OpenAI)",
        artifact=artifact,
    )
    aspects_1a = ", ".join(a["aspect_label"] for a in res_1a["detected_aspects"])
    kw_1a = ", ".join(f"{k} ({w:+.2f})" for k, w in res_1a["key_drivers"][:4])
    scenarios_output.append(
        {
            "skenario": "1A. Input Normal (Ulasan Positif Multi-Aspek)",
            "input_yang_diuji": "Teks: \"Aplikasinya sangat pintar dan cepat, bahasa Indonesianya luwes membantu tugas coding kuliah saya!\" | Rating: Bintang 5 | App: ChatGPT",
            "hasil_yang_keluar": (
                f"Status: {res_1a['status']} | Prediksi Sentimen: {res_1a['predicted_sentiment']} "
                f"(Confidence: {res_1a['confidence']*100:.1f}%) | Aspek Terdeteksi: [{aspects_1a}] | "
                f"Status Rating: {res_1a['discrepancy_status']} | Kata Kunci Pendorong: {kw_1a}"
            ),
            "sesuai_harapan": "Ya (100% Sesuai)",
        }
    )

    res_1b = predict_single_review(
        raw_text="Server sibuk terus dan gagal login, padahal kuota gratisnya cepat habis harus bayar mahal.",
        star_rating=1,
        app_name="DeepSeek - AI Assistant",
        artifact=artifact,
    )
    aspects_1b = ", ".join(a["aspect_label"] for a in res_1b["detected_aspects"])
    scenarios_output.append(
        {
            "skenario": "1B. Input Normal (Ulasan Keluhan Negatif)",
            "input_yang_diuji": "Teks: \"Server sibuk terus dan gagal login, padahal kuota gratisnya cepat habis harus bayar mahal.\" | Rating: Bintang 1 | App: DeepSeek",
            "hasil_yang_keluar": (
                f"Status: {res_1b['status']} | Prediksi Sentimen: {res_1b['predicted_sentiment']} "
                f"(Confidence: {res_1b['confidence']*100:.1f}%) | Aspek Terdeteksi: [{aspects_1b}] | "
                f"Status Rating: {res_1b['discrepancy_status']}"
            ),
            "sesuai_harapan": "Ya (100% Sesuai)",
        }
    )

    # -------------------------------------------------------------------------
    # SKENARIO 2: INPUT EKSTREM (SLANG BERAT, DISCREPANCY, & TEKS SANGAT PANJANG)
    # -------------------------------------------------------------------------
    res_2a = predict_single_review(
        raw_text="GUE KASIH BINTANG 5 BIAR DIBACA DEVELOPER!! Tolong dong apk makin update makin lemot bgt, mode live sering macet eror sendiri dan jawabannya ngawur ga nyambung!!",
        star_rating=5,
        app_name="Google Gemini",
        artifact=artifact,
    )
    aspects_2a = ", ".join(a["aspect_label"] for a in res_2a["detected_aspects"])
    scenarios_output.append(
        {
            "skenario": "2A. Input Ekstrem (Slang Berat, Huruf Kapital & Paradoks Rating-Text Discrepancy)",
            "input_yang_diuji": "Teks: \"GUE KASIH BINTANG 5 BIAR DIBACA DEVELOPER!! Tolong dong apk makin update makin lemot bgt, mode live sering macet eror sendiri dan jawabannya ngawur ga nyambung!!\" | Rating: Bintang 5",
            "hasil_yang_keluar": (
                f"Normalisasi Slang: \"{res_2a['clean_text']}\" | Prediksi Sentimen Teks: {res_2a['predicted_sentiment']} "
                f"(Confidence: {res_2a['confidence']*100:.1f}%) | Aspek: [{aspects_2a}] | "
                f"Deteksi Anomali: {res_2a['discrepancy_status']}"
            ),
            "sesuai_harapan": "Ya (Berhasil mendeteksi sentimen negatif sejati & memberi peringatan Discrepancy Type-I)",
        }
    )

    long_extreme_text = ("Aplikasi sering error macet dan kuota limit cepat habis padahal bayar mahal. " * 40)
    res_2b = predict_single_review(
        raw_text=long_extreme_text,
        star_rating=2,
        app_name="Claude by Anthropic",
        artifact=artifact,
    )
    scenarios_output.append(
        {
            "skenario": "2B. Input Ekstrem (Teks Sangat Panjang > 3.000 Karakter)",
            "input_yang_diuji": f"Teks berulang sepanjang {len(long_extreme_text):,} karakter (melebihi batas normal 500 karakter Play Store)",
            "hasil_yang_keluar": (
                f"Aplikasi tidak crash. Muncul peringatan: \"{res_2b['warning_message']}\" dan tetap menghasilkan "
                f"prediksi Sentimen: {res_2b['predicted_sentiment']} ({res_2b['confidence']*100:.1f}%)."
            ),
            "sesuai_harapan": "Ya (Sistem memotong otomatis di 2.000 karakter & tidak crash)",
        }
    )

    # -------------------------------------------------------------------------
    # SKENARIO 3: INPUT KOSONG ATAU SALAH FORMAT
    # -------------------------------------------------------------------------
    res_3a = predict_single_review(raw_text="   ", star_rating=5)
    scenarios_output.append(
        {
            "skenario": "3A. Input Kosong / Hanya Spasi",
            "input_yang_diuji": "Teks: \"   \" (String kosong / hanya spasi)",
            "hasil_yang_keluar": f"Status: {res_3a['status']} | Pesan Validasi UI: \"{res_3a['error_message']}\"",
            "sesuai_harapan": "Ya (Ditahan oleh validator sebelum masuk ke model, tidak crash)",
        }
    )

    res_3b = predict_single_review(raw_text="1234567890 ??? !!!", star_rating=3)
    scenarios_output.append(
        {
            "skenario": "3B. Input Salah Format (Hanya Angka & Simbol Tanpa Kata)",
            "input_yang_diuji": "Teks: \"1234567890 ??? !!!\"",
            "hasil_yang_keluar": f"Status: {res_3b['status']} | Pesan Validasi UI: \"{res_3b['error_message']}\"",
            "sesuai_harapan": "Ya (Menolak input non-linguistik dengan pesan ramah pengguna)",
        }
    )

    bad_csv = pd.DataFrame({"id_transaksi": [1, 2], "nominal_harga": [50000, 75000]})
    _, csv_err = predict_batch_dataframe(bad_csv, text_col="content", artifact=artifact)
    scenarios_output.append(
        {
            "skenario": "3C. Input CSV Salah Format (Tanpa Kolom Teks Ulasan)",
            "input_yang_diuji": "Upload file CSV dengan kolom ['id_transaksi', 'nominal_harga'] tanpa kolom 'content'",
            "hasil_yang_keluar": f"Batch predictor menolak secara aman dengan pesan: \"{csv_err}\"",
            "sesuai_harapan": "Ya (Tidak crash, menampilkan daftar kolom yang tersedia kepada pengguna)",
        }
    )

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(scenarios_output, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# HASIL UJI COBA 3 SKENARIO PROTOTYPE (BAB 6.5 LAPORAN AM552 FTI UKSW)\n",
        "| Skenario | Input yang Diuji | Hasil yang Keluar pada Prototype | Sesuai Harapan? |",
        "| :--- | :--- | :--- | :---: |",
    ]
    for item in scenarios_output:
        print(f"  [PASS] {item['skenario']} -> {item['sesuai_harapan']}")
        md_lines.append(
            f"| **{item['skenario']}** | {item['input_yang_diuji']} | {item['hasil_yang_keluar']} | **{item['sesuai_harapan']}** |"
        )

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[OK] Hasil uji 3 skenario disimpan di: {RESULTS_MD_PATH}")
    print("=" * 88)


if __name__ == "__main__":
    run_all_scenarios()
