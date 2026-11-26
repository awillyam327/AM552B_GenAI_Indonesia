"""
Generator ringkasan deret waktu harian & keluhan bertanggal untuk Dashboard Komparasi.
Menghasilkan:
  - models/dashboard_daily_metrics.csv
  - models/dashboard_dated_complaints.csv
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTOTYPE_DIR = os.path.join(ROOT_DIR, "prototype")
if PROTOTYPE_DIR not in sys.path:
    sys.path.insert(0, PROTOTYPE_DIR)

from core_predictor import ASPECT_DEFINITIONS, preprocess_text_indonesian

RAW_CSV_PATH = os.path.join(ROOT_DIR, "dataset", "RAW_DATASET_GENAI_INDONESIA_AM522B.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "genai_absa_pipeline.joblib")
DAILY_OUT_PATH = os.path.join(ROOT_DIR, "models", "dashboard_daily_metrics.csv")
COMPLAINTS_OUT_PATH = os.path.join(ROOT_DIR, "models", "dashboard_dated_complaints.csv")

ASPECT_SHORT_LABELS = [
    ("A1_Akurasi_Kecerdasan", "A1: Akurasi", "a1"),
    ("A2_Harga_Kuota", "A2: Harga/Kuota", "a2"),
    ("A3_Performa_UI_Server", "A3: Server/UI", "a3"),
    ("A4_Bahasa_Indonesia", "A4: Bahasa", "a4"),
]


def main() -> None:
    print("Memuat dataset mentah dan model pipeline...")
    df = pd.read_csv(
        RAW_CSV_PATH,
        usecols=["app_name", "content", "score", "thumbsUpCount", "at"],
        low_memory=False,
    )
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]

    dt = pd.to_datetime(df["at"], errors="coerce")
    df = df[dt.notna()].copy()
    df["date"] = dt[dt.notna()].dt.strftime("%Y-%m-%d")

    print(f"Menormalisasi dan memprediksi {len(df):,} ulasan...")
    df["clean_content"] = df["content"].fillna("").astype(str).apply(preprocess_text_indonesian)
    preds = pipeline.predict(df["clean_content"].values)
    df["pred"] = preds

    score_num = pd.to_numeric(df["score"], errors="coerce").fillna(3.0).astype(float)
    df["score_num"] = score_num
    df["n_reviews"] = 1
    df["sum_score"] = score_num

    disc_high_neg = (score_num >= 4) & (df["pred"] == "0_Negative")
    disc_low_pos = (score_num <= 2) & (df["pred"] == "2_Positive")
    df["disc_high_neg"] = disc_high_neg.astype(int)
    df["disc_low_pos"] = disc_low_pos.astype(int)

    adj_score = np.where(disc_high_neg, 1.5, np.where(disc_low_pos, 4.5, score_num))
    df["sum_adj_score"] = adj_score

    is_pos = df["pred"] == "2_Positive"
    is_neg = df["pred"] == "0_Negative"

    token_sets = df["clean_content"].apply(lambda s: set(s.split()))
    for code, _, prefix in ASPECT_SHORT_LABELS:
        kw_set = ASPECT_DEFINITIONS[code]["keywords"]
        hit = token_sets.apply(lambda ts: len(ts.intersection(kw_set)) > 0)
        df[f"{prefix}_n"] = hit.astype(int)
        df[f"{prefix}_pos"] = (hit & is_pos).astype(int)
        df[f"{prefix}_neg"] = (hit & is_neg).astype(int)

    agg_cols = [
        "n_reviews", "sum_score", "sum_adj_score", "disc_high_neg", "disc_low_pos",
        "a1_n", "a1_pos", "a1_neg",
        "a2_n", "a2_pos", "a2_neg",
        "a3_n", "a3_pos", "a3_neg",
        "a4_n", "a4_pos", "a4_neg",
    ]
    df_daily = (
        df.groupby(["date", "app_name"], as_index=False)[agg_cols]
        .sum()
        .sort_values(["date", "app_name"])
    )
    df_daily["sum_score"] = df_daily["sum_score"].round(2)
    df_daily["sum_adj_score"] = df_daily["sum_adj_score"].round(2)
    df_daily.to_csv(DAILY_OUT_PATH, index=False, encoding="utf-8")

    # Keluhan bertanggal dengan Thumbs Up tertinggi (per aplikasi + global top)
    df["thumbs"] = pd.to_numeric(df["thumbsUpCount"], errors="coerce").fillna(0).astype(int)
    complaint_mask = (df["pred"].isin(["0_Negative", "1_Neutral"]) | disc_high_neg) & (df["thumbs"] > 0)
    df_comp = df[complaint_mask].sort_values("thumbs", ascending=False)

    per_app_parts = []
    for _, grp in df_comp.groupby("app_name", sort=False):
        per_app_parts.append(grp.head(50))
    df_top_comp = (
        pd.concat(per_app_parts, ignore_index=True)
        .drop_duplicates(subset=["app_name", "date", "content"])
        .sort_values("thumbs", ascending=False)
        .head(400)
        .copy()
    )

    def _aspect_tags(row: pd.Series) -> str:
        tags = [lbl for _, lbl, pref in ASPECT_SHORT_LABELS if row[f"{pref}_n"] == 1]
        return ", ".join(tags) if tags else "Kesan Umum"

    def _snippet(txt: Any) -> str:
        s = " ".join(str(txt).split())
        return (s[:132] + "...") if len(s) > 135 else s

    df_comp_out = pd.DataFrame(
        {
            "Tanggal": df_top_comp["date"],
            "Aplikasi": df_top_comp["app_name"],
            "Bintang": df_top_comp["score_num"].astype(int),
            "Jumlah Like (Thumbs Up)": df_top_comp["thumbs"],
            "Aspek Terkait": df_top_comp.apply(_aspect_tags, axis=1),
            "Cuplikan Keluhan Pengguna": df_top_comp["content"].apply(_snippet),
        }
    )
    df_comp_out.to_csv(COMPLAINTS_OUT_PATH, index=False, encoding="utf-8")

    daily_kb = os.path.getsize(DAILY_OUT_PATH) / 1024
    comp_kb = os.path.getsize(COMPLAINTS_OUT_PATH) / 1024
    print(f"[OK] {DAILY_OUT_PATH} ({len(df_daily):,} baris, {daily_kb:.1f} KB, total ulasan={int(df_daily['n_reviews'].sum()):,})")
    print(f"[OK] {COMPLAINTS_OUT_PATH} ({len(df_comp_out):,} baris, {comp_kb:.1f} KB)")


if __name__ == "__main__":
    main()
